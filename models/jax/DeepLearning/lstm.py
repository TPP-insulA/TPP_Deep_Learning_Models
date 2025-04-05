import os, sys
import jax
import jax.numpy as jnp
import flax.linen as nn
from typing import Tuple, Dict, Any, Optional, List

PROJECT_ROOT = os.path.abspath(os.getcwd())
sys.path.append(PROJECT_ROOT) 

from config.models_config import LSTM_CONFIG

def create_lstm_attention_block(x: jnp.ndarray, units: int, num_heads: int = 4, 
                              dropout_rate: float = 0.2, deterministic: bool = False) -> jnp.ndarray:
    """
    Crea un bloque LSTM con self-attention y conexiones residuales.
    
    Parámetros:
    -----------
    x : jnp.ndarray
        Tensor de entrada
    units : int
        Número de unidades LSTM
    num_heads : int
        Número de cabezas de atención
    dropout_rate : float
        Tasa de dropout
    deterministic : bool
        Indica si está en modo de inferencia (no aplicar dropout)
        
    Retorna:
    --------
    jnp.ndarray
        Tensor procesado
    """
    # LSTM con skip connection
    skip1 = x
    
    # Preparar LSTM con scan
    lstm_cell = nn.LSTMCell(
        features=units,
        gate_fn=get_activation_fn(LSTM_CONFIG['activation']),
        activation_fn=get_activation_fn(LSTM_CONFIG['recurrent_activation'])
    )
    
    # Aplicar scan para procesar la secuencia
    lstm_scan = nn.scan(
        lstm_cell,
        variable_broadcast="params",
        split_rngs={"params": False, "dropout": True}
    )
    
    # Adaptar dimensiones para scan (time major)
    batch_size, _, _ = x.shape
    x_time_major = jnp.transpose(x, (1, 0, 2))  # [seq_len, batch, features]
    
    # Estado inicial
    c0 = jnp.zeros((batch_size, units))
    h0 = jnp.zeros((batch_size, units))
    
    # Aplicar LSTM cell
    (_, _), x = lstm_scan()(
        (c0, h0),
        x_time_major,
        deterministic=deterministic,
        dropout_rate=dropout_rate,
        recurrent_dropout_rate=LSTM_CONFIG['recurrent_dropout']
    )
    
    # Volver a batch major
    x = jnp.transpose(x, (1, 0, 2))  # [batch, seq_len, features]
    
    # Layer normalization
    x = nn.LayerNorm(epsilon=LSTM_CONFIG['epsilon'])(x)
    
    # Skip connection si las dimensiones coinciden
    if skip1.shape[-1] == units:
        x = x + skip1
    
    # Multi-head attention con gating mechanism
    skip2 = x
    
    # Atención con proyección de valores
    attention_output = nn.MultiHeadAttention(
        num_heads=num_heads,
        key_size=units // num_heads,
        value_size=units // num_heads,
        dropout_rate=dropout_rate,
        deterministic=deterministic
    )(x, x, x)
    
    # Mecanismo de gating para controlar flujo de información
    gate = nn.Dense(units)(skip2)
    gate = jax.nn.sigmoid(gate)
    attention_output = attention_output * gate
    
    # Conexión residual con normalización
    x = nn.LayerNorm(epsilon=LSTM_CONFIG['epsilon'])(attention_output + skip2)
    
    return x

def get_activation_fn(activation_name: str):
    """
    Devuelve la función de activación correspondiente al nombre.
    
    Parámetros:
    -----------
    activation_name : str
        Nombre de la función de activación
        
    Retorna:
    --------
    Callable
        Función de activación de JAX
    """
    if activation_name == 'relu':
        return jax.nn.relu
    elif activation_name == 'tanh':
        return jax.nn.tanh
    elif activation_name == 'sigmoid':
        return jax.nn.sigmoid
    elif activation_name == 'gelu':
        return jax.nn.gelu
    else:
        return jax.nn.tanh  # Por defecto

class LSTMModel(nn.Module):
    """
    Modelo LSTM avanzado con self-attention y conexiones residuales.
    
    Parámetros:
    -----------
    config : Dict
        Diccionario con la configuración del modelo
    cgm_shape : Tuple
        Forma de los datos CGM
    other_features_shape : Tuple
        Forma de otras características
    """
    config: Dict
    cgm_shape: Tuple
    other_features_shape: Tuple
    
    @nn.compact
    def __call__(self, inputs: Tuple[jnp.ndarray, jnp.ndarray], training: bool = True) -> jnp.ndarray:
        cgm_input, other_input = inputs
        deterministic = not training
        
        # Proyección inicial
        x = nn.Dense(self.config['hidden_units'][0])(cgm_input)
        x = nn.LayerNorm(epsilon=self.config['epsilon'])(x)
        
        # Bloques LSTM apilados con distinto nivel de complejidad
        for i, units in enumerate(self.config['hidden_units']):
            # Opción de bidireccional para primeras capas si está configurado
            if i < len(self.config['hidden_units'])-1 and self.config['use_bidirectional']:
                # Forward pass
                fwd_out = create_forward_lstm(
                    x, 
                    units, 
                    dropout_rate=self.config['dropout_rate'],
                    recurrent_dropout=self.config['recurrent_dropout'],
                    activation=self.config['activation'],
                    recurrent_activation=self.config['recurrent_activation'],
                    deterministic=deterministic
                )
                
                # Backward pass (invierte la secuencia)
                x_reversed = jnp.flip(x, axis=1)
                bwd_out = create_forward_lstm(
                    x_reversed, 
                    units, 
                    dropout_rate=self.config['dropout_rate'],
                    recurrent_dropout=self.config['recurrent_dropout'],
                    activation=self.config['activation'],
                    recurrent_activation=self.config['recurrent_activation'],
                    deterministic=deterministic
                )
                bwd_out = jnp.flip(bwd_out, axis=1)
                
                # Combinar salidas forward y backward
                x = jnp.concatenate([fwd_out, bwd_out], axis=-1)
                x = nn.LayerNorm(epsilon=self.config['epsilon'])(x)
            else:
                # Bloques con atención para capas posteriores
                x = create_lstm_attention_block(
                    x, 
                    units=units, 
                    num_heads=self.config['attention_heads'],
                    dropout_rate=self.config['dropout_rate'],
                    deterministic=deterministic
                )
        
        # Extracción de características con pooling estadístico
        avg_pool = jnp.mean(x, axis=1)  # equivalent to GlobalAveragePooling1D
        max_pool = jnp.max(x, axis=1)  # equivalent to GlobalMaxPooling1D
        x = jnp.concatenate([avg_pool, max_pool], axis=-1)
        
        # Combinar con otras características
        x = jnp.concatenate([x, other_input], axis=-1)
        
        # Red densa final con skip connections
        skip = x
        x = nn.Dense(self.config['dense_units'][0])(x)
        x = get_activation_fn(self.config['dense_activation'])(x)
        x = nn.LayerNorm(epsilon=self.config['epsilon'])(x)
        x = nn.Dropout(rate=self.config['dropout_rate'], deterministic=deterministic)(x)
        
        # Segunda capa densa con residual
        x = nn.Dense(self.config['dense_units'][1])(x)
        x = get_activation_fn(self.config['dense_activation'])(x)
        
        if skip.shape[-1] == self.config['dense_units'][1]:
            x = x + skip  # Skip connection si las dimensiones coinciden
            
        x = nn.LayerNorm(epsilon=self.config['epsilon'])(x)
        x = nn.Dropout(rate=self.config['dropout_rate'] * 0.5, deterministic=deterministic)(x)  # Menor dropout en capas finales
        
        # Capa de salida
        output = nn.Dense(1)(x)
        
        return output

def create_forward_lstm(x: jnp.ndarray, units: int, dropout_rate: float, recurrent_dropout: float,
                       activation: str, recurrent_activation: str, deterministic: bool) -> jnp.ndarray:
    """
    Crea una capa LSTM unidireccional.
    
    Parámetros:
    -----------
    x : jnp.ndarray
        Tensor de entrada
    units : int
        Número de unidades LSTM
    dropout_rate : float
        Tasa de dropout
    recurrent_dropout : float
        Tasa de dropout recurrente
    activation : str
        Nombre de la función de activación
    recurrent_activation : str
        Nombre de la función de activación recurrente
    deterministic : bool
        Indica si está en modo de inferencia
        
    Retorna:
    --------
    jnp.ndarray
        Tensor procesado
    """
    batch_size, _, _ = x.shape
    
    # Crear celda LSTM
    lstm_cell = nn.LSTMCell(
        features=units,
        gate_fn=get_activation_fn(activation),
        activation_fn=get_activation_fn(recurrent_activation)
    )
    
    # Aplicar scan para procesar la secuencia
    lstm_scan = nn.scan(
        lstm_cell,
        variable_broadcast="params",
        split_rngs={"params": False, "dropout": True}
    )
    
    # Adaptar dimensiones para scan (time major)
    x_time_major = jnp.transpose(x, (1, 0, 2))  # [seq_len, batch, features]
    
    # Estado inicial
    c0 = jnp.zeros((batch_size, units))
    h0 = jnp.zeros((batch_size, units))
    
    # Aplicar LSTM cell
    (_, _), outputs = lstm_scan()(
        (c0, h0),
        x_time_major,
        deterministic=deterministic,
        dropout_rate=dropout_rate,
        recurrent_dropout_rate=recurrent_dropout
    )
    
    # Volver a batch major
    outputs = jnp.transpose(outputs, (1, 0, 2))  # [batch, seq_len, features]
    
    return outputs

def create_lstm_model(cgm_shape: tuple, other_features_shape: tuple) -> LSTMModel:
    """
    Crea un modelo LSTM avanzado con self-attention y conexiones residuales con JAX/Flax.
    
    Parámetros:
    -----------
    cgm_shape : tuple
        Forma de los datos CGM (muestras, pasos_temporales, características)
    other_features_shape : tuple
        Forma de otras características (muestras, características)
        
    Retorna:
    --------
    lstm_model
        Modelo LSTM inicializado
    """
    model = LSTMModel(
        config=LSTM_CONFIG,
        cgm_shape=cgm_shape,
        other_features_shape=other_features_shape
    )
    
    return model