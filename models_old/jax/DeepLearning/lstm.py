import os, sys
import jax
import jax.numpy as jnp
import flax.linen as nn
import numpy as np
from typing import Tuple, Dict, Any, Optional, List, Callable

PROJECT_ROOT = os.path.abspath(os.getcwd())
sys.path.append(PROJECT_ROOT) 

from config.models_config import LSTM_CONFIG, EARLY_STOPPING_POLICY
from custom.DeepLearning.dl_model_wrapper import DLModelWrapper
from models_old.early_stopping import get_early_stopping_config

# Constantes para uso repetido
CONST_RELU = "relu"
CONST_TANH = "tanh"
CONST_SIGMOID = "sigmoid"
CONST_GELU = "gelu"
CONST_DROPOUT = "dropout"

def get_activation_fn(activation_name: str) -> Callable:
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
    if activation_name == CONST_RELU:
        return jax.nn.relu
    elif activation_name == CONST_TANH:
        return jax.nn.tanh
    elif activation_name == CONST_SIGMOID:
        return jax.nn.sigmoid
    elif activation_name == CONST_GELU:
        return jax.nn.gelu
    else:
        return jax.nn.tanh  # Por defecto

class LSTMAttentionBlock(nn.Module):
    """
    Bloque LSTM con self-attention y conexiones residuales.
    
    Parámetros:
    -----------
    units : int
        Número de unidades LSTM
    num_heads : int
        Número de cabezas de atención
    dropout_rate : float
        Tasa de dropout
    """
    units: int
    num_heads: int = 4
    dropout_rate: float = 0.2
    epsilon: float = 1e-6
    
    @nn.compact
    def __call__(self, x: jnp.ndarray, training: bool = True) -> jnp.ndarray:
        """
        Procesa la entrada con un bloque LSTM y atención.
        
        Parámetros:
        -----------
        x : jnp.ndarray
            Tensor de entrada
        training : bool
            Si es True, está en modo entrenamiento
            
        Retorna:
        --------
        jnp.ndarray
            Tensor procesado
        """
        deterministic = not training
        
        # LSTM con skip connection
        skip1 = x
        
        # Capa LSTM
        lstm = nn.RNN(nn.LSTMCell(features=self.units))
        x = lstm(x)
        
        # Layer normalization
        x = nn.LayerNorm(epsilon=self.epsilon)(x)
        
        # Skip connection si las dimensiones coinciden
        if skip1.shape[-1] == self.units:
            x = x + skip1
        
        # Multi-head attention con gating mechanism
        skip2 = x
        
        # Atención con proyección de valores
        attention_output = nn.MultiHeadAttention(
            num_heads=self.num_heads,
            qkv_features=self.units // self.num_heads,
            dropout_rate=self.dropout_rate,
            deterministic=deterministic
        )(x, x, x)
        
        # Mecanismo de gating para controlar flujo de información
        gate = nn.Dense(self.units)(skip2)
        gate = jax.nn.sigmoid(gate)
        attention_output = attention_output * gate
        
        # Conexión residual con normalización
        x = nn.LayerNorm(epsilon=self.epsilon)(attention_output + skip2)
        
        return x

class BidirectionalLSTMBlock(nn.Module):
    """
    Bloque LSTM bidireccional.
    
    Parámetros:
    -----------
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
    """
    units: int
    dropout_rate: float = 0.2
    recurrent_dropout: float = 0.1
    activation: str = CONST_TANH
    recurrent_activation: str = CONST_SIGMOID
    epsilon: float = 1e-6
    
    @nn.compact
    def __call__(self, x: jnp.ndarray, training: bool = True) -> jnp.ndarray:
        """
        Procesa la entrada con un bloque LSTM bidireccional.
        
        Parámetros:
        -----------
        x : jnp.ndarray
            Tensor de entrada
        training : bool
            Si es True, está en modo entrenamiento
            
        Retorna:
        --------
        jnp.ndarray
            Tensor procesado
        """
        deterministic = not training
        
        # Forward LSTM
        lstm_fwd = nn.RNN(
            nn.LSTMCell(
                features=self.units,
                gate_fn=get_activation_fn(self.activation),
                activation_fn=get_activation_fn(self.recurrent_activation)
            )
        )
        fwd_out = lstm_fwd(x)
        
        # Backward LSTM (invertimos la secuencia)
        x_reversed = jnp.flip(x, axis=1)
        lstm_bwd = nn.RNN(
            nn.LSTMCell(
                features=self.units,
                gate_fn=get_activation_fn(self.activation),
                activation_fn=get_activation_fn(self.recurrent_activation)
            )
        )
        bwd_out = lstm_bwd(x_reversed)
        bwd_out = jnp.flip(bwd_out, axis=1)
        
        # Combinar salidas forward y backward
        combined = jnp.concatenate([fwd_out, bwd_out], axis=-1)
        combined = nn.LayerNorm(epsilon=self.epsilon)(combined)
        
        # Dropout
        if not deterministic and self.dropout_rate > 0:
            combined = nn.Dropout(rate=self.dropout_rate)(combined, deterministic=deterministic)
            
        return combined

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
    def __call__(self, cgm_input: jnp.ndarray, other_input: jnp.ndarray, training: bool = True) -> jnp.ndarray:
        """
        Ejecuta el modelo LSTM sobre las entradas.
        
        Parámetros:
        -----------
        cgm_input : jnp.ndarray
            Datos de entrada CGM
        other_input : jnp.ndarray
            Otras características de entrada
        training : bool, opcional
            Indica si está en modo entrenamiento (default: True)
            
        Retorna:
        --------
        jnp.ndarray
            Predicciones del modelo
        """
        deterministic = not training
        
        # Proyección inicial
        x = nn.Dense(self.config['hidden_units'][0])(cgm_input)
        x = nn.LayerNorm(epsilon=self.config['epsilon'])(x)
        
        # Bloques LSTM apilados con distinto nivel de complejidad
        for i, units in enumerate(self.config['hidden_units']):
            # Opción de bidireccional para primeras capas si está configurado
            if i < len(self.config['hidden_units'])-1 and self.config['use_bidirectional']:
                x = BidirectionalLSTMBlock(
                    units=units,
                    dropout_rate=self.config['dropout_rate'],
                    recurrent_dropout=self.config['recurrent_dropout'],
                    activation=self.config['activation'],
                    recurrent_activation=self.config['recurrent_activation'],
                    epsilon=self.config['epsilon']
                )(x, training=training)
            else:
                # Bloques con atención para capas posteriores
                x = LSTMAttentionBlock(
                    units=units,
                    num_heads=self.config['attention_heads'],
                    dropout_rate=self.config['dropout_rate'],
                    epsilon=self.config['epsilon']
                )(x, training=training)
        
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

def create_lstm_model(cgm_shape: Tuple[int, ...], other_features_shape: Tuple[int, ...]) -> DLModelWrapper:
    """
    Crea un modelo LSTM avanzado con self-attention y conexiones residuales con JAX/Flax.
    
    Parámetros:
    -----------
    cgm_shape : Tuple[int, ...]
        Forma de los datos CGM (pasos_temporales, características)
    other_features_shape : Tuple[int, ...]
        Forma de otras características (características)
        
    Retorna:
    --------
    DLModelWrapper
        Modelo LSTM inicializado y envuelto en DLModelWrapper
    """
    # Función que crea una instancia del modelo
    def model_creator():
        return LSTMModel(
            config=LSTM_CONFIG,
            cgm_shape=cgm_shape,
            other_features_shape=other_features_shape
        )
    
    # Crear wrapper con framework JAX
    wrapper = DLModelWrapper(model_creator, 'jax')
    
    # Configurar early stopping
    es_patience, es_min_delta, es_restore_best = get_early_stopping_config()
    wrapper.add_early_stopping(
        patience=es_patience,
        min_delta=es_min_delta,
        restore_best_weights=es_restore_best
    )
    
    return wrapper

def model_creator() -> Callable[[Tuple[int, ...], Tuple[int, ...]], DLModelWrapper]:
    """
    Retorna una función para crear un modelo LSTM compatible con la API del sistema.
    
    Retorna:
    --------
    Callable[[Tuple[int, ...], Tuple[int, ...]], DLModelWrapper]
        Función para crear el modelo con las formas de entrada especificadas
    """
    return create_lstm_model