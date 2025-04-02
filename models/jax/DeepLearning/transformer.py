import os, sys
import jax
import jax.numpy as jnp
import flax.linen as nn
from typing import Dict, Tuple, List, Any, Optional, Callable

PROJECT_ROOT = os.path.abspath(os.getcwd())
sys.path.append(PROJECT_ROOT) 

from models.config import TRANSFORMER_CONFIG

class position_encoding(nn.Module):
    """
    Codificación posicional para el Transformer.
    
    Parámetros:
    -----------
    max_position : int
        Posición máxima a codificar
    d_model : int
        Dimensión del modelo
    """
    max_position: int
    d_model: int
    
    def setup(self) -> None:
        """
        Inicializa la codificación posicional.
        """
        positions = jnp.arange(self.max_position, dtype=jnp.float32)[:, jnp.newaxis]
        dimensions = jnp.arange(self.d_model, dtype=jnp.float32)[jnp.newaxis, :]
        angle_rates = 1 / jnp.power(10000.0, (2 * (dimensions // 2)) / self.d_model)
        angle_rads = positions * angle_rates

        # Apply sin to even indices, cos to odd indices
        pos_encoding = jnp.zeros((self.max_position, self.d_model))
        pos_encoding = pos_encoding.at[:, 0::2].set(jnp.sin(angle_rads[:, 0::2]))
        pos_encoding = pos_encoding.at[:, 1::2].set(jnp.cos(angle_rads[:, 1::2]))
        
        self.pos_encoding = pos_encoding
    
    def __call__(self, inputs: jnp.ndarray) -> jnp.ndarray:
        """
        Aplica la codificación posicional a las entradas.
        
        Parámetros:
        -----------
        inputs : jnp.ndarray
            Tensor de entrada
            
        Retorna:
        --------
        jnp.ndarray
            Tensor con codificación posicional añadida
        """
        sequence_length = inputs.shape[1]
        return inputs + self.pos_encoding[:sequence_length, :]

def apply_activation(x: jnp.ndarray, activation_name: str) -> jnp.ndarray:
    """
    Aplica una función de activación a un tensor.
    
    Parámetros:
    -----------
    x : jnp.ndarray
        Tensor de entrada
    activation_name : str
        Nombre de la función de activación
        
    Retorna:
    --------
    jnp.ndarray
        Tensor con la activación aplicada
    """
    if activation_name == 'gelu':
        return nn.gelu(x)
    elif activation_name == 'relu':
        return nn.relu(x)
    elif activation_name == 'selu':
        return nn.selu(x)
    elif activation_name == 'sigmoid':
        return jax.nn.sigmoid(x)
    elif activation_name == 'tanh':
        return jnp.tanh(x)
    else:
        return nn.relu(x)  # Por defecto

def create_transformer_block(inputs: jnp.ndarray, head_size: int, num_heads: int, 
                           ff_dim: int, dropout_rate: float, prenorm: bool = True, 
                           deterministic: bool = False) -> jnp.ndarray:
    """
    Crea un bloque Transformer mejorado con pre/post normalización.
    
    Parámetros:
    -----------
    inputs : jnp.ndarray
        Tensor de entrada
    head_size : int
        Tamaño de la cabeza de atención
    num_heads : int
        Número de cabezas de atención
    ff_dim : int
        Dimensión de la red feed-forward
    dropout_rate : float
        Tasa de dropout
    prenorm : bool
        Indica si se usa pre-normalización
    deterministic : bool
        Modo determinístico (para inferencia)
        
    Retorna:
    --------
    jnp.ndarray
        Tensor procesado
    """
    if prenorm:
        # Pre-normalization architecture (mejor estabilidad de entrenamiento)
        x = nn.LayerNorm(epsilon=TRANSFORMER_CONFIG['epsilon'])(inputs)
        x = nn.MultiHeadAttention(
            num_heads=num_heads,
            key_size=head_size,
            value_size=head_size,
            dropout_rate=dropout_rate,
            use_bias=TRANSFORMER_CONFIG['use_bias'],
            deterministic=deterministic
        )(x, x)
        x = nn.Dropout(rate=dropout_rate, deterministic=deterministic)(x)
        res1 = inputs + x
        
        # Red feed-forward
        x = nn.LayerNorm(epsilon=TRANSFORMER_CONFIG['epsilon'])(res1)
        x = nn.Dense(ff_dim)(x)
        x = apply_activation(x, TRANSFORMER_CONFIG['activation'])
        x = nn.Dropout(rate=dropout_rate, deterministic=deterministic)(x)
        x = nn.Dense(inputs.shape[-1])(x)
        x = nn.Dropout(rate=dropout_rate, deterministic=deterministic)(x)
        return res1 + x
    else:
        # Post-normalization architecture (original)
        attn = nn.MultiHeadAttention(
            num_heads=num_heads,
            key_size=head_size,
            value_size=head_size,
            dropout_rate=dropout_rate,
            use_bias=TRANSFORMER_CONFIG['use_bias'],
            deterministic=deterministic
        )(inputs, inputs)
        attn = nn.Dropout(rate=dropout_rate, deterministic=deterministic)(attn)
        res1 = nn.LayerNorm(epsilon=TRANSFORMER_CONFIG['epsilon'])(inputs + attn)
        
        # Red feed-forward
        ffn = nn.Dense(ff_dim)(res1)
        ffn = apply_activation(ffn, TRANSFORMER_CONFIG['activation'])
        ffn = nn.Dropout(rate=dropout_rate, deterministic=deterministic)(ffn)
        ffn = nn.Dense(inputs.shape[-1])(ffn)
        ffn = nn.Dropout(rate=dropout_rate, deterministic=deterministic)(ffn)
        return nn.LayerNorm(epsilon=TRANSFORMER_CONFIG['epsilon'])(res1 + ffn)

class transformer_model(nn.Module):
    """
    Modelo Transformer con entrada dual para datos CGM y otras características.
    
    Parámetros:
    -----------
    config : Dict
        Configuración del modelo
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
        """
        Aplica el modelo Transformer a las entradas.
        
        Parámetros:
        -----------
        inputs : Tuple[jnp.ndarray, jnp.ndarray]
            Tupla de (cgm_input, other_input)
        training : bool
            Indica si está en modo entrenamiento
            
        Retorna:
        --------
        jnp.ndarray
            Predicciones del modelo
        """
        cgm_input, other_input = inputs
        deterministic = not training
        
        # Proyección inicial y codificación posicional
        x = nn.Dense(self.config['key_dim'] * self.config['num_heads'])(cgm_input)
        if self.config['use_relative_pos']:
            x = position_encoding(
                max_position=self.config['max_position'],
                d_model=self.config['key_dim'] * self.config['num_heads']
            )(x)
        
        # Bloques Transformer
        for _ in range(self.config['num_layers']):
            x = create_transformer_block(
                x,
                self.config['head_size'],
                self.config['num_heads'],
                self.config['ff_dim'],
                self.config['dropout_rate'],
                self.config['prenorm'],
                deterministic
            )
        
        # Pooling con estadísticas
        avg_pool = jnp.mean(x, axis=1)
        max_pool = jnp.max(x, axis=1)
        x = jnp.concatenate([avg_pool, max_pool], axis=-1)
        
        # Combinar con otras características
        x = jnp.concatenate([x, other_input], axis=-1)
        
        # MLP final con residual connections
        skip = x
        x = nn.Dense(128)(x)
        x = apply_activation(x, self.config['activation'])
        x = nn.LayerNorm(epsilon=self.config['epsilon'])(x)
        x = nn.Dropout(rate=self.config['dropout_rate'], deterministic=deterministic)(x)
        
        x = nn.Dense(128)(x)
        x = apply_activation(x, self.config['activation'])
        
        # Skip connection si las dimensiones coinciden
        if skip.shape[-1] == 128:
            x = x + skip
        
        x = nn.Dense(64)(x)
        x = apply_activation(x, self.config['activation'])
        x = nn.LayerNorm(epsilon=self.config['epsilon'])(x)
        x = nn.Dropout(rate=self.config['dropout_rate'], deterministic=deterministic)(x)
        
        output = nn.Dense(1)(x)
        
        return output

def create_transformer_model(cgm_shape: tuple, other_features_shape: tuple) -> transformer_model:
    """
    Crea un modelo Transformer con entrada dual para datos CGM y otras características con JAX/Flax.
    
    Parámetros:
    -----------
    cgm_shape : tuple
        Forma de los datos CGM (muestras, pasos_temporales, características)
    other_features_shape : tuple
        Forma de otras características (muestras, características)
        
    Retorna:
    --------
    transformer_model
        Modelo Transformer inicializado
    """
    model = transformer_model(
        config=TRANSFORMER_CONFIG,
        cgm_shape=cgm_shape,
        other_features_shape=other_features_shape
    )
    
    return model