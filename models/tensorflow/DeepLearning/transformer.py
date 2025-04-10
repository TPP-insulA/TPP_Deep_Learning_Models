import os, sys
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input, Dense, Dropout, LayerNormalization,
    MultiHeadAttention, GlobalAveragePooling1D, Concatenate, Add
)
from keras.saving import register_keras_serializable
from typing import Tuple, Dict, Any, Optional, List, Union, Callable

PROJECT_ROOT = os.path.abspath(os.getcwd())
sys.path.append(PROJECT_ROOT) 

from config.models_config import TRANSFORMER_CONFIG

@register_keras_serializable()
class PositionEncoding(tf.keras.layers.Layer):
    """
    Codificación posicional para el Transformer.
    
    Parámetros:
    -----------
    max_position : int
        Posición máxima a codificar
    d_model : int
        Dimensión del modelo
    """
    def __init__(self, max_position: int, d_model: int, **kwargs):
        super().__init__(**kwargs)
        self.max_position = max_position
        self.d_model = d_model
        
    def build(self, input_shape: Tuple) -> None:
        """
        Construye la capa de codificación posicional.
        
        Parámetros:
        -----------
        input_shape : Tuple
            Forma de la entrada
        """
        positions = tf.range(self.max_position, dtype=tf.float32)[:, tf.newaxis]
        dimensions = tf.range(self.d_model, dtype=tf.float32)[tf.newaxis, :]
        angle_rates = 1 / tf.pow(10000.0, (2 * (dimensions // 2)) / tf.cast(self.d_model, tf.float32))
        angle_rads = positions * angle_rates

        # Apply sin to even indices, cos to odd indices
        pos_encoding = tf.stack([
            tf.sin(angle_rads[:, 0::2]),
            tf.cos(angle_rads[:, 1::2])
        ], axis=-1)

        self.pos_encoding = tf.reshape(pos_encoding, [self.max_position, self.d_model])
        
    def call(self, inputs: tf.Tensor) -> tf.Tensor:
        """
        Aplica la codificación posicional a las entradas.
        
        Parámetros:
        -----------
        inputs : tf.Tensor
            Tensor de entrada
            
        Retorna:
        --------
        tf.Tensor
            Tensor con codificación posicional añadida
        """
        sequence_length = tf.shape(inputs)[1]
        return inputs + self.pos_encoding[:sequence_length, :]
        
    def get_config(self) -> Dict:
        """
        Obtiene la configuración de la capa.
        
        Retorna:
        --------
        Dict
            Configuración de la capa
        """
        config = super().get_config()
        config.update({
            "max_position": self.max_position,
            "d_model": self.d_model
        })
        return config

def apply_activation(x: tf.Tensor, activation_name: str) -> tf.Tensor:
    """
    Aplica una función de activación a un tensor.
    
    Parámetros:
    -----------
    x : tf.Tensor
        Tensor de entrada
    activation_name : str
        Nombre de la función de activación
        
    Retorna:
    --------
    tf.Tensor
        Tensor con la activación aplicada
    """
    if activation_name == 'gelu':
        return tf.nn.gelu(x)
    elif activation_name == 'relu':
        return tf.nn.relu(x)
    elif activation_name == 'selu':
        return tf.nn.selu(x)
    elif activation_name == 'sigmoid':
        return tf.nn.sigmoid(x)
    elif activation_name == 'tanh':
        return tf.nn.tanh(x)
    else:
        return tf.nn.relu(x)  # Por defecto

def create_transformer_block(inputs: tf.Tensor, head_size: int, num_heads: int, 
                           ff_dim: int, dropout_rate: float, prenorm: bool = True) -> tf.Tensor:
    """
    Crea un bloque Transformer mejorado con pre/post normalización.
    
    Parámetros:
    -----------
    inputs : tf.Tensor
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
        
    Retorna:
    --------
    tf.Tensor
        Tensor procesado
    """
    if prenorm:
        # Pre-normalization architecture (mejor estabilidad de entrenamiento)
        x = LayerNormalization(epsilon=TRANSFORMER_CONFIG['epsilon'])(inputs)
        x = MultiHeadAttention(
            num_heads=num_heads,
            key_dim=head_size,
            value_dim=head_size,
            use_bias=TRANSFORMER_CONFIG['use_bias'],
            dropout=dropout_rate
        )(x, x)
        x = Dropout(dropout_rate)(x)
        res1 = Add()([inputs, x])
        
        # Red feed-forward
        x = LayerNormalization(epsilon=TRANSFORMER_CONFIG['epsilon'])(res1)
        x = Dense(ff_dim)(x)
        x = apply_activation(x, TRANSFORMER_CONFIG['activation'])
        x = Dropout(dropout_rate)(x)
        x = Dense(inputs.shape[-1])(x)
        x = Dropout(dropout_rate)(x)
        return Add()([res1, x])
    else:
        # Post-normalization architecture (original)
        attn = MultiHeadAttention(
            num_heads=num_heads,
            key_dim=head_size,
            value_dim=head_size,
            use_bias=TRANSFORMER_CONFIG['use_bias'],
            dropout=dropout_rate
        )(inputs, inputs)
        attn = Dropout(dropout_rate)(attn)
        res1 = LayerNormalization(epsilon=TRANSFORMER_CONFIG['epsilon'])(inputs + attn)
        
        # Red feed-forward
        ffn = Dense(ff_dim)(res1)
        ffn = apply_activation(ffn, TRANSFORMER_CONFIG['activation'])
        ffn = Dropout(dropout_rate)(ffn)
        ffn = Dense(inputs.shape[-1])(ffn)
        ffn = Dropout(dropout_rate)(ffn)
        return LayerNormalization(epsilon=TRANSFORMER_CONFIG['epsilon'])(res1 + ffn)

def create_transformer_model(cgm_shape: tuple, other_features_shape: tuple) -> Model:
    """
    Crea un modelo Transformer con entrada dual para datos CGM y otras características.
    
    Parámetros:
    -----------
    cgm_shape : tuple
        Forma de los datos CGM (muestras, pasos_temporales, características)
    other_features_shape : tuple
        Forma de otras características (muestras, características)
        
    Retorna:
    --------
    Model
        Modelo Transformer compilado
    """
    cgm_input = Input(shape=cgm_shape[1:], name='cgm_input')
    other_input = Input(shape=(other_features_shape[1],), name='other_input')
    
    # Proyección inicial y codificación posicional
    x = Dense(TRANSFORMER_CONFIG['key_dim'] * TRANSFORMER_CONFIG['num_heads'])(cgm_input)
    if TRANSFORMER_CONFIG['use_relative_pos']:
        x = PositionEncoding(
            TRANSFORMER_CONFIG['max_position'],
            TRANSFORMER_CONFIG['key_dim'] * TRANSFORMER_CONFIG['num_heads']
        )(x)
    
    # Bloques Transformer
    for _ in range(TRANSFORMER_CONFIG['num_layers']):
        x = create_transformer_block(
            x,
            TRANSFORMER_CONFIG['head_size'],
            TRANSFORMER_CONFIG['num_heads'],
            TRANSFORMER_CONFIG['ff_dim'],
            TRANSFORMER_CONFIG['dropout_rate'],
            TRANSFORMER_CONFIG['prenorm']
        )
    
    # Pooling con estadísticas
    avg_pool = GlobalAveragePooling1D()(x)
    max_pool = tf.keras.layers.GlobalMaxPooling1D()(x)
    x = Concatenate()([avg_pool, max_pool])
    
    # Combinar con otras características
    x = Concatenate()([x, other_input])
    
    # MLP final con residual connections
    skip = x
    x = Dense(128)(x)
    x = apply_activation(x, TRANSFORMER_CONFIG['activation'])
    x = LayerNormalization(epsilon=TRANSFORMER_CONFIG['epsilon'])(x)
    x = Dropout(TRANSFORMER_CONFIG['dropout_rate'])(x)
    
    x = Dense(128)(x)
    x = apply_activation(x, TRANSFORMER_CONFIG['activation'])
    
    # Skip connection si las dimensiones coinciden
    if skip.shape[-1] == 128:
        x = Add()([x, skip])
    
    x = Dense(64)(x)
    x = apply_activation(x, TRANSFORMER_CONFIG['activation'])
    x = LayerNormalization(epsilon=TRANSFORMER_CONFIG['epsilon'])(x)
    x = Dropout(TRANSFORMER_CONFIG['dropout_rate'])(x)
    
    output = Dense(1)(x)
    
    return Model(inputs=[cgm_input, other_input], outputs=output)