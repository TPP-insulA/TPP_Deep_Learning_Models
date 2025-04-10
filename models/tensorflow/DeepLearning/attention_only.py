import os, sys
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input, Dense, Dropout, LayerNormalization,
    MultiHeadAttention, GlobalAveragePooling1D, Concatenate,
    Add
)
from keras.saving import register_keras_serializable
from typing import Dict, Tuple, Any, Optional

PROJECT_ROOT = os.path.abspath(os.getcwd())
sys.path.append(PROJECT_ROOT) 

from config.models_config import ATTENTION_CONFIG

@register_keras_serializable
class RelativePositionEncoding(tf.keras.layers.Layer):
    """
    Codificación de posición relativa para mejorar la atención temporal.
    
    Parámetros:
    -----------
    max_position : int
        Posición máxima a codificar
    depth : int
        Profundidad de la codificación
        
    Retorna:
    --------
    tf.Tensor
        Tensor de codificación de posición
    """
    def __init__(self, max_position: int, depth: int, **kwargs):
        super().__init__(**kwargs)
        self.max_position = max_position
        self.depth = depth
        
    def build(self, input_shape):
        self.rel_embeddings = self.add_weight(
            name="rel_embeddings",
            shape=[2 * self.max_position - 1, self.depth],
            initializer="glorot_uniform"
        )
        self.built = True
        
    def call(self, inputs: tf.Tensor) -> tf.Tensor:
        length = tf.shape(inputs)[1]
        pos_indices = tf.range(length)[:, tf.newaxis] - tf.range(length)[tf.newaxis, :] + self.max_position - 1
        pos_emb = tf.gather(self.rel_embeddings, pos_indices)
        return pos_emb
    
    def get_config(self) -> Dict:
        config = super().get_config()
        config.update({
            "max_position": self.max_position,
            "depth": self.depth
        })
        return config

def get_activation(x: tf.Tensor, activation_name: str) -> tf.Tensor:
    """
    Aplica la función de activación según su nombre.
    
    Parámetros:
    -----------
    x : tf.Tensor
        Tensor al que aplicar la activación
    activation_name : str
        Nombre de la función de activación
        
    Retorna:
    --------
    tf.Tensor
        Tensor con la activación aplicada
    """
    if activation_name == 'relu':
        return tf.nn.relu(x)
    elif activation_name == 'gelu':
        return tf.nn.gelu(x)
    elif activation_name == 'swish':
        return tf.nn.swish(x)
    elif activation_name == 'silu':
        return tf.nn.silu(x)
    else:
        return tf.nn.relu(x)  # Valor por defecto

def create_attention_block(x: tf.Tensor, num_heads: int, key_dim: int, 
                         ff_dim: int, dropout_rate: float, training: bool = None) -> tf.Tensor:
    """
    Crea un bloque de atención mejorado con posición relativa y gating.

    Parámetros:
    -----------
    x : tf.Tensor
        Tensor de entrada
    num_heads : int
        Número de cabezas de atención
    key_dim : int
        Dimensión de la clave
    ff_dim : int
        Dimensión de la red feed-forward
    dropout_rate : float
        Tasa de dropout
    training : bool
        Indica si está en modo entrenamiento
    
    Retorna:
    --------
    tf.Tensor
        Tensor procesado
    """
    # Codificación de posición relativa
    if ATTENTION_CONFIG['use_relative_attention']:
        pos_encoding = RelativePositionEncoding(
            ATTENTION_CONFIG['max_relative_position'],
            key_dim
        )(x)
        
        attention_output = MultiHeadAttention(
            num_heads=num_heads,
            key_dim=key_dim,
            value_dim=ATTENTION_CONFIG['head_size']
        )(x, x, attention_bias=pos_encoding)
    else:
        attention_output = MultiHeadAttention(
            num_heads=num_heads,
            key_dim=key_dim
        )(x, x)
    
    # Mecanismo de gating
    gate = Dense(attention_output.shape[-1], activation='sigmoid')(x)
    attention_output = gate * attention_output
    
    attention_output = Dropout(dropout_rate)(attention_output, training=training)
    x = LayerNormalization(epsilon=1e-6)(x + attention_output)
    
    # Red feed-forward mejorada con GLU
    ffn = Dense(ff_dim)(x)
    ffn_gate = Dense(ff_dim, activation='sigmoid')(x)
    ffn = ffn * ffn_gate
    ffn = Dense(x.shape[-1])(ffn)
    ffn = Dropout(dropout_rate)(ffn, training=training)
    
    return LayerNormalization(epsilon=1e-6)(x + ffn)

def create_attention_model(cgm_shape: tuple, other_features_shape: tuple) -> Model:
    """
    Crea un modelo basado únicamente en mecanismos de atención.

    Parámetros:
    -----------
    cgm_shape : tuple
        Forma de los datos CGM (muestras, pasos_temporales, características)
    other_features_shape : tuple
        Forma de otras características (muestras, características)

    Retorna:
    --------
    Model
        Modelo de atención compilado
    """
    cgm_input = Input(shape=cgm_shape[1:])
    other_input = Input(shape=(other_features_shape[1],))
    
    # Proyección inicial
    x = Dense(ATTENTION_CONFIG['key_dim'] * ATTENTION_CONFIG['num_heads'])(cgm_input)
    
    # Stochastic depth (dropout de capas)
    survive_rates = tf.linspace(1.0, 0.5, ATTENTION_CONFIG['num_layers'])
    
    # Apilar bloques de atención con stochastic depth
    for i in range(ATTENTION_CONFIG['num_layers']):
        # En entrenamiento, aplicar dropout de capa
        # En inferencia, aplicar todas las capas con el factor de supervivencia
        if tf.keras.backend.learning_phase():
            if tf.random.uniform([]) < survive_rates[i]:
                x = create_attention_block(
                    x,
                    ATTENTION_CONFIG['num_heads'],
                    ATTENTION_CONFIG['key_dim'],
                    ATTENTION_CONFIG['ff_dim'],
                    ATTENTION_CONFIG['dropout_rate']
                )
        else:
            # En inferencia, escalar la salida
            block_output = create_attention_block(
                x,
                ATTENTION_CONFIG['num_heads'],
                ATTENTION_CONFIG['key_dim'],
                ATTENTION_CONFIG['ff_dim'],
                ATTENTION_CONFIG['dropout_rate'],
                training=False
            )
            x = x + survive_rates[i] * (block_output - x)
    
    # Contexto global
    attention_pooled = GlobalAveragePooling1D()(x)
    max_pooled = tf.keras.layers.GlobalMaxPooling1D()(x)
    x = Concatenate()([attention_pooled, max_pooled])
    
    # Combinar con otras características
    x = Concatenate()([x, other_input])
    
    # MLP final con conexión residual
    skip = x
    x = Dense(128)(x)
    x = get_activation(x, ATTENTION_CONFIG['activation'])
    x = LayerNormalization(epsilon=1e-6)(x)
    x = Dropout(ATTENTION_CONFIG['dropout_rate'])(x)
    x = Dense(128)(x)
    x = get_activation(x, ATTENTION_CONFIG['activation'])
    
    if skip.shape[-1] == 128:
        x = Add()([x, skip])
    
    output = Dense(1)(x)
    
    return Model(inputs=[cgm_input, other_input], outputs=output)