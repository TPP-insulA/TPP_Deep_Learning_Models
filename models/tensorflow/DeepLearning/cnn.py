import os, sys
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input, Conv1D, Dense, Dropout, BatchNormalization, LayerNormalization,
    MaxPooling1D, GlobalAveragePooling1D, GlobalMaxPooling1D, Concatenate,
    Activation, Add
)
from keras.saving import register_keras_serializable
from typing import Dict, Any, Tuple, Optional

PROJECT_ROOT = os.path.abspath(os.getcwd())
sys.path.append(PROJECT_ROOT) 

from config.models_config import CNN_CONFIG

@register_keras_serializable()
class SqueezeExcitationBlock(tf.keras.layers.Layer):
    """
    Bloque Squeeze-and-Excitation como capa personalizada.
    
    Parámetros:
    -----------
    filters : int
        Número de filtros del bloque
    se_ratio : int
        Factor de reducción para la capa de squeeze
        
    Retorna:
    --------
    tf.Tensor
        Tensor de entrada escalado por los pesos de atención
    """
    def __init__(self, filters: int, se_ratio: int = 16, **kwargs):
        super().__init__(**kwargs)
        self.filters = filters
        self.se_ratio = se_ratio
        
        # Define layers
        self.gap = GlobalAveragePooling1D()
        self.dense1 = Dense(filters // se_ratio, activation='gelu')
        self.dense2 = Dense(filters, activation='sigmoid')
    
    def call(self, inputs: tf.Tensor) -> tf.Tensor:
        # Squeeze
        x = self.gap(inputs)
        
        # Excitation
        x = self.dense1(x)
        x = self.dense2(x)
        
        # Reshape para broadcasting
        x = tf.expand_dims(x, axis=1)
        
        # Scale
        return inputs * x
    
    def get_config(self) -> Dict:
        config = super().get_config()
        config.update({
            "filters": self.filters,
            "se_ratio": self.se_ratio
        })
        return config

def get_activation(activation_name: str) -> Any:
    """
    Obtiene la función de activación por su nombre.
    
    Parámetros:
    -----------
    activation_name : str
        Nombre de la función de activación
        
    Retorna:
    --------
    Any
        Función de activación correspondiente
    """
    if activation_name == 'relu':
        return 'relu'
    elif activation_name == 'gelu':
        return 'gelu'
    elif activation_name == 'swish':
        return tf.keras.activations.swish
    elif activation_name == 'silu':
        return tf.nn.silu
    else:
        return 'relu'  # Valor por defecto

def create_residual_block(x: tf.Tensor, filters: int, dilation_rate: int = 1) -> tf.Tensor:
    """
    Crea un bloque residual mejorado con dilated convolutions y SE.
    
    Parámetros:
    -----------
    x : tf.Tensor
        Tensor de entrada
    filters : int
        Número de filtros
    dilation_rate : int
        Tasa de dilatación para las convoluciones
        
    Retorna:
    --------
    tf.Tensor
        Tensor procesado
    """
    skip = x
    
    # Camino convolucional
    x = Conv1D(
        filters=filters,
        kernel_size=CNN_CONFIG['kernel_size'],
        padding='same',
        dilation_rate=dilation_rate,
        kernel_initializer='glorot_uniform'
    )(x)
    x = LayerNormalization()(x)
    x = Activation(get_activation(CNN_CONFIG['activation']))(x)
    
    # Squeeze-and-Excitation
    if CNN_CONFIG['use_se_block']:
        x = SqueezeExcitationBlock(filters, CNN_CONFIG['se_ratio'])(x)
    
    # Proyección del residual si es necesario
    if skip.shape[-1] != filters:
        skip = Conv1D(filters, 1, padding='same', kernel_initializer='glorot_uniform')(skip)
    
    return Add()([x, skip])

def create_cnn_model(cgm_shape: tuple, other_features_shape: tuple) -> Model:
    """
    Crea un modelo CNN (Red Neuronal Convolucional) con entrada dual para datos CGM y otras características.
    
    Parámetros:
    -----------
    cgm_shape : tuple
        Forma de los datos CGM (muestras, pasos_temporales, características)
    other_features_shape : tuple
        Forma de otras características (muestras, características)
        
    Retorna:
    --------
    Model
        Modelo CNN compilado
    """
    # Entrada CGM
    cgm_input = Input(shape=cgm_shape[1:], name='cgm_input')
    
    # Proyección inicial
    x = Conv1D(CNN_CONFIG['filters'][0], 1, padding='same', kernel_initializer='glorot_uniform')(cgm_input)
    x = LayerNormalization()(x) if CNN_CONFIG['use_layer_norm'] else BatchNormalization()(x)
    
    # Bloques residuales con diferentes tasas de dilatación
    for filters in CNN_CONFIG['filters']:
        for dilation_rate in CNN_CONFIG['dilation_rates']:
            x = create_residual_block(x, filters, dilation_rate)
        x = MaxPooling1D(pool_size=CNN_CONFIG['pool_size'])(x)
    
    # Pooling global con concat de max y average
    avg_pool = GlobalAveragePooling1D()(x)
    max_pool = GlobalMaxPooling1D()(x)
    x = Concatenate()([avg_pool, max_pool])
    
    # Entrada de otras características
    other_input = Input(shape=(other_features_shape[1],), name='other_input')
    
    # Combinar características
    combined = Concatenate()([x, other_input])
    
    # Capas densas con conexiones residuales
    skip = combined
    dense = Dense(256, activation=get_activation(CNN_CONFIG['activation']), kernel_initializer='glorot_uniform')(combined)
    dense = LayerNormalization()(dense) if CNN_CONFIG['use_layer_norm'] else BatchNormalization()(dense)
    dense = Dropout(CNN_CONFIG['dropout_rate'])(dense)
    dense = Dense(256, activation=get_activation(CNN_CONFIG['activation']), kernel_initializer='glorot_uniform')(dense)
    if skip.shape[-1] == 256:
        dense = Add()([dense, skip])
    
    # Capas finales
    dense = Dense(128, activation=get_activation(CNN_CONFIG['activation']), kernel_initializer='glorot_uniform')(dense)
    dense = LayerNormalization()(dense) if CNN_CONFIG['use_layer_norm'] else BatchNormalization()(dense)
    dense = Dropout(CNN_CONFIG['dropout_rate'] / 2)(dense)
    
    output = Dense(1, kernel_initializer='glorot_uniform')(dense)
    
    return Model(inputs=[cgm_input, other_input], outputs=output)