import os, sys
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input, Conv1D, Dense, Dropout, LayerNormalization,
    BatchNormalization, Add, Activation, GlobalAveragePooling1D
)
from keras.saving import register_keras_serializable
from typing import Tuple, List, Dict, Any, Optional, Union

PROJECT_ROOT = os.path.abspath(os.getcwd())
sys.path.append(PROJECT_ROOT) 

from config.models_config import TCN_CONFIG

@register_keras_serializable()
class WeightNormalization(tf.keras.layers.Wrapper):
    """
    Normalización de pesos para capas convolucionales.
    
    Parámetros:
    -----------
    layer : tf.keras.layers.Layer
        Capa a la que aplicar normalización de pesos
    """
    def __init__(self, layer: tf.keras.layers.Layer, **kwargs):
        super().__init__(layer, **kwargs)
        self.layer = layer

    def build(self, input_shape: Tuple) -> None:
        """
        Construye la capa con normalización de pesos.
        
        Parámetros:
        -----------
        input_shape : Tuple
            Forma de entrada
        """
        self.layer.build(input_shape)
        self.g = self.add_weight(
            name='g',
            shape=(self.layer.filters,),
            initializer='ones',
            trainable=True
        )

    def call(self, inputs: tf.Tensor) -> tf.Tensor:
        """
        Aplica la capa con normalización de pesos.
        
        Parámetros:
        -----------
        inputs : tf.Tensor
            Tensor de entrada
            
        Retorna:
        --------
        tf.Tensor
            Tensor procesado
        """
        weights = self.layer.weights[0]
        norm = tf.sqrt(tf.sum(tf.square(weights), axis=[0, 1]) + 1e-5)
        self.layer.kernel = weights * (self.g / norm)
        outputs = self.layer.call(inputs)
        return outputs
    
    def get_config(self) -> Dict:
        """
        Obtiene la configuración de la capa.
        
        Retorna:
        --------
        Dict
            Diccionario de configuración
        """
        config = super().get_config()
        config.update({
            'layer': {
                'class_name': self.layer.__class__.__name__,
                'config': self.layer.get_config()
            }
        })
        return config

@register_keras_serializable()
class CausalPadding(tf.keras.layers.Layer):
    """
    Capa personalizada para padding causal.
    
    Parámetros:
    -----------
    padding_size : int
        Tamaño del padding
    """
    def __init__(self, padding_size: int, **kwargs):
        super().__init__(**kwargs)
        self.padding_size = padding_size

    def call(self, inputs: tf.Tensor) -> tf.Tensor:
        """
        Aplica padding causal a un tensor.
        
        Parámetros:
        -----------
        inputs : tf.Tensor
            Tensor de entrada
            
        Retorna:
        --------
        tf.Tensor
            Tensor con padding aplicado
        """
        return tf.pad(inputs, [[0, 0], [self.padding_size, 0], [0, 0]])

    def compute_output_shape(self, input_shape: Tuple) -> Tuple:
        """
        Calcula la forma de salida.
        
        Parámetros:
        -----------
        input_shape : Tuple
            Forma de entrada
            
        Retorna:
        --------
        Tuple
            Forma de salida
        """
        return (input_shape[0], input_shape[1] + self.padding_size, input_shape[2])
    
    def get_config(self) -> Dict:
        """
        Obtiene la configuración de la capa.
        
        Retorna:
        --------
        Dict
            Diccionario de configuración
        """
        config = super().get_config()
        config.update({
            'padding_size': self.padding_size
        })
        return config

def create_tcn_block(input_layer: tf.Tensor, filters: int, kernel_size: int, 
                    dilation_rate: int, dropout_rate: float) -> tf.Tensor:
    """
    Crea un bloque TCN (Temporal Convolutional Network).
    
    Parámetros:
    -----------
    input_layer : tf.Tensor
        Capa de entrada
    filters : int
        Número de filtros
    kernel_size : int
        Tamaño del kernel
    dilation_rate : int
        Tasa de dilatación
    dropout_rate : float
        Tasa de dropout
    
    Retorna:
    --------
    tf.Tensor
        Salida del bloque TCN
    """
    padding_size = (kernel_size - 1) * dilation_rate
    padded_input = CausalPadding(padding_size)(input_layer)
    
    # Convolución con weight normalization
    conv_layer = Conv1D(
        filters=filters * 2,  # Double for gating
        kernel_size=kernel_size,
        dilation_rate=dilation_rate,
        padding='valid',
        activation=None
    )
    
    if TCN_CONFIG['use_weight_norm']:
        conv_layer = WeightNormalization(conv_layer)
    
    conv = conv_layer(padded_input)
    
    # Gating mechanism (GLU)
    gate, linear = tf.split(conv, 2, axis=-1)
    gate = tf.nn.sigmoid(gate)
    conv = linear * gate
    
    # Normalization
    if TCN_CONFIG['use_layer_norm']:
        conv = LayerNormalization(epsilon=TCN_CONFIG['epsilon'])(conv)
    else:
        conv = BatchNormalization()(conv)
    
    # Spatial dropout
    if TCN_CONFIG['use_spatial_dropout']:
        conv = tf.keras.layers.SpatialDropout1D(dropout_rate)(conv)
    else:
        conv = Dropout(dropout_rate)(conv)
    
    # Residual connection
    if input_layer.shape[-1] == filters:
        cropped_input = input_layer[:, -conv.shape[1]:, :]
        if TCN_CONFIG['residual_dropout'] > 0:
            cropped_input = Dropout(TCN_CONFIG['residual_dropout'])(cropped_input)
        return Add()([conv, cropped_input])
    
    return conv

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
    if activation_name == 'relu':
        return tf.nn.relu(x)
    elif activation_name == 'gelu':
        return tf.nn.gelu(x)
    elif activation_name == 'selu':
        return tf.nn.selu(x)
    elif activation_name == 'sigmoid':
        return tf.nn.sigmoid(x)
    elif activation_name == 'tanh':
        return tf.nn.tanh(x)
    else:
        return tf.nn.relu(x)  # Default

def create_tcn_model(input_shape: tuple, other_features_shape: tuple) -> Model:
    """
    Crea un modelo TCN completo.
    
    Parámetros:
    -----------
    input_shape : tuple
        Forma de los datos CGM (muestras, pasos_temporales, características)
    other_features_shape : tuple
        Forma de otras características (muestras, características)
    
    Retorna:
    --------
    Model
        Modelo TCN compilado
    """
    cgm_input = Input(shape=input_shape[1:], name='cgm_input')
    other_input = Input(shape=(other_features_shape[1],), name='other_input')
    
    # Proyección inicial
    x = Conv1D(TCN_CONFIG['filters'][0], 1, padding='same')(cgm_input)
    
    # Bloques TCN con skip connections
    skip_connections = []
    
    for filters in TCN_CONFIG['filters']:
        for dilation_rate in TCN_CONFIG['dilations']:
            tcn_out = create_tcn_block(
                x,
                filters=filters,
                kernel_size=TCN_CONFIG['kernel_size'],
                dilation_rate=dilation_rate,
                dropout_rate=TCN_CONFIG['dropout_rate'][0]
            )
            skip_connections.append(tcn_out)
            x = tcn_out
    
    # Combinar skip connections con normalización
    if skip_connections:
        target_len = skip_connections[-1].shape[1]
        aligned_skips = [skip[:, -target_len:, :] for skip in skip_connections]
        x = Add()(aligned_skips)
        x = x / tf.sqrt(float(len(skip_connections)))  # Scale appropriately
    
    # Global pooling con concatenación de estadísticas
    avg_pool = GlobalAveragePooling1D()(x)
    max_pool = tf.keras.layers.GlobalMaxPooling1D()(x)
    x = tf.keras.layers.Concatenate()([avg_pool, max_pool])
    
    # Combinar con otras características
    x = tf.keras.layers.Concatenate()([x, other_input])
    
    # MLP final con residual connections
    skip = x
    
    # Usar la función de activación directamente en lugar de la capa Activation
    x = Dense(128)(x)
    x = apply_activation(x, TCN_CONFIG['activation'])
    x = LayerNormalization(epsilon=TCN_CONFIG['epsilon'])(x)
    x = Dropout(TCN_CONFIG['dropout_rate'][0])(x)
    
    x = Dense(128)(x)
    x = apply_activation(x, TCN_CONFIG['activation'])
    
    if skip.shape[-1] == 128:
        x = Add()([x, skip])
    
    x = Dense(64)(x)
    x = apply_activation(x, TCN_CONFIG['activation'])
    x = LayerNormalization(epsilon=TCN_CONFIG['epsilon'])(x)
    x = Dropout(TCN_CONFIG['dropout_rate'][1])(x)
    
    output = Dense(1)(x)
    
    return Model(inputs=[cgm_input, other_input], outputs=output)