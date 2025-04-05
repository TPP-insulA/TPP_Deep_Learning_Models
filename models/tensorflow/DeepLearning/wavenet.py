import os, sys
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input, Conv1D, Dense, Dropout, Add, Activation,
    BatchNormalization, GlobalAveragePooling1D, Concatenate
)
from keras.saving import register_keras_serializable
from typing import Tuple, Dict, List, Any, Optional, Union

PROJECT_ROOT = os.path.abspath(os.getcwd())
sys.path.append(PROJECT_ROOT) 

from config.models_config import WAVENET_CONFIG

@register_keras_serializable()
class WavenetBlock(tf.keras.layers.Layer):
    """
    Bloque WaveNet mejorado con activaciones gated y escalado adaptativo.
    
    Parámetros:
    -----------
    filters : int
        Número de filtros para las convoluciones
    kernel_size : int
        Tamaño del kernel convolucional
    dilation_rate : int
        Tasa de dilatación para la convolución
    dropout_rate : float
        Tasa de dropout
    """
    def __init__(self, filters: int, kernel_size: int, dilation_rate: int, dropout_rate: float, **kwargs):
        super().__init__(**kwargs)
        self.filters = filters
        self.kernel_size = kernel_size
        self.dilation_rate = dilation_rate
        self.dropout_rate = dropout_rate
        
        # Gated convolutions
        self.filter_conv = Conv1D(
            filters=filters,
            kernel_size=kernel_size,
            dilation_rate=dilation_rate,
            padding='causal'
        )
        self.gate_conv = Conv1D(
            filters=filters,
            kernel_size=kernel_size,
            dilation_rate=dilation_rate,
            padding='causal'
        )
        
        # Normalization and regularization
        self.filter_norm = BatchNormalization()
        self.gate_norm = BatchNormalization()
        self.dropout = Dropout(dropout_rate)
        
        # Projections
        self.residual_proj = Conv1D(filters, 1, padding='same')
        self.skip_proj = Conv1D(filters, 1, padding='same')
        
        # Scaling factors
        self.residual_scale = WAVENET_CONFIG['use_residual_scale']
        self.use_skip_scale = WAVENET_CONFIG['use_skip_scale']

    def call(self, inputs: tf.Tensor, training: Optional[bool] = None) -> Tuple[tf.Tensor, tf.Tensor]:
        """
        Aplica un bloque WaveNet a la entrada.
        
        Parámetros:
        -----------
        inputs : tf.Tensor
            Tensor de entrada
        training : bool, opcional
            Indica si está en modo entrenamiento
            
        Retorna:
        --------
        Tuple[tf.Tensor, tf.Tensor]
            (salida_residual, salida_skip)
        """
        # Gated activation
        filter_out = self.filter_conv(inputs)
        gate_out = self.gate_conv(inputs)
        
        filter_out = self.filter_norm(filter_out, training=training)
        gate_out = self.gate_norm(gate_out, training=training)
        
        # tanh(filter) * sigmoid(gate)
        gated_out = tf.nn.tanh(filter_out) * tf.nn.sigmoid(gate_out)
        gated_out = self.dropout(gated_out, training=training)
        
        # Residual connection
        residual = self.residual_proj(inputs)
        residual = residual[:, -gated_out.shape[1]:, :]
        residual_out = (gated_out * self.residual_scale) + residual
        
        # Skip connection
        skip_out = self.skip_proj(gated_out)
        if self.use_skip_scale:
            skip_out = skip_out * tf.math.sqrt(self.residual_scale)
        
        return residual_out, skip_out
    
    def get_config(self) -> Dict[str, Any]:
        """
        Obtiene la configuración del bloque.
        
        Retorna:
        --------
        Dict[str, Any]
            Configuración del bloque
        """
        config = super().get_config()
        config.update({
            'filters': self.filters,
            'kernel_size': self.kernel_size,
            'dilation_rate': self.dilation_rate,
            'dropout_rate': self.dropout_rate
        })
        return config

def create_wavenet_block(x: tf.Tensor, filters: int, kernel_size: int, 
                        dilation_rate: int, dropout_rate: float) -> Tuple[tf.Tensor, tf.Tensor]:
    """
    Crea un bloque WaveNet con conexiones residuales y skip connections.

    Parámetros:
    -----------
    x : tf.Tensor
        Tensor de entrada
    filters : int
        Número de filtros de la capa convolucional
    kernel_size : int
        Tamaño del kernel de la capa convolucional
    dilation_rate : int
        Tasa de dilatación de la capa convolucional
    dropout_rate : float
        Tasa de dropout

    Retorna:
    --------
    Tuple[tf.Tensor, tf.Tensor]
        (salida_residual, salida_skip)
    """
    # Convolución dilatada
    conv = Conv1D(filters=filters, kernel_size=kernel_size,
                 dilation_rate=dilation_rate, padding='causal')(x)
    conv = BatchNormalization()(conv)
    conv = Activation('relu')(conv)
    conv = Dropout(dropout_rate)(conv)
    
    # Conexión residual con proyección 1x1 si es necesario
    if x.shape[-1] != filters:
        x = Conv1D(filters, 1, padding='same')(x)
    
    # Alinear dimensiones temporales
    x = x[:, -conv.shape[1]:, :]
    res = Add()([conv, x])
    
    return res, conv

def apply_activation(x: tf.Tensor, activation_name: str) -> tf.Tensor:
    """
    Aplica la función de activación según su nombre.
    
    Parámetros:
    -----------
    x : tf.Tensor
        Tensor de entrada
    activation_name : str
        Nombre de la activación
        
    Retorna:
    --------
    tf.Tensor
        Tensor con activación aplicada
    """
    if activation_name == 'relu':
        return tf.nn.relu(x)
    elif activation_name == 'gelu':
        return tf.nn.gelu(x)
    else:
        return tf.nn.relu(x)  # Default

def create_wavenet_model(cgm_shape: tuple, other_features_shape: tuple) -> Model:
    """
    Crea un modelo WaveNet para predicción de series temporales.

    Parámetros:
    -----------
    cgm_shape : tuple
        Forma de los datos CGM (muestras, pasos_temporales, características)
    other_features_shape : tuple
        Forma de otras características (muestras, características)

    Retorna:
    --------
    Model
        Modelo WaveNet compilado
    """
    cgm_input = Input(shape=cgm_shape[1:])
    other_input = Input(shape=(other_features_shape[1],))
    
    # Proyección inicial
    x = Conv1D(WAVENET_CONFIG['filters'][0], 1, padding='same')(cgm_input)
    
    # Saltar conexiones
    skip_outputs = []
    
    # WaveNet stack
    for filters in WAVENET_CONFIG['filters']:
        for dilation in WAVENET_CONFIG['dilations']:
            wavenet = WavenetBlock(
                filters=filters,
                kernel_size=WAVENET_CONFIG['kernel_size'],
                dilation_rate=dilation,
                dropout_rate=WAVENET_CONFIG['dropout_rate']
            )
            x, skip = wavenet(x)
            skip_outputs.append(skip)
    
    # Combinar skip connections
    if skip_outputs:
        target_len = skip_outputs[-1].shape[1]
        aligned_skips = [skip[:, -target_len:, :] for skip in skip_outputs]
        x = Add()(aligned_skips)
        x = x / tf.sqrt(float(len(skip_outputs)))  # Escalar apropiadamente
    
    # Post-procesamiento
    x = apply_activation(x, WAVENET_CONFIG['activation'])
    x = Conv1D(WAVENET_CONFIG['filters'][-1], 1, padding='same')(x)
    x = apply_activation(x, WAVENET_CONFIG['activation'])
    x = GlobalAveragePooling1D()(x)
    
    # Combinación con otras features
    x = Concatenate()([x, other_input])
    
    # Capas densas finales con residual connections
    skip = x
    x = Dense(128)(x)
    x = BatchNormalization()(x)
    x = apply_activation(x, WAVENET_CONFIG['activation'])
    x = Dropout(WAVENET_CONFIG['dropout_rate'])(x)
    x = Dense(128)(x)
    
    if skip.shape[-1] == 128:
        x = Add()([x, skip])
    
    output = Dense(1)(x)
    
    return Model(inputs=[cgm_input, other_input], outputs=output)