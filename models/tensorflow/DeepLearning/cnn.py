import os, sys
import tensorflow as tf
from keras._tf_keras.keras.models import Model
from keras._tf_keras.keras.layers import (
    Input, Conv1D, Dense, Dropout, BatchNormalization, 
    LayerNormalization, Concatenate, Add, Activation, 
    GlobalAveragePooling1D, GlobalMaxPooling1D, SeparableConv1D
)
from keras._tf_keras.keras.saving import register_keras_serializable
from typing import Dict, Tuple, Any, Optional, Callable, Union, List

PROJECT_ROOT = os.path.abspath(os.getcwd())
sys.path.append(PROJECT_ROOT) 

from config.models_config import CNN_CONFIG
from custom.dl_model_wrapper import DLModelWrapper, DLModelWrapperTF

# Constantes para cadenas repetidas
CONST_EPSILON = 1e-6
CONST_RELU = "relu"
CONST_GELU = "gelu"
CONST_SWISH = "swish"
CONST_SILU = "silu"

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
    
    def call(self, inputs: tf.Tensor, training: Optional[bool] = None) -> tf.Tensor:
        """
        Aplica el mecanismo de Squeeze-and-Excitation a los inputs.
        
        Parámetros:
        -----------
        inputs : tf.Tensor
            Tensor de entrada
        training : bool, opcional
            Indica si está en modo entrenamiento
            
        Retorna:
        --------
        tf.Tensor
            Tensor procesado con atención de canal
        """
        # Squeeze
        x = self.gap(inputs)
        
        # Excitation
        x = self.dense1(x)
        x = self.dense2(x)
        
        # Scale - reshape para broadcasting con inputs
        x = tf.expand_dims(x, axis=1)
        
        return inputs * x
    
    def get_config(self) -> Dict:
        """
        Obtiene la configuración de la capa para serialización.
        
        Retorna:
        --------
        Dict
            Diccionario con la configuración de la capa
        """
        config = super().get_config()
        config.update({
            "filters": self.filters,
            "se_ratio": self.se_ratio
        })
        return config

@register_keras_serializable()
class ActivationLayer(tf.keras.layers.Layer):
    """
    Capa personalizada para funciones de activación.
    
    Parámetros:
    -----------
    activation_name : str
        Nombre de la función de activación a aplicar
    """
    
    def __init__(self, activation_name: str, **kwargs):
        super().__init__(**kwargs)
        self.activation_name = activation_name
    
    def call(self, inputs: tf.Tensor) -> tf.Tensor:
        """
        Aplica la función de activación especificada al tensor de entrada.
        
        Parámetros:
        -----------
        inputs : tf.Tensor
            Tensor de entrada
            
        Retorna:
        --------
        tf.Tensor
            Tensor con la activación aplicada
        """
        return get_activation(inputs, self.activation_name)
    
    def get_config(self) -> Dict:
        """
        Obtiene la configuración de la capa para serialización.
        
        Retorna:
        --------
        Dict
            Diccionario con la configuración de la capa
        """
        config = super().get_config()
        config.update({"activation_name": self.activation_name})
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
    if activation_name == CONST_RELU:
        return tf.nn.relu(x)
    elif activation_name == CONST_GELU:
        return tf.nn.gelu(x)
    elif activation_name == CONST_SWISH:
        return tf.nn.swish(x)
    elif activation_name == CONST_SILU:
        return tf.nn.silu(x)
    else:
        return tf.nn.relu(x)  # Valor por defecto

def create_residual_block(x: tf.Tensor, filters: int, 
                        kernel_size: int, dilation_rate: int = 1, 
                        dropout_rate: float = 0.1,
                        use_se_block: bool = True, 
                        se_ratio: int = 16) -> tf.Tensor:
    """
    Crea un bloque residual con convoluciones dilatadas y SE.
    
    Parámetros:
    -----------
    x : tf.Tensor
        Tensor de entrada
    filters : int
        Número de filtros para la convolución
    kernel_size : int
        Tamaño del kernel para la convolución
    dilation_rate : int, opcional
        Tasa de dilatación para las convoluciones (default: 1)
    dropout_rate : float, opcional
        Tasa de dropout a aplicar (default: 0.1)
    use_se_block : bool, opcional
        Si se debe usar un bloque Squeeze-and-Excitation (default: True)
    se_ratio : int, opcional
        Factor de reducción para el bloque SE (default: 16)
        
    Retorna:
    --------
    tf.Tensor
        Tensor procesado con conexión residual
    """
    # Guardar entrada para la conexión residual
    skip = x
    
    # Primera convolución con dilatación
    x = Conv1D(
        filters=filters,
        kernel_size=kernel_size,
        padding='same',
        dilation_rate=dilation_rate
    )(x)
    
    # Normalización y activación
    if CNN_CONFIG['use_layer_norm']:
        x = LayerNormalization(epsilon=CONST_EPSILON)(x)
    else:
        x = BatchNormalization()(x)
    
    x = Activation(CNN_CONFIG['activation'])(x)
    
    # Segunda convolución separable para reducir parámetros
    x = SeparableConv1D(
        filters=filters,
        kernel_size=kernel_size,
        padding='same'
    )(x)
    
    # Normalización
    if CNN_CONFIG['use_layer_norm']:
        x = LayerNormalization(epsilon=CONST_EPSILON)(x)
    else:
        x = BatchNormalization()(x)
    
    # Squeeze-and-Excitation si está habilitado
    if use_se_block:
        x = SqueezeExcitationBlock(filters, se_ratio)(x)
    
    # Dropout
    x = Dropout(dropout_rate)(x)
    
    # Conexión residual con proyección si es necesario
    if skip.shape[-1] != filters:
        skip = Conv1D(filters, 1, padding='same')(skip)
        
        if CNN_CONFIG['use_layer_norm']:
            skip = LayerNormalization(epsilon=CONST_EPSILON)(skip)
        else:
            skip = BatchNormalization()(skip)
    
    # Sumar con la conexión residual
    x = Add()([x, skip])
    
    # Activación final
    x = Activation(CNN_CONFIG['activation'])(x)
    
    return x

def apply_normalization(x: tf.Tensor) -> tf.Tensor:
    """Aplica normalización según la configuración"""
    if CNN_CONFIG['use_layer_norm']:
        return LayerNormalization(epsilon=CONST_EPSILON)(x)
    else:
        return BatchNormalization()(x)

def apply_conv_block(x: tf.Tensor, filters: int, kernel_size: int, strides: int = 1) -> tf.Tensor:
    """Aplica un bloque de convolución con normalización y activación"""
    x = Conv1D(filters=filters, kernel_size=kernel_size, strides=strides, padding='same')(x)
    x = apply_normalization(x)
    x = Activation(CNN_CONFIG['activation'])(x)
    return x

def apply_residual_blocks(x: tf.Tensor) -> tf.Tensor:
    """Aplica bloques residuales con diferentes tasas de dilatación y filtros"""
    for i, filters in enumerate(CNN_CONFIG['filters']):
        # Aplicar bloques residuales con diferentes tasas de dilatación
        for dilation_rate in CNN_CONFIG['dilation_rates']:
            x = create_residual_block(
                x, filters=filters, kernel_size=CNN_CONFIG['kernel_size'],
                dilation_rate=dilation_rate, dropout_rate=CNN_CONFIG['dropout_rate'],
                use_se_block=CNN_CONFIG['use_se_block'], se_ratio=CNN_CONFIG['se_ratio']
            )
        
        # Reducción opcional entre bloques de filtros (excepto el último)
        if i < len(CNN_CONFIG['filters']) - 1:
            x = apply_conv_block(
                x, filters=CNN_CONFIG['filters'][i+1], 
                kernel_size=CNN_CONFIG['pool_size'], 
                strides=CNN_CONFIG['pool_size']
            )
    
    return x

def create_mlp_head(x: tf.Tensor) -> tf.Tensor:
    """Crea la cabeza MLP para la clasificación final"""
    skip = x
    
    # Primera capa densa
    x = Dense(256)(x)
    x = ActivationLayer(activation_name=CNN_CONFIG['activation'])(x)
    x = apply_normalization(x)
    x = Dropout(CNN_CONFIG['dropout_rate'])(x)
    
    # Segunda capa densa
    x = Dense(256)(x)
    x = ActivationLayer(activation_name=CNN_CONFIG['activation'])(x)
    
    # Conexión residual si las dimensiones coinciden
    if skip.shape[-1] == 256:
        x = Add()([x, skip])
    
    # Capa final de proyección
    x = Dense(128)(x)
    x = ActivationLayer(activation_name=CNN_CONFIG['activation'])(x)
    x = apply_normalization(x)
    x = Dropout(CNN_CONFIG['dropout_rate'] / 2)(x)
    
    return Dense(1)(x)

def create_cnn_model(cgm_shape: Tuple[int, ...], other_features_shape: Tuple[int, ...]) -> Model:
    """
    Crea un modelo CNN avanzado con características modernas como bloques residuales,
    convoluciones dilatadas y Squeeze-and-Excitation.
    
    Parámetros:
    -----------
    cgm_shape : Tuple[int, ...]
        Forma de los datos CGM (pasos_temporales, características) sin incluir el batch
    other_features_shape : Tuple[int, ...]
        Forma de otras características (características) sin incluir el batch
        
    Retorna:
    --------
    Model
        Modelo CNN avanzado compilado
    """
    # Validar la forma de entrada
    if len(cgm_shape) < 2:
        raise ValueError(f"La entrada CGM debe tener al menos 2 dimensiones: (timesteps, features). Recibido: {cgm_shape}")
    
    # Crear entradas
    cgm_input = Input(shape=cgm_shape)
    other_input = Input(shape=other_features_shape)
    
    # Proyección inicial
    x = apply_conv_block(cgm_input, filters=CNN_CONFIG['filters'][0], kernel_size=1)
    
    # Bloques residuales
    x = apply_residual_blocks(x)
    
    # Pooling global
    avg_pool = GlobalAveragePooling1D()(x)
    max_pool = GlobalMaxPooling1D()(x)
    x = Concatenate()([avg_pool, max_pool, other_input])
    
    # MLP final
    output = create_mlp_head(x)
    
    return Model(inputs=[cgm_input, other_input], outputs=output)

def create_model_creator(cgm_shape: Tuple[int, ...], other_features_shape: Tuple[int, ...]) -> Callable[[], Model]:
    """
    Crea una función creadora de modelos compatible con DLModelWrapperTF.
    
    Parámetros:
    -----------
    cgm_shape : Tuple[int, ...]
        Forma de los datos CGM (muestras, pasos_temporales, características)
    other_features_shape : Tuple[int, ...]
        Forma de otras características (muestras, características)
        
    Retorna:
    --------
    Callable[[], Model]
        Función que crea un modelo CNN sin argumentos
    """
    def model_creator() -> Model:
        """
        Crea un modelo CNN sin argumentos.
        
        Retorna:
        --------
        Model
            Modelo CNN compilado
        """
        return create_cnn_model(cgm_shape, other_features_shape)
    
    return model_creator

def create_cnn_model_wrapper(cgm_shape: Tuple[int, ...], other_features_shape: Tuple[int, ...]) -> DLModelWrapper:
    """
    Crea un modelo CNN avanzado envuelto en DLModelWrapperTF.
    
    Parámetros:
    -----------
    cgm_shape : Tuple[int, ...]
        Forma de los datos CGM (muestras, pasos_temporales, características)
    other_features_shape : Tuple[int, ...]
        Forma de otras características (muestras, características)
        
    Retorna:
    --------
    DLModelWrapper
        Modelo CNN avanzado envuelto en DLModelWrapper
    """
    return DLModelWrapper(create_model_creator(cgm_shape, other_features_shape), 'tensorflow')