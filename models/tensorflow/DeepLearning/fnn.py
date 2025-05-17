import os, sys
import tensorflow as tf
from keras._tf_keras.keras.models import Model
from keras._tf_keras.keras.layers import (
    Input, Dense, Dropout, BatchNormalization, LayerNormalization,
    Concatenate, Add, Flatten, Activation
)
from keras._tf_keras.keras.saving import register_keras_serializable
from typing import Dict, Tuple, Any, Optional, Callable, Union, List

PROJECT_ROOT = os.path.abspath(os.getcwd())
sys.path.append(PROJECT_ROOT) 

from config.models_config import FNN_CONFIG
from custom.DeepLearning.dl_model_wrapper import DLModelWrapper

# Constantes para cadenas repetidas
CONST_EPSILON = 1e-6
CONST_RELU = "relu"
CONST_GELU = "gelu"
CONST_SWISH = "swish"
CONST_SILU = "silu"

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
        if self.activation_name == CONST_RELU:
            return tf.nn.relu(inputs)
        elif self.activation_name == CONST_GELU:
            return tf.nn.gelu(inputs)
        elif self.activation_name == CONST_SWISH:
            return tf.nn.swish(inputs)
        elif self.activation_name == CONST_SILU:
            return tf.nn.silu(inputs)
        else:
            return tf.nn.relu(inputs)  # Valor por defecto
    
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
    Aplica la función de activación según su nombre usando capas de Keras.
    
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
    # Usar capas de Keras en lugar de funciones de TF directamente para compatibilidad
    if activation_name == CONST_RELU:
        return Activation('relu')(x)
    elif activation_name == CONST_GELU:
        return ActivationLayer(CONST_GELU)(x)
    elif activation_name == CONST_SWISH:
        return Activation('swish')(x)
    elif activation_name == CONST_SILU:
        return ActivationLayer(CONST_SILU)(x)
    else:
        return Activation('relu')(x)  # Valor por defecto

def create_residual_block(x: tf.Tensor, units: int, dropout_rate: float, use_layer_norm: bool = True, 
                        activation: str = CONST_RELU, training: Optional[bool] = None) -> tf.Tensor:
    """
    Crea un bloque residual con normalización y conexiones skip.
    
    Parámetros:
    -----------
    x : tf.Tensor
        Tensor de entrada
    units : int
        Número de unidades en la capa densa
    dropout_rate : float
        Tasa de dropout
    use_layer_norm : bool
        Indica si usar normalización de capa en lugar de normalización por lotes
    activation : str
        Función de activación a utilizar
    training : bool (opcional)
        Indica si está en modo entrenamiento (default: None)
    
    Retorna:
    --------
    tf.Tensor
        Tensor procesado con conexión residual
    """
    # Guardar entrada para la conexión residual
    skip = x
    
    # Primera capa densa
    x = Dense(units)(x)
    
    # Normalización
    if use_layer_norm:
        x = LayerNormalization(epsilon=CONST_EPSILON)(x)
    else:
        x = BatchNormalization()(x, training=training)
    
    # Activación
    x = get_activation(x, activation)
    x = Dropout(dropout_rate)(x, training=training)
    
    # Segunda capa densa
    x = Dense(units)(x)
    
    # Normalización
    if use_layer_norm:
        x = LayerNormalization(epsilon=CONST_EPSILON)(x)
    else:
        x = BatchNormalization()(x, training=training)
    
    # Proyección para la conexión residual si es necesario
    if skip.shape[-1] != units:
        skip = Dense(units)(skip)
    
    # Añadir conexión residual
    x = Add()([x, skip])
    
    # Activación final
    x = get_activation(x, activation)
    
    return x

def create_fnn_model(cgm_shape: Tuple[int, ...], other_features_shape: Tuple[int, ...]) -> Model:
    """
    Crea un modelo de red neuronal feedforward (FNN) con conexiones residuales.
    
    Parámetros:
    -----------
    cgm_shape : Tuple[int, ...]
        Forma de los datos CGM (pasos_temporales, características) sin incluir el batch
    other_features_shape : Tuple[int, ...]
        Forma de otras características (características) sin incluir el batch
    
    Retorna:
    --------
    Model
        Modelo FNN compilado
    """
    # Entradas para datos CGM y otras características
    cgm_input = Input(shape=cgm_shape)
    
    # Determinar la forma para other_input de manera segura
    other_features_dim = 1  # Valor predeterminado
    
    if isinstance(other_features_shape, tuple):
        if len(other_features_shape) == 1:
            # Si es una tupla con un solo elemento, usar ese valor
            other_features_dim = other_features_shape[0]
        elif len(other_features_shape) > 1:
            # Si es una tupla con múltiples elementos, probablemente (muestras, características)
            other_features_dim = other_features_shape[-1]  # Usar el último elemento
    
    other_input = Input(shape=(other_features_dim,))
    
    # Aplanar los datos CGM para procesamiento en FNN
    x = Flatten()(cgm_input)
    
    # Combinar con otras características
    x = Concatenate()([x, other_input])
    
    # Capas ocultas con bloques residuales
    for i, units in enumerate(FNN_CONFIG['hidden_units']):
        dropout_rate = FNN_CONFIG['dropout_rates'][i] if i < len(FNN_CONFIG['dropout_rates']) else FNN_CONFIG['dropout_rates'][-1]
        
        x = create_residual_block(
            x=x,
            units=units,
            dropout_rate=dropout_rate,
            use_layer_norm=FNN_CONFIG['use_layer_norm'],
            activation=FNN_CONFIG['activation'],
            training=None
        )
    
    # Capas finales para regresión
    for units in FNN_CONFIG['final_units']:
        x = Dense(units)(x)
        
        if FNN_CONFIG['use_layer_norm']:
            x = LayerNormalization(epsilon=FNN_CONFIG['epsilon'])(x)
        else:
            x = BatchNormalization()(x)
            
        x = get_activation(x, FNN_CONFIG['activation'])
        x = Dropout(FNN_CONFIG['final_dropout_rate'])(x)
    
    # Capa de salida para regresión
    output = Dense(1)(x)
    
    # Crear y retornar el modelo
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
        Función que crea un modelo FNN sin argumentos
    """
    def model_creator() -> Model:
        """
        Crea un modelo FNN sin argumentos.
        
        Retorna:
        --------
        Model
            Modelo FNN compilado
        """
        return create_fnn_model(cgm_shape, other_features_shape)
    
    return model_creator

def create_fnn_model_wrapper(cgm_shape: Tuple[int, ...], other_features_shape: Tuple[int, ...]) -> DLModelWrapper:
    """
    Crea un modelo FNN envuelto en DLModelWrapperTF.
    
    Parámetros:
    -----------
    cgm_shape : Tuple[int, ...]
        Forma de los datos CGM (muestras, pasos_temporales, características)
    other_features_shape : Tuple[int, ...]
        Forma de otras características (muestras, características)
        
    Retorna:
    --------
    DLModelWrapper
        Modelo FNN envuelto en DLModelWrapper
    """
    return DLModelWrapper(create_model_creator(cgm_shape, other_features_shape), 'tensorflow')