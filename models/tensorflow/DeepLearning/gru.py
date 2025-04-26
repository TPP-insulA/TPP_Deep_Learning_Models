import os, sys
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input, Dense, Dropout, LayerNormalization,
    GRU, Bidirectional, GlobalAveragePooling1D, 
    Concatenate, Add, Attention
)
from keras.saving import register_keras_serializable
from typing import Dict, Tuple, Any, Optional, Callable, Union

PROJECT_ROOT = os.path.abspath(os.getcwd())
sys.path.append(PROJECT_ROOT) 

from config.models_config import GRU_CONFIG
from custom.dl_model_wrapper import DLModelWrapper, DLModelWrapperTF

# Constantes para cadenas repetidas
CONST_EPSILON = 1e-6

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
        if self.activation_name == 'relu':
            return tf.nn.relu(inputs)
        elif self.activation_name == 'gelu':
            return tf.nn.gelu(inputs)
        elif self.activation_name == 'tanh':
            return tf.nn.tanh(inputs)
        elif self.activation_name == 'swish':
            return tf.nn.swish(inputs)
        elif self.activation_name == 'silu':
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

def create_gru_attention_block(x: tf.Tensor, units: int, bidirectional: bool = True, 
                             dropout_rate: float = 0.2, recurrent_dropout: float = 0.1, 
                             training: bool = True) -> tf.Tensor:
    """
    Crea un bloque GRU con mecanismo de atención para la secuencia.
    
    Parámetros:
    -----------
    x : tf.Tensor
        Tensor de entrada
    units : int
        Número de unidades GRU
    bidirectional : bool, opcional
        Si usar GRU bidireccional (default: True)
    dropout_rate : float, opcional
        Tasa de dropout para la capa GRU (default: 0.2)
    recurrent_dropout : float, opcional
        Tasa de dropout recurrente para la capa GRU (default: 0.1)
    training : bool, opcional
        Indica si está en modo entrenamiento (default: True)
        
    Retorna:
    --------
    tf.Tensor
        Tensor procesado con GRU y atención
    """
    # Capa GRU principal
    if bidirectional:
        gru_output = Bidirectional(
            GRU(
                units,
                dropout=dropout_rate,
                recurrent_dropout=recurrent_dropout,
                return_sequences=True
            )
        )(x, training=training)
    else:
        gru_output = GRU(
            units,
            dropout=dropout_rate,
            recurrent_dropout=recurrent_dropout,
            return_sequences=True
        )(x, training=training)
    
    # Normalización de capa con skip connection
    if x.shape[-1] == gru_output.shape[-1]:
        gru_output = LayerNormalization(epsilon=CONST_EPSILON)(x + gru_output)
    else:
        gru_output = LayerNormalization(epsilon=CONST_EPSILON)(gru_output)
    
    # Mecanismo de atención - corregido para asegurar dimensiones compatibles
    # Proyectar tanto query como value al mismo tamaño para compatibilidad
    attention_dim = units  # Usar el mismo número de unidades para la dimensión de atención
    
    query = Dense(attention_dim)(gru_output)
    value = Dense(attention_dim)(gru_output)
    
    # Ahora query y value tienen la misma dimensión final (attention_dim)
    attention_scores = Attention()([query, value])
    
    # Combinar mediante skip connection
    combined = LayerNormalization(epsilon=CONST_EPSILON)(attention_scores + query)
    
    # Dropout final
    return Dropout(dropout_rate)(combined, training=training)

def create_gru_model(cgm_shape: Tuple[int, ...], other_features_shape: Tuple[int, ...]) -> Model:
    """
    Crea un modelo GRU con atención para series temporales de CGM y otras características.
    
    Parámetros:
    -----------
    cgm_shape : Tuple[int, ...]
        Forma de los datos CGM (pasos_temporales, características) sin incluir el batch
    other_features_shape : Tuple[int, ...]
        Forma de otras características (características) sin incluir el batch
        
    Retorna:
    --------
    Model
        Modelo GRU con atención compilado
    """
    # Validar la forma de entrada - debe tener al menos 2 dimensiones (tiempo, características)
    if len(cgm_shape) < 2:
        raise ValueError(f"La entrada CGM debe tener al menos 2 dimensiones: (timesteps, features). Recibido: {cgm_shape}")
    
    # Definir entrada CGM con la forma correcta (tiempo, características)
    cgm_input = Input(shape=cgm_shape)
    
    # Manejar diferentes formatos de other_features_shape
    if isinstance(other_features_shape, tuple) and len(other_features_shape) == 1:
        # Si es una tupla con un solo elemento (por ejemplo: (6,))
        other_input = Input(shape=other_features_shape)
    else:
        # Si es una tupla con más elementos o no es una tupla
        other_input = Input(shape=(other_features_shape[0],))
    
    # Capas GRU apiladas con diferente número de unidades
    x = cgm_input
    
    # Proyección inicial si es necesario
    x = Dense(GRU_CONFIG['hidden_units'][0])(x)
    x = LayerNormalization(epsilon=CONST_EPSILON)(x)
    
    # Aplicar bloques GRU en secuencia con diferentes tamaños
    for units in GRU_CONFIG['hidden_units']:
        x = create_gru_attention_block(
            x,
            units=units,
            bidirectional=GRU_CONFIG.get('use_bidirectional', True),
            dropout_rate=GRU_CONFIG['dropout_rate'],
            recurrent_dropout=GRU_CONFIG['recurrent_dropout']
        )
    
    # Agregación de características temporales
    x = GlobalAveragePooling1D()(x)
    
    # Combinar con otras características
    x = Concatenate()([x, other_input])
    
    # MLP final con residual connections
    skip = x
    
    # Primera capa
    x = Dense(128)(x)
    x = ActivationLayer('relu')(x)
    x = LayerNormalization(epsilon=CONST_EPSILON)(x)
    x = Dropout(GRU_CONFIG['dropout_rate'])(x)
    
    # Segunda capa con skip connection si las dimensiones coinciden
    x = Dense(128)(x)
    x = ActivationLayer('relu')(x)
    if skip.shape[-1] == 128:
        x = Add()([x, skip])
    
    # Capa final
    x = Dense(64)(x)
    x = ActivationLayer('relu')(x)
    x = LayerNormalization(epsilon=CONST_EPSILON)(x)
    x = Dropout(GRU_CONFIG['dropout_rate'])(x)
    
    # Salida de predicción
    output = Dense(1)(x)
    
    return Model(inputs=[cgm_input, other_input], outputs=output)

def create_model_creator(cgm_shape: Tuple[int, ...], other_features_shape: Tuple[int, ...]) -> Callable[[], Model]:
    """
    Crea una función creadora de modelos GRU compatible con DLModelWrapperTF.
    
    Parámetros:
    -----------
    cgm_shape : Tuple[int, ...]
        Forma de los datos CGM (muestras, pasos_temporales, características)
    other_features_shape : Tuple[int, ...]
        Forma de otras características (muestras, características)
        
    Retorna:
    --------
    Callable[[], Model]
        Función que crea un modelo GRU sin argumentos
    """
    def model_creator() -> Model:
        """
        Crea un modelo GRU sin argumentos.
        
        Retorna:
        --------
        Model
            Modelo GRU compilado
        """
        return create_gru_model(cgm_shape, other_features_shape)
    
    return model_creator

def create_gru_model_wrapper(cgm_shape: Tuple[int, ...], other_features_shape: Tuple[int, ...]) -> DLModelWrapper:
    """
    Crea un modelo GRU envuelto en DLModelWrapperTF.
    
    Parámetros:
    -----------
    cgm_shape : Tuple[int, ...]
        Forma de los datos CGM (muestras, pasos_temporales, características)
    other_features_shape : Tuple[int, ...]
        Forma de otras características (muestras, características)
        
    Retorna:
    --------
    DLModelWrapper
        Modelo GRU envuelto en DLModelWrapper
    """
    return DLModelWrapper(create_model_creator(cgm_shape, other_features_shape), 'tensorflow')