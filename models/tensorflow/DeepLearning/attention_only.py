import os, sys
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input, Dense, Dropout, LayerNormalization,
    MultiHeadAttention, GlobalAveragePooling1D, GlobalMaxPooling1D, 
    Concatenate, Add
)
from keras.saving import register_keras_serializable
from typing import Dict, Tuple, Any, Optional, Callable, Union

PROJECT_ROOT = os.path.abspath(os.getcwd())
sys.path.append(PROJECT_ROOT) 

from config.models_config import ATTENTION_CONFIG
from custom.dl_model_wrapper import DLModelWrapper, DLModelWrapperTF

# Constantes para cadenas repetidas
CONST_EPSILON = 1e-6

@register_keras_serializable()
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

@register_keras_serializable()
class ReshapeLayer(tf.keras.layers.Layer):
    """
    Capa personalizada para operaciones de reshape.
    
    Parámetros:
    -----------
    target_shape : tuple
        Forma objetivo para el tensor de salida
    """
    
    def __init__(self, target_shape: tuple, **kwargs):
        super().__init__(**kwargs)
        self.target_shape = target_shape
    
    def call(self, inputs: tf.Tensor) -> tf.Tensor:
        """
        Aplica la operación de reshape al tensor de entrada.
        
        Parámetros:
        -----------
        inputs : tf.Tensor
            Tensor de entrada
            
        Retorna:
        --------
        tf.Tensor
            Tensor con la forma modificada
        """
        return tf.reshape(inputs, self.target_shape)
    
    def get_config(self) -> Dict:
        """
        Obtiene la configuración de la capa para serialización.
        
        Retorna:
        --------
        Dict
            Diccionario con la configuración de la capa
        """
        config = super().get_config()
        config.update({"target_shape": self.target_shape})
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
        if self.activation_name == 'relu':
            return tf.nn.relu(inputs)
        elif self.activation_name == 'gelu':
            return tf.nn.gelu(inputs)
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
        
        # Modificado: eliminado el parámetro attention_bias que no está soportado
        mha = MultiHeadAttention(
            num_heads=num_heads,
            key_dim=key_dim,
            value_dim=ATTENTION_CONFIG['head_size']
        )
        # Aplicar atención primero, luego combinar con la codificación posicional manualmente
        attention_output = mha(x, x)
        
        # Si es necesario incorporar la codificación posicional, hacerlo de otra manera
        # como una suma directa o concatenación según sea apropiado
    else:
        attention_output = MultiHeadAttention(
            num_heads=num_heads,
            key_dim=key_dim
        )(x, x)
    
    # Mecanismo de gating
    gate = Dense(attention_output.shape[-1], activation='sigmoid')(x)
    attention_output = gate * attention_output
    
    attention_output = Dropout(dropout_rate)(attention_output, training=training)
    x = LayerNormalization(epsilon=CONST_EPSILON)(x + attention_output)
    
    # Red feed-forward mejorada con GLU
    ffn = Dense(ff_dim)(x)
    ffn_gate = Dense(ff_dim, activation='sigmoid')(x)
    ffn = ffn * ffn_gate
    ffn = Dense(x.shape[-1])(ffn)
    ffn = Dropout(dropout_rate)(ffn, training=training)
    
    return LayerNormalization(epsilon=CONST_EPSILON)(x + ffn)

def create_attention_model(cgm_shape: Tuple[int, ...], other_features_shape: Tuple[int, ...]) -> Model:
    """
    Crea un modelo basado únicamente en mecanismos de atención.

    Parámetros:
    -----------
    cgm_shape : Tuple[int, ...]
        Forma de los datos CGM (pasos_temporales, características) sin incluir el batch
    other_features_shape : Tuple[int, ...]
        Forma de otras características (características) sin incluir el batch

    Retorna:
    --------
    Model
        Modelo de atención compilado
    """
    # Validar la forma de entrada - debe tener al menos 2 dimensiones (tiempo, características)
    if len(cgm_shape) < 2:
        raise ValueError(f"La entrada CGM debe tener al menos 2 dimensiones: (timesteps, features). Recibido: {cgm_shape}")
    
    # Definir entrada CGM con la forma correcta (tiempo, características)
    cgm_input = Input(shape=cgm_shape)
    
    # Manejar diferentes formatos de other_features_shape
    if len(other_features_shape) == 1:
        # Si es una tupla con un solo elemento (por ejemplo: (6,))
        other_input = Input(shape=other_features_shape)
    else:
        # Si es una tupla con más elementos (asumimos que ya está en formato correcto)
        other_input = Input(shape=other_features_shape)
    
    # Proyección inicial - preservar la dimensionalidad temporal
    x = Dense(ATTENTION_CONFIG['key_dim'] * ATTENTION_CONFIG['num_heads'])(cgm_input)
    
    # Parámetros del modelo
    embedding_dim = ATTENTION_CONFIG['key_dim'] * ATTENTION_CONFIG['num_heads']
    
    # Stochastic depth (dropout de capas)
    survive_rates = tf.linspace(1.0, 0.5, ATTENTION_CONFIG['num_layers'])
    
    # Apilar bloques de atención con stochastic depth
    for i in range(ATTENTION_CONFIG['num_layers']):
        # Crear bloque de atención preservando dimensionalidad 3D
        block_output = create_attention_block(
            x,
            ATTENTION_CONFIG['num_heads'],
            ATTENTION_CONFIG['key_dim'],
            ATTENTION_CONFIG['ff_dim'],
            ATTENTION_CONFIG['dropout_rate'],
            training=True  # Por defecto asumimos entrenamiento
        )
        
        # Aplicar stochastic depth - técnica de regularización
        x = x + survive_rates[i] * (block_output - x)
    
    # Verificar que x tiene 3 dimensiones antes de aplicar pooling
    # En este punto x debe tener forma (batch_size, time_steps, features)
    if len(x.shape) != 3:
        raise ValueError(f"El tensor x debe tener 3 dimensiones para GlobalAveragePooling1D, pero tiene forma {x.shape}")
    
    # Contexto global
    attention_pooled = GlobalAveragePooling1D()(x)
    max_pooled = GlobalMaxPooling1D()(x)
    x = Concatenate()([attention_pooled, max_pooled])
    
    # Combinar con otras características
    x = Concatenate()([x, other_input])
    
    # MLP final con conexión residual
    skip = x
    x = Dense(128)(x)
    x = ActivationLayer(activation_name=ATTENTION_CONFIG['activation'])(x)
    x = LayerNormalization(epsilon=CONST_EPSILON)(x)
    x = Dropout(ATTENTION_CONFIG['dropout_rate'])(x)
    x = Dense(128)(x)
    x = ActivationLayer(activation_name=ATTENTION_CONFIG['activation'])(x)
    
    if skip.shape[-1] == 128:
        x = Add()([x, skip])
    
    output = Dense(1)(x)
    
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
        Función que crea un modelo de atención sin argumentos
    """
    def model_creator() -> Model:
        """
        Crea un modelo de atención sin argumentos.
        
        Retorna:
        --------
        Model
            Modelo de atención compilado
        """
        return create_attention_model(cgm_shape, other_features_shape)
    
    return model_creator

def create_attention_model_wrapper(cgm_shape: Tuple[int, ...], other_features_shape: Tuple[int, ...]) -> DLModelWrapper:
    """
    Crea un modelo de atención envuelto en DLModelWrapperTF.
    
    Parámetros:
    -----------
    cgm_shape : Tuple[int, ...]
        Forma de los datos CGM (muestras, pasos_temporales, características)
    other_features_shape : Tuple[int, ...]
        Forma de otras características (muestras, características)
        
    Retorna:
    --------
    DLModelWrapper
        Modelo de atención envuelto en DLModelWrapper
    """
    return DLModelWrapper(create_model_creator(cgm_shape, other_features_shape), 'tensorflow')