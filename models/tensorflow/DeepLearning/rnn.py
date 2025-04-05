import os, sys
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input, Dense, SimpleRNN, Dropout, BatchNormalization,
    Concatenate, Bidirectional, TimeDistributed, MaxPooling1D
)
from keras.saving import register_keras_serializable
from typing import Tuple, Dict, Any, Optional, List, Callable

PROJECT_ROOT = os.path.abspath(os.getcwd())
sys.path.append(PROJECT_ROOT) 

from config.models_config import RNN_CONFIG

def get_activation_fn(activation_name: str) -> Callable:
    """
    Obtiene la función de activación según su nombre.
    
    Parámetros:
    -----------
    activation_name : str
        Nombre de la función de activación
        
    Retorna:
    --------
    Callable
        Función de activación
    """
    if activation_name == 'relu':
        return tf.nn.relu
    elif activation_name == 'tanh':
        return tf.nn.tanh
    elif activation_name == 'sigmoid':
        return tf.nn.sigmoid
    elif activation_name == 'swish':
        return tf.nn.swish
    else:
        return tf.nn.relu  # Por defecto

def create_rnn_model(cgm_shape: tuple, other_features_shape: tuple) -> Model:
    """
    Crea un modelo RNN optimizado para velocidad con procesamiento temporal distribuido.
    
    Parámetros:
    -----------
    cgm_shape : tuple
        Forma de los datos CGM (samples, timesteps, features)
    other_features_shape : tuple
        Forma de otras características (samples, features)
        
    Retorna:
    --------
    Model
        Modelo RNN compilado
    """
    # Entradas
    cgm_input = Input(shape=cgm_shape[1:])
    other_input = Input(shape=(other_features_shape[1],))
    
    # Procesamiento temporal distribuido inicial
    if RNN_CONFIG['use_time_distributed']:
        # Usar función de activación explícita en lugar de string
        x = TimeDistributed(Dense(32))(cgm_input)
        x = tf.keras.layers.Activation(get_activation_fn(RNN_CONFIG['activation']))(x)
        x = TimeDistributed(BatchNormalization(epsilon=RNN_CONFIG['epsilon']))(x)
    else:
        x = cgm_input
    
    # Reducir secuencia temporal para procesamiento más rápido
    x = MaxPooling1D(pool_size=2)(x)
    
    # Capas RNN con menos unidades pero bidireccionales
    for units in RNN_CONFIG['hidden_units']:
        rnn_layer = SimpleRNN(
            units,
            activation=RNN_CONFIG['activation'],
            dropout=RNN_CONFIG['dropout_rate'],
            recurrent_dropout=RNN_CONFIG['recurrent_dropout'],
            return_sequences=True,
            unroll=True  # Desenrollar para secuencias cortas
        )
        
        if RNN_CONFIG['bidirectional']:
            x = Bidirectional(rnn_layer)(x)
        else:
            x = rnn_layer(x)
            
        x = BatchNormalization(
            epsilon=RNN_CONFIG['epsilon'],
            momentum=0.9  # Aumentar momentum para actualización más rápida
        )(x)
    
    # Último RNN sin return_sequences
    final_rnn = SimpleRNN(
        RNN_CONFIG['hidden_units'][-1],
        activation=RNN_CONFIG['activation'],
        dropout=RNN_CONFIG['dropout_rate'],
        recurrent_dropout=RNN_CONFIG['recurrent_dropout'],
        unroll=True
    )
    
    if RNN_CONFIG['bidirectional']:
        x = Bidirectional(final_rnn)(x)
    else:
        x = final_rnn(x)
    
    # Combinar características
    x = Concatenate()([x, other_input])
    
    # Reducir capas densas
    x = Dense(32)(x)
    x = tf.keras.layers.Activation(get_activation_fn(RNN_CONFIG['activation']))(x)
    x = BatchNormalization(epsilon=RNN_CONFIG['epsilon'])(x)
    x = Dropout(RNN_CONFIG['dropout_rate'])(x)
    
    output = Dense(1)(x)
    
    return Model(inputs=[cgm_input, other_input], outputs=output)