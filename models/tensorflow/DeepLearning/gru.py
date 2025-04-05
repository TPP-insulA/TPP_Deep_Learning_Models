import os, sys
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input, GRU, Dense, Dropout, LayerNormalization, Concatenate,
    MultiHeadAttention, Add, GlobalAveragePooling1D
)
from typing import Tuple, Any, Dict

PROJECT_ROOT = os.path.abspath(os.getcwd())
sys.path.append(PROJECT_ROOT) 

from config.models_config import GRU_CONFIG

def create_gru_attention_block(x: tf.Tensor, units: int, num_heads: int = 4) -> tf.Tensor:
    """
    Crea un bloque GRU con self-attention y conexiones residuales.
    
    Parámetros:
    -----------
    x : tf.Tensor
        Tensor de entrada
    units : int
        Número de unidades GRU
    num_heads : int
        Número de cabezas de atención
        
    Retorna:
    --------
    tf.Tensor
        Tensor procesado por el bloque GRU con atención
    """
    # GRU con skip connection
    skip1 = x
    x = GRU(
        units,
        return_sequences=True,
        dropout=GRU_CONFIG['dropout_rate'],
        recurrent_dropout=GRU_CONFIG['recurrent_dropout']
    )(x)
    x = LayerNormalization(epsilon=GRU_CONFIG['epsilon'])(x)
    if skip1.shape[-1] == units:
        x = Add()([x, skip1])
    
    # Multi-head attention con skip connection
    skip2 = x
    attention_output = MultiHeadAttention(
        num_heads=num_heads,
        key_dim=units // num_heads,
        dropout=GRU_CONFIG['dropout_rate']  # Agregado dropout a la capa de atención
    )(x, x)
    x = LayerNormalization(epsilon=GRU_CONFIG['epsilon'])(attention_output + skip2)
    
    return x

def create_gru_model(cgm_shape: tuple, other_features_shape: tuple) -> Model:
    """
    Crea un modelo GRU avanzado con self-attention y conexiones residuales.
    
    Parámetros:
    -----------
    cgm_shape : tuple
        Forma de los datos CGM (muestras, pasos_temporales, características)
    other_features_shape : tuple
        Forma de otras características (muestras, características)
        
    Retorna:
    --------
    Model
        Modelo GRU compilado
    """
    # Entradas
    cgm_input = Input(shape=cgm_shape[1:])
    other_input = Input(shape=(other_features_shape[1],))
    
    # Proyección inicial
    x = Dense(GRU_CONFIG['hidden_units'][0])(cgm_input)
    x = LayerNormalization(epsilon=GRU_CONFIG['epsilon'])(x)
    
    # Bloques GRU con attention
    for units in GRU_CONFIG['hidden_units']:
        x = create_gru_attention_block(x, units)
    
    # Pooling global
    x = GlobalAveragePooling1D()(x)
    
    # Combinar con otras características
    combined = Concatenate()([x, other_input])
    
    # Red densa final con skip connections
    for units in [128, 64]:
        skip = combined
        x = Dense(units)(combined)
        x = tf.nn.relu(x)  # Cambio a tf.nn.relu para consistencia con JAX
        x = LayerNormalization(epsilon=GRU_CONFIG['epsilon'])(x)
        x = Dropout(GRU_CONFIG['dropout_rate'])(x)
        if skip.shape[-1] == units:
            combined = Add()([x, skip])
        else:
            combined = x
    
    # Capa de salida
    output = Dense(1)(combined)
    
    return Model(inputs=[cgm_input, other_input], outputs=output)