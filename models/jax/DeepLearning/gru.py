import os, sys
import jax
import jax.numpy as jnp
import flax.linen as nn
from typing import List, Tuple, Dict, Any, Optional

PROJECT_ROOT = os.path.abspath(os.getcwd())
sys.path.append(PROJECT_ROOT) 

from config.models_config import GRU_CONFIG

def create_gru_attention_block(x: jnp.ndarray, units: int, num_heads: int = 4, 
                              deterministic: bool = False) -> jnp.ndarray:
    """
    Crea un bloque GRU con self-attention y conexiones residuales.
    
    Parámetros:
    -----------
    x : jnp.ndarray
        Tensor de entrada
    units : int
        Número de unidades GRU
    num_heads : int
        Número de cabezas de atención
    deterministic : bool
        Indica si está en modo de inferencia (no aplicar dropout)
        
    Retorna:
    --------
    jnp.ndarray
        Tensor procesado por el bloque GRU con atención
    """
    # GRU con skip connection
    skip1 = x
    
    # Definir y aplicar GRU
    gru = nn.scan(nn.GRUCell, 
                  variable_broadcast="params", 
                  split_rngs={"params": False, "dropout": True})
    
    batch_size, _, _ = x.shape
    x = x.transpose(1, 0, 2)  # Cambiar a [seq_len, batch_size, features] para scan
    
    # Crear estado inicial
    carry = jnp.zeros((batch_size, units))
    
    # Aplicar GRU con dropout
    _, x = gru()(
        carry, 
        x,
        dropout_rate=GRU_CONFIG['dropout_rate'],
        recurrent_dropout_rate=GRU_CONFIG['recurrent_dropout'],
        deterministic=deterministic
    )
    
    x = x.transpose(1, 0, 2)  # Volver a [batch_size, seq_len, features]
    x = nn.LayerNorm(epsilon=GRU_CONFIG['epsilon'])(x)
    
    # Skip connection si las dimensiones coinciden
    if skip1.shape[-1] == units:
        x = x + skip1
    
    # Multi-head attention con skip connection
    skip2 = x
    attention_output = nn.MultiHeadAttention(
        num_heads=num_heads,
        key_size=units // num_heads,
        dropout_rate=GRU_CONFIG['dropout_rate'],
        deterministic=deterministic
    )(x, x)
    
    x = nn.LayerNorm(epsilon=GRU_CONFIG['epsilon'])(attention_output + skip2)
    
    return x

class GRUModel(nn.Module):
    """
    Modelo GRU avanzado con self-attention y conexiones residuales.
    
    Parámetros:
    -----------
    config : Dict
        Diccionario con configuración del modelo
    cgm_shape : Tuple
        Forma de los datos CGM
    other_features_shape : Tuple
        Forma de otras características
    """
    config: Dict
    cgm_shape: Tuple
    other_features_shape: Tuple
    
    @nn.compact
    def __call__(self, inputs: Tuple[jnp.ndarray, jnp.ndarray], training: bool = True) -> jnp.ndarray:
        cgm_input, other_input = inputs
        deterministic = not training
        
        # Proyección inicial
        x = nn.Dense(self.config['hidden_units'][0])(cgm_input)
        x = nn.LayerNorm(epsilon=self.config['epsilon'])(x)
        
        # Bloques GRU con attention
        for units in self.config['hidden_units']:
            x = create_gru_attention_block(x, units, deterministic=deterministic)
        
        # Pooling global
        x = jnp.mean(x, axis=1)  # Equivalente a GlobalAveragePooling1D
        
        # Combinar con otras características
        combined = jnp.concatenate([x, other_input], axis=-1)
        
        # Red densa final con skip connections
        for units in [128, 64]:
            skip = combined
            x = nn.Dense(units)(combined)
            x = nn.relu(x)
            x = nn.LayerNorm(epsilon=self.config['epsilon'])(x)
            x = nn.Dropout(rate=self.config['dropout_rate'], deterministic=deterministic)(x)
            
            if skip.shape[-1] == units:
                combined = x + skip
            else:
                combined = x
        
        # Capa de salida
        output = nn.Dense(1)(combined)
        
        return output

def create_gru_model(cgm_shape: tuple, other_features_shape: tuple) -> GRUModel:
    """
    Crea un modelo GRU avanzado con self-attention y conexiones residuales con JAX/Flax.
    
    Parámetros:
    -----------
    cgm_shape : tuple
        Forma de los datos CGM (muestras, pasos_temporales, características)
    other_features_shape : tuple
        Forma de otras características (muestras, características)
        
    Retorna:
    --------
    gru_model
        Modelo GRU inicializado
    """
    model = GRUModel(
        config=GRU_CONFIG,
        cgm_shape=cgm_shape,
        other_features_shape=other_features_shape
    )
    
    return model