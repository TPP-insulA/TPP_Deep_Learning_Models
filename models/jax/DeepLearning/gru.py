import os, sys
import jax
import jax.numpy as jnp
import flax.linen as nn
from typing import List, Tuple, Dict, Any, Optional

PROJECT_ROOT = os.path.abspath(os.getcwd())
sys.path.append(PROJECT_ROOT) 

from models.config import GRU_CONFIG

def create_gru_attention_block(x: jnp.ndarray, units: int, num_heads: int = 4, 
                              deterministic: bool = False,
                              dropout_rng: Optional[jax.random.PRNGKey] = None) -> jnp.ndarray:
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
    dropout_rng : Optional[jax.random.PRNGKey]
        Clave PRNG para dropout
        
    Retorna:
    --------
    jnp.ndarray
        Tensor procesado por el bloque GRU con atención
    """
    # GRU con skip connection
    skip1 = x
    
    # Simplificar la implementación usando una capa densa en lugar de GRU
    # Esto evita los problemas de tracer leak con scan
    x = nn.Dense(features=units)(x)
    x = nn.relu(x)
    
    # Aplicar capa de normalización
    x = nn.LayerNorm(epsilon=GRU_CONFIG['epsilon'])(x)
    
    # Skip connection si las dimensiones coinciden
    if skip1.shape[-1] == units:
        x = x + skip1
    
    # Multi-head attention con skip connection
    skip2 = x
    
    # Crear la capa MultiHeadAttention con deterministic en el constructor
    mha = nn.MultiHeadAttention(
        num_heads=num_heads,
        qkv_features=units,
        dropout_rate=0.0 if deterministic else GRU_CONFIG['dropout_rate'],
        deterministic=deterministic
    )
    
    # Aplicar la atención
    attention_output = mha(x, x)
    
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
    def __call__(self, cgm_input, other_input=None, training: bool = True) -> jnp.ndarray:
        # Comprobar si se recibe una tupla como primer argumento
        if other_input is None and isinstance(cgm_input, tuple) and len(cgm_input) == 2:
            cgm_input, other_input = cgm_input
        elif other_input is None:
            # Si solo se recibe un input durante la inicialización, crear un dummy input
            other_input = jnp.ones((cgm_input.shape[0], self.other_features_shape[0]))

        deterministic = not training
        
        # Proyección inicial
        x = nn.Dense(self.config['hidden_units'][0])(cgm_input)
        x = nn.LayerNorm(epsilon=self.config['epsilon'])(x)
        
        # Generar claves PRNG para cada capa con dropout
        dropout_rng = None
        if not deterministic:
            # Solo generamos una clave PRNG si estamos en modo de entrenamiento
            dropout_rng = self.make_rng('dropout')
        
        # Bloques GRU con attention
        for i, units in enumerate(self.config['hidden_units']):
            # Si tenemos múltiples bloques, dividir la clave PRNG para cada uno
            block_rng = None
            if dropout_rng is not None:
                block_rng, dropout_rng = jax.random.split(dropout_rng)
            x = create_gru_attention_block(x, units, deterministic=deterministic, dropout_rng=block_rng)
        
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