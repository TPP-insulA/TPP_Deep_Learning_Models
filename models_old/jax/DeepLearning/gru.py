import os, sys
import jax
import jax.numpy as jnp
import flax.linen as nn
import numpy as np
from typing import List, Tuple, Dict, Any, Optional, Callable, Union, Sequence

PROJECT_ROOT = os.path.abspath(os.getcwd())
sys.path.append(PROJECT_ROOT) 

from config.models_config import GRU_CONFIG, EARLY_STOPPING_POLICY
from custom.DeepLearning.dl_model_wrapper import DLModelWrapper
from models_old.early_stopping import get_early_stopping_config

# Constantes para uso repetido
CONST_DROPOUT = "dropout"
CONST_PARAMS = "params"
CONST_TRAINING = "training"

class GRUCell(nn.Module):
    """
    Implementación de una celda GRU (Gated Recurrent Unit).
    
    Parámetros:
    -----------
    features : int
        Número de unidades ocultas
    gate_fn : Callable
        Función de activación para las puertas
    activation_fn : Callable
        Función de activación para la salida
    """
    features: int
    gate_fn: Callable = nn.sigmoid
    activation_fn: Callable = nn.tanh
    
    @nn.compact
    def __call__(self, inputs: jnp.ndarray, carry: jnp.ndarray, deterministic: bool = True) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """
        Ejecuta un paso de procesamiento de la celda GRU.
        
        Parámetros:
        -----------
        inputs : jnp.ndarray
            Entrada actual
        carry : jnp.ndarray
            Estado oculto actual
        deterministic : bool, opcional
            Indica si está en modo de inferencia (default: True)
            
        Retorna:
        --------
        Tuple[jnp.ndarray, jnp.ndarray]
            (nuevo estado, salida)
        """
        h = carry
        
        # Capas densas para las compuertas
        gates = nn.Dense(features=2 * self.features)(inputs)
        reset, update = jnp.split(self.gate_fn(gates), 2, axis=-1)
        
        # Capa densa para el nuevo candidato
        h_reset = reset * h
        candidate_x = nn.Dense(features=self.features)(inputs)
        candidate_h = nn.Dense(features=self.features)(h_reset)
        candidate = self.activation_fn(candidate_x + candidate_h)
        
        # Actualizar estado
        new_h = update * h + (1.0 - update) * candidate
        
        return new_h, new_h


class SimplifiedGRULayer(nn.Module):
    """
    Capa GRU optimizada para JAX/Flax con implementación simplificada.
    
    Parámetros:
    -----------
    hidden_size : int
        Tamaño del estado oculto
    return_sequences : bool
        Si devuelve toda la secuencia o solo último estado
    """
    hidden_size: int
    return_sequences: bool = True
    
    @nn.compact
    def __call__(self, inputs: jnp.ndarray, deterministic: bool = True) -> jnp.ndarray:
        """
        Procesa una secuencia con GRU.
        
        Parámetros:
        -----------
        inputs : jnp.ndarray
            Tensor de entrada [batch_size, seq_len, input_dim]
        deterministic : bool
            Si está en modo de inferencia
            
        Retorna:
        --------
        jnp.ndarray
            Secuencia procesada
        """
        batch_size, seq_len, _ = inputs.shape
        
        # Crear estado inicial
        h = jnp.zeros((batch_size, self.hidden_size))
        
        # Crear la celda GRU
        gru_cell = GRUCell(features=self.hidden_size)
        
        # Procesar secuencia manualmente para evitar problemas con scan
        outputs = []
        for i in range(seq_len):
            h, _ = gru_cell(inputs[:, i, :], h, deterministic=deterministic)
            outputs.append(h)
        
        # Apilar salidas
        outputs = jnp.stack(outputs, axis=1)
        
        # Retornar toda la secuencia o solo el último paso
        if self.return_sequences:
            return outputs
        else:
            return outputs[:, -1, :]


def create_attention_block(x: jnp.ndarray, num_heads: int = 4, key_dim: int = 64,
                          dropout_rate: float = 0.0, deterministic: bool = True) -> jnp.ndarray:
    """
    Crea un bloque de atención multi-cabeza con normalización de capa.
    
    Parámetros:
    -----------
    x : jnp.ndarray
        Entrada al bloque
    num_heads : int
        Número de cabezas de atención
    key_dim : int
        Dimensión de claves para cada cabeza
    dropout_rate : float
        Tasa de dropout
    deterministic : bool
        Si está en modo de inferencia
        
    Retorna:
    --------
    jnp.ndarray
        Salida del bloque de atención
    """
    # Guardar entrada para conexión residual
    skip = x
    
    # Normalización de capa
    x = nn.LayerNorm(epsilon=GRU_CONFIG['epsilon'])(x)
    
    # Aplicar mecanismo de atención
    attention = nn.MultiHeadAttention(
        num_heads=num_heads,
        qkv_features=key_dim,
        dropout_rate=dropout_rate
    )(x, x, deterministic=deterministic)
    
    # Aplicar dropout si no es determinístico
    if dropout_rate > 0.0 and not deterministic:
        attention = nn.Dropout(rate=dropout_rate)(attention, deterministic=deterministic)
    
    # Conexión residual
    x = attention + skip
    
    return x


def create_gru_block(x: jnp.ndarray, units: int, dropout_rate: float = 0.3,
                    deterministic: bool = True) -> jnp.ndarray:
    """
    Crea un bloque GRU con normalización de capa y conexiones residuales.
    
    Parámetros:
    -----------
    x : jnp.ndarray
        Entrada al bloque
    units : int
        Número de unidades GRU
    dropout_rate : float
        Tasa de dropout
    deterministic : bool
        Si está en modo de inferencia
        
    Retorna:
    --------
    jnp.ndarray
        Salida del bloque GRU
    """
    # Guardar entrada para conexión residual
    skip = x
    
    # Aplicar GRU
    x = SimplifiedGRULayer(
        hidden_size=units,
        return_sequences=True
    )(x, deterministic=deterministic)
    
    # Normalización de capa para estabilidad
    x = nn.LayerNorm(epsilon=GRU_CONFIG['epsilon'])(x)
    
    # Aplicar dropout si no es determinístico
    if dropout_rate > 0.0 and not deterministic:
        x = nn.Dropout(rate=dropout_rate)(x, deterministic=deterministic)
    
    # Aplicar conexión residual si las dimensiones coinciden
    if skip.shape[-1] == x.shape[-1]:
        x = x + skip
    
    return x


class GRUModel(nn.Module):
    """
    Modelo GRU con mecanismo de atención para series temporales.
    
    Parámetros:
    -----------
    hidden_units : List[int]
        Lista de unidades ocultas para cada capa GRU
    attention_heads : int
        Número de cabezas de atención
    dropout_rate : float
        Tasa de dropout
    epsilon : float
        Epsilon para normalización de capa
    """
    hidden_units: List[int]
    attention_heads: int
    dropout_rate: float
    epsilon: float
    
    @nn.compact
    def __call__(self, x_cgm: jnp.ndarray, x_other: jnp.ndarray, training: bool = True) -> jnp.ndarray:
        """
        Procesa datos CGM y otras características para predecir niveles de insulina.
        
        Parámetros:
        -----------
        x_cgm : jnp.ndarray
            Datos CGM [batch, seq_len, features]
        x_other : jnp.ndarray
            Otras características [batch, features]
        training : bool
            Si está en modo de entrenamiento
            
        Retorna:
        --------
        jnp.ndarray
            Predicciones de dosis de insulina
        """
        deterministic = not training
        
        # Proyección inicial
        x = nn.Dense(features=self.hidden_units[0])(x_cgm)
        x = nn.LayerNorm(epsilon=self.epsilon)(x)
        
        # Bloques GRU con atención
        for i, units in enumerate(self.hidden_units):
            # Bloque GRU
            x = create_gru_block(
                x=x,
                units=units,
                dropout_rate=self.dropout_rate,
                deterministic=deterministic
            )
            
            # Bloque de atención cada 2 capas GRU
            if (i + 1) % 2 == 0:
                x = create_attention_block(
                    x=x,
                    num_heads=self.attention_heads,
                    key_dim=units // self.attention_heads,
                    dropout_rate=self.dropout_rate,
                    deterministic=deterministic
                )
        
        # Pooling global para reducir la dimensión de secuencia
        x = jnp.mean(x, axis=1)  # [batch, features]
        
        # Combinar con otras características
        combined = jnp.concatenate([x, x_other], axis=-1)
        
        # Capas densas finales con skip connections
        for units in [128, 64]:
            skip = combined
            combined = nn.Dense(features=units)(combined)
            combined = nn.relu(combined)
            combined = nn.LayerNorm(epsilon=self.epsilon)(combined)
            
            if not deterministic:
                combined = nn.Dropout(rate=self.dropout_rate)(combined, deterministic=deterministic)
                
            if skip.shape[-1] == units:
                combined = combined + skip
        
        # Capa de salida
        outputs = nn.Dense(features=1)(combined)
        
        return outputs.squeeze(-1)


def create_gru_model(cgm_shape: Tuple[int, ...], other_features_shape: Tuple[int, ...]) -> nn.Module:
    """
    Crea un modelo GRU para predicción.
    
    Parámetros:
    -----------
    cgm_shape : Tuple[int, ...]
        Forma de los datos CGM
    other_features_shape : Tuple[int, ...]
        Forma de otras características
        
    Retorna:
    --------
    nn.Module
        Modelo GRU inicializado
    """
    return GRUModel(
        hidden_units=GRU_CONFIG['hidden_units'],
        attention_heads=GRU_CONFIG['attention_heads'],
        dropout_rate=GRU_CONFIG['dropout_rate'],
        epsilon=GRU_CONFIG['epsilon']
    )


def create_gru_model_wrapper(cgm_shape: Tuple[int, ...], other_features_shape: Tuple[int, ...]) -> DLModelWrapper:
    """
    Crea un wrapper para el modelo GRU que implementa la API estándar.
    
    Parámetros:
    -----------
    cgm_shape : Tuple[int, ...]
        Forma de los datos CGM
    other_features_shape : Tuple[int, ...]
        Forma de otras características
        
    Retorna:
    --------
    DLModelWrapper
        Wrapper del modelo GRU
    """
    # Función creadora de modelo para pasar al wrapper
    def model_fn():
        return create_gru_model(cgm_shape, other_features_shape)
    
    # Crear wrapper
    wrapper = DLModelWrapper(model_fn, 'jax')
    
    # Configurar early stopping
    es_patience, es_min_delta, es_restore_best = get_early_stopping_config()
    wrapper.add_early_stopping(
        patience=es_patience,
        min_delta=es_min_delta,
        restore_best_weights=es_restore_best
    )
    
    return wrapper


def model_creator() -> Callable[[Tuple[int, ...], Tuple[int, ...]], DLModelWrapper]:
    """
    Retorna un wrapper del modelo GRU compatible con la API del sistema.
    
    Retorna:
    --------
    Callable[[Tuple[int, ...], Tuple[int, ...]], DLModelWrapper]
        Función que crea un wrapper para el modelo GRU con las dimensiones especificadas
    """
    return create_gru_model_wrapper