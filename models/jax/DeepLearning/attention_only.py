import os
import sys
import jax
import jax.numpy as jnp
import flax.linen as nn
import numpy as np
from typing import Tuple, Dict, Any, Optional, Callable, Union

PROJECT_ROOT = os.path.abspath(os.getcwd())
sys.path.append(PROJECT_ROOT) 

from config.models_config import ATTENTION_CONFIG
from custom.dl_model_wrapper import DLModelWrapper

# Constantes para uso repetido
CONST_ATTENTION_BLOCK = "attention_block"
CONST_ACTOR_KEY = "actor_key"
CONST_CRITIC_KEY = "critic_key"

class RelativePositionEncoding(nn.Module):
    """
    Codificación de posición relativa para mejorar la atención temporal.
    
    Parámetros:
    -----------
    max_position : int
        Posición máxima a codificar
    depth : int
        Profundidad de la codificación
    """
    max_position: int
    depth: int
    
    def setup(self) -> None:
        """
        Inicializa los parámetros de codificación de posición.
        """
        self.rel_embeddings = self.param(
            "rel_embeddings", 
            nn.initializers.glorot_uniform(),
            (2 * self.max_position - 1, self.depth)
        )
    
    def __call__(self, inputs: jnp.ndarray) -> jnp.ndarray:
        """
        Aplica la codificación de posición relativa a las entradas.
        
        Parámetros:
        -----------
        inputs : jnp.ndarray
            Tensor de entrada
            
        Retorna:
        --------
        jnp.ndarray
            Tensor de codificación de posición
        """
        length = inputs.shape[1]
        pos_range = jnp.arange(length)
        pos_indices = pos_range[:, None] - pos_range[None, :] + self.max_position - 1
        pos_emb = self.rel_embeddings[pos_indices]
        return pos_emb

def create_attention_block(x: jnp.ndarray, 
                          num_heads: int, 
                          key_dim: int,
                          ff_dim: int, 
                          dropout_rate: float, 
                          deterministic: bool = False) -> jnp.ndarray:
    """
    Crea un bloque de atención mejorado con posición relativa y gating.

    Parámetros:
    -----------
    x : jnp.ndarray
        Tensor de entrada
    num_heads : int
        Número de cabezas de atención
    key_dim : int
        Dimensión de la clave
    ff_dim : int
        Dimensión de la red feed-forward
    dropout_rate : float
        Tasa de dropout
    deterministic : bool, opcional
        Indica si está en modo evaluación (no aplicar dropout) (default: False)
    
    Retorna:
    --------
    jnp.ndarray
        Tensor procesado
    """
    # Codificación de posición relativa
    if ATTENTION_CONFIG['use_relative_attention']:
        # Obtener la codificación de posición
        pos_encoding = RelativePositionEncoding(
            ATTENTION_CONFIG['max_relative_position'],
            key_dim
        )(x)
        
        # Implementación manual de atención multicabezal con soporte para bias de posición
        batch_size, seq_len, feature_dim = x.shape
        
        # Proyecciones para query, key y value
        query_dim = key_dim * num_heads
        value_dim = ATTENTION_CONFIG['head_size'] * num_heads if ATTENTION_CONFIG['head_size'] is not None else query_dim
        
        query = nn.Dense(query_dim)(x)
        key = nn.Dense(query_dim)(x)
        value = nn.Dense(value_dim)(x)
        
        # Reshape para atención multicabezal
        query = query.reshape(batch_size, seq_len, num_heads, key_dim)
        key = key.reshape(batch_size, seq_len, num_heads, key_dim)
        value = value.reshape(batch_size, seq_len, num_heads, 
                             ATTENTION_CONFIG['head_size'] if ATTENTION_CONFIG['head_size'] is not None else key_dim)
        
        # Calcular puntuaciones de atención
        scale = jnp.sqrt(key_dim).astype(x.dtype)
        attn_weights = jnp.einsum('bqhd,bkhd->bhqk', query, key) / scale
        
        # Añadir bias de posición relativa
        attn_weights = attn_weights + pos_encoding[:, None, :, :]
        
        # Softmax y aplicación de atención
        attn_weights = jax.nn.softmax(attn_weights, axis=-1)
        attn_output = jnp.einsum('bhqk,bkhd->bqhd', attn_weights, value)
        attn_output = attn_output.reshape(batch_size, seq_len, -1)
        
        # Proyección final
        attention_output = nn.Dense(feature_dim)(attn_output)
    else:
        # Usar implementación estándar sin posición relativa
        attention_output = nn.MultiHeadAttention(
            num_heads=num_heads,
            key_size=key_dim,
            dropout_rate=0.0
        )(x, x)
    
    # Mecanismo de gating
    gate = nn.Dense(attention_output.shape[-1])(x)
    gate = jax.nn.sigmoid(gate)
    attention_output = gate * attention_output
    
    attention_output = nn.Dropout(rate=dropout_rate)(attention_output, deterministic=deterministic)
    x = nn.LayerNorm(epsilon=1e-6)(x + attention_output)
    
    # Red feed-forward mejorada con GLU
    ffn = nn.Dense(ff_dim)(x)
    ffn_gate = nn.Dense(ff_dim)(x)
    ffn_gate = jax.nn.sigmoid(ffn_gate)
    ffn = ffn * ffn_gate
    ffn = nn.Dense(x.shape[-1])(ffn)
    ffn = nn.Dropout(rate=dropout_rate)(ffn, deterministic=deterministic)
    
    return nn.LayerNorm(epsilon=1e-6)(x + ffn)

class AttentionModel(nn.Module):
    """
    Modelo basado únicamente en mecanismos de atención.
    
    Parámetros:
    -----------
    config : Dict
        Diccionario con la configuración del modelo
    cgm_shape : Tuple
        Forma de los datos CGM
    other_features_shape : Tuple
        Forma de otras características
    """
    config: Dict
    cgm_shape: Tuple
    other_features_shape: Tuple
    
    @nn.compact
    def __call__(self, cgm_input: jnp.ndarray, other_input: jnp.ndarray, training: bool = True) -> jnp.ndarray:
        """
        Ejecuta el modelo de atención sobre las entradas.
        
        Parámetros:
        -----------
        cgm_input : jnp.ndarray
            Datos de entrada CGM
        other_input : jnp.ndarray
            Otras características de entrada
        training : bool, opcional
            Indica si está en modo entrenamiento (default: True)
            
        Retorna:
        --------
        jnp.ndarray
            Predicciones del modelo
        """
        deterministic = not training
        
        # Proyección inicial
        x = nn.Dense(self.config['key_dim'] * self.config['num_heads'])(cgm_input)
        
        # Stochastic depth (dropout de capas)
        survive_rates = jnp.linspace(1.0, 0.5, self.config['num_layers'])
        
        # Apilar bloques de atención con stochastic depth
        for i in range(self.config['num_layers']):
            should_apply = jax.random.uniform(self.make_rng('dropout'), shape=()) < survive_rates[i]
            if deterministic:
                # En evaluación, escalar la salida en vez de omitir capas
                block_output = create_attention_block(
                    x,
                    self.config['num_heads'],
                    self.config['key_dim'],
                    self.config['ff_dim'],
                    self.config['dropout_rate'],
                    deterministic=deterministic
                )
                x = x + survive_rates[i] * (block_output - x)
            elif should_apply:
                x = create_attention_block(
                    x,
                    self.config['num_heads'],
                    self.config['key_dim'],
                    self.config['ff_dim'],
                    self.config['dropout_rate'],
                    deterministic=deterministic
                )
        
        # Contexto global
        attention_pooled = jnp.mean(x, axis=1)
        max_pooled = jnp.max(x, axis=1)
        x = jnp.concatenate([attention_pooled, max_pooled], axis=-1)
        
        # Combinar con otras características
        x = jnp.concatenate([x, other_input], axis=-1)
        
        # MLP final con conexión residual
        skip = x
        x = nn.Dense(128)(x)
        x = get_activation(x, self.config['activation'])
        x = nn.LayerNorm(epsilon=1e-6)(x)
        x = nn.Dropout(rate=self.config['dropout_rate'])(x, deterministic=deterministic)
        x = nn.Dense(128)(x)
        x = get_activation(x, self.config['activation'])
        
        if skip.shape[-1] == 128:
            x = x + skip
        
        output = nn.Dense(1)(x)
        
        return output

def get_activation(x: jnp.ndarray, activation_name: str) -> jnp.ndarray:
    """
    Aplica la función de activación según su nombre.
    
    Parámetros:
    -----------
    x : jnp.ndarray
        Tensor al que aplicar la activación
    activation_name : str
        Nombre de la función de activación
        
    Retorna:
    --------
    jnp.ndarray
        Tensor con la activación aplicada
    """
    if activation_name == 'relu':
        return nn.relu(x)
    elif activation_name == 'gelu':
        return nn.gelu(x)
    elif activation_name == 'swish':
        return nn.swish(x)
    elif activation_name == 'silu':
        return nn.silu(x)
    else:
        return nn.relu(x)  # Valor por defecto

def create_attention_model(cgm_shape: Tuple[int, ...], other_features_shape: Tuple[int, ...]) -> DLModelWrapper:
    """
    Crea un modelo basado en mecanismos de atención envuelto en un DLModelWrapper.
    
    Parámetros:
    -----------
    cgm_shape : Tuple[int, ...]
        Forma de los datos CGM (pasos_temporales, características)
    other_features_shape : Tuple[int, ...]
        Forma de otras características (características)
        
    Retorna:
    --------
    DLModelWrapper
        Modelo envuelto en DLModelWrapper para compatibilidad con el sistema
    """
    model = AttentionModel(
        config=ATTENTION_CONFIG,
        cgm_shape=cgm_shape,
        other_features_shape=other_features_shape
    )
    
    # Envolver en DLModelWrapper para API consistente
    return DLModelWrapper(lambda **kwargs: model)

# Función creadora de modelo que será usada por el sistema
def model_creator() -> Callable[[Tuple[int, ...], Tuple[int, ...]], DLModelWrapper]:
    """
    Retorna una función para crear un modelo de atención compatible con la API del sistema.
    
    Retorna:
    --------
    Callable[[Tuple[int, ...], Tuple[int, ...]], DLModelWrapper]
        Función para crear el modelo con las formas de entrada especificadas
    """
    return create_attention_model