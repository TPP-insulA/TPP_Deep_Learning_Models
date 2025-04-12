import os
import sys
import jax
import jax.numpy as jnp
import flax.linen as nn
import numpy as np
from typing import Tuple, Dict, Any, Optional, Callable, Union

PROJECT_ROOT = os.path.abspath(os.getcwd())
sys.path.append(PROJECT_ROOT) 

from config.models_config import ATTENTION_CONFIG, EARLY_STOPPING_POLICY
from custom.dl_model_wrapper import DLModelWrapper
from models.early_stopping import EarlyStopping

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

class AttentionBlock(nn.Module):
    """
    Bloque de atención con soporte para codificación posicional.
    
    Parámetros:
    -----------
    num_heads : int
        Número de cabezas de atención
    key_dim : int
        Dimensión de las claves
    config : Optional[Dict], opcional
        Configuración adicional del bloque (default: None)
    """
    num_heads: int
    key_dim: int
    config: Optional[Dict] = None
    
    @nn.compact
    def __call__(self, x: jnp.ndarray, pos_encoding: jnp.ndarray, training: bool = False) -> jnp.ndarray:
        """
        Aplica el mecanismo de atención a la entrada.
        
        Parámetros:
        -----------
        x : jnp.ndarray
            Tensor de entrada
        pos_encoding : jnp.ndarray
            Codificación posicional
        training : bool, opcional
            Si está en modo entrenamiento (default: False)
            
        Retorna:
        --------
        jnp.ndarray
            Resultado del bloque de atención
        """
        config = self.config or ATTENTION_CONFIG
        
        # Codificación de posición relativa
        if config['use_relative_attention']:
            # Dimensiones
            batch_size, seq_len, feature_dim = x.shape
            
            # Proyecciones para query, key y value
            query_dim = self.key_dim * self.num_heads
            value_dim = config['head_size'] * self.num_heads if config['head_size'] is not None else query_dim
            
            query = nn.Dense(query_dim)(x)
            key = nn.Dense(query_dim)(x)
            value = nn.Dense(value_dim)(x)
            
            # Reshape para atención multicabezal
            query = query.reshape(batch_size, seq_len, self.num_heads, self.key_dim)
            key = key.reshape(batch_size, seq_len, self.num_heads, self.key_dim)
            value = value.reshape(batch_size, seq_len, self.num_heads, 
                                 config['head_size'] if config['head_size'] is not None else self.key_dim)
            
            # Calcular puntuaciones de atención
            scale = jnp.sqrt(self.key_dim).astype(x.dtype)
            attn_weights = jnp.einsum('bqhd,bkhd->bhqk', query, key) / scale
            
            # Adaptación de pos_encoding a dimensiones de attn_weights
            if pos_encoding is not None:
                pos_encoding_shape = pos_encoding.shape
                
                # Adaptación según dimensionalidad de pos_encoding
                if len(pos_encoding_shape) == 3:
                    # Si pos_encoding es 3D (por ejemplo, [seq_len, seq_len, features])
                    pos_encoding_reduced = jnp.mean(pos_encoding[:seq_len, :seq_len, :], axis=-1)
                    # Expandir para broadcasting con attn_weights (batch, heads, seq, seq)
                    pos_encoding_final = jnp.broadcast_to(
                        pos_encoding_reduced[None, None, :, :], 
                        attn_weights.shape
                    )
                else:
                    # Manejo para otras dimensionalidades
                    pos_encoding_final = jnp.zeros_like(attn_weights)
                    
                attn_weights = attn_weights + pos_encoding_final
            
            # Softmax y aplicación de atención
            attn_weights = jax.nn.softmax(attn_weights, axis=-1)
            attn_output = jnp.einsum('bhqk,bkhd->bqhd', attn_weights, value)
            attn_output = attn_output.reshape(batch_size, seq_len, -1)
            
            # Proyección final
            attention_output = nn.Dense(feature_dim)(attn_output)
        else:
            # Usar implementación estándar sin posición relativa
            attention_output = nn.MultiHeadAttention(
                num_heads=self.num_heads,
                key_size=self.key_dim,
                dropout_rate=0.0
            )(x, x)
        
        # Mecanismo de gating
        gate = nn.Dense(attention_output.shape[-1])(x)
        gate = jax.nn.sigmoid(gate)
        attention_output = gate * attention_output
        
        attention_output = nn.Dropout(rate=config['dropout_rate'])(attention_output, deterministic=not training)
        x = nn.LayerNorm(epsilon=1e-6)(x + attention_output)
        
        # Red feed-forward mejorada con GLU
        ffn = nn.Dense(config['ff_dim'])(x)
        ffn_gate = nn.Dense(config['ff_dim'])(x)
        ffn_gate = jax.nn.sigmoid(ffn_gate)
        ffn = ffn * ffn_gate
        ffn = nn.Dense(x.shape[-1])(ffn)
        ffn = nn.Dropout(rate=config['dropout_rate'])(ffn, deterministic=not training)
        
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
    
    def setup(self) -> None:
        """
        Inicializa los componentes del modelo de atención.
        """
        # Dimensiones y configuraciones
        _ = self.cgm_shape[-1]
        num_blocks = self.config['num_blocks']
        
        # Codificación posicional relativa
        self.pos_encoder = RelativePositionEncoding(
            self.config['max_relative_position'],
            self.config['key_dim']
        )
        
        # Bloques de atención - NUEVA ADICIÓN
        self.attention_blocks = [
            AttentionBlock(
                num_heads=self.config['num_heads'],
                key_dim=self.config['key_dim'],
                config=self.config
            )
            for _ in range(num_blocks)
        ]
        
        # Proyección para características adicionales
        if self.other_features_shape[0] > 0:
            self.other_projection = nn.Dense(self.config['embed_dim'])
            
        # Capas finales para la predicción
        final_layers = []
        for units in self.config['dense_units']:
            final_layers.append(nn.Dense(units))
            final_layers.append(nn.LayerNorm(epsilon=1e-6))
            final_layers.append(lambda x: get_activation(x, self.config['activation']))
        final_layers.append(nn.Dense(1))
        self.final_layers = nn.Sequential(final_layers)
        
        # Guardar tasa de dropout
        self.dropout_rate = self.config['dropout_rate']
    
    def __call__(self, x_cgm: jnp.ndarray, x_other: jnp.ndarray, training: bool = False) -> jnp.ndarray:
        """
        Ejecuta el modelo de atención con los datos de entrada.
        
        Parámetros:
        -----------
        x_cgm : jnp.ndarray
            Datos CGM con forma (batch_size, seq_length, features)
        x_other : jnp.ndarray
            Otras características con forma (batch_size, features)
        training : bool, opcional
            Indica si está en modo entrenamiento (default: False)
            
        Retorna:
        --------
        jnp.ndarray
            Predicciones del modelo
        """
        # Obtener dimensiones
        _ = x_cgm.shape[0]
        
        # Inicializar codificación posicional
        pos_encoding = self.pos_encoder(x_cgm)
        
        # Aplicar bloques de atención secuencialmente
        x = x_cgm
        # MODIFICADO: Usar los bloques predefinidos en lugar de crearlos en tiempo de ejecución
        for attention_block in self.attention_blocks:
            x = attention_block(x, pos_encoding, training)
        
        # Reducción para salida (media temporal)
        x = jnp.mean(x, axis=1)
        
        # Procesar características adicionales si las hay
        if x_other is not None and x_other.shape[-1] > 0 and hasattr(self, 'other_projection'):
            x_other_projected = self.other_projection(x_other)
            x = jnp.concatenate([x, x_other_projected], axis=-1)
        
        # Obtener predicción final
        x = self.final_layers(x)
        return x.squeeze(-1)

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
    
    early_stopping = None
    if EARLY_STOPPING_POLICY.get('early_stopping', False):
        early_stopping = EarlyStopping(
            patience=EARLY_STOPPING_POLICY.get('early_stopping_patience', 10),
            min_delta=EARLY_STOPPING_POLICY.get('early_stopping_min_delta', 0.001),
            restore_best_weights=EARLY_STOPPING_POLICY.get('early_stopping_restore_best', True)
        )
    
    # Envolver en DLModelWrapper para API consistente
    return DLModelWrapper(lambda **kwargs: model, early_stopping=early_stopping)

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