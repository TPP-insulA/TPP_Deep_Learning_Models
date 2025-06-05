import os
import sys
import jax
import jax.numpy as jnp
import flax.linen as nn
import numpy as np
from typing import Dict, Tuple, List, Any, Optional, Callable

PROJECT_ROOT = os.path.abspath(os.getcwd())
sys.path.append(PROJECT_ROOT)

from config.models_config_old import TRANSFORMER_CONFIG
from custom.DeepLearning.dl_model_wrapper import DLModelWrapper

# --- Constantes ---
CONST_GELU: str = "gelu"
CONST_RELU: str = "relu"
CONST_SELU: str = "selu"
CONST_SIGMOID: str = "sigmoid"
CONST_TANH: str = "tanh"
CONST_EPSILON: str = "epsilon"
CONST_VALID: str = "VALID"
CONST_SAME: str = "SAME"
CONST_DROPOUT: str = "dropout" # Clave para RNG de dropout

class PositionEncoding(nn.Module):
    """
    Codificación posicional para el Transformer (implementación Flax).

    Atributos:
    ----------
    max_position : int
        Posición máxima a codificar.
    d_model : int
        Dimensión del modelo (profundidad de la codificación).
    """
    max_position: int
    d_model: int

    def setup(self) -> None:
        """
        Inicializa la matriz de codificación posicional.
        """
        pos_encoding = np.zeros((self.max_position, self.d_model))
        position = np.arange(0, self.max_position)[:, np.newaxis]
        div_term = np.exp(np.arange(0, self.d_model, 2) * -(np.log(10000.0) / self.d_model))

        pos_encoding[:, 0::2] = np.sin(position * div_term)
        pos_encoding[:, 1::2] = np.cos(position * div_term)

        # Convertir a jnp.array y añadir dimensión de batch (aunque no se usa directamente)
        self.pos_encoding_matrix = jnp.array(pos_encoding)[jnp.newaxis, :, :]

    def __call__(self, inputs: jnp.ndarray) -> jnp.ndarray:
        """
        Aplica la codificación posicional a las entradas.

        Parámetros:
        -----------
        inputs : jnp.ndarray
            Tensor de entrada con forma (batch_size, sequence_length, d_model).

        Retorna:
        --------
        jnp.ndarray
            Tensor con codificación posicional añadida.
        """
        seq_len = inputs.shape[1]
        # Añadir la codificación posicional (broadcasting sobre batch)
        # Seleccionar solo la parte necesaria de la matriz precalculada
        return inputs + self.pos_encoding_matrix[:, :seq_len, :]

def _get_activation_fn(activation_name: str) -> Callable:
    """
    Obtiene la función de activación de Flax correspondiente a un nombre.

    Parámetros:
    -----------
    activation_name : str
        Nombre de la función de activación ('relu', 'gelu', 'selu', etc.).

    Retorna:
    --------
    Callable
        La función de activación de Flax.
    """
    if activation_name == CONST_RELU:
        return nn.relu
    elif activation_name == CONST_GELU:
        return nn.gelu
    elif activation_name == CONST_SELU:
        return nn.selu
    elif activation_name == CONST_SIGMOID:
        return jax.nn.sigmoid
    elif activation_name == CONST_TANH:
        return jnp.tanh
    else:
        print(f"Advertencia: Función de activación '{activation_name}' no reconocida. Usando ReLU por defecto.")
        return nn.relu

class TransformerBlock(nn.Module):
    """
    Bloque Transformer con atención multi-cabeza y red feed-forward (Flax).

    Atributos:
    ----------
    embed_dim : int
        Dimensión de embedding (igual a d_model).
    num_heads : int
        Número de cabezas de atención.
    ff_dim : int
        Dimensión de la capa oculta en la red feed-forward.
    dropout_rate : float
        Tasa de dropout.
    prenorm : bool
        Indica si se usa pre-normalización (LayerNorm antes de subcapas).
    use_bias : bool
        Indica si las capas Dense y MHA usan bias.
    epsilon : float
        Epsilon para LayerNormalization.
    activation_fn : Callable
        Función de activación para la red feed-forward.
    """
    embed_dim: int
    num_heads: int
    ff_dim: int
    dropout_rate: float
    prenorm: bool
    use_bias: bool
    epsilon: float
    activation_fn: Callable

    @nn.compact
    def __call__(self, inputs: jnp.ndarray, training: bool) -> jnp.ndarray:
        """
        Aplica el bloque Transformer.

        Parámetros:
        -----------
        inputs : jnp.ndarray
            Tensor de entrada (batch, seq_len, embed_dim).
        training : bool
            Indica si el modelo está en modo entrenamiento.

        Retorna:
        --------
        jnp.ndarray
            Tensor procesado por el bloque.
        """
        deterministic = not training

        if self.prenorm:
            # --- Arquitectura Pre-LN ---
            # 1. Multi-Head Attention
            x_norm = nn.LayerNorm(epsilon=self.epsilon, name="prenorm_att")(inputs)
            attn_output = nn.MultiHeadAttention(
                num_heads=self.num_heads,
                qkv_features=self.embed_dim, # Dimensión total para Q, K, V
                out_features=self.embed_dim, # Dimensión de salida
                dropout_rate=self.dropout_rate,
                deterministic=deterministic,
                use_bias=self.use_bias,
                name="multi_head_attention"
            )(x_norm, x_norm) # Self-attention
            attn_output = nn.Dropout(rate=self.dropout_rate, deterministic=deterministic, name="dropout_att")(attn_output)
            # Conexión residual
            x = inputs + attn_output

            # 2. Feed-Forward Network
            x_norm = nn.LayerNorm(epsilon=self.epsilon, name="prenorm_ffn")(x)
            ffn_output = nn.Dense(features=self.ff_dim, use_bias=self.use_bias, name="ffn_dense1")(x_norm)
            ffn_output = self.activation_fn(ffn_output)
            ffn_output = nn.Dropout(rate=self.dropout_rate, deterministic=deterministic, name="dropout_ffn1")(ffn_output)
            ffn_output = nn.Dense(features=self.embed_dim, use_bias=self.use_bias, name="ffn_dense2")(ffn_output)
            ffn_output = nn.Dropout(rate=self.dropout_rate, deterministic=deterministic, name="dropout_ffn2")(ffn_output)
            # Conexión residual
            output = x + ffn_output
        else:
            # --- Arquitectura Post-LN (Original) ---
            # 1. Multi-Head Attention
            attn_output = nn.MultiHeadAttention(
                num_heads=self.num_heads,
                qkv_features=self.embed_dim,
                out_features=self.embed_dim,
                dropout_rate=self.dropout_rate,
                deterministic=deterministic,
                use_bias=self.use_bias,
                name="multi_head_attention"
            )(inputs, inputs) # Self-attention
            attn_output = nn.Dropout(rate=self.dropout_rate, deterministic=deterministic, name="dropout_att")(attn_output)
            # Conexión residual y normalización
            x = nn.LayerNorm(epsilon=self.epsilon, name="postnorm_att")(inputs + attn_output)

            # 2. Feed-Forward Network
            ffn_output = nn.Dense(features=self.ff_dim, use_bias=self.use_bias, name="ffn_dense1")(x)
            ffn_output = self.activation_fn(ffn_output)
            ffn_output = nn.Dropout(rate=self.dropout_rate, deterministic=deterministic, name="dropout_ffn1")(ffn_output)
            ffn_output = nn.Dense(features=self.embed_dim, use_bias=self.use_bias, name="ffn_dense2")(ffn_output)
            ffn_output = nn.Dropout(rate=self.dropout_rate, deterministic=deterministic, name="dropout_ffn2")(ffn_output)
            # Conexión residual y normalización
            output = nn.LayerNorm(epsilon=self.epsilon, name="postnorm_ffn")(x + ffn_output)

        return output

class TransformerModel(nn.Module):
    """
    Modelo Transformer con Flax para datos CGM y otras características.

    Atributos:
    ----------
    config : Dict
        Configuración del modelo (TRANSFORMER_CONFIG).
    cgm_shape : Tuple
        Forma de los datos CGM (time_steps, cgm_features).
    other_features_shape : Tuple
        Forma de otras características (other_features,).
    """
    config: Dict
    cgm_shape: Tuple
    other_features_shape: Tuple

    def setup(self) -> None:
        """Inicializa los componentes del modelo."""
        self.embed_dim = self.config['key_dim'] * self.config['num_heads']
        self.num_heads = self.config['num_heads']
        self.ff_dim = self.config['ff_dim']
        self.num_layers = self.config['num_layers']
        self.dropout_rate = self.config['dropout_rate']
        self.prenorm = self.config['prenorm']
        self.use_bias = self.config['use_bias']
        self.epsilon = self.config[CONST_EPSILON]
        self.activation_name = self.config['activation']
        self.activation_fn = _get_activation_fn(self.activation_name)

        # Capa de proyección inicial para CGM
        self.input_projection = nn.Dense(self.embed_dim, use_bias=self.use_bias, name="input_projection")

        # Codificación posicional (si se usa)
        self.pos_encoding_layer = None
        if self.config['use_relative_pos']: # Asumiendo que 'use_relative_pos' controla si se usa PE
            self.pos_encoding_layer = PositionEncoding(
                max_position=self.config['max_position'], # Necesita max_position en config
                d_model=self.embed_dim,
                name="positional_encoding"
            )

        # Bloques Transformer
        self.transformer_blocks = [
            TransformerBlock(
                embed_dim=self.embed_dim,
                num_heads=self.num_heads,
                ff_dim=self.ff_dim,
                dropout_rate=self.dropout_rate,
                prenorm=self.prenorm,
                use_bias=self.use_bias,
                epsilon=self.epsilon,
                activation_fn=self.activation_fn,
                name=f"transformer_block_{i}"
            ) for i in range(self.num_layers)
        ]

        # MLP final
        self.dense1 = nn.Dense(128, use_bias=self.use_bias, name="final_dense1")
        self.norm1 = nn.LayerNorm(epsilon=self.epsilon, name="final_norm1")
        self.dropout1 = nn.Dropout(rate=self.dropout_rate, name="final_dropout1")

        self.dense2 = nn.Dense(64, use_bias=self.use_bias, name="final_dense2")
        self.norm2 = nn.LayerNorm(epsilon=self.epsilon, name="final_norm2")
        self.dropout2 = nn.Dropout(rate=self.dropout_rate, name="final_dropout2")

        # Capa de salida
        self.output_dense = nn.Dense(1, name="output_dense") # Para regresión

    @nn.compact
    def __call__(self, cgm_input: jnp.ndarray, other_input: jnp.ndarray, training: bool) -> jnp.ndarray:
        """
        Aplica el modelo Transformer a las entradas.

        Parámetros:
        -----------
        cgm_input : jnp.ndarray
            Datos CGM de entrada (batch, time_steps, cgm_features).
        other_input : jnp.ndarray
            Otras características de entrada (batch, other_features).
        training : bool
            Indica si el modelo está en modo entrenamiento.

        Retorna:
        --------
        jnp.ndarray
            Predicciones del modelo (batch,).
        """
        deterministic = not training

        # 1. Proyección y Codificación Posicional
        x = self.input_projection(cgm_input) # (batch, time_steps, embed_dim)
        if self.pos_encoding_layer is not None:
            x = self.pos_encoding_layer(x)
        # Dropout después de la proyección/codificación
        x = nn.Dropout(rate=self.dropout_rate, deterministic=deterministic, name="input_dropout")(x)

        # 2. Bloques Transformer
        for block in self.transformer_blocks:
            x = block(x, training=training) # (batch, time_steps, embed_dim)

        # 3. Pooling (Global Average Pooling)
        # Usar jnp.mean sobre la dimensión temporal (axis=1)
        pooled_output = jnp.mean(x, axis=1) # (batch, embed_dim)

        # 4. Combinar con otras características
        # Asegurar que other_input sea 2D (batch, features)
        if other_input.ndim > 2:
             other_input_flat = jnp.reshape(other_input, (other_input.shape[0], -1))
        else:
             other_input_flat = other_input
        combined_features = jnp.concatenate([pooled_output, other_input_flat], axis=-1)

        # 5. MLP final
        x = self.dense1(combined_features)
        x = self.activation_fn(x)
        x = self.norm1(x)
        x = self.dropout1(x, deterministic=deterministic)

        x = self.dense2(x)
        x = self.activation_fn(x)
        x = self.norm2(x)
        x = self.dropout2(x, deterministic=deterministic)

        # 6. Capa de salida
        output = self.output_dense(x) # (batch, 1)

        # Asegurar que la salida sea (batch,) para regresión MSE
        return output.squeeze(-1)


# --- Funciones de creación y envoltura ---

def _create_raw_transformer_model(cgm_shape: Tuple[int, ...], other_features_shape: Tuple[int, ...]) -> TransformerModel:
    """
    Crea una instancia cruda del modelo Transformer (nn.Module).

    Parámetros:
    -----------
    cgm_shape : Tuple[int, ...]
        Forma de los datos CGM (time_steps, cgm_features).
    other_features_shape : Tuple[int, ...]
        Forma de otras características (other_features,).

    Retorna:
    --------
    TransformerModel
        Instancia del modelo Transformer de Flax.
    """
    # Asegurarse de que 'max_position' esté en la config si se usa PE
    if TRANSFORMER_CONFIG.get('use_relative_pos', False) and 'max_position' not in TRANSFORMER_CONFIG:
         raise ValueError("TRANSFORMER_CONFIG debe incluir 'max_position' si 'use_relative_pos' es True.")

    return TransformerModel(
        config=TRANSFORMER_CONFIG,
        cgm_shape=cgm_shape,
        other_features_shape=other_features_shape
    )

def create_transformer_model(cgm_shape: Tuple[int, ...], other_features_shape: Tuple[int, ...]) -> DLModelWrapper:
    """
    Crea un modelo Transformer envuelto en DLModelWrapper para compatibilidad con el sistema de entrenamiento JAX.

    Esta función es la que se pasa al sistema de entrenamiento (e.g., train_multiple_models).
    Retorna un DLModelWrapper que internamente sabe cómo crear la instancia cruda del TransformerModel.

    Parámetros:
    -----------
    cgm_shape : Tuple[int, ...]
        Forma de los datos CGM (time_steps, cgm_features).
    other_features_shape : Tuple[int, ...]
        Forma de otras características (other_features,).

    Retorna:
    --------
    DLModelWrapper
        Modelo Transformer envuelto y listo para ser usado por el sistema de entrenamiento JAX.
    """
    # La función que DLModelWrapper necesita es una que *cree* el modelo crudo.
    raw_model_creator = lambda: _create_raw_transformer_model(cgm_shape, other_features_shape)

    # Envolver el *creador* del modelo crudo en DLModelWrapper (especificando backend 'jax')
    return DLModelWrapper(raw_model_creator, framework='jax')

def model_creator() -> Callable[[Tuple[int, ...], Tuple[int, ...]], DLModelWrapper]:
    """
    Retorna la función (`create_transformer_model`) que crea el modelo Transformer envuelto.

    Esta función (`model_creator`) es la que se importa y se usa en `params.py`.
    No toma argumentos y devuelve la función que sí los toma (`create_transformer_model`).

    Retorna:
    --------
    Callable[[Tuple[int, ...], Tuple[int, ...]], DLModelWrapper]
        Función (`create_transformer_model`) que, dadas las formas de entrada, crea el modelo Transformer envuelto.
    """
    return create_transformer_model
