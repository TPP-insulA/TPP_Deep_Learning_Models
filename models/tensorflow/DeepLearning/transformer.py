import os
import sys
import numpy as np
import tensorflow as tf
from  keras._tf_keras.keras.models import Model
from keras._tf_keras.keras import layers
from typing import Dict, Tuple, List, Any, Optional, Callable

# Asegurarse de que el directorio raíz del proyecto esté en el path
# Modificado para buscar correctamente desde la ubicación del archivo actual
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../.."))
sys.path.append(PROJECT_ROOT)

from config.models_config import TRANSFORMER_CONFIG
# DLModelWrapper no se usa directamente aquí para TF, pero se mantiene por consistencia si se necesita en el futuro
from custom.dl_model_wrapper import DLModelWrapper

# --- Constantes ---
CONST_GELU: str = "gelu"
CONST_RELU: str = "relu"
CONST_SELU: str = "selu"
CONST_SIGMOID: str = "sigmoid"
CONST_TANH: str = "tanh"
CONST_EPSILON: str = "epsilon"
CONST_VALID: str = "VALID"
CONST_SAME: str = "SAME"
CONST_NAME: str = "name"
CONST_CGM_INPUT_NAME: str = "cgm_input"
CONST_OTHER_INPUT_NAME: str = "other_input"
CONST_OUTPUT_DENSE_NAME: str = "output_dense"

class PositionEncoding(layers.Layer):
    """
    Codificación posicional para el Transformer (implementación Keras).

    Atributos:
    ----------
    max_position : int
        Posición máxima a codificar.
    d_model : int
        Dimensión del modelo (profundidad de la codificación).
    """
    def __init__(self, max_position: int, d_model: int, **kwargs) -> None:
        """
        Inicializa la capa de codificación posicional.

        Parámetros:
        -----------
        max_position : int
            Posición máxima a codificar.
        d_model : int
            Dimensión del modelo.
        """
        super().__init__(**kwargs)
        self.max_position = max_position
        self.d_model = d_model
        self.pos_encoding = None

    def build(self, input_shape: Tuple[Optional[int], ...]) -> None:
        """Construye la matriz de codificación posicional."""
        if self.pos_encoding is None:
            angle_rads = self._get_angles(
                np.arange(self.max_position)[:, np.newaxis],
                np.arange(self.d_model)[np.newaxis, :],
                self.d_model
            )

            # Aplicar seno a índices pares en el array; 2i
            sines = np.sin(angle_rads[:, 0::2])
            # Aplicar coseno a índices impares en el array; 2i+1
            cosines = np.cos(angle_rads[:, 1::2])

            pos_encoding_np = np.concatenate([sines, cosines], axis=-1)
            pos_encoding_np = pos_encoding_np[np.newaxis, ...] # Añadir dimensión de batch

            # Usar add_weight para registrar el tensor como no entrenable
            # Esto es más seguro dentro de build que asignar directamente un tf.Tensor
            self.pos_encoding = self.add_weight(
                name="positional_encoding_matrix",
                shape=(1, self.max_position, self.d_model),
                initializer=tf.constant_initializer(pos_encoding_np),
                trainable=False
            )
        super().build(input_shape)

    def _get_angles(self, position: np.ndarray, i: np.ndarray, d_model: int) -> np.ndarray:
        """
        Calcula los ángulos para la codificación posicional.

        Parámetros:
        -----------
        position : np.ndarray
            Posiciones (índices de secuencia).
        i : np.ndarray
            Dimensiones (índices de profundidad).
        d_model : int
            Dimensión del modelo.

        Retorna:
        --------
        np.ndarray
            Ángulos calculados.
        """
        angle_rates = 1 / np.power(10000, (2 * (i // 2)) / np.float32(d_model))
        return position * angle_rates

    def call(self, inputs: tf.Tensor) -> tf.Tensor:
        """
        Aplica la codificación posicional a las entradas.

        Parámetros:
        -----------
        inputs : tf.Tensor
            Tensor de entrada con forma (batch_size, sequence_length, d_model).

        Retorna:
        --------
        tf.Tensor
            Tensor con codificación posicional añadida.
        """
        seq_len = tf.shape(inputs)[1]
        # Asegurar que la codificación no exceda la longitud de la secuencia
        return inputs + self.pos_encoding[:, :seq_len, :]

    def get_config(self) -> Dict[str, Any]:
        """Obtiene la configuración de la capa."""
        config = super().get_config()
        config.update({
            "max_position": self.max_position,
            "d_model": self.d_model,
        })
        return config

def _get_activation_fn(activation_name: str) -> Callable:
    """
    Obtiene la función de activación de Keras correspondiente a un nombre.

    Parámetros:
    -----------
    activation_name : str
        Nombre de la función de activación ('relu', 'gelu', 'selu', etc.).

    Retorna:
    --------
    Callable
        La función de activación de Keras.
    """
    activation_map = {
        CONST_RELU: tf.keras.activations.relu,
        CONST_GELU: tf.keras.activations.gelu,
        CONST_SELU: tf.keras.activations.selu,
        CONST_SIGMOID: tf.keras.activations.sigmoid,
        CONST_TANH: tf.keras.activations.tanh,
    }
    if activation_name not in activation_map:
        print(f"Advertencia: Función de activación '{activation_name}' no reconocida. Usando ReLU por defecto.")
        return tf.keras.activations.relu
    return activation_map[activation_name]

class TransformerBlock(layers.Layer):
    """
    Bloque Transformer con atención multi-cabeza y red feed-forward.

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
    def __init__(self, embed_dim: int, num_heads: int, ff_dim: int, dropout_rate: float,
                 prenorm: bool, use_bias: bool, epsilon: float, activation_fn: Callable, **kwargs) -> None:
        """
        Inicializa el bloque Transformer.

        Parámetros:
        -----------
        embed_dim : int
            Dimensión de embedding.
        num_heads : int
            Número de cabezas de atención.
        ff_dim : int
            Dimensión de la capa oculta FFN.
        dropout_rate : float
            Tasa de dropout.
        prenorm : bool
            Si usar pre-normalización.
        use_bias : bool
            Si usar bias en capas.
        epsilon : float
            Epsilon para LayerNorm.
        activation_fn : Callable
            Función de activación FFN.
        """
        super().__init__(**kwargs)
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.ff_dim = ff_dim
        self.dropout_rate = dropout_rate
        self.prenorm = prenorm
        self.use_bias = use_bias
        self.epsilon = epsilon
        self.activation_fn = activation_fn

        # Capas del bloque
        self.att = layers.MultiHeadAttention(
            num_heads=num_heads,
            key_dim=embed_dim // num_heads, # Dimensión por cabeza
            dropout=dropout_rate,
            use_bias=use_bias,
            name="multi_head_attention"
        )
        self.ffn_dense1 = layers.Dense(ff_dim, activation=activation_fn, use_bias=use_bias, name="ffn_dense1")
        self.ffn_dense2 = layers.Dense(embed_dim, use_bias=use_bias, name="ffn_dense2")

        self.layernorm1 = layers.LayerNormalization(epsilon=epsilon, name="layernorm1")
        self.layernorm2 = layers.LayerNormalization(epsilon=epsilon, name="layernorm2")

        self.dropout1 = layers.Dropout(dropout_rate, name="dropout1")
        self.dropout2 = layers.Dropout(dropout_rate, name="dropout2")

    def build(self, input_shape: Tuple[Optional[int], ...]) -> None:
        """Construye las capas internas del bloque."""
        # Keras llama automáticamente a build en las capas hijas si es necesario,
        # pero podemos hacerlo explícito para mayor claridad.
        # La forma de entrada se propaga a través de las capas internas.
        self.layernorm1.build(input_shape)
        # MHA espera query, value, key con la misma forma que input_shape
        # Keras maneja la construcción interna de MHA
        # self.att.build(query_shape=input_shape, value_shape=input_shape, key_shape=input_shape) # No es necesario usualmente

        # FFN
        self.layernorm2.build(input_shape) # La entrada a la segunda parte también tiene input_shape
        # Dense layers build se llama implícitamente en la primera llamada
        super().build(input_shape) # Llama al build de la clase padre

    def call(self, inputs: tf.Tensor, training: Optional[bool] = None) -> tf.Tensor:
        """
        Aplica el bloque Transformer.

        Parámetros:
        -----------
        inputs : tf.Tensor
            Tensor de entrada.
        training : Optional[bool], opcional
            Indica si el modelo está en modo entrenamiento.

        Retorna:
        --------
        tf.Tensor
            Tensor procesado por el bloque.
        """
        if self.prenorm:
            # Arquitectura Pre-LN
            x = self.layernorm1(inputs)
            attn_output = self.att(query=x, value=x, key=x, training=training)
            attn_output = self.dropout1(attn_output, training=training)
            out1 = inputs + attn_output

            x = self.layernorm2(out1)
            ffn_output = self.ffn_dense1(x)
            ffn_output = self.ffn_dense2(ffn_output)
            ffn_output = self.dropout2(ffn_output, training=training)
            return out1 + ffn_output
        else:
            # Arquitectura Post-LN (original)
            attn_output = self.att(query=inputs, value=inputs, key=inputs, training=training)
            attn_output = self.dropout1(attn_output, training=training)
            out1 = self.layernorm1(inputs + attn_output)

            ffn_output = self.ffn_dense1(out1)
            ffn_output = self.ffn_dense2(ffn_output)
            ffn_output = self.dropout2(ffn_output, training=training)
            return self.layernorm2(out1 + ffn_output)

    def get_config(self) -> Dict[str, Any]:
        """Obtiene la configuración de la capa."""
        config = super().get_config()
        config.update({
            "embed_dim": self.embed_dim,
            "num_heads": self.num_heads,
            "ff_dim": self.ff_dim,
            "dropout_rate": self.dropout_rate,
            "prenorm": self.prenorm,
            "use_bias": self.use_bias,
            "epsilon": self.epsilon,
            # La función de activación no es serializable directamente, guardar su nombre
            "activation_fn_name": self.activation_fn.__name__ if hasattr(self.activation_fn, '__name__') else str(self.activation_fn)
        })
        return config

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> 'TransformerBlock':
        """Crea la capa desde su configuración."""
        # Reconstruir la función de activación desde su nombre
        activation_fn_name = config.pop("activation_fn_name", CONST_RELU)
        config["activation_fn"] = _get_activation_fn(activation_fn_name)
        return cls(**config)


class TransformerModel(Model):
    """
    Modelo Transformer con Keras para datos CGM y otras características.

    Atributos:
    ----------
    config : Dict
        Configuración del modelo (TRANSFORMER_CONFIG).
    cgm_shape : Tuple
        Forma de los datos CGM (time_steps, cgm_features).
    other_features_shape : Tuple
        Forma de otras características (other_features,).
    """
    def __init__(self, config: Dict, cgm_shape: Tuple, other_features_shape: Tuple, **kwargs) -> None:
        """
        Inicializa el modelo Transformer.

        Parámetros:
        -----------
        config : Dict
            Configuración del modelo.
        cgm_shape : Tuple
            Forma de los datos CGM.
        other_features_shape : Tuple
            Forma de otras características.
        """
        super().__init__(**kwargs)
        self.config = config
        self._cgm_input_shape_ref = cgm_shape
        self._other_features_shape_ref = other_features_shape

        # Extraer parámetros de configuración
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
        self.input_projection = layers.Dense(self.embed_dim, use_bias=self.use_bias, name="input_projection")

        # Codificación posicional (si se usa)
        self.pos_encoding = None
        if self.config['use_relative_pos']:
            self.pos_encoding = PositionEncoding(
                max_position=self.config['max_position'],
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

        # Pooling
        self.global_avg_pool = layers.GlobalAveragePooling1D(name="global_avg_pooling")
        self.global_max_pool = layers.GlobalMaxPooling1D(name="global_max_pooling")
        self.concat_pool = layers.Concatenate(axis=-1, name="concat_pooling")

        # Concatenación con otras características
        self.concat_features = layers.Concatenate(axis=-1, name="concat_features")

        # MLP final
        self.dense1 = layers.Dense(128, use_bias=self.use_bias, name="final_dense1")
        self.activation1 = layers.Activation(self.activation_fn, name="final_activation1")
        self.norm1 = layers.LayerNormalization(epsilon=self.epsilon, name="final_norm1")
        self.dropout1 = layers.Dropout(self.dropout_rate, name="final_dropout1")

        self.dense2 = layers.Dense(64, use_bias=self.use_bias, name="final_dense2")
        self.activation2 = layers.Activation(self.activation_fn, name="final_activation2")
        self.norm2 = layers.LayerNormalization(epsilon=self.epsilon, name="final_norm2")
        self.dropout2 = layers.Dropout(self.dropout_rate, name="final_dropout2")

        # Capa de salida
        self.output_dense = layers.Dense(1, name="output_dense") # Para regresión

    def build(self, input_shape: List[Tuple[Optional[int], ...]]) -> None:
        """
        Construye las capas internas del modelo basado en las formas de entrada.

        Parámetros:
        -----------
        input_shape : List[Tuple[Optional[int], ...]]
            Lista de formas de entrada: [cgm_input_shape, other_input_shape].
            Ej: [(None, 24, 3), (None, 6)]
        """
        cgm_input_shape, other_input_shape = input_shape

        # 1. Construir capas que procesan CGM secuencialmente
        # input_projection espera (None, seq_len, cgm_features)
        self.input_projection.build(cgm_input_shape)
        # La forma de salida es (None, seq_len, embed_dim)
        projected_shape = self.input_projection.compute_output_shape(cgm_input_shape)

        if self.pos_encoding is not None:
            self.pos_encoding.build(projected_shape)

        # 2. Construir bloques Transformer
        current_shape = projected_shape
        for block in self.transformer_blocks:
            # build se llama implícitamente en la primera llamada, pero podemos hacerlo explícito
            block.build(current_shape)
            # La forma no cambia a través de los bloques Transformer
            # current_shape = block.compute_output_shape(current_shape)

        # 3. Calcular la forma después del pooling
        # Global pooling reduce la dimensión temporal
        pooled_avg_shape = self.global_avg_pool.compute_output_shape(current_shape) # (None, embed_dim)
        pooled_max_shape = self.global_max_pool.compute_output_shape(current_shape) # (None, embed_dim)
        pooled_output_shape = self.concat_pool.compute_output_shape([pooled_avg_shape, pooled_max_shape]) # (None, 2 * embed_dim)

        # 4. Calcular la forma aplanada de other_input
        # other_input_shape es (None, num_features) o (None, steps, features), etc.
        # Necesitamos la dimensión de características aplanada (producto de dims > 0)
        if len(other_input_shape) < 2:
             raise ValueError(f"Forma inesperada para other_input_shape: {other_input_shape}. Debe tener al menos 2 dimensiones (batch, features).")
        # Asegurarse de que las dimensiones no sean None antes de multiplicar
        dims_to_multiply = [d for d in other_input_shape[1:] if d is not None]
        if not dims_to_multiply:
             # Si todas las dimensiones no-batch son None, no podemos determinar la forma
             # Intentar usar la forma de referencia si está disponible
             if self._other_features_shape_ref:
                 dims_to_multiply_ref = [d for d in self._other_features_shape_ref if d is not None]
                 if dims_to_multiply_ref:
                     other_features_flat_dim = np.prod(dims_to_multiply_ref)
                 else:
                     raise ValueError(f"No se pudo determinar la dimensión aplanada de other_input_shape: {other_input_shape} y la forma de referencia {self._other_features_shape_ref} tampoco es útil.")
             else:
                 raise ValueError(f"No se pudo determinar la dimensión aplanada de other_input_shape: {other_input_shape}")
        else:
            other_features_flat_dim = np.prod(dims_to_multiply)

        other_input_flat_shape = (None, other_features_flat_dim)

        # 5. Calcular la forma combinada
        combined_features_shape = self.concat_features.compute_output_shape([pooled_output_shape, other_input_flat_shape])

        # 6. Construir el MLP final explícitamente con la forma de entrada calculada
        self.dense1.build(combined_features_shape)
        dense1_output_shape = self.dense1.compute_output_shape(combined_features_shape)
        # La activación no tiene build
        self.norm1.build(dense1_output_shape)
        # Dropout no tiene build

        # La entrada a dense2 es la salida de norm1 (misma forma que dense1_output_shape)
        self.dense2.build(dense1_output_shape)
        dense2_output_shape = self.dense2.compute_output_shape(dense1_output_shape)
        # La activación no tiene build
        self.norm2.build(dense2_output_shape)
        # Dropout no tiene build

        # 7. Construir la capa de salida
        # La entrada a output_dense es la salida de norm2 (misma forma que dense2_output_shape)
        self.output_dense.build(dense2_output_shape)

        # Marcar el modelo como construido llamando al build del padre
        super().build(input_shape)
        # self.built = True # Keras >= 2.7 maneja esto automáticamente al llamar super().build()

    def call(self, inputs: List[tf.Tensor], training: Optional[bool] = None) -> tf.Tensor:
        """
        Aplica el modelo Transformer a las entradas.

        Parámetros:
        -----------
        inputs : List[tf.Tensor]
            Lista de tensores de entrada [cgm_input, other_input].
        training : Optional[bool], opcional
            Indica si el modelo está en modo entrenamiento.

        Retorna:
        --------
        tf.Tensor
            Predicciones del modelo (batch_size, 1).
        """
        cgm_input, other_input = inputs

        # 1. Proyección y Codificación Posicional
        x = self.input_projection(cgm_input)
        if self.pos_encoding is not None:
            x = self.pos_encoding(x)

        # 2. Bloques Transformer
        for block in self.transformer_blocks:
            x = block(x, training=training)

        # 3. Pooling
        avg_pool = self.global_avg_pool(x)
        max_pool = self.global_max_pool(x)
        pooled_output = self.concat_pool([avg_pool, max_pool])

        # 4. Combinar con otras características
        # Asegurar que other_input sea 2D (batch, features)
        # Usar tf.shape para obtener el tamaño de batch dinámico
        batch_size = tf.shape(other_input)[0]
        # Aplanar todas las dimensiones excepto la primera (batch)
        other_input_flat = tf.reshape(other_input, [batch_size, -1])
        combined_features = self.concat_features([pooled_output, other_input_flat])

        # 5. MLP final
        x = self.dense1(combined_features)
        x = self.activation1(x)
        x = self.norm1(x)
        x = self.dropout1(x, training=training)

        x = self.dense2(x)
        x = self.activation2(x)
        x = self.norm2(x)
        x = self.dropout2(x, training=training)

        # 6. Capa de salida
        output = self.output_dense(x)

        return output # Salida (batch_size, 1)

    def get_config(self) -> Dict[str, Any]:
        """Obtiene la configuración del modelo."""
        config = super().get_config()
        config.update({
            "config": self.config,
            "cgm_shape": self._cgm_input_shape_ref,
            "other_features_shape": self._other_features_shape_ref,
        })
        return config

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> 'TransformerModel':
        """Crea el modelo desde su configuración."""
        # Keras maneja la deserialización de capas anidadas estándar.
        # Recreamos la instancia usando los argumentos guardados.
        # Extraemos nuestros argumentos específicos del config.
        model_config = config.pop("config")
        cgm_shape = config.pop("cgm_shape")
        other_features_shape = config.pop("other_features_shape")

        # Creamos la instancia con los argumentos extraídos
        # Pasamos el resto de 'config' (que ahora contiene argumentos de keras.Model) a super().__init__
        return cls(config=model_config, cgm_shape=cgm_shape, other_features_shape=other_features_shape, **config)


def create_transformer_model(cgm_shape: Tuple[int, ...], other_features_shape: Tuple[int, ...]) -> Model:
    """
    Crea un modelo Transformer con Keras, listo para ser usado por DLModelWrapperTF.

    Parámetros:
    -----------
    cgm_shape : Tuple[int, ...]
        Forma de los datos CGM (pasos_temporales, características_cgm).
    other_features_shape : Tuple[int, ...]
        Forma de otras características (características_otras,).

    Retorna:
    --------
    tf.keras.Model
        Instancia del modelo Transformer de Keras.
    """
    # Crear la instancia del modelo Keras
    model_instance = TransformerModel(
        config=TRANSFORMER_CONFIG,
        cgm_shape=cgm_shape,
        other_features_shape=other_features_shape,
        name="transformer_model" 
    )

    return model_instance

# --- Función de Creador ---
# Esta función no es estrictamente necesaria para TF si `create_transformer_model`
# ya devuelve el modelo Keras directamente, pero se puede mantener por simetría.

def model_creator() -> Callable[[Tuple[int, ...], Tuple[int, ...]], Model]:
    """
    Retorna la función (`create_transformer_model`) que crea el modelo Transformer de Keras.

    Esta función (`model_creator`) es la que se puede importar y usar en `params.py`.
    No toma argumentos y devuelve la función que sí los toma (`create_transformer_model`).

    Retorna:
    --------
    Callable[[Tuple[int, ...], Tuple[int, ...]], tf.keras.Model]
        Función (`create_transformer_model`) que, dadas las formas de entrada, crea el modelo Keras.
    """
    return DLModelWrapper(create_transformer_model, framework="tensorflow")