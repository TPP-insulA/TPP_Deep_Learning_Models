import os, sys
import jax
import jax.numpy as jnp
import flax.linen as nn
import numpy as np
from typing import Tuple, List, Dict, Any, Optional, Callable, Sequence

PROJECT_ROOT = os.path.abspath(os.getcwd())
sys.path.append(PROJECT_ROOT)

from config.models_config import TCN_CONFIG
from custom.dl_model_wrapper import DLModelWrapper

# --- Constantes ---
CONST_RELU: str = "relu"
CONST_GELU: str = "gelu"
CONST_SELU: str = "selu"
CONST_SIGMOID: str = "sigmoid"
CONST_TANH: str = "tanh"
CONST_VALID: str = "VALID"
CONST_SAME: str = "SAME"
CONST_DROPOUT: str = "dropout"

class WeightNormalization(nn.Module):
    """
    Normalización de pesos para capas convolucionales.

    Atributos:
    ----------
    filters : int
        Número de filtros.
    kernel_size : int
        Tamaño del kernel.
    dilation_rate : int
        Tasa de dilatación.
    padding : str
        Tipo de padding ('VALID' o 'SAME').
    """
    filters: int
    kernel_size: int
    dilation_rate: int = 1
    padding: str = CONST_VALID

    @nn.compact
    def __call__(self, inputs: jnp.ndarray) -> jnp.ndarray:
        """
        Aplica normalización de pesos a una capa convolucional.

        Parámetros:
        -----------
        inputs : jnp.ndarray
            Tensor de entrada con forma (batch, time_steps, features).

        Retorna:
        --------
        jnp.ndarray
            Tensor procesado.
        """
        # Define la forma del kernel: (kernel_size, input_features, output_filters)
        kernel_shape = (self.kernel_size, inputs.shape[-1], self.filters)

        # Crea los parámetros para el kernel (dirección)
        kernel = self.param('kernel', nn.initializers.lecun_normal(), kernel_shape)

        # Crea el parámetro para la magnitud (g)
        g = self.param('g', nn.initializers.ones, (self.filters,))

        # Calcula la norma a lo largo de las dimensiones del kernel (0 y 1)
        norm = jnp.sqrt(jnp.sum(jnp.square(kernel), axis=(0, 1)))

        # Normaliza los pesos: kernel_norm = kernel * (g / ||kernel||)
        # Añadir reshape para broadcasting correcto: g y norm deben tener forma (1, 1, filters)
        normalized_kernel = kernel * (g / (norm + 1e-5))[jnp.newaxis, jnp.newaxis, :]

        # Aplica la convolución 1D usando lax.conv_general_dilated
        # JAX espera (N, H, W, C) o (N, C, H, W). Para 1D, usamos (N, W, C)
        # Dimension numbers: (batch, spatial, features) -> (batch, spatial_out, features_out)
        # Kernel: (spatial, features_in, features_out)
        output = jax.lax.conv_general_dilated(
            lhs=inputs,                     # (batch, time_steps, features_in) -> NWC
            rhs=normalized_kernel,          # (kernel_size, features_in, features_out) -> WIO
            window_strides=(1,),            # Stride de 1
            padding=self.padding,           # 'VALID' o 'SAME'
            lhs_dilation=(1,),              # Sin dilatación en la entrada
            rhs_dilation=(self.dilation_rate,), # Dilatación en el kernel
            dimension_numbers=('NWC', 'WIO', 'NWC') # lhs, rhs, out specs
        )
        return output

def causal_padding(inputs: jnp.ndarray, padding_size: int) -> jnp.ndarray:
    """
    Aplica padding causal (solo al principio) a un tensor 1D.

    Parámetros:
    -----------
    inputs : jnp.ndarray
        Tensor de entrada con forma (batch, time_steps, features).
    padding_size : int
        Tamaño del padding a añadir al inicio de la dimensión temporal.

    Retorna:
    --------
    jnp.ndarray
        Tensor con padding causal aplicado.
    """
    # Padear solo al inicio de la dimensión temporal (axis=1)
    return jnp.pad(inputs, [(0, 0), (padding_size, 0), (0, 0)])

class TCNResidualBlock(nn.Module):
    """
    Bloque residual TCN con convoluciones dilatadas, normalización y dropout.

    Atributos:
    ----------
    filters : int
        Número de filtros en las capas convolucionales.
    kernel_size : int
        Tamaño del kernel convolucional.
    dilation_rate : int
        Tasa de dilatación para la convolución.
    dropout_rate : float
        Tasa de dropout a aplicar.
    use_weight_norm : bool
        Indica si se debe usar normalización de pesos.
    use_layer_norm : bool
        Indica si se debe usar Layer Normalization (si False, usa BatchNorm).
    use_spatial_dropout : bool
        Indica si se debe usar Spatial Dropout.
    activation_fn : Callable
        Función de activación a usar.
    epsilon : float
        Epsilon para LayerNorm/BatchNorm.
    """
    filters: int
    kernel_size: int
    dilation_rate: int
    dropout_rate: float
    use_weight_norm: bool
    use_layer_norm: bool
    use_spatial_dropout: bool
    activation_fn: Callable
    epsilon: float

    @nn.compact
    def __call__(self, inputs: jnp.ndarray, training: bool) -> jnp.ndarray:
        """
        Aplica el bloque residual TCN.

        Parámetros:
        -----------
        inputs : jnp.ndarray
            Tensor de entrada (batch, time_steps, features).
        training : bool
            Indica si el modelo está en modo entrenamiento.

        Retorna:
        --------
        jnp.ndarray
            Salida del bloque TCN.
        """
        deterministic = not training
        residual = inputs
        x = inputs

        # Padding causal
        padding_size = (self.kernel_size - 1) * self.dilation_rate
        x = causal_padding(x, padding_size)

        # Primera convolución + activación + normalización + dropout
        if self.use_weight_norm:
            x = WeightNormalization(
                filters=self.filters,
                kernel_size=self.kernel_size,
                dilation_rate=self.dilation_rate,
                padding=CONST_VALID, # Padding ya aplicado manualmente
                name="wn_conv1"
            )(x)
        else:
            x = nn.Conv(
                features=self.filters,
                kernel_size=(self.kernel_size,),
                kernel_dilation=(self.dilation_rate,),
                padding=CONST_VALID, # Padding ya aplicado manualmente
                name="conv1"
            )(x)

        x = self.activation_fn(x)

        if self.use_layer_norm:
            x = nn.LayerNorm(epsilon=self.epsilon, name="norm1")(x)
        else:
            x = nn.BatchNorm(
                use_running_average=deterministic,
                momentum=0.9,
                epsilon=self.epsilon,
                name="bn1"
            )(x)

        x = nn.Dropout(rate=self.dropout_rate, deterministic=deterministic, name="dropout1")(x) # Dropout estándar

        # Segunda convolución + activación + normalización + dropout
        # Padding causal para la segunda convolución
        x = causal_padding(x, padding_size)

        if self.use_weight_norm:
            x = WeightNormalization(
                filters=self.filters,
                kernel_size=self.kernel_size,
                dilation_rate=self.dilation_rate,
                padding=CONST_VALID,
                name="wn_conv2"
            )(x)
        else:
            x = nn.Conv(
                features=self.filters,
                kernel_size=(self.kernel_size,),
                kernel_dilation=(self.dilation_rate,),
                padding=CONST_VALID,
                name="conv2"
            )(x)

        x = self.activation_fn(x)

        if self.use_layer_norm:
            x = nn.LayerNorm(epsilon=self.epsilon, name="norm2")(x)
        else:
            x = nn.BatchNorm(
                use_running_average=deterministic,
                momentum=0.9,
                epsilon=self.epsilon,
                name="bn2"
            )(x)

        x = nn.Dropout(rate=self.dropout_rate, deterministic=deterministic, name="dropout2")(x) # Dropout estándar

        # Conexión residual
        # Si las dimensiones de las características no coinciden, usar una convolución 1x1
        if residual.shape[-1] != self.filters:
            residual = nn.Conv(
                features=self.filters,
                kernel_size=(1,),
                padding=CONST_SAME, # Para mantener la dimensión temporal
                name="residual_conv"
            )(residual)

        # Asegurar que las longitudes temporales coincidan para la suma residual
        # x podría ser más corto debido al padding 'VALID'
        target_len = x.shape[1]
        residual_cropped = residual[:, -target_len:, :] # Tomar los últimos 'target_len' pasos

        return self.activation_fn(x + residual_cropped) # Activación después de la suma

class TCNModel(nn.Module):
    """
    Modelo TCN completo usando JAX/Flax.

    Atributos:
    ----------
    config : Dict
        Configuración del modelo TCN_CONFIG.
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
        self.num_filters = self.config['filters']
        self.kernel_size = self.config['kernel_size']
        self.dilations = self.config['dilations']
        self.dropout_rates = self.config['dropout_rate'] # Ahora es una lista
        self.use_weight_norm = self.config['use_weight_norm']
        self.use_layer_norm = self.config['use_layer_norm']
        self.use_spatial_dropout = self.config['use_spatial_dropout'] # No usado directamente en el bloque ahora
        self.activation_name = self.config['activation']
        self.epsilon = self.config['epsilon']

        # Seleccionar función de activación
        self.activation_fn = _get_activation_fn(self.activation_name)

        # Crear bloques residuales TCN usando list comprehension
        num_levels = len(self.dilations)
        self.tcn_blocks = [
            TCNResidualBlock(
                # Asignar filtros basados en la configuración (puede ser constante o variar por nivel)
                filters=self.num_filters[i] if isinstance(self.num_filters, list) and i < len(self.num_filters) else self.num_filters[0],
                kernel_size=self.kernel_size,
                dilation_rate=self.dilations[i],
                dropout_rate=self.dropout_rates[0], # Usar el primer dropout rate para TCN
                use_weight_norm=self.use_weight_norm,
                use_layer_norm=self.use_layer_norm,
                use_spatial_dropout=self.use_spatial_dropout,
                activation_fn=self.activation_fn,
                epsilon=self.epsilon,
                name=f"tcn_block_{i}"
            ) for i in range(num_levels)
        ]

        # Capas densas finales
        self.dense1 = nn.Dense(128, name="dense1")
        self.norm1 = nn.LayerNorm(epsilon=self.epsilon, name="final_norm1")
        self.dropout1 = nn.Dropout(rate=self.dropout_rates[1], name="final_dropout1") # Usar segundo dropout rate
        self.dense2 = nn.Dense(64, name="dense2")
        self.norm2 = nn.LayerNorm(epsilon=self.epsilon, name="final_norm2")
        self.dropout2 = nn.Dropout(rate=self.dropout_rates[1], name="final_dropout2") # Usar segundo dropout rate
        self.output_dense = nn.Dense(1, name="output_dense")

    @nn.compact
    def __call__(self, cgm_input: jnp.ndarray, other_input: jnp.ndarray, training: bool) -> jnp.ndarray:
        """
        Aplica el modelo TCN a las entradas.

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
            Predicciones del modelo (batch, 1).
        """
        x = cgm_input

        # Aplicar bloques TCN
        for block in self.tcn_blocks:
            x = block(x, training=training)

        # Tomar la salida del último paso temporal
        x = x[:, -1, :] # (batch, features)

        # Combinar con otras características
        # Asegurar que other_input sea 2D (batch, features)
        if other_input.ndim > 2:
             other_input_flat = jnp.reshape(other_input, (other_input.shape[0], -1))
        else:
             other_input_flat = other_input
        combined_features = jnp.concatenate([x, other_input_flat], axis=-1)

        # MLP final
        x = self.dense1(combined_features)
        x = self.activation_fn(x)
        x = self.norm1(x)
        x = self.dropout1(x, deterministic=not training)

        x = self.dense2(x)
        x = self.activation_fn(x)
        x = self.norm2(x)
        x = self.dropout2(x, deterministic=not training)

        # Capa de salida
        output = self.output_dense(x)

        # Asegurar que la salida sea (batch,) para regresión MSE
        return output.squeeze(-1)


def _get_activation_fn(activation_name: str) -> Callable:
    """
    Obtiene la función de activación correspondiente a un nombre.

    Parámetros:
    -----------
    activation_name : str
        Nombre de la función de activación ('relu', 'gelu', 'selu', etc.).

    Retorna:
    --------
    Callable
        La función de activación de JAX/Flax.
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

# --- Funciones de creación y envoltura ---

def _create_raw_tcn_model(cgm_shape: Tuple[int, ...], other_features_shape: Tuple[int, ...]) -> TCNModel:
    """
    Crea una instancia cruda del modelo TCN (nn.Module).

    Parámetros:
    -----------
    cgm_shape : Tuple[int, ...]
        Forma de los datos CGM (time_steps, cgm_features).
    other_features_shape : Tuple[int, ...]
        Forma de otras características (other_features,).

    Retorna:
    --------
    TCNModel
        Instancia del modelo TCN de Flax.
    """
    return TCNModel(
        config=TCN_CONFIG,
        cgm_shape=cgm_shape,
        other_features_shape=other_features_shape
    )

def create_tcn_model(cgm_shape: Tuple[int, ...], other_features_shape: Tuple[int, ...]) -> DLModelWrapper:
    """
    Crea un modelo TCN envuelto en DLModelWrapper para compatibilidad con el sistema de entrenamiento.

    Esta función es la que se pasa al sistema de entrenamiento (e.g., train_multiple_models).
    Retorna un DLModelWrapper que internamente sabe cómo crear la instancia cruda del TCNModel.

    Parámetros:
    -----------
    cgm_shape : Tuple[int, ...]
        Forma de los datos CGM (time_steps, cgm_features).
    other_features_shape : Tuple[int, ...]
        Forma de otras características (other_features,).

    Retorna:
    --------
    DLModelWrapper
        Modelo TCN envuelto y listo para ser usado por el sistema de entrenamiento JAX.
    """
    # La función que DLModelWrapper necesita es una que *cree* el modelo crudo.
    # Usamos una lambda para capturar las formas y pasarla al wrapper.
    # El wrapper llamará a esta lambda cuando necesite inicializar el modelo.
    raw_model_creator = lambda: _create_raw_tcn_model(cgm_shape, other_features_shape)

    # Envolver el *creador* del modelo crudo en DLModelWrapper (especificando backend 'jax')
    return DLModelWrapper(raw_model_creator, framework='jax')

def model_creator() -> Callable[[Tuple[int, ...], Tuple[int, ...]], DLModelWrapper]:
    """
    Retorna la función (`create_tcn_model`) que crea el modelo TCN envuelto.

    Esta función (`model_creator`) es la que se importa y se usa en `params.py`.
    No toma argumentos y devuelve la función que sí los toma (`create_tcn_model`).

    Retorna:
    --------
    Callable[[Tuple[int, ...], Tuple[int, ...]], DLModelWrapper]
        Función (`create_tcn_model`) que, dadas las formas de entrada, crea el modelo TCN envuelto.
    """
    return create_tcn_model