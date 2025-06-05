import os
import sys
import jax
import jax.numpy as jnp
import flax.linen as nn
import numpy as np
from typing import Tuple, List, Dict, Any, Optional, Callable, Sequence

PROJECT_ROOT = os.path.abspath(os.getcwd())
sys.path.append(PROJECT_ROOT) 

from config.models_config_old import WAVENET_CONFIG
from custom.DeepLearning.dl_model_wrapper import DLModelWrapper
from custom.printer import print_warning

# --- Constantes ---
CONST_ELU: str = "elu"
CONST_RELU: str = "relu"
CONST_GELU: str = "gelu"
CONST_SELU: str = "selu"
CONST_SIGMOID: str = "sigmoid"
CONST_TANH: str = "tanh"
CONST_CAUSAL: str = "CAUSAL"
CONST_VALID: str = "VALID"
CONST_SAME: str = "SAME"
CONST_DROPOUT: str = "dropout"

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

def _get_activation_fn(activation_name: str) -> Callable:
    """
    Obtiene la función de activación correspondiente a un nombre.

    Parámetros:
    -----------
    activation_name : str
        Nombre de la función de activación ('relu', 'gelu', 'elu', etc.).

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
    elif activation_name == CONST_ELU:
        return nn.elu
    else:
        print_warning(f"Advertencia: Función de activación '{activation_name}' no reconocida. Usando ReLU por defecto.")
        return nn.relu

class WaveNetBlock(nn.Module):
    """
    Bloque residual WaveNet con convoluciones dilatadas, compuertas y conexiones skip/residuales.

    Atributos:
    ----------
    filters : int
        Número de filtros en las capas convolucionales internas (filtro/compuerta).
    skip_filters : int  # <-- NUEVO: Dimensión para la salida skip
        Número de filtros para la conexión skip.
    kernel_size : int
        Tamaño del kernel convolucional.
    dilation_rate : int
        Tasa de dilatación para la convolución.
    dropout_rate : float
        Tasa de dropout a aplicar.
    use_gating : bool
        Indica si se debe usar el mecanismo de compuertas multiplicativas.
    activation_fn : Callable
        Función de activación a usar (usualmente tanh para el filtro).
    name : Optional[str]
        Nombre opcional para el bloque.
    """
    filters: int
    skip_filters: int 
    kernel_size: int
    dilation_rate: int
    dropout_rate: float
    use_gating: bool
    activation_fn: Callable # Usualmente tanh para el filtro
    name: Optional[str] = None

    @nn.compact
    def __call__(self, inputs: jnp.ndarray, training: bool) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """
        Aplica un bloque WaveNet a las entradas.

        Parámetros:
        -----------
        inputs : jnp.ndarray
            Tensor de entrada (batch, time_steps, features).
        training : bool
            Indica si el modelo está en modo entrenamiento (afecta dropout).

        Retorna:
        --------
        Tuple[jnp.ndarray, jnp.ndarray]
            - Salida para la conexión residual (mismas dimensiones que la entrada si es posible).
            - Salida para la conexión skip (con dimensión skip_filters).
        """
        deterministic = not training
        features_in = inputs.shape[-1]

        # Padding causal
        padding_size = (self.kernel_size - 1) * self.dilation_rate
        x_padded = causal_padding(inputs, padding_size)

        # Convolución dilatada para filtro
        x_filter = nn.Conv(
            features=self.filters, # Dimensión interna del bloque
            kernel_size=(self.kernel_size,),
            kernel_dilation=(self.dilation_rate,),
            padding=CONST_VALID, name="dilated_conv_filter"
        )(x_padded)

        if self.use_gating:
            # Convolución dilatada para compuerta
            x_gate = nn.Conv(
                features=self.filters, # Dimensión interna del bloque
                kernel_size=(self.kernel_size,),
                kernel_dilation=(self.dilation_rate,),
                padding=CONST_VALID, name="dilated_conv_gate"
            )(x_padded)
            # Aplicar activación (tanh) y compuerta (sigmoid)
            x_activated = self.activation_fn(x_filter) * jax.nn.sigmoid(x_gate)
        else:
            # Sin compuerta, solo aplicar activación
            x_activated = self.activation_fn(x_filter)

        # Dropout
        x_dropped = nn.Dropout(rate=self.dropout_rate, deterministic=deterministic)(x_activated)

        # Convolución 1x1 para salida residual
        # La salida residual debe tener la misma dimensión que la entrada para la suma
        residual_out = nn.Conv(features=features_in, kernel_size=(1,), name="residual_conv")(x_dropped)

        # Convolución 1x1 para salida skip
        # self.skip_filters para la dimensión de salida skip
        skip_out = nn.Conv(features=self.skip_filters, kernel_size=(1,), name="skip_conv")(x_dropped)

        # Conexión residual
        # Asegurar que las longitudes temporales coincidan después del padding='VALID'
        target_len = residual_out.shape[1]
        inputs_cropped = inputs[:, -target_len:, :] # Tomar los últimos pasos de la entrada original

        # La suma residual requiere que las dimensiones de características coincidan
        if inputs_cropped.shape[-1] != residual_out.shape[-1]:
             # Esto no debería ocurrir si residual_out usa features=features_in
             print_warning(f"Dimensión residual no coincide: input={inputs_cropped.shape}, residual={residual_out.shape}")
             # Si ocurriera, se necesitaría una proyección de 'inputs_cropped'
             output = residual_out # O manejar el error
        else:
             output = inputs_cropped + residual_out

        return output, skip_out

class WaveNetModel(nn.Module):
    """
    Modelo WaveNet completo usando JAX/Flax.

    Atributos:
    ----------
    config : Dict
        Configuración del modelo (WAVENET_CONFIG).
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
        self.filters = self.config['filters'] # Lista de filtros para los bloques
        self.kernel_size = self.config['kernel_size']
        self.dilations = self.config['dilations'] # Lista de dilataciones
        self.dropout_rate = self.config['dropout_rate']
        self.use_gating = self.config['use_gating']
        self.use_skip_scale = self.config['use_skip_scale']
        self.use_residual_scale = self.config['use_residual_scale']
        self.activation_name = self.config['activation']

        self.activation_fn = _get_activation_fn(self.activation_name)

        # Definir la dimensión skip deseada
        # Podría ser un valor fijo, el último filtro, o un nuevo parámetro en config
        # Usemos 128 por ahora, coincidiendo con post_skip_conv1
        self.skip_dim = 128

        # Capa causal inicial - proyecta a la dimensión del primer bloque
        self.causal_conv = nn.Conv(
            features=self.filters[0],
            kernel_size=(self.kernel_size,),
            padding=CONST_CAUSAL,
            name="causal_conv_initial"
        )

        # Bloques WaveNet
        self.wavenet_blocks = [
            WaveNetBlock(
                # Usar el tamaño de filtro correspondiente al nivel/bloque
                filters=self.filters[i % len(self.filters)],
                skip_filters=self.skip_dim,
                kernel_size=self.kernel_size,
                dilation_rate=dilation_rate,
                dropout_rate=self.dropout_rate,
                use_gating=self.use_gating,
                activation_fn=self.activation_fn, # Usualmente tanh para filtro
                name=f"wavenet_block_{i}"
            ) for i, dilation_rate in enumerate(self.dilations)
        ]

        # Capas finales después de sumar skips
        # Usar self.skip_dim para definir las características 
        self.post_skip_conv1 = nn.Conv(features=self.skip_dim, kernel_size=(1,), name="post_skip_conv1")

        # Capa densa después de pooling/flattening y concatenación
        # La dimensión combinada ahora depende de self.skip_dim
        combined_dim_calc = self.skip_dim + int(np.prod(self.other_features_shape)) # Cálculo para referencia
        self.dense1 = nn.Dense(combined_dim_calc, name="final_dense1")
        self.output_dense = nn.Dense(1, name="final_output_dense")

    @nn.compact
    def __call__(self, cgm_input: jnp.ndarray, other_input: jnp.ndarray, training: bool) -> jnp.ndarray:
        """
        Aplica el modelo WaveNet a las entradas.

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
        x = cgm_input
        x = self.causal_conv(x) # Salida: (batch, time_steps, filters[0])

        skip_outputs_list = [] # Lista para almacenar salidas skip

        # Aplicar bloques WaveNet
        for block in self.wavenet_blocks:
            x, skip_out = block(x, training=training) # skip_out ahora tiene self.skip_dim características
            skip_outputs_list.append(skip_out)

        # Sumar las salidas skip
        # Manejar lista vacía explícitamente
        if not skip_outputs_list:
            # Si no hay bloques, crear un tensor de ceros con la dimensión skip esperada
            # Necesitamos la longitud temporal de x después de causal_conv
            time_steps = x.shape[1]
            skip_outputs_sum = jnp.zeros((x.shape[0], time_steps, self.skip_dim))
        else:
            # Ahora todas las skip_out tienen la misma dimensión (self.skip_dim), la suma debería funcionar
            skip_outputs_sum = sum(skip_outputs_list) # Suma elemento a elemento

        # Post-procesamiento de skips
        x_post_skip = nn.relu(skip_outputs_sum)
        x_post_skip = self.post_skip_conv1(x_post_skip) # (batch, time_steps, self.skip_dim)
        x_post_skip = nn.relu(x_post_skip)

        # Tomar la salida del último paso temporal
        last_step_output = x_post_skip[:, -1, :] # (batch, self.skip_dim)

        # Combinar con otras características
        if other_input.ndim > 2:
             other_input_flat = jnp.reshape(other_input, (other_input.shape[0], -1))
        else:
             other_input_flat = other_input
        combined_features = jnp.concatenate([last_step_output, other_input_flat], axis=-1)

        # MLP final
        x_final = self.dense1(combined_features)
        x_final = nn.relu(x_final) # O self.activation_fn?
        output = self.output_dense(x_final)

        # Asegurar que la salida sea (batch,) para regresión MSE
        return output.squeeze(-1)

# --- Funciones de creación y envoltura ---
# (El resto de las funciones _create_raw_wavenet_model, create_wavenet_model, model_creator permanecen igual)

def _create_raw_wavenet_model(cgm_shape: Tuple[int, ...], other_features_shape: Tuple[int, ...]) -> WaveNetModel:
    """
    Crea una instancia cruda del modelo WaveNet (nn.Module).

    Parámetros:
    -----------
    cgm_shape : Tuple[int, ...]
        Forma de los datos CGM (time_steps, cgm_features).
    other_features_shape : Tuple[int, ...]
        Forma de otras características (other_features,).

    Retorna:
    --------
    WaveNetModel
        Instancia del modelo WaveNet de Flax.
    """
    # Validar configuración si es necesario
    if not isinstance(WAVENET_CONFIG.get('filters'), list) or not WAVENET_CONFIG['filters']:
         raise ValueError("WAVENET_CONFIG['filters'] debe ser una lista no vacía.")
    if not isinstance(WAVENET_CONFIG.get('dilations'), list) or not WAVENET_CONFIG['dilations']:
         raise ValueError("WAVENET_CONFIG['dilations'] debe ser una lista no vacía.")

    return WaveNetModel(
        config=WAVENET_CONFIG,
        cgm_shape=cgm_shape,
        other_features_shape=other_features_shape
    )

def create_wavenet_model(cgm_shape: Tuple[int, ...], other_features_shape: Tuple[int, ...]) -> DLModelWrapper:
    """
    Crea un modelo WaveNet envuelto en DLModelWrapper para compatibilidad con el sistema de entrenamiento JAX.

    Parámetros:
    -----------
    cgm_shape : Tuple[int, ...]
        Forma de los datos CGM (time_steps, cgm_features).
    other_features_shape : Tuple[int, ...]
        Forma de otras características (other_features,).

    Retorna:
    --------
    DLModelWrapper
        Modelo WaveNet envuelto y listo para ser usado por el sistema de entrenamiento JAX.
    """
    # Función que crea la instancia cruda del modelo
    raw_model_creator = lambda: _create_raw_wavenet_model(cgm_shape, other_features_shape)

    # Envolver el creador en DLModelWrapper
    return DLModelWrapper(raw_model_creator, framework='jax')

def model_creator() -> Callable[[Tuple[int, ...], Tuple[int, ...]], DLModelWrapper]:
    """
    Retorna la función (`create_wavenet_model`) que crea el modelo WaveNet envuelto.

    Esta función (`model_creator`) es la que se importa y se usa en `params.py`.
    No toma argumentos y devuelve la función que sí los toma (`create_wavenet_model`).

    Retorna:
    --------
    Callable[[Tuple[int, ...], Tuple[int, ...]], DLModelWrapper]
        Función (`create_wavenet_model`) que, dadas las formas de entrada, crea el modelo WaveNet envuelto.
    """
    return create_wavenet_model
