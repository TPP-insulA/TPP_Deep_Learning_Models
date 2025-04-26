import os
import sys
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input, Conv1D, Dense, Dropout, Add, Activation,
    BatchNormalization, GlobalAveragePooling1D, Concatenate
)
from keras.saving import register_keras_serializable # type: ignore # register_keras_serializable está en keras.saving
from typing import Tuple, Dict, List, Any, Optional, Union

PROJECT_ROOT = os.path.abspath(os.getcwd())
sys.path.append(PROJECT_ROOT) 

from config.models_config import WAVENET_CONFIG

# --- Constantes ---
CONST_RELU: str = 'relu'
CONST_GELU: str = 'gelu'
CONST_ELU: str = 'elu'
CONST_TANH: str = 'tanh'
CONST_SIGMOID: str = 'sigmoid'
CONST_SAME: str = 'same'
CONST_CAUSAL: str = 'causal'
CONST_INPUT_CGM: str = 'cgm_input'
CONST_INPUT_OTHER: str = 'other_input'
CONST_OUTPUT: str = 'output'


@register_keras_serializable()
class WavenetBlock(tf.keras.layers.Layer):
    """
    Bloque WaveNet mejorado con activaciones gated y escalado adaptativo.

    Atributos:
    ----------
    filters : int
        Número de filtros para las convoluciones.
    kernel_size : int
        Tamaño del kernel convolucional.
    dilation_rate : int
        Tasa de dilatación para la convolución.
    dropout_rate : float
        Tasa de dropout.
    residual_scale : float
        Factor de escala para la conexión residual.
    use_skip_scale : bool
        Indica si se debe escalar la conexión skip.
    """
    def __init__(self, filters: int, kernel_size: int, dilation_rate: int, dropout_rate: float, **kwargs) -> None:
        """
        Se inicializa el bloque WaveNet.

        Parámetros:
        -----------
        filters : int
            Número de filtros para las convoluciones.
        kernel_size : int
            Tamaño del kernel convolucional.
        dilation_rate : int
            Tasa de dilatación para la convolución.
        dropout_rate : float
            Tasa de dropout.
        """
        super().__init__(**kwargs)
        self.filters = filters
        self.kernel_size = kernel_size
        self.dilation_rate = dilation_rate
        self.dropout_rate = dropout_rate

        # Convoluciones Gated
        self.filter_conv = Conv1D(
            filters=filters,
            kernel_size=kernel_size,
            dilation_rate=dilation_rate,
            padding=CONST_CAUSAL,
            name=f"filter_conv_d{dilation_rate}"
        )
        self.gate_conv = Conv1D(
            filters=filters,
            kernel_size=kernel_size,
            dilation_rate=dilation_rate,
            padding=CONST_CAUSAL,
            name=f"gate_conv_d{dilation_rate}"
        )

        # Normalización y regularización
        self.filter_norm = BatchNormalization(name=f"filter_norm_d{dilation_rate}")
        self.gate_norm = BatchNormalization(name=f"gate_norm_d{dilation_rate}")
        self.dropout = Dropout(dropout_rate, name=f"dropout_d{dilation_rate}")

        # Proyecciones
        self.residual_proj = Conv1D(filters, 1, padding=CONST_SAME, name=f"residual_proj_d{dilation_rate}")
        self.skip_proj = Conv1D(filters, 1, padding=CONST_SAME, name=f"skip_proj_d{dilation_rate}")

        # Factores de escala (desde config)
        self.residual_scale = WAVENET_CONFIG['use_residual_scale']
        self.use_skip_scale = WAVENET_CONFIG['use_skip_scale']

    def call(self, inputs: tf.Tensor, training: Optional[bool] = None) -> Tuple[tf.Tensor, tf.Tensor]:
        """
        Se aplica un bloque WaveNet a la entrada.

        Parámetros:
        -----------
        inputs : tf.Tensor
            Tensor de entrada.
        training : Optional[bool], opcional
            Indica si está en modo entrenamiento (default: None).

        Retorna:
        --------
        Tuple[tf.Tensor, tf.Tensor]
            (salida_residual, salida_skip)
        """
        # Activación Gated
        filter_out = self.filter_conv(inputs)
        gate_out = self.gate_conv(inputs)

        filter_out = self.filter_norm(filter_out, training=training)
        gate_out = self.gate_norm(gate_out, training=training)

        # tanh(filter) * sigmoid(gate)
        gated_out = tf.nn.tanh(filter_out) * tf.nn.sigmoid(gate_out)
        gated_out = self.dropout(gated_out, training=training)

        # Conexión Residual
        # Proyectar la entrada original para que coincida con la dimensión de filtros
        residual_input_proj = self.residual_proj(inputs)
        # Alinear la dimensión temporal de la entrada proyectada con la salida gated
        # La convolución causal reduce la longitud, por eso se toma la parte final
        residual_input_aligned = residual_input_proj[:, -tf.shape(gated_out)[1]:, :]

        residual_out = (gated_out * self.residual_scale) + residual_input_aligned

        # Conexión Skip
        skip_out = self.skip_proj(gated_out)
        if self.use_skip_scale:
            # Escalar la salida skip
            skip_out = skip_out * tf.math.sqrt(1.0 - self.residual_scale) # Escala complementaria a residual

        return residual_out, skip_out

    def get_config(self) -> Dict[str, Any]:
        """
        Se obtiene la configuración del bloque para serialización.

        Retorna:
        --------
        Dict[str, Any]
            Configuración del bloque.
        """
        config = super().get_config()
        config.update({
            'filters': self.filters,
            'kernel_size': self.kernel_size,
            'dilation_rate': self.dilation_rate,
            'dropout_rate': self.dropout_rate
        })
        return config

def create_wavenet_model(cgm_shape: Tuple[int, ...], other_features_shape: Tuple[int, ...]) -> Model:
    """
    Se crea un modelo WaveNet usando la API funcional de Keras.
    # ... (docstring remains the same) ...
    """
    # Definición de Entradas
    cgm_input = Input(shape=cgm_shape, name=CONST_INPUT_CGM)
    other_input = Input(shape=other_features_shape, name=CONST_INPUT_OTHER)

    # Configuración
    activation_name = WAVENET_CONFIG['activation']
    config_filters = WAVENET_CONFIG['filters']
    kernel_size = WAVENET_CONFIG['kernel_size'] # Get kernel_size from config
    dilations = WAVENET_CONFIG['dilations']     # Get dilations from config
    dropout_rate = WAVENET_CONFIG['dropout_rate'] # Get dropout_rate from config
    # Dimensión objetivo para las skip connections
    skip_channels = config_filters[-1]

    # Proyección inicial
    initial_filters = config_filters[0]
    x = Conv1D(initial_filters, 1, padding=CONST_SAME, name='initial_projection')(cgm_input)
    x = BatchNormalization(name='initial_norm')(x)
    x = Activation(activation_name, name='initial_activation')(x)

    # Acumulador para las salidas de las conexiones skip
    skip_outputs: List[tf.Tensor] = []

    # Pila de bloques WaveNet
    current_input = x
    block_counter = 0 # Initialize counter outside the loops
    for i, filters in enumerate(config_filters):
        for dilation in dilations:
            # Crear y aplicar el bloque WaveNet personalizado
            block = WavenetBlock(
                filters=filters,
                kernel_size=kernel_size, # Use variable from config
                dilation_rate=dilation,
                dropout_rate=dropout_rate, # Use variable from config
                name=f"wavenet_block_{block_counter}"
            )
            current_input, skip = block(current_input)

            # Proyectar la salida skip a la dimensión 'skip_channels'
            projected_skip = Conv1D(
                filters=skip_channels,
                kernel_size=1,
                padding=CONST_SAME,
                # Use block_counter for unique projection name
                name=f"skip_projection_{block_counter}"
            )(skip)
            skip_outputs.append(projected_skip)

            block_counter += 1

    # Combinar las salidas de las conexiones skip
    if not skip_outputs:
        combined_skip = current_input
        # Asegurarse de que combined_skip tenga la dimensión correcta si se usa directamente
        if tf.keras.backend.int_shape(combined_skip)[-1] != skip_channels:
             combined_skip = Conv1D(skip_channels, 1, padding=CONST_SAME, name='final_skip_proj_fallback')(combined_skip)
    else:
        combined_skip = Add(name='add_skip_connections')(skip_outputs)

    # Post-procesamiento después de combinar skips
    post_skip = Activation(activation_name, name='post_skip_activation1')(combined_skip)
    post_skip = Conv1D(skip_channels, 1, padding=CONST_SAME, name='post_skip_conv1')(post_skip) # Use skip_channels here
    post_skip = Activation(activation_name, name='post_skip_activation2')(post_skip)

    # Pooling
    pooled_output = GlobalAveragePooling1D(name='global_avg_pooling')(post_skip)

    # Combinación con otras características
    combined_features = Concatenate(name='concat_features')([pooled_output, other_input])

    # Capas densas finales (MLP)
    dense_output = Dense(128, name='final_dense_1')(combined_features)
    dense_output = BatchNormalization(name='final_norm_1')(dense_output)
    dense_output = Activation(activation_name, name='final_activation_1')(dense_output)
    dense_output = Dropout(dropout_rate, name='final_dropout_1')(dense_output) # Use variable from config

    # Segunda capa densa
    dense_output = Dense(64, name='final_dense_2')(dense_output)
    dense_output = BatchNormalization(name='final_norm_2')(dense_output)
    dense_output = Activation(activation_name, name='final_activation_2')(dense_output)
    dense_output = Dropout(dropout_rate, name='final_dropout_2')(dense_output) # Use variable from config

    # Capa de salida final
    final_output = Dense(1, name=CONST_OUTPUT)(dense_output)

    # Crear el modelo final
    model = Model(inputs=[cgm_input, other_input], outputs=final_output, name='wavenet_model')

    return model