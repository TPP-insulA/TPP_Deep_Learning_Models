import os, sys
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input, Conv1D, Dense, Dropout, Add, Activation, LayerNormalization,
    BatchNormalization, GlobalAveragePooling1D, Concatenate, SpatialDropout1D,
    Reshape
)
from tensorflow.keras.constraints import max_norm
from typing import Callable, Tuple, Dict, List, Any, Optional, Union

PROJECT_ROOT = os.path.abspath(os.getcwd())
sys.path.append(PROJECT_ROOT) 

from config.models_config import TCN_CONFIG
from custom.dl_model_wrapper import DLModelWrapper

# Constantes para uso repetido
CONST_VALID = "valid"
CONST_SAME = "same"
CONST_CAUSAL = "causal"
CONST_RELU = "relu"
CONST_GELU = "gelu"

class TCNBlock(tf.keras.layers.Layer):
    """
    Bloque para Temporal Convolutional Network (TCN) con dilatación y skip connections.
    
    Parámetros:
    -----------
    filters : int
        Número de filtros para la capa convolucional
    kernel_size : int
        Tamaño del kernel para la convolución
    dilation_rate : int
        Tasa de dilatación para aumentar el campo receptivo
    dropout_rate : float
        Tasa de dropout para regularización
    use_layer_norm : bool
        Si usar normalización de capa en lugar de batch normalization
    use_spatial_dropout : bool
        Si usar dropout espacial (canales completos) o estándar
    activation : str
        Nombre de la función de activación a utilizar
    use_weight_norm : bool
        Si aplicar normalización de peso a la capa convolucional
    """
    def __init__(
        self, 
        filters: int, 
        kernel_size: int, 
        dilation_rate: int,
        dropout_rate: float,
        use_layer_norm: bool = True,
        use_spatial_dropout: bool = True,
        activation: str = "gelu",
        use_weight_norm: bool = True,
        **kwargs
    ):
        super(TCNBlock, self).__init__(**kwargs)
        self.filters = filters
        self.kernel_size = kernel_size
        self.dilation_rate = dilation_rate
        self.dropout_rate = dropout_rate
        self.use_layer_norm = use_layer_norm
        self.use_spatial_dropout = use_spatial_dropout
        self.activation = activation
        self.use_weight_norm = use_weight_norm
        
        # Inicializar capas para usar en call()
        # Convolución dilatada causal
        if self.use_weight_norm:
            # Para normalización de peso, usaremos un truco con constraints
            self.conv = Conv1D(
                filters=filters,
                kernel_size=kernel_size,
                padding=CONST_CAUSAL,
                dilation_rate=dilation_rate,
                kernel_constraint=max_norm(2.0),
                kernel_initializer='he_normal',
                name=f'tcn_conv_d{dilation_rate}'
            )
        else:
            self.conv = Conv1D(
                filters=filters,
                kernel_size=kernel_size,
                padding=CONST_CAUSAL,
                dilation_rate=dilation_rate,
                kernel_initializer='he_normal',
                name=f'tcn_conv_d{dilation_rate}'
            )
            
        # Normalización
        if use_layer_norm:
            self.norm = LayerNormalization(epsilon=TCN_CONFIG['epsilon'])
        else:
            self.norm = BatchNormalization()
        
        # Activación
        self.act = Activation(self.get_activation_fn())
        
        # Dropout
        if use_spatial_dropout:
            self.dropout = SpatialDropout1D(dropout_rate)
        else:
            self.dropout = Dropout(dropout_rate)
        
        # Projección para residual connection si es necesario
        self.residual_proj = Conv1D(filters, 1, padding=CONST_SAME)

    def get_activation_fn(self) -> str:
        """
        Devuelve el nombre de la función de activación para Keras.
        
        Retorna:
        --------
        str
            Nombre de la activación en formato Keras
        """
        if self.activation == CONST_GELU:
            return 'gelu'
        elif self.activation == CONST_RELU:
            return 'relu'
        else:
            return 'relu'  # Default

    def build(self, input_shape: tf.TensorShape) -> None:
        """
        Construye las capas del bloque cuando se conoce la forma de entrada.
        
        Parámetros:
        -----------
        input_shape : tf.TensorShape
            Forma del tensor de entrada
        """
        super(TCNBlock, self).build(input_shape)
        
    def call(self, inputs: tf.Tensor, training: Optional[bool] = None) -> tf.Tensor:
        """
        Aplica el bloque TCN a los datos de entrada.
        
        Parámetros:
        -----------
        inputs : tf.Tensor
            Tensor de entrada
        training : bool, opcional
            Si está en modo entrenamiento
            
        Retorna:
        --------
        tf.Tensor
            Tensor de salida
        """
        # Aplicar convolución causal dilatada
        x = self.conv(inputs)
        
        # Normalización
        x = self.norm(x, training=training)
        
        # Activación
        x = self.act(x)
            
        # Dropout
        x = self.dropout(x, training=training)
        
        # Conexión residual con proyección si es necesario
        if inputs.shape[-1] != self.filters:
            # Proyectar entrada a la misma dimensión que la salida
            residual = self.residual_proj(inputs)
        else:
            residual = inputs
            
        # Alinear dimensiones temporales
        # La operación de convolución causal puede reducir la longitud temporal
        if residual.shape[1] > x.shape[1]:
            residual = residual[:, -x.shape[1]:, :]
        
        # Conexión residual
        output = x + residual
        
        return output
    
    def get_config(self) -> Dict[str, Any]:
        """
        Obtiene la configuración del bloque.
        
        Retorna:
        --------
        Dict[str, Any]
            Configuración del bloque para serialización
        """
        config = super(TCNBlock, self).get_config()
        config.update({
            'filters': self.filters,
            'kernel_size': self.kernel_size,
            'dilation_rate': self.dilation_rate,
            'dropout_rate': self.dropout_rate,
            'use_layer_norm': self.use_layer_norm,
            'use_spatial_dropout': self.use_spatial_dropout,
            'activation': self.activation,
            'use_weight_norm': self.use_weight_norm
        })
        return config

def get_activation_layer(activation_name: str) -> Activation:
    """
    Devuelve una capa de activación Keras según el nombre.
    
    Parámetros:
    -----------
    activation_name : str
        Nombre de la activación a aplicar
        
    Retorna:
    --------
    Activation
        Capa de activación de Keras
    """
    if activation_name == CONST_GELU:
        return Activation('gelu')
    elif activation_name == CONST_RELU:
        return Activation('relu')
    else:
        return Activation('relu')  # Default

def _prepare_input_tensors(cgm_shape: Tuple[int, ...], other_features_shape: Tuple[int, ...]) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
    """
    Prepara los tensores de entrada para el modelo TCN.
    
    Parámetros:
    -----------
    cgm_shape : Tuple[int, ...]
        Forma de los datos CGM
    other_features_shape : Tuple[int, ...]
        Forma de otras características
        
    Retorna:
    --------
    Tuple[tf.Tensor, tf.Tensor, tf.Tensor]
        cgm_input, reshaped_cgm, other_input
    """
    # Validar other_features_shape
    if not isinstance(other_features_shape, tuple) or len(other_features_shape) < 1:
        raise ValueError(f"other_features_shape debe ser un tuple con al menos un elemento. Recibido: {other_features_shape}")
    
    # Preparar entrada CGM
    if len(cgm_shape) == 1:
        cgm_input = Input(shape=(cgm_shape[0],), name='cgm_input')
        reshaped_cgm = Reshape((1, cgm_shape[0]))(cgm_input)
    else:
        cgm_input = Input(shape=cgm_shape, name='cgm_input')
        reshaped_cgm = cgm_input
    
    # Preparar otras entradas
    other_input_shape = (other_features_shape[0],) if len(other_features_shape) == 1 else other_features_shape
    other_input = Input(shape=other_input_shape, name='other_input')
    
    return cgm_input, reshaped_cgm, other_input

def _build_tcn_blocks(x: tf.Tensor) -> Tuple[tf.Tensor, List[tf.Tensor]]:
    """
    Construye los bloques TCN y maneja las skip connections.
    
    Parámetros:
    -----------
    x : tf.Tensor
        Tensor de entrada
        
    Retorna:
    --------
    Tuple[tf.Tensor, List[tf.Tensor]]
        Tensor de salida y lista de skip connections combinadas
    """
    skip_connections_by_level = {}
    
    # Construir bloques TCN
    for layer_idx, filters in enumerate(TCN_CONFIG['filters']):
        skip_connections_by_level[layer_idx] = []
        
        for dilation_idx, dilation_rate in enumerate(TCN_CONFIG['dilations']):
            block_name = f'tcn_block_{layer_idx}_{dilation_idx}'
            
            x = TCNBlock(
                filters=filters,
                kernel_size=TCN_CONFIG['kernel_size'],
                dilation_rate=dilation_rate,
                dropout_rate=TCN_CONFIG['dropout_rate'][0],
                use_layer_norm=TCN_CONFIG['use_layer_norm'],
                use_spatial_dropout=TCN_CONFIG['use_spatial_dropout'],
                activation=TCN_CONFIG['activation'],
                use_weight_norm=TCN_CONFIG['use_weight_norm'],
                name=block_name
            )(x)
            
            skip_connections_by_level[layer_idx].append(x)
    
    # Procesar skip connections
    combined_skips_by_level = []
    
    for layer_idx, skips in skip_connections_by_level.items():
        combined = Add(name=f'skip_connections_level_{layer_idx}')(skips) if len(skips) > 1 else skips[0]
        
        if combined.shape[-1] != TCN_CONFIG['filters'][-1]:
            proj_name = f'skip_proj_level_{layer_idx}'
            combined = Conv1D(TCN_CONFIG['filters'][-1], 1, padding=CONST_SAME, name=proj_name)(combined)
        
        combined_skips_by_level.append(combined)
        
    return x, combined_skips_by_level

def _build_output_layers(x: tf.Tensor, other_input: tf.Tensor) -> tf.Tensor:
    """
    Construye las capas finales del modelo (pooling, concatenación y MLP).
    
    Parámetros:
    -----------
    x : tf.Tensor
        Tensor de salida de los bloques TCN
    other_input : tf.Tensor
        Tensor con otras características
        
    Retorna:
    --------
    tf.Tensor
        Tensor de salida final
    """
    # Post-procesamiento
    x = get_activation_layer(TCN_CONFIG['activation'])(x)
    x = GlobalAveragePooling1D(name='global_pooling')(x)
    
    # Combinar con otras características
    x = Concatenate(name='combine_features')([x, other_input])
    
    # Primera capa densa
    x = Dense(128, name='dense_1')(x)
    x = LayerNormalization(epsilon=TCN_CONFIG['epsilon'], name='norm_1')(x)
    x = get_activation_layer(TCN_CONFIG['activation'])(x)
    x = Dropout(TCN_CONFIG['dropout_rate'][1], name='dropout_1')(x)
    
    # Segunda capa densa
    x = Dense(64, name='dense_2')(x)
    x = LayerNormalization(epsilon=TCN_CONFIG['epsilon'], name='norm_2')(x)
    x = get_activation_layer(TCN_CONFIG['activation'])(x)
    x = Dropout(TCN_CONFIG['dropout_rate'][1], name='dropout_2')(x)
    
    # Salida
    return Dense(1, name='output')(x)

def create_tcn_model(cgm_shape: Tuple[int, ...], other_features_shape: Tuple[int, ...]) -> Model:
    """
    Crea un modelo TCN (Temporal Convolutional Network) para series temporales.
    
    Parámetros:
    -----------
    cgm_shape : Tuple[int, ...]
        Forma de los datos CGM (pasos_temporales, características)
    other_features_shape : Tuple[int, ...]
        Forma de otras características (características)
        
    Retorna:
    --------
    Model
        Modelo TCN compilado
    """
    # Preparar tensores de entrada
    cgm_input, reshaped_cgm, other_input = _prepare_input_tensors(cgm_shape, other_features_shape)
    
    # Proyección inicial
    x = Conv1D(
        filters=TCN_CONFIG['filters'][0], 
        kernel_size=1, 
        padding=CONST_SAME, 
        name='initial_projection'
    )(reshaped_cgm)
    
    # Construir bloques TCN
    _, combined_skips_by_level = _build_tcn_blocks(x)
    
    # Combinar todos los niveles
    if len(combined_skips_by_level) > 1:
        x = Add(name='skip_connections_final')(combined_skips_by_level)
    else:
        x = combined_skips_by_level[0]
    
    # Construir capas de salida
    output = _build_output_layers(x, other_input)
    
    # Crear y devolver modelo
    return Model(inputs=[cgm_input, other_input], outputs=output)

def model_creator() -> Callable:
    """
    Función fábrica para crear un modelo TCN compatible con la API del framework.
    
    Retorna:
    --------
    Callable
        Función para crear un modelo TCN
    """
    return DLModelWrapper(create_tcn_model, framework="tensorflow")