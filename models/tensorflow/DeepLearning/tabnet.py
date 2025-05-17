import os, sys
import tensorflow as tf
from keras._tf_keras.keras.models import Model
from keras._tf_keras.keras.layers import (
    Input, Dense, Dropout, BatchNormalization, LayerNormalization,
    Concatenate, Multiply, Add, Reshape, Activation, Layer
)
from typing import Tuple, Dict, List, Any, Optional, Union, Callable

PROJECT_ROOT = os.path.abspath(os.getcwd())
sys.path.append(PROJECT_ROOT) 

from config.models_config import TABNET_CONFIG
from custom.DeepLearning.dl_model_wrapper import DLModelWrapper

# Constantes para nombres repetidos
CONST_ACTIVACION_TANH = "tanh"
CONST_FEATURE_MASK = "feature_mask"
CONST_GATED_FEATURE_TRANSFORM = "gated_feature_transform"
CONST_STEP = "step"
CONST_ATTENTION = "attention"
CONST_TANH = "tanh"
CONST_SIGMOID = "sigmoid"
CONST_SAME = "same"


class GatedLinearUnit(tf.keras.layers.Layer):
    """
    Implementación de Unidad Lineal con Compuerta (GLU) como capa personalizada.
    
    Parámetros:
    -----------
    units : int
        Número de unidades de salida
    activation : str, opcional
        Activación para la compuerta (default: "sigmoid")
    use_bias : bool, opcional
        Si usar bias en la capa densa (default: True)
    """
    def __init__(self, units: int, activation: str = "sigmoid", 
                 use_bias: bool = True, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.units = units
        self.activation = activation
        self.use_bias = use_bias
        
    def build(self, input_shape: Tuple) -> None:
        """
        Construye la capa con los pesos necesarios.
        
        Parámetros:
        -----------
        input_shape : Tuple
            Forma del tensor de entrada
        """
        self.dense = Dense(self.units * 2, use_bias=self.use_bias)
        self.activation_fn = Activation(self.activation)
        
    def call(self, inputs: tf.Tensor) -> tf.Tensor:
        """
        Aplica la transformación GLU a las entradas.
        
        Parámetros:
        -----------
        inputs : tf.Tensor
            Tensor de entrada
            
        Retorna:
        --------
        tf.Tensor
            Salida transformada
        """
        dense_out = self.dense(inputs)
        
        # Dividir la salida en dos partes
        linear_out, gating_out = tf.split(dense_out, 2, axis=-1)
        
        # Aplicar función de activación a la compuerta
        gate = self.activation_fn(gating_out)
        
        # Multiplicar la parte lineal con la compuerta
        return Multiply()([linear_out, gate])
    
    def get_config(self) -> Dict[str, Any]:
        """
        Obtiene la configuración de la capa para serialización.
        
        Retorna:
        --------
        Dict[str, Any]
            Diccionario con la configuración
        """
        config = super().get_config()
        config.update({
            "units": self.units,
            "activation": self.activation,
            "use_bias": self.use_bias
        })
        return config


class GhostBatchNormalization(tf.keras.layers.Layer):
    """
    Implementación de Normalización por Lotes Fantasma para conjuntos de datos pequeños.
    
    Parámetros:
    -----------
    virtual_batch_size : int, opcional
        Tamaño de lote virtual para normalización (default: None)
    momentum : float, opcional
        Momentum para actualización de estadísticas (default: 0.9)
    epsilon : float, opcional
        Valor para estabilidad numérica (default: 1e-5)
    """
    def __init__(self, virtual_batch_size: Optional[int] = None, 
                 momentum: float = 0.9, epsilon: float = 1e-5, 
                 **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.virtual_batch_size = virtual_batch_size
        self.momentum = momentum
        self.epsilon = epsilon
        
    def build(self, input_shape: Tuple) -> None:
        """
        Construye la capa con los normalizadores necesarios.
        
        Parámetros:
        -----------
        input_shape : Tuple
            Forma del tensor de entrada
        """
        self.norm = BatchNormalization(
            momentum=self.momentum,
            epsilon=self.epsilon
        )
        
    def call(self, inputs: tf.Tensor, training: bool = False) -> tf.Tensor:
        """
        Aplica normalización por lotes fantasma.
        
        Parámetros:
        -----------
        inputs : tf.Tensor
            Tensor de entrada
        training : bool, opcional
            Si está en modo entrenamiento (default: False)
            
        Retorna:
        --------
        tf.Tensor
            Tensor normalizado
        """
        if not training or self.virtual_batch_size is None:
            return self.norm(inputs, training=training)
        
        # Para entrenamiento con tamaño virtual, dividir los lotes
        batch_size = tf.shape(inputs)[0]
        
        def apply_standard_normalization():
            return self.norm(inputs, training=training)
        
        def apply_ghost_batch_normalization():
            # Implementación de lotes virtuales
            # Calcular cuántos lotes virtuales completos caben
            num_virtual_batches = tf.cast(batch_size / self.virtual_batch_size, tf.int32)
            
            # División segura: solo procesar un múltiplo del tamaño de lote virtual
            valid_batch_size = num_virtual_batches * self.virtual_batch_size
            
            # Extraer la parte válida del lote
            valid_inputs = inputs[:valid_batch_size]
            
            # Reshape para tener [num_virtual_batches, virtual_batch_size, features]
            feature_dim = tf.shape(inputs)[1]
            reshaped_inputs = tf.reshape(valid_inputs, [num_virtual_batches, self.virtual_batch_size, feature_dim])
            
            # Normalizar cada lote virtual
            normalized_inputs = tf.TensorArray(tf.float32, size=num_virtual_batches)
            
            def normalize_batch(i, normalized_array):
                virtual_batch = reshaped_inputs[i]
                normalized = self.norm(virtual_batch, training=training)
                return i + 1, normalized_array.write(i, normalized)
            
            _, normalized_results = tf.while_loop(
                lambda i, _: i < num_virtual_batches,
                normalize_batch,
                [0, normalized_inputs]
            )
            
            # Concatenar resultados normalizados
            normalized_batches = tf.TensorArray.stack(normalized_results)
            concatenated = tf.reshape(normalized_batches, [-1, feature_dim])
            
            # Manejar el remanente (si existe)
            remainder_exists = tf.less(valid_batch_size, batch_size)
            
            def handle_remainder():
                remainder = inputs[valid_batch_size:]
                remainder_normalized = self.norm(remainder, training=training)
                return tf.concat([concatenated, remainder_normalized], axis=0)
            
            def no_remainder():
                return concatenated
            
            # Usar cond para manejar el remanente
            final_output = tf.cond(
                remainder_exists,
                handle_remainder,
                no_remainder
            )
            
            return final_output
        
        # Usar cond para elegir entre normalización estándar y fantasma
        return tf.cond(
            tf.less_equal(batch_size, self.virtual_batch_size), 
            apply_standard_normalization,
            apply_ghost_batch_normalization
        )
    
    def get_config(self) -> Dict[str, Any]:
        """
        Obtiene la configuración de la capa para serialización.
        
        Retorna:
        --------
        Dict[str, Any]
            Diccionario con la configuración
        """
        config = super().get_config()
        config.update({
            "virtual_batch_size": self.virtual_batch_size,
            "momentum": self.momentum,
            "epsilon": self.epsilon
        })
        return config


class SoftmaxLayer(Layer):
    """
    Capa personalizada para aplicar softmax a lo largo de un eje específico.
    
    Parámetros:
    -----------
    axis : int, opcional
        Eje a lo largo del cual aplicar softmax (default: -1)
    """
    def __init__(self, axis: int = -1, **kwargs):
        super().__init__(**kwargs)
        self.axis = axis
        
    def call(self, inputs):
        return tf.nn.softmax(inputs, axis=self.axis)
        
    def get_config(self):
        config = super().get_config()
        config.update({"axis": self.axis})
        return config


class ReduceMeanLayer(Layer):
    """
    Capa personalizada para reducir la media a lo largo de un eje específico.
    
    Parámetros:
    -----------
    axis : int, opcional
        Eje a lo largo del cual reducir (default: 2)
    """
    def __init__(self, axis: int = 2, **kwargs):
        super().__init__(**kwargs)
        self.axis = axis
        
    def call(self, inputs):
        return tf.reduce_mean(inputs, axis=self.axis)
        
    def get_config(self):
        config = super().get_config()
        config.update({"axis": self.axis})
        return config


class ReduceSumLayer(Layer):
    """
    Capa personalizada para reducir la suma a lo largo de un eje específico.
    
    Parámetros:
    -----------
    axis : int, opcional
        Eje a lo largo del cual reducir (default: 1)
    """
    def __init__(self, axis: int = 1, **kwargs):
        super().__init__(**kwargs)
        self.axis = axis
        
    def call(self, inputs):
        return tf.reduce_sum(inputs, axis=self.axis)
        
    def get_config(self):
        config = super().get_config()
        config.update({"axis": self.axis})
        return config


class ScaleLayer(Layer):
    """
    Capa personalizada para escalar un tensor.
    
    Parámetros:
    -----------
    scale_factor : float
        Factor de escala
    """
    def __init__(self, scale_factor: float = 1.0, **kwargs):
        super().__init__(**kwargs)
        self.scale_factor = scale_factor
        
    def call(self, inputs):
        return inputs * self.scale_factor
        
    def get_config(self):
        config = super().get_config()
        config.update({"scale_factor": self.scale_factor})
        return config


class OnesLikeLayer(Layer):
    """
    Capa personalizada que implementa tf.ones_like.
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        
    def call(self, inputs):
        return tf.ones_like(inputs)
        
    def get_config(self):
        return super().get_config()


class SubtractLayer(Layer):
    """
    Capa personalizada para realizar una resta (1 - x).
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        
    def call(self, inputs):
        return 1.0 - inputs
        
    def get_config(self):
        return super().get_config()


class ShapeValidationLayer(Layer):
    """
    Capa de validación que verifica las formas de los tensores y los registra.
    Útil para depurar problemas de forma en el modelo.
    
    Parámetros:
    -----------
    expected_shape : Optional[Tuple], opcional
        Forma esperada del tensor (excluyendo la dimensión del lote)
    name_for_error : str, opcional
        Nombre descriptivo para mensajes de error
    """
    def __init__(self, expected_shape: Optional[Tuple] = None, 
                 name_for_error: str = "tensor", **kwargs):
        super().__init__(**kwargs)
        self.expected_shape = expected_shape
        self.name_for_error = name_for_error
        
    def call(self, inputs):
        # Registrar la forma del tensor (para depuración)
        tf.print(f"Forma de {self.name_for_error}:", tf.shape(inputs), 
                 output_stream=tf.compat.v1.logging.info)
        
        # Validar forma si se especificó una esperada
        if self.expected_shape is not None:
            # Ignorar dimensión del lote (siempre es variable)
            actual_shape = inputs.shape[1:]
            
            # Comprobar compatibilidad (ignorando dimensiones None)
            if len(actual_shape) != len(self.expected_shape):
                tf.print(f"ERROR en {self.name_for_error}: El número de dimensiones no coincide.",
                         f"Esperado: {self.expected_shape}, Actual: {actual_shape}",
                         output_stream=tf.compat.v1.logging.error)
            else:
                for i, (actual, expected) in enumerate(zip(actual_shape, self.expected_shape)):
                    if expected is not None and actual is not None and actual != expected:
                        tf.print(f"ERROR en {self.name_for_error}: Dimensión {i+1} no coincide.",
                                f"Esperado: {expected}, Actual: {actual}",
                                output_stream=tf.compat.v1.logging.error)
        
        return inputs
    
    def get_config(self):
        config = super().get_config()
        config.update({
            "expected_shape": self.expected_shape,
            "name_for_error": self.name_for_error
        })
        return config


def feature_transformer(inputs: tf.Tensor, feature_dim: int, 
                        step_idx: int,
                        num_heads: int = 4, use_bn: bool = True, 
                        virtual_batch_size: Optional[int] = None,
                        dropout_rate: float = 0.2) -> tf.Tensor:
    """
    Aplica transformación de características con atención multi-cabeza.
    
    Parámetros:
    -----------
    inputs : tf.Tensor
        Tensor de entrada
    feature_dim : int
        Dimensión de características de salida
    step_idx : int
        Índice del paso actual para crear nombres únicos
    num_heads : int, opcional
        Número de cabezas de atención (default: 4)
    use_bn : bool, opcional
        Si usar normalización por lotes (default: True)
    virtual_batch_size : Optional[int], opcional
        Tamaño de lote virtual (default: None)
    dropout_rate : float, opcional
        Tasa de dropout (default: 0.2)
        
    Retorna:
    --------
    tf.Tensor
        Tensor transformado
    """
    # Primera GLU - usando nombres únicos con el índice del paso
    glu_out = GatedLinearUnit(feature_dim, 
                             name=f"{CONST_GATED_FEATURE_TRANSFORM}_{step_idx}_1")(inputs)
    
    # Normalización por lotes fantasma
    if use_bn:
        glu_out = GhostBatchNormalization(
            virtual_batch_size=virtual_batch_size,
            name=f"{CONST_GATED_FEATURE_TRANSFORM}_{step_idx}_bn_1"
        )(glu_out)
    
    # Procesamiento multi-cabeza
    head_outputs = []
    head_size = feature_dim // num_heads
    
    for i in range(num_heads):
        head_out = Dense(head_size, 
                        name=f"{CONST_ATTENTION}_{step_idx}_head_{i}")(glu_out)
        head_out = Activation(CONST_TANH, 
                             name=f"{CONST_ATTENTION}_{step_idx}_act_{i}")(head_out)
        head_outputs.append(head_out)
    
    # Concatenar cabezas y aplicar normalización
    multi_head = Concatenate(axis=-1, 
                            name=f"{CONST_ATTENTION}_{step_idx}_concat")(head_outputs)
    
    if use_bn:
        multi_head = GhostBatchNormalization(
            virtual_batch_size=virtual_batch_size,
            name=f"{CONST_ATTENTION}_{step_idx}_bn"
        )(multi_head)
    
    # Segunda GLU
    glu_out = GatedLinearUnit(feature_dim, 
                             name=f"{CONST_GATED_FEATURE_TRANSFORM}_{step_idx}_2")(multi_head)
    
    # Normalización final
    if use_bn:
        glu_out = GhostBatchNormalization(
            virtual_batch_size=virtual_batch_size,
            name=f"{CONST_GATED_FEATURE_TRANSFORM}_{step_idx}_bn_2"
        )(glu_out)
    
    # Dropout
    return Dropout(dropout_rate, 
                  name=f"{CONST_GATED_FEATURE_TRANSFORM}_{step_idx}_dropout")(glu_out)


def attention_block(inputs: tf.Tensor, input_dim: int, 
                   step_idx: int,
                   use_bn: bool = True, 
                   virtual_batch_size: Optional[int] = None) -> tf.Tensor:
    """
    Implementa un bloque de atención para selección de características.
    
    Parámetros:
    -----------
    inputs : tf.Tensor
        Tensor de entrada
    input_dim : int
        Dimensión de entrada
    step_idx : int
        Índice del paso actual para crear nombres únicos
    use_bn : bool, opcional
        Si usar normalización por lotes (default: True)
    virtual_batch_size : Optional[int], opcional
        Tamaño de lote virtual (default: None)
        
    Retorna:
    --------
    tf.Tensor
        Máscara de atención normalizada
    """
    attention_logits = Dense(input_dim, 
                            name=f"{CONST_FEATURE_MASK}_{step_idx}_logits")(inputs)
    
    # Normalizar la atención para suma a 1
    attention_mask = SoftmaxLayer(axis=-1, 
                                 name=f"{CONST_FEATURE_MASK}_{step_idx}_softmax")(attention_logits)
    
    return attention_mask


def _prepare_inputs(cgm_shape: Tuple[int, ...], 
                   other_features_shape: Tuple[int, ...]) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor, bool, int]:
    """
    Prepara las entradas del modelo TabNet.
    
    Parámetros:
    -----------
    cgm_shape : Tuple[int, ...]
        Forma de los datos CGM (pasos_temporales, características)
    other_features_shape : Tuple[int, ...]
        Forma de otras características
        
    Retorna:
    --------
    Tuple[tf.Tensor, tf.Tensor, tf.Tensor, bool, int]
        Tuple con (cgm_input, other_input, combined_input, has_other_features, input_dim)
    """
    # Si other_features_shape es una tupla vacía (), establecerla a (1,)
    if not other_features_shape:
        other_features_shape = (1,)
    
    # Asegurar que cgm_shape tenga al menos 2 dimensiones
    if len(cgm_shape) < 2:
        raise ValueError(f"CGM shape debe tener al menos 2 dimensiones, pero tiene {len(cgm_shape)}: {cgm_shape}")
    
    # Entradas
    cgm_input = Input(shape=cgm_shape, name='cgm_input')
    
    # Crear other_input solo si hay características adicionales
    if other_features_shape and other_features_shape[0] > 0:
        other_input = Input(shape=other_features_shape, name='other_input')
        has_other_features = True
    else:
        other_input = Input(shape=(1,), name='other_input')
        has_other_features = False
    
    # Aplanar entrada CGM
    flattened_cgm = Reshape((-1,), name='flatten_cgm')(cgm_input)
    
    # Combinar entradas
    if has_other_features:
        combined_input = Concatenate(axis=-1, name='combine_inputs')([flattened_cgm, other_input])
    else:
        combined_input = flattened_cgm
    
    # Obtener dimensión de entrada
    input_dim = combined_input.shape[-1]
    
    return cgm_input, other_input, combined_input, has_other_features, input_dim


def _process_decision_steps(x: tf.Tensor, combined_input: tf.Tensor, input_dim: int, 
                          feature_dim: int, num_decision_steps: int,
                          num_attention_heads: int, virtual_batch_size: Optional[int],
                          attention_dropout: float) -> Tuple[List[tf.Tensor], tf.Tensor]:
    """
    Procesa los pasos de decisión del modelo TabNet.
    
    Parámetros:
    -----------
    x : tf.Tensor
        Tensor de entrada procesado
    combined_input : tf.Tensor
        Entrada combinada original
    input_dim : int
        Dimensión de entrada
    feature_dim : int
        Dimensión de características
    num_decision_steps : int
        Número de pasos de decisión
    num_attention_heads : int
        Número de cabezas de atención
    virtual_batch_size : Optional[int]
        Tamaño de lote virtual
    attention_dropout : float
        Tasa de dropout para atención
        
    Retorna:
    --------
    Tuple[List[tf.Tensor], tf.Tensor]
        Tuple con (step_outputs, features)
    """
    # Salidas de los pasos de decisión para usarse en la combinación final
    step_outputs = []
    
    # Estado inicial
    features = combined_input
    
    # Pasos de decisión
    for step in range(num_decision_steps):
        # Transformador de características compartido
        transformer_output = feature_transformer(
            inputs=x,
            feature_dim=feature_dim,
            step_idx=step,
            num_heads=num_attention_heads,
            use_bn=True,
            virtual_batch_size=virtual_batch_size,
            dropout_rate=attention_dropout
        )
        
        # Bloque de atención
        attention_mask = attention_block(
            inputs=transformer_output,
            input_dim=input_dim,
            step_idx=step,
            use_bn=True,
            virtual_batch_size=virtual_batch_size
        )
        
        # Aplicar máscara y obtener features para este paso
        masked_features = Multiply(name=f"masked_features_{step}")([features, attention_mask])
        
        # Guardar salida del paso para uso en capas finales
        step_outputs.append(masked_features)
        
        # Actualizar features restantes para el siguiente paso
        if step < num_decision_steps - 1:
            features = _update_features(features, attention_mask, step)
    
    return step_outputs, features


def _update_features(features: tf.Tensor, attention_mask: tf.Tensor, step: int) -> tf.Tensor:
    """
    Actualiza las características disponibles para el siguiente paso.
    
    Parámetros:
    -----------
    features : tf.Tensor
        Características actuales
    attention_mask : tf.Tensor
        Máscara de atención
    step : int
        Índice del paso actual
        
    Retorna:
    --------
    tf.Tensor
        Características actualizadas
    """
    # Restar características usadas (escaladas por sparsity)
    sparsity_coeff = TABNET_CONFIG.get('sparsity_coefficient', 1e-4)
    relaxation_factor = TABNET_CONFIG.get('relaxation_factor', 1.5)
    
    # Calcular el factor de escala
    scale_value = sparsity_coeff * relaxation_factor
    
    # Usar capas personalizadas para evitar operaciones TF directas
    features_scaled = ScaleLayer(scale_value, name=f"scale_{step}")(attention_mask)
    ones = OnesLikeLayer(name=f"ones_{step}")(features_scaled)
    scaled_subtract = SubtractLayer(name=f"subtract_{step}")(features_scaled)
    scales = Add(name=f"add_scales_{step}")([ones, scaled_subtract])
    
    return Multiply(name=f"update_features_{step}")([features, scales])


def _combine_step_outputs(step_outputs: List[tf.Tensor], combined_input: tf.Tensor, 
                         num_decision_steps: int) -> tf.Tensor:
    """
    Combina las salidas de los pasos de decisión.
    
    Parámetros:
    -----------
    step_outputs : List[tf.Tensor]
        Lista de salidas de los pasos
    combined_input : tf.Tensor
        Entrada combinada original
    num_decision_steps : int
        Número de pasos de decisión
        
    Retorna:
    --------
    tf.Tensor
        Salida combinada
    """
    if not step_outputs:
        return combined_input
        
    # Transformar cada salida a la misma forma
    reshaped_outputs = []
    for i, output in enumerate(step_outputs):
        reshaped = Reshape((1, -1), name=f"reshape_step_{i}")(output)
        reshaped_outputs.append(reshaped)
    
    # Concatenar a lo largo del eje de pasos de decisión
    if len(reshaped_outputs) > 1:
        stacked_outputs = Concatenate(axis=1, name="stack_steps")(reshaped_outputs)
    else:
        stacked_outputs = reshaped_outputs[0]
    
    # Red para calcular pesos de atención para cada paso
    step_mean = ReduceMeanLayer(axis=2, name="step_mean")(stacked_outputs)
    step_attn = Dense(num_decision_steps, activation='softmax', name="step_attention")(step_mean)
    
    # Reshape para multiplicación por broadcasting
    step_attn_reshaped = Reshape((num_decision_steps, 1), name="reshape_attn")(step_attn)
    
    # Multiplicar directamente usando broadcasting de Keras
    weighted_outputs = Multiply(name="weight_outputs")([stacked_outputs, step_attn_reshaped])
    
    # Sumar a lo largo del eje de pasos
    return ReduceSumLayer(axis=1, name="combine_steps")(weighted_outputs)


def create_tabnet_model(cgm_shape: Tuple[int, ...], 
                       other_features_shape: Tuple[int, ...]) -> Model:
    """
    Crea un modelo TabNet para regresión.
    
    Parámetros:
    -----------
    cgm_shape : Tuple[int, ...]
        Forma de los datos CGM (pasos_temporales, características)
    other_features_shape : Tuple[int, ...]
        Forma de otras características
        
    Retorna:
    --------
    Model
        Modelo TabNet compilado
    """
    # Parámetros del modelo desde la configuración
    feature_dim = TABNET_CONFIG.get('feature_dim', 128)
    output_dim = TABNET_CONFIG.get('output_dim', 64)
    num_decision_steps = TABNET_CONFIG.get('num_decision_steps', 8)
    num_attention_heads = TABNET_CONFIG.get('num_attention_heads', 4)
    attention_dropout = TABNET_CONFIG.get('attention_dropout', 0.2)
    feature_dropout = TABNET_CONFIG.get('feature_dropout', 0.1)
    virtual_batch_size = TABNET_CONFIG.get('virtual_batch_size', 128)
    
    # Preparar entradas del modelo
    cgm_input, other_input, combined_input, _, input_dim = _prepare_inputs(
        cgm_shape, other_features_shape)
    
    # Transformación inicial
    x = Dense(feature_dim, name="initial_transform")(combined_input)
    x = LayerNormalization(name="initial_norm")(x)
    x = Dropout(feature_dropout, name="feature_dropout")(x)
    
    # Procesar pasos de decisión
    step_outputs, _ = _process_decision_steps(
        x, combined_input, input_dim, feature_dim, 
        num_decision_steps, num_attention_heads, 
        virtual_batch_size, attention_dropout
    )
    
    # Combinar salidas de los pasos
    combined = _combine_step_outputs(step_outputs, combined_input, num_decision_steps)
    
    # MLP final con conexiones residuales
    x = Dense(output_dim, activation='selu', name="final_dense_1")(combined)
    x = LayerNormalization(epsilon=1e-6, name="final_norm_1")(x)
    x = Dropout(attention_dropout, name="final_dropout")(x)
    
    # Conexión residual
    skip = x
    x = Dense(output_dim // 2, activation='selu', name="final_dense_2")(x)
    x = LayerNormalization(name="final_norm_2")(x)
    x = Dense(output_dim, activation='selu', name="final_dense_3")(x)
    
    # Sumar la conexión residual si las dimensiones coinciden
    if skip.shape[-1] == x.shape[-1]:
        x = Add(name="residual_connection")([x, skip])
    
    # Capa de salida
    output = Dense(1, name="output_layer")(x)
    
    # Construir modelo
    model = Model(inputs=[cgm_input, other_input], outputs=output, name='tabnet')
    
    # Compilar con Adam y MSE
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss='mse',
        metrics=['mae', tf.keras.metrics.RootMeanSquaredError(name='rmse')]
    )
    
    return model


def create_model_creator(cgm_shape: Tuple[int, ...], other_features_shape: Tuple[int, ...]) -> Callable[[], Model]:
    """
    Crea una función creadora de modelos compatible con DLModelWrapperTF.
    
    Parámetros:
    -----------
    cgm_shape : Tuple[int, ...]
        Forma de los datos CGM
    other_features_shape : Tuple[int, ...]
        Forma de otras características
        
    Retorna:
    --------
    Callable[[], Model]
        Función que crea un modelo TabNet sin argumentos
    """
    def model_creator() -> Model:
        """
        Crea un modelo TabNet sin argumentos.
        
        Retorna:
        --------
        Model
            Modelo TabNet compilado
        """
        return create_tabnet_model(cgm_shape, other_features_shape)
    
    return model_creator


def create_tabnet_model_wrapper(cgm_shape: Tuple[int, ...], other_features_shape: Tuple[int, ...]) -> DLModelWrapper:
    """
    Crea un modelo TabNet envuelto en DLModelWrapperTF.
    
    Parámetros:
    -----------
    cgm_shape : Tuple[int, ...]
        Forma de los datos CGM
    other_features_shape : Tuple[int, ...]
        Forma de otras características
        
    Retorna:
    --------
    DLModelWrapper
        Modelo TabNet envuelto en DLModelWrapper
    """
    # Crear una función model_creator
    model_creator_fn = create_model_creator(cgm_shape, other_features_shape)
    
    # Instanciar el wrapper específico para TensorFlow
    # El modelo será compilado en el método start() si aún no está compilado
    return DLModelWrapper(model_creator_fn, 'tensorflow')