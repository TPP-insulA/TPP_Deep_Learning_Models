import os, sys
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input, Dense, BatchNormalization, Concatenate, Activation,
    LayerNormalization, Dropout, Add, Flatten
)
from keras.saving import register_keras_serializable
from typing import Tuple, Dict, Any, Optional, List, Union

PROJECT_ROOT = os.path.abspath(os.getcwd())
sys.path.append(PROJECT_ROOT) 

from config.models_config import TABNET_CONFIG

@register_keras_serializable()
class GLU(tf.keras.layers.Layer):
    """
    Gated Linear Unit como capa personalizada.
    
    Parámetros:
    -----------
    units : int
        Número de unidades de salida
    """
    def __init__(self, units: int, **kwargs):
        super().__init__(**kwargs)
        self.units = units
        self.dense = Dense(units * 2)

    def call(self, inputs: tf.Tensor) -> tf.Tensor:
        """
        Aplica la capa GLU a las entradas.
        
        Parámetros:
        -----------
        inputs : tf.Tensor
            Tensor de entrada
            
        Retorna:
        --------
        tf.Tensor
            Tensor procesado
        """
        x = self.dense(inputs)
        return x[:, :self.units] * tf.nn.sigmoid(x[:, self.units:])
    
    def get_config(self) -> Dict:
        config = super().get_config()
        config.update({
            "units": self.units
        })
        return config

@register_keras_serializable()
class MultiHeadFeatureAttention(tf.keras.layers.Layer):
    """
    Atención multi-cabeza para características.
    
    Parámetros:
    -----------
    num_heads : int
        Número de cabezas de atención
    key_dim : int
        Dimensión de las claves
    dropout : float
        Tasa de dropout
    """
    def __init__(self, num_heads: int, key_dim: int, dropout: float = 0.0, **kwargs):
        super().__init__(**kwargs)
        self.num_heads = num_heads
        self.key_dim = key_dim
        self.dropout = dropout
        self.attention = tf.keras.layers.MultiHeadAttention(
            num_heads=num_heads,
            key_dim=key_dim,
            dropout=dropout
        )
        self.layernorm = LayerNormalization(epsilon=1e-6)
    
    def call(self, inputs: tf.Tensor, training: Optional[bool] = None) -> tf.Tensor:
        """
        Aplica atención multi-cabeza a las entradas.
        
        Parámetros:
        -----------
        inputs : tf.Tensor
            Tensor de entrada
        training : bool, opcional
            Indica si está en modo entrenamiento
            
        Retorna:
        --------
        tf.Tensor
            Tensor procesado
        """
        attention_output = self.attention(inputs, inputs, training=training)
        return self.layernorm(inputs + attention_output)
    
    def get_config(self) -> Dict:
        config = super().get_config()
        config.update({
            "num_heads": self.num_heads,
            "key_dim": self.key_dim,
            "dropout": self.dropout
        })
        return config

@register_keras_serializable()
class EnhancedFeatureTransformer(tf.keras.layers.Layer):
    """
    Transformador de características mejorado con atención y ghost batch norm.
    
    Parámetros:
    -----------
    feature_dim : int
        Dimensión de las características
    num_heads : int
        Número de cabezas de atención
    virtual_batch_size : int
        Tamaño del batch virtual
    dropout_rate : float
        Tasa de dropout
    """
    def __init__(self, feature_dim: int, num_heads: int, 
                virtual_batch_size: int, dropout_rate: float = 0.1, **kwargs):
        super().__init__(**kwargs)
        self.feature_dim = feature_dim
        self.num_heads = num_heads
        self.virtual_batch_size = virtual_batch_size
        self.dropout_rate = dropout_rate
        
        # GLU layers
        self.glu1 = GLU(feature_dim)
        self.glu2 = GLU(feature_dim)
        
        # Attention layer
        self.attention = MultiHeadFeatureAttention(
            num_heads=num_heads,
            key_dim=feature_dim // num_heads,
            dropout=dropout_rate
        )
        
        # Ghost Batch Normalization
        self.ghost_bn1 = BatchNormalization(
            virtual_batch_size=virtual_batch_size
        )
        self.ghost_bn2 = BatchNormalization(
            virtual_batch_size=virtual_batch_size
        )
        
        self.dropout = Dropout(dropout_rate)

    def call(self, inputs: tf.Tensor, training: Optional[bool] = None) -> tf.Tensor:
        """
        Aplica transformación a las características.
        
        Parámetros:
        -----------
        inputs : tf.Tensor
            Tensor de entrada
        training : bool, opcional
            Indica si está en modo entrenamiento
            
        Retorna:
        --------
        tf.Tensor
            Tensor transformado
        """
        x = self.glu1(inputs)
        x = self.ghost_bn1(x, training=training)
        x = self.attention(x, training=training)
        x = self.glu2(x)
        x = self.ghost_bn2(x, training=training)
        return self.dropout(x, training=training)
    
    def get_config(self) -> Dict:
        config = super().get_config()
        config.update({
            "feature_dim": self.feature_dim,
            "num_heads": self.num_heads,
            "virtual_batch_size": self.virtual_batch_size,
            "dropout_rate": self.dropout_rate
        })
        return config

def custom_softmax(x: tf.Tensor, axis: int = -1) -> tf.Tensor:
    """
    Implementación de softmax con estabilidad numérica.

    Parámetros:
    -----------
    x : tf.Tensor
        Tensor de entrada
    axis : int
        Eje de normalización
    
    Retorna:
    --------
    tf.Tensor
        Tensor normalizado
    """
    exp_x = tf.exp(x - tf.reduce_max(x, axis=axis, keepdims=True))
    return exp_x / tf.reduce_sum(exp_x, axis=axis, keepdims=True)

class TabnetModel(tf.keras.Model):
    """
    Modelo TabNet personalizado con manejo de pérdidas de entropía.
    
    Parámetros:
    -----------
    cgm_shape : tuple
        Forma de los datos CGM
    other_features_shape : tuple
        Forma de otras características
    """
    def __init__(self, cgm_shape: tuple, other_features_shape: tuple, **kwargs):
        super().__init__(**kwargs)
        self.cgm_shape = cgm_shape
        self.other_shape = other_features_shape
        self.entropy_tracker = tf.keras.metrics.Mean(name='entropy_loss')
        
        # Definir capas
        self.flatten = Flatten()
        self.feature_dropout = Dropout(TABNET_CONFIG['feature_dropout'])
        self.transformers = [
            EnhancedFeatureTransformer(
                feature_dim=TABNET_CONFIG['feature_dim'],
                num_heads=TABNET_CONFIG['num_attention_heads'],
                virtual_batch_size=TABNET_CONFIG['virtual_batch_size'],
                dropout_rate=TABNET_CONFIG['attention_dropout']
            ) for _ in range(TABNET_CONFIG['num_decision_steps'])
        ]
        
        # Capas finales
        self.final_dense1 = Dense(TABNET_CONFIG['output_dim'])
        self.final_activation1 = Activation('selu')
        self.final_norm1 = LayerNormalization(epsilon=1e-6)
        self.final_dropout = Dropout(TABNET_CONFIG['attention_dropout'])
        self.final_dense2 = Dense(TABNET_CONFIG['output_dim'] // 2)
        self.final_activation2 = Activation('selu')
        self.final_norm2 = LayerNormalization()
        self.final_dense3 = Dense(TABNET_CONFIG['output_dim'])
        self.final_activation3 = Activation('selu')
        self.add_layer = Add()
        self.output_dense = Dense(1)

    def call(self, inputs: Tuple[tf.Tensor, tf.Tensor], training: Optional[bool] = None) -> tf.Tensor:
        """
        Aplica el modelo TabNet a las entradas.
        
        Parámetros:
        -----------
        inputs : Tuple[tf.Tensor, tf.Tensor]
            Tupla de (cgm_input, other_input)
        training : bool, opcional
            Indica si está en modo entrenamiento
            
        Retorna:
        --------
        tf.Tensor
            Predicciones del modelo
        """
        cgm_input, other_input = inputs
        
        # Procesamiento inicial
        x = self.flatten(cgm_input)
        x = Concatenate()([x, other_input])
        
        # Feature masking
        if training:
            feature_mask = self.feature_dropout(tf.ones_like(x), training=True)
            x = tf.multiply(x, feature_mask)
        
        # Pasos de decisión
        step_outputs = []
        entropy_loss = 0.0
        
        for transformer in self.transformers:
            step_output = transformer(x, training=training)
            
            # Feature selection
            attention_mask = Dense(x.shape[-1])(step_output)
            mask = custom_softmax(attention_mask)
            masked_x = tf.multiply(x, mask)
            
            step_outputs.append(masked_x)
            
            if training:
                # Calcular entropía
                entropy = tf.reduce_mean(tf.reduce_sum(
                    -mask * tf.math.log(mask + 1e-15), axis=1
                ), axis=0)
                entropy_loss += entropy
        
        # Combinar salidas con atención
        combined = tf.stack(step_outputs, axis=1)
        attention_weights = Dense(len(step_outputs), activation='softmax')(
            tf.reduce_mean(combined, axis=2)
        )
        x = tf.reduce_sum(
            combined * tf.expand_dims(attention_weights, -1),
            axis=1
        )
        
        # Actualizar métrica de entropía
        if training:
            entropy_loss *= TABNET_CONFIG['sparsity_coefficient']
            self.entropy_tracker.update_state(entropy_loss)
            self.add_loss(entropy_loss)
        
        # Capas finales con residual
        x = self.final_dense1(x)
        x = self.final_activation1(x)
        x = self.final_norm1(x)
        x = self.final_dropout(x, training=training)
        
        skip = x
        x = self.final_dense2(x)
        x = self.final_activation2(x)
        x = self.final_norm2(x)
        x = self.final_dense3(x)
        x = self.final_activation3(x)
        x = self.add_layer([x, skip])
        
        return self.output_dense(x)
        
    def train_step(self, data: Tuple[Tuple[tf.Tensor, tf.Tensor], tf.Tensor]) -> Dict[str, float]:
        """
        Paso de entrenamiento personalizado con manejo de pérdida de entropía.
        
        Parámetros:
        -----------
        data : Tuple[Tuple[tf.Tensor, tf.Tensor], tf.Tensor]
            Tupla de ((cgm_input, other_input), targets)
            
        Retorna:
        --------
        Dict[str, float]
            Diccionario con métricas
        """
        inputs, targets = data
        
        with tf.GradientTape() as tape:
            predictions = self(inputs, training=True)
            loss = self.compiled_loss(targets, predictions, regularization_losses=self.losses)
            
        gradients = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))
        
        # Actualizar métricas
        self.compiled_metrics.update_state(targets, predictions)
        metrics = {m.name: m.result() for m in self.metrics}
        
        # Añadir pérdida de entropía
        metrics['entropy_loss'] = self.entropy_tracker.result()
        
        return metrics

def create_tabnet_model(cgm_shape: tuple, other_features_shape: tuple) -> tf.keras.Model:
    """
    Crea un modelo TabNet mejorado.
    
    Parámetros:
    -----------
    cgm_shape : tuple
        Forma de los datos CGM (muestras, pasos_temporales, características)
    other_features_shape : tuple
        Forma de otras características (muestras, características)
        
    Retorna:
    --------
    tf.keras.Model
        Modelo TabNet compilado
    """
    model = TabnetModel(cgm_shape, other_features_shape)
    
    # Build model
    dummy_cgm = Input(shape=cgm_shape[1:])
    dummy_other = Input(shape=(other_features_shape[1],))
    model([dummy_cgm, dummy_other])
    
    return model