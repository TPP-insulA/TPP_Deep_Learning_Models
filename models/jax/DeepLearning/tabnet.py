import os, sys
import jax
import jax.numpy as jnp
import flax.linen as nn
from flax.training import train_state
import numpy as np
from typing import List, Tuple, Dict, Any, Optional, Callable, Sequence

import optax

PROJECT_ROOT = os.path.abspath(os.getcwd())
sys.path.append(PROJECT_ROOT) 

from config.models_config import TABNET_CONFIG
from custom.dl_model_wrapper import DLModelWrapper

# Constantes para uso repetido
CONST_TRAINING = "training"
CONST_DROPOUT = "dropout"
CONST_PARAMS = "params"
CONST_BATCH_STATS = "batch_stats"
CONST_MEAN = "mean"
CONST_VAR = "var"
CONST_SCALE = "scale"
CONST_BIAS = "bias"

class GLU(nn.Module):
    """
    Unidad Lineal con Compuerta (Gated Linear Unit) como módulo de Flax.
    
    Parámetros:
    -----------
    units : int
        Número de unidades de salida
    """
    units: int
    
    @nn.compact
    def __call__(self, inputs: jnp.ndarray) -> jnp.ndarray:
        """
        Aplica la capa GLU a las entradas.
        
        Parámetros:
        -----------
        inputs : jnp.ndarray
            Tensor de entrada
            
        Retorna:
        --------
        jnp.ndarray
            Tensor procesado
        """
        x = nn.Dense(self.units * 2)(inputs)
        return x[:, :self.units] * jax.nn.sigmoid(x[:, self.units:])

class MultiHeadFeatureAttention(nn.Module):
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
    num_heads: int
    key_dim: int
    dropout: float = 0.0
    
    @nn.compact
    def __call__(self, inputs: jnp.ndarray, deterministic: bool = False) -> jnp.ndarray:
        """
        Aplica atención multi-cabeza a las entradas.
        
        Parámetros:
        -----------
        inputs : jnp.ndarray
            Tensor de entrada
        deterministic : bool
            Indica si está en modo inferencia
            
        Retorna:
        --------
        jnp.ndarray
            Tensor procesado
        """
        attention_output = nn.MultiHeadAttention(
            num_heads=self.num_heads,
            key_size=self.key_dim,
            dropout_rate=self.dropout,
            deterministic=deterministic
        )(inputs, inputs)
        
        return nn.LayerNorm(epsilon=1e-6)(inputs + attention_output)

class GhostBatchNorm(nn.Module):
    """
    Normalización por lotes fantasma para conjuntos de datos pequeños.
    
    Parámetros:
    -----------
    virtual_batch_size : int
        Tamaño del lote virtual
    momentum : float
        Factor de momentum
    epsilon : float
        Valor pequeño para estabilidad numérica
    """
    virtual_batch_size: int
    momentum: float = 0.9
    epsilon: float = 1e-5
    
    @nn.compact
    def __call__(self, x: jnp.ndarray, deterministic: bool = False) -> jnp.ndarray:
        """
        Aplica normalización por lotes fantasma.
        
        Parámetros:
        -----------
        x : jnp.ndarray
            Tensor de entrada
        deterministic : bool
            Indica si está en modo inferencia
            
        Retorna:
        --------
        jnp.ndarray
            Tensor normalizado
        """
        # Parámetros entrenables
        scale = self.param(CONST_SCALE, nn.initializers.ones, (x.shape[-1],))
        bias = self.param(CONST_BIAS, nn.initializers.zeros, (x.shape[-1],))
        
        # Variables de seguimiento de estadísticas
        mean_var = self.variable(
            CONST_BATCH_STATS, CONST_MEAN, 
            lambda s: jnp.zeros(s), x.shape[-1]
        )
        var_var = self.variable(
            CONST_BATCH_STATS, CONST_VAR, 
            lambda s: jnp.ones(s), x.shape[-1]
        )
        
        if deterministic:
            # Modo inferencia: usar estadísticas acumuladas
            mean = mean_var.value
            var = var_var.value
        else:
            # Modo entrenamiento: normalizar por lotes virtuales
            batch_size = x.shape[0]
            
            if self.virtual_batch_size is None or self.virtual_batch_size >= batch_size:
                # Usar BatchNorm normal si el lote virtual es mayor que el lote real
                mean = jnp.mean(x, axis=0)
                var = jnp.var(x, axis=0)
            else:
                # Dividir en lotes virtuales
                num_virtual_batches = max(batch_size // self.virtual_batch_size, 1)
                x_reshaped = x[:num_virtual_batches * self.virtual_batch_size]
                x_reshaped = x_reshaped.reshape(num_virtual_batches, self.virtual_batch_size, -1)
                
                # Calcular medias y varianzas por lote virtual
                mean = jnp.mean(x_reshaped, axis=1)  # (num_virtual_batches, features)
                var = jnp.var(x_reshaped, axis=1)    # (num_virtual_batches, features)
                
                # Promediar estadísticas entre lotes virtuales
                mean = jnp.mean(mean, axis=0)  # (features,)
                var = jnp.mean(var, axis=0)    # (features,)
                
            # Actualizar estadísticas de seguimiento
            mean_var.value = self.momentum * mean_var.value + (1 - self.momentum) * mean
            var_var.value = self.momentum * var_var.value + (1 - self.momentum) * var
            
        # Normalizar
        return scale * (x - mean) / jnp.sqrt(var + self.epsilon) + bias

class EnhancedFeatureTransformer(nn.Module):
    """
    Transformador de características mejorado con atención y normalización por lotes fantasma.
    
    Parámetros:
    -----------
    feature_dim : int
        Dimensión de las características
    num_heads : int
        Número de cabezas de atención
    virtual_batch_size : int
        Tamaño del lote virtual
    dropout_rate : float
        Tasa de dropout
    """
    feature_dim: int
    num_heads: int
    virtual_batch_size: int
    dropout_rate: float = 0.1
    
    @nn.compact
    def __call__(self, inputs: jnp.ndarray, deterministic: bool = False) -> jnp.ndarray:
        """
        Aplica transformación a las características.
        
        Parámetros:
        -----------
        inputs : jnp.ndarray
            Tensor de entrada
        deterministic : bool
            Indica si está en modo inferencia
            
        Retorna:
        --------
        jnp.ndarray
            Tensor transformado
        """
        # Capas GLU
        x = GLU(self.feature_dim)(inputs)
        x = GhostBatchNorm(self.virtual_batch_size)(x, deterministic=deterministic)
        x = MultiHeadFeatureAttention(
            num_heads=self.num_heads,
            key_dim=self.feature_dim // self.num_heads,
            dropout=self.dropout_rate
        )(x, deterministic=deterministic)
        
        x = GLU(self.feature_dim)(x)
        x = GhostBatchNorm(self.virtual_batch_size)(x, deterministic=deterministic)
        return nn.Dropout(rate=self.dropout_rate, deterministic=deterministic)(x)

def custom_softmax(x: jnp.ndarray, axis: int = -1) -> jnp.ndarray:
    """
    Implementación de softmax con estabilidad numérica.

    Parámetros:
    -----------
    x : jnp.ndarray
        Tensor de entrada
    axis : int
        Eje de normalización
    
    Retorna:
    --------
    jnp.ndarray
        Tensor normalizado
    """
    exp_x = jnp.exp(x - jnp.max(x, axis=axis, keepdims=True))
    return exp_x / jnp.sum(exp_x, axis=axis, keepdims=True)

class TabnetModel(nn.Module):
    """
    Modelo TabNet personalizado con manejo de pérdidas de entropía.
    
    Parámetros:
    -----------
    config : Dict
        Configuración del modelo
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
        Inicializa los componentes del modelo.
        """
        # Configuración de transformers
        self.transformers = [
            EnhancedFeatureTransformer(
                feature_dim=self.config['feature_dim'],
                num_heads=self.config['num_attention_heads'],
                virtual_batch_size=self.config['virtual_batch_size'],
                dropout_rate=self.config['attention_dropout']
            ) for _ in range(self.config['num_decision_steps'])
        ]
        
        # Capas finales
        self.final_dense1 = nn.Dense(self.config['output_dim'])
        self.final_norm1 = nn.LayerNorm(epsilon=1e-6)
        self.final_dense2 = nn.Dense(self.config['output_dim'] // 2)
        self.final_norm2 = nn.LayerNorm()
        self.final_dense3 = nn.Dense(self.config['output_dim'])
        self.output_layer = nn.Dense(1)
    
    @nn.compact
    def __call__(self, cgm_input: jnp.ndarray, other_input: jnp.ndarray, training: bool = True) -> jnp.ndarray:
        """
        Aplica el modelo TabNet a las entradas.
        
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
        
        # Procesamiento inicial
        x = jnp.reshape(cgm_input, (cgm_input.shape[0], -1))  # Flatten
        x = jnp.concatenate([x, other_input], axis=-1)
        
        # Feature masking en entrenamiento
        if training:
            feature_mask = jax.random.bernoulli(
                self.make_rng(CONST_DROPOUT),
                1.0 - self.config['feature_dropout'],
                shape=x.shape
            )
            x = x * feature_mask
        
        # Pasos de decisión
        step_outputs = []
        
        for transformer in self.transformers:
            step_output = transformer(x, deterministic=deterministic)
            
            # Feature selection
            attention_mask = nn.Dense(x.shape[-1])(step_output)
            mask = custom_softmax(attention_mask)
            masked_x = x * mask
            
            step_outputs.append(masked_x)
            
            if training:
                # Calcular entropía - No acumulamos ya que no se usa actualmente
                _ = jnp.mean(jnp.sum(
                    -mask * jnp.log(mask + 1e-15), axis=1
                ))
        
        # Combinar salidas con atención
        combined = jnp.stack(step_outputs, axis=1)
        attention_weights = nn.softmax(nn.Dense(len(step_outputs))(
            jnp.mean(combined, axis=2)
        ))
        x = jnp.sum(
            combined * jnp.expand_dims(attention_weights, -1),
            axis=1
        )
        
        # Calcular pérdida de entropía total
        # Capas finales con residual
        x = self.final_dense1(x)
        x = nn.selu(x)  # Activación SELU
        x = self.final_norm1(x)
        x = nn.Dropout(rate=self.config['attention_dropout'], deterministic=deterministic)(x)
        
        skip = x
        x = self.final_dense2(x)
        x = nn.selu(x)
        x = self.final_norm2(x)
        x = self.final_dense3(x)
        x = nn.selu(x)
        x = x + skip  # Conexión residual
        
        output = self.output_layer(x)
        
        return output

def create_tabnet_model(cgm_shape: Tuple[int, ...], other_features_shape: Tuple[int, ...]) -> DLModelWrapper:
    """
    Crea un modelo TabNet mejorado con JAX/Flax envuelto en DLModelWrapper.
    
    Parámetros:
    -----------
    cgm_shape : Tuple[int, ...]
        Forma de los datos CGM (pasos_temporales, características)
    other_features_shape : Tuple[int, ...]
        Forma de otras características (características)
        
    Retorna:
    --------
    DLModelWrapper
        Modelo TabNet inicializado y envuelto en DLModelWrapper
    """
    model = TabnetModel(
        config=TABNET_CONFIG,
        cgm_shape=cgm_shape,
        other_features_shape=other_features_shape
    )
    
    # Envolver el modelo en DLModelWrapper para compatibilidad con el sistema
    return DLModelWrapper(lambda **kwargs: model)

def model_creator() -> Callable[[Tuple[int, ...], Tuple[int, ...]], DLModelWrapper]:
    """
    Retorna una función para crear un modelo TabNet compatible con la API del sistema.
    
    Retorna:
    --------
    Callable[[Tuple[int, ...], Tuple[int, ...]], DLModelWrapper]
        Función para crear el modelo con las formas de entrada especificadas
    """
    return create_tabnet_model