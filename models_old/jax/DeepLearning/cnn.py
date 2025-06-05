import os, sys
import jax
import jax.numpy as jnp
import flax.linen as nn
import numpy as np
from typing import Tuple, Dict, Any, Optional, Callable
from models_old.early_stopping import get_early_stopping, get_early_stopping_config

PROJECT_ROOT = os.path.abspath(os.getcwd())
sys.path.append(PROJECT_ROOT) 

from config.models_config import CNN_CONFIG, EARLY_STOPPING_POLICY
from custom.DeepLearning.dl_model_wrapper import DLModelWrapper

# Constantes para uso repetido
CONST_RELU = "relu"
CONST_GELU = "gelu"
CONST_SWISH = "swish"
CONST_SILU = "silu"

class SqueezeExcitationBlock(nn.Module):
    """
    Bloque Squeeze-and-Excitation como módulo de Flax.
    
    Parámetros:
    -----------
    filters : int
        Número de filtros del bloque
    se_ratio : int
        Factor de reducción para la capa de squeeze
        
    Retorna:
    --------
    jnp.ndarray
        Tensor de entrada escalado por los pesos de atención
    """
    filters: int
    se_ratio: int = 16
    
    @nn.compact
    def __call__(self, inputs: jnp.ndarray) -> jnp.ndarray:
        # Squeeze - reducción por promedio global
        x = jnp.mean(inputs, axis=1)
        
        # Excitation - compresión y expansión
        x = nn.Dense(features=self.filters // self.se_ratio)(x)
        x = nn.gelu(x)
        x = nn.Dense(features=self.filters)(x)
        x = jax.nn.sigmoid(x)
        
        # Reshape para broadcasting
        x = jnp.expand_dims(x, axis=1)
        
        # Escalado
        return inputs * x

def create_residual_block(x: jnp.ndarray, filters: int, dilation_rate: int = 1) -> jnp.ndarray:
    """
    Crea un bloque residual con convoluciones dilatadas y SE.
    
    Parámetros:
    -----------
    x : jnp.ndarray
        Tensor de entrada
    filters : int
        Número de filtros para la convolución
    dilation_rate : int, opcional
        Tasa de dilatación para las convoluciones (default: 1)
        
    Retorna:
    --------
    jnp.ndarray
        Tensor procesado con conexión residual
    """
    skip = x
    
    # Camino convolucional
    x = nn.Conv(
        features=filters,
        kernel_size=(CNN_CONFIG['kernel_size'],),
        padding='SAME',
        kernel_dilation=(dilation_rate,)
    )(x)
    x = nn.LayerNorm()(x)
    x = get_activation(x, CNN_CONFIG['activation'])
    
    # Squeeze-and-Excitation
    if CNN_CONFIG['use_se_block']:
        x = SqueezeExcitationBlock(filters=filters, se_ratio=CNN_CONFIG['se_ratio'])(x)
    
    # Proyección del residual si es necesario
    if skip.shape[-1] != filters:
        skip = nn.Conv(
            features=filters,
            kernel_size=(1,),
            padding='SAME'
        )(skip)
    
    return x + skip

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
    if activation_name == CONST_RELU:
        return nn.relu(x)
    elif activation_name == CONST_GELU:
        return nn.gelu(x)
    elif activation_name == CONST_SWISH:
        return nn.swish(x)
    elif activation_name == CONST_SILU:
        return nn.silu(x)
    else:
        return nn.relu(x)  # Valor por defecto

class CNNModel(nn.Module):
    """
    Modelo CNN (Red Neuronal Convolucional) con entrada dual para datos CGM y otras características.
    
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
    
    @nn.compact
    def __call__(self, cgm_input: jnp.ndarray, other_input: jnp.ndarray, training: bool = True) -> jnp.ndarray:
        """
        Ejecuta el modelo CNN sobre las entradas.
        
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
        # Proyección inicial
        x = nn.Conv(
            features=self.config['filters'][0],
            kernel_size=(1,),
            padding='SAME'
        )(cgm_input)
        
        # Normalización por capas o por lotes
        if self.config['use_layer_norm']:
            x = nn.LayerNorm()(x)
        else:
            x = nn.BatchNorm(use_running_average=not training)(x)
        
        # Bloques residuales con diferentes tasas de dilatación
        for filters in self.config['filters']:
            for dilation_rate in self.config['dilation_rates']:
                x = create_residual_block(x, filters, dilation_rate)
            
            # MaxPooling implementado manualmente
            x = nn.max_pool(x, window_shape=(self.config['pool_size'],), strides=(self.config['pool_size'],), padding='VALID')
        
        # Pooling global con concatenación de máximo y promedio
        avg_pool = jnp.mean(x, axis=1)
        max_pool = jnp.max(x, axis=1)
        x = jnp.concatenate([avg_pool, max_pool], axis=-1)
        
        # Combinar características
        combined = jnp.concatenate([x, other_input], axis=-1)
        
        # Capas densas con conexiones residuales
        skip = combined
        dense = nn.Dense(features=256)(combined)
        dense = get_activation(dense, self.config['activation'])
        
        # Normalización por capas o por lotes
        if self.config['use_layer_norm']:
            dense = nn.LayerNorm()(dense)
        else:
            dense = nn.BatchNorm(use_running_average=not training)(dense)
            
        # Usar el método `dropout` para permitir a Flax manejar la generación PRNG internamente
        dense = nn.Dropout(
            rate=self.config['dropout_rate'], 
            deterministic=not training,
            rng_collection='dropout'
        )(dense)
        
        dense = nn.Dense(features=256)(dense)
        dense = get_activation(dense, self.config['activation'])
        
        # Conexión residual
        if skip.shape[-1] == 256:
            dense = dense + skip
        
        # Capas finales
        dense = nn.Dense(features=128)(dense)
        dense = get_activation(dense, self.config['activation'])
        
        # Normalización por capas o por lotes
        if self.config['use_layer_norm']:
            dense = nn.LayerNorm()(dense)
        else:
            dense = nn.BatchNorm(use_running_average=not training)(dense)
            
        # Usar el método `dropout` nuevamente
        dense = nn.Dropout(
            rate=self.config['dropout_rate'] / 2, 
            deterministic=not training,
            rng_collection='dropout'
        )(dense)
        
        output = nn.Dense(features=1)(dense)
        
        return output

def create_cnn_model(cgm_shape: Tuple[int, ...], other_features_shape: Tuple[int, ...]) -> DLModelWrapper:
    """
    Crea un modelo CNN (Red Neuronal Convolucional) con JAX/Flax.
    
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
    
    # Función de creación del modelo
    def model_creator():
        return CNNModel(
            config=CNN_CONFIG,
            cgm_shape=cgm_shape,
            other_features_shape=other_features_shape
        )
    
    # Crear wrapper con framework JAX
    model_wrapper = DLModelWrapper(model_creator, 'jax')
    
    # Configurar early stopping
    es_patience, es_min_delta, es_restore_best = get_early_stopping_config()
    model_wrapper.add_early_stopping(
        patience=es_patience,
        min_delta=es_min_delta,
        restore_best_weights=es_restore_best
    )
    
    return model_wrapper

def model_creator() -> Callable[[Tuple[int, ...], Tuple[int, ...]], DLModelWrapper]:
    """
    Retorna una función para crear un modelo CNN compatible con la API del sistema.
    
    Retorna:
    --------
    Callable[[Tuple[int, ...], Tuple[int, ...]], DLModelWrapper]
        Función para crear el modelo con las formas de entrada especificadas
    """
    return create_cnn_model