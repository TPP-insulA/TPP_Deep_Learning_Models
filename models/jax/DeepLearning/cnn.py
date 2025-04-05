import os, sys
import jax
import jax.numpy as jnp
import flax.linen as nn
from typing import Tuple, Dict, Any, Optional, Callable

PROJECT_ROOT = os.path.abspath(os.getcwd())
sys.path.append(PROJECT_ROOT) 

from config.models_config import CNN_CONFIG

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
    dilation_rate : int
        Tasa de dilatación para las convoluciones
        
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
    if activation_name == 'relu':
        return nn.relu(x)
    elif activation_name == 'gelu':
        return nn.gelu(x)
    elif activation_name == 'swish':
        return nn.swish(x)
    elif activation_name == 'silu':
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
    def __call__(self, inputs: Tuple[jnp.ndarray, jnp.ndarray], training: bool = True) -> jnp.ndarray:
        cgm_input, other_input = inputs
        
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
            
        dense = nn.Dropout(rate=self.config['dropout_rate'], deterministic=not training)(dense)
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
            
        dense = nn.Dropout(rate=self.config['dropout_rate'] / 2, deterministic=not training)(dense)
        
        output = nn.Dense(features=1)(dense)
        
        return output

def create_cnn_model(cgm_shape: tuple, other_features_shape: tuple) -> CNNModel:
    """
    Crea un modelo CNN (Red Neuronal Convolucional) con JAX/Flax.
    
    Parámetros:
    -----------
    cgm_shape : tuple
        Forma de los datos CGM (muestras, pasos_temporales, características)
    other_features_shape : tuple
        Forma de otras características (muestras, características)
        
    Retorna:
    --------
    cnn_model
        Modelo Flax inicializado
    """
    model = CNNModel(
        config=CNN_CONFIG,
        cgm_shape=cgm_shape,
        other_features_shape=other_features_shape
    )
    
    return model