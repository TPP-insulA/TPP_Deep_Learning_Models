import os, sys
import jax
import jax.numpy as jnp
import flax.linen as nn
import numpy as np
from typing import Tuple, Dict, List, Any, Optional, Callable

PROJECT_ROOT = os.path.abspath(os.getcwd())
sys.path.append(PROJECT_ROOT) 

from config.models_config import WAVENET_CONFIG
from custom.dl_model_wrapper import DLModelWrapper

# Constantes para uso repetido
CONST_RELU = "relu"
CONST_GELU = "gelu"
CONST_SAME = "SAME"
CONST_VALID = "VALID"

class WavenetBlock(nn.Module):
    """
    Bloque WaveNet mejorado con activaciones gated y escalado adaptativo.
    
    Parámetros:
    -----------
    filters : int
        Número de filtros para las convoluciones
    kernel_size : int
        Tamaño del kernel convolucional
    dilation_rate : int
        Tasa de dilatación para la convolución
    dropout_rate : float
        Tasa de dropout
    """
    filters: int
    kernel_size: int
    dilation_rate: int
    dropout_rate: float
    
    @nn.compact
    def __call__(self, inputs: jnp.ndarray, deterministic: bool = False) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """
        Aplica un bloque WaveNet a la entrada.
        
        Parámetros:
        -----------
        inputs : jnp.ndarray
            Tensor de entrada
        deterministic : bool
            Indica si está en modo inferencia
            
        Retorna:
        --------
        Tuple[jnp.ndarray, jnp.ndarray]
            (salida_residual, salida_skip)
        """
        # Gated convolutions
        # Padding causal: asegura que la convolución solo ve el pasado
        pad_width = self.kernel_size + (self.kernel_size - 1) * (self.dilation_rate - 1)
        padded_inputs = jnp.pad(
            inputs, 
            [(0, 0), (pad_width, 0), (0, 0)], 
            mode='constant'
        )
        
        # Convolución para filtro
        filter_out = nn.Conv(
            features=self.filters,
            kernel_size=(self.kernel_size,),
            kernel_dilation=(self.dilation_rate,),
            padding=CONST_VALID
        )(padded_inputs)
        
        # Convolución para gate
        gate_out = nn.Conv(
            features=self.filters,
            kernel_size=(self.kernel_size,),
            kernel_dilation=(self.dilation_rate,),
            padding=CONST_VALID
        )(padded_inputs)
        
        # Normalización
        filter_out = nn.BatchNorm(
            use_running_average=deterministic,
            momentum=0.9,
            epsilon=1e-5
        )(filter_out)
        
        gate_out = nn.BatchNorm(
            use_running_average=deterministic,
            momentum=0.9,
            epsilon=1e-5
        )(gate_out)
        
        # Activación gated: tanh(filter) * sigmoid(gate)
        gated_out = jnp.tanh(filter_out) * jax.nn.sigmoid(gate_out)
        gated_out = nn.Dropout(
            rate=self.dropout_rate,
            deterministic=deterministic
        )(gated_out)
        
        # Proyección residual
        residual = nn.Conv(
            features=self.filters,
            kernel_size=(1,),
            padding=CONST_SAME
        )(inputs)
        
        # Ajuste temporal para dimensiones
        residual = residual[:, -gated_out.shape[1]:, :]
        
        # Conexión residual con escalado
        residual_scale = WAVENET_CONFIG['use_residual_scale']
        residual_out = (gated_out * residual_scale) + residual
        
        # Proyección para skip connection
        skip_out = nn.Conv(
            features=self.filters,
            kernel_size=(1,),
            padding=CONST_SAME
        )(gated_out)
        
        # Opcional: escalado de skip connection
        if WAVENET_CONFIG['use_skip_scale']:
            skip_out = skip_out * jnp.sqrt(residual_scale)
        
        return residual_out, skip_out

def create_wavenet_block(x: jnp.ndarray, filters: int, kernel_size: int, 
                        dilation_rate: int, dropout_rate: float, 
                        deterministic: bool = False) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    Crea un bloque WaveNet con conexiones residuales y skip connections.

    Parámetros:
    -----------
    x : jnp.ndarray
        Tensor de entrada
    filters : int
        Número de filtros de la capa convolucional
    kernel_size : int
        Tamaño del kernel de la capa convolucional
    dilation_rate : int
        Tasa de dilatación de la capa convolucional
    dropout_rate : float
        Tasa de dropout
    deterministic : bool
        Indica si está en modo inferencia
        
    Retorna:
    --------
    Tuple[jnp.ndarray, jnp.ndarray]
        (salida_residual, salida_skip)
    """
    # Padding causal para convolución dilatada
    pad_width = kernel_size + (kernel_size - 1) * (dilation_rate - 1)
    padded_x = jnp.pad(x, [(0, 0), (pad_width, 0), (0, 0)], mode='constant')
    
    # Convolución dilatada
    conv = nn.Conv(
        features=filters,
        kernel_size=(kernel_size,),
        kernel_dilation=(dilation_rate,),
        padding=CONST_VALID
    )(padded_x)
    
    conv = nn.BatchNorm(
        use_running_average=deterministic,
        momentum=0.9,
        epsilon=1e-5
    )(conv)
    
    conv = nn.relu(conv)
    
    conv = nn.Dropout(
        rate=dropout_rate,
        deterministic=deterministic
    )(conv)
    
    # Conexión residual con proyección 1x1 si es necesario
    if x.shape[-1] != filters:
        x = nn.Conv(
            features=filters,
            kernel_size=(1,),
            padding=CONST_SAME
        )(x)
    
    # Alinear dimensiones temporales
    x = x[:, -conv.shape[1]:, :]
    res = x + conv
    
    return res, conv

class WavenetModel(nn.Module):
    """
    Modelo WaveNet para predicción de series temporales con JAX/Flax.
    
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
    
    @nn.compact
    def __call__(self, cgm_input: jnp.ndarray, other_input: jnp.ndarray, training: bool = True) -> jnp.ndarray:
        """
        Aplica el modelo WaveNet a las entradas.
        
        Parámetros:
        -----------
        cgm_input : jnp.ndarray
            Datos CGM de entrada
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
        
        # Proyección inicial
        x = nn.Conv(
            features=self.config['filters'][0],
            kernel_size=(1,),
            padding=CONST_SAME
        )(cgm_input)
        
        # Saltar conexiones
        skip_outputs = []
        
        # WaveNet stack
        for filters in self.config['filters']:
            for dilation in self.config['dilations']:
                wavenet = WavenetBlock(
                    filters=filters,
                    kernel_size=self.config['kernel_size'],
                    dilation_rate=dilation,
                    dropout_rate=self.config['dropout_rate']
                )
                x, skip = wavenet(x, deterministic=deterministic)
                skip_outputs.append(skip)
        
        # Combinar skip connections
        if skip_outputs:
            target_len = skip_outputs[-1].shape[1]
            aligned_skips = [skip[:, -target_len:, :] for skip in skip_outputs]
            x = sum(aligned_skips) / jnp.sqrt(float(len(skip_outputs)))
        
        # Post-procesamiento
        if self.config['activation'] == CONST_RELU:
            x = nn.relu(x)
        elif self.config['activation'] == CONST_GELU:
            x = nn.gelu(x)
        else:
            x = nn.relu(x)  # Default
            
        x = nn.Conv(
            features=self.config['filters'][-1],
            kernel_size=(1,),
            padding=CONST_SAME
        )(x)
        
        if self.config['activation'] == CONST_RELU:
            x = nn.relu(x)
        elif self.config['activation'] == CONST_GELU:
            x = nn.gelu(x)
        else:
            x = nn.relu(x)  # Default
            
        x = jnp.mean(x, axis=1)  # GlobalAveragePooling1D
        
        # Combinación con otras features
        x = jnp.concatenate([x, other_input], axis=-1)
        
        # Capas densas finales con residual connections
        skip = x
        x = nn.Dense(128)(x)
        x = nn.BatchNorm(
            use_running_average=deterministic,
            momentum=0.9,
            epsilon=1e-5
        )(x)
        
        if self.config['activation'] == CONST_RELU:
            x = nn.relu(x)
        elif self.config['activation'] == CONST_GELU:
            x = nn.gelu(x)
        else:
            x = nn.relu(x)  # Default
            
        x = nn.Dropout(
            rate=self.config['dropout_rate'],
            deterministic=deterministic
        )(x)
        
        x = nn.Dense(128)(x)
        
        # Conexión residual si las dimensiones coinciden
        if skip.shape[-1] == 128:
            x = x + skip
        
        output = nn.Dense(1)(x)
        
        return output

def create_wavenet_model(cgm_shape: Tuple[int, ...], other_features_shape: Tuple[int, ...]) -> DLModelWrapper:
    """
    Crea un modelo WaveNet para predicción de series temporales con JAX/Flax envuelto en DLModelWrapper.
    
    Parámetros:
    -----------
    cgm_shape : Tuple[int, ...]
        Forma de los datos CGM (pasos_temporales, características)
    other_features_shape : Tuple[int, ...]
        Forma de otras características (características)
        
    Retorna:
    --------
    DLModelWrapper
        Modelo WaveNet inicializado y envuelto en DLModelWrapper
    """
    model = WavenetModel(
        config=WAVENET_CONFIG,
        cgm_shape=cgm_shape,
        other_features_shape=other_features_shape
    )
    
    # Envolver el modelo en DLModelWrapper para compatibilidad con el sistema
    return DLModelWrapper(lambda **kwargs: model)

def model_creator() -> Callable[[Tuple[int, ...], Tuple[int, ...]], DLModelWrapper]:
    """
    Retorna una función para crear un modelo WaveNet compatible con la API del sistema.
    
    Retorna:
    --------
    Callable[[Tuple[int, ...], Tuple[int, ...]], DLModelWrapper]
        Función para crear el modelo con las formas de entrada especificadas
    """
    return create_wavenet_model