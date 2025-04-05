import os, sys
import jax
import jax.numpy as jnp
import flax.linen as nn
from typing import Tuple, List, Dict, Any, Optional, Callable, Sequence

PROJECT_ROOT = os.path.abspath(os.getcwd())
sys.path.append(PROJECT_ROOT) 

from config.models_config import TCN_CONFIG

class WeightNormalization(nn.Module):
    """
    Normalización de pesos para capas convolucionales.
    
    Parámetros:
    -----------
    filters : int
        Número de filtros
    kernel_size : int
        Tamaño del kernel
    dilation_rate : int
        Tasa de dilatación
    padding : str
        Tipo de padding
    """
    filters: int
    kernel_size: int
    dilation_rate: int = 1
    padding: str = 'VALID'
    
    @nn.compact
    def __call__(self, inputs: jnp.ndarray) -> jnp.ndarray:
        """
        Aplica normalización de pesos a una capa convolucional.
        
        Parámetros:
        -----------
        inputs : jnp.ndarray
            Tensor de entrada
            
        Retorna:
        --------
        jnp.ndarray
            Tensor procesado
        """
        # Define kernel shape
        kernel_shape = (self.kernel_size, inputs.shape[-1], self.filters)
        
        # Create parameters for the kernel
        kernel = self.param('kernel', nn.initializers.lecun_normal(), kernel_shape)
        
        # Weight normalization: separate magnitude and direction
        g = self.param('g', nn.initializers.ones, (self.filters,))
        
        # Calculate norm along kernel dimensions
        norm = jnp.sqrt(jnp.sum(jnp.square(kernel), axis=(0, 1)))
        
        # Normalize weights
        normalized_kernel = kernel * (g / (norm + 1e-5))
        
        # Apply convolution
        return jax.lax.conv_general_dilated(
            inputs,
            normalized_kernel,
            window_strides=(1,),
            padding=[(0, 0)],
            lhs_dilation=(1,),
            rhs_dilation=(self.dilation_rate,),
            dimension_numbers=jax.lax.ConvDimensionNumbers(
                lhs_spec=(0, 2, 1),  # (batch, features, spatial)
                rhs_spec=(2, 1, 0),  # (spatial, features, filters)
                out_spec=(0, 2, 1)   # (batch, features, spatial)
            )
        )

def causal_padding(inputs: jnp.ndarray, padding_size: int) -> jnp.ndarray:
    """
    Aplica padding causal a un tensor.
    
    Parámetros:
    -----------
    inputs : jnp.ndarray
        Tensor de entrada
    padding_size : int
        Tamaño del padding
        
    Retorna:
    --------
    jnp.ndarray
        Tensor con padding aplicado
    """
    return jnp.pad(inputs, [(0, 0), (padding_size, 0), (0, 0)])

def create_tcn_block(inputs: jnp.ndarray, filters: int, kernel_size: int,
                    dilation_rate: int, dropout_rate: float, 
                    deterministic: bool = False) -> jnp.ndarray:
    """
    Crea un bloque TCN (Temporal Convolutional Network).
    
    Parámetros:
    -----------
    inputs : jnp.ndarray
        Tensor de entrada
    filters : int
        Número de filtros
    kernel_size : int
        Tamaño del kernel
    dilation_rate : int
        Tasa de dilatación
    dropout_rate : float
        Tasa de dropout
    deterministic : bool
        Indica si está en modo inferencia
    
    Retorna:
    --------
    jnp.ndarray
        Salida del bloque TCN
    """
    padding_size = (kernel_size - 1) * dilation_rate
    padded_input = causal_padding(inputs, padding_size)
    
    # Convolución con weight normalization o convolución estándar
    if TCN_CONFIG['use_weight_norm']:
        conv = WeightNormalization(
            filters=filters * 2,  # Double for gating
            kernel_size=kernel_size,
            dilation_rate=dilation_rate
        )(padded_input)
    else:
        conv = nn.Conv(
            features=filters * 2,  # Double for gating
            kernel_size=(kernel_size,),
            kernel_dilation=(dilation_rate,),
            padding='VALID'
        )(padded_input)
    
    # Gating mechanism (GLU)
    gate, linear = jnp.split(conv, 2, axis=-1)
    gate = jax.nn.sigmoid(gate)
    conv = linear * gate
    
    # Normalization
    if TCN_CONFIG['use_layer_norm']:
        conv = nn.LayerNorm(epsilon=TCN_CONFIG['epsilon'])(conv)
    else:
        conv = nn.BatchNorm(
            use_running_average=deterministic,
            momentum=0.9,
            epsilon=1e-5
        )(conv)
    
    # Spatial or regular dropout
    if TCN_CONFIG['use_spatial_dropout']:
        # Simulate spatial dropout by repeating dropout mask across time dimension
        conv_shape = conv.shape
        dropout_mask = jax.random.bernoulli(
            jax.random.PRNGKey(0), 1.0 - dropout_rate, (conv_shape[0], 1, conv_shape[2])
        )
        dropout_mask = jnp.tile(dropout_mask, (1, conv_shape[1], 1))
        scale = 1.0 / (1.0 - dropout_rate) if not deterministic else 1.0
        conv = conv * dropout_mask * scale if not deterministic else conv
    else:
        conv = nn.Dropout(rate=dropout_rate, deterministic=deterministic)(conv)
    
    # Residual connection
    if inputs.shape[-1] == filters:
        cropped_input = inputs[:, -conv.shape[1]:, :]
        if TCN_CONFIG['residual_dropout'] > 0:
            cropped_input = nn.Dropout(
                rate=TCN_CONFIG['residual_dropout'], 
                deterministic=deterministic
            )(cropped_input)
        return conv + cropped_input
    
    return conv

class TCNModel(nn.Module):
    """
    Modelo TCN completo usando JAX/Flax.
    
    Parámetros:
    -----------
    config : Dict
        Configuración del modelo
    input_shape : Tuple
        Forma de los datos CGM
    other_features_shape : Tuple
        Forma de otras características
    """
    config: Dict
    input_shape: Tuple
    other_features_shape: Tuple
    
    @nn.compact
    def __call__(self, inputs: Tuple[jnp.ndarray, jnp.ndarray], training: bool = True) -> jnp.ndarray:
        """
        Aplica el modelo TCN a las entradas.
        
        Parámetros:
        -----------
        inputs : Tuple[jnp.ndarray, jnp.ndarray]
            Tupla de (cgm_input, other_input)
        training : bool
            Indica si está en modo entrenamiento
            
        Retorna:
        --------
        jnp.ndarray
            Predicciones del modelo
        """
        cgm_input, other_input = inputs
        deterministic = not training
        
        # Proyección inicial
        x = nn.Conv(
            features=self.config['filters'][0],
            kernel_size=(1,),
            padding='SAME'
        )(cgm_input)
        
        # Bloques TCN con skip connections
        skip_connections = []
        
        for filters in self.config['filters']:
            for dilation_rate in self.config['dilations']:
                tcn_out = create_tcn_block(
                    x,
                    filters=filters,
                    kernel_size=self.config['kernel_size'],
                    dilation_rate=dilation_rate,
                    dropout_rate=self.config['dropout_rate'][0],
                    deterministic=deterministic
                )
                skip_connections.append(tcn_out)
                x = tcn_out
        
        # Combinar skip connections con normalización
        if skip_connections:
            target_len = skip_connections[-1].shape[1]
            aligned_skips = [skip[:, -target_len:, :] for skip in skip_connections]
            x = sum(aligned_skips)
            x = x / jnp.sqrt(float(len(skip_connections)))  # Scale appropriately
        
        # Global pooling con concatenación de estadísticas
        avg_pool = jnp.mean(x, axis=1)
        max_pool = jnp.max(x, axis=1)
        x = jnp.concatenate([avg_pool, max_pool], axis=-1)
        
        # Combinar con otras características
        x = jnp.concatenate([x, other_input], axis=-1)
        
        # MLP final con residual connections
        skip = x
        x = nn.Dense(128)(x)
        
        # Aplicar activación según configuración
        if self.config['activation'] == 'relu':
            x = nn.relu(x)
        elif self.config['activation'] == 'gelu':
            x = nn.gelu(x)
        else:
            x = nn.relu(x)  # Default
            
        x = nn.LayerNorm(epsilon=self.config['epsilon'])(x)
        x = nn.Dropout(rate=self.config['dropout_rate'][0], deterministic=deterministic)(x)
        x = nn.Dense(128)(x)
        
        if self.config['activation'] == 'relu':
            x = nn.relu(x)
        elif self.config['activation'] == 'gelu':
            x = nn.gelu(x)
        else:
            x = nn.relu(x)  # Default
            
        # Residual connection si las dimensiones coinciden
        if skip.shape[-1] == 128:
            x = x + skip
        
        x = nn.Dense(64)(x)
        
        if self.config['activation'] == 'relu':
            x = nn.relu(x)
        elif self.config['activation'] == 'gelu':
            x = nn.gelu(x)
        else:
            x = nn.relu(x)  # Default
            
        x = nn.LayerNorm(epsilon=self.config['epsilon'])(x)
        x = nn.Dropout(rate=self.config['dropout_rate'][1], deterministic=deterministic)(x)
        
        output = nn.Dense(1)(x)
        
        return output

def create_tcn_model(input_shape: tuple, other_features_shape: tuple) -> TCNModel:
    """
    Crea un modelo TCN completo con JAX/Flax.
    
    Parámetros:
    -----------
    input_shape : tuple
        Forma de los datos CGM (muestras, pasos_temporales, características)
    other_features_shape : tuple
        Forma de otras características (muestras, características)
        
    Retorna:
    --------
    tcn_model
        Modelo TCN inicializado
    """
    model = TCNModel(
        config=TCN_CONFIG,
        input_shape=input_shape,
        other_features_shape=other_features_shape
    )
    
    return model

def apply_activation(x: jnp.ndarray, activation_name: str) -> jnp.ndarray:
    """
    Aplica una función de activación a un tensor.
    
    Parámetros:
    -----------
    x : jnp.ndarray
        Tensor de entrada
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
    elif activation_name == 'selu':
        return nn.selu(x)
    elif activation_name == 'sigmoid':
        return jax.nn.sigmoid(x)
    elif activation_name == 'tanh':
        return jnp.tanh(x)
    else:
        return nn.relu(x)  # Default