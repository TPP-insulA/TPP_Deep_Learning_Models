import os, sys
import jax
import jax.numpy as jnp
import flax.linen as nn
import numpy as np
from typing import Tuple, Dict, Any, Optional, Callable, List, Union
from functools import partial

PROJECT_ROOT = os.path.abspath(os.getcwd())
sys.path.append(PROJECT_ROOT) 

from config.models_config import RNN_CONFIG
from custom.DeepLearning.dl_model_wrapper import DLModelWrapper
from models.early_stopping import get_early_stopping_config

# Constantes para uso repetido
CONST_RELU = "relu"
CONST_TANH = "tanh"
CONST_SIGMOID = "sigmoid"
CONST_SWISH = "swish"

class TimeDistributed(nn.Module):
    """
    Aplica una capa a cada paso temporal de forma independiente.
    
    Parámetros:
    -----------
    module : nn.Module
        Módulo de Flax a aplicar a cada paso temporal
    """
    module: nn.Module
    
    @nn.compact
    def __call__(self, inputs: jnp.ndarray) -> jnp.ndarray:
        """
        Aplica el módulo a cada paso temporal.
        
        Parámetros:
        -----------
        inputs : jnp.ndarray
            Tensor de entrada [batch, tiempo, características]
            
        Retorna:
        --------
        jnp.ndarray
            Tensor procesado [batch, tiempo, características_salida]
        """
        batch_size, time_steps, features = inputs.shape
        reshaped_inputs = inputs.reshape(batch_size * time_steps, features)
        outputs = self.module(reshaped_inputs)
        return outputs.reshape(batch_size, time_steps, -1)

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
    elif activation_name == CONST_TANH:
        return nn.tanh(x)
    elif activation_name == CONST_SIGMOID:
        return jax.nn.sigmoid(x)
    elif activation_name == CONST_SWISH:
        return nn.swish(x)
    else:
        return nn.relu(x)  # Valor por defecto

def get_activation_fn(activation_name: str) -> Callable:
    """
    Obtiene la función de activación correspondiente al nombre.
    
    Parámetros:
    -----------
    activation_name : str
        Nombre de la función de activación
        
    Retorna:
    --------
    Callable
        Función de activación
    """
    if activation_name == CONST_RELU:
        return nn.relu
    elif activation_name == CONST_TANH:
        return nn.tanh
    elif activation_name == CONST_SIGMOID:
        return jax.nn.sigmoid
    elif activation_name == CONST_SWISH:
        return nn.swish
    else:
        return nn.relu  # Valor por defecto

class SimpleScanRNN(nn.Module):
    """
    Implementación simple y robusta de RNN utilizando directamente scan.
    
    Parámetros:
    -----------
    features : int
        Número de unidades en la capa RNN
    return_sequences : bool
        Si es True, devuelve la secuencia completa de salidas
    bidirectional : bool
        Si es True, procesa la secuencia en ambas direcciones
    activation : Callable
        Función de activación
    """
    features: int
    return_sequences: bool = False
    bidirectional: bool = False
    activation: Callable = nn.tanh
    
    @nn.compact
    def __call__(self, inputs: jnp.ndarray, deterministic: bool = True) -> jnp.ndarray:
        """
        Procesa una secuencia completa.
        
        Parámetros:
        -----------
        inputs : jnp.ndarray
            Tensor de entrada [batch, tiempo, características]
        deterministic : bool
            Si es True, se deshabilita el dropout
            
        Retorna:
        --------
        jnp.ndarray
            Salidas de la capa RNN
        """
        batch_size, _, input_dim = inputs.shape
        
        # Definir parámetros de manera explícita
        w_ih = self.param('w_ih', 
                         nn.initializers.glorot_uniform(), 
                         (input_dim, self.features))
        w_hh = self.param('w_hh', 
                         nn.initializers.orthogonal(), 
                         (self.features, self.features))
        bias = self.param('bias', 
                         nn.initializers.zeros, 
                         (self.features,))
        
        # Definir función pura para scan que solo usa parámetros de entrada
        def pure_cell_fn(carry: jnp.ndarray, x: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
            """
            Función pura de la celda RNN.
            
            Parámetros:
            -----------
            carry : jnp.ndarray
                Estado oculto anterior
            x : jnp.ndarray
                Entrada para el paso actual
                
            Retorna:
            --------
            Tuple[jnp.ndarray, jnp.ndarray]
                (Nuevo estado oculto, Salida)
            """
            h = carry
            h_next = self.activation(
                jnp.dot(x, w_ih) + 
                jnp.dot(h, w_hh) + 
                bias
            )
            return h_next, h_next
        
        # Transponer entrada para scan [tiempo, batch, características]
        inputs_t = jnp.swapaxes(inputs, 0, 1)
        
        # Estado inicial en ceros
        init_h = jnp.zeros((batch_size, self.features))
        
        # Forward scan
        _, outputs_forward = jax.lax.scan(
            pure_cell_fn, 
            init_h, 
            inputs_t
        )
        
        # Procesar bidireccionalidad si es necesario
        if self.bidirectional:
            # Invertir secuencia para backward pass
            inputs_reversed = jnp.flip(inputs_t, axis=0)
            _, outputs_backward = jax.lax.scan(
                pure_cell_fn, 
                init_h, 
                inputs_reversed
            )
            # Re-invertir salidas backward y concatenar con forward
            outputs_backward = jnp.flip(outputs_backward, axis=0)
            outputs = jnp.concatenate([outputs_forward, outputs_backward], axis=-1)
        else:
            outputs = outputs_forward
        
        # Transponer de vuelta a [batch, tiempo, características]
        outputs = jnp.swapaxes(outputs, 0, 1)
        
        # Retornar secuencias o solo último estado
        if self.return_sequences:
            return outputs
        else:
            return outputs[:, -1, :]

class RNNModel(nn.Module):
    """
    Modelo RNN completo con arquitectura personalizable.
    
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
        Ejecuta el modelo RNN sobre las entradas.
        
        Parámetros:
        -----------
        cgm_input : jnp.ndarray
            Datos de entrada CGM [batch, tiempo, características]
        other_input : jnp.ndarray
            Otras características [batch, características]
        training : bool
            Si es True, el modelo está en modo entrenamiento
            
        Retorna:
        --------
        jnp.ndarray
            Predicciones del modelo
        """
        deterministic = not training
        
        # Aplicar TimeDistributed si está configurado
        if self.config['use_time_distributed']:
            x = TimeDistributed(nn.Dense(32))(cgm_input)
            x = get_activation(x, self.config['activation'])
            x = nn.LayerNorm(epsilon=self.config['epsilon'])(x)
        else:
            x = cgm_input
        
        # Capas RNN apiladas
        for i, units in enumerate(self.config['hidden_units']):
            is_last_layer = i == len(self.config['hidden_units']) - 1
            return_sequences = not is_last_layer
            
            # Usar la implementación puramente funcional de RNN
            x = SimpleScanRNN(
                features=units,
                return_sequences=return_sequences,
                bidirectional=self.config['bidirectional'],
                activation=get_activation_fn(self.config['activation'])
            )(x, deterministic=deterministic)
            
            # Normalización para capas intermedias
            if return_sequences:
                x = nn.LayerNorm(epsilon=self.config['epsilon'])(x)
                # Aplicar dropout si está en modo entrenamiento
                if not deterministic:
                    x = nn.Dropout(rate=self.config['dropout_rate'])(
                        x, deterministic=deterministic
                    )
        
        # Combinar con otras características
        x = jnp.concatenate([x, other_input], axis=-1)
        
        # Capas densas finales
        x = nn.Dense(64)(x)
        x = get_activation(x, self.config['activation'])
        x = nn.LayerNorm(epsilon=self.config['epsilon'])(x)
        x = nn.Dropout(rate=self.config['dropout_rate'])(x, deterministic=deterministic)
        
        x = nn.Dense(32)(x)
        x = get_activation(x, self.config['activation'])
        
        # Capa de salida
        x = nn.Dense(1)(x)
        return x

def create_rnn_model(cgm_shape: Tuple[int, ...], other_features_shape: Tuple[int, ...]) -> DLModelWrapper:
    """
    Crea un modelo RNN con JAX/Flax.
    
    Parámetros:
    -----------
    cgm_shape : Tuple[int, ...]
        Forma de los datos CGM (pasos_temporales, características)
    other_features_shape : Tuple[int, ...]
        Forma de otras características (características)
        
    Retorna:
    --------
    DLModelWrapper
        Modelo RNN envuelto en DLModelWrapper
    """
    # Función creadora de modelo para pasar al wrapper
    def model_creator():
        return RNNModel(
            config=RNN_CONFIG,
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
    Retorna una función para crear un modelo RNN compatible con la API del sistema.
    
    Retorna:
    --------
    Callable[[Tuple[int, ...], Tuple[int, ...]], DLModelWrapper]
        Función para crear el modelo con las formas de entrada especificadas
    """
    return create_rnn_model