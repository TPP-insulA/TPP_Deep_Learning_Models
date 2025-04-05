import os, sys
import jax
import jax.numpy as jnp
import flax.linen as nn
from typing import List, Tuple, Dict, Any, Optional, Callable

PROJECT_ROOT = os.path.abspath(os.getcwd())
sys.path.append(PROJECT_ROOT) 

from config.models_config import RNN_CONFIG

class TimeDistributed(nn.Module):
    """
    Aplica una capa a cada paso temporal de forma independiente.
    
    Parámetros:
    -----------
    module : nn.Module
        Módulo de Flax a aplicar a cada paso temporal
    """
    module: nn.Module
    
    def __call__(self, inputs: jnp.ndarray, *args, **kwargs) -> jnp.ndarray:
        """
        Aplica el módulo a cada paso temporal.
        
        Parámetros:
        -----------
        inputs : jnp.ndarray
            Tensor de entrada de forma [batch, time, features]
        
        Retorna:
        --------
        jnp.ndarray
            Tensor procesado con la misma forma temporal
        """
        batch_size, time_steps, features = inputs.shape
        
        # Reshape para aplicar la capa a cada paso temporal
        reshaped_inputs = jnp.reshape(inputs, [batch_size * time_steps, features])
        
        # Aplicar la capa
        outputs = self.module(reshaped_inputs, *args, **kwargs)
        
        # Reshape de vuelta
        output_features = outputs.shape[-1]
        return jnp.reshape(outputs, [batch_size, time_steps, output_features])

class SimpleRNNCell(nn.Module):
    """
    Implementación de una celda RNN simple.
    
    Parámetros:
    -----------
    units : int
        Número de unidades
    activation : Callable
        Función de activación
    kernel_init : Callable
        Inicializador para los pesos
    recurrent_init : Callable
        Inicializador para los pesos recurrentes
    bias_init : Callable
        Inicializador para los sesgos
    """
    units: int
    activation: Callable = nn.tanh
    kernel_init: Callable = nn.initializers.glorot_uniform()
    recurrent_init: Callable = nn.initializers.orthogonal()
    bias_init: Callable = nn.initializers.zeros
    
    @nn.compact
    def __call__(self, h: jnp.ndarray, x: jnp.ndarray, 
                dropout_rate: float = 0.0, 
                recurrent_dropout_rate: float = 0.0,
                deterministic: bool = True) -> jnp.ndarray:
        """
        Aplica un paso de la celda RNN simple.
        
        Parámetros:
        -----------
        h : jnp.ndarray
            Estado oculto previo
        x : jnp.ndarray
            Entrada actual
        dropout_rate : float
            Tasa de dropout para la entrada
        recurrent_dropout_rate : float
            Tasa de dropout para la conexión recurrente
        deterministic : bool
            Si es True, no se aplica dropout
            
        Retorna:
        --------
        jnp.ndarray
            Nuevo estado oculto
        """
        # Dropout para la entrada
        if dropout_rate > 0 and not deterministic:
            x = nn.Dropout(rate=dropout_rate, deterministic=False)(x)
        
        # Dropout para la conexión recurrente
        if recurrent_dropout_rate > 0 and not deterministic:
            h = nn.Dropout(rate=recurrent_dropout_rate, deterministic=False)(h)
        
        # Proyecciones
        i2h = nn.Dense(self.units, kernel_init=self.kernel_init, bias_init=self.bias_init)(x)
        h2h = nn.Dense(self.units, kernel_init=self.recurrent_init, use_bias=False)(h)
        
        # Actualizar estado oculto
        h_new = self.activation(i2h + h2h)
        
        return h_new

def bidirectional_rnn(forward_cell: Callable, backward_cell: Callable, inputs: jnp.ndarray, 
                     initial_state: jnp.ndarray, reverse_initial_state: jnp.ndarray, 
                     dropout_rate: float = 0.0, recurrent_dropout_rate: float = 0.0,
                     deterministic: bool = True) -> jnp.ndarray:
    """
    Implementación de RNN bidireccional usando scan.
    
    Parámetros:
    -----------
    forward_cell : Callable
        Función de celda RNN para dirección forward
    backward_cell : Callable
        Función de celda RNN para dirección backward
    inputs : jnp.ndarray
        Tensor de entrada [tiempo, batch, características]
    initial_state : jnp.ndarray
        Estado inicial forward
    reverse_initial_state : jnp.ndarray
        Estado inicial backward
    dropout_rate : float
        Tasa de dropout
    recurrent_dropout_rate : float
        Tasa de dropout recurrente
    deterministic : bool
        Indica si está en modo inferencia
        
    Retorna:
    --------
    jnp.ndarray
        Salidas concatenadas de ambas direcciones [batch, tiempo, features*2]
    """
    # Forward pass
    def forward_scan_fn(h, x):
        h_next = forward_cell(h, x, dropout_rate, recurrent_dropout_rate, deterministic)
        return h_next, h_next
    
    _, forward_outputs = jax.lax.scan(
        forward_scan_fn,
        initial_state,
        inputs
    )
    
    # Backward pass (invertir secuencia)
    reversed_inputs = jnp.flip(inputs, axis=0)
    
    def backward_scan_fn(h, x):
        h_next = backward_cell(h, x, dropout_rate, recurrent_dropout_rate, deterministic)
        return h_next, h_next
    
    _, backward_outputs = jax.lax.scan(
        backward_scan_fn,
        reverse_initial_state,
        reversed_inputs
    )
    
    # Reordenar salidas backward
    backward_outputs = jnp.flip(backward_outputs, axis=0)
    
    # Concatenar outputs [tiempo, batch, características*2]
    outputs = jnp.concatenate([forward_outputs, backward_outputs], axis=-1)
    
    # Transponer a [batch, tiempo, características*2]
    outputs = jnp.transpose(outputs, (1, 0, 2))
    
    return outputs

def apply_rnn(cell_fn: Callable, inputs: jnp.ndarray, units: int, return_sequences: bool = False,
             dropout_rate: float = 0.0, recurrent_dropout_rate: float = 0.0, 
             bidirectional: bool = False, deterministic: bool = True) -> jnp.ndarray:
    """
    Aplica una capa RNN a un tensor de entrada.
    
    Parámetros:
    -----------
    cell_fn : Callable
        Función que crea la celda RNN
    inputs : jnp.ndarray
        Tensor de entrada [batch, tiempo, características]
    units : int
        Unidades de la RNN
    return_sequences : bool
        Si es True, devuelve toda la secuencia, si no solo el último estado
    dropout_rate : float
        Tasa de dropout
    recurrent_dropout_rate : float
        Tasa de dropout recurrente
    bidirectional : bool
        Si es True, aplica RNN bidireccional
    deterministic : bool
        Indica si está en modo inferencia
        
    Retorna:
    --------
    jnp.ndarray
        Salida procesada por la RNN
    """
    batch_size, _, _ = inputs.shape
    
    # Transponer para formato [tiempo, batch, características]
    inputs_time_major = jnp.transpose(inputs, (1, 0, 2))
    
    if not bidirectional:
        # RNN unidireccional
        def scan_fn(h, x):
            h_next = cell_fn()(h, x, dropout_rate, recurrent_dropout_rate, deterministic)
            return h_next, h_next
        
        # Estado inicial
        init_h = jnp.zeros((batch_size, units))
        
        # Aplicar scan para recorrer la secuencia
        final_state, outputs = jax.lax.scan(
            scan_fn,
            init_h,
            inputs_time_major
        )
        
        # Transponer de vuelta a [batch, tiempo, características]
        outputs = jnp.transpose(outputs, (1, 0, 2))
        
        # Devolver secuencia completa o solo estado final
        if return_sequences:
            return outputs
        else:
            return final_state
    else:
        # RNN bidireccional
        forward_cell = cell_fn()
        backward_cell = cell_fn()
        
        # Estados iniciales
        init_h_forward = jnp.zeros((batch_size, units))
        init_h_backward = jnp.zeros((batch_size, units))
        
        # Aplicar RNN bidireccional
        outputs = bidirectional_rnn(
            forward_cell, backward_cell,
            inputs_time_major, init_h_forward, init_h_backward,
            dropout_rate, recurrent_dropout_rate, deterministic
        )
        
        # Devolver secuencia completa o solo estados finales
        if return_sequences:
            return outputs
        else:
            # Para bidireccional, concatenar último estado forward y primer estado backward
            last_forward = outputs[:, -1, :units]
            first_backward = outputs[:, 0, units:]
            return jnp.concatenate([last_forward, first_backward], axis=-1)

class RNNModel(nn.Module):
    """
    Modelo RNN optimizado para velocidad con procesamiento temporal distribuido.
    
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
    def __call__(self, inputs: Tuple[jnp.ndarray, jnp.ndarray], training: bool = True) -> jnp.ndarray:
        """
        Aplica el modelo RNN a las entradas.
        
        Parámetros:
        -----------
        inputs : Tuple[jnp.ndarray, jnp.ndarray]
            Tupla de (cgm_input, other_input)
        training : bool
            Indica si está en modo entrenamiento
            
        Retorna:
        --------
        jnp.ndarray
            Predicción del modelo
        """
        cgm_input, other_input = inputs
        deterministic = not training
        
        # Procesamiento temporal distribuido inicial
        if self.config['use_time_distributed']:
            x = TimeDistributed(nn.Dense(32))(cgm_input)
            x = get_activation(x, self.config['activation'])
            x = TimeDistributed(nn.BatchNorm(
                epsilon=self.config['epsilon'],
                momentum=0.9,
                use_running_average=deterministic
            ))(x)
        else:
            x = cgm_input
        
        # Reducir secuencia temporal para procesamiento más rápido
        x = nn.max_pool(x, window_shape=(1, 2, 1), strides=(1, 2, 1))
        
        # Remodelar después de max_pool que devuelve un tensor 5D
        batch_size, reduced_seq_len, features = x.shape[0], x.shape[2], x.shape[3]
        x = jnp.reshape(x, (batch_size, reduced_seq_len, features))
        
        # Capas RNN con menos unidades pero bidireccionales
        for units in self.config['hidden_units']:
            # Crear la celda RNN
            def create_rnn_cell(units=units):
                return lambda h, x, dr, rdr, det: SimpleRNNCell(
                    units=units,
                    activation=get_activation_fn(self.config['activation'])
                )(h, x, dr, rdr, det)
            
            # Aplicar RNN
            x = apply_rnn(
                create_rnn_cell,
                x,
                units=units,
                return_sequences=True,
                dropout_rate=self.config['dropout_rate'],
                recurrent_dropout_rate=self.config['recurrent_dropout'],
                bidirectional=self.config['bidirectional'],
                deterministic=deterministic
            )
            
            # Normalización por lotes después de cada RNN
            x = nn.BatchNorm(
                epsilon=self.config['epsilon'],
                momentum=0.9,  # Aumentar momentum para actualización más rápida
                use_running_average=deterministic
            )(x)
        
        # Último RNN sin return_sequences
        def create_final_rnn_cell(units=self.config['hidden_units'][-1]):
            return lambda h, x, dr, rdr, det: SimpleRNNCell(
                units=units,
                activation=get_activation_fn(self.config['activation'])
            )(h, x, dr, rdr, det)
        
        x = apply_rnn(
            create_final_rnn_cell,
            x,
            units=self.config['hidden_units'][-1],
            return_sequences=False,
            dropout_rate=self.config['dropout_rate'],
            recurrent_dropout_rate=self.config['recurrent_dropout'],
            bidirectional=self.config['bidirectional'],
            deterministic=deterministic
        )
        
        # Combinar características
        x = jnp.concatenate([x, other_input], axis=-1)
        
        # Reducir capas densas
        x = nn.Dense(32)(x)
        x = get_activation(x, self.config['activation'])
        x = nn.BatchNorm(
            epsilon=self.config['epsilon'],
            use_running_average=deterministic
        )(x)
        x = nn.Dropout(rate=self.config['dropout_rate'], deterministic=deterministic)(x)
        
        # Capa de salida
        output = nn.Dense(1)(x)
        
        return output

def get_activation(x: jnp.ndarray, activation_name: str) -> jnp.ndarray:
    """
    Aplica la función de activación al tensor.
    
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
    return get_activation_fn(activation_name)(x)

def get_activation_fn(activation_name: str) -> Callable:
    """
    Obtiene la función de activación según su nombre.
    
    Parámetros:
    -----------
    activation_name : str
        Nombre de la función de activación
        
    Retorna:
    --------
    Callable
        Función de activación
    """
    if activation_name == 'relu':
        return jax.nn.relu
    elif activation_name == 'tanh':
        return jax.nn.tanh
    elif activation_name == 'sigmoid':
        return jax.nn.sigmoid
    elif activation_name == 'swish':
        return jax.nn.swish
    else:
        return jax.nn.relu  # Por defecto

def create_rnn_model(cgm_shape: tuple, other_features_shape: tuple) -> RNNModel:
    """
    Crea un modelo RNN optimizado para velocidad con procesamiento temporal distribuido con JAX/Flax.
    
    Parámetros:
    -----------
    cgm_shape : tuple
        Forma de los datos CGM (samples, timesteps, features)
    other_features_shape : tuple
        Forma de otras características (samples, features)
        
    Retorna:
    --------
    rnn_model
        Modelo RNN inicializado
    """
    model = RNNModel(
        config=RNN_CONFIG,
        cgm_shape=cgm_shape,
        other_features_shape=other_features_shape
    )
    
    return model