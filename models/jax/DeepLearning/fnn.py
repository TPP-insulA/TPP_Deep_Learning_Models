import os, sys
import jax
import jax.numpy as jnp
import flax.linen as nn
import optax
from flax.training import train_state
from typing import Tuple, Dict, List, Any, Optional, Callable, Union

PROJECT_ROOT = os.path.abspath(os.getcwd())
sys.path.append(PROJECT_ROOT) 

from config.models_config import FNN_CONFIG

def create_residual_block(x: jnp.ndarray, units: int, dropout_rate: float = 0.2, 
                         activation: str = 'relu', use_layer_norm: bool = True, 
                         training: bool = True, rng: Optional[jax.random.PRNGKey] = None) -> jnp.ndarray:
    """
    Crea un bloque residual para FNN con normalización y dropout.
    
    Parámetros:
    -----------
    x : jnp.ndarray
        Tensor de entrada
    units : int
        Número de unidades en la capa densa
    dropout_rate : float
        Tasa de dropout a aplicar
    activation : str
        Función de activación a utilizar
    use_layer_norm : bool
        Si se debe usar normalización de capa en lugar de normalización por lotes
    training : bool
        Indica si está en modo entrenamiento
    rng : jax.random.PRNGKey, opcional
        Clave para generación de números aleatorios
        
    Retorna:
    --------
    jnp.ndarray
        Salida del bloque residual
    """
    # Guarda la entrada para la conexión residual
    skip = x
    
    # Crear nuevas claves de aleatoriedad si se proporcionó una
    if rng is not None:
        dropout_rng, dropout_rng2 = jax.random.split(rng)
    else:
        dropout_rng = dropout_rng2 = None
    
    # Primera capa densa con normalización y activación
    x = nn.Dense(units)(x)
    if use_layer_norm:
        x = nn.LayerNorm(epsilon=FNN_CONFIG['epsilon'])(x)
    else:
        x = nn.BatchNorm(use_running_average=not training)(x)
    x = get_activation(x, activation)
    x = nn.Dropout(rate=dropout_rate, deterministic=not training)(x, rngs={'dropout': dropout_rng})
    
    # Segunda capa densa con normalización
    x = nn.Dense(units)(x)
    if use_layer_norm:
        x = nn.LayerNorm(epsilon=FNN_CONFIG['epsilon'])(x)
    else:
        x = nn.BatchNorm(use_running_average=not training)(x)
    
    # Proyección para la conexión residual si es necesario
    if skip.shape[-1] != units:
        skip = nn.Dense(units)(skip)
    
    # Conexión residual
    x = x + skip
    x = get_activation(x, activation)
    x = nn.Dropout(rate=dropout_rate, deterministic=not training)(x, rngs={'dropout': dropout_rng2})
    
    return x

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

class FNNModel(nn.Module):
    """
    Modelo de red neuronal feedforward (FNN) con características modernas implementado en JAX/Flax.
    
    Parámetros:
    -----------
    config : Dict
        Diccionario con la configuración del modelo
    input_shape : Tuple
        Forma del tensor de entrada principal
    other_features_shape : Tuple, opcional
        Forma de características adicionales
    """
    config: Dict
    input_shape: Tuple
    other_features_shape: Optional[Tuple] = None
    
    def _process_inputs(self, inputs):
        # Manejar entradas múltiples o única
        if self.other_features_shape is not None:
            main_input, other_input = inputs
        else:
            main_input = inputs
            other_input = None
        
        # Aplanar si es necesario (para entradas multidimensionales)
        if len(self.input_shape) > 1:
            x = jnp.reshape(main_input, (main_input.shape[0], -1))
        else:
            x = main_input
        
        # Entrada secundaria opcional
        if other_input is not None:
            x = jnp.concatenate([x, other_input], axis=-1)
            
        return x
    
    def _apply_normalization(self, x, training):
        if self.config['use_layer_norm']:
            return nn.LayerNorm(epsilon=self.config['epsilon'])(x)
        else:
            return nn.BatchNorm(use_running_average=not training)(x)
    
    def _build_output_layer(self, x):
        if self.config['regression']:
            return nn.Dense(1)(x)
        else:
            x = nn.Dense(self.config['num_classes'])(x)
            return nn.softmax(x)
    
    @nn.compact
    def __call__(self, inputs: Union[jnp.ndarray, Tuple[jnp.ndarray, jnp.ndarray]], 
                training: bool = True) -> jnp.ndarray:
        x = self._process_inputs(inputs)
        
        # Capa inicial
        x = nn.Dense(self.config['hidden_units'][0])(x)
        x = self._apply_normalization(x, training)
        x = get_activation(x, self.config['activation'])
        x = nn.Dropout(rate=self.config['dropout_rates'][0], deterministic=not training)(x)
        
        # Bloques residuales apilados
        for i, units in enumerate(self.config['hidden_units'][1:]):
            dropout_rate = self.config['dropout_rates'][min(i+1, len(self.config['dropout_rates'])-1)]
            dropout_rng = None
            if training:
                dropout_rng = self.make_rng('dropout')
            x = create_residual_block(
                x, 
                units,
                dropout_rate=dropout_rate,
                activation=self.config['activation'],
                use_layer_norm=self.config['use_layer_norm'],
                training=training,
                rng=dropout_rng
            )
        
        # Capas finales con estrechamiento progresivo
        for i, units in enumerate(self.config['final_units']):
            x = nn.Dense(units)(x)
            x = get_activation(x, self.config['activation'])
            x = self._apply_normalization(x, training)
            x = nn.Dropout(rate=self.config['final_dropout_rate'], deterministic=not training)(x)
        
        return self._build_output_layer(x)

def create_fnn_model(input_shape: tuple, 
                     other_features_shape: Optional[tuple] = None) -> FNNModel:
    """
    Crea un modelo de red neuronal feedforward (FNN) con JAX/Flax.
    
    Parámetros:
    -----------
    input_shape : tuple
        Forma del tensor de entrada principal
    other_features_shape : tuple, opcional
        Forma del tensor de características adicionales
    
    Retorna:
    --------
    fnn_model
        Modelo FNN inicializado
    """
    model = FNNModel(
        config=FNN_CONFIG,
        input_shape=input_shape,
        other_features_shape=other_features_shape
    )
    
    return model

def create_train_state(model: FNNModel, 
                       rng: jax.random.PRNGKey, 
                       input_shape: tuple,
                       other_features_shape: Optional[tuple] = None,
                       learning_rate: float = None) -> train_state.TrainState:
    """
    Crea un estado de entrenamiento para un modelo FNN.
    
    Parámetros:
    -----------
    model : fnn_model
        Modelo FNN a entrenar
    rng : jax.random.PRNGKey
        Clave para generación de números aleatorios
    input_shape : tuple
        Forma del tensor de entrada principal
    other_features_shape : tuple, opcional
        Forma del tensor de características adicionales
    learning_rate : float, opcional
        Tasa de aprendizaje para el optimizador
    
    Retorna:
    --------
    train_state.TrainState
        Estado de entrenamiento inicializado
    """
    if learning_rate is None:
        learning_rate = FNN_CONFIG['learning_rate']
    
    # Inicializar parámetros con datos ficticios
    if other_features_shape is not None:
        dummy_main = jnp.ones((1,) + input_shape)
        dummy_other = jnp.ones((1,) + other_features_shape)
        variables = model.init({'params': rng, 'dropout': rng}, (dummy_main, dummy_other), training=False)
    else:
        dummy_input = jnp.ones((1,) + input_shape)
        variables = model.init({'params': rng, 'dropout': rng}, dummy_input, training=False)
    
    # Crear optimizador
    tx = optax.adam(
        learning_rate=learning_rate,
        b1=FNN_CONFIG['beta_1'],
        b2=FNN_CONFIG['beta_2'],
        eps=FNN_CONFIG['optimizer_epsilon']
    )
    
    # Crear estado de entrenamiento
    return train_state.TrainState.create(
        apply_fn=model.apply,
        params=variables['params'],
        tx=tx
    )

def train_step(state: train_state.TrainState, 
               batch: Tuple, 
               rng: jax.random.PRNGKey, 
               loss_fn: Callable) -> Tuple[train_state.TrainState, Dict, jax.random.PRNGKey]:
    """
    Ejecuta un paso de entrenamiento para un modelo FNN.
    
    Parámetros:
    -----------
    state : train_state.TrainState
        Estado actual del entrenamiento
    batch : Tuple
        Lote de datos (x, y)
    rng : jax.random.PRNGKey
        Clave para generación de números aleatorios
    loss_fn : Callable
        Función de pérdida a utilizar
    
    Retorna:
    --------
    Tuple[train_state.TrainState, Dict, jax.random.PRNGKey]
        Nuevo estado de entrenamiento, métricas y nueva clave aleatoria
    """
    rng, dropout_rng = jax.random.split(rng)
    
    def loss_fn_with_params(params):
        logits = state.apply_fn(
            {'params': params}, 
            batch[0], 
            training=True,
            rngs={'dropout': dropout_rng}
        )
        loss = loss_fn(logits, batch[1])
        return loss, logits
    
    grad_fn = jax.value_and_grad(loss_fn_with_params, has_aux=True)
    (loss, logits), grads = grad_fn(state.params)
    
    # Actualizar parámetros
    state = state.apply_gradients(grads=grads)
    
    # Calcular métricas
    metrics = {
        'loss': loss,
    }
    
    # Añadir MAE para regresión, accuracy para clasificación
    if FNN_CONFIG['regression']:
        metrics['mae'] = jnp.mean(jnp.abs(logits - batch[1]))
    else:
        preds = jnp.argmax(logits, axis=-1)
        metrics['accuracy'] = jnp.mean(preds == batch[1])
    
    return state, metrics, rng

def eval_step(state: train_state.TrainState, 
              batch: Tuple, 
              loss_fn: Callable) -> Dict:
    """
    Ejecuta un paso de evaluación para un modelo FNN.
    
    Parámetros:
    -----------
    state : train_state.TrainState
        Estado actual del entrenamiento
    batch : Tuple
        Lote de datos (x, y)
    loss_fn : Callable
        Función de pérdida a utilizar
    
    Retorna:
    --------
    Dict
        Diccionario de métricas
    """
    logits = state.apply_fn({'params': state.params}, batch[0], training=False)
    loss = loss_fn(logits, batch[1])
    
    # Calcular métricas
    metrics = {
        'loss': loss,
    }
    
    # Añadir MAE para regresión, accuracy para clasificación
    if FNN_CONFIG['regression']:
        metrics['mae'] = jnp.mean(jnp.abs(logits - batch[1]))
    else:
        preds = jnp.argmax(logits, axis=-1)
        metrics['accuracy'] = jnp.mean(preds == batch[1])
    
    return metrics