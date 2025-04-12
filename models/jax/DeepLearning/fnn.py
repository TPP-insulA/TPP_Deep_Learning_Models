from flax.training import train_state
from typing import Tuple, Dict, List, Any, Optional, Callable, Union
import jax
import jax.numpy as jnp
import flax.linen as nn
import os
import sys

PROJECT_ROOT = os.path.abspath(os.getcwd())
sys.path.append(PROJECT_ROOT) 

from config.models_config import FNN_CONFIG, EARLY_STOPPING_POLICY
from custom.dl_model_wrapper import DLModelWrapper
from models.early_stopping import get_early_stopping_config

# Constantes para uso repetido
CONST_RELU = "relu"
CONST_GELU = "gelu"
CONST_SWISH = "swish"
CONST_SILU = "silu"
CONST_DROPOUT = "dropout"

def create_residual_block(x: jnp.ndarray, units: int, dropout_rate: float = 0.2, 
                         activation: str = CONST_RELU, use_layer_norm: bool = True, 
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
        x = nn.LayerNorm(epsilon=1e-6)(x)
    else:
        x = nn.BatchNorm(use_running_average=not training)(x)
    x = get_activation(x, activation)
    x = nn.Dropout(
            rate=dropout_rate,
            deterministic=not training
        )(x, rng=dropout_rng)
    
    # Segunda capa densa con normalización
    x = nn.Dense(units)(x)
    if use_layer_norm:
        x = nn.LayerNorm(epsilon=1e-6)(x)
    else:
        x = nn.BatchNorm(use_running_average=not training)(x)
    
    # Proyección para la conexión residual si es necesario
    if skip.shape[-1] != units:
        skip = nn.Dense(units)(skip)
    
    # Conexión residual
    x = x + skip
    x = get_activation(x, activation)
    x = nn.Dropout(
            rate=dropout_rate,
            deterministic=not training
        )(x, rng=dropout_rng2)
    
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
    
    def _process_inputs(self, inputs: Union[jnp.ndarray, Tuple[jnp.ndarray, jnp.ndarray]]) -> jnp.ndarray:
        """
        Procesa las entradas para el modelo.
        
        Parámetros:
        -----------
        inputs : Union[jnp.ndarray, Tuple[jnp.ndarray, jnp.ndarray]]
            Entradas al modelo, puede ser un tensor o una tupla (cgm_input, other_input)
            
        Retorna:
        --------
        jnp.ndarray
            Tensor de entrada procesado
        """
        if isinstance(inputs, tuple) and len(inputs) == 2:
            cgm_input, other_input = inputs
        else:
            cgm_input = inputs
            other_input = None
            
        # Aplanar si es necesario (para entradas multidimensionales)
        if len(self.input_shape) > 1:
            x = cgm_input.reshape(cgm_input.shape[0], -1)
        else:
            x = cgm_input
            
        # Combinar con otras características si están disponibles
        if other_input is not None:
            x = jnp.concatenate([x, other_input], axis=-1)
            
        return x
    
    def _apply_normalization(self, x: jnp.ndarray, training: bool) -> jnp.ndarray:
        """
        Aplica normalización dependiendo de la configuración.
        
        Parámetros:
        -----------
        x : jnp.ndarray
            Tensor para normalizar
        training : bool
            Si está en modo entrenamiento
            
        Retorna:
        --------
        jnp.ndarray
            Tensor normalizado
        """
        if self.config['use_layer_norm']:
            return nn.LayerNorm(epsilon=self.config['epsilon'])(x)
        else:
            return nn.BatchNorm(use_running_average=not training)(x)
    
    def _build_output_layer(self, x: jnp.ndarray) -> jnp.ndarray:
        """
        Construye la capa de salida según el tipo de tarea.
        
        Parámetros:
        -----------
        x : jnp.ndarray
            Tensor de entrada a la capa final
            
        Retorna:
        --------
        jnp.ndarray
            Salida del modelo
        """
        if self.config['regression']:
            return nn.Dense(1)(x)
        else:
            x = nn.Dense(self.config['num_classes'])(x)
            return nn.softmax(x)
    
    @nn.compact
    def __call__(self, cgm_input: jnp.ndarray, other_input: jnp.ndarray, training: bool = True) -> jnp.ndarray:
        """
        Ejecuta el modelo FNN sobre las entradas.
        
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
        # Procesar entradas combinadas
        inputs = (cgm_input, other_input) if self.other_features_shape is not None else cgm_input
        x = self._process_inputs(inputs)
        
        # Capa inicial
        x = nn.Dense(self.config['hidden_units'][0])(x)
        x = self._apply_normalization(x, training)
        x = get_activation(x, self.config['activation'])
        x = nn.Dropout(
                rate=self.config['dropout_rates'][0],
                deterministic=not training,
                rng_collection='dropout'
            )(x)
        
        # Bloques residuales apilados
        for i, units in enumerate(self.config['hidden_units'][1:]):
            dropout_rate = self.config['dropout_rates'][min(i+1, len(self.config['dropout_rates'])-1)]
            dropout_rng = None
            if training:
                dropout_rng = self.make_rng(CONST_DROPOUT)
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
            x = nn.Dropout(
                    rate=self.config['final_dropout_rate'],
                    deterministic=not training,
                    rng_collection='dropout'
                )(x)
        
        return self._build_output_layer(x)

def create_fnn_model(cgm_shape: Tuple[int, ...], other_features_shape: Tuple[int, ...]) -> DLModelWrapper:
    """
    Crea un modelo FNN (Red Neuronal Feedforward) con JAX/Flax envuelto en DLModelWrapper.
    
    Parámetros:
    -----------
    cgm_shape : Tuple[int, ...]
        Forma de los datos CGM (pasos_temporales, características)
    other_features_shape : Tuple[int, ...]
        Forma de otras características (características)
        
    Retorna:
    --------
    DLModelWrapper
        Modelo FNN inicializado y envuelto en DLModelWrapper
    """
    # Crear función para inicializar el modelo
    def model_creator(**kwargs):
        return FNNModel(
            config=FNN_CONFIG,
            input_shape=cgm_shape,
            other_features_shape=other_features_shape
        )
    
    # Envolver el modelo en DLModelWrapper para compatibilidad con el sistema
    wrapper = DLModelWrapper(model_creator, 'jax')
    
    # Configurar early stopping
    es_patience, es_min_delta, es_restore_best = get_early_stopping_config()
    wrapper.add_early_stopping(
        patience=es_patience,
        min_delta=es_min_delta,
        restore_best_weights=es_restore_best
    )
    
    return wrapper

def model_creator() -> Callable[[Tuple[int, ...], Tuple[int, ...]], DLModelWrapper]:
    """
    Retorna una función para crear un modelo FNN compatible con la API del sistema.
    
    Retorna:
    --------
    Callable[[Tuple[int, ...], Tuple[int, ...]], DLModelWrapper]
        Función para crear el modelo con las formas de entrada especificadas
    """
    return create_fnn_model