import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import jax
import jax.numpy as jnp
from flax.training import train_state
from flax.training.checkpoints import save_checkpoint, restore_checkpoint
import pickle
import tensorflow as tf
import time
from tqdm.auto import tqdm
from typing import Dict, List, Tuple, Any, Optional, Callable, Union

from custom.ReinforcementLearning.rl_jax import RLModelWrapperJAX
from custom.ReinforcementLearning.rl_pt import RLModelWrapperPyTorch
from custom.ReinforcementLearning.rl_tf import RLModelWrapperTF

from config.models_config_old import EARLY_STOPPING_POLICY
from constants.constants import CONST_DEFAULT_SEED, CONST_DEFAULT_EPOCHS, CONST_DEFAULT_BATCH_SIZE
from custom.model_wrapper import ModelWrapper
from custom.printer import print_debug, print_info, print_log, print_success, print_error, print_warning

# Constantes para mensajes de error y campos comunes
CONST_MODEL_INIT_ERROR = "El modelo debe ser inicializado antes de {}"
CONST_DEVICE = "device"
CONST_LOSS = "loss"
CONST_VAL_LOSS = "val_loss"

# Clase principal que selecciona el wrapper adecuado según el framework
class RLModelWrapper(ModelWrapper):
    """
    Wrapper para modelos de aprendizaje por refuerzo que selecciona el wrapper adecuado según el framework.

    Parámetros:
    -----------
    model_creator_func : Callable
        Función que crea la instancia del agente RL (ej. create_monte_carlo_agent para JAX,
        o la clase del modelo para TF). Debe aceptar cgm_shape y other_features_shape.
    framework : str, opcional
        Framework a utilizar ('jax' o 'tensorflow') (default: 'jax').
    cgm_shape : Tuple[int, ...]
        Forma de los datos CGM (necesario para JAX wrapper).
    other_features_shape : Tuple[int, ...]
        Forma de otras características (necesario para JAX wrapper).
    **model_kwargs
        Argumentos adicionales para el creador del agente/modelo.
    """

    def __init__(self, model_creator_func: Callable, framework: str = 'jax', cgm_shape: Optional[Tuple[int,...]]=None, other_features_shape: Optional[Tuple[int,...]]=None, **model_kwargs) -> None:
        """
        Inicializa el wrapper con un creador de modelo.
        
        Parámetros:
        -----------
        model_creator_func : Callable
            Función que crea la instancia del agente RL (ej. create_monte_carlo_agent para JAX,
            o la clase del modelo para TF). Debe aceptar cgm_shape y other_features_shape.
        framework : str, opcional
            Framework a utilizar ('pytorch', 'jax' o 'tensorflow') (default: 'jax').
        cgm_shape : Tuple[int, ...], opcional
            Forma de los datos CGM (necesario para JAX wrapper).
        other_features_shape : Tuple[int, ...], opcional
            Forma de otras características (necesario para JAX wrapper).
        **model_kwargs
            Argumentos adicionales para el creador del agente/modelo.
            
        Retorna:
        --------
        None
        """
        super().__init__()
        self.framework = framework
        if framework == 'jax':
            if cgm_shape is None or other_features_shape is None:
                 raise ValueError("cgm_shape y other_features_shape son requeridos para el framework JAX.")
            # Pasar formas y kwargs al wrapper JAX
            self.wrapper = RLModelWrapperJAX(model_creator_func, cgm_shape, other_features_shape, **model_kwargs)
        elif framework == 'tensorflow':
            self.wrapper = RLModelWrapperTF(model_creator_func, **model_kwargs)
        elif framework == 'pytorch':
            self.wrapper = RLModelWrapperPyTorch(model_creator_func, **model_kwargs)
        else:
            raise ValueError(f"Framework no soportado: {framework}")

    # Delegación de métodos al wrapper específico
    def start(self, x_cgm: np.ndarray, x_other: np.ndarray, y: np.ndarray,
             rng_key: Any = None) -> Any:
        """
        Inicializa el modelo/agente.
        
        Parámetros:
        -----------
        x_cgm : np.ndarray
            Datos CGM de entrada
        x_other : np.ndarray
            Otras características de entrada
        y : np.ndarray
            Valores objetivo
        rng_key : Any, opcional
            Clave para generación aleatoria (default: None)
            
        Retorna:
        --------
        Any
            Estado del modelo inicializado
        """
        return self.wrapper.start(x_cgm, x_other, y, rng_key)

    def train(self, x_cgm: np.ndarray, x_other: np.ndarray, y: np.ndarray,
             validation_data: Optional[Tuple[Tuple[np.ndarray, np.ndarray], np.ndarray]] = None,
             epochs: int = CONST_DEFAULT_EPOCHS, batch_size: int = CONST_DEFAULT_BATCH_SIZE) -> Dict[str, List[float]]:
        """
        Entrena el modelo/agente.
        
        Parámetros:
        -----------
        x_cgm : np.ndarray
            Datos CGM de entrenamiento
        x_other : np.ndarray
            Otras características de entrenamiento
        y : np.ndarray
            Valores objetivo
        validation_data : Optional[Tuple[Tuple[np.ndarray, np.ndarray], np.ndarray]], opcional
            Datos de validación como ((x_cgm_val, x_other_val), y_val) (default: None)
        epochs : int, opcional
            Número de épocas de entrenamiento (default: 10)
        batch_size : int, opcional
            Tamaño de lote (default: 32)
            
        Retorna:
        --------
        Dict[str, List[float]]
            Historial de entrenamiento con métricas
        """
        return self.wrapper.train(x_cgm, x_other, y, validation_data, epochs, batch_size)

    def predict(self, x_cgm: np.ndarray, x_other: np.ndarray) -> np.ndarray:
        """
        Realiza predicciones.
        
        Parámetros:
        -----------
        x_cgm : np.ndarray
            Datos CGM para predicción
        x_other : np.ndarray
            Otras características para predicción
        
        Retorna:
        --------
        np.ndarray
            Predicciones del modelo
        """
        return self.wrapper.predict(x_cgm, x_other)

    def save(self, filepath: str) -> None:
        """
        Guarda el modelo/agente (si el wrapper lo soporta).
        
        Parámetros:
        -----------
        filepath : str
            Ruta donde guardar el modelo
            
        Retorna:
        --------
        None
        """
        if hasattr(self.wrapper, 'save'):
            self.wrapper.save(filepath)
        else:
            print_warning(f"El guardado no está implementado para el wrapper {type(self.wrapper).__name__}.")

    def load(self, filepath: str) -> None:
        """
        Carga el modelo/agente (si el wrapper lo soporta).
        
        Parámetros:
        -----------
        filepath : str
            Ruta desde donde cargar el modelo
            
        Retorna:
        --------
        None
        """
        if hasattr(self.wrapper, 'load'):
            self.wrapper.load(filepath)
        else:
            print_warning(f"La carga no está implementada para el wrapper {type(self.wrapper).__name__}.")
