from typing import Dict, List, Tuple, Any, Optional, Callable, Union
import numpy as np
import jax
import jax.numpy as jnp
from flax.training import train_state
from flax.training.checkpoints import save_checkpoint, restore_checkpoint
import os
import pickle
import tensorflow as tf
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm.auto import tqdm

from custom.DeepReinforcementLearning.drl_jax import DRLModelWrapperJAX
from custom.DeepReinforcementLearning.drl_pt import DRLModelWrapperPyTorch
from custom.DeepReinforcementLearning.drl_tf import DRLModelWrapperTF

from constants.constants import CONST_DEFAULT_SEED, CONST_DEFAULT_EPOCHS, CONST_DEFAULT_BATCH_SIZE
from config.models_config import EARLY_STOPPING_POLICY
from custom.model_wrapper import ModelWrapper
from custom.printer import print_debug, print_info, print_warning, print_error, print_success

# Constantes para uso repetido
CONST_ACTOR = "actor"
CONST_CRITIC = "critic"
CONST_TARGET = "target"
CONST_PARAMS = "params"
CONST_DEVICE = "device"
CONST_MODEL_INIT_ERROR = "El modelo debe ser inicializado antes de {}"
CONST_LOSS = "loss"
CONST_VAL_LOSS = "val_loss"
CONST_EPSILON = 1e-10

class DRLModelWrapper(ModelWrapper):
    """
    Wrapper para modelos de aprendizaje por refuerzo profundo que selecciona el wrapper 
    adecuado según el framework.
    
    Parámetros:
    -----------
    model_cls : Callable
        Clase del modelo DRL a instanciar
    framework : str
        Framework a utilizar ('jax', 'tensorflow' o 'pytorch')
    **model_kwargs
        Argumentos para el constructor del modelo
    """
    
    def __init__(self, model_cls: Callable, framework: str = 'jax', algorithm: str = "", **model_kwargs) -> None:
        """
        Inicializa el wrapper seleccionando el backend adecuado.
        
        Parámetros:
        -----------
        model_cls : Callable
            Clase del modelo DRL a instanciar
        framework : str
            Framework a utilizar ('jax', 'tensorflow' o 'pytorch')
        algorithm : str
            Nombre del algoritmo (opcional)
        **model_kwargs
            Argumentos para el constructor del modelo
        """
        super().__init__()
        self.framework = framework.lower()
        
        # Seleccionar el wrapper adecuado según el framework
        if self.framework == 'jax':
            self.wrapper = DRLModelWrapperJAX(model_cls, algorithm, **model_kwargs)
        elif self.framework == 'pytorch':
            self.wrapper = DRLModelWrapperPyTorch(model_cls, algorithm, **model_kwargs)
        else:
            self.wrapper = DRLModelWrapperTF(model_cls, algorithm, **model_kwargs)
    
    def add_early_stopping(self, patience: int = 10, min_delta: float = 0.0, restore_best_weights: bool = True) -> None:
        """
        Agrega early stopping al modelo.

        Parámetros:
        -----------
        patience : int, opcional
            Número de épocas a esperar para detener el entrenamiento (default: 10)
        min_delta : float, opcional
            Mínima mejora requerida para considerar una mejora (default: 0.0)
        restore_best_weights : bool, opcional
            Si restaurar los mejores pesos al finalizar (default: True)
        """
        if hasattr(self.wrapper, 'add_early_stopping'):
            self.wrapper.add_early_stopping(patience, min_delta, restore_best_weights)
        else:
            super().add_early_stopping(patience, min_delta, restore_best_weights)
    
    def start(self, x_cgm: np.ndarray, x_other: np.ndarray, y: np.ndarray, 
             rng_key: Any = None) -> Any:
        """
        Inicializa el modelo DRL con los datos de entrada.
        
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
            Estado inicial del modelo o parámetros
        """
        return self.wrapper.start(x_cgm, x_other, y, rng_key)
    
    def train(self, x_cgm: np.ndarray, x_other: np.ndarray, y: np.ndarray, 
             validation_data: Optional[Tuple[Tuple[np.ndarray, np.ndarray], np.ndarray]] = None,
             epochs: int = CONST_DEFAULT_EPOCHS, batch_size: int = CONST_DEFAULT_BATCH_SIZE, verbose: int = 1) -> Dict[str, List[float]]:
        """
        Entrena el modelo DRL con los datos proporcionados.
        
        Parámetros:
        -----------
        x_cgm : np.ndarray
            Datos CGM de entrenamiento
        x_other : np.ndarray
            Otras características de entrenamiento
        y : np.ndarray
            Valores objetivo (acciones)
        validation_data : Optional[Tuple[Tuple[np.ndarray, np.ndarray], np.ndarray]], opcional
            Datos de validación como ((x_cgm_val, x_other_val), y_val) (default: None)
        epochs : int, opcional
            Número de épocas de entrenamiento (default: 10)
        batch_size : int, opcional
            Tamaño de lote (default: 32)
        verbose : int, opcional
            Nivel de verbosidad (0=silencioso, 1=progreso, 2=detallado)
            
        Retorna:
        --------
        Dict[str, List[float]]
            Historial de entrenamiento con métricas
        """
        if verbose > 0:
            print_info(f"Entrenando modelo {self.wrapper.algorithm} en {self.framework}...")
            print_info(f"Épocas: {epochs}, Batch size: {batch_size}, Ejemplos: {len(y)}")
        
        # Delegar el entrenamiento al wrapper específico con el nivel de verbosidad
        if hasattr(self.wrapper, 'fit') and callable(self.wrapper.fit):
            return self.wrapper.fit(x_cgm, x_other, y, validation_data, epochs, batch_size, verbose=verbose)
        else:
            return self.wrapper.train(x_cgm, x_other, y, validation_data, epochs, batch_size, verbose=verbose)
    
    def predict(self, x_cgm: np.ndarray, x_other: np.ndarray) -> np.ndarray:
        """
        Realiza predicciones con el modelo entrenado.
        
        Parámetros:
        -----------
        x_cgm : np.ndarray
            Datos CGM para predicción
        x_other : np.ndarray
            Otras características para predicción
            
        Retorna:
        --------
        np.ndarray
            Predicciones del modelo (acciones)
        """
        return self.wrapper.predict(x_cgm, x_other)
    def save(self, filepath: str) -> None:
        """
        Guarda el modelo/agente (si el wrapper lo soporta).
        
        Parámetros:
        -----------
        filepath : str
            Ruta del archivo donde guardar el modelo
            
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
            Ruta del archivo desde donde cargar el modelo
            
        Retorna:
        --------
        None
        """
        if hasattr(self.wrapper, 'load'):
            self.wrapper.load(filepath)
        else:
            print(f"Advertencia: La carga no está implementada para el wrapper {type(self.wrapper).__name__}.")
