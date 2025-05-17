import time
from typing import Dict, List, Tuple, Callable, Optional, Any, Union
import numpy as np
import jax
import jax.numpy as jnp
from flax.training import train_state
from flax.core.frozen_dict import FrozenDict
import optax
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from tqdm.auto import tqdm

from constants.constants import CONST_DEFAULT_SEED, CONST_DEFAULT_EPOCHS, CONST_DEFAULT_BATCH_SIZE
from custom.model_wrapper import ModelWrapper
from custom.printer import cprint, print_debug, print_info, print_error
from custom.DeepLearning.dl_tf import DLModelWrapperTF
from custom.DeepLearning.dl_jax import DLModelWrapperJAX
from custom.DeepLearning.dl_pt import DLModelWrapperPyTorch

# Constantes para mensajes de error
CONST_MODEL_INIT_ERROR = "El modelo debe ser inicializado antes de {}"
CONST_DEVICE = "device"

# Clase principal que selecciona el wrapper adecuado según el framework
class DLModelWrapper(ModelWrapper):
    """
    Wrapper para modelos de deep learning que selecciona el wrapper adecuado según el framework.
    
    Parámetros:
    -----------
    model_creator : Callable
        Función que crea una instancia del modelo
    framework : str
        Framework a utilizar ('jax' o 'tensorflow')
    """
    
    def __init__(self, model_creator: Callable, framework: str = 'jax') -> None:
        """
        Inicializa el wrapper con un creador de modelo.
        
        Parámetros:
        -----------
        model_creator : Callable
            Función que crea una instancia del modelo
        framework : str, opcional
            Framework a utilizar ('pytorch', 'jax' o 'tensorflow') (default: 'jax')
        """
        super().__init__()
        
        # Seleccionar wrapper específico según el framework
        if framework.lower() == 'tensorflow':
            self.wrapper = DLModelWrapperTF(model_creator)
        elif framework.lower() == 'pytorch':
            self.wrapper = DLModelWrapperPyTorch(model_creator)
        else:
            self.wrapper = DLModelWrapperJAX(model_creator)
    
    # Delegación de métodos al wrapper específico
    def start(self, x_cgm: np.ndarray, x_other: np.ndarray, y: np.ndarray, 
             rng_key: Any = None) -> Any:
        """
        Inicializa el modelo con los datos de entrada.
        
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
             epochs: int = CONST_DEFAULT_EPOCHS, batch_size: int = CONST_DEFAULT_BATCH_SIZE) -> Dict[str, List[float]]:
        """
        Entrena el modelo con los datos proporcionados.
        
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
            Predicciones del modelo
        """
        return self.wrapper.predict(x_cgm, x_other)
    
    def evaluate(self, x_cgm: np.ndarray, x_other: np.ndarray, y: np.ndarray) -> Dict[str, float]:
        """
        Evalúa el modelo con datos de prueba.
        
        Parámetros:
        -----------
        x_cgm : np.ndarray
            Datos CGM de prueba
        x_other : np.ndarray
            Otras características de prueba
        y : np.ndarray
            Valores objetivo reales
            
        Retorna:
        --------
        Dict[str, float]
            Diccionario con métricas de evaluación
        """
        return self.wrapper.evaluate(x_cgm, x_other, y)
    
    def save(self, path: str) -> None:
        """
        Guarda el modelo en disco.
        
        Parámetros:
        -----------
        path : str
            Ruta donde guardar el modelo
        """
        self.wrapper.save(path)
    
    def load(self, path: str) -> None:
        """
        Carga el modelo desde disco.
        
        Parámetros:
        -----------
        path : str
            Ruta desde donde cargar el modelo
        """
        self.wrapper.load(path)
    
    # Método para acceder al early stopping en JAX
    def add_early_stopping(self, patience: int = 10, min_delta: float = 0.0, restore_best_weights: bool = True) -> None:
        """
        Añade early stopping al modelo (solo para JAX).
        
        Parámetros:
        -----------
        patience : int, opcional
            Número de épocas a esperar para detener el entrenamiento (default: 10)
        min_delta : float, opcional
            Cambio mínimo considerado como mejora (default: 0.0)
        restore_best_weights : bool, opcional
            Si restaurar los mejores pesos al finalizar (default: True)
        """
        if isinstance(self.wrapper, DLModelWrapperJAX):
            self.wrapper.add_early_stopping(patience, min_delta, restore_best_weights)
        else:
            print_error("Early stopping solo está disponible para modelos JAX")