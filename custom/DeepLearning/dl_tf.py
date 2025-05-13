from typing import Dict, List, Tuple, Callable, Optional, Any
import numpy as np

from constants.constants import CONST_DEFAULT_SEED, CONST_DEFAULT_EPOCHS, CONST_DEFAULT_BATCH_SIZE
from custom.model_wrapper import ModelWrapper
from custom.printer import cprint, print_debug, print_info, print_error

# Constantes para mensajes de error
CONST_MODEL_INIT_ERROR = "El modelo debe ser inicializado antes de {}"
CONST_DEVICE = "device"


class DLModelWrapperTF(ModelWrapper):
    """
    Wrapper para modelos de deep learning implementados en TensorFlow.
    
    Parámetros:
    -----------
    model_creator : Callable
        Función que crea una instancia del modelo
    """
    
    def __init__(self, model_creator: Callable) -> None:
        """
        Inicializa el wrapper con un creador de modelo TensorFlow.
        
        Parámetros:
        -----------
        model_creator : Callable
            Función que crea una instancia del modelo
        """
        super().__init__()
        self.model_creator = model_creator
        self.model = None
    
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
        if self.model is None:
            self.model = self.model_creator()
        
        # Compilar si no está compilado
        if not hasattr(self.model, 'compiled_loss'):
            self.model.compile(optimizer='adam', loss='mse')
            
        return self.model
    
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
        if self.model is None:
            self.start(x_cgm, x_other, y)
            
        val_data = None
        if validation_data is not None:
            (x_cgm_val, x_other_val), y_val = validation_data
            val_data = ([x_cgm_val, x_other_val], y_val)
            
        history = self.model.fit(
            [x_cgm, x_other], y,
            validation_data=val_data,
            epochs=epochs,
            batch_size=batch_size,
            verbose=1
        )
        
        return history.history
    
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
        if self.model is None:
            raise ValueError("El modelo debe ser inicializado antes de predecir")
            
        return self.model.predict([x_cgm, x_other])
    
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
        if self.model is None:
            raise ValueError("El modelo debe ser inicializado antes de evaluar")
            
        loss = self.model.evaluate([x_cgm, x_other], y, verbose=0)
        preds = self.predict(x_cgm, x_other)
        
        return {
            "loss": float(loss),
            "mae": float(np.mean(np.abs(preds - y))),
            "rmse": float(np.sqrt(np.mean((preds - y) ** 2))),
            "r2": float(1 - np.sum((y - preds) ** 2) / np.sum((y - np.mean(y)) ** 2))
        }
    
    def save(self, path: str) -> None:
        """
        Guarda el modelo en disco.
        
        Parámetros:
        -----------
        path : str
            Ruta donde guardar el modelo
        """
        if self.model is None:
            raise ValueError("El modelo debe ser inicializado antes de guardarlo")
            
        self.model.save(path)
    
    def load(self, path: str) -> None:
        """
        Carga el modelo desde disco.
        
        Parámetros:
        -----------
        path : str
            Ruta desde donde cargar el modelo
        """
        import tensorflow as tf
        self.model = tf.keras.models.load_model(path)
