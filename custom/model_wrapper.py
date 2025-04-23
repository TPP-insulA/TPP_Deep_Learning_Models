from typing import Dict, List, Tuple, Any, Optional, Union
import numpy as np
import jax
import jax.numpy as jnp
import flax.linen as nn

class ModelWrapper:
    """
    Clase base para encapsular modelos de aprendizaje profundo y por refuerzo.
    
    Proporciona una interfaz unificada para inicialización, entrenamiento y predicción.
    Todos los modelos deben heredar de esta clase.
    """
    
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
        raise NotImplementedError("El método initialize debe ser implementado por las subclases")
    
    def train(self, x_cgm: np.ndarray, x_other: np.ndarray, y: np.ndarray, 
             validation_data: Optional[Tuple[Tuple[np.ndarray, np.ndarray], np.ndarray]] = None,
             epochs: int = 10, batch_size: int = 32) -> Dict[str, List[float]]:
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
        raise NotImplementedError("El método train debe ser implementado por las subclases")
    
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
        raise NotImplementedError("El método predict debe ser implementado por las subclases")
    
    def evaluate(self, x_cgm: np.ndarray, x_other: np.ndarray, y: np.ndarray) -> float:
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
        float
            Pérdida de evaluación
        """
        preds = self.predict(x_cgm, x_other)
        return float(np.mean((preds - y) ** 2))
    
    def add_early_stopping(self, patience: int = 10, min_delta: float = 0.0, restore_best_weights: bool = True) -> None:
        """
        Añade early stopping al modelo.
        
        Parámetros:
        -----------
        patience : int, opcional
            Número de épocas a esperar para detener el entrenamiento (default: 10)
        min_delta : float, opcional
            Cambio mínimo considerado como mejora (default: 0.0)
        restore_best_weights : bool, opcional
            Si restaurar los mejores pesos al finalizar (default: True)
        """
        self.early_stopping = {
            'patience': patience,
            'min_delta': min_delta,
            'restore_best_weights': restore_best_weights,
            'best_loss': float('inf'),
            'best_params': None,
            'wait': 0
        }