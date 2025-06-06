from typing import Dict, List, Tuple, Any, Optional, Union
import numpy as np
import jax
import jax.numpy as jnp
import flax.linen as nn

from constants.constants import CONST_DEFAULT_SEED, CONST_DEFAULT_EPOCHS, CONST_DEFAULT_BATCH_SIZE
from validation.evaluator import ClinicalMetricsEvaluator

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
    
    def predict_with_context(self, x_cgm: np.ndarray, x_other: np.ndarray, 
                        carb_intake: float, 
                        sleep_quality: int = None, 
                        work_intensity: int = None, 
                        exercise_intensity: int = None) -> float:
        """
        Realiza predicciones con el modelo entrenado, considerando contexto adicional.
        
        Parámetros:
        -----------
        x_cgm : np.ndarray
            Datos CGM para predicción (array con mediciones de glucosa recientes)
        x_other : np.ndarray
            Otras características para predicción
        carb_intake : float
            Ingesta de carbohidratos en gramos (obligatorio)
        sleep_quality : int, opcional
            Calidad del sueño (escala de 0-10)
        work_intensity : int, opcional
            Intensidad del trabajo (escala de 0-10)
        exercise_intensity : int, opcional
            Intensidad del ejercicio (escala de 0-10)
                
        Retorna:
        --------
        float
            Dosis de insulina recomendada en unidades
        """
        raise NotImplementedError("El método predict_with_context debe ser implementado por las subclases")
    
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
    
    def evaluate_clinical(self, simulator, x_cgm: np.ndarray, x_other: np.ndarray, 
                    initial_glucose: np.ndarray, carb_intake: np.ndarray, 
                    ground_truth_insulin: np.ndarray = None, 
                    simulation_hours: int = 24) -> Dict[str, float]:
        """
        Evalúa el modelo utilizando métricas clínicas con un simulador de glucosa.
        
        Parámetros:
        -----------
        simulator : GlucoseSimulator
            Simulador de glucosa
        x_cgm : np.ndarray
            Datos CGM para predicción
        x_other : np.ndarray
            Otras características para predicción
        initial_glucose : np.ndarray
            Valores iniciales de glucosa para simulación
        carb_intake : np.ndarray
            Valores de ingesta de carbohidratos
        ground_truth_insulin : np.ndarray, opcional
            Dosis reales de insulina para comparación
        simulation_hours : int, opcional
            Duración de la simulación en horas (default: 24)
                
        Retorna:
        --------
        Dict[str, float]
            Métricas clínicas de evaluación
        """
        
        # Predecir dosis de insulina con el modelo
        predicted_doses = []
        
        for i in range(len(x_cgm)):
            carbs = float(carb_intake[i])
            
            # Predecir dosis con contexto
            dose = self.predict_with_context(
                x_cgm[i:i+1], 
                x_other[i:i+1],
                carb_intake=carbs
            )
            
            predicted_doses.append(float(dose))
        
        # Simular trayectorias de glucosa usando las dosis predichas
        glucose_trajectories = []
        for i in range(len(initial_glucose)):
            glucose_trajectory = simulator.simulate(
                initial_glucose=initial_glucose[i],
                insulin_doses=[predicted_doses[i]],
                carb_intake=[carb_intake[i]],
                duration_hours=simulation_hours
            )
            glucose_trajectories.append(glucose_trajectory)
        
        # Calcular métricas clínicas para todas las trayectorias
        all_metrics = {}
        for i, trajectory in enumerate(glucose_trajectories):
            metrics = ClinicalMetricsEvaluator.evaluate_clinical_metrics(trajectory)
            
            # Agregar a métricas globales
            for key, value in metrics.items():
                if key not in all_metrics:
                    all_metrics[key] = []
                
                if isinstance(value, dict):
                    for subkey, subvalue in value.items():
                        subkey_full = f"{key}_{subkey}"
                        if subkey_full not in all_metrics:
                            all_metrics[subkey_full] = []
                        all_metrics[subkey_full].append(subvalue)
                else:
                    all_metrics[key].append(value)
        
        # Promediar métricas
        avg_metrics = {key: float(np.mean(values)) for key, values in all_metrics.items()}
        
        return avg_metrics
    
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