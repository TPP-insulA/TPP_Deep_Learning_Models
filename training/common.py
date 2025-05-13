import os, sys
import numpy as np
from typing import Dict, List, Tuple, Callable, Optional, Any, Union
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from scipy.optimize import minimize
from joblib import Parallel, delayed
from config.params import DEBUG
from constants.constants import CONST_VAL_LOSS, CONST_LOSS, CONST_METRIC_MAE, CONST_METRIC_RMSE, CONST_METRIC_R2, CONST_MODELS, CONST_BEST_PREFIX, CONST_LOGS_DIR, CONST_DEFAULT_EPOCHS, CONST_DEFAULT_BATCH_SIZE, CONST_DEFAULT_SEED, CONST_FIGURES_DIR, CONST_MODEL_TYPES
from custom.printer import print_debug, print_info, print_warning, print_error

def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """
    Calcula métricas de rendimiento para las predicciones del modelo.
    
    Parámetros:
    -----------
    y_true : np.ndarray
        Valores objetivo verdaderos
    y_pred : np.ndarray
        Valores predichos por el modelo
        
    Retorna:
    --------
    Dict[str, float]
        Diccionario con métricas MAE, RMSE y R²
    """
    print_debug(f"results['predictions']: {y_pred}")
    return {
        CONST_METRIC_MAE: float(mean_absolute_error(y_true, y_pred)),
        CONST_METRIC_RMSE: float(np.sqrt(mean_squared_error(y_true, y_pred))),
        CONST_METRIC_R2: float(r2_score(y_true, y_pred))
    }


def create_ensemble_prediction(predictions_dict: Dict[str, np.ndarray], 
                              weights: Optional[np.ndarray] = None) -> np.ndarray:
    """
    Combina predicciones de múltiples modelos usando promedio ponderado.
    
    Parámetros:
    -----------
    predictions_dict : Dict[str, np.ndarray]
        Diccionario con predicciones de cada modelo
    weights : Optional[np.ndarray], opcional
        Pesos para cada modelo. Si es None, usa promedio simple (default: None)
        
    Retorna:
    --------
    np.ndarray
        Predicciones combinadas del ensemble
    """
    # Normalizar todos los arrays de predicciones a 1D
    normalized_preds = {}
    for model_name, preds in predictions_dict.items():
        # Convertir a numpy si es necesario
        if not isinstance(preds, np.ndarray):
            preds = np.array(preds)
            
        # Asegurar que el array sea 1D
        if preds.ndim > 1:
            preds = preds.reshape(-1)
            
        normalized_preds[model_name] = preds
    
    # Verificar si todos los arrays tienen la misma longitud después de normalizar
    lengths = [len(preds) for preds in normalized_preds.values()]
    if len(set(lengths)) > 1:
        print(f"Advertencia: Los modelos tienen longitudes de predicción diferentes: {lengths}")
        # Usar la longitud común mínima
        min_length = min(lengths)
        for model_name, preds in normalized_preds.items():
            normalized_preds[model_name] = preds[:min_length]
    
    # Apilar predicciones y aplicar pesos
    all_preds = np.stack(list(normalized_preds.values()))
    if weights is None:
        weights = np.ones(len(normalized_preds)) / len(normalized_preds)
    return np.average(all_preds, axis=0, weights=weights)


def optimize_ensemble_weights(predictions_dict: Dict[str, np.ndarray], 
                             y_true: np.ndarray) -> np.ndarray:
    """
    Optimiza pesos del ensemble usando optimización.
    
    Parámetros:
    -----------
    predictions_dict : Dict[str, np.ndarray]
        Diccionario con predicciones de cada modelo
    y_true : np.ndarray
        Valores objetivo verdaderos
        
    Retorna:
    --------
    np.ndarray
        Pesos optimizados para cada modelo
    """
    # Asegurar que y_true sea 1D para consistencia
    if y_true.ndim > 1:
        y_true = y_true.reshape(-1)
    
    # Registrar los modelos y sus formas de predicción
    print("\nFormas de predicciones de modelos antes de optimización:")
    for model_name, preds in predictions_dict.items():
        print(f"  {model_name}: {np.array(preds).shape}")
    
    def objective(weights: np.ndarray) -> float:
        """
        Función objetivo para optimización de pesos.
        
        Parámetros:
        -----------
        weights : np.ndarray
            Pesos a optimizar
            
        Retorna:
        --------
        float
            Error cuadrático medio
        """
        # Normalizar pesos
        weights = weights / np.sum(weights)
        # Obtener predicción del ensemble
        ensemble_pred = create_ensemble_prediction(predictions_dict, weights)
        # Asegurar que las predicciones del ensemble coincidan con la longitud de y_true
        if len(ensemble_pred) > len(y_true):
            ensemble_pred = ensemble_pred[:len(y_true)]
        elif len(ensemble_pred) < len(y_true):
            y_true_temp = y_true[:len(ensemble_pred)]
            return mean_squared_error(y_true_temp, ensemble_pred)
        # Calcular error
        return mean_squared_error(y_true, ensemble_pred)
    
    n_models = len(predictions_dict)
    initial_weights = np.ones(n_models) / n_models
    bounds = [(0, 1) for _ in range(n_models)]
    
    try:
        result = minimize(
            objective,
            initial_weights,
            bounds=bounds,
            constraints={'type': 'eq', 'fun': lambda w: np.sum(w) - 1}
        )
        optimized_weights = result.x / np.sum(result.x)
        return optimized_weights
    except Exception as e:
        print(f"Error en optimización: {e}")
        print("Volviendo a pesos iguales")
        return initial_weights


def enhance_features(x_cgm: np.ndarray, x_other: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Mejora las características de entrada añadiendo derivadas y estadísticas.
    
    Parámetros:
    -----------
    x_cgm : np.ndarray
        Datos CGM de forma (muestras, pasos_tiempo, características)
    x_other : np.ndarray
        Otras características de forma (muestras, características)
        
    Retorna:
    --------
    Tuple[np.ndarray, np.ndarray]
        Tupla con (cgm_features_mejoradas, other_features_mejoradas)
    """
    # Calcular diferencia entre puntos de tiempo consecutivos (derivada)
    cgm_diff = np.diff(x_cgm, axis=1)
    
    # Verificar la forma de cgm_diff antes de aplicar padding
    print(f"Forma de cgm_diff: {cgm_diff.shape}")
    
    # Aplicar padding según la dimensionalidad real
    if cgm_diff.ndim == 3:
        cgm_diff = np.pad(cgm_diff, ((0, 0), (1, 0), (0, 0)), mode='edge')
    else:
        cgm_diff = np.pad(cgm_diff, ((0, 0), (1, 0)), mode='edge')
    
    # Añadir estadísticas móviles
    window = 5
    rolling_mean = np.apply_along_axis(
        lambda x: np.convolve(x, np.ones(window)/window, mode='same'),
        1, x_cgm.squeeze()
    )
    
    # Concatenar características mejoradas
    x_cgm_enhanced = np.concatenate([
        x_cgm,
        cgm_diff,
        rolling_mean[..., np.newaxis]
    ], axis=-1)
    
    return x_cgm_enhanced, x_other


def get_model_type(model_name: str) -> str:
    """
    Determina el tipo de modelo basado en su nombre.
    
    Parámetros:
    -----------
    model_name : str
        Nombre del modelo
        
    Retorna:
    --------
    str
        Tipo de modelo: "dl", "rl" o "drl"
    """
    if any(x in model_name for x in ["monte_carlo", "policy_iteration", "q_learning", "sarsa", "value_iteration", "reinforce"]):
        return "rl"
    elif any(x in model_name for x in ["a2c", "a3c", "ddpg", "dqn", "ppo", "sac", "trpo"]):
        return "drl"
    else:
        return "dl"


def process_training_results(model_results: List[Dict], 
                            y_test: np.ndarray) -> Tuple[Dict[str, Dict], Dict[str, np.ndarray], Dict[str, Dict]]:
    """
    Procesa resultados de múltiples modelos entrenados.
    
    Parámetros:
    -----------
    model_results : List[Dict]
        Lista de resultados de modelos con keys 'name', 'history', 'predictions'
    y_test : np.ndarray
        Valores objetivo de prueba
        
    Retorna:
    --------
    Tuple[Dict[str, Dict], Dict[str, np.ndarray], Dict[str, Dict]]
        (historiales, predicciones, métricas) diccionarios
    """
    from config.params import FRAMEWORK
    
    # Procesar resultados secuencialmente cuando se usa JAX para evitar deadlocks
    if FRAMEWORK == "jax":
        print("\nCalculando métricas secuencialmente (compatible con JAX)...")
        metric_results = [
            calculate_metrics(
                y_test, 
                np.array(result['predictions'])
            ) for result in model_results
        ]
    else:
        # Para TensorFlow u otros frameworks, mantener el paralelismo
        print("\nCalculando métricas en paralelo...")
        with Parallel(n_jobs=-1, verbose=1) as parallel:
            metric_results = parallel(
                delayed(calculate_metrics)(
                    y_test, 
                    np.array(result['predictions'])
                ) for result in model_results
            )
    
    # Almacenar resultados
    histories = {}
    predictions = {}
    metrics = {}
    
    for result, metric in zip(model_results, metric_results):
        name = result['name']
        histories[name] = result['history']
        predictions[name] = np.array(result['predictions'])
        metrics[name] = metric
    
    return histories, predictions, metrics