import os, sys
import numpy as np
from typing import Dict, List, Tuple, Callable, Optional, Any, Union
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from scipy.optimize import minimize
from joblib import Parallel, delayed
from config.params import DEBUG

# Constantes compartidas
CONST_VAL_LOSS = "val_loss"
CONST_LOSS = "loss"
CONST_METRIC_MAE = "mae"
CONST_METRIC_RMSE = "rmse"
CONST_METRIC_R2 = "r2"
CONST_MODELS = "models"
CONST_BEST_PREFIX = "best_"
CONST_LOGS_DIR = "logs"
CONST_DEFAULT_EPOCHS = 10 if DEBUG else 100
CONST_DEFAULT_BATCH_SIZE = 32
CONST_DEFAULT_SEED = 42
CONST_FIGURES_DIR = "figures"
CONST_MODEL_TYPES = {
    "dl": "deep_learning",
    "drl": "deep_reinforcement_learning",
    "rl": "reinforcement_learning"
}

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
    all_preds = np.stack(list(predictions_dict.values()))
    if weights is None:
        weights = np.ones(len(predictions_dict)) / len(predictions_dict)
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
    def objective(weights):
        # Normalizar pesos
        weights = weights / np.sum(weights)
        # Obtener predicción del ensemble
        ensemble_pred = create_ensemble_prediction(predictions_dict, weights)
        # Calcular error
        return mean_squared_error(y_true, ensemble_pred)
    
    n_models = len(predictions_dict)
    initial_weights = np.ones(n_models) / n_models
    bounds = [(0, 1) for _ in range(n_models)]
    
    result = minimize(
        objective,
        initial_weights,
        bounds=bounds,
        constraints={'type': 'eq', 'fun': lambda w: np.sum(w) - 1}
    )
    
    return result.x / np.sum(result.x)


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