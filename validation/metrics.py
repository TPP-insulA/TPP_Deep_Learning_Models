import numpy as np
from typing import Dict, List, Tuple, Optional

def calculate_time_in_range(
    glucose_values: np.ndarray,
    low_threshold: float = 70.0,
    high_threshold: float = 180.0
) -> float:
    """
    Calcula el porcentaje de tiempo en que la glucosa estuvo en el rango objetivo.
    
    Parámetros:
    -----------
    glucose_values : np.ndarray
        Valores de glucosa en mg/dL
    low_threshold : float, opcional
        Límite inferior del rango objetivo (default: 70.0)
    high_threshold : float, opcional
        Límite superior del rango objetivo (default: 180.0)
        
    Retorna:
    --------
    float
        Porcentaje de tiempo en rango (0-100)
    """
    in_range = np.logical_and(
        glucose_values >= low_threshold,
        glucose_values <= high_threshold
    )
    return 100 * np.mean(in_range)

def calculate_glucose_risk_indices(glucose_values: np.ndarray) -> Dict[str, float]:
    """
    Calcula índices de riesgo de glucosa (LBGI, HBGI, BGRI).
    
    Parámetros:
    -----------
    glucose_values : np.ndarray
        Valores de glucosa en mg/dL
        
    Retorna:
    --------
    Dict[str, float]
        Diccionario con índices de riesgo
    """
    # Transformar valores de glucosa a escala logarítmica
    f_glucose = 1.509 * ((np.log(glucose_values))**1.084 - 5.381)
    
    # Funciones de riesgo
    r_glucose = 10 * (f_glucose**2)
    
    # Separar componentes de riesgo
    rl = r_glucose * (f_glucose < 0)  # Riesgo de valores bajos
    rh = r_glucose * (f_glucose > 0)  # Riesgo de valores altos
    
    # Calcular índices
    lbgi = np.mean(rl)  # Low Blood Glucose Index
    hbgi = np.mean(rh)  # High Blood Glucose Index
    bgri = lbgi + hbgi  # Blood Glucose Risk Index
    
    return {
        "lbgi": lbgi,
        "hbgi": hbgi,
        "bgri": bgri
    }

def evaluate_glucose_control(
    glucose_values: np.ndarray,
    time_interval_minutes: float = 5.0
) -> Dict[str, float]:
    """
    Evalúa el control de glucosa con múltiples métricas.
    
    Parámetros:
    -----------
    glucose_values : np.ndarray
        Valores de glucosa en mg/dL
    time_interval_minutes : float, opcional
        Intervalo de tiempo entre mediciones en minutos (default: 5.0)
        
    Retorna:
    --------
    Dict[str, float]
        Diccionario con métricas de control glucémico
    """
    # Tiempo en rangos
    time_in_range = calculate_time_in_range(glucose_values, 70, 180)
    time_below_range = calculate_time_in_range(glucose_values, 0, 70)
    time_above_range = calculate_time_in_range(glucose_values, 180, float('inf'))
    
    # Tiempo en hipoglucemia severa (<54 mg/dL)
    time_severe_hypo = calculate_time_in_range(glucose_values, 0, 54)
    
    # Índices de riesgo
    risk_indices = calculate_glucose_risk_indices(glucose_values)
    
    # Variabilidad
    glucose_std = np.std(glucose_values)
    glucose_cv = 100 * glucose_std / np.mean(glucose_values)  # Coeficiente de variación
    
    # Eventos
    hypo_events = count_events(glucose_values < 70, time_interval_minutes)
    severe_hypo_events = count_events(glucose_values < 54, time_interval_minutes)
    hyper_events = count_events(glucose_values > 180, time_interval_minutes)
    
    return {
        "time_in_range": time_in_range,
        "time_below_range": time_below_range,
        "time_above_range": time_above_range,
        "time_severe_hypo": time_severe_hypo,
        "lbgi": risk_indices["lbgi"],
        "hbgi": risk_indices["hbgi"],
        "bgri": risk_indices["bgri"],
        "glucose_std": glucose_std,
        "glucose_cv": glucose_cv,
        "hypo_events": hypo_events,
        "severe_hypo_events": severe_hypo_events,
        "hyper_events": hyper_events
    }

def count_events(condition_array: np.ndarray, time_interval_minutes: float = 5.0) -> int:
    """
    Cuenta el número de eventos en una serie temporal, donde un evento
    se define como una secuencia continua donde la condición es verdadera.
    
    Parámetros:
    -----------
    condition_array : np.ndarray
        Array booleano que indica cuándo se cumple la condición
    time_interval_minutes : float, opcional
        Intervalo de tiempo entre mediciones en minutos (default: 5.0)
        
    Retorna:
    --------
    int
        Número de eventos detectados
    """
    # Convertir a enteros (0 y 1)
    condition_int = condition_array.astype(int)
    
    # Detectar cambios en la condición
    changes = np.diff(np.concatenate(([0], condition_int, [0])))
    
    # Inicios de eventos son cambios de 0 a 1
    starts = np.where(changes == 1)[0]
    
    # Finales de eventos son cambios de 1 a 0
    ends = np.where(changes == -1)[0]
    
    # Calcular la duración de cada evento en minutos
    durations = (ends - starts) * time_interval_minutes
    
    # Contar solo eventos que duran al menos 15 minutos
    valid_events = durations >= 15
    
    return np.sum(valid_events)