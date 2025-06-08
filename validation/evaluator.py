import numpy as np
from typing import Dict, List, Tuple, Optional, Union
from constants.constants import SEVERE_HYPOGLYCEMIA_THRESHOLD, HYPOGLYCEMIA_THRESHOLD, HYPERGLYCEMIA_THRESHOLD, SEVERE_HYPERGLYCEMIA_THRESHOLD, IDEAL_LOWER_BOUND, IDEAL_UPPER_BOUND

class ClinicalMetricsEvaluator:
    """
    Evaluador de métricas clínicas para modelos de dosificación de insulina.
    
    Calcula métricas relevantes clínicamente como Tiempo en Rango (TIR),
    Tiempo por Encima del Rango (TAR), Tiempo por Debajo del Rango (TBR), etc.
    """
    
    @staticmethod
    def calculate_time_in_range(glucose_values: np.ndarray, 
                              low_threshold: float = HYPOGLYCEMIA_THRESHOLD,
                              high_threshold: float = HYPERGLYCEMIA_THRESHOLD) -> float:
        """
        Calcula el porcentaje de tiempo en rango.
        
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
        in_range = np.logical_and(glucose_values >= low_threshold, 
                                 glucose_values <= high_threshold)
        return 100.0 * np.mean(in_range)
    
    @staticmethod
    def calculate_time_below_range(glucose_values: np.ndarray, 
                                threshold: float = HYPOGLYCEMIA_THRESHOLD,
                                severe_threshold: float = SEVERE_HYPOGLYCEMIA_THRESHOLD) -> Tuple[float, float]:
        """
        Calcula el porcentaje de tiempo por debajo del rango.
        
        Parámetros:
        -----------
        glucose_values : np.ndarray
            Valores de glucosa en mg/dL
        threshold : float, opcional
            Límite de hipoglucemia (default: 70.0)
        severe_threshold : float, opcional
            Límite de hipoglucemia severa (default: 54.0)
            
        Retorna:
        --------
        Tuple[float, float]
            (Porcentaje tiempo < threshold, Porcentaje tiempo < severe_threshold)
        """
        below_range = glucose_values < threshold
        severe_below = glucose_values < severe_threshold
        return 100.0 * np.mean(below_range), 100.0 * np.mean(severe_below)
    
    @staticmethod
    def calculate_time_above_range(glucose_values: np.ndarray, 
                                threshold: float = HYPERGLYCEMIA_THRESHOLD,
                                severe_threshold: float = SEVERE_HYPERGLYCEMIA_THRESHOLD) -> Tuple[float, float]:
        """
        Calcula el porcentaje de tiempo por encima del rango.
        
        Parámetros:
        -----------
        glucose_values : np.ndarray
            Valores de glucosa en mg/dL
        threshold : float, opcional
            Límite de hiperglucemia (default: 180.0)
        severe_threshold : float, opcional
            Límite de hiperglucemia severa (default: 250.0)
            
        Retorna:
        --------
        Tuple[float, float]
            (Porcentaje tiempo > threshold, Porcentaje tiempo > severe_threshold)
        """
        above_range = glucose_values > threshold
        severe_above = glucose_values > severe_threshold
        return 100.0 * np.mean(above_range), 100.0 * np.mean(severe_above)
    
    @staticmethod
    def calculate_glucose_variability(glucose_values: np.ndarray) -> Dict[str, float]:
        """
        Calcula métricas de variabilidad de glucosa.
        
        Parámetros:
        -----------
        glucose_values : np.ndarray
            Valores de glucosa en mg/dL
            
        Retorna:
        --------
        Dict[str, float]
            Diccionario con métricas de variabilidad (SD, CV, MAGE, etc.)
        """
        metrics = {
            'mean_glucose': float(np.mean(glucose_values)),
            'median_glucose': float(np.median(glucose_values)),
            'std_glucose': float(np.std(glucose_values)),
            'cv_glucose': float(np.std(glucose_values) / np.mean(glucose_values) * 100) if np.mean(glucose_values) > 0 else 0.0,
            'min_glucose': float(np.min(glucose_values)),
            'max_glucose': float(np.max(glucose_values)),
            'glucose_range': float(np.max(glucose_values) - np.min(glucose_values))
        }
        
        # Calcular MAGE (Mean Amplitude of Glycemic Excursions) de manera simplificada
        diff = np.abs(np.diff(glucose_values))
        significant_excursions = diff > (np.std(glucose_values) * 1.0)
        if np.any(significant_excursions):
            metrics['mage'] = float(np.mean(diff[significant_excursions]))
        else:
            metrics['mage'] = 0.0
            
        return metrics
    
    @staticmethod
    def evaluate_clinical_metrics(glucose_values: np.ndarray) -> Dict[str, Union[float, Dict[str, float]]]:
        """
        Evalúa todas las métricas clínicas para una serie de valores de glucosa.
        
        Parámetros:
        -----------
        glucose_values : np.ndarray
            Valores de glucosa en mg/dL
            
        Retorna:
        --------
        Dict[str, Union[float, Dict[str, float]]]
            Diccionario con todas las métricas clínicas
        """
        time_in_range = ClinicalMetricsEvaluator.calculate_time_in_range(glucose_values)
        tbr, severe_tbr = ClinicalMetricsEvaluator.calculate_time_below_range(glucose_values)
        tar, severe_tar = ClinicalMetricsEvaluator.calculate_time_above_range(glucose_values)
        variability = ClinicalMetricsEvaluator.calculate_glucose_variability(glucose_values)
        
        return {
            'time_in_range': time_in_range,
            'time_below_range': tbr,
            'time_above_range': tar,
            'severe_hypoglycemia': severe_tbr,
            'severe_hyperglycemia': severe_tar,
            'variability': variability
        }
    
    @staticmethod
    def calculate_risk_index(glucose_values: np.ndarray) -> Dict[str, float]:
        """
        Calcula índices de riesgo basados en valores de glucosa.
        
        Parámetros:
        -----------
        glucose_values : np.ndarray
            Valores de glucosa en mg/dL
            
        Retorna:
        --------
        Dict[str, float]
            Índices de riesgo LBGI (bajo) y HBGI (alto)
        """
        # Convertir de mg/dL a mmol/L para fórmulas estándar
        glucose_mmol = glucose_values / 18.0
        
        # Función de transformación para índice de riesgo
        def risk_function(bg):
            return 10.0 * (np.log(bg) ** 2)
        
        # Calcular riesgo para cada valor
        transformed = risk_function(glucose_mmol)
        
        # Separar riesgos altos y bajos
        r_low = np.where(glucose_mmol < 5.6, transformed, 0)
        r_high = np.where(glucose_mmol > 5.6, transformed, 0)
        
        # Calcular LBGI y HBGI
        lbgi = np.mean(r_low)
        hbgi = np.mean(r_high)
        
        return {
            'lbgi': float(lbgi),  # Low Blood Glucose Index
            'hbgi': float(hbgi),  # High Blood Glucose Index
            'bgri': float(lbgi + hbgi)  # Blood Glucose Risk Index
        }