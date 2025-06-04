import os
import gymnasium as gym
import numpy as np
import polars as pl
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import argparse
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.callbacks import EvalCallback, BaseCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.utils import set_random_seed
import tensorboard
from tensorboard.backend.event_processing import event_accumulator
import logging
import glob
from datetime import datetime, timedelta
from types import SimpleNamespace
import pandas as pd
import jax as jnp
from typing import Any
import time
from tqdm import tqdm

# Configuración de colores para logging
class ColoredFormatter(logging.Formatter):
    """Formateador personalizado para agregar colores a los mensajes de logging"""
    
    COLORS = {
        'DEBUG': '\033[94m',     # Azul
        'INFO': '\033[92m',      # Verde
        'WARNING': '\033[93m',   # Amarillo
        'ERROR': '\033[91m',     # Rojo
        'CRITICAL': '\033[91m\033[1m'  # Rojo negrita
    }
    
    def format(self, record):
        if record.levelname in self.COLORS:
            record.msg = f"{self.COLORS[record.levelname]}{record.msg}\033[0m"
        return super().format(record)

# Configurar logging con colores
logger = logging.getLogger()
handler = logging.StreamHandler()
handler.setFormatter(ColoredFormatter('%(asctime)s - %(levelname)s - %(message)s'))
logger.addHandler(handler)
logger.setLevel(logging.INFO)

# Color magenta para logs de tests de paciente anciana
MAGENTA = '\033[95m'
RESET = '\033[0m'

def compute_glucose_patterns(cgm_values: List[float], extended_cgm: List[float] = None) -> Dict[str, float]:
    """
    Calcula patrones de glucosa a largo plazo y métricas de estabilidad.
    
    Args:
        cgm_values: Valores recientes de CGM (ventana de 2h)
        extended_cgm: Valores extendidos de CGM (ventana de 24h) si están disponibles
        
    Returns:
        Diccionario con características de patrones de glucosa
    """
    patterns = {}
    
    # Use extended CGM if available, otherwise use recent values
    values_24h = extended_cgm if extended_cgm is not None else cgm_values
    
    if len(values_24h) > 0:
        values_array = np.array(values_24h)
        
        # Basic statistics
        patterns['cgm_mean_24h'] = np.mean(values_array)
        patterns['cgm_std_24h'] = np.std(values_array)
        patterns['cgm_median_24h'] = np.median(values_array)
        patterns['cgm_range_24h'] = np.max(values_array) - np.min(values_array)
        
        # Hypoglycemia episodes (< 70 mg/dL)
        hypo_episodes = np.sum(values_array < 70)
        patterns['hypo_episodes_24h'] = hypo_episodes
        patterns['hypo_percentage_24h'] = (hypo_episodes / len(values_array)) * 100
        
        # Hyperglycemia episodes (> 180 mg/dL)
        hyper_episodes = np.sum(values_array > 180)
        patterns['hyper_episodes_24h'] = hyper_episodes
        patterns['hyper_percentage_24h'] = (hyper_episodes / len(values_array)) * 100
        
        # Time in range (70-180 mg/dL)
        in_range = np.sum((values_array >= 70) & (values_array <= 180))
        patterns['time_in_range_24h'] = (in_range / len(values_array)) * 100
        
        # Glucose variability
        if len(values_array) > 1:
            # Coefficient of variation
            patterns['cv_24h'] = (patterns['cgm_std_24h'] / patterns['cgm_mean_24h']) * 100
            
            # Mean absolute glucose change
            patterns['mage_24h'] = np.mean(np.abs(np.diff(values_array)))
            
            # Trend analysis
            if len(values_array) >= 3:
                time_points = np.arange(len(values_array))
                slope, _ = np.polyfit(time_points, values_array, 1)
                patterns['glucose_trend_24h'] = slope
            else:
                patterns['glucose_trend_24h'] = 0.0
        else:
            patterns['cv_24h'] = 0.0
            patterns['mage_24h'] = 0.0
            patterns['glucose_trend_24h'] = 0.0
    else:
        # Default values if no data available
        for key in ['cgm_mean_24h', 'cgm_std_24h', 'cgm_median_24h', 'cgm_range_24h',
                   'hypo_episodes_24h', 'hypo_percentage_24h', 'hyper_episodes_24h',
                   'hyper_percentage_24h', 'time_in_range_24h', 'cv_24h', 'mage_24h',
                   'glucose_trend_24h']:
            patterns[key] = 0.0
    
    return patterns

def encode_time_cyclical(timestamp: datetime) -> Tuple[float, float]:
    """
    Codifica la hora del día como características cíclicas usando seno y coseno.
    
    Args:
        timestamp: Timestamp actual
        
    Returns:
        Tupla de valores (hour_sin, hour_cos)
    """
    hour_decimal = timestamp.hour + timestamp.minute / 60.0 + timestamp.second / 3600.0
    hour_normalized = hour_decimal / 24.0  # Normalizar a [0, 1]
    
    # Convertir a radianes y calcular seno/coseno
    hour_radians = 2 * np.pi * hour_normalized
    hour_sin = np.sin(hour_radians)
    hour_cos = np.cos(hour_radians)
    
    return hour_sin, hour_cos

def compute_clinical_impact(current_cgm: float, predicted_dose: float, carbs: float = 0, iob: float = 0) -> Dict[str, float]:
    """
    Estima el impacto clínico de la dosis predicha en los niveles futuros de glucosa.
    
    Args:
        current_cgm: Nivel actual de glucosa
        predicted_dose: Dosis de insulina predicha
        carbs: Ingesta de carbohidratos
        iob: Insulina activa
        
    Returns:
        Diccionario con resultados clínicos estimados
    """
    # Modelo simplificado de predicción de glucosa
    # Estas son aproximaciones y deberían calibrarse con datos reales
    
    # Factor de sensibilidad a la insulina (mg/dL por unidad)
    ISF = 50  # Rango típico: 30-100
    
    # Ratio de carbohidratos (gramos por unidad)
    CR = 15  # Rango típico: 10-20
    
    # Duración de la acción de la insulina (horas)
    DIA = 4
    
    # Estimar cambio de glucosa por insulina
    insulin_effect = -(predicted_dose + iob) * ISF
    
    # Estimar cambio de glucosa por carbohidratos
    carb_dose_needed = carbs / CR
    carb_effect = carbs * 3  # Aproximación: 1g carb = 3 mg/dL aumento
    
    # Efecto neto (simplificado)
    estimated_glucose_change = insulin_effect + carb_effect
    estimated_future_glucose = current_cgm + estimated_glucose_change
    
    # Evaluación de riesgo
    hypo_risk = 1.0 if estimated_future_glucose < 70 else 0.0
    hyper_risk = 1.0 if estimated_future_glucose > 250 else 0.0
    
    # Puntuación de seguridad (0 = inseguro, 1 = seguro)
    safety_score = 1.0 - (hypo_risk * 0.8 + hyper_risk * 0.2)  # Hipoglucemia es más peligrosa
    
    return {
        'estimated_glucose_change': estimated_glucose_change,
        'estimated_future_glucose': estimated_future_glucose,
        'hypo_risk': hypo_risk,
        'hyper_risk': hyper_risk,
        'safety_score': safety_score,
        'insulin_effect': insulin_effect,
        'carb_effect': carb_effect
    }

class OhioT1DMEnhancedEnv(gym.Env):
    """
    Entorno OhioT1DM mejorado con preprocesamiento y función de recompensa clínica mejorados.
    """
    def __init__(self, df_windows: pl.DataFrame, df_final: pl.DataFrame, patient_id: str = None):
        super().__init__()
        self.df_windows = df_windows
        self.df_final = df_final
        self.patient_id = patient_id
        self.current_idx = 0
        self.episode_length = len(df_windows)

        # Espacio de acción: dosis de bolo continua (0-30 unidades)
        self.action_space = gym.spaces.Box(low=0.0, high=30.0, shape=(1,), dtype=np.float32)
        
        # Espacio de observación mejorado: 24 CGM + 16 características mejoradas + 12 patrones de glucosa
        self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(52,), dtype=np.float32)
        
        # Seguimiento de métricas clínicas
        self.episode_metrics = {
            'time_in_range': [],
            'hypo_events': 0,
            'hyper_events': 0,
            'safety_violations': 0,
            'predicted_doses': [],
            'true_doses': [],
            'clinical_scores': []
        }

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        super().reset(seed=seed)
        self.current_idx = 0
        
        # Reiniciar métricas del episodio
        self.episode_metrics = {
            'time_in_range': [],
            'hypo_events': 0,
            'hyper_events': 0,
            'safety_violations': 0,
            'predicted_doses': [],
            'true_doses': [],
            'clinical_scores': []
        }
        
        obs = self._get_enhanced_observation()
        return obs, {}

    def step(self, action: np.ndarray):
        # Datos del paso actual
        true_bolus = self.df_windows['bolus'][self.current_idx]
        current_cgm = self.df_windows['cgm_0'][self.current_idx]
        carb_input = self.df_windows['carb_input'][self.current_idx]
        iob = self.df_windows['insulin_on_board'][self.current_idx]
        
        # Redondear dosis a incrementos realistas (0.5 unidades)
        dose = np.round(action[0] * 2) / 2
        dose = np.clip(dose, 0.0, 30.0)

        # Calcular impacto clínico
        clinical_impact = compute_clinical_impact(current_cgm, dose, carb_input, iob)

        # Cálculo de recompensa mejorada
        reward, reward_components = self._compute_enhanced_reward(
            dose, true_bolus, current_cgm, carb_input, iob, clinical_impact
        )

        # Seguimiento de métricas del episodio
        self.episode_metrics['predicted_doses'].append(dose)
        self.episode_metrics['true_doses'].append(true_bolus)
        self.episode_metrics['clinical_scores'].append(clinical_impact['safety_score'])
        
        # Seguimiento de tiempo en rango
        if 70 <= current_cgm <= 180:
            self.episode_metrics['time_in_range'].append(1)
        else:
            self.episode_metrics['time_in_range'].append(0)
            
        # Seguimiento de eventos de seguridad
        if clinical_impact['hypo_risk'] > 0.5:
            self.episode_metrics['hypo_events'] += 1
            self.episode_metrics['safety_violations'] += 1
        elif clinical_impact['hyper_risk'] > 0.5:
            self.episode_metrics['hyper_events'] += 1
            self.episode_metrics['safety_violations'] += 1

        # Avanzar episodio
        self.current_idx += 1
        terminated = self.current_idx >= self.episode_length
        truncated = False

        obs = self._get_enhanced_observation() if not terminated else np.zeros(52, dtype=np.float32)

        # Diccionario de info mejorado
        info = {
            'true_bolus': true_bolus,
            'predicted_bolus': dose,
            'current_cgm': current_cgm,
            'carb_input': carb_input,
            'iob': iob,
            'reward_components': reward_components,
            'clinical_impact': clinical_impact,
            'patient_id': self.patient_id,
            'time_in_range_episode': np.mean(self.episode_metrics['time_in_range']) if self.episode_metrics['time_in_range'] else 0.0,
            'safety_score_episode': np.mean(self.episode_metrics['clinical_scores']) if self.episode_metrics['clinical_scores'] else 1.0
        }

        return obs, reward, terminated, truncated, info

    def _compute_enhanced_reward(self, dose: float, true_bolus: float, current_cgm: float, 
                               carb_input: float, iob: float, clinical_impact: Dict) -> Tuple[float, Dict]:
        """
        Enhanced reward function with clinical considerations.
        """
        reward = 0.0
        components = {}

        # Base accuracy reward
        abs_error = abs(dose - true_bolus)
        relative_error = abs_error / max(true_bolus, 0.1)  # Avoid division by zero
        
        components['accuracy'] = -abs_error
        reward += components['accuracy']

        # Clinical safety rewards/penalties
        components['safety'] = clinical_impact['safety_score'] * 2.0
        reward += components['safety']

        # Severe hypoglycemia prevention (most critical)
        if current_cgm < 70 and dose > 0.5:
            components['hypo_prevention'] = -10.0  # Severe penalty
            reward += components['hypo_prevention']
        else:
            components['hypo_prevention'] = 0.0

        # Hyperglycemia management
        if current_cgm > 250 and dose < 1.0 and carb_input > 30:
            components['hyper_management'] = -3.0  # Penalty for under-treatment
            reward += components['hyper_management']
        else:
            components['hyper_management'] = 0.0

        # Meal bolus appropriateness
        if carb_input > 15:  # Significant meal
            expected_meal_bolus = carb_input / 15  # Rough ICR estimate
            meal_appropriateness = 1.0 - min(abs(dose - expected_meal_bolus) / expected_meal_bolus, 1.0)
            components['meal_appropriateness'] = meal_appropriateness * 1.5
            reward += components['meal_appropriateness']
        else:
            components['meal_appropriateness'] = 0.0

        # IOB consideration
        if iob > 3.0 and dose > 2.0:
            components['iob_awareness'] = -2.0  # Penalty for stacking insulin
            reward += components['iob_awareness']
        else:
            components['iob_awareness'] = 0.0

        # Time in range bonus
        if 70 <= current_cgm <= 180:
            if relative_error < 0.2:  # Good prediction while in range
                components['tir_bonus'] = 1.0
                reward += components['tir_bonus']
            else:
                components['tir_bonus'] = 0.0
        else:
            components['tir_bonus'] = 0.0

        # Conservative dosing reward (when uncertain)
        if abs_error <= 0.1 * true_bolus:  # Very accurate prediction
            components['precision_bonus'] = 2.0
            reward += components['precision_bonus']
        else:
            components['precision_bonus'] = 0.0

        return reward, components

    def _get_enhanced_observation(self) -> np.ndarray:
        """Get enhanced observation with all clinical features."""
        if self.current_idx >= len(self.df_windows):
            return np.zeros(52, dtype=np.float32)

        # Basic CGM values (24 values)
        cgm_values = [float(self.df_windows[f'cgm_{i}'][self.current_idx]) for i in range(24)]
        
        # Get extended CGM for pattern analysis if available
        extended_cgm = cgm_values  # In practice, this could be a longer history
        
        # Cyclical time encoding
        if hasattr(self.df_windows, 'timestamp'):
            timestamp = self.df_windows['timestamp'][self.current_idx]
        else:
            # Fallback to hour of day
            hour = float(self.df_windows['hour_of_day'][self.current_idx]) if 'hour_of_day' in self.df_windows.columns else 12.0
            timestamp = datetime.now().replace(hour=int(hour), minute=0, second=0)
        
        hour_sin, hour_cos = encode_time_cyclical(timestamp)

        # Basic features with log transformation for stability - ensure float conversion
        bolus_value = float(self.df_windows['bolus'][self.current_idx]) if 'bolus' in self.df_windows.columns else 0.0
        carb_input_value = float(self.df_windows['carb_input'][self.current_idx]) if 'carb_input' in self.df_windows.columns else 0.0
        iob_value = float(self.df_windows['insulin_on_board'][self.current_idx]) if 'insulin_on_board' in self.df_windows.columns else 0.0
        
        bolus_log = np.log1p(bolus_value)
        carb_log = np.log1p(carb_input_value)
        iob_log = np.log1p(iob_value)

        # Enhanced meal features - ensure float conversion
        meal_carbs = float(self.df_windows.get_column("meal_carbs").to_numpy()[self.current_idx]) if "meal_carbs" in self.df_windows.columns else 0.0
        meal_time_diff = float(self.df_windows.get_column("meal_time_diff_hours").to_numpy()[self.current_idx]) if "meal_time_diff_hours" in self.df_windows.columns else 0.0
        has_meal = float(self.df_windows.get_column("has_meal").to_numpy()[self.current_idx]) if "has_meal" in self.df_windows.columns else 0.0
        meals_in_window = float(self.df_windows.get_column("meals_in_window").to_numpy()[self.current_idx]) if "meals_in_window" in self.df_windows.columns else 0.0

        meal_carbs_log = np.log1p(meal_carbs)

        # Short-term glucose patterns - ensure float conversion
        cgm_trend = float(self.df_windows.get_column("cgm_trend").to_numpy()[self.current_idx]) if "cgm_trend" in self.df_windows.columns else 0.0
        cgm_std = float(self.df_windows.get_column("cgm_std").to_numpy()[self.current_idx]) if "cgm_std" in self.df_windows.columns else 0.0

        # Long-term glucose patterns
        glucose_patterns = compute_glucose_patterns(cgm_values, extended_cgm)

        # Additional clinical context
        current_cgm = cgm_values[0] if cgm_values else 120.0  # Most recent CGM
        
        # Risk indicators
        hypo_risk = 1.0 if current_cgm < 80 else 0.0
        hyper_risk = 1.0 if current_cgm > 200 else 0.0
        
        # Glucose rate of change
        glucose_rate = cgm_trend  # Already computed
        
        # Day of week (cyclical encoding)
        day_of_week = timestamp.weekday() / 7.0
        day_sin = np.sin(2 * np.pi * day_of_week)
        day_cos = np.cos(2 * np.pi * day_of_week)

        # Combine all features
        obs = np.concatenate([
            cgm_values,  # 24 values
            [
                # Time features (4)
                hour_sin, hour_cos, day_sin, day_cos,
                
                # Basic features (4)
                bolus_log, carb_log, iob_log, meal_carbs_log,
                
                # Meal context (4)
                meal_time_diff, has_meal, meals_in_window, 
                1.0 if meal_carbs > 15 else 0.0,  # significant_meal
                
                # Short-term glucose patterns (4)
                cgm_trend, cgm_std, hypo_risk, hyper_risk,
                
                # Long-term glucose patterns (12)
                glucose_patterns['cgm_mean_24h'] / 200.0,  # Normalize
                glucose_patterns['cgm_std_24h'] / 100.0,
                glucose_patterns['time_in_range_24h'] / 100.0,
                glucose_patterns['hypo_percentage_24h'] / 100.0,
                glucose_patterns['hyper_percentage_24h'] / 100.0,
                glucose_patterns['cv_24h'] / 100.0,
                glucose_patterns['mage_24h'] / 50.0,
                glucose_patterns['glucose_trend_24h'] / 10.0,
                glucose_patterns['cgm_range_24h'] / 300.0,
                glucose_patterns['cgm_median_24h'] / 200.0,
                glucose_patterns['hypo_episodes_24h'] / 24.0,
                glucose_patterns['hyper_episodes_24h'] / 24.0
            ]
        ])

        return obs.astype(np.float32)

class OhioT1DMInferenceEnv(gym.Env):
    """
    Enhanced inference environment with improved preprocessing.
    """
    
    def __init__(self):
        super().__init__()
        
        # Define action space (continuous bolus dose)
        self.action_space = gym.spaces.Box(
            low=0.0,
            high=30.0,
            shape=(1,),
            dtype=np.float32
        )
        
        # Enhanced observation space: 24 CGM + 16 enhanced features + 12 glucose patterns
        self.observation_space = gym.spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(52,),
            dtype=np.float32
        )
        
        # Store current observation
        self.current_observation = None
    
    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None) -> Tuple[np.ndarray, dict]:
        """Reset environment with a new observation."""
        super().reset(seed=seed)
        if self.current_observation is None:
            self.current_observation = np.zeros(52, dtype=np.float32)
        return self.current_observation, {}
    
    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """
        Execute one time step within the environment.
        During inference, we only need to return the action as the prediction.
        """
        return self.current_observation, 0.0, True, False, {'predicted_bolus': action[0]}
    
    def set_observation(self, observation: np.ndarray):
        """Set the current observation for prediction."""
        self.current_observation = observation.astype(np.float32)

def prepare_enhanced_observation(
    cgm_values: List[float],
    carb_input: float,
    iob: float,
    timestamp: datetime,
    meal_carbs: float = 0.0,
    meal_time_diff: float = 0.0,
    has_meal: float = 0.0,
    meals_in_window: int = 0,
    extended_cgm: List[float] = None
) -> np.ndarray:
    """
    Prepara una observación mejorada para el modelo PPO.
    
    Args:
        cgm_values: Valores recientes de CGM
        carb_input: Entrada de carbohidratos
        iob: Insulina activa
        timestamp: Timestamp actual
        meal_carbs: Carbohidratos de la comida
        meal_time_diff: Diferencia de tiempo con la comida
        has_meal: Indicador de presencia de comida
        meals_in_window: Número de comidas en la ventana
        extended_cgm: Valores extendidos de CGM (24h)
        
    Returns:
        Enhanced observation vector of 52 elements
    """
    # Process CGM values (24 values)
    cgm_processed = []
    for i in range(24):
        if i < len(cgm_values):
            cgm_processed.append(cgm_values[i])
        else:
            # Pad with last known value or reasonable default
            cgm_processed.append(cgm_values[-1] if cgm_values else 120.0)
    
    # Cyclical time encoding
    hour_sin, hour_cos = encode_time_cyclical(timestamp)
    
    # Day of week encoding
    day_of_week = timestamp.weekday() / 7.0
    day_sin = np.sin(2 * np.pi * day_of_week)
    day_cos = np.cos(2 * np.pi * day_of_week)
    
    # Basic features with log transformation
    carb_log = np.log1p(carb_input)
    iob_log = np.log1p(iob)
    bolus_log = 0.0  # Set to 0 during inference
    meal_carbs_log = np.log1p(meal_carbs)
    
    # Short-term glucose patterns
    if len(cgm_values) >= 2:
        cgm_trend = np.polyfit(range(len(cgm_values)), cgm_values, 1)[0]
        cgm_std = np.std(cgm_values)
    else:
        cgm_trend = 0.0
        cgm_std = 0.0
    
    # Long-term glucose patterns
    glucose_patterns = compute_glucose_patterns(cgm_values, extended_cgm)
    
    # Additional clinical context
    current_cgm = cgm_processed[0]
    hypo_risk = 1.0 if current_cgm < 80 else 0.0
    hyper_risk = 1.0 if current_cgm > 200 else 0.0
    
    # Combine all features
    obs = np.concatenate([
        cgm_processed,  # 24 values
        [
            # Time features (4)
            hour_sin, hour_cos, day_sin, day_cos,
            
            # Basic features (4)
            bolus_log, carb_log, iob_log, meal_carbs_log,
            
            # Meal context (4)
            meal_time_diff, has_meal, meals_in_window, 
            1.0 if meal_carbs > 15 else 0.0,  # significant_meal
            
            # Short-term glucose patterns (4)
            cgm_trend, cgm_std, hypo_risk, hyper_risk,
            
            # Long-term glucose patterns (12)
            glucose_patterns['cgm_mean_24h'] / 200.0,  # Normalize
            glucose_patterns['cgm_std_24h'] / 100.0,
            glucose_patterns['time_in_range_24h'] / 100.0,
            glucose_patterns['hypo_percentage_24h'] / 100.0,
            glucose_patterns['hyper_percentage_24h'] / 100.0,
            glucose_patterns['cv_24h'] / 100.0,
            glucose_patterns['mage_24h'] / 50.0,
            glucose_patterns['glucose_trend_24h'] / 10.0,
            glucose_patterns['cgm_range_24h'] / 300.0,
            glucose_patterns['cgm_median_24h'] / 200.0,
            glucose_patterns['hypo_episodes_24h'] / 24.0,
            glucose_patterns['hyper_episodes_24h'] / 24.0
        ]
    ])
    
    return obs.astype(np.float32)

def safe_bolus_prediction(predicted_dose: float, current_cgm: float, carbs: float = 0, iob: float = 0, max_dose: float = 20.0, min_cgm: float = 80.0) -> float:
    """
    Ajusta la dosis predicha para maximizar la seguridad clínica.
    - Si la glucosa actual < min_cgm, devuelve 0 y loggea advertencia.
    - Simula el efecto de la dosis. Si la glucosa futura estimada < min_cgm o hay riesgo de hipo, reduce la dosis.
    - Limita la dosis máxima.
    """
    if current_cgm < min_cgm:
        logging.warning(f"[Seguridad] Glucosa actual {current_cgm:.1f} < {min_cgm}. Dosis forzada a 0.")
        return 0.0
    # Limitar dosis máxima
    dose = min(predicted_dose, max_dose)
    # Simular efecto clínico
    clinical = compute_clinical_impact(current_cgm, dose, carbs, iob)
    # Si hay riesgo de hipo, reducir dosis iterativamente
    step = 0.5
    while (clinical['estimated_future_glucose'] < min_cgm or clinical['hypo_risk'] > 0) and dose > 0:
        dose = max(0.0, dose - step)
        clinical = compute_clinical_impact(current_cgm, dose, carbs, iob)
        logging.info(f"[Seguridad] Ajustando dosis a {dose:.2f} por riesgo de hipo/futuro {clinical['estimated_future_glucose']:.1f} mg/dL")
    if dose < predicted_dose:
        logging.info(f"[Seguridad] Dosis ajustada de {predicted_dose:.2f} a {dose:.2f} por validación clínica.")
    return dose

def predict_insulin_dose(
    model_path: str,
    observation: np.ndarray,
    current_cgm: float = None,
    carbs: float = 0,
    iob: float = 0,
    max_dose: float = 20.0,
    min_cgm: float = 80.0
) -> float:
    """
    Predice la dosis de insulina usando un modelo PPO entrenado y la valida clínicamente.
    """
    # Load trained model
    model = PPO.load(model_path)
    # Create inference environment
    env = OhioT1DMInferenceEnv()
    env.set_observation(observation)
    # Get prediction
    obs, _ = env.reset()
    action, _ = model.predict(obs, deterministic=True)
    predicted_dose = float(action[0])
    # Si se proveen datos de contexto, validar seguridad
    if current_cgm is not None:
        safe_dose = safe_bolus_prediction(predicted_dose, current_cgm, carbs, iob, max_dose, min_cgm)
        return safe_dose
    return predicted_dose

def evaluate_clinical_metrics(predictions: List[float], actuals: List[float], 
                            cgm_values: List[List[float]] = None) -> Dict[str, float]:
    """
    Calcula métricas clínicas de evaluación integral.
    
    Args:
        predictions: Lista de dosis de bolo predichas
        actuals: Lista de dosis de bolo reales
        cgm_values: Lista de secuencias de valores CGM para cálculo de TIR
        
    Returns:
        Diccionario con métricas clínicas y estadísticas
    """
    predictions = np.array(predictions)
    actuals = np.array(actuals)
    
    # Métricas estadísticas básicas
    mae = np.mean(np.abs(predictions - actuals))
    mse = np.mean((predictions - actuals) ** 2)
    rmse = np.sqrt(mse)
    mape = np.mean(np.abs((predictions - actuals) / np.maximum(actuals, 0.1))) * 100
    
    # Análisis de error
    error = predictions - actuals
    overdose_count = np.sum(error > 0)
    underdose_count = np.sum(error < 0)
    overdose_rate = (overdose_count / len(error)) * 100
    underdose_rate = (underdose_count / len(error)) * 100
    
    # Métricas de seguridad clínica
    # Porcentaje dentro del rango seguro (±20% de la dosis real)
    safe_range_20 = np.abs(error) <= (0.2 * np.maximum(actuals, 0.1))
    safe_rate_20 = np.mean(safe_range_20) * 100
    
    # Porcentaje dentro del rango estrecho (±10% de la dosis real)
    safe_range_10 = np.abs(error) <= (0.1 * np.maximum(actuals, 0.1))
    safe_rate_10 = np.mean(safe_range_10) * 100
    
    # Análisis de error severo
    severe_overdose = np.sum(error > 2.0)  # Más de 2 unidades por encima
    severe_underdose = np.sum(error < -2.0)  # Más de 2 unidades por debajo
    severe_error_rate = ((severe_overdose + severe_underdose) / len(error)) * 100
    
    # Apropiación clínica
    # Predicciones de bolo pequeño (< 1 unidad) deben ser precisas
    small_bolus_mask = actuals < 1.0
    if np.any(small_bolus_mask):
        small_bolus_mae = np.mean(np.abs(error[small_bolus_mask]))
    else:
        small_bolus_mae = 0.0
    
    # Predicciones de bolo grande (> 5 unidades) deben ser conservadoras
    large_bolus_mask = actuals > 5.0
    if np.any(large_bolus_mask):
        large_bolus_mae = np.mean(np.abs(error[large_bolus_mask]))
        large_bolus_conservatism = np.mean(error[large_bolus_mask])  # Negativo = conservador
    else:
        large_bolus_mae = 0.0
        large_bolus_conservatism = 0.0
    
    metrics = {
        # Métricas estadísticas
        'mae': mae,
        'mse': mse,
        'rmse': rmse,
        'mape': mape,
        
        # Distribución de error
        'overdose_rate': overdose_rate,
        'underdose_rate': underdose_rate,
        'mean_error': np.mean(error),
        'std_error': np.std(error),
        
        # Métricas de seguridad
        'safe_rate_20': safe_rate_20,
        'safe_rate_10': safe_rate_10,
        'severe_error_rate': severe_error_rate,
        'severe_overdose_count': severe_overdose,
        'severe_underdose_count': severe_underdose,
        
        # Apropiación clínica
        'small_bolus_mae': small_bolus_mae,
        'large_bolus_mae': large_bolus_mae,
        'large_bolus_conservatism': large_bolus_conservatism,
        
        # Resumen de datos
        'total_predictions': len(predictions),
        'mean_predicted_dose': np.mean(predictions),
        'mean_actual_dose': np.mean(actuals),
    }
    
    # Cálculo de Tiempo en Rango si se proporcionan datos CGM
    if cgm_values is not None:
        tir_scores = []
        hypo_percentages = []
        hyper_percentages = []
        
        for cgm_sequence in cgm_values:
            if len(cgm_sequence) > 0:
                cgm_array = np.array(cgm_sequence)
                
                # Tiempo en rango (70-180 mg/dL)
                in_range = np.sum((cgm_array >= 70) & (cgm_array <= 180))
                tir = (in_range / len(cgm_array)) * 100
                tir_scores.append(tir)
                
                # Porcentaje de hipoglucemia
                hypo_count = np.sum(cgm_array < 70)
                hypo_pct = (hypo_count / len(cgm_array)) * 100
                hypo_percentages.append(hypo_pct)
                
                # Porcentaje de hiperglucemia
                hyper_count = np.sum(cgm_array > 180)
                hyper_pct = (hyper_count / len(cgm_array)) * 100
                hyper_percentages.append(hyper_pct)
        
        if tir_scores:
            metrics['time_in_range_mean'] = np.mean(tir_scores)
            metrics['time_in_range_std'] = np.std(tir_scores)
            metrics['hypo_percentage_mean'] = np.mean(hypo_percentages)
            metrics['hyper_percentage_mean'] = np.mean(hyper_percentages)
        else:
            metrics['time_in_range_mean'] = 0.0
            metrics['time_in_range_std'] = 0.0
            metrics['hypo_percentage_mean'] = 0.0
            metrics['hyper_percentage_mean'] = 0.0
    
    return metrics

def train_patient_specific_model(
    patient_data_path: str,
    base_model_path: str = None,
    output_dir: str = None,
    patient_id: str = None,
    total_timesteps: int = 100000
) -> PPO:
    """
    Entrena un modelo específico para un paciente usando transfer learning.
    
    Args:
        patient_data_path: Ruta a los datos de entrenamiento específicos del paciente
        base_model_path: Ruta al modelo base pre-entrenado (opcional)
        output_dir: Directorio para guardar el modelo específico del paciente
        patient_id: Identificador del paciente
        total_timesteps: Número de pasos de entrenamiento
        
    Returns:
        Modelo PPO específico del paciente entrenado
    """
    # Cargar datos del paciente
    df_windows = pl.read_parquet(patient_data_path)
    df_final = pl.read_parquet(patient_data_path)
    
    logging.info(f"Entrenando modelo específico para paciente {patient_id}")
    logging.info(f"Forma de datos del paciente: {df_windows.shape}")
    
    # Crear entorno
    def make_patient_env():
        env = OhioT1DMEnhancedEnv(df_windows, df_final, patient_id=patient_id)
        env = Monitor(env)
        return env
    
    vec_env = DummyVecEnv([make_patient_env])
    
    # Inicializar modelo
    if base_model_path and os.path.exists(base_model_path):
        # Cargar modelo pre-entrenado para transfer learning
        logging.info(f"Cargando modelo base desde {base_model_path}")
        model = PPO.load(base_model_path, env=vec_env)
        # Reducir tasa de aprendizaje para fine-tuning
        model.learning_rate = 1e-6
    else:
        # Crear nuevo modelo
        logging.info("Creando nuevo modelo específico para paciente")
        model = PPO(
            'MlpPolicy',
            vec_env,
            learning_rate=1e-5,
            n_steps=2048,
            batch_size=128,
            n_epochs=10,
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=0.1,
            ent_coef=0.01,
            vf_coef=0.5,
            max_grad_norm=0.5,
            policy_kwargs=dict(
                net_arch=[dict(pi=[128, 128], vf=[128, 128])]
            ),
            verbose=1
        )
    
    # Entrenar modelo
    logging.info(f"Entrenando modelo específico para paciente por {total_timesteps} pasos")
    model.learn(total_timesteps=total_timesteps, progress_bar=True)
    
    # Guardar modelo
    if output_dir:
        model_path = f"{output_dir}/ppo_patient_{patient_id}"
        model.save(model_path)
        logging.info(f"Modelo específico para paciente guardado en {model_path}")
    
    return model

def evaluate_patient_models(
    patient_models: Dict[str, str],
    test_data_paths: Dict[str, str],
    general_model_path: str = None
) -> Dict[str, Dict[str, float]]:
    """
    Evalúa modelos específicos de pacientes vs modelo general.
    
    Args:
        patient_models: Diccionario que mapea patient_id a model_path
        test_data_paths: Diccionario que mapea patient_id a test_data_path
        general_model_path: Ruta al modelo general para comparación
        
    Returns:
        Diccionario con resultados de evaluación para cada paciente
    """
    results = {}
    
    for patient_id, test_path in test_data_paths.items():
        logging.info(f"Evaluando paciente {patient_id}")
        
        # Cargar datos de prueba
        df_windows = pl.read_parquet(test_path)
        df_final = pl.read_parquet(test_path)
        
        patient_results = {'patient_id': patient_id}
        
        # Probar modelo específico del paciente si está disponible
        if patient_id in patient_models:
            patient_model_path = patient_models[patient_id]
            if os.path.exists(patient_model_path + '.zip'):
                logging.info(f"Probando modelo específico del paciente: {patient_model_path}")
                
                env = OhioT1DMEnhancedEnv(df_windows, df_final, patient_id=patient_id)
                model = PPO.load(patient_model_path)
                
                obs, _ = env.reset()
                done = False
                predictions = []
                actuals = []
                cgm_sequences = []
                
                while not done:
                    action, _ = model.predict(obs, deterministic=True)
                    obs, _, terminated, truncated, info = env.step(action)
                    done = terminated or truncated
                    
                    predictions.append(info['predicted_bolus'])
                    actuals.append(info['true_bolus'])
                    
                    # Recolectar secuencia CGM para esta predicción
                    cgm_seq = [env.df_windows[f'cgm_{i}'][env.current_idx-1] for i in range(24)]
                    cgm_sequences.append(cgm_seq)
                
                # Calcular métricas
                patient_metrics = evaluate_clinical_metrics(predictions, actuals, cgm_sequences)
                patient_results['patient_specific'] = patient_metrics
        
        # Probar modelo general si está disponible
        if general_model_path and os.path.exists(general_model_path + '.zip'):
            logging.info(f"Probando modelo general: {general_model_path}")
            
            env = OhioT1DMEnhancedEnv(df_windows, df_final, patient_id=patient_id)
            model = PPO.load(general_model_path)
            
            obs, _ = env.reset()
            done = False
            predictions = []
            actuals = []
            cgm_sequences = []
            
            while not done:
                action, _ = model.predict(obs, deterministic=True)
                obs, _, terminated, truncated, info = env.step(action)
                done = terminated or truncated
                
                predictions.append(info['predicted_bolus'])
                actuals.append(info['true_bolus'])
                
                # Recolectar secuencia CGM
                cgm_seq = [env.df_windows[f'cgm_{i}'][env.current_idx-1] for i in range(24)]
                cgm_sequences.append(cgm_seq)
            
            # Calcular métricas
            general_metrics = evaluate_clinical_metrics(predictions, actuals, cgm_sequences)
            patient_results['general'] = general_metrics
        
        results[patient_id] = patient_results
    
    return results

def predict_from_preprocessed(
    model_path: str,
    preprocessed_data_path: str
) -> Dict[str, Union[List[float], float]]:
    """
    Realiza predicciones usando archivos de datos preprocesados con métricas mejoradas.
    
    Args:
        model_path: Ruta al modelo PPO entrenado
        preprocessed_data_path: Ruta al archivo parquet preprocesado
        
    Returns:
        Diccionario con predicciones, valores reales y métricas clínicas
    """
    try:
        # Cargar datos preprocesados
        df_windows = pl.read_parquet(preprocessed_data_path)
        df_final = pl.read_parquet(preprocessed_data_path)
        
        # Verificar y convertir tipos de datos
        numeric_columns = ['bolus', 'carb_input', 'insulin_on_board']
        for col in numeric_columns:
            if col in df_windows.columns:
                try:
                    df_windows = df_windows.with_columns(pl.col(col).cast(pl.Float64))
                    df_final = df_final.with_columns(pl.col(col).cast(pl.Float64))
                except Exception as e:
                    logging.error(f"Error al convertir columna {col}: {str(e)}")
                    raise
        
        # Verificar columnas CGM
        cgm_columns = [f'cgm_{i}' for i in range(24)]
        for col in cgm_columns:
            if col in df_windows.columns:
                try:
                    df_windows = df_windows.with_columns(pl.col(col).cast(pl.Float64))
                    df_final = df_final.with_columns(pl.col(col).cast(pl.Float64))
                except Exception as e:
                    logging.error(f"Error al convertir columna {col}: {str(e)}")
                    raise
        
        # Crear entorno
        env = OhioT1DMEnhancedEnv(df_windows, df_final)
        
        # Cargar modelo
        model = PPO.load(model_path)
        
        # Ejecutar predicciones
        obs, _ = env.reset()
        done = False
        predictions = []
        true_values = []
        cgm_sequences = []
        
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, _, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            
            # Asegurar que los valores sean float
            predicted_dose = float(info['predicted_bolus'])
            true_bolus = float(info['true_bolus'])
            
            predictions.append(predicted_dose)
            true_values.append(true_bolus)
            
            # Recolectar secuencia CGM para esta predicción
            if env.current_idx > 0:
                cgm_seq = [float(env.df_windows[f'cgm_{i}'][env.current_idx-1]) for i in range(24)]
                cgm_sequences.append(cgm_seq)
        
        # Calcular métricas mejoradas
        clinical_metrics = evaluate_clinical_metrics(predictions, true_values, cgm_sequences)
        
        return {
            'predictions': predictions,
            'true_values': true_values,
            'clinical_metrics': clinical_metrics,
            'mae': clinical_metrics['mae']
        }
        
    except Exception as e:
        logging.error(f"Error en predict_from_preprocessed: {str(e)}")
        raise

def make_env(df_windows: pl.DataFrame, df_final: pl.DataFrame, rank: int, seed: int = 0) -> gym.Env:
    """
    Crea un entorno envuelto para entrenamiento paralelo.
    
    Args:
        df_windows: DataFrame con ventanas de CGM
        df_final: DataFrame con características finales
        rank: Rango del entorno
        seed: Semilla aleatoria
        
    Returns:
        Entorno envuelto
    """
    def _init():
        env = OhioT1DMEnhancedEnv(df_windows, df_final)
        env.reset(seed=seed + rank)  # Establecer semilla durante el reinicio
        env = Monitor(env)
        return env
    return _init

class ProgressBarCallback(BaseCallback):
    """
    Callback personalizado para mostrar una barra de progreso durante el entrenamiento.
    """
    def __init__(self, total_timesteps: int, verbose: int = 0):
        super().__init__(verbose)
        self.total_timesteps = total_timesteps
        self.pbar = None
        self.start_time = None
        self.last_time = None
        self.last_timestep = 0
        
    def _on_training_start(self):
        """Inicializa la barra de progreso al inicio del entrenamiento."""
        self.pbar = tqdm(total=self.total_timesteps, 
                        desc="Entrenando modelo PPO",
                        unit="steps",
                        bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]')
        self.start_time = time.time()
        self.last_time = self.start_time
        
    def _on_step(self) -> bool:
        """Actualiza la barra de progreso en cada paso."""
        if self.pbar is not None:
            # Calcular métricas
            current_time = time.time()
            steps_since_last = self.num_timesteps - self.last_timestep
            time_since_last = current_time - self.last_time
            
            if time_since_last > 0:
                steps_per_second = steps_since_last / time_since_last
                remaining_steps = self.total_timesteps - self.num_timesteps
                estimated_time_remaining = remaining_steps / steps_per_second if steps_per_second > 0 else 0
                
                # Actualizar descripción con métricas
                self.pbar.set_description(
                    f"Entrenando modelo PPO | "
                    f"Velocidad: {steps_per_second:.1f} steps/s | "
                    f"Tiempo restante: {timedelta(seconds=int(estimated_time_remaining))}"
                )
            
            # Actualizar barra de progreso
            self.pbar.update(steps_since_last)
            self.last_timestep = self.num_timesteps
            self.last_time = current_time
            
        return True
    
    def _on_training_end(self):
        """Cierra la barra de progreso al finalizar el entrenamiento."""
        if self.pbar is not None:
            self.pbar.close()
            total_time = time.time() - self.start_time
            logging.info(f"Entrenamiento completado en {timedelta(seconds=int(total_time))}")

def train_with_enhanced_hyperparameters(
    train_files: List[str],
    output_dir: str,
    tensorboard_log: str,
    total_timesteps: int,
    n_envs: int = 4,
    learning_rate: float = 1e-5,
    batch_size: int = 256,
    net_arch: List[int] = [256, 256, 128],
    gamma: float = 0.99,
    n_steps: int = 4096,
    n_epochs: int = 20,
    clip_range: float = 0.2,
    ent_coef: float = 0.01
) -> PPO:
    """
    Entrena un modelo PPO con hiperparámetros mejorados y arquitectura de red optimizada.
    
    Args:
        train_files: Lista de rutas a archivos de entrenamiento
        output_dir: Directorio para guardar el modelo
        tensorboard_log: Directorio para logs de tensorboard
        total_timesteps: Número total de pasos de tiempo para entrenar
        n_envs: Número de entornos paralelos
        learning_rate: Tasa de aprendizaje
        batch_size: Tamaño del batch
        net_arch: Arquitectura de la red neuronal
        gamma: Factor de descuento
        n_steps: Número de pasos por actualización
        n_epochs: Número de épocas por actualización
        clip_range: Rango de clip para la actualización de política
        ent_coef: Coeficiente de entropía
        
    Returns:
        Modelo PPO entrenado
    """
    # Crear directorio de salida
    os.makedirs(output_dir, exist_ok=True)
    
    # Columnas que deben mantenerse como string (identificadores)
    id_columns = {'SubjectID', 'patient_id', 'id'}
    
    # Cargar y combinar datos de entrenamiento
    train_dfs = []
    for file in train_files:
        logging.info(f"Cargando archivo: {file}")
        df = pl.read_parquet(file)
        
        # Asegurar tipos de datos consistentes
        for col in df.columns:
            if df[col].dtype == pl.String and col not in id_columns:
                # Convertir columnas string a float si contienen números
                try:
                    df = df.with_columns(pl.col(col).cast(pl.Float64))
                    logging.info(f"Columna {col} convertida a Float64")
                except:
                    logging.info(f"Columna {col} mantenida como String (no numérica)")
        
        train_dfs.append(df)
        logging.info(f"Archivo {file} procesado. Forma: {df.shape}")
    
    # Verificar esquemas antes de concatenar
    if len(train_dfs) > 1:
        first_schema = train_dfs[0].schema
        for i, df in enumerate(train_dfs[1:], 1):
            if df.schema != first_schema:
                logging.info(f"Alineando esquema del archivo {i}")
                # Intentar alinear esquemas
                for col in df.columns:
                    if col in first_schema and df[col].dtype != first_schema[col] and col not in id_columns:
                        try:
                            df = df.with_columns(pl.col(col).cast(first_schema[col]))
                            logging.info(f"Columna {col} alineada con el esquema principal")
                        except:
                            logging.warning(f"No se pudo alinear el tipo de la columna {col}")
    
    # Concatenar DataFrames
    logging.info("Concatenando DataFrames...")
    train_df = pl.concat(train_dfs)
    logging.info(f"DataFrame concatenado final. Forma: {train_df.shape}")

    # Loggear estadísticas descriptivas del dataset de entrenamiento combinado
    if not train_df.is_empty():
        logging.info("Estadísticas descriptivas del dataset de entrenamiento combinado:")
        # Seleccionar columnas numéricas relevantes para el análisis. Excluir identificadores y ventanas CGM expandidas.
        relevant_cols_for_stats = [
            col for col in train_df.columns 
            if train_df[col].dtype in [pl.Float64, pl.Int64] and 
            not col.startswith('cgm_') and 
            col not in ['SubjectID', 'timestamp'] # Añadir otros IDs o no numéricos si es necesario
        ]
        if relevant_cols_for_stats:
            desc_stats = train_df.select(relevant_cols_for_stats).describe()
            logging.info(f"\n{desc_stats}")
            # Loggear distribución de eventos de hipo/hiper si existen
            if 'hypo_episodes_24h' in train_df.columns:
                logging.info(f"Distribución de episodios de hipoglucemia (24h):\n{train_df['hypo_episodes_24h'].value_counts().sort('count', descending=True)}")
            if 'hyper_episodes_24h' in train_df.columns:
                logging.info(f"Distribución de episodios de hiperglucemia (24h):\n{train_df['hyper_episodes_24h'].value_counts().sort('count', descending=True)}")
            if 'bolus' in train_df.columns:
                logging.info(f"Distribución de dosis de bolus (cuantiles):")
                logging.info(train_df.select(pl.col('bolus').quantile(q).alias(f"bolus_q{int(q*100)}") for q in [0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99]))
            if 'carb_input' in train_df.columns:
                logging.info(f"Distribución de carbohidratos (cuantiles):")
                logging.info(train_df.select(pl.col('carb_input').quantile(q).alias(f"carbs_q{int(q*100)}") for q in [0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99]))
        else:
            logging.info("No se encontraron columnas numéricas relevantes para estadísticas descriptivas.")
    else:
        logging.warning("El DataFrame de entrenamiento está vacío. No se pueden calcular estadísticas.")
    
    # Crear entornos paralelos
    logging.info("Creando entornos paralelos...")
    env = SubprocVecEnv([
        make_env(train_df, train_df, i) for i in range(n_envs)
    ])
    
    # Configurar callbacks
    eval_callback = EvalCallback(
        env,
        best_model_save_path=output_dir,
        log_path=output_dir,
        eval_freq=10000,
        n_eval_episodes=5,
        deterministic=True,
        render=False
    )
    
    progress_callback = ProgressBarCallback(total_timesteps=total_timesteps)
    
    # Configurar modelo PPO con hiperparámetros mejorados
    logging.info("Configurando modelo PPO...")
    model = PPO(
        "MlpPolicy",
        env,
        learning_rate=learning_rate,
        n_steps=n_steps,
        batch_size=batch_size,
        n_epochs=n_epochs,
        gamma=gamma,
        gae_lambda=0.95,
        clip_range=clip_range,
        ent_coef=ent_coef,
        vf_coef=0.5,
        max_grad_norm=0.5,
        use_sde=False,
        sde_sample_freq=-1,
        target_kl=None,
        tensorboard_log=tensorboard_log,
        policy_kwargs=dict(
            net_arch=net_arch
        ),
        device='cpu',  # Forzar uso de CPU
        verbose=0  # Desactivar la barra de progreso por defecto
    )
    
    # Entrenar modelo
    logging.info("Iniciando entrenamiento del modelo...")
    model.learn(
        total_timesteps=total_timesteps,
        callback=[eval_callback, progress_callback]
    )
    
    # Guardar modelo final
    model_path = os.path.join(output_dir, 'final_model')
    model.save(model_path)
    logging.info(f"Modelo guardado en: {model_path}")
    
    return model

def test_enhanced_user_inputs(
    model_path: str,
    test_cases: List[Dict[str, Union[List[float], float, datetime]]]
) -> Dict[str, Dict[str, float]]:
    """
    Prueba el modelo con casos de prueba personalizados que incluyen entradas de usuario.
    
    Args:
        model_path: Ruta al modelo PPO entrenado
        test_cases: Lista de diccionarios con casos de prueba
        
    Returns:
        Diccionario con resultados de predicción y métricas clínicas para cada caso
    """
    # Cargar modelo
    try:
        model = PPO.load(model_path)
    except FileNotFoundError:
        logging.error(f"No se pudo encontrar el modelo en: {model_path}")
        logging.error("Asegúrese de que el modelo existe y la ruta es correcta.")
        return {}
    
    results = {}
    
    for i, test_case in enumerate(test_cases):
        # Preparar observación
        observation = prepare_enhanced_observation(
            cgm_values=test_case['cgm_values'],
            carb_input=test_case['carb_input'],
            iob=test_case['iob'],
            timestamp=test_case['timestamp'],
            meal_carbs=test_case.get('meal_carbs', 0.0),
            meal_time_diff=test_case.get('meal_time_diff', 0.0),
            has_meal=test_case.get('has_meal', 0.0),
            meals_in_window=test_case.get('meals_in_window', 0),
            extended_cgm=test_case.get('extended_cgm', None)
        )
        
        # Realizar predicción
        action, _ = model.predict(observation, deterministic=True)
        predicted_dose = float(action[0])
        
        # Calcular impacto clínico
        current_cgm = float(test_case['cgm_values'][0])
        clinical_impact = compute_clinical_impact(
            current_cgm=current_cgm,
            predicted_dose=predicted_dose,
            carbs=float(test_case['carb_input']),
            iob=float(test_case['iob'])
        )
        
        # Almacenar resultados
        results[f'case_{i}'] = {
            'predicted_dose': predicted_dose,
            'current_cgm': current_cgm,
            'clinical_impact': clinical_impact,
            'estimated_future_glucose': clinical_impact['estimated_future_glucose'],
            'hypo_risk': clinical_impact['hypo_risk'],
            'hyper_risk': clinical_impact['hyper_risk'],
            'safety_score': clinical_impact['safety_score']
        }
    
    return results

def plot_enhanced_training(tensorboard_log: str, output_dir: str):
    """
    Genera gráficos mejorados del entrenamiento usando datos de TensorBoard.
    
    Args:
        tensorboard_log: Directorio con logs de TensorBoard
        output_dir: Directorio para guardar los gráficos
    """
    try:
        # Crear directorio de salida
        os.makedirs(output_dir, exist_ok=True)
        
        # Cargar datos de TensorBoard
        ea = event_accumulator.EventAccumulator(
            tensorboard_log,
            size_guidance={
                event_accumulator.SCALARS: 0,
            }
        )
        ea.Reload()
        
        # Obtener métricas de entrenamiento con manejo de errores
        metrics = {}
        metric_names = {
            'train/reward': 'Recompensa de Entrenamiento',
            'eval/mean_reward': 'Recompensa de Evaluación',
            'train/loss': 'Pérdida Total',
            'train/value_loss': 'Pérdida de Valor',
            'train/policy_loss': 'Pérdida de Política'
        }
        
        for metric_key, metric_name in metric_names.items():
            try:
                metrics[metric_key] = ea.Scalars(metric_key)
                logging.info(f"Métrica encontrada: {metric_name}")
            except KeyError:
                logging.warning(f"Métrica no encontrada: {metric_name}")
                metrics[metric_key] = []
        
        # Verificar si hay datos para graficar
        if not any(metrics.values()):
            logging.warning("No se encontraron métricas de entrenamiento para graficar")
            return
        
        # Crear figura con subplots
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Métricas de Entrenamiento Mejoradas', fontsize=16)
        
        # Gráfico de recompensa
        ax = axes[0, 0]
        if metrics['train/reward'] and metrics['eval/mean_reward']:
            ax.plot([x.step for x in metrics['train/reward']], 
                   [x.value for x in metrics['train/reward']], 
                   label='Entrenamiento')
            ax.plot([x.step for x in metrics['eval/mean_reward']], 
                   [x.value for x in metrics['eval/mean_reward']], 
                   label='Evaluación')
            ax.set_title('Recompensa Media')
            ax.set_xlabel('Pasos')
            ax.set_ylabel('Recompensa')
            ax.legend()
            ax.grid(True)
        else:
            ax.text(0.5, 0.5, 'Datos de recompensa no disponibles', 
                   ha='center', va='center', transform=ax.transAxes)
        
        # Gráfico de pérdida total
        ax = axes[0, 1]
        if metrics['train/loss']:
            ax.plot([x.step for x in metrics['train/loss']], 
                   [x.value for x in metrics['train/loss']])
            ax.set_title('Pérdida Total')
            ax.set_xlabel('Pasos')
            ax.set_ylabel('Pérdida')
            ax.grid(True)
        else:
            ax.text(0.5, 0.5, 'Datos de pérdida total no disponibles', 
                   ha='center', va='center', transform=ax.transAxes)
        
        # Gráfico de pérdida de valor
        ax = axes[1, 0]
        if metrics['train/value_loss']:
            ax.plot([x.step for x in metrics['train/value_loss']], 
                   [x.value for x in metrics['train/value_loss']])
            ax.set_title('Pérdida de Valor')
            ax.set_xlabel('Pasos')
            ax.set_ylabel('Pérdida')
            ax.grid(True)
        else:
            ax.text(0.5, 0.5, 'Datos de pérdida de valor no disponibles', 
                   ha='center', va='center', transform=ax.transAxes)
        
        # Gráfico de pérdida de política
        ax = axes[1, 1]
        if metrics['train/policy_loss']:
            ax.plot([x.step for x in metrics['train/policy_loss']], 
                   [x.value for x in metrics['train/policy_loss']])
            ax.set_title('Pérdida de Política')
            ax.set_xlabel('Pasos')
            ax.set_ylabel('Pérdida')
            ax.grid(True)
        else:
            ax.text(0.5, 0.5, 'Datos de pérdida de política no disponibles', 
                   ha='center', va='center', transform=ax.transAxes)
        
        # Ajustar layout y guardar
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'training_metrics.png'))
        plt.close()
        
        # Crear gráfico de distribución de recompensas si hay datos disponibles
        if metrics['train/reward']:
            plt.figure(figsize=(10, 6))
            sns.histplot([x.value for x in metrics['train/reward']], bins=50)
            plt.title('Distribución de Recompensas de Entrenamiento')
            plt.xlabel('Recompensa')
            plt.ylabel('Frecuencia')
            plt.grid(True)
            plt.savefig(os.path.join(output_dir, 'reward_distribution.png'))
            plt.close()
            
            # Crear gráfico de tendencia de recompensa
            plt.figure(figsize=(10, 6))
            train_rewards = [x.value for x in metrics['train/reward']]
            window_size = min(100, len(train_rewards))
            if window_size > 0:
                moving_avg = np.convolve(train_rewards, np.ones(window_size)/window_size, mode='valid')
                plt.plot(moving_avg)
                plt.title('Tendencia de Recompensa (Media Móvil)')
                plt.xlabel('Pasos')
                plt.ylabel('Recompensa Media')
                plt.grid(True)
                plt.savefig(os.path.join(output_dir, 'reward_trend.png'))
                plt.close()
        
        logging.info("Gráficos de entrenamiento generados exitosamente")
        
    except Exception as e:
        logging.error(f"Error al generar gráficos de entrenamiento: {str(e)}")
        logging.error("Continuando con la ejecución...")

def plot_clinical_results(
    results: Dict[str, Dict[str, float]],
    output_dir: str
):
    """
    Genera gráficos de evaluación clínica integral.
    """
    try:
        # Extraer métricas de los resultados
        patients = list(results.keys())
        
        # Preparar datos para graficar
        mae_values = []
        tir_values = []
        safe_rate_values = []
        hypo_risk_values = []
        
        for patient in patients:
            if 'clinical_metrics' in results[patient]:
                metrics = results[patient]['clinical_metrics']
                mae_values.append(metrics.get('mae', 0))
                tir_values.append(metrics.get('time_in_range_mean', 0))
                safe_rate_values.append(metrics.get('safe_rate_20', 0))
                hypo_risk_values.append(metrics.get('hypo_percentage_mean', 0))
            else:
                mae_values.append(0)
                tir_values.append(0)
                safe_rate_values.append(0)
                hypo_risk_values.append(0)
        
        # Crear subplots
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
        
        # MAE por paciente
        bars1 = ax1.bar(patients, mae_values, color='skyblue', alpha=0.7)
        ax1.set_title('Error Absoluto Medio (MAE) por Paciente')
        ax1.set_xlabel('Paciente')
        ax1.set_ylabel('MAE (unidades)')
        ax1.tick_params(axis='x', rotation=45)
        ax1.grid(True, linestyle='--', alpha=0.3)
        
        # Etiquetas de valor
        for bar, value in zip(bars1, mae_values):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height,
                    f'{value:.3f}', ha='center', va='bottom')
        
        # Tiempo en Rango
        bars2 = ax2.bar(patients, tir_values, color='lightgreen', alpha=0.7)
        ax2.set_title('Tiempo en Rango (70-180 mg/dL)')
        ax2.set_xlabel('Paciente')
        ax2.set_ylabel('TIR (%)')
        ax2.tick_params(axis='x', rotation=45)
        ax2.grid(True, linestyle='--', alpha=0.3)
        ax2.axhline(y=70, color='red', linestyle='--', alpha=0.5, label='Objetivo: >70%')
        ax2.legend()
        
        # Etiquetas de valor
        for bar, value in zip(bars2, tir_values):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height,
                    f'{value:.1f}%', ha='center', va='bottom')
        
        # Tasa de predicción segura
        bars3 = ax3.bar(patients, safe_rate_values, color='orange', alpha=0.7)
        ax3.set_title('Tasa de Predicción Segura (±20% de la real)')
        ax3.set_xlabel('Paciente')
        ax3.set_ylabel('Tasa Segura (%)')
        ax3.tick_params(axis='x', rotation=45)
        ax3.grid(True, linestyle='--', alpha=0.3)
        
        # Etiquetas de valor
        for bar, value in zip(bars3, safe_rate_values):
            height = bar.get_height()
            ax3.text(bar.get_x() + bar.get_width()/2., height,
                    f'{value:.1f}%', ha='center', va='bottom')
        
        # Riesgo de hipoglucemia
        bars4 = ax4.bar(patients, hypo_risk_values, color='red', alpha=0.7)
        ax4.set_title('Riesgo de Hipoglucemia (%)')
        ax4.set_xlabel('Paciente')
        ax4.set_ylabel('Riesgo Hipo (%)')
        ax4.tick_params(axis='x', rotation=45)
        ax4.grid(True, linestyle='--', alpha=0.3)
        ax4.axhline(y=4, color='red', linestyle='--', alpha=0.5, label='Objetivo: <4%')
        ax4.legend()
        
        # Etiquetas de valor
        for bar, value in zip(bars4, hypo_risk_values):
            height = bar.get_height()
            ax4.text(bar.get_x() + bar.get_width()/2., height,
                    f'{value:.1f}%', ha='center', va='bottom')
        
        plt.tight_layout()
        plt.savefig(f"{output_dir}/clinical_evaluation_results.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        # Gráfico resumen adicional
        fig, ax = plt.subplots(1, 1, figsize=(10, 6))
        
        # Métricas resumen generales
        avg_mae = np.mean(mae_values)
        avg_tir = np.mean(tir_values)
        avg_safe_rate = np.mean(safe_rate_values)
        avg_hypo_risk = np.mean(hypo_risk_values)
        
        metrics_names = ['MAE\n(unidades)', 'TIR\n(%)', 'Tasa Segura\n(%)', 'Riesgo Hipo\n(%)']
        metrics_values = [avg_mae, avg_tir, avg_safe_rate, avg_hypo_risk]
        colors = ['skyblue', 'lightgreen', 'orange', 'red']
        
        bars = ax.bar(metrics_names, metrics_values, color=colors, alpha=0.7)
        ax.set_title('Resumen General de Desempeño Clínico')
        ax.set_ylabel('Valor de Métrica')
        
        # Etiquetas de valor
        for bar, value in zip(bars, metrics_values):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                    f'{value:.2f}', ha='center', va='bottom', fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(f"{output_dir}/clinical_summary.png", dpi=300, bbox_inches='tight')
        plt.close()
        
    except Exception as e:
        logging.error(f"Error al graficar los resultados clínicos: {e}")

def test_elderly_patient_cases(model_path: str):
    """
    Testea el modelo con casos de una paciente anciana sedentaria con reglas fijas de dosis.
    """
    from datetime import datetime
    elderly_cases = [
        # Glucosa < 150 mg/dL → 0 unidades
        {'cgm': 120, 'expected': 0},
        {'cgm': 149, 'expected': 0},
        # 150-200 mg/dL → 2 unidades
        {'cgm': 150, 'expected': 2},
        {'cgm': 180, 'expected': 2},
        {'cgm': 200, 'expected': 2},
        # 200-250 mg/dL → 4 unidades
        {'cgm': 201, 'expected': 4},
        {'cgm': 225, 'expected': 4},
        {'cgm': 249, 'expected': 4},
        # 250-300 mg/dL → 6 unidades
        {'cgm': 250, 'expected': 6},
        {'cgm': 275, 'expected': 6},
        {'cgm': 299, 'expected': 6},
        # >300 mg/dL → 8 unidades
        {'cgm': 301, 'expected': 8},
        {'cgm': 350, 'expected': 8},
        {'cgm': 400, 'expected': 8},
    ]
    # Simular observaciones para cada caso
    for i, case in enumerate(elderly_cases):
        cgm_values = [case['cgm']] * 24  # Paciente estable, sin ejercicio
        obs = prepare_enhanced_observation(
            cgm_values=cgm_values,
            carb_input=0.0,  # Sin ingesta de carbos
            iob=0.0,         # Sin insulina activa
            timestamp=datetime.now().replace(hour=7, minute=0),
            meal_carbs=0.0,
            meal_time_diff=0.0,
            has_meal=0.0,
            meals_in_window=0,
            extended_cgm=cgm_values
        )
        pred = predict_insulin_dose(
            model_path=model_path,
            observation=obs,
            current_cgm=case['cgm'],
            carbs=0.0,
            iob=0.0
        )
        print(f"{MAGENTA}[ELDERLY TEST] Caso {i+1}: CGM={case['cgm']} mg/dL | Dosis esperada={case['expected']} | Dosis modelo={pred:.2f}{RESET}")

def main():
    """Función principal mejorada con todas las mejoras."""
    # Directorios base
    train_base_dir = 'new_ohio/processed_data/train'
    test_base_dir = 'new_ohio/processed_data/test'
    output_dir = 'new_ohio/models/enhanced_output'
    tensorboard_log = 'new_ohio/models/runs/ppo_ohio_enhanced'
    total_timesteps = 1_000_000  # Aumentado para mejor convergencia
    n_envs = 8

    # Buscar todos los archivos parquet
    train_files = sorted(glob.glob(f"{train_base_dir}/processed_*.parquet"))
    test_files = sorted(glob.glob(f"{test_base_dir}/processed_*.parquet"))

    # Crear directorios de salida
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    Path(tensorboard_log).mkdir(parents=True, exist_ok=True)

    logging.info("Iniciando entrenamiento mejorado con características clínicas...")
    logging.info(f"Archivos de entrenamiento encontrados: {len(train_files)}")
    logging.info(f"Archivos de prueba encontrados: {len(test_files)}")

    #Entrenar un modelo nuevo
    model_object = train_with_enhanced_hyperparameters( # Renombrado a model_object para evitar confusión
        train_files=train_files,
        output_dir=output_dir,
        tensorboard_log=tensorboard_log,
        total_timesteps=total_timesteps,
        n_envs=n_envs,
        # Los hiperparámetros específicos como learning_rate, batch_size, etc., 
        # se toman de los argumentos por defecto de la función o se pueden pasar aquí.
    )
    # Después del entrenamiento, la ruta al modelo guardado es conocida
    model_path_to_evaluate = os.path.join(output_dir, 'final_model.zip')
    logging.info(f"Modelo entrenado y guardado en: {model_path_to_evaluate}")

    # Evaluar modelo si existen datos de prueba
    if test_files:
        logging.info("Evaluando modelo mejorado...")
        all_results = {}
        
        for test_file in test_files:
            try:
                patient_id = Path(test_file).stem.replace('processed_', '')
                logging.info(f"Evaluando paciente: {patient_id}")
                
                results = predict_from_preprocessed(
                    model_path=model_path_to_evaluate, # Usar la ruta del modelo recién entrenado
                    preprocessed_data_path=test_file
                )
                    
                all_results[patient_id] = results
                    
                # Imprimir resultados detallados de evaluación clínica
                metrics = results['clinical_metrics']
                logging.info(f"\nResultados de Evaluación Clínica para {patient_id}:")
                logging.info("=" * 60)
                logging.info(f"Métricas Estadísticas:")
                logging.info(f"  MAE: {metrics['mae']:.4f} unidades")
                logging.info(f"  RMSE: {metrics['rmse']:.4f} unidades")
                logging.info(f"  MAPE: {metrics['mape']:.2f}%")
                
                logging.info(f"\nMétricas de Seguridad:")
                logging.info(f"  Tasa Segura (±20%): {metrics['safe_rate_20']:.2f}%")
                logging.info(f"  Tasa Segura (±10%): {metrics['safe_rate_10']:.2f}%")
                logging.info(f"  Tasa de Error Severo: {metrics['severe_error_rate']:.2f}%")
                
                logging.info(f"\nMétricas Clínicas:")
                if 'time_in_range_mean' in metrics:
                    logging.info(f"  Tiempo en Rango: {metrics['time_in_range_mean']:.2f}%")
                    logging.info(f"  Riesgo de Hipoglucemia: {metrics['hypo_percentage_mean']:.2f}%")
                    logging.info(f"  Riesgo de Hiperglucemia: {metrics['hyper_percentage_mean']:.2f}%")
                
                logging.info(f"\nAnálisis de Dosis:")
                logging.info(f"  Dosis Media Predicha: {metrics['mean_predicted_dose']:.2f} unidades")
                logging.info(f"  Dosis Media Real: {metrics['mean_actual_dose']:.2f} unidades")
                logging.info(f"  Conservadurismo en Bolos Grandes: {metrics['large_bolus_conservatism']:.2f}")
                logging.info("=" * 60)
                
            except Exception as e:
                logging.error(f"Error al evaluar {test_file}: {e}")
                continue

        # Probar escenarios mejorados de entrada de usuario
        logging.info("\nProbando escenarios mejorados de entrada de usuario...")
        enhanced_test_cases = [
            # Caso Mejorado 1: Comida normal con glucosa estable
            {
                'name': 'Comida Normal, Glucosa Estable',
                'cgm_values': [float(120 + np.sin(i/4)*5) for i in range(24)],
                'carb_input': 50.0, 'iob': 2.5, 'timestamp': datetime.now().replace(hour=12, minute=0),
                'meal_carbs': 45.0, 'meal_time_diff': 0.5, 'has_meal': 1.0, 'meals_in_window': 1,
                'extended_cgm': [float(115 + np.random.normal(0, 10)) for _ in range(48)]
            },
            # Caso Mejorado 2: Glucosa alta con comida grande
            {
                'name': 'Glucosa Alta, Comida Grande',
                'cgm_values': [float(280 + i*2) for i in range(24)],
                'carb_input': 100.0, 'iob': 0.0, 'timestamp': datetime.now().replace(hour=18, minute=30),
                'meal_carbs': 95.0, 'meal_time_diff': 0.25, 'has_meal': 1.0, 'meals_in_window': 1,
                'extended_cgm': [float(250 + np.random.normal(0, 20)) for _ in range(48)]
            },
            # Caso Mejorado 3: Riesgo de hipoglucemia
            {
                'name': 'Riesgo Hipoglucemia (CGM Bajo, IOB Alto)',
                'cgm_values': [float(75 - i) for i in range(24)], # Tendencia descendente
                'carb_input': 0.0, 'iob': 5.0, 'timestamp': datetime.now().replace(hour=3, minute=0),
                'meal_carbs': 0.0, 'meal_time_diff': 0.0, 'has_meal': 0.0, 'meals_in_window': 0,
                'extended_cgm': [float(85 + np.random.normal(0, 15)) for _ in range(48)]
            },
            # Caso Mejorado 4: Fenómeno del alba
            {
                'name': 'Fenómeno del Alba (Hiperglucemia Matutina)',
                'cgm_values': [float(140 + i*3) for i in range(24)],
                'carb_input': 30.0, 'iob': 1.0, 'timestamp': datetime.now().replace(hour=6, minute=30),
                'meal_carbs': 25.0, 'meal_time_diff': 0.5, 'has_meal': 1.0, 'meals_in_window': 1,
                'extended_cgm': [float(130 + np.random.normal(0, 10)) for _ in range(48)]
            },
            # Nuevos Casos de Prueba Diversos
            {
                'name': 'Alta Sensibilidad Insulina (Hipoglucemia con Dosis Mínima)',
                'cgm_values': [90.0] * 24, # CGM estable pero bajo
                'carb_input': 10.0, # Comida muy pequeña
                'iob': 1.0, # Algo de IOB
                'timestamp': datetime.now().replace(hour=10, minute=0),
                'meal_carbs': 10.0, 'meal_time_diff': 0.1, 'has_meal': 1.0, 'meals_in_window': 1,
                'extended_cgm': [90.0] * 48
            },
            {
                'name': 'Resistencia Insulina (Hiperglucemia Persistente)',
                'cgm_values': [220.0] * 24, # CGM estable pero alto
                'carb_input': 60.0, # Comida normal
                'iob': 0.5, # Poca IOB
                'timestamp': datetime.now().replace(hour=13, minute=0),
                'meal_carbs': 60.0, 'meal_time_diff': 0.0, 'has_meal': 1.0, 'meals_in_window': 1,
                'extended_cgm': [220.0] * 48 
            },
            {
                'name': 'Deportista (Post-Ejercicio, Riesgo Hipo)',
                'cgm_values': [100.0, 95.0, 90.0, 85.0, 80.0, 75.0] + [75.0]*18, # CGM bajando
                'carb_input': 20.0, # Carbs de recuperación
                'iob': 0.0, # Sin IOB pre-ejercicio
                'timestamp': datetime.now().replace(hour=17, minute=0),
                'meal_carbs': 20.0, 'meal_time_diff': 0.0, 'has_meal': 1.0, 'meals_in_window': 1,
                'extended_cgm': [110.0 - i for i in range(24)] + [86.0]*24 # Historial de CGM descendente
            },
            {
                'name': 'Comida Alta Grasa/Proteína (Efecto Retrasado)',
                'cgm_values': [130.0] * 24, # CGM estable pre-comida
                'carb_input': 50.0, # Carbs moderados, pero efecto lento esperado
                'iob': 1.0,
                'timestamp': datetime.now().replace(hour=19, minute=0),
                'meal_carbs': 50.0, 'meal_time_diff': 0.0, 'has_meal': 1.0, 'meals_in_window': 1,
                'extended_cgm': [130.0]*24 + [140.0, 150.0, 160.0, 170.0]*6 # Simula subida lenta post-comida
            }
        ]
        
        user_input_results = test_enhanced_user_inputs(
            model_path=model_path_to_evaluate, # Usar la ruta del modelo recién entrenado
            test_cases=enhanced_test_cases
        )
        
        # Imprimir resultados de prueba de entrada de usuario mejorada
        logging.info("\nResultados de Prueba de Entrada de Usuario Mejorada:")
        logging.info("=" * 60)
        for case_name, case_results in user_input_results.items():
            logging.info(f"\n{case_name}:")
            logging.info(f"  Dosis Predicha: {case_results['predicted_dose']:.2f} unidades")
            logging.info(f"  CGM Actual: {case_results['current_cgm']:.2f} mg/dL")
            logging.info(f"  Glucosa Futura Estimada: {case_results['estimated_future_glucose']:.2f} mg/dL")
            logging.info(f"  Riesgo de Hipoglucemia: {case_results['hypo_risk']:.0f}")
            logging.info(f"  Riesgo de Hiperglucemia: {case_results['hyper_risk']:.0f}")
            logging.info(f"  Puntuación de Seguridad: {case_results['safety_score']:.2f}")
        logging.info("=" * 60)

        # Generar visualizaciones mejoradas
        logging.info("Generando visualizaciones mejoradas...")
        plot_enhanced_training(tensorboard_log, output_dir)
        plot_clinical_results(all_results, output_dir)
        
        # Calcular desempeño general
        if all_results:
            all_maes = [result['clinical_metrics']['mae'] for result in all_results.values() if 'clinical_metrics' in result]
            all_tirs = [result['clinical_metrics'].get('time_in_range_mean', 0) for result in all_results.values() if 'clinical_metrics' in result]
            all_safe_rates = [result['clinical_metrics']['safe_rate_20'] for result in all_results.values() if 'clinical_metrics' in result]
            
            if all_maes:
                logging.info(f"\nResumen de Desempeño General:")
                logging.info(f"MAE Promedio: {np.mean(all_maes):.4f} ± {np.std(all_maes):.4f} unidades")
                if all_tirs:
                    logging.info(f"Tiempo en Rango Promedio: {np.mean(all_tirs):.2f} ± {np.std(all_tirs):.2f}%")
                logging.info(f"Tasa Segura Promedio: {np.mean(all_safe_rates):.2f} ± {np.std(all_safe_rates):.2f}%")

    else:
        logging.warning("No se encontraron datos de prueba. Omitiendo evaluación.")
        logging.info("Para evaluar el modelo, por favor asegúrese de que existan datos de prueba en:")
        logging.info(f"  - {test_base_dir}/processed_*.parquet")

    logging.info("¡Entrenamiento y evaluación del modelo mejorado de predicción de bolo de insulina completados!")

    # Al final del main, correr los tests de la paciente anciana
    test_elderly_patient_cases(model_path=model_path_to_evaluate) # Usar la ruta del modelo recién entrenado

if __name__ == "__main__":
    main()