import os
import sys

import gymnasium as gym
import numpy as np
import polars as pl
from gymnasium import spaces
from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.buffers import ReplayBuffer
from typing import Dict, List, Optional, Tuple, Union
import logging
from pathlib import Path
import os
import sys
from stable_baselines3.common.callbacks import BaseCallback
from tqdm import tqdm
import torch
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.policies import ActorCriticPolicy
from torch import nn
import torch.nn.functional as F
from scipy import stats
import json
import glob

## --entrenamiento-rápido para ejecutar con muestra del 10% para depuración

# Agregar directorio padre al path para importar desde processing
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from new_ohio.models.tests.test_sac_user_cases import test_user_cases
from new_ohio.processing.ohio_polars import CONFIG, ColoredFormatter, simulate_glucose

# Configurar logging
logger = logging.getLogger()
handler = logging.StreamHandler()
handler.setFormatter(ColoredFormatter('%(asctime)s - %(levelname)s - %(message)s'))
logger.addHandler(handler)
logger.setLevel(logging.INFO)

class OhioT1DMEnv(gym.Env):
    """
    Entorno personalizado para OhioT1DM usando Gym.
    Implementa un espacio de acción continua para predicción de bolus.
    """
    
    def __init__(self, df: pl.DataFrame, state_mean: Optional[np.ndarray] = None, state_std: Optional[np.ndarray] = None):
        super().__init__()
        
        # Guardar DataFrame
        self.df = df
        self.current_step = 0
        
        # Definir columnas base que siempre deben estar presentes
        self.base_columns = [
            'Timestamp', 'SubjectID', 'bolus'  # Solo estas son realmente requeridas
        ]
        
        # Definir mapeo de columnas alternativas
        self.column_aliases = {
            'carb_input': ['carb_input', 'carbs', 'meal_carbs', 'carbohydrates'],
            'insulin_on_board': ['insulin_on_board', 'iob', 'active_insulin'],
            'meal_carbs': ['meal_carbs', 'carbs', 'carb_input', 'carbohydrates'],
            'meal_time_diff_hours': ['meal_time_diff_hours', 'meal_time_diff', 'time_since_meal'],
            'has_meal': ['has_meal', 'meal_occurred', 'meal_indicator'],
            'meals_in_window': ['meals_in_window', 'meals_in_period', 'meal_count']
        }
        
        # Definir columnas de características opcionales
        self.optional_columns = [
            # Valores CGM (24h)
            *[f'cgm_{i}' for i in range(24)],
            
            # Características temporales
            'hour_sin', 'hour_cos',
            
            # Características básicas (con alias)
            'carb_input', 'insulin_on_board',
            
            # Contexto de comidas (con alias)
            'meal_carbs', 'meal_time_diff_hours', 'has_meal', 'meals_in_window',
            
            # Patrones de glucosa a corto plazo
            'cgm_trend', 'cgm_std',
            
            # Patrones de glucosa a largo plazo (24h)
            'cgm_mean_24h', 'cgm_std_24h', 'time_in_range_24h',
            'hypo_percentage_24h', 'hyper_percentage_24h', 'cv_24h',
            'mage_24h', 'glucose_trend_24h', 'cgm_range_24h',
            'cgm_median_24h', 'hypo_episodes_24h', 'hyper_episodes_24h'
        ]
        
        # Verificar columnas base
        missing_base = [col for col in self.base_columns if col not in df.columns]
        if missing_base:
            raise ValueError(f"Faltan columnas requeridas: {missing_base}")
        
        # Función para encontrar la primera columna disponible de una lista de alternativas
        def find_available_column(column_options):
            if isinstance(column_options, str):
                return column_options if column_options in df.columns else None
            for col in column_options:
                if col in df.columns:
                    return col
            return None
        
        # Encontrar columnas disponibles usando alias
        available_columns = []
        for col in self.optional_columns:
            if col in df.columns:
                available_columns.append(col)
            elif col in self.column_aliases:
                alias_col = find_available_column(self.column_aliases[col])
                if alias_col:
                    available_columns.append(alias_col)
                    logger.info(f"Usando columna alternativa '{alias_col}' para '{col}'")
        
        # Combinar columnas base (excluyendo Timestamp y SubjectID) con opcionales disponibles
        self.feature_columns = [col for col in self.base_columns if col not in ['Timestamp', 'SubjectID']] + available_columns
        
        # Si se proporcionan estadísticas pre-calculadas, usarlas
        if state_mean is not None and state_std is not None:
            self.state_mean = state_mean
            self.state_std = state_std
            # Asegurar que las columnas coincidan con las estadísticas
            if len(self.state_mean) != len(self.feature_columns):
                logger.warning(f"Ajustando columnas de características para coincidir con estadísticas de normalización")
                self.feature_columns = self.feature_columns[:len(self.state_mean)]
        else:
            # Calcular estadísticas para normalización
            feature_data = df.select(self.feature_columns)
            
            # Verificar y manejar valores NaN antes de calcular estadísticas
            for col in self.feature_columns:
                if feature_data[col].null_count() > 0:
                    logger.warning(f"Columna {col} tiene {feature_data[col].null_count()} valores nulos. Reemplazando con 0.")
                    feature_data = feature_data.with_columns(pl.col(col).fill_null(0))
            
            # Calcular estadísticas
            self.state_mean = feature_data.mean().to_numpy().flatten()
            self.state_std = feature_data.std().to_numpy().flatten()
            
            # Filtrar columnas con std=0 o muy cercana a cero
            valid_indices = self.state_std > 1e-8
            self.feature_columns = [col for i, col in enumerate(self.feature_columns) if valid_indices[i]]
            self.state_mean = self.state_mean[valid_indices]
            self.state_std = self.state_std[valid_indices]
            
            logger.info(f"Columnas filtradas: {len(self.feature_columns)} características válidas")
            logger.info(f"Columnas eliminadas: {sum(~valid_indices)} características no informativas")
            
            # Verificar y ajustar desviación estándar
            min_std = 1e-8
            self.state_std = np.maximum(self.state_std, min_std)
        
        # Definir espacios de observación y acción
        self.observation_space = spaces.Box(
            low=-np.inf, 
            high=np.inf, 
            shape=(len(self.feature_columns),),
            dtype=np.float32
        )
        
        self.action_space = spaces.Box(
            low=CONFIG['bolus_min'],
            high=CONFIG['bolus_max'],
            shape=(1,),
            dtype=np.float32
        )
        
        # Inicializar estado actual
        self.current_state = None
        self.current_glucose = None
        self.current_subject = None
        
        logger.info(f"Forma del espacio de observación: {len(self.feature_columns)}")
        logger.info(f"Columnas de características: {self.feature_columns}")
        logger.info(f"Características opcionales disponibles: {available_columns}")

    def save_normalization_stats(self, path: str):
        """Guarda las estadísticas de normalización en un archivo."""
        stats = {
            'state_mean': self.state_mean.tolist(),
            'state_std': self.state_std.tolist(),
            'feature_columns': self.feature_columns
        }
        with open(path, 'w') as f:
            json.dump(stats, f)
        logger.info(f"Estadísticas de normalización guardadas en {path}")

    @classmethod
    def load_normalization_stats(cls, path: str) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """Carga las estadísticas de normalización desde un archivo."""
        with open(path, 'r') as f:
            stats = json.load(f)
        return (
            np.array(stats['state_mean']),
            np.array(stats['state_std']),
            stats['feature_columns']
        )

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None) -> Tuple[np.ndarray, dict]:
        """Reinicia el entorno al primer paso."""
        self.current_step = 0
        self._update_state()
        return self._get_observation(), {}  # Retorna tupla (observation, info)
    
    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """
        Ejecuta un paso en el entorno.
        
        Args:
            action: Dosis de bolus a administrar
            
        Returns:
            observation: Estado normalizado
            reward: Recompensa calculada
            terminated: Si el episodio terminó naturalmente
            truncated: Si el episodio fue truncado
            info: Información adicional
        """
        # Asegurar que action sea un array numpy con la forma correcta
        action_np = np.array(action, dtype=np.float32).reshape(-1)
        current_action_value = action_np[0]
        
        reward = 0.0  # Inicializar recompensa para este paso

        # 1. Manejar acción NaN
        is_nan_action = np.isnan(current_action_value)
        if is_nan_action:
            logger.warning(f"Acción NaN detectada: {current_action_value}. Reemplazando con 0 y penalizando.")
            current_action_value = 0.0
            reward = -100.0  # Penalización grande específicamente para acciones NaN
            
        # 2. Aplicar capa de seguridad en el valor de acción (posiblemente corregido por NaN)
        # Esta capa de seguridad se aplica si la acción original NO era NaN pero era insegura.
        # Si la acción era NaN (ahora 0.0), la condición current_action_value > 10 es falsa, por lo que la capa de seguridad no se activará.
        if not is_nan_action and self.current_glucose < 100 and current_action_value > 10:
            logger.warning(f"Capa de seguridad: Recortando acción {current_action_value} a 10.0 para glucosa {self.current_glucose}")
            current_action_value = 10.0 # Aplicar corrección
            reward = -5.0  # Penalización por acción insegura (sobrescribe reward = 0 del caso no-NaN)
        
        # Obtener estado actual (estado en t antes de la acción)
        current_state_for_reward_calc = self._get_state_dict()
        
        # Simular siguiente estado de glucosa usando la acción (posiblemente modificada)
        actual_bolus_for_sim = float(current_action_value)
        
        next_glucose_simulation_results = simulate_glucose(
            cgm_values=[self.current_glucose], # Glucosa en tiempo t (antes de la acción)
            bolus=actual_bolus_for_sim,      # Acción tomada en tiempo t
            carbs=current_state_for_reward_calc.get('meal_carbs', 0.0),
            basal_rate=current_state_for_reward_calc.get('effective_basal_rate', 0.0),
            exercise_intensity=current_state_for_reward_calc.get('exercise_intensity', 0.0)
        )
        
        # Avanzar al siguiente paso de tiempo en los datos del DataFrame
        self.current_step += 1
        self._update_state() # Actualiza self.current_state, self.current_glucose para el TIEMPO t+1
        
        # Calcular recompensa basada en el resultado de la acción tomada en el estado t
        computed_reward_component = self._compute_reward(current_state_for_reward_calc, actual_bolus_for_sim, next_glucose_simulation_results)
        
        # Sumar la recompensa calculada a las penalizaciones existentes (NaN o capa de seguridad)
        reward += computed_reward_component
        
        # Verificar si terminó el episodio
        terminated = self.current_step >= len(self.df)
        truncated = False  # No hay truncamiento en este entorno
        
        # La observación retornada es para el estado en t+1
        return self._get_observation(), reward, terminated, truncated, {}
    
    def _get_observation(self) -> np.ndarray:
        """Obtiene la observación normalizada del estado actual."""
        if self.current_state is None:
            return np.zeros(len(self.feature_columns), dtype=np.float32)
        
        try:
            state_values = []
            for col in self.feature_columns:
                val = self.current_state.get(col, 0.0)
                
                # Convertir a float de manera segura
                if isinstance(val, (list, np.ndarray)):
                    val = float(val[0]) if len(val) > 0 else 0.0
                elif isinstance(val, (int, float)):
                    val = float(val)
                else:
                    val = 0.0
                
                # Manejar NaN explícitamente
                if np.isnan(val):
                    logger.warning(f"Valor NaN encontrado para columna {col}. Reemplazando con 0.0")
                    val = 0.0
                
                state_values.append(val)
            
            state_array = np.array(state_values, dtype=np.float32).flatten()
            
            # Verificar NaN después de la conversión
            if np.isnan(state_array).any():
                logger.warning("NaN detectado después de conversión. Reemplazando con 0.0")
                state_array = np.nan_to_num(state_array, nan=0.0)
            
            # Normalizar con manejo de división por cero y NaN
            normalized_state = np.zeros_like(state_array)
            
            # Verificar dimensiones
            if len(self.state_mean) != len(state_array) or len(self.state_std) != len(state_array):
                logger.error(f"Error de dimensiones: state_array={len(state_array)}, state_mean={len(self.state_mean)}, state_std={len(self.state_std)}")
                return np.zeros(len(self.feature_columns), dtype=np.float32)
            
            # Aplicar normalización solo donde std != 0 y no es NaN
            mask = (self.state_std != 0) & ~np.isnan(self.state_std)
            normalized_state[mask] = (state_array[mask] - self.state_mean[mask]) / self.state_std[mask]
            
            # Para columnas con std=0 o NaN, solo centrar
            mask = ~mask
            normalized_state[mask] = state_array[mask] - self.state_mean[mask]
            
            # Verificar NaN después de normalización
            if np.isnan(normalized_state).any():
                logger.warning("NaN detectado después de normalización. Reemplazando con 0.0")
                normalized_state = np.nan_to_num(normalized_state, nan=0.0)
            
            # Clipear valores normalizados
            normalized_state = np.clip(normalized_state, -10, 10)
            
            return normalized_state.astype(np.float32)
        
        except Exception as e:
            logger.error(f"Error en _get_observation: {e}")
            logger.error(f"Estado actual: {self.current_state}")
            logger.error(f"Columnas esperadas: {self.feature_columns}")
            logger.error(f"Columnas disponibles: {list(self.current_state.keys())}")
            return np.zeros(len(self.feature_columns), dtype=np.float32)
    
    def _get_state_dict(self) -> Dict:
        """Obtiene el estado actual como diccionario."""
        if self.current_state is None:
            return {}
        return self.current_state
    
    def _update_state(self):
        """Actualiza el estado actual con los datos del paso actual."""
        if self.current_step < len(self.df):
            row = self.df.row(self.current_step, named=True)
            self.current_state = row
            self.current_glucose = row.get('glucose_last', 120.0)
            self.current_subject = row.get('SubjectID')
        else:
            self.current_state = None
            self.current_glucose = None
            self.current_subject = None
    
    def _compute_reward(self, state: Dict, action: float, next_glucose: Dict) -> float:
        """
        Calcula la recompensa basada en el estado, acción y siguiente estado.
        Implementa una función de recompensa escalonada según el artículo.
        
        Args:
            state: Estado actual
            action: Acción tomada (bolus)
            next_glucose: Diccionario con valores de glucosa simulados
            
        Returns:
            float: Recompensa total (TIR + TBR)
        """
        # Obtener glucosa a 2h tras bolus
        glucose_2h = next_glucose.get('simulated_glucose_2h', 120.0)
        
        # Asegurar que current_glucose sea float
        try:
            current_glucose = float(state.get('glucose_last', 120.0))
        except (ValueError, TypeError):
            current_glucose = 120.0  # Valor por defecto si no es convertible
        
        # Componente TIR
        tir_reward = 0.0
        if 70 <= glucose_2h <= 180:
            tir_reward = 100.0  # Recompensa máxima por TIR
        elif glucose_2h < 60:
            tir_reward = -200.0  # Penalización severa por hipoglucemia severa
        elif 60 <= glucose_2h < 70:
            tir_reward = -100.0  # Penalización por hipoglucemia leve
        elif 180 < glucose_2h <= 250:
            tir_reward = -50.0  # Penalización por hiperglucemia leve
        elif glucose_2h > 250:
            tir_reward = -150.0  # Penalización por hiperglucemia severa
            
        # Componente TBR
        tbr_penalty = -50.0 if glucose_2h < 70 else 0.0
        
        # Componente de acción
        action_penalty = -0.1 * abs(action)  # Pequeña penalización por usar insulina
        
        # Componente de cambio de glucosa
        glucose_change = glucose_2h - current_glucose
        glucose_change_penalty = -0.1 * abs(glucose_change)  # Penalización por cambios bruscos
        
        # Recompensa total
        total_reward = tir_reward + tbr_penalty + action_penalty + glucose_change_penalty
        
        # Registro de componentes de recompensa
        logger.debug(f"Componentes de recompensa - Glucosa_2h: {glucose_2h:.1f}, TIR: {tir_reward:.1f}, "
                    f"TBR: {tbr_penalty:.1f}, Action: {action_penalty:.1f}, "
                    f"Change: {glucose_change_penalty:.1f}, Total: {total_reward:.1f}")
        
        return total_reward
    
    def prefill_replay_buffer(self, model: SAC, n_transitions: int = 10000):
        """
        Prellena el buffer de repetición con transiciones simuladas usando datos históricos.
        
        Args:
            model: Modelo SAC
            n_transitions: Número de transiciones a generar
        """
        logger.info(f"Prellenando buffer con {n_transitions} transiciones usando datos históricos...")
        
        # Limitar n_transitions al tamaño del dataset
        n_transitions = min(n_transitions, len(self.df) - 1)
        
        for i in range(n_transitions):
            # Establecer paso actual y actualizar estado
            self.current_step = i
            self._update_state()
            
            # Obtener estado actual
            state = self._get_observation()
            
            # Usar acción histórica del dataset
            action = np.array([self.df['bolus'][i]], dtype=np.float32)
            
            # Simular siguiente estado
            next_state, reward, terminated, truncated, infos = self.step(action)
            
            # Añadir transición al buffer
            model.replay_buffer.add(
                np.expand_dims(state, axis=0),           # obs
                np.expand_dims(next_state, axis=0),      # next_obs
                np.expand_dims(action, axis=0),          # action
                np.array([reward], dtype=np.float32),    # reward
                np.array([terminated or truncated], dtype=bool),  # done
                [infos]                                  # infos
            )
        
        logger.info("Buffer prellenado completado")

class FQE:
    """Evaluación Q Ajustada para evaluación de política fuera de línea."""
    
    def __init__(self, state_dim: int, action_dim: int, device: str = 'cuda'):
        self.device = device
        self.q_network = Critic(state_dim, action_dim).to(device)
        self.optimizer = torch.optim.Adam(self.q_network.parameters(), lr=1e-3)
        
    def evaluate(self, policy, dataset: Union[pl.DataFrame, DummyVecEnv], n_iterations: int = 1000) -> float:
        """
        Evalúa la política usando FQE.
        
        Args:
            policy: Modelo SAC a evaluar
            dataset: DataFrame con datos de validación o entorno de validación
            n_iterations: Número de iteraciones para FQE
            
        Returns:
            float: Valor estimado de la política
        """
        logger.info("Iniciando evaluación FQE...")
        
        # Obtener datos del entorno si se proporciona un DummyVecEnv
        if isinstance(dataset, DummyVecEnv):
            env = dataset.envs[0]
            df = env.df
        else:
            df = dataset
        
        # Limitar el dataset para debug/velocidad
        if len(df) > 1000:
            logger.warning(f"Reduciendo dataset de FQE de {len(df)} a 1000 muestras para acelerar debug.")
            df = df.sample(n=1000, seed=42)
        
        # Obtener columnas de características del entorno
        if isinstance(dataset, DummyVecEnv):
            feature_columns = env.feature_columns
        else:
            feature_columns = [col for col in df.columns if col not in ['Timestamp', 'SubjectID']]
        
        # Convertir dataset a tensores
        states = torch.FloatTensor(df.select(feature_columns).to_numpy()).to(self.device)
        actions = torch.FloatTensor(df.select(['bolus']).to_numpy()).to(self.device)
        
        # Crear entorno temporal para calcular rewards
        temp_env = OhioT1DMEnv(df)
        rewards = []
        next_states = []
        dones = []
        
        # Calcular rewards y next_states usando el entorno
        for i in range(len(df)):
            state = states[i].cpu().numpy()
            action = actions[i].cpu().numpy()
            current_state = dict(zip(df.columns, df.row(i, named=True)))
            
            # Asegurar que current_glucose sea float
            try:
                logger.info(f"current_state: {current_state}")
                current_glucose = float(current_state['glucose_last'])
            except (KeyError, ValueError):
                current_glucose = 120.0  # Valor por defecto si no existe o no es convertible
                
            temp_env.current_state = current_state
            temp_env.current_glucose = current_glucose
            
            # Simular paso
            next_glucose = simulate_glucose(
                cgm_values=[current_glucose],  # Asegurar que sea una lista de floats
                bolus=float(action[0]),
                carbs=float(current_state.get('meal_carbs', 0.0)),
                basal_rate=float(current_state.get('effective_basal_rate', 0.0)),
                exercise_intensity=float(current_state.get('exercise_intensity', 0.0))
            )
            
            # Calcular reward
            reward = temp_env._compute_reward(temp_env.current_state, float(action[0]), next_glucose)
            rewards.append(reward)
            
            # Obtener siguiente estado
            if i < len(df) - 1:
                # Convertir el siguiente estado a tensor manteniendo el orden de las características
                next_state_dict = dict(zip(df.columns, df.row(i + 1, named=True)))
                next_state_values = []
                for col in feature_columns:
                    try:
                        val = next_state_dict.get(col, 0.0)
                        if isinstance(val, (list, np.ndarray)):
                            val = float(val[0]) if len(val) > 0 else 0.0
                        next_state_values.append(float(val))
                    except (ValueError, TypeError):
                        next_state_values.append(0.0)
                next_state = torch.FloatTensor(next_state_values).to(self.device)
                next_states.append(next_state)
                dones.append(False)
            else:
                next_states.append(torch.zeros_like(states[i]))
                dones.append(True)
        
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.stack(next_states).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device)
        
        # Iteraciones FQE
        for i in tqdm(range(n_iterations), desc="Progreso FQE"):
            # Obtener valores Q para pares estado-acción actuales
            q1, q2 = self.q_network(states, actions)
            q_values = torch.min(q1, q2).view(-1)  # [N]
            
            # Obtener siguientes acciones de la política usando predict()
            with torch.no_grad():
                next_actions = []
                for next_state in next_states:
                    action, _ = policy.predict(next_state.cpu().numpy(), deterministic=True)
                    # Asegurarse de que la acción sea 1D
                    action_tensor = torch.FloatTensor(action).flatten().to(self.device)
                    next_actions.append(action_tensor)
                next_actions = torch.stack(next_actions)  # [N, action_dim]
                next_q1, next_q2 = self.q_network(next_states, next_actions)
                next_q_values = torch.min(next_q1, next_q2).view(-1)  # [N]
            
            # Calcular objetivos
            targets = rewards + (1 - dones) * 0.99 * next_q_values
            targets = targets.view(-1)  # [N]
            
            # Actualizar red Q
            loss = F.mse_loss(q_values, targets)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            if i % 100 == 0:
                logger.info(f"Iteración FQE {i}, pérdida: {loss.item():.4f}")
        
        # Calcular valor final de la política
        with torch.no_grad():
            initial_states = states[:1000]  # Usar primeros 1000 estados como estados iniciales
            initial_actions = []
            for state in initial_states:
                action, _ = policy.predict(state.cpu().numpy(), deterministic=True)
                action_tensor = torch.FloatTensor(action).flatten().to(self.device)
                initial_actions.append(action_tensor)
            initial_actions = torch.stack(initial_actions)
            q1, q2 = self.q_network(initial_states, initial_actions)
            policy_value = torch.min(q1, q2).mean().item()
        
        logger.info(f"Evaluación FQE completada. Valor estimado de la política: {policy_value:.4f}")
        return policy_value

def calculate_clinical_metrics(glucose_values: np.ndarray) -> Dict[str, float]:
    """
    Calcula métricas clínicas a partir de valores de glucosa.
    
    Args:
        glucose_values: Array de valores de glucosa en mg/dL
        
    Returns:
        Dict conteniendo métricas clínicas
    """
    # Métricas TIR
    tir = np.mean((glucose_values >= 70) & (glucose_values <= 180)) * 100
    
    # Métricas TAR
    tar_l1 = np.mean((glucose_values > 180) & (glucose_values <= 250)) * 100
    tar_l2 = np.mean(glucose_values > 250) * 100
    tar = tar_l1 + tar_l2
    
    # Métricas TBR
    tbr_l1 = np.mean((glucose_values >= 54) & (glucose_values < 70)) * 100
    tbr_l2 = np.mean(glucose_values < 54) * 100
    tbr = tbr_l1 + tbr_l2
    
    # Cálculo LBGI/HBGI
    def transform_glucose(g):
        f = 1.509 * (np.log(g)**1.084 - 5.381)
        return f
    
    f_values = transform_glucose(glucose_values)
    lbgi = np.mean(np.clip(-f_values, 0, None))
    hbgi = np.mean(np.clip(f_values, 0, None))
    
    return {
        'TIR': float(tir),
        'TAR_L1': float(tar_l1),
        'TAR_L2': float(tar_l2),
        'TAR': float(tar),
        'TBR_L1': float(tbr_l1),
        'TBR_L2': float(tbr_l2),
        'TBR': float(tbr),
        'LBGI': float(lbgi),
        'HBGI': float(hbgi)
    }

def evaluate_model(model: SAC, env: DummyVecEnv, df_test: pl.DataFrame) -> Dict:
    """
    Evalúa el modelo usando métricas clínicas y FQE.
    
    Args:
        model: Modelo SAC entrenado
        env: Entorno vectorizado
        df_test: Dataset de prueba
        
    Returns:
        Dict conteniendo métricas de evaluación
    """
    logger.info("Iniciando evaluación del modelo...")
    
    # Recolectar predicciones y valores reales
    predictions = []
    actuals = []
    glucose_values = []
    
    obs = env.reset()
    done = False
    
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        action_value = action[0].item() if isinstance(action[0], np.ndarray) else float(action[0])
        predictions.append(action_value)
        
        current_state = env.envs[0]._get_state_dict()
        actuals.append(float(current_state.get('bolus', 0.0)))
        glucose_values.append(float(current_state.get('glucose_last', 120.0)))
        
        step_result = env.step(action)
        if len(step_result) == 4:
            obs, reward, done, info = step_result
            terminated, truncated = done, False
        else:
            obs, reward, terminated, truncated, info = step_result
            done = terminated or truncated
    
    # Convertir a arrays numpy
    predictions = np.array(predictions, dtype=np.float32)
    actuals = np.array(actuals, dtype=np.float32)
    glucose_values = np.array(glucose_values, dtype=np.float32)
    
    # Calcular métricas clínicas
    clinical_metrics = calculate_clinical_metrics(glucose_values)
    
    # Calcular valor FQE
    fqe = FQE(
        state_dim=env.observation_space.shape[0],
        action_dim=env.action_space.shape[0],
        device='cuda' if torch.cuda.is_available() else 'cpu'
    )
    fqe_value = fqe.evaluate(model, df_test)
    
    # Calcular comparación estadística con línea base
    baseline_metrics = calculate_clinical_metrics(df_test.select(['glucose_last']).to_numpy().flatten())
    t_stat, p_value = stats.ttest_rel(glucose_values, df_test.select(['glucose_last']).to_numpy().flatten())
    
    # Combinar todas las métricas
    results = {
        'clinical_metrics': clinical_metrics,
        'fqe_value': fqe_value,
        'statistical_comparison': {
            't_statistic': float(t_stat),
            'p_value': float(p_value)
        },
        'baseline_metrics': baseline_metrics
    }
    
    # Guardar resultados
    results_path = "new_ohio/models/evaluation_results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=4)
    
    logger.info(f"Evaluación completada. Resultados guardados en {results_path}")
    return results

def evaluate_saved_model(model_path: str, test_files: List[str]):
    """
    Evalúa un modelo guardado previamente.
    
    Args:
        model_path: Ruta al modelo guardado
        test_files: Lista de archivos de prueba
    """
    logger.info(f"Cargando modelo desde {model_path}")
    
    # Cargar datos de prueba
    test_df = pl.concat([pl.read_parquet(f) for f in test_files])
    logger.info(f"Datos de prueba cargados: {test_df.shape}")
    
    # Crear entorno de prueba
    test_env = DummyVecEnv([lambda: OhioT1DMEnv(test_df)])
    
    # Cargar modelo
    model = SAC.load(model_path)
    logger.info("Modelo cargado exitosamente")
    
    # Evaluar modelo
    results = evaluate_model(model, test_env, test_df)
    
    # Guardar resultados
    results_path = "new_ohio/models/evaluation_results.txt"
    with open(results_path, "w") as f:
        f.write(str(results))
    logger.info(f"Resultados guardados en {results_path}")

class Actor(nn.Module):
    """Red del actor para la política."""
    def __init__(self, obs_dim, action_dim, lr=1e-3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, action_dim),
            nn.Tanh()
        )
        self.optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        
    def forward(self, obs):
        return self.net(obs)
        
    def action_log_prob(self, obs):
        """Calcula la acción y su log-probabilidad."""
        action = self.forward(obs)
        # Para una distribución uniforme en [-1, 1], la log-probabilidad es constante
        # log(1/2) = -log(2) ≈ -0.693
        log_prob = torch.full_like(action, -0.693)
        return action, log_prob

class Critic(nn.Module):
    """Red del crítico para estimar valores Q."""
    def __init__(self, obs_dim, action_dim, lr=1e-3):
        super().__init__()
        self.q1 = nn.Sequential(
            nn.Linear(obs_dim + action_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 1)
        )
        self.q2 = nn.Sequential(
            nn.Linear(obs_dim + action_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 1)
        )
        self.optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        
    def forward(self, obs, action):
        x = torch.cat([obs, action], dim=1)
        return self.q1(x), self.q2(x)

class CustomActorCriticPolicy(ActorCriticPolicy):
    """Política personalizada que implementa TD3+BC para aprendizaje fuera de línea."""
    
    def __init__(self, observation_space, action_space, lr_schedule, *args, **kwargs):
        super().__init__(observation_space, action_space, lr_schedule, *args, **kwargs)
        
        # Parámetros TD3+BC
        self.alpha_bc = 0.1  # Coeficiente de clonación de comportamiento
        self.policy_delay = 2  # Retraso de política TD3
        self.noise_clip = 0.5  # Recorte de ruido TD3
        self.policy_noise = 0.2  # Ruido de política TD3
        self.tau = 0.005  # Tasa de actualización suave
        
        # Inicializar redes
        obs_dim = observation_space.shape[0]
        action_dim = action_space.shape[0]
        
        self.actor = Actor(obs_dim, action_dim, lr=lr_schedule(1))
        self.critic = Critic(obs_dim, action_dim, lr=lr_schedule(1))
        self.critic_target = Critic(obs_dim, action_dim, lr=lr_schedule(1))
        
        # Copiar parámetros del crítico al crítico objetivo
        self.critic_target.load_state_dict(self.critic.state_dict())
        
    def forward(self, obs, deterministic=False):
        """Forward pass para obtener acciones."""
        # Validar y normalizar entrada
        if isinstance(obs, np.ndarray):
            obs = torch.FloatTensor(obs).to(self.device)
        
        # Verificar NaN en entrada
        if torch.isnan(obs).any():
            logger.warning("NaN detectado en observación. Reemplazando con ceros.")
            obs = torch.nan_to_num(obs, nan=0.0)
        
        # Obtener acción del actor
        action = self.actor(obs)
        
        # Verificar NaN en salida
        if torch.isnan(action).any():
            logger.warning("NaN detectado en acción. Reemplazando con ceros.")
            action = torch.nan_to_num(action, nan=0.0)
        
        # Clipear acción al rango válido
        action = torch.clamp(action, -1.0, 1.0)
        
        return action
        
    def _train_step(self, replay_buffer, batch_size):
        """Paso de entrenamiento personalizado que implementa TD3+BC."""
        # Obtener batch de datos
        obs, next_obs, actions, rewards, dones = replay_buffer.sample(batch_size)
        
        # Validar y normalizar datos
        obs = torch.FloatTensor(obs).to(self.device)
        next_obs = torch.FloatTensor(next_obs).to(self.device)
        actions = torch.FloatTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device)
        
        # Verificar y manejar NaN
        for tensor, name in [(obs, "obs"), (next_obs, "next_obs"), (actions, "actions"), 
                           (rewards, "rewards"), (dones, "dones")]:
            if torch.isnan(tensor).any():
                logger.warning(f"NaN detectado en {name}. Reemplazando con ceros.")
                tensor = torch.nan_to_num(tensor, nan=0.0)
        
        # Calcular valores Q objetivo
        with torch.no_grad():
            # Agregar ruido a las acciones
            noise = (torch.randn_like(actions) * self.policy_noise).clamp(-self.noise_clip, self.noise_clip)
            next_actions = (self.actor(next_obs) + noise).clamp(-1, 1)
            
            # Calcular Q mínimo
            target_q1, target_q2 = self.critic_target(next_obs, next_actions)
            target_q = torch.min(target_q1, target_q2)
            target_q = rewards + (1 - dones) * self.gamma * target_q
        
        # Actualizar crítico
        current_q1, current_q2 = self.critic(obs, actions)
        critic_loss = F.mse_loss(current_q1, target_q) + F.mse_loss(current_q2, target_q)
        self.critic.optimizer.zero_grad()
        critic_loss.backward()
        self.critic.optimizer.step()
        
        # Actualizar actor menos frecuentemente
        if self.num_timesteps % self.policy_delay == 0:
            # Obtener acciones predichas
            pred_actions = self.actor(obs)
            
            # Pérdida de actor TD3
            actor_loss = -self.critic.q1(obs, pred_actions).mean()
            
            # Pérdida de clonación de comportamiento
            bc_loss = F.mse_loss(pred_actions, actions)
            
            # Pérdida total
            total_loss = actor_loss + self.alpha_bc * bc_loss
            
            self.actor.optimizer.zero_grad()
            total_loss.backward()
            self.actor.optimizer.step()
            
            # Actualizar crítico objetivo
            self._update_target(self.critic_target, self.critic)
            
            # Registrar pérdidas para depuración
            self.logger.record('train/actor_loss', actor_loss.item())
            self.logger.record('train/bc_loss', bc_loss.item())
            self.logger.record('train/critic_loss', critic_loss.item())
            
            # Verificar NaN en parámetros
            if torch.isnan(next(self.actor.parameters())) or torch.isnan(next(self.critic.parameters())):
                raise ValueError("NaN detectado en parámetros del modelo")
                
    def _update_target(self, target, source):
        """Actualiza la red objetivo usando soft update."""
        for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(target_param.data * (1.0 - self.tau) + param.data * self.tau)

class ProgressBarCallback(BaseCallback):
    def __init__(self, total_timesteps, fqe_iterations=1000):
        super().__init__()
        self.total_timesteps = total_timesteps
        self.fqe_iterations = fqe_iterations
        self.pbar = None
        self.best_fqe_value = float('-inf')
        self.best_model_path = None
        
    def _on_training_start(self):
        self.pbar = tqdm(total=self.total_timesteps, desc="Entrenando SAC")
        
    def _on_step(self):
        self.pbar.update(self.training_env.num_envs)
        
        # Obtener valores registrados
        ep_reward_mean = self.model.logger.name_to_value.get('rollout/ep_rew_mean', 0.0)
        actor_loss = self.model.logger.name_to_value.get('train/actor_loss', 0.0)
        critic_loss = self.model.logger.name_to_value.get('train/critic_loss', 0.0)
        bc_loss = self.model.logger.name_to_value.get('train/bc_loss', 0.0)
        
        self.pbar.set_postfix({
            'ep_reward_mean': f"{ep_reward_mean:.2f}",
            'actor_loss': f"{actor_loss:.2f}",
            'critic_loss': f"{critic_loss:.2f}",
            'bc_loss': f"{bc_loss:.2f}"
        })
        
        # Evaluar con FQE periódicamente
        if self.num_timesteps % 2000 == 0:
            # Verificación rápida de NaN antes de FQE
            validation_sample = self.training_env.envs[0].df.sample(n=100, seed=42)
            validation_env = DummyVecEnv([lambda: OhioT1DMEnv(validation_sample)])
            
            # Probar algunas predicciones
            obs = validation_env.reset()
            has_nan = False
            for _ in range(10):  # Probar 10 predicciones
                action, _ = self.model.predict(obs, deterministic=True)
                if np.isnan(action).any():
                    has_nan = True
                    logger.error("¡ALERTA! Modelo produciendo NaN en predicciones. Deteniendo entrenamiento.")
                    self.training_env.close()
                    return False
                obs, _, _, _ = validation_env.step(action)
            
            if not has_nan:
                logger.info("Verificación rápida pasada: No se detectaron NaN en predicciones. Procediendo con FQE...")
                fqe = FQE(
                    state_dim=self.model.observation_space.shape[0],
                    action_dim=self.model.action_space.shape[0],
                    device=self.model.device
                )
                fqe_value = fqe.evaluate(self.model, validation_env, n_iterations=self.fqe_iterations)
                
                if fqe_value > self.best_fqe_value:
                    self.best_fqe_value = fqe_value
                    self.best_model_path = f"new_ohio/models/sac_best_fqe_{fqe_value:.2f}"
                    self.model.save(self.best_model_path)
                    logger.info(f"Nuevo mejor modelo guardado con valor FQE: {fqe_value:.2f}")
        
        return True
        
    def _on_training_end(self):
        self.pbar.close()
        if self.best_model_path:
            logger.info(f"Mejor modelo guardado en: {self.best_model_path}")

def main():
    """Función principal para entrenar y evaluar el modelo SAC con DRL fuera de línea."""
    import argparse
    from stable_baselines3.common.policies import ActorCriticPolicy
    from torch import nn
    import torch.nn.functional as F
    
    # Habilitar detección de anomalías
    torch.autograd.set_detect_anomaly(True)
    
    parser = argparse.ArgumentParser(description='Entrenar o evaluar modelo SAC para OhioT1DM')
    parser.add_argument('--evaluate-only', action='store_true', 
                      help='Solo evaluar modelo guardado sin entrenar')
    parser.add_argument('--model-path', type=str, 
                      default='new_ohio/models/sac_ohiot1dm',
                      help='Ruta al modelo guardado (solo para evaluación)')
    parser.add_argument('--quick-train', action='store_true',
                      help='Entrenamiento rápido con 10% del dataset para validación/debug')
    parser.add_argument('--validation-split', type=float, default=0.2,
                      help='Fracción de datos de entrenamiento para validación')
    parser.add_argument('--fqe-iterations', type=int, default=1000,
                      help='Número de iteraciones FQE para evaluación de política')
    parser.add_argument('--test-user-cases-only', action='store_true',
                      help='Solo ejecutar pruebas de casos de usuario')
    parser.add_argument('--test-fqe-only', action='store_true',
                      help='Solo ejecutar evaluación FQE')
    
    args = parser.parse_args()
    
    # Definir rutas de archivos
    train_files = [
        "new_ohio/processed_data/train/processed_enhanced_2018_train.parquet",
        "new_ohio/processed_data/train/processed_enhanced_2020_train.parquet"
    ]
    test_files = [
        "new_ohio/processed_data/test/processed_enhanced_2018_test.parquet",
        "new_ohio/processed_data/test/processed_enhanced_2020_test.parquet"
    ]
    
    if args.evaluate_only:
        evaluate_saved_model(args.model_path, test_files)
        return
    
    logger.info("Iniciando entrenamiento SAC fuera de línea para OhioT1DM")
    
    # Cargar y validar datos
    train_dfs = []
    for f in train_files:
        df = pl.read_parquet(f)
        logger.info(f"Cargando {f}: {df.shape}")
        train_dfs.append(df)
    
    # Verificar columnas comunes
    common_columns = set.intersection(*[set(df.columns) for df in train_dfs])
    logger.info(f"Columnas comunes en datos de entrenamiento: {len(common_columns)}")
    
    # Seleccionar solo columnas comunes
    train_dfs = [df.select(list(common_columns)) for df in train_dfs]
    train_df = pl.concat(train_dfs)
    
    # Dividir en conjuntos de entrenamiento y validación
    if args.quick_train:
        SAMPLE_FRAC = 0.1
        logger.info(f"Realizando entrenamiento rápido de validación con {SAMPLE_FRAC*100:.0f}% del dataset...")
        train_df = train_df.sample(fraction=SAMPLE_FRAC, seed=42)
        validation_df = train_df.sample(fraction=args.validation_split, seed=42)
        train_df = train_df.filter(~train_df['Timestamp'].is_in(validation_df['Timestamp']))
        total_timesteps = 5000
    else:
        logger.info("Entrenando con dataset completo.")
        validation_df = train_df.sample(fraction=args.validation_split, seed=42)
        train_df = train_df.filter(~train_df['Timestamp'].is_in(validation_df['Timestamp']))
        total_timesteps = 20000
    
    logger.info(f"Conjunto de entrenamiento: {train_df.shape}")
    logger.info(f"Conjunto de validación: {validation_df.shape}")
    
    # Cargar datos de prueba
    test_dfs = []
    for f in test_files:
        df = pl.read_parquet(f)
        logger.info(f"Cargando {f}: {df.shape}")
        test_dfs.append(df)
    
    test_dfs = [df.select(list(common_columns)) for df in test_dfs]
    test_df = pl.concat(test_dfs)
    logger.info(f"Conjunto de prueba: {test_df.shape}")
    
    # Crear entornos
    train_env = DummyVecEnv([lambda: OhioT1DMEnv(train_df)])
    
    # Guardar estadísticas de normalización del entorno de entrenamiento
    train_env.envs[0].save_normalization_stats('new_ohio/models/normalization_stats.json')
    
    # Crear entornos de validación y prueba usando las mismas estadísticas
    state_mean, state_std, feature_columns = OhioT1DMEnv.load_normalization_stats('new_ohio/models/normalization_stats.json')
    validation_env = DummyVecEnv([lambda: OhioT1DMEnv(validation_df, state_mean, state_std)])
    test_env = DummyVecEnv([lambda: OhioT1DMEnv(test_df, state_mean, state_std)])
    
    # Configurar y entrenar SAC con TD3+BC
    model = SAC(
        CustomActorCriticPolicy,
        train_env,
        learning_rate=1e-3,
        buffer_size=20000,
        batch_size=128,
        tau=0.005,
        gamma=0.99,
        ent_coef=0.2,
        device='cuda' if torch.cuda.is_available() else 'cpu',
        verbose=1,
        policy_kwargs=dict(
            net_arch=dict(
                pi=[256, 256],
                qf=[256, 256]
            )
        )
    )
    
    # Prellenar buffer con datos históricos más diversos
    logger.info("Prellenando buffer de replay con datos históricos...")
    train_env.env_method('prefill_replay_buffer', model, n_transitions=5000)
    
    # Crear callback para barra de progreso y evaluación
    logger.info("Iniciando entrenamiento...")
    model.learn(
        total_timesteps=total_timesteps,
        callback=[ProgressBarCallback(total_timesteps, fqe_iterations=args.fqe_iterations)]
    )
    
    # Cargar mejor modelo basado en FQE
    if os.path.exists("new_ohio/models/sac_best_fqe_*"):
        best_model_path = max(glob.glob("new_ohio/models/sac_best_fqe_*"), key=os.path.getctime)
        model = SAC.load(best_model_path)
        logger.info(f"Cargado mejor modelo desde {best_model_path}")
    else:
        model_path = "new_ohio/models/sac_ohiot1dm"
        model.save(model_path)
        logger.info(f"Modelo guardado en {model_path}")
    
    # Ejecutar pruebas según los argumentos
    if args.test_user_cases_only:
        logger.info("Ejecutando solo pruebas de casos de usuario...")
        test_user_cases()
    elif args.test_fqe_only:
        logger.info("Ejecutando solo evaluación FQE...")
        results = evaluate_model(model, test_env, test_df)
    else:
        # Ejecutar ambas pruebas en orden
        logger.info("Ejecutando pruebas de casos de usuario...")
        test_user_cases()
        
        logger.info("Ejecutando evaluación FQE...")
        results = evaluate_model(model, test_env, test_df)

if __name__ == "__main__":
    main() 