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

# Add parent directory to path to import from processing
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from new_ohio.processing.ohio_polars import CONFIG, ColoredFormatter, simulate_glucose

# Configure logging
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
    
    def __init__(self, df: pl.DataFrame):
        super().__init__()
        
        # Guardar DataFrame
        self.df = df
        self.current_step = 0
        
        # Definir columnas de características (excluyendo Timestamp, SubjectID, bolus)
        self.feature_columns = [col for col in df.columns 
                              if col not in ['Timestamp', 'SubjectID', 'bolus']]
        
        # Calcular estadísticas para normalización
        self.state_mean = df.select(self.feature_columns).mean().to_numpy()
        self.state_std = df.select(self.feature_columns).std().to_numpy() + 1e-8
        
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
        action = np.array(action, dtype=np.float32).reshape(-1)
        
        # Aplicar capa de seguridad
        if self.current_glucose < 100 and action[0] > 10:
            action[0] = min(action[0], 10.0)
            reward = -5.0  # Penalización por acción insegura
        else:
            reward = 0.0
        
        # Obtener estado actual
        current_state = self._get_state_dict()
        
        # Simular siguiente estado de glucosa
        next_glucose = simulate_glucose(
            cgm_values=[self.current_glucose],
            bolus=float(action[0]),  # Convertir a float para simulate_glucose
            carbs=current_state.get('meal_carbs', 0.0),
            basal_rate=current_state.get('effective_basal_rate', 0.0),
            exercise_intensity=current_state.get('exercise_intensity', 0.0)
        )
        
        # Avanzar al siguiente paso
        self.current_step += 1
        self._update_state()
        
        # Calcular recompensa
        reward += self._compute_reward(current_state, float(action[0]), next_glucose)
        
        # Verificar si terminó el episodio
        terminated = self.current_step >= len(self.df)
        truncated = False  # No hay truncamiento en este entorno
        
        return self._get_observation(), reward, terminated, truncated, {}
    
    def _get_observation(self) -> np.ndarray:
        """Obtiene la observación normalizada del estado actual."""
        if self.current_state is None:
            return np.zeros(len(self.feature_columns), dtype=np.float32)
        
        # Convertir estado a array y normalizar
        try:
            # Asegurar que todos los valores sean float32
            state_values = []
            for col in self.feature_columns:
                val = self.current_state[col]
                if isinstance(val, (list, np.ndarray)):
                    val = float(val[0])  # Tomar el primer valor si es una secuencia
                state_values.append(float(val))
            
            state_array = np.array(state_values, dtype=np.float32)
            normalized_state = (state_array - self.state_mean) / self.state_std
            
            return normalized_state.astype(np.float32)
        except Exception as e:
            logger.error(f"Error en _get_observation: {e}")
            logger.error(f"Estado actual: {self.current_state}")
            logger.error(f"Columnas: {self.feature_columns}")
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
        
        Args:
            state: Estado actual
            action: Acción tomada (dosis de bolus)
            next_glucose: Siguiente estado de glucosa simulado
            
        Returns:
            float: Recompensa calculada
        """
        glucose = next_glucose['simulated_mean_6h']
        
        # Recompensa por tiempo en rango
        tir = 1.0 if CONFIG['tir_lower'] <= glucose <= CONFIG['tir_upper'] else 0.0
        
        # Penalizaciones por hipo/hiperglucemia
        hypo_penalty = -10.0 if glucose < CONFIG['hypoglycemia_threshold'] else 0.0
        hyper_penalty = -2.0 if glucose > CONFIG['hyperglycemia_threshold'] else 0.0
        
        # Penalización por desviación del bolus real
        real_bolus = state.get('bolus', 0.0)
        bolus_diff = -0.1 * (action - real_bolus)**2
        
        # Calcular recompensa total como escalar
        total_reward = tir + hypo_penalty + hyper_penalty + bolus_diff
        return total_reward
    
    def prefill_replay_buffer(self, model: SAC, n_transitions: int = 10000):
        """
        Prellena el buffer de repetición con transiciones simuladas.
        
        Args:
            model: Modelo SAC
            n_transitions: Número de transiciones a generar
        """
        logger.info(f"Prellenando buffer con {n_transitions} transiciones...")
        
        for _ in range(n_transitions):
            state, _ = self.reset()  # Unpack the tuple returned by reset()
            # Generar acción como array de shape (1,)
            action = np.array([np.random.uniform(
                CONFIG['bolus_min'], 
                CONFIG['bolus_max']
            )], dtype=np.float32)
            next_state, reward, terminated, truncated, infos = self.step(action)
            # Logging de shapes para depuración
            logger.info(f"state shape: {state.shape}, action shape: {action.shape}, reward shape: {np.array([reward], dtype=np.float32).shape}, next_state shape: {next_state.shape}, done shape: {np.array([terminated or truncated], dtype=bool).shape}")
            model.replay_buffer.add(
                np.expand_dims(state, axis=0),           # obs
                np.expand_dims(next_state, axis=0),      # next_obs
                np.expand_dims(action, axis=0),          # action
                np.array([reward], dtype=np.float32),    # reward
                np.array([terminated or truncated], dtype=bool),  # done
                [infos]                                  # infos
            )
        
        logger.info("Buffer prellenado completado")

def evaluate_model(model: SAC, env: DummyVecEnv, df_test: pl.DataFrame) -> Dict:
    """
    Evalúa el modelo en datos de prueba.
    
    Args:
        model: Modelo SAC entrenado
        env: Entorno vectorizado
        df_test: DataFrame con datos de prueba
        
    Returns:
        Dict con métricas de evaluación
    """
    logger.info("Iniciando evaluación del modelo...")
    
    predictions = []
    actuals = []
    glucose_values = []
    
    obs = env.reset()
    done = False
    
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        # Convertir action[0] a float de manera segura
        action_value = action[0].item() if isinstance(action[0], np.ndarray) else float(action[0])
        predictions.append(action_value)
        
        # Obtener valores reales
        current_state = env.envs[0]._get_state_dict()
        actuals.append(float(current_state.get('bolus', 0.0)))
        glucose_values.append(float(current_state.get('glucose_last', 120.0)))
        
        # Manejar el paso del entorno vectorizado
        step_result = env.step(action)
        if len(step_result) == 4:  # Versión antigua de Gym
            obs, reward, done, info = step_result
            terminated, truncated = done, False
        else:  # Nueva versión de Gymnasium
            obs, reward, terminated, truncated, info = step_result
            done = terminated or truncated
    
    # Convertir a arrays numpy y asegurar que tengan la forma correcta
    predictions = np.array(predictions, dtype=np.float32)
    actuals = np.array(actuals, dtype=np.float32)
    glucose_values = np.array(glucose_values, dtype=np.float32)
    
    # Calcular métricas
    mae = np.mean(np.abs(predictions - actuals))
    corr = np.corrcoef(predictions, actuals)[0, 1]
    
    tir = np.mean([
        1 if CONFIG['tir_lower'] <= g <= CONFIG['tir_upper'] else 0 
        for g in glucose_values
    ])
    
    hypo = np.mean([
        1 if g < CONFIG['hypoglycemia_threshold'] else 0 
        for g in glucose_values
    ])
    
    hyper = np.mean([
        1 if g > CONFIG['hyperglycemia_threshold'] else 0 
        for g in glucose_values
    ])
    
    results = {
        'MAE': float(mae),
        'Correlation': float(corr),
        'TIR': float(tir),
        'Hypo': float(hypo),
        'Hyper': float(hyper)
    }
    
    logger.info(f"Resultados de evaluación: {results}")
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

def main():
    """Función principal para entrenar y evaluar el modelo SAC."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Entrenar o evaluar modelo SAC para OhioT1DM')
    parser.add_argument('--evaluate-only', action='store_true', 
                      help='Solo evaluar el modelo guardado sin entrenar')
    parser.add_argument('--model-path', type=str, 
                      default='new_ohio/models/sac_ohiot1dm',
                      help='Ruta al modelo guardado (solo para evaluación)')
    
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
    
    logger.info("Iniciando entrenamiento SAC para OhioT1DM")
    
    # Cargar datos
    train_df = pl.concat([pl.read_parquet(f) for f in train_files])
    test_df = pl.concat([pl.read_parquet(f) for f in test_files])
    
    logger.info(f"Datos de entrenamiento cargados: {train_df.shape}")
    logger.info(f"Datos de prueba cargados: {test_df.shape}")
    
    # Crear entorno
    env = DummyVecEnv([lambda: OhioT1DMEnv(train_df)])
    
    # Configurar y entrenar SAC
    model = SAC(
        "MlpPolicy",
        env,
        learning_rate=3e-4,
        buffer_size=100000,
        batch_size=256,
        tau=0.005,
        gamma=0.99,
        ent_coef='auto',
        verbose=1
    )
    
    # Prellenar buffer
    env.env_method('prefill_replay_buffer', model, n_transitions=10000)
    
    # Crear callback para la barra de progreso
    class ProgressBarCallback(BaseCallback):
        def __init__(self, total_timesteps):
            super().__init__()
            self.total_timesteps = total_timesteps
            self.pbar = None
            
        def _on_training_start(self):
            self.pbar = tqdm(total=self.total_timesteps, desc="Entrenando SAC")
            
        def _on_step(self):
            self.pbar.update(self.training_env.num_envs)
            
            # Obtener la recompensa media del episodio si hay episodios completados
            ep_reward = 0.0
            if len(self.model.ep_info_buffer) > 0:
                ep_reward = self.model.ep_info_buffer[-1]['r']
            
            self.pbar.set_postfix({
                'ep_reward_mean': f"{ep_reward:.2f}",
                'loss': f"{self.model.logger.name_to_value.get('train/loss', 0):.2f}"
            })
            return True
            
        def _on_training_end(self):
            self.pbar.close()
    
    # Entrenar modelo con barra de progreso
    total_timesteps = 100000
    logger.info("Iniciando entrenamiento...")
    model.learn(
        total_timesteps=total_timesteps,
        callback=ProgressBarCallback(total_timesteps)
    )
    
    # Guardar modelo
    model_path = "new_ohio/models/sac_ohiot1dm"
    model.save(model_path)
    logger.info(f"Modelo guardado en {model_path}")
    
    # Evaluar
    test_env = DummyVecEnv([lambda: OhioT1DMEnv(test_df)])
    results = evaluate_model(model, test_env, test_df)
    
    # Guardar resultados
    results_path = "new_ohio/models/evaluation_results.txt"
    with open(results_path, "w") as f:
        f.write(str(results))
    logger.info(f"Resultados guardados en {results_path}")

if __name__ == "__main__":
    main() 