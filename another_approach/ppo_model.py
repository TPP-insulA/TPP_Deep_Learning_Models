# %% CELL: Imports and Config
import gymnasium as gym
from gymnasium import spaces
import numpy as np
import polars as pl
import json
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
import os
from config import CONFIG, PREV_SAMPLES, POST_SAMPLES, MODEL_ID, PREPROCESSSING_ID

# %% CELL: Custom Gym Environment
class InsulinEnv(gym.Env):
    def __init__(self, data_path, standardization_params_path):
        super(InsulinEnv, self).__init__()
        
        # Verificar que los archivos existen
        if not os.path.exists(data_path):
            raise FileNotFoundError(f"No se encontró el archivo: {data_path}")
        if not os.path.exists(standardization_params_path):
            raise FileNotFoundError(f"No se encontró el archivo: {standardization_params_path}")
        
        # Cargar datos (ya normalizados)
        self.data = pl.read_parquet(data_path)
        
        # Cargar parámetros de estandarización (para referencia)
        with open(standardization_params_path, "r") as f:
            params = json.load(f)
        self.means = params["means"]
        self.stds = params["stds"]
        
        # Definir columnas
        self.state_cols = [
            *[f"mg/dl_prev_{i+1}" for i in range(PREV_SAMPLES)],
            "carbInput", "insulinCarbRatio", "bgInput", "insulinOnBoard", "targetBloodGlucose"
        ]
        self.action_col = "normal"
        self.post_cols = [f"mg/dl_post_{i+1}" for i in range(POST_SAMPLES)]
        
        # Verificar que las columnas necesarias existen
        required_cols = self.state_cols + [self.action_col] + self.post_cols + ["subject_id", "bolus_date"]
        missing_cols = [col for col in required_cols if col not in self.data.columns]
        if missing_cols:
            raise ValueError(f"Faltan columnas en los datos: {missing_cols}")
        
        # Espacios de estado y acción
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(len(self.state_cols),), dtype=np.float32
        )
        self.action_space = spaces.Box(
            low=0, high=20, shape=(1,), dtype=np.float32  # Restaurado a [0, 20]
        )
        
        self.current_step = 0
        self.max_steps = self.data.height
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step = 0
        return self._get_state(), {}
    
    def step(self, action):
        # Obtener acción predicha (dosis) directamente
        pred_dose = action[0]  # Sin desnormalización, ya que action_space es [0, 20]
        
        # Obtener datos reales
        real_dose = self.data[self.action_col][self.current_step]
        mgdl_post = self.data[self.post_cols].row(self.current_step)
        
        # Calcular recompensa
        reward = self._calculate_reward(pred_dose, real_dose, mgdl_post)
        
        # Construir info antes de incrementar el paso
        info = {
            "pred_dose": float(pred_dose),
            "real_dose": float(real_dose),
            "mgdl_prev": self.data[self.state_cols[:PREV_SAMPLES]].row(self.current_step),
            "mgdl_post": mgdl_post,
            "subject_id": self.data["subject_id"][self.current_step],
            "bolus_date": self.data["bolus_date"][self.current_step],
            "reward": float(reward)
        }
        
        # Avanzar paso
        self.current_step += 1
        terminated = self.current_step >= self.max_steps
        truncated = False
        next_state = self._get_state() if not terminated else np.zeros(len(self.state_cols), dtype=np.float32)
        
        return next_state, reward, terminated, truncated, info
    
    def _get_state(self):
        # Obtener estado directamente (ya está normalizado)
        state = self.data[self.state_cols].row(self.current_step)
        return np.array(state, dtype=np.float32)
    
    def _calculate_reward(self, pred_dose, real_dose, mgdl_post):
        avg_mgdl = np.mean(mgdl_post)
        
        NORMAL_LOW = 70
        NORMAL_HIGH = 180
        DOSE_THRESHOLD = 1.0
        
        if NORMAL_LOW <= avg_mgdl <= NORMAL_HIGH:
            if abs(pred_dose - real_dose) < DOSE_THRESHOLD:
                return 1.0
            else:
                return -0.5
        elif avg_mgdl > NORMAL_HIGH:
            if pred_dose > real_dose:
                return 0.5
            else:
                return -1.0
        else:
            if pred_dose < real_dose:
                return 0.5
            else:
                return -1.0
    
    def render(self):
        pass

# %% CELL: Train PPO Model
def train_ppo():
    data_path = os.path.join(CONFIG["processed_data_path"], f"train_all_{PREPROCESSSING_ID}.parquet")
    params_path = os.path.join(CONFIG["params_path"], f"state_standardization_params_{PREPROCESSSING_ID}.json")
    
    try:
        env = InsulinEnv(data_path, params_path)
    except Exception as e:
        print(f"Error al crear el ambiente de entrenamiento: {e}")
        raise
    
    check_env(env)
    
    model = PPO(
        "MlpPolicy",
        env,
        verbose=1,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        device="cpu"  # Forzar CPU
    )
    
    model.learn(total_timesteps=500000)  # Aumentado para mejor convergencia
    
    # Descomentar para guardar el modelo
    # model.save(os.path.join(CONFIG["processed_data_path"], f"ppo_insulin_model_{PREPROCESSING_ID}_{MODEL_ID}"))
    
    return model, env

# %% CELL: Evaluate Model
def evaluate_model(model, env, dataset_type="val"):
    obs, _ = env.reset()
    total_reward = 0
    steps = 0
    results = []
    
    while True:
        action, _states = model.predict(obs)
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        steps += 1
        
        # Construir fila para el CSV
        result_row = {
            "subject_id": info["subject_id"],
            "bolus_date": info["bolus_date"],
            "pred_dose": info["pred_dose"],
            "real_dose": info["real_dose"],
            **{f"mg/dl_prev_{i+1}": info["mgdl_prev"][i] for i in range(PREV_SAMPLES)},
            **{f"mg/dl_post_{i+1}": info["mgdl_post"][i] for i in range(POST_SAMPLES)},
            "reward": info["reward"]
        }
        results.append(result_row)
        
        if terminated or truncated:
            break
    
    # Guardar resultados en CSV
    try:
        results_df = pl.DataFrame(results)
        csv_path = os.path.join(CONFIG["processed_data_path"], f"ppo_predictions_{dataset_type}_{PREPROCESSSING_ID}_{MODEL_ID}.csv")
        results_df.write_csv(csv_path)
        print(f"Predicciones guardadas en: {csv_path}")
    except Exception as e:
        print(f"Error al guardar el CSV: {e}")
        raise
    
    print(f"Evaluación completada ({dataset_type}). Recompensa total: {total_reward}, Pasos: {steps}")
    return total_reward

# %% CELL: Run Training and Evaluation
if __name__ == "__main__":
    try:
        model, train_env = train_ppo()
        
        val_data_path = os.path.join(CONFIG["processed_data_path"], f"val_all_{PREPROCESSSING_ID}.parquet")
        val_env = InsulinEnv(val_data_path, os.path.join(CONFIG["params_path"], f"state_standardization_params_{PREPROCESSSING_ID}.json"))
        evaluate_model(model, val_env, dataset_type="val")
    except Exception as e:
        print(f"Error en la ejecución principal: {e}")
        raise
# %%
