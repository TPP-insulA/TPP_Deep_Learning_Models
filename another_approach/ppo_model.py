# %% CELL: Imports and Config
import gymnasium as gym
from gymnasium import spaces
import numpy as np
import polars as pl
import json
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
from copy import deepcopy
import os
import matplotlib.pyplot as plt
import seaborn as sns
from config import CONFIG, PREV_SAMPLES, POST_SAMPLES, MODEL_ID, PREPROCESSSING_ID

print("MODEL_ID:", MODEL_ID)
print("PREPROCESSSING_ID:", PREPROCESSSING_ID)


# %% CELL: Custom Gym Environment
class InsulinEnv(gym.Env):
    def __init__(self, data_path, standardization_params_path, permute_feature=None):
        super().__init__()

        if not os.path.exists(data_path):
            raise FileNotFoundError(f"No se encontró el archivo: {data_path}")
        if not os.path.exists(standardization_params_path):
            raise FileNotFoundError(f"No se encontró el archivo: {standardization_params_path}")

        self.data = pl.read_parquet(data_path)

        if permute_feature is not None:
            if permute_feature not in self.data.columns:
                raise ValueError(f"Feature {permute_feature} not in data columns")
            permuted_values = np.random.permutation(self.data[permute_feature].to_numpy())
            self.data = self.data.with_columns(pl.Series(permute_feature, permuted_values))

        with open(standardization_params_path, "r") as f:
            params = json.load(f)
        self.means = params["means"]
        self.stds = params["stds"]

        self.state_cols = [
            *[f"mg/dl_prev_{i+1}" for i in range(PREV_SAMPLES)],
            "carbInput",
            "insulinCarbRatio",
            "bgInput",
            "insulinOnBoard",
            "targetBloodGlucose",
        ]
        self.action_col = "normal"
        self.post_cols = [f"mg/dl_post_{i+1}" for i in range(POST_SAMPLES)]

        required_cols = self.state_cols + [self.action_col] + self.post_cols + ["subject_id", "bolus_date"]
        missing_cols = [col for col in required_cols if col not in self.data.columns]
        if missing_cols:
            raise ValueError(f"Faltan columnas: {missing_cols}")

        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(len(self.state_cols) + 1,), dtype=np.float32)
        self.action_space = spaces.Box(low=0.0, high=20.0, shape=(1,), dtype=np.float32)

        self.current_step = 0
        self.max_steps = self.data.height
        self.last_pred_dose = 0.0

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step = 0
        self.last_pred_dose = 0.0
        return self._get_state(), {}

    def step(self, action):
        pred_dose = round(float(action[0]) * 2) / 2
        real_dose = self.data[self.action_col][self.current_step]
        mgdl_post = self.data[self.post_cols].row(self.current_step)

        reward = self._calculate_reward(pred_dose, real_dose, mgdl_post)

        info = {
            "pred_dose": float(pred_dose),
            "real_dose": float(real_dose),
            "mgdl_prev": self.data[self.state_cols[:PREV_SAMPLES]].row(self.current_step),
            "mgdl_post": mgdl_post,
            "subject_id": self.data["subject_id"][self.current_step],
            "bolus_date": self.data["bolus_date"][self.current_step],
            "reward": float(reward),
        }

        self.last_pred_dose = float(pred_dose)
        self.current_step += 1
        terminated = self.current_step >= self.max_steps
        truncated = False
        next_state = self._get_state() if not terminated else np.zeros(len(self.state_cols) + 1, dtype=np.float32)

        return next_state, reward, terminated, truncated, info

    def _get_state(self):
        state = self.data[self.state_cols].row(self.current_step)
        return np.array(list(state) + [self.last_pred_dose], dtype=np.float32)

    def _calculate_reward(self, pred_dose, real_dose, mgdl_post):
        avg_mgdl = np.mean(mgdl_post)
        final_mgdl = mgdl_post[-1]
        std_post = np.std(mgdl_post)
        rel_error = abs(pred_dose - real_dose) / (real_dose + 1e-5)

        reward = np.exp(-1.5 * rel_error)  # penaliza error relativo en general

        # 🔴 HIPOgulcemia: foco en evitar sobredosis
        if final_mgdl < 70:
            if pred_dose > real_dose:
                reward -= 2.0  # penaliza fuertemente sobredosificación
            elif pred_dose < real_dose:
                reward += 1.0  # premia corrección adecuada
            else:
                reward -= 0.5  # castigo por inacción

        # 🟢 RANGO NORMAL: bonus + penalización de variabilidad
        elif 70 <= final_mgdl <= 180:
            reward += 0.5
            reward -= std_post / 100  # penaliza variabilidad solo si está en rango

        # 🔶 HIPERglucemia severa
        elif final_mgdl > 300:
            reward -= 0.5

        # 🟡 HIPER leve: refuerzo explícito
        if avg_mgdl > 180:
            if pred_dose > real_dose:
                reward += 0.5  # incentiva subir dosis ante hiper
                if pred_dose > real_dose * 1.5:
                    reward -= 0.3  # penaliza sobrecorrección leve
            elif final_mgdl > 180:
                reward -= 0.5  # no hizo nada y sigue alto

        # ✅ BONUS por corrección efectiva hacia el rango
        if avg_mgdl > 180 and 70 <= final_mgdl <= 180:
            reward += 0.7  # corregiste hiper
        elif avg_mgdl < 70 and 70 <= final_mgdl <= 180:
            reward += 0.5  # corregiste hipo

        return float(reward)

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
        learning_rate=1e-4,
        n_steps=2048,
        batch_size=256,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        device="cpu",
    )

    model.learn(total_timesteps=1000000)

    return model, env


# %% CELL: Evaluate Model
def evaluate_model(model, env, dataset_type="val", save_predictions=True):
    obs, _ = env.reset()
    total_reward = 0
    steps = 0
    results = []

    while True:
        action, _states = model.predict(obs)
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        steps += 1
        if save_predictions:
            results.append(
                {
                    "subject_id": info["subject_id"],
                    "bolus_date": info["bolus_date"],
                    "pred_dose": info["pred_dose"],
                    "real_dose": info["real_dose"],
                    **{f"mg/dl_prev_{i+1}": info["mgdl_prev"][i] for i in range(PREV_SAMPLES)},
                    **{f"mg/dl_post_{i+1}": info["mgdl_post"][i] for i in range(POST_SAMPLES)},
                    "reward": info["reward"],
                }
            )
        if terminated or truncated:
            break

    if save_predictions:
        try:
            results_df = pl.DataFrame(results)
            csv_path = os.path.join(CONFIG["processed_data_path"], f"ppo_predictions_{dataset_type}_{PREPROCESSSING_ID}_{MODEL_ID}.csv")
            results_df.write_csv(csv_path)
            print(f"Predicciones guardadas en: {csv_path}")
        except Exception as e:
            print(f"Error al guardar el CSV: {e}")
            raise

    print(f"Evaluación completada ({dataset_type}). Recompensa total: {total_reward}, Pasos: {steps}")
    return total_reward, results


# %% CELL: Feature Importance
def permutation_importance(model, env, n_repeats=5, dataset_type="val"):
    """
    Calculate feature importance using Permutation Importance for a PPO model.

    Args:
        model: Trained PPO model.
        env: Validation environment (InsulinEnv).
        n_repeats: Number of repetitions to average results.
        dataset_type: Dataset type (e.g., "val").

    Returns:
        pl.DataFrame: DataFrame with feature names and their importance scores.
    """
    state_cols = env.state_cols
    results = {col: [] for col in state_cols}

    # Evaluate baseline performance
    print("Evaluating baseline performance...")
    base_rewards = []
    for _ in range(n_repeats):
        total_reward, _ = evaluate_model(model, deepcopy(env), dataset_type=dataset_type, save_predictions=False)
        base_rewards.append(total_reward)
    base_reward_mean = np.mean(base_rewards)
    print(f"Average baseline reward: {base_reward_mean:.2f}")

    # Evaluate each feature
    for col in state_cols:
        print(f"Evaluating feature: {col}")
        perm_rewards = []

        for _ in range(n_repeats):
            # Create a permuted environment
            perm_env = InsulinEnv(
                os.path.join(CONFIG["processed_data_path"], f"{dataset_type}_all_{PREPROCESSSING_ID}.parquet"),
                os.path.join(CONFIG["params_path"], f"state_standardization_params_{PREPROCESSSING_ID}.json"),
                permute_feature=col,
            )
            # Evaluate model in permuted environment
            total_reward, _ = evaluate_model(model, perm_env, dataset_type=dataset_type, save_predictions=False)
            perm_rewards.append(total_reward)

        # Calculate importance as the decrease in reward
        perm_reward_mean = np.mean(perm_rewards)
        importance = base_reward_mean - perm_reward_mean
        results[col].append(importance)
        print(f"Importance of {col}: {importance:.2f} (Permuted reward: {perm_reward_mean:.2f})")

    # Save results to CSV
    results_df = pl.DataFrame({"feature": state_cols, "importance": [np.mean(results[col]) for col in state_cols]})
    csv_path = os.path.join(CONFIG["processed_data_path"], f"feature_importance_{dataset_type}_{PREPROCESSSING_ID}_{MODEL_ID}.csv")
    results_df.write_csv(csv_path)
    print(f"Feature importance results saved to: {csv_path}")

    return results_df


# %% CELL: Plot Feature Importance
def plot_feature_importance(importance_df):
    """
    Plot feature importance as a bar chart.

    Args:
        importance_df: DataFrame with feature names and importance scores.
    """
    plt.figure(figsize=(12, 6))
    sns.barplot(data=importance_df.to_pandas(), x="feature", y="importance")
    plt.xticks(rotation=45, ha="right")
    plt.xlabel("Features")
    plt.ylabel("Importance (Reward Decrease)")
    plt.title("Feature Importance (Permutation Importance)")
    plt.tight_layout()

    # Save plot
    plot_path = os.path.join(CONFIG["processed_data_path"], f"feature_importance_plot_{PREPROCESSSING_ID}_{MODEL_ID}.png")
    plt.savefig(plot_path)
    print(f"Feature importance plot saved to: {plot_path}")
    plt.close()


# %% CELL: Run Training, Evaluation, and Feature Importance
if __name__ == "__main__":
    try:
        # Train the model
        model, train_env = train_ppo()

        # Create validation environment
        val_data_path = os.path.join(CONFIG["processed_data_path"], f"val_all_{PREPROCESSSING_ID}.parquet")
        params_path = os.path.join(CONFIG["params_path"], f"state_standardization_params_{PREPROCESSSING_ID}.json")
        val_env = InsulinEnv(val_data_path, params_path)

        # Step 1: Evaluate the model
        print("Running model evaluation...")
        total_reward, eval_results = evaluate_model(model, val_env, dataset_type="val", save_predictions=True)

        # Step 2: Compute feature importance
        # print("Computing feature importance...")
        # importance_df = permutation_importance(model, val_env, n_repeats=5, dataset_type="val")

        # Step 3: Plot feature importance
        # print("Generating feature importance plot...")
        # plot_feature_importance(importance_df)

        # print("Execution completed successfully.")
        # print(importance_df)
    except Exception as e:
        print(f"Error in main execution: {e}")
        raise
