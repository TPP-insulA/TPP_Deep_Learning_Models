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
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.utils import set_random_seed
import tensorboard
from tensorboard.backend.event_processing import event_accumulator
import logging
import glob
from datetime import datetime
from types import SimpleNamespace
import pandas as pd
import jax as jnp
from typing import Any

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class OhioT1DMEnv(gym.Env):
    """
    Custom Gym environment for OhioT1DM insulin bolus prediction.
    
    Observation:
        - CGM values (24 timesteps from cgm_0 to cgm_23)
        - hour_of_day (normalized)
        - bolus_log1p
        - carb_input_log1p
        - insulin_on_board_log1p
        - meal_carbs_log1p
        - meal_time_diff_hours
        - has_meal
        - meals_in_window
    
    Action:
        - Continuous bolus dose in [0, 30] units
    
    Reward:
        - Negative absolute error between predicted and true bolus
    """
    
    def __init__(self, df_windows: pl.DataFrame, df_final: pl.DataFrame):
        super().__init__()
        
        # Validate required columns
        required_cols = ['hour_of_day', 'bolus', 'carb_input'] + [f'cgm_{i}' for i in range(24)]
        for col in required_cols:
            if col not in df_windows.columns:
                raise ValueError(f"Required column '{col}' not found in DataFrame")
        
        self.df_windows = df_windows
        self.df_final = df_final
        self.current_idx = 0
        
        # Define action space (continuous bolus dose)
        self.action_space = gym.spaces.Box(
            low=0.0,
            high=30.0,
            shape=(1,),
            dtype=np.float32
        )
        
        # Define observation space
        self.observation_space = gym.spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(32,),  # 24 CGM + 8 features
            dtype=np.float32
        )
        
        # Store current episode data
        self.current_episode_data = None
        self.episode_length = 0
        
    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None) -> Tuple[np.ndarray, dict]:
        """Reset environment to start of episode."""
        super().reset(seed=seed)
        self.current_idx = 0
        self.episode_length = len(self.df_windows)
        
        # Get initial observation
        obs = self._get_observation()
        return obs, {}  # Return empty info dict as required by gymnasium
    
    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """
        Execute one time step within the environment.
        
        Args:
            action: Predicted bolus dose
            
        Returns:
            observation: Next observation
            reward: Reward for the current step
            terminated: Whether episode is done
            truncated: Whether episode was truncated
            info: Additional information
        """
        # Get true bolus for current timestep
        true_bolus = self.df_windows['bolus'][self.current_idx]
        
        # Calculate reward (negative absolute error)
        reward = -abs(action[0] - true_bolus)
        
        # Move to next timestep
        self.current_idx += 1
        terminated = self.current_idx >= self.episode_length
        truncated = False  # We don't use truncation in this environment
        
        # Get next observation
        obs = self._get_observation() if not terminated else np.zeros_like(self._get_observation())
        
        info = {
            'true_bolus': true_bolus,
            'predicted_bolus': action[0]
        }
        
        return obs, reward, terminated, truncated, info
    
    def _get_observation(self) -> np.ndarray:
        """Construct observation vector from current timestep."""
        if self.current_idx >= len(self.df_windows):
            return np.zeros(32, dtype=np.float32)
            
        # Get CGM values for the current timestep
        cgm_values = [self.df_windows[f'cgm_{i}'][self.current_idx] for i in range(24)]
        
        # Get additional features
        hour_of_day = self.df_windows['hour_of_day'][self.current_idx]
        bolus_log = np.log1p(self.df_windows['bolus'][self.current_idx])
        carb_log = np.log1p(self.df_windows['carb_input'][self.current_idx])
        
        # Handle insulin_on_board (use 0.0 if column doesn't exist)
        try:
            iob_log = np.log1p(self.df_windows['insulin_on_board'][self.current_idx])
        except:
            iob_log = 0.0
            
        # Handle meal-related features
        try:
            meal_carbs_log = np.log1p(self.df_windows['meal_carbs'][self.current_idx])
            meal_time_diff = self.df_windows['meal_time_diff_hours'][self.current_idx]
            has_meal = self.df_windows['has_meal'][self.current_idx]
            meals_in_window = self.df_windows['meals_in_window'][self.current_idx]
        except:
            meal_carbs_log = 0.0
            meal_time_diff = 0.0
            has_meal = 0.0
            meals_in_window = 0.0
        
        # Combine into observation vector
        obs = np.concatenate([
            cgm_values,
            [hour_of_day, bolus_log, carb_log, iob_log,
             meal_carbs_log, meal_time_diff, has_meal, meals_in_window]
        ])
        
        return obs.astype(np.float32)

class OhioT1DMInferenceEnv(gym.Env):
    """
    Optimized Gym environment for OhioT1DM insulin bolus prediction during inference.
    This version doesn't require true bolus values and is designed for single-step predictions.
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
        
        # Define observation space
        self.observation_space = gym.spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(32,),  # 24 CGM + 8 features
            dtype=np.float32
        )
        
        # Store current observation
        self.current_observation = None
    
    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None) -> Tuple[np.ndarray, dict]:
        """Reset environment with a new observation."""
        super().reset(seed=seed)
        if self.current_observation is None:
            self.current_observation = np.zeros(32, dtype=np.float32)
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

def prepare_observation(
    cgm_values: List[float],
    carb_input: float,
    iob: float,
    timestamp: datetime,
    meal_carbs: float = 0.0,
    meal_time_diff: float = 0.0,
    has_meal: float = 0.0,
    meals_in_window: int = 0
) -> np.ndarray:
    """
    Prepare observation vector from user inputs for model prediction.
    
    Args:
        cgm_values: List of glucose values (up to 24)
        carb_input: Carbohydrate input in grams
        iob: Insulin on board in units
        timestamp: Current timestamp
        meal_carbs: Carbohydrates from meal in grams
        meal_time_diff: Time difference to meal in hours
        has_meal: Whether there is a meal (0 or 1)
        meals_in_window: Number of meals in the window
        
    Returns:
        Observation vector of 32 elements
    """
    # Process CGM values
    cgm_processed = []
    for i in range(24):
        if i < len(cgm_values):
            # Apply log1p to valid CGM values
            cgm_processed.append(np.log1p(cgm_values[i]))
        else:
            # Pad with zeros for missing values
            cgm_processed.append(0.0)
    
    # Process hour of day (normalize between 0 and 1)
    hour_of_day = timestamp.hour + timestamp.minute / 60.0
    hour_of_day = hour_of_day / 24.0
    
    # Process other features
    carb_log = np.log1p(carb_input)
    iob_log = np.log1p(iob)
    bolus_log = 0.0  # Set to 0 during inference
    
    # Process meal-related features
    meal_carbs_log = np.log1p(meal_carbs)
    
    # Combine into observation vector
    obs = np.concatenate([
        cgm_processed,
        [hour_of_day, bolus_log, carb_log, iob_log,
         meal_carbs_log, meal_time_diff, has_meal, meals_in_window]
    ])
    
    return obs.astype(np.float32)

def predict_insulin_dose(
    model_path: str,
    observation: np.ndarray
) -> float:
    """
    Predict insulin dose using trained PPO model.
    
    Args:
        model_path: Path to trained PPO model
        observation: Prepared observation vector
        
    Returns:
        Predicted insulin dose
    """
    # Load trained model
    model = PPO.load(model_path)
    
    # Create inference environment
    env = OhioT1DMInferenceEnv()
    env.set_observation(observation)
    
    # Get prediction
    obs, _ = env.reset()
    action, _ = model.predict(obs, deterministic=True)
    
    return float(action[0])

def evaluate_metrics(predictions: List[float], actuals: List[float]) -> Dict[str, float]:
    """
    Calculate comprehensive evaluation metrics.
    
    Args:
        predictions: List of predicted bolus doses
        actuals: List of actual bolus doses
        
    Returns:
        Dictionary containing MAE, MSE, RMSE, and percentage of predictions within safe range
    """
    predictions = np.array(predictions)
    actuals = np.array(actuals)
    
    # Basic metrics
    mae = np.mean(np.abs(predictions - actuals))
    mse = np.mean((predictions - actuals) ** 2)
    rmse = np.sqrt(mse)
    
    # Safety metrics
    error = predictions - actuals
    overdose_rate = np.mean(error > 0) * 100
    underdose_rate = np.mean(error < 0) * 100
    
    # Percentage within safe range (±20% of actual dose)
    safe_range = np.abs(error) <= (0.2 * actuals)
    safe_rate = np.mean(safe_range) * 100
    
    return {
        'mae': mae,
        'mse': mse,
        'rmse': rmse,
        'overdose_rate': overdose_rate,
        'underdose_rate': underdose_rate,
        'safe_rate': safe_rate
    }

def predict_from_preprocessed(
    model_path: str,
    preprocessed_data_path: str
) -> Dict[str, float]:
    """
    Make predictions using preprocessed data files.
    
    Args:
        model_path: Path to trained PPO model
        preprocessed_data_path: Path to preprocessed parquet file
        
    Returns:
        Dictionary containing predictions and true values
    """
    # Load preprocessed data
    df_windows = pl.read_parquet(preprocessed_data_path)
    df_final = pl.read_parquet(preprocessed_data_path)
    
    # Create environment
    env = OhioT1DMEnv(df_windows, df_final)
    
    # Load model
    model = PPO.load(model_path)
    
    # Run predictions
    obs, _ = env.reset()
    done = False
    predictions = []
    true_values = []
    
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, _, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        predictions.append(info['predicted_bolus'])
        true_values.append(info['true_bolus'])
    
    return {
        'predictions': predictions,
        'true_values': true_values,
        'mae': np.mean(np.abs(np.array(predictions) - np.array(true_values)))
    }

def make_env(df_windows: pl.DataFrame, df_final: pl.DataFrame, rank: int, seed: int = 0) -> gym.Env:
    """
    Create a wrapped environment for parallel training.
    
    Args:
        df_windows: DataFrame with CGM windows
        df_final: DataFrame with final features
        rank: Environment rank
        seed: Random seed
        
    Returns:
        Wrapped environment
    """
    def _init():
        env = OhioT1DMEnv(df_windows, df_final)
        env.reset(seed=seed + rank)  # Set seed during reset
        env = Monitor(env)
        return env
    set_random_seed(seed)
    return _init

def train_with_hyperparameters(
    train_dirs: List[str],
    output_dir: str,
    tensorboard_log: str,
    total_timesteps: int,
    n_envs: int = 4,
    learning_rate: float = 0.0001,
    batch_size: int = 128,
    net_arch: List[int] = [128, 128],
    gamma: float = 0.99,
    n_steps: int = 2048,
    n_epochs: int = 10
) -> PPO:
    """
    Train PPO agent with customizable hyperparameters.
    
    Args:
        train_dirs: List of training data directories
        output_dir: Directory to save model
        tensorboard_log: Directory for TensorBoard logs
        total_timesteps: Total training timesteps
        n_envs: Number of parallel environments
        learning_rate: Learning rate for optimizer
        batch_size: Batch size for training
        net_arch: Network architecture
        gamma: Discount factor
        n_steps: Number of steps per update
        n_epochs: Number of epochs per update
        
    Returns:
        Trained PPO model
    """
    # Load training data
    envs = []
    for data_dir in train_dirs:
        try:
            df_windows = pl.read_parquet(f"{data_dir}/processed_{Path(data_dir).name}.parquet")
            df_final = pl.read_parquet(f"{data_dir}/processed_{Path(data_dir).name}.parquet")
            
            logging.info(f"Loaded DataFrame from {data_dir}")
            logging.info(f"Columns: {df_windows.columns}")
            logging.info(f"Shape: {df_windows.shape}")
            
            env = make_env(df_windows, df_final, len(envs))
            envs.append(env)
        except Exception as e:
            logging.error(f"Error loading data from {data_dir}: {e}")
            continue
    
    if not envs:
        raise ValueError("No environments could be created. Check your data directories.")
    
    # Create vectorized environment
    if n_envs > 1:
        vec_env = SubprocVecEnv(envs)
    else:
        vec_env = DummyVecEnv(envs)
    
    # Create evaluation environment
    eval_env = DummyVecEnv([make_env(df_windows, df_final, 0)])
    
    # Create evaluation callback
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=output_dir,
        log_path=output_dir,
        eval_freq=10000,
        deterministic=True,
        render=False
    )
    
    # Create model with custom hyperparameters
    model = PPO(
        'MlpPolicy',
        vec_env,
        learning_rate=learning_rate,
        n_steps=n_steps,
        batch_size=batch_size,
        n_epochs=n_epochs,
        gamma=gamma,
        gae_lambda=0.95,
        clip_range=0.2,
        clip_range_vf=None,
        normalize_advantage=True,
        ent_coef=0.01,
        vf_coef=0.5,
        max_grad_norm=0.5,
        use_sde=False,
        sde_sample_freq=-1,
        target_kl=None,
        tensorboard_log=tensorboard_log,
        policy_kwargs=dict(
            net_arch=[dict(pi=net_arch, vf=net_arch)]
        ),
        verbose=1
    )
    
    # Train model
    logging.info("Starting training with custom hyperparameters...")
    model.learn(
        total_timesteps=total_timesteps,
        callback=eval_callback,
        progress_bar=True
    )
    
    # Save final model
    model.save(f"{output_dir}/ppo_ohio_final")
    logging.info(f"Model saved to {output_dir}/ppo_ohio_final")
    
    return model

def test_user_inputs(
    model_path: str,
    test_cases: List[Dict[str, Union[List[float], float, datetime]]]
) -> Dict[str, Dict[str, float]]:
    """
    Test model predictions with various user input scenarios.
    
    Args:
        model_path: Path to trained model
        test_cases: List of test cases, each containing cgm_values, carb_input, iob, and timestamp
        
    Returns:
        Dictionary of predictions and metrics for each test case
    """
    results = {}
    
    for i, case in enumerate(test_cases):
        # Prepare observation
        obs = prepare_observation(
            case['cgm_values'],
            case['carb_input'],
            case['iob'],
            case['timestamp'],
            case.get('meal_carbs', 0.0),
            case.get('meal_time_diff', 0.0),
            case.get('has_meal', 0.0),
            case.get('meals_in_window', 0)
        )
        
        # Get prediction
        predicted_dose = predict_insulin_dose(model_path, obs)
        
        # Store results
        results[f'test_case_{i+1}'] = {
            'predicted_dose': predicted_dose,
            'cgm_mean': np.mean(case['cgm_values']),
            'carb_input': case['carb_input'],
            'iob': case['iob'],
            'meal_carbs': case.get('meal_carbs', 0.0),
            'meal_time_diff': case.get('meal_time_diff', 0.0),
            'has_meal': case.get('has_meal', 0.0),
            'meals_in_window': case.get('meals_in_window', 0)
        }
    
    return results

def train(
    self, 
    env: Any, 
    epochs: int = 100, 
    steps_per_epoch: int = 4000, 
    batch_size: int = 64, 
    update_iters: int = 10, 
    gae_lambda: float = 0.95,
    log_interval: int = 10
) -> Dict[str, List[float]]:
    """
    Entrena el agente con PPO en un entorno dado.
    
    Parámetros:
    -----------
    env : Any
        Entorno de OpenAI Gym o compatible
    epochs : int, opcional
        Número de épocas de entrenamiento (default: 100)
    steps_per_epoch : int, opcional
        Pasos por época (default: 4000)
    batch_size : int, opcional
        Tamaño de lote para actualización (default: 64)
    update_iters : int, opcional
        Número de iteraciones de actualización por lote (default: 10)
    gae_lambda : float, opcional
        Factor lambda para GAE (default: 0.95)
    log_interval : int, opcional
        Intervalo para mostrar información (default: 10)
        
    Retorna:
    --------
    Dict[str, List[float]]
        Historial de entrenamiento
    """
    history = {
        'reward': [],       # Recompensa por episodio
        'avg_reward': [],   # Recompensa promedio por época
        'policy_loss': [],  # Pérdida de política por época
        'value_loss': [],   # Pérdida de valor por época
        'entropy': [],      # Entropía por época
        'total_loss': [],   # Pérdida total por época
        'optional_features_used': []  # Registro de uso de características opcionales
    }
    
    for epoch in range(epochs):
        # 1. Recolectar trayectorias
        trajectory_data, episode_history = self._collect_trajectories(env, steps_per_epoch)
        states, actions, rewards, values, dones, next_values, log_probs = trajectory_data
        
        # Registrar recompensas de episodios
        history['reward'].extend(episode_history['reward'])
        history['avg_reward'].append(np.mean(episode_history['reward']))
        
        # Registrar uso de características opcionales
        if hasattr(env, 'has_optional_features'):
            history['optional_features_used'].append(env.has_optional_features)
        
        # 2. Calcular ventajas y retornos usando GAE
        advantages, returns = self.compute_gae(
            rewards, values, next_values, dones, self.gamma, gae_lambda)
        
        # 3. Actualizar política
        metrics = self._update_policy(
            states, actions, log_probs, returns, advantages, values,
            batch_size, update_iters)
        
        # Registrar métricas
        history['policy_loss'].append(metrics['policy_loss'])
        history['value_loss'].append(metrics['value_loss']) 
        history['entropy'].append(metrics['entropy'])
        history['total_loss'].append(metrics['total_loss'])
        
        # Mostrar progreso
        if (epoch + 1) % log_interval == 0 or epoch == 0:
            print(f"Época {epoch+1}/{epochs} - Recompensa Promedio: {history['avg_reward'][-1]:.2f}, "
                  f"Pérdida Política: {history['policy_loss'][-1]:.4f}, "
                  f"Pérdida Valor: {history['value_loss'][-1]:.4f}")
            if hasattr(env, 'has_optional_features'):
                print(f"Características opcionales: {'Usadas' if env.has_optional_features else 'No usadas'}")
    
    return history

def evaluate(
    model: PPO,
    test_dirs: List[str]
) -> Dict[str, float]:
    """
    Evaluate trained model on test data.
    
    Args:
        model: Trained PPO model
        test_dirs: List of test data directories
        
    Returns:
        Dictionary of per-subject MAE
    """
    results = {}
    
    for data_dir in test_dirs:
        # Load test data
        df_windows = pl.read_parquet(f"{data_dir}/processed_{Path(data_dir).name}.parquet")
        df_final = pl.read_parquet(f"{data_dir}/processed_{Path(data_dir).name}.parquet")
        
        # Create environment
        env = OhioT1DMEnv(df_windows, df_final)
        
        # Run evaluation
        obs, _ = env.reset()  # Unpack the tuple from reset()
        done = False
        predictions = []
        true_values = []
        
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, _, terminated, truncated, info = env.step(action)  # Unpack all return values
            done = terminated or truncated
            predictions.append(info['predicted_bolus'])
            true_values.append(info['true_bolus'])
        
        # Calculate MAE
        mae = np.mean(np.abs(np.array(predictions) - np.array(true_values)))
        results[Path(data_dir).name] = mae
        
    return results

def plot_training(tensorboard_log: str, output_dir: str):
    """
    Plot training curves from TensorBoard logs.
    
    Args:
        tensorboard_log: Directory containing TensorBoard logs
        output_dir: Directory to save plots
    """
    try:
        # Find the most recent event file directory (Stable-Baselines3 may create subdirs)
        event_dirs = [tensorboard_log] + [os.path.join(tensorboard_log, d) for d in os.listdir(tensorboard_log) if os.path.isdir(os.path.join(tensorboard_log, d))]
        event_file = None
        for d in event_dirs:
            event_files = glob.glob(os.path.join(d, 'events.out.tfevents.*'))
            if event_files:
                event_file = event_files[0]
                logging.info(f"Using TensorBoard event file: {event_file}")
                break
        if not event_file:
            logging.warning("No TensorBoard event file found for plotting.")
            return
        
        # Load TensorBoard data
        ea = event_accumulator.EventAccumulator(
            os.path.dirname(event_file),
            size_guidance={event_accumulator.SCALARS: 0}
        )
        ea.Reload()
        
        # Get available tags
        available_tags = ea.Tags()['scalars']
        logging.info(f"Available TensorBoard metrics: {available_tags}")
        
        if not available_tags:
            logging.warning("No training metrics found in TensorBoard logs")
            return
        
        # Create plots
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
        
        # Plot reward if available
        if 'rollout/ep_rew_mean' in available_tags:
            reward_data = ea.Scalars('rollout/ep_rew_mean')
            ax1.plot([x.step for x in reward_data], [x.value for x in reward_data])
            ax1.set_title('Training Reward')
            ax1.set_xlabel('Steps')
            ax1.set_ylabel('Mean Episode Reward')
            ax1.grid(True, linestyle='--', alpha=0.3)
        else:
            ax1.text(0.5, 0.5, 'No reward data available', 
                    horizontalalignment='center', verticalalignment='center')
            ax1.set_title('Training Reward (No Data)')
        
        # Plot losses if available (using correct tags)
        has_policy = 'train/policy_gradient_loss' in available_tags
        has_value = 'train/value_loss' in available_tags
        has_total = 'train/loss' in available_tags
        if has_policy or has_value or has_total:
            if has_policy:
                policy_loss = ea.Scalars('train/policy_gradient_loss')
                ax2.plot([x.step for x in policy_loss], [x.value for x in policy_loss], label='Policy Gradient Loss')
            if has_value:
                value_loss = ea.Scalars('train/value_loss')
                ax2.plot([x.step for x in value_loss], [x.value for x in value_loss], label='Value Loss')
            if has_total:
                total_loss = ea.Scalars('train/loss')
                ax2.plot([x.step for x in total_loss], [x.value for x in total_loss], label='Total Loss')
            ax2.set_title('Training Losses')
            ax2.set_xlabel('Steps')
            ax2.set_ylabel('Loss')
            ax2.legend()
            ax2.grid(True, linestyle='--', alpha=0.3)
        else:
            ax2.text(0.5, 0.5, 'No loss data available', 
                    horizontalalignment='center', verticalalignment='center')
            ax2.set_title('Training Losses (No Data)')
        
        plt.tight_layout()
        plt.savefig(f"{output_dir}/training_curves.png")
        plt.close()
        
    except Exception as e:
        logging.error(f"Error plotting training curves: {e}")

def plot_test_results(
    results: Dict[str, float],
    output_dir: str
):
    """
    Plot test results including MAE per subject.
    
    Args:
        results: Dictionary of per-subject MAE
        output_dir: Directory to save plots
    """
    try:
        # Convert results to numpy arrays with consistent types
        subjects = np.array(list(results.keys()))
        maes = np.array(list(results.values()), dtype=np.float64)
        
        plt.figure(figsize=(10, 6))
        bars = plt.bar(subjects, maes)
        
        # Add value labels on top of bars
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.4f}',
                    ha='center', va='bottom')
        
        plt.title('MAE per Subject')
        plt.xlabel('Subject')
        plt.ylabel('Mean Absolute Error')
        plt.xticks(rotation=45)
        plt.grid(True, linestyle='--', alpha=0.3)
        plt.tight_layout()
        
        # Add note if only one subject
        if len(subjects) == 1:
            plt.text(0.5, 0.5, 'Only one subject in test set',
                     transform=plt.gca().transAxes,
                     fontsize=12, color='gray', ha='center', va='center', alpha=0.5)
        
        # Save plot
        plt.savefig(f"{output_dir}/mae_per_subject.png")
        plt.close()
        
    except Exception as e:
        logging.error(f"Error plotting test results: {e}")
        # Print more detailed error information
        logging.error(f"Results dictionary: {results}")
        logging.error(f"Subjects: {subjects if 'subjects' in locals() else 'Not created'}")
        logging.error(f"MAEs: {maes if 'maes' in locals() else 'Not created'}")

def plot_overall_mae(mae: float, output_dir: str):
    """Plot a single bar for overall MAE."""
    plt.figure(figsize=(4, 6))
    plt.bar(['Overall MAE'], [mae], color='skyblue')
    plt.ylabel('Mean Absolute Error')
    plt.title('Overall MAE')
    plt.ylim(0, max(2, mae * 1.2))
    plt.tight_layout()
    plt.savefig(f"{output_dir}/overall_mae.png")
    plt.close()

def main():
    """Main function to train and evaluate PPO agent."""
    # Hardcoded parameters
    train_dirs = ['new_ohio/processed_data/train']
    test_dirs = ['new_ohio/processed_data/test']
    output_dir = 'new_ohio/models/output'
    tensorboard_log = 'new_ohio/models/runs/ppo_ohio'
    total_timesteps = 300_000
    n_envs = 4
    
    # Create output directories
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    Path(tensorboard_log).mkdir(parents=True, exist_ok=True)
    
    # Train model with optimized hyperparameters
    logging.info("Starting training with optimized hyperparameters...")
    model = train_with_hyperparameters(
        train_dirs=train_dirs,
        output_dir=output_dir,
        tensorboard_log=tensorboard_log,
        total_timesteps=total_timesteps,
        n_envs=n_envs,
        learning_rate=5e-5,  # Lower learning rate
        batch_size=128,  # Increased batch size
        net_arch=[256, 256, 128],  # Deeper and wider network
        gamma=0.99,
        n_steps=2048,
        n_epochs=10
    )
    
    # Evaluate model if test data exists
    logging.info("Checking test data...")
    test_data_exists = False
    for test_dir in test_dirs:
        test_file = f"{test_dir}/processed_{Path(test_dir).name}.parquet"
        if Path(test_file).exists():
            test_data_exists = True
            break
    
    if test_data_exists:
        logging.info("Evaluating model...")
        results = predict_from_preprocessed(
            model_path=f"{output_dir}/ppo_ohio_final",
            preprocessed_data_path=test_file
        )
        metrics = evaluate_metrics(results['predictions'], results['true_values'])
        
        # Print evaluation results
        logging.info("\nEvaluation Results:")
        logging.info("=" * 50)
        logging.info(f"MAE: {metrics['mae']:.4f}")
        logging.info(f"MSE: {metrics['mse']:.4f}")
        logging.info(f"RMSE: {metrics['rmse']:.4f}")
        logging.info(f"Overdose Rate: {metrics['overdose_rate']:.2f}%")
        logging.info(f"Underdose Rate: {metrics['underdose_rate']:.2f}%")
        logging.info(f"Safe Rate: {metrics['safe_rate']:.2f}%")
        logging.info("=" * 50)
        
        # Test user input scenarios
        logging.info("\nTesting user input scenarios...")
        test_cases = [
            # Caso 1: Comida normal con glucosa estable
            {
                'cgm_values': [120 + i*5 for i in range(24)],  # Glucosa estable con ligera tendencia al alza
                'carb_input': 50.0,
                'iob': 2.5,
                'timestamp': datetime.now(),
                'meal_carbs': 45.0,
                'meal_time_diff': 0.5,  # Comida en 30 minutos
                'has_meal': 1.0,
                'meals_in_window': 1
            },
            # Caso 2: Comida grande con glucosa alta
            {
                'cgm_values': [300] * 24,  # Glucosa alta
                'carb_input': 100.0,
                'iob': 0.0,
                'timestamp': datetime.now(),
                'meal_carbs': 95.0,
                'meal_time_diff': 0.25,  # Comida en 15 minutos
                'has_meal': 1.0,
                'meals_in_window': 1
            },
            # Caso 3: Glucosa baja sin comida
            {
                'cgm_values': [70] * 24,  # Glucosa baja
                'carb_input': 0.0,
                'iob': 5.0,
                'timestamp': datetime.now(),
                'meal_carbs': 0.0,
                'meal_time_diff': 0.0,
                'has_meal': 0.0,
                'meals_in_window': 0
            },
            # Caso 4: Comida pequeña con glucosa normal
            {
                'cgm_values': [140] * 24,  # Glucosa normal
                'carb_input': 20.0,
                'iob': 1.0,
                'timestamp': datetime.now(),
                'meal_carbs': 15.0,
                'meal_time_diff': 0.75,  # Comida en 45 minutos
                'has_meal': 1.0,
                'meals_in_window': 1
            },
            # Caso 5: Múltiples comidas en la ventana
            {
                'cgm_values': [160 + i*3 for i in range(24)],  # Glucosa subiendo
                'carb_input': 80.0,
                'iob': 3.0,
                'timestamp': datetime.now(),
                'meal_carbs': 75.0,
                'meal_time_diff': 0.33,  # Comida en 20 minutos
                'has_meal': 1.0,
                'meals_in_window': 2  # Dos comidas en la ventana
            },
            # Caso 6: Comida con glucosa descendente
            {
                'cgm_values': [200 - i*5 for i in range(24)],  # Glucosa bajando
                'carb_input': 60.0,
                'iob': 4.0,
                'timestamp': datetime.now(),
                'meal_carbs': 55.0,
                'meal_time_diff': 0.17,  # Comida en 10 minutos
                'has_meal': 1.0,
                'meals_in_window': 1
            },
            # Caso 7: Comida con glucosa muy alta
            {
                'cgm_values': [400] * 24,  # Glucosa muy alta
                'carb_input': 40.0,
                'iob': 0.0,
                'timestamp': datetime.now(),
                'meal_carbs': 35.0,
                'meal_time_diff': 0.5,  # Comida en 30 minutos
                'has_meal': 1.0,
                'meals_in_window': 1
            },
            # Caso 8: Sin comida con glucosa normal
            {
                'cgm_values': [130] * 24,  # Glucosa normal
                'carb_input': 0.0,
                'iob': 2.0,
                'timestamp': datetime.now(),
                'meal_carbs': 0.0,
                'meal_time_diff': 0.0,
                'has_meal': 0.0,
                'meals_in_window': 0
            }
        ]
        
        user_input_results = test_user_inputs(
            model_path=f"{output_dir}/ppo_ohio_final",
            test_cases=test_cases
        )
        
        # Print user input test results
        logging.info("\nUser Input Test Results:")
        logging.info("=" * 50)
        for case_name, case_results in user_input_results.items():
            logging.info(f"\n{case_name}:")
            logging.info(f"Predicted Dose: {case_results['predicted_dose']:.2f} units")
            logging.info(f"Mean CGM: {case_results['cgm_mean']:.2f} mg/dL")
            logging.info(f"Carb Input: {case_results['carb_input']:.2f} g")
            logging.info(f"IOB: {case_results['iob']:.2f} units")
            logging.info(f"Meal Carbs: {case_results['meal_carbs']:.2f} g")
            logging.info(f"Meal Time Diff: {case_results['meal_time_diff']:.2f} hours")
            logging.info(f"Has Meal: {case_results['has_meal']}")
            logging.info(f"Meals in Window: {case_results['meals_in_window']}")
        logging.info("=" * 50)
        
        # Plot results
        logging.info("Generating plots...")
        plot_training(tensorboard_log, output_dir)
        # Only plot per-subject MAE if results is a subject-to-mae dict
        if isinstance(results, dict) and all(isinstance(v, (float, np.floating)) for v in results.values()):
            plot_test_results(results, output_dir)
        elif 'mae' in results:
            plot_overall_mae(results['mae'], output_dir)
    else:
        logging.warning("No test data found. Skipping evaluation and plotting.")
        logging.info("To evaluate the model, please ensure test data exists at:")
        for test_dir in test_dirs:
            logging.info(f"  - {test_dir}/processed_{Path(test_dir).name}.parquet")
    
    logging.info("Done!")

if __name__ == "__main__":
    main()
