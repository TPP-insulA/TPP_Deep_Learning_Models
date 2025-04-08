# %% CELL: Required Imports
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns
import os
from joblib import Parallel, delayed
from datetime import timedelta
import time
from tqdm import tqdm

# DRL-specific imports
import gym
import gymnasium as gym  
from gymnasium import spaces  
from stable_baselines3 import PPO, SAC, TD3  
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.callbacks import BaseCallback

PROJECT_DIR = os.path.abspath(os.path.join(os.getcwd(), ".."))

# Global configuration
CONFIG = {
    "batch_size": 128,
    "window_hours": 2,
    "cap_normal": 30,
    "cap_bg": 300,
    "cap_iob": 5,
    "cap_carb": 150,
    "data_path": os.path.join(PROJECT_DIR, "data", "subjects")
}

# %% CELL: Data Processing Functions
def get_cgm_window(bolus_time, cgm_df, window_hours=CONFIG["window_hours"]):
    """
    Obtiene una ventana de datos CGM alrededor del tiempo del bolo de insulina.
    
    Args:
        bolus_time: Tiempo del bolo de insulina
        cgm_df: DataFrame con datos CGM
        window_hours: Tamaño de la ventana en horas
    
    Returns:
        Array numpy con los últimos 24 valores CGM o None si no hay suficientes datos
    """
    window_start = bolus_time - timedelta(hours=window_hours)
    window = cgm_df[(cgm_df['date'] >= window_start) & (cgm_df['date'] <= bolus_time)]
    window = window.sort_values('date').tail(24)
    return window['mg/dl'].values if len(window) >= 24 else None

def calculate_iob(bolus_time, basal_df, half_life_hours=4):
    """
    Calcula la insulina activa (IOB) en un momento dado.
    
    Args:
        bolus_time: Tiempo para calcular IOB
        basal_df: DataFrame con datos de insulina basal
        half_life_hours: Vida media de la insulina en horas
    
    Returns:
        Float con cantidad de insulina activa
    """
    if basal_df is None or basal_df.empty:
        return 0.0
        
    iob = 0
    for _, row in basal_df.iterrows():
        start_time = row['date']
        duration_hours = row['duration'] / (1000 * 3600)
        end_time = start_time + timedelta(hours=duration_hours)
        rate = row['rate'] if pd.notna(row['rate']) else 0.9
        rate = min(rate, 2.0)
        
        if start_time <= bolus_time <= end_time:
            time_since_start = (bolus_time - start_time).total_seconds() / 3600
            remaining = rate * (1 - (time_since_start / half_life_hours))
            iob += max(0, remaining)
            
    return min(iob, 5.0)

def process_subject(subject_path, idx):
    """
    Procesa los datos de un sujeto del estudio.
    
    Args:
        subject_path: Ruta al archivo Excel con datos del sujeto
        idx: Índice/ID del sujeto
    
    Returns:
        Lista de diccionarios con características procesadas
    """
    start_time = time.time()
    
    try:
        excel_file = pd.ExcelFile(subject_path)
        cgm_df = pd.read_excel(excel_file, sheet_name="CGM")
        bolus_df = pd.read_excel(excel_file, sheet_name="Bolus")
        try:
            basal_df = pd.read_excel(excel_file, sheet_name="Basal")
        except ValueError:
            basal_df = None
    except Exception as e:
        print(f"Error al cargar {os.path.basename(subject_path)}: {e}")
        return []

    cgm_df['date'] = pd.to_datetime(cgm_df['date'])
    cgm_df = cgm_df.sort_values('date')
    bolus_df['date'] = pd.to_datetime(bolus_df['date'])
    if basal_df is not None:
        basal_df['date'] = pd.to_datetime(basal_df['date'])

    non_zero_carbs = bolus_df[bolus_df['carbInput'] > 0]['carbInput']
    carb_median = non_zero_carbs.median() if not non_zero_carbs.empty else 10.0

    iob_values = [calculate_iob(row['date'], basal_df) for _, row in bolus_df.iterrows()]
    non_zero_iob = [iob for iob in iob_values if iob > 0]
    iob_median = np.median(non_zero_iob) if non_zero_iob else 0.5

    processed_data = []
    for _, row in tqdm(bolus_df.iterrows(), total=len(bolus_df), desc=f"Procesando {os.path.basename(subject_path)}", leave=False):
        bolus_time = row['date']
        cgm_window = get_cgm_window(bolus_time, cgm_df)
        
        if cgm_window is not None:
            iob = calculate_iob(bolus_time, basal_df)
            iob = iob_median if iob == 0 else iob
            hour_of_day = bolus_time.hour / 23.0
            bg_input = row['bgInput'] if pd.notna(row['bgInput']) else cgm_window[-1]
            
            normal = row['normal'] if pd.notna(row['normal']) else 0.0
            normal = np.clip(normal, 0, CONFIG["cap_normal"])
            
            bg_input = max(bg_input, 50.0)
            isf_custom = 50.0 if normal <= 0 else (bg_input - 100) / normal
            isf_custom = np.clip(isf_custom, 10, 100)
            
            bg_input = np.clip(bg_input, 0, CONFIG["cap_bg"])
            iob = np.clip(iob, 0, CONFIG["cap_iob"])
            carb_input = row['carbInput'] if pd.notna(row['carbInput']) else 0.0
            carb_input = carb_median if carb_input == 0 else carb_input
            carb_input = np.clip(carb_input, 0, CONFIG["cap_carb"])
            
            features = {
                'subject_id': idx,
                'cgm_window': cgm_window,
                'carbInput': carb_input,
                'bgInput': bg_input,
                'insulinCarbRatio': np.clip(row['insulinCarbRatio'] if pd.notna(row['insulinCarbRatio']) else 10.0, 5, 20),
                'insulinSensitivityFactor': isf_custom,
                'insulinOnBoard': iob,
                'hour_of_day': hour_of_day,
                'normal': normal
            }
            processed_data.append(features)

    elapsed_time = time.time() - start_time
    print(f"Procesado {os.path.basename(subject_path)} (Sujeto {idx+1}) en {elapsed_time:.2f} segundos")
    return processed_data

def preprocess_data(subject_folder):
    
    
    """
    Preprocesa los datos de todos los sujetos para el entrenamiento del modelo.
    
    Args:
        subject_folder: Ruta a la carpeta que contiene los archivos Excel de los sujetos
        
    Returns:
        X_cgm: Array de datos CGM normalizados
        X_other: Array de otras características normalizadas
        X_subject: Array de IDs de sujetos
        y: Array de dosis objetivo normalizadas
        df_final: DataFrame con todos los datos procesados
        scaler_cgm: Scaler usado para normalizar datos CGM
        scaler_other: Scaler usado para normalizar otras características
        scaler_y: Scaler usado para normalizar objetivos
    """
    start_time = time.time()
    
    subject_files = [f for f in os.listdir(subject_folder) if f.startswith("Subject") and f.endswith(".xlsx")]
    print(f"\nFound Subject files ({len(subject_files)}):")
    for f in subject_files:
        print(f)

    all_processed_data = Parallel(n_jobs=-1)(delayed(process_subject)(os.path.join(subject_folder, f), idx) 
                                            for idx, f in enumerate(subject_files))
    all_processed_data = [item for sublist in all_processed_data for item in sublist]

    df_processed = pd.DataFrame(all_processed_data)
    print("Muestra de datos procesados combinados:")
    print(df_processed.head())
    print(f"Total de muestras: {len(df_processed)}")

    df_processed['normal'] = np.log1p(df_processed['normal'])
    df_processed['carbInput'] = np.log1p(df_processed['carbInput'])
    df_processed['insulinOnBoard'] = np.log1p(df_processed['insulinOnBoard'])
    df_processed['bgInput'] = np.log1p(df_processed['bgInput'])
    df_processed['cgm_window'] = df_processed['cgm_window'].apply(lambda x: np.log1p(x))

    cgm_columns = [f'cgm_{i}' for i in range(24)]
    df_cgm = pd.DataFrame(df_processed['cgm_window'].tolist(), columns=cgm_columns, index=df_processed.index)
    df_final = pd.concat([df_cgm, df_processed.drop(columns=['cgm_window'])], axis=1)
    
    df_final = df_final.dropna()
    print("Verificación de NaN en df_final:")
    print(df_final.isna().sum())

    elapsed_time = time.time() - start_time
    print(f"Preprocesamiento completo en {elapsed_time:.2f} segundos")
    return df_final


def split_data(df_final):
    """
    Divide los datos en conjuntos de entrenamiento, validación y prueba, asegurando distribuciones similares.
    Forces Subject 49 into the test set for evaluation and balances means and stds across sets.
    
    Args:
        df_final: DataFrame con todos los datos
        
    Returns:
        Tupla con datos divididos para entrenamiento, validación y prueba
    """
    start_time = time.time()
    
    # Compute the mean and std insulin dose for each subject
    subject_stats = df_final.groupby('subject_id')['normal'].agg(['mean', 'std']).reset_index()
    subject_stats.columns = ['subject_id', 'mean_dose', 'std_dose']
    subject_ids = df_final['subject_id'].unique()

    # Sort subjects by their mean insulin dose
    sorted_subjects = subject_stats.sort_values('mean_dose')['subject_id'].values
    n_subjects = len(sorted_subjects)

    # Aim for 80%-10%-10% split
    train_size = int(0.8 * n_subjects)  # 80% for training
    val_size = int(0.1 * n_subjects)    # 10% for validation
    test_size = n_subjects - train_size - val_size  # 10% for test

    # Force Subject 49 into the test set
    test_subjects = [49] if 49 in sorted_subjects else []
    remaining_subjects = [s for s in sorted_subjects if s != 49]
    train_subjects = []
    val_subjects = []

    # Initialize lists and track stats
    train_subjects = []
    val_subjects = []
    test_subjects = [49] if 49 in sorted_subjects else []

    # Assign subjects to minimize differences in means and stds
    remaining_subjects_list = list(remaining_subjects)
    np.random.shuffle(remaining_subjects_list)  # Shuffle to avoid bias in assignment order

    for i, subject in enumerate(remaining_subjects_list):
        # Compute current means and stds of each set
        train_mean = df_final[df_final['subject_id'].isin(train_subjects)]['normal'].mean() if train_subjects else 0
        val_mean = df_final[df_final['subject_id'].isin(val_subjects)]['normal'].mean() if val_subjects else 0
        test_mean = df_final[df_final['subject_id'].isin(test_subjects)]['normal'].mean() if test_subjects else 0
        train_std = df_final[df_final['subject_id'].isin(train_subjects)]['normal'].std() if train_subjects else 0
        val_std = df_final[df_final['subject_id'].isin(val_subjects)]['normal'].std() if val_subjects else 0
        test_std = df_final[df_final['subject_id'].isin(test_subjects)]['normal'].std() if test_subjects else 0

        # Compute the stats if we add this subject to each set
        train_temp = train_subjects + [subject]
        val_temp = val_subjects + [subject]
        test_temp = test_subjects + [subject]

        train_mean_new = df_final[df_final['subject_id'].isin(train_temp)]['normal'].mean()
        val_mean_new = df_final[df_final['subject_id'].isin(val_temp)]['normal'].mean()
        test_mean_new = df_final[df_final['subject_id'].isin(test_temp)]['normal'].mean()
        train_std_new = df_final[df_final['subject_id'].isin(train_temp)]['normal'].std()
        val_std_new = df_final[df_final['subject_id'].isin(val_temp)]['normal'].std()
        test_std_new = df_final[df_final['subject_id'].isin(test_temp)]['normal'].std()

        # Compute the mean and std differences if we add to each set
        means_if_train = [train_mean_new, val_mean, test_mean]
        means_if_val = [train_mean, val_mean_new, test_mean]
        means_if_test = [train_mean, val_mean, test_mean_new]
        stds_if_train = [train_std_new, val_std, test_std]
        stds_if_val = [train_std, val_std_new, test_std]
        stds_if_test = [train_std, val_std, test_std_new]

        # Compute the range of means and stds (max - min) for each option
        range_means_if_train = max(means_if_train) - min(means_if_train) if all(m != 0 for m in means_if_train) else float('inf')
        range_means_if_val = max(means_if_val) - min(means_if_val) if all(m != 0 for m in means_if_val) else float('inf')
        range_means_if_test = max(means_if_test) - min(means_if_test) if all(m != 0 for m in means_if_test) else float('inf')
        range_stds_if_train = max(stds_if_train) - min(stds_if_train) if all(s != 0 for s in stds_if_train) else float('inf')
        range_stds_if_val = max(stds_if_val) - min(stds_if_val) if all(s != 0 for s in stds_if_val) else float('inf')
        range_stds_if_test = max(stds_if_test) - min(stds_if_test) if all(s != 0 for s in stds_if_test) else float('inf')

        # Combine mean and std differences (weighted equally)
        score_if_train = range_means_if_train + range_stds_if_train
        score_if_val = range_means_if_val + range_stds_if_val
        score_if_test = range_means_if_test + range_stds_if_test

        # Respect the target split sizes
        if len(train_subjects) < train_size and score_if_train <= min(score_if_val, score_if_test):
            train_subjects.append(subject)
        elif len(val_subjects) < val_size and score_if_val <= min(score_if_train, score_if_test):
            val_subjects.append(subject)
        elif len(test_subjects) < test_size:
            test_subjects.append(subject)
        else:
            # If sizes are exceeded, default to train
            train_subjects.append(subject)

    # Create masks for splitting
    train_mask = df_final['subject_id'].isin(train_subjects)
    val_mask = df_final['subject_id'].isin(val_subjects)
    test_mask = df_final['subject_id'].isin(test_subjects)

    # Check distributions after splitting
    y_train_temp = df_final.loc[train_mask, 'normal']
    y_val_temp = df_final.loc[val_mask, 'normal']
    y_test_temp = df_final.loc[test_mask, 'normal']
    print("Post-split Train y: mean =", y_train_temp.mean(), "std =", y_train_temp.std())
    print("Post-split Val y: mean =", y_val_temp.mean(), "std =", y_val_temp.std())
    print("Post-split Test y: mean =", y_test_temp.mean(), "std =", y_test_temp.std())

    # Initialize scalers
    scaler_cgm = StandardScaler()
    scaler_other = StandardScaler()
    scaler_y = StandardScaler()
    cgm_columns = [f'cgm_{i}' for i in range(24)]
    other_features = ['carbInput', 'bgInput', 'insulinOnBoard', 'insulinCarbRatio', 
                      'insulinSensitivityFactor', 'hour_of_day']

    # Scale only on training data
    X_cgm_train = scaler_cgm.fit_transform(df_final.loc[train_mask, cgm_columns]).reshape(-1, 24, 1)
    X_cgm_val = scaler_cgm.transform(df_final.loc[val_mask, cgm_columns]).reshape(-1, 24, 1)
    X_cgm_test = scaler_cgm.transform(df_final.loc[test_mask, cgm_columns]).reshape(-1, 24, 1)
    
    X_other_train = scaler_other.fit_transform(df_final.loc[train_mask, other_features])
    X_other_val = scaler_other.transform(df_final.loc[val_mask, other_features])
    X_other_test = scaler_other.transform(df_final.loc[test_mask, other_features])
    
    y_train = scaler_y.fit_transform(df_final.loc[train_mask, 'normal'].values.reshape(-1, 1)).flatten()
    y_val = scaler_y.transform(df_final.loc[val_mask, 'normal'].values.reshape(-1, 1)).flatten()
    y_test = scaler_y.transform(df_final.loc[test_mask, 'normal'].values.reshape(-1, 1)).flatten()

    X_subject_train = df_final.loc[train_mask, 'subject_id'].values
    X_subject_val = df_final.loc[val_mask, 'subject_id'].values
    X_subject_test = df_final.loc[test_mask, 'subject_id'].values
    subject_test = X_subject_test

    print(f"Entrenamiento CGM: {X_cgm_train.shape}, Validación CGM: {X_cgm_val.shape}, Prueba CGM: {X_cgm_test.shape}")
    print(f"Entrenamiento Otros: {X_other_train.shape}, Validación Otros: {X_other_val.shape}, Prueba Otros: {X_other_test.shape}")
    print(f"Entrenamiento Subject: {X_subject_train.shape}, Validación Subject: {X_subject_val.shape}, Prueba Subject: {X_subject_test.shape}")
    print(f"Sujetos de prueba: {test_subjects}")

    elapsed_time = time.time() - start_time
    print(f"División de datos completa en {elapsed_time:.2f} segundos")
    return (X_cgm_train, X_cgm_val, X_cgm_test,
            X_other_train, X_other_val, X_other_test,
            X_subject_train, X_subject_val, X_subject_test,
            y_train, y_val, y_test, subject_test,
            scaler_cgm, scaler_other, scaler_y)

def rule_based_prediction(X_other, scaler_other, scaler_y, target_bg=100):
    """
    Genera predicciones basadas en reglas médicas estándar.
    
    Args:
        X_other: Características adicionales normalizadas
        scaler_other: Scaler para desnormalizar características
        scaler_y: Scaler para normalizar predicciones
        target_bg: Nivel objetivo de glucosa en sangre
        
    Returns:
        Array con predicciones de dosis
    """
    start_time = time.time()
    
    X_other_np = X_other
    inverse_transformed = scaler_other.inverse_transform(X_other_np)
    carb_input, bg_input, icr, isf = (inverse_transformed[:, 0],
                                     inverse_transformed[:, 1],
                                     inverse_transformed[:, 3],
                                     inverse_transformed[:, 4])
    
    icr = np.where(icr == 0, 1e-6, icr)
    isf = np.where(isf == 0, 1e-6, isf)
    
    carb_component = np.divide(carb_input, icr, out=np.zeros_like(carb_input), where=icr!=0)
    bg_component = np.divide(bg_input - target_bg, isf, out=np.zeros_like(bg_input), where=isf!=0)
    
    prediction = carb_component + bg_component
    prediction = np.clip(prediction, 0, CONFIG["cap_normal"])

    elapsed_time = time.time() - start_time
    print(f"Predicción basada en reglas completa en {elapsed_time:.2f} segundos")
    return prediction
# %% CELL: Visualization Functions
def compute_metrics(y_true, y_pred, scaler_y):
    y_true_denorm = scaler_y.inverse_transform(y_true.reshape(-1, 1)).flatten()
    y_pred_denorm = y_pred  # Already denormalized
    mae = mean_absolute_error(y_true_denorm, y_pred_denorm)
    rmse = np.sqrt(mean_squared_error(y_true_denorm, y_pred_denorm))
    r2 = r2_score(y_true_denorm, y_pred_denorm)
    return mae, rmse, r2

def plot_evaluation(y_test, y_pred_ppo, y_rule, subject_test, scaler_y):
    start_time = time.time()
    y_test_denorm = scaler_y.inverse_transform(y_test.reshape(-1, 1)).flatten()

    colors = {'PPO': 'green', 'Rules': 'orange'}
    offset = 1e-2

    # 1. Predictions vs Real (Density Scatter Plot for PPO)
    plt.figure(figsize=(8, 6))
    sns.kdeplot(x=y_test_denorm + offset, y=y_pred_ppo + offset, cmap="viridis", fill=True, levels=5, thresh=.05)
    plt.plot([offset, 15], [offset, 15], 'k--', label='Perfect Prediction')
    plt.xscale('log')
    plt.yscale('log')
    plt.xticks([0.01, 0.1, 1, 10, 15], ['0.01', '0.1', '1', '10', '15'])
    plt.yticks([0.01, 0.1, 1, 10, 15], ['0.01', '0.1', '1', '10', '15'])
    plt.xlabel('Real Dose (units)', fontsize=10)
    plt.ylabel('Predicted Dose (units)', fontsize=10)
    plt.title('PPO: Predictions vs Real (Density)', fontsize=12)
    plt.legend()
    plt.show()

    # 2. Residual Distribution (KDE Plots)
    plt.figure(figsize=(10, 6))
    residuals_ppo = y_test_denorm - y_pred_ppo
    residuals_rule = y_test_denorm - y_rule

    sns.kdeplot(residuals_ppo, label='PPO', color=colors['PPO'], fill=True, alpha=0.3)
    sns.kdeplot(residuals_rule, label='Rules', color=colors['Rules'], fill=True, alpha=0.3)
    plt.xlabel('Residual (units)', fontsize=10)
    plt.ylabel('Density', fontsize=10)
    plt.title('Residual Distribution (KDE)', fontsize=12)
    plt.legend(fontsize=8)
    plt.grid(True, alpha=0.3)
    plt.show()

    # 3. MAE by Subject
    plt.figure(figsize=(10, 6))
    test_subjects = np.unique(subject_test)
    mae_ppo, mae_rule = [], []
    for sid in test_subjects:
        mask = subject_test == sid
        if np.sum(mask) > 0:
            mae_ppo.append(mean_absolute_error(y_test_denorm[mask], y_pred_ppo[mask]))
            mae_rule.append(mean_absolute_error(y_test_denorm[mask], y_rule[mask]))

    bar_width = 0.35
    x = np.arange(len(test_subjects))
    plt.bar(x - bar_width/2, mae_ppo, width=bar_width, label='PPO', color=colors['PPO'], alpha=0.8)
    plt.bar(x + bar_width/2, mae_rule, width=bar_width, label='Rules', color=colors['Rules'], alpha=0.8)
    plt.xlabel('Subject', fontsize=10)
    plt.ylabel('MAE (units)', fontsize=10)
    plt.xticks(x, test_subjects, rotation=45, ha='right', fontsize=8)
    plt.ylim(0, 2.5)  # Adjusted to accommodate Rules MAE
    plt.title('MAE by Subject', fontsize=12)
    plt.legend(fontsize=8)
    plt.grid(True, axis='y', alpha=0.3)
    plt.show()

    elapsed_time = time.time() - start_time
    print(f"Visualización completa en {elapsed_time:.2f} segundos")

# %% CELL: DRL Environment Definition
class InsulinDoseEnv(gym.Env):
    def __init__(self, X_cgm, X_other, y, scaler_y):
        super(InsulinDoseEnv, self).__init__()
        self.X_cgm = X_cgm.astype(np.float32)  # Ensure input data is float32
        self.X_other = X_other.astype(np.float32)  # Ensure input data is float32
        self.y = y.astype(np.float32)  # Ensure target data is float32
        self.scaler_y = scaler_y  # Scaler for denormalizing doses
        self.current_step = 0
        self.n_samples = len(X_cgm)
        
        # Define the state space (CGM + other features)
        state_dim = X_cgm.shape[2] * X_cgm.shape[1] + X_other.shape[1]  # Flatten CGM (24*1) + other features (6)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(state_dim,), dtype=np.float32)
        
        # Define the action space (insulin dose, continuous)
        self.action_space = spaces.Box(low=-3, high=3, shape=(1,), dtype=np.float32)  # Adjust based on your scaler
        
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step = np.random.randint(0, self.n_samples)
        cgm_state = self.X_cgm[self.current_step].flatten()  # Shape: (24,)
        other_state = self.X_other[self.current_step]  # Shape: (6,)
        state = np.concatenate([cgm_state, other_state]).astype(np.float32)  # Ensure state is float32
        self.current_state = state  # Store the current state for use in step()
        return state, {}  # Return state and info dict
    
    def step(self, action):
        true_dose = self.scaler_y.inverse_transform(self.y[self.current_step].reshape(-1, 1)).flatten()[0]
        predicted_dose = self.scaler_y.inverse_transform(action.reshape(-1, 1)).flatten()[0]
        error = predicted_dose - true_dose
        weight = 1.0 + (true_dose / 3.0)  # Slightly more aggressive weighting
        reward = -min(abs(error), 2.0) * weight  # Apply weight to clipped absolute error
        reward = float(reward)
        done = True
        truncated = False
        info = {"true_dose": true_dose, "predicted_dose": predicted_dose}
        next_state = self.current_state
        return next_state, reward, done, truncated, info
    
    def render(self, mode='human'):
        pass

class RewardCallback(BaseCallback):
    def __init__(self, val_env, verbose=0):
        super().__init__(verbose)
        self.train_rewards = []
        self.val_rewards = []
        self.val_env = val_env

    def _on_step(self):
        self.train_rewards.append(self.locals['rewards'][0])
        # Evaluate on validation set every 1000 steps
        if self.num_timesteps % 1000 == 0:
            val_reward = self.evaluate_val()
            self.val_rewards.append(val_reward)
        return True

    def evaluate_val(self):
        total_reward = 0
        for _ in range(10):  # Evaluate on 10 random validation samples
            state, _ = self.val_env.reset()
            action, _ = self.model.predict(state, deterministic=True)
            _, reward, _, _, _ = self.val_env.step(action)
            total_reward += reward
        return total_reward / 10
    
# %% CELL: Main Execution - Preprocess Data
# df_final = preprocess_data(CONFIG["data_path"])
# #Save the df_final in a csv
# df_final.to_csv('df_final.csv', index=False)

# %%
#Load the csv
df_final = pd.read_csv('df_final.csv')

(X_cgm_train, X_cgm_val, X_cgm_test,
 X_other_train, X_other_val, X_other_test,
 X_subject_train, X_subject_val, X_subject_test,
 y_train, y_val, y_test, subject_test,
 scaler_cgm, scaler_other, scaler_y) = split_data(df_final)

# %% CELL: Main Execution - DRL (PPO) Training
# Create the training and validation environments
train_env_ppo = InsulinDoseEnv(X_cgm_train, X_other_train, y_train, scaler_y)
val_env_ppo = InsulinDoseEnv(X_cgm_val, X_other_val, y_val, scaler_y)
callback = RewardCallback(val_env=val_env_ppo)

# Verify that the environment is correctly implemented
check_env(train_env_ppo)

# Initialize the PPO model
model_ppo = PPO("MlpPolicy", 
                train_env_ppo, 
                verbose=1, 
                learning_rate=1e-4,  # Reduced learning rate
                n_steps=2048, 
                batch_size=64, 
                clip_range=0.15, 
                ent_coef=0.01)   # Reduce entropy coefficient
# Train the model
total_timesteps = 50000  # Adjust based on convergence
model_ppo.learn(total_timesteps=total_timesteps, callback=callback)

# Plot training and validation rewards
plt.plot(callback.train_rewards, label='Train Reward')
plt.plot(np.arange(len(callback.val_rewards)) * 1000, callback.val_rewards, label='Val Reward')
plt.xlabel('Timestep')
plt.ylabel('Reward')
plt.legend()
plt.title('PPO Training vs Validation Reward')
plt.show()

# Save the model (optional)
model_ppo.save("ppo_insulin_dose")

# Function to generate predictions with PPO
def predict_with_ppo(model, X_cgm, X_other):
    predictions = []
    env = InsulinDoseEnv(X_cgm, X_other, np.zeros(len(X_cgm)), scaler_y)  # Dummy y since we only need X for prediction
    
    for i in range(len(X_cgm)):
        cgm_state = X_cgm[i].flatten()  # Shape: (24,)
        other_state = X_other[i]  # Shape: (6,)
        state = np.concatenate([cgm_state, other_state])  # Shape: (30,)
        
        action, _ = model.predict(state, deterministic=True)  # Use deterministic mode for evaluation
        predicted_dose = scaler_y.inverse_transform(action.reshape(-1, 1)).flatten()[0]
        predictions.append(predicted_dose)
    
    return np.array(predictions)

# Generate predictions on train, validation, and test sets
y_pred_ppo_train = predict_with_ppo(model_ppo, X_cgm_train, X_other_train)
y_pred_ppo_val = predict_with_ppo(model_ppo, X_cgm_val, X_other_val)
y_pred_ppo = predict_with_ppo(model_ppo, X_cgm_test, X_other_test)


# Generate rule-based predictions for comparison
y_rule = rule_based_prediction(X_other_test, scaler_other, scaler_y)

# %% CELL: Main Execution - Print Metrics
# Metrics for PPO
mae_ppo = mean_absolute_error(scaler_y.inverse_transform(y_test.reshape(-1, 1)), y_pred_ppo)
rmse_ppo = np.sqrt(mean_squared_error(scaler_y.inverse_transform(y_test.reshape(-1, 1)), y_pred_ppo))
r2_ppo = r2_score(scaler_y.inverse_transform(y_test.reshape(-1, 1)), y_pred_ppo)
print(f"PPO Test - MAE: {mae_ppo:.2f}, RMSE: {rmse_ppo:.2f}, R²: {r2_ppo:.2f}")

# Metrics for the Rules-based model
mae_rule = mean_absolute_error(scaler_y.inverse_transform(y_test.reshape(-1, 1)), y_rule)
rmse_rule = np.sqrt(mean_squared_error(scaler_y.inverse_transform(y_test.reshape(-1, 1)), y_rule))
r2_rule = r2_score(scaler_y.inverse_transform(y_test.reshape(-1, 1)), y_rule)
print(f"Rules Test - MAE: {mae_rule:.2f}, RMSE: {rmse_rule:.2f}, R²: {r2_rule:.2f}")

# %% CELL: Main Execution - PPO Metrics Across Sets to Check Overfitting
# Metrics for train
mae_ppo_train = mean_absolute_error(scaler_y.inverse_transform(y_train.reshape(-1, 1)), y_pred_ppo_train)
rmse_ppo_train = np.sqrt(mean_squared_error(scaler_y.inverse_transform(y_train.reshape(-1, 1)), y_pred_ppo_train))
r2_ppo_train = r2_score(scaler_y.inverse_transform(y_train.reshape(-1, 1)), y_pred_ppo_train)
print(f"PPO Train - MAE: {mae_ppo_train:.2f}, RMSE: {rmse_ppo_train:.2f}, R²: {r2_ppo_train:.2f}")

# Metrics for validation
mae_ppo_val = mean_absolute_error(scaler_y.inverse_transform(y_val.reshape(-1, 1)), y_pred_ppo_val)
rmse_ppo_val = np.sqrt(mean_squared_error(scaler_y.inverse_transform(y_val.reshape(-1, 1)), y_pred_ppo_val))
r2_ppo_val = r2_score(scaler_y.inverse_transform(y_val.reshape(-1, 1)), y_pred_ppo_val)
print(f"PPO Val - MAE: {mae_ppo_val:.2f}, RMSE: {rmse_ppo_val:.2f}, R²: {r2_ppo_val:.2f}")

# Metrics for test (already computed)
print(f"PPO Test - MAE: {mae_ppo:.2f}, RMSE: {rmse_ppo:.2f}, R²: {r2_ppo:.2f}")

#%% CELL: Main Execution - Cross-Validation for PPO
from sklearn.model_selection import KFold

kf = KFold(n_splits=5, shuffle=True, random_state=42)
subject_ids = df_final['subject_id'].unique()
mae_scores = []
cgm_columns = [f'cgm_{i}' for i in range(24)]
other_features = ['carbInput', 'bgInput', 'insulinOnBoard', 'insulinCarbRatio', 
                  'insulinSensitivityFactor', 'hour_of_day']

for train_idx, test_idx in kf.split(subject_ids):
    train_subs = subject_ids[train_idx]
    test_subs = subject_ids[test_idx]
    train_mask = df_final['subject_id'].isin(train_subs)
    test_mask = df_final['subject_id'].isin(test_subs)
    
    X_cgm_train_cv = scaler_cgm.transform(df_final.loc[train_mask, cgm_columns]).reshape(-1, 24, 1)
    X_other_train_cv = scaler_other.transform(df_final.loc[train_mask, other_features])
    y_train_cv = scaler_y.transform(df_final.loc[train_mask, 'normal'].values.reshape(-1, 1)).flatten()
    X_cgm_test_cv = scaler_cgm.transform(df_final.loc[test_mask, cgm_columns]).reshape(-1, 24, 1)
    X_other_test_cv = scaler_other.transform(df_final.loc[test_mask, other_features])
    y_test_cv = scaler_y.transform(df_final.loc[test_mask, 'normal'].values.reshape(-1, 1)).flatten()
    
    env_cv = InsulinDoseEnv(X_cgm_train_cv, X_other_train_cv, y_train_cv, scaler_y)
    model_cv = PPO("MlpPolicy", env_cv, verbose=0, learning_rate=3e-4, n_steps=2048, batch_size=64)
    model_cv.learn(total_timesteps=50000)  # Reduced for speed
    y_pred_cv = predict_with_ppo(model_cv, X_cgm_test_cv, X_other_test_cv)
    mae_cv = mean_absolute_error(scaler_y.inverse_transform(y_test_cv.reshape(-1, 1)), y_pred_cv)
    mae_scores.append(mae_cv)

print(f"Cross-validated PPO MAE: {np.mean(mae_scores):.2f} ± {np.std(mae_scores):.2f}")

# %% CELL: Main Execution - Debug Test Set Distribution
y_train_denorm = scaler_y.inverse_transform(y_train.reshape(-1, 1)).flatten()
y_val_denorm = scaler_y.inverse_transform(y_val.reshape(-1, 1)).flatten()
y_test_denorm = scaler_y.inverse_transform(y_test.reshape(-1, 1)).flatten()

print("Train y: mean =", np.mean(y_train_denorm), "std =", np.std(y_train_denorm))
print("Val y: mean =", np.mean(y_val_denorm), "std =", np.std(y_val_denorm))
print("Test y: mean =", np.mean(y_test_denorm), "std =", np.std(y_test_denorm))

plt.figure(figsize=(10, 6))
sns.kdeplot(y_train_denorm, label='Train', color='blue')
sns.kdeplot(y_val_denorm, label='Val', color='orange')
sns.kdeplot(y_test_denorm, label='Test', color='green')
plt.xlabel('Insulin Dose (units)')
plt.ylabel('Density')
plt.legend()
plt.title('Distribution of Target Insulin Doses')
plt.show()

# Plot predictions vs true values for test set
plt.figure(figsize=(8, 6))
plt.scatter(y_test_denorm, y_pred_ppo, alpha=0.5, label='PPO Predictions')
plt.plot([0, 4], [0, 4], 'k--', label='Perfect Prediction')
plt.xlabel('True Dose (units)')
plt.ylabel('Predicted Dose (units)')
plt.legend()
plt.title('PPO Predictions vs True Values (Test Set)')
plt.show()

# %% CELL: Main Execution - Visualization
# Visualize the results

# %% CELL: Main Execution - Metrics per Subject
# Analyze performance per subject
print("\nRendimiento por sujeto (Test Set):")
for subject_id in np.unique(subject_test):
    mask = subject_test == subject_id
    if np.sum(mask) > 0:
        y_test_sub = scaler_y.inverse_transform(y_test[mask].reshape(-1, 1)).flatten()
        print(f"Sujeto {subject_id}: ", end="")
        mae_ppo_sub = mean_absolute_error(y_test_sub, y_pred_ppo[mask])
        print(f"PPO MAE={mae_ppo_sub:.2f}, ", end="")
        mae_rule_sub = mean_absolute_error(y_test_sub, y_rule[mask])
        print(f"Rules MAE={mae_rule_sub:.2f}")

# Check if subject 49 exists in test set
if 49 in np.unique(subject_test):
    mask_49 = subject_test == 49
    y_test_49 = scaler_y.inverse_transform(y_test[mask_49].reshape(-1, 1)).flatten()
    print(f"Subject 49 - Mean dose: {y_test_49.mean():.2f}, Std: {y_test_49.std():.2f}")
    plt.figure(figsize=(8, 6))
    plt.scatter(y_test_49, y_pred_ppo[mask_49], alpha=0.5, label='PPO Predictions (Subject 49)')
    plt.plot([0, 4], [0, 4], 'k--', label='Perfect Prediction')
    plt.xlabel('True Dose (units)')
    plt.ylabel('Predicted Dose (units)')
    plt.legend()
    plt.title('PPO Predictions vs True Values (Subject 49)')
    plt.show()
else:
    print("Subject 49 not found in test set")

plot_evaluation(y_test, y_pred_ppo, y_rule, subject_test, scaler_y)



# %%
