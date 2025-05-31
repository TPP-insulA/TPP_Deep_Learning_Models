# %% CELL: Required Imports
import polars as pl
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns
import os
from joblib import Parallel, delayed
from datetime import timedelta
import time
from tqdm import tqdm
from sklearn.inspection import permutation_importance
from sklearn.model_selection import KFold, train_test_split
%matplotlib inline
# DRL-specific imports
import gymnasium as gym
from gymnasium import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.callbacks import BaseCallback, EvalCallback
import sys
import os

# Add the project root directory to Python path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(PROJECT_ROOT)


PROJECT_DIR = os.path.abspath(os.path.join(os.getcwd(), ".."))

# Global configuration
CONFIG = {
    "batch_size": 128,
    "window_hours": 2,
    "cap_normal": 30,
    "cap_bg": 300,
    "cap_iob": 5,
    "cap_carb": 150,
    "data_path": os.path.join(PROJECT_DIR, "data", "subjects"),
    "metadata_path": os.path.join(PROJECT_DIR, "data", "metadata.csv")
}

# %% CELL: Data Processing Functions
def load_metadata(metadata_path):
    """Load and clean demographic and clinical metadata."""
    try:
        metadata = pl.read_csv(metadata_path)
        
        metadata = metadata.with_columns([
            (pl.col('Subject') - 1).cast(int).alias('subject_id'),
            pl.col('Age').map_elements(
                lambda x: (
                    int(x) if x.isdigit() else
                    (int(x.split('-')[0].strip()) + int(x.split('-')[1].replace(' yrs', '').strip())) / 2
                ),
                return_dtype=pl.Float64
            ).alias('age'),
            pl.col('Gender').map_elements(lambda x: 0 if x == 'Female' else 1, return_dtype=pl.Float64).alias('gender'),
            pl.col('Race').map_elements(
                lambda x: 1.0 if x == 'White/Caucasian' else 0.0,
                return_dtype=pl.Float64
            ).alias('race'),
            pl.col('Hemoglobin A1C').map_elements(
                lambda x: float(x.replace(',', '.')) if x != 'NR' else 7.5,
                return_dtype=pl.Float64
            ).alias('hbA1c')
        ])
        
        metadata = metadata.with_columns([
            ((pl.col('age') - 19) / (74 - 19)).alias('age_normalized'),
            ((pl.col('hbA1c') - 4.0) / (12.0 - 4.0)).alias('hbA1c_normalized')  # Normalize HbA1c
        ])
        
        metadata = metadata.select(['subject_id', 'age_normalized', 'gender', 'race', 'hbA1c', 'hbA1c_normalized'])
        
        print("Metadata after cleaning:", flush=True)
        print(metadata.head(), flush=True)
        return metadata
    except Exception as e:
        print(f"Error loading metadata: {e}", flush=True)
        return None

def get_cgm_window(bolus_time, cgm_df, window_hours=CONFIG["window_hours"]):
    window_start = bolus_time - timedelta(hours=window_hours)
    window = (cgm_df.filter((pl.col('date') >= window_start) & (pl.col('date') <= bolus_time))
             .sort('date'))
    
    if window.height == 0:
        return None
    
    values = window['mg/dl'].to_numpy()
    if window.height < 24:
        last_value = values[-1] if len(values) > 0 else 100.0
        values = np.pad(values, (0, 24 - len(values)), mode='constant', constant_values=last_value)
    elif window.height > 24:
        values = values[-24:]
    
    return values

def calculate_iob(bolus_time, basal_df, half_life_hours=4):
    if basal_df is None or basal_df.height == 0:
        return 0.0
        
    iob = 0
    for row in basal_df.rows(named=True):
        start_time = row['date']
        duration_hours = row['duration'] / (1000 * 3600)
        end_time = start_time + timedelta(hours=duration_hours)
        rate = row['rate'] if row['rate'] is not None else 0.9
        rate = min(rate, 2.0)
        
        if start_time <= bolus_time <= end_time:
            time_since_start = (bolus_time - start_time).total_seconds() / 3600
            remaining = rate * (1 - (time_since_start / half_life_hours))
            iob += max(0, remaining)
            
    return min(iob, 5.0)

def process_subject(subject_path, idx, metadata):
    start_time = time.time()
    
    try:
        # Load sheets, making Basal optional
        excel_file = pl.read_excel(subject_path, sheet_name=["CGM", "Bolus", "Basal"])
        cgm_df = excel_file["CGM"].with_columns(pl.col('date').cast(pl.Datetime))
        bolus_df = excel_file["Bolus"].with_columns(pl.col('date').cast(pl.Datetime))
        basal_df = None
        try:
            basal_df = pl.read_excel(subject_path, sheet_name="Basal")
        except Exception:
            basal_df = None
            
        if basal_df is not None:
            basal_df = basal_df.with_columns(pl.col('date').cast(pl.Datetime))
    
    except Exception as e:
        print(f"Error loading {os.path.basename(subject_path)}: {e}", flush=True)
        return []

    try:
        # Rename columns safely, including insulinOnBoard
        cgm_df = cgm_df.rename({"mg/dl": "mg/dl"})
        bolus_df = bolus_df.rename({
            "date": "date",
            "normal": "normal",
            "carbInput": "carbInput",
            "bgInput": "bgInput",
            "insulinCarbRatio": "insulinCarbRatio",
            "insulinOnBoard": "insulinOnBoard"  # Ensure this column is renamed if present
        })
        if basal_df is not None:
            basal_df = basal_df.rename({"rate": "rate", "duration": "duration"})
    except Exception as e:
        print(f"Error renaming columns in {os.path.basename(subject_path)}: {e}", flush=True)
        return []

    cgm_df = cgm_df.sort('date')
    bolus_df = bolus_df.sort('date')

    # Calculate median carb value for fallbacks
    non_zero_carbs = bolus_df.filter(pl.col('carbInput') > 0)['carbInput'].to_numpy()
    carb_median = np.median(non_zero_carbs) if len(non_zero_carbs) > 0 else 10.0

    # Calculate median IOB from insulinOnBoard for subjects without Basal sheet as a fallback
    if basal_df is None and 'insulinOnBoard' in bolus_df.columns:
        iob_values = bolus_df['insulinOnBoard'].filter(pl.col('insulinOnBoard') > 0).to_numpy()
        iob_median = np.median(iob_values) if len(iob_values) > 0 else 0.5
    else:
        iob_median = 0.5  # Default for cases where Basal exists or insulinOnBoard is missing

    # Load metadata for the subject
    subject_metadata = metadata.filter(pl.col('subject_id') == idx)
    if subject_metadata.height == 0:
        age_normalized = 0.5
        gender = 0.0
        race = 1.0
        hbA1c = 7.5
        hbA1c_normalized = 0.5
    else:
        age_normalized = subject_metadata['age_normalized'][0]
        gender = subject_metadata['gender'][0]
        race = subject_metadata['race'][0]
        hbA1c = subject_metadata['hbA1c'][0]
        hbA1c_normalized = subject_metadata['hbA1c_normalized'][0]

    processed_data = []
    last_bolus_time = None
    for row in tqdm(bolus_df.rows(named=True), total=bolus_df.height, desc=f"Processing {os.path.basename(subject_path)}", leave=False):
        try:
            bolus_time = row['date']
            cgm_window = get_cgm_window(bolus_time, cgm_df)
            
            if cgm_window is not None:
                # Determine IOB based on availability of Basal sheet or insulinOnBoard
                if basal_df is not None:
                    iob = calculate_iob(bolus_time, basal_df)
                else:
                    # Use insulinOnBoard from Bolus sheet if Basal is absent
                    iob = row.get('insulinOnBoard', None)
                    if iob is None or iob < 0:
                        iob = iob_median  # Fallback if insulinOnBoard is missing or invalid
                
                # Ensure IOB is non-zero and capped
                iob = iob_median if iob == 0 else iob
                
                hour_of_day = bolus_time.hour / 23.0
                time_since_last_bolus = (bolus_time - last_bolus_time).total_seconds() / 3600 if last_bolus_time else 0.0
                last_bolus_time = bolus_time
                
                bg_input = row.get('bgInput', None)
                if bg_input is None or bg_input < 40:
                    bg_input = cgm_window[-1]
                bg_input = max(bg_input, 50.0)
                
                normal = row.get('normal', 0.0)
                if normal is None or normal < 0:
                    normal = 0.0
                normal = np.clip(normal, 0, CONFIG["cap_normal"])
                
                # Adjusted ISF calculation with cgm_trend
                isf_base = 50.0 if normal <= 0 else (bg_input - 100) / normal
                isf_adjustment = 1.0 + np.tanh((hbA1c - 7.0) * 0.1)
                cgm_trend = np.polyfit(np.arange(5), cgm_window[-5:], 1)[0] if len(cgm_window) >= 5 else 0.0
                trend_adjustment = 1.0 - 0.1 * cgm_trend
                isf_custom = np.clip(isf_base * isf_adjustment * trend_adjustment, 20, 80)
                
                carb_input = row.get('carbInput', None)
                if carb_input is None or carb_input < 0:
                    carb_input = carb_median
                
                icr = row.get('insulinCarbRatio', None)
                if icr is None or icr < 5 or icr > 20:
                    icr = 10.0
                
                hbA1c_bg_interaction = hbA1c_normalized * bg_input
                
                features = {
                    'subject_id': idx,
                    'cgm_window': cgm_window,
                    'normal': normal,
                    'carbInput': np.clip(carb_input, 0, CONFIG["cap_carb"]),
                    'bgInput': np.clip(bg_input, 0, CONFIG["cap_bg"]),
                    'insulinOnBoard': np.clip(iob, 0, CONFIG["cap_iob"]),
                    'insulinCarbRatio': np.clip(icr, 5, 20),
                    'insulinSensitivityFactor': isf_custom,
                    'hour_of_day': hour_of_day,
                    'time_since_last_bolus': time_since_last_bolus,
                    'cgm_trend': cgm_trend,
                    'age_normalized': age_normalized,
                    'gender': gender,
                    'hbA1c_normalized': hbA1c_normalized,
                    'hbA1c_bg_interaction': hbA1c_bg_interaction
                }
                processed_data.append(features)
        except Exception as e:
            print(f"Error processing row in {os.path.basename(subject_path)}: {e}", flush=True)
            continue

    elapsed_time = time.time() - start_time
    print(f"Processed {os.path.basename(subject_path)} (Subject {idx+1}) in {elapsed_time:.2f} seconds", flush=True)
    return processed_data

def preprocess_data(subject_folder, metadata):
    start_time = time.time()
    
    subject_files = [f for f in os.listdir(subject_folder) if f.startswith("Subject") and f.endswith(".xlsx")]
    print(f"\nFound Subject files ({len(subject_files)}):", flush=True)
    for f in subject_files:
        print(f, flush=True)

    all_processed_data = Parallel(n_jobs=-1)(delayed(process_subject)(os.path.join(subject_folder, f), idx, metadata) 
                                            for idx, f in enumerate(subject_files))
    all_processed_data = [item for sublist in all_processed_data for item in sublist if item]

    if not all_processed_data:
        raise ValueError("No valid data was processed from any subject file")

    try:
        df_processed = pl.DataFrame(all_processed_data)
    except Exception as e:
        print("Error creating DataFrame:", e)
        print("Sample data:", all_processed_data[0] if all_processed_data else "No data")
        raise

    print("Muestra de datos procesados combinados:", flush=True)
    print(df_processed.head(), flush=True)
    print(f"Total de muestras: {df_processed.height}", flush=True)
    print("Schema de df_processed:", flush=True)
    print(df_processed.schema, flush=True)

    expected_columns = ['normal', 'carbInput', 'insulinOnBoard', 'bgInput']
    missing_columns = [col for col in expected_columns if col not in df_processed.columns]
    if missing_columns:
        print("Available columns:", df_processed.columns, flush=True)
        print("Missing columns:", missing_columns, flush=True)
        raise ValueError(f"Missing expected columns in df_processed: {missing_columns}")

    transforms = []
    for col in expected_columns:
        transforms.append(pl.col(col).fill_null(0).log1p().alias(col))
    
    df_processed = df_processed.with_columns(transforms)
    
    cgm_data = np.array(df_processed['cgm_window'].to_list())
    cgm_data = np.log1p(cgm_data)
    cgm_columns = [f'cgm_{i}' for i in range(24)]
    df_cgm = pl.DataFrame({col: cgm_data[:, i] for i, col in enumerate(cgm_columns)})
    
    df_final = pl.concat([df_cgm, df_processed.drop('cgm_window')], how='horizontal')
    df_final = df_final.drop_nulls()
    
    print("Verificación de NaN en df_final:", flush=True)
    print(df_final.null_count(), flush=True)

    elapsed_time = time.time() - start_time
    print(f"Preprocesamiento completo en {elapsed_time:.2f} segundos", flush=True)
    return df_final

def split_folds_data(df_final, n_splits=5):
    start_time = time.time()
    
    subject_ids = df_final['subject_id'].unique().to_numpy()
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    
    results = []
    for fold, (train_val_idx, test_idx) in enumerate(kf.split(subject_ids)):
        train_val_subjects = subject_ids[train_val_idx]
        test_subjects = subject_ids[test_idx]
        
        train_subjects, val_subjects = train_test_split(train_val_subjects, test_size=0.1, random_state=42)
        
        train_mask = df_final['subject_id'].is_in(train_subjects)
        val_mask = df_final['subject_id'].is_in(val_subjects)
        test_mask = df_final['subject_id'].is_in(test_subjects)

        y_train_temp = df_final.filter(train_mask)['normal'].to_numpy()
        y_val_temp = df_final.filter(val_mask)['normal'].to_numpy()
        y_test_temp = df_final.filter(test_mask)['normal'].to_numpy()
        print(f"Fold {fold+1} - Post-split Train y: mean = {y_train_temp.mean():.2f}, std = {y_train_temp.std():.2f}", flush=True)
        print(f"Fold {fold+1} - Post-split Val y: mean = {y_val_temp.mean():.2f}, std = {y_val_temp.std():.2f}", flush=True)
        print(f"Fold {fold+1} - Post-split Test y: mean = {y_test_temp.mean():.2f}, std = {y_test_temp.std():.2f}", flush=True)

        scaler_cgm = StandardScaler()
        scaler_other = StandardScaler()
        scaler_y = StandardScaler()
        cgm_columns = [f'cgm_{i}' for i in range(24)]
        other_features = ['carbInput', 'bgInput', 'insulinOnBoard', 'insulinCarbRatio', 
                         'insulinSensitivityFactor', 'time_since_last_bolus', 'cgm_trend', 
                         'age_normalized', 'hbA1c_normalized', 'hbA1c_bg_interaction']

        X_cgm_train = scaler_cgm.fit_transform(df_final.filter(train_mask)[cgm_columns].to_numpy()).reshape(-1, 24, 1)
        X_cgm_val = scaler_cgm.transform(df_final.filter(val_mask)[cgm_columns].to_numpy()).reshape(-1, 24, 1)
        X_cgm_test = scaler_cgm.transform(df_final.filter(test_mask)[cgm_columns].to_numpy()).reshape(-1, 24, 1)
        
        # Add noise to training data for augmentation
        noise = np.random.normal(0, 0.05, X_cgm_train.shape)
        X_cgm_train = X_cgm_train + noise
        
        X_other_train = scaler_other.fit_transform(df_final.filter(train_mask)[other_features].to_numpy())
        X_other_val = scaler_other.transform(df_final.filter(val_mask)[other_features].to_numpy())
        X_other_test = scaler_other.transform(df_final.filter(test_mask)[other_features].to_numpy())
        
        y_train = scaler_y.fit_transform(df_final.filter(train_mask)['normal'].to_numpy().reshape(-1, 1)).flatten()
        y_val = scaler_y.transform(df_final.filter(val_mask)['normal'].to_numpy().reshape(-1, 1)).flatten()
        y_test = scaler_y.transform(df_final.filter(test_mask)['normal'].to_numpy().reshape(-1, 1)).flatten()

        X_subject_train = df_final.filter(train_mask)['subject_id'].to_numpy()
        X_subject_val = df_final.filter(val_mask)['subject_id'].to_numpy()
        X_subject_test = df_final.filter(test_mask)['subject_id'].to_numpy()

        print(f"Fold {fold+1} - Entrenamiento CGM: {X_cgm_train.shape}, Validación CGM: {X_cgm_val.shape}, Prueba CGM: {X_cgm_test.shape}", flush=True)
        print(f"Fold {fold+1} - Entrenamiento Otros: {X_other_train.shape}, Validación Otros: {X_other_val.shape}, Prueba Otros: {X_other_test.shape}", flush=True)
        print(f"Fold {fold+1} - Sujetos de prueba: {test_subjects}",flush=True)

        results.append((
            X_cgm_train, X_cgm_val, X_cgm_test,
            X_other_train, X_other_val, X_other_test,
            X_subject_train, X_subject_val, X_subject_test,
            y_train, y_val, y_test, X_subject_test,
            scaler_cgm, scaler_other, scaler_y
        ))

    elapsed_time = time.time() - start_time
    print(f"División de datos completa en {elapsed_time:.2f} segundos", flush=True)
    return results


def split_data(df_final):
    start_time = time.time()
    
    subject_stats = (df_final.group_by('subject_id')
                    .agg([
                        pl.col('normal').mean().alias('mean_dose'),
                        pl.col('normal').std().alias('std_dose')
                    ]))
    subject_ids = df_final['subject_id'].unique().to_numpy()
    sorted_subjects = subject_stats.sort('mean_dose')['subject_id'].to_numpy()
    n_subjects = len(sorted_subjects)

    train_size = int(0.8 * n_subjects)
    val_size = int(0.1 * n_subjects)
    test_size = n_subjects - train_size - val_size

    train_subjects = []
    val_subjects = []
    test_subjects = []
    remaining_subjects = list(sorted_subjects)
    np.random.shuffle(remaining_subjects)

    for subject in remaining_subjects:
        train_mean = df_final.filter(pl.col('subject_id').is_in(train_subjects))['normal'].mean() if train_subjects else 0
        val_mean = df_final.filter(pl.col('subject_id').is_in(val_subjects))['normal'].mean() if val_subjects else 0
        test_mean = df_final.filter(pl.col('subject_id').is_in(test_subjects))['normal'].mean() if test_subjects else 0
        train_std = df_final.filter(pl.col('subject_id').is_in(train_subjects))['normal'].std() if train_subjects else 0
        val_std = df_final.filter(pl.col('subject_id').is_in(val_subjects))['normal'].std() if val_subjects else 0
        test_std = df_final.filter(pl.col('subject_id').is_in(test_subjects))['normal'].std() if test_subjects else 0

        train_temp = train_subjects + [subject]
        val_temp = val_subjects + [subject]
        test_temp = test_subjects + [subject]

        train_mean_new = df_final.filter(pl.col('subject_id').is_in(train_temp))['normal'].mean()
        val_mean_new = df_final.filter(pl.col('subject_id').is_in(val_temp))['normal'].mean()
        test_mean_new = df_final.filter(pl.col('subject_id').is_in(test_temp))['normal'].mean()
        train_std_new = df_final.filter(pl.col('subject_id').is_in(train_temp))['normal'].std()
        val_std_new = df_final.filter(pl.col('subject_id').is_in(val_temp))['normal'].std()
        test_std_new = df_final.filter(pl.col('subject_id').is_in(test_temp))['normal'].std()

        means_if_train = [train_mean_new, val_mean, test_mean]
        means_if_val = [train_mean, val_mean_new, test_mean]
        means_if_test = [train_mean, val_mean, test_mean_new]
        stds_if_train = [train_std_new, val_std, test_std]
        stds_if_val = [train_std, val_std_new, test_std]
        stds_if_test = [train_std, val_std, test_std_new]

        range_means_if_train = max(means_if_train) - min(means_if_train) if all(m != 0 for m in means_if_train) else float('inf')
        range_means_if_val = max(means_if_val) - min(means_if_val) if all(m != 0 for m in means_if_val) else float('inf')
        range_means_if_test = max(means_if_test) - min(means_if_test) if all(m != 0 for m in means_if_test) else float('inf')
        range_stds_if_train = max(stds_if_train) - min(stds_if_train) if all(s != 0 for s in stds_if_train) else float('inf')
        range_stds_if_val = max(stds_if_val) - min(stds_if_val) if all(s != 0 for s in stds_if_val) else float('inf')
        range_stds_if_test = max(stds_if_test) - min(stds_if_test) if all(s != 0 for s in stds_if_test) else float('inf')

        score_if_train = range_means_if_train + range_stds_if_train
        score_if_val = range_means_if_val + range_stds_if_val
        score_if_test = range_means_if_test + range_stds_if_test

        if len(train_subjects) < train_size and score_if_train <= min(score_if_val, score_if_test):
            train_subjects.append(subject)
        elif len(val_subjects) < val_size and score_if_val <= min(score_if_train, score_if_test):
            val_subjects.append(subject)
        elif len(test_subjects) < test_size:
            test_subjects.append(subject)
        else:
            train_subjects.append(subject)

    train_mask = df_final['subject_id'].is_in(train_subjects)
    val_mask = df_final['subject_id'].is_in(val_subjects)
    test_mask = df_final['subject_id'].is_in(test_subjects)

    y_train_temp = df_final.filter(train_mask)['normal'].to_numpy()
    y_val_temp = df_final.filter(val_mask)['normal'].to_numpy()
    y_test_temp = df_final.filter(test_mask)['normal'].to_numpy()
    print("Post-split Train y: mean =", y_train_temp.mean(), "std =", y_train_temp.std())
    print("Post-split Val y: mean =", y_val_temp.mean(), "std =", y_val_temp.std())
    print("Post-split Test y: mean =", y_test_temp.mean(), "std =", y_test_temp.std())

    scaler_cgm = StandardScaler()
    scaler_other = StandardScaler()
    scaler_y = StandardScaler()
    cgm_columns = [f'cgm_{i}' for i in range(24)]
    other_features = ['carbInput', 'bgInput', 'insulinOnBoard', 'insulinCarbRatio', 
                    'insulinSensitivityFactor', 'hour_of_day', 'cgm_trend']

    X_cgm_train = scaler_cgm.fit_transform(df_final.filter(train_mask)[cgm_columns].to_numpy()).reshape(-1, 24, 1)
    X_cgm_val = scaler_cgm.transform(df_final.filter(val_mask)[cgm_columns].to_numpy()).reshape(-1, 24, 1)
    X_cgm_test = scaler_cgm.transform(df_final.filter(test_mask)[cgm_columns].to_numpy()).reshape(-1, 24, 1)
    
    X_other_train = scaler_other.fit_transform(df_final.filter(train_mask)[other_features].to_numpy())
    X_other_val = scaler_other.transform(df_final.filter(val_mask)[other_features].to_numpy())
    X_other_test = scaler_other.transform(df_final.filter(test_mask)[other_features].to_numpy())
    
    y_train = scaler_y.fit_transform(df_final.filter(train_mask)['normal'].to_numpy().reshape(-1, 1)).flatten()
    y_val = scaler_y.transform(df_final.filter(val_mask)['normal'].to_numpy().reshape(-1, 1)).flatten()
    y_test = scaler_y.transform(df_final.filter(test_mask)['normal'].to_numpy().reshape(-1, 1)).flatten()

    X_subject_train = df_final.filter(train_mask)['subject_id'].to_numpy()
    X_subject_val = df_final.filter(val_mask)['subject_id'].to_numpy()
    X_subject_test = df_final.filter(test_mask)['subject_id'].to_numpy()

    print(f"Entrenamiento CGM: {X_cgm_train.shape}, Validación CGM: {X_cgm_val.shape}, Prueba CGM: {X_cgm_test.shape}")
    print(f"Entrenamiento Otros: {X_other_train.shape}, Validación Otros: {X_other_val.shape}, Prueba Otros: {X_other_test.shape}")
    print(f"Entrenamiento Subject: {X_subject_train.shape}, Validación Subject: {X_subject_val.shape}, Prueba Subject: {X_subject_test.shape}")
    print(f"Sujetos de prueba: {test_subjects}")

    elapsed_time = time.time() - start_time
    print(f"División de datos completa en {elapsed_time:.2f} segundos")
    return (X_cgm_train, X_cgm_val, X_cgm_test,
            X_other_train, X_other_val, X_other_test,
            X_subject_train, X_subject_val, X_subject_test,
            y_train, y_val, y_test, X_subject_test,
            scaler_cgm, scaler_other, scaler_y)

def rule_based_prediction(X_other, scaler_other, scaler_y, target_bg=100):
    start_time = time.time()
    
    inverse_transformed = scaler_other.inverse_transform(X_other)
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

# %% CELL: Clinical Evaluation Functions
def estimate_future_glucose(cgm_window, insulin_dose, iob, isf, carb_input, icr):
    """Improved glucose prediction model with non-linear insulin effect and delayed carb absorption."""
    current_glucose = cgm_window[-1]
    
    # Non-linear insulin effect: use a logarithmic decay to model diminishing returns
    total_insulin = insulin_dose + iob * 0.5
    insulin_effect = isf * np.log1p(total_insulin)  # Logarithmic effect to prevent over-correction
    
    # Carb effect: model delayed absorption with a two-phase response
    immediate_carb_effect = (carb_input * 0.2 / icr) * isf  # 20% immediate
    delayed_carb_effect = (carb_input * 0.8 / icr) * isf * 0.5  # 80% delayed, scaled down
    total_carb_effect = immediate_carb_effect + delayed_carb_effect
    
    # Adjust for CGM trend
    trend = np.polyfit(np.arange(5), cgm_window[-5:], 1)[0] if len(cgm_window) >= 5 else 0.0
    trend_effect = trend * 5.0  # Amplify trend effect over the next 5 minutes
    
    future_glucose = current_glucose - insulin_effect + total_carb_effect + trend_effect
    future_glucose = np.clip(future_glucose, 30, 400)
    return future_glucose

def evaluate_clinical_metrics(y_true, y_pred, X_cgm, X_other, scaler_y, scaler_other):
    y_true_denorm = scaler_y.inverse_transform(y_true.reshape(-1, 1)).flatten()
    y_pred_denorm = y_pred
    
    X_other_denorm = scaler_other.inverse_transform(X_other)
    carb_input = X_other_denorm[:, 0]
    iob = X_other_denorm[:, 2]
    isf = X_other_denorm[:, 4]
    icr = X_other_denorm[:, 3]
    
    hypo_count = 0
    hyper_count = 0
    in_range_count = 0
    total = len(y_pred)
    
    for i in range(total):
        cgm_window = X_cgm[i].flatten()
        cgm_window = np.expm1(cgm_window)
        future_glucose = estimate_future_glucose(cgm_window, y_pred_denorm[i], iob[i], isf[i], carb_input[i], icr[i])
        
        if future_glucose < 70:
            hypo_count += 1
        elif future_glucose > 180:
            hyper_count += 1
        else:
            in_range_count += 1
    
    print("\nClinical Metrics:", flush=True)
    print(f"Hypoglycemia Risk (<70 mg/dL): {hypo_count} ({hypo_count/total*100:.2f}%)", flush=True)
    print(f"Hyperglycemia Risk (>180 mg/dL): {hyper_count} ({hyper_count/total*100:.2f}%)", flush=True)
    print(f"Time in Range (70-180 mg/dL): {in_range_count} ({in_range_count/total*100:.2f}%)", flush=True)

# %% CELL: Visualization and Analysis Functions
def compute_metrics(y_true, y_pred, scaler_y):
    y_true_denorm = scaler_y.inverse_transform(y_true.reshape(-1, 1)).flatten()
    y_pred_denorm = y_pred
    mae = mean_absolute_error(y_true_denorm, y_pred_denorm)
    rmse = np.sqrt(mean_squared_error(y_true_denorm, y_pred_denorm))
    r2 = r2_score(y_true_denorm, y_pred_denorm)
    return mae, rmse, r2

def plot_evaluation(y_test, y_pred_ppo, y_rule, subject_test, scaler_y):
    start_time = time.time()
    y_test_denorm = scaler_y.inverse_transform(y_test.reshape(-1, 1)).flatten()

    colors = {'PPO': 'green', 'Rules': 'orange'}
    offset = 1e-2

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
    fig = plt.gcf()
    plt.show()
    plt.close(fig)  # FIX: Close the figure

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
    fig = plt.gcf()
    plt.show()
    plt.close(fig)  # FIX: Close the figure

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
    plt.ylim(0, 2.5)
    plt.title('MAE by Subject', fontsize=12)
    plt.legend(fontsize=8)
    plt.grid(True, axis='y', alpha=0.3)
    fig = plt.gcf()
    plt.show()
    plt.close(fig)  # FIX: Close the figure

    elapsed_time = time.time() - start_time

def predict_with_ppo(model, X_cgm, X_other):
    predictions = []
    for i in range(len(X_cgm)):
        cgm_state = X_cgm[i].flatten()
        other_state = X_other[i]
        state = np.concatenate([cgm_state, other_state])
        action, _ = model.predict(state, deterministic=True)
        predictions.append(action[0])
    return np.array(predictions)

def plot_feature_importance(model, X_cgm, X_other, y, scaler_y, feature_names):
    start_time = time.time()
    
    X_cgm_flat = X_cgm.reshape(X_cgm.shape[0], -1)
    X_combined = np.hstack((X_cgm_flat, X_other))
    all_feature_names = [f'cgm_{i}' for i in range(24)] + feature_names
    
    class PPOWrapper:
        def __init__(self, ppo_model, scaler_y):
            self.ppo_model = ppo_model
            self.scaler_y = scaler_y
            
        def fit(self, X, y):
            return self
            
        def predict(self, X):
            X_cgm = X[:, :24].reshape(-1, 24, 1)
            X_other = X[:, 24:]
            return predict_with_ppo(self.ppo_model, X_cgm, X_other)
    
    wrapped_model = PPOWrapper(model, scaler_y)
    
    result = permutation_importance(
        wrapped_model, X_combined, y=scaler_y.inverse_transform(y.reshape(-1, 1)).flatten(),
        n_repeats=3, random_state=42, scoring='neg_mean_absolute_error'  # Reduced n_repeats
    )
    
    plt.figure(figsize=(10, 6))
    sorted_idx = result.importances_mean.argsort()
    plt.barh(np.array(all_feature_names)[sorted_idx], result.importances_mean[sorted_idx])
    plt.xlabel('Importance (Increase in MAE when permuted)')
    plt.title('Feature Importance (Permutation Importance)')
    plt.tight_layout()
    fig = plt.gcf()
    plt.show()
    plt.close(fig)  # FIX: Close the figure
    
    top_features = np.array(all_feature_names)[sorted_idx][-5:]
    top_importances = result.importances_mean[sorted_idx][-5:]
    print("\nTop 5 Features by Importance:", flush=True)
    for feat, imp in zip(top_features, top_importances):
        print(f"{feat}: {imp:.4f}", flush=True)
    
    elapsed_time = time.time() - start_time
    print(f"Feature importance computation complete in {elapsed_time:.2f} seconds", flush=True)

def count_prediction_types(y_true, y_pred, scaler_y):
    y_true_denorm = scaler_y.inverse_transform(y_true.reshape(-1, 1)).flatten()
    y_pred_denorm = y_pred
    
    low_threshold = np.percentile(y_true_denorm, 25)
    high_threshold = np.percentile(y_true_denorm, 75)
    
    low_dose = np.sum(y_pred_denorm < low_threshold)
    medium_dose = np.sum((y_pred_denorm >= low_threshold) & (y_pred_denorm <= high_threshold))
    high_dose = np.sum(y_pred_denorm > high_threshold)
    
    total = len(y_pred_denorm)
    print("\nPrediction Type Counts:", flush=True)
    print(f"Low Dose (< {low_threshold:.2f} units, risk of hyperglycemia): {low_dose} ({low_dose/total*100:.2f}%)", flush=True)
    print(f"Medium Dose ({low_threshold:.2f} to {high_threshold:.2f} units, normal range): {medium_dose} ({medium_dose/total*100:.2f}%)", flush=True)
    print(f"High Dose (> {high_threshold:.2f} units, risk of hypoglycemia): {high_dose} ({high_dose/total*100:.2f}%)", flush=True)

# %% CELL: DRL Environment Definition
class InsulinDoseEnv(gym.Env):
    def __init__(self, X_cgm, X_other, y, scaler_y, scaler_other):
        super(InsulinDoseEnv, self).__init__()
        self.X_cgm = X_cgm.astype(np.float32)
        self.X_other = X_other.astype(np.float32)
        self.y = y.astype(np.float32)
        self.scaler_y = scaler_y
        self.scaler_other = scaler_other
        self.current_step = 0
        self.n_samples = len(X_cgm)
        
        state_dim = X_cgm.shape[2] * X_cgm.shape[1] + X_other.shape[1]
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(state_dim,), dtype=np.float32)
        # Using symmetric normalized action space [-1, 1] as recommended
        self.action_space = spaces.Box(low=-1, high=1, shape=(1,), dtype=np.float32)
        
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step = np.random.randint(0, self.n_samples)
        cgm_state = self.X_cgm[self.current_step].flatten()
        other_state = self.X_other[self.current_step]
        state = np.concatenate([cgm_state, other_state]).astype(np.float32)
        self.current_state = state
        return state, {}
    
    def step(self, action):
        # Scale action from [-1, 1] to appropriate insulin dose range
        scaled_action = action * 3  # Scale to approximately [-3, 3] as in the original code
        
        true_dose = self.scaler_y.inverse_transform(self.y[self.current_step].reshape(-1, 1)).flatten()[0]
        predicted_dose = self.scaler_y.inverse_transform(scaled_action.reshape(-1, 1)).flatten()[0]
        
        X_other_denorm = self.scaler_other.inverse_transform(self.X_other[self.current_step].reshape(1, -1)).flatten()
        carb_input = X_other_denorm[0]
        iob = X_other_denorm[2]
        isf = X_other_denorm[4]
        icr = X_other_denorm[3]
        cgm_window = np.expm1(self.X_cgm[self.current_step].flatten())
        future_glucose = estimate_future_glucose(cgm_window, predicted_dose, iob, isf, carb_input, icr)
        
        error = predicted_dose - true_dose
        weight = 1.0 + (true_dose / 3.0)
        reward = -min(abs(error), 2.0) * weight
        
        # Adjusted reward to strongly discourage hypoglycemia
        if future_glucose < 70:
            reward -= 20.0 * (70 - future_glucose) / 70  # Increased penalty
        elif future_glucose > 180:
            reward -= 5.0 * (future_glucose - 180) / 180
        else:
            reward += 5.0  # Increased positive reward for staying in range
        
        # Penalize extreme doses
        if predicted_dose > 10.0:
            reward -= 10.0 * (predicted_dose - 10.0) / 10.0
        
        reward = float(reward)
        done = True
        truncated = False
        info = {"true_dose": true_dose, "predicted_dose": predicted_dose, "future_glucose": future_glucose}
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
        if self.num_timesteps % 1000 == 0:
            val_reward = self.evaluate_val()
            self.val_rewards.append(val_reward)
        return True

    def evaluate_val(self):
        total_reward = 0
        for _ in range(10):
            state, _ = self.val_env.reset()
            action, _ = self.model.predict(state, deterministic=True)
            _, reward, _, _, _ = self.val_env.step(action)
            total_reward += reward
        return total_reward / 10



# %% CELL: Main Execution - Preprocess Data
metadata = load_metadata(CONFIG["metadata_path"])
df_final = preprocess_data(CONFIG["data_path"], metadata)

(X_cgm_train, X_cgm_val, X_cgm_test,
 X_other_train, X_other_val, X_other_test,
 X_subject_train, X_subject_val, X_subject_test,
 y_train, y_val, y_test, subject_test,
 scaler_cgm, scaler_other, scaler_y) = split_data(df_final)

# %% CELL: Main Execution - DRL (PPO) Training
train_env_ppo = InsulinDoseEnv(X_cgm_train, X_other_train, y_train, scaler_y, scaler_other)
val_env_ppo = InsulinDoseEnv(X_cgm_val, X_other_val, y_val, scaler_y, scaler_other)
callback = RewardCallback(val_env=val_env_ppo)
check_env(train_env_ppo)

# Adjusted hyperparameters to reduce overfitting
model_ppo = PPO("MlpPolicy", 
                train_env_ppo, 
                verbose=1, 
                learning_rate=5e-5,  # Reduced learning rate
                n_steps=2048, 
                batch_size=128,  # Increased batch size
                clip_range=0.1,  # Tighter clipping
                ent_coef=0.02,  # Increased entropy coefficient
                gamma=0.99,  # Discount factor
                gae_lambda=0.95)  # GAE lambda
total_timesteps = 100000  # Increased training steps
model_ppo.learn(total_timesteps=total_timesteps, callback=callback)

plt.plot(callback.train_rewards, label='Train Reward')
plt.plot(np.arange(len(callback.val_rewards)) * 1000, callback.val_rewards, label='Val Reward')
plt.xlabel('Timestep')
plt.ylabel('Reward')
plt.legend()
plt.title('PPO Training vs Validation Reward')
plt.show()
plt.close()

model_ppo.save("ppo_insulin_dose")

def predict_with_ppo(model, X_cgm, X_other):
    predictions = []
    env = InsulinDoseEnv(X_cgm, X_other, np.zeros(len(X_cgm)), scaler_y, scaler_other)
    for i in range(len(X_cgm)):
        cgm_state = X_cgm[i].flatten()
        other_state = X_other[i]
        state = np.concatenate([cgm_state, other_state])
        action, _ = model.predict(state, deterministic=True)
        predicted_dose = scaler_y.inverse_transform(action.reshape(-1, 1)).flatten()[0]
        predictions.append(predicted_dose)
    return np.array(predictions)

y_pred_ppo_train = predict_with_ppo(model_ppo, X_cgm_train, X_other_train)
y_pred_ppo_val = predict_with_ppo(model_ppo, X_cgm_val, X_other_val)
y_pred_ppo = predict_with_ppo(model_ppo, X_cgm_test, X_other_test)
y_rule = rule_based_prediction(X_other_test, scaler_other, scaler_y)

# %% CELL: Main Execution - Print Metrics
mae_ppo = mean_absolute_error(scaler_y.inverse_transform(y_test.reshape(-1, 1)), y_pred_ppo)
rmse_ppo = np.sqrt(mean_squared_error(scaler_y.inverse_transform(y_test.reshape(-1, 1)), y_pred_ppo))
r2_ppo = r2_score(scaler_y.inverse_transform(y_test.reshape(-1, 1)), y_pred_ppo)
print(f"PPO Test - MAE: {mae_ppo:.2f}, RMSE: {rmse_ppo:.2f}, R²: {r2_ppo:.2f}")

mae_ppo_train = mean_absolute_error(scaler_y.inverse_transform(y_train.reshape(-1, 1)), y_pred_ppo_train)
rmse_ppo_train = np.sqrt(mean_squared_error(scaler_y.inverse_transform(y_train.reshape(-1, 1)), y_pred_ppo_train))
r2_ppo_train = r2_score(scaler_y.inverse_transform(y_train.reshape(-1, 1)), y_pred_ppo_train)
print(f"PPO Train - MAE: {mae_ppo_train:.2f}, RMSE: {rmse_ppo_train:.2f}, R²: {r2_ppo_train:.2f}")

mae_ppo_val = mean_absolute_error(scaler_y.inverse_transform(y_val.reshape(-1, 1)), y_pred_ppo_val)
rmse_ppo_val = np.sqrt(mean_squared_error(scaler_y.inverse_transform(y_val.reshape(-1, 1)), y_pred_ppo_val))
r2_ppo_val = r2_score(scaler_y.inverse_transform(y_val.reshape(-1, 1)), y_pred_ppo_val)
print(f"PPO Val - MAE: {mae_ppo_val:.2f}, RMSE: {rmse_ppo_val:.2f}, R²: {r2_ppo_val:.2f}")

mae_rule = mean_absolute_error(scaler_y.inverse_transform(y_test.reshape(-1, 1)), y_rule)
rmse_rule = np.sqrt(mean_squared_error(scaler_y.inverse_transform(y_test.reshape(-1, 1)), y_rule))
r2_rule = r2_score(scaler_y.inverse_transform(y_test.reshape(-1, 1)), y_rule)
print(f"Rules Test - MAE: {mae_rule:.2f}, RMSE: {rmse_rule:.2f}, R²: {r2_rule:.2f}")

# %% CELL: Main Execution - Feature Importance
feature_names = ['carbInput', 'bgInput', 'insulinOnBoard', 'insulinCarbRatio', 
                 'insulinSensitivityFactor', 'hour_of_day', 'cgm_trend']
plot_feature_importance(model_ppo, X_cgm_test, X_other_test, y_test, scaler_y, feature_names)

# %% CELL: Main Execution - Prediction Type Counter
count_prediction_types(y_test, y_pred_ppo, scaler_y)

# %% CELL: Main Execution - Visualization
plot_evaluation(y_test, y_pred_ppo, y_rule, subject_test, scaler_y)

# %% CELL: Main Execution - Metrics per Subject
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


