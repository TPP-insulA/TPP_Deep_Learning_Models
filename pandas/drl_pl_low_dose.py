# %% FILE: drl_low_polars.py
# Description: Deep Reinforcement Learning (DRL) model optimized for low-dose insulin predictions with Polars.

import polars as pl
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns
import os
from joblib import Parallel, delayed
import time
from tqdm import tqdm
from datetime import timedelta, datetime
from sklearn.ensemble import RandomForestRegressor
# DRL-specific imports
import gym
import gymnasium as gym
from gymnasium import spaces
from stable_baselines3 import PPO
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
    "data_path": os.path.join(PROJECT_DIR, "data", "subjects"),
    "low_dose_threshold": 7.0  # Clinical threshold for low-dose insulin
}

# %% CELL: Data Processing Functions
def get_cgm_window(bolus_time, cgm_df, window_hours=CONFIG["window_hours"]):
    window_start = bolus_time - timedelta(hours=window_hours)
    window = cgm_df.filter((pl.col('date') >= window_start) & (pl.col('date') <= bolus_time))
    window = window.sort('date').tail(24)
    return window['mg/dl'].to_numpy() if len(window) >= 24 else None

def calculate_iob(bolus_time, basal_df, half_life_hours=4):
    if basal_df is None or len(basal_df) == 0:
        return 0.0
    iob = 0
    for row in basal_df.iter_rows(named=True):
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

def process_subject(subject_path, idx):
    start_time = time.time()
    try:
        # Try reading with pandas first (more reliable for Excel)
        import pandas as pd
        pd_cgm = pd.read_excel(subject_path, sheet_name="CGM")
        pd_bolus = pd.read_excel(subject_path, sheet_name="Bolus")
        
        # Convert to datetime
        if 'date' in pd_cgm.columns:
            pd_cgm['date'] = pd.to_datetime(pd_cgm['date'])
        if 'date' in pd_bolus.columns:
            pd_bolus['date'] = pd.to_datetime(pd_bolus['date'])
            
        # Convert to Polars
        cgm_df = pl.from_pandas(pd_cgm)
        bolus_df = pl.from_pandas(pd_bolus)
        
        try:
            pd_basal = pd.read_excel(subject_path, sheet_name="Basal")
            if 'date' in pd_basal.columns:
                pd_basal['date'] = pd.to_datetime(pd_basal['date'])
            basal_df = pl.from_pandas(pd_basal)
        except:
            basal_df = None
            print(f"No Basal sheet in {os.path.basename(subject_path)}")
            
    except Exception as e:
        print(f"Error al cargar {os.path.basename(subject_path)}: {e}")
        return []

    print(f"Successfully loaded {os.path.basename(subject_path)}: CGM rows={len(cgm_df)}, Bolus rows={len(bolus_df)}")

    # Check for required columns
    required_cgm_cols = ['date', 'mg/dl']
    required_bolus_cols = ['date', 'normal', 'carbInput', 'bgInput']
    
    for col in required_cgm_cols:
        if col not in cgm_df.columns:
            print(f"ERROR: Required column '{col}' missing from CGM sheet in {os.path.basename(subject_path)}")
            if col == 'mg/dl' and 'value' in cgm_df.columns:
                print(f"Found 'value' column instead of 'mg/dl', renaming...")
                cgm_df = cgm_df.rename({'value': 'mg/dl'})
            else:
                return []
    
    for col in required_bolus_cols:
        if col not in bolus_df.columns:
            print(f"ERROR: Required column '{col}' missing from Bolus sheet in {os.path.basename(subject_path)}")
            if col == 'normal' and 'insulin' in bolus_df.columns:
                print(f"Found 'insulin' column instead of 'normal', renaming...")
                bolus_df = bolus_df.rename({'insulin': 'normal'})
            else:
                return []

    # Sort data
    cgm_df = cgm_df.sort('date')
    bolus_df = bolus_df.sort('date')
    if basal_df is not None:
        basal_df = basal_df.sort('date')

    # Impute missing basal rates
    basal_rate_est = 0.9 if basal_df is None or len(basal_df) == 0 else basal_df['rate'].mean()

    # Median calculations for imputation
    non_zero_carbs = bolus_df.filter(pl.col('carbInput') > 0)['carbInput']
    carb_median = non_zero_carbs.median() if len(non_zero_carbs) > 0 else 10.0
    
    # Safe IOB calculation
    iob_values = []
    for row in bolus_df.iter_rows(named=True):
        try:
            iob_values.append(calculate_iob(row['date'], basal_df))
        except:
            iob_values.append(0.5)
    non_zero_iob = [iob for iob in iob_values if iob > 0]
    iob_median = np.median(non_zero_iob) if non_zero_iob else 0.5

    # Subject-specific baselines
    mean_normal = bolus_df['normal'].mean() if 'normal' in bolus_df else 0.0
    std_normal = bolus_df['normal'].std() if 'normal' in bolus_df else 0.0
    mean_carb = bolus_df['carbInput'].mean() if 'carbInput' in bolus_df else carb_median
    mean_cgm = cgm_df['mg/dl'].mean() if 'mg/dl' in cgm_df else 100.0
    hba1c_proxy = (mean_cgm + 46.7) / 28.7

    # Simplified processing loop
    processed_data = []
    for row in tqdm(bolus_df.iter_rows(named=True), total=len(bolus_df), desc=f"Procesando {os.path.basename(subject_path)}", leave=False):
        try:
            bolus_time = row['date']
            cgm_window = get_cgm_window(bolus_time, cgm_df)
            if cgm_window is not None:
                # Basic features
                iob = calculate_iob(bolus_time, basal_df)
                iob = iob_median if iob == 0 else iob
                hour_of_day = bolus_time.hour
                bg_input = row['bgInput'] if row['bgInput'] is not None else cgm_window[-1]
                normal = row['normal'] if row['normal'] is not None else 0.0
                normal = np.clip(normal, 0, CONFIG["cap_normal"])
                bg_input = max(bg_input, 50.0)
                isf_custom = row.get('insulinSensitivityFactor', 50.0)
                isf_custom = np.clip(isf_custom, 10, 100)
                bg_input = np.clip(bg_input, 0, CONFIG["cap_bg"])
                iob = np.clip(iob, 0, CONFIG["cap_iob"])
                carb_input = row['carbInput'] if row['carbInput'] is not None else 0.0
                carb_input = carb_median if carb_input == 0 else carb_input
                carb_input = np.clip(carb_input, 0, CONFIG["cap_carb"])
                icr = np.clip(row.get('insulinCarbRatio', 10.0), 5, 20)

                # Enhanced CGM features
                cgm_mean = np.mean(cgm_window)
                cgm_std = np.std(cgm_window)
                cgm_cv = (cgm_std / cgm_mean) * 100 if cgm_mean != 0 else 0.0
                cgm_slope = np.polyfit(np.arange(5), cgm_window[-5:], 1)[0] if len(cgm_window) >= 5 else 0.0
                cgm_accel = np.polyfit(np.arange(5), cgm_window[-5:], 2)[0] if len(cgm_window) >= 5 else 0.0
                tir = np.mean((cgm_window >= 70) & (cgm_window <= 180)) * 100
                time_below_70 = np.mean(cgm_window < 70) * 100
                time_above_250 = np.mean(cgm_window > 250) * 100
                hypo_risk = 1 if np.any(cgm_window < 70) else 0
                cgm_max = np.max(cgm_window)
                time_to_peak = np.argmax(cgm_window) / 24

                # Cyclical encoding
                hour_sin = np.sin(2 * np.pi * hour_of_day / 24)
                hour_cos = np.cos(2 * np.pi * hour_of_day / 24)
                day_sin = np.sin(2 * np.pi * bolus_time.weekday() / 7)
                day_cos = np.cos(2 * np.pi * bolus_time.weekday() / 7)

                # Safe feature calculations
                try:
                    recent_bolus_window = bolus_df.filter((pl.col('date') >= bolus_time - timedelta(hours=4)) & 
                                                         (pl.col('date') < bolus_time))
                    time_since_last_bolus = (bolus_time - recent_bolus_window['date'].max()).total_seconds() / 3600 if len(recent_bolus_window) > 0 else 4.0
                    num_recent_boluses = len(recent_bolus_window)
                    total_recent_insulin = recent_bolus_window['normal'].sum() if len(recent_bolus_window) > 0 else 0.0
                except:
                    time_since_last_bolus = 4.0
                    num_recent_boluses = 0
                    total_recent_insulin = 0.0
                
                is_correction_bolus = 1 if row.get('recommended.correction', 0) is not None and row.get('recommended.correction', 0) > 0 else 0

                # Behavioral features
                carb_rate = total_recent_insulin / 4 if time_since_last_bolus < 4 else carb_input / 4
                carb_size = 'small' if carb_input < 20 else 'medium' if carb_input < 60 else 'large'

                # Interaction terms
                carb_insulin_ratio = carb_input / icr if icr != 0 else 0.0
                bg_iob_interaction = bg_input * iob
                isf_cgm_slope = isf_custom * cgm_slope
                carb_tir = carb_input * tir

                features = {
                    'subject_id': idx,
                    'cgm_window': cgm_window,
                    'carbInput': carb_input,
                    'bgInput': bg_input,
                    'insulinCarbRatio': icr,
                    'insulinSensitivityFactor': isf_custom,
                    'insulinOnBoard': iob,
                    'hour_sin': hour_sin,
                    'hour_cos': hour_cos,
                    'day_sin': day_sin,
                    'day_cos': day_cos,
                    'normal': normal,
                    'cgm_mean': cgm_mean,
                    'cgm_std': cgm_std,
                    'cgm_cv': cgm_cv,
                    'cgm_slope': cgm_slope,
                    'cgm_accel': cgm_accel,
                    'tir': tir,
                    'time_below_70': time_below_70,
                    'time_above_250': time_above_250,
                    'hypo_risk': hypo_risk,
                    'cgm_max': cgm_max,
                    'time_to_peak': time_to_peak,
                    'time_since_last_bolus': time_since_last_bolus,
                    'num_recent_boluses': num_recent_boluses,
                    'total_recent_insulin': total_recent_insulin,
                    'is_correction_bolus': is_correction_bolus,
                    'carb_rate': carb_rate,
                    'carb_size': carb_size,
                    'carb_insulin_ratio': carb_insulin_ratio,
                    'bg_iob_interaction': bg_iob_interaction,
                    'isf_cgm_slope': isf_cgm_slope,
                    'carb_tir': carb_tir,
                    'hba1c_proxy': hba1c_proxy,
                    'mean_normal': mean_normal,
                    'std_normal': std_normal,
                    'mean_carb': mean_carb
                }
                processed_data.append(features)
        except Exception as e:
            print(f"Error processing row: {e}")
            continue

    elapsed_time = time.time() - start_time
    print(f"Procesado {os.path.basename(subject_path)} (Sujeto {idx+1}) en {elapsed_time:.2f} segundos - {len(processed_data)} records")
    return processed_data

# The rest of your code (split_data, training, etc.) remains largely unchanged but needs minor adjustments
# to work with Polars DataFrames instead of Pandas. For simplicity, convert to Pandas at the end if needed:

def preprocess_data(subject_folder):
    start_time = time.time()
    subject_files = [f for f in os.listdir(subject_folder) if f.startswith("Subject") and f.endswith(".xlsx")]
    print(f"\nFound Subject files ({len(subject_files)}):")
    for f in subject_files:
        print(f)

    processed_data = Parallel(n_jobs=-1)(delayed(process_subject)(os.path.join(subject_folder, f), idx) 
                                       for idx, f in enumerate(subject_files))
    all_processed_data = [item for sublist in processed_data if sublist for item in sublist]

    if not all_processed_data:
        print("Error: No data processed from Excel files")
        return pl.DataFrame()

    df_processed = pl.DataFrame(all_processed_data)
    print("Muestra de datos procesados combinados:")
    print(df_processed.head())
    print(f"Total de muestras: {len(df_processed)}")
    
    print("Available columns in df_processed:")
    print(df_processed.columns)

    # Convert to pandas for processing that's easier in pandas
    df_processed_pd = df_processed.to_pandas()
    
    print("Available columns in df_processed_pd:")
    print(df_processed_pd.columns.tolist())
    
    # Handle column name based on availability
    if 'normal' in df_processed_pd.columns:
        normal_col = 'normal'
    elif 'Normal' in df_processed_pd.columns:
        normal_col = 'Normal'
    else:
        print("Error: No data processed from Excel files")
        return pl.DataFrame()
    
    # Early low-dose filtering
    scaler_y_temp = StandardScaler()
    y_temp = scaler_y_temp.fit_transform(df_processed_pd[normal_col].values.reshape(-1, 1)).flatten()
    dosis = scaler_y_temp.inverse_transform(y_temp.reshape(-1, 1)).flatten()
    low_dose_mask = dosis < CONFIG["low_dose_threshold"]
    
    # Use polars filter with the mask
    df_low_dose = df_processed.filter(pl.Series(low_dose_mask))
    print(f"Total low-dose samples (dosis < {CONFIG['low_dose_threshold']}): {len(df_low_dose)}")

    # Outlier removal within low-dose data (IQR method)
    df_low_dose_pd = df_low_dose.to_pandas()
    dosis_low = scaler_y_temp.inverse_transform(df_low_dose_pd[normal_col].values.reshape(-1, 1)).flatten()
    q1, q3 = np.percentile(dosis_low, [25, 75])
    iqr = q3 - q1
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr
    low_dose_outlier_mask = (dosis_low >= lower_bound) & (dosis_low <= upper_bound)
    
    # Use polars filter with the new mask
    df_low_dose_clean = df_low_dose.filter(pl.Series(low_dose_outlier_mask))
    print(f"Total low-dose samples after outlier removal: {len(df_low_dose_clean)}")

    # Interpolate CGM data
    def interpolate_cgm(cgm_window):
        if len(cgm_window) < 24:
            x = np.linspace(0, 23, len(cgm_window))
            x_new = np.arange(24)
            cgm_interp = np.interp(x_new, x, cgm_window)
            interp_ratio = (24 - len(cgm_window)) / 24
            return cgm_interp, interp_ratio
        return cgm_window, 0.0

    df_low_dose_clean = df_low_dose_clean.with_columns([
        pl.col('cgm_window').map_elements(lambda x: interpolate_cgm(x)[0], return_dtype=pl.List(pl.Float64)).alias('cgm_window'),
        pl.col('cgm_window').map_elements(lambda x: interpolate_cgm(x)[1], return_dtype=pl.Float64).alias('cgm_interpolated_ratio')
    ])

    # Logarithmic transformations
    log_cols = ['carbInput', 'insulinOnBoard', 'bgInput', 'cgm_mean', 'cgm_std', 'cgm_cv', 'total_recent_insulin']
    df_low_dose_clean = df_low_dose_clean.with_columns([pl.col(col).log1p().alias(col) for col in log_cols])
    df_low_dose_clean = df_low_dose_clean.with_columns(pl.col('cgm_window').map_elements(lambda x: np.log1p(x), return_dtype=pl.List(pl.Float64)))

    # Create dummy variables for carb_size using with_columns
    df_low_dose_clean = df_low_dose_clean.with_columns([
        (pl.col('carb_size') == 'small').cast(pl.Int64).alias('carb_size_small'),
        (pl.col('carb_size') == 'medium').cast(pl.Int64).alias('carb_size_medium'),
        (pl.col('carb_size') == 'large').cast(pl.Int64).alias('carb_size_large')
    ])

    # Expand CGM window into columns
    cgm_df = pl.DataFrame({f'cgm_{i}': df_low_dose_clean['cgm_window'].map_elements(lambda x: x[i]) for i in range(24)})
    df_final = pl.concat([cgm_df, df_low_dose_clean.drop('cgm_window')], how='horizontal')
    df_final = df_final.drop_nulls()

    print("Verificación de NaN en df_final:")
    print(df_final.null_count())

    elapsed_time = time.time() - start_time
    print(f"Preprocesamiento completo en {elapsed_time:.2f} segundos")
    return df_final

def split_data(df_final):
    # Convert to pandas if it's a Polars DataFrame
    if isinstance(df_final, pl.DataFrame):
        df_final = df_final.to_pandas()
    
    # Debug: Check DataFrame columns
    print("Columns in df_final before processing:")
    print(df_final.columns.tolist())
    
    # Check if 'subject_id' exists
    if 'subject_id' not in df_final.columns:
        # Try case-insensitive matching
        subject_cols = [col for col in df_final.columns if col.lower() == 'subject_id']
        if subject_cols:
            # Rename the column to 'subject_id'
            df_final = df_final.rename(columns={subject_cols[0]: 'subject_id'})
            print(f"Renamed '{subject_cols[0]}' to 'subject_id'")
        else:
            # Create dummy subject_id if it doesn't exist at all
            print("Warning: No 'subject_id' column found. Creating a dummy column.")
            unique_subjects = 10  # Arbitrary number of subjects
            n_samples = len(df_final)
            # Assign roughly equal samples to each subject
            df_final['subject_id'] = np.array([i % unique_subjects for i in range(n_samples)])
    
    start_time = time.time()
    
    # Print unique subject IDs
    print(f"Unique subject IDs: {df_final['subject_id'].unique()}")
    
    # Perform groupby operation with error checking
    try:
        subject_stats = df_final.groupby('subject_id')['normal'].agg(['mean', 'std']).reset_index()
        subject_stats.columns = ['subject_id', 'mean_dose', 'std_dose']
    except KeyError as e:
        print(f"Error in groupby: {e}")
        # Create a fallback subject_stats DataFrame
        unique_subjects = df_final['subject_id'].unique()
        means = [df_final[df_final['subject_id'] == sid]['normal'].mean() for sid in unique_subjects]
        stds = [df_final[df_final['subject_id'] == sid]['normal'].std() for sid in unique_subjects]
        subject_stats = pd.DataFrame({
            'subject_id': unique_subjects,
            'mean_dose': means,
            'std_dose': stds
        })
    
    subject_ids = df_final['subject_id'].unique()

    # Sort subjects by mean dose for stratified split
    sorted_subjects = subject_stats.sort_values('mean_dose')['subject_id'].values
    n_subjects = len(sorted_subjects)
    train_size = int(0.8 * n_subjects)
    val_size = int(0.1 * n_subjects)
    test_size = n_subjects - train_size - val_size

    # Define special test subject if it exists
    test_subjects = [49] if 49 in sorted_subjects else []
    remaining_subjects = [s for s in sorted_subjects if s != 49]
    train_subjects = []
    val_subjects = []

    # Shuffle for randomization
    remaining_subjects_list = list(remaining_subjects)
    np.random.shuffle(remaining_subjects_list)

    # Remainder of the function remains unchanged
    for i, subject in enumerate(remaining_subjects_list):
        train_mean = df_final[df_final['subject_id'].isin(train_subjects)]['normal'].mean() if train_subjects else 0
        val_mean = df_final[df_final['subject_id'].isin(val_subjects)]['normal'].mean() if val_subjects else 0
        test_mean = df_final[df_final['subject_id'].isin(test_subjects)]['normal'].mean() if test_subjects else 0
        train_std = df_final[df_final['subject_id'].isin(train_subjects)]['normal'].std() if train_subjects else 0
        val_std = df_final[df_final['subject_id'].isin(val_subjects)]['normal'].std() if val_subjects else 0
        test_std = df_final[df_final['subject_id'].isin(test_subjects)]['normal'].std() if test_subjects else 0

        # ... existing code for subject distribution ...
        train_temp = train_subjects + [subject]
        val_temp = val_subjects + [subject]
        test_temp = test_subjects + [subject]

        train_mean_new = df_final[df_final['subject_id'].isin(train_temp)]['normal'].mean()
        val_mean_new = df_final[df_final['subject_id'].isin(val_temp)]['normal'].mean()
        test_mean_new = df_final[df_final['subject_id'].isin(test_temp)]['normal'].mean()
        train_std_new = df_final[df_final['subject_id'].isin(train_temp)]['normal'].std()
        val_std_new = df_final[df_final['subject_id'].isin(val_temp)]['normal'].std()
        test_std_new = df_final[df_final['subject_id'].isin(test_temp)]['normal'].std()

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

    # Create masks for each set
    train_mask = df_final['subject_id'].isin(train_subjects)
    val_mask = df_final['subject_id'].isin(val_subjects)
    test_mask = df_final['subject_id'].isin(test_subjects)

    # Calculate statistics
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
    
    # Define the feature set
    # Check if all features exist
    all_other_features = ['carbInput', 'bgInput', 'insulinOnBoard', 'insulinCarbRatio', 
                      'insulinSensitivityFactor', 'hour_sin', 'hour_cos', 
                      'time_since_last_bolus', 'carb_insulin_ratio', 'mean_normal', 
                      'std_normal', 'hba1c_proxy', 'cgm_23']
    
    # Filter to only include features that exist in the DataFrame
    other_features = [feat for feat in all_other_features if feat in df_final.columns]
    
    if not other_features:
        print("Warning: None of the expected features found. Using all non-CGM columns")
        other_features = [col for col in df_final.columns if col not in cgm_columns 
                         and col != 'subject_id' and col != 'normal']
    
    print(f"Using the following other features: {other_features}")

    # Check if we have enough data in each split
    if sum(train_mask) == 0 or sum(val_mask) == 0 or sum(test_mask) == 0:
        print("Warning: One or more splits have no data! Using a default 80/10/10 split")
        # Perform a simple split without subject consideration
        train_size = int(0.8 * len(df_final))
        val_size = int(0.1 * len(df_final))
        indices = np.arange(len(df_final))
        np.random.shuffle(indices)
        train_mask = np.zeros(len(df_final), dtype=bool)
        val_mask = np.zeros(len(df_final), dtype=bool)
        test_mask = np.zeros(len(df_final), dtype=bool)
        train_mask[indices[:train_size]] = True
        val_mask[indices[train_size:train_size+val_size]] = True
        test_mask[indices[train_size+val_size:]] = True

    # Prepare data for the models
    # Scale features
    X_cgm_train = scaler_cgm.fit_transform(df_final.loc[train_mask, cgm_columns]).reshape(-1, 24, 1)
    X_cgm_val = scaler_cgm.transform(df_final.loc[val_mask, cgm_columns]).reshape(-1, 24, 1)
    X_cgm_test = scaler_cgm.transform(df_final.loc[test_mask, cgm_columns]).reshape(-1, 24, 1)
    X_other_train = scaler_other.fit_transform(df_final.loc[train_mask, other_features])
    X_other_val = scaler_other.transform(df_final.loc[val_mask, other_features])
    X_other_test = scaler_other.transform(df_final.loc[test_mask, other_features])
    y_train = scaler_y.fit_transform(df_final.loc[train_mask, 'normal'].values.reshape(-1, 1)).flatten()
    y_val = scaler_y.transform(df_final.loc[val_mask, 'normal'].values.reshape(-1, 1)).flatten()
    y_test = scaler_y.transform(df_final.loc[test_mask, 'normal'].values.reshape(-1, 1)).flatten()

    # Extract subject IDs for each split
    X_subject_train = df_final.loc[train_mask, 'subject_id'].values
    X_subject_val = df_final.loc[val_mask, 'subject_id'].values
    X_subject_test = df_final.loc[test_mask, 'subject_id'].values
    subject_test = X_subject_test

    # Print dataset dimensions
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
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    plt.savefig(f'plots/ppo_predictions_vs_real_{timestamp}.png', bbox_inches='tight')

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
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    plt.savefig(f'plots/residual_distribution_{timestamp}.png', bbox_inches='tight')

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
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    plt.savefig(f'plots/mae_by_subject_{timestamp}.png', bbox_inches='tight')

    elapsed_time = time.time() - start_time
    print(f"Visualización completa en {elapsed_time:.2f} segundos")

# %% CELL: DRL Environment Definition (Modified for Low-Dose)
class InsulinDoseEnv(gym.Env):
    def __init__(self, X_cgm, X_other, y, scaler_y):
        super(InsulinDoseEnv, self).__init__()
        self.X_cgm = X_cgm.astype(np.float32)
        self.X_other = X_other.astype(np.float32)
        self.y = y.astype(np.float32)
        self.scaler_y = scaler_y
        self.current_step = 0
        self.n_samples = len(X_cgm)
        # Update state dimension to account for new features
        state_dim = X_cgm.shape[2] * X_cgm.shape[1] + X_other.shape[1]
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(state_dim,), dtype=np.float32)
        self.action_space = spaces.Box(low=-3, high=3, shape=(1,), dtype=np.float32)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step = np.random.randint(0, self.n_samples)
        cgm_state = self.X_cgm[self.current_step].flatten()
        other_state = self.X_other[self.current_step]
        state = np.concatenate([cgm_state, other_state]).astype(np.float32)
        self.current_state = state
        return state, {}

    def step(self, action):
        true_dose = self.scaler_y.inverse_transform(self.y[self.current_step].reshape(-1, 1)).flatten()[0]
        predicted_dose = self.scaler_y.inverse_transform(action.reshape(-1, 1)).flatten()[0]
        error = predicted_dose - true_dose

        # Linear penalty for error, scaled by the true dose to normalize
        base_reward = -abs(error) / (true_dose + 1e-6)  # Normalize by true dose to avoid division by zero

        # Stronger hypo penalty for overprediction
        hypo_penalty = 0.0
        if predicted_dose > true_dose:
            hypo_penalty = -2.0 * (predicted_dose - true_dose)  # Linear penalty for overprediction
        hypo_penalty = np.clip(hypo_penalty, -2.0, 0.0)

        # Continuous precision bonus for small errors
        precision_bonus = np.exp(-abs(error) / 0.1)  # Exponential decay, peaks at 1.0 when error=0

        reward = base_reward + hypo_penalty + precision_bonus
        reward = float(reward)

        done = True
        truncated = False
        info = {"true_dose": true_dose, "predicted_dose": predicted_dose}
        next_state = self.current_state
        return next_state, reward, done, truncated, info

    def render(self, mode='human'):
        pass

class RewardCallback(BaseCallback):
    def __init__(self, val_env, verbose=0, patience=50):
        super().__init__(verbose)
        self.train_rewards = []
        self.val_rewards = []
        self.val_env = val_env
        self.patience = patience
        self.best_val_reward = -float('inf')
        self.steps_without_improvement = 0

    def _on_step(self):
        self.train_rewards.append(self.locals['rewards'][0])
        if self.num_timesteps % 1000 == 0:
            val_reward = self.evaluate_val()
            self.val_rewards.append(val_reward)
            print(f"Timestep {self.num_timesteps}: Train Reward = {self.train_rewards[-1]:.3f}, Val Reward = {val_reward:.3f}")
            if val_reward > self.best_val_reward:
                self.best_val_reward = val_reward
                self.steps_without_improvement = 0
            else:
                self.steps_without_improvement += 1
            if self.steps_without_improvement >= self.patience:
                print("Early stopping due to no improvement in validation reward.")
                return False
        return True

    def evaluate_val(self):
        total_reward = 0
        for _ in range(10):
            state, _ = self.val_env.reset()
            action, _ = self.model.predict(state, deterministic=True)
            _, reward, _, _, _ = self.val_env.step(action)
            total_reward += reward
        return total_reward / 10

# %% CELL: Main Execution - Preprocess and Filter Low-Dose Data
# Preprocess the full dataset
df_final = preprocess_data(CONFIG["data_path"])

# First split to get the scaler
(_, _, _,
 _, _, _,
 _, _, _,
 _, _, _, _,
 _, _, scaler_y) = split_data(df_final)

# Filter for low-dose data
low_dose_threshold = CONFIG["low_dose_threshold"]
# Convert Polars Series to numpy array properly
dosis = scaler_y.inverse_transform(df_final.select('normal').to_numpy().reshape(-1, 1)).flatten()
low_dose_mask = dosis < low_dose_threshold
df_low_dose = df_final.filter(pl.Series(low_dose_mask))
print(f"Total low-dose samples (dosis < {low_dose_threshold}): {len(df_low_dose)}")

# Remove outliers within low-dose data
dosis_low = scaler_y.inverse_transform(df_low_dose.select('normal').to_numpy().reshape(-1, 1)).flatten()
low_dose_outlier_threshold = np.percentile(dosis_low, 95)
low_dose_outlier_mask = dosis_low < low_dose_outlier_threshold
df_low_dose_clean = df_low_dose.filter(pl.Series(low_dose_outlier_mask))
print(f"Total low-dose samples after outlier removal: {len(df_low_dose_clean)}")

# Split the low-dose data
(X_cgm_train_low, X_cgm_val_low, X_cgm_test_low,
 X_other_train_low, X_other_val_low, X_other_test_low,
 X_subject_train_low, X_subject_val_low, X_subject_test_low,
 y_train_low, y_val_low, y_test_low, subject_test_low,
 scaler_cgm_low, scaler_other_low, scaler_y_low) = split_data(df_low_dose_clean)

# %% CELL: Feature Importance Analysis for Low-Dose Data
# # Feature Importance Analysis for Low-Dose Data
# from sklearn.ensemble import RandomForestRegressor
# import seaborn as sns
# import matplotlib.pyplot as plt

# # Define features based on what's actually in df_low_dose_clean
# cgm_columns = [f'cgm_{i}' for i in range(24)]
# other_features = ['carbInput', 'bgInput', 'insulinOnBoard', 'insulinCarbRatio', 
#                   'insulinSensitivityFactor', 'hour_sin', 'hour_cos', 'cgm_mean', 
#                   'cgm_std', 'cgm_cv', 'cgm_slope', 'tir', 'hypo_risk', 
#                   'carb_insulin_ratio', 'num_recent_boluses', 'total_recent_insulin',
#                   'time_below_70', 'time_above_250', 'cgm_accel', 'cgm_max', 
#                   'time_to_peak', 'time_since_last_bolus', 'is_correction_bolus', 
#                   'carb_rate', 'bg_iob_interaction', 'isf_cgm_slope', 'carb_tir', 
#                   'hba1c_proxy', 'mean_normal', 'std_normal', 'mean_carb']

# # Include one-hot encoded carb_size columns
# carb_size_columns = ['carb_size_small', 'carb_size_medium', 'carb_size_large']

# # Combine all features
# all_features = other_features + cgm_columns + carb_size_columns

# # Extract features and target
# X_low = df_low_dose_clean[all_features]
# y_low = df_low_dose_clean['normal']

# # Train Random Forest
# rf = RandomForestRegressor(n_estimators=100, random_state=42)
# rf.fit(X_low, y_low)

# # Get feature importances
# importances = rf.feature_importances_
# feature_names = all_features

# # Plot feature importances
# plt.figure(figsize=(12, 8))
# sns.barplot(x=importances, y=feature_names)
# plt.title('Feature Importances for Low-Dose Predictions (Updated)')
# plt.xlabel('Importance')
# plt.ylabel('Feature')
# plt.tight_layout()
# timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
# plt.savefig(f'plots/feature_importances_low_dose_updated_{timestamp}.png', bbox_inches='tight')
# plt.show()
# plt.close()

# # Print top 10 features for inspection
# feature_importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': importances})
# feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=False)
# print("Top 10 Features by Importance:")
# print(feature_importance_df.head(10))

# %% CELL: DRL (PPO) Training for Low-Dose Data
train_env_ppo_low = InsulinDoseEnv(X_cgm_train_low, X_other_train_low, y_train_low, scaler_y_low)
val_env_ppo_low = InsulinDoseEnv(X_cgm_val_low, X_other_val_low, y_val_low, scaler_y_low)
callback_low = RewardCallback(val_env=val_env_ppo_low, patience=50)

check_env(train_env_ppo_low)

from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
import torch.nn as nn

class CustomPolicy(BaseFeaturesExtractor):
    def __init__(self, observation_space):
        super(CustomPolicy, self).__init__(observation_space, features_dim=64)
        self.net = nn.Sequential(
            nn.Linear(observation_space.shape[0], 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, self.features_dim),
            nn.ReLU()
        )

    def forward(self, observations):
        return self.net(observations)

model_ppo_low = PPO("MlpPolicy", 
                    train_env_ppo_low, 
                    verbose=1, 
                    learning_rate=1e-4,  # Lower learning rate for stability
                    n_steps=2048,  # Reduce n_steps to update more frequently
                    batch_size=32,  # Smaller batch size for better gradient estimates
                    clip_range=0.1,  # Smaller clip range for more conservative updates
                    ent_coef=0.01,  # Lower entropy coefficient to focus on exploitation
                    policy_kwargs={"net_arch": [128, 128],  # Larger network for better capacity
                                   "ortho_init": False, 
                                   "optimizer_kwargs": {"weight_decay": 1e-4},  # Adjust weight decay
                                   "features_extractor_class": CustomPolicy})

# Train for more timesteps
total_timesteps = 500000  # Increased timesteps
model_ppo_low.learn(total_timesteps=total_timesteps, callback=callback_low)

#%%
# Plot training and validation rewards
plt.plot(callback_low.train_rewards, label='Train Reward')
plt.plot(np.arange(len(callback_low.val_rewards)) * 1000, callback_low.val_rewards, label='Val Reward')
plt.xlabel('Timestep')
plt.ylabel('Reward')
plt.legend()
plt.title('PPO Training vs Validation Reward (Low-Dose)')
timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
plt.savefig(f'plots/ppo_training_validation_reward_low_dose_{timestamp}.png', bbox_inches='tight')
plt.show()
plt.close()

# %%
# Function to generate predictions with PPO
def predict_with_ppo(model, X_cgm, X_other):
    predictions = []
    env = InsulinDoseEnv(X_cgm, X_other, np.zeros(len(X_cgm)), scaler_y_low)
    for i in range(len(X_cgm)):
        cgm_state = X_cgm[i].flatten()
        other_state = X_other[i]
        state = np.concatenate([cgm_state, other_state])
        action, _ = model.predict(state, deterministic=True)
        predicted_dose = scaler_y_low.inverse_transform(action.reshape(-1, 1)).flatten()[0]
        predictions.append(predicted_dose)
    return np.array(predictions)

# Generate predictions with updated model
y_pred_ppo_low_train = predict_with_ppo(model_ppo_low, X_cgm_train_low, X_other_train_low)
y_pred_ppo_low_val = predict_with_ppo(model_ppo_low, X_cgm_val_low, X_other_val_low)
y_pred_ppo_low = predict_with_ppo(model_ppo_low, X_cgm_test_low, X_other_test_low)
y_rule_low = rule_based_prediction(X_other_test_low, scaler_other_low, scaler_y_low)

# Metrics for Low-Dose Model
# Train set
mae_ppo_train = mean_absolute_error(scaler_y_low.inverse_transform(y_train_low.reshape(-1, 1)), y_pred_ppo_low_train)
rmse_ppo_train = np.sqrt(mean_squared_error(scaler_y_low.inverse_transform(y_train_low.reshape(-1, 1)), y_pred_ppo_low_train))
r2_ppo_train = r2_score(scaler_y_low.inverse_transform(y_train_low.reshape(-1, 1)), y_pred_ppo_low_train)
print(f"Updated Low-Dose PPO Train - MAE: {mae_ppo_train:.2f}, RMSE: {rmse_ppo_train:.2f}, R²: {r2_ppo_train:.2f}")

# Validation set
mae_ppo_val = mean_absolute_error(scaler_y_low.inverse_transform(y_val_low.reshape(-1, 1)), y_pred_ppo_low_val)
rmse_ppo_val = np.sqrt(mean_squared_error(scaler_y_low.inverse_transform(y_val_low.reshape(-1, 1)), y_pred_ppo_low_val))
r2_ppo_val = r2_score(scaler_y_low.inverse_transform(y_val_low.reshape(-1, 1)), y_pred_ppo_low_val)
print(f"Updated Low-Dose PPO Val - MAE: {mae_ppo_val:.2f}, RMSE: {rmse_ppo_val:.2f}, R²: {r2_ppo_val:.2f}")

# Test set
mae_ppo = mean_absolute_error(scaler_y_low.inverse_transform(y_test_low.reshape(-1, 1)), y_pred_ppo_low)
rmse_ppo = np.sqrt(mean_squared_error(scaler_y_low.inverse_transform(y_test_low.reshape(-1, 1)), y_pred_ppo_low))
r2_ppo = r2_score(scaler_y_low.inverse_transform(y_test_low.reshape(-1, 1)), y_pred_ppo_low)
print(f"Updated Low-Dose PPO Test - MAE: {mae_ppo:.2f}, RMSE: {rmse_ppo:.2f}, R²: {r2_ppo:.2f}")

# Rule-based model (for comparison)
mae_rule = mean_absolute_error(scaler_y_low.inverse_transform(y_test_low.reshape(-1, 1)), y_rule_low)
rmse_rule = np.sqrt(mean_squared_error(scaler_y_low.inverse_transform(y_test_low.reshape(-1, 1)), y_rule_low))
r2_rule = r2_score(scaler_y_low.inverse_transform(y_test_low.reshape(-1, 1)), y_rule_low)
print(f"Updated Low-Dose Rules Test - MAE: {mae_rule:.2f}, RMSE: {rmse_rule:.2f}, R²: {r2_rule:.2f}")

# %% 
# Analyze test set characteristics
test_df = df_low_dose_clean.filter(pl.col('subject_id').is_in(subject_test_low))
train_df = df_low_dose_clean.filter(pl.col('subject_id').is_in(X_subject_train_low))

print("Test Set Statistics:")
print(test_df.select(['normal', 'carbInput', 'bgInput', 'hba1c_proxy']).describe())
print("\nTrain Set Statistics:")
print(train_df.select(['normal', 'carbInput', 'bgInput', 'hba1c_proxy']).describe())

val_df = df_low_dose_clean.filter(pl.col('subject_id').is_in(X_subject_val_low))
print("Validation Set Statistics:")
print(val_df.select(['normal', 'carbInput', 'bgInput', 'hba1c_proxy']).describe())

# Plot distributions
plt.figure(figsize=(12, 6))
sns.kdeplot(val_df.select('normal').to_numpy().flatten(), label='Validation', color='green')
sns.kdeplot(train_df.select('normal').to_numpy().flatten(), label='Train', color='blue')
sns.kdeplot(test_df.select('normal').to_numpy().flatten(), label='Test', color='red')
plt.title('Distribution of Insulin Doses (Normal)')
plt.xlabel('Insulin Dose (units)')
plt.legend()
plt.show()

# %%
# Visualization
plot_evaluation(y_test_low, y_pred_ppo_low, y_rule_low, subject_test_low, scaler_y_low)



