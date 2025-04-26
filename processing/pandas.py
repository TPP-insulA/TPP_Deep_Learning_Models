# %% CELL: Required Imports
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import os
from joblib import Parallel, delayed
import time
from tqdm import tqdm
from datetime import timedelta

# Global configuration
CONFIG = {
    "batch_size": 128,
    "window_hours": 2,
    "cap_normal": 30,
    "cap_bg": 300,
    "cap_iob": 5,
    "cap_carb": 150,
    "data_path": os.path.join(os.getcwd(), "data", "subjects"),
    "low_dose_threshold": 7.0  # Clinical threshold for low-dose insulin
}

# %% CELL: Data Processing Functions
def get_cgm_window(bolus_time: pd.Timestamp, cgm_df: pd.DataFrame, window_hours: int=CONFIG["window_hours"]) -> np.ndarray:
    """
    Obtiene la ventana de datos CGM para un tiempo de bolo específico.

    Parámetros:
    -----------
    bolus_time : pd.Timestamp
        Tiempo del bolo de insulina
    cgm_df : pd.DataFrame
        DataFrame con datos CGM
    window_hours : int, opcional
        Horas de la ventana de datos (default: 2)

    Retorna:
    --------
    np.ndarray
        Ventana de datos CGM o None si no hay suficientes datos
    """
    window_start = bolus_time - timedelta(hours=window_hours)
    window = cgm_df[(cgm_df['date'] >= window_start) & (cgm_df['date'] <= bolus_time)]
    window = window.sort_values('date').tail(24)
    return window['mg/dl'].values if len(window) >= 24 else None

def calculate_iob(bolus_time: pd.Timestamp, basal_df: pd.DataFrame, half_life_hours: float=4) -> float:
    """
    Calcula la insulina activa en el cuerpo (IOB).

    Parámetros:
    -----------
    bolus_time : pd.Timestamp
        Tiempo del bolo de insulina
    basal_df : pd.DataFrame
        DataFrame con datos de insulina basal
    half_life_hours : float, opcional
        Vida media de la insulina en horas (default: 4)

    Retorna:
    --------
    float
        Cantidad de insulina activa en el organismo
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

def load_subject_data(subject_path: str) -> tuple:
    """
    Carga los datos de un sujeto desde archivos excel.

    Parámetros:
    -----------
    subject_path : str
        Ruta al archivo del sujeto

    Retorna:
    --------
    tuple
        Tupla con (cgm_df, bolus_df, basal_df), donde cada elemento es un DataFrame
        o None si hubo error en la carga
    """
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
        return None, None, None
    
    return cgm_df, bolus_df, basal_df

def prepare_dataframes(cgm_df: pd.DataFrame, bolus_df: pd.DataFrame, basal_df: pd.DataFrame) -> tuple:
    """
    Prepara los DataFrames convirtiendo fechas y ordenando datos.

    Parámetros:
    -----------
    cgm_df : pd.DataFrame
        DataFrame con datos de CGM
    bolus_df : pd.DataFrame
        DataFrame con datos de bolos
    basal_df : pd.DataFrame
        DataFrame con datos basales, puede ser None

    Retorna:
    --------
    tuple
        Tupla con (cgm_df, bolus_df, basal_df) procesados
    """
    cgm_df['date'] = pd.to_datetime(cgm_df['date'])
    cgm_df = cgm_df.sort_values('date')
    bolus_df['date'] = pd.to_datetime(bolus_df['date'])
    if basal_df is not None:
        basal_df['date'] = pd.to_datetime(basal_df['date'])
    return cgm_df, bolus_df, basal_df

def calculate_medians(bolus_df: pd.DataFrame, basal_df: pd.DataFrame) -> tuple:
    """
    Calcula valores medianos para imputación de datos faltantes.

    Parámetros:
    -----------
    bolus_df : pd.DataFrame
        DataFrame con datos de bolos de insulina
    basal_df : pd.DataFrame
        DataFrame con datos de insulina basal

    Retorna:
    --------
    tuple
        Tupla con (carb_median, iob_median) donde:
        - carb_median: mediana de carbohidratos no nulos
        - iob_median: mediana de IOB no nulos
    """
    non_zero_carbs = bolus_df[bolus_df['carbInput'] > 0]['carbInput']
    carb_median = non_zero_carbs.median() if not non_zero_carbs.empty else 10.0

    iob_values = [calculate_iob(row['date'], basal_df) for _, row in bolus_df.iterrows()]
    non_zero_iob = [iob for iob in iob_values if iob > 0]
    iob_median = np.median(non_zero_iob) if non_zero_iob else 0.5
    
    return carb_median, iob_median

def process_bolus_row(row: pd.Series, bolus_time: pd.Timestamp, cgm_window: np.ndarray, 
                     idx: int, carb_median: float, iob_median: float, basal_df: pd.DataFrame) -> dict:
    """
    Procesa una fila de bolo para extraer características.

    Parámetros:
    -----------
    row : pd.Series
        Serie con datos del bolo
    bolus_time : pd.Timestamp
        Tiempo del bolo
    cgm_window : np.ndarray
        Ventana de datos CGM
    idx : int
        Índice del sujeto
    carb_median : float
        Mediana de carbohidratos para imputación
    iob_median : float
        Mediana de IOB para imputación
    basal_df : pd.DataFrame
        DataFrame con datos de insulina basal

    Retorna:
    --------
    dict
        Diccionario con características extraídas
    """
    iob = calculate_iob(bolus_time, basal_df)
    iob = iob_median if iob == 0 else iob
    hour_of_day = bolus_time.hour / 23.0
    
    # Process blood glucose input
    bg_input = row['bgInput'] if pd.notna(row['bgInput']) else cgm_window[-1]
    bg_input = max(bg_input, 50.0)
    bg_input = np.clip(bg_input, 0, CONFIG["cap_bg"])
    
    # Process normal insulin
    normal = row['normal'] if pd.notna(row['normal']) else 0.0
    normal = np.clip(normal, 0, CONFIG["cap_normal"])
    
    # Calculate insulin sensitivity factor
    isf_custom = 50.0 if normal <= 0 else (bg_input - 100) / normal
    isf_custom = np.clip(isf_custom, 10, 100)
    
    # Process carb input
    carb_input = row['carbInput'] if pd.notna(row['carbInput']) else 0.0
    carb_input = carb_median if carb_input == 0 else carb_input
    carb_input = np.clip(carb_input, 0, CONFIG["cap_carb"])
    
    # Process insulin on board
    iob = np.clip(iob, 0, CONFIG["cap_iob"])
    
    # Process insulin carb ratio
    insulin_carb_ratio = np.clip(
        row['insulinCarbRatio'] if pd.notna(row['insulinCarbRatio']) else 10.0, 
        5, 20
    )
    
    return {
        'subject_id': idx,
        'cgm_window': cgm_window,
        'carbInput': carb_input,
        'bgInput': bg_input,
        'insulinCarbRatio': insulin_carb_ratio,
        'insulinSensitivityFactor': isf_custom,
        'insulinOnBoard': iob,
        'hour_of_day': hour_of_day,
        'normal': normal
    }

def process_subject(subject_path: str, idx: int) -> list:
    """
    Procesa los datos de un sujeto para extraer características.

    Parámetros:
    -----------
    subject_path : str
        Ruta al archivo del sujeto
    idx : int
        Índice del sujeto

    Retorna:
    --------
    list
        Lista de diccionarios con características procesadas
    """
    start_time = time.time()
    
    # Load data
    cgm_df, bolus_df, basal_df = load_subject_data(subject_path)
    if cgm_df is None:
        return []
    
    # Prepare dataframes
    cgm_df, bolus_df, basal_df = prepare_dataframes(cgm_df, bolus_df, basal_df)
    
    # Calculate median values
    carb_median, iob_median = calculate_medians(bolus_df, basal_df)

    # Process each bolus event
    processed_data = []
    for _, row in tqdm(bolus_df.iterrows(), total=len(bolus_df), desc=f"Procesando {os.path.basename(subject_path)}", leave=False):
        bolus_time = row['date']
        cgm_window = get_cgm_window(bolus_time, cgm_df)
        if cgm_window is not None:
            features = process_bolus_row(row, bolus_time, cgm_window, idx, carb_median, iob_median, basal_df)
            processed_data.append(features)

    elapsed_time = time.time() - start_time
    print(f"Procesado {os.path.basename(subject_path)} (Sujeto {idx+1}) en {elapsed_time:.2f} segundos")
    return processed_data

def preprocess_data(subject_folder: str) -> pd.DataFrame:
    """
    Preprocesa todos los datos de los sujetos.

    Parámetros:
    -----------
    subject_folder : str
        Carpeta que contiene los archivos de los sujetos

    Retorna:
    --------
    pd.DataFrame
        DataFrame con todos los datos preprocesados
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

def calculate_score_for_subject(df_final: pd.DataFrame, train_subjects: list, 
                               val_subjects: list, test_subjects: list, subject: int) -> tuple:
    """
    Calcula la puntuación de distribución si se agrega un sujeto a cada grupo.

    Parámetros:
    -----------
    df_final : pd.DataFrame
        DataFrame con datos procesados
    train_subjects : list
        Lista de sujetos de entrenamiento
    val_subjects : list
        Lista de sujetos de validación
    test_subjects : list
        Lista de sujetos de prueba
    subject : int
        ID del sujeto a evaluar

    Retorna:
    --------
    tuple
        Puntuaciones (score_if_train, score_if_val, score_if_test) para cada opción
    """
    # Create temporary subject lists
    train_temp = train_subjects + [subject]
    val_temp = val_subjects + [subject]
    test_temp = test_subjects + [subject]
    
    # Calculate stats for current distributions
    train_mean = df_final[df_final['subject_id'].isin(train_subjects)]['normal'].mean() if train_subjects else 0
    val_mean = df_final[df_final['subject_id'].isin(val_subjects)]['normal'].mean() if val_subjects else 0
    test_mean = df_final[df_final['subject_id'].isin(test_subjects)]['normal'].mean() if test_subjects else 0
    train_std = df_final[df_final['subject_id'].isin(train_subjects)]['normal'].std() if train_subjects else 0
    val_std = df_final[df_final['subject_id'].isin(val_subjects)]['normal'].std() if val_subjects else 0
    test_std = df_final[df_final['subject_id'].isin(test_subjects)]['normal'].std() if test_subjects else 0
    
    # Calculate stats for potential new distributions
    train_mean_new = df_final[df_final['subject_id'].isin(train_temp)]['normal'].mean()
    val_mean_new = df_final[df_final['subject_id'].isin(val_temp)]['normal'].mean()
    test_mean_new = df_final[df_final['subject_id'].isin(test_temp)]['normal'].mean()
    train_std_new = df_final[df_final['subject_id'].isin(train_temp)]['normal'].std()
    val_std_new = df_final[df_final['subject_id'].isin(val_temp)]['normal'].std()
    test_std_new = df_final[df_final['subject_id'].isin(test_temp)]['normal'].std()
    
    # Prepare arrays for comparison
    means_if_train = [train_mean_new, val_mean, test_mean]
    means_if_val = [train_mean, val_mean_new, test_mean]
    means_if_test = [train_mean, val_mean, test_mean_new]
    stds_if_train = [train_std_new, val_std, test_std]
    stds_if_val = [train_std, val_std_new, test_std]
    stds_if_test = [train_std, val_std, test_std_new]
    
    # Calculate range scores
    range_means_if_train = max(means_if_train) - min(means_if_train) if all(m != 0 for m in means_if_train) else float('inf')
    range_means_if_val = max(means_if_val) - min(means_if_val) if all(m != 0 for m in means_if_val) else float('inf')
    range_means_if_test = max(means_if_test) - min(means_if_test) if all(m != 0 for m in means_if_test) else float('inf')
    range_stds_if_train = max(stds_if_train) - min(stds_if_train) if all(s != 0 for s in stds_if_train) else float('inf')
    range_stds_if_val = max(stds_if_val) - min(stds_if_val) if all(s != 0 for s in stds_if_val) else float('inf')
    range_stds_if_test = max(stds_if_test) - min(stds_if_test) if all(s != 0 for s in stds_if_test) else float('inf')
    
    # Final combined scores
    score_if_train = range_means_if_train + range_stds_if_train
    score_if_val = range_means_if_val + range_stds_if_val
    score_if_test = range_means_if_test + range_stds_if_test
    
    return score_if_train, score_if_val, score_if_test

def assign_subjects_to_sets(df_final: pd.DataFrame, sorted_subjects: np.ndarray, 
                           train_size: int, val_size: int, test_size: int) -> tuple:
    """
    Asigna sujetos a los conjuntos de entrenamiento, validación y prueba.

    Parámetros:
    -----------
    df_final : pd.DataFrame
        DataFrame con datos procesados
    sorted_subjects : np.ndarray
        Array con IDs de sujetos ordenados
    train_size : int
        Tamaño del conjunto de entrenamiento
    val_size : int
        Tamaño del conjunto de validación
    test_size : int
        Tamaño del conjunto de prueba

    Retorna:
    --------
    tuple
        Tupla con (train_subjects, val_subjects, test_subjects)
    """
    # Initialize with subject 49 in test set if present
    test_subjects = [49] if 49 in sorted_subjects else []
    remaining_subjects = [s for s in sorted_subjects if s != 49]
    train_subjects = []
    val_subjects = []
    
    # Shuffle remaining subjects for randomness
    remaining_subjects_list = list(remaining_subjects)
    rng = np.random.default_rng(seed=42)
    rng.shuffle(remaining_subjects_list)
    
    # Assign each subject to minimize distribution differences
    for subject in remaining_subjects_list:
        score_if_train, score_if_val, score_if_test = calculate_score_for_subject(
            df_final, train_subjects, val_subjects, test_subjects, subject
        )
        
        if len(train_subjects) < train_size and score_if_train <= min(score_if_val, score_if_test):
            train_subjects.append(subject)
        elif len(val_subjects) < val_size and score_if_val <= min(score_if_train, score_if_test):
            val_subjects.append(subject)
        elif len(test_subjects) < test_size:
            test_subjects.append(subject)
        else:
            train_subjects.append(subject)
    
    return train_subjects, val_subjects, test_subjects

def scale_and_prepare_data(df_final: pd.DataFrame, train_mask: pd.Series, 
                          val_mask: pd.Series, test_mask: pd.Series) -> tuple:
    """
    Escala y prepara los datos para entrenamiento de modelos.

    Parámetros:
    -----------
    df_final : pd.DataFrame
        DataFrame con datos procesados
    train_mask : pd.Series
        Máscara booleana para datos de entrenamiento
    val_mask : pd.Series
        Máscara booleana para datos de validación
    test_mask : pd.Series
        Máscara booleana para datos de prueba

    Retorna:
    --------
    tuple
        Tupla con múltiples elementos:
        - x_cgm_train, x_cgm_val, x_cgm_test: datos CGM para cada conjunto
        - x_other_train, x_other_val, x_other_test: otras características para cada conjunto
        - x_subject_train, x_subject_val, x_subject_test: IDs de sujetos para cada conjunto
        - y_train, y_val, y_test: etiquetas para cada conjunto
        - x_subject_test: IDs de sujetos de prueba
        - scaler_cgm, scaler_other, scaler_y: escaladores ajustados
    """
    # Initialize scalers
    scaler_cgm = StandardScaler()
    scaler_other = StandardScaler()
    scaler_y = StandardScaler()
    
    # Define feature columns
    cgm_columns = [f'cgm_{i}' for i in range(24)]
    other_features = ['carbInput', 'bgInput', 'insulinOnBoard', 'insulinCarbRatio', 
                      'insulinSensitivityFactor', 'hour_of_day']
    
    # Scale CGM data
    x_cgm_train = scaler_cgm.fit_transform(df_final.loc[train_mask, cgm_columns]).reshape(-1, 24, 1)
    x_cgm_val = scaler_cgm.transform(df_final.loc[val_mask, cgm_columns]).reshape(-1, 24, 1)
    x_cgm_test = scaler_cgm.transform(df_final.loc[test_mask, cgm_columns]).reshape(-1, 24, 1)
    
    # Scale other features
    x_other_train = scaler_other.fit_transform(df_final.loc[train_mask, other_features])
    x_other_val = scaler_other.transform(df_final.loc[val_mask, other_features])
    x_other_test = scaler_other.transform(df_final.loc[test_mask, other_features])
    
    # Scale target variable
    y_train = scaler_y.fit_transform(df_final.loc[train_mask, 'normal'].values.reshape(-1, 1)).flatten()
    y_val = scaler_y.transform(df_final.loc[val_mask, 'normal'].values.reshape(-1, 1)).flatten()
    y_test = scaler_y.transform(df_final.loc[test_mask, 'normal'].values.reshape(-1, 1)).flatten()
    
    # Get subject IDs
    x_subject_train = df_final.loc[train_mask, 'subject_id'].values
    x_subject_val = df_final.loc[val_mask, 'subject_id'].values
    x_subject_test = df_final.loc[test_mask, 'subject_id'].values
    
    return (x_cgm_train, x_cgm_val, x_cgm_test,
            x_other_train, x_other_val, x_other_test,
            x_subject_train, x_subject_val, x_subject_test,
            y_train, y_val, y_test, x_subject_test,
            scaler_cgm, scaler_other, scaler_y)

def split_data(df_final: pd.DataFrame) -> tuple:
    """
    Divide los datos en conjuntos de entrenamiento, validación y prueba.

    Parámetros:
    -----------
    df_final : pd.DataFrame
        DataFrame con todos los datos preprocesados

    Retorna:
    --------
    tuple
        Tupla con múltiples elementos para los conjuntos de datos escalados
        y los escaladores ajustados para su uso posterior
    """
    start_time = time.time()
    
    # Get subject statistics
    subject_stats = df_final.groupby('subject_id')['normal'].agg(['mean', 'std']).reset_index()
    subject_stats.columns = ['subject_id', 'mean_dose', 'std_dose']
    sorted_subjects = subject_stats.sort_values('mean_dose')['subject_id'].values
    
    # Calculate split sizes
    n_subjects = len(sorted_subjects)
    train_size = int(0.8 * n_subjects)
    val_size = int(0.1 * n_subjects)
    test_size = n_subjects - train_size - val_size
    
    # Assign subjects to sets
    train_subjects, val_subjects, test_subjects = assign_subjects_to_sets(
        df_final, sorted_subjects, train_size, val_size, test_size
    )
    
    # Create masks for the sets
    train_mask = df_final['subject_id'].isin(train_subjects)
    val_mask = df_final['subject_id'].isin(val_subjects)
    test_mask = df_final['subject_id'].isin(test_subjects)
    
    # Print stats about the split
    STD_TEXT = "std ="
    y_train_temp = df_final.loc[train_mask, 'normal']
    y_val_temp = df_final.loc[val_mask, 'normal']
    y_test_temp = df_final.loc[test_mask, 'normal']
    print("Post-split Train y: mean =", y_train_temp.mean(), STD_TEXT, y_train_temp.std())
    print("Post-split Val y: mean =", y_val_temp.mean(), STD_TEXT, y_val_temp.std())
    print("Post-split Test y: mean =", y_test_temp.mean(), STD_TEXT, y_test_temp.std())
    
    # Scale and prepare data
    result = scale_and_prepare_data(df_final, train_mask, val_mask, test_mask)
    
    # Print information about the data shapes
    x_cgm_train, x_cgm_val, x_cgm_test = result[0:3]
    x_other_train, x_other_val, x_other_test = result[3:6]
    x_subject_train, x_subject_val, x_subject_test = result[6:9]
    
    print(f"Entrenamiento CGM: {x_cgm_train.shape}, Validación CGM: {x_cgm_val.shape}, Prueba CGM: {x_cgm_test.shape}")
    print(f"Entrenamiento Otros: {x_other_train.shape}, Validación Otros: {x_other_val.shape}, Prueba Otros: {x_other_test.shape}")
    print(f"Entrenamiento Subject: {x_subject_train.shape}, Validación Subject: {x_subject_val.shape}, Prueba Subject: {x_subject_test.shape}")
    print(f"Sujetos de prueba: {test_subjects}")
    
    elapsed_time = time.time() - start_time
    print(f"División de datos completa en {elapsed_time:.2f} segundos")
    
    return result