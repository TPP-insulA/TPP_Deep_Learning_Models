import polars as pl
import numpy as np
from sklearn.preprocessing import StandardScaler
import sys
import os
from joblib import Parallel, delayed
import time
from datetime import timedelta
from tqdm import tqdm

# Configuración de Matplotlib para evitar errores con Tkinter
import matplotlib
matplotlib.use('Agg')

PROJECT_ROOT = os.path.abspath(os.path.join(os.getcwd(), ".."))
sys.path.append(PROJECT_ROOT) 

# Global configuration
CONFIG = {
    "batch_size": 128,
    "window_hours": 2,
    "cap_normal": 30,
    "cap_bg": 300,
    "cap_iob": 5,
    "cap_carb": 150,
    "data_path": os.path.join(os.getcwd(), "subjects"),
    "low_dose_threshold": 7.0  # Clinical threshold for low-dose insulin
}

def get_cgm_window(bolus_time, cgm_df: pl.DataFrame, window_hours: int=CONFIG["window_hours"]) -> np.ndarray:
    """
    Obtiene la ventana de datos CGM para un tiempo de bolo específico.

    Parámetros:
    -----------
    bolus_time : datetime
        Tiempo del bolo de insulina
    cgm_df : pl.DataFrame
        DataFrame con datos CGM
    window_hours : int, opcional
        Horas de la ventana de datos (default: 2)

    Retorna:
    --------
    np.ndarray
        Ventana de datos CGM o None si no hay suficientes datos
    """
    window_start = bolus_time - timedelta(hours=window_hours)
    window = cgm_df.filter(
        (pl.col("date") >= window_start) & (pl.col("date") <= bolus_time)
    ).sort("date").tail(24)
    
    if window.height < 24:
        return None
    return window.get_column("mg/dl").to_numpy()

def calculate_iob(bolus_time, basal_df: pl.DataFrame, half_life_hours: float = 4.0) -> float:
    """
    Calcula la insulina activa en el cuerpo (IOB).

    Parámetros:
    -----------
    bolus_time : datetime
        Tiempo del bolo de insulina
    basal_df : pl.DataFrame
        DataFrame con datos de insulina basal
    half_life_hours : float, opcional
        Vida media de la insulina en horas (default: 4.0)

    Retorna:
    --------
    float
        Cantidad de insulina activa en el organismo
    """
    if basal_df is None or basal_df.is_empty():
        return 0.0
    
    iob = 0.0
    for row in basal_df.iter_rows(named=True):
        start_time = row["date"]
        duration_hours = row["duration"] / (1000 * 3600)
        end_time = start_time + timedelta(hours=duration_hours)
        rate = row["rate"] if row["rate"] is not None else 0.9
        rate = min(rate, 2.0)
        
        if start_time <= bolus_time <= end_time:
            time_since_start = (bolus_time - start_time).total_seconds() / 3600
            remaining = rate * (1 - (time_since_start / half_life_hours))
            iob += max(0.0, remaining)
    
    return min(iob, CONFIG["cap_iob"])

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
        cgm_df = pl.read_excel(subject_path, sheet_name="CGM")
        bolus_df = pl.read_excel(subject_path, sheet_name="Bolus")
        try:
            basal_df = pl.read_excel(subject_path, sheet_name="Basal")
        except Exception:
            basal_df = None
            
        # Conversión de fechas
        cgm_df = cgm_df.with_columns(pl.col("date").cast(pl.Datetime))
        cgm_df = cgm_df.sort("date")
        bolus_df = bolus_df.with_columns(pl.col("date").cast(pl.Datetime))
        if basal_df is not None:
            basal_df = basal_df.with_columns(pl.col("date").cast(pl.Datetime))
            
        return cgm_df, bolus_df, basal_df
    except Exception as e:
        print(f"Error al cargar {os.path.basename(subject_path)}: {e}")
        return None, None, None

def calculate_medians(bolus_df: pl.DataFrame, basal_df: pl.DataFrame) -> tuple:
    """
    Calcula valores medianos para imputación de datos faltantes.

    Parámetros:
    -----------
    bolus_df : pl.DataFrame
        DataFrame con datos de bolos de insulina
    basal_df : pl.DataFrame
        DataFrame con datos de insulina basal

    Retorna:
    --------
    tuple
        Tupla con (carb_median, iob_median) donde:
        - carb_median: mediana de carbohidratos no nulos
        - iob_median: mediana de IOB no nulos
    """
    # Carb median
    non_zero_carbs = bolus_df.filter(pl.col("carbInput") > 0).get_column("carbInput")
    carb_median = non_zero_carbs.median() if len(non_zero_carbs) > 0 else 10.0
    
    # IOB median
    iob_values = []
    for row in bolus_df.iter_rows(named=True):
        iob = calculate_iob(row['date'], basal_df)
        iob_values.append(iob)
    
    non_zero_iob = [iob for iob in iob_values if iob > 0]
    iob_median = np.median(non_zero_iob) if non_zero_iob else 0.5
    
    return carb_median, iob_median

def extract_features(row: dict, cgm_window: np.ndarray, carb_median: float, 
                    iob_median: float, basal_df: pl.DataFrame, idx: int) -> dict | None:
    """
    Extrae características para una instancia de bolo individual.

    Parámetros:
    -----------
    row : dict
        Fila con datos del bolo
    cgm_window : np.ndarray
        Ventana de datos CGM
    carb_median : float
        Valor mediano de carbohidratos para imputación
    iob_median : float
        Valor mediano de IOB para imputación
    basal_df : pl.DataFrame
        DataFrame con datos de insulina basal
    idx : int
        Índice del sujeto

    Retorna:
    --------
    dict
        Diccionario con características extraídas o None si no hay datos suficientes
    """
    bolus_time = row["date"]
    if cgm_window is None:
        return None
    
    # Calculate IOB
    iob = calculate_iob(bolus_time, basal_df)
    iob = iob_median if iob == 0 else iob
    iob = np.clip(iob, 0, CONFIG["cap_iob"])
    
    # Time features
    hour_of_day = bolus_time.hour / 23.0
    
    # BG features
    bg_input = row["bgInput"] if row["bgInput"] is not None else cgm_window[-1]
    bg_input = max(bg_input, 50.0)
    bg_input = np.clip(bg_input, 0, CONFIG["cap_bg"])
    
    # Insulin features
    normal = row["normal"] if row["normal"] is not None else 0.0
    normal = np.clip(normal, 0, CONFIG["cap_normal"])
    
    # Calculate custom insulin sensitivity factor
    isf_custom = 50.0 if normal <= 0 else (bg_input - 100) / normal
    isf_custom = np.clip(isf_custom, 10, 100)
    
    # Carb features
    carb_input = row["carbInput"] if row["carbInput"] is not None else 0.0
    carb_input = carb_median if carb_input == 0 else carb_input
    carb_input = np.clip(carb_input, 0, CONFIG["cap_carb"])
    
    insulin_carb_ratio = row["insulinCarbRatio"] if row["insulinCarbRatio"] is not None else 10.0
    insulin_carb_ratio = np.clip(insulin_carb_ratio, 5, 20)
    
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
    Procesa los datos de un sujeto.

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
    print(f"Procesando {os.path.basename(subject_path)} (Sujeto {idx+1})...")
    
    # Load data
    cgm_df, bolus_df, basal_df = load_subject_data(subject_path)
    if cgm_df is None or bolus_df is None:
        return []

    # Calculate medians for imputation
    carb_median, iob_median = calculate_medians(bolus_df, basal_df)

    # Process each bolus row
    processed_data = []
    for row in tqdm(bolus_df.iter_rows(named=True), total=len(bolus_df), 
                    desc=f"Procesando {os.path.basename(subject_path)}", leave=False):
        bolus_time = row["date"]
        cgm_window = get_cgm_window(bolus_time, cgm_df)
        features = extract_features(row, cgm_window, carb_median, iob_median, basal_df, idx)
        if features is not None:
            processed_data.append(features)

    elapsed_time = time.time() - start_time
    print(f"Procesado {os.path.basename(subject_path)} (Sujeto {idx+1}) en {elapsed_time:.2f} segundos")
    return processed_data

def preprocess_data(subject_folder: str) -> pl.DataFrame:
    """
    Preprocesa todos los datos de los sujetos.

    Parámetros:
    -----------
    subject_folder : str
        Carpeta que contiene los archivos de los sujetos

    Retorna:
    --------
    pl.DataFrame
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

    df_processed = pl.DataFrame(all_processed_data)
    print("Muestra de datos procesados combinados:")
    print(df_processed.head())
    print(f"Total de muestras: {len(df_processed)}")

    # Aplicar transformaciones logarítmicas (np.log1p) como en pandas.py
    df_processed = df_processed.with_columns([
        pl.col("normal").log1p().alias("normal"),
        pl.col("carbInput").log1p().alias("carbInput"),
        pl.col("insulinOnBoard").log1p().alias("insulinOnBoard"),
        pl.col("bgInput").log1p().alias("bgInput")
    ])
    
    # Para cgm_window, necesitamos extraer, transformar y volver a empaquetar
    # Extraer los arrays como numpy
    cgm_windows = df_processed["cgm_window"].to_numpy()
    # Transformar con NumPy
    transformed_windows = np.array([np.log1p(window) for window in cgm_windows])
    # Reemplazar en el DataFrame
    df_processed = df_processed.with_columns(
        pl.lit(transformed_windows).alias("cgm_window")
    )

    # Creación de columnas para las ventanas CGM
    cgm_columns = [f'cgm_{i}' for i in range(24)]
    cgm_data = np.array([row['cgm_window'] for row in df_processed.to_dicts()])
    
    # Crear DataFrame separado para CGM y luego unirlo
    df_cgm = pl.DataFrame({col: cgm_data[:, i] for i, col in enumerate(cgm_columns)})
    df_final = pl.concat([
        df_cgm, 
        df_processed.drop("cgm_window")
    ], how="horizontal")
    
    # Verificar valores nulos
    print("Verificación de NaN en df_final:")
    null_counts = df_final.null_count()
    print(null_counts)
    df_final = df_final.drop_nulls()

    elapsed_time = time.time() - start_time
    print(f"Preprocesamiento completo en {elapsed_time:.2f} segundos")
    return df_final

def calculate_stats_for_group(df_final_pd: pl.DataFrame, subjects: list, feature: str='normal') -> tuple:
    """
    Calcula media y desviación estándar para un grupo de sujetos.

    Parámetros:
    -----------
    df_final_pd : pl.DataFrame
        DataFrame con datos procesados
    subjects : list
        Lista de IDs de sujetos
    feature : str, opcional
        Característica para calcular estadísticas (default: 'normal')

    Retorna:
    --------
    tuple
        Tupla con (media, desviación estándar)
    """
    if not subjects:
        return 0, 0
    mask = df_final_pd['subject_id'].isin(subjects)
    values = df_final_pd.loc[mask, feature]
    return values.mean(), values.std()

def calculate_distribution_score(means: list, stds: list) -> float:
    """
    Calcula una puntuación de distribución basada en medias y desviaciones estándar.

    Parámetros:
    -----------
    means : list
        Lista de valores medios
    stds : list
        Lista de desviaciones estándar

    Retorna:
    --------
    float
        Puntuación que representa la variabilidad de la distribución
    """
    if not all(m != 0 for m in means):
        return float('inf')
    
    range_means = max(means) - min(means)
    range_stds = max(stds) - min(stds) if all(s != 0 for s in stds) else float('inf')
    return range_means + range_stds

def assign_subject_to_group(df_final_pd: pl.DataFrame, subject: int, 
                           train_subjects: list, val_subjects: list, test_subjects: list,
                           train_size: int, val_size: int, test_size: int) -> tuple:
    """
    Asigna un sujeto a un grupo de entrenamiento, validación o prueba basado en balance.

    Parámetros:
    -----------
    df_final_pd : pl.DataFrame
        DataFrame con datos procesados
    subject : int
        ID del sujeto a asignar
    train_subjects : list
        Lista actual de sujetos de entrenamiento
    val_subjects : list
        Lista actual de sujetos de validación
    test_subjects : list
        Lista actual de sujetos de prueba
    train_size : int
        Tamaño máximo del grupo de entrenamiento
    val_size : int
        Tamaño máximo del grupo de validación
    test_size : int
        Tamaño máximo del grupo de prueba

    Retorna:
    --------
    tuple
        Tupla con listas actualizadas (train_subjects, val_subjects, test_subjects)
    """
    # Calculate current stats
    train_mean, train_std = calculate_stats_for_group(df_final_pd, train_subjects)
    val_mean, val_std = calculate_stats_for_group(df_final_pd, val_subjects)
    test_mean, test_std = calculate_stats_for_group(df_final_pd, test_subjects)
    
    # Calculate potential stats if subject is added to each group
    train_temp = train_subjects + [subject]
    val_temp = val_subjects + [subject]
    test_temp = test_subjects + [subject]
    
    train_mean_new, train_std_new = calculate_stats_for_group(df_final_pd, train_temp)
    val_mean_new, val_std_new = calculate_stats_for_group(df_final_pd, val_temp)
    test_mean_new, test_std_new = calculate_stats_for_group(df_final_pd, test_temp)
    
    # Calculate scores for each option
    score_if_train = calculate_distribution_score(
        [train_mean_new, val_mean, test_mean], 
        [train_std_new, val_std, test_std]
    )
    score_if_val = calculate_distribution_score(
        [train_mean, val_mean_new, test_mean], 
        [train_std, val_std_new, test_std]
    )
    score_if_test = calculate_distribution_score(
        [train_mean, val_mean, test_mean_new], 
        [train_std, val_std, test_std_new]
    )
    
    # Assign to the group with best balance
    if len(train_subjects) < train_size and score_if_train <= min(score_if_val, score_if_test):
        train_subjects.append(subject)
    elif len(val_subjects) < val_size and score_if_val <= min(score_if_train, score_if_test):
        val_subjects.append(subject)
    elif len(test_subjects) < test_size:
        test_subjects.append(subject)
    else:
        train_subjects.append(subject)
    
    return train_subjects, val_subjects, test_subjects

def prepare_data_with_scaler(df_final_pd: pl.DataFrame, mask: pl.Series, 
                            columns: list, scaler: StandardScaler, reshape: tuple=None) -> np.ndarray:
    """
    Prepara datos con transformación StandardScaler.

    Parámetros:
    -----------
    df_final_pd : pl.DataFrame
        DataFrame con datos procesados
    mask : pd.Series
        Máscara booleana para seleccionar filas
    columns : list
        Lista de columnas para seleccionar
    scaler : StandardScaler
        Escalador ajustado previamente
    reshape : tuple, opcional
        Nueva forma para los datos transformados (default: None)

    Retorna:
    --------
    np.ndarray
        Array con datos transformados y opcionalmente reshapeados
    """
    data = scaler.transform(df_final_pd.loc[mask, columns])
    if reshape:
        data = data.reshape(*reshape)
    return data

def split_data(df_final: pl.DataFrame) -> tuple:
    """
    Divide los datos siguiendo una estrategia para asegurar distribuciones 
    equilibradas entre los conjuntos de entrenamiento, validación y prueba.

    Parámetros:
    -----------
    df_final : pl.DataFrame
        DataFrame con todos los datos preprocesados

    Retorna:
    --------
    tuple
        Tupla con múltiples elementos:
        - x_cgm_train, x_cgm_val, x_cgm_test: datos CGM para cada conjunto
        - x_other_train, x_other_val, x_other_test: otras características para cada conjunto
        - x_subject_train, x_subject_val, x_subject_test: IDs de sujetos para cada conjunto
        - y_train, y_val, y_test: etiquetas para cada conjunto
        - subject_test: IDs de sujetos de prueba
        - scaler_cgm, scaler_other, scaler_y: escaladores ajustados
    """
    start_time = time.time()
    
    # Estadísticas por sujeto
    subject_stats = df_final.group_by("subject_id").agg([
        pl.col("normal").mean().alias("mean_dose"),
        pl.col("normal").std().alias("std_dose")
    ])
    
    # Obtener lista de sujetos ordenados por dosis media
    sorted_subjects = subject_stats.sort("mean_dose").get_column("subject_id").to_list()
    n_subjects = len(sorted_subjects)
    train_size = int(0.8 * n_subjects)
    val_size = int(0.1 * n_subjects)
    test_size = n_subjects - train_size - val_size

    # Iniciar con sujeto específico para pruebas si está disponible
    test_subjects = [49] if 49 in sorted_subjects else []
    remaining_subjects = [s for s in sorted_subjects if s != 49]
    train_subjects = []
    val_subjects = []

    # Aleatorizar la lista restante y convertir a pandas para cálculos
    rng = np.random.default_rng(seed=42)
    rng.shuffle(remaining_subjects)
    df_final_pd = df_final.to_pandas()

    # Distribuir sujetos entre los grupos
    for subject in remaining_subjects:
        train_subjects, val_subjects, test_subjects = assign_subject_to_group(
            df_final_pd, subject, train_subjects, val_subjects, test_subjects,
            train_size, val_size, test_size
        )

    # Crear máscaras para división de datos
    train_mask = df_final_pd['subject_id'].isin(train_subjects)
    val_mask = df_final_pd['subject_id'].isin(val_subjects)
    test_mask = df_final_pd['subject_id'].isin(test_subjects)

    # Mostrar estadísticas post-división
    for set_name, mask in [("Train", train_mask), ("Val", val_mask), ("Test", test_mask)]:
        y_temp = df_final_pd.loc[mask, 'normal']
        print(f"Post-split {set_name} y: mean = {y_temp.mean()}, std = {y_temp.std()}")

    # Definir columnas para diferentes grupos de características
    cgm_columns = [f'cgm_{i}' for i in range(24)]
    other_features = ['carbInput', 'bgInput', 'insulinOnBoard', 'insulinCarbRatio', 
                      'insulinSensitivityFactor', 'hour_of_day']

    # Inicializar escaladores
    scaler_cgm = StandardScaler().fit(df_final_pd.loc[train_mask, cgm_columns])
    scaler_other = StandardScaler().fit(df_final_pd.loc[train_mask, other_features])
    scaler_y = StandardScaler().fit(df_final_pd.loc[train_mask, 'normal'].values.reshape(-1, 1))

    # Preparar datos CGM
    x_cgm_train = prepare_data_with_scaler(df_final_pd, train_mask, cgm_columns, scaler_cgm, reshape=(-1, 24, 1))
    x_cgm_val = prepare_data_with_scaler(df_final_pd, val_mask, cgm_columns, scaler_cgm, reshape=(-1, 24, 1))
    x_cgm_test = prepare_data_with_scaler(df_final_pd, test_mask, cgm_columns, scaler_cgm, reshape=(-1, 24, 1))
    
    # Preparar otras características
    x_other_train = prepare_data_with_scaler(df_final_pd, train_mask, other_features, scaler_other)
    x_other_val = prepare_data_with_scaler(df_final_pd, val_mask, other_features, scaler_other)
    x_other_test = prepare_data_with_scaler(df_final_pd, test_mask, other_features, scaler_other)
    
    # Preparar etiquetas
    y_train = scaler_y.transform(df_final_pd.loc[train_mask, 'normal'].values.reshape(-1, 1)).flatten()
    y_val = scaler_y.transform(df_final_pd.loc[val_mask, 'normal'].values.reshape(-1, 1)).flatten()
    y_test = scaler_y.transform(df_final_pd.loc[test_mask, 'normal'].values.reshape(-1, 1)).flatten()

    # Obtener IDs de sujeto
    x_subject_train = df_final_pd.loc[train_mask, 'subject_id'].values
    x_subject_val = df_final_pd.loc[val_mask, 'subject_id'].values
    x_subject_test = df_final_pd.loc[test_mask, 'subject_id'].values
    
    # Imprimir resumen
    print(f"Entrenamiento CGM: {x_cgm_train.shape}, Validación CGM: {x_cgm_val.shape}, Prueba CGM: {x_cgm_test.shape}")
    print(f"Entrenamiento Otros: {x_other_train.shape}, Validación Otros: {x_other_val.shape}, Prueba Otros: {x_other_test.shape}")
    print(f"Entrenamiento Subject: {x_subject_train.shape}, Validación Subject: {x_subject_val.shape}, Prueba Subject: {x_subject_test.shape}")
    print(f"Sujetos de prueba: {test_subjects}")

    elapsed_time = time.time() - start_time
    print(f"División de datos completa en {elapsed_time:.2f} segundos")
    
    return (x_cgm_train, x_cgm_val, x_cgm_test,
            x_other_train, x_other_val, x_other_test,
            x_subject_train, x_subject_val, x_subject_test,
            y_train, y_val, y_test, x_subject_test,
            scaler_cgm, scaler_other, scaler_y)