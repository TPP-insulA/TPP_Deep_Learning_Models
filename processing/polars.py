import polars as pl
import numpy as np
from sklearn.preprocessing import StandardScaler
import sys
import os
from joblib import Parallel, delayed
import time
from datetime import timedelta
from tqdm import tqdm
import xml.etree.ElementTree as ET
import glob

# Configuración de Matplotlib para evitar errores con Tkinter
import matplotlib
matplotlib.use('Agg')

PROJECT_ROOT = os.path.abspath(os.path.join(os.getcwd(), ".."))
sys.path.append(PROJECT_ROOT) 

from constants.constants import CONST_DEFAULT_SEED

# Global configuration
CONFIG = {
    "batch_size": 128,
    "window_hours": 2,
    "cap_normal": 30,
    "cap_bg": 300,
    "cap_iob": 5,
    "cap_carb": 150,
    "cap_exercise": 3,
    "cap_sleep": 10,
    "cap_stress": 3,
    "data_path": os.path.join(os.getcwd(), "subjects"),
    "ohio_data_path": os.path.join(PROJECT_ROOT, "data", "OhioT1DM"),
    "low_dose_threshold": 7.0,  # Clinical threshold for low-dose insulin
    "use_ohio_data": True
}

# Constantes para columnas comunes
COL_DATE = "date"
COL_TS = "ts"
COL_TIMESTAMP = "timestamp"
COL_VALUE = "value"
COL_MGDL = "mg/dl"
COL_PATIENT_ID = "patient_id"
COL_SUBJECT_ID = "subject_id"
COL_TENDENCY = "tendency"
COL_NORMAL = "normal"

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

def parse_patient_xml(file_path: str) -> dict:
    """
    Analiza un archivo XML y extrae los datos del paciente.

    Parámetros:
    -----------
    file_path : str
        Ruta al archivo XML

    Retorna:
    --------
    dict
        Diccionario con los datos del paciente extraídos
    """
    tree = ET.parse(file_path)
    root = tree.getroot()
    
    # Extraer metadatos del paciente
    patient_id = root.get('id')
    
    # Definir características a extraer
    features = ['glucose_level', 'finger_stick', 'basal', 'temp_basal', 'bolus', 'meal', 
                'sleep', 'work', 'stressors', 'hypo_event', 'illness', 'exercise', 
                'basis_heart_rate', 'basis_gsr', 'basis_skin_temperature', 'basis_sleep', 
                'acceleration']
    
    # Diccionario para almacenar DataFrames por característica
    feature_dfs = {}
    
    # Extraer datos para cada característica
    for feature in features:
        feature_data = []
        feature_element = root.find(feature)
        if feature_element is not None:
            for event in feature_element.findall('event'):
                # Extraer todos los atributos de este evento
                event_data = {k: v for k, v in event.attrib.items()}
                # Añadir patient_id
                event_data[COL_PATIENT_ID] = patient_id
                feature_data.append(event_data)
        
        # Crear DataFrame si hay datos
        if feature_data:
            df = pl.DataFrame(feature_data)
            
            # Convertir columnas de timestamp a datetime
            timestamp_cols = [col for col in df.columns if 'ts' in col]
            for col in timestamp_cols:
                df = df.with_columns(
                    pl.col(col).str.strptime(
                        pl.Datetime, 
                        format="%d-%m-%Y %H:%M:%S",
                        strict=False
                    )
                )
            
            # Convertir columnas numéricas
            numeric_cols = [COL_VALUE, 'dose', 'carbs', 'duration']
            for col in [c for c in numeric_cols if c in df.columns]:
                df = df.with_columns(
                    pl.col(col).cast(pl.Float64, strict=False)
                )
                
            feature_dfs[feature] = df
        else:
            # Crear DataFrame vacío con columna patient_id
            feature_dfs[feature] = pl.DataFrame(schema={
                COL_PATIENT_ID: pl.Utf8
            })
    
    # Para glucose_level específicamente, renombrar columnas para compatibilidad
    if 'glucose_level' in feature_dfs and COL_TS in feature_dfs['glucose_level'].columns:
        feature_dfs['glucose'] = feature_dfs['glucose_level'].rename({
            COL_TS: COL_TIMESTAMP, 
            COL_VALUE: COL_MGDL
        })
    
    return {
        'patient_info': {'id': patient_id},
        **feature_dfs
    }

def load_ohio_data() -> dict:
    """
    Carga y organiza el dataset OhioT1DM.

    Retorna:
    --------
    dict
        Diccionario con datos de pacientes de OhioT1DM
    """
    print("Cargando dataset OhioT1DM...")
    
    # Rutas al dataset Ohio
    base_paths = [
        os.path.join(CONFIG["ohio_data_path"], "2020", "train"),
        os.path.join(CONFIG["ohio_data_path"], "2018", "train"),
        os.path.join(CONFIG["ohio_data_path"], "2020", "test"),
        os.path.join(CONFIG["ohio_data_path"], "2018", "test")
    ]
    
    # Diccionario para almacenar datos de pacientes
    patient_data = {}
    
    # Procesar archivos XML
    for base_path in base_paths:
        for xml_file in glob.glob(os.path.join(base_path, "*.xml")):
            print(f"  Procesando {os.path.basename(xml_file)}")
            data = parse_patient_xml(xml_file)
            patient_id = f"ohio_{data['patient_info']['id']}"  # Prefijo "ohio_" para distinción
            
            if patient_id in patient_data:
                # Combinar datos si el paciente ya existe
                for feature in data:
                    if feature == 'patient_info':
                        continue
                    
                    if feature in patient_data[patient_id] and isinstance(data[feature], pl.DataFrame) and len(data[feature]) > 0:
                        patient_data[patient_id][feature] = pl.concat([
                            patient_data[patient_id][feature],
                            data[feature]
                        ])
                    else:
                        patient_data[patient_id][feature] = data[feature]
            else:
                patient_data[patient_id] = data
    
    print(f"Datos cargados para {len(patient_data)} pacientes del dataset OhioT1DM")
    return patient_data

def classify_patient_tendency(cgm_df: pl.DataFrame) -> str:
    """
    Analiza si un paciente tiende a hiperglucemia o hipoglucemia.
    
    Parámetros:
    -----------
    cgm_df : pl.DataFrame
        DataFrame con lecturas CGM incluyendo columna 'mg/dl' o 'value'
    
    Retorna:
    --------
    str
        Clasificación: 'hyperglycemic', 'hypoglycemic', o 'mixed'
    """
    if cgm_df.height == 0:
        return "unknown"
    
    # Determinar el nombre de columna para valores de glucosa
    value_col = COL_MGDL if COL_MGDL in cgm_df.columns else COL_VALUE
    
    # Calcular tiempo en diferentes rangos
    hypo_time = (cgm_df[value_col] < 70).mean() * 100
    normal_time = ((cgm_df[value_col] >= 70) & (cgm_df[value_col] <= 180)).mean() * 100
    hyper_time = (cgm_df[value_col] > 180).mean() * 100
    
    # Contar eventos (lecturas consecutivas bajo/sobre umbral)
    # Ordenar por columna de timestamp
    time_col = COL_TIMESTAMP if COL_TIMESTAMP in cgm_df.columns else COL_TS
    if time_col not in cgm_df.columns:
        time_col = COL_DATE
    
    glucose_vals = cgm_df.sort(time_col)
    
    prev_type = None
    hypo_events = 0
    hyper_events = 0
    
    for row in glucose_vals.iter_rows(named=True):
        if row[value_col] < 70:  # Hipoglucemia
            if prev_type != 'hypo':
                hypo_events += 1
            prev_type = 'hypo'
        elif row[value_col] > 180:  # Hiperglucemia
            if prev_type != 'hyper':
                hyper_events += 1
            prev_type = 'hyper'
        else:  # Rango normal
            prev_type = 'normal'
    
    # Determinar tendencia general
    if hypo_events > hyper_events and hypo_time > hyper_time:
        tendency = "hypoglycemic"
    elif hyper_events > hypo_events and hyper_time > hypo_time:
        tendency = "hyperglycemic"
    else:
        tendency = "mixed"
    
    return tendency

def get_sleep_quality(patient_data: dict, date) -> float:
    """
    Extrae métricas de calidad del sueño alrededor de una fecha específica.

    Parámetros:
    -----------
    patient_data : dict
        Diccionario con datos del paciente incluyendo información de sueño
    date : datetime
        Fecha de referencia para los datos de sueño

    Retorna:
    --------
    float
        Métrica de calidad del sueño (escala 0-10, donde 10 es mejor)
    """
    sleep_quality = 5.0  # Valor intermedio por defecto
    
    # Verificar fuentes de datos de sueño
    sleep_sources = ['basis_sleep', 'sleep']
    
    for source in sleep_sources:
        if source in patient_data and len(patient_data[source]) > 0:
            sleep_df = patient_data[source]
            
            # Determinar nombres de columnas de timestamp
            start_col = 'tbegin' if 'tbegin' in sleep_df.columns else 'ts_begin'
            end_col = 'tend' if 'tend' in sleep_df.columns else 'ts_end'
            
            if start_col not in sleep_df.columns or end_col not in sleep_df.columns:
                continue
                
            # Buscar episodios de sueño que ocurrieron antes de la fecha
            # (dentro de 24 horas es una ventana razonable)
            time_window = date - timedelta(hours=24)
            
            recent_sleep = sleep_df.filter(
                (pl.col(start_col) >= time_window) & (pl.col(start_col) <= date)
            )
            
            if len(recent_sleep) > 0:
                # Si hay una columna de calidad, usarla
                if 'quality' in recent_sleep.columns:
                    qualities = recent_sleep['quality'].to_list()
                    try:
                        # Convertir a numérico y tomar promedio
                        qualities = [float(q) for q in qualities if q is not None]
                        if qualities:
                            sleep_quality = sum(qualities) / len(qualities)
                    except (ValueError, TypeError):
                        pass
                
                # De lo contrario calcular basado en duración
                elif start_col in recent_sleep.columns and end_col in recent_sleep.columns:
                    try:
                        # Calcular duraciones en horas
                        durations = []
                        for row in recent_sleep.iter_rows(named=True):
                            start = row[start_col]
                            end = row[end_col]
                            duration_hours = (end - start).total_seconds() / 3600
                            if 0 < duration_hours <= 12:  # Rango razonable para duración de sueño
                                durations.append(duration_hours)
                        
                        if durations:
                            # Convertir duración a calidad (7-9 horas es ideal)
                            avg_duration = sum(durations) / len(durations)
                            if 6.5 <= avg_duration <= 9:
                                sleep_quality = 10.0
                            elif avg_duration < 4 or avg_duration > 11:
                                sleep_quality = 2.0
                            else:
                                # Escalar entre 5-9 basado en qué tan cercano al rango ideal
                                deviation = min(abs(avg_duration - 7.5), 3)
                                sleep_quality = 9.5 - (deviation * 1.5)
                    except Exception:
                        pass
    
    return np.clip(sleep_quality, 0, CONFIG["cap_sleep"])

def get_exercise_level(patient_data: dict, date) -> float:
    """
    Extrae métricas de nivel de ejercicio alrededor de una fecha específica.

    Parámetros:
    -----------
    patient_data : dict
        Diccionario con datos del paciente incluyendo información de ejercicio
    date : datetime
        Fecha de referencia para los datos de ejercicio

    Retorna:
    --------
    float
        Métrica de nivel de ejercicio (escala 0-3, donde 3 es intenso)
    """
    exercise_level = 0.0  # Por defecto sin ejercicio
    
    if 'exercise' in patient_data and len(patient_data['exercise']) > 0:
        exercise_df = patient_data['exercise']
        
        # Determinar nombre de columna de timestamp
        time_col = COL_TS if COL_TS in exercise_df.columns else COL_TIMESTAMP
        
        if time_col not in exercise_df.columns:
            return exercise_level
            
        # Buscar episodios de ejercicio que ocurrieron antes de la fecha
        # (12 horas es una ventana razonable para impacto en glucosa)
        time_window = date - timedelta(hours=12)
        
        recent_exercise = exercise_df.filter(
            (pl.col(time_col) >= time_window) & (pl.col(time_col) <= date)
        )
        
        if len(recent_exercise) > 0:
            # Si hay columna de intensidad, usarla
            if 'intensity' in recent_exercise.columns:
                intensities = [float(i) if i is not None else 0.0 for i in recent_exercise['intensity'].to_list()]
                if intensities:
                    exercise_level = max(intensities)  # Tomar el ejercicio más intenso
            
            # Si hay duración, usarla para estimar intensidad
            elif 'duration' in recent_exercise.columns:
                durations = [float(d) if d is not None else 0.0 for d in recent_exercise['duration'].to_list()]
                if durations:
                    total_duration = sum(durations)
                    # Mapear duración a nivel: > 60 min = 3, > 30 min = 2, > 0 min = 1
                    if total_duration > 60:
                        exercise_level = 3.0
                    elif total_duration > 30:
                        exercise_level = 2.0
                    elif total_duration > 0:
                        exercise_level = 1.0
            
            # Si hay entradas pero sin detalles, asumir ejercicio moderado
            else:
                exercise_level = 1.5
    
    return np.clip(exercise_level, 0, CONFIG["cap_exercise"])

def get_stress_level(patient_data: dict, date) -> float:
    """
    Extrae métricas de nivel de estrés alrededor de una fecha específica.

    Parámetros:
    -----------
    patient_data : dict
        Diccionario con datos del paciente incluyendo información de estrés
    date : datetime
        Fecha de referencia para los datos de estrés

    Retorna:
    --------
    float
        Métrica de nivel de estrés (escala 0-3, donde 3 es máximo estrés)
    """
    stress_level = 0.0  # Por defecto sin estrés
    
    if 'stressors' in patient_data and len(patient_data['stressors']) > 0:
        stress_df = patient_data['stressors']
        
        # Determinar nombre de columna de timestamp
        time_col = COL_TS if COL_TS in stress_df.columns else COL_TIMESTAMP
        
        if time_col not in stress_df.columns:
            return stress_level
            
        # Buscar eventos de estrés que ocurrieron antes de la fecha
        # (6 horas es una ventana razonable para impacto significativo)
        time_window = date - timedelta(hours=6)
        
        recent_stress = stress_df.filter(
            (pl.col(time_col) >= time_window) & (pl.col(time_col) <= date)
        )
        
        if len(recent_stress) > 0:
            # Si hay columna de nivel, usarla
            if 'level' in recent_stress.columns:
                levels = [float(l) if l is not None else 0.0 for l in recent_stress['level'].to_list()]
                if levels:
                    stress_level = max(levels)  # Tomar el nivel más alto de estrés
            
            # De lo contrario basar en cantidad de estresores
            else:
                # Más estresores = nivel más alto
                num_stressors = len(recent_stress)
                if num_stressors >= 3:
                    stress_level = 3.0
                elif num_stressors == 2:
                    stress_level = 2.0
                elif num_stressors == 1:
                    stress_level = 1.0
    
    return np.clip(stress_level, 0, CONFIG["cap_stress"])

def convert_ohio_to_subject_format(ohio_patient_data: dict) -> dict:
    """
    Convierte los datos de OhioT1DM al formato usado en el dataset original.

    Parámetros:
    -----------
    ohio_patient_data : dict
        Diccionario con datos de pacientes de OhioT1DM

    Retorna:
    --------
    dict
        Diccionario con datos convertidos al formato esperado por
        las funciones de procesamiento existentes
    """
    subject_data = {}
    
    for patient_id, data in ohio_patient_data.items():
        # Extraer datos CGM
        if 'glucose' in data and len(data['glucose']) > 0:
            cgm_df = data['glucose']
            if COL_TIMESTAMP in cgm_df.columns and COL_MGDL not in cgm_df.columns:
                if COL_VALUE in cgm_df.columns:
                    cgm_df = cgm_df.rename({COL_VALUE: COL_MGDL})
                else:
                    continue
            
            # Preparar datos de bolo
            bolus_df = None
            if 'bolus' in data and len(data['bolus']) > 0:
                orig_bolus = data['bolus']
                
                # Mapear datos de bolo al formato esperado
                bolus_rows = []
                for row in orig_bolus.iter_rows(named=True):
                    ts = row.get(COL_TS, None)
                    if not ts:
                        ts = row.get('ts_begin', None)
                    
                    if not ts:
                        continue
                    
                    # Obtener valor de glucosa al momento del bolo
                    bg_input = None
                    bolus_time = ts
                    closest_cgm = cgm_df.filter(
                        (pl.col(COL_TIMESTAMP) <= bolus_time) & 
                        (pl.col(COL_TIMESTAMP) >= bolus_time - timedelta(minutes=20))
                    ).sort(pl.col(COL_TIMESTAMP).diff().abs())
                    
                    if len(closest_cgm) > 0:
                        bg_input = closest_cgm[0, COL_MGDL]
                    
                    # Obtener info de comida si existe
                    carb_input = None
                    insulin_carb_ratio = None
                    if 'meal' in data and len(data['meal']) > 0:
                        meal_df = data['meal']
                        if COL_TS in meal_df.columns:
                            closest_meal = meal_df.filter(
                                (pl.col(COL_TS) >= bolus_time - timedelta(minutes=15)) &
                                (pl.col(COL_TS) <= bolus_time + timedelta(minutes=15))
                            )
                            
                            if len(closest_meal) > 0 and 'carbs' in closest_meal.columns:
                                carb_input = closest_meal[0, 'carbs']
                    
                    # Obtener características adicionales para predicción mejorada
                    sleep_quality = get_sleep_quality(data, bolus_time)
                    exercise_level = get_exercise_level(data, bolus_time)
                    stress_level = get_stress_level(data, bolus_time)
                    
                    bolus_rows.append({
                        COL_DATE: ts,
                        COL_NORMAL: row.get('dose', None),
                        'carbInput': carb_input,
                        'insulinCarbRatio': insulin_carb_ratio,
                        'bgInput': bg_input,
                        'insulinSensitivityFactor': None,
                        'targetBloodGlucose': None,
                        'insulinOnBoard': None,  # Se calculará después
                        'sleep_quality': sleep_quality,
                        'exercise_level': exercise_level,
                        'stress_level': stress_level
                    })
                
                # Crear DataFrame de bolus
                if bolus_rows:
                    bolus_df = pl.DataFrame(bolus_rows)
            
            # Si tenemos tanto datos CGM como bolus, añadir este paciente
            if bolus_df is not None and len(bolus_df) > 0:
                # Extraer datos basales si están disponibles
                basal_df = None
                if 'basal' in data and len(data['basal']) > 0:
                    orig_basal = data['basal']
                    if COL_TS in orig_basal.columns and COL_VALUE in orig_basal.columns:
                        basal_rows = []
                        for row in orig_basal.iter_rows(named=True):
                            basal_rows.append({
                                COL_DATE: row[COL_TS],
                                'duration': 3600000,  # 1 hora en milisegundos (por defecto)
                                'rate': row[COL_VALUE]
                            })
                        
                        if basal_rows:
                            basal_df = pl.DataFrame(basal_rows)
                
                # Añadir los datos de este paciente
                subject_data[patient_id] = {
                    'cgm_df': cgm_df,
                    'bolus_df': bolus_df,
                    'basal_df': basal_df,
                    # Añadir clasificación del paciente basada en datos CGM
                    COL_TENDENCY: classify_patient_tendency(cgm_df)
                }
    
    return subject_data

def process_ohio_subject(subject_data: dict, subject_id: str) -> list:
    """
    Procesa los datos de un sujeto de Ohio.

    Parámetros:
    -----------
    subject_data : dict
        Diccionario con los datos del sujeto
    subject_id : str
        ID del sujeto

    Retorna:
    --------
    list
        Lista de diccionarios con características procesadas
    """
    start_time = time.time()
    print(f"Procesando sujeto Ohio {subject_id}...")
    
    # Acceder a los datos del sujeto
    cgm_df = subject_data['cgm_df']
    bolus_df = subject_data['bolus_df']
    basal_df = subject_data.get('basal_df', None)
    tendency = subject_data.get(COL_TENDENCY, 'mixed')
    
    # Calcular medianas para imputación
    carb_median, iob_median = calculate_medians(bolus_df, basal_df)
    
    # Procesar cada fila de bolus
    processed_data = []
    date_col = COL_DATE if COL_DATE in bolus_df.columns else COL_TS
    
    for row in tqdm(bolus_df.iter_rows(named=True), total=len(bolus_df), 
                   desc=f"Procesando Ohio {subject_id}", leave=False):
        bolus_time = row[date_col]
        cgm_window = get_cgm_window(bolus_time, cgm_df)
        
        # Añadir tendencia a la fila
        row_with_tendency = dict(row)
        row_with_tendency[COL_TENDENCY] = tendency
        
        features = extract_features(row_with_tendency, cgm_window, carb_median, iob_median, basal_df, subject_id)
        if features is not None:
            processed_data.append(features)

    elapsed_time = time.time() - start_time
    print(f"Procesado sujeto Ohio {subject_id} en {elapsed_time:.2f} segundos")
    return processed_data

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
    # Manejar diferentes nombres de columnas entre datasets
    date_col = COL_DATE if COL_DATE in row else COL_TS
    bolus_time = row[date_col]
    
    if cgm_window is None:
        return None
    
    # Calcular IOB
    iob = calculate_iob(bolus_time, basal_df)
    iob = iob_median if iob == 0 else iob
    iob = np.clip(iob, 0, CONFIG["cap_iob"])
    
    # Características temporales
    hour_of_day = bolus_time.hour / 23.0
    
    # Características BG
    bg_col = 'bgInput' if 'bgInput' in row else 'bg'
    bg_input = row.get(bg_col) if row.get(bg_col) is not None else cgm_window[-1]
    bg_input = max(bg_input, 50.0)
    bg_input = np.clip(bg_input, 0, CONFIG["cap_bg"])
    
    # Características de insulina
    dose_col = COL_NORMAL if COL_NORMAL in row else 'dose'
    normal = row.get(dose_col) if row.get(dose_col) is not None else 0.0
    normal = np.clip(normal, 0, CONFIG["cap_normal"])
    
    # Calcular factor de sensibilidad a insulina personalizado
    isf_col = 'insulinSensitivityFactor'
    if isf_col in row and row[isf_col] is not None:
        isf_custom = row[isf_col]
    else:
        isf_custom = 50.0 if normal <= 0 else (bg_input - 100) / normal
    isf_custom = np.clip(isf_custom, 10, 100)
    
    # Características de carbohidratos
    carb_col = 'carbInput' if 'carbInput' in row else 'carbs'
    carb_input = row.get(carb_col) if row.get(carb_col) is not None else 0.0
    carb_input = carb_median if carb_input == 0 else carb_input
    carb_input = np.clip(carb_input, 0, CONFIG["cap_carb"])
    
    # Ratio insulina-carbohidratos
    icr_col = 'insulinCarbRatio'
    if icr_col in row and row[icr_col] is not None:
        insulin_carb_ratio = row[icr_col]
    else:
        insulin_carb_ratio = 10.0
    insulin_carb_ratio = np.clip(insulin_carb_ratio, 5, 20)
    
    # Características adicionales para el modelo mejorado
    sleep_quality = row.get('sleep_quality', 5.0)  # Valor por defecto si no está presente
    exercise_level = row.get('exercise_level', 0.0)
    stress_level = row.get('stress_level', 0.0)
    
    # Obtener ID del sujeto con prefijo - extraer prefijo ohio_ si existe
    if isinstance(idx, str) and idx.startswith('ohio_'):
        subject_id = idx
    else:
        subject_id = idx
    
    return {
        COL_SUBJECT_ID: subject_id,
        'cgm_window': cgm_window,
        'carbInput': carb_input,
        'bgInput': bg_input,
        'insulinCarbRatio': insulin_carb_ratio,
        'insulinSensitivityFactor': isf_custom,
        'insulinOnBoard': iob,
        'hour_of_day': hour_of_day,
        COL_NORMAL: normal,
        'sleep_quality': sleep_quality,
        'exercise_level': exercise_level,
        'stress_level': stress_level,
        'patient_tendency': row.get(COL_TENDENCY, 'mixed')  # Obtener tendencia si está disponible
    }

def preprocess_data(subject_folder: str) -> pl.DataFrame:
    """
    Preprocesa todos los datos de los sujetos, incluyendo OhioT1DM si está habilitado.

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
    all_processed_data = []
    
    # Procesar dataset original
    subject_files = [f for f in os.listdir(subject_folder) if f.startswith("Subject") and f.endswith(".xlsx")]
    print(f"\nFound Subject files ({len(subject_files)}):")
    for f in subject_files:
        print(f"  {f}")

    original_data = Parallel(n_jobs=-1)(delayed(process_subject)(os.path.join(subject_folder, f), idx) 
                                      for idx, f in enumerate(subject_files))
    original_data = [item for sublist in original_data for item in sublist]
    all_processed_data.extend(original_data)
    
    # Procesar dataset OhioT1DM si está habilitado
    if CONFIG["use_ohio_data"]:
        # Cargar datos OhioT1DM
        ohio_data = load_ohio_data()
        
        # Convertir al formato correcto
        ohio_subjects = convert_ohio_to_subject_format(ohio_data)
        
        # Procesar cada sujeto de Ohio
        ohio_processed_data = []
        for subject_id, subject_data in ohio_subjects.items():
            subject_processed = process_ohio_subject(subject_data, subject_id)
            ohio_processed_data.extend(subject_processed)
        
        all_processed_data.extend(ohio_processed_data)
    
    # Crear DataFrame con todos los datos procesados
    df_processed = pl.DataFrame(all_processed_data)
    print("\nMuestra de datos procesados combinados:")
    print(df_processed.head())
    print(f"Total de muestras: {len(df_processed)}")

    # Aplicar transformaciones logarítmicas a características numéricas
    numeric_columns = [COL_NORMAL, "carbInput", "insulinOnBoard", "bgInput"]
    for col in numeric_columns:
        if col in df_processed.columns:
            df_processed = df_processed.with_columns(
                pl.col(col).log1p().alias(col)
            )
    
    # Transformar datos de ventana CGM
    cgm_windows = df_processed["cgm_window"].to_numpy()
    transformed_windows = np.array([np.log1p(window) for window in cgm_windows])
    df_processed = df_processed.with_columns(
        pl.lit(transformed_windows).alias("cgm_window")
    )

    # Crear columnas para valores de ventana CGM
    cgm_columns = [f'cgm_{i}' for i in range(24)]
    cgm_data = np.array([row['cgm_window'] for row in df_processed.to_dicts()])
    
    # Crear DataFrame para datos CGM y unirlo con datos principales
    df_cgm = pl.DataFrame({col: cgm_data[:, i] for i, col in enumerate(cgm_columns)})
    
    # Excluir patient_tendency de las columnas regulares si lo usamos como objetivo
    regular_columns = [col for col in df_processed.columns if col not in ['cgm_window']]
    
    df_final = pl.concat([
        df_cgm, 
        df_processed.select(regular_columns)
    ], how="horizontal")
    
    # Verificar valores nulos
    print("\nVerificación de NaN en df_final:")
    null_counts = df_final.null_count()
    print(null_counts)
    df_final = df_final.drop_nulls()

    elapsed_time = time.time() - start_time
    print(f"\nPreprocesamiento completo en {elapsed_time:.2f} segundos")
    return df_final

def classify_patients_by_tendency(df: pl.DataFrame) -> pl.DataFrame:
    """
    Añade clasificación de tendencia glicémica de pacientes al DataFrame.

    Parámetros:
    -----------
    df : pl.DataFrame
        DataFrame con datos procesados

    Retorna:
    --------
    pl.DataFrame
        DataFrame con clasificación de tendencia de pacientes añadida
    """
    # Verificar si la columna patient_tendency ya existe
    if 'patient_tendency' in df.columns:
        return df
    
    # Agrupar por subject_id para calcular estadísticas
    subject_stats = df.group_by(COL_SUBJECT_ID).agg([
        pl.col('bgInput').mean().alias('avg_bg'),
        pl.col('bgInput').quantile(0.25).alias('q1_bg'),
        pl.col('bgInput').quantile(0.75).alias('q3_bg')
    ])
    
    # Clasificar cada sujeto
    classifications = []
    for row in subject_stats.iter_rows(named=True):
        subject_id = row[COL_SUBJECT_ID]
        avg_bg = row['avg_bg']
        
        if avg_bg > 5.3:  # Transformado usando log1p, ~200 mg/dL
            tendency = 'hyperglycemic'
        elif avg_bg < 4.5:  # ~90 mg/dL
            tendency = 'hypoglycemic'
        else:
            tendency = 'mixed'
            
        classifications.append({COL_SUBJECT_ID: subject_id, COL_TENDENCY: tendency})
    
    # Crear DataFrame con clasificaciones
    classifications_df = pl.DataFrame(classifications)
    
    # Unir con datos originales
    return df.join(classifications_df, on=COL_SUBJECT_ID, how='left')

def prepare_patient_classification_data(df: pl.DataFrame) -> tuple:
    """
    Prepara los datos para la tarea de clasificación de pacientes.

    Parámetros:
    -----------
    df : pl.DataFrame
        DataFrame con todos los datos procesados

    Retorna:
    --------
    tuple
        Tupla con (features, labels)
        - features: Características agregadas por paciente
        - labels: Etiquetas de tendencia glicémica
    """
    # Asegurar que tenemos la columna de tendencia
    if 'patient_tendency' not in df.columns:
        df = classify_patients_by_tendency(df)
    
    # Agregar características por paciente
    patient_features = df.group_by(COL_SUBJECT_ID).agg([
        pl.col('bgInput').mean().alias('avg_bg'),
        pl.col('bgInput').std().alias('bg_variability'),
        pl.col(COL_NORMAL).mean().alias('avg_insulin'),
        pl.col('carbInput').mean().alias('avg_carbs'),
        pl.col('insulinOnBoard').mean().alias('avg_iob'),
        pl.col('sleep_quality').mean().alias('avg_sleep_quality'),
        pl.col('exercise_level').mean().alias('avg_exercise'),
        pl.col('stress_level').mean().alias('avg_stress'),
        pl.col('patient_tendency').mode().first().alias(COL_TENDENCY)
    ])
    
    # Convertir a NumPy para compatibilidad con modelos
    features_cols = [col for col in patient_features.columns 
                   if col not in [COL_SUBJECT_ID, COL_TENDENCY]]
    
    features = patient_features.select(features_cols).to_numpy()
    labels = patient_features[COL_TENDENCY].to_list()
    
    # Codificar etiquetas como números (0: mixed, 1: hypoglycemic, 2: hyperglycemic)
    label_mapping = {'mixed': 0, 'hypoglycemic': 1, 'hyperglycemic': 2}
    numeric_labels = np.array([label_mapping[label] for label in labels])
    
    return features, numeric_labels

def split_data(df_final: pl.DataFrame) -> tuple:
    """
    Divide los datos siguiendo una estrategia para asegurar distribuciones 
    equilibradas entre los conjuntos de entrenamiento, validación y prueba.
    
    Prepara datos para ambas tareas: predicción de dosis y clasificación de pacientes.

    Parámetros:
    -----------
    df_final : pl.DataFrame
        DataFrame con todos los datos procesados

    Retorna:
    --------
    tuple
        Tupla con múltiples elementos para ambas tareas y metadatos
    """
    start_time = time.time()
    
    # Asegurar que tenemos clasificación de tendencia de pacientes
    if 'patient_tendency' not in df_final.columns:
        df_final = classify_patients_by_tendency(df_final)
    
    # Agregamos estadísticas por sujeto, incluyendo tendencia
    subject_stats = df_final.group_by(COL_SUBJECT_ID).agg([
        pl.col(COL_NORMAL).mean().alias("mean_dose"),
        pl.col(COL_NORMAL).std().alias("std_dose"),
        pl.col("patient_tendency").first().alias(COL_TENDENCY)
    ])
    
    # Obtener lista ordenada de sujetos
    sorted_subjects = subject_stats.sort("mean_dose").get_column(COL_SUBJECT_ID).to_list()
    n_subjects = len(sorted_subjects)
    train_size = int(0.8 * n_subjects)
    val_size = int(0.1 * n_subjects)
    test_size = n_subjects - train_size - val_size

    # Iniciar con sujeto específico de prueba si está disponible
    test_subjects = [49] if 49 in sorted_subjects else []
    remaining_subjects = [s for s in sorted_subjects if s != 49]
    train_subjects = []
    val_subjects = []

    # Aleatorizar la lista restante
    rng = np.random.default_rng(seed=CONST_DEFAULT_SEED)
    rng.shuffle(remaining_subjects)
    df_final_pd = df_final.to_pandas()

    # Distribuir sujetos mientras se balancean las tendencias glicémicas
    hyper_subjects = []
    hypo_subjects = []
    mixed_subjects = []
    
    for subject in remaining_subjects:
        # Obtener tendencia del sujeto
        tendency = subject_stats.filter(pl.col(COL_SUBJECT_ID) == subject)[COL_TENDENCY][0]
        
        if tendency == "hyperglycemic":
            hyper_subjects.append(subject)
        elif tendency == "hypoglycemic":
            hypo_subjects.append(subject)
        else:
            mixed_subjects.append(subject)
    
    # Función para distribuir sujetos manteniendo balance de tendencias
    def distribute_subjects(subjects_list, target_list, max_size):
        for subject in subjects_list:
            if len(target_list) < max_size:
                target_list.append(subject)
            else:
                break
        return [s for s in subjects_list if s not in target_list]
    
    # Distribuir sujetos hiperglucémicos
    hyper_train_target = int(0.8 * len(hyper_subjects))
    hyper_val_target = int(0.1 * len(hyper_subjects))
    
    hyper_subjects = distribute_subjects(hyper_subjects, train_subjects, hyper_train_target)
    hyper_subjects = distribute_subjects(hyper_subjects, val_subjects, hyper_val_target)
    distribute_subjects(hyper_subjects, test_subjects, len(hyper_subjects))
    
    # Distribuir sujetos hipoglucémicos
    hypo_train_target = int(0.8 * len(hypo_subjects))
    hypo_val_target = int(0.1 * len(hypo_subjects))
    
    hypo_subjects = distribute_subjects(hypo_subjects, train_subjects, hypo_train_target)
    hypo_subjects = distribute_subjects(hypo_subjects, val_subjects, hypo_val_target)
    distribute_subjects(hypo_subjects, test_subjects, len(hypo_subjects))
    
    # Distribuir sujetos mixtos
    mixed_train_target = int(0.8 * len(mixed_subjects))
    mixed_val_target = int(0.1 * len(mixed_subjects))
    
    mixed_subjects = distribute_subjects(mixed_subjects, train_subjects, mixed_train_target)
    mixed_subjects = distribute_subjects(mixed_subjects, val_subjects, mixed_val_target)
    distribute_subjects(mixed_subjects, test_subjects, len(mixed_subjects))

    # Crear máscaras para división de datos
    train_mask = df_final_pd[COL_SUBJECT_ID].isin(train_subjects)
    val_mask = df_final_pd[COL_SUBJECT_ID].isin(val_subjects)
    test_mask = df_final_pd[COL_SUBJECT_ID].isin(test_subjects)

    # Mostrar estadísticas post-división
    for set_name, mask in [("Train", train_mask), ("Val", val_mask), ("Test", test_mask)]:
        y_temp = df_final_pd.loc[mask, COL_NORMAL]
        print(f"Post-split {set_name} y: mean = {y_temp.mean()}, std = {y_temp.std()}")

    # Definir grupos de columnas
    cgm_columns = [f'cgm_{i}' for i in range(24)]
    other_features = [
        'carbInput', 'bgInput', 'insulinOnBoard', 'insulinCarbRatio', 
        'insulinSensitivityFactor', 'hour_of_day',
        'sleep_quality', 'exercise_level', 'stress_level'  # Nuevas características
    ]
    
    # Asegurar que todas las columnas existen en el DataFrame
    other_features = [col for col in other_features if col in df_final_pd.columns]

    # Inicializar escaladores
    scaler_cgm = StandardScaler().fit(df_final_pd.loc[train_mask, cgm_columns])
    scaler_other = StandardScaler().fit(df_final_pd.loc[train_mask, other_features])
    scaler_y = StandardScaler().fit(df_final_pd.loc[train_mask, COL_NORMAL].values.reshape(-1, 1))

    # Preparar datos CGM
    x_cgm_train = prepare_data_with_scaler(df_final_pd, train_mask, cgm_columns, scaler_cgm, reshape=(-1, 24, 1))
    x_cgm_val = prepare_data_with_scaler(df_final_pd, val_mask, cgm_columns, scaler_cgm, reshape=(-1, 24, 1))
    x_cgm_test = prepare_data_with_scaler(df_final_pd, test_mask, cgm_columns, scaler_cgm, reshape=(-1, 24, 1))
    
    # Preparar otras características
    x_other_train = prepare_data_with_scaler(df_final_pd, train_mask, other_features, scaler_other)
    x_other_val = prepare_data_with_scaler(df_final_pd, val_mask, other_features, scaler_other)
    x_other_test = prepare_data_with_scaler(df_final_pd, test_mask, other_features, scaler_other)
    
    # Preparar etiquetas de dosis de insulina
    y_train = scaler_y.transform(df_final_pd.loc[train_mask, COL_NORMAL].values.reshape(-1, 1)).flatten()
    y_val = scaler_y.transform(df_final_pd.loc[val_mask, COL_NORMAL].values.reshape(-1, 1)).flatten()
    y_test = scaler_y.transform(df_final_pd.loc[test_mask, COL_NORMAL].values.reshape(-1, 1)).flatten()

    # Obtener IDs de sujeto
    x_subject_train = df_final_pd.loc[train_mask, COL_SUBJECT_ID].values
    x_subject_val = df_final_pd.loc[val_mask, COL_SUBJECT_ID].values
    x_subject_test = df_final_pd.loc[test_mask, COL_SUBJECT_ID].values
    
    # Crear mapeo de ID de sujeto a tendencia
    patient_tendencies = {}
    for subject_id, tendency in zip(subject_stats[COL_SUBJECT_ID].to_list(), subject_stats[COL_TENDENCY].to_list()):
        patient_tendencies[subject_id] = tendency
    
    # Preparar datos para clasificación de pacientes
    # Esto usa las características agregadas por paciente
    class_features, class_labels = prepare_patient_classification_data(df_final)
    
    # Crear máscaras a nivel de paciente para clasificación
    unique_subjects = list(set(subject_stats[COL_SUBJECT_ID].to_list()))
    class_train_mask = np.array([subject in train_subjects for subject in unique_subjects])
    class_val_mask = np.array([subject in val_subjects for subject in unique_subjects])
    class_test_mask = np.array([subject in test_subjects for subject in unique_subjects])
    
    # Normalizar características para clasificación usando solo datos de entrenamiento
    train_class_features = class_features[class_train_mask]
    scaler_class = StandardScaler().fit(train_class_features)
    class_features_scaled = scaler_class.transform(class_features)
    
    # Preparar datos de clasificación para cada conjunto
    x_class_train = class_features_scaled[class_train_mask]
    y_class_train = class_labels[class_train_mask]
    
    x_class_val = class_features_scaled[class_val_mask]
    y_class_val = class_labels[class_val_mask]
    
    x_class_test = class_features_scaled[class_test_mask]
    y_class_test = class_labels[class_test_mask]
    
    # Imprimir resumen
    print(f"Entrenamiento CGM: {x_cgm_train.shape}, Validación CGM: {x_cgm_val.shape}, Prueba CGM: {x_cgm_test.shape}")
    print(f"Entrenamiento Otros: {x_other_train.shape}, Validación Otros: {x_other_val.shape}, Prueba Otros: {x_other_test.shape}")
    
    # Contar sujetos por tendencia en cada conjunto
    def count_by_tendency(ids):
        hyper = sum(1 for id in ids if patient_tendencies.get(id) == 'hyperglycemic')
        hypo = sum(1 for id in ids if patient_tendencies.get(id) == 'hypoglycemic')
        mixed = sum(1 for id in ids if patient_tendencies.get(id) == 'mixed')
        return {'hyperglycemic': hyper, 'hypoglycemic': hypo, 'mixed': mixed}
    
    train_counts = count_by_tendency(train_subjects)
    val_counts = count_by_tendency(val_subjects)
    test_counts = count_by_tendency(test_subjects)
    
    print("Distribución de tendencias de pacientes:")
    print(f"  Train: {train_counts}")
    print(f"  Val: {val_counts}")
    print(f"  Test: {test_counts}")
    print(f"Sujetos de prueba: {test_subjects}")

    elapsed_time = time.time() - start_time
    print(f"División de datos completa en {elapsed_time:.2f} segundos")
    
    return (x_cgm_train, x_cgm_val, x_cgm_test,
            x_other_train, x_other_val, x_other_test,
            x_subject_train, x_subject_val, x_subject_test,
            y_train, y_val, y_test, 
            x_class_train, x_class_val, x_class_test,
            y_class_train, y_class_val, y_class_test,
            patient_tendencies, test_subjects,
            scaler_cgm, scaler_other, scaler_y, scaler_class)