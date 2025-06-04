from typing import Union
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import sys
import os
from joblib import Parallel, delayed
import time
from datetime import timedelta, datetime
from tqdm import tqdm
import matplotlib
matplotlib.use('Agg')
import xml.etree.ElementTree as ET
import glob
import logging
from zoneinfo import ZoneInfo

# Constantes para evitar repetición de strings
PROJECT_ROOT: str = os.path.abspath(os.path.join(os.getcwd(), ".."))
sys.path.append(PROJECT_ROOT)
DATA_PATH_SUBJECTS: str = os.path.join(os.getcwd(), "data", "subjects")
OHIO_DATA_DIRS: list[str] = [
    os.path.join(os.getcwd(), "data", "OhioT1DM","2018","train"), 
    os.path.join(os.getcwd(), "data", "OhioT1DM","2018","test"), 
    os.path.join(os.getcwd(), "data", "OhioT1DM","2020","train"), 
    os.path.join(os.getcwd(), "data", "OhioT1DM","2020","test")
]
OUTPUT_DIR: str = 'new_ohio/processed_data'
PLOTS_DIR: str = 'new_ohio/processed_data/plots'
DATE_FORMAT: str = "%d-%m-%Y %H:%M:%S"
TIMESTAMP_COL: str = "Timestamp"
SUBJECT_ID_COL: str = "SubjectID"
GLUCOSE_COL: str = "value"
BOLUS_COL: str = "bolus"
MEAL_COL: str = "meal_carbs"
BASAL_COL: str = "basal_rate"
TEMP_BASAL_COL: str = "temp_basal_rate"
CONST_DEFAULT_SEED: int = 42

# Configuración de logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Configuración global
CONFIG: dict[str, Union[int, float, str]] = {
    "batch_size": 128,
    "window_hours": 2,
    "window_steps": 24,  # 5-min steps in 2 hours
    "insulin_lifetime_hours": 4.0,
    "cap_normal": 30,
    "cap_bg": 300,
    "cap_iob": 5,
    "cap_carb": 150,
    "min_carbs": 0,
    "max_carbs": 150,
    "min_bg": 40,
    "max_bg": 400,
    "min_insulin": 0,
    "max_insulin": 30,
    "min_icr": 5,
    "max_icr": 20,
    "min_isf": 10,
    "max_isf": 100,
    "timezone": "UTC",
    "max_work_intensity": 10,
    "max_sleep_quality": 10,
    "max_activity_intensity": 10,
    "low_dose_threshold": 7.0,  # Umbral clínico para dosis baja de insulina
}

def load_excel_data(subject_path: str) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Carga datos de un sujeto desde un archivo Excel con hojas CGM, Bolus y Basal.

    Parámetros:
    -----------
    subject_path : str
        Ruta al archivo Excel del sujeto.

    Retorna:
    --------
    tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]
        Tupla con (cgm_df, bolus_df, basal_df), donde cada elemento es un DataFrame
        o None si hubo error en la carga.
    """
    try:
        cgm_df: pd.DataFrame = pd.read_excel(subject_path, sheet_name="CGM")
        bolus_df: pd.DataFrame = pd.read_excel(subject_path, sheet_name="Bolus")
        try:
            basal_df: pd.DataFrame = pd.read_excel(subject_path, sheet_name="Basal")
        except Exception:
            basal_df = None

        # Conversión de fechas con precisión
        cgm_df[TIMESTAMP_COL] = pd.to_datetime(cgm_df["date"], errors='coerce')
        cgm_df = cgm_df.sort_values(TIMESTAMP_COL).rename(columns={"mg/dl": GLUCOSE_COL})
        bolus_df[TIMESTAMP_COL] = pd.to_datetime(bolus_df["date"], errors='coerce')
        # Asegurar que columnas numéricas sean numéricas, convirtiendo valores no válidos a NaN
        numeric_cols = ["bgInput", "normal", "carbInput", "insulinCarbRatio"]
        for col in numeric_cols:
            if col in bolus_df.columns:
                bolus_df[col] = pd.to_numeric(bolus_df[col], errors='coerce')
        if basal_df is not None:
            basal_df[TIMESTAMP_COL] = pd.to_datetime(basal_df["date"], errors='coerce')
        return cgm_df, bolus_df, basal_df
    except Exception as e:
        logging.error(f"Error al cargar {os.path.basename(subject_path)}: {e}")
        return None, None, None

def load_xml_data(data_dir: str) -> dict[str, pd.DataFrame]:
    """
    Carga datos desde archivos XML en el directorio especificado.

    Parámetros:
    -----------
    data_dir : str
        Directorio con archivos XML.

    Retorna:
    --------
    dict[str, pd.DataFrame]
        Diccionario con DataFrames por tipo de dato (glucose_level, bolus, meal, etc.).
    """
    logging.info(f"Cargando datos desde {data_dir}")
    data_dict: dict[str, pd.DataFrame] = {}
    for xml_file in glob.glob(os.path.join(data_dir, "*.xml")):
        try:
            tree = ET.parse(xml_file)
            root = tree.getroot()
            subject_id: str = os.path.basename(xml_file).split('.')[0]
            for data_type_elem in root:
                data_type: str = data_type_elem.tag
                if data_type == 'patient':
                    continue
                records: list[dict] = []
                for event in data_type_elem:
                    record_dict: dict = dict(event.attrib)
                    record_dict[SUBJECT_ID_COL] = subject_id
                    records.append(record_dict)
                if records:
                    df: pd.DataFrame = pd.DataFrame(records)
                    data_dict[data_type] = df
        except Exception as e:
            logging.error(f"Error procesando {xml_file}: {e}")
            continue
    return data_dict

def preprocess_xml_bolus_meal(data: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    """
    Preprocesa los datos de bolus y meal de XML, renombrando columnas y convirtiendo timestamps.

    Parámetros:
    -----------
    data : dict[str, pd.DataFrame]
        Diccionario con DataFrames de datos XML.

    Retorna:
    --------
    dict[str, pd.DataFrame]
        Diccionario con DataFrames preprocesados de bolus y meal.
    """
    processed: dict[str, pd.DataFrame] = {}
    # Procesar bolus
    if "bolus" in data:
        bolus: pd.DataFrame = data["bolus"].copy()
        if "dose" in bolus.columns:
            bolus = bolus.rename(columns={"dose": BOLUS_COL})
        if "ts_begin" in bolus.columns:
            bolus[TIMESTAMP_COL] = pd.to_datetime(bolus["ts_begin"], format=DATE_FORMAT, errors='coerce')
        processed["bolus"] = bolus
    # Procesar meal
    if "meal" in data:
        meal: pd.DataFrame = data["meal"].copy()
        if "carbs" in meal.columns:
            meal = meal.rename(columns={"carbs": MEAL_COL})
        if "ts" in meal.columns:
            meal[TIMESTAMP_COL] = pd.to_datetime(meal["ts"], format=DATE_FORMAT, errors='coerce')
        processed["meal"] = meal
    return processed

def align_events_to_cgm(cgm_df: pd.DataFrame, event_df: pd.DataFrame, event_time_col: str = TIMESTAMP_COL, tolerance_minutes: int = 5) -> pd.DataFrame:
    """
    Alinea eventos (bolus, meal, etc.) con el timestamp de CGM más cercano dentro de una tolerancia.

    Parámetros:
    -----------
    cgm_df : pd.DataFrame
        DataFrame con datos CGM.
    event_df : pd.DataFrame
        DataFrame con eventos a alinear.
    event_time_col : str, opcional
        Columna de timestamp en event_df (default: "Timestamp").
    tolerance_minutes : int, opcional
        Tolerancia en minutos para la alineación (default: 5).

    Retorna:
    --------
    pd.DataFrame
        DataFrame de eventos con timestamps alineados.
    """
    if cgm_df.empty or event_df.empty:
        return event_df

    # Convertir los timestamps a formato datetime64 para cálculos numéricos
    cgm_times: np.ndarray = cgm_df[TIMESTAMP_COL].values.astype("datetime64[ns]")
    aligned_rows: list[dict] = []
    for idx, row in event_df.iterrows():
        event_time = row[event_time_col]
        if not isinstance(event_time, pd.Timestamp):
            try:
                event_time = pd.to_datetime(event_time)
            except Exception:
                continue
        if pd.isna(event_time):  # Saltar si event_time es NaT
            continue
        # Convertir event_time a datetime64 para la resta
        event_time_np = np.datetime64(event_time)
        # Calcular la diferencia absoluta en nanosegundos
        time_diffs = np.abs(cgm_times - event_time_np)
        idx_nearest = np.argmin(time_diffs)
        nearest_cgm_time = cgm_df[TIMESTAMP_COL].iloc[idx_nearest]
        # Convertir la diferencia a minutos
        diff_minutes: float = time_diffs[idx_nearest].astype("timedelta64[ns]").astype(float) / (60 * 1e9)
        if diff_minutes <= tolerance_minutes:
            row_dict = row.to_dict()
            row_dict[TIMESTAMP_COL] = nearest_cgm_time
            aligned_rows.append(row_dict)
    return pd.DataFrame(aligned_rows)

def preprocess_cgm(cgm: pd.DataFrame) -> pd.DataFrame:
    """
    Preprocesa los datos de CGM, convirtiendo la columna de timestamp.

    Parámetros:
    -----------
    cgm : pd.DataFrame
        DataFrame con datos CGM.

    Retorna:
    --------
    pd.DataFrame
        DataFrame con la columna de timestamp convertida.
    """
    if "ts" in cgm.columns:
        cgm[TIMESTAMP_COL] = pd.to_datetime(cgm["ts"], format=DATE_FORMAT, errors='coerce')
    return cgm

def join_signals(data: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """
    Une señales de CGM, bolus, meal, basal y temp_basal alineadas por Timestamp y SubjectID.

    Parámetros:
    -----------
    data : dict[str, pd.DataFrame]
        Diccionario con DataFrames de diferentes tipos de datos.

    Retorna:
    --------
    pd.DataFrame
        DataFrame unificado con todas las señales.
    """
    df: pd.DataFrame = data["glucose_level"].copy()
    # Join bolus
    if "bolus" in data and not data["bolus"].empty:
        df = df.merge(
            data["bolus"][[TIMESTAMP_COL, BOLUS_COL, SUBJECT_ID_COL]],
            on=[TIMESTAMP_COL, SUBJECT_ID_COL],
            how="left"
        )
    else:
        df[BOLUS_COL] = None
    # Join meal
    if "meal" in data and not data["meal"].empty:
        df = df.merge(
            data["meal"][[TIMESTAMP_COL, MEAL_COL, SUBJECT_ID_COL]],
            on=[TIMESTAMP_COL, SUBJECT_ID_COL],
            how="left"
        )
    else:
        df[MEAL_COL] = None
    # Join basal
    if "basal" in data and "dose" in data["basal"].columns:
        basal: pd.DataFrame = data["basal"].copy()
        basal[BASAL_COL] = basal["dose"].astype(float)
        if "ts_begin" in basal.columns:
            basal[TIMESTAMP_COL] = pd.to_datetime(basal["ts_begin"], format=DATE_FORMAT, errors='coerce')
        df = df.merge(
            basal[[TIMESTAMP_COL, SUBJECT_ID_COL, BASAL_COL]],
            on=[TIMESTAMP_COL, SUBJECT_ID_COL],
            how="left"
        )
    else:
        df[BASAL_COL] = 0.0
    # Join temp_basal
    if "temp_basal" in data and "dose" in data["temp_basal"].columns:
        temp: pd.DataFrame = data["temp_basal"].copy()
        temp[TEMP_BASAL_COL] = temp["dose"].astype(float)
        if "ts_begin" in temp.columns:
            temp[TIMESTAMP_COL] = pd.to_datetime(temp["ts_begin"], format=DATE_FORMAT, errors='coerce')
        df = df.merge(
            temp[[TIMESTAMP_COL, SUBJECT_ID_COL, TEMP_BASAL_COL]],
            on=[TIMESTAMP_COL, SUBJECT_ID_COL],
            how="left"
        )
    else:
        df[TEMP_BASAL_COL] = 0.0
    return df

def ensure_timestamp_datetime(df: pd.DataFrame, col: str = TIMESTAMP_COL) -> pd.DataFrame:
    """
    Convierte la columna de timestamp a pd.Timestamp, tolerando diferentes formatos.

    Parámetros:
    -----------
    df : pd.DataFrame
        DataFrame con la columna de timestamp.
    col : str, opcional
        Nombre de la columna de timestamp (default: "Timestamp").

    Retorna:
    --------
    pd.DataFrame
        DataFrame con la columna de timestamp convertida a pd.Timestamp.
    """
    if col in df.columns:
        df[col] = pd.to_datetime(df[col], errors='coerce')
    return df

def calculate_iob(bolus_time: datetime, basal_df: pd.DataFrame, half_life_hours: float = 4.0) -> float:
    """
    Calcula la insulina activa en el cuerpo (IOB).

    Parámetros:
    -----------
    bolus_time : datetime
        Tiempo del bolo de insulina.
    basal_df : pd.DataFrame
        DataFrame con datos de insulina basal.
    half_life_hours : float, opcional
        Vida media de la insulina en horas (default: 4.0).

    Retorna:
    --------
    float
        Cantidad de insulina activa en el organismo.
    """
    if basal_df is None or basal_df.empty:
        return 0.0
    
    iob: float = 0.0
    for _, row in basal_df.iterrows():
        start_time: datetime = row[TIMESTAMP_COL]
        duration_hours: float = row["duration"] / (1000 * 3600) if "duration" in row else 0.0
        end_time: datetime = start_time + timedelta(hours=duration_hours)
        rate: float = row[BASAL_COL] if BASAL_COL in row and row[BASAL_COL] is not None else 0.9
        rate = min(rate, 2.0)
        
        if start_time <= bolus_time <= end_time:
            time_since_start: float = (bolus_time - start_time).total_seconds() / 3600
            remaining: float = rate * (1 - (time_since_start / half_life_hours))
            iob += max(0.0, remaining)
    
    return min(iob, CONFIG["cap_iob"])

def get_cgm_window(bolus_time: datetime, cgm_df: pd.DataFrame, window_hours: int = CONFIG["window_hours"]) -> np.ndarray:
    """
    Obtiene la ventana de datos CGM para un tiempo de bolo específico.

    Parámetros:
    -----------
    bolus_time : datetime
        Tiempo del bolo de insulina.
    cgm_df : pd.DataFrame
        DataFrame con datos CGM.
    window_hours : int, opcional
        Horas de la ventana de datos (default: 2).

    Retorna:
    --------
    np.ndarray
        Ventana de datos CGM o None si no hay suficientes datos.
    """
    window_start: datetime = bolus_time - timedelta(hours=window_hours)
    window: pd.DataFrame = cgm_df[
        (cgm_df[TIMESTAMP_COL] >= window_start) & (cgm_df[TIMESTAMP_COL] <= bolus_time)
    ].sort_values(TIMESTAMP_COL).tail(24)
    
    if len(window) < 24:
        return None
    return window[GLUCOSE_COL].values

def generate_windows(df: pd.DataFrame, window_size: int = CONFIG["window_steps"]) -> pd.DataFrame:
    """
    Genera ventanas de CGM de tamaño fijo antes de cada evento bolus.

    Parámetros:
    -----------
    df : pd.DataFrame
        DataFrame con datos de CGM y bolus.
    window_size : int, opcional
        Cantidad de pasos en la ventana (default: 24 para 2 horas con datos cada 5 min).

    Retorna:
    --------
    pd.DataFrame
        DataFrame con ventanas generadas.
    """
    windows: list[dict] = []
    bolus_events: pd.DataFrame = df[df[BOLUS_COL] > 0]
    for _, row in bolus_events.iterrows():
        ts: datetime = row[TIMESTAMP_COL]
        subject_id: str = row[SUBJECT_ID_COL]
        cgm_window: pd.DataFrame = df[
            (df[SUBJECT_ID_COL] == subject_id) &
            (df[TIMESTAMP_COL] <= ts) &
            (df[TIMESTAMP_COL] > ts - timedelta(minutes=window_size*5))
        ].sort_values(TIMESTAMP_COL)
        if len(cgm_window) == window_size:
            windows.append({
                SUBJECT_ID_COL: subject_id,
                TIMESTAMP_COL: ts,
                "cgm_window": cgm_window[GLUCOSE_COL].tolist(),
                BOLUS_COL: row[BOLUS_COL],
                MEAL_COL: row.get(MEAL_COL, 0.0)
            })
    return pd.DataFrame(windows)

def calculate_medians(bolus_df: pd.DataFrame, basal_df: pd.DataFrame) -> tuple[float, float]:
    """
    Calcula valores medianos para imputación de datos faltantes.

    Parámetros:
    -----------
    bolus_df : pd.DataFrame
        DataFrame con datos de bolos de insulina.
    basal_df : pd.DataFrame
        DataFrame con datos de insulina basal.

    Retorna:
    --------
    tuple[float, float]
        Tupla con (carb_median, iob_median).
    """
    non_zero_carbs: pd.Series = bolus_df[bolus_df["carbInput"] > 0]["carbInput"]
    carb_median: float = non_zero_carbs.median() if len(non_zero_carbs) > 0 else 10.0
    
    iob_values: list[float] = []
    for _, row in bolus_df.iterrows():
        iob: float = calculate_iob(row[TIMESTAMP_COL], basal_df)
        iob_values.append(iob)
    
    non_zero_iob: list[float] = [iob for iob in iob_values if iob > 0]
    iob_median: float = np.median(non_zero_iob) if non_zero_iob else 0.5
    
    return carb_median, iob_median

def extract_features(df: pd.DataFrame, meal_df: pd.DataFrame) -> pd.DataFrame:
    """
    Extrae características de las ventanas CGM y agrega información de meal.

    Parámetros:
    -----------
    df : pd.DataFrame
        DataFrame con ventanas CGM y datos de bolus.
    meal_df : pd.DataFrame
        DataFrame con datos de meals.

    Retorna:
    --------
    pd.DataFrame
        DataFrame con características extraídas.
    """
    df["glucose_last"] = df["cgm_window"].apply(lambda x: x[-1] if len(x) > 0 else np.nan)
    df["glucose_mean"] = df["cgm_window"].apply(lambda x: np.mean(x) if len(x) > 0 else np.nan)
    df["glucose_std"] = df["cgm_window"].apply(lambda x: np.std(x) if len(x) > 0 else np.nan)
    df["glucose_min"] = df["cgm_window"].apply(lambda x: np.min(x) if len(x) > 0 else np.nan)
    df["glucose_max"] = df["cgm_window"].apply(lambda x: np.max(x) if len(x) > 0 else np.nan)
    df["hour_of_day"] = df[TIMESTAMP_COL].apply(lambda x: (x.hour * 60 + x.minute) / (24 * 60))
    df["bg_input"] = df["cgm_window"].apply(lambda x: x[-1] if len(x) > 0 else 0.0)
    df["insulin_on_board"] = 0.0

    required_cols: list[str] = ["carb_input", BASAL_COL, TEMP_BASAL_COL, "meal_time_diff", "meal_time_diff_hours", "has_meal", "meals_in_window"]
    for col in required_cols:
        default: Union[float, int] = 0.0 if "time" not in col and "meals" not in col else 0
        df[col] = default

    df["meal_time_diff_hours"] = df["meal_time_diff"] / 60.0
    df["has_meal"] = np.where(df["meal_time_diff"] > 0, 1.0, 0.0)

    if not meal_df.empty:
        meal_df[TIMESTAMP_COL] = pd.to_datetime(meal_df[TIMESTAMP_COL], errors='coerce')
        matched: list[dict] = []
        for _, row in df.iterrows():
            bolus_time = row.get(TIMESTAMP_COL, None)
            if bolus_time is None:
                continue
            start: datetime = bolus_time
            end: datetime = bolus_time + timedelta(hours=1)
            meals: pd.DataFrame = meal_df[(meal_df[TIMESTAMP_COL] >= start) & (meal_df[TIMESTAMP_COL] <= end)]
            if not meals.empty:
                meals = meals.copy()
                meals["time_diff"] = (meals[TIMESTAMP_COL] - bolus_time).dt.total_seconds() / 60.0
                meals = meals.sort_values("time_diff")
                closest: dict = meals.iloc[0].to_dict()
                meal_time_diff = closest.get("time_diff", 0.0)
                matched.append({
                    TIMESTAMP_COL: bolus_time,
                    MEAL_COL: float(closest.get(MEAL_COL, 0.0)),
                    "meal_time_diff": float(meal_time_diff),
                    "meals_in_window": float(len(meals))
                })
            else:
                matched.append({
                    TIMESTAMP_COL: bolus_time,
                    MEAL_COL: 0.0,
                    "meal_time_diff": 0.0,
                    "meals_in_window": 0.0
                })

        meals_info: pd.DataFrame = pd.DataFrame(matched)
        if TIMESTAMP_COL in meals_info.columns:
            meals_info[TIMESTAMP_COL] = pd.to_datetime(meals_info[TIMESTAMP_COL], errors='coerce')
            df[TIMESTAMP_COL] = pd.to_datetime(df[TIMESTAMP_COL], errors='coerce')
            df = df.merge(meals_info, on=TIMESTAMP_COL, how="left", suffixes=('', '_meal'))
            df[MEAL_COL] = df[MEAL_COL].astype(float).fillna(0.0)
            df["meal_time_diff"] = df["meal_time_diff"].astype(float).fillna(0.0)
            df["meals_in_window"] = df["meals_in_window"].astype(float).fillna(0.0)
            df["meal_time_diff_hours"] = df["meal_time_diff"] / 60.0
            df["has_meal"] = np.where(df["meals_in_window"] > 0, 1.0, 0.0)

    return df

def transform_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aplica transformaciones finales a las características para entrenamiento.

    Parámetros:
    -----------
    df : pd.DataFrame
        DataFrame con características extraídas.

    Retorna:
    --------
    pd.DataFrame
        DataFrame con características transformadas.
    """
    for col in ["glucose_last", "glucose_mean", "glucose_std", "glucose_min", "glucose_max"]:
        df[f"{col}_log1p"] = np.log1p(df[col].astype(float))

    df["glucose_trend"] = df.apply(
        lambda x: (x["cgm_window"][-1] - x["cgm_window"][0]) if len(x["cgm_window"]) > 1 else 0.0,
        axis=1
    ).astype(float)
    df["glucose_variability"] = df["cgm_window"].apply(lambda x: np.std(x) if len(x) > 0 else 0.0).astype(float)

    df["cgm_trend"] = df["glucose_trend"].astype(float)
    df["cgm_std"] = df["glucose_variability"].astype(float)

    for col in [BOLUS_COL, "carb_input", MEAL_COL, "insulin_on_board"]:
        df[f"{col}_log1p"] = np.log1p(df[col].astype(float))

    optional_features: list[str] = ['work_intensity', 'sleep_quality', 'activity_intensity']
    for feature in optional_features:
        if feature in df.columns:
            df[feature] = np.where(
                df[feature].notna(),
                df[feature] / CONFIG[f'max_{feature}'],
                None
            ).astype(float)

    window_size: int = CONFIG["window_steps"]
    cgm_cols: list[str] = [f"cgm_{i}" for i in range(window_size)]
    for i in range(window_size):
        df[f"cgm_{i}"] = df["cgm_window"].apply(lambda x: x[i] if len(x) > i else np.nan).astype(float)

    for col in cgm_cols:
        df[f"{col}_log"] = np.log1p(df[col].astype(float))

    return df.drop(columns=["cgm_window"])

def extract_features_excel(row: dict, cgm_window: np.ndarray, carb_median: float, iob_median: float, basal_df: pd.DataFrame, idx: int) -> dict:
    """
    Extrae características para una instancia de bolo individual desde datos Excel.

    Parámetros:
    -----------
    row : dict
        Fila con datos del bolo.
    cgm_window : np.ndarray
        Ventana de datos CGM.
    carb_median : float
        Valor mediano de carbohidratos para imputación.
    iob_median : float
        Valor mediano de IOB para imputación.
    basal_df : pd.DataFrame
        DataFrame con datos de insulina basal.
    idx : int
        Índice del sujeto.

    Retorna:
    --------
    dict
        Diccionario con características extraídas o None si no hay datos suficientes.
    """
    bolus_time: datetime = row[TIMESTAMP_COL]
    if cgm_window is None:
        return None
    
    iob: float = calculate_iob(bolus_time, basal_df)
    iob = iob_median if iob == 0 else iob
    iob = np.clip(iob, 0, CONFIG["cap_iob"])
    
    hour_of_day: float = bolus_time.hour / 23.0
    
    # Manejar bgInput, asegurando que sea un valor numérico
    bg_input = row["bgInput"]
    if pd.isna(bg_input):  # Si es NaN o NaT, usar el último valor de cgm_window
        bg_input = cgm_window[-1]
    else:
        try:
            bg_input = float(bg_input)  # Asegurar que sea float
        except (ValueError, TypeError):
            bg_input = cgm_window[-1]  # Fallback si no se puede convertir
    
    bg_input = max(bg_input, 50.0)
    bg_input = np.clip(bg_input, 0, CONFIG["cap_bg"])
    
    # Manejar normal, asegurando que sea un valor numérico
    normal = row["normal"]
    if pd.isna(normal):  # Si es NaN o NaT, usar 0.0 como valor predeterminado
        normal = 0.0
    else:
        try:
            normal = float(normal)  # Asegurar que sea float
        except (ValueError, TypeError):
            normal = 0.0  # Fallback si no se puede convertir
    normal = np.clip(normal, 0, CONFIG["cap_normal"])
    
    isf_custom: float = 50.0 if normal <= 0 else (bg_input - 100) / normal
    isf_custom = np.clip(isf_custom, 10, 100)
    
    # Manejar carbInput, asegurando que sea un valor numérico
    carb_input = row["carbInput"]
    if pd.isna(carb_input):  # Si es NaN o NaT, usar 0.0 como valor predeterminado
        carb_input = 0.0
    else:
        try:
            carb_input = float(carb_input)  # Asegurar que sea float
        except (ValueError, TypeError):
            carb_input = 0.0  # Fallback si no se puede convertir
    carb_input = carb_median if carb_input == 0 else carb_input
    carb_input = np.clip(carb_input, 0, CONFIG["cap_carb"])
    
    # Manejar insulinCarbRatio, asegurando que sea un valor numérico
    insulin_carb_ratio = row["insulinCarbRatio"]
    if pd.isna(insulin_carb_ratio):  # Si es NaN o NaT, usar 10.0 como valor predeterminado
        insulin_carb_ratio = 10.0
    else:
        try:
            insulin_carb_ratio = float(insulin_carb_ratio)  # Asegurar que sea float
        except (ValueError, TypeError):
            insulin_carb_ratio = 10.0  # Fallback si no se puede convertir
    insulin_carb_ratio = np.clip(insulin_carb_ratio, 5, 20)
    
    return {
        'subject_id': idx,  # Mantener como int
        TIMESTAMP_COL: bolus_time,  # Añadir Timestamp
        'cgm_window': cgm_window.tolist(),  # Convertir a lista para evitar problemas con NumPy array
        'carb_input': float(carb_input),
        'bg_input': float(bg_input),
        'insulin_carb_ratio': float(insulin_carb_ratio),
        'insulin_sensitivity_factor': float(isf_custom),
        'insulin_on_board': float(iob),
        'hour_of_day': float(hour_of_day),
        BOLUS_COL: float(normal)
    }

def process_excel_subject(subject_path: str, idx: int) -> list[dict]:
    """
    Procesa los datos de un sujeto desde un archivo Excel.

    Parámetros:
    -----------
    subject_path : str
        Ruta al archivo del sujeto.
    idx : int
        Índice del sujeto.

    Retorna:
    --------
    list[dict]
        Lista de diccionarios con características procesadas.
    """
    start_time: float = time.time()
    logging.info(f"Procesando {os.path.basename(subject_path)} (Sujeto {idx+1})...")
    
    cgm_df, bolus_df, basal_df = load_excel_data(subject_path)
    if cgm_df is None or bolus_df is None:
        return []

    carb_median, iob_median = calculate_medians(bolus_df, basal_df)
    processed_data: list[dict] = []
    for _, row in tqdm(bolus_df.iterrows(), total=len(bolus_df), desc=f"Procesando {os.path.basename(subject_path)}"):
        bolus_time: datetime = row[TIMESTAMP_COL]
        cgm_window: np.ndarray = get_cgm_window(bolus_time, cgm_df)
        features: dict = extract_features_excel(row.to_dict(), cgm_window, carb_median, iob_median, basal_df, idx)
        if features is not None:
            processed_data.append(features)

    elapsed_time: float = time.time() - start_time
    logging.info(f"Procesado {os.path.basename(subject_path)} (Sujeto {idx+1}) en {elapsed_time:.2f} segundos")
    return processed_data

def calculate_stats_for_group(df_final_pd: pd.DataFrame, subjects: list, feature: str = 'bolus') -> tuple:
    """
    Calcula media y desviación estándar para un grupo de sujetos.

    Parámetros:
    -----------
    df_final_pd : pd.DataFrame
        DataFrame con datos procesados
    subjects : list
        Lista de IDs de sujetos
    feature : str, opcional
        Característica para calcular estadísticas (default: 'bolus')

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

def assign_subject_to_group(df_final_pd: pd.DataFrame, subject: int, 
                           train_subjects: list, val_subjects: list, test_subjects: list,
                           train_size: int, val_size: int, test_size: int) -> tuple:
    """
    Asigna un sujeto a un grupo de entrenamiento, validación o prueba basado en balance.

    Parámetros:
    -----------
    df_final_pd : pd.DataFrame
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

def prepare_data_with_scaler(df_final_pd: pd.DataFrame, mask: pd.Series, 
                            columns: list, scaler: StandardScaler, reshape: tuple = None) -> np.ndarray:
    """
    Prepara datos con transformación StandardScaler.

    Parámetros:
    -----------
    df_final_pd : pd.DataFrame
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

def preprocess_data() -> pd.DataFrame:
    """
    Preprocesa todos los datos de sujetos (Excel y XML) y los unifica en un DataFrame.

    Retorna:
    --------
    pd.DataFrame
        DataFrame con todos los datos preprocesados y unificados.
    """
    start_time: float = time.time()
    
    # Procesar datos Excel
    subject_files: list[str] = [f for f in os.listdir(DATA_PATH_SUBJECTS) if f.startswith("Subject") and f.endswith(".xlsx")]
    logging.info(f"\nArchivos de sujetos encontrados ({len(subject_files)}):")
    for f in subject_files:
        logging.info(f)

    excel_data: list[dict] = Parallel(n_jobs=-1)(
        delayed(process_excel_subject)(os.path.join(DATA_PATH_SUBJECTS, f), idx)
        for idx, f in enumerate(subject_files)
    )
    excel_data = [item for sublist in excel_data for item in sublist]
    df_excel: pd.DataFrame = pd.DataFrame(excel_data) if excel_data else pd.DataFrame()

    # Aplicar transform_features a df_excel para procesar cgm_window
    if not df_excel.empty:
        # Asegurar que Timestamp sea Datetime
        df_excel = ensure_timestamp_datetime(df_excel, TIMESTAMP_COL)
        # Añadir columnas requeridas por extract_features (como meal_carbs, etc.)
        required_cols: list[str] = [MEAL_COL, BASAL_COL, TEMP_BASAL_COL]
        for col in required_cols:
            if col not in df_excel.columns:
                df_excel[col] = 0.0
        # Aplicar extract_features y transform_features
        df_excel = extract_features(df_excel, pd.DataFrame())
        df_excel = transform_features(df_excel)

    # Procesar datos XML
    all_xml_dfs: list[pd.DataFrame] = []
    print(f"{OHIO_DATA_DIRS=}")
    for data_dir in OHIO_DATA_DIRS:
        print(f"Procesando directorio: {data_dir}")
        logging.info(f"\nProcesando directorio: {data_dir}")
        data: dict[str, pd.DataFrame] = load_xml_data(data_dir)
        if not data:
            logging.warning(f"No se encontraron datos en {data_dir}")
            continue
        processed: dict[str, pd.DataFrame] = preprocess_xml_bolus_meal(data)
        cgm: pd.DataFrame = data["glucose_level"].copy()
        cgm[TIMESTAMP_COL] = pd.to_datetime(cgm["ts"], format=DATE_FORMAT, errors='coerce')
        bolus_aligned: pd.DataFrame = align_events_to_cgm(cgm, processed["bolus"])
        meal_aligned: pd.DataFrame = align_events_to_cgm(cgm, processed["meal"])
        data['glucose_level'] = preprocess_cgm(data['glucose_level'])
        data['bolus'] = bolus_aligned
        data['meal'] = meal_aligned

        for key in ["glucose_level", "bolus", "meal"]:
            if key in data and TIMESTAMP_COL in data[key].columns:
                data[key] = ensure_timestamp_datetime(data[key], TIMESTAMP_COL)

        df_xml: pd.DataFrame = join_signals(data)
        if GLUCOSE_COL in df_xml.columns:
            df_xml[GLUCOSE_COL] = df_xml[GLUCOSE_COL].astype(float)
        if BOLUS_COL in df_xml.columns:
            df_xml[BOLUS_COL] = df_xml[BOLUS_COL].astype(float)

        df_windows: pd.DataFrame = generate_windows(df_xml, window_size=CONFIG["window_steps"])
        df_features: pd.DataFrame = extract_features(df_windows, data.get('meal'))
        # Renombrar SubjectID a subject_id para consistencia y extraer la parte numérica
        if SUBJECT_ID_COL in df_features.columns:
            df_features = df_features.rename(columns={SUBJECT_ID_COL: "subject_id"})
            # Extraer la parte numérica de subject_id (e.g., "588-ws-training" -> "588")
            df_features["subject_id"] = df_features["subject_id"].str.extract(r"^(\d+)").astype("Int64")
        df_final_xml: pd.DataFrame = transform_features(df_features)
        all_xml_dfs.append(df_final_xml)

    df_xml_combined = pd.concat(all_xml_dfs) if all_xml_dfs else pd.DataFrame()

    # Unificar datos
    if not df_excel.empty and not df_xml_combined.empty:
        # Obtener todas las columnas (unión)
        excel_cols: set[str] = set(df_excel.columns)
        xml_cols: set[str] = set(df_xml_combined.columns)
        all_cols: list[str] = sorted(list(excel_cols.union(xml_cols)))
        logging.info(f"Todas las columnas (unión): {all_cols}")

        # Asegurar que ambos DataFrames tengan las mismas columnas
        for col in all_cols:
            if col not in df_excel.columns:
                if df_xml_combined[col].dtype in [np.float64, np.int64]:
                    df_excel[col] = 0.0
                elif df_xml_combined[col].dtype == 'datetime64[ns]':
                    df_excel[col] = pd.NaT
                else:
                    df_excel[col] = None
            if col not in df_xml_combined.columns:
                if df_excel[col].dtype in [np.float64, np.int64]:
                    df_xml_combined[col] = 0.0
                elif df_excel[col].dtype == 'datetime64[ns]':
                    df_xml_combined[col] = pd.NaT
                else:
                    df_xml_combined[col] = None

        # Reordenar columnas para que coincidan con all_cols
        df_excel = df_excel[all_cols]
        df_xml_combined = df_xml_combined[all_cols]

        # Asegurar tipos consistentes
        logging.info("Tipos en df_excel después de agregar columnas:")
        for col in all_cols:
            logging.info(f"{col}: {df_excel[col].dtype}")
        logging.info("Tipos en df_xml después de agregar columnas:")
        for col in all_cols:
            logging.info(f"{col}: {df_xml_combined[col].dtype}")

        # Identificar columnas que deben ser numéricas (excluir columnas no numéricas)
        numeric_cols = [
            col for col in all_cols
            if col not in ['subject_id', 'Timestamp']
        ]

        # Convertir columnas numéricas a float64
        for col in numeric_cols:
            for df, df_name in [(df_excel, "df_excel"), (df_xml_combined, "df_xml_combined")]:
                if df[col].dtype == 'object':
                    logging.warning(f"Columna '{col}' en {df_name} es de tipo Object: {df[col].dtype}")
                    df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0.0)
                df[col] = df[col].astype(float)

        # Asegurar que subject_id sea Int64
        df_excel['subject_id'] = df_excel['subject_id'].astype("Int64")
        df_xml_combined['subject_id'] = df_xml_combined['subject_id'].astype("Int64")

        # Verificar tipos después de la conversión
        logging.info("Tipos en df_excel después de casteo:")
        for col in all_cols:
            logging.info(f"{col}: {df_excel[col].dtype}")
        logging.info("Tipos en df_xml después de casteo:")
        for col in all_cols:
            logging.info(f"{col}: {df_xml_combined[col].dtype}")

        df_final: pd.DataFrame = pd.concat([df_excel, df_xml_combined], ignore_index=True)
    elif not df_excel.empty:
        df_final = df_excel
    else:
        df_final = df_xml_combined

    logging.info(f"Datos unificados: {df_final.shape}")
    elapsed_time: float = time.time() - start_time
    logging.info(f"Preprocesamiento completo en {elapsed_time:.2f} segundos")
    return df_final

def split_data(df_final: pd.DataFrame) -> tuple:
    """
    Divide los datos siguiendo una estrategia para asegurar distribuciones 
    equilibradas entre los conjuntos de entrenamiento, validación y prueba.

    Parámetros:
    -----------
    df_final : pd.DataFrame
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
        - mean_std_cgm, mean_std_other, mean_std_y: diccionarios con medias y desviaciones estándar
    """
    start_time = time.time()
    logging.info("Iniciando división de datos...")
    
    # Estadísticas por sujeto
    subject_stats = df_final.groupby("subject_id").agg({
        "bolus": ["mean", "std"]
    }).reset_index()
    subject_stats.columns = ["subject_id", "mean_dose", "std_dose"]
    
    # Obtener lista de sujetos ordenados por dosis media
    sorted_subjects = subject_stats.sort_values("mean_dose")["subject_id"].tolist()
    n_subjects = len(sorted_subjects)
    train_size = int(0.8 * n_subjects)
    val_size = int(0.1 * n_subjects)
    test_size = n_subjects - train_size - val_size
    logging.info(f"Total de sujetos: {n_subjects}, Train: {train_size}, Val: {val_size}, Test: {test_size}")

    # Iniciar con sujeto específico para pruebas si está disponible
    test_subjects = [49] if 49 in sorted_subjects else []
    remaining_subjects = [s for s in sorted_subjects if s != 49]
    train_subjects = []
    val_subjects = []

    # Aleatorizar la lista restante y mantener como pandas
    rng = np.random.default_rng(seed=CONST_DEFAULT_SEED)
    rng.shuffle(remaining_subjects)
    df_final_pd = df_final
    logging.info("Datos ya están en formato pandas para procesamiento.")

    # Distribuir sujetos entre los grupos
    for subject in tqdm(remaining_subjects, desc="Asignando sujetos a grupos"):
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
        y_temp = df_final_pd.loc[mask, 'bolus']
        logging.info(f"Post-split {set_name} y: mean = {y_temp.mean()}, std = {y_temp.std()}")

    # Definir columnas para diferentes grupos de características
    cgm_columns = [f'cgm_{i}' for i in range(24)]
    other_features = ['carb_input', 'bg_input', 'insulin_on_board', 'insulin_carb_ratio', 
                      'insulin_sensitivity_factor', 'hour_of_day']

    # Inicializar escaladores
    scaler_cgm = StandardScaler().fit(df_final_pd.loc[train_mask, cgm_columns])
    scaler_other = StandardScaler().fit(df_final_pd.loc[train_mask, other_features])
    scaler_y = StandardScaler().fit(df_final_pd.loc[train_mask, 'bolus'].values.reshape(-1, 1))

    # Preparar datos CGM
    x_cgm_train = prepare_data_with_scaler(df_final_pd, train_mask, cgm_columns, scaler_cgm, reshape=(-1, 24, 1))
    x_cgm_val = prepare_data_with_scaler(df_final_pd, val_mask, cgm_columns, scaler_cgm, reshape=(-1, 24, 1))
    x_cgm_test = prepare_data_with_scaler(df_final_pd, test_mask, cgm_columns, scaler_cgm, reshape=(-1, 24, 1))
    
    # Preparar otras características
    x_other_train = prepare_data_with_scaler(df_final_pd, train_mask, other_features, scaler_other)
    x_other_val = prepare_data_with_scaler(df_final_pd, val_mask, other_features, scaler_other)
    x_other_test = prepare_data_with_scaler(df_final_pd, test_mask, other_features, scaler_other)
    
    # Preparar etiquetas
    y_train = scaler_y.transform(df_final_pd.loc[train_mask, 'bolus'].values.reshape(-1, 1)).flatten()
    y_val = scaler_y.transform(df_final_pd.loc[val_mask, 'bolus'].values.reshape(-1, 1)).flatten()
    y_test = scaler_y.transform(df_final_pd.loc[test_mask, 'bolus'].values.reshape(-1, 1)).flatten()

    # Obtener IDs de sujeto
    x_subject_train = df_final_pd.loc[train_mask, 'subject_id'].values
    x_subject_val = df_final_pd.loc[val_mask, 'subject_id'].values
    x_subject_test = df_final_pd.loc[test_mask, 'subject_id'].values
    
    # Crear diccionarios de medias y desviaciones estándar
    mean_std_cgm = {col: (mean, std) for col, mean, std in zip(cgm_columns, scaler_cgm.mean_, scaler_cgm.scale_)}
    mean_std_other = {col: (mean, std) for col, mean, std in zip(other_features, scaler_other.mean_, scaler_other.scale_)}
    mean_std_y = {'bolus': (scaler_y.mean_[0], scaler_y.scale_[0])}

    # Imprimir resumen
    logging.info(f"Entrenamiento CGM: {x_cgm_train.shape}, Validación CGM: {x_cgm_val.shape}, Prueba CGM: {x_cgm_test.shape}")
    logging.info(f"Entrenamiento Otros: {x_other_train.shape}, Validación Otros: {x_other_val.shape}, Prueba Otros: {x_other_test.shape}")
    logging.info(f"Entrenamiento Subject: {x_subject_train.shape}, Validación Subject: {x_subject_val.shape}, Prueba Subject: {x_subject_test.shape}")
    logging.info(f"Sujetos de prueba: {test_subjects}")

    elapsed_time = time.time() - start_time
    logging.info(f"División de datos completa en {elapsed_time:.2f} segundos")
    
    return (x_cgm_train, x_cgm_val, x_cgm_test,
            x_other_train, x_other_val, x_other_test,
            x_subject_train, x_subject_val, x_subject_test,
            y_train, y_val, y_test, test_subjects,
            mean_std_cgm, mean_std_other, mean_std_y)