from typing import Union
import polars as pl
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

def load_excel_data(subject_path: str) -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]:
    """
    Carga datos de un sujeto desde un archivo Excel con hojas CGM, Bolus y Basal.

    Parámetros:
    -----------
    subject_path : str
        Ruta al archivo Excel del sujeto.

    Retorna:
    --------
    tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]
        Tupla con (cgm_df, bolus_df, basal_df), donde cada elemento es un DataFrame
        o None si hubo error en la carga.
    """
    try:
        cgm_df: pl.DataFrame = pl.read_excel(subject_path, sheet_name="CGM")
        bolus_df: pl.DataFrame = pl.read_excel(subject_path, sheet_name="Bolus")
        try:
            basal_df: pl.DataFrame = pl.read_excel(subject_path, sheet_name="Basal")
        except Exception:
            basal_df = None

        # Conversión de fechas con precisión en microsegundos
        cgm_df = cgm_df.with_columns(
            pl.col("date").cast(pl.Datetime(time_unit="us")).alias(TIMESTAMP_COL)
        )
        cgm_df = cgm_df.sort(TIMESTAMP_COL).rename({"mg/dl": GLUCOSE_COL})
        bolus_df = bolus_df.with_columns(
            pl.col("date").cast(pl.Datetime(time_unit="us")).alias(TIMESTAMP_COL)
        )
        if basal_df is not None:
            basal_df = basal_df.with_columns(
                pl.col("date").cast(pl.Datetime(time_unit="us")).alias(TIMESTAMP_COL)
            )
        return cgm_df, bolus_df, basal_df
    except Exception as e:
        logging.error(f"Error al cargar {os.path.basename(subject_path)}: {e}")
        return None, None, None

def load_xml_data(data_dir: str) -> dict[str, pl.DataFrame]:
    """
    Carga datos desde archivos XML en el directorio especificado.

    Parámetros:
    -----------
    data_dir : str
        Directorio con archivos XML.

    Retorna:
    --------
    dict[str, pl.DataFrame]
        Diccionario con DataFrames por tipo de dato (glucose_level, bolus, meal, etc.).
    """
    logging.info(f"Cargando datos desde {data_dir}")
    data_dict: dict[str, pl.DataFrame] = {}
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
                    df: pl.DataFrame = pl.DataFrame(records)
                    data_dict[data_type] = df
        except Exception as e:
            logging.error(f"Error procesando {xml_file}: {e}")
            continue
    return data_dict

def preprocess_xml_bolus_meal(data: dict[str, pl.DataFrame]) -> dict[str, pl.DataFrame]:
    """
    Preprocesa los datos de bolus y meal de XML, renombrando columnas y convirtiendo timestamps.

    Parámetros:
    -----------
    data : dict[str, pl.DataFrame]
        Diccionario con DataFrames de datos XML.

    Retorna:
    --------
    dict[str, pl.DataFrame]
        Diccionario con DataFrames preprocesados de bolus y meal.
    """
    processed: dict[str, pl.DataFrame] = {}
    # Procesar bolus
    if "bolus" in data:
        bolus: pl.DataFrame = data["bolus"].clone()
        if "dose" in bolus.columns:
            bolus = bolus.rename({"dose": BOLUS_COL})
        if "ts_begin" in bolus.columns:
            bolus = bolus.with_columns(
                pl.col("ts_begin")
                .str.strptime(pl.Datetime(time_unit="us"), DATE_FORMAT)
                .alias(TIMESTAMP_COL)
            )
        processed["bolus"] = bolus
    # Procesar meal
    if "meal" in data:
        meal: pl.DataFrame = data["meal"].clone()
        if "carbs" in meal.columns:
            meal = meal.rename({"carbs": MEAL_COL})
        if "ts" in meal.columns:
            meal = meal.with_columns(
                pl.col("ts")
                .str.strptime(pl.Datetime(time_unit="us"), DATE_FORMAT)
                .alias(TIMESTAMP_COL)
            )
        processed["meal"] = meal
    return processed

def align_events_to_cgm(cgm_df: pl.DataFrame, event_df: pl.DataFrame, event_time_col: str = TIMESTAMP_COL, tolerance_minutes: int = 5) -> pl.DataFrame:
    """
    Alinea eventos (bolus, meal, etc.) con el timestamp de CGM más cercano dentro de una tolerancia.

    Parámetros:
    -----------
    cgm_df : pl.DataFrame
        DataFrame con datos CGM.
    event_df : pl.DataFrame
        DataFrame con eventos a alinear.
    event_time_col : str, opcional
        Columna de timestamp en event_df (default: "Timestamp").
    tolerance_minutes : int, opcional
        Tolerancia en minutos para la alineación (default: 5).

    Retorna:
    --------
    pl.DataFrame
        DataFrame de eventos con timestamps alineados.
    """
    if cgm_df.is_empty() or event_df.is_empty():
        return event_df

    cgm_times: np.ndarray = cgm_df[TIMESTAMP_COL].to_numpy()
    aligned_rows: list[dict] = []
    for row in event_df.iter_rows(named=True):
        event_time = row[event_time_col]
        if not isinstance(event_time, np.datetime64):
            try:
                event_time = np.datetime64(event_time)
            except Exception:
                continue
        idx: int = np.argmin(np.abs(cgm_times - event_time))
        nearest_cgm_time = cgm_times[idx]
        diff_minutes: float = np.abs((nearest_cgm_time - event_time) / np.timedelta64(1, 'm'))
        if diff_minutes <= tolerance_minutes:
            row[TIMESTAMP_COL] = nearest_cgm_time
            aligned_rows.append(row)
    return pl.DataFrame(aligned_rows)

def preprocess_cgm(cgm: pl.DataFrame) -> pl.DataFrame:
    """
    Preprocesa los datos de CGM, convirtiendo la columna de timestamp.

    Parámetros:
    -----------
    cgm : pl.DataFrame
        DataFrame con datos CGM.

    Retorna:
    --------
    pl.DataFrame
        DataFrame con la columna de timestamp convertida.
    """
    if "ts" in cgm.columns:
        cgm = cgm.with_columns(
            pl.col("ts")
            .str.strptime(pl.Datetime(time_unit="us"), DATE_FORMAT)
            .alias(TIMESTAMP_COL)
        )
    return cgm

def join_signals(data: dict[str, pl.DataFrame]) -> pl.DataFrame:
    """
    Une señales de CGM, bolus, meal, basal y temp_basal alineadas por Timestamp y SubjectID.

    Parámetros:
    -----------
    data : dict[str, pl.DataFrame]
        Diccionario con DataFrames de diferentes tipos de datos.

    Retorna:
    --------
    pl.DataFrame
        DataFrame unificado con todas las señales.
    """
    df: pl.DataFrame = data["glucose_level"].clone()
    # Join bolus
    if "bolus" in data and not data["bolus"].is_empty():
        df = df.join(
            data["bolus"].select([TIMESTAMP_COL, BOLUS_COL, SUBJECT_ID_COL]),
            on=[TIMESTAMP_COL, SUBJECT_ID_COL],
            how="left"
        )
    else:
        df = df.with_columns(pl.lit(None).alias(BOLUS_COL))
    # Join meal
    if "meal" in data and not data["meal"].is_empty():
        df = df.join(
            data["meal"].select([TIMESTAMP_COL, MEAL_COL, SUBJECT_ID_COL]),
            on=[TIMESTAMP_COL, SUBJECT_ID_COL],
            how="left"
        )
    else:
        df = df.with_columns(pl.lit(None).alias(MEAL_COL))
    # Join basal
    if "basal" in data and "dose" in data["basal"].columns:
        basal: pl.DataFrame = data["basal"].with_columns(
            pl.col("dose").cast(pl.Float64)
        ).rename({"dose": BASAL_COL})
        if "ts_begin" in basal.columns:
            basal = basal.with_columns(pl.col("ts_begin").str.strptime(pl.Datetime, DATE_FORMAT).alias(TIMESTAMP_COL))
        df = df.join(
            basal.select([TIMESTAMP_COL, SUBJECT_ID_COL, BASAL_COL]),
            on=[TIMESTAMP_COL, SUBJECT_ID_COL],
            how="left"
        )
    else:
        df = df.with_columns(pl.lit(0.0).alias(BASAL_COL))
    # Join temp_basal
    if "temp_basal" in data and "dose" in data["temp_basal"].columns:
        temp: pl.DataFrame = data["temp_basal"].with_columns(
            pl.col("dose").cast(pl.Float64)
        ).rename({"dose": TEMP_BASAL_COL})
        if "ts_begin" in temp.columns:
            temp = temp.with_columns(pl.col("ts_begin").str.strptime(pl.Datetime, DATE_FORMAT).alias(TIMESTAMP_COL))
        df = df.join(
            temp.select([TIMESTAMP_COL, SUBJECT_ID_COL, TEMP_BASAL_COL]),
            on=[TIMESTAMP_COL, SUBJECT_ID_COL],
            how="left"
        )
    else:
        df = df.with_columns(pl.lit(0.0).alias(TEMP_BASAL_COL))
    return df

def ensure_timestamp_datetime(df: pl.DataFrame, col: str = TIMESTAMP_COL) -> pl.DataFrame:
    """
    Convierte la columna de timestamp a pl.Datetime, tolerando diferentes formatos.

    Parámetros:
    -----------
    df : pl.DataFrame
        DataFrame con la columna de timestamp.
    col : str, opcional
        Nombre de la columna de timestamp (default: "Timestamp").

    Retorna:
    --------
    pl.DataFrame
        DataFrame con la columna de timestamp convertida a pl.Datetime.
    """
    if col in df.columns:
        if df[col].dtype == pl.Object:
            df = df.with_columns(
                pl.col(col).map_elements(
                    lambda x: str(x) if x is not None else None,
                    return_dtype=pl.Utf8
                ).alias(col)
            )
        df = df.with_columns(
            pl.col(col).cast(pl.Datetime(time_unit="us"))
        )
    return df

def calculate_iob(bolus_time: datetime, basal_df: pl.DataFrame, half_life_hours: float = 4.0) -> float:
    """
    Calcula la insulina activa en el cuerpo (IOB).

    Parámetros:
    -----------
    bolus_time : datetime
        Tiempo del bolo de insulina.
    basal_df : pl.DataFrame
        DataFrame con datos de insulina basal.
    half_life_hours : float, opcional
        Vida media de la insulina en horas (default: 4.0).

    Retorna:
    --------
    float
        Cantidad de insulina activa en el organismo.
    """
    if basal_df is None or basal_df.is_empty():
        return 0.0
    
    iob: float = 0.0
    for row in basal_df.iter_rows(named=True):
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

def get_cgm_window(bolus_time: datetime, cgm_df: pl.DataFrame, window_hours: int = CONFIG["window_hours"]) -> np.ndarray:
    """
    Obtiene la ventana de datos CGM para un tiempo de bolo específico.

    Parámetros:
    -----------
    bolus_time : datetime
        Tiempo del bolo de insulina.
    cgm_df : pl.DataFrame
        DataFrame con datos CGM.
    window_hours : int, opcional
        Horas de la ventana de datos (default: 2).

    Retorna:
    --------
    np.ndarray
        Ventana de datos CGM o None si no hay suficientes datos.
    """
    window_start: datetime = bolus_time - timedelta(hours=window_hours)
    window: pl.DataFrame = cgm_df.filter(
        (pl.col(TIMESTAMP_COL) >= window_start) & (pl.col(TIMESTAMP_COL) <= bolus_time)
    ).sort(TIMESTAMP_COL).tail(24)
    
    if window.height < 24:
        return None
    return window.get_column(GLUCOSE_COL).to_numpy()

def generate_windows(df: pl.DataFrame, window_size: int = CONFIG["window_steps"]) -> pl.DataFrame:
    """
    Genera ventanas de CGM de tamaño fijo antes de cada evento bolus.

    Parámetros:
    -----------
    df : pl.DataFrame
        DataFrame con datos de CGM y bolus.
    window_size : int, opcional
        Cantidad de pasos en la ventana (default: 24 para 2 horas con datos cada 5 min).

    Retorna:
    --------
    pl.DataFrame
        DataFrame con ventanas generadas.
    """
    windows: list[dict] = []
    bolus_events: pl.DataFrame = df.filter(pl.col(BOLUS_COL) > 0)
    for row in bolus_events.iter_rows(named=True):
        ts: datetime = row[TIMESTAMP_COL]
        subject_id: str = row[SUBJECT_ID_COL]
        cgm_window: pl.DataFrame = df.filter(
            (pl.col(SUBJECT_ID_COL) == subject_id) &
            (pl.col(TIMESTAMP_COL) <= ts) &
            (pl.col(TIMESTAMP_COL) > ts - timedelta(minutes=window_size*5))
        ).sort(TIMESTAMP_COL)
        if cgm_window.height == window_size:
            windows.append({
                SUBJECT_ID_COL: subject_id,
                TIMESTAMP_COL: ts,
                "cgm_window": cgm_window[GLUCOSE_COL].to_list(),
                BOLUS_COL: row[BOLUS_COL],
                MEAL_COL: row.get(MEAL_COL, 0.0)
            })
    return pl.DataFrame(windows)

def calculate_medians(bolus_df: pl.DataFrame, basal_df: pl.DataFrame) -> tuple[float, float]:
    """
    Calcula valores medianos para imputación de datos faltantes.

    Parámetros:
    -----------
    bolus_df : pl.DataFrame
        DataFrame con datos de bolos de insulina.
    basal_df : pl.DataFrame
        DataFrame con datos de insulina basal.

    Retorna:
    --------
    tuple[float, float]
        Tupla con (carb_median, iob_median).
    """
    non_zero_carbs: pl.Series = bolus_df.filter(pl.col("carbInput") > 0).get_column("carbInput")
    carb_median: float = non_zero_carbs.median() if len(non_zero_carbs) > 0 else 10.0
    
    iob_values: list[float] = []
    for row in bolus_df.iter_rows(named=True):
        iob: float = calculate_iob(row[TIMESTAMP_COL], basal_df)
        iob_values.append(iob)
    
    non_zero_iob: list[float] = [iob for iob in iob_values if iob > 0]
    iob_median: float = np.median(non_zero_iob) if non_zero_iob else 0.5
    
    return carb_median, iob_median

def extract_features(df: pl.DataFrame, meal_df: pl.DataFrame) -> pl.DataFrame:
    """
    Extrae características de las ventanas CGM y agrega información de meal.

    Parámetros:
    -----------
    df : pl.DataFrame
        DataFrame con ventanas CGM y datos de bolus.
    meal_df : pl.DataFrame
        DataFrame con datos de meals.

    Retorna:
    --------
    pl.DataFrame
        DataFrame con características extraídas.
    """
    df = df.with_columns([
        pl.col("cgm_window").list.get(-1).alias("glucose_last"),
        pl.col("cgm_window").list.mean().alias("glucose_mean"),
        pl.col("cgm_window").list.std().alias("glucose_std"),
        pl.col("cgm_window").list.min().alias("glucose_min"),
        pl.col("cgm_window").list.max().alias("glucose_max"),
        ((pl.col(TIMESTAMP_COL).dt.hour() * 60 + pl.col(TIMESTAMP_COL).dt.minute()) / (24 * 60)).alias("hour_of_day"),
        pl.when(pl.col("cgm_window").list.len() > 0)
        .then(pl.col("cgm_window").list.get(-1))
        .otherwise(0.0)
        .alias("bg_input"),
        pl.lit(0.0).alias("insulin_on_board")
    ])

    required_cols: list[str] = ["carb_input", BASAL_COL, TEMP_BASAL_COL, "meal_time_diff", "meal_time_diff_hours", "has_meal", "meals_in_window"]
    for col in required_cols:
        default: Union[float, int] = 0.0 if "time" not in col and "meals" not in col else 0
        df = df.with_columns(pl.lit(default).cast(pl.Float64).alias(col))

    df = df.with_columns([
        (pl.col("meal_time_diff") / 60.0).cast(pl.Float64).alias("meal_time_diff_hours"),
        pl.when(pl.col("meal_time_diff") > 0).then(1.0).otherwise(0.0).cast(pl.Float64).alias("has_meal")
    ])

    if not meal_df.is_empty():
        meal_df = meal_df.with_columns(pl.col(TIMESTAMP_COL).cast(pl.Datetime(time_unit="us")))
        matched: list[dict] = []
        for row in df.iter_rows(named=True):
            bolus_time = row.get(TIMESTAMP_COL, None)
            if bolus_time is None:
                continue
            start: datetime = bolus_time
            end: datetime = bolus_time + timedelta(hours=1)
            meals: pl.DataFrame = meal_df.filter((pl.col(TIMESTAMP_COL) >= start) & (pl.col(TIMESTAMP_COL) <= end))
            if meals.height > 0:
                meals = meals.with_columns((pl.col(TIMESTAMP_COL) - bolus_time).alias("time_diff")).sort("time_diff")
                closest: dict = meals.to_dicts()[0]
                meal_time_diff = closest.get("time_diff")
                matched.append({
                    TIMESTAMP_COL: bolus_time,
                    MEAL_COL: float(closest.get(MEAL_COL, 0.0)),
                    "meal_time_diff": float(meal_time_diff.total_seconds() / 60.0 if meal_time_diff else 0.0),
                    "meals_in_window": float(meals.height)
                })
            else:
                matched.append({
                    TIMESTAMP_COL: bolus_time,
                    MEAL_COL: 0.0,
                    "meal_time_diff": 0.0,
                    "meals_in_window": 0.0
                })

        meals_info: pl.DataFrame = pl.DataFrame(matched)
        if TIMESTAMP_COL in meals_info.columns:
            meals_info = meals_info.with_columns(pl.col(TIMESTAMP_COL).cast(pl.Datetime(time_unit="us")))
            df = df.with_columns(pl.col(TIMESTAMP_COL).cast(pl.Datetime(time_unit="us")))
            df = df.join(meals_info, on=TIMESTAMP_COL, how="left")
            df = df.with_columns([
                pl.col(MEAL_COL).cast(pl.Float64).fill_null(0.0),
                pl.col("meal_time_diff").cast(pl.Float64).fill_null(0.0),
                pl.col("meals_in_window").cast(pl.Float64).fill_null(0.0),
                (pl.col("meal_time_diff") / 60.0).cast(pl.Float64).alias("meal_time_diff_hours"),
                pl.when(pl.col("meals_in_window") > 0).then(1.0).otherwise(0.0).cast(pl.Float64).alias("has_meal")
            ])

    return df

def transform_features(df: pl.DataFrame) -> pl.DataFrame:
    """
    Aplica transformaciones finales a las características para entrenamiento.

    Parámetros:
    -----------
    df : pl.DataFrame
        DataFrame con características extraídas.

    Retorna:
    --------
    pl.DataFrame
        DataFrame con características transformadas.
    """
    for col in ["glucose_last", "glucose_mean", "glucose_std", "glucose_min", "glucose_max"]:
        df = df.with_columns(
            pl.col(col).cast(pl.Float64).log1p().alias(f"{col}_log1p")
        )

    df = df.with_columns([
        pl.when(pl.col("cgm_window").list.len() > 1)
        .then(pl.col("cgm_window").list.get(-1) - pl.col("cgm_window").list.get(0))
        .otherwise(0.0)
        .cast(pl.Float64)
        .alias("glucose_trend"),
        pl.col("cgm_window").list.std().fill_null(0.0).cast(pl.Float64).alias("glucose_variability")
    ])

    df = df.with_columns([
        pl.col("glucose_trend").cast(pl.Float64).alias("cgm_trend"),
        pl.col("glucose_variability").cast(pl.Float64).alias("cgm_std")
    ])

    for col in [BOLUS_COL, "carb_input", MEAL_COL, "insulin_on_board"]:
        df = df.with_columns(
            pl.col(col).cast(pl.Float64).log1p().alias(f"{col}_log1p")
        )

    optional_features: list[str] = ['work_intensity', 'sleep_quality', 'activity_intensity']
    for feature in optional_features:
        if feature in df.columns:
            df = df.with_columns([
                pl.when(pl.col(feature).is_not_null())
                .then(pl.col(feature) / CONFIG[f'max_{feature}'])
                .otherwise(None)
                .cast(pl.Float64)
                .alias(feature)
            ])

    window_size: int = CONFIG["window_steps"]
    cgm_cols: list[str] = [f"cgm_{i}" for i in range(window_size)]
    df = df.with_columns([
        pl.col("cgm_window").list.get(i).cast(pl.Float64).alias(f"cgm_{i}")
        for i in range(window_size)
    ])

    for col in cgm_cols:
        df = df.with_columns(
            pl.col(col).cast(pl.Float64).log1p().alias(f"{col}_log")
        )

    return df.drop("cgm_window")

def extract_features_excel(row: dict, cgm_window: np.ndarray, carb_median: float, iob_median: float, basal_df: pl.DataFrame, idx: int) -> dict:
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
    basal_df : pl.DataFrame
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
    
    bg_input: float = row["bgInput"] if row["bgInput"] is not None else cgm_window[-1]
    bg_input = max(bg_input, 50.0)
    bg_input = np.clip(bg_input, 0, CONFIG["cap_bg"])
    
    normal: float = row["normal"] if row["normal"] is not None else 0.0
    normal = np.clip(normal, 0, CONFIG["cap_normal"])
    
    isf_custom: float = 50.0 if normal <= 0 else (bg_input - 100) / normal
    isf_custom = np.clip(isf_custom, 10, 100)
    
    carb_input: float = row["carbInput"] if row["carbInput"] is not None else 0.0
    carb_input = carb_median if carb_input == 0 else carb_input
    carb_input = np.clip(carb_input, 0, CONFIG["cap_carb"])
    
    insulin_carb_ratio: float = row["insulinCarbRatio"] if row["insulinCarbRatio"] is not None else 10.0
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
    for row in tqdm(bolus_df.iter_rows(named=True), total=len(bolus_df), desc=f"Procesando {os.path.basename(subject_path)}"):
        bolus_time: datetime = row[TIMESTAMP_COL]
        cgm_window: np.ndarray = get_cgm_window(bolus_time, cgm_df)
        features: dict = extract_features_excel(row, cgm_window, carb_median, iob_median, basal_df, idx)
        if features is not None:
            processed_data.append(features)

    elapsed_time: float = time.time() - start_time
    logging.info(f"Procesado {os.path.basename(subject_path)} (Sujeto {idx+1}) en {elapsed_time:.2f} segundos")
    return processed_data

def calculate_stats_for_group(df_final_pd: pl.DataFrame, subjects: list, feature: str = 'bolus') -> tuple:
    """
    Calcula media y desviación estándar para un grupo de sujetos.

    Parámetros:
    -----------
    df_final_pd : pl.DataFrame
        DataFrame con datos procesados en formato pandas
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

def assign_subject_to_group(df_final_pd: pl.DataFrame, subject: int, 
                           train_subjects: list, val_subjects: list, test_subjects: list,
                           train_size: int, val_size: int, test_size: int) -> tuple:
    """
    Asigna un sujeto a un grupo de entrenamiento, validación o prueba basado en balance.

    Parámetros:
    -----------
    df_final_pd : pl.DataFrame
        DataFrame con datos procesados en formato pandas
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
                            columns: list, scaler: StandardScaler, reshape: tuple = None) -> np.ndarray:
    """
    Prepara datos con transformación StandardScaler.

    Parámetros:
    -----------
    df_final_pd : pl.DataFrame
        DataFrame con datos procesados en formato pandas
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

def preprocess_data() -> pl.DataFrame:
    """
    Preprocesa todos los datos de sujetos (Excel y XML) y los unifica en un DataFrame.

    Retorna:
    --------
    pl.DataFrame
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
    df_excel: pl.DataFrame = pl.DataFrame(excel_data) if excel_data else pl.DataFrame()

    # Aplicar transform_features a df_excel para procesar cgm_window
    if not df_excel.is_empty():
        # Asegurar que Timestamp sea Datetime con microsegundos
        df_excel = ensure_timestamp_datetime(df_excel, TIMESTAMP_COL)
        # Asegurar que GLUCOSE_COL sea Float64 en cgm_window
        df_excel = df_excel.with_columns(
            pl.col("cgm_window").list.eval(pl.element().cast(pl.Float64)).alias("cgm_window")
        )
        # Añadir columnas requeridas por extract_features (como meal_carbs, etc.)
        required_cols: list[str] = [MEAL_COL, BASAL_COL, TEMP_BASAL_COL]
        for col in required_cols:
            if col not in df_excel.columns:
                df_excel = df_excel.with_columns(pl.lit(0.0).cast(pl.Float64).alias(col))
        # Aplicar extract_features y transform_features
        df_excel = extract_features(df_excel, pl.DataFrame())
        df_excel = transform_features(df_excel)

    # Procesar datos XML
    all_xml_dfs: list[pl.DataFrame] = []
    print(f"{OHIO_DATA_DIRS=}")
    for data_dir in OHIO_DATA_DIRS:
        print(f"Procesando directorio: {data_dir}")
        logging.info(f"\nProcesando directorio: {data_dir}")
        data: dict[str, pl.DataFrame] = load_xml_data(data_dir)
        if not data:
            logging.warning(f"No se encontraron datos en {data_dir}")
            continue
        processed: dict[str, pl.DataFrame] = preprocess_xml_bolus_meal(data)
        cgm: pl.DataFrame = data["glucose_level"].with_columns(
            pl.col("ts").str.strptime(pl.Datetime(time_unit="us"), DATE_FORMAT).alias(TIMESTAMP_COL)
        )
        bolus_aligned: pl.DataFrame = align_events_to_cgm(cgm, processed["bolus"])
        meal_aligned: pl.DataFrame = align_events_to_cgm(cgm, processed["meal"])
        data['glucose_level'] = preprocess_cgm(data['glucose_level'])
        data['bolus'] = bolus_aligned
        data['meal'] = meal_aligned

        for key in ["glucose_level", "bolus", "meal"]:
            if key in data and TIMESTAMP_COL in data[key].columns:
                data[key] = ensure_timestamp_datetime(data[key], TIMESTAMP_COL)

        df_xml: pl.DataFrame = join_signals(data)
        if GLUCOSE_COL in df_xml.columns:
            df_xml = df_xml.with_columns(pl.col(GLUCOSE_COL).cast(pl.Float64))
        if BOLUS_COL in df_xml.columns:
            df_xml = df_xml.with_columns(pl.col(BOLUS_COL).cast(pl.Float64))

        df_windows: pl.DataFrame = generate_windows(df_xml, window_size=CONFIG["window_steps"])
        df_features: pl.DataFrame = extract_features(df_windows, data.get('meal'))
        # Renombrar SubjectID a subject_id para consistencia y extraer la parte numérica
        if SUBJECT_ID_COL in df_features.columns:
            df_features = df_features.rename({SUBJECT_ID_COL: "subject_id"})
            # Extraer la parte numérica de subject_id (e.g., "588-ws-training" -> "588")
            df_features = df_features.with_columns(
                pl.col("subject_id")
                .str.extract(r"^(\d+)")
                .cast(pl.Int64)
                .alias("subject_id")
            )
        df_final_xml: pl.DataFrame = transform_features(df_features)
        all_xml_dfs.append(df_final_xml)

    df_xml_combined = pl.concat(all_xml_dfs) if all_xml_dfs else pl.DataFrame()

    # Unificar datos
    if not df_excel.is_empty() and not df_xml_combined.is_empty():
        # Obtener todas las columnas (unión)
        excel_cols: set[str] = set(df_excel.columns)
        xml_cols: set[str] = set(df_xml_combined.columns)
        all_cols: list[str] = sorted(list(excel_cols.union(xml_cols)))
        logging.info(f"Todas las columnas (unión): {all_cols}")

        # Asegurar que ambos DataFrames tengan las mismas columnas
        for col in all_cols:
            if col not in df_excel.columns:
                col_dtype = df_xml_combined[col].dtype
                if col_dtype in [pl.Float64, pl.Int64]:
                    df_excel = df_excel.with_columns(pl.lit(0.0).cast(pl.Float64).alias(col))
                elif col_dtype == pl.Datetime:
                    df_excel = df_excel.with_columns(pl.lit(None).cast(pl.Datetime(time_unit="us")).alias(col))
                else:
                    df_excel = df_excel.with_columns(pl.lit(None).cast(col_dtype).alias(col))
            if col not in df_xml_combined.columns:
                col_dtype = df_excel[col].dtype
                if col_dtype in [pl.Float64, pl.Int64]:
                    df_xml_combined = df_xml_combined.with_columns(pl.lit(0.0).cast(pl.Float64).alias(col))
                elif col_dtype == pl.Datetime:
                    df_xml_combined = df_xml_combined.with_columns(pl.lit(None).cast(pl.Datetime(time_unit="us")).alias(col))
                else:
                    df_xml_combined = df_xml_combined.with_columns(pl.lit(None).cast(col_dtype).alias(col))

        # Reordenar columnas para que coincidan con all_cols
        df_excel = df_excel.select(all_cols)
        df_xml_combined = df_xml_combined.select(all_cols)

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

        # Verificar si hay columnas de tipo List
        for col in all_cols:
            for df, df_name in [(df_excel, "df_excel"), (df_xml_combined, "df_xml_combined")]:
                if isinstance(df[col].dtype, pl.List):
                    logging.error(f"Columna '{col}' en {df_name} es de tipo List: {df[col].dtype}")
                    df = df.with_columns(
                        pl.col(col).list.first().cast(pl.Float64).fill_null(0.0).alias(col)
                    )

        # Convertir columnas numéricas a Float64, manejar columnas Object si existen
        for col in numeric_cols:
            for df, df_name in [(df_excel, "df_excel"), (df_xml_combined, "df_xml_combined")]:
                if isinstance(df[col].dtype, pl.List):
                    logging.warning(f"Columna '{col}' en {df_name} es de tipo List: {df[col].dtype}")
                    df = df.with_columns(
                        pl.col(col).list.first().cast(pl.Float64).fill_null(0.0).alias(col)
                    )
                if df[col].dtype == pl.Object:
                    logging.warning(f"Columna '{col}' en {df_name} es de tipo Object: {df[col].dtype}")
                    df = df.with_columns(
                        pl.col(col)
                        .cast(pl.Utf8)
                        .cast(pl.Float64, strict=False)
                        .fill_null(0.0)
                        .alias(col)
                    )
                df = df.with_columns(pl.col(col).cast(pl.Float64).alias(col))

        # Asegurar que subject_id sea Int64
        df_excel = df_excel.with_columns(pl.col('subject_id').cast(pl.Int64))
        df_xml_combined = df_xml_combined.with_columns(pl.col('subject_id').cast(pl.Int64))

        # Verificar tipos después de la conversión
        logging.info("Tipos en df_excel después de casteo:")
        for col in all_cols:
            logging.info(f"{col}: {df_excel[col].dtype}")
        logging.info("Tipos en df_xml después de casteo:")
        for col in all_cols:
            logging.info(f"{col}: {df_xml_combined[col].dtype}")

        df_final: pl.DataFrame = pl.concat([df_excel, df_xml_combined])
    elif not df_excel.is_empty():
        df_final = df_excel
    else:
        df_final = df_xml_combined

    logging.info(f"Datos unificados: {df_final.shape}")
    elapsed_time: float = time.time() - start_time
    logging.info(f"Preprocesamiento completo en {elapsed_time:.2f} segundos")
    return df_final


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
    logging.info("Iniciando división de datos...")
    
    # Estadísticas por sujeto
    subject_stats = df_final.group_by("subject_id").agg([
        pl.col("bolus").mean().alias("mean_dose"),
        pl.col("bolus").std().alias("std_dose")
    ])
    
    # Obtener lista de sujetos ordenados por dosis media
    sorted_subjects = subject_stats.sort("mean_dose").get_column("subject_id").to_list()
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

    # Aleatorizar la lista restante y convertir a pandas para cálculos
    rng = np.random.default_rng(seed=CONST_DEFAULT_SEED)
    rng.shuffle(remaining_subjects)
    df_final_pd = df_final.to_pandas()
    logging.info("Datos convertidos a pandas para procesamiento.")

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
            scaler_cgm, scaler_other, scaler_y)

def calculate_stats_for_group(df: pl.DataFrame, subjects: list, feature: str = 'bolus') -> tuple:
    """
    Calcula media y desviación estándar para un grupo de sujetos usando Polars.

    Parámetros:
    -----------
    df : pl.DataFrame
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
        return 0.0, 0.0
    filtered_df = df.filter(pl.col('subject_id').is_in(subjects))
    mean_val = filtered_df[feature].mean()
    std_val = filtered_df[feature].std()
    return mean_val if mean_val is not None else 0.0, std_val if std_val is not None else 0.0

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

def assign_subject_to_group(df: pl.DataFrame, subject: int, 
                           train_subjects: list, val_subjects: list, test_subjects: list,
                           train_size: int, val_size: int, test_size: int) -> tuple:
    """
    Asigna un sujeto a un grupo de entrenamiento, validación o prueba basado en balance.

    Parámetros:
    -----------
    df : pl.DataFrame
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
    train_mean, train_std = calculate_stats_for_group(df, train_subjects)
    val_mean, val_std = calculate_stats_for_group(df, val_subjects)
    test_mean, test_std = calculate_stats_for_group(df, test_subjects)
    
    # Calculate potential stats if subject is added to each group
    train_temp = train_subjects + [subject]
    val_temp = val_subjects + [subject]
    test_temp = test_subjects + [subject]
    
    train_mean_new, train_std_new = calculate_stats_for_group(df, train_temp)
    val_mean_new, val_std_new = calculate_stats_for_group(df, val_temp)
    test_mean_new, test_std_new = calculate_stats_for_group(df, test_temp)
    
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

def standardize_columns(df: pl.DataFrame, columns: list, mean_std_dict: dict = None) -> tuple:
    """
    Estandariza columnas específicas de un DataFrame usando Polars.

    Parámetros:
    -----------
    df : pl.DataFrame
        DataFrame con datos a estandarizar
    columns : list
        Lista de columnas a estandarizar
    mean_std_dict : dict, opcional
        Diccionario con medias y desviaciones estándar precalculadas (default: None)

    Retorna:
    --------
    tuple
        (DataFrame estandarizado, diccionario con medias y desviaciones estándar)
    """
    if mean_std_dict is None:
        mean_std_dict = {}
        # Calcular media y desviación estándar para cada columna
        for col in columns:
            mean_val = df[col].mean()
            std_val = df[col].std()
            mean_std_dict[col] = (mean_val if mean_val is not None else 0.0, 
                                 std_val if std_val is not None else 1.0)
    
    # Estandarizar columnas
    exprs = []
    for col in columns:
        mean_val, std_val = mean_std_dict[col]
        # Evitar división por cero
        std_val = 1.0 if std_val == 0 else std_val
        expr = ((pl.col(col) - mean_val) / std_val).alias(col)
        exprs.append(expr)
    
    df_standardized = df.with_columns(exprs)
    return df_standardized, mean_std_dict

def split_data(df_final: pl.DataFrame) -> tuple:
    """
    Divide los datos siguiendo una estrategia para asegurar distribuciones 
    equilibradas entre los conjuntos de entrenamiento, validación y prueba, usando Polars.

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
        - mean_std_cgm, mean_std_other, mean_std_y: diccionarios con medias y desviaciones estándar
    """
    start_time = time.time()
    logging.info("Iniciando división de datos...")
    
    # Estadísticas por sujeto
    subject_stats = df_final.group_by("subject_id").agg([
        pl.col("bolus").mean().alias("mean_dose"),
        pl.col("bolus").std().alias("std_dose")
    ])
    
    # Obtener lista de sujetos ordenados por dosis media
    sorted_subjects = subject_stats.sort("mean_dose").get_column("subject_id").to_list()
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

    # Aleatorizar la lista restante
    rng = np.random.default_rng(seed=CONST_DEFAULT_SEED)
    rng.shuffle(remaining_subjects)
    logging.info("Sujetos aleatorizados para asignación.")

    # Distribuir sujetos entre los grupos
    for subject in tqdm(remaining_subjects, desc="Asignando sujetos a grupos"):
        train_subjects, val_subjects, test_subjects = assign_subject_to_group(
            df_final, subject, train_subjects, val_subjects, test_subjects,
            train_size, val_size, test_size
        )

    # Dividir el DataFrame en conjuntos
    df_train = df_final.filter(pl.col('subject_id').is_in(train_subjects))
    df_val = df_final.filter(pl.col('subject_id').is_in(val_subjects))
    df_test = df_final.filter(pl.col('subject_id').is_in(test_subjects))

    # Mostrar estadísticas post-división
    for set_name, df_set in [("Train", df_train), ("Val", df_val), ("Test", df_test)]:
        y_mean = df_set['bolus'].mean()
        y_std = df_set['bolus'].std()
        logging.info(f"Post-split {set_name} y: mean = {y_mean}, std = {y_std}")

    # Definir columnas para diferentes grupos de características
    cgm_columns = [f'cgm_{i}' for i in range(24)]
    other_features = ['carb_input', 'bg_input', 'insulin_on_board', 'insulin_carb_ratio', 
                      'insulin_sensitivity_factor', 'hour_of_day']

    # Estandarizar datos CGM
    df_train_cgm, mean_std_cgm = standardize_columns(df_train, cgm_columns)
    df_val_cgm, _ = standardize_columns(df_val, cgm_columns, mean_std_cgm)
    df_test_cgm, _ = standardize_columns(df_test, cgm_columns, mean_std_cgm)

    # Convertir a NumPy y reshape
    x_cgm_train = df_train_cgm.select(cgm_columns).to_numpy().reshape(-1, 24, 1)
    x_cgm_val = df_val_cgm.select(cgm_columns).to_numpy().reshape(-1, 24, 1)
    x_cgm_test = df_test_cgm.select(cgm_columns).to_numpy().reshape(-1, 24, 1)

    # Estandarizar otras características
    df_train_other, mean_std_other = standardize_columns(df_train, other_features)
    df_val_other, _ = standardize_columns(df_val, other_features, mean_std_other)
    df_test_other, _ = standardize_columns(df_test, other_features, mean_std_other)

    # Convertir a NumPy
    x_other_train = df_train_other.select(other_features).to_numpy()
    x_other_val = df_val_other.select(other_features).to_numpy()
    x_other_test = df_test_other.select(other_features).to_numpy()

    # Estandarizar etiquetas (bolus)
    df_train_y, mean_std_y = standardize_columns(df_train, ['bolus'])
    df_val_y, _ = standardize_columns(df_val, ['bolus'], mean_std_y)
    df_test_y, _ = standardize_columns(df_test, ['bolus'], mean_std_y)

    # Convertir etiquetas a NumPy y flatten
    y_train = df_train_y['bolus'].to_numpy()
    y_val = df_val_y['bolus'].to_numpy()
    y_test = df_test_y['bolus'].to_numpy()

    # Obtener IDs de sujeto
    x_subject_train = df_train['subject_id'].to_numpy()
    x_subject_val = df_val['subject_id'].to_numpy()
    x_subject_test = df_test['subject_id'].to_numpy()
    
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
