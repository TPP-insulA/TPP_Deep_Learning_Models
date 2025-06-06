from collections import defaultdict
from typing import Optional, Union
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

from constants.constants import CONST_DEFAULT_SEED, DATE_FORMAT, TIMESTAMP_COL, SUBJECT_ID_COL, GLUCOSE_COL, BOLUS_COL, MEAL_COL, BASAL_COL, TEMP_BASAL_COL
from config.params import CONFIG_PROCESSING

# Configuración de logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

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
    Carga datos desde archivos XML en el directorio especificado, con validación de sujetos y soporte para múltiples tipos de datos.

    Parámetros:
    -----------
    data_dir : str
        Directorio con archivos XML.

    Retorna:
    --------
    dict[str, pl.DataFrame]
        Diccionario con DataFrames por tipo de dato (glucose_level, bolus, meal, basal, temp_basal, etc.).
    """
    logging.info(f"Cargando datos desde {data_dir}")
    expected_subjects = {
        '2018': ['559-ws-training', '563-ws-training', '570-ws-training', '575-ws-training', '588-ws-training', '591-ws-training'],
        '2020': ['540-ws-training', '544-ws-training', '552-ws-training', '567-ws-training', '584-ws-training', '596-ws-training']
    }
    year = '2018' if '2018' in data_dir else '2020' if '2020' in data_dir else None
    if year is None:
        raise ValueError(f"No se pudo determinar el año del directorio: {data_dir}")
    suffix = '-ws-training' if 'train' in data_dir else '-ws-testing'
    expected_subjects[year] = [s.replace('-ws-training', suffix).replace('-ws-testing', suffix) for s in expected_subjects[year]]
    xml_files = glob.glob(os.path.join(data_dir, "*.xml"))
    found_subjects = [os.path.basename(f).split('.')[0] for f in xml_files]
    missing_subjects = [s for s in expected_subjects[year] if s not in found_subjects]
    if missing_subjects:
        logging.error(f"Faltan datos para sujetos del año {year}: {missing_subjects}")
        if not found_subjects:
            raise ValueError(f"No se encontraron archivos XML en {data_dir}")
    data_dict: dict[str, pl.DataFrame] = {}
    subject_stats = defaultdict(lambda: defaultdict(int))
    expected_types = [
        'glucose_level', 'bolus', 'meal', 'basal', 'temp_basal', 'exercise', 'basis_steps', 'hypo_event',
        'finger_stick', 'sleep', 'work', 'stressors', 'illness', 'basis_heart_rate', 'basis_gsr',
        'basis_skin_temperature', 'basis_air_temperature', 'basis_sleep', 'acceleration'
    ]
    for xml_file in xml_files:
        subject_id = os.path.basename(xml_file).split('.')[0]
        logging.info(f"Procesando SubjectID: {subject_id} (Año {year})")
        try:
            tree = ET.parse(xml_file)
            root = tree.getroot()
            for data_type_elem in root:
                data_type = data_type_elem.tag
                if data_type == 'patient':
                    continue
                if data_type not in expected_types:
                    continue
                records = []
                for event in data_type_elem:
                    record_dict = dict(event.attrib)
                    record_dict[SUBJECT_ID_COL] = subject_id
                    record_dict['Year'] = year
                    records.append(record_dict)
                if records:
                    df = pl.DataFrame(records)
                    if 'value' in df.columns:
                        df = df.with_columns(pl.col('value').cast(pl.Float64))
                    data_dict[data_type] = pl.concat([data_dict.get(data_type, pl.DataFrame()), df])
                    logging.info(f"SubjectID {subject_id}: {data_type}={len(records)} registros")
                    subject_stats[subject_id][data_type] += len(records)
        except Exception as e:
            logging.error(f"Error procesando {xml_file}: {e}")
            continue
    logging.info(f"\nEstadísticas por sujeto (Año {year}):")
    for subject_id in sorted(subject_stats.keys()):
        stats = subject_stats[subject_id]
        stat_str = ", ".join([f"{k}={v}" for k, v in stats.items()])
        logging.info(f"SubjectID {subject_id}: {stat_str}")
    missing_types = [t for t in expected_types if t not in data_dict]
    if missing_types:
        logging.warning(f"Faltan tipos de datos: {missing_types}")
    if len(subject_stats) != len(expected_subjects[year]):
        logging.error(f"Se encontraron datos para {len(subject_stats)}/{len(expected_subjects[year])} sujetos")
    return data_dict

def preprocess_xml_bolus_meal(data: dict[str, pl.DataFrame]) -> dict[str, pl.DataFrame]:
    """
    Preprocesa los datos de bolus, meal, basal, temp_basal, exercise, basis_steps, hypo_event, etc., renombrando columnas y convirtiendo timestamps.

    Parámetros:
    -----------
    data : dict[str, pl.DataFrame]
        Diccionario con DataFrames de datos XML.

    Retorna:
    --------
    dict[str, pl.DataFrame]
        Diccionario con DataFrames preprocesados.
    """
    processed: dict[str, pl.DataFrame] = {}
    # Bolus
    if "bolus" in data:
        bolus = data["bolus"].clone()
        if "dose" in bolus.columns:
            bolus = bolus.rename({"dose": BOLUS_COL})
            bolus = bolus.with_columns(pl.col(BOLUS_COL).cast(pl.Float64))
        if "ts_begin" in bolus.columns:
            bolus = bolus.with_columns(
                pl.col("ts_begin")
                .str.strptime(pl.Datetime(time_unit="us"), DATE_FORMAT)
                .alias(TIMESTAMP_COL)
            )
        valid_bolus = bolus.filter(pl.col(BOLUS_COL).is_not_null() & (pl.col(BOLUS_COL) > 0))
        processed["bolus"] = valid_bolus
        logging.info(f"Eventos bolus válidos: {valid_bolus.height}")

    # Meal
    if "meal" in data:
        meal = data["meal"].clone()
        if "carbs" in meal.columns:
            meal = meal.rename({"carbs": MEAL_COL})
            meal = meal.with_columns(pl.col(MEAL_COL).cast(pl.Float64))
        if "ts" in meal.columns:
            meal = meal.with_columns(
                pl.col("ts")
                .str.strptime(pl.Datetime(time_unit="us"), DATE_FORMAT)
                .alias(TIMESTAMP_COL)
            )
        valid_meal = meal.filter(pl.col(MEAL_COL).is_not_null() & (pl.col(MEAL_COL) > 0))
        processed["meal"] = valid_meal
        logging.info(f"Eventos meal válidos: {valid_meal.height}")

    # Basal
    if "basal" in data:
        basal = data["basal"].clone()
        basal = basal.rename({"value": BASAL_COL}).with_columns(
            pl.col(BASAL_COL).cast(pl.Float64),
            pl.col("ts").str.strptime(pl.Datetime(time_unit="us"), DATE_FORMAT).alias(TIMESTAMP_COL)
        )
        processed["basal"] = basal.filter(pl.col(BASAL_COL).is_not_null())
        logging.info(f"Eventos basal válidos: {processed['basal'].height}")

    # Temp Basal
    if "temp_basal" in data:
        temp_basal = data["temp_basal"].clone()
        temp_basal = temp_basal.rename({"value": TEMP_BASAL_COL}).with_columns(
            pl.col(TEMP_BASAL_COL).cast(pl.Float64),
            pl.col("ts_begin").str.strptime(pl.Datetime(time_unit="us"), DATE_FORMAT).alias(TIMESTAMP_COL)
        )
        processed["temp_basal"] = temp_basal.filter(pl.col(TEMP_BASAL_COL).is_not_null())
        logging.info(f"Eventos temp_basal válidos: {processed['temp_basal'].height}")

    # Exercise
    if "exercise" in data:
        exercise = data["exercise"].clone()
        exercise = exercise.with_columns(
            pl.col("intensity").cast(pl.Float64),
            pl.col("duration").cast(pl.Float64),
            pl.col("ts").str.strptime(pl.Datetime(time_unit="us"), DATE_FORMAT).alias(TIMESTAMP_COL)
        )
        processed["exercise"] = exercise.filter(pl.col("intensity").is_not_null())
        logging.info(f"Eventos exercise válidos: {processed['exercise'].height}")

    # Steps
    if "basis_steps" in data:
        steps = data["basis_steps"].clone()
        steps = steps.rename({"value": "steps"}).with_columns(
            pl.col("steps").cast(pl.Float64),
            pl.col("ts").str.strptime(pl.Datetime(time_unit="us"), DATE_FORMAT).alias(TIMESTAMP_COL)
        )
        processed["basis_steps"] = steps.filter(pl.col("steps").is_not_null())
        logging.info(f"Eventos steps válidos: {processed['basis_steps'].height}")

    # Hypo Event
    if "hypo_event" in data:
        hypo = data["hypo_event"].clone()
        if "ts" in hypo.columns:
            hypo = hypo.with_columns(
                pl.col("ts").str.strptime(pl.Datetime(time_unit="us"), DATE_FORMAT).alias(TIMESTAMP_COL)
            )
        processed["hypo_event"] = hypo
        logging.info(f"Eventos hypo_event válidos: {processed['hypo_event'].height}")

    # Finger Stick
    if "finger_stick" in data:
        finger_stick = data["finger_stick"].clone()
        finger_stick = finger_stick.rename({"value": "finger_stick_bg"}).with_columns(
            pl.col("finger_stick_bg").cast(pl.Float64),
            pl.col("ts").str.strptime(pl.Datetime(time_unit="us"), DATE_FORMAT).alias(TIMESTAMP_COL)
        )
        processed["finger_stick"] = finger_stick.filter(pl.col("finger_stick_bg").is_not_null())
        logging.info(f"Eventos finger_stick válidos: {processed['finger_stick'].height}")

    # Sleep
    if "sleep" in data:
        sleep = data["sleep"].clone()
        if "ts_begin" in sleep.columns and "ts_end" in sleep.columns:
            sleep = sleep.with_columns(
                pl.col("quality").cast(pl.Float64),
                pl.col("ts_begin").str.strptime(pl.Datetime(time_unit="us"), DATE_FORMAT).alias("Timestamp_begin"),
                pl.col("ts_end").str.strptime(pl.Datetime(time_unit="us"), DATE_FORMAT).alias("Timestamp_end")
            )
        elif "ts" in sleep.columns:
            sleep = sleep.with_columns(
                pl.col("quality").cast(pl.Float64),
                pl.col("ts").str.strptime(pl.Datetime(time_unit="us"), DATE_FORMAT).alias(TIMESTAMP_COL)
            )
        processed["sleep"] = sleep.filter(pl.col("quality").is_not_null())
        logging.info(f"Eventos sleep válidos: {processed['sleep'].height}")

    # Work
    if "work" in data:
        work = data["work"].clone()
        if "ts_begin" in work.columns and "ts_end" in work.columns:
            work = work.with_columns(
                pl.col("intensity").cast(pl.Float64),
                pl.col("ts_begin").str.strptime(pl.Datetime(time_unit="us"), DATE_FORMAT).alias("Timestamp_begin"),
                pl.col("ts_end").str.strptime(pl.Datetime(time_unit="us"), DATE_FORMAT).alias("Timestamp_end")
            )
        elif "ts" in work.columns:
            work = work.with_columns(
                pl.col("intensity").cast(pl.Float64),
                pl.col("ts").str.strptime(pl.Datetime(time_unit="us"), DATE_FORMAT).alias(TIMESTAMP_COL)
            )
        processed["work"] = work.filter(pl.col("intensity").is_not_null())
        logging.info(f"Eventos work válidos: {processed['work'].height}")

    # Stressors
    if "stressors" in data:
        stressors = data["stressors"].clone()
        if "ts" in stressors.columns:
            stressors = stressors.with_columns(
                pl.col("ts").str.strptime(pl.Datetime(time_unit="us"), DATE_FORMAT).alias(TIMESTAMP_COL)
            )
        processed["stressors"] = stressors
        logging.info(f"Eventos stressors válidos: {processed['stressors'].height}")

    # Illness
    if "illness" in data:
        illness = data["illness"].clone()
        if "ts" in illness.columns:
            illness = illness.with_columns(
                pl.col("ts").str.strptime(pl.Datetime(time_unit="us"), DATE_FORMAT).alias(TIMESTAMP_COL)
            )
        processed["illness"] = illness
        logging.info(f"Eventos illness válidos: {processed['illness'].height}")

    # Basis Heart Rate
    if "basis_heart_rate" in data:
        heart_rate = data["basis_heart_rate"].clone()
        heart_rate = heart_rate.rename({"value": "heart_rate"}).with_columns(
            pl.col("heart_rate").cast(pl.Float64),
            pl.col("ts").str.strptime(pl.Datetime(time_unit="us"), DATE_FORMAT).alias(TIMESTAMP_COL)
        )
        processed["basis_heart_rate"] = heart_rate.filter(pl.col("heart_rate").is_not_null())
        logging.info(f"Eventos heart_rate válidos: {processed['basis_heart_rate'].height}")

    # Basis GSR
    if "basis_gsr" in data:
        gsr = data["basis_gsr"].clone()
        gsr = gsr.rename({"value": "gsr"}).with_columns(
            pl.col("gsr").cast(pl.Float64),
            pl.col("ts").str.strptime(pl.Datetime(time_unit="us"), DATE_FORMAT).alias(TIMESTAMP_COL)
        )
        processed["basis_gsr"] = gsr.filter(pl.col("gsr").is_not_null())
        logging.info(f"Eventos gsr válidos: {processed['basis_gsr'].height}")

    # Basis Skin Temperature
    if "basis_skin_temperature" in data:
        skin_temp = data["basis_skin_temperature"].clone()
        skin_temp = skin_temp.rename({"value": "skin_temperature"}).with_columns(
            pl.col("skin_temperature").cast(pl.Float64),
            pl.col("ts").str.strptime(pl.Datetime(time_unit="us"), DATE_FORMAT).alias(TIMESTAMP_COL)
        )
        processed["basis_skin_temperature"] = skin_temp.filter(pl.col("skin_temperature").is_not_null())
        logging.info(f"Eventos skin_temperature válidos: {processed['basis_skin_temperature'].height}")

    # Basis Air Temperature
    if "basis_air_temperature" in data:
        air_temp = data["basis_air_temperature"].clone()
        air_temp = air_temp.rename({"value": "air_temperature"}).with_columns(
            pl.col("air_temperature").cast(pl.Float64),
            pl.col("ts").str.strptime(pl.Datetime(time_unit="us"), DATE_FORMAT).alias(TIMESTAMP_COL)
        )
        processed["basis_air_temperature"] = air_temp.filter(pl.col("air_temperature").is_not_null())
        logging.info(f"Eventos air_temperature válidos: {processed['basis_air_temperature'].height}")

    # Basis Sleep
    if "basis_sleep" in data:
        basis_sleep = data["basis_sleep"].clone()
        if "ts_begin" in basis_sleep.columns and "ts_end" in basis_sleep.columns:
            basis_sleep = basis_sleep.with_columns(
                pl.col("quality").cast(pl.Float64),
                pl.col("ts_begin").str.strptime(pl.Datetime(time_unit="us"), DATE_FORMAT).alias("Timestamp_begin"),
                pl.col("ts_end").str.strptime(pl.Datetime(time_unit="us"), DATE_FORMAT).alias("Timestamp_end")
            )
        elif "ts" in basis_sleep.columns:
            basis_sleep = basis_sleep.with_columns(
                pl.col("quality").cast(pl.Float64),
                pl.col("ts").str.strptime(pl.Datetime(time_unit="us"), DATE_FORMAT).alias(TIMESTAMP_COL)
            )
        processed["basis_sleep"] = basis_sleep.filter(pl.col("quality").is_not_null())
        logging.info(f"Eventos basis_sleep válidos: {processed['basis_sleep'].height}")

    # Acceleration
    if "acceleration" in data:
        accel = data["acceleration"].clone()
        accel = accel.rename({"value": "acceleration"}).with_columns(
            pl.col("acceleration").cast(pl.Float64),
            pl.col("ts").str.strptime(pl.Datetime(time_unit="us"), DATE_FORMAT).alias(TIMESTAMP_COL)
        )
        processed["acceleration"] = accel.filter(pl.col("acceleration").is_not_null())
        logging.info(f"Eventos acceleration válidos: {processed['acceleration'].height}")

    return processed

def align_events_to_cgm(cgm_df: pl.DataFrame, event_df: pl.DataFrame, event_time_col: str = TIMESTAMP_COL, tolerance_minutes: int = 5) -> pl.DataFrame:
    """
    Alinea eventos (bolus, meal, etc.) con el timestamp de CGM más cercano dentro de una tolerancia, con mejor logging.

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

    # Asegurar que las columnas de tiempo son datetime
    cgm_df = ensure_timestamp_datetime(cgm_df)
    event_df = ensure_timestamp_datetime(event_df, event_time_col)
    
    # Convertir tolerancia a timedelta
    tolerance = timedelta(minutes=tolerance_minutes)
    
    # Obtener timestamps de CGM
    cgm_times = cgm_df[TIMESTAMP_COL].to_list()
    
    # Función para encontrar el timestamp CGM más cercano
    def find_nearest_cgm(event_time: datetime) -> Optional[datetime]:
        if not cgm_times:
            return None
            
        # Encontrar el timestamp CGM más cercano
        nearest_time = min(cgm_times, key=lambda x: abs(x - event_time))
        time_diff = abs(nearest_time - event_time)
        
        # Verificar si está dentro de la tolerancia
        if time_diff <= tolerance:
            return nearest_time
        return None
    
    # Aplicar la función a cada evento
    aligned_times = []
    for event_time in event_df[event_time_col]:
        nearest_cgm = find_nearest_cgm(event_time)
        if nearest_cgm is not None:
            aligned_times.append(nearest_cgm)
        else:
            aligned_times.append(event_time)
    
    # Crear nuevo DataFrame con tiempos alineados
    aligned_df = event_df.with_columns(
        pl.Series(name=TIMESTAMP_COL, values=aligned_times)
    )
    
    # Filtrar eventos que no se pudieron alinear
    aligned_df = aligned_df.filter(
        pl.col(TIMESTAMP_COL).is_in(cgm_times)
    )
    
    # Calcular estadísticas de alineación
    total_events = len(event_df)
    aligned_events = len(aligned_df)
    lost_events = total_events - aligned_events
    
    if lost_events > 0:
        logging.warning(f"Eventos descartados por estar fuera de tolerancia: {lost_events}")
        logging.warning(f"Eventos sin CGM correspondiente: {lost_events}")
    
    logging.info(f"Eventos alineados: {aligned_events}/{total_events} ({aligned_events/total_events*100:.1f}%)")
    
    return aligned_df

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
    Une señales de CGM, bolus, meal, basal, temp_basal, exercise, steps, hypo_event, etc., alineadas por Timestamp y SubjectID.

    Parámetros:
    -----------
    data : dict[str, pl.DataFrame]
        Diccionario con DataFrames de diferentes tipos de datos.

    Retorna:
    --------
    pl.DataFrame
        DataFrame unificado con todas las señales.
    """
    logging.info("Uniendo señales...")
    cgm_data = data["glucose_level"]
    cgm_data = preprocess_cgm(cgm_data)
    cgm_data = cgm_data.fill_null(strategy="forward").fill_null(strategy="backward")
    
    # Datos base
    bolus_data = data.get("bolus", pl.DataFrame())
    meal_data = data.get("meal", pl.DataFrame())
    basal_data = data.get("basal", pl.DataFrame())
    temp_basal_data = data.get("temp_basal", pl.DataFrame())
    exercise_data = data.get("exercise", pl.DataFrame())
    steps_data = data.get("basis_steps", pl.DataFrame())
    hypo_data = data.get("hypo_event", pl.DataFrame())
    
    # Nuevos datos
    finger_stick_data = data.get("finger_stick", pl.DataFrame())
    sleep_data = data.get("sleep", pl.DataFrame())
    work_data = data.get("work", pl.DataFrame())
    stressors_data = data.get("stressors", pl.DataFrame())
    illness_data = data.get("illness", pl.DataFrame())
    heart_rate_data = data.get("basis_heart_rate", pl.DataFrame())
    gsr_data = data.get("basis_gsr", pl.DataFrame())
    skin_temp_data = data.get("basis_skin_temperature", pl.DataFrame())
    air_temp_data = data.get("basis_air_temperature", pl.DataFrame())
    basis_sleep_data = data.get("basis_sleep", pl.DataFrame())
    acceleration_data = data.get("acceleration", pl.DataFrame())

    # Join bolus y meal
    df = cgm_data.join(bolus_data, on=[TIMESTAMP_COL, SUBJECT_ID_COL], how="left", suffix="_bolus")
    df = df.join(meal_data, on=[TIMESTAMP_COL, SUBJECT_ID_COL], how="left", suffix="_meal")
    
    # Basal
    if not basal_data.is_empty() and TIMESTAMP_COL in basal_data.columns and TIMESTAMP_COL in df.columns:
        df = df.join(basal_data, on=[TIMESTAMP_COL, SUBJECT_ID_COL], how="left", suffix="_basal")
        df = df.with_columns(pl.col(BASAL_COL).fill_null(0.0))
    else:
        df = df.with_columns(pl.lit(0.0).alias(BASAL_COL))
    
    # Temp Basal
    if not temp_basal_data.is_empty() and TIMESTAMP_COL in temp_basal_data.columns and TIMESTAMP_COL in df.columns:
        df = df.join(temp_basal_data, on=[TIMESTAMP_COL, SUBJECT_ID_COL], how="left", suffix="_temp_basal")
        df = df.with_columns(
            pl.col(TEMP_BASAL_COL).fill_null(0.0),
            pl.col(TEMP_BASAL_COL).gt(0).alias("temp_basal_active"),
            pl.when(pl.col("temp_basal_active")).then(pl.col(TEMP_BASAL_COL)).otherwise(pl.col(BASAL_COL)).alias("effective_basal_rate")
        )
    else:
        df = df.with_columns(
            pl.col(BASAL_COL).alias("effective_basal_rate"),
            pl.lit(0.0).alias("temp_basal_active")
        )
    
    # Exercise
    if not exercise_data.is_empty() and TIMESTAMP_COL in exercise_data.columns and TIMESTAMP_COL in df.columns:
        exercise_aligned = align_events_to_cgm(df, exercise_data, tolerance_minutes=15)
        df = df.join(exercise_aligned, on=[TIMESTAMP_COL, SUBJECT_ID_COL], how="left", suffix="_exercise")
        df = df.with_columns(
            pl.col("intensity").fill_null(0.0).alias("exercise_intensity"),
            pl.col("duration").fill_null(0.0).alias("exercise_duration")
        )
    else:
        df = df.with_columns(
            pl.lit(0.0).alias("exercise_intensity"),
            pl.lit(0.0).alias("exercise_duration")
        )
    
    # Steps
    if not steps_data.is_empty() and TIMESTAMP_COL in steps_data.columns and TIMESTAMP_COL in df.columns:
        steps_aligned = align_events_to_cgm(df, steps_data, tolerance_minutes=15)
        df = df.join(steps_aligned, on=[TIMESTAMP_COL, SUBJECT_ID_COL], how="left", suffix="_steps")
        df = df.with_columns(
            pl.col("steps").fill_null(0.0).alias("steps")
        )
    else:
        df = df.with_columns(
            pl.lit(0.0).alias("steps")
        )
    
    # Hypo event (binaria)
    if not hypo_data.is_empty() and TIMESTAMP_COL in hypo_data.columns and TIMESTAMP_COL in df.columns:
        hypo_aligned = align_events_to_cgm(df, hypo_data, tolerance_minutes=15)
        hypo_timestamps = hypo_aligned[TIMESTAMP_COL].unique().to_list()
        df = df.with_columns(
            pl.col(TIMESTAMP_COL).map_elements(lambda x: float(x in hypo_timestamps), return_dtype=pl.Float64).alias("hypo_event")
        )
    else:
        df = df.with_columns(
            pl.lit(0.0).alias("hypo_event")
        )

    # Finger Stick
    if not finger_stick_data.is_empty() and TIMESTAMP_COL in finger_stick_data.columns and TIMESTAMP_COL in df.columns:
        finger_stick_aligned = align_events_to_cgm(df, finger_stick_data, tolerance_minutes=15)
        df = df.join(finger_stick_aligned, on=[TIMESTAMP_COL, SUBJECT_ID_COL], how="left", suffix="_finger_stick")
        df = df.with_columns(
            pl.col("finger_stick_bg").fill_null(0.0)
        )
    else:
        df = df.with_columns(
            pl.lit(0.0).alias("finger_stick_bg")
        )

    # Sleep (binaria)
    if not sleep_data.is_empty() and "Timestamp_begin" in sleep_data.columns and TIMESTAMP_COL in df.columns:
        sleep_aligned = align_events_to_cgm(df, sleep_data, event_time_col="Timestamp_begin", tolerance_minutes=15)
        sleep_timestamps = sleep_aligned[TIMESTAMP_COL].unique().to_list()
        df = df.with_columns(
            pl.col(TIMESTAMP_COL).map_elements(lambda x: float(x in sleep_timestamps), return_dtype=pl.Float64).alias("sleep_event")
        )
    else:
        df = df.with_columns(
            pl.lit(0.0).alias("sleep_event")
        )

    # Work (binaria)
    if not work_data.is_empty() and "Timestamp_begin" in work_data.columns and TIMESTAMP_COL in df.columns:
        work_aligned = align_events_to_cgm(df, work_data, event_time_col="Timestamp_begin", tolerance_minutes=15)
        work_timestamps = work_aligned[TIMESTAMP_COL].unique().to_list()
        df = df.with_columns(
            pl.col(TIMESTAMP_COL).map_elements(lambda x: float(x in work_timestamps), return_dtype=pl.Float64).alias("work_event")
        )
    else:
        df = df.with_columns(
            pl.lit(0.0).alias("work_event")
        )

    # Stressors (binaria)
    if not stressors_data.is_empty() and TIMESTAMP_COL in stressors_data.columns and TIMESTAMP_COL in df.columns:
        stressors_aligned = align_events_to_cgm(df, stressors_data, tolerance_minutes=15)
        stressors_timestamps = stressors_aligned[TIMESTAMP_COL].unique().to_list()
        df = df.with_columns(
            pl.col(TIMESTAMP_COL).map_elements(lambda x: float(x in stressors_timestamps), return_dtype=pl.Float64).alias("stressors_event")
        )
    else:
        df = df.with_columns(
            pl.lit(0.0).alias("stressors_event")
        )

    # Illness (binaria)
    if not illness_data.is_empty() and TIMESTAMP_COL in illness_data.columns and TIMESTAMP_COL in df.columns:
        illness_aligned = align_events_to_cgm(df, illness_data, tolerance_minutes=15)
        illness_timestamps = illness_aligned[TIMESTAMP_COL].unique().to_list()
        df = df.with_columns(
            pl.col(TIMESTAMP_COL).map_elements(lambda x: float(x in illness_timestamps), return_dtype=pl.Float64).alias("illness_event")
        )
    else:
        df = df.with_columns(
            pl.lit(0.0).alias("illness_event")
        )

    # Basis Heart Rate
    if not heart_rate_data.is_empty() and TIMESTAMP_COL in heart_rate_data.columns and TIMESTAMP_COL in df.columns:
        heart_rate_aligned = align_events_to_cgm(df, heart_rate_data, tolerance_minutes=15)
        df = df.join(heart_rate_aligned, on=[TIMESTAMP_COL, SUBJECT_ID_COL], how="left", suffix="_heart_rate")
        df = df.with_columns(
            pl.col("heart_rate").fill_null(0.0)
        )
    else:
        df = df.with_columns(
            pl.lit(0.0).alias("heart_rate")
        )

    # Basis GSR
    if not gsr_data.is_empty() and TIMESTAMP_COL in gsr_data.columns and TIMESTAMP_COL in df.columns:
        gsr_aligned = align_events_to_cgm(df, gsr_data, tolerance_minutes=15)
        df = df.join(gsr_aligned, on=[TIMESTAMP_COL, SUBJECT_ID_COL], how="left", suffix="_gsr")
        df = df.with_columns(
            pl.col("gsr").fill_null(0.0)
        )
    else:
        df = df.with_columns(
            pl.lit(0.0).alias("gsr")
        )

    # Basis Skin Temperature
    if not skin_temp_data.is_empty() and TIMESTAMP_COL in skin_temp_data.columns and TIMESTAMP_COL in df.columns:
        skin_temp_aligned = align_events_to_cgm(df, skin_temp_data, tolerance_minutes=15)
        df = df.join(skin_temp_aligned, on=[TIMESTAMP_COL, SUBJECT_ID_COL], how="left", suffix="_skin_temp")
        df = df.with_columns(
            pl.col("skin_temperature").fill_null(0.0)
        )
    else:
        df = df.with_columns(
            pl.lit(0.0).alias("skin_temperature")
        )

    # Basis Air Temperature
    if not air_temp_data.is_empty() and TIMESTAMP_COL in air_temp_data.columns and TIMESTAMP_COL in df.columns:
        air_temp_aligned = align_events_to_cgm(df, air_temp_data, tolerance_minutes=15)
        df = df.join(air_temp_aligned, on=[TIMESTAMP_COL, SUBJECT_ID_COL], how="left", suffix="_air_temp")
        df = df.with_columns(
            pl.col("air_temperature").fill_null(0.0)
        )
    else:
        df = df.with_columns(
            pl.lit(0.0).alias("air_temperature")
        )

    # Basis Sleep (binaria)
    if not basis_sleep_data.is_empty() and "Timestamp_begin" in basis_sleep_data.columns and TIMESTAMP_COL in df.columns:
        basis_sleep_aligned = align_events_to_cgm(df, basis_sleep_data, event_time_col="Timestamp_begin", tolerance_minutes=15)
        basis_sleep_timestamps = basis_sleep_aligned[TIMESTAMP_COL].unique().to_list()
        df = df.with_columns(
            pl.col(TIMESTAMP_COL).map_elements(lambda x: float(x in basis_sleep_timestamps), return_dtype=pl.Float64).alias("basis_sleep_event")
        )
    else:
        df = df.with_columns(
            pl.lit(0.0).alias("basis_sleep_event")
        )

    # Acceleration
    if not acceleration_data.is_empty() and TIMESTAMP_COL in acceleration_data.columns and TIMESTAMP_COL in df.columns:
        acceleration_aligned = align_events_to_cgm(df, acceleration_data, tolerance_minutes=15)
        df = df.join(acceleration_aligned, on=[TIMESTAMP_COL, SUBJECT_ID_COL], how="left", suffix="_acceleration")
        df = df.with_columns(
            pl.col("acceleration").fill_null(0.0)
        )
    else:
        df = df.with_columns(
            pl.lit(0.0).alias("acceleration")
        )

    # Imputar valores faltantes
    for col in [BOLUS_COL, MEAL_COL, "effective_basal_rate", "exercise_intensity", "exercise_duration", 
                "steps", "hypo_event", "finger_stick_bg", "sleep_event", "work_event", "stressors_event", 
                "illness_event", "heart_rate", "gsr", "skin_temperature", "air_temperature", 
                "basis_sleep_event", "acceleration"]:
        if col in df.columns:
            df = df.with_columns(
                pl.col(col).fill_null(0.0)
            )
    
    logging.info(f"Señales unidas: {df.shape}")
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
    
    return min(iob, CONFIG_PROCESSING["cap_iob"])

def get_cgm_window(bolus_time: datetime, cgm_df: pl.DataFrame, window_hours: int = CONFIG_PROCESSING["window_hours"]) -> np.ndarray:
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

def generate_windows(df: pl.DataFrame, window_size: int = CONFIG_PROCESSING["window_steps"]) -> pl.DataFrame:
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

def compute_glucose_patterns_24h(cgm_values: list[float]) -> dict[str, float]:
    """
    Calcula patrones de glucosa de 24 horas para análisis clínico.
    """
    if not cgm_values or len(cgm_values) == 0:
        return {
            'cgm_mean_24h': 120.0,
            'cgm_std_24h': 0.0,
            'cgm_median_24h': 120.0,
            'cgm_range_24h': 0.0,
            'hypo_episodes_24h': 0,
            'hypo_percentage_24h': 0.0,
            'hyper_episodes_24h': 0,
            'hyper_percentage_24h': 0.0,
            'time_in_range_24h': 100.0,
            'cv_24h': 0.0,
            'mage_24h': 0.0,
            'glucose_trend_24h': 0.0
        }
    
    values_array = np.array(cgm_values)
    
    # Basic statistics
    patterns = {
        'cgm_mean_24h': float(np.mean(values_array)),
        'cgm_std_24h': float(np.std(values_array)),
        'cgm_median_24h': float(np.median(values_array)),
        'cgm_range_24h': float(np.max(values_array) - np.min(values_array))
    }
    
    # Hypoglycemia analysis (< 70 mg/dL)
    hypo_episodes = np.sum(values_array < CONFIG_PROCESSING['hypoglycemia_threshold'])
    patterns['hypo_episodes_24h'] = int(hypo_episodes)
    patterns['hypo_percentage_24h'] = float((hypo_episodes / len(values_array)) * 100)
    
    # Hyperglycemia analysis (> 180 mg/dL)
    hyper_episodes = np.sum(values_array > CONFIG_PROCESSING['hyperglycemia_threshold'])
    patterns['hyper_episodes_24h'] = int(hyper_episodes)
    patterns['hyper_percentage_24h'] = float((hyper_episodes / len(values_array)) * 100)
    
    # Time in Range (70-180 mg/dL)
    in_range = np.sum((values_array >= CONFIG_PROCESSING['tir_lower']) & (values_array <= CONFIG_PROCESSING['tir_upper']))
    patterns['time_in_range_24h'] = float((in_range / len(values_array)) * 100)
    
    # Glucose variability
    if len(values_array) > 1 and patterns['cgm_mean_24h'] > 0:
        patterns['cv_24h'] = float((patterns['cgm_std_24h'] / patterns['cgm_mean_24h']) * 100)
        glucose_changes = np.abs(np.diff(values_array))
        patterns['mage_24h'] = float(np.mean(glucose_changes))
        if len(values_array) >= 3:
            time_points = np.arange(len(values_array))
            slope, _ = np.polyfit(time_points, values_array, 1)
            patterns['glucose_trend_24h'] = float(slope)
        else:
            patterns['glucose_trend_24h'] = 0.0
    else:
        patterns['cv_24h'] = 0.0
        patterns['mage_24h'] = 0.0
        patterns['glucose_trend_24h'] = 0.0
    
    return patterns

def encode_time_cyclical(timestamp: datetime) -> dict[str, float]:
    """
    Codifica características de tiempo cíclicamente usando seno y coseno.
    """
    hour_decimal = timestamp.hour + timestamp.minute / 60.0 + timestamp.second / 3600.0
    hour_normalized = hour_decimal / 24.0
    hour_radians = 2 * np.pi * hour_normalized
    day_of_week = timestamp.weekday() / 7.0
    day_radians = 2 * np.pi * day_of_week
    return {
        'hour_sin': float(np.sin(hour_radians)),
        'hour_cos': float(np.cos(hour_radians)),
        'day_sin': float(np.sin(day_radians)),
        'day_cos': float(np.cos(day_radians)),
        'hour_of_day_normalized': float(hour_normalized),
        'day_of_week_normalized': float(day_of_week)
    }

def compute_enhanced_meal_context(bolus_time: datetime, meal_df: pl.DataFrame, 
                                window_hours: float = 2.0) -> dict[str, Union[float, int]]:
    """
    Calcula características mejoradas del contexto de comidas alrededor del tiempo del bolo.
    """
    if meal_df.is_empty():
        return {
            'meal_carbs': 0.0,
            'meal_time_diff_minutes': 0.0,
            'meal_time_diff_hours': 0.0,
            'has_meal': 0.0,
            'meals_in_window': 0,
            'significant_meal': 0.0,
            'total_carbs_window': 0.0,
            'largest_meal_carbs': 0.0,
            'meal_timing_score': 0.0
        }
    
    start_time = bolus_time - timedelta(hours=window_hours/2)
    end_time = bolus_time + timedelta(hours=window_hours)
    meals_in_window = meal_df.filter(
        (pl.col(TIMESTAMP_COL) >= start_time) & 
        (pl.col(TIMESTAMP_COL) <= end_time)
    )
    
    if meals_in_window.is_empty():
        return {
            'meal_carbs': 0.0,
            'meal_time_diff_minutes': 0.0,
            'meal_time_diff_hours': 0.0,
            'has_meal': 0.0,
            'meals_in_window': 0,
            'significant_meal': 0.0,
            'total_carbs_window': 0.0,
            'largest_meal_carbs': 0.0,
            'meal_timing_score': 0.0
        }
    
    meals_with_diff = meals_in_window.with_columns(
        (pl.col(TIMESTAMP_COL) - bolus_time).abs().alias("time_diff_abs")
    ).sort("time_diff_abs")
    
    closest_meal = meals_with_diff.row(0, named=True)
    time_diff = (closest_meal[TIMESTAMP_COL] - bolus_time).total_seconds()
    meal_carbs = float(closest_meal.get(MEAL_COL, 0.0))
    
    all_meals = meals_in_window.to_dicts()
    total_carbs = sum(float(meal.get(MEAL_COL, 0.0)) for meal in all_meals)
    largest_meal = max((float(meal.get(MEAL_COL, 0.0)) for meal in all_meals), default=0.0)
    
    max_diff_seconds = window_hours * 3600
    timing_score = max(0.0, 1.0 - abs(time_diff) / max_diff_seconds)
    
    return {
        'meal_carbs': meal_carbs,
        'meal_time_diff_minutes': float(time_diff / 60.0),
        'meal_time_diff_hours': float(time_diff / 3600.0),
        'has_meal': 1.0,
        'meals_in_window': len(all_meals),
        'significant_meal': 1.0 if meal_carbs > CONFIG_PROCESSING['significant_meal_threshold'] else 0.0,
        'total_carbs_window': total_carbs,
        'largest_meal_carbs': largest_meal,
        'meal_timing_score': timing_score
    }

def compute_clinical_risk_indicators(cgm_values: list[float], current_iob: float = 0.0) -> dict[str, float]:
    """
    Calcula indicadores de riesgo clínico en tiempo real.
    """
    if not cgm_values:
        return {
            'current_hypo_risk': 0.0,
            'current_hyper_risk': 0.0,
            'glucose_rate_of_change': 0.0,
            'glucose_acceleration': 0.0,
            'stability_score': 1.0,
            'iob_risk_factor': 0.0
        }
    
    current_glucose = cgm_values[-1] if cgm_values else 120.0
    
    hypo_risk = 1.0 if current_glucose < CONFIG_PROCESSING['hypoglycemia_threshold'] else 0.0
    hyper_risk = 1.0 if current_glucose > CONFIG_PROCESSING['hyperglycemia_threshold'] else 0.0
    
    if len(cgm_values) >= 2:
        glucose_rate = cgm_values[-1] - cgm_values[-2]
    else:
        glucose_rate = 0.0
    
    if len(cgm_values) >= 3:
        glucose_acceleration = (cgm_values[-1] - cgm_values[-2]) - (cgm_values[-2] - cgm_values[-3])
    else:
        glucose_acceleration = 0.0
    
    if len(cgm_values) >= 6:
        recent_values = cgm_values[-6:]
        stability_score = max(0.0, 1.0 - (np.std(recent_values) / 50.0))
    else:
        stability_score = 1.0
    
    iob_risk = min(1.0, current_iob / 5.0)
    
    return {
        'current_hypo_risk': hypo_risk,
        'current_hyper_risk': hyper_risk,
        'glucose_rate_of_change': float(glucose_rate),
        'glucose_acceleration': float(glucose_acceleration),
        'stability_score': float(stability_score),
        'iob_risk_factor': float(iob_risk)
    }

def extract_features(df: pl.DataFrame, meal_df: pl.DataFrame, extended_cgm_df: pl.DataFrame = None) -> pl.DataFrame:
    """
    Extrae características mejoradas para DRL, incluyendo contexto de basal, ejercicio, pasos, hipo_event.
    """
    logging.info("Extrayendo características mejoradas para DRL...")
    enhanced_rows = []
    for row in df.iter_rows(named=True):
        # Features base
        cgm_window = row.get("cgm_window", [])
        basic_stats = {
            "glucose_last": cgm_window[-1] if cgm_window else 120.0,
            "glucose_mean": float(np.mean(cgm_window)) if cgm_window else 120.0,
            "glucose_std": float(np.std(cgm_window)) if cgm_window else 0.0,
            "glucose_min": float(np.min(cgm_window)) if cgm_window else 120.0,
            "glucose_max": float(np.max(cgm_window)) if cgm_window else 120.0,
        }
        # Contexto de tiempo
        time_features = encode_time_cyclical(row[TIMESTAMP_COL])
        # Contexto de comidas
        meal_context = compute_enhanced_meal_context(row[TIMESTAMP_COL], meal_df)
        # Riesgo clínico
        iob = row.get("insulin_on_board", 0.0)
        risk_indicators = compute_clinical_risk_indicators(cgm_window, iob)
        # Contexto de ejercicio y pasos
        exercise_features = {
            "exercise_intensity": row.get("exercise_intensity", 0.0),
            "exercise_duration": row.get("exercise_duration", 0.0),
            "exercise_in_window": row.get("exercise_in_window", 0.0),
            "steps_in_window": row.get("steps_in_window", 0.0),
        }
        # Hipo-evento
        hypo_features = {
            "hypo_event_in_window": row.get("hypo_event_in_window", 0.0)
        }
        # Basal
        basal_features = {
            "effective_basal_rate": row.get("effective_basal_rate", 0.0),
            "temp_basal_active": row.get("temp_basal_active", 0.0)
        }
        # Patrones de glucosa de 24h si hay datos extendidos
        glucose_patterns = {}
        if extended_cgm_df is not None:
            subject_id = row[SUBJECT_ID_COL]
            ts = row[TIMESTAMP_COL]
            window_start = ts - timedelta(hours=24)
            extended_window = extended_cgm_df.filter(
                (pl.col(SUBJECT_ID_COL) == subject_id) &
                (pl.col(TIMESTAMP_COL) >= window_start) &
                (pl.col(TIMESTAMP_COL) <= ts)
            ).sort(TIMESTAMP_COL)
            if extended_window.height >= 288:  # 24 horas a 5 min
                glucose_patterns = compute_glucose_patterns_24h(extended_window[GLUCOSE_COL].to_list())
            else:
                glucose_patterns = compute_glucose_patterns_24h([])
        
        enhanced_row = {
            **row,
            **basic_stats,
            **time_features,
            **meal_context,
            **risk_indicators,
            **exercise_features,
            **hypo_features,
            **basal_features,
            **glucose_patterns
        }
        enhanced_rows.append(enhanced_row)
    enhanced_df = pl.DataFrame(enhanced_rows)
    # Normalización y log1p
    for col in [BOLUS_COL, MEAL_COL, "effective_basal_rate", "exercise_intensity", "exercise_duration", "steps_in_window"]:
        if col in enhanced_df.columns:
            enhanced_df = enhanced_df.with_columns(
                pl.col(col).log1p().alias(f"{col}_log1p")
            )
    # Normalizar features binarios
    for col in ["exercise_in_window", "hypo_event_in_window", "temp_basal_active", "has_meal", "significant_meal"]:
        if col in enhanced_df.columns:
            enhanced_df = enhanced_df.with_columns(
                pl.col(col).cast(pl.Float64)
            )
    logging.info(f"Características DRL extraídas. Forma: {enhanced_df.shape}")
    return enhanced_df

def transform_features(df: pl.DataFrame) -> pl.DataFrame:
    """
    Aplica transformaciones mejoradas incluyendo transformaciones logarítmicas, normalización y expansión de características.
    """
    logging.info("Aplicando transformaciones mejoradas...")
    
    # Log1p transformations
    log_transform_cols = [
        BOLUS_COL, "carb_input", MEAL_COL, "insulin_on_board",
        "total_carbs_window", "largest_meal_carbs"
    ]
    
    for col in log_transform_cols:
        if col in df.columns:
            df = df.with_columns(
                pl.col(col).log1p().alias(f"{col}_log1p")
            )

    # Normalize percentage features
    percentage_cols = [
        "hypo_percentage_24h", "hyper_percentage_24h", "time_in_range_24h", "cv_24h"
    ]
    
    for col in percentage_cols:
        if col in df.columns:
            df = df.with_columns(
                (pl.col(col) / 100.0).alias(f"{col}_normalized")
            )
    
    # Normalize time features
    if "meal_time_diff_hours" in df.columns:
        df = df.with_columns(
            (pl.col("meal_time_diff_hours") / 24.0).alias("meal_time_diff_normalized")
        )
    
    # Normalize glucose-related features
    glucose_norm_cols = [
        ("cgm_mean_24h", 200.0),
        ("cgm_std_24h", 100.0),
        ("cgm_median_24h", 200.0),
        ("cgm_range_24h", 300.0),
        ("mage_24h", 50.0),
        ("glucose_trend_24h", 10.0)
    ]
    
    for col, norm_factor in glucose_norm_cols:
        if col in df.columns:
            df = df.with_columns(
                (pl.col(col) / norm_factor).alias(f"{col}_normalized")
            )
    
    # Expand CGM window to individual columns
    if "cgm_window" in df.columns:
        window_size = CONFIG_PROCESSING["window_steps"]
        for i in range(window_size):
            df = df.with_columns(
                pl.col("cgm_window").list.get(i, null_on_oob=True)
                .fill_null(120.0)
                .alias(f"cgm_{i}")
            )
        df = df.drop("cgm_window")

    # Create risk composite scores
    if all(col in df.columns for col in ["current_hypo_risk", "stability_score", "iob_risk_factor"]):
        df = df.with_columns(
            (pl.col("current_hypo_risk") + pl.col("iob_risk_factor") * 0.5).alias("composite_hypo_risk")
        )
    
    if all(col in df.columns for col in ["current_hyper_risk", "glucose_rate_of_change"]):
        df = df.with_columns(
            (pl.col("current_hyper_risk") + (pl.col("glucose_rate_of_change") / 10.0).clip(0, 1)).alias("composite_hyper_risk")
        )
    
    logging.info(f"Transformaciones mejoradas completadas. Forma final: {df.shape}")
    return df

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
    iob = np.clip(iob, 0, CONFIG_PROCESSING["cap_iob"])
    
    hour_of_day: float = bolus_time.hour / 23.0
    
    bg_input: float = row["bgInput"] if row["bgInput"] is not None else cgm_window[-1]
    bg_input = max(bg_input, 50.0)
    bg_input = np.clip(bg_input, 0, CONFIG_PROCESSING["cap_bg"])
    
    normal: float = row["normal"] if row["normal"] is not None else 0.0
    normal = np.clip(normal, 0, CONFIG_PROCESSING["cap_normal"])
    
    isf_custom: float = 50.0 if normal <= 0 else (bg_input - 100) / normal
    isf_custom = np.clip(isf_custom, 10, 100)
    
    carb_input: float = row["carbInput"] if row["carbInput"] is not None else 0.0
    carb_input = carb_median if carb_input == 0 else carb_input
    carb_input = np.clip(carb_input, 0, CONFIG_PROCESSING["cap_carb"])
    
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
        df_excel = ensure_timestamp_datetime(df_excel, TIMESTAMP_COL)
        df_excel = df_excel.with_columns(
            pl.col("cgm_window").list.eval(pl.element().cast(pl.Float64)).alias("cgm_window")
        )
        required_cols: list[str] = [MEAL_COL, BASAL_COL, TEMP_BASAL_COL]
        for col in required_cols:
            if col not in df_excel.columns:
                df_excel = df_excel.with_columns(pl.lit(0.0).cast(pl.Float64).alias(col))
        df_excel = extract_features(df_excel, pl.DataFrame(), extended_cgm_df=None)
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

        df_windows: pl.DataFrame = generate_windows(df_xml, window_size=CONFIG_PROCESSING["window_steps"])
        df_features: pl.DataFrame = extract_features(df_windows, data.get('meal'), extended_cgm_df=df_xml)
        if SUBJECT_ID_COL in df_features.columns:
            df_features = df_features.rename({SUBJECT_ID_COL: "subject_id"})
            df_features = df_features.with_columns(
                pl.col("subject_id")
                .str.extract(r"^(\d+)")
                .cast(pl.Int64)
                .alias("subject_id")
            )
        df_final_xml: pl.DataFrame = transform_features(df_features)
        all_xml_dfs.append(df_final_xml)

    df_xml_combined = pl.concat(all_xml_dfs) if all_xml_dfs else pl.DataFrame()
    
    print(f"Datos XML combinados: {df_xml_combined.shape}")

    # Unificar datos
    if not df_excel.is_empty() and not df_xml_combined.is_empty():
        excel_cols: set[str] = set(df_excel.columns)
        xml_cols: set[str] = set(df_xml_combined.columns)
        all_cols: list[str] = sorted(list(excel_cols.union(xml_cols)))
        logging.info(f"Todas las columnas (unión): {all_cols}")

        # Asegurar que las columnas estén presentes en ambos DataFrames
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

        df_excel = df_excel.select(all_cols)
        df_xml_combined = df_xml_combined.select(all_cols)

        # Definir columnas numéricas (excluyendo 'subject_id' y 'Timestamp')
        numeric_cols = [
            col for col in all_cols
            if col not in ['subject_id', 'Timestamp']
        ]

        # Asegurar que 'subject_id' sea Int64 en ambos DataFrames
        df_excel = df_excel.with_columns(pl.col('subject_id').cast(pl.Int64))
        df_xml_combined = df_xml_combined.with_columns(pl.col('subject_id').cast(pl.Int64))

        # Manejar columnas de tipo List u Object y castear a Float64 para columnas numéricas
        for col in all_cols:
            for df, df_name in [(df_excel, "df_excel"), (df_xml_combined, "df_xml_combined")]:
                if isinstance(df[col].dtype, pl.List):
                    logging.warning(f"Columna '{col}' en {df_name} es de tipo List: {df[col].dtype}")
                    df = df.with_columns(
                        pl.col(col).list.first().cast(pl.Float64).fill_null(0.0).alias(col)
                    )
                elif df[col].dtype == pl.Object:
                    logging.warning(f"Columna '{col}' en {df_name} es de tipo Object: {df[col].dtype}")
                    df = df.with_columns(
                        pl.col(col)
                        .cast(pl.Utf8)
                        .cast(pl.Float64, strict=False)
                        .fill_null(0.0)
                        .alias(col)
                    )

        # Castear todas las columnas numéricas a Float64
        for col in numeric_cols:
            df_excel = df_excel.with_columns(pl.col(col).cast(pl.Float64).fill_null(0.0).alias(col))
            df_xml_combined = df_xml_combined.with_columns(pl.col(col).cast(pl.Float64).fill_null(0.0).alias(col))

        # Verificar tipos antes de la concatenación
        logging.info("Tipos en df_excel antes de concatenar:")
        for col in all_cols:
            logging.info(f"{col}: {df_excel[col].dtype}")
        logging.info("Tipos en df_xml_combined antes de concatenar:")
        for col in all_cols:
            logging.info(f"{col}: {df_xml_combined[col].dtype}")

        # Concatenar los DataFrames
        df_final: pl.DataFrame = pl.concat([df_excel, df_xml_combined])
    elif not df_excel.is_empty():
        df_final = df_excel
    else:
        df_final = df_xml_combined

    logging.info(f"Datos unificados: {df_final.shape}")
    elapsed_time: float = time.time() - start_time
    logging.info(f"Preprocesamiento completo en {elapsed_time:.2f} segundos")
    return df_final

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