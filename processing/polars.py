from collections import defaultdict
import re
from typing import Dict, List, Optional, Union
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

from custom.printer import print_debug, print_info
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
from config.params import CONFIG_PROCESSING, USE_EXCEL_DATA

# Configuración de logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def load_excel_data(subject_path: str) -> tuple[Optional[pl.DataFrame], Optional[pl.DataFrame], Optional[pl.DataFrame]]:
    """
    Carga datos de un sujeto desde un archivo Excel con hojas CGM, Bolus y Basal.

    Parámetros:
    -----------
    subject_path : str
        Ruta al archivo Excel del sujeto.

    Retorna:
    --------
    tuple[Optional[pl.DataFrame], Optional[pl.DataFrame], Optional[pl.DataFrame]]
        Tupla con (cgm_df, bolus_df, basal_df), donde cada elemento es un DataFrame
        o None si hubo error en la carga.
    """
    cgm_df, bolus_df, basal_df = None, None, None
    
    try:
        # Verificar si el archivo existe y es accesible
        if not os.path.exists(subject_path) or not os.access(subject_path, os.R_OK):
            logging.error(f"Archivo no existe o no es accesible: {subject_path}")
            return None, None, None
            
        # Verificar el tamaño del archivo
        file_size = os.path.getsize(subject_path)
        if file_size == 0:
            logging.error(f"Archivo vacío: {subject_path}")
            return None, None, None
            
        # Intenta cargar la hoja CGM
        try:
            cgm_df = pl.read_excel(subject_path, sheet_name="CGM")
            if cgm_df.is_empty():
                logging.warning(f"Hoja CGM vacía en {os.path.basename(subject_path)}")
                cgm_df = None
            else:
                # Verificar columnas requeridas
                if "date" not in cgm_df.columns or "mg/dl" not in cgm_df.columns:
                    missing_cols = []
                    if "date" not in cgm_df.columns: missing_cols.append("date")
                    if "mg/dl" not in cgm_df.columns: missing_cols.append("mg/dl")
                    logging.warning(f"Faltan columnas en hoja CGM de {os.path.basename(subject_path)}: {missing_cols}")
                    logging.warning(f"Columnas disponibles: {cgm_df.columns}")
                    cgm_df = None
                else:
                    # Procesamiento de CGM
                    cgm_df = cgm_df.with_columns(
                        pl.col("date").cast(pl.Datetime(time_unit="us")).alias(TIMESTAMP_COL)
                    )
                    cgm_df = cgm_df.sort(TIMESTAMP_COL).rename({"mg/dl": GLUCOSE_COL})
        except Exception as e:
            logging.error(f"Error cargando hoja CGM de {os.path.basename(subject_path)}: {e}")
            cgm_df = None
            
        # Intenta cargar la hoja Bolus
        try:
            bolus_df = pl.read_excel(subject_path, sheet_name="Bolus")
            if bolus_df.is_empty():
                logging.warning(f"Hoja Bolus vacía en {os.path.basename(subject_path)}")
                bolus_df = None
            else:
                # Verificar columnas requeridas
                if "date" not in bolus_df.columns:
                    logging.warning(f"Falta columna 'date' en hoja Bolus de {os.path.basename(subject_path)}")
                    logging.warning(f"Columnas disponibles: {bolus_df.columns}")
                    bolus_df = None
                else:
                    # Procesamiento de Bolus
                    bolus_df = bolus_df.with_columns(
                        pl.col("date").cast(pl.Datetime(time_unit="us")).alias(TIMESTAMP_COL)
                    )
        except Exception as e:
            logging.error(f"Error cargando hoja Bolus de {os.path.basename(subject_path)}: {e}")
            bolus_df = None
            
        # Intenta cargar la hoja Basal (opcional)
        try:
            basal_df = pl.read_excel(subject_path, sheet_name="Basal")
            if not basal_df.is_empty() and "date" in basal_df.columns:
                basal_df = basal_df.with_columns(
                    pl.col("date").cast(pl.Datetime(time_unit="us")).alias(TIMESTAMP_COL)
                )
            else:
                basal_df = None
        except Exception:
            basal_df = None
            
        # Verificar si se pudo cargar al menos una de las hojas principales
        if cgm_df is None and bolus_df is None:
            logging.error(f"No se pudo cargar ninguna hoja útil de {os.path.basename(subject_path)}")
            return None, None, None
            
        return cgm_df, bolus_df, basal_df
        
    except Exception as e:
        logging.error(f"Error general al cargar {os.path.basename(subject_path)}: {e}")
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
                    record_dict[SUBJECT_ID_COL] = extract_numeric_id(subject_id)
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

def join_signals(cgm_df: pl.DataFrame, 
                bolus_df: pl.DataFrame, 
                meal_df: pl.DataFrame,
                physiological_df: Optional[pl.DataFrame] = None) -> pl.DataFrame:
    """Une las diferentes señales temporales usando join_asof"""
    
    def standardize_timestamp_column(df: pl.DataFrame) -> pl.DataFrame:
        """Estandariza el nombre de la columna de timestamp y su tipo de datos"""
        if 'Timestamp' in df.columns:
            df = df
        elif 'ts' in df.columns:
            df = df.rename({'ts': 'Timestamp'})
        elif 'Time' in df.columns:
            df = df.rename({'Time': 'Timestamp'})
        else:
            raise ValueError(f"No se encontró columna de timestamp en DataFrame. Columnas disponibles: {df.columns}")
        
        # Asegurar que Timestamp es datetime
        if df['Timestamp'].dtype != pl.Datetime:
            try:
                # Intentar diferentes formatos de fecha
                formats = [
                    "%Y-%m-%d %H:%M:%S",
                    "%d-%m-%Y %H:%M:%S",
                    "%m-%d-%Y %H:%M:%S",
                    "%Y/%m/%d %H:%M:%S",
                    "%d/%m/%Y %H:%M:%S"
                ]
                
                for fmt in formats:
                    try:
                        df = df.with_columns(
                            pl.col('Timestamp').str.strptime(pl.Datetime, fmt)
                        )
                        break
                    except Exception:
                        continue
                else:
                    raise ValueError("No se pudo convertir Timestamp a datetime con ningún formato conocido")
                    
            except Exception as e:
                logging.error(f"No se pudo convertir Timestamp a datetime: {e}")
                raise
        
        return df
    
    # Estandarizar nombres de columnas de timestamp y tipos de datos
    cgm_df = standardize_timestamp_column(cgm_df)
    bolus_df = standardize_timestamp_column(bolus_df)
    meal_df = standardize_timestamp_column(meal_df)
    
    if physiological_df is not None:
        physiological_df = standardize_timestamp_column(physiological_df)
    
    # Asegurar que los DataFrames estén ordenados por timestamp y SubjectID
    cgm_df = cgm_df.sort(['SubjectID', 'Timestamp'])
    bolus_df = bolus_df.sort(['SubjectID', 'Timestamp'])
    meal_df = meal_df.sort(['SubjectID', 'Timestamp'])
    
    if physiological_df is not None:
        physiological_df = physiological_df.sort(['SubjectID', 'Timestamp'])
    
    # Unir bolus y meal primero
    try:
        # Asegurar que la columna value existe en cgm_df
        if 'value' not in cgm_df.columns:
            logging.error("Columna 'value' no encontrada en cgm_df")
            raise ValueError("Columna 'value' no encontrada en cgm_df")
            
        # Asegurar que la columna bolus existe en bolus_df
        if 'bolus' not in bolus_df.columns:
            logging.error("Columna 'bolus' no encontrada en bolus_df")
            raise ValueError("Columna 'bolus' no encontrada en bolus_df")
        
        # Unir con bolus sin sufijo para preservar el nombre de la columna
        joined_df = cgm_df.join_asof(
            bolus_df,
            on='Timestamp',
            by='SubjectID',
            tolerance='5m'
        )
        
        # Verificar que las columnas críticas están presentes
        if 'value' not in joined_df.columns:
            logging.error("Columna 'value' perdida después de join con bolus")
            raise ValueError("Columna 'value' perdida después de join con bolus")
        if 'bolus' not in joined_df.columns:
            logging.error("Columna 'bolus' perdida después de join con bolus")
            raise ValueError("Columna 'bolus' perdida después de join con bolus")
        
        # Unir con meal usando sufijo específico
        joined_df = joined_df.join_asof(
            meal_df,
            on='Timestamp',
            by='SubjectID',
            tolerance='5m',
            suffix='_meal'
        )
        
        # Verificar nuevamente las columnas críticas
        if 'value' not in joined_df.columns:
            logging.error("Columna 'value' perdida después de join con meal")
            raise ValueError("Columna 'value' perdida después de join con meal")
        if 'bolus' not in joined_df.columns:
            logging.error("Columna 'bolus' perdida después de join con meal")
            raise ValueError("Columna 'bolus' perdida después de join con meal")
        
        # Unir datos fisiológicos si existen
        if physiological_df is not None:
            # Filtrar valores nulos antes de unir
            physiological_df = physiological_df.filter(pl.col('value').is_not_null())
            
            # Renombrar columna value a physiological_value antes de unir
            physiological_df = physiological_df.rename({'value': 'physiological_value'})
            
            # Unir datos fisiológicos usando sufijo específico
            joined_df = joined_df.join_asof(
                physiological_df,
                on='Timestamp',
                by='SubjectID',
                tolerance='5m',
                suffix='_physio'
            )
            
            # Calcular porcentaje de valores no nulos
            if 'physiological_value' in joined_df.columns:
                non_null_percentage = (joined_df['physiological_value'].is_not_null().sum() / len(joined_df)) * 100
                # logging.info(f"Señal physiological_value: {non_null_percentage:.1f}% valores no nulos")
        
        # Verificar columnas críticas una última vez
        if 'value' not in joined_df.columns:
            logging.error("Columna 'value' perdida después de join con physiological")
            raise ValueError("Columna 'value' perdida después de join con physiological")
        if 'bolus' not in joined_df.columns:
            logging.error("Columna 'bolus' perdida después de join con physiological")
            raise ValueError("Columna 'bolus' perdida después de join con physiological")
        
        # Registrar forma final y columnas presentes
        # logging.info(f"Forma de datos unidos: {joined_df.shape}")
        # logging.info(f"Columnas presentes: {joined_df.columns}")
        
        return joined_df
        
    except Exception as e:
        logging.error(f"Error en join_signals: {e}")
        raise

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
    
    Parámetros:
    -----------
    bolus_time : datetime
        Tiempo del bolo de insulina
    meal_df : pl.DataFrame
        DataFrame con datos de comidas
    window_hours : float, opcional
        Horas de la ventana alrededor del bolo (default: 2.0)
        
    Retorna:
    --------
    Dict[str, Union[float, int]]
        Diccionario con características del contexto de comidas
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

def compute_clinical_risk_indicators(glucose_values: List[float], 
                                   physiological_data: Optional[Dict[str, List[float]]] = None,
                                   time_values: Optional[List[datetime]] = None) -> Dict[str, float]:
    """
    Calcula indicadores de riesgo clínico basados en valores de glucosa y señales fisiológicas.
    
    Parámetros:
    -----------
    glucose_values : List[float]
        Lista de valores de glucosa
    physiological_data : Optional[Dict[str, List[float]]], opcional
        Diccionario con señales fisiológicas (default: None)
    time_values : Optional[List[datetime]], opcional
        Lista de timestamps correspondientes a los valores (default: None)
        
    Retorna:
    --------
    Dict[str, float]
        Diccionario con indicadores de riesgo clínico
    """
    if not glucose_values:
        return {
            'hypo_risk': 0.0,
            'hyper_risk': 0.0,
            'variability_risk': 0.0,
            'sleep_hypo_risk': 0.0,
            'activity_hypo_risk': 0.0,
            'stress_hyper_risk': 0.0,
            'overall_risk': 0.0
        }
    
    # Obtener umbrales con valores por defecto si no están en CONFIG
    hypo_threshold = CONFIG_PROCESSING.get('hypo_threshold', 70)
    hyper_threshold = CONFIG_PROCESSING.get('hyper_threshold', 180)
    
    # Calcular riesgos básicos
    hypo_count = sum(1 for g in glucose_values if g < hypo_threshold)
    hyper_count = sum(1 for g in glucose_values if g > hyper_threshold)
    
    hypo_risk = hypo_count / len(glucose_values)
    hyper_risk = hyper_count / len(glucose_values)
    
    # Calcular riesgo de variabilidad
    if len(glucose_values) > 1:
        glucose_std = np.std(glucose_values)
        variability_risk = min(1.0, glucose_std / 50.0)  # Normalizar a [0,1]
    else:
        variability_risk = 0.0
    
    # Inicializar riesgos específicos
    sleep_hypo_risk = 0.0
    activity_hypo_risk = 0.0
    stress_hyper_risk = 0.0
    
    # Ajustar riesgos basados en señales fisiológicas si están disponibles
    if physiological_data is not None and time_values is not None:
        # Riesgo de hipoglucemia durante el sueño
        if 'sleep_event' in physiological_data:
            sleep_indices = [i for i, sleep in enumerate(physiological_data['sleep_event']) 
                           if sleep > 0 and i < len(glucose_values)]
            if sleep_indices:
                sleep_glucose = [glucose_values[i] for i in sleep_indices]
                sleep_hypo_count = sum(1 for g in sleep_glucose if g < hypo_threshold)
                sleep_hypo_risk = sleep_hypo_count / len(sleep_glucose)
        
        # Riesgo de hipoglucemia durante actividad
        if 'work_event' in physiological_data:
            activity_indices = [i for i, work in enumerate(physiological_data['work_event']) 
                              if work > 0 and i < len(glucose_values)]
            if activity_indices:
                activity_glucose = [glucose_values[i] for i in activity_indices]
                activity_hypo_count = sum(1 for g in activity_glucose if g < hypo_threshold)
                activity_hypo_risk = activity_hypo_count / len(activity_glucose)
        
        # Riesgo de hiperglucemia durante estrés
        if 'stressors_event' in physiological_data:
            stress_indices = [i for i, stress in enumerate(physiological_data['stressors_event']) 
                            if stress > 0 and i < len(glucose_values)]
            if stress_indices:
                stress_glucose = [glucose_values[i] for i in stress_indices]
                stress_hyper_count = sum(1 for g in stress_glucose if g > hyper_threshold)
                stress_hyper_risk = stress_hyper_count / len(stress_glucose)
    
    # Obtener pesos de riesgo con valores por defecto
    risk_weights = CONFIG_PROCESSING.get('risk_weights', {
        'hypo': 0.3,
        'hyper': 0.3,
        'variability': 0.2,
        'sleep_hypo': 0.1,
        'activity_hypo': 0.05,
        'stress_hyper': 0.05
    })
    
    # Calcular riesgo general ponderado
    overall_risk = (
        risk_weights['hypo'] * hypo_risk +
        risk_weights['hyper'] * hyper_risk +
        risk_weights['variability'] * variability_risk +
        risk_weights['sleep_hypo'] * sleep_hypo_risk +
        risk_weights['activity_hypo'] * activity_hypo_risk +
        risk_weights['stress_hyper'] * stress_hyper_risk
    )
    
    return {
        'hypo_risk': hypo_risk,
        'hyper_risk': hyper_risk,
        'variability_risk': variability_risk,
        'sleep_hypo_risk': sleep_hypo_risk,
        'activity_hypo_risk': activity_hypo_risk,
        'stress_hyper_risk': stress_hyper_risk,
        'overall_risk': overall_risk
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

def process_xml_directory(data_dir: str) -> Optional[pl.DataFrame]:
    """
    Procesa un directorio XML completo, extrayendo y transformando características.
    
    Parámetros:
    -----------
    data_dir : str
        Directorio con archivos XML
        
    Retorna:
    --------
    Optional[pl.DataFrame]
        DataFrame procesado o None si hubo un error
    """
    try:
        logging.info(f"Procesando directorio XML: {data_dir}")
        data = load_data(data_dir)
        
        # Preprocesar datos de bolus y meal
        processed_data = preprocess_bolus_meal(data)
        
        # Preprocesar datos de CGM
        cgm_data = preprocess_cgm(data.get("glucose_level"))
        
        # Extraer los DataFrames necesarios
        bolus_df = processed_data.get("bolus", pl.DataFrame())
        meal_df = processed_data.get("meal", pl.DataFrame())
        
        # Unir señales con los parámetros correctos
        df = join_signals(cgm_data, bolus_df, meal_df)
        
        # Extraer características mejoradas
        df = extract_enhanced_features(df, processed_data.get("meal"))
        
        # Transformar características
        df = transform_enhanced_features(df)
        
        logging.info(f"Procesado exitoso para {data_dir}: {df.shape}")
        return df
    except Exception as e:
        logging.error(f"Error procesando {data_dir}: {str(e)}")
        return None

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

def load_data(data_dir: str) -> Dict[str, pl.DataFrame]:
    """
    Carga los datos y muestra las columnas de cada DataFrame.
    Verifica que estén presentes los 6 sujetos esperados para cada año (2018 y 2020).
    Ahora soporta: glucose_level, bolus, meal, basal, temp_basal, exercise, basis_steps, hypo_event,
    finger_stick, sleep, work, stressors, illness, basis_heart_rate, basis_gsr, basis_skin_temperature,
    basis_air_temperature, basis_sleep, acceleration.
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
    data_dict = {}
    subject_stats = defaultdict(lambda: defaultdict(int))
    expected_types = [
        'glucose_level', 'bolus', 'meal', 'basal', 'temp_basal', 'exercise', 'basis_steps', 'hypo_event',
        'finger_stick', 'sleep', 'work', 'stressors', 'illness', 'basis_heart_rate', 'basis_gsr',
        'basis_skin_temperature', 'basis_air_temperature', 'basis_sleep', 'acceleration'
    ]
    
    # Diccionario para almacenar columnas por tipo de evento y sujeto
    event_columns_by_subject = defaultdict(lambda: defaultdict(set))
    
    for xml_file in xml_files:
        subject_id = os.path.basename(xml_file).split('.')[0]
        numeric_id = extract_numeric_id(subject_id)
        logging.info(f"\n{'='*50}")
        logging.info(f"Procesando SubjectID: {subject_id} (ID numérico: {numeric_id}, Año {year})")
        logging.info(f"{'='*50}")
        
        try:
            tree = ET.parse(xml_file)
            root = tree.getroot()
            
            # Procesar cada tipo de dato
            for data_type_elem in root:
                data_type = data_type_elem.tag
                if data_type == 'patient':
                    continue
                if data_type not in expected_types:
                    continue
                
                records = []
                for event in data_type_elem:
                    record_dict = dict(event.attrib)
                    record_dict['SubjectID'] = subject_id
                    record_dict['Year'] = year
                    records.append(record_dict)
                    
                    # Registrar columnas por tipo de evento
                    for col in record_dict.keys():
                        if col not in ['SubjectID', 'Year']:
                            event_columns_by_subject[subject_id][data_type].add(col)
                
                if records:
                    df = pl.DataFrame(records)
                    if 'value' in df.columns:
                        df = df.with_columns(pl.col('value').cast(pl.Float64))
                    data_dict[data_type] = pl.concat([data_dict.get(data_type, pl.DataFrame()), df])
                    # logging.info(f"SubjectID {subject_id}: {data_type}={len(records)} registros")
                    subject_stats[subject_id][data_type] += len(records)
                    
                    # Mostrar estadísticas de valores para columnas numéricas
                    if 'value' in df.columns:
                        value_stats = df.select(pl.col('value')).describe()
                        # logging.info(f"Estadísticas de valores para {data_type}:")
                        # logging.info(f"  Min: {value_stats['value'].min():.2f}")
                        # logging.info(f"  Max: {value_stats['value'].max():.2f}")
                        # logging.info(f"  Mean: {value_stats['value'].mean():.2f}")
                        # logging.info(f"  Std: {value_stats['value'].std():.2f}")
        
        except Exception as e:
            logging.error(f"Error procesando {xml_file}: {e}")
            continue
    
    # Mostrar columnas por tipo de evento y sujeto
    # logging.info("\nColumnas por tipo de evento y sujeto:")
    # for subject_id in sorted(event_columns_by_subject.keys()):
    #     logging.info(f"\n{'-'*50}")
    #     logging.info(f"SubjectID: {subject_id}")
    #     for event_type, columns in sorted(event_columns_by_subject[subject_id].items()):
    #         logging.info(f"  {event_type}: {sorted(columns)}")
    
    # logging.info(f"\nEstadísticas por sujeto (Año {year}):")
    # for subject_id in sorted(subject_stats.keys()):
    #     stats = subject_stats[subject_id]
    #     stat_str = ", ".join([f"{k}={v}" for k, v in stats.items()])
    #     logging.info(f"SubjectID {subject_id}: {stat_str}")
    
    missing_types = [t for t in expected_types if t not in data_dict]
    if missing_types:
        logging.warning(f"Faltan tipos de datos: {missing_types}")
    if len(subject_stats) != len(expected_subjects[year]):
        logging.error(f"Se encontraron datos para {len(subject_stats)}/{len(expected_subjects[year])} sujetos")
    
    return data_dict

def preprocess_bolus_meal(data: Dict[str, pl.DataFrame]) -> Dict[str, pl.DataFrame]:
    """
    Renombra y convierte columnas clave de bolus, meal, basal, temp_basal, exercise, basis_steps, hypo_event,
    finger_stick, sleep, work, stressors, illness, basis_heart_rate, basis_gsr, basis_skin_temperature,
    basis_air_temperature, basis_sleep, acceleration.
    """
    processed = {}
    # Bolus
    if "bolus" in data:
        bolus = data["bolus"].clone()
        if "dose" in bolus.columns:
            bolus = bolus.rename({"dose": "bolus"})
            bolus = bolus.with_columns(pl.col("bolus").cast(pl.Float64))
        if "ts_begin" in bolus.columns:
            bolus = bolus.with_columns(
                pl.col("ts_begin").str.strptime(pl.Datetime, "%d-%m-%Y %H:%M:%S").alias("Timestamp")
            )
        valid_bolus = bolus.filter(pl.col("bolus").is_not_null() & (pl.col("bolus") > 0))
        processed["bolus"] = valid_bolus
        logging.info(f"Eventos bolus válidos: {valid_bolus.height}")

    # Meal
    if "meal" in data:
        meal = data["meal"].clone()
        if "carbs" in meal.columns:
            meal = meal.rename({"carbs": "meal_carbs"})
            meal = meal.with_columns(pl.col("meal_carbs").cast(pl.Float64))
        if "ts" in meal.columns:
            meal = meal.with_columns(
                pl.col("ts").str.strptime(pl.Datetime, "%d-%m-%Y %H:%M:%S").alias("Timestamp")
            )
        valid_meal = meal.filter(pl.col("meal_carbs").is_not_null() & (pl.col("meal_carbs") > 0))
        processed["meal"] = valid_meal
        logging.info(f"Eventos meal válidos: {valid_meal.height}")

    # Basal
    if "basal" in data:
        basal = data["basal"].clone()
        basal = basal.rename({"value": "basal_rate"}).with_columns(
            pl.col("basal_rate").cast(pl.Float64),
            pl.col("ts").str.strptime(pl.Datetime, "%d-%m-%Y %H:%M:%S").alias("Timestamp")
        )
        processed["basal"] = basal.filter(pl.col("basal_rate").is_not_null())
        logging.info(f"Eventos basal válidos: {processed['basal'].height}")

    # Temp Basal
    if "temp_basal" in data:
        temp_basal = data["temp_basal"].clone()
        temp_basal = temp_basal.rename({"value": "temp_basal_rate"}).with_columns(
            pl.col("temp_basal_rate").cast(pl.Float64),
            pl.col("ts_begin").str.strptime(pl.Datetime, "%d-%m-%Y %H:%M:%S").alias("Timestamp")
        )
        processed["temp_basal"] = temp_basal.filter(pl.col("temp_basal_rate").is_not_null())
        logging.info(f"Eventos temp_basal válidos: {processed['temp_basal'].height}")

    # Exercise
    if "exercise" in data:
        exercise = data["exercise"].clone()
        exercise = exercise.with_columns(
            pl.col("intensity").cast(pl.Float64),
            pl.col("duration").cast(pl.Float64),
            pl.col("ts").str.strptime(pl.Datetime, "%d-%m-%Y %H:%M:%S").alias("Timestamp")
        )
        processed["exercise"] = exercise.filter(pl.col("intensity").is_not_null())
        logging.info(f"Eventos exercise válidos: {processed['exercise'].height}")

    # Steps
    if "basis_steps" in data:
        steps = data["basis_steps"].clone()
        steps = steps.rename({"value": "steps"}).with_columns(
            pl.col("steps").cast(pl.Float64),
            pl.col("ts").str.strptime(pl.Datetime, "%d-%m-%Y %H:%M:%S").alias("Timestamp")
        )
        processed["basis_steps"] = steps.filter(pl.col("steps").is_not_null())
        logging.info(f"Eventos steps válidos: {processed['basis_steps'].height}")

    # Hypo Event
    if "hypo_event" in data:
        hypo = data["hypo_event"].clone()
        if "ts" in hypo.columns:
            hypo = hypo.with_columns(
                pl.col("ts").str.strptime(pl.Datetime, "%d-%m-%Y %H:%M:%S").alias("Timestamp")
            )
        processed["hypo_event"] = hypo
        logging.info(f"Eventos hypo_event válidos: {processed['hypo_event'].height}")

    # Finger Stick
    if "finger_stick" in data:
        finger_stick = data["finger_stick"].clone()
        finger_stick = finger_stick.rename({"value": "finger_stick_bg"}).with_columns(
            pl.col("finger_stick_bg").cast(pl.Float64),
            pl.col("ts").str.strptime(pl.Datetime, "%d-%m-%Y %H:%M:%S").alias("Timestamp")
        )
        processed["finger_stick"] = finger_stick.filter(pl.col("finger_stick_bg").is_not_null())
        logging.info(f"Eventos finger_stick válidos: {processed['finger_stick'].height}")

    # Sleep
    if "sleep" in data:
        sleep = data["sleep"].clone()
        if "ts_begin" in sleep.columns and "ts_end" in sleep.columns:
            sleep = sleep.with_columns(
                pl.col("quality").cast(pl.Float64),
                pl.col("ts_begin").str.strptime(pl.Datetime, "%d-%m-%Y %H:%M:%S").alias("Timestamp_begin"),
                pl.col("ts_end").str.strptime(pl.Datetime, "%d-%m-%Y %H:%M:%S").alias("Timestamp_end")
            )
        elif "ts" in sleep.columns:
            sleep = sleep.with_columns(
                pl.col("quality").cast(pl.Float64),
                pl.col("ts").str.strptime(pl.Datetime, "%d-%m-%Y %H:%M:%S").alias("Timestamp")
            )
        processed["sleep"] = sleep.filter(pl.col("quality").is_not_null())
        logging.info(f"Eventos sleep válidos: {processed['sleep'].height}")

    # Work
    if "work" in data:
        work = data["work"].clone()
        if "ts_begin" in work.columns and "ts_end" in work.columns:
            work = work.with_columns(
                pl.col("intensity").cast(pl.Float64),
                pl.col("ts_begin").str.strptime(pl.Datetime, "%d-%m-%Y %H:%M:%S").alias("Timestamp_begin"),
                pl.col("ts_end").str.strptime(pl.Datetime, "%d-%m-%Y %H:%M:%S").alias("Timestamp_end")
            )
        elif "ts" in work.columns:
            work = work.with_columns(
                pl.col("intensity").cast(pl.Float64),
                pl.col("ts").str.strptime(pl.Datetime, "%d-%m-%Y %H:%M:%S").alias("Timestamp")
            )
        processed["work"] = work.filter(pl.col("intensity").is_not_null())
        logging.info(f"Eventos work válidos: {processed['work'].height}")

    # Stressors
    if "stressors" in data:
        stressors = data["stressors"].clone()
        if "ts" in stressors.columns:
            stressors = stressors.with_columns(
                pl.col("ts").str.strptime(pl.Datetime, "%d-%m-%Y %H:%M:%S").alias("Timestamp")
            )
        processed["stressors"] = stressors
        logging.info(f"Eventos stressors válidos: {processed['stressors'].height}")

    # Illness
    if "illness" in data:
        illness = data["illness"].clone()
        if "ts" in illness.columns:
            illness = illness.with_columns(
                pl.col("ts").str.strptime(pl.Datetime, "%d-%m-%Y %H:%M:%S").alias("Timestamp")
            )
        processed["illness"] = illness
        logging.info(f"Eventos illness válidos: {processed['illness'].height}")

    # Basis Heart Rate
    if "basis_heart_rate" in data:
        heart_rate = data["basis_heart_rate"].clone()
        heart_rate = heart_rate.rename({"value": "heart_rate"}).with_columns(
            pl.col("heart_rate").cast(pl.Float64),
            pl.col("ts").str.strptime(pl.Datetime, "%d-%m-%Y %H:%M:%S").alias("Timestamp")
        )
        processed["basis_heart_rate"] = heart_rate.filter(pl.col("heart_rate").is_not_null())
        logging.info(f"Eventos heart_rate válidos: {processed['basis_heart_rate'].height}")

    # Basis GSR
    if "basis_gsr" in data:
        gsr = data["basis_gsr"].clone()
        gsr = gsr.rename({"value": "gsr"}).with_columns(
            pl.col("gsr").cast(pl.Float64),
            pl.col("ts").str.strptime(pl.Datetime, "%d-%m-%Y %H:%M:%S").alias("Timestamp")
        )
        processed["basis_gsr"] = gsr.filter(pl.col("gsr").is_not_null())
        logging.info(f"Eventos gsr válidos: {processed['basis_gsr'].height}")

    # Basis Skin Temperature
    if "basis_skin_temperature" in data:
        skin_temp = data["basis_skin_temperature"].clone()
        skin_temp = skin_temp.rename({"value": "skin_temperature"}).with_columns(
            pl.col("skin_temperature").cast(pl.Float64),
            pl.col("ts").str.strptime(pl.Datetime, "%d-%m-%Y %H:%M:%S").alias("Timestamp")
        )
        processed["basis_skin_temperature"] = skin_temp.filter(pl.col("skin_temperature").is_not_null())
        logging.info(f"Eventos skin_temperature válidos: {processed['basis_skin_temperature'].height}")

    # Basis Air Temperature
    if "basis_air_temperature" in data:
        air_temp = data["basis_air_temperature"].clone()
        air_temp = air_temp.rename({"value": "air_temperature"}).with_columns(
            pl.col("air_temperature").cast(pl.Float64),
            pl.col("ts").str.strptime(pl.Datetime, "%d-%m-%Y %H:%M:%S").alias("Timestamp")
        )
        processed["basis_air_temperature"] = air_temp.filter(pl.col("air_temperature").is_not_null())
        logging.info(f"Eventos air_temperature válidos: {processed['basis_air_temperature'].height}")

    # Basis Sleep
    if "basis_sleep" in data:
        basis_sleep = data["basis_sleep"].clone()
        if "ts_begin" in basis_sleep.columns and "ts_end" in basis_sleep.columns:
            basis_sleep = basis_sleep.with_columns(
                pl.col("quality").cast(pl.Float64),
                pl.col("ts_begin").str.strptime(pl.Datetime, "%d-%m-%Y %H:%M:%S").alias("Timestamp_begin"),
                pl.col("ts_end").str.strptime(pl.Datetime, "%d-%m-%Y %H:%M:%S").alias("Timestamp_end")
            )
        elif "ts" in basis_sleep.columns:
            basis_sleep = basis_sleep.with_columns(
                pl.col("quality").cast(pl.Float64),
                pl.col("ts").str.strptime(pl.Datetime, "%d-%m-%Y %H:%M:%S").alias("Timestamp")
            )
        processed["basis_sleep"] = basis_sleep.filter(pl.col("quality").is_not_null())
        logging.info(f"Eventos basis_sleep válidos: {processed['basis_sleep'].height}")

    # Acceleration
    if "acceleration" in data:
        accel = data["acceleration"].clone()
        accel = accel.rename({"value": "acceleration"}).with_columns(
            pl.col("acceleration").cast(pl.Float64),
            pl.col("ts").str.strptime(pl.Datetime, "%d-%m-%Y %H:%M:%S").alias("Timestamp")
        )
        processed["acceleration"] = accel.filter(pl.col("acceleration").is_not_null())
        logging.info(f"Eventos acceleration válidos: {processed['acceleration'].height}")

    return processed

def extract_numeric_id(subject_id: str) -> int:
    """
    Extrae el ID numérico de una cadena de identificación de sujeto.
    
    Parámetros:
    -----------
    subject_id : str
        Cadena de identificación del sujeto (ej: '559-ws-training')
        
    Retorna:
    --------
    int
        ID numérico extraído
    """
    # Usar regex para extraer la parte numérica
    match = re.search(r'^(\d+)', subject_id)
    if match:
        return int(match.group(1))
    else:
        # Si no hay número, usar un valor predeterminado o lanzar un error
        logging.warning(f"No se pudo extraer ID numérico de: {subject_id}")
        return 9999  # Un valor que no colisione con IDs reales

def extract_enhanced_features(df: pl.DataFrame, meal_df: Optional[pl.DataFrame] = None,
                            extended_cgm_df: Optional[pl.DataFrame] = None) -> pl.DataFrame:
    """
    Extrae características mejoradas del DataFrame, incluyendo características CGM,
    fisiológicas y de eventos.
    
    Parámetros:
    -----------
    df : pl.DataFrame
        DataFrame con datos unidos
    meal_df : Optional[pl.DataFrame], opcional
        DataFrame con datos de comidas (default: None)
    extended_cgm_df : Optional[pl.DataFrame], opcional
        DataFrame con datos CGM extendidos (default: None)
        
    Retorna:
    --------
    pl.DataFrame
        DataFrame con características extraídas
    """
    # Verificar columnas críticas
    if 'value' not in df.columns:
        logging.error("Columna 'value' no encontrada en el DataFrame de entrada")
        raise ValueError("Columna 'value' no encontrada en el DataFrame de entrada")
    
    # Preservar columnas críticas
    critical_columns = ['value', 'bolus', 'SubjectID', 'Timestamp']
    preserved_columns = {col: df[col] for col in critical_columns if col in df.columns}
    
    # Definir grupos de características
    feature_groups = {
        'cgm': [
            'glucose_last', 'glucose_mean', 'glucose_std', 'glucose_min',
            'glucose_max', 'glucose_range', 'glucose_slope'
        ],
        'physiological': [
            'heart_rate', 'gsr', 'skin_temperature', 'air_temperature',
            'acceleration'
        ],
        'events': [
            'sleep_event', 'work_event', 'stressors_event',
            'illness_event', 'basis_sleep_event'
        ],
        'meal_context': [
            'meal_carbs', 'meal_time_diff_minutes', 'meal_time_diff_hours',
            'has_meal', 'meals_in_window', 'significant_meal',
            'total_carbs_window', 'largest_meal_carbs', 'meal_timing_score'
        ]
    }
    
    # Verificar columnas disponibles
    available_features = {}
    for group, features in feature_groups.items():
        available_features[group] = [f for f in features if f in df.columns]
        logging.info(f"{group}: {len(available_features[group])}/{len(features)} características presentes")
    
    # Extraer características CGM
    if 'value' in df.columns:
        # Asegurarse de que 'value' es una columna numérica, no una lista
        if df['value'].dtype == pl.List:
            df = df.with_columns(pl.col('value').list.first().alias('value'))
        
        # Calcular características CGM
        df = df.with_columns([
            pl.col('value').alias('glucose_last'),
            pl.col('value').rolling_mean(window_size=5).alias('glucose_mean'),
            pl.col('value').rolling_std(window_size=5).alias('glucose_std'),
            pl.col('value').rolling_min(window_size=5).alias('glucose_min'),
            pl.col('value').rolling_max(window_size=5).alias('glucose_max'),
            (pl.col('value').rolling_max(window_size=5) - 
             pl.col('value').rolling_min(window_size=5)).alias('glucose_range'),
            pl.col('value').diff().alias('glucose_slope')
        ])
        
        # Calcular características de 24h si hay suficientes datos
        if extended_cgm_df is not None and not extended_cgm_df.is_empty():
            patterns_24h = compute_glucose_patterns_24h(
                extended_cgm_df.get_column('value').to_list()
            )
            for key, value in patterns_24h.items():
                df = df.with_columns(pl.lit(value).alias(key))
        else:
            # Si no hay datos extendidos, calcular usando los datos actuales
            patterns_24h = compute_glucose_patterns_24h(
                df.get_column('value').to_list()
            )
            for key, value in patterns_24h.items():
                df = df.with_columns(pl.lit(value).alias(key))
            
            logging.info("Usando datos actuales para calcular patrones de 24h")
    
    # Extraer características fisiológicas
    for signal in available_features.get('physiological', []):
        if signal in df.columns:
            # Asegurarse de que la señal es una columna numérica, no una lista
            if df[signal].dtype == pl.List:
                df = df.with_columns(pl.col(signal).list.first().alias(signal))
            
            # Calcular estadísticas de la señal
            df = df.with_columns([
                pl.col(signal).rolling_mean(window_size=5).alias(f'{signal}_mean'),
                pl.col(signal).rolling_std(window_size=5).alias(f'{signal}_std'),
                pl.col(signal).rolling_min(window_size=5).alias(f'{signal}_min'),
                pl.col(signal).rolling_max(window_size=5).alias(f'{signal}_max')
            ])
    
    # Extraer características de eventos
    for event in available_features.get('events', []):
        if event in df.columns:
            # Asegurarse de que el evento es una columna numérica, no una lista
            if df[event].dtype == pl.List:
                df = df.with_columns(pl.col(event).list.first().alias(event))
            
            # Calcular estadísticas de eventos
            df = df.with_columns([
                pl.col(event).rolling_sum(window_size=5).alias(f'{event}_count'),
                pl.col(event).rolling_mean(window_size=5).alias(f'{event}_density')
            ])
    
    # Extraer características de contexto de comidas
    if meal_df is not None and not meal_df.is_empty() and 'Timestamp' in df.columns:
        # Calcular tiempo desde última comida
        df = df.with_columns([
            pl.col('Timestamp').diff().dt.total_minutes().alias('time_since_last_meal')
        ])
        
        # Calcular características de comidas en ventana
        window_hours = 2.0
        for row in df.iter_rows(named=True):
            bolus_time = row['Timestamp']
            meal_context = compute_enhanced_meal_context(
                bolus_time, meal_df, window_hours=window_hours
            )
            
            # Actualizar características de comidas
            for key, value in meal_context.items():
                if key in df.columns:
                    df = df.with_columns(pl.lit(value).alias(key))
    else:
        # Si no hay datos de comidas o Timestamp, inicializar columnas con valores por defecto
        default_meal_features = {
            'time_since_last_meal': 0.0,
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
        for key, value in default_meal_features.items():
            if key not in df.columns:
                df = df.with_columns(pl.lit(value).alias(key))
    
    # Calcular indicadores de riesgo clínico
    if 'value' in df.columns:
        glucose_values = df.get_column('value').to_list()
        physiological_data = {
            signal: df.get_column(signal).to_list()
            for signal in available_features.get('physiological', [])
            if signal in df.columns
        }
        time_values = df.get_column('Timestamp').to_list() if 'Timestamp' in df.columns else None
        
        risk_indicators = compute_clinical_risk_indicators(
            glucose_values, physiological_data, time_values
        )
        
        # Actualizar indicadores de riesgo
        for key, value in risk_indicators.items():
            df = df.with_columns(pl.lit(value).alias(key))
    
    # Añadir características de tiempo cíclicas
    if 'Timestamp' in df.columns:
        time_features = []
        for ts in df.get_column('Timestamp'):
            time_features.append(encode_time_cyclical(ts))
        
        # Convertir a DataFrame y unir
        time_df = pl.DataFrame(time_features)
        df = df.hstack(time_df)
    
    # Rellenar valores nulos
    df = df.fill_null(0)
    
    # Restaurar columnas críticas preservadas
    for col, values in preserved_columns.items():
        if col not in df.columns:
            df = df.with_columns(values.alias(col))
    
    # Verificar características generadas
    for group, features in feature_groups.items():
        present = [f for f in features if f in df.columns]
        logging.info(f"{group} generadas: {len(present)}/{len(features)}")
        if len(present) < len(features):
            missing = set(features) - set(present)
            logging.warning(f"Faltan características de {group}: {missing}")
    
    # Verificar columnas críticas al final
    critical_columns = ['value', 'time_in_range_24h', 'bolus']
    missing_critical = [col for col in critical_columns if col not in df.columns]
    if missing_critical:
        logging.error(f"Faltan columnas críticas al final: {missing_critical}")
        raise ValueError(f"Faltan columnas críticas al final: {missing_critical}")
    
    return df

def transform_enhanced_features(df: pl.DataFrame) -> pl.DataFrame:
    """
    Aplica transformaciones mejoradas incluyendo transformaciones logarítmicas, 
    normalización y expansión de características para un espacio de observación 
    de 52 dimensiones.
    
    Parámetros:
    -----------
    df : pl.DataFrame
        DataFrame con características sin transformar
        
    Retorna:
    --------
    pl.DataFrame
        DataFrame con características transformadas
    """
    logging.info("Aplicando transformaciones mejoradas...")
    
    # Log1p transformations for skewed features
    log_transform_cols = [
        "bolus", "carb_input", "meal_carbs", "insulin_on_board",
        "total_carbs_window", "largest_meal_carbs"
    ]
    
    for col in log_transform_cols:
        if col in df.columns:
            df = df.with_columns(
                pl.col(col).log1p().alias(f"{col}_log1p")
            )

    # Normalize percentage features (0-100 to 0-1)
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
    
    # Normalize glucose-related features for stability
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
    
    # Add derived features for model compatibility
    compatibility_features = {
        "hour_of_day": "hour_of_day_normalized",
        "has_meal_binary": "has_meal",
        "significant_meal_binary": "significant_meal"
    }
    
    for new_col, source_col in compatibility_features.items():
        if source_col in df.columns and new_col not in df.columns:
            df = df.with_columns(pl.col(source_col).alias(new_col))
    
    logging.info(f"Transformaciones mejoradas completadas. Forma final: {df.shape}")
    
    return df

def extract_enhanced_features_excel(df: pl.DataFrame) -> pl.DataFrame:
    """
    Extrae características mejoradas específicamente para datos Excel.
    
    Parámetros:
    -----------
    df : pl.DataFrame
        DataFrame con datos Excel
        
    Retorna:
    --------
    pl.DataFrame
        DataFrame con características extraídas
    """
    logging.info("Extrayendo características mejoradas para datos Excel...")
    
    # 1. Procesar ventana CGM
    if "cgm_window" in df.columns:
        # Extraer estadísticas de la ventana CGM
        enhanced_rows = []
        
        for row in df.iter_rows(named=True):
            cgm_window = row.get("cgm_window", [])
            if not cgm_window:
                continue
                
            # Características básicas de CGM
            cgm_stats = {
                "glucose_last": cgm_window[-1] if cgm_window else 120.0,
                "glucose_mean": float(np.mean(cgm_window)) if cgm_window else 120.0,
                "glucose_std": float(np.std(cgm_window)) if cgm_window else 0.0,
                "glucose_min": float(np.min(cgm_window)) if cgm_window else 120.0,
                "glucose_max": float(np.max(cgm_window)) if cgm_window else 120.0,
                "glucose_range": float(np.max(cgm_window) - np.min(cgm_window)) if cgm_window else 0.0,
                "glucose_slope": float(cgm_window[-1] - cgm_window[0]) / len(cgm_window) if len(cgm_window) > 1 else 0.0,
            }
            
            # Características de tiempo
            timestamp = row.get("Timestamp")
            time_features = encode_time_cyclical(timestamp) if timestamp else {}
            
            # Características para compatibilidad con XML
            xml_compat = {
                "value": cgm_stats["glucose_last"],  # 'value' en XML corresponde al último valor de glucosa
                "bwz_carb_input": row.get("carb_input", 0.0),
                "SubjectID": row.get("subject_id", 0),
            }
            
            # Características de riesgo
            risk_indicators = compute_clinical_risk_indicators(
                cgm_window, 
                physiological_data=None,
                time_values=None
            )
            
            # Patrones de glucosa de 24h
            glucose_patterns = compute_glucose_patterns_24h(cgm_window)
            
            # Crear fila con todas las características
            enhanced_row = {
                **row,
                **cgm_stats,
                **time_features,
                **xml_compat,
                **risk_indicators,
                **glucose_patterns
            }
            
            enhanced_rows.append(enhanced_row)
        
        # Crear nuevo DataFrame con características mejoradas
        if enhanced_rows:
            df = pl.DataFrame(enhanced_rows)
    
    # 2. Transformar columnas numéricas
    numeric_cols = ["bolus", "carb_input", "insulin_on_board", "insulin_carb_ratio", "insulin_sensitivity_factor"]
    for col in numeric_cols:
        if col in df.columns:
            # Asegurar que es numérica
            df = df.with_columns(pl.col(col).cast(pl.Float64))
            
            # Reemplazar valores extremos
            df = df.with_columns(
                pl.when(pl.col(col) < 0).then(0.0)
                .otherwise(pl.col(col))
                .alias(col)
            )
    
    # 3. Agregar features derivadas para comidas
    if "carb_input" in df.columns:
        df = df.with_columns(
            # Has meal binario (1 si hay carbohidratos, 0 si no)
            (pl.col("carb_input") > 0).cast(pl.Float64).alias("has_meal"),
            
            # Comida significativa (1 si es más de 15g, 0 si no)
            (pl.col("carb_input") > 15).cast(pl.Float64).alias("significant_meal")
        )
    
    # 4. Verificar columnas críticas para compatibilidad con XML
    critical_columns = ["value", "bolus", "SubjectID", "Timestamp"]
    missing_critical = [col for col in critical_columns if col not in df.columns]
    if missing_critical:
        logging.warning(f"Faltan columnas críticas: {missing_critical}")
        
        # Agregar columnas faltantes con valores por defecto
        for col in missing_critical:
            if col == "value" and "glucose_last" in df.columns:
                df = df.with_columns(pl.col("glucose_last").alias("value"))
            elif col == "SubjectID" and "subject_id" in df.columns:
                df = df.with_columns(pl.col("subject_id").alias("SubjectID"))
            else:
                # Usar valor por defecto según el tipo de columna
                if col == "bolus":
                    df = df.with_columns(pl.lit(0.0).alias(col))
                elif col == "value":
                    df = df.with_columns(pl.lit(120.0).alias(col))
                elif col == "Timestamp" and "ts" in df.columns:
                    df = df.with_columns(pl.col("ts").alias(col))
                else:
                    df = df.with_columns(pl.lit(None).alias(col))
    
    logging.info(f"Extracción de características Excel completada. Forma: {df.shape}")
    return df

def transform_enhanced_features_excel(df: pl.DataFrame) -> pl.DataFrame:
    """
    Aplica transformaciones a características de datos Excel para hacerlas compatibles
    con el formato esperado por los modelos.
    
    Parámetros:
    -----------
    df : pl.DataFrame
        DataFrame con características Excel extraídas
        
    Retorna:
    --------
    pl.DataFrame
        DataFrame con características transformadas
    """
    logging.info("Transformando características de datos Excel...")
    
    # 1. Transformaciones logarítmicas para características sesgadas
    log_transform_cols = [
        "bolus", "carb_input", "insulin_on_board", "insulin_carb_ratio"
    ]
    
    for col in log_transform_cols:
        if col in df.columns:
            df = df.with_columns(
                # Usar log1p para manejar valores cero
                pl.col(col).log1p().alias(f"{col}_log1p")
            )
    
    # 2. Normalizar características porcentuales
    if "cgm_window" in df.columns:
        percentage_cols = [
            "hypo_percentage_24h", "hyper_percentage_24h", "time_in_range_24h", "cv_24h"
        ]
        
        for col in percentage_cols:
            if col in df.columns:
                df = df.with_columns(
                    (pl.col(col) / 100.0).alias(f"{col}_normalized")
                )
    
    # 3. Expandir la ventana CGM a columnas individuales
    if "cgm_window" in df.columns:
        window_size = min(CONFIG_PROCESSING["window_steps"], 24)  # Usar hasta 24 puntos
        
        for i in range(window_size):
            df = df.with_columns(
                pl.col("cgm_window").list.get(i, null_on_oob=True)
                .fill_null(120.0)  # Valor por defecto para missing
                .alias(f"cgm_{i}")
            )
        
        # Eliminar columna de lista original para ahorrar espacio
        df = df.drop("cgm_window")
    
    # 4. Crear variables dummy para características categóricas
    if "hour_of_day" in df.columns:
        # Características cíclicas para hora del día (si no existen ya)
        if "hour_sin" not in df.columns:
            hour_radians = 2 * np.pi * (df["hour_of_day"] / 24.0)
            df = df.with_columns([
                pl.Series(name="hour_sin", values=np.sin(hour_radians.to_numpy())),
                pl.Series(name="hour_cos", values=np.cos(hour_radians.to_numpy()))
            ])
    
    # 5. Asegurar compatibilidad de nombres con formato XML
    compatibility_mappings = {
        "carb_input": "bwz_carb_input",
        "bg_input": "glucose_last",
        "hour_of_day": "hour_of_day_normalized",
        "has_meal": "has_meal_binary",
        "significant_meal": "significant_meal_binary"
    }
    
    for excel_col, xml_col in compatibility_mappings.items():
        if excel_col in df.columns and xml_col not in df.columns:
            df = df.with_columns(pl.col(excel_col).alias(xml_col))
    
    logging.info(f"Transformación de características Excel completada. Forma: {df.shape}")
    return df

def unify_datetime_precision(df: pl.DataFrame) -> pl.DataFrame:
    """
    Unifica la precisión de todas las columnas datetime a microsegundos.
    
    Parámetros:
    -----------
    df : pl.DataFrame
        DataFrame con columnas datetime que pueden tener diferentes precisiones
        
    Retorna:
    --------
    pl.DataFrame
        DataFrame con todas las columnas datetime convertidas a microsegundos
    """
    datetime_cols = [col for col in df.columns if df[col].dtype.base_type() == pl.Datetime]
    
    if not datetime_cols:
        return df
    
    exprs = []
    for col in datetime_cols:
        # Convertir explícitamente a microsegundos
        exprs.append(pl.col(col).cast(pl.Datetime(time_unit="us")).alias(col))
    
    if exprs:
        df = df.with_columns(exprs)
    
    return df

def unify_column_types(data_frames: list[pl.DataFrame]) -> list[pl.DataFrame]:
    """
    Unifica los tipos de datos entre DataFrames para evitar conflictos en la concatenación.
    
    Parámetros:
    -----------
    data_frames : list[pl.DataFrame]
        Lista de DataFrames a unificar
        
    Retorna:
    --------
    list[pl.DataFrame]
        Lista de DataFrames con tipos de datos consistentes
    """
    if not data_frames:
        return data_frames
    
    # Determinar el tipo más apropiado para cada columna
    column_types = {}
    
    # Primera pasada: identificar todas las columnas y sus tipos
    for df in data_frames:
        for col in df.columns:
            col_type = df[col].dtype
            
            # Saltar si la columna es de tipo Null
            if str(col_type) == "Null":
                continue
            
            # Inicializar o actualizar la preferencia de tipo de columna
            if col not in column_types:
                column_types[col] = col_type
            else:
                current_type = column_types[col]
                
                # Preferir tipos numéricos sobre string
                if str(current_type) == "Utf8" and (str(col_type) == "Float64" or str(col_type) == "Int32"):
                    column_types[col] = col_type
                # Preferir Float64 sobre Int32 para columnas numéricas
                elif str(current_type) == "Int32" and str(col_type) == "Float64":
                    column_types[col] = col_type
                # Para columnas de fecha, usar siempre microsegundos
                elif "Datetime" in str(current_type) and "Datetime" in str(col_type):
                    column_types[col] = pl.Datetime(time_unit="us")
    
    logging.info("Unificando tipos de columnas para concatenación...")
    
    # Segunda pasada: actualizar todos los DataFrames para usar tipos consistentes
    for i, df in enumerate(data_frames):
        exprs = []
        for col in df.columns:
            if col in column_types:
                target_type = column_types[col]
                # Solo convertir si el tipo actual es diferente
                if str(df[col].dtype) != str(target_type):
                    try:
                        exprs.append(pl.col(col).cast(target_type).alias(col))
                    except Exception as e:
                        # Si la conversión falla, intentar método alternativo
                        logging.warning(f"Error al convertir columna {col}: {e}")
                        if str(target_type) == "Utf8":
                            exprs.append(pl.col(col).cast(pl.Utf8).alias(col))
                        elif str(target_type) == "Float64":
                            exprs.append(pl.lit(None).cast(pl.Float64).alias(col))
                        elif str(target_type) == "Int32":
                            exprs.append(pl.lit(None).cast(pl.Int32).alias(col))
                        elif "Datetime" in str(target_type):
                            exprs.append(pl.lit(None).cast(pl.Datetime(time_unit="us")).alias(col))
        
        # Aplicar conversiones si es necesario
        if exprs:
            df = df.with_columns(exprs)
            data_frames[i] = df
    
    return data_frames

def preprocess_data() -> pl.DataFrame:
    """
    Preprocesa los datos utilizando preferentemente las funciones de pl_ohio_only.py
    para datos XML, y uniendo con Excel si es necesario.
    
    Retorna:
    --------
    pl.DataFrame
        DataFrame con datos preprocesados.
    """
    logging.info("Procesando datos priorizando XML sobre Excel...")
    
    # Procesar datos XML usando pl_ohio_only.py
    xml_data_frames = []
    column_mapping = {}  # Para mapear nombres de columnas entre fuentes
    all_columns = set()  # Para rastrear todas las columnas de todos los DataFrames
    
    # 1. Procesar datos XML en paralelo (Ohio dataset)
    logging.info(f"Iniciando procesamiento paralelo de {len(OHIO_DATA_DIRS)} directorios XML...")
    xml_data_frames_results = Parallel(n_jobs=-1)(
        delayed(process_xml_directory)(data_dir)
        for data_dir in OHIO_DATA_DIRS
    )

    # Filtrar resultados None y extraer columnas
    xml_data_frames = [df for df in xml_data_frames_results if df is not None]
    for df in xml_data_frames:
        for col in df.columns:
            all_columns.add(col)

    logging.info(f"Procesamiento XML completado. Obtenidos {len(xml_data_frames)} DataFrames válidos.")
    
    # Setear SubjectIDs a numericos
    for i, df in enumerate(xml_data_frames):
        if "SubjectID" in df.columns:
            # Asegurar que SubjectID es numérico
            if df["SubjectID"].dtype != pl.Int64:
                xml_data_frames[i] = df.with_columns(
                    pl.col("SubjectID").map_elements(
                        lambda x: extract_numeric_id(str(x)) if x is not None else None
                    ).cast(pl.Int64)
                )
        
    # 2. Procesar datos Excel desde la carpeta de sujetos
    excel_data_frames = []  # Lista para almacenar DataFrames procesados
    if USE_EXCEL_DATA:
        subject_files: list[str] = [f for f in os.listdir(DATA_PATH_SUBJECTS) if f.startswith("Subject") and f.endswith(".xlsx")]
        logging.info(f"\nArchivos de sujetos encontrados ({len(subject_files)}):")
        for f in subject_files:
            logging.info(f)

        # Procesar datos Excel en paralelo usando joblib
        excel_data: list[dict] = Parallel(n_jobs=-1)(
            delayed(process_excel_subject)(os.path.join(DATA_PATH_SUBJECTS, f), idx)
            for idx, f in enumerate(subject_files)
        )
        # Aplanar la lista de resultados
        excel_data = [item for sublist in excel_data for item in sublist if item is not None]

        if excel_data:
            # Convertir a DataFrame
            df_excel: pl.DataFrame = pl.DataFrame(excel_data)
            
            # Realizar mapeo de columnas si es necesario
            column_mappings = {
                "subject_id": "SubjectID",      # Mapear ID del sujeto
                "cgm_window": "cgm_window",     # Mantener ventana CGM
                "carb_input": "bwz_carb_input", # Mapear carbohidratos
                "bg_input": "glucose_last",     # Mapear entrada de glucosa
                "bolus": "bolus"                # Mantener nombre de bolus
            }
            
            # Aplicar mapeos de columnas necesarios
            for excel_col, xml_col in column_mappings.items():
                if excel_col in df_excel.columns and xml_col not in df_excel.columns:
                    df_excel = df_excel.rename({excel_col: xml_col})
            
            # Aplicar las mismas transformaciones de características que a los datos XML
            print_debug(f"Columnas df_excel: {df_excel.columns}")
            df_excel = extract_enhanced_features_excel(df_excel)
            df_excel = transform_enhanced_features_excel(df_excel)
            
            # Guardar nombres de columnas
            for col in df_excel.columns:
                all_columns.add(col)
            
            # Añadir a la lista de DataFrames
            excel_data_frames.append(df_excel)
            logging.info(f"DataFrame Excel procesado: {df_excel.shape}")
        else:
            excel_data_frames = []
            logging.warning("No se pudieron procesar datos Excel")
    
    # 3. Unificar todos los DataFrames
    print_debug(f"All columns: {all_columns}") 
    all_data_frames = xml_data_frames + excel_data_frames
    
    if not all_data_frames:
        raise ValueError("No se pudieron procesar datos de ninguna fuente")
    
    print_info(f"Total de DataFrames procesados: {len(all_data_frames)}")
    
    # 4. Asegurar que todos los DataFrames tengan las mismas columnas antes de concatenar
    logging.info(f"Armonizando {len(all_data_frames)} DataFrames con {len(all_columns)} columnas...")

    # Primero determinar el tipo de cada columna a partir de los DataFrames existentes
    column_types = {}
    for df in all_data_frames:
        for col in df.columns:
            column_types[col] = df[col].dtype

    for i, df in enumerate(all_data_frames):
        # Identificar columnas faltantes en este DataFrame
        missing_cols = all_columns - set(df.columns)
        
        # Añadir columnas faltantes con valores nulos y tipo correcto
        if missing_cols:
            logging.info(f"Añadiendo {len(missing_cols)} columnas faltantes al DataFrame {i+1}")
            for col in missing_cols:
                # Si conocemos el tipo, crear columna con ese tipo específico
                if col in column_types:
                    df = df.with_columns(
                        pl.lit(None).cast(column_types[col]).alias(col)
                    )
                else:
                    # Tipo por defecto para columnas donde no conocemos el tipo
                    # Integer para columnas que podrían ser numéricas (excepto algunas específicas)
                    if col in ['meals_in_window', 'hypo_episodes_24h', 'hyper_episodes_24h']:
                        df = df.with_columns(
                            pl.lit(None).cast(pl.Int32).alias(col)
                        )
                    # Float para columnas que probablemente sean numéricas con valores decimales
                    elif any(prefix in col for prefix in ['glucose_', 'time_', 'meal_', 'risk_', 'percentage']):
                        df = df.with_columns(
                            pl.lit(None).cast(pl.Float64).alias(col)
                        )
                    else:
                        df = df.with_columns(pl.lit(None).alias(col))
        
        # Actualizar el DataFrame en la lista
        all_data_frames[i] = df
    
    # 5. Asegurar el mismo orden de columnas y precisión en todos los DataFrames
    ordered_columns = sorted(all_columns)
    for i, df in enumerate(all_data_frames):
        # Primero seleccionar las columnas en el mismo orden
        df = df.select(ordered_columns)
        # Luego unificar la precisión de las columnas datetime
        df = unify_datetime_precision(df)
        all_data_frames[i] = df

    # 6. Unificar tipos de columnas antes de concatenar
    all_data_frames = unify_column_types(all_data_frames)

    # 6. Concatenar todos los DataFrames
    logging.info(f"Concatenando {len(all_data_frames)} DataFrames...")
    final_df = pl.concat(all_data_frames)
    
    logging.info(f"Procesamiento completado. Forma final: {final_df.shape}")
    return final_df

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