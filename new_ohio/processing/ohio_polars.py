import polars as pl
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import glob
import numpy as np
from typing import List, Dict, Union, Optional, Tuple
import xml.etree.ElementTree as ET
from collections import defaultdict
from datetime import timedelta, datetime
import pandas as pd
import argparse
from joblib import Parallel, delayed
import logging
from zoneinfo import ZoneInfo
import matplotlib.dates as mdates
from scipy import stats
from sklearn.linear_model import TheilSenRegressor
from sklearn.metrics import r2_score
import os

# Configuración de colores para logging
class ColoredFormatter(logging.Formatter):
    """Formateador personalizado para agregar colores a los mensajes de logging"""
    
    COLORS = {
        'DEBUG': '\033[94m',     # Azul
        'INFO': '\033[92m',      # Verde
        'WARNING': '\033[93m',   # Amarillo
        'ERROR': '\033[91m',     # Rojo
        'CRITICAL': '\033[91m\033[1m'  # Rojo negrita
    }
    
    def format(self, record):
        if record.levelname in self.COLORS:
            record.msg = f"{self.COLORS[record.levelname]}{record.msg}\033[0m"
        return super().format(record)

# Configuración del logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# Crear directorio de logs si no existe
log_dir = os.path.join(os.path.dirname(__file__), 'logs')
os.makedirs(log_dir, exist_ok=True)

# Configurar el formateador con colores
formatter = ColoredFormatter(
    '%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

# Handler para consola
console_handler = logging.StreamHandler()
console_handler.setFormatter(formatter)
logger.addHandler(console_handler)

# Handler para archivo (modo 'w' para sobrescribir en cada ejecución)
file_handler = logging.FileHandler(
    os.path.join(log_dir, 'logs.txt'),
    mode='w',
    encoding='utf-8'
)
file_handler.setFormatter(logging.Formatter(
    '%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
))
logger.addHandler(file_handler)

# Configuración global
CONFIG = {
    # Parámetros de procesamiento
    "window_size": 12,  # Tamaño de ventana en puntos CGM (1 hora)
    "extended_window_size": 288,  # Tamaño de ventana extendida (24 horas)
    "min_cgm_points": 12,  # Mínimo de puntos CGM para ventana válida
    "alignment_tolerance_minutes": 15,  # Tolerancia para alineación de eventos
    "random_seed": 42,  # Semilla para reproducibilidad
    
    # Parámetros de simulación glucémica
    "carb_effect_factor": 5,  # mg/dL por gramo de carbohidratos
    "insulin_effect_factor": 50,  # mg/dL por unidad de insulina
    "glucose_min": 40,  # Límite inferior de glucosa
    "glucose_max": 400,  # Límite superior de glucosa
    
    # Parámetros de bolus
    "bolus_max": 20.0,  # Máximo valor de bolus considerado válido
    "bolus_min": 0.1,  # Mínimo valor de bolus considerado válido
    
    # Parámetros de ventanas
    "window_steps": 12,  # Pasos en cada ventana
    "partial_window_threshold": 6,  # Mínimo de pasos para ventana parcial
    
    # Parámetros de logging
    "log_bolus_loss": True,  # Registrar pérdida de eventos bolus
    "log_window_discard": True,  # Registrar ventanas descartadas
    "log_simulation": True,  # Registrar resultados de simulación
    "log_basal_estimation": True,  # Registrar estimación de basal
    "log_outliers": True,  # Registrar outliers
    "log_cv": True,  # Registrar validación cruzada
    "log_clinical_metrics": True,  # Registrar métricas clínicas
    
    # Parámetros de eventos
    "event_tolerance": timedelta(minutes=15),  # Tolerancia para unir eventos
    "basal_estimation_hours": (0, 24),  # Horas para estimar tasa basal
    "basal_estimation_factor": 0.5,  # Factor para estimar tasa basal
    
    # Umbrales clínicos
    "hypo_threshold": 70,  # Umbral para hipoglucemia (mg/dL)
    "hyper_threshold": 180,  # Umbral para hiperglucemia (mg/dL)
    "tir_lower": 70,  # Límite inferior de rango glucémico (mg/dL)
    "tir_upper": 180,  # Límite superior de rango glucémico (mg/dL)
    "significant_meal_threshold": 20,  # Umbral de gramos para considerar una comida significativa
    
    # Parámetros de simulación
    "simulation_steps": 72,  # Número de pasos para simular glucosa
    
    # Parámetros de riesgo
    "risk_weights": {
        'hypo': 0.3,
        'hyper': 0.3,
        'variability': 0.2,
        'sleep_hypo': 0.1,
        'activity_hypo': 0.05,
        'stress_hyper': 0.05
    }
}

def load_data(data_dir: str) -> Dict[str, pl.DataFrame]:
    """
    Carga los datos y muestra las columnas de cada DataFrame.
    Verifica que estén presentes los 6 sujetos esperados para cada año (2018 y 2020).
    Ahora soporta: glucose_level, bolus, meal, basal, temp_basal, exercise, basis_steps, hypo_event,
    finger_stick, sleep, work, stressors, illness, basis_heart_rate, basis_gsr, basis_skin_temperature,
    basis_air_temperature, basis_sleep, acceleration.
    """
    logger.info(f"Cargando datos desde {data_dir}")
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
        logger.error(f"Faltan datos para sujetos del año {year}: {missing_subjects}")
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
        logger.info(f"\n{'='*50}")
        logger.info(f"Procesando SubjectID: {subject_id} (Año {year})")
        logger.info(f"{'='*50}")
        
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
                    # logger.info(f"SubjectID {subject_id}: {data_type}={len(records)} registros")
                    subject_stats[subject_id][data_type] += len(records)
                    
                    # Mostrar estadísticas de valores para columnas numéricas
                    if 'value' in df.columns:
                        value_stats = df.select(pl.col('value')).describe()
                        # logger.info(f"Estadísticas de valores para {data_type}:")
                        # logger.info(f"  Min: {value_stats['value'].min():.2f}")
                        # logger.info(f"  Max: {value_stats['value'].max():.2f}")
                        # logger.info(f"  Mean: {value_stats['value'].mean():.2f}")
                        # logger.info(f"  Std: {value_stats['value'].std():.2f}")
        
        except Exception as e:
            logger.error(f"Error procesando {xml_file}: {e}")
            continue
    
    # Mostrar columnas por tipo de evento y sujeto
    # logger.info("\nColumnas por tipo de evento y sujeto:")
    # for subject_id in sorted(event_columns_by_subject.keys()):
    #     logger.info(f"\n{'-'*50}")
    #     logger.info(f"SubjectID: {subject_id}")
    #     for event_type, columns in sorted(event_columns_by_subject[subject_id].items()):
    #         logger.info(f"  {event_type}: {sorted(columns)}")
    
    # logger.info(f"\nEstadísticas por sujeto (Año {year}):")
    # for subject_id in sorted(subject_stats.keys()):
    #     stats = subject_stats[subject_id]
    #     stat_str = ", ".join([f"{k}={v}" for k, v in stats.items()])
    #     logger.info(f"SubjectID {subject_id}: {stat_str}")
    
    missing_types = [t for t in expected_types if t not in data_dict]
    if missing_types:
        logger.warning(f"Faltan tipos de datos: {missing_types}")
    if len(subject_stats) != len(expected_subjects[year]):
        logger.error(f"Se encontraron datos para {len(subject_stats)}/{len(expected_subjects[year])} sujetos")
    
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
        logger.info(f"Eventos bolus válidos: {valid_bolus.height}")

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
        logger.info(f"Eventos meal válidos: {valid_meal.height}")

    # Basal
    if "basal" in data:
        basal = data["basal"].clone()
        basal = basal.rename({"value": "basal_rate"}).with_columns(
            pl.col("basal_rate").cast(pl.Float64),
            pl.col("ts").str.strptime(pl.Datetime, "%d-%m-%Y %H:%M:%S").alias("Timestamp")
        )
        processed["basal"] = basal.filter(pl.col("basal_rate").is_not_null())
        logger.info(f"Eventos basal válidos: {processed['basal'].height}")

    # Temp Basal
    if "temp_basal" in data:
        temp_basal = data["temp_basal"].clone()
        temp_basal = temp_basal.rename({"value": "temp_basal_rate"}).with_columns(
            pl.col("temp_basal_rate").cast(pl.Float64),
            pl.col("ts_begin").str.strptime(pl.Datetime, "%d-%m-%Y %H:%M:%S").alias("Timestamp")
        )
        processed["temp_basal"] = temp_basal.filter(pl.col("temp_basal_rate").is_not_null())
        logger.info(f"Eventos temp_basal válidos: {processed['temp_basal'].height}")

    # Exercise
    if "exercise" in data:
        exercise = data["exercise"].clone()
        exercise = exercise.with_columns(
            pl.col("intensity").cast(pl.Float64),
            pl.col("duration").cast(pl.Float64),
            pl.col("ts").str.strptime(pl.Datetime, "%d-%m-%Y %H:%M:%S").alias("Timestamp")
        )
        processed["exercise"] = exercise.filter(pl.col("intensity").is_not_null())
        logger.info(f"Eventos exercise válidos: {processed['exercise'].height}")

    # Steps
    if "basis_steps" in data:
        steps = data["basis_steps"].clone()
        steps = steps.rename({"value": "steps"}).with_columns(
            pl.col("steps").cast(pl.Float64),
            pl.col("ts").str.strptime(pl.Datetime, "%d-%m-%Y %H:%M:%S").alias("Timestamp")
        )
        processed["basis_steps"] = steps.filter(pl.col("steps").is_not_null())
        logger.info(f"Eventos steps válidos: {processed['basis_steps'].height}")

    # Hypo Event
    if "hypo_event" in data:
        hypo = data["hypo_event"].clone()
        if "ts" in hypo.columns:
            hypo = hypo.with_columns(
                pl.col("ts").str.strptime(pl.Datetime, "%d-%m-%Y %H:%M:%S").alias("Timestamp")
            )
        processed["hypo_event"] = hypo
        logger.info(f"Eventos hypo_event válidos: {processed['hypo_event'].height}")

    # Finger Stick
    if "finger_stick" in data:
        finger_stick = data["finger_stick"].clone()
        finger_stick = finger_stick.rename({"value": "finger_stick_bg"}).with_columns(
            pl.col("finger_stick_bg").cast(pl.Float64),
            pl.col("ts").str.strptime(pl.Datetime, "%d-%m-%Y %H:%M:%S").alias("Timestamp")
        )
        processed["finger_stick"] = finger_stick.filter(pl.col("finger_stick_bg").is_not_null())
        logger.info(f"Eventos finger_stick válidos: {processed['finger_stick'].height}")

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
        logger.info(f"Eventos sleep válidos: {processed['sleep'].height}")

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
        logger.info(f"Eventos work válidos: {processed['work'].height}")

    # Stressors
    if "stressors" in data:
        stressors = data["stressors"].clone()
        if "ts" in stressors.columns:
            stressors = stressors.with_columns(
                pl.col("ts").str.strptime(pl.Datetime, "%d-%m-%Y %H:%M:%S").alias("Timestamp")
            )
        processed["stressors"] = stressors
        logger.info(f"Eventos stressors válidos: {processed['stressors'].height}")

    # Illness
    if "illness" in data:
        illness = data["illness"].clone()
        if "ts" in illness.columns:
            illness = illness.with_columns(
                pl.col("ts").str.strptime(pl.Datetime, "%d-%m-%Y %H:%M:%S").alias("Timestamp")
            )
        processed["illness"] = illness
        logger.info(f"Eventos illness válidos: {processed['illness'].height}")

    # Basis Heart Rate
    if "basis_heart_rate" in data:
        heart_rate = data["basis_heart_rate"].clone()
        heart_rate = heart_rate.rename({"value": "heart_rate"}).with_columns(
            pl.col("heart_rate").cast(pl.Float64),
            pl.col("ts").str.strptime(pl.Datetime, "%d-%m-%Y %H:%M:%S").alias("Timestamp")
        )
        processed["basis_heart_rate"] = heart_rate.filter(pl.col("heart_rate").is_not_null())
        logger.info(f"Eventos heart_rate válidos: {processed['basis_heart_rate'].height}")

    # Basis GSR
    if "basis_gsr" in data:
        gsr = data["basis_gsr"].clone()
        gsr = gsr.rename({"value": "gsr"}).with_columns(
            pl.col("gsr").cast(pl.Float64),
            pl.col("ts").str.strptime(pl.Datetime, "%d-%m-%Y %H:%M:%S").alias("Timestamp")
        )
        processed["basis_gsr"] = gsr.filter(pl.col("gsr").is_not_null())
        logger.info(f"Eventos gsr válidos: {processed['basis_gsr'].height}")

    # Basis Skin Temperature
    if "basis_skin_temperature" in data:
        skin_temp = data["basis_skin_temperature"].clone()
        skin_temp = skin_temp.rename({"value": "skin_temperature"}).with_columns(
            pl.col("skin_temperature").cast(pl.Float64),
            pl.col("ts").str.strptime(pl.Datetime, "%d-%m-%Y %H:%M:%S").alias("Timestamp")
        )
        processed["basis_skin_temperature"] = skin_temp.filter(pl.col("skin_temperature").is_not_null())
        logger.info(f"Eventos skin_temperature válidos: {processed['basis_skin_temperature'].height}")

    # Basis Air Temperature
    if "basis_air_temperature" in data:
        air_temp = data["basis_air_temperature"].clone()
        air_temp = air_temp.rename({"value": "air_temperature"}).with_columns(
            pl.col("air_temperature").cast(pl.Float64),
            pl.col("ts").str.strptime(pl.Datetime, "%d-%m-%Y %H:%M:%S").alias("Timestamp")
        )
        processed["basis_air_temperature"] = air_temp.filter(pl.col("air_temperature").is_not_null())
        logger.info(f"Eventos air_temperature válidos: {processed['basis_air_temperature'].height}")

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
        logger.info(f"Eventos basis_sleep válidos: {processed['basis_sleep'].height}")

    # Acceleration
    if "acceleration" in data:
        accel = data["acceleration"].clone()
        accel = accel.rename({"value": "acceleration"}).with_columns(
            pl.col("acceleration").cast(pl.Float64),
            pl.col("ts").str.strptime(pl.Datetime, "%d-%m-%Y %H:%M:%S").alias("Timestamp")
        )
        processed["acceleration"] = accel.filter(pl.col("acceleration").is_not_null())
        logger.info(f"Eventos acceleration válidos: {processed['acceleration'].height}")

    return processed

def align_events_to_cgm(cgm_df: pl.DataFrame, event_df: pl.DataFrame, event_time_col: str = "Timestamp", tolerance_minutes: int = 5) -> pl.DataFrame:
    """
    Alinea los eventos con las mediciones CGM más cercanas dentro de una tolerancia temporal.
    
    Args:
        cgm_df: DataFrame con datos CGM
        event_df: DataFrame con eventos a alinear
        event_time_col: Nombre de la columna de tiempo en event_df
        tolerance_minutes: Tolerancia máxima en minutos para alinear eventos
        
    Returns:
        DataFrame con eventos alineados a CGM
    """
    # Asegurar que las columnas de tiempo son datetime
    cgm_df = ensure_timestamp_datetime(cgm_df)
    event_df = ensure_timestamp_datetime(event_df, event_time_col)
    
    # Convertir tolerancia a timedelta
    tolerance = timedelta(minutes=tolerance_minutes)
    
    # Obtener timestamps de CGM
    cgm_times = cgm_df["Timestamp"].to_list()
    
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
            # Si no hay CGM cercano, mantener el tiempo original
            aligned_times.append(event_time)
    
    # Crear nuevo DataFrame con tiempos alineados
    aligned_df = event_df.with_columns(
        pl.Series(name="Timestamp", values=aligned_times)
    )
    
    # Filtrar eventos que no se pudieron alinear
    aligned_df = aligned_df.filter(
        pl.col("Timestamp").is_in(cgm_times)
    )
    
    # Calcular estadísticas de alineación
    total_events = len(event_df)
    aligned_events = len(aligned_df)
    lost_events = total_events - aligned_events
    
    if lost_events > 0:
        logger.warning(f"Eventos descartados por estar fuera de tolerancia: {lost_events}")
        logger.warning(f"Eventos sin CGM correspondiente: {lost_events}")
    
    # logger.info(f"Eventos alineados: {aligned_events}/{total_events} ({aligned_events/total_events*100:.1f}%)")
    
    return aligned_df

def preprocess_cgm(cgm: pl.DataFrame) -> pl.DataFrame:
    if "ts" in cgm.columns:
        cgm = cgm.with_columns(
            pl.col("ts").str.strptime(pl.Datetime, "%d-%m-%Y %H:%M:%S").alias("Timestamp")
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
                logger.error(f"No se pudo convertir Timestamp a datetime: {e}")
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
            logger.error("Columna 'value' no encontrada en cgm_df")
            raise ValueError("Columna 'value' no encontrada en cgm_df")
            
        # Asegurar que la columna bolus existe en bolus_df
        if 'bolus' not in bolus_df.columns:
            logger.error("Columna 'bolus' no encontrada en bolus_df")
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
            logger.error("Columna 'value' perdida después de join con bolus")
            raise ValueError("Columna 'value' perdida después de join con bolus")
        if 'bolus' not in joined_df.columns:
            logger.error("Columna 'bolus' perdida después de join con bolus")
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
            logger.error("Columna 'value' perdida después de join con meal")
            raise ValueError("Columna 'value' perdida después de join con meal")
        if 'bolus' not in joined_df.columns:
            logger.error("Columna 'bolus' perdida después de join con meal")
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
                # logger.info(f"Señal physiological_value: {non_null_percentage:.1f}% valores no nulos")
        
        # Verificar columnas críticas una última vez
        if 'value' not in joined_df.columns:
            logger.error("Columna 'value' perdida después de join con physiological")
            raise ValueError("Columna 'value' perdida después de join con physiological")
        if 'bolus' not in joined_df.columns:
            logger.error("Columna 'bolus' perdida después de join con physiological")
            raise ValueError("Columna 'bolus' perdida después de join con physiological")
        
        # Registrar forma final y columnas presentes
        # logger.info(f"Forma de datos unidos: {joined_df.shape}")
        # logger.info(f"Columnas presentes: {joined_df.columns}")
        
        return joined_df
        
    except Exception as e:
        logger.error(f"Error en join_signals: {e}")
        raise

def ensure_timestamp_datetime(df: pl.DataFrame, col: str = "Timestamp") -> pl.DataFrame:
    """
    Convierte la columna 'Timestamp' a pl.Datetime, tolerando np.datetime64, string y datetime.
    """
    if col in df.columns:
        # Si es object (np.datetime64 o string), primero a string
        if df[col].dtype == pl.Object:
            df = df.with_columns(
                pl.col(col).map_elements(
                    lambda x: str(x) if x is not None else None,
                    return_dtype=pl.Utf8
                ).alias(col)
            )
        # Finalmente casteá a datetime
        df = df.with_columns(
            pl.col(col).cast(pl.Datetime)
        )
    return df

def generate_windows(df: pl.DataFrame, window_size: int = 12) -> pl.DataFrame:
    """
    Genera ventanas de datos para entrenamiento, manejando ventanas parciales
    y mejorando el logging de ventanas descartadas.
    """
    logger.info("Generando ventanas de datos...")
    
    # Asegurar que los datos están ordenados por tiempo
    df = df.sort("Timestamp")
    
    # Inicializar lista para ventanas
    windows = []
    discarded_windows = []
    
    # Obtener timestamps únicos
    timestamps = df["Timestamp"].unique()
    
    for timestamp in timestamps:
        # Obtener datos de la ventana
        window_start = timestamp - timedelta(minutes=5 * (window_size - 1))
        window_data = df.filter(
            (pl.col("Timestamp") >= window_start) & 
            (pl.col("Timestamp") <= timestamp)
        )
        
        # Verificar si la ventana es válida
        if len(window_data) == window_size:
            windows.append(window_data)
        else:
            # Intentar generar ventana parcial si hay suficientes puntos
            if len(window_data) >= CONFIG["partial_window_threshold"]:
                # Calcular valor medio para rellenar
                mean_value = window_data["value"].mean()
                padding_size = window_size - len(window_data)
                
                # Crear datos de relleno
                padding_data = pl.DataFrame({
                    "Timestamp": [window_start + timedelta(minutes=5 * i) for i in range(padding_size)],
                    "value": [mean_value] * padding_size,
                    "SubjectID": [window_data["SubjectID"][0]] * padding_size
                })
                
                # Combinar datos originales y relleno
                window_data = pl.concat([padding_data, window_data])
                windows.append(window_data)
                
                logger.info(
                    f"Ventana parcial generada para SubjectID {window_data['SubjectID'][0]}, "
                    f"Timestamp {timestamp}: {len(window_data)}/{window_size} pasos"
                )
            else:
                discarded_windows.append({
                    "SubjectID": window_data["SubjectID"][0] if len(window_data) > 0 else "Unknown",
                    "Timestamp": timestamp,
                    "Steps": len(window_data),
                    "Reason": "Pasos insuficientes"
                })
    
    # Reportar estadísticas de ventanas
    total_windows = len(timestamps)
    valid_windows = len(windows)
    discarded_count = len(discarded_windows)
    
    logger.info(f"Total de ventanas: {total_windows}")
    logger.info(f"Ventanas válidas: {valid_windows}")
    logger.info(f"Ventanas descartadas: {discarded_count}")
    
    # Reportar detalles de ventanas descartadas
    if discarded_count > 0:
        # logger.warning("Detalles de ventanas descartadas:")
        # for window in discarded_windows:
        #     logger.warning(
        #         f"SubjectID: {window['SubjectID']}, "
        #         f"Timestamp: {window['Timestamp']}, "
        #         f"Pasos: {window['Steps']}, "
        #         f"Razón: {window['Reason']}"
        #     )
        pass
    
    # Combinar todas las ventanas válidas
    if windows:
        result = pl.concat(windows)
        # logger.info(f"Ventanas generadas: {result.shape}")
        return result
    else:
        logger.error("No se generaron ventanas válidas")
        return pl.DataFrame()

def extract_features(df: pl.DataFrame, meal_df: Optional[pl.DataFrame] = None) -> pl.DataFrame:
    """
    Wrapper legado para extracción de características mejorada para mantener compatibilidad.
    """
    logger.info("Usando pipeline de extracción de características mejorado...")
    return extract_enhanced_features(df, meal_df)

def transform_features(df: pl.DataFrame) -> pl.DataFrame:
    """
    Wrapper legado para transformación de características mejorada para mantener compatibilidad.
    """
    logger.info("Usando pipeline de transformación de características mejorado...")
    return transform_enhanced_features(df)

def generate_extended_windows(df: pl.DataFrame, 
                            window_size: int = 12,
                            prediction_horizon: int = 12) -> Tuple[pl.DataFrame, pl.DataFrame]:
    """Genera ventanas extendidas con características adicionales"""
    
    def safe_mean(series):
        """Calcula la media de forma segura, manejando valores nulos"""
        if series.is_null().all():
            return 0.0
        non_null = series.drop_nulls()
        if len(non_null) == 0:
            return 0.0
        return float(non_null.mean())
    
    def safe_std(series):
        """Calcula la desviación estándar de forma segura, manejando valores nulos"""
        if series.is_null().all():
            return 0.0
        non_null = series.drop_nulls()
        if len(non_null) <= 1:
            return 0.0
        return float(non_null.std())
    
    def safe_min(series):
        """Calcula el mínimo de forma segura, manejando valores nulos"""
        if series.is_null().all():
            return 0.0
        non_null = series.drop_nulls()
        if len(non_null) == 0:
            return 0.0
        return float(non_null.min())
    
    def safe_max(series):
        """Calcula el máximo de forma segura, manejando valores nulos"""
        if series.is_null().all():
            return 0.0
        non_null = series.drop_nulls()
        if len(non_null) == 0:
            return 0.0
        return float(non_null.max())
    
    def safe_float(value):
        """Convierte un valor a float de forma segura, manejando None"""
        if value is None or pd.isna(value):
            return 0.0
        return float(value)
    
    # Lista para almacenar las características
    X_features = []
    y_values = []
    
    # Obtener columnas numéricas (excluyendo Timestamp y SubjectID)
    numeric_cols = [col for col in df.columns 
                   if col not in ['Timestamp', 'SubjectID'] 
                   and df[col].dtype in [pl.Float64, pl.Int64]]
    
    # Iterar sobre cada ventana
    for i in range(len(df) - window_size - prediction_horizon + 1):
        window = df.slice(i, window_size)
        future_window = df.slice(i + window_size, prediction_horizon)
        
        # Características básicas
        features = {
            'Timestamp': window['Timestamp'].last(),
            'SubjectID': window['SubjectID'].last()
        }
        
        # Preservar el valor actual de glucosa y bolus de forma segura
        if 'value' in window.columns:
            features['value'] = safe_float(window['value'].last())
        if 'bolus' in window.columns:
            features['bolus'] = safe_float(window['bolus'].last())
        
        # Calcular características para cada columna numérica
        for col in numeric_cols:
            try:
                features[f'{col}_mean'] = safe_mean(window[col])
                features[f'{col}_std'] = safe_std(window[col])
                features[f'{col}_min'] = safe_min(window[col])
                features[f'{col}_max'] = safe_max(window[col])
            except Exception as e:
                logger.warning(f"Error calculando estadísticas para columna {col}: {e}")
                features[f'{col}_mean'] = 0.0
                features[f'{col}_std'] = 0.0
                features[f'{col}_min'] = 0.0
                features[f'{col}_max'] = 0.0
        
        X_features.append(features)
        y_values.append(future_window['value'].to_list())
    
    # Convertir a DataFrames con tipos de datos explícitos
    schema = {
        'Timestamp': pl.Datetime,
        'SubjectID': pl.Utf8,
        'value': pl.Float64,
        'bolus': pl.Float64
    }
    
    # Añadir esquema para columnas derivadas
    for col in numeric_cols:
        schema.update({
            f'{col}_mean': pl.Float64,
            f'{col}_std': pl.Float64,
            f'{col}_min': pl.Float64,
            f'{col}_max': pl.Float64
        })
    
    # Crear DataFrame con esquema explícito
    X_df = pl.DataFrame(X_features, schema=schema)
    y_df = pl.DataFrame({'future_values': y_values})
    
    # Verificar que las columnas críticas están presentes
    critical_columns = ['value', 'bolus']
    missing_critical = [col for col in critical_columns if col not in X_df.columns]
    if missing_critical:
        logger.error(f"Faltan columnas críticas después de generar ventanas: {missing_critical}")
        raise ValueError(f"Faltan columnas críticas después de generar ventanas: {missing_critical}")
    
    logger.info(f"Ventanas generadas: X={X_df.shape}, y={y_df.shape}")
    logger.info(f"Columnas en X_df: {X_df.columns}")
    
    return X_df, y_df

def compute_glucose_patterns_24h(cgm_values: List[float], physiological_data: Optional[Dict[str, List[float]]] = None) -> Dict[str, float]:
    """
    Calcula patrones de glucosa de 24 horas para análisis clínico, incluyendo señales fisiológicas.
    
    Args:
        cgm_values: Lista de valores de glucosa (idealmente 24h)
        physiological_data: Diccionario opcional con señales fisiológicas (heart_rate, gsr, etc.)
        
    Returns:
        Diccionario con características de patrones de glucosa y fisiológicas
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
            'glucose_trend_24h': 0.0,
            'heart_rate_std_24h': 0.0,
            'sleep_percentage_24h': 0.0,
            'stress_percentage_24h': 0.0,
            'activity_percentage_24h': 0.0
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
    hypo_episodes = np.sum(values_array < CONFIG['hypo_threshold'])
    patterns['hypo_episodes_24h'] = int(hypo_episodes)
    patterns['hypo_percentage_24h'] = float((hypo_episodes / len(values_array)) * 100)
    
    # Hyperglycemia analysis (> 180 mg/dL)
    hyper_episodes = np.sum(values_array > CONFIG['hyper_threshold'])
    patterns['hyper_episodes_24h'] = int(hyper_episodes)
    patterns['hyper_percentage_24h'] = float((hyper_episodes / len(values_array)) * 100)
    
    # Time in Range (70-180 mg/dL)
    in_range = np.sum((values_array >= CONFIG['tir_lower']) & (values_array <= CONFIG['tir_upper']))
    patterns['time_in_range_24h'] = float((in_range / len(values_array)) * 100)
    
    # Glucose variability
    if len(values_array) > 1 and patterns['cgm_mean_24h'] > 0:
        # Coefficient of variation
        patterns['cv_24h'] = float((patterns['cgm_std_24h'] / patterns['cgm_mean_24h']) * 100)
        
        # Mean Absolute Glucose change (MAGE)
        glucose_changes = np.abs(np.diff(values_array))
        patterns['mage_24h'] = float(np.mean(glucose_changes))
        
        # Long-term trend analysis
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
    
    # Physiological patterns if available
    if physiological_data:
        # Heart rate variability
        if 'heart_rate' in physiological_data and len(physiological_data['heart_rate']) > 0:
            hr_values = np.array(physiological_data['heart_rate'])
            patterns['heart_rate_std_24h'] = float(np.std(hr_values))
        else:
            patterns['heart_rate_std_24h'] = 0.0
        
        # Sleep analysis
        if 'sleep_event' in physiological_data and len(physiological_data['sleep_event']) > 0:
            sleep_events = np.array(physiological_data['sleep_event'])
            patterns['sleep_percentage_24h'] = float((np.sum(sleep_events) / len(sleep_events)) * 100)
        else:
            patterns['sleep_percentage_24h'] = 0.0
        
        # Stress analysis
        if 'stressors_event' in physiological_data and len(physiological_data['stressors_event']) > 0:
            stress_events = np.array(physiological_data['stressors_event'])
            patterns['stress_percentage_24h'] = float((np.sum(stress_events) / len(stress_events)) * 100)
        else:
            patterns['stress_percentage_24h'] = 0.0
        
        # Activity analysis
        if 'work_event' in physiological_data and len(physiological_data['work_event']) > 0:
            work_events = np.array(physiological_data['work_event'])
            patterns['activity_percentage_24h'] = float((np.sum(work_events) / len(work_events)) * 100)
        else:
            patterns['activity_percentage_24h'] = 0.0
    else:
        patterns['heart_rate_std_24h'] = 0.0
        patterns['sleep_percentage_24h'] = 0.0
        patterns['stress_percentage_24h'] = 0.0
        patterns['activity_percentage_24h'] = 0.0
    
    return patterns

def encode_time_cyclical(timestamp: datetime) -> Dict[str, float]:
    """
    Codifica características de tiempo cíclicamente usando seno y coseno.
    
    Args:
        timestamp: Objeto datetime
        
    Returns:
        Diccionario con características de tiempo cíclicas
    """
    # Codificación de hora del día (0-24)
    hour_decimal = timestamp.hour + timestamp.minute / 60.0 + timestamp.second / 3600.0
    hour_normalized = hour_decimal / 24.0
    hour_radians = 2 * np.pi * hour_normalized
    
    # Codificación de día de la semana (0-7)
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
                                physiological_df: Optional[pl.DataFrame] = None,
                                window_hours: float = 2.0) -> Dict[str, Union[float, int]]:
    """
    Calcula características mejoradas del contexto de comidas alrededor del tiempo del bolo,
    incluyendo señales fisiológicas y eventos.
    
    Args:
        bolus_time: Tiempo de administración del bolo
        meal_df: DataFrame con datos de comidas
        physiological_df: DataFrame opcional con señales fisiológicas y eventos
        window_hours: Horas para buscar alrededor del tiempo del bolo
        
    Returns:
        Diccionario con características del contexto de comidas y fisiológicas
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
            'meal_timing_score': 0.0,
            'activity_level': 0.0,
            'stress_level': 0.0,
            'sleep_state': 0.0,
            'heart_rate_mean': 0.0,
            'gsr_mean': 0.0,
            'skin_temperature_mean': 0.0
        }
    
    # Definir ventana de tiempo alrededor del bolo
    start_time = bolus_time - timedelta(hours=window_hours/2)
    end_time = bolus_time + timedelta(hours=window_hours)
    
    # Filtrar comidas en la ventana
    meals_in_window = meal_df.filter(
        (pl.col("Timestamp") >= start_time) & 
        (pl.col("Timestamp") <= end_time)
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
            'meal_timing_score': 0.0,
            'activity_level': 0.0,
            'stress_level': 0.0,
            'sleep_state': 0.0,
            'heart_rate_mean': 0.0,
            'gsr_mean': 0.0,
            'skin_temperature_mean': 0.0
        }
    
    # Encontrar comida más cercana
    meals_with_diff = meals_in_window.with_columns(
        (pl.col("Timestamp") - bolus_time).abs().alias("time_diff_abs")
    ).sort("time_diff_abs")
    
    closest_meal = meals_with_diff.row(0, named=True)
    time_diff = (closest_meal["Timestamp"] - bolus_time).total_seconds()
    meal_carbs = float(closest_meal.get("meal_carbs", 0.0))
    
    # Calcular métricas adicionales
    all_meals = meals_in_window.to_dicts()
    total_carbs = sum(float(meal.get("meal_carbs", 0.0)) for meal in all_meals)
    largest_meal = max((float(meal.get("meal_carbs", 0.0)) for meal in all_meals), default=0.0)
    
    # Puntuación de tiempo de comida (1.0 = tiempo perfecto, 0.0 = tiempo pobre)
    max_diff_seconds = window_hours * 3600
    timing_score = max(0.0, 1.0 - abs(time_diff) / max_diff_seconds)
    
    # Características fisiológicas y eventos si están disponibles
    physiological_features = {
        'activity_level': 0.0,
        'stress_level': 0.0,
        'sleep_state': 0.0,
        'heart_rate_mean': 0.0,
        'gsr_mean': 0.0,
        'skin_temperature_mean': 0.0
    }
    
    if physiological_df is not None:
        # Filtrar datos fisiológicos en la ventana
        phys_window = physiological_df.filter(
            (pl.col("Timestamp") >= start_time) & 
            (pl.col("Timestamp") <= end_time)
        )
        
        if not phys_window.is_empty():
            # Calcular nivel de actividad
            if 'work_event' in phys_window.columns:
                physiological_features['activity_level'] = float(phys_window['work_event'].mean())
            
            # Calcular nivel de estrés
            if 'stressors_event' in phys_window.columns:
                physiological_features['stress_level'] = float(phys_window['stressors_event'].mean())
            
            # Estado de sueño
            if 'sleep_event' in phys_window.columns:
                physiological_features['sleep_state'] = float(phys_window['sleep_event'].mean())
            
            # Señales fisiológicas promedio
            if 'heart_rate' in phys_window.columns:
                physiological_features['heart_rate_mean'] = float(phys_window['heart_rate'].mean())
            
            if 'gsr' in phys_window.columns:
                physiological_features['gsr_mean'] = float(phys_window['gsr'].mean())
            
            if 'skin_temperature' in phys_window.columns:
                physiological_features['skin_temperature_mean'] = float(phys_window['skin_temperature'].mean())
    
    return {
        'meal_carbs': meal_carbs,
        'meal_time_diff_minutes': float(time_diff / 60.0),
        'meal_time_diff_hours': float(time_diff / 3600.0),
        'has_meal': 1.0,
        'meals_in_window': len(all_meals),
        'significant_meal': 1.0 if meal_carbs > CONFIG['significant_meal_threshold'] else 0.0,
        'total_carbs_window': total_carbs,
        'largest_meal_carbs': largest_meal,
        'meal_timing_score': timing_score,
        **physiological_features
    }

def compute_clinical_risk_indicators(glucose_values: List[float], 
                                   physiological_data: Optional[Dict[str, List[float]]] = None,
                                   time_values: Optional[List[datetime]] = None) -> Dict[str, float]:
    """
    Calcula indicadores de riesgo clínico basados en valores de glucosa y señales fisiológicas.
    
    Args:
        glucose_values: Lista de valores de glucosa
        physiological_data: Diccionario opcional con señales fisiológicas
        time_values: Lista opcional de timestamps correspondientes a los valores
        
    Returns:
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
    hypo_threshold = CONFIG.get('hypo_threshold', 70)
    hyper_threshold = CONFIG.get('hyper_threshold', 180)
    
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
    risk_weights = CONFIG.get('risk_weights', {
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

def calculate_insulin_on_board(df: pl.DataFrame, current_time: datetime, 
                             insulin_duration_hours: float = 4.0) -> float:
    """
    Calcula la insulina activa usando modelo de decaimiento exponencial.
    
    Args:
        df: DataFrame con historial de bolus
        current_time: Timestamp actual
        insulin_duration_hours: Duración de la acción de la insulina
        
    Returns:
        Valor actual de insulina activa
    """
    # Filtrar eventos bolus dentro de la duración de la insulina
    cutoff_time = current_time - timedelta(hours=insulin_duration_hours)
    recent_bolus = df.filter(
        (pl.col("Timestamp") >= cutoff_time) & 
        (pl.col("Timestamp") <= current_time) &
        (pl.col("bolus").is_not_null()) &
        (pl.col("bolus") > 0)
    )
    
    if recent_bolus.is_empty():
        return 0.0
    
    iob = 0.0
    decay_constant = np.log(2) / (insulin_duration_hours * 60)  # Decaimiento basado en vida media
    
    for row in recent_bolus.iter_rows(named=True):
        bolus_time = row["Timestamp"]
        bolus_dose = float(row["bolus"])
        
        # Tiempo desde el bolo en minutos
        time_diff_minutes = (current_time - bolus_time).total_seconds() / 60.0
        
        # Decaimiento exponencial
        remaining_fraction = np.exp(-decay_constant * time_diff_minutes)
        iob += bolus_dose * remaining_fraction
    
    return float(iob)

def extract_enhanced_features(df: pl.DataFrame, meal_df: Optional[pl.DataFrame] = None,
                            extended_cgm_df: Optional[pl.DataFrame] = None) -> pl.DataFrame:
    """
    Extrae características mejoradas del DataFrame, incluyendo características CGM,
    fisiológicas y de eventos.
    
    Args:
        df: DataFrame con datos unidos
        meal_df: DataFrame opcional con datos de comidas
        extended_cgm_df: DataFrame opcional con datos CGM extendidos
        
    Returns:
        DataFrame con características extraídas
    """
    # Verificar columnas críticas
    if 'value' not in df.columns:
        logger.error("Columna 'value' no encontrada en el DataFrame de entrada")
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
        logger.info(f"{group}: {len(available_features[group])}/{len(features)} características presentes")
    
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
            
            logger.info("Usando datos actuales para calcular patrones de 24h")
    
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
        logger.info(f"{group} generadas: {len(present)}/{len(features)}")
        if len(present) < len(features):
            missing = set(features) - set(present)
            logger.warning(f"Faltan características de {group}: {missing}")
    
    # Verificar columnas críticas al final
    critical_columns = ['value', 'time_in_range_24h', 'bolus']
    missing_critical = [col for col in critical_columns if col not in df.columns]
    if missing_critical:
        logger.error(f"Faltan columnas críticas al final: {missing_critical}")
        raise ValueError(f"Faltan columnas críticas al final: {missing_critical}")
    
    return df

def transform_enhanced_features(df: pl.DataFrame) -> pl.DataFrame:
    """
    Apply enhanced transformations including log transforms, normalization,
    and feature expansion for the 52-dimensional observation space.
    """
    logger.info("Aplicando transformaciones mejoradas...")
    
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
        window_size = CONFIG["window_steps"]
        
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
    
    logger.info(f"Transformaciones mejoradas completadas. Forma final: {df.shape}")
    
    return df

def simulate_glucose(cgm_values: list, bolus: float, carbs: float, basal_rate: float, exercise_intensity: float, steps: int = 72):
    """
    Simula glucosa para DRL, considerando basal y ejercicio.
    Devuelve también la glucosa simulada a las 2 horas (24 pasos) y el TIR en 2 horas.
    """
    if not cgm_values:
        return {
            'simulated_tir_6h': 0.0,
            'simulated_mean_6h': 120.0,
            'simulated_std_6h': 0.0,
            'simulated_hypo_6h': 0.0,
            'simulated_hyper_6h': 0.0,
            'simulated_glucose_2h': 120.0,
            'simulated_tir_2h': 0.0
        }
    
    current_glucose = cgm_values[-1]
    
    # Cálculo simplificado de glucosa a 2 horas
    glucose_change = carbs * 5 - bolus * 10  # 5 mg/dL por gramo de carbos, -10 mg/dL por unidad de insulina
    simulated_glucose_2h = current_glucose + glucose_change
    
    # Limitar valores entre 40 y 400 mg/dL
    simulated_glucose_2h = max(CONFIG['glucose_min'], min(CONFIG['glucose_max'], simulated_glucose_2h))
    
    # Calcular TIR en 2 horas
    tir_2h = 1.0 if CONFIG['tir_lower'] <= simulated_glucose_2h <= CONFIG['tir_upper'] else 0.0
    
    # Para mantener compatibilidad con el resto del código, generamos valores simulados para 6h
    simulated_values = [simulated_glucose_2h] * steps
    
    return {
        'simulated_tir_6h': float(tir_2h),  # Usamos el mismo valor para mantener compatibilidad
        'simulated_mean_6h': float(simulated_glucose_2h),
        'simulated_std_6h': 0.0,
        'simulated_hypo_6h': float(simulated_glucose_2h < CONFIG['hypo_threshold']),
        'simulated_hyper_6h': float(simulated_glucose_2h > CONFIG['hyper_threshold']),
        'simulated_glucose_2h': float(simulated_glucose_2h),
        'simulated_tir_2h': float(tir_2h)
    }

def estimate_basal_rate(df: pl.DataFrame, subject_id: str) -> float:
    """
    Estima la tasa basal promedio usando datos nocturnos sin bolus ni comidas.
    
    Args:
        df: DataFrame con datos del sujeto
        subject_id: ID del sujeto
        
    Returns:
        Tasa basal estimada en U/h
    """
    # Filtrar datos nocturnos sin bolus ni comidas
    night_data = df.filter(
        (pl.col("SubjectID") == subject_id) &
        (pl.col("bolus").is_null()) &
        (pl.col("meal_carbs").is_null()) &
        (pl.col("Timestamp").dt.hour().is_between(*CONFIG['basal_estimation_hours']))
    )
    
    if night_data.is_empty():
        return 0.0
    
    # Calcular tasa basal basada en la variación de glucosa
    glucose_changes = night_data["value"].diff().abs()
    basal_estimate = glucose_changes.mean() / CONFIG['basal_estimation_factor']
    
    if CONFIG['log_basal_estimation']:
        logger.info(f"Insulina basal estimada para SubjectID {subject_id}: {basal_estimate:.2f} U/h")
    
    return float(basal_estimate)

def plot_clinical_metrics(df: pl.DataFrame, output_dir: str):
    """Genera visualizaciones de métricas clínicas y las guarda en el directorio especificado"""
    import matplotlib.pyplot as plt
    import seaborn as sns
    from pathlib import Path
    import pandas as pd
    import numpy as np
    
    # Crear directorio si no existe
    plot_dir = Path(output_dir)
    plot_dir.mkdir(parents=True, exist_ok=True)
    
    # Convertir a pandas para visualización
    df_pd = df.to_pandas()
    
    # Configurar estilo
    plt.style.use('seaborn')
    
    # Función auxiliar para guardar plots de forma segura
    def safe_save_plot(fig, filename):
        try:
            plt.savefig(str(plot_dir / filename), dpi=100, bbox_inches='tight')
            logger.info(f"Plot generado: {filename}")
        except Exception as e:
            logger.error(f"Error guardando plot {filename}: {e}")
        finally:
            plt.close(fig)
    
    # Función para filtrar valores extremos
    def filter_extremes(data, column, lower_percentile=1, upper_percentile=99):
        lower = np.percentile(data[column].dropna(), lower_percentile)
        upper = np.percentile(data[column].dropna(), upper_percentile)
        return data[(data[column] >= lower) & (data[column] <= upper)]
    
    try:
        # 1. Distribución de glucosa
        fig, ax = plt.subplots(figsize=(10, 6))
        filtered_data = filter_extremes(df_pd, 'value')
        sns.histplot(data=filtered_data, x='value', bins=50)
        plt.title('Distribución de Valores de Glucosa')
        plt.xlabel('Glucosa (mg/dL)')
        plt.ylabel('Frecuencia')
        safe_save_plot(fig, 'glucose_distribution.png')
        
        # 2. Tiempo en rango por sujeto
        if 'time_in_range_24h' in df_pd.columns:
            fig, ax = plt.subplots(figsize=(12, 6))
            x_col = 'SubjectID' if 'SubjectID' in df_pd.columns else df_pd.index
            sns.boxplot(data=df_pd, x=x_col, y='time_in_range_24h')
            plt.title('Tiempo en Rango por Sujeto')
            plt.xlabel('SubjectID' if 'SubjectID' in df_pd.columns else 'Índice')
            plt.ylabel('TIR (%)')
            plt.xticks(rotation=45)
            safe_save_plot(fig, 'time_in_range.png')
        
        # 3. Distribución de bolus
        if 'bolus' in df_pd.columns:
            fig, ax = plt.subplots(figsize=(10, 6))
            valid_bolus = df_pd[(df_pd['bolus'] > 0) & (df_pd['bolus'] < 20)]
            sns.histplot(data=valid_bolus, x='bolus', bins=30)
            plt.title('Distribución de Dosis de Bolus')
            plt.xlabel('Dosis (U)')
            plt.ylabel('Frecuencia')
            safe_save_plot(fig, 'bolus_distribution.png')
        
        # 4. Patrones de glucosa por hora
        if 'Timestamp' in df_pd.columns:
            fig, ax = plt.subplots(figsize=(12, 6))
            df_pd['hour'] = pd.to_datetime(df_pd['Timestamp']).dt.hour
            valid_data = filter_extremes(df_pd, 'value')
            sns.boxplot(data=valid_data, x='hour', y='value')
            plt.title('Patrones de Glucosa por Hora')
            plt.xlabel('Hora del Día')
            plt.ylabel('Glucosa (mg/dL)')
            safe_save_plot(fig, 'glucose_by_hour.png')
        
        # 5. Correlación entre variables
        if len(df_pd.columns) > 2:
            fig, ax = plt.subplots(figsize=(12, 8))
            numeric_cols = df_pd.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) > 1:
                corr_matrix = df_pd[numeric_cols].corr()
                sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0)
                plt.title('Matriz de Correlación')
                safe_save_plot(fig, 'correlation_matrix.png')
        
    except Exception as e:
        logger.error(f"Error generando plots: {e}")
    finally:
        plt.close('all')  # Cerrar todas las figuras al finalizar

def generate_cv_splits(df: pl.DataFrame, output_dir: str):
    """
    Genera splits de validación cruzada y los guarda como conjuntos de prueba sintéticos.
    
    Args:
        df: DataFrame con datos procesados
        output_dir: Directorio para guardar los splits
    """
    if not CONFIG['log_cv']:
        return
    
    from sklearn.model_selection import KFold
    
    # Crear directorio si no existe
    Path(output_dir).mkdir(exist_ok=True, parents=True)
    
    # Inicializar KFold
    kf = KFold(n_splits=CONFIG['cv_folds'], shuffle=True, random_state=CONFIG['random_seed'])
    
    # Generar splits
    for fold, (train_idx, test_idx) in enumerate(kf.split(df)):
        train_data = df[train_idx]
        test_data = df[test_idx]
        
        if CONFIG['log_cv']:
            logger.info(f"Fold {fold+1}: Train={len(train_data)} filas, Test={len(test_data)} filas")
        
        # Guardar split de prueba
        test_data.write_parquet(f"{output_dir}/synthetic_test_fold_{fold+1}.parquet")
    
    logger.info(f"Splits de validación cruzada guardados en {output_dir}")

def main():
    """Enhanced main function with improved preprocessing pipeline."""
    parser = argparse.ArgumentParser(description='Procesar dataset OhioT1DM con características mejoradas')
    parser.add_argument('--data-dirs', nargs='+', 
                      default=['data/OhioT1DM/2018/train', 'data/OhioT1DM/2020/train',
                              'data/OhioT1DM/2018/test', 'data/OhioT1DM/2020/test'],
                      help='Lista de directorios de datos a procesar')
    parser.add_argument('--output-dir', default='new_ohio/processed_data',
                      help='Directorio de salida para datos procesados')
    parser.add_argument('--plots-dir', default='new_ohio/processed_data/plots',
                      help='Directorio de salida para gráficos')
    parser.add_argument('--timezone', default='UTC',
                      help='Zona horaria a usar para timestamps')
    parser.add_argument('--n-jobs', type=int, default=-1,
                      help='Número de trabajos paralelos (-1 para todos los núcleos)')
    parser.add_argument('--enhanced', action='store_true', default=True,
                      help='Usar características de preprocesamiento mejoradas')
    args = parser.parse_args()
    
    CONFIG['timezone'] = args.timezone
    
    # Crear directorios necesarios
    Path(args.output_dir).mkdir(exist_ok=True)
    Path(args.plots_dir).mkdir(exist_ok=True, parents=True)
    
    logger.info("=" * 60)
    logger.info("PIPELINE DE PREPROCESAMIENTO MEJORADO OHIO T1DM")
    logger.info("=" * 60)
    logger.info(f"Características mejoradas: {args.enhanced}")
    logger.info(f"Umbrales clínicos: Hipoglucemia<{CONFIG['hypo_threshold']}, Hiperglucemia>{CONFIG['hyper_threshold']}")
    logger.info(f"Tiempo en Rango: {CONFIG['tir_lower']}-{CONFIG['tir_upper']} mg/dL")
    
    for data_dir in args.data_dirs:
        logger.info(f"\nProcesando directorio: {data_dir}")
        logger.info("=" * 50)
        
        try:
            logger.info("Cargando datos crudos...")
            data = load_data(data_dir)
            if not data:
                logger.warning(f"No se encontraron datos en {data_dir}")
                continue
                
            logger.info("Preprocesando y alineando eventos...")
            processed = preprocess_bolus_meal(data)
            
            # Preprocesar glucose_level una sola vez
            data['glucose_level'] = preprocess_cgm(data['glucose_level'])
            
            # Alinear eventos con CGM
            bolus_aligned = align_events_to_cgm(data['glucose_level'], processed["bolus"])
            meal_aligned = align_events_to_cgm(data['glucose_level'], processed["meal"])
            
            # Actualizar datos con eventos alineados
            data['bolus'] = bolus_aligned
            data['meal'] = meal_aligned

            logger.info(f"Alineando y uniendo señales para {data_dir}")            
            for key in ["glucose_level", "bolus", "meal"]:
                if key in data and "Timestamp" in data[key].columns:
                    data[key] = ensure_timestamp_datetime(data[key], "Timestamp")
                    
            # Unir señales con los DataFrames correctos
            df = join_signals(
                cgm_df=data['glucose_level'],
                bolus_df=data['bolus'],
                meal_df=data['meal'],
                physiological_df=data.get('basis_heart_rate')  # Opcional
            )
            logger.info(f"Forma de datos unidos: {df.shape}")

            if "value" in df.columns:
                df = df.with_columns(pl.col("value").cast(pl.Float64))
            if "bolus" in df.columns:
                df = df.with_columns(pl.col("bolus").cast(pl.Float64))
                    
            n_bolus_total = df.filter(pl.col("bolus") > 0).height
            logger.info(f"Total de eventos bolus: {n_bolus_total}")

            logger.info("Generando ventanas mejoradas con patrones a largo plazo...")
            if args.enhanced:
                X_df, y_df = generate_extended_windows(
                    df, 
                    window_size=CONFIG["window_steps"],
                    prediction_horizon=CONFIG["window_steps"]  # Usar el mismo tamaño para predicción
                )
                logger.info(f"Ventanas generadas: X={X_df.shape}, y={y_df.shape}")
            else:
                df_windows = generate_windows(df, window_size=CONFIG["window_size"])
                logger.info(f"Ventanas generadas: {df_windows.shape}")
            
            logger.info("Extrayendo características clínicas mejoradas...")
            if args.enhanced:
                df_features = extract_enhanced_features(
                    X_df,  # Usar X_df en lugar de df_windows
                    meal_df=data.get('meal'),
                    extended_cgm_df=df
                )
            else:
                df_features = extract_features(df_windows, data.get('meal'))
            
            logger.info(f"Características extraídas: {df_features.shape}")
            
            if args.enhanced:
                new_features = [col for col in df_features.columns if any(keyword in col for keyword in 
                    ['24h', '_sin', '_cos', 'hypo_risk', 'hyper_risk', 'stability', 'timing_score', 'composite'])]
                logger.info(f"Características mejoradas añadidas: {len(new_features)} nuevas características clínicas")
                logger.info(f"Muestra de nuevas características: {new_features[:10]}")

            logger.info("Aplicando transformaciones mejoradas...")
            if args.enhanced:
                df_final = transform_enhanced_features(df_features)
            else:
                df_final = transform_features(df_features)
                
            logger.info(f"Datos procesados finales: {df_final.shape}")
            
            if args.enhanced:
                expected_cgm_cols = [f"cgm_{i}" for i in range(CONFIG["window_steps"])]
                cgm_cols_present = [col for col in df_final.columns if col in expected_cgm_cols]
                logger.info(f"Columnas CGM: {len(cgm_cols_present)}/{len(expected_cgm_cols)}")
                
                enhanced_feature_categories = {
                    'tiempo_cíclico': ['hour_sin', 'hour_cos', 'day_sin', 'day_cos'],
                    'patrones_glucosa': ['cgm_mean_24h', 'time_in_range_24h', 'hypo_percentage_24h'],
                    'contexto_comida': ['meal_timing_score', 'significant_meal', 'total_carbs_window'],
                    'indicadores_riesgo': ['current_hypo_risk', 'current_hyper_risk', 'stability_score'],
                    'métricas_clínicas': ['composite_hypo_risk', 'composite_hyper_risk', 'iob_risk_factor']
                }
                
                for category, features in enhanced_feature_categories.items():
                    present = [f for f in features if f in df_final.columns]
                    logger.info(f"{category}: {len(present)}/{len(features)} características presentes")
            
            is_test = 'test' in data_dir
            output_subdir = 'test' if is_test else 'train'
            year = Path(data_dir).parts[-2]
            split = Path(data_dir).name
            
            if args.enhanced:
                output_filename = f"processed_enhanced_{year}_{split}.parquet"
            else:
                output_filename = f"processed_{year}_{split}.parquet"

            output_path = f"{args.output_dir}/{output_subdir}/{output_filename}"
            Path(f"{args.output_dir}/{output_subdir}").mkdir(exist_ok=True, parents=True)
            
            df_final.write_parquet(output_path)
            logger.info(f"Datos mejorados exportados a {output_path}")
            
            if 'bolus' in df_final.columns:
                bolus_stats = df_final.select(pl.col('bolus')).describe()
                # logger.info(f"Estadísticas de bolus:\n{bolus_stats}")
            
            if args.enhanced and 'time_in_range_24h' in df_final.columns:
                tir_mean = df_final.select(pl.col('time_in_range_24h')).mean().item()
                # logger.info(f"Tiempo en Rango promedio: {tir_mean:.1f}%")

            # Generar plots para cada conjunto de datos
            if CONFIG['log_clinical_metrics']:
                plot_dir = f"{args.plots_dir}/{year}_{split}"
                Path(plot_dir).mkdir(exist_ok=True, parents=True)
                # logger.info(f"Generando plots en {plot_dir}")
                
                try:
                    # Convertir a pandas para plotting
                    df_pd = df_final.to_pandas()
                    
                    # Verificar columnas requeridas
                    required_columns = {
                        'value': 'glucose_distribution.png',
                        'time_in_range_24h': 'tir_by_subject.png',
                        'bolus': 'bolus_distribution.png',
                        'Timestamp': 'glucose_by_hour.png'
                    }
                    
                    # Generar plots solo si las columnas requeridas están presentes
                    for col, plot_name in required_columns.items():
                        if col in df_pd.columns:
                            try:
                                # Cerrar todas las figuras existentes antes de crear una nueva
                                plt.close('all')
                                
                                # Crear nueva figura para cada plot
                                fig = plt.figure(figsize=(12, 6))
                                
                                if col == 'value':
                                    # Filtrar valores extremos para mejor visualización
                                    valid_data = df_pd[df_pd[col].between(0, 400)]
                                    sns.histplot(data=valid_data, x=col, bins=50)
                                    plt.title(f'Distribución de Glucosa - {year} {split}')
                                    plt.xlabel('Glucosa (mg/dL)')
                                    plt.ylabel('Frecuencia')
                                
                                elif col == 'time_in_range_24h':
                                    # Usar índice como x si no hay SubjectID
                                    x_col = 'SubjectID' if 'SubjectID' in df_pd.columns else df_pd.index
                                    sns.boxplot(data=df_pd, x=x_col, y=col)
                                    plt.title(f'Tiempo en Rango por Sujeto - {year} {split}')
                                    plt.xlabel('SubjectID' if 'SubjectID' in df_pd.columns else 'Índice')
                                    plt.ylabel('TIR (%)')
                                    plt.xticks(rotation=45)
                                
                                elif col == 'bolus':
                                    # Filtrar solo valores positivos y razonables
                                    valid_bolus = df_pd[(df_pd[col] > 0) & (df_pd[col] < 20)]
                                    sns.histplot(data=valid_bolus, x=col, bins=30)
                                    plt.title(f'Distribución de Dosis de Bolus - {year} {split}')
                                    plt.xlabel('Dosis (U)')
                                    plt.ylabel('Frecuencia')
                                
                                elif col == 'Timestamp':
                                    df_pd['hour'] = pd.to_datetime(df_pd[col]).dt.hour
                                    # Filtrar valores extremos para mejor visualización
                                    valid_data = df_pd[df_pd['value'].between(0, 400)]
                                    sns.boxplot(data=valid_data, x='hour', y='value')
                                    plt.title(f'Patrones de Glucosa por Hora - {year} {split}')
                                    plt.xlabel('Hora del Día')
                                    plt.ylabel('Glucosa (mg/dL)')
                                
                                plt.tight_layout()
                                
                                # Guardar figura con manejo de errores
                                try:
                                    plt.savefig(f"{plot_dir}/{plot_name}", dpi=100, bbox_inches='tight')
                                    # logger.info(f"Plot generado: {plot_name}")
                                except Exception as save_error:
                                    logger.error(f"Error guardando plot {plot_name}: {save_error}")
                                finally:
                                    plt.close(fig)  # Cerrar la figura específica
                                
                            except Exception as e:
                                logger.error(f"Error generando plot {plot_name}: {e}")
                                plt.close('all')  # Cerrar todas las figuras en caso de error
                        else:
                            logger.warning(f"No se pudo generar {plot_name}: columna {col} no encontrada")
                    
                    logger.info(f"Plots generados y guardados en {plot_dir}")
                    
                except Exception as e:
                    logger.error(f"Error en la generación de plots: {e}")
                    plt.close('all')  # Cerrar todas las figuras en caso de error
                finally:
                    plt.close('all')  # Asegurar que todas las figuras se cierran al finalizar
                    # Liberar memoria
                    if 'df_pd' in locals():
                        del df_pd
                    import gc
                    gc.collect()

        except Exception as e:
            logger.error(f"Error procesando {data_dir}: {e}")
            import traceback
            logger.error(traceback.format_exc())
            continue

    logger.info("=" * 60)
    logger.info("PREPROCESAMIENTO MEJORADO COMPLETADO")
    logger.info("=" * 60)

if __name__ == "__main__":
    main()
