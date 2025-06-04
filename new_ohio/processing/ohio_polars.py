import polars as pl
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import glob
import numpy as np
from typing import List, Dict, Union, Optional
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

# Configurar logging con colores
logger = logging.getLogger()
handler = logging.StreamHandler()
handler.setFormatter(ColoredFormatter('%(asctime)s - %(levelname)s - %(message)s'))
logger.addHandler(handler)
logger.setLevel(logging.INFO)

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
    "event_tolerance": timedelta(minutes=15),  # Tolerancia para unir eventos
    "basal_estimation_hours": (0, 24),  # Horas para estimar tasa basal
    "basal_estimation_factor": 0.5,  # Factor para estimar tasa basal
    "log_basal_estimation": True,  # Registrar estimación de basal
    "hypoglycemia_threshold": 70,  # Umbral para hipoglucemia
    "hyperglycemia_threshold": 180,  # Umbral para hiperglucemia
    "tir_lower": 70,  # Límite inferior de rango glucémico
    "tir_upper": 180,  # Límite superior de rango glucémico
    "simulation_steps": 72,  # Número de pasos para simular glucosa
    # --- Agregados para evitar KeyError ---
    "hypo_risk_threshold": 70,  # Igual a hypoglycemia_threshold
    "hyper_risk_threshold": 180,  # Igual a hyperglycemia_threshold
    "significant_meal_threshold": 20,  # Umbral de gramos para considerar una comida significativa
}

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
                    record_dict['SubjectID'] = subject_id
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
        logging.warning(f"Eventos descartados por estar fuera de tolerancia: {lost_events}")
        logging.warning(f"Eventos sin CGM correspondiente: {lost_events}")
    
    logging.info(f"Eventos alineados: {aligned_events}/{total_events} ({aligned_events/total_events*100:.1f}%)")
    
    return aligned_df

def preprocess_cgm(cgm: pl.DataFrame) -> pl.DataFrame:
    if "ts" in cgm.columns:
        cgm = cgm.with_columns(
            pl.col("ts").str.strptime(pl.Datetime, "%d-%m-%Y %H:%M:%S").alias("Timestamp")
        )
    return cgm

def join_signals(data: Dict[str, pl.DataFrame]) -> pl.DataFrame:
    """
    Une todas las señales CGM, bolus, meal, basal, temp_basal, exercise, steps, hypo_event,
    finger_stick, sleep, work, stressors, illness, basis_heart_rate, basis_gsr, basis_skin_temperature,
    basis_air_temperature, basis_sleep, acceleration en un único DataFrame.
    Optimizado para DRL. Robusto ante ausencia de columnas.
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

    # Join bolus y meal (siempre tienen Timestamp por preprocess)
    df = cgm_data.join(bolus_data, on="Timestamp", how="left", suffix="_bolus")
    df = df.join(meal_data, on="Timestamp", how="left", suffix="_meal")
    
    # Basal
    if not basal_data.is_empty() and "Timestamp" in basal_data.columns and "Timestamp" in df.columns:
        df = df.join(basal_data, on="Timestamp", how="left", suffix="_basal")
        df = df.with_columns(pl.col("basal_rate").fill_null(0.0))
    else:
        df = df.with_columns(pl.lit(0.0).alias("basal_rate"))
    
    # Temp Basal
    if not temp_basal_data.is_empty() and "Timestamp" in temp_basal_data.columns and "Timestamp" in df.columns:
        df = df.join(temp_basal_data, on="Timestamp", how="left", suffix="_temp_basal")
        df = df.with_columns(
            pl.col("temp_basal_rate").fill_null(0.0),
            pl.col("temp_basal_rate").gt(0).alias("temp_basal_active"),
            pl.when(pl.col("temp_basal_active")).then(pl.col("temp_basal_rate")).otherwise(pl.col("basal_rate")).alias("effective_basal_rate")
        )
    else:
        df = df.with_columns(
            pl.col("basal_rate").alias("effective_basal_rate"),
            pl.lit(0.0).alias("temp_basal_active")
        )
    
    # Exercise
    if not exercise_data.is_empty() and "Timestamp" in exercise_data.columns and "Timestamp" in df.columns:
        exercise_aligned = align_events_to_cgm(df, exercise_data, tolerance_minutes=15)
        df = df.join(exercise_aligned, on="Timestamp", how="left", suffix="_exercise")
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
    if not steps_data.is_empty() and "Timestamp" in steps_data.columns and "Timestamp" in df.columns:
        steps_aligned = align_events_to_cgm(df, steps_data, tolerance_minutes=15)
        df = df.join(steps_aligned, on="Timestamp", how="left", suffix="_steps")
        df = df.with_columns(
            pl.col("steps").fill_null(0.0).alias("steps")
        )
    else:
        df = df.with_columns(
            pl.lit(0.0).alias("steps")
        )
    
    # Hypo event (binaria)
    if not hypo_data.is_empty() and "Timestamp" in hypo_data.columns and "Timestamp" in df.columns:
        hypo_aligned = align_events_to_cgm(df, hypo_data, tolerance_minutes=15)
        hypo_timestamps = hypo_aligned["Timestamp"].unique().to_list()
        df = df.with_columns(
            pl.col("Timestamp").map_elements(lambda x: float(x in hypo_timestamps), return_dtype=pl.Float64).alias("hypo_event")
        )
    else:
        df = df.with_columns(
            pl.lit(0.0).alias("hypo_event")
        )

    # Finger Stick
    if not finger_stick_data.is_empty() and "Timestamp" in finger_stick_data.columns and "Timestamp" in df.columns:
        finger_stick_aligned = align_events_to_cgm(df, finger_stick_data, tolerance_minutes=15)
        df = df.join(finger_stick_aligned, on="Timestamp", how="left", suffix="_finger_stick")
        df = df.with_columns(
            pl.col("finger_stick_bg").fill_null(0.0)
        )
    else:
        df = df.with_columns(
            pl.lit(0.0).alias("finger_stick_bg")
        )

    # Sleep (binaria)
    if not sleep_data.is_empty() and "Timestamp_begin" in sleep_data.columns and "Timestamp" in df.columns:
        sleep_aligned = align_events_to_cgm(df, sleep_data, tolerance_minutes=15)
        sleep_timestamps = sleep_aligned["Timestamp"].unique().to_list()
        df = df.with_columns(
            pl.col("Timestamp").map_elements(lambda x: float(x in sleep_timestamps), return_dtype=pl.Float64).alias("sleep_event")
        )
    else:
        df = df.with_columns(
            pl.lit(0.0).alias("sleep_event")
        )

    # Work (binaria)
    if not work_data.is_empty() and "Timestamp_begin" in work_data.columns and "Timestamp" in df.columns:
        work_aligned = align_events_to_cgm(df, work_data, tolerance_minutes=15)
        work_timestamps = work_aligned["Timestamp"].unique().to_list()
        df = df.with_columns(
            pl.col("Timestamp").map_elements(lambda x: float(x in work_timestamps), return_dtype=pl.Float64).alias("work_event")
        )
    else:
        df = df.with_columns(
            pl.lit(0.0).alias("work_event")
        )

    # Stressors (binaria)
    if not stressors_data.is_empty() and "Timestamp" in stressors_data.columns and "Timestamp" in df.columns:
        stressors_aligned = align_events_to_cgm(df, stressors_data, tolerance_minutes=15)
        stressors_timestamps = stressors_aligned["Timestamp"].unique().to_list()
        df = df.with_columns(
            pl.col("Timestamp").map_elements(lambda x: float(x in stressors_timestamps), return_dtype=pl.Float64).alias("stressors_event")
        )
    else:
        df = df.with_columns(
            pl.lit(0.0).alias("stressors_event")
        )

    # Illness (binaria)
    if not illness_data.is_empty() and "Timestamp" in illness_data.columns and "Timestamp" in df.columns:
        illness_aligned = align_events_to_cgm(df, illness_data, tolerance_minutes=15)
        illness_timestamps = illness_aligned["Timestamp"].unique().to_list()
        df = df.with_columns(
            pl.col("Timestamp").map_elements(lambda x: float(x in illness_timestamps), return_dtype=pl.Float64).alias("illness_event")
        )
    else:
        df = df.with_columns(
            pl.lit(0.0).alias("illness_event")
        )

    # Basis Heart Rate
    if not heart_rate_data.is_empty() and "Timestamp" in heart_rate_data.columns and "Timestamp" in df.columns:
        heart_rate_aligned = align_events_to_cgm(df, heart_rate_data, tolerance_minutes=15)
        df = df.join(heart_rate_aligned, on="Timestamp", how="left", suffix="_heart_rate")
        df = df.with_columns(
            pl.col("heart_rate").fill_null(0.0)
        )
    else:
        df = df.with_columns(
            pl.lit(0.0).alias("heart_rate")
        )

    # Basis GSR
    if not gsr_data.is_empty() and "Timestamp" in gsr_data.columns and "Timestamp" in df.columns:
        gsr_aligned = align_events_to_cgm(df, gsr_data, tolerance_minutes=15)
        df = df.join(gsr_aligned, on="Timestamp", how="left", suffix="_gsr")
        df = df.with_columns(
            pl.col("gsr").fill_null(0.0)
        )
    else:
        df = df.with_columns(
            pl.lit(0.0).alias("gsr")
        )

    # Basis Skin Temperature
    if not skin_temp_data.is_empty() and "Timestamp" in skin_temp_data.columns and "Timestamp" in df.columns:
        skin_temp_aligned = align_events_to_cgm(df, skin_temp_data, tolerance_minutes=15)
        df = df.join(skin_temp_aligned, on="Timestamp", how="left", suffix="_skin_temp")
        df = df.with_columns(
            pl.col("skin_temperature").fill_null(0.0)
        )
    else:
        df = df.with_columns(
            pl.lit(0.0).alias("skin_temperature")
        )

    # Basis Air Temperature
    if not air_temp_data.is_empty() and "Timestamp" in air_temp_data.columns and "Timestamp" in df.columns:
        air_temp_aligned = align_events_to_cgm(df, air_temp_data, tolerance_minutes=15)
        df = df.join(air_temp_aligned, on="Timestamp", how="left", suffix="_air_temp")
        df = df.with_columns(
            pl.col("air_temperature").fill_null(0.0)
        )
    else:
        df = df.with_columns(
            pl.lit(0.0).alias("air_temperature")
        )

    # Basis Sleep (binaria)
    if not basis_sleep_data.is_empty() and "Timestamp_begin" in basis_sleep_data.columns and "Timestamp" in df.columns:
        basis_sleep_aligned = align_events_to_cgm(df, basis_sleep_data, tolerance_minutes=15)
        basis_sleep_timestamps = basis_sleep_aligned["Timestamp"].unique().to_list()
        df = df.with_columns(
            pl.col("Timestamp").map_elements(lambda x: float(x in basis_sleep_timestamps), return_dtype=pl.Float64).alias("basis_sleep_event")
        )
    else:
        df = df.with_columns(
            pl.lit(0.0).alias("basis_sleep_event")
        )

    # Acceleration
    if not acceleration_data.is_empty() and "Timestamp" in acceleration_data.columns and "Timestamp" in df.columns:
        acceleration_aligned = align_events_to_cgm(df, acceleration_data, tolerance_minutes=15)
        df = df.join(acceleration_aligned, on="Timestamp", how="left", suffix="_acceleration")
        df = df.with_columns(
            pl.col("acceleration").fill_null(0.0)
        )
    else:
        df = df.with_columns(
            pl.lit(0.0).alias("acceleration")
        )

    # Imputar valores faltantes para DRL
    for col in ["bolus", "meal_carbs", "effective_basal_rate", "exercise_intensity", "exercise_duration", 
                "steps", "hypo_event", "finger_stick_bg", "sleep_event", "work_event", "stressors_event", 
                "illness_event", "heart_rate", "gsr", "skin_temperature", "air_temperature", 
                "basis_sleep_event", "acceleration"]:
        if col in df.columns:
            df = df.with_columns(
                pl.col(col).fill_null(0.0)
            )
    
    logging.info(f"Señales unidas: {df.shape}")
    return df

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
    logging.info("Generando ventanas de datos...")
    
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
                
                logging.info(
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
    
    logging.info(f"Total de ventanas: {total_windows}")
    logging.info(f"Ventanas válidas: {valid_windows}")
    logging.info(f"Ventanas descartadas: {discarded_count}")
    
    # Reportar detalles de ventanas descartadas
    if discarded_count > 0:
        logging.warning("Detalles de ventanas descartadas:")
        for window in discarded_windows:
            logging.warning(
                f"SubjectID: {window['SubjectID']}, "
                f"Timestamp: {window['Timestamp']}, "
                f"Pasos: {window['Steps']}, "
                f"Razón: {window['Reason']}"
            )
    
    # Combinar todas las ventanas válidas
    if windows:
        result = pl.concat(windows)
        logging.info(f"Ventanas generadas: {result.shape}")
        return result
    else:
        logging.error("No se generaron ventanas válidas")
        return pl.DataFrame()

def extract_features(df: pl.DataFrame, meal_df: Optional[pl.DataFrame] = None) -> pl.DataFrame:
    """
    Wrapper legado para extracción de características mejorada para mantener compatibilidad.
    """
    logging.info("Usando pipeline de extracción de características mejorado...")
    return extract_enhanced_features(df, meal_df)

def transform_features(df: pl.DataFrame) -> pl.DataFrame:
    """
    Wrapper legado para transformación de características mejorada para mantener compatibilidad.
    """
    logging.info("Usando pipeline de transformación de características mejorado...")
    return transform_enhanced_features(df)

def generate_extended_windows(df: pl.DataFrame, window_size: int = 12, 
                             extended_window_size: int = 288) -> pl.DataFrame:
    """
    Genera ventanas para DRL, incluyendo contexto de basal, ejercicio, pasos, hipo_event.
    """
    from datetime import timedelta
    windows = []
    bolus_events = df.filter(pl.col("bolus") > 0)
    logging.info(f"Generando ventanas para {bolus_events.height} eventos bolus...")
    for row in bolus_events.iter_rows(named=True):
        ts = row["Timestamp"]
        subject_id = row["SubjectID"]
        # Ventana CGM regular
        cgm_window = df.filter(
            (pl.col("SubjectID") == subject_id) &
            (pl.col("Timestamp") <= ts) &
            (pl.col("Timestamp") > ts - timedelta(minutes=window_size*5))
        ).sort("Timestamp")
        cgm_values = cgm_window["value"].to_list() if cgm_window.height > 0 else []
        # Imputar ventana parcial
        if len(cgm_values) < window_size:
            padding = [float(np.mean(cgm_values)) if cgm_values else 120.0] * (window_size - len(cgm_values))
            cgm_values = padding + cgm_values
            logging.info(f"Ventana parcial generada para SubjectID {subject_id}, Timestamp {ts}")
        # Contexto de ejercicio en ventana (últimas 2h)
        exercise_window = df.filter(
            (pl.col("SubjectID") == subject_id) &
            (pl.col("Timestamp") <= ts) &
            (pl.col("Timestamp") > ts - timedelta(hours=2))
        )
        exercise_intensity = float(exercise_window["exercise_intensity"].mean() or 0.0) if "exercise_intensity" in exercise_window.columns else 0.0
        exercise_duration = float(exercise_window["exercise_duration"].sum() or 0.0) if "exercise_duration" in exercise_window.columns else 0.0
        exercise_in_window = 1.0 if exercise_window.height > 0 and exercise_intensity > 0 else 0.0
        # Pasos en ventana (2h)
        steps_in_window = float(exercise_window["steps"].sum() or 0.0) if "steps" in exercise_window.columns else 0.0
        # Hipo-evento en ventana (6h)
        hypo_window = df.filter(
            (pl.col("SubjectID") == subject_id) &
            (pl.col("Timestamp") <= ts) &
            (pl.col("Timestamp") > ts - timedelta(hours=6))
        )
        hypo_event_in_window = 1.0 if "hypo_event" in hypo_window.columns and hypo_window["hypo_event"].sum() > 0 else 0.0
        window_data = {
            "SubjectID": subject_id,
            "Timestamp": ts,
            "cgm_window": cgm_values,
            "bolus": row.get("bolus", 0.0),
            "meal_carbs": row.get("meal_carbs", 0.0),
            "effective_basal_rate": row.get("effective_basal_rate", 0.0),
            "temp_basal_active": row.get("temp_basal_active", 0.0),
            "exercise_intensity": exercise_intensity,
            "exercise_duration": exercise_duration,
            "exercise_in_window": exercise_in_window,
            "steps_in_window": steps_in_window,
            "hypo_event_in_window": hypo_event_in_window
        }
        windows.append(window_data)
    logging.info(f"Generadas {len(windows)} ventanas válidas")
    return pl.DataFrame(windows)

def compute_glucose_patterns_24h(cgm_values: List[float]) -> Dict[str, float]:
    """
    Calcula patrones de glucosa de 24 horas para análisis clínico.
    
    Args:
        cgm_values: Lista de valores de glucosa (idealmente 24h)
        
    Returns:
        Diccionario con características de patrones de glucosa
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
    hypo_episodes = np.sum(values_array < CONFIG['hypoglycemia_threshold'])
    patterns['hypo_episodes_24h'] = int(hypo_episodes)
    patterns['hypo_percentage_24h'] = float((hypo_episodes / len(values_array)) * 100)
    
    # Hyperglycemia analysis (> 180 mg/dL)
    hyper_episodes = np.sum(values_array > CONFIG['hyperglycemia_threshold'])
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
                                window_hours: float = 2.0) -> Dict[str, Union[float, int]]:
    """
    Calcula características mejoradas del contexto de comidas alrededor del tiempo del bolo.
    
    Args:
        bolus_time: Tiempo de administración del bolo
        meal_df: DataFrame con datos de comidas
        window_hours: Horas para buscar alrededor del tiempo del bolo
        
    Returns:
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
            'meal_timing_score': 0.0  # Qué tan cerca del tiempo de la comida
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
            'meal_timing_score': 0.0
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
    
    return {
        'meal_carbs': meal_carbs,
        'meal_time_diff_minutes': float(time_diff / 60.0),
        'meal_time_diff_hours': float(time_diff / 3600.0),
        'has_meal': 1.0,
        'meals_in_window': len(all_meals),
        'significant_meal': 1.0 if meal_carbs > CONFIG['significant_meal_threshold'] else 0.0,
        'total_carbs_window': total_carbs,
        'largest_meal_carbs': largest_meal,
        'meal_timing_score': timing_score
    }

def compute_clinical_risk_indicators(cgm_values: List[float], current_iob: float = 0.0) -> Dict[str, float]:
    """
    Calcula indicadores de riesgo clínico en tiempo real.
    
    Args:
        cgm_values: Valores recientes de glucosa
        current_iob: Insulina activa actual
        
    Returns:
        Diccionario con indicadores de riesgo
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
    
    # Riesgo de hipoglucemia e hiperglucemia
    hypo_risk = 1.0 if current_glucose < CONFIG['hypoglycemia_threshold'] else 0.0
    hyper_risk = 1.0 if current_glucose > CONFIG['hyperglycemia_threshold'] else 0.0
    
    # Tasa de cambio de glucosa (tendencia)
    if len(cgm_values) >= 2:
        glucose_rate = cgm_values[-1] - cgm_values[-2]  # mg/dL por 5 min
    else:
        glucose_rate = 0.0
    
    # Aceleración de glucosa (segunda derivada)
    if len(cgm_values) >= 3:
        glucose_acceleration = (cgm_values[-1] - cgm_values[-2]) - (cgm_values[-2] - cgm_values[-3])
    else:
        glucose_acceleration = 0.0
    
    # Puntuación de estabilidad (basada en variabilidad reciente)
    if len(cgm_values) >= 6:  # Últimos 30 minutos
        recent_values = cgm_values[-6:]
        stability_score = max(0.0, 1.0 - (np.std(recent_values) / 50.0))  # Normalizar por 50 mg/dL std
    else:
        stability_score = 1.0
    
    # Factor de riesgo IOB
    iob_risk = min(1.0, current_iob / 5.0)  # Normalizar por 5 unidades
    
    return {
        'current_hypo_risk': hypo_risk,
        'current_hyper_risk': hyper_risk,
        'glucose_rate_of_change': float(glucose_rate),
        'glucose_acceleration': float(glucose_acceleration),
        'stability_score': float(stability_score),
        'iob_risk_factor': float(iob_risk)
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
    Extrae features para DRL, incluyendo contexto de basal, ejercicio, pasos, hipo_event.
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
        enhanced_row = {
            **row,
            **basic_stats,
            **exercise_features,
            **hypo_features,
            **basal_features
        }
        enhanced_rows.append(enhanced_row)
    enhanced_df = pl.DataFrame(enhanced_rows)
    # Normalización y log1p para DRL
    for col in ["bolus", "meal_carbs", "effective_basal_rate", "exercise_intensity", "exercise_duration", "steps_in_window"]:
        if col in enhanced_df.columns:
            enhanced_df = enhanced_df.with_columns(
                pl.col(col).log1p().alias(f"{col}_log1p")
            )
    # Normalizar features binarios
    for col in ["exercise_in_window", "hypo_event_in_window", "temp_basal_active"]:
        if col in enhanced_df.columns:
            enhanced_df = enhanced_df.with_columns(
                pl.col(col).cast(pl.Float64)
            )
    logging.info(f"Características DRL extraídas. Forma: {enhanced_df.shape}")
    return enhanced_df

def transform_enhanced_features(df: pl.DataFrame) -> pl.DataFrame:
    """
    Apply enhanced transformations including log transforms, normalization,
    and feature expansion for the 52-dimensional observation space.
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
    
    logging.info(f"Transformaciones mejoradas completadas. Forma final: {df.shape}")
    
    return df

def simulate_glucose(cgm_values: list, bolus: float, carbs: float, basal_rate: float, exercise_intensity: float, steps: int = 72):
    """
    Simula glucosa para DRL, considerando basal y ejercicio.
    """
    if not cgm_values:
        return {
            'simulated_tir_6h': 0.0,
            'simulated_mean_6h': 120.0,
            'simulated_std_6h': 0.0,
            'simulated_hypo_6h': 0.0,
            'simulated_hyper_6h': 0.0
        }
    current_glucose = cgm_values[-1]
    simulated_values = []
    for _ in range(steps):
        carb_effect = carbs * CONFIG['carb_effect_factor']
        insulin_effect = (bolus + basal_rate * 0.083) * CONFIG['insulin_effect_factor']
        exercise_effect = exercise_intensity * 2.0  # Reducción por actividad
        glucose_change = carb_effect - insulin_effect - exercise_effect
        current_glucose = max(CONFIG['glucose_min'], min(CONFIG['glucose_max'], current_glucose + glucose_change))
        simulated_values.append(current_glucose)
    tir = sum(CONFIG['tir_lower'] <= g <= CONFIG['tir_upper'] for g in simulated_values) / len(simulated_values)
    hypo = sum(g < CONFIG['hypoglycemia_threshold'] for g in simulated_values) / len(simulated_values)
    hyper = sum(g > CONFIG['hyperglycemia_threshold'] for g in simulated_values) / len(simulated_values)
    return {
        'simulated_tir_6h': float(tir),
        'simulated_mean_6h': float(np.mean(simulated_values)),
        'simulated_std_6h': float(np.std(simulated_values)),
        'simulated_hypo_6h': float(hypo),
        'simulated_hyper_6h': float(hyper)
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
        logging.info(f"Insulina basal estimada para SubjectID {subject_id}: {basal_estimate:.2f} U/h")
    
    return float(basal_estimate)

def plot_clinical_metrics(df: pl.DataFrame, output_dir: str):
    """
    Genera y guarda gráficos de métricas clínicas.
    
    Args:
        df: DataFrame con datos procesados
        output_dir: Directorio para guardar los gráficos
    """
    if not CONFIG['log_clinical_metrics']:
        return
    
    # Crear directorio si no existe
    Path(output_dir).mkdir(exist_ok=True, parents=True)
    
    # Distribución de TIR
    plt.figure(figsize=(10, 6))
    plt.hist(df['time_in_range_24h'], bins=20)
    plt.xlabel('Tiempo en Rango (24h)')
    plt.ylabel('Frecuencia')
    plt.title('Distribución de TIR')
    plt.savefig(f"{output_dir}/tir_distribution.png")
    plt.close()
    
    # TIR por sujeto
    plt.figure(figsize=(12, 6))
    tir_by_subject = df.groupby('SubjectID').agg(pl.col('time_in_range_24h').mean())
    plt.bar(tir_by_subject['SubjectID'], tir_by_subject['time_in_range_24h'])
    plt.xlabel('SubjectID')
    plt.ylabel('TIR Promedio')
    plt.title('TIR por Sujeto')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/tir_by_subject.png")
    plt.close()
    
    # Distribución de bolus
    plt.figure(figsize=(10, 6))
    plt.hist(df.filter(pl.col('bolus').is_not_null())['bolus'], bins=30)
    plt.xlabel('Dosis de Bolus (U)')
    plt.ylabel('Frecuencia')
    plt.title('Distribución de Dosis de Bolus')
    plt.savefig(f"{output_dir}/bolus_distribution.png")
    plt.close()
    
    logging.info(f"Gráficos de métricas clínicas guardados en {output_dir}")

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
            logging.info(f"Fold {fold+1}: Train={len(train_data)} filas, Test={len(test_data)} filas")
        
        # Guardar split de prueba
        test_data.write_parquet(f"{output_dir}/synthetic_test_fold_{fold+1}.parquet")
    
    logging.info(f"Splits de validación cruzada guardados en {output_dir}")

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
    
    Path(args.output_dir).mkdir(exist_ok=True)
    Path(args.plots_dir).mkdir(exist_ok=True, parents=True)
    
    logging.info("=" * 60)
    logging.info("PIPELINE DE PREPROCESAMIENTO MEJORADO OHIO T1DM")
    logging.info("=" * 60)
    logging.info(f"Características mejoradas: {args.enhanced}")
    logging.info(f"Umbrales clínicos: Hipoglucemia<{CONFIG['hypoglycemia_threshold']}, Hiperglucemia>{CONFIG['hyperglycemia_threshold']}")
    logging.info(f"Tiempo en Rango: {CONFIG['tir_lower']}-{CONFIG['tir_upper']} mg/dL")
    
    for data_dir in args.data_dirs:
        logging.info(f"\nProcesando directorio: {data_dir}")
        logging.info("=" * 50)
        
        try:
            logging.info("Cargando datos crudos...")
            data = load_data(data_dir)
            if not data:
                logging.warning(f"No se encontraron datos en {data_dir}")
                continue
                
            logging.info("Preprocesando y alineando eventos...")
            processed = preprocess_bolus_meal(data)
            
            # Preprocesar glucose_level una sola vez
            data['glucose_level'] = preprocess_cgm(data['glucose_level'])
            
            # Alinear eventos con CGM
            bolus_aligned = align_events_to_cgm(data['glucose_level'], processed["bolus"])
            meal_aligned = align_events_to_cgm(data['glucose_level'], processed["meal"])
            
            # Actualizar datos con eventos alineados
            data['bolus'] = bolus_aligned
            data['meal'] = meal_aligned

            logging.info(f"Alineando y uniendo señales para {data_dir}")            
            for key in ["glucose_level", "bolus", "meal"]:
                if key in data and "Timestamp" in data[key].columns:
                    data[key] = ensure_timestamp_datetime(data[key], "Timestamp")
                    
            df = join_signals(data)
            logging.info(f"Forma de datos unidos: {df.shape}")

            if "value" in df.columns:
                df = df.with_columns(pl.col("value").cast(pl.Float64))
            if "bolus" in df.columns:
                df = df.with_columns(pl.col("bolus").cast(pl.Float64))
                    
            n_bolus_total = df.filter(pl.col("bolus") > 0).height
            logging.info(f"Total de eventos bolus: {n_bolus_total}")

            logging.info("Generando ventanas mejoradas con patrones a largo plazo...")
            if args.enhanced:
                df_windows = generate_extended_windows(
                    df, 
                    window_size=CONFIG["window_steps"],
                    extended_window_size=CONFIG["extended_window_size"]
                )
            else:
                df_windows = generate_windows(df, window_size=CONFIG["window_size"])
            
            logging.info(f"Ventanas generadas: {df_windows.shape}")

            logging.info("Extrayendo características clínicas mejoradas...")
            if args.enhanced:
                df_features = extract_enhanced_features(
                    df_windows, 
                    meal_df=data.get('meal'),
                    extended_cgm_df=df
                )
            else:
                df_features = extract_features(df_windows, data.get('meal'))
            
            logging.info(f"Características extraídas: {df_features.shape}")
            
            if args.enhanced:
                new_features = [col for col in df_features.columns if any(keyword in col for keyword in 
                    ['24h', '_sin', '_cos', 'hypo_risk', 'hyper_risk', 'stability', 'timing_score', 'composite'])]
                logging.info(f"Características mejoradas añadidas: {len(new_features)} nuevas características clínicas")
                logging.info(f"Muestra de nuevas características: {new_features[:10]}")

            logging.info("Aplicando transformaciones mejoradas...")
            if args.enhanced:
                df_final = transform_enhanced_features(df_features)
            else:
                df_final = transform_features(df_features)
                
            logging.info(f"Datos procesados finales: {df_final.shape}")
            
            if args.enhanced:
                expected_cgm_cols = [f"cgm_{i}" for i in range(CONFIG["window_steps"])]
                cgm_cols_present = [col for col in df_final.columns if col in expected_cgm_cols]
                logging.info(f"Columnas CGM: {len(cgm_cols_present)}/{len(expected_cgm_cols)}")
                
                enhanced_feature_categories = {
                    'tiempo_cíclico': ['hour_sin', 'hour_cos', 'day_sin', 'day_cos'],
                    'patrones_glucosa': ['cgm_mean_24h', 'time_in_range_24h', 'hypo_percentage_24h'],
                    'contexto_comida': ['meal_timing_score', 'significant_meal', 'total_carbs_window'],
                    'indicadores_riesgo': ['current_hypo_risk', 'current_hyper_risk', 'stability_score'],
                    'métricas_clínicas': ['composite_hypo_risk', 'composite_hyper_risk', 'iob_risk_factor']
                }
                
                for category, features in enhanced_feature_categories.items():
                    present = [f for f in features if f in df_final.columns]
                    logging.info(f"{category}: {len(present)}/{len(features)} características presentes")
            
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
            logging.info(f"Datos mejorados exportados a {output_path}")
            
            if 'bolus' in df_final.columns:
                bolus_stats = df_final.select(pl.col('bolus')).describe()
                logging.info(f"Estadísticas de bolus:\n{bolus_stats}")
            
            if args.enhanced and 'time_in_range_24h' in df_final.columns:
                tir_mean = df_final.select(pl.col('time_in_range_24h')).mean().item()
                logging.info(f"Tiempo en Rango promedio: {tir_mean:.1f}%")

        except Exception as e:
            logging.error(f"Error procesando {data_dir}: {e}")
            import traceback
            logging.error(traceback.format_exc())
            continue

    logging.info("=" * 60)
    logging.info("PREPROCESAMIENTO MEJORADO COMPLETADO")
    logging.info("=" * 60)

if __name__ == "__main__":
    main()
