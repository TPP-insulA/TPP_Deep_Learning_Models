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

# Configuración global mejorada
CONFIG = {
    "window_size_hours": 2,
    "window_steps": 24,  # Pasos de 5 min en 24 horas
    "extended_window_hours": 24,  # Para patrones a largo plazo
    "extended_window_steps": 288,  # Pasos de 5 min en 24 horas
    "insulin_lifetime_hours": 4.0,  # Duración predeterminada de la insulina
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
    # Umbrales clínicos
    "hypoglycemia_threshold": 70,  # mg/dL
    "hyperglycemia_threshold": 180,  # mg/dL
    "severe_hyperglycemia_threshold": 250,  # mg/dL
    "tir_lower": 70,  # Límite inferior de Tiempo en Rango
    "tir_upper": 180,  # Límite superior de Tiempo en Rango
    # Indicadores de riesgo
    "hypo_risk_threshold": 80,  # Glucosa actual para riesgo de hipoglucemia
    "hyper_risk_threshold": 200,  # Glucosa actual para riesgo de hiperglucemia
    # Contexto de comidas
    "significant_meal_threshold": 15,  # gramos de carbohidratos
    "meal_window_hours": 2,  # Horas para buscar comidas alrededor del bolo
    # Parámetros mejorados
    "max_work_intensity": 10,
    "max_sleep_quality": 10,
    "max_activity_intensity": 10
}

def load_data(data_dir: str) -> Dict[str, pl.DataFrame]:
    """
    Carga los datos y muestra las columnas de cada DataFrame.
    """
    logging.info(f"Cargando datos desde {data_dir}")
    data_dict = {}
    for xml_file in glob.glob(os.path.join(data_dir, "*.xml")):
        try:
            tree = ET.parse(xml_file)
            root = tree.getroot()
            subject_id = os.path.basename(xml_file).split('.')[0]
            for data_type_elem in root:
                data_type = data_type_elem.tag
                if data_type == 'patient':
                    continue
                records = []
                for event in data_type_elem:
                    record_dict = dict(event.attrib)
                    record_dict['SubjectID'] = subject_id
                    records.append(record_dict)
                if records:
                    df = pl.DataFrame(records)
                    data_dict[data_type] = df
        except Exception as e:
            logging.error(f"Error procesando {xml_file}: {e}")
            continue
    return data_dict

def preprocess_bolus_meal(data: Dict[str, pl.DataFrame]) -> Dict[str, pl.DataFrame]:
    """
    Renombra y convierte columnas clave de bolus y meal para facilitar el join.
    """
    processed = {}
    # Procesar bolus
    if "bolus" in data:
        bolus = data["bolus"].clone()
        if "dose" in bolus.columns:
            bolus = bolus.rename({"dose": "bolus"})
        if "ts_begin" in bolus.columns:
            bolus = bolus.with_columns(
                pl.col("ts_begin").str.strptime(pl.Datetime, "%d-%m-%Y %H:%M:%S").alias("Timestamp")
            )
        processed["bolus"] = bolus
    # Procesar meal
    if "meal" in data:
        meal = data["meal"].clone()
        if "carbs" in meal.columns:
            meal = meal.rename({"carbs": "meal_carbs"})
        if "ts" in meal.columns:
            meal = meal.with_columns(
                pl.col("ts").str.strptime(pl.Datetime, "%d-%m-%Y %H:%M:%S").alias("Timestamp")
            )
        processed["meal"] = meal
    return processed

def align_events_to_cgm(cgm_df: pl.DataFrame, event_df: pl.DataFrame, event_time_col: str = "Timestamp", tolerance_minutes: int = 5) -> pl.DataFrame:
    """
    Asigna cada evento (bolus, meal, etc.) al timestamp de CGM más cercano dentro de una tolerancia.
    Devuelve un DataFrame de eventos con el timestamp alineado.
    """
    if cgm_df.is_empty() or event_df.is_empty():
        return event_df

    cgm_times = cgm_df["Timestamp"].to_numpy()
    aligned_rows = []
    for row in event_df.iter_rows(named=True):
        event_time = row[event_time_col]
        # Asegura que event_time sea numpy.datetime64
        if not isinstance(event_time, np.datetime64):
            try:
                event_time = np.datetime64(event_time)
            except Exception:
                continue
        idx = np.argmin(np.abs(cgm_times - event_time))
        nearest_cgm_time = cgm_times[idx]
        # Chequear tolerancia
        diff_minutes = np.abs((nearest_cgm_time - event_time) / np.timedelta64(1, 'm'))
        if diff_minutes <= tolerance_minutes:
            row["Timestamp"] = nearest_cgm_time
            aligned_rows.append(row)
    return pl.DataFrame(aligned_rows)

def preprocess_cgm(cgm: pl.DataFrame) -> pl.DataFrame:
    if "ts" in cgm.columns:
        cgm = cgm.with_columns(
            pl.col("ts").str.strptime(pl.Datetime, "%d-%m-%Y %H:%M:%S").alias("Timestamp")
        )
    return cgm

def join_signals(data: Dict[str, pl.DataFrame]) -> pl.DataFrame:
    """
    Une CGM, bolus y meal alineados por Timestamp y SubjectID.
    """
    df = data["glucose_level"].clone()
    # Join bolus
    if "bolus" in data and not data["bolus"].is_empty():
        df = df.join(
            data["bolus"].select(["Timestamp", "bolus", "SubjectID"]),
            on=["Timestamp", "SubjectID"],
            how="left"
        )
    else:
        df = df.with_columns(pl.lit(None).alias("bolus"))
    # Join meal
    if "meal" in data and not data["meal"].is_empty():
        df = df.join(
            data["meal"].select(["Timestamp", "meal_carbs", "SubjectID"]),
            on=["Timestamp", "SubjectID"],
            how="left"
        )
    else:
        df = df.with_columns(pl.lit(None).alias("meal_carbs"))
        
    # Join opcional: carb_input (de bolus)
    if "bolus" in data and "bwz_carb_input" in data["bolus"].columns:
        df = df.join(
            data["bolus"].select(["Timestamp", "SubjectID", "bwz_carb_input"]).rename({"bwz_carb_input": "carb_input"}),
            on=["Timestamp", "SubjectID"],
            how="left"
        )
    else:
        df = df.with_columns(pl.lit(0.0).alias("carb_input"))

    # Join basal
    if "basal" in data and "dose" in data["basal"].columns:
        basal = data["basal"].with_columns(
            pl.col("dose").cast(pl.Float64)
        ).rename({"dose": "basal_rate"})
        if "ts_begin" in basal.columns:
            basal = basal.with_columns(pl.col("ts_begin").str.strptime(pl.Datetime, "%d-%m-%Y %H:%M:%S").alias("Timestamp"))
        df = df.join(
            basal.select(["Timestamp", "SubjectID", "basal_rate"]),
            on=["Timestamp", "SubjectID"],
            how="left"
        )
    else:
        df = df.with_columns(pl.lit(0.0).alias("basal_rate"))

    # Join temp_basal
    if "temp_basal" in data and "dose" in data["temp_basal"].columns:
        temp = data["temp_basal"].with_columns(
            pl.col("dose").cast(pl.Float64)
        ).rename({"dose": "temp_basal_rate"})
        if "ts_begin" in temp.columns:
            temp = temp.with_columns(pl.col("ts_begin").str.strptime(pl.Datetime, "%d-%m-%Y %H:%M:%S").alias("Timestamp"))
        df = df.join(
            temp.select(["Timestamp", "SubjectID", "temp_basal_rate"]),
            on=["Timestamp", "SubjectID"],
            how="left"
        )
    else:
        df = df.with_columns(pl.lit(0.0).alias("temp_basal_rate"))

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
    Genera ventanas de CGM de tamaño fijo antes de cada evento bolus.
    window_size: cantidad de pasos (por ejemplo, 12 para 1 hora si los datos son cada 5 min)
    """
    from datetime import timedelta

    windows = []
    # Filtrar solo eventos bolus válidos
    bolus_events = df.filter(pl.col("bolus") > 0)
    for row in bolus_events.iter_rows(named=True):
        ts = row["Timestamp"]
        subject_id = row["SubjectID"]
        # Filtrar CGM del mismo sujeto y ventana temporal
        cgm_window = df.filter(
            (pl.col("SubjectID") == subject_id) &
            (pl.col("Timestamp") <= ts) &
            (pl.col("Timestamp") > ts - timedelta(minutes=window_size*5))
        ).sort("Timestamp")
        # Si hay suficientes puntos, guardar la ventana
        if cgm_window.height == window_size:
            windows.append({
                "SubjectID": subject_id,
                "Timestamp": ts,
                "cgm_window": cgm_window["value"].to_list(),
                "bolus": row["bolus"],
                "meal_carbs": row.get("meal_carbs", 0.0)
            })
    return pl.DataFrame(windows)
 
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
    Genera tanto ventanas regulares como extendidas para análisis de patrones a largo plazo.
    """
    from datetime import timedelta

    windows = []
    # Filtrar solo eventos bolus válidos
    bolus_events = df.filter(pl.col("bolus") > 0)
    
    logging.info(f"Generando ventanas para {bolus_events.height} eventos bolus...")
    
    for row in bolus_events.iter_rows(named=True):
        ts = row["Timestamp"]
        subject_id = row["SubjectID"]
        
        # Ventana CGM regular (2 horas antes del bolus)
        cgm_window = df.filter(
            (pl.col("SubjectID") == subject_id) &
            (pl.col("Timestamp") <= ts) &
            (pl.col("Timestamp") > ts - timedelta(minutes=window_size*5))
        ).sort("Timestamp")
        
        # Ventana CGM extendida (24 horas antes del bolus) para patrones a largo plazo
        extended_cgm_window = df.filter(
            (pl.col("SubjectID") == subject_id) &
            (pl.col("Timestamp") <= ts) &
            (pl.col("Timestamp") > ts - timedelta(hours=24))
        ).sort("Timestamp")
        
        # Asegurar que tenemos suficientes datos para la ventana regular
        if cgm_window.height >= window_size:
            # Tomar los últimos window_size puntos para consistencia
            cgm_values = cgm_window["value"].tail(window_size).to_list()
            
            # Obtener valores extendidos si están disponibles (para patrones de 24h)
            extended_cgm_values = extended_cgm_window["value"].to_list() if extended_cgm_window.height > 0 else cgm_values
            
            window_data = {
                "SubjectID": subject_id,
                "Timestamp": ts,
                "cgm_window": cgm_values,
                "extended_cgm_window": extended_cgm_values,
                "bolus": row["bolus"],
                "meal_carbs": row.get("meal_carbs", 0.0),
                "carb_input": row.get("carb_input", 0.0),
                "basal_rate": row.get("basal_rate", 0.0),
                "temp_basal_rate": row.get("temp_basal_rate", 0.0)
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
    hypo_risk = 1.0 if current_glucose < CONFIG['hypo_risk_threshold'] else 0.0
    hyper_risk = 1.0 if current_glucose > CONFIG['hyper_risk_threshold'] else 0.0
    
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
    Extract enhanced features including long-term patterns, cyclical time encoding,
    enhanced meal context, and clinical risk indicators.
    """
    logging.info("Extrayendo características mejoradas con patrones clínicos...")
    
    enhanced_rows = []
    
    for row in df.iter_rows(named=True):
        bolus_time = row["Timestamp"]
        subject_id = row["SubjectID"]
        cgm_window = row.get("cgm_window", [])
        
        # Basic CGM statistics (existing)
        basic_stats = {
            "glucose_last": cgm_window[-1] if cgm_window else 120.0,
            "glucose_mean": float(np.mean(cgm_window)) if cgm_window else 120.0,
            "glucose_std": float(np.std(cgm_window)) if cgm_window else 0.0,
            "glucose_min": float(np.min(cgm_window)) if cgm_window else 120.0,
            "glucose_max": float(np.max(cgm_window)) if cgm_window else 120.0,
        }
        
        # Enhanced Feature 1: Long-term glucose patterns (24h)
        glucose_patterns = compute_glucose_patterns_24h(cgm_window)
        
        # Enhanced Feature 2: Cyclical time encoding
        time_features = encode_time_cyclical(bolus_time)
        
        # Enhanced Feature 3: Enhanced meal context
        if meal_df is not None and not meal_df.is_empty():
            subject_meals = meal_df.filter(pl.col("SubjectID") == subject_id)
            meal_context = compute_enhanced_meal_context(bolus_time, subject_meals)
        else:
            meal_context = compute_enhanced_meal_context(bolus_time, pl.DataFrame())
        
        # Enhanced Feature 4: Clinical risk indicators
        subject_data = extended_cgm_df.filter(pl.col("SubjectID") == subject_id) if extended_cgm_df is not None else pl.DataFrame()
        current_iob = calculate_insulin_on_board(subject_data, bolus_time) if not subject_data.is_empty() else 0.0
        risk_indicators = compute_clinical_risk_indicators(cgm_window, current_iob)
        
        # Enhanced Feature 5: Advanced glucose analysis
        advanced_glucose = {}
        if cgm_window and len(cgm_window) > 1:
            recent_trend = np.polyfit(range(len(cgm_window)), cgm_window, 1)[0]
            advanced_glucose['glucose_trend'] = float(recent_trend)
            advanced_glucose['cgm_trend'] = float(recent_trend)
            
            advanced_glucose['glucose_variability'] = float(np.std(cgm_window))
            advanced_glucose['cgm_std'] = float(np.std(cgm_window))
        else:
            advanced_glucose['glucose_trend'] = 0.0
            advanced_glucose['cgm_trend'] = 0.0
            advanced_glucose['glucose_variability'] = 0.0
            advanced_glucose['cgm_std'] = 0.0
        
        enhanced_row = {
            **row,
            **basic_stats,
            **glucose_patterns,
            **time_features,
            **meal_context,
            **risk_indicators,
            **advanced_glucose,
            'insulin_on_board': current_iob,
            'bg_input': basic_stats["glucose_last"]
        }
        
        enhanced_rows.append(enhanced_row)
    
    enhanced_df = pl.DataFrame(enhanced_rows)
    
    required_legacy_cols = [
        "carb_input", "basal_rate", "temp_basal_rate"
    ]
    
    for col in required_legacy_cols:
        if col not in enhanced_df.columns:
            enhanced_df = enhanced_df.with_columns(pl.lit(0.0).alias(col))
    
    logging.info(f"Características mejoradas extraídas. Forma: {enhanced_df.shape}")
    logging.info(f"Nuevas columnas de características añadidas: {[col for col in enhanced_df.columns if col not in df.columns]}")
    
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
            
            cgm = data["glucose_level"].with_columns(
                pl.col("ts").str.strptime(pl.Datetime, "%d-%m-%Y %H:%M:%S").alias("Timestamp")
            )
            
            bolus_aligned = align_events_to_cgm(cgm, processed["bolus"])
            meal_aligned = align_events_to_cgm(cgm, processed["meal"])
            data['glucose_level'] = preprocess_cgm(data['glucose_level'])

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
                    extended_window_size=CONFIG["extended_window_steps"]
                )
            else:
                df_windows = generate_windows(df, window_size=CONFIG["window_steps"])
            
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
