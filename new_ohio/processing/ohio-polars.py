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

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Global configuration
CONFIG = {
    "window_size_hours": 2,
    "window_steps": 24,  # 5-min steps in 2 hours
    "insulin_lifetime_hours": 4.0,  # Default insulin lifetime
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
    # Nuevos parámetros para work, sleep y activity
    "max_work_intensity": 10,
    "max_sleep_quality": 10,
    "max_activity_intensity": 10
}

def load_data(data_dir: str) -> Dict[str, pl.DataFrame]:
    """
    Carga los datos y muestra las columnas de cada DataFrame.
    """
    logging.info(f"Loading data from {data_dir}")
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
                    # logging.info(f"Data type '{data_type}' columns: {df.columns}")
        except Exception as e:
            logging.error(f"Error processing {xml_file}: {e}")
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

import numpy as np

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
    Extrae features simples de las ventanas CGM y agrega información de meal si está disponible,
    incluyendo columnas requeridas por el pipeline legacy.
    """

    # Últimos valores estadísticos de la ventana CGM
    df = df.with_columns([
        pl.col("cgm_window").list.get(-1).alias("glucose_last"),
        pl.col("cgm_window").list.mean().alias("glucose_mean"),
        pl.col("cgm_window").list.std().alias("glucose_std"),
        pl.col("cgm_window").list.min().alias("glucose_min"),
        pl.col("cgm_window").list.max().alias("glucose_max"),
    ])

    # Hora del día normalizada
    df = df.with_columns([
        ((pl.col("Timestamp").dt.hour() * 60 + pl.col("Timestamp").dt.minute()) / (24 * 60)).alias("hour_of_day")
    ])

    # Última glucosa como input al modelo
    df = df.with_columns([
        pl.when(pl.col("cgm_window").list.len() > 0)
        .then(pl.col("cgm_window").list.get(-1))
        .otherwise(0.0)
        .alias("bg_input")
    ])

    # Agregar columnas legacy si no existen
    required_cols = [
        "carb_input", "basal_rate", "temp_basal_rate",
        "meal_time_diff", "meal_time_diff_hours",
        "has_meal", "meals_in_window"
    ]

    for col in required_cols:
        if col not in df.columns:
            default = 0.0 if "time" not in col and "meals" not in col else 0
            df = df.with_columns(pl.lit(default).alias(col))

    # Derivar columnas relacionadas
    if "meal_time_diff_hours" not in df.columns and "meal_time_diff" in df.columns:
        df = df.with_columns((pl.col("meal_time_diff") / 60.0).alias("meal_time_diff_hours"))

    if "has_meal" not in df.columns:
        df = df.with_columns(
            pl.when(pl.col("meal_time_diff") > 0).then(1.0).otherwise(0.0).alias("has_meal")
        )

    if "meals_in_window" not in df.columns:
        df = df.with_columns(pl.lit(0).alias("meals_in_window"))

    # Placeholder para meal_carbs si no está
    if "meal_carbs" not in df.columns and meal_df is not None:
        df = df.with_columns(pl.lit(0.0).alias("meal_carbs"))

    # Asegurar columna insulin_on_board
    if "insulin_on_board" not in df.columns:
        df = df.with_columns(pl.lit(0.0).alias("insulin_on_board"))

    if meal_df is not None and not meal_df.is_empty():
        meal_df = meal_df.with_columns(pl.col("Timestamp").cast(pl.Datetime("ns")))

        matched = []
        for row in df.iter_rows(named=True):
            bolus_time = row.get("Timestamp", None)
            if bolus_time is None:
                continue  # saltear si no hay timestamp válido

            try:
                start = bolus_time
                end = bolus_time + timedelta(hours=1)
            except Exception as e:
                logging.warning(f"Skipping row with invalid timestamp: {e}")
                continue

            meals = meal_df.filter((pl.col("Timestamp") >= start) & (pl.col("Timestamp") <= end))
            if meals.height > 0:
                meals = meals.with_columns((pl.col("Timestamp") - bolus_time).alias("time_diff")).sort("time_diff")
                closest = meals.to_dicts()[0]
                meal_time_diff = closest.get("time_diff")
                matched.append({
                    "Timestamp": bolus_time,
                    "meal_carbs": closest.get("meal_carbs", 0.0),
                    "meal_time_diff": meal_time_diff.total_seconds() / 60.0 if meal_time_diff else 0.0,
                    "meals_in_window": meals.height
                })
            else:
                matched.append({
                    "Timestamp": bolus_time,
                    "meal_carbs": 0.0,
                    "meal_time_diff": 0.0,
                    "meals_in_window": 0
                })

        meals_info = pl.DataFrame(matched)
        if "Timestamp" in meals_info.columns:
            meals_info = meals_info.with_columns(pl.col("Timestamp").cast(pl.Datetime("ns")))
            df = df.with_columns(pl.col("Timestamp").cast(pl.Datetime("ns")))
            df = df.join(meals_info, on="Timestamp", how="left")

            df = df.with_columns([
                pl.col("meal_carbs").fill_null(0.0),
                pl.col("meal_time_diff").fill_null(0.0),
                pl.col("meals_in_window").fill_null(0),
                (pl.col("meal_time_diff") / 60.0).alias("meal_time_diff_hours"),
                pl.when(pl.col("meals_in_window") > 0).then(1.0).otherwise(0.0).alias("has_meal")
            ])

    return df

def transform_features(df: pl.DataFrame) -> pl.DataFrame:
    """
    Aplica transformaciones finales a los features para entrenamiento.
    """
    # log1p a valores agregados de CGM
    for col in ["glucose_last", "glucose_mean", "glucose_std", "glucose_min", "glucose_max"]:
        if col in df.columns:
            df = df.with_columns(
                pl.col(col).log1p().alias(f"{col}_log1p")
            )

    # Tendencia y variabilidad de glucosa
    df = df.with_columns([
        pl.when(pl.col("cgm_window").list.len() > 1)
        .then(pl.col("cgm_window").list.get(-1) - pl.col("cgm_window").list.get(0))
        .otherwise(0.0)
        .alias("glucose_trend"),

        pl.col("cgm_window").list.std().fill_null(0.0).alias("glucose_variability")
    ])

    # Aliases para compatibilidad con entorno
    df = df.with_columns([
        pl.col("glucose_trend").alias("cgm_trend"),
        pl.col("glucose_variability").alias("cgm_std")
    ])

    # log1p a features relacionados con entrada del entorno
    for col in ["bolus", "carb_input", "meal_carbs", "insulin_on_board"]:
        if col in df.columns:
            df = df.with_columns(
                pl.col(col).log1p().alias(f"{col}_log1p")
            )

    # Normalización de features opcionales (work, sleep, activity)
    optional_features = ['work_intensity', 'sleep_quality', 'activity_intensity']
    for feature in optional_features:
        if feature in df.columns:
            df = df.with_columns([
                pl.when(pl.col(feature).is_not_null())
                .then(pl.col(feature) / CONFIG[f'max_{feature}'])
                .otherwise(None)
                .alias(feature)
            ])

    # Expandir cgm_window en columnas cgm_0 a cgm_23
    if "cgm_window" in df.columns:
        window_size = CONFIG["window_steps"]
        cgm_cols = [f"cgm_{i}" for i in range(window_size)]

        df = df.with_columns([
            pl.col("cgm_window").list.get(i).alias(f"cgm_{i}")
            for i in range(window_size)
        ])

        # log1p para cgm_*
        for col in cgm_cols:
            df = df.with_columns(
                pl.col(col).log1p().alias(f"{col}_log")
            )

        # Borrar cgm_window si no se usa más
        df = df.drop("cgm_window")

    return df


def main():
    """Main function to process OhioT1DM dataset."""
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Process OhioT1DM dataset')
    parser.add_argument('--data-dirs', nargs='+', 
                      default=['data/OhioT1DM/2018/train', 'data/OhioT1DM/2020/train',
                              'data/OhioT1DM/2018/test', 'data/OhioT1DM/2020/test'],
                      help='List of data directories to process')
    parser.add_argument('--output-dir', default='new_ohio/processed_data',
                      help='Output directory for processed data')
    parser.add_argument('--plots-dir', default='new_ohio/processed_data/plots',
                      help='Output directory for plots')
    parser.add_argument('--timezone', default='UTC',
                      help='Timezone to use for timestamps')
    parser.add_argument('--n-jobs', type=int, default=-1,
                      help='Number of parallel jobs (-1 for all cores)')
    args = parser.parse_args()
    
    # Update config with command line arguments
    CONFIG['timezone'] = args.timezone
    
    # Create output directories
    Path(args.output_dir).mkdir(exist_ok=True)
    Path(args.plots_dir).mkdir(exist_ok=True, parents=True)
    
    # Process each data directory
    for data_dir in args.data_dirs:
        logging.info(f"\nProcessing directory: {data_dir}")
        logging.info("=" * 50)
        
        try:
            # Load data
            data = load_data(data_dir)
            if not data:
                logging.warning(f"No data found in {data_dir}")
                continue
            processed = preprocess_bolus_meal(data)
            cgm = data["glucose_level"].with_columns(
                pl.col("ts").str.strptime(pl.Datetime, "%d-%m-%Y %H:%M:%S").alias("Timestamp")
            )
            bolus_aligned = align_events_to_cgm(cgm, processed["bolus"])
            meal_aligned = align_events_to_cgm(cgm, processed["meal"])
            data['glucose_level'] = preprocess_cgm(data['glucose_level'])

            # Reemplazá bolus y meal por los alineados
            data['bolus'] = bolus_aligned
            data['meal'] = meal_aligned

            logging.info(f"Aligning and joining signals for {data_dir}")            
            for key in ["glucose_level", "bolus", "meal"]:
                if key in data and "Timestamp" in data[key].columns:
                    data[key] = ensure_timestamp_datetime(data[key], "Timestamp")
                    
            df = join_signals(data)

            # Imprimir tipos y ejemplos antes del casteo
            if "value" in df.columns:
                df = df.with_columns(pl.col("value").cast(pl.Float64))
            if "bolus" in df.columns:
                df = df.with_columns(pl.col("bolus").cast(pl.Float64))
                    
            n_bolus_total = df.filter(pl.col("bolus") > 0).height
            logging.info(f"Total bolus events (en el join): {n_bolus_total}")

            # Ventanas alrededor de bolus
            df_windows = generate_windows(df, window_size=CONFIG["window_steps"])
            logging.info(f"Generated windows: {df_windows.shape}")

            # Features
            df_features = extract_features(df_windows, data.get('meal'))
            logging.info(f"Extracted features: {df_features.shape}")
    
            # Transformación final (opcional)
            df_final = transform_features(df_features)
            logging.info(f"Transformed features: {df_final.shape}")
            
            # Guardar en estructura train/test (¡solo este archivo!)
            is_test = 'test' in data_dir
            output_subdir = 'test' if is_test else 'train'
            # Extraer año y subcarpeta del path como nombre de archivo único
            year = Path(data_dir).parts[-2]  # "2018" o "2020"
            split = Path(data_dir).name      # "train" o "test"
            output_filename = f"processed_{year}_{split}.parquet"

            output_path = f"{args.output_dir}/{output_subdir}/{output_filename}"
            Path(f"{args.output_dir}/{output_subdir}").mkdir(exist_ok=True)
            df_final.write_parquet(output_path)
            logging.info(f"Exported processed data to {output_path}")

        except Exception as e:
            logging.error(f"Error processing {data_dir}: {e}")
            continue

if __name__ == "__main__":
    main()
