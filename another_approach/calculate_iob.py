from config import IOB_WINDOW_HOURS
from datetime import timedelta
import numpy as np
import polars as pl

def calculate_iob(iob_id, date_target, basal_df, bolus_df, subject_id):
    if iob_id == 0:
        return calculate_iob_0(date_target, basal_df, bolus_df, subject_id)
    else:
        raise ValueError(f"ID de IOB no soportado: {iob_id}. Solo se admite el ID 0.")
    

def calculate_iob_0(date_target, basal_df, bolus_df, subject_id):
    window_start = date_target - timedelta(hours=IOB_WINDOW_HOURS)
    iob = 0.0
    
    # Verificar si basal_df tiene las columnas necesarias
    if basal_df.height > 0 and "date" in basal_df.columns and "rate" in basal_df.columns and "duration" in basal_df.columns:
        basal_window = basal_df.filter(
            (pl.col("date") >= window_start) & (pl.col("date") <= date_target)
        )
        # Calcular el promedio de rates no nulos
        valid_rates = [row["rate"] for row in basal_window.rows(named=True) if row["rate"] is not None]
        default_rate = np.mean(valid_rates) if valid_rates else 0.0  # Promedio o 0 si no hay valores válidos
        for row in basal_window.rows(named=True):
            time_elapsed = (date_target - row["date"]).total_seconds() / 3600  # En horas
            duration_hr = row["duration"] / 3_600_000  # ms a horas
            rate = row["rate"] if row["rate"] is not None else default_rate  # Usar promedio si rate es None
            insulin_delivered = rate * duration_hr
            fraction_remaining = max(0, 1 - (time_elapsed / IOB_WINDOW_HOURS))  # Curva lineal
            iob += insulin_delivered * fraction_remaining
    elif basal_df.height > 0:
        print(f"Advertencia: {subject_id} - Basal no tiene las columnas esperadas: {basal_df.columns}")
    
    # Verificar si bolus_df tiene las columnas necesarias
    if bolus_df.height > 0 and "date" in bolus_df.columns and "normal" in bolus_df.columns:
        bolus_window = bolus_df.filter(
            (pl.col("date") >= window_start) & (pl.col("date") <= date_target)
        )
        for row in bolus_window.rows(named=True):
            time_elapsed = (date_target - row["date"]).total_seconds() / 3600  # En horas
            fraction_remaining = max(0, 1 - (time_elapsed / IOB_WINDOW_HOURS))  # Curva lineal
            iob += row["normal"] * fraction_remaining
    elif bolus_df.height > 0:
        print(f"Advertencia: {subject_id} - Bolus no tiene las columnas esperadas: {bolus_df.columns}")
    
    return iob