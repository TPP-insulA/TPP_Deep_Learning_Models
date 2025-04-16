# %% CELL: Required Imports
import polars as pl
import pandas as pd
import os
from pathlib import Path
import numpy as np
from datetime import timedelta
import json
from config import CONFIG, WINDOW_PREV_HOURS, WINDOW_POST_HOURS, IOB_WINDOW_HOURS, PREV_SAMPLES, POST_SAMPLES

# %% CELL: Calculate IOB Function

def calculate_iob(date_target, basal_df, bolus_df, subject_id):
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

# %% CELL: Load data

all_data = []

# Iterar sobre los archivos .xlsx en el directorio
for file_path in Path(CONFIG["data_path"]).glob("*.xlsx"):
    subject_id = file_path.stem  # Nombre del archivo sin extensión
    
    # Leer las hojas con pandas como puente
    xlsx_data = pd.read_excel(file_path, sheet_name=None, engine="openpyxl")
    
    # Convertir a Polars
    cgm_df = pl.from_pandas(xlsx_data.get("CGM", pd.DataFrame())) if "CGM" in xlsx_data else pl.DataFrame()
    bolus_df = pl.from_pandas(xlsx_data.get("Bolus", pd.DataFrame())) if "Bolus" in xlsx_data else pl.DataFrame()
    basal_df = pl.from_pandas(xlsx_data.get("Basal", pd.DataFrame())) if "Basal" in xlsx_data else pl.DataFrame()
    
    # Verificar columnas mínimas necesarias
    required_cgm_cols = {"date", "mg/dl"}
    required_bolus_cols = {"date", "normal"}
    required_basal_cols = {"date", "rate", "duration"}
    
    if cgm_df.height > 0 and not required_cgm_cols.issubset(set(cgm_df.columns)):
        print(f"Advertencia: {subject_id} - CGM no tiene todas las columnas requeridas: {cgm_df.columns}")
        continue
    if bolus_df.height > 0 and not required_bolus_cols.issubset(set(bolus_df.columns)):
        print(f"Advertencia: {subject_id} - Bolus no tiene todas las columnas requeridas: {bolus_df.columns}")
        continue
    if basal_df.height > 0 and not required_basal_cols.issubset(set(basal_df.columns)):
        print(f"Advertencia: {subject_id} - Basal no tiene todas las columnas requeridas: {basal_df.columns}")
        basal_df = pl.DataFrame()  # Ignorar Basal si faltan columnas
    
    # Asegurarse de que 'date' sea datetime y 'mg/dl' sea Float64
    if cgm_df.height > 0:
        if cgm_df["date"].dtype == pl.Utf8:
            cgm_df = cgm_df.with_columns(pl.col("date").str.strptime(pl.Datetime, "%Y-%m-%d %H:%M:%S"))
        cgm_df = cgm_df.with_columns([
            pl.col("date").cast(pl.Datetime("us")),
            pl.col("mg/dl").cast(pl.Float64)  # Forzar mg/dl a Float64 desde el inicio
        ])

    if bolus_df.height > 0:
        if bolus_df["date"].dtype == pl.Utf8:
            bolus_df = bolus_df.with_columns(pl.col("date").str.strptime(pl.Datetime, "%Y-%m-%d %H:%M:%S"))
        bolus_df = bolus_df.with_columns(pl.col("date").cast(pl.Datetime("us")))
        bolus_df = bolus_df.filter((pl.col("normal").is_not_null()) & (pl.col("normal") != 0))
    if basal_df.height > 0:
        if basal_df["date"].dtype == pl.Utf8:
            basal_df = basal_df.with_columns(pl.col("date").str.strptime(pl.Datetime, "%Y-%m-%d %H:%M:%S"))
        basal_df = basal_df.with_columns(pl.col("date").cast(pl.Datetime("us")))



    if bolus_df.height == 0:
        print(f"Advertencia: {subject_id} - No hay datos válidos en Bolus después de filtrar normal (todos son None o 0)")
        continue    
    
    # Procesar cada evento de bolo
    subject_rows = []
    for bolus_row in bolus_df.rows(named=True):
        bolus_date = bolus_row["date"]
        
        # Ventana previa de CGM (2 horas antes)
        prev_start = bolus_date - timedelta(hours=WINDOW_PREV_HOURS)
        cgm_prev = cgm_df.filter(
            (pl.col("date") >= prev_start) & (pl.col("date") <= bolus_date)
        ).sort("date").tail(PREV_SAMPLES)
        
        # Rellenar con último valor si hay menos de 24 datos
        if cgm_prev.height < PREV_SAMPLES:
            last_value = cgm_prev["mg/dl"][-1] if cgm_prev.height > 0 else 100.0  # TODO: descartar dato? Por ahora valor por defecto si no hay datos
            missing_count = PREV_SAMPLES - cgm_prev.height
            filler = pl.DataFrame({
                "date": [bolus_date - timedelta(minutes=5 * i) for i in range(missing_count)],
                "mg/dl": [float(last_value)] * missing_count  # Convertir a float explícitamente
            }).with_columns([
                pl.col("date").cast(pl.Datetime("us")),
                pl.col("mg/dl").cast(pl.Float64)  # Forzar tipo Float64
            ])
            # Depuración: verificar tipos antes de concatenar
            # print(f"Subject: {subject_id}, filler dtypes: {filler.dtypes}, cgm_prev dtypes: {cgm_prev.dtypes}")
            cgm_prev = pl.concat([filler, cgm_prev]).sort("date").tail(PREV_SAMPLES)
        
        # Ventana posterior de CGM (2 horas después)
        post_end = bolus_date + timedelta(hours=WINDOW_POST_HOURS)
        cgm_post = cgm_df.filter(
            (pl.col("date") > bolus_date) & (pl.col("date") <= post_end)
        ).sort("date").head(POST_SAMPLES)
        
        # Rellenar con último valor si hay menos de 24 datos (para evaluación)
        if cgm_post.height < POST_SAMPLES:
            last_value = cgm_post["mg/dl"][-1] if cgm_post.height > 0 else cgm_prev["mg/dl"][-1]
            missing_count = POST_SAMPLES - cgm_post.height
            filler = pl.DataFrame({
                "date": [bolus_date + timedelta(minutes=5 * (i + 1)) for i in range(missing_count)],
                "mg/dl": [float(last_value)] * missing_count  # Convertir a float explícitamente
            }).with_columns([
                pl.col("date").cast(pl.Datetime("us")),
                pl.col("mg/dl").cast(pl.Float64)  # Forzar tipo Float64
            ])
            cgm_post = pl.concat([cgm_post, filler]).sort("date").head(POST_SAMPLES)
        
        # Calcular insulinOnBoard con chequeo adicional
        iob = calculate_iob(bolus_date, basal_df, bolus_df, subject_id) if basal_df.height > 0 or bolus_df.height > 0 else 0.0
        
        # Asignar targetBloodGlucose (aleatorio si falta)
        target_bg = bolus_row.get("targetBloodGlucose", None)
        if target_bg is None or target_bg < 70.0 or target_bg > 160.0:
            target_bg = np.random.choice([110.0, 130.0, 160.0])  # Valor por defecto si no hay datos o fuera de rango
        
        # Manejar valores nulos en Bolus con valores por defecto
        carb_input = bolus_row.get("carbInput", None)  # 0 si falta
        if carb_input is None:
            carb_input = 0.0
        insulin_carb_ratio = bolus_row.get("insulinCarbRatio", None)  # Valor típico si falta
        if insulin_carb_ratio is None:
            #Obtener algun valor no nulo de la columna
            insulin_carb_ratio = bolus_df.filter(pl.col("insulinCarbRatio").is_not_null() & (pl.col("insulinCarbRatio") != 0)).select("insulinCarbRatio").to_numpy().flatten()[0] if bolus_df.height > 0 else 10.0  # Valor por defecto si no hay datos
        bg_input = bolus_row.get("bgInput", None)  # Última glucosa o 100
        if bg_input is None:
            bg_input = cgm_prev["mg/dl"][-1] if cgm_prev.height > 0 else np.random.choice([60.0, 80.0, 110.0, 130.0, 160.0, 190.0, 220.0, 250.0])  # Valor por defecto si no hay datos

        # Construir fila
        try: 
            row = {
            "subject_id": subject_id,
            "bolus_date": bolus_date,
            **{f"mg/dl_prev_{i+1}": cgm_prev["mg/dl"][i] for i in range(PREV_SAMPLES)},
            "carbInput": float(carb_input),
            "insulinCarbRatio": float(insulin_carb_ratio),
            "bgInput": float(bg_input),
            "insulinOnBoard": float(iob),
            "targetBloodGlucose": float(target_bg),
            "normal": bolus_row["normal"],
            **{f"mg/dl_post_{i+1}": cgm_post["mg/dl"][i] for i in range(POST_SAMPLES)},
        }
        except:
            print(f"Error with values: {subject_id}, insulin_carb_ratio: {insulin_carb_ratio}, bg_input: {bg_input}, iob: {iob}, carb_input: {carb_input}, target_bg: {target_bg}")
            continue
        subject_rows.append(row)
    
    # Convertir a DataFrame de Polars
    if subject_rows:
        subject_df = pl.DataFrame(subject_rows)
        all_data.append(subject_df)

# %% CELL: Save dataset from every subject
for subject_df in all_data:
    subject_id = subject_df["subject_id"][0]
    output_path = os.path.join(CONFIG["processed_data_path"], f"{subject_id}_processed.parquet")
    subject_df.write_parquet(output_path)
    print(f"Guardado: {output_path}")

# %% CELL: Split datasets
train_ratio = 0.7  # Porcentaje de datos para entrenamiento
val_ratio = 0.15  # Porcentaje de datos para validación
size_ratio = 1 - train_ratio - val_ratio  # Porcentaje de datos para prueba

train_dfs = []
val_dfs = []
test_dfs = []


for subject_df in all_data:
    
    subject_df = subject_df.sort("bolus_date")
    total_rows = subject_df.height
    train_end = int(total_rows * train_ratio)
    val_end = int(total_rows * (train_ratio + val_ratio))

    train_df = subject_df.slice(0, train_end)
    val_df = subject_df.slice(train_end, val_end - train_end)
    test_df = subject_df.slice(val_end, total_rows - val_end)

    train_dfs.append(train_df)
    val_dfs.append(val_df)
    test_dfs.append(test_df)

#for i, df in enumerate(train_dfs):
#        print(f"Train DF {i} (sujeto {train_dfs[i]['subject_id'][0]}) tipos: {dict(zip(df.columns, df.dtypes))}")

train_combined = pl.concat(train_dfs)
val_combined = pl.concat(val_dfs)
test_combined = pl.concat(test_dfs)

print("Valores únicos de targetBloodGlucose en train_combined:", train_combined["targetBloodGlucose"].unique().to_list())
print("Estadísticas de targetBloodGlucose en train_combined:", train_combined["targetBloodGlucose"].describe())


# %% CELL: Normalize data

# Columnas del estado a estandarizar
state_cols = [
    *[f"mg/dl_prev_{i+1}" for i in range(PREV_SAMPLES)],
    "carbInput", "insulinCarbRatio", "bgInput", "insulinOnBoard", "targetBloodGlucose"
]
action_col = ["normal"]  # Acción, no se estandariza
post_cols = [f"mg/dl_post_{i+1}" for i in range(POST_SAMPLES)]  # Para reward, no se estandariza

means = train_combined.select([pl.col(col).mean() for col in state_cols]).to_dicts()[0]
stds = train_combined.select([pl.col(col).std() for col in state_cols]).to_dicts()[0]

# Save params for standardization
with open(os.path.join(CONFIG["params_path"], "state_standardization_params.json"), "w") as f:
    json.dump({"means": means, "stds": stds}, f)

# %% CELL: Normalize train, val and test data
def standardize_state(df, means, stds):
    for col in state_cols:
        df = df.with_columns(
            ((pl.col(col) - means[col]) / stds[col]).alias(col)
        )
    return df

train_standardized = standardize_state(train_combined, means, stds)
val_standardized = standardize_state(val_combined, means, stds)
test_standardized = standardize_state(test_combined, means, stds)

train_standardized.write_parquet(os.path.join(CONFIG["processed_data_path"], "train_all.parquet"))
val_standardized.write_parquet(os.path.join(CONFIG["processed_data_path"], "val_all.parquet"))
test_standardized.write_parquet(os.path.join(CONFIG["processed_data_path"], "test_all.parquet"))

# %% CELL: Data about train, test and val
print("Conjuntos preparados para PPO (estado estandarizado).")
print(f"Train: {train_standardized.height} filas, Val: {val_standardized.height} filas, Test: {test_standardized.height} filas")
# %%
