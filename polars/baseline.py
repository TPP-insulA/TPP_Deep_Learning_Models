# %%
import polars as pl
import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
import matplotlib.pyplot as plt
import os
from joblib import Parallel, delayed
from datetime import timedelta
import openpyxl

# Configuración de Matplotlib para evitar errores con Tkinter
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

# %%
# Obtener los archivos de los sujetos
# Definición de la ruta del proyecto
PROJECT_ROOT = os.path.abspath(os.path.join(os.getcwd(), ".."))
SUBJECTS_RELATIVE_PATH = "data/Subjects"
SUBJECTS_PATH = os.path.join(PROJECT_ROOT, SUBJECTS_RELATIVE_PATH)

# Crear directorio para resultados si no existe
FIGURES_DIR = os.path.join(PROJECT_ROOT, "figures", "lstm")
os.makedirs(FIGURES_DIR, exist_ok=True)
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")
os.makedirs(MODELS_DIR, exist_ok=True)

subject_files = [f for f in os.listdir(SUBJECTS_PATH) if f.startswith("Subject") and f.endswith(".xlsx")]
print(f"Total sujetos: {len(subject_files)}")

# %%
# Paso 1: Preprocesamiento.
# Leer los datos de los sujetos y preprocesarlos para el modelo, seleccionando las columnas de interés.
def get_cgm_window(bolus_time, cgm_df, window_hours=2) -> np.ndarray:
    '''
    Obtiene la ventana de datos de CGM para un tiempo de bolo específico.

    Parámetros:
    - bolus_time: datetime
        Tiempo del bolo de insulina.
    - cgm_df: pl.DataFrame
        Datos de CGM.
    - window_hours: int
        Número de horas de la ventana de datos.
    
    Retorna:
    - np.ndarray
        Ventana de datos
    '''
    window_start = bolus_time - timedelta(hours=window_hours)
    # Filtro y ordenamiento de los datos de CGM
    window = cgm_df.filter(
        (pl.col("date") >= window_start) & (pl.col("date") <= bolus_time)
    ).sort("date").tail(24)
    
    if window.height < 24:
        return None
    return window.get_column("mg/dl").to_numpy()

def calculate_iob(bolus_time, basal_df, half_life_hours=4) -> float:
    '''
    Calcula la insulina activa en el cuerpo (IOB) para un tiempo de bolo específico.

    Parámetros:
    - bolus_time: datetime
        Tiempo del bolo de insulina.
    - basal_df: pl.DataFrame
        Datos de insulina basal.
    - half_life_hours: float
        Vida media de la insulina en horas.
    
    Retorna:
    - float
        Insulina activa en el cuerpo.
    '''
    if basal_df is None or basal_df.is_empty():
        return 0.0
    
    iob = 0
    for row in basal_df.iter_rows(named=True):
        start_time = row["date"]
        duration_hours = row["duration"] / (1000 * 3600)
        end_time = start_time + timedelta(hours=duration_hours)
        rate = row["rate"] if row["rate"] is not None else 0.9
        
        if start_time <= bolus_time <= end_time:
            time_since_start = (bolus_time - start_time).total_seconds() / 3600
            remaining = rate * (1 - (time_since_start / half_life_hours))
            iob += max(0, remaining)
    return iob

def process_subject(subject_path, idx) -> list:
    '''
    Procesa los datos de un sujeto específico.

    Parámetros:
    - subject_path: str
        Ruta del archivo de datos del sujeto.
    - idx: int
        Índice del sujeto.
    
    Retorna:
    - list
        Datos procesados del sujeto.
    '''
    print(f"Procesando {os.path.basename(subject_path)} ({idx+1}/{len(subject_files)})...")
    
    try:
        # Carga de los datos del sujeto
        cgm_df = pl.read_excel(subject_path, sheet_name="CGM")
        bolus_df = pl.read_excel(subject_path, sheet_name="Bolus")
        try:
            basal_df = pl.read_excel(subject_path, sheet_name="Basal")
        except Exception:
            basal_df = None
            
    except Exception as e:
        print(f"Error loading {os.path.basename(subject_path)}: {e}")
        return []

    # Conversión de fechas y preordenamiento CGM para cada sujeto
    cgm_df = cgm_df.with_columns(pl.col("date").cast(pl.Datetime))
    bolus_df = bolus_df.with_columns(pl.col("date").cast(pl.Datetime))
    if basal_df is not None:
        basal_df = basal_df.with_columns(pl.col("date").cast(pl.Datetime))
    
    # Preordenamiento para eficiencia
    cgm_df = cgm_df.sort("date")

    processed_data = []
    for row in bolus_df.iter_rows(named=True):
        bolus_time = row["date"]
        cgm_window = get_cgm_window(bolus_time, cgm_df)
        
        if cgm_window is not None:
            iob = calculate_iob(bolus_time, basal_df)
            features = {
                'subject_id': idx,
                'cgm_window': cgm_window,
                'carbInput': row["carbInput"] if row["carbInput"] is not None else 0.0,
                'bgInput': row["bgInput"] if row["bgInput"] is not None else cgm_window[-1],
                'insulinCarbRatio': row["insulinCarbRatio"] if row["insulinCarbRatio"] is not None else 10.0,
                'insulinSensitivityFactor': 50.0,
                'insulinOnBoard': iob,
                'normal': row["normal"]
            }
            processed_data.append(features)
    
    return processed_data

# Ejecución en paralelo
all_processed_data = Parallel(n_jobs=-1)(
    delayed(process_subject)(
        os.path.join(SUBJECTS_PATH, f), 
        idx
    ) for idx, f in enumerate(subject_files)
)

all_processed_data = [item for sublist in all_processed_data for item in sublist]

# Conversión a DataFrame
df_processed = pl.DataFrame(all_processed_data)
print("Muestra de datos procesados combinados:")
print(df_processed.head())
print(f"Total muestras: {len(df_processed)}")

# %%
# División de la ventana CGM en columnas
cgm_columns = [f'cgm_{i}' for i in range(24)]
df_cgm = pl.DataFrame({
    col: [row['cgm_window'][i] for row in all_processed_data]
    for i, col in enumerate(cgm_columns)
},
schema={col: pl.Float64 for col in cgm_columns})

# Combinación con los datos procesados
df_final = pl.concat([
    df_cgm,
    df_processed.drop('cgm_window')
], how="horizontal")

# %%
# Drop rows with null values
print("Check de valores nulos en df_final:")
df_final = df_final.drop_nulls()
print(df_final.null_count())

# %%
# Paso 2: Entrenamiento del modelo.
# Normalización de los datos y división en conjuntos de entrenamiento, validación y prueba.
scaler_cgm = MinMaxScaler(feature_range=(0, 1))
scaler_other = StandardScaler()

# Normalizar características
X_cgm = scaler_cgm.fit_transform(df_final.select(cgm_columns).to_numpy())
X_other = scaler_other.fit_transform(
    df_final.select(['carbInput', 'bgInput', 'insulinOnBoard']).to_numpy()
)

# Combinar características
X = np.hstack([
    X_cgm, 
    X_other, 
    df_final.select(['insulinCarbRatio', 'insulinSensitivityFactor', 'subject_id']).to_numpy()
])
y = df_final.get_column('normal').to_numpy()

# %%
# Conversión de arrays a NumPy
X = np.asarray(X)
y = np.asarray(y)

print("X dtype:", X.dtype)
print("y dtype:", y.dtype)

def check_nan(arr) -> int:
    '''
    Función para verificar valores NaN en un array NumPy.

    Parámetros:
    - arr: np.ndarray
        Array NumPy a verificar.

    Retorna:
    - int
        Número de valores NaN en el array.
    '''
    if np.issubdtype(arr.dtype, np.number):
        return np.isnan(arr).sum()
    else:
        return np.sum([x is None for x in arr])

print("NaN in X:", check_nan(X))
print("NaN in y:", check_nan(y))

# Levantar error si hay valores en NaN
if check_nan(X) > 0 or check_nan(y) > 0:
    raise ValueError("Valores NaN detectados en X o y")

# %%
# Dividir los datos por sujeto
subject_ids = df_final.get_column('subject_id').unique().to_numpy()
train_subjects, temp_subjects = train_test_split(subject_ids, test_size=0.2, random_state=42)
val_subjects, test_subjects = train_test_split(temp_subjects, test_size=0.5, random_state=42)

train_mask = np.isin(df_final.get_column('subject_id').to_numpy(), train_subjects)
val_mask = np.isin(df_final.get_column('subject_id').to_numpy(), val_subjects)
test_mask = np.isin(df_final.get_column('subject_id').to_numpy(), test_subjects)

X_train, y_train = X[train_mask], y[train_mask]
X_val, y_val = X[val_mask], y[val_mask]
X_test, y_test = X[test_mask], y[test_mask]
subject_test = df_final.filter(pl.lit(test_mask)).get_column('subject_id').to_numpy()

print(f"Entrenamiento: {X_train.shape}, Validación: {X_val.shape}, Test: {X_test.shape}")
print(f"Sujetos de prueba: {test_subjects}")

# %%
# Paso 3: Establecimiento del modelo base.
print("\n=== Step 3: Establecimiento del modelo base ===")

# Definición del modelo FNN (Feedforward Neural Network), ajustado para un conjunto más grande de datos.
model = Sequential([
    Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
    Dropout(0.2),
    Dense(32, activation='relu'),
    Dropout(0.2),
    Dense(1, activation='linear')
])
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), loss='mse')
model.summary()

# Convertir datos to float32
X_train = np.array(X_train, dtype=np.float32)
X_val = np.array(X_val, dtype=np.float32)
y_train = np.array(y_train, dtype=np.float32)
y_val = np.array(y_val, dtype=np.float32)

# Verificar formas y tipos de datos.
DTYPE_TEXT = "dtype:"
print("X_train shape:", X_train.shape, DTYPE_TEXT, X_train.dtype)
print("X_val shape:", X_val.shape, DTYPE_TEXT, X_val.dtype)
print("y_train shape:", y_train.shape, DTYPE_TEXT, y_train.dtype)
print("y_val shape:", y_val.shape, DTYPE_TEXT, y_val.dtype)

# Entrenar el modelo.
print("\nEntrenando el modelo...")
EPOCHS = 100
BATCH_SIZE = 32
history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    callbacks=[tf.keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True)],
    verbose=1
)

# %%
# Paso 4: Graficación de los datos.
# Graficar el historial de entrenamiento.
figure_path = os.path.join(FIGURES_DIR, 'evolucion.png')

plt.figure(figsize=(10, 5))
plt.plot(history.history['loss'], label='Pérdida del Entrenamiento')
plt.plot(history.history['val_loss'], label='Pérdida de la Validación')
plt.xlabel('Épocas/Epochs')
plt.ylabel('Pérdida ECM/MSE Loss')
plt.legend()
plt.title('Historial de Entrenamiento (todos los sujetos)')
plt.savefig(figure_path, dpi=300, bbox_inches='tight')

print(f"Figura guardada en: {figure_path}")

# %%
# Paso 5: Evaluación del modelo base.
print("\n=== Paso 5: Evaluación del modelo base ===")

# Predicciones y métricas generales
y_pred = model.predict(X_test).flatten()

mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)
print(f"General MAE (Mean Absolute Error): {mae:.2f} units")
print(f"General RMSE (Root Meean Squared Error): {rmse:.2f} units")
print(f"General R²: {r2:.2f}")

# %%
# Limpieza de datos
def clean_array(arr: np.ndarray, return_counts: bool = False) -> tuple[np.ndarray, dict] | np.ndarray:
    """
    Limpia un array numpy eliminando valores infinitos y demasiado grandes para float64.
    
    Parámeteros:
    -----------
    arr : np.ndarray
        Array a limpiar
    return_counts : bool, optional
        Si es True, se devolverá un diccionario con los conteos de valores eliminados
        
    Retorna:
    --------
    np.ndarray o tuple[np.ndarray, dict]
        Array limpio o tupla con array limpio y diccionario
    """
    # Verificar dimensiones del array
    if arr.ndim != 2:
        raise ValueError(f"El array debe ser 2D. Forma actual: {arr.shape}")
    
    max_float = np.finfo(np.float64).max
    min_float = np.finfo(np.float64).min
    
    # Crear máscaras para valores problemáticos
    inf_mask = np.isinf(arr)
    too_large_mask = (arr > max_float) | (arr < min_float)
    nan_mask = np.isnan(arr)
    
    # Combinar máscaras
    problem_mask = inf_mask | too_large_mask | nan_mask
    # Convertir a máscara por filas
    row_mask = ~np.any(problem_mask, axis=1)
    
    # Contar valores problemáticos
    counts = {
        'inf_values': np.sum(inf_mask),
        'too_large_values': np.sum(too_large_mask & ~inf_mask),
        'nan_values': np.sum(nan_mask),
        'total_rows_removed': np.sum(~row_mask)
    }
    
    # Limpiar array manteniendo la forma 2D
    cleaned_arr = arr[row_mask]
    
    if return_counts:
        return cleaned_arr, counts
    return cleaned_arr

X_test_cleaned, counts = clean_array(X_test, return_counts=True)
print(f"Resumen de valores eliminados:")
for k, v in counts.items():
    print(f"{k}: {v}")

# %%
# Línea base basada en reglas.
def rule_based_prediction(arr: np.ndarray, target_bg: float = 100.0) -> np.ndarray:
    """
    Función para predecir la dosis de insulina basada en reglas.
    
    Parámetros:
    -----------
    arr : np.ndarray
        Array 2D de entrada con features
    target_bg : float
        Valor objetivo de glucosa en sangre
        
    Retorna:
    --------
    np.ndarray
        Predicciones de dosis de insulina
    """
    # Verificar dimensiones
    if arr.ndim != 2:
        raise ValueError(f"El array debe ser 2D. Forma actual: {arr.shape}")
    
    if arr.shape[1] < 29:
        raise ValueError(f"El array debe tener al menos 29 columnas. Actuales: {arr.shape[1]}")
    
    # Transformar datos una sola vez
    transformed_data = scaler_other.inverse_transform(arr[:, 24:27])
    
    # Extraer features
    carb_input = transformed_data[:, 0]
    bg_input = transformed_data[:, 1]
    icr = arr[:, 27]
    isf = arr[:, 28]
    
    # Crear máscara para valores válidos
    valid_mask = (icr != 0) & (isf != 0)
    
    # Inicializar array de predicciones con NaN
    predictions = np.full(len(arr), np.nan)
    
    # Calcular predicciones solo donde ICR e ISF son válidos
    predictions[valid_mask] = (
        carb_input[valid_mask] / icr[valid_mask] + 
        (bg_input[valid_mask] - target_bg) / isf[valid_mask]
    )
    
    # Opcionalmente, puedes manejar los casos inválidos:
    if not np.all(valid_mask):
        print(f"Advertencia: {np.sum(~valid_mask)} predicciones no pudieron calcularse debido a ICR o ISF igual a cero")
    
    return predictions

try:
    y_rule = rule_based_prediction(X_test_cleaned)
    mae_rule = mean_absolute_error(y_test, y_rule)
    rmse_rule = np.sqrt(mean_squared_error(y_test, y_rule))
    r2_rule = r2_score(y_test, y_rule)
    print(f"MAE basado en reglas: {mae_rule:.2f} u. de insulina.")
    print(f"RMSE basado en reglas: {rmse_rule:.2f} u. de insulina.")
    print(f"R² basado en reglas: {r2_rule:.2f}")
except ValueError as e:
    print(f"Error en la predicción basada en reglas: {str(e)}")
    print("\nDetalles de los datos:")
    print(f"Shape de X_test: {X_test.shape}")
    print(f"NaN en X_test: {np.isnan(X_test).sum()}")
    print(f"NaN en y_test: {np.isnan(y_test).sum()}")

# %%
# Métricas por sujeto
print("\nRendimiento por sujeto:")
for subject_id in test_subjects:
    mask = subject_test == subject_id
    y_test_sub = y_test[mask]
    y_pred_sub = y_pred[mask]
    
    # Clean data for subject-wise metrics
    sub_data = np.column_stack([y_test_sub, y_pred_sub])
    is_finite = np.all(np.isfinite(sub_data), axis=1)
    
    if len(y_test_sub) > 0 and np.any(is_finite):
        # Use only finite values for metrics
        y_test_clean = y_test_sub[is_finite]
        y_pred_clean = y_pred_sub[is_finite]
        
        # Calculate metrics
        mae_sub = mean_absolute_error(y_test_clean, y_pred_clean)
        rmse_sub = np.sqrt(mean_squared_error(y_test_clean, y_pred_clean))
        r2_sub = r2_score(y_test_clean, y_pred_clean)
        
        # Rule-based metrics if available
        try:
            y_rule_sub = y_rule[mask][is_finite]
            mae_rule_sub = mean_absolute_error(y_test_clean, y_rule_sub)
            print(
                f"Sujeto {subject_id}: "
                f"FNN MAE={mae_sub:.2f}, "
                f"RMSE={rmse_sub:.2f}, "
                f"R²={r2_sub:.2f}, "
                f"MAE basado en reglas={mae_rule_sub:.2f}"
            )
        except (IndexError, ValueError):
            print(
                f"Sujeto {subject_id}: "
                f"FNN MAE={mae_sub:.2f}, "
                f"RMSE={rmse_sub:.2f}, "
                f"R²={r2_sub:.2f}, "
                f"MAE basado en reglas=N/A"
            )
    else:
        print(f"Sujeto {subject_id}: No hay suficientes datos válidos para calcular métricas")


# %%
# Paso 6: Visualización de los resultados.
# Separación de Datos
def clean_residuals(actual, predicted):
    """
    Limpia los residuos eliminando valores infinitos.
    
    Retorna:
    --------
    - np.ndarray
        Residuos limpios
    - int
        Número de valores eliminados
    """
    residuals = actual - predicted
    mask = np.isfinite(residuals)
    n_dropped = np.sum(~mask)
    return residuals[mask], n_dropped

fnn_residuals, fnn_dropped = clean_residuals(y_test, y_pred)
rule_residuals, rule_dropped = clean_residuals(y_test, y_rule)

print(f"\nValores infinitos removidos:")
print(f"FNN: {fnn_dropped} valores")
print(f"Basado en reglas: {rule_dropped} valores")

plt.figure(figsize=(12, 5))

# %%
# Predicción vs Real
plt.subplot(1, 2, 1)
mask_pred = np.isfinite(y_pred)
mask_rule = np.isfinite(y_rule)

plt.scatter(y_test[mask_pred], y_pred[mask_pred], label='Predicción FNN', alpha=0.5)
plt.scatter(y_test[mask_rule], y_rule[mask_rule], label='Basado en Reglas', alpha=0.5)
plt.plot([0, 15], [0, 15], 'r--')
plt.xlabel('Dosis Real (u. de insulina)')
plt.ylabel('Dosis Predicha (u. de insulina)')
plt.legend()
plt.title('Predicción vs. Real (Todos los Sujetos)')

# %%
# Residuos
plt.subplot(1, 2, 2)
plt.hist(fnn_residuals, bins=20, label='Residuales FNN', alpha=0.5)
plt.hist(rule_residuals, bins=20, label='Residuos Basados en Reglas', alpha=0.5)
plt.xlabel('Residuo (u. de insulina)')
plt.ylabel('Frecuencia')
plt.legend()
plt.title('Distribución de Residuos (Todos los Sujetos)')

# %%
# Guardar figura
plt.tight_layout()
plt.savefig(os.path.join(FIGURES_DIR, 'pred_vs_real.png'), dpi=300, bbox_inches='tight')
plt.close()