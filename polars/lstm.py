# %%
import polars as pl
import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Dropout, LSTM, Input, Concatenate, BatchNormalization
import matplotlib.pyplot as plt
import os
from joblib import Parallel, delayed
from datetime import timedelta
import openpyxl

# Configuración de Matplotlib para evitar errores con Tkinter
import matplotlib
matplotlib.use('TkAgg')

# %%
# Definición de la ruta del proyecto
PROJECT_ROOT = os.path.abspath(os.path.join(os.getcwd(), ".."))
SUBJECTS_RELATIVE_PATH = "data/Subjects"
SUBJECTS_PATH = os.path.join(PROJECT_ROOT, SUBJECTS_RELATIVE_PATH)
# Crear directorio para resultados si no existe
FIGURES_DIR = os.path.join(PROJECT_ROOT, "figures", "lstm")
os.makedirs(FIGURES_DIR, exist_ok=True)
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")
os.makedirs(MODELS_DIR, exist_ok=True)

# %%
def get_cgm_window(bolus_time: datetime, cgm_df: pl.DataFrame, window_hours: int = 2) -> np.ndarray:
    """
    Obtiene la ventana de datos CGM para un tiempo de bolo específico.

    Parámetros:
    -----------
    bolus_time : datetime
        Tiempo del bolo de insulina
    cgm_df : pl.DataFrame
        DataFrame con datos CGM
    window_hours : int, opcional
        Horas de la ventana de datos (default: 2)

    Retorna:
    --------
    np.ndarray
        Ventana de datos CGM o None si no hay suficientes datos
    """
    window_start = bolus_time - timedelta(hours=window_hours)
    window = cgm_df.filter(
        (pl.col("date") >= window_start) & (pl.col("date") <= bolus_time)
    ).sort("date").tail(24)
    
    if window.height < 24:
        return None
    return window.get_column("mg/dl").to_numpy()

def calculate_iob(bolus_time: datetime, basal_df: pl.DataFrame, half_life_hours: float = 4.0) -> float:
    """
    Calcula la insulina activa en el cuerpo (IOB).

    Parámetros:
    -----------
    bolus_time : datetime
        Tiempo del bolo de insulina
    basal_df : pl.DataFrame
        DataFrame con datos de insulina basal
    half_life_hours : float, opcional
        Vida media de la insulina en horas (default: 4.0)

    Retorna:
    --------
    float
        Cantidad de insulina activa
    """
    if basal_df is None or basal_df.is_empty():
        return 0.0
    
    iob = 0.0
    for row in basal_df.iter_rows(named=True):
        start_time = row["date"]
        duration_hours = row["duration"] / (1000 * 3600)
        end_time = start_time + timedelta(hours=duration_hours)
        rate = row["rate"] if row["rate"] is not None else 0.9
        
        if start_time <= bolus_time <= end_time:
            time_since_start = (bolus_time - start_time).total_seconds() / 3600
            remaining = rate * (1 - (time_since_start / half_life_hours))
            iob += max(0.0, remaining)
    return iob

def process_subject(subject_path: str, idx: int) -> list:
    """
    Procesa los datos de un sujeto.

    Parámetros:
    -----------
    subject_path : str
        Ruta al archivo del sujeto
    idx : int
        Índice del sujeto

    Retorna:
    --------
    list
        Lista de diccionarios con características procesadas
    """
    print(f"Procesando {os.path.basename(subject_path)} ({idx+1}/{len(SUBJECTS_PATH)})...")
    
    try:
        cgm_df = pl.read_excel(subject_path, sheet_name="CGM")
        bolus_df = pl.read_excel(subject_path, sheet_name="Bolus")
        try:
            basal_df = pl.read_excel(subject_path, sheet_name="Basal")
        except Exception:
            basal_df = None
    except Exception as e:
        print(f"Error al cargar {os.path.basename(subject_path)}: {e}")
        return []

    # Conversión de fechas
    cgm_df = cgm_df.with_columns(pl.col("date").cast(pl.Datetime))
    bolus_df = bolus_df.with_columns(pl.col("date").cast(pl.Datetime))
    if basal_df is not None:
        basal_df = basal_df.with_columns(pl.col("date").cast(pl.Datetime))
    
    cgm_df = cgm_df.sort("date")

    processed_data = []
    for row in bolus_df.iter_rows(named=True):
        bolus_time = row["date"]
        cgm_window = get_cgm_window(bolus_time, cgm_df)
        
        if cgm_window is not None:
            iob = calculate_iob(bolus_time, basal_df)
            hour_of_day = bolus_time.hour / 23.0
            bg_input = row["bgInput"] if row["bgInput"] is not None else cgm_window[-1]
            normal = row["normal"] if row["normal"] is not None else 0.0
            
            # Cálculo del factor de sensibilidad personalizado
            isf_custom = 50.0
            if normal > 0 and bg_input > 100:
                isf_custom = (bg_input - 100) / normal
            
            features = {
                'subject_id': idx,
                'cgm_window': cgm_window,
                'carbInput': row["carbInput"] if row["carbInput"] is not None else 0.0,
                'bgInput': bg_input,
                'insulinCarbRatio': row["insulinCarbRatio"] if row["insulinCarbRatio"] is not None else 10.0,
                'insulinSensitivityFactor': isf_custom,
                'insulinOnBoard': iob,
                'hour_of_day': hour_of_day,
                'normal': normal
            }
            processed_data.append(features)
    
    return processed_data

# Ejecución en paralelo
all_processed_data = Parallel(n_jobs=-1)(
    delayed(process_subject)(
        os.path.join(SUBJECTS_PATH, f), 
        idx
    ) for idx, f in enumerate(SUBJECTS_PATH)
)

all_processed_data = [item for sublist in all_processed_data for item in sublist]

# Conversión a DataFrame
df_processed = pl.DataFrame(all_processed_data)
print("Muestra de datos procesados combinados:")
print(df_processed.head())
print(f"Total muestras: {len(df_processed)}")

# %%
# División de ventana CGM y otras características
cgm_columns = [f'cgm_{i}' for i in range(24)]
df_cgm = pl.DataFrame({
    col: [row['cgm_window'][i] for row in all_processed_data]
    for i, col in enumerate(cgm_columns)
}, schema={col: pl.Float64 for col in cgm_columns})

# Combinar con otras características
df_final = pl.concat([
    df_cgm,
    df_processed.drop('cgm_window')
], how="horizontal")

# Verificar valores nulos
print("Verificación de valores nulos en df_final:")
df_final = df_final.drop_nulls()
print(df_final.null_count())

# %%
# Normalizar características
scaler_cgm = MinMaxScaler(feature_range=(0, 1))
scaler_other = StandardScaler()

# Normalizar CGM y reshape para LSTM
X_cgm = scaler_cgm.fit_transform(df_final.select(cgm_columns).to_numpy())
X_cgm = X_cgm.reshape(X_cgm.shape[0], X_cgm.shape[1], 1)

# Normalizar otras características (incluyendo hour_of_day)
other_features = ['carbInput', 'bgInput', 'insulinOnBoard', 'insulinCarbRatio', 
                 'insulinSensitivityFactor', 'subject_id', 'hour_of_day']
X_other = scaler_other.fit_transform(df_final.select(other_features).to_numpy())

# Etiquetas
y = df_final.get_column('normal').to_numpy()

# Verificar NaN
print("\nVerificación de valores NaN:")
print(f"NaN en X_cgm: {np.isnan(X_cgm).sum()}")
print(f"NaN en X_other: {np.isnan(X_other).sum()}")
print(f"NaN en y: {np.isnan(y).sum()}")

if np.isnan(X_cgm).sum() > 0 or np.isnan(X_other).sum() > 0 or np.isnan(y).sum() > 0:
    raise ValueError("Valores NaN detectados en X_cgm, X_other o y")

# %%
# División por sujeto
subject_ids = df_final.get_column('subject_id').unique().to_numpy()
train_subjects, temp_subjects = train_test_split(subject_ids, test_size=0.2, random_state=42)
val_subjects, test_subjects = train_test_split(temp_subjects, test_size=0.5, random_state=42)

train_mask = np.isin(df_final.get_column('subject_id').to_numpy(), train_subjects)
val_mask = np.isin(df_final.get_column('subject_id').to_numpy(), val_subjects)
test_mask = np.isin(df_final.get_column('subject_id').to_numpy(), test_subjects)

X_cgm_train, X_cgm_val, X_cgm_test = X_cgm[train_mask], X_cgm[val_mask], X_cgm[test_mask]
X_other_train, X_other_val, X_other_test = X_other[train_mask], X_other[val_mask], X_other[test_mask]
y_train, y_val, y_test = y[train_mask], y[val_mask], y[test_mask]
subject_test = df_final.filter(pl.lit(test_mask)).get_column('subject_id').to_numpy()

print("\nFormas de los conjuntos de datos:")
print(f"Entrenamiento CGM: {X_cgm_train.shape}, Validación CGM: {X_cgm_val.shape}, Prueba CGM: {X_cgm_test.shape}")
print(f"Entrenamiento Otros: {X_other_train.shape}, Validación Otros: {X_other_val.shape}, Prueba Otros: {X_other_test.shape}")
print(f"Sujetos de prueba: {test_subjects}")

# %%
# Modelo LSTM
def create_lstm_model(cgm_shape: tuple, other_features_shape: tuple) -> Model:
    """
    Crea un modelo LSTM con dos entradas.
    
    Parámetros:
    -----------
    cgm_shape : tuple
        Forma de los datos CGM
    other_features_shape : tuple
        Forma de otras características
        
    Retorna:
    --------
    Model
        Modelo LSTM compilado
    """
    # Definir las entradas
    cgm_input = Input(shape=cgm_shape[1:], name='cgm_input')
    other_input = Input(shape=(other_features_shape[1],), name='other_input')

    # Capas LSTM apiladas para procesar CGM
    lstm_out = LSTM(128, return_sequences=True)(cgm_input)
    lstm_out = LSTM(64, return_sequences=False)(lstm_out)
    lstm_out = BatchNormalization()(lstm_out)
    lstm_out = Dropout(0.2)(lstm_out)

    # Combinar con otras características
    combined = Concatenate()([lstm_out, other_input])

    # Capas densas
    dense = Dense(64, activation='relu')(combined)
    dense = BatchNormalization()(dense)
    dense = Dropout(0.2)(dense)
    output = Dense(1, activation='linear')(dense)

    # Crear y compilar modelo
    model = Model(inputs=[cgm_input, other_input], outputs=output)
    
    return model

# Función de pérdida personalizada
def custom_mse(y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
    """
    Función de pérdida MSE personalizada que penaliza más las sobrepredicciones.
    
    Parámetros:
    -----------
    y_true : tf.Tensor
        Valores reales
    y_pred : tf.Tensor
        Valores predichos
        
    Retorna:
    --------
    tf.Tensor
        Valor de pérdida
    """
    error = y_true - y_pred
    overprediction_penalty = tf.where(error < 0, 2 * tf.square(error), tf.square(error))
    return tf.reduce_mean(overprediction_penalty)

# %%
# Crear y compilar modelo
model = create_lstm_model(X_cgm_train.shape, X_other_train.shape)
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss=custom_mse
)
model.summary()

# %%
# Entrenar modelo
print("\nEntrenando modelo LSTM...")
history = model.fit(
    [X_cgm_train, X_other_train],
    y_train,
    validation_data=([X_cgm_val, X_other_val], y_val),
    epochs=100,
    batch_size=32,
    callbacks=[
        tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True
        )
    ],
    verbose=1
)

# %%
# Evaluación del modelo
print("\nEvaluación del modelo LSTM:")
y_pred = model.predict([X_cgm_test, X_other_test]).flatten()

# Limpiar datos para métricas
def clean_predictions(y_true, y_pred):
    """
    Limpia predicciones eliminando valores infinitos o NaN.
    
    Parámetros:
    -----------
    y_true : np.ndarray
        Valores reales
    y_pred : np.ndarray
        Valores predichos
        
    Retorna:
    --------
    tuple
        Valores limpios (y_true, y_pred) y número de valores eliminados
    """
    mask = np.isfinite(y_pred) & np.isfinite(y_true)
    return y_true[mask], y_pred[mask], np.sum(~mask)

# Calcular métricas
y_test_clean, y_pred_clean, dropped = clean_predictions(y_test, y_pred)
print(f"\nValores eliminados en la evaluación: {dropped}")

mae = mean_absolute_error(y_test_clean, y_pred_clean)
rmse = np.sqrt(mean_squared_error(y_test_clean, y_pred_clean))
r2 = r2_score(y_test_clean, y_pred_clean)

print(f"MAE LSTM: {mae:.2f} u. de insulina")
print(f"RMSE LSTM: {rmse:.2f} u. de insulina")
print(f"R² LSTM: {r2:.2f}")

# %%
# Evaluación por sujeto
print("\nRendimiento por sujeto:")
for subject_id in test_subjects:
    mask = subject_test == subject_id
    y_test_sub = y_test[mask]
    y_pred_sub = y_pred[mask]
    
    # Limpiar datos del sujeto
    y_test_sub_clean, y_pred_sub_clean, dropped_sub = clean_predictions(y_test_sub, y_pred_sub)
    
    if len(y_test_sub_clean) > 0:
        mae_sub = mean_absolute_error(y_test_sub_clean, y_pred_sub_clean)
        rmse_sub = np.sqrt(mean_squared_error(y_test_sub_clean, y_pred_sub_clean))
        r2_sub = r2_score(y_test_sub_clean, y_pred_sub_clean)
        print(
            f"Sujeto {subject_id}: "
            f"MAE={mae_sub:.2f}, "
            f"RMSE={rmse_sub:.2f}, "
            f"R²={r2_sub:.2f}, "
            f"Valores eliminados={dropped_sub}"
        )
    else:
        print(f"Sujeto {subject_id}: No hay suficientes datos válidos")

# %%
# Visualización del entrenamiento
plt.figure(figsize=(10, 5))
plt.plot(history.history['loss'], label='Pérdida del Entrenamiento')
plt.plot(history.history['val_loss'], label='Pérdida de la Validación')
plt.xlabel('Épocas')
plt.ylabel('Pérdida MSE Personalizada')
plt.legend()
plt.title('Historial de Entrenamiento LSTM')
plt.savefig(os.path.join(FIGURES_DIR, 'training_history.png'), dpi=300, bbox_inches='tight')
plt.close()

# Visualización de predicciones
plt.figure(figsize=(12, 5))

# %%
# Predicciones vs valores reales
plt.subplot(1, 2, 1)
plt.scatter(y_test_clean, y_pred_clean, alpha=0.5, label='LSTM')
plt.plot([0, 15], [0, 15], 'r--')
plt.xlabel('Dosis Real (u. de insulina)')
plt.ylabel('Dosis Predicha (u. de insulina)')
plt.legend()
plt.title('Predicción vs. Real (LSTM)')

# %%
# Distribución de residuos
plt.subplot(1, 2, 2)
residuals = y_test_clean - y_pred_clean
plt.hist(residuals, bins=20, alpha=0.5, label='Residuos LSTM')
plt.xlabel('Residuo (u. de insulina)')
plt.ylabel('Frecuencia')
plt.legend()
plt.title('Distribución de Residuos (LSTM)')

plt.tight_layout()
plt.savefig(os.path.join(FIGURES_DIR, 'predictions.png'), dpi=300, bbox_inches='tight')
plt.close()

# %%
# Guardar el modelo
model.save(os.path.join(FIGURES_DIR, 'lstm_model.h5'))
print(f"\nModelo guardado en: {os.path.join(FIGURES_DIR, 'lstm_model.h5')}")