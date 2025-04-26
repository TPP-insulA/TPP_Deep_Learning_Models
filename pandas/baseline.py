# %%
import pandas as pd
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

# Asegurarse de que matplotlib esté instalado (ya está en tu código)
# %pip install matplotlib
import matplotlib
matplotlib.use('TkAgg')  # o 'Agg' para no interactivo
import matplotlib.pyplot as plt

# %%
# Los sujetos están en la raíz y cada archivo comienza con "Subject"
PROJECT_DIR = os.path.abspath(os.path.join(os.getcwd(), ".."))
subject_folder = os.path.join(PROJECT_DIR, "data", "Subjects")
subject_files = [f for f in os.listdir(subject_folder) if f.startswith("Subject") and f.endswith(".xlsx")]
print(f"Total de sujetos: {len(subject_files)}")

# %%
# Paso 1: Preprocesar datos para todos los sujetos
def get_cgm_window(bolus_time, cgm_df, window_hours=2, interval_minutes=5):
    window_start = bolus_time - timedelta(hours=window_hours)
    window = cgm_df[(cgm_df['date'] >= window_start) & (cgm_df['date'] <= bolus_time)]
    window = window.sort_values('date').tail(24)  # Últimas 24 lecturas (~2 horas)
    if len(window) < 24:
        return None
    return window['mg/dl'].values

def calculate_iob(bolus_time, basal_df, half_life_hours=4):
    if basal_df is None or basal_df.empty:
        return 0.0  # IOB predeterminado si no hay datos basales
    iob = 0
    for _, row in basal_df.iterrows():
        start_time = row['date']
        duration_hours = row['duration'] / (1000 * 3600)  # Convertir ms a horas
        end_time = start_time + timedelta(hours=duration_hours)
        rate = row['rate'] if pd.notna(row['rate']) else 0.9  # Tasa predeterminada
        if start_time <= bolus_time <= end_time:
            time_since_start = (bolus_time - start_time).total_seconds() / 3600
            remaining = rate * (1 - (time_since_start / half_life_hours))
            iob += max(0, remaining)
    return iob

# Función optimizada para procesar un solo sujeto
def process_subject(subject_path, idx):
    print(f"Procesando {os.path.basename(subject_path)} ({idx+1}/{len(subject_files)})...")
    
    # Cargar datos eficientemente usando pd.ExcelFile
    try:
        excel_file = pd.ExcelFile(subject_path)
        cgm_df = pd.read_excel(excel_file, sheet_name="CGM")
        bolus_df = pd.read_excel(excel_file, sheet_name="Bolus")
        try:
            basal_df = pd.read_excel(excel_file, sheet_name="Basal")
        except ValueError:
            basal_df = None  # No hay hoja Basal
    except Exception as e:
        print(f"Error al cargar {os.path.basename(subject_path)}: {e}")
        return []

    # Convertir fechas y preordenar CGM para eficiencia
    cgm_df['date'] = pd.to_datetime(cgm_df['date'])
    cgm_df = cgm_df.sort_values('date')  # Preordenar una vez
    bolus_df['date'] = pd.to_datetime(bolus_df['date'])
    if basal_df is not None:
        basal_df['date'] = pd.to_datetime(basal_df['date'])

    # Proceso de preprocesamiento
    processed_data = []
    for _, row in bolus_df.iterrows():
        bolus_time = row['date']
        cgm_window = get_cgm_window(bolus_time, cgm_df)
        if cgm_window is not None:
            iob = calculate_iob(bolus_time, basal_df)
            features = {
                'subject_id': idx,
                'cgm_window': cgm_window,
                'carbInput': row['carbInput'] if pd.notna(row['carbInput']) else 0.0,
                'bgInput': row['bgInput'] if pd.notna(row['bgInput']) else cgm_window[-1],
                'insulinCarbRatio': row['insulinCarbRatio'] if pd.notna(row['insulinCarbRatio']) else 10.0,
                'insulinSensitivityFactor': 50.0,
                'insulinOnBoard': iob,
                'normal': row['normal']
            }
            processed_data.append(features)
    
    return processed_data

# Ejecución en paralelo
all_processed_data = Parallel(n_jobs=-1)(
    delayed(process_subject)(
        os.path.join(subject_folder, f), 
        idx
    ) for idx, f in enumerate(subject_files)
)

# Aplanar la lista de listas
all_processed_data = [item for sublist in all_processed_data for item in sublist]

# Convertir a DataFrame
df_processed = pd.DataFrame(all_processed_data)
print("Muestra de datos procesados combinados:")
print(df_processed.head())
print(f"Total de muestras: {len(df_processed)}")

# %%
# Dividir ventana CGM en columnas
cgm_columns = [f'cgm_{i}' for i in range(24)]
df_cgm = pd.DataFrame(df_processed['cgm_window'].tolist(), columns=cgm_columns, index=df_processed.index)
df_final = pd.concat([df_cgm, df_processed.drop(columns=['cgm_window'])], axis=1)

# %%
# Verificar valores NaN
print("Verificación de NaN en df_final:")
df_final = df_final.dropna()
print(df_final.isna().sum())

# %%
# Normalizar características
scaler_cgm = MinMaxScaler(feature_range=(0, 1))
scaler_other = StandardScaler()
X_cgm = scaler_cgm.fit_transform(df_final[cgm_columns])
X_other = scaler_other.fit_transform(df_final[['carbInput', 'bgInput', 'insulinOnBoard']])
X = np.hstack([X_cgm, X_other, df_final[['insulinCarbRatio', 'insulinSensitivityFactor', 'subject_id']].values])
y = df_final['normal'].values

# %%

# Convertir a arrays numpy y manejar diferentes tipos de datos
X = np.asarray(X)
y = np.asarray(y)

# Imprimir tipos de datos para depuración
print("X dtype:", X.dtype)
print("y dtype:", y.dtype)

# Verificar valores NaN según el tipo de datos
def check_nan(arr):
    if np.issubdtype(arr.dtype, np.number):
        return np.isnan(arr).sum()
    else:
        # Para tipos no numéricos, verificar usando pandas
        return pd.isna(arr).sum()

print("NaN en X:", check_nan(X))
print("NaN en y:", check_nan(y))

# Generar error si se encuentran valores NaN
if check_nan(X) > 0 or check_nan(y) > 0:
    raise ValueError("Valores NaN detectados en X o y")
# %%
# División por sujeto en conjuntos de entrenamiento-validación-prueba
subject_ids = df_final['subject_id'].unique()
train_subjects, temp_subjects = train_test_split(subject_ids, test_size=0.2, random_state=42)
val_subjects, test_subjects = train_test_split(temp_subjects, test_size=0.5, random_state=42)

train_mask = df_final['subject_id'].isin(train_subjects)
val_mask = df_final['subject_id'].isin(val_subjects)
test_mask = df_final['subject_id'].isin(test_subjects)

X_train, y_train = X[train_mask], y[train_mask]
X_val, y_val = X[val_mask], y[val_mask]
X_test, y_test = X[test_mask], y[test_mask]
subject_test = df_final[test_mask]['subject_id'].values

print(f"Entrenamiento: {X_train.shape}, Validación: {X_val.shape}, Prueba: {X_test.shape}")
print(f"Sujetos de prueba: {test_subjects}")

# %%
# Paso 2: Establecer un modelo base
print("\n=== Paso 2: Establecer un modelo base ===")

# Definir modelo FNN (ajustado para conjunto de datos más grande)
model = Sequential([
    Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
    Dropout(0.2),
    Dense(32, activation='relu'),
    Dropout(0.2),
    Dense(1, activation='linear')
])
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), loss='mse')
model.summary()

# %%
import numpy as np

# Convertir datos a float32
X_train = np.array(X_train, dtype=np.float32)
X_val = np.array(X_val, dtype=np.float32)
y_train = np.array(y_train, dtype=np.float32)
y_val = np.array(y_val, dtype=np.float32)

# Verificar formas y tipos de datos
print("Forma de X_train:", X_train.shape, "dtype:", X_train.dtype)
print("Forma de X_val:", X_val.shape, "dtype:", X_val.dtype)
print("Forma de y_train:", y_train.shape, "dtype:", y_train.dtype)
print("Forma de y_val:", y_val.shape, "dtype:", y_val.dtype)

# Entrenar modelo
history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=100,  # Incremento de épocas para conjunto de datos más grande
    batch_size=32,  # Tamaño de lote incrementado
    callbacks=[tf.keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True)],
    verbose=1
)

# %%
# Graficar historial de entrenamiento
plt.figure(figsize=(10, 5))
plt.plot(history.history['loss'], label='Pérdida de Entrenamiento')
plt.plot(history.history['val_loss'], label='Pérdida de Validación')
plt.xlabel('Época')
plt.ylabel('Pérdida MSE')
plt.legend()
plt.title('Historial de Entrenamiento (Todos los Sujetos)')
plt.show()

# %%
# Paso 3: Evaluar el modelo base
print("\n=== Paso 3: Evaluar el modelo base ===")

# Predicciones
y_pred = model.predict(X_test).flatten()

# Métricas generales
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)
print(f"MAE general: {mae:.2f} unidades")
print(f"RMSE general: {rmse:.2f} unidades")
print(f"R² general: {r2:.2f}")

# %%
# Línea base basada en reglas
def rule_based_prediction(X, target_bg=100):
    carb_input = scaler_other.inverse_transform(X[:, 24:27])[:, 0]
    bg_input = scaler_other.inverse_transform(X[:, 24:27])[:, 1]
    icr = X[:, 27]
    isf = X[:, 28]
    print(f"carb_input: {carb_input}, bg_input: {bg_input}, icr: {icr}, isf: {isf}")
    return carb_input / icr + (bg_input - target_bg) / isf

y_rule = rule_based_prediction(X_test)
mae_rule = mean_absolute_error(y_test, y_rule)
rmse_rule = np.sqrt(mean_squared_error(y_test, y_rule))
r2_rule = r2_score(y_test, y_rule)
print(f"MAE basado en reglas: {mae_rule:.2f} unidades")
print(f"RMSE basado en reglas: {rmse_rule:.2f} unidades")
print(f"R² basado en reglas: {r2_rule:.2f}")

# %%
# Métricas por sujeto
print("\nRendimiento por sujeto:")
for subject_id in test_subjects:
    mask = subject_test == subject_id
    y_test_sub = y_test[mask]
    y_pred_sub = y_pred[mask]
    y_rule_sub = y_rule[mask]
    if len(y_test_sub) > 0:
        mae_sub = mean_absolute_error(y_test_sub, y_pred_sub)
        rmse_sub = np.sqrt(mean_squared_error(y_test_sub, y_pred_sub))
        r2_sub = r2_score(y_test_sub, y_pred_sub)
        mae_rule_sub = mean_absolute_error(y_test_sub, y_rule_sub)
        print(f"Sujeto {subject_id}: FNN MAE={mae_sub:.2f}, RMSE={rmse_sub:.2f}, R²={r2_sub:.2f}, MAE basado en reglas={mae_rule_sub:.2f}")

# %%
# Visualización
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.scatter(y_test, y_pred, label='Predicciones FNN', alpha=0.5)
plt.scatter(y_test, y_rule, label='Basado en Reglas', alpha=0.5)
plt.plot([0, 15], [0, 15], 'r--')
plt.xlabel('Dosis Real (unidades)')
plt.ylabel('Dosis Predicha (unidades)')
plt.legend()
plt.title('Predicciones vs Real (Todos los Sujetos)')

plt.subplot(1, 2, 2)
plt.hist(y_test - y_pred, bins=20, label='Residuos FNN', alpha=0.5)
plt.hist(y_test - y_rule, bins=20, label='Residuos Basados en Reglas', alpha=0.5)
plt.xlabel('Residuo (unidades)')
plt.ylabel('Frecuencia')
plt.legend()
plt.title('Distribución de Residuos (Todos los Sujetos)')
plt.tight_layout()
plt.show()
# %%
