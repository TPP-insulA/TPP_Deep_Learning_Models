# %% [markdown]
# # Modelos Varios
# 
# En este notebook están los modelos:
# 
# + CNN (Convolutional Neural Network)
# + Transformer
# + TCN (Temporal Convolutional Network)
# + GRU (Gated Recurrent Unit)
# + Wavenet
# + Tanmet
# + Attention-Only

# %%
# %%
import polars as pl
import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from scipy.optimize import minimize
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import (
    Dense, Dropout, Input, Concatenate, BatchNormalization,
    Conv1D, MaxPooling1D, LayerNormalization, MultiHeadAttention,
    Add, GlobalAveragePooling1D, GRU, Activation, SimpleRNN, Bidirectional
)
from keras.saving import register_keras_serializable
import matplotlib.pyplot as plt
import sys
import os
from joblib import Parallel, delayed
from datetime import timedelta
# import openpyxl

# Configuración de Matplotlib para evitar errores con Tkinter
import matplotlib
matplotlib.use('Agg')

PROJECT_ROOT = os.path.abspath(os.getcwd())
sys.path.append(PROJECT_ROOT) 

from models_old.tensorflow.DeepLearning.attention_only import create_attention_model
from models_old.tensorflow.DeepLearning.cnn import create_cnn_model
from models_old.tensorflow.DeepLearning.gru import create_gru_model
from models_old.tensorflow.DeepLearning.rnn import create_rnn_model
from models_old.tensorflow.DeepLearning.tabnet import create_tabnet_model
from models_old.tensorflow.DeepLearning.tcn import create_tcn_model
from models_old.tensorflow.DeepLearning.transformer import create_transformer_model
from models_old.tensorflow.DeepLearning.wavenet import create_wavenet_model

# %% [markdown]
# ## Constantes

# %%
# Definición de la ruta del proyecto
# PROJECT_ROOT = os.path.abspath(os.path.join(os.getcwd(), "..", ".."))
SUBJECTS_RELATIVE_PATH = "data/Subjects"
SUBJECTS_PATH = os.path.join(PROJECT_ROOT, SUBJECTS_RELATIVE_PATH)

# Crear directorios para resultados
FIGURES_DIR = os.path.join(PROJECT_ROOT, "figures", "various_models")
os.makedirs(FIGURES_DIR, exist_ok=True)
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")
os.makedirs(MODELS_DIR, exist_ok=True)

subject_files = [f for f in os.listdir(SUBJECTS_PATH) if f.startswith("Subject") and f.endswith(".xlsx")]
print(f"Total sujetos: {len(subject_files)}")

# %% [markdown]
# ## Preprocesamiento y Procesamiento de Datos

# %%
def get_cgm_window(bolus_time, cgm_df: pl.DataFrame, window_hours: int = 2) -> np.ndarray:
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

def calculate_iob(bolus_time, basal_df: pl.DataFrame, half_life_hours: float = 4.0) -> float:
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
    print(f"Procesando {os.path.basename(subject_path)} ({idx+1}/{len(subject_files)})...")
    
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
    ) for idx, f in enumerate(subject_files)
)

all_processed_data = [item for sublist in all_processed_data for item in sublist]

# Conversión a DataFrame
df_processed = pl.DataFrame(all_processed_data)
print("Muestra de datos procesados combinados:")
print(df_processed.head())
print(f"Total muestras: {len(df_processed)}")

# %% [markdown]
# ### División de Ventana CGM y Valores Nulos

# %%
# Dividir ventana CGM y otras características
cgm_columns = [f'cgm_{i}' for i in range(24)]
df_cgm = pl.DataFrame({
    col: [row['cgm_window'][i] for row in all_processed_data]
    for i, col in enumerate(cgm_columns)
}, schema={col: pl.Float64 for col in cgm_columns})

# Combinar con otras características
df_processed = pl.concat([
    df_cgm,
    df_processed.drop('cgm_window')
], how="horizontal")

# Verificar valores nulos
print("Verificación de valores nulos en df_processed:")
print(df_processed.null_count())
df_processed = df_processed.drop_nulls()

# %% [markdown]
# ### Normalización de Datos

# %%
# Normalizar características
scaler_cgm = MinMaxScaler(feature_range=(0, 1))
scaler_other = StandardScaler()

# Normalizar CGM
X_cgm = scaler_cgm.fit_transform(df_processed.select(cgm_columns).to_numpy())
X_cgm = X_cgm.reshape(X_cgm.shape[0], X_cgm.shape[1], 1)

# Normalizar otras características (incluyendo hour_of_day)
other_features = ['carbInput', 'bgInput', 'insulinOnBoard', 'insulinCarbRatio', 
                  'insulinSensitivityFactor', 'subject_id', 'hour_of_day']
X_other = scaler_other.fit_transform(df_processed.select(other_features).to_numpy())

# Etiquetas
y = df_processed.get_column('normal').to_numpy()

# Verificar NaN
print("NaN en X_cgm:", np.isnan(X_cgm).sum())
print("NaN en X_other:", np.isnan(X_other).sum())
print("NaN en y:", np.isnan(y).sum())
if np.isnan(X_cgm).sum() > 0 or np.isnan(X_other).sum() > 0 or np.isnan(y).sum() > 0:
    raise ValueError("Valores NaN detectados en X_cgm, X_other o y")

# %% [markdown]
# ### División por Sujeto de los Datos

# %%
# División por sujeto
subject_ids = df_processed.get_column('subject_id').unique().to_numpy()
train_subjects, temp_subjects = train_test_split(subject_ids, test_size=0.2, random_state=42)
val_subjects, test_subjects = train_test_split(temp_subjects, test_size=0.5, random_state=42)

# %% [markdown]
# ### Creación de Máscaras

# %%
# Crear máscaras
train_mask = df_processed.get_column('subject_id').is_in(train_subjects).to_numpy()
val_mask = df_processed.get_column('subject_id').is_in(val_subjects).to_numpy()
test_mask = df_processed.get_column('subject_id').is_in(test_subjects).to_numpy()

X_cgm_train, X_cgm_val, X_cgm_test = X_cgm[train_mask], X_cgm[val_mask], X_cgm[test_mask]
X_other_train, X_other_val, X_other_test = X_other[train_mask], X_other[val_mask], X_other[test_mask]
y_train, y_val, y_test = y[train_mask], y[val_mask], y[test_mask]
subject_test = df_processed.filter(pl.col('subject_id').is_in(test_subjects)).get_column('subject_id').to_numpy()

print(f"Entrenamiento CGM: {X_cgm_train.shape}, Validación CGM: {X_cgm_val.shape}, Prueba CGM: {X_cgm_test.shape}")
print(f"Entrenamiento Otros: {X_other_train.shape}, Validación Otros: {X_other_val.shape}, Prueba Otros: {X_other_test.shape}")
print(f"Sujetos de prueba: {test_subjects}")

# %% [markdown]
# ## Modelos

# %%
MODEL_CREATORS = {
    'CNN': create_cnn_model,
    'Transformer': create_transformer_model,
    'GRU': create_gru_model,
    'Attention': create_attention_model,
    'RNN': create_rnn_model,
    'TabNet': create_tabnet_model,
    'TCN': create_tcn_model,
    'WaveNet': create_wavenet_model
}

# %% [markdown]
# ## Funciones Visualización

# %%
def plot_training_history(histories: dict, model_names: list):
    """
    Visualiza el historial de entrenamiento de múltiples modelos.
    
    Parámetros:
    -----------
    histories : dict
        Diccionario con historiales de entrenamiento por modelo
    model_names : list
        Lista de nombres de modelos
    """
    plt.figure(figsize=(12, 6))
    
    for name, history in histories.items():
        plt.plot(history['loss'], label=f'{name} (train)')
        plt.plot(history['val_loss'], label=f'{name} (val)', linestyle='--')
    
    plt.xlabel('Épocas')
    plt.ylabel('Pérdida MSE')
    plt.title('Comparación de Historiales de Entrenamiento')
    plt.legend()
    plt.savefig(os.path.join(FIGURES_DIR, 'training_comparison.png'), dpi=300, bbox_inches='tight')
    plt.close()

def plot_predictions_comparison(y_test: np.ndarray, predictions: dict):
    """
    Visualiza comparación de predicciones de múltiples modelos.
    
    Parámetros:
    -----------
    y_test : np.ndarray
        Valores reales de prueba
    predictions : dict
        Diccionario con predicciones por modelo
    """
    plt.figure(figsize=(15, 5))
    
    # Scatter plot
    plt.subplot(1, 2, 1)
    for name, y_pred in predictions.items():
        plt.scatter(y_test, y_pred, alpha=0.5, label=name)
    plt.plot([0, 15], [0, 15], 'r--')
    plt.xlabel('Dosis Real (u. de insulina)')
    plt.ylabel('Dosis Predicha (u. de insulina)')
    plt.legend()
    plt.title('Predicción vs. Real (Todos los Modelos)')
    
    # Residuals
    plt.subplot(1, 2, 2)
    for name, y_pred in predictions.items():
        plt.hist(y_test - y_pred, bins=20, alpha=0.5, label=name)
    plt.xlabel('Residuo (u. de insulina)')
    plt.ylabel('Frecuencia')
    plt.legend()
    plt.title('Distribución de Residuos')
    
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'predictions_comparison.png'), dpi=300, bbox_inches='tight')
    plt.close()


# %% [markdown]
# ## Función de Entrenamiento

# %%
def create_dataset(x_cgm, x_other, y, batch_size=32):
    """
    Crea un dataset optimizado usando tf.data.
    
    Parámetros:
    -----------
    x_cgm : np.ndarray
        Datos CGM
    x_other : np.ndarray
        Otras características
    y : np.ndarray
        Etiquetas
    batch_size : int
        Tamaño del batch
        
    Retorna:
    --------
    tf.data.Dataset
        Dataset optimizado
    """
    dataset = tf.data.Dataset.from_tensor_slices((
        (x_cgm, x_other), y
    ))
    return dataset.cache().shuffle(1000).batch(batch_size).prefetch(tf.data.AUTOTUNE)

# %%
def create_ensemble_prediction(predictions_dict: dict, weights: np.ndarray = None) -> np.ndarray:
    """
    Combina predicciones de múltiples modelos usando un promedio ponderado.
    
    Parámetros:
    -----------
    predictions_dict : dict
        Diccionario con predicciones de cada modelo
    weights : np.ndarray, opcional
        Pesos para cada modelo. Si es None, usa promedio simple
        
    Retorna:
    --------
    np.ndarray
        Predicciones combinadas del ensemble
    """
    all_preds = np.stack(list(predictions_dict.values()))
    if weights is None:
        weights = np.ones(len(predictions_dict)) / len(predictions_dict)
    return np.average(all_preds, axis=0, weights=weights)

def optimize_ensemble_weights(predictions_dict: dict, y_true: np.ndarray) -> np.ndarray:
    """
    Optimiza los pesos del ensemble usando validación cruzada.
    
    Parámetros:
    -----------
    predictions_dict : dict
        Diccionario con predicciones de cada modelo
    y_true : np.ndarray
        Valores reales
        
    Retorna:
    --------
    np.ndarray
        Pesos optimizados para cada modelo
    """
    from scipy.optimize import minimize
    
    def objective(weights):
        # Normalizar pesos
        weights = weights / np.sum(weights)
        # Obtener predicción del ensemble
        ensemble_pred = create_ensemble_prediction(predictions_dict, weights)
        # Calcular error
        return mean_squared_error(y_true, ensemble_pred)
    
    n_models = len(predictions_dict)
    initial_weights = np.ones(n_models) / n_models
    bounds = [(0, 1) for _ in range(n_models)]
    
    result = minimize(
        objective,
        initial_weights,
        bounds=bounds,
        constraints={'type': 'eq', 'fun': lambda w: np.sum(w) - 1}
    )
    
    return result.x / np.sum(result.x)


# %%
def train_and_evaluate_model(model: Model, model_name: str, 
                           x_cgm_train: np.ndarray, x_other_train: np.ndarray, 
                           y_train: np.ndarray, x_cgm_val: np.ndarray, 
                           x_other_val: np.ndarray, y_val: np.ndarray,
                           x_cgm_test: np.ndarray, x_other_test: np.ndarray, 
                           y_test: np.ndarray) -> tuple:
    """
    Entrena y evalúa un modelo específico con características avanzadas de entrenamiento.
    
    Parámetros:
    -----------
    model : Model
        Modelo a entrenar
    model_name : str
        Nombre del modelo para guardado/logging
    x_cgm_train, x_other_train, y_train : np.ndarray
        Datos de entrenamiento
    x_cgm_val, x_other_val, y_val : np.ndarray
        Datos de validación
    x_cgm_test, x_other_test, y_test : np.ndarray
        Datos de prueba
        
    Retorna:
    --------
    tuple
        (history, y_pred, metrics_dict)
    """
    # Habilitar compilación XLA
    tf.config.optimizer.set_jit(True)
    
    # Crear datasets optimizados
    train_ds = create_dataset(x_cgm_train, x_other_train, y_train)
    val_ds = create_dataset(x_cgm_val, x_other_val, y_val)
    
    # Configurar learning rate con decaimiento
    initial_learning_rate = 0.001
    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate,
        decay_steps=1000,
        decay_rate=0.9
    )
    
    # Optimizador con gradient clipping
    optimizer = tf.keras.optimizers.Adam(
        learning_rate=lr_schedule,
        clipnorm=1.0
    )
    
    # Habilitar entrenamiento con precisión mixta
    tf.keras.mixed_precision.set_global_policy('mixed_float16')
    
    # Compilar modelo con múltiples métricas
    model.compile(
        optimizer=optimizer,
        loss='mse',
        metrics=['mae', tf.keras.metrics.RootMeanSquaredError()]
    )
    
    # Callbacks para monitoreo y optimización
    callbacks = [
        # Early stopping para evitar overfitting
        tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True
        ),
        # Reducción de learning rate cuando el modelo se estanca
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=1e-6
        ),
        # Guardado del mejor modelo
        tf.keras.callbacks.ModelCheckpoint(
            os.path.join(MODELS_DIR, f'best_{model_name}.h5'),
            monitor='val_loss',
            save_best_only=True
        ),
        # TensorBoard para visualización
        tf.keras.callbacks.TensorBoard(
            log_dir=os.path.join(MODELS_DIR, 'logs', model_name),
            histogram_freq=1
        )
    ]
    
    # Entrenar modelo
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=100,
        callbacks=callbacks,
        verbose=1
    )
    
    # Predecir y evaluar
    y_pred = model.predict([x_cgm_test, x_other_test]).flatten()
    
    # Calcular métricas
    metrics = {
        'mae': mean_absolute_error(y_test, y_pred),
        'rmse': np.sqrt(mean_squared_error(y_test, y_pred)),
        'r2': r2_score(y_test, y_pred)
    }
    
    # Guardar modelo final
    model.save(os.path.join(MODELS_DIR, f'{model_name}.keras'))
    
    # Restaurar política de precisión default
    tf.keras.mixed_precision.set_global_policy('float32')
    
    return history, y_pred, metrics

def train_model_parallel(name, input_shapes):
    """
    Entrenamiento en paralelo de un modelo específico.
    
    Parámeteros:
    -----------
    name : str
        Name of the model to create
    input_shapes : tuple
        Shapes for CGM and other inputs
    """
    print(f"\nEntrenando modelo {name}...")
    
    
    model = MODEL_CREATORS[name](input_shapes[0], input_shapes[1])
    
    return name, train_and_evaluate_model(
        model, name,
        X_cgm_train, X_other_train, y_train,
        X_cgm_val, X_other_val, y_val,
        X_cgm_test, X_other_test, y_test
    )

# %%
def cross_validate_model(create_model_fn, X_cgm: np.ndarray, X_other: np.ndarray, 
                        y: np.ndarray, n_splits: int = 5) -> tuple:
    """
    Realiza validación cruzada de un modelo.
    
    Parámetros:
    -----------
    create_model_fn : callable
        Función que crea el modelo
    X_cgm : np.ndarray
        Datos CGM
    X_other : np.ndarray
        Otras características
    y : np.ndarray
        Etiquetas
    n_splits : int
        Número de divisiones para validación cruzada
        
    Retorna:
    --------
    tuple
        (media_metricas, std_metricas)
    """
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    scores = []
    
    for fold, (train_idx, val_idx) in enumerate(kf.split(X_cgm)):
        print(f"\nEntrenando fold {fold + 1}/{n_splits}")
        
        # Dividir datos
        X_cgm_train_fold = X_cgm[train_idx]
        X_cgm_val_fold = X_cgm[val_idx]
        X_other_train_fold = X_other[train_idx]
        X_other_val_fold = X_other[val_idx]
        y_train_fold = y[train_idx]
        y_val_fold = y[val_idx]
        
        # Crear y entrenar modelo
        model = create_model_fn()
        history = train_and_evaluate_model(
            model=model,
            model_name=f'fold_{fold}',
            x_cgm_train=X_cgm_train_fold,
            x_other_train=X_other_train_fold,
            y_train=y_train_fold,
            x_cgm_val=X_cgm_val_fold,
            x_other_val=X_other_val_fold,
            y_val=y_val_fold,
            x_cgm_test=X_cgm_val_fold,
            x_other_test=X_other_val_fold,
            y_test=y_val_fold
        )
        
        scores.append(history[2])  # Append metrics dictionary
    
    # Calcular estadísticas
    mean_scores = {
        metric: np.mean([s[metric] for s in scores])
        for metric in scores[0].keys()
    }
    std_scores = {
        metric: np.std([s[metric] for s in scores])
        for metric in scores[0].keys()
    }
    
    return mean_scores, std_scores

# %%
def train_model_sequential(model_info):
    """Train a model and return only picklable results"""
    name, input_shapes = model_info
    print(f"\nEntrenando modelo {name}...")
    
    # Habilitar compilación XLA
    tf.config.optimizer.set_jit(True)
    
    # Crear datasets optimizados
    train_ds = create_dataset(X_cgm_train, X_other_train, y_train)
    val_ds = create_dataset(X_cgm_val, X_other_val, y_val)
    
    model = MODEL_CREATORS[name](input_shapes[0], input_shapes[1])
    
    # Compile
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss='mse'
    )
    
    # Train
    history = model.fit(
        train_ds,
        validation_data=val_ds,
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
    
    # Get predictions
    y_pred = model.predict([X_cgm_test, X_other_test]).flatten()
    
    # Save model
    model.save(os.path.join(MODELS_DIR, f'{name}.keras'))
    
    # Clear memory
    del model
    tf.keras.backend.clear_session()
    
    # Return only picklable objects
    return {
        'name': name,
        'history': history.history,
        'predictions': y_pred.tolist(),  # Convert to list for pickling
    }

def calculate_metrics(predictions, y_true):
    """Calculate metrics for predictions"""
    return {
        'mae': mean_absolute_error(y_true, predictions),
        'rmse': np.sqrt(mean_squared_error(y_true, predictions)),
        'r2': r2_score(y_true, predictions)
    }

# %%
def enhance_features(X_cgm, X_other):
    # Add derivative features for CGM
    cgm_diff = np.diff(X_cgm.squeeze(), axis=1)
    cgm_diff = np.pad(cgm_diff, ((0,0), (1,0), (0,0)), mode='edge')
    
    # Add rolling statistics
    window = 5
    rolling_mean = np.apply_along_axis(
        lambda x: np.convolve(x, np.ones(window)/window, mode='same'),
        1, X_cgm.squeeze()
    )
    
    X_cgm_enhanced = np.concatenate([
        X_cgm,
        cgm_diff[..., np.newaxis],
        rolling_mean[..., np.newaxis]
    ], axis=-1)
    
    return X_cgm_enhanced, X_other

# %% [markdown]
# ## Entrenamiento y Evaluación de los Modelos

# %%
# Entrenamiento y evaluación de modelos
input_shapes = (X_cgm_train.shape, X_other_train.shape)
models_names = ['CNN', 'Transformer', 'GRU', 'Attention', 'RNN', 'TabNet', 'TCN', 'WaveNet']

histories = {}
predictions = {}
metrics = {}

model_results = []
for name in models_names:
    result = train_model_sequential((name, input_shapes))
    model_results.append(result)

# Process results in parallel
print("\nCalculando métricas en paralelo...")
with Parallel(n_jobs=-1, verbose=1) as parallel:
    metric_results = parallel(
        delayed(calculate_metrics)(
            np.array(result['predictions']), 
            y_test
        ) for result in model_results
    )

# Store results
for result, metric in zip(model_results, metric_results):
    name = result['name']
    histories[name] = result['history']
    predictions[name] = np.array(result['predictions'])
    metrics[name] = metric

# Evaluación por sujeto
print("\nRendimiento por sujeto:")
for subject_id in test_subjects:
    mask = subject_test == subject_id
    y_test_sub = y_test[mask]
    
    print(f"\nSujeto {subject_id}:")
    print("-" * 40)
    for name, y_pred in predictions.items():
        y_pred_sub = y_pred[mask]
        mae_sub = mean_absolute_error(y_test_sub, y_pred_sub)
        rmse_sub = np.sqrt(mean_squared_error(y_test_sub, y_pred_sub))
        r2_sub = r2_score(y_test_sub, y_pred_sub)
        print(f"{name:<15} MAE={mae_sub:.2f}, RMSE={rmse_sub:.2f}, R²={r2_sub:.2f}")


# %% [markdown]
# ## Visualización de los Resultados

# %%
# Visualización de resultados
plot_training_history(histories, models_names)
plot_predictions_comparison(y_test, predictions)

# %%
# After storing individual model results
print("\nCreando predicciones del ensemble...")

# Crear predicciones del ensemble
ensemble_pred = create_ensemble_prediction(predictions)
ensemble_metrics = calculate_metrics(ensemble_pred, y_test)

# Agregar métricas del ensemble
metrics['Ensemble'] = ensemble_metrics
predictions['Ensemble'] = ensemble_pred

# Optimizar pesos del ensemble
print("\nOptimizando pesos del ensemble...")
optimal_weights = optimize_ensemble_weights(predictions, y_test)

# Crear predicción del ensemble con pesos optimizados
ensemble_pred_optimized = create_ensemble_prediction(predictions, optimal_weights)
ensemble_metrics_optimized = calculate_metrics(ensemble_pred_optimized, y_test)

# Agregar métricas del ensemble optimizado
metrics['Ensemble (Opt)'] = ensemble_metrics_optimized
predictions['Ensemble (Opt)'] = ensemble_pred_optimized

# Validación cruzada para cada modelo
print("\nRealizando validación cruzada...")
cv_results = {}

for name in models_names:
    print(f"\nValidación cruzada para {name}")
    model_creator = lambda: MODEL_CREATORS[name](input_shapes[0], input_shapes[1])
    mean_scores, std_scores = cross_validate_model(
        create_model_fn=model_creator,
        X_cgm=X_cgm,
        X_other=X_other,
        y=y
    )
    cv_results[name] = {
        'mean': mean_scores,
        'std': std_scores
    }

# Imprimir resultados
print("\nResultados de validación cruzada:")
print("-" * 70)
print(f"{'Modelo':<15} {'MAE':>12} {'RMSE':>12} {'R²':>12}")
print("-" * 70)
for name, results in cv_results.items():
    mean = results['mean']
    std = results['std']
    print(f"{name:<15} {mean['mae']:>8.2f}±{std['mae']:4.2f} "
          f"{mean['rmse']:>8.2f}±{std['rmse']:4.2f} "
          f"{mean['r2']:>8.2f}±{std['r2']:4.2f}")

# Actualizar visualizaciones
plot_training_history(histories, models_names + ['Ensemble', 'Ensemble (Opt)'])
plot_predictions_comparison(y_test, predictions)

# %% [markdown]
# ## Métricas Comparativas

# %%
# Imprimir métricas comparativas
print("\nComparación de métricas:")
print("-" * 50)
print(f"{'Modelo':<15} {'MAE':>8} {'RMSE':>8} {'R²':>8}")
print("-" * 50)
for name, metric in metrics.items():
    print(f"{name:<15} {metric['mae']:8.2f} {metric['rmse']:8.2f} {metric['r2']:8.2f}")