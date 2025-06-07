# Imports
import os
import sys
import pandas as pd
import polars as pl
import numpy as np
import matplotlib.pyplot as plt
from sklearn.discriminant_analysis import StandardScaler
import torch
import json
from datetime import datetime
from typing import Dict, List, Tuple, Any
from validation.model_validation import validate_dosing_model
from validation.simulator import GlucoseSimulator
from constants.constants import CONST_METRIC_MAE, CONST_METRIC_R2, CONST_METRIC_RMSE, SEVERE_HYPOGLYCEMIA_THRESHOLD, HYPOGLYCEMIA_THRESHOLD, HYPERGLYCEMIA_THRESHOLD, SEVERE_HYPERGLYCEMIA_THRESHOLD, TARGET_GLUCOSE
from training.pytorch import (
    train_multiple_models, calculate_metrics, evaluate_clinical_metrics, 
    optimize_ensemble_weights_clinical, enhance_features
)



PROJECT_ROOT = os.path.abspath(os.getcwd())
sys.path.append(PROJECT_ROOT)

# Printer
from custom.printer import cprint, coloured, print_warning

# Configuración 
from config.params import FRAMEWORK, PROCESSING, MODELS, MODELS_USAGE, EVALUATE, EVALUATE_USAGE

# Procesamiento
from processing.pandas import preprocess_data as pd_preprocess, split_data as pd_split
from processing.polars import preprocess_data as pl_preprocess, split_data as pl_split

# Visualización
from visualization.plotting import visualize_model_results, plot_model_evaluation_summary

# Reporte
from report.generate_report import create_report, render_to_pdf

# Validación
from validation.model_validation import validate_model_with_simulator
from validation.simulator import GlucoseSimulator

# Constantes para los directorios y nombres
CONST_MODELS_DIR = "models"
CONST_RESULTS_DIR = "results"
CONST_ENSEMBLE = "ensemble"

# Auxiliary Functions
def is_model_creator(fn: Any) -> bool:
    """
    Verifica si una función es un model creator que debe ser llamada para obtener
    la función de creación del modelo.
    
    Parámetros:
    -----------
    fn : Any
        Función a verificar
        
    Retorna:
    --------
    bool
        True si es un model creator, False si ya es una función de creación de modelo
    """
    if callable(fn):
        try:
            import inspect
            sig = inspect.signature(fn)
            # Si no tiene parámetros, es probable que es un model creator
            # que debe llamarse para obtener la función de creación real
            return len(sig.parameters) == 0
        except Exception:
            pass
    return False

# Importación dinámica de módulos de entrenamiento según el framework seleccionado
cprint(f"Framework seleccionado: {FRAMEWORK}", 'blue', 'bold')

if torch.cuda.is_available():
    cprint(f"GPU available: {torch.cuda.device_count()} devices", 'green')
    cprint(f"Using: {torch.cuda.get_device_name(0)}", 'green')
else:
    cprint("No GPUs detected, using CPU", 'yellow')

# Constante para mensaje repetido
CONST_MODEL_ACTIVATED = "Modelo {} activado."
CONST_MODEL_DEACTIVATED = "Modelo {} desactivado."

use_models = {}

for model_name, use in MODELS_USAGE.items():
    if use:
        model_fn = MODELS[model_name]
        if is_model_creator(model_fn):
            model_fn = model_fn()
        use_models[model_name] = model_fn
        cprint(CONST_MODEL_ACTIVATED.format(model_name), 'green', 'bold')
    else:
        cprint(CONST_MODEL_DEACTIVATED.format(model_name), 'red', 'bold')

# Validaciones Previas
if PROCESSING not in ["pandas", "polars"]:
    cprint(f"Error: El procesamiento debe ser 'pandas' o 'polars'. Se recibió '{PROCESSING}'", 'red', 'bold')
    sys.exit(1)
if not use_models:
    cprint("Error: No se ha activado ningún modelo. Por favor, activa al menos un modelo en 'MODELS_USAGE'.", 'red', 'bold')
    sys.exit(1)
if MODELS_USAGE.values() == [False] * len(use_models):
    cprint("Error: Todos los modelos están desactivados. Por favor, activa al menos un modelo en 'MODELS_USAGE'.", 'red', 'bold')
    sys.exit(1)
if len(use_models) == 0:
    cprint("Error: No se ha activado ningún modelo. Por favor, activa al menos un modelo en 'MODELS_USAGE'.", 'red', 'bold')
    sys.exit(1)

# Rutas de datos y figuras
SUBJECTS_PATH = os.path.join(PROJECT_ROOT, "data", "subjects")
cprint(f"Ruta de sujetos: {SUBJECTS_PATH}", 'yellow', 'bold')

# Crear directorio para modelos según el framework
MODELS_SAVE_DIR = os.path.join(PROJECT_ROOT, CONST_RESULTS_DIR, CONST_MODELS_DIR, FRAMEWORK)
os.makedirs(MODELS_SAVE_DIR, exist_ok=True)
cprint(f"Ruta para guardar modelos: {MODELS_SAVE_DIR}", 'yellow', 'bold')

# Crear directorio para resultados según el framework
RESULTS_SAVE_DIR = os.path.join(PROJECT_ROOT, CONST_RESULTS_DIR, FRAMEWORK)
os.makedirs(RESULTS_SAVE_DIR, exist_ok=True)
cprint(f"Ruta para guardar resultados: {RESULTS_SAVE_DIR}", 'yellow', 'bold')

# Crear directorios para figuras
FIGURES_DIR = os.path.join(PROJECT_ROOT, "figures", "various_models", FRAMEWORK)
os.makedirs(FIGURES_DIR, exist_ok=True)
cprint(f"Ruta de figuras: {FIGURES_DIR}", 'yellow', 'bold')

subject_files = [f for f in os.listdir(SUBJECTS_PATH) if f.startswith("Subject") and f.endswith(".xlsx")]
cprint(f"Total sujetos: {len(subject_files)}", 'yellow', 'bold')

# Procesamiento de datos
(x_cgm_train, x_cgm_val, x_cgm_test, x_other_train, x_other_val, x_other_test, 
 x_subject_train, x_subject_val, x_subject_test, y_train, y_val, y_test, 
 x_subject_test, scaler_cgm, scaler_other, scaler_y) = (None, None, None, None, None, None, 
                                                        None, None, None, None, None, None, 
                                                        None, None, None, None)

if PROCESSING == "pandas":
    cprint("Procesando datos con pandas...", 'blue', 'bold')
    df_pd: pd.DataFrame = pd_preprocess()
    # No usamos pd_split, trabajamos con todo el dataset
    
    # Extraer características
    x_cgm = np.stack(df_pd['cgm_window'].to_numpy())
    x_other = df_pd.drop(['cgm_window', 'bolus'], axis=1).to_numpy()
    y = df_pd['bolus'].to_numpy()
    
    # Normalizar datos si es necesario
    scaler_cgm = StandardScaler().fit(x_cgm.reshape(x_cgm.shape[0], -1))
    scaler_other = StandardScaler().fit(x_other)
    scaler_y = StandardScaler().fit(y.reshape(-1, 1))
    
    x_cgm = scaler_cgm.transform(x_cgm.reshape(x_cgm.shape[0], -1)).reshape(x_cgm.shape)
    x_other = scaler_other.transform(x_other)
    y = y.reshape(-1, 1).flatten()
elif PROCESSING == "polars":
    cprint("Procesando datos con polars...", 'blue', 'bold')
    df_pl: pl.DataFrame = pl_preprocess()
    
    print(df_pl.head())
    print(df_pl.columns)
    
    # Verificar si existen columnas CGM secuenciales
    cgm_cols = [col for col in df_pl.columns if col.startswith('cgm_') and col[4:].isdigit()]
    cgm_cols.sort(key=lambda x: int(x.split('_')[1]))  # Ordenar numéricamente
    
    if not cgm_cols:
        cprint("Advertencia: No se encontraron columnas CGM secuenciales (cgm_0, cgm_1, etc.)", 'yellow', 'bold')
        cprint("Generando datos CGM a partir de columnas de glucosa disponibles...", 'blue')
        
        # Utilizar columnas de glucosa disponibles para crear datos de CGM
        glucose_cols = [col for col in df_pl.columns if 'glucose' in col.lower() and df_pl[col].dtype in 
                       [pl.Float64, pl.Float32, pl.Int64, pl.Int32, pl.Int16, pl.Int8]]
        
        if not glucose_cols:
            cprint("Error: No se encontraron columnas de glucosa utilizables", 'red', 'bold')
            sys.exit(1)
            
        cprint(f"Usando columnas de glucosa: {', '.join(glucose_cols)}", 'blue')
        
        # Crear un array de CGM sintético a partir de las columnas de glucosa disponibles
        glucose_values = df_pl.select(glucose_cols).to_numpy()
        # Asegurar que tenemos al menos 12 puntos (replicando si es necesario)
        if glucose_values.shape[1] < 12:
            repeats = int(np.ceil(12 / glucose_values.shape[1]))
            glucose_values = np.tile(glucose_values, (1, repeats))
            glucose_values = glucose_values[:, :12]  # Limitar a 12 puntos
            
        # Reshape para obtener la forma [muestras, pasos_tiempo, 1]
        x_cgm = glucose_values.reshape(glucose_values.shape[0], glucose_values.shape[1], 1)
    else:
        # Proceder con columnas CGM secuenciales como antes
        cprint(f"Usando {len(cgm_cols)} columnas CGM secuenciales", 'blue')
        cgm_values = df_pl.select(cgm_cols).to_numpy()
        x_cgm = cgm_values.reshape(cgm_values.shape[0], len(cgm_cols), 1)
    
    # Extraer columnas para features adicionales, SOLO NUMÉRICAS
    exclude_cols = cgm_cols + ['bolus']
    
    # Identificar columnas de fecha/hora y strings para excluir
    datetime_cols = [col for col in df_pl.columns if df_pl[col].dtype == pl.Datetime]
    string_cols = [col for col in df_pl.columns if df_pl[col].dtype == pl.Utf8]
    
    # Agregar todas las columnas no numéricas a la lista de exclusión
    exclude_cols += datetime_cols + string_cols
    
    # Seleccionar solo columnas numéricas para features
    feature_cols = [col for col in df_pl.columns 
                   if col not in exclude_cols 
                   and col != 'bolus' 
                   and df_pl[col].dtype in [pl.Float64, pl.Float32, pl.Int64, pl.Int32, pl.Int16, pl.Int8]]
    
    # Verificar que tenemos columnas para usar
    if not feature_cols:
        cprint("Error: No se encontraron columnas numéricas para features", 'red', 'bold')
        sys.exit(1)
    
    cprint(f"Usando {len(feature_cols)} columnas numéricas como features", 'blue')
    
    x_other = df_pl.select(feature_cols).to_numpy()
    y = df_pl['bolus'].to_numpy()
    
    # Normalizar datos
    scaler_cgm = StandardScaler().fit(x_cgm.reshape(x_cgm.shape[0], -1))
    scaler_other = StandardScaler().fit(x_other)
    scaler_y = StandardScaler().fit(y.reshape(-1, 1))
    
    x_cgm = scaler_cgm.transform(x_cgm.reshape(x_cgm.shape[0], -1)).reshape(x_cgm.shape)
    x_other = scaler_other.transform(x_other)
    y = y.reshape(-1, 1).flatten()

# Mostrar información sobre los datos
cprint("\n==== INFORMACIÓN DE DATOS ====", 'cyan', 'bold')
print(f"x_cgm: {x_cgm.shape}")
print(f"x_other: {x_other.shape}")
print(f"y: {y.shape}")

# Crear conjuntos de validación (10% de los datos)
# Esto mantiene algo de validación para monitoreo del entrenamiento
val_size = int(0.1 * len(y))
indices = np.random.permutation(len(y))
val_indices = indices[:val_size]
train_indices = indices[val_size:]

x_cgm_train, x_cgm_val = x_cgm[train_indices], x_cgm[val_indices]
x_other_train, x_other_val = x_other[train_indices], x_other[val_indices]
y_train, y_val = y[train_indices], y[val_indices]

# Para test, usamos también datos de validación (temporalmente)
x_cgm_test, x_other_test, y_test = x_cgm_val, x_other_val, y_val

# Mejorar características utilizando la función del framework seleccionado
cprint("\n==== GENERACIÓN DE CARACTERÍSTICAS ADICIONALES ====", 'cyan', 'bold')
x_cgm_train_enhanced, x_other_train_enhanced = enhance_features(x_cgm_train, x_other_train)
x_cgm_val_enhanced, x_other_val_enhanced = enhance_features(x_cgm_val, x_other_val)
x_cgm_test_enhanced, x_other_test_enhanced = enhance_features(x_cgm_test, x_other_test)

cprint(f"Forma de datos mejorados - CGM: {x_cgm_train_enhanced.shape}, Otros: {x_other_train_enhanced.shape}", 'green')

# Definir formas de entrada para los modelos
input_shapes = (x_cgm_train_enhanced.shape[1:], x_other_train_enhanced.shape[1:])
cprint(f"Formas de entrada para los modelos: CGM {input_shapes[0]}, Otros {input_shapes[1]}", 'green')

# Entrenamiento de modelos
cprint("\n==== ENTRENAMIENTO DE MODELOS ====", 'cyan', 'bold')

histories, predictions, clinical_metrics, trained_models = train_multiple_models(
    model_creators=use_models,
    input_shapes=input_shapes,
    x_cgm_train=x_cgm_train_enhanced,
    x_other_train=x_other_train_enhanced,
    y_train=y_train,
    x_cgm_val=x_cgm_val_enhanced,
    x_other_val=x_other_val_enhanced,
    y_val=y_val,
    x_cgm_test=x_cgm_test_enhanced,
    x_other_test=x_other_test_enhanced,
    y_test=y_test,
    models_dir=MODELS_SAVE_DIR
)

# Creación del Ensamble
cprint("\n==== CREACIÓN DEL ENSAMBLE ====", 'cyan', 'bold')
ensemble_prediction = None
ensemble_metrics = None
clinical_results = {}

# Inicializar simulador para métricas clínicas
simulator = GlucoseSimulator()

# Extraer valores iniciales de glucosa y carbohidratos del conjunto de prueba
initial_glucose = np.array([x_cgm_test_enhanced[i, -1, 0] for i in range(len(x_cgm_test_enhanced))])

# Manera más robusta de identificar la columna de carbohidratos
carb_intake_idx = 0  # Valor por defecto en caso de no encontrarla

# Si x_other_test es un DataFrame con columnas nombradas
if hasattr(x_other_test, 'columns'):
    columns = x_other_test.columns
    carb_columns = [i for i, col in enumerate(columns) if 'carb' in str(col).lower()]
    if carb_columns:
        carb_intake_idx = carb_columns[0]
# Si no tiene columnas nombradas, asumimos que es la primera columna de x_other_test_enhanced
elif x_other_test_enhanced.shape[1] > 0:
    # Usamos la primera columna como fallback
    carb_intake_idx = 0
    print_warning("No se encontraron nombres de columnas. Asumiendo que los carbohidratos están en la primera columna.")

carb_intake = np.array([x_other_test_enhanced[i, carb_intake_idx] for i in range(len(x_other_test_enhanced))])

# Evaluar métricas clínicas para cada modelo
for model_name, model_pred in predictions.items():
    clinical_metrics = evaluate_clinical_metrics(
        simulator=simulator,
        predictions=model_pred,
        initial_glucose=initial_glucose,
        carb_intake=carb_intake
    )
    clinical_results[model_name] = clinical_metrics
    
    cprint(f"\nMétricas clínicas para {model_name}:", 'green', 'bold')
    cprint(f"  Tiempo Severamente Bajo Rango: {clinical_metrics['time_severe_below']:.2f}%", 'red')
    cprint(f"  Tiempo Bajo Rango: {clinical_metrics['time_below_range']:.2f}%", 'yellow')
    cprint(f"  Tiempo en Rango: {clinical_metrics['time_in_range']:.2f}%", 'green')
    cprint(f"  Tiempo Sobre Rango: {clinical_metrics['time_above_range']:.2f}%", 'yellow')
    cprint(f"  Tiempo Severamente Sobre Rango: {clinical_metrics['time_severe_above']:.2f}%", 'red')
    
    # Guardar métricas clínicas
    with open(os.path.join(RESULTS_SAVE_DIR, f"{model_name}_clinical_metrics.json"), 'w') as f:
        json.dump(clinical_metrics, f, indent=2)

# Crear ensamble si hay más de un modelo
if len(predictions) > 1:
    cprint("\nCreando ensamble optimizado para métricas clínicas...", 'blue', 'bold')
    
    # Optimizar pesos para tiempo en rango
    weights, ensemble_prediction = optimize_ensemble_weights_clinical(
        predictions=predictions,
        initial_glucose=initial_glucose,
        carb_intake=carb_intake,
        simulator=simulator,
        y_true=y_test
    )
    
    # Calcular métricas para el ensamble
    ensemble_metrics = calculate_metrics(y_test, ensemble_prediction)
    ensemble_clinical = evaluate_clinical_metrics(
        simulator=simulator,
        predictions=ensemble_prediction,
        initial_glucose=initial_glucose,
        carb_intake=carb_intake
    )
    
    # Mostrar pesos y métricas del ensamble
    cprint("\nPesos del ensamble:", 'green', 'bold')
    for i, (model_name, weight) in enumerate(zip(predictions.keys(), weights)):
        cprint(f"  {model_name}: {weight:.4f}", 'green')
    
    cprint("\nMétricas clínicas del ensamble:", 'green', 'bold')
    cprint(f"  Tiempo en Rango: {ensemble_clinical['time_in_range']:.2f}%", 'green')
    cprint(f"  Tiempo Bajo Rango: {ensemble_clinical['time_below_range']:.2f}%", 'yellow')
    cprint(f"  Tiempo Sobre Rango: {ensemble_clinical['time_above_range']:.2f}%", 'yellow')
    
    # Guardar predicciones y métricas del ensamble
    clinical_results[CONST_ENSEMBLE] = ensemble_clinical
    np.save(os.path.join(RESULTS_SAVE_DIR, f"{CONST_ENSEMBLE}_predictions.npy"), ensemble_prediction)
    with open(os.path.join(RESULTS_SAVE_DIR, f"{CONST_ENSEMBLE}_metrics.json"), 'w') as f:
        json.dump({**ensemble_metrics, **ensemble_clinical}, f, indent=2)
else:
    cprint("No se puede crear ensamble con menos de 2 modelos", 'yellow', 'bold')
    
# Evaluación con FQE y Doubly Robust
cprint("\n==== EVALUACIÓN OFFLINE RL ====", 'cyan', 'bold')

# Seleccionar evaluadores activados
active_evaluators = {}
for eval_name, use in EVALUATE_USAGE.items():
    if use:
        evaluator_fn = EVALUATE[eval_name]
        if is_model_creator(evaluator_fn):
            evaluator_fn = evaluator_fn()
        active_evaluators[eval_name] = evaluator_fn
        cprint(f"Evaluador {eval_name} activado", 'green', 'bold')
    else:
        cprint(f"Evaluador {eval_name} desactivado", 'red', 'bold')

# Realizar evaluación si hay evaluadores activos
if active_evaluators:
    offline_results = {}
    
    # Evaluar cada modelo con cada evaluador activo
    for model_name, model in trained_models.items():
        cprint(f"\nEvaluando modelo {model_name}...", 'blue')
        model_results = {}
        
        for eval_name, evaluator_creator in active_evaluators.items():
            cprint(f"  Aplicando evaluador {eval_name}...", 'blue')
            eval_instance = evaluator_creator(input_shapes[0], input_shapes[1])
            
            # Entrenar evaluador con los datos de entrenamiento
            eval_instance.fit(
                x_cgm_train_enhanced, 
                x_other_train_enhanced, 
                y_train,
                validation_data=((x_cgm_val_enhanced, x_other_val_enhanced), y_val),
                epochs=50,
                batch_size=64
            )
            
            # Evaluar política del modelo usando directamente el modelo entrenado
            eval_metrics = eval_instance.evaluate_policy(
                policy=model,
                x_cgm_test=x_cgm_test_enhanced,
                x_other_test=x_other_test_enhanced,
                y_test=y_test,
                simulator=simulator
            )
            
            model_results[eval_name] = eval_metrics
            
            # Mostrar métricas principales
            cprint(f"    Valor Estimado: {eval_metrics.get('estimated_value', 0):.4f}", 'green')
            cprint(f"    Límite Inferior de Confianza: {eval_metrics.get('confidence_lower', 0):.4f}", 'yellow')
            cprint(f"    Límite Superior de Confianza: {eval_metrics.get('confidence_upper', 0):.4f}", 'yellow')
        
        offline_results[model_name] = model_results
        
        # Guardar resultados
        with open(os.path.join(RESULTS_SAVE_DIR, f"{model_name}_offline_eval.json"), 'w') as f:
            json.dump(model_results, f, indent=2)
    
    # Evaluar ensamble si existe
    if ensemble_prediction is not None:
        cprint(f"\nEvaluando ensamble...", 'blue')
        ensemble_offline_results = {}
        
        # Crear un wrapper temporal para el ensamble
        ensemble_wrapper = type('EnsembleWrapper', (), {
            'predict': lambda x_cgm, x_other: ensemble_prediction
        })()
        
        for eval_name, evaluator in active_evaluators.items():
            eval_instance = evaluator(input_shapes[0], input_shapes[1])
            eval_instance.fit(
                x_cgm_train_enhanced, x_other_train_enhanced, y_train,
                validation_data=((x_cgm_val_enhanced, x_other_val_enhanced), y_val)
            )
            
            eval_metrics = eval_instance.evaluate_policy(
                ensemble_wrapper, 
                x_cgm_test_enhanced, 
                x_other_test_enhanced, 
                y_test,
                simulator=simulator
            )
            
            ensemble_offline_results[eval_name] = eval_metrics
        
        offline_results[CONST_ENSEMBLE] = ensemble_offline_results
        
        # Guardar resultados
        with open(os.path.join(RESULTS_SAVE_DIR, f"{CONST_ENSEMBLE}_offline_eval.json"), 'w') as f:
            json.dump(ensemble_offline_results, f, indent=2)
else:
    cprint("No hay evaluadores offline activos", 'yellow', 'bold')

# Visualización de resultados
cprint("\n==== VISUALIZACIÓN DE RESULTADOS ====", 'cyan', 'bold')

# 1. Visualizar historial de entrenamiento para cada modelo
for model_name, history in histories.items():
    if not isinstance(history, dict) or not history:
        continue
        
    plt.figure(figsize=(12, 8))
    
    # Gráfico de pérdida
    plt.subplot(2, 2, 1)
    if 'loss' in history:
        plt.plot(history['loss'], label='Training Loss')
    if 'val_loss' in history:
        plt.plot(history['val_loss'], label='Validation Loss')
    plt.title(f'{model_name} - Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    
    # Gráfico de error absoluto medio
    plt.subplot(2, 2, 2)
    if 'mae' in history:
        plt.plot(history['mae'], label='Training MAE')
    if 'val_mae' in history:
        plt.plot(history['val_mae'], label='Validation MAE')
    plt.title(f'{model_name} - MAE')
    plt.xlabel('Epochs')
    plt.ylabel('MAE')
    plt.legend()
    plt.grid(True)
    
    # Gráfico de pérdidas específicas para DRL (actor, critic)
    plt.subplot(2, 2, 3)
    for metric in ['actor_loss', 'critic_loss', 'q_loss', 'policy_loss', 'value_loss']:
        if metric in history and history[metric]:
            plt.plot(history[metric], label=metric.replace('_', ' ').title())
    plt.title(f'{model_name} - DRL Losses')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    
    # Gráfico de recompensas (si es aplicable)
    plt.subplot(2, 2, 4)
    for metric in ['reward', 'average_reward', 'episode_reward']:
        if metric in history and history[metric]:
            plt.plot(history[metric], label=metric.replace('_', ' ').title())
    plt.title(f'{model_name} - Rewards')
    plt.xlabel('Epochs')
    plt.ylabel('Reward')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, f'{model_name}_training.png'))
    plt.close()

# 2. Visualización de métricas clínicas
if clinical_results:
    plt.figure(figsize=(14, 8))
    
    # Gráfico de tiempo en rango
    models = list(clinical_results.keys())
    tir_values = [clinical_results[m]['time_in_range'] for m in models]
    tbr_values = [clinical_results[m]['time_below_range'] for m in models]
    tsb_values = [clinical_results[m]['time_severe_below'] for m in models]  # Nuevo
    tar_values = [clinical_results[m]['time_above_range'] for m in models]
    tsa_values = [clinical_results[m]['time_severe_above'] for m in models]  # Nuevo

    # Barras apiladas
    plt.subplot(1, 2, 1)
    bars_tsb = plt.bar(models, tsb_values, label='Hipoglucemia Severa (<54 mg/dL)', color='darkred')
    bars_tbr = plt.bar(models, tbr_values, bottom=tsb_values, label='Hipoglucemia (54-70 mg/dL)', color='red')

    # Posición para TIR
    tsb_tbr = [tsb + tbr for tsb, tbr in zip(tsb_values, tbr_values)]
    bars_tir = plt.bar(models, tir_values, bottom=tsb_tbr, label='Tiempo en Rango (70-180 mg/dL)', color='green')

    # Posición para TAR
    tsb_tbr_tir = [tsb + tbr + tir for tsb, tbr, tir in zip(tsb_values, tbr_values, tir_values)]
    bars_tar = plt.bar(models, tar_values, bottom=tsb_tbr_tir, label='Hiperglucemia (180-250 mg/dL)', color='orange')

    # Posición para TSA
    tsb_tbr_tir_tar = [tsb + tbr + tir + tar for tsb, tbr, tir, tar in zip(tsb_values, tbr_values, tir_values, tar_values)]
    bars_tsa = plt.bar(models, tsa_values, bottom=tsb_tbr_tir_tar, label='Hiperglucemia Severa (>250 mg/dL)', color='darkred')
    
    plt.title('Distribución de Métricas Clínicas')
    plt.xlabel('Modelo')
    plt.ylabel('Porcentaje (%)')
    plt.legend()
    plt.ylim(0, 100)
    plt.xticks(rotation=45)
    plt.grid(True, axis='y')
    
    # Comparación de tiempo en rango
    plt.subplot(1, 2, 2)
    plt.bar(models, tir_values, color='green')
    plt.axhline(y=70, color='r', linestyle='--', label='Objetivo (70%)')
    plt.title('Tiempo en Rango por Modelo')
    plt.xlabel('Modelo')
    plt.ylabel('Tiempo en Rango (%)')
    plt.xticks(rotation=45)
    plt.ylim(0, 100)
    plt.legend()
    plt.grid(True, axis='y')
    
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'clinical_metrics.png'))
    plt.close()

# 3. Visualización de evaluaciones offline (si existen)
if 'offline_results' in locals() and offline_results:
    evaluators = list(next(iter(offline_results.values())).keys())
    
    for evaluator in evaluators:
        plt.figure(figsize=(10, 6))
        
        models = list(offline_results.keys())
        values = [offline_results[m][evaluator].get('estimated_value', 0) for m in models]
        lower = [offline_results[m][evaluator].get('confidence_lower', 0) for m in models]
        upper = [offline_results[m][evaluator].get('confidence_upper', 0) for m in models]
        
        plt.bar(models, values, color='skyblue')
        plt.errorbar(models, values, yerr=[
            [v - l for v, l in zip(values, lower)],
            [u - v for v, u in zip(upper, values)]
        ], fmt='o', color='black', capsize=5)
        
        plt.title(f'Evaluación {evaluator}')
        plt.xlabel('Modelo')
        plt.ylabel('Valor Estimado')
        plt.xticks(rotation=45)
        plt.grid(True, axis='y')
        plt.tight_layout()
        
        plt.savefig(os.path.join(FIGURES_DIR, f'{evaluator}_evaluation.png'))
        plt.close()

# Generación de Reporte
cprint("\n==== GENERACIÓN DE REPORTE ====", 'cyan', 'bold')


# Finalización del proceso

cprint("\n==== PROCESO COMPLETADO ====", 'cyan', 'bold')
cprint(f"Resultados guardados en: {RESULTS_SAVE_DIR}", 'green')
cprint(f"Visualizaciones guardadas en: {FIGURES_DIR}", 'green')