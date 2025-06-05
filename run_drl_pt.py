# Imports
import os
import sys
import pandas as pd
import polars as pl
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Any
from validation.model_validation import validate_dosing_model
from validation.simulator import GlucoseSimulator
from constants.constants import LOWER_BOUND_NORMAL_GLUCOSE_RANGE, UPPER_BOUND_NORMAL_GLUCOSE_RANGE


PROJECT_ROOT = os.path.abspath(os.getcwd())
sys.path.append(PROJECT_ROOT)

# Printer
from custom.printer import cprint, coloured

# Configuración 
from config.params import FRAMEWORK, PROCESSING, USE_TF_MODELS, USE_JAX_MODELS, USE_PT_MODELS, TF_MODELS, JAX_MODELS, PT_MODELS

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
            # Si no tiene parámetros, es probable que sea un model creator
            # que debe llamarse para obtener la función de creación real
            return len(sig.parameters) == 0
        except Exception:
            pass
    return False

# Importación dinámica de módulos de entrenamiento según el framework seleccionado
cprint(f"Framework seleccionado: {FRAMEWORK}", 'blue', 'bold')

if FRAMEWORK == "tensorflow":
    from training.tensorflow import (
        train_multiple_models, calculate_metrics, create_ensemble_prediction, 
        optimize_ensemble_weights, enhance_features
    )
    
    # Configurar TensorFlow para uso eficiente de memoria GPU
    import tensorflow as tf
    gpus = tf.config.list_physical_devices('GPU')
    if (gpus):
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            cprint(f"GPU disponible: {len(gpus)} dispositivos", 'green')
        except RuntimeError as e:
            cprint(f"Error al configurar GPU: {e}", 'red')
    else:
        cprint("No se detectaron GPUs, usando CPU", 'yellow')
        
    cprint("Usando backend de TensorFlow para entrenamiento", 'green', 'bold')
        
elif FRAMEWORK == "jax":
    from training.jax import (
        train_multiple_models, calculate_metrics, create_ensemble_prediction, 
        optimize_ensemble_weights, enhance_features
    )
    
    # Configuración específica para JAX
    import jax
    cprint(f"Dispositivos JAX disponibles: {jax.devices()}", 'green')
    cprint("Usando backend de JAX para entrenamiento", 'green', 'bold')
elif FRAMEWORK == "pytorch":
    from training.pytorch import (
        train_multiple_models, calculate_metrics, create_ensemble_prediction, 
        optimize_ensemble_weights, enhance_features
    )
    
    # Configure PyTorch
    import torch
    if torch.cuda.is_available():
        cprint(f"GPU available: {torch.cuda.device_count()} devices", 'green')
        cprint(f"Using: {torch.cuda.get_device_name(0)}", 'green')
    else:
        cprint("No GPUs detected, using CPU", 'yellow')
        
    cprint("Using PyTorch backend for training", 'green', 'bold')
else:
    cprint(f"Error: Framework not supported: {FRAMEWORK}. Must be 'tensorflow', 'jax', or 'pytorch'.", 'red', 'bold')
    sys.exit(1)

# Constante para mensaje repetido
CONST_MODEL_ACTIVATED = "Modelo {} activado."
CONST_MODEL_DEACTIVATED = "Modelo {} desactivado."

use_models = {}
if FRAMEWORK == "tensorflow":
    MODELS_TO_USE = USE_TF_MODELS
    AVAILABLE_MODELS = TF_MODELS
elif FRAMEWORK == "jax":
    MODELS_TO_USE = USE_JAX_MODELS
    AVAILABLE_MODELS = JAX_MODELS
elif FRAMEWORK == "pytorch":
    MODELS_TO_USE = USE_PT_MODELS
    AVAILABLE_MODELS = PT_MODELS

for model_name, use in MODELS_TO_USE.items():
    if use:
        model_fn = AVAILABLE_MODELS[model_name]
        
        # Si estamos usando JAX o PyTorch, asegurar que todos los modelos estén en formato uniforme
        if FRAMEWORK == "jax" or FRAMEWORK == "pytorch":
            # Verificar si es un model creator que necesita ser llamado
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
    cprint("Error: No se ha activado ningún modelo. Por favor, activa al menos un modelo en 'MODELS_TO_USE'.", 'red', 'bold')
    sys.exit(1)
if MODELS_TO_USE.values() == [False] * len(use_models):
    cprint("Error: Todos los modelos están desactivados. Por favor, activa al menos un modelo en 'MODELS_TO_USE'.", 'red', 'bold')
    sys.exit(1)
if len(use_models) == 0:
    cprint("Error: No se ha activado ningún modelo. Por favor, activa al menos un modelo en 'MODELS_TO_USE'.", 'red', 'bold')
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
    (x_cgm_train, x_cgm_val, x_cgm_test, x_other_train, x_other_val, x_other_test, 
     x_subject_train, x_subject_val, x_subject_test, y_train, y_val, y_test, 
     x_subject_test, scaler_cgm, scaler_other, scaler_y) = pd_split(df_pd)
elif PROCESSING == "polars":
    cprint("Procesando datos con polars...", 'blue', 'bold')
    df_pl: pl.DataFrame = pl_preprocess()
    (x_cgm_train, x_cgm_val, x_cgm_test, x_other_train, x_other_val, x_other_test, 
     x_subject_train, x_subject_val, x_subject_test, y_train, y_val, y_test, 
     x_subject_test, scaler_cgm, scaler_other, scaler_y) = pl_split(df_pl)

# Mostrar información sobre los datos
cprint("\n==== INFORMACIÓN DE DATOS ====", 'cyan', 'bold')
print(f"x_cgm_train: {x_cgm_train.shape}")
print(f"x_cgm_val: {x_cgm_val.shape}")
print(f"x_cgm_test: {x_cgm_test.shape}")
print(f"x_other_train: {x_other_train.shape}")
print(f"x_other_val: {x_other_val.shape}")
print(f"x_other_test: {x_other_test.shape}")
print(f"x_subject_train: {x_subject_train.shape}")
print(f"x_subject_val: {x_subject_val.shape}")
#print(f"x_subject_test: {x_subject_test.shape}")
print(f"{x_subject_test=}")
print(f"y_train: {y_train.shape}")
print(f"y_val: {y_val.shape}")
print(f"y_test: {y_test.shape}")

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
histories, predictions, metrics = train_multiple_models(
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

# Mostrar resultados de cada modelo
cprint("\n==== RESULTADOS DE LOS MODELOS ====", 'cyan', 'bold')
for model_name, model_metrics in metrics.items():
    mae = model_metrics["mae"]
    rmse = model_metrics["rmse"]
    r2 = model_metrics["r2"]
    cprint(f"Modelo: {model_name}", 'green', 'bold')
    print(f"  MAE:  {mae:.4f}")
    print(f"  RMSE: {rmse:.4f}")
    print(f"  R²:   {r2:.4f}")

# Crear y evaluar ensemble
cprint("\n==== CREACIÓN DE ENSEMBLE ====", 'cyan', 'bold')

# Optimizar pesos para ensemble
ensemble_weights = optimize_ensemble_weights(predictions, y_test)
ensemble_pred = create_ensemble_prediction(predictions, ensemble_weights)
ensemble_metrics = calculate_metrics(y_test, ensemble_pred)

cprint("\n==== GENERANDO VISUALIZACIONES POR MODELO ====", 'cyan', 'bold')

# Crear directorio para visualizaciones individuales por modelo
MODEL_FIGURES_DIR = os.path.join(FIGURES_DIR, "individual_models")
os.makedirs(MODEL_FIGURES_DIR, exist_ok=True)

# Visualizar resultados de cada modelo individualmente
model_figures = {}
for model_name, model_history in histories.items():
    cprint(f"Generando visualizaciones para modelo: {model_name}", 'green')
    
    # Usar la función de visualización completa por modelo
    figure_paths = visualize_model_results(
        history=model_history,
        predictions=predictions[model_name],
        y_test=y_test,
        model_name=model_name,
        save_dir=MODEL_FIGURES_DIR,
        show_plots=False  # Cambiar a True si deseas ver los gráficos durante la ejecución
    )
    
    model_figures[model_name] = figure_paths
    
    # Mostrar rutas de los archivos generados
    for fig_type, fig_path in figure_paths.items():
        print(f"  - {fig_type}: {os.path.basename(fig_path)}")

cprint("\n==== GENERANDO VISUALIZACIONES COMPLETAS ====", 'cyan', 'bold')

# Generar todas las visualizaciones de forma organizada
all_figures = plot_model_evaluation_summary(
    histories=histories,
    predictions=predictions,
    y_true=y_test,
    metrics=metrics,
    ensemble_predictions=ensemble_pred,
    ensemble_weights=ensemble_weights,
    ensemble_metrics=ensemble_metrics,
    save_dir=FIGURES_DIR,
    sample_size=100,
    show_plots=False
)

cprint(f"Visualizaciones comparativas guardadas en: {os.path.join(FIGURES_DIR, 'comparative')}", 'green')
cprint(f"Visualizaciones por modelo guardadas en: {os.path.join(FIGURES_DIR, 'individual_models')}", 'green')


# Mostrar métricas del ensemble
cprint("Métricas del ensemble:", 'green', 'bold')
print(f"  MAE:  {ensemble_metrics['mae']:.4f}")
print(f"  RMSE: {ensemble_metrics['rmse']:.4f}")
print(f"  R²:   {ensemble_metrics['r2']:.4f}")

# Mostrar pesos del ensemble
cprint("Pesos optimizados del ensemble:", 'green', 'bold')
for model_name, weight in zip(predictions.keys(), ensemble_weights):
    print(f"  {model_name}: {weight:.4f}")

# Guardar resultados
cprint("\n==== GUARDANDO RESULTADOS ====", 'cyan', 'bold')
import json
import pickle

# Guardar métricas como JSON
metrics_with_ensemble = {**metrics, CONST_ENSEMBLE: ensemble_metrics}
with open(os.path.join(RESULTS_SAVE_DIR, 'model_metrics.json'), 'w') as f:
    json.dump({model: {k: float(v) for k, v in metrics.items()} 
               for model, metrics in metrics_with_ensemble.items()}, f, indent=2)

# Guardar predicciones y pesos del ensemble
predictions_with_ensemble = {**predictions, CONST_ENSEMBLE: ensemble_pred}
with open(os.path.join(RESULTS_SAVE_DIR, 'predictions.pkl'), 'wb') as f:
    pickle.dump(predictions_with_ensemble, f)

with open(os.path.join(RESULTS_SAVE_DIR, 'ensemble_weights.pkl'), 'wb') as f:
    pickle.dump({
        'weights': ensemble_weights,
        'model_names': list(predictions.keys())
    }, f)

# Visualizaciones
cprint("\n==== GENERANDO VISUALIZACIONES ====", 'cyan', 'bold')

# Gráfico de comparación de pérdida
plt.figure(figsize=(12, 6))
for model_name, history in histories.items():
    plt.plot(history['loss'], label=f'{model_name} (Train)')
    plt.plot(history['val_loss'], label=f'{model_name} (Val)')
# Verificar si hay elementos etiquetados en la figura actual antes de agregar leyenda
if len(plt.gca().get_legend_handles_labels()[0]) > 0:
    plt.legend()
plt.title('Comparación de Pérdida Durante Entrenamiento')
plt.xlabel('Época')
plt.ylabel('Pérdida (MSE)')
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(FIGURES_DIR, 'loss_comparison.png'), dpi=300)

# Gráfico de comparación de MAE
plt.figure(figsize=(12, 6))
for model_name, history in histories.items():
    if 'mae' in history:
        plt.plot(history['mae'], label=f'{model_name} (Train)')
        plt.plot(history['val_mae'], label=f'{model_name} (Val)')
# Verificar si hay elementos etiquetados en la figura actual antes de agregar leyenda
if len(plt.gca().get_legend_handles_labels()[0]) > 0:
    plt.legend()
plt.title('Comparación de MAE Durante Entrenamiento')
plt.xlabel('Época')
plt.ylabel('MAE')
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(FIGURES_DIR, 'mae_comparison.png'), dpi=300)

# Gráfico de predicciones vs reales (muestra limitada)
plt.figure(figsize=(14, 7))
# Mostrar solo los primeros 100 puntos para claridad
sample_size = min(100, len(y_test))
indices = np.arange(sample_size)

plt.plot(indices, y_test[:sample_size], 'o-', label='Real', color='black', alpha=0.7)
for model_name, pred in predictions.items():
    plt.plot(indices, pred[:sample_size], 'o-', label=model_name, alpha=0.5)
plt.plot(indices, ensemble_pred[:sample_size], 'o-', label='Ensemble', linewidth=2)
# Verificar si hay elementos etiquetados en la figura actual antes de agregar leyenda
if len(plt.gca().get_legend_handles_labels()[0]) > 0:
    plt.legend()
plt.title('Predicciones vs Valores Reales')
plt.xlabel('Índice de Muestra')
plt.ylabel('Dosis de Insulina')
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(FIGURES_DIR, 'predictions_comparison.png'), dpi=300)

# Gráficos de barras para métricas
plt.figure(figsize=(15, 8))

# MAE
plt.subplot(1, 3, 1)
all_names = list(metrics.keys()) + [CONST_ENSEMBLE]
all_mae = [m['mae'] for m in metrics.values()] + [ensemble_metrics['mae']]
plt.bar(all_names, all_mae, color=['blue'] * len(metrics) + ['red'])
plt.title('MAE (menor es mejor)')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()

# RMSE
plt.subplot(1, 3, 2)
all_rmse = [m['rmse'] for m in metrics.values()] + [ensemble_metrics['rmse']]
plt.bar(all_names, all_rmse, color=['blue'] * len(metrics) + ['red'])
plt.title('RMSE (menor es mejor)')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()

# R²
plt.subplot(1, 3, 3)
all_r2 = [m['r2'] for m in metrics.values()] + [ensemble_metrics['r2']]
plt.bar(all_names, all_r2, color=['blue'] * len(metrics) + ['red'])
plt.title('R² (mayor es mejor)')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()

plt.savefig(os.path.join(FIGURES_DIR, 'metrics_comparison.png'), dpi=300)

# Gráfico de pesos del ensemble
plt.figure(figsize=(10, 6))
plt.bar(list(predictions.keys()), ensemble_weights, color='teal')
plt.title('Pesos Optimizados del Ensemble')
plt.xticks(rotation=45, ha='right')
plt.ylabel('Peso')
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.savefig(os.path.join(FIGURES_DIR, 'ensemble_weights.png'), dpi=300)

# Cerrar todas las figuras
plt.close('all')

# Crear directorio para resultados de validación de glucosa
VALIDATION_DIR = os.path.join(RESULTS_SAVE_DIR, "glucose_validation")
os.makedirs(VALIDATION_DIR, exist_ok=True)

cprint("\n==== VALIDACIÓN DE MODELOS CON SIMULACIÓN DE GLUCOSA ====", 'cyan', 'bold')

# Convertir datos de prueba a DataFrame de Polars para simulación
test_data_cols = {
    "subject_id": x_subject_test,  # IDs de pacientes
    "timestamp": np.arange(len(y_test)).tolist(),  # Timestamps simulados
    "glucose": x_cgm_test_enhanced[:, 0, -1].tolist(),  # Último valor de glucosa en cada secuencia
    "carb_intake": x_other_test_enhanced[:, 0].tolist(),  # Ingesta de carbohidratos (primera columna)
    "insulin_on_board": np.zeros(len(y_test)).tolist()  # Valor predeterminado si no está disponible
}

# Crear DataFrame con Polars
test_df = pl.DataFrame(test_data_cols)

# Definir parámetros específicos por paciente
patient_params = {}
for subject_id in np.unique(x_subject_test):
    # Parámetros personalizados por paciente
    patient_params[subject_id] = {
        "insulin_sensitivity": 50,  # mg/dL por unidad de insulina
        "carb_ratio": 10,           # gramos por unidad de insulina
        "basal_glucose_impact": 20, # aumento mg/dL por hora sin insulina
        "insulin_duration_hours": 4  # duración de acción de insulina
    }

# Diccionario para almacenar resultados de simulación
simulation_results = {}

# Validar cada modelo entrenado
for model_name, model_metrics in metrics.items():
    model_output_dir = os.path.join(VALIDATION_DIR, model_name)
    os.makedirs(model_output_dir, exist_ok=True)
    
    cprint(f"Validando modelo {model_name} con simulación de glucosa...", 'green')
    
    # Cargar modelo entrenado - aquí necesitas asegurarte de tener acceso a los modelos
    # Esto puede requerir que los modelos se guarden durante el entrenamiento
    try:
        # Cargar modelo desde archivo si está disponible
        # PYTORCH GUARDA MODELOS COMO .pt
        model_path = os.path.join(MODELS_SAVE_DIR, f"{model_name}.pt")
        if os.path.exists(model_path):
            with open(model_path, 'rb') as f:
                model = pickle.load(f)
        else:
            cprint(f"Advertencia: No se pudo cargar el modelo {model_name}", 'yellow')
            continue
        
        # Ejecutar validación
        results = validate_dosing_model(
            model=model,
            test_data=test_df,
            patient_params=patient_params,
            output_dir=model_output_dir,
            visualize=True
        )
        
        simulation_results[model_name] = results
        
        # Mostrar resultados principales
        time_in_range = results["global"]["time_in_range"]
        hypo_time = results["global"]["time_below_range"]
        hyper_time = results["global"]["time_above_range"]
        
        cprint(f"Resultados de simulación para {model_name}:", 'blue')
        print(f"  Tiempo en rango (70-180 mg/dL): {time_in_range:.1f}%")
        print(f"  Tiempo en hipoglucemia (<70 mg/dL): {hypo_time:.1f}%")
        print(f"  Tiempo en hiperglucemia (>180 mg/dL): {hyper_time:.1f}%")
        
    except Exception as e:
        cprint(f"Error al validar modelo {model_name}: {e}", 'red')

# Visualizar comparativa de resultados si hay datos disponibles
if simulation_results:
    plt.figure(figsize=(14, 8))
    
    # Crear listas de modelos y métricas
    models = list(simulation_results.keys())
    tir_values = [simulation_results[m]["global"]["time_in_range"] for m in models]
    hypo_values = [simulation_results[m]["global"]["time_below_range"] for m in models]
    hyper_values = [simulation_results[m]["global"]["time_above_range"] for m in models]
    
    # Crear gráfico de barras apiladas
    bar_width = 0.6
    plt.bar(models, tir_values, bar_width, label='Tiempo en Rango', color='green')
    plt.bar(models, hypo_values, bar_width, bottom=tir_values, label='Hipoglucemia', color='red')
    plt.bar(models, hyper_values, bar_width, bottom=[sum(x) for x in zip(tir_values, hypo_values)], 
           label='Hiperglucemia', color='orange')
    
    plt.xlabel('Modelo')
    plt.ylabel('Porcentaje de Tiempo (%)')
    plt.title('Distribución de Control Glucémico por Modelo')
    plt.legend()
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    
    # Guardar figura
    plt.savefig(os.path.join(FIGURES_DIR, "glucose_control_comparison.png"), dpi=300)
    plt.close()
    
    # Guardar resultados como JSON para referencia futura
    with open(os.path.join(VALIDATION_DIR, 'simulation_results.json'), 'w') as f:
        # Convertir valores numpy a Python nativos para JSON
        json_results = {}
        for model, result in simulation_results.items():
            json_results[model] = {}
            for patient, metrics in result.items():
                json_results[model][patient] = {k: float(v) for k, v in metrics.items()}
        
        json.dump(json_results, f, indent=2)

    cprint(f"Resultados de validación con simulación guardados en: {VALIDATION_DIR}", 'green')
    cprint(f"Gráfico comparativo guardado en: {os.path.join(FIGURES_DIR, 'glucose_control_comparison.png')}", 'green')


cprint("\n==== PROCESO COMPLETADO ====", 'cyan', 'bold')
cprint(f"Resultados guardados en: {RESULTS_SAVE_DIR}", 'green')
cprint(f"Visualizaciones guardadas en: {FIGURES_DIR}", 'green')

# Generar reporte Typst
cprint("\n==== GENERANDO REPORTE EN TYPST ====", 'cyan', 'bold')

# Crear reporte Typst
report_path = create_report(
    model_figures=model_figures,            # Diccionario con rutas a figuras por modelo
    ensemble_metrics=ensemble_metrics,      # Métricas del modelo ensemble
    framework=FRAMEWORK,                    # 'tensorflow', 'jax' o 'pytorch'
    project_root=PROJECT_ROOT,              # Ruta base del proyecto
    figures_dir=FIGURES_DIR,                # Directorio de figuras (constante definida)
    metrics=metrics                         # Métricas de todos los modelos
)

cprint(f"Reporte Typst generado: {report_path}", 'green')

# # Intentar renderizar el PDF
# pdf_path = render_to_pdf(report_path)
# if pdf_path:
#     cprint(f"PDF generado: {pdf_path}", 'green')

cprint("\n==== PROCESO COMPLETADO ====", 'cyan', 'bold')
cprint(f"Resultados guardados en: {RESULTS_SAVE_DIR}", 'green')
cprint(f"Visualizaciones guardadas en: {FIGURES_DIR}", 'green')
cprint(f"Reporte guardado en: {report_path}", 'green')
# if pdf_path:
#     cprint(f"PDF guardado en: {pdf_path}", 'green')