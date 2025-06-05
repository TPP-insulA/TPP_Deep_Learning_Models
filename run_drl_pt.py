# Imports
import os
import sys
import pandas as pd
import polars as pl
import numpy as np
import matplotlib.pyplot as plt
import torch
from typing import Dict, List, Tuple, Any
from validation.model_validation import validate_dosing_model
from validation.simulator import GlucoseSimulator
from constants.constants import LOWER_BOUND_NORMAL_GLUCOSE_RANGE, UPPER_BOUND_NORMAL_GLUCOSE_RANGE
from training.pytorch import (
    train_multiple_models, calculate_metrics, create_ensemble_prediction, 
    optimize_ensemble_weights, enhance_features
)



PROJECT_ROOT = os.path.abspath(os.getcwd())
sys.path.append(PROJECT_ROOT)

# Printer
from custom.printer import cprint, coloured

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
            # Si no tiene parámetros, es probable que sea un model creator
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

# Creación del Ensamble

# Visualización de resultados

# Visualización de Métricas Clínicas

# Generación de Reporte

cprint("\n==== PROCESO COMPLETADO ====", 'cyan', 'bold')
cprint(f"Resultados guardados en: {RESULTS_SAVE_DIR}", 'green')
cprint(f"Visualizaciones guardadas en: {FIGURES_DIR}", 'green')