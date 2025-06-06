from datetime import timedelta
from typing import Dict, Union
# Modelos
## PyTorch
### Modelos de Aprendizaje por Refuerzo Profundo

# Evaluadores
from validation.fqe import create_fqe_evaluator
from validation.dre import create_dre_evaluator

# Modo de Ejecución
DEBUG = False
FRAMEWORK_OP = 2
PROCESSING_OP = 1

# Configuración de procesamiento
## Framework a utilizar durante la ejecución. Puede ser con TensorFlow o JAX.
## Opciones: "tensorflow", "jax", "pytorch"
FRAMEWORK_OPTIONS = ["tensorflow", "jax", "pytorch"]
# FRAMEWORK = FRAMEWORK_OPTIONS[FRAMEWORK_OP] 
# Se corrigen solo los modelos DRL de PyTorch
FRAMEWORK = "pytorch" 
## Procesamiento de datos. Puede ser con pandas o polars.
## Opciones: "pandas", "polars"
PROCESSING_OPTIONS = ["pandas", "polars"]
PROCESSING = PROCESSING_OPTIONS[PROCESSING_OP]

# Configuración de Procesamiento
CONFIG_PROCESSING: dict[str, Union[int, float, str]] = {
    "batch_size": 128,
    "window_hours": 2,
    "window_steps": 24,  # 5-min steps in 2 hours
    "window_size": 12,
    "extended_window_size": 288,
    "insulin_lifetime_hours": 4.0,
    "cap_normal": 30,
    "cap_bg": 300,
    "cap_iob": 5,
    "cap_carb": 150,
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
    "max_work_intensity": 10,
    "max_sleep_quality": 10,
    "max_activity_intensity": 10,
    "low_dose_threshold": 7.0,
    "min_cgm_points": 12,
    "alignment_tolerance_minutes": 15,
    "random_seed": 42,
    "carb_effect_factor": 5,
    "insulin_effect_factor": 50,
    "glucose_min": 40,
    "glucose_max": 400,
    "bolus_max": 20.0,
    "bolus_min": 0.1,
    "partial_window_threshold": 6,
    "event_tolerance": timedelta(minutes=15),
    "basal_estimation_hours": (0, 24),
    "basal_estimation_factor": 0.5,
    "hypoglycemia_threshold": 70,
    "hyperglycemia_threshold": 180,
    "tir_lower": 70,
    "tir_upper": 180,
    "simulation_steps": 72,
    "hypo_risk_threshold": 70,
    "hyper_risk_threshold": 180,
    "significant_meal_threshold": 20,
}

## Modelos PyTorch disponibles.
MODELS = {
    ## Modelos de Aprendizaje por Refuerzo Profundo
    "pt_ddpg": "",
    "pt_dqn": "",
    "pt_sac": "",
    "pt_td3_bc": "",
}

# Modelos Pytorch a utilizar
MODELS_USAGE: Dict[str, bool] = {
    ## Modelos de Aprendizaje por Refuerzo Profundo
    "pt_ddpg": True,
    "pt_dqn": True,
    "pt_sac": True,
    "pt_td3_bc": True,
}

EVALUATE = {
    "pt_fqe": create_fqe_evaluator,
    "pt_dr": create_dre_evaluator
}

EVALUATE_USAGE: Dict[str, bool] = {
    "pt_fqe": True,
    "pt_dr": True
}