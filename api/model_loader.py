import tensorflow as tf
import jax
import torch
from typing import List

MODELS_PATH = "results/models/"  # Ruta donde se almacenan los modelos

def load_models(framework: str, models: List[str]) -> List:
    """
    Carga los modelos especificados en la lista.
    
    Parámetros:
    -----------
    framework : str
        Framework utilizado (ej. 'tensorflow', 'jax', 'pytorch').
    models : List[str]
        Lista de modelos a cargar.
        
    Retorna:
    --------
    List
        Lista de modelos cargados.
    """
    loaded_models = []
    
    for model in models:
        if framework == 'tensorflow':
            # Cargar modelo de TensorFlow
            loaded_model = tf.keras.models.load_model(f"{MODELS_PATH}/tensorflow/{model}.h5")
        # TODO: Temporalmente deshabilitado JAX porque no se está guardando bien el modelo.
        # elif framework == 'jax':
        #     # Cargar modelo de JAX
        #     loaded_model = jax.jit(jax.experimental.stax.serial(f"{MODELS_PATH}/jax/{model}.npz"))
        elif framework == 'pytorch':
            # Cargar modelo de PyTorch
            loaded_model = torch.load(f"{MODELS_PATH}/pytorch/{model}.pt")
        else:
            raise ValueError(f"Framework '{framework}' no soportado.")
        
        loaded_models.append(loaded_model)
    
    return loaded_models