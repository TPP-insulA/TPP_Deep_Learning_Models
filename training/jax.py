import os
import time
import numpy as np
import jax
import jax.numpy as jnp
from jax import random, grad, jit, value_and_grad
from flax import linen as nn
from flax.training import train_state
from flax.training.checkpoints import save_checkpoint, restore_checkpoint
import optax
from typing import Dict, List, Tuple, Callable, Optional, Any, Union
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold
from joblib import Parallel, delayed
from scipy.optimize import minimize
import orbax.checkpoint as orbax_ckpt
from custom.printer import print_debug
from training.common import (
    calculate_metrics, create_ensemble_prediction, optimize_ensemble_weights,
    enhance_features, get_model_type, process_training_results
)
from constants.constants import (
    CONST_VAL_LOSS, CONST_LOSS, CONST_METRIC_MAE, CONST_METRIC_RMSE, CONST_METRIC_R2,
    CONST_MODELS, CONST_BEST_PREFIX, CONST_LOGS_DIR, CONST_DEFAULT_EPOCHS, 
    CONST_DEFAULT_BATCH_SIZE, CONST_DEFAULT_SEED, CONST_FIGURES_DIR, CONST_MODEL_TYPES
)
from tqdm.auto import tqdm
from config.params import DEBUG

CONST_EPOCHS = 3 if DEBUG else CONST_DEFAULT_EPOCHS

def create_batched_dataset(x_cgm: np.ndarray, 
                          x_other: np.ndarray, 
                          y: np.ndarray, 
                          batch_size: int = CONST_DEFAULT_BATCH_SIZE, 
                          shuffle: bool = True, 
                          rng: Optional[jax.random.PRNGKey] = None) -> Tuple[List[Tuple[Tuple[np.ndarray, np.ndarray], np.ndarray]], int]:
    """
    Crea un dataset en batches para entrenamiento con JAX.

    Parámetros:
    -----------
    x_cgm : np.ndarray
        Datos CGM con forma (muestras, pasos_tiempo, características)
    x_other : np.ndarray
        Otras características con forma (muestras, características)
    y : np.ndarray
        Valores objetivo con forma (muestras,)
    batch_size : int, opcional
        Tamaño del batch para entrenamiento (default: 32)
    shuffle : bool, opcional
        Indica si se deben mezclar los datos (default: True)
    rng : Optional[jax.random.PRNGKey], opcional
        Clave de generación aleatoria para reproducibilidad (default: None)
        
    Retorna:
    --------
    Tuple[List[Tuple[Tuple[np.ndarray, np.ndarray], np.ndarray]], int]
        Lista de batches y cantidad de batches
    """
    # Obtener número de muestras
    n_samples = x_cgm.shape[0]
    
    # Crear índices y mezclarlos si es necesario
    indices = np.arange(n_samples)
    if shuffle and rng is not None:
        indices = jax.random.permutation(rng, indices)
    elif shuffle:
        rng_np = np.random.Generator(np.random.PCG64(CONST_DEFAULT_SEED))
        rng_np.shuffle(indices)
    
    # Calcular número de batches
    n_batches = int(np.ceil(n_samples / batch_size))
    
    # Crear lista de batches
    batches = []
    for i in range(n_batches):
        start_idx = i * batch_size
        end_idx = min(start_idx + batch_size, n_samples)
        batch_indices = indices[start_idx:end_idx]
        
        # Seleccionar datos correspondientes a los índices
        x_cgm_batch = x_cgm[batch_indices]
        x_other_batch = x_other[batch_indices]
        y_batch = y[batch_indices]
        
        batches.append(((x_cgm_batch, x_other_batch), y_batch))
    
    return batches, n_batches

@jit
def mse_loss(params: Dict, apply_fn: Callable, x_cgm: jnp.ndarray, x_other: jnp.ndarray, y: jnp.ndarray) -> jnp.ndarray:
    """
    Función de pérdida de error cuadrático medio para entrenamiento.
    
    Parámetros:
    -----------
    params : Dict
        Parámetros del modelo
    apply_fn : Callable
        Función para aplicar el modelo
    x_cgm : jnp.ndarray
        Datos CGM
    x_other : jnp.ndarray
        Otras características
    y : jnp.ndarray
        Valores objetivo
        
    Retorna:
    --------
    jnp.ndarray
        Valor de pérdida MSE
    """
    # Realizar predicción
    y_pred = apply_fn(params, x_cgm, x_other).flatten()
    
    # Calcular error cuadrático medio
    return jnp.mean(jnp.square(y_pred - y))


@jit
def train_step(state: train_state.TrainState, 
               x_cgm: jnp.ndarray, 
               x_other: jnp.ndarray, 
               y: jnp.ndarray) -> Tuple[train_state.TrainState, Dict[str, jnp.ndarray]]:
    """
    Ejecuta un paso de entrenamiento del modelo.
    
    Parámetros:
    -----------
    state : train_state.TrainState
        Estado actual del entrenamiento
    x_cgm : jnp.ndarray
        Datos CGM para el batch
    x_other : jnp.ndarray
        Otras características para el batch
    y : jnp.ndarray
        Valores objetivo para el batch
        
    Retorna:
    --------
    Tuple[train_state.TrainState, Dict[str, jnp.ndarray]]
        Nuevo estado de entrenamiento y métricas
    """
    # Generar clave PRNG para dropout
    dropout_rng = jax.random.PRNGKey(0)  # O bien usar una clave diferente para cada paso
    
    # Definir función de pérdida con manejo de PRNG
    def loss_fn(params):
        # Pasar rngs para operaciones estocásticas como dropout
        y_pred = state.apply_fn(
            params, 
            x_cgm, 
            x_other, 
            training=True,  # Modo entrenamiento
            rngs={'dropout': dropout_rng}  # Clave PRNG para dropout
        ).flatten()
        return jnp.mean(jnp.square(y_pred - y))
    
    # Calcular gradiente y pérdida
    grad_fn = value_and_grad(loss_fn, has_aux=False)
    loss, grads = grad_fn(state.params)
    
    # Aplicar gradientes y actualizar estado
    new_state = state.apply_gradients(grads=grads)
    
    # Calcular métricas adicionales
    metrics = {
        CONST_LOSS: loss,
    }
    
    return new_state, metrics


@jit
def eval_step(state: train_state.TrainState, 
              x_cgm: jnp.ndarray, 
              x_other: jnp.ndarray, 
              y: jnp.ndarray) -> Dict[str, jnp.ndarray]:
    """
    Evalúa el modelo en datos de validación.
    
    Parámetros:
    -----------
    state : train_state.TrainState
        Estado actual del entrenamiento
    x_cgm : jnp.ndarray
        Datos CGM para validación
    x_other : jnp.ndarray
        Otras características para validación
    y : jnp.ndarray
        Valores objetivo para validación
        
    Retorna:
    --------
    Dict[str, jnp.ndarray]
        Métricas de evaluación
    """
    # Calcular pérdida
    loss = mse_loss(state.params, state.apply_fn, x_cgm, x_other, y)
    
    # Calcular predicciones
    y_pred = state.apply_fn(state.params, x_cgm, x_other).flatten()
    
    # Calcular error absoluto medio
    mae = jnp.mean(jnp.abs(y_pred - y))
    
    # Calcular raíz del error cuadrático medio
    rmse = jnp.sqrt(jnp.mean(jnp.square(y_pred - y)))
    
    return {
        CONST_LOSS: loss,
        CONST_METRIC_MAE: mae,
        CONST_METRIC_RMSE: rmse
    }


def _setup_training(model: nn.Module,
                   training_config: Dict[str, Any],
                   x_cgm_train: np.ndarray,
                   x_other_train: np.ndarray,
                   models_dir: str) -> Tuple[train_state.TrainState, Dict, jax.random.PRNGKey]:
    """
    Prepara el entorno de entrenamiento inicializando el modelo y optimizador.
    """
    # Extraer parámetros de configuración
    learning_rate = training_config.get('learning_rate', 0.001)
    seed = training_config.get('seed', CONST_DEFAULT_SEED)
    
    # Crear directorio para modelos si no existe
    os.makedirs(models_dir, exist_ok=True)
    
    # Inicializar generador de números aleatorios
    rng = random.PRNGKey(seed)
    rng, init_rng = random.split(rng)
    
    # Inicializar modelo
    x_cgm_shape = (1,) + x_cgm_train.shape[1:]
    x_other_shape = (1,) + x_other_train.shape[1:]
    
    params = model.init(init_rng, jnp.ones(x_cgm_shape), jnp.ones(x_other_shape))
    
    # Configurar tasa de aprendizaje con decaimiento
    schedule_fn = optax.exponential_decay(
        init_value=learning_rate,
        transition_steps=1000,
        decay_rate=0.9
    )
    
    # Configurar optimizador con recorte de gradiente
    tx = optax.chain(
        optax.clip_by_global_norm(1.0),  # Recorte de gradiente
        optax.adam(learning_rate=schedule_fn)
    )
    
    # Crear estado de entrenamiento
    state = train_state.TrainState.create(
        apply_fn=model.apply,
        params=params,
        tx=tx
    )
    
    return state, params, rng

def _prepare_data(x_cgm_train: np.ndarray, 
                x_other_train: np.ndarray, 
                y_train: np.ndarray,
                x_cgm_val: np.ndarray,
                x_other_val: np.ndarray,
                y_val: np.ndarray,
                batch_size: int,
                rng: jax.random.PRNGKey) -> Tuple[List, int, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """
    Prepara los datos para entrenamiento y validación.
    """
    # Crear conjuntos de datos en batches
    train_batches, n_train_batches = create_batched_dataset(
        x_cgm_train, x_other_train, y_train, batch_size=batch_size, shuffle=True, rng=rng
    )
    
    # Convertir datos de validación para evaluación
    x_cgm_val_array = jnp.array(x_cgm_val)
    x_other_val_array = jnp.array(x_other_val)
    y_val_array = jnp.array(y_val)
    
    return train_batches, n_train_batches, x_cgm_val_array, x_other_val_array, y_val_array

def _train_epoch(state: train_state.TrainState,
               train_batches: List,
               n_train_batches: int,
               x_cgm_val_array: jnp.ndarray,
               x_other_val_array: jnp.ndarray,
               y_val_array: jnp.ndarray) -> Tuple[float, Dict, train_state.TrainState]:
    """
    Entrena una época completa y evalúa en validación.
    """
    # Variables para métricas de época
    epoch_loss = 0.0
    
    # Usar tqdm para mostrar progreso dentro de la época
    for i, batch in enumerate(tqdm(train_batches, desc="Batches", leave=False)):
        # Desempaquetar batch
        (x_cgm_batch, x_other_batch), y_batch = batch
        
        # Convertir a arrays de JAX
        x_cgm_batch = jnp.array(x_cgm_batch)
        x_other_batch = jnp.array(x_other_batch)
        y_batch = jnp.array(y_batch)
        
        # Ejecutar paso de entrenamiento
        state, metrics = train_step(state, x_cgm_batch, x_other_batch, y_batch)
        
        # Actualizar pérdida de época
        batch_loss = float(metrics[CONST_LOSS])
        epoch_loss += batch_loss / n_train_batches
        
        # Mostrar progreso del batch actual (cada 10 batches)
        if (i + 1) % 10 == 0 or i == 0 or i == len(train_batches) - 1:
            print(f"  Batch {i+1}/{len(train_batches)}, Loss: {batch_loss:.4f}")
    
    # Evaluar en validación
    val_metrics = eval_step(state, x_cgm_val_array, x_other_val_array, y_val_array)
    
    # Asegurarse de que CONST_METRIC_R2 existe en val_metrics
    val_metrics_dict = {k: float(v) for k, v in val_metrics.items()}
    if CONST_METRIC_R2 not in val_metrics_dict:
        val_metrics_dict[CONST_METRIC_R2] = 0.0  # Valor por defecto
    
    return epoch_loss, val_metrics_dict, state


def _update_history(history: Dict[str, List[float]],
                  epoch_loss: float,
                  val_metrics: Dict[str, jnp.ndarray]) -> Dict[str, List[float]]:
    """
    Actualiza el historial de métricas.
    """
    # Actualizar históricos
    history[CONST_LOSS].append(epoch_loss)
    history[CONST_VAL_LOSS].append(float(val_metrics[CONST_LOSS]))
    history[CONST_METRIC_MAE].append(float(val_metrics[CONST_METRIC_MAE]))
    history[f"val_{CONST_METRIC_MAE}"].append(float(val_metrics[CONST_METRIC_MAE]))
    history[CONST_METRIC_RMSE].append(float(val_metrics[CONST_METRIC_RMSE]))
    history[f"val_{CONST_METRIC_RMSE}"].append(float(val_metrics[CONST_METRIC_RMSE]))
    
    return history

def _extract_and_organize_data(data: Dict[str, Dict[str, np.ndarray]]) -> Dict[str, Any]:
    """
    Extrae y organiza los datos de entrenamiento, validación y prueba.
    """
    return {
        'train': {
            'x_cgm': data['train']['x_cgm'],
            'x_other': data['train']['x_other'],
            'y': data['train']['y']
        },
        'val': {
            'x_cgm': data['val']['x_cgm'],
            'x_other': data['val']['x_other'],
            'y': data['val']['y']
        },
        'test': {
            'x_cgm': data['test']['x_cgm'],
            'x_other': data['test']['x_other'],
            'y': data['test']['y']
        }
    }

def _init_training_config() -> Dict[str, Any]:
    """
    Inicializa la configuración de entrenamiento por defecto.
    """
    return {
        'epochs': CONST_EPOCHS,
        'batch_size': CONST_DEFAULT_BATCH_SIZE,
        'learning_rate': 0.001,
        'patience': 10,
        'seed': CONST_DEFAULT_SEED
    }

def _initialize_history() -> Dict[str, List[float]]:
    """
    Inicializa el diccionario de historial de métricas.
    """
    return {
        CONST_LOSS: [],
        CONST_VAL_LOSS: [],
        CONST_METRIC_MAE: [],
        f"val_{CONST_METRIC_MAE}": [],
        CONST_METRIC_RMSE: [],
        f"val_{CONST_METRIC_RMSE}": []
    }

def _handle_early_stopping(model, epoch, val_loss, state_params, best_params):
    """
    Maneja la lógica de early stopping personalizada.
    """
    if hasattr(model, 'early_stopping') and model.early_stopping is not None:
        if model.early_stopping(epoch, val_loss, state_params):
            print(f"\nEarly stopping activado en época {epoch+1}")
            if model.early_stopping.restore_best_weights:
                return model.early_stopping.get_best_params(), True
    return best_params, False

def _predict_and_evaluate(best_state, test_data):
    """
    Realiza predicciones con el mejor modelo y evalúa métricas.
    """
    x_cgm_test_array = jnp.array(test_data['x_cgm'])
    x_other_test_array = jnp.array(test_data['x_other'])
    y_test = test_data['y']
    
    y_pred_array = best_state.apply_fn(best_state.params, x_cgm_test_array, x_other_test_array)
    y_pred = np.array(y_pred_array).flatten()
    
    metrics = calculate_metrics(y_test, y_pred)
    
    return y_pred, metrics

def train_and_evaluate_model(model: nn.Module, 
                            model_name: str, 
                            data: Dict[str, Dict[str, np.ndarray]],
                            models_dir: str = CONST_MODELS,
                            training_config: Dict[str, Any] = None) -> Tuple[Dict[str, List[float]], np.ndarray, Dict[str, float]]:
    """
    Entrena y evalúa un modelo con características avanzadas de entrenamiento.
    
    Parámetros:
    -----------
    model : nn.Module
        Modelo a entrenar
    model_name : str
        Nombre del modelo para guardado y registro
    data : Dict[str, Dict[str, np.ndarray]]
        Diccionario con datos de entrenamiento, validación y prueba
        Estructura esperada:
        {
            'train': {'x_cgm': array, 'x_other': array, 'y': array},
            'val': {'x_cgm': array, 'x_other': array, 'y': array},
            'test': {'x_cgm': array, 'x_other': array, 'y': array}
        }
    models_dir : str, opcional
        Directorio para guardar modelos (default: "models")
    training_config : Dict[str, Any], opcional
        Configuración de entrenamiento con los siguientes valores (y sus defaults):
        {
            'epochs': 100,
            'batch_size': 32,
            'learning_rate': 0.001,
            'patience': 10,
            'seed': 42
        }
        
    Retorna:
    --------
    Tuple[Dict[str, List[float]], np.ndarray, Dict[str, float]]
        (historial, predicciones, métricas)
    """
    # Inicializar configuración
    if training_config is None:
        training_config = _init_training_config()
    
    # Extraer datos
    extracted_data = _extract_and_organize_data(data)
    x_cgm_train = extracted_data['train']['x_cgm']
    x_other_train = extracted_data['train']['x_other']
    y_train = extracted_data['train']['y']
    
    # Extraer parámetros de configuración
    epochs = training_config.get('epochs', CONST_EPOCHS)
    batch_size = training_config.get('batch_size', CONST_DEFAULT_BATCH_SIZE)
    patience = training_config.get('patience', 10)
    
    # Configurar entrenamiento
    state, _, rng = _setup_training(model, training_config, x_cgm_train, x_other_train, models_dir)
    
    # Preparar datos
    train_batches, n_train_batches, x_cgm_val_array, x_other_val_array, y_val_array = _prepare_data(
        x_cgm_train, x_other_train, y_train, 
        extracted_data['val']['x_cgm'], extracted_data['val']['x_other'], extracted_data['val']['y'], 
        batch_size, rng
    )
    
    # Inicializar históricos y variables para early stopping
    history = _initialize_history()
    wait, best_val_loss = 0, float('inf')
    best_state, best_params = state, state.params
    
    # Bucle de entrenamiento con early stopping

    print(f"\nEntrenando modelo {model_name}...")
    print(f"Configuración: {epochs} épocas, batch size: {batch_size}")
    print(f"Datos: {len(x_cgm_train)} ejemplos de entrenamiento, {len(x_cgm_val_array)} ejemplos de validación")
    
    for epoch in range(epochs):
        print(f"\nÉpoca {epoch+1}/{epochs}")
        
        # Mezclar datos para esta época
        rng, shuffle_rng = random.split(rng)
        train_batches, _ = create_batched_dataset(
            x_cgm_train, x_other_train, y_train, 
            batch_size=batch_size, shuffle=True, rng=shuffle_rng
        )
        
        # Mostrar información antes de cada época
        print(f"Procesando {n_train_batches} batches ({len(train_batches)} batches reales)")
        
        # Entrenar una época y actualizar históricos
        start_time = time.time()
        epoch_loss, val_metrics, state = _train_epoch(
            state, train_batches, n_train_batches, 
            x_cgm_val_array, x_other_val_array, y_val_array
        )
        epoch_time = time.time() - start_time
        history = _update_history(history, epoch_loss, val_metrics)
        
        # Verificar que CONST_METRIC_R2 exista
        r2_value = val_metrics.get(CONST_METRIC_R2, 0.0)
        
        # Imprimir métricas con tiempo
        print(f"Época {epoch+1}/{epochs} completada en {epoch_time:.2f}s - loss: {epoch_loss:.4f} - val_loss: {float(val_metrics[CONST_LOSS]):.4f} - MAE: {float(val_metrics[CONST_METRIC_MAE]):.4f} - RMSE: {float(val_metrics[CONST_METRIC_RMSE]):.4f} - R2: {float(r2_value):.4f}")
        
        # Manejo de early stopping
        val_loss = float(val_metrics[CONST_LOSS])
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_params = state.params
            best_state = state
            wait = 0
            
            # Guardar mejor modelo
            save_checkpoint(
                os.path.join(models_dir, f"{CONST_BEST_PREFIX}{model_name}"),
                best_state,
                step=epoch
            )
        else:
            wait += 1
            if wait >= patience:
                print(f"Early stopping en época {epoch+1}")
                break
        
        # Custom early stopping
        best_params, should_stop = _handle_early_stopping(model, epoch, val_loss, state.params, best_params)
        if should_stop:
            break
    
    # Restaurar los mejores parámetros y guardar modelo final
    state = state.replace(params=best_params)
    save_checkpoint(os.path.join(models_dir, model_name), state, step=epoch)
    
    # Hacer predicciones y calcular métricas
    y_pred, metrics = _predict_and_evaluate(best_state, extracted_data['test'])
    
    return history, y_pred, metrics


def train_model_sequential(model_creator: Callable, 
                          name: str, 
                          input_shapes: Tuple[Tuple[int, ...], Tuple[int, ...]], 
                          x_cgm_train: np.ndarray, 
                          x_other_train: np.ndarray, 
                          y_train: np.ndarray,
                          x_cgm_val: np.ndarray, 
                          x_other_val: np.ndarray, 
                          y_val: np.ndarray,
                          x_cgm_test: np.ndarray, 
                          x_other_test: np.ndarray, 
                          y_test: np.ndarray,
                          models_dir: str = CONST_MODELS) -> Dict[str, Any]:
    """
    Entrena un modelo secuencialmente y devuelve resultados serializables.
    
    Parámetros:
    -----------
    model_creator : Callable
        Función que crea el modelo
    name : str
        Nombre del modelo
    input_shapes : Tuple[Tuple[int, ...], Tuple[int, ...]]
        Formas de las entradas (CGM, otras)
    x_cgm_train : np.ndarray
        Datos CGM de entrenamiento
    x_other_train : np.ndarray
        Otras características de entrenamiento
    y_train : np.ndarray
        Valores objetivo de entrenamiento
    x_cgm_val : np.ndarray
        Datos CGM de validación
    x_other_val : np.ndarray
        Otras características de validación
    y_val : np.ndarray
        Valores objetivo de validación
    x_cgm_test : np.ndarray
        Datos CGM de prueba
    x_other_test : np.ndarray
        Otras características de prueba
    y_test : np.ndarray
        Valores objetivo de prueba
    models_dir : str, opcional
        Directorio para guardar modelos (default: "models")
        
    Retorna:
    --------
    Dict[str, Any]
        Diccionario con nombre del modelo, historial y predicciones
    """
    print(f"\nEntrenando modelo {name}...")
    
    # Identificar tipo de modelo para organización de figuras
    model_type = get_model_type(name)
    figures_path = os.path.join(CONST_FIGURES_DIR, CONST_MODEL_TYPES[model_type], name)
    os.makedirs(figures_path, exist_ok=True)
    
    # Crear modelo usando el model_creator
    model_wrapper = model_creator(input_shapes[0], input_shapes[1])
    
    # Verificar si es un wrapper (DLModelWrapper, RLModelWrapper, DRLModelWrapper)
    # Modificación: mejorar la condición para detectar wrappers
    is_wrapper = (hasattr(model_wrapper, 'model') or not hasattr(model_wrapper, 'init')) and not isinstance(model_wrapper, nn.Module)
    
    if is_wrapper:
        # Caso ModelWrapper: usar su API interna
        # Inicializar con clave aleatoria
        rng_key = jax.random.PRNGKey(CONST_DEFAULT_SEED)
        model_wrapper.start(x_cgm_train, x_other_train, y_train, rng_key)
        
        # Entrenar
        history = model_wrapper.train(
            x_cgm_train, x_other_train, y_train,
            validation_data=((x_cgm_val, x_other_val), y_val),
            epochs=CONST_EPOCHS, batch_size=CONST_DEFAULT_BATCH_SIZE
        )
        
        # Predecir
        y_pred = model_wrapper.predict(x_cgm_test, x_other_test)
        
    else:
        # Caso nn.Module: usar la lógica existente
        # Organizar datos en estructura esperada
        data = {
            'train': {'x_cgm': x_cgm_train, 'x_other': x_other_train, 'y': y_train},
            'val': {'x_cgm': x_cgm_val, 'x_other': x_other_val, 'y': y_val},
            'test': {'x_cgm': x_cgm_test, 'x_other': x_other_test, 'y': y_test}
        }
        
        # Configuración por defecto
        training_config = {
            'epochs': CONST_EPOCHS,
            'batch_size': CONST_DEFAULT_BATCH_SIZE
        }
        
        # Entrenar y evaluar modelo
        history, y_pred, _ = train_and_evaluate_model(
            model=model_wrapper,
            model_name=name,
            data=data,
            models_dir=models_dir,
            training_config=training_config
        )
    
    # Limpiar memoria
    jax.clear_caches()
    
    # Devolver sólo objetos serializables
    return {
        'name': name,
        'history': history,
        'predictions': y_pred.tolist() if hasattr(y_pred, 'tolist') else y_pred,
    }

def cross_validate_model(create_model_fn: Callable, 
                        x_cgm: np.ndarray, 
                        x_other: np.ndarray, 
                        y: np.ndarray, 
                        n_splits: int = 5, 
                        models_dir: str = CONST_MODELS) -> Tuple[Dict[str, float], Dict[str, float]]:
    """
    Realiza validación cruzada para un modelo.
    
    Parámetros:
    -----------
    create_model_fn : Callable
        Función que crea el modelo
    x_cgm : np.ndarray
        Datos CGM
    x_other : np.ndarray
        Otras características
    y : np.ndarray
        Valores objetivo
    n_splits : int, opcional
        Número de divisiones para validación cruzada (default: 5)
    models_dir : str, opcional
        Directorio para guardar modelos (default: "models")
        
    Retorna:
    --------
    Tuple[Dict[str, float], Dict[str, float]]
        (métricas_promedio, métricas_desviación)
    """
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=CONST_EPOCHS)
    scores = []
    
    for fold, (train_idx, val_idx) in enumerate(kf.split(x_cgm)):
        print(f"\nEntrenando fold {fold + 1}/{n_splits}")
        
        # Dividir datos
        x_cgm_train_fold = x_cgm[train_idx]
        x_cgm_val_fold = x_cgm[val_idx]
        x_other_train_fold = x_other[train_idx]
        x_other_val_fold = x_other[val_idx]
        y_train_fold = y[train_idx]
        y_val_fold = y[val_idx]
        
        # Crear modelo
        model = create_model_fn()
        
        # Entrenar y evaluar modelo
        # Organizar datos en estructura esperada
        data = {
            'train': {'x_cgm': x_cgm_train_fold, 'x_other': x_other_train_fold, 'y': y_train_fold},
            'val': {'x_cgm': x_cgm_val_fold, 'x_other': x_other_val_fold, 'y': y_val_fold},
            'test': {'x_cgm': x_cgm_val_fold, 'x_other': x_other_val_fold, 'y': y_val_fold}
        }
        
        _, _, metrics = train_and_evaluate_model(
            model=model,
            model_name=f'fold_{fold}',
            data=data,
            models_dir=models_dir
        )
        
        scores.append(metrics)
    
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

def predict_model(model_path: str, 
                 model_creator: Callable, 
                 x_cgm: np.ndarray, 
                 x_other: np.ndarray, 
                 input_shapes: Optional[Tuple[Tuple[int, ...], Tuple[int, ...]]] = None) -> np.ndarray:
    """
    Realiza predicciones con un modelo guardado.
    
    Parámetros:
    -----------
    model_path : str
        Ruta al modelo guardado
    model_creator : Callable
        Función que crea el modelo
    x_cgm : np.ndarray
        Datos CGM para predicción
    x_other : np.ndarray
        Otras características para predicción
    input_shapes : Optional[Tuple[Tuple[int, ...], Tuple[int, ...]]], opcional
        Formas de las entradas. Si es None, se infieren (default: None)
        
    Retorna:
    --------
    np.ndarray
        Predicciones del modelo
    """
    # Determinar formas de entrada si no se proporcionan
    if input_shapes is None:
        input_shapes = ((x_cgm.shape[1:]), (x_other.shape[1:]))
    
    # Crear modelo
    model = model_creator(input_shapes[0], input_shapes[1])
    
    # Inicializar modelo
    rng = random.PRNGKey(0)
    x_cgm_shape = (1,) + input_shapes[0]
    x_other_shape = (1,) + input_shapes[1]
    
    params = model.init(rng, jnp.ones(x_cgm_shape), jnp.ones(x_other_shape))
    
    # Crear estado de entrenamiento mínimo
    state = train_state.TrainState.create(
        apply_fn=model.apply,
        params=params,
        tx=optax.sgd(0.1)  # Optimizador ficticio, no se usa para predicciones
    )
    
    # Cargar parámetros guardados
    state = restore_checkpoint(model_path, state)
    
    # Convertir entradas a arrays JAX
    x_cgm_array = jnp.array(x_cgm)
    x_other_array = jnp.array(x_other)
    
    # Realizar predicción
    y_pred = state.apply_fn(state.params, x_cgm_array, x_other_array)
    
    return np.array(y_pred).flatten()


def train_multiple_models(model_creators: Dict[str, Callable], 
                         input_shapes: Tuple[Tuple[int, ...], Tuple[int, ...]],
                         x_cgm_train: np.ndarray, 
                         x_other_train: np.ndarray, 
                         y_train: np.ndarray,
                         x_cgm_val: np.ndarray, 
                         x_other_val: np.ndarray, 
                         y_val: np.ndarray,
                         x_cgm_test: np.ndarray, 
                         x_other_test: np.ndarray, 
                         y_test: np.ndarray,
                         models_dir: str = CONST_MODELS) -> Tuple[Dict[str, Dict], Dict[str, np.ndarray], Dict[str, Dict]]:
    """
    Entrena múltiples modelos y recopila sus resultados.
    
    Parámetros:
    -----------
    model_creators : Dict[str, Callable]
        Diccionario de funciones creadoras de modelos indexadas por nombre
    input_shapes : Tuple[Tuple[int, ...], Tuple[int, ...]]
        Formas de las entradas (CGM, otras)
    x_cgm_train : np.ndarray
        Datos CGM de entrenamiento
    x_other_train : np.ndarray
        Otras características de entrenamiento
    y_train : np.ndarray
        Valores objetivo de entrenamiento
    x_cgm_val : np.ndarray
        Datos CGM de validación
    x_other_val : np.ndarray
        Otras características de validación
    y_val : np.ndarray
        Valores objetivo de validación
    x_cgm_test : np.ndarray
        Datos CGM de prueba
    x_other_test : np.ndarray
        Otras características de prueba
    y_test : np.ndarray
        Valores objetivo de prueba
    models_dir : str, opcional
        Directorio para guardar modelos (default: "models")
        
    Retorna:
    --------
    Tuple[Dict[str, Dict], Dict[str, np.ndarray], Dict[str, Dict]]
        (historiales, predicciones, métricas) diccionarios
    """
    models_names = list(model_creators.keys())
    
    model_results = []
    for name in models_names:
        result = train_model_sequential(
            model_creators[name], name, input_shapes,
            x_cgm_train, x_other_train, y_train,
            x_cgm_val, x_other_val, y_val,
            x_cgm_test, x_other_test, y_test,
            models_dir
        )
        model_results.append(result)
    
    # Procesar resultados en paralelo
    print("\nCalculando métricas en paralelo...")
    with Parallel(n_jobs=-1, verbose=1) as parallel:
        metric_results = parallel(
            delayed(calculate_metrics)(
                y_test, 
                np.array(result['predictions'])
            ) for result in model_results
        )
    
    # metric_results = process_training_results(model_results=model_results, y_test=y_test)
    print_debug("metric_results:")
    print_debug(metric_results)
    
    # Almacenar resultados
    histories = {}
    predictions = {}
    metrics = {}
    
    for result, metric in zip(model_results, metric_results):
        name = result['name']
        histories[name] = result['history']
        predictions[name] = np.array(result['predictions'])
        metrics[name] = metric
    
    return histories, predictions, metrics