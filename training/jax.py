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

# Constantes para uso común
CONST_VAL_LOSS = "val_loss"
CONST_LOSS = "loss"
CONST_METRIC_MAE = "mae"
CONST_METRIC_RMSE = "rmse"
CONST_METRIC_R2 = "r2"
CONST_MODELS = "models"
CONST_BEST_PREFIX = "best_"
CONST_LOGS_DIR = "logs"


def create_batched_dataset(x_cgm: np.ndarray, 
                          x_other: np.ndarray, 
                          y: np.ndarray, 
                          batch_size: int = 32, 
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
        indices = random.permutation(rng, indices)
    elif shuffle:
        rng = np.random.Generator(np.random.PCG64(42))
        rng.shuffle(indices)
    
    # Calcular número de batches
    n_batches = int(np.ceil(n_samples / batch_size))
    
    # Crear lista de batches
    batches = []
    for i in range(n_batches):
        # Obtener índices para el batch actual
        batch_indices = indices[i * batch_size:(i + 1) * batch_size]
        
        # Crear batch
        batch = (
            (x_cgm[batch_indices], x_other[batch_indices]),
            y[batch_indices]
        )
        batches.append(batch)
    
    return batches, n_batches


def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """
    Calcula métricas de rendimiento para las predicciones del modelo.
    
    Parámetros:
    -----------
    y_true : np.ndarray
        Valores objetivo verdaderos
    y_pred : np.ndarray
        Valores predichos por el modelo
        
    Retorna:
    --------
    Dict[str, float]
        Diccionario con métricas MAE, RMSE y R²
    """
    return {
        CONST_METRIC_MAE: float(mean_absolute_error(y_true, y_pred)),
        CONST_METRIC_RMSE: float(np.sqrt(mean_squared_error(y_true, y_pred))),
        CONST_METRIC_R2: float(r2_score(y_true, y_pred))
    }


def mse_loss(params, model, x_cgm, x_other, y):
    """
    Función de pérdida de error cuadrático medio para entrenamiento.
    
    Parámetros:
    -----------
    params : Dict
        Parámetros del modelo
    model : nn.Module o callable
        Modelo a utilizar o función apply del modelo
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
    # Crear PRNG para inferencia
    dropout_rng = jax.random.PRNGKey(0)  # No afecta en modo deterministic/eval
    
    # Realizar predicción - comprobar si model es una función o un objeto con método apply
    if hasattr(model, 'apply'):
        y_pred = model.apply(params, x_cgm, x_other, rngs={'dropout': dropout_rng}, training=False).flatten()
    else:
        # Si es una función (como state.apply_fn), llamarla directamente
        y_pred = model(params, x_cgm, x_other, rngs={'dropout': dropout_rng}, training=False).flatten()
    
    # Calcular error cuadrático medio
    return jnp.mean(jnp.square(y_pred - y))

# Versión JIT-compilada de mse_loss
mse_loss_jit = jit(mse_loss, static_argnames=['model'])


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
    # Definir una función de pérdida para este paso que no necesite el modelo como argumento
    def loss_fn(params):
        return mse_loss_jit(params, state.apply_fn, x_cgm, x_other, y)
    
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
    # Usar una función local para calcular la pérdida sin pasar el modelo directamente
    def loss_fn(params):
        return mse_loss_jit(params, state.apply_fn, x_cgm, x_other, y)
    
    # Calcular pérdida
    loss = loss_fn(state.params)
    
    # Crear un PRNG para la evaluación (modo deterministic)
    eval_rng = jax.random.PRNGKey(0)
    
    # Calcular predicciones con rngs explícito
    y_pred = state.apply_fn(state.params, x_cgm, x_other, 
                           rngs={'dropout': eval_rng},
                           training=False).flatten()
    
    # Calcular error absoluto medio
    mae = jnp.mean(jnp.abs(y_pred - y))
    
    # Calcular raíz del error cuadrático medio
    rmse = jnp.sqrt(jnp.mean(jnp.square(y_pred - y)))
    
    return {
        CONST_LOSS: loss,
        CONST_METRIC_MAE: mae,
        CONST_METRIC_RMSE: rmse
    }


def normalize_array(arr_data, is_cgm=False):
    """
    Normaliza arrays con formas potencialmente heterogéneas.
    
    Parámetros:
    -----------
    arr_data : array-like
        Datos a normalizar
    is_cgm : bool
        Indica si los datos son CGM (tienen forma especial)
    """
    if isinstance(arr_data, (np.ndarray, jnp.ndarray)):
        return np.array(arr_data, dtype=np.float32)
        
    if isinstance(arr_data, list):
        # Si es una lista vacía
        if not arr_data:
            return np.array([], dtype=np.float32)
            
        # Para datos CGM, necesitamos un manejo especial
        if is_cgm:
            # Verificar la estructura de los datos
            print(f"Forma de los datos CGM (primeros elementos):")
            for i in range(min(3, len(arr_data))):
                print(f"Elemento {i}: {np.shape(arr_data[i])}")
            
            try:
                # Intentar convertir cada secuencia a un array numpy primero
                normalized_sequences = []
                max_time_steps = 0
                max_features = 0
                
                for seq in arr_data:
                    if seq is None:
                        continue
                    
                    seq_array = np.array(seq, dtype=np.float32)
                    if len(seq_array.shape) == 1:
                        seq_array = seq_array.reshape(-1, 1)
                    elif len(seq_array.shape) > 2:
                        # Si tiene más de 2 dimensiones, aplanar las dimensiones extra
                        seq_array = seq_array.reshape(seq_array.shape[0], -1)
                    
                    max_time_steps = max(max_time_steps, seq_array.shape[0])
                    max_features = max(max_features, seq_array.shape[1])
                    normalized_sequences.append(seq_array)
                
                # Ahora que tenemos las dimensiones máximas, podemos padear todas las secuencias
                padded_sequences = []
                for seq in normalized_sequences:
                    # Padear tiempo y características si es necesario
                    pad_time = max_time_steps - seq.shape[0]
                    pad_features = max_features - seq.shape[1]
                    
                    if pad_time > 0 or pad_features > 0:
                        padded_seq = np.pad(
                            seq,
                            ((0, pad_time), (0, pad_features)),
                            mode='constant',
                            constant_values=0
                        )
                    else:
                        padded_seq = seq
                    
                    padded_sequences.append(padded_seq)
                
                result = np.array(padded_sequences, dtype=np.float32)
                print(f"Forma final del array CGM: {result.shape}")
                return result
                
            except Exception as e:
                print(f"Error al procesar datos CGM: {str(e)}")
                # Si falla el método anterior, intentar un enfoque más simple
                try:
                    # Convertir cada secuencia a un vector 1D
                    flattened = [np.array(x).flatten() if x is not None else np.array([]) for x in arr_data]
                    # Encontrar la longitud máxima
                    max_len = max(len(x) for x in flattened)
                    # Padear todas las secuencias a la misma longitud
                    padded = [np.pad(x, (0, max_len - len(x)), mode='constant') if len(x) < max_len else x[:max_len] 
                             for x in flattened]
                    return np.array(padded, dtype=np.float32)
                except Exception as e2:
                    print(f"Error al procesar datos CGM (método alternativo): {str(e2)}")
                    raise
        else:
            # Para otros datos, intentar convertir directamente
            try:
                return np.array(arr_data, dtype=np.float32)
            except ValueError:
                # Si falla, intentar aplanar los datos
                flattened = [np.array(x).flatten() if x is not None else np.array([]) for x in arr_data]
                max_len = max(len(x) for x in flattened)
                padded = [np.pad(x, (0, max_len - len(x)), mode='constant') if len(x) < max_len else x[:max_len] 
                         for x in flattened]
                return np.array(padded, dtype=np.float32)
    else:
        # Si no es lista ni array, intentar convertir directamente
        return np.array(arr_data, dtype=np.float32)

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
    # Configuración por defecto
    if training_config is None:
        training_config = {
            'epochs': 100,
            'batch_size': 32,
            'learning_rate': 0.001,
            'patience': 10,
            'seed': 42
        }
    
    # Extraer y normalizar datos
    try:
        x_cgm_train = normalize_array(data['train']['x_cgm'], is_cgm=True)
        x_other_train = normalize_array(data['train']['x_other'])
        y_train = normalize_array(data['train']['y'])
        
        x_cgm_val = normalize_array(data['val']['x_cgm'], is_cgm=True)
        x_other_val = normalize_array(data['val']['x_other'])
        y_val = normalize_array(data['val']['y'])
        
        x_cgm_test = normalize_array(data['test']['x_cgm'], is_cgm=True)
        x_other_test = normalize_array(data['test']['x_other'])
        y_test = normalize_array(data['test']['y'])
    
        # Verificar si hubo algún problema en la conversión
        print(f"Forma de x_cgm_train: {x_cgm_train.shape}")
        print(f"Forma de x_other_train: {x_other_train.shape}")
        print(f"Forma de y_train: {y_train.shape}")
    except Exception as e:
        print(f"Error al convertir datos en {model_name}: {str(e)}")
        raise
    
    # Extraer parámetros de configuración
    epochs = training_config.get('epochs', 100)
    batch_size = training_config.get('batch_size', 32)
    learning_rate = training_config.get('learning_rate', 0.001)
    patience = training_config.get('patience', 10)
    seed = training_config.get('seed', 42)
    
    # Crear directorio para modelos si no existe
    os.makedirs(models_dir, exist_ok=True)
    
    # Inicializar generador de números aleatorios
    rng = random.PRNGKey(seed)
    rng, init_rng = random.split(rng)
    
    # Inicializar modelo
    x_cgm_shape = (1,) + x_cgm_train.shape[1:]
    x_other_shape = (1,) + x_other_train.shape[1:]
    
    # Separar un PRNG específico para dropout
    init_rng, dropout_rng = random.split(init_rng)
    
    # Inicializar modelo con PRNG para dropout
    params = model.init({'params': init_rng, 'dropout': dropout_rng}, 
                        jnp.ones(x_cgm_shape), jnp.ones(x_other_shape),
                        training=True)
    
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
    
    # Crear conjuntos de datos en batches
    train_batches, n_train_batches = create_batched_dataset(
        x_cgm_train, x_other_train, y_train, batch_size=batch_size, shuffle=True, rng=rng
    )
    
    # Convertir datos de validación para evaluación
    x_cgm_val_array = jnp.array(x_cgm_val)
    x_other_val_array = jnp.array(x_other_val)
    y_val_array = jnp.array(y_val)
    
    # Inicializar históricos
    history = {
        CONST_LOSS: [],
        CONST_VAL_LOSS: [],
        CONST_METRIC_MAE: [],
        f"val_{CONST_METRIC_MAE}": [],
        CONST_METRIC_RMSE: [],
        f"val_{CONST_METRIC_RMSE}": []
    }
    
    # Variables para early stopping
    wait = 0
    best_val_loss = float('inf')
    best_state = state
    
    # Bucle de entrenamiento
    print(f"\nEntrenando modelo {model_name}...")
    for epoch in range(epochs):
        # Mezclar datos para esta época
        rng, shuffle_rng = random.split(rng)
        train_batches, _ = create_batched_dataset(
            x_cgm_train, x_other_train, y_train, 
            batch_size=batch_size, shuffle=True, rng=shuffle_rng
        )
        
        # Variables para métricas de época
        epoch_loss = 0.0
        
        # Bucle sobre batches
        for batch in train_batches:
            # Desempaquetar batch
            (x_cgm_batch, x_other_batch), y_batch = batch
            
            # Convertir a arrays de JAX
            x_cgm_batch = jnp.array(x_cgm_batch)
            x_other_batch = jnp.array(x_other_batch)
            y_batch = jnp.array(y_batch)
            
            # Ejecutar paso de entrenamiento
            state, metrics = train_step(state, x_cgm_batch, x_other_batch, y_batch)
            
            # Actualizar pérdida de época
            epoch_loss += float(metrics[CONST_LOSS]) / n_train_batches
        
        # Evaluar en validación
        val_metrics = eval_step(state, x_cgm_val_array, x_other_val_array, y_val_array)
        
        # Actualizar históricos
        history[CONST_LOSS].append(epoch_loss)
        history[CONST_VAL_LOSS].append(float(val_metrics[CONST_LOSS]))
        history[CONST_METRIC_MAE].append(float(val_metrics[CONST_METRIC_MAE]))
        history[f"val_{CONST_METRIC_MAE}"].append(float(val_metrics[CONST_METRIC_MAE]))
        history[CONST_METRIC_RMSE].append(float(val_metrics[CONST_METRIC_RMSE]))
        history[f"val_{CONST_METRIC_RMSE}"].append(float(val_metrics[CONST_METRIC_RMSE]))
        
        # Mostrar progreso cada 10 épocas
        if epoch % 10 == 0 or epoch == epochs - 1:
            print(f"Época {epoch+1}/{epochs} - "
                  f"loss: {epoch_loss:.4f} - "
                  f"val_loss: {float(val_metrics[CONST_LOSS]):.4f}")
        
        # Early stopping
        val_loss = float(val_metrics[CONST_LOSS])
        if val_loss < best_val_loss:
            best_val_loss = val_loss
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
    
    # Guardar modelo final
    save_checkpoint(
        os.path.join(models_dir, model_name),
        state,
        step=epoch
    )
    
    # Hacer predicciones con el mejor modelo
    x_cgm_test_array = jnp.array(x_cgm_test)
    x_other_test_array = jnp.array(x_other_test)
    y_pred_array = best_state.apply_fn(best_state.params, x_cgm_test_array, x_other_test_array)
    y_pred = np.array(y_pred_array).flatten()
    
    # Calcular métricas finales
    metrics = calculate_metrics(y_test, y_pred)
    
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
    
    # Crear modelo
    model = model_creator(input_shapes[0], input_shapes[1])
    
    # Organizar datos en estructura esperada
    data = {
        'train': {'x_cgm': x_cgm_train, 'x_other': x_other_train, 'y': y_train},
        'val': {'x_cgm': x_cgm_val, 'x_other': x_other_val, 'y': y_val},
        'test': {'x_cgm': x_cgm_test, 'x_other': x_other_test, 'y': y_test}
    }
    
    # Configuración por defecto
    training_config = {
        'epochs': 100,
        'batch_size': 32
    }
    
    # Entrenar y evaluar modelo
    history, y_pred, _ = train_and_evaluate_model(
        model=model,
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
        'predictions': y_pred.tolist(),
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
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
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


def create_ensemble_prediction(predictions_dict: Dict[str, np.ndarray], 
                             weights: Optional[np.ndarray] = None) -> np.ndarray:
    """
    Combina predicciones de múltiples modelos usando promedio ponderado.
    
    Parámetros:
    -----------
    predictions_dict : Dict[str, np.ndarray]
        Diccionario con predicciones de cada modelo
    weights : Optional[np.ndarray], opcional
        Pesos para cada modelo. Si es None, usa promedio simple (default: None)
        
    Retorna:
    --------
    np.ndarray
        Predicciones combinadas del ensemble
    """
    all_preds = np.stack(list(predictions_dict.values()))
    if weights is None:
        weights = np.ones(len(predictions_dict)) / len(predictions_dict)
    return np.average(all_preds, axis=0, weights=weights)


def optimize_ensemble_weights(predictions_dict: Dict[str, np.ndarray], 
                            y_true: np.ndarray) -> np.ndarray:
    """
    Optimiza pesos del ensemble usando optimización.
    
    Parámetros:
    -----------
    predictions_dict : Dict[str, np.ndarray]
        Diccionario con predicciones de cada modelo
    y_true : np.ndarray
        Valores objetivo verdaderos
        
    Retorna:
    --------
    np.ndarray
        Pesos optimizados para cada modelo
    """
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


def enhance_features(x_cgm: np.ndarray, x_other: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Mejora las características de entrada con características derivadas.
    
    Parámetros:
    -----------
    x_cgm : np.ndarray
        Datos CGM
    x_other : np.ndarray
        Otras características
        
    Retorna:
    --------
    Tuple[np.ndarray, np.ndarray]
        (x_cgm_mejorado, x_other)
    """
    # Añadir características derivadas para CGM
    cgm_diff = np.diff(x_cgm.squeeze(), axis=1)
    
    # Ajustar padding según la forma real del array
    padding = [(0,0) for _ in range(cgm_diff.ndim)]
    padding[1] = (1,0)  # Añadir padding solo en la segunda dimensión
    cgm_diff = np.pad(cgm_diff, padding, mode='edge')
    
    # Añadir estadísticas móviles
    window = 5
    rolling_mean = np.apply_along_axis(
        lambda x: np.convolve(x, np.ones(window)/window, mode='same'),
        1, x_cgm.squeeze()
    )
    
    # Concatenar características mejoradas
    x_cgm_enhanced = np.concatenate([
        x_cgm,
        cgm_diff[..., np.newaxis],
        rolling_mean[..., np.newaxis]
    ], axis=-1)
    
    return x_cgm_enhanced, x_other


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
                         x_cgm_train: Union[np.ndarray, Tuple[np.ndarray, np.ndarray]], 
                         x_other_train: np.ndarray, 
                         y_train: np.ndarray,
                         x_cgm_val: Union[np.ndarray, Tuple[np.ndarray, np.ndarray]], 
                         x_other_val: np.ndarray, 
                         y_val: np.ndarray,
                         x_cgm_test: Union[np.ndarray, Tuple[np.ndarray, np.ndarray]], 
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
    x_cgm_train : Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]
        Datos CGM de entrenamiento (o tupla con CGM y otras características)
    x_other_train : np.ndarray
        Otras características de entrenamiento
    y_train : np.ndarray
        Valores objetivo de entrenamiento
    x_cgm_val : Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]
        Datos CGM de validación (o tupla con CGM y otras características)
    x_other_val : np.ndarray
        Otras características de validación
    y_val : np.ndarray
        Valores objetivo de validación
    x_cgm_test : Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]
        Datos CGM de prueba (o tupla con CGM y otras características)
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
    
    # Detectar si los datos CGM vienen como tupla y extraerlos
    if isinstance(x_cgm_train, tuple) and len(x_cgm_train) == 2:
        print("Detectada tupla de datos (CGM, otros) - Extrayendo correctamente...")
        x_cgm_train_actual = x_cgm_train[0]  # Tomar solo el primer elemento
    else:
        x_cgm_train_actual = x_cgm_train
        
    if isinstance(x_cgm_val, tuple) and len(x_cgm_val) == 2:
        x_cgm_val_actual = x_cgm_val[0]
    else:
        x_cgm_val_actual = x_cgm_val
        
    if isinstance(x_cgm_test, tuple) and len(x_cgm_test) == 2:
        x_cgm_test_actual = x_cgm_test[0]
    else:
        x_cgm_test_actual = x_cgm_test
    
    model_results = []
    for name in models_names:
        print(f"\nEntrenando modelo {name}...")
        
        # Imprimir información de formas
        print(f"Forma de datos CGM de entrenamiento: {x_cgm_train_actual.shape}")
        print(f"Forma de otras características de entrenamiento: {x_other_train.shape}")
        
        result = train_model_sequential(
            model_creators[name], name, input_shapes,
            x_cgm_train_actual, x_other_train, y_train,
            x_cgm_val_actual, x_other_val, y_val,
            x_cgm_test_actual, x_other_test, y_test,
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