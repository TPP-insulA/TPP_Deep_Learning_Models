import os
import numpy as np
import tensorflow as tf
from keras._tf_keras.keras.models import Model
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold
from joblib import Parallel, delayed
from scipy.optimize import minimize
from typing import Dict, List, Tuple, Callable, Optional, Any, Union
from config.params import DEBUG
from training.common import (
    calculate_metrics, create_ensemble_prediction, optimize_ensemble_weights,
    enhance_features, get_model_type, process_training_results
)
from constants.constants import (
    CONST_VAL_LOSS, CONST_LOSS, CONST_METRIC_MAE, CONST_METRIC_RMSE, CONST_METRIC_R2,
    CONST_MODELS, CONST_BEST_PREFIX, CONST_LOGS_DIR, CONST_DEFAULT_EPOCHS, 
    CONST_DEFAULT_BATCH_SIZE, CONST_DEFAULT_SEED, CONST_FIGURES_DIR, CONST_MODEL_TYPES
)
from custom.printer import print_info, print_warning, print_error, print_success

import contextlib

@contextlib.contextmanager
def model_device_context(model_name) -> Any:
    """
    Maneja el contexto del dispositivo para modelos específicos.
    Si el modelo es incompatible con GPU, fuerza el uso de CPU.
    
    Parámetros:
    -----------
    model_name : str
        Nombre del modelo para determinar el dispositivo a usar
    """
    # Guardar la configuración original de CUDA
    original_cuda_visible = os.environ.get('CUDA_VISIBLE_DEVICES', '')
    
    # Modelos problemáticos
    cpu_only_models = ['tf_tabnet', 'tf_sarsa', 'tf_monte_carlo']
    
    try:
        # Forzar el uso de CPU para modelos específicos
        if any(model_name.lower() == cpu_model for cpu_model in cpu_only_models):
            print_info(f"Model {model_name} using CPU only (GPU-incompatible operations)")
            os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
        yield
    finally:
        # Restaurar la configuración original de CUDA
        os.environ['CUDA_VISIBLE_DEVICES'] = original_cuda_visible

CONST_EPOCHS = 2 if DEBUG else CONST_DEFAULT_EPOCHS

def configure_tensorflow_gpu():
    """
    Configurar adecuadamente TensorFlow para uso de GPU con fallback a CPU.
    """
    
    # Intentar encontrar CUDA y libdevice
    cuda_locations = [
        "/usr/local/cuda-11.8",
        "/usr/local/cuda",
        "/usr/lib/cuda",
        "/opt/cuda",
    ]
    
    cuda_found = False
    for cuda_path in cuda_locations:
        libdevice_path = os.path.join(cuda_path, "nvvm", "libdevice")
        if os.path.exists(os.path.join(libdevice_path, "libdevice.10.bc")):
            os.environ["XLA_FLAGS"] = f"--xla_gpu_cuda_data_dir={cuda_path}"
            print_info(f"Librerías CUDA encontradas en:  {cuda_path}")
            cuda_found = True
            break
    
    if not cuda_found:
        print_warning("No se encontraron librerías CUDA.")
    
    # Configurar la memoria de GPU para crecimiento dinámico
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            print_info(f"GPU detected: {len(gpus)} device(s) with memory growth enabled")
            
            # Test GPU 
            try:
                with tf.device('/GPU:0'):
                    a = tf.random.normal([1000, 1000])
                    b = tf.random.normal([1000, 1000])
                    c = tf.matmul(a, b)
                    _ = c.numpy() 
                print_success("Test de GPU exitoso")
                return True
            except Exception as e:
                print_error(f"Test de GPU fallido: {e}")
                print_error("Fallback a CPU.")
                os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
                return False
        except RuntimeError as e:
            print_error(f"Error de configuración de GPU: {e}")
            print_warning("Fallback a CPU.")
            os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
            return False
    else:
        print_info("No se detectaron GPU. Usando CPU.")
        return False

use_gpu = configure_tensorflow_gpu()

def create_dataset(x_cgm: np.ndarray, 
                  x_other: np.ndarray, 
                  y: np.ndarray, 
                  batch_size: int = CONST_DEFAULT_BATCH_SIZE) -> tf.data.Dataset:
    """
    Crea un dataset optimizado usando tf.data.
    
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
        
    Retorna:
    --------
    tf.data.Dataset
        Dataset optimizado para entrenamiento
    """
    dataset = tf.data.Dataset.from_tensor_slices((
        (x_cgm, x_other), y
    ))
    return dataset.cache().shuffle(1000).batch(batch_size).prefetch(tf.data.AUTOTUNE)

def train_and_evaluate_model(model: Model, 
                           model_name: str, 
                           data: Dict[str, Dict[str, np.ndarray]],
                           models_dir: str = CONST_MODELS,
                           training_config: Dict[str, Any] = None) -> Tuple[Dict[str, List[float]], np.ndarray, Dict[str, float]]:
    """
    Entrena y evalúa un modelo con características avanzadas de entrenamiento.
    
    Parámetros:
    -----------
    model : Model
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
            'patience': 10
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
            'patience': 10
        }
    
    # Extraer datos
    x_cgm_train = data['train']['x_cgm']
    x_other_train = data['train']['x_other']
    y_train = data['train']['y']
    
    x_cgm_val = data['val']['x_cgm']
    x_other_val = data['val']['x_other'] 
    y_val = data['val']['y']
    
    x_cgm_test = data['test']['x_cgm']
    x_other_test = data['test']['x_other']
    y_test = data['test']['y']
    
    # Extraer parámetros de configuración
    epochs = training_config.get('epochs', 100)
    batch_size = training_config.get('batch_size', 32)
    learning_rate = training_config.get('learning_rate', 0.001)
    patience = training_config.get('patience', 10)
    
    # Crear directorios necesarios
    os.makedirs(models_dir, exist_ok=True)
    log_dir = os.path.join(models_dir, CONST_LOGS_DIR, model_name)
    os.makedirs(log_dir, exist_ok=True)
    
    # Habilitar compilación XLA
    if model_name in ['tf_tabnet'] and use_gpu:
        tf.config.optimizer.set_jit(False)
    else:
        tf.config.optimizer.set_jit(True)
    
    # Identificar el tipo de modelo (RL o DL)
    is_rl_model = model_name.startswith('tf_policy_iteration') or model_name.startswith('tf_value_iteration') or \
                  model_name.startswith('tf_monte_carlo') or model_name.startswith('tf_q_learning') or \
                  model_name.startswith('tf_sarsa') or model_name.startswith('tf_reinforce')
    
    if is_rl_model:
        # Para modelos de RL, usamos una estrategia diferente
        print(f"Entrenando modelo RL: {model_name}")
        
        # Callbacks para monitoreo y optimización
        callbacks = [
            # Early stopping para evitar sobreajuste
            tf.keras.callbacks.EarlyStopping(
                monitor=CONST_VAL_LOSS,
                patience=patience,
                restore_best_weights=True
            ),
            # Reducir tasa de aprendizaje cuando el modelo se estanca
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor=CONST_VAL_LOSS,
                factor=0.5,
                patience=patience // 2,
                min_lr=1e-6
            ),
            # Guardar mejor modelo
            tf.keras.callbacks.ModelCheckpoint(
                os.path.join(models_dir, f'{CONST_BEST_PREFIX}{model_name}.keras'),
                monitor=CONST_VAL_LOSS,
                save_best_only=True
            ),
            # TensorBoard para visualización
            tf.keras.callbacks.TensorBoard(
                log_dir=log_dir,
                histogram_freq=1
            )
        ]

        optimizer = tf.keras.optimizers.Adam(
            learning_rate=learning_rate,
            clipnorm=1.0
        )
        # Compilar modelo con múltiples métricas
        model.compile(
            optimizer=optimizer,
            loss='mse',
            metrics=[CONST_METRIC_MAE, tf.keras.metrics.RootMeanSquaredError(name=CONST_METRIC_RMSE)]
        )
        
        # Para modelos RL, pasamos los datos de forma explícita
        history = model.fit(
            x=[x_cgm_train, x_other_train],
            y=y_train,
            batch_size=batch_size,
            epochs=epochs,
            verbose=1,
            callbacks=callbacks,
            validation_data=([x_cgm_val, x_other_val], y_val)
        )
        
        # Historial simulado para compatibilidad
        history_dict = {
            'loss': [0.0],
            'val_loss': [0.0]
        }
    else:
        # Crear datasets optimizados para modelos DL
        train_ds = create_dataset(x_cgm_train, x_other_train, y_train, batch_size=batch_size)
        val_ds = create_dataset(x_cgm_val, x_other_val, y_val, batch_size=batch_size)
        
        # Usar una tasa de aprendizaje fija en lugar de un planificador
        # para permitir que ReduceLROnPlateau funcione correctamente
        optimizer = tf.keras.optimizers.Adam(
            learning_rate=learning_rate,
            clipnorm=1.0
        )
        
        # Habilitar entrenamiento con precisión mixta
        tf.keras.mixed_precision.set_global_policy('mixed_float16')
        
        # Compilar modelo con múltiples métricas
        model.compile(
            optimizer=optimizer,
            loss='mse',
            metrics=[CONST_METRIC_MAE, tf.keras.metrics.RootMeanSquaredError(name=CONST_METRIC_RMSE)]
        )
        
        # Callbacks para monitoreo y optimización
        callbacks = [
            # Early stopping para evitar sobreajuste
            tf.keras.callbacks.EarlyStopping(
                monitor=CONST_VAL_LOSS,
                patience=patience,
                restore_best_weights=True
            ),
            # Reducir tasa de aprendizaje cuando el modelo se estanca
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor=CONST_VAL_LOSS,
                factor=0.5,
                patience=patience // 2,
                min_lr=1e-6
            ),
            # Guardar mejor modelo
            tf.keras.callbacks.ModelCheckpoint(
                os.path.join(models_dir, f'{CONST_BEST_PREFIX}{model_name}.keras'),
                monitor=CONST_VAL_LOSS,
                save_best_only=True
            ),
            # TensorBoard para visualización
            tf.keras.callbacks.TensorBoard(
                log_dir=log_dir,
                histogram_freq=1
            )
        ]
        
        # Entrenar modelo
        print(f"\nEntrenando modelo {model_name}...")
        history = model.fit(
            train_ds,
            validation_data=val_ds,
            epochs=epochs,
            callbacks=callbacks,
            verbose=1
        )
        
        # Obtener el historial de entrenamiento
        history_dict = history.history
        
        # Restaurar política de precisión predeterminada
        tf.keras.mixed_precision.set_global_policy('float32')
    
    # Predecir y evaluar
    y_pred = model.predict([x_cgm_test, x_other_test]).flatten()
    
    # Calcular métricas
    metrics = calculate_metrics(y_test, y_pred)
    
    # Guardar modelo final
    model.save(os.path.join(models_dir, f'{model_name}.keras'))
    
    return history_dict, y_pred, metrics


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
        'epochs': CONST_EPOCHS,
        'batch_size': CONST_DEFAULT_BATCH_SIZE
    }
    
    # Entrenar y evaluar modelo
    with model_device_context(name):
        history, y_pred, _metrics = train_and_evaluate_model(
            model=model,
            model_name=name,
            data=data,
            models_dir=models_dir,
            training_config=training_config
        )
    
    # Limpiar memoria
    del model
    tf.keras.backend.clear_session()
    
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
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=CONST_DEFAULT_SEED)
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
        
        # Organizar datos en estructura esperada
        data = {
            'train': {'x_cgm': x_cgm_train_fold, 'x_other': x_other_train_fold, 'y': y_train_fold},
            'val': {'x_cgm': x_cgm_val_fold, 'x_other': x_other_val_fold, 'y': y_val_fold},
            'test': {'x_cgm': x_cgm_val_fold, 'x_other': x_other_val_fold, 'y': y_val_fold}
        }
        
        # Entrenar y evaluar modelo
        _, _, metrics = train_and_evaluate_model(
            model=model,
            model_name=f'fold_{fold}',
            data=data,
            models_dir=models_dir
        )
        
        scores.append(metrics)
        
        # Limpiar memoria
        del model
        tf.keras.backend.clear_session()
    
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
        Función que crea el modelo (no utilizada para TensorFlow, pero mantiene interfaz consistente)
    x_cgm : np.ndarray
        Datos CGM para predicción
    x_other : np.ndarray
        Otras características para predicción
    input_shapes : Optional[Tuple[Tuple[int, ...], Tuple[int, ...]]], opcional
        Formas de las entradas. No utilizado en TensorFlow pero mantiene interfaz consistente (default: None)
        
    Retorna:
    --------
    np.ndarray
        Predicciones del modelo
    """
    # Cargar modelo guardado
    model = tf.keras.models.load_model(model_path)
    
    # Realizar predicciones
    y_pred = model.predict([x_cgm, x_other]).flatten()
    
    # Limpiar memoria
    del model
    tf.keras.backend.clear_session()
    
    return y_pred