from typing import Dict, List, Tuple, Any, Optional, Callable, Union
import numpy as np
import jax
import jax.numpy as jnp
from flax.training import train_state
from flax.training.checkpoints import save_checkpoint, restore_checkpoint
import os
import pickle
import tensorflow as tf
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm.auto import tqdm

from constants.constants import CONST_DEFAULT_SEED, CONST_DEFAULT_EPOCHS, CONST_DEFAULT_BATCH_SIZE
from config.models_config import EARLY_STOPPING_POLICY
from custom.model_wrapper import ModelWrapper
from custom.printer import print_debug, print_info, print_warning, print_error, print_success

# Constantes para uso repetido
CONST_ACTOR = "actor"
CONST_CRITIC = "critic"
CONST_TARGET = "target"
CONST_PARAMS = "params"
CONST_DEVICE = "device"
CONST_MODEL_INIT_ERROR = "El modelo debe ser inicializado antes de {}"
CONST_LOSS = "loss"
CONST_VAL_LOSS = "val_loss"


class DRLModelWrapperTF(ModelWrapper):
    """
    Wrapper para modelos de aprendizaje por refuerzo profundo implementados en TensorFlow.
    
    Parámetros:
    -----------
    model_cls : Callable
        Clase del modelo DRL a instanciar
    **model_kwargs
        Argumentos para el constructor del modelo
    """
    
    def __init__(self, model_cls: Callable, **model_kwargs) -> None:
        """
        Inicializa un wrapper para modelos de aprendizaje por refuerzo profundo en TensorFlow.
        
        Parámetros:
        -----------
        model_cls : Callable
            Clase del modelo DRL a instanciar
        **model_kwargs
            Argumentos para el constructor del modelo
        """
        super().__init__()
        self.model_cls = model_cls
        self.model_kwargs = model_kwargs
        self.model = None
        self.buffer = None
        self.algorithm = model_kwargs.get('algorithm', 'generic')
    
    def start(self, x_cgm: np.ndarray, x_other: np.ndarray, y: np.ndarray, 
             rng_key: Any = None) -> Any:
        """
        Inicializa el modelo DRL con los datos de entrada.
        
        Parámetros:
        -----------
        x_cgm : np.ndarray
            Datos CGM de entrada
        x_other : np.ndarray
            Otras características de entrada
        y : np.ndarray
            Valores objetivo
        rng_key : Any, opcional
            Clave para generación aleatoria (default: None)
            
        Retorna:
        --------
        Any
            Estado inicial del modelo o parámetros
        """
        if self.model is None:
            self.model = self.model_cls(**self.model_kwargs)
        
        # Dimensiones del espacio de estados y acciones
        state_dim = (x_cgm.shape[1:], x_other.shape[1:])
        action_dim = 1  # Para regresión en el caso de dosis de insulina
        
        # Inicializar el modelo según su interfaz disponible
        if hasattr(self.model, 'initialize'):
            self.model.initialize(state_dim, action_dim)
        elif hasattr(self.model, 'setup'):
            self.model.setup(state_dim, action_dim)
        elif hasattr(self.model, 'build'):
            # Crear datos dummy para build
            x_cgm_dummy = np.zeros((1,) + x_cgm.shape[1:])
            x_other_dummy = np.zeros((1,) + x_other.shape[1:])
            self.model.build([x_cgm_dummy.shape, x_other_dummy.shape])
        
        # Inicializar buffer de experiencia si el modelo lo requiere
        if hasattr(self.model, 'init_buffer'):
            self.buffer = self.model.init_buffer()
        
        return self.model
    
    def _unpack_validation_data(self, validation_data: Optional[Tuple]) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray]]:
        """
        Desempaqueta los datos de validación si están disponibles.
        
        Parámetros:
        -----------
        validation_data : Optional[Tuple]
            Datos de validación en formato ((x_cgm_val, x_other_val), y_val)
            
        Retorna:
        --------
        Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray]]
            Tupla con (x_cgm_val, x_other_val, y_val)
        """
        x_cgm_val, x_other_val, y_val = None, None, None
        if validation_data is not None:
            (x_cgm_val, x_other_val), y_val = validation_data
        return x_cgm_val, x_other_val, y_val
    
    def _compute_rewards(self, x_cgm: np.ndarray, x_other: np.ndarray, y: np.ndarray) -> np.ndarray:
        """
        Calcula recompensas a partir de los datos y objetivos.
        
        Parámetros:
        -----------
        x_cgm : np.ndarray
            Datos CGM de entrada
        x_other : np.ndarray
            Otras características de entrada
        y : np.ndarray
            Valores objetivo
            
        Retorna:
        --------
        np.ndarray
            Recompensas calculadas para el entrenamiento RL
        """
        # Si el modelo proporciona una función para calcular recompensas, úsala
        if hasattr(self.model, 'compute_rewards'):
            return self.model.compute_rewards(x_cgm, x_other, y)
        
        # Implementación por defecto: recompensa negativa basada en el error
        predicted = self.predict(x_cgm, x_other)
        error = np.abs(predicted - y)
        # Normalizar error al rango [-1, 0] donde -1 es el peor error y 0 es perfecto
        max_error = np.max(error) if np.max(error) > 0 else 1.0
        rewards = -error / max_error
        return rewards
    
    def _fill_buffer(self, x_cgm: np.ndarray, x_other: np.ndarray, rewards: np.ndarray, y: np.ndarray) -> None:
        """
        Llena el buffer de experiencia con transiciones.
        
        Parámetros:
        -----------
        x_cgm : np.ndarray
            Estados (datos CGM)
        x_other : np.ndarray
            Estados (otras características)
        rewards : np.ndarray
            Recompensas calculadas
        y : np.ndarray
            Acciones objetivo (dosis)
        """
        if self.buffer is None or not hasattr(self.model, 'add_to_buffer'):
            return
        
        for i in range(len(rewards)):
            state = (x_cgm[i], x_other[i])
            action = y[i]
            reward = rewards[i]
            # En un entorno supervisado, el siguiente estado puede ser el mismo
            # y done es siempre True (episodio de un paso)
            next_state = (x_cgm[i], x_other[i])
            done = True
            self.model.add_to_buffer(state, action, reward, next_state, done)
    
    def _update_networks(self, batch_size: int, updates_per_batch: int = 1) -> Dict[str, float]:
        """
        Actualiza las redes del modelo usando experiencias del buffer.
        
        Parámetros:
        -----------
        batch_size : int
            Tamaño del lote para actualizaciones
        updates_per_batch : int
            Número de actualizaciones por lote
            
        Retorna:
        --------
        Dict[str, float]
            Diccionario con métricas de actualización
        """
        metrics = {
            "actor_loss": 0.0,
            "critic_loss": 0.0,
            "q_loss": 0.0,
            "total_loss": 0.0
        }
        
        if not hasattr(self.model, 'update'):
            return metrics
        
        for _ in range(updates_per_batch):
            update_metrics = self.model.update(batch_size)
            if isinstance(update_metrics, dict):
                for key, value in update_metrics.items():
                    if key in metrics:
                        metrics[key] += value
            elif isinstance(update_metrics, (int, float)):
                metrics["total_loss"] += update_metrics
        
        # Promediar las métricas
        for key in metrics:
            metrics[key] /= updates_per_batch
            
        return metrics
    
    def _process_epoch_metrics(self, epoch_metrics: Dict[str, float], 
                               updates_per_epoch: int, history: Dict[str, List[float]]) -> float:
        """
        Procesa y registra las métricas de una época.
        
        Parámetros:
        -----------
        epoch_metrics : Dict[str, float]
            Métricas acumuladas de la época
        updates_per_epoch : int
            Número de actualizaciones por época
        history : Dict[str, List[float]]
            Historial de entrenamiento a actualizar
            
        Retorna:
        --------
        float
            Pérdida total calculada
        """
        # Promediar métricas de la época
        for key in epoch_metrics:
            epoch_metrics[key] /= updates_per_epoch
            if key in history:
                history[key].append(epoch_metrics[key])
        
        # Pérdida total para seguimiento
        total_loss = epoch_metrics["total_loss"]
        epsilon = 1e-10  # Pequeño valor para comparación de punto flotante
        if abs(total_loss) < epsilon:
            # Si no hay pérdida total (o es muy cercana a cero), usar la suma de otras pérdidas
            total_loss = sum(epoch_metrics[key] for key in ["actor_loss", "critic_loss", "q_loss"])
        
        return total_loss
    
    def _run_epoch_updates(self, batch_size: int, updates_per_epoch: int) -> Dict[str, float]:
        """
        Ejecuta las actualizaciones durante una época.
        
        Parámetros:
        -----------
        batch_size : int
            Tamaño del lote
        updates_per_epoch : int
            Número de actualizaciones por época
            
        Retorna:
        --------
        Dict[str, float]
            Métricas acumuladas de la época
        """
        epoch_metrics = {key: 0.0 for key in ["actor_loss", "critic_loss", "q_loss", "total_loss"]}
        
        for _ in range(updates_per_epoch):
            batch_metrics = self._update_networks(batch_size)
            for key, value in batch_metrics.items():
                epoch_metrics[key] += value
                
        return epoch_metrics
    
    def _validate_model(self, x_cgm_val: np.ndarray, x_other_val: np.ndarray, 
                       y_val: np.ndarray, history: Dict[str, List[float]]) -> None:
        """
        Realiza validación del modelo y registra la pérdida.
        
        Parámetros:
        -----------
        x_cgm_val : np.ndarray
            Datos CGM de validación
        x_other_val : np.ndarray
            Otras características de validación
        y_val : np.ndarray
            Valores objetivo de validación
        history : Dict[str, List[float]]
            Historial donde registrar la pérdida de validación
        """
        if x_cgm_val is not None and y_val is not None:
            val_preds = self.predict(x_cgm_val, x_other_val)
            val_loss = float(np.mean((val_preds - y_val) ** 2))
            history["val_loss"].append(val_loss)
        
    def fit(self, x_cgm: np.ndarray, x_other: np.ndarray, y: np.ndarray, 
           validation_data: Optional[Tuple[Tuple[np.ndarray, np.ndarray], np.ndarray]] = None,
           epochs: int = CONST_DEFAULT_EPOCHS, batch_size: int = CONST_DEFAULT_BATCH_SIZE, verbose: int = 1) -> Dict[str, List[float]]:
        """
        Entrena el modelo DRL con los datos proporcionados.
        
        Parámetros:
        -----------
        x_cgm : np.ndarray
            Datos CGM de entrenamiento
        x_other : np.ndarray
            Otras características de entrenamiento
        y : np.ndarray
            Valores objetivo (acciones)
        validation_data : Optional[Tuple[Tuple[np.ndarray, np.ndarray], np.ndarray]], opcional
            Datos de validación como ((x_cgm_val, x_other_val), y_val) (default: None)
        epochs : int, opcional
            Número de épocas de entrenamiento (default: 10)
        batch_size : int, opcional
            Tamaño de lote (default: 32)
        verbose : int, opcional
            Nivel de verbosidad (0=silencioso, 1=progreso, 2=detallado)
            
        Retorna:
        --------
        Dict[str, List[float]]
            Historial de entrenamiento con métricas
        """
        # Preparar datos de validación
        x_cgm_val, x_other_val, y_val = self._unpack_validation_data(validation_data)
        
        # Historial de entrenamiento
        history = {
            "loss": [],
            "actor_loss": [],
            "critic_loss": [],
            "q_loss": [],
            "val_loss": []
        }
        
        # Si el modelo tiene un método fit nativo, úsalo
        if hasattr(self.model, 'fit'):
            return self._train_with_fit(
                x_cgm, x_other, y,
                (x_cgm_val, x_other_val, y_val) if x_cgm_val is not None else None,
                epochs, batch_size
            )
        
        # Entrenamiento RL personalizado
        for _ in range(epochs):
            # Calcular recompensas y llenar buffer
            rewards = self._compute_rewards(x_cgm, x_other, y)
            self._fill_buffer(x_cgm, x_other, rewards, y)
            
            # Actualizar redes y procesar métricas
            updates_per_epoch = max(1, len(x_cgm) // batch_size)
            epoch_metrics = self._run_epoch_updates(batch_size, updates_per_epoch)
            total_loss = self._process_epoch_metrics(epoch_metrics, updates_per_epoch, history)
            history["loss"].append(total_loss)
            
            # Validación
            self._validate_model(x_cgm_val, x_other_val, y_val, history)
        
        return history
    
    def _train_with_fit(self, x_cgm: np.ndarray, x_other: np.ndarray, y: np.ndarray,
                      validation_data: Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]],
                      epochs: int, batch_size: int) -> Dict[str, List[float]]:
        """
        Entrena el modelo usando su método fit nativo.
        
        Parámetros:
        -----------
        x_cgm : np.ndarray
            Datos CGM de entrenamiento
        x_other : np.ndarray
            Otras características de entrenamiento
        y : np.ndarray
            Valores objetivo (acciones)
        validation_data : Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]]
            Datos de validación
        epochs : int
            Número de épocas
        batch_size : int
            Tamaño de lote
            
        Retorna:
        --------
        Dict[str, List[float]]
            Historial de entrenamiento
        """
        history = {"loss": [], "val_loss": []}
        fit_history = self.model.fit(
            [x_cgm, x_other], y,
            validation_data=validation_data,
            epochs=epochs,
            batch_size=batch_size
        )
        
        # Copiar el historial del modelo
        for key, values in fit_history.history.items():
            history[key] = values
            
        return history
    
    def predict(self, x_cgm: np.ndarray, x_other: np.ndarray) -> np.ndarray:
        """
        Realiza predicciones con el modelo entrenado.
        
        Parámetros:
        -----------
        x_cgm : np.ndarray
            Datos CGM para predicción
        x_other : np.ndarray
            Otras características para predicción
            
        Retorna:
        --------
        np.ndarray
            Predicciones del modelo (acciones)
        """
        # Si el modelo tiene un método predict directo, úsalo
        if hasattr(self.model, 'predict'):
            return self.model.predict([x_cgm, x_other])
        
        # Si el modelo tiene un método act, úsalo (común en DRL)
        elif hasattr(self.model, 'act') or hasattr(self.model, 'get_action'):
            return self._predict_with_act(x_cgm, x_other)
        
        # Fallback para otros casos
        else:
            # Predicción uno por uno, ya que los modelos DRL suelen estar diseñados para estados individuales
            predictions = np.zeros((len(x_cgm),), dtype=np.float32)
            for i in range(len(x_cgm)):
                state = (x_cgm[i:i+1], x_other[i:i+1])
                if hasattr(self.model, '__call__'):
                    predictions[i] = self.model(state)
                else:
                    # Si no hay forma clara de predecir, devolver ceros
                    pass
            return predictions
    
    def _predict_with_act(self, x_cgm: np.ndarray, x_other: np.ndarray) -> np.ndarray:
        """
        Realiza predicciones usando el método act o get_action del modelo.
        
        Parámetros:
        -----------
        x_cgm : np.ndarray
            Datos CGM para predicción
        x_other : np.ndarray
            Otras características para predicción
            
        Retorna:
        --------
        np.ndarray
            Predicciones del modelo (acciones)
        """
        predictions = np.zeros((len(x_cgm),), dtype=np.float32)
        
        act_method = getattr(self.model, 'act', None) or getattr(self.model, 'get_action', None)
        
        for i in range(len(x_cgm)):
            state = (x_cgm[i:i+1], x_other[i:i+1])
            # Los modelos DRL suelen tener un parámetro deterministic para inferencia
            if 'deterministic' in act_method.__code__.co_varnames:
                predictions[i] = act_method(state, deterministic=True)
            else:
                predictions[i] = act_method(state)
                
        return predictions
    
    def save(self, path: str) -> None:
        """
        Guarda el estado del modelo DRL JAX en disco.

        Parámetros:
        -----------
        path : str
            Ruta donde guardar el modelo
    
        Retorna:
        --------
        None
        """
        # Verificar si el modelo está inicializado
        if self.model is None:
            raise ValueError(CONST_MODEL_INIT_ERROR.format("guardar"))
        
        # Crear directorio si es necesario
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)

        try:
            # Intentar guardar usando el método del modelo subyacente si existe
            if hasattr(self.model, 'save'):
                self.model.save(path)
                print_success(f"Modelo guardado usando método nativo en: {path}")
                return
                
            # Determinar qué estado guardar
            state_to_save = None
            if self.params is not None:
                state_to_save = self.params
            elif self.state is not None:
                state_to_save = self.state
            elif self.states:  # Si hay estados (actor, crítico, etc.)
                state_to_save = self.states
                
            # Intentar guardar con checkpoint de Flax
            if state_to_save is not None:
                try:
                    save_checkpoint(path, state_to_save, step=0, keep=1)
                    print_success(f"Estado del modelo guardado en: {path}")
                    return
                except Exception as e:
                    print_warning(f"No se pudo guardar con checkpoint de Flax: {e}")
                    
            # Fallback: guardar con pickle
            print_warning("Guardando con pickle como alternativa.")
            save_data = {
                'params': self.params,
                'state': self.state,
                'states': self.states,
                'model_kwargs': self.model_kwargs,
                'algorithm': self.algorithm,
                'rng_key': self.rng_key
            }
                
            # Asegurar que no haya duplicación de extensión
            save_path = path if path.endswith('.pkl') else f"{path}.pkl"
            
            with open(save_path, 'wb') as f:
                pickle.dump(save_data, f)
            print_success(f"Estado del modelo guardado como pickle en: {save_path}")
                
        except Exception as e:
            print_error(f"Error al guardar modelo: {e}")
            
            # Último recurso - intentar guardar solo información básica
            try:
                basic_info = {
                    'model_kwargs': self.model_kwargs,
                    'algorithm': self.algorithm
                }
                
                fallback_path = f"{path}_info.pkl"
                with open(fallback_path, 'wb') as f:
                    pickle.dump(basic_info, f)
                print_warning(f"No se pudo guardar el modelo completo, pero la información básica fue guardada en: {fallback_path}")
            except Exception as inner_e:
                print_error(f"No se pudo guardar ni siquiera la información básica: {inner_e}")
        
    def load(self, path: str) -> None:
        """
        Carga el modelo desde disco.
        
        Parámetros:
        -----------
        path : str
            Ruta desde donde cargar el modelo
        
        Retorna:
        --------
        None
        """
        # Verificar si es un archivo pickle
        if path.endswith('.pkl'):
            self._load_from_pickle(path)
        # Verificar si es un archivo de pesos
        elif path.endswith('_weights.h5') or path.endswith('.h5'):
            self._load_weights(path)
        # Intentar cargar como modelo completo
        elif os.path.isdir(path) or path.endswith('.keras'):
            self._load_complete_model(path)
        else:
            raise ValueError(f"Formato de archivo no soportado: {path}")
            
    def _load_from_pickle(self, path: str) -> None:
        """
        Carga el modelo desde un archivo pickle.
        
        Parámetros:
        -----------
        path : str
            Ruta del archivo pickle
        
        Retorna:
        --------
        None
        """
        try:
            with open(path, 'rb') as f:
                data = pickle.load(f)
            
            # Recrear modelo si es necesario
            if self.model is None:
                self._create_model_instance((1,), (1,))
            
            # Establecer pesos si están disponibles
            if 'weights' in data and data['weights'] is not None and hasattr(self.model, 'set_weights'):
                self.model.set_weights(data['weights'])
                print_success(f"Pesos cargados desde {path}")
            else:
                print_warning("No se encontraron pesos en el archivo pickle o el modelo no soporta set_weights")
        except Exception as e:
            print_error(f"Error al cargar desde pickle: {e}")
            
    def _load_weights(self, path: str) -> None:
        """
        Carga los pesos del modelo desde un archivo h5.
        
        Parámetros:
        -----------
        path : str
            Ruta del archivo de pesos
        
        Retorna:
        --------
        None
        """
        try:
            # Asegurar que el modelo existe
            if self.model is None:
                self._create_model_instance((1,), (1,))
                
            # Cargar pesos
            if hasattr(self.model, 'load_weights'):
                self.model.load_weights(path)
                print_success(f"Pesos cargados desde {path}")
            else:
                print_warning("El modelo no tiene método load_weights")
        except Exception as e:
            print_error(f"Error al cargar pesos: {e}")
            
    def _load_complete_model(self, path: str) -> None:
        """
        Carga el modelo completo.
        
        Parámetros:
        -----------
        path : str
            Ruta del modelo completo
        
        Retorna:
        --------
        None
        """
        try:
            self.model = tf.keras.models.load_model(path)
            print_success(f"Modelo cargado desde {path}")
        except Exception as e:
            print_error(f"Error al cargar el modelo completo: {e}")
            print_warning("Intentando cargar como pesos...")
            self._load_weights(path)


class DRLModelWrapperJAX(ModelWrapper):
    """
    Wrapper para modelos de aprendizaje por refuerzo profundo implementados en JAX.
    
    Parámetros:
    -----------
    model_cls : Callable
        Clase del modelo DRL a instanciar
    **model_kwargs
        Argumentos para el constructor del modelo
    """
    
    def __init__(self, model_cls: Callable, **model_kwargs) -> None:
        """
        Inicializa un wrapper para modelos de aprendizaje por refuerzo profundo en JAX.
        
        Parámetros:
        -----------
        model_cls : Callable
            Clase del modelo DRL a instanciar
        **model_kwargs
            Argumentos para el constructor del modelo
        """
        super().__init__()
        self.model_cls = model_cls
        self.model_kwargs = model_kwargs
        self.model = None
        self.buffer = None
        self.algorithm = model_kwargs.get('algorithm', 'generic')
        self.rng_key = jax.random.PRNGKey(0)
        self.params = None
        self.state = None
        self.states = {}  # Múltiples estados (actor, crítico, etc.)
        
        self.early_stopping_config = {
            'patience': EARLY_STOPPING_POLICY['early_stopping_patience'],
            'min_delta': EARLY_STOPPING_POLICY['early_stopping_min_delta'],
            'restore_best_weights': EARLY_STOPPING_POLICY['early_stopping_restore_best_weights'],
            'best_val_loss': EARLY_STOPPING_POLICY['early_stopping_best_val_loss'],
            'counter': EARLY_STOPPING_POLICY['early_stopping_counter'],
            'best_weights': EARLY_STOPPING_POLICY['early_stopping_best_weights']
        }
    
    def start(self, x_cgm: np.ndarray, x_other: np.ndarray, y: np.ndarray, 
             rng_key: Any = None) -> Any:
        """
        Inicializa el modelo DRL con los datos de entrada.
        
        Parámetros:
        -----------
        x_cgm : np.ndarray
            Datos CGM de entrada
        x_other : np.ndarray
            Otras características de entrada
        y : np.ndarray
            Valores objetivo
        rng_key : Any, opcional
            Clave para generación aleatoria (default: None)
            
        Retorna:
        --------
        Any
            Estado inicial del modelo o parámetros
        """
        if rng_key is not None:
            self.rng_key = rng_key
            
        if self.model is None:
            self.model = self.model_cls(**self.model_kwargs)
        
        # Dimensiones del espacio de estados y acciones
        state_dim = (x_cgm.shape[1:], x_other.shape[1:])
        action_dim = 1  # Para regresión en el caso de dosis de insulina
        
        # Inicializar según el tipo de algoritmo
        if self.algorithm in ['ppo', 'a2c', 'a3c', 'sac', 'trpo']:
            self._initialize_actor_critic(state_dim, action_dim)
        elif self.algorithm in ['dqn', 'ddpg', 'td3']:
            self._initialize_q_networks(state_dim, action_dim)
        else:
            # Inicialización genérica
            self._initialize_generic(state_dim, action_dim)
        
        return self.params or self.state or self.states
    
    def _initialize_actor_critic(self, state_dim: Tuple, action_dim: int) -> None:
        """
        Inicializa redes de actor y crítico para algoritmos basados en política.
        
        Parámetros:
        -----------
        state_dim : Tuple
            Dimensiones del espacio de estados
        action_dim : int
            Dimensión del espacio de acciones
        """
        # Crear claves para inicialización
        self.rng_key, actor_key, critic_key = jax.random.split(self.rng_key, 3)
        
        # Formas de entrada para inicialización
        cgm_shape, other_shape = state_dim
        x_cgm_dummy = jnp.ones((1,) + cgm_shape)
        x_other_dummy = jnp.ones((1,) + other_shape)
        
        # Inicializar actor
        if hasattr(self.model, 'init_actor'):
            self.states[CONST_ACTOR] = self.model.init_actor(
                actor_key, x_cgm_dummy, x_other_dummy
            )
        
        # Inicializar crítico
        if hasattr(self.model, 'init_critic'):
            self.states[CONST_CRITIC] = self.model.init_critic(
                critic_key, x_cgm_dummy, x_other_dummy
            )
        
        # Inicializar target networks si son necesarias
        if hasattr(self.model, 'init_target_networks'):
            self.rng_key, target_key = jax.random.split(self.rng_key)
            self.states[CONST_TARGET] = self.model.init_target_networks(
                target_key, self.states[CONST_ACTOR], self.states[CONST_CRITIC]
            )
    
    def _initialize_q_networks(self, state_dim: Tuple, action_dim: int) -> None:
        """
        Inicializa redes Q para algoritmos basados en valor.
        
        Parámetros:
        -----------
        state_dim : Tuple
            Dimensiones del espacio de estados
        action_dim : int
            Dimensión del espacio de acciones
        """
        # Crear claves para inicialización
        self.rng_key, q_key, target_key = jax.random.split(self.rng_key, 3)
        
        # Formas de entrada para inicialización
        cgm_shape, other_shape = state_dim
        x_cgm_dummy = jnp.ones((1,) + cgm_shape)
        x_other_dummy = jnp.ones((1,) + other_shape)
        
        # Inicializar Q-network
        if hasattr(self.model, 'init_q_network'):
            self.states['q_network'] = self.model.init_q_network(
                q_key, x_cgm_dummy, x_other_dummy
            )
        
        # Inicializar target Q-network
        if hasattr(self.model, 'init_target_q_network'):
            self.states['target_q_network'] = self.model.init_target_q_network(
                target_key, self.states['q_network']
            )
        
        # Para DDPG/TD3, también inicializar el actor
        if self.algorithm in ['ddpg', 'td3'] and hasattr(self.model, 'init_actor'):
            self.rng_key, actor_key = jax.random.split(self.rng_key)
            self.states[CONST_ACTOR] = self.model.init_actor(
                actor_key, x_cgm_dummy, x_other_dummy
            )
    
    def _initialize_generic(self, state_dim: Tuple, action_dim: int) -> None:
        """
        Inicialización genérica para otros tipos de modelos DRL.
        
        Parámetros:
        -----------
        state_dim : Tuple
            Dimensiones del espacio de estados
        action_dim : int
            Dimensión del espacio de acciones
        """
        # Crear clave para inicialización
        self.rng_key, init_key = jax.random.split(self.rng_key)
        
        # Formas de entrada para inicialización
        cgm_shape, other_shape = state_dim
        x_cgm_dummy = jnp.ones((1,) + cgm_shape)
        x_other_dummy = jnp.ones((1,) + other_shape)
        
        # Inicializar modelo
        if hasattr(self.model, 'init'):
            self.params = self.model.init(
                init_key, x_cgm_dummy, x_other_dummy
            )
        elif hasattr(self.model, 'initialize'):
            self.state = self.model.initialize(
                init_key, x_cgm_dummy, x_other_dummy
            )
    
    def _compute_rewards(self, x_cgm: np.ndarray, x_other: np.ndarray, y: np.ndarray) -> np.ndarray:
        """
        Calcula recompensas a partir de los datos y objetivos.
        
        Parámetros:
        -----------
        x_cgm : np.ndarray
            Datos CGM de entrada
        x_other : np.ndarray
            Otras características de entrada
        y : np.ndarray
            Valores objetivo
            
        Retorna:
        --------
        np.ndarray
            Recompensas calculadas para el entrenamiento RL
        """
        # Si el modelo proporciona una función para calcular recompensas, úsala
        if hasattr(self.model, 'compute_rewards'):
            self.rng_key, reward_key = jax.random.split(self.rng_key)
            return self.model.compute_rewards(
                self.params or self.state or self.states,
                jnp.array(x_cgm), jnp.array(x_other), jnp.array(y),
                reward_key
            )
        
        # Implementación por defecto: recompensa negativa basada en el error
        predicted = self.predict(x_cgm, x_other)
        error = np.abs(predicted - y)
        # Normalizar error al rango [-1, 0] donde -1 es el peor error y 0 es perfecto
        max_error = np.max(error) if np.max(error) > 0 else 1.0
        rewards = -error / max_error
        return rewards
    
    def _update_networks(self, batch: Tuple) -> Dict[str, float]:
        """
        Actualiza las redes del modelo usando una muestra del buffer.
        
        Parámetros:
        -----------
        batch : Tuple
            Lote de experiencias del buffer
            
        Retorna:
        --------
        Dict[str, float]
            Diccionario con métricas de actualización
        """
        # Actualizar según el tipo de algoritmo
        if self.algorithm in ['ppo', 'a2c', 'a3c', 'sac', 'trpo']:
            return self._update_actor_critic(batch)
        elif self.algorithm in ['dqn', 'ddpg', 'td3']:
            return self._update_q_networks(batch)
        else:
            return self._update_generic(batch)
    
    def _update_actor_critic(self, batch: Tuple) -> Dict[str, float]:
        """
        Actualiza las redes de actor y crítico.
        
        Parámetros:
        -----------
        batch : Tuple
            Lote de experiencias del buffer
            
        Retorna:
        --------
        Dict[str, float]
            Métricas de actualización
        """
        metrics = {"actor_loss": 0.0, "critic_loss": 0.0, "total_loss": 0.0}
        
        if not hasattr(self.model, 'update_actor_critic'):
            return metrics
        
        self.rng_key, update_key = jax.random.split(self.rng_key)
        actor_state, critic_state, actor_loss, critic_loss = self.model.update_actor_critic(
            self.states[CONST_ACTOR], self.states[CONST_CRITIC], batch, update_key
        )
        
        self.states[CONST_ACTOR] = actor_state
        self.states[CONST_CRITIC] = critic_state
        metrics["actor_loss"] = float(actor_loss)
        metrics["critic_loss"] = float(critic_loss)
        metrics["total_loss"] = float(actor_loss) + float(critic_loss)
        
        # Actualizar target networks si existe el método
        if hasattr(self.model, 'update_target_networks'):
            self.states[CONST_TARGET] = self.model.update_target_networks(
                self.states[CONST_ACTOR], self.states[CONST_CRITIC], self.states[CONST_TARGET]
            )
            
        return metrics
    
    def _update_q_networks(self, batch: Tuple) -> Dict[str, float]:
        """
        Actualiza las redes Q.
        
        Parámetros:
        -----------
        batch : Tuple
            Lote de experiencias del buffer
            
        Retorna:
        --------
        Dict[str, float]
            Métricas de actualización
        """
        metrics = {"q_loss": 0.0, "actor_loss": 0.0, "total_loss": 0.0}
        
        if not hasattr(self.model, 'update_q_network'):
            return metrics
        
        self.rng_key, update_key = jax.random.split(self.rng_key)
        q_state, q_loss = self.model.update_q_network(
            self.states['q_network'], self.states.get('target_q_network'), batch, update_key
        )
        
        self.states['q_network'] = q_state
        metrics["q_loss"] = float(q_loss)
        metrics["total_loss"] = float(q_loss)
        
        # Para DDPG/TD3, también actualizar el actor
        if self.algorithm in ['ddpg', 'td3'] and hasattr(self.model, 'update_actor'):
            self.rng_key, actor_key = jax.random.split(self.rng_key)
            actor_state, actor_loss = self.model.update_actor(
                self.states[CONST_ACTOR], self.states['q_network'], batch, actor_key
            )
            self.states[CONST_ACTOR] = actor_state
            metrics["actor_loss"] = float(actor_loss)
            metrics["total_loss"] += float(actor_loss)
        
        # Actualizar target networks si existe el método
        if hasattr(self.model, 'update_target_networks'):
            target_q = self.model.update_target_networks(
                self.states['q_network'], self.states.get('target_q_network')
            )
            if target_q is not None:
                self.states['target_q_network'] = target_q
            
        return metrics
    
    def _update_generic(self, batch: Tuple) -> Dict[str, float]:
        """
        Actualización genérica para otros tipos de modelos DRL.
        
        Parámetros:
        -----------
        batch : Tuple
            Lote de experiencias del buffer
            
        Retorna:
        --------
        Dict[str, float]
            Métricas de actualización
        """
        metrics = {"total_loss": 0.0}
        
        if hasattr(self.model, 'update'):
            self.rng_key, update_key = jax.random.split(self.rng_key)
            
            if self.params is not None:
                self.params, loss = self.model.update(self.params, batch, update_key)
                metrics["total_loss"] = float(loss)
            elif self.state is not None:
                self.state, loss = self.model.update(self.state, batch, update_key)
                metrics["total_loss"] = float(loss)
                
        return metrics
    
    def _fill_buffer(self, x_cgm: jnp.ndarray, x_other: jnp.ndarray, rewards: jnp.ndarray, y: jnp.ndarray) -> None:
        """
        Llena el buffer de experiencia con transiciones.
        
        Parámetros:
        -----------
        x_cgm : jnp.ndarray
            Estados (datos CGM)
        x_other : jnp.ndarray
            Estados (otras características)
        rewards : jnp.ndarray
            Recompensas calculadas
        y : jnp.ndarray
            Acciones objetivo (dosis)
        """
        if self.buffer is None or not hasattr(self.model, 'add_to_buffer'):
            return
        
        for i in range(len(rewards)):
            state = (x_cgm[i], x_other[i])
            action = y[i]
            reward = rewards[i]
            # En un entorno supervisado, el siguiente estado puede ser el mismo
            # y done es siempre True (episodio de un paso)
            next_state = (x_cgm[i], x_other[i])
            done = True
            
            self.rng_key, add_key = jax.random.split(self.rng_key)
            self.model.add_to_buffer(self.buffer, state, action, reward, next_state, done, add_key)
    
    def _unpack_validation_data(self, validation_data: Optional[Tuple]) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray]]:
        """
        Desempaqueta los datos de validación si están disponibles.
        
        Parámetros:
        -----------
        validation_data : Optional[Tuple]
            Datos de validación en formato ((x_cgm_val, x_other_val), y_val)
            
        Retorna:
        --------
        Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray]]
            Tupla con (x_cgm_val, x_other_val, y_val)
        """
        x_cgm_val, x_other_val, y_val = None, None, None
        if validation_data is not None:
            (x_cgm_val, x_other_val), y_val = validation_data
        return x_cgm_val, x_other_val, y_val
    
    def _prepare_training_data(self, x_cgm: np.ndarray, x_other: np.ndarray, y: np.ndarray, 
                              validation_data: Optional[Tuple]) -> Tuple:
        """
        Prepara los datos para el entrenamiento.
        
        Parámetros:
        -----------
        x_cgm : np.ndarray
            Datos CGM de entrenamiento
        x_other : np.ndarray
            Otras características de entrenamiento
        y : np.ndarray
            Valores objetivo (acciones)
        validation_data : Optional[Tuple]
            Datos de validación
            
        Retorna:
        --------
        Tuple
            Datos preparados para entrenamiento
        """
        # Convertir datos a jnp.array
        x_cgm_arr = jnp.array(x_cgm)
        x_other_arr = jnp.array(x_other)
        y_arr = jnp.array(y)
        
        # Preparar datos de validación
        x_cgm_val, x_other_val, y_val = self._unpack_validation_data(validation_data)
        x_cgm_val_arr = jnp.array(x_cgm_val) if x_cgm_val is not None else None
        x_other_val_arr = jnp.array(x_other_val) if x_other_val is not None else None
        y_val_arr = jnp.array(y_val) if y_val is not None else None
        
        return (x_cgm_arr, x_other_arr, y_arr, 
                x_cgm_val, x_other_val, y_val,
                x_cgm_val_arr, x_other_val_arr, y_val_arr)
    
    def _get_batch_from_buffer(self, x_cgm_arr: jnp.ndarray, x_other_arr: jnp.ndarray, 
                               y_arr: jnp.ndarray, rewards: jnp.ndarray, batch_size: int) -> Tuple:
        """
        Obtiene un lote del buffer o genera uno aleatorio.
        
        Parámetros:
        -----------
        x_cgm_arr : jnp.ndarray
            Datos CGM
        x_other_arr : jnp.ndarray
            Otras características
        y_arr : jnp.ndarray
            Valores objetivo
        rewards : jnp.ndarray
            Recompensas
        batch_size : int
            Tamaño del lote
            
        Retorna:
        --------
        Tuple
            Lote para entrenamiento
        """
        self.rng_key, sample_key = jax.random.split(self.rng_key)
        
        if hasattr(self.model, 'sample_buffer'):
            return self.model.sample_buffer(self.buffer, batch_size, sample_key)
        
        # Si no hay método sample_buffer, crear un lote aleatorio de índices
        indices = jax.random.choice(
            sample_key, jnp.arange(len(x_cgm_arr)), (batch_size,), replace=True
        )
        return (
            x_cgm_arr[indices], 
            x_other_arr[indices], 
            y_arr[indices], 
            rewards[indices]
        )
    
    def _process_epoch_metrics(self, epoch_metrics: Dict[str, float], 
                               updates_per_epoch: int, history: Dict[str, List[float]]) -> float:
        """
        Procesa y registra las métricas de una época.
        
        Parámetros:
        -----------
        epoch_metrics : Dict[str, float]
            Métricas acumuladas de la época
        updates_per_epoch : int
            Número de actualizaciones por época
        history : Dict[str, List[float]]
            Historial de entrenamiento a actualizar
            
        Retorna:
        --------
        float
            Pérdida total calculada
        """
        # Promediar métricas de la época
        for key in epoch_metrics:
            epoch_metrics[key] /= updates_per_epoch
            if key in history:
                history[key].append(epoch_metrics[key])
        
        # Pérdida total para seguimiento
        total_loss = epoch_metrics["total_loss"]
        epsilon = 1e-10  # Pequeño valor para comparación de punto flotante
        
        if abs(total_loss) < epsilon:
            # Si no hay pérdida total (o es muy cercana a cero), usar la suma de otras pérdidas
            total_loss = sum(epoch_metrics[key] for key in ["actor_loss", "critic_loss", "q_loss"])
        
        return total_loss
    
    def _run_epoch_updates(self, x_cgm_arr: jnp.ndarray, x_other_arr: jnp.ndarray, 
                          y_arr: jnp.ndarray, rewards: jnp.ndarray, 
                          batch_size: int, updates_per_epoch: int, verbose: int = 1) -> Dict[str, float]:
        """
        Ejecuta las actualizaciones durante una época.
        
        Parámetros:
        -----------
        x_cgm_arr : jnp.ndarray
            Datos CGM
        x_other_arr : jnp.ndarray
            Otras características
        y_arr : jnp.ndarray
            Valores objetivo
        rewards : jnp.ndarray
            Recompensas
        batch_size : int
            Tamaño del lote
        updates_per_epoch : int
            Número de actualizaciones por época
        verbose : int, opcional
            Nivel de verbosidad (0=silencioso, 1=progreso) (default: 1)
            
        Retorna:
        --------
        Dict[str, float]
            Métricas acumuladas de la época
        """
        epoch_metrics = {key: 0.0 for key in ["actor_loss", "critic_loss", "q_loss", "total_loss"]}
        
        # Crear iterador con barra de progreso
        iterator = tqdm(
            range(updates_per_epoch), 
            desc="Procesando lotes", 
            leave=False,  # No dejar la barra al terminar
            disable=verbose == 0,  # Deshabilitar si verbose=0
            unit="batch"
        )
        
        # Iterar con la barra de progreso
        for _ in iterator:
            # Obtener un lote del buffer
            batch = self._get_batch_from_buffer(x_cgm_arr, x_other_arr, y_arr, rewards, batch_size)
            
            # Actualizar redes
            batch_metrics = self._update_networks(batch)
            
            # Actualizar métricas acumuladas
            for key, value in batch_metrics.items():
                epoch_metrics[key] += value
            
            # Actualizar la barra de progreso con métricas actuales
            if verbose > 0:
                postfix = {
                    "loss": f"{batch_metrics.get('total_loss', 0.0):.4f}"
                }
                # Use an epsilon value for floating point comparisons
                epsilon = 1e-10
                if abs(batch_metrics.get('actor_loss', 0.0)) > epsilon:
                    postfix["actor"] = f"{batch_metrics.get('actor_loss', 0.0):.4f}"
                if abs(batch_metrics.get('critic_loss', 0.0)) > epsilon:
                    postfix["critic"] = f"{batch_metrics.get('critic_loss', 0.0):.4f}"
                iterator.set_postfix(**postfix)
                
        return epoch_metrics
    
    def add_early_stopping(self, patience: int = 10, min_delta: float = 0.0, restore_best_weights: bool = True) -> None:
        """
        Agrega early stopping al modelo.

        Parámetros:
        -----------
        patience : int, opcional
            Número de épocas a esperar para detener el entrenamiento (default: 10)
        min_delta : float, opcional
            Mínima mejora requerida para considerar una mejora (default: 0.0)
        restore_best_weights : bool, opcional
            Si restaurar los mejores pesos al finalizar (default: True)
        """
        if hasattr(self.wrapper, 'add_early_stopping'):
            self.wrapper.add_early_stopping(patience, min_delta, restore_best_weights)
        else:
            super().add_early_stopping(patience, min_delta, restore_best_weights)
    
    def train(self, x_cgm: np.ndarray, x_other: np.ndarray, y: np.ndarray, 
             validation_data: Optional[Tuple[Tuple[np.ndarray, np.ndarray], np.ndarray]] = None,
             epochs: int = CONST_DEFAULT_EPOCHS, batch_size: int = CONST_DEFAULT_BATCH_SIZE, verbose: int = 1) -> Dict[str, List[float]]:
        """
        Entrena el modelo DRL con los datos proporcionados.
        
        Parámetros:
        -----------
        x_cgm : np.ndarray
            Datos CGM de entrenamiento
        x_other : np.ndarray
            Otras características de entrenamiento
        y : np.ndarray
            Valores objetivo (acciones)
        validation_data : Optional[Tuple[Tuple[np.ndarray, np.ndarray], np.ndarray]], opcional
            Datos de validación como ((x_cgm_val, x_other_val), y_val) (default: None)
        epochs : int, opcional
            Número de épocas de entrenamiento (default: 10)
        batch_size : int, opcional
            Tamaño de lote (default: 32)
        verbose : int, opcional
            Nivel de verbosidad (0=silencioso, 1=progreso, 2=detallado)
            
        Retorna:
        --------
        Dict[str, List[float]]
            Historial de entrenamiento con métricas
        """
        # Preparar los datos
        (x_cgm_arr, x_other_arr, y_arr, 
         x_cgm_val, x_other_val, y_val,
         x_cgm_val_arr, x_other_val_arr, y_val_arr) = self._prepare_training_data(
            x_cgm, x_other, y, validation_data
        )
        
        # Inicializar historial
        history = {
            "loss": [],
            "actor_loss": [],
            "critic_loss": [],
            "q_loss": [],
            "val_loss": []
        }
        
        # Si el modelo tiene un método train directo, úsalo
        if hasattr(self.model, 'train'):
            return self.model.train(
                self.params or self.state or self.states,
                x_cgm_arr, x_other_arr, y_arr,
                (x_cgm_val_arr, x_other_val_arr, y_val_arr) if x_cgm_val is not None else None,
                epochs, batch_size, verbose=verbose
            )
        
        # Inicializar buffer si es necesario
        if hasattr(self.model, 'init_buffer') and self.buffer is None:
            self.rng_key, buffer_key = jax.random.split(self.rng_key)
            self.buffer = self.model.init_buffer(buffer_key)
        
        # Entrenamiento por épocas
        for _ in range(epochs):
            self.rng_key, _ = jax.random.split(self.rng_key)
            
            # Calcular recompensas y llenar buffer
            rewards = self._compute_rewards(x_cgm, x_other, y)
            self._fill_buffer(x_cgm_arr, x_other_arr, rewards, y_arr)
            
            # Ejecutar actualizaciones durante la época
            updates_per_epoch = max(1, len(x_cgm) // batch_size)
            epoch_metrics = self._run_epoch_updates(
                x_cgm_arr, x_other_arr, y_arr, rewards, batch_size, updates_per_epoch, verbose=verbose
            )
            
            # Procesar métricas y registrarlas en el historial
            total_loss = self._process_epoch_metrics(epoch_metrics, updates_per_epoch, history)
            history["loss"].append(total_loss)
            
            # Validación si hay datos disponibles
            if x_cgm_val is not None and y_val is not None:
                val_preds = self.predict(x_cgm_val, x_other_val)
                val_loss = float(np.mean((val_preds - y_val) ** 2))
                history["val_loss"].append(val_loss)
        
        return history
    
    def predict(self, x_cgm: np.ndarray, x_other: np.ndarray) -> np.ndarray:
        """
        Realiza predicciones con el modelo entrenado.
        
        Parámetros:
        -----------
        x_cgm : np.ndarray
            Datos CGM para predicción
        x_other : np.ndarray
            Otras características para predicción
            
        Retorna:
        --------
        np.ndarray
            Predicciones del modelo (acciones)
        """
        # Convertir a jnp.array para procesamiento JAX
        x_cgm_arr = jnp.array(x_cgm)
        x_other_arr = jnp.array(x_other)
        
        # Según el tipo de algoritmo, usar el método apropiado
        if self.algorithm in ['ppo', 'a2c', 'a3c', 'sac', 'trpo', 'ddpg', 'td3']:
            return self._predict_with_actor(x_cgm_arr, x_other_arr)
        elif self.algorithm in ['dqn']:
            return self._predict_with_q(x_cgm_arr, x_other_arr)
        else:
            return self._predict_generic(x_cgm_arr, x_other_arr)
    
    def _predict_with_actor(self, x_cgm: jnp.ndarray, x_other: jnp.ndarray) -> np.ndarray:
        """
        Predice usando la red del actor.
        
        Parámetros:
        -----------
        x_cgm : jnp.ndarray
            Datos CGM para predicción
        x_other : jnp.ndarray
            Otras características para predicción
            
        Retorna:
        --------
        np.ndarray
            Predicciones del actor
        """
        if not hasattr(self.model, 'predict_with_actor'):
            return np.zeros((len(x_cgm),), dtype=np.float32)
            
        self.rng_key, predict_key = jax.random.split(self.rng_key)
        actions = self.model.predict_with_actor(
            self.states[CONST_ACTOR], x_cgm, x_other, predict_key, deterministic=True
        )
        return np.array(actions)
    
    def _predict_with_q(self, x_cgm: jnp.ndarray, x_other: jnp.ndarray) -> np.ndarray:
        """
        Predice usando la red Q.
        
        Parámetros:
        -----------
        x_cgm : jnp.ndarray
            Datos CGM para predicción
        x_other : jnp.ndarray
            Otras características para predicción
            
        Retorna:
        --------
        np.ndarray
            Predicciones basadas en Q
        """
        if not hasattr(self.model, 'predict_with_q'):
            return np.zeros((len(x_cgm),), dtype=np.float32)
            
        self.rng_key, predict_key = jax.random.split(self.rng_key)
        actions = self.model.predict_with_q(
            self.states['q_network'], x_cgm, x_other, predict_key, deterministic=True
        )
        return np.array(actions)
    
    def _predict_generic(self, x_cgm: jnp.ndarray, x_other: jnp.ndarray) -> np.ndarray:
        """
        Predicción genérica para otros tipos de modelos.
        
        Parámetros:
        -----------
        x_cgm : jnp.ndarray
            Datos CGM para predicción
        x_other : jnp.ndarray
            Otras características para predicción
            
        Retorna:
        --------
        np.ndarray
            Predicciones genéricas
        """
        if hasattr(self.model, 'predict'):
            self.rng_key, predict_key = jax.random.split(self.rng_key)
            return np.array(self.model.predict(
                self.params or self.state or self.states,
                x_cgm, x_other, predict_key
            ))
        
        # Si no hay método predict, devolver ceros
        return np.zeros((len(x_cgm),), dtype=np.float32)
    
    def save(self, path: str) -> None:
        """
        Guarda el estado del modelo DRL JAX en disco.

        Parámetros:
        -----------
        path : str
            Ruta donde guardar el modelo
    
        Retorna:
        --------
        None
        """
        # Verificar si el modelo está inicializado
        if self.model is None:
            raise ValueError(CONST_MODEL_INIT_ERROR.format("guardar"))
        
        # Crear directorio si es necesario
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)

        try:
            # Intentar guardar usando el método del modelo subyacente si existe
            if hasattr(self.model, 'save'):
                self.model.save(path)
                print_success(f"Modelo guardado usando método nativo en: {path}")
                return
                
            # Determinar qué estado guardar
            state_to_save = None
            if self.params is not None:
                state_to_save = self.params
            elif self.state is not None:
                state_to_save = self.state
            elif self.states:  # Si hay estados (actor, crítico, etc.)
                state_to_save = self.states
                
            # Intentar guardar con checkpoint de Flax
            if state_to_save is not None:
                try:
                    save_checkpoint(path, state_to_save, step=0, keep=1)
                    print_success(f"Estado del modelo guardado en: {path}")
                    return
                except Exception as e:
                    print_warning(f"No se pudo guardar con checkpoint de Flax: {e}")
                
        except Exception as e:
            print_error(f"Error al guardar modelo: {e}")
        
        # Fallback: guardar con pickle
        print_warning("Guardando con pickle como alternativa.")
        save_data = {
            'params': self.params,
            'state': self.state,
            'states': self.states,
            'model_kwargs': self.model_kwargs,
            'algorithm': self.algorithm,
            'rng_key': self.rng_key
        }
            
        # Asegurar que no haya duplicación de extensión
        save_path = path if path.endswith('.pkl') else f"{path}.pkl"
        
        with open(save_path, 'wb') as f:
            pickle.dump(save_data, f)
        print_success(f"Estado del modelo guardado como pickle en: {save_path}")
            
    def _try_load_checkpoint(self, path: str) -> bool:
        """
        Intenta cargar el modelo desde un checkpoint de Flax.
        
        Parámetros:
        -----------
        path : str
            Ruta desde donde cargar el checkpoint
            
        Retorna:
        --------
        bool
            True si el checkpoint se cargó correctamente, False en caso contrario
        """
        try:
            # Determinar qué estado cargar (usando el mismo orden de prioridad que en save)
            if self.params is not None:
                restored_params = restore_checkpoint(path, self.params)
                if restored_params is not None:
                    self.params = restored_params
                    print_success(f"Parámetros restaurados desde: {path}")
                    return True
                return False
                
            if self.state is not None:
                restored_state = restore_checkpoint(path, self.state)
                if restored_state is not None:
                    self.state = restored_state
                    print_success(f"Estado restaurado desde: {path}")
                    return True
                return False
                
            if self.states:
                restored_states = restore_checkpoint(path, self.states)
                if restored_states is not None:
                    self.states = restored_states
                    print_success(f"Estados restaurados desde: {path}")
                    return True
                return False
                
            # Si no hay estado previo, intentar restaurar en un diccionario nuevo
            loaded_state = restore_checkpoint(path, {})
            if loaded_state is not None and loaded_state:  # Verificar que no sea None y no esté vacío
                self._assign_loaded_state(loaded_state)
                print_success(f"Estado del modelo restaurado desde: {path}")
                return True
            
            # Si llegamos aquí, no se pudo cargar ningún estado
            print_warning(f"No se pudo cargar un estado válido desde: {path}")
            return False
            
        except Exception as e:
            print_error(f"Error al cargar checkpoint: {e}")
            return False
    
    def _assign_loaded_state(self, loaded_state):
        """
        Asigna el estado cargado a la propiedad correcta.
        
        Parámetros:
        -----------
        loaded_state : Any
            Estado cargado desde el checkpoint
        """
        if isinstance(loaded_state, dict) and 'params' in loaded_state:
            self.params = loaded_state
        else:
            self.state = loaded_state
    
    def _load_from_pickle(self, path: str) -> None:
        """
        Carga el modelo desde un archivo pickle.
        
        Parámetros:
        -----------
        path : str
            Ruta del archivo pickle
        """
        pickle_path = path if path.endswith('.pkl') else f"{path}.pkl"
        with open(pickle_path, 'rb') as f:
            data = pickle.load(f)
            
        # Restaurar los componentes relevantes
        self._restore_model_components(data)
        
        # Restaurar metadatos y configuración
        self._restore_model_metadata(data)
        
        print_success(f"Estado del modelo restaurado desde pickle: {pickle_path}")
    
    def _restore_model_components(self, data: Dict) -> None:
        """
        Restaura los componentes principales del modelo desde los datos cargados.
        
        Parámetros:
        -----------
        data : Dict
            Datos del modelo cargados
        """
        if 'params' in data and data['params'] is not None:
            self.params = data['params']
            
        if 'state' in data and data['state'] is not None:
            self.state = data['state']
            
        if 'states' in data and data['states'] is not None:
            self.states = data['states']
    
    def _restore_model_metadata(self, data: Dict) -> None:
        """
        Restaura los metadatos del modelo desde los datos cargados.
        
        Parámetros:
        -----------
        data : Dict
            Datos del modelo cargados
        """
        if 'model_kwargs' in data:
            self.model_kwargs.update(data['model_kwargs'])
            
        if 'algorithm' in data:
            self.algorithm = data['algorithm']
            
        if 'rng_key' in data:
            self.rng_key = data['rng_key']
    
    def load(self, path: str) -> None:
        """
        Carga el estado del modelo DRL JAX desde disco.

        Parámetros:
        -----------
        path : str
            Ruta desde donde cargar el modelo

        Retorna:
        --------
        None
        """
        try:
            # Intentar cargar con checkpoint de Flax
            self._try_load_checkpoint(path)
        except Exception as e:
            print_warning(f"No se pudo cargar desde checkpoint de Flax: {e}")
            
            # Intentar cargar desde pickle como último recurso
            print_warning("Intentando cargar desde pickle.")
            try:
                self._load_from_pickle(path)
            except Exception as inner_e:
                print_error(f"Error al cargar desde pickle: {inner_e}")
                print_error("No se pudo restaurar el modelo.")
        
    def add_early_stopping(self, patience: int = 10, min_delta: float = 0.0, restore_best_weights: bool = True) -> None:
        """
        Agrega early stopping al modelo.

        Parámetros:
        -----------
        patience : int, opcional
            Número de épocas a esperar para detener el entrenamiento (default: 10)
        min_delta : float, opcional
            Mínima mejora requerida para considerar una mejora (default: 0.0)
        restore_best_weights : bool, opcional
            Si restaurar los mejores pesos al finalizar (default: True)
        """
        # Actualizar la configuración existente de early stopping
        self.early_stopping_config['patience'] = patience
        self.early_stopping_config['min_delta'] = min_delta
        self.early_stopping_config['restore_best_weights'] = restore_best_weights
class DRLModelWrapperPyTorch(ModelWrapper, nn.Module):
    """
    Wrapper para modelos de aprendizaje por refuerzo profundo implementados en PyTorch.
    
    Parámetros:
    -----------
    model_or_cls : Union[Callable, nn.Module]
        Clase del modelo DRL a instanciar o instancia ya creada del modelo
    **model_kwargs
        Argumentos para el constructor del modelo (usado solo si se pasa una clase)
    """
    
    def __init__(self, model_or_cls: Union[Callable, nn.Module], **model_kwargs) -> None:
        """
        Inicializa un wrapper para modelos de aprendizaje por refuerzo profundo en PyTorch.
        
        Parámetros:
        -----------
        model_or_cls : Union[Callable, nn.Module]
            Clase del modelo DRL a instanciar o instancia ya creada del modelo
        **model_kwargs
            Argumentos para el constructor del modelo (usado solo si se pasa una clase)
        """
        ModelWrapper.__init__(self)
        nn.Module.__init__(self)
        super().__init__()
        
        # Determinar si es una clase o una instancia
        self.is_class = isinstance(model_or_cls, type) or callable(model_or_cls)
        
        if self.is_class:
            self.model_cls = model_or_cls
            self.model_kwargs = model_kwargs
            # Inicializar el modelo inmediatamente si es posible
            try:
                self.model = self.model_cls(**self.model_kwargs)
            except Exception:
                self.model = None
        else:
            self.model = model_or_cls
            self.model_kwargs = {}
            
        self.buffer = None
        self.algorithm = model_kwargs.get('algorithm', 'generic')
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Inicializar el modelo si ya se pasó una instancia
        if not self.is_class and self.model is not None:
            if isinstance(self.model, nn.Module):
                self.model = self.model.to(self.device)
            
        # Crear un generador de numpy para operaciones aleatorias
        self.rng = np.random.Generator(np.random.PCG64(model_kwargs.get('seed', CONST_DEFAULT_SEED)))
        
        # Configuración para early stopping
        self.early_stopping_config = {
            'patience': EARLY_STOPPING_POLICY['early_stopping_patience'],
            'min_delta': EARLY_STOPPING_POLICY['early_stopping_min_delta'],
            'restore_best_weights': EARLY_STOPPING_POLICY['early_stopping_restore_best_weights'],
            'best_val_loss': EARLY_STOPPING_POLICY['early_stopping_best_val_loss'],
            'counter': EARLY_STOPPING_POLICY['early_stopping_counter'],
            'best_weights': EARLY_STOPPING_POLICY['early_stopping_best_weights']
        }
    
    def to(self, device: torch.device) -> 'DRLModelWrapperPyTorch':
        """
        Mueve el modelo al dispositivo especificado.
        
        Parámetros:
        -----------
        device : torch.device
            Dispositivo de destino (CPU/GPU)
            
        Retorna:
        --------
        DRLModelWrapperPyTorch
            Self para permitir encadenamiento
        """
        self.device = device
        if self.model is not None and isinstance(self.model, nn.Module):
                self.model = self.model.to(device)
        return self
    
    def forward(self, x_cgm: torch.Tensor, x_other: torch.Tensor) -> torch.Tensor:
        """
        Implementación requerida del método forward para nn.Module.
        
        Parámetros:
        -----------
        x_cgm : torch.Tensor
            Datos CGM de entrada
        x_other : torch.Tensor
            Otras características de entrada
            
        Retorna:
        --------
        torch.Tensor
            Predicciones del modelo
        """
        # Si el modelo no está inicializado, intentar inicializarlo ahora
        if self.model is None:
            if self.is_class:
                try:
                    self.model = self.model_cls(**self.model_kwargs)
                    if isinstance(self.model, nn.Module):
                        self.model = self.model.to(self.device)
                except Exception as e:
                    print_debug(f"Error al inicializar el modelo: {e}")
                    raise ValueError(CONST_MODEL_INIT_ERROR.format("realizar forward pass"))
            else:
                raise ValueError(CONST_MODEL_INIT_ERROR.format("realizar forward pass"))
            
        # Delegamos el forward al modelo subyacente
        try:
            if (hasattr(self.model, 'forward') and callable(self.model.forward)) or hasattr(self.model, '__call__'):
                return self.model(x_cgm, x_other)
        except Exception as e:
            print_debug(f"Error en forward: {e}")
            
        # Si no podemos usar forward directamente, intentar con predict
        with torch.no_grad():
            try:
                x_cgm_np = x_cgm.cpu().numpy()
                x_other_np = x_other.cpu().numpy()
                predictions_np = self.predict([x_cgm_np, x_other_np])
                return torch.FloatTensor(predictions_np).to(self.device)
            except Exception as e:
                print_debug(f"Error al predecir con el modelo: {e}")
                # Devolver un tensor de ceros como fallback
                return torch.zeros(x_cgm.shape[0], 1, device=self.device)
    
    def start(self, x_cgm: np.ndarray, x_other: np.ndarray, y: np.ndarray, 
             rng_key: Any = None) -> Any:
        """
        Inicializa el modelo DRL con los datos de entrada.
        
        Parámetros:
        -----------
        x_cgm : np.ndarray
            Datos CGM de entrada
        x_other : np.ndarray
            Otras características de entrada
        y : np.ndarray
            Valores objetivo
        rng_key : Any, opcional
            Clave para generación aleatoria (default: None)
            
        Retorna:
        --------
        Any
            Estado inicial del modelo o parámetros
        """
        # Si el modelo ya está inicializado y no es una clase, devolver directamente
        if self.model is not None and not self.is_class:
            return self.model

        # Si el modelo no está inicializado pero tenemos una clase, crearlo
        if self.model is None and self.is_class:
            self.model = self.model_cls(**self.model_kwargs)
            
        # Mover modelo al dispositivo adecuado si es un módulo PyTorch
        if isinstance(self.model, nn.Module):
            self.model = self.model.to(self.device)
        
        # Inicialización específica si el modelo lo requiere
        if hasattr(self.model, 'start'):
            return self.model.start(x_cgm, x_other, y, rng_key)
        elif hasattr(self.model, 'initialize'):
            state_dim = (x_cgm.shape[1:], x_other.shape[1:])
            action_dim = 1  # Para regresión en el caso de dosis de insulina
            return self.model.initialize(state_dim, action_dim)
            
        return self.model

    def parameters(self, recurse=True):
        """
        Retorna los parámetros entrenables del modelo.
        Necesario para que optimizadores de PyTorch funcionen correctamente.
        
        Parámetros:
        -----------
        recurse : bool, opcional
            Si True, devuelve parámetros de esta instancia y todos los submódulos
            recursivamente (default: True)
            
        Retorna:
        --------
        Iterator
            Iterador sobre los parámetros del modelo
        """
        # Si el modelo existe y es un módulo PyTorch, delegar a sus parámetros
        if self.model is not None and isinstance(self.model, nn.Module):
            return self.model.parameters(recurse=recurse)
        # De lo contrario, devolver los parámetros de este wrapper (que podría estar vacío)
        return super().parameters(recurse=recurse)
        
    def train(self, mode: bool = True) -> 'DRLModelWrapperPyTorch':
        """
        Establece el módulo en modo entrenamiento.
        
        Parámetros:
        -----------
        mode : bool, opcional
            Si True, establece en modo entrenamiento, si False en modo evaluación (default: True)
            
        Retorna:
        --------
        DRLModelWrapperPyTorch
            Self para permitir encadenamiento
        """
        nn.Module.train(self, mode)
        if self.model is not None and isinstance(self.model, nn.Module):
            self.model.train(mode)
        return self
    
    def eval(self) -> 'DRLModelWrapperPyTorch':
        """
        Establece el módulo en modo evaluación.
        
        Retorna:
        --------
        DRLModelWrapperPyTorch
            Self para permitir encadenamiento
        """
        # Call nn.Module's eval method directly
        nn.Module.eval(self)
        if self.model is not None and isinstance(self.model, nn.Module):
            self.model.eval()
        return self
    
    def _initialize_optimizer(self, learning_rate: float = 1e-3) -> None:
        """
        Inicializa el optimizador si aún no existe.

        Parámetros:
        -----------
        learning_rate : float, opcional
            Tasa de aprendizaje para el optimizador (default: 1e-3)

        Retorna:
        --------
        None
        """
        if not hasattr(self, 'optimizer') or self.optimizer is None:
            if self.model is not None and isinstance(self.model, nn.Module):
                # Usar Adam por defecto si no se especifica otro
                self.optimizer = optim.Adam(self.parameters(), lr=learning_rate, weight_decay=0.01)
            else:
                print_warning("No se pudo inicializar el optimizador: el modelo no es un nn.Module o no está inicializado.")

    def _compute_rewards(self, x_cgm: np.ndarray, x_other: np.ndarray, y: np.ndarray) -> np.ndarray:
        """
        Calcula recompensas a partir de los datos y objetivos. (Adaptado para PyTorch)

        Parámetros:
        -----------
        x_cgm : np.ndarray
            Datos CGM de entrada
        x_other : np.ndarray
            Otras características de entrada
        y : np.ndarray
            Valores objetivo

        Retorna:
        --------
        np.ndarray
            Recompensas calculadas para el entrenamiento RL
        """
        # Si el modelo proporciona una función para calcular recompensas, úsala
        if hasattr(self.model, 'compute_rewards'):
            with torch.no_grad():
                x_cgm_t = torch.FloatTensor(x_cgm).to(self.device)
                x_other_t = torch.FloatTensor(x_other).to(self.device)
                y_t = torch.FloatTensor(y).to(self.device)
                rewards_t = self.model.compute_rewards(x_cgm_t, x_other_t, y_t)
                return rewards_t.cpu().numpy()

        # Implementación por defecto: recompensa negativa basada en el error
        predicted = self.predict(x_cgm, x_other) # Usa el método predict del wrapper
        error = np.abs(predicted.flatten() - y.flatten())
        # Normalizar error al rango [-1, 0] donde -1 es el peor error y 0 es perfecto
        max_error = np.max(error) if np.max(error) > 0 else 1.0
        rewards = -error / max_error
        return rewards

    def _fill_buffer(self, x_cgm: np.ndarray, x_other: np.ndarray, rewards: np.ndarray, y: np.ndarray) -> None:
        """
        Llena el buffer de experiencia con transiciones. (Adaptado para PyTorch)

        Parámetros:
        -----------
        x_cgm : np.ndarray
            Estados (datos CGM)
        x_other : np.ndarray
            Estados (otras características)
        rewards : np.ndarray
            Recompensas calculadas
        y : np.ndarray
            Acciones objetivo (dosis)
        """
        if not hasattr(self.model, 'add_to_buffer') or not hasattr(self.model, 'buffer'):
            return

        buffer = getattr(self.model, 'buffer', None)
        if buffer is None:
            return

        for i in range(len(rewards)):
            state = (x_cgm[i], x_other[i])
            action = y[i]
            reward = rewards[i]
            # En un entorno supervisado, el siguiente estado puede ser el mismo
            # y done es siempre True (episodio de un paso)
            next_state = (x_cgm[i], x_other[i])
            done = True
            # El método add_to_buffer del modelo debe manejar la conversión a tensores si es necesario
            self.model.add_to_buffer(buffer, state, action, reward, next_state, done)

    def _update_networks(self, batch_size: int) -> Dict[str, float]:
        """
        Actualiza las redes del modelo usando experiencias del buffer. (Adaptado para PyTorch)

        Parámetros:
        -----------
        batch_size : int
            Tamaño del lote para actualizaciones

        Retorna:
        --------
        Dict[str, float]
            Diccionario con métricas de actualización (pérdidas)
        """
        # Inicializar métricas
        metrics = self._init_update_metrics()
        
        # Verificar si el modelo tiene los métodos necesarios
        if not self._check_model_readiness_for_update():
            return metrics
            
        # Obtener batch del buffer
        batch = self._get_batch_from_buffer(batch_size)
        if batch is None:
            return metrics
            
        # Ejecutar actualización y procesar resultados
        return self._process_update_results(batch, metrics)
        
    def _init_update_metrics(self) -> Dict[str, float]:
        """Inicializa las métricas de actualización con valores por defecto."""
        return {
            "actor_loss": 0.0,
            "critic_loss": 0.0,
            "q_loss": 0.0,
            "total_loss": 0.0
        }
        
    def _check_model_readiness_for_update(self) -> bool:
        """Verifica si el modelo está listo para actualizaciones."""
        if not hasattr(self.model, 'update') or not hasattr(self.model, 'sample_buffer'):
            print_warning("El modelo no implementa 'update' o 'sample_buffer'. No se realizarán actualizaciones.")
            return False
        return True
        
    def _get_batch_from_buffer(self, batch_size: int):
        """Obtiene un batch de experiencias del buffer."""
        buffer = getattr(self.model, 'buffer', None)
        if buffer is None:
            print_warning("El modelo no tiene un atributo 'buffer'. No se puede muestrear.")
            return None
        return self.model.sample_buffer(buffer, batch_size)
        
    def _process_update_results(self, batch, metrics: Dict[str, float]) -> Dict[str, float]:
        """Procesa los resultados de la actualización del modelo."""
        # Preparar para actualización
        self.optimizer.zero_grad()
        update_metrics = self.model.update(batch)
        
        # Manejar diferentes tipos de resultado
        if isinstance(update_metrics, dict) and 'total_loss' in update_metrics:
            self._handle_dict_with_total_loss(update_metrics, metrics)
        elif isinstance(update_metrics, torch.Tensor) and update_metrics.requires_grad:
            self._handle_tensor_loss(update_metrics, metrics)
        elif isinstance(update_metrics, (int, float)):
            metrics["total_loss"] = float(update_metrics)
        elif isinstance(update_metrics, dict):
            self._handle_dict_without_total_loss(update_metrics, metrics)
            
        return metrics
        
    def _handle_dict_with_total_loss(self, update_metrics: Dict, metrics: Dict[str, float]) -> None:
        """Maneja el caso donde update devuelve un diccionario con total_loss."""
        loss = update_metrics['total_loss']
        if isinstance(loss, torch.Tensor) and loss.requires_grad:
            loss.backward()
            self.optimizer.step()
            self._convert_tensor_values_to_float(update_metrics, metrics)
        else:
            self._convert_all_values_to_float(update_metrics, metrics)
            
    def _handle_tensor_loss(self, loss: torch.Tensor, metrics: Dict[str, float]) -> None:
        """Maneja el caso donde update devuelve un tensor como pérdida total."""
        loss.backward()
        self.optimizer.step()
        metrics["total_loss"] = loss.item()
        
    def _handle_dict_without_total_loss(self, update_metrics: Dict, metrics: Dict[str, float]) -> None:
        """Maneja el caso donde update devuelve un diccionario sin total_loss."""
        self._convert_tensor_values_to_float(update_metrics, metrics)
        
    def _convert_tensor_values_to_float(self, source: Dict, target: Dict[str, float]) -> None:
        """Convierte valores tensor a float en las métricas."""
        for key, value in source.items():
            if key in target:
                if isinstance(value, torch.Tensor):
                    target[key] = value.item()
                else:
                    target[key] = float(value)
                    
    def _convert_all_values_to_float(self, source: Dict, target: Dict[str, float]) -> None:
        """Convierte todos los valores a float en las métricas."""
        for key, value in source.items():
            if key in target:
                target[key] = float(value)

    def _run_epoch_updates(self, batch_size: int, updates_per_epoch: int) -> Dict[str, float]:
        """
        Ejecuta las actualizaciones durante una época. (Adaptado para PyTorch)

        Parámetros:
        -----------
        batch_size : int
            Tamaño del lote
        updates_per_epoch : int
            Número de actualizaciones por época

        Retorna:
        --------
        Dict[str, float]
            Métricas acumuladas de la época
        """
        epoch_metrics = {key: 0.0 for key in ["actor_loss", "critic_loss", "q_loss", "total_loss"]}
        updates_done = 0
        for _ in range(updates_per_epoch):
            try:
                batch_metrics = self._update_networks(batch_size)
                for key, value in batch_metrics.items():
                    if key in epoch_metrics:
                        epoch_metrics[key] += value
                updates_done += 1
            except Exception as e:
                print_warning(f"Error durante la actualización de redes: {e}. Saltando esta actualización.")
                # Podría ser útil agregar un traceback aquí para depuración
                # import traceback
                # traceback.print_exc()
                continue # Saltar a la siguiente actualización

        # Promediar métricas si se realizaron actualizaciones
        if updates_done > 0:
            for key in epoch_metrics:
                epoch_metrics[key] /= updates_done
        return epoch_metrics

    def _validate_model(self, x_cgm_val: np.ndarray, x_other_val: np.ndarray,
                       y_val: np.ndarray) -> float:
        """
        Realiza validación del modelo y calcula la pérdida (MSE por defecto).

        Parámetros:
        -----------
        x_cgm_val : np.ndarray
            Datos CGM de validación
        x_other_val : np.ndarray
            Otras características de validación
        y_val : np.ndarray
            Valores objetivo de validación

        Retorna:
        --------
        float
            Pérdida de validación calculada
        """
        self.eval() # Poner el modelo en modo evaluación
        with torch.no_grad():
            val_preds = self.predict(x_cgm_val, x_other_val) # Usar el método predict del wrapper
            # Calcular MSE
            val_loss = float(np.mean((val_preds.flatten() - y_val.flatten()) ** 2))
        self.train() # Volver a poner el modelo en modo entrenamiento
        return val_loss

    def _process_epoch_metrics(self, epoch_metrics: Dict[str, float],
                               history: Dict[str, List[float]]) -> float:
        """
        Procesa y registra las métricas de una época en el historial.

        Parámetros:
        -----------
        epoch_metrics : Dict[str, float]
            Métricas promediadas de la época
        history : Dict[str, List[float]]
            Historial de entrenamiento a actualizar

        Retorna:
        --------
        float
            Pérdida total calculada para la época
        """
        # Registrar métricas en el historial
        for key, value in epoch_metrics.items():
            if key in history:
                history[key].append(value)

        # Pérdida total para seguimiento y early stopping
        total_loss = epoch_metrics.get("total_loss", 0.0)
        epsilon = 1e-10  # Pequeño valor para comparación de punto flotante

        if abs(total_loss) < epsilon:
            # Si no hay pérdida total (o es muy cercana a cero), usar la suma de otras pérdidas si existen
            other_losses = [epoch_metrics.get(key, 0.0) for key in ["actor_loss", "critic_loss", "q_loss"]]
            if any(abs(l) > epsilon for l in other_losses):
                 total_loss = sum(other_losses)
            # Si todas las pérdidas son cero, total_loss permanece cero

        history[CONST_LOSS].append(total_loss) # Asegurarse de que 'loss' siempre se registre
        return total_loss

    def _initialize_training(self, x_cgm: np.ndarray, x_other: np.ndarray, y: np.ndarray) -> None:
        """
        Inicializa el modelo y el optimizador para entrenamiento.
        
        Parámetros:
        -----------
        x_cgm : np.ndarray
            Datos CGM de entrada
        x_other : np.ndarray
            Otras características de entrada
        y : np.ndarray
            Valores objetivo
        """
        if self.model is None:
            self.start(x_cgm, x_other, y)
        self._initialize_optimizer()
        
    def _setup_history(self) -> Dict[str, List[float]]:
        """
        Prepara el diccionario para almacenar el historial de entrenamiento.
        
        Retorna:
        --------
        Dict[str, List[float]]
            Diccionario de historial inicializado
        """
        return {
            CONST_LOSS: [],
            "actor_loss": [],
            "critic_loss": [],
            "q_loss": [],
            CONST_VAL_LOSS: []
        }
        
    def _setup_early_stopping(self) -> None:
        """
        Inicializa la configuración para early stopping.
        """
        es_config = self.early_stopping_config
        es_config['best_val_loss'] = float('inf')
        es_config['counter'] = 0
        es_config['best_weights'] = None
    
    def _run_training_epoch(self, epoch: int, x_cgm: np.ndarray, x_other: np.ndarray, 
                           y: np.ndarray, batch_size: int, do_validation: bool,
                           x_cgm_val: Optional[np.ndarray], x_other_val: Optional[np.ndarray],
                           y_val: Optional[np.ndarray], history: Dict[str, List[float]]) -> Tuple[float, Optional[float]]:
        """
        Ejecuta una época completa de entrenamiento.
        
        Retorna:
        --------
        Tuple[float, Optional[float]]
            Tupla con (pérdida de entrenamiento, pérdida de validación)
        """
        self.train()  # Asegurar modo entrenamiento
        
        # Calcular recompensas y llenar buffer
        rewards = self._compute_rewards(x_cgm, x_other, y)
        self._fill_buffer(x_cgm, x_other, rewards, y)
        
        # Ejecutar actualizaciones de redes
        updates_per_epoch = max(1, len(x_cgm) // batch_size)
        epoch_metrics = self._run_epoch_updates(batch_size, updates_per_epoch)
        
        # Procesar métricas
        epoch_loss = self._process_epoch_metrics(epoch_metrics, history)
        
        # Validación
        val_loss = None
        if do_validation:
            val_loss = self._validate_model(x_cgm_val, x_other_val, y_val)
            history[CONST_VAL_LOSS].append(val_loss)
            
        return epoch_loss, val_loss
    
    def _log_epoch_progress(self, epoch: int, epochs: int, epoch_loss: float, 
                           val_loss: Optional[float], history: Dict[str, List[float]]) -> None:
        """
        Registra el progreso de la época actual.
        
        Parámetros:
        -----------
        epoch : int
            Época actual
        epochs : int
            Total de épocas
        epoch_loss : float
            Pérdida de entrenamiento
        val_loss : Optional[float]
            Pérdida de validación
        history : Dict[str, List[float]]
            Historial de entrenamiento
        """
        log_msg = f"Época {epoch + 1}/{epochs} - Pérdida: {epoch_loss:.4f}"
        if val_loss is not None:
            log_msg += f" - Pérdida Val: {val_loss:.4f}"
        
        # Agregar otras métricas si existen
        for metric in ["actor_loss", "critic_loss", "q_loss"]:
            if history[metric] and abs(history[metric][-1]) > 1e-10:
                log_msg += f" - {metric}: {history[metric][-1]:.4f}"
        
        print_info(log_msg)
    
    def _check_early_stopping(self, epoch_loss: float, val_loss: Optional[float]) -> bool:
        """
        Comprueba si se debe detener el entrenamiento según la política de early stopping.
        
        Parámetros:
        -----------
        epoch_loss : float
            Pérdida de entrenamiento
        val_loss : Optional[float]
            Pérdida de validación
            
        Retorna:
        --------
        bool
            True si se debe detener el entrenamiento, False en caso contrario
        """
        es_config = self.early_stopping_config
        current_val_loss = val_loss if val_loss is not None else epoch_loss
        
        if current_val_loss < es_config['best_val_loss'] - es_config['min_delta']:
            es_config['best_val_loss'] = current_val_loss
            es_config['counter'] = 0
            if es_config['restore_best_weights']:
                es_config['best_weights'] = self.model.state_dict()
            return False
        
        es_config['counter'] += 1
        return es_config['counter'] >= es_config['patience']
    
    def _restore_best_weights(self) -> None:
        """
        Restaura los mejores pesos encontrados durante el entrenamiento.
        """
        es_config = self.early_stopping_config
        if es_config['restore_best_weights'] and es_config['best_weights'] is not None:
            print_info(f"Restaurando mejores pesos con pérdida de validación: {es_config['best_val_loss']:.4f}")
            self.model.load_state_dict(es_config['best_weights'])
    
    def fit(self, x: List[np.ndarray], y: np.ndarray,
           validation_data: Optional[Tuple] = None,
           epochs: int = CONST_DEFAULT_EPOCHS, batch_size: int = CONST_DEFAULT_BATCH_SIZE, verbose: int = 1) -> Dict[str, List[float]]:
        """
        Entrena el modelo DRL con los datos proporcionados usando PyTorch.

        Parámetros:
        -----------
        x : List[np.ndarray]
            Lista con [x_cgm, x_other]
        y : np.ndarray
            Valores objetivo (acciones)
        validation_data : Optional[Tuple], opcional
            Datos de validación como ([x_cgm_val, x_other_val], y_val) (default: None)
        epochs : int, opcional
            Número de épocas de entrenamiento (default: 10)
        batch_size : int, opcional
            Tamaño de lote (default: 32)
        verbose : int, opcional
            Nivel de verbosidad (0=silencioso, 1=progreso, 2=detallado)

        Retorna:
        --------
        Dict[str, List[float]]
            Historial de entrenamiento con métricas
        """
        x_cgm, x_other = x
        
        # Inicialización del entrenamiento
        self._initialize_training(x_cgm, x_other, y)
        
        # Preparar datos y estructuras
        x_cgm_val, x_other_val, y_val = self._unpack_validation_data(validation_data)
        do_validation = x_cgm_val is not None
        history = self._setup_history()
        self._setup_early_stopping()
        
        # Anunciar inicio del entrenamiento
        if verbose > 0:
            print_info(f"Iniciando entrenamiento DRL ({self.algorithm}) por {epochs} épocas (PyTorch)...")
        
        # Bucle de entrenamiento por épocas
        for epoch in tqdm(range(epochs), desc="Entrenando (PyTorch)", disable=verbose == 0):
            # Ejecutar una época completa
            epoch_loss, val_loss = self._run_training_epoch(
                epoch, x_cgm, x_other, y, batch_size, do_validation,
                x_cgm_val, x_other_val, y_val, history
            )
            
            # Registrar progreso
            if verbose > 0:
                self._log_epoch_progress(epoch, epochs, epoch_loss, val_loss, history)
                
            # Comprobar early stopping
            if self._check_early_stopping(epoch_loss, val_loss):
                print_info(f"Early stopping en época {epoch + 1}")
                break
                
        # Restaurar mejores pesos al finalizar
        self._restore_best_weights()
        
        return history

    def save(self, path: str) -> None:
        """
        Guarda el modelo DRL de PyTorch en disco.

        Parámetros:
        -----------
        path : str
            Ruta donde guardar el modelo.

        Retorna:
        --------
        None
        """
        if self.model is None:
            raise ValueError("El modelo debe estar inicializado antes de guardarlo.")

        save_data = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict() if self.optimizer else None,
            'model_kwargs': self.model_kwargs,
            'rng_state': torch.get_rng_state()
        }
        torch.save(save_data, path)
        print_success(f"Modelo guardado en: {path}")

    def load(self, path: str) -> None:
        """
        Carga el modelo DRL de PyTorch desde disco.

        Parámetros:
        -----------
        path : str
            Ruta desde donde cargar el modelo.

        Retorna:
        --------
        None
        """
        checkpoint = torch.load(path, map_location=self.device)
        if self.model is None:
            self.model = self.model_cls(**self.model_kwargs)
            self.model.to(self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        if 'optimizer_state_dict' in checkpoint and self.optimizer:
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if 'rng_state' in checkpoint:
            torch.set_rng_state(checkpoint['rng_state'])
        print_success(f"Modelo cargado desde: {path}")

# Actualizar la clase DRLModelWrapper para incluir PyTorch
class DRLModelWrapper(ModelWrapper):
    """
    Wrapper para modelos de aprendizaje por refuerzo profundo que selecciona el wrapper 
    adecuado según el framework.
    
    Parámetros:
    -----------
    model_cls : Callable
        Clase del modelo DRL a instanciar
    framework : str
        Framework a utilizar ('jax', 'tensorflow' o 'pytorch')
    **model_kwargs
        Argumentos para el constructor del modelo
    """
    
    def __init__(self, model_cls: Callable, framework: str = 'jax', **model_kwargs) -> None:
        """
        Inicializa el wrapper seleccionando el backend adecuado.
        
        Parámetros:
        -----------
        model_cls : Callable
            Clase del modelo DRL a instanciar
        framework : str
            Framework a utilizar ('jax', 'tensorflow' o 'pytorch')
        **model_kwargs
            Argumentos para el constructor del modelo
        """
        super().__init__()
        self.framework = framework.lower()
        
        # Seleccionar el wrapper adecuado según el framework
        if self.framework == 'jax':
            self.wrapper = DRLModelWrapperJAX(model_cls, **model_kwargs)
        elif self.framework == 'pytorch':
            self.wrapper = DRLModelWrapperPyTorch(model_cls, **model_kwargs)
        else:
            self.wrapper = DRLModelWrapperTF(model_cls, **model_kwargs)
    
    def add_early_stopping(self, patience: int = 10, min_delta: float = 0.0, restore_best_weights: bool = True) -> None:
        """
        Agrega early stopping al modelo.

        Parámetros:
        -----------
        patience : int, opcional
            Número de épocas a esperar para detener el entrenamiento (default: 10)
        min_delta : float, opcional
            Mínima mejora requerida para considerar una mejora (default: 0.0)
        restore_best_weights : bool, opcional
            Si restaurar los mejores pesos al finalizar (default: True)
        """
        if hasattr(self.wrapper, 'add_early_stopping'):
            self.wrapper.add_early_stopping(patience, min_delta, restore_best_weights)
        else:
            super().add_early_stopping(patience, min_delta, restore_best_weights)
    
    def start(self, x_cgm: np.ndarray, x_other: np.ndarray, y: np.ndarray, 
             rng_key: Any = None) -> Any:
        """
        Inicializa el modelo DRL con los datos de entrada.
        
        Parámetros:
        -----------
        x_cgm : np.ndarray
            Datos CGM de entrada
        x_other : np.ndarray
            Otras características de entrada
        y : np.ndarray
            Valores objetivo
        rng_key : Any, opcional
            Clave para generación aleatoria (default: None)
            
        Retorna:
        --------
        Any
            Estado inicial del modelo o parámetros
        """
        return self.wrapper.start(x_cgm, x_other, y, rng_key)
    
    def train(self, x_cgm: np.ndarray, x_other: np.ndarray, y: np.ndarray, 
             validation_data: Optional[Tuple[Tuple[np.ndarray, np.ndarray], np.ndarray]] = None,
             epochs: int = CONST_DEFAULT_EPOCHS, batch_size: int = CONST_DEFAULT_BATCH_SIZE, verbose: int = 1) -> Dict[str, List[float]]:
        """
        Entrena el modelo DRL con los datos proporcionados.
        
        Parámetros:
        -----------
        x_cgm : np.ndarray
            Datos CGM de entrenamiento
        x_other : np.ndarray
            Otras características de entrenamiento
        y : np.ndarray
            Valores objetivo (acciones)
        validation_data : Optional[Tuple[Tuple[np.ndarray, np.ndarray], np.ndarray]], opcional
            Datos de validación como ((x_cgm_val, x_other_val), y_val) (default: None)
        epochs : int, opcional
            Número de épocas de entrenamiento (default: 10)
        batch_size : int, opcional
            Tamaño de lote (default: 32)
        verbose : int, opcional
            Nivel de verbosidad (0=silencioso, 1=progreso, 2=detallado)
            
        Retorna:
        --------
        Dict[str, List[float]]
            Historial de entrenamiento con métricas
        """
        if verbose > 0:
            print_info(f"Entrenando modelo {self.wrapper.algorithm} en {self.framework}...")
            print_info(f"Épocas: {epochs}, Batch size: {batch_size}, Ejemplos: {len(y)}")
        
        # Delegar el entrenamiento al wrapper específico con el nivel de verbosidad
        if hasattr(self.wrapper, 'fit') and callable(self.wrapper.fit):
            return self.wrapper.fit(x_cgm, x_other, y, validation_data, epochs, batch_size, verbose=verbose)
        else:
            return self.wrapper.train(x_cgm, x_other, y, validation_data, epochs, batch_size, verbose=verbose)
    
    def predict(self, x_cgm: np.ndarray, x_other: np.ndarray) -> np.ndarray:
        """
        Realiza predicciones con el modelo entrenado.
        
        Parámetros:
        -----------
        x_cgm : np.ndarray
            Datos CGM para predicción
        x_other : np.ndarray
            Otras características para predicción
            
        Retorna:
        --------
        np.ndarray
            Predicciones del modelo (acciones)
        """
        return self.wrapper.predict(x_cgm, x_other)
    def save(self, filepath: str) -> None:
        """
        Guarda el modelo/agente (si el wrapper lo soporta).
        
        Parámetros:
        -----------
        filepath : str
            Ruta del archivo donde guardar el modelo
            
        Retorna:
        --------
        None
        """
        if hasattr(self.wrapper, 'save'):
            self.wrapper.save(filepath)
        else:
            print_warning(f"El guardado no está implementado para el wrapper {type(self.wrapper).__name__}.")

    def load(self, filepath: str) -> None:
        """
        Carga el modelo/agente (si el wrapper lo soporta).
        
        Parámetros:
        -----------
        filepath : str
            Ruta del archivo desde donde cargar el modelo
            
        Retorna:
        --------
        None
        """
        if hasattr(self.wrapper, 'load'):
            self.wrapper.load(filepath)
        else:
            print(f"Advertencia: La carga no está implementada para el wrapper {type(self.wrapper).__name__}.")
