from typing import Dict, List, Tuple, Any, Optional, Callable
import numpy as np
import jax
import jax.numpy as jnp
from flax.training.checkpoints import save_checkpoint, restore_checkpoint
import os
import pickle
from tqdm.auto import tqdm

from constants.constants import CONST_DEFAULT_SEED, CONST_DEFAULT_EPOCHS, CONST_DEFAULT_BATCH_SIZE
from config.models_config_old import EARLY_STOPPING_POLICY
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
CONST_EPSILON = 1e-10


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
    
    def __init__(self, model_cls: Callable, algorithm: str = 'generic', **model_kwargs) -> None:
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
        self.algorithm = model_kwargs.get('algorithm', algorithm)
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
            # Verificar si model_cls es una clase o una instancia
            if isinstance(self.model_cls, type) or (callable(self.model_cls) and not hasattr(self.model_cls, 'predict')):
                # Es una clase o función, llamarla con los argumentos
                self.model = self.model_cls(**self.model_kwargs)
            else:
                # Es probablemente una instancia, usarla directamente
                self.model = self.model_cls
        
        # Dimensiones del espacio de estados y acciones
        state_dim = (x_cgm.shape[1:], x_other.shape[1:])
        action_dim = 1  # Para regresión en el caso de dosis de insulina
        
        # Inicializar según el tipo de algoritmo
        if self.algorithm in ['ppo', 'a2c', 'a3c', 'sac', 'trpo']:
            self._initialize_actor_critic(state_dim, action_dim)
        elif self.algorithm in ['dqn', 'ddpg']:
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
        
        if abs(total_loss) < CONST_EPSILON:
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
                if abs(batch_metrics.get('actor_loss', 0.0)) > CONST_EPSILON:
                    postfix["actor"] = f"{batch_metrics.get('actor_loss', 0.0):.4f}"
                if abs(batch_metrics.get('critic_loss', 0.0)) > CONST_EPSILON:
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
        super().add_early_stopping(patience, min_delta, restore_best_weights)
    
    def _init_training_buffer(self):
        """
        Inicializa el buffer de experiencia si es necesario.
        
        Retorna:
        --------
        None
        """
        if hasattr(self.model, 'init_buffer') and self.buffer is None:
            self.rng_key, buffer_key = jax.random.split(self.rng_key)
            self.buffer = self.model.init_buffer(buffer_key)
            
    def _handle_validation(self, x_cgm_val, x_other_val, y_val, history):
        """
        Realiza validación del modelo y actualiza el historial.
        
        Parámetros:
        -----------
        x_cgm_val : np.ndarray
            Datos CGM de validación
        x_other_val : np.ndarray
            Otras características de validación
        y_val : np.ndarray
            Valores objetivo de validación
        history : Dict[str, List[float]]
            Historial donde registrar métricas
            
        Retorna:
        --------
        float or None
            Pérdida de validación
        """
        val_loss = None
        if x_cgm_val is not None and y_val is not None:
            val_preds = self.predict(x_cgm_val, x_other_val)
            val_loss = float(np.mean((val_preds - y_val) ** 2))
            history["val_loss"].append(val_loss)
        return val_loss
            
    def _update_progress_bar(self, epoch_iterator, total_loss, val_loss, epoch_metrics):
        """
        Actualiza la barra de progreso con las métricas actuales.
        
        Parámetros:
        -----------
        epoch_iterator : tqdm
            Iterador de época
        total_loss : float
            Pérdida total
        val_loss : float or None
            Pérdida de validación
        epoch_metrics : Dict[str, float]
            Métricas de la época
        """
        postfix = {"loss": f"{total_loss:.4f}"}
        if val_loss is not None:
            postfix["val_loss"] = f"{val_loss:.4f}"
        
        for metric in ["actor_loss", "critic_loss", "q_loss"]:
            if epoch_metrics.get(metric, 0.0) > CONST_EPSILON:
                postfix[metric] = f"{epoch_metrics[metric]:.4f}"
                
        epoch_iterator.set_postfix(**postfix)
    
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
        if self.model is None:
            self.start(x_cgm, x_other, y)
        
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
        if hasattr(self.model, 'fit'):
            return self.model.fit(
                [x_cgm_arr, x_other_arr],
                y_arr,
                validation_data=([x_cgm_val_arr, x_other_val_arr], y_val_arr) if x_cgm_val is not None else None,
                epochs=epochs,
                batch_size=batch_size,
                verbose=verbose
            )
        
        # # Inicializar buffer si es necesario
        # self._init_training_buffer()
        
        # # Crear iterador con barra de progreso para épocas
        # epoch_iterator = tqdm(
        #     range(epochs),
        #     desc=f"Entrenando {self.algorithm} (JAX)",
        #     disable=verbose == 0,
        #     unit="época"
        # )
        
        # # Entrenamiento por épocas
        # for _ in epoch_iterator:
        #     self.rng_key, _ = jax.random.split(self.rng_key)
            
        #     # Calcular recompensas y llenar buffer
        #     rewards = self._compute_rewards(x_cgm, x_other, y)
        #     self._fill_buffer(x_cgm_arr, x_other_arr, rewards, y_arr)
            
        #     # Ejecutar actualizaciones durante la época
        #     updates_per_epoch = max(1, len(x_cgm) // batch_size)
        #     epoch_metrics = self._run_epoch_updates(
        #         x_cgm_arr, x_other_arr, y_arr, rewards, batch_size, updates_per_epoch, verbose=verbose
        #     )
            
        #     # Procesar métricas y registrarlas en el historial
        #     total_loss = self._process_epoch_metrics(epoch_metrics, updates_per_epoch, history)
        #     history["loss"].append(total_loss)
            
        #     # Validación si hay datos disponibles
        #     val_loss = self._handle_validation(x_cgm_val, x_other_val, y_val, history)

        #     # Actualizar la barra de progreso con métricas actuales
        #     self._update_progress_bar(epoch_iterator, total_loss, val_loss, epoch_metrics)
            
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
        if self.model is None:
            raise ValueError(CONST_MODEL_INIT_ERROR.format("predecir"))
        
        try:
            # Pasar los inputs como una lista, que es lo que esperan los wrappers DRL
            return np.array(self.model.predict([x_cgm, x_other]))
        except Exception as e:
            print_error(f"Error en predicción: {e}")
            return np.zeros((x_cgm.shape[0], 1))
    
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
