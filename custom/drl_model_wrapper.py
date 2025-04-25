from flax.training import train_state
from typing import Dict, List, Tuple, Any, Optional, Callable, Union
import numpy as np
import jax
import jax.numpy as jnp
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm.auto import tqdm

from custom.model_wrapper import ModelWrapper
from custom.printer import print_debug, print_info

# Constantes para uso repetido
CONST_ACTOR = "actor"
CONST_CRITIC = "critic"
CONST_TARGET = "target"
CONST_PARAMS = "params"
CONST_DEVICE = "device"
CONST_MODEL_INIT_ERROR = "El modelo debe ser inicializado antes de {}"


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
        
    def train(self, x_cgm: np.ndarray, x_other: np.ndarray, y: np.ndarray, 
             validation_data: Optional[Tuple[Tuple[np.ndarray, np.ndarray], np.ndarray]] = None,
             epochs: int = 10, batch_size: int = 32, verbose: int = 1) -> Dict[str, List[float]]:
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
                          batch_size: int, updates_per_epoch: int) -> Dict[str, float]:
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
            
        Retorna:
        --------
        Dict[str, float]
            Métricas acumuladas de la época
        """
        epoch_metrics = {key: 0.0 for key in ["actor_loss", "critic_loss", "q_loss", "total_loss"]}
        
        for _ in range(updates_per_epoch):
            # Obtener un lote del buffer
            batch = self._get_batch_from_buffer(x_cgm_arr, x_other_arr, y_arr, rewards, batch_size)
            
            # Actualizar redes
            batch_metrics = self._update_networks(batch)
            for key, value in batch_metrics.items():
                epoch_metrics[key] += value
                
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
             epochs: int = 10, batch_size: int = 32, verbose: int = 1) -> Dict[str, List[float]]:
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
                x_cgm_arr, x_other_arr, y_arr, rewards, batch_size, updates_per_epoch
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

class DRLModelWrapperPyTorch(ModelWrapper):
    """
    Wrapper para modelos de aprendizaje por refuerzo profundo implementados en PyTorch.
    
    Parámetros:
    -----------
    model_cls : Callable
        Clase del modelo DRL a instanciar
    **model_kwargs
        Argumentos para el constructor del modelo
    """
    
    def __init__(self, model_cls: Callable, **model_kwargs) -> None:
        """
        Inicializa un wrapper para modelos de aprendizaje por refuerzo profundo en PyTorch.
        
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
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # Crear un generador de numpy para operaciones aleatorias
        self.rng = np.random.Generator(np.random.PCG64(model_kwargs.get('seed', 42)))
        # Configuración para early stopping
        self.early_stopping_config = {
            'patience': 10,
            'min_delta': 0.0,
            'restore_best_weights': True,
            'best_val_loss': float('inf'),
            'counter': 0,
            'best_weights': None
        }
    
    def add_early_stopping(self, patience: int = 10, min_delta: float = 0.0, restore_best_weights: bool = True) -> None:
        """
        Configura early stopping para el entrenamiento.
        
        Parámetros:
        -----------
        patience : int, opcional
            Número de épocas a esperar para detener el entrenamiento (default: 10)
        min_delta : float, opcional
            Cambio mínimo para considerar mejora (default: 0.0)
        restore_best_weights : bool, opcional
            Si se deben restaurar los mejores pesos al finalizar (default: True)
        """
        self.early_stopping_config['patience'] = patience
        self.early_stopping_config['min_delta'] = min_delta
        self.early_stopping_config['restore_best_weights'] = restore_best_weights
    
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
            
            # Mover modelo al dispositivo disponible
            if isinstance(self.model, nn.Module):
                self.model = self.model.to(self.device)
        
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
            x_cgm_dummy = torch.zeros((1,) + tuple(x_cgm.shape[1:])).to(self.device)
            x_other_dummy = torch.zeros((1,) + tuple(x_other.shape[1:])).to(self.device)
            self.model.build([x_cgm_dummy.shape, x_other_dummy.shape])
        
        # Inicializar buffer de experiencia si el modelo lo requiere
        if hasattr(self.model, 'init_buffer'):
            self.buffer = self.model.init_buffer()
        
        return self.model
    
    def _check_early_stopping(self, val_loss: float) -> bool:
        """
        Verifica si debe activarse el early stopping.
        
        Parámetros:
        -----------
        val_loss : float
            Pérdida de validación actual
            
        Retorna:
        --------
        bool
            True si debe detenerse el entrenamiento, False en caso contrario
        """
        if val_loss < self.early_stopping_config['best_val_loss'] - self.early_stopping_config['min_delta']:
            self.early_stopping_config['best_val_loss'] = val_loss
            
            if self.early_stopping_config['restore_best_weights'] and isinstance(self.model, nn.Module):
                self.early_stopping_config['best_weights'] = {
                    k: v.cpu().clone() for k, v in self.model.state_dict().items()
                }
            
            self.early_stopping_config['counter'] = 0
            return False
        else:
            self.early_stopping_config['counter'] += 1
            
            if self.early_stopping_config['counter'] >= self.early_stopping_config['patience']:
                # Restaurar mejores pesos si corresponde
                if (self.early_stopping_config['restore_best_weights'] and 
                    self.early_stopping_config['best_weights'] is not None and
                    isinstance(self.model, nn.Module)):
                    self.model.load_state_dict(self.early_stopping_config['best_weights'])
                
                return True
            
            return False
    
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
            
            # Verificar early stopping
            should_stop = self._check_early_stopping(val_loss)
            if should_stop:
                print_info("Deteniendo entrenamiento por early stopping")
    
    def _init_training_history(self) -> Dict[str, List[float]]:
        """
        Inicializa el historial de entrenamiento.
        
        Retorna:
        --------
        Dict[str, List[float]]
            Diccionario con listas vacías para métricas
        """
        return {
            "loss": [],
            "actor_loss": [],
            "critic_loss": [],
            "q_loss": [],
            "val_loss": []
        }
    
    def _setup_model_for_training(self) -> None:
        """
        Configura el modelo para el entrenamiento.
        """
        if isinstance(self.model, nn.Module):
            self.model.train()
    
    def _update_progress_description(self, progress_iter, epoch: int, epochs: int, 
                                    total_loss: float, x_cgm_val, history: Dict[str, List[float]]) -> None:
        """
        Actualiza la descripción de la barra de progreso.
        
        Parámetros:
        -----------
        progress_iter : tqdm
            Iterador de progreso
        epoch : int
            Época actual
        epochs : int
            Total de épocas
        total_loss : float
            Pérdida total
        x_cgm_val : np.ndarray
            Datos CGM de validación
        history : Dict[str, List[float]]
            Historial de entrenamiento
        """
        desc = f"Época {epoch+1}/{epochs}, loss: {total_loss:.4f}"
        if x_cgm_val is not None and "val_loss" in history and history["val_loss"]:
            desc += f", val_loss: {history['val_loss'][-1]:.4f}"
        progress_iter.set_description(desc)
    
    def _check_early_stopping_epoch(self, epoch: int, verbose: int) -> bool:
        """
        Verifica si el entrenamiento debe detenerse por early stopping.
        
        Parámetros:
        -----------
        epoch : int
            Época actual
        verbose : int
            Nivel de verbosidad
            
        Retorna:
        --------
        bool
            True si se debe detener, False en caso contrario
        """
        if self.early_stopping_config['counter'] >= self.early_stopping_config['patience']:
            if verbose > 0:
                print_info(f"Early stopping en época {epoch+1}")
            return True
        return False
    
    def _run_training_epoch(self, epoch: int, epochs: int, x_cgm: np.ndarray, x_other: np.ndarray, 
                           y: np.ndarray, x_cgm_val: np.ndarray, x_other_val: np.ndarray, 
                           y_val: np.ndarray, batch_size: int, history: Dict[str, List[float]], 
                           progress_iter, verbose: int) -> bool:
        """
        Ejecuta una época de entrenamiento.
        
        Parámetros:
        -----------
        epoch : int
            Número de época actual
        epochs : int
            Total de épocas
        x_cgm : np.ndarray
            Datos CGM de entrenamiento
        x_other : np.ndarray
            Otras características de entrenamiento
        y : np.ndarray
            Valores objetivo
        x_cgm_val : np.ndarray
            Datos CGM de validación
        x_other_val : np.ndarray
            Otras características de validación
        y_val : np.ndarray
            Valores objetivo de validación
        batch_size : int
            Tamaño del lote
        history : Dict[str, List[float]]
            Historial de entrenamiento
        progress_iter : tqdm
            Iterador de progreso
        verbose : int
            Nivel de verbosidad
            
        Retorna:
        --------
        bool
            True si el entrenamiento debe continuar, False si debe detenerse
        """
        # Calcular recompensas y llenar buffer
        rewards = self._compute_rewards(x_cgm, x_other, y)
        self._fill_buffer(x_cgm, x_other, rewards, y)
        
        # Actualizar redes y procesar métricas
        updates_per_epoch = max(1, len(x_cgm) // batch_size)
        epoch_metrics = self._run_epoch_updates(batch_size, updates_per_epoch)
        total_loss = self._process_epoch_metrics(epoch_metrics, updates_per_epoch, history)
        history["loss"].append(total_loss)
        
        # Validación
        if isinstance(self.model, nn.Module):
            self.model.eval()
        self._validate_model(x_cgm_val, x_other_val, y_val, history)
        if isinstance(self.model, nn.Module):
            self.model.train()
        
        # Mostrar progreso
        if verbose > 0:
            self._update_progress_description(progress_iter, epoch, epochs, total_loss, x_cgm_val, history)
        
        # Verificar early stopping
        return not self._check_early_stopping_epoch(epoch, verbose)
    
    def train(self, x_cgm: np.ndarray, x_other: np.ndarray, y: np.ndarray, 
             validation_data: Optional[Tuple[Tuple[np.ndarray, np.ndarray], np.ndarray]] = None,
             epochs: int = 10, batch_size: int = 32, verbose: int = 1) -> Dict[str, List[float]]:
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
        # Asegurar que el modelo está inicializado
        if self.model is None:
            self.start(x_cgm, x_other, y)
        
        # Preparar datos de validación
        x_cgm_val, x_other_val, y_val = self._unpack_validation_data(validation_data)
        
        # Inicializar historial
        history = self._init_training_history()
        
        # Si el modelo tiene un método fit nativo, úsalo
        if hasattr(self.model, 'fit'):
            return self._train_with_fit(
                x_cgm, x_other, y,
                (x_cgm_val, x_other_val, y_val) if x_cgm_val is not None else None,
                epochs, batch_size
            )
        
        # Configurar modelo para entrenamiento
        self._setup_model_for_training()
        
        # Entrenamiento por épocas
        progress_iter = tqdm(range(epochs)) if verbose > 0 else range(epochs)
        for epoch in progress_iter:
            continue_training = self._run_training_epoch(
                epoch, epochs, x_cgm, x_other, y, 
                x_cgm_val, x_other_val, y_val, 
                batch_size, history, progress_iter, verbose
            )
            
            if not continue_training:
                break
        
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
        
        # Convertir datos a tensores PyTorch si es necesario
        x_cgm_tensor = torch.FloatTensor(x_cgm).to(self.device)
        x_other_tensor = torch.FloatTensor(x_other).to(self.device)
        y_tensor = torch.FloatTensor(y).to(self.device)
        
        # Preparar datos de validación si están disponibles
        val_tensors = None
        if validation_data is not None:
            x_cgm_val, x_other_val, y_val = validation_data
            x_cgm_val_tensor = torch.FloatTensor(x_cgm_val).to(self.device)
            x_other_val_tensor = torch.FloatTensor(x_other_val).to(self.device)
            y_val_tensor = torch.FloatTensor(y_val).to(self.device)
            val_tensors = ([x_cgm_val_tensor, x_other_val_tensor], y_val_tensor)
        
        # Llamar al método fit
        fit_history = self.model.fit(
            [x_cgm_tensor, x_other_tensor], y_tensor,
            validation_data=val_tensors,
            epochs=epochs,
            batch_size=batch_size
        )
        
        # Copiar el historial del modelo
        for key, values in fit_history.items():
            history[key] = values
            
        return history
    
    def _predict_with_pytorch_module(self, x_cgm: np.ndarray, x_other: np.ndarray) -> np.ndarray:
        """
        Realiza predicciones con un módulo PyTorch.
        
        Parámetros:
        -----------
        x_cgm : np.ndarray
            Datos CGM para predicción
        x_other : np.ndarray
            Otras características para predicción
            
        Retorna:
        --------
        np.ndarray
            Predicciones del modelo o None si hay error
        """
        self.model.eval()
        
        # Convertir datos a tensores PyTorch
        x_cgm_tensor = torch.FloatTensor(x_cgm).to(self.device)
        x_other_tensor = torch.FloatTensor(x_other).to(self.device)
        
        with torch.no_grad():
            # Si el modelo tiene un método forward adecuado
            if callable(getattr(self.model, 'forward', None)):
                try:
                    predictions = self.model(x_cgm_tensor, x_other_tensor)
                    return predictions.cpu().numpy()
                except Exception as e:
                    print_debug(f"Error en forward: {e}")
        
        return None
    
    def _predict_with_fallback(self, x_cgm: np.ndarray, x_other: np.ndarray) -> np.ndarray:
        """
        Realiza predicciones usando llamadas directas al modelo.
        
        Parámetros:
        -----------
        x_cgm : np.ndarray
            Datos CGM para predicción
        x_other : np.ndarray
            Otras características para predicción
            
        Retorna:
        --------
        np.ndarray
            Predicciones del modelo
        """
        predictions = np.zeros((len(x_cgm),), dtype=np.float32)
        for i in range(len(x_cgm)):
            state = (x_cgm[i:i+1], x_other[i:i+1])
            if hasattr(self.model, '__call__'):
                predictions[i] = self.model(state)
        return predictions
    
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
        if self.model is None:
            raise ValueError(CONST_MODEL_INIT_ERROR.format("realizar predicciones"))
        
        # Si el modelo es un módulo PyTorch
        if isinstance(self.model, nn.Module):
            result = self._predict_with_pytorch_module(x_cgm, x_other)
            if result is not None:
                return result
        
        # Si el modelo tiene un método predict directo
        if hasattr(self.model, 'predict'):
            return self.model.predict([x_cgm, x_other])
        
        # Si el modelo tiene un método act o get_action
        if hasattr(self.model, 'act') or hasattr(self.model, 'get_action'):
            return self._predict_with_act(x_cgm, x_other)
        
        # Fallback para otros casos
        return self._predict_with_fallback(x_cgm, x_other)
    
    def _get_act_method(self):
        """
        Obtiene el método de acción del modelo.
        
        Retorna:
        --------
        tuple
            (método de acción, booleano indicando si acepta deterministic)
        """
        if hasattr(self.model, 'act'):
            act_method = self.model.act
        elif hasattr(self.model, 'get_action'):
            act_method = self.model.get_action
        else:
            return None, False
        
        has_deterministic_param = 'deterministic' in act_method.__code__.co_varnames
        return act_method, has_deterministic_param
    
    def _process_pytorch_action(self, act_method, state, has_deterministic_param):
        """
        Procesa una acción para un modelo PyTorch.
        
        Parámetros:
        -----------
        act_method : callable
            Método para obtener acciones
        state : tuple
            Estado de entrada
        has_deterministic_param : bool
            Si el método acepta un parámetro deterministic
            
        Retorna:
        --------
        numpy.ndarray
            Acción procesada
        """
        state_tensors = (
            torch.FloatTensor(state[0]).to(self.device),
            torch.FloatTensor(state[1]).to(self.device)
        )
        
        with torch.no_grad():
            action = act_method(state_tensors, deterministic=True) if has_deterministic_param else act_method(state_tensors)
            
            if isinstance(action, torch.Tensor):
                action = action.cpu().numpy()
                
        return action
    
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
        # Obtener el método de acción
        act_method, has_deterministic_param = self._get_act_method()
        if act_method is None:
            return np.zeros((len(x_cgm),), dtype=np.float32)
        
        # Realizar predicciones
        predictions = np.zeros((len(x_cgm),), dtype=np.float32)
        
        for i in range(len(x_cgm)):
            state = (x_cgm[i:i+1], x_other[i:i+1])
            
            # Procesar según el tipo de modelo
            if isinstance(self.model, nn.Module):
                action = self._process_pytorch_action(act_method, state, has_deterministic_param)
            else:
                # Modelo no-PyTorch
                action = act_method(state, deterministic=True) if has_deterministic_param else act_method(state)
            
            predictions[i] = action
                
        return predictions


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
             epochs: int = 10, batch_size: int = 32, verbose: int = 1) -> Dict[str, List[float]]:
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

class DRLModelWrapper(ModelWrapper):
    """
    Wrapper para modelos de aprendizaje por refuerzo profundo que selecciona el wrapper 
    adecuado según el framework.
    
    Parámetros:
    -----------
    model_cls : Callable
        Clase del modelo DRL a instanciar
    framework : str
        Framework a utilizar ('jax' o 'tensorflow')
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
            Framework a utilizar ('jax' o 'tensorflow')
        **model_kwargs
            Argumentos para el constructor del modelo
        """
        super().__init__()
        self.framework = framework.lower()
        
        # Seleccionar el wrapper adecuado según el framework
        if self.framework == 'jax':
            self.wrapper = DRLModelWrapperJAX(model_cls, **model_kwargs)
        else:
            self.wrapper = DRLModelWrapperTF(model_cls, **model_kwargs)
    
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
             epochs: int = 10, batch_size: int = 32, verbose: int = 1) -> Dict[str, List[float]]:
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