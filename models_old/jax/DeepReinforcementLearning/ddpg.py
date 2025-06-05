import os, sys
import jax
import jax.numpy as jnp
import flax
import flax.linen as nn
from flax.training import train_state
import flax.serialization as serialization
import numpy as np
import optax
from typing import Dict, List, Tuple, Any, Optional, Callable, Sequence
from collections import deque
from types import SimpleNamespace
import pickle
import tqdm

PROJECT_ROOT = os.path.abspath(os.getcwd())
sys.path.append(PROJECT_ROOT) 

from constants.constants import CONST_DEFAULT_SEED, CONST_DEFAULT_EPOCHS, CONST_DEFAULT_BATCH_SIZE
from config.models_config_old import DDPG_CONFIG
from custom.DeepReinforcementLearning.drl_model_wrapper import DRLModelWrapper
from custom.printer import print_success, print_error, print_warning, print_info, print_debug

# Constantes para uso repetido
CONST_RELU = "relu"
CONST_TANH = "tanh"
CONST_LEAKY_RELU = "leaky_relu"
CONST_GELU = "gelu"
CONST_DROPOUT = "dropout"
CONST_PARAMS = "params"
CONST_ACTOR_PREFIX = "actor"
CONST_CRITIC_PREFIX = "critic"

FIGURES_DIR = os.path.join(PROJECT_ROOT, "figures", "jax", "ddpg")


class ReplayBuffer:
    """
    Buffer de experiencias para el algoritmo DDPG.
    
    Almacena transiciones (state, action, reward, next_state, done)
    y permite muestrear lotes de manera aleatoria para el entrenamiento.
    
    Parámetros:
    -----------
    capacity : int, opcional
        Capacidad máxima del buffer (default: 100000)
    """
    def __init__(self, capacity: int = 100000):
        self.buffer = deque(maxlen=capacity)
    
    def add(self, state: np.ndarray, action: np.ndarray, reward: float, 
           next_state: np.ndarray, done: float) -> None:
        """
        Añade una transición al buffer.
        
        Parámetros:
        -----------
        state : np.ndarray
            Estado actual
        action : np.ndarray
            Acción tomada
        reward : float
            Recompensa recibida
        next_state : np.ndarray
            Estado siguiente
        done : float
            Indicador de fin de episodio (1.0 si terminó, 0.0 si no)
        """
        self.buffer.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Muestrea un lote aleatorio de transiciones.
        
        Parámetros:
        -----------
        batch_size : int
            Tamaño del lote a muestrear
            
        Retorna:
        --------
        Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]
            (states, actions, rewards, next_states, dones)
        """
        if len(self.buffer) < batch_size:
            # Si no hay suficientes transiciones, devuelve lo que haya
            indices = self.rng.choice(len(self.buffer), size=len(self.buffer), replace=False)
            batch = [self.buffer[i] for i in indices]
        else:
            indices = self.rng.choice(len(self.buffer), size=batch_size, replace=False)
            batch = [self.buffer[i] for i in indices]
            
        states, actions, rewards, next_states, dones = zip(*batch)
        
        return (
            np.array(states, dtype=np.float32),
            np.array(actions, dtype=np.float32),
            np.array(rewards, dtype=np.float32),
            np.array(next_states, dtype=np.float32),
            np.array(dones, dtype=np.float32)
        )
    
    def __len__(self) -> int:
        """
        Retorna la cantidad de transiciones almacenadas.
        
        Retorna:
        --------
        int
            Número de transiciones en el buffer
        """
        return len(self.buffer)


class OUActionNoise:
    """
    Implementa el proceso de ruido de Ornstein-Uhlenbeck para exploración.
    
    Este ruido añade correlación temporal a las acciones para una exploración más efectiva
    en espacios continuos.
    
    Parámetros:
    -----------
    mean : np.ndarray
        Valor medio al que tiende el proceso
    std_deviation : np.ndarray
        Desviación estándar del ruido
    theta : float, opcional
        Velocidad de reversión a la media (default: 0.15)
    dt : float, opcional
        Delta de tiempo para la discretización (default: 1e-2)
    x_initial : Optional[np.ndarray], opcional
        Valor inicial del proceso (default: None)
    seed : int, opcional
        Semilla para la generación de números aleatorios (default: 42)
    """
    def __init__(self, mean: np.ndarray, std_deviation: np.ndarray, theta: float = 0.15, 
                dt: float = 1e-2, x_initial: Optional[np.ndarray] = None, seed: int = CONST_DEFAULT_SEED):
        self.theta = theta
        self.mean = mean
        self.std_dev = std_deviation
        self.dt = dt
        self.x_initial = x_initial
        self.seed = seed
        self.key = jax.random.PRNGKey(seed)
        self.reset()
        
    def __call__(self) -> np.ndarray:
        """
        Genera un nuevo valor de ruido siguiendo el proceso de Ornstein-Uhlenbeck.
        
        Retorna:
        --------
        np.ndarray
            Valor de ruido generado
        """
        # Dividir clave para mantener independencia estadística
        self.key, subkey = jax.random.split(self.key)
        
        # Fórmula para el proceso de Ornstein-Uhlenbeck
        noise = jax.random.normal(subkey, shape=self.mean.shape)
        x = (
            self.x_prev
            + self.theta * (self.mean - self.x_prev) * self.dt
            + self.std_dev * np.sqrt(self.dt) * noise
        )
        self.x_prev = x
        return x
    
    def reset(self) -> None:
        """
        Reinicia el estado del ruido.
        """
        if self.x_initial is not None:
            self.x_prev = self.x_initial
        else:
            self.x_prev = np.zeros_like(self.mean)


class ActorNetwork(nn.Module):
    """
    Red de Actor para DDPG que mapea estados a acciones determinísticas.
    
    Parámetros:
    -----------
    state_dim : int
        Dimensión del espacio de estados
    action_dim : int
        Dimensión del espacio de acciones
    action_high : np.ndarray
        Límite superior del rango de acciones
    action_low : np.ndarray
        Límite inferior del rango de acciones
    hidden_units : Optional[List[int]], opcional
        Unidades en capas ocultas (default: None, usa configuración por defecto)
    """
    state_dim: int
    action_dim: int
    action_high: np.ndarray
    action_low: np.ndarray
    hidden_units: Optional[List[int]] = None
    
    @nn.compact
    def __call__(self, x: jnp.ndarray, training: bool = False) -> jnp.ndarray:
        """
        Pasa la entrada por la red del actor.
        
        Parámetros:
        -----------
        x : jnp.ndarray
            Estado de entrada
        training : bool, opcional
            Indica si está en modo entrenamiento (default: False)
            
        Retorna:
        --------
        jnp.ndarray
            Acción determinística
        """
        # Valores predeterminados para capas ocultas
        hidden_units = self.hidden_units or DDPG_CONFIG['actor_hidden_units']
        action_range = self.action_high - self.action_low
        
        # Procesar a través de capas ocultas
        for i, units in enumerate(hidden_units):
            x = nn.Dense(units, name=f'actor_dense_{i}')(x)
            x = activation_function(x, DDPG_CONFIG['actor_activation'])
            x = nn.LayerNorm(epsilon=DDPG_CONFIG['epsilon'], name=f'actor_ln_{i}')(x)
            x = nn.Dropout(rate=DDPG_CONFIG['dropout_rate'], deterministic=not training)(x)
        
        # Capa de salida con activación tanh y escalado
        raw_actions = nn.Dense(self.action_dim, name='actor_output')(x)
        raw_actions = jnp.tanh(raw_actions)
        
        # Escalar desde [-1, 1] al rango de acción [low, high]
        scaled_actions = 0.5 * (raw_actions + 1.0) * action_range + self.action_low
        
        return scaled_actions


class CriticNetwork(nn.Module):
    """
    Red de Crítico para DDPG que mapea pares (estado, acción) a valores-Q.
    
    Parámetros:
    -----------
    state_dim : int
        Dimensión del espacio de estados
    action_dim : int
        Dimensión del espacio de acciones
    hidden_units : Optional[List[int]], opcional
        Unidades en capas ocultas (default: None, usa configuración por defecto)
    """
    state_dim: int
    action_dim: int
    hidden_units: Optional[List[int]] = None
    
    @nn.compact
    def __call__(self, inputs: Tuple[jnp.ndarray, jnp.ndarray], training: bool = False) -> jnp.ndarray:
        """
        Pasa la entrada por la red del crítico.
        
        Parámetros:
        -----------
        inputs : Tuple[jnp.ndarray, jnp.ndarray]
            Tupla de (estados, acciones)
        training : bool, opcional
            Indica si está en modo entrenamiento (default: False)
            
        Retorna:
        --------
        jnp.ndarray
            Valores Q estimados
        """
        # Valores predeterminados para capas ocultas
        hidden_units = self.hidden_units or DDPG_CONFIG['critic_hidden_units']
        states, actions = inputs
        
        # Procesar el estado inicialmente
        x = states
        for i, units in enumerate(hidden_units[:1]):
            x = nn.Dense(units, name=f'critic_state_dense_{i}')(x)
            x = activation_function(x, DDPG_CONFIG['critic_activation'])
            x = nn.LayerNorm(epsilon=DDPG_CONFIG['epsilon'], name=f'critic_state_ln_{i}')(x)
        
        # Combinar estado procesado con acción
        x = jnp.concatenate([x, actions], axis=-1)
        
        # Procesar a través de capas combinadas
        for i, units in enumerate(hidden_units[1:]):
            x = nn.Dense(units, name=f'critic_combined_dense_{i}')(x)
            x = activation_function(x, DDPG_CONFIG['critic_activation'])
            x = nn.LayerNorm(epsilon=DDPG_CONFIG['epsilon'], name=f'critic_combined_ln_{i}')(x)
            x = nn.Dropout(rate=DDPG_CONFIG['dropout_rate'], deterministic=not training)(x)
        
        # Capa de salida (sin activación)
        q_value = nn.Dense(1, name='critic_output')(x)
        
        return q_value


class DDPGTrainState:
    """
    Clase para almacenar el estado del entrenamiento del algoritmo DDPG.
    
    Parámetros:
    -----------
    actor : train_state.TrainState
        Estado de entrenamiento para la red del actor
    critic : train_state.TrainState
        Estado de entrenamiento para la red del crítico
    target_actor_params : Any
        Parámetros de la red target del actor
    target_critic_params : Any
        Parámetros de la red target del crítico
    key : jax.random.PRNGKey
        Clave para generación de números aleatorios
    """
    def __init__(self, actor: train_state.TrainState, critic: train_state.TrainState,
                target_actor_params: Any, target_critic_params: Any,
                key: jnp.ndarray):
        self.actor = actor
        self.critic = critic
        self.target_actor_params = target_actor_params
        self.target_critic_params = target_critic_params
        self.key = key
        
        # Métricas (como valores acumulados)
        self.actor_loss_sum = 0.0
        self.critic_loss_sum = 0.0
        self.q_value_sum = 0.0
        self.count = 0
    
    def update_metrics(self, actor_loss: float, critic_loss: float, q_value: float) -> None:
        """
        Actualiza las métricas del entrenamiento.
        
        Parámetros:
        -----------
        actor_loss : float
            Pérdida del actor
        critic_loss : float
            Pérdida del crítico
        q_value : float
            Valor Q promedio
        """
        self.actor_loss_sum += actor_loss
        self.critic_loss_sum += critic_loss
        self.q_value_sum += q_value
        self.count += 1
    
    def reset_metrics(self) -> None:
        """
        Reinicia las métricas de entrenamiento.
        """
        self.actor_loss_sum = 0.0
        self.critic_loss_sum = 0.0
        self.q_value_sum = 0.0
        self.count = 0
    
    def get_metrics(self) -> Tuple[float, float, float]:
        """
        Obtiene los valores promedio de las métricas.
        
        Retorna:
        --------
        Tuple[float, float, float]
            (actor_loss_avg, critic_loss_avg, q_value_avg)
        """
        if self.count > 0:
            return (
                self.actor_loss_sum / self.count,
                self.critic_loss_sum / self.count,
                self.q_value_sum / self.count
            )
        return 0.0, 0.0, 0.0


def activation_function(x: jnp.ndarray, name: str) -> jnp.ndarray:
    """
    Aplica una función de activación según su nombre.
    
    Parámetros:
    -----------
    x : jnp.ndarray
        Tensor de entrada
    name : str
        Nombre de la activación ('relu', 'tanh', 'leaky_relu', etc.)
        
    Retorna:
    --------
    jnp.ndarray
        Tensor con activación aplicada
    """
    if name == 'relu':
        return nn.relu(x)
    elif name == 'tanh':
        return jnp.tanh(x)
    elif name == 'leaky_relu':
        return nn.leaky_relu(x)
    elif name == 'gelu':
        return nn.gelu(x)
    else:
        # Por defecto usar ReLU
        return nn.relu(x)


def soft_update(params: Any, target_params: Any, tau: float) -> Any:
    """
    Realiza una actualización suave de los parámetros target.
    
    Parámetros:
    -----------
    params : Any
        Parámetros fuente
    target_params : Any
        Parámetros target a actualizar
    tau : float
        Factor de actualización suave (0 < tau ≤ 1)
        
    Retorna:
    --------
    Any
        Nuevos parámetros target
    """
    # Función para aplicar actualización suave a parámetros individuales
    return jax.tree_map(lambda p, tp: tau * p + (1.0 - tau) * tp, params, target_params)


class DDPG:
    """
    Implementación del algoritmo Deep Deterministic Policy Gradient (DDPG).
    
    DDPG combina ideas de DQN y métodos de policy gradient para manejar
    espacios de acción continuos con políticas determinísticas.
    
    Parámetros:
    -----------
    state_dim : int
        Dimensión del espacio de estados
    action_dim : int
        Dimensión del espacio de acciones
    action_high : np.ndarray
        Límite superior del rango de acciones
    action_low : np.ndarray
        Límite inferior del rango de acciones
    actor_lr : float, opcional
        Tasa de aprendizaje del actor
    critic_lr : float, opcional
        Tasa de aprendizaje del crítico
    gamma : float, opcional
        Factor de descuento
    tau : float, opcional
        Factor de actualización suave para redes target
    buffer_capacity : int, opcional
        Capacidad del buffer de experiencias
    batch_size : int, opcional
        Tamaño del lote para entrenamiento
    noise_std : float, opcional
        Desviación estándar del ruido para exploración
    actor_hidden_units : Optional[List[int]], opcional
        Unidades en capas ocultas del actor
    critic_hidden_units : Optional[List[int]], opcional
        Unidades en capas ocultas del crítico
    seed : int, opcional
        Semilla para generación de números aleatorios
    """
    def __init__(
        self, 
        state_dim: int, 
        action_dim: int,
        action_high: np.ndarray,
        action_low: np.ndarray,
        config: Optional[Dict[str, Any]] = None,
        seed: int = CONST_DEFAULT_SEED
    ):
        # Use provided config or default
        self.config = config or DDPG_CONFIG
        
        # Extract hyperparameters from config
        actor_lr = self.config.get('actor_lr', DDPG_CONFIG['actor_lr'])
        critic_lr = self.config.get('critic_lr', DDPG_CONFIG['critic_lr'])
        self.gamma = self.config.get('gamma', DDPG_CONFIG['gamma'])
        self.tau = self.config.get('tau', DDPG_CONFIG['tau'])
        buffer_capacity = self.config.get('buffer_capacity', DDPG_CONFIG['buffer_capacity'])
        self.batch_size = self.config.get('batch_size', DDPG_CONFIG['batch_size'])
        noise_std = self.config.get('noise_std', DDPG_CONFIG['noise_std'])
        actor_hidden_units = self.config.get('actor_hidden_units')
        critic_hidden_units = self.config.get('critic_hidden_units')
        # Parámetros del entorno y del modelo
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.action_high = action_high
        self.action_low = action_low
        
        # Crear directorio para figuras si no existe
        os.makedirs(FIGURES_DIR, exist_ok=True)
        
        # Inicializar clave para generación de números aleatorios
        self.key = jax.random.PRNGKey(seed)
        
        # Valores predeterminados para capas ocultas
        self.actor_hidden_units = actor_hidden_units or DDPG_CONFIG['actor_hidden_units']
        self.critic_hidden_units = critic_hidden_units or DDPG_CONFIG['critic_hidden_units']
        
        # Definir modelos
        self.actor_def = ActorNetwork(
            state_dim=state_dim, 
            action_dim=action_dim, 
            action_high=action_high, 
            action_low=action_low, 
            hidden_units=self.actor_hidden_units
        )
        self.critic_def = CriticNetwork(
            state_dim=state_dim, 
            action_dim=action_dim, 
            hidden_units=self.critic_hidden_units
        )
        
        # Inicializar modelos
        self.key, actor_key, critic_key = jax.random.split(self.key, 3)
        dummy_state = jnp.zeros((1, state_dim))
        dummy_action = jnp.zeros((1, action_dim))
        
        actor_params = self.actor_def.init(actor_key, dummy_state)
        critic_params = self.critic_def.init(critic_key, (dummy_state, dummy_action))
        
        # Optimizadores
        actor_tx = optax.adam(learning_rate=actor_lr)
        critic_tx = optax.adam(learning_rate=critic_lr)
        
        # Estado de entrenamiento
        actor_state = train_state.TrainState.create(
            apply_fn=self.actor_def.apply,
            params=actor_params,
            tx=actor_tx
        )
        critic_state = train_state.TrainState.create(
            apply_fn=self.critic_def.apply,
            params=critic_params,
            tx=critic_tx
        )
        
        # Parámetros target (inicializados con los mismos valores)
        target_actor_params = actor_params
        target_critic_params = critic_params
        
        # Crear estado de entrenamiento completo
        self.train_state = DDPGTrainState(
            actor=actor_state,
            critic=critic_state,
            target_actor_params=target_actor_params,
            target_critic_params=target_critic_params,
            key=self.key
        )
        
        # Buffer de experiencias
        self.replay_buffer = ReplayBuffer(buffer_capacity)
        
        # Ruido para exploración
        self.noise = OUActionNoise(
            mean=np.zeros(action_dim),
            std_deviation=noise_std * np.ones(action_dim),
            seed=seed
        )
    
    def update_target_networks(self, state: DDPGTrainState, tau: Optional[float] = None) -> DDPGTrainState:
        """
        Actualiza los pesos de las redes target usando soft update.
        
        Parámetros:
        -----------
        state : ddpg_train_state
            Estado actual del entrenamiento
        tau : Optional[float], opcional
            Factor de actualización suave (si None, usa el valor por defecto)
            
        Retorna:
        --------
        ddpg_train_state
            Estado de entrenamiento actualizado
        """
        if tau is None:
            tau = self.tau
        
        target_actor_params = soft_update(state.actor.params, state.target_actor_params, tau)
        target_critic_params = soft_update(state.critic.params, state.target_critic_params, tau)
        
        return DDPGTrainState(
            actor=state.actor,
            critic=state.critic,
            target_actor_params=target_actor_params,
            target_critic_params=target_critic_params,
            key=state.key
        )
    
    def get_action(self, state: np.ndarray, add_noise: bool = True) -> np.ndarray:
        """
        Obtiene una acción determinística para un estado, opcionalmente añadiendo ruido.
        
        Parámetros:
        -----------
        state : np.ndarray
            Estado actual
        add_noise : bool, opcional
            Si se debe añadir ruido para exploración
            
        Retorna:
        --------
        np.ndarray
            Acción seleccionada
        """
        state = jnp.array(state)[jnp.newaxis, :]
        action = self.train_state.actor.apply_fn(
            self.train_state.actor.params, 
            state
        )[0]
        
        action = np.array(action)
        
        if add_noise:
            action += self.noise()
            
        # Clipear al rango válido de acciones
        action = np.clip(action, self.action_low, self.action_high)
        
        return action
    
    def train_step(self, state: DDPGTrainState, batch: Tuple) -> Tuple[DDPGTrainState, Dict]:
        """
        Realiza un paso de entrenamiento DDPG con un lote de experiencias.
        
        Parámetros:
        -----------
        state : ddpg_train_state
            Estado actual del entrenamiento
        batch : Tuple
            Lote de experiencias (states, actions, rewards, next_states, dones)
            
        Retorna:
        --------
        Tuple[ddpg_train_state, Dict]
            Estado actualizado y diccionario con métricas
        """
        states, actions, rewards, next_states, dones = [jnp.array(x) for x in batch]
        
        # Dividir clave para operaciones aleatorias
        key, _, _ = jax.random.split(state.key, 3)
        new_state = DDPGTrainState(
            actor=state.actor,
            critic=state.critic,
            target_actor_params=state.target_actor_params,
            target_critic_params=state.target_critic_params,
            key=key
        )
        
        # Función para actualizar el crítico
        def update_critic(cstate, inputs):
            states, actions, rewards, next_states, dones = inputs
            
            # Predecir acciones target para los siguientes estados
            target_actions = self.actor_def.apply(
                new_state.target_actor_params, 
                next_states
            )
            
            # Predecir Q-values target
            target_q_values = self.critic_def.apply(
                new_state.target_critic_params, 
                (next_states, target_actions)
            )
            
            # Calcular Q-values objetivo usando la ecuación de Bellman
            target_q = rewards[:, jnp.newaxis] + (1 - dones[:, jnp.newaxis]) * self.gamma * target_q_values
            
            # Función de pérdida del crítico
            def critic_loss_fn(params):
                # Predecir Q-values actuales
                current_q = self.critic_def.apply(
                    params, 
                    (states, actions)
                )
                
                # Calcular pérdida (error cuadrático medio)
                loss = jnp.mean(jnp.square(target_q - current_q))
                return loss, current_q
            
            # Calcular pérdida y gradientes
            (critic_loss, current_q), grads = jax.value_and_grad(
                critic_loss_fn, has_aux=True)(cstate.params)
            
            # Actualizar pesos del crítico
            new_cstate = cstate.apply_gradients(grads=grads)
            
            return new_cstate, (critic_loss, jnp.mean(current_q))
        
        # Función para actualizar el actor
        def update_actor(astate, states):
            # Función de pérdida del actor
            def actor_loss_fn(params):
                # Predecir acciones para los estados actuales
                actor_actions = self.actor_def.apply(
                    params, 
                    states
                )
                
                # Calcular Q-values para estas acciones
                actor_q_values = self.critic_def.apply(
                    new_state.critic.params, 
                    (states, actor_actions)
                )
                
                # Pérdida del actor (negativo del Q-value promedio)
                # Queremos maximizar Q-value, así que minimizamos su negativo
                loss = -jnp.mean(actor_q_values)
                return loss
            
            # Calcular pérdida y gradientes
            actor_loss, grads = jax.value_and_grad(actor_loss_fn)(astate.params)
            
            # Actualizar pesos del actor
            new_astate = astate.apply_gradients(grads=grads)
            
            return new_astate, actor_loss
        
        # Actualizar crítico
        new_critic, (critic_loss, q_value) = update_critic(
            state.critic, 
            (states, actions, rewards, next_states, dones)
        )
        
        # Actualizar actor
        new_actor, actor_loss = update_actor(state.actor, states)
        
        # Actualizar estado de entrenamiento
        new_state = DDPGTrainState(
            actor=new_actor,
            critic=new_critic,
            target_actor_params=state.target_actor_params,
            target_critic_params=state.target_critic_params,
            key=key
        )
        
        # Actualizar redes target
        new_state = self.update_target_networks(new_state)
        
        # Guardar métricas
        metrics = {
            'actor_loss': actor_loss,
            'critic_loss': critic_loss,
            'q_value': q_value
        }
        
        return new_state, metrics
    
    def _select_action(self, state: np.ndarray, step_counter: int, warmup_steps: int) -> np.ndarray:
        """
        Selecciona una acción basada en el estado actual y fase de entrenamiento.
        
        Parámetros:
        -----------
        state : np.ndarray
            Estado actual
        step_counter : int
            Contador de pasos
        warmup_steps : int
            Pasos iniciales de exploración pura
            
        Retorna:
        --------
        np.ndarray
            Acción seleccionada
        """
        if step_counter < warmup_steps:
            # Exploración inicial uniforme
            self.key, subkey = jax.random.split(self.key)
            return jax.random.uniform(
                subkey, 
                shape=(self.action_dim,),
                minval=self.action_low,
                maxval=self.action_high
            )
        else:
            # Política del actor con ruido
            return self.get_action(state, add_noise=True)
    
    def _update_model(self, step_counter: int, update_every: int) -> Tuple[Optional[float], Optional[float]]:
        """
        Actualiza los modelos si es momento de hacerlo.
        
        Parámetros:
        -----------
        step_counter : int
            Contador de pasos totales
        update_every : int
            Frecuencia de actualización
            
        Retorna:
        --------
        Tuple[Optional[float], Optional[float]]
            (actor_loss, critic_loss) o (None, None) si no se actualizó
        """
        if step_counter % update_every == 0:
            # Muestrear batch
            batch = self.replay_buffer.sample(self.batch_size)
            
            # Realizar actualización
            self.train_state, metrics = self.train_step(self.train_state, batch)
            
            return metrics['actor_loss'], metrics['critic_loss']
        return None, None
    
    def _update_history(self, history: Dict, episode_reward: float, 
                      episode_actor_loss: List, episode_critic_loss: List) -> Dict:
        """
        Actualiza el historial de entrenamiento con los resultados del episodio.
        
        Parámetros:
        -----------
        history : Dict
            Historial de entrenamiento a actualizar
        episode_reward : float
            Recompensa del episodio
        episode_actor_loss : List
            Lista de pérdidas del actor durante el episodio
        episode_critic_loss : List
            Lista de pérdidas del crítico durante el episodio
            
        Retorna:
        --------
        Dict
            Historial actualizado
        """
        history['episode_rewards'].append(episode_reward)
        
        if episode_actor_loss:
            avg_actor_loss = sum(episode_actor_loss) / len(episode_actor_loss)
            avg_critic_loss = sum(episode_critic_loss) / len(episode_critic_loss)
            history['actor_losses'].append(avg_actor_loss)
            history['critic_losses'].append(avg_critic_loss)
            
            q_values = []
            for i in range(len(episode_actor_loss)):
                # Podemos estimar los valores Q a partir de las pérdidas
                # Valor Q aproximado es lo inverso de la pérdida del actor
                q_value = -episode_actor_loss[i]
                q_values.append(q_value)
            
            avg_q = sum(q_values) / len(q_values) if q_values else 0.0
            history['avg_q_values'].append(avg_q)
        else:
            # Si no hay pérdidas, registrar NaN para mantener consistencia
            history['actor_losses'].append(float('nan'))
            history['critic_losses'].append(float('nan'))
            history['avg_q_values'].append(float('nan'))
        
        return history
    
    def _log_progress(self, episode: int, episodes: int, episode_reward_history: List, 
                    history: Dict, log_interval: int, best_reward: float) -> float:
        """
        Registra y muestra el progreso del entrenamiento periódicamente.
        
        Parámetros:
        -----------
        episode : int
            Episodio actual
        episodes : int
            Total de episodios
        episode_reward_history : List
            Historial reciente de recompensas
        history : Dict
            Historial completo de entrenamiento
        log_interval : int
            Intervalo para mostrar información
        best_reward : float
            Mejor recompensa hasta el momento
            
        Retorna:
        --------
        float
            Mejor recompensa actualizada
        """
        if (episode + 1) % log_interval == 0:
            # Calcular promedio de recompensas recientes
            avg_reward = sum(episode_reward_history) / len(episode_reward_history)
            
            # Verificar si hay una nueva mejor recompensa
            new_best = ""
            if avg_reward > best_reward:
                best_reward = avg_reward
                new_best = " (nuevo mejor)"
            
            # Obtener promedios de pérdidas recientes (si están disponibles)
            actor_losses = [l for l in history['actor_losses'][-log_interval:] if not np.isnan(l)]
            critic_losses = [l for l in history['critic_losses'][-log_interval:] if not np.isnan(l)]
            
            avg_actor_loss = sum(actor_losses) / len(actor_losses) if actor_losses else float('nan')
            avg_critic_loss = sum(critic_losses) / len(critic_losses) if critic_losses else float('nan')
            
            # Mostrar progreso
            print(f"Episodio {episode+1}/{episodes} - "
                  f"Recompensa promedio: {avg_reward:.2f}{new_best} - "
                  f"Actor loss: {avg_actor_loss:.4f} - "
                  f"Critic loss: {avg_critic_loss:.4f}")
        
        return best_reward
    
    def _run_episode(self, env, max_steps: int, step_counter: int, warmup_steps: int, 
                   update_every: int, render: bool) -> Tuple[float, List[float], List[float], int]:
        """
        Ejecuta un episodio completo de entrenamiento.
        
        Parámetros:
        -----------
        env : gym.Env
            Entorno donde ejecutar el episodio
        max_steps : int
            Pasos máximos por episodio
        step_counter : int
            Contador de pasos totales
        warmup_steps : int
            Pasos iniciales de exploración pura
        update_every : int
            Frecuencia de actualización del modelo
        render : bool
            Si se debe renderizar el entorno
            
        Retorna:
        --------
        Tuple[float, List[float], List[float], int]
            (episode_reward, episode_actor_loss, episode_critic_loss, step_counter)
        """
        state, _ = env.reset()
        state = np.array(state, dtype=np.float32)
        episode_reward = 0
        episode_actor_loss = []
        episode_critic_loss = []
        
        for _ in range(max_steps):
            step_counter += 1
            
            # Seleccionar acción
            action = self._select_action(state, step_counter, warmup_steps)
            
            # Ejecutar acción
            next_state, reward, done, _, _ = env.step(action)
            next_state = np.array(next_state, dtype=np.float32)
            
            # Renderizar si es necesario
            if render:
                env.render()
            
            # Guardar transición en buffer
            self.replay_buffer.add(state, action, reward, next_state, float(done))
            
            # Actualizar estado y recompensa
            state = next_state
            episode_reward += reward
            
            # Entrenar modelo si hay suficientes datos
            if len(self.replay_buffer) > self.batch_size and step_counter >= warmup_steps:
                actor_loss, critic_loss = self._update_model(step_counter, update_every)
                if actor_loss is not None:
                    episode_actor_loss.append(float(actor_loss))
                    episode_critic_loss.append(float(critic_loss))
            
            if done:
                break
                
        return episode_reward, episode_actor_loss, episode_critic_loss, step_counter
    
    def train(self, env, episodes: int = 1000, max_steps: int = 1000, warmup_steps: int = 1000, 
             update_every: int = 1, render: bool = False, log_interval: int = 10) -> Dict:
        """
        Entrena el agente DDPG en un entorno dado.
        
        Parámetros:
        -----------
        env : gym.Env
            Entorno de Gym o compatible
        episodes : int, opcional
            Número máximo de episodios (default: 1000)
        max_steps : int, opcional
            Pasos máximos por episodio (default: 1000)
        warmup_steps : int, opcional
            Pasos iniciales para recolectar experiencias antes de entrenar (default: 1000)
        update_every : int, opcional
            Frecuencia de actualización (default: 1)
        render : bool, opcional
            Mostrar entorno gráficamente (default: False)
        log_interval : int, opcional
            Intervalo para mostrar información (default: 10)
            
        Retorna:
        --------
        Dict
            Historia de entrenamiento
        """
        history = {
            'episode_rewards': [],
            'actor_losses': [],
            'critic_losses': [],
            'avg_q_values': []
        }
        
        # Variables para seguimiento de progreso
        best_reward = -float('inf')
        episode_reward_history = []
        step_counter = 0
        
        for episode in range(episodes):
            # Ejecutar un episodio completo
            episode_reward, episode_actor_loss, episode_critic_loss, step_counter = self._run_episode(
                env, max_steps, step_counter, warmup_steps, update_every, render
            )
            
            # Actualizar historial
            history = self._update_history(history, episode_reward, episode_actor_loss, episode_critic_loss)
            
            # Resetear el ruido para el siguiente episodio
            self.noise.reset()
            
            # Actualizar historial de recompensas
            episode_reward_history.append(episode_reward)
            if len(episode_reward_history) > log_interval:
                episode_reward_history.pop(0)
            
            # Registrar progreso
            best_reward = self._log_progress(
                episode, episodes, episode_reward_history, history, log_interval, best_reward
            )
        
        return history
    
    def evaluate(self, env, episodes: int = 10, render: bool = False) -> float:
        """
        Evalúa el agente DDPG en un entorno dado sin exploración.
        
        Parámetros:
        -----------
        env : gym.Env
            Entorno de Gym o compatible
        episodes : int, opcional
            Número de episodios para evaluar (default: 10)
        render : bool, opcional
            Si se debe renderizar el entorno (default: False)
            
        Retorna:
        --------
        float
            Recompensa promedio obtenida durante la evaluación
        """
        rewards = []
        
        for episode in range(episodes):
            state, _ = env.reset()
            state = np.array(state, dtype=np.float32)
            episode_reward = 0
            done = False
            
            while not done:
                # Seleccionar acción determinística (sin ruido)
                action = self.get_action(state, add_noise=False)
                
                # Ejecutar acción
                next_state, reward, done, _, _ = env.step(action)
                next_state = np.array(next_state, dtype=np.float32)
                
                # Renderizar si es necesario
                if render:
                    env.render()
                
                # Actualizar estado y recompensa
                state = next_state
                episode_reward += reward
            
            rewards.append(episode_reward)
            print(f"Episodio de Evaluación {episode+1}/{episodes} - Recompensa: {episode_reward:.2f}")
        
        avg_reward = np.mean(rewards)
        print(f"Recompensa Promedio de Evaluación: {avg_reward:.2f}")
        
        return avg_reward
    
    def save_models(self, actor_path: str, critic_path: str) -> None:
        """
        Guarda los modelos del actor y crítico en archivos separados.
        
        Parámetros:
        -----------
        actor_path : str
            Ruta para guardar el modelo del actor
        critic_path : str
            Ruta para guardar el modelo del crítico
        """
        import flax.serialization as serialization
        
        # Guardar parámetros
        with open(actor_path, 'wb') as f:
            f.write(serialization.to_bytes(self.train_state.actor.params))
            
        with open(critic_path, 'wb') as f:
            f.write(serialization.to_bytes(self.train_state.critic.params))
            
        print(f"Modelos guardados en {actor_path} y {critic_path}")
    
    def load_models(self, actor_path: str, critic_path: str) -> None:
        """
        Carga los modelos del actor y crítico desde archivos.
        
        Parámetros:
        -----------
        actor_path : str
            Ruta para cargar el modelo del actor
        critic_path : str
            Ruta para cargar el modelo del crítico
        """
        import flax.serialization as serialization
        
        # Cargar parámetros
        with open(actor_path, 'rb') as f:
            actor_params = serialization.from_bytes(self.train_state.actor.params, f.read())
            
        with open(critic_path, 'rb') as f:
            critic_params = serialization.from_bytes(self.train_state.critic.params, f.read())
            
        # Actualizar estados
        self.train_state = DDPGTrainState(
            actor=train_state.TrainState(
                step=self.train_state.actor.step,
                apply_fn=self.train_state.actor.apply_fn,
                params=actor_params,
                tx=self.train_state.actor.tx,
                opt_state=self.train_state.actor.opt_state
            ),
            critic=train_state.TrainState(
                step=self.train_state.critic.step,
                apply_fn=self.train_state.critic.apply_fn,
                params=critic_params,
                tx=self.train_state.critic.tx,
                opt_state=self.train_state.critic.opt_state
            ),
            target_actor_params=actor_params,
            target_critic_params=critic_params,
            key=self.train_state.key
        )
        
        print(f"Modelos cargados desde {actor_path} y {critic_path}")
    
    def visualize_training(self, history: Dict, window_size: int = 10) -> None:
        """
        Visualiza los resultados del entrenamiento.
        
        Parámetros:
        -----------
        history : Dict
            Historia de entrenamiento
        window_size : int, opcional
            Tamaño de ventana para suavizado (default: 10)
        """
        import matplotlib.pyplot as plt
        
        # Función para aplicar suavizado
        def smooth(data, window_size):
            kernel = np.ones(window_size) / window_size
            return np.convolve(data, kernel, mode='valid')
        
        # Crear figura con múltiples subplots
        _, axs = plt.subplots(2, 2, figsize=(15, 10))
        
        # Graficar recompensas
        rewards = history['episode_rewards']
        axs[0, 0].plot(rewards, alpha=0.3, color='blue', label='Raw')
        if len(rewards) > window_size:
            axs[0, 0].plot(range(window_size-1, len(rewards)), smooth(rewards, window_size), 
                         color='blue', label=f'Smoothed (window={window_size})')
        axs[0, 0].set_title('Episode Rewards')
        axs[0, 0].set_xlabel('Episode')
        axs[0, 0].set_ylabel('Reward')
        axs[0, 0].legend()
        axs[0, 0].grid(alpha=0.3)
        
        # Graficar pérdida del actor
        actor_losses = history['actor_losses']
        valid_losses = [l for l in actor_losses if not np.isnan(l)]
        if valid_losses:
            axs[0, 1].plot(valid_losses, alpha=0.3, color='green', label='Raw')
            if len(valid_losses) > window_size:
                axs[0, 1].plot(range(window_size-1, len(valid_losses)), 
                             smooth(valid_losses, window_size), 
                             color='green', label=f'Smoothed (window={window_size})')
        axs[0, 1].set_title('Actor Loss')
        axs[0, 1].set_xlabel('Episode')
        axs[0, 1].set_ylabel('Loss')
        axs[0, 1].legend()
        axs[0, 1].grid(alpha=0.3)
        
        # Graficar pérdida del crítico
        critic_losses = history['critic_losses']
        valid_losses = [l for l in critic_losses if not np.isnan(l)]
        if valid_losses:
            axs[1, 0].plot(valid_losses, alpha=0.3, color='red', label='Raw')
            if len(valid_losses) > window_size:
                axs[1, 0].plot(range(window_size-1, len(valid_losses)), 
                             smooth(valid_losses, window_size), 
                             color='red', label=f'Smoothed (window={window_size})')
        axs[1, 0].set_title('Critic Loss')
        axs[1, 0].set_xlabel('Episode')
        axs[1, 0].set_ylabel('Loss')
        axs[1, 0].legend()
        axs[1, 0].grid(alpha=0.3)
        
        # Graficar valores Q promedio
        q_values = history['avg_q_values']
        valid_q = [q for q in q_values if not np.isnan(q)]
        if valid_q:
            axs[1, 1].plot(valid_q, alpha=0.3, color='purple', label='Raw')
            if len(valid_q) > window_size:
                axs[1, 1].plot(range(window_size-1, len(valid_q)), 
                             smooth(valid_q, window_size), 
                             color='purple', label=f'Smoothed (window={window_size})')
        axs[1, 1].set_title('Average Q Values')
        axs[1, 1].set_xlabel('Episode')
        axs[1, 1].set_ylabel('Q Value')
        axs[1, 1].legend()
        axs[1, 1].grid(alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(FIGURES_DIR, 'training_results.png'))
        plt.show()

class DDPGWrapper:
    """
    Wrapper para hacer que el agente DDPG sea compatible con la interfaz de modelos de aprendizaje profundo.
    
    Parámetros:
    -----------
    ddpg_agent : DDPG
        Agente DDPG inicializado
    cgm_shape : Tuple[int, ...]
        Forma de los datos CGM
    other_features_shape : Tuple[int, ...]
        Forma de otras características
    """
    
    def __init__(
        self, 
        ddpg_agent: DDPG, 
        cgm_shape: Tuple[int, ...], 
        other_features_shape: Tuple[int, ...],
    ) -> None:
        """
        Inicializa el wrapper DDPG.
        
        Parámetros:
        -----------
        ddpg_agent : DDPG
            Agente DDPG inicializado
        cgm_shape : Tuple[int, ...]
            Forma de los datos CGM
        other_features_shape : Tuple[int, ...]
            Forma de otras características
        """
        self.ddpg_agent = ddpg_agent
        self.cgm_shape = cgm_shape
        self.other_features_shape = other_features_shape
        self.state_dim = ddpg_agent.state_dim
        
        # Inicializar los codificadores
        self._setup_encoders()
    
    def _setup_encoders(self) -> None:
        """
        Configura los codificadores para proyectar datos de entrada al espacio de estados.
        """
        # Semilla para reproducibilidad
        rng_key = jax.random.PRNGKey(CONST_DEFAULT_SEED)
        key_cgm, key_other = jax.random.split(rng_key)
        
        # Obtener dimensiones y convertirlas explícitamente a enteros
        cgm_dim = int(np.prod(self.cgm_shape))
        other_dim = int(np.prod(self.other_features_shape))
        
        # Inicializar pesos para la proyección de dimensiones
        self.cgm_weight = jax.random.normal(key_cgm, (cgm_dim, self.ddpg_agent.state_dim // 2))
        self.other_weight = jax.random.normal(key_other, (other_dim, self.ddpg_agent.state_dim // 2))
    
    def _create_encoder_fn(self, weights: jnp.ndarray) -> Callable:
        """
        Crea una función de codificación.
        
        Parámetros:
        -----------
        weights : jnp.ndarray
            Pesos para la codificación
            
        Retorna:
        --------
        Callable
            Función de codificación
        """
        def encoder_fn(x: jnp.ndarray) -> jnp.ndarray:
            # Aplanar la entrada si es necesario
            if x.ndim > 2:
                x_flat = x.reshape(x.shape[0], -1)
            else:
                x_flat = x
            
            # Proyectar a la dimensión del espacio de estado
            encoded = jnp.dot(x_flat, weights)
            return jax.nn.tanh(encoded)  # Activación tanh para limitar valores
            
        return encoder_fn
    
    def __call__(self, cgm_input: jnp.ndarray, other_input: jnp.ndarray, training: bool = False) -> jnp.ndarray:
        """
        Llamada principal del modelo.
        
        Parámetros:
        -----------
        cgm_input : jnp.ndarray
            Datos de entrada CGM
        other_input : jnp.ndarray
            Otras características de entrada
        training : bool, opcional
            Indica si está en modo entrenamiento (default: False)
            
        Retorna:
        --------
        jnp.ndarray
            Predicciones del modelo
        """
        return self.predict([cgm_input, other_input])
    
    def predict(self, inputs: List[jnp.ndarray]) -> jnp.ndarray:
        """
        Realiza predicciones usando el modelo DDPG entrenado.
        
        Parámetros:
        -----------
        inputs : List[jnp.ndarray]
            Lista de tensores de entrada [cgm_data, other_features]
            
        Retorna:
        --------
        jnp.ndarray
            Predicciones del modelo (dosis de insulina)
        """
        # Extraer entradas
        if isinstance(inputs, list) and len(inputs) == 2:
            cgm_data, other_features = inputs
        elif isinstance(inputs, tuple) and len(inputs) == 2:
            cgm_data, other_features = inputs
        else:
            raise ValueError("La entrada debe ser una lista o tupla con [cgm_data, other_features]")
    
        # Obtener tamaño del batch
        batch_size = cgm_data.shape[0]
    
        # Inicializar array de predicciones
        predictions = np.zeros((batch_size, 1), dtype=np.float32)
    
        # Añadir barra de progreso
        batch_iterator = tqdm.tqdm(range(batch_size), desc="Prediciendo dosis DDPG", disable=False)
    
        # Procesar cada muestra del batch con barra de progreso
        for i in batch_iterator:
            # Extraer muestra
            cgm_sample = cgm_data[i:i+1]
            other_sample = other_features[i:i+1]
        
            # Codificar entradas
            cgm_encoder = self._create_encoder_fn(self.cgm_weight)
            other_encoder = self._create_encoder_fn(self.other_weight)
        
            encoded_cgm = cgm_encoder(cgm_sample)
            encoded_other = other_encoder(other_sample)
        
            # Combinar características
            encoded_state = jnp.concatenate([encoded_cgm.flatten(), encoded_other.flatten()])
        
            # Obtener acción determinística (sin ruido)
            action = self.ddpg_agent.get_action(encoded_state, add_noise=False)
        
            # Guardar predicción
            predictions[i, 0] = action[0]  # Tomar el primer valor de la acción
    
        return predictions
    
    def _calculate_metrics(self, train_preds: jnp.ndarray, y: jnp.ndarray, 
                      validation_data: Optional[Tuple] = None, verbose: int = 1) -> Dict:
        """
        Calcula métricas de rendimiento del modelo.
        
        Parámetros:
        -----------
        train_preds : jnp.ndarray
            Predicciones del modelo en datos de entrenamiento
        y : jnp.ndarray
            Valores objetivo (dosis de insulina)
        validation_data : Optional[Tuple], opcional
            Datos de validación como (x_val, y_val) (default: None)
        verbose : int, opcional
            Nivel de verbosidad (default: 1)
            
        Retorna:
        --------
        Dict
            Diccionario con métricas calculadas
        """
        # Asegurar que las formas sean compatibles para cálculos
        y_flat = y.reshape(-1)
        train_preds_flat = train_preds.reshape(-1)
        
        # Calcular métricas de rendimiento
        mae = float(jnp.mean(jnp.abs(train_preds_flat - y_flat)))
        mse = float(jnp.mean((train_preds_flat - y_flat) ** 2))
        rmse = float(jnp.sqrt(mse))
        
        # Calcular R²
        y_mean = jnp.mean(y_flat)
        ss_total = jnp.sum((y_flat - y_mean) ** 2)
        ss_residual = jnp.sum((y_flat - train_preds_flat) ** 2)
        r2 = float(1 - (ss_residual / (ss_total + 1e-10)))
        
        # Guardar métricas en el historial
        self.history['mae'] = [mae]
        self.history['mse'] = [mse]
        self.history['rmse'] = [rmse]
        self.history['r2'] = [r2]
        
        # Calcular métricas de validación si se proporcionan
        if validation_data:
            val_x, val_y = validation_data
            val_preds = self.predict(val_x)
            
            # Aplanar para cálculos seguros
            val_y_flat = val_y.reshape(-1)
            val_preds_flat = val_preds.reshape(-1)
            
            val_mae = float(jnp.mean(jnp.abs(val_preds_flat - val_y_flat)))
            val_mse = float(jnp.mean((val_preds_flat - val_y_flat) ** 2))
            val_rmse = float(jnp.sqrt(val_mse))
            
            val_y_mean = jnp.mean(val_y_flat)
            val_ss_total = jnp.sum((val_y_flat - val_y_mean) ** 2)
            val_ss_residual = jnp.sum((val_y_flat - val_preds_flat) ** 2)
            val_r2 = float(1 - (val_ss_residual / (val_ss_total + 1e-10)))
            
            self.history['val_mae'] = [val_mae]
            self.history['val_mse'] = [val_mse]
            self.history['val_rmse'] = [val_rmse]
            self.history['val_r2'] = [val_r2]
            self.history['val_predictions'] = val_preds
        
        if verbose > 0:
            print(f"Training metrics - MAE: {mae:.4f}, RMSE: {rmse:.4f}, R²: {r2:.4f}")
            if validation_data:
                print(f"Validation metrics - MAE: {val_mae:.4f}, RMSE: {val_rmse:.4f}, R²: {val_r2:.4f}")
        
        print_debug(f"self.history: {self.history}")
        return self.history
    
    def fit(
        self, 
        x: List[jnp.ndarray], 
        y: jnp.ndarray, 
        validation_data: Optional[Tuple] = None, 
        epochs: int = CONST_DEFAULT_EPOCHS,
        batch_size: int = CONST_DEFAULT_BATCH_SIZE,
        callbacks: List = None,
        verbose: int = 0
    ) -> Dict:
        """
        Entrena el modelo con los datos proporcionados.
        
        Parámetros:
        -----------
        x : List[jnp.ndarray]
            Lista de entradas [cgm_input, other_input]
        y : jnp.ndarray
            Valores objetivo (acciones)
        validation_data : Optional[Tuple], opcional
            Datos de validación como ([x_cgm_val, x_other_val], y_val) (default: None)
        epochs : int, opcional
            Número de épocas de entrenamiento (default: 10)
        batch_size : int, opcional
            Tamaño de lote (default: 32)
        callbacks : List, opcional
            Lista de callbacks (default: None)
        verbose : int, opcional
            Nivel de verbosidad (default: 0)
            
        Retorna:
        --------
        Dict
            Historial de entrenamiento
        """
        # Extraer datos de entrenamiento
        x_cgm, x_other = x
        
        # Crear entorno de entrenamiento para el agente DDPG
        env = self._create_training_environment(x_cgm, x_other, y)
        
        # Entrenar el agente DDPG
        history = self.ddpg_agent.train(
            env, 
            episodes=epochs,
            max_steps=batch_size,
            warmup_steps=min(500, batch_size * 5),
            update_every=1,
            render=False,
            log_interval=max(1, epochs // 10)
        )
        
        # Inicializar claves estándar en el historial si no existen
        if 'loss' not in history:
            history['loss'] = []
        if 'val_loss' not in history:
            history['val_loss'] = []
        if 'predictions' not in history:
            history['predictions'] = []
        if 'val_predictions' not in history:
            history['val_predictions'] = []
        
        print_info("Calculando métricas...")
        # Calcular pérdida en datos de entrenamiento
        train_preds = self.predict(x)
        train_loss = float(jnp.mean((train_preds.flatten() - y) ** 2))
        history['predictions'] = train_preds
        history['loss'].append(train_loss)
        
        # Evaluar en datos de validación si se proporcionan
        if validation_data:
            val_x, val_y = validation_data
            val_preds = self.predict(val_x)
            val_loss = float(jnp.mean((val_preds.flatten() - val_y) ** 2))
            history['val_predictions'] = val_preds
            history['val_loss'].append(val_loss)
        
        # Para compatibilidad con la API de visualización
        self.history = history
        
        self._calculate_metrics(train_preds, y, validation_data, verbose=verbose)
        
        if verbose > 0:
            print_success("Entrenamiento del modelo DDPG completado.")
            print_info(f"Pérdida de entrenamiento: {train_loss:.4f}")
            if validation_data:
                print_info(f"Pérdida de validación: {val_loss:.4f}")
        
        return self.history
    
    def _create_training_environment(
        self, 
        cgm_data: jnp.ndarray, 
        other_features: jnp.ndarray, 
        targets: jnp.ndarray
    ) -> Any:
        """
        Crea un entorno de entrenamiento para el agente DDPG.
        
        Parámetros:
        -----------
        cgm_data : jnp.ndarray
            Datos CGM
        other_features : jnp.ndarray
            Otras características
        targets : jnp.ndarray
            Valores objetivo (acciones)
            
        Retorna:
        --------
        Any
            Entorno de entrenamiento
        """
        state_dim = self.ddpg_agent.state_dim
        # Crear un entorno personalizado para el entrenamiento DDPG
        class DDPGEnv:
            def __init__(self, cgm_data, other_features, targets, cgm_encoder, other_encoder):
                self.cgm_data = cgm_data
                self.other_features = other_features
                self.targets = targets
                self.cgm_encoder = cgm_encoder
                self.other_encoder = other_encoder
                self.current_idx = 0
                self.num_samples = len(targets)
                self.rng = np.random.default_rng(seed=CONST_DEFAULT_SEED)
                self.action_space = SimpleNamespace(
                    sample=lambda: self.rng.uniform(0, 15, (1,)),
                    n=1,
                    shape=(1,)
                )
                self.observation_space = SimpleNamespace(
                    shape=(state_dim,)
                )
                
            def reset(self):
                # Seleccionar un punto aleatorio para iniciar
                self.current_idx = self.rng.integers(0, self.num_samples)
                
                # Obtener estado actual
                state = self._get_state(self.current_idx)
                
                return state, {}
                
            def step(self, action):
                # Calcular error con respecto al objetivo
                true_action = self.targets[self.current_idx]
                error = np.abs(action[0] - true_action)
                
                # Recompensa inversamente proporcional al error
                reward = -error
                
                # Avanzar al siguiente punto
                self.current_idx = (self.current_idx + 1) % self.num_samples
                
                # Obtener siguiente estado
                next_state = self._get_state(self.current_idx)
                
                # Verificar si el episodio ha terminado
                done = False  # En este caso, los episodios no terminan naturalmente
                
                return next_state, reward, done, False, {}
                
            def _get_state(self, idx):
                # Codificar estado actual
                cgm_sample = self.cgm_data[idx:idx+1]
                other_sample = self.other_features[idx:idx+1]
                
                encoded_cgm = self.cgm_encoder(cgm_sample)[0]
                encoded_other = self.other_encoder(other_sample)[0]
                
                # Combinar características codificadas
                return np.concatenate([encoded_cgm, encoded_other])
        
        # Crear encoders
        cgm_encoder = self._create_encoder_fn(self.cgm_weight)
        other_encoder = self._create_encoder_fn(self.other_weight)
        
        # Crear y retornar el entorno
        return DDPGEnv(cgm_data, other_features, targets, cgm_encoder, other_encoder)
    
    def save(self, filepath: str) -> None:
        """
        Guarda el modelo en disco usando serialización apropiada para JAX.
        
        Parámetros:
        -----------
        filepath : str
            Ruta donde guardar el modelo
        """
        try:
            # Crear directorio si no existe
            os.makedirs(os.path.dirname(filepath), exist_ok=True)
            
            serialized_data = {
                'cgm_weight': np.array(self.cgm_weight),
                'other_weight': np.array(self.other_weight),
                'actor_params': serialization.to_bytes(self.ddpg_agent.train_state.actor.params),
                'critic_params': serialization.to_bytes(self.ddpg_agent.train_state.critic.params),
                'target_actor_params': serialization.to_bytes(self.ddpg_agent.train_state.target_actor_params),
                'target_critic_params': serialization.to_bytes(self.ddpg_agent.train_state.target_critic_params),
                'agent_params': {
                    'state_dim': self.ddpg_agent.state_dim,
                    'action_dim': self.ddpg_agent.action_dim,
                    'action_high': np.array(self.ddpg_agent.action_high),
                    'action_low': np.array(self.ddpg_agent.action_low),
                    'config': self.ddpg_agent.config,
                }
            }
            
            with open(filepath, 'wb') as f:
                pickle.dump(serialized_data, f)
            
            print_success(f"Modelo DDPG guardado en: {filepath}")
        except Exception as e:
            print_error(f"Error al guardar modelo: {e}")
            print_warning("Guardando con pickle como alternativa.")
            
            # Guardar estado completo como respaldo
            with open(filepath, 'wb') as f:
                pickle.dump({
                    'cgm_weight': np.array(self.cgm_weight),
                    'other_weight': np.array(self.other_weight),
                    'actor_params': flax.core.frozen_dict.unfreeze(self.ddpg_agent.train_state.actor.params),
                    'critic_params': flax.core.frozen_dict.unfreeze(self.ddpg_agent.train_state.critic.params)
                }, f)
            
            print_success(f"Estado del modelo guardado como pickle en: {filepath}")

    def load(self, filepath: str) -> None:
        """
        Carga el modelo desde disco.
        
        Parámetros:
        -----------
        filepath : str
            Ruta desde donde cargar el modelo
        """
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)
        
        # Cargar pesos y estado
        self.cgm_weight = model_data['cgm_weight']
        self.other_weight = model_data['other_weight']
        if 'train_state' in model_data:
            self.ddpg_agent.train_state = model_data['train_state']
    
        print_success(f"Modelo DDPG cargado desde: {filepath}")
    
    def get_config(self) -> Dict:
        """
        Obtiene la configuración del modelo.
        
        Retorna:
        --------
        Dict
            Configuración del modelo
        """
        return {
            'cgm_shape': self.cgm_shape,
            'other_features_shape': self.other_features_shape,
            'state_dim': self.state_dim,
            'action_dim': self.ddpg_agent.action_dim
        }

def create_ddpg_model(cgm_shape: Tuple[int, ...], other_features_shape: Tuple[int, ...]) -> DRLModelWrapper:
    """
    Crea un modelo basado en DDPG (Deep Deterministic Policy Gradient) para predicción de dosis de insulina.
    
    Parámetros:
    -----------
    cgm_shape : Tuple[int, ...]
        Forma de los datos CGM (batch_size, time_steps, features)
    other_features_shape : Tuple[int, ...]
        Forma de otras características (batch_size, n_features)
        
    Retorna:
    --------
    DRLModelWrapper
        Wrapper de DDPG que implementa la interfaz compatible con el sistema de model creators
    """
    # Configurar el espacio de estados y acciones
    state_dim = 64  # Dimensión del espacio de estado latente
    action_dim = 1  # Una dimensión para la dosis continua
    
    # Límites de acción para dosis de insulina (0-15 unidades)
    action_high = jnp.array([15.0])  # Máximo 15 unidades de insulina
    action_low = jnp.array([0.0])    # Mínimo 0 unidades
    
    # Crear configuración para el agente DDPG
    config = {
        'actor_lr': DDPG_CONFIG['actor_lr'],
        'critic_lr': DDPG_CONFIG['critic_lr'],
        'gamma': DDPG_CONFIG['gamma'],
        'tau': DDPG_CONFIG['tau'],
        'batch_size': min(DDPG_CONFIG['batch_size'], 64),  # Adaptado para este problema
        'buffer_capacity': DDPG_CONFIG['buffer_capacity'],
        'noise_std': DDPG_CONFIG['noise_std'],
        'actor_hidden_units': DDPG_CONFIG['actor_hidden_units'],
        'critic_hidden_units': DDPG_CONFIG['critic_hidden_units'],
        'actor_activation': DDPG_CONFIG['actor_activation'],
        'critic_activation': DDPG_CONFIG['critic_activation'],
        'dropout_rate': DDPG_CONFIG['dropout_rate'],
        'epsilon': DDPG_CONFIG['epsilon'],
        'seed': CONST_DEFAULT_SEED
    }
    
    # Crear agente DDPG
    ddpg_agent = DDPG(
        state_dim=state_dim,
        action_dim=action_dim,
        action_high=action_high,
        action_low=action_low,
        config=config,
        seed=CONST_DEFAULT_SEED
    )
    
    # Crear wrapper DDPG
    ddpg_wrapper = DDPGWrapper(
        ddpg_agent=ddpg_agent,
        cgm_shape=cgm_shape,
        other_features_shape=other_features_shape
    )
    
    # Envolver en DRLModelWrapper para compatibilidad con el sistema
    return DRLModelWrapper(model_cls=ddpg_wrapper, framework="jax", algorithm="ddpg")

def model_creator() -> Callable[[Tuple[int, ...], Tuple[int, ...]], DRLModelWrapper]:
    """
    Retorna una función para crear un modelo DDPG compatible con la API del sistema.
    
    Retorna:
    --------
    Callable[[Tuple[int, ...], Tuple[int, ...]], DRLModelWrapper]
        Función para crear el modelo con las formas de entrada especificadas
    """
    return create_ddpg_model