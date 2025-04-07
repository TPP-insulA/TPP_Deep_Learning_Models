import os, sys
from types import SimpleNamespace
import jax
import jax.numpy as jnp
import numpy as np
import flax.linen as nn
from flax.training import train_state
import optax
from typing import Dict, List, Tuple, Any, Optional, Callable, Union, Sequence
from collections import deque
import random
from functools import partial
import matplotlib.pyplot as plt

PROJECT_ROOT = os.path.abspath(os.getcwd())
sys.path.append(PROJECT_ROOT) 

from config.models_config import DQN_CONFIG
from custom.drl_model_wrapper import DRLModelWrapper

# Constantes para uso repetido
CONST_RELU = "relu"
CONST_GELU = "gelu"
CONST_TANH = "tanh"
CONST_SELU = "selu"
CONST_SIGMOID = "sigmoid"
CONST_DROPOUT = "dropout"
CONST_PARAMS = "params"
CONST_TARGET = "target"
CONST_Q_VALUE = "q_value"
CONST_LOSS = "loss"

FIGURES_DIR = os.path.join(PROJECT_ROOT, "figures", "jax", "dqn")


class ReplayBuffer:
    """
    Buffer de experiencias para el algoritmo DQN.
    
    Almacena transiciones (state, action, reward, next_state, done)
    y permite muestrear lotes de manera aleatoria para el entrenamiento.
    
    Parámetros:
    -----------
    capacity : int, opcional
        Capacidad máxima del buffer (default: 10000)
    """
    def __init__(self, capacity: int = 10000) -> None:
        self.buffer = deque(maxlen=capacity)
    
    def add(self, state: np.ndarray, action: int, reward: float, 
           next_state: np.ndarray, done: float) -> None:
        """
        Añade una transición al buffer.
        
        Parámetros:
        -----------
        state : np.ndarray
            Estado actual
        action : int
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
            # Si no hay suficientes experiencias, muestrear con reemplazo
            rng = np.random.Generator(np.random.PCG64(42))
            indices = rng.integers(0, len(self.buffer), size=batch_size)
            batch = [self.buffer[i] for i in indices]
        else:
            # Uso del nuevo método de muestreo con Generator
            rng = np.random.Generator(np.random.PCG64(42))
            batch = rng.choice(self.buffer, batch_size, replace=False)
        
        states, actions, rewards, next_states, dones = zip(*batch)
        
        return (
            np.array(states, dtype=np.float32),
            np.array(actions, dtype=np.int32),
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


class PrioritizedReplayBuffer(ReplayBuffer):
    """
    Buffer de experiencias con muestreo prioritario.
    
    Prioriza experiencias basadas en el TD error para muestreo eficiente.
    
    Parámetros:
    -----------
    capacity : int
        Capacidad máxima del buffer
    """
    def __init__(self, capacity: int) -> None:
        super().__init__(capacity)
        self.priorities = np.zeros(capacity, dtype=np.float32) + 1e-5
        self.pos = 0
    
    def add(self, state: np.ndarray, action: int, reward: float, 
           next_state: np.ndarray, done: float) -> None:
        """
        Añade una transición al buffer con prioridad máxima.
        
        Parámetros:
        -----------
        state : np.ndarray
            Estado actual
        action : int
            Acción tomada
        reward : float
            Recompensa recibida
        next_state : np.ndarray
            Estado siguiente
        done : float
            Indicador de fin de episodio (1.0 si terminó, 0.0 si no)
        """
        max_priority = max(np.max(self.priorities), 1e-5)
        
        if len(self.buffer) < self.buffer.maxlen:
            super().add(state, action, reward, next_state, done)
        else:
            # Reemplazar en posición actual
            self.buffer[self.pos] = (state, action, reward, next_state, done)
        
        self.priorities[self.pos] = max_priority
        self.pos = (self.pos + 1) % self.buffer.maxlen
    
    def sample(self, batch_size: int, beta: float = 0.4) -> Tuple[np.ndarray, np.ndarray, 
                                                                np.ndarray, np.ndarray, 
                                                                np.ndarray, List[int], np.ndarray]:
        """
        Muestrea un lote basado en prioridades.
        
        Parámetros:
        -----------
        batch_size : int
            Tamaño del lote
        beta : float, opcional
            Factor para corrección de sesgo (0-1) (default: 0.4)
            
        Retorna:
        --------
        Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, List[int], np.ndarray]
            (states, actions, rewards, next_states, dones, índices, pesos de importancia)
        """
        if len(self.buffer) < batch_size:
            # Si no hay suficientes experiencias, usar todas con reemplazo
            idx = np.random.Generator(np.random.PCG64(42)).choice(
                len(self.buffer), batch_size, replace=True).tolist()
        else:
            # Muestreo basado en prioridades
            priorities = self.priorities[:len(self.buffer)]
            probabilities = priorities ** DQN_CONFIG['priority_alpha']
            probabilities /= np.sum(probabilities)
            
            # Muestreo según distribución de probabilidad
            rng = np.random.Generator(np.random.PCG64(42))
            idx = rng.choice(len(self.buffer), batch_size, p=probabilities, replace=False).tolist()
        
        # Extraer batch
        states, actions, rewards, next_states, dones = [], [], [], [], []
        for i in idx:
            s, a, r, ns, d = self.buffer[i]
            states.append(s)
            actions.append(a)
            rewards.append(r)
            next_states.append(ns)
            dones.append(d)
        
        # Calcular pesos de importancia (corrección del sesgo)
        weights = np.zeros(batch_size, dtype=np.float32)
        priorities = self.priorities[idx]
        # Evitar división por cero
        probabilities = priorities / np.sum(self.priorities[:len(self.buffer)])
        weights = (len(self.buffer) * probabilities) ** -beta
        weights /= np.max(weights)  # Normalizar
        
        return (
            np.array(states, dtype=np.float32),
            np.array(actions, dtype=np.int32),
            np.array(rewards, dtype=np.float32),
            np.array(next_states, dtype=np.float32),
            np.array(dones, dtype=np.float32),
            idx,
            np.array(weights, dtype=np.float32)
        )
    
    def update_priorities(self, indices: List[int], priorities: np.ndarray) -> None:
        """
        Actualiza las prioridades para los índices dados.
        
        Parámetros:
        -----------
        indices : List[int]
            Índices de las transiciones a actualizar
        priorities : np.ndarray
            Nuevas prioridades
        """
        for idx, priority in zip(indices, priorities):
            self.priorities[idx] = priority + 1e-5


class QNetwork(nn.Module):
    """
    Red Q para DQN que mapea estados a valores Q.
    
    Implementa una arquitectura flexible para estimación de valores Q-state-action.
    
    Parámetros:
    -----------
    action_dim : int
        Dimensión del espacio de acciones
    hidden_units : Sequence[int], opcional
        Unidades en capas ocultas (default: None)
    dueling : bool, opcional
        Si usar arquitectura dueling (default: False)
    activation : str, opcional
        Función de activación a utilizar (default: 'relu')
    dropout_rate : float, opcional
        Tasa de dropout para regularización (default: 0.0)
    """
    action_dim: int
    hidden_units: Optional[Sequence[int]] = None
    dueling: bool = False
    activation: str = CONST_RELU
    dropout_rate: float = 0.0
    
    def _get_activation_fn(self, activation_name):
        """
        Obtiene la función de activación según el nombre.
        
        Parámetros:
        -----------
        activation_name : str
            Nombre de la función de activación
            
        Retorna:
        --------
        Callable
            Función de activación correspondiente
        """
        activation_map = {
            CONST_RELU: nn.relu,
            CONST_GELU: nn.gelu,
            CONST_SELU: nn.selu,
            CONST_TANH: jnp.tanh,
            CONST_SIGMOID: jax.nn.sigmoid
        }
        return activation_map.get(activation_name, nn.relu)
    
    @nn.compact
    def __call__(self, x: jnp.ndarray, training: bool = False) -> jnp.ndarray:
        """
        Aplica la red Q a las entradas.
        
        Parámetros:
        -----------
        x : jnp.ndarray
            Tensor de entrada (estados)
        training : bool, opcional
            Indica si está en modo entrenamiento (default: False)
            
        Retorna:
        --------
        jnp.ndarray
            Valores Q para cada acción
        """
        # Default para capas ocultas si no se especifican
        hidden_units = self.hidden_units or [256, 256]
        
        # Obtener función de activación
        activation_fn = self._get_activation_fn(self.activation)
        
        # Procesar características de estado
        for i, units in enumerate(hidden_units):
            x = nn.Dense(units, name=f'feature_dense_{i}')(x)
            x = nn.LayerNorm(epsilon=DQN_CONFIG['epsilon'], name=f'feature_ln_{i}')(x)
            x = activation_fn(x)
            if self.dropout_rate > 0 and training:
                x = nn.Dropout(rate=self.dropout_rate, deterministic=not training)(x)
        
        # Para arquitectura Dueling DQN
        if self.dueling:
            # Ventaja: un valor por acción
            advantage = x
            for i, units in enumerate(hidden_units[-2:]):
                advantage = nn.Dense(units, name=f'advantage_dense_{i}')(advantage)
                advantage = activation_fn(advantage)
            advantage = nn.Dense(self.action_dim, name='advantage')(advantage)
            
            # Valor del estado: un valor único
            value = x
            for i, units in enumerate(hidden_units[-2:]):
                value = nn.Dense(units, name=f'value_dense_{i}')(value)
                value = activation_fn(value)
            value = nn.Dense(1, name='value')(value)
            
            # Combinar ventaja y valor (restando la media de ventajas)
            q_values = value + (advantage - advantage.mean(axis=-1, keepdims=True))
        else:
            # DQN clásica: predecir valor Q para cada acción
            q_values = nn.Dense(self.action_dim, name='q_values')(x)
            
        return q_values


class DQNTrainState(train_state.TrainState):
    """
    Estado de entrenamiento para DQN que extiende TrainState de Flax.
    
    Incluye el modelo target además del modelo principal.
    
    Atributos:
    ----------
    target_params : Any
        Parámetros del modelo target
    rng: jax.random.PRNGKey
        Clave para generación aleatoria
    """
    target_params: Any
    rng: jax.random.PRNGKey


class DQN:
    """
    Implementación del algoritmo Deep Q-Network (DQN) usando JAX.
    
    Incluye mecanismos de Experience Replay y Target Network para
    mejorar la estabilidad del aprendizaje.
    
    Parámetros:
    -----------
    state_dim : int
        Dimensión del espacio de estados
    action_dim : int
        Dimensión del espacio de acciones
    config : Optional[Dict[str, Any]], opcional
        Configuración personalizada (default: None)
    hidden_units : Optional[List[int]], opcional
        Unidades en capas ocultas (default: None)
    seed : int, opcional
        Semilla para los generadores de números aleatorios (default: 42)
    """
    def __init__(
        self, 
        state_dim: int, 
        action_dim: int,
        config: Optional[Dict[str, Any]] = None,
        hidden_units: Optional[List[int]] = None,
        seed: int = 42
    ) -> None:
        # Use provided config or default
        self.config = config or DQN_CONFIG
        
        # Extract configuration parameters
        learning_rate = self.config.get('learning_rate', 0.001)
        self.gamma = self.config.get('gamma', 0.99)
        self.epsilon_start = self.config.get('epsilon_start', 1.0)
        self.epsilon_end = self.config.get('epsilon_end', 0.01)
        self.epsilon_decay = self.config.get('epsilon_decay', 0.995)
        buffer_capacity = self.config.get('buffer_capacity', 10000)
        self.batch_size = self.config.get('batch_size', 64)
        self.target_update_freq = self.config.get('target_update_freq', 100)
        dueling = self.config.get('dueling', False)
        self.double = self.config.get('double', False)
        self.prioritized = self.config.get('prioritized', False)
        dropout_rate = self.config.get('dropout_rate', 0.0)
        
        # Parámetros del entorno y del modelo
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.epsilon = self.epsilon_start
        
        # Cantidad de actualizaciones realizadas
        self.update_counter = 0
        
        # Configurar semillas para reproducibilidad
        self.seed = seed
        self.rng = jax.random.PRNGKey(seed)
        np.random.seed(seed)
        random.seed(seed)
        
        # Crear directorio para figuras si no existe
        os.makedirs(FIGURES_DIR, exist_ok=True)
        
        # Dividir la clave para inicialización y uso posterior
        self.rng, init_rng = jax.random.split(self.rng)
        
        # Valores predeterminados para capas ocultas
        if hidden_units is None:
            self.hidden_units = DQN_CONFIG['hidden_units']
        else:
            self.hidden_units = hidden_units
        
        # Crear modelo Q
        self.q_network = QNetwork(
            action_dim=action_dim,
            hidden_units=self.hidden_units,
            dueling=dueling,
            activation=DQN_CONFIG['activation'],
            dropout_rate=dropout_rate
        )
        
        # Inicializar parámetros con una muestra de estado
        dummy_state = jnp.zeros((1, state_dim))
        params = self.q_network.init(init_rng, dummy_state)
        
        # Crear optimizador
        tx = optax.adam(learning_rate=learning_rate)
        
        # Crear estado de entrenamiento
        self.state = DQNTrainState(
            step=0,
            apply_fn=self.q_network.apply,
            params=params,
            target_params=params,  # Inicializar target = modelo principal
            tx=tx,
            rng=init_rng
        )
        
        # Buffer de experiencias
        if self.prioritized:
            # Con prioridad de muestreo basada en TD-error
            self.replay_buffer = PrioritizedReplayBuffer(buffer_capacity)
            self.alpha = DQN_CONFIG['priority_alpha']
            self.beta = DQN_CONFIG['priority_beta']
            self.beta_increment = DQN_CONFIG['priority_beta_increment']
        else:
            # Buffer uniforme clásico
            self.replay_buffer = ReplayBuffer(buffer_capacity)
        
        # Métricas acumuladas
        self.loss_sum = 0.0
        self.q_value_sum = 0.0
        self.updates_count = 0
        
        # Precompilar funciones para entrenamiento
        self._compile_jitted_functions()
        
    def _compile_jitted_functions(self) -> None:
        """
        Compila versiones JIT de las funciones principales para acelerar la ejecución.
        """
        self.update_target_jit = jax.jit(self._update_target)
        self.train_step_jit = jax.jit(self._train_step)
        
    def update_target_network(self) -> None:
        """
        Actualiza los pesos del modelo target con los del modelo principal.
        """
        self.state = self.update_target_jit(self.state)
    
    def _update_target(self, state: DQNTrainState) -> DQNTrainState:
        """
        Actualiza los parámetros target.
        
        Parámetros:
        -----------
        state : DQNTrainState
            Estado actual del entrenamiento
            
        Retorna:
        --------
        DQNTrainState
            Nuevo estado con parámetros target actualizados
        """
        return state.replace(target_params=state.params)
    
    def _train_step(self, state: DQNTrainState, 
                  batch: Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray],
                  importance_weights: Optional[jnp.ndarray] = None) -> Tuple[DQNTrainState, jnp.ndarray, jnp.ndarray]:
        """
        Realiza un paso de entrenamiento para actualizar la red Q.
        
        Parámetros:
        -----------
        state : DQNTrainState
            Estado actual del entrenamiento
        batch : Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]
            (estados, acciones, recompensas, siguientes_estados, terminados)
        importance_weights : Optional[jnp.ndarray], opcional
            Pesos de importancia para muestreo prioritario (default: None)
            
        Retorna:
        --------
        Tuple[DQNTrainState, jnp.ndarray, jnp.ndarray]
            (nuevo_estado, pérdida, td_errors)
        """
        states, actions, rewards, next_states, dones = batch
        
        # Dividir rng para posibles necesidades estocásticas
        new_rng, dropout_rng = jax.random.split(state.rng)
        
        # Función de pérdida y su gradiente
        def loss_fn(params):
            # Q-values para los estados actuales
            q_values = state.apply_fn(params, states, rngs={'dropout': dropout_rng})
            q_values_selected = q_values[jnp.arange(q_values.shape[0]), actions]
            
            # Q-values objetivos para los siguientes estados
            if self.double:
                # Double DQN: seleccionar acción con red primaria
                next_q_values_online = state.apply_fn(params, next_states)
                next_actions = jnp.argmax(next_q_values_online, axis=1)
                next_q_values_target = state.apply_fn(state.target_params, next_states)
                next_q_values = next_q_values_target[jnp.arange(next_q_values_target.shape[0]), next_actions]
            else:
                # DQN estándar: target Q-network para seleccionar y evaluar
                next_q_values = state.apply_fn(state.target_params, next_states)
                next_q_values = jnp.max(next_q_values, axis=1)
                
            # Calcular targets usando ecuación de Bellman
            targets = rewards + (1.0 - dones) * self.gamma * next_q_values
            
            # Calcular TD-error
            td_errors = targets - q_values_selected
            
            # Aplicar pesos de importancia para PER si es necesario
            if importance_weights is not None:
                loss = jnp.mean(importance_weights * jnp.square(td_errors))
            else:
                loss = jnp.mean(optax.huber_loss(q_values_selected, targets))
            
            metrics = {
                CONST_LOSS: loss,
                CONST_Q_VALUE: jnp.mean(q_values_selected),
                'td_errors': td_errors
            }
            
            return loss, metrics
        
        # Calcular gradientes y actualizar parámetros
        grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
        (loss, metrics), grads = grad_fn(state.params)
        new_state = state.apply_gradients(grads=grads, rng=new_rng)
        
        return new_state, loss, metrics['td_errors']
    
    def get_action(self, state: np.ndarray, epsilon: float = 0.0) -> int:
        """
        Obtiene una acción usando la política epsilon-greedy.
        
        Parámetros:
        -----------
        state : np.ndarray
            Estado actual del entorno
        epsilon : float, opcional
            Probabilidad de exploración (0-1) (default: 0.0)
            
        Retorna:
        --------
        int
            Acción seleccionada según la política
        """
        # Dividir clave para exploración
        self.rng, explore_rng = jax.random.split(self.rng)
        
        # Exploración
        if jax.random.uniform(explore_rng) < epsilon:
            # Explorar: acción aleatoria
            self.rng, action_rng = jax.random.split(self.rng)
            return int(jax.random.randint(action_rng, (), 0, self.action_dim))
        else:
            # Explotar: mejor acción según la red
            state_tensor = jnp.array([state], dtype=jnp.float32)
            q_values = self.q_network.apply(self.state.params, state_tensor)
            return int(jnp.argmax(q_values[0]))
    
    def _sample_batch(self) -> Tuple:
        """
        Muestrea un lote del buffer de experiencias.
        
        Retorna:
        --------
        Tuple
            Datos muestreados, según el tipo de buffer
        """
        if self.prioritized:
            (states, actions, rewards, next_states, dones, 
             indices, importance_weights) = self.replay_buffer.sample(
                 self.batch_size, self.beta)
            self.beta = min(1.0, self.beta + self.beta_increment)
            
            # Convertir a JAX arrays
            batch = (
                jnp.array(states, dtype=jnp.float32),
                jnp.array(actions, dtype=jnp.int32),
                jnp.array(rewards, dtype=jnp.float32),
                jnp.array(next_states, dtype=jnp.float32),
                jnp.array(dones, dtype=jnp.float32)
            )
            importance_weights_array = jnp.array(importance_weights, dtype=jnp.float32)
            return batch, indices, importance_weights_array
        else:
            states, actions, rewards, next_states, dones = self.replay_buffer.sample(
                self.batch_size)
            
            # Convertir a JAX arrays
            batch = (
                jnp.array(states, dtype=jnp.float32),
                jnp.array(actions, dtype=jnp.int32),
                jnp.array(rewards, dtype=jnp.float32),
                jnp.array(next_states, dtype=jnp.float32),
                jnp.array(dones, dtype=jnp.float32)
            )
            return batch, None, None
    
    def _update_model(self, episode_loss: List[float], update_every: int, update_after: int) -> None:
        """
        Actualiza el modelo si hay suficientes datos y es momento de hacerlo.
        
        Parámetros:
        -----------
        episode_loss : List[float]
            Lista para almacenar pérdidas del episodio
        update_every : int
            Frecuencia de actualización (cada cuántos pasos)
        update_after : int
            Número de pasos antes de empezar a actualizar
        """
        # Actualizar sólo si el buffer tiene suficientes datos y es momento de actualizar
        if len(self.replay_buffer) < self.batch_size or self.update_counter < update_after:
            return
            
        if self.update_counter % update_every != 0:
            self.update_counter += 1
            return
            
        # Muestrear batch
        batch, indices, importance_weights = self._sample_batch()
        
        # Actualizar modelo
        self.state, loss, td_errors = self.train_step_jit(self.state, batch, importance_weights)
        episode_loss.append(float(loss))
        
        # Actualizar métricas acumuladas
        self.loss_sum += float(loss)
        # Calcular q_values del batch
        states, actions = batch[0], batch[1]
        q_values = self.q_network.apply(self.state.params, states)
        q_values_selected = q_values[jnp.arange(q_values.shape[0]), actions]
        self.q_value_sum += float(jnp.mean(q_values_selected))
        self.updates_count += 1
        
        # Actualizar prioridades si es PER
        if self.prioritized and indices is not None:
            priorities = np.abs(np.array(td_errors)) + 1e-6
            self.replay_buffer.update_priorities(indices, priorities)
                
        # Actualizar target network periódicamente
        if self.update_counter % self.target_update_freq == 0 and self.update_counter > 0:
            self.update_target_network()
        
        self.update_counter += 1
    
    def _run_episode(self, env: Any, max_steps: int, render: bool, 
                   update_every: int, update_after: int) -> Tuple[float, List[float]]:
        """
        Ejecuta un episodio completo de entrenamiento.
        
        Parámetros:
        -----------
        env : Any
            Entorno donde ejecutar el episodio
        max_steps : int
            Pasos máximos por episodio
        render : bool
            Si se debe renderizar el entorno
        update_every : int
            Frecuencia de actualización del modelo
        update_after : int
            Pasos antes de empezar a actualizar la red
            
        Retorna:
        --------
        Tuple[float, List[float]]
            (episode_reward, episode_loss)
        """
        state, _ = env.reset()
        state = np.array(state, dtype=np.float32)
        episode_reward = 0.0
        episode_loss = []
        
        for _ in range(max_steps):
            # Seleccionar acción
            action = self.get_action(state, self.epsilon)
            
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
            
            # Actualizar modelo
            self._update_model(episode_loss, update_every, update_after)
            
            if done:
                break
                
        return episode_reward, episode_loss
    
    def _update_history(self, history: Dict, episode_reward: float, episode_loss: List[float], 
                      episode_reward_history: List[float], log_interval: int) -> List[float]:
        """
        Actualiza el historial de entrenamiento con los resultados del episodio.
        
        Parámetros:
        -----------
        history : Dict
            Diccionario de historial a actualizar
        episode_reward : float
            Recompensa obtenida en el episodio
        episode_loss : List[float]
            Lista de pérdidas durante el episodio
        episode_reward_history : List[float]
            Historial de recompensas para cálculos de promedio
        log_interval : int
            Intervalo para limitar el tamaño del historial
            
        Retorna:
        --------
        List[float]
            Historial de recompensas actualizado
        """
        # Guardar recompensa
        history['episode_rewards'].append(float(episode_reward))
        
        # Guardar epsilon
        history['epsilons'].append(float(self.epsilon))
        
        # Guardar pérdida media si hay actualizaciones
        if episode_loss:
            history['losses'].append(float(np.mean(episode_loss)))
        else:
            history['losses'].append(0.0)
            
        # Calcular y guardar promedio del valor Q
        if self.updates_count > 0:
            avg_q_value = self.q_value_sum / self.updates_count
            self.q_value_sum = 0.0
            self.updates_count = 0
        else:
            avg_q_value = 0.0
            
        history['avg_q_values'].append(avg_q_value)
        
        # Actualizar epsilon (decaimiento)
        self.epsilon = max(
            self.epsilon_end, 
            self.epsilon * self.epsilon_decay
        )
        
        # Guardar últimas recompensas para promedio
        episode_reward_history.append(episode_reward)
        if len(episode_reward_history) > log_interval:
            episode_reward_history.pop(0)
            
        return episode_reward_history
    
    def train(self, env: Any, episodes: int = 1000, max_steps: int = 1000, 
             update_after: int = 1000, update_every: int = 4, 
             render: bool = False, log_interval: int = 10) -> Dict:
        """
        Entrena el agente DQN en un entorno dado.
        
        Parámetros:
        -----------
        env : Any
            Entorno de Gym o compatible
        episodes : int, opcional
            Número máximo de episodios (default: 1000)
        max_steps : int, opcional
            Pasos máximos por episodio (default: 1000)
        update_after : int, opcional
            Pasos antes de empezar a actualizar la red (default: 1000)
        update_every : int, opcional
            Frecuencia de actualización (default: 4)
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
            'losses': [],
            'epsilons': [],
            'avg_q_values': []
        }
        
        # Variables para seguimiento de progreso
        best_reward = -float('inf')
        episode_reward_history = []
        
        for episode in range(episodes):
            # Ejecutar un episodio
            episode_reward, episode_loss = self._run_episode(
                env, max_steps, render, update_every, update_after)
            
            # Actualizar historial
            episode_reward_history = self._update_history(
                history, episode_reward, episode_loss, episode_reward_history, log_interval)
            
            # Mostrar progreso periódicamente
            if (episode + 1) % log_interval == 0:
                avg_reward = np.mean(episode_reward_history)
                print(f"Episodio {episode+1}/{episodes} - Recompensa Promedio: {avg_reward:.2f}, "
                      f"Epsilon: {self.epsilon:.4f}, Pérdida: {history['losses'][-1]:.4f}")
                
                # Guardar mejor modelo
                if avg_reward > best_reward:
                    best_reward = avg_reward
                    print(f"Nuevo mejor modelo con recompensa: {best_reward:.2f}")
        
        return history
    
    def evaluate(self, env: Any, episodes: int = 10, render: bool = False) -> float:
        """
        Evalúa el agente DQN entrenado en un entorno.
        
        Parámetros:
        -----------
        env : Any
            Entorno de Gym o compatible
        episodes : int, opcional
            Número de episodios para evaluar (default: 10)
        render : bool, opcional
            Mostrar entorno gráficamente (default: False)
            
        Retorna:
        --------
        float
            Recompensa promedio durante la evaluación
        """
        rewards = []
        
        for episode in range(episodes):
            state, _ = env.reset()
            state = np.array(state, dtype=np.float32)
            episode_reward = 0
            done = False
            step = 0
            
            while not done:
                # Seleccionar acción determinística (sin exploración)
                action = self.get_action(state)
                
                # Ejecutar acción
                next_state, reward, done, _, _ = env.step(action)
                next_state = np.array(next_state, dtype=np.float32)
                
                # Renderizar si es necesario
                if render:
                    env.render()
                
                # Actualizar estado y recompensa
                state = next_state
                episode_reward += reward
                step += 1
            
            rewards.append(episode_reward)
            print(f"Episodio de evaluación {episode+1}/{episodes} - "
                  f"Recompensa: {episode_reward:.2f}, Pasos: {step}")
        
        avg_reward = np.mean(rewards)
        print(f"Recompensa promedio de evaluación: {avg_reward:.2f}")
        
        return avg_reward
    
    def save_model(self, filepath: str) -> None:
        """
        Guarda el modelo en disco.
        
        Parámetros:
        -----------
        filepath : str
            Ruta donde guardar el modelo
        """
        from flax.training import checkpoints
        
        # Crear directorio si no existe
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        # Guardar parámetros del modelo
        checkpoints.save_checkpoint(
            filepath, 
            self.state,
            step=int(self.update_counter),
            overwrite=True
        )
    
    def load_model(self, filepath: str) -> None:
        """
        Carga el modelo desde disco.
        
        Parámetros:
        -----------
        filepath : str
            Ruta desde donde cargar el modelo
        """
        from flax.training import checkpoints
        
        # Cargar parámetros
        self.state = checkpoints.restore_checkpoint(filepath, self.state)
    
    def visualize_training(self, history: Dict, window_size: int = 10) -> None:
        """
        Visualiza el historial de entrenamiento con gráficos.
        
        Parámetros:
        -----------
        history : Dict
            Historial de entrenamiento
        window_size : int, opcional
            Tamaño de ventana para suavizado (default: 10)
        """
        import matplotlib.pyplot as plt
        
        def smooth(data, window_size):
            kernel = np.ones(window_size) / window_size
            return np.convolve(data, kernel, mode='valid')
        
        # Crear figura
        _, axs = plt.subplots(2, 2, figsize=(14, 10))
        
        # Recompensas
        rewards = history['episode_rewards']
        axs[0, 0].plot(rewards, alpha=0.3, color='blue', label='Original')
        if len(rewards) > window_size:
            axs[0, 0].plot(range(window_size-1, len(rewards)), smooth(rewards, window_size), 
                         color='blue', label=f'Suavizado (ventana={window_size})')
        axs[0, 0].set_title('Recompensas por Episodio')
        axs[0, 0].set_xlabel('Episodio')
        axs[0, 0].set_ylabel('Recompensa')
        axs[0, 0].legend()
        axs[0, 0].grid(alpha=0.3)
        
        # Epsilon
        epsilons = history['epsilons']
        axs[0, 1].plot(epsilons, color='green')
        axs[0, 1].set_title('Valor Epsilon')
        axs[0, 1].set_xlabel('Episodio')
        axs[0, 1].set_ylabel('Epsilon')
        axs[0, 1].grid(alpha=0.3)
        
        # Pérdida
        losses = history['losses']
        axs[1, 0].plot(losses, alpha=0.3, color='red', label='Original')
        if len(losses) > window_size:
            axs[1, 0].plot(range(window_size-1, len(losses)), smooth(losses, window_size), 
                         color='red', label=f'Suavizado (ventana={window_size})')
        axs[1, 0].set_title('Pérdida')
        axs[1, 0].set_xlabel('Episodio')
        axs[1, 0].set_ylabel('Pérdida')
        axs[1, 0].legend()
        axs[1, 0].grid(alpha=0.3)
        
        # Valores Q promedio
        q_values = history['avg_q_values']
        axs[1, 1].plot(q_values, alpha=0.3, color='purple', label='Original')
        if len(q_values) > window_size:
            axs[1, 1].plot(range(window_size-1, len(q_values)), smooth(q_values, window_size), 
                         color='purple', label=f'Suavizado (ventana={window_size})')
        axs[1, 1].set_title('Valores Q Promedio')
        axs[1, 1].set_xlabel('Episodio')
        axs[1, 1].set_ylabel('Valor Q')
        axs[1, 1].legend()
        axs[1, 1].grid(alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(FIGURES_DIR, 'training_results.png'))
        plt.show()


class DQNWrapper:
    """
    Wrapper para hacer que el agente DQN sea compatible con la interfaz de modelos de aprendizaje profundo.
    """
    
    def __init__(
        self, 
        dqn_agent: DQN, 
        cgm_shape: Tuple[int, ...], 
        other_features_shape: Tuple[int, ...],
    ) -> None:
        """
        Inicializa el wrapper DQN.
        
        Parámetros:
        -----------
        dqn_agent : DQN
            Agente DQN a encapsular
        cgm_shape : Tuple[int, ...]
            Forma de los datos CGM
        other_features_shape : Tuple[int, ...]
            Forma de otras características
        """
        self.dqn_agent = dqn_agent
        self.cgm_shape = cgm_shape
        self.other_features_shape = other_features_shape
        
        # Inicializar generador de números aleatorios
        self.key = jax.random.PRNGKey(42)
        
        # Configurar funciones de codificación para entradas
        self._setup_encoders()
        
        # Historial de entrenamiento
        self.history = {
            'loss': [],
            'val_loss': [],
            'losses': [],
            'episode_rewards': []
        }
    
    def _setup_encoders(self) -> None:
        """
        Configura las funciones de codificación para procesar las entradas.
        """
        # Calcular dimensiones de entrada aplanadas
        cgm_dim = np.prod(self.cgm_shape[1:])
        other_dim = np.prod(self.other_features_shape[1:])
        
        # Inicializar matrices de transformación
        self.key, key_cgm, key_other = jax.random.split(self.key, 3)
        
        # Crear matrices de proyección para la codificación de entradas
        self.cgm_weight = jax.random.normal(key_cgm, (cgm_dim, self.dqn_agent.state_dim // 2))
        self.other_weight = jax.random.normal(key_other, (other_dim, self.dqn_agent.state_dim // 2))
        
        # JIT-compilar transformaciones para mayor rendimiento
        self.encode_cgm = jax.jit(self._create_encoder_fn(self.cgm_weight))
        self.encode_other = jax.jit(self._create_encoder_fn(self.other_weight))
    
    def _create_encoder_fn(self, weights: jnp.ndarray) -> Callable:
        """
        Crea una función de codificación para transformar entradas.
        
        Parámetros:
        -----------
        weights : jnp.ndarray
            Pesos de la transformación
            
        Retorna:
        --------
        Callable
            Función de codificación
        """
        def encode_fn(x: jnp.ndarray) -> jnp.ndarray:
            # Aplanar entrada
            flat_x = x.reshape(x.shape[0], -1)
            # Transformar a espacio de estados
            encoded = jnp.tanh(jnp.matmul(flat_x, weights))
            return encoded
        return encode_fn
    
    def __call__(self, cgm_input: jnp.ndarray, other_input: jnp.ndarray, training: bool = False) -> jnp.ndarray:
        """
        Realiza una predicción usando el modelo DQN.
        
        Parámetros:
        -----------
        cgm_input : jnp.ndarray
            Datos CGM de entrada
        other_input : jnp.ndarray
            Otras características de entrada
        training : bool, opcional
            Indica si está en modo entrenamiento (default: False)
            
        Retorna:
        --------
        jnp.ndarray
            Predicciones de dosis
        """
        return self.predict([cgm_input, other_input])
    
    def predict(self, inputs: List[jnp.ndarray]) -> jnp.ndarray:
        """
        Realiza predicciones con el modelo.
        
        Parámetros:
        -----------
        inputs : List[jnp.ndarray]
            Lista con [cgm_input, other_input]
            
        Retorna:
        --------
        jnp.ndarray
            Predicciones de dosis
        """
        cgm_input, other_input = inputs
        
        # Codificar entradas a espacio de estado
        cgm_encoded = self.encode_cgm(jnp.array(cgm_input))
        other_encoded = self.encode_other(jnp.array(other_input))
        
        # Concatenar codificaciones
        states = jnp.concatenate([cgm_encoded, other_encoded], axis=1)
        
        # Convertir a dosis usando política DQN (sin exploración)
        batch_size = states.shape[0]
        actions = np.zeros((batch_size, 1))
        
        # Para cada muestra en el batch, obtener acción determinística
        for i in range(batch_size):
            state = np.array(states[i])
            # Usar DQN para obtener acción sin exploración
            action = self.dqn_agent.get_action(state, epsilon=0.0)
            # Convertir índice discreto a valor continuo de dosis (0-15 unidades)
            action_value = action / (self.dqn_agent.action_dim - 1) * 15.0
            actions[i, 0] = action_value
            
        return actions
    
    def fit(
        self, 
        x: List[jnp.ndarray], 
        y: jnp.ndarray, 
        validation_data: Optional[Tuple] = None, 
        epochs: int = 1,
        batch_size: int = 32,
        callbacks: List = None,
        verbose: int = 0
    ) -> Dict:
        """
        Entrena el modelo DQN con los datos proporcionados.
        
        Parámetros:
        -----------
        x : List[jnp.ndarray]
            Lista con [cgm_input, other_input]
        y : jnp.ndarray
            Valores objetivo de dosis
        validation_data : Optional[Tuple], opcional
            Datos de validación como ([x_cgm_val, x_other_val], y_val) (default: None)
        epochs : int, opcional
            Número de episodios de entrenamiento (default: 1)
        batch_size : int, opcional
            Tamaño del lote (default: 32)
        callbacks : List, opcional
            Lista de callbacks (default: None)
        verbose : int, opcional
            Nivel de verbosidad (default: 0)
            
        Retorna:
        --------
        Dict
            Historia del entrenamiento
        """
        if verbose > 0:
            print("Entrenando modelo DQN...")
            
        # Crear entorno simulado para RL a partir de los datos
        env = self._create_training_environment(x[0], x[1], y)
        
        # Entrenar el agente DQN
        training_history = self.dqn_agent.train(
            env=env,
            episodes=epochs,
            max_steps=batch_size,
            update_after=min(1000, batch_size // 2),
            update_every=1,
            render=False,
            log_interval=max(1, epochs // 10) if verbose > 0 else epochs + 1
        )
        
        # Actualizar historial con métricas del entrenamiento
        self.history['losses'].extend(training_history.get('losses', []))
        self.history['episode_rewards'].extend(training_history.get('episode_rewards', []))
        
        # Calcular pérdida en datos de entrenamiento
        train_preds = self.predict(x)
        train_loss = float(np.mean((train_preds.flatten() - y) ** 2))
        self.history['loss'].append(train_loss)
        
        # Evaluar en datos de validación si se proporcionan
        if validation_data:
            val_x, val_y = validation_data
            val_preds = self.predict(val_x)
            val_loss = float(np.mean((val_preds.flatten() - val_y) ** 2))
            self.history['val_loss'].append(val_loss)
        
        if verbose > 0:
            print(f"Entrenamiento completado. Pérdida final: {train_loss:.4f}")
            if validation_data:
                print(f"Pérdida de validación: {val_loss:.4f}")
        
        return self.history
    
    def _create_training_environment(
        self, 
        cgm_data: jnp.ndarray, 
        other_features: jnp.ndarray, 
        targets: jnp.ndarray
    ) -> Any:
        """
        Crea un entorno de entrenamiento personalizado basado en los datos.
        
        Parámetros:
        -----------
        cgm_data : jnp.ndarray
            Datos CGM
        other_features : jnp.ndarray
            Otras características
        targets : jnp.ndarray
            Valores objetivo de dosis
            
        Retorna:
        --------
        Any
            Entorno de entrenamiento
        """
        class InsulinDosingEnv:
            """Entorno personalizado para problema de dosificación de insulina."""
            
            def __init__(self, cgm, features, targets, model_wrapper):
                self.cgm = np.array(cgm)
                self.features = np.array(features)
                self.targets = np.array(targets)
                self.model = model_wrapper
                self.rng = np.random.Generator(np.random.PCG64(42))
                self.current_idx = 0
                self.max_idx = len(targets) - 1
                
                # Para compatibilidad con algoritmos RL
                self.observation_space = SimpleNamespace(
                    shape=(model_wrapper.dqn_agent.state_dim,),
                    low=-np.ones(model_wrapper.dqn_agent.state_dim) * 10,
                    high=np.ones(model_wrapper.dqn_agent.state_dim) * 10
                )
                
                self.action_space = SimpleNamespace(
                    shape=(1,),
                    n=model_wrapper.dqn_agent.action_dim,
                    sample=self._sample_action
                )
            
            def _sample_action(self):
                """Muestrea una acción aleatoria del espacio discreto."""
                return self.rng.integers(0, self.action_space.n)
                
            def reset(self):
                """Reinicia el entorno eligiendo un ejemplo aleatorio."""
                self.current_idx = self.rng.integers(0, self.max_idx)
                state = self._get_state()
                return state, {}
            
            def step(self, action):
                """Ejecuta un paso con la acción dada."""
                # Convertir acción discreta a dosis continua
                dose = action / (self.model.dqn_agent.action_dim - 1) * 15.0
                
                # Calcular recompensa como negativo del error absoluto
                target = self.targets[self.current_idx]
                reward = -abs(dose - target)
                
                # Avanzar al siguiente ejemplo
                self.current_idx = (self.current_idx + 1) % self.max_idx
                
                # Obtener próximo estado
                next_state = self._get_state()
                
                # Episodio siempre termina después de un paso
                done = True
                truncated = False
                
                return next_state, reward, done, truncated, {}
            
            def _get_state(self):
                """Obtiene el estado codificado para el ejemplo actual."""
                # Obtener datos actuales
                cgm_batch = self.cgm[self.current_idx:self.current_idx+1]
                features_batch = self.features[self.current_idx:self.current_idx+1]
                
                # Codificar a espacio de estado
                cgm_encoded = self.model.encode_cgm(jnp.array(cgm_batch))
                other_encoded = self.model.encode_other(jnp.array(features_batch))
                
                # Combinar características
                state = np.concatenate([cgm_encoded[0], other_encoded[0]])
                
                return state
        
        # Crear y devolver el entorno
        return InsulinDosingEnv(cgm_data, other_features, targets, self)
    
    def save(self, filepath: str) -> None:
        """
        Guarda el modelo en disco.
        
        Parámetros:
        -----------
        filepath : str
            Ruta donde guardar el modelo
        """
        self.dqn_agent.save_model(filepath)
        
        print(f"Modelo guardado en {filepath}")
    
    def load(self, filepath: str) -> None:
        """
        Carga el modelo desde disco.
        
        Parámetros:
        -----------
        filepath : str
            Ruta desde donde cargar el modelo
        """
        self.dqn_agent.load_model(filepath)
        
        print(f"Modelo cargado desde {filepath}")
    
    def get_config(self) -> Dict:
        """
        Obtiene la configuración del modelo.
        
        Retorna:
        --------
        Dict
            Configuración del modelo
        """
        return {
            "cgm_shape": self.cgm_shape,
            "other_features_shape": self.other_features_shape,
            "state_dim": self.dqn_agent.state_dim,
            "action_dim": self.dqn_agent.action_dim,
            "hidden_units": self.dqn_agent.hidden_units
        }


def create_dqn_model(cgm_shape: Tuple[int, ...], other_features_shape: Tuple[int, ...]) -> DRLModelWrapper:
    """
    Crea un modelo basado en DQN (Deep Q-Network) para predicción de dosis de insulina.
    
    Parámetros:
    -----------
    cgm_shape : Tuple[int, ...]
        Forma de los datos CGM (batch_size, time_steps, features)
    other_features_shape : Tuple[int, ...]
        Forma de otras características (batch_size, n_features)
        
    Retorna:
    --------
    DRLModelWrapper
        Wrapper del modelo DQN compatible con la interfaz del sistema
    """
    # Configurar el espacio de estados y acciones
    state_dim = 64  # Dimensión del espacio de estado latente
    action_dim = 20  # 20 niveles discretos para dosis (0 a 15 unidades)
    
    # Crear configuración para el agente DQN
    config = {
        'learning_rate': DQN_CONFIG['learning_rate'],
        'gamma': DQN_CONFIG['gamma'],
        'epsilon_start': DQN_CONFIG['epsilon_start'],
        'epsilon_end': DQN_CONFIG['epsilon_end'],
        'epsilon_decay': DQN_CONFIG['epsilon_decay'],
        'buffer_capacity': DQN_CONFIG['buffer_capacity'],
        'batch_size': DQN_CONFIG['batch_size'],
        'target_update_freq': DQN_CONFIG['target_update_freq'],
        'dueling': DQN_CONFIG['dueling'],
        'double': DQN_CONFIG['double'],
        'prioritized': DQN_CONFIG['prioritized'],
        'hidden_units': DQN_CONFIG['hidden_units'],
        'dropout_rate': DQN_CONFIG['dropout_rate'],
        'activation': DQN_CONFIG['activation'],
        'priority_alpha': DQN_CONFIG['priority_alpha'],
        'priority_beta': DQN_CONFIG['priority_beta'],
        'priority_beta_increment': DQN_CONFIG['priority_beta_increment'],
        'epsilon': DQN_CONFIG['epsilon']
    }
    
    # Crear agente DQN
    dqn_agent = DQN(
        state_dim=state_dim,
        action_dim=action_dim,
        config=config,
        hidden_units=DQN_CONFIG['hidden_units'],
        seed=42
    )
    
    # Crear y devolver wrapper
    return DRLModelWrapper(
        dqn_agent=dqn_agent,
        cgm_shape=cgm_shape,
        other_features_shape=other_features_shape
    )


def model_creator() -> Callable[[Tuple[int, ...], Tuple[int, ...]], DRLModelWrapper]:
    """
    Retorna una función para crear un modelo DQN compatible con la API del sistema.
    
    Retorna:
    --------
    Callable[[Tuple[int, ...], Tuple[int, ...]], DRLModelWrapper]
        Función para crear el modelo con las formas de entrada especificadas
    """
    return create_dqn_model