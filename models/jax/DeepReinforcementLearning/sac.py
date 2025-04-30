import os, sys
import jax
import jax.numpy as jnp
import numpy as np
import flax
import flax.linen as nn
import optax
import random
from flax.training import train_state
from typing import Dict, List, Tuple, Any, Optional, Union, Callable, Sequence
from functools import partial
from collections import deque
import matplotlib.pyplot as plt
from types import SimpleNamespace
import pickle

PROJECT_ROOT = os.path.abspath(os.getcwd())
sys.path.append(PROJECT_ROOT) 

from constants.constants import CONST_DEFAULT_SEED
from config.models_config import SAC_CONFIG
from custom.drl_model_wrapper import DRLModelWrapper

# Constantes para uso repetido
CONST_RELU = "relu"
CONST_TANH = "tanh"
CONST_GELU = "gelu"
CONST_LEAKY_RELU = "leaky_relu"
CONST_SIGMOID = "sigmoid"
CONST_DROPOUT = "dropout"
CONST_EPSILON = "epsilon"
CONST_PARAMS = "params"
CONST_ACTOR_PREFIX = "actor"
CONST_CRITIC_PREFIX = "critic"
CONST_TARGET_PREFIX = "target"

FIGURES_DIR = os.path.join(PROJECT_ROOT, "figures", "jax", "sac")


class ReplayBuffer:
    """
    Buffer de experiencias para el algoritmo SAC.
    
    Almacena transiciones (state, action, reward, next_state, done)
    y permite muestrear lotes de manera aleatoria para el entrenamiento.
    
    Parámetros:
    -----------
    capacity : int, opcional
        Capacidad máxima del buffer (default: 100000)
    """
    def __init__(self, capacity: int = 100000) -> None:
        self.buffer = deque(maxlen=capacity)
    
    def add(self, state: np.ndarray, action: np.ndarray, 
            reward: float, next_state: np.ndarray, done: float) -> None:
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
            (estados, acciones, recompensas, siguientes_estados, terminados)
        """
        if len(self.buffer) < batch_size:
            # Si no hay suficientes experiencias, muestrear con reemplazo
            rng = np.random.Generator(np.random.PCG64(CONST_DEFAULT_SEED))
            batch = rng.choice(self.buffer, batch_size, replace=True)
        else:
            # Muestreo aleatorio sin reemplazo
            rng = np.random.Generator(np.random.PCG64(CONST_DEFAULT_SEED))
            batch = rng.choice(self.buffer, batch_size, replace=False)
        
        states, actions, rewards, next_states, dones = zip(*batch)
        
        return (
            np.array(states, dtype=np.float32),
            np.array(actions, dtype=np.float32),
            np.array(rewards, dtype=np.float32).reshape(-1, 1),
            np.array(next_states, dtype=np.float32),
            np.array(dones, dtype=np.float32).reshape(-1, 1)
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


class ActorNetwork(nn.Module):
    """
    Red del Actor para SAC que produce una distribución de política gaussiana.
    
    Esta red mapea estados a distribuciones de probabilidad sobre acciones
    mediante una política estocástica parametrizada por una distribución normal.
    
    Parámetros:
    -----------
    action_dim : int
        Dimensión del espacio de acciones
    action_high : jnp.ndarray
        Límite superior del espacio de acciones
    action_low : jnp.ndarray
        Límite inferior del espacio de acciones
    hidden_units : Optional[Sequence[int]], opcional
        Unidades en capas ocultas (default: None)
    """
    action_dim: int
    action_high: jnp.ndarray
    action_low: jnp.ndarray
    hidden_units: Optional[Sequence[int]] = None
    
    @nn.compact
    def __call__(self, x: jnp.ndarray, training: bool = False) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """
        Realiza el forward pass del modelo actor.
        
        Parámetros:
        -----------
        x : jnp.ndarray
            Tensor de estados de entrada
        training : bool, opcional
            Si está en modo entrenamiento (default: False)
            
        Retorna:
        --------
        Tuple[jnp.ndarray, jnp.ndarray]
            (mean, std) - Media y desviación estándar de la distribución de política
        """
        hidden_units = self.hidden_units or SAC_CONFIG['actor_hidden_units']
        log_std_min = SAC_CONFIG['log_std_min']
        log_std_max = SAC_CONFIG['log_std_max']
        dropout_rate = SAC_CONFIG['dropout_rate']
        
        # Capas para procesar el estado
        for i, units in enumerate(hidden_units):
            x = nn.Dense(units, name=f'actor_dense_{i}')(x)
            x = getattr(nn, SAC_CONFIG['actor_activation'])(x)
            x = nn.LayerNorm(epsilon=SAC_CONFIG['epsilon'], name=f'actor_ln_{i}')(x)
            x = nn.Dropout(rate=dropout_rate, deterministic=not training, name=f'actor_dropout_{i}')(x)
        
        # Capas de salida para media y log-desviación estándar
        mean = nn.Dense(self.action_dim, name='actor_mean')(x)
        log_std = nn.Dense(self.action_dim, name='actor_log_std')(x)
        
        # Restringir log_std al rango especificado
        log_std = jnp.clip(log_std, log_std_min, log_std_max)
        std = jnp.exp(log_std)
        
        return mean, std
    
    def sample_action(self, mean: jnp.ndarray, std: jnp.ndarray, key: jnp.ndarray, 
                     deterministic: bool = False) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        """
        Muestrea una acción de la distribución de política.
        
        Parámetros:
        -----------
        mean : jnp.ndarray
            Media de la distribución gaussiana
        std : jnp.ndarray
            Desviación estándar de la distribución
        key : jnp.ndarray
            PRNG key para muestreo aleatorio
        deterministic : bool, opcional
            Si es True, devuelve la acción media (sin ruido) (default: False)
            
        Retorna:
        --------
        Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]
            (actions, log_probs, next_key) - Acciones, logaritmos de probabilidad y siguiente llave PRNG
        """
        if deterministic:
            # Para evaluación o explotación
            actions = mean
            return jax.lax.stop_gradient(self._scale_actions(actions)), None, key
        
        # Muestrear usando el truco de reparametrización para permitir backprop
        noise = jax.random.normal(key, mean.shape)
        z = mean + std * noise
        
        # Aplicar tanh para acotar las acciones
        actions = jnp.tanh(z)
        
        # Escalar acciones al rango deseado
        scaled_actions = self._scale_actions(actions)
        
        # Calcular log-probabilidad con corrección para tanh
        log_probs = self._log_prob(z, std, actions)
        
        return scaled_actions, log_probs, key
    
    def _scale_actions(self, actions: jnp.ndarray) -> jnp.ndarray:
        """
        Escala las acciones al rango deseado.
        
        Parámetros:
        -----------
        actions : jnp.ndarray
            Acciones normalizadas en el rango [-1, 1]
            
        Retorna:
        --------
        jnp.ndarray
            Acciones escaladas al rango [action_low, action_high]
        """
        return actions * (self.action_high - self.action_low) / 2 + (self.action_high + self.action_low) / 2
    
    def _log_prob(self, z: jnp.ndarray, std: jnp.ndarray, actions: jnp.ndarray) -> jnp.ndarray:
        """
        Calcula el logaritmo de la probabilidad de una acción.
        
        Parámetros:
        -----------
        z : jnp.ndarray
            Valor antes de aplicar tanh
        std : jnp.ndarray
            Desviación estándar
        actions : jnp.ndarray
            Acción muestreada
            
        Retorna:
        --------
        jnp.ndarray
            Log-probabilidad de la acción
        """
        # Log-prob de distribución normal
        log_prob_gaussian = -0.5 * (jnp.square(z) + 2 * jnp.log(std) + jnp.log(2.0 * jnp.pi))
        log_prob_gaussian = jnp.sum(log_prob_gaussian, axis=-1, keepdims=True)
        
        # Corrección por transformación tanh
        # Deriva de cambio de variable (ver paper SAC)
        squash_correction = jnp.sum(
            jnp.log(1.0 - jnp.square(jnp.tanh(z)) + 1e-6),
            axis=-1,
            keepdims=True
        )
        
        return log_prob_gaussian - squash_correction


class CriticNetwork(nn.Module):
    """
    Red de Crítico para SAC que mapea pares (estado, acción) a valores-Q.
    
    Parámetros:
    -----------
    hidden_units : Optional[Sequence[int]], opcional
        Unidades en capas ocultas (default: None)
    """
    hidden_units: Optional[Sequence[int]] = None
    
    @nn.compact
    def __call__(self, states: jnp.ndarray, actions: jnp.ndarray, training: bool = False) -> jnp.ndarray:
        """
        Realiza el forward pass del modelo crítico.
        
        Parámetros:
        -----------
        states : jnp.ndarray
            Tensor de estados
        actions : jnp.ndarray
            Tensor de acciones
        training : bool, opcional
            Si está en modo entrenamiento (default: False)
            
        Retorna:
        --------
        jnp.ndarray
            Valores Q estimados
        """
        hidden_units = self.hidden_units or SAC_CONFIG['critic_hidden_units']
        dropout_rate = SAC_CONFIG['dropout_rate']
        
        # Procesar el estado inicialmente
        x = states
        for i, units in enumerate(hidden_units[:1]):  # Primera capa solo para estado
            x = nn.Dense(units, name=f'critic_state_dense_{i}')(x)
            x = getattr(nn, SAC_CONFIG['critic_activation'])(x)
            x = nn.LayerNorm(epsilon=SAC_CONFIG['epsilon'], name=f'critic_state_ln_{i}')(x)
        
        # Combinar estado procesado con acción
        x = jnp.concatenate([x, actions], axis=-1)
        
        # Procesar la combinación
        for i, units in enumerate(hidden_units[1:]):
            x = nn.Dense(units, name=f'critic_combined_dense_{i}')(x)
            x = getattr(nn, SAC_CONFIG['critic_activation'])(x)
            x = nn.LayerNorm(epsilon=SAC_CONFIG['epsilon'], name=f'critic_combined_ln_{i}')(x)
            x = nn.Dropout(rate=dropout_rate, deterministic=not training, name=f'critic_dropout_{i}')(x)
        
        # Capa de salida: valor Q
        q_value = nn.Dense(1, name='critic_output')(x)
        
        return q_value


class SACTrainState(train_state.TrainState):
    """
    Estado de entrenamiento para SAC que extiende el TrainState de Flax.
    
    Atributos adicionales:
    --------------------
    target_params : flax.core.FrozenDict
        Parámetros de las redes target
    key : jnp.ndarray
        Llave PRNG para generación de números aleatorios
    """
    target_params: flax.core.FrozenDict
    key: jnp.ndarray


class SAC:
    """
    Implementación del algoritmo Soft Actor-Critic (SAC).
    
    SAC es un algoritmo de aprendizaje por refuerzo fuera de política (off-policy)
    basado en el marco de máxima entropía, que busca maximizar tanto el retorno
    esperado como la entropía de la política.
    
    Parámetros:
    -----------
    state_dim : int
        Dimensión del espacio de estados
    action_dim : int
        Dimensión del espacio de acciones
    action_high : np.ndarray
        Límite superior del espacio de acciones
    action_low : np.ndarray
        Límite inferior del espacio de acciones
    config : Optional[Dict[str, Any]], opcional
        Configuración personalizada (default: None)
    seed : int, opcional
        Semilla para reproducibilidad (default: 42)
    """
    def __init__(
        self, 
        state_dim: int, 
        action_dim: int,
        action_high: np.ndarray,
        action_low: np.ndarray,
        config: Optional[Dict[str, Any]] = None,
        seed: int = CONST_DEFAULT_SEED
    ) -> None:
        # Configurar semillas para reproducibilidad
        key = jax.random.PRNGKey(seed)
        np.random.seed(seed)
        random.seed(seed)
        
        # Use default config if none provided
        if config is None:
            config = {}
            
        # Set parameters from config with defaults from SAC_CONFIG
        actor_lr = config.get('actor_lr', SAC_CONFIG['actor_lr'])
        critic_lr = config.get('critic_lr', SAC_CONFIG['critic_lr'])
        alpha_lr = config.get('alpha_lr', SAC_CONFIG['alpha_lr'])
        self.gamma = config.get('gamma', SAC_CONFIG['gamma'])
        self.tau = config.get('tau', SAC_CONFIG['tau'])
        buffer_capacity = config.get('buffer_capacity', SAC_CONFIG['buffer_capacity'])
        self.batch_size = config.get('batch_size', SAC_CONFIG['batch_size'])
        initial_alpha = config.get('initial_alpha', SAC_CONFIG['initial_alpha'])
        target_entropy = config.get('target_entropy', None)
        actor_hidden_units = config.get('actor_hidden_units', None)
        critic_hidden_units = config.get('critic_hidden_units', None)
        
        # Parámetros del entorno y del modelo
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.action_high = jnp.array(action_high, dtype=jnp.float32)
        self.action_low = jnp.array(action_low, dtype=jnp.float32)
        
        # Crear directorio para figuras si no existe
        os.makedirs(FIGURES_DIR, exist_ok=True)
        
        # Valores predeterminados para capas ocultas
        if actor_hidden_units is None:
            self.actor_hidden_units = SAC_CONFIG['actor_hidden_units']
        else:
            self.actor_hidden_units = actor_hidden_units
            
        if critic_hidden_units is None:
            self.critic_hidden_units = SAC_CONFIG['critic_hidden_units']
        else:
            self.critic_hidden_units = critic_hidden_units
        
        # Crear modelos
        self.actor_network = ActorNetwork(
            action_dim=action_dim, 
            action_high=self.action_high, 
            action_low=self.action_low,
            hidden_units=self.actor_hidden_units
        )
        
        self.critic_network_1 = CriticNetwork(hidden_units=self.critic_hidden_units)
        self.critic_network_2 = CriticNetwork(hidden_units=self.critic_hidden_units)
        
        # Inicializar parámetros
        key, init_key_actor = jax.random.split(key)
        key, init_key_critic1 = jax.random.split(key)
        key, init_key_critic2 = jax.random.split(key)
        
        dummy_state = jnp.ones((1, state_dim), dtype=jnp.float32)
        dummy_action = jnp.ones((1, action_dim), dtype=jnp.float32)
        
        actor_params = self.actor_network.init(init_key_actor, dummy_state)
        critic_params_1 = self.critic_network_1.init(init_key_critic1, dummy_state, dummy_action)
        critic_params_2 = self.critic_network_2.init(init_key_critic2, dummy_state, dummy_action)
        
        # Crear optimizadores
        actor_tx = optax.adam(learning_rate=actor_lr)
        critic_tx = optax.adam(learning_rate=critic_lr)
        
        # Inicializar estados de entrenamiento
        self.actor_state = train_state.TrainState.create(
            apply_fn=self.actor_network.apply,
            params=actor_params,
            tx=actor_tx
        )
        
        self.critic_state_1 = SACTrainState.create(
            apply_fn=self.critic_network_1.apply,
            params=critic_params_1,
            tx=critic_tx,
            target_params=critic_params_1,
            key=key
        )
        
        key, subkey = jax.random.split(key)
        self.critic_state_2 = SACTrainState.create(
            apply_fn=self.critic_network_2.apply,
            params=critic_params_2,
            tx=critic_tx,
            target_params=critic_params_2,
            key=subkey
        )
        
        # Inicializar alpha (coeficiente de temperatura)
        self.log_alpha = jnp.array(jnp.log(initial_alpha), dtype=jnp.float32)
        self.alpha_optimizer = optax.adam(learning_rate=alpha_lr)
        self.alpha_opt_state = self.alpha_optimizer.init(self.log_alpha)
        
        # Entropía objetivo (heurística: -dim(A))
        if target_entropy is None:
            self.target_entropy = -action_dim  # Valor predeterminado: -dim(A)
        else:
            self.target_entropy = target_entropy
        
        # Buffer de experiencias
        self.replay_buffer = ReplayBuffer(buffer_capacity)
        
        # Métricas acumuladas
        self.actor_loss_sum = 0.0
        self.critic_loss_sum = 0.0
        self.alpha_loss_sum = 0.0
        self.entropy_sum = 0.0
        self.updates_count = 0
        
        # PRNG key
        self.key = key
        
        # Compilar funciones para mejorar rendimiento
        self.update_actor = jax.jit(self._update_actor)
        self.update_critics = jax.jit(self._update_critics)
        self.update_alpha = jax.jit(self._update_alpha)
        self.update_target_networks = jax.jit(self._update_target_networks)
    
    def _update_target_networks(
        self, 
        critic_state_1: SACTrainState, 
        critic_state_2: SACTrainState, 
        tau: Optional[float] = None
    ) -> Tuple[SACTrainState, SACTrainState]:
        """
        Actualiza los parámetros de las redes target con soft update.
        
        Parámetros:
        -----------
        critic_state_1 : SACTrainState
            Estado del primer crítico
        critic_state_2 : SACTrainState
            Estado del segundo crítico
        tau : Optional[float], opcional
            Factor de interpolación (default: None, usa el valor del objeto)
            
        Retorna:
        --------
        Tuple[SACTrainState, SACTrainState]
            Estados actualizados de los críticos
        """
        tau = tau if tau is not None else self.tau
        
        new_target_params_1 = jax.tree_map(
            lambda tp, p: (1 - tau) * tp + tau * p,
            critic_state_1.target_params,
            critic_state_1.params
        )
        
        new_target_params_2 = jax.tree_map(
            lambda tp, p: (1 - tau) * tp + tau * p,
            critic_state_2.target_params,
            critic_state_2.params
        )
        
        new_critic_state_1 = critic_state_1.replace(target_params=new_target_params_1)
        new_critic_state_2 = critic_state_2.replace(target_params=new_target_params_2)
        
        return new_critic_state_1, new_critic_state_2
    
    def _update_critics(
        self,
        critic_state_1: SACTrainState,
        critic_state_2: SACTrainState,
        actor_state: train_state.TrainState,
        log_alpha: jnp.ndarray,
        states: jnp.ndarray,
        actions: jnp.ndarray,
        rewards: jnp.ndarray,
        next_states: jnp.ndarray,
        dones: jnp.ndarray
    ) -> Tuple[SACTrainState, SACTrainState, jnp.ndarray, jnp.ndarray]:
        """
        Actualiza las redes de crítica.
        
        Parámetros:
        -----------
        critic_state_1 : SACTrainState
            Estado del primer crítico
        critic_state_2 : SACTrainState
            Estado del segundo crítico
        actor_state : train_state.TrainState
            Estado del actor
        log_alpha : jnp.ndarray
            Logaritmo del parámetro alpha
        states : jnp.ndarray
            Estados actuales
        actions : jnp.ndarray
            Acciones tomadas
        rewards : jnp.ndarray
            Recompensas recibidas
        next_states : jnp.ndarray
            Estados siguientes
        dones : jnp.ndarray
            Indicadores de fin de episodio
            
        Retorna:
        --------
        Tuple[SACTrainState, SACTrainState, jnp.ndarray, jnp.ndarray]
            Estados actualizados de los críticos, pérdida, y valores Q
        """
        key, actor_key = jax.random.split(critic_state_1.key)
        
        # Calcular alpha actual
        alpha = jnp.exp(log_alpha)
        
        # Obtener acciones y log_probs para el siguiente estado
        mean, std = actor_state.apply_fn(actor_state.params, next_states)
        next_actions, next_log_probs, _ = self.actor_network.sample_action(mean, std, actor_key)
        
        # Calcular valores Q para el siguiente estado usando redes target
        q1_next = critic_state_1.apply_fn(
            critic_state_1.target_params, next_states, next_actions)
        q2_next = critic_state_2.apply_fn(
            critic_state_2.target_params, next_states, next_actions)
        
        # Tomar el mínimo para evitar sobreestimación
        q_next = jnp.minimum(q1_next, q2_next)
        
        # Añadir término de entropía al Q-target (soft Q-learning)
        soft_q_next = q_next - alpha * next_log_probs
        
        # Calcular target usando ecuación de Bellman
        q_target = rewards + (1 - dones) * self.gamma * soft_q_next
        q_target = jax.lax.stop_gradient(q_target)
        
        # Definir función de pérdida para crítico 1
        def critic_1_loss_fn(params):
            q1_pred = critic_state_1.apply_fn(params, states, actions)
            return jnp.mean(jnp.square(q_target - q1_pred))
        
        # Definir función de pérdida para crítico 2
        def critic_2_loss_fn(params):
            q2_pred = critic_state_2.apply_fn(params, states, actions)
            return jnp.mean(jnp.square(q_target - q2_pred))
        
        # Calcular gradientes y actualizar crítico 1
        grad_fn_1 = jax.value_and_grad(critic_1_loss_fn)
        critic_1_loss, critic_1_grads = grad_fn_1(critic_state_1.params)
        
        # Calcular gradientes y actualizar crítico 2
        grad_fn_2 = jax.value_and_grad(critic_2_loss_fn)
        critic_2_loss, critic_2_grads = grad_fn_2(critic_state_2.params)
        
        # Aplicar gradientes
        new_critic_state_1 = critic_state_1.apply_gradients(
            grads=critic_1_grads, key=key)
        
        key, subkey = jax.random.split(key)
        new_critic_state_2 = critic_state_2.apply_gradients(
            grads=critic_2_grads, key=subkey)
        
        # Calcular pérdida total
        critic_loss = critic_1_loss + critic_2_loss
        
        # Para métricas, calcular un Q-value para seguimiento
        q_value = critic_state_1.apply_fn(critic_state_1.params, states, actions)
        
        return new_critic_state_1, new_critic_state_2, critic_loss, jnp.mean(q_value)
    
    def _update_actor(
        self,
        actor_state: train_state.TrainState,
        critic_state_1: SACTrainState,
        critic_state_2: SACTrainState,
        log_alpha: jnp.ndarray,
        states: jnp.ndarray
    ) -> Tuple[train_state.TrainState, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        """
        Actualiza la red del actor.
        
        Parámetros:
        -----------
        actor_state : train_state.TrainState
            Estado del actor
        critic_state_1 : SACTrainState
            Estado del primer crítico
        critic_state_2 : SACTrainState
            Estado del segundo crítico
        log_alpha : jnp.ndarray
            Logaritmo del parámetro alpha
        states : jnp.ndarray
            Estados actuales
            
        Retorna:
        --------
        Tuple[train_state.TrainState, jnp.ndarray, jnp.ndarray, jnp.ndarray]
            Estado actualizado del actor, pérdida, log_probs, y entropía
        """
        alpha = jnp.exp(log_alpha)
        key = critic_state_1.key
        
        def actor_loss_fn(params):
            # Obtener distribución de política
            mean, std = actor_state.apply_fn(params, states)
            actions, log_probs, _ = self.actor_network.sample_action(mean, std, key)
            
            # Calcular valores Q
            q1 = critic_state_1.apply_fn(critic_state_1.params, states, actions)
            q2 = critic_state_2.apply_fn(critic_state_2.params, states, actions)
            
            # Tomar el mínimo para evitar sobreestimación
            q = jnp.minimum(q1, q2)
            
            # Pérdida del actor: minimizar KL divergence entre política y Q suavizada
            actor_loss = jnp.mean(alpha * log_probs - q)
            
            return actor_loss, (log_probs, -jnp.mean(log_probs))
        
        # Calcular gradientes y actualizar actor
        grad_fn = jax.value_and_grad(actor_loss_fn, has_aux=True)
        (actor_loss, (log_probs, entropy)), actor_grads = grad_fn(actor_state.params)
        
        new_actor_state = actor_state.apply_gradients(grads=actor_grads)
        
        return new_actor_state, actor_loss, log_probs, entropy
    
    def _update_alpha(
        self,
        log_alpha: jnp.ndarray,
        opt_state: optax.OptState,
        log_probs: jnp.ndarray
    ) -> Tuple[jnp.ndarray, optax.OptState, jnp.ndarray]:
        """
        Actualiza el parámetro de temperatura alpha.
        
        Parámetros:
        -----------
        log_alpha : jnp.ndarray
            Logaritmo del parámetro alpha
        opt_state : optax.OptState
            Estado del optimizador de alpha
        log_probs : jnp.ndarray
            Log-probabilidades de las acciones muestreadas
            
        Retorna:
        --------
        Tuple[jnp.ndarray, optax.OptState, jnp.ndarray]
            Nuevo log_alpha, estado del optimizador, y pérdida
        """
        def alpha_loss_fn(log_alpha):
            alpha = jnp.exp(log_alpha)
            # Objetivo: ajustar alpha para alcanzar la entropía objetivo
            alpha_loss = -jnp.mean(
                alpha * (jnp.mean(log_probs) + self.target_entropy)
            )
            return alpha_loss
        
        # Calcular gradientes y actualizar alpha
        alpha_loss, alpha_grads = jax.value_and_grad(alpha_loss_fn)(log_alpha)
        
        updates, new_opt_state = self.alpha_optimizer.update(
            alpha_grads, opt_state, log_alpha)
        new_log_alpha = optax.apply_updates(log_alpha, updates)
        
        return new_log_alpha, new_opt_state, alpha_loss
    
    def train_step(self) -> Tuple[Optional[float], Optional[float], Optional[float], Optional[float]]:
        """
        Realiza un paso de entrenamiento completo (actor, crítico y alpha).
        
        Retorna:
        --------
        Tuple[Optional[float], Optional[float], Optional[float], Optional[float]]
            (actor_loss, critic_loss, alpha_loss, entropy) o None si no hay suficientes datos
        """
        # Si no hay suficientes datos, no hacer nada
        if len(self.replay_buffer) < self.batch_size:
            return None, None, None, None
        
        # Muestrear batch del buffer
        states, actions, rewards, next_states, dones = self.replay_buffer.sample(self.batch_size)
        
        # Convertir a jnp arrays
        states = jnp.array(states, dtype=jnp.float32)
        actions = jnp.array(actions, dtype=jnp.float32)
        rewards = jnp.array(rewards, dtype=jnp.float32)
        next_states = jnp.array(next_states, dtype=jnp.float32)
        dones = jnp.array(dones, dtype=jnp.float32)
        
        # Actualizar críticos
        self.critic_state_1, self.critic_state_2, critic_loss, _ = self.update_critics(
            self.critic_state_1,
            self.critic_state_2,
            self.actor_state,
            self.log_alpha,
            states,
            actions,
            rewards,
            next_states,
            dones
        )
        
        # Actualizar actor
        self.actor_state, actor_loss, log_probs, entropy = self.update_actor(
            self.actor_state,
            self.critic_state_1,
            self.critic_state_2,
            self.log_alpha,
            states
        )
        
        # Actualizar alpha
        self.log_alpha, self.alpha_opt_state, alpha_loss = self.update_alpha(
            self.log_alpha,
            self.alpha_opt_state,
            log_probs
        )
        
        # Actualizar redes target
        self.critic_state_1, self.critic_state_2 = self.update_target_networks(
            self.critic_state_1,
            self.critic_state_2
        )
        
        # Actualizar métricas acumuladas
        self.actor_loss_sum += float(actor_loss)
        self.critic_loss_sum += float(critic_loss)
        self.alpha_loss_sum += float(alpha_loss)
        self.entropy_sum += float(entropy)
        self.updates_count += 1
        
        return float(actor_loss), float(critic_loss), float(alpha_loss), float(entropy)
    
    def get_action(self, state: np.ndarray, deterministic: bool = False) -> np.ndarray:
        """
        Obtiene una acción basada en el estado actual.
        
        Parámetros:
        -----------
        state : np.ndarray
            Estado actual
        deterministic : bool, opcional
            Si es True, devuelve acción determinística (para evaluación) (default: False)
            
        Retorna:
        --------
        np.ndarray
            Acción seleccionada
        """
        # Convertir a jnp array y añadir dimensión de batch
        state = jnp.array(state, dtype=jnp.float32)[None, :]
        
        # Obtener distribución de política
        mean, std = self.actor_state.apply_fn(self.actor_state.params, state)
        
        # Obtener acción
        self.key, subkey = jax.random.split(self.key)
        action, _, _ = self.actor_network.sample_action(mean, std, subkey, deterministic)
        
        # Convertir a numpy y eliminar dimensión de batch
        return np.array(action[0], dtype=np.float32)
    
    def _init_training_history(self) -> Dict[str, List[float]]:
        """
        Inicializa el diccionario para almacenar la historia del entrenamiento.
        
        Retorna:
        --------
        Dict[str, List[float]]
            Estructura para almacenar la historia del entrenamiento
        """
        return {
            'episode_rewards': [],
            'actor_losses': [],
            'critic_losses': [],
            'alpha_losses': [],
            'entropies': [],
            'alphas': [],
            'eval_rewards': []
        }
        
    def _get_action_for_training(self, state: np.ndarray, total_steps: int, warmup_steps: int) -> np.ndarray:
        """
        Obtiene la acción para entrenamiento, considerando el período de calentamiento.
        
        Parámetros:
        -----------
        state : np.ndarray
            Estado actual
        total_steps : int
            Pasos totales ejecutados
        warmup_steps : int
            Pasos de calentamiento antes de usar la política
            
        Retorna:
        --------
        np.ndarray
            Acción seleccionada
        """
        if total_steps < warmup_steps:
            # Use the same seed pattern as JAX for consistency
            rng = np.random.default_rng(seed=int(self.key[0]))
            return rng.uniform(self.action_low, self.action_high, self.action_dim)
        else:
            return self.get_action(state, deterministic=False)
    
    def _update_model_multiple_times(self, update_count: int) -> Tuple[List[float], List[float], List[float], List[float]]:
        """
        Actualiza el modelo múltiples veces y devuelve las métricas.
        
        Parámetros:
        -----------
        update_count : int
            Número de actualizaciones a realizar
            
        Retorna:
        --------
        Tuple[List[float], List[float], List[float], List[float]]
            Listas con las pérdidas y entropía de cada actualización
        """
        episode_actor_loss = []
        episode_critic_loss = []
        episode_alpha_loss = []
        episode_entropy = []
        
        for _ in range(update_count):
            actor_loss, critic_loss, alpha_loss, entropy = self.train_step()
            
            # Almacenar pérdidas si hubo actualización
            if actor_loss is not None:
                episode_actor_loss.append(actor_loss)
                episode_critic_loss.append(critic_loss)
                episode_alpha_loss.append(alpha_loss)
                episode_entropy.append(entropy)
        
        return episode_actor_loss, episode_critic_loss, episode_alpha_loss, episode_entropy
    
    def _update_history_with_metrics(self, history: Dict[str, List[float]], 
                                     episode_reward: float,
                                     episode_actor_loss: List[float],
                                     episode_critic_loss: List[float],
                                     episode_alpha_loss: List[float],
                                     episode_entropy: List[float]) -> None:
        """
        Actualiza el historial con las métricas del episodio.
        
        Parámetros:
        -----------
        history : Dict[str, List[float]]
            Historial de entrenamiento
        episode_reward : float
            Recompensa del episodio
        episode_actor_loss : List[float]
            Pérdidas del actor
        episode_critic_loss : List[float]
            Pérdidas del crítico
        episode_alpha_loss : List[float]
            Pérdidas de alpha
        episode_entropy : List[float]
            Entropía
        """
        history['episode_rewards'].append(episode_reward)
        
        if episode_actor_loss:
            history['actor_losses'].append(np.mean(episode_actor_loss))
            history['critic_losses'].append(np.mean(episode_critic_loss))
            history['alpha_losses'].append(np.mean(episode_alpha_loss))
            history['entropies'].append(np.mean(episode_entropy))
        else:
            history['actor_losses'].append(float('nan'))
            history['critic_losses'].append(float('nan'))
            history['alpha_losses'].append(float('nan'))
            history['entropies'].append(float('nan'))
        
        history['alphas'].append(float(jnp.exp(self.log_alpha)))
    
    def _reset_metrics(self) -> None:
        """
        Resetea las métricas acumuladas.
        """
        self.actor_loss_sum = 0.0
        self.critic_loss_sum = 0.0
        self.alpha_loss_sum = 0.0
        self.entropy_sum = 0.0
        self.updates_count = 0
    
    def _run_episode(self, env: Any, max_steps: int, warmup_steps: int, 
                   update_after: int, update_every: int, render: bool, 
                   total_steps: int) -> Tuple[float, int, List[float], List[float], List[float], List[float]]:
        """
        Ejecuta un episodio completo de entrenamiento.
        
        Parámetros:
        -----------
        env : Any
            Entorno de Gym o compatible
        max_steps : int
            Pasos máximos por episodio
        warmup_steps : int
            Pasos iniciales con acciones aleatorias
        update_after : int
            Pasos antes de empezar a entrenar
        update_every : int
            Frecuencia de actualización
        render : bool
            Si se debe renderizar el entorno
        total_steps : int
            Pasos totales acumulados
            
        Retorna:
        --------
        Tuple[float, int, List[float], List[float], List[float], List[float]]
            Recompensa del episodio, pasos totales actualizados, y listas de métricas
        """
        state, _ = env.reset()
        state = np.array(state, dtype=np.float32)
        episode_reward = 0
        
        # Variables para almacenar pérdidas del episodio
        episode_actor_loss = []
        episode_critic_loss = []
        episode_alpha_loss = []
        episode_entropy = []
        
        for _ in range(max_steps):
            if render:
                env.render()
                
            # Obtener acción según la etapa de entrenamiento
            action = self._get_action_for_training(state, total_steps, warmup_steps)
            
            # Ejecutar acción
            next_state, reward, done, _, _ = env.step(action)
            next_state = np.array(next_state, dtype=np.float32)
            
            # Guardar transición en buffer
            self.replay_buffer.add(state, action, reward, next_state, float(done))
            
            # Actualizar estado y contador
            state = next_state
            episode_reward += reward
            total_steps += 1
            
            # Entrenar si es momento
            if total_steps >= update_after and total_steps % update_every == 0:
                actor_losses, critic_losses, alpha_losses, entropies = self._update_model_multiple_times(update_every)
                episode_actor_loss.extend(actor_losses)
                episode_critic_loss.extend(critic_losses)
                episode_alpha_loss.extend(alpha_losses)
                episode_entropy.extend(entropies)
            
            if done:
                break
                
        return episode_reward, total_steps, episode_actor_loss, episode_critic_loss, episode_alpha_loss, episode_entropy
    
    def _evaluate_and_log(self, env: Any, episode: int, episodes: int, 
                        episode_reward_history: List[float], evaluate_interval: int, 
                        best_reward: float, history: Dict[str, List[float]]) -> float:
        """
        Evalúa el modelo y registra los resultados.
        
        Parámetros:
        -----------
        env : Any
            Entorno de Gym o compatible
        episode : int
            Número de episodio actual
        episodes : int
            Número total de episodios
        episode_reward_history : List[float]
            Historial de recompensas para cálculo de promedio
        evaluate_interval : int
            Intervalo entre evaluaciones
        best_reward : float
            Mejor recompensa hasta el momento
        history : Dict[str, List[float]]
            Historial de entrenamiento
            
        Retorna:
        --------
        float
            Mejor recompensa actualizada
        """
        if (episode + 1) % evaluate_interval == 0:
            avg_reward = np.mean(episode_reward_history)
            print(f"Episodio {episode+1}/{episodes} - Recompensa Promedio: {avg_reward:.2f}, "
                  f"Alpha: {float(jnp.exp(self.log_alpha)):.4f}")
            
            # Evaluar rendimiento actual
            eval_reward = self.evaluate(env, episodes=3, render=False)
            history['eval_rewards'].append(eval_reward)
            
            # Guardar mejor modelo
            if eval_reward > best_reward:
                best_reward = eval_reward
                print(f"Nuevo mejor modelo con recompensa de evaluación: {best_reward:.2f}")
                
        return best_reward
    
    def train(self, env: Any, episodes: int = 1000, max_steps: int = 1000, 
              warmup_steps: int = 10000, update_after: int = 1000, update_every: int = 50,
              evaluate_interval: int = 10, render: bool = False) -> Dict[str, List[float]]:
        """
        Entrena el agente SAC en un entorno dado.
        
        Parámetros:
        -----------
        env : Any
            Entorno de Gym o compatible
        episodes : int, opcional
            Número máximo de episodios (default: 1000)
        max_steps : int, opcional
            Pasos máximos por episodio (default: 1000)
        warmup_steps : int, opcional
            Pasos iniciales con acciones aleatorias para explorar (default: 10000)
        update_after : int, opcional
            Pasos antes de empezar a entrenar (default: 1000)
        update_every : int, opcional
            Frecuencia de actualización (default: 50)
        evaluate_interval : int, opcional
            Episodios entre evaluaciones (default: 10)
        render : bool, opcional
            Mostrar entorno gráficamente (default: False)
            
        Retorna:
        --------
        Dict[str, List[float]]
            Historia de entrenamiento
        """
        history = self._init_training_history()
        
        # Variables para seguimiento de progreso
        best_reward = -float('inf')
        episode_reward_history = []
        total_steps = 0
        
        for episode in range(episodes):
            # Ejecutar un episodio completo
            episode_reward, total_steps, episode_actor_loss, episode_critic_loss, episode_alpha_loss, episode_entropy = self._run_episode(
                env, max_steps, warmup_steps, update_after, update_every, render, total_steps
            )
            
            # Actualizar historial con métricas
            self._update_history_with_metrics(
                history, episode_reward, episode_actor_loss, episode_critic_loss, 
                episode_alpha_loss, episode_entropy
            )
            
            # Resetear métricas acumuladas
            self._reset_metrics()
            
            # Guardar últimas recompensas para promedio
            episode_reward_history.append(episode_reward)
            if len(episode_reward_history) > evaluate_interval:
                episode_reward_history.pop(0)
            
            # Evaluar y mostrar progreso periódicamente
            best_reward = self._evaluate_and_log(
                env, episode, episodes, episode_reward_history, evaluate_interval, 
                best_reward, history
            )
        
        return history
    
    def evaluate(self, env: Any, episodes: int = 10, render: bool = False) -> float:
        """
        Evalúa el agente SAC en un entorno dado sin exploración.
        
        Parámetros:
        -----------
        env : Any
            Entorno de Gym o compatible
        episodes : int, opcional
            Número de episodios para evaluar (default: 10)
        render : bool, opcional
            Si se debe renderizar el entorno (default: False)
            
        Retorna:
        --------
        float
            Recompensa promedio obtenida
        """
        rewards = []
        
        for _ in range(episodes):
            state, _ = env.reset()
            state = np.array(state, dtype=np.float32)
            episode_reward = 0
            done = False
            
            while not done:
                # Seleccionar acción determinística
                action = self.get_action(state, deterministic=True)
                
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
        
        avg_reward = np.mean(rewards)
        return avg_reward
    
    def save_models(self, directory: str) -> None:
        """
        Guarda los modelos y parámetros del agente.
        
        Parámetros:
        -----------
        directory : str
            Directorio donde guardar los modelos
        """
        if not os.path.exists(directory):
            os.makedirs(directory)
        
        # Guardar parámetros de las redes
        with open(os.path.join(directory, 'actor_params.pkl'), 'wb') as f:
            f.write(flax.serialization.to_bytes(self.actor_state.params))
            
        with open(os.path.join(directory, 'critic_1_params.pkl'), 'wb') as f:
            f.write(flax.serialization.to_bytes(self.critic_state_1.params))
            
        with open(os.path.join(directory, 'critic_2_params.pkl'), 'wb') as f:
            f.write(flax.serialization.to_bytes(self.critic_state_2.params))
        
        # Guardar alpha
        alpha = jnp.exp(self.log_alpha).item()
        np.save(os.path.join(directory, 'alpha.npy'), alpha)
        
        print(f"Modelos guardados en {directory}")
    
    def load_models(self, directory: str) -> None:
        """
        Carga los modelos y parámetros del agente.
        
        Parámetros:
        -----------
        directory : str
            Directorio de donde cargar los modelos
        """
        # Cargar parámetros de las redes
        try:
            with open(os.path.join(directory, 'actor_params.pkl'), 'rb') as f:
                actor_params = flax.serialization.from_bytes(
                    self.actor_state.params, f.read())
                self.actor_state = self.actor_state.replace(params=actor_params)
            
            with open(os.path.join(directory, 'critic_1_params.pkl'), 'rb') as f:
                critic_1_params = flax.serialization.from_bytes(
                    self.critic_state_1.params, f.read())
                self.critic_state_1 = self.critic_state_1.replace(
                    params=critic_1_params, target_params=critic_1_params)
            
            with open(os.path.join(directory, 'critic_2_params.pkl'), 'rb') as f:
                critic_2_params = flax.serialization.from_bytes(
                    self.critic_state_2.params, f.read())
                self.critic_state_2 = self.critic_state_2.replace(
                    params=critic_2_params, target_params=critic_2_params)
            
            # Guardar alpha
            try:
                alpha = np.load(os.path.join(directory, 'alpha.npy')).item()
                self.log_alpha = jnp.array(np.log(alpha), dtype=jnp.float32)
                self.alpha_opt_state = self.alpha_optimizer.init(self.log_alpha)
            except (FileNotFoundError, IOError) as e:
                print(f"No se pudo cargar alpha, usando el valor actual: {str(e)}")
            
            print(f"Modelos cargados desde {directory}")
        except Exception as e:
            print(f"Error al cargar los modelos: {str(e)}")
    
    def visualize_training(self, history: Dict[str, List[float]], smoothing_window: int = 10) -> None:
        """
        Visualiza los resultados del entrenamiento.
        
        Parámetros:
        -----------
        history : Dict[str, List[float]]
            Historia de entrenamiento
        smoothing_window : int, opcional
            Ventana para suavizado de gráficos (default: 10)
        """
        # Función para suavizar datos
        def smooth(data, window_size):
            if len(data) < window_size:
                return data
            kernel = np.ones(window_size) / window_size
            return np.convolve(data, kernel, mode='valid')
        
        # Crear figura con múltiples subplots
        _, axs = plt.subplots(3, 2, figsize=(16, 12))
        
        # 1. Graficar recompensas de episodio
        rewards = history['episode_rewards']
        axs[0, 0].plot(rewards, alpha=0.3, color='blue')
        if len(rewards) > smoothing_window:
            smoothed_rewards = smooth(rewards, smoothing_window)
            axs[0, 0].plot(range(smoothing_window-1, len(rewards)), 
                         smoothed_rewards, color='blue', label='Suavizado')
        axs[0, 0].set_title('Recompensa por Episodio')
        axs[0, 0].set_xlabel('Episodio')
        axs[0, 0].set_ylabel('Recompensa')
        axs[0, 0].grid(alpha=0.3)
        axs[0, 0].legend()
        
        # 2. Graficar recompensas de evaluación
        eval_rewards = history['eval_rewards']
        if eval_rewards:
            eval_interval = len(rewards) // len(eval_rewards)
            x_eval = [i * eval_interval for i in range(len(eval_rewards))]
            axs[0, 1].plot(x_eval, eval_rewards, color='green', marker='o')
            axs[0, 1].set_title('Recompensa de Evaluación')
            axs[0, 1].set_xlabel('Episodio')
            axs[0, 1].set_ylabel('Recompensa Promedio')
            axs[0, 1].grid(alpha=0.3)
        
        # 3. Graficar pérdidas del actor
        actor_losses = [l for l in history['actor_losses'] if not np.isnan(l)]
        if actor_losses:
            axs[1, 0].plot(actor_losses, alpha=0.3, color='red')
            if len(actor_losses) > smoothing_window:
                smoothed_actor_losses = smooth(actor_losses, smoothing_window)
                axs[1, 0].plot(range(smoothing_window-1, len(actor_losses)), 
                             smoothed_actor_losses, color='red', label='Suavizado')
            axs[1, 0].set_title('Pérdida del Actor')
            axs[1, 0].set_xlabel('Episodio')
            axs[1, 0].set_ylabel('Pérdida')
            axs[1, 0].grid(alpha=0.3)
            axs[1, 0].legend()
        
        # 4. Graficar pérdidas del crítico
        critic_losses = [l for l in history['critic_losses'] if not np.isnan(l)]
        if critic_losses:
            axs[1, 1].plot(critic_losses, alpha=0.3, color='purple')
            if len(critic_losses) > smoothing_window:
                smoothed_critic_losses = smooth(critic_losses, smoothing_window)
                axs[1, 1].plot(range(smoothing_window-1, len(critic_losses)), 
                              smoothed_critic_losses, color='purple', label='Suavizado')
            axs[1, 1].set_title('Pérdida del Crítico')
            axs[1, 1].set_xlabel('Episodio')
            axs[1, 1].set_ylabel('Pérdida')
            axs[1, 1].grid(alpha=0.3)
            axs[1, 1].legend()
        
        # 5. Graficar entropía
        entropies = [e for e in history['entropies'] if not np.isnan(e)]
        if entropies:
            axs[2, 0].plot(entropies, alpha=0.3, color='orange')
            if len(entropies) > smoothing_window:
                smoothed_entropies = smooth(entropies, smoothing_window)
                axs[2, 0].plot(range(smoothing_window-1, len(entropies)), 
                              smoothed_entropies, color='orange', label='Suavizado')
            axs[2, 0].set_title('Entropía')
            axs[2, 0].set_xlabel('Episodio')
            axs[2, 0].set_ylabel('Entropía')
            axs[2, 0].grid(alpha=0.3)
            axs[2, 0].legend()
        
        # 6. Graficar alpha
        alphas = history['alphas']
        axs[2, 1].plot(alphas, color='brown')
        axs[2, 1].set_title('Coeficiente Alpha')
        axs[2, 1].set_xlabel('Episodio')
        axs[2, 1].set_ylabel('Alpha')
        axs[2, 1].grid(alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(FIGURES_DIR, 'training_history.png'))
        plt.show()

class SACWrapper:
    """
    Wrapper para hacer que el agente SAC sea compatible con la interfaz de modelos de aprendizaje profundo.
    """
    def __init__(
        self,
        sac_agent: SAC,
        cgm_shape: Tuple[int, ...],
        other_features_shape: Tuple[int, ...],
    ) -> None:
        """
        Inicializa el wrapper para SAC.
        
        Parámetros:
        -----------
        sac_agent : SAC
            Agente SAC a utilizar
        cgm_shape : Tuple[int, ...]
            Forma de entrada para datos CGM
        other_features_shape : Tuple[int, ...]
            Forma de entrada para otras características
        """
        self.sac_agent = sac_agent
        self.cgm_shape = cgm_shape
        self.other_features_shape = other_features_shape
        
        # Crear preprocesadores para convertir las entradas en estados para SAC
        self.key = jax.random.PRNGKey(CONST_DEFAULT_SEED)
        self.key, self.encoder_key = jax.random.split(self.key)
        
        # Configurar funciones de codificación para entradas
        self._setup_encoders()
        
        # Historial de entrenamiento
        self.history = {
            'loss': [],
            'val_loss': [],
            'actor_losses': [],
            'critic_losses': [],
            'alpha_losses': [],
            'episode_rewards': []
        }
    
    def _setup_encoders(self) -> None:
        """
        Configura las funciones de codificación para procesar las entradas.
        """
        # Calcular dimensiones de entrada aplanadas de manera segura
        if len(self.cgm_shape) <= 1:
            cgm_dim = 1
        else:
            cgm_dim = int(np.prod(self.cgm_shape[1:]))
        
        if len(self.other_features_shape) <= 1:
            other_dim = 1 
        else:
            other_dim = int(np.prod(self.other_features_shape[1:]))
        
        # Inicializar matrices de transformación
        self.key, key_cgm, key_other = jax.random.split(self.key, 3)
        
        # Crear matrices de proyección para la codificación de entradas
        self.cgm_weight = jax.random.normal(key_cgm, (int(cgm_dim), self.sac_agent.state_dim // 2))
        self.other_weight = jax.random.normal(key_other, (int(other_dim), self.sac_agent.state_dim // 2))
        
        # JIT-compilar transformaciones para mayor rendimiento
        self.encode_cgm = jax.jit(self._create_encoder_fn(self.cgm_weight))
        self.encode_other = jax.jit(self._create_encoder_fn(self.other_weight))
    
    def _create_encoder_fn(self, weights: jnp.ndarray) -> Callable:
        """
        Crea una función de codificación.
        
        Parámetros:
        -----------
        weights : jnp.ndarray
            Matriz de pesos para la transformación
            
        Retorna:
        --------
        Callable
            Función de codificación JIT-compilada
        """
        def encoder_fn(x):
            x_flat = x.reshape((x.shape[0], -1))
            return jnp.tanh(jnp.dot(x_flat, weights))
        return encoder_fn
    
    def __call__(self, inputs: list) -> jnp.ndarray:
        """
        Implementa la interfaz de llamada para predicción.
        
        Parámetros:
        -----------
        inputs : list
            Lista con [cgm_data, other_features]
            
        Retorna:
        --------
        jnp.ndarray
            Predicciones de dosis de insulina
        """
        return self.predict(inputs)
    
    def predict(self, inputs: list) -> jnp.ndarray:
        """
        Realiza predicciones con el modelo SAC.
        
        Parámetros:
        -----------
        inputs : list
            Lista con [cgm_data, other_features]
            
        Retorna:
        --------
        jnp.ndarray
            Predicciones de dosis de insulina
        """
        # Obtener entradas
        cgm_data, other_features = inputs
        
        # Convertir a arrays de JAX si no lo son
        cgm_data = jnp.array(cgm_data)
        other_features = jnp.array(other_features)
        
        # Codificar entradas a estado
        cgm_encoded = self.encode_cgm(cgm_data)
        other_encoded = self.encode_other(other_features)
        states = jnp.concatenate([cgm_encoded, other_encoded], axis=1)
        
        # Obtener acciones usando el agente SAC (en modo determinístico)
        batch_size = states.shape[0]
        actions = np.zeros((batch_size, 1))
        
        for i in range(batch_size):
            state = np.array(states[i])
            action = self.sac_agent.get_action(state, deterministic=True)
            actions[i] = action
        
        return actions
    
    def fit(
        self, 
        x: list, 
        y: jnp.ndarray, 
        validation_data: Optional[Tuple] = None, 
        epochs: int = 1,
        batch_size: int = 32,
        callbacks: list = None,
        verbose: int = 0
    ) -> Dict:
        """
        Entrena el modelo SAC en los datos proporcionados.
        
        Parámetros:
        -----------
        x : list
            Lista con [cgm_data, other_features]
        y : jnp.ndarray
            Etiquetas (dosis objetivo)
        validation_data : Optional[Tuple], opcional
            Datos de validación (default: None)
        epochs : int, opcional
            Número de épocas (default: 1)
        batch_size : int, opcional
            Tamaño del lote (default: 32)
        callbacks : list, opcional
            Lista de callbacks (default: None)
        verbose : int, opcional
            Nivel de verbosidad (default: 0)
            
        Retorna:
        --------
        Dict
            Historia del entrenamiento
        """
        if verbose > 0:
            print("Entrenando modelo SAC...")
            
        # Crear entorno simulado para RL a partir de los datos
        env = self._create_training_environment(x[0], x[1], y)
        
        # Entrenar el agente SAC
        sac_history = self.sac_agent.train(
            env=env,
            episodes=epochs,
            max_steps=batch_size,
            warmup_steps=min(1000, batch_size),
            update_after=min(500, batch_size // 2),
            update_every=1,
            evaluate_interval=max(1, epochs // 5) if verbose > 0 else epochs + 1,
            render=False
        )
        
        # Actualizar historial con las métricas del entrenamiento
        self.history['episode_rewards'].extend(sac_history.get('episode_rewards', []))
        self.history['actor_losses'].extend(sac_history.get('actor_losses', []))
        self.history['critic_losses'].extend(sac_history.get('critic_losses', []))
        self.history['alpha_losses'].extend(sac_history.get('alpha_losses', []))
        
        # Calcular pérdida en los datos de entrenamiento
        train_preds = self.predict(x)
        train_loss = float(jnp.mean((train_preds.flatten() - y) ** 2))
        self.history['loss'].append(train_loss)
        
        # Evaluar en datos de validación si se proporcionan
        if validation_data:
            val_x, val_y = validation_data
            val_preds = self.predict(val_x)
            val_loss = float(jnp.mean((val_preds.flatten() - val_y) ** 2))
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
        Crea un entorno de entrenamiento para RL a partir de los datos.
        
        Parámetros:
        -----------
        cgm_data : jnp.ndarray
            Datos CGM
        other_features : jnp.ndarray
            Otras características
        targets : jnp.ndarray
            Dosis objetivo
            
        Retorna:
        --------
        Any
            Entorno simulado para RL
        """
        # Crear entorno personalizado para SAC
        class InsulinDosingEnv:
            def __init__(self, cgm, features, targets, model_wrapper):
                self.cgm = np.array(cgm)
                self.features = np.array(features)
                self.targets = np.array(targets)
                self.model = model_wrapper
                self.rng = np.random.Generator(np.random.PCG64(CONST_DEFAULT_SEED))
                self.current_idx = 0
                self.max_idx = len(targets) - 1
                
                # Para compatibilidad con algoritmos RL
                # Espacio de observación
                self.observation_space = SimpleNamespace(
                    shape=(model_wrapper.sac_agent.state_dim,),
                    low=-np.ones(model_wrapper.sac_agent.state_dim) * 10,
                    high=np.ones(model_wrapper.sac_agent.state_dim) * 10
                )
                
                # Espacio de acción
                self.action_space = SimpleNamespace(
                    shape=model_wrapper.sac_agent.action_high.shape,
                    low=model_wrapper.sac_agent.action_low,
                    high=model_wrapper.sac_agent.action_high,
                    sample=self._sample_action
                )
            
            def _sample_action(self):
                """Muestrea una acción aleatoria del espacio continuo."""
                return self.rng.uniform(
                    self.action_space.low, 
                    self.action_space.high
                )
                
            def reset(self):
                """Reinicia el entorno eligiendo un ejemplo aleatorio."""
                self.current_idx = self.rng.integers(0, self.max_idx)
                state = self._get_state()
                return state, {}
                
            def step(self, action):
                """Ejecuta un paso con la acción dada."""
                # Obtener valor de dosis (primera dimensión de la acción)
                dose = float(action[0])
                
                # Calcular recompensa como negativo del error absoluto
                target = self.targets[self.current_idx]
                reward = -abs(dose - target)
                
                # Avanzar al siguiente ejemplo
                self.current_idx = (self.current_idx + 1) % self.max_idx
                
                # Obtener próximo estado
                next_state = self._get_state()
                
                # Episodio siempre termina después de un paso (para este problema)
                done = True
                truncated = False
                
                return next_state, reward, done, truncated, {}
            
            def _get_state(self):
                """Obtiene el estado codificado para el ejemplo actual."""
                # Obtener datos actuales
                cgm_batch = self.cgm[self.current_idx:self.current_idx+1]
                features_batch = self.features[self.current_idx:self.current_idx+1]
                
                # Codificar entradas a espacio de estado
                cgm_encoded = self.model.encode_cgm(jnp.array(cgm_batch))
                other_encoded = self.model.encode_other(jnp.array(features_batch))
                
                # Combinar características
                state = np.concatenate([cgm_encoded[0], other_encoded[0]])
                
                return state
        
        # Crear y devolver el entorno
        return InsulinDosingEnv(cgm_data, other_features, targets, self)
    
    def save(self, filepath: str) -> None:
        """
        Guarda el modelo SAC en un archivo.
        
        Parámetros:
        -----------
        filepath : str
            Ruta donde guardar el modelo
        """
        # Crear directorio si no existe
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        # Guardar datos del modelo
        model_data = {
            'cgm_shape': self.cgm_shape,
            'other_features_shape': self.other_features_shape,
            'cgm_weight': self.cgm_weight,
            'other_weight': self.other_weight,
            'state_dim': self.sac_agent.state_dim,
            'action_dim': self.sac_agent.action_dim,
            'action_high': self.sac_agent.action_high.tolist(),
            'action_low': self.sac_agent.action_low.tolist()
        }
        
        with open(filepath + "_wrapper.pkl", 'wb') as f:
            pickle.dump(model_data, f)
        
        # Guardar el agente SAC
        self.sac_agent.save_models(filepath + "_sac")
        
        print(f"Modelo guardado en {filepath}")
    
    def load(self, filepath: str) -> None:
        """
        Carga el modelo SAC desde un archivo.
        
        Parámetros:
        -----------
        filepath : str
            Ruta desde donde cargar el modelo
        """
        
        # Cargar datos del wrapper
        with open(filepath + "_wrapper.pkl", 'rb') as f:
            model_data = pickle.load(f)
            
        self.cgm_shape = model_data['cgm_shape']
        self.other_features_shape = model_data['other_features_shape']
        self.cgm_weight = model_data['cgm_weight']
        self.other_weight = model_data['other_weight']
        
        # Re-compilar funciones de codificación
        self.encode_cgm = jax.jit(self._create_encoder_fn(self.cgm_weight))
        self.encode_other = jax.jit(self._create_encoder_fn(self.other_weight))
        
        # Cargar el agente SAC
        self.sac_agent.load_models(filepath + "_sac")
        
        print(f"Modelo cargado desde {filepath}")
    
    def get_config(self) -> Dict:
        """
        Obtiene la configuración del modelo.
        
        Retorna:
        --------
        Dict
            Diccionario con configuración del modelo
        """
        config = {
            'cgm_shape': self.cgm_shape,
            'other_features_shape': self.other_features_shape,
            'state_dim': self.sac_agent.state_dim,
            'action_dim': self.sac_agent.action_dim,
            'action_high': self.sac_agent.action_high.tolist(),
            'action_low': self.sac_agent.action_low.tolist()
        }
        
        # Añadir configuración del agente SAC
        sac_config = {
            'gamma': self.sac_agent.gamma,
            'tau': self.sac_agent.tau,
            'batch_size': self.sac_agent.batch_size,
            'alpha': float(jnp.exp(self.sac_agent.log_alpha))
        }
        
        config['sac_config'] = sac_config
        return config


def create_sac_model(cgm_shape: Tuple[int, ...], other_features_shape: Tuple[int, ...]) -> DRLModelWrapper:
    """
    Crea un modelo basado en SAC (Soft Actor-Critic) para predicción de dosis de insulina.
    
    Parámetros:
    -----------
    cgm_shape : Tuple[int, ...]
        Forma de los datos CGM (batch_size, time_steps, features)
    other_features_shape : Tuple[int, ...]
        Forma de otras características (batch_size, n_features)
        
    Retorna:
    --------
    DRLModelWrapper
        Wrapper de SAC que implementa la interfaz compatible con el sistema
    """
    # Configurar el espacio de estados y acciones
    state_dim = 64  # Dimensión del espacio de estado latente
    action_dim = 1  # Una dimensión para la dosis continua
    
    # Límites de acción para dosis de insulina (0-15 unidades)
    action_high = np.array([15.0])  # Máximo 15 unidades de insulina
    action_low = np.array([0.0])    # Mínimo 0 unidades
    
    # Crear agente SAC
    sac_agent = SAC(
        state_dim=state_dim,
        action_dim=action_dim,
        action_high=action_high,
        action_low=action_low,
        config=SAC_CONFIG,
        seed=CONST_DEFAULT_SEED
    )
    
    # Crear wrapper del agente
    wrapper = SACWrapper(
        sac_agent=sac_agent,
        cgm_shape=cgm_shape,
        other_features_shape=other_features_shape
    )
    
    # Envolver en DRLModelWrapper para compatibilidad completa con la interfaz del sistema
    return DRLModelWrapper(lambda **kwargs: wrapper, algorithm="sac")

def model_creator() -> Callable[[Tuple[int, ...], Tuple[int, ...]], DRLModelWrapper]:
    """
    Retorna una función para crear un modelo SAC compatible con la API del sistema.
    
    Retorna:
    --------
    Callable[[Tuple[int, ...], Tuple[int, ...]], DRLModelWrapper]
        Función para crear el modelo con las formas de entrada especificadas
    """
    return create_sac_model