import os, sys
import flax
import jax
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
import time
from typing import Dict, List, Tuple, Optional, Union, Any, Callable, NamedTuple
from functools import partial
from flax import linen as nn
from flax.training import train_state
import optax

PROJECT_ROOT = os.path.abspath(os.getcwd())
sys.path.append(PROJECT_ROOT) 

from config.models_config import REINFORCE_CONFIG

# Constantes para rutas de figuras y etiquetas comunes
CONST_FIGURES_DIR = "figures/reinforcement_learning/reinforce_mcgp"
CONST_EPISODE = "Episodio"
CONST_REWARD = "Recompensa"
CONST_POLICY_LOSS = "Pérdida de Política"
CONST_ENTROPY = "Entropía"
CONST_BASELINE_LOSS = "Pérdida de Baseline"
CONST_RETURNS = "Retornos"
CONST_STEPS = "Pasos"
CONST_ORIGINAL = "Original"
CONST_SMOOTHED = "Suavizado"

class PolicyNetworkState(NamedTuple):
    """Estado para la red neuronal de política"""
    params: Dict
    rng_key: jnp.ndarray
    

class PolicyNetwork(nn.Module):
    """
    Red neuronal de política para REINFORCE.
    
    Esta red mapea estados a distribuciones de probabilidad sobre acciones.
    """
    state_dim: int
    action_dim: int
    hidden_units: Tuple[int, ...]
    continuous: bool
    
    @nn.compact
    def __call__(self, x: jnp.ndarray, training: bool = False) -> Union[jnp.ndarray, Tuple[jnp.ndarray, jnp.ndarray]]:
        """
        Ejecuta la red para obtener los parámetros de la distribución de política.
        
        Parámetros:
        -----------
        x : jnp.ndarray
            Tensor de estados de entrada
        training : bool, opcional
            Si está en modo entrenamiento (default: False)
        
        Retorna:
        --------
        Union[jnp.ndarray, Tuple[jnp.ndarray, jnp.ndarray]]
            Para espacios discretos: logits
            Para espacios continuos: (mu, log_sigma)
        """
        # Capas ocultas
        for i, units in enumerate(self.hidden_units):
            x = nn.Dense(features=units, name=f'hidden_{i}')(x)
            x = nn.LayerNorm(epsilon=REINFORCE_CONFIG['epsilon'], name=f'ln_{i}')(x)
            x = nn.relu(x)
            if REINFORCE_CONFIG['dropout_rate'] > 0 and training:
                x = nn.Dropout(
                    rate=REINFORCE_CONFIG['dropout_rate'], 
                    deterministic=not training,
                    name=f'dropout_{i}'
                )(x)
        
        # Capa de salida: depende de si el espacio de acciones es continuo o discreto
        if self.continuous:
            # Para espacios continuos: política gaussiana
            mu = nn.Dense(features=self.action_dim, name='mu')(x)
            mu = nn.tanh(mu)
            
            log_sigma = nn.Dense(features=self.action_dim, name='log_sigma')(x)
            # Limitar el rango de log_sigma para estabilidad
            log_sigma = jnp.clip(log_sigma, -20, 2)
            
            return mu, log_sigma
        else:
            # Para espacios discretos: política categórica
            logits = nn.Dense(features=self.action_dim, name='logits')(x)
            return logits


class REINFORCEState(NamedTuple):
    """Estado para el algoritmo REINFORCE"""
    policy_state: train_state.TrainState
    value_state: Optional[train_state.TrainState] = None
    rng_key: jnp.ndarray = None


class REINFORCE:
    """
    Implementación del algoritmo REINFORCE (Monte Carlo Policy Gradient).
    
    REINFORCE utiliza retornos Monte Carlo completos para actualizar la política,
    haciéndolo un algoritmo simple pero efectivo para aprendizaje de políticas.
    """
    
    def __init__(
        self, 
        state_dim: int, 
        action_dim: int,
        continuous: bool = False,
        learning_rate: float = REINFORCE_CONFIG['learning_rate'],
        gamma: float = REINFORCE_CONFIG['gamma'],
        hidden_units: Optional[List[int]] = None,
        baseline: bool = REINFORCE_CONFIG['use_baseline'],
        entropy_coef: float = REINFORCE_CONFIG['entropy_coef'],
        seed: int = 42
    ) -> None:
        """
        Inicializa el agente REINFORCE.
        
        Parámetros:
        -----------
        state_dim : int
            Dimensión del espacio de estados
        action_dim : int
            Dimensión del espacio de acciones
        continuous : bool, opcional
            Si el espacio de acciones es continuo (default: False)
        learning_rate : float, opcional
            Tasa de aprendizaje (default: REINFORCE_CONFIG['learning_rate'])
        gamma : float, opcional
            Factor de descuento (default: REINFORCE_CONFIG['gamma'])
        hidden_units : Optional[List[int]], opcional
            Lista con unidades en capas ocultas (default: None)
        baseline : bool, opcional
            Si usar baseline para reducir varianza (default: REINFORCE_CONFIG['use_baseline'])
        entropy_coef : float, opcional
            Coeficiente para regularización por entropía (default: REINFORCE_CONFIG['entropy_coef'])
        seed : int, opcional
            Semilla para reproducibilidad (default: 42)
        """
        # Configurar semillas para reproducibilidad
        self.rng_key = jax.random.PRNGKey(seed)
        
        # Parámetros básicos
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.continuous = continuous
        self.gamma = gamma
        self.use_baseline = baseline
        self.entropy_coef = entropy_coef
        
        # Valores predeterminados para capas ocultas
        if hidden_units is None:
            self.hidden_units = tuple(REINFORCE_CONFIG['hidden_units'])
        else:
            self.hidden_units = tuple(hidden_units)
        
        # Crear red de política con Flax
        self.policy_model = PolicyNetwork(
            state_dim=state_dim,
            action_dim=action_dim,
            hidden_units=self.hidden_units,
            continuous=continuous
        )
        
        # Inicializar parámetros del modelo
        self.rng_key, init_key = jax.random.split(self.rng_key)
        dummy_input = jnp.ones((1, state_dim))
        policy_params = self.policy_model.init(init_key, dummy_input)
        
        # Crear optimizador
        policy_tx = optax.adam(learning_rate=learning_rate)
        self.policy_state = train_state.TrainState.create(
            apply_fn=self.policy_model.apply,
            params=policy_params,
            tx=policy_tx
        )
        
        # Red de valor (baseline) opcional para reducir varianza
        self.value_state = None
        if self.use_baseline:
            self.value_model = self._create_value_network()
            self.rng_key, init_key = jax.random.split(self.rng_key)
            value_params = self.value_model.init(init_key, dummy_input)
            value_tx = optax.adam(learning_rate=learning_rate)
            self.value_state = train_state.TrainState.create(
                apply_fn=self.value_model.apply,
                params=value_params,
                tx=value_tx
            )
        
        # Estado del agente
        self.state = REINFORCEState(
            policy_state=self.policy_state,
            value_state=self.value_state,
            rng_key=self.rng_key
        )
        
        # Compilar funciones para rendimiento óptimo
        self._get_action_discrete = jax.jit(self._get_action_discrete)
        self._get_action_continuous = jax.jit(self._get_action_continuous)
        self._get_log_prob_discrete = jax.jit(self._get_log_prob_discrete)
        self._get_log_prob_continuous = jax.jit(self._get_log_prob_continuous)
        self._get_entropy_discrete = jax.jit(self._get_entropy_discrete)
        self._get_entropy_continuous = jax.jit(self._get_entropy_continuous)
        self._policy_loss_fn = jax.jit(self._policy_loss_fn)
        if self.use_baseline:
            self._value_loss_fn = jax.jit(self._value_loss_fn)
        
        # Métricas
        self.policy_loss_metric = 0.0
        self.entropy_metric = 0.0
        self.baseline_loss_metric = 0.0
        self.returns_metric = 0.0
    
    def _create_value_network(self) -> nn.Module:
        """
        Crea una red neuronal para estimar el valor de estado (baseline).
        
        Retorna:
        --------
        nn.Module
            Módulo de red de valor
        """
        class ValueNetwork(nn.Module):
            """Red neuronal para estimar valores de estado."""
            hidden_units: Tuple[int, ...]
            
            @nn.compact
            def __call__(self, x: jnp.ndarray, training: bool = False) -> jnp.ndarray:
                # Capas ocultas
                for i, units in enumerate(self.hidden_units):
                    x = nn.Dense(features=units, name=f'value_hidden_{i}')(x)
                    x = nn.LayerNorm(epsilon=REINFORCE_CONFIG['epsilon'], name=f'value_ln_{i}')(x)
                    x = nn.relu(x)
                    if REINFORCE_CONFIG['dropout_rate'] > 0 and training:
                        x = nn.Dropout(
                            rate=REINFORCE_CONFIG['dropout_rate'], 
                            deterministic=not training,
                            name=f'value_dropout_{i}'
                        )(x)
                
                # Capa de salida: un solo valor
                x = nn.Dense(features=1, name='value')(x)
                return x
        
        return ValueNetwork(hidden_units=self.hidden_units)
    
    def _get_action_discrete(
        self, 
        params: Dict, 
        state: jnp.ndarray, 
        rng_key: jnp.ndarray, 
        deterministic: bool
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """
        Obtiene una acción para espacio discreto.
        
        Parámetros:
        -----------
        params : Dict
            Parámetros de la red de política
        state : jnp.ndarray
            Estado actual
        rng_key : jnp.ndarray
            Llave para generación de números aleatorios
        deterministic : bool
            Si seleccionar la acción determinísticamente
            
        Retorna:
        --------
        Tuple[jnp.ndarray, jnp.ndarray]
            Acción seleccionada y nueva llave aleatoria
        """
        logits = self.policy_model.apply(params, state[None])
        
        if deterministic:
            action = jnp.argmax(logits, axis=-1)[0]
        else:
            rng_key, subkey = jax.random.split(rng_key)
            action = jax.random.categorical(subkey, logits)[0]
        
        return action, rng_key
    
    def _get_action_continuous(
        self, 
        params: Dict, 
        state: jnp.ndarray, 
        rng_key: jnp.ndarray, 
        deterministic: bool
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """
        Obtiene una acción para espacio continuo.
        
        Parámetros:
        -----------
        params : Dict
            Parámetros de la red de política
        state : jnp.ndarray
            Estado actual
        rng_key : jnp.ndarray
            Llave para generación de números aleatorios
        deterministic : bool
            Si seleccionar la acción determinísticamente
            
        Retorna:
        --------
        Tuple[jnp.ndarray, jnp.ndarray]
            Acción seleccionada y nueva llave aleatoria
        """
        mu, log_sigma = self.policy_model.apply(params, state[None])
        
        if deterministic:
            action = mu[0]
        else:
            rng_key, subkey = jax.random.split(rng_key)
            sigma = jnp.exp(log_sigma)
            noise = jax.random.normal(subkey, shape=mu.shape)
            action = mu + sigma * noise
            action = action[0]
        
        return action, rng_key
    
    def get_action(
        self, 
        state: np.ndarray, 
        deterministic: bool = False
    ) -> Union[np.ndarray, int]:
        """
        Obtiene una acción según la política actual.
        
        Parámetros:
        -----------
        state : np.ndarray
            Estado actual
        deterministic : bool, opcional
            Si se usa la acción determinística o se muestrea (default: False)

        Retorna:
        --------
        Union[np.ndarray, int]
            La acción seleccionada
        """
        state = jnp.asarray(state)
        
        # Seleccionar acción según el tipo de espacio
        if self.continuous:
            action, new_key = self._get_action_continuous(
                self.state.policy_state.params, 
                state, 
                self.state.rng_key, 
                deterministic
            )
        else:
            action, new_key = self._get_action_discrete(
                self.state.policy_state.params, 
                state, 
                self.state.rng_key, 
                deterministic
            )
        
        # Actualizar llave de aleatoriedad
        self.state = self.state._replace(rng_key=new_key)
        
        return np.asarray(action)
    
    def _get_log_prob_discrete(
        self, 
        params: Dict, 
        states: jnp.ndarray, 
        actions: jnp.ndarray
    ) -> jnp.ndarray:
        """
        Calcula log-probabilidades para acciones discretas.
        
        Parámetros:
        -----------
        params : Dict
            Parámetros de la red de política
        states : jnp.ndarray
            Estados observados
        actions : jnp.ndarray
            Acciones tomadas
            
        Retorna:
        --------
        jnp.ndarray
            Log-probabilidades de las acciones
        """
        logits = self.policy_model.apply(params, states)
        action_masks = jax.nn.one_hot(actions, self.action_dim)
        log_probs = jnp.sum(action_masks * jax.nn.log_softmax(logits), axis=1)
        return log_probs
    
    def _get_log_prob_continuous(
        self, 
        params: Dict, 
        states: jnp.ndarray, 
        actions: jnp.ndarray
    ) -> jnp.ndarray:
        """
        Calcula log-probabilidades para acciones continuas.
        
        Parámetros:
        -----------
        params : Dict
            Parámetros de la red de política
        states : jnp.ndarray
            Estados observados
        actions : jnp.ndarray
            Acciones tomadas
            
        Retorna:
        --------
        jnp.ndarray
            Log-probabilidades de las acciones
        """
        mu, log_sigma = self.policy_model.apply(params, states)
        sigma = jnp.exp(log_sigma)
        
        # Log-prob para distribución gaussiana
        log_probs = -0.5 * (
            jnp.sum(
                jnp.square((actions - mu) / sigma) + 
                2 * log_sigma + 
                jnp.log(2.0 * np.pi),
                axis=1
            )
        )
        return log_probs
    
    def get_log_prob(self, states: jnp.ndarray, actions: jnp.ndarray) -> jnp.ndarray:
        """
        Calcula el logaritmo de la probabilidad de acciones dadas.
        
        Parámetros:
        -----------
        states : jnp.ndarray
            Estados observados
        actions : jnp.ndarray
            Acciones tomadas
            
        Retorna:
        --------
        jnp.ndarray
            Log-probabilidades de las acciones
        """
        # Seleccionar función según el tipo de espacio
        if self.continuous:
            return self._get_log_prob_continuous(self.state.policy_state.params, states, actions)
        else:
            return self._get_log_prob_discrete(self.state.policy_state.params, states, actions)
    
    def _get_entropy_discrete(self, params: Dict, states: jnp.ndarray) -> jnp.ndarray:
        """
        Calcula la entropía para la política discreta.
        
        Parámetros:
        -----------
        params : Dict
            Parámetros de la red de política
        states : jnp.ndarray
            Estados observados
            
        Retorna:
        --------
        jnp.ndarray
            Entropía de la política
        """
        logits = self.policy_model.apply(params, states)
        probs = jax.nn.softmax(logits)
        entropy = -jnp.sum(probs * jnp.log(probs + 1e-10), axis=1)
        return entropy
    
    def _get_entropy_continuous(self, params: Dict, states: jnp.ndarray) -> jnp.ndarray:
        """
        Calcula la entropía para la política continua.
        
        Parámetros:
        -----------
        params : Dict
            Parámetros de la red de política
        states : jnp.ndarray
            Estados observados
            
        Retorna:
        --------
        jnp.ndarray
            Entropía de la política
        """
        _, log_sigma = self.policy_model.apply(params, states)
        # Entropía de distribución gaussiana: 0.5 * log(2*pi*e*sigma^2)
        entropy = jnp.sum(
            0.5 * jnp.log(2.0 * np.pi * np.e * jnp.exp(2 * log_sigma)),
            axis=1
        )
        return entropy
    
    def get_entropy(self, states: jnp.ndarray) -> jnp.ndarray:
        """
        Calcula la entropía de la política para los estados dados.
        
        Parámetros:
        -----------
        states : jnp.ndarray
            Estados para evaluar
            
        Retorna:
        --------
        jnp.ndarray
            Entropía de la política
        """
        # Seleccionar función según el tipo de espacio
        if self.continuous:
            return self._get_entropy_continuous(self.state.policy_state.params, states)
        else:
            return self._get_entropy_discrete(self.state.policy_state.params, states)
    
    def _policy_loss_fn(
        self, 
        params: Dict, 
        states: jnp.ndarray, 
        actions: jnp.ndarray, 
        returns: jnp.ndarray, 
        values: Optional[jnp.ndarray] = None
    ) -> Tuple[jnp.ndarray, Tuple[jnp.ndarray, jnp.ndarray]]:
        """
        Función de pérdida para la red de política.
        
        Parámetros:
        -----------
        params : Dict
            Parámetros de la red de política
        states : jnp.ndarray
            Estados observados
        actions : jnp.ndarray
            Acciones tomadas
        returns : jnp.ndarray
            Retornos calculados
        values : Optional[jnp.ndarray], opcional
            Valores estimados del baseline (default: None)
            
        Retorna:
        --------
        Tuple[jnp.ndarray, Tuple[jnp.ndarray, jnp.ndarray]]
            Pérdida y métricas auxiliares (pérdida de política, entropía)
        """
        # Calcular log-probabilidades de acciones tomadas
        if self.continuous:
            log_probs = self._get_log_prob_continuous(params, states, actions)
            entropy = self._get_entropy_continuous(params, states)
        else:
            log_probs = self._get_log_prob_discrete(params, states, actions)
            entropy = self._get_entropy_discrete(params, states)
        
        # Si se usa baseline, restar el valor predicho de los retornos
        if values is not None:
            advantages = returns - values
        else:
            advantages = returns
        
        # Calcular pérdida de política (negativa porque queremos maximizar)
        policy_loss = -jnp.mean(log_probs * advantages)
        
        # Calcular entropía media
        mean_entropy = jnp.mean(entropy)
        
        # Pérdida total con regularización de entropía
        loss = policy_loss - self.entropy_coef * mean_entropy
        
        return loss, (policy_loss, mean_entropy)
    
    def _value_loss_fn(self, params: Dict, states: jnp.ndarray, returns: jnp.ndarray) -> jnp.ndarray:
        """
        Función de pérdida para la red de valor.
        
        Parámetros:
        -----------
        params : Dict
            Parámetros de la red de valor
        states : jnp.ndarray
            Estados observados
        returns : jnp.ndarray
            Retornos calculados
            
        Retorna:
        --------
        jnp.ndarray
            Pérdida de la red de valor
        """
        values = jnp.squeeze(self.value_model.apply(params, states))
        loss = jnp.mean(jnp.square(values - returns))
        return loss
    
    def _update_policy(
        self, 
        policy_state: train_state.TrainState, 
        states: jnp.ndarray, 
        actions: jnp.ndarray, 
        returns: jnp.ndarray, 
        values: Optional[jnp.ndarray] = None
    ) -> Tuple[train_state.TrainState, Tuple[float, float]]:
        """
        Actualiza los parámetros de la red de política.
        
        Parámetros:
        -----------
        policy_state : train_state.TrainState
            Estado actual de la red de política
        states : jnp.ndarray
            Estados observados
        actions : jnp.ndarray
            Acciones tomadas
        returns : jnp.ndarray
            Retornos calculados
        values : Optional[jnp.ndarray], opcional
            Valores de estado si se usa baseline (default: None)
            
        Retorna:
        --------
        Tuple[train_state.TrainState, Tuple[float, float]]
            Nuevo estado de la red y métricas (pérdida de política, entropía)
        """
        # Calcular gradientes y actualizar red
        grad_fn = jax.value_and_grad(self._policy_loss_fn, has_aux=True)
        (_, (policy_loss, entropy)), grads = grad_fn(
            policy_state.params, states, actions, returns, values
        )
        
        # Actualizar parámetros
        new_policy_state = policy_state.apply_gradients(grads=grads)
        
        return new_policy_state, (policy_loss, entropy)
    
    def _update_value(
        self, 
        value_state: train_state.TrainState, 
        states: jnp.ndarray, 
        returns: jnp.ndarray
    ) -> Tuple[train_state.TrainState, float]:
        """
        Actualiza los parámetros de la red de valor.
        
        Parámetros:
        -----------
        value_state : train_state.TrainState
            Estado actual de la red de valor
        states : jnp.ndarray
            Estados observados
        returns : jnp.ndarray
            Retornos calculados
            
        Retorna:
        --------
        Tuple[train_state.TrainState, float]
            Nuevo estado de la red y pérdida de valor
        """
        # Calcular gradientes y actualizar red
        grad_fn = jax.value_and_grad(self._value_loss_fn)
        value_loss, grads = grad_fn(value_state.params, states, returns)
        
        # Actualizar parámetros
        new_value_state = value_state.apply_gradients(grads=grads)
        
        return new_value_state, value_loss
    
    def compute_returns(self, rewards: List[float]) -> np.ndarray:
        """
        Calcula los retornos descontados para cada paso de tiempo.
        
        Parámetros:
        -----------
        rewards : List[float]
            Lista de recompensas recibidas
            
        Retorna:
        --------
        np.ndarray
            Array de retornos descontados
        """
        returns = np.zeros_like(rewards, dtype=np.float32)
        future_return = 0.0
        
        # Calcular retornos desde el final del episodio
        for t in reversed(range(len(rewards))):
            future_return = rewards[t] + self.gamma * future_return
            returns[t] = future_return
            
        # Normalizar retornos para estabilidad
        if len(returns) > 1:
            returns = (returns - np.mean(returns)) / (np.std(returns) + 1e-8)
        
        return returns
    
    def _run_episode(
        self, 
        env: Any, 
        render: bool = False
    ) -> Tuple[List[np.ndarray], List[Union[int, np.ndarray]], List[float], float, int]:
        """
        Ejecuta un episodio completo.
        
        Parámetros:
        -----------
        env : Any
            Entorno a interactuar
        render : bool, opcional
            Si renderizar o no el entorno (default: False)
            
        Retorna:
        --------
        Tuple[List[np.ndarray], List[Union[int, np.ndarray]], List[float], float, int]
            Estados, acciones, recompensas, recompensa total, longitud del episodio
        """
        state, _ = env.reset()
        done = False
        states, actions, rewards = [], [], []
        total_reward = 0
        step = 0
        
        while not done and step < REINFORCE_CONFIG.get('max_steps', 1000):
            if render:
                env.render()
            
            # Convertir estado a array de JAX si es necesario
            if not isinstance(state, np.ndarray):
                state = np.array(state, dtype=np.float32)
            
            # Tomar acción según la política actual
            action = self.get_action(state, deterministic=False)
            
            # Ejecutar acción en el entorno
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            
            # Almacenar transición
            states.append(state)
            actions.append(action)
            rewards.append(reward)
            
            # Actualizar
            total_reward += reward
            state = next_state
            step += 1
        
        return states, actions, rewards, total_reward, step
    
    def _update_networks(
        self, 
        states: List[np.ndarray], 
        actions: List[Union[int, np.ndarray]], 
        rewards: List[float]
    ) -> Tuple[float, float]:
        """
        Actualiza las redes de política y valor.
        
        Parámetros:
        -----------
        states : List[np.ndarray]
            Lista de estados
        actions : List[Union[int, np.ndarray]]
            Lista de acciones
        rewards : List[float]
            Lista de recompensas
            
        Retorna:
        --------
        Tuple[float, float]
            Tupla con (pérdida_política, entropía)
        """
        # Calcular retornos
        returns = self.compute_returns(rewards)
        
        # Convertir a arrays de JAX
        states = jnp.asarray(states, dtype=jnp.float32)
        if self.continuous:
            actions = jnp.asarray(actions, dtype=jnp.float32)
        else:
            actions = jnp.asarray(actions, dtype=jnp.int32)
        returns = jnp.asarray(returns, dtype=jnp.float32)
        
        # Calcular valores de baseline si se usa
        values = None
        if self.use_baseline:
            values = jnp.squeeze(self.value_model.apply(self.state.value_state.params, states))
            # Actualizar red de valor
            new_value_state, value_loss = self._update_value(
                self.state.value_state, 
                states, 
                returns
            )
            self.baseline_loss_metric = float(value_loss)
        
        # Actualizar red de política
        new_policy_state, (policy_loss, entropy) = self._update_policy(
            self.state.policy_state, 
            states, 
            actions, 
            returns, 
            values
        )
        
        # Actualizar estado del agente
        self.state = REINFORCEState(
            policy_state=new_policy_state,
            value_state=new_value_state if self.use_baseline else None,
            rng_key=self.state.rng_key
        )
        
        # Actualizar métricas
        self.policy_loss_metric = float(policy_loss)
        self.entropy_metric = float(entropy)
        self.returns_metric = float(jnp.mean(returns))
        
        return float(policy_loss), float(entropy)
    
    def _update_history(
        self, 
        history: Dict[str, List[float]], 
        episode_reward: float, 
        episode_length: int
    ) -> None:
        """
        Actualiza la historia de entrenamiento con las métricas actuales.
        
        Parámetros:
        -----------
        history : Dict[str, List[float]]
            Diccionario de historia
        episode_reward : float
            Recompensa total del episodio
        episode_length : int
            Longitud del episodio
        """
        history['episode_rewards'].append(episode_reward)
        history['episode_lengths'].append(episode_length)
        history['policy_losses'].append(self.policy_loss_metric)
        if self.use_baseline:
            history['baseline_losses'].append(self.baseline_loss_metric)
        history['entropies'].append(self.entropy_metric)
        history['mean_returns'].append(self.returns_metric)
    
    def train(
        self, 
        env: Any, 
        episodes: Optional[int] = None, 
        render: bool = False
    ) -> Dict[str, List[float]]:
        """
        Entrena el agente REINFORCE en el entorno dado.
        
        Parámetros:
        -----------
        env : Any
            Entorno de OpenAI Gym o compatible
        episodes : Optional[int], opcional
            Número de episodios de entrenamiento (default: None)
        render : bool, opcional
            Si renderizar el entorno durante entrenamiento (default: False)
            
        Retorna:
        --------
        Dict[str, List[float]]
            Historia de entrenamiento
        """
        if episodes is None:
            episodes = REINFORCE_CONFIG['episodes']
        
        history = {
            'episode_rewards': [],
            'episode_lengths': [],
            'policy_losses': [],
            'baseline_losses': [] if self.use_baseline else None,
            'entropies': [],
            'mean_returns': []
        }
        
        start_time = time.time()
        
        for episode in range(episodes):
            # Ejecutar episodio
            states, actions, rewards, episode_reward, episode_length = self._run_episode(env, render)
            
            # Actualizar redes
            policy_loss, entropy = self._update_networks(states, actions, rewards)
            
            # Actualizar historia
            self._update_history(history, episode_reward, episode_length)
            
            # Mostrar progreso periódicamente
            if (episode + 1) % REINFORCE_CONFIG['log_interval'] == 0:
                elapsed_time = time.time() - start_time
                avg_reward = np.mean(history['episode_rewards'][-REINFORCE_CONFIG['log_interval']:])
                print(f"Episodio {episode+1}/{episodes} - "
                      f"Recompensa Media: {avg_reward:.2f}, "
                      f"Pérdida Política: {policy_loss:.4f}, "
                      f"Entropía: {entropy:.4f}, "
                      f"Tiempo: {elapsed_time:.2f}s")
                start_time = time.time()
        
        return history
    
    def evaluate(
        self, 
        env: Any, 
        episodes: int = 10, 
        render: bool = False
    ) -> float:
        """
        Evalúa el agente REINFORCE con su política actual.
        
        Parámetros:
        -----------
        env : Any
            Entorno de OpenAI Gym o compatible
        episodes : int, opcional
            Número de episodios para evaluación (default: 10)
        render : bool, opcional
            Si renderizar el entorno durante evaluación (default: False)
            
        Retorna:
        --------
        float
            Recompensa media obtenida
        """
        rewards = []
        lengths = []
        
        for episode in range(episodes):
            state, _ = env.reset()
            done = False
            episode_reward = 0
            steps = 0
            
            while not done:
                if render:
                    env.render()
                
                # Usar política determinística para evaluación
                action = self.get_action(state, deterministic=True)
                
                # Dar paso en el entorno
                next_state, reward, terminated, truncated, _ = env.step(action)
                done = terminated or truncated
                
                # Actualizar estado y contadores
                state = next_state
                episode_reward += reward
                steps += 1
            
            rewards.append(episode_reward)
            lengths.append(steps)
            
            print(f"Episodio Evaluación {episode+1}: Recompensa = {episode_reward:.2f}, Pasos = {steps}")
        
        avg_reward = np.mean(rewards)
        avg_length = np.mean(lengths)
        
        print(f"Evaluación Completada - Recompensa Media: {avg_reward:.2f}, Pasos Medios: {avg_length:.2f}")
        
        return avg_reward
    
    def save(self, policy_path: str, baseline_path: Optional[str] = None) -> None:
        """
        Guarda los modelos del agente.
        
        Parámetros:
        -----------
        policy_path : str
            Ruta para guardar la política
        baseline_path : Optional[str], opcional
            Ruta para guardar el baseline (default: None)
        """
        # Guardar política
        with open(policy_path, 'wb') as f:
            f.write(flax.serialization.to_bytes(self.state.policy_state.params))
        
        # Guardar baseline si existe
        if self.use_baseline and baseline_path:
            with open(baseline_path, 'wb') as f:
                f.write(flax.serialization.to_bytes(self.state.value_state.params))
        
        print(f"Modelo guardado en {policy_path}")
    
    def load(self, policy_path: str, baseline_path: Optional[str] = None) -> None:
        """
        Carga los modelos del agente.
        
        Parámetros:
        -----------
        policy_path : str
            Ruta para cargar la política
        baseline_path : Optional[str], opcional
            Ruta para cargar el baseline (default: None)
        """
        # Cargar política
        with open(policy_path, 'rb') as f:
            policy_params = flax.serialization.from_bytes(
                self.state.policy_state.params,
                f.read()
            )
        
        # Actualizar estado de la política
        policy_state = self.state.policy_state.replace(params=policy_params)
        
        # Cargar baseline si existe
        value_state = self.state.value_state
        if self.use_baseline and baseline_path:
            with open(baseline_path, 'rb') as f:
                value_params = flax.serialization.from_bytes(
                    self.state.value_state.params,
                    f.read()
                )
            value_state = self.state.value_state.replace(params=value_params)
        
        # Actualizar estado del agente
        self.state = REINFORCEState(
            policy_state=policy_state,
            value_state=value_state,
            rng_key=self.state.rng_key
        )
        
        print(f"Modelo cargado desde {policy_path}")
    
    def _get_default_history(self) -> Dict[str, List[float]]:
        """
        Crea un historial predeterminado con las métricas actuales.
        
        Retorna:
        --------
        Dict[str, List[float]]
            Historial predeterminado
        """
        history = {
            'episode_rewards': [],
            'episode_lengths': [],
            'policy_losses': [self.policy_loss_metric],
            'entropy': [self.entropy_metric],
            'returns': [self.returns_metric]
        }
        if self.use_baseline:
            history['baseline_losses'] = [self.baseline_loss_metric]
        return history
    
    def _smooth_data(self, data: List[float], window_size: int) -> np.ndarray:
        """
        Suaviza datos usando una ventana deslizante.
        
        Parámetros:
        -----------
        data : List[float]
            Datos a suavizar
        window_size : int
            Tamaño de la ventana de suavizado
            
        Retorna:
        --------
        np.ndarray
            Datos suavizados
        """
        if len(data) < window_size:
            return np.array(data)
        kernel = np.ones(window_size) / window_size
        return np.convolve(np.array(data), kernel, mode='valid')
    
    def _plot_metric(
        self, 
        ax: plt.Axes, 
        data: List[float], 
        smoothing_window: int, 
        color: str,
        title: str, 
        xlabel: str, 
        ylabel: str
    ) -> None:
        """
        Dibuja un gráfico de métrica con datos originales y suavizados.
        
        Parámetros:
        -----------
        ax : plt.Axes
            Ejes donde dibujar
        data : List[float]
            Datos a graficar
        smoothing_window : int
            Tamaño de ventana para suavizado
        color : str
            Color del gráfico
        title : str
            Título del gráfico
        xlabel : str
            Etiqueta del eje X
        ylabel : str
            Etiqueta del eje Y
        """
        ax.plot(data, alpha=0.3, color=color, label=CONST_ORIGINAL)
        
        if len(data) > smoothing_window:
            smoothed = self._smooth_data(data, smoothing_window)
            ax.plot(
                range(smoothing_window-1, len(data)),
                smoothed,
                color=color,
                label=f'{CONST_SMOOTHED} (ventana={smoothing_window})'
            )
            
        ax.set_title(title)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.legend()
        ax.grid(alpha=0.3)
    
    def visualize_training(
        self, 
        history: Optional[Dict[str, List[float]]] = None, 
        smoothing_window: Optional[int] = None
    ) -> None:
        """
        Visualiza el historial de entrenamiento.
        
        Parámetros:
        -----------
        history : Optional[Dict[str, List[float]]], opcional
            Historial de entrenamiento (default: None, usa el historial interno)
        smoothing_window : Optional[int], opcional
            Tamaño de ventana para suavizado (default: None, usa valor configurado)
        """
        # Usar historial proporcionado o interno
        history = history or self._get_default_history()
        
        # Determinar tamaño de ventana de suavizado
        smoothing_window = smoothing_window or REINFORCE_CONFIG.get('smoothing_window', 10)
        
        # Crear directorio para figuras si no existe
        os.makedirs(CONST_FIGURES_DIR, exist_ok=True)
        
        # Comprobar métricas disponibles
        metrics_config = [
            ('episode_rewards', 'blue', f'{CONST_REWARD}s por {CONST_EPISODE}', CONST_EPISODE, CONST_REWARD),
            ('episode_lengths', 'green', f'Longitud de {CONST_EPISODE}s', CONST_EPISODE, CONST_STEPS),
            ('policy_losses', 'red', CONST_POLICY_LOSS, CONST_EPISODE, CONST_POLICY_LOSS),
            ('entropy', 'purple', CONST_ENTROPY, CONST_EPISODE, CONST_ENTROPY),
            ('baseline_losses', 'orange', CONST_BASELINE_LOSS, CONST_EPISODE, CONST_BASELINE_LOSS),
            ('returns', 'brown', CONST_RETURNS, CONST_EPISODE, CONST_RETURNS)
        ]
        
        # Filtrar métricas disponibles en el historial
        available_metrics = [(key, color, title, xlabel, ylabel) 
                             for key, color, title, xlabel, ylabel in metrics_config
                             if key in history and history[key]]
        
        n_plots = len(available_metrics)
        
        # Crear figura si hay datos para mostrar
        if n_plots > 0:
            _, axs = plt.subplots(n_plots, 1, figsize=(12, 3 * n_plots))
            axs = [axs] if n_plots == 1 else axs
            
            # Graficar cada métrica disponible
            for i, (key, color, title, xlabel, ylabel) in enumerate(available_metrics):
                self._plot_metric(
                    axs[i], history[key], smoothing_window, 
                    color, title, xlabel, ylabel
                )
            
            plt.tight_layout()
            
            # Guardar figura
            plt.savefig(os.path.join(CONST_FIGURES_DIR, "entrenamiento_resumen.png"), dpi=300)
            plt.show()

class REINFORCEWrapper:
    """
    Wrapper para hacer que REINFORCE sea compatible con la interfaz de modelos de aprendizaje profundo.
    """
    def __init__(
        self,
        reinforce_agent: REINFORCE,
        cgm_shape: Tuple[int, ...],
        other_features_shape: Tuple[int, ...],
    ) -> None:
        """
        Inicializa el wrapper para REINFORCE.
        
        Parámetros:
        -----------
        reinforce_agent : REINFORCE
            Agente REINFORCE a utilizar
        cgm_shape : Tuple[int, ...]
            Forma de entrada para datos CGM
        other_features_shape : Tuple[int, ...]
            Forma de entrada para otras características
        """
        self.reinforce_agent = reinforce_agent
        self.cgm_shape = cgm_shape
        self.other_features_shape = other_features_shape
        
        # Inicializar clave para generación de números aleatorios
        self.key = jax.random.PRNGKey(42)
        self.key, self.encoder_key = jax.random.split(self.key)
        
        # Configurar funciones de codificación para entradas
        self._setup_encoders()
        
        # Historial de entrenamiento
        self.history = {
            'loss': [], 
            'val_loss': [],
            'policy_loss': [],
            'episode_rewards': []
        }
    
    def _setup_encoders(self) -> None:
        """
        Configura las funciones de codificación para procesar las entradas.
        """
        # Calcular dimensiones de características aplanadas
        cgm_dim = np.prod(self.cgm_shape[1:])
        other_dim = np.prod(self.other_features_shape[1:])
        
        # Inicializar matrices de transformación
        self.key, key_cgm, key_other = jax.random.split(self.key, 3)
        
        # Crear matrices de proyección para entradas
        self.cgm_weight = jax.random.normal(key_cgm, (cgm_dim, self.reinforce_agent.state_dim // 2))
        self.other_weight = jax.random.normal(key_other, (other_dim, self.reinforce_agent.state_dim // 2))
        
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
    
    def __call__(self, inputs: List[jnp.ndarray]) -> jnp.ndarray:
        """
        Implementa la interfaz de llamada para predicción.
        
        Parámetros:
        -----------
        inputs : List[jnp.ndarray]
            Lista con [cgm_data, other_features]
            
        Retorna:
        --------
        jnp.ndarray
            Predicciones de dosis de insulina
        """
        return self.predict(inputs)
    
    def predict(self, inputs: List[jnp.ndarray]) -> jnp.ndarray:
        """
        Realiza predicciones con el modelo REINFORCE.
        
        Parámetros:
        -----------
        inputs : List[jnp.ndarray]
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
        
        # Obtener acciones usando el agente REINFORCE (modo determinístico para predicciones)
        batch_size = states.shape[0]
        actions = np.zeros((batch_size, 1))
        
        for i in range(batch_size):
            state = np.array(states[i])
            action = self.reinforce_agent.get_action(state, deterministic=True)
            
            # Convertir a dosis de insulina (escalar según el tipo de acción)
            if self.reinforce_agent.continuous:
                # Para acciones continuas, escalar de [-1,1] a [0,15]
                action_scaled = (action[0] + 1.0) * 7.5
            else:
                # Para acciones discretas, convertir índice a valor
                action_scaled = action / (self.reinforce_agent.action_dim - 1) * 15.0
                
            actions[i, 0] = action_scaled
            
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
        Entrena el modelo REINFORCE en los datos proporcionados.
        
        Parámetros:
        -----------
        x : List[jnp.ndarray]
            Lista con [cgm_data, other_features]
        y : jnp.ndarray
            Etiquetas (dosis objetivo)
        validation_data : Optional[Tuple], opcional
            Datos de validación (default: None)
        epochs : int, opcional
            Número de épocas (default: 1)
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
            print("Entrenando modelo REINFORCE...")
            
        # Crear entorno simulado para RL a partir de los datos
        env = self._create_training_environment(x[0], x[1], y)
        
        # Entrenar el agente REINFORCE
        reinforce_history = self.reinforce_agent.train(
            env=env,
            episodes=epochs,
            render=False
        )
        
        # Actualizar historial con métricas del entrenamiento
        self.history['episode_rewards'] = reinforce_history['episode_rewards']
        self.history['policy_loss'] = reinforce_history['policy_losses']
        
        if self.reinforce_agent.use_baseline:
            self.history['baseline_loss'] = reinforce_history['baseline_losses']
        
        self.history['entropy'] = reinforce_history.get('entropies', [])
        
        # Calcular pérdida en datos de entrenamiento
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
        # Crear entorno personalizado para REINFORCE
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
                    shape=(model_wrapper.reinforce_agent.state_dim,),
                    low=np.full((model_wrapper.reinforce_agent.state_dim,), -10.0),
                    high=np.full((model_wrapper.reinforce_agent.state_dim,), 10.0)
                )
                
                if model_wrapper.reinforce_agent.continuous:
                    self.action_space = SimpleNamespace(
                        shape=(1,),
                        low=np.array([-1.0]),
                        high=np.array([1.0]),
                        sample=self._sample_continuous_action
                    )
                else:
                    self.action_space = SimpleNamespace(
                        n=model_wrapper.reinforce_agent.action_dim,
                        sample=lambda: self.rng.integers(0, model_wrapper.reinforce_agent.action_dim)
                    )
            
            def _sample_continuous_action(self):
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
                # Convertir acción a dosis según tipo de espacio de acción
                if hasattr(self.action_space, 'shape'):  # Acción continua
                    dose = (action[0] + 1.0) * 7.5  # Escalar de [-1,1] a [0,15]
                else:  # Acción discreta
                    dose = action / (self.model.reinforce_agent.action_dim - 1) * 15.0
                
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
        
        from types import SimpleNamespace
        return InsulinDosingEnv(cgm_data, other_features, targets, self)
    
    def save(self, filepath: str) -> None:
        """
        Guarda el modelo en un archivo.
        
        Parámetros:
        -----------
        filepath : str
            Ruta donde guardar el modelo
        """
        # Crear directorio si no existe
        import os
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        # Guardar el agente REINFORCE
        policy_path = f"{filepath}_policy.h5"
        baseline_path = None
        if self.reinforce_agent.use_baseline:
            baseline_path = f"{filepath}_baseline.h5"
        self.reinforce_agent.save(policy_path, baseline_path)
        
        # Guardar datos adicionales del wrapper
        import pickle
        wrapper_data = {
            'cgm_shape': self.cgm_shape,
            'other_features_shape': self.other_features_shape,
            'cgm_weight': self.cgm_weight,
            'other_weight': self.other_weight,
            'state_dim': self.reinforce_agent.state_dim,
            'action_dim': self.reinforce_agent.action_dim,
            'continuous': self.reinforce_agent.continuous
        }
        
        with open(f"{filepath}_wrapper.pkl", 'wb') as f:
            pickle.dump(wrapper_data, f)
        
        print(f"Modelo guardado en {filepath}")
    
    def load(self, filepath: str) -> None:
        """
        Carga el modelo desde un archivo.
        
        Parámetros:
        -----------
        filepath : str
            Ruta desde donde cargar el modelo
        """
        # Cargar el agente REINFORCE
        policy_path = f"{filepath}_policy.h5"
        baseline_path = None
        if self.reinforce_agent.use_baseline:
            baseline_path = f"{filepath}_baseline.h5"
        self.reinforce_agent.load(policy_path, baseline_path)
        
        # Cargar datos adicionales del wrapper
        import pickle
        with open(f"{filepath}_wrapper.pkl", 'rb') as f:
            wrapper_data = pickle.load(f)
        
        self.cgm_shape = wrapper_data['cgm_shape']
        self.other_features_shape = wrapper_data['other_features_shape']
        self.cgm_weight = wrapper_data['cgm_weight']
        self.other_weight = wrapper_data['other_weight']
        
        # Recompilar funciones de codificación
        self.encode_cgm = jax.jit(self._create_encoder_fn(self.cgm_weight))
        self.encode_other = jax.jit(self._create_encoder_fn(self.other_weight))
        
        print(f"Modelo cargado desde {filepath}")
    
    def get_config(self) -> Dict:
        """
        Obtiene la configuración del modelo.
        
        Retorna:
        --------
        Dict
            Diccionario con configuración del modelo
        """
        return {
            'cgm_shape': self.cgm_shape,
            'other_features_shape': self.other_features_shape,
            'state_dim': self.reinforce_agent.state_dim,
            'action_dim': self.reinforce_agent.action_dim,
            'continuous': self.reinforce_agent.continuous,
            'gamma': self.reinforce_agent.gamma,
            'entropy_coef': self.reinforce_agent.entropy_coef,
            'use_baseline': self.reinforce_agent.use_baseline
        }


def create_reinforce_mcgp_model(cgm_shape: Tuple[int, ...], other_features_shape: Tuple[int, ...]) -> REINFORCEWrapper:
    """
    Crea un modelo basado en REINFORCE (Monte Carlo Policy Gradient) para predicción de dosis de insulina.
    
    Parámetros:
    -----------
    cgm_shape : Tuple[int, ...]
        Forma de los datos CGM (batch_size, time_steps, features)
    other_features_shape : Tuple[int, ...]
        Forma de otras características (batch_size, n_features)
        
    Retorna:
    --------
    REINFORCEWrapper
        Wrapper de REINFORCE que implementa la interfaz compatible con modelos de aprendizaje profundo
    """
    # Configurar el espacio de estados y acciones
    state_dim = 64  # Dimensión del espacio de estado latente
    action_dim = 1  # Una dimensión para dosis continua
    continuous = True  # Usar espacio de acción continuo
    
    # Crear agente REINFORCE
    reinforce_agent = REINFORCE(
        state_dim=state_dim,
        action_dim=action_dim,
        continuous=continuous,
        learning_rate=REINFORCE_CONFIG['learning_rate'],
        gamma=REINFORCE_CONFIG['gamma'],
        hidden_units=REINFORCE_CONFIG['hidden_units'],
        baseline=REINFORCE_CONFIG['use_baseline'],
        entropy_coef=REINFORCE_CONFIG['entropy_coef'],
        seed=42
    )
    
    # Crear y devolver wrapper
    return REINFORCEWrapper(reinforce_agent, cgm_shape, other_features_shape)

def model_creator() -> Callable[[Tuple[int, ...], Tuple[int, ...]], REINFORCEWrapper]:
    """
    Retorna una función para crear un modelo REINFORCE compatible con la API del sistema.
    
    Retorna:
    --------
    Callable[[Tuple[int, ...], Tuple[int, ...]], REINFORCEWrapper]
        Función para crear el modelo con las formas de entrada especificadas
    """
    return create_reinforce_mcgp_model