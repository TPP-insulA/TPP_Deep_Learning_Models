import os
import sys
import numpy as np
import jax
import jax.numpy as jnp
from jax import random, grad, jit, value_and_grad
import optax
import flax
import flax.linen as nn
from flax.training import train_state
from typing import Dict, List, Tuple, Any, Optional, Union, Callable, NamedTuple
from functools import partial
import time
import matplotlib.pyplot as plt

PROJECT_ROOT = os.path.abspath(os.getcwd())
sys.path.append(PROJECT_ROOT) 

from config.models_config import REINFORCE_CONFIG
from custom.rl_model_wrapper import RLModelWrapperJAX
from custom.printer import print_debug, print_warning

# Constantes para rutas y mensajes
CONST_FIGURES_DIR = os.path.join(PROJECT_ROOT, "figures", "jax", "reinforce_mcpg")
CONST_EPISODE = "Episodio"
CONST_REWARD = "Recompensa"
CONST_POLICY_LOSS = "Pérdida de Política"
CONST_BASELINE_LOSS = "Pérdida de Línea Base"
CONST_ENTROPY = "Entropía"
CONST_ELAPSED_TIME = "Tiempo Transcurrido"
CONST_RETURNS = "Retornos"
CONST_STEPS = "Pasos"
CONST_ORIGINAL = "Original"
CONST_SMOOTHED = "Suavizado"

# Crear directorio para figuras si no existe
os.makedirs(CONST_FIGURES_DIR, exist_ok=True)

class PolicyNetworkState(NamedTuple):
    """Estado para la red neuronal de política"""
    params: Dict
    rng_key: jnp.ndarray
    

class PolicyNetwork(nn.Module):
    """
    Red de política para el algoritmo REINFORCE.
    
    Implementa una red neuronal que parametriza una política estocástica.
    Para acciones continuas, produce la media y la desviación estándar
    de una distribución normal. Para acciones discretas, produce logits
    para una distribución categórica.
    """
    hidden_units: List[int]
    action_dim: int
    continuous: bool = True
    
    @nn.compact
    def __call__(self, x: jnp.ndarray, training: bool = False) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """
        Ejecuta la red de política en la entrada dada.
        
        Parámetros:
        -----------
        x : jnp.ndarray
            Observación o estado de entrada
        training : bool, opcional
            Indica si está en modo entrenamiento (default: False)
            
        Retorna:
        --------
        Tuple[jnp.ndarray, jnp.ndarray]
            Para acciones continuas: (media, log_std)
            Para acciones discretas: (logits, None)
        """
        # Feature extraction
        for units in self.hidden_units:
            x = nn.Dense(units)(x)
            x = nn.relu(x)
            if training:
                x = nn.Dropout(rate=0.1)(x, deterministic=not training)
        
        if self.continuous:
            # Para acciones continuas, predecimos media y log_std
            mean = nn.Dense(self.action_dim)(x)
            log_std = nn.Dense(self.action_dim)(x)
            # Limitar log_std para estabilidad numérica
            log_std = jnp.clip(log_std, -20.0, 2.0)
            return mean, log_std
        else:
            # Para acciones discretas, predecimos logits
            logits = nn.Dense(self.action_dim)(x)
            return logits, None

class REINFORCEState(NamedTuple):
    """Estructura para almacenar el estado del agente REINFORCE"""
    policy_params: Any
    baseline_params: Optional[Any]
    optimizer_state: Any
    baseline_optimizer_state: Optional[Any]
    rng_key: jnp.ndarray
    total_steps: int

class BaselineNetwork(nn.Module):
    """
    Red de valor para reducir la varianza en REINFORCE.
    
    Implementa una red que estima el valor esperado del retorno
    desde un estado dado, actuando como línea base para reducir
    la varianza del gradiente de política.
    """
    hidden_units: List[int]
    
    @nn.compact
    def __call__(self, x: jnp.ndarray, training: bool = False) -> jnp.ndarray:
        """
        Ejecuta la red de valor en la entrada dada.
        
        Parámetros:
        -----------
        x : jnp.ndarray
            Observación o estado de entrada
        training : bool, opcional
            Indica si está en modo entrenamiento (default: False)
            
        Retorna:
        --------
        jnp.ndarray
            Estimación del valor del estado
        """
        for units in self.hidden_units:
            x = nn.Dense(units)(x)
            x = nn.relu(x)
            if training:
                x = nn.Dropout(rate=0.1)(x, deterministic=not training)
        
        # Retornamos un único valor escalar
        return nn.Dense(1)(x)

class REINFORCE:
    """
    Implementación del algoritmo Monte Carlo Policy Gradient (REINFORCE) con JAX.
    
    Este algoritmo actualiza una política parametrizada con redes neuronales
    utilizando el gradiente de política, calculado a partir de episodios completos.
    """
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_units: List[int] = REINFORCE_CONFIG['hidden_units'],
        learning_rate: float = REINFORCE_CONFIG['learning_rate'],
        gamma: float = REINFORCE_CONFIG['gamma'],
        continuous: bool = True,
        use_baseline: bool = REINFORCE_CONFIG['use_baseline'],
        entropy_coef: float = REINFORCE_CONFIG['entropy_coef'],
        seed: int = 42,
        cgm_shape: Optional[Tuple[int, ...]] = None,
        other_features_shape: Optional[Tuple[int, ...]] = None
    ) -> None:
        """
        Inicializa el agente REINFORCE.
        
        Parámetros:
        -----------
        state_dim : int
            Dimensión del espacio de estados
        action_dim : int
            Dimensión del espacio de acciones
        hidden_units : List[int], opcional
            Unidades en capas ocultas (default: REINFORCE_CONFIG['hidden_units'])
        learning_rate : float, opcional
            Tasa de aprendizaje (default: REINFORCE_CONFIG['learning_rate'])
        gamma : float, opcional
            Factor de descuento para recompensas futuras (default: REINFORCE_CONFIG['gamma'])
        continuous : bool, opcional
            Si el espacio de acción es continuo (default: True)
        use_baseline : bool, opcional
            Si usar una red de valor como línea base (default: REINFORCE_CONFIG['use_baseline'])
        entropy_coef : float, opcional
            Coeficiente para el término de entropía (default: REINFORCE_CONFIG['entropy_coef'])
        seed : int, opcional
            Semilla para reproducibilidad (default: 42)
        cgm_shape : Optional[Tuple[int, ...]], opcional
            Forma de los datos CGM (default: None)
        other_features_shape : Optional[Tuple[int, ...]], opcional
            Forma de otras características (default: None)
        """
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.hidden_units = hidden_units
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.continuous = continuous
        self.use_baseline = use_baseline
        self.entropy_coef = entropy_coef
        self.cgm_shape = cgm_shape
        self.other_features_shape = other_features_shape
        
        # Inicializar clave PRNG
        self.rng_key = jax.random.PRNGKey(seed)
        
        # Crear redes de política y valor
        self.policy_network = PolicyNetwork(
            hidden_units=self.hidden_units,
            action_dim=self.action_dim,
            continuous=self.continuous
        )
        
        if self.use_baseline:
            self.baseline_network = BaselineNetwork(
                hidden_units=self.hidden_units
            )
        
        # Inicializar los parámetros de las redes
        dummy_state = jnp.zeros((1, self.state_dim))
        self.rng_key, policy_key, baseline_key = random.split(self.rng_key, 3)
        
        # Inicializar parámetros de la política
        policy_variables = self.policy_network.init(policy_key, dummy_state)
        self.policy_params = policy_variables['params']
        
        # Inicializar parámetros de la línea base si se usa
        self.baseline_params = None
        if self.use_baseline:
            baseline_variables = self.baseline_network.init(baseline_key, dummy_state)
            self.baseline_params = baseline_variables['params']
        
        # Configurar optimizadores
        self.optimizer = optax.adam(learning_rate=self.learning_rate)
        self.optimizer_state = self.optimizer.init(self.policy_params)
        
        self.baseline_optimizer = None
        self.baseline_optimizer_state = None
        if self.use_baseline:
            self.baseline_optimizer = optax.adam(learning_rate=self.learning_rate)
            self.baseline_optimizer_state = self.baseline_optimizer.init(self.baseline_params)
        
        # Estado para seguimiento
        self.state = REINFORCEState(
            policy_params=self.policy_params,
            baseline_params=self.baseline_params,
            optimizer_state=self.optimizer_state,
            baseline_optimizer_state=self.baseline_optimizer_state,
            rng_key=self.rng_key,
            total_steps=0
        )
        
        # Historial para métricas
        self.history = {'loss': [], 'avg_reward': [], 'entropy': []}
        
        # Compilar funciones puras con JAX
        self._jit_sample_action = jit(self._sample_action)
        self._jit_get_action = jit(self._get_action)
        self._jit_compute_returns = jit(self._compute_returns)
    
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
    
    def _create_policy_network(self) -> nn.Module:
        """
        Crea una red neuronal para la política.

        Retorna:
        --------
        nn.Module
            Módulo de red de política
        """
        class PolicyNetwork(nn.Module):
            """Red neuronal para la política del agente."""
            hidden_units: Tuple[int, ...]
            continuous: bool
            action_dim: int
            
            @nn.compact
            def __call__(self, x: jnp.ndarray, training: bool = False) -> Tuple[jnp.ndarray, jnp.ndarray]:
                # Validación de entrada
                # expected_shape = x.shape
                
                # Capas ocultas
                for i, units in enumerate(self.hidden_units):
                    x = nn.Dense(features=units, name=f'policy_hidden_{i}')(x)
                    x = nn.LayerNorm(epsilon=REINFORCE_CONFIG['epsilon'], name=f'policy_ln_{i}')(x)
                    x = nn.relu(x)
                    if REINFORCE_CONFIG['dropout_rate'] > 0 and training:
                        x = nn.Dropout(
                            rate=REINFORCE_CONFIG['dropout_rate'], 
                            deterministic=not training,
                            name=f'policy_dropout_{i}'
                        )(x)
                
                # Salida depende del tipo de espacio de acción
                if self.continuous:
                    # Para espacio continuo: media y log-desviación estándar
                    mean = nn.Dense(features=self.action_dim, name='policy_mean')(x)
                    log_std = nn.Dense(features=self.action_dim, name='policy_log_std')(x)
                    
                    # Limitar log_std para estabilidad (-5, 2)
                    log_std = jnp.clip(log_std, -5.0, 2.0)
                    
                    return mean, log_std
                else:
                    # Para espacio discreto: logits
                    logits = nn.Dense(features=self.action_dim, name='policy_logits')(x)
                    return logits, None
        
        return PolicyNetwork(
            hidden_units=self.hidden_units,
            continuous=self.continuous,
            action_dim=self.action_dim
        )
    
    def _sample_action(
        self,
        policy_params: Any,
        state: jnp.ndarray,
        rng_key: jnp.ndarray,
        deterministic: bool = False
    ) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        """
        Muestrea una acción de la política y devuelve información adicional.
        
        Parámetros:
        -----------
        policy_params : Any
            Parámetros de la red de política
        state : jnp.ndarray
            Estado actual
        rng_key : jnp.ndarray
            Llave para generación de números aleatorios
        deterministic : bool, opcional
            Si seleccionar la acción determinísticamente (default: False)
            
        Retorna:
        --------
        Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]
            (acción, log-probabilidad, entropía)
        """
        # Verificación y corrección de dimensiones
        expected_dim = self.reinforce_agent.state_dim
        
        # Validar dimensión
        if state.ndim == 1:
            if state.shape[0] != expected_dim:
                print_warning(f"State dimension mismatch: got {state.shape[0]}, expected {expected_dim}")
                
                # Adaptación de estado a la dimensión correcta
                if state.shape[0] > expected_dim:
                    state = state[:expected_dim]
                else:
                    padding = jnp.zeros(expected_dim - state.shape[0])
                    state = jnp.concatenate([state, padding])
                    
            # Añadir dimensión de batch
            state = state.reshape(1, -1)
        
        # Obtener outputs de la red de política
        if self.reinforce_agent.continuous:
            # Espacio de acción continuo
            mean, log_std = self.reinforce_agent.policy_network.apply({'params': policy_params}, state)
            std = jnp.exp(log_std)
            
            if deterministic:
                # En modo determinístico, devolver la media
                action = mean
            else:
                # Muestrear de la distribución normal
                rng_key, subkey = random.split(rng_key)
                action = mean + std * random.normal(subkey, mean.shape)
            
            # Calcular log-probabilidad
            log_prob = -0.5 * jnp.sum(
                ((action - mean) / std) ** 2 + 2 * log_std + jnp.log(2 * jnp.pi)
            )
            
            # Calcular entropía
            entropy = jnp.sum(log_std + 0.5 * jnp.log(2 * jnp.pi * jnp.e))
        else:
            # Espacio de acción discreto
            logits = self.reinforce_agent.policy_network.apply({'params': policy_params}, state)
            
            if deterministic:
                # En modo determinístico, devolver acción con mayor probabilidad
                action = jnp.argmax(logits, axis=-1)
            else:
                # Muestrear de la distribución categórica
                rng_key, subkey = random.split(rng_key)
                action = random.categorical(subkey, logits)
            
            # Calcular log-probabilidad y entropía
            probs = jax.nn.softmax(logits)
            log_probs = jnp.log(probs + 1e-8)  # Evitar log(0)
            log_prob = log_probs[0, action]
            entropy = -jnp.sum(probs * log_probs, axis=-1)[0]
        
        return action, log_prob, entropy
    
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
        logits = self.policy_network.apply({'params': params}, state[None])
        
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
        # Asegurar que el estado tenga dimensión de lote
        if state.ndim == 1:
            state = state.reshape(1, -1)
            
        mu, log_sigma = self.policy_network.apply({'params': params}, state)
        
        if deterministic:
            action = mu[0]
        else:
            # Calcular sigma a partir de log_sigma
            sigma = jnp.exp(log_sigma)
            
            # Generar ruido aleatorio
            rng_key, subkey = jax.random.split(rng_key)
            noise = jax.random.normal(subkey, mu.shape)
            
            # Calcular acción estocástica
            stochastic_action = mu + sigma * noise
            action = stochastic_action[0]
        
        return action, rng_key
    
    def _get_action(
        self,
        state: jnp.ndarray,
        rng_key: jnp.ndarray,
        policy_params: Any,
        deterministic: bool = False
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """
        Selecciona una acción según la política actual.
        
        Parámetros:
        -----------
        state : jnp.ndarray
            Estado actual
        rng_key : jnp.ndarray
            Clave para generación aleatoria
        policy_params : Any
            Parámetros de la política
        deterministic : bool, opcional
            Si es True, selecciona la acción determinísticamente (default: False)
            
        Retorna:
        --------
        Tuple[jnp.ndarray, jnp.ndarray]
            Tupla con (acción seleccionada, nueva clave PRNG)
        """
        action, _, _ = self._sample_action(policy_params, state, rng_key, deterministic)
        _, next_key = random.split(rng_key)
        return action, next_key

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
                self.state.policy_params,
                state, 
                self.state.rng_key, 
                deterministic
            )
        else:
            action, new_key = self._get_action_discrete(
                self.state.policy_params,
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
        logits = self.policy_network.apply({'params': params}, states)
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
        mu, log_sigma = self.policy_network.apply({'params': params}, states)
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
            return self._get_log_prob_continuous(self.state.policy_params, states, actions)
        else:
            return self._get_log_prob_discrete(self.state.policy_params, states, actions)
    
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
        logits = self.policy_network.apply({'params': params}, states)
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
        _, log_sigma = self.policy_network.apply({'params': params}, states)
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
            return self._get_entropy_continuous(self.state.policy_params, states)
        else:
            return self._get_entropy_discrete(self.state.policy_params, states)
    
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
    
    def _value_loss_fn(
        self, 
        params: Dict, 
        states: jnp.ndarray, 
        returns: jnp.ndarray
    ) -> jnp.ndarray:
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
        # Obtener valores predichos por la red de valor
        values = jnp.squeeze(self.baseline_network.apply({'params': params}, states))
        
        # Calcular pérdida por error cuadrático medio
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
    
    def _compute_returns(self, rewards: jnp.ndarray, dones: jnp.ndarray) -> jnp.ndarray:
        """
        Calcula los retornos descontados para un episodio.
        
        Parámetros:
        -----------
        rewards : jnp.ndarray
            Array de recompensas
        dones : jnp.ndarray
            Array de indicadores de fin de episodio
            
        Retorna:
        --------
        jnp.ndarray
            Retornos descontados
        """
        # Calcular retornos Monte Carlo descontados
        n_steps = len(rewards)
        returns = jnp.zeros_like(rewards)
        future_return = 0.0
        
        # Iteramos en orden inverso (del final al principio) para calcular retornos acumulados
        for t in reversed(range(n_steps)):
            # Si done=1, reiniciamos el retorno futuro (fin de episodio)
            # Si done=0, propagamos el retorno futuro descontado
            future_return = rewards[t] + (1.0 - dones[t]) * self.gamma * future_return
            returns = returns.at[t].set(future_return)
        
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
            # Usar baseline_network para obtener valores de estado
            values = jnp.squeeze(self.baseline_network.apply({'params': self.state.baseline_params}, states))
            
            # Calcular pérdida de baseline
            baseline_loss = self._value_loss_fn(
                self.state.baseline_params, 
                states, 
                returns
            )
            self.baseline_loss_metric = float(baseline_loss)
            
            # Actualizar optimizer state para baseline
            grads = jax.grad(self._value_loss_fn)(self.state.baseline_params, states, returns)
            updates, new_baseline_optimizer_state = self.baseline_optimizer.update(
                grads, self.state.baseline_optimizer_state
            )
            new_baseline_params = optax.apply_updates(self.state.baseline_params, updates)
        else:
            new_baseline_params = self.state.baseline_params
            new_baseline_optimizer_state = self.state.baseline_optimizer_state
        
        # Actualizar red de política - CORRECCIÓN AQUÍ
        # Usar value_and_grad en lugar de grad para obtener tanto la pérdida como los gradientes
        (_, (policy_loss, entropy)), grads = jax.value_and_grad(self._policy_loss_fn, has_aux=True)(
            self.state.policy_params, 
            states, 
            actions, 
            returns, 
            values
        )
        
        updates, new_optimizer_state = self.optimizer.update(
            grads, self.state.optimizer_state
        )
        new_policy_params = optax.apply_updates(self.state.policy_params, updates)
        
        # Actualizar estado del agente
        self.state = REINFORCEState(
            policy_params=new_policy_params,
            baseline_params=new_baseline_params,
            optimizer_state=new_optimizer_state,
            baseline_optimizer_state=new_baseline_optimizer_state,
            rng_key=self.state.rng_key,
            total_steps=self.state.total_steps + 1
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
            Si se debe renderizar el entorno durante evaluación (default: False)
            
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
            f.write(flax.serialization.to_bytes(self.state.policy_params))
        
        # Guardar baseline si existe
        if self.use_baseline and baseline_path:
            with open(baseline_path, 'wb') as f:
                f.write(flax.serialization.to_bytes(self.state.baseline_params))
        
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
                self.state.policy_params,
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
            optimizer_state=self.state.optimizer_state,
            baseline_optimizer_state=self.state.baseline_optimizer_state,
            rng_key=self.state.rng_key,
            total_steps=self.state.total_steps
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

    def train_batch(
        self, 
        agent_state: REINFORCEState, 
        batch_data: Tuple, 
        rng_key: jax.random.PRNGKey
    ) -> Tuple[REINFORCEState, Dict[str, float]]:
        """
        Entrena el agente con un lote de datos.
        
        Parámetros:
        -----------
        agent_state : REINFORCEState
            Estado actual del agente
        batch_data : Tuple
            Datos del lote (x_cgm, x_other, y)
        rng_key : jax.random.PRNGKey
            Clave para generación aleatoria
            
        Retorna:
        --------
        Tuple[REINFORCEState, Dict[str, float]]
            Tupla con (estado actualizado del agente, métricas)
        """
        # Desempaquetar datos del lote
        x_cgm, x_other, y = batch_data
        batch_size = x_cgm.shape[0]
        
        # Inicializar métricas
        policy_loss = 0.0
        # value_loss = 0.0
        entropy = 0.0
        rewards = []
        
        # Actualizar estado con nueva clave
        updated_state = agent_state._replace(rng_key=rng_key)
        key_update = rng_key
        
        # Entrenar para cada muestra en el lote
        for i in range(batch_size):
            # Codificar estado
            cgm_encoded = self.encode_cgm(x_cgm[i:i+1])
            other_encoded = self.encode_other(x_other[i:i+1])
            state = jnp.concatenate([cgm_encoded[0], other_encoded[0]])
            
            # Muestrear acción
            key_update, subkey = jax.random.split(key_update)
            action, log_prob, sample_entropy = self._sample_action(
                updated_state.policy_params,
                state,
                subkey,
                deterministic=False
            )
            
            # Calcular recompensa (error negativo cuadrático)
            if self.reinforce_agent.continuous:
                pred_dose = (action[0] + 1.0) * 7.5  # Mapear de [-1,1] a [0,15]
            else:
                pred_dose = action * (15.0 / (self.reinforce_agent.action_dim - 1))
                
            reward = -((pred_dose - y[i]) ** 2)
            rewards.append(reward)
            
            # Actualizar pérdidas acumuladas
            policy_loss -= log_prob * reward
            entropy -= sample_entropy
        
        # Calcular gradientes para actualización
        key_update, subkey = jax.random.split(key_update)
        grad_fn = lambda p: self._compute_policy_gradient(p, updated_state.baseline_params, x_cgm, x_other, y, key_update)
        grads = grad_fn(updated_state.policy_params)
        
        # Aplicar actualizaciones usando el optimizador
        updates, new_optimizer_state = self.reinforce_agent.optimizer.update(
            grads, 
            updated_state.optimizer_state, 
            updated_state.policy_params
        )
        new_policy_params = optax.apply_updates(updated_state.policy_params, updates)
        
        # Actualizar el estado del agente
        new_state = updated_state._replace(
            policy_params=new_policy_params,
            optimizer_state=new_optimizer_state,
            rng_key=key_update,
            total_steps=updated_state.total_steps + batch_size
        )
        
        # Métricas para monitoreo
        metrics = {
            'loss': float(policy_loss / batch_size),
            'entropy': float(entropy / batch_size),
            'reward': float(np.mean(rewards))
        }
        
        return new_state, metrics

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
        cgm_flatten_dim = int(np.prod(self.cgm_shape[1:]))  # 24*3=72
        other_flatten_dim = int(np.prod(self.other_features_shape[1:]))  # 6
        
        print_debug(f"Dimensión aplanada de CGM: {cgm_flatten_dim}")
        print_debug(f"Dimensión aplanada de otras características: {other_flatten_dim}")
        
        # Inicializar matrices de transformación
        self.key, key_cgm, key_other = jax.random.split(self.key, 3)
        
        # Crear matrices de proyección para entradas
        state_dim_half = max(1, self.reinforce_agent.state_dim // 2)
        self.cgm_weight = jax.random.normal(key_cgm, (cgm_flatten_dim, state_dim_half))
        self.other_weight = jax.random.normal(key_other, (other_flatten_dim, state_dim_half))
        
        print_debug(f"Forma matriz de pesos CGM: {self.cgm_weight.shape}")
        print_debug(f"Forma matriz de pesos otras características: {self.other_weight.shape}")
        
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
            # Guardar dimensión de lote original
            batch_size = x.shape[0]
            
            # Obtener dimensión de entrada esperada desde la forma de los pesos
            expected_input_dim = weights.shape[0]
            
            # Garantizar que x se aplana a la dimensión correcta
            # Primero aplanamos todas las dimensiones excepto la de lote
            x_flat = x.reshape(batch_size, -1)
            
            # Verificar si la dimensión aplanada coincide con la esperada
            if x_flat.shape[1] != expected_input_dim:
                # Imprimir advertencia para depuración
                print_debug(f"Incompatibilidad de dimensiones en encoder: Esperada {expected_input_dim}, "
                        f"obtenida {x_flat.shape[1]}. Ajustando.")
                
                # Si las dimensiones no coinciden, ajustar mediante truncado o padding
                if x_flat.shape[1] > expected_input_dim:
                    # Truncar si es mayor
                    x_flat = x_flat[:, :expected_input_dim]
                else:
                    # Padding con ceros si es menor
                    padding = jnp.zeros((batch_size, expected_input_dim - x_flat.shape[1]))
                    x_flat = jnp.concatenate([x_flat, padding], axis=1)
            
            # Aplicar transformación
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
    
    def get_action(
        self, 
        cgm_data: np.ndarray, 
        other_data: np.ndarray = None, 
        deterministic: bool = False
    ) -> Union[np.ndarray, int]:
        """
        Obtiene una acción para la entrada dada usando el agente REINFORCE.
        
        Parámetros:
        -----------
        cgm_data : np.ndarray
            Datos CGM para predicción
        other_data : np.ndarray, opcional
            Otras características para predicción (default: None)
        deterministic : bool, opcional
            Si es True, usa modo determinista (default: False)
            
        Retorna:
        --------
        Union[np.ndarray, int]
            Acción seleccionada por la política
        """
        # Verificar si tenemos otras características
        has_other_features = other_data is not None and other_data.size > 0
        
        # Para datos CGM, asegurar la dimensión de lote y forma correctas
        # Asegurar que tiene dimensión de lote
        if cgm_data.ndim == 1:
            cgm_data_2d = cgm_data.reshape(1, -1)
        elif cgm_data.ndim == 2:
            # Si ya tiene forma (batch, features)
            if cgm_data.shape[0] == 1:
                cgm_data_2d = cgm_data
            else:
                # Asumir que es una muestra sin dimensión de lote
                cgm_data_2d = cgm_data.reshape(1, *cgm_data.shape)
        elif cgm_data.ndim == 3:
            # Ya tiene el formato correcto (batch, timesteps, features)
            cgm_data_2d = cgm_data
        else:
            # Formato inesperado, convertir a 2D con dimensión de lote
            cgm_data_2d = cgm_data.reshape(1, -1)
        
        # Codificar CGM
        cgm_encoded = self.encode_cgm(jnp.array(cgm_data_2d))
        
        # Procesar otras características si las hay
        if has_other_features:
            if other_data.ndim == 1:
                other_data_2d = other_data.reshape(1, -1)
            else:
                other_data_2d = other_data
            
            other_encoded = self.encode_other(jnp.array(other_data_2d))
        else:
            # Si no hay otras características, crear tensor de ceros
            batch_size = cgm_encoded.shape[0]
            feature_dim = self.reinforce_agent.state_dim // 2
            other_encoded = jnp.zeros((batch_size, feature_dim))
        
        # Concatenar características codificadas
        estado = jnp.concatenate([cgm_encoded, other_encoded], axis=-1)
        
        # Convertir a NumPy
        estado_np = np.array(estado)
        
        # Si solo hay una muestra, eliminar dimensión de batch
        if estado_np.shape[0] == 1:
            estado_np = estado_np[0]
        
        # Delegar al método get_action del agente REINFORCE
        return self.reinforce_agent.get_action(estado_np, deterministic=deterministic)
    
    def _extract_input_data(self, *args) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """
        Extrae los datos de entrada de los argumentos proporcionados.
        
        Parámetros:
        -----------
        *args : Union[List[np.ndarray], Tuple[np.ndarray, np.ndarray], np.ndarray]
            Argumentos de entrada con datos CGM y otras características
            
        Retorna:
        --------
        Tuple[jnp.ndarray, jnp.ndarray]
            Tupla con (cgm_data, other_features) como arrays JAX
        """
        if len(args) == 1:
            if isinstance(args[0], (list, tuple)) and len(args[0]) == 2:
                cgm_data, other_features = args[0]
            else:
                cgm_data = args[0]
                other_features = None
        elif len(args) == 2:
            cgm_data, other_features = args
        else:
            raise ValueError("Formato de entrada no válido. Proporcione una lista [cgm_data, other_features] o dos argumentos separados.")
        
        # Convertir a arrays JAX
        cgm_data = jnp.array(cgm_data)
        if other_features is not None:
            other_features = jnp.array(other_features)
        else:
            other_features = jnp.zeros((cgm_data.shape[0], 0))
            
        return cgm_data, other_features

    @partial(jax.jit, static_argnums=0)
    def _process_batch(self, cgm_batch, other_batch):
        """
        Procesa un lote de datos para predicción.
        
        Parámetros:
        -----------
        cgm_batch : jnp.ndarray
            Lote de datos CGM
        other_batch : jnp.ndarray
            Lote de otras características
            
        Retorna:
        --------
        jnp.ndarray
            Predicciones para el lote
        """
        # Codificar entradas
        cgm_encoded = self.encode_cgm(cgm_batch)
        other_encoded = self.encode_other(other_batch) if other_batch.shape[1] > 0 else jnp.zeros((cgm_batch.shape[0], self.reinforce_agent.state_dim // 2))
        
        # Concatenar características
        states = jnp.concatenate([cgm_encoded, other_encoded], axis=1)
        
        # Obtener acciones según el tipo de espacio
        if self.reinforce_agent.continuous:
            mean, _ = self.reinforce_agent.policy_network.apply(
                {'params': self.reinforce_agent.state.policy_params}, 
                states
            )
            # Convertir a rango de dosis [0, 15]
            actions = (mean + 1.0) * 7.5
        else:
            logits = self.reinforce_agent.policy_network.apply(
                {'params': self.reinforce_agent.state.policy_params}, 
                states
            )
            actions = jnp.argmax(logits, axis=1) / (self.reinforce_agent.action_dim - 1) * 15.0
            
        return actions.reshape(-1, 1)
        
    def predict(self, *args) -> jnp.ndarray:
        """
        Realiza predicciones utilizando la política entrenada.
        
        Parámetros:
        -----------
        *args : Union[List[np.ndarray], Tuple[np.ndarray, np.ndarray], np.ndarray]
            Puede ser:
            - Una lista o tupla [cgm_data, other_features]
            - Dos argumentos separados: cgm_data, other_features
            
        Retorna:
        --------
        jnp.ndarray
            Predicciones (acciones) del modelo
        """
        # Extraer datos de entrada usando método auxiliar
        cgm_data, other_features = self._extract_input_data(*args)
        
        # Configurar parámetros de procesamiento por lotes
        batch_size = 512
        num_samples = cgm_data.shape[0]
        predictions = np.zeros((num_samples, 1))
        
        print(f"Procesando predicciones para {num_samples} muestras...")
        
        # Procesar por lotes con indicador de progreso
        start_time = time.time()
        for i in range(0, num_samples, batch_size):
            # Mostrar progreso periódicamente
            if i % (batch_size * 10) == 0 and i > 0:
                elapsed = time.time() - start_time
                percent_done = i / num_samples * 100
                est_total = elapsed / percent_done * 100
                remaining = est_total - elapsed
                print(f"Progreso: {i}/{num_samples} ({percent_done:.1f}%) - Tiempo restante estimado: {remaining:.1f}s")
            
            # Extraer lote actual
            end_idx = min(i + batch_size, num_samples)
            batch_cgm = cgm_data[i:end_idx]
            batch_other = other_features[i:end_idx]
            
            # Procesar lote usando método auxiliar
            batch_predictions = self._process_batch(batch_cgm, batch_other)
            predictions[i:end_idx] = np.array(batch_predictions)
        
        # Mostrar tiempo total
        total_time = time.time() - start_time
        print(f"Predicciones completadas en {total_time:.2f}s")
        
        return predictions
    
    def _get_verbose_int(self, verbose: Any) -> int:
        """
        Convierte el parámetro verbose a entero.
        
        Parámetros:
        -----------
        verbose : Any
            Valor de verbose que puede ser entero, float o array
            
        Retorna:
        --------
        int
            Valor entero para verbose
        """
        verbose_int = 1  # Valor predeterminado
        
        if isinstance(verbose, (int, float)):
            verbose_int = int(verbose)
        elif isinstance(verbose, (np.ndarray, jnp.ndarray)):
            # Para arrays numpy/jax, verificar tamaño adecuadamente
            if verbose.size == 1:
                verbose_int = int(verbose.item())
            elif verbose.size > 1:
                # Para arrays con múltiples elementos, usar el primero
                verbose_int = int(verbose[0])
            # Si size=0, usar el valor predeterminado 1
        elif hasattr(verbose, 'item'):
            try:
                # Intentar convertir a entero para otros objetos array-like
                verbose_int = int(verbose.item())
            except (ValueError, TypeError):
                # Si falla, usar el valor predeterminado
                print_warning("No se pudo convertir verbose a entero, usando valor predeterminado 1")
        
        return verbose_int
    
    def _update_history_from_reinforce(self, reinforce_history: Dict) -> None:
        """
        Actualiza el historial del modelo con las métricas del entrenamiento REINFORCE.
        
        Parámetros:
        -----------
        reinforce_history : Dict
            Historia de entrenamiento del agente REINFORCE
        """
        self.history['episode_rewards'] = reinforce_history['episode_rewards']
        self.history['policy_loss'] = reinforce_history['policy_losses']
        
        if self.reinforce_agent.use_baseline:
            self.history['baseline_loss'] = reinforce_history['baseline_losses']
        
        self.history['entropy'] = reinforce_history.get('entropies', [])
    
    def _evaluate_validation_data(self, validation_data: Tuple) -> Optional[float]:
        """
        Evalúa el modelo en los datos de validación.
        
        Parámetros:
        -----------
        validation_data : Tuple
            Datos de validación
            
        Retorna:
        --------
        Optional[float]
            Pérdida de validación si los datos son válidos, None en caso contrario
        """
        if len(validation_data) != 2:
            print_warning(f"Formato de datos de validación incorrecto. Se esperaba una tupla (val_x, val_y) pero se recibió {type(validation_data)}")
            return None
        
        val_x, val_y = validation_data
        
        # Verificar formato de val_x
        if not (isinstance(val_x, (list, tuple)) and len(val_x) == 2):
            print_warning("Formato de datos de validación incorrecto. Se esperaba una tupla (val_x, val_y) donde val_x=[cgm_data, other_features]")
            return None
        
        # Evaluar en datos de validación
        val_preds = self.predict(val_x)
        val_loss = float(jnp.mean((val_preds.flatten() - val_y) ** 2))
        self.history['val_loss'].append(val_loss)
        
        return val_loss
        
    def fit(
        self, 
        x: List[jnp.ndarray], 
        y: jnp.ndarray, 
        validation_data: Optional[Tuple] = None, 
        epochs: int = 1,
        batch_size: int = 32,
        callbacks: List = None,
        verbose: Any = 0
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
        verbose : Any, opcional
            Nivel de verbosidad (default: 0)
            
        Retorna:
        --------
        Dict
            Historia del entrenamiento
        """
        # Procesar nivel de verbosidad
        verbose_int = self._get_verbose_int(verbose)
        
        if verbose_int > 0:
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
        self._update_history_from_reinforce(reinforce_history)
        
        # Calcular pérdida en datos de entrenamiento
        train_preds = self.predict(x)
        train_loss = float(jnp.mean((train_preds.flatten() - y) ** 2))
        self.history['loss'].append(train_loss)
        
        # Evaluar en datos de validación si se proporcionan
        val_loss = None
        if validation_data:
            val_loss = self._evaluate_validation_data(validation_data)
        
        # Mostrar resumen si verbose
        if verbose_int > 0:
            print(f"Entrenamiento completado. Pérdida final: {train_loss:.4f}")
            if val_loss is not None:
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

    def start(
        self, 
        x_cgm: jnp.ndarray, 
        x_other: jnp.ndarray, 
        y: jnp.ndarray, 
        rng_key: jnp.ndarray
    ) -> None:
        """
        Inicializa el agente con los datos de entrenamiento.
        
        Parámetros:
        -----------
        x_cgm : jnp.ndarray
            Datos de monitoreo continuo de glucosa para entrenamiento
        x_other : jnp.ndarray
            Otras características para entrenamiento
        y : jnp.ndarray
            Etiquetas de entrenamiento (dosis de insulina objetivo)
        rng_key : jnp.ndarray
            Clave para generación de números aleatorios
            
        Retorna:
        --------
        None
            No retorna valor
        """
        # Guardar las referencias a los datos
        self.x_cgm = x_cgm
        self.x_other = x_other
        self.y = y
        
        # Actualizar la clave de aleatoriedad
        self.key = rng_key
        
        # Reinicializar las funciones de codificación con las formas correctas
        # basadas en los datos de entrenamiento reales
        self.cgm_shape = (x_cgm.shape[0],) + x_cgm.shape[1:]
        self.other_features_shape = (x_other.shape[0],) + x_other.shape[1:]
        
        print_debug(f"CGM shape: {self.cgm_shape}")
        print_debug(f"Other features shape: {self.other_features_shape}")
        
        # Reconfigura los codificadores usando la forma real de los datos
        self._setup_encoders()
        
        # Crear el entorno de entrenamiento
        self.env = self._create_training_environment(x_cgm, x_other, y)
        
        # Configurar el estado inicial del agente
        self.agent_state = self.reinforce_agent.state

    def setup(self, rng_key: jax.random.PRNGKey) -> Any:
        """
        Inicializa el agente REINFORCE para interactuar con el entorno.
        
        Parámetros:
        -----------
        rng_key : jax.random.PRNGKey
            Clave para generación aleatoria
            
        Retorna:
        --------
        Any
            Estado del agente inicializado
        """
        self.rng_key = rng_key
        
        # Inicializar policy_network si es necesario
        if self.policy_network is None:
            self.policy_network = self._create_policy_network()
        
        # Inicializar value_network si es necesario y está habilitado
        if self.use_baseline and self.value_network is None:
            self.value_network = self._create_value_network()
        
        # Asegurarse de que el estado esté adecuadamente inicializado
        if self.state is None:
            # Inicializar estado con parámetros aleatorios
            dummy_state = jnp.zeros((1, self.state_dim))
            policy_key, value_key, rng_key = jax.random.split(rng_key, 3)
            
            # Inicializar parámetros de la política
            policy_variables = self.policy_network.init(policy_key, dummy_state)
            policy_params = policy_variables['params']
            
            # Crear estado de entrenamiento para la política
            policy_state = train_state.TrainState.create(
                apply_fn=self.policy_network.apply,
                params=policy_params,
                tx=self.optimizer
            )
            
            # Inicializar baseline si es necesario
            value_params = None
            value_state = None
            baseline_optimizer_state = None
            
            if self.use_baseline:
                value_variables = self.value_network.init(value_key, dummy_state)
                value_params = value_variables['params']
                
                # Crear estado de entrenamiento para el baseline
                value_state = train_state.TrainState.create(
                    apply_fn=self.value_network.apply,
                    params=value_params,
                    tx=self.baseline_optimizer
                )
                
                baseline_optimizer_state = self.baseline_optimizer.init(value_params)
            
            # Crear estado REINFORCE completo
            self.state = REINFORCEState(
                policy_state=policy_state,
                value_state=value_state,
                optimizer_state=self.optimizer.init(policy_params),
                baseline_optimizer_state=baseline_optimizer_state,
                rng_key=rng_key,
                total_steps=0
            )
        
        return self.state

    def _parse_verbose(self, verbose: Any) -> int:
        """
        Procesa el parámetro verbose y lo convierte a entero.
        
        Parámetros:
        -----------
        verbose : Any
            Nivel de verbosidad en diversos formatos
            
        Retorna:
        --------
        int
            Nivel de verbosidad como entero
        """
        verbose_int = 1  # Valor predeterminado
        
        try:
            if isinstance(verbose, (int, float)):
                verbose_int = int(verbose)
            elif isinstance(verbose, (np.ndarray, jnp.ndarray)):
                verbose_int = self._parse_verbose_array(verbose)
            elif hasattr(verbose, 'item'):
                try:
                    verbose_int = int(verbose.item())
                except (ValueError, TypeError):
                    print_warning("No se pudo convertir verbose.item() a entero, usando valor predeterminado 1")
        except Exception as e:
            print_warning(f"Error al procesar el parámetro verbose: {str(e)}. Usando valor predeterminado 1")
            
        return verbose_int
    
    def _parse_verbose_array(self, verbose_arr: Union[np.ndarray, jnp.ndarray]) -> int:
        """
        Procesa un array de verbosidad para extraer un valor entero.
        
        Parámetros:
        -----------
        verbose_arr : Union[np.ndarray, jnp.ndarray]
            Array que contiene el nivel de verbosidad
            
        Retorna:
        --------
        int
            Nivel de verbosidad como entero
        """
        if verbose_arr.size == 1:
            return int(verbose_arr.item())
        elif verbose_arr.size > 1:
            try:
                return int(verbose_arr.flatten()[0])
            except (IndexError, ValueError, TypeError):
                print_warning("No se pudo extraer un valor escalar válido del array verbose, usando valor predeterminado 1")
        
        return 1  # Valor predeterminado
    
    def _extract_training_data(self, training_data: Any, kwargs: Dict) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        """
        Extrae los datos de entrenamiento de los argumentos proporcionados.
        
        Parámetros:
        -----------
        training_data : Any
            Datos de entrenamiento en formato aceptado por el framework
        kwargs : Dict
            Argumentos adicionales que pueden contener los datos
            
        Retorna:
        --------
        Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]
            Tupla con (x_cgm, x_other, y)
        """
        # Verificar si los datos ya han sido inicializados por el método start
        if hasattr(self, 'x_cgm') and hasattr(self, 'x_other') and hasattr(self, 'y'):
            return self.x_cgm, self.x_other, self.y
        
        # Si no están inicializados, intentar extraer de kwargs
        x_cgm = kwargs.get('x_cgm_train', None)
        x_other = kwargs.get('x_other_train', None)
        y = kwargs.get('y_train', None)
        
        # Si todavía faltan datos, verificar si training_data es una tupla
        if (x_cgm is None or x_other is None or y is None) and isinstance(training_data, (tuple, list)) and len(training_data) >= 3:
            x_cgm, x_other, y = training_data[0], training_data[1], training_data[2]
        
        # Si aún no tenemos los datos completos, mostrar un error específico
        if x_cgm is None or x_other is None or y is None:
            raise ValueError(f"No se pudieron extraer los datos de entrenamiento en formato válido. "
                            f"Asegúrese de inicializar el modelo con el método 'start' o proporcionar "
                            f"los datos en el formato correcto. Tipo de training_data: {type(training_data)}")
        
        return x_cgm, x_other, y
    
    def _prepare_validation_data(self, validation_data: Any) -> Optional[Tuple]:
        """
        Prepara los datos de validación en el formato esperado.
        
        Parámetros:
        -----------
        validation_data : Any
            Datos de validación en diversos formatos
            
        Retorna:
        --------
        Optional[Tuple]
            Datos de validación en formato ([x_cgm_val, x_other_val], y_val) o None
        """
        if validation_data is None:
            return None
            
        if isinstance(validation_data, tuple) and len(validation_data) == 3:
            x_cgm_val, x_other_val, y_val = validation_data
            return ([x_cgm_val, x_other_val], y_val)
            
        if isinstance(validation_data, tuple) and len(validation_data) == 2:
            first_elem = validation_data[0]
            if isinstance(first_elem, (list, tuple)) and len(first_elem) == 2:
                return validation_data
            else:
                print_warning("Formato de datos de validación no compatible. Se esperan 3 elementos (x_cgm_val, x_other_val, y_val) o 2 elementos donde el primero sea [x_cgm_val, x_other_val]")
        else:
            print_warning(f"Formato de validation_data no reconocido: {type(validation_data)}")
            
        return None
    
    def train(
        self,
        training_data: Any,
        verbose: Any = 1,
        callbacks: Optional[List] = None,
        **kwargs
    ) -> Dict[str, List[float]]:
        """
        Entrena el modelo con los datos proporcionados.
        
        Parámetros:
        -----------
        training_data : Any
            Datos de entrenamiento en formato esperado por el framework
        verbose : Any, opcional
            Nivel de verbosidad (0: silencioso, 1: progreso) (default: 1)
        callbacks : Optional[List], opcional
            Lista de callbacks para el entrenamiento (default: None)
            
        Retorna:
        --------
        Dict[str, List[float]]
            Historial de entrenamiento
        """
        # Procesar parámetros
        verbose_int = self._parse_verbose(verbose)
        x_cgm, x_other, y = self._extract_training_data(training_data, kwargs)
        
        # Obtener parámetros de kwargs
        epochs = kwargs.get('epochs', 1)
        batch_size = kwargs.get('batch_size', 32)
        val_data = self._prepare_validation_data(kwargs.get('validation_data', None))
        
        # Llamar al método fit existente
        return self.fit(
            x=[x_cgm, x_other],
            y=y,
            validation_data=val_data,
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=verbose_int
        )


def create_reinforce_mcgp_model(cgm_shape: Tuple[int, ...], other_features_shape: Tuple[int, ...], **kwargs) -> REINFORCEWrapper:
    """
    Crea un modelo basado en REINFORCE (Monte Carlo Policy Gradient) para predicción de dosis de insulina.
    
    Parámetros:
    -----------
    cgm_shape : Tuple[int, ...]
        Forma de los datos CGM (batch_size, time_steps, features)
    other_features_shape : Tuple[int, ...]
        Forma de otras características (batch_size, n_features)
    **kwargs
        Argumentos adicionales para configurar el modelo REINFORCE
        
    Retorna:
    --------
    REINFORCEWrapper
        Wrapper de REINFORCE que implementa la interfaz compatible con modelos de aprendizaje profundo
    """
    # Configurar el espacio de estados y acciones
    state_dim = 64  # Dimensión del espacio de estado latente (ajustar según necesidad)
    action_dim = 1  # Una dimensión para dosis continua
    continuous = True  # Usar espacio de acción continuo
    
    # Crear agente REINFORCE con configuración específica
    reinforce_agent = REINFORCE(
        state_dim=state_dim,
        action_dim=action_dim,
        continuous=continuous,
        learning_rate=REINFORCE_CONFIG['learning_rate'],
        gamma=REINFORCE_CONFIG['gamma'],
        hidden_units=REINFORCE_CONFIG['hidden_units'],
        use_baseline=REINFORCE_CONFIG['use_baseline'],
        entropy_coef=REINFORCE_CONFIG['entropy_coef'],
        seed=kwargs.get('seed', 42),
        cgm_shape=cgm_shape,
        other_features_shape=other_features_shape
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