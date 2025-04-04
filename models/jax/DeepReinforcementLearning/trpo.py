import os, sys
import jax
import jax.numpy as jnp
import numpy as np
import time
import flax
import flax.linen as nn
import optax
import pickle
from flax.training import train_state
from typing import Dict, List, Tuple, Any, Optional, Union, Callable, Sequence

PROJECT_ROOT = os.path.abspath(os.getcwd())
sys.path.append(PROJECT_ROOT) 

from models.config import TRPO_CONFIG


class ActorNetwork(nn.Module):
    """
    Red del Actor para TRPO que produce parámetros para una distribución de política.
    
    Esta red mapea estados a distribuciones de probabilidad sobre acciones,
    ya sean continuas (gaussiana) o discretas (categórica).
    
    Parámetros:
    -----------
    action_dim : int
        Dimensión del espacio de acciones
    continuous : bool
        Si el espacio de acciones es continuo o discreto
    hidden_units : Optional[Sequence[int]]
        Unidades en capas ocultas (default: None)
    """
    action_dim: int
    continuous: bool
    hidden_units: Optional[Sequence[int]] = None

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> Union[Tuple[jnp.ndarray, jnp.ndarray], jnp.ndarray]:
        """
        Realiza el forward pass del modelo actor.
        
        Parámetros:
        -----------
        x : jnp.ndarray
            Tensor de estados
            
        Retorna:
        --------
        Union[Tuple[jnp.ndarray, jnp.ndarray], jnp.ndarray]
            Para espacios continuos: (mu, log_std)
            Para espacios discretos: logits
        """
        hidden_units = self.hidden_units or TRPO_CONFIG['hidden_units']
        
        # Capas ocultas
        for i, units in enumerate(hidden_units):
            x = nn.Dense(units, name=f'actor_dense_{i}')(x)
            x = nn.tanh(x)
            if TRPO_CONFIG['use_layer_norm']:
                x = nn.LayerNorm(epsilon=TRPO_CONFIG['epsilon'], name=f'actor_ln_{i}')(x)
        
        # Salida depende de si las acciones son continuas o discretas
        if self.continuous:
            # Para acciones continuas (política gaussiana)
            mu = nn.Dense(self.action_dim, name='actor_mu')(x)
            # En Flax, los parámetros deben ser parte del módulo, así que usamos un parámetro
            log_std = self.param('log_std', 
                                lambda _, shape: -0.5 * jnp.ones(shape),
                                (self.action_dim,))
            return mu, log_std
        else:
            # Para acciones discretas (política categórica)
            logits = nn.Dense(self.action_dim, name='actor_logits')(x)
            return logits


class CriticNetwork(nn.Module):
    """
    Red del Crítico para TRPO que estima el valor del estado.
    
    Parámetros:
    -----------
    hidden_units : Optional[Sequence[int]]
        Unidades en capas ocultas (default: None)
    """
    hidden_units: Optional[Sequence[int]] = None
    
    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        """
        Realiza el forward pass del modelo crítico.
        
        Parámetros:
        -----------
        x : jnp.ndarray
            Tensor de estados
            
        Retorna:
        --------
        jnp.ndarray
            Valor estimado del estado
        """
        hidden_units = self.hidden_units or TRPO_CONFIG['hidden_units']
        
        # Capas ocultas
        for i, units in enumerate(hidden_units):
            x = nn.Dense(units, name=f'critic_dense_{i}')(x)
            x = nn.tanh(x)
            if TRPO_CONFIG['use_layer_norm']:
                x = nn.LayerNorm(epsilon=TRPO_CONFIG['epsilon'], name=f'critic_ln_{i}')(x)
        
        # Capa de salida (valor del estado)
        value = nn.Dense(1, name='critic_output')(x)
        return value


class CriticTrainState(train_state.TrainState):
    """
    Estado de entrenamiento para el Crítico que extiende el TrainState de Flax.
    """
    pass


class TRPOState:
    """
    Clase para mantener el estado del algoritmo TRPO.
    
    Esta clase mantiene los parámetros y estado de entrenamiento del actor y crítico.
    
    Parámetros:
    -----------
    actor_params : flax.core.FrozenDict
        Parámetros del actor
    critic_state : CriticTrainState
        Estado de entrenamiento del crítico
    key : jnp.ndarray
        Llave PRNG para generación de números aleatorios
    """
    def __init__(
        self, 
        actor_params: flax.core.FrozenDict,
        critic_state: CriticTrainState,
        key: jnp.ndarray
    ):
        self.actor_params = actor_params
        self.critic_state = critic_state
        self.key = key


class TRPO:
    """
    Implementación de Trust Region Policy Optimization (TRPO) usando JAX.
    
    TRPO es un algoritmo de optimización de política que actualiza la política
    de forma conservadora, manteniendo las actualizaciones dentro de una región
    de confianza definida por una restricción de divergencia KL.
    
    Parámetros:
    -----------
    state_dim : int
        Dimensión del espacio de estados
    action_dim : int
        Dimensión del espacio de acciones
    continuous : bool, opcional
        Si el espacio de acciones es continuo o discreto (default: True)
    gamma : float, opcional
        Factor de descuento (default: TRPO_CONFIG['gamma'])
    delta : float, opcional
        Límite de divergencia KL (default: TRPO_CONFIG['delta'])
    hidden_units : Optional[List[int]], opcional
        Unidades en capas ocultas (default: None)
    backtrack_iters : int, opcional
        Iteraciones para búsqueda de línea (default: TRPO_CONFIG['backtrack_iters'])
    backtrack_coeff : float, opcional
        Coeficiente de reducción para búsqueda de línea (default: TRPO_CONFIG['backtrack_coeff'])
    cg_iters : int, opcional
        Iteraciones para gradiente conjugado (default: TRPO_CONFIG['cg_iters'])
    damping : float, opcional
        Término de regularización para estabilidad numérica (default: TRPO_CONFIG['damping'])
    """
    def __init__(
        self, 
        state_dim: int, 
        action_dim: int,
        continuous: bool = True,
        gamma: float = TRPO_CONFIG['gamma'],
        delta: float = TRPO_CONFIG['delta'],
        hidden_units: Optional[List[int]] = None,
        backtrack_iters: int = TRPO_CONFIG['backtrack_iters'],
        backtrack_coeff: float = TRPO_CONFIG['backtrack_coeff'],
        cg_iters: int = TRPO_CONFIG['cg_iters'],
        damping: float = TRPO_CONFIG['damping']
    ) -> None:
        # Configurar semilla para reproducibilidad
        seed = TRPO_CONFIG.get('seed', 42)
        key = jax.random.PRNGKey(seed)
        np.random.seed(seed)
        
        # Parámetros del modelo
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.continuous = continuous
        self.gamma = gamma
        self.delta = delta
        self.backtrack_iters = backtrack_iters
        self.backtrack_coeff = backtrack_coeff
        self.cg_iters = cg_iters
        self.damping = damping
        
        # Valores predeterminados para capas ocultas
        if hidden_units is None:
            self.hidden_units = TRPO_CONFIG['hidden_units']
        else:
            self.hidden_units = hidden_units
        
        # Crear modelos
        self.actor = ActorNetwork(action_dim=action_dim, continuous=continuous, hidden_units=self.hidden_units)
        self.critic = CriticNetwork(hidden_units=self.hidden_units)
        
        # Inicializar parámetros
        dummy_state = jnp.ones((1, state_dim))
        key, actor_key, critic_key = jax.random.split(key, 3)
        
        self.actor_params = self.actor.init(actor_key, dummy_state)
        self.critic_params = self.critic.init(critic_key, dummy_state)
        
        # Optimizador del crítico
        critic_tx = optax.adam(learning_rate=TRPO_CONFIG['critic_learning_rate'])
        self.critic_state = CriticTrainState.create(
            apply_fn=self.critic.apply,
            params=self.critic_params,
            tx=critic_tx
        )
        
        # Estado TRPO
        self.state = TRPOState(
            actor_params=self.actor_params,
            critic_state=self.critic_state,
            key=key
        )
        
        # Métricas
        self.metrics = {
            'policy_loss': 0.0,
            'value_loss': 0.0,
            'kl_divergence': 0.0,
            'entropy': 0.0
        }
        
        # JIT-compilar funciones clave
        self.get_action_and_log_prob = jax.jit(self._get_action_and_log_prob)
        self.get_action = jax.jit(self._get_action)
        self.get_value = jax.jit(self._get_value)
        self.get_log_prob = jax.jit(self._get_log_prob)
        self.get_kl_divergence = jax.jit(self._get_kl_divergence)
    
    def _get_action_continuous(
        self, 
        params: flax.core.FrozenDict, 
        state: jnp.ndarray, 
        key: jnp.ndarray, 
        deterministic: bool = False
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """
        Obtiene una acción continua basada en el estado actual.
        
        Parámetros:
        -----------
        params : flax.core.FrozenDict
            Parámetros del actor
        state : jnp.ndarray
            Estado actual
        key : jnp.ndarray
            Llave PRNG
        deterministic : bool, opcional
            Si es True, devuelve la acción media (sin ruido) (default: False)
            
        Retorna:
        --------
        Tuple[jnp.ndarray, jnp.ndarray]
            (acción, nueva_llave)
        """
        mu, log_std = self.actor.apply(params, state)
        
        if deterministic:
            return mu, key
        
        key, subkey = jax.random.split(key)
        std = jnp.exp(log_std)
        noise = jax.random.normal(subkey, mu.shape)
        action = mu + noise * std
        
        return action, key
    
    def _get_action_discrete(
        self, 
        params: flax.core.FrozenDict, 
        state: jnp.ndarray, 
        key: jnp.ndarray, 
        deterministic: bool = False
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """
        Obtiene una acción discreta basada en el estado actual.
        
        Parámetros:
        -----------
        params : flax.core.FrozenDict
            Parámetros del actor
        state : jnp.ndarray
            Estado actual
        key : jnp.ndarray
            Llave PRNG
        deterministic : bool, opcional
            Si es True, devuelve la acción más probable (default: False)
            
        Retorna:
        --------
        Tuple[jnp.ndarray, jnp.ndarray]
            (acción, nueva_llave)
        """
        logits = self.actor.apply(params, state)
        
        if deterministic:
            return jnp.argmax(logits, axis=-1), key
        
        key, subkey = jax.random.split(key)
        action = jax.random.categorical(subkey, logits)
        
        return action, key
    
    def _get_action(
        self, 
        params: flax.core.FrozenDict, 
        state: jnp.ndarray, 
        key: jnp.ndarray, 
        deterministic: bool = False
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """
        Obtiene una acción basada en el estado actual.
        
        Parámetros:
        -----------
        params : flax.core.FrozenDict
            Parámetros del actor
        state : jnp.ndarray
            Estado actual
        key : jnp.ndarray
            Llave PRNG
        deterministic : bool, opcional
            Si es True, devuelve la acción con máxima probabilidad (default: False)
            
        Retorna:
        --------
        Tuple[jnp.ndarray, jnp.ndarray]
            (acción, nueva_llave)
        """
        if self.continuous:
            return self._get_action_continuous(params, state, key, deterministic)
        else:
            return self._get_action_discrete(params, state, key, deterministic)
    
    def _get_action_and_log_prob(
        self, 
        params: flax.core.FrozenDict, 
        state: jnp.ndarray, 
        key: jnp.ndarray
    ) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        """
        Obtiene acción y log-probabilidad para un estado dado.
        
        Parámetros:
        -----------
        params : flax.core.FrozenDict
            Parámetros del actor
        state : jnp.ndarray
            Estado actual
        key : jnp.ndarray
            Llave PRNG
            
        Retorna:
        --------
        Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]
            (acción, log_prob, nueva_llave)
        """
        action, key = self._get_action(params, state, key)
        log_prob = self._get_log_prob(params, state, action)
        
        return action, log_prob, key
    
    def _get_value(self, params: flax.core.FrozenDict, state: jnp.ndarray) -> jnp.ndarray:
        """
        Obtiene el valor estimado para un estado.
        
        Parámetros:
        -----------
        params : flax.core.FrozenDict
            Parámetros del crítico
        state : jnp.ndarray
            Estado para evaluar
            
        Retorna:
        --------
        jnp.ndarray
            Valor estimado del estado
        """
        value = self.critic.apply(params, state)
        return value.squeeze()
    
    def _get_log_prob_continuous(
        self, 
        params: flax.core.FrozenDict, 
        states: jnp.ndarray, 
        actions: jnp.ndarray
    ) -> jnp.ndarray:
        """
        Calcula el logaritmo de probabilidad para acciones continuas.
        
        Parámetros:
        -----------
        params : flax.core.FrozenDict
            Parámetros del actor
        states : jnp.ndarray
            Estados observados
        actions : jnp.ndarray
            Acciones tomadas
            
        Retorna:
        --------
        jnp.ndarray
            Log-probabilidad de las acciones
        """
        mu, log_std = self.actor.apply(params, states)
        std = jnp.exp(log_std)
        
        # Log prob para política gaussiana
        log_probs = -0.5 * (
            jnp.sum(
                jnp.square((actions - mu) / std) + 
                2 * log_std + 
                jnp.log(2.0 * jnp.pi),
                axis=1
            )
        )
        return log_probs
    
    def _get_log_prob_discrete(
        self, 
        params: flax.core.FrozenDict, 
        states: jnp.ndarray, 
        actions: jnp.ndarray
    ) -> jnp.ndarray:
        """
        Calcula el logaritmo de probabilidad para acciones discretas.
        
        Parámetros:
        -----------
        params : flax.core.FrozenDict
            Parámetros del actor
        states : jnp.ndarray
            Estados observados
        actions : jnp.ndarray
            Acciones tomadas
            
        Retorna:
        --------
        jnp.ndarray
            Log-probabilidad de las acciones
        """
        logits = self.actor.apply(params, states)
        # Log prob para política categórica
        action_masks = jax.nn.one_hot(actions.astype(jnp.int32), self.action_dim)
        log_probs = jnp.sum(
            action_masks * jax.nn.log_softmax(logits),
            axis=1
        )
        return log_probs
    
    def _get_log_prob(
        self, 
        params: flax.core.FrozenDict, 
        states: jnp.ndarray, 
        actions: jnp.ndarray
    ) -> jnp.ndarray:
        """
        Calcula el logaritmo de probabilidad de acciones bajo la política actual.
        
        Parámetros:
        -----------
        params : flax.core.FrozenDict
            Parámetros del actor
        states : jnp.ndarray
            Estados observados
        actions : jnp.ndarray
            Acciones tomadas
            
        Retorna:
        --------
        jnp.ndarray
            Log-probabilidad de las acciones
        """
        if self.continuous:
            return self._get_log_prob_continuous(params, states, actions)
        else:
            return self._get_log_prob_discrete(params, states, actions)
    
    def _get_kl_divergence_continuous(
        self, 
        params: flax.core.FrozenDict, 
        states: jnp.ndarray, 
        old_mu: jnp.ndarray, 
        old_log_std: jnp.ndarray
    ) -> jnp.ndarray:
        """
        Calcula la divergencia KL para distribuciones continuas.
        
        Parámetros:
        -----------
        params : flax.core.FrozenDict
            Parámetros del actor
        states : jnp.ndarray
            Estados para evaluar
        old_mu : jnp.ndarray
            Medias de la política antigua
        old_log_std : jnp.ndarray
            Log-std de la política antigua
            
        Retorna:
        --------
        jnp.ndarray
            Divergencia KL media
        """
        mu, log_std = self.actor.apply(params, states)
        std = jnp.exp(log_std)
        old_std = jnp.exp(old_log_std)
        
        # KL para distribuciones normales
        kl = jnp.sum(
            log_std - old_log_std + 
            (jnp.square(old_std) + jnp.square(old_mu - mu)) / (2.0 * jnp.square(std)) - 0.5,
            axis=1
        )
        return jnp.mean(kl)
    
    def _get_kl_divergence_discrete(
        self, 
        params: flax.core.FrozenDict, 
        states: jnp.ndarray, 
        old_logits: jnp.ndarray
    ) -> jnp.ndarray:
        """
        Calcula la divergencia KL para distribuciones discretas.
        
        Parámetros:
        -----------
        params : flax.core.FrozenDict
            Parámetros del actor
        states : jnp.ndarray
            Estados para evaluar
        old_logits : jnp.ndarray
            Logits de la política antigua
            
        Retorna:
        --------
        jnp.ndarray
            Divergencia KL media
        """
        logits = self.actor.apply(params, states)
        
        # KL para distribuciones categóricas
        old_logp = jax.nn.log_softmax(old_logits)
        logp = jax.nn.log_softmax(logits)
        kl = jnp.sum(
            jnp.exp(old_logp) * (old_logp - logp),
            axis=1
        )
        return jnp.mean(kl)
    
    def _get_kl_divergence(
        self, 
        params: flax.core.FrozenDict, 
        states: jnp.ndarray, 
        old_dist_params: Tuple[jnp.ndarray, ...]
    ) -> jnp.ndarray:
        """
        Calcula la divergencia KL entre la política antigua y la actual.
        
        Parámetros:
        -----------
        params : flax.core.FrozenDict
            Parámetros del actor
        states : jnp.ndarray
            Estados para evaluar
        old_dist_params : Tuple[jnp.ndarray, ...]
            Parámetros de la distribución antigua
            
        Retorna:
        --------
        jnp.ndarray
            Divergencia KL media
        """
        if self.continuous:
            old_mu, old_log_std = old_dist_params
            return self._get_kl_divergence_continuous(params, states, old_mu, old_log_std)
        else:
            old_logits = old_dist_params[0]
            return self._get_kl_divergence_discrete(params, states, old_logits)
    
    def _get_entropy(
        self, 
        params: flax.core.FrozenDict, 
        states: jnp.ndarray
    ) -> jnp.ndarray:
        """
        Calcula la entropía de la política para estados dados.
        
        Parámetros:
        -----------
        params : flax.core.FrozenDict
            Parámetros del actor
        states : jnp.ndarray
            Estados para evaluar
            
        Retorna:
        --------
        jnp.ndarray
            Entropía media de la política
        """
        if self.continuous:
            _, log_std = self.actor.apply(params, states)
            entropy = jnp.sum(
                log_std + 0.5 * jnp.log(2.0 * jnp.pi * jnp.e),
                axis=1
            )
        else:
            logits = self.actor.apply(params, states)
            probs = jax.nn.softmax(logits)
            entropy = -jnp.sum(
                probs * jnp.log(probs + 1e-8),
                axis=1
            )
        
        return jnp.mean(entropy)
    
    def compute_gae(
        self, 
        rewards: np.ndarray, 
        values: np.ndarray, 
        next_values: np.ndarray, 
        dones: np.ndarray, 
        lam: float = TRPO_CONFIG['lambda']
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Calcula el Estimador de Ventaja Generalizada (GAE).
        
        Parámetros:
        -----------
        rewards : np.ndarray
            Recompensas recibidas
        values : np.ndarray
            Valores estimados para los estados actuales
        next_values : np.ndarray
            Valores estimados para los estados siguientes
        dones : np.ndarray
            Indicadores de fin de episodio
        lam : float, opcional
            Factor lambda para GAE (default: TRPO_CONFIG['lambda'])
            
        Retorna:
        --------
        Tuple[np.ndarray, np.ndarray]
            (ventajas, retornos)
        """
        # Añadir el valor del último estado
        values = np.append(values, next_values[-1])
        
        advantages = np.zeros_like(rewards, dtype=np.float32)
        last_gae = 0
        
        for t in reversed(range(len(rewards))):
            # Si es terminal, el valor del siguiente estado es 0
            next_non_terminal = 1.0 - dones[t]
            next_value = values[t + 1]
            
            # Delta temporal para GAE
            delta = rewards[t] + self.gamma * next_value * next_non_terminal - values[t]
            
            # Calcular ventaja con GAE
            advantages[t] = last_gae = delta + self.gamma * lam * next_non_terminal * last_gae
            
        # Calcular retornos para entrenar el crítico
        returns = advantages + values[:-1]
        
        # Normalizar ventajas para reducir varianza
        advantages = (advantages - np.mean(advantages)) / (np.std(advantages) + 1e-8)
        
        return advantages, returns
    
    def fisher_vector_product(
        self, 
        params: flax.core.FrozenDict, 
        states: jnp.ndarray, 
        old_dist_params: Tuple[jnp.ndarray, ...], 
        p: jnp.ndarray
    ) -> jnp.ndarray:
        """
        Calcula el producto Fisher-vector para el método de gradiente conjugado.
        
        Parámetros:
        -----------
        params : flax.core.FrozenDict
            Parámetros del actor
        states : jnp.ndarray
            Estados
        old_dist_params : Tuple[jnp.ndarray, ...]
            Parámetros de la distribución antigua
        p : jnp.ndarray
            Vector para multiplicar con la matriz Fisher
            
        Retorna:
        --------
        jnp.ndarray
            Producto Fisher-vector
        """
        # Calcular KL
        kl_func = lambda params: self._get_kl_divergence(params, states, old_dist_params)
        
        # Calcular producto vector-hessiano
        kl_grad_func = jax.grad(kl_func)
        grads = kl_grad_func(params)
        
        # Aplanar gradientes
        flat_grads = self._flatten_params(grads)
        
        # Producto punto gradiente y p
        grad_dot_p = jnp.sum(flat_grads * p)
        
        # Calcular producto hessiano-vector
        hvp_func = jax.grad(lambda params: grad_dot_p)
        hessian_p = hvp_func(params)
        flat_hessian_p = self._flatten_params(hessian_p)
        
        # Añadir término de amortiguación para estabilidad numérica
        return flat_hessian_p + self.damping * p
    
    def conjugate_gradient(
        self, 
        params: flax.core.FrozenDict, 
        states: jnp.ndarray, 
        old_dist_params: Tuple[jnp.ndarray, ...], 
        b: jnp.ndarray
    ) -> jnp.ndarray:
        """
        Resuelve el sistema lineal Ax = b usando el método de gradiente conjugado.
        
        Parámetros:
        -----------
        params : flax.core.FrozenDict
            Parámetros del actor
        states : jnp.ndarray
            Estados
        old_dist_params : Tuple[jnp.ndarray, ...]
            Parámetros de la distribución antigua
        b : jnp.ndarray
            Vector lado derecho de la ecuación
            
        Retorna:
        --------
        jnp.ndarray
            Solución aproximada x
        """
        x = np.zeros_like(b)
        r = b.copy()
        p = r.copy()
        r_dot_r = np.dot(r, r)
        
        for _ in range(self.cg_iters):
            # Convertir a numpy para iterabilidad
            Ap = self.fisher_vector_product(params, states, old_dist_params, jnp.array(p)).numpy()
            alpha = r_dot_r / (np.dot(p, Ap) + 1e-8)
            x += alpha * p
            r -= alpha * Ap
            
            new_r_dot_r = np.dot(r, r)
            beta = new_r_dot_r / (r_dot_r + 1e-8)
            p = r + beta * p
            r_dot_r = new_r_dot_r
            
            # Criterio de parada
            if np.sqrt(r_dot_r) < 1e-6:
                break
        
        return x
    
    def _flatten_params(self, params: Union[flax.core.FrozenDict, Dict]) -> jnp.ndarray:
        """
        Convierte parámetros anidados en un vector plano.
        
        Parámetros:
        -----------
        params : Union[flax.core.FrozenDict, Dict]
            Parámetros a aplanar
            
        Retorna:
        --------
        jnp.ndarray
            Vector plano de parámetros
        """
        flat_params = []
        for _, param in jax.tree_util.tree_leaves(params):
            flat_params.append(param.reshape(-1))
        return jnp.concatenate(flat_params)
    
    def _unflatten_params(
        self, 
        flat_params: jnp.ndarray, 
        template: flax.core.FrozenDict
    ) -> flax.core.FrozenDict:
        """
        Convierte un vector plano en parámetros anidados usando una plantilla.
        
        Parámetros:
        -----------
        flat_params : jnp.ndarray
            Vector plano de parámetros
        template : flax.core.FrozenDict
            Plantilla con la estructura anidada deseada
            
        Retorna:
        --------
        flax.core.FrozenDict
            Parámetros con la estructura anidada
        """
        shapes = [(k, p.shape) for k, p in jax.tree_util.tree_leaves_with_path(template)]
        
        new_params = {}
        start_idx = 0
        for path, shape in shapes:
            size = np.prod(shape)
            param = flat_params[start_idx:start_idx + size].reshape(shape)
            
            # Crear estructura anidada
            current = new_params
            for key in path[:-1]:
                if key not in current:
                    current[key] = {}
                current = current[key]
            current[path[-1]] = param
            
            start_idx += size
            
        return flax.core.freeze(new_params)
    
    def update_policy(
        self, 
        states: np.ndarray, 
        actions: np.ndarray, 
        advantages: np.ndarray
    ) -> Dict[str, float]:
        """
        Actualiza la política utilizando el método TRPO.
        
        Parámetros:
        -----------
        states : np.ndarray
            Estados
        actions : np.ndarray
            Acciones
        advantages : np.ndarray
            Ventajas
            
        Retorna:
        --------
        Dict[str, float]
            Métricas de actualización
        """
        start_time = time.time()
        
        # Convertir a arrays de JAX
        states = jnp.array(states)
        actions = jnp.array(actions)
        advantages = jnp.array(advantages)
        
        # Guardar parámetros de distribución antiguos para KL
        if self.continuous:
            old_mu, old_log_std = self.actor.apply(self.state.actor_params, states)
            old_dist_params = (old_mu, old_log_std)
        else:
            old_logits = self.actor.apply(self.state.actor_params, states)
            old_dist_params = (old_logits,)
        
        # Calcular log probs antiguos
        old_log_probs = self._get_log_prob(self.state.actor_params, states, actions)
        
        # Calcular gradiente del objetivo surrogate
        def surrogate_loss(params):
            log_probs = self._get_log_prob(params, states, actions)
            ratio = jnp.exp(log_probs - old_log_probs)
            return -jnp.mean(ratio * advantages)
        
        # Calcular gradiente
        policy_grad = jax.grad(surrogate_loss)(self.state.actor_params)
        flat_policy_grad = self._flatten_params(policy_grad)
        
        # Parámetros actuales
        old_params = self.state.actor_params
        
        # Calcular dirección de actualización usando gradiente conjugado
        step_direction = self.conjugate_gradient(
            old_params, 
            states, 
            old_dist_params, 
            -flat_policy_grad.numpy()
        )
        step_direction = jnp.array(step_direction)
        
        # Calcular tamaño de paso
        fvp = self.fisher_vector_product(old_params, states, old_dist_params, step_direction)
        shs = 0.5 * jnp.dot(step_direction, fvp)
        lm = jnp.sqrt(shs / self.delta)
        full_step = step_direction / lm
        
        # Guardar valor de pérdida actual
        current_loss = surrogate_loss(old_params)
        
        # Backtracking line search
        for i in range(self.backtrack_iters):
            # Probar nuevos parámetros
            step_size = self.backtrack_coeff**i
            flat_old_params = self._flatten_params(old_params)
            new_flat_params = flat_old_params + step_size * full_step
            new_params = self._unflatten_params(new_flat_params, old_params)
            
            # Calcular nuevo surrogate loss
            new_loss = surrogate_loss(new_params)
            
            # Calcular nueva KL divergence
            kl = self._get_kl_divergence(new_params, states, old_dist_params)
            
            improvement = current_loss - new_loss
            
            # Verificar si mejora y cumple restricción KL
            if improvement > 0 and kl < self.delta:
                print(f"Policy update: iter {i}, improvement: {improvement:.6f}, kl: {kl:.6f}")
                self.state.actor_params = new_params
                break
                
            # Si llegamos a la última iteración, mantener parámetros originales
            if i == self.backtrack_iters - 1:
                print("Line search failed. Keeping old parameters.")
        
        # Calcular entropía después de la actualización
        entropy = self._get_entropy(self.state.actor_params, states)
        
        # Actualizar métricas
        self.metrics['policy_loss'] = float(current_loss)
        self.metrics['kl_divergence'] = float(kl)
        self.metrics['entropy'] = float(entropy)
        
        elapsed_time = time.time() - start_time
        return {
            'policy_loss': float(current_loss),
            'kl_divergence': float(kl),
            'entropy': float(entropy),
            'elapsed_time': elapsed_time
        }
    
    def update_value_step(
        self, 
        critic_state: CriticTrainState, 
        batch_states: jnp.ndarray, 
        batch_returns: jnp.ndarray
    ) -> Tuple[CriticTrainState, jnp.ndarray]:
        """
        Realiza un paso de actualización de la red de valor.
        
        Parámetros:
        -----------
        critic_state : CriticTrainState
            Estado actual del crítico
        batch_states : jnp.ndarray
            Lote de estados
        batch_returns : jnp.ndarray
            Lote de retornos
            
        Retorna:
        --------
        Tuple[CriticTrainState, jnp.ndarray]
            (nuevo_estado_critico, perdida)
        """
        def value_loss_fn(params):
            values = self.critic.apply(params, batch_states).squeeze()
            return jnp.mean(jnp.square(values - batch_returns))
        
        # Calcular gradiente y pérdida
        grad_fn = jax.value_and_grad(value_loss_fn)
        loss, grads = grad_fn(critic_state.params)
        
        # Actualizar crítico
        critic_state = critic_state.apply_gradients(grads=grads)
        
        return critic_state, loss
    
    def update_value(
        self, 
        states: np.ndarray, 
        returns: np.ndarray, 
        epochs: int = TRPO_CONFIG['value_epochs'], 
        batch_size: int = TRPO_CONFIG['batch_size']
    ) -> Dict[str, float]:
        """
        Actualiza la red de valor (crítico) usando descenso de gradiente.
        
        Parámetros:
        -----------
        states : np.ndarray
            Estados
        returns : np.ndarray
            Retornos objetivo
        epochs : int, opcional
            Número de épocas para entrenar (default: TRPO_CONFIG['value_epochs'])
        batch_size : int, opcional
            Tamaño de lote (default: TRPO_CONFIG['batch_size'])
            
        Retorna:
        --------
        Dict[str, float]
            Estadísticas de actualización
        """
        start_time = time.time()
        
        # Convertir a arrays de JAX
        states = jnp.array(states)
        returns = jnp.array(returns)
        
        dataset_size = len(states)
        indices = np.arange(dataset_size)
        losses = []
        
        # Create a Generator instance with a fixed seed for reproducibility
        seed = TRPO_CONFIG.get('seed', 42)
        rng = np.random.default_rng(seed)
        
        for _ in range(epochs):
            # Mezclar datos en cada época
            rng.shuffle(indices)
            
            for start_idx in range(0, dataset_size, batch_size):
                # Obtener lote
                idx = indices[start_idx:min(start_idx + batch_size, dataset_size)]
                batch_states = jnp.take(states, idx, axis=0)
                batch_returns = jnp.take(returns, idx, axis=0)
                
                # Actualizar red de valor
                self.state.critic_state, loss = self.update_value_step(
                    self.state.critic_state, 
                    batch_states, 
                    batch_returns
                )
                
                losses.append(float(loss))
        
        # Actualizar métrica
        mean_loss = np.mean(losses)
        self.metrics['value_loss'] = mean_loss
        
        elapsed_time = time.time() - start_time
        return {
            'value_loss': mean_loss,
            'elapsed_time': elapsed_time
        }
    
    def get_action(
        self, 
        state: np.ndarray, 
        deterministic: bool = False
    ) -> np.ndarray:
        """
        Obtiene una acción basada en el estado actual.
        
        Parámetros:
        -----------
        state : np.ndarray
            Estado actual
        deterministic : bool, opcional
            Si es True, devuelve la acción con máxima probabilidad (default: False)
            
        Retorna:
        --------
        np.ndarray
            Una acción muestreada de la distribución de política
        """
        # Convertir a tensor y añadir dimensión de batch si es necesario
        if state.ndim == 1:
            state = jnp.expand_dims(state, axis=0)
        else:
            state = jnp.array(state)
        
        action, key = self._get_action(
            self.state.actor_params, 
            state, 
            self.state.key, 
            deterministic
        )
        
        # Actualizar llave
        self.state.key = key
        
        # Convertir a numpy y eliminar dimensión de batch si es necesaria
        action_np = np.array(action)
        if action_np.shape[0] == 1:
            action_np = action_np[0]
            
        return action_np
    
    def get_value(self, state: np.ndarray) -> float:
        """
        Obtiene el valor estimado para un estado.
        
        Parámetros:
        -----------
        state : np.ndarray
            Estado para evaluar
            
        Retorna:
        --------
        float
            Valor estimado del estado
        """
        # Convertir a tensor y añadir dimensión de batch si es necesario
        if state.ndim == 1:
            state = jnp.expand_dims(state, axis=0)
        else:
            state = jnp.array(state)
        
        value = self._get_value(self.state.critic_state.params, state)
        
        # Si es un único estado, retornar un escalar
        if state.shape[0] == 1:
            return float(value[0])
        return np.array(value)
    
    def collect_trajectories(
        self, 
        env: Any, 
        min_steps: int = TRPO_CONFIG['min_steps_per_update'], 
        render: bool = False
    ) -> Dict[str, np.ndarray]:
        """
        Recolecta trayectorias ejecutando la política actual en el entorno.
        
        Parámetros:
        -----------
        env : Any
            Entorno de OpenAI Gym o compatible
        min_steps : int, opcional
            Mínimo número de pasos antes de actualizar (default: TRPO_CONFIG['min_steps_per_update'])
        render : bool, opcional
            Si renderizar el entorno o no (default: False)
            
        Retorna:
        --------
        Dict[str, np.ndarray]
            Datos recolectados
        """
        states, actions, rewards, dones, next_states, values = [], [], [], [], [], []
        
        state, _ = env.reset()
        done = False
        episode_reward = 0
        episode_length = 0
        episode_rewards = []
        episode_lengths = []
        total_steps = 0
        
        # Recolectar hasta alcanzar min_steps
        while total_steps < min_steps:
            if render:
                env.render()
            
            # Guardar estado actual
            states.append(state)
            
            # Seleccionar acción
            action = self.get_action(state)
            actions.append(action)
            
            # Calcular valor del estado
            value = self.get_value(state)
            values.append(value)
            
            # Ejecutar acción
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            
            # Guardar resultado
            rewards.append(reward)
            dones.append(done)
            next_states.append(next_state)
            
            # Actualizar
            state = next_state
            episode_reward += reward
            episode_length += 1
            total_steps += 1
            
            # Si episodio termina, resetear
            if done:
                print(f"Episodio terminado con recompensa: {episode_reward}, longitud: {episode_length}")
                episode_rewards.append(episode_reward)
                episode_lengths.append(episode_length)
                episode_reward = 0
                episode_length = 0
                state, _ = env.reset()
                done = False
        
        # Calcular ventajas y retornos
        next_values = [self.get_value(next_state) for next_state in next_states]
        advantages, returns = self.compute_gae(
            np.array(rewards), 
            np.array(values), 
            np.array(next_values), 
            np.array(dones)
        )
        
        return {
            'states': np.array(states, dtype=np.float32),
            'actions': np.array(actions, dtype=np.float32 if self.continuous else np.int32),
            'rewards': np.array(rewards, dtype=np.float32),
            'dones': np.array(dones, dtype=np.float32),
            'next_states': np.array(next_states, dtype=np.float32),
            'values': np.array(values, dtype=np.float32),
            'advantages': advantages,
            'returns': returns,
            'episode_rewards': episode_rewards,
            'episode_lengths': episode_lengths,
            'total_steps': total_steps
        }
    
    def train(
        self, 
        env: Any, 
        iterations: int = TRPO_CONFIG['iterations'], 
        min_steps_per_update: int = TRPO_CONFIG['min_steps_per_update'],
        render: bool = False, 
        evaluate_interval: int = TRPO_CONFIG['evaluate_interval']
    ) -> Dict[str, List[float]]:
        """
        Entrena el agente TRPO.
        
        Parámetros:
        -----------
        env : Any
            Entorno para entrenar
        iterations : int, opcional
            Número de iteraciones de entrenamiento (default: TRPO_CONFIG['iterations'])
        min_steps_per_update : int, opcional
            Mínimo de pasos antes de actualizar la política (default: TRPO_CONFIG['min_steps_per_update'])
        render : bool, opcional
            Si renderizar el entorno o no (default: False)
        evaluate_interval : int, opcional
            Intervalos para evaluación (default: TRPO_CONFIG['evaluate_interval'])
            
        Retorna:
        --------
        Dict[str, List[float]]
            Historia de entrenamiento
        """
        history = {
            'iterations': [],
            'policy_losses': [],
            'value_losses': [],
            'kl_divergences': [],
            'entropies': [],
            'mean_episode_rewards': [],
            'mean_episode_lengths': [],
            'steps_per_iteration': [],
            'evaluation_rewards': []
        }
        
        for i in range(iterations):
            print(f"\nIteración {i+1}/{iterations}")
            start_time = time.time()
            
            # Recolectar experiencias
            data = self.collect_trajectories(env, min_steps_per_update, render)
            
            # Actualizar política
            policy_stats = self.update_policy(data['states'], data['actions'], data['advantages'])
            
            # Actualizar red de valor
            value_stats = self.update_value(data['states'], data['returns'])
            
            # Calcular métricas
            mean_episode_reward = np.mean(data['episode_rewards']) if data['episode_rewards'] else 0
            mean_episode_length = np.mean(data['episode_lengths']) if data['episode_lengths'] else 0
            
            # Guardar historia
            history['iterations'].append(i)
            history['policy_losses'].append(policy_stats['policy_loss'])
            history['value_losses'].append(value_stats['value_loss'])
            history['kl_divergences'].append(policy_stats['kl_divergence'])
            history['entropies'].append(policy_stats['entropy'])
            history['mean_episode_rewards'].append(mean_episode_reward)
            history['mean_episode_lengths'].append(mean_episode_length)
            history['steps_per_iteration'].append(data['total_steps'])
            
            # Evaluar
            if (i + 1) % evaluate_interval == 0:
                eval_reward = self.evaluate(env, episodes=5)
                history['evaluation_rewards'].append(eval_reward)
                print(f"Evaluación: Recompensa media = {eval_reward:.2f}")
            
            # Mostrar estadísticas
            elapsed_time = time.time() - start_time
            print(f"Tiempo total: {elapsed_time:.2f}s")
            print(f"KL Divergence: {policy_stats['kl_divergence']:.6f}")
            print(f"Entropía: {policy_stats['entropy']:.6f}")
            print(f"Recompensa media: {mean_episode_reward:.2f}")
            print(f"Longitud media: {mean_episode_length:.2f}")
        
        return history
    
    def evaluate(
        self, 
        env: Any, 
        episodes: int = 10, 
        render: bool = False
    ) -> float:
        """
        Evalúa el agente TRPO en el entorno.
        
        Parámetros:
        -----------
        env : Any
            Entorno para evaluar
        episodes : int, opcional
            Número de episodios (default: 10)
        render : bool, opcional
            Si renderizar el entorno o no (default: False)
            
        Retorna:
        --------
        float
            Recompensa media
        """
        rewards = []
        
        for episode in range(episodes):
            state, _ = env.reset()
            done = False
            episode_reward = 0
            
            while not done:
                if render:
                    env.render()
                
                # Usar política determinística para evaluación
                action = self.get_action(state, deterministic=True)
                
                # Ejecutar acción
                next_state, reward, terminated, truncated, _ = env.step(action)
                done = terminated or truncated
                
                # Actualizar
                state = next_state
                episode_reward += reward
            
            rewards.append(episode_reward)
            print(f"Episodio {episode+1}: Recompensa = {episode_reward:.2f}")
        
        return np.mean(rewards)
    
    def save_model(self, actor_path: str, critic_path: str) -> None:
        """
        Guarda los modelos del actor y crítico.
        
        Parámetros:
        -----------
        actor_path : str
            Ruta para guardar el actor
        critic_path : str
            Ruta para guardar el crítico
        """
        # Guardar actor
        with open(actor_path, 'wb') as f:
            f.write(flax.serialization.to_bytes(self.state.actor_params))
        
        # Guardar crítico
        with open(critic_path, 'wb') as f:
            f.write(flax.serialization.to_bytes(self.state.critic_state.params))
            
        print(f"Modelo guardado en {actor_path} y {critic_path}")
    
    def load_model(self, actor_path: str, critic_path: str) -> None:
        """
        Carga los modelos del actor y crítico.
        
        Parámetros:
        -----------
        actor_path : str
            Ruta para cargar el actor
        critic_path : str
            Ruta para cargar el crítico
        """
        try:
            # Cargar actor
            with open(actor_path, 'rb') as f:
                actor_bytes = f.read()
                actor_params = flax.serialization.from_bytes(self.state.actor_params, actor_bytes)
            
            # Cargar crítico
            with open(critic_path, 'rb') as f:
                critic_bytes = f.read()
                critic_params = flax.serialization.from_bytes(self.state.critic_state.params, critic_bytes)
            
            # Actualizar parámetros
            self.state.actor_params = actor_params
            self.state.critic_state = self.state.critic_state.replace(params=critic_params)
            
            print(f"Modelo cargado desde {actor_path} y {critic_path}")
        except Exception as e:
            print(f"Error al cargar los modelos: {str(e)}")
    
    def visualize_training(self, history: Dict[str, List[float]], smoothing_window: int = 10) -> None:
        """
        Visualiza la historia de entrenamiento.
        
        Parámetros:
        -----------
        history : Dict[str, List[float]]
            Historia de entrenamiento
        smoothing_window : int, opcional
            Tamaño de ventana para suavizado (default: 10)
        """
        import matplotlib.pyplot as plt
        
        # Constantes para etiquetas
        LABEL_ITERATION = 'Iteración'
        LABEL_ENTROPY = 'Entropía'
        
        # Función para suavizar datos
        def smooth(data, window_size):
            if len(data) < window_size:
                return data
            kernel = np.ones(window_size) / window_size
            return np.convolve(data, kernel, mode='valid')
        
        # Crear figura con múltiples subplots
        _, axs = plt.subplots(3, 2, figsize=(15, 12))
        
        # 1. Recompensas de episodio
        axs[0, 0].plot(history['mean_episode_rewards'], alpha=0.3, color='blue')
        if len(history['mean_episode_rewards']) > smoothing_window:
            axs[0, 0].plot(
                range(smoothing_window-1, len(history['mean_episode_rewards'])),
                smooth(history['mean_episode_rewards'], smoothing_window),
                color='blue',
                label='Suavizado'
            )
        axs[0, 0].set_title('Recompensa Media por Episodio')
        axs[0, 0].set_xlabel(LABEL_ITERATION)
        axs[0, 0].set_ylabel('Recompensa')
        axs[0, 0].grid(alpha=0.3)
        axs[0, 0].legend()
        
        # 2. Recompensas de evaluación
        if history['evaluation_rewards']:
            eval_interval = len(history['iterations']) // len(history['evaluation_rewards'])
            x_eval = [i * eval_interval for i in range(len(history['evaluation_rewards']))]
            axs[0, 1].plot(x_eval, history['evaluation_rewards'], color='green', marker='o')
            axs[0, 1].set_title('Recompensa de Evaluación')
            axs[0, 1].set_xlabel(LABEL_ITERATION)
            axs[0, 1].set_ylabel('Recompensa Media')
            axs[0, 1].grid(alpha=0.3)
        
        # 3. Pérdida de política
        axs[1, 0].plot(history['policy_losses'], alpha=0.3, color='red')
        if len(history['policy_losses']) > smoothing_window:
            axs[1, 0].plot(
                range(smoothing_window-1, len(history['policy_losses'])),
                smooth(history['policy_losses'], smoothing_window),
                color='red',
                label='Suavizado'
            )
        axs[1, 0].set_title('Pérdida de Política')
        axs[1, 0].set_xlabel(LABEL_ITERATION)
        axs[1, 0].set_ylabel('Pérdida')
        axs[1, 0].grid(alpha=0.3)
        axs[1, 0].legend()
        
        # 4. Pérdida de valor
        axs[1, 1].plot(history['value_losses'], alpha=0.3, color='purple')
        if len(history['value_losses']) > smoothing_window:
            axs[1, 1].plot(
                range(smoothing_window-1, len(history['value_losses'])),
                smooth(history['value_losses'], smoothing_window),
                color='purple',
                label='Suavizado'
            )
        axs[1, 1].set_title('Pérdida de Valor')
        axs[1, 1].set_xlabel(LABEL_ITERATION)
        axs[1, 1].set_ylabel('Pérdida')
        axs[1, 1].grid(alpha=0.3)
        axs[1, 1].legend()
        
        # 5. KL Divergence
        axs[2, 0].plot(history['kl_divergences'], alpha=0.3, color='orange')
        if len(history['kl_divergences']) > smoothing_window:
            axs[2, 0].plot(
                range(smoothing_window-1, len(history['kl_divergences'])),
                smooth(history['kl_divergences'], smoothing_window),
                color='orange',
                label='Suavizado'
            )
        axs[2, 0].set_title('KL Divergence')
        axs[2, 0].set_xlabel(LABEL_ITERATION)
        axs[2, 0].set_ylabel('KL')
        axs[2, 0].grid(alpha=0.3)
        axs[2, 0].legend()
        
        # 6. Entropía
        axs[2, 1].plot(history['entropies'], alpha=0.3, color='brown')
        if len(history['entropies']) > smoothing_window:
            axs[2, 1].plot(
                range(smoothing_window-1, len(history['entropies'])),
                smooth(history['entropies'], smoothing_window),
                color='brown',
                label='Suavizado'
            )
        axs[2, 1].set_title(LABEL_ENTROPY)
        axs[2, 1].set_xlabel(LABEL_ITERATION)
        axs[2, 1].set_ylabel(LABEL_ENTROPY)
        axs[2, 1].grid(alpha=0.3)
        axs[2, 1].legend()
        axs[2, 1].legend()
        
        plt.tight_layout()
        plt.show()
class TRPOWrapper:
    """
    Wrapper para el algoritmo TRPO que implementa la interfaz compatible con modelos de aprendizaje profundo.
    """
    
    def __init__(
        self, 
        trpo_agent: TRPO,
        cgm_shape: Tuple[int, ...],
        other_features_shape: Tuple[int, ...],
    ) -> None:
        """
        Inicializa el wrapper para TRPO.
        
        Parámetros:
        -----------
        trpo_agent : TRPO
            Agente TRPO a utilizar
        cgm_shape : Tuple[int, ...]
            Forma de entrada para datos CGM
        other_features_shape : Tuple[int, ...]
            Forma de entrada para otras características
        """
        self.trpo_agent = trpo_agent
        self.cgm_shape = cgm_shape
        self.other_features_shape = other_features_shape
        
        # Preprocesadores para convertir entradas a espacio de estado
        self.cgm_encoder = None
        self.other_encoder = None
        
        # Configurar funciones de codificación
        self._setup_encoders()
        
        # Historial de entrenamiento
        self.history = {
            'loss': [],
            'val_loss': [],
            'policy_loss': [],
            'value_loss': [],
            'kl': []
        }
    
    def _setup_encoders(self) -> None:
        """
        Configura las funciones de codificación para procesar datos de entrada.
        """
        # Inicializa parámetros para la codificación
        self.key = jax.random.key(42)
        
        # Funciones de codificación con hparams
        def init_encoder_params(key, input_shape, hidden_sizes):
            """Inicializa parámetros para un codificador MLP."""
            sizes = [np.prod(input_shape)] + hidden_sizes
            keys = jax.random.split(key, len(sizes))
            
            params = []
            for i in range(len(sizes) - 1):
                w_key, b_key = jax.random.split(keys[i])
                w = jax.random.normal(w_key, (sizes[i], sizes[i + 1])) * 0.01
                b = jax.random.normal(b_key, (sizes[i + 1],)) * 0.01
                params.append((w, b))
            
            return params
        
        # Inicializar parámetros para ambos codificadores
        self.key, key1, key2 = jax.random.split(self.key, 3)
        self.cgm_params = init_encoder_params(key1, self.cgm_shape[1:], [32, 16])
        self.other_params = init_encoder_params(key2, self.other_features_shape[1:], [16, 8])
        
        # Compilar funciones con JIT para mayor rendimiento
        self.cgm_encoder = jax.jit(self._create_encoder_fn(self.cgm_params))
        self.other_encoder = jax.jit(self._create_encoder_fn(self.other_params))
    
    def _create_encoder_fn(self, params: List) -> Callable:
        """
        Crea una función de codificación JAX.
        
        Parámetros:
        -----------
        params : List
            Parámetros del codificador
            
        Retorna:
        --------
        Callable
            Función de codificación JIT-compilada
        """
        def encoder_fn(x):
            x = x.reshape((x.shape[0], -1))  # Aplanar entrada
            for i, (w, b) in enumerate(params):
                x = jnp.dot(x, w) + b
                if i < len(params) - 1:
                    x = jnp.tanh(x)  # Activación no lineal
            return x
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
        Realiza predicciones con el modelo TRPO.
        
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
        
        # Codificar entradas a representación de estado
        cgm_encoded = self.cgm_encoder(cgm_data)
        other_encoded = self.other_encoder(other_features)
        
        # Combinar características en representación de estado
        states = jnp.concatenate([cgm_encoded, other_encoded], axis=-1)
        
        # Usar política del agente para predecir acciones (dosis)
        batch_size = states.shape[0]
        actions = np.zeros((batch_size, 1))
        
        # Predecir para cada ejemplo en el batch
        for i in range(batch_size):
            state = np.array(states[i])
            # Usar política determinística (media de la distribución)
            action = self.trpo_agent.get_action(state, deterministic=True)
            actions[i] = action
        
        return actions
    
    def fit(
        self, 
        x: List[jnp.ndarray], 
        y: jnp.ndarray, 
        validation_data: Optional[Tuple] = None, 
        epochs: int = 1,
        batch_size: int = 32,
        callbacks: list = None,
        verbose: int = 0
    ) -> Dict:
        """
        Entrena el modelo TRPO en los datos proporcionados.
        
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
            print("Entrenando modelo TRPO...")
        
        # Crear entorno simulado para RL a partir de los datos
        env = self._create_training_environment(x[0], x[1], y)
        
        # Entrenar el agente TRPO
        train_metrics = self.trpo_agent.train(
            env=env,
            iterations=epochs,
            batch_size=batch_size
        )
        
        # Actualizar historial con métricas del entrenamiento
        for key, values in train_metrics.items():
            if key in self.history:
                self.history[key].extend(values)
        
        # Evaluar en datos de validación si se proporcionan
        if validation_data:
            val_x, val_y = validation_data
            val_preds = self.predict(val_x)
            val_loss = jnp.mean((val_preds.flatten() - val_y) ** 2)
            self.history['val_loss'].append(float(val_loss))
        
        if verbose > 0:
            print(f"Entrenamiento completado en {len(train_metrics.get('iterations', []))} iteraciones")
        
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
        # Crear entorno personalizado para TRPO
        class InsulinDosingEnv:
            def __init__(self, cgm, features, targets, model_wrapper):
                self.cgm = np.array(cgm)
                self.features = np.array(features)
                self.targets = np.array(targets)
                self.model = model_wrapper
                self.rng = np.random.Generator(np.random.PCG64(42))
                self.current_idx = 0
                self.max_idx = len(targets) - 1
                
                # Para compatibilidad con RL
                self.observation_space = SimpleNamespace(
                    shape=(model_wrapper.trpo_agent.state_dim,),
                    low=np.full((model_wrapper.trpo_agent.state_dim,), -10.0),
                    high=np.full((model_wrapper.trpo_agent.state_dim,), 10.0)
                )
                
                if model_wrapper.trpo_agent.continuous:
                    self.action_space = SimpleNamespace(
                        shape=(model_wrapper.trpo_agent.action_dim,),
                        low=np.zeros(model_wrapper.trpo_agent.action_dim),
                        high=np.ones(model_wrapper.trpo_agent.action_dim) * 15.0,  # Max 15 unidades
                        sample=self._sample_action
                    )
                else:
                    self.action_space = SimpleNamespace(
                        n=model_wrapper.trpo_agent.action_dim,
                        sample=lambda: self.rng.integers(0, model_wrapper.trpo_agent.action_dim)
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
                # Convertir acción a dosis según el tipo de espacio
                if self.model.trpo_agent.continuous:
                    dose = float(action[0])  # Ya está en escala correcta
                else:
                    # Para acciones discretas, mapear a rango de dosis
                    dose = action / (self.model.trpo_agent.action_dim - 1) * 15.0
                
                # Calcular recompensa como negativo del error absoluto
                target = self.targets[self.current_idx]
                reward = -abs(dose - target)
                
                # Avanzar al siguiente ejemplo
                self.current_idx = (self.current_idx + 1) % self.max_idx
                
                # Obtener próximo estado
                next_state = self._get_state()
                
                # Episodio siempre termina después de una acción
                done = True
                truncated = False
                
                return next_state, reward, done, truncated, {}
            
            def _get_state(self):
                """Obtiene el estado codificado para el ejemplo actual."""
                # Obtener datos actuales
                cgm_batch = self.cgm[self.current_idx:self.current_idx+1]
                features_batch = self.features[self.current_idx:self.current_idx+1]
                
                # Codificar a estado
                cgm_encoded = self.model.cgm_encoder(cgm_batch)
                other_encoded = self.model.other_encoder(features_batch)
                
                # Combinar características 
                state = np.concatenate([cgm_encoded[0], other_encoded[0]])
                
                return state
        
        # Importar lo necesario para el entorno
        from types import SimpleNamespace
        
        # Crear y devolver el entorno
        return InsulinDosingEnv(cgm_data, other_features, targets, self)
    
    def save(self, filepath: str) -> None:
        """
        Guarda el modelo TRPO en un archivo.
        
        Parámetros:
        -----------
        filepath : str
            Ruta donde guardar el modelo
        """
        # Guardar parámetros del modelo
        model_data = {
            'trpo_params': self.trpo_agent.get_params(),
            'cgm_params': self.cgm_params,
            'other_params': self.other_params,
            'cgm_shape': self.cgm_shape,
            'other_features_shape': self.other_features_shape,
            'state_dim': self.trpo_agent.state_dim,
            'action_dim': self.trpo_agent.action_dim,
            'continuous': self.trpo_agent.continuous
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
        
        print(f"Modelo guardado en {filepath}")
    
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
            'state_dim': self.trpo_agent.state_dim,
            'action_dim': self.trpo_agent.action_dim,
            'continuous': self.trpo_agent.continuous
        }

def create_trpo_model(cgm_shape: Tuple[int, ...], other_features_shape: Tuple[int, ...]) -> TRPOWrapper:
    """
    Crea un modelo basado en TRPO (Trust Region Policy Optimization) para predicción de dosis de insulina.
    
    Parámetros:
    -----------
    cgm_shape : Tuple[int, ...]
        Forma de los datos CGM (batch_size, time_steps, features)
    other_features_shape : Tuple[int, ...]
        Forma de otras características (batch_size, n_features)
        
    Retorna:
    --------
    TRPOWrapper
        Modelo TRPO que implementa la interfaz compatible con modelos de aprendizaje profundo
    """
    # Configurar el espacio de estados y acciones
    state_dim = 64  # Dimensión del espacio de estado latente
    action_dim = 1  # Una dimensión para dosis continua
    continuous = True  # TRPO funciona bien con espacios continuos
    
    # Crear agente TRPO
    trpo_agent = TRPO(
        state_dim=state_dim,
        action_dim=action_dim,
        continuous=continuous,
        gamma=TRPO_CONFIG['gamma'],
        delta=TRPO_CONFIG['delta'],  # Restricción KL para la región de confianza
        vf_coeff=TRPO_CONFIG['vf_coeff'],
        cg_iters=TRPO_CONFIG['cg_iters'],
        max_backtrack=TRPO_CONFIG['max_backtrack'],
        backtrack_ratio=TRPO_CONFIG['backtrack_ratio'],
        hidden_sizes=TRPO_CONFIG['hidden_units'],
        seed=TRPO_CONFIG.get('seed', 42)
    )
    
    # Crear y devolver wrapper para compatibilidad con la interfaz de modelos
    return TRPOWrapper(trpo_agent, cgm_shape, other_features_shape)
