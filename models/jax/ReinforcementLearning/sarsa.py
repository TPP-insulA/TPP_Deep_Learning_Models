import os
import sys
import time
import pickle
import matplotlib.pyplot as plt
import jax
import jax.numpy as jnp
import numpy as np
from types import SimpleNamespace
from typing import Dict, List, Tuple, Optional, Union, Any, NamedTuple, Callable
from functools import partial

PROJECT_ROOT = os.path.abspath(os.getcwd())
sys.path.append(PROJECT_ROOT) 

from config.models_config import SARSA_CONFIG
from custom.rl_model_wrapper import RLModelWrapperJAX

# Constantes para rutas de figuras y etiquetas comunes
FIGURES_DIR = os.path.join(PROJECT_ROOT, "figures", "jax", "sarsa")
CONST_EPISODE_LABEL = "Episodio"
CONST_REWARD_LABEL = "Recompensa" 
CONST_STEPS_LABEL = "Pasos"
CONST_EPSILON_LABEL = "Epsilon"
CONST_ORIGINAL_LABEL = "Original"
CONST_SMOOTHED_LABEL = "Suavizado"

# Crear directorio para figuras si no existe
os.makedirs(FIGURES_DIR, exist_ok=True)

class SARSAState(NamedTuple):
    """Estado interno para agente SARSA con JAX (inmutabilidad)"""
    q_table: jnp.ndarray
    rng_key: jnp.ndarray
    episode_rewards: List[float]
    episode_lengths: List[int]
    epsilon_history: List[float]
    epsilon: float


class SARSA:
    """
    Implementación del algoritmo SARSA (State-Action-Reward-State-Action) con JAX.
    
    SARSA es un algoritmo de aprendizaje por refuerzo on-policy que actualiza
    los valores Q basándose en la política actual, incluyendo la exploración.
    """
    
    def __init__(
        self, 
        env: Any, 
        config: Optional[Dict] = None,
        seed: int = 42,
        cgm_shape: Optional[Tuple[int, ...]] = None,
        other_features_shape: Optional[Tuple[int, ...]] = None
    ) -> None:
        """
        Inicializa el agente SARSA.
        
        Parámetros:
        -----------
        env : Any
            Entorno de OpenAI Gym o compatible
        config : Optional[Dict], opcional
            Configuración personalizada (default: None)
        seed : int, opcional
            Semilla para reproducibilidad (default: 42)
        cgm_shape : Optional[Tuple[int, ...]], opcional
            Forma de los datos CGM (puede ser necesario para mapeo estado-observación) (default: None)
        other_features_shape : Optional[Tuple[int, ...]], opcional
            Forma de otras características (puede ser necesario para mapeo estado-observación) (default: None)
        """
        self.env = env
        self.config = config or SARSA_CONFIG
        self.cgm_shape = cgm_shape
        self.other_features_shape = other_features_shape
        
        # Inicializar parámetros básicos
        self._init_learning_params()
        self._validate_action_space()
        
        # Configurar llave para generación de números aleatorios
        self.rng_key = jax.random.PRNGKey(seed)
        
        # Configurar espacio de estados y tabla Q
        self.discrete_state_space = hasattr(env.observation_space, 'n')
        
        if self.discrete_state_space:
            self._setup_discrete_state_space()
        else:
            self._setup_continuous_state_space()
        
        # Inicializar estado mutable para JAX (con funcionalidad inmutable)
        self.state = SARSAState(
            q_table=self.q_table,
            rng_key=self.rng_key,
            episode_rewards=[],
            episode_lengths=[],
            epsilon_history=[],
            epsilon=self.epsilon
        )
        
        # Compilar funciones para mejor rendimiento
        self._update_q_value_fn = jax.jit(self._update_q_value_fn)
        self._get_action_fn = jax.jit(self._get_action_fn, static_argnums=(3,))
        
        # Historial para RLModelWrapperJAX
        self.history = {'loss': [], 'avg_reward': [], 'epsilon': []}
    
    def setup(self, rng_key: jax.random.PRNGKey) -> Any:
        """
        Inicializa el agente para interactuar con el entorno.
        Compatible con RLModelWrapperJAX.
        
        Parámetros:
        -----------
        rng_key : jax.random.PRNGKey
            Clave para generación aleatoria.
            
        Retorna:
        --------
        Any
            El estado del agente inicializado (self).
        """
        # Actualizar la clave RNG del agente
        self.rng_key = rng_key
        self.state = self.state._replace(rng_key=rng_key)
        
        return self
    
    def _init_learning_params(self) -> None:
        """
        Inicializa los parámetros de aprendizaje desde la configuración.
        """
        self.alpha = self.config['learning_rate']
        self.gamma = self.config['gamma']
        self.epsilon = self.config['epsilon_start']
        self.epsilon_min = self.config['epsilon_end']
        self.epsilon_decay = self.config['epsilon_decay']
        self.decay_type = self.config['epsilon_decay_type']
    
    def _validate_action_space(self) -> None:
        """
        Valida que el espacio de acción sea compatible con SARSA.
        
        Genera:
        -------
        ValueError
            Si el espacio de acción no es discreto
        """
        if not hasattr(self.env.action_space, 'n'):
            raise ValueError("SARSA requiere un espacio de acción discreto")
        
        self.action_space_size = self.env.action_space.n
    
    def _setup_discrete_state_space(self) -> None:
        """
        Configura SARSA para un espacio de estados discreto.
        """
        self.state_space_size = self.env.observation_space.n
        self.q_table = jnp.zeros((self.state_space_size, self.action_space_size))
        
        if self.config['optimistic_initialization']:
            self.q_table = jnp.ones((self.state_space_size, self.action_space_size)) * self.config['optimistic_initial_value']
    
    def _setup_continuous_state_space(self) -> None:
        """
        Configura SARSA para un espacio de estados continuo con discretización.
        """
        self.state_dim = self.env.observation_space.shape[0]
        self.bins_per_dim = self.config['bins']
        
        # Configurar límites de estado y bins
        self._setup_state_bounds()
        self._create_discretization_bins()
        
        # Crear y configurar tabla Q
        q_shape = tuple([self.bins_per_dim] * self.state_dim + [self.action_space_size])
        self.q_table = jnp.zeros(q_shape)
        
        if self.config['optimistic_initialization']:
            self.q_table = jnp.ones(q_shape) * self.config['optimistic_initial_value']
    
    def _setup_state_bounds(self) -> None:
        """
        Determina los límites para cada dimensión del espacio de estados.
        """
        if self.config['state_bounds'] is None:
            self.state_bounds = self._get_default_state_bounds()
        else:
            self.state_bounds = self.config['state_bounds']
    
    def _get_default_state_bounds(self) -> List[Tuple[float, float]]:
        """
        Calcula límites predeterminados para cada dimensión del estado.
        
        Retorna:
        --------
        List[Tuple[float, float]]
            Lista de tuplas (min, max) para cada dimensión
        """
        bounds = []
        for i in range(self.state_dim):
            low = self.env.observation_space.low[i]
            high = self.env.observation_space.high[i]
            bounds.append((low, high))
        return bounds
    
    def _create_discretization_bins(self) -> None:
        """
        Crea bins para discretización del espacio de estados continuo.
        """
        self.discrete_states = []
        for low, high in self.state_bounds:
            bins = np.linspace(low, high, self.bins_per_dim + 1)[1:-1]
            self.discrete_states.append(bins)
    
    def discretize_state(self, state: np.ndarray) -> Union[Tuple, int]:
        """
        Discretiza un estado continuo.
        
        Parámetros:
        -----------
        state : np.ndarray
            Estado continuo del entorno
            
        Retorna:
        --------
        Union[Tuple, int]
            Índice discretizado como entero (para espacios discretos) o
            tupla con índices discretizados (para espacios continuos)
        """
        if self.discrete_state_space:
            return int(state)
        
        discrete_state = []
        for i, val in enumerate(state):
            bin_idx = np.digitize(val, self.discrete_states[i])
            discrete_state.append(bin_idx)
        
        return tuple(discrete_state)
    
    def _get_action_fn(
        self, 
        q_table: jnp.ndarray, 
        discrete_state: Union[Tuple, int], 
        rng_key: jnp.ndarray, 
        explore: bool
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """
        Función pura para seleccionar acción con política epsilon-greedy.
        
        Parámetros:
        -----------
        q_table : jnp.ndarray
            Tabla Q actual
        discrete_state : Union[Tuple, int]
            Estado discretizado
        rng_key : jnp.ndarray
            Llave para generación de números aleatorios
        explore : bool
            Si se debe explorar o ser greedy
            
        Retorna:
        --------
        Tuple[jnp.ndarray, jnp.ndarray]
            Acción seleccionada (como array) y nueva llave aleatoria
        """
        q_values = q_table[discrete_state]
        best_action = jnp.argmax(q_values)
        
        if not explore:
            return best_action, rng_key  # Devolver como array, no como int
        
        rng_key, explore_key, action_key = jax.random.split(rng_key, 3)
        should_explore = jax.random.uniform(explore_key) < self.epsilon
        random_action = jax.random.randint(action_key, shape=(), minval=0, maxval=self.action_space_size)
        
        action = jnp.where(should_explore, random_action, best_action)
        return action, rng_key  # Devolver como array, no como int
    
    def get_action(self, state: np.ndarray, explore: bool = True) -> int:
        """
        Selecciona una acción usando política epsilon-greedy.
        
        Parámetros:
        -----------
        state : np.ndarray
            Estado actual
        explore : bool, opcional
            Si debe explorar (True) o ser greedy (False) (default: True)
            
        Retorna:
        --------
        int
            Acción seleccionada
        """
        discrete_state = self.discretize_state(state)
        action_array, new_rng_key = self._get_action_fn(
            self.state.q_table, discrete_state, self.state.rng_key, explore
        )
        
        # Actualizar estado de JAX (inmutable)
        self.state = self.state._replace(rng_key=new_rng_key)
        
        # Convertir a int de Python fuera de la función JIT-compilada
        return int(action_array)
    
    def update_q_value(
        self, 
        state: np.ndarray, 
        action: int, 
        reward: float, 
        next_state: np.ndarray, 
        next_action: int
    ) -> None:
        """
        Actualiza un valor Q usando la regla de actualización SARSA.
        
        Parámetros:
        -----------
        state : np.ndarray
            Estado actual
        action : int
            Acción tomada
        reward : float
            Recompensa recibida
        next_state : np.ndarray
            Siguiente estado
        next_action : int
            Siguiente acción (según política actual)
        """
        discrete_state = self.discretize_state(state)
        discrete_next_state = self.discretize_state(next_state)
        
        # Actualizar tabla Q con función pura
        new_q_table = self._update_q_value_fn(
            self.state.q_table,
            discrete_state,
            action,
            reward,
            discrete_next_state,
            next_action
        )
        
        # Actualizar estado de JAX (inmutable)
        self.state = self.state._replace(q_table=new_q_table)
    
    def _update_q_value_fn(
        self, 
        q_table: jnp.ndarray,
        discrete_state: Tuple,
        action: int,
        reward: float,
        discrete_next_state: Tuple,
        next_action: int
    ) -> jnp.ndarray:
        """
        Función pura para actualizar un valor Q usando la regla SARSA.
        
        Parámetros:
        -----------
        q_table : jnp.ndarray
            Tabla Q actual
        discrete_state : Tuple
            Estado discretizado actual
        action : int
            Acción tomada
        reward : float
            Recompensa recibida
        discrete_next_state : Tuple
            Siguiente estado discretizado
        next_action : int
            Siguiente acción (según política actual)
            
        Retorna:
        --------
        jnp.ndarray
            Tabla Q actualizada
        """
        # Valor Q actual
        current_q = q_table[discrete_state][action]
        
        # Valor Q del siguiente estado-acción
        next_q = q_table[discrete_next_state][next_action]
        
        # Calcular nuevo valor Q usando la regla SARSA
        # Q(s,a) = Q(s,a) + α[r + γQ(s',a') - Q(s,a)]
        new_q = current_q + self.alpha * (reward + self.gamma * next_q - current_q)
        
        # Actualizar tabla Q
        return q_table.at[discrete_state].set(q_table[discrete_state].at[action].set(new_q))
    
    def _map_observation_to_state(self, cgm_obs: np.ndarray, other_obs: np.ndarray) -> int:
        """
        Mapea las observaciones a un estado discreto para la tabla Q.
        
        Parámetros:
        -----------
        cgm_obs : np.ndarray
            Datos CGM de una muestra
        other_obs : np.ndarray
            Otras características de una muestra
            
        Retorna:
        --------
        int
            Estado discreto
        """
        # Extraer características relevantes de CGM
        cgm_flat = cgm_obs.flatten()
        
        cgm_mean = np.mean(cgm_flat) if cgm_flat.size > 0 else 0
        cgm_last = cgm_flat[-1] if cgm_flat.size > 0 else 0
        
        # Calcular pendiente si hay suficientes puntos
        if cgm_flat.size >= 5:
            cgm_slope = (cgm_flat[-1] - cgm_flat[-5]) / 5
        else:
            cgm_slope = 0.0
        
        cgm_std = np.std(cgm_flat) if cgm_flat.size > 0 else 0
        
        # Discretizar características CGM
        cgm_bins = 10  # Número de bins por dimensión
        cgm_mean_bin = min(int(cgm_mean / 300 * cgm_bins), cgm_bins - 1)
        cgm_last_bin = min(int(cgm_last / 300 * cgm_bins), cgm_bins - 1)
        cgm_slope_bin = min(int((cgm_slope + 100) / 200 * cgm_bins), cgm_bins - 1)
        cgm_std_bin = min(int(cgm_std / 50 * cgm_bins), cgm_bins - 1)
        
        # Extraer características relevantes de otras variables
        other_bins = 5  # Menos bins para otras características
        
        # Por defecto, usar los primeros 3 valores o ceros si no hay suficientes
        carb_bin = min(int(other_obs[0] / 100 * other_bins), other_bins - 1) if other_obs.size > 0 else 0
        bg_bin = min(int(other_obs[1] / 300 * other_bins), other_bins - 1) if other_obs.size > 1 else 0
        iob_bin = min(int(other_obs[2] / 10 * other_bins), other_bins - 1) if other_obs.size > 2 else 0
        
        # Combinar bins en un único índice de estado
        state_index = cgm_mean_bin
        state_index = state_index * cgm_bins + cgm_last_bin
        state_index = state_index * cgm_bins + cgm_slope_bin
        state_index = state_index * cgm_bins + cgm_std_bin
        state_index = state_index * other_bins + carb_bin
        state_index = state_index * other_bins + bg_bin
        state_index = state_index * other_bins + iob_bin
        
        return state_index
    
    def _convert_action_to_dose(self, action: int) -> float:
        """
        Convierte una acción discreta a una dosis continua.
        
        Parámetros:
        -----------
        action : int
            Índice de acción discreto
            
        Retorna:
        --------
        float
            Dosis de insulina (valor continuo)
        """
        # Mapear de acción discreta [0, n_actions-1] a dosis [0, 15]
        max_dose = 15.0  # Dosis máxima (unidades)
        return (action / (self.action_space_size - 1)) * max_dose
    
    def train_batch(self, agent_state: Any, batch_data: Tuple, rng_key: jax.random.PRNGKey) -> Tuple[Any, Dict[str, float]]:
        """
        Entrena el agente con un lote de datos.
        
        Parámetros:
        -----------
        agent_state : Any
            Estado actual del agente (self)
        batch_data : Tuple
            Datos del lote ((cgm_data, other_data), targets)
        rng_key : jax.random.PRNGKey
            Clave para generación aleatoria
            
        Retorna:
        --------
        Tuple[Any, Dict[str, float]]
            (Nuevo estado del agente, métricas)
        """
        # Extraer datos del lote
        (cgm_data, other_data), targets = batch_data
        batch_size = cgm_data.shape[0]
        
        # Actualizar clave RNG
        self.state = self.state._replace(rng_key=rng_key)
        
        # Métricas a reportar
        total_loss = 0.0
        total_reward = 0.0
        
        # Procesar cada muestra en el lote
        for i in range(batch_size):
            # Mapear observación a estado
            state = self._map_observation_to_state(
                np.array(cgm_data[i]),
                np.array(other_data[i])
            )
            
            # Obtener acción actual según la política
            action = self.get_action(state, explore=True)
            dose = self._convert_action_to_dose(action)
            
            # Calcular recompensa como negativo del error absoluto
            target_dose = float(targets[i])
            reward = -abs(dose - target_dose)
            total_reward += reward
            
            # Simular siguiente estado (en este caso, podría ser el mismo estado)
            next_state = state
            
            # Seleccionar siguiente acción
            next_action = self.get_action(next_state, explore=True)
            
            # Actualizar Q-values
            self.update_q_value(state, action, reward, next_state, next_action)
            
            # Calcular error para métricas
            predicted_dose = self._convert_action_to_dose(action)
            error = (predicted_dose - target_dose) ** 2
            total_loss += error
        
        # Calcular métricas promedio
        avg_loss = total_loss / batch_size if batch_size > 0 else 0.0
        avg_reward = total_reward / batch_size if batch_size > 0 else 0.0
        
        # Actualizar epsilon según esquema de decaimiento
        self._decay_epsilon(0)  # Solo usamos decaimiento exponencial
        
        # Actualizar historial
        self.history['loss'].append(float(avg_loss))
        self.history['avg_reward'].append(float(avg_reward))
        self.history['epsilon'].append(float(self.state.epsilon))
        
        # Preparar métricas
        metrics = {
            'loss': float(avg_loss),
            'reward': float(avg_reward),
            'epsilon': float(self.state.epsilon)
        }
        
        return self, metrics
    
    def _decay_epsilon(self, episode: Optional[int] = None) -> None:
        """
        Decae epsilon según el esquema configurado.
        
        Parámetros:
        -----------
        episode : Optional[int], opcional
            Episodio actual (no usado en decaimiento exponencial)
        """
        # Aplicar decaimiento exponencial
        new_epsilon = max(self.epsilon_min, self.state.epsilon * self.epsilon_decay)
        self.state = self.state._replace(epsilon=new_epsilon)
    
    def predict_batch(self, agent_state: Any, observations: Tuple[jnp.ndarray, jnp.ndarray], rng_key: jax.random.PRNGKey) -> jnp.ndarray:
        """
        Predice acciones para un lote de observaciones.
        
        Parámetros:
        -----------
        agent_state : Any
            Estado del agente (self)
        observations : Tuple[jnp.ndarray, jnp.ndarray]
            (cgm_data, other_data)
        rng_key : jax.random.PRNGKey
            Clave para generación aleatoria
            
        Retorna:
        --------
        jnp.ndarray
            Predicciones de dosis para el lote
        """
        cgm_data, other_data = observations
        batch_size = cgm_data.shape[0]
        
        # Actualizar clave RNG
        self.state = self.state._replace(rng_key=rng_key)
        
        # Predecir para cada muestra
        predictions = np.zeros((batch_size, 1), dtype=np.float32)
        
        for i in range(batch_size):
            # Mapear observación a estado
            state = self._map_observation_to_state(
                np.array(cgm_data[i]),
                np.array(other_data[i])
            )
            
            # Obtener mejor acción (sin exploración)
            action = self.get_action(state, explore=False)
            
            # Convertir a dosis
            dose = self._convert_action_to_dose(action)
            predictions[i, 0] = dose
        
        return jnp.array(predictions)
    
    def evaluate(self, agent_state: Any, batch_data: Tuple, rng_key: jax.random.PRNGKey) -> Dict[str, float]:
        """
        Evalúa el rendimiento en un conjunto de datos.
        
        Parámetros:
        -----------
        agent_state : Any
            Estado del agente (self)
        batch_data : Tuple
            ((cgm_data, other_data), targets)
        rng_key : jax.random.PRNGKey
            Clave para generación aleatoria
            
        Retorna:
        --------
        Dict[str, float]
            Métricas de evaluación
        """
        # Extraer datos
        (cgm_data, other_data), targets = batch_data
        
        # Realizar predicciones
        predictions = self.predict_batch(agent_state, (cgm_data, other_data), rng_key)
        
        # Convertir a arrays numpy
        pred_np = np.array(predictions).squeeze()
        targets_np = np.array(targets).squeeze()
        
        # Calcular métricas
        mse = np.mean((pred_np - targets_np) ** 2)
        mae = np.mean(np.abs(pred_np - targets_np))
        
        return {
            'loss': float(mse),
            'mae': float(mae)
        }


def create_sarsa_agent(cgm_shape: Tuple[int, ...], other_features_shape: Tuple[int, ...], **kwargs) -> SARSA:
    """
    Crea un agente SARSA para el problema de dosificación de insulina.
    
    Parámetros:
    -----------
    cgm_shape : Tuple[int, ...]
        Forma de los datos CGM
    other_features_shape : Tuple[int, ...]
        Forma de otras características
    **kwargs
        Argumentos adicionales para configurar el agente
        
    Retorna:
    --------
    SARSA
        Agente SARSA inicializado
    """
    # Crear entorno ficticio para inicialización
    class TempEnv:
        def __init__(self):
            # Configurar espacios de acción y observación
            # Espacio de acción: 20 niveles discretos (0-15 unidades)
            self.action_space = SimpleNamespace(n=20)
            
            # Espacio de observación: basado en la discretización de características
            cgm_bins = 10
            other_bins = 5
            n_states = cgm_bins**4 * other_bins**3  # 4 características CGM, 3 otras
            self.observation_space = SimpleNamespace(n=n_states)
    
    # Crear entorno temporal
    temp_env = TempEnv()
    
    # Personalizar configuración SARSA
    sarsa_config = SARSA_CONFIG.copy()
    sarsa_config.update({
        'learning_rate': kwargs.get('learning_rate', 0.1),
        'epsilon_start': kwargs.get('epsilon_start', 0.3),
        'epsilon_end': kwargs.get('epsilon_end', 0.01),
        'epsilon_decay': kwargs.get('epsilon_decay', 0.99),
        'gamma': kwargs.get('gamma', 0.95),
        'episodes': kwargs.get('episodes', 1000),
        'max_steps': kwargs.get('max_steps', 100),
        'log_interval': kwargs.get('log_interval', 100)
    })
    
    # Crear agente SARSA
    sarsa_agent = SARSA(
        env=temp_env, 
        config=sarsa_config, 
        seed=kwargs.get('seed', 42),
        cgm_shape=cgm_shape,
        other_features_shape=other_features_shape
    )
    
    return sarsa_agent


def create_sarsa_model(cgm_shape: Tuple[int, ...], other_features_shape: Tuple[int, ...], **kwargs) -> RLModelWrapperJAX:
    """
    Crea un modelo SARSA envuelto en RLModelWrapperJAX.
    
    Parámetros:
    -----------
    cgm_shape : Tuple[int, ...]
        Forma de los datos CGM
    other_features_shape : Tuple[int, ...]
        Forma de otras características
    **kwargs
        Argumentos adicionales para el agente
        
    Retorna:
    --------
    RLModelWrapperJAX
        Agente SARSA envuelto en RLModelWrapperJAX
    """
    # Devolver el wrapper con el agente SARSA
    return RLModelWrapperJAX(
        agent_creator=create_sarsa_agent,
        cgm_shape=cgm_shape,
        other_features_shape=other_features_shape,
        **kwargs
    )


def model_creator() -> Callable[[Tuple[int, ...], Tuple[int, ...]], RLModelWrapperJAX]:
    """
    Devuelve una función para crear un modelo SARSA compatible con la API del sistema.
    
    Retorna:
    --------
    Callable[[Tuple[int, ...], Tuple[int, ...]], RLModelWrapperJAX]
        Función para crear el modelo con las formas de entrada especificadas
    """
    return create_sarsa_model


# Se mantiene la clase SARSAWrapper original para compatibilidad con código existente,
# pero se recomienda usar el enfoque RLModelWrapperJAX para nueva funcionalidad
class SARSAWrapper:
    """
    Wrapper para hacer que SARSA sea compatible con la interfaz de modelos de aprendizaje profundo.
    NOTA: Para nueva funcionalidad, use RLModelWrapperJAX en su lugar.
    """
    
    def __init__(
        self, 
        sarsa_agent: SARSA, 
        cgm_shape: Tuple[int, ...], 
        other_features_shape: Tuple[int, ...],
    ) -> None:
        """
        Inicializa el wrapper para SARSA.
        
        Parámetros:
        -----------
        sarsa_agent : SARSA
            Agente SARSA inicializado
        cgm_shape : Tuple[int, ...]
            Forma de los datos CGM
        other_features_shape : Tuple[int, ...]
            Forma de otras características
        """
        self.sarsa_agent = sarsa_agent
        self.cgm_shape = cgm_shape
        self.other_features_shape = other_features_shape
        
        # Determinar dimensionalidad para discretización
        self.cgm_bins = 10  # Bins por dimensión CGM
        self.other_bins = 5  # Bins por dimensión de otras características
        
        # Para compatibilidad con la interfaz de entrenamiento
        self.rl_wrapper = RLModelWrapperJAX(
            agent_creator=lambda cgm_shape, other_features_shape, **kwargs: sarsa_agent,
            cgm_shape=cgm_shape,
            other_features_shape=other_features_shape
        )
    
    def start(self, x_cgm: np.ndarray, x_other: np.ndarray, y: np.ndarray, rng_key: Any = None) -> Any:
        """
        Inicializa el agente con los datos de entrada.
        
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
            Estado del modelo inicializado
        """
        return self.rl_wrapper.start(x_cgm, x_other, y, rng_key)
    
    def __call__(self, inputs: List[np.ndarray]) -> np.ndarray:
        """
        Realiza una predicción con el modelo.
        
        Parámetros:
        -----------
        inputs : List[np.ndarray]
            Lista de entradas [cgm_data, other_data]
            
        Retorna:
        --------
        np.ndarray
            Predicciones del modelo
        """
        return self.predict(inputs)
    
    def predict(self, inputs: List[np.ndarray]) -> np.ndarray:
        """
        Realiza predicciones con el modelo.
        
        Parámetros:
        -----------
        inputs : List[np.ndarray]
            Lista de entradas [cgm_data, other_data]
            
        Retorna:
        --------
        np.ndarray
            Predicciones del modelo
        """
        if isinstance(inputs, list) and len(inputs) == 2:
            x_cgm, x_other = inputs
        else:
            raise ValueError("Las entradas deben ser una lista [cgm_data, other_data]")
        
        return self.rl_wrapper.predict(x_cgm, x_other)
    
    def _discretize_state(self, cgm_data: np.ndarray, other_features: np.ndarray) -> int:
        """
        Discretiza el estado para la tabla Q.
        
        Parámetros:
        -----------
        cgm_data : np.ndarray
            Datos CGM
        other_features : np.ndarray
            Otras características
            
        Retorna:
        --------
        int
            Estado discretizado
        """
        return self.sarsa_agent._map_observation_to_state(cgm_data, other_features)
    
    def _convert_action_to_dose(self, action: int) -> float:
        """
        Convierte una acción discreta a una dosis.
        
        Parámetros:
        -----------
        action : int
            Acción discreta
            
        Retorna:
        --------
        float
            Dosis correspondiente
        """
        return self.sarsa_agent._convert_action_to_dose(action)
    
    def fit(
        self, 
        x: List[np.ndarray], 
        y: np.ndarray, 
        validation_data: Optional[Tuple] = None, 
        epochs: int = 1,
        batch_size: int = 32,
        callbacks: List = None,
        verbose: int = 0
    ) -> Dict:
        """
        Entrena el modelo con los datos proporcionados.
        
        Parámetros:
        -----------
        x : List[np.ndarray]
            Lista de entradas [cgm_data, other_data]
        y : np.ndarray
            Valores objetivo
        validation_data : Optional[Tuple], opcional
            Datos de validación como ([x_cgm_val, x_other_val], y_val) (default: None)
        epochs : int, opcional
            Número de épocas (default: 1)
        batch_size : int, opcional
            Tamaño del lote (default: 32)
        callbacks : List, opcional
            Callbacks para el entrenamiento (default: None)
        verbose : int, opcional
            Nivel de verbosidad (0 = silencioso, 1 = progreso) (default: 0)
            
        Retorna:
        --------
        Dict
            Historial de entrenamiento
        """
        if isinstance(x, list) and len(x) == 2:
            x_cgm, x_other = x
        else:
            raise ValueError("x debe ser una lista [cgm_data, other_data]")
        
        val_data = None
        if validation_data is not None:
            val_data = (validation_data[0], validation_data[1])
        
        # Delegar al wrapper RLModelWrapperJAX
        return self.rl_wrapper.train(
            x_cgm=x_cgm,
            x_other=x_other,
            y=y,
            validation_data=val_data,
            epochs=epochs,
            batch_size=batch_size
        )
    
    def save(self, filepath: str) -> None:
        """
        Guarda el modelo en un archivo.
        
        Parámetros:
        -----------
        filepath : str
            Ruta donde guardar el modelo
        """
        self.rl_wrapper.save(filepath)
    
    def load(self, filepath: str) -> None:
        """
        Carga el modelo desde un archivo.
        
        Parámetros:
        -----------
        filepath : str
            Ruta desde donde cargar el modelo
        """
        self.rl_wrapper.load(filepath)
    
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
            'other_features_shape': self.other_features_shape
        }