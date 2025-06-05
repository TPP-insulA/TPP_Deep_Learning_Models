import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import time
from typing import Dict, List, Tuple, Optional, Union, Any, Callable, NamedTuple
import jax
import jax.numpy as jnp
from jax import jit, random
from functools import partial

from custom.printer import print_debug

PROJECT_ROOT = os.path.abspath(os.getcwd())
sys.path.append(PROJECT_ROOT) 

from config.models_config_old import EARLY_STOPPING_POLICY, QLEARNING_CONFIG
from constants.constants import CONST_DEFAULT_SEED
from custom.ReinforcementLearning.rl_jax import RLModelWrapperJAX

# Constantes para rutas y mensajes
FIGURES_DIR = os.path.join(PROJECT_ROOT, "figures", "jax", "q_learning")
CONST_EPISODE = "Episodio"
CONST_REWARD = "Recompensa"
CONST_STEPS = "Pasos"
CONST_AVG_REWARD = "Recompensa Promedio"
CONST_EPSILON = "Epsilon"
CONST_ELAPSED_TIME = "Tiempo Transcurrido"

# Crear directorio para figuras si no existe
os.makedirs(FIGURES_DIR, exist_ok=True)

class QTableState(NamedTuple):
    """Estructura para almacenar el estado del agente Q-Learning"""
    q_table: jnp.ndarray
    rng_key: jnp.ndarray
    epsilon: float
    total_steps: int


class QLearning:
    """
    Implementación de Q-Learning tabular con JAX para espacios de estados y acciones discretos.
    
    Este algoritmo aprende una función de valor-acción (Q) a través de experiencias
    recolectadas mediante interacción con el entorno. Utiliza una tabla para almacenar
    los valores Q para cada par estado-acción.
    """
    def __init__(
        self,
        n_states: int,
        n_actions: int,
        learning_rate: float = QLEARNING_CONFIG['learning_rate'],
        gamma: float = QLEARNING_CONFIG['gamma'],
        epsilon_start: float = QLEARNING_CONFIG['epsilon_start'],
        epsilon_end: float = QLEARNING_CONFIG['epsilon_end'],
        epsilon_decay: float = QLEARNING_CONFIG['epsilon_decay'],
        use_decay_schedule: str = QLEARNING_CONFIG['use_decay_schedule'],
        decay_steps: int = QLEARNING_CONFIG['decay_steps'],
        seed: int = CONST_DEFAULT_SEED,
        cgm_shape: Optional[Tuple[int, ...]] = None,
        other_features_shape: Optional[Tuple[int, ...]] = None
    ) -> None:
        """
        Inicializa el agente Q-Learning.
        
        Parámetros:
        -----------
        n_states : int
            Número de estados en el espacio de estados discreto
        n_actions : int
            Número de acciones en el espacio de acciones discreto
        learning_rate : float, opcional
            Tasa de aprendizaje (alpha) (default: QLEARNING_CONFIG['learning_rate'])
        gamma : float, opcional
            Factor de descuento (default: QLEARNING_CONFIG['gamma'])
        epsilon_start : float, opcional
            Valor inicial de epsilon para política epsilon-greedy (default: QLEARNING_CONFIG['epsilon_start'])
        epsilon_end : float, opcional
            Valor final de epsilon para política epsilon-greedy (default: QLEARNING_CONFIG['epsilon_end'])
        epsilon_decay : float, opcional
            Factor de decaimiento para epsilon (default: QLEARNING_CONFIG['epsilon_decay'])
        use_decay_schedule : str, opcional
            Tipo de decaimiento ('linear', 'exponential', o None) (default: QLEARNING_CONFIG['use_decay_schedule'])
        decay_steps : int, opcional
            Número de pasos para decaer epsilon (si se usa decay schedule) (default: QLEARNING_CONFIG['decay_steps'])
        seed : int, opcional
            Semilla para reproducibilidad (default: 42)
        cgm_shape : Optional[Tuple[int, ...]], opcional
            Forma de los datos CGM (necesario para compatibilidad con RLModelWrapperJAX) (default: None)
        other_features_shape : Optional[Tuple[int, ...]], opcional
            Forma de otras características (necesario para compatibilidad con RLModelWrapperJAX) (default: None)
        """
        self.n_states = n_states
        self.n_actions = n_actions
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.use_decay_schedule = use_decay_schedule
        self.decay_steps = decay_steps
        self.cgm_shape = cgm_shape
        self.other_features_shape = other_features_shape
        
        # Crear llave PRNG inicial para JAX
        self.rng_key = random.PRNGKey(seed)
        
        # Inicializar la tabla Q con valores optimistas o ceros
        if QLEARNING_CONFIG['optimistic_init']:
            self.q_table = jnp.ones((n_states, n_actions)) * QLEARNING_CONFIG['optimistic_value']
        else:
            self.q_table = jnp.zeros((n_states, n_actions))
        
        # Estado mutable para JAX (que es funcionalmente pura)
        self.state = QTableState(
            q_table=self.q_table,
            rng_key=self.rng_key,
            epsilon=epsilon_start,
            total_steps=0
        )
        
        # Métricas
        self.rewards_history = []
        self.history = {'loss': [], 'avg_reward': [], 'epsilon': []}
        
        # Compilar funciones puras
        self._update_q_value = jit(self._update_q_value)
    
    def setup(self, rng_key: jax.random.PRNGKey) -> Any:
        """
        Inicializa el agente para interactuar con el entorno.
        
        Parámetros:
        -----------
        rng_key : jax.random.PRNGKey
            Clave para generación aleatoria
            
        Retorna:
        --------
        Any
            El estado del agente inicializado
        """
        # Inicializar la tabla Q con valores optimistas o ceros
        if QLEARNING_CONFIG['optimistic_init']:
            q_table = jnp.ones((self.n_states, self.n_actions)) * QLEARNING_CONFIG['optimistic_value']
        else:
            q_table = jnp.zeros((self.n_states, self.n_actions))
        
        # Crear estado inicial del agente
        agent_state = QTableState(
            q_table=q_table,
            rng_key=rng_key,
            epsilon=self.epsilon_start,
            total_steps=0
        )
        
        return agent_state
    
    def _get_action(self, state_idx: int, rng_key: jnp.ndarray, q_table: jnp.ndarray, 
                   epsilon: float) -> Tuple[int, jnp.ndarray]:
        """
        Selecciona una acción usando política epsilon-greedy.
        
        Parámetros:
        -----------
        state_idx : int
            Índice del estado actual
        rng_key : jnp.ndarray
            Clave para generación aleatoria
        q_table : jnp.ndarray
            Tabla Q actual
        epsilon : float
            Valor actual de epsilon
            
        Retorna:
        --------
        Tuple[int, jnp.ndarray]
            Tupla con (acción seleccionada, nueva clave PRNG)
        """
        # Dividir la clave para tomar dos decisiones aleatorias
        key_decision, key_action, next_key = random.split(rng_key, 3)
        
        # Decidir si explorar o explotar
        explore = random.uniform(key_decision) < epsilon
        
        # Si explorar, seleccionar acción aleatoria
        random_action = random.randint(key_action, shape=(), minval=0, maxval=self.n_actions)
        
        # Si explotar, seleccionar mejor acción según la tabla Q
        best_action = jnp.argmax(q_table[state_idx])
        
        # Seleccionar entre exploración y explotación
        action = jnp.where(explore, random_action, best_action)
        
        return int(action), next_key
    
    def get_action(self, state: int) -> int:
        """
        Obtiene una acción para un estado dado usando la política actual.
        
        Parámetros:
        -----------
        state : int
            Índice del estado actual
            
        Retorna:
        --------
        int
            Acción seleccionada
        """
        # Usar política epsilon-greedy
        action, next_key = self._get_action(
            state,
            self.state.rng_key,
            self.state.q_table,
            self.state.epsilon
        )
        
        # Actualizar la llave PRNG
        self.state = self.state._replace(rng_key=next_key)
        
        return action
    
    def _update_q_value(self, q_table: jnp.ndarray, state: int, action: int, 
                       reward: float, next_state: int, done: bool) -> Tuple[jnp.ndarray, float]:
        """
        Actualiza el valor Q para un par estado-acción.
        
        Parámetros:
        -----------
        q_table : jnp.ndarray
            Tabla Q actual
        state : int
            Estado actual
        action : int
            Acción tomada
        reward : float
            Recompensa recibida
        next_state : int
            Estado siguiente
        done : bool
            Indica si el episodio ha terminado
            
        Retorna:
        --------
        Tuple[jnp.ndarray, float]
            Tupla con (tabla Q actualizada, error TD)
        """
        # Obtener valor Q actual
        current_q = q_table[state, action]
        
        # Calcular valor objetivo (target) usando jnp.where en lugar de if/else
        # para compatibilidad con JIT
        max_next_q = jnp.max(q_table[next_state])
        target_q = jnp.where(done, reward, reward + self.gamma * max_next_q)
        
        # Calcular TD error
        td_error = target_q - current_q
        
        # Actualizar valor Q
        new_q_value = current_q + self.learning_rate * td_error
        
        # Crear tabla Q actualizada (JAX arrays son inmutables)
        new_q_table = q_table.at[state, action].set(new_q_value)
        
        return new_q_table, td_error
    
    def update(self, state: int, action: int, reward: float, next_state: int, done: bool) -> float:
        """
        Actualiza la tabla Q con una nueva experiencia.
        
        Parámetros:
        -----------
        state : int
            Estado actual
        action : int
            Acción tomada
        reward : float
            Recompensa recibida
        next_state : int
            Estado siguiente
        done : bool
            Indica si el episodio ha terminado
            
        Retorna:
        --------
        float
            Error TD de la actualización
        """
        # Actualizar tabla Q utilizando la función compilada con jit
        new_q_table, td_error = self._update_q_value(
            self.state.q_table, 
            state, 
            action, 
            reward, 
            next_state, 
            done
        )
        
        # Actualizar el estado del agente
        self.state = self.state._replace(
            q_table=new_q_table,
            total_steps=self.state.total_steps + 1
        )
        
        # Actualizar epsilon según el esquema de decaimiento
        self.update_epsilon(self.state.total_steps)
        
        return float(td_error)
    
    def _update_epsilon_value(self, epsilon: float, step: Optional[int], use_decay_schedule: str) -> float:
        """
        Calcula el nuevo valor de epsilon según el esquema de decaimiento.
        
        Parámetros:
        -----------
        epsilon : float
            Valor actual de epsilon
        step : Optional[int]
            Paso actual del entrenamiento
        use_decay_schedule : str
            Tipo de esquema de decaimiento ('linear', 'exponential', o None)
            
        Retorna:
        --------
        float
            Nuevo valor de epsilon
        """
        if use_decay_schedule is None or step is None:
            return epsilon
        
        if use_decay_schedule == 'linear':
            # Decaimiento lineal
            fraction = min(1.0, step / self.decay_steps)
            return self.epsilon_end + (self.epsilon_start - self.epsilon_end) * (1 - fraction)
        
        if use_decay_schedule == 'exponential':
            # Decaimiento exponencial
            return self.epsilon_end + (self.epsilon_start - self.epsilon_end) * self.epsilon_decay ** step
        
        # Por defecto, mantener el mismo epsilon
        return epsilon
    
    def update_epsilon(self, step: Optional[int] = None) -> None:
        """
        Actualiza el valor epsilon según el esquema de decaimiento.
        
        Parámetros:
        -----------
        step : Optional[int], opcional
            Paso actual del entrenamiento (default: None)
        """
        # Calcular nuevo epsilon
        new_epsilon = self._update_epsilon_value(
            self.state.epsilon, 
            step, 
            self.use_decay_schedule
        )
        
        # Actualizar el estado con el nuevo epsilon
        self.state = self.state._replace(epsilon=new_epsilon)
    
    def train_batch(self, agent_state: QTableState, batch_data: Tuple, rng_key: jax.random.PRNGKey) -> Tuple[QTableState, Dict[str, float]]:
        """
        Entrena el agente con un lote de datos.
        
        Parámetros:
        -----------
        agent_state : QTableState
            Estado actual del agente
        batch_data : Tuple
            Tupla con datos del lote (estructura puede variar)
        rng_key : jax.random.PRNGKey
            Clave para generación aleatoria
            
        Retorna:
        --------
        Tuple[QTableState, Dict[str, float]]
            Tupla con (nuevo estado del agente, métricas)
        """
        (x_cgm, x_other), y = batch_data
        
        # Asegurar que x_cgm y x_other sean arrays
        if not isinstance(x_cgm, (np.ndarray, jnp.ndarray)) or not isinstance(x_other, (np.ndarray, jnp.ndarray)):
            raise ValueError(f"x_cgm y x_other deben ser arrays. Tipos: x_cgm={type(x_cgm)}, x_other={type(x_other)}")
        
        # Obtener batch_size de manera segura
        batch_size = x_cgm.shape[0] if hasattr(x_cgm, 'shape') else len(x_cgm)
        total_loss = 0.0
        rewards = jnp.zeros(batch_size)
        
        # Actualizar la llave PRNG del estado del agente
        updated_state = agent_state._replace(rng_key=rng_key)
        
        # Procesar cada muestra en el lote
        for i in range(batch_size):
            # Mapear observación a estado discreto
            state_idx = self._map_observation_to_state(x_cgm[i], x_other[i])
            
            # Calcular recompensa (error negativo entre predicción y objetivo)
            pred = self._convert_action_to_dose(jnp.argmax(updated_state.q_table[state_idx]))
            reward = -((pred - y[i]) ** 2)  # Recompensa negativa del error cuadrático
            rewards = rewards.at[i].set(reward)
            
            # Obtener acción usando política epsilon-greedy
            action, next_key = self._get_action(
                state_idx, 
                updated_state.rng_key, 
                updated_state.q_table, 
                updated_state.epsilon
            )
            
            # Estado siguiente (mismo estado para entrenamiento por lotes)
            next_state_idx = state_idx
            done = True  # Terminamos después de cada muestra
            
            # Actualizar Q-valor
            new_q_table, td_error = self._update_q_value(
                updated_state.q_table, 
                state_idx, 
                action, 
                reward, 
                next_state_idx, 
                done
            )
            
            # Actualizar estado del agente
            updated_state = updated_state._replace(
                q_table=new_q_table,
                rng_key=next_key,
                total_steps=updated_state.total_steps + 1
            )
            
            # Acumular pérdida
            total_loss += jnp.abs(td_error)
        
        # Actualizar epsilon según esquema de decaimiento
        new_epsilon = self._update_epsilon_value(
            updated_state.epsilon, 
            updated_state.total_steps, 
            self.use_decay_schedule
        )
        updated_state = updated_state._replace(epsilon=new_epsilon)
        
        # Calcular métricas promedio
        avg_loss = total_loss / batch_size if batch_size > 0 else 0.0
        avg_reward = jnp.mean(rewards)
        
        # Retornar estado actualizado y métricas
        metrics = {
            'loss': float(avg_loss),
            'avg_reward': float(avg_reward),
            'epsilon': float(updated_state.epsilon)
        }
        
        return updated_state, metrics
    
    def _map_observation_to_state(self, cgm_obs: np.ndarray, other_obs: np.ndarray) -> int:
        """
        Mapea una observación continua a un índice de estado discreto.
        
        Parámetros:
        -----------
        cgm_obs : np.ndarray
            Observación CGM
        other_obs : np.ndarray
            Otras características
            
        Retorna:
        --------
        int
            Índice de estado discreto
        """
        # Aplanar y concatenar observaciones
        flattened_cgm = cgm_obs.flatten()
        flattened_other = other_obs.flatten()
        combined = np.concatenate([flattened_cgm, flattened_other])
        
        # Aplicar hash para obtener un valor discreto
        # Usar solo los primeros elementos si hay muchos
        num_features = min(len(combined), 5)
        selected_features = combined[:num_features]
        
        # Hacer hash de las características seleccionadas
        features_sum = np.sum(selected_features)
        features_mean = np.mean(selected_features)
        features_std = np.std(selected_features) if len(selected_features) > 1 else 0
        
        # Calcular índice de estado simple basado en características agregadas
        state_idx = int((features_sum + features_mean + features_std) * 100) % self.n_states
        
        return state_idx
    
    def _convert_action_to_dose(self, action: int) -> float:
        """
        Convierte un índice de acción a una dosis.
        
        Parámetros:
        -----------
        action : int
            Índice de acción discreta
            
        Retorna:
        --------
        float
            Valor de dosis correspondiente
        """
        # Escalar el índice de acción al rango de dosis
        # Rango de dosis asumido: 0 a 20 unidades
        max_dose = 20.0
        return (action / (self.n_actions - 1)) * max_dose
    
    def predict_batch(self, agent_state: QTableState, observations: Tuple[jnp.ndarray, jnp.ndarray], rng_key: jax.random.PRNGKey) -> jnp.ndarray:
        """
        Realiza predicciones para un lote de observaciones.
        
        Parámetros:
        -----------
        agent_state : QTableState
            Estado actual del agente
        observations : Tuple[jnp.ndarray, jnp.ndarray]
            Tupla con (x_cgm, x_other) para predecir
        rng_key : jax.random.PRNGKey
            Clave para generación aleatoria
            
        Retorna:
        --------
        jnp.ndarray
            Predicciones del modelo
        """
        x_cgm, x_other = observations
        batch_size = x_cgm.shape[0]
        predictions = np.zeros(batch_size)
        
        for i in range(batch_size):
            # Mapear observación a estado
            state_idx = self._map_observation_to_state(x_cgm[i], x_other[i])
            
            # Seleccionar mejor acción según tabla Q
            best_action = jnp.argmax(agent_state.q_table[state_idx])
            
            # Convertir acción a dosis
            predictions[i] = self._convert_action_to_dose(int(best_action))
        
        return jnp.array(predictions)
    
    def evaluate(self, agent_state: QTableState, batch_data: Tuple, rng_key: jax.random.PRNGKey) -> Dict[str, float]:
        """
        Evalúa el modelo con datos de prueba.
        
        Parámetros:
        -----------
        agent_state : QTableState
            Estado actual del agente
        batch_data : Tuple
            Tupla con ((x_cgm, x_other), y) para evaluar
        rng_key : jax.random.PRNGKey
            Clave para generación aleatoria
            
        Retorna:
        --------
        Dict[str, float]
            Métricas de evaluación
        """
        # Desempaquetar datos
        (x_cgm, x_other), y = batch_data
        
        # Obtener predicciones
        predictions = self.predict_batch(agent_state, (x_cgm, x_other), rng_key)
        
        # Calcular métricas
        mse = jnp.mean((predictions - y) ** 2)
        mae = jnp.mean(jnp.abs(predictions - y))
        
        # Retornar métricas
        return {
            'loss': float(mse),
            'mae': float(mae)
        }
    
    def get_q_table(self) -> np.ndarray:
        """
        Obtiene la tabla Q actual.
        
        Retorna:
        --------
        np.ndarray
            Tabla Q actual
        """
        return np.array(self.state.q_table)


def create_q_learning_agent(cgm_shape: Tuple[int, ...], other_features_shape: Tuple[int, ...], **kwargs) -> QLearning:
    """
    Crea una instancia del agente QLearning.
    
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
    QLearning
        Instancia del agente Q-Learning
    """
    # Configurar el tamaño del espacio de estados y acciones
    n_states = kwargs.get('n_states', 1000)  # Estado discretizado
    n_actions = kwargs.get('n_actions', 20)   # Por ejemplo: 20 niveles discretos de dosis
    
    # Crear agente Q-Learning
    q_agent = QLearning(
        n_states=n_states,
        n_actions=n_actions,
        learning_rate=kwargs.get('learning_rate', QLEARNING_CONFIG['learning_rate']),
        gamma=kwargs.get('gamma', QLEARNING_CONFIG['gamma']),
        epsilon_start=kwargs.get('epsilon_start', QLEARNING_CONFIG['epsilon_start']),
        epsilon_end=kwargs.get('epsilon_end', QLEARNING_CONFIG['epsilon_end']),
        epsilon_decay=kwargs.get('epsilon_decay', QLEARNING_CONFIG['epsilon_decay']),
        use_decay_schedule=kwargs.get('use_decay_schedule', QLEARNING_CONFIG['use_decay_schedule']),
        decay_steps=kwargs.get('decay_steps', QLEARNING_CONFIG['decay_steps']),
        seed=kwargs.get('seed', CONST_DEFAULT_SEED),
        cgm_shape=cgm_shape,
        other_features_shape=other_features_shape
    )
    
    return q_agent


def create_q_learning_model(cgm_shape: Tuple[int, ...], other_features_shape: Tuple[int, ...], **model_kwargs) -> RLModelWrapperJAX:
    """
    Crea un modelo Q-Learning envuelto en RLModelWrapperJAX.
    
    Parámetros:
    -----------
    cgm_shape : Tuple[int, ...]
        Forma de los datos CGM
    other_features_shape : Tuple[int, ...]
        Forma de otras características
    **model_kwargs
        Argumentos adicionales para el agente Q-Learning
        
    Retorna:
    --------
    RLModelWrapperJAX
        Wrapper RL para el agente Q-Learning
    """
    # Crear el wrapper RLModelWrapperJAX con la función creadora del agente
    model = RLModelWrapperJAX(
        agent_creator=create_q_learning_agent,
        cgm_shape=cgm_shape,
        other_features_shape=other_features_shape,
        **model_kwargs
    )
    
    # Configurar early stopping
    patience = EARLY_STOPPING_POLICY.get('patience', 10)
    min_delta = EARLY_STOPPING_POLICY.get('min_delta', 0.01)
    restore_best_weights = EARLY_STOPPING_POLICY.get('restore_best_weights', True)
    model.add_early_stopping(patience=patience, min_delta=min_delta, restore_best_weights=restore_best_weights)
    
    return model


def model_creator() -> Callable[[Tuple[int, ...], Tuple[int, ...], Dict], RLModelWrapperJAX]:
    """
    Retorna una función para crear un modelo Q-Learning compatible con la API del sistema.
    
    Retorna:
    --------
    Callable[[Tuple[int, ...], Tuple[int, ...], Dict], RLModelWrapperJAX]
        Función para crear un modelo Q-Learning con las formas de entrada especificadas
    """
    def creator_with_kwargs(cgm_shape: Tuple[int, ...], other_features_shape: Tuple[int, ...], **kwargs) -> RLModelWrapperJAX:
        return create_q_learning_model(cgm_shape, other_features_shape, **kwargs)
    
    return creator_with_kwargs