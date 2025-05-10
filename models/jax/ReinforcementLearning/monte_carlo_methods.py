import os
import sys
import numpy as np
import jax
import jax.numpy as jnp
from jax import jit, grad, random, vmap
from typing import Dict, List, Tuple, Optional, Union, Any, Callable
from functools import partial
from tqdm.auto import tqdm

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../.."))
sys.path.append(PROJECT_ROOT)

from constants.constants import CONST_DEFAULT_SEED
from config.models_config import EARLY_STOPPING_POLICY, MONTE_CARLO_CONFIG
from custom.rl_model_wrapper import RLModelWrapperJAX
from custom.printer import print_warning

FIGURES_DIR = os.path.join(PROJECT_ROOT, 'figures', 'jax', 'monte_carlo')

class MonteCarlo:
    """
    Implementación de métodos Monte Carlo para predicción y control en aprendizaje por refuerzo usando JAX.
    
    Esta clase proporciona implementaciones de:
    1. Predicción Monte Carlo (first-visit y every-visit) para evaluar políticas
    2. Control Monte Carlo (on-policy y off-policy) para encontrar políticas óptimas
    
    Se incluyen algoritmos como:
    - First-visit MC prediction
    - Every-visit MC prediction
    - Monte Carlo Exploring Starts (MCES)
    - On-policy MC control con epsilon-greedy
    - Off-policy MC control con importance sampling
    """
    
    def __init__(
        self,
        n_states: int,
        n_actions: int,
        gamma: float = MONTE_CARLO_CONFIG['gamma'],
        epsilon_start: float = MONTE_CARLO_CONFIG['epsilon_start'],
        epsilon_end: float = MONTE_CARLO_CONFIG['epsilon_end'],
        epsilon_decay: float = MONTE_CARLO_CONFIG['epsilon_decay'],
        first_visit: bool = MONTE_CARLO_CONFIG['first_visit'],
        evaluation_mode: bool = False,
        seed: int = CONST_DEFAULT_SEED,
        cgm_shape: Optional[Tuple[int, ...]] = None,
        other_features_shape: Optional[Tuple[int, ...]] = None
    ) -> None:
        """
        Inicializa el agente Monte Carlo.

        Parámetros:
        -----------
        n_states : int
            Número de estados discretos.
        n_actions : int
            Número de acciones discretas.
        gamma : float, opcional
            Factor de descuento (default: MONTE_CARLO_CONFIG['gamma']).
        epsilon_start : float, opcional
            Valor inicial de epsilon para exploración (default: MONTE_CARLO_CONFIG['epsilon_start']).
        epsilon_end : float, opcional
            Valor final de epsilon (default: MONTE_CARLO_CONFIG['epsilon_end']).
        epsilon_decay : float, opcional
            Tasa de decaimiento de epsilon (default: MONTE_CARLO_CONFIG['epsilon_decay']).
        first_visit : bool, opcional
            Si usar first-visit MC (True) o every-visit MC (False) (default: MONTE_CARLO_CONFIG['first_visit']).
        evaluation_mode : bool, opcional
            Si el agente está en modo evaluación (sin exploración) (default: False).
        seed : int, opcional
            Semilla para el generador de números aleatorios (default: 42).
        cgm_shape : Optional[Tuple[int, ...]], opcional
            Forma de los datos CGM (puede ser necesario para mapeo estado-observación) (default: None).
        other_features_shape : Optional[Tuple[int, ...]], opcional
            Forma de otras características (puede ser necesario para mapeo estado-observación) (default: None).
        """
        self.n_states = n_states
        self.n_actions = n_actions
        self.gamma = gamma
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.epsilon = epsilon_start
        self.first_visit = first_visit # Nota: La lógica first-visit es difícil de implementar sin episodios completos.
        self.evaluation_mode = evaluation_mode
        # Usar np.random.Generator
        self.rng_np = np.random.default_rng(seed)
        
        self.q_table = np.zeros((n_states, n_actions))
        self.n_table = np.zeros((n_states, n_actions)) # Contador de visitas (state, action)
        # self.c_table = np.zeros((n_states, n_actions)) # Para Importance Sampling (Off-policy) - No implementado aquí
        
        # Política (inicialmente equiprobable o basada en Q-table si se carga)
        # Se actualiza implícitamente al elegir acciones basadas en Q-table
        
        self.cgm_shape = cgm_shape
        self.other_features_shape = other_features_shape
        
        self.history: Dict[str, List[float]] = {'loss': [], 'avg_reward': [], 'epsilon': []}

    def setup(self, rng_key: jax.random.PRNGKey) -> Any:
        """
        Método de inicialización compatible con RLModelWrapperJAX.
        Devuelve el estado inicial del agente (puede ser self o parámetros específicos).

        Parámetros:
        -----------
        rng_key : jax.random.PRNGKey
            Clave JAX para aleatoriedad.

        Retorna:
        --------
        Any
            El propio agente como estado inicial.
        """
        # La inicialización principal ya se hizo en __init__
        return self

    def train_batch(self, agent_state: Any, batch_data: Tuple, rng_key: jax.random.PRNGKey) -> Tuple[Any, Dict[str, float]]:
        """
        Actualiza la Q-table usando un batch de datos preprocesados.
        Requiere que batch_data contenga información interpretable como (estado, acción, retorno G).

        Parámetros:
        -----------
        agent_state : Any
            Estado actual del agente (la propia instancia 'self').
        batch_data : Tuple
            Tupla conteniendo ((observaciones_cgm, observaciones_other), targets).
            **ASUNCIÓN FUERTE:** Se asume que `targets` (batch_y) representa el retorno Monte Carlo (G)
            calculado previamente para una acción específica tomada en el estado derivado de las observaciones.
            La acción tomada NO está explícitamente en `batch_data` y debe ser inferida o asumida.
        rng_key : jax.random.PRNGKey
            Clave JAX para aleatoriedad (no usada directamente aquí).

        Retorna:
        --------
        Tuple[Any, Dict[str, float]]
            Nuevo estado del agente (self) y métricas de entrenamiento.
        """
        current_agent = agent_state # La instancia MonteCarlo
        (observations_cgm, observations_other), returns_g = batch_data

        batch_size = observations_cgm.shape[0]
        total_update_magnitude = 0.0

        # --- Lógica de Actualización Monte Carlo con Datos de Batch ---
        # Iterar sobre cada muestra en el batch
        for i in range(batch_size):
            # 1. Obtener el estado discreto de la observación
            state = current_agent._map_observation_to_state(
                np.array(observations_cgm[i]), # Convertir a numpy si son JAX arrays
                np.array(observations_other[i])
            )

            # 2. Obtener el retorno G (asumido de batch_y)
            G = float(returns_g[i])

            # 3. Determinar la acción tomada (¡¡PROBLEMA!!)
            #    No tenemos la acción que llevó a este retorno G.
            #    OPCIONES (con fuertes limitaciones):
            #    a) Asumir que la acción fue la greedy actual: action = np.argmax(current_agent.q_table[state])
            #    b) Asumir que batch_y codifica la acción (requiere cambio en preprocesamiento)
            #    c) Asumir una acción aleatoria (incorrecto para MC estándar)
            #    d) Omitir la actualización si la acción no se conoce.

            # Implementando opción (a) con una advertencia:
            action = np.argmax(current_agent.q_table[state])
            # print_warning("Asumiendo acción greedy para actualización MC en train_batch. "
            #               "Esto puede ser incorrecto si los retornos G no corresponden a la acción greedy.")

            # 4. Actualizar Q(s, a) usando la fórmula incremental de MC
            current_agent.n_table[state, action] += 1
            old_q = current_agent.q_table[state, action]
            # Q(s, a) <- Q(s, a) + (1 / N(s, a)) * (G - Q(s, a))
            update_step = (1.0 / current_agent.n_table[state, action]) * (G - old_q)
            current_agent.q_table[state, action] += update_step
            total_update_magnitude += abs(update_step)

            # Nota: La lógica first-visit/every-visit no se puede aplicar correctamente
            #       sin la estructura completa del episodio. Esta implementación
            #       corresponde a una actualización tipo every-visit si el mismo (s, a)
            #       aparece múltiples veces en el batch o en batches sucesivos.

        # Actualizar epsilon para exploración futura (aunque no se use en esta actualización)
        current_agent.epsilon = max(current_agent.epsilon_end, current_agent.epsilon * current_agent.epsilon_decay)

        # Calcular métricas: 'loss' puede representar la magnitud promedio de la actualización
        avg_update = total_update_magnitude / batch_size if batch_size > 0 else 0.0
        metrics = {
            'loss': float(avg_update), # Usar la magnitud del cambio en Q como 'loss'
            'epsilon': float(current_agent.epsilon)
            # 'reward' no es directamente aplicable aquí, G ya es el retorno acumulado
        }

        # Devolver el estado actualizado (la propia instancia) y las métricas
        return current_agent, metrics

    def predict_batch(self, agent_state: Any, observations: Tuple[jnp.ndarray, jnp.ndarray], rng_key: jax.random.PRNGKey) -> jnp.ndarray:
        """
        Predice acciones para un batch de observaciones. Compatible con RLModelWrapperJAX.

        Parámetros:
        -----------
        agent_state : Any
             Estado actual del agente (la instancia 'self').
        observations : Tuple[jnp.ndarray, jnp.ndarray]
            Tupla con (batch_cgm, batch_other). Convertidos a JAX arrays por el wrapper.
        rng_key : jax.random.PRNGKey
            Clave JAX para aleatoriedad (no usada en predicción determinista).

        Retorna:
        --------
        jnp.ndarray
            Array JAX de acciones predichas para el batch.
        """
        current_agent = agent_state # La instancia MonteCarlo
        x_cgm, x_other = observations
        batch_size = x_cgm.shape[0]

        # Convertir JAX arrays a NumPy para procesamiento interno
        x_cgm_np = np.array(x_cgm)
        x_other_np = np.array(x_other)

        actions = np.zeros(batch_size, dtype=int)

        for i in range(batch_size):
            state = current_agent._map_observation_to_state(x_cgm_np[i], x_other_np[i])
            # Predicción greedy (sin exploración) basada en la Q-table aprendida
            actions[i] = np.argmax(current_agent.q_table[state])

        # Devolver acciones como predicciones (reshape y convertir a JAX array float32)
        # La predicción en este contexto es la acción óptima estimada (dosis discreta).
        return jnp.array(actions.reshape(-1, 1), dtype=jnp.float32)

    def evaluate(self, agent_state: Any, batch_data: Tuple, rng_key: jax.random.PRNGKey) -> Dict[str, float]:
        """
        Evalúa el rendimiento del agente en un conjunto de datos. Compatible con RLModelWrapperJAX.
        Calcula métricas comparando las acciones predichas (dosis discretas) con los targets (dosis reales/continuas).

        Parámetros:
        -----------
        agent_state : Any
             Estado actual del agente (la instancia 'self').
        batch_data : Tuple
            Tupla conteniendo ((observaciones_cgm, observaciones_other), targets).
            `targets` son los valores reales (posiblemente continuos).
        rng_key : jax.random.PRNGKey
            Clave JAX para aleatoriedad.

        Retorna:
        --------
        Dict[str, float]
            Diccionario con métricas de evaluación (ej. 'loss' como MSE, 'mae').
        """
        current_agent = agent_state
        (observations_cgm, observations_other), targets = batch_data

        # Realizar predicciones (acciones discretas)
        predictions_jax = current_agent.predict_batch(agent_state, (observations_cgm, observations_other), rng_key)

        # Convertir a NumPy para cálculo de métricas
        # Las predicciones son acciones discretas, los targets son valores reales (posiblemente continuos)
        predictions_np = np.array(predictions_jax).flatten() # Acciones discretas predichas
        targets_np = np.array(targets).flatten() # Valores reales

        # Calcular métricas comparando la acción discreta predicha con el target real.
        # MSE y MAE se calculan directamente.
        loss = np.mean((predictions_np - targets_np)**2) # MSE entre acción discreta y target real
        mae = np.mean(np.abs(predictions_np - targets_np)) # MAE entre acción discreta y target real

        return {'loss': float(loss), 'mae': float(mae)}

    def _map_observation_to_state(self, cgm_obs: np.ndarray, other_obs: np.ndarray) -> int:
        """
        Mapea una observación continua/compleja a un estado discreto.
        Esta es una función placeholder y necesita una implementación real
        basada en la naturaleza de los datos y la discretización elegida.

        Parámetros:
        -----------
        cgm_obs : np.ndarray
            Observación CGM para una muestra.
        other_obs : np.ndarray
            Otras características para una muestra.

        Retorna:
        --------
        int
            El estado discreto correspondiente.
        """
        # ¡¡¡ESTA IMPLEMENTACIÓN ES UN PLACEHOLDER MUY BÁSICO!!!
        # Se necesita una discretización significativa del espacio de estados.
        # Ejemplo: usar cuantiles, clustering, o una red neuronal para extraer características clave.
        
        # Ejemplo simplista basado en media CGM y alguna otra característica:
        mean_cgm = np.mean(cgm_obs) if cgm_obs.size > 0 else 0.0
        
        other_feature_summary = 0.0
        if other_obs is not None and other_obs.size > 0 and np.issubdtype(other_obs.dtype, np.number):
             # Usar la media si hay múltiples características, o el valor si es una sola
             other_feature_summary = np.mean(other_obs) if other_obs.ndim > 0 and other_obs.size > 1 else other_obs.item()

        # Combinar características (ejemplo: suma ponderada o concatenación seguida de hash)
        # Aquí usamos una combinación lineal simple para el ejemplo
        combined_feature = 0.7 * mean_cgm + 0.3 * other_feature_summary

        # Discretizar la característica combinada en n_states bins
        # Ejemplo: Mapeo lineal simple (puede no ser efectivo)
        min_val, max_val = 50, 250 # Rango esperado aproximado de combined_feature
        state = int(np.floor(((combined_feature - min_val) / (max_val - min_val)) * self.n_states))

        # Asegurar que el estado esté dentro de los límites [0, n_states-1]
        return max(0, min(state, self.n_states - 1))

    # --- Métodos auxiliares (ejemplos que podrían ser útiles con episodios completos) ---
    def run_episode(self, env): # Necesitaría un entorno
        episode = []
        state_obs = env.reset()
        done = False
        while not done:
            state = self._map_observation_to_state(state_obs['cgm'], state_obs['other'])
            action = self.select_action(state, explore=True) # Necesita método select_action
            next_state_obs, reward, done, _ = env.step(action)
            episode.append((state, action, reward))
            state_obs = next_state_obs
        return episode

    def update_q_table_from_episode(self, episode):
        G = 0
        visited_state_action = set()
        for t in reversed(range(len(episode))):
            state, action, reward = episode[t]
            G = self.gamma * G + reward
            pair = (state, action)
    
            is_first_visit = pair not in visited_state_action
            if self.first_visit:
                visited_state_action.add(pair)
    
            if (self.first_visit and is_first_visit) or (not self.first_visit):
                self.n_table[state, action] += 1
                self.q_table[state, action] += (G - self.q_table[state, action]) / self.n_table[state, action]
                # Actualizar política si es control (ej. epsilon-greedy implícito por argmax)

    def select_action(self, state, explore=True):
        if explore and self.rng_np.random() < self.epsilon:
            return self.rng_np.integers(self.n_actions)
        else:
            return np.argmax(self.q_table[state])

def create_monte_carlo_agent(cgm_shape: Tuple[int, ...], other_features_shape: Tuple[int, ...], **kwargs) -> MonteCarlo:
    """
    Crea una instancia del agente MonteCarlo.
    Esta función será usada por RLModelWrapperJAX.

    Parámetros:
    -----------
    cgm_shape : Tuple[int, ...]
        Forma de los datos CGM.
    other_features_shape : Tuple[int, ...]
        Forma de otras características.
    **kwargs
        Argumentos adicionales (seed, n_states, n_actions, etc.).

    Retorna:
    --------
    MonteCarlo
        Instancia del agente MonteCarlo.
    """
    # Configurar el tamaño del espacio de estados y acciones
    n_states = kwargs.get('n_states', 100)  # Ejemplo: discretización del espacio de estados
    n_actions = kwargs.get('n_actions', 20)   # Ejemplo: niveles discretos de dosis de insulina
    seed = kwargs.get('seed', CONST_DEFAULT_SEED)

    mc_agent = MonteCarlo(
        n_states=n_states,
        n_actions=n_actions,
        gamma=kwargs.get('gamma', MONTE_CARLO_CONFIG['gamma']),
        epsilon_start=kwargs.get('epsilon_start', MONTE_CARLO_CONFIG['epsilon_start']),
        epsilon_end=kwargs.get('epsilon_end', MONTE_CARLO_CONFIG['epsilon_end']),
        epsilon_decay=kwargs.get('epsilon_decay', MONTE_CARLO_CONFIG['epsilon_decay']),
        first_visit=kwargs.get('first_visit', MONTE_CARLO_CONFIG['first_visit']),
        seed=seed,
        cgm_shape=cgm_shape,
        other_features_shape=other_features_shape
    )
    return mc_agent


def create_monte_carlo_model(cgm_shape: Tuple[int, ...], other_features_shape: Tuple[int, ...], **model_kwargs) -> RLModelWrapperJAX:
    """
    Crea un wrapper RLModelWrapperJAX para el agente Monte Carlo.

    Parámetros:
    -----------
    cgm_shape : Tuple[int, ...]
        Forma de los datos CGM (time_steps, features).
    other_features_shape : Tuple[int, ...]
        Forma de otras características (n_features,).
    **model_kwargs
        Argumentos adicionales para `create_monte_carlo_agent`.

    Retorna:
    --------
    RLModelWrapperJAX
        Wrapper RL para el agente Monte Carlo.
    """
    # Pasar la función creadora del agente y los kwargs al wrapper JAX
    model = RLModelWrapperJAX(
        agent_creator=create_monte_carlo_agent,
        cgm_shape=cgm_shape,
        other_features_shape=other_features_shape,
        **model_kwargs # Pasar kwargs al wrapper, que los pasará al creador
    )
    
    # Configurar early stopping
    patience = EARLY_STOPPING_POLICY.get('patience', 10)
    min_delta = EARLY_STOPPING_POLICY.get('min_delta', 0.01)
    restore_best_weights = EARLY_STOPPING_POLICY.get('restore_best_weights', True)
    model.add_early_stopping(patience=patience, min_delta=min_delta, restore_best_weights=restore_best_weights)
    
    return model

def model_creator() -> Callable[[Tuple[int, ...], Tuple[int, ...], Dict], RLModelWrapperJAX]:
    """
    Retorna una función para crear un modelo Monte Carlo envuelto en RLModelWrapperJAX,
    compatible con la API del sistema.

    Retorna:
    --------
    Callable[[Tuple[int, ...], Tuple[int, ...], Dict], RLModelWrapperJAX]
        Función que, dadas las formas de entrada y kwargs, crea el wrapper RL para Monte Carlo.
    """
    # La función devuelta ahora acepta kwargs
    def creator_with_kwargs(cgm_shape: Tuple[int, ...], other_features_shape: Tuple[int, ...], **kwargs) -> RLModelWrapperJAX:
        # Asegurarse de que n_states y n_actions estén en kwargs si no se proporcionan explícitamente
        kwargs.setdefault('n_states', 100)
        kwargs.setdefault('n_actions', 20)
        return create_monte_carlo_model(cgm_shape, other_features_shape, **kwargs)
    return creator_with_kwargs