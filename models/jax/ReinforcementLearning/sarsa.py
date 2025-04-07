import os, sys
import jax
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
import pickle
import time
from typing import Dict, List, Tuple, Optional, Union, Any, NamedTuple, Callable
from functools import partial
from types import SimpleNamespace

PROJECT_ROOT = os.path.abspath(os.getcwd())
sys.path.append(PROJECT_ROOT) 

from config.models_config import SARSA_CONFIG

# Constantes para rutas de figuras y etiquetas comunes
FIGURES_DIR = os.path.join(PROJECT_ROOT, "figures", "jax", "sarsa")
CONST_EPISODE_LABEL = "Episodio"
CONST_REWARD_LABEL = "Recompensa" 
CONST_STEPS_LABEL = "Pasos"
CONST_EPSILON_LABEL = "Epsilon"
CONST_ORIGINAL_LABEL = "Original"
CONST_SMOOTHED_LABEL = "Suavizado"

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
        seed: int = 42
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
        """
        self.env = env
        self.config = config or SARSA_CONFIG
        
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
            
            # Manejar valores infinitos
            if low == float("-inf") or low < -1e6:
                low = -10.0
            if high == float("inf") or high > 1e6:
                high = 10.0
                
            bounds.append((low, high))
        return bounds
    
    def _create_discretization_bins(self) -> None:
        """
        Crea bins para discretización del espacio de estados continuo.
        """
        self.discrete_states = []
        for low, high in self.state_bounds:
            self.discrete_states.append(np.linspace(low, high, self.bins_per_dim + 1)[1:-1])
    
    def discretize_state(self, state: np.ndarray) -> Tuple:
        """
        Discretiza un estado continuo.
        
        Parámetros:
        -----------
        state : np.ndarray
            Estado continuo del entorno
            
        Retorna:
        --------
        Tuple
            Tupla con índices discretizados
        """
        if self.discrete_state_space:
            return state
        
        discrete_state = []
        for i, val in enumerate(state):
            # Limitar val al rango definido
            low, high = self.state_bounds[i]
            val = max(low, min(val, high))
            
            # Encontrar el bin correspondiente
            bins = self.discrete_states[i]
            digitized = np.digitize(val, bins)
            discrete_state.append(digitized)
        
        return tuple(discrete_state)
    
    def _get_action_fn(
        self, 
        q_table: jnp.ndarray, 
        discrete_state: Tuple, 
        rng_key: jnp.ndarray, 
        explore: bool
    ) -> Tuple[int, jnp.ndarray]:
        """
        Función pura para seleccionar acción con política epsilon-greedy.
        
        Parámetros:
        -----------
        q_table : jnp.ndarray
            Tabla Q actual
        discrete_state : Tuple
            Estado discretizado
        rng_key : jnp.ndarray
            Llave para generación de números aleatorios
        explore : bool
            Si se debe explorar o ser greedy
            
        Retorna:
        --------
        Tuple[int, jnp.ndarray]
            Acción seleccionada y nueva llave aleatoria
        """
        q_values = q_table[discrete_state]
        best_action = jnp.argmax(q_values)
        
        if not explore:
            return best_action, rng_key
        
        rng_key, explore_key, action_key = jax.random.split(rng_key, 3)
        should_explore = jax.random.uniform(explore_key) < self.epsilon
        random_action = jax.random.randint(action_key, shape=(), minval=0, maxval=self.action_space_size)
        
        action = jnp.where(should_explore, random_action, best_action)
        return int(action), rng_key
    
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
        action, new_rng_key = self._get_action_fn(
            self.state.q_table, discrete_state, self.state.rng_key, explore
        )
        
        # Actualizar estado de JAX (inmutable)
        self.state = self.state._replace(rng_key=new_rng_key)
        
        return action
    
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
    
    def _decay_epsilon_fn(self, epsilon: float, episode: Optional[int] = None) -> float:
        """
        Función pura para actualizar epsilon según el esquema de decaimiento.
        
        Parámetros:
        -----------
        epsilon : float
            Valor actual de epsilon
        episode : Optional[int], opcional
            Número de episodio actual (para decaimiento lineal) (default: None)
            
        Retorna:
        --------
        float
            Nuevo valor de epsilon
        """
        if self.decay_type == 'exponential':
            return max(self.epsilon_min, epsilon * self.epsilon_decay)
        elif self.decay_type == 'linear':
            # Requiere conocer el número total de episodios para el decaimiento lineal
            if episode is not None and self.config['episodes'] > 0:
                return max(
                    self.epsilon_min,
                    self.epsilon_min + (self.config['epsilon_start'] - self.epsilon_min) * 
                    (1 - episode / self.config['episodes'])
                )
        return epsilon
    
    def decay_epsilon(self, episode: Optional[int] = None) -> None:
        """
        Actualiza epsilon según el esquema de decaimiento configurado.
        
        Parámetros:
        -----------
        episode : Optional[int], opcional
            Número de episodio actual (para decaimiento lineal) (default: None)
        """
        new_epsilon = self._decay_epsilon_fn(self.state.epsilon, episode)
        self.state = self.state._replace(epsilon=new_epsilon)
    
    def train(
        self, 
        episodes: Optional[int] = None, 
        max_steps: Optional[int] = None, 
        render: bool = False
    ) -> Dict[str, List[float]]:
        """
        Entrena el agente SARSA.
        
        Parámetros:
        -----------
        episodes : Optional[int], opcional
            Número de episodios de entrenamiento (default: None)
        max_steps : Optional[int], opcional
            Límite de pasos por episodio (default: None)
        render : bool, opcional
            Si renderizar el entorno durante el entrenamiento (default: False)
            
        Retorna:
        --------
        Dict[str, List[float]]
            Diccionario con métricas de entrenamiento
        """
        episodes = episodes or self.config['episodes']
        max_steps = max_steps or self.config['max_steps']
        
        start_time = time.time()
        
        # Reiniciar listas para métricas
        episode_rewards = []
        episode_lengths = []
        epsilon_history = []
        
        for episode in range(episodes):
            state, _ = self.env.reset()
            action = self.get_action(state)  # Seleccionar primera acción
            
            episode_reward = 0
            episode_steps = 0
            
            # Interactuar con el entorno hasta terminar episodio
            for _ in range(max_steps):
                if render:
                    self.env.render()
                
                # Ejecutar acción y observar resultado
                next_state, reward, terminated, truncated, _ = self.env.step(action)
                done = terminated or truncated
                
                # Acumular recompensa
                episode_reward += reward
                episode_steps += 1
                
                if done:
                    # Para estados terminales, el valor Q del siguiente estado es 0
                    self.update_q_value(state, action, reward, next_state, 0)
                    break
                else:
                    # Seleccionar siguiente acción según política actual
                    next_action = self.get_action(next_state)
                    
                    # Actualizar valor Q
                    self.update_q_value(state, action, reward, next_state, next_action)
                    
                    # Actualizar estado y acción para la siguiente iteración
                    state = next_state
                    action = next_action
            
            # Recopilar métricas del episodio
            episode_rewards.append(episode_reward)
            episode_lengths.append(episode_steps)
            epsilon_history.append(float(self.state.epsilon))
            
            # Actualizar métricas en el estado inmutable
            self.state = self.state._replace(
                episode_rewards=episode_rewards,
                episode_lengths=episode_lengths,
                epsilon_history=epsilon_history
            )
            
            # Actualizar epsilon para el siguiente episodio
            self.decay_epsilon(episode)
            
            # Mostrar progreso periódicamente
            if (episode + 1) % self.config['log_interval'] == 0:
                avg_reward = np.mean(episode_rewards[-self.config['log_interval']:])
                avg_steps = np.mean(episode_lengths[-self.config['log_interval']:])
                elapsed = time.time() - start_time
                print(f"Episodio {episode+1}/{episodes} - Recompensa: {avg_reward:.2f}, "
                      f"Pasos: {avg_steps:.2f}, Epsilon: {self.state.epsilon:.4f}, "
                      f"Tiempo: {elapsed:.2f}s")
        
        print("Entrenamiento completado!")
        
        # Retornar historial de entrenamiento
        return {
            'rewards': episode_rewards,
            'lengths': episode_lengths,
            'epsilons': epsilon_history
        }
    
    def evaluate(
        self, 
        episodes: int = 10, 
        render: bool = True, 
        verbose: bool = True
    ) -> float:
        """
        Evalúa la política aprendida.
        
        Parámetros:
        -----------
        episodes : int, opcional
            Número de episodios para evaluación (default: 10)
        render : bool, opcional
            Si renderizar el entorno durante evaluación (default: True)
        verbose : bool, opcional
            Si mostrar resultados detallados (default: True)
            
        Retorna:
        --------
        float
            Recompensa promedio obtenida
        """
        rewards = []
        steps = []
        
        for episode in range(episodes):
            state, _ = self.env.reset()
            episode_reward = 0
            episode_steps = 0
            done = False
            
            while not done and episode_steps < self.config['max_steps']:
                if render:
                    self.env.render()
                
                # Seleccionar acción sin exploración
                action = self.get_action(state, explore=False)
                
                # Ejecutar acción
                next_state, reward, terminated, truncated, _ = self.env.step(action)
                done = terminated or truncated
                
                # Actualizar contadores
                episode_reward += reward
                episode_steps += 1
                state = next_state
            
            rewards.append(episode_reward)
            steps.append(episode_steps)
            
            if verbose:
                print(f"Episodio {episode+1}: Recompensa = {episode_reward}, Pasos = {episode_steps}")
        
        avg_reward = np.mean(rewards)
        avg_steps = np.mean(steps)
        
        print(f"Evaluación completada - Recompensa Media: {avg_reward:.2f}, Pasos Medios: {avg_steps:.2f}")
        
        return avg_reward
    
    def save(self, filepath: str) -> None:
        """
        Guarda la tabla Q y otra información del modelo.
        
        Parámetros:
        -----------
        filepath : str
            Ruta donde guardar el modelo
        """
        # Convertir la tabla Q de JAX a NumPy para serialización
        q_table_np = np.array(self.state.q_table)
        
        data = {
            'q_table': q_table_np,
            'discrete_states': self.discrete_states if not self.discrete_state_space else None,
            'state_bounds': self.state_bounds if not self.discrete_state_space else None,
            'discrete_state_space': self.discrete_state_space,
            'config': self.config,
            'episode_rewards': self.state.episode_rewards,
            'episode_lengths': self.state.episode_lengths,
            'epsilon_history': self.state.epsilon_history,
            'epsilon': float(self.state.epsilon)
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(data, f)
        
        print(f"Modelo guardado en {filepath}")
    
    def load(self, filepath: str) -> None:
        """
        Carga la tabla Q y otra información del modelo.
        
        Parámetros:
        -----------
        filepath : str
            Ruta desde donde cargar el modelo
        """
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
        
        # Cargar datos
        q_table = jnp.array(data['q_table'])
        self.discrete_state_space = data['discrete_state_space']
        
        if not self.discrete_state_space:
            self.discrete_states = data['discrete_states']
            self.state_bounds = data['state_bounds']
        
        self.config = data['config']
        self.alpha = self.config['learning_rate']
        self.gamma = self.config['gamma']
        
        # Actualizar estado inmutable
        self.state = SARSAState(
            q_table=q_table,
            rng_key=self.state.rng_key,  # Mantener llave aleatoria
            episode_rewards=data.get('episode_rewards', []),
            episode_lengths=data.get('episode_lengths', []),
            epsilon_history=data.get('epsilon_history', []),
            epsilon=data.get('epsilon', self.epsilon)
        )
        
        print(f"Modelo cargado desde {filepath}")
    
    def visualize_training(
        self, 
        training_history: Optional[Dict[str, List[float]]] = None, 
        smoothing_window: Optional[int] = None
    ) -> None:
        """
        Visualiza el historial de entrenamiento.
        
        Parámetros:
        -----------
        training_history : Optional[Dict[str, List[float]]], opcional
            Historial de entrenamiento (default: None, usa historial interno)
        smoothing_window : Optional[int], opcional
            Tamaño de ventana para suavizado (default: None, usa valor de configuración)
        """
        # Usar historial proporcionado o interno
        if training_history is None:
            rewards = self.state.episode_rewards
            lengths = self.state.episode_lengths
            epsilons = self.state.epsilon_history
        else:
            rewards = training_history.get('rewards', [])
            lengths = training_history.get('lengths', [])
            epsilons = training_history.get('epsilons', [])
        
        # Verificar que hay datos para graficar
        if not rewards:
            print("No hay datos de entrenamiento para visualizar")
            return
        
        # Usar ventana de suavizado de configuración si no se especifica
        smoothing_window = smoothing_window or self.config['smoothing_window']
        
        # Función para suavizar datos
        def smooth(data: List[float], window_size: int) -> np.ndarray:
            """Aplica suavizado con media móvil"""
            if len(data) < window_size:
                return np.array(data)
            kernel = np.ones(window_size) / window_size
            return np.convolve(np.array(data), kernel, mode='valid')
        
        # Crear directorio para figuras si no existe
        os.makedirs(FIGURES_DIR, exist_ok=True)
        
        # Crear figura con tres subplots
        _, axs = plt.subplots(3, 1, figsize=(10, 12))
        
        # 1. Gráfico de recompensas
        axs[0].plot(rewards, alpha=0.3, color='blue', label=CONST_ORIGINAL_LABEL)
        if len(rewards) > smoothing_window:
            smoothed_rewards = smooth(rewards, smoothing_window)
            axs[0].plot(
                range(smoothing_window-1, len(rewards)),
                smoothed_rewards,
                color='blue',
                label=f'{CONST_SMOOTHED_LABEL} (ventana={smoothing_window})'
            )
        axs[0].set_title(f'{CONST_REWARD_LABEL}s por {CONST_EPISODE_LABEL}')
        axs[0].set_xlabel(CONST_EPISODE_LABEL)
        axs[0].set_ylabel(CONST_REWARD_LABEL)
        axs[0].legend()
        axs[0].grid(alpha=0.3)
        
        # 2. Gráfico de longitud de episodios
        axs[1].plot(lengths, alpha=0.3, color='green', label=CONST_ORIGINAL_LABEL)
        if len(lengths) > smoothing_window:
            smoothed_lengths = smooth(lengths, smoothing_window)
            axs[1].plot(
                range(smoothing_window-1, len(lengths)),
                smoothed_lengths,
                color='green',
                label=f'{CONST_SMOOTHED_LABEL} (ventana={smoothing_window})'
            )
        axs[1].set_title(f'Longitud de {CONST_EPISODE_LABEL}s')
        axs[1].set_xlabel(CONST_EPISODE_LABEL)
        axs[1].set_ylabel(CONST_STEPS_LABEL)
        axs[1].legend()
        axs[1].grid(alpha=0.3)
        
        # 3. Gráfico de epsilon
        axs[2].plot(epsilons, color='red')
        axs[2].set_title(f'Valor de {CONST_EPSILON_LABEL} (Exploración)')
        axs[2].set_xlabel(CONST_EPISODE_LABEL)
        axs[2].set_ylabel(CONST_EPSILON_LABEL)
        axs[2].grid(alpha=0.3)
        
        plt.tight_layout()
        
        # Guardar figura
        plt.savefig(os.path.join(FIGURES_DIR, "entrenamiento_resumen.png"), dpi=300)
        plt.show()
    
    def visualize_policy(
        self, 
        state_dims: Tuple[int, int] = (0, 1), 
        resolution: int = 100
    ) -> None:
        """
        Visualiza la política aprendida para entornos con estados continuos.
        
        Parámetros:
        -----------
        state_dims : Tuple[int, int], opcional
            Tupla con las dos dimensiones del estado a visualizar (default: (0, 1))
        resolution : int, opcional
            Resolución de la visualización (default: 100)
        """
        if self.discrete_state_space:
            print("Visualización de política no disponible para espacios de estado discretos nativos")
            return
        
        if self.state_dim < 2:
            print("Visualización requiere al menos 2 dimensiones de estado")
            return
        
        # Crear directorio para figuras si no existe
        os.makedirs(FIGURES_DIR, exist_ok=True)
        
        # Crear malla para visualización
        dim1, dim2 = state_dims
        x_range = np.linspace(self.state_bounds[dim1][0], self.state_bounds[dim1][1], resolution)
        y_range = np.linspace(self.state_bounds[dim2][0], self.state_bounds[dim2][1], resolution)
        X, Y = np.meshgrid(x_range, y_range)
        
        # Calcular acciones para cada punto de la malla
        policy = np.zeros(X.shape, dtype=int)
        
        # Estado base (valores medios para dimensiones no visualizadas)
        base_state = []
        for i in range(self.state_dim):
            if i == dim1 or i == dim2:
                base_state.append(0)  # Placeholder
            else:
                # Usar punto medio para dimensiones no visualizadas
                base_state.append((self.state_bounds[i][0] + self.state_bounds[i][1]) / 2)
        
        for i in range(resolution):
            for j in range(resolution):
                # Crear estado para este punto
                state = base_state.copy()
                state[dim1] = X[i, j]
                state[dim2] = Y[i, j]
                
                # Obtener acción según política actual
                action = self.get_action(state, explore=False)
                policy[i, j] = action
        
        # Visualizar política
        plt.figure(figsize=(10, 8))
        policy_plot = plt.pcolormesh(X, Y, policy, cmap='viridis', alpha=0.7, shading='auto')
        plt.colorbar(policy_plot, label='Acción')
        plt.title('Visualización de Política')
        plt.xlabel(f'Dimensión de Estado {dim1}')
        plt.ylabel(f'Dimensión de Estado {dim2}')
        plt.grid(alpha=0.2)
        
        # Guardar figura
        plt.savefig(os.path.join(FIGURES_DIR, "politica_visualizacion.png"), dpi=300)
        plt.show()
    
    def visualize_value_function(
        self, 
        state_dims: Tuple[int, int] = (0, 1), 
        resolution: int = 100
    ) -> None:
        """
        Visualiza la función de valor aprendida para entornos con estados continuos.
        
        Parámetros:
        -----------
        state_dims : Tuple[int, int], opcional
            Tupla con las dos dimensiones del estado a visualizar (default: (0, 1))
        resolution : int, opcional
            Resolución de la visualización (default: 100)
        """
        if self.discrete_state_space:
            print("Visualización de valores no disponible para espacios de estado discretos nativos")
            return
        
        if self.state_dim < 2:
            print("Visualización requiere al menos 2 dimensiones de estado")
            return
        
        # Crear malla para visualización
        dim1, dim2 = state_dims
        x_range = np.linspace(self.state_bounds[dim1][0], self.state_bounds[dim1][1], resolution)
        y_range = np.linspace(self.state_bounds[dim2][0], self.state_bounds[dim2][1], resolution)
        X, Y = np.meshgrid(x_range, y_range)
        
        # Calcular valores para cada punto de la malla
        values = np.zeros(X.shape)
        
        # Estado base (valores medios para dimensiones no visualizadas)
        base_state = []
        for i in range(self.state_dim):
            if i == dim1 or i == dim2:
                base_state.append(0)  # Placeholder
            else:
                # Usar punto medio para dimensiones no visualizadas
                base_state.append((self.state_bounds[i][0] + self.state_bounds[i][1]) / 2)
        
        q_table_np = np.array(self.state.q_table)  # Convertir a NumPy para indexación
        
        for i in range(resolution):
            for j in range(resolution):
                # Crear estado para este punto
                state = base_state.copy()
                state[dim1] = X[i, j]
                state[dim2] = Y[i, j]
                
                # Obtener valor máximo para este estado
                discrete_state = self.discretize_state(state)
                values[i, j] = np.max(q_table_np[discrete_state])
        
        # Visualizar valores
        plt.figure(figsize=(10, 8))
        value_plot = plt.pcolormesh(X, Y, values, cmap='plasma', alpha=0.7, shading='auto')
        plt.colorbar(value_plot, label='Valor')
        plt.title('Visualización de Función de Valor')
        plt.xlabel(f'Dimensión de Estado {dim1}')
        plt.ylabel(f'Dimensión de Estado {dim2}')
        plt.grid(alpha=0.2)
        plt.show()

class SARSAWrapper:
    """
    Wrapper para hacer que SARSA sea compatible con la interfaz de modelos de aprendizaje profundo.
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
            Agente SARSA a utilizar
        cgm_shape : Tuple[int, ...]
            Forma de entrada para datos CGM
        other_features_shape : Tuple[int, ...]
            Forma de entrada para otras características
        """
        self.sarsa_agent = sarsa_agent
        self.cgm_shape = cgm_shape
        self.other_features_shape = other_features_shape
        
        # Para discretizar entradas continuas
        self.cgm_bins = 10
        self.other_bins = 5
        
        # Historial de entrenamiento
        self.history = {'loss': [], 'val_loss': []}
    
    def __call__(self, inputs: List[np.ndarray]) -> np.ndarray:
        """
        Implementa la interfaz de llamada para predicción.
        
        Parámetros:
        -----------
        inputs : List[np.ndarray]
            Lista con [cgm_data, other_features]
            
        Retorna:
        --------
        np.ndarray
            Predicciones de dosis de insulina
        """
        return self.predict(inputs)
    
    def predict(self, inputs: List[np.ndarray]) -> np.ndarray:
        """
        Realiza predicciones con el modelo SARSA.
        
        Parámetros:
        -----------
        inputs : List[np.ndarray]
            Lista con [cgm_data, other_features]
            
        Retorna:
        --------
        np.ndarray
            Predicciones de dosis de insulina
        """
        # Obtener entradas
        cgm_data, other_features = inputs
        batch_size = cgm_data.shape[0]
        
        # Crear array para resultados
        predictions = np.zeros((batch_size, 1))
        
        for i in range(batch_size):
            # Convertir datos continuos a estado discreto
            state = self._discretize_state(cgm_data[i], other_features[i])
            
            # Obtener acción según la política actual (sin exploración)
            action = self.sarsa_agent.get_action(state, explore=False)
            
            # Convertir acción discreta a dosis continua
            predictions[i, 0] = self._convert_action_to_dose(action)
        
        return predictions
    
    def _discretize_state(self, cgm_data: np.ndarray, other_features: np.ndarray) -> int:
        """
        Discretiza las entradas continuas a un índice de estado.
        
        Parámetros:
        -----------
        cgm_data : np.ndarray
            Datos CGM para un ejemplo
        other_features : np.ndarray
            Otras características para un ejemplo
            
        Retorna:
        --------
        int
            Índice de estado discretizado
        """
        # Extraer características clave de CGM
        cgm_flat = cgm_data.flatten()
        
        # Extraer métricas: promedio, último valor, pendiente, variabilidad
        cgm_mean = np.mean(cgm_flat)
        cgm_last = cgm_flat[-1]
        cgm_slope = cgm_flat[-1] - cgm_flat[0]
        cgm_std = np.std(cgm_flat)
        
        # Discretizar características de CGM
        cgm_mean_bin = min(int(cgm_mean / 300 * self.cgm_bins), self.cgm_bins - 1)
        cgm_last_bin = min(int(cgm_last / 300 * self.cgm_bins), self.cgm_bins - 1)
        cgm_slope_bin = min(int((cgm_slope + 100) / 200 * self.cgm_bins), self.cgm_bins - 1)
        cgm_std_bin = min(int(cgm_std / 50 * self.cgm_bins), self.cgm_bins - 1)
        
        # Usar las características más relevantes de otras variables
        # Asumiendo: carbInput, bgInput, IOB, ICR son las características importantes
        carb_bin = min(int(other_features[0] / 100 * self.other_bins), self.other_bins - 1)
        bg_bin = min(int(other_features[1] / 300 * self.other_bins), self.other_bins - 1)
        iob_bin = min(int(other_features[2] / 10 * self.other_bins), self.other_bins - 1)
        
        # Combinar bins para crear un índice de estado único
        # Usar una codificación posicional
        state_index = cgm_mean_bin
        state_index = state_index * self.cgm_bins + cgm_last_bin
        state_index = state_index * self.cgm_bins + cgm_slope_bin
        state_index = state_index * self.cgm_bins + cgm_std_bin
        state_index = state_index * self.other_bins + carb_bin
        state_index = state_index * self.other_bins + bg_bin
        state_index = state_index * self.other_bins + iob_bin
        
        return state_index
    
    def _convert_action_to_dose(self, action: int) -> float:
        """
        Convierte una acción discreta a dosis continua.
        
        Parámetros:
        -----------
        action : int
            Índice de acción discreta
            
        Retorna:
        --------
        float
            Dosis de insulina
        """
        # Mapear desde [0, action_space_size-1] a [0, 15] unidades de insulina
        return action * 15.0 / (self.sarsa_agent.action_space_size - 1)
    
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
        Entrena el modelo SARSA en los datos proporcionados.
        
        Parámetros:
        -----------
        x : List[np.ndarray]
            Lista con [cgm_data, other_features]
        y : np.ndarray
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
        cgm_data, other_features = x
        
        # Crear entorno personalizado para SARSA
        _ = self._create_training_environment(cgm_data, other_features, y)
        
        if verbose > 0:
            print("Entrenando modelo SARSA...")
        
        # Entrenar agente SARSA
        training_history = self.sarsa_agent.train(
            episodes=epochs,
            max_steps=batch_size,
            render=False
        )
        
        # Calcular pérdida en los datos de entrenamiento
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
        
        # Combinar historial de SARSA con métricas adicionales
        combined_history = {
            'episode_rewards': training_history['rewards'],
            'episode_lengths': training_history['lengths'],
            'epsilons': training_history['epsilons'],
            'loss': self.history['loss']
        }
        
        if validation_data:
            combined_history['val_loss'] = self.history['val_loss']
            
        return combined_history
    
    def _create_training_environment(
        self, 
        cgm_data: np.ndarray, 
        other_features: np.ndarray, 
        targets: np.ndarray
    ) -> Any:
        """
        Crea un entorno de entrenamiento personalizado para SARSA.
        
        Parámetros:
        -----------
        cgm_data : np.ndarray
            Datos CGM
        other_features : np.ndarray
            Otras características
        targets : np.ndarray
            Dosis objetivo
            
        Retorna:
        --------
        Any
            Entorno de entrenamiento compatible con SARSA
        """
        class InsulinDosingEnv:
            """Entorno personalizado para problema de dosificación de insulina."""
            
            def __init__(self, cgm, features, targets, model_wrapper):
                self.cgm = cgm
                self.features = features
                self.targets = targets
                self.model = model_wrapper
                self.rng = np.random.Generator(np.random.PCG64(42))
                self.current_idx = 0
                self.max_idx = len(targets) - 1
                
                # Para compatibilidad con interface de entorno
                # Definir espacios de acción y observación discretos
                self.action_space = SimpleNamespace(
                    n=model_wrapper.sarsa_agent.action_space_size,
                    sample=self._sample_action
                )
                
                # Espacio de observación discreto basado en bins de discretización
                n_states = model_wrapper.cgm_bins**4 * model_wrapper.other_bins**3
                self.observation_space = SimpleNamespace(
                    n=n_states,
                    low=0,
                    high=n_states-1
                )
            
            def _sample_action(self):
                """Muestrea una acción aleatoria del espacio discreto."""
                return self.rng.integers(0, self.action_space.n)
            
            def reset(self):
                """Reinicia el entorno eligiendo un ejemplo aleatorio."""
                self.current_idx = self.rng.integers(0, self.max_idx)
                state = self.model._discretize_state(
                    self.cgm[self.current_idx],
                    self.features[self.current_idx]
                )
                return state, {}
            
            def step(self, action):
                """Ejecuta un paso con la acción dada."""
                # Convertir acción a dosis
                dose = self.model._convert_action_to_dose(action)
                
                # Calcular recompensa como negativo del error absoluto
                target = self.targets[self.current_idx]
                reward = -abs(dose - target)
                
                # Avanzar al siguiente ejemplo
                _ = self.current_idx
                self.current_idx = (self.current_idx + 1) % self.max_idx
                
                # Discretizar próximo estado
                next_state = self.model._discretize_state(
                    self.cgm[self.current_idx],
                    self.features[self.current_idx]
                )
                
                # Terminal después de cada paso para entrenamiento episódico
                done = True
                
                return next_state, reward, done, False, {}
        
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
        self.sarsa_agent.save(filepath + "_sarsa.pkl")
        
        import pickle
        wrapper_data = {
            'cgm_shape': self.cgm_shape,
            'other_features_shape': self.other_features_shape,
            'cgm_bins': self.cgm_bins,
            'other_bins': self.other_bins
        }
        
        with open(filepath + "_wrapper.pkl", 'wb') as f:
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
        self.sarsa_agent.load(filepath + "_sarsa.pkl")
        
        import pickle
        with open(filepath + "_wrapper.pkl", 'rb') as f:
            wrapper_data = pickle.load(f)
        
        self.cgm_shape = wrapper_data['cgm_shape']
        self.other_features_shape = wrapper_data['other_features_shape']
        self.cgm_bins = wrapper_data['cgm_bins']
        self.other_bins = wrapper_data['other_bins']
        
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
            'cgm_bins': self.cgm_bins,
            'other_bins': self.other_bins,
            'sarsa_config': self.sarsa_agent.config
        }


def create_sarsa_model(cgm_shape: Tuple[int, ...], other_features_shape: Tuple[int, ...]) -> SARSAWrapper:
    """
    Crea un modelo basado en SARSA para predicción de dosis de insulina.
    
    Parámetros:
    -----------
    cgm_shape : Tuple[int, ...]
        Forma de los datos CGM (batch_size, time_steps, features)
    other_features_shape : Tuple[int, ...]
        Forma de otras características (batch_size, n_features)
        
    Retorna:
    --------
    SARSAWrapper
        Wrapper de SARSA que implementa la interfaz compatible con modelos de aprendizaje profundo
    """
    # Crear un entorno temporal para pasar al constructor de SARSA
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
        'learning_rate': 0.1,
        'epsilon_start': 0.3,
        'epsilon_end': 0.01,
        'epsilon_decay': 0.99,
        'gamma': 0.95,
        'episodes': 1000,
        'max_steps': 100,
        'log_interval': 100
    })
    
    # Crear agente SARSA
    sarsa_agent = SARSA(temp_env, sarsa_config, seed=42)
    
    # Crear y devolver wrapper
    return SARSAWrapper(sarsa_agent, cgm_shape, other_features_shape)

def model_creator() -> Callable[[Tuple[int, ...], Tuple[int, ...]], SARSAWrapper]:
    """
    Retorna una función para crear un modelo SARSA compatible con la API del sistema.
    
    Retorna:
    --------
    Callable[[Tuple[int, ...], Tuple[int, ...]], SARSAWrapper]
        Función para crear el modelo con las formas de entrada especificadas
    """
    return create_sarsa_model