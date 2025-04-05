import os, sys
import numpy as np
import matplotlib.pyplot as plt
import pickle
import time
import gym
from typing import Dict, List, Tuple, Optional, Union, Any
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Input, Concatenate, GlobalAveragePooling1D
from keras.saving import register_keras_serializable

PROJECT_ROOT = os.path.abspath(os.getcwd())
sys.path.append(PROJECT_ROOT) 

from config.models_config import SARSA_CONFIG


class SARSA:
    """
    Implementación del algoritmo SARSA (State-Action-Reward-State-Action).
    
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
        
        # Configurar generador de números aleatorios
        self.rng = np.random.Generator(np.random.PCG64(seed))
        
        # Configurar espacio de estados y tabla Q
        self.discrete_state_space = hasattr(env.observation_space, 'n')
        
        if self.discrete_state_space:
            self._setup_discrete_state_space()
        else:
            self._setup_continuous_state_space()
        
        # Métricas para seguimiento del entrenamiento
        self.episode_rewards = []
        self.episode_lengths = []
        self.epsilon_history = []
        
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
        self.q_table = np.zeros((self.state_space_size, self.action_space_size))
        
        if self.config['optimistic_initialization']:
            self.q_table += self.config['optimistic_initial_value']
    
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
        self.q_table = np.zeros(q_shape)
        
        if self.config['optimistic_initialization']:
            self.q_table += self.config['optimistic_initial_value']
    
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
        
        # Exploración con probabilidad epsilon
        if explore and self.rng.random() < self.epsilon:
            return self.rng.integers(0, self.action_space_size)
        
        # Explotación: elegir acción con mayor valor Q
        return np.argmax(self.q_table[discrete_state])
    
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
        
        # Valor Q actual
        current_q = self.q_table[discrete_state][action]
        
        # Valor Q del siguiente estado-acción
        next_q = self.q_table[discrete_next_state][next_action]
        
        # Actualizar Q usando la regla SARSA
        # Q(s,a) = Q(s,a) + α[r + γQ(s',a') - Q(s,a)]
        new_q = current_q + self.alpha * (reward + self.gamma * next_q - current_q)
        self.q_table[discrete_state][action] = new_q
    
    def decay_epsilon(self, episode: Optional[int] = None) -> None:
        """
        Actualiza epsilon según el esquema de decaimiento configurado.
        
        Parámetros:
        -----------
        episode : Optional[int], opcional
            Número de episodio actual (para decaimiento lineal) (default: None)
        """
        if self.decay_type == 'exponential':
            self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
        elif self.decay_type == 'linear':
            # Requiere conocer el número total de episodios para el decaimiento lineal
            if episode is not None and self.config['episodes'] > 0:
                self.epsilon = max(
                    self.epsilon_min,
                    self.epsilon_min + (self.config['epsilon_start'] - self.epsilon_min) * 
                    (1 - episode / self.config['episodes'])
                )
    
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
        self.episode_rewards = []
        self.episode_lengths = []
        self.epsilon_history = []
        
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
            self.episode_rewards.append(episode_reward)
            self.episode_lengths.append(episode_steps)
            self.epsilon_history.append(self.epsilon)
            
            # Actualizar epsilon para el siguiente episodio
            self.decay_epsilon(episode)
            
            # Mostrar progreso periódicamente
            if (episode + 1) % self.config['log_interval'] == 0:
                avg_reward = np.mean(self.episode_rewards[-self.config['log_interval']:])
                avg_steps = np.mean(self.episode_lengths[-self.config['log_interval']:])
                elapsed = time.time() - start_time
                print(f"Episodio {episode+1}/{episodes} - Recompensa: {avg_reward:.2f}, "
                      f"Pasos: {avg_steps:.2f}, Epsilon: {self.epsilon:.4f}, "
                      f"Tiempo: {elapsed:.2f}s")
        
        print("Entrenamiento completado!")
        
        # Retornar historial de entrenamiento
        return {
            'rewards': self.episode_rewards,
            'lengths': self.episode_lengths,
            'epsilons': self.epsilon_history
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
        data = {
            'q_table': self.q_table,
            'discrete_states': self.discrete_states if not self.discrete_state_space else None,
            'state_bounds': self.state_bounds if not self.discrete_state_space else None,
            'discrete_state_space': self.discrete_state_space,
            'config': self.config,
            'episode_rewards': self.episode_rewards,
            'episode_lengths': self.episode_lengths,
            'epsilon_history': self.epsilon_history,
            'epsilon': self.epsilon
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
        
        self.q_table = data['q_table']
        self.discrete_state_space = data['discrete_state_space']
        
        if not self.discrete_state_space:
            self.discrete_states = data['discrete_states']
            self.state_bounds = data['state_bounds']
        
        self.config = data['config']
        self.alpha = self.config['learning_rate']
        self.gamma = self.config['gamma']
        
        # Cargar métricas de entrenamiento si están disponibles
        self.episode_rewards = data.get('episode_rewards', [])
        self.episode_lengths = data.get('episode_lengths', [])
        self.epsilon_history = data.get('epsilon_history', [])
        self.epsilon = data.get('epsilon', self.epsilon)
        
        print(f"Modelo cargado desde {filepath}")
    
    def visualize_training(
        self, 
        training_history: Optional[Dict[str, List[float]]] = None, 
        smoothing_window: Optional[int] = None
    ) -> None:
        """
        Visualiza las métricas de entrenamiento.
        
        Parámetros:
        -----------
        training_history : Optional[Dict[str, List[float]]], opcional
            Historia de entrenamiento (default: None)
        smoothing_window : Optional[int], opcional
            Tamaño de ventana para suavizado (default: None)
        """
        if training_history is None:
            # Usar historial almacenado internamente
            rewards = self.episode_rewards
            lengths = self.episode_lengths
            epsilons = self.epsilon_history
        else:
            # Usar historial proporcionado
            rewards = training_history['rewards']
            lengths = training_history['lengths']
            epsilons = training_history['epsilons']
        
        smoothing_window = smoothing_window or self.config['smoothing_window']
        
        # Función para suavizar datos
        def smooth(data: List[float], window_size: int) -> np.ndarray:
            """Aplica suavizado con media móvil"""
            if len(data) < window_size:
                return np.array(data)
            kernel = np.ones(window_size) / window_size
            return np.convolve(np.array(data), kernel, mode='valid')
        
        # Crear figura con tres subplots
        _, axs = plt.subplots(3, 1, figsize=(10, 12))
        
        # 1. Gráfico de recompensas
        axs[0].plot(rewards, alpha=0.3, color='blue', label='Original')
        if len(rewards) > smoothing_window:
            smoothed_rewards = smooth(rewards, smoothing_window)
            axs[0].plot(
                range(smoothing_window-1, len(rewards)),
                smoothed_rewards,
                color='blue',
                label=f'Suavizado (ventana={smoothing_window})'
            )
        axs[0].set_title('Recompensas por Episodio')
        axs[0].set_xlabel('Episodio')
        axs[0].set_ylabel('Recompensa')
        axs[0].legend()
        axs[0].grid(alpha=0.3)
        
        # 2. Gráfico de longitud de episodios
        axs[1].plot(lengths, alpha=0.3, color='green', label='Original')
        if len(lengths) > smoothing_window:
            smoothed_lengths = smooth(lengths, smoothing_window)
            axs[1].plot(
                range(smoothing_window-1, len(lengths)),
                smoothed_lengths,
                color='green',
                label=f'Suavizado (ventana={smoothing_window})'
            )
        axs[1].set_title('Longitud de Episodios')
        axs[1].set_xlabel('Episodio')
        axs[1].set_ylabel('Pasos')
        axs[1].legend()
        axs[1].grid(alpha=0.3)
        
        # 3. Gráfico de epsilon
        axs[2].plot(epsilons, color='red')
        axs[2].set_title('Valor de Epsilon (Exploración)')
        axs[2].set_xlabel('Episodio')
        axs[2].set_ylabel('Epsilon')
        axs[2].grid(alpha=0.3)
        
        plt.tight_layout()
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
        
        for i in range(resolution):
            for j in range(resolution):
                # Crear estado para este punto
                state = base_state.copy()
                state[dim1] = X[i, j]
                state[dim2] = Y[i, j]
                
                # Obtener valor máximo para este estado
                discrete_state = self.discretize_state(state)
                values[i, j] = np.max(self.q_table[discrete_state])
        
        # Visualizar valores
        plt.figure(figsize=(10, 8))
        value_plot = plt.pcolormesh(X, Y, values, cmap='plasma', alpha=0.7, shading='auto')
        plt.colorbar(value_plot, label='Valor')
        plt.title('Visualización de Función de Valor')
        plt.xlabel(f'Dimensión de Estado {dim1}')
        plt.ylabel(f'Dimensión de Estado {dim2}')
        plt.grid(alpha=0.2)
        plt.show()

@register_keras_serializable
class SARSAModel(Model):
    """
    Wrapper para el algoritmo SARSA que implementa la interfaz de Model de Keras.
    """
    
    def __init__(
        self, 
        sarsa_agent: SARSA,
        cgm_shape: Tuple[int, ...],
        other_features_shape: Tuple[int, ...],
        discretizer: Optional[Any] = None
    ) -> None:
        """
        Inicializa el modelo wrapper para SARSA.
        
        Parámetros:
        -----------
        sarsa_agent : SARSA
            Agente SARSA a utilizar
        cgm_shape : Tuple[int, ...]
            Forma de entrada para datos CGM
        other_features_shape : Tuple[int, ...]
            Forma de entrada para otras características
        discretizer : Optional[Any], opcional
            Función de discretización personalizada (default: None)
        """
        super(SARSAModel, self).__init__()
        self.sarsa_agent = sarsa_agent
        self.cgm_shape = cgm_shape
        self.other_features_shape = other_features_shape
        self.discretizer = discretizer
        
        # Capas para procesar entrada de CGM
        self.cgm_encoder = Dense(64, activation='relu', name='cgm_encoder')
        self.cgm_pooling = GlobalAveragePooling1D(name='cgm_pooling')
        
        # Capas para procesar otras características
        self.other_encoder = Dense(32, activation='relu', name='other_encoder')
        
        # Capa de concatenación
        self.concat = Concatenate(name='concat_layer')
        
        # Capa para codificar estado discreto
        self.state_encoder = Dense(
            sarsa_agent.state_space_size if hasattr(sarsa_agent, 'state_space_size') else 1000, 
            activation='softmax', 
            name='state_encoder'
        )
        
        # Capa para convertir acciones discretas a dosis continuas
        self.action_decoder = Dense(1, kernel_initializer='glorot_uniform', name='action_decoder')
        
    def call(self, inputs: List[tf.Tensor], training: bool = False) -> tf.Tensor:
        """
        Implementa la llamada del modelo para predicciones.
        
        Parámetros:
        -----------
        inputs : List[tf.Tensor]
            Lista de tensores [cgm_data, other_features]
        training : bool, opcional
            Indica si está en modo de entrenamiento (default: False)
            
        Retorna:
        --------
        tf.Tensor
            Predicciones basadas en la política actual
        """
        # Obtener y procesar entradas
        cgm_input, other_input = inputs
        batch_size = tf.shape(cgm_input)[0]
        
        # Codificar estado a partir de las entradas
        states = self._encode_states(cgm_input, other_input)
        
        # Inicializar array para acciones
        actions = tf.TensorArray(tf.float32, size=batch_size)
        
        # Para cada muestra en el batch, obtener acción correspondiente
        for i in range(batch_size):
            # Obtener índice del estado más probable
            state = tf.argmax(states[i]).numpy()
            # Usar sarsa para obtener acción (sin exploración en modo predicción)
            action = self.sarsa_agent.get_action(state, explore=False)
            actions = actions.write(i, tf.cast(action, tf.float32))
        
        # Convertir a tensor y ajustar forma
        actions_tensor = tf.reshape(actions.stack(), [batch_size, 1])
        
        # Decodificar acción discreta a valor de dosis
        dose_predictions = self._decode_actions(actions_tensor)
        
        return dose_predictions
    
    def _encode_states(self, cgm_data: tf.Tensor, other_features: tf.Tensor) -> tf.Tensor:
        """
        Codifica las características en estados discretos.
        
        Parámetros:
        -----------
        cgm_data : tf.Tensor
            Datos de monitoreo continuo de glucosa
        other_features : tf.Tensor
            Otras características (carbohidratos, insulina a bordo, etc.)
            
        Retorna:
        --------
        tf.Tensor
            Distribución suave sobre estados discretos
        """
        # Procesar CGM con capas densas
        cgm_encoded = self.cgm_encoder(cgm_data)
        cgm_features = self.cgm_pooling(cgm_encoded)
        
        # Procesar otras características
        other_encoded = self.other_encoder(other_features)
        
        # Concatenar características
        combined = self.concat([cgm_features, other_encoded])
        
        # Codificar a estados discretos
        state_distribution = self.state_encoder(combined)
        
        return state_distribution
    
    def _decode_actions(self, actions: tf.Tensor) -> tf.Tensor:
        """
        Decodifica acciones discretas a valores continuos de dosis.
        
        Parámetros:
        -----------
        actions : tf.Tensor
            Índices de acciones discretas
            
        Retorna:
        --------
        tf.Tensor
            Valores de dosis de insulina
        """
        # Convertir índices a one-hot para procesamiento
        action_one_hot = tf.one_hot(tf.cast(actions, tf.int32), self.sarsa_agent.action_space_size)
        # Mapear a valores continuos
        doses = self.action_decoder(action_one_hot)
        return doses
    
    def fit(
        self, 
        x: List[tf.Tensor], 
        y: tf.Tensor, 
        batch_size: int = 32, 
        epochs: int = 1, 
        verbose: int = 0,
        callbacks: Optional[List[Any]] = None,
        validation_data: Optional[Tuple] = None,
        **kwargs
    ) -> Dict:
        """
        Simula la interfaz de entrenamiento de Keras para el agente SARSA.
        
        Parámetros:
        -----------
        x : List[tf.Tensor]
            Lista con [cgm_data, other_features]
        y : tf.Tensor
            Etiquetas (dosis objetivo)
        batch_size : int, opcional
            Tamaño del lote (default: 32)
        epochs : int, opcional
            Número de épocas (default: 1)
        verbose : int, opcional
            Nivel de verbosidad (default: 0)
        callbacks : Optional[List[Any]], opcional
            Lista de callbacks (default: None)
        validation_data : Optional[Tuple], opcional
            Datos de validación (default: None)
        **kwargs
            Argumentos adicionales
            
        Retorna:
        --------
        Dict
            Historia simulada de entrenamiento
        """
        if verbose > 0:
            print("Entrenando modelo SARSA...")
        
        # Crear entorno personalizado para SARSA
        _ = self._create_environment(x[0], x[1], y)
        
        # Entrenar agente SARSA
        history = self.sarsa_agent.train(
            episodes=epochs * batch_size,
            max_steps=100,
            render=False
        )
        
        # Calibrar decodificador de acciones
        self._calibrate_action_decoder(y)
        
        if verbose > 0:
            print(f"Entrenamiento completado en {len(history.get('rewards', []))} episodios")
        
        # Crear historia simulada para compatibilidad con Keras
        keras_history = {
            'loss': history.get('rewards', [0]),
            'val_loss': [history.get('rewards', [0])[-1]] if validation_data is not None else None
        }
        
        return {'history': keras_history}
    
    def _create_environment(self, cgm_data: tf.Tensor, other_features: tf.Tensor, 
                           target_doses: tf.Tensor) -> Any:
        """
        Crea un entorno personalizado para entrenar el agente SARSA.
        
        Parámetros:
        -----------
        cgm_data : tf.Tensor
            Datos CGM
        other_features : tf.Tensor
            Otras características
        target_doses : tf.Tensor
            Dosis objetivo
            
        Retorna:
        --------
        Any
            Entorno compatible con OpenAI Gym
        """
        # Convertir tensores a numpy arrays
        cgm_np = cgm_data.numpy() if hasattr(cgm_data, 'numpy') else cgm_data
        other_np = other_features.numpy() if hasattr(other_features, 'numpy') else other_features
        target_np = target_doses.numpy() if hasattr(target_doses, 'numpy') else target_doses
        
        # Crear clase de entorno personalizada
        class InsulinDosingEnv:
            """Entorno personalizado para dosificación de insulina con SARSA."""
            
            def __init__(self, cgm, features, targets, model):
                self.cgm = cgm
                self.features = features
                self.targets = targets
                self.model = model
                self.rng = np.random.Generator(np.random.PCG64(42))
                self.current_idx = 0
                self.max_idx = len(targets) - 1
                
                # Espacios para compatibilidad con Gym
                self.observation_space = gym.spaces.Discrete(
                    model.sarsa_agent.state_space_size 
                    if hasattr(model.sarsa_agent, 'state_space_size') 
                    else 1000
                )
                self.action_space = gym.spaces.Discrete(model.sarsa_agent.action_space_size)
            
            def reset(self):
                """Reinicia el entorno a un estado aleatorio."""
                self.current_idx = self.rng.integers(0, self.max_idx)
                state = self._get_state()
                return state, {}
            
            def step(self, action):
                """Ejecuta un paso en el entorno con la acción dada."""
                # Convertir acción a dosis
                dose = action / (self.model.sarsa_agent.action_space_size - 1) * 15  # Max 15 U
                
                # Calcular recompensa (negativo del error absoluto)
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
                """Obtiene el estado discreto para el ejemplo actual."""
                # Obtener datos actuales
                cgm_batch = self.cgm[self.current_idx:self.current_idx+1]
                features_batch = self.features[self.current_idx:self.current_idx+1]
                
                # Codificar estado
                states = self.model._encode_states(
                    tf.convert_to_tensor(cgm_batch, dtype=tf.float32),
                    tf.convert_to_tensor(features_batch, dtype=tf.float32)
                )
                
                # Obtener índice del estado más probable
                state = tf.argmax(states, axis=1)[0].numpy()
                return int(state)
                
        # Actualizar el entorno en el agente SARSA
        env = InsulinDosingEnv(cgm_np, other_np, target_np, self)
        self.sarsa_agent.env = env
        
        return env
    
    def _calibrate_action_decoder(self, target_doses: tf.Tensor) -> None:
        """
        Calibra la capa decodificadora para mapear acciones a dosis apropiadas.
        
        Parámetros:
        -----------
        target_doses : tf.Tensor
            Dosis objetivo para calibración
        """
        doses_np = target_doses.numpy() if hasattr(target_doses, 'numpy') else target_doses
        max_dose = np.max(doses_np)
        min_dose = np.min(doses_np)
        
        # Configurar pesos para mapear del espacio discreto al rango de dosis
        self.action_decoder.set_weights([
            np.ones((self.sarsa_agent.action_space_size, 1)) * 
            (max_dose - min_dose) / self.sarsa_agent.action_space_size,
            np.array([min_dose])
        ])
    
    def predict(self, x: List[tf.Tensor], **kwargs) -> np.ndarray:
        """
        Implementa la interfaz de predicción de Keras.
        
        Parámetros:
        -----------
        x : List[tf.Tensor]
            Lista con [cgm_data, other_features]
        **kwargs
            Argumentos adicionales
            
        Retorna:
        --------
        np.ndarray
            Predicciones de dosis
        """
        return self.call(x).numpy()
    
    def get_config(self) -> Dict:
        """
        Obtiene la configuración del modelo para serialización.
        
        Retorna:
        --------
        Dict
            Configuración del modelo
        """
        return {
            "cgm_shape": self.cgm_shape,
            "other_features_shape": self.other_features_shape
        }
    
    def save(self, filepath: str) -> None:
        """
        Guarda el modelo SARSA.
        
        Parámetros:
        -----------
        filepath : str
            Ruta donde guardar el modelo
        """
        # Guardar pesos del modelo
        super().save_weights(filepath + WEIGHTS_FILE_SUFFIX)
        
        # Guardar agente SARSA
        self.sarsa_agent.save(filepath + SARSA_AGENT_SUFFIX)
        
    def load_weights(self, filepath: str) -> None:
        """
        Carga el modelo SARSA.
        
        Parámetros:
        -----------
        filepath : str
            Ruta desde donde cargar el modelo
        """
        # Determinar ruta correcta según formato de filepath
        if filepath.endswith(WEIGHTS_FILE_SUFFIX):
            weights_path = filepath
            sarsa_path = filepath.replace(WEIGHTS_FILE_SUFFIX, SARSA_AGENT_SUFFIX)
        else:
            weights_path = filepath + WEIGHTS_FILE_SUFFIX
            sarsa_path = filepath + SARSA_AGENT_SUFFIX
        
        # Cargar pesos del modelo
        super().load_weights(weights_path)
        
        # Cargar agente SARSA
        self.sarsa_agent.load(sarsa_path)


# Constantes para prevenir duplicación
CGM_ENCODER = 'cgm_encoder'
OTHER_ENCODER = 'other_encoder'
STATE_ENCODER = 'state_encoder'
ACTION_DECODER = 'action_decoder'
WEIGHTS_FILE_SUFFIX = '_weights.h5'
SARSA_AGENT_SUFFIX = '_sarsa_agent'


def create_sarsa_model(cgm_shape: Tuple[int, ...], other_features_shape: Tuple[int, ...]) -> Model:
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
    Model
        Modelo SARSA que implementa la interfaz de Keras
    """
    # Configuración del espacio de estados y acciones
    n_states = 1000  # Número de estados discretos
    n_actions = 20   # Número de acciones discretas (niveles de dosis)
    
    # Crear entorno dummy para inicializar SARSA
    class DummyEnv:
        def __init__(self):
            self.observation_space = gym.spaces.Discrete(n_states)
            self.action_space = gym.spaces.Discrete(n_actions)
    
    dummy_env = DummyEnv()
    
    # Crear agente SARSA
    sarsa_agent = SARSA(
        env=dummy_env,
        config={
            'learning_rate': SARSA_CONFIG['learning_rate'],
            'gamma': SARSA_CONFIG['gamma'],
            'epsilon_start': SARSA_CONFIG['epsilon_start'],
            'epsilon_end': SARSA_CONFIG['epsilon_end'],
            'epsilon_decay': SARSA_CONFIG['epsilon_decay'],
            'epsilon_decay_type': SARSA_CONFIG['epsilon_decay_type'],
            'optimistic_initialization': SARSA_CONFIG['optimistic_initialization'],
            'optimistic_initial_value': SARSA_CONFIG['optimistic_initial_value'],
            'log_interval': SARSA_CONFIG['log_interval'],
            'episodes': SARSA_CONFIG['episodes'],
            'max_steps': SARSA_CONFIG['max_steps'],
            'smoothing_window': SARSA_CONFIG['smoothing_window']
        },
        seed=42
    )
    
    # Crear y devolver el modelo wrapper
    return SARSAModel(
        sarsa_agent=sarsa_agent,
        cgm_shape=cgm_shape,
        other_features_shape=other_features_shape
    )