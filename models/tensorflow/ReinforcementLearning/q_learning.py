import os, sys
import numpy as np
import matplotlib.pyplot as plt
import time
from typing import Dict, List, Tuple, Optional, Union, Any, Callable
import tensorflow as tf
from keras._tf_keras.keras.saving import register_keras_serializable
from types import SimpleNamespace

PROJECT_ROOT = os.path.abspath(os.getcwd())
sys.path.append(PROJECT_ROOT) 

from config.models_config import QLEARNING_CONFIG
from constants.constants import CONST_DEFAULT_SEED

class QLearning:
    """
    Implementación de Q-Learning tabular para espacios de estados y acciones discretos.
    
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
        seed: int = CONST_DEFAULT_SEED
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
        """
        self.n_states = n_states
        self.n_actions = n_actions
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.use_decay_schedule = use_decay_schedule
        self.decay_steps = decay_steps
        
        # Configurar generador de números aleatorios con semilla fija
        self.rng = np.random.Generator(np.random.PCG64(seed))
        
        # Inicializar la tabla Q con valores optimistas o ceros
        if QLEARNING_CONFIG['optimistic_init']:
            self.q_table = np.ones((n_states, n_actions)) * QLEARNING_CONFIG['optimistic_value']
        else:
            self.q_table = np.zeros((n_states, n_actions))
        
        # Contador total de pasos
        self.total_steps = 0
        
        # Métricas
        self.rewards_history = []
    
    def get_action(self, state: int) -> int:
        """
        Selecciona una acción usando la política epsilon-greedy.
        
        Parámetros:
        -----------
        state : int
            Estado actual
            
        Retorna:
        --------
        int
            Acción seleccionada
        """
        if self.rng.random() < self.epsilon:
            # Explorar: acción aleatoria
            return self.rng.integers(0, self.n_actions)
        else:
            # Explotar: mejor acción según la tabla Q
            return np.argmax(self.q_table[state])
    
    def update(self, state: int, action: int, reward: float, next_state: int, done: bool) -> float:
        """
        Actualiza la tabla Q usando la regla de Q-Learning.
        
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
            Si el episodio ha terminado
            
        Retorna:
        --------
        float
            TD-error calculado
        """
        # Calcular valor Q objetivo con Q-Learning (off-policy TD control)
        if done:
            # Si es estado terminal, no hay recompensa futura
            target_q = reward
        else:
            # Q-Learning: seleccionar máxima acción en estado siguiente (greedy)
            target_q = reward + self.gamma * np.max(self.q_table[next_state])
        
        # Valor Q actual
        current_q = self.q_table[state, action]
        
        # Calcular TD error
        td_error = target_q - current_q
        
        # Actualizar valor Q
        self.q_table[state, action] += self.learning_rate * td_error
        
        return td_error
    
    def update_epsilon(self, step: Optional[int] = None) -> None:
        """
        Actualiza el valor de epsilon según la política de decaimiento.
        
        Parámetros:
        -----------
        step : Optional[int], opcional
            Paso actual para decaimiento programado (default: None)
        """
        if self.use_decay_schedule == 'linear':
            # Decaimiento lineal
            if step is not None:
                fraction = min(1.0, step / self.decay_steps)
                self.epsilon = self.epsilon_start + fraction * (self.epsilon_end - self.epsilon_start)
            else:
                self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)
        elif self.use_decay_schedule == 'exponential':
            # Decaimiento exponencial
            self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)
        # Si no hay decay schedule, epsilon se mantiene constante
    
    def _run_episode(self, env: Any, max_steps: int, render: bool) -> Tuple[float, int]:
        """
        Ejecuta un episodio de entrenamiento.
        
        Parámetros:
        -----------
        env : Any
            Entorno compatible con OpenAI Gym
        max_steps : int
            Máximo número de pasos por episodio
        render : bool
            Si renderizar el entorno durante el entrenamiento
            
        Retorna:
        --------
        Tuple[float, int]
            Recompensa total y número de pasos del episodio
        """
        state, _ = env.reset()
        episode_reward = 0
        steps = 0
        
        for _ in range(max_steps):
            if render:
                env.render()
            
            # Seleccionar y ejecutar acción
            action = self.get_action(state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            
            # Actualizar tabla Q
            self.update(state, action, reward, next_state, done)
            
            # Actualizar contadores y estadísticas
            state = next_state
            episode_reward += reward
            steps += 1
            self.total_steps += 1
            
            # Actualizar epsilon por paso si corresponde
            if self.use_decay_schedule:
                self.update_epsilon(self.total_steps)
            
            if done:
                break
                
        return episode_reward, steps
    
    def _update_history(
        self, 
        history: Dict[str, List[float]], 
        episode_reward: float, 
        steps: int, 
        episode_rewards_window: List[float], 
        log_interval: int
    ) -> float:
        """
        Actualiza el historial de entrenamiento con los resultados del episodio.
        
        Parámetros:
        -----------
        history : Dict[str, List[float]]
            Historial de entrenamiento
        episode_reward : float
            Recompensa del episodio
        steps : int
            Pasos del episodio
        episode_rewards_window : List[float]
            Ventana de recompensas recientes
        log_interval : int
            Intervalo para el cálculo de promedio móvil
            
        Retorna:
        --------
        float
            Recompensa promedio actual
        """
        history['episode_rewards'].append(episode_reward)
        history['episode_lengths'].append(steps)
        history['epsilons'].append(self.epsilon)
        
        # Mantener una ventana de recompensas para promedio móvil
        episode_rewards_window.append(episode_reward)
        if len(episode_rewards_window) > log_interval:
            episode_rewards_window.pop(0)
        
        avg_reward = np.mean(episode_rewards_window)
        history['avg_rewards'].append(avg_reward)
        
        return avg_reward
    
    def train(
        self, 
        env: Any, 
        episodes: int = QLEARNING_CONFIG['episodes'],
        max_steps: int = QLEARNING_CONFIG['max_steps_per_episode'],
        render: bool = False,
        log_interval: int = QLEARNING_CONFIG['log_interval']
    ) -> Dict[str, List[float]]:
        """
        Entrena el agente Q-Learning en el entorno.
        
        Parámetros:
        -----------
        env : Any
            Entorno compatible con OpenAI Gym
        episodes : int, opcional
            Número de episodios para entrenar (default: QLEARNING_CONFIG['episodes'])
        max_steps : int, opcional
            Máximo número de pasos por episodio (default: QLEARNING_CONFIG['max_steps_per_episode'])
        render : bool, opcional
            Si renderizar el entorno durante el entrenamiento (default: False)
        log_interval : int, opcional
            Cada cuántos episodios mostrar estadísticas (default: QLEARNING_CONFIG['log_interval'])
            
        Retorna:
        --------
        Dict[str, List[float]]
            Historia de entrenamiento
        """
        history = {
            'episode_rewards': [],
            'episode_lengths': [],
            'epsilons': [],
            'avg_rewards': []
        }
        
        # Variables para seguimiento
        episode_rewards_window = []
        start_time = time.time()
        
        for episode in range(episodes):
            # Ejecutar episodio
            episode_reward, steps = self._run_episode(env, max_steps, render)
            
            # Actualizar epsilon después de cada episodio si no se hace por paso
            if not self.use_decay_schedule:
                self.update_epsilon()
            
            # Actualizar estadísticas
            avg_reward = self._update_history(history, episode_reward, steps, 
                                            episode_rewards_window, log_interval)
            
            # Mostrar progreso
            if (episode + 1) % log_interval == 0:
                elapsed_time = time.time() - start_time
                print(f"Episodio {episode+1}/{episodes} - "
                      f"Recompensa: {episode_reward:.2f}, "
                      f"Promedio: {avg_reward:.2f}, "
                      f"Epsilon: {self.epsilon:.4f}, "
                      f"Tiempo: {elapsed_time:.2f}s")
                start_time = time.time()
        
        return history
    
    def evaluate(self, env: Any, episodes: int = 10, render: bool = False) -> float:
        """
        Evalúa el agente entrenado.
        
        Parámetros:
        -----------
        env : Any
            Entorno compatible con OpenAI Gym
        episodes : int, opcional
            Número de episodios para evaluar (default: 10)
        render : bool, opcional
            Si renderizar el entorno durante la evaluación (default: False)
            
        Retorna:
        --------
        float
            Recompensa promedio
        """
        rewards = []
        episode_lengths = []
        
        for episode in range(episodes):
            state, _ = env.reset()
            episode_reward = 0
            steps = 0
            done = False
            
            while not done:
                if render:
                    env.render()
                
                # Seleccionar la mejor acción (sin exploración)
                action = np.argmax(self.q_table[state])
                
                # Ejecutar acción
                next_state, reward, terminated, truncated, _ = env.step(action)
                done = terminated or truncated
                
                # Actualizar estado y recompensa
                state = next_state
                episode_reward += reward
                steps += 1
            
            rewards.append(episode_reward)
            episode_lengths.append(steps)
            print(f"Episodio de evaluación {episode+1}/{episodes} - Recompensa: {episode_reward:.2f}")
        
        avg_reward = np.mean(rewards)
        avg_length = np.mean(episode_lengths)
        print(f"Recompensa promedio de evaluación: {avg_reward:.2f}, "
              f"Longitud promedio: {avg_length:.2f}")
        
        return avg_reward
    
    def save_qtable(self, filepath: str) -> None:
        """
        Guarda la tabla Q en un archivo.
        
        Parámetros:
        -----------
        filepath : str
            Ruta para guardar la tabla Q
        """
        np.save(filepath, self.q_table)
        print(f"Tabla Q guardada en {filepath}")
    
    def load_qtable(self, filepath: str) -> None:
        """
        Carga la tabla Q desde un archivo.
        
        Parámetros:
        -----------
        filepath : str
            Ruta para cargar la tabla Q
        """
        self.q_table = np.load(filepath)
        print(f"Tabla Q cargada desde {filepath}")
    
    def visualize_training(self, history: Dict[str, List[float]], window_size: int = 10) -> None:
        """
        Visualiza los resultados del entrenamiento.
        
        Parámetros:
        -----------
        history : Dict[str, List[float]]
            Historia de entrenamiento
        window_size : int, opcional
            Tamaño de ventana para suavizado (default: 10)
        """
        def smooth(data: List[float], window_size: int) -> np.ndarray:
            """Suaviza los datos usando una media móvil."""
            kernel = np.ones(window_size) / window_size
            return np.convolve(data, kernel, mode='valid')
        
        _, axs = plt.subplots(2, 2, figsize=(15, 10))
        
        # 1. Recompensas por episodio
        axs[0, 0].plot(history['episode_rewards'], alpha=0.3, color='blue', label='Original')
        if len(history['episode_rewards']) > window_size:
            axs[0, 0].plot(
                range(window_size-1, len(history['episode_rewards'])),
                smooth(history['episode_rewards'], window_size),
                color='blue',
                label=f'Suavizado (ventana={window_size})'
            )
        axs[0, 0].set_title('Recompensa por Episodio')
        axs[0, 0].set_xlabel('Episodio')
        axs[0, 0].set_ylabel('Recompensa')
        axs[0, 0].grid(alpha=0.3)
        axs[0, 0].legend()
        
        # 2. Recompensa promedio
        axs[0, 1].plot(history['avg_rewards'], color='green')
        axs[0, 1].set_title('Recompensa Promedio')
        axs[0, 1].set_xlabel('Episodio')
        axs[0, 1].set_ylabel('Recompensa Promedio')
        axs[0, 1].grid(alpha=0.3)
        
        # 3. Epsilon
        axs[1, 0].plot(history['epsilons'], color='red')
        axs[1, 0].set_title('Epsilon (Exploración)')
        axs[1, 0].set_xlabel('Episodio')
        axs[1, 0].set_ylabel('Epsilon')
        axs[1, 0].grid(alpha=0.3)
        
        # 4. Longitud de episodios
        axs[1, 1].plot(history['episode_lengths'], alpha=0.3, color='purple', label='Original')
        if len(history['episode_lengths']) > window_size:
            axs[1, 1].plot(
                range(window_size-1, len(history['episode_lengths'])),
                smooth(history['episode_lengths'], window_size),
                color='purple',
                label=f'Suavizado (ventana={window_size})'
            )
        axs[1, 1].set_title('Longitud de Episodio')
        axs[1, 1].set_xlabel('Episodio')
        axs[1, 1].set_ylabel('Pasos')
        axs[1, 1].grid(alpha=0.3)
        axs[1, 1].legend()
        
        plt.tight_layout()
        plt.show()
    
    def _setup_grid_visualization(self, ax: plt.Axes, rows: int, cols: int) -> None:
        """
        Configura la visualización básica de la cuadrícula.
        
        Parámetros:
        -----------
        ax : plt.Axes
            Eje para la visualización
        rows : int
            Número de filas en la grilla
        cols : int
            Número de columnas en la grilla
        """
        ax.set_xlim(0, cols)
        ax.set_ylim(0, rows)
        ax.invert_yaxis()
        
        # Dibujar líneas de cuadrícula
        for i in range(rows+1):
            ax.axhline(i, color='black', linewidth=0.5)
        
        for j in range(cols+1):
            ax.axvline(j, color='black', linewidth=0.5)
    
    def _draw_policy_arrows(self, ax: plt.Axes, row: int, col: int, state: int, arrows: Dict[int, Tuple[float, float]]) -> None:
        """
        Dibuja flechas para visualizar la política en una celda específica.
        
        Parámetros:
        -----------
        ax : plt.Axes
            Eje para la visualización
        row : int
            Fila de la celda
        col : int
            Columna de la celda
        state : int
            Estado correspondiente a la celda
        arrows : Dict[int, Tuple[float, float]]
            Mapeo de acciones a direcciones de flechas
        """
        q_values = self.q_table[state]
        
        # Verificar si todos los valores Q son iguales (sin preferencia)
        if np.all(q_values == q_values[0]):
            # Dibujar todas las acciones con color gris
            for action, (dx, dy) in arrows.items():
                ax.arrow(col + 0.5, row + 0.5, dx, dy, head_width=0.1, head_length=0.1, 
                        fc='gray', ec='gray', alpha=0.3)
        else:
            # Dibujar la acción con mayor valor Q
            best_action = np.argmax(q_values)
            dx, dy = arrows[best_action]
            ax.arrow(col + 0.5, row + 0.5, dx, dy, head_width=0.1, head_length=0.1, 
                    fc='blue', ec='blue')
            
            # Mostrar el valor Q
            ax.text(col + 0.5, row + 0.7, f"{float(q_values.max()):.2f}", 
                   ha='center', va='center', fontsize=8)
    
    def visualize_policy(self, env: Any, grid_size: Optional[Tuple[int, int]] = None) -> None:
        """
        Visualiza la política aprendida en un entorno de cuadrícula.
        
        Parámetros:
        -----------
        env : Any
            Entorno para obtener información sobre el espacio
        grid_size : Optional[Tuple[int, int]], opcional
            Tamaño de la cuadrícula (filas, columnas) (default: None)
        """
        # Determinar tamaño de la cuadrícula si no se proporciona
        if grid_size is None:
            # Intentar inferir del entorno
            if hasattr(env, 'shape'):
                rows, cols = env.shape
            else:
                # Valor por defecto
                rows = cols = int(np.sqrt(self.n_states))
        else:
            rows, cols = grid_size
        
        _, ax = plt.subplots(figsize=(8, 8))
        self._setup_grid_visualization(ax, rows, cols)
        
        # Mapeo de acciones a flechas: 0=izquierda, 1=abajo, 2=derecha, 3=arriba
        arrows = {
            0: (-0.2, 0),   # Izquierda
            1: (0, 0.2),    # Abajo
            2: (0.2, 0),    # Derecha
            3: (0, -0.2)    # Arriba
        }
        
        # Dibujar flechas para cada celda
        for row in range(rows):
            for col in range(cols):
                state = row * cols + col  # Esto puede variar según cómo se mapeen los estados
                self._draw_policy_arrows(ax, row, col, state, arrows)
        
        plt.title('Política Aprendida')
        plt.tight_layout()
        plt.show()
    
    def visualize_value_function(self, env: Any, grid_size: Optional[Tuple[int, int]] = None) -> None:
        """
        Visualiza la función de valor derivada de la tabla Q.
        
        Parámetros:
        -----------
        env : Any
            Entorno para obtener información sobre el espacio
        grid_size : Optional[Tuple[int, int]], opcional
            Tamaño de la cuadrícula (filas, columnas) (default: None)
        """
        # Determinar tamaño de la cuadrícula si no se proporciona
        if grid_size is None:
            # Intentar inferir del entorno
            if hasattr(env, 'shape'):
                rows, cols = env.shape
            else:
                # Valor por defecto
                rows = cols = int(np.sqrt(self.n_states))
        else:
            rows, cols = grid_size
        
        # Calcular el valor de cada estado como el máximo valor Q
        value_function = np.max(self.q_table, axis=1).reshape(rows, cols)
        
        _, ax = plt.subplots(figsize=(10, 8))
        im = ax.imshow(value_function, cmap='viridis')
        
        # Añadir barra de color
        cbar = plt.colorbar(im)
        cbar.set_label('Valor Esperado')
        
        # Añadir etiquetas
        for i in range(rows):
            for j in range(cols):
                state = i * cols + j
                value = np.max(self.q_table[state])
                ax.text(j, i, f"{value:.2f}", ha="center", va="center", 
                        color="w" if value < (value_function.max() + value_function.min())/2 else "black")
        
        plt.title('Función de Valor (V)')
        plt.tight_layout()
        plt.show()
    
    def compare_with_optimal(self, optimal_policy: np.ndarray) -> float:
        """
        Compara la política aprendida con una política óptima.
        
        Parámetros:
        -----------
        optimal_policy : np.ndarray
            Política óptima como array de acciones para cada estado
            
        Retorna:
        --------
        float
            Porcentaje de concordancia entre políticas
        """
        # Determinar la política greedy actual
        current_policy = np.argmax(self.q_table, axis=1)
        
        # Calcular concordancia
        matches = np.sum(current_policy == optimal_policy)
        match_percentage = matches / len(optimal_policy) * 100
        
        print(f"Concordancia con política óptima: {match_percentage:.2f}%")
        return match_percentage
    
    def get_q_table(self) -> np.ndarray:
        """
        Obtiene la tabla Q actual.
        
        Retorna:
        --------
        np.ndarray
            Tabla Q actual
        """
        return self.q_table.copy()

# Constantes para archivos
QTABLE_SUFFIX = "_qtable.npy"
WRAPPER_WEIGHTS_SUFFIX = "_wrapper_weights.h5"

# Constantes para mensajes
CONST_CREANDO_ENTORNO = "Creando entorno de entrenamiento..."
CONST_ENTRENANDO_AGENTE = "Entrenando agente Q-Learning..."
CONST_EVALUANDO = "Evaluando modelo..."
CONST_TRANSFORMANDO_ESTADO = "Transformando estado discreto..."
CONST_ERROR_VALOR = "Error: Valor fuera de rango detectado:"

@register_keras_serializable()
class QLearningModel(tf.keras.models.Model):
    """
    Wrapper para el agente Q-Learning que implementa la interfaz de Keras.Model.
    
    Esta clase permite que el algoritmo Q-Learning tabular se comporte
    como un modelo de Keras, facilitando su uso con la infraestructura
    existente de entrenamiento.
    """
    
    def __init__(
        self, 
        q_learning_agent: 'QLearning',
        cgm_shape: Tuple[int, ...],
        other_features_shape: Tuple[int, ...],
        **kwargs
    ) -> None:
        """
        Inicializa el modelo wrapper para Q-Learning.
        
        Parámetros:
        -----------
        q_learning_agent : QLearning
            Agente Q-Learning a encapsular
        cgm_shape : Tuple[int, ...]
            Forma de los datos CGM
        other_features_shape : Tuple[int, ...]
            Forma de otras características
        **kwargs
            Argumentos adicionales para el constructor de tf.keras.models.Model
        """
        super().__init__(**kwargs)
        
        self.q_learning_agent = q_learning_agent
        self.cgm_shape = cgm_shape
        self.other_features_shape = other_features_shape
        
        # Determinar dimensiones para discretización
        if len(cgm_shape) > 1:
            self.cgm_length = cgm_shape[0]
            self.cgm_features = cgm_shape[1] if len(cgm_shape) > 1 else 1
        else:
            self.cgm_length = cgm_shape[0]
            self.cgm_features = 1
            
        self.other_features_length = other_features_shape[0] if len(other_features_shape) > 0 else 0
        
        # Definir capas para codificación inicial (no se usarán para cálculo real,
        # solo para mantener la estructura esperada en Keras)
        self.cgm_dense = tf.keras.layers.Dense(self.cgm_length, name='cgm_encoder')
        self.other_dense = tf.keras.layers.Dense(self.other_features_length, name='other_encoder')
        self.output_dense = tf.keras.layers.Dense(1, name='output_layer')
        
        # Para mapeo de acciones discretas a dosis continuas
        self.action_space = q_learning_agent.n_actions
        self.min_dose = 0.0
        self.max_dose = 15.0  # Máxima dosis de insulina en unidades
        self.dose_values = np.linspace(self.min_dose, self.max_dose, self.action_space)
        
        # Historial de entrenamiento para compatibilidad con Keras
        self.history = {'loss': [], 'val_loss': []}
        
        # Para discretización de estados
        self.cgm_bins = 10  # Número de bins para discretizar CGM
        self.other_bins = 5  # Número de bins para discretizar otras características
        
        # Dimensiones de entrada calculadas para evitar errores
        self._build_inputs()
    
    def _build_inputs(self) -> None:
        """
        Construye las entradas del modelo para asegurar compatibilidad con Keras.
        """
        # Crear entradas simuladas para construir el modelo
        dummy_cgm = np.zeros((1,) + self.cgm_shape)
        dummy_other = np.zeros((1,) + self.other_features_shape)
        
        # Llamada inicial para construir
        _ = self([dummy_cgm, dummy_other])
    
    def call(self, inputs: List[tf.Tensor], training: bool = False) -> tf.Tensor:
        """
        Realiza una llamada al modelo con las entradas proporcionadas.
        
        Parámetros:
        -----------
        inputs : List[tf.Tensor]
            Lista conteniendo [cgm_data, other_features]
        training : bool, opcional
            Indica si está en modo entrenamiento (default: False)
            
        Retorna:
        --------
        tf.Tensor
            Predicciones del modelo
        """
        # Extraer inputs
        cgm_data, other_features = inputs
        batch_size = tf.shape(cgm_data)[0]
        
        # Para cada muestra en el lote, obtener la acción óptima según la tabla Q
        # y mapear a valores de dosis
        doses = tf.TensorArray(tf.float32, size=batch_size)
        
        for i in range(batch_size):
            # Extraer datos individuales
            single_cgm = cgm_data[i]
            single_other = other_features[i]
            
            # Convertir a numpy para procesamiento
            cgm_np = single_cgm.numpy() if hasattr(single_cgm, 'numpy') else single_cgm
            other_np = single_other.numpy() if hasattr(single_other, 'numpy') else single_other
            
            # Discretizar estado
            state = self._discretize_state(cgm_np, other_np)
            
            # Obtener acción óptima
            action = self.q_learning_agent.get_action(state)
            
            # Convertir acción a dosis continua
            dose = self._convert_action_to_dose(action)
            
            # Almacenar en el array
            doses = doses.write(i, dose)
        
        # Convertir a tensor
        return doses.stack()
    
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
        # Aplanar CGM si es necesario
        cgm_flat = cgm_data.flatten()
        
        # Extraer características clave de CGM
        cgm_mean = np.mean(cgm_flat) if len(cgm_flat) > 0 else 0.0
        cgm_last = cgm_flat[-1] if len(cgm_flat) > 0 else 0.0
        cgm_slope = cgm_flat[-1] - cgm_flat[0] if len(cgm_flat) > 1 else 0.0
        
        # Normalizar [0,1] y discretizar
        max_cgm = 300.0  # Valor máximo típico de CGM
        cgm_mean_bin = min(int(cgm_mean / max_cgm * self.cgm_bins), self.cgm_bins - 1)
        cgm_last_bin = min(int(cgm_last / max_cgm * self.cgm_bins), self.cgm_bins - 1)
        cgm_slope_bin = min(int((cgm_slope + 50) / 100.0 * self.cgm_bins), self.cgm_bins - 1)
        
        # Procesamiento de otras características
        other_flat = other_features.flatten()
        other_bins = []
        
        # Usar solo las primeras características más relevantes
        n_features = min(3, len(other_flat))
        for i in range(n_features):
            if i < len(other_flat):
                # Normalizar a [0,1] y discretizar
                normalized_value = min(max(0.0, (other_flat[i] + 1) / 2), 1.0)
                bin_value = min(int(normalized_value * self.other_bins), self.other_bins - 1)
                other_bins.append(bin_value)
            else:
                other_bins.append(0)
        
        # Calcular índice de estado combinado
        state = cgm_mean_bin
        state = state * self.cgm_bins + cgm_last_bin
        state = state * self.cgm_bins + cgm_slope_bin
        
        for b in other_bins:
            state = state * self.other_bins + b
            
        return min(state, self.q_learning_agent.n_states - 1)
    
    def _convert_action_to_dose(self, action: int) -> float:
        """
        Convierte una acción discreta a un valor de dosis continuo.
        
        Parámetros:
        -----------
        action : int
            Índice de acción discreta
            
        Retorna:
        --------
        float
            Valor de dosis correspondiente
        """
        if action < 0 or action >= len(self.dose_values):
            print(f"{CONST_ERROR_VALOR} {action}")
            return self.min_dose
        
        return self.dose_values[action]
    
    def _create_environment(self, cgm_data: tf.Tensor, other_features: tf.Tensor, 
                           target_doses: tf.Tensor) -> Any:
        """
        Crea un entorno de entrenamiento para el agente de Q-Learning.
        
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
            Entorno para entrenamiento
        """
        print(CONST_CREANDO_ENTORNO)
        
        # Convertir tensores a numpy
        cgm_np = cgm_data.numpy() if hasattr(cgm_data, 'numpy') else cgm_data
        other_np = other_features.numpy() if hasattr(other_features, 'numpy') else other_features
        targets_np = target_doses.numpy() if hasattr(target_doses, 'numpy') else target_doses
        
        class InsulinDosingEnv:
            """Entorno personalizado para problema de dosificación de insulina."""
            
            def __init__(self, cgm, features, targets, model):
                self.cgm = cgm
                self.features = features
                self.targets = targets
                self.model = model
                self.rng = np.random.Generator(np.random.PCG64(CONST_DEFAULT_SEED))
                self.current_idx = 0
                self.max_idx = len(targets) - 1
                
                # Para compatibilidad con algoritmos RL
                self.observation_space = SimpleNamespace(
                    shape=(1,),
                    low=0,
                    high=model.q_learning_agent.n_states - 1
                )
                
                self.action_space = SimpleNamespace(
                    n=model.q_learning_agent.n_actions,
                    sample=self._sample_action
                )
            
            def _sample_action(self):
                """Muestrea una acción aleatoria del espacio discreto."""
                return self.rng.integers(0, self.model.q_learning_agent.n_actions)
            
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
                error = abs(dose - target)
                # Función de recompensa que prioriza precisión
                if error < 0.5:
                    reward = 1.0 - error  # Recompensa alta para error bajo
                else:
                    reward = -error  # Penalización para errores grandes
                
                # Avanzar al siguiente ejemplo
                self.current_idx = (self.current_idx + 1) % self.max_idx
                
                # Obtener próximo estado
                next_state = self.model._discretize_state(
                    self.cgm[self.current_idx],
                    self.features[self.current_idx]
                )
                
                # Terminal después de cada paso
                done = True
                
                # Información adicional
                info = {
                    'dose': dose,
                    'target': target,
                    'error': error
                }
                
                return next_state, reward, done, False, info
        
        return InsulinDosingEnv(cgm_np, other_np, targets_np, self)
    
    def _calibrate_action_decoder(self, y: tf.Tensor) -> None:
        """
        Calibra el decodificador de acciones basado en los objetivos.
        
        Parámetros:
        -----------
        y : tf.Tensor
            Tensor con valores objetivo (dosis)
        """
        # Convertir a numpy
        y_np = y.numpy() if hasattr(y, 'numpy') else y
        
        # Determinar rango de dosis
        min_dose = max(0.0, np.min(y_np) * 0.8)  # 20% menos que el mínimo, pero no negativo
        max_dose = np.max(y_np) * 1.2  # 20% más que el máximo
        
        # Actualizar valores de dosis
        self.min_dose = min_dose
        self.max_dose = max_dose
        self.dose_values = np.linspace(self.min_dose, self.max_dose, self.action_space)
    
    def fit(
        self, 
        x: List[tf.Tensor], 
        y: tf.Tensor, 
        validation_data: Optional[Tuple] = None, 
        epochs: int = 1,
        batch_size: int = 32,
        callbacks: List = None,
        verbose: int = 0
    ) -> Dict:
        """
        Entrena el modelo Q-Learning con los datos proporcionados.
        
        Parámetros:
        -----------
        x : List[tf.Tensor]
            Lista conteniendo [cgm_data, other_features]
        y : tf.Tensor
            Tensor con valores objetivo (dosis)
        validation_data : Optional[Tuple], opcional
            Datos de validación como (x_val, y_val) (default: None)
        epochs : int, opcional
            Número de épocas para entrenamiento (default: 1)
        batch_size : int, opcional
            Tamaño de lote (default: 32)
        callbacks : List, opcional
            Lista de callbacks de Keras (default: None)
        verbose : int, opcional
            Nivel de detalle para logs (default: 0)
            
        Retorna:
        --------
        Dict
            Historial de entrenamiento
        """
        # Extraer datos
        cgm_data, other_features = x
        
        # Calibrar decodificador de acciones
        self._calibrate_action_decoder(y)
        
        # Crear entorno para entrenamiento
        env = self._create_environment(cgm_data, other_features, y)
        
        print(CONST_ENTRENANDO_AGENTE)
        
        # Entrenar agente Q-Learning
        history = self.q_learning_agent.train(
            env=env,
            episodes=epochs * (len(y) // batch_size or 1),  # Aproximar al número total de pasos
            max_steps=1,  # Cada ejemplo es un episodio
            render=False,
            log_interval=max(10, (epochs * len(y) // batch_size) // 10)  # 10 logs durante entrenamiento
        )
        
        # Guardar historial para compatibilidad con Keras
        self.history = {
            'loss': history['episode_rewards'],
            'val_loss': []
        }
        
        return self.history
    
    def predict(self, x: List[tf.Tensor], **kwargs) -> np.ndarray:
        """
        Realiza predicciones con el modelo entrenado.
        
        Parámetros:
        -----------
        x : List[tf.Tensor]
            Lista conteniendo [cgm_data, other_features]
        **kwargs
            Argumentos adicionales para compatibilidad con Keras
            
        Retorna:
        --------
        np.ndarray
            Predicciones del modelo
        """
        # Usar el método call para predicción
        predictions = self.call(x)
        
        # Convertir a numpy para consistencia
        if hasattr(predictions, 'numpy'):
            return predictions.numpy()
        return predictions
    
    def get_config(self) -> Dict:
        """
        Obtiene la configuración del modelo para serialización.
        
        Retorna:
        --------
        Dict
            Configuración del modelo
        """
        config = super().get_config()
        config.update({
            'cgm_shape': self.cgm_shape,
            'other_features_shape': self.other_features_shape,
            'cgm_bins': self.cgm_bins,
            'other_bins': self.other_bins,
            'min_dose': self.min_dose,
            'max_dose': self.max_dose
        })
        return config
    
    def save(self, filepath: str, **kwargs) -> None:
        """
        Guarda el modelo y la tabla Q.
        
        Parámetros:
        -----------
        filepath : str
            Ruta donde guardar el modelo
        **kwargs
            Argumentos adicionales para compatibilidad con Keras
        """
        # Guardar tabla Q por separado
        q_table_path = filepath + QTABLE_SUFFIX
        np.save(q_table_path, self.q_learning_agent.q_table)
        
        # Guardar el modelo wrapper
        super().save(filepath, **kwargs)
    
    def load_weights(self, filepath: str, **kwargs) -> None:
        """
        Carga los pesos del modelo y la tabla Q.
        
        Parámetros:
        -----------
        filepath : str
            Ruta desde donde cargar el modelo
        **kwargs
            Argumentos adicionales para compatibilidad con Keras
        """
        # Cargar tabla Q
        q_table_path = filepath + QTABLE_SUFFIX
        if os.path.exists(q_table_path):
            self.q_learning_agent.q_table = np.load(q_table_path)
        
        # Cargar pesos del modelo wrapper
        super().load_weights(filepath, **kwargs)

def create_q_learning_model(cgm_shape: Tuple[int, ...], other_features_shape: Tuple[int, ...]) -> tf.keras.models.Model:
    """
    Crea un modelo basado en Q-Learning para predicción de dosis de insulina.
    
    Parámetros:
    -----------
    cgm_shape : Tuple[int, ...]
        Forma de los datos CGM (batch_size, time_steps, features)
    other_features_shape : Tuple[int, ...]
        Forma de otras características (batch_size, n_features)
        
    Retorna:
    --------
    tf.keras.models.Model
        Modelo de Q-Learning que implementa la interfaz de Keras
    """
    # Configuración del espacio de estados y acciones
    n_states = 1000  # Estados discretos (ajustar según complejidad del problema)
    n_actions = 20   # Acciones discretas (niveles de dosis de insulina)
    
    # Crear agente Q-Learning
    q_learning_agent = QLearning(
        n_states=n_states,
        n_actions=n_actions,
        learning_rate=QLEARNING_CONFIG['learning_rate'],
        gamma=QLEARNING_CONFIG['gamma'],
        epsilon_start=QLEARNING_CONFIG['epsilon_start'],
        epsilon_end=QLEARNING_CONFIG['epsilon_end'],
        epsilon_decay=QLEARNING_CONFIG['epsilon_decay'],
        use_decay_schedule=QLEARNING_CONFIG['use_decay_schedule'],
        decay_steps=QLEARNING_CONFIG['decay_steps'],
        seed=QLEARNING_CONFIG.get('seed', CONST_DEFAULT_SEED)
    )
    
    # Crear y devolver el modelo wrapper
    return QLearningModel(
        q_learning_agent=q_learning_agent,
        cgm_shape=cgm_shape,
        other_features_shape=other_features_shape
    )