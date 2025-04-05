import os, sys
import numpy as np
import matplotlib.pyplot as plt
import time
from typing import Dict, List, Tuple, Optional, Union, Any, Callable, NamedTuple
import jax
import jax.numpy as jnp
from jax import jit, random
from functools import partial

PROJECT_ROOT = os.path.abspath(os.getcwd())
sys.path.append(PROJECT_ROOT) 

from config.models_config import QLEARNING_CONFIG


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
        seed: int = 42
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
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.use_decay_schedule = use_decay_schedule
        self.decay_steps = decay_steps
        
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
        
        # Compilar funciones puras
        self._update_q_value = jit(self._update_q_value)
    
    def _get_action(self, state_idx: int, rng_key: jnp.ndarray, q_table: jnp.ndarray, 
                  epsilon: float) -> Tuple[int, jnp.ndarray]:
        """
        Selecciona una acción usando la política epsilon-greedy.
        
        Parámetros:
        -----------
        state_idx : int
            Índice del estado actual
        rng_key : jnp.ndarray
            Llave PRNG para generación de números aleatorios
        q_table : jnp.ndarray
            Tabla Q actual
        epsilon : float
            Valor actual de epsilon para exploración
            
        Retorna:
        --------
        Tuple[int, jnp.ndarray]
            Tupla con (acción seleccionada, nueva llave PRNG)
        """
        # Dividir la llave PRNG para permitir múltiples operaciones aleatorias
        key_epsilon, key_action, next_key = random.split(rng_key, 3)
        
        # Determinar si explorar
        explore = random.uniform(key_epsilon) < epsilon
        
        # Si explorar, seleccionar acción aleatoria
        random_action = random.randint(key_action, shape=(), minval=0, maxval=self.n_actions)
        
        # Si explotar, seleccionar mejor acción según la tabla Q
        best_action = jnp.argmax(q_table[state_idx])
        
        # Seleccionar entre exploración y explotación
        action = jnp.where(explore, random_action, best_action)
        
        return int(action), next_key
    
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
        action, next_key = self._get_action(
            state, self.state.rng_key, self.state.q_table, self.state.epsilon
        )
        # Actualizar la llave PRNG
        self.state = self.state._replace(rng_key=next_key)
        
        return action
    
    def _update_q_value(self, q_table: jnp.ndarray, state: int, action: int, 
                        reward: float, next_state: int, done: bool) -> Tuple[jnp.ndarray, float]:
        """
        Actualiza un valor Q específico usando la regla de Q-Learning.
        
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
            Si el episodio ha terminado
            
        Retorna:
        --------
        Tuple[jnp.ndarray, float]
            Tupla con (tabla Q actualizada, TD-error)
        """
        # Valor Q actual
        current_q = q_table[state, action]
        
        # Calcular valor Q objetivo con Q-Learning (off-policy TD control)
        if done:
            # Si es estado terminal, no hay recompensa futura
            target_q = reward
        else:
            # Q-Learning: seleccionar máxima acción en estado siguiente (greedy)
            target_q = reward + self.gamma * jnp.max(q_table[next_state])
        
        # Calcular TD error
        td_error = target_q - current_q
        
        # Actualizar valor Q
        new_q_value = current_q + self.learning_rate * td_error
        
        # Crear tabla Q actualizada (JAX arrays son inmutables)
        new_q_table = q_table.at[state, action].set(new_q_value)
        
        return new_q_table, td_error
    
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
        # Actualizar q_table y obtener TD error
        new_q_table, td_error = self._update_q_value(
            self.state.q_table, state, action, reward, next_state, done
        )
        
        # Actualizar estado
        self.state = self.state._replace(q_table=new_q_table)
        
        return float(td_error)
    
    def _update_epsilon_value(self, epsilon: float, step: Optional[int], use_decay_schedule: str) -> float:
        """
        Calcula el nuevo valor de epsilon según la política de decaimiento.
        
        Parámetros:
        -----------
        epsilon : float
            Valor actual de epsilon
        step : Optional[int]
            Paso actual para decaimiento programado
        use_decay_schedule : str
            Tipo de decaimiento a usar
            
        Retorna:
        --------
        float
            Nuevo valor de epsilon
        """
        if use_decay_schedule == 'linear':
            # Decaimiento lineal
            if step is not None:
                fraction = min(1.0, step / self.decay_steps)
                new_epsilon = self.epsilon_start + fraction * (self.epsilon_end - self.epsilon_start)
            else:
                new_epsilon = max(self.epsilon_end, epsilon * self.epsilon_decay)
        elif use_decay_schedule == 'exponential':
            # Decaimiento exponencial
            new_epsilon = max(self.epsilon_end, epsilon * self.epsilon_decay)
        else:
            # Sin decaimiento
            new_epsilon = epsilon
            
        return new_epsilon
    
    def update_epsilon(self, step: Optional[int] = None) -> None:
        """
        Actualiza el valor de epsilon según la política de decaimiento.
        
        Parámetros:
        -----------
        step : Optional[int], opcional
            Paso actual para decaimiento programado (default: None)
        """
        new_epsilon = self._update_epsilon_value(
            self.state.epsilon, step, self.use_decay_schedule
        )
        
        # Actualizar estado
        self.state = self.state._replace(epsilon=new_epsilon)
    
    def _run_episode(self, env: Any, max_steps: int, render: bool) -> Tuple[float, int, QTableState]:
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
        Tuple[float, int, QTableState]
            Tupla con (recompensa total, número de pasos, estado actualizado del agente)
        """
        state, _ = env.reset()
        episode_reward = 0
        steps = 0
        
        q_table = self.state.q_table
        rng_key = self.state.rng_key
        epsilon = self.state.epsilon
        total_steps = self.state.total_steps
        
        for _ in range(max_steps):
            if render:
                env.render()
            
            # Seleccionar y ejecutar acción
            action, rng_key = self._get_action(state, rng_key, q_table, epsilon)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            
            # Actualizar tabla Q
            q_table, _ = self._update_q_value(q_table, state, action, reward, next_state, done)
            
            # Actualizar contadores y estadísticas
            state = next_state
            episode_reward += reward
            steps += 1
            total_steps += 1
            
            # Actualizar epsilon por paso si corresponde
            if self.use_decay_schedule:
                epsilon = self._update_epsilon_value(epsilon, total_steps, self.use_decay_schedule)
            
            if done:
                break
        
        # Crear estado actualizado
        updated_state = QTableState(
            q_table=q_table,
            rng_key=rng_key,
            epsilon=epsilon,
            total_steps=total_steps
        )
                
        return episode_reward, steps, updated_state
    
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
        history['epsilons'].append(float(self.state.epsilon))
        
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
            episode_reward, steps, updated_state = self._run_episode(env, max_steps, render)
            
            # Actualizar estado del agente
            self.state = updated_state
            
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
                      f"Epsilon: {float(self.state.epsilon):.4f}, "
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
        
        for episode in range(episodes):
            state, _ = env.reset()
            episode_reward = 0
            done = False
            
            while not done:
                if render:
                    env.render()
                
                # Seleccionar la mejor acción (sin exploración)
                action = int(jnp.argmax(self.state.q_table[state]))
                
                # Ejecutar acción
                next_state, reward, terminated, truncated, _ = env.step(action)
                done = terminated or truncated
                
                # Actualizar estado y recompensa
                state = next_state
                episode_reward += reward
            
            rewards.append(episode_reward)
            print(f"Episodio de evaluación {episode+1}/{episodes} - Recompensa: {episode_reward:.2f}")
        
        avg_reward = np.mean(rewards)
        print(f"Recompensa promedio de evaluación: {avg_reward:.2f}")
        
        return avg_reward
    
    def save_qtable(self, filepath: str) -> None:
        """
        Guarda la tabla Q en un archivo.
        
        Parámetros:
        -----------
        filepath : str
            Ruta para guardar la tabla Q
        """
        with open(filepath, 'wb') as f:
            np.save(f, np.array(self.state.q_table))
        print(f"Tabla Q guardada en {filepath}")
    
    def load_qtable(self, filepath: str) -> None:
        """
        Carga la tabla Q desde un archivo.
        
        Parámetros:
        -----------
        filepath : str
            Ruta para cargar la tabla Q
        """
        with open(filepath, 'rb') as f:
            q_table = jnp.array(np.load(f))
        self.state = self.state._replace(q_table=q_table)
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
            """Aplica suavizado con media móvil a los datos"""
            kernel = np.ones(window_size) / window_size
            return np.convolve(np.array(data), kernel, mode='valid')
        
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
            ax.axhline(i, color='black', alpha=0.3)
        for j in range(cols+1):
            ax.axvline(j, color='black', alpha=0.3)
    
    def _draw_policy_arrows(self, ax: plt.Axes, row: int, col: int, state: int, arrows: Dict[int, Tuple[float, float]]) -> None:
        """
        Dibuja flechas que representan la política para un estado.
        
        Parámetros:
        -----------
        ax : plt.Axes
            Eje para la visualización
        row : int
            Fila del estado en la grilla
        col : int
            Columna del estado en la grilla
        state : int
            Índice del estado
        arrows : Dict[int, Tuple[float, float]]
            Diccionario que mapea acciones a vectores de dirección
        """
        q_values = np.array(self.state.q_table[state])
        
        if np.all(q_values == 0):
            # Si todas las acciones son iguales, no hay preferencia clara
            for action in range(self.n_actions):
                dx, dy = arrows[action]
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
        Visualiza la política aprendida para entornos de tipo grilla.
        
        Parámetros:
        -----------
        env : Any
            Entorno, preferentemente de tipo grilla
        grid_size : Optional[Tuple[int, int]], opcional
            Tamaño de la grilla (filas, columnas) (default: None)
        """
        if grid_size is None:
            try:
                grid_size = (env.nrow, env.ncol)  # Para FrozenLake
            except AttributeError:
                print("El entorno no parece ser una grilla o no se proporcionó grid_size")
                return
        
        rows, cols = grid_size
        
        # Crear un mapa de flechas para visualizar la política
        _, ax = plt.subplots(figsize=(10, 10))
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
        Visualiza la función de valor aprendida para entornos de tipo grilla.
        
        Parámetros:
        -----------
        env : Any
            Entorno, preferentemente de tipo grilla
        grid_size : Optional[Tuple[int, int]], opcional
            Tamaño de la grilla (filas, columnas) (default: None)
        """
        if grid_size is None:
            try:
                grid_size = (env.nrow, env.ncol)  # Para FrozenLake
            except AttributeError:
                print("El entorno no parece ser una grilla o no se proporcionó grid_size")
                return
        
        rows, cols = grid_size
        
        # Calcular la función de valor: V(s) = max_a Q(s,a)
        q_table_np = np.array(self.state.q_table)
        value_function = np.max(q_table_np, axis=1).reshape(rows, cols)
        
        _, ax = plt.subplots(figsize=(10, 8))
        im = ax.imshow(value_function, cmap='viridis')
        
        # Añadir barra de color
        cbar = plt.colorbar(im)
        cbar.set_label('Valor Esperado')
        
        # Añadir etiquetas
        for i in range(rows):
            for j in range(cols):
                state = i * cols + j
                value = float(np.max(q_table_np[state]))
                ax.text(j, i, f"{value:.2f}", ha="center", va="center", 
                        color="w" if value < (value_function.max() + value_function.min())/2 else "black")
        
        plt.title('Función de Valor (V)')
        plt.tight_layout()
        plt.show()
    
    def compare_with_optimal(self, optimal_policy: np.ndarray) -> float:
        """
        Compara la política aprendida con una política óptima conocida.
        
        Parámetros:
        -----------
        optimal_policy : np.ndarray
            Política óptima conocida como array de acciones para cada estado
            
        Retorna:
        --------
        float
            Porcentaje de acciones que coinciden con la política óptima
        """
        # Obtener la política actual
        current_policy = np.array([np.argmax(self.state.q_table[s]) for s in range(self.n_states)])
        
        # Calcular coincidencias
        matches = np.sum(current_policy == optimal_policy)
        
        # Calcular porcentaje de coincidencia
        match_percentage = matches / self.n_states * 100
        
        print(f"Coincidencia con política óptima: {match_percentage:.2f}%")
        
        return match_percentage
    
    def get_q_table(self) -> np.ndarray:
        """
        Devuelve la tabla Q actual como un array NumPy.
        
        Retorna:
        --------
        np.ndarray
            Tabla Q actual
        """
        return np.array(self.state.q_table)

class QLearningWrapper:
    """
    Wrapper para hacer que el agente Q-Learning sea compatible con la interfaz de modelos de aprendizaje profundo.
    """
    
    def __init__(
        self, 
        q_agent: QLearning, 
        cgm_shape: Tuple[int, ...], 
        other_features_shape: Tuple[int, ...],
    ) -> None:
        """
        Inicializa el wrapper para Q-Learning.
        
        Parámetros:
        -----------
        q_agent : QLearning
            Agente Q-Learning a utilizar
        cgm_shape : Tuple[int, ...]
            Forma de entrada para datos CGM
        other_features_shape : Tuple[int, ...]
            Forma de entrada para otras características
        """
        self.q_agent = q_agent
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
        Realiza predicciones con el modelo Q-Learning.
        
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
            # Discretizar estado
            state = self._discretize_state(cgm_data[i], other_features[i])
            
            # Obtener acción según la tabla Q (greedy, sin exploración)
            action = int(jnp.argmax(self.q_agent.state.q_table[state]))
            
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
        # Simplificar CGM usando valores clave (último valor, pendiente, promedio)
        cgm_flat = cgm_data.flatten()
        cgm_mean = np.mean(cgm_flat)
        cgm_last = cgm_flat[-1]
        cgm_slope = cgm_flat[-1] - cgm_flat[0]
        
        # Discretizar características clave
        cgm_mean_bin = min(int(cgm_mean * self.cgm_bins), self.cgm_bins - 1)
        cgm_last_bin = min(int(cgm_last * self.cgm_bins), self.cgm_bins - 1)
        cgm_slope_bin = min(int((cgm_slope + 1) * self.cgm_bins / 2), self.cgm_bins - 1)
        
        # Usar solo las primeras características más relevantes de other_features
        relevant_features = other_features[:min(3, len(other_features))]
        other_bins = [min(int(f * self.other_bins), self.other_bins - 1) for f in relevant_features]
        
        # Calcular índice de estado combinado
        state = cgm_mean_bin
        state = state * self.cgm_bins + cgm_last_bin
        state = state * self.cgm_bins + cgm_slope_bin
        
        for b in other_bins:
            state = state * self.other_bins + b
            
        return min(state, self.q_agent.n_states - 1)
    
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
        # Mapear desde [0, n_actions-1] a [0, 15] unidades de insulina
        return action * 15.0 / (self.q_agent.n_actions - 1)
    
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
        Entrena el modelo Q-Learning en los datos proporcionados.
        
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
        
        # Crear entorno personalizado para Q-Learning
        env = self._create_training_environment(cgm_data, other_features, y)
        
        if verbose > 0:
            print("Entrenando modelo Q-Learning...")
        
        # Entrenar agente con el número de episodios igual a epochs
        training_history = self.q_agent.train(
            env=env,
            episodes=epochs,
            max_steps=batch_size,
            render=False,
            log_interval=max(1, epochs // 10) if verbose > 0 else epochs + 1
        )
        
        # Actualizar historial con métricas del entrenamiento
        self.history = {
            'episode_rewards': training_history['episode_rewards'],
            'episode_lengths': training_history['episode_lengths'],
            'epsilons': training_history['epsilons'],
            'avg_rewards': training_history['avg_rewards']
        }
        
        # Calcular pérdida en los datos de entrenamiento
        train_preds = self.predict(x)
        train_loss = float(np.mean((train_preds.flatten() - y) ** 2))
        self.history['loss'] = [train_loss]
        
        # Evaluar en datos de validación si se proporcionan
        if validation_data:
            val_x, val_y = validation_data
            val_preds = self.predict(val_x)
            val_loss = float(np.mean((val_preds.flatten() - val_y) ** 2))
            self.history['val_loss'] = [val_loss]
            
            if verbose > 0:
                print(f"Pérdida de validación: {val_loss:.4f}")
        
        if verbose > 0:
            print(f"Entrenamiento completado. Pérdida final: {train_loss:.4f}")
        
        return self.history
    
    def _create_training_environment(
        self, 
        cgm_data: np.ndarray, 
        other_features: np.ndarray, 
        targets: np.ndarray
    ) -> Any:
        """
        Crea un entorno de entrenamiento compatible con el agente Q-Learning.
        
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
            Entorno simulado para Q-Learning
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
                
                # Para compatibilidad con algoritmos RL
                self.observation_space = SimpleNamespace(
                    shape=(1,),
                    low=0,
                    high=model_wrapper.q_agent.n_states - 1
                )
                
                self.action_space = SimpleNamespace(
                    n=model_wrapper.q_agent.n_actions,
                    sample=self._sample_action
                )
            
            def _sample_action(self):
                """Muestrea una acción aleatoria del espacio discreto."""
                return self.rng.integers(0, self.model.q_agent.n_actions)
            
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
                self.current_idx = (self.current_idx + 1) % self.max_idx
                
                # Obtener próximo estado
                next_state = self.model._discretize_state(
                    self.cgm[self.current_idx],
                    self.features[self.current_idx]
                )
                
                # Terminal después de cada paso
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
        self.q_agent.save_qtable(filepath + "_qtable.npy")
        
        import pickle
        wrapper_data = {
            'cgm_shape': self.cgm_shape,
            'other_features_shape': self.other_features_shape,
            'cgm_bins': self.cgm_bins,
            'other_bins': self.other_bins,
            'n_states': self.q_agent.n_states,
            'n_actions': self.q_agent.n_actions
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
        self.q_agent.load_qtable(filepath + "_qtable.npy")
        
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
            'n_states': self.q_agent.n_states,
            'n_actions': self.q_agent.n_actions,
            'gamma': self.q_agent.gamma,
            'epsilon': float(self.q_agent.state.epsilon),
            'learning_rate': self.q_agent.learning_rate
        }


def create_q_learning_model(cgm_shape: Tuple[int, ...], other_features_shape: Tuple[int, ...]) -> QLearningWrapper:
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
    QLearningWrapper
        Wrapper de Q-Learning que implementa la interfaz compatible con modelos de aprendizaje profundo
    """
    # Configurar el tamaño del espacio de estados y acciones
    # Esto es una simplificación - en un caso real habría que definirlo según los datos
    n_states = 1000  # Estado discretizado (más estados para mayor precisión)
    n_actions = 20   # Por ejemplo: 20 niveles discretos de dosis (0 a 15 unidades)
    
    # Crear agente Q-Learning
    q_agent = QLearning(
        n_states=n_states,
        n_actions=n_actions,
        learning_rate=QLEARNING_CONFIG['learning_rate'],
        gamma=QLEARNING_CONFIG['gamma'],
        epsilon_start=QLEARNING_CONFIG['epsilon_start'],
        epsilon_end=QLEARNING_CONFIG['epsilon_end'],
        epsilon_decay=QLEARNING_CONFIG['epsilon_decay'],
        use_decay_schedule=QLEARNING_CONFIG['use_decay_schedule'],
        decay_steps=QLEARNING_CONFIG['decay_steps'],
        seed=QLEARNING_CONFIG.get('seed', 42)
    )
    
    # Crear y devolver wrapper
    return QLearningWrapper(q_agent, cgm_shape, other_features_shape)