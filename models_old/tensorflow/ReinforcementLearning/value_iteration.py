import os, sys
import numpy as np
import matplotlib.pyplot as plt
import time
import pickle
from typing import Dict, List, Tuple, Optional, Union, Any
import tensorflow as tf
from keras._tf_keras.keras.models import Model
from keras._tf_keras.keras.layers import Input, Dense, Concatenate, GlobalAveragePooling1D
from keras._tf_keras.keras.saving import register_keras_serializable

PROJECT_ROOT = os.path.abspath(os.getcwd())
sys.path.append(PROJECT_ROOT) 

from config.models_config_old import VALUE_ITERATION_CONFIG
from constants.constants import CONST_DEFAULT_SEED, CONST_DEFAULT_EPOCHS, CONST_DEFAULT_BATCH_SIZE

# Constantes para cadenas repetidas
CONST_ITERACION = "Iteración"
CONST_RECOMPENSA = "Recompensa"
CONST_VALOR = "Valor"
CONST_TIEMPO = "Tiempo (segundos)"
CONST_POLITICA = "Política"
CONST_PROB_TRANSICION = 1.0
FIGURAS_DIR = "figures/reinforcement_learning/value_iteration"

class ValueIteration:
    """
    Implementación del algoritmo de Iteración de Valor (Value Iteration).
    
    La Iteración de Valor es un método de programación dinámica que encuentra la política
    óptima calculando directamente la función de valor óptima utilizando la ecuación de
    optimalidad de Bellman, sin mantener explícitamente una política durante el proceso.
    """
    
    def __init__(
        self,
        n_states: int,
        n_actions: int,
        gamma: float = VALUE_ITERATION_CONFIG['gamma'],
        theta: float = VALUE_ITERATION_CONFIG['theta'],
        max_iterations: int = VALUE_ITERATION_CONFIG['max_iterations']
    ) -> None:
        """
        Inicializa el agente de Iteración de Valor.
        
        Parámetros:
        -----------
        n_states : int
            Número de estados en el entorno
        n_actions : int
            Número de acciones en el entorno
        gamma : float, opcional
            Factor de descuento (default: VALUE_ITERATION_CONFIG['gamma'])
        theta : float, opcional
            Umbral para convergencia (default: VALUE_ITERATION_CONFIG['theta'])
        max_iterations : int, opcional
            Número máximo de iteraciones (default: VALUE_ITERATION_CONFIG['max_iterations'])
        """
        self.n_states = n_states
        self.n_actions = n_actions
        self.gamma = gamma
        self.theta = theta
        self.max_iterations = max_iterations
        
        # Inicializar función de valor
        self.V = np.zeros(n_states)
        
        # La política se deriva de la función de valor (no se mantiene explícitamente)
        self.policy = np.zeros((n_states, n_actions))
        
        # Para métricas
        self.value_changes = []
        self.iteration_times = []
    
    def _calculate_action_values(self, state: int, transitions: Dict[int, Dict[int, List]]) -> np.ndarray:
        """
        Calcula los valores Q para todas las acciones en un estado.
        
        Parámetros:
        -----------
        state : int
            Estado para calcular valores de acción
        transitions : Dict[int, Dict[int, List]]
            Diccionario con las transiciones del entorno
            
        Retorna:
        --------
        np.ndarray
            Valores Q para todas las acciones en el estado dado
        """
        action_values = np.zeros(self.n_actions)
        
        for a in range(self.n_actions):
            # Verificar si esta acción está definida para este estado
            if state in transitions and a in transitions[state]:
                # Iterar sobre todas las posibles transiciones para esta acción
                for prob, next_state, reward, done in transitions[state][a]:
                    # Actualizar valor de acción utilizando la ecuación de Bellman (optimalidad)
                    action_values[a] += prob * (reward + self.gamma * self.V[next_state] * (not done))
        
        return action_values
    
    def value_update(self, env: Any) -> float:
        """
        Realiza una iteración de actualización de la función de valor.
        
        Parámetros:
        -----------
        env : Any
            Entorno con dinámicas de transición
            
        Retorna:
        --------
        float
            Delta máximo (cambio máximo en la función de valor)
        """
        delta = 0
        
        for s in range(self.n_states):
            # Solo procesar estados que existen en el entorno
            if s in env.P:
                # Guardar valor antiguo
                v_old = self.V[s]
                
                # Calcular valores para cada acción y seleccionar el máximo (ecuación de optimalidad de Bellman)
                action_values = self._calculate_action_values(s, env.P)
                self.V[s] = np.max(action_values)
                
                # Actualizar delta (cambio máximo)
                delta = max(delta, abs(v_old - self.V[s]))
        
        return delta
    
    def extract_policy(self, env: Any) -> np.ndarray:
        """
        Extrae la política óptima a partir de la función de valor.
        
        Parámetros:
        -----------
        env : Any
            Entorno con dinámicas de transición
            
        Retorna:
        --------
        np.ndarray
            Política óptima (determinística)
        """
        policy = np.zeros((self.n_states, self.n_actions))
        
        for s in range(self.n_states):
            # Solo procesar estados que existen en el entorno
            if s in env.P:
                # Calcular valores Q para cada acción
                action_values = self._calculate_action_values(s, env.P)
                
                # Manejar el caso de valores iguales de manera determinista
                best_actions = np.nonzero(action_values == np.max(action_values))[0]
                if len(best_actions) > 1:
                    # Seleccionar de manera determinista basada en el estado
                    selected_action = best_actions[hash(str(s)) % len(best_actions)]
                else:
                    selected_action = best_actions[0]
                
                # Asignar probabilidad 1.0 a la mejor acción (política determinística)
                policy[s, selected_action] = 1.0
        
        return policy
    
    def train(self, env: Any) -> Dict[str, List]:
        """
        Entrena al agente usando iteración de valor.
        
        Parámetros:
        -----------
        env : Any
            Entorno con dinámicas de transición
            
        Retorna:
        --------
        Dict[str, List]
            Diccionario con historial de entrenamiento
        """
        print("Iniciando iteración de valor...")
        
        iterations = 0
        start_time = time.time()
        self.value_changes = []
        self.iteration_times = []
        
        for i in range(self.max_iterations):
            iteration_start = time.time()
            
            # Actualizar la función de valor
            delta = self.value_update(env)
            
            # Guardar cambio de valor y tiempo
            self.value_changes.append(delta)
            iteration_time = time.time() - iteration_start
            self.iteration_times.append(iteration_time)
            
            iterations += 1
            
            # Log de progreso
            print(f"Iteración {i+1}/{self.max_iterations}, Delta={delta:.6f}, Tiempo={iteration_time:.2f}s")
            
            # Verificar convergencia
            if delta < self.theta:
                print(f"Convergido después de {i+1} iteraciones (delta={delta:.6f})")
                break
            
            # Si el entrenamiento está tomando demasiado tiempo, detenerlo
            if time.time() - start_time > 300:  # Limitar a 5 minutos
                print("Tiempo de entrenamiento excedido. Deteniendo entrenamiento.")
                break
        
        # Extraer política óptima
        self.policy = self.extract_policy(env)
        
        total_time = time.time() - start_time
        print(f"Iteración de valor completada en {iterations} iteraciones, {total_time:.2f} segundos")
        
        history = {
            'iterations': iterations,
            'value_changes': self.value_changes,
            'iteration_times': self.iteration_times,
            'total_time': total_time
        }
        
        return history
    
    def get_action(self, state: int) -> int:
        """
        Devuelve la mejor acción para un estado dado según la política actual.
        
        Parámetros:
        -----------
        state : int
            Estado actual
            
        Retorna:
        --------
        int
            Mejor acción
        """
        return np.argmax(self.policy[state])
    
    def get_value(self, state: int) -> float:
        """
        Devuelve el valor de un estado según la función de valor actual.
        
        Parámetros:
        -----------
        state : int
            Estado para obtener su valor
            
        Retorna:
        --------
        float
            Valor del estado
        """
        return self.V[state]
    
    def evaluate(self, env: Any, max_steps: int = 100, episodes: int = 10) -> float:
        """
        Evalúa la política actual en el entorno.
        
        Parámetros:
        -----------
        env : Any
            Entorno para evaluar
        max_steps : int, opcional
            Pasos máximos por episodio (default: 100)
        episodes : int, opcional
            Número de episodios para evaluar (default: 10)
            
        Retorna:
        --------
        float
            Recompensa promedio en los episodios
        """
        total_rewards = []
        episode_lengths = []
        
        for _ in range(episodes):
            state, _ = env.reset()
            total_reward = 0
            steps = 0
            
            for _ in range(max_steps):
                # Seleccionar acción según la política actual
                action = self.get_action(state)
                
                # Ejecutar acción en el entorno
                next_state, reward, terminated, truncated, _ = env.step(action)
                done = terminated or truncated
                
                # Actualizar estado y acumular recompensa
                state = next_state
                total_reward += reward
                steps += 1
                
                # Terminar si el episodio ha concluido
                if done:
                    break
            
            total_rewards.append(total_reward)
            episode_lengths.append(steps)
        
        avg_reward = np.mean(total_rewards)
        avg_length = np.mean(episode_lengths)
        print(f"Evaluación: recompensa media en {episodes} episodios = {avg_reward:.2f}, " +
              f"longitud media = {avg_length:.2f}")
        
        return avg_reward
    
    def save(self, filepath: str) -> None:
        """
        Guarda la política y función de valor en un archivo.
        
        Parámetros:
        -----------
        filepath : str
            Ruta donde guardar el modelo
        """
        data = {
            'policy': self.policy,
            'V': self.V,
            'n_states': self.n_states,
            'n_actions': self.n_actions,
            'gamma': self.gamma
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(data, f)
        
        print(f"Modelo guardado en {filepath}")
    
    def load(self, filepath: str) -> None:
        """
        Carga la política y función de valor desde un archivo.
        
        Parámetros:
        -----------
        filepath : str
            Ruta desde donde cargar el modelo
        """
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
        
        self.policy = data['policy']
        self.V = data['V']
        self.n_states = data['n_states']
        self.n_actions = data['n_actions']
        self.gamma = data['gamma']
        
        print(f"Modelo cargado desde {filepath}")
    
    def _get_grid_position(self, env: Any, state: int) -> Tuple[int, int]:
        """
        Obtiene la posición en la cuadrícula para un estado.
        
        Parámetros:
        -----------
        env : Any
            Entorno con estructura de cuadrícula
        state : int
            Estado a convertir en posición
            
        Retorna:
        --------
        Tuple[int, int]
            Posición (i, j) en la cuadrícula
        """
        if hasattr(env, 'state_mapping'):
            return env.state_mapping[state]
        else:
            # Asumir cuadrícula nxn
            n = int(np.sqrt(self.n_states))
            return (state // n, state % n)
    
    def _is_terminal_state(self, env: Any, state: int) -> bool:
        """
        Determina si un estado es terminal.
        
        Parámetros:
        -----------
        env : Any
            Entorno con transiciones
        state : int
            Estado a comprobar
            
        Retorna:
        --------
        bool
            True si el estado es terminal, False en caso contrario
        """
        for a in range(self.n_actions):
            if state in env.P and a in env.P[state]:
                for _, _, _, done in env.P[state][a]:
                    if done:
                        return True
        return False
    
    def _setup_grid(self, ax: plt.Axes, grid_shape: Tuple[int, int]) -> None:
        """
        Configura la cuadrícula para visualización.
        
        Parámetros:
        -----------
        ax : plt.Axes
            Ejes para dibujar
        grid_shape : Tuple[int, int]
            Forma de la cuadrícula (filas, columnas)
        """
        # Configurar límites
        ax.set_xlim([0, grid_shape[1]])
        ax.set_ylim([0, grid_shape[0]])
        
        # Dibujar líneas de cuadrícula
        for i in range(grid_shape[1] + 1):
            ax.axvline(i, color='black', linewidth=0.5)
        for j in range(grid_shape[0] + 1):
            ax.axhline(j, color='black', linewidth=0.5)
            
    def _draw_action_arrows(self, ax: plt.Axes, env: Any, grid_shape: Tuple[int, int]) -> None:
        """
        Dibuja flechas para las acciones en cada estado no terminal.
        
        Parámetros:
        -----------
        ax : plt.Axes
            Ejes para dibujar
        env : Any
            Entorno con estructura de cuadrícula
        grid_shape : Tuple[int, int]
            Forma de la cuadrícula (filas, columnas)
        """
        # Definir direcciones de flechas
        directions = {
            0: (0, -0.4),  # Arriba
            1: (0, 0.4),   # Abajo
            2: (-0.4, 0),  # Izquierda
            3: (0.4, 0)    # Derecha
        }
        
        for s in range(self.n_states):
            if s in env.P and not self._is_terminal_state(env, s):
                i, j = self._get_grid_position(env, s)
                action = self.get_action(s)
                
                if action in directions:
                    dx, dy = directions[action]
                    ax.arrow(j + 0.5, grid_shape[0] - i - 0.5, dx, dy, 
                             head_width=0.1, head_length=0.1, fc='blue', ec='blue')
    
    def _show_state_values(self, ax: plt.Axes, env: Any, grid_shape: Tuple[int, int]) -> None:
        """
        Muestra los valores de cada estado en la cuadrícula.
        
        Parámetros:
        -----------
        ax : plt.Axes
            Ejes para dibujar
        env : Any
            Entorno con estructura de cuadrícula
        grid_shape : Tuple[int, int]
            Forma de la cuadrícula (filas, columnas)
        """
        for s in range(self.n_states):
            if s in env.P:
                i, j = self._get_grid_position(env, s)
                ax.text(j + 0.5, grid_shape[0] - i - 0.5, f"{self.V[s]:.2f}", 
                       ha='center', va='center', color='red', fontsize=9)
    
    def visualize_policy(self, env: Any, title: str = "Política Óptima") -> None:
        """
        Visualiza la política para entornos de tipo cuadrícula.
        
        Parámetros:
        -----------
        env : Any
            Entorno con estructura de cuadrícula
        title : str, opcional
            Título para la visualización (default: "Política Óptima")
        """
        if not hasattr(env, 'shape'):
            print("El entorno no tiene una forma definida para visualización.")
            return
        
        grid_shape = env.shape
        _, ax = plt.subplots(figsize=(8, 8))
        
        # Configurar y dibujar cuadrícula
        self._setup_grid(ax, grid_shape)
        
        # Dibujar flechas para las acciones
        self._draw_action_arrows(ax, env, grid_shape)
        
        # Mostrar valores de estados
        self._show_state_values(ax, env, grid_shape)
        
        ax.set_title(title)
        plt.tight_layout()
        
        # Guardar figura
        os.makedirs(FIGURAS_DIR, exist_ok=True)
        plt.savefig(os.path.join(FIGURAS_DIR, f"{title.lower().replace(' ', '_')}.png"), dpi=300)
        
        plt.show()
    
    def visualize_value_function(self, env: Any, title: str = "Función de Valor") -> None:
        """
        Visualiza la función de valor para entornos de tipo cuadrícula.
        
        Parámetros:
        -----------
        env : Any
            Entorno con estructura de cuadrícula
        title : str, opcional
            Título para la visualización (default: "Función de Valor")
        """
        if not hasattr(env, 'shape'):
            print("El entorno no tiene una forma definida para visualización.")
            return
        
        grid_shape = env.shape
        value_grid = np.zeros(grid_shape)
        
        # Llenar la cuadrícula con valores
        for s in range(self.n_states):
            if s in env.P:
                i, j = self._get_grid_position(env, s)
                value_grid[i, j] = self.V[s]
        
        # Crear figura
        _, ax = plt.subplots(figsize=(10, 8))
        im = ax.imshow(value_grid, cmap='viridis')
        plt.colorbar(im, ax=ax, label='Valor')
        
        # Añadir valores como texto
        for i in range(grid_shape[0]):
            for j in range(grid_shape[1]):
                s = i * grid_shape[1] + j
                if s in env.P:
                    ax.text(j, i, f"{value_grid[i, j]:.2f}", 
                           ha='center', va='center', color='white', fontsize=9)
        
        ax.set_title(title)
        
        # Guardar figura
        os.makedirs(FIGURAS_DIR, exist_ok=True)
        plt.savefig(os.path.join(FIGURAS_DIR, f"{title.lower().replace(' ', '_')}.png"), dpi=300)
        
        plt.show()
    
    def visualize_training(self, history: Dict[str, List]) -> None:
        """
        Visualiza el progreso de entrenamiento.
        
        Parámetros:
        -----------
        history : Dict[str, List]
            Historial de entrenamiento con métricas
        """
        # iterations = history.get('iterations', 0)
        
        # Crear figura con 2 subplots
        _, axs = plt.subplots(1, 2, figsize=(15, 6))
        
        # Gráfico de cambios de valor
        if 'value_changes' in history and history['value_changes']:
            x = range(1, len(history['value_changes']) + 1)
            axs[0].plot(x, history['value_changes'], marker='o')
            axs[0].set_title('Cambios en la Función de Valor')
            axs[0].set_xlabel(CONST_ITERACION)
            axs[0].set_ylabel('Delta')
            axs[0].set_yscale('log')
            axs[0].grid(True)
        
        # Gráfico de tiempos de iteración
        if 'iteration_times' in history and history['iteration_times']:
            x = range(1, len(history['iteration_times']) + 1)
            axs[1].plot(x, history['iteration_times'], marker='o')
            axs[1].set_title('Tiempos de Iteración')
            axs[1].set_xlabel(CONST_ITERACION)
            axs[1].set_ylabel(CONST_TIEMPO)
            axs[1].grid(True)
        
        plt.tight_layout()
        
        # Guardar figura
        os.makedirs(FIGURAS_DIR, exist_ok=True)
        plt.savefig(os.path.join(FIGURAS_DIR, "entrenamiento_resumen.png"), dpi=300)
        
        plt.show()
        
    def _initialize_transition_matrices(self, env: Any) -> Tuple[list, dict, np.ndarray, np.ndarray, np.ndarray]:
        """
        Inicializa las matrices de transición para iteración de valor vectorizada.
        
        Parámetros:
        -----------
        env : Any
            Entorno con dinámicas de transición
            
        Retorna:
        --------
        Tuple[list, dict, np.ndarray, np.ndarray, np.ndarray]
            Estados, mapeo de índices, matrices de probabilidad, recompensa y máscara terminal
        """
        # Compilar el modelo de transición para todos los estados existentes
        states = sorted(list(env.P.keys()))
        n_existing_states = len(states)
        
        # Mapeo de índices de estado
        state_to_idx = {s: i for i, s in enumerate(states)}
        
        # Matrices de transición
        P_matrix = np.zeros((n_existing_states, self.n_actions, n_existing_states))
        R_matrix = np.zeros((n_existing_states, self.n_actions, n_existing_states))
        terminal_mask = np.zeros((n_existing_states, self.n_actions), dtype=bool)
        
        # Llenar matrices
        for s_idx, s in enumerate(states):
            for a in range(self.n_actions):
                if a in env.P[s]:
                    for prob, next_s, reward, done in env.P[s][a]:
                        next_s_idx = state_to_idx.get(next_s, 0)  # Usar 0 como fallback
                        P_matrix[s_idx, a, next_s_idx] = prob
                        R_matrix[s_idx, a, next_s_idx] = reward
                        if done:
                            terminal_mask[s_idx, a] = True
        
        return states, state_to_idx, P_matrix, R_matrix, terminal_mask
    
    def _calculate_q_values(self, V: np.ndarray, P_matrix: np.ndarray, R_matrix: np.ndarray, 
                          terminal_mask: np.ndarray, n_existing_states: int) -> np.ndarray:
        """
        Calcula los valores Q para todos los pares estado-acción.
        
        Parámetros:
        -----------
        V : np.ndarray
            Vector de valores de estado actual
        P_matrix : np.ndarray
            Matriz de probabilidades de transición
        R_matrix : np.ndarray
            Matriz de recompensas
        terminal_mask : np.ndarray
            Máscara de estados terminales
        n_existing_states : int
            Número de estados existentes
            
        Retorna:
        --------
        np.ndarray
            Matriz de valores Q
        """
        Q = np.zeros((n_existing_states, self.n_actions))
        for s_idx in range(n_existing_states):
            for a in range(self.n_actions):
                if not terminal_mask[s_idx, a]:
                    # Sumar sobre todos los posibles estados siguientes
                    Q[s_idx, a] = np.sum(P_matrix[s_idx, a] * (R_matrix[s_idx, a] + self.gamma * V))
                else:
                    # Para estados terminales, solo considerar la recompensa inmediata
                    Q[s_idx, a] = np.sum(P_matrix[s_idx, a] * R_matrix[s_idx, a])
        return Q
    
    def _update_full_value_function(self, V: np.ndarray, states: list) -> None:
        """
        Actualiza la función de valor completa con los valores calculados.
        
        Parámetros:
        -----------
        V : np.ndarray
            Vector de valores de estado calculados
        states : list
            Lista de estados existentes
        """
        full_V = np.zeros(self.n_states)
        for s_idx, s in enumerate(states):
            full_V[s] = V[s_idx]
        self.V = full_V
    
    def vectorized_value_iteration(self, env: Any) -> Dict[str, List]:
        """
        Implementación vectorizada de iteración de valor para mayor eficiencia.
        
        Parámetros:
        -----------
        env : Any
            Entorno con dinámicas de transición
            
        Retorna:
        --------
        Dict[str, List]
            Historial de entrenamiento
        """
        print("Iniciando iteración de valor vectorizada...")
        
        # Inicializar matrices de transición
        states, _, P_matrix, R_matrix, terminal_mask = self._initialize_transition_matrices(env)
        n_existing_states = len(states)
        
        # Vectores de valor
        V = np.zeros(n_existing_states)
        
        # Historial de métricas
        value_changes = []
        iteration_times = []
        
        start_time = time.time()
        i = 0
        
        # Iteración de valor vectorizada
        for i in range(self.max_iterations):
            iteration_start = time.time()
            
            # Calcular valores Q
            Q = self._calculate_q_values(V, P_matrix, R_matrix, terminal_mask, n_existing_states)
            
            # Actualizar valores de estado (ecuación de optimalidad de Bellman)
            V_new = np.max(Q, axis=1)
            
            # Calcular delta (convergencia)
            delta = np.max(np.abs(V - V_new))
            value_changes.append(delta)
            
            # Actualizar función de valor
            V = V_new
            
            # Registrar tiempo
            iteration_time = time.time() - iteration_start
            iteration_times.append(iteration_time)
            
            # Log de progreso
            print(f"Iteración {i+1}/{self.max_iterations}, Delta={delta:.6f}, Tiempo={iteration_time:.2f}s")
            
            # Verificar convergencia
            if delta < self.theta:
                print(f"Convergido después de {i+1} iteraciones (delta={delta:.6f})")
                break
        
        # Actualizar función de valor completa
        self._update_full_value_function(V, states)
        
        # Extraer política óptima
        self.policy = self.extract_policy(env)
        
        total_time = time.time() - start_time
        print(f"Iteración de valor vectorizada completada en {i+1} iteraciones, {total_time:.2f} segundos")
        
        return {
            'iterations': i + 1,
            'value_changes': value_changes,
            'iteration_times': iteration_times,
            'total_time': total_time
        }


@register_keras_serializable()
class ValueIterationModel(Model):
    """
    Modelo wrapper para Iteración de Valor compatible con la interfaz de Keras.
    """
    
    def __init__(
        self, 
        value_iteration_agent: ValueIteration,
        cgm_shape: Tuple[int, ...],
        other_features_shape: Tuple[int, ...],
        discretizer: Optional[Any] = None
    ) -> None:
        """
        Inicializa el modelo wrapper.
        
        Parámetros:
        -----------
        value_iteration_agent : ValueIteration
            Agente de Iteración de Valor
        cgm_shape : Tuple[int, ...]
            Forma de los datos CGM
        other_features_shape : Tuple[int, ...]
            Forma de otras características
        discretizer : Optional[Any], opcional
            Discretizador de estados (default: None)
        """
        super().__init__()
        self.value_iteration_agent = value_iteration_agent
        
        # Formas de entrada
        self.cgm_shape = cgm_shape
        self.other_features_shape = other_features_shape
        
        # Discretizador de estados (puede ser None)
        self.discretizer = discretizer
        
        # Crear codificadores para traducir datos continuos a estados discretos
        self.cgm_encoder = Dense(64, activation='relu', name='cgm_encoder')
        self.other_encoder = Dense(32, activation='relu', name='other_encoder')
        self.combined_encoder = Dense(self.value_iteration_agent.n_states, 
                                      activation='softmax', name='state_encoder')
        
        # Decodificador para traducir acciones discretas a valores continuos
        self.action_decoder = Dense(1, name='action_decoder')
        
        # Crear entorno sintético para entrenamiento
        self.env = None
        
        # Métricas para seguimiento
        self.training_metrics = {}
    
    def _encode_states(self, cgm_data: tf.Tensor, other_features: tf.Tensor) -> tf.Tensor:
        """
        Codifica los datos de entrada en distribuciones sobre estados discretos.
        
        Parámetros:
        -----------
        cgm_data : tf.Tensor
            Datos CGM
        other_features : tf.Tensor
            Otras características
            
        Retorna:
        --------
        tf.Tensor
            Distribución sobre estados discretos
        """
        # Procesar CGM data a través del encoder
        cgm_encoded = self.cgm_encoder(cgm_data)
        
        # Aplanar el tensor de salida de CGM para que sea 2D (batch_size, features)
        # Esto es necesario porque la salida del encoder CGM tiene forma (batch, timesteps, features)
        cgm_encoded_flat = tf.keras.layers.Flatten()(cgm_encoded)
        
        # Procesar other features a través de su encoder
        other_encoded = self.other_encoder(other_features)
        
        # Concatenar las representaciones aplanadas
        combined = Concatenate()([cgm_encoded_flat, other_encoded])
        
        # Obtener distribución sobre estados discretos
        state_distribution = self.combined_encoder(combined)
        
        return state_distribution
    
    def call(self, inputs: List[tf.Tensor], training: bool = False) -> tf.Tensor:
        """
        Realiza la predicción con el modelo.
        
        Parámetros:
        -----------
        inputs : List[tf.Tensor]
            Lista de tensores de entrada [cgm_data, other_features]
        training : bool, opcional
            Indica si está en modo entrenamiento (default: False)
            
        Retorna:
        --------
        tf.Tensor
            Predicción de dosis de insulina
        """
        # Extraer entradas
        cgm_data, other_features = inputs
        
        # Codificar estados
        states = self._encode_states(cgm_data, other_features)
        
        # Obtener la matriz de política
        policy_matrix = tf.convert_to_tensor(self.value_iteration_agent.policy, dtype=tf.float32)
        
        # Extraer las acciones para los estados calculados
        actions = tf.matmul(states, policy_matrix)
        
        # Mapear acción discreta a valor continuo de dosis
        action_values = self.action_decoder(actions)
        
        # Asegurar que la salida tenga la forma correcta (batch_size,)
        return tf.reshape(action_values, [-1])
    
    def _process_dataset_input(self, x: tf.data.Dataset) -> None:
        """
        Procesa un dataset de TensorFlow para la entrada.
        
        Parámetros:
        -----------
        x : tf.data.Dataset
            Dataset con datos de entrada
        """
        # Iterar un lote del dataset para obtener formas representativas
        for (cgm_batch, other_batch), y_batch in x.take(1):
            self._update_encoders(cgm_batch, other_batch, y_batch)
            break
    
    def _process_tensor_input(self, x: List[tf.Tensor], y: tf.Tensor) -> None:
        """
        Procesa la entrada cuando son tensores.
        
        Parámetros:
        -----------
        x : List[tf.Tensor]
            Lista con [cgm_data, other_features]
        y : tf.Tensor
            Etiquetas (dosis objetivo)
        """
        if not isinstance(x, list) or len(x) != 2:
            raise ValueError("Si 'x' no es un dataset, debe ser una lista [cgm_data, other_features]")
            
        cgm_data, other_features = x
        self._update_encoders(cgm_data, other_features, y)
    
    def _update_encoders_from_dataset(self, dataset: tf.data.Dataset) -> None:
        """
        Actualiza los encoders basado en un dataset.
        
        Parámetros:
        -----------
        dataset : tf.data.Dataset
            Dataset a procesar
        """
        # Tomar un lote para calibrar los encoders
        for (cgm_batch, other_batch), y_batch in dataset.take(1):
            self._update_encoders(cgm_batch, other_batch, y_batch)
            break
    
    def _update_encoders(self, x_cgm: tf.Tensor, x_other: tf.Tensor, y: tf.Tensor) -> None:
        """
        Actualiza los encoders para mapear adecuadamente los valores continuos.
        
        Parámetros:
        -----------
        x_cgm : tf.Tensor
            Datos CGM
        x_other : tf.Tensor
            Otras características
        y : tf.Tensor
            Dosis objetivo
        """
        # Asegurar que el action_decoder esté construido
        if not hasattr(self.action_decoder, 'built') or not self.action_decoder.built:
            # Construir explícitamente la capa con la forma correcta
            dummy_input = tf.zeros([1, self.value_iteration_agent.n_actions])
            _ = self.action_decoder(dummy_input)
        
        # Establecer pesos para mapear linealmente desde espacio de acciones discretas a dosis continuas
        min_dose = tf.reduce_min(y, axis=0)
        max_dose = tf.reduce_max(y, axis=0)
        dose_spread = max_dose - min_dose
        
        # Asegurar que todos los valores son del mismo tipo (float32)
        dose_ratio = tf.cast(dose_spread / self.value_iteration_agent.n_actions, tf.float32)
        min_dose_float32 = tf.cast(min_dose, tf.float32)
        
        # Verificar que la forma de los pesos coincida con lo que espera la capa
        kernel_shape = self.action_decoder.kernel.shape
        bias_shape = self.action_decoder.bias.shape
        
        # Configurar capa decodificadora para mapear acciones a dosis en el rango [min_dose, max_dose]
        self.action_decoder.set_weights([
            tf.ones(kernel_shape, dtype=tf.float32) * (dose_ratio / kernel_shape[0]),
            tf.ones(bias_shape, dtype=tf.float32) * min_dose_float32
        ])
    
    def fit(
        self, 
        x: Any, 
        y: Optional[tf.Tensor] = None, 
        batch_size: int = CONST_DEFAULT_BATCH_SIZE, 
        epochs: int = CONST_DEFAULT_EPOCHS, 
        verbose: int = 1, 
        validation_data: Optional[Tuple] = None, 
        **kwargs
    ) -> Dict[str, List[float]]:
        """
        Entrena el modelo con datos de entrada.
        
        Parámetros:
        -----------
        x : Any
            Datos de entrada. Puede ser un dataset o una lista [cgm_data, other_features]
        y : Optional[tf.Tensor], opcional
            Etiquetas cuando x es una lista de tensores (default: None)
        batch_size : int, opcional
            Tamaño del lote (default: 32)
        epochs : int, opcional
            Número de épocas (default: 10)
        verbose : int, opcional
            Nivel de verbosidad (default: 1)
        validation_data : Optional[Tuple], opcional
            Datos de validación (default: None)
            
        Retorna:
        --------
        Dict[str, List[float]]
            Historial de entrenamiento
        """
        # Procesar entradas y configurar encoders/decoders
        if isinstance(x, tf.data.Dataset):
            self._process_dataset_input(x)
        else:
            self._process_tensor_input(x, y)
        
        # Extraer datos de entrenamiento
        if isinstance(x, tf.data.Dataset):
            # Extraer datos del dataset
            all_cgm = []
            all_other = []
            all_y = []
            
            for (cgm_batch, other_batch), y_batch in x:
                all_cgm.append(cgm_batch.numpy())
                all_other.append(other_batch.numpy())
                all_y.append(y_batch.numpy())
            
            cgm_data = np.vstack(all_cgm)
            other_features = np.vstack(all_other)
            targets = np.concatenate(all_y)
        else:
            # Usar tensores directamente - verificar si ya son arrays de numpy
            cgm_data = x[0].numpy() if hasattr(x[0], 'numpy') else x[0]
            other_features = x[1].numpy() if hasattr(x[1], 'numpy') else x[1]
            targets = y.numpy() if hasattr(y, 'numpy') else y
        
        # Crear entorno para el agente de Iteración de Valor
        if self.env is None:
            self.env = _create_training_environment(
                cgm_data, other_features, targets, self.value_iteration_agent
            )
        
        # Entrenar el agente de Iteración de Valor
        print("Entrenando modelo de Iteración de Valor...")
        history = self.value_iteration_agent.train(self.env)
        
        # Actualizar métricas de entrenamiento
        self.training_metrics = history
        
        # Crear historial en formato esperado por Keras
        return {
            'loss': history.get('value_changes', [0.0])[-10:],  # Últimos 10 cambios como aproximación de pérdida
            'value_changes': history.get('value_changes', []),
            'iteration_times': history.get('iteration_times', [])
        }
    
    def evaluate(
        self, 
        x: Any, 
        y: Optional[tf.Tensor] = None, 
        batch_size: int = CONST_DEFAULT_BATCH_SIZE, 
        verbose: int = 1, 
        **kwargs
    ) -> float:
        """
        Evalúa el modelo con datos de prueba.
        
        Parámetros:
        -----------
        x : Any
            Datos de entrada. Puede ser un dataset o una lista [cgm_data, other_features]
        y : Optional[tf.Tensor], opcional
            Etiquetas cuando x es una lista de tensores (default: None)
        batch_size : int, opcional
            Tamaño del lote (default: 32)
        verbose : int, opcional
            Nivel de verbosidad (default: 1)
            
        Retorna:
        --------
        float
            Pérdida de evaluación (MSE)
        """
        # Realizar predicciones
        predictions = self.predict(x, batch_size=batch_size)
        
        # Extraer etiquetas reales
        if isinstance(x, tf.data.Dataset):
            all_y = []
            for _, y_batch in x:
                all_y.append(y_batch.numpy())
            actual = np.concatenate(all_y)
        else:
            actual = y.numpy()
        
        # Calcular error cuadrático medio
        mse = np.mean((predictions - actual) ** 2)
        
        return float(mse)
    
    def predict(
        self, 
        x: Any, 
        batch_size: int = CONST_DEFAULT_BATCH_SIZE, 
        verbose: int = 0, 
        **kwargs
    ) -> np.ndarray:
        """
        Realiza predicciones con el modelo.
        
        Parámetros:
        -----------
        x : Any
            Datos de entrada. Puede ser un dataset o una lista [cgm_data, other_features]
        batch_size : int, opcional
            Tamaño del lote (default: 32)
        verbose : int, opcional
            Nivel de verbosidad (default: 0)
            
        Retorna:
        --------
        np.ndarray
            Predicciones de dosis de insulina
        """
        # Manejar diferentes tipos de entrada
        if isinstance(x, tf.data.Dataset):
            all_preds = []
            for (cgm_batch, other_batch), _ in x:
                batch_preds = self([cgm_batch, other_batch], training=False).numpy()
                all_preds.append(batch_preds)
            return np.concatenate(all_preds)
        else:
            return self(x, training=False).numpy()
    
    def get_config(self) -> Dict[str, Any]:
        """
        Obtiene la configuración del modelo para serialización.
        
        Retorna:
        --------
        Dict[str, Any]
            Configuración del modelo
        """
        config = super().get_config()
        config.update({
            'cgm_shape': self.cgm_shape,
            'other_features_shape': self.other_features_shape,
            # No podemos serializar fácilmente value_iteration_agent o discretizer
            # Para una implementación completa se necesitaría código adicional
        })
        return config


# Constantes para uso en el modelo
STATE_ENCODER = 'state_encoder'
ACTION_DECODER = 'action_decoder'
CGM_ENCODER = 'cgm_encoder'
OTHER_ENCODER = 'other_encoder'
WRAPPER_WEIGHTS_SUFFIX = '_wrapper_weights.h5'
VI_AGENT_SUFFIX = '_vi_agent'


def _prepare_data_for_env(
    cgm_data: tf.Tensor, 
    other_features: tf.Tensor, 
    targets: tf.Tensor
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Prepara los datos para su uso en el entorno de entrenamiento.
    
    Parámetros:
    -----------
    cgm_data : tf.Tensor
        Datos CGM 
    other_features : tf.Tensor
        Otras características
    targets : tf.Tensor
        Dosis objetivo
        
    Retorna:
    --------
    Tuple[np.ndarray, np.ndarray, np.ndarray]
        Datos preparados (cgm, otras características, objetivos)
    """
    # Convertir tensores a numpy para procesamiento
    cgm_np = cgm_data.numpy() if hasattr(cgm_data, 'numpy') else cgm_data
    other_np = other_features.numpy() if hasattr(other_features, 'numpy') else other_features
    target_np = targets.numpy() if hasattr(targets, 'numpy') else targets
    
    # Limitar la cantidad de datos para procesamiento más rápido
    max_samples = min(500, len(target_np))  # Reducido a 500 muestras máximo
    if len(target_np) > max_samples:
        # Usar una semilla fija para reproducibilidad
        rng = np.random.Generator(np.random.PCG64(CONST_DEFAULT_SEED))
        indices = rng.choice(len(target_np), max_samples, replace=False)
        cgm_np = cgm_np[indices]
        other_np = other_np[indices]
        target_np = target_np[indices]
    
    return cgm_np, other_np, target_np


def _process_state_features(cgm_flat: np.ndarray, feat_flat: np.ndarray) -> Tuple[float, float, float, float, int]:
    """
    Procesa las características de un estado para su discretización.
    
    Parámetros:
    -----------
    cgm_flat : np.ndarray
        Datos CGM aplanados
    feat_flat : np.ndarray
        Otras características aplanadas
        
    Retorna:
    --------
    Tuple[float, float, float, float, int]
        Características procesadas (cgm_mean_norm, cgm_last_norm, cgm_trend_norm, feat_mean_norm, patient_id)
    """
    # CGM media, último valor, y tendencia
    cgm_mean = float(np.mean(cgm_flat)) if len(cgm_flat) > 0 else 0.0
    cgm_last = float(cgm_flat[-1]) if len(cgm_flat) > 0 else 0.0
    cgm_trend = float(cgm_flat[-1] - cgm_flat[0]) if len(cgm_flat) > 1 else 0.0
    
    # Añadir algunas características de las otras variables
    feat_mean = float(np.mean(feat_flat)) if len(feat_flat) > 0 else 0.0
    
    # Valores específicos para cada paciente
    patient_id = int(hash(str(feat_flat)) % 5)  # Un ID pseudo-único basado en las características
    
    # Normalizar todo a valores entre 0 y 1
    max_cgm = 300.0  # Valor máximo típico de CGM
    cgm_mean_norm = min(1.0, max(0.0, cgm_mean / max_cgm))
    cgm_last_norm = min(1.0, max(0.0, cgm_last / max_cgm))
    cgm_trend_norm = min(1.0, max(0.0, (cgm_trend + 50) / 100.0))  # Tendencia dentro de ±50
    feat_mean_norm = min(1.0, max(0.0, abs(feat_mean) / 10.0))  # Normalización arbitraria
    
    return cgm_mean_norm, cgm_last_norm, cgm_trend_norm, feat_mean_norm, patient_id


def _handle_unique_states(cached_states: np.ndarray, n_states: int) -> np.ndarray:
    """
    Maneja los estados únicos para crear un modelo de transición adecuado.
    
    Parámetros:
    -----------
    cached_states : np.ndarray
        Estados precalculados
    n_states : int
        Número total de estados posibles
        
    Retorna:
    --------
    np.ndarray
        Estados únicos a procesar
    """
    unique_states = np.unique(cached_states)
    print(f"Procesando {len(unique_states)} estados únicos de {n_states} posibles...")
    
    if len(unique_states) == 0:
        # Si no hay estados únicos, usar un subconjunto predeterminado
        unique_states = np.array([0, n_states // 2, n_states - 1])
        print(f"No se encontraron estados únicos. Usando {len(unique_states)} estados predeterminados.")
        
    if len(unique_states) < 3:
        # Asegurar un mínimo de estados para construir un modelo útil
        additional_states = np.array([s+1 for s in unique_states if s+1 < n_states] + 
                                   [s-1 for s in unique_states if s-1 >= 0])
        unique_states = np.unique(np.concatenate([unique_states, additional_states]))
        print(f"Ahora procesando {len(unique_states)} estados.")
    
    return unique_states


def _create_training_environment(
    cgm_data: tf.Tensor, 
    other_features: tf.Tensor, 
    targets: tf.Tensor,
    value_iteration_agent: ValueIteration
) -> Any:
    """
    Crea un entorno de entrenamiento para el agente de Iteración de Valor.
    
    Parámetros:
    -----------
    cgm_data : tf.Tensor
        Datos CGM 
    other_features : tf.Tensor
        Otras características
    targets : tf.Tensor
        Dosis objetivo
    value_iteration_agent: ValueIteration
        Agente de Iteración de Valor
        
    Retorna:
    --------
    Any
        Entorno para entrenamiento
    """
    # Preparar datos
    cgm_np, other_np, target_np = _prepare_data_for_env(cgm_data, other_features, targets)
    print(f"Creando entorno con {len(target_np)} muestras...")
    
    class InsulinDosingEnv:
        """Entorno personalizado para problema de dosificación de insulina."""
        
        def __init__(self, cgm: np.ndarray, features: np.ndarray, targets: np.ndarray, 
                    agent: ValueIteration) -> None:
            """
            Inicializa el entorno.
            
            Parámetros:
            -----------
            cgm : np.ndarray
                Datos CGM
            features : np.ndarray
                Otras características
            targets : np.ndarray
                Dosis objetivo
            agent : ValueIteration
                Agente de iteración de valor
            """
            self.cgm = cgm
            self.features = features
            self.targets = targets
            self.agent = agent
            self.current_state = None
            self.n_states = agent.n_states
            self.n_actions = agent.n_actions
            self.shape = (4, 5)  # Forma arbitraria para visualización
            
            # Crear modelo de transición para todos los estados
            self.P = {}
            
            # Asignar IDs únicos a cada muestra para estado
            self.prepare_environment()
        
        def prepare_environment(self) -> None:
            """Prepara el modelo de transición del entorno."""
            print("Preparando modelo de transición del entorno...")
            
            # Calcular estados para cada muestra
            states = []
            for i in range(len(self.cgm)):
                state = self._discretize_state(self.cgm[i], self.features[i])
                states.append(state)
            
            states_array = np.array(states)
            
            # Procesar solo estados únicos para eficiencia
            unique_states = _handle_unique_states(states_array, self.n_states)
            
            # Construir modelo de transición
            for s in unique_states:
                self.P[s] = {}
                
                # Para cada acción, definir transiciones
                for a in range(self.n_actions):
                    self.P[s][a] = []
                    
                    # Encuentra muestras cercanas al estado actual
                    dose = self._action_to_dose(a)
                    
                    # Encuentra la muestra cuya dosis objetivo está más cerca de esta acción
                    distances = np.abs(self.targets - dose)
                    closest_idx = np.argmin(distances)
                    
                    # El siguiente estado será el estado correspondiente a la siguiente muestra
                    next_idx = min(closest_idx + 1, len(self.cgm) - 1)
                    next_state = self._discretize_state(self.cgm[next_idx], self.features[next_idx])
                    
                    # Calcular recompensa basada en qué tan cerca está la acción de la dosis objetivo
                    reward = -abs(dose - self.targets[closest_idx])
                    
                    # Añadir transición: (probabilidad, siguiente_estado, recompensa, terminal)
                    self.P[s][a].append((CONST_PROB_TRANSICION, next_state, reward, False))
            
            print(f"Modelo de transición creado con {len(self.P)} estados y {self.n_actions} acciones.")
        
        def _discretize_state(self, cgm: np.ndarray, features: np.ndarray) -> int:
            """Discretiza el estado continuo en un índice de estado discreto."""
            try:
                # Convertir y aplanar arrays
                cgm_array = np.array(cgm)
                cgm_flat = cgm_array.flatten() if len(cgm_array.shape) > 1 else cgm_array
                feat_flat = np.array(features).flatten() if hasattr(features, 'shape') and len(features.shape) > 1 else features
                
                # Procesar características del estado
                cgm_mean_norm, cgm_last_norm, cgm_trend_norm, feat_mean_norm, patient_id = _process_state_features(cgm_flat, feat_flat)
                
                # Discretizar con bins
                mean_bins, last_bins, trend_bins, feat_bins, patient_bins = 8, 8, 5, 5, 5
                
                # Calcular índices discretizados
                mean_idx = min(int(cgm_mean_norm * mean_bins), mean_bins - 1)
                last_idx = min(int(cgm_last_norm * last_bins), last_bins - 1)
                trend_idx = min(int(cgm_trend_norm * trend_bins), trend_bins - 1)
                feat_idx = min(int(feat_mean_norm * feat_bins), feat_bins - 1)
                
                # Combinar en un único estado discreto (enfoque de codificación posicional)
                state = (mean_idx * last_bins * trend_bins * feat_bins * patient_bins + 
                        last_idx * trend_bins * feat_bins * patient_bins + 
                        trend_idx * feat_bins * patient_bins + 
                        feat_idx * patient_bins + 
                        patient_id)
                
                return min(state, self.n_states - 1)
                
            except Exception as e:
                print(f"Error en discretización: {e}")
                return 0
        
        def _action_to_dose(self, action: int) -> float:
            """Convierte una acción discreta a una dosis continua."""
            min_dose = np.min(self.targets)
            max_dose = np.max(self.targets)
            
            return min_dose + (action / (self.n_actions - 1)) * (max_dose - min_dose)
        
        def reset(self) -> Tuple[int, Dict]:
            """
            Reinicia el entorno a un estado inicial.
            
            Retorna:
            --------
            Tuple[int, Dict]
                Estado inicial y metadata
            """
            # Seleccionar un estado inicial aleatorio de los disponibles
            available_states = list(self.P.keys())
            if not available_states:
                # Si no hay estados disponibles, usar un valor por defecto
                self.current_state = 0
            else:
                # Elegir un estado aleatorio usando índice 0 para reproducibilidad
                self.current_state = available_states[0]
            
            return self.current_state, {}
        
        def step(self, action: int) -> Tuple[int, float, bool, bool, Dict]:
            """
            Ejecuta un paso en el entorno.
            
            Parámetros:
            -----------
            action : int
                Acción a ejecutar
                
            Retorna:
            --------
            Tuple[int, float, bool, bool, Dict]
                (siguiente_estado, recompensa, terminado, truncado, info)
            """
            if self.current_state not in self.P or action not in self.P[self.current_state]:
                # Manejar caso donde el estado o acción no están en el modelo
                # Volver al estado inicial con recompensa negativa
                self.current_state, _ = self.reset()
                return self.current_state, -10.0, False, True, {}
            
            # Obtener transición según el modelo
            # En este caso simple, solo hay una transición por par estado-acción
            _, next_state, reward, done = self.P[self.current_state][action][0]
            
            # Actualizar estado actual
            self.current_state = next_state
            
            return next_state, reward, done, False, {}
    
    return InsulinDosingEnv(cgm_np, other_np, target_np, value_iteration_agent)


def create_value_iteration_model(cgm_shape: Tuple[int, ...], other_features_shape: Tuple[int, ...]) -> Model:
    """
    Crea un modelo basado en Iteración de Valor para predicción de dosis de insulina.
    
    Parámetros:
    -----------
    cgm_shape : Tuple[int, ...]
        Forma de los datos CGM (batch_size, time_steps, features)
    other_features_shape : Tuple[int, ...]
        Forma de otras características (batch_size, n_features)
        
    Retorna:
    --------
    Model
        Modelo de Iteración de Valor que implementa la interfaz de Keras
    """
    # Configuración del espacio de estados y acciones
    n_states = 1000  # Estados discretos (ajustar según complejidad del problema)
    n_actions = 20   # Acciones discretas (niveles de dosis de insulina)
    
    # Crear agente de Iteración de Valor
    value_iteration_agent = ValueIteration(
        n_states=n_states,
        n_actions=n_actions,
        gamma=VALUE_ITERATION_CONFIG['gamma'],
        theta=VALUE_ITERATION_CONFIG['theta'],
        max_iterations=VALUE_ITERATION_CONFIG['max_iterations']
    )
    
    # Crear y devolver el modelo wrapper
    return ValueIterationModel(
        value_iteration_agent=value_iteration_agent,
        cgm_shape=cgm_shape,
        other_features_shape=other_features_shape
    )