import os, sys
import numpy as np
import matplotlib.pyplot as plt
import time
from typing import Dict, List, Tuple, Optional, Union, Any
import pickle
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Concatenate
from keras.saving import register_keras_serializable

PROJECT_ROOT = os.path.abspath(os.getcwd())
sys.path.append(PROJECT_ROOT) 

from config.models_config import POLICY_ITERATION_CONFIG


class PolicyIteration:
    """
    Implementación del algoritmo de Iteración de Política.
    
    La Iteración de Política alterna entre Evaluación de Política (calcular la función
    de valor para la política actual) y Mejora de Política (hacer la política codiciosa
    respecto a la función de valor actual).
    """
    # Constantes para etiquetas de gráficos
    ITERATION_LABEL = 'Iteración'
    
    def __init__(
        self,
        n_states: int,
        n_actions: int,
        gamma: float = POLICY_ITERATION_CONFIG['gamma'],
        theta: float = POLICY_ITERATION_CONFIG['theta'],
        max_iterations: int = POLICY_ITERATION_CONFIG['max_iterations'],
        max_iterations_eval: int = POLICY_ITERATION_CONFIG['max_iterations_eval'],
        seed: int = POLICY_ITERATION_CONFIG.get('seed', 42)
    ) -> None:
        """
        Inicializa el agente de Iteración de Política.
        
        Parámetros:
        -----------
        n_states : int
            Número de estados en el entorno
        n_actions : int
            Número de acciones en el entorno
        gamma : float, opcional
            Factor de descuento (default: POLICY_ITERATION_CONFIG['gamma'])
        theta : float, opcional
            Umbral para convergencia (default: POLICY_ITERATION_CONFIG['theta'])
        max_iterations : int, opcional
            Número máximo de iteraciones de iteración de política 
            (default: POLICY_ITERATION_CONFIG['max_iterations'])
        max_iterations_eval : int, opcional
            Número máximo de iteraciones para evaluación de política 
            (default: POLICY_ITERATION_CONFIG['max_iterations_eval'])
        seed : int, opcional
            Semilla para reproducibilidad (default: POLICY_ITERATION_CONFIG.get('seed', 42))
        """
        self.n_states = n_states
        self.n_actions = n_actions
        self.gamma = gamma
        self.theta = theta
        self.max_iterations = max_iterations
        self.max_iterations_eval = max_iterations_eval
        
        # Configurar generador de números aleatorios para reproducibilidad
        self.rng = np.random.Generator(np.random.PCG64(seed))
        
        # Inicializar función de valor
        self.v = np.zeros(n_states)
        
        # Inicializar política (aleatoria uniforme)
        self.policy = np.ones((n_states, n_actions)) / n_actions
        
        # Para métricas
        self.policy_changes = []
        self.value_changes = []
        self.policy_iteration_times = []
        self.eval_iteration_counts = []
    
    def _calculate_state_value(
        self, 
        env: Any, 
        state: int, 
        policy: np.ndarray, 
        v: np.ndarray
    ) -> float:
        """
        Calcula el valor de un estado dado según la política actual.
        
        Parámetros:
        -----------
        env : Any
            Entorno con dinámicas de transición
        state : int
            Estado a evaluar
        policy : np.ndarray
            Política actual
        v : np.ndarray
            Función de valor actual
            
        Retorna:
        --------
        float
            Nuevo valor del estado
        """
        v_new = 0
        
        # Calcular valor esperado al seguir la política en el estado
        for a in range(self.n_actions):
            if policy[state, a] > 0:  # Solo considerar acciones con probabilidad no cero
                # Obtener información del siguiente estado y recompensa
                for prob, next_s, r, done in env.P[state][a]:
                    # Actualizar valor del estado usando la ecuación de Bellman
                    v_new += policy[state, a] * prob * (r + self.gamma * v[next_s] * (not done))
        
        return v_new
    
    def policy_evaluation(self, env: Any, policy: np.ndarray) -> np.ndarray:
        """
        Evalúa la política actual calculando su función de valor.
        
        Parámetros:
        -----------
        env : Any
            Entorno con dinámicas de transición
        policy : np.ndarray
            Política a evaluar (distribución de probabilidad sobre acciones para cada estado)
            
        Retorna:
        --------
        np.ndarray
            Función de valor para la política dada
        """
        v = np.zeros(self.n_states)
        
        for i in range(self.max_iterations_eval):
            delta = 0
            
            for s in range(self.n_states):
                v_old = v[s]
                v[s] = self._calculate_state_value(env, s, policy, v)
                delta = max(delta, abs(v_old - v[s]))
            
            # Verificar convergencia
            if delta < self.theta:
                break
        
        self.eval_iteration_counts.append(i + 1)
        return v
    
    def policy_improvement(self, env: Any, v: np.ndarray) -> Tuple[np.ndarray, bool]:
        """
        Mejora la política haciéndola codiciosa respecto a la función de valor.
        
        Parámetros:
        -----------
        env : Any
            Entorno con dinámicas de transición
        v : np.ndarray
            Función de valor actual
            
        Retorna:
        --------
        Tuple[np.ndarray, bool]
            Tupla de (nueva política, política_estable)
        """
        policy = np.zeros((self.n_states, self.n_actions))
        policy_stable = True
        
        for s in range(self.n_states):
            # Encontrar la mejor acción anterior
            old_action = np.argmax(self.policy[s])
            
            # Calcular valores de acción para el estado actual
            action_values = np.zeros(self.n_actions)
            
            for a in range(self.n_actions):
                for prob, next_s, r, done in env.P[s][a]:
                    action_values[a] += prob * (r + self.gamma * v[next_s] * (not done))
            
            # Obtener la nueva mejor acción (con valor máximo)
            best_action = np.argmax(action_values)
            
            # Manejar acciones con valores idénticos (romper empates de manera determinista)
            if np.sum(action_values == action_values[best_action]) > 1:
                # Encontrar todas las acciones con valor máximo
                max_indices = np.nonzero(action_values == action_values[best_action])[0]
                # Elegir una de manera determinista basada en el estado
                best_action = max_indices[hash(s) % len(max_indices)]
            
            # Actualizar política: determinística (probabilidad 1.0 para la mejor acción)
            policy[s, best_action] = 1.0
            
            # Verificar si la política cambió
            if old_action != best_action:
                policy_stable = False
        
        return policy, policy_stable
    
    def train(self, env: Any) -> Dict[str, List]:
        """
        Entrena al agente usando iteración de política.
        
        Parámetros:
        -----------
        env : Any
            Entorno con dinámicas de transición
            
        Retorna:
        --------
        Dict[str, List]
            Diccionario con historial de entrenamiento
        """
        print("Iniciando iteración de política...")
        
        policy_stable = False
        iterations = 0
        
        start_time = time.time()
        
        while not policy_stable and iterations < self.max_iterations:
            iteration_start = time.time()
            
            # Evaluación de Política: Calcular función de valor para la política actual
            self.v = self.policy_evaluation(env, self.policy)
            
            # Calcular cambio de valor para métricas
            if iterations > 0:
                value_change = np.mean(np.abs(self.v - old_v))
                self.value_changes.append(value_change)
            old_v = self.v.copy()
            
            # Mejora de Política: Actualizar política basada en nueva función de valor
            new_policy, policy_stable = self.policy_improvement(env, self.v)
            
            # Calcular cambio de política para métricas
            if iterations > 0:
                policy_change = np.sum(np.abs(new_policy - self.policy)) / (2 * self.n_states)
                self.policy_changes.append(policy_change)
            
            self.policy = new_policy
            iterations += 1
            
            # Registrar tiempo transcurrido
            iteration_time = time.time() - iteration_start
            self.policy_iteration_times.append(iteration_time)
            
            print(f"Iteración {iterations}: {iteration_time:.2f} segundos, " + 
                  f"Iteraciones de evaluación: {self.eval_iteration_counts[-1]}")
            
            if policy_stable:
                print("¡Política convergida!")
        
        total_time = time.time() - start_time
        print(f"Iteración de política completada en {iterations} iteraciones, {total_time:.2f} segundos")
        
        history = {
            'iterations': iterations,
            'policy_changes': self.policy_changes,
            'value_changes': self.value_changes,
            'iteration_times': self.policy_iteration_times,
            'eval_iterations': self.eval_iteration_counts,
            'total_time': total_time
        }
        
        return history
    
    def modified_policy_iteration(self, env: Any, k_eval: int = 5) -> Dict[str, List]:
        """
        Implementa la Iteración de Política Modificada que utiliza un número fijo
        de iteraciones para la evaluación de política.
        
        Parámetros:
        -----------
        env : Any
            Entorno con dinámicas de transición
        k_eval : int, opcional
            Número de iteraciones para evaluación de política (default: 5)
            
        Retorna:
        --------
        Dict[str, List]
            Diccionario con historial de entrenamiento
        """
        print(f"Iniciando iteración de política modificada (k={k_eval})...")
        
        policy_stable = False
        iterations = 0
        
        start_time = time.time()
        
        # Inicializar función de valor
        v = np.zeros(self.n_states)
        
        while not policy_stable and iterations < self.max_iterations:
            iteration_start = time.time()
            
            # Evaluación de política parcial: k iteraciones
            for _ in range(k_eval):
                new_v = np.zeros_like(v)
                
                for s in range(self.n_states):
                    new_v[s] = self._calculate_state_value(env, s, self.policy, v)
                
                v = new_v
            
            # Calcular cambio de valor para métricas
            if iterations > 0:
                value_change = np.mean(np.abs(v - old_v))
                self.value_changes.append(value_change)
            old_v = v.copy()
            
            # Mejora de Política: Actualizar política basada en nueva función de valor
            new_policy, policy_stable = self.policy_improvement(env, v)
            
            # Calcular cambio de política para métricas
            if iterations > 0:
                policy_change = np.sum(np.abs(new_policy - self.policy)) / (2 * self.n_states)
                self.policy_changes.append(policy_change)
            
            self.policy = new_policy
            iterations += 1
            
            # Registrar tiempo transcurrido
            iteration_time = time.time() - iteration_start
            self.policy_iteration_times.append(iteration_time)
            
            print(f"Iteración {iterations}: {iteration_time:.2f} segundos")
            
            if policy_stable:
                print("¡Política convergida!")
        
        # Evaluación final para obtener valores precisos
        self.v = self.policy_evaluation(env, self.policy)
        
        total_time = time.time() - start_time
        print(f"Iteración de política modificada completada en {iterations} iteraciones, {total_time:.2f} segundos")
        
        history = {
            'iterations': iterations,
            'policy_changes': self.policy_changes,
            'value_changes': self.value_changes,
            'iteration_times': self.policy_iteration_times,
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
        return self.v[state]
    
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
            done = False
            total_reward = 0
            steps = 0
            
            while not done and steps < max_steps:
                action = self.get_action(state)
                next_state, reward, terminated, truncated, _ = env.step(action)
                done = terminated or truncated
                
                total_reward += reward
                state = next_state
                steps += 1
            
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
            'v': self.v,
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
        self.v = data['v'] if 'v' in data else data.get('V', np.zeros(data['n_states']))
        self.n_states = data['n_states']
        self.n_actions = data['n_actions']
        self.gamma = data['gamma']
        
        print(f"Modelo cargado desde {filepath}")
    
    def _run_value_iteration(self, env: Any, max_iterations: int) -> Tuple[np.ndarray, int, np.ndarray]:
        """
        Ejecuta el algoritmo de Iteración de Valor.
        
        Parámetros:
        -----------
        env : Any
            Entorno con dinámicas de transición
        max_iterations : int
            Número máximo de iteraciones
            
        Retorna:
        --------
        Tuple[np.ndarray, int, np.ndarray]
            Tupla de (valores, número de iteraciones, política)
        """
        v = np.zeros(self.n_states)
        iterations = 0
        
        for i in range(max_iterations):
            delta = 0
            
            for s in range(self.n_states):
                v_old = v[s]
                action_values = self._calculate_action_values(env, s, v)
                v[s] = np.max(action_values)
                delta = max(delta, abs(v_old - v[s]))
            
            iterations = i + 1
            
            if delta < self.theta:
                break
        
        # Derivar política de los valores
        policy = self._derive_policy_from_values(env, v)
        
        return v, iterations, policy
    
    def _calculate_action_values(self, env: Any, state: int, v: np.ndarray) -> np.ndarray:
        """
        Calcula los valores de acción para un estado dado.
        
        Parámetros:
        -----------
        env : Any
            Entorno con dinámicas de transición
        state : int
            Estado para calcular valores de acción
        v : np.ndarray
            Función de valor actual
            
        Retorna:
        --------
        np.ndarray
            Array con valores para cada acción
        """
        action_values = np.zeros(self.n_actions)
        
        for a in range(self.n_actions):
            for prob, next_s, reward, done in env.P[state][a]:
                action_values[a] += prob * (reward + self.gamma * v[next_s] * (not done))
        
        return action_values
    
    def _derive_policy_from_values(self, env: Any, v: np.ndarray) -> np.ndarray:
        """
        Deriva una política determinística a partir de una función de valor.
        
        Parámetros:
        -----------
        env : Any
            Entorno con dinámicas de transición
        v : np.ndarray
            Función de valor
            
        Retorna:
        --------
        np.ndarray
            Política determinística
        """
        policy = np.zeros((self.n_states, self.n_actions))
        
        for s in range(self.n_states):
            action_values = self._calculate_action_values(env, s, v)
            best_action = np.argmax(action_values)
            policy[s, best_action] = 1.0
        
        return policy
    
    def compare_with_value_iteration(
        self, 
        env: Any, 
        max_iterations: int = POLICY_ITERATION_CONFIG['max_iterations']
    ) -> Dict[str, List]:
        """
        Compara Iteración de Política con Iteración de Valor.
        
        Parámetros:
        -----------
        env : Any
            Entorno con dinámicas de transición
        max_iterations : int, opcional
            Número máximo de iteraciones (default: POLICY_ITERATION_CONFIG['max_iterations'])
            
        Retorna:
        --------
        Dict[str, List]
            Diccionario con resultados de la comparación
        """
        # Guardar configuración original
        original_max_iterations = self.max_iterations
        self.max_iterations = max_iterations
        
        # Resultados de la comparación
        comparison = {
            'policy_iteration': {'time': 0, 'iterations': 0, 'values': []},
            'value_iteration': {'time': 0, 'iterations': 0, 'values': []}
        }
        
        # Ejecutar Iteración de Política
        print("\n--- Iteración de Política ---")
        pi_start = time.time()
        pi_history = self.train(env)
        pi_time = time.time() - pi_start
        pi_values = self.v.copy()
        
        comparison['policy_iteration']['time'] = pi_time
        comparison['policy_iteration']['iterations'] = pi_history['iterations']
        comparison['policy_iteration']['values'] = pi_values
        
        # Ejecutar Iteración de Valor
        print("\n--- Iteración de Valor ---")
        vi_start = time.time()
        v, vi_iterations, value_based_policy = self._run_value_iteration(env, max_iterations)
        vi_time = time.time() - vi_start
        
        comparison['value_iteration']['time'] = vi_time
        comparison['value_iteration']['iterations'] = vi_iterations
        comparison['value_iteration']['values'] = v
        
        # Calcular diferencia entre políticas y valores
        pi_actions = np.argmax(self.policy, axis=1)
        vi_actions = np.argmax(value_based_policy, axis=1)
        policy_difference = np.sum(pi_actions != vi_actions) / self.n_states
        value_difference = np.mean(np.abs(pi_values - v))
        
        # Mostrar resultados
        print("\n--- Comparación ---")
        print(f"Iteración de Política: {pi_history['iterations']} iteraciones, {pi_time:.2f} segundos")
        print(f"Iteración de Valor: {vi_iterations} iteraciones, {vi_time:.2f} segundos")
        print(f"Diferencia en política: {policy_difference:.2%}")
        print(f"Diferencia en valores: {value_difference:.6f}")
        
        # Restaurar configuración original
        self.max_iterations = original_max_iterations
        
        return comparison
    
    def _get_grid_position(self, env: Any, s: int, grid_shape: Tuple[int, int]) -> Tuple[int, int]:
        """
        Obtiene la posición en la cuadrícula para un estado dado.
        
        Parámetros:
        -----------
        env : Any
            Entorno con estructura de cuadrícula
        s : int
            Índice del estado
        grid_shape : Tuple[int, int]
            Forma de la cuadrícula
            
        Retorna:
        --------
        Tuple[int, int]
            Posición (fila, columna) en la cuadrícula
        """
        if hasattr(env, 'state_mapping'):
            return env.state_mapping(s)
        else:
            # Asumir orden row-major
            return s // grid_shape[1], s % grid_shape[1]
    
    def _is_terminal_state(self, env: Any, s: int) -> bool:
        """
        Verifica si un estado es terminal.
        
        Parámetros:
        -----------
        env : Any
            Entorno con dinámicas de transición
        s : int
            Índice del estado
            
        Retorna:
        --------
        bool
            True si el estado es terminal, False en caso contrario
        """
        for a in range(self.n_actions):
            for prob, next_s, done, *_ in env.P[s][a]:
                if done:
                    return True
        return False
    
    def _draw_grid_lines(self, ax, grid_shape: Tuple[int, int]) -> None:
        """
        Dibuja las líneas de la cuadrícula.
        
        Parámetros:
        -----------
        ax : matplotlib.axes.Axes
            Ejes para dibujar
        grid_shape : Tuple[int, int]
            Forma de la cuadrícula
        """
        # Configurar límites
        ax.set_xlim([0, grid_shape[1]])
        ax.set_ylim([0, grid_shape[0]])
        
        # Dibujar líneas verticales
        for i in range(grid_shape[1] + 1):
            ax.axvline(i, color='black', linestyle='-')
        
        # Dibujar líneas horizontales
        for j in range(grid_shape[0] + 1):
            ax.axhline(j, color='black', linestyle='-')
    
    def _draw_policy_arrows(self, ax, env: Any, grid_shape: Tuple[int, int]) -> None:
        """
        Dibuja flechas para representar la política.
        
        Parámetros:
        -----------
        ax : matplotlib.axes.Axes
            Ejes para dibujar
        env : Any
            Entorno con estructura de cuadrícula
        grid_shape : Tuple[int, int]
            Forma de la cuadrícula
        """
        # Definir direcciones de flechas
        directions = {
            0: (0, -0.4),  # Izquierda
            1: (0, 0.4),   # Derecha
            2: (-0.4, 0),  # Abajo
            3: (0.4, 0)    # Arriba
        }
        
        for s in range(self.n_states):
            # Omitir estados terminales
            if self._is_terminal_state(env, s):
                continue
            
            i, j = self._get_grid_position(env, s, grid_shape)
            action = self.get_action(s)
            
            if action in directions:
                dx, dy = directions[action]
                ax.arrow(j + 0.5, grid_shape[0] - i - 0.5, dx, dy, 
                         head_width=0.1, head_length=0.1, fc='blue', ec='blue')
    
    def _draw_state_values(self, ax, env: Any, grid_shape: Tuple[int, int]) -> None:
        """
        Dibuja los valores de los estados en la cuadrícula.
        
        Parámetros:
        -----------
        ax : matplotlib.axes.Axes
            Ejes para dibujar
        env : Any
            Entorno con estructura de cuadrícula
        grid_shape : Tuple[int, int]
            Forma de la cuadrícula
        """
        for s in range(self.n_states):
            i, j = self._get_grid_position(env, s, grid_shape)
            value = self.get_value(s)
            ax.text(j + 0.5, grid_shape[0] - i - 0.5, f"{value:.2f}", 
                   ha='center', va='center', color='red', fontsize=9)
    
    def visualize_policy(self, env: Any, title: str = "Política") -> None:
        """
        Visualiza la política para entornos de tipo cuadrícula.
        
        Parámetros:
        -----------
        env : Any
            Entorno con estructura de cuadrícula
        title : str, opcional
            Título para la visualización (default: "Política")
        """
        if not hasattr(env, 'shape'):
            print("El entorno no tiene estructura de cuadrícula para visualización")
            return
        
        grid_shape = env.shape
        _, ax = plt.subplots(figsize=(8, 8))
        
        # Dibujar componentes de la visualización
        self._draw_grid_lines(ax, grid_shape)
        self._draw_policy_arrows(ax, env, grid_shape)
        self._draw_state_values(ax, env, grid_shape)
        
        ax.set_title(title)
        plt.tight_layout()
        plt.show()
    
    def visualize_training(self, history: Dict[str, List]) -> None:
        """
        Visualiza las métricas de entrenamiento.
        
        Parámetros:
        -----------
        history : Dict[str, List]
            Diccionario con historial de entrenamiento
        """
        _, axs = plt.subplots(2, 2, figsize=(15, 10))
        
        # Gráfico de cambios en la política
        if 'policy_changes' in history and history['policy_changes']:
            x = range(1, len(history['policy_changes']) + 1)
            axs[0, 0].plot(x, history['policy_changes'], marker='o')
            axs[0, 0].set_title('Cambios en la Política')
            axs[0, 0].set_xlabel(self.ITERATION_LABEL)
            axs[0, 0].set_ylabel('Cambio de Política')
            axs[0, 0].grid(True)
        
        # Gráfico de cambios en la función de valor
        if 'value_changes' in history and history['value_changes']:
            x = range(1, len(history['value_changes']) + 1)
            axs[0, 1].plot(x, history['value_changes'], marker='o')
            axs[0, 1].set_title('Cambios en la Función de Valor')
            axs[0, 1].set_xlabel(self.ITERATION_LABEL)
            axs[0, 1].set_ylabel('Cambio Promedio de Valor')
            axs[0, 1].grid(True)
        
        # Gráfico de tiempos de iteración
        if 'iteration_times' in history and history['iteration_times']:
            x = range(1, len(history['iteration_times']) + 1)
            axs[1, 0].plot(x, history['iteration_times'], marker='o')
            axs[1, 0].set_title('Tiempos de Iteración')
            axs[1, 0].set_xlabel(self.ITERATION_LABEL)
            axs[1, 0].set_ylabel('Tiempo (segundos)')
            axs[1, 0].grid(True)
        
        # Gráfico de iteraciones de evaluación
        if 'eval_iterations' in history and history['eval_iterations']:
            x = range(1, len(history['eval_iterations']) + 1)
            axs[1, 1].plot(x, history['eval_iterations'], marker='o')
            axs[1, 1].set_title('Iteraciones de Evaluación de Política')
            axs[1, 1].set_xlabel('Iteración de Política')
            axs[1, 1].set_ylabel('Número de Iteraciones')
            axs[1, 1].grid(True)
        
        plt.tight_layout()
        plt.show()

@register_keras_serializable
class PolicyIterationModel(Model):
    """
    Modelo wrapper para Iteración de Política compatible con la interfaz de Keras.
    """
    
    def __init__(
        self, 
        policy_iteration_agent: PolicyIteration,
        cgm_shape: Tuple[int, ...],
        other_features_shape: Tuple[int, ...],
        discretizer: Optional[Any] = None
    ) -> None:
        """
        Inicializa el modelo wrapper para Iteración de Política.
        
        Parámetros:
        -----------
        policy_iteration_agent : PolicyIteration
            Agente de Iteración de Política a utilizar
        cgm_shape : Tuple[int, ...]
            Forma de entrada para datos CGM
        other_features_shape : Tuple[int, ...]
            Forma de entrada para otras características
        discretizer : Optional[Any], opcional
            Función para discretizar estados continuos (default: None)
        """
        super(PolicyIterationModel, self).__init__()
        self.policy_iteration_agent = policy_iteration_agent
        self.cgm_shape = cgm_shape
        self.other_features_shape = other_features_shape
        
        # Red simple para combinar inputs en una representación de estado
        self.cgm_encoder = Dense(32, activation="relu")
        self.other_encoder = Dense(16, activation="relu")
        self.combined_encoder = Dense(policy_iteration_agent.n_states, activation="softmax")
        self.action_decoder = Dense(1, kernel_initializer="glorot_uniform")
        
        # Constantes
        self.policy_file_ext = ".policy"
        self.weight_file_ext = ".weights.h5"
        
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
        cgm_data, other_features = inputs
        
        # Mapear inputs a estados discretos
        states = self._encode_states(cgm_data, other_features)
        _ = tf.shape(states)[0]
        
        # Construir seleccionador de acción basado en la política óptima
        policy_matrix = tf.constant(self.policy_iteration_agent.policy, dtype=tf.float32)
        
        # Extraer las acciones para los estados calculados
        actions = tf.matmul(states, policy_matrix)
        
        # Mapear acción discreta a valor continuo de dosis
        # Usar la representación continua para predecir la dosis
        action_values = self.action_decoder(actions)
        
        return action_values
        
    def _encode_states(self, cgm_data: tf.Tensor, other_features: tf.Tensor) -> tf.Tensor:
        """
        Convierte CGM y otras características en una representación de estados discretos.
        
        Parámetros:
        -----------
        cgm_data : tf.Tensor
            Datos de monitoreo continuo de glucosa
        other_features : tf.Tensor
            Otras características (carbohidratos, insulina a bordo, etc.)
            
        Retorna:
        --------
        tf.Tensor
            Representación one-hot de estados discretos
        """
        # Extraer características relevantes de CGM
        batch_size = tf.shape(cgm_data)[0]
        cgm_flat = tf.reshape(cgm_data, [batch_size, -1])
        
        # Procesar ambos tipos de características
        cgm_encoded = self.cgm_encoder(cgm_flat)
        other_encoded = self.other_encoder(other_features)
        
        # Concatenar y obtener distribución sobre estados discretos
        combined = Concatenate()([cgm_encoded, other_encoded])
        state_distribution = self.combined_encoder(combined)
        
        return state_distribution
        
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
        Simula el entrenamiento con interfaz compatible con Keras.
        
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
        # Actualizar encoders en base a los datos
        self._update_encoders(x, y)
        
        # Simulación de entrenamiento
        history = {
            'loss': [0.0],
            'val_loss': [0.0] if validation_data is not None else None
        }
        
        # Usado solo para simulación de interfaz compatible
        return {'history': history}
        
    def _update_encoders(self, x: List[tf.Tensor], y: tf.Tensor) -> None:
        """
        Actualiza los encoders para mapear mejor los datos a estados.
        
        Parámetros:
        -----------
        x : List[tf.Tensor]
            Lista con [cgm_data, other_features]
        y : tf.Tensor
            Etiquetas (dosis objetivo)
        """
        _, _ = x
        
        # Calcular rangos para parámetros de la capa de salida
        max_dose = tf.reduce_max(y, axis=0)
        min_dose = tf.reduce_min(y, axis=0)
        
        # Actualizar capa para mapear a rango correcto de dosis
        dose_spread = max_dose - min_dose
        self.action_decoder.set_weights([
            tf.ones([self.policy_iteration_agent.n_actions, 1]) * (dose_spread / self.policy_iteration_agent.n_actions),
            tf.ones([1]) * min_dose
        ])
        
    def predict(self, x: List[tf.Tensor], **kwargs) -> tf.Tensor:
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
        tf.Tensor
            Predicciones de dosis
        """
        return self.call(x)
        
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
            "other_features_shape": self.other_features_shape,
            "n_states": self.policy_iteration_agent.n_states,
            "n_actions": self.policy_iteration_agent.n_actions,
            "gamma": self.policy_iteration_agent.gamma
        }
        
    def save(self, filepath: str, **kwargs) -> None:
        """
        Guarda el modelo y el agente de Iteración de Política.
        
        Parámetros:
        -----------
        filepath : str
            Ruta donde guardar el modelo
        **kwargs
            Argumentos adicionales
        """
        # Guardar el agente de iteración de política
        self.policy_iteration_agent.save(filepath + self.policy_file_ext)
        
        # Guardar la parte del modelo de keras (encoders/decoders)
        super().save_weights(filepath + self.weight_file_ext)
        
    def load_weights(self, filepath: str, **kwargs) -> None:
        """
        Carga el modelo y el agente de Iteración de Política.
        
        Parámetros:
        -----------
        filepath : str
            Ruta desde donde cargar el modelo
        **kwargs
            Argumentos adicionales
        """
        # Cargar el agente de iteración de política
        if filepath.endswith(self.weight_file_ext):
            policy_file = filepath.replace(self.weight_file_ext, self.policy_file_ext)
        else:
            policy_file = filepath + self.policy_file_ext
            
        self.policy_iteration_agent.load(policy_file)
        
        # Cargar la parte del modelo de keras
        super().load_weights(filepath + self.weight_file_ext if not filepath.endswith(self.weight_file_ext) else filepath)


def create_policy_iteration_model(cgm_shape: Tuple[int, ...], other_features_shape: Tuple[int, ...]) -> Model:
    """
    Crea un modelo basado en Iteración de Política para predicción de dosis de insulina.
    
    Parámetros:
    -----------
    cgm_shape : Tuple[int, ...]
        Forma de los datos CGM (batch_size, time_steps, features)
    other_features_shape : Tuple[int, ...]
        Forma de otras características (batch_size, n_features)
        
    Retorna:
    --------
    Model
        Modelo de Iteración de Política que implementa la interfaz de Keras
    """
    # Configuración del espacio de estados y acciones
    n_states = 1000  # Estados discretos (ajustar según complejidad del problema)
    n_actions = 20   # Acciones discretas (niveles de dosis de insulina)
    
    # Crear agente de Iteración de Política
    policy_iteration_agent = PolicyIteration(
        n_states=n_states,
        n_actions=n_actions,
        gamma=POLICY_ITERATION_CONFIG['gamma'],
        theta=POLICY_ITERATION_CONFIG['theta'],
        max_iterations=POLICY_ITERATION_CONFIG['max_iterations'],
        max_iterations_eval=POLICY_ITERATION_CONFIG['max_iterations_eval']
    )
    
    # Crear y devolver el modelo wrapper
    return PolicyIterationModel(
        policy_iteration_agent=policy_iteration_agent,
        cgm_shape=cgm_shape,
        other_features_shape=other_features_shape
    )