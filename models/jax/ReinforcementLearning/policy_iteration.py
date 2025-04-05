import os, sys
import jax
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
import time
from typing import Dict, List, Tuple, Optional, Union, Any, Callable
import pickle
from functools import partial

PROJECT_ROOT = os.path.abspath(os.getcwd())
sys.path.append(PROJECT_ROOT) 

from config.models_config import POLICY_ITERATION_CONFIG


class PolicyIteration:
    """
    Implementación del algoritmo de Iteración de Política usando JAX.
    
    La Iteración de Política alterna entre Evaluación de Política (calcular la función
    de valor para la política actual) y Mejora de Política (hacer la política codiciosa
    respecto a la función de valor actual).
    """
    # Constantes para etiquetas de gráficos y operaciones
    ITERATION_LABEL = 'Iteración'
    EINSUM_PATTERN = 'san,n->sa'  # Patrón para calcular valores esperados
    
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
        
        # Configurar claves para aleatorización
        self.key = jax.random.key(seed)
        self.np_rng = np.random.Generator(np.random.PCG64(seed))
        
        # Inicializar función de valor
        self.v = jnp.zeros(n_states)
        
        # Inicializar política (aleatoria uniforme)
        self.policy = jnp.ones((n_states, n_actions)) / n_actions
        
        # Para métricas
        self.policy_changes = []
        self.value_changes = []
        self.policy_iteration_times = []
        self.eval_iteration_counts = []
        
        # Compilar funciones clave para mejorar rendimiento
        self._init_jitted_functions()
    
    def _init_jitted_functions(self) -> None:
        """
        Inicializa funciones JIT-compiladas para mejorar el rendimiento.
        """
        # Definimos las funciones que serán compiladas con JIT
        self._jit_policy_evaluation_step = jax.jit(self._policy_evaluation_step)
        self._jit_calculate_state_values = jax.jit(self._calculate_state_values)
    
    def _prepare_transition_matrices(self, env: Any) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        """
        Prepara matrices de transición para cálculos JAX eficientes.
        
        Parámetros:
        -----------
        env : Any
            Entorno con dinámicas de transición
            
        Retorna:
        --------
        Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]
            Matrices de transición: (probabilidades, recompensas, terminales)
        """
        # Crear matrices para probabilidades de transición, recompensas y estados terminales
        transition_probs = np.zeros((self.n_states, self.n_actions, self.n_states))
        rewards = np.zeros((self.n_states, self.n_actions))
        terminals = np.zeros((self.n_states, self.n_actions), dtype=bool)
        
        # Llenar matrices desde la información del entorno
        for s in range(self.n_states):
            for a in range(self.n_actions):
                for prob, next_s, r, done in env.P[s][a]:
                    transition_probs[s, a, next_s] += prob
                    rewards[s, a] += prob * r  # Recompensa esperada
                    if done:
                        terminals[s, a] = True
        
        return jnp.array(transition_probs), jnp.array(rewards), jnp.array(terminals)
    
    def _calculate_state_values(
        self, 
        policy: jnp.ndarray, 
        v: jnp.ndarray,
        transition_probs: jnp.ndarray, 
        rewards: jnp.ndarray, 
        terminals: jnp.ndarray
    ) -> jnp.ndarray:
        """
        Calcula valores de estado para todos los estados en un solo paso.
        
        Parámetros:
        -----------
        policy : jnp.ndarray
            Política actual
        v : jnp.ndarray
            Función de valor actual
        transition_probs : jnp.ndarray
            Matriz de probabilidades de transición
        rewards : jnp.ndarray
            Matriz de recompensas
        terminals : jnp.ndarray
            Matriz de indicadores de terminal
            
        Retorna:
        --------
        jnp.ndarray
            Nuevos valores de estado
        """
        # Calcular el valor de cada estado siguiendo la política
        # Para cada estado s, acción a, y estado siguiente s':
        # v_new[s] = sum_a policy[s,a] * (rewards[s,a] + gamma * sum_s' P[s,a,s'] * v[s'] * (not terminal[s,a]))
        expected_values = rewards + self.gamma * jnp.einsum(self.EINSUM_PATTERN, transition_probs, v)
        expected_values = jnp.where(terminals, rewards, expected_values)
        
        # Promediar sobre la política
        v_new = jnp.einsum('sa,sa->s', policy, expected_values)
        
        return v_new
    
    def _policy_evaluation_step(
        self, 
        v: jnp.ndarray, 
        policy: jnp.ndarray,
        transition_probs: jnp.ndarray, 
        rewards: jnp.ndarray, 
        terminals: jnp.ndarray
    ) -> Tuple[jnp.ndarray, float]:
        """
        Realiza un paso de evaluación de política y calcula el delta.
        
        Parámetros:
        -----------
        v : jnp.ndarray
            Función de valor actual
        policy : jnp.ndarray
            Política a evaluar
        transition_probs : jnp.ndarray
            Matriz de probabilidades de transición
        rewards : jnp.ndarray
            Matriz de recompensas
        terminals : jnp.ndarray
            Matriz de indicadores de terminal
            
        Retorna:
        --------
        Tuple[jnp.ndarray, jnp.float64]
            Nueva función de valor y delta (cambio máximo)
        """
        v_new = self._calculate_state_values(policy, v, transition_probs, rewards, terminals)
        delta = jnp.max(jnp.abs(v_new - v))
        
        return v_new, delta
    
    def policy_evaluation(
        self, 
        env: Any, 
        policy: jnp.ndarray,
        use_jit: bool = True
    ) -> jnp.ndarray:
        """
        Evalúa la política actual calculando su función de valor.
        
        Parámetros:
        -----------
        env : Any
            Entorno con dinámicas de transición
        policy : jnp.ndarray
            Política a evaluar (distribución de probabilidad sobre acciones para cada estado)
        use_jit : bool, opcional
            Si usar compilación JIT (default: True)
            
        Retorna:
        --------
        jnp.ndarray
            Función de valor para la política dada
        """
        # Preparar matrices de transición
        transition_probs, rewards, terminals = self._prepare_transition_matrices(env)
        
        # Inicializar valor
        v = jnp.zeros(self.n_states)
        
        # Función de evaluación paso a paso
        eval_step = self._jit_policy_evaluation_step if use_jit else self._policy_evaluation_step
        
        # Iteración de valor para la política dada
        for i in range(self.max_iterations_eval):
            v, delta = eval_step(v, policy, transition_probs, rewards, terminals)
            
            # Verificar convergencia
            if delta < self.theta:
                break
        
        self.eval_iteration_counts.append(i + 1)
        return v
    
    def policy_improvement(
        self, 
        env: Any, 
        v: jnp.ndarray
    ) -> Tuple[jnp.ndarray, bool]:
        """
        Mejora la política haciéndola codiciosa respecto a la función de valor.
        
        Parámetros:
        -----------
        env : Any
            Entorno con dinámicas de transición
        v : jnp.ndarray
            Función de valor actual
            
        Retorna:
        --------
        Tuple[jnp.ndarray, bool]
            Tupla de (nueva política, política_estable)
        """
        # Preparar matrices de transición
        transition_probs, rewards, terminals = self._prepare_transition_matrices(env)
        
        # Convertir a arrays numpy para manipulación
        v_np = np.array(v)
        old_policy_np = np.array(self.policy)
        new_policy_np = np.zeros_like(old_policy_np)
        
        # Calcular todos los valores de acción
        action_values = np.zeros((self.n_states, self.n_actions))
        
        for s in range(self.n_states):
            for a in range(self.n_actions):
                if terminals[s, a]:
                    action_values[s, a] = rewards[s, a]
                else:
                    for next_s in range(self.n_states):
                        action_values[s, a] += transition_probs[s, a, next_s] * (
                            rewards[s, a] + self.gamma * v_np[next_s]
                        )
        
        # Determinar la mejor acción para cada estado
        best_actions = np.argmax(action_values, axis=1)
        
        # Crear nueva política
        for s in range(self.n_states):
            new_policy_np[s, best_actions[s]] = 1.0
        
        # Verificar estabilidad de la política
        policy_stable = np.all(np.argmax(old_policy_np, axis=1) == best_actions)
        
        return jnp.array(new_policy_np), policy_stable
    
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
                value_change = float(jnp.mean(jnp.abs(self.v - old_v)))
                self.value_changes.append(value_change)
            old_v = self.v
            
            # Mejora de Política: Actualizar política basada en nueva función de valor
            new_policy, policy_stable = self.policy_improvement(env, self.v)
            
            # Calcular cambio de política para métricas
            if iterations > 0:
                policy_change = float(jnp.sum(jnp.abs(new_policy - self.policy)) / (2 * self.n_states))
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
        
        # Preparar matrices de transición
        transition_probs, rewards, terminals = self._prepare_transition_matrices(env)
        
        policy_stable = False
        iterations = 0
        
        start_time = time.time()
        
        # Inicializar función de valor
        v = jnp.zeros(self.n_states)
        
        while not policy_stable and iterations < self.max_iterations:
            iteration_start = time.time()
            
            # Evaluación de política parcial: k iteraciones
            for _ in range(k_eval):
                v = self._calculate_state_values(self.policy, v, transition_probs, rewards, terminals)
            
            # Calcular cambio de valor para métricas
            if iterations > 0:
                value_change = float(jnp.mean(jnp.abs(v - old_v)))
                self.value_changes.append(value_change)
            old_v = v
            
            # Mejora de Política: Actualizar política basada en nueva función de valor
            new_policy, policy_stable = self.policy_improvement(env, v)
            
            # Calcular cambio de política para métricas
            if iterations > 0:
                policy_change = float(jnp.sum(jnp.abs(new_policy - self.policy)) / (2 * self.n_states))
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
        return int(jnp.argmax(self.policy[state]))
    
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
        return float(self.v[state])
    
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
        
        avg_reward = float(np.mean(total_rewards))
        avg_length = float(np.mean(episode_lengths))
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
        # Convertir arrays JAX a NumPy para serialización
        data = {
            'policy': np.array(self.policy),
            'v': np.array(self.v),
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
        
        # Convertir arrays NumPy a JAX
        self.policy = jnp.array(data['policy'])
        self.v = jnp.array(data['v'] if 'v' in data else data.get('V', np.zeros(data['n_states'])))
        self.n_states = data['n_states']
        self.n_actions = data['n_actions']
        self.gamma = data['gamma']
        
        print(f"Modelo cargado desde {filepath}")
    
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
        
        # Preparar matrices de transición
        transition_probs, rewards, terminals = self._prepare_transition_matrices(env)
        
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
        pi_values = np.array(self.v)
        
        comparison['policy_iteration']['time'] = pi_time
        comparison['policy_iteration']['iterations'] = pi_history['iterations']
        comparison['policy_iteration']['values'] = pi_values
        
        # Compilar función de valor-iteración para mayor velocidad
        @jax.jit
        def value_iteration_step(v: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
            # Calcular valores para todas las acciones en todos los estados
            expected_values = rewards + self.gamma * jnp.einsum(self.EINSUM_PATTERN, transition_probs, v)
            expected_values = jnp.where(terminals, rewards, expected_values)
            expected_values = jnp.where(terminals, rewards, expected_values)
            
            # Tomar el máximo valor para cada estado
            new_v = jnp.max(expected_values, axis=1)
            delta = jnp.max(jnp.abs(new_v - v))
            
            return new_v, delta
        
        # Ejecutar Iteración de Valor
        print("\n--- Iteración de Valor ---")
        vi_start = time.time()
        v = jnp.zeros(self.n_states)
        vi_iterations = 0
        
        for i in range(max_iterations):
            v, delta = value_iteration_step(v)
            vi_iterations = i + 1
            
            # Verificar convergencia
            if delta < self.theta:
                break
        
        # Calcular valores para todas las acciones
        expected_values = rewards + self.gamma * jnp.einsum(self.EINSUM_PATTERN, transition_probs, v)
        expected_values = jnp.where(terminals, rewards, expected_values)
        expected_values = jnp.where(terminals, rewards, expected_values)
        
        # La política es determinista, eligiendo la mejor acción
        best_actions = jnp.argmax(expected_values, axis=1)
        value_based_policy = jnp.zeros((self.n_states, self.n_actions))
        value_based_policy = value_based_policy.at[jnp.arange(self.n_states), best_actions].set(1.0)
        
        vi_time = time.time() - vi_start
        
        comparison['value_iteration']['time'] = vi_time
        comparison['value_iteration']['iterations'] = vi_iterations
        comparison['value_iteration']['values'] = np.array(v)
        
        # Calcular diferencia entre políticas
        pi_actions = np.argmax(np.array(self.policy), axis=1)
        vi_actions = np.array(best_actions)
        policy_difference = np.sum(pi_actions != vi_actions) / self.n_states
        value_difference = np.mean(np.abs(pi_values - np.array(v)))
        
        # Mostrar resultados
        print("\n--- Comparación ---")
        print(f"Iteración de Política: {pi_history['iterations']} iteraciones, {pi_time:.2f} segundos")
        print(f"Iteración de Valor: {vi_iterations} iteraciones, {vi_time:.2f} segundos")
        print(f"Diferencia en política: {policy_difference:.2%}")
        print(f"Diferencia en valores: {value_difference:.6f}")
        
        # Restaurar configuración original
        self.max_iterations = original_max_iterations
        
        return comparison
    
    def _get_grid_position(self, env: Any, state: int, grid_shape: Tuple[int, int]) -> Tuple[int, int]:
        """
        Obtiene la posición en la cuadrícula para un estado.
        
        Parámetros:
        -----------
        env : Any
            Entorno con estructura de cuadrícula
        state : int
            Estado a convertir
        grid_shape : Tuple[int, int]
            Forma de la cuadrícula
            
        Retorna:
        --------
        Tuple[int, int]
            Posición (fila, columna) en la cuadrícula
        """
        if hasattr(env, 'state_mapping'):
            return env.state_mapping(state)
        # Asumir orden row-major
        return state // grid_shape[1], state % grid_shape[1]
    
    def _is_terminal_state(self, env: Any, state: int) -> bool:
        """
        Verifica si un estado es terminal.
        
        Parámetros:
        -----------
        env : Any
            Entorno
        state : int
            Estado a verificar
            
        Retorna:
        --------
        bool
            True si el estado es terminal, False en caso contrario
        """
        for action in range(self.n_actions):
            for transitions in env.P[state][action]:
                if len(transitions) >= 3 and transitions[2]:  # done
                    return True
        return False
    
    def _setup_grid(self, ax: plt.Axes, grid_shape: Tuple[int, int]) -> None:
        """
        Configura la cuadrícula en los ejes.
        
        Parámetros:
        -----------
        ax : plt.Axes
            Ejes donde dibujar
        grid_shape : Tuple[int, int]
            Forma de la cuadrícula
        """
        # Crear cuadrícula
        ax.set_xlim([0, grid_shape[1]])
        ax.set_ylim([0, grid_shape[0]])
        
        # Dibujar líneas de cuadrícula
        for i in range(grid_shape[1] + 1):
            ax.axvline(i, color='black', linestyle='-')
        for j in range(grid_shape[0] + 1):
            ax.axhline(j, color='black', linestyle='-')
    
    def _draw_policy_arrows(self, ax: plt.Axes, env: Any, policy_np: np.ndarray, grid_shape: Tuple[int, int]) -> None:
        """
        Dibuja flechas representando la política.
        
        Parámetros:
        -----------
        ax : plt.Axes
            Ejes donde dibujar
        env : Any
            Entorno
        policy_np : np.ndarray
            Política en formato NumPy
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
            if self._is_terminal_state(env, s):
                continue
                
            i, j = self._get_grid_position(env, s, grid_shape)
            action = np.argmax(policy_np[s])
            
            if action in directions:
                dx, dy = directions[action]
                ax.arrow(j + 0.5, grid_shape[0] - i - 0.5, dx, dy, 
                        head_width=0.1, head_length=0.1, fc='blue', ec='blue')
    
    def _draw_state_values(self, ax: plt.Axes, env: Any, v_np: np.ndarray, grid_shape: Tuple[int, int]) -> None:
        """
        Dibuja los valores de los estados en la cuadrícula.
        
        Parámetros:
        -----------
        ax : plt.Axes
            Ejes donde dibujar
        env : Any
            Entorno
        v_np : np.ndarray
            Función de valor en formato NumPy
        grid_shape : Tuple[int, int]
            Forma de la cuadrícula
        """
        for s in range(self.n_states):
            i, j = self._get_grid_position(env, s, grid_shape)
            value = v_np[s]
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
        
        # Convertir a NumPy para visualización
        policy_np = np.array(self.policy)
        v_np = np.array(self.v)
        
        grid_shape = env.shape
        _, ax = plt.subplots(figsize=(8, 8))
        
        # Configurar y dibujar elementos
        self._setup_grid(ax, grid_shape)
        self._draw_policy_arrows(ax, env, policy_np, grid_shape)
        self._draw_state_values(ax, env, v_np, grid_shape)
        
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

class PolicyIterationWrapper:
    """
    Wrapper para hacer que el agente de Iteración de Política sea compatible con la interfaz de modelos de aprendizaje profundo.
    """
    
    def __init__(
        self, 
        pi_agent: PolicyIteration, 
        cgm_shape: Tuple[int, ...], 
        other_features_shape: Tuple[int, ...],
    ) -> None:
        """
        Inicializa el wrapper para Iteración de Política.
        
        Parámetros:
        -----------
        pi_agent : PolicyIteration
            Agente de Iteración de Política a utilizar
        cgm_shape : Tuple[int, ...]
            Forma de entrada para datos CGM
        other_features_shape : Tuple[int, ...]
            Forma de entrada para otras características
        """
        self.pi_agent = pi_agent
        self.cgm_shape = cgm_shape
        self.other_features_shape = other_features_shape
        
        # Para discretizar entradas continuas
        self.cgm_bins = 10
        self.other_bins = 5
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
        Realiza predicciones con el modelo de Iteración de Política.
        
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
            
            # Obtener acción según la política
            action = self.pi_agent.get_action(state)
            
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
            
        return min(state, self.pi_agent.n_states - 1)
    
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
        return action * 15.0 / (self.pi_agent.n_actions - 1)
    
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
        Entrena el modelo de Iteración de Política en los datos proporcionados.
        
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
        env = self._create_training_environment(cgm_data, other_features, y)
        
        if verbose > 0:
            print("Entrenando modelo de Iteración de Política...")
        
        # Entrenar agente con un número de iteraciones basado en epochs
        self.pi_agent.max_iterations = max(epochs, 10)
        _ = self.pi_agent.train(env)
        
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
        
        return self.history
    
    def _create_training_environment(
        self, 
        cgm_data: np.ndarray, 
        other_features: np.ndarray, 
        targets: np.ndarray
    ) -> Any:
        """
        Crea un entorno de entrenamiento compatible con el agente de Iteración de Política.
        
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
            Entorno simulado para RL
        """
        # Constantes para el entorno
        TRANSITION_PROB = 1.0
        
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
                
                # Definir espacios de acción y observación para compatibilidad con RL
                self.n_actions = model_wrapper.pi_agent.n_actions
                self.n_states = model_wrapper.pi_agent.n_states
                
                # Modelar las dinámicas de transición
                self.P = {}
                for s in range(self.n_states):
                    self.P[s] = {}
                    for a in range(self.n_actions):
                        # Para simplificar, cada acción lleva al terminal con recompensa basada en error
                        dose = model_wrapper._convert_action_to_dose(a)
                        self.P[s][a] = []
                        
                        # Calcular error promedio para esta acción en todos los estados
                        errors = []
                        for i in range(len(self.targets)):
                            state = model_wrapper._discretize_state(self.cgm[i], self.features[i])
                            if state == s:
                                target = self.targets[i]
                                error = -abs(dose - target)
                                errors.append(error)
                        
                        # Si no hay ejemplos de este estado, usar recompensa neutral
                        if not errors:
                            errors = [-5.0]  # Penalización por defecto
                            
                        avg_error = np.mean(errors)
                        
                        # Transición al terminar episodio con la recompensa promedio
                        self.P[s][a].append((TRANSITION_PROB, s, avg_error, True))
            
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
                state = self.model._discretize_state(
                    self.cgm[self.current_idx],
                    self.features[self.current_idx]
                )
                
                # Convertir acción a dosis
                dose = self.model._convert_action_to_dose(action)
                
                # Calcular recompensa
                target = self.targets[self.current_idx]
                reward = -abs(dose - target)
                
                # Avanzar al siguiente ejemplo
                self.current_idx = (self.current_idx + 1) % self.max_idx
                
                # Terminal después de cada paso
                done = True
                
                return state, reward, done, False, {}
        
        return InsulinDosingEnv(cgm_data, other_features, targets, self)
    
    def save(self, filepath: str) -> None:
        """
        Guarda el modelo en un archivo.
        
        Parámetros:
        -----------
        filepath : str
            Ruta donde guardar el modelo
        """
        self.pi_agent.save(filepath)
    
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
            'n_states': self.pi_agent.n_states,
            'n_actions': self.pi_agent.n_actions,
            'gamma': self.pi_agent.gamma,
            'theta': self.pi_agent.theta
        }


def create_policy_iteration_model(cgm_shape: Tuple[int, ...], other_features_shape: Tuple[int, ...]) -> PolicyIterationWrapper:
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
    PolicyIterationWrapper
        Wrapper de Iteración de Política que implementa la interfaz compatible con modelos de aprendizaje profundo
    """
    # Configurar el tamaño del espacio de estados y acciones
    # Esto es una simplificación - en un caso real habría que definirlo según los datos
    n_states = 1000  # Estado discretizado (más estados para mayor precisión)
    n_actions = 20   # Por ejemplo: 20 niveles discretos de dosis (0 a 15 unidades)
    
    # Crear agente de Iteración de Política
    pi_agent = PolicyIteration(
        n_states=n_states,
        n_actions=n_actions,
        gamma=POLICY_ITERATION_CONFIG['gamma'],
        theta=POLICY_ITERATION_CONFIG['theta'],
        max_iterations=POLICY_ITERATION_CONFIG['max_iterations'],
        max_iterations_eval=POLICY_ITERATION_CONFIG['max_iterations_eval'],
        seed=POLICY_ITERATION_CONFIG.get('seed', 42)
    )
    
    # Crear y devolver wrapper
    return PolicyIterationWrapper(pi_agent, cgm_shape, other_features_shape)