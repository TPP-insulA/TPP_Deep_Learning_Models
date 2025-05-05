import os, sys
import numpy as np
import matplotlib.pyplot as plt
import time
import jax
import jax.numpy as jnp
from typing import Dict, List, Tuple, Optional, Union, Any, NamedTuple, Callable
import pickle

PROJECT_ROOT = os.path.abspath(os.getcwd())
sys.path.append(PROJECT_ROOT) 

from constants.constants import CONST_DEFAULT_SEED, CONST_DEFAULT_EPOCHS, CONST_DEFAULT_BATCH_SIZE
from config.models_config import VALUE_ITERATION_CONFIG, EARLY_STOPPING_POLICY
from custom.rl_model_wrapper import RLModelWrapperJAX
from custom.printer import print_success, print_error

# Constantes para rutas de figuras y etiquetas comunes
CONST_FIGURES_DIR = "figures/reinforcement_learning/value_iteration"
CONST_ITERATION = "Iteración"
CONST_VALUE = "Valor"
CONST_DELTA = "Delta"
CONST_TIME = "Tiempo (segundos)"
CONST_POLICY = "Política"
CONST_PROBABILITY = 1.0  # Probabilidad de transición para modelo determinista
CONST_EINSUM_PATTERN = 'san,n->sa'  # Patrón para cálculos vectorizados


class ValueIterationState(NamedTuple):
    """Estado interno para la iteración de valor."""
    V: jnp.ndarray
    policy: jnp.ndarray
    value_changes: List[float]
    iteration_times: List[float]


class ValueIteration:
    """
    Implementación del algoritmo de Iteración de Valor (Value Iteration) con JAX.
    
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
        
        # Inicializar función de valor y política
        V = jnp.zeros(n_states)
        policy = jnp.zeros((n_states, n_actions))
        
        # Crear estado interno
        self.state = ValueIterationState(
            V=V,
            policy=policy,
            value_changes=[],
            iteration_times=[]
        )
        
        # NO compilar esta función, ya que usa diccionarios Python
        # self._calculate_action_values = jax.jit(self._calculate_action_values)

    def _calculate_action_values(
        self, 
        V: jnp.ndarray, 
        transitions: Dict[int, Dict[int, List]], 
        state: int
    ) -> jnp.ndarray:
        """
        Calcula los valores Q para todas las acciones en un estado.
        
        Parámetros:
        -----------
        V : jnp.ndarray
            Función de valor actual
        transitions : Dict[int, Dict[int, List]]
            Diccionario con las transiciones del entorno
        state : int
            Estado para calcular valores de acción
            
        Retorna:
        --------
        jnp.ndarray
            Valores Q para todas las acciones en el estado dado
        """
        action_values = jnp.zeros(self.n_actions)
        
        for a in range(self.n_actions):
            for prob, next_s, r, done in transitions[state][a]:
                # Valor esperado usando la ecuación de Bellman
                not_done = jnp.logical_not(done)
                action_values = action_values.at[a].add(
                    prob * (r + self.gamma * V[next_s] * not_done)
                )
        
        return action_values

    def _compute_action_value(self, env: Any, s: int, a: int, V_numpy: np.ndarray) -> float:
        """
        Calcula el valor Q para una acción en un estado determinado.
        
        Parámetros:
        -----------
        env : Any
            Entorno con dinámicas de transición
        s : int
            Índice del estado
        a : int
            Índice de la acción
        V_numpy : np.ndarray
            Función de valor actual
            
        Retorna:
        --------
        float
            Valor Q calculado
        """
        action_value = 0.0
        
        try:
            transitions = env.P.get(s, {}).get(a, [])
            for prob, next_s, r, done in transitions:
                not_done = 1.0 if not done else 0.0
                next_state_value = V_numpy[next_s] if next_s < len(V_numpy) else 0.0
                action_value += prob * (r + self.gamma * next_state_value * not_done)
        except Exception as e:
            print(f"Error en value_update, estado {s}, acción {a}: {e}")
            return 0.0  # Valor seguro en caso de error
            
        return action_value

    def _get_max_action_value(self, action_values: np.ndarray) -> float:
        """
        Obtiene el valor máximo de las acciones, manejando valores no finitos.
        
        Parámetros:
        -----------
        action_values : np.ndarray
            Array con valores Q para todas las acciones
            
        Retorna:
        --------
        float
            Valor máximo o valor predeterminado si no hay valores finitos
        """
        if np.any(np.isfinite(action_values)):
            return np.max(action_values)
        return 0.0  # Valor predeterminado seguro
    
    def value_update(self, env: Any) -> float:
        """
        Actualiza la función de valor usando la ecuación de Bellman.
        
        Parámetros:
        -----------
        env : Any
            Entorno con dinámicas de transición
            
        Retorna:
        --------
        float
            Delta máximo (diferencia máxima en valores)
        """
        # Convertir la matriz de valor a NumPy para mayor seguridad
        V_numpy = np.array(self.state.V)
        
        # Limitar estados a procesar para entornos grandes
        max_states_to_process = min(100, self.n_states)
        
        delta = 0.0
        new_values = np.copy(V_numpy)  # Hacer una copia explícita
        
        for s in range(max_states_to_process):
            v_old = V_numpy[s]
            
            # Calcular valores de acción manualmente
            action_values = np.zeros(self.n_actions)
            
            for a in range(self.n_actions):
                action_values[a] = self._compute_action_value(env, s, a, V_numpy)
            
            # Obtener el nuevo valor para el estado
            v_new = self._get_max_action_value(action_values)
            
            # Actualizar valor
            new_values[s] = v_new
            
            # Actualizar delta
            delta = max(delta, abs(v_old - v_new))
        
        # Actualizar función de valor, convirtiéndola a JAX array
        self.state = self.state._replace(V=jnp.array(new_values))
        
        return float(delta)

    def extract_policy(self, env: Any) -> jnp.ndarray:
        """
        Extrae la política óptima a partir de la función de valor.
        
        Parámetros:
        -----------
        env : Any
            Entorno con dinámicas de transición
            
        Retorna:
        --------
        jnp.ndarray
            Política óptima (determinística)
        """
        policy = np.zeros((self.n_states, self.n_actions))
        
        for s in range(self.n_states):
            # Calcular el valor Q para cada acción
            action_values = np.zeros(self.n_actions)
            
            # Calcular valores directamente sin usar _calculate_action_values
            for a in range(self.n_actions):
                # Usar try/except para manejar estados que pueden no estar en el diccionario
                try:
                    for prob, next_s, r, done in env.P.get(s, {}).get(a, []):
                        # Valor esperado usando la ecuación de Bellman
                        not_done = not done
                        action_values[a] += prob * (r + self.gamma * float(self.state.V[next_s]) * not_done)
                except (KeyError, TypeError):
                    # En caso de error, asignar un valor bajo
                    action_values[a] = -float('inf')
            
            # Política determinística: asignar probabilidad 1.0 a la mejor acción
            if np.all(np.isneginf(action_values)):
                # Si todos los valores son -inf, elegir acción aleatoria
                best_action = 0  # Acción predeterminada segura
            else:
                best_action = np.argmax(action_values)
            
            policy[s, best_action] = 1.0
        
        return jnp.array(policy)  # Convertir a array JAX al final

    def _check_early_stopping(self, delta: float, best_delta: float, no_improvement_count: int, 
                             patience: int, min_delta: float) -> Tuple[float, int, bool]:
        """
        Verifica si el early stopping debe activarse.
        
        Retorna:
        --------
        Tuple[float, int, bool]
            Nueva mejor delta, nuevo contador de no mejora, y bandera si debe detenerse
        """
        should_stop = False
        new_best_delta = best_delta
        new_count = no_improvement_count
        
        if delta < best_delta - min_delta:
            new_best_delta = delta
            new_count = 0
        else:
            new_count += 1
            
        if new_count >= patience:
            should_stop = True
            
        return new_best_delta, new_count, should_stop
    
    def _check_convergence(self, delta: float, start_time: float) -> bool:
        """
        Verifica si se ha alcanzado la convergencia o tiempo límite.
        
        Retorna:
        --------
        bool
            True si debe detenerse, False en caso contrario
        """
        # Verificar convergencia con un umbral razonable
        if delta < max(self.theta, 0.01):
            print_success(f"¡Convergencia alcanzada (delta < {max(self.theta, 0.01)})!")
            return True
            
        # Agregar límite de tiempo para evitar bucles
        if time.time() - start_time > 30:  # 30 segundos máximo
            print_error("¡Tiempo máximo de entrenamiento alcanzado!")
            return True
            
        return False
    
    def train(self, env: Any) -> Dict[str, List]:
        """
        Entrena al agente usando iteración de valor.
        """
        print("Iniciando iteración de valor con espacio de estados reducido...")
        
        iterations = 0
        start_time = time.time()
        value_changes = []
        iteration_times = []
        
        # Usar el valor de paciencia del early stopping si está definido
        patience = EARLY_STOPPING_POLICY.get('early_stopping_patience', 5)
        min_delta = EARLY_STOPPING_POLICY.get('early_stopping_min_delta', 0.001)
        use_early_stopping = EARLY_STOPPING_POLICY.get('early_stopping', True)
        
        # Contador para early stopping
        no_improvement_count = 0
        best_delta = float('inf')
        
        # Definir límite de iteraciones reducido para entrenamiento inicial
        actual_max_iterations = min(self.max_iterations, 50)
        print(f"Limitando a {actual_max_iterations} iteraciones máximas")
        
        for i in range(actual_max_iterations):
            iteration_start = time.time()
            
            # Actualizar función de valor
            delta = self.value_update(env)
            
            # Registrar cambio de valor y tiempo
            value_changes.append(float(delta))
            iteration_time = time.time() - iteration_start
            iteration_times.append(iteration_time)
            
            iterations = i + 1
            
            # Mostrar progreso cada 10 iteraciones o al principio
            if i % 10 == 0 or i < 5:
                print(f"Iteración {iterations}: Delta = {delta:.6f}, Tiempo = {iteration_time:.2f} segundos")
            
            # Comprobar early stopping si está habilitado
            if use_early_stopping:
                best_delta, no_improvement_count, should_stop = self._check_early_stopping(
                    delta, best_delta, no_improvement_count, patience, min_delta)
                
                if should_stop:
                    print_success(f"¡Early stopping activado después de {iterations} iteraciones!")
                    break
            
            # Verificar convergencia o tiempo límite
            if self._check_convergence(delta, start_time):
                break
        
        # Extraer política óptima y actualizar estado
        policy = self.extract_policy(env)
        self.state = self.state._replace(
            policy=policy,
            value_changes=value_changes,
            iteration_times=iteration_times
        )
        
        total_time = time.time() - start_time
        print(f"Iteración de valor completada en {iterations} iteraciones, {total_time:.2f} segundos")
        
        return {
            'iterations': iterations,
            'value_changes': value_changes,
            'iteration_times': iteration_times,
            'total_time': total_time
        }

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
        return int(jnp.argmax(self.state.policy[state]))

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
        return float(self.state.V[state])

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
        
        for ep in range(episodes):
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
            print(f"Episodio {ep+1}: Recompensa = {total_reward}, Pasos = {steps}")
        
        avg_reward = np.mean(total_rewards)
        print(f"Evaluación: recompensa media en {episodes} episodios = {avg_reward:.2f}")
        
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
            'policy': np.array(self.state.policy),
            'V': np.array(self.state.V),
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
        
        # Actualizar estado
        self.state = self.state._replace(
            V=jnp.array(data['V']),
            policy=jnp.array(data['policy'])
        )
        self.n_states = data['n_states']
        self.n_actions = data['n_actions']
        self.gamma = data['gamma']
        
        print(f"Modelo cargado desde {filepath}")

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
            print("El entorno no tiene estructura de cuadrícula para visualización")
            return
        
        # Crear directorio para figuras si no existe
        os.makedirs(CONST_FIGURES_DIR, exist_ok=True)
        
        grid_shape = env.shape
        _, ax = plt.subplots(figsize=(8, 8))
        
        # Crear cuadrícula
        ax.set_xlim([0, grid_shape[1]])
        ax.set_ylim([0, grid_shape[0]])
        
        # Dibujar líneas de cuadrícula
        for i in range(grid_shape[1] + 1):
            ax.axvline(i, color='black', linestyle='-')
        for j in range(grid_shape[0] + 1):
            ax.axhline(j, color='black', linestyle='-')
        
        # Dibujar flechas para las acciones
        for s in range(self.n_states):
            if hasattr(env, 'state_mapping'):
                # Convertir índice de estado a posición en cuadrícula
                i, j = env.state_mapping(s)
            else:
                # Asumir orden row-major
                i, j = s // grid_shape[1], s % grid_shape[1]
            
            # Omitir estados terminales
            if any(info[2] for a in range(self.n_actions) for _, _, info, _ in env.P[s][a]):
                continue
                
            action = self.get_action(s)
            
            # Definir direcciones de flechas
            directions = {
                0: (0, -0.4),  # Izquierda
                1: (0, 0.4),   # Derecha
                2: (-0.4, 0),  # Abajo
                3: (0.4, 0)    # Arriba
            }
            
            if action in directions:
                dx, dy = directions[action]
                ax.arrow(j + 0.5, grid_shape[0] - i - 0.5, dx, dy, 
                         head_width=0.1, head_length=0.1, fc='blue', ec='blue')
        
        # Mostrar valores de estados
        for s in range(self.n_states):
            if hasattr(env, 'state_mapping'):
                i, j = env.state_mapping(s)
            else:
                i, j = s // grid_shape[1], s % grid_shape[1]
            
            value = self.get_value(s)
            ax.text(j + 0.5, grid_shape[0] - i - 0.5, f"{value:.2f}", 
                   ha='center', va='center', color='red', fontsize=9)
        
        ax.set_title(title)
        plt.tight_layout()
        
        # Guardar figura
        plt.savefig(os.path.join(CONST_FIGURES_DIR, f"politica_{title.lower().replace(' ', '_')}.png"), dpi=300)
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
            print("El entorno no tiene estructura de cuadrícula para visualización")
            return
        
        # Crear directorio para figuras si no existe
        os.makedirs(CONST_FIGURES_DIR, exist_ok=True)
        
        grid_shape = env.shape
        
        # Crear matriz para visualización
        value_grid = np.zeros(grid_shape)
        
        # Llenar matriz con valores
        for s in range(self.n_states):
            if hasattr(env, 'state_mapping'):
                i, j = env.state_mapping(s)
            else:
                i, j = s // grid_shape[1], s % grid_shape[1]
                
            value_grid[i, j] = self.get_value(s)
        
        _, ax = plt.subplots(figsize=(10, 8))
        
        # Crear mapa de calor
        im = ax.imshow(value_grid, cmap='viridis')
        
        # Añadir barra de color
        plt.colorbar(im, ax=ax, label='Valor')
        
        # Mostrar valores en cada celda
        for i in range(grid_shape[0]):
            for j in range(grid_shape[1]):
                ax.text(j, i, f"{value_grid[i, j]:.2f}", ha='center', va='center',
                        color='white' if value_grid[i, j] < np.max(value_grid)/1.5 else 'black')
        
        ax.set_title(title)
        plt.tight_layout()
        
        # Guardar figura
        plt.savefig(os.path.join(CONST_FIGURES_DIR, f"funcion_valor_{title.lower().replace(' ', '_')}.png"), dpi=300)
        plt.show()

    def visualize_training(self, history: Dict[str, List]) -> None:
        """
        Visualiza las métricas de entrenamiento.
        
        Parámetros:
        -----------
        history : Dict[str, List]
            Diccionario con historial de entrenamiento
        """
        # Crear directorio para figuras si no existe
        os.makedirs(CONST_FIGURES_DIR, exist_ok=True)
        
        _, axs = plt.subplots(2, 1, figsize=(12, 8))
        
        # Gráfico de cambios en la función de valor (delta)
        axs[0].plot(range(1, len(history['value_changes']) + 1), 
                    history['value_changes'])
        axs[0].set_title(f'Cambios en la Función de {CONST_VALUE} ({CONST_DELTA})')
        axs[0].set_xlabel(CONST_ITERATION)
        axs[0].set_ylabel(CONST_DELTA)
        axs[0].set_yscale('log')  # Escala logarítmica para ver mejor la convergencia
        axs[0].grid(True)
        
        # Gráfico de tiempos de iteración
        axs[1].plot(range(1, len(history['iteration_times']) + 1), 
                    history['iteration_times'])
        axs[1].set_title(f'Tiempos de {CONST_ITERATION}')
        axs[1].set_xlabel(CONST_ITERATION)
        axs[1].set_ylabel(CONST_TIME)
        axs[1].grid(True)
        
        plt.tight_layout()
        
        # Guardar figura
        plt.savefig(os.path.join(CONST_FIGURES_DIR, "entrenamiento_resumen.png"), dpi=300)
        plt.show()

    def parallel_value_iteration(self, env: Any) -> Dict[str, List]:
        """
        Implementa iteración de valor con cálculos paralelizados usando JAX.
        
        Parámetros:
        -----------
        env : Any
            Entorno con dinámicas de transición
            
        Retorna:
        --------
        Dict[str, List]
            Diccionario con historial de entrenamiento
        """
        print("Iniciando iteración de valor paralela...")
        
        iterations = 0
        start_time = time.time()
        value_changes = []
        iteration_times = []
        
        # Preparar las funciones para cálculo en paralelo
        value_fn = jax.vmap(
            lambda s, V: self._calculate_action_values(V, env.P, s),
            in_axes=(0, None)
        )
        
        for i in range(self.max_iterations):
            iteration_start = time.time()
            
            # Calcular valores de acción para todos los estados en paralelo
            states = jnp.arange(self.n_states)
            all_action_values = value_fn(states, self.state.V)
            
            # Actualizar función de valor
            new_V = jnp.max(all_action_values, axis=1)
            
            # Calcular delta
            delta = jnp.max(jnp.abs(new_V - self.state.V))
            
            # Actualizar estado
            _ = self.state.V
            self.state = self.state._replace(V=new_V)
            
            # Registrar métricas
            value_changes.append(float(delta))
            iteration_time = time.time() - iteration_start
            iteration_times.append(iteration_time)
            
            iterations = i + 1
            
            print(f"Iteración {iterations}: Delta = {float(delta):.6f}, "
                  f"Tiempo = {iteration_time:.2f} segundos")
            
            # Verificar convergencia
            if delta < self.theta:
                print("¡Convergencia alcanzada!")
                break
        
        # Extraer política óptima
        policy = self.extract_policy(env)
        
        # Actualizar estado final
        self.state = self.state._replace(
            policy=policy,
            value_changes=value_changes,
            iteration_times=iteration_times
        )
        
        total_time = time.time() - start_time
        print(f"Iteración de valor paralela completada en {iterations} iteraciones, "
              f"{total_time:.2f} segundos")
        
        history = {
            'iterations': iterations,
            'value_changes': value_changes,
            'iteration_times': iteration_times,
            'total_time': total_time
        }
        
        return history

    def _build_transition_dynamics(self):
        """Construye el modelo de transición con eficiencia de memoria."""
        print("Construyendo modelo de transición eficiente...")
        
        # Diccionario minimalista que solo carga estados necesarios
        transitions = {}
        
        # Solo precalcular transiciones para unos pocos estados iniciales
        for s in range(min(10, self.model.vi_agent.n_states)):
            transitions[s] = {}
            for a in range(self.model.vi_agent.n_actions):
                # Simplificar cada transición
                transitions[s][a] = [(1.0, s, -1.0, True)]  # Probabilidad, estado, recompensa, terminal
        
        return transitions

class ValueIterationWrapper:
    """
    Wrapper para hacer que el agente de Iteración de Valor sea compatible con la interfaz de modelos de aprendizaje profundo.
    """
    
    def __init__(
        self, 
        vi_agent: ValueIteration, 
        cgm_shape: Tuple[int, ...], 
        other_features_shape: Tuple[int, ...],
    ) -> None:
        """
        Inicializa el wrapper para Iteración de Valor.
        
        Parámetros:
        -----------
        vi_agent : ValueIteration
            Agente de Iteración de Valor a utilizar
        cgm_shape : Tuple[int, ...]
            Forma de entrada para datos CGM
        other_features_shape : Tuple[int, ...]
            Forma de entrada para otras características
        """
        self.vi_agent = vi_agent
        self.cgm_shape = cgm_shape
        self.other_features_shape = other_features_shape
        
        # Para discretizar entradas continuas
        self.cgm_bins = 8
        self.other_bins = 4
        
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
        Realiza predicciones con el modelo de Iteración de Valor.
        
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
            
            # Obtener acción según la política óptima
            action = self.vi_agent.get_action(state)
            
            # Convertir acción discreta a dosis continua
            predictions[i, 0] = self._convert_action_to_dose(action)
        
        return predictions
    
    def _discretize_state(self, cgm_data: np.ndarray, other_features: np.ndarray) -> int:
        """
        Discretiza las entradas continuas a un índice de estado.
        """
        # Simplificar extracción de características relevantes
        cgm_flat = cgm_data.flatten()
        
        # Usar solo características esenciales
        cgm_last = cgm_flat[-1] if len(cgm_flat) > 0 else 0
        cgm_slope = cgm_flat[-1] - cgm_flat[0] if len(cgm_flat) > 1 else 0
        
        # Discretizar solo características esenciales
        cgm_last_bin = min(int(cgm_last / 400 * self.cgm_bins), self.cgm_bins - 1)
        cgm_slope_bin = min(int((cgm_slope + 200) / 400 * self.cgm_bins), self.cgm_bins - 1)
        
        # Solo usar carbohidratos e insulina a bordo
        carb_input = other_features[0] if len(other_features) > 0 else 0
        iob = other_features[2] if len(other_features) > 2 else 0
        
        # Discretizar
        carb_bin = min(int(carb_input / 100 * self.other_bins), self.other_bins - 1)
        iob_bin = min(int(iob / 10 * self.other_bins), self.other_bins - 1)
        
        # Combinar para formar estado (con menos dimensiones)
        state_idx = 0
        state_idx = state_idx * self.cgm_bins + cgm_last_bin
        state_idx = state_idx * self.cgm_bins + cgm_slope_bin
        state_idx = state_idx * self.other_bins + carb_bin
        state_idx = state_idx * self.other_bins + iob_bin
        
        return min(state_idx, self.vi_agent.n_states - 1)
    
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
        return action * 15.0 / (self.vi_agent.n_actions - 1)
    
    def fit(
        self, 
        x: List[np.ndarray], 
        y: np.ndarray, 
        validation_data: Optional[Tuple] = None, 
        epochs: int = CONST_DEFAULT_EPOCHS,
        batch_size: int = CONST_DEFAULT_BATCH_SIZE,
        callbacks: List = None,
        verbose: int = 0
    ) -> Dict:
        """
        Entrena el modelo de Iteración de Valor en los datos proporcionados.
        
        Parámetros:
        -----------
        x : List[np.ndarray]
            Lista con [cgm_data, other_features]
        y : np.ndarray
            Etiquetas (dosis objetivo)
        validation_data : Optional[Tuple], opcional
            Datos de validación (default: None)
        epochs : int, opcional
            Número de épocas (default: 10)
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
        
        # Crear entorno de entrenamiento que modeliza las dinámicas del problema
        env = self._create_training_environment(cgm_data, other_features, y)
        
        if verbose > 0:
            print("Entrenando modelo de Iteración de Valor...")
        
        # Configurar máximo de iteraciones basado en epochs
        self.vi_agent.max_iterations = max(epochs * 10, self.vi_agent.max_iterations)
        
        # Entrenar agente
        vi_history = self.vi_agent.train(env)
        
        # Calcular pérdida en datos de entrenamiento
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
        
        # Combinar historiales
        combined_history = {
            'loss': self.history['loss'],
            'iterations': vi_history['iterations'],
            'value_changes': vi_history['value_changes'],
            'iteration_times': vi_history['iteration_times'],
            'total_time': vi_history['total_time']
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
        Crea un entorno de entrenamiento compatible con el agente de Iteración de Valor.
        
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
        from types import SimpleNamespace
        
        # Crear e inicializar el entorno con los datos proporcionados
        env = self._create_env_class()
        return env(cgm_data, other_features, targets, self)
    
    def _create_env_class(self):
        """Crea y devuelve la clase del entorno de dosificación de insulina."""
        class InsulinDosingEnv:
            """Entorno personalizado para problema de dosificación de insulina."""
            
            def __init__(self, cgm, features, targets, model_wrapper):
                self.cgm = cgm
                self.features = features
                self.targets = targets
                self.model = model_wrapper
                self.rng = np.random.Generator(np.random.PCG64(CONST_DEFAULT_SEED))
                self.current_idx = 0
                self.max_idx = len(targets) - 1
                
                # Modelar las dinámicas de transición necesarias para Value Iteration
                self.P = self._build_transition_dynamics()
                
                # Inicializar espacios para compatibilidad con algoritmos RL
                self._init_spaces()
            
            def _init_spaces(self):
                """Inicializa los espacios de observación y acción."""
                from types import SimpleNamespace
                
                self.observation_space = SimpleNamespace(
                    shape=(1,),
                    low=0,
                    high=self.model.vi_agent.n_states - 1
                )
                
                self.action_space = SimpleNamespace(
                    n=self.model.vi_agent.n_actions,
                    sample=self._sample_action
                )
                
                # Agregar shape para visualización de la política
                self.shape = (
                    self.model.cgm_bins**2, 
                    self.model.cgm_bins**2 * self.model.other_bins
                )
            
            def _sample_action(self):
                """Muestrea una acción aleatoria del espacio discreto."""
                return self.rng.integers(0, self.model.vi_agent.n_actions)
            
            def _get_reward_for_state_action(self, s, a, dose):
                """Calcula la recompensa para un estado y acción dados."""
                rewards = []
                
                # Tomar muestras aleatorias para estimar recompensas
                sample_indices = self.rng.integers(0, self.max_idx, 10)
                for idx in sample_indices:
                    state_idx = self.model._discretize_state(
                        self.cgm[idx], self.features[idx]
                    )
                    if state_idx == s:
                        target = self.targets[idx]
                        # Recompensa como negativo del error absoluto
                        reward = -abs(dose - target)
                        rewards.append(reward)
                
                # Si no hay ejemplos relevantes, usar una estimación
                if not rewards:
                    # Penalización por defecto más alta para acciones extremas
                    default_reward = -5.0
                    if a == 0 or a == self.model.vi_agent.n_actions - 1:
                        default_reward = -10.0
                    rewards = [default_reward]
                
                return float(np.mean(rewards))
            
            def _build_transition_dynamics(self):
                """Construye el modelo de dinámicas de transición para Value Iteration."""
                return self._build_transition_batches()
            
            def _build_transition_batches(self):
                """Construye las transiciones en lotes para optimizar memoria."""
                P = {}
                n_states = self.model.vi_agent.n_states
                _ = self.model.vi_agent.n_actions
                
                # Cantidad de estados a procesar por iteración
                batch_size = 1000
                
                # Construir modelo de transiciones por lotes
                for state_batch_start in range(0, n_states, batch_size):
                    state_batch_end = min(state_batch_start + batch_size, n_states)
                    self._build_transitions_for_batch(P, state_batch_start, state_batch_end)
                
                return P
            
            def _build_transitions_for_batch(self, P, start_state, end_state):
                """Construye transiciones para un lote de estados."""
                n_actions = self.model.vi_agent.n_actions
                
                for s in range(start_state, end_state):
                    P[s] = {}
                    for a in range(n_actions):
                        P[s][a] = []
                        
                        # Calcular dosis para esta acción
                        dose = self.model._convert_action_to_dose(a)
                        
                        # Calcular recompensa para esta acción en este estado
                        avg_reward = self._get_reward_for_state_action(s, a, dose)
                        
                        # Para Value Iteration, asumimos estado terminal después de cada acción
                        P[s][a].append((CONST_PROBABILITY, s, avg_reward, True))
                
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
                
                # En este caso, consideramos episodios de un solo paso
                done = True
                truncated = False
                
                return next_state, reward, done, truncated, {}
            
            def state_mapping(self, state_idx):
                """Convierte un índice de estado a coordenadas para visualización."""
                total_bins = self.model.cgm_bins**4 * self.model.other_bins**3
                relative_idx = state_idx / total_bins
                
                # Convertir a coordenadas 2D aproximadas para visualización
                grid_size = self.shape
                i = int(relative_idx * grid_size[0])
                j = int((relative_idx * total_bins) % grid_size[1])
                
                return i, j
        
        return InsulinDosingEnv
    
    def save(self, filepath: str) -> None:
        """
        Guarda el modelo en un archivo.
        
        Parámetros:
        -----------
        filepath : str
            Ruta donde guardar el modelo
        """
        # Crear directorio si no existe
        import os
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        # Guardar agente principal
        self.vi_agent.save(filepath + "_vi_agent.pkl")
        
        # Guardar configuración del wrapper
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
        # Cargar agente principal
        self.vi_agent.load(filepath + "_vi_agent.pkl")
        
        # Cargar configuración del wrapper
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
            'n_states': self.vi_agent.n_states,
            'n_actions': self.vi_agent.n_actions,
            'gamma': self.vi_agent.gamma,
            'theta': self.vi_agent.theta,
            'cgm_bins': self.cgm_bins,
            'other_bins': self.other_bins
        }

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
        return self

    def _map_observation_to_state(self, cgm_obs: np.ndarray, other_obs: np.ndarray) -> int:
        """
        Mapea una observación continua a un índice de estado discreto.
        
        Parámetros:
        -----------
        cgm_obs : np.ndarray
            Datos CGM para una muestra.
        other_obs : np.ndarray
            Otras características para una muestra.
            
        Retorna:
        --------
        int
            Índice del estado discretizado.
        """
        # Reutilizar el método existente de discretización
        return self._discretize_state(cgm_obs, other_obs)

    def train_batch(self, agent_state: Any, batch_data: Tuple, rng_key: jax.random.PRNGKey) -> Tuple[Any, Dict[str, float]]:
        """
        Entrena el agente con un lote de datos.
        
        Parámetros:
        -----------
        agent_state : Any
            Estado actual del agente (la propia instancia).
        batch_data : Tuple
            Tupla conteniendo ((observaciones_cgm, observaciones_other), targets).
        rng_key : jax.random.PRNGKey
            Clave para generación aleatoria.
            
        Retorna:
        --------
        Tuple[Any, Dict[str, float]]
            Tupla con (nuevo estado del agente, métricas de entrenamiento).
        """
        # Extraer datos del lote
        (observations_cgm, observations_other), targets = batch_data
        
        # Convertir a arrays de NumPy si son arrays de JAX
        obs_cgm_np = np.array(observations_cgm)
        obs_other_np = np.array(observations_other)
        targets_np = np.array(targets)
        
        # Solo entrenar realmente en el primer lote de la época
        # Para los lotes subsiguientes, solo aplicar la política ya aprendida
        if not hasattr(self, '_trained_once') or not self._trained_once:
            # Crear entorno de entrenamiento solo la primera vez
            env = self._create_training_environment(obs_cgm_np, obs_other_np, targets_np)
            
            # Entrenar el agente de VI
            history = self.vi_agent.train(env)
            
            # Marcar que ya hemos entrenado una vez
            self._trained_once = True
            
            # Registrar métricas de entrenamiento
            metrics = {
                'loss': float(history.get('value_changes', [0.0])[-1] if history.get('value_changes') else 0.0),
                'iterations': history.get('iterations', 0),
                'total_time': history.get('total_time', 0.0)
            }
        else:
            # Para lotes posteriores, saltamos el entrenamiento y reportamos métricas fijas
            metrics = {
                'loss': 0.0,
                'iterations': 0,
                'total_time': 0.0,
                'skipped': True  # Indicador de que se saltó el entrenamiento
            }
        
        return self, metrics

    def predict_batch(self, agent_state: Any, observations: Tuple[jnp.ndarray, jnp.ndarray], rng_key: jax.random.PRNGKey) -> jnp.ndarray:
        """
        Realiza predicciones para un lote de observaciones.
        
        Parámetros:
        -----------
        agent_state : Any
            Estado actual del agente (la propia instancia).
        observations : Tuple[jnp.ndarray, jnp.ndarray]
            Tupla con (observaciones_cgm, observaciones_other).
        rng_key : jax.random.PRNGKey
            Clave para generación aleatoria.
            
        Retorna:
        --------
        jnp.ndarray
            Predicciones para el lote.
        """
        x_cgm, x_other = observations
        batch_size = x_cgm.shape[0]
        
        # Convertir a arrays de NumPy para procesamiento
        x_cgm_np = np.array(x_cgm)
        x_other_np = np.array(x_other)
        
        # Inicializar array para predicciones
        predictions = np.zeros((batch_size, 1), dtype=np.float32)
        
        # Realizar predicciones para cada muestra
        for i in range(batch_size):
            # Discretizar estado
            state_idx = self._map_observation_to_state(x_cgm_np[i], x_other_np[i])
            
            # Obtener acción según la política actual
            action = self.vi_agent.get_action(state_idx)
            
            # Convertir acción discreta a dosis continua
            dose = self._convert_action_to_dose(action)
            
            # Almacenar predicción
            predictions[i, 0] = dose
        
        # Convertir a array de JAX para interfaz consistente
        return jnp.array(predictions)

    def evaluate(self, agent_state: Any, batch_data: Tuple, rng_key: jax.random.PRNGKey) -> Dict[str, float]:
        """
        Evalúa el rendimiento del agente en un conjunto de datos.
        
        Parámetros:
        -----------
        agent_state : Any
            Estado actual del agente (la propia instancia).
        batch_data : Tuple
            Tupla conteniendo ((observaciones_cgm, observaciones_other), targets).
        rng_key : jax.random.PRNGKey
            Clave para generación aleatoria.
            
        Retorna:
        --------
        Dict[str, float]
            Diccionario con métricas de evaluación.
        """
        (observations_cgm, observations_other), targets = batch_data
        
        # Realizar predicciones
        predictions = self.predict_batch(agent_state, (observations_cgm, observations_other), rng_key)
        
        # Convertir a NumPy para cálculos
        preds_np = np.array(predictions).flatten()
        targets_np = np.array(targets).flatten()
        
        # Calcular métricas
        mse = np.mean((preds_np - targets_np) ** 2)
        mae = np.mean(np.abs(preds_np - targets_np))
        
        # Si hay suficientes muestras para calcular R²
        if len(targets_np) > 1:
            ss_tot = np.sum((targets_np - np.mean(targets_np)) ** 2)
            ss_res = np.sum((targets_np - preds_np) ** 2)
            r2 = 1 - (ss_res / (ss_tot + 1e-10))  # Evitar división por cero
        else:
            r2 = 0.0
        
        return {'loss': float(mse), 'mae': float(mae), 'r2': float(r2)}

def create_value_iteration_agent(cgm_shape: Tuple[int, ...], other_features_shape: Tuple[int, ...], **kwargs) -> ValueIterationWrapper:
    """
    Crea un agente de Iteración de Valor para el problema de dosificación de insulina.
    """
    # Reducir drásticamente el número de bins y features
    cgm_bins = kwargs.get('cgm_bins', 2)  # Reducción a solo 2 bins
    other_bins = kwargs.get('other_bins', 2)  # Reducción a solo 2 bins
    
    # Solo las features absolutamente esenciales
    cgm_features = 1  # Solo último valor de CGM
    other_features = 1  # Solo carbohidratos
    
    # Esto da un espacio manejable: 2^1 * 2^1 = 4 estados
    n_states = cgm_bins**cgm_features * other_bins**other_features
    
    print(f"Espacio de estados redimensionado a {n_states} estados discretos")
    
    # Para dosificación de insulina, discretizamos en niveles de dosis
    n_actions = kwargs.get('n_actions', 20)  # 20 niveles discretos (0 a 15 unidades)
    
    # Crear agente de Iteración de Valor con configuración óptima
    vi_agent = ValueIteration(
        n_states=n_states,
        n_actions=n_actions,
        gamma=kwargs.get('gamma', VALUE_ITERATION_CONFIG['gamma']),
        theta=kwargs.get('theta', VALUE_ITERATION_CONFIG['theta']),
        max_iterations=kwargs.get('max_iterations', VALUE_ITERATION_CONFIG['max_iterations'])
    )
    
    # Crear wrapper con bins configurables
    wrapper = ValueIterationWrapper(
        vi_agent=vi_agent,
        cgm_shape=cgm_shape,
        other_features_shape=other_features_shape
    )
    wrapper.cgm_bins = cgm_bins
    wrapper.other_bins = other_bins
    
    return wrapper

def create_value_iteration_model(cgm_shape: Tuple[int, ...], other_features_shape: Tuple[int, ...], **kwargs) -> RLModelWrapperJAX:
    """
    Crea un modelo basado en Iteración de Valor para predicción de dosis de insulina.
    
    Parámetros:
    -----------
    cgm_shape : Tuple[int, ...]
        Forma de los datos CGM (batch_size, time_steps, features)
    other_features_shape : Tuple[int, ...]
        Forma de otras características (batch_size, n_features)
    **kwargs
        Argumentos adicionales para el agente
        
    Retorna:
    --------
    RLModelWrapperJAX
        Wrapper de Iteración de Valor que implementa la interfaz compatible con la API del sistema
    """
    # Devolver el wrapper con el agente de Iteración de Valor
    model = RLModelWrapperJAX(
        agent_creator=create_value_iteration_agent,
        cgm_shape=cgm_shape,
        other_features_shape=other_features_shape,
        **kwargs
    )

    # Configurar early stopping
    patience = EARLY_STOPPING_POLICY.get('patience', 10)
    min_delta = EARLY_STOPPING_POLICY.get('min_delta', 0.01)
    restore_best_weights = EARLY_STOPPING_POLICY.get('restore_best_weights', True)
    model.add_early_stopping(patience=patience, min_delta=min_delta, restore_best_weights=restore_best_weights)
    
    return model

def model_creator() -> Callable[[Tuple[int, ...], Tuple[int, ...]], RLModelWrapperJAX]:
    """
    Retorna una función para crear un modelo de Iteración de Valor compatible con la API del sistema.
    
    Retorna:
    --------
    Callable[[Tuple[int, ...], Tuple[int, ...]], RLModelWrapperJAX]
        Función para crear el modelo con las formas de entrada especificadas
    """
    return create_value_iteration_model