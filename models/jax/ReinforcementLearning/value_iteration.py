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

from config.models_config import VALUE_ITERATION_CONFIG

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
        
        # Compilar funciones puras para mejor rendimiento
        self._calculate_action_values = jax.jit(self._calculate_action_values)

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
        delta = 0.0
        new_V = self.state.V
        
        for s in range(self.n_states):
            v_old = self.state.V[s]
            
            # Calcular el valor Q para cada acción y tomar el máximo
            action_values = self._calculate_action_values(self.state.V, env.P, s)
            
            # Actualizar el valor del estado con el máximo valor de acción
            new_V = new_V.at[s].set(jnp.max(action_values))
            
            # Actualizar delta
            delta = max(delta, abs(v_old - new_V[s]))
        
        # Actualizar estado
        self.state = self.state._replace(V=new_V)
        
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
        policy = jnp.zeros((self.n_states, self.n_actions))
        
        for s in range(self.n_states):
            # Calcular el valor Q para cada acción
            action_values = self._calculate_action_values(self.state.V, env.P, s)
            
            # Política determinística: asignar probabilidad 1.0 a la mejor acción
            best_action = jnp.argmax(action_values)
            policy = policy.at[s, best_action].set(1.0)
        
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
        value_changes = []
        iteration_times = []
        
        for i in range(self.max_iterations):
            iteration_start = time.time()
            
            # Actualizar función de valor
            delta = self.value_update(env)
            
            # Registrar cambio de valor
            value_changes.append(float(delta))
            
            # Registrar tiempo de iteración
            iteration_time = time.time() - iteration_start
            iteration_times.append(iteration_time)
            
            iterations = i + 1
            
            print(f"Iteración {iterations}: Delta = {delta:.6f}, Tiempo = {iteration_time:.2f} segundos")
            
            # Verificar convergencia
            if delta < self.theta:
                print("¡Convergencia alcanzada!")
                break
        
        # Extraer política óptima
        policy = self.extract_policy(env)
        
        # Actualizar estado
        self.state = self.state._replace(
            policy=policy,
            value_changes=value_changes,
            iteration_times=iteration_times
        )
        
        total_time = time.time() - start_time
        print(f"Iteración de valor completada en {iterations} iteraciones, {total_time:.2f} segundos")
        
        history = {
            'iterations': iterations,
            'value_changes': value_changes,
            'iteration_times': iteration_times,
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
        # Extraer características relevantes de los datos CGM
        cgm_flat = cgm_data.flatten()
        
        # Calcular estadísticas de CGM: último valor, pendiente, promedio, variabilidad
        cgm_last = cgm_flat[-1]
        cgm_mean = np.mean(cgm_flat)
        cgm_slope = cgm_flat[-1] - cgm_flat[0] if len(cgm_flat) > 1 else 0
        cgm_std = np.std(cgm_flat)
        
        # Discretizar cada característica CGM
        cgm_last_bin = min(int(cgm_last / 400 * self.cgm_bins), self.cgm_bins - 1)
        cgm_mean_bin = min(int(cgm_mean / 400 * self.cgm_bins), self.cgm_bins - 1)
        cgm_slope_bin = min(int((cgm_slope + 100) / 200 * self.cgm_bins), self.cgm_bins - 1)
        cgm_std_bin = min(int(cgm_std / 50 * self.cgm_bins), self.cgm_bins - 1)
        
        # Extraer características relevantes de otras entradas (asumiendo carbInput, bgInput, IOB)
        carb_input = other_features[0]
        bg_input = other_features[1]
        iob = other_features[2]
        
        # Discretizar características adicionales
        carb_bin = min(int(carb_input / 100 * self.other_bins), self.other_bins - 1)
        bg_bin = min(int(bg_input / 400 * self.other_bins), self.other_bins - 1)
        iob_bin = min(int(iob / 10 * self.other_bins), self.other_bins - 1)
        
        # Combinar todas las características discretizadas en un único índice de estado
        # Usar codificación posicional
        state_idx = 0
        state_idx = state_idx * self.cgm_bins + cgm_last_bin
        state_idx = state_idx * self.cgm_bins + cgm_mean_bin
        state_idx = state_idx * self.cgm_bins + cgm_slope_bin
        state_idx = state_idx * self.cgm_bins + cgm_std_bin
        state_idx = state_idx * self.other_bins + carb_bin
        state_idx = state_idx * self.other_bins + bg_bin
        state_idx = state_idx * self.other_bins + iob_bin
        
        # Asegurarse de que el índice esté en el rango válido
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
        epochs: int = 1,
        batch_size: int = 32,
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
                self.rng = np.random.Generator(np.random.PCG64(42))
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


def create_value_iteration_model(cgm_shape: Tuple[int, ...], other_features_shape: Tuple[int, ...]) -> ValueIterationWrapper:
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
    ValueIterationWrapper
        Wrapper de Iteración de Valor que implementa la interfaz compatible con modelos de aprendizaje profundo
    """
    # Configurar el tamaño del espacio de estados y acciones
    # Esto es una simplificación - en un caso real habría que definirlo según los datos
    
    # Estimación del espacio de estados basado en la discretización
    cgm_bins = 8  # Bins para cada característica CGM
    other_bins = 4  # Bins para cada característica adicional
    cgm_features = 4  # Último valor, promedio, pendiente, variabilidad
    other_features = 3  # Carbohidratos, glucosa, insulina a bordo
    
    # Calcular espacio de estados total (discretizado)
    n_states = cgm_bins**cgm_features * other_bins**other_features
    
    # Para un problema de dosificación, discretizamos en niveles de dosis
    n_actions = 20  # 20 niveles discretos (0 a 15 unidades)
    
    # Crear agente de Iteración de Valor con configuración óptima
    vi_agent = ValueIteration(
        n_states=n_states,
        n_actions=n_actions,
        gamma=VALUE_ITERATION_CONFIG['gamma'],
        theta=VALUE_ITERATION_CONFIG['theta'],
        max_iterations=VALUE_ITERATION_CONFIG['max_iterations']
    )
    
    # Crear y devolver wrapper
    return ValueIterationWrapper(vi_agent, cgm_shape, other_features_shape)

def model_creator() -> Callable[[Tuple[int, ...], Tuple[int, ...]], ValueIterationWrapper]:
    """
    Retorna una función para crear un modelo de Iteración de Valor compatible con la API del sistema.
    
    Retorna:
    --------
    Callable[[Tuple[int, ...], Tuple[int, ...]], ValueIterationWrapper]
        Función para crear un modelo con las formas de entrada especificadas
    """
    return create_value_iteration_model