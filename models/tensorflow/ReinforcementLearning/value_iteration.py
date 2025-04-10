import os, sys
import numpy as np
import matplotlib.pyplot as plt
import time
from typing import Dict, List, Tuple, Optional, Union, Any
import pickle
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Concatenate, GlobalAveragePooling1D
from keras.saving import register_keras_serializable

PROJECT_ROOT = os.path.abspath(os.getcwd())
sys.path.append(PROJECT_ROOT) 

from config.models_config import VALUE_ITERATION_CONFIG


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
            # Calcular el valor esperado para la acción a desde el estado state
            for prob, next_s, r, done in transitions[state][a]:
                # Valor esperado usando la ecuación de Bellman
                action_values[a] += prob * (r + self.gamma * self.V[next_s] * (not done))
        
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
            v_old = self.V[s]
            
            # Calcular el valor Q para cada acción y tomar el máximo
            action_values = self._calculate_action_values(s, env.P)
            
            # Actualizar el valor del estado con el máximo valor de acción
            self.V[s] = np.max(action_values)
            
            # Actualizar delta
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
            # Calcular el valor Q para cada acción
            action_values = self._calculate_action_values(s, env.P)
            
            # Política determinística: asignar probabilidad 1.0 a la mejor acción
            best_action = np.argmax(action_values)
            policy[s, best_action] = 1.0
        
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
            
            # Actualizar función de valor
            delta = self.value_update(env)
            
            # Registrar cambio de valor
            self.value_changes.append(delta)
            
            # Registrar tiempo de iteración
            iteration_time = time.time() - iteration_start
            self.iteration_times.append(iteration_time)
            
            iterations = i + 1
            
            print(f"Iteración {iterations}: Delta = {delta:.6f}, Tiempo = {iteration_time:.2f} segundos")
            
            # Verificar convergencia
            if delta < self.theta:
                print("¡Convergencia alcanzada!")
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
            episode_lengths.append(steps)
            print(f"Episodio {ep+1}: Recompensa = {total_reward}, Pasos = {steps}")
        
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
            return env.state_mapping(state)
        else:
            # Asumir orden row-major
            return state // env.shape[1], state % env.shape[1]
    
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
            ax.axvline(i, color='black', linestyle='-')
        for j in range(grid_shape[0] + 1):
            ax.axhline(j, color='black', linestyle='-')
            
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
            0: (0, -0.4),  # Izquierda
            1: (0, 0.4),   # Derecha
            2: (-0.4, 0),  # Abajo
            3: (0.4, 0)    # Arriba
        }
        
        for s in range(self.n_states):
            if self._is_terminal_state(env, s):
                continue
                
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
            i, j = self._get_grid_position(env, s)
            value = self.get_value(s)
            ax.text(j + 0.5, grid_shape[0] - i - 0.5, f"{value:.2f}", 
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
            print("El entorno no tiene estructura de cuadrícula para visualización")
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
        
        grid_shape = env.shape
        
        # Crear matriz para visualización
        value_grid = np.zeros(grid_shape)
        
        # Llenar matriz con valores
        for s in range(self.n_states):
            if hasattr(env, 'state_mapping'):
                i, j = env.state_mapping(s)
            else:
                i, j = s // grid_shape[1], s % grid_shape[1]
                
            value_grid[i, j] = self.V[s]
        
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
        plt.show()
    
    def visualize_training(self, history: Dict[str, List]) -> None:
        """
        Visualiza las métricas de entrenamiento.
        
        Parámetros:
        -----------
        history : Dict[str, List]
            Diccionario con historial de entrenamiento
        """
        _, axs = plt.subplots(2, 1, figsize=(12, 8))
        
        # Gráfico de cambios en la función de valor (delta)
        axs[0].plot(range(1, len(history['value_changes']) + 1), 
                    history['value_changes'])
        axs[0].set_title('Cambios en la Función de Valor (Delta)')
        axs[0].set_xlabel('Iteración')
        axs[0].set_ylabel('Delta')
        axs[0].set_yscale('log')  # Escala logarítmica para ver mejor la convergencia
        axs[0].grid(True)
        
        # Gráfico de tiempos de iteración
        axs[1].plot(range(1, len(history['iteration_times']) + 1), 
                    history['iteration_times'])
        axs[1].set_title('Tiempos de Iteración')
        axs[1].set_xlabel('Iteración')
        axs[1].set_ylabel('Tiempo (segundos)')
        axs[1].grid(True)
        
        plt.tight_layout()
        plt.show()
        
    def vectorized_value_iteration(self, env: Any) -> Dict[str, List]:
        """
        Implementa una versión vectorizada de iteración de valor usando operaciones de NumPy.
        
        Parámetros:
        -----------
        env : Any
            Entorno con dinámicas de transición
            
        Retorna:
        --------
        Dict[str, List]
            Diccionario con historial de entrenamiento
        """
        print("Iniciando iteración de valor vectorizada...")
        
        # Preparar estructuras de datos para cálculo vectorizado
        transitions = np.zeros((self.n_states, self.n_actions, self.n_states))
        rewards = np.zeros((self.n_states, self.n_actions, self.n_states))
        terminals = np.zeros((self.n_states, self.n_actions, self.n_states), dtype=bool)
        
        # Convertir modelo de transición a matrices
        for s in range(self.n_states):
            for a in range(self.n_actions):
                for prob, next_s, r, done in env.P[s][a]:
                    transitions[s, a, next_s] += prob
                    rewards[s, a, next_s] = r
                    terminals[s, a, next_s] = done
        
        iterations = 0
        start_time = time.time()
        self.value_changes = []
        self.iteration_times = []
        
        for i in range(self.max_iterations):
            iteration_start = time.time()
            
            # Calcular los valores Q para todos los estados y acciones
            # Q(s,a) = Σ_s' P(s'|s,a) * [R(s,a,s') + γ*V(s')*(1-terminal(s'))]
            v_expanded = np.expand_dims(self.V, axis=(0, 1))
            q_values = np.sum(
                transitions * (rewards + self.gamma * v_expanded * ~terminals), 
                axis=2
            )
            
            # Actualizar con los valores máximos
            v_old = self.V.copy()
            self.V = np.max(q_values, axis=1)
            
            # Calcular delta
            delta = np.max(np.abs(v_old - self.V))
            
            # Registrar métricas
            self.value_changes.append(delta)
            iteration_time = time.time() - iteration_start
            self.iteration_times.append(iteration_time)
            
            iterations = i + 1
            
            print(f"Iteración {iterations}: Delta = {delta:.6f}, Tiempo = {iteration_time:.2f} segundos")
            
            # Verificar convergencia
            if delta < self.theta:
                print("¡Convergencia alcanzada!")
                break
        
        # Extraer política óptima
        q_values = np.zeros((self.n_states, self.n_actions))
        for s in range(self.n_states):
            for a in range(self.n_actions):
                for prob, next_s, r, done in env.P[s][a]:
                    q_values[s, a] += prob * (r + self.gamma * self.V[next_s] * (not done))
        
        # Política determinística
        best_actions = np.argmax(q_values, axis=1)
        self.policy = np.zeros((self.n_states, self.n_actions))
        self.policy[np.arange(self.n_states), best_actions] = 1.0
        
        total_time = time.time() - start_time
        print(f"Iteración de valor vectorizada completada en {iterations} iteraciones, "
              f"{total_time:.2f} segundos")
        
        history = {
            'iterations': iterations,
            'value_changes': self.value_changes,
            'iteration_times': self.iteration_times,
            'total_time': total_time
        }
        
        return history

@register_keras_serializable
class ValueIterationModel(Model):
    """
    Wrapper para el algoritmo de Iteración de Valor que implementa la interfaz de Keras.Model.
    """
    
    def __init__(
        self, 
        value_iteration_agent: ValueIteration,
        cgm_shape: Tuple[int, ...],
        other_features_shape: Tuple[int, ...]
    ) -> None:
        """
        Inicializa el modelo wrapper para Iteración de Valor.
        
        Parámetros:
        -----------
        value_iteration_agent : ValueIteration
            Agente de Iteración de Valor a utilizar
        cgm_shape : Tuple[int, ...]
            Forma de entrada para datos CGM
        other_features_shape : Tuple[int, ...]
            Forma de entrada para otras características
        """
        super(ValueIterationModel, self).__init__()
        self.value_iteration_agent = value_iteration_agent
        self.cgm_shape = cgm_shape
        self.other_features_shape = other_features_shape
        
        # Capas para procesar entradas CGM
        self.cgm_encoder = Dense(64, activation='relu', name='cgm_encoder')
        self.cgm_pooling = GlobalAveragePooling1D(name='cgm_pooling')
        
        # Capas para procesar otras características
        self.other_encoder = Dense(32, activation='relu', name='other_encoder')
        
        # Capa para combinar características
        self.combined_encoder = Dense(128, activation='relu', name='combined_encoder')
        
        # Capa para codificar a estados discretos
        self.state_encoder = Dense(value_iteration_agent.n_states, activation='softmax', name='state_encoder')
        
        # Capa para decodificar acciones discretas a valores de dosis
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
        # Procesar entradas
        cgm_data, other_features = inputs
        batch_size = tf.shape(cgm_data)[0]
        
        # Codificar estados
        states_distribution = self._encode_states(cgm_data, other_features)
        
        # Convertir a estados discretos
        states = tf.argmax(states_distribution, axis=1)
        
        # Inicializar tensor para acciones
        actions = tf.TensorArray(tf.float32, size=batch_size)
        
        # Para cada ejemplo, determinar acción óptima según la política
        for i in range(batch_size):
            state_idx = states[i]
            action = self.value_iteration_agent.get_action(state_idx.numpy())
            actions = actions.write(i, float(action))
        
        # Convertir a tensor y ajustar forma
        actions_tensor = tf.reshape(actions.stack(), [batch_size, 1])
        
        # Decodificar acciones a valores de dosis
        doses = self._decode_actions(actions_tensor)
        
        return doses
    
    def _encode_states(self, cgm_data: tf.Tensor, other_features: tf.Tensor) -> tf.Tensor:
        """
        Codifica las entradas en estados discretos.
        
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
        # Procesar datos CGM
        cgm_encoded = self.cgm_encoder(cgm_data)
        cgm_features = self.cgm_pooling(cgm_encoded)
        
        # Procesar otras características
        other_encoded = self.other_encoder(other_features)
        
        # Combinar características
        combined = tf.concat([cgm_features, other_encoded], axis=1)
        combined_encoded = self.combined_encoder(combined)
        
        # Codificar a distribución de estados discretos
        states_distribution = self.state_encoder(combined_encoded)
        
        return states_distribution
    
    def _decode_actions(self, actions: tf.Tensor) -> tf.Tensor:
        """
        Decodifica índices de acción a valores de dosis.
        
        Parámetros:
        -----------
        actions : tf.Tensor
            Índices de acciones
            
        Retorna:
        --------
        tf.Tensor
            Valores de dosis correspondientes
        """
        # Convertir índices de acción a representación one-hot
        one_hot = tf.one_hot(tf.cast(actions, tf.int32), self.value_iteration_agent.n_actions)
        
        # Mapear a valores continuos (dosis)
        doses = self.action_decoder(one_hot)
        
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
        Simula la interfaz de entrenamiento de Keras para el agente de Iteración de Valor.
        
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
            print("Entrenando modelo de Iteración de Valor...")
        
        # Crear entorno para entrenar el agente
        env = self._create_environment(x[0], x[1], y)
        
        # Entrenar el agente de Iteración de Valor
        history = self.value_iteration_agent.train(env)
        
        # Calibrar la capa de decodificación
        self._calibrate_action_decoder(y)
        
        if verbose > 0:
            print(f"Entrenamiento completado en {history.get('iterations', 0)} iteraciones")
        
        # Crear historia simulada para compatibilidad con Keras
        keras_history = {
            'loss': history.get('value_changes', [0.0]),
            'val_loss': [history.get('value_changes', [0.0])[-1]] if validation_data is not None else None
        }
        
        return {'history': keras_history}
    
    def _create_environment(self, cgm_data: tf.Tensor, other_features: tf.Tensor, 
                           target_doses: tf.Tensor) -> Any:
        """
        Crea un entorno compatible con el agente de Iteración de Valor.
        
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
            Entorno compatible para entrenamiento
        """
        # Convertir tensores a numpy para procesamiento
        cgm_np = cgm_data.numpy() if hasattr(cgm_data, 'numpy') else cgm_data
        other_np = other_features.numpy() if hasattr(other_features, 'numpy') else other_features
        target_np = target_doses.numpy() if hasattr(target_doses, 'numpy') else target_doses
        
        # Clase de entorno personalizada
        class InsulinDosingEnv:
            """Entorno para simular problema de dosificación de insulina."""
            
            def __init__(self, cgm, features, targets, model):
                self.cgm = cgm
                self.features = features
                self.targets = targets
                self.model = model
                self.rng = np.random.Generator(np.random.PCG64(42))
                self.current_idx = 0
                self.max_idx = len(targets) - 1
                
                # Para compatibilidad con algoritmos RL
                self.n_states = model.value_iteration_agent.n_states
                self.n_actions = model.value_iteration_agent.n_actions
                self.shape = (1, 1)  # Forma ficticia para visualización
                
                # Crear modelo dinámico de transición para VI
                self.P = self._create_transition_model()
                
            def _create_transition_model(self):
                """Crea modelo de transición para Value Iteration."""
                # Inicializar modelo de transición (diccionario anidado)
                # Formato P[state][action] = [(prob, next_state, reward, done), ...]
                P = {}
                
                # Para cada estado posible
                for s in range(self.n_states):
                    P[s] = {}
                    
                    # Para cada acción posible
                    for a in range(self.n_actions):
                        # Mapear acción a dosis
                        dose = a * 15.0 / (self.n_actions - 1)  # Max 15 U
                        
                        # Buscar ejemplos similares al estado actual
                        similar_indices = self._find_similar_states(s, max_samples=10)
                        if len(similar_indices) == 0:
                            # Si no hay ejemplos similares, usar transición por defecto
                            P[s][a] = [(1.0, s, 0.0, True)]
                            continue
                        
                        # Calcular recompensa basada en diferencia con dosis objetivo
                        rewards = -np.abs(self.targets[similar_indices] - dose)
                        avg_reward = float(np.mean(rewards))
                        
                        # Simplificación: siempre termina después de una acción
                        P[s][a] = [(1.0, s, avg_reward, True)]
                
                return P
                
            def _find_similar_states(self, state_idx, max_samples=10):
                """Encuentra índices de ejemplos con estados similares."""
                # Codificar todos los estados
                all_states = self.model._encode_states(
                    tf.convert_to_tensor(self.cgm),
                    tf.convert_to_tensor(self.features)
                )
                
                # Obtener estados con mayor probabilidad para el índice dado
                state_probs = all_states.numpy()[:, state_idx]
                indices = np.argsort(-state_probs)[:max_samples]
                
                # Filtrar por umbral mínimo de similaridad
                threshold = 0.01
                indices = indices[state_probs[indices] > threshold]
                
                return indices
                
            def reset(self):
                """Reinicia el entorno a un estado aleatorio."""
                self.current_idx = self.rng.integers(0, self.max_idx)
                state = self._get_state()
                return state, {}
                
            def step(self, action):
                """Ejecuta un paso en el entorno con la acción dada."""
                # Convertir acción a dosis
                dose = action * 15.0 / (self.n_actions - 1)  # Max 15 U
                
                # Calcular recompensa (negativo del error absoluto)
                target = self.targets[self.current_idx]
                reward = -abs(dose - target)
                
                # Avanzar al siguiente ejemplo
                self.current_idx = (self.current_idx + 1) % self.max_idx
                
                # Obtener próximo estado
                next_state = self._get_state()
                
                # Siempre termina después de un paso
                done = True
                truncated = False
                
                return next_state, reward, done, truncated, {}
                
            def _get_state(self):
                """Obtiene el estado discreto para el ejemplo actual."""
                # Codificar estado actual
                states = self.model._encode_states(
                    tf.convert_to_tensor(self.cgm[self.current_idx:self.current_idx+1]),
                    tf.convert_to_tensor(self.features[self.current_idx:self.current_idx+1])
                )
                
                # Obtener el estado más probable
                state = tf.argmax(states, axis=1)[0].numpy()
                return int(state)
        
        return InsulinDosingEnv(cgm_np, other_np, target_np, self)
    
    def _calibrate_action_decoder(self, y: tf.Tensor) -> None:
        """
        Calibra la capa de decodificación para mapear acciones a dosis adecuadas.
        
        Parámetros:
        -----------
        y : tf.Tensor
            Dosis objetivo para calibración
        """
        # Extraer rango de dosis
        y_np = y.numpy() if hasattr(y, 'numpy') else y
        max_dose = np.max(y_np)
        min_dose = np.min(y_np)
        dose_range = max_dose - min_dose
        
        # Configurar pesos para mapear del espacio discreto al rango de dosis
        weights = [
            np.ones((self.value_iteration_agent.n_actions, 1)) * dose_range / self.value_iteration_agent.n_actions,
            np.array([min_dose])
        ]
        self.action_decoder.set_weights(weights)
    
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
            "other_features_shape": self.other_features_shape,
            "n_states": self.value_iteration_agent.n_states,
            "n_actions": self.value_iteration_agent.n_actions,
            "gamma": self.value_iteration_agent.gamma,
            "theta": self.value_iteration_agent.theta,
        }
    
    def save(self, filepath: str, **kwargs) -> None:
        """
        Guarda el modelo de Iteración de Valor.
        
        Parámetros:
        -----------
        filepath : str
            Ruta donde guardar el modelo
        **kwargs
            Argumentos adicionales
        """
        # Guardar agente de VI
        self.value_iteration_agent.save(filepath + VI_AGENT_SUFFIX)
        
        # Guardar pesos de capas de keras
        super().save_weights(filepath + WRAPPER_WEIGHTS_SUFFIX)
        
    def load_weights(self, filepath: str, **kwargs) -> None:
        """
        Carga el modelo de Iteración de Valor.
        
        Parámetros:
        -----------
        filepath : str
            Ruta desde donde cargar el modelo
        **kwargs
            Argumentos adicionales
        """
        # Determinar rutas correctas
        if filepath.endswith(WRAPPER_WEIGHTS_SUFFIX):
            agent_path = filepath.replace(WRAPPER_WEIGHTS_SUFFIX, VI_AGENT_SUFFIX)
            wrapper_path = filepath
        else:
            agent_path = filepath + VI_AGENT_SUFFIX
            wrapper_path = filepath + WRAPPER_WEIGHTS_SUFFIX
        
        # Cargar agente de VI
        self.value_iteration_agent.load(agent_path)
        
        # Cargar pesos de capas de keras
        super().load_weights(wrapper_path)


# Constantes para uso en el modelo
STATE_ENCODER = 'state_encoder'
ACTION_DECODER = 'action_decoder'
CGM_ENCODER = 'cgm_encoder'
OTHER_ENCODER = 'other_encoder'
WRAPPER_WEIGHTS_SUFFIX = '_wrapper_weights.h5'
VI_AGENT_SUFFIX = '_vi_agent'


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