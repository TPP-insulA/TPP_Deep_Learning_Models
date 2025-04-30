import os, sys
import numpy as np
import matplotlib.pyplot as plt
import time
import pickle
from typing import Dict, List, Tuple, Optional, Union, Any
import tensorflow as tf
from keras._tf_keras.keras.models import Model
from keras._tf_keras.keras.layers import Input, Dense, Concatenate
from keras._tf_keras.keras.saving import register_keras_serializable

PROJECT_ROOT = os.path.abspath(os.getcwd())
sys.path.append(PROJECT_ROOT) 

from config.models_config import POLICY_ITERATION_CONFIG
from constants.constants import CONST_DEFAULT_SEED
from custom.printer import print_warning, print_success

# Constantes para rutas y mensajes
FIGURAS_DIR = "figures/reinforcement_learning/policy_iteration"
CONST_ITERACION = "Iteración"
CONST_RECOMPENSA = "Recompensa"
CONST_VALOR = "Valor"
CONST_TIEMPO = "Tiempo (segundos)"
CONST_POLITICA = "Política"
CONST_PROB_TRANSICION = 1.0

class PolicyIteration:
    """
    Implementación del algoritmo de Iteración de Política.
    
    La Iteración de Política alterna entre Evaluación de Política (calcular la función
    de valor para la política actual) y Mejora de Política (hacer la política codiciosa
    respecto a la función de valor actual).
    """
    
    def __init__(
        self,
        n_states: int,
        n_actions: int,
        gamma: float = POLICY_ITERATION_CONFIG['gamma'],
        theta: float = POLICY_ITERATION_CONFIG['theta'],
        max_iterations: int = POLICY_ITERATION_CONFIG['max_iterations'],
        max_iterations_eval: int = POLICY_ITERATION_CONFIG['max_iterations_eval'],
        seed: int = POLICY_ITERATION_CONFIG.get('seed', CONST_DEFAULT_SEED)
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
            if policy[state, a] > 0:  # Solo considerar acciones con probabilidad > 0
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
        
        # Solo evaluar estados que están presentes en el entorno
        states_to_evaluate = list(env.P.keys())
        print(f"Evaluando política para {len(states_to_evaluate)} estados...")
        
        for i in range(self.max_iterations_eval):
            delta = 0
            
            for s in states_to_evaluate:
                v_old = v[s]
                v[s] = self._calculate_state_value(env, s, policy, v)
                delta = max(delta, abs(v_old - v[s]))
            
            # Reportar progreso
            if i % 2 == 0:
                print(f"  Iteración de evaluación {i+1}/{self.max_iterations_eval}, delta={delta:.5f}")
            
            # Verificar convergencia
            if delta < self.theta:
                print(f"  Evaluación convergió después de {i+1} iteraciones (delta={delta:.5f})")
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
        # Inicializar nueva política con la política actual
        policy = np.copy(self.policy)
        policy_stable = True
        
        # Solo procesar estados que existen en el entorno
        states_to_process = list(env.P.keys())
        print(f"Mejorando política para {len(states_to_process)} estados...")
        
        for s in states_to_process:
            # Encontrar la mejor acción anterior
            old_action = np.argmax(self.policy[s])
            
            # Calcular valores de acción para el estado actual
            action_values = np.zeros(self.n_actions)
            
            for a in range(self.n_actions):
                if a in env.P[s]:  # Verificar si la acción está definida para este estado
                    for prob, next_s, r, done in env.P[s][a]:
                        # Actualizar valor de acción utilizando la ecuación de Bellman
                        action_values[a] += prob * (r + self.gamma * v[next_s] * (not done))
            
            # Obtener la nueva mejor acción (con valor máximo)
            best_action = np.argmax(action_values)
            
            # Manejar acciones con valores idénticos (romper empates de manera determinista)
            if np.sum(action_values == action_values[best_action]) > 1:
                # Encontrar todas las acciones con valor máximo
                max_indices = np.nonzero(action_values == action_values[best_action])[0]
                # Elegir una de manera determinista basada en el estado
                best_action = max_indices[hash(str(s)) % len(max_indices)]
            
            # Actualizar política: determinística (probabilidad 1.0 para la mejor acción)
            policy[s] = np.zeros(self.n_actions)
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
        max_training_time = 300  # Limitar tiempo de entrenamiento a 5 minutos
        
        old_v = np.zeros(self.n_states)
        
        while not policy_stable and iterations < self.max_iterations:
            iteration_start = time.time()
            
            # Verificar si excedimos el tiempo máximo
            if time.time() - start_time > max_training_time:
                print("¡Tiempo máximo de entrenamiento excedido!")
                break
            
            print(f"Iniciando iteración {iterations+1}/{self.max_iterations}...")
            
            # Evaluación de Política: Calcular función de valor para la política actual
            self.v = self.policy_evaluation(env, self.policy)
            
            # Calcular cambio de valor para métricas
            if iterations > 0:
                value_change = np.mean(np.abs(self.v - old_v))
                self.value_changes.append(value_change)
                print(f"Cambio de valor: {value_change:.5f}")
            old_v = self.v.copy()
            
            # Mejora de Política: Actualizar política basada en nueva función de valor
            print("Realizando mejora de política...")
            new_policy, policy_stable = self.policy_improvement(env, self.v)
            
            # Calcular cambio de política para métricas
            if iterations > 0:
                # Solo calcular cambio para estados en env.P
                states_in_env = list(env.P.keys())
                policy_diff = np.abs(new_policy[states_in_env] - self.policy[states_in_env])
                policy_change = np.sum(policy_diff) / (2 * len(states_in_env))
                self.policy_changes.append(policy_change)
                print(f"Cambio de política: {policy_change:.5f}")
            
            self.policy = new_policy
            iterations += 1
            
            # Registrar tiempo transcurrido
            iteration_time = time.time() - iteration_start
            self.policy_iteration_times.append(iteration_time)
            
            print(f"Iteración {iterations} completada en {iteration_time:.2f} segundos, " + 
                  f"Iteraciones de evaluación: {self.eval_iteration_counts[-1]}")
            
            if policy_stable:
                print_success("¡Política convergida!")
            
            # Agregar un límite adicional para iteraciones de evaluación costosas
            if iterations >= 3 and self.eval_iteration_counts[-1] >= self.max_iterations_eval * 0.8:
                print("Las evaluaciones de política están tomando demasiado tiempo. Finalizando entrenamiento.")
                break
        
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
            total_reward = 0
            steps = 0
            terminated = False
            truncated = False
            
            while not (terminated or truncated) and steps < max_steps:
                action = self.get_action(state)
                next_state, reward, terminated, truncated, _ = env.step(action)
                
                # done = terminated or truncated
                
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
    
    def _determine_grid_shape(self, env: Any) -> Tuple[int, int]:
        """
        Determina la forma de la cuadrícula para visualización.
        
        Parámetros:
        -----------
        env : Any
            Entorno con dinámicas de transición
            
        Retorna:
        --------
        Tuple[int, int]
            Forma de la cuadrícula (filas, columnas)
        """
        if hasattr(env, 'shape'):
            return env.shape
        elif hasattr(env, 'grid_shape'):
            return env.grid_shape
        else:
            # Intentar inferir la forma de la cuadrícula a partir del número de estados
            grid_size = int(np.sqrt(self.n_states))
            return (grid_size, grid_size)
    
    def _draw_grid_lines(self, ax: plt.Axes, grid_shape: Tuple[int, int]) -> None:
        """
        Dibuja las líneas de la cuadrícula.
        
        Parámetros:
        -----------
        ax : plt.Axes
            Ejes para dibujar
        grid_shape : Tuple[int, int]
            Forma de la cuadrícula (filas, columnas)
        """
        for i in range(grid_shape[1] + 1):
            ax.axvline(i, color='black', linestyle='-')
        for j in range(grid_shape[0] + 1):
            ax.axhline(j, color='black', linestyle='-')
    
    def _is_terminal_state(self, env: Any, s: int) -> bool:
        """
        Verifica si un estado es terminal.
        
        Parámetros:
        -----------
        env : Any
            Entorno con dinámicas de transición
        s : int
            Estado a verificar
            
        Retorna:
        --------
        bool
            True si el estado es terminal, False en caso contrario
        """
        if not hasattr(env, 'P') or s not in env.P:
            return False
            
        for a in env.P[s]:
            for _, _, _, done in env.P[s][a]:
                if done:
                    return True
        return False
    
    def _draw_action_arrows(self, ax: plt.Axes, grid_shape: Tuple[int, int], env: Any) -> None:
        """
        Dibuja flechas representando acciones de la política.
        
        Parámetros:
        -----------
        ax : plt.Axes
            Ejes para dibujar
        grid_shape : Tuple[int, int]
            Forma de la cuadrícula
        env : Any
            Entorno con dinámicas de transición
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
                
            # Obtener posición en cuadrícula del estado
            i, j = s // grid_shape[1], s % grid_shape[1]
            
            # Obtener acción según la política
            action = self.get_action(s)
            
            if action in directions:
                dx, dy = directions[action]
                ax.arrow(j + 0.5, grid_shape[0] - i - 0.5, dx, dy, 
                         head_width=0.1, head_length=0.1, fc='blue', ec='blue')
    
    def _draw_state_values(self, ax: plt.Axes, grid_shape: Tuple[int, int]) -> None:
        """
        Dibuja los valores de los estados.
        
        Parámetros:
        -----------
        ax : plt.Axes
            Ejes para dibujar
        grid_shape : Tuple[int, int]
            Forma de la cuadrícula
        """
        for s in range(self.n_states):
            i, j = s // grid_shape[1], s % grid_shape[1]
            ax.text(j + 0.5, grid_shape[0] - i - 0.5, f"{self.v[s]:.2f}", 
                   ha='center', va='center', color='red', fontsize=9)
    
    def visualize_policy(self, env: Any, title: str = "Política") -> None:
        """
        Visualiza la política actual.
        
        Parámetros:
        -----------
        env : Any
            Entorno con dinámicas de transición
        title : str, opcional
            Título del gráfico (default: "Política")
        """
        grid_shape = self._determine_grid_shape(env)
        _, ax = plt.subplots(figsize=(8, 8))
        
        self._draw_grid_lines(ax, grid_shape)
        self._draw_action_arrows(ax, grid_shape, env)
        self._draw_state_values(ax, grid_shape)
        
        ax.set_title(title)
        plt.tight_layout()
        
        # Guardar figura
        os.makedirs(FIGURAS_DIR, exist_ok=True)
        plt.savefig(os.path.join(FIGURAS_DIR, f"politica_{title.lower().replace(' ', '_')}.png"), dpi=300)
        plt.show()
    
    def visualize_training(self, history: Dict[str, List]) -> None:
        """
        Visualiza las métricas de entrenamiento.
        
        Parámetros:
        -----------
        history : Dict[str, List]
            Diccionario con historiales de métricas
        """
        _, axs = plt.subplots(2, 2, figsize=(12, 10))
        
        # Gráfico de cambios de política
        if 'policy_changes' in history and history['policy_changes']:
            x = range(1, len(history['policy_changes']) + 1)
            axs[0, 0].plot(x, history['policy_changes'], marker='o')
            axs[0, 0].set_title('Cambios en la Política')
            axs[0, 0].set_xlabel(CONST_ITERACION)
            axs[0, 0].set_ylabel('Cambio Promedio de Política')
            axs[0, 0].grid(True)
        
        # Gráfico de cambios de valor
        if 'value_changes' in history and history['value_changes']:
            x = range(1, len(history['value_changes']) + 1)
            axs[0, 1].plot(x, history['value_changes'], marker='o')
            axs[0, 1].set_title('Cambios en la Función de Valor')
            axs[0, 1].set_xlabel(CONST_ITERACION)
            axs[0, 1].set_ylabel('Cambio Promedio de Valor')
            axs[0, 1].grid(True)
        
        # Gráfico de tiempos de iteración
        if 'iteration_times' in history and history['iteration_times']:
            x = range(1, len(history['iteration_times']) + 1)
            axs[1, 0].plot(x, history['iteration_times'], marker='o')
            axs[1, 0].set_title('Tiempos de Iteración')
            axs[1, 0].set_xlabel(CONST_ITERACION)
            axs[1, 0].set_ylabel(CONST_TIEMPO)
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
        os.makedirs(FIGURAS_DIR, exist_ok=True)
        plt.savefig(os.path.join(FIGURAS_DIR, "entrenamiento_resumen.png"), dpi=300)
        plt.show()


@register_keras_serializable()
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
            Agente de Iteración de Política
        cgm_shape : Tuple[int, ...]
            Forma de entrada para datos CGM
        other_features_shape : Tuple[int, ...]
            Forma de entrada para otras características
        discretizer : Optional[Any], opcional
            Función o objeto para discretizar estados (default: None)
        """
        super().__init__()
        self.policy_iteration_agent = policy_iteration_agent
        self.cgm_shape = cgm_shape
        self.other_features_shape = other_features_shape
        self.discretizer = discretizer
        
        # Mejora: usar LSTM para manejar la serie temporal de CGM
        # Esto conserva la forma 3D, así que necesitaremos aplanar después
        self.cgm_encoder = tf.keras.Sequential([
            # Capa para preservar la dimensionalidad temporal
            Dense(64, activation='relu', input_shape=cgm_shape),
            Dense(32, activation='relu')
            # No usar softmax aquí, mantenemos dimensionalidad para posterior flatten
        ])
        
        # Encoder para otras características
        self.other_encoder = tf.keras.Sequential([
            Dense(64, activation='relu', input_shape=other_features_shape),
            Dense(32, activation='relu')
        ])
        
        # Después del flatten y concatenación, usamos esta capa para mapear a estados
        self.combined_encoder = Dense(policy_iteration_agent.n_states, activation='softmax')
        
        # Capa para decodificar acciones a valores continuos
        # Asegurar que la capa se construya completamente
        self.action_decoder = Dense(1, use_bias=True)
        
        # Construir las capas inmediatamente con formas de entrada apropiadas
        # Esto asegura que estén listas para configurar pesos
        dummy_cgm = tf.zeros((1,) + cgm_shape)
        dummy_other = tf.zeros((1,) + other_features_shape)
        _ = self.cgm_encoder(dummy_cgm)
        _ = self.other_encoder(dummy_other)
        
        # Calculamos formas intermedias para evitar errores
        dummy_cgm_encoded = self.cgm_encoder(dummy_cgm)
        dummy_cgm_flat = tf.keras.layers.Flatten()(dummy_cgm_encoded)
        dummy_other_encoded = self.other_encoder(dummy_other)
        dummy_combined = Concatenate()([dummy_cgm_flat, dummy_other_encoded])
        
        _ = self.combined_encoder(dummy_combined)
        
        dummy_actions = tf.zeros((1, policy_iteration_agent.n_actions))
        _ = self.action_decoder(dummy_actions)
    
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
        Ejecuta el modelo en los datos de entrada.
        
        Parámetros:
        -----------
        inputs : List[tf.Tensor]
            Lista con [cgm_data, other_features]
        training : bool, opcional
            Si está en modo entrenamiento (default: False)
            
        Retorna:
        --------
        tf.Tensor
            Predicción de dosis de insulina
        """
        # Extraer entradas
        cgm_data, other_features = inputs
        
        # Codificar entrada como distribución sobre estados
        states = self._encode_states(cgm_data, other_features)
        
        # Extraer matriz de política del agente
        policy_matrix = tf.constant(self.policy_iteration_agent.policy, dtype=tf.float32)
        
        # Extraer las acciones para los estados calculados
        actions = tf.matmul(states, policy_matrix)
        
        # Mapear acción discreta a valor continuo de dosis
        action_values = self.action_decoder(actions)
        
        # Asegurar que la salida tenga la forma correcta (batch_size,)
        return tf.reshape(action_values, [-1])
    
    def _process_dataset_input(self, x: tf.data.Dataset) -> None:
        """
        Procesa la entrada cuando es un objeto tf.data.Dataset.
        
        Parámetros:
        -----------
        x : tf.data.Dataset
            Dataset con datos de entrenamiento
        """
        dataset_iter = iter(x.take(1))
        first_batch = next(dataset_iter)
        
        if not isinstance(first_batch, tuple) or len(first_batch) != 2:
            raise ValueError("Formato de datos no compatible. Se espera que el dataset contenga tuplas (x, y)")
            
        x_sample, _ = first_batch
        if not isinstance(x_sample, (list, tuple)):
            raise ValueError("Formato de datos no compatible. Se espera que el dataset contenga tuplas (x, y) donde x es [cgm_data, other_features]")
            
        # Usar un lote representativo para calibrar el decodificador
        self._update_encoders_from_dataset(x)
    
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
    
    def fit(
        self, 
        x: Any, 
        y: Optional[tf.Tensor] = None, 
        batch_size: int = 32, 
        epochs: int = 1, 
        verbose: int = 0,
        callbacks: Optional[List[Any]] = None,
        validation_data: Optional[Any] = None,
        **kwargs
    ) -> Dict:
        """
        Entrena el modelo con los datos proporcionados.
        
        Parámetros:
        -----------
        x : Any
            Datos de entrada, puede ser una lista de tensores [cgm_data, other_features]
            o un objeto tf.data.Dataset
        y : Optional[tf.Tensor], opcional
            Etiquetas (dosis objetivo), no necesario si x es un Dataset (default: None)
        batch_size : int, opcional
            Tamaño del lote (default: 32)
        epochs : int, opcional
            Número de épocas (default: 1)
        verbose : int, opcional
            Nivel de verbosidad (default: 0)
        callbacks : Optional[List[Any]], opcional
            Lista de callbacks (default: None)
        validation_data : Optional[Any], opcional
            Datos de validación (default: None)
        **kwargs
            Argumentos adicionales
            
        Retorna:
        --------
        Dict
            Historia simulada de entrenamiento
        """
        # Procesar entrada según su tipo
        if isinstance(x, tf.data.Dataset):
            self._process_dataset_input(x)
        else:
            if y is None:
                raise ValueError("Si 'x' no es un dataset, 'y' debe ser proporcionado")
            self._process_tensor_input(x, y)
        
        # Simulación de entrenamiento
        history = {
            'loss': [0.0],
            'val_loss': [0.0] if validation_data is not None else None
        }
        
        return {'history': history}
    
    def _update_encoders_from_dataset(self, dataset: tf.data.Dataset) -> None:
        """
        Actualiza los encoders basados en un dataset.
        
        Parámetros:
        -----------
        dataset : tf.data.Dataset
            Dataset con datos de entrenamiento
        """
        # Extraer suficientes muestras para entrenamiento
        batch_size = 32
        max_samples = 1000
        x_cgm_list = []
        x_other_list = []
        y_list = []
        
        for batch in dataset.take(max_samples // batch_size):
            x_batch, y_batch = batch
            cgm_batch, other_batch = x_batch
            
            x_cgm_list.append(cgm_batch)
            x_other_list.append(other_batch)
            y_list.append(y_batch)
        
        x_cgm = tf.concat(x_cgm_list, axis=0)
        x_other = tf.concat(x_other_list, axis=0)
        y = tf.concat(y_list, axis=0)
        
        # Crear un entorno de entrenamiento y entrenar el agente
        self._update_encoders(x_cgm, x_other, y)
    
    def _update_encoders(self, x_cgm: tf.Tensor, x_other: tf.Tensor, y: tf.Tensor) -> None:
        """
        Actualiza los encoders basados en los datos.
        
        Parámetros:
        -----------
        x_cgm : tf.Tensor
            Datos CGM
        x_other : tf.Tensor
            Otras características
        y : tf.Tensor
            Etiquetas (dosis objetivo)
        """
        # Crear un entorno de entrenamiento
        env = _create_training_environment(x_cgm, x_other, y, self.policy_iteration_agent)
        
        # Entrenar el agente en este entorno
        self.policy_iteration_agent.train(env)
        
        # Actualizar decodificador de acciones
        # Asegurar que la capa esté construida antes de establecer pesos
        if not hasattr(self.action_decoder, 'built') or not self.action_decoder.built:
            # Construir explícitamente la capa con la forma correcta
            dummy_input = tf.zeros([1, self.policy_iteration_agent.n_actions])
            _ = self.action_decoder(dummy_input)
        
        # Establecer pesos para mapear linealmente desde espacio de acciones discretas a dosis continuas
        min_dose = tf.reduce_min(y, axis=0)
        max_dose = tf.reduce_max(y, axis=0)
        dose_spread = max_dose - min_dose
        
        # Asegurar que todos los valores son del mismo tipo (float32)
        dose_ratio = tf.cast(dose_spread / self.policy_iteration_agent.n_actions, tf.float32)
        min_dose_float32 = tf.cast(min_dose, tf.float32)
        
        # Verificar que la forma de los pesos coincida con lo que espera la capa
        kernel_shape = self.action_decoder.kernel.shape
        bias_shape = self.action_decoder.bias.shape
        
        # Configurar capa decodificadora para mapear acciones a dosis en el rango [min_dose, max_dose]
        self.action_decoder.set_weights([
            tf.ones(kernel_shape, dtype=tf.float32) * (dose_ratio / kernel_shape[0]),
            tf.ones(bias_shape, dtype=tf.float32) * min_dose_float32
        ])
    
    def predict(self, x: List[tf.Tensor], **kwargs) -> tf.Tensor:
        """
        Realiza predicciones con el modelo.
        
        Parámetros:
        -----------
        x : List[tf.Tensor]
            Lista con [cgm_data, other_features]
        **kwargs
            Argumentos adicionales
            
        Retorna:
        --------
        tf.Tensor
            Predicción de dosis de insulina
        """
        return super().predict(x, **kwargs)
    
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
            'other_features_shape': self.other_features_shape
        })
        return config
    
    def save(self, filepath: str, **kwargs) -> None:
        """
        Guarda el modelo en disco.
        
        Parámetros:
        -----------
        filepath : str
            Ruta donde guardar el modelo
        **kwargs
            Argumentos adicionales
        """
        # Guardar el agente de Policy Iteration por separado
        pi_path = filepath + '_pi_agent.pkl'
        self.policy_iteration_agent.save(pi_path)
        
        # Guardar el modelo Keras
        super().save(filepath, **kwargs)
    
    def load_weights(self, filepath: str, **kwargs) -> None:
        """
        Carga pesos del modelo.
        
        Parámetros:
        -----------
        filepath : str
            Ruta desde donde cargar los pesos
        **kwargs
            Argumentos adicionales
        """
        # Cargar el agente de Policy Iteration si existe
        pi_path = filepath + '_pi_agent.pkl'
        if os.path.exists(pi_path):
            self.policy_iteration_agent.load(pi_path)
        
        # Cargar pesos Keras
        super().load_weights(filepath, **kwargs)


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
        print_warning("No se encontraron estados únicos. Creando estado por defecto.")
        unique_states = np.array([0])
        
    if len(unique_states) < 3:
        print(f"Pocos estados únicos ({len(unique_states)}). Creando estados adicionales.")
        additional_states = np.array([s+1 for s in unique_states if s+1 < n_states] + 
                                   [s-1 for s in unique_states if s-1 >= 0])
        unique_states = np.unique(np.concatenate([unique_states, additional_states]))
        print(f"Ahora procesando {len(unique_states)} estados.")
    
    return unique_states


def _create_training_environment(
    cgm_data: tf.Tensor, 
    other_features: tf.Tensor, 
    targets: tf.Tensor,
    policy_iteration_agent: PolicyIteration
) -> Any:
    """
    Crea un entorno de entrenamiento para el agente de Iteración de Política.
    
    Parámetros:
    -----------
    cgm_data : tf.Tensor
        Datos CGM 
    other_features : tf.Tensor
        Otras características
    targets : tf.Tensor
        Dosis objetivo
    policy_iteration_agent: PolicyIteration
        Agente de Iteración de Política
        
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
        
        def __init__(self, cgm: np.ndarray, features: np.ndarray, targets: np.ndarray, agent: PolicyIteration):
            """Inicializa el entorno con los datos proporcionados."""
            self.cgm = cgm
            self.features = features
            self.targets = targets
            self.agent = agent
            self.rng = np.random.Generator(np.random.PCG64(CONST_DEFAULT_SEED))
            self.current_idx = 0
            self.max_idx = len(targets) - 1
            
            # Para compatibilidad con algoritmos RL
            self.n_states = agent.n_states
            self.n_actions = agent.n_actions
            self.shape = (1, 1)  # Forma ficticia para visualización
            
            # Precalcular estados discretizados
            print("Precalculando estados discretizados...")
            self.cached_states = np.zeros(len(targets), dtype=np.int32)
            for i in range(len(targets)):
                self.cached_states[i] = self._discretize_state(cgm[i], features[i])
            
            # Crear modelo dinámico de transición
            print("Creando modelo de dinámica de transición...")
            self.P = self._create_dynamics()
            print("Modelo de dinámica creado con éxito.")
        
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
        
        def _create_dynamics(self) -> Dict:
            """Crea el modelo de dinámica de transición para Policy Iteration."""
            P = {}
            
            # Obtener estados únicos a procesar
            unique_states = _handle_unique_states(self.cached_states, self.n_states)
            
            # Crear transiciones para cada estado único
            for s in unique_states:
                P[s] = {}
                
                for a in range(self.n_actions):
                    dose = self._action_to_dose(a)
                    P[s][a] = []
                    
                    # Encontrar ejemplos similares a este estado
                    similar_indices = np.nonzero(self.cached_states == s)[0]
                    
                    if len(similar_indices) > 0:
                        # Calcular error promedio para estos ejemplos
                        errors = [-abs(dose - self.targets[idx]) for idx in similar_indices]
                        avg_error = np.mean(errors)
                        P[s][a].append((CONST_PROB_TRANSICION, s, avg_error, True))
                    else:
                        # Si no hay ejemplos similares, usar penalización por defecto
                        P[s][a].append((CONST_PROB_TRANSICION, s, -2.0, True))
            
            # Asegurar que existe al menos un estado con todas las acciones definidas
            if 0 not in P:
                P[0] = {}
                for a in range(self.n_actions):
                    P[0][a] = [(CONST_PROB_TRANSICION, 0, -1.0, True)]
            
            return P
    
    return InsulinDosingEnv(cgm_np, other_np, target_np, policy_iteration_agent)


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
    # Reducir drásticamente el número de estados y acciones para mejorar rendimiento
    n_states = 200  # Reducido de 500 para acelerar el entrenamiento
    n_actions = 8   # Reducido de 10 para acelerar el entrenamiento
    
    # Usar valores por defecto más conservadores para una convergencia más rápida
    gamma_default = 0.95 if 'gamma' not in POLICY_ITERATION_CONFIG else POLICY_ITERATION_CONFIG['gamma']
    theta_default = 0.1 if 'theta' not in POLICY_ITERATION_CONFIG else POLICY_ITERATION_CONFIG['theta']
    
    # Crear agente de Iteración de Política con límites estrictos
    policy_iteration_agent = PolicyIteration(
        n_states=n_states,
        n_actions=n_actions,
        gamma=gamma_default,
        theta=theta_default,
        max_iterations=10,   # Limitar a 10 iteraciones máximo (en lugar de 20)
        max_iterations_eval=5 # Limitar a 5 iteraciones de evaluación (en lugar de 10)
    )
    
    # Crear y devolver el modelo wrapper
    return PolicyIterationModel(
        policy_iteration_agent=policy_iteration_agent,
        cgm_shape=cgm_shape,
        other_features_shape=other_features_shape
    )