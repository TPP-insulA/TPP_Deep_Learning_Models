from typing import Dict, List, Tuple, Optional, Union, Any
import numpy as np
import time
import pickle
import os
import sys
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Input, Concatenate, GlobalAveragePooling1D
from keras.saving import register_keras_serializable
from types import SimpleNamespace
import matplotlib.pyplot as plt

PROJECT_ROOT = os.path.abspath(os.getcwd())
sys.path.append(PROJECT_ROOT) 

from config.models_config import SARSA_CONFIG

# Constantes para prevenir duplicación
CONST_CGM_ENCODER = 'cgm_encoder'
CONST_OTHER_ENCODER = 'other_encoder'
CONST_STATE_ENCODER = 'state_encoder'
CONST_ACTION_DECODER = 'action_decoder'
CONST_WEIGHTS_FILE_SUFFIX = '_weights.h5'
CONST_SARSA_AGENT_SUFFIX = '_sarsa_agent'

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
    
    def discretize_state(self, state: np.ndarray) -> Union[int, Tuple[int, ...]]:
        """
        Discretiza un estado continuo.
        
        Parámetros:
        -----------
        state : np.ndarray
            Estado continuo del entorno
            
        Retorna:
        --------
        Union[int, Tuple[int, ...]]
            Índice discretizado o tupla de índices discretizados
        """
        if self.discrete_state_space:
            # Si el espacio de estados ya es discreto, convertir a entero
            if isinstance(state, np.ndarray):
                # Si es un array, tomar el primer elemento como índice
                if len(state.shape) > 0:
                    return int(state[0])
                return int(state)
            return int(state)
        
        # Para espacio continuo, discretizar cada dimensión
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
        
        # Verificar el tipo de índice y acceder a la tabla Q de manera adecuada
        if isinstance(discrete_state, tuple):
            # Para espacio de estados multidimensional, usamos indexación con tuplas
            current_q = self.q_table[discrete_state][action]
            next_q = self.q_table[discrete_next_state][next_action]
            
            # Actualizar Q usando la regla SARSA
            new_q = current_q + self.alpha * (reward + self.gamma * next_q - current_q)
            self.q_table[discrete_state][action] = new_q
        else:
            # Para espacio de estados unidimensional, usamos indexación simple
            current_q = self.q_table[discrete_state, action]
            next_q = self.q_table[discrete_next_state, next_action]
            
            # Actualizar Q usando la regla SARSA
            new_q = current_q + self.alpha * (reward + self.gamma * next_q - current_q)
            self.q_table[discrete_state, action] = new_q
    
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
        
        _ = time.time()
        
        # Reiniciar listas para métricas
        self.episode_rewards = []
        self.episode_lengths = []
        self.epsilon_history = []
        
        for episode in range(episodes):
            # Inicializar el episodio
            state, _ = self.env.reset()
            state = np.array(state, dtype=np.float32)
            
            # Seleccionar acción inicial
            action = self.get_action(state)
            
            episode_reward = 0
            steps = 0
            
            # Registrar tiempo de inicio del episodio
            episode_start = time.time()
            
            for _ in range(max_steps):
                # Tomar la acción en el entorno
                next_state, reward, done, _, _ = self.env.step(action)
                next_state = np.array(next_state, dtype=np.float32)
                
                # Renderizar si es necesario
                if render:
                    self.env.render()
                
                # Seleccionar la siguiente acción (política actual)
                next_action = self.get_action(next_state)
                
                # Actualizar la tabla Q usando la regla SARSA
                self.update_q_value(state, action, reward, next_state, next_action)
                
                # Actualizar para el próximo paso
                state = next_state
                action = next_action
                
                # Actualizar métricas
                episode_reward += reward
                steps += 1
                
                if done:
                    break
            
            # Actualizar métricas del episodio
            self.episode_rewards.append(episode_reward)
            self.episode_lengths.append(steps)
            self.epsilon_history.append(self.epsilon)
            
            # Decaer epsilon
            self.decay_epsilon(episode)
            
            # Calcular duración del episodio
            episode_duration = time.time() - episode_start
            
            # Mostrar progreso 
            if episode % self.config['log_interval'] == 0 or episode == episodes - 1:
                print(f"Episodio {episode+1}/{episodes} - Recompensa: {episode_reward:.2f}, Pasos: {steps:.2f}, Epsilon: {self.epsilon:.4f}, Tiempo: {episode_duration:.2f}s")
        
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
            state = np.array(state, dtype=np.float32)
            episode_reward = 0
            episode_steps = 0
            
            done = False
            while not done:
                # Seleccionar mejor acción (sin exploración)
                action = self.get_action(state, explore=False)
                
                # Ejecutar acción en el entorno
                next_state, reward, done, _, _ = self.env.step(action)
                next_state = np.array(next_state, dtype=np.float32)
                
                if render:
                    self.env.render()
                
                # Actualizar estado y métricas
                state = next_state
                episode_reward += reward
                episode_steps += 1
            
            rewards.append(episode_reward)
            steps.append(episode_steps)
            
            if verbose:
                print(f"Episodio de evaluación {episode+1}/{episodes} - Recompensa: {episode_reward:.2f}, Pasos: {episode_steps}")
        
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

@register_keras_serializable()
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
        self.cgm_encoder = Dense(64, activation='relu', name=CONST_CGM_ENCODER)
        self.cgm_pooling = GlobalAveragePooling1D(name='cgm_pooling')
        
        # Capas para procesar otras características
        self.other_encoder = Dense(32, activation='relu', name=CONST_OTHER_ENCODER)
        
        # Capa de concatenación
        self.concat = Concatenate(name='concat_layer')
        
        # Capa para codificar estado discreto
        self.state_encoder = Dense(
            sarsa_agent.state_space_size if hasattr(sarsa_agent, 'state_space_size') else 1000, 
            activation='softmax', 
            name=CONST_STATE_ENCODER
        )
        
        # Capa para convertir acciones discretas a dosis continuas
        self.action_decoder = Dense(1, kernel_initializer='glorot_uniform', name=CONST_ACTION_DECODER)
        
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

        # Codificar estados
        state_encodings = self._encode_states(cgm_input, other_input)
        
        # Obtener acciones discretas para cada muestra usando tf.py_function
        # para encapsular el código Python/NumPy en un op de TensorFlow
        def get_actions(encodings):
            # Esta función se ejecutará en modo eager
            batch_actions = []
            for i in range(len(encodings)):
                # Extraer codificación individual
                encoding = encodings[i]
                
                # Discretizar estado
                discrete_state = self._discretize_state(encoding)
                
                # Obtener mejor acción
                action = np.argmax(self.sarsa_agent.q_table[discrete_state])
                batch_actions.append(action)
            
            return np.array(batch_actions, dtype=np.int32)
        
        # Convertir el código Python en una operación de TensorFlow
        actions = tf.py_function(
            func=get_actions,
            inp=[state_encodings],
            Tout=tf.int32
        )
        
        # Asegurar forma correcta después de tf.py_function
        actions = tf.reshape(actions, [batch_size])
        
        # Decodificar acciones discretas a dosis continuas
        actions_float = tf.cast(actions, tf.float32)
        actions_reshaped = tf.reshape(actions_float, [batch_size, 1])
        
        # Mapear acciones discretas a valores continuos usando el decodificador
        dose_predictions = self.action_decoder(actions_reshaped)
        
        return dose_predictions
    
    def _encode_states(self, cgm_data: tf.Tensor, other_features: tf.Tensor) -> tf.Tensor:
        """
        Codifica los datos de entrada en una representación de estado.
        
        Parámetros:
        -----------
        cgm_data : tf.Tensor
            Datos CGM
        other_features : tf.Tensor
            Otras características
            
        Retorna:
        --------
        tf.Tensor
            Estados codificados
        """
        batch_size = tf.shape(cgm_data)[0]
        
        # Procesar datos CGM
        if len(cgm_data.shape) > 2:  # Si es serie temporal
            # Aplanar la dimensión temporal
            cgm_flattened = tf.reshape(cgm_data, [batch_size, -1])
            cgm_encoded = self.cgm_encoder(cgm_flattened)
        else:  # Si ya es plano
            cgm_encoded = self.cgm_encoder(cgm_data)
        
        # Procesar otras características
        other_encoded = self.other_encoder(other_features)
        
        # Concatenar las características codificadas
        combined = self.concat([cgm_encoded, other_encoded])
        
        # Codificar a un espacio de estados discreto para SARSA
        state_encoded = self.state_encoder(combined)
        
        return state_encoded
    
    def _discretize_state(self, state_encoding: tf.Tensor) -> int:
        """
        Discretiza la codificación de estado para consultar la tabla Q.
        
        Parámetros:
        -----------
        state_encoding : tf.Tensor
            Codificación de estado
            
        Retorna:
        --------
        int
            Índice discretizado para la tabla Q
        """
        # Obtener índice del valor máximo como estado discreto
        discrete_state = tf.argmax(state_encoding).numpy()
        
        # Convertir a entero para indexación
        return int(discrete_state)
    
    def fit(
        self, 
        x: List[tf.Tensor], 
        y: np.ndarray, 
        validation_data: Optional[Tuple] = None, 
        epochs: int = 1,
        batch_size: int = 32,
        callbacks: list = None,
        verbose: int = 0
    ) -> Dict:
        """
        Entrena el modelo utilizando el agente SARSA subyacente.
        
        Parámetros:
        -----------
        x : List[tf.Tensor]
            Lista de tensores de entrada [cgm_data, other_features]
        y : np.ndarray
            Valores objetivo (dosis de insulina)
        validation_data : Optional[Tuple], opcional
            Datos de validación (default: None)
        epochs : int, opcional
            Número de épocas (default: 1)
        batch_size : int, opcional
            Tamaño de lote (default: 32)
        callbacks : list, opcional
            Lista de callbacks (default: None)
        verbose : int, opcional
            Nivel de verbosidad (default: 0)
            
        Retorna:
        --------
        Dict
            Historia de entrenamiento
        """
        # Calibrar el decodificador de acciones para mapear acciones discretas a dosis continuas
        self._calibrate_action_decoder(y)
        
        # Crear entorno de entrenamiento
        train_env = self._create_training_environment(x[0], x[1], y)
        
        # Asignar entorno al agente SARSA
        self.sarsa_agent.env = train_env
        
        # Multiplicar épocas por batch_size para simular épocas completas
        effective_episodes = epochs * (len(y) // batch_size) if batch_size < len(y) else epochs
        
        # Entrenar el agente SARSA
        history = self.sarsa_agent.train(
            episodes=effective_episodes,
            max_steps=1,  # Un paso por episodio es suficiente para este problema
            render=False
        )
        
        # Preparar datos para validación si están disponibles
        if validation_data is not None:
            val_x, val_y = validation_data
            val_env = self._create_training_environment(val_x[0], val_x[1], val_y)
            
            # Guardar el entorno actual
            original_env = self.sarsa_agent.env
            
            # Configurar entorno de validación
            self.sarsa_agent.env = val_env
            
            # Evaluar (sin pasar el entorno como argumento)
            val_reward = self.sarsa_agent.evaluate(
                episodes=len(val_y) // batch_size if batch_size < len(val_y) else 1,
                render=False,
                verbose=verbose > 0
            )
            
            # Restaurar el entorno original
            self.sarsa_agent.env = original_env
            
            print(f"Recompensa de validación: {val_reward:.4f}")
            
            # Añadir métricas de validación al historial
            history['val_reward'] = val_reward
        
        return history
    
    def _calibrate_action_decoder(self, y: np.ndarray) -> None:
        """
        Calibra la capa de decodificación de acciones para mapear acciones discretas a dosis continuas.
        
        Parámetros:
        -----------
        y : np.ndarray
            Valores objetivo (dosis de insulina)
        """
        # Obtener rango de dosis
        min_dose = np.min(y)
        max_dose = np.max(y)
        
        # Determinar espacio de acciones del agente SARSA
        n_actions = self.sarsa_agent.action_space_size
        
        # Calcular pendiente y sesgo para mapeo lineal
        slope = (max_dose - min_dose) / (n_actions - 1)
        intercept = min_dose
        
        # Asegurar que la capa esté inicializada antes de asignar pesos
        dummy_input = tf.zeros((1, 1))
        _ = self.action_decoder(dummy_input)
        
        # Crear pesos para la capa action_decoder
        w = np.array([[slope]], dtype=np.float32)
        b = np.array([intercept], dtype=np.float32)
        
        # Ahora configurar los pesos
        self.action_decoder.set_weights([w, b])
    
    def _create_training_environment(
        self, 
        cgm_data: tf.Tensor, 
        other_features: tf.Tensor, 
        targets: np.ndarray
    ) -> Any:
        """
        Crea un entorno de entrenamiento compatible con el agente SARSA.
        
        Parámetros:
        -----------
        cgm_data : tf.Tensor
            Datos CGM
        other_features : tf.Tensor
            Otras características
        targets : np.ndarray
            Valores objetivo (dosis de insulina)
            
        Retorna:
        --------
        Any
            Entorno compatible con Gym para el agente SARSA
        """
        # Convertir tensores a numpy para procesamiento
        cgm_np = cgm_data.numpy() if hasattr(cgm_data, 'numpy') else cgm_data
        other_np = other_features.numpy() if hasattr(other_features, 'numpy') else other_features
        targets_np = targets.flatten()
        
        # Definir clase de entorno
        class InsulinDosingEnv:
            def __init__(self, cgm, other, targets, model):
                self.cgm = cgm
                self.other = other
                self.targets = targets
                self.model = model
                self.current_idx = 0
                self.max_idx = len(targets) - 1
                self.rng = np.random.Generator(np.random.PCG64(42))
                
                # Definir espacio de acción discreto
                self.action_space = SimpleNamespace(
                    n=self.model.sarsa_agent.action_space_size,
                    sample=lambda: self.rng.integers(0, self.model.sarsa_agent.action_space_size)
                )
                
                # Definir espacio de observación
                flat_dim = np.prod(cgm.shape[1:]) + np.prod(other.shape[1:])
                self.observation_space = SimpleNamespace(
                    shape=(flat_dim,),
                    n=self.model.sarsa_agent.state_space_size if hasattr(self.model.sarsa_agent, 'state_space_size') else None
                )
            
            def reset(self):
                """Reinicia el entorno, elige un ejemplo aleatorio."""
                self.current_idx = self.rng.integers(0, self.max_idx)
                state = self._get_state()
                return state, {}
            
            def step(self, action):
                """Ejecuta un paso con la acción dada."""
                # La acción es un índice discreto, obtener la dosis correspondiente
                action_tensor = tf.convert_to_tensor([[float(action)]], dtype=tf.float32)
                dose = float(self.model.action_decoder(action_tensor).numpy()[0, 0])
                
                # Calcular error y recompensa (negativo del error absoluto)
                target = self.targets[self.current_idx]
                error = abs(dose - target)
                reward = -error  # Recompensa es negativo del error
                
                # Avanzar al siguiente ejemplo (o volver al inicio si es el último)
                self.current_idx = (self.current_idx + 1) % (self.max_idx + 1)
                
                # Obtener siguiente estado
                next_state = self._get_state()
                
                # Para este problema, considerar que un episodio termina después de cada paso
                done = True
                truncated = False
                info = {}
                
                return next_state, reward, done, truncated, info
            
            def _get_state(self):
                """Obtiene el estado actual codificado."""
                # Extraer datos actuales
                current_cgm = self.cgm[self.current_idx:self.current_idx+1]
                current_other = self.other[self.current_idx:self.current_idx+1]
                
                # Codificar estado
                state_encoding = self.model._encode_states(
                    tf.convert_to_tensor(current_cgm, dtype=tf.float32),
                    tf.convert_to_tensor(current_other, dtype=tf.float32)
                )
                
                return state_encoding.numpy()[0]
        
        return InsulinDosingEnv(cgm_np, other_np, targets_np, self)
    
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
        })
        return config
    
    def save_weights(self, filepath: str, **kwargs) -> None:
        """
        Guarda los pesos del modelo y el agente SARSA.
        
        Parámetros:
        -----------
        filepath : str
            Ruta donde guardar los pesos
        **kwargs : dict
            Argumentos adicionales para el método de guardado original
        """
        # Guardar pesos de las capas del wrapper
        super().save_weights(filepath + CONST_WEIGHTS_FILE_SUFFIX, **kwargs)
        
        # Guardar el agente SARSA
        self.sarsa_agent.save(filepath + CONST_SARSA_AGENT_SUFFIX)
        
    def load_weights(self, filepath: str, **kwargs) -> None:
        """
        Carga los pesos del modelo y el agente SARSA.
        
        Parámetros:
        -----------
        filepath : str
            Ruta desde donde cargar los pesos
        **kwargs : dict
            Argumentos adicionales para el método de carga original
        """
        # Cargar pesos de las capas del wrapper
        super().load_weights(filepath + CONST_WEIGHTS_FILE_SUFFIX, **kwargs)
        
        # Cargar el agente SARSA
        self.sarsa_agent.load(filepath + CONST_SARSA_AGENT_SUFFIX)


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
            self.action_space = SimpleNamespace(n=n_actions)
            self.observation_space = SimpleNamespace(n=n_states)
            
        def reset(self):
            return np.zeros(n_states), {}
            
        def step(self, _):
            return np.zeros(n_states), 0.0, True, False, {}
    
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