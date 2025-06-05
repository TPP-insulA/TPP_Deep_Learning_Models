import os, sys
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import time
from types import SimpleNamespace
from typing import Dict, List, Tuple, Optional, Union, Any, Callable
from keras._tf_keras.keras.models import Model
from keras._tf_keras.keras.layers import (
    Input, Dense, LayerNormalization, Dropout, Activation, GlobalAveragePooling1D,
    Flatten, Reshape, Conv1D, LSTM, GRU, BatchNormalization, Concatenate
)
from keras._tf_keras.keras.optimizers import Adam
from keras._tf_keras.keras.saving import register_keras_serializable

PROJECT_ROOT = os.path.abspath(os.getcwd())
sys.path.append(PROJECT_ROOT) 

from config.models_config_old import REINFORCE_CONFIG
from constants.constants import CONST_DEFAULT_SEED, CONST_DEFAULT_EPOCHS, CONST_DEFAULT_BATCH_SIZE

# Constantes para mensajes y valores
CONST_ENTRENANDO = "Entrenando agente REINFORCE..."
CONST_EPISODIO = "Episodio"
CONST_RECOMPENSA = "Recompensa"
CONST_PROMEDIO = "Promedio"
CONST_POLITICA_LOSS = "Pérdida política"
CONST_ENTROPIA = "Entropía"
CONST_EVALUANDO = "Evaluando modelo..."
CONST_RESULTADOS = "Resultados de evaluación:"

class REINFORCEPolicy:
    """
    Implementación de la política para el algoritmo REINFORCE.
    
    Esta clase maneja la distribución de probabilidad sobre acciones
    y proporciona métodos para calcular log-probabilidades y entropía.
    """
    
    def __init__(
        self, 
        state_dim: int, 
        action_dim: int, 
        continuous: bool = False,
        hidden_sizes: List[int] = REINFORCE_CONFIG['hidden_units'],
        activation: str = 'relu',
        learning_rate: float = 0.001,
        log_std_init: float = -0.5
    ) -> None:
        """
        Inicializa la política REINFORCE.
        
        Parámetros:
        -----------
        state_dim : int
            Dimensión del espacio de estados
        action_dim : int
            Dimensión del espacio de acciones
        continuous : bool, opcional
            Si el espacio de acciones es continuo (default: False)
        hidden_sizes : List[int], opcional
            Tamaños de las capas ocultas (default: [64, 32])
        activation : str, opcional
            Función de activación para capas ocultas (default: 'relu')
        learning_rate : float, opcional
            Tasa de aprendizaje (default: 0.001)
        log_std_init : float, opcional
            Valor inicial para el logaritmo de la desviación estándar en caso continuo (default: -0.5)
        """
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.continuous = continuous
        self.hidden_sizes = hidden_sizes
        self.activation = activation
        self.learning_rate = learning_rate
        
        # Construir modelo
        self.model = self._build_model()
        
        # Para acciones continuas, inicializar log_std
        if continuous:
            self.log_std = tf.Variable(
                initial_value=tf.ones(action_dim) * log_std_init,
                trainable=True,
                name="log_std"
            )
    
    def _build_model(self) -> tf.keras.Model:
        """
        Construye la red neuronal para la política.
        
        Retorna:
        --------
        tf.keras.Model
            Modelo de red neuronal
        """
        inputs = Input(shape=(self.state_dim,), name="policy_input")
        x = inputs
        
        # Capas ocultas
        for i, size in enumerate(self.hidden_sizes):
            x = Dense(
                size, 
                activation=self.activation,
                name=f"policy_hidden_{i}"
            )(x)
        
        # Capa de salida
        if self.continuous:
            # Para acciones continuas, predecir la media
            outputs = Dense(
                self.action_dim, 
                activation=None, 
                name="policy_mean"
            )(x)
        else:
            # Para acciones discretas, predecir logits
            outputs = Dense(
                self.action_dim, 
                activation=None,
                name="policy_logits"
            )(x)
        
        return tf.keras.Model(inputs=inputs, outputs=outputs, name="policy_network")
    
    def get_action(self, state: np.ndarray) -> Union[int, np.ndarray]:
        """
        Selecciona una acción basada en el estado actual.
        
        Parámetros:
        -----------
        state : np.ndarray
            Estado actual
            
        Retorna:
        --------
        Union[int, np.ndarray]
            Acción seleccionada (int para discreto, np.ndarray para continuo)
        """
        # Asegurar que el estado tenga la forma correcta
        if len(state.shape) < 2:
            state = np.expand_dims(state, axis=0)
        
        # Obtener salida del modelo
        network_output = self.model(state, training=False).numpy()
        
        if self.continuous:
            # Para acciones continuas, muestrear de distribución normal
            mean = network_output[0]
            std = np.exp(self.log_std.numpy())
            
            # Generador con semilla fija para reproducibilidad
            rng = np.random.Generator(np.random.PCG64(int(time.time())))
            action = mean + rng.normal(size=mean.shape) * std
            
            return action
        else:
            # Para acciones discretas, muestrear categóricamente
            logits = network_output[0]
            
            # Convertir logits a probabilidades
            probs = tf.nn.softmax(logits).numpy()
            
            # Muestrear acción
            rng = np.random.Generator(np.random.PCG64(int(time.time())))
            return rng.choice(self.action_dim, p=probs)
    
    def get_log_prob(self, states: tf.Tensor, actions: tf.Tensor) -> tf.Tensor:
        """
        Calcula log-probabilidades para estados y acciones dados.
        
        Parámetros:
        -----------
        states : tf.Tensor
            Tensor de estados
        actions : tf.Tensor
            Tensor de acciones
            
        Retorna:
        --------
        tf.Tensor
            Log-probabilidades
        """
        # Obtener salida del modelo
        network_output = self.model(states, training=True)
        
        if self.continuous:
            # Para acciones continuas, usar distribución normal
            mean = network_output
            std = tf.exp(self.log_std)
            
            # Calcular log probabilidad para distribución normal
            log_probs = -0.5 * (
                tf.reduce_sum(
                    tf.square((actions - mean) / std) + 
                    2 * self.log_std + 
                    tf.math.log(2 * np.pi), 
                    axis=-1
                )
            )
        else:
            # Para acciones discretas, usar distribución categórica
            logits = network_output
            
            # One-hot encoding para acciones
            if len(actions.shape) == 1:
                actions = tf.one_hot(actions, depth=self.action_dim)
            
            # Log probabilidad para distribución categórica
            log_probs = tf.reduce_sum(
                actions * tf.nn.log_softmax(logits),
                axis=-1
            )
        
        return log_probs
    
    def get_entropy(self, states: tf.Tensor) -> tf.Tensor:
        """
        Calcula la entropía de la política para estados dados.
        
        Parámetros:
        -----------
        states : tf.Tensor
            Tensor de estados
            
        Retorna:
        --------
        tf.Tensor
            Entropía de la política
        """
        network_output = self.model(states, training=True)
        
        if self.continuous:
            # Entropía para distribución normal
            std = tf.exp(self.log_std)
            entropy = 0.5 * tf.reduce_sum(
                tf.math.log(2 * np.pi * std**2) + 1,
                axis=-1
            )
        else:
            # Entropía para distribución categórica
            logits = network_output
            probs = tf.nn.softmax(logits)
            log_probs = tf.nn.log_softmax(logits)
            entropy = -tf.reduce_sum(probs * log_probs, axis=-1)
        
        return entropy
    
    @property
    def trainable_variables(self) -> List[tf.Variable]:
        """
        Obtiene las variables entrenables de la política.
        
        Retorna:
        --------
        List[tf.Variable]
            Lista de variables entrenables
        """
        # Incluir log_std para el caso continuo
        if self.continuous:
            return self.model.trainable_variables + [self.log_std]
        return self.model.trainable_variables

class REINFORCEValueNetwork:
    """
    Red de valor (baseline) para REINFORCE.
    
    Esta red estima el valor esperado de los estados para reducir
    la varianza del estimador del gradiente de política.
    """
    
    def __init__(
        self, 
        state_dim: int, 
        hidden_sizes: List[int] = REINFORCE_CONFIG['hidden_units'],
        activation: str = 'relu',
        learning_rate: float = 0.001
    ) -> None:
        """
        Inicializa la red de valor.
        
        Parámetros:
        -----------
        state_dim : int
            Dimensión del espacio de estados
        hidden_sizes : List[int], opcional
            Tamaños de las capas ocultas (default: [64, 32])
        activation : str, opcional
            Función de activación para capas ocultas (default: 'relu')
        learning_rate : float, opcional
            Tasa de aprendizaje (default: 0.001)
        """
        self.state_dim = state_dim
        self.hidden_sizes = hidden_sizes
        self.activation = activation
        self.learning_rate = learning_rate
        
        # Construir modelo
        self.model = self._build_model()
    
    def _build_model(self) -> tf.keras.Model:
        """
        Construye la red neuronal para la función de valor.
        
        Retorna:
        --------
        tf.keras.Model
            Modelo de red neuronal
        """
        inputs = Input(shape=(self.state_dim,), name="value_input")
        x = inputs
        
        # Capas ocultas
        for i, size in enumerate(self.hidden_sizes):
            x = Dense(
                size, 
                activation=self.activation,
                name=f"value_hidden_{i}"
            )(x)
        
        # Capa de salida (valor escalar)
        outputs = Dense(1, activation=None, name="value_output")(x)
        
        return tf.keras.Model(inputs=inputs, outputs=outputs, name="value_network")
    
    def __call__(self, states: tf.Tensor) -> tf.Tensor:
        """
        Predice valores para estados dados.
        
        Parámetros:
        -----------
        states : tf.Tensor
            Tensor de estados
            
        Retorna:
        --------
        tf.Tensor
            Valores predichos
        """
        return self.model(states, training=True)
    
    @property
    def trainable_variables(self) -> List[tf.Variable]:
        """
        Obtiene las variables entrenables de la red de valor.
        
        Retorna:
        --------
        List[tf.Variable]
            Lista de variables entrenables
        """
        return self.model.trainable_variables

class REINFORCE:
    """
    Implementación del algoritmo REINFORCE (Monte Carlo Policy Gradient).
    
    Este algoritmo aprende una política parametrizada directamente
    maximizando el retorno esperado utilizando ascenso por gradiente.
    """
    
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        continuous: bool = False,
        gamma: float = REINFORCE_CONFIG['gamma'],
        policy_lr: float = REINFORCE_CONFIG['policy_lr'],
        value_lr: float = REINFORCE_CONFIG['value_lr'],
        use_baseline: bool = REINFORCE_CONFIG['use_baseline'],
        entropy_coef: float = REINFORCE_CONFIG['entropy_coef'],
        hidden_sizes: List[int] = REINFORCE_CONFIG['hidden_units'],
        seed: int = CONST_DEFAULT_SEED
    ) -> None:
        """
        Inicializa el agente REINFORCE.
        
        Parámetros:
        -----------
        state_dim : int
            Dimensión del espacio de estados
        action_dim : int
            Dimensión del espacio de acciones
        continuous : bool, opcional
            Si el espacio de acciones es continuo (default: False)
        gamma : float, opcional
            Factor de descuento (default: REINFORCE_CONFIG['gamma'])
        policy_lr : float, opcional
            Tasa de aprendizaje para la política (default: REINFORCE_CONFIG['policy_lr'])
        value_lr : float, opcional
            Tasa de aprendizaje para la red de valor (default: REINFORCE_CONFIG['value_lr'])
        use_baseline : bool, opcional
            Si usar una función de valor como baseline (default: REINFORCE_CONFIG['use_baseline'])
        entropy_coef : float, opcional
            Coeficiente para regularización de entropía (default: REINFORCE_CONFIG['entropy_coef'])
        hidden_sizes : List[int], opcional
            Tamaños de las capas ocultas (default: REINFORCE_CONFIG['hidden_units'])
        seed : int, opcional
            Semilla para reproducibilidad (default: 42)
        """
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.continuous = continuous
        self.gamma = gamma
        self.use_baseline = use_baseline
        self.entropy_coef = entropy_coef
        
        # Fijar semilla para reproducibilidad
        tf.random.set_seed(seed)
        np.random.seed(seed)
        
        # Crear política
        self.policy = REINFORCEPolicy(
            state_dim=state_dim,
            action_dim=action_dim,
            continuous=continuous,
            hidden_sizes=hidden_sizes,
            learning_rate=policy_lr
        )
        
        # Crear red de valor si se usa baseline
        if use_baseline:
            self.value_network = REINFORCEValueNetwork(
                state_dim=state_dim,
                hidden_sizes=hidden_sizes,
                learning_rate=value_lr
            )
        
        # Optimizadores
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=policy_lr)
        if use_baseline:
            self.value_optimizer = tf.keras.optimizers.Adam(learning_rate=value_lr)
        
        # Métricas
        self.policy_loss_metric = tf.keras.metrics.Mean(name="policy_loss")
        self.entropy_metric = tf.keras.metrics.Mean(name="entropy")
        self.returns_metric = tf.keras.metrics.Mean(name="returns")
        if use_baseline:
            self.baseline_loss_metric = tf.keras.metrics.Mean(name="baseline_loss")
        
        # Historiales
        self.episode_rewards = []
        self.avg_rewards = []
        self.policy_losses = []
        self.entropy_values = []
    
    @tf.function
    def train_policy_step(
        self, 
        states: tf.Tensor, 
        actions: tf.Tensor, 
        returns: tf.Tensor
    ) -> Tuple[tf.Tensor, tf.Tensor]:
        """
        Realiza un paso de entrenamiento para la red de política.
        
        Parámetros:
        -----------
        states : tf.Tensor
            Estados visitados
        actions : tf.Tensor
            Acciones tomadas
        returns : tf.Tensor
            Retornos calculados
            
        Retorna:
        --------
        Tuple[tf.Tensor, tf.Tensor]
            Tupla con (pérdida_política, entropía)
        """
        with tf.GradientTape() as tape:
            # Obtener log probabilities de las acciones tomadas
            log_probs = self.policy.get_log_prob(states, actions)
            
            # Calcular entropía para regularización
            entropy = self.policy.get_entropy(states)
            mean_entropy = tf.reduce_mean(entropy, axis=0)
            
            # Si se usa baseline, usar valores como ventaja
            if self.use_baseline:
                values = self.value_network(states)
                values = tf.squeeze(values, axis=-1)
                # Calcular ventaja (returns - valores predichos)
                advantages = returns - values
            else:
                advantages = returns
            
            # Calcular pérdida de política (negativo porque queremos maximizar)
            policy_loss = -tf.reduce_mean(log_probs * advantages, axis=0)
            
            # Agregar término de entropía para fomentar exploración
            loss = policy_loss - self.entropy_coef * mean_entropy
        
        # Calcular gradientes y actualizar política
        grads = tape.gradient(loss, self.policy.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.policy.trainable_variables))
        
        # Actualizar métricas
        self.policy_loss_metric.update_state(policy_loss)
        self.entropy_metric.update_state(mean_entropy)
        self.returns_metric.update_state(tf.reduce_mean(returns, axis=0))
        
        return policy_loss, mean_entropy
    
    @tf.function
    def train_baseline_step(self, states: tf.Tensor, returns: tf.Tensor) -> tf.Tensor:
        """
        Realiza un paso de entrenamiento para la red de valor (baseline).
        
        Parámetros:
        -----------
        states : tf.Tensor
            Estados visitados
        returns : tf.Tensor
            Retornos calculados
            
        Retorna:
        --------
        tf.Tensor
            Pérdida del baseline
        """
        with tf.GradientTape() as tape:
            # Predecir valores
            values = self.value_network(states)
            values = tf.squeeze(values, axis=-1)
            
            # Calcular pérdida MSE
            baseline_loss = tf.reduce_mean(tf.square(returns - values), axis=0)
        
        # Calcular gradientes y actualizar red de valor
        grads = tape.gradient(baseline_loss, self.value_network.trainable_variables)
        self.value_optimizer.apply_gradients(zip(grads, self.value_network.trainable_variables))
        
        # Actualizar métrica
        self.baseline_loss_metric.update_state(baseline_loss)
        
        return baseline_loss
    
    def compute_returns(self, rewards: List[float]) -> np.ndarray:
        """
        Calcula los retornos descontados para cada paso de tiempo.
        
        Parámetros:
        -----------
        rewards : List[float]
            Lista de recompensas recibidas
            
        Retorna:
        --------
        np.ndarray
            Array de retornos descontados
        """
        returns = np.zeros_like(rewards, dtype=np.float32)
        future_return = 0.0
        
        # Calcular retornos desde el final del episodio
        for t in reversed(range(len(rewards))):
            future_return = rewards[t] + self.gamma * future_return
            returns[t] = future_return
        
        # Normalizar retornos para estabilidad
        if len(returns) > 1:
            returns = (returns - np.mean(returns)) / (np.std(returns) + 1e-8)
        
        return returns
    
    def _run_episode(
        self, 
        env: Any, 
        render: bool = False
    ) -> Tuple[List[np.ndarray], List[Union[int, np.ndarray]], List[float], float, int]:
        """
        Ejecuta un episodio completo y recolecta la experiencia.
        
        Parámetros:
        -----------
        env : Any
            Entorno de OpenAI Gym o compatible
        render : bool, opcional
            Si renderizar el entorno durante entrenamiento (default: False)
            
        Retorna:
        --------
        Tuple[List[np.ndarray], List[Union[int, np.ndarray]], List[float], float, int]
            Tupla con (estados, acciones, recompensas, recompensa_total, longitud_episodio)
        """
        state, _ = env.reset()
        done = False
        
        # Almacenar datos del episodio
        states = []
        actions = []
        rewards = []
        
        # Interactuar con el entorno hasta finalizar episodio
        while not done:
            if render:
                env.render()
            
            # Convertir estado a numpy array si no lo es
            if not isinstance(state, np.ndarray):
                state = np.array(state, dtype=np.float32)
            
            # Seleccionar acción según la política actual
            action = self.policy.get_action(state)
            
            # Ejecutar acción en el entorno
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            
            # Guardar transición
            states.append(state)
            actions.append(action)
            rewards.append(reward)
            
            # Actualizar estado
            state = next_state
        
        return states, actions, rewards, sum(rewards), len(rewards)
    
    def _update_networks(
        self, 
        states: List[np.ndarray], 
        actions: List[Union[int, np.ndarray]], 
        rewards: List[float]
    ) -> Tuple[float, float]:
        """
        Actualiza las redes de política y valor.
        
        Parámetros:
        -----------
        states : List[np.ndarray]
            Lista de estados
        actions : List[Union[int, np.ndarray]]
            Lista de acciones
        rewards : List[float]
            Lista de recompensas
            
        Retorna:
        --------
        Tuple[float, float]
            Tupla con (pérdida_política, entropía)
        """
        # Calcular retornos
        returns = self.compute_returns(rewards)
        
        # Convertir a tensores
        states = np.array(states, dtype=np.float32)
        if self.continuous:
            actions = np.array(actions, dtype=np.float32)
        else:
            actions = np.array(actions, dtype=np.int32)
        returns = np.array(returns, dtype=np.float32)
        
        # Actualizar política
        policy_loss, entropy = self.train_policy_step(
            tf.convert_to_tensor(states), 
            tf.convert_to_tensor(actions), 
            tf.convert_to_tensor(returns)
        )
        
        # Actualizar baseline si se usa
        if self.use_baseline:
            self.train_baseline_step(
                tf.convert_to_tensor(states),
                tf.convert_to_tensor(returns)
            )
        
        return policy_loss, entropy
    
    def train(
        self, 
        env: Any, 
        episodes: int = REINFORCE_CONFIG['episodes'],
        render: bool = False,
        log_interval: int = REINFORCE_CONFIG['log_interval'],
        max_steps: int = REINFORCE_CONFIG['max_steps']
    ) -> Dict[str, List[float]]:
        """
        Entrena el agente REINFORCE.
        
        Parámetros:
        -----------
        env : Any
            Entorno de OpenAI Gym o compatible
        episodes : int, opcional
            Número de episodios para entrenar (default: REINFORCE_CONFIG['episodes'])
        render : bool, opcional
            Si renderizar el entorno durante entrenamiento (default: False)
        log_interval : int, opcional
            Cada cuántos episodios mostrar estadísticas (default: REINFORCE_CONFIG['log_interval'])
        max_steps : int, opcional
            Número máximo de pasos por episodio (default: REINFORCE_CONFIG['max_steps_per_episode'])
            
        Retorna:
        --------
        Dict[str, List[float]]
            Historial de entrenamiento
        """
        print(CONST_ENTRENANDO)
        
        # Historial para seguimiento
        history = {
            'episode_rewards': [],
            'avg_rewards': [],
            'policy_losses': [],
            'entropy_values': []
        }
        
        # Ventana para promedio móvil
        reward_window = []
        
        for episode in range(episodes):
            # Ejecutar episodio
            states, actions, rewards, total_reward, _ = self._run_episode(env, render)
            
            # Actualizar redes
            policy_loss, entropy = self._update_networks(states, actions, rewards)
            
            # Seguimiento de métricas
            history['episode_rewards'].append(total_reward)
            history['policy_losses'].append(float(policy_loss))
            history['entropy_values'].append(float(entropy))
            
            # Actualizar ventana de recompensas
            reward_window.append(total_reward)
            if len(reward_window) > 100:  # Mantener ventana de 100 episodios
                reward_window.pop(0)
            
            # Calcular promedio
            avg_reward = np.mean(reward_window)
            history['avg_rewards'].append(avg_reward)
            
            # Mostrar progreso
            if (episode + 1) % log_interval == 0:
                print(f"{CONST_EPISODIO} {episode+1}/{episodes} - "
                      f"{CONST_RECOMPENSA}: {total_reward:.2f}, "
                      f"{CONST_PROMEDIO}: {avg_reward:.2f}, "
                      f"{CONST_POLITICA_LOSS}: {float(policy_loss):.4f}, "
                      f"{CONST_ENTROPIA}: {float(entropy):.4f}")
        
        return history
    
    def evaluate(
        self, 
        env: Any, 
        episodes: int = 10, 
        render: bool = False
    ) -> float:
        """
        Evalúa el agente entrenado.
        
        Parámetros:
        -----------
        env : Any
            Entorno de OpenAI Gym o compatible
        episodes : int, opcional
            Número de episodios para evaluar (default: 10)
        render : bool, opcional
            Si renderizar el entorno durante evaluación (default: False)
            
        Retorna:
        --------
        float
            Recompensa promedio
        """
        print(CONST_EVALUANDO)
        rewards = []
        
        for episode in range(episodes):
            state, _ = env.reset()
            done = False
            episode_reward = 0
            
            while not done:
                if render:
                    env.render()
                
                # Convertir estado a numpy array si no lo es
                if not isinstance(state, np.ndarray):
                    state = np.array(state, dtype=np.float32)
                
                # Seleccionar acción según la política actual (sin exploración)
                if self.continuous:
                    # Para continuo, usar media directamente
                    network_output = self.policy.model(
                        np.expand_dims(state, axis=0), 
                        training=False
                    ).numpy()
                    action = network_output[0]
                else:
                    # Para discreto, elegir acción más probable
                    network_output = self.policy.model(
                        np.expand_dims(state, axis=0), 
                        training=False
                    ).numpy()
                    logits = network_output[0]
                    action = np.argmax(logits)
                
                # Ejecutar acción
                next_state, reward, terminated, truncated, _ = env.step(action)
                done = terminated or truncated
                
                episode_reward += reward
                state = next_state
            
            rewards.append(episode_reward)
            print(f"{CONST_EPISODIO} {episode+1}/{episodes} - {CONST_RECOMPENSA}: {episode_reward:.2f}")
        
        avg_reward = np.mean(rewards)
        print(f"{CONST_RESULTADOS} {CONST_PROMEDIO} {CONST_RECOMPENSA}: {avg_reward:.2f}")
        
        return avg_reward

@register_keras_serializable()
class REINFORCEModel(tf.keras.models.Model):
    """
    Modelo Keras que encapsula el agente REINFORCE para uso con la API de Keras.
    
    Esta clase permite que el algoritmo REINFORCE se integre con el
    flujo de trabajo de entrenamiento de Keras.
    """
    
    def __init__(
        self, 
        reinforce_agent: REINFORCE,
        cgm_shape: Tuple[int, ...],
        other_features_shape: Tuple[int, ...],
        **kwargs
    ) -> None:
        """
        Inicializa el modelo wrapper para REINFORCE.
        
        Parámetros:
        -----------
        reinforce_agent : REINFORCE
            Agente REINFORCE a encapsular
        cgm_shape : Tuple[int, ...]
            Forma de los datos CGM
        other_features_shape : Tuple[int, ...]
            Forma de otras características
        **kwargs
            Argumentos adicionales para tf.keras.models.Model
        """
        super().__init__(**kwargs)
        
        self.reinforce_agent = reinforce_agent
        self.cgm_shape = cgm_shape
        self.other_features_shape = other_features_shape
        
        # Capas de procesamiento de entrada
        self.cgm_flatten = Flatten()
        self.cgm_dense = Dense(64, activation='relu')
        
        # Para otras características
        self.other_features_dense = Dense(32, activation='relu')
        
        # Capa de salida
        self.output_layer = Dense(1)
        
        # Historial de entrenamiento
        self.history = {'loss': [], 'val_loss': []}
    
    def _create_environment(
        self, 
        cgm_data: tf.Tensor, 
        other_features: tf.Tensor, 
        target_doses: tf.Tensor
    ) -> Any:
        """
        Crea un entorno personalizado para entrenamiento.
        
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
        # Convertir a numpy para procesamiento
        cgm_np = cgm_data.numpy() if hasattr(cgm_data, 'numpy') else cgm_data
        other_np = other_features.numpy() if hasattr(other_features, 'numpy') else other_features
        targets_np = target_doses.numpy() if hasattr(target_doses, 'numpy') else target_doses
        
        class InsulinDosingEnv:
            """Entorno para problema de dosificación de insulina."""
            
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
                    shape=(model.reinforce_agent.state_dim,),
                    low=-np.inf,
                    high=np.inf
                )
                
                self.action_space = SimpleNamespace(
                    n=model.reinforce_agent.action_dim if not model.reinforce_agent.continuous else None,
                    shape=(model.reinforce_agent.action_dim,) if model.reinforce_agent.continuous else None,
                    sample=self._sample_action,
                    low=-1.0,
                    high=1.0
                )
            
            def _sample_action(self):
                """Muestrea una acción aleatoria del espacio correspondiente."""
                if self.model.reinforce_agent.continuous:
                    return self.rng.uniform(
                        -1.0, 1.0, 
                        size=(self.model.reinforce_agent.action_dim,)
                    )
                else:
                    return self.rng.integers(0, self.model.reinforce_agent.action_dim)
            
            def reset(self):
                """Reinicia el entorno eligiendo un ejemplo aleatorio."""
                self.current_idx = self.rng.integers(0, self.max_idx)
                
                # Procesar estado
                state = self._get_state()
                
                return state, {}
            
            def step(self, action):
                """Ejecuta un paso con la acción dada."""
                # Convertir acción a dosis
                if self.model.reinforce_agent.continuous:
                    # Mapear acción continua [-1,1] a rango de dosis [0, max_dose]
                    max_dose = np.max(self.targets) * 1.2
                    dose = (action[0] + 1) * max_dose / 2.0
                else:
                    # Mapear acción discreta a rango de dosis
                    max_dose = np.max(self.targets) * 1.2
                    dose = action * max_dose / (self.model.reinforce_agent.action_dim - 1)
                
                # Calcular recompensa como negativo del error absoluto
                target = self.targets[self.current_idx]
                error = np.abs(dose - target)
                
                # Función de recompensa que prioriza precisión
                if error < 0.5:
                    reward = 1.0 - error  # Recompensa alta para error bajo
                else:
                    reward = -error  # Penalización para errores grandes
                
                # Avanzar al siguiente ejemplo
                self.current_idx = (self.current_idx + 1) % self.max_idx
                
                # Obtener próximo estado
                next_state = self._get_state()
                
                # Episodio termina después de un paso
                done = True
                
                return next_state, reward, done, False, {'dose': dose, 'target': target}
            
            def _get_state(self):
                """Obtiene el estado actual procesado."""
                # Extraer datos CGM y otras características para el índice actual
                current_cgm = self.cgm[self.current_idx:self.current_idx+1]
                current_features = self.features[self.current_idx:self.current_idx+1]
                
                # Procesar con las capas del modelo
                cgm_flat = self.model.cgm_flatten(current_cgm)
                cgm_encoded = self.model.cgm_dense(cgm_flat)
                
                other_encoded = self.model.other_features_dense(current_features)
                
                # Concatenar características
                combined = tf.concat([cgm_encoded, other_encoded], axis=1)
                
                # Devolver numpy array
                return combined.numpy()[0]
        
        return InsulinDosingEnv(cgm_np, other_np, targets_np, self)
    
    def call(self, inputs: List[tf.Tensor], training: bool = False) -> tf.Tensor:
        """
        Procesa las entradas para generar predicciones.
        
        Parámetros:
        -----------
        inputs : List[tf.Tensor]
            Lista con [cgm_data, other_features]
        training : bool, opcional
            Si está en modo entrenamiento (default: False)
            
        Retorna:
        --------
        tf.Tensor
            Predicciones del modelo
        """
        cgm_data, other_features = inputs
        batch_size = tf.shape(cgm_data)[0]
        
        # Procesar entradas
        cgm_flat = self.cgm_flatten(cgm_data)
        cgm_encoded = self.cgm_dense(cgm_flat)
        
        other_encoded = self.other_features_dense(other_features)
        
        # Concatenar características
        combined = tf.concat([cgm_encoded, other_encoded], axis=1)
        
        # Obtener predicciones para cada muestra en el lote
        predictions = tf.TensorArray(tf.float32, size=batch_size)
        
        for i in range(batch_size):
            # Extraer estado para esta muestra
            state = combined[i]
            
            # Usar política para predecir
            if self.reinforce_agent.continuous:
                action = self.reinforce_agent.policy.model(
                    tf.expand_dims(state, axis=0), 
                    training=False
                )[0]
                
                # Convertir acción a dosis
                max_dose = 15.0  # Valor típico máximo de dosis
                pred = (action[0] + 1) * max_dose / 2.0
                
            else:
                logits = self.reinforce_agent.policy.model(
                    tf.expand_dims(state, axis=0), 
                    training=False
                )[0]
                action = tf.argmax(logits)
                
                # Convertir acción a dosis
                max_dose = 15.0  # Valor típico máximo de dosis
                pred = tf.cast(action, tf.float32) * max_dose / (self.reinforce_agent.action_dim - 1)
            
            predictions = predictions.write(i, pred)
        
        return predictions.stack()
    
    def fit(
        self, 
        x: List[tf.Tensor], 
        y: tf.Tensor, 
        epochs: int = CONST_DEFAULT_EPOCHS,
        batch_size: int = CONST_DEFAULT_BATCH_SIZE,
        callbacks: List = None,
        validation_data: Optional[Tuple] = None,
        verbose: int = 0
    ) -> Dict:
        """
        Entrena el modelo con los datos proporcionados.
        
        Parámetros:
        -----------
        x : List[tf.Tensor]
            Lista con [cgm_data, other_features]
        y : tf.Tensor
            Valores objetivo (dosis)
        epochs : int, opcional
            Número de épocas (default: 10)
        batch_size : int, opcional
            Tamaño del lote (default: 32)
        callbacks : List, opcional
            Lista de callbacks (default: None)
        validation_data : Optional[Tuple], opcional
            Datos de validación (default: None)
        verbose : int, opcional
            Nivel de verbosidad (default: 0)
            
        Retorna:
        --------
        Dict
            Historia de entrenamiento
        """
        # Extraer datos
        cgm_data, other_features = x
        
        # Crear entorno para entrenamiento
        env = self._create_environment(cgm_data, other_features, y)
        
        # Entrenar agente REINFORCE
        history = self.reinforce_agent.train(
            env=env,
            episodes=epochs * (len(y) // batch_size or 1),
            render=False,
            log_interval=max(1, (epochs * (len(y) // batch_size)) // 10)
        )
        
        # Guardar historial para compatibilidad con Keras
        self.history = {
            'loss': history['policy_losses'],
            'val_loss': []
        }
        
        return self.history
    
    def predict(self, x: List[tf.Tensor], **kwargs) -> np.ndarray:
        """
        Realiza predicciones con el modelo entrenado.
        
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
            'other_features_shape': self.other_features_shape
        })
        return config

def create_reinforce_mcpg_model(cgm_shape: Tuple[int, ...], other_features_shape: Tuple[int, ...]) -> tf.keras.models.Model:
    """
    Crea un modelo REINFORCE para predicción de dosis de insulina.
    
    Parámetros:
    -----------
    cgm_shape : Tuple[int, ...]
        Forma de los datos CGM (batch_size, time_steps, features) o (batch_size, features)
    other_features_shape : Tuple[int, ...]
        Forma de otras características (batch_size, n_features)
        
    Retorna:
    --------
    tf.keras.models.Model
        Modelo REINFORCE que implementa la interfaz de Keras
    """
    # Definir dimensiones de las capas de codificación
    # Estas deben coincidir con las dimensiones en REINFORCEModel.__init__
    cgm_encoded_dim = 64
    other_encoded_dim = 32
    state_dim = cgm_encoded_dim + other_encoded_dim  # = 96
    
    # Configuración para espacio de acciones
    if REINFORCE_CONFIG['continuous']:
        action_dim = 1  # Una dimensión para dosis continua
    else:
        action_dim = 20  # 20 niveles discretos de dosis
    
    # Crear agente REINFORCE
    reinforce_agent = REINFORCE(
        state_dim=state_dim,  # Usar la dimensión correcta del estado
        action_dim=action_dim,
        continuous=REINFORCE_CONFIG['continuous'],
        gamma=REINFORCE_CONFIG['gamma'],
        policy_lr=REINFORCE_CONFIG['policy_lr'],
        value_lr=REINFORCE_CONFIG['value_lr'],
        use_baseline=REINFORCE_CONFIG['use_baseline'],
        entropy_coef=REINFORCE_CONFIG['entropy_coef'],
        hidden_sizes=REINFORCE_CONFIG['hidden_units'],
        seed=REINFORCE_CONFIG.get('seed', CONST_DEFAULT_SEED)
    )
    
    # Crear y devolver el modelo wrapper
    return REINFORCEModel(
        reinforce_agent=reinforce_agent,
        cgm_shape=cgm_shape,
        other_features_shape=other_features_shape
    )