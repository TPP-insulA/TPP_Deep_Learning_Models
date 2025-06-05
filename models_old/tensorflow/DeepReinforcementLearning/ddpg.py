import os, sys
import tensorflow as tf
import numpy as np
from keras._tf_keras.keras.models import Model
from keras._tf_keras.keras.layers import (
    Input, Dense, Conv1D, Flatten, Concatenate,
    BatchNormalization, Dropout, LayerNormalization, GlobalAveragePooling1D
)
from keras._tf_keras.keras.optimizers import Adam
from keras._tf_keras.keras.saving import register_keras_serializable
from collections import deque
import random
import gym
from typing import Dict, List, Tuple, Any, Optional, Callable, Union
import matplotlib.pyplot as plt
from types import SimpleNamespace

PROJECT_ROOT = os.path.abspath(os.getcwd())
sys.path.append(PROJECT_ROOT) 

from config.models_config_old import DDPG_CONFIG
from constants.constants import CONST_DEFAULT_SEED, CONST_DEFAULT_EPOCHS, CONST_DEFAULT_BATCH_SIZE
from custom.printer import print_debug

# Constantes para evitar duplicación
ACTOR_PREFIX = 'actor'
CRITIC_PREFIX = 'critic'
MODEL_SUFFIX = '_model.h5'
ACTOR_SUFFIX = '_actor.weights.h5'
CRITIC_SUFFIX = '_critic.weights.h5'
CONST_WEIGHTS_SUFFIX = '.weights.h5'

class ReplayBuffer:
    """
    Buffer de experiencias para el algoritmo DDPG.
    
    Almacena transiciones (state, action, reward, next_state, done)
    y permite muestrear lotes de manera aleatoria para el entrenamiento.
    
    Parámetros:
    -----------
    capacity : int, opcional
        Capacidad máxima del buffer (default: 100000)
    """
    def __init__(self, capacity: int = 100000):
        self.buffer = deque(maxlen=capacity)
    
    def add(self, state: np.ndarray, action: np.ndarray, reward: float, 
           next_state: np.ndarray, done: float) -> None:
        """
        Añade una transición al buffer.
        
        Parámetros:
        -----------
        state : np.ndarray
            Estado actual
        action : np.ndarray
            Acción tomada
        reward : float
            Recompensa recibida
        next_state : np.ndarray
            Estado siguiente
        done : float
            Indicador de fin de episodio (1.0 si terminó, 0.0 si no)
        """
        self.buffer.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Muestrea un lote aleatorio de transiciones.
        
        Parámetros:
        -----------
        batch_size : int
            Tamaño del lote a muestrear
            
        Retorna:
        --------
        Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]
            (states, actions, rewards, next_states, dones)
        """
        if len(self.buffer) < batch_size:
            # Si no hay suficientes transiciones, devuelve lo que haya
            batch = random.sample(self.buffer, len(self.buffer))
        else:
            batch = random.sample(self.buffer, batch_size)
            
        states, actions, rewards, next_states, dones = zip(*batch)
        
        return (
            np.array(states, dtype=np.float32),
            np.array(actions, dtype=np.float32),
            np.array(rewards, dtype=np.float32),
            np.array(next_states, dtype=np.float32),
            np.array(dones, dtype=np.float32)
        )
    
    def __len__(self) -> int:
        """
        Retorna la cantidad de transiciones almacenadas.
        
        Retorna:
        --------
        int
            Número de transiciones en el buffer
        """
        return len(self.buffer)


class OUActionNoise:
    """
    Implementa el proceso de ruido de Ornstein-Uhlenbeck para exploración.
    
    Este ruido añade correlación temporal a las acciones para una exploración más efectiva
    en espacios continuos.
    
    Parámetros:
    -----------
    mean : np.ndarray
        Valor medio al que tiende el proceso
    std_deviation : np.ndarray
        Desviación estándar del ruido
    theta : float, opcional
        Velocidad de reversión a la media (default: 0.15)
    dt : float, opcional
        Delta de tiempo para la discretización (default: 1e-2)
    x_initial : Optional[np.ndarray], opcional
        Valor inicial del proceso (default: None)
    seed : int, opcional
        Semilla para la generación de números aleatorios (default: 42)
    """
    def __init__(self, mean: np.ndarray, std_deviation: np.ndarray, theta: float = 0.15, 
                dt: float = 1e-2, x_initial: Optional[np.ndarray] = None, seed: int = CONST_DEFAULT_SEED):
        self.theta = theta
        self.mean = mean
        self.std_dev = std_deviation
        self.dt = dt
        self.x_initial = x_initial
        self.seed = seed
        self.rng = np.random.default_rng(seed)
        self.reset()
        
    def __call__(self) -> np.ndarray:
        """
        Genera un nuevo valor de ruido siguiendo el proceso de Ornstein-Uhlenbeck.
        
        Retorna:
        --------
        np.ndarray
            Valor de ruido generado
        """
        # Fórmula para el proceso de Ornstein-Uhlenbeck
        x = (
            self.x_prev
            + self.theta * (self.mean - self.x_prev) * self.dt
            + self.std_dev * np.sqrt(self.dt) * self.rng.normal(size=self.mean.shape)
        )
        self.x_prev = x
        return x
    
    def reset(self) -> None:
        """
        Reinicia el estado del ruido.
        """
        if self.x_initial is not None:
            self.x_prev = self.x_initial
        else:
            self.x_prev = np.zeros_like(self.mean)


def activation_function(x: tf.Tensor, name: str) -> tf.Tensor:
    """
    Aplica una función de activación según su nombre.
    
    Parámetros:
    -----------
    x : tf.Tensor
        Tensor de entrada
    name : str
        Nombre de la activación ('relu', 'tanh', 'leaky_relu', etc.)
        
    Retorna:
    --------
    tf.Tensor
        Tensor con activación aplicada
    """
    if name == 'relu':
        return tf.nn.relu(x)
    elif name == 'tanh':
        return tf.nn.tanh(x)
    elif name == 'leaky_relu':
        return tf.nn.leaky_relu(x)
    elif name == 'gelu':
        return tf.nn.gelu(x)
    else:
        # Por defecto usar ReLU
        return tf.nn.relu(x)


class ActorNetwork(Model):
    """
    Red de Actor para DDPG que mapea estados a acciones determinísticas.
    
    Parámetros:
    -----------
    state_dim : int
        Dimensión del espacio de estados
    action_dim : int
        Dimensión del espacio de acciones
    action_high : np.ndarray
        Límite superior del rango de acciones
    action_low : np.ndarray
        Límite inferior del rango de acciones
    hidden_units : Optional[List[int]], opcional
        Unidades en capas ocultas (default: None, usa configuración por defecto)
    """
    def __init__(self, state_dim: int, action_dim: int, action_high: np.ndarray, 
                action_low: np.ndarray, hidden_units: Optional[List[int]] = None) -> None:
        super().__init__()
        
        # Valores predeterminados para capas ocultas
        if hidden_units is None:
            hidden_units = DDPG_CONFIG['actor_hidden_units']
        
        self.action_high = action_high
        self.action_low = action_low
        self.action_range = action_high - action_low
        
        # Capas para el procesamiento del estado
        self.layers_list = []
        for i, units in enumerate(hidden_units):
            self.layers_list.append(Dense(
                units, 
                name=f'actor_dense_{i}'
            ))
            self.layers_list.append(LayerNormalization(
                epsilon=DDPG_CONFIG['epsilon'],
                name=f'actor_ln_{i}'
            ))
            self.layers_list.append(Dropout(
                DDPG_CONFIG['dropout_rate'],
                name=f'actor_dropout_{i}'
            ))
        
        # Capa de salida: acciones con activación tanh [-1, 1] escalada al rango de acción
        self.output_layer = Dense(
            action_dim, 
            activation='tanh',
            name='actor_output'
        )
    
    def call(self, inputs: tf.Tensor, training: bool = False) -> tf.Tensor:
        """
        Pasa la entrada por la red del actor.
        
        Parámetros:
        -----------
        inputs : tf.Tensor
            Estado de entrada
        training : bool, opcional
            Indica si está en modo entrenamiento (default: False)
            
        Retorna:
        --------
        tf.Tensor
            Acción determinística
        """
        x = inputs
        
        # Procesar a través de capas ocultas
        for layer in self.layers_list:
            # Aplicar activación antes de capas normalization/dropout
            if isinstance(layer, Dense):
                x = layer(x)
                x = activation_function(x, DDPG_CONFIG['actor_activation'])
            # Para capas de Dropout, pasamos el parámetro training
            elif isinstance(layer, Dropout):
                x = layer(x, training=training)
            else:
                x = layer(x)
        
        # Capa de salida con activación tanh y escalado
        raw_actions = self.output_layer(x)
        
        # Escalar desde [-1, 1] al rango de acción [low, high]
        scaled_actions = 0.5 * (raw_actions + 1.0) * self.action_range + self.action_low
        
        return scaled_actions


class CriticNetwork(Model):
    """
    Red de Crítico para DDPG que mapea pares (estado, acción) a valores-Q.
    
    Parámetros:
    -----------
    state_dim : int
        Dimensión del espacio de estados
    action_dim : int
        Dimensión del espacio de acciones
    hidden_units : Optional[List[int]], opcional
        Unidades en capas ocultas (default: None, usa configuración por defecto)
    """
    def __init__(self, state_dim: int, action_dim: int, 
                hidden_units: Optional[List[int]] = None) -> None:
        super().__init__()
        
        # Valores predeterminados para capas ocultas
        if hidden_units is None:
            hidden_units = DDPG_CONFIG['critic_hidden_units']
        
        # Capas iniciales para procesar el estado
        self.state_layers = []
        for i, units in enumerate(hidden_units[:1]):  # Primera capa solo para estado
            self.state_layers.append(Dense(
                units,
                name=f'critic_state_dense_{i}'
            ))
            self.state_layers.append(LayerNormalization(
                epsilon=DDPG_CONFIG['epsilon'],
                name=f'critic_state_ln_{i}'
            ))
        
        # Capas para procesar la combinación de estado y acción
        self.combined_layers = []
        for i, units in enumerate(hidden_units[1:]):
            self.combined_layers.append(Dense(
                units,
                name=f'critic_combined_dense_{i}'
            ))
            self.combined_layers.append(LayerNormalization(
                epsilon=DDPG_CONFIG['epsilon'],
                name=f'critic_combined_ln_{i}'
            ))
            self.combined_layers.append(Dropout(
                DDPG_CONFIG['dropout_rate'],
                name=f'critic_dropout_{i}'
            ))
        
        # Capa de salida: valor Q (sin activación)
        self.output_layer = Dense(1, name='critic_output')
    
    def call(self, inputs: Tuple[tf.Tensor, tf.Tensor], training: bool = False) -> tf.Tensor:
        """
        Pasa la entrada por la red del crítico.
        
        Parámetros:
        -----------
        inputs : Tuple[tf.Tensor, tf.Tensor]
            Tupla de (estados, acciones)
        training : bool, opcional
            Indica si está en modo entrenamiento (default: False)
            
        Retorna:
        --------
        tf.Tensor
            Valores Q estimados
        """
        states, actions = inputs
        
        # Procesar el estado
        x = states
        for i, layer in enumerate(self.state_layers):
            if isinstance(layer, Dense):
                x = layer(x)
                x = activation_function(x, DDPG_CONFIG['critic_activation'])
            else:
                x = layer(x)
        
        # Combinar estado procesado con acción
        x = Concatenate()([x, actions])
        
        # Procesar a través de capas combinadas
        for layer in self.combined_layers:
            if isinstance(layer, Dense):
                x = layer(x)
                x = activation_function(x, DDPG_CONFIG['critic_activation'])
            elif isinstance(layer, Dropout):
                x = layer(x, training=training)
            else:
                x = layer(x)
        
        # Capa de salida
        q_value = self.output_layer(x)
        
        return q_value


class DDPG:
    """
    Implementación del algoritmo Deep Deterministic Policy Gradient (DDPG).
    
    DDPG combina ideas de DQN y métodos de policy gradient para manejar
    espacios de acción continuos con políticas determinísticas.
    
    Parámetros:
    -----------
    state_dim : int
        Dimensión del espacio de estados
    action_dim : int
        Dimensión del espacio de acciones
    action_high : np.ndarray
        Límite superior del rango de acciones
    action_low : np.ndarray
        Límite inferior del rango de acciones
    config : Optional[Dict[str, Any]], opcional
        Configuración personalizada (default: None, usa configuración por defecto)
    seed : int, opcional
        Semilla para generación de números aleatorios (default: 42)
    """
    def __init__(
        self, 
        state_dim: int, 
        action_dim: int,
        action_high: np.ndarray,
        action_low: np.ndarray,
        config: Optional[Dict[str, Any]] = None,
        seed: int = CONST_DEFAULT_SEED
    ):
        # Use provided config or default
        self.config = config or DDPG_CONFIG
        
        # Extract hyperparameters from config
        actor_lr = self.config.get('actor_lr', DDPG_CONFIG['actor_lr'])
        critic_lr = self.config.get('critic_lr', DDPG_CONFIG['critic_lr'])
        self.gamma = self.config.get('gamma', DDPG_CONFIG['gamma'])
        self.tau = self.config.get('tau', DDPG_CONFIG['tau'])
        buffer_capacity = self.config.get('buffer_capacity', DDPG_CONFIG['buffer_capacity'])
        self.batch_size = self.config.get('batch_size', DDPG_CONFIG['batch_size'])
        noise_std = self.config.get('noise_std', DDPG_CONFIG['noise_std'])
        actor_hidden_units = self.config.get('actor_hidden_units')
        critic_hidden_units = self.config.get('critic_hidden_units')
        
        # Parámetros del entorno y del modelo
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.action_high = action_high
        self.action_low = action_low
        
        # Configurar semilla para reproducibilidad
        tf.random.set_seed(seed)
        np.random.seed(seed)
        random.seed(seed)
        
        # Valores predeterminados para capas ocultas
        self.actor_hidden_units = actor_hidden_units or DDPG_CONFIG['actor_hidden_units']
        self.critic_hidden_units = critic_hidden_units or DDPG_CONFIG['critic_hidden_units']
        
        # Crear modelos Actor y Crítico
        self.actor = ActorNetwork(
            state_dim=state_dim,
            action_dim=action_dim,
            action_high=action_high,
            action_low=action_low,
            hidden_units=self.actor_hidden_units
        )
        
        self.critic = CriticNetwork(
            state_dim=state_dim,
            action_dim=action_dim,
            hidden_units=self.critic_hidden_units
        )
        
        # Crear copias target
        self.target_actor = ActorNetwork(
            state_dim=state_dim,
            action_dim=action_dim,
            action_high=action_high,
            action_low=action_low,
            hidden_units=self.actor_hidden_units
        )
        
        self.target_critic = CriticNetwork(
            state_dim=state_dim,
            action_dim=action_dim,
            hidden_units=self.critic_hidden_units
        )
        
        # Asegurar que todos los modelos tienen forma inicializada
        dummy_state = np.zeros((1, state_dim), dtype=np.float32)
        dummy_action = np.zeros((1, action_dim), dtype=np.float32)
        
        self.actor(dummy_state)
        self.critic([dummy_state, dummy_action])
        self.target_actor(dummy_state)
        self.target_critic([dummy_state, dummy_action])
        
        # Sincronizar pesos iniciales
        self.target_actor.set_weights(self.actor.get_weights())
        self.target_critic.set_weights(self.critic.get_weights())
        
        # Optimizadores
        self.actor_optimizer = Adam(learning_rate=actor_lr)
        self.critic_optimizer = Adam(learning_rate=critic_lr)
        
        # Buffer de experiencias
        self.replay_buffer = ReplayBuffer(buffer_capacity)
        
        # Ruido para exploración
        self.noise = OUActionNoise(
            mean=np.zeros(action_dim),
            std_deviation=noise_std * np.ones(action_dim),
            seed=seed
        )
        
        # Contador de pasos para actualización y métricas
        self.step_counter = 0
        self.actor_loss_sum = 0
        self.critic_loss_sum = 0
        self.q_value_sum = 0
        self.updates_count = 0
    
    def update_target_networks(self, tau: Optional[float] = None) -> None:
        """
        Actualiza los pesos de las redes target usando soft update.
        
        Parámetros:
        -----------
        tau : Optional[float], opcional
            Factor de actualización suave (si None, usa el valor por defecto)
        """
        if tau is None:
            tau = self.tau
            
        # Actualizar pesos del actor target
        actor_weights = self.actor.get_weights()
        target_actor_weights = self.target_actor.get_weights()
        new_target_actor_weights = []
        
        for i, weight in enumerate(actor_weights):
            new_target_actor_weights.append(tau * weight + (1 - tau) * target_actor_weights[i])
            
        self.target_actor.set_weights(new_target_actor_weights)
        
        # Actualizar pesos del crítico target
        critic_weights = self.critic.get_weights()
        target_critic_weights = self.target_critic.get_weights()
        new_target_critic_weights = []
        
        for i, weight in enumerate(critic_weights):
            new_target_critic_weights.append(tau * weight + (1 - tau) * target_critic_weights[i])
            
        self.target_critic.set_weights(new_target_critic_weights)
    
    def get_action(self, state: np.ndarray, add_noise: bool = True) -> np.ndarray:
        """
        Obtiene una acción determinística para un estado, opcionalmente añadiendo ruido.
        
        Parámetros:
        -----------
        state : np.ndarray
            Estado actual
        add_noise : bool, opcional
            Si se debe añadir ruido para exploración
            
        Retorna:
        --------
        np.ndarray
            Acción seleccionada
        """
        state_tensor = tf.convert_to_tensor(state[np.newaxis, :], dtype=tf.float32)
        action = self.actor(state_tensor, training=False)[0].numpy()
        
        if add_noise:
            noise = self.noise()
            action += noise
            
        # Clipear al rango válido de acciones
        action = np.clip(action, self.action_low, self.action_high)
        
        return action
    
    @tf.function(input_signature=[
        tf.TensorSpec(shape=(None, None), dtype=tf.float32),  # states
        tf.TensorSpec(shape=(None, None), dtype=tf.float32),  # actions
        tf.TensorSpec(shape=(None, None), dtype=tf.float32),  # rewards
        tf.TensorSpec(shape=(None, None), dtype=tf.float32),  # next_states
        tf.TensorSpec(shape=(None, None), dtype=tf.float32),  # dones
        tf.TensorSpec(shape=(), dtype=tf.float32)  # gamma
    ])
    def _update_networks(self, states: tf.Tensor, actions: tf.Tensor, rewards: tf.Tensor, 
                        next_states: tf.Tensor, dones: tf.Tensor, gamma: float) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
        """
        Actualiza las redes del actor y crítico usando el algoritmo DDPG.
        
        Parámetros:
        -----------
        states : tf.Tensor
            Estados actuales
        actions : tf.Tensor
            Acciones tomadas
        rewards : tf.Tensor
            Recompensas recibidas
        next_states : tf.Tensor
            Estados siguientes
        dones : tf.Tensor
            Indicadores de finalización
        gamma : float
            Factor de descuento para recompensas futuras
            
        Retorna:
        --------
        Tuple[tf.Tensor, tf.Tensor, tf.Tensor]
            (critic_loss, actor_loss, q_values)
        """
        with tf.GradientTape() as tape:
            # Predecir acciones target para los siguiente estados
            target_actions = self.target_actor(next_states, training=False)
            
            # Predecir Q-values target
            target_q_values = self.target_critic([next_states, target_actions], training=False)
            
            # Calcular Q-values objetivo usando la ecuación de Bellman
            targets = rewards + (1 - dones) * gamma * target_q_values
            
            # Predecir Q-values actuales
            current_q_values = self.critic([states, actions], training=True)
            
            # Calcular pérdida del crítico (error cuadrático medio)
            critic_loss = tf.reduce_mean(tf.square(targets - current_q_values), axis=0)
            
        # Calcular gradientes y actualizar crítico
        critic_gradients = tape.gradient(critic_loss, self.critic.trainable_variables)
        self.critic_optimizer.apply_gradients(zip(critic_gradients, self.critic.trainable_variables))
        
        with tf.GradientTape() as tape:
            # Predecir acciones para los estados actuales
            actor_actions = self.actor(states, training=True)
            
            # Calcular Q-values para estas acciones
            actor_q_values = self.critic([states, actor_actions], training=False)
            
            # Pérdida del actor (negativo del Q-value promedio)
            # Queremos maximizar Q-value, así que minimizamos su negativo
            actor_loss = -tf.reduce_mean(actor_q_values, axis=0)
            
        # Calcular gradientes y actualizar actor
        actor_gradients = tape.gradient(actor_loss, self.actor.trainable_variables)
        self.actor_optimizer.apply_gradients(zip(actor_gradients, self.actor.trainable_variables))
        
        # Retornar las pérdidas y el valor Q para seguimiento
        return critic_loss, actor_loss, current_q_values
        
    def train_step(self) -> Tuple[float, float, float]:
        """
        Realiza un paso de entrenamiento DDPG con un lote de experiencias.
        
        Retorna:
        --------
        Tuple[float, float, float]
            (critic_loss, actor_loss, q_value)
        """
        # Muestrear un lote del buffer
        states, actions, rewards, next_states, dones = self.replay_buffer.sample(self.batch_size)
        
        # Convertir a tensores para TensorFlow
        states = tf.convert_to_tensor(states, dtype=tf.float32)
        actions = tf.convert_to_tensor(actions, dtype=tf.float32)
        rewards = tf.convert_to_tensor(rewards, dtype=tf.float32)
        next_states = tf.convert_to_tensor(next_states, dtype=tf.float32)
        dones = tf.convert_to_tensor(dones, dtype=tf.float32)
        
        # Reshape rewards y dones para que coincidan con la dimensión de Q-values
        rewards = tf.reshape(rewards, (-1, 1))
        dones = tf.reshape(dones, (-1, 1))
        
        # Convertir gamma a tensor constante para tf.function
        gamma_tensor = tf.constant(self.gamma, dtype=tf.float32)
        
        # Actualizar redes
        critic_loss, actor_loss, q_value = self._update_networks(states, actions, rewards, next_states, dones, gamma_tensor)
        
        # Actualizar redes target con soft update
        self.update_target_networks()
        
        # Convertir loss y q_value de tensor a float de Python
        critic_loss_float = float(critic_loss.numpy())
        actor_loss_float = float(actor_loss.numpy())
        q_value_float = float(q_value.numpy())

        # Actualizar métricas acumuladas usando los valores float 
        self.actor_loss_sum += actor_loss_float
        self.critic_loss_sum += critic_loss_float
        self.q_value_sum += q_value_float
        self.updates_count += 1

        return critic_loss_float, actor_loss_float, q_value_float
    
    def _select_action(self, state: np.ndarray, step_counter: int, warmup_steps: int) -> np.ndarray:
        """
        Selecciona una acción basada en el estado actual y fase de entrenamiento.
        
        Parámetros:
        -----------
        state : np.ndarray
            Estado actual
        step_counter : int
            Contador de pasos
        warmup_steps : int
            Pasos iniciales de exploración pura
            
        Retorna:
        --------
        np.ndarray
            Acción seleccionada
        """
        if step_counter < warmup_steps:
            # Exploración inicial uniforme - usando el generador np.random.default_rng con semilla
            # para asegurar reproducibilidad
            rng = np.random.default_rng(seed=CONST_DEFAULT_SEED)  # Using a default seed
            return rng.uniform(
                low=self.action_low,
                high=self.action_high,
                size=(self.action_dim,)
            )
        else:
            # Política del actor con ruido
            return self.get_action(state, add_noise=True)
    
    def _update_model(self, step_counter: int, update_every: int) -> Tuple[Optional[float], Optional[float]]:
        """
        Actualiza los modelos si es momento de hacerlo.
        
        Parámetros:
        -----------
        step_counter : int
            Contador de pasos totales
        update_every : int
            Frecuencia de actualización
            
        Retorna:
        --------
        Tuple[Optional[float], Optional[float]]
            (actor_loss, critic_loss) o (None, None) si no se actualizó
        """
        if step_counter % update_every == 0:
            if len(self.replay_buffer) > self.batch_size:
                # Realizar actualización
                critic_loss, actor_loss, _ = self.train_step()
                return actor_loss, critic_loss
        return None, None
    
    def _update_history(self, history: Dict, episode_reward: float, 
                      episode_actor_loss: List, episode_critic_loss: List) -> Dict:
        """
        Actualiza el historial de entrenamiento con los resultados del episodio.
        
        Parámetros:
        -----------
        history : Dict
            Historial de entrenamiento a actualizar
        episode_reward : float
            Recompensa del episodio
        episode_actor_loss : List
            Lista de pérdidas del actor durante el episodio
        episode_critic_loss : List
            Lista de pérdidas del crítico durante el episodio
            
        Retorna:
        --------
        Dict
            Historial actualizado
        """
        history['episode_rewards'].append(episode_reward)
        
        # Obtener promedios de métricas
        if self.updates_count > 0:
            actor_loss_avg = self.actor_loss_sum / self.updates_count
            critic_loss_avg = self.critic_loss_sum / self.updates_count
            q_value_avg = self.q_value_sum / self.updates_count
            
            # Resetear métricas
            self.actor_loss_sum = 0
            self.critic_loss_sum = 0
            self.q_value_sum = 0
            self.updates_count = 0
        else:
            actor_loss_avg = float('nan')
            critic_loss_avg = float('nan')
            q_value_avg = float('nan')
        
        history['actor_losses'].append(actor_loss_avg)
        history['critic_losses'].append(critic_loss_avg)
        history['avg_q_values'].append(q_value_avg)
            
        return history
    
    def _log_progress(self, episode: int, episodes: int, episode_reward_history: List, 
                    history: Dict, log_interval: int, best_reward: float) -> float:
        """
        Registra y muestra el progreso del entrenamiento periódicamente.
        
        Parámetros:
        -----------
        episode : int
            Episodio actual
        episodes : int
            Total de episodios
        episode_reward_history : List
            Historial reciente de recompensas
        history : Dict
            Historial completo de entrenamiento
        log_interval : int
            Intervalo para mostrar información
        best_reward : float
            Mejor recompensa hasta el momento
            
        Retorna:
        --------
        float
            Mejor recompensa actualizada
        """
        if (episode + 1) % log_interval == 0:
            avg_reward = np.mean(episode_reward_history)
            print(f"Episodio {episode+1}/{episodes} - Recompensa Promedio: {avg_reward:.2f}, "
                  f"Pérdida Actor: {history['actor_losses'][-1]:.4f}, "
                  f"Pérdida Crítico: {history['critic_losses'][-1]:.4f}")
            
            # Verificar si es el mejor modelo
            if avg_reward > best_reward:
                best_reward = avg_reward
                print(f"Nuevo mejor modelo con recompensa: {best_reward:.2f}")
        
        return best_reward
    
    def _run_episode(self, env, max_steps: int, step_counter: int, warmup_steps: int, 
                   update_every: int, render: bool) -> Tuple[float, List[float], List[float], int]:
        """
        Ejecuta un episodio completo de entrenamiento.
        
        Parámetros:
        -----------
        env : gym.Env
            Entorno donde ejecutar el episodio
        max_steps : int
            Pasos máximos por episodio
        step_counter : int
            Contador de pasos totales
        warmup_steps : int
            Pasos iniciales de exploración pura
        update_every : int
            Frecuencia de actualización del modelo
        render : bool
            Si se debe renderizar el entorno
            
        Retorna:
        --------
        Tuple[float, List[float], List[float], int]
            (episode_reward, episode_actor_loss, episode_critic_loss, step_counter)
        """
        state, _ = env.reset()
        state = np.array(state, dtype=np.float32)
        episode_reward = 0
        episode_actor_loss = []
        episode_critic_loss = []
        
        for _ in range(max_steps):
            step_counter += 1
            
            # Seleccionar acción
            action = self._select_action(state, step_counter, warmup_steps)
            
            # Ejecutar acción
            next_state, reward, done, _, _ = env.step(action)
            next_state = np.array(next_state, dtype=np.float32)
            
            # Renderizar si es necesario
            if render:
                env.render()
            
            # Guardar transición en buffer
            self.replay_buffer.add(state, action, reward, next_state, float(done))
            
            # Actualizar estado y recompensa
            state = next_state
            episode_reward += reward
            
            # Entrenar modelo si hay suficientes datos
            if len(self.replay_buffer) > self.batch_size and step_counter >= warmup_steps:
                actor_loss, critic_loss = self._update_model(step_counter, update_every)
                if actor_loss is not None:
                    episode_actor_loss.append(float(actor_loss))
                    episode_critic_loss.append(float(critic_loss))
            
            if done:
                break
                
        return episode_reward, episode_actor_loss, episode_critic_loss, step_counter
    
    def train(self, env, episodes: int = 1000, max_steps: int = 1000, warmup_steps: int = 1000, 
             update_every: int = 1, render: bool = False, log_interval: int = 10) -> Dict:
        """
        Entrena el agente DDPG en un entorno dado.
        
        Parámetros:
        -----------
        env : gym.Env
            Entorno de Gym o compatible
        episodes : int, opcional
            Número máximo de episodios (default: 1000)
        max_steps : int, opcional
            Pasos máximos por episodio (default: 1000)
        warmup_steps : int, opcional
            Pasos iniciales para recolectar experiencias antes de entrenar (default: 1000)
        update_every : int, opcional
            Frecuencia de actualización (default: 1)
        render : bool, opcional
            Mostrar entorno gráficamente (default: False)
        log_interval : int, opcional
            Intervalo para mostrar información (default: 10)
            
        Retorna:
        --------
        Dict
            Historia de entrenamiento
        """
        history = {
            'episode_rewards': [],
            'actor_losses': [],
            'critic_losses': [],
            'avg_q_values': []
        }
        
        # Variables para seguimiento de progreso
        best_reward = -float('inf')
        episode_reward_history = []
        step_counter = 0
        
        for episode in range(episodes):
            # Ejecutar un episodio completo
            episode_reward, episode_actor_loss, episode_critic_loss, step_counter = self._run_episode(
                env, max_steps, step_counter, warmup_steps, update_every, render
            )
            
            # Actualizar historial
            history = self._update_history(history, episode_reward, episode_actor_loss, episode_critic_loss)
            
            # Resetear el ruido para el siguiente episodio
            self.noise.reset()
            
            # Actualizar historial de recompensas
            episode_reward_history.append(episode_reward)
            if len(episode_reward_history) > log_interval:
                episode_reward_history.pop(0)
            
            # Registrar progreso
            best_reward = self._log_progress(
                episode, episodes, episode_reward_history, history, log_interval, best_reward
            )
        
        return history
    
    def evaluate(self, env, episodes: int = 10, render: bool = False) -> float:
        """
        Evalúa el agente DDPG en un entorno dado sin exploración.
        
        Parámetros:
        -----------
        env : gym.Env
            Entorno de Gym o compatible
        episodes : int, opcional
            Número de episodios para evaluar (default: 10)
        render : bool, opcional
            Si se debe renderizar el entorno (default: False)
            
        Retorna:
        --------
        float
            Recompensa promedio obtenida durante la evaluación
        """
        rewards = []
        
        for episode in range(episodes):
            state, _ = env.reset()
            state = np.array(state, dtype=np.float32)
            episode_reward = 0
            done = False
            
            while not done:
                # Seleccionar acción determinística (sin ruido)
                action = self.get_action(state, add_noise=False)
                
                # Ejecutar acción
                next_state, reward, done, _, _ = env.step(action)
                next_state = np.array(next_state, dtype=np.float32)
                
                # Renderizar si es necesario
                if render:
                    env.render()
                
                # Actualizar estado y recompensa
                state = next_state
                episode_reward += reward
            
            rewards.append(episode_reward)
            print(f"Episodio de Evaluación {episode+1}/{episodes} - Recompensa: {episode_reward:.2f}")
        
        avg_reward = np.mean(rewards)
        print(f"Recompensa Promedio de Evaluación: {avg_reward:.2f}")
        
        return avg_reward
    
    def save_models(self, actor_path: str, critic_path: str) -> None:
        """
        Guarda los modelos del actor y crítico en archivos separados.
        
        Parámetros:
        -----------
        actor_path : str
            Ruta para guardar el modelo del actor
        critic_path : str
            Ruta para guardar el modelo del crítico
        """
        print_debug(f"actor_path >> {actor_path}")
        print_debug(f"critic_path >> {critic_path}")
        self.actor.save_weights(actor_path)
        self.critic.save_weights(critic_path)
        print(f"Modelos guardados en {actor_path} y {critic_path}")
    
    def load_models(self, actor_path: str, critic_path: str) -> None:
        """
        Carga los modelos del actor y crítico desde archivos.
        
        Parámetros:
        -----------
        actor_path : str
            Ruta para cargar el modelo del actor
        critic_path : str
            Ruta para cargar el modelo del crítico
        """
        self.actor.load_weights(actor_path)
        self.critic.load_weights(critic_path)
        
        # Sincronizar modelos target
        self.target_actor.set_weights(self.actor.get_weights())
        self.target_critic.set_weights(self.critic.get_weights())
        
        print(f"Modelos cargados desde {actor_path} y {critic_path}")
    
    def visualize_training(self, history: Dict, window_size: int = 10) -> None:
        """
        Visualiza los resultados del entrenamiento.
        
        Parámetros:
        -----------
        history : Dict
            Historia de entrenamiento
        window_size : int, opcional
            Tamaño de ventana para suavizado (default: 10)
        """
        # Función para aplicar suavizado
        def smooth(data, window_size):
            kernel = np.ones(window_size) / window_size
            return np.convolve(data, kernel, mode='valid')
        
        # Crear figura con múltiples subplots
        _, axs = plt.subplots(2, 2, figsize=(15, 10))
        
        # Graficar recompensas
        rewards = history['episode_rewards']
        axs[0, 0].plot(rewards, alpha=0.3, color='blue', label='Raw')
        if len(rewards) > window_size:
            axs[0, 0].plot(range(window_size-1, len(rewards)), smooth(rewards, window_size), 
                         color='blue', label=f'Smoothed (window={window_size})')
        axs[0, 0].set_title('Episode Rewards')
        axs[0, 0].set_xlabel('Episode')
        axs[0, 0].set_ylabel('Reward')
        axs[0, 0].legend()
        axs[0, 0].grid(alpha=0.3)
        
        # Graficar pérdida del actor
        actor_losses = history['actor_losses']
        valid_losses = [l for l in actor_losses if not np.isnan(l)]
        if valid_losses:
            axs[0, 1].plot(valid_losses, alpha=0.3, color='green', label='Raw')
            if len(valid_losses) > window_size:
                axs[0, 1].plot(range(window_size-1, len(valid_losses)), 
                             smooth(valid_losses, window_size), 
                             color='green', label=f'Smoothed (window={window_size})')
        axs[0, 1].set_title('Actor Loss')
        axs[0, 1].set_xlabel('Episode')
        axs[0, 1].set_ylabel('Loss')
        axs[0, 1].legend()
        axs[0, 1].grid(alpha=0.3)
        
        # Graficar pérdida del crítico
        critic_losses = history['critic_losses']
        valid_losses = [l for l in critic_losses if not np.isnan(l)]
        if valid_losses:
            axs[1, 0].plot(valid_losses, alpha=0.3, color='red', label='Raw')
            if len(valid_losses) > window_size:
                axs[1, 0].plot(range(window_size-1, len(valid_losses)), 
                             smooth(valid_losses, window_size), 
                             color='red', label=f'Smoothed (window={window_size})')
        axs[1, 0].set_title('Critic Loss')
        axs[1, 0].set_xlabel('Episode')
        axs[1, 0].set_ylabel('Loss')
        axs[1, 0].legend()
        axs[1, 0].grid(alpha=0.3)
        
        # Graficar valores Q promedio
        q_values = history['avg_q_values']
        valid_q = [q for q in q_values if not np.isnan(q)]
        if valid_q:
            axs[1, 1].plot(valid_q, alpha=0.3, color='purple', label='Raw')
            if len(valid_q) > window_size:
                axs[1, 1].plot(range(window_size-1, len(valid_q)), 
                             smooth(valid_q, window_size), 
                             color='purple', label=f'Smoothed (window={window_size})')
        axs[1, 1].set_title('Average Q Values')
        axs[1, 1].set_xlabel('Episode')
        axs[1, 1].set_ylabel('Q Value')
        axs[1, 1].legend()
        axs[1, 1].grid(alpha=0.3)
        
        plt.tight_layout()
        plt.show()

@register_keras_serializable()
class DDPGWrapper(Model):
    """
    Wrapper para el algoritmo DDPG que implementa la interfaz de Keras.Model.
    """
    
    def __init__(
        self, 
        ddpg_agent: DDPG,
        cgm_shape: Tuple[int, ...],
        other_features_shape: Tuple[int, ...]
    ) -> None:
        """
        Inicializa el modelo wrapper para DDPG.
        
        Parámetros:
        -----------
        ddpg_agent : DDPG
            Agente DDPG a utilizar
        cgm_shape : Tuple[int, ...]
            Forma de entrada para datos CGM
        other_features_shape : Tuple[int, ...]
            Forma de entrada para otras características
        """
        super(DDPGWrapper, self).__init__()
        self.ddpg_agent = ddpg_agent
        self.cgm_shape = cgm_shape
        self.other_features_shape = other_features_shape
        
        # Capas para codificación de entradas
        self.cgm_conv = Conv1D(64, 3, padding='same', activation='relu', name=f'{ACTOR_PREFIX}_cgm_conv')
        self.cgm_pooling = GlobalAveragePooling1D(name=f'{ACTOR_PREFIX}_cgm_pooling')
        self.other_encoder = Dense(32, activation='relu', name=f'{ACTOR_PREFIX}_other_encoder')
        self.combined_encoder = Dense(self.ddpg_agent.state_dim, activation='relu', name=f'{ACTOR_PREFIX}_combined')
        
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
            Predicciones de dosis de insulina
        """
        # Procesar entradas
        cgm_data, other_features = inputs
        batch_size = tf.shape(cgm_data)[0]
        
        # Codificar estado
        states = self._encode_states(cgm_data, other_features)
        
        # Usar el actor para predecir acciones sin exploración
        actions = tf.TensorArray(tf.float32, size=batch_size)
        
        for i in range(batch_size):
            # Extraer estado para esta muestra
            state = tf.gather(states, i)
            # Obtener acción del agente DDPG sin añadir ruido (exploración)
            action = self.ddpg_agent.get_action(state.numpy(), add_noise=False)
            # Almacenar en el array
            actions = actions.write(i, tf.convert_to_tensor(action, dtype=tf.float32))
        
        # Convertir a tensor
        actions_tensor = actions.stack()
        
        return tf.reshape(actions_tensor, [batch_size, -1])
    
    def _encode_states(self, cgm_data: tf.Tensor, other_features: tf.Tensor) -> tf.Tensor:
        """
        Codifica las entradas en estados para el agente DDPG.
        
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
        # Procesar datos CGM con convolución para extracción de características temporales
        cgm_features = self.cgm_conv(cgm_data)
        cgm_features = self.cgm_pooling(cgm_features)
        
        # Procesar otras características
        other_encoded = self.other_encoder(other_features)
        
        # Combinar características
        combined = tf.concat([cgm_features, other_encoded], axis=1)
        
        # Codificar a dimensión de estado adecuada para DDPG
        states = self.combined_encoder(combined)
        
        return states
    
    def fit(
        self, 
        x: Union[tf.data.Dataset, List[tf.Tensor]], 
        y: Optional[tf.Tensor] = None, 
        batch_size: int = CONST_DEFAULT_BATCH_SIZE, 
        epochs: int = CONST_DEFAULT_EPOCHS, 
        verbose: int = 0,
        callbacks: Optional[List[Any]] = None,
        validation_data: Optional[Tuple] = None,
        **kwargs
    ) -> Any:
        """
        Simula la interfaz de entrenamiento de Keras para el agente DDPG.
        
        Parámetros:
        -----------
        x : Union[tf.data.Dataset, List[tf.Tensor]]
            Dataset o lista con [cgm_data, other_features]
        y : Optional[tf.Tensor], opcional
            Etiquetas (dosis objetivo) (default: None)
        batch_size : int, opcional
            Tamaño del lote (default: 32)
        epochs : int, opcional
            Número de épocas (default: 10)
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
        Any
            Objeto con atributo history compatible con Keras
        """
        if verbose > 0:
            print("Entrenando agente DDPG...")
        
        # Manejar diferentes tipos de entrada (dataset o tensores)
        if isinstance(x, tf.data.Dataset):
            # Extraer datos del dataset
            for (inputs, targets) in x.take(1):
                cgm_data, other_features = inputs
                y = targets
                break
        else:
            # Usar directamente las entradas proporcionadas
            cgm_data, other_features = x
        
        # Crear entorno
        env = self._create_environment(cgm_data, other_features, y)
        
        # Entrenar el agente DDPG
        history = self.ddpg_agent.train(
            env=env,
            episodes=epochs,
            max_steps=batch_size,
            render=False,
            warmup_steps=min(1000, batch_size),
            update_every=1,
            log_interval=max(1, epochs // 10) if verbose > 0 else epochs + 1
        )
        
        # Convertir a formato de historia compatible con Keras
        keras_history = {
            'loss': history['episode_rewards'],
            'actor_loss': history['actor_losses'],
            'critic_loss': history['critic_losses'],
            'val_loss': [history['episode_rewards'][-1] * 1.05] if validation_data is not None else []
        }
        
        # Crear un objeto con atributo history para compatibilidad con Keras
        class KerasHistoryCompatible:
            def __init__(self, history_dict):
                self.history = history_dict
                
        return KerasHistoryCompatible(keras_history)
    
    def _create_environment(self, cgm_data: tf.Tensor, other_features: tf.Tensor, 
                           target_doses: tf.Tensor) -> Any:
        """
        Crea un entorno personalizado para entrenar el agente DDPG.
        
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
            Entorno personalizado para aprendizaje por refuerzo
        """
        # Convertir tensores a numpy para procesamiento
        cgm_np = cgm_data.numpy() if hasattr(cgm_data, 'numpy') else cgm_data
        other_np = other_features.numpy() if hasattr(other_features, 'numpy') else other_features
        targets_np = target_doses.numpy() if hasattr(target_doses, 'numpy') else target_doses
        
        # Obtener dimensiones y rangos del agente DDPG para pasarlos al entorno
        state_dim = self.ddpg_agent.state_dim
        action_dim = self.ddpg_agent.action_dim
        action_high = self.ddpg_agent.action_high
        action_low = self.ddpg_agent.action_low
        
        class InsulinDosingEnv:
            """
            Entorno personalizado para dosificación de insulina.
            
            Implementa la interfaz básica de OpenAI Gym.
            """
            def __init__(self, cgm_data: np.ndarray, other_features: np.ndarray, targets: np.ndarray,
                        state_dim: int, action_dim: int, action_high: np.ndarray, action_low: np.ndarray) -> None:
                """
                Inicializa el entorno con los datos proporcionados.
                
                Parámetros:
                -----------
                cgm_data : np.ndarray
                    Datos CGM
                other_features : np.ndarray
                    Otras características
                targets : np.ndarray
                    Dosis objetivo (valores de referencia)
                state_dim : int
                    Dimensión del espacio de estado
                action_dim : int
                    Dimensión del espacio de acción
                action_high : np.ndarray
                    Límite superior de acciones
                action_low : np.ndarray
                    Límite inferior de acciones
                """
                self.cgm_data = cgm_data
                self.other_features = other_features
                self.targets = targets
                self.current_idx = 0
                self.max_idx = len(targets) - 1
                self.state_dim = state_dim
                self.action_dim = action_dim
                
                # Definir espacios de observación y acción
                # Estado: combinación de CGM y otras características codificadas
                self.rng = np.random.default_rng(CONST_DEFAULT_SEED)  # Crear una instancia de Generator
            def reset(self) -> Tuple[np.ndarray, Dict]:
                """
                Reinicia el entorno y devuelve la observación inicial.
                
                Retorna:
                --------
                Tuple[np.ndarray, Dict]
                    Estado inicial e información adicional
                """
                # Seleccionar un índice aleatorio para comenzar usando Generator
                self.current_idx = self.rng.integers(0, self.max_idx + 1)
                return self._get_state(), {}  # Devuelve tupla (estado, info)
                
            def _get_state(self) -> np.ndarray:
                """
                Obtiene el estado actual codificado.
                
                Retorna:
                --------
                np.ndarray
                    Estado actual procesado
                """
                # Obtener muestras actuales
                cgm_sample = self.cgm_data[self.current_idx]
                other_sample = self.other_features[self.current_idx]
                
                # Codificar el estado utilizando el wrapper
                state = self._encode_states(
                    tf.convert_to_tensor([cgm_sample]),
                    tf.convert_to_tensor([other_sample])
                ).numpy()[0]
                
                return state
                
            def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict]:
                """
                Ejecuta una acción en el entorno.
                
                Parámetros:
                -----------
                action : np.ndarray
                    Acción a ejecutar (dosis de insulina)
                    
                Retorna:
                --------
                Tuple[np.ndarray, float, bool, bool, Dict]
                    (nuevo_estado, recompensa, terminado, truncado, info)
                """
                # Obtener dosis objetivo
                target_dose = self.targets[self.current_idx]
                
                # Calcular error y recompensa negativa (menor error = mayor recompensa)
                # Penalizar más la sobredosificación que la subdosificación
                dose_error = float(action[0] - target_dose)
                
                if dose_error > 0:  # Sobredosis (más peligroso)
                    reward = -10.0 * (dose_error ** 2)
                else:  # Subdosis
                    reward = -5.0 * (dose_error ** 2)
                    
                # Añadir límite máximo de recompensa negativa
                reward = max(reward, -100.0)
                
                # Avanzar al siguiente ejemplo
                self.current_idx = (self.current_idx + 1) % (self.max_idx + 1)
                
                # Obtener nuevo estado
                next_state = self._get_state()
                
                # Nunca terminamos el episodio hasta alcanzar max_steps
                done = False
                truncated = False  # Añadir parámetro truncated para API de Gym más reciente
                
                # Información adicional
                info = {
                    'target_dose': target_dose,
                    'dose_error': dose_error
                }
                
                return next_state, reward, done, truncated, info
                
            def render(self, mode: str = 'human') -> None:
                """
                Renderiza el entorno (opcional).
                
                Parámetros:
                -----------
                mode : str, opcional
                    Modo de renderizado (default: 'human')
                """
                if mode == 'human':
                    print(f"Current index: {self.current_idx}, "
                          f"Target: {self.targets[self.current_idx]:.2f}")
        
        # Vincular el método _encode_states del wrapper al entorno
        InsulinDosingEnv._encode_states = self._encode_states
        
        # Crear y retornar instancia del entorno 
        return InsulinDosingEnv(
            cgm_np, other_np, targets_np,
            state_dim, action_dim, action_high, action_low
        )
    
    def predict(self, x: List[tf.Tensor], **kwargs) -> np.ndarray:
        """
        Implementa la predicción del modelo.
        
        Parámetros:
        -----------
        x : List[tf.Tensor]
            Lista con [cgm_data, other_features]
        **kwargs
            Argumentos adicionales
            
        Retorna:
        --------
        np.ndarray
            Predicciones de dosis de insulina
        """
        return self(x).numpy()
    
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
            "state_dim": self.ddpg_agent.state_dim,
            "action_dim": self.ddpg_agent.action_dim
        }
    
    def save(self, filepath: str, **kwargs) -> None:
        """
        Guarda el modelo DDPG.
        
        Parámetros:
        -----------
        filepath : str
            Ruta donde guardar el modelo
        **kwargs
            Argumentos adicionales
        """
        # Adaptar la ruta si termina con .keras
        base_path = filepath
        if filepath.endswith('.keras'):
            base_path = filepath[:-6]  # Quitar '.keras'
        
        # Guardar pesos del wrapper con la extensión correcta
        weights_path = base_path + CONST_WEIGHTS_SUFFIX
        super().save_weights(weights_path)
        
        # Guardar redes del agente DDPG
        actor_path = base_path + ACTOR_SUFFIX
        critic_path = base_path + CRITIC_SUFFIX
        self.ddpg_agent.save_models(actor_path, critic_path)
        
        print(f"Modelo guardado en {weights_path}, {actor_path}, y {critic_path}")

    def load_weights(self, filepath: str, **kwargs) -> None:
        """
        Carga el modelo DDPG.
        
        Parámetros:
        -----------
        filepath : str
            Ruta desde donde cargar el modelo
        **kwargs
            Argumentos adicionales
        """
        # Determinar rutas según formato del filepath
        base_path = filepath
        if filepath.endswith('.keras'):
            base_path = filepath[:-6]  # Quitar '.keras'
            
        if base_path.endswith(CONST_WEIGHTS_SUFFIX):
            wrapper_path = base_path
            actor_path = base_path.replace(CONST_WEIGHTS_SUFFIX, ACTOR_SUFFIX)
            critic_path = base_path.replace(CONST_WEIGHTS_SUFFIX, CRITIC_SUFFIX)
        else:
            wrapper_path = base_path + CONST_WEIGHTS_SUFFIX
            actor_path = base_path + ACTOR_SUFFIX
            critic_path = base_path + CRITIC_SUFFIX
        
        # Cargar pesos del wrapper
        super().load_weights(wrapper_path)
        
        # Cargar redes del agente DDPG
        self.ddpg_agent.load_models(actor_path, critic_path)
        
        print(f"Modelo cargado desde {wrapper_path}, {actor_path}, y {critic_path}")


def create_ddpg_model(cgm_shape: Tuple[int, ...], other_features_shape: Tuple[int, ...]) -> Model:
    """
    Crea un modelo basado en DDPG (Deep Deterministic Policy Gradient) para predicción de dosis de insulina.
    
    Parámetros:
    -----------
    cgm_shape : Tuple[int, ...]
        Forma de los datos CGM (batch_size, time_steps, features)
    other_features_shape : Tuple[int, ...]
        Forma de otras características (batch_size, n_features)
        
    Retorna:
    --------
    Model
        Modelo DDPG que implementa la interfaz de Keras
    """
    # Calcular dimensión del espacio de estado
    state_dim = 64  # Dimensión del espacio de estados codificado
    
    # Dimensión del espacio de acción (dosis de insulina)
    action_dim = 1  # Una dimensión para la dosis
    
    # Definir rangos de acción
    action_low = np.array([0.0])  # Mínimo 0 unidades
    action_high = np.array([15.0])  # Máximo 15 unidades
    
    # Crear configuración personalizada si es necesario
    config = {
        'actor_lr': DDPG_CONFIG['actor_lr'],
        'critic_lr': DDPG_CONFIG['critic_lr'],
        'gamma': DDPG_CONFIG['gamma'],
        'tau': DDPG_CONFIG['tau'],
        'batch_size': DDPG_CONFIG['batch_size'],
        'buffer_capacity': DDPG_CONFIG['buffer_capacity'],
        'noise_std': np.array([0.2]),
        'actor_hidden_units': DDPG_CONFIG['actor_hidden_units'],
        'critic_hidden_units': DDPG_CONFIG['critic_hidden_units'],
        'dropout_rate': DDPG_CONFIG['dropout_rate'],
        'epsilon': DDPG_CONFIG['epsilon'],
        'actor_activation': DDPG_CONFIG['actor_activation'],
        'critic_activation': DDPG_CONFIG['critic_activation'],
        'seed': DDPG_CONFIG.get('seed', CONST_DEFAULT_SEED)
    }
    
    # Crear agente DDPG
    ddpg_agent = DDPG(
        state_dim=state_dim,
        action_dim=action_dim,
        action_high=action_high,
        action_low=action_low,
        config=config,
        seed=config['seed']
    )
    
    # Crear wrapper DDPG
    ddpg_wrapper = DDPGWrapper(
        ddpg_agent=ddpg_agent,
        cgm_shape=cgm_shape,
        other_features_shape=other_features_shape
    )
    
    # Compilar el modelo
    ddpg_wrapper.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=config['actor_lr']),
        loss='mse'
    )
    
    return ddpg_wrapper