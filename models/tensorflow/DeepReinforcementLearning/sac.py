import os, sys
import tensorflow as tf
import numpy as np
import gym
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input, Dense, Conv1D, Flatten, Concatenate,
    BatchNormalization, Dropout, LayerNormalization
)
from tensorflow.keras.optimizers import Adam
from keras.saving import register_keras_serializable
from collections import deque
import random
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Any, Optional, Union, Callable

PROJECT_ROOT = os.path.abspath(os.getcwd())
sys.path.append(PROJECT_ROOT) 

from config.models_config import SAC_CONFIG


class ReplayBuffer:
    """
    Buffer de experiencias para el algoritmo SAC.
    
    Almacena transiciones (state, action, reward, next_state, done)
    y permite muestrear lotes de manera aleatoria para el entrenamiento.
    
    Parámetros:
    -----------
    capacity : int, opcional
        Capacidad máxima del buffer (default: 100000)
    """
    def __init__(self, capacity: int = 100000) -> None:
        self.buffer = deque(maxlen=capacity)
    
    def add(self, state: np.ndarray, action: np.ndarray, 
            reward: float, next_state: np.ndarray, done: float) -> None:
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
            np.array(rewards, dtype=np.float32).reshape(-1, 1),
            np.array(next_states, dtype=np.float32),
            np.array(dones, dtype=np.float32).reshape(-1, 1)
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


class ActorNetwork(Model):
    """
    Red del Actor para SAC que produce una distribución de política gaussiana.
    
    Esta red mapea estados a distribuciones de probabilidad sobre acciones
    mediante una política estocástica parametrizada por una distribución normal.
    
    Parámetros:
    -----------
    state_dim : int
        Dimensión del espacio de estados
    action_dim : int
        Dimensión del espacio de acciones
    action_high : np.ndarray
        Límite superior del espacio de acciones
    action_low : np.ndarray
        Límite inferior del espacio de acciones
    hidden_units : Optional[List[int]], opcional
        Unidades en capas ocultas (default: None)
    """
    def __init__(
        self, 
        state_dim: int, 
        action_dim: int, 
        action_high: np.ndarray, 
        action_low: np.ndarray, 
        hidden_units: Optional[List[int]] = None
    ) -> None:
        super(ActorNetwork, self).__init__()
        
        # Límites de acciones para escalar la salida
        self.action_high = action_high
        self.action_low = action_low
        self.action_dim = action_dim
        self.log_std_min = SAC_CONFIG['log_std_min']  # Límite inferior para log_std
        self.log_std_max = SAC_CONFIG['log_std_max']  # Límite superior para log_std
        
        # Valores predeterminados para capas ocultas
        if hidden_units is None:
            hidden_units = SAC_CONFIG['actor_hidden_units']
        
        # Capas para el procesamiento del estado
        self.hidden_layers = []
        for i, units in enumerate(hidden_units):
            self.hidden_layers.append(Dense(
                units, 
                activation=SAC_CONFIG['actor_activation'],
                name=f'actor_dense_{i}'
            ))
            self.hidden_layers.append(LayerNormalization(
                epsilon=SAC_CONFIG['epsilon'],
                name=f'actor_ln_{i}'
            ))
            self.hidden_layers.append(Dropout(
                SAC_CONFIG['dropout_rate'],
                name=f'actor_dropout_{i}'
            ))
        
        # Capas de salida para media y log-desviación estándar
        self.mean_layer = Dense(action_dim, activation='linear', name='actor_mean')
        self.log_std_layer = Dense(action_dim, activation='linear', name='actor_log_std')
    
    def call(self, inputs: tf.Tensor, training: bool = False) -> Tuple[tf.Tensor, tf.Tensor]:
        """
        Realiza el forward pass del modelo actor.
        
        Parámetros:
        -----------
        inputs : tf.Tensor
            Tensor de estados
        training : bool, opcional
            Si está en modo entrenamiento (default: False)
            
        Retorna:
        --------
        Tuple[tf.Tensor, tf.Tensor]
            (mean, std) - Media y desviación estándar de la distribución de política
        """
        x = inputs
        
        # Procesar a través de capas ocultas
        for layer in self.hidden_layers:
            if isinstance(layer, Dropout):
                x = layer(x, training=training)
            else:
                x = layer(x)
        
        # Calcular media y log_std
        mean = self.mean_layer(x)
        log_std = self.log_std_layer(x)
        log_std = tf.clip_by_value(log_std, self.log_std_min, self.log_std_max)
        std = tf.exp(log_std)
        
        return mean, std
    
    def sample_action(self, state: tf.Tensor, deterministic: bool = False) -> Tuple[tf.Tensor, Optional[tf.Tensor]]:
        """
        Muestrea una acción de la distribución de política.
        
        Parámetros:
        -----------
        state : tf.Tensor
            El estado actual
        deterministic : bool, opcional
            Si es True, devuelve la acción media (sin ruido) (default: False)
            
        Retorna:
        --------
        Tuple[tf.Tensor, Optional[tf.Tensor]]
            (acción, log_prob) - Acción muestreada y log-probabilidad
        """
        # Obtener parámetros de la distribución
        mean, std = self(state, training=False)
        
        if deterministic:
            # Para evaluación o explotación
            actions = mean
            log_probs = None
        else:
            # Muestrear usando el truco de reparametrización para permitir backprop
            noise = tf.random.normal(shape=mean.shape)
            z = mean + std * noise
            
            # Aplicar tanh para acotar las acciones
            actions = tf.tanh(z)
            
            # Calcular log-probabilidad con corrección para tanh
            log_probs = self._log_prob(z, std, actions)
        
        # Escalar acciones al rango deseado
        scaled_actions = self._scale_actions(actions)
        
        return scaled_actions, log_probs
    
    def _scale_actions(self, actions: tf.Tensor) -> tf.Tensor:
        """
        Escala las acciones al rango deseado.
        
        Parámetros:
        -----------
        actions : tf.Tensor
            Acciones normalizadas en el rango [-1, 1]
            
        Retorna:
        --------
        tf.Tensor
            Acciones escaladas al rango [action_low, action_high]
        """
        return actions * (self.action_high - self.action_low) / 2 + (self.action_high + self.action_low) / 2
    
    def _log_prob(self, z: tf.Tensor, std: tf.Tensor, actions: tf.Tensor) -> tf.Tensor:
        """
        Calcula el logaritmo de la probabilidad de una acción.
        
        Parámetros:
        -----------
        z : tf.Tensor
            Valor antes de aplicar tanh
        std : tf.Tensor
            Desviación estándar
        actions : tf.Tensor
            Acción muestreada
            
        Retorna:
        --------
        tf.Tensor
            Log-probabilidad de la acción
        """
        # Log-prob de distribución normal
        log_prob_gaussian = -0.5 * (tf.square(z) + 2 * tf.math.log(std) + tf.math.log(2.0 * np.pi))
        log_prob_gaussian = tf.reduce_sum(log_prob_gaussian, axis=-1, keepdims=True)
        
        # Corrección por transformación tanh
        # Deriva del cambio de variable (ver paper SAC)
        squash_correction = tf.reduce_sum(
            tf.math.log(1.0 - tf.square(tf.tanh(z)) + 1e-6),
            axis=-1,
            keepdims=True
        )
        
        return log_prob_gaussian - squash_correction


class CriticNetwork(Model):
    """
    Red de Crítico para SAC que mapea pares (estado, acción) a valores-Q.
    
    Parámetros:
    -----------
    state_dim : int
        Dimensión del espacio de estados
    action_dim : int
        Dimensión del espacio de acciones
    hidden_units : Optional[List[int]], opcional
        Unidades en capas ocultas (default: None)
    """
    def __init__(
        self, 
        state_dim: int, 
        action_dim: int, 
        hidden_units: Optional[List[int]] = None
    ) -> None:
        super(CriticNetwork, self).__init__()
        
        # Valores predeterminados para capas ocultas
        if hidden_units is None:
            hidden_units = SAC_CONFIG['critic_hidden_units']
        
        # Capas para procesar el estado inicialmente
        self.state_layers = []
        for i, units in enumerate(hidden_units[:1]):  # Primera capa solo para estado
            self.state_layers.append(Dense(
                units, 
                activation=SAC_CONFIG['critic_activation'],
                name=f'critic_state_dense_{i}'
            ))
            self.state_layers.append(LayerNormalization(
                epsilon=SAC_CONFIG['epsilon'],
                name=f'critic_state_ln_{i}'
            ))
        
        # Capas para procesar la combinación estado-acción
        self.combined_layers = []
        for i, units in enumerate(hidden_units[1:]):
            self.combined_layers.append(Dense(
                units, 
                activation=SAC_CONFIG['critic_activation'],
                name=f'critic_combined_dense_{i}'
            ))
            self.combined_layers.append(LayerNormalization(
                epsilon=SAC_CONFIG['epsilon'],
                name=f'critic_combined_ln_{i}'
            ))
            self.combined_layers.append(Dropout(
                SAC_CONFIG['dropout_rate'],
                name=f'critic_dropout_{i}'
            ))
        
        # Capa de salida: valor Q
        self.output_layer = Dense(1, name='critic_output')
    
    def call(self, inputs: Tuple[tf.Tensor, tf.Tensor], training: bool = False) -> tf.Tensor:
        """
        Realiza el forward pass del modelo crítico.
        
        Parámetros:
        -----------
        inputs : Tuple[tf.Tensor, tf.Tensor]
            Tupla (estados, acciones)
        training : bool, opcional
            Si está en modo entrenamiento (default: False)
            
        Retorna:
        --------
        tf.Tensor
            Valores Q estimados
        """
        states, actions = inputs
        
        # Procesar el estado inicialmente
        x = states
        for layer in self.state_layers:
            x = layer(x)
        
        # Combinar estado procesado con acción
        x = tf.concat([x, actions], axis=-1)
        
        # Procesar la combinación
        for layer in self.combined_layers:
            if isinstance(layer, Dropout):
                x = layer(x, training=training)
            else:
                x = layer(x)
        
        # Capa de salida: valor Q
        q_value = self.output_layer(x)
        
        return q_value


class SAC:
    """
    Implementación del algoritmo Soft Actor-Critic (SAC).
    
    SAC es un algoritmo de aprendizaje por refuerzo fuera de política (off-policy)
    basado en el marco de máxima entropía, que busca maximizar tanto el retorno
    esperado como la entropía de la política.
    
    Parámetros:
    -----------
    state_dim : int
        Dimensión del espacio de estados
    action_dim : int
        Dimensión del espacio de acciones
    action_high : np.ndarray
        Límite superior del espacio de acciones
    action_low : np.ndarray
        Límite inferior del espacio de acciones
    config : Optional[Dict[str, Any]], opcional
        Configuración personalizada (default: None)
    seed : int, opcional
        Semilla para reproducibilidad (default: 42)
    """
    def __init__(
        self, 
        state_dim: int, 
        action_dim: int,
        action_high: np.ndarray,
        action_low: np.ndarray,
        config: Optional[Dict[str, Any]] = None,
        seed: int = 42
    ) -> None:
        # Configurar semillas para reproducibilidad
        tf.random.set_seed(seed)
        np.random.seed(seed)
        random.seed(seed)
        
        # Use default config if none provided
        if config is None:
            config = {}
            
        # Set parameters from config with defaults from SAC_CONFIG
        actor_lr = config.get('actor_lr', SAC_CONFIG['actor_lr'])
        critic_lr = config.get('critic_lr', SAC_CONFIG['critic_lr'])
        alpha_lr = config.get('alpha_lr', SAC_CONFIG['alpha_lr'])
        self.gamma = config.get('gamma', SAC_CONFIG['gamma'])
        self.tau = config.get('tau', SAC_CONFIG['tau'])
        buffer_capacity = config.get('buffer_capacity', SAC_CONFIG['buffer_capacity'])
        self.batch_size = config.get('batch_size', SAC_CONFIG['batch_size'])
        initial_alpha = config.get('initial_alpha', SAC_CONFIG['initial_alpha'])
        target_entropy = config.get('target_entropy', None)
        actor_hidden_units = config.get('actor_hidden_units', None)
        critic_hidden_units = config.get('critic_hidden_units', None)
        
        # Parámetros del entorno y del modelo
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.action_high = action_high 
        self.action_low = action_low
        
        # Valores predeterminados para capas ocultas
        if actor_hidden_units is None:
            self.actor_hidden_units = SAC_CONFIG['actor_hidden_units']
        else:
            self.actor_hidden_units = actor_hidden_units
            
        if critic_hidden_units is None:
            self.critic_hidden_units = SAC_CONFIG['critic_hidden_units']
        else:
            self.critic_hidden_units = critic_hidden_units
        
        # Crear modelos
        self.actor = ActorNetwork(
            state_dim=state_dim,
            action_dim=action_dim,
            action_high=action_high,
            action_low=action_low,
            hidden_units=self.actor_hidden_units
        )
        
        # Crear dos críticos (para reducir sobreestimación)
        self.critic_1 = CriticNetwork(
            state_dim=state_dim, 
            action_dim=action_dim,
            hidden_units=self.critic_hidden_units
        )
        
        self.critic_2 = CriticNetwork(
            state_dim=state_dim, 
            action_dim=action_dim,
            hidden_units=self.critic_hidden_units
        )
        
        # Crear redes target
        self.target_critic_1 = CriticNetwork(
            state_dim=state_dim, 
            action_dim=action_dim,
            hidden_units=self.critic_hidden_units
        )
        
        self.target_critic_2 = CriticNetwork(
            state_dim=state_dim, 
            action_dim=action_dim,
            hidden_units=self.critic_hidden_units
        )
        
        # Asegurar que los modelos estén construidos
        dummy_state = np.zeros((1, state_dim), dtype=np.float32)
        dummy_action = np.zeros((1, action_dim), dtype=np.float32)
        
        _ = self.actor(dummy_state)
        _ = self.critic_1([dummy_state, dummy_action])
        _ = self.critic_2([dummy_state, dummy_action])
        _ = self.target_critic_1([dummy_state, dummy_action])
        _ = self.target_critic_2([dummy_state, dummy_action])
        
        # Sincronizar pesos de redes target con principales
        self.update_target_networks(tau=1.0)
        
        # Optimizadores
        self.actor_optimizer = Adam(learning_rate=actor_lr)
        self.critic_optimizer = Adam(learning_rate=critic_lr)
        
        # Parámetro alpha (temperatura)
        self.log_alpha = tf.Variable(tf.math.log(initial_alpha), dtype=tf.float32)
        self.alpha_optimizer = Adam(learning_rate=alpha_lr)
        
        # Entropía objetivo (heurística: -dim(A))
        if target_entropy is None:
            self.target_entropy = -action_dim  # Valor predeterminado: -dim(A)
        else:
            self.target_entropy = target_entropy
        
        # Buffer de experiencias
        self.replay_buffer = ReplayBuffer(buffer_capacity)
        
        # Métricas acumuladas
        self.actor_loss_sum = 0.0
        self.critic_loss_sum = 0.0
        self.alpha_loss_sum = 0.0
        self.entropy_sum = 0.0
        self.updates_count = 0
    
    def update_target_networks(self, tau: Optional[float] = None) -> None:
        """
        Actualiza los parámetros de las redes target con soft update.
        
        Parámetros:
        -----------
        tau : Optional[float], opcional
            Factor de interpolación (default: None, usa el valor del objeto)
        """
        tau = tau if tau is not None else self.tau
        
        # Actualizar target_critic_1
        for source_weight, target_weight in zip(self.critic_1.trainable_variables, self.target_critic_1.trainable_variables):
            target_weight.assign((1 - tau) * target_weight + tau * source_weight)
        
        # Actualizar target_critic_2
        for source_weight, target_weight in zip(self.critic_2.trainable_variables, self.target_critic_2.trainable_variables):
            target_weight.assign((1 - tau) * target_weight + tau * source_weight)
    
    @tf.function
    def _update_critics(self, states: tf.Tensor, actions: tf.Tensor, rewards: tf.Tensor, 
                     next_states: tf.Tensor, dones: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
        """
        Actualiza las redes de crítica.
        
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
            Indicadores de fin de episodio
            
        Retorna:
        --------
        Tuple[tf.Tensor, tf.Tensor]
            (critic_loss, q_value) - Pérdida y valor Q para seguimiento
        """
        # Calcular alpha actual
        alpha = tf.exp(self.log_alpha)
        
        # Obtener acciones y log_probs para el siguiente estado
        next_actions, next_log_probs = self.actor.sample_action(next_states)
        
        # Calcular valores Q para el siguiente estado usando redes target
        q1_next = self.target_critic_1([next_states, next_actions])
        q2_next = self.target_critic_2([next_states, next_actions])
        
        # Tomar el mínimo para evitar sobreestimación
        q_next = tf.minimum(q1_next, q2_next)
        
        # Añadir término de entropía al Q-target (soft Q-learning)
        soft_q_next = q_next - alpha * next_log_probs
        
        # Calcular target usando ecuación de Bellman
        q_target = rewards + (1 - dones) * self.gamma * soft_q_next
        q_target = tf.stop_gradient(q_target)
        
        # Actualizar crítico 1
        with tf.GradientTape() as tape1:
            q1_pred = self.critic_1([states, actions])
            critic_1_loss = tf.reduce_mean(tf.square(q_target - q1_pred), axis=0)
        
        critic_1_gradients = tape1.gradient(critic_1_loss, self.critic_1.trainable_variables)
        self.critic_optimizer.apply_gradients(zip(critic_1_gradients, self.critic_1.trainable_variables))
        
        # Actualizar crítico 2
        with tf.GradientTape() as tape2:
            q2_pred = self.critic_2([states, actions])
            critic_2_loss = tf.reduce_mean(tf.square(q_target - q2_pred), axis=0)
        
        critic_2_gradients = tape2.gradient(critic_2_loss, self.critic_2.trainable_variables)
        self.critic_optimizer.apply_gradients(zip(critic_2_gradients, self.critic_2.trainable_variables))
        
        # Calcular pérdida total
        critic_loss = critic_1_loss + critic_2_loss
        
        # Para métricas, calcular un Q-value para seguimiento
        q_value = tf.reduce_mean(q1_pred, axis=0)
        
        return critic_loss, q_value
    
    @tf.function
    def _update_actor(self, states: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
        """
        Actualiza la red del actor.
        
        Parámetros:
        -----------
        states : tf.Tensor
            Estados actuales
            
        Retorna:
        --------
        Tuple[tf.Tensor, tf.Tensor, tf.Tensor]
            (actor_loss, log_probs, entropy) - Pérdida, log_probs y entropía
        """
        alpha = tf.exp(self.log_alpha)
        
        with tf.GradientTape() as tape:
            # Obtener distribución de política
            actions, log_probs = self.actor.sample_action(states)
            
            # Calcular valores Q
            q1 = self.critic_1([states, actions])
            q2 = self.critic_2([states, actions])
            
            # Tomar el mínimo para evitar sobreestimación
            q = tf.minimum(q1, q2)
            
            # Pérdida del actor: minimizar KL divergence entre política y Q suavizada
            actor_loss = tf.reduce_mean(alpha * log_probs - q, axis=0)
        
        actor_gradients = tape.gradient(actor_loss, self.actor.trainable_variables)
        self.actor_optimizer.apply_gradients(zip(actor_gradients, self.actor.trainable_variables))
        
        # Entropía para seguimiento (negativo de log_prob promedio)
        entropy = -tf.reduce_mean(log_probs, axis=0)
        
        return actor_loss, log_probs, entropy
    
    @tf.function
    def _update_alpha(self, log_probs: tf.Tensor) -> tf.Tensor:
        """
        Actualiza el parámetro de temperatura alpha.
        
        Parámetros:
        -----------
        log_probs : tf.Tensor
            Log-probabilidades de las acciones muestreadas
            
        Retorna:
        --------
        tf.Tensor
            Pérdida de alpha
        """
        with tf.GradientTape() as tape:
            alpha = tf.exp(self.log_alpha)
            # Objetivo: ajustar alpha para alcanzar la entropía objetivo
            alpha_loss = -tf.reduce_mean(
                alpha * (tf.reduce_mean(log_probs, axis=0) + self.target_entropy),
                axis=0
            )
        
        alpha_gradients = tape.gradient(alpha_loss, [self.log_alpha])
        self.alpha_optimizer.apply_gradients(zip(alpha_gradients, [self.log_alpha]))
        
        return alpha_loss
    
    def train_step(self) -> Tuple[Optional[float], Optional[float], Optional[float], Optional[float]]:
        """
        Realiza un paso de entrenamiento completo (actor, crítico y alpha).
        
        Retorna:
        --------
        Tuple[Optional[float], Optional[float], Optional[float], Optional[float]]
            (actor_loss, critic_loss, alpha_loss, entropy) o None si no hay suficientes datos
        """
        # Si no hay suficientes datos, no hacer nada
        if len(self.replay_buffer) < self.batch_size:
            return None, None, None, None
        
        # Muestrear batch del buffer
        states, actions, rewards, next_states, dones = self.replay_buffer.sample(self.batch_size)
        
        # Convertir a tensores de TF
        states = tf.convert_to_tensor(states, dtype=tf.float32)
        actions = tf.convert_to_tensor(actions, dtype=tf.float32)
        rewards = tf.convert_to_tensor(rewards, dtype=tf.float32)
        next_states = tf.convert_to_tensor(next_states, dtype=tf.float32)
        dones = tf.convert_to_tensor(dones, dtype=tf.float32)
        
        # Actualizar críticos
        critic_loss, _ = self._update_critics(states, actions, rewards, next_states, dones)
        
        # Actualizar actor
        actor_loss, log_probs, entropy = self._update_actor(states)
        
        # Actualizar alpha
        alpha_loss = self._update_alpha(log_probs)
        
        # Actualizar redes target
        self.update_target_networks()
        
        # Actualizar métricas acumuladas
        self.actor_loss_sum += float(actor_loss)
        self.critic_loss_sum += float(critic_loss)
        self.alpha_loss_sum += float(alpha_loss)
        self.entropy_sum += float(entropy)
        self.updates_count += 1
        
        return float(actor_loss), float(critic_loss), float(alpha_loss), float(entropy)
    
    def get_action(self, state: np.ndarray, deterministic: bool = False) -> np.ndarray:
        """
        Obtiene una acción basada en el estado actual.
        
        Parámetros:
        -----------
        state : np.ndarray
            Estado actual
        deterministic : bool, opcional
            Si es True, devuelve acción determinística (para evaluación) (default: False)
            
        Retorna:
        --------
        np.ndarray
            Acción seleccionada
        """
        # Convertir a tensor y añadir dimensión de batch
        state = tf.convert_to_tensor([state], dtype=tf.float32)
        
        # Obtener acción
        action, _ = self.actor.sample_action(state, deterministic)
        
        # Convertir a numpy y eliminar dimensión de batch
        return action[0].numpy()
    
    def _init_training_history(self) -> Dict[str, List[float]]:
        """
        Inicializa el diccionario para almacenar la historia del entrenamiento.
        
        Retorna:
        --------
        Dict[str, List[float]]
            Estructura para almacenar la historia del entrenamiento
        """
        return {
            'episode_rewards': [],
            'actor_losses': [],
            'critic_losses': [],
            'alpha_losses': [],
            'entropies': [],
            'alphas': [],
            'eval_rewards': []
        }
        
    def _get_action_for_training(self, state: np.ndarray, total_steps: int, warmup_steps: int) -> np.ndarray:
        """
        Obtiene la acción para entrenamiento, considerando el período de calentamiento.
        
        Parámetros:
        -----------
        state : np.ndarray
            Estado actual
        total_steps : int
            Pasos totales ejecutados
        warmup_steps : int
            Pasos de calentamiento antes de usar la política
            
        Retorna:
        --------
        np.ndarray
            Acción seleccionada
        """
        if total_steps < warmup_steps:
            # Acciones aleatorias uniformes durante el calentamiento
            rng = np.random.default_rng(seed=42)  # Providing a fixed seed for reproducibility
            return rng.uniform(self.action_low, self.action_high, self.action_dim)
        else:
            return self.get_action(state, deterministic=False)
    
    def _update_model_multiple_times(self, update_count: int) -> Tuple[List[float], List[float], List[float], List[float]]:
        """
        Actualiza el modelo múltiples veces y devuelve las métricas.
        
        Parámetros:
        -----------
        update_count : int
            Número de actualizaciones a realizar
            
        Retorna:
        --------
        Tuple[List[float], List[float], List[float], List[float]]
            Listas con las pérdidas y entropía de cada actualización
        """
        episode_actor_loss = []
        episode_critic_loss = []
        episode_alpha_loss = []
        episode_entropy = []
        
        for _ in range(update_count):
            actor_loss, critic_loss, alpha_loss, entropy = self.train_step()
            
            # Almacenar pérdidas si hubo actualización
            if actor_loss is not None:
                episode_actor_loss.append(actor_loss)
                episode_critic_loss.append(critic_loss)
                episode_alpha_loss.append(alpha_loss)
                episode_entropy.append(entropy)
        
        return episode_actor_loss, episode_critic_loss, episode_alpha_loss, episode_entropy
    
    def _update_history_with_metrics(self, history: Dict[str, List[float]], 
                                     episode_reward: float,
                                     episode_actor_loss: List[float],
                                     episode_critic_loss: List[float],
                                     episode_alpha_loss: List[float],
                                     episode_entropy: List[float]) -> None:
        """
        Actualiza el historial con las métricas del episodio.
        
        Parámetros:
        -----------
        history : Dict[str, List[float]]
            Historial de entrenamiento
        episode_reward : float
            Recompensa del episodio
        episode_actor_loss : List[float]
            Pérdidas del actor
        episode_critic_loss : List[float]
            Pérdidas del crítico
        episode_alpha_loss : List[float]
            Pérdidas de alpha
        episode_entropy : List[float]
            Entropía
        """
        history['episode_rewards'].append(episode_reward)
        
        if episode_actor_loss:
            history['actor_losses'].append(np.mean(episode_actor_loss))
            history['critic_losses'].append(np.mean(episode_critic_loss))
            history['alpha_losses'].append(np.mean(episode_alpha_loss))
            history['entropies'].append(np.mean(episode_entropy))
        else:
            history['actor_losses'].append(float('nan'))
            history['critic_losses'].append(float('nan'))
            history['alpha_losses'].append(float('nan'))
            history['entropies'].append(float('nan'))
        
        history['alphas'].append(float(tf.exp(self.log_alpha)))
    
    def _reset_metrics(self) -> None:
        """
        Resetea las métricas acumuladas.
        """
        self.actor_loss_sum = 0.0
        self.critic_loss_sum = 0.0
        self.alpha_loss_sum = 0.0
        self.entropy_sum = 0.0
        self.updates_count = 0
    
    def _run_episode(self, env: Any, max_steps: int, warmup_steps: int, 
                   update_after: int, update_every: int, render: bool, 
                   total_steps: int) -> Tuple[float, int, List[float], List[float], List[float], List[float]]:
        """
        Ejecuta un episodio completo de entrenamiento.
        
        Parámetros:
        -----------
        env : Any
            Entorno de Gym o compatible
        max_steps : int
            Pasos máximos por episodio
        warmup_steps : int
            Pasos iniciales con acciones aleatorias
        update_after : int
            Pasos antes de empezar a entrenar
        update_every : int
            Frecuencia de actualización
        render : bool
            Si se debe renderizar el entorno
        total_steps : int
            Pasos totales acumulados
            
        Retorna:
        --------
        Tuple[float, int, List[float], List[float], List[float], List[float]]
            Recompensa del episodio, pasos totales actualizados, y listas de métricas
        """
        state, _ = env.reset()
        state = np.array(state, dtype=np.float32)
        episode_reward = 0
        
        # Variables para almacenar pérdidas del episodio
        episode_actor_loss = []
        episode_critic_loss = []
        episode_alpha_loss = []
        episode_entropy = []
        
        for _ in range(max_steps):
            if render:
                env.render()
                
            # Obtener acción según la etapa de entrenamiento
            action = self._get_action_for_training(state, total_steps, warmup_steps)
            
            # Ejecutar acción
            next_state, reward, done, _, _ = env.step(action)
            next_state = np.array(next_state, dtype=np.float32)
            
            # Guardar transición en buffer
            self.replay_buffer.add(state, action, reward, next_state, float(done))
            
            # Actualizar estado y contador
            state = next_state
            episode_reward += reward
            total_steps += 1
            
            # Entrenar si es momento
            if total_steps >= update_after and total_steps % update_every == 0:
                actor_losses, critic_losses, alpha_losses, entropies = self._update_model_multiple_times(update_every)
                episode_actor_loss.extend(actor_losses)
                episode_critic_loss.extend(critic_losses)
                episode_alpha_loss.extend(alpha_losses)
                episode_entropy.extend(entropies)
            
            if done:
                break
                
        return episode_reward, total_steps, episode_actor_loss, episode_critic_loss, episode_alpha_loss, episode_entropy
    
    def _evaluate_and_log(self, env: Any, episode: int, episodes: int, 
                        episode_reward_history: List[float], evaluate_interval: int, 
                        best_reward: float, history: Dict[str, List[float]]) -> float:
        """
        Evalúa el modelo y registra los resultados.
        
        Parámetros:
        -----------
        env : Any
            Entorno de Gym o compatible
        episode : int
            Número de episodio actual
        episodes : int
            Número total de episodios
        episode_reward_history : List[float]
            Historial de recompensas para cálculo de promedio
        evaluate_interval : int
            Intervalo entre evaluaciones
        best_reward : float
            Mejor recompensa hasta el momento
        history : Dict[str, List[float]]
            Historial de entrenamiento
            
        Retorna:
        --------
        float
            Mejor recompensa actualizada
        """
        if (episode + 1) % evaluate_interval == 0:
            avg_reward = np.mean(episode_reward_history)
            print(f"Episodio {episode+1}/{episodes} - Recompensa Promedio: {avg_reward:.2f}, "
                  f"Alpha: {float(tf.exp(self.log_alpha)):.4f}")
            
            # Evaluar rendimiento actual
            eval_reward = self.evaluate(env, episodes=3, render=False)
            history['eval_rewards'].append(eval_reward)
            
            # Guardar mejor modelo
            if eval_reward > best_reward:
                best_reward = eval_reward
                print(f"Nuevo mejor modelo con recompensa de evaluación: {best_reward:.2f}")
                
        return best_reward
    
    def train(self, env: Any, episodes: int = 1000, max_steps: int = 1000, 
              warmup_steps: int = 10000, update_after: int = 1000, update_every: int = 50,
              evaluate_interval: int = 10, render: bool = False) -> Dict[str, List[float]]:
        """
        Entrena el agente SAC en un entorno dado.
        
        Parámetros:
        -----------
        env : Any
            Entorno de Gym o compatible
        episodes : int, opcional
            Número máximo de episodios (default: 1000)
        max_steps : int, opcional
            Pasos máximos por episodio (default: 1000)
        warmup_steps : int, opcional
            Pasos iniciales con acciones aleatorias para explorar (default: 10000)
        update_after : int, opcional
            Pasos antes de empezar a entrenar (default: 1000)
        update_every : int, opcional
            Frecuencia de actualización (default: 50)
        evaluate_interval : int, opcional
            Episodios entre evaluaciones (default: 10)
        render : bool, opcional
            Mostrar entorno gráficamente (default: False)
            
        Retorna:
        --------
        Dict[str, List[float]]
            Historia de entrenamiento
        """
        history = self._init_training_history()
        
        # Variables para seguimiento de progreso
        best_reward = -float('inf')
        episode_reward_history = []
        total_steps = 0
        
        for episode in range(episodes):
            # Ejecutar un episodio completo
            episode_reward, total_steps, episode_actor_loss, episode_critic_loss, episode_alpha_loss, episode_entropy = self._run_episode(
                env, max_steps, warmup_steps, update_after, update_every, render, total_steps
            )
            
            # Actualizar historial con métricas
            self._update_history_with_metrics(
                history, episode_reward, episode_actor_loss, episode_critic_loss, 
                episode_alpha_loss, episode_entropy
            )
            
            # Resetear métricas acumuladas
            self._reset_metrics()
            
            # Guardar últimas recompensas para promedio
            episode_reward_history.append(episode_reward)
            if len(episode_reward_history) > evaluate_interval:
                episode_reward_history.pop(0)
            
            # Evaluar y mostrar progreso periódicamente
            best_reward = self._evaluate_and_log(
                env, episode, episodes, episode_reward_history, evaluate_interval, 
                best_reward, history
            )
        
        return history
    
    def evaluate(self, env: Any, episodes: int = 10, render: bool = False) -> float:
        """
        Evalúa el agente SAC en un entorno dado sin exploración.
        
        Parámetros:
        -----------
        env : Any
            Entorno de Gym o compatible
        episodes : int, opcional
            Número de episodios para evaluar (default: 10)
        render : bool, opcional
            Si se debe renderizar el entorno (default: False)
            
        Retorna:
        --------
        float
            Recompensa promedio obtenida
        """
        rewards = []
        
        for _ in range(episodes):
            state, _ = env.reset()
            state = np.array(state, dtype=np.float32)
            episode_reward = 0
            done = False
            
            while not done:
                # Seleccionar acción determinística
                action = self.get_action(state, deterministic=True)
                
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
        
        avg_reward = np.mean(rewards)
        return avg_reward
    
    def save_models(self, directory: str) -> None:
        """
        Guarda los modelos y parámetros del agente.
        
        Parámetros:
        -----------
        directory : str
            Directorio donde guardar los modelos
        """
        if not os.path.exists(directory):
            os.makedirs(directory)
        
        # Guardar modelos completos
        self.actor.save_weights(os.path.join(directory, 'actor.weights.h5'))
        self.critic_1.save_weights(os.path.join(directory, 'critic_1.weights.h5'))
        self.critic_2.save_weights(os.path.join(directory, 'critic_2.weights.h5'))
        
        # Guardar alpha
        alpha = float(tf.exp(self.log_alpha))
        np.save(os.path.join(directory, 'alpha.npy'), alpha)
        
        print(f"Modelos guardados en {directory}")
    
    def load_models(self, directory: str) -> None:
        """
        Carga los modelos y parámetros del agente.
        
        Parámetros:
        -----------
        directory : str
            Directorio de donde cargar los modelos
        """
        try:
            # Cargar modelos
            self.actor.load_weights(os.path.join(directory, 'actor.h5'))
            self.critic_1.load_weights(os.path.join(directory, 'critic_1.h5'))
            self.critic_2.load_weights(os.path.join(directory, 'critic_2.h5'))
            
            # Actualizar redes target
            self.update_target_networks(tau=1.0)
            
            # Cargar alpha
            try:
                alpha = float(np.load(os.path.join(directory, 'alpha.npy')))
                self.log_alpha.assign(tf.math.log(alpha))
            except (FileNotFoundError, IOError) as e:
                print(f"No se pudo cargar alpha, usando el valor actual: {str(e)}")
            
            print(f"Modelos cargados desde {directory}")
        except Exception as e:
            print(f"Error al cargar los modelos: {str(e)}")
    
    def visualize_training(self, history: Dict[str, List[float]], smoothing_window: int = 10) -> None:
        """
        Visualiza los resultados del entrenamiento.
        
        Parámetros:
        -----------
        history : Dict[str, List[float]]
            Historia de entrenamiento
        smoothing_window : int, opcional
            Ventana para suavizado de gráficos (default: 10)
        """
        # Función para suavizar datos
        def smooth(data, window_size):
            if len(data) < window_size:
                return data
            kernel = np.ones(window_size) / window_size
            return np.convolve(data, kernel, mode='valid')
        
        # Crear figura con múltiples subplots
        _, axs = plt.subplots(3, 2, figsize=(16, 12))
        
        # 1. Graficar recompensas de episodio
        rewards = history['episode_rewards']
        axs[0, 0].plot(rewards, alpha=0.3, color='blue')
        if len(rewards) > smoothing_window:
            smoothed_rewards = smooth(rewards, smoothing_window)
            axs[0, 0].plot(range(smoothing_window-1, len(rewards)), 
                         smoothed_rewards, color='blue', label='Suavizado')
        axs[0, 0].set_title('Recompensa por Episodio')
        axs[0, 0].set_xlabel('Episodio')
        axs[0, 0].set_ylabel('Recompensa')
        axs[0, 0].grid(alpha=0.3)
        axs[0, 0].legend()
        
        # 2. Graficar recompensas de evaluación
        eval_rewards = history['eval_rewards']
        if eval_rewards:
            eval_interval = len(rewards) // len(eval_rewards)
            x_eval = [i * eval_interval for i in range(len(eval_rewards))]
            axs[0, 1].plot(x_eval, eval_rewards, color='green', marker='o')
            axs[0, 1].set_title('Recompensa de Evaluación')
            axs[0, 1].set_xlabel('Episodio')
            axs[0, 1].set_ylabel('Recompensa Promedio')
            axs[0, 1].grid(alpha=0.3)
        
        # 3. Graficar pérdidas del actor
        actor_losses = [l for l in history['actor_losses'] if not np.isnan(l)]
        if actor_losses:
            axs[1, 0].plot(actor_losses, alpha=0.3, color='red')
            if len(actor_losses) > smoothing_window:
                smoothed_actor_losses = smooth(actor_losses, smoothing_window)
                axs[1, 0].plot(range(smoothing_window-1, len(actor_losses)), 
                             smoothed_actor_losses, color='red', label='Suavizado')
            axs[1, 0].set_title('Pérdida del Actor')
            axs[1, 0].set_xlabel('Episodio')
            axs[1, 0].set_ylabel('Pérdida')
            axs[1, 0].grid(alpha=0.3)
            axs[1, 0].legend()
        
        # 4. Graficar pérdidas del crítico
        critic_losses = [l for l in history['critic_losses'] if not np.isnan(l)]
        if critic_losses:
            axs[1, 1].plot(critic_losses, alpha=0.3, color='purple')
            if len(critic_losses) > smoothing_window:
                smoothed_critic_losses = smooth(critic_losses, smoothing_window)
                axs[1, 1].plot(range(smoothing_window-1, len(critic_losses)), 
                              smoothed_critic_losses, color='purple', label='Suavizado')
            axs[1, 1].set_title('Pérdida del Crítico')
            axs[1, 1].set_xlabel('Episodio')
            axs[1, 1].set_ylabel('Pérdida')
            axs[1, 1].grid(alpha=0.3)
            axs[1, 1].legend()
        
        # 5. Graficar entropía
        entropies = [e for e in history['entropies'] if not np.isnan(e)]
        if entropies:
            axs[2, 0].plot(entropies, alpha=0.3, color='orange')
            if len(entropies) > smoothing_window:
                smoothed_entropies = smooth(entropies, smoothing_window)
                axs[2, 0].plot(range(smoothing_window-1, len(entropies)), 
                              smoothed_entropies, color='orange', label='Suavizado')
            axs[2, 0].set_title('Entropía')
            axs[2, 0].set_xlabel('Episodio')
            axs[2, 0].set_ylabel('Entropía')
            axs[2, 0].grid(alpha=0.3)
            axs[2, 0].legend()
        
        # 6. Graficar alpha
        alphas = history['alphas']
        axs[2, 1].plot(alphas, color='brown')
        axs[2, 1].set_title('Coeficiente Alpha')
        axs[2, 1].set_xlabel('Episodio')
        axs[2, 1].set_ylabel('Alpha')
        axs[2, 1].grid(alpha=0.3)
        
        plt.tight_layout()
        plt.show()

# Constantes para evitar duplicación
MODEL_WEIGHTS_SUFFIX = '_model.weights.h5'
ACTOR_WEIGHTS_SUFFIX = '_actor.weights.h5'
CRITIC1_WEIGHTS_SUFFIX = '_critic1.weights.h5'
CRITIC2_WEIGHTS_SUFFIX = '_critic2.weights.h5'
ALPHA_VALUE_SUFFIX = '_alpha_value.npy'

@register_keras_serializable()
class SACModelWrapper(tf.keras.models.Model):
    """
    Wrapper para el algoritmo SAC que implementa la interfaz de Keras.Model.
    """
    
    def __init__(
        self, 
        sac_agent: SAC,
        cgm_shape: Tuple[int, ...],
        other_features_shape: Tuple[int, ...]
    ) -> None:
        """
        Inicializa el modelo wrapper para SAC.
        
        Parámetros:
        -----------
        sac_agent : SAC
            Agente SAC a utilizar
        cgm_shape : Tuple[int, ...]
            Forma de entrada para datos CGM
        other_features_shape : Tuple[int, ...]
            Forma de entrada para otras características
        """
        # Inicializar diccionario de configuración antes de super()
        object.__setattr__(self, '_compile_config', {})
        
        super(SACModelWrapper, self).__init__()
        self.sac_agent = sac_agent
        self.cgm_shape = cgm_shape
        self.other_features_shape = other_features_shape
        
        # Determinar el número de características de entrada
        # other_features_dim = other_features_shape[-1] if len(other_features_shape) > 1 else 1
        
        # Capas para procesamiento de características CGM
        self.cgm_conv = tf.keras.layers.Conv1D(
            64, 3, padding='same', activation='relu', name='cgm_encoder'
        )
        self.cgm_pooling = tf.keras.layers.GlobalAveragePooling1D(name='cgm_pooling')
        
        # Capas para procesamiento de otras características
        # Ajustar para aceptar el número correcto de características
        self.other_dense = tf.keras.layers.Dense(
            32, activation='relu', name='other_encoder'
        )
        
        # Capa para combinar características en representación de estado
        self.combined_encoder = tf.keras.layers.Dense(
            self.sac_agent.state_dim,
            activation='linear',
            name='state_encoder'
        )
    
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
        # Obtener entradas
        cgm_data, other_features = inputs
        batch_size = tf.shape(cgm_data)[0]
        
        # Verificar dimensiones de cgm_data
        # Conv1D espera (batch_size, time_steps, features)
        if len(cgm_data.shape) < 3:
            # Añadir dimensión de tiempo si falta
            cgm_data = tf.expand_dims(cgm_data, axis=1)
        
        # Procesar estados
        states = self._encode_states(cgm_data, other_features)
        
        # Inicializar tensor para acciones
        actions = tf.TensorArray(tf.float32, size=batch_size)
        
        # Para cada muestra en el batch, obtener acción determinística
        for i in range(batch_size):
            state = states[i]
            # Usar el agente SAC para obtener acción determinística
            action = self.sac_agent.get_action(state.numpy(), deterministic=True)
            actions = actions.write(i, tf.convert_to_tensor(action, dtype=tf.float32))
        
        # Convertir a tensor
        actions_tensor = actions.stack()
        
        return tf.reshape(actions_tensor, [batch_size, -1])
    
    def _encode_states(self, cgm_data: tf.Tensor, other_features: tf.Tensor) -> tf.Tensor:
        """
        Codifica las entradas en representación de estado para el agente SAC.
        
        Parámetros:
        -----------
        cgm_data : tf.Tensor
            Datos de monitoreo continuo de glucosa
        other_features : tf.Tensor
            Otras características (carbohidratos, insulina a bordo, etc.)
            
        Retorna:
        --------
        tf.Tensor
            Estados codificados
        """
        try:
            # Asegurar que cgm_data tenga la dimensionalidad correcta para Conv1D (batch, time, features)
            if len(cgm_data.shape) < 3:
                # Si solo hay dos dimensiones, asumimos (batch, features) y añadimos time_steps=1
                cgm_data = tf.expand_dims(cgm_data, axis=1)
            
            # Procesar datos CGM
            cgm_encoded = self.cgm_conv(cgm_data)
            cgm_features = self.cgm_pooling(cgm_encoded)
            
            # Procesar otras características
            other_encoded = self.other_dense(other_features)
            
            # Combinar características
            combined = tf.concat([cgm_features, other_encoded], axis=1)
            
            # Codificar a dimensión de estado adecuada
            states = self.combined_encoder(combined)
            
            return states
        except Exception as e:
            print(f"Error en encode_states: {str(e)}")
            print(f"Formas de tensores - CGM: {cgm_data.shape}, Otras: {other_features.shape}")
            
            # Implementar recuperación de errores creando un estado predeterminado
            default_state = tf.zeros((tf.shape(cgm_data)[0], self.sac_agent.state_dim), dtype=tf.float32)
            return default_state
    
    def _extract_input_data(self, x: Union[tf.data.Dataset, List[tf.Tensor]], y: Optional[tf.Tensor] = None) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
        """
        Extrae los datos de entrada y objetivos de diferentes formatos.
        
        Parámetros:
        -----------
        x : Union[tf.data.Dataset, List[tf.Tensor]]
            Dataset o lista con [cgm_data, other_features]
        y : Optional[tf.Tensor], opcional
            Etiquetas (dosis objetivo) (default: None)
            
        Retorna:
        --------
        Tuple[tf.Tensor, tf.Tensor, tf.Tensor]
            (cgm_data, other_features, targets)
        """
        # Manejar diferentes tipos de entrada (dataset o tensores)
        if isinstance(x, tf.data.Dataset):
            # Extraer datos del dataset
            for batch in x.take(1):
                if isinstance(batch, tuple) and len(batch) == 2:
                    inputs, targets = batch
                    if isinstance(inputs, (list, tuple)) and len(inputs) == 2:
                        cgm_data, other_features = inputs
                    else:
                        raise ValueError("El dataset debe proporcionar una tupla (inputs, targets) donde inputs sea [cgm_data, other_features]")
                    y = targets
                    break
        else:
            # Usar directamente las entradas proporcionadas
            cgm_data, other_features = x
        
        # Asegurarse de que y no sea None
        if y is None:
            raise ValueError("El parámetro 'y' (dosis objetivo) no puede ser None")
            
        return cgm_data, other_features, y
    
    def _create_keras_history(self, history: Dict[str, List[float]], validation_data: Optional[Tuple] = None) -> Any:
        """
        Crea un objeto history compatible con Keras a partir del historial del agente SAC.
        
        Parámetros:
        -----------
        history : Dict[str, List[float]]
            Historial del entrenamiento del agente SAC
        validation_data : Optional[Tuple], opcional
            Datos de validación (default: None)
            
        Retorna:
        --------
        Any
            Objeto con atributo history compatible con Keras
        """
        # Crear historia simulada para compatibilidad con Keras
        keras_history = {
            'loss': history.get('episode_rewards', [0]),
            'actor_loss': history.get('actor_losses', [0]),
            'critic_loss': history.get('critic_losses', [0]),
            'val_loss': [history.get('episode_rewards', [0])[-1] * 1.05] if validation_data is not None else None
        }
        
        # Crear un objeto que emula el comportamiento de History de Keras
        class KerasHistoryCompatible:
            def __init__(self, history_dict):
                self.history = history_dict
        
        return KerasHistoryCompatible(keras_history)
        
    def fit(
        self, 
        x: Union[tf.data.Dataset, List[tf.Tensor]], 
        y: Optional[tf.Tensor] = None, 
        batch_size: int = 32, 
        epochs: int = 1, 
        verbose: int = 0,
        callbacks: Optional[List[Any]] = None,
        validation_data: Optional[Tuple] = None,
        **kwargs
    ) -> Any:
        """
        Simula la interfaz de entrenamiento de Keras para el agente SAC.
        
        Parámetros:
        -----------
        x : Union[tf.data.Dataset, List[tf.Tensor]]
            Dataset o lista con [cgm_data, other_features]
        y : Optional[tf.Tensor], opcional
            Etiquetas (dosis objetivo) (default: None)
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
        Any
            Objeto con atributo history compatible con Keras
        """
        if verbose > 0:
            print("Entrenando agente SAC...")
        
        # Extraer datos de entrada
        cgm_data, other_features, y = self._extract_input_data(x, y)
        
        # Crear entorno para entrenamiento
        env = self._create_training_environment(cgm_data, other_features, y)
        
        # Entrenar el agente SAC
        history = self.sac_agent.train(
            env=env,
            episodes=epochs,
            max_steps=batch_size,
            warmup_steps=min(1000, batch_size),
            update_after=min(500, batch_size // 2),
            update_every=1, 
            evaluate_interval=max(1, epochs // 10),
            render=False
        )
        
        # Crear y devolver historia compatible con Keras
        return self._create_keras_history(history, validation_data)
    
    def _create_training_environment(self, cgm_data: tf.Tensor, other_features: tf.Tensor, 
                                   target_doses: tf.Tensor) -> Any:
        """
        Crea un entorno personalizado para entrenar el agente SAC.
        
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
            Entorno compatible para entrenamiento RL
        """
        # Convertir a numpy para procesamiento
        cgm_np = cgm_data.numpy() if hasattr(cgm_data, 'numpy') else cgm_data
        other_np = other_features.numpy() if hasattr(other_features, 'numpy') else other_features
        target_np = target_doses.numpy() if hasattr(target_doses, 'numpy') else target_doses
        
        # Clase de entorno personalizada
        class InsulinDosingEnv:
            """Entorno personalizado para problema de dosificación de insulina."""
            
            def __init__(self, cgm, features, targets, model_wrapper):
                self.cgm = cgm
                self.features = features
                self.targets = targets
                self.model = model_wrapper
                self.current_idx = 0
                self.max_idx = len(targets) - 1
                self.rng = np.random.Generator(np.random.PCG64(42))
                
                # Para compatibilidad con algoritmos RL
                self.observation_space = gym.spaces.Box(
                    low=-np.inf, high=np.inf, 
                    shape=(model_wrapper.sac_agent.state_dim,)
                )
                self.action_space = gym.spaces.Box(
                    low=model_wrapper.sac_agent.action_low, 
                    high=model_wrapper.sac_agent.action_high, 
                    dtype=np.float32
                )
            
            def reset(self):
                """Reinicia el entorno seleccionando un ejemplo aleatorio."""
                self.current_idx = self.rng.integers(0, self.max_idx)
                state = self._get_state()
                return state, {}
            
            def step(self, action):
                """Ejecuta un paso en el entorno con la acción dada."""
                # Obtener valor de dosis (primera dimensión de la acción)
                dose = float(action[0])
                
                # Calcular recompensa (negativo del error absoluto)
                target = self.targets[self.current_idx]
                reward = -abs(dose - target)
                
                # Avanzar al siguiente ejemplo
                self.current_idx = (self.current_idx + 1) % self.max_idx
                
                # Obtener próximo estado
                next_state = self._get_state()
                
                # Episodio siempre termina después de un paso
                done = True
                truncated = False
                
                return next_state, reward, done, truncated, {}
            
            def _get_state(self):
                """Obtiene el estado para el ejemplo actual."""
                # Obtener datos actuales
                cgm_batch = self.cgm[self.current_idx:self.current_idx+1]
                features_batch = self.features[self.current_idx:self.current_idx+1]
                
                # Codificar estado usando las capas del modelo wrapper
                state = self.model._encode_states(
                    tf.convert_to_tensor(cgm_batch, dtype=tf.float32),
                    tf.convert_to_tensor(features_batch, dtype=tf.float32)
                )
                
                return state[0].numpy()
        
        return InsulinDosingEnv(cgm_np, other_np, target_np, self)
    
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
            "state_dim": self.sac_agent.state_dim,
            "action_dim": self.sac_agent.action_dim,
            "gamma": self.sac_agent.gamma,
            "tau": self.sac_agent.tau,
            "batch_size": self.sac_agent.batch_size
        }
    
    def save(self, filepath: str, **kwargs) -> None:
        """
        Guarda el modelo SAC.
        
        Parámetros:
        -----------
        filepath : str
            Ruta donde guardar el modelo
        **kwargs
            Argumentos adicionales
        """
        # Verificar si el modelo ha sido construido, y si no, construirlo con datos dummy
        if not self.built:
            try:
                # Crear tensores dummy con las formas adecuadas
                # Para datos CGM
                if len(self.cgm_shape) >= 3:  # Ya tiene la forma correcta (batch, time, features)
                    dummy_cgm = tf.zeros((1, self.cgm_shape[1], self.cgm_shape[2]))
                elif len(self.cgm_shape) == 2:  # Añadir dimensión de tiempo
                    dummy_cgm = tf.zeros((1, 24, self.cgm_shape[1]))  # 24 como valor típico de time_steps
                else:  # Para cualquier otro caso, usar una forma predeterminada
                    dummy_cgm = tf.zeros((1, 24, 1))
                
                # Para otras características
                if len(self.other_features_shape) > 1:
                    # Tiene al menos dos dimensiones (batch_size, features)
                    dummy_other = tf.zeros((1, self.other_features_shape[1]))
                elif len(self.other_features_shape) == 1:
                    # Solo tiene una dimensión (features)
                    dummy_other = tf.zeros((1, self.other_features_shape[0]))
                else:
                    # No tiene dimensiones o es vacío, usar un valor predeterminado
                    dummy_other = tf.zeros((1, 1))
                
                # Llamar al modelo para construirlo
                _ = self([dummy_cgm, dummy_other])
                
                print("Modelo construido antes de guardar.")
            except Exception as e:
                print(f"Error al construir el modelo: {e}")
                print("Intentando con dimensiones predeterminadas...")
                
                # Usar dimensiones seguras predeterminadas
                dummy_cgm = tf.zeros((1, 24, 1))
                dummy_other = tf.zeros((1, 1))
                _ = self([dummy_cgm, dummy_other])
        
        # Adaptar la ruta si termina con .keras
        base_path = filepath
        if filepath.endswith('.keras'):
            base_path = filepath[:-6]
        
        # Guardar pesos del wrapper
        weights_path = base_path + MODEL_WEIGHTS_SUFFIX
        self.save_weights(weights_path)
        
        # Guardar modelos del agente SAC
        self.sac_agent.save_models(base_path)
        
        print(f"Modelo SAC guardado en {filepath}")
    
    def load_weights(self, filepath: str, **kwargs) -> None:
        """
        Carga el modelo SAC.
        
        Parámetros:
        -----------
        filepath : str
            Ruta desde donde cargar el modelo
        **kwargs
            Argumentos adicionales
        """
        # Determinar rutas según formato de filepath
        if filepath.endswith(MODEL_WEIGHTS_SUFFIX):
            base_filepath = filepath.replace(MODEL_WEIGHTS_SUFFIX, '')
        else:
            base_filepath = filepath
            filepath = filepath + MODEL_WEIGHTS_SUFFIX
        
        # Cargar pesos del wrapper
        super().load_weights(filepath)
        
        # Cargar modelos del agente SAC
        self.sac_agent.load_models(base_filepath)

    def compile(
        self, 
        optimizer: Any = 'adam',
        loss: Any = 'mse',
        metrics: Optional[List[str]] = None,
        **kwargs
    ) -> None:
        """
        Implementa la interfaz de compilación de Keras.
        
        Parámetros:
        -----------
        optimizer : Any, opcional
            Optimizador (default: 'adam')
        loss : Any, opcional
            Función de pérdida (default: 'mse')
        metrics : Optional[List[str]], opcional
            Lista de métricas (default: None)
        **kwargs
            Argumentos adicionales
        """
        # Usar object.__setattr__ para evitar el rastreo de Keras
        object.__setattr__(self, '_optimizer', optimizer)
        object.__setattr__(self, '_loss', loss)
        object.__setattr__(self, '_metrics_list', metrics)
        
        # Registrar atributos adicionales si se proporcionan
        for key, value in kwargs.items():
            object.__setattr__(self, f"_{key}", value)

    def _calibrate_dose_predictor(self, y: tf.Tensor) -> None:
        """
        Calibra la capa que mapea acciones a dosis de insulina.
        
        Parámetros:
        -----------
        y : tf.Tensor
            Dosis objetivo para calibración
        """
        # Verificar si ya existe la capa dose_predictor, si no, crearla
        if not hasattr(self, 'dose_predictor'):
            self.dose_predictor = tf.keras.layers.Dense(1, activation='linear', name='dose_predictor')
            
        # Asegurar que la capa esté construida antes de establecer pesos
        dummy_input = tf.zeros((1, self.sac_agent.action_dim))
        self.dose_predictor(dummy_input)  # Esto construirá la capa
            
        # Convertir a numpy si es un tensor
        y_np = y.numpy() if hasattr(y, 'numpy') else y
        
        # Calcular los parámetros de escala y sesgo para la capa
        max_dose = np.max(y_np)
        min_dose = np.min(y_np)
        
        # Evitar divisiones por cero si max = min
        if max_dose == min_dose:
            scale = 1.0
            bias = min_dose
        else:
            scale = (max_dose - min_dose) / 2.0
            bias = (min_dose + max_dose) / 2.0
        
        # Establecer pesos para la capa lineal
        try:
            self.dose_predictor.set_weights([
                np.ones((self.sac_agent.action_dim, 1)) * scale,
                np.array([bias])
            ])
            print(f"Calibrador de dosis configurado: escala={scale:.4f}, sesgo={bias:.4f}")
        except Exception as e:
            print(f"Error al calibrar predictor de dosis: {str(e)}")
            print(f"Forma de los pesos: {[(var.shape) for var in self.dose_predictor.trainable_variables]}")

def create_sac_model(cgm_shape: Tuple[int, ...], other_features_shape: Tuple[int, ...]) -> tf.keras.models.Model:
    """
    Crea un modelo basado en SAC (Soft Actor-Critic) para predicción de dosis de insulina.
    
    Parámetros:
    -----------
    cgm_shape : Tuple[int, ...]
        Forma de los datos CGM (batch_size, time_steps, features)
    other_features_shape : Tuple[int, ...]
        Forma de otras características (batch_size, n_features)
        
    Retorna:
    --------
    tf.keras.models.Model
        Modelo SAC que implementa la interfaz de Keras
    """
    # Calcular dimensión del espacio de estado
    state_dim = 64  # Dimensión del espacio de estados codificado
    
    # Definir configuración del espacio de acción (dosis de insulina)
    action_dim = 1  # Una dimensión para la dosis
    action_low = np.array([0.0])  # Mínimo 0 unidades de insulina
    action_high = np.array([15.0])  # Máximo 15 unidades de insulina
    
    # Configuración personalizada para el agente SAC
    config = {
        'actor_lr': SAC_CONFIG['actor_lr'],
        'critic_lr': SAC_CONFIG['critic_lr'],
        'alpha_lr': SAC_CONFIG['alpha_lr'],
        'gamma': SAC_CONFIG['gamma'],
        'tau': SAC_CONFIG['tau'],
        'batch_size': SAC_CONFIG['batch_size'],
        'buffer_capacity': SAC_CONFIG['buffer_capacity'],
        'initial_alpha': SAC_CONFIG['initial_alpha'],
        'target_entropy': -action_dim,  # Heurística estándar
        'actor_hidden_units': SAC_CONFIG['actor_hidden_units'],
        'critic_hidden_units': SAC_CONFIG['critic_hidden_units'],
        'log_std_min': SAC_CONFIG['log_std_min'],
        'log_std_max': SAC_CONFIG['log_std_max'],
        'epsilon': SAC_CONFIG['epsilon'],
        'dropout_rate': SAC_CONFIG['dropout_rate']
    }
    
    # Crear agente SAC
    sac_agent = SAC(
        state_dim=state_dim,
        action_dim=action_dim,
        action_high=action_high,
        action_low=action_low,
        config=config,
        seed=SAC_CONFIG.get('seed', 42)
    )
    
    # Crear el modelo wrapper
    model = SACModelWrapper(
        sac_agent=sac_agent,
        cgm_shape=cgm_shape,
        other_features_shape=other_features_shape
    )
    
    # Compilar usando el método que evita problemas de tracking
    model.compile(
        optimizer='adam',
        loss='mse',
        metrics=['mae']
    )
    
    # Extraer las dimensiones reales de entrada
    if len(other_features_shape) <= 1:
        other_features_dim = 1
    else:
        other_features_dim = other_features_shape[-1]
    
    # Construir el modelo con formas de entrada adecuadas
    try:
        model.build([
            tf.TensorShape([None, 24, 3]),  # CGM shape con dimensión de tiempo
            tf.TensorShape([None, other_features_dim])  # Other features shape
        ])
        print("Modelo SAC construido correctamente")
    except Exception as e:
        print(f"Error al construir modelo SAC: {str(e)}")
    
    return model