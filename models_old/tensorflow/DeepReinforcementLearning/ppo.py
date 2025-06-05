import os, sys
import tensorflow as tf
import numpy as np
import gym
from keras._tf_keras.keras.models import Model
from keras._tf_keras.keras.layers import (
    Input, Dense, Conv1D, LSTM, Flatten, Concatenate,
    BatchNormalization, Dropout, LayerNormalization, GlobalAveragePooling1D
)
from keras._tf_keras.keras.optimizers import Adam
from keras._tf_keras.keras.losses import MeanSquaredError
from keras._tf_keras.keras.callbacks import EarlyStopping
from keras._tf_keras.keras.saving import register_keras_serializable
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Any, Optional, Union, Callable

PROJECT_ROOT = os.path.abspath(os.getcwd())
sys.path.append(PROJECT_ROOT) 

from config.models_config_old import PPO_CONFIG
from constants.constants import CONST_DEFAULT_SEED, CONST_DEFAULT_EPOCHS, CONST_DEFAULT_BATCH_SIZE

class ActorCriticModel(Model):
    """
    Modelo Actor-Crítico para PPO que divide la arquitectura en redes para
    política (actor) y valor (crítico).
    
    Parámetros:
    -----------
    state_dim : int
        Dimensión del espacio de estados
    action_dim : int
        Dimensión del espacio de acciones
    hidden_units : Optional[List[int]], opcional
        Unidades en capas ocultas (default: None)
    """
    def __init__(self, 
                state_dim: int, 
                action_dim: int, 
                hidden_units: Optional[List[int]] = None) -> None:
        super(ActorCriticModel, self).__init__()
        
        # Valores predeterminados para capas ocultas
        if hidden_units is None:
            hidden_units = PPO_CONFIG['hidden_units']
        
        # Capas compartidas para procesamiento de estados
        self.shared_layers = []
        for i, units in enumerate(hidden_units[:2]):
            self.shared_layers.append(Dense(units, activation='tanh', name=f'shared_dense_{i}'))
            self.shared_layers.append(LayerNormalization(epsilon=PPO_CONFIG['epsilon'], name=f'shared_ln_{i}'))
            self.shared_layers.append(Dropout(PPO_CONFIG['dropout_rate'], name=f'shared_dropout_{i}'))
        
        # Red del Actor (política)
        self.actor_layers = []
        for i, units in enumerate(hidden_units[2:]):
            self.actor_layers.append(Dense(units, activation='tanh', name=f'actor_dense_{i}'))
            self.actor_layers.append(LayerNormalization(epsilon=PPO_CONFIG['epsilon'], name=f'actor_ln_{i}'))
        
        # Capa de salida del actor (mu y sigma para política gaussiana)
        self.mu = Dense(action_dim, activation='linear', name='actor_mu')
        self.log_sigma = Dense(action_dim, activation='linear', name='actor_log_sigma')
        
        # Red del Crítico (valor)
        self.critic_layers = []
        for i, units in enumerate(hidden_units[2:]):
            self.critic_layers.append(Dense(units, activation='tanh', name=f'critic_dense_{i}'))
            self.critic_layers.append(LayerNormalization(epsilon=PPO_CONFIG['epsilon'], name=f'critic_ln_{i}'))
        
        # Capa de salida del crítico (valor del estado)
        self.value = Dense(1, activation='linear', name='critic_value')
    
    def call(self, inputs: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
        """
        Realiza el forward pass del modelo actor-crítico.
        
        Parámetros:
        -----------
        inputs : tf.Tensor
            Tensor de estados de entrada
            
        Retorna:
        --------
        Tuple[tf.Tensor, tf.Tensor, tf.Tensor]
            (mu, sigma, value) - Parámetros de la distribución de política y valor estimado
        """
        x = inputs
        
        # Capas compartidas
        for layer in self.shared_layers:
            x = layer(x)
        
        # Red del Actor
        actor_x = x
        for layer in self.actor_layers:
            actor_x = layer(actor_x)
        
        mu = self.mu(actor_x)
        log_sigma = self.log_sigma(actor_x)
        sigma = tf.exp(log_sigma)
        
        # Red del Crítico
        critic_x = x
        for layer in self.critic_layers:
            critic_x = layer(critic_x)
        
        value = self.value(critic_x)
        
        return mu, sigma, value
    
    def get_action(self, state: np.ndarray, deterministic: bool = False) -> np.ndarray:
        """
        Obtiene una acción basada en el estado actual.
        
        Parámetros:
        -----------
        state : np.ndarray
            El estado actual
        deterministic : bool, opcional
            Si es True, devuelve la acción con máxima probabilidad (default: False)
        
        Retorna:
        --------
        np.ndarray
            Una acción muestreada de la distribución de política
        """
        state = tf.convert_to_tensor([state], dtype=tf.float32)
        mu, sigma, _ = self.call(state)
        
        if deterministic:
            return mu[0].numpy()
        
        # Muestrear de la distribución normal
        dist = tf.random.normal(shape=mu.shape)
        action = mu + sigma * dist
        
        return action[0].numpy()
    
    def get_value(self, state: np.ndarray) -> np.ndarray:
        """
        Obtiene el valor estimado para un estado.
        
        Parámetros:
        -----------
        state : np.ndarray
            El estado para evaluar
        
        Retorna:
        --------
        np.ndarray
            El valor estimado del estado
        """
        state = tf.convert_to_tensor([state], dtype=tf.float32)
        _, _, value = self.call(state)
        return value[0].numpy()


class PPO:
    """
    Implementación de Proximal Policy Optimization (PPO).
    
    Esta implementación utiliza el clipping de PPO para actualizar la política
    y un estimador de ventaja generalizada (GAE) para mejorar el aprendizaje.
    
    Parámetros:
    -----------
    state_dim : int
        Dimensión del espacio de estados
    action_dim : int
        Dimensión del espacio de acciones
    learning_rate : float, opcional
        Tasa de aprendizaje (default: PPO_CONFIG['learning_rate'])
    gamma : float, opcional
        Factor de descuento (default: PPO_CONFIG['gamma'])
    epsilon : float, opcional
        Parámetro de clipping para PPO (default: PPO_CONFIG['clip_epsilon'])
    hidden_units : Optional[List[int]], opcional
        Unidades en capas ocultas (default: None)
    entropy_coef : float, opcional
        Coeficiente para término de entropía (default: PPO_CONFIG['entropy_coef'])
    value_coef : float, opcional
        Coeficiente para pérdida de valor (default: PPO_CONFIG['value_coef'])
    max_grad_norm : Optional[float], opcional
        Norma máxima para clipping de gradientes (default: PPO_CONFIG['max_grad_norm'])
    seed : int, opcional
        Semilla para reproducibilidad (default: 42)
    """
    def __init__(
        self, 
        state_dim: int, 
        action_dim: int,
        learning_rate: float = PPO_CONFIG['learning_rate'],
        gamma: float = PPO_CONFIG['gamma'],
        epsilon: float = PPO_CONFIG['clip_epsilon'],
        hidden_units: Optional[List[int]] = None,
        entropy_coef: float = PPO_CONFIG['entropy_coef'],
        value_coef: float = PPO_CONFIG['value_coef'],
        max_grad_norm: Optional[float] = PPO_CONFIG['max_grad_norm'],
        seed: int = CONST_DEFAULT_SEED
    ) -> None:
        # Configurar semillas para reproducibilidad
        tf.random.set_seed(seed)
        np.random.seed(seed)
        
        # Parámetros del modelo
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.epsilon = epsilon
        self.entropy_coef = entropy_coef
        self.value_coef = value_coef
        self.max_grad_norm = max_grad_norm
        self.seed = seed
        
        # Valores predeterminados para capas ocultas
        if hidden_units is None:
            self.hidden_units = PPO_CONFIG['hidden_units']
        else:
            self.hidden_units = hidden_units
        
        # Crear modelo y optimizador
        self.model = ActorCriticModel(state_dim, action_dim, self.hidden_units)
        self.optimizer = Adam(learning_rate=learning_rate)
        
        # Métricas
        self.policy_loss_metric = tf.keras.metrics.Mean('policy_loss')
        self.value_loss_metric = tf.keras.metrics.Mean('value_loss')
        self.entropy_metric = tf.keras.metrics.Mean('entropy')
        self.total_loss_metric = tf.keras.metrics.Mean('total_loss')
        
        # Inicializar modelo para evitar errores en la primera llamada
        dummy_state = np.zeros((1, state_dim), dtype=np.float32)
        self.model(dummy_state)
    
    def log_prob(self, mu: tf.Tensor, sigma: tf.Tensor, actions: tf.Tensor) -> tf.Tensor:
        """
        Calcula el logaritmo de la probabilidad de acciones bajo una política gaussiana.
        
        Parámetros:
        -----------
        mu : tf.Tensor
            Media de la distribución gaussiana
        sigma : tf.Tensor
            Desviación estándar de la distribución gaussiana
        actions : tf.Tensor
            Acciones para calcular su probabilidad
        
        Retorna:
        --------
        tf.Tensor
            Logaritmo de probabilidad de las acciones
        """
        logp_normal = -0.5 * tf.square((actions - mu) / sigma) - 0.5 * tf.math.log(2.0 * np.pi) - tf.math.log(sigma)
        return tf.reduce_sum(logp_normal, axis=-1, keepdims=True)
    
    @tf.function(input_signature=[
        tf.TensorSpec(shape=[None, None], dtype=tf.float32),  # states
        tf.TensorSpec(shape=[None, None], dtype=tf.float32),  # actions
        tf.TensorSpec(shape=[None, None], dtype=tf.float32),  # old_log_probs
        tf.TensorSpec(shape=[None], dtype=tf.float32),        # rewards
        tf.TensorSpec(shape=[None], dtype=tf.float32),        # advantages
        tf.TensorSpec(shape=[None], dtype=tf.float32)         # values
    ])
    def train_step(self, states: tf.Tensor, actions: tf.Tensor, old_log_probs: tf.Tensor, 
                  rewards: tf.Tensor, advantages: tf.Tensor, values: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor]:
        """
        Realiza un paso de entrenamiento para actualizar el modelo.
        
        Parámetros:
        -----------
        states : tf.Tensor
            Estados observados en el entorno
        actions : tf.Tensor
            Acciones tomadas para esos estados
        old_log_probs : tf.Tensor
            Log de probabilidades de acciones bajo la política antigua
        rewards : tf.Tensor
            Recompensas recibidas (o retornos calculados)
        advantages : tf.Tensor
            Ventajas estimadas
        values : tf.Tensor
            Valores antiguos estimados por el crítico
        
        Retorna:
        --------
        Tuple[tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor]
            (pérdida total, pérdida de política, pérdida de valor, entropía)
        """
        with tf.GradientTape() as tape:
            # Pasar estados por el modelo
            mu, sigma, new_values = self.model(states, training=True)
            
            # Calcular nueva probabilidad de acciones
            new_log_probs = self.log_prob(mu, sigma, actions)
            
            # Ratio entre nuevas y antiguas probabilidades
            ratio = tf.exp(new_log_probs - old_log_probs)
            
            # Términos del clipping de PPO
            p1 = ratio * advantages
            p2 = tf.clip_by_value(ratio, 1 - self.epsilon, 1 + self.epsilon) * advantages
            
            # Pérdida de política (negativa porque queremos maximizar)
            policy_loss = -tf.reduce_mean(tf.minimum(p1, p2), axis=0)
            
            # Pérdida de valor (predicción vs retorno real)
            # Usamos la recompensa total como objetivo para el crítico
            value_loss = tf.reduce_mean(tf.square(rewards - new_values), axis=0)
            
            # Término de entropía para fomentar la exploración
            entropy = tf.reduce_mean(
                tf.reduce_sum(
                    0.5 * tf.math.log(2.0 * np.pi * tf.square(sigma)) + 0.5,
                    axis=-1
                ),
                axis=0
            )
            
            # Pérdida total combinada
            total_loss = policy_loss + self.value_coef * value_loss - self.entropy_coef * entropy
            
        # Calcular gradientes y actualizar pesos
        grads = tape.gradient(total_loss, self.model.trainable_variables)
        
        # Clipping de gradientes para estabilidad
        if self.max_grad_norm is not None:
            grads, _ = tf.clip_by_global_norm(grads, self.max_grad_norm)
            
        self.optimizer.apply_gradients(zip(grads, self.model.trainable_variables))
        
        # Actualizar métricas
        self.policy_loss_metric.update_state(policy_loss)
        self.value_loss_metric.update_state(value_loss)
        self.entropy_metric.update_state(entropy)
        self.total_loss_metric.update_state(total_loss)
        
        return total_loss, policy_loss, value_loss, entropy
    
    def compute_gae(self, rewards: np.ndarray, values: np.ndarray, next_values: np.ndarray, 
                  dones: np.ndarray, gamma: float = 0.99, lam: float = 0.95) -> Tuple[np.ndarray, np.ndarray]:
        """
        Calcula el Estimador de Ventaja Generalizada (GAE).
        
        Parámetros:
        -----------
        rewards : np.ndarray
            Recompensas recibidas
        values : np.ndarray
            Valores estimados para los estados actuales
        next_values : np.ndarray
            Valores estimados para los estados siguientes
        dones : np.ndarray
            Indicadores de fin de episodio
        gamma : float, opcional
            Factor de descuento (default: 0.99)
        lam : float, opcional
            Factor lambda para GAE (default: 0.95)
        
        Retorna:
        --------
        Tuple[np.ndarray, np.ndarray]
            (ventajas, retornos) - Ventajas y retornos calculados
        """
        advantages = np.zeros_like(rewards)
        last_gae = 0
        
        for t in reversed(range(len(rewards))):
            if t == len(rewards) - 1:
                next_value = next_values[t]
            else:
                next_value = values[t + 1]
                
            delta = rewards[t] + gamma * next_value * (1 - dones[t]) - values[t]
            advantages[t] = last_gae = delta + gamma * lam * (1 - dones[t]) * last_gae
            
        returns = advantages + values
        
        # Normalizar ventajas
        advantages = (advantages - np.mean(advantages)) / (np.std(advantages) + 1e-10)
        
        return advantages, returns
    
    def _collect_trajectories(self, env: Any, steps_per_epoch: int) -> Tuple[List[np.ndarray], Dict[str, List[float]]]:
        """
        Recolecta trayectorias de experiencia en el entorno.
        
        Parámetros:
        -----------
        env : Any
            Entorno de OpenAI Gym o compatible
        steps_per_epoch : int
            Número de pasos a ejecutar
            
        Retorna:
        --------
        Tuple[List[np.ndarray], Dict[str, List[float]]]
            (datos_trayectoria, historial_episodios) - Datos recopilados y métricas por episodio
        """
        # Contenedores para almacenar experiencias
        states = []
        actions = []
        rewards = []
        values = []
        dones = []
        next_values = []
        log_probs = []
        
        # Para tracking de episodios
        episode_rewards = []
        episode_lengths = []
        episode_reward = 0
        episode_length = 0
        
        # Recolectar experiencias
        state, _ = env.reset()
        for _ in range(steps_per_epoch):
            episode_length += 1
            states.append(state)
            
            # Obtener acción y valor
            action = self.model.get_action(state)
            mu, sigma, value = self.model(tf.convert_to_tensor([state], dtype=tf.float32))
            
            # Log prob de la acción
            log_prob = self.log_prob(mu, sigma, tf.convert_to_tensor([action], dtype=tf.float32))[0]
            
            # Dar paso en el entorno
            next_state, reward, done, _, _ = env.step(action)
            
            # Guardar experiencia
            actions.append(action)
            rewards.append(reward)
            values.append(value[0][0].numpy())
            dones.append(float(done))
            log_probs.append(log_prob.numpy())
            
            # Actualizar estado y recompensa acumulada
            state = next_state
            episode_reward += reward
            
            # Si el episodio termina, resetear y registrar métricas
            if done:
                state, _ = env.reset()
                episode_rewards.append(episode_reward)
                episode_lengths.append(episode_length)
                episode_reward = 0
                episode_length = 0
            
            # Obtener valor del siguiente estado
            next_value = self.model.get_value(next_state)
            next_values.append(next_value[0])
        
        # Si el último episodio no terminó, guardar sus métricas parciales
        if episode_length > 0:
            episode_rewards.append(episode_reward)
            episode_lengths.append(episode_length)
            
        # Empaquetar datos
        trajectory_data = [
            np.array(states, dtype=np.float32),
            np.array(actions, dtype=np.float32),
            np.array(rewards, dtype=np.float32),
            np.array(values, dtype=np.float32),
            np.array(dones, dtype=np.float32),
            np.array(next_values, dtype=np.float32),
            np.array(log_probs, dtype=np.float32)
        ]
        
        episode_history = {
            'reward': episode_rewards,
            'length': episode_lengths
        }
        
        return trajectory_data, episode_history
    
    def _update_policy(self, states: np.ndarray, actions: np.ndarray, old_log_probs: np.ndarray, 
                     returns: np.ndarray, advantages: np.ndarray, values: np.ndarray,
                     batch_size: int, update_iters: int) -> Dict[str, float]:
        """
        Actualiza la política y el crítico con los datos recolectados.
        
        Parámetros:
        -----------
        states : np.ndarray
            Estados observados
        actions : np.ndarray
            Acciones tomadas
        old_log_probs : np.ndarray
            Log probs originales de las acciones
        returns : np.ndarray
            Retornos calculados
        advantages : np.ndarray
            Ventajas estimadas
        values : np.ndarray
            Valores originales estimados
        batch_size : int
            Tamaño de lote para actualización
        update_iters : int
            Número de iteraciones de actualización
            
        Retorna:
        --------
        Dict[str, float]
            Métricas promedio de entrenamiento
        """
        # Convertir a tensores
        states_tensor = tf.convert_to_tensor(states, dtype=tf.float32)
        actions_tensor = tf.convert_to_tensor(actions, dtype=tf.float32)
        old_log_probs_tensor = tf.convert_to_tensor(old_log_probs, dtype=tf.float32)
        returns_tensor = tf.convert_to_tensor(returns, dtype=tf.float32)
        advantages_tensor = tf.convert_to_tensor(advantages, dtype=tf.float32)
        values_tensor = tf.convert_to_tensor(values, dtype=tf.float32)
        
        # Crear conjunto de datos para entrenar
        dataset = tf.data.Dataset.from_tensor_slices(
            (states_tensor, actions_tensor, old_log_probs_tensor, returns_tensor, advantages_tensor, values_tensor)
        )
        dataset = dataset.shuffle(buffer_size=states.shape[0]).batch(batch_size)
        
        # Resetear métricas
        self.policy_loss_metric.reset_state()
        self.value_loss_metric.reset_state()
        self.entropy_metric.reset_state()
        self.total_loss_metric.reset_state()
        
        # Entrenar durante varias iteraciones
        for _ in range(update_iters):
            for batch in dataset:
                self.train_step(*batch)
        
        # Recopilar métricas
        metrics = {
            'policy_loss': float(self.policy_loss_metric.result().numpy()),
            'value_loss': float(self.value_loss_metric.result().numpy()),
            'entropy': float(self.entropy_metric.result().numpy()),
            'total_loss': float(self.total_loss_metric.result().numpy())
        }
        
        return metrics
    
    def train(self, env: Any, epochs: int = CONST_DEFAULT_EPOCHS, steps_per_epoch: int = 4000, batch_size: int = 64, 
             update_iters: int = 10, gae_lambda: float = 0.95,
             log_interval: int = 10) -> Dict[str, List[float]]:
        """
        Entrena el agente con PPO en un entorno dado.
        
        Parámetros:
        -----------
        env : Any
            Entorno de OpenAI Gym o compatible
        epochs : int, opcional
            Número de épocas de entrenamiento (default: 100)
        steps_per_epoch : int, opcional
            Pasos por época (default: 4000)
        batch_size : int, opcional
            Tamaño de lote para actualización (default: 64)
        update_iters : int, opcional
            Número de iteraciones de actualización por lote (default: 10)
        gae_lambda : float, opcional
            Factor lambda para GAE (default: 0.95)
        log_interval : int, opcional
            Intervalo para mostrar información (default: 10)
            
        Retorna:
        --------
        Dict[str, List[float]]
            Historial de entrenamiento
        """
        history = {
            'reward': [],       # Recompensa por episodio
            'avg_reward': [],   # Recompensa promedio por época
            'policy_loss': [],  # Pérdida de política por época
            'value_loss': [],   # Pérdida de valor por época
            'entropy': [],      # Entropía por época
            'total_loss': []    # Pérdida total por época
        }
        
        for epoch in range(epochs):
            # 1. Recolectar trayectorias
            trajectory_data, episode_history = self._collect_trajectories(env, steps_per_epoch)
            states, actions, rewards, values, dones, next_values, log_probs = trajectory_data
            
            # Registrar recompensas de episodios
            history['reward'].extend(episode_history['reward'])
            history['avg_reward'].append(np.mean(episode_history['reward']))
            
            # 2. Calcular ventajas y retornos usando GAE
            advantages, returns = self.compute_gae(
                rewards, values, next_values, dones, self.gamma, gae_lambda)
            
            # 3. Actualizar política
            metrics = self._update_policy(
                states, actions, log_probs, returns, advantages, values,
                batch_size, update_iters)
            
            # Registrar métricas
            history['policy_loss'].append(metrics['policy_loss'])
            history['value_loss'].append(metrics['value_loss']) 
            history['entropy'].append(metrics['entropy'])
            history['total_loss'].append(metrics['total_loss'])
            
            # Mostrar progreso
            if (epoch + 1) % log_interval == 0 or epoch == 0:
                print(f"Época {epoch+1}/{epochs} - Recompensa Promedio: {history['avg_reward'][-1]:.2f}, "
                      f"Pérdida Política: {history['policy_loss'][-1]:.4f}, "
                      f"Pérdida Valor: {history['value_loss'][-1]:.4f}")
        
        return history
    
    def evaluate(self, env: Any, n_episodes: int = 10, deterministic: bool = True, render: bool = False) -> float:
        """
        Evalúa el agente entrenado en un entorno.
        
        Parámetros:
        -----------
        env : Any
            Entorno de OpenAI Gym o compatible
        n_episodes : int, opcional
            Número de episodios para evaluar (default: 10)
        deterministic : bool, opcional
            Si usar acciones determinísticas (default: True)
        render : bool, opcional
            Si renderizar el entorno (default: False)
            
        Retorna:
        --------
        float
            Recompensa promedio por episodio
        """
        total_rewards = []
        total_lengths = []
        
        for episode in range(n_episodes):
            state, _ = env.reset()
            done = False
            episode_reward = 0
            episode_length = 0
            
            while not done:
                # Obtener acción (determinística o estocástica)
                action = self.model.get_action(state, deterministic=deterministic)
                
                # Dar paso en el entorno
                next_state, reward, done, _, _ = env.step(action)
                
                # Renderizar si es necesario
                if render:
                    env.render()
                
                # Actualizar estado y contadores
                state = next_state
                episode_reward += reward
                episode_length += 1
                
            # Registrar resultados
            total_rewards.append(episode_reward)
            total_lengths.append(episode_length)
            print(f"Episodio {episode+1}/{n_episodes} - Recompensa: {episode_reward:.2f}, Longitud: {episode_length}")
        
        # Calcular estadísticas
        mean_reward = np.mean(total_rewards)
        mean_length = np.mean(total_lengths)
        std_reward = np.std(total_rewards)
        
        print(f"\nEvaluación completada - Recompensa Promedio: {mean_reward:.2f} ± {std_reward:.2f}, "
              f"Longitud Promedio: {mean_length:.1f}")
              
        return mean_reward
    
    def save_model(self, filepath: str) -> None:
        """
        Guarda el modelo en un archivo.
        
        Parámetros:
        -----------
        filepath : str
            Ruta donde guardar el modelo
        """
        self.model.save_weights(filepath)
        print(f"Modelo guardado en {filepath}")
        
    def load_model(self, filepath: str) -> None:
        """
        Carga el modelo desde un archivo.
        
        Parámetros:
        -----------
        filepath : str
            Ruta desde donde cargar el modelo
        """
        # Asegurarse de que el modelo está construido primero
        dummy_state = np.zeros((1, self.state_dim), dtype=np.float32)
        self.model(dummy_state)
        self.model.load_weights(filepath)
        print(f"Modelo cargado desde {filepath}")
    
    def visualize_training(self, history: Dict[str, List[float]], window_size: int = 5) -> None:
        """
        Visualiza los resultados del entrenamiento.
        
        Parámetros:
        -----------
        history : Dict[str, List[float]]
            Historial de entrenamiento
        window_size : int, opcional
            Tamaño de ventana para suavizado (default: 5)
        """
        # Constantes para etiquetas de ejes
        LABEL_EPOCA = 'Época'
        LABEL_EPISODIO = 'Episodio'
        
        def smooth(y, window_size):
            """Aplica suavizado con media móvil."""
            box = np.ones(window_size) / window_size
            return np.convolve(y, box, mode='valid')
        
        plt.figure(figsize=(15, 10))
        
        # Graficar recompensas por episodio
        plt.subplot(2, 2, 1)
        rewards = history['reward']
        plt.plot(rewards, alpha=0.3, color='blue')
        if len(rewards) > window_size:
            plt.plot(range(window_size-1, len(rewards)), 
                    smooth(rewards, window_size), 
                    color='blue', label='Suavizado')
        plt.title('Recompensa por Episodio')
        plt.xlabel(LABEL_EPISODIO)
        plt.ylabel('Recompensa')
        plt.grid(alpha=0.3)
        
        # Graficar recompensa promedio por época
        plt.subplot(2, 2, 2)
        avg_rewards = history['avg_reward']
        plt.plot(avg_rewards, marker='o', color='green')
        plt.title('Recompensa Promedio por Época')
        plt.xlabel(LABEL_EPOCA)
        plt.ylabel('Recompensa Promedio')
        plt.grid(alpha=0.3)
        
        # Graficar pérdidas
        plt.subplot(2, 2, 3)
        plt.plot(history['policy_loss'], label='Política', color='red')
        plt.plot(history['value_loss'], label='Valor', color='orange')
        plt.title('Pérdidas')
        plt.xlabel(LABEL_EPOCA)
        plt.ylabel('Pérdida')
        plt.legend()
        plt.grid(alpha=0.3)
        
        # Graficar entropía
        plt.subplot(2, 2, 4)
        plt.plot(history['entropy'], color='purple')
        plt.title('Entropía')
        plt.xlabel(LABEL_EPOCA)
        plt.ylabel('Entropía')
        plt.grid(alpha=0.3)
        
        plt.tight_layout()
        plt.show()

# Constantes para evitar duplicación
MODEL_WEIGHTS_SUFFIX = '_model.weights.h5'
ACTOR_CRITIC_WEIGHTS_SUFFIX = '_actor_critic.weights.h5'
CGM_ENCODER = 'cgm_encoder'
OTHER_ENCODER = 'other_encoder'
COMBINED_LAYER = 'combined_layer'

@register_keras_serializable()
class PPOModelWrapper(tf.keras.models.Model):
    """
    Wrapper para el algoritmo PPO que implementa la interfaz de Keras.Model.
    """
    
    def __init__(
        self, 
        ppo_agent: PPO,
        cgm_shape: Tuple[int, ...],
        other_features_shape: Tuple[int, ...]
    ) -> None:
        """
        Inicializa el modelo wrapper para PPO.
        
        Parámetros:
        -----------
        ppo_agent : PPO
            Agente PPO a utilizar
        cgm_shape : Tuple[int, ...]
            Forma de entrada para datos CGM
        other_features_shape : Tuple[int, ...]
            Forma de entrada para otras características
        """
        super(PPOModelWrapper, self).__init__()
        self.ppo_agent = ppo_agent
        self.cgm_shape = cgm_shape
        self.other_features_shape = other_features_shape
        
        # Capas para procesar entrada CGM
        self.cgm_conv = Conv1D(64, 3, padding='same', activation='relu', name=f'{CGM_ENCODER}_conv')
        self.cgm_pooling = GlobalAveragePooling1D(name=f'{CGM_ENCODER}_pooling')
        
        # Capas para procesar otras características
        self.other_dense = Dense(32, activation='relu', name=OTHER_ENCODER)
        
        # Capa para combinar características en representación de estado
        self.combined_dense = Dense(self.ppo_agent.state_dim, activation='relu', name=COMBINED_LAYER)
        
        # Capa para convertir salidas de política a dosis
        self.dose_predictor = Dense(1, kernel_initializer='glorot_uniform', name='dose_predictor')
    
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
        
        # Codificar entradas a estados
        states = self._encode_states(cgm_data, other_features)
        
        # Inicializar tensor para acciones
        actions = tf.TensorArray(tf.float32, size=batch_size)
        
        # Para cada muestra en el batch, obtener acción del agente PPO
        for i in range(batch_size):
            state = states[i]
            # Usar política del agente PPO (modo determinístico para predicción)
            action = self.ppo_agent.model.get_action(state.numpy(), deterministic=True)
            actions = actions.write(i, tf.convert_to_tensor(action, dtype=tf.float32))
        
        # Convertir a tensor
        actions_tensor = actions.stack()
        
        # Mapear acciones a dosis de insulina
        doses = self.dose_predictor(tf.reshape(actions_tensor, [batch_size, -1]))
        
        return doses
    
    def _encode_states(self, cgm_data: tf.Tensor, other_features: tf.Tensor) -> tf.Tensor:
        """
        Codifica las entradas en una representación de estado para el agente PPO.
        
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
        # Procesar CGM con capas convolucionales
        cgm_encoded = self.cgm_conv(cgm_data)
        cgm_features = self.cgm_pooling(cgm_encoded)
        
        # Procesar otras características
        other_encoded = self.other_dense(other_features)
        
        # Combinar características
        combined = tf.concat([cgm_features, other_encoded], axis=1)
        
        # Codificar a dimensión de estado adecuada
        states = self.combined_dense(combined)
        
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
        Simula la interfaz de entrenamiento de Keras para el agente PPO.
        
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
            print("Entrenando agente PPO...")
        
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
        
        # Crear entorno personalizado para RL a partir de los datos
        env = self._create_training_environment(cgm_data, other_features, y)
        
        # Entrenar el agente PPO
        history = self.ppo_agent.train(
            env=env,
            epochs=epochs,
            steps_per_epoch=batch_size,
            batch_size=min(32, batch_size),
            log_interval=max(1, epochs // 10) if verbose > 0 else epochs + 1
        )
        
        # Calibrar capa de predicción de dosis
        self._calibrate_dose_predictor(y)
        
        # Crear historia simulada para compatibilidad con Keras
        keras_history = {
            'loss': history.get('total_loss', [0.0]),
            'policy_loss': history.get('policy_loss', [0.0]),
            'value_loss': history.get('value_loss', [0.0]),
            'entropy': history.get('entropy', [0.0]),
            'val_loss': [history.get('total_loss', [0.0])[-1]] if validation_data is not None else None
        }
        
        # Crear un objeto que emula el comportamiento de History de Keras
        class KerasHistoryCompatible:
            def __init__(self, history_dict):
                self.history = history_dict
        
        return KerasHistoryCompatible(keras_history)
    
    def _create_training_environment(self, cgm_data: tf.Tensor, other_features: tf.Tensor, 
                                   target_doses: tf.Tensor) -> Any:
        """
        Crea un entorno personalizado compatible con el agente PPO.
        
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
            Entorno compatible con Open AI Gym
        """
        # Convertir tensores a numpy
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
                self.rng = np.random.Generator(np.random.PCG64(CONST_DEFAULT_SEED))
                
                # Para compatibilidad con algoritmos RL
                self.observation_space = gym.spaces.Box(
                    low=-np.inf, high=np.inf,
                    shape=(model_wrapper.ppo_agent.state_dim,)
                )
                self.action_space = gym.spaces.Box(
                    low=-1.0, high=1.0,
                    shape=(model_wrapper.ppo_agent.model.mu.units,)
                )
            
            def reset(self):
                """Reinicia el entorno seleccionando un ejemplo aleatorio."""
                self.current_idx = self.rng.integers(0, self.max_idx)
                state = self._get_state()
                return state, {}
            
            def step(self, action):
                """Ejecuta un paso en el entorno con la acción dada."""
                # Mapear acción (típicamente en [-1, 1]) a dosis de insulina (0-15 unidades)
                dose = (action[0] + 1) * 7.5  # Escalar de [-1,1] a [0,15]
                
                # Calcular recompensa (negativo del error absoluto)
                target = self.targets[self.current_idx]
                reward = -abs(dose - target)
                
                # Avanzar al siguiente ejemplo
                self.current_idx = (self.current_idx + 1) % self.max_idx
                
                # Obtener próximo estado
                next_state = self._get_state()
                
                # Episodio siempre termina después de un paso (para este problema)
                done = True
                truncated = False
                
                return next_state, reward, done, truncated, {}
            
            def _get_state(self):
                """Obtiene el estado codificado para el ejemplo actual."""
                # Obtener datos actuales
                cgm_batch = self.cgm[self.current_idx:self.current_idx+1]
                features_batch = self.features[self.current_idx:self.current_idx+1]
                
                # Codificar a espacio de estado usando el wrapper
                state = self.model._encode_states(
                    tf.convert_to_tensor(cgm_batch, dtype=tf.float32),
                    tf.convert_to_tensor(features_batch, dtype=tf.float32)
                )
                
                return state[0].numpy()
        
        return InsulinDosingEnv(cgm_np, other_np, target_np, self)
    
    def _calibrate_dose_predictor(self, y: tf.Tensor) -> None:
        """
        Calibra la capa que mapea acciones a dosis de insulina.
        
        Parámetros:
        -----------
        y : tf.Tensor
            Dosis objetivo para calibración
        """
        y_np = y.numpy() if hasattr(y, 'numpy') else y
        max_dose = np.max(y_np)
        min_dose = np.min(y_np)
        
        # Asegurar que la capa esté construida antes de establecer pesos
        # Crear entrada dummy con la forma correcta y llamar a la capa
        dummy_input = tf.zeros((1, self.ppo_agent.action_dim))
        self.dose_predictor(dummy_input)  # Esto construirá la capa
        
        # Configurar pesos para convertir acciones a dosis apropiadas
        # Suponemos acciones en [-1, 1] que se escalan a [min_dose, max_dose]
        scale = (max_dose - min_dose) / 2.0
        bias = (min_dose + max_dose) / 2.0
        
        self.dose_predictor.set_weights([
            np.ones((self.ppo_agent.action_dim, 1)) * scale,
            np.array([bias])
        ])
    
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
            "state_dim": self.ppo_agent.state_dim,
            "action_dim": self.ppo_agent.model.mu.units,
            "hidden_units": self.ppo_agent.hidden_units,
            "gamma": self.ppo_agent.gamma,
            "epsilon": self.ppo_agent.epsilon
        }
    
    def save(self, filepath: str, **kwargs) -> None:
        """
        Guarda el modelo PPO.
        
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
            base_path = filepath[:-6]  # Quitar '.keras'
        
        # Guardar pesos del wrapper
        self.save_weights(base_path + MODEL_WEIGHTS_SUFFIX)
        
        # Guardar modelo actor-crítico del agente PPO
        self.ppo_agent.save_model(base_path + ACTOR_CRITIC_WEIGHTS_SUFFIX)
    
    def load_weights(self, filepath: str, **kwargs) -> None:
        """
        Carga el modelo PPO.
        
        Parámetros:
        -----------
        filepath : str
            Ruta desde donde cargar el modelo
        **kwargs
            Argumentos adicionales
        """
        # Determinar rutas según formato de filepath
        if filepath.endswith(MODEL_WEIGHTS_SUFFIX):
            wrapper_path = filepath
            ac_path = filepath.replace(MODEL_WEIGHTS_SUFFIX, ACTOR_CRITIC_WEIGHTS_SUFFIX)
        else:
            wrapper_path = filepath + MODEL_WEIGHTS_SUFFIX
            ac_path = filepath + ACTOR_CRITIC_WEIGHTS_SUFFIX
            
        # Cargar pesos del wrapper
        super().load_weights(wrapper_path)
        
        # Cargar modelo actor-crítico
        self.ppo_agent.load_model(ac_path)


def create_ppo_model(cgm_shape: Tuple[int, ...], other_features_shape: Tuple[int, ...]) -> tf.keras.models.Model:
    """
    Crea un modelo basado en PPO (Proximal Policy Optimization) para predicción de dosis de insulina.
    
    Parámetros:
    -----------
    cgm_shape : Tuple[int, ...]
        Forma de los datos CGM (batch_size, time_steps, features)
    other_features_shape : Tuple[int, ...]
        Forma de otras características (batch_size, n_features)
        
    Retorna:
    --------
    tf.keras.models.Model
        Modelo PPO que implementa la interfaz de Keras
    """
    # Determinar dimensión del espacio de estado
    state_dim = 64  # Dimensión del espacio de estado codificado
    
    # Configurar espacio de acción (dosis de insulina)
    action_dim = 1  # Una dimensión para la dosis continua
    
    # Crear agente PPO con configuración desde PPO_CONFIG
    ppo_agent = PPO(
        state_dim=state_dim,
        action_dim=action_dim,
        learning_rate=PPO_CONFIG['learning_rate'],
        gamma=PPO_CONFIG['gamma'],
        epsilon=PPO_CONFIG['clip_epsilon'],
        hidden_units=PPO_CONFIG['hidden_units'],
        entropy_coef=PPO_CONFIG['entropy_coef'],
        value_coef=PPO_CONFIG['value_coef'],
        max_grad_norm=PPO_CONFIG['max_grad_norm'],
        seed=CONST_DEFAULT_SEED
    )
    
    ppo_model = PPOModelWrapper(
        ppo_agent=ppo_agent,
        cgm_shape=cgm_shape,
        other_features_shape=other_features_shape
    )
    
    # Pre-compilar el modelo
    ppo_model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=PPO_CONFIG['learning_rate']),
        loss='mse'  # Loss dummy, no se usa realmente para el entrenamiento RL
    )
    
    return ppo_model