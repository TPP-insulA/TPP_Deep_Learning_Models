import os, sys
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import time
from typing import Dict, List, Tuple, Optional, Union, Any, Callable
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input, Dense, LayerNormalization, Dropout, Activation, GlobalAveragePooling1D
)
from tensorflow.keras.optimizers import Adam
from keras.saving import register_keras_serializable

PROJECT_ROOT = os.path.abspath(os.getcwd())
sys.path.append(PROJECT_ROOT) 

from config.models_config import REINFORCE_CONFIG


class PolicyNetwork(Model):
    """
    Red neuronal de política para REINFORCE.
    
    Esta red mapea estados a distribuciones de probabilidad sobre acciones.
    """
    def __init__(
        self, 
        state_dim: int, 
        action_dim: int, 
        hidden_units: Optional[List[int]] = None, 
        continuous: bool = False
    ) -> None:
        """
        Inicializa la red de política.
        
        Parámetros:
        -----------
        state_dim : int
            Dimensión del espacio de estados
        action_dim : int
            Dimensión del espacio de acciones
        hidden_units : Optional[List[int]], opcional
            Lista con unidades en capas ocultas (default: None)
        continuous : bool, opcional
            Si el espacio de acciones es continuo (default: False)
        """
        super(PolicyNetwork, self).__init__()
        
        # Configuración de la red
        if hidden_units is None:
            hidden_units = REINFORCE_CONFIG['hidden_units']
        
        self.continuous = continuous
        self.action_dim = action_dim
        
        # Capas ocultas
        self.hidden_layers = []
        for i, units in enumerate(hidden_units):
            self.hidden_layers.append(Dense(units, name=f'hidden_{i}'))
            self.hidden_layers.append(LayerNormalization(epsilon=REINFORCE_CONFIG['epsilon'], name=f'ln_{i}'))
            self.hidden_layers.append(Activation('relu', name=f'relu_{i}'))
            if REINFORCE_CONFIG['dropout_rate'] > 0:
                self.hidden_layers.append(Dropout(REINFORCE_CONFIG['dropout_rate'], name=f'dropout_{i}'))
        
        # Capa de salida: depende de si el espacio de acciones es continuo o discreto
        if continuous:
            # Para espacios continuos: política gaussiana
            self.mu = Dense(action_dim, activation='tanh', name='mu')
            self.log_sigma = Dense(action_dim, activation='linear', name='log_sigma')
        else:
            # Para espacios discretos: política categórica
            self.logits = Dense(action_dim, activation='linear', name='logits')
    
    def call(self, inputs: tf.Tensor, training: bool = False) -> Union[tf.Tensor, Tuple[tf.Tensor, tf.Tensor]]:
        """
        Ejecuta la red para obtener los parámetros de la distribución de política.
        
        Parámetros:
        -----------
        inputs : tf.Tensor
            Tensor de estados de entrada
        training : bool, opcional
            Si está en modo entrenamiento (default: False)
        
        Retorna:
        --------
        Union[tf.Tensor, Tuple[tf.Tensor, tf.Tensor]]
            Para espacios discretos: logits
            Para espacios continuos: (mu, log_sigma)
        """
        x = inputs
        
        # Pasar por capas ocultas
        for layer in self.hidden_layers:
            if isinstance(layer, Dropout):
                x = layer(x, training=training)
            else:
                x = layer(x)
        
        # Generar salida según el tipo de política
        if self.continuous:
            mu = self.mu(x)
            log_sigma = self.log_sigma(x)
            # Limitar el rango de log_sigma para estabilidad
            log_sigma = tf.clip_by_value(log_sigma, -20, 2)
            return mu, log_sigma
        else:
            logits = self.logits(x)
            return logits
    
    def get_action(self, state: np.ndarray, deterministic: bool = False) -> Union[np.ndarray, int]:
        """
        Obtiene una acción según la política actual.
        
        Parámetros:
        -----------
        state : np.ndarray
            Estado actual
        deterministic : bool, opcional
            Si se usa la acción determinística o se muestrea (default: False)

        Retorna:
        --------
        Union[np.ndarray, int]
            La acción seleccionada
        """
        # Convertir a tensor y obtener distribución
        state_tensor = tf.convert_to_tensor([state], dtype=tf.float32)
        
        if self.continuous:
            mu, log_sigma = self(state_tensor, training=False)
            
            if deterministic:
                return mu[0].numpy()
            
            # Muestrear de la distribución gaussiana para exploración
            sigma = tf.exp(log_sigma)
            normal_dist = tf.random.normal(shape=mu.shape)
            action = mu + sigma * normal_dist
            return action[0].numpy()
        else:
            logits = self(state_tensor, training=False)
            
            if deterministic:
                return tf.argmax(logits[0]).numpy()
            
            # Muestrear de la distribución categórica para exploración
            probs = tf.nn.softmax(logits)
            action = tf.random.categorical(tf.math.log(probs), 1)
            return action[0, 0].numpy()
    
    def get_log_prob(self, states: tf.Tensor, actions: tf.Tensor) -> tf.Tensor:
        """
        Calcula el logaritmo de la probabilidad de acciones dadas.
        
        Parámetros:
        -----------
        states : tf.Tensor
            Estados observados
        actions : tf.Tensor
            Acciones tomadas
            
        Retorna:
        --------
        tf.Tensor
            Log-probabilidades de las acciones
        """
        if self.continuous:
            mu, log_sigma = self(states)
            sigma = tf.exp(log_sigma)
            
            # Log-prob para distribución gaussiana
            log_probs = -0.5 * (
                tf.reduce_sum(
                    tf.square((actions - mu) / sigma) + 
                    2 * log_sigma + 
                    tf.math.log(2.0 * np.pi),
                    axis=1
                )
            )
            return log_probs
        else:
            logits = self(states)
            
            # Log-prob para distribución categórica
            action_masks = tf.one_hot(tf.cast(actions, tf.int32), self.action_dim)
            log_probs = tf.reduce_sum(
                action_masks * tf.nn.log_softmax(logits),
                axis=1
            )
            return log_probs
    
    def get_entropy(self, states: tf.Tensor) -> tf.Tensor:
        """
        Calcula la entropía de la política para los estados dados.
        
        Parámetros:
        -----------
        states : tf.Tensor
            Estados para evaluar
            
        Retorna:
        --------
        tf.Tensor
            Entropía de la política
        """
        if self.continuous:
            _, log_sigma = self(states)
            # Entropía de distribución gaussiana: 0.5 * log(2*pi*e*sigma^2)
            entropy = tf.reduce_sum(
                0.5 * tf.math.log(2.0 * np.pi * np.e * tf.exp(2 * log_sigma)),
                axis=1
            )
        else:
            logits = self(states)
            probs = tf.nn.softmax(logits)
            # Entropía de distribución categórica: -sum(p * log(p))
            entropy = -tf.reduce_sum(
                probs * tf.math.log(probs + 1e-10),
                axis=1
            )
        return entropy


class REINFORCE:
    """
    Implementación del algoritmo REINFORCE (Monte Carlo Policy Gradient).
    
    REINFORCE utiliza retornos Monte Carlo completos para actualizar la política,
    haciéndolo un algoritmo simple pero efectivo para aprendizaje de políticas.
    """
    
    def __init__(
        self, 
        state_dim: int, 
        action_dim: int,
        continuous: bool = False,
        learning_rate: float = REINFORCE_CONFIG['learning_rate'],
        gamma: float = REINFORCE_CONFIG['gamma'],
        hidden_units: Optional[List[int]] = None,
        baseline: bool = REINFORCE_CONFIG['use_baseline'],
        entropy_coef: float = REINFORCE_CONFIG['entropy_coef'],
        seed: int = 42
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
        learning_rate : float, opcional
            Tasa de aprendizaje (default: REINFORCE_CONFIG['learning_rate'])
        gamma : float, opcional
            Factor de descuento (default: REINFORCE_CONFIG['gamma'])
        hidden_units : Optional[List[int]], opcional
            Lista con unidades en capas ocultas (default: None)
        baseline : bool, opcional
            Si usar baseline para reducir varianza (default: REINFORCE_CONFIG['use_baseline'])
        entropy_coef : float, opcional
            Coeficiente para regularización por entropía (default: REINFORCE_CONFIG['entropy_coef'])
        seed : int, opcional
            Semilla para reproducibilidad (default: 42)
        """
        # Configurar semillas para reproducibilidad
        tf.random.set_seed(seed)
        self.rng = np.random.Generator(np.random.PCG64(seed))
        
        # Parámetros básicos
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.continuous = continuous
        self.gamma = gamma
        self.use_baseline = baseline
        self.entropy_coef = entropy_coef
        
        # Valores predeterminados para capas ocultas
        if hidden_units is None:
            self.hidden_units = REINFORCE_CONFIG['hidden_units']
        else:
            self.hidden_units = hidden_units
        
        # Crear red de política
        self.policy = PolicyNetwork(state_dim, action_dim, self.hidden_units, continuous)
        self.optimizer = Adam(learning_rate=learning_rate)
        
        # Red de valor (baseline) opcional para reducir varianza
        if self.use_baseline:
            self.value_network = self._create_value_network()
            self.value_optimizer = Adam(learning_rate=learning_rate)
        
        # Métricas
        self.policy_loss_metric = tf.keras.metrics.Mean('policy_loss')
        self.entropy_metric = tf.keras.metrics.Mean('entropy')
        self.baseline_loss_metric = tf.keras.metrics.Mean('baseline_loss')
        self.returns_metric = tf.keras.metrics.Mean('returns')
    
    def _create_value_network(self) -> Model:
        """
        Crea una red neuronal para estimar el valor de estado (baseline).
        
        Retorna:
        --------
        Model
            Modelo de red de valor
        """
        inputs = Input(shape=(self.state_dim,))
        x = inputs
        
        # Capas ocultas
        for i, units in enumerate(self.hidden_units):
            x = Dense(units, activation='relu', name=f'value_hidden_{i}')(x)
            x = LayerNormalization(epsilon=REINFORCE_CONFIG['epsilon'], name=f'value_ln_{i}')(x)
            if REINFORCE_CONFIG['dropout_rate'] > 0:
                x = Dropout(REINFORCE_CONFIG['dropout_rate'], name=f'value_dropout_{i}')(x)
        
        # Capa de salida: un solo valor
        outputs = Dense(1, activation='linear', name='value')(x)
        
        model = Model(inputs=inputs, outputs=outputs)
        return model
    
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
            # Calcular log-probabilidades de acciones tomadas
            log_probs = self.policy.get_log_prob(states, actions)
            
            # Si se usa baseline, restar el valor predicho de los retornos
            if self.use_baseline:
                values = tf.squeeze(self.value_network(states))
                advantages = returns - values
            else:
                advantages = returns
            
            # Calcular pérdida de política (negativa porque queremos maximizar)
            policy_loss = -tf.reduce_mean(log_probs * advantages, axis=0)
            
            # Calcular entropía y añadir regularización
            entropy = tf.reduce_mean(self.policy.get_entropy(states), axis=0)
            loss = policy_loss - self.entropy_coef * entropy
        
        # Calcular gradientes y actualizar política
        grads = tape.gradient(loss, self.policy.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.policy.trainable_variables))
        
        # Actualizar métricas
        self.policy_loss_metric.update_state(policy_loss)
        self.entropy_metric.update_state(entropy)
        self.returns_metric.update_state(tf.reduce_mean(returns, axis=0))
        
        return policy_loss, entropy
    
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
            Pérdida de la red de valor
        """
        with tf.GradientTape() as tape:
            values = tf.squeeze(self.value_network(states))
            value_loss = tf.reduce_mean(tf.square(returns - values), axis=0)
        
        # Calcular gradientes y actualizar red de valor
        grads = tape.gradient(value_loss, self.value_network.trainable_variables)
        self.value_optimizer.apply_gradients(zip(grads, self.value_network.trainable_variables))
        
        # Actualizar métrica
        self.baseline_loss_metric.update_state(value_loss)
        
        return value_loss
    
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
            
            # Seleccionar acción según política actual
            action = self.policy.get_action(state)
            
            # Dar paso en el entorno
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            
            # Guardar transición
            states.append(state)
            actions.append(action)
            rewards.append(reward)
            
            # Avanzar al siguiente estado
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
        
        return policy_loss.numpy(), entropy.numpy()
    
    def _update_history(
        self, 
        history: Dict[str, List[float]], 
        episode_reward: float, 
        episode_length: int
    ) -> None:
        """
        Actualiza la historia de entrenamiento con las métricas actuales.
        
        Parámetros:
        -----------
        history : Dict[str, List[float]]
            Diccionario de historia
        episode_reward : float
            Recompensa total del episodio
        episode_length : int
            Longitud del episodio
        """
        history['episode_rewards'].append(episode_reward)
        history['episode_lengths'].append(episode_length)
        history['policy_losses'].append(self.policy_loss_metric.result().numpy())
        if self.use_baseline:
            history['baseline_losses'].append(self.baseline_loss_metric.result().numpy())
        history['entropies'].append(self.entropy_metric.result().numpy())
        history['mean_returns'].append(self.returns_metric.result().numpy())
        
        # Resetear métricas
        self.policy_loss_metric.reset_states()
        self.entropy_metric.reset_states()
        self.returns_metric.reset_states()
        if self.use_baseline:
            self.baseline_loss_metric.reset_states()
    
    def train(
        self, 
        env: Any, 
        episodes: Optional[int] = None, 
        render: bool = False
    ) -> Dict[str, List[float]]:
        """
        Entrena el agente REINFORCE en el entorno dado.
        
        Parámetros:
        -----------
        env : Any
            Entorno de OpenAI Gym o compatible
        episodes : Optional[int], opcional
            Número de episodios de entrenamiento (default: None)
        render : bool, opcional
            Si renderizar el entorno durante entrenamiento (default: False)
            
        Retorna:
        --------
        Dict[str, List[float]]
            Historia de entrenamiento
        """
        if episodes is None:
            episodes = REINFORCE_CONFIG['episodes']
        
        history = {
            'episode_rewards': [],
            'episode_lengths': [],
            'policy_losses': [],
            'baseline_losses': [] if self.use_baseline else None,
            'entropies': [],
            'mean_returns': []
        }
        
        start_time = time.time()
        
        for episode in range(episodes):
            # Ejecutar episodio
            states, actions, rewards, episode_reward, episode_length = self._run_episode(env, render)
            
            # Actualizar redes
            policy_loss, entropy = self._update_networks(states, actions, rewards)
            
            # Actualizar historia
            self._update_history(history, episode_reward, episode_length)
            
            # Mostrar progreso periódicamente
            if (episode + 1) % REINFORCE_CONFIG['log_interval'] == 0:
                elapsed_time = time.time() - start_time
                avg_reward = np.mean(history['episode_rewards'][-REINFORCE_CONFIG['log_interval']:])
                print(f"Episodio {episode+1}/{episodes} - "
                      f"Recompensa Media: {avg_reward:.2f}, "
                      f"Pérdida Política: {policy_loss:.4f}, "
                      f"Entropía: {entropy:.4f}, "
                      f"Tiempo: {elapsed_time:.2f}s")
                start_time = time.time()
        
        return history
    
    def evaluate(
        self, 
        env: Any, 
        episodes: int = 10, 
        render: bool = False
    ) -> float:
        """
        Evalúa el agente REINFORCE con su política actual.
        
        Parámetros:
        -----------
        env : Any
            Entorno de OpenAI Gym o compatible
        episodes : int, opcional
            Número de episodios para evaluación (default: 10)
        render : bool, opcional
            Si renderizar el entorno durante evaluación (default: False)
            
        Retorna:
        --------
        float
            Recompensa media obtenida
        """
        rewards = []
        lengths = []
        
        for episode in range(episodes):
            state, _ = env.reset()
            done = False
            episode_reward = 0
            steps = 0
            
            while not done:
                if render:
                    env.render()
                
                # Usar política determinística para evaluación
                action = self.policy.get_action(state, deterministic=True)
                
                # Dar paso en el entorno
                next_state, reward, terminated, truncated, _ = env.step(action)
                done = terminated or truncated
                
                # Actualizar estado y contadores
                state = next_state
                episode_reward += reward
                steps += 1
            
            rewards.append(episode_reward)
            lengths.append(steps)
            
            print(f"Episodio Evaluación {episode+1}: Recompensa = {episode_reward:.2f}, Pasos = {steps}")
        
        avg_reward = np.mean(rewards)
        avg_length = np.mean(lengths)
        
        print(f"Evaluación Completada - Recompensa Media: {avg_reward:.2f}, Pasos Medios: {avg_length:.2f}")
        
        return avg_reward
    
    def save(self, policy_path: str, baseline_path: Optional[str] = None) -> None:
        """
        Guarda los modelos del agente.
        
        Parámetros:
        -----------
        policy_path : str
            Ruta para guardar la política
        baseline_path : Optional[str], opcional
            Ruta para guardar el baseline (default: None)
        """
        # Guardar política
        self.policy.save_weights(policy_path)
        
        # Guardar baseline si existe
        if self.use_baseline and baseline_path:
            self.value_network.save_weights(baseline_path)
        
        print(f"Modelo guardado en {policy_path}")
    
    def load(self, policy_path: str, baseline_path: Optional[str] = None) -> None:
        """
        Carga los modelos del agente.
        
        Parámetros:
        -----------
        policy_path : str
            Ruta para cargar la política
        baseline_path : Optional[str], opcional
            Ruta para cargar el baseline (default: None)
        """
        # Asegurarse que el modelo está construido antes de cargar
        dummy_state = np.zeros((1, self.state_dim))
        _ = self.policy(dummy_state)
        
        # Cargar política
        self.policy.load_weights(policy_path)
        
        # Cargar baseline si existe
        if self.use_baseline and baseline_path:
            _ = self.value_network(dummy_state)
            self.value_network.load_weights(baseline_path)
        
        print(f"Modelo cargado desde {policy_path}")
    
    def visualize_training(
        self, 
        history: Optional[Dict[str, List[float]]] = None, 
        smoothing_window: Optional[int] = None
    ) -> None:
        """
        Visualiza las métricas de entrenamiento.
        
        Parámetros:
        -----------
        history : Optional[Dict[str, List[float]]], opcional
            Historia de entrenamiento (default: None)
        smoothing_window : Optional[int], opcional
            Tamaño de ventana para suavizado (default: None)
        """
        if history is None:
            return
        
        if smoothing_window is None:
            smoothing_window = REINFORCE_CONFIG['smoothing_window']
        
        # Función para suavizar datos
        def smooth(data: List[float], window_size: int) -> np.ndarray:
            """Aplica suavizado con media móvil"""
            if len(data) < window_size:
                return data
            kernel = np.ones(window_size) / window_size
            return np.convolve(data, kernel, mode='valid')
        
        # Determinar número de subplots
        n_plots = 4 if self.use_baseline else 3
        _, axs = plt.subplots(n_plots, 1, figsize=(10, 3*n_plots))
        
        # 1. Gráfico de recompensas
        axs[0].plot(history['episode_rewards'], alpha=0.3, color='blue', label='Original')
        if len(history['episode_rewards']) > smoothing_window:
            smoothed_rewards = smooth(history['episode_rewards'], smoothing_window)
            axs[0].plot(range(smoothing_window-1, len(history['episode_rewards'])), 
                      smoothed_rewards, color='blue', 
                      label=f'Suavizado (ventana={smoothing_window})')
        axs[0].set_title('Recompensa por Episodio')
        axs[0].set_xlabel('Episodio')
        axs[0].set_ylabel('Recompensa')
        axs[0].grid(alpha=0.3)
        axs[0].legend()
        
        # 2. Gráfico de longitud de episodios
        axs[1].plot(history['episode_lengths'], alpha=0.3, color='green', label='Original')
        if len(history['episode_lengths']) > smoothing_window:
            smoothed_lengths = smooth(history['episode_lengths'], smoothing_window)
            axs[1].plot(range(smoothing_window-1, len(history['episode_lengths'])), 
                      smoothed_lengths, color='green', 
                      label=f'Suavizado (ventana={smoothing_window})')
        axs[1].set_title('Longitud de Episodios')
        axs[1].set_xlabel('Episodio')
        axs[1].set_ylabel('Pasos')
        axs[1].grid(alpha=0.3)
        axs[1].legend()
        
        # 3. Gráfico de pérdida de política
        axs[2].plot(history['policy_losses'], alpha=0.3, color='red', label='Original')
        if len(history['policy_losses']) > smoothing_window:
            smoothed_losses = smooth(history['policy_losses'], smoothing_window)
            axs[2].plot(range(smoothing_window-1, len(history['policy_losses'])), 
                      smoothed_losses, color='red', 
                      label=f'Suavizado (ventana={smoothing_window})')
        axs[2].set_title('Pérdida de Política')
        axs[2].set_xlabel('Episodio')
        axs[2].set_ylabel('Pérdida')
        axs[2].grid(alpha=0.3)
        axs[2].legend()
        
        # 4. Gráfico de pérdida de baseline (si se usa)
        if self.use_baseline:
            axs[3].plot(history['baseline_losses'], alpha=0.3, color='purple', label='Original')
            if len(history['baseline_losses']) > smoothing_window:
                smoothed_baseline = smooth(history['baseline_losses'], smoothing_window)
                axs[3].plot(range(smoothing_window-1, len(history['baseline_losses'])), 
                          smoothed_baseline, color='purple', 
                          label=f'Suavizado (ventana={smoothing_window})')
            axs[3].set_title('Pérdida de Baseline')
            axs[3].set_xlabel('Episodio')
            axs[3].set_ylabel('Pérdida')
            axs[3].grid(alpha=0.3)
            axs[3].legend()
        
        plt.tight_layout()
        plt.show()

@register_keras_serializable
class REINFORCEModel(Model):
    """
    Wrapper para el algoritmo REINFORCE que implementa la interfaz de Keras.Model.
    """
    
    def __init__(
        self, 
        reinforce_agent: REINFORCE,
        cgm_shape: Tuple[int, ...],
        other_features_shape: Tuple[int, ...]
    ) -> None:
        """
        Inicializa el modelo wrapper para REINFORCE.
        
        Parámetros:
        -----------
        reinforce_agent : REINFORCE
            Agente REINFORCE a utilizar
        cgm_shape : Tuple[int, ...]
            Forma de entrada para datos CGM
        other_features_shape : Tuple[int, ...]
            Forma de entrada para otras características
        """
        super(REINFORCEModel, self).__init__()
        self.reinforce_agent = reinforce_agent
        self.cgm_shape = cgm_shape
        self.other_features_shape = other_features_shape
        
        # Capas para procesamiento de entradas
        self.cgm_encoder = Dense(64, activation='relu', name='cgm_encoder')
        self.cgm_pooling = GlobalAveragePooling1D(name='cgm_pooling')
        self.other_encoder = Dense(32, activation='relu', name='other_encoder')
        self.combined_encoder = Dense(reinforce_agent.state_dim, activation='linear', name='state_encoder')
        
        # Capa para convertir acciones de política a dosis de insulina
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
        cgm_data, other_features = inputs
        batch_size = tf.shape(cgm_data)[0]
        
        # Codificar estado a partir de entradas
        states = self._encode_states(cgm_data, other_features)
        
        # Inicializar tensor para acciones
        actions = tf.TensorArray(tf.float32, size=batch_size)
        
        # Para cada muestra en el batch, obtener acción determinística (sin exploración)
        for i in range(batch_size):
            state = states[i]
            # Usar el agente para seleccionar acción sin exploración
            action = self.reinforce_agent.policy.get_action(state.numpy(), deterministic=True)
            actions = actions.write(i, tf.convert_to_tensor(action, dtype=tf.float32))
        
        # Obtener tensor de acciones y ajustar forma
        actions_tensor = actions.stack()
        
        # Para espacio continuo, usar directamente las acciones
        if self.reinforce_agent.continuous:
            # Escalar a rango de dosis apropiado
            scaled_actions = self.action_decoder(tf.reshape(actions_tensor, [batch_size, -1]))
        else:
            # Para espacio discreto, decodificar valores de acción a dosis
            action_one_hot = tf.one_hot(tf.cast(actions_tensor, tf.int32), self.reinforce_agent.action_dim)
            scaled_actions = self.action_decoder(action_one_hot)
        
        return scaled_actions
    
    def _encode_states(self, cgm_data: tf.Tensor, other_features: tf.Tensor) -> tf.Tensor:
        """
        Codifica las entradas CGM y otras características en representaciones de estado.
        
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
        cgm_encoded = self.cgm_encoder(cgm_data)
        cgm_features = self.cgm_pooling(cgm_encoded)
        
        # Procesar otras características
        other_encoded = self.other_encoder(other_features)
        
        # Combinar características
        combined = tf.concat([cgm_features, other_encoded], axis=1)
        
        # Codificar a dimensión de estado esperada por REINFORCE
        states = self.combined_encoder(combined)
        
        return states
    
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
    ) -> Dict[str, Any]:
        """
        Simula la interfaz de entrenamiento de Keras para el agente REINFORCE.
        
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
        Dict[str, Any]
            Historia simulada de entrenamiento
        """
        if verbose > 0:
            print("Entrenando agente REINFORCE...")
        
        # Crear entorno para entrenar el agente REINFORCE
        env = self._create_environment(x, y)
        
        # Entrenar agente REINFORCE
        history = self.reinforce_agent.train(
            env=env,
            episodes=batch_size * epochs,
            render=False
        )
        
        # Calibrar la capa de acción para mapeo apropiado
        self._calibrate_action_decoder(y)
        
        if verbose > 0:
            print(f"Entrenamiento completado en {len(history['episode_rewards'])} episodios.")
        
        # Convertir métricas a formato compatible con Keras
        keras_history = {
            'loss': history['policy_losses'],
            'val_loss': [np.mean(history['policy_losses'][-10:])] if validation_data else None
        }
        
        # Retornar directamente un diccionario
        return {'history': keras_history}
    
    def _create_environment(self, x: List[tf.Tensor], y: tf.Tensor) -> Any:
        """
        Crea un entorno personalizado para entrenar el agente REINFORCE.
        
        Parámetros:
        -----------
        x : List[tf.Tensor]
            Lista con [cgm_data, other_features]
        y : tf.Tensor
            Etiquetas (dosis objetivo)
            
        Retorna:
        --------
        Any
            Entorno para entrenamiento de RL
        """
        # Convertir a numpy para procesamiento
        cgm_data = x[0].numpy() if hasattr(x[0], 'numpy') else x[0]
        other_features = x[1].numpy() if hasattr(x[1], 'numpy') else x[1]
        targets = y.numpy() if hasattr(y, 'numpy') else y
        
        # Clase de entorno personalizado
        class InsulinDosingEnvironment:
            def __init__(self, cgm, features, targets, model):
                self.cgm = cgm
                self.features = features
                self.targets = targets
                self.model = model
                self.current_idx = 0
                self.max_idx = len(targets) - 1
                self.rng = np.random.Generator(np.random.PCG64(REINFORCE_CONFIG.get('seed', 42)))
                
            def reset(self):
                """Reinicia el entorno con un ejemplo aleatorio."""
                self.current_idx = self.rng.integers(0, self.max_idx)
                state = self._get_state()
                return state, {}
                
            def step(self, action):
                """Ejecuta un paso en el entorno con la acción dada."""
                # Convertir acción a dosis según el tipo de espacio de acción
                if self.model.reinforce_agent.continuous:
                    # Escalar la acción continua al rango de dosis
                    dose = action[0]  # Primera dimensión si es vector multidimensional
                else:
                    # Mapear el índice discreto a valor de dosis
                    max_dose = np.max(self.targets)
                    dose = (action / (self.model.reinforce_agent.action_dim - 1)) * max_dose
                
                # Calcular recompensa (negativo del error absoluto)
                target = self.targets[self.current_idx]
                reward = -abs(dose - target)
                
                # Avanzar al siguiente caso
                self.current_idx = (self.current_idx + 1) % self.max_idx
                
                # Obtener próximo estado
                next_state = self._get_state()
                
                # Episodio siempre termina después de un paso
                done = True
                
                return next_state, reward, done, False, {}
                
            def _get_state(self):
                """Obtiene el estado actual codificado."""
                # Obtener datos actuales
                cgm_batch = self.cgm[self.current_idx:self.current_idx+1]
                features_batch = self.features[self.current_idx:self.current_idx+1]
                
                # Codificar estado usando el modelo
                state = self.model._encode_states(
                    tf.convert_to_tensor(cgm_batch, dtype=tf.float32),
                    tf.convert_to_tensor(features_batch, dtype=tf.float32)
                )
                
                return state[0].numpy()  # Extraer y convertir a numpy
        
        return InsulinDosingEnvironment(cgm_data, other_features, targets, self)
    
    def _calibrate_action_decoder(self, y: tf.Tensor) -> None:
        """
        Calibra la capa que mapea acciones a dosis de insulina.
        
        Parámetros:
        -----------
        y : tf.Tensor
            Dosis objetivo para calibración
        """
        y_numpy = y.numpy() if hasattr(y, 'numpy') else y
        max_dose = np.max(y_numpy)
        min_dose = np.min(y_numpy)
        
        if self.reinforce_agent.continuous:
            # Para política continua, escalar a rango adecuado
            self.action_decoder.set_weights([
                np.ones((1, 1)) * (max_dose - min_dose),
                np.array([min_dose])
            ])
        else:
            # Para política discreta, establecer pesos para mapear adecuadamente
            self.action_decoder.set_weights([
                np.ones((self.reinforce_agent.action_dim, 1)) * (max_dose - min_dose) / self.reinforce_agent.action_dim,
                np.array([min_dose])
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
            "state_dim": self.reinforce_agent.state_dim,
            "action_dim": self.reinforce_agent.action_dim,
            "continuous": self.reinforce_agent.continuous,
            "gamma": self.reinforce_agent.gamma
        }
    
    def save(self, filepath: str, **kwargs) -> None:
        """
        Guarda el modelo y los pesos del agente REINFORCE.
        
        Parámetros:
        -----------
        filepath : str
            Ruta donde guardar el modelo
        **kwargs
            Argumentos adicionales
        """
        # Guardar pesos de la política y del modelo wrapper
        self.save_weights(filepath + MODEL_WEIGHTS_EXT)
        
        # Guardar pesos de la política por separado
        policy_path = filepath + POLICY_WEIGHTS_EXT
        baseline_path = filepath + BASELINE_WEIGHTS_EXT if self.reinforce_agent.use_baseline else None
        
        self.reinforce_agent.save(policy_path, baseline_path)
        
    def load_weights(self, filepath: str, **kwargs) -> None:
        """
        Carga el modelo y los pesos del agente REINFORCE.
        
        Parámetros:
        -----------
        filepath : str
            Ruta desde donde cargar el modelo
        **kwargs
            Argumentos adicionales
        """
        # Determinar rutas de archivos según el filepath proporcionado
        if filepath.endswith(MODEL_WEIGHTS_EXT):
            base_path = filepath[:-len(MODEL_WEIGHTS_EXT)]
        else:
            base_path = filepath
            filepath = base_path + MODEL_WEIGHTS_EXT
            
        # Cargar pesos del wrapper
        super().load_weights(filepath)
        
        # Cargar pesos de la política
        policy_path = base_path + POLICY_WEIGHTS_EXT
        baseline_path = base_path + BASELINE_WEIGHTS_EXT if self.reinforce_agent.use_baseline else None
        
        self.reinforce_agent.load(policy_path, baseline_path)


# Constantes para evitar duplicación de strings
CGM_ENCODER = 'cgm_encoder'
OTHER_ENCODER = 'other_encoder'
STATE_ENCODER = 'state_encoder'
ACTION_DECODER = 'action_decoder'
# Constantes para evitar duplicación de archivos
MODEL_WEIGHTS_EXT = "_model.h5"
POLICY_WEIGHTS_EXT = "_policy.h5"
BASELINE_WEIGHTS_EXT = "_baseline.h5"


def create_reinforce_mcpg_model(cgm_shape: Tuple[int, ...], other_features_shape: Tuple[int, ...]) -> Model:
    """
    Crea un modelo basado en REINFORCE (Monte Carlo Policy Gradient) para predicción de dosis de insulina.
    
    Parámetros:
    -----------
    cgm_shape : Tuple[int, ...]
        Forma de los datos CGM (batch_size, time_steps, features)
    other_features_shape : Tuple[int, ...]
        Forma de otras características (batch_size, n_features)
        
    Retorna:
    --------
    Model
        Modelo REINFORCE que implementa la interfaz de Keras
    """
    # Definir dimensiones de entrada
    _ = cgm_shape[1] * cgm_shape[2] + other_features_shape[1]
    state_dim = 64  # Dimensión del espacio de estado
    
    # Configurar espacio de acciones
    action_dim = 20  # Acciones discretas (niveles de dosis)
    continuous = REINFORCE_CONFIG['continuous_action']
    
    # Crear agente REINFORCE
    reinforce_agent = REINFORCE(
        state_dim=state_dim,
        action_dim=action_dim if not continuous else 1,
        continuous=continuous,
        learning_rate=REINFORCE_CONFIG['learning_rate'],
        gamma=REINFORCE_CONFIG['gamma'],
        hidden_units=REINFORCE_CONFIG['hidden_units'],
        baseline=REINFORCE_CONFIG['use_baseline'],
        entropy_coef=REINFORCE_CONFIG['entropy_coef'],
        seed=REINFORCE_CONFIG.get('seed', 42)
    )
    
    # Crear y devolver el modelo wrapper
    return REINFORCEModel(
        reinforce_agent=reinforce_agent,
        cgm_shape=cgm_shape,
        other_features_shape=other_features_shape
    )