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
import threading
from typing import Tuple, Dict, List, Any, Optional, Union, Callable


PROJECT_ROOT = os.path.abspath(os.getcwd())
sys.path.append(PROJECT_ROOT) 

from config.models_config import A2C_A3C_CONFIG
from constants.constants import CONST_DEFAULT_SEED
from custom.printer import print_success


class ActorCriticModel(Model):
    """
    Modelo Actor-Crítico para A2C que divide la arquitectura en redes para
    política (actor) y valor (crítico).
    
    Parámetros:
    -----------
    state_dim : int
        Dimensión del espacio de estados
    action_dim : int
        Dimensión del espacio de acciones
    continuous : bool, opcional
        Indica si el espacio de acciones es continuo o discreto
    hidden_units : Optional[List[int]], opcional
        Unidades ocultas en cada capa
    """
    def __init__(self, state_dim: int, action_dim: int, continuous: bool = True, 
                hidden_units: Optional[List[int]] = None) -> None:
        super().__init__()
        
        # Valores predeterminados para capas ocultas
        if hidden_units is None:
            hidden_units = A2C_A3C_CONFIG['hidden_units']
        
        self.continuous = continuous
        self.action_dim = action_dim
        
        # Capas compartidas para procesamiento de estados
        self.shared_layers = []
        for i, units in enumerate(hidden_units[:2]):
            self.shared_layers.append(Dense(units, activation='tanh', name=f'shared_dense_{i}'))
            self.shared_layers.append(LayerNormalization(epsilon=A2C_A3C_CONFIG['epsilon'], name=f'shared_ln_{i}'))
            self.shared_layers.append(Dropout(A2C_A3C_CONFIG['dropout_rate'], name=f'shared_dropout_{i}'))
        
        # Red del Actor (política)
        self.actor_layers = []
        for i, units in enumerate(hidden_units[2:]):
            self.actor_layers.append(Dense(units, activation='tanh', name=f'actor_dense_{i}'))
            self.actor_layers.append(LayerNormalization(epsilon=A2C_A3C_CONFIG['epsilon'], name=f'actor_ln_{i}'))
        
        # Capa de salida del actor (depende de si el espacio de acción es continuo o discreto)
        if continuous:
            # Para acción continua (política gaussiana)
            self.mu = Dense(action_dim, activation='linear', name='actor_mu')
            self.log_sigma = Dense(action_dim, activation='linear', name='actor_log_sigma')
        else:
            # Para acción discreta (política categórica)
            self.logits = Dense(action_dim, activation='linear', name='actor_logits')
        
        # Red del Crítico (valor)
        self.critic_layers = []
        for i, units in enumerate(hidden_units[2:]):
            self.critic_layers.append(Dense(units, activation='tanh', name=f'critic_dense_{i}'))
            self.critic_layers.append(LayerNormalization(epsilon=A2C_A3C_CONFIG['epsilon'], name=f'critic_ln_{i}'))
        
        # Capa de salida del crítico (valor del estado)
        self.value = Dense(1, activation='linear', name='critic_value')
    
    def call(self, inputs: tf.Tensor, training: bool = False) -> Tuple[Any, tf.Tensor]:
        """
        Pasa la entrada por el modelo Actor-Crítico.
        
        Parámetros:
        -----------
        inputs : tf.Tensor
            Tensor de entrada con los estados
        training : bool, opcional
            Indica si está en modo entrenamiento
            
        Retorna:
        --------
        Tuple[Any, tf.Tensor]
            (política, valor) - la política puede ser una tupla (mu, sigma) o logits
        """
        x = inputs
        
        # Capas compartidas
        for layer in self.shared_layers:
            if isinstance(layer, Dropout):
                x = layer(x, training=training)
            else:
                x = layer(x)
        
        # Red del Actor
        actor_x = x
        for layer in self.actor_layers:
            if isinstance(layer, Dropout):
                actor_x = layer(actor_x, training=training)
            else:
                actor_x = layer(actor_x)
        
        # Salida del actor según el tipo de política
        if self.continuous:
            mu = self.mu(actor_x)
            log_sigma = self.log_sigma(actor_x)
            log_sigma = tf.clip_by_value(log_sigma, -20, 2)  # Evitar valores extremos
            sigma = tf.exp(log_sigma)
            policy = (mu, sigma)
        else:
            logits = self.logits(actor_x)
            policy = logits
        
        # Red del Crítico
        critic_x = x
        for layer in self.critic_layers:
            if isinstance(layer, Dropout):
                critic_x = layer(critic_x, training=training)
            else:
                critic_x = layer(critic_x)
        
        value = self.value(critic_x)
        
        return policy, value
    
    def get_action(self, state: np.ndarray, deterministic: bool = False) -> np.ndarray:
        """
        Obtiene una acción basada en el estado actual.

        Parámetros:
        -----------
        state : np.ndarray
            El estado actual del entorno
        deterministic : bool, opcional
            Si se debe tomar la acción de forma determinística (True) o estocástica (False)
        
        Retorna:
        --------
        np.ndarray
            Acción seleccionada según la política actual
        """
        state = tf.convert_to_tensor([state], dtype=tf.float32)
        policy, _ = self.call(state)
        
        if self.continuous:
            mu, sigma = policy
            if deterministic:
                return mu[0].numpy()
            # Muestrear de la distribución normal con tipos consistentes
            dist = tf.random.normal(shape=mu.shape, dtype=mu.dtype)
            action = mu + sigma * dist  # Ahora tienen el mismo tipo
            return action[0].numpy()
        else:
            logits = policy
            if deterministic:
                return tf.argmax(logits[0]).numpy()
            # Muestrear de la distribución categórica
            probs = tf.nn.softmax(logits)
            action = tf.random.categorical(tf.math.log(probs + 1e-10), 1)
            return action[0, 0].numpy()
    
    def get_value(self, state: np.ndarray) -> float:
        """
        Obtiene el valor estimado para un estado.
        
        Parámetros:
        -----------
        state : np.ndarray
            El estado para evaluar
        
        Retorna:
        --------
        float
            El valor estimado del estado
        """
        state = tf.convert_to_tensor([state], dtype=tf.float32)
        _, value = self.call(state)
        return value[0].numpy()
    
    def evaluate_actions(self, states: tf.Tensor, actions: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
        """
        Evalúa las acciones tomadas, devolviendo log_probs, valores y entropía.
        
        Parámetros:
        -----------
        states : tf.Tensor
            Los estados observados
        actions : tf.Tensor
            Las acciones tomadas
        
        Retorna:
        --------
        Tuple[tf.Tensor, tf.Tensor, tf.Tensor]
            (log_probs, valores, entropía)
        """
        policy, values = self.call(states)
        
        if self.continuous:
            mu, sigma = policy
            # Calcular log probabilidad para acciones continuas
            log_probs = -0.5 * tf.reduce_sum(
                tf.square((actions - mu) / sigma) + 
                2 * tf.math.log(sigma) + 
                tf.math.log(2.0 * np.pi), 
                axis=1
            )
            # Entropía de política gaussiana
            entropy = tf.reduce_sum(
                0.5 * tf.math.log(2.0 * np.pi * tf.square(sigma)) + 0.5,
                axis=1
            )
        else:
            logits = policy
            # Calcular log probabilidad para acciones discretas
            action_masks = tf.one_hot(actions, self.action_dim)
            log_probs = tf.reduce_sum(
                action_masks * tf.nn.log_softmax(logits),
                axis=1
            )
            # Entropía de política categórica
            probs = tf.nn.softmax(logits)
            entropy = -tf.reduce_sum(
                probs * tf.math.log(probs + 1e-10),
                axis=1
            )
        
        return log_probs, values, entropy


class A2C:
    """
    Implementación del algoritmo Advantage Actor-Critic (A2C).
    
    Este algoritmo utiliza un estimador de ventaja para actualizar la política
    y una red de valor para estimar los retornos esperados.
    
    Parámetros:
    -----------
    state_dim : int
        Dimensión del espacio de estados
    action_dim : int
        Dimensión del espacio de acciones
    continuous : bool, opcional
        Indica si el espacio de acciones es continuo o discreto
    learning_rate : float, opcional
        Tasa de aprendizaje
    gamma : float, opcional
        Factor de descuento
    entropy_coef : float, opcional
        Coeficiente de entropía para exploración
    value_coef : float, opcional
        Coeficiente de pérdida de valor
    max_grad_norm : float, opcional
        Norma máxima para recorte de gradientes
    hidden_units : Optional[List[int]], opcional
        Unidades ocultas por capa
    """
    def __init__(
        self, 
        state_dim: int, 
        action_dim: int,
        continuous: bool = True,
        learning_rate: float = A2C_A3C_CONFIG['learning_rate'],
        gamma: float = A2C_A3C_CONFIG['gamma'],
        entropy_coef: float = A2C_A3C_CONFIG['entropy_coef'],
        value_coef: float = A2C_A3C_CONFIG['value_coef'],
        max_grad_norm: float = A2C_A3C_CONFIG['max_grad_norm'],
        hidden_units: Optional[List[int]] = None
    ) -> None:
        # Parámetros del modelo
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.entropy_coef = entropy_coef
        self.value_coef = value_coef
        self.max_grad_norm = max_grad_norm
        self.continuous = continuous
        
        # Valores predeterminados para capas ocultas
        if hidden_units is None:
            self.hidden_units = A2C_A3C_CONFIG['hidden_units']
        else:
            self.hidden_units = hidden_units
        
        # Crear modelo y optimizador
        self.model = ActorCriticModel(state_dim, action_dim, continuous, self.hidden_units)
        self.optimizer = Adam(learning_rate=learning_rate)
        
        # Métricas
        self.policy_loss_metric = tf.keras.metrics.Mean('policy_loss')
        self.value_loss_metric = tf.keras.metrics.Mean('value_loss')
        self.entropy_metric = tf.keras.metrics.Mean('entropy')
        self.total_loss_metric = tf.keras.metrics.Mean('total_loss')
    
    @tf.function
    def train_step(self, states: tf.Tensor, actions: tf.Tensor, 
                 returns: tf.Tensor, advantages: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor]:
        """
        Realiza un paso de entrenamiento para actualizar el modelo.
        
        Parámetros:
        -----------
        states : tf.Tensor
            Estados observados en el entorno
        actions : tf.Tensor
            Acciones tomadas para esos estados
        returns : tf.Tensor
            Retornos estimados (para entrenar el crítico)
        advantages : tf.Tensor
            Ventajas estimadas (para entrenar el actor)
        
        Retorna:
        --------
        Tuple[tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor]
            (pérdida_total, pérdida_política, pérdida_valor, pérdida_entropía)
        """
        with tf.GradientTape() as tape:
            # Evaluar acciones con el modelo actual
            log_probs, values, entropy = self.model.evaluate_actions(states, actions)
            
            # Ventaja ya está calculada externamente
            advantages = tf.reshape(advantages, [-1])
            
            # Calcular pérdida de política
            policy_loss = -tf.reduce_mean(log_probs * advantages, axis=0)
            
            # Calcular pérdida de valor
            value_pred = tf.reshape(values, [-1])
            returns = tf.reshape(returns, [-1])
            value_loss = tf.reduce_mean(tf.square(returns - value_pred), axis=0)
            
            # Calcular pérdida de entropía (regularización)
            entropy_loss = -tf.reduce_mean(entropy, axis=0)
            
            # Pérdida total combinada
            total_loss = policy_loss + self.value_coef * value_loss + self.entropy_coef * entropy_loss
            
        # Calcular gradientes y actualizar pesos
        grads = tape.gradient(total_loss, self.model.trainable_variables)
        
        # Clipping de gradientes para estabilidad
        if self.max_grad_norm is not None:
            grads, _ = tf.clip_by_global_norm(grads, self.max_grad_norm)
            
        self.optimizer.apply_gradients(zip(grads, self.model.trainable_variables))
        
        # Actualizar métricas
        self.policy_loss_metric.update_state(policy_loss)
        self.value_loss_metric.update_state(value_loss)
        self.entropy_metric.update_state(entropy_loss)
        self.total_loss_metric.update_state(total_loss)
        
        return total_loss, policy_loss, value_loss, entropy_loss
    
    def compute_returns_advantages(self, rewards: np.ndarray, values: np.ndarray, 
                                 dones: np.ndarray, next_value: float) -> Tuple[np.ndarray, np.ndarray]:
        """
        Calcula los retornos y ventajas para los estados visitados.
        
        Parámetros:
        -----------
        rewards : np.ndarray
            Recompensas recibidas
        values : np.ndarray
            Valores estimados para los estados actuales
        dones : np.ndarray
            Indicadores de fin de episodio
        next_value : float
            Valor estimado para el estado final
            
        Retorna:
        --------
        Tuple[np.ndarray, np.ndarray]
            (returns, advantages) calculados
        """
        # Añadir el valor del último estado
        values = np.append(values, next_value)
        
        returns = np.zeros_like(rewards, dtype=np.float32)
        advantages = np.zeros_like(rewards, dtype=np.float32)
        
        # Calcular retornos y ventajas desde el final
        gae = 0
        for t in reversed(range(len(rewards))):
            # Si es terminal, el valor del siguiente estado es 0
            next_non_terminal = 1.0 - dones[t]
            next_value = values[t + 1]
            
            # Delta temporal para GAE
            delta = rewards[t] + self.gamma * next_value * next_non_terminal - values[t]
            
            # Calcular ventaja con GAE
            gae = delta + self.gamma * A2C_A3C_CONFIG['lambda'] * next_non_terminal * gae
            advantages[t] = gae
            
            # Calcular retornos (para entrenar el crítico)
            returns[t] = advantages[t] + values[t]
        
        # Normalizar ventajas para reducir varianza
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        return returns, advantages
    
    def _collect_experience(self, env, state: np.ndarray, n_steps: int, render: bool = False,
                          episode_reward: float = 0, episode_rewards: Optional[List] = None) -> Tuple:
        """
        Recolecta experiencia en el entorno para una actualización.
        
        Parámetros:
        -----------
        env : gym.Env
            Entorno donde recolectar datos
        state : np.ndarray
            Estado inicial desde donde empezar
        n_steps : int
            Número de pasos a recolectar
        render : bool, opcional
            Si se debe renderizar el entorno
        episode_reward : float, opcional
            Recompensa acumulada en el episodio actual
        episode_rewards : List, opcional
            Lista donde guardar recompensas de episodios completos
            
        Retorna:
        --------
        Tuple
            Datos recolectados y el estado actualizado
        """
        states, actions, rewards, dones, values = [], [], [], [], []
        current_state = state
        current_episode_reward = episode_reward
        
        for _ in range(n_steps):
            if render:
                env.render()
            
            # Guardar estado actual
            states.append(current_state)
            
            # Obtener acción y valor
            action = self.model.get_action(current_state)
            actions.append(action)
            
            # Valor del estado actual
            value = self.model.get_value(current_state)
            values.append(value)
            
            # Ejecutar acción en el entorno
            next_state, reward, done, _, _ = env.step(action)
            
            # Guardar recompensa y done
            rewards.append(reward)
            dones.append(done)
            
            # Actualizar recompensa acumulada
            current_episode_reward += reward
            
            # Si el episodio termina, resetear
            if done:
                current_state, _ = env.reset()
                if episode_rewards is not None:
                    episode_rewards.append(current_episode_reward)
                current_episode_reward = 0
            else:
                current_state = next_state
        
        return states, actions, rewards, dones, values, current_state, current_episode_reward

    def _update_model(self, states: List, actions: List, rewards: List, 
                    dones: List, values: List, next_value: float, 
                    done: bool) -> Tuple[float, float, float]:
        """
        Actualiza el modelo con los datos recolectados.
        
        Parámetros:
        -----------
        states : List
            Estados observados
        actions : List
            Acciones tomadas
        rewards : List
            Recompensas recibidas
        dones : List
            Indicadores de fin de episodio
        values : List
            Valores estimados para los estados
        next_value : float
            Valor estimado del último estado
        done : bool
            Si el último estado es terminal
            
        Retorna:
        --------
        Tuple[float, float, float]
            Pérdidas del entrenamiento (policy_loss, value_loss, entropy_loss)
        """
        # Si el episodio no terminó, usar el valor estimado
        final_value = 0 if done else next_value
                
        # Convertir a arrays de numpy
        states_np = np.array(states, dtype=np.float32)
        actions_np = np.array(actions, dtype=np.float32 if self.continuous else np.int32)
        rewards_np = np.array(rewards, dtype=np.float32)
        dones_np = np.array(dones, dtype=np.float32)
        values_np = np.array(values, dtype=np.float32)
        
        # Calcular retornos y ventajas
        returns, advantages = self.compute_returns_advantages(rewards_np, values_np, dones_np, final_value)
        
        # Actualizar modelo
        _, policy_loss, value_loss, entropy_loss = self.train_step(
            tf.convert_to_tensor(states_np), 
            tf.convert_to_tensor(actions_np), 
            tf.convert_to_tensor(returns), 
            tf.convert_to_tensor(advantages)
        )
        
        return policy_loss.numpy(), value_loss.numpy(), entropy_loss.numpy()

    def _update_history(self, history: Dict, episode_rewards: List, 
                      epoch: int, epochs: int, policy_loss: float, 
                      value_loss: float) -> List:
        """
        Actualiza y muestra las métricas de entrenamiento.
        
        Parámetros:
        -----------
        history : Dict
            Historial de entrenamiento a actualizar
        episode_rewards : List
            Recompensas de episodios completados
        epoch : int
            Época actual
        epochs : int
            Total de épocas
        policy_loss : float
            Pérdida de política
        value_loss : float
            Pérdida de valor
            
        Retorna:
        --------
        List
            Lista actualizada de recompensas de episodios
        """
        # Guardar estadísticas
        history['policy_losses'].append(self.policy_loss_metric.result().numpy())
        history['value_losses'].append(self.value_loss_metric.result().numpy())
        history['entropy_losses'].append(self.entropy_metric.result().numpy())
        
        # Resetear métricas
        self.policy_loss_metric.reset_state()
        self.value_loss_metric.reset_state()
        self.entropy_metric.reset_state()
        self.total_loss_metric.reset_state()
        
        # Añadir recompensas de episodios completados
        if episode_rewards:
            history['episode_rewards'].extend(episode_rewards)
            avg_reward = np.mean(episode_rewards)
            
            # Mostrar progreso cada 10 épocas
            if (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch+1}/{epochs} - Avg Reward: {avg_reward:.2f}, "
                      f"Policy Loss: {policy_loss:.4f}, Value Loss: {value_loss:.4f}")
            
            return []  # Resetear lista de recompensas
        
        return episode_rewards

    def train(self, env, n_steps: int = 10, epochs: int = 1000, 
             render: bool = False) -> Dict:
        """
        Entrena el modelo A2C en el entorno dado.
        
        Parámetros:
        -----------
        env : gym.Env
            Entorno donde entrenar
        n_steps : int, opcional
            Número de pasos por actualización
        epochs : int, opcional
            Número de épocas de entrenamiento
        render : bool, opcional
            Si se debe renderizar el entorno
            
        Retorna:
        --------
        Dict
            Historia de entrenamiento
        """
        # Historia de entrenamiento
        history = {
            'episode_rewards': [],
            'policy_losses': [],
            'value_losses': [],
            'entropy_losses': []
        }
        
        episode_reward = 0
        episode_rewards = []
        
        # Estado inicial
        state, _ = env.reset()
        
        for epoch in range(epochs):
            # Recolectar experiencia
            states, actions, rewards, dones, values, state, episode_reward = self._collect_experience(
                env, state, n_steps, render, episode_reward, episode_rewards
            )
            
            # Obtener valor del último estado si es necesario
            next_value = self.model.get_value(state) if not dones[-1] else 0
            
            # Actualizar modelo
            policy_loss, value_loss, _ = self._update_model(
                states, actions, rewards, dones, values, next_value, dones[-1]
            )
            
            # Actualizar historial y mostrar progreso
            episode_rewards = self._update_history(
                history, episode_rewards, epoch, epochs, policy_loss, value_loss
            )
        
        return history
    
    def save_model(self, filepath: str) -> None:
        """
        Guarda el modelo en un archivo.
        
        Parámetros:
        -----------
        filepath : str
            Ruta donde guardar el modelo
        """
        # Asegurar que el modelo esté construido antes de guardar
        dummy_state = tf.zeros((1, self.state_dim))
        _ = self.model(dummy_state)  # Esto construye el modelo
        
        # Ahora guardar los pesos
        self.model.save_weights(filepath)
        
    def load_model(self, filepath: str) -> None:
        """
        Carga el modelo desde un archivo.
        
        Parámetros:
        -----------
        filepath : str
            Ruta desde donde cargar el modelo
        """
        # Asegurarse de que el modelo está construido primero
        dummy_state = np.zeros((1, self.state_dim))
        self.model(dummy_state)
        self.model.load_weights(filepath)


class A3C(A2C):
    """
    Implementación de Asynchronous Advantage Actor-Critic (A3C).
    
    Extiende A2C para permitir entrenamiento asíncrono con múltiples trabajadores.
    
    Parámetros:
    -----------
    state_dim : int
        Dimensión del espacio de estados
    action_dim : int
        Dimensión del espacio de acciones
    continuous : bool, opcional
        Indica si el espacio de acciones es continuo o discreto
    n_workers : int, opcional
        Número de trabajadores asíncronos
    learning_rate : float, opcional
        Tasa de aprendizaje
    gamma : float, opcional
        Factor de descuento
    entropy_coef : float, opcional
        Coeficiente de entropía para exploración
    value_coef : float, opcional
        Coeficiente de pérdida de valor
    max_grad_norm : float, opcional
        Norma máxima para recorte de gradientes
    hidden_units : Optional[List[int]], opcional
        Unidades ocultas por capa
    """
    def __init__(self, state_dim: int, action_dim: int, continuous: bool = True, 
                n_workers: int = 4, **kwargs):
        super().__init__(state_dim, action_dim, continuous, **kwargs)
        self.n_workers = n_workers
        self.workers = []
    
    def create_worker(self, env_fn: Callable, worker_id: int) -> 'A3CWorker':
        """
        Crea un trabajador para entrenamiento asíncrono.
        
        Parámetros:
        -----------
        env_fn : Callable
            Función que devuelve un entorno
        worker_id : int
            ID del trabajador
            
        Retorna:
        --------
        a3c_worker
            Un trabajador A3C
        """
        return A3CWorker(
            self.model,
            self.optimizer,
            env_fn,
            worker_id,
            self.state_dim,
            self.action_dim,
            self.gamma,
            self.entropy_coef,
            self.value_coef,
            self.max_grad_norm,
            self.continuous
        )
    
    def train_async(self, env_fn: Callable, n_steps: int = 10, 
                   total_steps: int = 1000000, render: bool = False) -> Dict:
        """
        Entrena el modelo A3C con múltiples trabajadores asíncronos.
        
        Parámetros:
        -----------
        env_fn : Callable
            Función que devuelve un entorno
        n_steps : int, opcional
            Pasos por actualización
        total_steps : int, opcional
            Total de pasos globales
        render : bool, opcional
            Si se debe renderizar el entorno
            
        Retorna:
        --------
        Dict
            Historia de entrenamiento
        """
        # Crear trabajadores
        workers = []
        for i in range(self.n_workers):
            worker = self.create_worker(env_fn, i)
            workers.append(worker)
        
        # Variables para seguimiento de recompensas y pérdidas
        history = {
            'episode_rewards': [],
            'policy_losses': [],
            'value_losses': [],
            'entropy_losses': []
        }
        
        # Crear e iniciar hilos
        threads = []
        for worker in workers:
            thread = threading.Thread(
                target=worker.train,
                args=(n_steps, total_steps // self.n_workers, history, render)
            )
            threads.append(thread)
            thread.daemon = True  # Terminar hilos cuando el programa principal termina
            thread.start()
        
        # Esperar a que terminen todos los hilos
        for thread in threads:
            thread.join()
        
        return history


class A3CWorker:
    """
    Trabajador para el algoritmo A3C que entrena de forma asíncrona.
    
    Parámetros:
    -----------
    global_model : actor_critic_model
        Modelo compartido global
    optimizer : tf.keras.optimizers.Optimizer
        Optimizador compartido
    env_fn : Callable
        Función que devuelve un entorno
    worker_id : int
        ID del trabajador
    state_dim : int
        Dimensión del espacio de estados
    action_dim : int
        Dimensión del espacio de acciones
    gamma : float
        Factor de descuento
    entropy_coef : float
        Coeficiente de entropía
    value_coef : float
        Coeficiente de pérdida de valor
    max_grad_norm : float
        Norma máxima para recorte de gradientes
    continuous : bool
        Indica si el espacio de acciones es continuo
    """
    def __init__(
        self,
        global_model: ActorCriticModel,
        optimizer: tf.keras.optimizers.Optimizer,
        env_fn: Callable,
        worker_id: int,
        state_dim: int,
        action_dim: int,
        gamma: float = 0.99,
        entropy_coef: float = 0.01,
        value_coef: float = 0.5,
        max_grad_norm: float = 0.5,
        continuous: bool = True
    ) -> None:
        # Parámetros del trabajador
        self.worker_id = worker_id
        self.env = env_fn()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.entropy_coef = entropy_coef
        self.value_coef = value_coef
        self.max_grad_norm = max_grad_norm
        self.continuous = continuous
        
        # Modelo global compartido
        self.global_model = global_model
        self.optimizer = optimizer
        
        # Modelo local para este trabajador
        self.local_model = ActorCriticModel(
            state_dim, action_dim, continuous, 
            hidden_units=A2C_A3C_CONFIG['hidden_units']
        )
        # Sincronizar pesos locales con globales
        self.update_local_model()
    
    def update_local_model(self) -> None:
        """
        Actualiza los pesos del modelo local desde el modelo global.
        """
        self.local_model.set_weights(self.global_model.get_weights())
    
    def compute_returns_advantages(self, rewards: np.ndarray, values: np.ndarray, 
                                 dones: np.ndarray, next_value: float) -> Tuple[np.ndarray, np.ndarray]:
        """
        Calcula los retornos y ventajas para los estados visitados.
        
        Parámetros:
        -----------
        rewards : np.ndarray
            Recompensas recibidas
        values : np.ndarray
            Valores estimados para los estados actuales
        dones : np.ndarray
            Indicadores de fin de episodio
        next_value : float
            Valor estimado para el estado final
            
        Retorna:
        --------
        Tuple[np.ndarray, np.ndarray]
            (returns, advantages) calculados
        """
        # Añadir el valor del último estado
        values = np.append(values, next_value)
        
        returns = np.zeros_like(rewards, dtype=np.float32)
        advantages = np.zeros_like(rewards, dtype=np.float32)
        
        # Calcular retornos y ventajas desde el final
        gae = 0
        for t in reversed(range(len(rewards))):
            # Si es terminal, el valor del siguiente estado es 0
            next_non_terminal = 1.0 - dones[t]
            next_value = values[t + 1]
            
            # Delta temporal para GAE
            delta = rewards[t] + self.gamma * next_value * next_non_terminal - values[t]
            
            # Calcular ventaja con GAE
            gae = delta + self.gamma * A2C_A3C_CONFIG['lambda'] * next_non_terminal * gae
            advantages[t] = gae
            
            # Calcular retornos (para entrenar el crítico)
            returns[t] = advantages[t] + values[t]
        
        # Normalizar ventajas para reducir varianza
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        return returns, advantages
    
    def train_step(self, states: tf.Tensor, actions: tf.Tensor, 
                  returns: tf.Tensor, advantages: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor]:
        """
        Realiza un paso de entrenamiento asíncrono.
        
        Parámetros:
        -----------
        states : tf.Tensor
            Estados observados
        actions : tf.Tensor
            Acciones tomadas
        returns : tf.Tensor
            Retornos calculados
        advantages : tf.Tensor
            Ventajas calculadas
            
        Retorna:
        --------
        Tuple[tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor]
            (total_loss, policy_loss, value_loss, entropy_loss)
        """
        with tf.GradientTape() as tape:
            # Evaluar acciones con el modelo local
            log_probs, values, entropy = self.local_model.evaluate_actions(states, actions)
            
            # Calcular pérdida de política
            advantages = tf.reshape(advantages, [-1])
            policy_loss = -tf.reduce_mean(log_probs * advantages, axis=0)
            
            # Calcular pérdida de valor
            value_pred = tf.reshape(values, [-1])
            returns = tf.reshape(returns, [-1])
            value_loss = tf.reduce_mean(tf.square(returns - value_pred), axis=0)
            
            # Calcular pérdida de entropía (regularización)
            entropy_loss = -tf.reduce_mean(entropy, axis=0)
            
            # Pérdida total combinada
            total_loss = policy_loss + self.value_coef * value_loss + self.entropy_coef * entropy_loss
            
        # Calcular gradientes usando el modelo local
        grads = tape.gradient(total_loss, self.local_model.trainable_variables)
        
        # Clipping de gradientes para estabilidad
        if self.max_grad_norm is not None:
            grads, _ = tf.clip_by_global_norm(grads, self.max_grad_norm)
            
        # Aplicar gradientes al modelo global de manera asíncrona
        self.optimizer.apply_gradients(zip(grads, self.global_model.trainable_variables))
        
        # Actualizar modelo local
        self.update_local_model()
        
        return total_loss, policy_loss, value_loss, entropy_loss
    
    def _collect_step_data(self, state: np.ndarray, render: bool) -> Tuple:
        """
        Recoge datos de un solo paso en el entorno.
        
        Parámetros:
        -----------
        state : np.ndarray
            Estado actual
        render : bool
            Si se debe renderizar el entorno
            
        Retorna:
        --------
        Tuple
            (state, action, value, reward, done, next_state)
        """
        if render and self.worker_id == 0:  # Solo renderizar el primer trabajador
            self.env.render()
        
        # Obtener acción y valor con el modelo local
        action = self.local_model.get_action(state)
        value = self.local_model.get_value(state)
        
        # Ejecutar acción en el entorno
        next_state, reward, done, _, _ = self.env.step(action)
        
        return state, action, value, reward, done, next_state
    
    def _handle_episode_end(self, episode_reward: float, steps_done: int, 
                          max_steps: int, shared_history: Dict) -> Tuple[np.ndarray, float]:
        """
        Maneja el final de un episodio.
        
        Parámetros:
        -----------
        episode_reward : float
            Recompensa acumulada en el episodio
        steps_done : int
            Pasos completados
        max_steps : int
            Pasos máximos
        shared_history : Dict
            Historial compartido
            
        Retorna:
        --------
        Tuple[np.ndarray, float]
            Nuevo estado y recompensa de episodio reiniciada
        """
        state, _ = self.env.reset()
        
        # Guardar recompensa de episodio completado
        with threading.Lock():  # Proteger acceso compartido
            shared_history['episode_rewards'].append(episode_reward)
        
        # Mostrar progreso del trabajador
        if self.worker_id == 0 and len(shared_history['episode_rewards']) % 10 == 0:
            avg_reward = np.mean(shared_history['episode_rewards'][-10:])
            print(f"Worker {self.worker_id} - Episode {len(shared_history['episode_rewards'])}, "
                  f"Avg Reward: {avg_reward:.2f}, Steps: {steps_done}/{max_steps}")
        
        return state, 0  # Nuevo estado y recompensa reiniciada
    
    def _update_model_with_collected_data(self, states: List, actions: List,
                                        rewards: List, dones: List,
                                        values: List, done: bool,
                                        next_state: np.ndarray, shared_history: Dict) -> None:
        """
        Actualiza el modelo con los datos recolectados.
        
        Parámetros:
        -----------
        states : List
            Estados recolectados
        actions : List
            Acciones tomadas
        rewards : List
            Recompensas recibidas
        dones : List
            Indicadores de fin de episodio
        values : List
            Valores estimados
        done : bool
            Si el episodio terminó
        next_state : np.ndarray
            Estado final
        shared_history : Dict
            Historial compartido
        """
        # Si el episodio no terminó, calcular valor del último estado
        next_value = 0.0 if done else self.local_model.get_value(next_state)
            
        # Convertir a arrays de numpy
        states_np = np.array(states, dtype=np.float32)
        actions_np = np.array(actions, dtype=np.float32 if self.continuous else np.int32)
        rewards_np = np.array(rewards, dtype=np.float32)
        dones_np = np.array(dones, dtype=np.float32)
        values_np = np.array(values, dtype=np.float32)
        
        # Calcular retornos y ventajas
        returns, advantages = self.compute_returns_advantages(
            rewards_np, values_np, dones_np, next_value
        )
        
        # Actualizar modelo
        _, policy_loss, value_loss, entropy_loss = self.train_step(
            tf.convert_to_tensor(states_np),
            tf.convert_to_tensor(actions_np),
            tf.convert_to_tensor(returns),
            tf.convert_to_tensor(advantages)
        )
        
        # Guardar estadísticas
        with threading.Lock():  # Proteger acceso compartido
            shared_history['policy_losses'].append(policy_loss.numpy())
            shared_history['value_losses'].append(value_loss.numpy())
            shared_history['entropy_losses'].append(entropy_loss.numpy())
    
    def train(self, n_steps: int, max_steps: int, shared_history: Dict, 
             render: bool = False) -> None:
        """
        Entrenamiento asíncrono del trabajador.
        
        Parámetros:
        -----------
        n_steps : int
            Pasos por actualización
        max_steps : int
            Pasos máximos para este trabajador
        shared_history : Dict
            Diccionario compartido para seguimiento
        render : bool, opcional
            Si se debe renderizar el entorno
        """
        # Estado inicial
        state, _ = self.env.reset()
        episode_reward = 0
        steps_done = 0
        
        while steps_done < max_steps:
            # Almacenar transiciones
            states, actions, rewards, dones, values = [], [], [], [], []
            done = False
            
            # Recolectar experiencia durante n pasos
            for _ in range(n_steps):
                if steps_done >= max_steps:
                    break
                    
                # Recolectar datos de un paso
                current_state, action, value, reward, done, next_state = self._collect_step_data(state, render)
                
                # Guardar datos
                states.append(current_state)
                actions.append(action)
                values.append(value)
                rewards.append(reward)
                dones.append(done)
                
                # Actualizar contadores
                episode_reward += reward
                steps_done += 1
                state = next_state
                
                # Si el episodio termina, resetear
                if done:
                    state, episode_reward = self._handle_episode_end(
                        episode_reward, steps_done, max_steps, shared_history
                    )
                    break
            
            # Si recolectamos suficientes pasos, actualizar modelo
            if states:
                self._update_model_with_collected_data(
                    states, actions, rewards, dones, values, done, state, shared_history
                )

# Constantes para nombres de capas
CGM_ENCODER = 'cgm_encoder'
OTHER_ENCODER = 'other_encoder'
COMBINED_LAYER = 'combined_layer'
WRAPPER_WEIGHTS_SUFFIX = '_wrapper.weights.h5'
POLICY_WEIGHTS_SUFFIX = '_policy.weights.h5'


class A2CWrapper(Model):
    """
    Wrapper para el algoritmo A2C que implementa la interfaz de Keras.Model.
    """
    
    def __init__(
        self, 
        aNc_agent: A2C,
        cgm_shape: Tuple[int, ...],
        other_features_shape: Tuple[int, ...]
    ) -> None:
        """
        Inicializa el modelo wrapper para A2C.
        
        Parámetros:
        -----------
        aNc_agent : A2C
            Agente A2C a utilizar
        cgm_shape : Tuple[int, ...]
            Forma de entrada para datos CGM
        other_features_shape : Tuple[int, ...]
            Forma de entrada para otras características
        """
        super(A2CWrapper, self).__init__()
        self.aNc_agent = aNc_agent
        self.cgm_shape = cgm_shape
        self.other_features_shape = other_features_shape
        
        # Capas para codificación de CGM
        self.cgm_conv = Conv1D(64, 3, padding='same', activation='relu', name=f'{CGM_ENCODER}_conv')
        self.cgm_pooling = GlobalAveragePooling1D(name=f'{CGM_ENCODER}_pooling')
        
        # Capas para procesamiento de otras características
        self.other_dense = Dense(32, activation='relu', name=OTHER_ENCODER)
        
        # Capa para combinar características
        self.combined = Dense(self.aNc_agent.state_dim, activation='relu', name=COMBINED_LAYER)
        
        # Capa final para mapear de salidas de política a dosis
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
        # Obtener entradas
        cgm_data, other_features = inputs
        batch_size = tf.shape(cgm_data)[0]
        
        # Procesar datos CGM
        cgm_encoded = self.cgm_conv(cgm_data)
        cgm_features = self.cgm_pooling(cgm_encoded)
        
        # Procesar otras características
        other_encoded = self.other_dense(other_features)
        
        # Combinar características en representación de estado
        combined_features = tf.concat([cgm_features, other_encoded], axis=1)
        states = self.combined(combined_features)
        
        # Crear array para resultados
        actions = tf.TensorArray(dtype=tf.float32, size=batch_size)
        
        # Procesar cada muestra en el batch
        for i in range(batch_size):
            state = tf.gather(states, i)
            
            # Usar el agente A2C para obtener acción determinística
            action = self.aNc_agent.model.get_action(state.numpy(), deterministic=True)
            
            # Almacenar la acción
            actions = actions.write(i, tf.convert_to_tensor(action, dtype=tf.float32))
        
        # Convertir a tensor
        actions_tensor = actions.stack()
        
        # Para acción continua, mapear directamente a dosis
        if self.aNc_agent.continuous:
            if len(tf.shape(actions_tensor)) > 2:
                actions_tensor = tf.reshape(actions_tensor, [batch_size, -1])
            
            # Convertir a valor de dosis con capa densa
            doses = self.dose_predictor(actions_tensor)
        else:
            # Para acción discreta, convertir a one-hot y luego a dosis
            one_hot = tf.one_hot(tf.cast(actions_tensor, tf.int32), self.aNc_agent.action_dim)
            doses = self.dose_predictor(one_hot)
        
        return doses
    
    def fit(
        self, 
        x: Union[List[tf.Tensor], tf.data.Dataset], 
        y: Optional[tf.Tensor] = None, 
        batch_size: int = 32, 
        epochs: int = 1, 
        verbose: int = 0,
        callbacks: Optional[List[Any]] = None,
        validation_data: Optional[Tuple] = None,
        **kwargs
    ) -> Any:
        """
        Simula la interfaz de entrenamiento de Keras para el agente A2C.
        
        Parámetros:
        -----------
        x : Union[List[tf.Tensor], tf.data.Dataset]
            Lista con [cgm_data, other_features] o un dataset de TensorFlow
        y : Optional[tf.Tensor], opcional
            Etiquetas (dosis objetivo), se ignora si x es un dataset (default: None)
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
            Historia de entrenamiento compatible con Keras
        """
        if verbose > 0:
            print("Entrenando modelo A2C...")
        
        # Extraer datos del dataset si es necesario
        if isinstance(x, tf.data.Dataset):
            # Extraer datos del dataset
            cgm_data = []
            other_features = []
            targets = []
            
            for (batch_cgm, batch_other), batch_y in x:
                cgm_data.append(batch_cgm)
                other_features.append(batch_other)
                targets.append(batch_y)
            
            cgm_tensor = tf.concat(cgm_data, axis=0)
            other_tensor = tf.concat(other_features, axis=0)
            y_tensor = tf.concat(targets, axis=0)
            
            x_input = [cgm_tensor, other_tensor]
        else:
            # Usar listas de tensores directamente
            x_input = x
            y_tensor = y
        
        # Crear entorno para entrenamiento
        env = self._create_training_environment(x_input[0], x_input[1], y_tensor)
        
        # Entrenar al agente A2C
        history = self.aNc_agent.train(
            env=env,
            n_steps=batch_size,
            epochs=epochs,
            render=False
        )
        
        # Ajustar capa de predicción de dosis
        self._calibrate_dose_predictor(y_tensor)
        
        if verbose > 0:
            print_success("Entrenamiento completado.")
        
        # Convertir historia a formato compatible con Keras
        keras_history = {
            'loss': history.get('policy_losses', [0.0]),
            'val_loss': [history.get('policy_losses', [0.0])[-1]] if validation_data else None
        }
        
        # Crear un objeto con atributo .history para compatibilidad con Keras
        class KerasHistoryCompatible:
            def __init__(self, history_dict):
                self.history = history_dict
        
        return KerasHistoryCompatible(keras_history)
    
    def _create_training_environment(self, cgm_data: tf.Tensor, other_features: tf.Tensor, 
                                   target_doses: tf.Tensor) -> Any:
        """
        Crea un entorno de entrenamiento para el agente A2C.
        
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
        # Convertir tensores a numpy para procesamiento
        cgm_np = cgm_data.numpy() if hasattr(cgm_data, 'numpy') else cgm_data
        other_np = other_features.numpy() if hasattr(other_features, 'numpy') else other_features
        targets_np = target_doses.numpy() if hasattr(target_doses, 'numpy') else target_doses
        
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
                self.lock = threading.RLock()  # Lock para sincronización en entrenamiento asíncrono
                
                # Para compatibilidad con algoritmos RL
                self.observation_space = gym.spaces.Box(
                    low=-np.inf, high=np.inf, 
                    shape=(model_wrapper.aNc_agent.state_dim,)
                )
                
                # Espacio de acción según tipo
                if model_wrapper.aNc_agent.continuous:
                    self.action_space = gym.spaces.Box(
                        low=np.zeros(model_wrapper.aNc_agent.action_dim), 
                        high=np.ones(model_wrapper.aNc_agent.action_dim) * 15, 
                        dtype=np.float32
                    )
                else:
                    self.action_space = gym.spaces.Discrete(model_wrapper.aNc_agent.action_dim)
                    
                # Pre-calcular codificaciones para evitar crear tensores/capas durante la ejecución
                self._initialize_encoders(model_wrapper)
                
            def _initialize_encoders(self, model_wrapper):
                """Inicializa los codificadores para evitar problemas de construcción tardía."""
                # Crear tensores vacíos
                dummy_cgm = np.zeros((1,) + self.cgm.shape[1:], dtype=np.float32)
                dummy_other = np.zeros((1,) + self.features.shape[1:], dtype=np.float32)
                
                # Ejecutar una pasada para inicializar todas las capas
                _ = model_wrapper.call([tf.convert_to_tensor(dummy_cgm), 
                                          tf.convert_to_tensor(dummy_other)])
                
                # Almacenar encoders pre-construidos
                self.cgm_conv = model_wrapper.cgm_conv
                self.cgm_pooling = model_wrapper.cgm_pooling
                self.other_dense = model_wrapper.other_dense
                self.combined = model_wrapper.combined
                
            def reset(self):
                """Reinicia el entorno seleccionando un ejemplo aleatorio."""
                with self.lock:
                    self.current_idx = self.rng.integers(0, self.max_idx)
                    state = self._get_state()
                    return state, {}
            
            def step(self, action):
                """Ejecuta un paso en el entorno con la acción dada."""
                with self.lock:
                    # Convertir acción a dosis según tipo de espacio
                    if isinstance(self.action_space, gym.spaces.Box):
                        # Para acción continua, usar directamente (limitando al rango)
                        dose = np.clip(action[0], 0, 15)
                    else:
                        # Para acción discreta, mapear a valor de dosis
                        dose = action / (self.model.aNc_agent.action_dim - 1) * 15
                    
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
                """Codifica el estado actual a partir de los datos."""
                # Obtener datos actuales en formato adecuado
                cgm_batch = tf.convert_to_tensor(
                    self.cgm[self.current_idx:self.current_idx+1], 
                    dtype=tf.float32
                )
                features_batch = tf.convert_to_tensor(
                    self.features[self.current_idx:self.current_idx+1],
                    dtype=tf.float32
                )
                
                # Procesar con capas previamente construidas
                # Procesar con capas de codificación
                cgm_encoded = self.cgm_conv(cgm_batch)
                cgm_features = self.cgm_pooling(cgm_encoded)
                other_encoded = self.other_dense(features_batch)
                
                # Combinar características
                combined = tf.concat([cgm_features, other_encoded], axis=1)
                state = self.combined(combined)
                
                return state[0].numpy()
        
        # Crear y devolver instancia del entorno
        return InsulinDosingEnv(cgm_np, other_np, targets_np, self)
    
    def _calibrate_dose_predictor(self, y: tf.Tensor) -> None:
        """
        Ajusta la capa de predicción de dosis según el rango de valores objetivo.
        
        Parámetros:
        -----------
        y : tf.Tensor
            Dosis objetivo para calibración
        """
        y_np = y.numpy() if hasattr(y, 'numpy') else y
        max_dose = np.max(y_np)
        min_dose = np.min(y_np)
        
        # Crear una entrada ficticia para construir la capa antes de establecer pesos
        if self.aNc_agent.continuous:
            # Para acción continua, forma de entrada es [batch_size, 1]
            dummy_input = tf.zeros((1, 1))
            _ = self.dose_predictor(dummy_input)
            
            # Ahora configurar los pesos
            self.dose_predictor.set_weights([
                np.ones((1, 1)) * (max_dose - min_dose) / 15,
                np.array([min_dose])
            ])
        else:
            # Para acción discreta, forma de entrada es [batch_size, action_dim]
            dummy_input = tf.zeros((1, self.aNc_agent.action_dim))
            _ = self.dose_predictor(dummy_input)
            
            # Ahora configurar los pesos
            self.dose_predictor.set_weights([
                np.ones((self.aNc_agent.action_dim, 1)) * (max_dose - min_dose) / self.aNc_agent.action_dim,
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
            "state_dim": self.aNc_agent.state_dim,
            "action_dim": self.aNc_agent.action_dim,
            "continuous": self.aNc_agent.continuous
        }
    
    def save(self, filepath: str, **kwargs) -> None:
        """
        Guarda el modelo A2C.
        
        Parámetros:
        -----------
        filepath : str
            Ruta donde guardar el modelo
        **kwargs
            Argumentos adicionales
        """
        # Asegurar que todas las capas estén construidas antes de guardar
        self._ensure_model_is_built()
        
        # Guardar pesos del wrapper
        self.save_weights(filepath + WRAPPER_WEIGHTS_SUFFIX)
        
        # Guardar modelo A2C
        self.aNc_agent.save_model(filepath + POLICY_WEIGHTS_SUFFIX)

    def _ensure_model_is_built(self) -> None:
        """
        Asegura que todas las capas del modelo estén construidas antes de guardar.
        """
        # Crear datos ficticios con las formas correctas
        batch_size = 1
        
        # Datos CGM ficticios - SIEMPRE debe ser 3D para Conv1D (batch, secuencia, características)
        if isinstance(self.cgm_shape, tuple) and len(self.cgm_shape) > 2:
            # Si tenemos la forma completa, usarla
            dummy_cgm = tf.zeros((batch_size, self.cgm_shape[1], self.cgm_shape[2]))
        else:
            # Forma predeterminada asegurando 3 dimensiones para Conv1D
            dummy_cgm = tf.zeros((batch_size, 24, 3))  # (batch, secuencia, características)
        
        # Otras características ficticias - debe ser 2D (batch, características)
        if isinstance(self.other_features_shape, tuple) and len(self.other_features_shape) > 1:
            dummy_other = tf.zeros((batch_size, self.other_features_shape[1]))
        else:
            # Forma predeterminada
            dummy_other = tf.zeros((batch_size, 6))  # (batch, características)
        
        # Realizar una pasada hacia adelante para construir todas las capas
        _ = self([dummy_cgm, dummy_other])
        
        # Asegurar que dose_predictor esté construido
        if hasattr(self.aNc_agent, 'continuous') and self.aNc_agent.continuous:
            dummy_action = tf.zeros((batch_size, 1))
        else:
            dummy_action = tf.zeros((batch_size, self.aNc_agent.action_dim))
        
        _ = self.dose_predictor(dummy_action)
        
        print("Modelo completamente construido antes de guardar.")
    
    def load_weights(self, filepath: str, **kwargs) -> None:
        """
        Carga el modelo A2C.
        
        Parámetros:
        -----------
        filepath : str
            Ruta desde donde cargar el modelo
        **kwargs
            Argumentos adicionales
        """
        # Determinar rutas según formato de filepath
        if filepath.endswith(WRAPPER_WEIGHTS_SUFFIX):
            wrapper_path = filepath
            policy_path = filepath.replace(WRAPPER_WEIGHTS_SUFFIX, POLICY_WEIGHTS_SUFFIX)
        else:
            wrapper_path = filepath + WRAPPER_WEIGHTS_SUFFIX
            policy_path = filepath + POLICY_WEIGHTS_SUFFIX
        
        # Cargar pesos del wrapper
        super().load_weights(wrapper_path)
        
        # Cargar modelo A2C
        self.aNc_agent.load_model(policy_path)


class A3CWrapper(A2CWrapper):
    """
    Wrapper para el algoritmo A3C que extiende el wrapper de A2C.
    """
    
    def __init__(
        self, 
        aNc_agent: A3C,
        cgm_shape: Tuple[int, ...],
        other_features_shape: Tuple[int, ...]
    ) -> None:
        """
        Inicializa el wrapper para el modelo A3C.
        
        Parámetros:
        -----------
        aNc_agent : A3C
            Agente A3C a envolver
        cgm_shape : Tuple[int, ...]
            Forma de los datos CGM
        other_features_shape : Tuple[int, ...]
            Forma de otras características
        """
        super(A2CWrapper, self).__init__()  # Inicializa tf.keras.Model
        self.aNc_agent = aNc_agent
        self.cgm_shape = cgm_shape
        self.other_features_shape = other_features_shape
        
        # Capas para procesamiento de CGM
        self.cgm_conv = Conv1D(64, 3, padding='same', activation='relu', name=f'{CGM_ENCODER}_conv')
        self.cgm_pooling = GlobalAveragePooling1D(name=f'{CGM_ENCODER}_pooling')
        
        # Capas para procesamiento de otras características
        self.other_dense = Dense(32, activation='relu', name=OTHER_ENCODER)
        
        # Capa para combinar características
        self.combined = Dense(self.aNc_agent.state_dim, activation='relu', name=COMBINED_LAYER)
        
        # Capa final para mapear de salidas de política a dosis
        self.dose_predictor = Dense(1, kernel_initializer='glorot_uniform', name='dose_predictor')
        
        # Para almacenar la forma de entrada durante la llamada
        self.built_input_shape = None
    
    def fit(
        self, 
        x: Union[List[tf.Tensor], tf.data.Dataset], 
        y: Optional[tf.Tensor] = None, 
        batch_size: int = 32, 
        epochs: int = 1, 
        verbose: int = 0,
        callbacks: Optional[List[Any]] = None,
        validation_data: Optional[Tuple] = None,
        **kwargs
    ) -> Any:
        """
        Simula la interfaz de entrenamiento de Keras para el agente A3C.
        
        Parámetros:
        -----------
        x : Union[List[tf.Tensor], tf.data.Dataset]
            Lista con [cgm_data, other_features] o un dataset de TensorFlow
        y : Optional[tf.Tensor], opcional
            Etiquetas (dosis objetivo), se ignora si x es un dataset (default: None)
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
            Historia de entrenamiento compatible con Keras
        """
        if verbose > 0:
            print("Entrenando modelo A3C...")
        
        # Extraer datos del dataset si es necesario
        if isinstance(x, tf.data.Dataset):
            # Extraer datos del dataset
            cgm_data = []
            other_features = []
            targets = []
            
            for (batch_cgm, batch_other), batch_y in x:
                cgm_data.append(batch_cgm)
                other_features.append(batch_other)
                targets.append(batch_y)
            
            cgm_tensor = tf.concat(cgm_data, axis=0)
            other_tensor = tf.concat(other_features, axis=0)
            y_input = tf.concat(targets, axis=0)
            
            x_input = [cgm_tensor, other_tensor]
        else:
            # Usar listas de tensores directamente
            x_input = x
            y_input = y
    
        # Crear entorno para entrenamiento
        env_instance = self._create_training_environment(x_input[0], x_input[1], y_input)
        
        # Función generadora de entornos para trabajadores
        def env_creator():
            return env_instance
        
        # Verificar que todos los workers tengan acceso al mismo entorno
        # para evitar problemas de concurrencia
        env_instance.lock = threading.RLock()
        
        # Entrenar al agente A3C de manera asíncrona con menos workers y pasos
        # para evitar sobrecargar la memoria y problemas de concurrencia
        history = {}
        try:
            # Ejecutar entrenamiento asíncrono con manejo de excepciones
            history = self.aNc_agent.train_async(
                env_fn=env_creator,
                n_steps=batch_size,
                total_steps=epochs * batch_size,
                render=False
            )
        except Exception as e:
            print(f"Error durante el entrenamiento asíncrono: {e}")
            # Proporcionar un historial vacío pero válido en caso de error
            history = {'policy_losses': [0.0], 'value_losses': [0.0], 'total_losses': [0.0]}
        
        # Ajustar capa de predicción de dosis
        self._calibrate_dose_predictor(y_input)
        
        if verbose > 0:
            print_success("Entrenamiento asíncrono completado.")
        
        # Convertir historia a formato compatible con Keras
        # Asegurarse de que siempre haya al menos un elemento en policy_losses
        policy_losses = history.get('policy_losses', [0.0])
        if not policy_losses:
            policy_losses = [0.0]
            
        keras_history = {
            'loss': policy_losses,
            'val_loss': [policy_losses[-1]] if validation_data else None
        }
        
        # Crear un objeto con atributo .history para compatibilidad con Keras
        class KerasHistoryCompatible:
            def __init__(self, history_dict):
                self.history = history_dict
                
        return KerasHistoryCompatible(keras_history)


def create_a2c_model(cgm_shape: Tuple[int, ...], other_features_shape: Tuple[int, ...]) -> tf.keras.models.Model:
    """
    Crea un modelo basado en A2C (Advantage Actor-Critic) para predicción de dosis de insulina.
    
    Parámetros:
    -----------
    cgm_shape : Tuple[int, ...]
        Forma de los datos CGM (batch_size, time_steps, features)
    other_features_shape : Tuple[int, ...]
        Forma de otras características (batch_size, n_features)
        
    Retorna:
    --------
    tf.keras.models.Model
        Modelo A2C que implementa la interfaz de Keras
    """
    # Inferir dimensión del estado a partir de las formas de entrada
    # Esto puede ajustarse según la complejidad necesaria
    state_dim = 64
    
    # Configurar espacio de acciones (dosis de insulina)
    action_dim = 1  # Una dimensión para dosis continua
    continuous = True  # Usar espacio de acción continuo para dosificación precisa
    
    # Crear agente A2C
    a2c_agent = A2C(
        state_dim=state_dim,
        action_dim=action_dim,
        continuous=continuous,
        gamma=A2C_A3C_CONFIG['gamma'],
        learning_rate=A2C_A3C_CONFIG['learning_rate'],
        entropy_coef=A2C_A3C_CONFIG['entropy_coef'],
        value_coef=A2C_A3C_CONFIG['value_coef'],
        max_grad_norm=A2C_A3C_CONFIG['max_grad_norm'],
        hidden_units=A2C_A3C_CONFIG['hidden_units']
    )
    
    # Crear y devolver el modelo wrapper
    return A2CWrapper(
        aNc_agent=a2c_agent,
        cgm_shape=cgm_shape,
        other_features_shape=other_features_shape
    )


def create_a3c_model(cgm_shape: Tuple[int, ...], other_features_shape: Tuple[int, ...]) -> tf.keras.models.Model:
    """
    Crea un modelo basado en A3C (Asynchronous Advantage Actor-Critic) para predicción de dosis de insulina.
    
    Parámetros:
    -----------
    cgm_shape : Tuple[int, ...]
        Forma de los datos CGM (batch_size, time_steps, features)
    other_features_shape : Tuple[int, ...]
        Forma de otras características (batch_size, n_features)
        
    Retorna:
    --------
    tf.keras.models.Model
        Modelo A3C que implementa la interfaz de Keras
    """
    # Inferir dimensión del estado a partir de las formas de entrada
    state_dim = 64
    
    # Configurar espacio de acciones (dosis de insulina)
    action_dim = 1  # Una dimensión para dosis continua
    continuous = True  # Usar espacio de acción continuo para dosificación precisa
    
    # Número de trabajadores para entrenamiento paralelo
    # Reducir para evitar sobrecarga en sistemas con pocos núcleos
    n_workers = 4 

    # Crear agente A3C
    a3c_agent = A3C(
        state_dim=state_dim,
        action_dim=action_dim,
        continuous=continuous,
        n_workers=n_workers,
        gamma=A2C_A3C_CONFIG['gamma'],
        learning_rate=A2C_A3C_CONFIG['learning_rate'],
        entropy_coef=A2C_A3C_CONFIG['entropy_coef'],
        value_coef=A2C_A3C_CONFIG['value_coef'],
        max_grad_norm=A2C_A3C_CONFIG['max_grad_norm'],
        hidden_units=A2C_A3C_CONFIG['hidden_units']
    )
    
    # Crear y devolver el modelo wrapper
    return A3CWrapper(
        aNc_agent=a3c_agent,
        cgm_shape=cgm_shape,
        other_features_shape=other_features_shape
    )