import os, sys
import jax
import jax.numpy as jnp
import numpy as np
import flax
import flax.linen as nn
import optax
from flax.training import train_state
from typing import Dict, List, Tuple, Any, Optional, Union, Callable, Sequence
from functools import partial
import matplotlib.pyplot as plt

PROJECT_ROOT = os.path.abspath(os.getcwd())
sys.path.append(PROJECT_ROOT) 

from config.models_config import PPO_CONFIG


class ActorCriticModel(nn.Module):
    """
    Modelo Actor-Crítico para PPO que divide la arquitectura en redes para
    política (actor) y valor (crítico).
    
    Parámetros:
    -----------
    action_dim : int
        Dimensión del espacio de acciones
    hidden_units : Optional[Sequence[int]], opcional
        Unidades en capas ocultas (default: None)
    """
    action_dim: int
    hidden_units: Optional[Sequence[int]] = None
    
    @nn.compact
    def __call__(self, x: jnp.ndarray, training: bool = False) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        """
        Realiza el forward pass del modelo actor-crítico.
        
        Parámetros:
        -----------
        x : jnp.ndarray
            Tensor de estados de entrada
        training : bool, opcional
            Si está en modo entrenamiento (default: False)
            
        Retorna:
        --------
        Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]
            (mu, sigma, value) - Parámetros de la distribución de política y valor estimado
        """
        hidden_units = self.hidden_units or PPO_CONFIG['hidden_units']
        
        # Capas compartidas para procesamiento de estados
        for i, units in enumerate(hidden_units[:2]):
            x = nn.Dense(units, name=f'shared_dense_{i}')(x)
            x = nn.tanh(x)
            x = nn.LayerNorm(epsilon=PPO_CONFIG['epsilon'], name=f'shared_ln_{i}')(x)
            x = nn.Dropout(rate=PPO_CONFIG['dropout_rate'], deterministic=not training, name=f'shared_dropout_{i}')(x)
        
        # Red del Actor (política)
        actor_x = x
        for i, units in enumerate(hidden_units[2:]):
            actor_x = nn.Dense(units, name=f'actor_dense_{i}')(actor_x)
            actor_x = nn.tanh(actor_x)
            actor_x = nn.LayerNorm(epsilon=PPO_CONFIG['epsilon'], name=f'actor_ln_{i}')(actor_x)
        
        # Capa de salida del actor (mu y sigma para política gaussiana)
        mu = nn.Dense(self.action_dim, name='actor_mu')(actor_x)
        log_sigma = nn.Dense(self.action_dim, name='actor_log_sigma')(actor_x)
        sigma = jnp.exp(log_sigma)
        
        # Red del Crítico (valor)
        critic_x = x
        for i, units in enumerate(hidden_units[2:]):
            critic_x = nn.Dense(units, name=f'critic_dense_{i}')(critic_x)
            critic_x = nn.tanh(critic_x)
            critic_x = nn.LayerNorm(epsilon=PPO_CONFIG['epsilon'], name=f'critic_ln_{i}')(critic_x)
        
        # Capa de salida del crítico (valor del estado)
        value = nn.Dense(1, name='critic_value')(critic_x)
        
        return mu, sigma, value


class PPOTrainState(train_state.TrainState):
    """
    Estado de entrenamiento para PPO que extiende el TrainState de Flax.
    
    Atributos adicionales:
    --------------------
    apply_fn : Callable
        Función del modelo para inferencia
    key : jnp.ndarray
        Llave PRNG para generación de números aleatorios
    """
    key: jnp.ndarray
    

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
        seed: int = 42
    ) -> None:
        # Configurar semillas para reproducibilidad
        key = jax.random.PRNGKey(seed)
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
        
        # Crear modelo
        self.model = ActorCriticModel(action_dim=action_dim, hidden_units=self.hidden_units)
        
        # Inicializar parámetros
        key, init_key = jax.random.split(key)
        dummy_state = jnp.ones((1, state_dim))
        params = self.model.init(init_key, dummy_state)
        
        # Crear optimizador con clipping de gradientes opcional
        if max_grad_norm is not None:
            tx = optax.chain(
                optax.clip_by_global_norm(max_grad_norm),
                optax.adam(learning_rate=learning_rate)
            )
        else:
            tx = optax.adam(learning_rate=learning_rate)
        
        # Inicializar estado de entrenamiento
        self.state = PPOTrainState.create(
            apply_fn=self.model.apply,
            params=params,
            tx=tx,
            key=key
        )
        
        # Métricas acumuladas
        self.total_loss_sum = 0.0
        self.policy_loss_sum = 0.0
        self.value_loss_sum = 0.0
        self.entropy_sum = 0.0
        self.count = 0
        
        # Compilar funciones con jit para acelerar
        self.get_action_and_value = jax.jit(self._get_action_and_value)
        self.get_action = jax.jit(self._get_action)
        self.get_value = jax.jit(self._get_value)
        self.train_step = jax.jit(self._train_step)
    
    def _get_action_and_value(self, params: flax.core.FrozenDict, state: jnp.ndarray, 
                            key: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        """
        Obtiene acción, log_prob y valor para un estado dado.
        
        Parámetros:
        -----------
        params : flax.core.FrozenDict
            Parámetros del modelo
        state : jnp.ndarray
            Estado actual
        key : jnp.ndarray
            Llave PRNG para muestreo aleatorio
            
        Retorna:
        --------
        Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]
            (acción, log_prob, valor, nueva_llave)
        """
        mu, sigma, value = self.model.apply(params, state)
        key, subkey = jax.random.split(key)
        
        # Muestrear de la distribución normal
        noise = jax.random.normal(subkey, mu.shape)
        action = mu + sigma * noise
        
        # Calcular log prob
        log_prob = self._log_prob(mu, sigma, action)
        
        return action, log_prob, value, key
    
    def _get_action(self, params: flax.core.FrozenDict, state: jnp.ndarray, 
                  key: jnp.ndarray, deterministic: bool = False) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """
        Obtiene una acción basada en el estado actual.
        
        Parámetros:
        -----------
        params : flax.core.FrozenDict
            Parámetros del modelo
        state : jnp.ndarray
            Estado actual
        key : jnp.ndarray
            Llave PRNG para muestreo aleatorio
        deterministic : bool, opcional
            Si es True, devuelve la acción con máxima probabilidad (default: False)
            
        Retorna:
        --------
        Tuple[jnp.ndarray, jnp.ndarray]
            (acción, nueva_llave)
        """
        mu, sigma, _ = self.model.apply(params, state)
        
        if deterministic:
            return mu, key
        
        # Muestrear de la distribución normal
        key, subkey = jax.random.split(key)
        noise = jax.random.normal(subkey, mu.shape)
        action = mu + sigma * noise
        
        return action, key
    
    def _get_value(self, params: flax.core.FrozenDict, state: jnp.ndarray) -> jnp.ndarray:
        """
        Obtiene el valor estimado para un estado.
        
        Parámetros:
        -----------
        params : flax.core.FrozenDict
            Parámetros del modelo
        state : jnp.ndarray
            Estado para evaluar
            
        Retorna:
        --------
        jnp.ndarray
            El valor estimado del estado
        """
        _, _, value = self.model.apply(params, state)
        return value
    
    def _log_prob(self, mu: jnp.ndarray, sigma: jnp.ndarray, actions: jnp.ndarray) -> jnp.ndarray:
        """
        Calcula el logaritmo de la probabilidad de acciones bajo una política gaussiana.
        
        Parámetros:
        -----------
        mu : jnp.ndarray
            Media de la distribución gaussiana
        sigma : jnp.ndarray
            Desviación estándar de la distribución gaussiana
        actions : jnp.ndarray
            Acciones para calcular su probabilidad
            
        Retorna:
        --------
        jnp.ndarray
            Logaritmo de probabilidad de las acciones
        """
        logp_normal = -0.5 * ((actions - mu) / sigma) ** 2 - 0.5 * jnp.log(2.0 * jnp.pi) - jnp.log(sigma)
        return jnp.sum(logp_normal, axis=-1, keepdims=True)
    
    def _compute_gae(self, rewards: np.ndarray, values: np.ndarray, next_values: np.ndarray, 
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
    
    def _train_step(self, state: PPOTrainState, states: jnp.ndarray, actions: jnp.ndarray, 
                  old_log_probs: jnp.ndarray, returns: jnp.ndarray, advantages: jnp.ndarray) -> Tuple[PPOTrainState, Dict[str, jnp.ndarray]]:
        """
        Realiza un paso de entrenamiento para actualizar el modelo.
        
        Parámetros:
        -----------
        state : ppo_train_state
            Estado actual del modelo
        states : jnp.ndarray
            Estados observados en el entorno
        actions : jnp.ndarray
            Acciones tomadas para esos estados
        old_log_probs : jnp.ndarray
            Log de probabilidades de acciones bajo la política antigua
        returns : jnp.ndarray
            Retornos estimados
        advantages : jnp.ndarray
            Ventajas estimadas
            
        Retorna:
        --------
        Tuple[ppo_train_state, Dict[str, jnp.ndarray]]
            (nuevo_estado, métricas)
        """
        def loss_fn(params):
            # Pasar estados por el modelo
            mu, sigma, values = self.model.apply(params, states, training=True)
            
            # Calcular nueva probabilidad de acciones
            new_log_probs = self._log_prob(mu, sigma, actions)
            
            # Ratio entre nuevas y antiguas probabilidades
            ratio = jnp.exp(new_log_probs - old_log_probs)
            
            # Términos del clipping de PPO
            p1 = ratio * advantages
            p2 = jnp.clip(ratio, 1 - self.epsilon, 1 + self.epsilon) * advantages
            
            # Pérdida de política (negativa porque queremos maximizar)
            policy_loss = -jnp.mean(jnp.minimum(p1, p2))
            
            # Pérdida de valor (predicción vs retorno real)
            value_loss = jnp.mean(jnp.square(returns - values[:, 0]))
            
            # Término de entropía para fomentar la exploración
            entropy = jnp.mean(
                jnp.sum(
                    0.5 * jnp.log(2.0 * jnp.pi * jnp.square(sigma)) + 0.5,
                    axis=-1
                )
            )
            
            # Pérdida total combinada
            total_loss = policy_loss + self.value_coef * value_loss - self.entropy_coef * entropy
            
            return total_loss, (policy_loss, value_loss, entropy)
        
        # Calcular pérdida y gradientes
        grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
        (total_loss, (policy_loss, value_loss, entropy)), grads = grad_fn(state.params)
        
        # Actualizar parámetros
        new_state = state.apply_gradients(grads=grads)
        
        # Recopilar métricas
        metrics = {
            'total_loss': total_loss,
            'policy_loss': policy_loss,
            'value_loss': value_loss,
            'entropy': entropy
        }
        
        return new_state, metrics
    
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
        state = jnp.asarray(state)[None, :]  # Add batch dimension
        action, key = self._get_action(self.state.params, state, self.state.key, deterministic)
        # Update key
        self.state = self.state.replace(key=key)
        return np.asarray(action[0])
    
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
        state = jnp.asarray(state)[None, :]  # Add batch dimension
        value = self._get_value(self.state.params, state)
        return float(value[0][0])
    
    def _collect_trajectories(self, env: Any, steps_per_epoch: int) -> Tuple[Dict[str, np.ndarray], Dict[str, List[float]]]:
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
        Tuple[Dict[str, np.ndarray], Dict[str, List[float]]]
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
            
            # Obtener acción y valor usando JAX
            state_tensor = jnp.asarray(state)[None, :]
            action, key = self._get_action(self.state.params, state_tensor, self.state.key)
            self.state = self.state.replace(key=key)
            action = np.asarray(action[0])
            
            # Obtener valor y log_prob
            mu, sigma, value = self.model.apply(self.state.params, state_tensor)
            log_prob = self._log_prob(mu, sigma, jnp.asarray(action)[None, :])
            
            # Dar paso en el entorno
            next_state, reward, done, _, _ = env.step(action)
            
            # Guardar experiencia
            actions.append(action)
            rewards.append(reward)
            values.append(float(value[0][0]))
            dones.append(float(done))
            log_probs.append(float(log_prob[0][0]))
            
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
            next_state_tensor = jnp.asarray(next_state)[None, :]
            next_value = self._get_value(self.state.params, next_state_tensor)
            next_values.append(float(next_value[0][0]))
        
        # Si el último episodio no terminó, guardar sus métricas parciales
        if episode_length > 0:
            episode_rewards.append(episode_reward)
            episode_lengths.append(episode_length)
            
        # Empaquetar datos
        trajectory_data = {
            'states': np.array(states, dtype=np.float32),
            'actions': np.array(actions, dtype=np.float32),
            'rewards': np.array(rewards, dtype=np.float32),
            'values': np.array(values, dtype=np.float32),
            'dones': np.array(dones, dtype=np.float32),
            'next_values': np.array(next_values, dtype=np.float32),
            'log_probs': np.array(log_probs, dtype=np.float32).reshape(-1, 1)
        }
        
        episode_history = {
            'reward': episode_rewards,
            'length': episode_lengths
        }
        
        return trajectory_data, episode_history
    
    def _update_policy(self, data: Dict[str, np.ndarray], batch_size: int, 
                     update_iters: int) -> Dict[str, float]:
        """
        Actualiza la política y el crítico con los datos recolectados.
        
        Parámetros:
        -----------
        data : Dict[str, np.ndarray]
            Datos de trayectorias
        batch_size : int
            Tamaño de lote para actualización
        update_iters : int
            Número de iteraciones de actualización
            
        Retorna:
        --------
        Dict[str, float]
            Métricas promedio de entrenamiento
        """
        states = data['states']
        actions = data['actions']
        log_probs = data['log_probs']
        returns = data['returns']
        advantages = data['advantages']
        
        # Convertir a jnp arrays
        states = jnp.asarray(states)
        actions = jnp.asarray(actions)
        old_log_probs = jnp.asarray(log_probs)
        returns = jnp.asarray(returns).reshape(-1, 1)
        advantages = jnp.asarray(advantages).reshape(-1, 1)
        
        # Resetear métricas acumuladas
        self.total_loss_sum = 0.0
        self.policy_loss_sum = 0.0
        self.value_loss_sum = 0.0
        self.entropy_sum = 0.0
        self.count = 0
        
        # Crear índices para batching
        indices = np.arange(len(states))
        
        # Crear una instancia de Generator para operaciones aleatorias con semilla para reproducibilidad
        rng = np.random.default_rng(self.seed)
        
        # Entrenar durante varias iteraciones
        for _ in range(update_iters):
            # Shuffle indices
            rng.shuffle(indices)
            
            # Actualizar en mini-lotes
            for start in range(0, len(states), batch_size):
                end = start + batch_size
                if end > len(states):
                    end = len(states)
                
                mb_indices = indices[start:end]
                mb_states = states[mb_indices]
                mb_actions = actions[mb_indices]
                mb_old_log_probs = old_log_probs[mb_indices]
                mb_returns = returns[mb_indices]
                mb_advantages = advantages[mb_indices]
                
                # Actualizar modelo
                self.state, metrics = self.train_step(
                    self.state, mb_states, mb_actions, 
                    mb_old_log_probs, mb_returns, mb_advantages)
                
                # Acumular métricas
                self.total_loss_sum += float(metrics['total_loss'])
                self.policy_loss_sum += float(metrics['policy_loss'])
                self.value_loss_sum += float(metrics['value_loss'])
                self.entropy_sum += float(metrics['entropy'])
                self.count += 1
        
        # Calcular promedios
        metrics = {
            'total_loss': self.total_loss_sum / self.count if self.count > 0 else 0.0,
            'policy_loss': self.policy_loss_sum / self.count if self.count > 0 else 0.0,
            'value_loss': self.value_loss_sum / self.count if self.count > 0 else 0.0,
            'entropy': self.entropy_sum / self.count if self.count > 0 else 0.0
        }
        
        return metrics
    
    def train(self, env: Any, epochs: int = 100, steps_per_epoch: int = 4000, batch_size: int = 64, 
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
            
            # Registrar recompensas de episodios
            history['reward'].extend(episode_history['reward'])
            history['avg_reward'].append(np.mean(episode_history['reward']))
            
            # 2. Calcular ventajas y retornos usando GAE
            advantages, returns = self._compute_gae(
                trajectory_data['rewards'], 
                trajectory_data['values'], 
                trajectory_data['next_values'], 
                trajectory_data['dones'], 
                self.gamma, 
                gae_lambda
            )
            
            # Añadir a los datos de trayectoria
            trajectory_data['advantages'] = advantages
            trajectory_data['returns'] = returns
            
            # 3. Actualizar política
            metrics = self._update_policy(trajectory_data, batch_size, update_iters)
            
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
                action = self.get_action(state, deterministic=deterministic)
                
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
        with open(filepath, 'wb') as f:
            f.write(flax.serialization.to_bytes(self.state.params))
        print(f"Modelo guardado en {filepath}")
        
    def load_model(self, filepath: str) -> None:
        """
        Carga el modelo desde un archivo.
        
        Parámetros:
        -----------
        filepath : str
            Ruta desde donde cargar el modelo
        """
        with open(filepath, 'rb') as f:
            params = flax.serialization.from_bytes(self.state.params, f.read())
            self.state = self.state.replace(params=params)
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
        def smooth(y, window_size):
            """Aplica suavizado con media móvil."""
            box = np.ones(window_size) / window_size
            return np.convolve(y, box, mode='valid')
        
        # Definir constantes para etiquetas repetidas
        LABEL_EPOCA = 'Época'
        
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
        plt.xlabel('Episodio')
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

class PPOWrapper:
    """
    Wrapper para hacer que el agente PPO sea compatible con la interfaz de modelos de aprendizaje profundo.
    """
    
    def __init__(
        self, 
        ppo_agent: PPO, 
        cgm_shape: Tuple[int, ...], 
        other_features_shape: Tuple[int, ...],
    ) -> None:
        """
        Inicializa el wrapper para PPO.
        
        Parámetros:
        -----------
        ppo_agent : PPO
            Agente PPO a utilizar
        cgm_shape : Tuple[int, ...]
            Forma de entrada para datos CGM
        other_features_shape : Tuple[int, ...]
            Forma de entrada para otras características
        """
        self.ppo_agent = ppo_agent
        self.cgm_shape = cgm_shape
        self.other_features_shape = other_features_shape
        
        # Inicializar clave para generación de números aleatorios
        self.key = jax.random.PRNGKey(42)
        self.key, self.encoder_key = jax.random.split(self.key)
        
        # Configurar funciones de codificación para entradas
        self._setup_encoders()
        
        # Historial de entrenamiento
        self.history = {
            'loss': [],
            'val_loss': [],
            'policy_loss': [],
            'value_loss': [],
            'entropy': [],
            'episode_rewards': []
        }
    
    def _setup_encoders(self) -> None:
        """
        Configura las funciones de codificación para procesar las entradas.
        """
        # En un modelo real, esto inicializaría redes neuronales para codificar entradas
        # Simplificado para este ejemplo, usando transformaciones lineales
        
        # Calcular dimensiones de características aplanadas
        cgm_dim = np.prod(self.cgm_shape[1:])
        other_dim = np.prod(self.other_features_shape[1:])
        
        # Inicializar matrices de transformación
        self.key, key_cgm, key_other = jax.random.split(self.key, 3)
        
        # Crear matrices de proyección para entradas
        self.cgm_weight = jax.random.normal(key_cgm, (cgm_dim, self.ppo_agent.state_dim // 2))
        self.other_weight = jax.random.normal(key_other, (other_dim, self.ppo_agent.state_dim // 2))
        
        # JIT-compilar transformaciones para mayor rendimiento
        self.encode_cgm = jax.jit(self._create_encoder_fn(self.cgm_weight))
        self.encode_other = jax.jit(self._create_encoder_fn(self.other_weight))
    
    def _create_encoder_fn(self, weights: jnp.ndarray) -> Callable:
        """
        Crea una función de codificación.
        
        Parámetros:
        -----------
        weights : jnp.ndarray
            Matriz de pesos para la transformación
            
        Retorna:
        --------
        Callable
            Función de codificación JIT-compilada
        """
        def encoder_fn(x):
            x_flat = x.reshape((x.shape[0], -1))
            return jnp.tanh(jnp.dot(x_flat, weights))
        return encoder_fn
    
    def __call__(self, inputs: List[jnp.ndarray]) -> jnp.ndarray:
        """
        Implementa la interfaz de llamada para predicción.
        
        Parámetros:
        -----------
        inputs : List[jnp.ndarray]
            Lista con [cgm_data, other_features]
            
        Retorna:
        --------
        jnp.ndarray
            Predicciones de dosis de insulina
        """
        return self.predict(inputs)
    
    def predict(self, inputs: List[jnp.ndarray]) -> jnp.ndarray:
        """
        Realiza predicciones con el modelo PPO.
        
        Parámetros:
        -----------
        inputs : List[jnp.ndarray]
            Lista con [cgm_data, other_features]
            
        Retorna:
        --------
        jnp.ndarray
            Predicciones de dosis de insulina
        """
        # Obtener entradas
        cgm_data, other_features = inputs
        
        # Convertir a arrays de JAX si no lo son
        cgm_data = jnp.array(cgm_data)
        other_features = jnp.array(other_features)
        
        # Codificar entradas a estado
        cgm_encoded = self.encode_cgm(cgm_data)
        other_encoded = self.encode_other(other_features)
        states = jnp.concatenate([cgm_encoded, other_encoded], axis=1)
        
        # Obtener acciones usando el agente PPO (modo determinístico para predicciones)
        batch_size = states.shape[0]
        actions = np.zeros((batch_size, 1))
        
        for i in range(batch_size):
            state = np.array(states[i])
            # Usar la política del agente PPO para obtener acción determinística
            action = self.ppo_agent.get_action(state, deterministic=True)
            # Escalar acción al rango de dosis (asumiendo que PPO usa [-1, 1])
            action_scaled = (action + 1.0) / 2.0 * 15.0  # Escalar de [-1,1] a [0,15]
            actions[i] = action_scaled
            
        return actions
    
    def fit(
        self, 
        x: List[jnp.ndarray], 
        y: jnp.ndarray, 
        validation_data: Optional[Tuple] = None, 
        epochs: int = 1,
        batch_size: int = 32,
        callbacks: List = None,
        verbose: int = 0
    ) -> Dict:
        """
        Entrena el modelo PPO en los datos proporcionados.
        
        Parámetros:
        -----------
        x : List[jnp.ndarray]
            Lista con [cgm_data, other_features]
        y : jnp.ndarray
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
        if verbose > 0:
            print("Entrenando modelo PPO...")
            
        # Crear entorno simulado para RL a partir de los datos
        env = self._create_training_environment(x[0], x[1], y)
        
        # Entrenar el agente PPO
        ppo_history = self.ppo_agent.train(
            env=env,
            epochs=epochs,
            steps_per_epoch=batch_size,
            batch_size=min(32, batch_size),
            log_interval=max(1, epochs // 10) if verbose > 0 else epochs + 1
        )
        
        # Actualizar historial con métricas del entrenamiento
        self.history['policy_loss'].extend(ppo_history.get('policy_loss', [0.0]))
        self.history['value_loss'].extend(ppo_history.get('value_loss', [0.0]))
        self.history['entropy'].extend(ppo_history.get('entropy', [0.0]))
        self.history['episode_rewards'].extend(ppo_history.get('reward', [0.0]))
        
        # Calcular pérdida en los datos de entrenamiento
        train_preds = self.predict(x)
        train_loss = float(jnp.mean((train_preds.flatten() - y) ** 2))
        self.history['loss'].append(train_loss)
        
        # Evaluar en datos de validación si se proporcionan
        if validation_data:
            val_x, val_y = validation_data
            val_preds = self.predict(val_x)
            val_loss = float(jnp.mean((val_preds.flatten() - val_y) ** 2))
            self.history['val_loss'].append(val_loss)
        
        if verbose > 0:
            print(f"Entrenamiento completado. Pérdida final: {train_loss:.4f}")
            if validation_data:
                print(f"Pérdida de validación: {val_loss:.4f}")
        
        return self.history
    
    def _create_training_environment(
        self, 
        cgm_data: jnp.ndarray, 
        other_features: jnp.ndarray, 
        targets: jnp.ndarray
    ) -> Any:
        """
        Crea un entorno de entrenamiento para RL a partir de los datos.
        
        Parámetros:
        -----------
        cgm_data : jnp.ndarray
            Datos CGM
        other_features : jnp.ndarray
            Otras características
        targets : jnp.ndarray
            Dosis objetivo
            
        Retorna:
        --------
        Any
            Entorno simulado para RL
        """
        # Crear entorno personalizado para PPO
        class InsulinDosingEnv:
            def __init__(self, cgm, features, targets, model_wrapper):
                self.cgm = np.array(cgm)
                self.features = np.array(features)
                self.targets = np.array(targets)
                self.model = model_wrapper
                self.rng = np.random.Generator(np.random.PCG64(42))
                self.current_idx = 0
                self.max_idx = len(targets) - 1
                
                # Para compatibilidad con algoritmos RL
                self.observation_space = SimpleNamespace(
                    shape=(model_wrapper.ppo_agent.state_dim,),
                    low=np.full((model_wrapper.ppo_agent.state_dim,), -10.0),
                    high=np.full((model_wrapper.ppo_agent.state_dim,), 10.0)
                )
                
                self.action_space = SimpleNamespace(
                    shape=(1,),
                    low=np.array([-1.0]),
                    high=np.array([1.0]),
                    sample=self._sample_action
                )
            
            def _sample_action(self):
                """Muestrea una acción aleatoria del espacio continuo."""
                return self.rng.uniform(
                    self.action_space.low,
                    self.action_space.high
                )
            
            def reset(self):
                """Reinicia el entorno eligiendo un ejemplo aleatorio."""
                self.current_idx = self.rng.integers(0, self.max_idx)
                state = self._get_state()
                return state, {}
            
            def step(self, action):
                """Ejecuta un paso con la acción dada."""
                # Convertir acción normalizada a dosis de insulina
                action_scaled = (action[0] + 1.0) / 2.0 * 15.0  # Escalar de [-1,1] a [0,15]
                
                # Calcular recompensa como negativo del error absoluto
                target = self.targets[self.current_idx]
                reward = -abs(action_scaled - target)
                
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
                
                # Codificar a espacio de estado
                cgm_encoded = self.model.encode_cgm(jnp.array(cgm_batch))
                other_encoded = self.model.encode_other(jnp.array(features_batch))
                
                # Combinar características
                state = np.concatenate([cgm_encoded[0], other_encoded[0]])
                
                return state
        
        # Importar lo necesario para el entorno
        from types import SimpleNamespace
        
        # Crear y devolver el entorno
        return InsulinDosingEnv(cgm_data, other_features, targets, self)
    
    def save(self, filepath: str) -> None:
        """
        Guarda el modelo PPO en un archivo.
        
        Parámetros:
        -----------
        filepath : str
            Ruta donde guardar el modelo
        """
        # Crear directorio si no existe
        import os
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        # Guardar el agente PPO
        self.ppo_agent.save_model(f"{filepath}_ppo.h5")
        
        # Guardar datos adicionales del wrapper
        import pickle
        wrapper_data = {
            'cgm_shape': self.cgm_shape,
            'other_features_shape': self.other_features_shape,
            'cgm_weight': self.cgm_weight,
            'other_weight': self.other_weight,
            'state_dim': self.ppo_agent.state_dim
        }
        
        with open(f"{filepath}_wrapper.pkl", 'wb') as f:
            pickle.dump(wrapper_data, f)
        
        print(f"Modelo guardado en {filepath}")
    
    def load(self, filepath: str) -> None:
        """
        Carga el modelo PPO desde un archivo.
        
        Parámetros:
        -----------
        filepath : str
            Ruta desde donde cargar el modelo
        """
        # Cargar el agente PPO
        self.ppo_agent.load_model(f"{filepath}_ppo.h5")
        
        # Cargar datos adicionales del wrapper
        import pickle
        with open(f"{filepath}_wrapper.pkl", 'rb') as f:
            wrapper_data = pickle.load(f)
        
        self.cgm_shape = wrapper_data['cgm_shape']
        self.other_features_shape = wrapper_data['other_features_shape']
        self.cgm_weight = wrapper_data['cgm_weight']
        self.other_weight = wrapper_data['other_weight']
        
        # Recompilar funciones de codificación
        self.encode_cgm = jax.jit(self._create_encoder_fn(self.cgm_weight))
        self.encode_other = jax.jit(self._create_encoder_fn(self.other_weight))
        
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
            'state_dim': self.ppo_agent.state_dim,
            'hidden_units': self.ppo_agent.hidden_units,
            'gamma': self.ppo_agent.gamma,
            'epsilon': self.ppo_agent.epsilon
        }


def create_ppo_model(cgm_shape: Tuple[int, ...], other_features_shape: Tuple[int, ...]) -> PPOWrapper:
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
    PPOWrapper
        Wrapper de PPO que implementa la interfaz compatible con modelos de aprendizaje profundo
    """
    # Configurar el espacio de estados y acciones
    state_dim = 64  # Dimensión del espacio de estado latente
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
        seed=42
    )
    
    # Crear y devolver wrapper
    return PPOWrapper(
        ppo_agent=ppo_agent,
        cgm_shape=cgm_shape,
        other_features_shape=other_features_shape
    )