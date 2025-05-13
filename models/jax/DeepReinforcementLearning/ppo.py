import os, sys
from types import SimpleNamespace
import jax
import jax.numpy as jnp
import numpy as np
import flax
import flax.linen as nn
import optax
import tqdm
from flax.training import train_state
from typing import Dict, List, Tuple, Any, Optional, Union, Callable, Sequence
from functools import partial
import matplotlib.pyplot as plt

PROJECT_ROOT = os.path.abspath(os.getcwd())
sys.path.append(PROJECT_ROOT) 

from constants.constants import CONST_DEFAULT_SEED, CONST_DEFAULT_EPOCHS, CONST_DEFAULT_BATCH_SIZE
from config.models_config import PPO_CONFIG
from custom.DeepReinforcementLearning.drl_model_wrapper import DRLModelWrapper
from custom.printer import print_debug, print_log, print_success, print_error, print_info

# Constantes para uso repetido
CONST_RELU = "relu"
CONST_GELU = "gelu"
CONST_TANH = "tanh"
CONST_SELU = "selu"
CONST_SIGMOID = "sigmoid"
CONST_EPSILON = "epsilon"
CONST_GAMMA = 'gamma'
CONST_LAMBDA = 'lambda'
CONST_DROPOUT = "dropout_rate"
CONST_ENTROPIA = "Entropía"

FIGURES_DIR = os.path.join(PROJECT_ROOT, 'figures', 'jax', 'ppo')

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
            x = jnp.tanh(x)
            x = nn.LayerNorm(epsilon=PPO_CONFIG[CONST_EPSILON], name=f'shared_ln_{i}')(x)
            x = nn.Dropout(rate=PPO_CONFIG[CONST_DROPOUT], deterministic=not training, name=f'shared_dropout_{i}')(x)
        
        # Red del Actor (política)
        actor_x = x
        for i, units in enumerate(hidden_units[2:]):
            actor_x = nn.Dense(units, name=f'actor_dense_{i}')(actor_x)
            actor_x = jnp.tanh(actor_x)
            actor_x = nn.LayerNorm(epsilon=PPO_CONFIG[CONST_EPSILON], name=f'actor_ln_{i}')(actor_x)
        
        # Capa de salida del actor (mu y sigma para política gaussiana)
        mu = nn.Dense(self.action_dim, name='actor_mu')(actor_x)
        log_sigma = nn.Dense(self.action_dim, name='actor_log_sigma')(actor_x)
        sigma = jnp.exp(log_sigma)
        
        # Red del Crítico (valor)
        critic_x = x
        for i, units in enumerate(hidden_units[2:]):
            critic_x = nn.Dense(units, name=f'critic_dense_{i}')(critic_x)
            critic_x = jnp.tanh(critic_x)
            critic_x = nn.LayerNorm(epsilon=PPO_CONFIG[CONST_EPSILON], name=f'critic_ln_{i}')(critic_x)
        
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
        seed: int = CONST_DEFAULT_SEED
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
        self.hidden_units = PPO_CONFIG['hidden_units'] if hidden_units is None else hidden_units
        
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
        # Crear dos versiones compiladas con deterministic fijo
        self.get_action_jit_det = jax.jit(lambda p, s, k: self._get_action(p, s, k, True))
        self.get_action_jit_stoch = jax.jit(lambda p, s, k: self._get_action(p, s, k, False))
        self.get_value_jit = jax.jit(self._get_value)
        self.train_step = jax.jit(self._train_step)
        
        # Crear directorio para figuras si no existe
        os.makedirs(FIGURES_DIR, exist_ok=True)
    
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
               dones: np.ndarray, gamma: float = PPO_CONFIG[CONST_GAMMA], lam: float = PPO_CONFIG[CONST_LAMBDA]) -> Tuple[np.ndarray, np.ndarray]:
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
            # Si el episodio ha terminado, no hay recompensa futura
            mask = 1.0 - dones[t]
            # Calcular delta (diferencia temporal)
            delta = rewards[t] + gamma * next_values[t] * mask - values[t]
            # Actualizar la ventaja usando lambda
            last_gae = delta + gamma * lam * mask * last_gae
            advantages[t] = last_gae
        
        returns = advantages + values
        
        # Normalizar ventajas para estabilidad numérica
        advantages = (advantages - np.mean(advantages)) / (np.std(advantages) + 1e-10)
        
        return advantages, returns
    
    def _train_step(self, state: PPOTrainState, states: jnp.ndarray, actions: jnp.ndarray, 
              old_log_probs: jnp.ndarray, returns: jnp.ndarray, advantages: jnp.ndarray) -> Tuple[PPOTrainState, Dict[str, jnp.ndarray]]:
        """
        Ejecuta un paso de entrenamiento PPO.
        
        Parámetros:
        -----------
        state : PPOTrainState
            Estado actual del entrenamiento
        states : jnp.ndarray
            Estados para el paso de entrenamiento
        actions : jnp.ndarray
            Acciones tomadas
        old_log_probs : jnp.ndarray
            Log-probabilidades de la política antigua
        returns : jnp.ndarray
            Estimaciones de retorno
        advantages : jnp.ndarray
            Ventajas estimadas
            
        Retorna:
        --------
        Tuple[PPOTrainState, Dict[str, jnp.ndarray]]
            Estado actualizado y métricas
        """
        # Función de pérdida para este paso
        def loss_fn(params):
            # Obtener distribuciones de la política actual
            mu, sigma, values = self.model.apply(params, states)
            
            # Calcular log_probs actuales
            new_log_probs = self._log_prob(mu, sigma, actions)
            
            # Ratio de probabilidades (nueva/vieja)
            ratio = jnp.exp(new_log_probs - old_log_probs)
            
            # Calcular pérdidas clipeadas y no clipeadas
            policy_loss1 = -ratio * advantages
            policy_loss2 = -jnp.clip(ratio, 1.0 - self.epsilon, 1.0 + self.epsilon) * advantages
            
            # Tomar el máximo para minimizar
            policy_loss = jnp.mean(jnp.maximum(policy_loss1, policy_loss2))
            
            # Pérdida de valor
            value_pred = values.squeeze(-1)
            value_loss = jnp.mean(jnp.square(value_pred - returns))
            
            # Términos de entropía para exploración
            entropy = jnp.mean(0.5 * (jnp.log(2.0 * jnp.pi) + 1.0 + jnp.log(jnp.square(sigma))))
            
            # Pérdida total combinada
            total_loss = policy_loss + self.value_coef * value_loss - self.entropy_coef * entropy
            
            return total_loss, (policy_loss, value_loss, entropy)
        
        # Calcular gradientes
        grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
        (total_loss, (policy_loss, value_loss, entropy)), grads = grad_fn(state.params)
        
        # Actualizar parámetros con optimizador
        new_state = state.apply_gradients(grads=grads)
        
        # Preparar métricas para seguimiento
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
        state = jnp.asarray(state)[None, :]  # Añadir dimensión de lote
        
        # Usar la versión compilada adecuada
        if deterministic:
            action, key = self.get_action_jit_det(self.state.params, state, self.state.key)
        else:
            action, key = self.get_action_jit_stoch(self.state.params, state, self.state.key)
        
        # Actualizar key
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
        value = self.get_value_jit(self.state.params, state)
        return float(value[0][0])
    
    def _collect_trajectories(self, env: Any, steps_per_epoch: int) -> Tuple[Dict[str, np.ndarray], Dict[str, List[float]]]:
        """
        Recolecta trayectorias de experiencia interactuando con el entorno.
        
        Parámetros:
        -----------
        env : Any
            Entorno de interacción
        steps_per_epoch : int
            Número de pasos a recolectar
            
        Retorna:
        --------
        Tuple[Dict[str, np.ndarray], Dict[str, List[float]]]
            Datos de trayectorias y métricas de episodios
        """
        # Listas para almacenar experiencias
        states, actions, rewards, values, log_probs, dones = [], [], [], [], [], []
        
        # Métricas de episodios
        episode_rewards = []
        episode_lengths = []
        
        # Estado inicial
        state, _ = env.reset()
        done = False
        episode_reward = 0
        episode_length = 0
        
        # Recolectar experiencias
        for _ in range(steps_per_epoch):
            states.append(state)
            
            # Obtener acción, valor y log-prob
            key = self.state.key
            action, log_prob, value, new_key = self.get_action_and_value(
                self.state.params, 
                jnp.array([state]), 
                key
            )
            self.state = self.state.replace(key=new_key)
            
            actions.append(action[0])
            log_probs.append(log_prob[0])
            values.append(value[0, 0])
            
            # Ejecutar acción en el entorno
            next_state, reward, done, truncated, _ = env.step(action[0])
            done = done or truncated
            
            # Almacenar recompensa y flag de finalización
            rewards.append(reward)
            dones.append(float(done))
            
            # Actualizar métricas del episodio
            episode_reward += reward
            episode_length += 1
            
            # Si el episodio terminó
            if done:
                # Registrar métricas del episodio
                episode_rewards.append(episode_reward)
                episode_lengths.append(episode_length)
                
                # Reiniciar episodio
                state, _ = env.reset()
                episode_reward = 0
                episode_length = 0
            else:
                state = next_state
        
        # Calcular el valor del estado final (0 si terminado)
        if done:
            last_value = 0
        else:
            # Estimar el valor del último estado
            last_value = self.get_value_jit(self.state.params, jnp.array([state]))[0, 0]
        
        # Calcular ventajas y retornos usando GAE
        values_np = np.array(values)
        rewards_np = np.array(rewards)
        dones_np = np.array(dones)
        
        # Crear array de valores "siguiente"
        next_values = np.append(values_np[1:], last_value)
        
        # Calcular GAE
        advantages, returns = self._compute_gae(
            rewards=rewards_np,
            values=values_np,
            next_values=next_values,
            dones=dones_np,
            gamma=self.gamma,
            lam=0.95
        )
        
        # Empaquetar datos
        trajectory_data = {
            'states': np.array(states),
            'actions': np.array(actions),
            'log_probs': np.array(log_probs),
            'returns': returns,
            'advantages': advantages
        }
        
        episode_info = {
            'episode_rewards': episode_rewards,
            'episode_lengths': episode_lengths
        }
        
        return trajectory_data, episode_info
    
    def _update_policy(self, data: Dict[str, np.ndarray], batch_size: int, 
                 update_iters: int) -> Dict[str, float]:
        """
        Actualiza la política usando los datos recolectados.
        
        Parámetros:
        -----------
        data : Dict[str, np.ndarray]
            Datos de experiencias recolectadas
        batch_size : int
            Tamaño de lote para actualizaciones
        update_iters : int
            Número de iteraciones de actualización
            
        Retorna:
        --------
        Dict[str, float]
            Métricas promedio de las actualizaciones
        """
        # Extraer datos
        states = data['states']
        actions = data['actions']
        log_probs = data['log_probs']
        returns = data['returns']
        advantages = data['advantages']
        
        # Métricas acumuladas
        metrics_sum = {
            'total_loss': 0.0,
            'policy_loss': 0.0,
            'value_loss': 0.0,
            'entropy': 0.0
        }
        
        # Índices para muestreo aleatorio
        indices = np.arange(len(states))
        
        # Crear un generador de números aleatorios con semilla fija para reproducibilidad
        rng = np.random.Generator(np.random.PCG64(seed=CONST_DEFAULT_SEED))
        
        # Múltiples épocas de actualización con los mismos datos
        for _ in range(update_iters):
            # Mezclar índices para muestreo
            rng.shuffle(indices)
            
            # Procesar por lotes
            for start in range(0, len(states), batch_size):
                end = start + batch_size
                batch_indices = indices[start:end]
                
                # Extraer lote
                batch_states = states[batch_indices]
                batch_actions = actions[batch_indices]
                batch_log_probs = log_probs[batch_indices]
                batch_returns = returns[batch_indices]
                batch_advantages = advantages[batch_indices]
                
                # Actualizar parámetros con este lote
                self.state, batch_metrics = self.train_step(
                    self.state,
                    jnp.array(batch_states),
                    jnp.array(batch_actions),
                    jnp.array(batch_log_probs),
                    jnp.array(batch_returns),
                    jnp.array(batch_advantages)
                )
                
                # Acumular métricas
                for k, v in batch_metrics.items():
                    metrics_sum[k] += float(v)
        
        # Calcular promedios
        num_updates = update_iters * ((len(states) + batch_size - 1) // batch_size)
        metrics_avg = {k: v / max(1, num_updates) for k, v in metrics_sum.items()}
        
        return metrics_avg
    
    def train(self, env: Any, epochs: int = CONST_DEFAULT_EPOCHS, steps_per_epoch: int = 4000, batch_size: int = 64, 
         update_iters: int = 10, gae_lambda: float = 0.95,
         log_interval: int = 10) -> Dict[str, List[float]]:
        """
        Entrena el modelo PPO en el entorno dado.
        
        Parámetros:
        -----------
        env : Any
            Entorno de OpenAI Gym o compatible
        epochs : int, opcional
            Número de épocas de entrenamiento (default: 100)
        steps_per_epoch : int, opcional
            Número de pasos por época (default: 4000)
        batch_size : int, opcional
            Tamaño del lote para actualizaciones (default: 64)
        update_iters : int, opcional
            Número de iteraciones de actualización por época (default: 10) 
        gae_lambda : float, opcional
            Factor lambda para GAE (default: 0.95)
        log_interval : int, opcional
            Intervalo para mostrar información de progreso (default: 10)
            
        Retorna:
        --------
        Dict[str, List[float]]
            Historial de entrenamiento con métricas
        """
        # Historial para seguimiento de métricas
        history = {
            'total_loss': [],
            'policy_loss': [],
            'value_loss': [],
            'entropy': [],
            'episode_rewards': [],
            'episode_lengths': []
        }
        
        # Listas para gráfico de recompensas
        all_rewards = []
        
        # Bucle principal de entrenamiento
        for epoch in range(epochs):
            # Recolectar trayectorias
            trajectory_data, episode_history = self._collect_trajectories(env, steps_per_epoch)
            
            # Actualizar política con datos recolectados
            metrics = self._update_policy(trajectory_data, batch_size, update_iters)
            
            # Registrar métricas
            history['total_loss'].append(metrics['total_loss'])
            history['policy_loss'].append(metrics['policy_loss'])
            history['value_loss'].append(metrics['value_loss'])
            history['entropy'].append(metrics['entropy'])
            
            # Registrar recompensas y longitudes de episodio
            for reward in episode_history['episode_rewards']:
                history['episode_rewards'].append(reward)
                all_rewards.append(reward)
            for length in episode_history['episode_lengths']:
                history['episode_lengths'].append(length)

            # Mostrar información de progreso
            if (epoch + 1) % log_interval == 0 or epoch == 0:
                avg_reward = np.mean(episode_history['episode_rewards']) if episode_history['episode_rewards'] else 0
                print(f"Época {epoch+1}/{epochs} | " 
                      f"Recompensa media: {avg_reward:.2f} | "
                      f"Pérdida de Política: {metrics['policy_loss']:.4f} | "
                      f"Pérdida de Valor: {metrics['value_loss']:.4f}")
                
                # Graficar recompensas si hay suficientes datos
                if len(all_rewards) >= 10:
                    plt.figure(figsize=(10, 5))
                    plt.plot(all_rewards)
                    plt.title('Recompensas por episodio')
                    plt.xlabel('Episodio')
                    plt.ylabel('Recompensa total')
                    plt.grid(True)
                    plt.savefig(os.path.join(FIGURES_DIR, f'rewards_epoch_{epoch+1}.png'))
                    plt.close()
        
        # Graficar recompensas finales
        plt.figure(figsize=(10, 5))
        plt.plot(history['episode_rewards'])
        plt.title('Recompensas por episodio - Entrenamiento completo')
        plt.xlabel('Episodio')
        plt.ylabel('Recompensa total')
        plt.grid(True)
        plt.savefig(os.path.join(FIGURES_DIR, 'rewards_final.png'))
        plt.close()
        
        # Graficar pérdidas
        plt.figure(figsize=(15, 5))
        plt.subplot(1, 3, 1)
        plt.plot(history['policy_loss'], label='Política')
        plt.title('Pérdida de Política')
        plt.grid(True)
        
        plt.subplot(1, 3, 2)
        plt.plot(history['value_loss'], label='Valor')
        plt.title('Pérdida de Valor')
        plt.grid(True)
        
        plt.subplot(1, 3, 3)
        plt.plot(history['entropy'], label=CONST_ENTROPIA)
        plt.title(CONST_ENTROPIA)
        plt.grid(True)
        
        plt.tight_layout()
        plt.savefig(os.path.join(FIGURES_DIR, 'loss_metrics.png'))
        plt.close()
        
        return history
    
    def evaluate(self, env: Any, n_episodes: int = 10, deterministic: bool = True, render: bool = False) -> float:
        """
        Evalúa el rendimiento del modelo en un entorno.
        
        Parámetros:
        -----------
        env : Any
            Entorno de OpenAI Gym o compatible
        n_episodes : int, opcional
            Número de episodios para evaluar (default: 10)
        deterministic : bool, opcional
            Si usar política determinista (default: True)
        render : bool, opcional
            Si renderizar el entorno (default: False)
            
        Retorna:
        --------
        float
            Recompensa promedio por episodio
        """
        total_rewards = []
        
        for episode in range(n_episodes):
            state, _ = env.reset()
            episode_reward = 0
            done = False
            
            while not done:
                # Seleccionar acción (determinística o estocástica)
                action = self.get_action(state, deterministic=deterministic)
                
                # Ejecutar acción
                state, reward, terminated, truncated, _ = env.step(action)
                done = terminated or truncated
                episode_reward += reward
                
                # Renderizar si es necesario
                if render:
                    env.render()
            
            total_rewards.append(episode_reward)
            print(f"Episodio {episode+1}/{n_episodes}: Recompensa = {episode_reward:.2f}")
        
        mean_reward = np.mean(total_rewards)
        print(f"Recompensa promedio sobre {n_episodes} episodios: {mean_reward:.2f}")
        return mean_reward
    
    def save_model(self, filepath: str) -> None:
        """
        Guarda los parámetros del modelo en el disco.
        
        Parámetros:
        -----------
        filepath : str
            Ruta donde guardar el modelo
        """
        # Crear directorio si no existe
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        # Guardar sólo los parámetros del modelo
        with open(filepath, 'wb') as f:
            f.write(flax.serialization.to_bytes(self.state.params))
        
        print(f"Modelo guardado en: {filepath}")
        
    def load_model(self, filepath: str) -> None:
        """
        Carga los parámetros del modelo desde el disco.
        
        Parámetros:
        -----------
        filepath : str
            Ruta desde donde cargar el modelo
        """
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"No se encontró el modelo en {filepath}")
            
        # Cargar parámetros
        with open(filepath, 'rb') as f:
            params_bytes = f.read()
        
        # Restaurar parámetros en el modelo
        self.state = self.state.replace(
            params=flax.serialization.from_bytes(self.state.params, params_bytes)
        )
        
        print(f"Modelo cargado desde: {filepath}")
    
    def visualize_training(self, history: Dict[str, List[float]], window_size: int = 5) -> None:
        """
        Visualiza el progreso del entrenamiento con gráficos.
        
        Parámetros:
        -----------
        history : Dict[str, List[float]]
            Historial de métricas de entrenamiento
        window_size : int, opcional
            Tamaño de ventana para suavizado (default: 5)
        """
        # Crear directorio para figuras si no existe
        os.makedirs(FIGURES_DIR, exist_ok=True)
        
        # Graficar recompensas
        if 'episode_rewards' in history and len(history['episode_rewards']) > 0:
            plt.figure(figsize=(12, 6))
            rewards = np.array(history['episode_rewards'])
            plt.plot(rewards, alpha=0.5, label='Original')
            
            # Aplicar suavizado si hay suficientes datos
            if len(rewards) > window_size:
                smoothed = np.convolve(rewards, np.ones(window_size)/window_size, mode='valid')
                plt.plot(range(window_size-1, len(rewards)), smoothed, label=f'Suavizado (ventana={window_size})')
            
            plt.title('Recompensas por Episodio')
            plt.xlabel('Episodio')
            plt.ylabel('Recompensa Total')
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.savefig(os.path.join(FIGURES_DIR, 'rewards_history.png'))
            plt.close()
        
        # Graficar pérdidas
        plt.figure(figsize=(15, 5))
        
        plt.subplot(1, 3, 1)
        if 'policy_loss' in history:
            plt.plot(history['policy_loss'])
            plt.title('Pérdida de Política')
            plt.grid(True, alpha=0.3)
        
        plt.subplot(1, 3, 2)
        if 'value_loss' in history:
            plt.plot(history['value_loss'])
            plt.title('Pérdida de Valor')
            plt.grid(True, alpha=0.3)
        
        plt.subplot(1, 3, 3)
        if 'entropy' in history:
            plt.plot(history['entropy'])
            plt.title(CONST_ENTROPIA)
            plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(FIGURES_DIR, 'training_metrics.png'))
        plt.close()


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
        Inicializa el wrapper de PPO.
        
        Parámetros:
        -----------
        ppo_agent : PPO
            Agente PPO a encapsular
        cgm_shape : Tuple[int, ...]
            Forma de los datos CGM
        other_features_shape : Tuple[int, ...]
            Forma de otras características
        """
        self.ppo_agent = ppo_agent
        self.cgm_shape = cgm_shape
        self.other_features_shape = other_features_shape
        
        # Inicializar generador de números aleatorios
        self.key = jax.random.PRNGKey(CONST_DEFAULT_SEED)
        self.key, self.encoder_key = jax.random.split(self.key)
        
        # Configurar funciones de codificación para entradas
        self._setup_encoders()
        
        # Historial de entrenamiento
        self.history = {
            'loss': [],
            'val_loss': [],
            'policy_loss': [],
            'value_loss': [],
            'episode_rewards': [],
            'predictions': [],
            'val_predictions': [],
        }
    
    def _setup_encoders(self) -> None:
        """
        Configura las funciones de codificación para procesar las entradas.
        """
        # Calcular dimensiones de entrada aplanadas de manera segura
        if len(self.cgm_shape) <= 1:
            cgm_dim = max(1, self.cgm_shape[0])
        else:
            cgm_dim = int(np.prod(self.cgm_shape[1:]))
        
        if len(self.other_features_shape) <= 1:
            other_dim = max(1, self.other_features_shape[0])
        else:
            other_dim = int(np.prod(self.other_features_shape[1:]))
        
        print_debug(f"Configurando encoders PPO. CGM dim: {cgm_dim}, Other dim: {other_dim}")
        
        # Inicializar matrices de transformación
        self.key, key_cgm, key_other = jax.random.split(self.key, 3)
        
        # Crear matrices de proyección para la codificación de entradas
        self.cgm_weight = jax.random.normal(key_cgm, (int(cgm_dim), self.ppo_agent.state_dim // 2))
        self.other_weight = jax.random.normal(key_other, (int(other_dim), self.ppo_agent.state_dim // 2))
        
        # JIT-compilar transformaciones para mayor rendimiento
        self.encode_cgm = jax.jit(self._create_encoder_fn(self.cgm_weight))
        self.encode_other = jax.jit(self._create_encoder_fn(self.other_weight))
    
    def _create_encoder_fn(self, weights: jnp.ndarray) -> Callable:
        """
        Crea una función de codificación robusta que maneja discrepancias de dimensiones.
        
        Parámetros:
        -----------
        weights : jnp.ndarray
            Pesos de la transformación
            
        Retorna:
        --------
        Callable
            Función de codificación
        """
        def encode_fn(x: jnp.ndarray) -> jnp.ndarray:
            # Asegurar que la entrada tiene al menos 2 dimensiones (batch, features)
            if x.ndim == 1:
                x = x.reshape(1, -1)
            
            # Aplanar entrada manteniendo la dimensión del batch
            batch_size = x.shape[0]
            x_flat = x.reshape(batch_size, -1)
            
            # Verificar si las dimensiones coinciden
            input_dim = x_flat.shape[1]
            weight_dim = weights.shape[0]
            
            # Manejar posibles discrepancias de dimensiones
            if input_dim != weight_dim:
                print(f"Adaptando forma de entrada: {input_dim} a {weight_dim}")
                if input_dim > weight_dim:
                    # Truncar si los datos son más grandes que los pesos
                    x_flat = x_flat[:, :weight_dim]
                else:
                    # Rellenar con ceros si los datos son más pequeños
                    padding = jnp.zeros((batch_size, weight_dim - input_dim))
                    x_flat = jnp.concatenate([x_flat, padding], axis=1)
            
            # Aplicar la transformación y activación
            return jnp.tanh(jnp.dot(x_flat, weights))
        
        return encode_fn
    
    def __call__(self, cgm_input: jnp.ndarray, other_input: jnp.ndarray, training: bool = False) -> jnp.ndarray:
        """
        Realiza una predicción usando el modelo PPO.
        
        Parámetros:
        -----------
        cgm_input : jnp.ndarray
            Datos CGM de entrada
        other_input : jnp.ndarray
            Otras características de entrada
        training : bool, opcional
            Indica si está en modo entrenamiento (default: False)
            
        Retorna:
        --------
        jnp.ndarray
            Predicciones de dosis
        """
        return self.predict([cgm_input, other_input])
    
    def predict(self, inputs: List[jnp.ndarray]) -> jnp.ndarray:
        """
        Realiza predicciones usando el modelo PPO entrenado.
        
        Parámetros:
        -----------
        inputs : List[jnp.ndarray]
            Lista de tensores de entrada [cgm_data, other_features]
            
        Retorna:
        --------
        jnp.ndarray
            Predicciones del modelo (dosis de insulina)
        """
        # Extraer entradas
        if isinstance(inputs, list) and len(inputs) == 2:
            cgm_data, other_features = inputs
        elif isinstance(inputs, tuple) and len(inputs) == 2:
            cgm_data, other_features = inputs
        else:
            raise ValueError("La entrada debe ser una lista o tupla con [cgm_data, other_features]")
    
        # Obtener tamaño del batch
        batch_size = cgm_data.shape[0]
    
        # Inicializar array de predicciones
        predictions = np.zeros((batch_size, 1), dtype=np.float32)
    
        # Añadir barra de progreso
        batch_iterator = tqdm.tqdm(range(batch_size), desc="Prediciendo dosis PPO", disable=False)
    
        # Procesar cada muestra del batch con barra de progreso
        for i in batch_iterator:
            # Extraer muestra
            cgm_sample = cgm_data[i:i+1]
            other_sample = other_features[i:i+1]
        
            # Codificar entradas a espacio de estado
            cgm_encoded = self.encode_cgm(jnp.array(cgm_sample))
            other_encoded = self.encode_other(jnp.array(other_sample))
        
            # Combinar características en un estado
            state = np.concatenate([cgm_encoded.flatten(), other_encoded.flatten()])
        
            # Usar acción determinista para predicción
            action = self.ppo_agent.get_action(state, deterministic=True)
            predictions[i, 0] = action[0] if isinstance(action, np.ndarray) and action.size > 0 else action
    
        return predictions
    
    def _calculate_metrics(self, train_preds: jnp.ndarray, y: jnp.ndarray, 
                      validation_data: Optional[Tuple] = None, verbose: int = 1) -> Dict:
        """
        Calcula métricas de rendimiento del modelo.
        
        Parámetros:
        -----------
        train_preds : jnp.ndarray
            Predicciones del modelo en datos de entrenamiento
        y : jnp.ndarray
            Valores objetivo (dosis de insulina)
        validation_data : Optional[Tuple], opcional
            Datos de validación como (x_val, y_val) (default: None)
        verbose : int, opcional
            Nivel de verbosidad (default: 1)
            
        Retorna:
        --------
        Dict
            Diccionario con métricas calculadas
        """
        # Asegurar que las formas sean compatibles para cálculos
        y_flat = y.reshape(-1)
        train_preds_flat = train_preds.reshape(-1)
        
        # Calcular métricas de rendimiento
        mae = float(jnp.mean(jnp.abs(train_preds_flat - y_flat)))
        mse = float(jnp.mean((train_preds_flat - y_flat) ** 2))
        rmse = float(jnp.sqrt(mse))
        
        # Calcular R²
        y_mean = jnp.mean(y_flat)
        ss_total = jnp.sum((y_flat - y_mean) ** 2)
        ss_residual = jnp.sum((y_flat - train_preds_flat) ** 2)
        r2 = float(1 - (ss_residual / (ss_total + 1e-10)))
        
        # Guardar métricas en el historial
        self.history['mae'] = [mae]
        self.history['mse'] = [mse]
        self.history['rmse'] = [rmse]
        self.history['r2'] = [r2]
        
        # Calcular métricas de validación si se proporcionan
        if validation_data:
            val_x, val_y = validation_data
            val_preds = self.predict(val_x)
            
            # Aplanar para cálculos seguros
            val_y_flat = val_y.reshape(-1)
            val_preds_flat = val_preds.reshape(-1)
            
            val_mae = float(jnp.mean(jnp.abs(val_preds_flat - val_y_flat)))
            val_mse = float(jnp.mean((val_preds_flat - val_y_flat) ** 2))
            val_rmse = float(jnp.sqrt(val_mse))
            
            val_y_mean = jnp.mean(val_y_flat)
            val_ss_total = jnp.sum((val_y_flat - val_y_mean) ** 2)
            val_ss_residual = jnp.sum((val_y_flat - val_preds_flat) ** 2)
            val_r2 = float(1 - (val_ss_residual / (val_ss_total + 1e-10)))
            
            self.history['val_mae'] = [val_mae]
            self.history['val_mse'] = [val_mse]
            self.history['val_rmse'] = [val_rmse]
            self.history['val_r2'] = [val_r2]
            self.history['val_predictions'] = val_preds
        
        if verbose > 0:
            print(f"Training metrics - MAE: {mae:.4f}, RMSE: {rmse:.4f}, R²: {r2:.4f}")
            if validation_data:
                print(f"Validation metrics - MAE: {val_mae:.4f}, RMSE: {val_rmse:.4f}, R²: {val_r2:.4f}")
        
        print_debug(f"self.history: {self.history}")
        return self.history
    
    def fit(
        self, 
        x: List[jnp.ndarray], 
        y: jnp.ndarray, 
        validation_data: Optional[Tuple] = None, 
        epochs: int = CONST_DEFAULT_EPOCHS,
        batch_size: int = CONST_DEFAULT_BATCH_SIZE,
        callbacks: List = None,
        verbose: int = 0
    ) -> Dict:
        """
        Entrena el modelo PPO con los datos proporcionados.
        
        Parámetros:
        -----------
        x : List[jnp.ndarray]
            Lista con [cgm_input, other_input]
        y : jnp.ndarray
            Valores objetivo de dosis
        validation_data : Optional[Tuple], opcional
            Datos de validación como ([x_cgm_val, x_other_val], y_val) (default: None)
        epochs : int, opcional
            Número de episodios de entrenamiento (default: 10)
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
        training_history = self.ppo_agent.train(
            env=env,
            epochs=epochs,
            steps_per_epoch=min(2000, len(y)),
            batch_size=batch_size,
            update_iters=4,
            log_interval=max(1, epochs // 10) if verbose > 0 else epochs + 1
        )
        
        # Actualizar historial con métricas del entrenamiento
        self.history['policy_loss'] = training_history.get('policy_loss', [])
        self.history['value_loss'] = training_history.get('value_loss', [])
        self.history['episode_rewards'] = training_history.get('episode_rewards', [])
        
        # Calcular pérdida en datos de entrenamiento
        train_preds = self.predict(x)
        train_loss = float(jnp.mean((train_preds.flatten() - y) ** 2))
        self.history['loss'].append(train_loss)
        self.history['predictions'].extend(train_preds.flatten().tolist())
        
        # Evaluar en datos de validación si se proporcionan
        if validation_data:
            val_x, val_y = validation_data
            val_preds = self.predict(val_x)
            val_loss = float(jnp.mean((val_preds.flatten() - val_y) ** 2))
            self.history['val_loss'] = [val_loss]
            self.history['val_predictions'].extend(val_preds.flatten().tolist())
        
        self._calculate_metrics(train_preds, y, validation_data, verbose=verbose)
        
        if verbose > 0:
            print_success("Entrenamiento del modelo PPO completado.")
            print_info(f"Pérdida final: {train_loss:.4f}")
            if validation_data:
                print_info(f"Pérdida de validación: {val_loss:.4f}")
        
        return self.history
    
    def _create_training_environment(
        self, 
        cgm_data: jnp.ndarray, 
        other_features: jnp.ndarray, 
        targets: jnp.ndarray
    ) -> Any:
        """
        Crea un entorno de entrenamiento personalizado basado en los datos.
        
        Parámetros:
        -----------
        cgm_data : jnp.ndarray
            Datos CGM
        other_features : jnp.ndarray
            Otras características
        targets : jnp.ndarray
            Valores objetivo de dosis
            
        Retorna:
        --------
        Any
            Entorno de entrenamiento
        """
        class InsulinDosingEnv:
            """Entorno personalizado para problema de dosificación de insulina."""
            
            def __init__(self, cgm, features, targets, model_wrapper):
                self.cgm = np.array(cgm)
                self.features = np.array(features)
                self.targets = np.array(targets)
                self.model = model_wrapper
                self.rng = np.random.Generator(np.random.PCG64(CONST_DEFAULT_SEED))
                self.current_idx = 0
                self.max_idx = len(targets) - 1
                
                # Para compatibilidad con algoritmos RL
                self.observation_space = SimpleNamespace(
                    shape=(model_wrapper.ppo_agent.state_dim,),
                    low=-np.ones(model_wrapper.ppo_agent.state_dim) * 10,
                    high=np.ones(model_wrapper.ppo_agent.state_dim) * 10
                )
                
                self.action_space = SimpleNamespace(
                    shape=(1,),
                    low=np.array([0.0]),
                    high=np.array([15.0]),
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
                # Obtener valor de dosis (acción continua)
                dose = float(action[0])
                
                # Calcular recompensa como negativo del error absoluto
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
        
        # Crear y devolver el entorno
        return InsulinDosingEnv(cgm_data, other_features, targets, self)
    
    def save(self, filepath: str) -> None:
        """
        Guarda el modelo en disco.
        
        Parámetros:
        -----------
        filepath : str
            Ruta donde guardar el modelo
        """
        # Asegurar que el directorio existe
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        # Guardar modelo PPO
        self.ppo_agent.save_model(filepath)
        
        print_success(f"Modelo guardado en {filepath}")
    
    def load(self, filepath: str) -> None:
        """
        Carga el modelo desde disco.
        
        Parámetros:
        -----------
        filepath : str
            Ruta desde donde cargar el modelo
        """
        self.ppo_agent.load_model(filepath)
        print_success(f"Modelo cargado desde {filepath}")
    
    def get_config(self) -> Dict:
        """
        Obtiene la configuración del modelo.
        
        Retorna:
        --------
        Dict
            Configuración del modelo
        """
        return {
            "cgm_shape": self.cgm_shape,
            "other_features_shape": self.other_features_shape,
            "state_dim": self.ppo_agent.state_dim,
            "action_dim": self.ppo_agent.action_dim,
            "hidden_units": self.ppo_agent.hidden_units
        }


def create_ppo_model(cgm_shape: Tuple[int, ...], other_features_shape: Tuple[int, ...]) -> DRLModelWrapper:
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
    DRLModelWrapper
        Wrapper de PPO que implementa la interfaz compatible con el sistema
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
        seed=CONST_DEFAULT_SEED
    )
    
    # Crear wrapper del agente para compatibilidad con sistema de model creators
    wrapper = PPOWrapper(
        ppo_agent=ppo_agent,
        cgm_shape=cgm_shape,
        other_features_shape=other_features_shape
    )
    
    # Envolver en DRLModelWrapper para compatibilidad completa con la interfaz del sistema
    return DRLModelWrapper(model_cls=wrapper, framework="jax", algorithm="ppo")

def model_creator() -> Callable[[Tuple[int, ...], Tuple[int, ...]], DRLModelWrapper]:
    """
    Retorna una función para crear un modelo PPO compatible con la API del sistema.
    
    Retorna:
    --------
    Callable[[Tuple[int, ...], Tuple[int, ...]], DRLModelWrapper]
        Función para crear el modelo con las formas de entrada especificadas
    """
    return create_ppo_model