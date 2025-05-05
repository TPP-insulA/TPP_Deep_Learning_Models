import os, sys
import flax
import jax
import jax.numpy as jnp
import numpy as np
import flax.linen as nn
import optax
from flax.training import train_state
from typing import Tuple, Dict, List, Any, Optional, Union, Callable, Sequence
import threading
from functools import partial
from tqdm import tqdm

PROJECT_ROOT = os.path.abspath(os.getcwd())
sys.path.append(PROJECT_ROOT) 

from constants.constants import CONST_DEFAULT_SEED, CONST_DEFAULT_EPOCHS, CONST_DEFAULT_BATCH_SIZE
from config.models_config import A2C_A3C_CONFIG
from custom.drl_model_wrapper import DRLModelWrapper

# Constantes para uso repetido
CONST_DROPOUT = "dropout"
CONST_PARAMS = "params"
CONST_POLICY_LOSS = "policy_loss"
CONST_VALUE_LOSS = "value_loss"
CONST_ENTROPY_LOSS = "entropy_loss"
CONST_TOTAL_LOSS = "total_loss"
CONST_EPISODE_REWARDS = "episode_rewards"


class ActorCriticModel(nn.Module):
    """
    Modelo Actor-Crítico para A2C que divide la arquitectura en redes para
    política (actor) y valor (crítico).
    
    Parámetros:
    -----------
    state_dim : int
        Dimensión del espacio de estados
    action_dim : int
        Dimensión del espacio de acciones
    continuous : bool
        Indica si el espacio de acciones es continuo o discreto
    hidden_units : Optional[List[int]]
        Unidades ocultas en cada capa
    """
    state_dim: int
    action_dim: int
    continuous: bool = True
    hidden_units: Optional[Sequence[int]] = None
    
    def setup(self) -> None:
        """
        Inicializa las capas del modelo.
        """
        # Valores predeterminados para capas ocultas
        if self.hidden_units is None:
            self.hidden_units = A2C_A3C_CONFIG['hidden_units']
        
        # # Inicializamos con listas vacías
        # shared_layers = []
        # actor_layers = []
        # critic_layers = []
        
        # # Capas compartidas para procesamiento de estados
        # for i, units in enumerate(self.hidden_units[:2]):
        #     shared_layers.append((
        #         nn.Dense(units, name=f'shared_dense_{i}'),
        #         nn.LayerNorm(epsilon=A2C_A3C_CONFIG['epsilon'], name=f'shared_ln_{i}'),
        #         nn.Dropout(A2C_A3C_CONFIG['dropout_rate'], name=f'shared_dropout_{i}')
        #     ))
        
        # # Red del Actor (política)
        # for i, units in enumerate(self.hidden_units[2:]):
        #     actor_layers.append((
        #         nn.Dense(units, name=f'actor_dense_{i}'),
        #         nn.LayerNorm(epsilon=A2C_A3C_CONFIG['epsilon'], name=f'actor_ln_{i}')
        #     ))
        
        # # Red del Crítico (valor)
        # for i, units in enumerate(self.hidden_units[2:]):
        #     critic_layers.append((
        #         nn.Dense(units, name=f'critic_dense_{i}'),
        #         nn.LayerNorm(epsilon=A2C_A3C_CONFIG['epsilon'], name=f'critic_ln_{i}')
        #     ))
        
        # # Asignar las listas completas
        # self.shared_layers = shared_layers
        # self.actor_layers = actor_layers
        # self.critic_layers = critic_layers
        
        # # Capas de salida del actor (depende de si el espacio de acción es continuo o discreto)
        # if self.continuous:
        #     # Para acción continua (política gaussiana)
        #     self.mu = nn.Dense(self.action_dim, name='actor_mu')
        #     self.log_sigma = nn.Dense(self.action_dim, name='actor_log_sigma')
        # else:
        #     # Para acción discreta (política categórica)
        #     self.logits = nn.Dense(self.action_dim, name='actor_logits')
        
        # # Capa de salida del crítico (valor del estado)
        # self.value = nn.Dense(1, name='critic_value')

    @nn.compact
    def __call__(self, inputs: jnp.ndarray, training: bool = False, 
                rngs: Optional[Dict[str, jnp.ndarray]] = None) -> Tuple[Any, jnp.ndarray]:
        """
        Pasa la entrada por el modelo Actor-Crítico.
        
        Parámetros:
        -----------
        inputs : jnp.ndarray
            Tensor de entrada con los estados
        training : bool, opcional
            Indica si está en modo entrenamiento
        rngs : Optional[Dict[str, jnp.ndarray]], opcional
            Claves PRNG para procesos estocásticos
            
        Retorna:
        --------
        Tuple[Any, jnp.ndarray]
            (política, valor) - la política puede ser una tupla (mu, sigma) o logits
        """
        x = inputs
        dropout_rng = None if rngs is None else rngs.get(CONST_DROPOUT)
        
        # Capas compartidas
        for i, units in enumerate(self.hidden_units[:2]):
            x = nn.Dense(units, name=f'shared_dense_{i}')(x)
            x = jnp.tanh(x)
            x = nn.LayerNorm(epsilon=A2C_A3C_CONFIG['epsilon'], name=f'shared_ln_{i}')(x)
            x = nn.Dropout(A2C_A3C_CONFIG['dropout_rate'], name=f'shared_dropout_{i}')(
                x, deterministic=not training, rng=dropout_rng)
        
        # Actor network
        actor_x = x
        for i, units in enumerate(self.hidden_units[2:]):
            actor_x = nn.Dense(units, name=f'actor_dense_{i}')(actor_x)
            actor_x = jnp.tanh(actor_x)
            actor_x = nn.LayerNorm(epsilon=A2C_A3C_CONFIG['epsilon'], name=f'actor_ln_{i}')(actor_x)
        
        # Salida del actor
        if self.continuous:
            mu = nn.Dense(self.action_dim, name='actor_mu')(actor_x)
            log_sigma = nn.Dense(self.action_dim, name='actor_log_sigma')(actor_x)
            log_sigma = jnp.clip(log_sigma, -20.0, 2.0)
            sigma = jnp.exp(log_sigma)
            policy = (mu, sigma)
        else:
            logits = nn.Dense(self.action_dim, name='actor_logits')(actor_x)
            policy = logits
        
        # Crítico (valor)
        critic_x = x
        for i, units in enumerate(self.hidden_units[2:]):
            critic_x = nn.Dense(units, name=f'critic_dense_{i}')(critic_x)
            critic_x = jnp.tanh(critic_x)
            critic_x = nn.LayerNorm(epsilon=A2C_A3C_CONFIG['epsilon'], name=f'critic_ln_{i}')(critic_x)
        
        value = nn.Dense(1, name='critic_value')(critic_x)
        
        return policy, value

    def get_action(self, params: Dict, state: jnp.ndarray, rng: jnp.ndarray, 
                  deterministic: bool = False) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """
        Obtiene una acción basada en el estado actual.
        
        Parámetros:
        -----------
        params : Dict
            Parámetros del modelo
        state : jnp.ndarray
            Estado actual
        rng : jnp.ndarray
            Clave PRNG para generación de números aleatorios
        deterministic : bool, opcional
            Si es True, devuelve la acción con máxima probabilidad
        
        Retorna:
        --------
        Tuple[jnp.ndarray, jnp.ndarray]
            (acción, nueva_clave_rng)
        """
        rng, subkey = jax.random.split(rng)
        
        # Asegurar formato batch
        if state.ndim == 1:
            state = state[jnp.newaxis, :]
        
        # Aplicar modelo
        policy, _ = self.apply({"params": params}, state)
        
        if self.continuous:
            mu, sigma = policy
            if deterministic:
                return mu[0], rng
            
            # Muestrear de la distribución normal
            action = mu + sigma * jax.random.normal(subkey, mu.shape)
            return action[0], rng
        else:
            logits = policy
            if deterministic:
                return jnp.argmax(logits[0]), rng
            
            # Muestrear de la distribución categórica
            _ = jax.nn.softmax(logits)
            action = jax.random.categorical(subkey, logits[0])
            return action, rng

    def get_value(self, params: Dict, state: jnp.ndarray) -> jnp.ndarray:
        """
        Obtiene el valor estimado para un estado.
        
        Parámetros:
        -----------
        params : Dict
            Parámetros del modelo
        state : jnp.ndarray
            El estado para evaluar
        
        Retorna:
        --------
        jnp.ndarray
            El valor estimado del estado
        """
        # Asegurar formato batch
        if state.ndim == 1:
            state = state[jnp.newaxis, :]
        
        _, value = self.apply({"params": params}, state)
        return value[0]

    def evaluate_actions(self, params: Dict, states: jnp.ndarray, actions: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        """
        Evalúa las acciones tomadas, devolviendo log_probs, valores y entropía.
        
        Parámetros:
        -----------
        params : Dict
            Parámetros del modelo
        states : jnp.ndarray
            Los estados observados
        actions : jnp.ndarray
            Las acciones tomadas
        
        Retorna:
        --------
        Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]
            (log_probs, valores, entropía)
        """
        policy, values = self.apply({"params": params}, states)
        
        if self.continuous:
            mu, sigma = policy
            # Calcular log probabilidad para acciones continuas
            log_probs = -0.5 * jnp.sum(
                jnp.square((actions - mu) / sigma) + 
                2 * jnp.log(sigma) + 
                jnp.log(2.0 * np.pi), 
                axis=1
            )
            # Entropía de política gaussiana
            entropy = jnp.sum(
                0.5 * jnp.log(2.0 * np.pi * jnp.square(sigma)) + 0.5,
                axis=1
            )
        else:
            logits = policy
            # Calcular log probabilidad para acciones discretas
            action_masks = jax.nn.one_hot(actions, self.action_dim)
            log_probs = jnp.sum(
                action_masks * jax.nn.log_softmax(logits),
                axis=1
            )
            # Entropía de política categórica
            probs = jax.nn.softmax(logits)
            entropy = -jnp.sum(
                probs * jnp.log(probs + 1e-10),
                axis=1
            )
        
        return log_probs, values, entropy


class A2CTrainState(train_state.TrainState):
    """
    Estado de entrenamiento para A2C, extendiendo el TrainState de Flax.
    
    Parámetros:
    -----------
    metrics : Optional[Dict]
        Métricas de entrenamiento
    rng : Optional[jnp.ndarray]
        PRNG key
    """
    metrics: Optional[Dict] = None
    rng: Optional[jnp.ndarray] = None


class A2C:
    """
    Implementación del algoritmo Advantage Actor-Critic (A2C) con JAX.
    
    Este algoritmo utiliza un estimador de ventaja para actualizar la política
    y una red de valor para estimar los retornos esperados.
    
    Parámetros:
    -----------
    state_dim : int
        Dimensión del espacio de estados
    action_dim : int
        Dimensión del espacio de acciones
    continuous : bool
        Indica si el espacio de acciones es continuo o discreto
    learning_rate : float
        Tasa de aprendizaje
    gamma : float
        Factor de descuento
    entropy_coef : float
        Coeficiente de entropía para exploración
    value_coef : float
        Coeficiente de pérdida de valor
    max_grad_norm : float
        Norma máxima para recorte de gradientes
    hidden_units : Optional[List[int]]
        Unidades ocultas por capa
    seed : int
        Semilla para reproducibilidad
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
        hidden_units: Optional[List[int]] = None,
        seed: int = 0
    ):
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
            
        # Crear modelo
        self.rng = jax.random.PRNGKey(seed)
        self.rng, init_rng = jax.random.split(self.rng)
        self.model = ActorCriticModel(
            state_dim=state_dim, 
            action_dim=action_dim,
            continuous=continuous,
            hidden_units=self.hidden_units
        )
        
        # Inicializar parámetros con entrada ficticia
        dummy_input = jnp.ones((1, state_dim))
        params = self.model.init(init_rng, dummy_input)["params"]
        
        # Crear optimizador
        optimizer = optax.adam(learning_rate=learning_rate)
        
        # Crear estado de entrenamiento
        self.state = A2CTrainState.create(
            apply_fn=self.model.apply,
            params=params,
            tx=optimizer,
            metrics={
                "policy_loss": [],
                "value_loss": [],
                "entropy_loss": [],
                "total_loss": [],
                "episode_rewards": []
            },
            rng=self.rng
        )
    
    @partial(jax.jit, static_argnums=(0,))
    def _train_step(self, state: A2CTrainState, states: jnp.ndarray, 
                   actions: jnp.ndarray, returns: jnp.ndarray, 
                   advantages: jnp.ndarray) -> Tuple[A2CTrainState, Dict]:
        """
        Realiza un paso de entrenamiento para actualizar el modelo.
        
        Parámetros:
        -----------
        state : A2CTrainState
            Estado actual del entrenamiento
        states : jnp.ndarray
            Estados observados en el entorno
        actions : jnp.ndarray
            Acciones tomadas para esos estados
        returns : jnp.ndarray
            Retornos estimados (para entrenar el crítico)
        advantages : jnp.ndarray
            Ventajas estimadas (para entrenar el actor)
            
        Retorna:
        --------
        Tuple[A2CTrainState, Dict]
            Nuevo estado de entrenamiento y métricas
        """
        # Función para calcular la pérdida y gradientes
        def loss_fn(params):
            # Obtener log_probs, valores y entropía de las acciones tomadas
            log_probs, values, entropy = self.model.evaluate_actions(params, states, actions)
            
            # Calcular pérdida de la política (actor)
            policy_loss = -jnp.mean(log_probs * advantages)
            
            # Calcular pérdida del valor (crítico)
            value_loss = jnp.mean(jnp.square(returns - values))
            
            # Calcular pérdida de entropía para promover exploración
            entropy_loss = -jnp.mean(entropy)
            
            # Pérdida total ponderada
            total_loss = policy_loss + self.value_coef * value_loss + self.entropy_coef * entropy_loss
            
            # Devolver pérdida total y componentes individuales
            return total_loss, (policy_loss, value_loss, entropy_loss)
        
        # Calcular pérdida y gradientes
        grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
        (total_loss, (policy_loss, value_loss, entropy_loss)), grads = grad_fn(state.params)
        
        # Recortar gradientes para estabilidad
        if self.max_grad_norm > 0:
            grads = optax.clip_by_global_norm(grads, self.max_grad_norm)
        
        # Actualizar parámetros
        new_state = state.apply_gradients(grads=grads)
        
        # Crear diccionario de métricas
        metrics = {
            "policy_loss": policy_loss,
            "value_loss": value_loss,
            "entropy_loss": entropy_loss,
            "total_loss": total_loss
        }
        
        return new_state, metrics
    
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
            returns y ventajas calculados
        """
        # Añadir el valor del último estado
        values_extended = np.append(values, next_value)
        
        returns = np.zeros_like(rewards, dtype=np.float32)
        advantages = np.zeros_like(rewards, dtype=np.float32)
        
        # Calcular retornos y ventajas desde el final
        gae = 0
        for t in reversed(range(len(rewards))):
            # Si es terminal, el valor del siguiente estado es 0
            next_non_terminal = 1.0 - dones[t]
            next_value = values_extended[t + 1]
            
            # Delta temporal para GAE
            delta = rewards[t] + self.gamma * next_value * next_non_terminal - values_extended[t]
            
            # Calcular ventaja con GAE
            gae = delta + self.gamma * A2C_A3C_CONFIG['lambda'] * next_non_terminal * gae
            advantages[t] = gae
            
            # Calcular retornos (para entrenar el crítico)
            returns[t] = advantages[t] + values_extended[t]
        
        # Normalizar ventajas para reducir varianza
        if len(advantages) > 1:
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
            self.rng, action_rng = jax.random.split(self.rng)
            action, self.rng = self.model.get_action(
                self.state.params, jnp.array(current_state), action_rng
            )
            actions.append(action)
            
            # Valor del estado actual
            value = self.model.get_value(self.state.params, jnp.array(current_state)).item()
            values.append(value)
            
            # Ejecutar acción en el entorno
            next_state, reward, done, _, _ = env.step(np.array(action))
            
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
        self.state, metrics = self._train_step(
            self.state, 
            jnp.array(states_np), 
            jnp.array(actions_np), 
            jnp.array(returns), 
            jnp.array(advantages)
        )
        
        # Actualizar métricas
        self.state.metrics["policy_loss"].append(metrics["policy_loss"].item())
        self.state.metrics["value_loss"].append(metrics["value_loss"].item())
        self.state.metrics["entropy_loss"].append(metrics["entropy_loss"].item())
        self.state.metrics["total_loss"].append(metrics["total_loss"].item())
        
        return metrics["policy_loss"].item(), metrics["value_loss"].item(), metrics["entropy_loss"].item()

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
        history['policy_losses'].append(np.mean(self.state.metrics["policy_loss"]))
        history['value_losses'].append(np.mean(self.state.metrics["value_loss"]))
        history['entropy_losses'].append(np.mean(self.state.metrics["entropy_loss"]))
        
        # Resetear métricas
        self.state.metrics["policy_loss"] = []
        self.state.metrics["value_loss"] = []
        self.state.metrics["entropy_loss"] = []
        self.state.metrics["total_loss"] = []
        
        # Añadir recompensas de episodios completados
        if episode_rewards:
            history['episode_rewards'].extend(episode_rewards)
            avg_reward = np.mean(episode_rewards)
            
            # Mostrar progreso en cada época para mejor visibilidad
            print(f"Época {epoch+1}/{epochs} - Recompensa media: {avg_reward:.2f}, "
                  f"P. política: {policy_loss:.4f}, P. valor: {value_loss:.4f}")
            
            return []  # Resetear lista de recompensas
        
        # Si no hay nuevos episodios completos, mostrar al menos el progreso
        if (epoch + 1) % 5 == 0:
            print(f"Época {epoch+1}/{epochs} - P. política: {policy_loss:.4f}, P. valor: {value_loss:.4f}")
        
        return episode_rewards

    def train(self, env, n_steps: int = 10, epochs: int = CONST_DEFAULT_EPOCHS, 
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
            if not dones[-1]:
                next_value = self.model.get_value(
                    self.state.params, jnp.array(state)
                ).item()
            else:
                next_value = 0
            
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
        with open(filepath, 'wb') as f:
            params_bytes = flax.serialization.to_bytes(self.state.params)
            f.write(params_bytes)
        
    def load_model(self, filepath: str) -> None:
        """
        Carga el modelo desde un archivo.
        
        Parámetros:
        -----------
        filepath : str
            Ruta desde donde cargar el modelo
        """
        with open(filepath, 'rb') as f:
            params_bytes = f.read()
            params = flax.serialization.from_bytes(self.state.params, params_bytes)
            self.state = self.state.replace(params=params)


class A3C(A2C):
    """
    Implementación de Asynchronous Advantage Actor-Critic (A3C) con JAX.
    
    Extiende A2C para permitir entrenamiento asíncrono con múltiples trabajadores.
    
    Parámetros:
    -----------
    state_dim : int
        Dimensión del espacio de estados
    action_dim : int
        Dimensión del espacio de acciones
    continuous : bool
        Indica si el espacio de acciones es continuo o discreto
    n_workers : int
        Número de trabajadores asíncronos
    learning_rate : float
        Tasa de aprendizaje
    gamma : float
        Factor de descuento
    entropy_coef : float
        Coeficiente de entropía para exploración
    value_coef : float
        Coeficiente de pérdida de valor
    max_grad_norm : float
        Norma máxima para recorte de gradientes
    hidden_units : Optional[List[int]]
        Unidades ocultas por capa
    seed : int
        Semilla para reproducibilidad
    """
    def __init__(self, state_dim: int, action_dim: int, continuous: bool = True, 
                n_workers: int = 4, **kwargs):
        super(A3C, self).__init__(state_dim, action_dim, continuous, **kwargs)
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
            self.state,
            self.rng,
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
            # Crear nueva llave PRNG para cada trabajador
            self.rng, worker_rng = jax.random.split(self.rng)
            worker = self.create_worker(env_fn, i)
            worker.rng = worker_rng
            workers.append(worker)
            print(f"Worker {i} inicializado")
        
        # Variables para seguimiento de recompensas y pérdidas
        history = {
            'episode_rewards': [],
            'policy_losses': [],
            'value_losses': [],
            'entropy_losses': []
        }
        
        # Lock para proteger el acceso concurrente al modelo global
        model_lock = threading.Lock()
        
        progress = tqdm(total=total_steps, desc="Progreso A3C global")
        self.progress_counter = 0
        
        def update_progress(steps_completed):
            with model_lock:
                self.progress_counter += steps_completed
                progress.update(steps_completed)
        
        # Crear e iniciar hilos
        threads = []
        for worker in workers:
            thread = threading.Thread(
                target=worker.train,
                args=(n_steps, total_steps // self.n_workers, history, model_lock, render, update_progress)
            )
            threads.append(thread)
            thread.daemon = True  # Terminar hilos cuando el programa principal termina
            thread.start()
        
        # Esperar a que terminen todos los hilos
        for thread in threads:
            thread.join()
        
        progress.close()
        print(f"Entrenamiento A3C completado. {len(history['episode_rewards'])} episodios finalizados.")
        
        return history


class A3CWorker:
    """
    Trabajador para el algoritmo A3C que entrena de forma asíncrona con JAX.
    
    Parámetros:
    -----------
    global_model : actor_critic_model
        Modelo compartido global
    global_state : a2c_train_state
        Estado de entrenamiento global
    rng : jnp.ndarray
        Clave PRNG para aleatoriedad
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
        Coeficiente de entropía para exploración
    value_coef : float
        Coeficiente de pérdida de valor
    max_grad_norm : float
        Norma máxima para recorte de gradientes
    continuous : bool
        Indica si el espacio de acciones es continuo o discreto
    """
    def __init__(
        self,
        global_model: ActorCriticModel,
        global_state: A2CTrainState,
        rng: jnp.ndarray,
        env_fn: Callable,
        worker_id: int,
        state_dim: int,
        action_dim: int,
        gamma: float = 0.99,
        entropy_coef: float = 0.01,
        value_coef: float = 0.5,
        max_grad_norm: float = 0.5,
        continuous: bool = True
    ):
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
        self.global_state = global_state
        self.rng = rng
        
        # Modelo local para este trabajador (misma arquitectura, diferentes pesos)
        self.local_model = global_model
        
        # Estado local de entrenamiento
        self.local_state = A2CTrainState.create(
            apply_fn=global_state.apply_fn,
            params=jax.tree_util.tree_map(lambda x: jnp.copy(x), global_state.params),
            tx=global_state.tx,
            metrics=dict(global_state.metrics),  # Copiar métrica
            rng=rng
        )
        
        # Actualizar pesos locales
        self.update_local_model()
    
    def update_local_model(self) -> None:
        """
        Actualiza los pesos del modelo local desde el modelo global.
        """
        # Copiar parámetros del modelo global al local
        self.local_state = self.local_state.replace(
            params=jax.tree_util.tree_map(lambda x: jnp.copy(x), self.global_state.params)
        )
    
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
            returns y ventajas calculados
        """
        # Añadir el valor del último estado
        values_extended = np.append(values, next_value)
        
        returns = np.zeros_like(rewards, dtype=np.float32)
        advantages = np.zeros_like(rewards, dtype=np.float32)
        
        # Calcular retornos y ventajas desde el final
        gae = 0
        for t in reversed(range(len(rewards))):
            # Si es terminal, el valor del siguiente estado es 0
            next_non_terminal = 1.0 - dones[t]
            next_value = values_extended[t + 1]
            
            # Delta temporal para GAE
            delta = rewards[t] + self.gamma * next_value * next_non_terminal - values_extended[t]
            
            # Calcular ventaja con GAE
            gae = delta + self.gamma * A2C_A3C_CONFIG['lambda'] * next_non_terminal * gae
            advantages[t] = gae
            
            # Calcular retornos (para entrenar el crítico)
            returns[t] = advantages[t] + values_extended[t]
        
        # Normalizar ventajas para reducir varianza
        if len(advantages) > 1:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        return returns, advantages
    
    def train_step(self, states: jnp.ndarray, actions: jnp.ndarray, 
                  returns: jnp.ndarray, advantages: jnp.ndarray) -> Tuple[Dict, Dict]:
        """
        Realiza un paso de entrenamiento asíncrono.
        
        Parámetros:
        -----------
        states : jnp.ndarray
            Estados observados
        actions : jnp.ndarray
            Acciones tomadas
        returns : jnp.ndarray
            Retornos calculados
        advantages : jnp.ndarray
            Ventajas calculadas
            
        Retorna:
        --------
        Tuple[Dict, Dict]
            (gradientes calculados, métricas)
        """
        def loss_fn(params):
            # Evaluar acciones con el modelo local
            log_probs, values, entropy = self.local_model.evaluate_actions(params, states, actions)
            
            # Calcular pérdida de política
            advantages_flat = advantages.reshape(-1)
            policy_loss = -jnp.mean(log_probs * advantages_flat, axis=0)
            
            # Calcular pérdida de valor
            value_pred = values.reshape(-1)
            returns_flat = returns.reshape(-1)
            value_loss = jnp.mean(jnp.square(returns_flat - value_pred), axis=0)
            
            # Calcular pérdida de entropía (regularización)
            entropy_loss = -jnp.mean(entropy, axis=0)
            
            # Pérdida total combinada
            total_loss = policy_loss + self.value_coef * value_loss + self.entropy_coef * entropy_loss
            
            return total_loss, (policy_loss, value_loss, entropy_loss)
        
        # Calcular pérdida y gradientes
        grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
        (total_loss, (policy_loss, value_loss, entropy_loss)), grads = grad_fn(self.local_state.params)
        
        # Recortar gradientes si es necesario
        if self.max_grad_norm is not None:
            grads = optax.clip_by_global_norm(grads, self.max_grad_norm)
            
        # Retornar gradientes y métricas (sin actualizar directamente)
        metrics = {
            "policy_loss": policy_loss,
            "value_loss": value_loss,
            "entropy_loss": entropy_loss,
            "total_loss": total_loss
        }
        
        return grads, metrics
    
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
        self.rng, action_rng = jax.random.split(self.rng)
        action, self.rng = self.local_model.get_action(
            self.local_state.params, jnp.array(state), action_rng
        )
        
        value = self.local_model.get_value(
            self.local_state.params, jnp.array(state)
        ).item()
        
        # Ejecutar acción en el entorno
        next_state, reward, done, _, _ = self.env.step(np.array(action))
        
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
                                        next_state: np.ndarray, 
                                        shared_history: Dict, model_lock: threading.Lock) -> None:
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
        model_lock : threading.Lock
            Lock para proteger acceso al modelo global
        """
        # Si el episodio no terminó, calcular valor del último estado
        next_value = 0.0 if done else self.local_model.get_value(
            self.local_state.params, jnp.array(next_state)
        ).item()
            
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
        
        # Calcular gradientes con el modelo local (sin actualizar aún)
        grads, metrics = self.train_step(
            jnp.array(states_np),
            jnp.array(actions_np),
            jnp.array(returns),
            jnp.array(advantages)
        )
        
        # Acceder al modelo global de manera segura para aplicar gradientes
        with model_lock:
            # Actualizar modelo global usando los gradientes calculados
            updates, new_opt_state = self.global_state.tx.update(
                grads, self.global_state.opt_state, self.global_state.params
            )
            new_params = optax.apply_updates(self.global_state.params, updates)
            
            # Actualizar estado global
            self.global_state = self.global_state.replace(
                params=new_params,
                opt_state=new_opt_state
            )
            
            # Guardar estadísticas
            with threading.Lock():  # Proteger acceso compartido
                shared_history['policy_losses'].append(metrics["policy_loss"].item())
                shared_history['value_losses'].append(metrics["value_loss"].item())
                shared_history['entropy_losses'].append(metrics["entropy_loss"].item())
                
        # Actualizar modelo local desde el global
        self.update_local_model()
    
    def train(self, n_steps: int, max_steps: int, shared_history: Dict, 
             model_lock: threading.Lock, render: bool = False, update_progress: Optional[Callable] = None, verbose: int = 1) -> None:
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
        model_lock : threading.Lock
            Lock para proteger acceso al modelo global
        render : bool, opcional
            Si se debe renderizar el entorno
        verbose : int, opcional
            Nivel de verbosidad (0=silencioso, 1=progreso, 2=detallado)
        """
        # Estado inicial
        state, _ = self.env.reset()
        episode_reward = 0.0
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
                    states, actions, rewards, dones, values, done, state, shared_history, model_lock
                )
            
            # Actualizar progreso global
            if update_progress:
                update_progress(n_steps)

class A2CWrapper:
    """
    Wrapper para hacer que el agente A2C sea compatible con la interfaz de modelos de aprendizaje profundo.
    """
    
    def __init__(
        self, 
        a2c_agent: A2C, 
        cgm_shape: Tuple[int, ...], 
        other_features_shape: Tuple[int, ...],
    ) -> None:
        """
        Inicializa el wrapper para A2C.
        
        Parámetros:
        -----------
        a2c_agent : A2C
            Agente A2C a utilizar
        cgm_shape : Tuple[int, ...]
            Forma de entrada para datos CGM
        other_features_shape : Tuple[int, ...]
            Forma de entrada para otras características
        """
        self.a2c_agent = a2c_agent
        self.cgm_shape = cgm_shape
        self.other_features_shape = other_features_shape
        
        # Inicializar clave para generación de números aleatorios
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
            'entropy_loss': [],
            'episode_rewards': []
        }
    
    def _setup_encoders(self) -> None:
        """
        Configura las funciones de codificación para procesar las entradas.
        """
        # Calcular dimensiones de características aplanadas asegurando que sean enteros
        if len(self.cgm_shape) > 1:
            cgm_dim = int(np.prod(self.cgm_shape[1:]))
        else:
            cgm_dim = int(self.cgm_shape[0])
        
        if len(self.other_features_shape) > 1:
            other_dim = int(np.prod(self.other_features_shape[1:]))
        else:
            other_dim = int(self.other_features_shape[0])
        
        # Inicializar matrices de transformación
        self.key, key_cgm, key_other = jax.random.split(self.key, 3)
        
        # Crear matrices de proyección para entradas
        self.cgm_weight = jax.random.normal(key_cgm, (cgm_dim, self.a2c_agent.state_dim // 2))
        self.other_weight = jax.random.normal(key_other, (other_dim, self.a2c_agent.state_dim // 2))
        
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
        Realiza predicciones usando el modelo entrenado.
        
        Parámetros:
        -----------
        inputs : List[jnp.ndarray]
            Lista de tensores de entrada [cgm_data, other_features]
            
        Retorna:
        --------
        jnp.ndarray
            Predicciones del modelo (dosis de insulina)
        """
        # Preparar las entradas
        if isinstance(inputs, list) and len(inputs) == 2:
            cgm_data, other_features = inputs
        elif isinstance(inputs, tuple) and len(inputs) == 2:
            cgm_data, other_features = inputs
        else:
            raise ValueError("La entrada debe ser una lista o tupla con [cgm_data, other_features]")
        
        # Obtener tamaño del batch
        batch_size = cgm_data.shape[0]
        
        # Inicializar array de acciones
        actions = np.zeros((batch_size, 1), dtype=np.float32)
        
        # Procesar cada muestra del batch
        for i in range(batch_size):
            # Extraer muestra
            cgm_sample = cgm_data[i:i+1]
            other_sample = other_features[i:i+1]
            
            # Codificar estado
            state = self._encode_state(cgm_sample, other_sample)
            
            # Obtener acción determinística (para predicción)
            action, _ = self.a2c_agent.model.get_action(
                self.a2c_agent.state.params,
                state,
                self.a2c_agent.rng,
                deterministic=True
            )
            
            # Convertir a valor de dosis (asumiendo espacio continuo)
            if self.a2c_agent.continuous:
                action_value = action[0]  # Acción ya continua
            else:
                # Para acciones discretas, convertir índice a valor continuo
                action_value = action / (self.a2c_agent.action_dim - 1) * 15.0
                
            actions[i, 0] = action_value
        
        return actions
    
    def fit(
        self, 
        x: List[jnp.ndarray], 
        y: jnp.ndarray, 
        validation_data: Optional[Tuple] = None, 
        epochs: int = CONST_DEFAULT_EPOCHS,
        batch_size: int = CONST_DEFAULT_BATCH_SIZE,
        callbacks: List = None,
        verbose: int = 1
    ) -> Dict:
        """
        Entrena el modelo A2C con los datos proporcionados.
        
        Parámetros:
        -----------
        x : List[jnp.ndarray]
            Lista de tensores de entrada [cgm_data, other_features]
        y : jnp.ndarray
            Valores objetivo (dosis de insulina)
        validation_data : Optional[Tuple], opcional
            Datos de validación como (x_val, y_val) (default: None)
        epochs : int, opcional
            Número de épocas de entrenamiento (default: 1)
        batch_size : int, opcional
            Tamaño de lote (default: 32)
        callbacks : List, opcional
            Callbacks para el entrenamiento (default: None)
        verbose : int, opcional
            Nivel de verbosidad (default: 0)
            
        Retorna:
        --------
        Dict
            Historial de entrenamiento con métricas
        """
        # Verificar que las entradas son correctas
        if not isinstance(x, list) or len(x) != 2:
            raise ValueError("x debe ser una lista con [cgm_data, other_features]")
        
        # Configurar reportes de progreso
        if verbose > 0:
            progress_bar = tqdm(total=epochs, desc="Entrenando A2C")
        
        # Crear entorno para entrenamiento
        env = self._create_training_environment(x[0], x[1], y)
        
        # Calcular número de pasos totales basados en epochs y batch_size
        # total_steps = epochs * (len(y) // batch_size) * batch_size
        
        # Imprimir información inicial
        if verbose > 0:
            print(f"Iniciando entrenamiento A2C | Épocas: {epochs} | Batch: {batch_size} | Ejemplos: {len(y)}")
        
        # Entrenar al agente A2C
        history = self.a2c_agent.train(
            env=env,
            n_steps=batch_size,
            epochs=epochs,
            render=False
        )
        
        # Calibrar predictor de dosis basado en los datos objetivo
        self._calibrate_dose_predictor(y)
        
        # Actualizar historial
        self.history['episode_rewards'].extend(history.get('episode_rewards', []))
        self.history['policy_loss'].extend(history.get('policy_losses', []))
        self.history['value_loss'].extend(history.get('value_losses', []))
        self.history['entropy_loss'].extend(history.get('entropy_losses', []))
        
        # Calcular pérdida en datos de entrenamiento
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
            progress_bar.close()
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
        # Crear entorno personalizado para A2C/A3C
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
                    shape=(model_wrapper.a2c_agent.state_dim,),
                    low=np.full((model_wrapper.a2c_agent.state_dim,), -10.0),
                    high=np.full((model_wrapper.a2c_agent.state_dim,), 10.0)
                )
                
                if model_wrapper.a2c_agent.model.continuous:
                    self.action_space = SimpleNamespace(
                        shape=(1,),
                        low=np.array([0.0]),
                        high=np.array([15.0]), # Max 15 unidades de insulina
                        sample=self._sample_continuous_action
                    )
                else:
                    self.action_space = SimpleNamespace(
                        n=model_wrapper.a2c_agent.model.action_dim,
                        sample=lambda: self.rng.integers(0, model_wrapper.a2c_agent.model.action_dim)
                    )
            
            def _sample_continuous_action(self):
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
                # Convertir acción a dosis según tipo de espacio de acción
                if hasattr(self.action_space, 'shape'):  # Acción continua
                    dose = action[0]
                else:  # Acción discreta
                    dose = action / (self.model.a2c_agent.model.action_dim - 1) * 15.0
                
                # Calcular recompensa como negativo del error absoluto
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
        Guarda el modelo A2C en un archivo.
        
        Parámetros:
        -----------
        filepath : str
            Ruta donde guardar el modelo
        """
        # Crear directorio si no existe
        import os
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        # Guardar el agente A2C
        self.a2c_agent.save_model(filepath + "_a2c.h5")
        
        # Guardar datos adicionales del wrapper
        import pickle
        wrapper_data = {
            'cgm_shape': self.cgm_shape,
            'other_features_shape': self.other_features_shape,
            'cgm_weight': self.cgm_weight,
            'other_weight': self.other_weight,
            'state_dim': self.a2c_agent.state_dim
        }
        
        with open(filepath + "_wrapper.pkl", 'wb') as f:
            pickle.dump(wrapper_data, f)
        
        print(f"Modelo guardado en {filepath}")
    
    def load(self, filepath: str) -> None:
        """
        Carga el modelo A2C desde un archivo.
        
        Parámetros:
        -----------
        filepath : str
            Ruta desde donde cargar el modelo
        """
        # Cargar el agente A2C
        self.a2c_agent.load_model(filepath + "_a2c.h5")
        
        # Cargar datos adicionales del wrapper
        import pickle
        with open(filepath + "_wrapper.pkl", 'rb') as f:
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
            'state_dim': self.a2c_agent.state_dim,
            'continuous': self.a2c_agent.model.continuous,
            'gamma': self.a2c_agent.gamma,
            'entropy_coef': self.a2c_agent.entropy_coef,
            'value_coef': self.a2c_agent.value_coef
        }
    
    def start(self, x_cgm: jnp.ndarray, x_other: jnp.ndarray, y: jnp.ndarray, 
         rng_key: Optional[jax.random.PRNGKey] = None) -> Any:
        """
        Inicializa el agente con los datos de entrada.
        
        Parámetros:
        -----------
        x_cgm : jnp.ndarray
            Datos CGM de entrada
        x_other : jnp.ndarray
            Otras características de entrada
        y : jnp.ndarray
            Valores objetivo
        rng_key : Optional[jax.random.PRNGKey], opcional
            Clave para generación aleatoria (default: None)
            
        Retorna:
        --------
        Any
            Estado inicial del agente
        """
        # Inicializar codificadores si es necesario
        if not hasattr(self, 'encode_cgm') or self.encode_cgm is None:
            self._setup_encoders()
        
        # Inicializar pesos de encoders si no existen
        if not hasattr(self, 'cgm_weight') or self.cgm_weight is None:
            # Inicializar pesos aleatorios para los codificadores
            key1, key2 = jax.random.split(rng_key)
            self.cgm_weight = jax.random.normal(key1, shape=(self.cgm_shape[1], 64))
            self.other_weight = jax.random.normal(key2, shape=(self.other_features_shape[0], 32))
            
            # Crear funciones de codificación
            self.encode_cgm = jax.jit(self._create_encoder_fn(self.cgm_weight))
            self.encode_other = jax.jit(self._create_encoder_fn(self.other_weight))
        
        # Inicializar historial de entrenamiento
        self.history = {
            'loss': [],
            'val_loss': [],
            'episode_rewards': [],
            'policy_loss': [],
            'value_loss': [],
            'entropy_loss': []
        }
        
        return self

    def _encode_state(self, cgm_sample: jnp.ndarray, other_sample: jnp.ndarray) -> jnp.ndarray:
        """
        Codifica cgm y otras características en un estado para el agente RL.
        
        Parámetros:
        -----------
        cgm_sample : jnp.ndarray
            Datos CGM para una muestra
        other_sample : jnp.ndarray
            Otras características para una muestra
            
        Retorna:
        --------
        jnp.ndarray
            Estado codificado para el agente RL
        """
        # Codificar CGM y otras características a la dimensión del estado
        cgm_encoded = self.encode_cgm(cgm_sample)
        other_encoded = self.encode_other(other_sample)
        
        # Concatenar para formar el estado completo
        state = jnp.concatenate([cgm_encoded[0], other_encoded[0]])
        
        return state

    def _calibrate_dose_predictor(self, targets: jnp.ndarray) -> None:
        """
        Calibra el predictor de dosis basado en los datos objetivo.
        
        Parámetros:
        -----------
        targets : jnp.ndarray
            Dosis objetivo observadas
        """
        # Obtener estadísticas de las dosis objetivo
        min_dose = float(jnp.min(targets))
        max_dose = float(jnp.max(targets))
        mean_dose = float(jnp.mean(targets))
        std_dose = float(jnp.std(targets))
        
        # Almacenar para escalar predicciones
        self.dose_stats = {
            'min': min_dose,
            'max': max_dose,
            'mean': mean_dose,
            'std': std_dose
        }
        
        # Actualizar límites para acciones continuas
        if self.a2c_agent.continuous:
            # Si las acciones son valores normalizados entre -1 y 1
            # necesitamos mapearlos al rango de dosis [min_dose, max_dose]
            self.output_scale = (max_dose - min_dose) / 2.0
            self.output_shift = (max_dose + min_dose) / 2.0


class A3CWrapper(A2CWrapper):
    """
    Wrapper para hacer que el agente A3C sea compatible con la interfaz de modelos de aprendizaje por refuerzo profundo.
    Extiende el wrapper A2C añadiendo funcionalidad para entrenamiento asíncrono.
    """
    
    def __init__(
        self, 
        a3c_agent: A3C, 
        cgm_shape: Tuple[int, ...], 
        other_features_shape: Tuple[int, ...],
    ) -> None:
        """
        Inicializa el wrapper para A3C.
        
        Parámetros:
        -----------
        a3c_agent : A3C
            Agente A3C a envolver
        cgm_shape : Tuple[int, ...]
            Forma de los datos CGM
        other_features_shape : Tuple[int, ...]
            Forma de otras características
        """
        super().__init__(a3c_agent, cgm_shape, other_features_shape)
        # Especificamos que es un agente A3C para gestión apropiada
        self.algorithm_type = "a3c"
    
    def fit(
        self, 
        x: List[jnp.ndarray], 
        y: jnp.ndarray, 
        validation_data: Optional[Tuple] = None, 
        epochs: int = CONST_DEFAULT_EPOCHS,
        batch_size: int = CONST_DEFAULT_BATCH_SIZE,
        callbacks: List = None,
        verbose: int = 1
    ) -> Dict:
        """
        Entrena el modelo A3C en los datos proporcionados utilizando múltiples workers.
        
        Parámetros:
        -----------
        x : List[jnp.ndarray]
            Lista con [cgm_data, other_features]
        y : jnp.ndarray
            Etiquetas (dosis objetivo)
        validation_data : Optional[Tuple], opcional
            Datos de validación (default: None)
        epochs : int, opcional
            Número de épocas (default: 10)
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
            print("Entrenando modelo A3C con múltiples workers...")
        
        # Función para crear entornos
        def create_env_fn():
            return self._create_training_environment(x[0], x[1], y)
        
        # Entrenar el agente A3C en modo asíncrono
        # Calcula pasos totales como epochs * batch_size
        total_steps = epochs * batch_size
        
        a3c_history = self.a2c_agent.train_async(
            env_fn=create_env_fn,
            n_steps=batch_size // 4,  # Tamaño del lote para actualización A3C por worker
            total_steps=total_steps,
            render=False
        )
        
        # Actualizar historial con métricas del entrenamiento
        self.history['episode_rewards'].extend(a3c_history.get('episode_rewards', [0.0]))
        self.history['policy_loss'].extend(a3c_history.get('policy_losses', [0.0]))
        self.history['value_loss'].extend(a3c_history.get('value_losses', [0.0]))
        self.history['entropy_loss'].extend(a3c_history.get('entropy_losses', [0.0]))
        
        # Calcular pérdida en datos de entrenamiento
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
            print(f"Entrenamiento asíncrono completado. Pérdida final: {train_loss:.4f}")
            if validation_data:
                print(f"Pérdida de validación: {val_loss:.4f}")
        
        return self.history
    
    def save(self, filepath: str) -> None:
        """
        Guarda el modelo A3C en un archivo.
        
        Parámetros:
        -----------
        filepath : str
            Ruta donde guardar el modelo
        """
        super(A3CWrapper, self).save(filepath.replace("a2c", "a3c"))
    
    def load(self, filepath: str) -> None:
        """
        Carga el modelo A3C desde un archivo.
        
        Parámetros:
        -----------
        filepath : str
            Ruta desde donde cargar el modelo
        """
        super(A3CWrapper, self).load(filepath.replace("a2c", "a3c"))
    
    def get_config(self) -> Dict:
        """
        Obtiene la configuración del modelo.
        
        Retorna:
        --------
        Dict
            Diccionario con configuración del modelo
        """
        config = super(A3CWrapper, self).get_config()
        config['n_workers'] = self.a2c_agent.n_workers
        return config


def create_a2c_model(cgm_shape: Tuple[int, ...], other_features_shape: Tuple[int, ...]) -> DRLModelWrapper:
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
    DRLModelWrapper
        Wrapper del modelo A2C compatible con la interfaz del sistema
    """
    # Configurar el espacio de estados y acciones
    state_dim = 64  # Dimensión del espacio de estado latente
    action_dim = 1  # Una dimensión para dosis continua
    continuous = True  # Usar espacio de acción continuo
    
    # Crear agente A2C con configuración específica
    a2c_agent = A2C(
        state_dim=state_dim,
        action_dim=action_dim,
        continuous=continuous,
        learning_rate=A2C_A3C_CONFIG['learning_rate'],
        gamma=A2C_A3C_CONFIG['gamma'],
        entropy_coef=A2C_A3C_CONFIG['entropy_coef'],
        value_coef=A2C_A3C_CONFIG['value_coef'],
        max_grad_norm=A2C_A3C_CONFIG['max_grad_norm'],
        hidden_units=A2C_A3C_CONFIG['hidden_units'],
        seed=CONST_DEFAULT_SEED
    )
    
    # Crear wrapper del agente para compatibilidad con sistema de model creators
    wrapper = A2CWrapper(
        a2c_agent=a2c_agent,
        cgm_shape=cgm_shape,
        other_features_shape=other_features_shape
    )
    
    # Usar DRLModelWrapper en lugar de DLModelWrapper
    return DRLModelWrapper(lambda **kwargs: wrapper, framework="jax", algorithm="a2c")

def model_creator_a2c() -> Callable[[Tuple[int, ...], Tuple[int, ...]], DRLModelWrapper]:
    """
    Retorna una función para crear un modelo A2C compatible con la API del sistema.
    
    Retorna:
    --------
    Callable[[Tuple[int, ...], Tuple[int, ...]], DRLModelWrapper]
        Función para crear el modelo con las formas de entrada especificadas
    """
    return create_a2c_model

def create_a3c_model(cgm_shape: Tuple[int, ...], other_features_shape: Tuple[int, ...]) -> DRLModelWrapper:
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
    DRLModelWrapper
        Wrapper del modelo A3C compatible con la interfaz del sistema
    """
    # Configurar el espacio de estados y acciones
    state_dim = 64  # Dimensión del espacio de estado latente
    action_dim = 1  # Una dimensión para dosis continua
    continuous = True  # Usar espacio de acción continuo
    n_workers = A2C_A3C_CONFIG.get('n_workers', 4)  # Número de workers para entrenamiento asíncrono
    
    # Crear agente A3C con configuración específica
    a3c_agent = A3C(
        state_dim=state_dim,
        action_dim=action_dim,
        continuous=continuous,
        n_workers=n_workers,
        learning_rate=A2C_A3C_CONFIG['learning_rate'],
        gamma=A2C_A3C_CONFIG['gamma'],
        entropy_coef=A2C_A3C_CONFIG['entropy_coef'],
        value_coef=A2C_A3C_CONFIG['value_coef'],
        max_grad_norm=A2C_A3C_CONFIG['max_grad_norm'],
        hidden_units=A2C_A3C_CONFIG['hidden_units'],
        seed=CONST_DEFAULT_SEED
    )
    
    # Crear wrapper del agente para compatibilidad con sistema de model creators
    wrapper = A3CWrapper(
        a3c_agent=a3c_agent,
        cgm_shape=cgm_shape,
        other_features_shape=other_features_shape
    )
    
    # Envolver en DRLModelWrapper para compatibilidad total con el sistema
    return DRLModelWrapper(lambda **kwargs: wrapper, algorithm="a3c")

def model_creator_a3c() -> Callable[[Tuple[int, ...], Tuple[int, ...]], DRLModelWrapper]:
    """
    Retorna una función para crear un modelo A3C compatible con la API del sistema.
    
    Retorna:
    --------
    Callable[[Tuple[int, ...], Tuple[int, ...]], DRLModelWrapper]
        Función para crear el modelo con las formas de entrada especificadas
    """
    return create_a3c_model