import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Any, Optional, Callable, Union
from collections import deque
from types import SimpleNamespace

from models.early_stopping import get_early_stopping_config

PROJECT_ROOT = os.path.abspath(os.getcwd())
sys.path.append(PROJECT_ROOT) 

from config.models_config import DDPG_CONFIG, EARLY_STOPPING_POLICY
from custom.drl_model_wrapper import DRLModelWrapperPyTorch

# Constantes para uso repetido
CONST_RELU = "relu"
CONST_TANH = "tanh"
CONST_LEAKY_RELU = "leaky_relu"
CONST_GELU = "gelu"
CONST_ACTOR_PREFIX = "actor"
CONST_CRITIC_PREFIX = "critic"
CONST_DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

FIGURES_DIR = os.path.join(PROJECT_ROOT, "figures", "pytorch", "ddpg")
os.makedirs(FIGURES_DIR, exist_ok=True)


class ReplayBuffer:
    """
    Buffer de experiencias para el algoritmo DDPG.
    
    Almacena transiciones (estado, acción, recompensa, estado siguiente, done)
    y permite muestrear lotes de manera aleatoria para el entrenamiento.
    
    Parámetros:
    -----------
    capacity : int, opcional
        Capacidad máxima del buffer (default: 100000)
    """
    def __init__(self, capacity: int = 100000) -> None:
        self.buffer = deque(maxlen=capacity)
        self.rng = np.random.Generator(np.random.PCG64(42))
    
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
    
    def sample(self, batch_size: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Muestrea un lote aleatorio de transiciones.
        
        Parámetros:
        -----------
        batch_size : int
            Tamaño del lote a muestrear
            
        Retorna:
        --------
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]
            (states, actions, rewards, next_states, dones)
        """
        if len(self.buffer) < batch_size:
            batch_size = len(self.buffer)
            
        indices = self.rng.choice(len(self.buffer), size=batch_size, replace=False)
        batch = [self.buffer[idx] for idx in indices]
            
        states, actions, rewards, next_states, dones = zip(*batch)
        
        return (
            torch.FloatTensor(np.array(states)).to(CONST_DEVICE),
            torch.FloatTensor(np.array(actions)).to(CONST_DEVICE),
            torch.FloatTensor(np.array(rewards).reshape(-1, 1)).to(CONST_DEVICE),
            torch.FloatTensor(np.array(next_states)).to(CONST_DEVICE),
            torch.FloatTensor(np.array(dones).reshape(-1, 1)).to(CONST_DEVICE)
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
                dt: float = 1e-2, x_initial: Optional[np.ndarray] = None, seed: int = 42) -> None:
        self.theta = theta
        self.mean = mean
        self.std_dev = std_deviation
        self.dt = dt
        self.x_initial = x_initial
        self.rng = np.random.Generator(np.random.PCG64(seed))
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
        noise = self.rng.normal(size=self.mean.shape)
        x = (
            self.x_prev
            + self.theta * (self.mean - self.x_prev) * self.dt
            + self.std_dev * np.sqrt(self.dt) * noise
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


class ActorNetwork(nn.Module):
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
    activation : str, opcional
        Función de activación a utilizar (default: "relu")
    """
    def __init__(self, state_dim: int, action_dim: int, action_high: np.ndarray, 
                action_low: np.ndarray, hidden_units: Optional[List[int]] = None,
                activation: str = CONST_RELU) -> None:
        super(ActorNetwork, self).__init__()
        
        # Valores predeterminados para capas ocultas
        if hidden_units is None:
            hidden_units = DDPG_CONFIG['actor_hidden_units']
        
        self.action_high = torch.FloatTensor(action_high).to(CONST_DEVICE)
        self.action_low = torch.FloatTensor(action_low).to(CONST_DEVICE)
        self.action_range = self.action_high - self.action_low
        
        # Inicializar layers
        self.layers = nn.ModuleList()
        
        # Primera capa desde state_dim
        self.layers.append(nn.Linear(state_dim, hidden_units[0]))
        
        # Capas ocultas
        for i in range(len(hidden_units) - 1):
            self.layers.append(nn.Linear(hidden_units[i], hidden_units[i + 1]))
        
        # Capa de salida
        self.output_layer = nn.Linear(hidden_units[-1], action_dim)
        
        # Almacenar activación
        self.activation_name = activation
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Pasa la entrada por la red del actor.
        
        Parámetros:
        -----------
        x : torch.Tensor
            Estado de entrada
            
        Retorna:
        --------
        torch.Tensor
            Acción determinística
        """
        # Procesar a través de capas ocultas
        for layer in self.layers:
            x = layer(x)
            x = self._get_activation(x)
        
        # Capa de salida con activación tanh y escalado
        raw_actions = torch.tanh(self.output_layer(x))
        
        # Escalar desde [-1, 1] al rango de acción [low, high]
        scaled_actions = 0.5 * (raw_actions + 1.0) * self.action_range + self.action_low
        
        return scaled_actions
    
    def _get_activation(self, x: torch.Tensor) -> torch.Tensor:
        """
        Aplica la función de activación especificada.
        
        Parámetros:
        -----------
        x : torch.Tensor
            Entrada a la función de activación
            
        Retorna:
        --------
        torch.Tensor
            Salida de la función de activación
        """
        if self.activation_name == CONST_RELU:
            return F.relu(x)
        elif self.activation_name == CONST_TANH:
            return torch.tanh(x)
        elif self.activation_name == CONST_LEAKY_RELU:
            return F.leaky_relu(x)
        elif self.activation_name == CONST_GELU:
            return F.gelu(x)
        else:
            return F.relu(x)  # Valor por defecto


class CriticNetwork(nn.Module):
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
    activation : str, opcional
        Función de activación a utilizar (default: "relu")
    """
    def __init__(self, state_dim: int, action_dim: int, 
                hidden_units: Optional[List[int]] = None,
                activation: str = CONST_RELU) -> None:
        super(CriticNetwork, self).__init__()
        
        # Valores predeterminados para capas ocultas
        if hidden_units is None:
            hidden_units = DDPG_CONFIG['critic_hidden_units']
        
        # Almacenar activación
        self.activation_name = activation
        
        # Capas iniciales para procesar el estado
        self.state_layers = nn.ModuleList()
        self.state_layers.append(nn.Linear(state_dim, hidden_units[0]))
        
        # Dimensión del estado procesado
        state_processed_dim = hidden_units[0]
        
        # Dimensión combinada después de unir estado procesado y acción
        combined_dim = state_processed_dim + action_dim
        
        # Capas para procesar la combinación de estado y acción
        self.combined_layers = nn.ModuleList()
        for i in range(1, len(hidden_units)):
            self.combined_layers.append(nn.Linear(combined_dim if i == 1 else hidden_units[i-1], 
                                               hidden_units[i]))
        
        # Capa de salida: valor Q (sin activación)
        self.output_layer = nn.Linear(hidden_units[-1], 1)
    
    def forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """
        Pasa la entrada por la red del crítico.
        
        Parámetros:
        -----------
        state : torch.Tensor
            Estado de entrada
        action : torch.Tensor
            Acción de entrada
            
        Retorna:
        --------
        torch.Tensor
            Valor Q estimado
        """
        # Procesar el estado
        x = state
        for layer in self.state_layers:
            x = layer(x)
            x = self._get_activation(x)
        
        # Combinar estado procesado con acción
        x = torch.cat([x, action], dim=1)
        
        # Procesar a través de capas combinadas
        for layer in self.combined_layers:
            x = layer(x)
            x = self._get_activation(x)
        
        # Capa de salida
        q_value = self.output_layer(x)
        
        return q_value
    
    def _get_activation(self, x: torch.Tensor) -> torch.Tensor:
        """
        Aplica la función de activación especificada.
        
        Parámetros:
        -----------
        x : torch.Tensor
            Entrada a la función de activación
            
        Retorna:
        --------
        torch.Tensor
            Salida de la función de activación
        """
        if self.activation_name == CONST_RELU:
            return F.relu(x)
        elif self.activation_name == CONST_TANH:
            return torch.tanh(x)
        elif self.activation_name == CONST_LEAKY_RELU:
            return F.leaky_relu(x)
        elif self.activation_name == CONST_GELU:
            return F.gelu(x)
        else:
            return F.relu(x)  # Valor por defecto


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
        seed: int = 42
    ) -> None:
        # Usar configuración proporcionada o por defecto
        self.config = config or DDPG_CONFIG
        
        # Extraer hiperparámetros de la configuración
        actor_lr = self.config.get('actor_lr', DDPG_CONFIG['actor_lr'])
        critic_lr = self.config.get('critic_lr', DDPG_CONFIG['critic_lr'])
        self.gamma = self.config.get('gamma', DDPG_CONFIG['gamma'])
        self.tau = self.config.get('tau', DDPG_CONFIG['tau'])
        buffer_capacity = self.config.get('buffer_capacity', DDPG_CONFIG['buffer_capacity'])
        self.batch_size = self.config.get('batch_size', DDPG_CONFIG['batch_size'])
        noise_std = self.config.get('noise_std', DDPG_CONFIG['noise_std'])
        actor_hidden_units = self.config.get('actor_hidden_units')
        critic_hidden_units = self.config.get('critic_hidden_units')
        actor_activation = self.config.get('actor_activation', DDPG_CONFIG['actor_activation'])
        critic_activation = self.config.get('critic_activation', DDPG_CONFIG['critic_activation'])
        
        # Parámetros del entorno y del modelo
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.action_high = action_high
        self.action_low = action_low
        
        # Configurar semilla para reproducibilidad
        torch.manual_seed(seed)
        np.random.seed(seed)
        
        # Valores predeterminados para capas ocultas
        self.actor_hidden_units = actor_hidden_units or DDPG_CONFIG['actor_hidden_units']
        self.critic_hidden_units = critic_hidden_units or DDPG_CONFIG['critic_hidden_units']
        
        # Crear modelos Actor y Crítico
        self.actor = ActorNetwork(
            state_dim=state_dim,
            action_dim=action_dim,
            action_high=action_high,
            action_low=action_low,
            hidden_units=self.actor_hidden_units,
            activation=actor_activation
        ).to(CONST_DEVICE)
        
        self.critic = CriticNetwork(
            state_dim=state_dim,
            action_dim=action_dim,
            hidden_units=self.critic_hidden_units,
            activation=critic_activation
        ).to(CONST_DEVICE)
        
        # Crear copias target
        self.target_actor = ActorNetwork(
            state_dim=state_dim,
            action_dim=action_dim,
            action_high=action_high,
            action_low=action_low,
            hidden_units=self.actor_hidden_units,
            activation=actor_activation
        ).to(CONST_DEVICE)
        
        self.target_critic = CriticNetwork(
            state_dim=state_dim,
            action_dim=action_dim,
            hidden_units=self.critic_hidden_units,
            activation=critic_activation
        ).to(CONST_DEVICE)
        
        # Inicializar pesos de las redes target con los mismos valores que las principales
        self.target_actor.load_state_dict(self.actor.state_dict())
        self.target_critic.load_state_dict(self.critic.state_dict())
        
        # Optimizadores
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=actor_lr, weight_decay=1e-5)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=critic_lr, weight_decay=1e-5)
        
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
        self.actor_loss_sum = 0.0
        self.critic_loss_sum = 0.0
        self.q_value_sum = 0.0
        self.updates_count = 0

        self.rng = np.random.Generator(np.random.PCG64(seed))
    
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
        for target_param, param in zip(self.target_actor.parameters(), self.actor.parameters()):
            target_param.data.copy_(tau * param.data + (1.0 - tau) * target_param.data)
        
        # Actualizar pesos del crítico target
        for target_param, param in zip(self.target_critic.parameters(), self.critic.parameters()):
            target_param.data.copy_(tau * param.data + (1.0 - tau) * target_param.data)
    
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
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(CONST_DEVICE)
        
        self.actor.eval()
        with torch.no_grad():
            action = self.actor(state_tensor).cpu().numpy()[0]
        self.actor.train()
        
        if add_noise:
            action += self.noise()
            
        # Clipear al rango válido de acciones
        action = np.clip(action, self.action_low, self.action_high)
        
        return action
    
    def train_step(self) -> Tuple[float, float, float]:
        """
        Realiza un paso de entrenamiento con un lote de experiencias del buffer.
        
        Retorna:
        --------
        Tuple[float, float, float]
            (critic_loss, actor_loss, q_value)
        """
        if len(self.replay_buffer) < self.batch_size:
            return 0.0, 0.0, 0.0
            
        # Muestrear un lote del buffer
        states, actions, rewards, next_states, dones = self.replay_buffer.sample(self.batch_size)
        
        # --------------------- #
        # Actualizar el crítico #
        # --------------------- #
        
        # Obtener acciones del estado siguiente usando el actor target
        with torch.no_grad():
            next_actions = self.target_actor(next_states)
            
            # Predecir Q-values target
            target_q_values = self.target_critic(next_states, next_actions)
            
            # Calcular Q-values objetivo usando la ecuación de Bellman
            targets = rewards + (1 - dones) * self.gamma * target_q_values
            
        # Predecir Q-values actuales
        current_q_values = self.critic(states, actions)
        
        # Calcular pérdida del crítico (error cuadrático medio)
        critic_loss = F.mse_loss(current_q_values, targets.detach())
        
        # Reiniciar gradientes y actualizar crítico
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()
        
        # --------------------- #
        # Actualizar el actor   #
        # --------------------- #
        
        # Predecir acciones para los estados actuales
        actor_actions = self.actor(states)
        
        # Calcular pérdida del actor (negativo del Q-value promedio)
        # Queremos maximizar Q-value, así que minimizamos su negativo
        actor_loss = -self.critic(states, actor_actions).mean()
        
        # Reiniciar gradientes y actualizar actor
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()
        
        # Actualizar redes target con soft update
        self.update_target_networks()
        
        # Actualizar métricas acumuladas
        self.actor_loss_sum += float(actor_loss.item())
        self.critic_loss_sum += float(critic_loss.item())
        q_value = float(current_q_values.mean().item())
        self.q_value_sum += q_value
        self.updates_count += 1
        
        return float(critic_loss.item()), float(actor_loss.item()), q_value
    
    def _select_action(self, state: np.ndarray, step_counter: int, warmup_steps: int) -> np.ndarray:
        """
        Selecciona una acción según la política actual, con exploración.
        
        Parámetros:
        -----------
        state : np.ndarray
            Estado actual
        step_counter : int
            Contador de pasos, usado para determinar si explorar con acciones aleatorias
        warmup_steps : int
            Número de pasos iniciales con acciones aleatorias
            
        Retorna:
        --------
        np.ndarray
            Acción seleccionada
        """
        # Durante la fase de calentamiento, usar acciones aleatorias
        if step_counter < warmup_steps:
            return self.rng.uniform(self.action_low, self.action_high, size=self.action_dim)
        
        # Después de la fase de calentamiento, usar la política con ruido
        return self.get_action(state, add_noise=True)

    def _update_model(self, step_counter: int, update_every: int) -> Tuple[Optional[float], Optional[float]]:
        """
        Actualiza el modelo si es momento de hacerlo.
        
        Parámetros:
        -----------
        step_counter : int
            Contador global de pasos
        update_every : int
            Frecuencia de actualización (cada cuántos pasos)
            
        Retorna:
        --------
        Tuple[Optional[float], Optional[float]]
            (actor_loss, critic_loss) si se realizó actualización, None en caso contrario
        """
        if step_counter % update_every != 0:
            return None, None
        
        critic_loss, actor_loss, _ = self.train_step()
        return actor_loss, critic_loss

    def _update_history(self, history: Dict, episode_reward: float, 
                      episode_actor_loss: List, episode_critic_loss: List) -> Dict:
        """
        Actualiza el historial de entrenamiento con los resultados del episodio.
        
        Parámetros:
        -----------
        history : Dict
            Historial acumulado hasta el momento
        episode_reward : float
            Recompensa obtenida en el episodio actual
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
        
        if episode_actor_loss:
            avg_actor_loss = sum(episode_actor_loss) / len(episode_actor_loss)
            history['actor_losses'].append(avg_actor_loss)
        else:
            history['actor_losses'].append(float('nan'))
        
        if episode_critic_loss:
            avg_critic_loss = sum(episode_critic_loss) / len(episode_critic_loss)
            history['critic_losses'].append(avg_critic_loss)
        else:
            history['critic_losses'].append(float('nan'))
        
        # Calcular Q-values promedio si hay actualizaciones
        if self.updates_count > 0:
            avg_q = self.q_value_sum / self.updates_count
            history['avg_q_values'].append(avg_q)
            
            # Reiniciar contadores para la siguiente época
            self.actor_loss_sum = 0.0
            self.critic_loss_sum = 0.0
            self.q_value_sum = 0.0
            self.updates_count = 0
        else:
            history['avg_q_values'].append(float('nan'))
        
        return history
    
    def _log_progress(self, episode: int, episodes: int, episode_reward_history: List, 
                    history: Dict, log_interval: int, best_reward: float) -> float:
        """
        Registra y muestra el progreso del entrenamiento.
        
        Parámetros:
        -----------
        episode : int
            Episodio actual
        episodes : int
            Número total de episodios
        episode_reward_history : List
            Historial de recompensas de episodios recientes
        history : Dict
            Historial completo de entrenamiento
        log_interval : int
            Intervalo para mostrar información
        best_reward : float
            Mejor recompensa obtenida hasta el momento
            
        Retorna:
        --------
        float
            Mejor recompensa actualizada
        """
        if (episode + 1) % log_interval == 0:
            avg_reward = sum(episode_reward_history) / len(episode_reward_history)
            
            # Actualizar mejor recompensa si corresponde
            new_best = ""
            if avg_reward > best_reward:
                best_reward = avg_reward
                new_best = " (nueva mejor)"
            
            # Obtener últimas pérdidas si están disponibles
            recent_actors = [loss for loss in history['actor_losses'][-log_interval:] if not np.isnan(loss)]
            recent_critics = [loss for loss in history['critic_losses'][-log_interval:] if not np.isnan(loss)]
            
            avg_actor_loss = sum(recent_actors) / len(recent_actors) if recent_actors else float('nan')
            avg_critic_loss = sum(recent_critics) / len(recent_critics) if recent_critics else float('nan')
            
            print(f"Episodio {episode+1}/{episodes} - "
                      f"Recompensa promedio: {avg_reward:.2f}{new_best} - "
                      f"Actor loss: {avg_actor_loss:.4f} - "
                      f"Critic loss: {avg_critic_loss:.4f}")
            
        return best_reward

    def _run_episode(self, env, max_steps: int, step_counter: int, warmup_steps: int, 
                   update_every: int, render: bool) -> Tuple[float, List[float], List[float], int]:
        """
        Ejecuta un episodio completo de entrenamiento.
        
        Parámetros:
        -----------
        env : Any
            Entorno de entrenamiento
        max_steps : int
            Número máximo de pasos por episodio
        step_counter : int
            Contador global de pasos
        warmup_steps : int
            Pasos iniciales con acciones aleatorias
        update_every : int
            Frecuencia de actualización de la red
        render : bool
            Si renderizar el entorno durante el entrenamiento
            
        Retorna:
        --------
        Tuple[float, List[float], List[float], int]
            (recompensa_episodio, pérdidas_actor, pérdidas_crítico, contador_pasos)
        """
        state, _ = env.reset()
        state = np.array(state, dtype=np.float32)
        episode_reward = 0.0
        episode_actor_loss = []
        episode_critic_loss = []
        
        for _ in range(max_steps):
            step_counter += 1
            
            # Seleccionar acción
            action = self._select_action(state, step_counter, warmup_steps)
            
            # Ejecutar acción
            next_state, reward, done, truncated, _ = env.step(action)
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
                    episode_actor_loss.append(actor_loss)
                    episode_critic_loss.append(critic_loss)
            
            if done or truncated:
                break
                
        return episode_reward, episode_actor_loss, episode_critic_loss, step_counter
    
    def train(self, env, episodes: int = 1000, max_steps: int = 1000, warmup_steps: int = 1000, 
             update_every: int = 1, render: bool = False, log_interval: int = 10) -> Dict:
        """
        Entrena el agente DDPG en el entorno proporcionado.
        
        Parámetros:
        -----------
        env : Any
            Entorno de entrenamiento
        episodes : int, opcional
            Número de episodios (default: 1000)
        max_steps : int, opcional
            Pasos máximos por episodio (default: 1000)
        warmup_steps : int, opcional
            Pasos iniciales con acciones aleatorias (default: 1000)
        update_every : int, opcional
            Frecuencia de actualización de la red (default: 1)
        render : bool, opcional
            Si renderizar el entorno durante el entrenamiento (default: False)
        log_interval : int, opcional
            Intervalo para mostrar información (default: 10)
            
        Retorna:
        --------
        Dict
            Historial de entrenamiento con métricas
        """
        history = {
            'episode_rewards': [],
            'actor_losses': [],
            'critic_losses': [],
            'avg_q_values': []
        }
        
        episode_reward_history = []
        step_counter = 0
        best_reward = float('-inf')
        
        for episode in range(episodes):
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
        Evalúa el agente DDPG en el entorno proporcionado.
        
        Parámetros:
        -----------
        env : Any
            Entorno de evaluación
        episodes : int, opcional
            Número de episodios para evaluación (default: 10)
        render : bool, opcional
            Si renderizar el entorno durante evaluación (default: False)
            
        Retorna:
        --------
        float
            Recompensa promedio de evaluación
        """
        rewards = []
        
        for episode in range(episodes):
            state, _ = env.reset()
            state = np.array(state, dtype=np.float32)
            episode_reward = 0.0
            done = False
            truncated = False
            
            while not (done or truncated):
                # Obtener acción determinística (sin ruido)
                action = self.get_action(state, add_noise=False)
                
                # Ejecutar acción
                next_state, reward, done, truncated, _ = env.step(action)
                next_state = np.array(next_state, dtype=np.float32)
                
                # Renderizar si es necesario
                if render:
                    env.render()
                
                # Actualizar estado y recompensa
                state = next_state
                episode_reward += reward
            
            rewards.append(episode_reward)
            print(f"Episodio de Evaluación {episode+1}/{episodes} - Recompensa: {episode_reward:.2f}")
        
        avg_reward = sum(rewards) / len(rewards)
        print(f"Recompensa Promedio de Evaluación: {avg_reward:.2f}")
        
        return avg_reward
    
    def save_models(self, actor_path: str, critic_path: str) -> None:
        """
        Guarda los modelos entrenados en disco.
        
        Parámetros:
        -----------
        actor_path : str
            Ruta donde guardar el modelo del actor
        critic_path : str
            Ruta donde guardar el modelo del crítico
        """
        torch.save(self.actor.state_dict(), actor_path)
        torch.save(self.critic.state_dict(), critic_path)
        print(f"Modelos guardados en {actor_path} y {critic_path}")
    
    def load_models(self, actor_path: str, critic_path: str) -> None:
        """
        Carga los modelos desde disco.
        
        Parámetros:
        -----------
        actor_path : str
            Ruta desde donde cargar el modelo del actor
        critic_path : str
            Ruta desde donde cargar el modelo del crítico
        """
        self.actor.load_state_dict(torch.load(actor_path, map_location=CONST_DEVICE))
        self.target_actor.load_state_dict(torch.load(actor_path, map_location=CONST_DEVICE))
        self.critic.load_state_dict(torch.load(critic_path, map_location=CONST_DEVICE))
        self.target_critic.load_state_dict(torch.load(critic_path, map_location=CONST_DEVICE))
        print(f"Modelos cargados desde {actor_path} y {critic_path}")
    
    def visualize_training(self, history: Dict, window_size: int = 10) -> None:
        """
        Visualiza los resultados del entrenamiento.
        
        Parámetros:
        -----------
        history : Dict
            Historial de entrenamiento con métricas
        window_size : int, opcional
            Tamaño de la ventana para suavizado (default: 10)
        """
        
        def smooth(data, window_size):
            """Aplica suavizado a los datos para mejor visualización."""
            if len(data) < window_size:
                return data
            kern = np.ones(window_size) / window_size
            return np.convolve(data, kern, mode='valid')
        
        # Crear directorio para guardar figuras si no existe
        os.makedirs(FIGURES_DIR, exist_ok=True)
        
        _, axs = plt.subplots(2, 2, figsize=(14, 10))
        
        # Graficar recompensas
        rewards = history['episode_rewards']
        axs[0, 0].plot(rewards, alpha=0.3, color='blue', label='Raw')
        if len(rewards) > window_size:
            smooth_rewards = smooth(rewards, window_size)
            axs[0, 0].plot(range(window_size-1, len(rewards)), 
                         smooth_rewards, 
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
                smooth_losses = smooth(valid_losses, window_size)
                axs[0, 1].plot(range(window_size-1, len(valid_losses)),
                             smooth_losses,
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
                smooth_losses = smooth(valid_losses, window_size)
                axs[1, 0].plot(range(window_size-1, len(valid_losses)),
                             smooth_losses,
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
                smooth_q = smooth(valid_q, window_size)
                axs[1, 1].plot(range(window_size-1, len(valid_q)),
                             smooth_q,
                             color='purple', label=f'Smoothed (window={window_size})')
        axs[1, 1].set_title('Average Q Values')
        axs[1, 1].set_xlabel('Episode')
        axs[1, 1].set_ylabel('Q Value')
        axs[1, 1].legend()
        axs[1, 1].grid(alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(FIGURES_DIR, 'training_results.png'))
        plt.show()


class DDPGWrapper(nn.Module):
    """
    Wrapper para el agente DDPG que implementa la interfaz compatible con el sistema.
    
    Parámetros:
    -----------
    ddpg_agent : DDPG
        Agente DDPG inicializado
    cgm_shape : Tuple[int, ...]
        Forma de los datos CGM
    other_features_shape : Tuple[int, ...]
        Forma de otras características
    """
    def __init__(self, ddpg_agent: DDPG, cgm_shape: Tuple[int, ...], other_features_shape: Tuple[int, ...]) -> None:
        super(DDPGWrapper, self).__init__()
        self.ddpg_agent = ddpg_agent
        self.cgm_shape = cgm_shape
        self.other_features_shape = other_features_shape
        self._setup_encoders()
    
    def _setup_encoders(self) -> None:
        """
        Configura los codificadores para proyectar datos de entrada al espacio de estados.
        """
        # Dimensiones de entrada
        cgm_dim = int(np.prod(self.cgm_shape))
        other_dim = int(np.prod(self.other_features_shape))
        
        # Crear codificadores como matrices de proyección lineales
        self.cgm_encoder = nn.Linear(cgm_dim, self.ddpg_agent.state_dim // 2).to(CONST_DEVICE)
        self.other_encoder = nn.Linear(other_dim, self.ddpg_agent.state_dim // 2).to(CONST_DEVICE)
        
        # Inicializar pesos
        torch.nn.init.xavier_uniform_(self.cgm_encoder.weight)
        torch.nn.init.xavier_uniform_(self.other_encoder.weight)
    
    def forward(self, cgm_input: torch.Tensor, other_input: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for the model.
        
        Parámetros:
        -----------
        cgm_input : torch.Tensor
            Datos CGM de entrada
        other_input : torch.Tensor
            Otras características de entrada
            
        Retorna:
        --------
        torch.Tensor
            Acciones predichas
        """
        # Aplanar entradas si es necesario
        batch_size = cgm_input.shape[0]
        cgm_flat = cgm_input.reshape(batch_size, -1)
        other_flat = other_input.reshape(batch_size, -1)
        
        # Codificar entradas
        cgm_encoded = self.cgm_encoder(cgm_flat)
        other_encoded = self.other_encoder(other_flat)
        
        # Combinar características codificadas
        state = torch.cat([cgm_encoded, other_encoded], dim=1)
        
        # Ensure the actor network is in training mode and directly use it
        # rather than going through the get_action method that uses no_grad()
        self.ddpg_agent.actor.train()
        actions = self.ddpg_agent.actor(state)
        
        return actions
    
    def __call__(self, cgm_input: torch.Tensor, other_input: torch.Tensor) -> torch.Tensor:
        """
        Método de llamada directa para compatibilidad con PyTorch.
        
        Parámetros:
        -----------
        cgm_input : torch.Tensor
            Datos CGM de entrada
        other_input : torch.Tensor
            Otras características de entrada
            
        Retorna:
        --------
        torch.Tensor
            Acciones predichas
        """
        # Forward to parent class __call__ which handles module's forward
        return nn.Module.__call__(self, cgm_input, other_input)
    
    # When we need to predict without gradient tracking, use this method
    def predict(self, inputs: List[torch.Tensor]) -> np.ndarray:
        """
        Realiza predicciones con el modelo sin trackear gradientes.
        
        Parámetros:
        -----------
        inputs : List[torch.Tensor]
            Lista con [cgm_data, other_data]
            
        Retorna:
        --------
        np.ndarray
            Predicciones (acciones)
        """
        cgm_data, other_data = inputs
        
        if not isinstance(cgm_data, torch.Tensor):
            cgm_data = torch.FloatTensor(cgm_data).to(CONST_DEVICE)
            other_data = torch.FloatTensor(other_data).to(CONST_DEVICE)
        
        with torch.no_grad():
            self.ddpg_agent.actor.eval()  # Set actor to evaluation mode
            actions = self.forward(cgm_data, other_data)
            self.ddpg_agent.actor.train() # Restore training mode after prediction
            
        return actions.cpu().numpy()
    
    def fit(
        self, 
        x: List[torch.Tensor], 
        y: torch.Tensor, 
        validation_data: Optional[Tuple] = None, 
        epochs: int = 1,
        batch_size: int = 32,
        callbacks: List = None,
        verbose: int = 0
    ) -> Dict:
        """
        Entrena el modelo DDPG con los datos proporcionados.
        
        Parámetros:
        -----------
        x : List[torch.Tensor]
            Lista con [cgm_data, other_data]
        y : torch.Tensor
            Valores objetivo (acciones)
        validation_data : Optional[Tuple], opcional
            Datos de validación como ([cgm_val, other_val], y_val) (default: None)
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
            Historial de entrenamiento
        """
        # Extraer datos de entrenamiento
        x_cgm, x_other = x
        
        # Crear entorno de entrenamiento para el agente DDPG
        env = self._create_training_environment(x_cgm, x_other, y)
        
        # Entrenar el agente DDPG
        history = self.ddpg_agent.train(
            env, 
            episodes=epochs,
            max_steps=batch_size,
            warmup_steps=min(500, batch_size * 5),
            update_every=1,
            render=False,
            log_interval=max(1, epochs // 10) if verbose > 0 else epochs + 1
        )
        
        return history
    
    def _create_training_environment(
        self, 
        cgm_data: torch.Tensor, 
        other_features: torch.Tensor, 
        targets: torch.Tensor
    ) -> Any:
        """
        Crea un entorno de entrenamiento para el agente DDPG.
        
        Parámetros:
        -----------
        cgm_data : torch.Tensor
            Datos CGM
        other_features : torch.Tensor
            Otras características
        targets : torch.Tensor
            Valores objetivo (dosis)
            
        Retorna:
        --------
        Any
            Entorno de entrenamiento compatible con gym
        """
        # Convertir tensores a numpy si es necesario
        cgm_np = cgm_data.cpu().numpy() if hasattr(cgm_data, 'cpu') else cgm_data
        other_np = other_features.cpu().numpy() if hasattr(other_features, 'cpu') else other_features
        targets_np = targets.cpu().numpy() if hasattr(targets, 'cpu') else targets
        
        # Si los objetivos son 2D, aplanarlos
        if len(targets_np.shape) > 1:
            targets_np = targets_np.reshape(-1)
        
        # Obtener referencias a los codificadores para uso en el entorno
        cgm_encoder = self.cgm_encoder
        other_encoder = self.other_encoder
        
        class DDPGEnv:
            """
            Entorno personalizado para entrenamiento de DDPG.
            
            Implementa una interfaz compatible con gym para entrenar el agente.
            """
            def __init__(self, cgm_data, other_features, targets, cgm_encoder, other_encoder):
                self.cgm_data = cgm_data
                self.other_features = other_features
                self.targets = targets
                self.cgm_encoder = cgm_encoder
                self.other_encoder = other_encoder
                self.current_idx = 0
                self.num_samples = len(targets)
                self.rng = np.random.Generator(np.random.PCG64(42))
                self.action_space = SimpleNamespace(
                    sample=lambda: self.rng.uniform(0, 15, (1,)),
                    n=1,
                    shape=(1,)
                )
                self.observation_space = SimpleNamespace(
                    shape=(self.ddpg_agent.state_dim,)
                )
            
            def reset(self):
                """
                Reinicia el entorno para un nuevo episodio.
                
                Retorna:
                --------
                Tuple[np.ndarray, Dict]
                    Estado inicial e información adicional
                """
                # Seleccionar un punto aleatorio para iniciar
                self.current_idx = self.rng.integers(0, self.num_samples)
                
                # Obtener estado actual
                state = self._get_state(self.current_idx)
                
                return state, {}
            
            def step(self, action):
                """
                Ejecuta una acción en el entorno.
                
                Parámetros:
                -----------
                action : np.ndarray
                    Acción a ejecutar
                    
                Retorna:
                --------
                Tuple[np.ndarray, float, bool, bool, Dict]
                    (estado_siguiente, recompensa, terminado, truncado, info)
                """
                # Calcular error con respecto al objetivo
                true_action = self.targets[self.current_idx]
                error = np.abs(action[0] - true_action)
                
                # Penalizar más la sobredosificación que la subdosificación
                if action[0] > true_action:  # Sobredosis (más peligroso)
                    reward = -10.0 * (error ** 2)
                else:  # Subdosis
                    reward = -5.0 * (error ** 2)
                
                # Limitar recompensa negativa extrema
                reward = max(reward, -100.0)
                
                # Avanzar al siguiente punto
                self.current_idx = (self.current_idx + 1) % self.num_samples
                
                # Obtener siguiente estado
                next_state = self._get_state(self.current_idx)
                
                # Verificar si el episodio ha terminado
                done = False  # En este caso, los episodios no terminan naturalmente
                
                return next_state, reward, done, False, {}
            
            def _get_state(self, idx):
                """
                Obtiene el estado codificado para un índice dado.
                
                Parámetros:
                -----------
                idx : int
                    Índice para obtener el estado
                    
                Retorna:
                --------
                np.ndarray
                    Estado procesado
                """
                # Codificar estado actual
                cgm_sample = torch.FloatTensor(self.cgm_data[idx:idx+1]).reshape(1, -1).to(CONST_DEVICE)
                other_sample = torch.FloatTensor(self.other_features[idx:idx+1]).reshape(1, -1).to(CONST_DEVICE)
                
                with torch.no_grad():
                    encoded_cgm = self.cgm_encoder(cgm_sample)[0].cpu().numpy()
                    encoded_other = self.other_encoder(other_sample)[0].cpu().numpy()
                
                # Combinar características codificadas
                return np.concatenate([encoded_cgm, encoded_other])
            
            def render(self, mode='human'):
                """Placeholder para compatibilidad con gym."""
                pass
        
        return DDPGEnv(cgm_np, other_np, targets_np, cgm_encoder, other_encoder)
    
    def save(self, filepath: str) -> None:
        """
        Guarda el modelo en disco.
        
        Parámetros:
        -----------
        filepath : str
            Ruta base donde guardar el modelo
        """
        # Crear directorios si no existen
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        # Guardar modelos del agente DDPG
        actor_path = f"{filepath}_actor.pt"
        critic_path = f"{filepath}_critic.pt"
        self.ddpg_agent.save_models(actor_path, critic_path)
        
        # Guardar codificadores
        encoders_path = f"{filepath}_encoders.pt"
        torch.save({
            'cgm_encoder': self.cgm_encoder.state_dict(),
            'other_encoder': self.other_encoder.state_dict()
        }, encoders_path)
        
        print(f"Modelo guardado en {filepath}")
    
    def load(self, filepath: str) -> None:
        """
        Carga el modelo desde disco.
        
        Parámetros:
        -----------
        filepath : str
            Ruta base desde donde cargar el modelo
        """
        actor_path = f"{filepath}_actor.pt"
        critic_path = f"{filepath}_critic.pt"
        encoders_path = f"{filepath}_encoders.pt"
        
        # Cargar modelos del agente DDPG
        self.ddpg_agent.load_models(actor_path, critic_path)
        
        # Cargar codificadores
        encoders_state = torch.load(encoders_path, map_location=CONST_DEVICE)
        self.cgm_encoder.load_state_dict(encoders_state['cgm_encoder'])
        self.other_encoder.load_state_dict(encoders_state['other_encoder'])
        
        print(f"Modelo cargado desde {filepath}")
    
    def get_config(self) -> Dict:
        """
        Retorna la configuración del modelo.
        
        Retorna:
        --------
        Dict
            Configuración del modelo
        """
        return {
            'state_dim': self.ddpg_agent.state_dim,
            'action_dim': self.ddpg_agent.action_dim,
            'action_high': self.ddpg_agent.action_high.tolist(),
            'action_low': self.ddpg_agent.action_low.tolist(),
            'cgm_shape': self.cgm_shape,
            'other_features_shape': self.other_features_shape,
            'config': self.ddpg_agent.config
        }


def create_ddpg_model(cgm_shape: Tuple[int, ...], other_features_shape: Tuple[int, ...]) -> DRLModelWrapperPyTorch:
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
    DRLModelWrapperPyTorch
        Wrapper de DDPG que implementa la interfaz compatible
    """
    # Configurar el espacio de estados y acciones
    state_dim = 64  # Dimensión del espacio de estado latente
    action_dim = 1  # Una dimensión para la dosis continua
    
    # Límites de acción para dosis de insulina (0-15 unidades)
    action_high = np.array([15.0])  # Máximo 15 unidades de insulina
    action_low = np.array([0.0])    # Mínimo 0 unidades
    
    # Crear configuración para el agente DDPG
    config = {
        'actor_lr': DDPG_CONFIG['actor_lr'],
        'critic_lr': DDPG_CONFIG['critic_lr'],
        'gamma': DDPG_CONFIG['gamma'],
        'tau': DDPG_CONFIG['tau'],
        'batch_size': min(DDPG_CONFIG['batch_size'], 64),  # Adaptado para este problema
        'buffer_capacity': DDPG_CONFIG['buffer_capacity'],
        'noise_std': DDPG_CONFIG['noise_std'],
        'actor_hidden_units': DDPG_CONFIG['actor_hidden_units'],
        'critic_hidden_units': DDPG_CONFIG['critic_hidden_units'],
        'actor_activation': DDPG_CONFIG['actor_activation'],
        'critic_activation': DDPG_CONFIG['critic_activation'],
        'dropout_rate': DDPG_CONFIG['dropout_rate'],
        'epsilon': DDPG_CONFIG['epsilon']
    }

    # Crear agente DDPG
    ddpg_agent = DDPG(
        state_dim=state_dim,
        action_dim=action_dim,
        action_high=action_high,
        action_low=action_low,
        config=config,
        seed=42
    )

    # Crear wrapper para DDPG
    ddpg_wrapper = DDPGWrapper(
        ddpg_agent=ddpg_agent,
        cgm_shape=cgm_shape,
        other_features_shape=other_features_shape
    )

    # Devolver el wrapper compatible con DRLModelWrapperPyTorch
    return ddpg_wrapper

def create_ddpg_model_wrapper(cgm_shape: Tuple[int, ...], other_features_shape: Tuple[int, ...]) -> DRLModelWrapperPyTorch:
    """
    Crea un modelo DDPG envuelto en DRLModelWrapperPyTorch.
    
    Parámetros:
    -----------
    cgm_shape : Tuple[int, ...]
        Forma de los datos CGM (pasos_temporales, características)
    other_features_shape : Tuple[int, ...]
        Forma de otras características (características)
        
    Retorna:
    --------
    DRLModelWrapperPyTorch
        Modelo envuelto en DRLModelWrapperPyTorch para compatibilidad con el sistema
    """
    # Define critical parameters for the model
    state_dim = 64  # Dimensión del espacio de estado latente
    action_dim = 1  # Una dimensión para la dosis continua
    action_high = np.array([15.0])
    action_low = np.array([0.0])
    
    # Model creator function that creates the DDPG model
    def model_creator(**kwargs) -> nn.Module:
        """
        Creates a DDPG model instance.
        
        Returns:
        --------
        nn.Module
            DDPG model instance
        """
        # Create configuration for DDPG agent
        config = {
            'actor_lr': DDPG_CONFIG['actor_lr'],
            'critic_lr': DDPG_CONFIG['critic_lr'],
            'gamma': DDPG_CONFIG['gamma'],
            'tau': DDPG_CONFIG['tau'],
            'batch_size': min(DDPG_CONFIG['batch_size'], 64),
            'buffer_capacity': DDPG_CONFIG['buffer_capacity'],
            'noise_std': DDPG_CONFIG['noise_std'],
            'actor_hidden_units': DDPG_CONFIG['actor_hidden_units'],
            'critic_hidden_units': DDPG_CONFIG['critic_hidden_units'],
            'actor_activation': DDPG_CONFIG['actor_activation'],
            'critic_activation': DDPG_CONFIG['critic_activation'],
            'dropout_rate': DDPG_CONFIG['dropout_rate'],
            'epsilon': DDPG_CONFIG['epsilon']
        }
        
        # Create DDPG agent
        ddpg_agent = DDPG(
            state_dim=state_dim,
            action_dim=action_dim,
            action_high=action_high,
            action_low=action_low,
            config=config,
            seed=42
        )
        
        # Create and return the wrapper model (important: inherits from nn.Module)
        return DDPGWrapper(
            ddpg_agent=ddpg_agent,
            cgm_shape=cgm_shape,
            other_features_shape=other_features_shape
        )
    
    # Create the DRLModelWrapperPyTorch with our model creator function
    wrapper = DRLModelWrapperPyTorch(model_creator, algorithm='ddpg')
    
    # Add early stopping if enabled
    es_patience, es_min_delta, es_restore_best = get_early_stopping_config()
    if EARLY_STOPPING_POLICY.get('early_stopping', False):
        wrapper.add_early_stopping(
            patience=es_patience,
            min_delta=es_min_delta,
            restore_best_weights=es_restore_best
        )
    
    return wrapper

def model_creator() -> Callable[[Tuple[int, ...], Tuple[int, ...]], DRLModelWrapperPyTorch]:
    """
    Retorna una función para crear un modelo DDPG compatible con la API del sistema.
    
    Retorna:
    --------
    Callable[[Tuple[int, ...], Tuple[int, ...]], DRLModelWrapperPyTorch]
        Función para crear el modelo con las formas de entrada especificadas
    """
    return create_ddpg_model_wrapper