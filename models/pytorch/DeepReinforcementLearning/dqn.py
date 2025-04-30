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

from config.models_config import DQN_CONFIG, EARLY_STOPPING_POLICY
from constants.constants import CONST_DEFAULT_SEED
from custom.drl_model_wrapper import DRLModelWrapperPyTorch

# Constantes para uso repetido
CONST_RELU = "relu"
CONST_TANH = "tanh"
CONST_LEAKY_RELU = "leaky_relu"
CONST_GELU = "gelu"
CONST_DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CONST_LOSS = "loss"
CONST_Q_VALUE = "q_value"
CONST_TRAINING = "training"
CONST_FIGURES_DIR = os.path.join(PROJECT_ROOT, "figures", "pytorch", "dqn")

# Asegurar que existe el directorio para figuras
os.makedirs(CONST_FIGURES_DIR, exist_ok=True)


class ReplayBuffer:
    """
    Buffer de experiencias para el algoritmo DQN.
    
    Almacena transiciones (estado, acción, recompensa, estado siguiente, done)
    y permite muestrear lotes de manera aleatoria para el entrenamiento.
    
    Parámetros:
    -----------
    capacity : int, opcional
        Capacidad máxima del buffer (default: 10000)
    """
    def __init__(self, capacity: int = 10000) -> None:
        self.buffer = deque(maxlen=capacity)
        self.rng = np.random.Generator(np.random.PCG64(CONST_DEFAULT_SEED))
    
    def add(self, state: np.ndarray, action: int, reward: float, 
           next_state: np.ndarray, done: float) -> None:
        """
        Añade una transición al buffer.
        
        Parámetros:
        -----------
        state : np.ndarray
            Estado actual
        action : int
            Acción tomada
        reward : float
            Recompensa recibida
        next_state : np.ndarray
            Estado siguiente
        done : float
            Indicador de fin de episodio (1.0 si terminó, 0.0 si no)
        """
        self.buffer.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, 
                                             torch.Tensor, torch.Tensor]:
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
        batch = [self.buffer[i] for i in indices]
        
        states, actions, rewards, next_states, dones = zip(*batch)
        
        return (
            torch.FloatTensor(np.array(states)).to(CONST_DEVICE),
            torch.LongTensor(np.array(actions)).to(CONST_DEVICE),
            torch.FloatTensor(np.array(rewards)).to(CONST_DEVICE),
            torch.FloatTensor(np.array(next_states)).to(CONST_DEVICE),
            torch.FloatTensor(np.array(dones)).to(CONST_DEVICE)
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


class PrioritizedReplayBuffer(ReplayBuffer):
    """
    Buffer de experiencias con muestreo prioritario.
    
    Prioriza experiencias basadas en el TD error para muestreo eficiente.
    
    Parámetros:
    -----------
    capacity : int
        Capacidad máxima del buffer
    """
    def __init__(self, capacity: int) -> None:
        super().__init__(capacity)
        self.priorities = np.zeros(capacity, dtype=np.float32) + 1e-5
        self.pos = 0
    
    def add(self, state: np.ndarray, action: int, reward: float, 
           next_state: np.ndarray, done: float) -> None:
        """
        Añade una transición al buffer con prioridad máxima.
        
        Parámetros:
        -----------
        state : np.ndarray
            Estado actual
        action : int
            Acción tomada
        reward : float
            Recompensa recibida
        next_state : np.ndarray
            Estado siguiente
        done : float
            Indicador de fin de episodio (1.0 si terminó, 0.0 si no)
        """
        max_priority = np.max(self.priorities) if len(self.buffer) > 0 else 1.0
        
        if len(self.buffer) < self.buffer.maxlen:
            self.buffer.append((state, action, reward, next_state, done))
        else:
            self.buffer[self.pos] = (state, action, reward, next_state, done)
        
        self.priorities[self.pos] = max_priority
        self.pos = (self.pos + 1) % self.buffer.maxlen
    
    def sample(self, batch_size: int, beta: float = 0.4) -> Tuple[torch.Tensor, torch.Tensor, 
                                                                torch.Tensor, torch.Tensor, 
                                                                torch.Tensor, List[int], torch.Tensor]:
        """
        Muestrea un lote basado en prioridades.
        
        Parámetros:
        -----------
        batch_size : int
            Tamaño del lote
        beta : float, opcional
            Factor para corrección de sesgo (0-1) (default: 0.4)
            
        Retorna:
        --------
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, List[int], torch.Tensor]
            (states, actions, rewards, next_states, dones, índices, pesos de importancia)
        """
        if len(self.buffer) < batch_size:
            batch_size = len(self.buffer)
        
        buffer_length = len(self.buffer)
        priorities = self.priorities[:buffer_length]
        
        # Calcular probabilidades basadas en prioridad
        probs = priorities / np.sum(priorities)
        indices = self.rng.choice(buffer_length, batch_size, replace=False, p=probs)
        
        # Extraer batch
        states, actions, rewards, next_states, dones = [], [], [], [], []
        for i in indices:
            state, action, reward, next_state, done = self.buffer[i]
            states.append(state)
            actions.append(action)
            rewards.append(reward)
            next_states.append(next_state)
            dones.append(done)
        
        # Calcular pesos de importancia (corrección del sesgo)
        weights = np.zeros(batch_size, dtype=np.float32)
        # Evitar división por cero
        sampled_probs = probs[indices]
        weights = (buffer_length * sampled_probs) ** -beta
        weights /= np.max(weights)  # Normalizar
        
        return (
            torch.FloatTensor(np.array(states)).to(CONST_DEVICE),
            torch.LongTensor(np.array(actions)).to(CONST_DEVICE),
            torch.FloatTensor(np.array(rewards)).to(CONST_DEVICE),
            torch.FloatTensor(np.array(next_states)).to(CONST_DEVICE),
            torch.FloatTensor(np.array(dones)).to(CONST_DEVICE),
            indices,
            torch.FloatTensor(weights).to(CONST_DEVICE)
        )
    
    def update_priorities(self, indices: List[int], priorities: np.ndarray) -> None:
        """
        Actualiza las prioridades para los índices dados.
        
        Parámetros:
        -----------
        indices : List[int]
            Índices de las transiciones a actualizar
        priorities : np.ndarray
            Nuevas prioridades
        """
        for idx, priority in zip(indices, priorities):
            self.priorities[idx] = priority


class QNetwork(nn.Module):
    """
    Red Q para DQN que mapea estados a valores Q.
    
    Implementa una arquitectura flexible para estimación de valores Q-state-action.
    
    Parámetros:
    -----------
    state_dim : int
        Dimensión del espacio de estados
    action_dim : int
        Dimensión del espacio de acciones
    hidden_units : Optional[List[int]], opcional
        Unidades en capas ocultas (default: None)
    dueling : bool, opcional
        Si usar arquitectura dueling (default: False)
    activation : str, opcional
        Función de activación a utilizar (default: 'relu')
    dropout_rate : float, opcional
        Tasa de dropout para regularización (default: 0.0)
    """
    def __init__(self, state_dim: int, action_dim: int, 
                 hidden_units: Optional[List[int]] = None,
                 dueling: bool = False,
                 activation: str = CONST_RELU,
                 dropout_rate: float = 0.0) -> None:
        super(QNetwork, self).__init__()
        
        # Valores predeterminados para capas ocultas
        if hidden_units is None:
            hidden_units = DQN_CONFIG['hidden_units']
        
        self.dueling = dueling
        self.action_dim = action_dim
        self.activation_name = activation
        self.dropout_rate = dropout_rate
        
        # Capas para el procesamiento del estado
        self.feature_layers = nn.ModuleList()
        for i, units in enumerate(hidden_units):
            if i == 0:
                self.feature_layers.append(nn.Linear(state_dim, units))
            else:
                self.feature_layers.append(nn.Linear(hidden_units[i-1], units))
            
            self.feature_layers.append(nn.LayerNorm(units, eps=1e-6))
            self.feature_layers.append(self._get_activation_layer())
            
            if dropout_rate > 0:
                self.feature_layers.append(nn.Dropout(dropout_rate))
        
        if dueling:
            # Arquitectura dueling DQN
            # Valor de estado (V)
            self.value_stream = nn.Sequential(
                nn.Linear(hidden_units[-1], hidden_units[-1] // 2),
                nn.LayerNorm(hidden_units[-1] // 2, eps=1e-6),
                self._get_activation_layer(),
                nn.Linear(hidden_units[-1] // 2, 1)
            )
            
            # Ventaja (A)
            self.advantage_stream = nn.Sequential(
                nn.Linear(hidden_units[-1], hidden_units[-1] // 2),
                nn.LayerNorm(hidden_units[-1] // 2, eps=1e-6),
                self._get_activation_layer(),
                nn.Linear(hidden_units[-1] // 2, action_dim)
            )
        else:
            # DQN estándar
            self.output_layer = nn.Linear(hidden_units[-1], action_dim)
        
        # Enviar al dispositivo disponible
        self.to(CONST_DEVICE)
    
    def _get_activation_layer(self) -> nn.Module:
        """
        Retorna la capa de activación según el nombre especificado.
        
        Retorna:
        --------
        nn.Module
            Capa de activación correspondiente
        """
        if self.activation_name == CONST_RELU:
            return nn.ReLU()
        elif self.activation_name == CONST_TANH:
            return nn.Tanh()
        elif self.activation_name == CONST_LEAKY_RELU:
            return nn.LeakyReLU(0.01)
        elif self.activation_name == CONST_GELU:
            return nn.GELU()
        else:
            return nn.ReLU()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Pasa la entrada por la red Q.
        
        Parámetros:
        -----------
        x : torch.Tensor
            Tensor de entrada (estados)
            
        Retorna:
        --------
        torch.Tensor
            Valores Q para cada acción
        """
        # Procesamiento a través de capas compartidas
        for layer in self.feature_layers:
            x = layer(x)
        
        if self.dueling:
            # Arquitectura dueling
            value = self.value_stream(x)
            advantage = self.advantage_stream(x)
            
            # Combinar V y A: Q = V + (A - mean(A))
            # La resta de la media de ventajas asegura identificabilidad
            q_values = value + (advantage - advantage.mean(dim=1, keepdim=True))
        else:
            # DQN estándar
            q_values = self.output_layer(x)
        
        return q_values
    
    def get_action(self, state: np.ndarray, epsilon: float = 0.0) -> int:
        """
        Obtiene una acción usando la política epsilon-greedy.
        
        Parámetros:
        -----------
        state : np.ndarray
            Estado actual del entorno
        epsilon : float, opcional
            Probabilidad de exploración (0-1) (default: 0.0)
            
        Retorna:
        --------
        int
            Acción seleccionada según la política
        """
        rng = np.random.Generator(np.random.PCG64(CONST_DEFAULT_SEED))
        if rng.random() < epsilon:
            # Exploración: acción aleatoria
            return rng.integers(0, self.action_dim)
        else:
            # Explotación: acción con mayor valor Q
            with torch.no_grad():
                state_tensor = torch.FloatTensor(state).unsqueeze(0).to(CONST_DEVICE)
                q_values = self(state_tensor)
                action = q_values.argmax(dim=1).item()
            return action


class DQN:
    """
    Implementación del algoritmo Deep Q-Network (DQN).
    
    Incluye mecanismos de Experience Replay y Target Network para
    mejorar la estabilidad del aprendizaje.
    
    Parámetros:
    -----------
    state_dim : int
        Dimensión del espacio de estados
    action_dim : int
        Dimensión del espacio de acciones
    config : Optional[Dict[str, Any]], opcional
        Configuración personalizada (default: None)
    hidden_units : Optional[List[int]], opcional
        Unidades en capas ocultas (default: None)
    seed : int, opcional
        Semilla para los generadores de números aleatorios (default: 42)
    """
    def __init__(
        self, 
        state_dim: int, 
        action_dim: int,
        config: Optional[Dict[str, Any]] = None,
        hidden_units: Optional[List[int]] = None,
        seed: int = CONST_DEFAULT_SEED
    ) -> None:
        # Configurar semillas para reproducibilidad
        torch.manual_seed(seed)
        np.random.seed(seed)
        
        # Usar configuración proporcionada o predeterminada
        self.config = config or DQN_CONFIG
        
        # Extraer parámetros de configuración
        learning_rate = self.config.get('learning_rate', 0.001)
        self.gamma = self.config.get('gamma', 0.99)
        self.epsilon_start = self.config.get('epsilon_start', 1.0)
        self.epsilon_end = self.config.get('epsilon_end', 0.01)
        self.epsilon_decay = self.config.get('epsilon_decay', 0.995)
        buffer_capacity = self.config.get('buffer_capacity', 10000)
        self.batch_size = self.config.get('batch_size', 64)
        self.target_update_freq = self.config.get('target_update_freq', 100)
        dueling = self.config.get('dueling', False)
        self.double = self.config.get('double', False)
        self.prioritized = self.config.get('prioritized', False)
        dropout_rate = self.config.get('dropout_rate', 0.0)
        activation = self.config.get('activation', CONST_RELU)
        
        # Parámetros del entorno y del modelo
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.epsilon = self.epsilon_start
        
        # Cantidad de actualizaciones realizadas
        self.update_counter = 0
        
        # Valores predeterminados para capas ocultas
        if hidden_units is None:
            self.hidden_units = DQN_CONFIG['hidden_units']
        else:
            self.hidden_units = hidden_units
        
        # Crear modelo Q y modelo Q Target
        self.q_network = QNetwork(
            state_dim=state_dim, 
            action_dim=action_dim, 
            hidden_units=self.hidden_units, 
            dueling=dueling,
            activation=activation,
            dropout_rate=dropout_rate
        )
        
        self.target_q_network = QNetwork(
            state_dim=state_dim, 
            action_dim=action_dim, 
            hidden_units=self.hidden_units, 
            dueling=dueling,
            activation=activation,
            dropout_rate=dropout_rate
        )
        
        # Inicializar el target network con los mismos pesos
        self.update_target_network()
        
        # Optimizador con weight_decay para reducir sobreajuste
        self.optimizer = optim.Adam(
            self.q_network.parameters(), 
            lr=learning_rate, 
            weight_decay=1e-5
        )
        
        # Buffer de experiencias
        if self.prioritized:
            # Con prioridad de muestreo basada en TD-error
            self.replay_buffer = PrioritizedReplayBuffer(buffer_capacity)
            self.alpha = DQN_CONFIG['priority_alpha']
            self.beta = DQN_CONFIG['priority_beta']
            self.beta_increment = DQN_CONFIG['priority_beta_increment']
        else:
            # Buffer uniforme clásico
            self.replay_buffer = ReplayBuffer(buffer_capacity)
        
        # Métricas acumuladas
        self.loss_sum = 0.0
        self.q_value_sum = 0.0
        self.updates_count = 0
        
        # Generador de números aleatorios moderno
        self.rng = np.random.Generator(np.random.PCG64(seed))
    
    def update_target_network(self) -> None:
        """
        Actualiza los pesos del target network con los del Q-network principal.
        """
        self.target_q_network.load_state_dict(self.q_network.state_dict())
    
    def get_action(self, state: np.ndarray, epsilon: Optional[float] = None) -> int:
        """
        Selecciona una acción siguiendo una política epsilon-greedy.
        
        Parámetros:
        -----------
        state : np.ndarray
            Estado actual del entorno
        epsilon : Optional[float]
            Valor de epsilon a utilizar (si es None, usa el valor actual)
            
        Retorna:
        --------
        int
            Acción seleccionada
        """
        if epsilon is None:
            epsilon = self.epsilon
        
        return self.q_network.get_action(state, epsilon)
    
    def _sample_batch(self) -> Union[Tuple, Tuple[Tuple, List[int], torch.Tensor]]:
        """
        Muestrea un lote de experiencias del buffer.
        
        Retorna:
        --------
        Union[Tuple, Tuple[Tuple, List[int], torch.Tensor]]
            Batch de transiciones y, si se usa PER, índices y pesos
        """
        if self.prioritized:
            # Actualiza beta según el schedule
            self.beta = min(1.0, self.beta + self.beta_increment)
            batch, indices, weights = self.replay_buffer.sample(self.batch_size, self.beta)
            return batch, indices, weights
        else:
            return self.replay_buffer.sample(self.batch_size)
    
    def train_step(self) -> Tuple[float, float]:
        """
        Realiza un paso de entrenamiento utilizando un lote de experiencias.
        
        Retorna:
        --------
        Tuple[float, float]
            (pérdida, q_value promedio)
        """
        if len(self.replay_buffer) < self.batch_size:
            # No hay suficientes muestras para entrenar
            return 0.0, 0.0
        
        # Muestrear batch de experiencias
        if self.prioritized:
            (states, actions, rewards, next_states, dones), indices, importance_weights = self._sample_batch()
        else:
            states, actions, rewards, next_states, dones = self._sample_batch()
            importance_weights = torch.ones_like(rewards).to(CONST_DEVICE)
        
        # Calcular valores Q actuales: Q(s, a)
        q_values = self.q_network(states)
        q_values_for_actions = q_values.gather(1, actions.unsqueeze(1)).squeeze(1)
        
        with torch.no_grad():
            if self.double:
                # Double DQN: Seleccionar acción usando la red principal
                next_actions = self.q_network(next_states).argmax(dim=1, keepdim=True)
                # Estimar el valor usando la red target
                next_q_values = self.target_q_network(next_states).gather(1, next_actions).squeeze(1)
            else:
                # DQN estándar: Estimar valor usando max sobre todas las acciones
                next_q_values = self.target_q_network(next_states).max(dim=1)[0]
            
            # Calcular targets usando ecuación de Bellman
            targets = rewards + (1 - dones) * self.gamma * next_q_values
        
        # Calcular pérdida TD y aplicar pesos de importancia si se usa PER
        td_errors = targets - q_values_for_actions
        
        # Aplicar pesos de importancia
        if self.prioritized:
            # Usar pérdida de Huber ponderada por pesos de importancia
            loss = F.smooth_l1_loss(q_values_for_actions, targets, reduction='none')
            loss = (importance_weights * loss).mean()
        else:
            # Usar pérdida de Huber estándar
            loss = F.smooth_l1_loss(q_values_for_actions, targets)
        
        # Optimización
        self.optimizer.zero_grad()
        loss.backward()
        # Recorte de gradiente para estabilidad
        torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), 1.0)
        self.optimizer.step()
        
        # Actualizar prioridades si se usa PER
        if self.prioritized:
            with torch.no_grad():
                new_priorities = np.abs(td_errors.cpu().numpy()) + 1e-6
            self.replay_buffer.update_priorities(indices, new_priorities)
        
        # Actualizaciones de métricas
        self.loss_sum += loss.item()
        self.q_value_sum += q_values_for_actions.mean().item()
        self.updates_count += 1
        
        # Actualizar target network periódicamente
        self.update_counter += 1
        if self.update_counter % self.target_update_freq == 0:
            self.update_target_network()
        
        return loss.item(), q_values_for_actions.mean().item()
    
    def _update_model(self, episode_loss: List[float], update_every: int, update_after: int) -> None:
        """
        Actualiza el modelo con los datos del buffer de experiencia.
        
        Parámetros:
        -----------
        episode_loss : List[float]
            Lista para almacenar pérdidas durante el episodio
        update_every : int
            Frecuencia de actualización (cada cuántos pasos)
        update_after : int
            Cantidad de pasos antes de empezar a actualizar
        """
        if len(self.replay_buffer) < update_after:
            return
        
        if self.updates_count % update_every == 0:
            loss, _ = self.train_step()
            episode_loss.append(loss)
    
    def _decay_epsilon(self) -> None:
        """
        Actualiza el valor de epsilon según el esquema de decaimiento.
        """
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)
    
    def _run_episode(self, env: Any, max_steps: int, render: bool, 
                   update_every: int, update_after: int) -> Tuple[float, List[float]]:
        """
        Ejecuta un episodio de entrenamiento.
        
        Parámetros:
        -----------
        env : Any
            Entorno de aprendizaje por refuerzo
        max_steps : int
            Máximo número de pasos por episodio
        render : bool
            Si renderizar el entorno durante la ejecución
        update_every : int
            Frecuencia de actualización del modelo
        update_after : int
            Cantidad de pasos antes de empezar a actualizar
            
        Retorna:
        --------
        Tuple[float, List[float]]
            (recompensa total del episodio, lista de pérdidas durante el episodio)
        """
        state, _ = env.reset()
        episode_reward = 0.0
        episode_loss = []
        
        for _ in range(max_steps):
            # Seleccionar acción según política epsilon-greedy
            action = self.get_action(state, self.epsilon)
            
            # Ejecutar acción en el entorno
            next_state, reward, done, truncated, _ = env.step(action)
            
            # Almacenar experiencia en el buffer
            self.replay_buffer.add(state, action, reward, next_state, float(done or truncated))
            
            # Actualizar modelo
            self._update_model(episode_loss, update_every, update_after)
            
            # Mostrar entorno si se solicita
            if render:
                env.render()
            
            # Actualizar estado y recompensa acumulada
            state = next_state
            episode_reward += reward
            
            # Terminar episodio si corresponde
            if done or truncated:
                break
        
        # Actualizar epsilon al final del episodio
        self._decay_epsilon()
        
        return episode_reward, episode_loss
    
    def _update_history(self, history: Dict, episode_reward: float, episode_loss: List[float], 
                      episode_reward_history: List[float], log_interval: int) -> List[float]:
        """
        Actualiza el historial de entrenamiento con los resultados del episodio.
        
        Parámetros:
        -----------
        history : Dict
            Historial de entrenamiento
        episode_reward : float
            Recompensa total del episodio
        episode_loss : List[float]
            Lista de pérdidas durante el episodio
        episode_reward_history : List[float]
            Historial de recompensas para calcular promedios
        log_interval : int
            Intervalo para logs y cálculos
            
        Retorna:
        --------
        List[float]
            Historial de recompensas actualizado
        """
        # Calcular métricas del episodio
        avg_loss = np.mean(episode_loss) if episode_loss else 0.0
        metrics = {
            CONST_LOSS: avg_loss,
        }
        
        # Actualizar métricas acumuladas
        if self.updates_count > 0:
            metrics[CONST_Q_VALUE] = self.q_value_sum / self.updates_count
        
        # Reiniciar métricas acumuladas
        self.loss_sum = 0.0
        self.q_value_sum = 0.0
        self.updates_count = 0
        
        # Registrar en historial
        for key, value in metrics.items():
            if key not in history:
                history[key] = []
            history[key].append(value)
        
        # Añadir recompensa al historial
        if "episode_rewards" not in history:
            history["episode_rewards"] = []
        history["episode_rewards"].append(episode_reward)
        
        # Actualizar historial para promedios móviles
        episode_reward_history.append(episode_reward)
        if len(episode_reward_history) > log_interval:
            episode_reward_history.pop(0)
            
        return episode_reward_history
    
    def train(self, env: Any, episodes: int = 1000, max_steps: int = 1000, 
             update_after: int = 1000, update_every: int = 4, 
             render: bool = False, log_interval: int = 10) -> Dict:
        """
        Entrena el agente DQN en el entorno dado.
        
        Parámetros:
        -----------
        env : Any
            Entorno de aprendizaje por refuerzo
        episodes : int, opcional
            Número de episodios de entrenamiento (default: 1000)
        max_steps : int, opcional
            Máximo número de pasos por episodio (default: 1000)
        update_after : int, opcional
            Cantidad de pasos antes de empezar a actualizar (default: 1000)
        update_every : int, opcional
            Frecuencia de actualización del modelo (default: 4)
        render : bool, opcional
            Si renderizar el entorno durante la ejecución (default: False)
        log_interval : int, opcional
            Intervalo para logs y cálculos (default: 10)
            
        Retorna:
        --------
        Dict
            Historial de entrenamiento con métricas
        """
        history = {}
        episode_reward_history = []
        best_reward = float('-inf')
        
        for episode in range(episodes):
            # Ejecutar episodio
            episode_reward, episode_loss = self._run_episode(
                env, max_steps, render, update_every, update_after
            )
            
            # Actualizar historial
            episode_reward_history = self._update_history(
                history, episode_reward, episode_loss, episode_reward_history, log_interval
            )
            
            # Calcular recompensa promedio móvil
            if len(episode_reward_history) >= log_interval:
                avg_reward = np.mean(episode_reward_history)
                
                # Mostrar progreso
                if (episode + 1) % log_interval == 0:
                    print(f"Episodio {episode + 1}/{episodes} | "
                          f"Epsilon: {self.epsilon:.4f} | "
                          f"Recompensa: {episode_reward:.2f} | "
                          f"Recompensa Promedio: {avg_reward:.2f} | "
                          f"Pérdida: {np.mean(episode_loss) if episode_loss else 0:.6f}")
                
                # Guardar mejor modelo
                if avg_reward > best_reward:
                    best_reward = avg_reward
                    print(f"Nuevo mejor modelo con recompensa: {best_reward:.2f}")
        
        return history
    
    def evaluate(self, env: Any, episodes: int = 10, render: bool = False) -> float:
        """
        Evalúa el agente DQN entrenado.
        
        Parámetros:
        -----------
        env : Any
            Entorno de aprendizaje por refuerzo
        episodes : int, opcional
            Número de episodios para evaluación (default: 10)
        render : bool, opcional
            Si renderizar el entorno durante la evaluación (default: False)
            
        Retorna:
        --------
        float
            Recompensa promedio obtenida
        """
        rewards = []
        
        for _ in range(episodes):
            state, _ = env.reset()
            episode_reward = 0
            done = False
            truncated = False
            
            while not (done or truncated):
                # Seleccionar mejor acción (sin exploración)
                action = self.get_action(state, epsilon=0.0)
                
                # Ejecutar acción en el entorno
                next_state, reward, done, truncated, _ = env.step(action)
                
                if render:
                    env.render()
                
                state = next_state
                episode_reward += reward
            
            rewards.append(episode_reward)
        
        avg_reward = np.mean(rewards)
        print(f"Evaluación: Recompensa promedio = {avg_reward:.2f}")
        
        return avg_reward
    
    def save_model(self, filepath: str) -> None:
        """
        Guarda el modelo en disco.
        
        Parámetros:
        -----------
        filepath : str
            Ruta donde guardar el modelo
        """
        torch.save({
            'q_network_state_dict': self.q_network.state_dict(),
            'target_q_network_state_dict': self.target_q_network.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
            'update_counter': self.update_counter,
            'config': self.config
        }, filepath)
        print(f"Modelo guardado en {filepath}")
    
    def load_model(self, filepath: str) -> None:
        """
        Carga el modelo desde disco.
        
        Parámetros:
        -----------
        filepath : str
            Ruta desde donde cargar el modelo
        """
        checkpoint = torch.load(filepath, map_location=CONST_DEVICE)
        self.q_network.load_state_dict(checkpoint['q_network_state_dict'])
        self.target_q_network.load_state_dict(checkpoint['target_q_network_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.epsilon = checkpoint['epsilon']
        self.update_counter = checkpoint['update_counter']
        self.config = checkpoint['config']
        print(f"Modelo cargado desde {filepath}")
    
    def visualize_training(self, history: Dict, window_size: int = 10) -> None:
        """
        Visualiza los resultados del entrenamiento.
        
        Parámetros:
        -----------
        history : Dict
            Historial de entrenamiento con métricas
        window_size : int, opcional
            Tamaño de la ventana para suavizado de curvas (default: 10)
        """
        def smooth(data: List[float], window_size: int) -> np.ndarray:
            """
            Suaviza una serie de datos usando promedio móvil.
            
            Parámetros:
            -----------
            data : List[float]
                Datos a suavizar
            window_size : int
                Tamaño de la ventana para el promedio móvil
                
            Retorna:
            --------
            np.ndarray
                Datos suavizados
            """
            if len(data) < window_size:
                return np.array(data)
            
            smoothed = []
            for i in range(len(data)):
                start = max(0, i - window_size + 1)
                smoothed.append(np.mean(data[start:i+1]))
            
            return np.array(smoothed)
        
        plt.figure(figsize=(15, 10))
        
        # Graficar recompensas
        plt.subplot(2, 2, 1)
        rewards = history.get("episode_rewards", [])
        plt.plot(rewards, alpha=0.5, label='Original')
        if len(rewards) > window_size:
            smoothed_rewards = smooth(rewards, window_size)
            plt.plot(smoothed_rewards, label=f'Suavizado (ventana={window_size})')
        plt.title('Recompensas por Episodio')
        plt.xlabel('Episodio')
        plt.ylabel('Recompensa')
        plt.legend()
        plt.grid(alpha=0.3)
        
        # Graficar pérdida
        plt.subplot(2, 2, 2)
        losses = history.get(CONST_LOSS, [])
        if losses:
            plt.plot(losses, alpha=0.5, label='Original')
            if len(losses) > window_size:
                smoothed_losses = smooth(losses, window_size)
                plt.plot(smoothed_losses, label=f'Suavizado (ventana={window_size})')
            plt.title('Pérdida durante Entrenamiento')
            plt.xlabel('Episodio')
            plt.ylabel('Pérdida')
            plt.legend()
            plt.grid(alpha=0.3)
        
        # Graficar valores Q promedio
        plt.subplot(2, 2, 3)
        q_values = history.get(CONST_Q_VALUE, [])
        if q_values:
            plt.plot(q_values, alpha=0.5, label='Original')
            if len(q_values) > window_size:
                smoothed_q_values = smooth(q_values, window_size)
                plt.plot(smoothed_q_values, label=f'Suavizado (ventana={window_size})')
            plt.title('Valores Q Promedio')
            plt.xlabel('Episodio')
            plt.ylabel('Valor Q Promedio')
            plt.legend()
            plt.grid(alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(CONST_FIGURES_DIR, 'dqn_training.png'), dpi=300)
        plt.show()


class DQNWrapper(nn.Module):
    """
    Wrapper para el agente DQN que implementa la interfaz compatible con el sistema.
    
    Parámetros:
    -----------
    dqn_agent : DQN
        Agente DQN inicializado
    cgm_shape : Tuple[int, ...]
        Forma de los datos CGM
    other_features_shape : Tuple[int, ...]
        Forma de otras características
    """
    def __init__(self, dqn_agent: DQN, cgm_shape: Tuple[int, ...], other_features_shape: Tuple[int, ...]) -> None:
        super(DQNWrapper, self).__init__()
        self.dqn_agent = dqn_agent
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
        self.cgm_encoder = nn.Linear(cgm_dim, self.dqn_agent.state_dim // 2)
        self.other_encoder = nn.Linear(other_dim, self.dqn_agent.state_dim // 2)
        
        # Inicializar pesos
        torch.nn.init.xavier_uniform_(self.cgm_encoder.weight)
        torch.nn.init.xavier_uniform_(self.other_encoder.weight)
    
    def forward(self, cgm_input: torch.Tensor, other_input: torch.Tensor) -> torch.Tensor:
        """
        Forward pass para el modelo.
        
        Parámetros:
        -----------
        cgm_input : torch.Tensor
            Datos CGM de entrada
        other_input : torch.Tensor
            Otras características de entrada
            
        Retorna:
        --------
        torch.Tensor
            Predicción de dosis de insulina (valor continuo)
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
        
        # Obtener Q-valores para todas las acciones
        q_values = self.dqn_agent.q_network(state)
        
        # Durante entrenamiento, necesitamos una ruta diferenciable
        if self.training:
            # Usar softmax para obtener una distribución de probabilidad sobre acciones
            # Temperatura alta para aproximar argmax de forma diferenciable
            action_probs = F.softmax(q_values * 10.0, dim=1)
            
            # Crear tensor de valores de acción (0 a action_dim-1)
            action_values = torch.arange(0, self.dqn_agent.action_dim, 
                                         device=q_values.device).float()
            
            # Calcular valor esperado de acción (suma ponderada)
            # Esto mantiene el flujo de gradientes
            expected_action = torch.matmul(action_probs, action_values).unsqueeze(1)
            
            # Escalar al rango de dosis (0-15)
            doses = expected_action * (15.0 / (self.dqn_agent.action_dim - 1))
        else:
            # Durante inferencia, usar argmax (no importa que no sea diferenciable)
            with torch.no_grad():
                action_indices = torch.argmax(q_values, dim=1, keepdim=True)
                doses = action_indices.float() * (15.0 / (self.dqn_agent.action_dim - 1))
        
        return doses
    
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
            Predicciones de dosis de insulina
        """
        cgm_data, other_data = inputs
        
        if not isinstance(cgm_data, torch.Tensor):
            cgm_data = torch.FloatTensor(cgm_data)
            other_data = torch.FloatTensor(other_data)
        
        with torch.no_grad():
            self.eval()  # Asegurar modo evaluación
            doses = self.forward(cgm_data, other_data)
            
        return doses.cpu().numpy()


def create_dqn_model(cgm_shape: Tuple[int, ...], other_features_shape: Tuple[int, ...]) -> DRLModelWrapperPyTorch:
    """
    Crea un modelo basado en DQN (Deep Q-Network) para predicción de dosis de insulina.
    
    Parámetros:
    -----------
    cgm_shape : Tuple[int, ...]
        Forma de los datos CGM (batch_size, time_steps, features)
    other_features_shape : Tuple[int, ...]
        Forma de otras características (batch_size, n_features)
        
    Retorna:
    --------
    DRLModelWrapperPyTorch
        Modelo DQN que implementa la interfaz del sistema
    """
    # Calcular dimensión del espacio de estados codificado
    state_dim = 64  # Dimensión del espacio de estados latente
    
    # Configurar espacio de acciones (niveles discretos de dosis)
    action_dim = 20  # 20 niveles discretos para dosis de 0 a 15 unidades
    
    # Crear configuración para el agente DQN
    config = {
        'learning_rate': DQN_CONFIG['learning_rate'],
        'gamma': DQN_CONFIG['gamma'],
        'epsilon_start': DQN_CONFIG['epsilon_start'],
        'epsilon_end': DQN_CONFIG['epsilon_end'],
        'epsilon_decay': DQN_CONFIG['epsilon_decay'],
        'buffer_capacity': DQN_CONFIG['buffer_capacity'],
        'batch_size': min(DQN_CONFIG['batch_size'], 64),  # Adaptado para este problema
        'target_update_freq': DQN_CONFIG['target_update_freq'],
        'dueling': DQN_CONFIG['dueling'],
        'double': DQN_CONFIG['double'],
        'prioritized': DQN_CONFIG['prioritized'],
        'hidden_units': DQN_CONFIG['hidden_units'],
        'dropout_rate': DQN_CONFIG['dropout_rate'],
        'activation': DQN_CONFIG['activation']
    }
    
    # Crear agente DQN
    dqn_agent = DQN(
        state_dim=state_dim,
        action_dim=action_dim,
        config=config,
        hidden_units=config['hidden_units'],
        seed=CONST_DEFAULT_SEED
    )
    
    # Crear wrapper para DQN
    dqn_wrapper = DQNWrapper(
        dqn_agent=dqn_agent,
        cgm_shape=cgm_shape,
        other_features_shape=other_features_shape
    )
    
    # Envolver en DRLModelWrapperPyTorch para compatibilidad con el sistema
    dqn_model = DRLModelWrapperPyTorch(
        dqn_wrapper, 
        algorithm="dqn"
    )
    
    return dqn_model


def create_dqn_model_wrapper(cgm_shape: Tuple[int, ...], other_features_shape: Tuple[int, ...]) -> DRLModelWrapperPyTorch:
    """
    Crea un modelo DQN envuelto en DRLModelWrapperPyTorch.
    
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
    action_dim = 20  # Número de acciones discretas que representan dosis de 0-15
    
    # Model creator function that creates the DQN model
    def model_creator(**kwargs) -> nn.Module:
        """
        Creates a DQN model instance.
        
        Returns:
        --------
        nn.Module
            DQN model instance
        """
        # Crear configuración para el agente DQN
        config = {
            'learning_rate': DQN_CONFIG['learning_rate'],
            'gamma': DQN_CONFIG['gamma'],
            'epsilon_start': DQN_CONFIG['epsilon_start'],
            'epsilon_end': DQN_CONFIG['epsilon_end'],
            'epsilon_decay': DQN_CONFIG['epsilon_decay'],
            'buffer_capacity': DQN_CONFIG['buffer_capacity'],
            'batch_size': min(DQN_CONFIG['batch_size'], 64),
            'target_update_freq': DQN_CONFIG['target_update_freq'],
            'dueling': DQN_CONFIG['dueling'],
            'double': DQN_CONFIG['double'],
            'prioritized': DQN_CONFIG['prioritized'],
            'hidden_units': DQN_CONFIG['hidden_units'],
            'dropout_rate': DQN_CONFIG['dropout_rate'],
            'activation': DQN_CONFIG['activation']
        }
        
        # Crear agente DQN
        dqn_agent = DQN(
            state_dim=state_dim,
            action_dim=action_dim,
            config=config
        )
        
        # Crear y devolver el wrapper
        return DQNWrapper(
            dqn_agent=dqn_agent,
            cgm_shape=cgm_shape,
            other_features_shape=other_features_shape
        )
    
    # Crear el wrapper DRLModelWrapperPyTorch con nuestro creador de modelos
    wrapper = DRLModelWrapperPyTorch(model_creator, algorithm='dqn')
    
    # Añadir early stopping si está habilitado
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
    Retorna una función para crear un modelo DQN compatible con la API del sistema.
    
    Retorna:
    --------
    Callable[[Tuple[int, ...], Tuple[int, ...]], DRLModelWrapperPyTorch]
        Función para crear el modelo con las formas de entrada especificadas
    """
    return create_dqn_model_wrapper