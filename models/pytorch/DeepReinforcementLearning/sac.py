import os
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Normal
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Any, Optional, Union, Callable, Sequence
from collections import deque, namedtuple
from types import SimpleNamespace

PROJECT_ROOT = os.path.abspath(os.getcwd())
sys.path.append(PROJECT_ROOT) 

from config.models_config import SAC_CONFIG
from constants.constants import CONST_DEFAULT_SEED, CONST_DEFAULT_EPOCHS, CONST_DEFAULT_BATCH_SIZE
from models.early_stopping import get_early_stopping_config
from custom.DeepReinforcementLearning.drl_pt import DRLModelWrapperPyTorch

# Constantes para uso repetido
CONST_DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CONST_ACTOR_LOSS = "actor_loss"
CONST_CRITIC_LOSS = "critic_loss"
CONST_ALPHA_LOSS = "alpha_loss"
CONST_TOTAL_LOSS = "total_loss"
CONST_ENTROPY = "entropy"
CONST_EPISODE_REWARDS = "episode_rewards"
CONST_FIGURES_DIR = os.path.join(PROJECT_ROOT, "figures", "pytorch", "sac")
os.makedirs(CONST_FIGURES_DIR, exist_ok=True)

# Definir estructura para el buffer de repetición
Experience = namedtuple('Experience', ['state', 'action', 'reward', 'next_state', 'done'])


class ReplayBuffer:
    """
    Buffer de repetición para almacenar y muestrear experiencias.
    
    Parámetros:
    -----------
    capacity : int
        Capacidad máxima del buffer
    seed : int, opcional
        Semilla para reproducibilidad (default: 42)
    """
    def __init__(self, capacity: int, seed: int = CONST_DEFAULT_SEED) -> None:
        self.buffer = deque(maxlen=capacity)
        self.rng = np.random.Generator(np.random.PCG64(seed))
    
    def add(self, state: np.ndarray, action: np.ndarray, reward: float, 
            next_state: np.ndarray, done: bool) -> None:
        """
        Añade una experiencia al buffer.
        
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
        done : bool
            Indicador de si el episodio terminó
        """
        experience = Experience(state, action, reward, next_state, done)
        self.buffer.append(experience)
    
    def sample(self, batch_size: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Muestrea un lote de experiencias del buffer.
        
        Parámetros:
        -----------
        batch_size : int
            Tamaño del lote a muestrear
            
        Retorna:
        --------
        Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]
            Tupla con (estados, acciones, recompensas, estados_siguientes, terminados)
        """
        # Limitar batch_size al tamaño del buffer
        batch_size = min(batch_size, len(self.buffer))
        
        # Muestrear índices aleatorios sin reemplazo
        indices = self.rng.choice(len(self.buffer), batch_size, replace=False)
        
        # Extraer experiencias
        states, actions, rewards, next_states, dones = zip(*[self.buffer[i] for i in indices])
        
        # Convertir a arrays numpy
        return (
            np.array(states),
            np.array(actions),
            np.array(rewards, dtype=np.float32).reshape(-1, 1),
            np.array(next_states),
            np.array(dones, dtype=np.float32).reshape(-1, 1)
        )
    
    def __len__(self) -> int:
        """
        Obtiene el número de experiencias en el buffer.
        
        Retorna:
        --------
        int
            Número de experiencias almacenadas
        """
        return len(self.buffer)


class GaussianPolicy(nn.Module):
    """
    Red de política que produce distribuciones gaussianas para acciones continuas.
    
    Parámetros:
    -----------
    state_dim : int
        Dimensión del espacio de estados
    action_dim : int
        Dimensión del espacio de acciones
    hidden_units : List[int]
        Lista con unidades en cada capa oculta
    min_log_std : float, opcional
        Valor mínimo para log de desviación estándar (default: -20)
    max_log_std : float, opcional
        Valor máximo para log de desviación estándar (default: 2)
    """
    def __init__(
        self, 
        state_dim: int, 
        action_dim: int, 
        hidden_units: List[int],
        min_log_std: float = -20.0,
        max_log_std: float = 2.0
    ) -> None:
        super(GaussianPolicy, self).__init__()
        
        # Límites para log_std
        self.min_log_std = min_log_std
        self.max_log_std = max_log_std
        self.action_dim = action_dim
        
        # Capas ocultas
        layers = []
        input_dim = state_dim
        
        for units in hidden_units:
            layers.append(nn.Linear(input_dim, units))
            layers.append(nn.ReLU())
            input_dim = units
        
        self.hidden_layers = nn.Sequential(*layers)
        
        # Capas de salida para mu y log_std
        self.mean_layer = nn.Linear(hidden_units[-1], action_dim)
        self.log_std_layer = nn.Linear(hidden_units[-1], action_dim)
    
    def forward(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Pasa un estado por la red de política para obtener distribución de acciones.
        
        Parámetros:
        -----------
        state : torch.Tensor
            Tensor con estados
            
        Retorna:
        --------
        Tuple[torch.Tensor, torch.Tensor]
            Tupla con (media, desviación estándar) de la distribución gaussiana
        """
        x = self.hidden_layers(state)
        
        # Calcular media y log_std
        mean = self.mean_layer(x)
        log_std = self.log_std_layer(x)
        
        # Aplicar límites a log_std
        log_std = torch.clamp(log_std, self.min_log_std, self.max_log_std)
        
        # Convertir log_std a std
        std = torch.exp(log_std)
        
        return mean, std
    
    def sample(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Muestrea acciones y calcula log_probs a partir del estado.
        
        Parámetros:
        -----------
        state : torch.Tensor
            Tensor con estados
            
        Retorna:
        --------
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor]
            Tupla con (acciones, log_probs, media)
        """
        mean, std = self.forward(state)
        
        # Crear distribución Normal
        normal = Normal(mean, std)
        
        # Muestrear acciones usando el truco de reparametrización
        x_t = normal.rsample()
        
        # Aplicar tanh para limitar acciones a [-1, 1]
        action = torch.tanh(x_t)
        
        # Calcular log_prob con corrección para tanh
        log_prob = normal.log_prob(x_t)
        
        # Corrección para la transformación tanh
        log_prob -= torch.log(1 - action.pow(2) + 1e-6)
        log_prob = log_prob.sum(1, keepdim=True)
        
        return action, log_prob, mean
    
    def get_action(self, state: np.ndarray, deterministic: bool = False) -> np.ndarray:
        """
        Obtiene acción a partir de un estado.
        
        Parámetros:
        -----------
        state : np.ndarray
            Estado para el cual obtener acción
        deterministic : bool, opcional
            Si es True, usa la media sin exploración (default: False)
            
        Retorna:
        --------
        np.ndarray
            Acción seleccionada
        """
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(CONST_DEVICE)
            
            if deterministic:
                # Retornar la media para comportamiento determinista
                mean, _ = self.forward(state_tensor)
                return torch.tanh(mean).cpu().numpy()[0]
            else:
                # Muestrear acción para exploración
                action, _, _ = self.sample(state_tensor)
                return action.cpu().numpy()[0]


class QNetwork(nn.Module):
    """
    Red Q para estimar el valor de pares estado-acción.
    
    Parámetros:
    -----------
    state_dim : int
        Dimensión del espacio de estados
    action_dim : int
        Dimensión del espacio de acciones
    hidden_units : List[int]
        Lista con unidades en cada capa oculta
    """
    def __init__(
        self, 
        state_dim: int, 
        action_dim: int, 
        hidden_units: List[int]
    ) -> None:
        super(QNetwork, self).__init__()
        
        # Capas ocultas
        layers = []
        input_dim = state_dim + action_dim
        
        for units in hidden_units:
            layers.append(nn.Linear(input_dim, units))
            layers.append(nn.ReLU())
            input_dim = units
        
        self.hidden_layers = nn.Sequential(*layers)
        
        # Capa de salida para el valor Q
        self.q_value = nn.Linear(hidden_units[-1], 1)
    
    def forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """
        Calcula el valor Q para un par estado-acción.
        
        Parámetros:
        -----------
        state : torch.Tensor
            Tensor con estados
        action : torch.Tensor
            Tensor con acciones
            
        Retorna:
        --------
        torch.Tensor
            Valores Q estimados
        """
        # Concatenar estado y acción
        x = torch.cat([state, action], dim=1)
        
        # Pasar por capas ocultas
        x = self.hidden_layers(x)
        
        # Calcular valor Q
        q_value = self.q_value(x)
        
        return q_value


class SAC(nn.Module):
    """
    Implementación del algoritmo Soft Actor-Critic (SAC).
    
    SAC es un algoritmo actor-crítico fuera de política (off-policy) que maximiza
    tanto la recompensa acumulada como la entropía de la política.
    
    Parámetros:
    -----------
    state_dim : int
        Dimensión del espacio de estados
    action_dim : int
        Dimensión del espacio de acciones
    config : Dict, opcional
        Diccionario con parámetros de configuración (default: SAC_CONFIG)
    hidden_units : Optional[List[int]], opcional
        Unidades en capas ocultas (default: None)
    seed : int, opcional
        Semilla para reproducibilidad (default: 42)
    """
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        config: Dict = SAC_CONFIG,
        hidden_units: Optional[List[int]] = None,
        seed: int = CONST_DEFAULT_SEED
    ) -> None:
        super(SAC, self).__init__()
        
        # Configurar semillas para reproducibilidad
        self.seed = seed
        torch.manual_seed(seed)
        np.random.seed(seed)
        self.rng = np.random.Generator(np.random.PCG64(seed))
        
        # Parámetros del algoritmo
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = config.get('gamma', 0.99)
        self.tau = config.get('tau', 0.005)
        self.alpha_value = config.get('alpha', 0.2)
        self.lr_actor = config.get('lr_actor', 3e-4)
        self.lr_critic = config.get('lr_critic', 3e-4)
        self.lr_alpha = config.get('lr_alpha', 3e-4)
        self.weight_decay = config.get('weight_decay', 1e-5)
        self.batch_size = config.get('batch_size', 64)
        self.auto_entropy_tuning = config.get('auto_entropy_tuning', True)
        
        # Unidades ocultas
        if hidden_units is None:
            self.hidden_units = config.get('hidden_units', [256, 256])
        else:
            self.hidden_units = hidden_units
        
        # Configurar dispositivo
        self.device = CONST_DEVICE
        
        # Inicializar política
        self.policy = GaussianPolicy(
            state_dim=state_dim,
            action_dim=action_dim,
            hidden_units=self.hidden_units
        ).to(self.device)
        
        # Inicializar redes Q
        self.q1 = QNetwork(
            state_dim=state_dim,
            action_dim=action_dim,
            hidden_units=self.hidden_units
        ).to(self.device)
        
        self.q2 = QNetwork(
            state_dim=state_dim,
            action_dim=action_dim,
            hidden_units=self.hidden_units
        ).to(self.device)
        
        # Inicializar redes Q objetivo
        self.target_q1 = QNetwork(
            state_dim=state_dim,
            action_dim=action_dim,
            hidden_units=self.hidden_units
        ).to(self.device)
        
        self.target_q2 = QNetwork(
            state_dim=state_dim,
            action_dim=action_dim,
            hidden_units=self.hidden_units
        ).to(self.device)
        
        # Copiar pesos a las redes objetivo
        self._copy_weights(self.q1, self.target_q1)
        self._copy_weights(self.q2, self.target_q2)
        
        # Configurar ajuste automático de entropía
        if self.auto_entropy_tuning:
            # Valor objetivo de entropía (heurística: -dim(A))
            self.target_entropy = -action_dim
            # Parámetro log(alpha) para optimización
            self.log_alpha = torch.zeros(1, requires_grad=True, device=self.device)
            self.alpha = self.log_alpha.exp()
        else:
            # Alpha fijo
            self.alpha = torch.tensor(self.alpha_value).to(self.device)
        
        # Buffer de repetición
        self.replay_buffer = ReplayBuffer(
            capacity=config.get('buffer_capacity', 1000000),
            seed=seed
        )
        
        # Métricas para seguimiento
        self.metrics = {
            CONST_ACTOR_LOSS: [],
            CONST_CRITIC_LOSS: [],
            CONST_ALPHA_LOSS: [],
            CONST_TOTAL_LOSS: [],
            CONST_ENTROPY: [],
            CONST_EPISODE_REWARDS: []
        }
        
        # Contadores
        self.training_step = 0
        self.global_step = 0
    
    def _copy_weights(self, source: nn.Module, target: nn.Module) -> None:
        """
        Copia los pesos de una red a otra.
        
        Parámetros:
        -----------
        source : nn.Module
            Red de origen
        target : nn.Module
            Red de destino
        """
        for target_param, source_param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(source_param.data)
    
    def _soft_update(self, source: nn.Module, target: nn.Module) -> None:
        """
        Actualiza suavemente los pesos de la red objetivo.
        
        Parámetros:
        -----------
        source : nn.Module
            Red de origen
        target : nn.Module
            Red de destino
        """
        for target_param, source_param in zip(target.parameters(), source.parameters()):
            target_value = target_param.data
            source_value = source_param.data
            target_param.data.copy_(self.tau * source_value + (1.0 - self.tau) * target_value)
    
    def _compute_q_target(self, next_state_batch: torch.Tensor, reward_batch: torch.Tensor, 
                          done_batch: torch.Tensor) -> torch.Tensor:
        """
        Calcula el valor Q objetivo para actualizar las redes Q.
        
        Parámetros:
        -----------
        next_state_batch : torch.Tensor
            Batch de estados siguientes
        reward_batch : torch.Tensor
            Batch de recompensas
        done_batch : torch.Tensor
            Batch de indicadores de finalización
            
        Retorna:
        --------
        torch.Tensor
            Valores Q objetivo
        """
        with torch.no_grad():
            # Muestrear acción del estado siguiente según la política actual
            next_action, next_log_prob, _ = self.policy.sample(next_state_batch)
            
            # Calcular valores Q para el estado siguiente
            next_q1 = self.target_q1(next_state_batch, next_action)
            next_q2 = self.target_q2(next_state_batch, next_action)
            
            # Usar el mínimo para reducir sobreestimación
            next_q = torch.min(next_q1, next_q2)
            
            # Restar término de entropía (regulado por alpha)
            next_q = next_q - self.alpha * next_log_prob
            
            # Calcular Q objetivo con ecuación de Bellman
            q_target = reward_batch + (1.0 - done_batch) * self.gamma * next_q
        
        return q_target
    
    def _update_critic(self, state_batch: torch.Tensor, action_batch: torch.Tensor, 
                      q_target: torch.Tensor, 
                      q1_optimizer: torch.optim.Optimizer,
                      q2_optimizer: torch.optim.Optimizer) -> float:
        """
        Actualiza las redes Q minimizando el error cuadrático.
        
        Parámetros:
        -----------
        state_batch : torch.Tensor
            Batch de estados
        action_batch : torch.Tensor
            Batch de acciones
        q_target : torch.Tensor
            Valores Q objetivo
        q1_optimizer : torch.optim.Optimizer
            Optimizador para la primera red Q
        q2_optimizer : torch.optim.Optimizer
            Optimizador para la segunda red Q
            
        Retorna:
        --------
        float
            Pérdida del crítico
        """
        # Calcular valores Q actuales
        q1_value = self.q1(state_batch, action_batch)
        q2_value = self.q2(state_batch, action_batch)
        
        # Calcular pérdidas MSE
        q1_loss = F.mse_loss(q1_value, q_target)
        q2_loss = F.mse_loss(q2_value, q_target)
        
        # Pérdida total del crítico
        critic_loss = q1_loss + q2_loss
        
        # Optimizar Q1
        q1_optimizer.zero_grad()
        q1_loss.backward(retain_graph=True)
        q1_optimizer.step()
        
        # Optimizar Q2
        q2_optimizer.zero_grad()
        q2_loss.backward()
        q2_optimizer.step()
        
        return critic_loss.item()
    
    def _update_actor_and_alpha(self, state_batch: torch.Tensor, 
                              policy_optimizer: torch.optim.Optimizer,
                              alpha_optimizer: Optional[torch.optim.Optimizer] = None) -> Tuple[float, float, float]:
        """
        Actualiza la política del actor y el parámetro alpha.
        
        Parámetros:
        -----------
        state_batch : torch.Tensor
            Batch de estados
        policy_optimizer : torch.optim.Optimizer
            Optimizador para la política
        alpha_optimizer : Optional[torch.optim.Optimizer], opcional
            Optimizador para alpha (default: None)
            
        Retorna:
        --------
        Tuple[float, float, float]
            (actor_loss, alpha_loss, entropy)
        """
        # Muestrear acciones y log_probs
        action, log_prob, _ = self.policy.sample(state_batch)
        
        # Calcular valores Q para la acción muestreada
        q1_value = self.q1(state_batch, action)
        q2_value = self.q2(state_batch, action)
        
        # Usar mínimo para reducir sobreestimación
        q_value = torch.min(q1_value, q2_value)
        
        # Calcular pérdida del actor (maximizar Q - alpha*log_prob)
        actor_loss = (self.alpha * log_prob - q_value).mean()
        
        # Optimizar actor
        policy_optimizer.zero_grad()
        actor_loss.backward(retain_graph=True)
        policy_optimizer.step()
        
        # Calcular entropía promedio
        entropy = -log_prob.mean().item()
        
        # Actualizar alpha si está habilitado el ajuste automático
        alpha_loss = 0.0
        if self.auto_entropy_tuning and alpha_optimizer is not None:
            # Pérdida de alpha (minimizar alpha*(-log_prob - target_entropy))
            alpha_loss = -(self.log_alpha * (log_prob + self.target_entropy).detach()).mean()
            
            # Optimizar alpha
            alpha_optimizer.zero_grad()
            alpha_loss.backward()
            alpha_optimizer.step()
            
            # Actualizar valor de alpha
            self.alpha = self.log_alpha.exp()
            alpha_loss = alpha_loss.item()
        
        return actor_loss.item(), alpha_loss, entropy
    
    def _update_targets(self) -> None:
        """
        Actualiza suavemente las redes Q objetivo.
        """
        self._soft_update(self.q1, self.target_q1)
        self._soft_update(self.q2, self.target_q2)
    
    def optimize_model(self, 
                      policy_optimizer: torch.optim.Optimizer, 
                      q1_optimizer: torch.optim.Optimizer,
                      q2_optimizer: torch.optim.Optimizer,
                      alpha_optimizer: Optional[torch.optim.Optimizer] = None) -> Tuple[float, float, float, float]:
        """
        Realiza un paso de optimización del modelo usando experiencias del buffer.
        
        Parámetros:
        -----------
        policy_optimizer : torch.optim.Optimizer
            Optimizador para la política
        q1_optimizer : torch.optim.Optimizer
            Optimizador para la primera red Q
        q2_optimizer : torch.optim.Optimizer
            Optimizador para la segunda red Q
        alpha_optimizer : Optional[torch.optim.Optimizer], opcional
            Optimizador para alpha (default: None)
            
        Retorna:
        --------
        Tuple[float, float, float, float]
            (critic_loss, actor_loss, alpha_loss, entropy)
        """
        # Verificar si hay suficientes experiencias
        if len(self.replay_buffer) < self.batch_size:
            return 0.0, 0.0, 0.0, 0.0
        
        # Muestrear batch de experiencias
        state_batch, action_batch, reward_batch, next_state_batch, done_batch = self.replay_buffer.sample(self.batch_size)
        
        # Convertir a tensores
        state_batch = torch.FloatTensor(state_batch).to(self.device)
        action_batch = torch.FloatTensor(action_batch).to(self.device)
        reward_batch = torch.FloatTensor(reward_batch).to(self.device)
        next_state_batch = torch.FloatTensor(next_state_batch).to(self.device)
        done_batch = torch.FloatTensor(done_batch).to(self.device)
        
        # Calcular Q objetivo
        q_target = self._compute_q_target(next_state_batch, reward_batch, done_batch)
        
        # Actualizar redes críticas
        critic_loss = self._update_critic(state_batch, action_batch, q_target, q1_optimizer, q2_optimizer)
        
        # Actualizar actor y alpha
        actor_loss, alpha_loss, entropy = self._update_actor_and_alpha(state_batch, policy_optimizer, alpha_optimizer)
        
        # Actualizar redes objetivo
        self._update_targets()
        
        # Incrementar contador de entrenamiento
        self.training_step += 1
        
        return critic_loss, actor_loss, alpha_loss, entropy
    
    def get_action(self, state: np.ndarray, deterministic: bool = False) -> np.ndarray:
        """
        Obtiene acción a partir de un estado.
        
        Parámetros:
        -----------
        state : np.ndarray
            Estado para el cual obtener acción
        deterministic : bool, opcional
            Si es True, usa la media sin exploración (default: False)
            
        Retorna:
        --------
        np.ndarray
            Acción seleccionada
        """
        return self.policy.get_action(state, deterministic)
    
    def save_model(self, filepath: str) -> None:
        """
        Guarda el modelo en disco.
        
        Parámetros:
        -----------
        filepath : str
            Ruta donde guardar el modelo
        """
        # Crear directorio si no existe
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        # Guardar estado del modelo y configuración
        torch.save({
            'policy_state_dict': self.policy.state_dict(),
            'q1_state_dict': self.q1.state_dict(),
            'q2_state_dict': self.q2.state_dict(),
            'target_q1_state_dict': self.target_q1.state_dict(),
            'target_q2_state_dict': self.target_q2.state_dict(),
            'log_alpha': self.log_alpha if self.auto_entropy_tuning else torch.log(self.alpha),
            'state_dim': self.state_dim,
            'action_dim': self.action_dim,
            'hidden_units': self.hidden_units,
            'gamma': self.gamma,
            'tau': self.tau,
            'auto_entropy_tuning': self.auto_entropy_tuning,
            'target_entropy': self.target_entropy if self.auto_entropy_tuning else -self.action_dim,
            'training_step': self.training_step,
            'global_step': self.global_step,
            'seed': self.seed,
            'metrics': self.metrics
        }, filepath)
    
    def load_model(self, filepath: str) -> None:
        """
        Carga el modelo desde disco.
        
        Parámetros:
        -----------
        filepath : str
            Ruta desde donde cargar el modelo
        """
        # Cargar checkpoint
        checkpoint = torch.load(filepath, map_location=self.device)
        
        # Configurar parámetros
        self.state_dim = checkpoint['state_dim']
        self.action_dim = checkpoint['action_dim']
        self.hidden_units = checkpoint['hidden_units']
        self.gamma = checkpoint['gamma']
        self.tau = checkpoint['tau']
        self.auto_entropy_tuning = checkpoint['auto_entropy_tuning']
        self.target_entropy = checkpoint['target_entropy']
        self.training_step = checkpoint['training_step']
        self.global_step = checkpoint['global_step']
        self.seed = checkpoint['seed']
        self.metrics = checkpoint['metrics']
        
        # Cargar estados de los modelos
        self.policy.load_state_dict(checkpoint['policy_state_dict'])
        self.q1.load_state_dict(checkpoint['q1_state_dict'])
        self.q2.load_state_dict(checkpoint['q2_state_dict'])
        self.target_q1.load_state_dict(checkpoint['target_q1_state_dict'])
        self.target_q2.load_state_dict(checkpoint['target_q2_state_dict'])
        
        # Configurar alpha
        if self.auto_entropy_tuning:
            self.log_alpha = checkpoint['log_alpha']
            self.alpha = self.log_alpha.exp()
        else:
            self.alpha = torch.exp(checkpoint['log_alpha']).item()


class SACWrapper(nn.Module):
    """
    Wrapper para integrar el algoritmo SAC con la interfaz de entrenamiento.
    
    Parámetros:
    -----------
    sac_agent : SAC
        Agente SAC
    cgm_shape : Tuple[int, ...]
        Forma de los datos CGM de entrada
    other_features_shape : Tuple[int, ...]
        Forma de otras características de entrada
    """
    def __init__(
        self, 
        sac_agent: SAC,
        cgm_shape: Tuple[int, ...],
        other_features_shape: Tuple[int, ...]
    ) -> None:
        super(SACWrapper, self).__init__()
        
        # Registrar agente SAC y sus componentes como submodules 
        # para que PyTorch pueda encontrar los parámetros
        self.sac_agent = sac_agent
        
        # Registrar explícitamente redes para que los parámetros sean detectados
        self.policy = self.sac_agent.policy
        self.q1 = self.sac_agent.q1
        self.q2 = self.sac_agent.q2
        self.target_q1 = self.sac_agent.target_q1
        self.target_q2 = self.sac_agent.target_q2
        
        self.cgm_shape = cgm_shape
        self.other_features_shape = other_features_shape
        
        # Codificadores para procesar entradas
        self._setup_encoders()
        
        # Transformación para mapear acciones a dosis
        self.action_to_dose = nn.Linear(1, 1)
        # Inicializar para mapear de [-1,1] a [0,15]
        with torch.no_grad():
            self.action_to_dose.weight.fill_(7.5)  # Escala: (15-0)/2 = 7.5
            self.action_to_dose.bias.fill_(7.5)   # Desplazamiento: (15+0)/2 = 7.5
        
        # Historia para seguimiento
        self.history = {'loss': [], 'val_loss': []}
        
        # Optimizadores
        self.policy_optimizer = optim.Adam(
            self.policy.parameters(), 
            lr=self.sac_agent.lr_actor,
            weight_decay=self.sac_agent.weight_decay
        )
        
        self.q1_optimizer = optim.Adam(
            self.q1.parameters(), 
            lr=self.sac_agent.lr_critic,
            weight_decay=self.sac_agent.weight_decay
        )
        
        self.q2_optimizer = optim.Adam(
            self.q2.parameters(), 
            lr=self.sac_agent.lr_critic,
            weight_decay=self.sac_agent.weight_decay
        )
        
        if self.sac_agent.auto_entropy_tuning:
            self.alpha_optimizer = optim.Adam(
                [self.sac_agent.log_alpha],
                lr=self.sac_agent.lr_alpha,
                weight_decay=self.sac_agent.weight_decay
            )
        else:
            self.alpha_optimizer = None
        
        # Mover al dispositivo disponible
        self.to(CONST_DEVICE)
    
    def _setup_encoders(self) -> None:
        """
        Configura codificadores para transformar datos de entrada en estados.
        """
        # Codificador para datos CGM
        cgm_in_channels = self.cgm_shape[1]
        cnn_out_channels = 32
        
        self.cgm_encoder = nn.Sequential(
            nn.Conv1d(cgm_in_channels, cnn_out_channels, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten()
        )
        
        # Codificador para otras características
        self.other_encoder = nn.Sequential(
            nn.Linear(self.other_features_shape[0], 32),
            nn.ReLU()
        )
        
        # Capa de combinación
        combined_size = cnn_out_channels + 32
        self.combined_layer = nn.Sequential(
            nn.Linear(combined_size, self.sac_agent.state_dim),
            nn.ReLU()
        )
    
    def _process_state(self, cgm_data: torch.Tensor, other_features: torch.Tensor) -> torch.Tensor:
        """
        Procesa datos de entrada para obtener representación de estado.
        
        Parámetros:
        -----------
        cgm_data : torch.Tensor
            Datos CGM
        other_features : torch.Tensor
            Otras características
            
        Retorna:
        --------
        torch.Tensor
            Estado procesado
        """
        # Codificar CGM (cambiar formato para conv1d)
        cgm_encoded = self.cgm_encoder(cgm_data.permute(0, 2, 1))
        
        # Codificar otras características
        other_encoded = self.other_encoder(other_features)
        
        # Combinar características
        combined = torch.cat([cgm_encoded, other_encoded], dim=1)
        
        # Proyectar al espacio de estados
        state = self.combined_layer(combined)
        
        return state
    
    def forward(self, cgm_data: torch.Tensor, other_features: torch.Tensor) -> torch.Tensor:
        """
        Obtiene predicciones de dosis a partir de los datos de entrada.
        
        Parámetros:
        -----------
        cgm_data : torch.Tensor
            Datos CGM
        other_features : torch.Tensor
            Otras características
            
        Retorna:
        --------
        torch.Tensor
            Predicciones de dosis
        """
        # Procesar estado
        batch_size = cgm_data.shape[0]
        states = self._process_state(cgm_data, other_features)
        
        # Predicciones
        with torch.no_grad():
            actions = torch.zeros(batch_size, 1, device=CONST_DEVICE)
            
            for i in range(batch_size):
                # Usar acción determinista para predicción
                action = self.sac_agent.get_action(states[i].cpu().numpy(), deterministic=True)
                actions[i, 0] = torch.tensor(action[0], device=CONST_DEVICE)
        
        # Mapear de [-1,1] a dosis
        doses = self.action_to_dose(actions)
        
        return doses
    
    def _create_training_environment(
        self, 
        cgm_data: torch.Tensor, 
        other_features: torch.Tensor, 
        targets: torch.Tensor
    ) -> Any:
        """
        Crea un entorno de entrenamiento para SAC.
        
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
            Entorno de dosificación de insulina
        """
        # Convertir tensores a arrays numpy
        cgm_np = cgm_data.cpu().numpy()
        other_np = other_features.cpu().numpy()
        targets_np = targets.cpu().numpy()
        
        # Definir entorno personalizado
        class InsulinDosingEnv:
            """Entorno personalizado para dosificación de insulina."""
            
            def __init__(self, cgm, features, targets, wrapper):
                self.cgm = cgm
                self.features = features
                self.targets = targets
                self.wrapper = wrapper
                self.current_idx = 0
                self.max_idx = len(targets) - 1
                self.rng = np.random.Generator(np.random.PCG64(CONST_DEFAULT_SEED))
                
                # Definir espacios
                self.observation_space = SimpleNamespace(
                    shape=(wrapper.sac_agent.state_dim,)
                )
                self.action_space = SimpleNamespace(
                    shape=(wrapper.sac_agent.action_dim,),
                    high=1.0,
                    low=-1.0,
                    sample=self._sample_action
                )
            
            def _sample_action(self):
                """Muestrea una acción aleatoria."""
                return self.rng.uniform(-1, 1, size=(self.wrapper.sac_agent.action_dim,))
            
            def reset(self):
                """Reinicia el entorno a un estado aleatorio."""
                self.current_idx = self.rng.integers(0, self.max_idx)
                state = self._get_state()
                return state, {}
            
            def step(self, action):
                """Ejecuta un paso con la acción dada."""
                # Convertir acción a dosis (de [-1,1] a [0,15])
                action_tensor = torch.FloatTensor([[action[0]]]).to(CONST_DEVICE)
                dose = self.wrapper.action_to_dose(action_tensor).item()
                
                # Calcular recompensa
                target = self.targets[self.current_idx]
                error = abs(dose - target)
                
                # Penalizar más la sobredosificación
                if dose > target:
                    reward = -10.0 * error
                else:
                    reward = -5.0 * error
                
                # Avanzar al siguiente ejemplo
                self.current_idx = (self.current_idx + 1) % self.max_idx
                
                # Obtener próximo estado
                next_state = self._get_state()
                
                # Episodio termina después de un paso
                done = True
                truncated = False
                
                return next_state, reward, done, truncated, {}
            
            def _get_state(self):
                """Obtiene estado codificado para el ejemplo actual."""
                cgm_batch = torch.FloatTensor(self.cgm[self.current_idx:self.current_idx+1]).to(CONST_DEVICE)
                features_batch = torch.FloatTensor(self.features[self.current_idx:self.current_idx+1]).to(CONST_DEVICE)
                
                with torch.no_grad():
                    state = self.wrapper._process_state(cgm_batch, features_batch)
                
                return state.cpu().numpy()[0]
        
        # Crear instancia del entorno
        return InsulinDosingEnv(cgm_np, other_np, targets_np, self)
    
    def _run_episode(self, env: Any, max_steps: int, render: bool = False) -> Tuple[float, int]:
        """
        Ejecuta un único episodio de entrenamiento.
        
        Parámetros:
        -----------
        env : Any
            Entorno de entrenamiento
        max_steps : int
            Número máximo de pasos por episodio
        render : bool, opcional
            Si renderizar el entorno durante entrenamiento (default: False)
            
        Retorna:
        --------
        Tuple[float, int]
            (recompensa_acumulada, número_de_pasos)
        """
        state, _ = env.reset()
        episode_reward = 0
        episode_steps = 0
        
        for _ in range(max_steps):
            # Seleccionar acción
            action = self.sac_agent.get_action(state)
            
            # Ejecutar acción en el entorno
            next_state, reward, done, truncated, _ = env.step(action)
            
            # Renderizar si está habilitado
            if render:
                env.render()
            
            # Almacenar experiencia en el buffer
            self.sac_agent.replay_buffer.add(state, action, reward, next_state, done)
            
            # Actualizar estado y acumular recompensa
            state = next_state
            episode_reward += reward
            episode_steps += 1
            self.sac_agent.global_step += 1
            
            # Optimizar modelo
            self._optimize_if_possible()
            
            # Terminar episodio si está hecho
            if done or truncated:
                break
                
        return episode_reward, episode_steps
    
    def _optimize_if_possible(self) -> None:
        """
        Optimiza el modelo si hay suficientes experiencias en el buffer.
        """
        if len(self.sac_agent.replay_buffer) >= self.sac_agent.batch_size:
            critic_loss, actor_loss, alpha_loss, entropy = self.sac_agent.optimize_model(
                self.policy_optimizer, 
                self.q1_optimizer, 
                self.q2_optimizer, 
                self.alpha_optimizer
            )
            
            # Registrar métricas
            self.sac_agent.metrics[CONST_CRITIC_LOSS].append(critic_loss)
            self.sac_agent.metrics[CONST_ACTOR_LOSS].append(actor_loss)
            self.sac_agent.metrics[CONST_ALPHA_LOSS].append(alpha_loss)
            self.sac_agent.metrics[CONST_ENTROPY].append(entropy)
            self.sac_agent.metrics[CONST_TOTAL_LOSS].append(critic_loss + actor_loss)
    
    def train_agent(self, env: Any, episodes: int, max_steps: int, 
                  evaluation_episodes: int, eval_frequency: int, 
                  render: bool = False) -> Dict:
        """
        Entrena el agente SAC en un entorno.
        
        Parámetros:
        -----------
        env : Any
            Entorno de entrenamiento
        episodes : int
            Número de episodios de entrenamiento
        max_steps : int
            Número máximo de pasos por episodio
        evaluation_episodes : int
            Número de episodios para evaluación
        eval_frequency : int
            Frecuencia de evaluación en episodios
        render : bool, opcional
            Si renderizar el entorno durante entrenamiento (default: False)
            
        Retorna:
        --------
        Dict
            Métricas de entrenamiento
        """
        # Listas para seguimiento
        episode_rewards = []
        average_rewards = []
        eval_rewards = []
        
        for episode in range(episodes):
            # Ejecutar un episodio completo
            episode_reward, episode_steps = self._run_episode(env, max_steps, render)
            
            # Registrar recompensa del episodio
            episode_rewards.append(episode_reward)
            self.sac_agent.metrics[CONST_EPISODE_REWARDS].append(episode_reward)
            
            # Calcular recompensa promedio de los últimos 10 episodios
            avg_reward = np.mean(episode_rewards[-10:])
            average_rewards.append(avg_reward)
            
            # Mostrar progreso
            if episode % 10 == 0:
                print(f"Episodio: {episode+1}/{episodes}, Recompensa: {episode_reward:.2f}, "
                      f"Promedio: {avg_reward:.2f}, Pasos: {episode_steps}")
            
            # Evaluar agente periódicamente
            if (episode + 1) % eval_frequency == 0:
                eval_reward = self._evaluate_agent(env, evaluation_episodes)
                eval_rewards.append(eval_reward)
                print(f"Evaluación en episodio {episode+1}: Recompensa = {eval_reward:.2f}")
                
                # Guardar gráfico
                self._plot_rewards(episode_rewards, eval_rewards, episode)
        
        return {
            'episode_rewards': episode_rewards,
            'average_rewards': average_rewards,
            'eval_rewards': eval_rewards,
            'metrics': self.sac_agent.metrics
        }
    
    def _evaluate_agent(self, env: Any, episodes: int = 10) -> float:
        """
        Evalúa el agente durante un número determinado de episodios.
        
        Parámetros:
        -----------
        env : Any
            Entorno de evaluación
        episodes : int, opcional
            Número de episodios (default: 10)
            
        Retorna:
        --------
        float
            Recompensa promedio de evaluación
        """
        total_rewards = []
        
        for _ in range(episodes):
            state, _ = env.reset()
            done = False
            truncated = False
            episode_reward = 0
            
            while not (done or truncated):
                # Seleccionar acción determinista
                action = self.sac_agent.get_action(state, deterministic=True)
                
                # Ejecutar acción
                next_state, reward, done, truncated, _ = env.step(action)
                
                # Actualizar estado y recompensa
                state = next_state
                episode_reward += reward
            
            total_rewards.append(episode_reward)
        
        # Calcular promedio
        avg_reward = np.mean(total_rewards)
        
        return avg_reward
    
    def _plot_rewards(self, episode_rewards: List[float], eval_rewards: List[float], episode: int) -> None:
        """
        Genera y guarda un gráfico con las recompensas de entrenamiento y evaluación.
        
        Parámetros:
        -----------
        episode_rewards : List[float]
            Lista de recompensas por episodio
        eval_rewards : List[float]
            Lista de recompensas de evaluación
        episode : int
            Episodio actual
        """
        plt.figure(figsize=(10, 6))
        plt.plot(episode_rewards, label='Recompensa por episodio')
        
        # Calcular promedio móvil
        window_size = min(10, len(episode_rewards))
        if window_size > 0:
            moving_avg = np.convolve(
                episode_rewards, 
                np.ones(window_size) / window_size, 
                mode='valid'
            )
            plt.plot(range(window_size-1, len(episode_rewards)), moving_avg, 
                     label=f'Media móvil ({window_size} episodios)')
        
        # Graficar recompensas de evaluación
        if eval_rewards:
            eval_x = np.linspace(0, episode, len(eval_rewards))
            plt.plot(eval_x, eval_rewards, 'ro-', label='Recompensa de evaluación')
        
        plt.title('Recompensas durante el entrenamiento SAC')
        plt.xlabel('Episodio')
        plt.ylabel('Recompensa')
        plt.legend()
        plt.grid(True)
        
        # Guardar figura
        plt.savefig(os.path.join(CONST_FIGURES_DIR, f'sac_rewards_ep{episode}.png'))
        plt.close()
    
    def fit(
        self, 
        x: List[torch.Tensor], 
        y: torch.Tensor, 
        validation_data: Optional[Tuple] = None, 
        epochs: int = CONST_DEFAULT_EPOCHS,
        batch_size: int = CONST_DEFAULT_BATCH_SIZE,
        callbacks: List = None,
        verbose: int = 1
    ) -> Dict:
        """
        Entrena el modelo con los datos proporcionados.
        
        Parámetros:
        -----------
        x : List[torch.Tensor]
            Lista con [cgm_data, other_features]
        y : torch.Tensor
            Valores objetivo (dosis)
        validation_data : Optional[Tuple], opcional
            Datos de validación como ([x_cgm_val, x_other_val], y_val) (default: None)
        epochs : int, opcional
            Número de épocas (default: CONST_DEFAULT_EPOCHS)
        batch_size : int, opcional
            Tamaño de lote (default: 32)
        callbacks : List, opcional
            Lista de callbacks (default: None)
        verbose : int, opcional
            Nivel de verbosidad (0=silencioso, 1=progreso) (default: 1)
            
        Retorna:
        --------
        Dict
            Historial de entrenamiento
        """
        # Extraer datos de entrada
        cgm_data, other_features = x
        
        # Verificar datos de validación
        if validation_data is not None:
            val_x, val_y = validation_data
            val_cgm_data, val_other_features = val_x
        
        # Crear entorno de entrenamiento
        env = self._create_training_environment(cgm_data, other_features, y)
        
        # Entrenar agente SAC
        if verbose:
            print("Entrenando agente SAC...")
        
        # Configurar parámetros para SAC
        episodes_per_epoch = max(1, min(1000, len(cgm_data) // batch_size))
        total_episodes = episodes_per_epoch * epochs
        
        # Entrenar durante el número especificado de episodios
        self.train_agent(
            env=env,
            episodes=total_episodes,
            max_steps=1,  # Un paso por episodio
            evaluation_episodes=min(10, len(cgm_data) // 10),
            eval_frequency=episodes_per_epoch,
            render=False
        )
        
        # Actualizar historial con métricas
        self.history[CONST_ACTOR_LOSS] = self.sac_agent.metrics.get(CONST_ACTOR_LOSS, [])
        self.history[CONST_CRITIC_LOSS] = self.sac_agent.metrics.get(CONST_CRITIC_LOSS, [])
        self.history[CONST_ALPHA_LOSS] = self.sac_agent.metrics.get(CONST_ALPHA_LOSS, [])
        self.history[CONST_TOTAL_LOSS] = self.sac_agent.metrics.get(CONST_TOTAL_LOSS, [])
        self.history[CONST_EPISODE_REWARDS] = self.sac_agent.metrics.get(CONST_EPISODE_REWARDS, [])
        
        # Compatibilidad con interfaz DL
        self.history['loss'] = self.sac_agent.metrics.get(CONST_TOTAL_LOSS, [])
        
        # Calcular error en datos de entrenamiento
        train_preds = self.predict(x)
        train_loss = np.mean((train_preds - y.cpu().numpy()) ** 2)
        
        # Calcular error en datos de validación si existen
        if validation_data is not None:
            val_preds = self.predict([val_cgm_data, val_other_features])
            val_loss = np.mean((val_preds - val_y.cpu().numpy()) ** 2)
            self.history['val_loss'] = [val_loss]
            
            if verbose:
                print(f"Error de entrenamiento: {train_loss:.4f}, "
                      f"Error de validación: {val_loss:.4f}")
        elif verbose:
            print(f"Error de entrenamiento: {train_loss:.4f}")
        
        return self.history
    
    def predict(self, x: List[torch.Tensor]) -> np.ndarray:
        """
        Realiza predicciones con el modelo entrenado.
        
        Parámetros:
        -----------
        x : List[torch.Tensor]
            Lista con [cgm_data, other_features]
            
        Retorna:
        --------
        np.ndarray
            Predicciones de dosis
        """
        self.eval()  # Establecer modo evaluación
        
        # Extraer datos
        cgm_data, other_features = x
        
        # Convertir a tensores si son arrays
        if isinstance(cgm_data, np.ndarray):
            cgm_data = torch.FloatTensor(cgm_data).to(CONST_DEVICE)
        if isinstance(other_features, np.ndarray):
            other_features = torch.FloatTensor(other_features).to(CONST_DEVICE)
        
        # Realizar predicciones
        with torch.no_grad():
            predictions = self(cgm_data, other_features)
            
            # Convertir a numpy
            return predictions.cpu().numpy().flatten()
    
    def save(self, filepath: str) -> None:
        """
        Guarda el modelo en disco.
        
        Parámetros:
        -----------
        filepath : str
            Ruta donde guardar el modelo
        """
        # Crear directorio si no existe
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        # Guardar agente SAC
        self.sac_agent.save_model(f"{filepath}_sac_agent.pt")
        
        # Guardar wrapper
        torch.save({
            'wrapper_state_dict': self.state_dict(),
            'policy_optimizer_state_dict': self.policy_optimizer.state_dict(),
            'q1_optimizer_state_dict': self.q1_optimizer.state_dict(),
            'q2_optimizer_state_dict': self.q2_optimizer.state_dict(),
            'alpha_optimizer_state_dict': self.alpha_optimizer.state_dict() if self.alpha_optimizer else None,
            'history': self.history,
            'cgm_shape': self.cgm_shape,
            'other_features_shape': self.other_features_shape
        }, f"{filepath}_wrapper.pt")
    
    def load(self, filepath: str) -> None:
        """
        Carga el modelo desde disco.
        
        Parámetros:
        -----------
        filepath : str
            Ruta desde donde cargar el modelo
        """
        # Cargar agente SAC
        self.sac_agent.load_model(f"{filepath}_sac_agent.pt")
        
        # Cargar wrapper
        checkpoint = torch.load(f"{filepath}_wrapper.pt", map_location=CONST_DEVICE)
        self.load_state_dict(checkpoint['wrapper_state_dict'])
        
        # Restaurar optimizadores
        self.policy_optimizer.load_state_dict(checkpoint['policy_optimizer_state_dict'])
        self.q1_optimizer.load_state_dict(checkpoint['q1_optimizer_state_dict'])
        self.q2_optimizer.load_state_dict(checkpoint['q2_optimizer_state_dict'])
        
        if self.alpha_optimizer and checkpoint['alpha_optimizer_state_dict']:
            self.alpha_optimizer.load_state_dict(checkpoint['alpha_optimizer_state_dict'])
        
        # Restaurar historia y formas
        self.history = checkpoint['history']
        self.cgm_shape = checkpoint['cgm_shape']
        self.other_features_shape = checkpoint['other_features_shape']
    
    def get_config(self) -> Dict:
        """
        Obtiene la configuración del modelo.
        
        Retorna:
        --------
        Dict
            Configuración del modelo
        """
        return {
            'cgm_shape': self.cgm_shape,
            'other_features_shape': self.other_features_shape,
            'state_dim': self.sac_agent.state_dim,
            'action_dim': self.sac_agent.action_dim,
            'hidden_units': self.sac_agent.hidden_units,
            'gamma': self.sac_agent.gamma,
            'tau': self.sac_agent.tau,
            'auto_entropy_tuning': self.sac_agent.auto_entropy_tuning
        }


def create_sac_model(cgm_shape: Tuple[int, ...], other_features_shape: Tuple[int, ...]) -> DRLModelWrapperPyTorch:
    """
    Crea un modelo SAC para predicción de dosis de insulina.
    
    Parámetros:
    -----------
    cgm_shape : Tuple[int, ...]
        Forma de los datos CGM (batch_size, time_steps, features)
    other_features_shape : Tuple[int, ...]
        Forma de otras características (batch_size, n_features)
        
    Retorna:
    --------
    DRLModelWrapperPyTorch
        Modelo SAC envuelto en la interfaz del sistema
    """
    # Configurar dimensiones
    state_dim = 64  # Dimensión del espacio de estado latente
    action_dim = 1  # Dimensión para la dosis continua
    
    # Función creadora del modelo
    def model_creator(**kwargs) -> nn.Module:
        """
        Crea una instancia del modelo SAC.
        
        Retorna:
        --------
        nn.Module
            Instancia del modelo SAC
        """
        # Crear agente SAC
        sac_agent = SAC(
            state_dim=state_dim,
            action_dim=action_dim,
            config=SAC_CONFIG,
            hidden_units=SAC_CONFIG.get('hidden_units', [256, 256]),
            seed=CONST_DEFAULT_SEED
        )
        
        # Crear wrapper SAC
        wrapper = SACWrapper(
            sac_agent=sac_agent,
            cgm_shape=cgm_shape,
            other_features_shape=other_features_shape
        )
        
        # Verificar que haya parámetros
        param_count = sum(p.numel() for p in wrapper.parameters())
        if param_count == 0:
            raise ValueError("El modelo SAC no tiene parámetros entrenables")
        
        return wrapper
    
    # Devolver wrapper para sistema de entrenamiento
    return DRLModelWrapperPyTorch(model_creator, algorithm='sac')


def model_creator() -> Callable[[Tuple[int, ...], Tuple[int, ...]], DRLModelWrapperPyTorch]:
    """
    Retorna una función para crear un modelo SAC compatible con la API del sistema.
    
    Retorna:
    --------
    Callable[[Tuple[int, ...], Tuple[int, ...]], DRLModelWrapperPyTorch]
        Función para crear el modelo con las formas de entrada especificadas
    """
    return create_sac_model