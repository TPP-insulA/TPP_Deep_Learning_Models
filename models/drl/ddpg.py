"""
Implementación de Deep Deterministic Policy Gradient (DDPG) para dosificación de insulina.

DDPG es un algoritmo de aprendizaje por refuerzo profundo que combina:
1. Una red de política (actor) que determina la mejor acción para un estado dado
2. Una red de valor (crítico) que evalúa la calidad de pares estado-acción
3. Redes objetivo (target networks) para ambos para estabilizar el entrenamiento
"""

import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from typing import Dict, List, Tuple, Any, Optional, Union
import random
from collections import deque

from custom.DeepReinforcementLearning.drl_pt import DRLModelWrapperPyTorch
from custom.printer import print_debug, print_warning
from models.baseDRL import BaseDRLModel
from constants.constants import (
    IDEAL_LOWER_BOUND, IDEAL_UPPER_BOUND, SEVERE_HYPOGLYCEMIA_THRESHOLD, HYPOGLYCEMIA_THRESHOLD, HYPERGLYCEMIA_THRESHOLD, SEVERE_HYPERGLYCEMIA_THRESHOLD, SEVERE_HYPO_PENALTY, HYPO_PENALTY_BASE, HYPER_PENALTY_BASE, SEVERE_HYPER_PENALTY, MAX_REWARD
)
from config.models_config import DDPG_CONFIG
from validation.simulator import GlucoseSimulator

class DDPGActor(nn.Module):
    """
    Red del actor para DDPG que determina la acción óptima para un estado dado.
    
    Parámetros:
    -----------
    cgm_input_dim : tuple
        Dimensiones de entrada para datos CGM
    other_input_dim : tuple
        Dimensiones de entrada para otras características
    action_dim : int
        Dimensión de la acción (dosis de insulina)
    hidden_dim : int
        Dimensión de las capas ocultas
    max_action : float
        Valor máximo de acción permitido
    """
    
    def __init__(self, cgm_input_dim: tuple, other_input_dim: tuple, 
                action_dim: int = DDPG_CONFIG['action_dim'], hidden_dim: int = DDPG_CONFIG['hidden_dim'], max_action: float = DDPG_CONFIG['max_action']) -> None:
        super().__init__()
        
        # Encoder para datos CGM
        self.cgm_encoder = nn.Sequential(
            nn.Linear(np.prod(cgm_input_dim), hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU()
        )
        
        # Encoder para otras características
        self.other_encoder = nn.Sequential(
            nn.Linear(np.prod(other_input_dim), hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU()
        )
        
        # Capas combinadas
        self.combined_layer = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU()
        )
        
        # Capa de salida para acción (dosis de insulina)
        self.action_head = nn.Linear(hidden_dim // 2, action_dim)
        
        self.max_action = max_action
        
    def forward(self, x_cgm: torch.Tensor, x_other: torch.Tensor) -> torch.Tensor:
        """
        Paso hacia adelante de la red del actor.
        
        Parámetros:
        -----------
        x_cgm : torch.Tensor
            Datos CGM de entrada
        x_other : torch.Tensor
            Otras características de entrada
                
        Retorna:
        --------
        torch.Tensor
            Acción predicha (dosis de insulina)
        """
        # Aplanar entradas si es necesario
        if len(x_cgm.shape) > 2:
            x_cgm = x_cgm.reshape(x_cgm.shape[0], -1)
        if len(x_other.shape) > 2:
            x_other = x_other.reshape(x_other.shape[0], -1)
            
        # Codificar cada componente
        cgm_features = self.cgm_encoder(x_cgm)
        other_features = self.other_encoder(x_other)
        
        # Combinar características
        combined = torch.cat([cgm_features, other_features], dim=1)
        features = self.combined_layer(combined)
        
        # Generar acción
        action = torch.sigmoid(self.action_head(features)) * self.max_action
        
        return action

class DDPGCritic(nn.Module):
    """
    Red del crítico para DDPG que evalúa la calidad de pares estado-acción.
    
    Parámetros:
    -----------
    cgm_input_dim : tuple
        Dimensiones de entrada para datos CGM
    other_input_dim : tuple
        Dimensiones de entrada para otras características
    action_dim : int
        Dimensión de la acción (dosis de insulina)
    hidden_dim : int
        Dimensión de las capas ocultas
    """
    
    def __init__(self, cgm_input_dim: tuple, other_input_dim: tuple, 
                action_dim: int = DDPG_CONFIG['action_dim'], hidden_dim: int = DDPG_CONFIG['hidden_dim']) -> None:
        super().__init__()
        
        # Encoder para datos CGM
        self.cgm_encoder = nn.Sequential(
            nn.Linear(np.prod(cgm_input_dim), hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU()
        )
        
        # Encoder para otras características
        self.other_encoder = nn.Sequential(
            nn.Linear(np.prod(other_input_dim), hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU()
        )
        
        # Encoder para acciones
        self.action_encoder = nn.Sequential(
            nn.Linear(action_dim, hidden_dim // 4),
            nn.ReLU()
        )
        
        # Capa combinada
        combined_dim = hidden_dim + hidden_dim // 4
        self.combined_layer = nn.Sequential(
            nn.Linear(combined_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1)  # Valor Q
        )
        
    def forward(self, x_cgm: torch.Tensor, x_other: torch.Tensor, 
           action: torch.Tensor) -> torch.Tensor:
        """
        Paso hacia adelante de la red del crítico.
        
        Parámetros:
        -----------
        x_cgm : torch.Tensor
            Datos CGM de entrada
        x_other : torch.Tensor
            Otras características de entrada
        action : torch.Tensor
            Acción (dosis de insulina)
            
        Retorna:
        --------
        torch.Tensor
            Valor Q estimado
        """
        # Aplanar entradas si es necesario
        if len(x_cgm.shape) > 2:
            x_cgm = x_cgm.reshape(x_cgm.shape[0], -1)
        if len(x_other.shape) > 2:
            x_other = x_other.reshape(x_other.shape[0], -1)
            
        # Codificar cada componente
        cgm_features = self.cgm_encoder(x_cgm)
        other_features = self.other_encoder(x_other)
        action_features = self.action_encoder(action)
        
        # Combinar características de estado
        state_features = torch.cat([cgm_features, other_features], dim=1)
        
        # Combinar con características de acción
        combined = torch.cat([state_features, action_features], dim=1)
        
        # Estimar valor Q
        q_value = self.combined_layer(combined)
        
        return q_value


class ReplayBuffer:
    """
    Buffer de experiencia para almacenar y muestrear transiciones.
    
    Parámetros:
    -----------
    capacity : int
        Capacidad máxima del buffer
    cgm_dim : tuple
        Dimensiones de los datos CGM
    other_dim : tuple
        Dimensiones de otras características
    action_dim : int
        Dimensión de la acción
    rng : np.random.Generator
        Generador de números aleatorios
    """
    
    def __init__(self, capacity: int, cgm_dim: tuple, other_dim: tuple, 
                action_dim: int = DDPG_CONFIG['action_dim'], rng: np.random.Generator = None) -> None:
        self.capacity = capacity
        self.cgm_dim = cgm_dim
        self.other_dim = other_dim
        self.action_dim = action_dim
        self.rng = rng or np.random.default_rng(self.seed)
        
        self.buffer = []
        self.position = 0
    
    def __len__(self) -> int:
        """
        Obtiene el número de transiciones en el buffer.
        
        Retorna:
        --------
        int
            Número de transiciones almacenadas
        """
        return len(self.buffer)
        
    def _log_debug_info(self, state: Tuple[Any, Any], action: Any, next_state: Tuple[Any, Any]) -> None:
        """
        Registra información de depuración sobre los tipos de datos y dispositivos.
        
        Parámetros:
        -----------
        state : Tuple[Any, Any]
            Estado actual
        action : Any
            Acción tomada
        next_state : Tuple[Any, Any]
            Estado siguiente
        """
        print(f"Adding to buffer: state types: {type(state[0])}, {type(state[1])}, "
              f"action type: {type(action)}, next_state types: {type(next_state[0])}, {type(next_state[1])}")
        
        if isinstance(state[0], torch.Tensor):
            print(f"state[0] device: {state[0].device}")
        if isinstance(state[1], torch.Tensor):
            print(f"state[1] device: {state[1].device}")
        if isinstance(action, torch.Tensor):
            print(f"action device: {action.device}")
    
    def _validate_input_types(self, state: Tuple[Any, Any], reward: float, next_state: Tuple[Any, Any], done: bool) -> None:
        """
        Valida los tipos de los datos de entrada.
        
        Parámetros:
        -----------
        state : Tuple[Any, Any]
            Estado actual
        reward : float
            Recompensa recibida
        next_state : Tuple[Any, Any]
            Estado siguiente
        done : bool
            Indicador de fin de episodio
        """
        if not isinstance(state, tuple) or len(state) != 2:
            raise ValueError("State debe ser una tupla de dos elementos")
        if not isinstance(next_state, tuple) or len(next_state) != 2:
            raise ValueError("Next_state debe ser una tupla de dos elementos")
        if not isinstance(reward, (int, float)):
            raise ValueError("Reward debe ser un número (int o float)")
        if not isinstance(done, bool):
            raise ValueError("Done debe ser un booleano")
    
    def _convert_to_numpy(self, state: Tuple[Any, Any], action: Any, next_state: Tuple[Any, Any]) -> Tuple:
        """
        Convierte los datos de entrada a arrays de NumPy.
        
        Parámetros:
        -----------
        state : Tuple[Any, Any]
            Estado actual
        action : Any
            Acción tomada
        next_state : Tuple[Any, Any]
            Estado siguiente
            
        Retorna:
        --------
        Tuple
            Datos convertidos a NumPy
        """
        # Convertir state y next_state
        state_np = (
            # Eliminar dimensión de batch si existe
            state[0].detach().cpu().numpy().squeeze(0) if isinstance(state[0], torch.Tensor) else np.asarray(state[0], dtype=np.float32),
            state[1].detach().cpu().numpy().squeeze(0) if isinstance(state[1], torch.Tensor) else np.asarray(state[1], dtype=np.float32)
        )
        
        next_state_np = (
            # Eliminar dimensión de batch si existe
            next_state[0].detach().cpu().numpy().squeeze(0) if isinstance(next_state[0], torch.Tensor) else np.asarray(next_state[0], dtype=np.float32),
            next_state[1].detach().cpu().numpy().squeeze(0) if isinstance(next_state[1], torch.Tensor) else np.asarray(next_state[1], dtype=np.float32)
        )
        
        # Convertir action
        if isinstance(action, torch.Tensor):
            action_np = action.detach().cpu().numpy()
        elif isinstance(action, np.ndarray):
            action_np = action
        elif np.isscalar(action):
            action_np = np.array([action], dtype=np.float32)
        else:
            action_np = np.asarray(action, dtype=np.float32)
    
        # Ensure action_np is 1D
        action_np = action_np.flatten()
        
        return state_np, action_np, next_state_np
    
    def _validate_dimensions(self, state_np: Tuple[np.ndarray, np.ndarray], action_np: np.ndarray) -> None:
        """
        Valida las dimensiones de los datos convertidos.
        
        Parámetros:
        -----------
        state_np : Tuple[np.ndarray, np.ndarray]
            Estado convertido a NumPy
        action_np : np.ndarray
            Acción convertida a NumPy
        """
        if state_np[0].shape != self.cgm_dim:
            raise ValueError(f"CGM state dimension {state_np[0].shape} does not match expected {self.cgm_dim}")
        if state_np[1].shape != self.other_dim:
            raise ValueError(f"Other state dimension {state_np[1].shape} does not match expected {self.other_dim}")
        if action_np.shape != (self.action_dim,):
            raise ValueError(f"Action dimension {action_np.shape} does not match expected {(self.action_dim,)}")
    
    def add(self, state: Tuple[Any, Any], action: Any, 
        reward: float, next_state: Tuple[Any, Any], 
        done: bool) -> None:
        """
        Agrega una transición al buffer.
        
        Parámetros:
        -----------
        state : Tuple[Any, Any]
            Estado actual (x_cgm, x_other)
        action : Any
            Acción tomada
        reward : float
            Recompensa recibida
        next_state : Tuple[Any, Any]
            Estado siguiente (x_cgm_next, x_other_next)
        done : bool
            Indicador de fin de episodio
            
        Retorna:
        --------
        None
        """
        # self._log_debug_info(state, action, next_state)
        self._validate_input_types(state, reward, next_state, done)
        
        state_np, action_np, next_state_np = self._convert_to_numpy(state, action, next_state)
        self._validate_dimensions(state_np, action_np)
        
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
            
        self.buffer[self.position] = (state_np, action_np, reward, next_state_np, done)
        self.position = (self.position + 1) % self.capacity
        
    def push(self, state: Tuple[Any, Any], action: Any, 
         reward: float, next_state: Tuple[Any, Any], 
         done: bool) -> None:
        """
        Añade una transición al buffer (alias para add).
        
        Parámetros:
        -----------
        state : Tuple[Any, Any]
            Estado actual (x_cgm, x_other)
        action : Any
            Acción tomada
        reward : float
            Recompensa recibida
        next_state : Tuple[Any, Any]
            Estado siguiente (x_cgm_next, x_other_next)
        done : bool
            Indicador de fin de episodio
            
        Retorna:
        --------
        None
        """
        self.add(state, action, reward, next_state, done)

    def sample(self, batch_size: int) -> Tuple:
        """
        Muestrea un batch de transiciones del buffer.
        
        Parámetros:
        -----------
        batch_size : int
            Tamaño del batch a muestrear
            
        Retorna:
        --------
        Tuple
            Batch de transiciones (estados, acciones, recompensas, siguientes estados, done flags)
        """
        # Verificar si hay elementos en el buffer
        buffer_size = len(self.buffer)
        if buffer_size == 0:
            raise ValueError("No hay elementos en el buffer para muestrear")
        
        # Ajustar el batch_size al tamaño disponible
        effective_batch_size = min(buffer_size, batch_size)
        if effective_batch_size < batch_size:
            print_warning(f"Ajustando batch_size de {batch_size} a {effective_batch_size} (elementos disponibles)")
        
        # Generar índices aleatorios con reemplazo para garantizar batch completo
        indices = self.rng.choice(buffer_size, effective_batch_size, replace=True)
        
        # Extraer elementos según los índices
        batch = [self.buffer[i] for i in indices]
        
        # Desempaquetar batch
        states, actions, rewards, next_states, dones = zip(*batch)
        
        # Desempaquetar estados y siguientes estados
        x_cgm, x_other = zip(*states)
        x_cgm_next, x_other_next = zip(*next_states)
        
        # Convertir a arrays de numpy
        x_cgm_np = np.array(x_cgm)
        x_other_np = np.array(x_other)
        actions_np = np.array(actions)
        rewards_np = np.array(rewards).reshape(-1, 1)
        x_cgm_next_np = np.array(x_cgm_next)
        x_other_next_np = np.array(x_other_next)
        dones_np = np.array(dones).reshape(-1, 1)
        
        return (
            (x_cgm_np, x_other_np),
            actions_np,
            rewards_np,
            (x_cgm_next_np, x_other_next_np),
            dones_np
        )
        
    def __len__(self) -> int:
        """
        Obtiene el número de transiciones en el buffer.
        
        Retorna:
        --------
        int
            Número de transiciones almacenadas
        """
        return len(self.buffer)


class DDPGModel(BaseDRLModel):
    """
    Implementación de Deep Deterministic Policy Gradient (DDPG) para dosificación de insulina.
    
    Parámetros:
    -----------
    cgm_input_dim : tuple
        Dimensiones de entrada para datos CGM
    other_input_dim : tuple
        Dimensiones de entrada para otras características
    config : Dict[str, Any]
        Configuración del modelo DDPG
    rewards_function : callable
        Función para calcular recompensas
    """
    
    def __init__(self,
                 cgm_input_dim: tuple, 
                 other_input_dim: tuple,
                 config: Dict[str, Any] = DDPG_CONFIG,
                 rewards_function = None) -> None:
        """
        Inicializa el modelo DDPG para dosificación de insulina.
        """
        action_dim = config.get('action_dim', DDPG_CONFIG['action_dim'])
        hidden_dim = config.get('hidden_dim', DDPG_CONFIG['hidden_dim'])
        super().__init__(cgm_input_dim, other_input_dim, action_dim, hidden_dim)
        
        # Inicializar semilla aleatoria para reproducibilidad
        seed = DDPG_CONFIG['seed']
        self.seed = seed
        torch.manual_seed(seed)
        self.rng = np.random.Generator(np.random.PCG64(seed))
        
        self.config = config
        
        # Parámetros del algoritmo
        self.gamma = config.get('gamma', DDPG_CONFIG['gamma'])
        self.tau = config.get('tau', DDPG_CONFIG['tau'])
        self.max_action = config.get('max_action', DDPG_CONFIG['max_action'])
        self.min_action = config.get('min_action', DDPG_CONFIG['min_action'])
        self.exploration_noise = config.get('exploration_noise', DDPG_CONFIG['exploration_noise'])
        self.actor_lr = config.get('actor_lr', DDPG_CONFIG['actor_lr'])
        self.critic_lr = config.get('critic_lr', DDPG_CONFIG['critic_lr'])
        self.buffer_size = config.get('buffer_size', DDPG_CONFIG['buffer_size'])
        
        # Inicializar redes de actor y crítico
        self.actor = DDPGActor(cgm_input_dim, other_input_dim, action_dim, hidden_dim, self.max_action)
        self.actor_target = DDPGActor(cgm_input_dim, other_input_dim, action_dim, hidden_dim, self.max_action)
        self.actor_target.load_state_dict(self.actor.state_dict())
        
        self.critic = DDPGCritic(cgm_input_dim, other_input_dim, action_dim, hidden_dim)
        self.critic_target = DDPGCritic(cgm_input_dim, other_input_dim, action_dim, hidden_dim)
        self.critic_target.load_state_dict(self.critic.state_dict())
        
        # Optimizadores
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=self.actor_lr, weight_decay=1e-5)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=self.critic_lr, weight_decay=1e-5)
        
        # Buffer de experiencia
        self.buffer = ReplayBuffer(self.buffer_size, cgm_input_dim, other_input_dim, action_dim, self.rng)
        
        # Función de recompensa
        self.compute_rewards = rewards_function
        
        # Enviar redes al dispositivo correcto (CPU/GPU)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.to(self.device)
        
    def forward(self, x_cgm: torch.Tensor, x_other: torch.Tensor) -> torch.Tensor:
        """
        Realiza el paso hacia adelante del modelo.
        
        Parámetros:
        -----------
        x_cgm : torch.Tensor
            Datos CGM de entrada
        x_other : torch.Tensor
            Otras características de entrada
            
        Retorna:
        --------
        torch.Tensor
            Acción predicha (dosis de insulina)
        """
        return self.actor(x_cgm, x_other)

    def select_action(self, x_cgm: torch.Tensor, x_other: torch.Tensor, 
                  add_noise: bool = True) -> torch.Tensor:
        """
        Selecciona una acción basada en el estado actual.
        
        Parámetros:
        -----------
        x_cgm : torch.Tensor
            Datos CGM de entrada
        x_other : torch.Tensor
            Otras características de entrada
        add_noise : bool, opcional
            Si agregar ruido de exploración (default: True)
        
        Retorna:
        --------
        torch.Tensor
            Acción seleccionada
        """
        # Asegurar modo de evaluación
        self.actor.eval()
    
        # Aplanar tensores si es necesario
        if len(x_cgm.shape) > 2:
            x_cgm = x_cgm.reshape(x_cgm.shape[0], -1)
        if len(x_other.shape) > 2:
            x_other = x_other.reshape(x_other.shape[0], -1)
    
        # Obtener acción del actor
        with torch.no_grad():
            action = self.actor(x_cgm, x_other)
    
        # Agregar ruido de exploración si está habilitado
        if add_noise:
            noise = torch.normal(0, self.exploration_noise, size=action.shape).to(action.device)
            action = action + noise
    
        # Recortar a los límites de acción
        action = torch.clamp(action, self.min_action, self.max_action)
    
        # Volver a modo de entrenamiento
        self.actor.train()
    
        return action
    
    def add_to_buffer(self, buffer: Any, state: Tuple[Any, Any], 
                  action: Any, reward: float, 
                  next_state: Tuple[Any, Any], done: bool) -> None:
        """
        Añade una transición al buffer de experiencia.
        
        Parámetros:
        -----------
        buffer : Any
            Buffer de experiencia donde añadir la transición
        state : Tuple[Any, Any]
            Estado actual (x_cgm, x_other)
        action : Any
            Acción tomada (dosis de insulina)
        reward : float
            Recompensa recibida
        next_state : Tuple[Any, Any]
            Estado siguiente (x_cgm_next, x_other_next)
        done : bool
            Indicador de fin de episodio
            
        Retorna:
        --------
        None
        """
        # Log input types and devices for debugging
        print(f"add_to_buffer: state types: {type(state[0])}, {type(state[1])}, "
            f"action type: {type(action)}, next_state types: {type(next_state[0])}, {type(next_state[1])}")
        if isinstance(state[0], torch.Tensor):
            print(f"state[0] device: {state[0].device}")
        if isinstance(state[1], torch.Tensor):
            print(f"state[1] device: {state[1].device}")
        if isinstance(action, torch.Tensor):
            print(f"action device: {action.device}")
        
        # Convertir acción
        if isinstance(action, torch.Tensor):
            action_np = action.detach().cpu().numpy()
        elif isinstance(action, np.ndarray):
            action_np = action
        elif np.isscalar(action):
            action_np = np.array([action], dtype=np.float32)
        else:
            action_np = np.asarray(action, dtype=np.float32)
        
        # Ensure action_np is 1D
        action_np = action_np.flatten()
        
        # Convertir state y next_state
        state_np = (
            state[0].detach().cpu().numpy() if isinstance(state[0], torch.Tensor) else np.asarray(state[0], dtype=np.float32),
            state[1].detach().cpu().numpy() if isinstance(state[1], torch.Tensor) else np.asarray(state[1], dtype=np.float32)
        )
        next_state_np = (
            next_state[0].detach().cpu().numpy() if isinstance(next_state[0], torch.Tensor) else np.asarray(next_state[0], dtype=np.float32),
            next_state[1].detach().cpu().numpy() if isinstance(next_state[1], torch.Tensor) else np.asarray(next_state[1], dtype=np.float32)
        )
        
        # Añadir al buffer
        self.buffer.add(state_np, action_np, reward, next_state_np, done)

    def sample_buffer(self, buffer: Any, batch_size: int) -> Tuple:
        """
        Muestrea un batch de transiciones del buffer.
        
        Parámetros:
        -----------
        buffer : Any
            Buffer de donde muestrear
        batch_size : int
            Tamaño del batch a muestrear
            
        Retorna:
        --------
        Tuple
            Batch de transiciones
        """
        return self.buffer.sample(batch_size)
    
    def compute_rewards(self, x_cgm: np.ndarray, x_other: np.ndarray, 
                      actions: np.ndarray, simulator=None) -> np.ndarray:
        """
        Calcula recompensas para un batch de estados y acciones.
        
        Parámetros:
        -----------
        x_cgm : np.ndarray
            Datos CGM
        x_other : np.ndarray
            Otras características
        actions : np.ndarray
            Acciones tomadas
        simulator : Any, opcional
            Simulador de glucosa (si está disponible)
            
        Retorna:
        --------
        np.ndarray
            Recompensas calculadas
        """
        rewards = np.zeros(len(actions))
        
        # Si hay un simulador disponible, usarlo para calcular recompensas
        if simulator is not None:
            # Código para usar el simulador
            pass
        else:
            # Calcular recompensas basadas en el último valor de CGM
            for i in range(len(actions)):
                # Extraer el último valor de glucosa del CGM
                current_glucose = x_cgm[i, -1, 0] if len(x_cgm.shape) == 3 else x_cgm[i, -1]
                
                # Calcular recompensa
                rewards[i] = self.compute_reward(current_glucose)
        
        return rewards
    
    def update(self, batch: Tuple) -> Dict[str, float]:
        """
        Actualiza las redes de actor y crítico usando un batch de experiencias.
        
        Parámetros:
        -----------
        batch : Tuple
            Batch de experiencias (states, actions, rewards, next_states, dones)
            
        Retorna:
        --------
        Dict[str, float]
            Diccionario con pérdidas de actor y crítico
        """
        states, actions, rewards, next_states, dones = batch
        states_cgm, states_other = states
        next_states_cgm, next_states_other = next_states
        
        # Convertir a tensores
        states_cgm_t = torch.FloatTensor(states_cgm).to(self.device)
        states_other_t = torch.FloatTensor(states_other).to(self.device)
        actions_t = torch.FloatTensor(actions).to(self.device)
        rewards_t = torch.FloatTensor(rewards).to(self.device)
        next_states_cgm_t = torch.FloatTensor(next_states_cgm).to(self.device)
        next_states_other_t = torch.FloatTensor(next_states_other).to(self.device)
        dones_t = torch.FloatTensor(dones).to(self.device)
        
        # Asegurar que los tensores no contengan NaN
        states_cgm_t = torch.nan_to_num(states_cgm_t)
        states_other_t = torch.nan_to_num(states_other_t)
        actions_t = torch.nan_to_num(actions_t)
        rewards_t = torch.nan_to_num(rewards_t)
        next_states_cgm_t = torch.nan_to_num(next_states_cgm_t)
        next_states_other_t = torch.nan_to_num(next_states_other_t)
        
        # Actualizar crítico: minimizar error TD
        with torch.no_grad():
            next_actions = self.actor_target(next_states_cgm_t, next_states_other_t)
            next_actions = torch.nan_to_num(next_actions, nan=0.0)
            
            target_q = self.critic_target(next_states_cgm_t, next_states_other_t, next_actions)
            target_q = torch.nan_to_num(target_q, nan=0.0)
            
            target_q = rewards_t + (1 - dones_t) * self.gamma * target_q
            target_q = torch.nan_to_num(target_q, nan=0.0)
    
        # Q actual
        current_q = self.critic(states_cgm_t, states_other_t, actions_t)
        
        # Pérdida del crítico
        critic_loss = F.mse_loss(current_q, target_q)
        
        # Optimizar crítico
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 1.0)  # Recortar gradientes para estabilidad
        self.critic_optimizer.step()
        
        # Actualizar actor: maximizar Q esperado
        actor_actions = self.actor(states_cgm_t, states_other_t)
        actor_actions = torch.nan_to_num(actor_actions, nan=0.0)
        
        actor_loss = -self.critic(states_cgm_t, states_other_t, actor_actions).mean()
        
        # Optimizar actor
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 1.0)  # Recortar gradientes para estabilidad
        self.actor_optimizer.step()
        
        # Actualizar redes objetivo
        self._update_target_networks()
        
        return {
            'actor_loss': actor_loss.item(),
            'critic_loss': critic_loss.item(),
            'total_loss': actor_loss.item() + critic_loss.item()
        }
        
    def _update_target_networks(self) -> None:
        """
        Actualiza las redes objetivo usando actualización suave (soft update).
        
        Retorna:
        --------
        None
        """
        # Actualizar parámetros del actor objetivo
        for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
            
        # Actualizar parámetros del crítico objetivo
        for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
    
    def predict(self, x_cgm: np.ndarray, x_other: np.ndarray) -> np.ndarray:
        """
        Predice dosis de insulina para los estados dados.
        
        Parámetros:
        -----------
        x_cgm : np.ndarray
            Datos CGM para predicción
        x_other : np.ndarray
            Otras características para predicción
            
        Retorna:
        --------
        np.ndarray
            Dosis de insulina predichas
        """
        # Verificar y manejar entradas nulas o NaN
        if x_cgm is None or x_other is None or np.isnan(x_cgm).any() or np.isnan(x_other).any():
            print_warning("Se detectaron valores nulos o NaN en las entradas. Aplicando corrección.")
            if x_cgm is None:
                raise ValueError("Los datos CGM no pueden ser nulos")
            if x_other is None:
                raise ValueError("Los datos de otras características no pueden ser nulos")
            
            # Reemplazar NaN con valores seguros
            if np.isnan(x_cgm).any():
                x_cgm = np.nan_to_num(x_cgm, nan=100.0)  # Valor CGM predeterminado seguro
            if np.isnan(x_other).any():
                x_other = np.nan_to_num(x_other, nan=0.0)
        
        # Convertir a tensores
        x_cgm_tensor = torch.FloatTensor(x_cgm).to(self.device)
        x_other_tensor = torch.FloatTensor(x_other).to(self.device)
        
        # Asegurar dimensiones correctas
        if len(x_cgm_tensor.shape) == 2:
            x_cgm_tensor = x_cgm_tensor.unsqueeze(0)
        if len(x_other_tensor.shape) == 1:
            x_other_tensor = x_other_tensor.unsqueeze(0)
        
        # Predecir sin ruido
        with torch.no_grad():
            actions = self.select_action(x_cgm_tensor, x_other_tensor, add_noise=False)
        
        # Verificar NaN en las predicciones
        pred_numpy = actions.cpu().numpy()
        if np.isnan(pred_numpy).any():
            print_warning("Se detectaron NaN en las predicciones. Aplicando corrección.")
            pred_numpy = np.nan_to_num(pred_numpy, nan=0.5)  # Valor predeterminado seguro
        
        return pred_numpy
    
    def predict_with_context(self, x_cgm: np.ndarray, x_other: np.ndarray, 
                  carb_intake: float,
                  sleep_quality: float = None,
                  work_intensity: float = None,
                  exercise_intensity: float = None,
                  current_glucose: float = None,
                  iob: float = None) -> float:
        """
        Predice dosis de insulina con información contextual adicional.
        
        Parámetros:
        -----------
        x_cgm : np.ndarray
            Datos CGM para predicción
        x_other : np.ndarray
            Otras características para predicción
        carb_intake : float
            Ingesta de carbohidratos en gramos
        sleep_quality : float, opcional
            Calidad del sueño (escala 0-10)
        work_intensity : float, opcional
            Intensidad del trabajo (escala 0-10)
        exercise_intensity : float, opcional
            Intensidad del ejercicio (escala 0-10)
        current_glucose : float, opcional
            Nivel actual de glucosa en mg/dL
        iob : float, opcional
            Insulina activa en el cuerpo (Insulin On Board)
        
        Retorna:
        --------
        float
            Dosis de insulina recomendada
        """
        # Extraer nivel de glucosa si no se proporciona
        if current_glucose is None:
            if len(x_cgm.shape) > 2:
                current_glucose = float(x_cgm[0, -1, 0])
            elif len(x_cgm.shape) == 2:
                current_glucose = float(x_cgm[0, -1])
            else:
                current_glucose = float(x_cgm[-1])
    
        # Manejar IOB no proporcionado con estimación
        if iob is None:
            from training.common import calculate_iob
            iob = calculate_iob(x_cgm, carb_intake)
    
        # Valores por defecto para parámetros opcionales
        sleep_quality = 5.0 if sleep_quality is None else float(sleep_quality)
        work_intensity = 0.0 if work_intensity is None else float(work_intensity)
        exercise_intensity = 0.0 if exercise_intensity is None else float(exercise_intensity)
    
        # Preparar entrada modificada con contexto
        x_other_with_context = np.copy(x_other)
    
        # Si x_other es un array vacío o muy pequeño, ampliarlo
        if x_other_with_context.shape[1] < 5:
            extended_x_other = np.zeros((x_other_with_context.shape[0], 5))
            # Copiar los valores existentes
            for i in range(min(x_other_with_context.shape[1], 5)):
                extended_x_other[:, i] = x_other_with_context[:, i]
            x_other_with_context = extended_x_other
    
        # Actualizar con valores contextuales
        x_other_with_context[0, 0] = carb_intake
        x_other_with_context[0, 1] = iob
        if x_other_with_context.shape[1] > 2:
            x_other_with_context[0, 2] = sleep_quality
        if x_other_with_context.shape[1] > 3:
            x_other_with_context[0, 3] = work_intensity
        if x_other_with_context.shape[1] > 4:
            x_other_with_context[0, 4] = exercise_intensity
    
        # Predecir usando la entrada modificada con todo el contexto
        prediction = self.predict(x_cgm, x_other_with_context)
        
        print_debug(f"Predicción inicial: {prediction}, glucosa actual: {current_glucose}, IOB: {iob}, ingesta de carbohidratos: {carb_intake}")
    
        # Aplicar ajustes basados en reglas para seguridad adicional
        prediction_value = float(prediction.item() if hasattr(prediction, 'item') else prediction[0])

        # Ajustar dosis basada en glucosa actual para seguridad
        if current_glucose < SEVERE_HYPOGLYCEMIA_THRESHOLD:  # < 54 mg/dL
            # Reducir significativamente la dosis si hay hipoglucemia severa
            prediction_value *= 0.1  # Reducción drástica
        elif current_glucose < HYPOGLYCEMIA_THRESHOLD:  # < 70 mg/dL
            # Reducir moderadamente la dosis si hay hipoglucemia leve
            prediction_value *= 0.5
        elif HYPOGLYCEMIA_THRESHOLD <= current_glucose <= HYPERGLYCEMIA_THRESHOLD:  # 70-180 mg/dL
            # No ajustar dosis si está en rango normal
            prediction_value *= 1.0
        elif current_glucose < SEVERE_HYPERGLYCEMIA_THRESHOLD:# > 180 mg/dL
            # Aumentar moderadamente la dosis si hay hiperglucemia
            prediction_value *= 1.2
        else:  # > 250 mg/dL
            # Aumentar significativamente la dosis si hay hiperglucemia severa
            prediction_value *= 1.2
    
        # Carb-based dosing as a fallback component (based on typical insulin:carb ratio)
        carb_based_component = carb_intake / 10.0  # Typical I:C ratio
    
        # Blend the model prediction with the carb-based component
        # This ensures we're never giving zero insulin for meals
        if carb_intake > 0:
            # Blend more heavily toward carb-based dosing for larger meals
            blend_factor = min(0.8, carb_intake / 100.0)
            blended_dose = (1 - blend_factor) * prediction_value + blend_factor * carb_based_component
            prediction_value = max(prediction_value, blended_dose * 0.8)  # At least 80% of blended dose
    
        # Ajustar por IOB: reducir dosis si ya hay mucha insulina activa
        if iob > 3.0:
            prediction_value *= max(0.5, 1.0 - (iob - 3.0) / 4.0)
    
        # Basic minimum dose for meals
        if carb_intake > 20:
            minimum_dose = carb_intake / 30.0  # Conservative minimum
            prediction_value = max(prediction_value, minimum_dose)
    
        # Asegurar límites seguros
        return max(0.0, min(prediction_value, self.max_action))
    
    def evaluate_with_context(self, x_cgm: np.ndarray, x_other: np.ndarray, 
                    carb_intake: np.ndarray,
                    true_doses: np.ndarray = None,
                    sleep_quality: np.ndarray = None,
                    work_intensity: np.ndarray = None,
                    exercise_intensity: np.ndarray = None,
                    simulator: GlucoseSimulator = None) -> Dict[str, float]:
        """
        Evalúa el modelo usando contexto adicional y un simulador.
        
        Parámetros:
        -----------
        x_cgm : np.ndarray
            Datos CGM para evaluación
        x_other : np.ndarray
            Otras características para evaluación
        carb_intake : np.ndarray
            Valores de ingesta de carbohidratos
        true_doses : np.ndarray, opcional
            Dosis reales para comparación
        sleep_quality, work_intensity, exercise_intensity : np.ndarray, opcional
            Características contextuales adicionales
        simulator : GlucoseSimulator, opcional
            Simulador para métricas clínicas
        
        Retorna:
        --------
        Dict[str, float]
            Métricas de evaluación
        """
        predictions = []
        time_in_range_total = []
        time_in_range_ideal = []
        time_in_range_low = []
        time_in_range_high = []
        time_below_range = []
        time_severely_below_range = []
        time_above_range = []
        time_severely_above_range = []
        
        # Extraer valores de glucosa para simulación
        if len(x_cgm.shape) > 2:
            initial_glucose = np.array([x_cgm[i, -1, 0] for i in range(len(x_cgm))])
        else:
            initial_glucose = np.array([x_cgm[i, -1] if len(x_cgm[i]) > 0 else 120.0 for i in range(len(x_cgm))])

        for i in range(len(x_cgm)):
            # Preparar contexto para esta muestra
            context = {
                'carb_intake': float(carb_intake[i]),
                'current_glucose': float(initial_glucose[i])
            }
            
            # Agregar contexto adicional si está disponible
            if sleep_quality is not None:
                context['sleep_quality'] = float(sleep_quality[i])
            if work_intensity is not None:
                context['work_intensity'] = float(work_intensity[i])
            if exercise_intensity is not None:
                context['exercise_intensity'] = float(exercise_intensity[i])
    
        # Estimar IOB si es posible
        # Podemos usar la tendencia de glucosa como proxy
        if len(x_cgm[i]) > 2 and len(x_cgm.shape) > 2:
            glucose_trend = x_cgm[i, -1, 0] - x_cgm[i, -3, 0]
            if glucose_trend < -10:  # Bajando
                estimated_iob = max(1.0, context['carb_intake'] / 20.0)
            else:
                estimated_iob = context['carb_intake'] / 30.0
            context['iob'] = min(estimated_iob, 5.0)
    
        # Predecir dosis
        dose = self.predict_with_context(
            x_cgm[i:i+1], x_other[i:i+1], **context
        )
        predictions.append(dose)
        
        # Evaluar con simulador si está disponible
        if simulator is not None:
            trajectory = simulator.predict_glucose_trajectory(
                initial_glucose=initial_glucose[i],
                insulin_doses=[dose],
                carb_intakes=[float(carb_intake[i])],
                timestamps=[0],
                prediction_horizon=6
            )
            
            # Definir correctamente los rangos de glucosa
            severely_below_range = trajectory < SEVERE_HYPOGLYCEMIA_THRESHOLD
            below_range = np.logical_and(trajectory >= SEVERE_HYPOGLYCEMIA_THRESHOLD, trajectory < HYPOGLYCEMIA_THRESHOLD)
            in_range_low = np.logical_and(trajectory >= HYPOGLYCEMIA_THRESHOLD, trajectory < IDEAL_LOWER_BOUND)
            in_range_ideal = np.logical_and(trajectory >= IDEAL_LOWER_BOUND, trajectory <= IDEAL_UPPER_BOUND)
            in_range_high = np.logical_and(trajectory > IDEAL_UPPER_BOUND, trajectory <= HYPERGLYCEMIA_THRESHOLD)
            above_range = np.logical_and(trajectory > HYPERGLYCEMIA_THRESHOLD, 
                                        trajectory < SEVERE_HYPERGLYCEMIA_THRESHOLD)
            severely_above_range = trajectory >= SEVERE_HYPERGLYCEMIA_THRESHOLD
            
            # Calcular tiempo total en rango (70-180 mg/dL)
            in_range_total = np.logical_and(trajectory >= HYPOGLYCEMIA_THRESHOLD, 
                                          trajectory <= HYPERGLYCEMIA_THRESHOLD)
            
            # Calcular porcentajes para cada rango
            time_severely_below_range.append(100.0 * np.mean(severely_below_range))
            time_below_range.append(100.0 * np.mean(below_range))
            time_in_range_low.append(100.0 * np.mean(in_range_low))
            time_in_range_ideal.append(100.0 * np.mean(in_range_ideal))
            time_in_range_high.append(100.0 * np.mean(in_range_high))
            time_in_range_total.append(100.0 * np.mean(in_range_total))
            time_above_range.append(100.0 * np.mean(above_range))
            time_severely_above_range.append(100.0 * np.mean(severely_above_range))

        # Preparar métricas
        metrics = {}

        # Métricas clínicas si se utilizó simulador
        if simulator is not None:
            metrics.update({
                'time_in_range_total': float(np.mean(time_in_range_total)),
                'time_in_range_ideal': float(np.mean(time_in_range_ideal)),
                'time_in_range_low': float(np.mean(time_in_range_low)),
                'time_in_range_high': float(np.mean(time_in_range_high)),
                'time_below_range': float(np.mean(time_below_range)),
                'time_severely_below_range': float(np.mean(time_severely_below_range)),
                'time_above_range': float(np.mean(time_above_range)),
                'time_severely_above_range': float(np.mean(time_severely_above_range))
            })

        # Métricas de error si hay dosis reales
        if true_doses is not None:
            mae = np.mean(np.abs(np.array(predictions) - true_doses))
            rmse = np.sqrt(np.mean(np.square(np.array(predictions) - true_doses)))
            metrics.update({
                'mae': float(mae),
                'rmse': float(rmse)
            })

        return metrics

    def to(self, device: torch.device) -> 'DDPGModel':
        """
        Mueve todas las redes al dispositivo especificado.
        
        Parámetros:
        -----------
        device : torch.device
            Dispositivo donde mover las redes
            
        Retorna:
        --------
        DDPGModel
            Self para permitir encadenamiento
        """
        self.device = device
        self.actor = self.actor.to(device)
        self.actor_target = self.actor_target.to(device)
        self.critic = self.critic.to(device)
        self.critic_target = self.critic_target.to(device)
        return self


def create_ddpg_model(cgm_input_dim: tuple, other_input_dim: tuple) -> DRLModelWrapperPyTorch:
    """
    Crea un modelo DDPG para dosificación de insulina.
    
    Parámetros:
    -----------
    cgm_input_dim : tuple
        Dimensiones de entrada para datos CGM
    other_input_dim : tuple
        Dimensiones de entrada para otras características
        
    Retorna:
    --------
    DRLModelWrapperPyTorch
        Modelo DDPG inicializado envuelto en DRLModelWrapperPyTorch
    """
    from training.common import compute_reward
    
    model = DDPGModel(
        cgm_input_dim=cgm_input_dim,
        other_input_dim=other_input_dim,
        config=DDPG_CONFIG,
        rewards_function=compute_reward
    )

    return DRLModelWrapperPyTorch(model, algorithm="DDPG")