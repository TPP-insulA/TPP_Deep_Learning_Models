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
from models.baseDRL import BaseDRLModel
from constants.constants import (
    SEVERE_HYPOGLYCEMIA_THRESHOLD, HYPOGLYCEMIA_THRESHOLD, HYPERGLYCEMIA_THRESHOLD, SEVERE_HYPERGLYCEMIA_THRESHOLD, TARGET_GLUCOSE, SEVERE_HYPO_PENALTY, HYPO_PENALTY_BASE, HYPER_PENALTY_BASE, SEVERE_HYPER_PENALTY, MAX_REWARD
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
        self.rng = rng or np.random.default_rng()
        
        self.buffer = []
        self.position = 0
        
    def add(self, state: Tuple[np.ndarray, np.ndarray], action: np.ndarray, 
           reward: float, next_state: Tuple[np.ndarray, np.ndarray], 
           done: bool) -> None:
        """
        Agrega una transición al buffer.
        
        Parámetros:
        -----------
        state : Tuple[np.ndarray, np.ndarray]
            Estado actual (x_cgm, x_other)
        action : np.ndarray
            Acción tomada
        reward : float
            Recompensa recibida
        next_state : Tuple[np.ndarray, np.ndarray]
            Estado siguiente (x_cgm_next, x_other_next)
        done : bool
            Indicador de fin de episodio
            
        Retorna:
        --------
        None
        """
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
            
        self.buffer[self.position] = (state, action, reward, next_state, done)
        self.position = (self.position + 1) % self.capacity
        
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
        batch = self.rng.choice(self.buffer, batch_size)
        
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
    action_dim : int
        Dimensión de la acción (dosis de insulina)
    hidden_dim : int
        Dimensión de las capas ocultas
    actor_lr : float
        Tasa de aprendizaje para el actor
    critic_lr : float
        Tasa de aprendizaje para el crítico
    gamma : float
        Factor de descuento para recompensas futuras
    tau : float
        Parámetro de actualización suave para redes objetivo
    buffer_size : int
        Tamaño del buffer de experiencia
    max_action : float
        Valor máximo de acción
    min_action : float
        Valor mínimo de acción
    exploration_noise : float
        Desviación estándar del ruido de exploración
    """
    
    def __init__(self, 
                 cgm_input_dim: tuple, 
                 other_input_dim: tuple,
                 action_dim: int = DDPG_CONFIG['action_dim'], 
                 hidden_dim: int = DDPG_CONFIG['hidden_dim'],
                 actor_lr: float = DDPG_CONFIG['actor_lr'],
                 critic_lr: float = DDPG_CONFIG['critic_lr'],
                 gamma: float = DDPG_CONFIG['gamma'],
                 tau: float = DDPG_CONFIG['tau'],
                 buffer_size: int = DDPG_CONFIG['buffer_size'],
                 max_action: float = DDPG_CONFIG['max_action'],
                 min_action: float = DDPG_CONFIG['min_action'],
                 exploration_noise: float = DDPG_CONFIG['exploration_noise'],
                 seed: int = DDPG_CONFIG['seed']) -> None:
        """
        Inicializa el modelo DDPG para dosificación de insulina.
        """
        super().__init__(cgm_input_dim, other_input_dim, action_dim, hidden_dim)
        
        # Inicializar semilla aleatoria para reproducibilidad
        self.seed = seed
        torch.manual_seed(seed)
        self.rng = np.random.Generator(np.random.PCG64(seed))
        
        # Parámetros del algoritmo
        self.gamma = gamma
        self.tau = tau
        self.max_action = max_action
        self.min_action = min_action
        self.exploration_noise = exploration_noise
        
        # Inicializar redes de actor y crítico
        self.actor = DDPGActor(cgm_input_dim, other_input_dim, action_dim, hidden_dim, max_action)
        self.actor_target = DDPGActor(cgm_input_dim, other_input_dim, action_dim, hidden_dim, max_action)
        self.actor_target.load_state_dict(self.actor.state_dict())
        
        self.critic = DDPGCritic(cgm_input_dim, other_input_dim, action_dim, hidden_dim)
        self.critic_target = DDPGCritic(cgm_input_dim, other_input_dim, action_dim, hidden_dim)
        self.critic_target.load_state_dict(self.critic.state_dict())
        
        # Optimizadores
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=critic_lr)
        
        # Buffer de experiencia
        self.buffer = ReplayBuffer(buffer_size, cgm_input_dim, other_input_dim, action_dim, self.rng)
        
        # Enviar redes al dispositivo correcto (CPU/GPU)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.to(self.device)
        
    def forward(self, x_cgm: torch.Tensor, x_other: torch.Tensor) -> torch.Tensor:
        """
        Paso hacia adelante que delega al actor para predecir acciones.
        
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
        Selecciona una acción para un estado dado, opcionalmente añadiendo ruido para exploración.
        
        Parámetros:
        -----------
        x_cgm : torch.Tensor
            Datos CGM de entrada
        x_other : torch.Tensor
            Otras características de entrada
        add_noise : bool
            Si se debe añadir ruido para exploración
            
        Retorna:
        --------
        torch.Tensor
            Acción seleccionada
        """
        with torch.no_grad():
            action = self.actor(x_cgm, x_other)
            
            if add_noise:
                noise = torch.randn_like(action) * self.exploration_noise
                action = action + noise
                
            # Limitar acciones al rango válido
            action = torch.clamp(action, self.min_action, self.max_action)
            
        return action
    
    def add_to_buffer(self, buffer: Any, state: Tuple[np.ndarray, np.ndarray], 
                     action: float, reward: float, 
                     next_state: Tuple[np.ndarray, np.ndarray], 
                     done: bool) -> None:
        """
        Añade una transición al buffer de experiencia.
        
        Parámetros:
        -----------
        buffer : Any
            Buffer donde añadir la transición
        state : Tuple[np.ndarray, np.ndarray]
            Estado actual (x_cgm, x_other)
        action : float
            Acción tomada
        reward : float
            Recompensa recibida
        next_state : Tuple[np.ndarray, np.ndarray]
            Siguiente estado (x_cgm_next, x_other_next)
        done : bool
            Indicador de fin de episodio
            
        Retorna:
        --------
        None
        """
        # Convertir la acción a array de numpy
        action_np = np.array([action]) if np.isscalar(action) else np.array(action)
        
        # Añadir al buffer
        self.buffer.add(state, action_np, reward, next_state, done)
    
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
    
    def compute_reward(self, glucose_level: float) -> float:
        """
        Calcula la recompensa basada en el nivel actual de glucosa en sangre.
        
        La función penaliza niveles fuera del rango objetivo, con penalizaciones más
        severas para hipoglucemia que para hiperglucemia debido a su mayor peligro inmediato.
        La recompensa máxima se obtiene cuando el nivel de glucosa está en el valor objetivo.
        
        Parámetros:
        -----------
        glucose_level : float
            Nivel de glucosa en mg/dL
            
        Retorna:
        --------
        float
            Valor de recompensa: positivo para niveles saludables, negativo para niveles peligrosos
        """
        # Hipoglucemia severa (muy peligroso)
        if glucose_level < SEVERE_HYPOGLYCEMIA_THRESHOLD:
            return SEVERE_HYPO_PENALTY
        
        # Hipoglucemia (peligroso)
        elif glucose_level < HYPOGLYCEMIA_THRESHOLD:
            # Penalización lineal que aumenta a medida que la glucosa disminuye
            severity_factor = (HYPOGLYCEMIA_THRESHOLD - glucose_level) / (HYPOGLYCEMIA_THRESHOLD - SEVERE_HYPOGLYCEMIA_THRESHOLD)
            return HYPO_PENALTY_BASE * (1 + severity_factor)
        
        # Rango objetivo (saludable)
        elif HYPOGLYCEMIA_THRESHOLD <= glucose_level <= HYPERGLYCEMIA_THRESHOLD:
            # Recompensa máxima en TARGET_GLUCOSE, disminuyendo a medida que nos alejamos
            deviation = abs(glucose_level - TARGET_GLUCOSE)
            max_deviation = max(TARGET_GLUCOSE - HYPOGLYCEMIA_THRESHOLD, HYPERGLYCEMIA_THRESHOLD - TARGET_GLUCOSE)
            return MAX_REWARD * (1 - deviation / max_deviation)
        
        # Hiperglucemia (preocupante)
        elif glucose_level <= SEVERE_HYPERGLYCEMIA_THRESHOLD:
            # Penalización lineal que aumenta a medida que la glucosa aumenta
            severity_factor = (glucose_level - HYPERGLYCEMIA_THRESHOLD) / (SEVERE_HYPERGLYCEMIA_THRESHOLD - HYPERGLYCEMIA_THRESHOLD)
            return HYPER_PENALTY_BASE * (1 + severity_factor)
        
        # Hiperglucemia severa (peligroso)
        else:  # glucose_level > SEVERE_HYPERGLYCEMIA_THRESHOLD
            return SEVERE_HYPER_PENALTY
    
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
            Batch de transiciones (estados, acciones, recompensas, estados siguientes, done flags)
            
        Retorna:
        --------
        Dict[str, float]
            Diccionario con las pérdidas calculadas
        """
        # Desempaquetar batch
        states, actions, rewards, next_states, dones = batch
        x_cgm, x_other = states
        x_cgm_next, x_other_next = next_states
        
        # Convertir a tensores y enviar al dispositivo correcto
        x_cgm_tensor = torch.FloatTensor(x_cgm).to(self.device)
        x_other_tensor = torch.FloatTensor(x_other).to(self.device)
        actions_tensor = torch.FloatTensor(actions).to(self.device)
        rewards_tensor = torch.FloatTensor(rewards).to(self.device)
        x_cgm_next_tensor = torch.FloatTensor(x_cgm_next).to(self.device)
        x_other_next_tensor = torch.FloatTensor(x_other_next).to(self.device)
        dones_tensor = torch.FloatTensor(dones).to(self.device)
        
        # Actualizar el crítico
        with torch.no_grad():
            # Seleccionar siguiente acción según el actor objetivo
            next_actions = self.actor_target(x_cgm_next_tensor, x_other_next_tensor)
            
            # Calcular valor Q objetivo
            target_q = self.critic_target(x_cgm_next_tensor, x_other_next_tensor, next_actions)
            target_q = rewards_tensor + (1 - dones_tensor) * self.gamma * target_q
        
        # Calcular valor Q actual
        current_q = self.critic(x_cgm_tensor, x_other_tensor, actions_tensor)
        
        # Calcular pérdida del crítico (MSE)
        critic_loss = F.mse_loss(current_q, target_q)
        
        # Actualizar crítico
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()
        
        # Actualizar el actor
        actor_actions = self.actor(x_cgm_tensor, x_other_tensor)
        actor_loss = -self.critic(x_cgm_tensor, x_other_tensor, actor_actions).mean()
        
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()
        
        # Actualizar redes objetivo suavemente
        self._update_target_networks()
        
        return {
            "actor_loss": actor_loss.item(),
            "critic_loss": critic_loss.item(),
            "total_loss": actor_loss.item() + critic_loss.item()
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
        Predice acciones para los estados dados.
        
        Parámetros:
        -----------
        x_cgm : np.ndarray
            Datos CGM de entrada
        x_other : np.ndarray
            Otras características de entrada
            
        Retorna:
        --------
        np.ndarray
            Acciones predichas (dosis de insulina)
        """
        self.eval()  # Cambiar a modo evaluación
        
        with torch.no_grad():
            # Convertir a tensores
            x_cgm_tensor = torch.FloatTensor(x_cgm).to(self.device)
            x_other_tensor = torch.FloatTensor(x_other).to(self.device)
            
            # Predecir acción
            actions = self.actor(x_cgm_tensor, x_other_tensor)
            
        return actions.cpu().numpy()
    
    def predict_with_context(self, x_cgm: np.ndarray, x_other: np.ndarray, 
                            carb_intake: float,
                            sleep_quality: float = None,
                            work_intensity: float = None,
                            exercise_intensity: float = None,
                            current_glucose: float = None,
                            iob: float = None) -> float:
        """
        Predice la dosis de insulina con información contextual adicional.
        
        Parámetros:
        -----------
        x_cgm : np.ndarray
            Datos CGM para predicción (array con mediciones de glucosa recientes)
        x_other : np.ndarray
            Otras características para predicción
        carb_intake : float
            Ingesta de carbohidratos en gramos (obligatorio)
        sleep_quality : float, opcional
            Calidad del sueño (escala de 0-10)
        work_intensity : float, opcional
            Intensidad del trabajo (escala de 0-10)
        exercise_intensity : float, opcional
            Intensidad del ejercicio (escala de 0-10)
        current_glucose : float, opcional
            Nivel actual de glucosa en mg/dL
        iob : float, opcional
            Insulina activa en unidades
            
        Retorna:
        --------
        float
            Dosis de insulina recomendada en unidades
        """
        # Verificar parámetros obligatorios
        if carb_intake is None:
            raise ValueError("El parámetro 'carb_intake' es obligatorio")
        
        # Convertir a tensores
        x_cgm_t = torch.FloatTensor(x_cgm).to(self.device)
        x_other_t = torch.FloatTensor(x_other).to(self.device)
        
        # Extraer valor de glucosa actual del array x_cgm o usar el proporcionado
        if current_glucose is None:
            current_glucose = float(x_cgm[-1, -1] if len(x_cgm.shape) > 1 else x_cgm[-1])
            
        # Usar IOB proporcionado o extraer del x_other si está disponible
        if iob is None and x_other.shape[1] > 2:
            iob = float(x_other[0, 2])  # Asumiendo que IOB está en la tercera columna
        else:
            iob = float(iob if iob is not None else 0.0)
        
        # Valores por defecto para parámetros opcionales
        sleep_quality = float(sleep_quality if sleep_quality is not None else 5.0)
        work_intensity = float(work_intensity if work_intensity is not None else 0.0)
        exercise_intensity = float(exercise_intensity if exercise_intensity is not None else 0.0)
        
        # Asegurar dimensiones correctas para el modelo
        if len(x_cgm_t.shape) == 2:
            x_cgm_t = x_cgm_t.unsqueeze(0)
        if len(x_other_t.shape) == 1:
            x_other_t = x_other_t.unsqueeze(0)
    
        # Predecir dosis usando directamente el actor sin ajustes manuales
        with torch.no_grad():
            self.actor.eval()
            dose = self.actor(x_cgm_t, x_other_t).cpu().numpy().flatten()[0]
            self.actor.train()
        
        # Asegurar límites de seguridad
        final_dose = max(self.min_action, min(dose, self.max_action))
        
        return float(final_dose)

    def evaluate_with_context(self, x_cgm: np.ndarray, x_other: np.ndarray, 
                            carb_intake: np.ndarray,
                            true_doses: np.ndarray = None,
                            sleep_quality: np.ndarray = None,
                            work_intensity: np.ndarray = None,
                            exercise_intensity: np.ndarray = None,
                            simulator: GlucoseSimulator = None) -> Dict[str, float]:
        """
        Evalúa el rendimiento del modelo con métricas clínicas usando contexto adicional.
        
        Parámetros:
        -----------
        x_cgm : np.ndarray
            Datos CGM para evaluación (array con mediciones de glucosa recientes)
        x_other : np.ndarray
            Otras características para evaluación
        carb_intake : np.ndarray
            Ingesta de carbohidratos en gramos para cada instancia (obligatorio)
        true_doses : np.ndarray, opcional
            Dosis reales de insulina para comparación
        sleep_quality : np.ndarray, opcional
            Calidad del sueño para cada instancia (escala de 0-10)
        work_intensity : np.ndarray, opcional
            Intensidad del trabajo para cada instancia (escala de 0-10)
        exercise_intensity : np.ndarray, opcional
            Intensidad del ejercicio para cada instancia (escala de 0-10)
        simulator : object, opcional
            Simulador de glucosa para evaluar las métricas clínicas
            
        Retorna:
        --------
        Dict[str, float]
            Diccionario con métricas de evaluación clínica
        """
        # Validar que las dimensiones coincidan
        n_samples = len(x_cgm)
        if len(carb_intake) != n_samples:
            raise ValueError("La cantidad de valores de ingesta de carbohidratos debe coincidir con la cantidad de muestras")
        
        # Inicializar resultados
        predictions = []
        time_in_range = []
        time_below_range = []
        time_above_range = []
        clinical_metrics = {}
        
        # Realizar predicciones para cada muestra
        for i in range(n_samples):
            # Extraer valores de contexto
            context = {
                'carb_intake': float(carb_intake[i])
            }
            
            # Agregar parámetros opcionales si están disponibles
            if sleep_quality is not None:
                context['sleep_quality'] = float(sleep_quality[i])
            if work_intensity is not None:
                context['work_intensity'] = float(work_intensity[i])
            if exercise_intensity is not None:
                context['exercise_intensity'] = float(exercise_intensity[i])
            
            # Predecir dosis
            dose = self.predict_with_context(x_cgm[i:i+1], x_other[i:i+1], **context)
            predictions.append(dose)
            
            # Evaluar con simulador si está disponible
            if simulator is not None:
                # Extraer glucosa inicial del CGM
                initial_glucose = float(x_cgm[i, -1] if len(x_cgm[i].shape) > 0 else x_cgm[i])
                
                # Simular trayectoria de glucosa
                trajectory = simulator.predict_glucose_trajectory(
                    initial_glucose=initial_glucose,
                    insulin_doses=[dose],
                    carb_intakes=[float(carb_intake[i])],
                    timestamps=[0],
                    prediction_horizon=6
                )
                
                # Calcular métricas clínicas
                in_range = np.logical_and(trajectory >= HYPOGLYCEMIA_THRESHOLD, 
                                        trajectory <= HYPERGLYCEMIA_THRESHOLD)
                below_range = trajectory < HYPOGLYCEMIA_THRESHOLD
                above_range = trajectory > HYPERGLYCEMIA_THRESHOLD
                
                time_in_range.append(100.0 * np.mean(in_range))
                time_below_range.append(100.0 * np.mean(below_range))
                time_above_range.append(100.0 * np.mean(above_range))
        
        # Calcular métricas clínicas agregadas
        if simulator is not None:
            clinical_metrics = {
                'time_in_range': float(np.mean(time_in_range)),
                'time_below_range': float(np.mean(time_below_range)),
                'time_above_range': float(np.mean(time_above_range))
            }
        
        # Calcular métricas de error si se proporcionan las dosis reales
        if true_doses is not None:
            mae = np.mean(np.abs(np.array(predictions) - true_doses))
            rmse = np.sqrt(np.mean(np.square(np.array(predictions) - true_doses)))
            clinical_metrics.update({
                'mae': float(mae),
                'rmse': float(rmse)
            })
        
        return clinical_metrics
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
    # Primero creamos el modelo DDPG
    model = DDPGModel(
        cgm_input_dim=cgm_input_dim,
        other_input_dim=other_input_dim,
        action_dim=DDPG_CONFIG["action_dim"],
        hidden_dim=DDPG_CONFIG["hidden_dim"],
        actor_lr=DDPG_CONFIG["actor_lr"],
        critic_lr=DDPG_CONFIG["critic_lr"],
        gamma=DDPG_CONFIG["gamma"],
        tau=DDPG_CONFIG["tau"],
        buffer_size=DDPG_CONFIG["buffer_size"],
        max_action=DDPG_CONFIG["max_action"],
        min_action=DDPG_CONFIG["min_action"],
        exploration_noise=DDPG_CONFIG["exploration_noise"],
        seed=DDPG_CONFIG["seed"]
    )

    return DRLModelWrapperPyTorch(model, algorithm="DDPG")