import os
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.multiprocessing as mp
from torch.distributions import Normal, Categorical
from typing import Dict, List, Tuple, Any, Optional, Union, Callable
import threading
import time
import gym
from tqdm.auto import tqdm

from custom.printer import print_debug
from constants.constants import CONST_MODEL_INIT_ERROR

PROJECT_ROOT = os.path.abspath(os.getcwd())
sys.path.append(PROJECT_ROOT) 

from config.models_config_old import A2C_A3C_CONFIG
from constants.constants import CONST_DEFAULT_SEED, CONST_DEFAULT_EPOCHS, CONST_DEFAULT_BATCH_SIZE, LOWER_BOUND_NORMAL_GLUCOSE_RANGE, UPPER_BOUND_NORMAL_GLUCOSE_RANGE, TARGET_GLUCOSE, POSITIVE_REWARD, MILD_PENALTY_REWARD, SEVERE_PENALTY_REWARD, CONST_DROPOUT, CONST_POLICY_LOSS, CONST_VALUE_LOSS, CONST_ENTROPY_LOSS, CONST_TOTAL_LOSS, CONST_EPISODE_REWARDS
from custom.DeepReinforcementLearning.drl_pt import DRLModelWrapperPyTorch

# Constantes para uso repetido
CONST_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


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
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        continuous: bool = True,
        hidden_units: Optional[List[int]] = None
    ) -> None:
        super(ActorCriticModel, self).__init__()
        
        # Valores predeterminados para capas ocultas
        if hidden_units is None:
            hidden_units = A2C_A3C_CONFIG['hidden_units']
        
        self.continuous = continuous
        self.action_dim = action_dim
        
        # Capas compartidas para procesamiento de estados
        self.shared_layers = nn.ModuleList()
        for i, units in enumerate(hidden_units[:2]):
            self.shared_layers.append(nn.Sequential(
                nn.Linear(state_dim if i == 0 else hidden_units[i-1], units),
                nn.LayerNorm(units, eps=A2C_A3C_CONFIG['epsilon']),
                nn.Dropout(A2C_A3C_CONFIG['dropout_rate']),
                nn.Tanh()
            ))
        
        # Red del Actor (política)
        self.actor_layers = nn.ModuleList()
        for i, units in enumerate(hidden_units[2:]):
            self.actor_layers.append(nn.Sequential(
                nn.Linear(hidden_units[1] if i == 0 else hidden_units[i+2-1], units),
                nn.LayerNorm(units, eps=A2C_A3C_CONFIG['epsilon']),
                nn.Tanh()
            ))
        
        # Capa de salida del actor (depende de si el espacio de acción es continuo o discreto)
        if continuous:
            self.mu = nn.Linear(hidden_units[-1], action_dim)
            self.log_sigma = nn.Linear(hidden_units[-1], action_dim)
        else:
            self.logits = nn.Linear(hidden_units[-1], action_dim)
        
        # Red del Crítico (valor)
        self.critic_layers = nn.ModuleList()
        for i, units in enumerate(hidden_units[2:]):
            self.critic_layers.append(nn.Sequential(
                nn.Linear(hidden_units[1] if i == 0 else hidden_units[i+2-1], units),
                nn.LayerNorm(units, eps=A2C_A3C_CONFIG['epsilon']),
                nn.Tanh()
            ))
        
        # Capa de salida del crítico (valor del estado)
        self.value = nn.Linear(hidden_units[-1], 1)
        
        # Enviar modelo al dispositivo adecuado
        self.to(CONST_DEVICE)
    
    def forward(self, x: torch.Tensor) -> Tuple[Any, torch.Tensor]:
        """
        Pasa la entrada por el modelo Actor-Crítico.
        
        Parámetros:
        -----------
        x : torch.Tensor
            Tensor de entrada con los estados
            
        Retorna:
        --------
        Tuple[Any, torch.Tensor]
            (política, valor) - la política puede ser una tupla (mu, sigma) o logits
        """
        x = x.to(CONST_DEVICE)
        
        # Capas compartidas
        for layer in self.shared_layers:
            x = layer(x)
        
        # Red del Actor
        actor_x = x
        for layer in self.actor_layers:
            actor_x = layer(actor_x)
        
        # Salida del actor según el tipo de política
        if self.continuous:
            mu = self.mu(actor_x)
            log_sigma = self.log_sigma(actor_x)
            log_sigma = torch.clamp(log_sigma, -20, 2)  # Limitar para estabilidad
            policy = (mu, log_sigma)
        else:
            policy = self.logits(actor_x)
        
        # Red del Crítico
        critic_x = x
        for layer in self.critic_layers:
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
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).to(CONST_DEVICE)
            if state_tensor.dim() == 1:
                state_tensor = state_tensor.unsqueeze(0)
                
            policy, _ = self.forward(state_tensor)
            
            if self.continuous:
                mu, log_sigma = policy
                if deterministic:
                    action = mu
                else:
                    sigma = torch.exp(log_sigma)
                    dist = Normal(mu, sigma)
                    action = dist.sample()
                action = action.cpu().numpy()[0]
            else:
                if deterministic:
                    action = torch.argmax(policy, dim=1)
                else:
                    dist = Categorical(logits=policy)
                    action = dist.sample()
                action = action.cpu().numpy()[0]
            
            return action
    
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
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).to(CONST_DEVICE)
            if state_tensor.dim() == 1:
                state_tensor = state_tensor.unsqueeze(0)
                
            _, value = self.forward(state_tensor)
            return value.cpu().numpy()[0, 0]
    
    def evaluate_actions(self, states: torch.Tensor, actions: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Evalúa las acciones tomadas, devolviendo log_probs, valores y entropía.
        
        Parámetros:
        -----------
        states : torch.Tensor
            Los estados observados
        actions : torch.Tensor
            Las acciones tomadas
        
        Retorna:
        --------
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor]
            (log_probs, valores, entropía)
        """
        policy, values = self.forward(states)
        
        if self.continuous:
            mu, log_sigma = policy
            sigma = torch.exp(log_sigma)
            dist = Normal(mu, sigma)
            entropy = 0.5 + 0.5 * np.log(2 * np.pi) + log_sigma
            entropy = entropy.sum(dim=-1)
            log_probs = dist.log_prob(actions).sum(dim=-1)
        else:
            dist = Categorical(logits=policy)
            entropy = dist.entropy()
            log_probs = dist.log_prob(actions.squeeze(-1))
        
        return log_probs, values.squeeze(-1), entropy


class A2C:
    """
    Implementación del algoritmo Advantage Actor-Critic (A2C) con PyTorch.
    
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
        seed: int = CONST_DEFAULT_SEED
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
            
        # Crear modelo
        self.model = ActorCriticModel(
            state_dim=state_dim, 
            action_dim=action_dim,
            continuous=continuous,
            hidden_units=self.hidden_units
        )
        
        # Optimizador
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate, weight_decay=A2C_A3C_CONFIG.get('weight_decay', 1e-4))
        
        # Métricas
        self.metrics = {
            CONST_POLICY_LOSS: [],
            CONST_VALUE_LOSS: [],
            CONST_ENTROPY_LOSS: [],
            CONST_TOTAL_LOSS: [],
            CONST_EPISODE_REWARDS: []
        }
        
        # Inicializar semilla
        torch.manual_seed(seed)
        np.random.seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
        
        # Generador de números aleatorios
        self.rng = np.random.Generator(np.random.PCG64(seed))
    
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
            # Delta para Ventaja Generalizada (GAE)
            delta = rewards[t] + self.gamma * values_extended[t + 1] * (1 - dones[t]) - values_extended[t]
            
            # Actualizar GAE
            gae = delta + self.gamma * 0.95 * (1 - dones[t]) * gae
            
            # Almacenar ventaja y retorno
            advantages[t] = gae
            returns[t] = gae + values_extended[t]
        
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
        episode_rewards : Optional[List], opcional
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
            
            # Obtener valor y acción
            current_value = self.model.get_value(current_state)
            action = self.model.get_action(current_state)
            
            # Ejecutar acción en el entorno
            next_state, reward, done, truncated, _ = env.step(action)
            
            # Guardar experiencia
            states.append(current_state)
            actions.append(action)
            rewards.append(reward)
            dones.append(done)
            values.append(current_value)
            
            # Actualizar recompensa del episodio
            current_episode_reward += reward
            
            # Si el episodio termina
            if done or truncated:
                # Guardar recompensa del episodio y resetear entorno
                if episode_rewards is not None:
                    episode_rewards.append(current_episode_reward)
                
                next_state, _ = env.reset()
                current_episode_reward = 0
            
            # Actualizar estado
            current_state = next_state
        
        return states, actions, rewards, dones, values, current_state, current_episode_reward, episode_rewards

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
        
        # Convertir a tensores de PyTorch
        states_tensor = torch.FloatTensor(states_np).to(CONST_DEVICE)
        actions_tensor = torch.FloatTensor(actions_np).to(CONST_DEVICE)
        returns_tensor = torch.FloatTensor(returns).to(CONST_DEVICE)
        advantages_tensor = torch.FloatTensor(advantages).to(CONST_DEVICE)
        
        # Actualizar modelo
        policy_loss, value_loss, entropy_loss = self.train_step(
            states_tensor, actions_tensor, returns_tensor, advantages_tensor
        )
        
        return policy_loss, value_loss, entropy_loss
    
    def train_step(self, states: torch.Tensor, actions: torch.Tensor, 
                  returns: torch.Tensor, advantages: torch.Tensor) -> Tuple[float, float, float]:
        """
        Ejecuta un paso de entrenamiento con los datos proporcionados.
        
        Parámetros:
        -----------
        states : torch.Tensor
            Estados observados
        actions : torch.Tensor
            Acciones tomadas
        returns : torch.Tensor
            Retornos calculados
        advantages : torch.Tensor
            Ventajas calculadas
            
        Retorna:
        --------
        Tuple[float, float, float]
            Pérdidas del entrenamiento (policy_loss, value_loss, entropy_loss)
        """
        # Reiniciar gradientes
        self.optimizer.zero_grad()
        
        # Calcular log_probs, valores y entropía
        log_probs, values, entropy = self.model.evaluate_actions(states, actions)
        
        # Pérdida de política
        policy_loss = -(log_probs * advantages).mean()
        
        # Pérdida de valor
        value_loss = F.mse_loss(values, returns)
        
        # Pérdida de entropía (negativa para fomentar exploración)
        entropy_loss = -entropy.mean()
        
        # Pérdida total
        total_loss = policy_loss + self.value_coef * value_loss + self.entropy_coef * entropy_loss
        
        # Retropropagación
        total_loss.backward()
        
        # Recorte de gradientes para estabilidad
        if self.max_grad_norm is not None:
            nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
            
        # Actualizar parámetros
        self.optimizer.step()
        
        # Actualizar métricas
        self.metrics[CONST_POLICY_LOSS].append(policy_loss.item())
        self.metrics[CONST_VALUE_LOSS].append(value_loss.item())
        self.metrics[CONST_ENTROPY_LOSS].append(entropy_loss.item())
        self.metrics[CONST_TOTAL_LOSS].append(total_loss.item())
        
        return policy_loss.item(), value_loss.item(), entropy_loss.item()

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
        history['policy_losses'].append(policy_loss)
        history['value_losses'].append(value_loss)
        history['entropy_losses'].append(np.mean(self.metrics[CONST_ENTROPY_LOSS]))
        
        # Añadir recompensas de episodios completados
        if episode_rewards:
            history['episode_rewards'].extend(episode_rewards)
            avg_reward = np.mean(episode_rewards)
            self.metrics[CONST_EPISODE_REWARDS].extend(episode_rewards)
            episode_rewards = []
            print(f"Época {epoch+1}/{epochs}, Pérdida Política: {policy_loss:.4f}, "
                  f"Pérdida Valor: {value_loss:.4f}, Recompensa Media: {avg_reward:.2f}")
        else:
            print(f"Época {epoch+1}/{epochs}, Pérdida Política: {policy_loss:.4f}, "
                  f"Pérdida Valor: {value_loss:.4f}")
        
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
            states, actions, rewards, dones, values, state, episode_reward, episode_rewards = self._collect_experience(
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
        Guarda el modelo en disco.
        
        Parámetros:
        -----------
        filepath : str
            Ruta donde guardar el modelo
        """
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'metrics': self.metrics,
            'continuous': self.continuous,
            'state_dim': self.state_dim,
            'action_dim': self.action_dim,
            'hidden_units': self.hidden_units,
            'gamma': self.gamma,
            'entropy_coef': self.entropy_coef,
            'value_coef': self.value_coef,
            'max_grad_norm': self.max_grad_norm
        }, filepath)
    
    def load_model(self, filepath: str) -> None:
        """
        Carga el modelo desde disco.
        
        Parámetros:
        -----------
        filepath : str
            Ruta desde donde cargar el modelo
        """
        checkpoint = torch.load(filepath, map_location=CONST_DEVICE)
        
        # Recrear el modelo con los parámetros guardados
        self.state_dim = checkpoint['state_dim']
        self.action_dim = checkpoint['action_dim']
        self.continuous = checkpoint['continuous']
        self.hidden_units = checkpoint['hidden_units']
        self.gamma = checkpoint.get('gamma', self.gamma)
        self.entropy_coef = checkpoint.get('entropy_coef', self.entropy_coef)
        self.value_coef = checkpoint.get('value_coef', self.value_coef)
        self.max_grad_norm = checkpoint.get('max_grad_norm', self.max_grad_norm)
        
        # Recrear el modelo y cargar los pesos
        self.model = ActorCriticModel(
            state_dim=self.state_dim,
            action_dim=self.action_dim,
            continuous=self.continuous,
            hidden_units=self.hidden_units
        )
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        
        # Recrear el optimizador y cargar su estado
        self.optimizer = optim.Adam(self.model.parameters(), 
                                    lr=checkpoint.get('learning_rate', A2C_A3C_CONFIG['learning_rate']), 
                                    weight_decay=A2C_A3C_CONFIG.get('weight_decay', 1e-4))
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        # Cargar métricas si existen
        self.metrics = checkpoint.get('metrics', {
            CONST_POLICY_LOSS: [],
            CONST_VALUE_LOSS: [],
            CONST_ENTROPY_LOSS: [],
            CONST_TOTAL_LOSS: [],
            CONST_EPISODE_REWARDS: []
        })


class A3C(A2C):
    """
    Implementación del algoritmo Asynchronous Advantage Actor-Critic (A3C) con PyTorch.
    
    Este algoritmo extiende A2C para habilitar entrenamiento asíncrono utilizando múltiples
    procesos trabajadores.
    
    Parámetros:
    -----------
    state_dim : int
        Dimensión del espacio de estados
    action_dim : int
        Dimensión del espacio de acciones
    continuous : bool
        Indica si el espacio de acciones es continuo o discreto
    n_workers : int
        Número de trabajadores para entrenamiento asíncrono
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
        n_workers: int = 4,
        learning_rate: float = A2C_A3C_CONFIG['learning_rate'],
        gamma: float = A2C_A3C_CONFIG['gamma'],
        entropy_coef: float = A2C_A3C_CONFIG['entropy_coef'],
        value_coef: float = A2C_A3C_CONFIG['value_coef'],
        max_grad_norm: float = A2C_A3C_CONFIG['max_grad_norm'],
        hidden_units: Optional[List[int]] = None,
        seed: int = CONST_DEFAULT_SEED
    ) -> None:
        super(A3C, self).__init__(
            state_dim, action_dim, continuous,
            learning_rate, gamma, entropy_coef,
            value_coef, max_grad_norm, hidden_units, seed
        )
        
        # Configurar para entrenamiento asíncrono
        self.n_workers = n_workers
        self.global_model = self.model
        self.global_optimizer = self.optimizer
        
        # Asegurar que el modelo global esté en CUDA si está disponible
        if hasattr(self.global_model, 'to') and torch.cuda.is_available():
            self.global_model = self.global_model.to(CONST_DEVICE)
        
        # Lista de trabajadores
        self.workers = []
        
        # Bloqueo para actualizaciones al modelo global
        self.model_lock = mp.Lock()
    
    def create_worker(self, env_fn: Callable, worker_id: int) -> 'A3CWorker':
        """
        Crea un trabajador A3C para el entrenamiento asíncrono.
        
        Parámetros:
        -----------
        env_fn : Callable
            Función que crea el entorno para el trabajador
        worker_id : int
            Identificador del trabajador
            
        Retorna:
        --------
        A3CWorker
            Trabajador creado
        """
        worker = A3CWorker(
            self.global_model, 
            self.global_optimizer,
            self.model_lock,
            env_fn,
            worker_id,
            self.state_dim,
            self.action_dim,
            self.gamma,
            self.entropy_coef,
            self.value_coef,
            self.max_grad_norm,
            self.continuous,
            self.hidden_units
        )
        self.workers.append(worker)
        return worker
    
    def train_async(self, env_fn: Callable, n_steps: int = 10, 
                   total_steps: int = 1000000, render: bool = False) -> Dict:
        """
        Entrena el modelo A3C usando múltiples trabajadores de forma asíncrona.
        
        Parámetros:
        -----------
        env_fn : Callable
            Función que crea un entorno para cada trabajador
        n_steps : int, opcional
            Número de pasos por actualización (default: 10)
        total_steps : int, opcional
            Número total de pasos de entrenamiento (default: 1000000)
        render : bool, opcional
            Si se debe renderizar el entorno (default: False)
            
        Retorna:
        --------
        Dict
            Historia de entrenamiento
        """
        # Historia compartida de entrenamiento
        shared_history = {
            'episode_rewards': [],
            'policy_losses': [],
            'value_losses': [],
            'entropy_losses': []
        }
        
        # Calcular pasos por trabajador
        steps_per_worker = total_steps // self.n_workers
        
        # Crear trabajadores si no existen
        if not self.workers:
            for i in range(self.n_workers):
                self.create_worker(env_fn, i)
        
        # Asegurarse de que todos los trabajadores estén en el dispositivo correcto
        for worker in self.workers:
            if hasattr(worker.local_model, 'to'):
                worker.local_model = worker.local_model.to(CONST_DEVICE)
        
        # Crear y comenzar procesos
        processes = []
        
        # Iniciar entrenamiento asíncrono
        for i, worker in enumerate(self.workers):
            p = mp.Process(
                target=worker.train,
                args=(n_steps, steps_per_worker, shared_history, self.model_lock, render)
            )
            p.start()
            processes.append(p)
        
        # Esperar a que terminen todos los procesos
        for p in processes:
            p.join()
        
        # Actualizar métricas del modelo global
        self.metrics[CONST_EPISODE_REWARDS].extend(shared_history['episode_rewards'])
        
        return shared_history


class A3CWorker:
    """
    Trabajador para el algoritmo A3C que ejecuta entrenamiento asíncrono.
    
    Parámetros:
    -----------
    global_model : ActorCriticModel
        Modelo compartido para todos los trabajadores
    global_optimizer : torch.optim.Optimizer
        Optimizador compartido para actualizar el modelo global
    model_lock : multiprocessing.Lock
        Bloqueo para actualizaciones sincronizadas al modelo global
    env_fn : Callable
        Función que crea el entorno para el trabajador
    worker_id : int
        Identificador del trabajador
    state_dim : int
        Dimensión del espacio de estados
    action_dim : int
        Dimensión del espacio de acciones
    gamma : float
        Factor de descuento para recompensas futuras
    entropy_coef : float
        Coeficiente para el término de entropía
    value_coef : float
        Coeficiente para la pérdida de la función de valor
    max_grad_norm : float
        Valor máximo para recorte de norma del gradiente
    continuous : bool
        Indica si el espacio de acciones es continuo o discreto
    hidden_units : Optional[List[int]]
        Unidades ocultas por capa
    """
    def __init__(
        self,
        global_model: ActorCriticModel,
        global_optimizer: torch.optim.Optimizer,
        model_lock: threading.Lock,
        env_fn: Callable,
        worker_id: int,
        state_dim: int,
        action_dim: int,
        gamma: float = 0.99,
        entropy_coef: float = 0.01,
        value_coef: float = 0.5,
        max_grad_norm: float = 0.5,
        continuous: bool = True,
        hidden_units: Optional[List[int]] = None
    ) -> None:
        # Guardar parámetros
        self.global_model = global_model
        self.global_optimizer = global_optimizer
        self.model_lock = model_lock
        self.env_fn = env_fn
        self.worker_id = worker_id
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.entropy_coef = entropy_coef
        self.value_coef = value_coef
        self.max_grad_norm = max_grad_norm
        self.continuous = continuous
        self.hidden_units = hidden_units if hidden_units is not None else A2C_A3C_CONFIG['hidden_units']
        
        # Crear entorno para este trabajador
        self.env = env_fn()
        
        # Crear modelo local (copia del global)
        self.local_model = ActorCriticModel(
            state_dim=state_dim,
            action_dim=action_dim,
            continuous=continuous,
            hidden_units=self.hidden_units
        )
        
        # Inicializar modelo local con los parámetros del global
        self.update_local_model()
        
        # Generador de números aleatorios con semilla basada en ID
        self.rng = np.random.Generator(np.random.PCG64(CONST_DEFAULT_SEED + worker_id))
    
    def update_local_model(self) -> None:
        """
        Actualiza los parámetros del modelo local desde el modelo global.
        """
        # Copiar parámetros del modelo global al local
        with self.model_lock:
            self.local_model.load_state_dict(self.global_model.state_dict())
    
    def compute_returns_advantages(self, rewards: np.ndarray, values: np.ndarray, 
                                 dones: np.ndarray, next_value: float) -> Tuple[np.ndarray, np.ndarray]:
        """
        Calcula los retornos y ventajas para los estados visitados.
        
        Parámetros:
        -----------
        rewards : np.ndarray
            Recompensas recibidas
        values : np.ndarray
            Valores estimados para los estados
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
            # Delta para Ventaja Generalizada (GAE)
            delta = rewards[t] + self.gamma * values_extended[t + 1] * (1 - dones[t]) - values_extended[t]
            
            # Actualizar GAE
            gae = delta + self.gamma * 0.95 * (1 - dones[t]) * gae
            
            # Almacenar ventaja y retorno
            advantages[t] = gae
            returns[t] = gae + values_extended[t]
        
        # Normalizar ventajas para reducir varianza
        if len(advantages) > 1:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        return returns, advantages
    
    def train_step(self, states: torch.Tensor, actions: torch.Tensor, 
                  returns: torch.Tensor, advantages: torch.Tensor) -> Tuple[float, float, float]:
        """
        Ejecuta un paso de entrenamiento y actualiza el modelo global.
        
        Parámetros:
        -----------
        states : torch.Tensor
            Estados observados
        actions : torch.Tensor
            Acciones tomadas
        returns : torch.Tensor
            Retornos calculados
        advantages : torch.Tensor
            Ventajas calculadas
            
        Retorna:
        --------
        Tuple[float, float, float]
            Pérdidas del entrenamiento (policy_loss, value_loss, entropy_loss)
        """
        # Calcular log_probs, valores y entropía con el modelo local
        log_probs, values, entropy = self.local_model.evaluate_actions(states, actions)
        
        # Pérdida de política
        policy_loss = -(log_probs * advantages).mean()
        
        # Pérdida de valor
        value_loss = F.mse_loss(values, returns)
        
        # Pérdida de entropía (negativa para fomentar exploración)
        entropy_loss = -entropy.mean()
        
        # Pérdida total
        total_loss = policy_loss + self.value_coef * value_loss + self.entropy_coef * entropy_loss
        
        # Calcular gradientes en el modelo local
        total_loss.backward()
        
        # Recortar gradientes para estabilidad
        if self.max_grad_norm is not None:
            nn.utils.clip_grad_norm_(self.local_model.parameters(), self.max_grad_norm)
        
        # Actualizar modelo global con los gradientes calculados
        with self.model_lock:
            for local_param, global_param in zip(self.local_model.parameters(), self.global_model.parameters()):
                if global_param.grad is None:
                    global_param.grad = local_param.grad.clone()
                else:
                    global_param.grad += local_param.grad.clone()
            
            # Paso de optimización en el modelo global
            self.global_optimizer.step()
            self.global_optimizer.zero_grad()
        
        # Actualizar modelo local desde el global
        self.update_local_model()
        
        return policy_loss.item(), value_loss.item(), entropy_loss.item()
    
    def _collect_step_data(self, state: np.ndarray, render: bool) -> Tuple:
        """
        Recolecta un paso de experiencia en el entorno.
        
        Parámetros:
        -----------
        state : np.ndarray
            Estado actual
        render : bool
            Si renderizar o no el entorno
            
        Retorna:
        --------
        Tuple
            (estado, acción, recompensa, estado siguiente, terminado, valor)
        """
        if render:
            self.env.render()
        
        # Obtener valor y acción
        current_value = self.local_model.get_value(state)
        action = self.local_model.get_action(state)
        
        # Ejecutar acción en el entorno
        next_state, reward, done, truncated, _ = self.env.step(action)
        
        return state, action, reward, next_state, done or truncated, current_value
    
    def _handle_episode_end(self, episode_reward: float, steps_done: int, 
                          max_steps: int, shared_history: Dict) -> Tuple[np.ndarray, float]:
        """
        Maneja el fin de un episodio y registra estadísticas.
        
        Parámetros:
        -----------
        episode_reward : float
            Recompensa acumulada del episodio
        steps_done : int
            Pasos realizados
        max_steps : int
            Pasos máximos para este trabajador
        shared_history : Dict
            Historial compartido para registrar métricas
            
        Retorna:
        --------
        Tuple[np.ndarray, float]
            (nuevo estado inicial, recompensa inicial del episodio)
        """
        # Registrar recompensa del episodio
        with self.model_lock:
            shared_history['episode_rewards'].append(episode_reward)
        
        # Mostrar progreso si es el trabajador principal
        if self.worker_id == 0 and len(shared_history['episode_rewards']) % 10 == 0:
            recent_rewards = shared_history['episode_rewards'][-10:]
            avg_reward = sum(recent_rewards) / len(recent_rewards)
            progress = steps_done / max_steps * 100
            print(f"Trabajador {self.worker_id}: {steps_done}/{max_steps} pasos ({progress:.1f}%), "
                  f"Recompensa media: {avg_reward:.2f}")
        
        # Resetear entorno para un nuevo episodio
        initial_state, _ = self.env.reset()
        
        return initial_state, 0.0
    
    def _update_model_with_collected_data(self, states: List, actions: List,
                                        rewards: List, dones: List,
                                        values: List, done: bool,
                                        next_state: np.ndarray, 
                                        shared_history: Dict) -> None:
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
        done : bool
            Si el último estado es terminal
        next_state : np.ndarray
            Estado siguiente al último en la secuencia
        shared_history : Dict
            Historial compartido para registrar métricas
        """
        # Si el episodio no terminó, usar el valor estimado del siguiente estado
        next_value = 0 if done else self.local_model.get_value(next_state)
        
        # Convertir a arrays de numpy
        states_np = np.array(states, dtype=np.float32)
        actions_np = np.array(actions, dtype=np.float32 if self.continuous else np.int32)
        rewards_np = np.array(rewards, dtype=np.float32)
        dones_np = np.array(dones, dtype=np.float32)
        values_np = np.array(values, dtype=np.float32)
        
        # Calcular retornos y ventajas
        returns, advantages = self.compute_returns_advantages(rewards_np, values_np, dones_np, next_value)
        
        # Convertir a tensores para el entrenamiento
        states_tensor = torch.FloatTensor(states_np).to(CONST_DEVICE)
        actions_tensor = torch.FloatTensor(actions_np).to(CONST_DEVICE)
        returns_tensor = torch.FloatTensor(returns).to(CONST_DEVICE)
        advantages_tensor = torch.FloatTensor(advantages).to(CONST_DEVICE)
        
        # Actualizar modelo
        policy_loss, value_loss, entropy_loss = self.train_step(
            states_tensor, actions_tensor, returns_tensor, advantages_tensor
        )
        
        # Registrar pérdidas
        with self.model_lock:
            shared_history['policy_losses'].append(policy_loss)
            shared_history['value_losses'].append(value_loss)
            shared_history['entropy_losses'].append(entropy_loss)
    
    def train(self, n_steps: int, max_steps: int, shared_history: Dict, 
             model_lock: threading.Lock, render: bool = False) -> None:
        """
        Entrena el modelo A3C como un trabajador.
        
        Parámetros:
        -----------
        n_steps : int
            Número de pasos por actualización
        max_steps : int
            Número máximo de pasos para este trabajador
        shared_history : Dict
            Historial compartido para registrar métricas
        model_lock : mp.Lock
            Bloqueo para actualizaciones al modelo global
        render : bool, opcional
            Si se debe renderizar el entorno (default: False)
        """
        # Inicializar estado
        state, _ = self.env.reset()
        episode_reward = 0.0
        steps_done = 0
        
        # Bucle principal de entrenamiento
        while steps_done < max_steps:
            # Listas para almacenar datos de n_steps
            states, actions, rewards, dones, values = [], [], [], [], []
            
            # Recolectar n_steps de experiencia
            for _ in range(n_steps):
                # Recolectar datos de un paso
                current_state, action, reward, next_state, done, value = self._collect_step_data(state, render)
                
                # Guardar experiencia
                states.append(current_state)
                actions.append(action)
                rewards.append(reward)
                dones.append(done)
                values.append(value)
                
                # Actualizar recompensa del episodio
                episode_reward += reward
                steps_done += 1
                
                # Actualizar estado
                state = next_state
                
                # Si el episodio termina o se alcanzan los pasos máximos
                if done or steps_done >= max_steps:
                    state, episode_reward = self._handle_episode_end(
                        episode_reward, steps_done, max_steps, shared_history
                    )
                    break
            
            # Actualizar modelo con los datos recolectados
            if states:
                self._update_model_with_collected_data(
                    states, actions, rewards, dones, values,
                    dones[-1], state, shared_history
                )


class A2CWrapper(nn.Module):
    """
    Wrapper para integrar el algoritmo A2C con la interfaz de entrenamiento PyTorch.
    
    Parámetros:
    -----------
    a2c_agent : A2C
        Agente A2C a encapsular
    cgm_shape : Tuple[int, ...]
        Forma de los datos CGM de entrada
    other_features_shape : Tuple[int, ...]
        Forma de otras características de entrada
    """
    def __init__(
        self, 
        a2c_agent: A2C,
        cgm_shape: Tuple[int, ...],
        other_features_shape: Tuple[int, ...]
    ) -> None:
        super(A2CWrapper, self).__init__()
        
        self.a2c_agent = a2c_agent
        self.cgm_shape = cgm_shape
        self.other_features_shape = other_features_shape
        
        # Crear capas de codificación para CGM
        self.cgm_encoder = nn.Sequential(
            nn.Conv1d(cgm_shape[-1], 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten()
        )
        
        # Crear capa de codificación para otras características
        self.other_encoder = nn.Sequential(
            nn.Linear(other_features_shape[-1], 32),
            nn.LayerNorm(32),
            nn.ReLU()
        )
        
        # Capa de codificación para variables contextuales
        self.context_encoder = nn.Sequential(
            nn.Linear(6, 16),  # 6 variables contextuales
            nn.LayerNorm(16),
            nn.ReLU()
        )
        
        # Capa para combinar características
        self.combined_layer = nn.Linear(64 + 32 + 16, a2c_agent.state_dim)
        
        # Capa para mapear salidas del agente a dosis
        output_dim = 1 if a2c_agent.continuous else a2c_agent.action_dim
        self.dose_predictor = nn.Linear(output_dim, 1)
        
        # Enviar a dispositivo adecuado
        self.to(CONST_DEVICE)
    
    def forward(self, cgm_data: torch.Tensor, other_features: torch.Tensor) -> torch.Tensor:
        """
        Realiza la pasada hacia adelante, integrando la codificación y el agente A2C.
        
        Parámetros:
        -----------
        cgm_data : torch.Tensor
            Datos CGM de entrada
        other_features : torch.Tensor
            Otras características de entrada
            
        Retorna:
        --------
        torch.Tensor
            Predicciones de dosis
        """
        # Codificar datos CGM (ajustar dimensiones para Conv1D)
        if cgm_data.dim() == 3:  # [batch, seq_len, features]
            cgm_data = cgm_data.permute(0, 2, 1)  # -> [batch, features, seq_len]
        elif cgm_data.dim() == 2:  # [seq_len, features]
            cgm_data = cgm_data.permute(1, 0).unsqueeze(0)  # -> [1, features, seq_len]
        
        cgm_features = self.cgm_encoder(cgm_data)
        
        # Codificar otras características
        other_encoded = self.other_encoder(other_features)
        
        # Codificar contexto
        context_encoded = self.context_encoder(other_features[:, -6:])
        
        # Combinar características
        combined = torch.cat([cgm_features, other_encoded, context_encoded], dim=1)
        state = self.combined_layer(combined)
        
        # Obtener acciones del agente A2C
        batch_size = cgm_data.shape[0]
        actions = []
        
        for i in range(batch_size):
            sample_state = state[i].detach().cpu().numpy()
            action = self.a2c_agent.model.get_action(sample_state, deterministic=True)
            actions.append(action)
        
        actions_np = np.array(actions)
        actions_tensor = torch.FloatTensor(actions_np).to(state.device)
        
        # Convertir acciones a dosis
        if self.a2c_agent.continuous:
            # Para acción continua, mapear directamente
            doses = self.dose_predictor(actions_tensor.view(batch_size, -1))
        else:
            # Para acción discreta, convertir a one-hot y luego a dosis
            one_hot = F.one_hot(actions_tensor.long(), self.a2c_agent.action_dim).float()
            doses = self.dose_predictor(one_hot)
        
        return doses
    
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
        Entrena el modelo A2C con los datos proporcionados.
        
        Parámetros:
        -----------
        x : List[torch.Tensor]
            Lista con [x_cgm, x_other]
        y : torch.Tensor
            Valores objetivo (dosis)
        validation_data : Optional[Tuple], opcional
            Datos de validación como ((x_cgm_val, x_other_val), y_val) (default: None)
        epochs : int, opcional
            Número de épocas (default: 10)
        batch_size : int, opcional
            Tamaño de lote (default: 32)
        callbacks : List, opcional
            Lista de callbacks (default: None)
        verbose : int, opcional
            Nivel de verbosidad (default: 1)
            
        Retorna:
        --------
        Dict
            Historial de entrenamiento
        """
        if verbose > 0:
            print("Entrenando modelo A2C...")
        
        # Extraer x_cgm y x_other de la lista
        x_cgm, x_other = x
        
        # Convertir tensores a numpy para el entorno
        x_cgm_np = x_cgm.cpu().numpy() if isinstance(x_cgm, torch.Tensor) else x_cgm
        x_other_np = x_other.cpu().numpy() if isinstance(x_other, torch.Tensor) else x_other
        y_np = y.cpu().numpy() if isinstance(y, torch.Tensor) else y
        
        # Crear entorno de entrenamiento
        env = self._create_training_environment(x_cgm_np, x_other_np, y_np)
        
        # Entrenar agente A2C
        history = self.a2c_agent.train(
            env=env,
            n_steps=batch_size,
            epochs=epochs,
            render=False
        )
        
        # Calibrar predictor de dosis
        self._calibrate_dose_predictor(y_np)
        
        if verbose > 0:
            print("Entrenamiento completado.")
        
        return history
    
    def _create_training_environment(self, cgm_data: np.ndarray, other_features: np.ndarray, 
                                  target_doses: np.ndarray) -> Any:
        """
        Crea un entorno de entrenamiento para el agente A2C.
        
        Parámetros:
        -----------
        cgm_data : np.ndarray
            Datos CGM
        other_features : np.ndarray
            Otras características
        target_doses : np.ndarray
            Dosis objetivo
            
        Retorna:
        --------
        Any
            Entorno para entrenamiento
        """
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
                
                # Para compatibilidad con algoritmos RL
                action_dim = model_wrapper.a2c_agent.action_dim
                
                # Definir espacios de observación y acción
                import gym
                from gym import spaces
                
                # Definir el espacio de observación basado en el estado del agente
                self.observation_space = spaces.Box(
                    low=-np.inf, high=np.inf, 
                    shape=(model_wrapper.a2c_agent.state_dim,)
                )
                
                # Definir espacio de acción según tipo (continuo o discreto)
                if model_wrapper.a2c_agent.continuous:
                    self.action_space = spaces.Box(
                        low=np.zeros(action_dim), 
                        high=np.ones(action_dim) * 15, 
                        dtype=np.float32
                    )
                else:
                    self.action_space = spaces.Discrete(action_dim)
                
                # Precargar codificadores para evitar problemas de inicialización
                self._initialize_encoders(model_wrapper)
            
            def _initialize_encoders(self, model_wrapper):
                """Inicializa los codificadores para evitar problemas durante el entrenamiento."""
                # Crear tensores vacíos para la inicialización
                dummy_cgm = torch.zeros((1,) + self.cgm.shape[1:], dtype=torch.float32).to(CONST_DEVICE)
                dummy_other = torch.zeros((1,) + self.features.shape[1:], dtype=torch.float32).to(CONST_DEVICE)
                
                # Pasar por el modelo para inicializar todas las capas
                _ = model_wrapper(dummy_cgm, dummy_other)
                
                # Guardar referencias a los codificadores
                self.cgm_encoder = model_wrapper.cgm_encoder
                self.other_encoder = model_wrapper.other_encoder
                self.combined_layer = model_wrapper.combined_layer
            
            def reset(self):
                """Reinicia el entorno seleccionando un ejemplo aleatorio."""
                self.current_idx = self.rng.integers(0, self.max_idx)
                state = self._get_state()
                return state, {}
            
            def step(self, action):
                """Ejecuta un paso en el entorno con la acción dada."""
                # Convertir acción a dosis
                if isinstance(self.action_space, gym.spaces.Box):
                    dose = np.clip(action[0], 0, 15)
                else:
                    dose = action / (self.model.a2c_agent.action_dim - 1) * 15
                
                # Obtener target y calcular error de dosis
                target = self.targets[self.current_idx]
                dose_error = abs(dose - target)
                
                # Normalizar error de dosis (asumiendo error máximo de 5 unidades)
                normalized_dose_error = min(1.0, dose_error / 5.0)
                dose_reward = -normalized_dose_error
                
                # Obtener y denormalizar glucosa
                current_cgm = self.cgm[self.current_idx]
                current_glucose_normalized = current_cgm[0, -1]  # Último valor
                
                # Denormalización segura
                try:
                    if hasattr(self, 'scaler_cgm') and self.scaler_cgm is not None:
                        current_glucose = self.scaler_cgm.inverse_transform([[current_glucose_normalized]])[0, 0]
                    else:
                        # Si no hay escaler, aproximar usando estadísticas conocidas
                        current_glucose = current_glucose_normalized * 60 + 140
                except Exception as e:
                    print(f"Error en denormalización: {e}")
                    current_glucose = current_glucose_normalized
                
                # Recompensa de glucosa usando constantes definidas
                if LOWER_BOUND_NORMAL_GLUCOSE_RANGE <= current_glucose <= UPPER_BOUND_NORMAL_GLUCOSE_RANGE:
                    # Dentro del rango: recompensa basada en cercanía al óptimo
                    distance_from_target = abs(current_glucose - TARGET_GLUCOSE)
                    max_distance_in_range = max(TARGET_GLUCOSE - LOWER_BOUND_NORMAL_GLUCOSE_RANGE, 
                                                UPPER_BOUND_NORMAL_GLUCOSE_RANGE - TARGET_GLUCOSE)
                    glucose_reward = POSITIVE_REWARD * (1 - distance_from_target / max_distance_in_range)
                elif current_glucose < LOWER_BOUND_NORMAL_GLUCOSE_RANGE:
                    # Hipoglucemia: penalización proporcional
                    hypoglycemia_severity = (LOWER_BOUND_NORMAL_GLUCOSE_RANGE - current_glucose) / 30
                    glucose_reward = SEVERE_PENALTY_REWARD * min(1.0, hypoglycemia_severity)
                else:  # current_glucose > UPPER_BOUND_NORMAL_GLUCOSE_RANGE
                    # Hiperglucemia: penalización proporcional
                    hyperglycemia_severity = (current_glucose - UPPER_BOUND_NORMAL_GLUCOSE_RANGE) / 120
                    glucose_reward = MILD_PENALTY_REWARD * min(1.0, hyperglycemia_severity)
                
                # Combinar recompensas con mejor balance
                # Ajustar los pesos de la recompensa para enfatizar la precisión de la dosis
                # reward = 0.8 * dose_reward + 0.2 * glucose_reward  # Dar más peso a la precisión de dosis

                # Alternativamente, usar una función de recompensa adaptativa
                dose_weight = 0.9 - 0.3 * min(1, abs(current_glucose - TARGET_GLUCOSE) / 50)  # Peso dinámico
                reward = dose_weight * dose_reward + (1 - dose_weight) * glucose_reward
                
                # Avanzar al siguiente ejemplo
                self.current_idx = (self.current_idx + 1) % self.max_idx
                
                # Obtener próximo estado
                next_state = self._get_state()
                
                # Episodio siempre termina después de un paso
                done = True
                truncated = False
                
                return next_state, reward, done, truncated, {}
            
            def _get_state(self):
                """Codifica el estado current al a partir de los datos."""
                # Obtener datos actuales
                cgm_sample = torch.tensor(
                    self.cgm[self.current_idx:self.current_idx+1],
                    dtype=torch.float32
                ).to(CONST_DEVICE)
                
                features_sample = torch.tensor(
                    self.features[self.current_idx:self.current_idx+1],
                    dtype=torch.float32
                ).to(CONST_DEVICE)
                
                # Ajustar dimensiones para Conv1d
                if cgm_sample.dim() == 3:
                    cgm_sample = cgm_sample.permute(0, 2, 1)
                
                # Procesar con capas de codificación
                with torch.no_grad():
                    cgm_encoded = self.cgm_encoder(cgm_sample)
                    other_encoded = self.other_encoder(features_sample)
                    
                    # Combinar características
                    combined = torch.cat([cgm_encoded, other_encoded], dim=1)
                    state = self.combined_layer(combined)
                
                return state.cpu().numpy()[0]
            
            def render(self, mode='human'):
                """Renderiza el entorno (no implementado)."""
                pass
        
        # Crear y devolver instancia del entorno
        return InsulinDosingEnv(cgm_data, other_features, target_doses, self)
    
    def _calibrate_dose_predictor(self, y: np.ndarray) -> None:
        """
        Ajusta la capa de predicción de dosis según el rango de valores objetivo.
        
        Parámetros:
        -----------
        y : np.ndarray
            Dosis objetivo para calibración
        """
        max_dose = np.max(y)
        min_dose = np.min(y)
        
        with torch.no_grad():
            # Ajustar pesos según el tipo de acción
            if self.a2c_agent.continuous:
                # Para acción continua, mapear [0, 1] a [min_dose, max_dose]
                self.dose_predictor.weight.data.fill_((max_dose - min_dose) / 5.0)
                self.dose_predictor.bias.data.fill_(min_dose)
            else:
                # Para acción discreta, distribuir valores equitativamente
                weight_val = (max_dose - min_dose) / self.a2c_agent.action_dim
                self.dose_predictor.weight.data.fill_(weight_val)
                self.dose_predictor.bias.data.fill_(min_dose)
    
    def predict(self, x: List[torch.Tensor], **context) -> np.ndarray:
        """
        Realiza predicciones con el modelo entrenado.
        
        Parámetros:
        -----------
        x : List[torch.Tensor]
            Lista con [x_cgm, x_other]
        context : dict, opcional
            Contexto adicional:
            - Nivel objetivo de glucosa (glucose_target)
            - Insulin on Board (IoB)
            - Consumo de carbohidratos (carb_intake)
            - Calidad de Sueño (sleep_quality, 1-10)
            - Intensidad del estrés laboral (work_stress_level, 1-10)
            - Intensidad de E (physical_activity, 1-10)
            
        Retorna:
        --------
        np.ndarray
            Predicciones de dosis
        """
        self.eval()
        with torch.no_grad():
            cgm_data, other_features = x
            
            # Convertir a tensores si son arrays numpy
            if not isinstance(cgm_data, torch.Tensor):
                cgm_data = torch.tensor(cgm_data, dtype=torch.float32)
            if not isinstance(other_features, torch.Tensor):
                other_features = torch.tensor(other_features, dtype=torch.float32)
            
            # Mover a dispositivo
            cgm_data = cgm_data.to(CONST_DEVICE)
            other_features = other_features.to(CONST_DEVICE)
            
            # Realizar predicciones
            doses = self(cgm_data, other_features)
            
            return doses.cpu().numpy()
    
    def predict_with_context(self, x: List[torch.Tensor], **context) -> float:
        """
        Realiza predicciones con el modelo entrenado considerando variables contextuales.
        
        Parámetros:
        -----------
        x : List[torch.Tensor]
            Lista con [x_cgm, x_other]
        context : dict, opcional
            Contexto adicional:
            - objective_glucose : float - Nivel objetivo de glucosa
            - iob : float - Insulin on Board
            - carb_intake : float - Consumo de carbohidratos
            - sleep_quality : float - Calidad de sueño (1-10)
            - work_intensity : float - Intensidad del trabajo (1-10)
            - exercise_intensity : float - Intensidad del ejercicio (1-10)
                
        Retorna:
        --------
        float
            Dosis recomendada a inyectar
        """
        self.eval()  # Establecer en modo evaluación
        
        with torch.no_grad():
            cgm_data, other_features = x
            
            # Convertir a tensores si son arrays numpy
            if not isinstance(cgm_data, torch.Tensor):
                cgm_data = torch.tensor(cgm_data, dtype=torch.float32)
            if not isinstance(other_features, torch.Tensor):
                other_features = torch.tensor(other_features, dtype=torch.float32)
            
            # Mover tensores al dispositivo adecuado
            cgm_data = cgm_data.to(CONST_DEVICE)
            other_features = other_features.to(CONST_DEVICE)
            
            # Crear tensor para variables contextuales
            context_values = [
                context.get('carb_intake', 0.0),
                context.get('iob', 0.0),
                context.get('objective_glucose', 0.0),
                context.get('sleep_quality', 5.0),
                context.get('work_intensity', 0.0),
                context.get('exercise_intensity', 0.0)
            ]
            
            context_tensor = torch.tensor(context_values, dtype=torch.float32).to(CONST_DEVICE)
            
            # Asegurar que tiene la dimensión correcta (añadir dimensión de lote si es necesario)
            if len(cgm_data.shape) > 1 and cgm_data.shape[0] > 1:
                context_tensor = context_tensor.unsqueeze(0).repeat(cgm_data.shape[0], 1)
            else:
                context_tensor = context_tensor.unsqueeze(0)
            
            # Codificar datos CGM (ajustar dimensiones para Conv1D)
            if cgm_data.dim() == 3:  # [batch, seq_len, features]
                cgm_data = cgm_data.permute(0, 2, 1)  # -> [batch, features, seq_len]
            elif cgm_data.dim() == 2:  # [seq_len, features]
                cgm_data = cgm_data.permute(1, 0).unsqueeze(0)  # -> [1, features, seq_len]
            
            # Procesar con el modelo
            cgm_features = self.cgm_encoder(cgm_data)
            other_encoded = self.other_encoder(other_features)
            
            # Combinar características y contexto
            if hasattr(self, 'context_encoder'):
                # Si hay un encoder específico para el contexto, usarlo
                context_encoded = self.context_encoder(context_tensor)
                combined = torch.cat([cgm_features, other_encoded, context_encoded], dim=1)
            else:
                # Concatenar contexto directamente con otras características
                enhanced_other = torch.cat([other_encoded, context_tensor], dim=1)
                combined = torch.cat([cgm_features, enhanced_other], dim=1)
            
            # Codificar el estado combinado
            state = self.combined_layer(combined)
            
            # Obtener acción del agente A2C
            sample_state = state[0].detach().cpu().numpy()
            action = self.a2c_agent.model.get_action(sample_state, deterministic=True)
            
            # Convertir acción a dosis
            action_tensor = torch.FloatTensor([action]).to(CONST_DEVICE)
            dose = self.dose_predictor(action_tensor.view(1, -1))
            
            # Retornar la dosis como un valor float
            return float(dose.item())
    
    def forward_with_context(self, x_cgm: torch.Tensor, x_other: torch.Tensor, 
                       carb_intake: float = None, iob: float = None, 
                       objective_glucose: float = None, sleep_quality: float = None, 
                       work_intensity: float = None, exercise_intensity: float = None) -> torch.Tensor:
        """
        Realiza un forward pass incluyendo variables contextuales adicionales.
        
        Parámetros:
        -----------
        x_cgm : torch.Tensor
            Datos CGM de entrada
        x_other : torch.Tensor
            Otras características de entrada
        carb_intake : float, opcional
            Ingesta de carbohidratos en gramos
        iob : float, opcional
            Insulina a bordo (insulina activa en el cuerpo)
        objective_glucose : float, opcional
            Nivel objetivo de glucosa en sangre
        sleep_quality : float, opcional
            Calidad del sueño (escala de 0-10)
        work_intensity : float, opcional
            Intensidad del trabajo/estrés (escala de 0-10)
        exercise_intensity : float, opcional
            Intensidad del ejercicio (escala de 0-10)
            
        Retorna:
        --------
        torch.Tensor
            Predicciones del modelo con contexto incorporado
        """
        # Verificar que el modelo esté inicializado
        if self.model is None:
            if self.is_class:
                try:
                    self.model = self.model_cls(**self.model_kwargs)
                    if isinstance(self.model, nn.Module):
                        self.model = self.model.to(self.device)
                except Exception as e:
                    print_debug(f"Error al inicializar el modelo: {e}")
                    raise ValueError(CONST_MODEL_INIT_ERROR.format("realizar forward pass"))
            else:
                raise ValueError(CONST_MODEL_INIT_ERROR.format("realizar forward pass"))
        
        # Crear tensor de contexto si hay variables contextuales
        context_values = [
            carb_intake if carb_intake is not None else 0.0,
            iob if iob is not None else 0.0,
            objective_glucose if objective_glucose is not None else 0.0,
            sleep_quality if sleep_quality is not None else 5.0,
            work_intensity if work_intensity is not None else 0.0,
            exercise_intensity if exercise_intensity is not None else 0.0
        ]
        
        # Crear tensor y repetir para cada muestra en el batch
        context_tensor = torch.tensor(context_values, dtype=torch.float32).to(self.device)
        if len(x_cgm.shape) > 1 and x_cgm.shape[0] > 1:
            context_tensor = context_tensor.unsqueeze(0).repeat(x_cgm.shape[0], 1)
        else:
            context_tensor = context_tensor.unsqueeze(0)
        
        # Verificar si el modelo tiene un método específico para forward con contexto
        if hasattr(self.model, 'forward_with_context'):
            return self.model.forward_with_context(x_cgm, x_other, context_tensor)
        
        # Alternativa: concatenar con otras características y usar forward estándar
        enhanced_features = torch.cat([x_other, context_tensor], dim=1)
        
        # Intentar usar el forward del modelo subyacente
        try:
            if (hasattr(self.model, 'forward') and callable(self.model.forward)) or hasattr(self.model, '__call__'):
                return self.model(x_cgm, enhanced_features)
        except Exception as e:
            print_debug(f"Error en forward con contexto: {e}")
        
        # Si no podemos usar forward directamente, intentar con predict_with_context
        try:
            # Convertir tensores a numpy para llamar a predict_with_context
            x_cgm_np = x_cgm.cpu().numpy()
            x_other_np = x_other.cpu().numpy()
            
            context_dict = {
                'carb_intake': carb_intake,
                'iob': iob,
                'objective_glucose': objective_glucose,
                'sleep_quality': sleep_quality,
                'work_intensity': work_intensity,
                'exercise_intensity': exercise_intensity
            }
            
            # Filtrar None values
            context_dict = {k: v for k, v in context_dict.items() if v is not None}
            
            # Llamar a predict_with_context
            predictions_np = self.predict_with_context(x_cgm_np, x_other_np, **context_dict)
            return torch.FloatTensor(predictions_np).to(self.device)
        except Exception as e:
            print_debug(f"Error al predecir con contexto: {e}")
            # Devolver un tensor de ceros como fallback
            return torch.zeros(x_cgm.shape[0], 1, device=self.device)


class A3CWrapper(A2CWrapper):
    """
    Wrapper para integrar el algoritmo A3C con la interfaz de entrenamiento PyTorch.
    
    Parámetros:
    -----------
    a3c_agent : A3C
        Agente A3C a encapsular
    cgm_shape : Tuple[int, ...]
        Forma de los datos CGM de entrada
    other_features_shape : Tuple[int, ...]
        Forma de otras características de entrada
    """
    def __init__(
        self, 
        a3c_agent: A3C,
        cgm_shape: Tuple[int, ...],
        other_features_shape: Tuple[int, ...]
    ) -> None:
        super(A3CWrapper, self).__init__(a3c_agent, cgm_shape, other_features_shape)
    
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
        Entrena el modelo A3C con los datos proporcionados de forma asíncrona.
        
        Parámetros:
        -----------
        x : List[torch.Tensor]
            Lista con [x_cgm, x_other]
        y : torch.Tensor
            Valores objetivo (dosis)
        validation_data : Optional[Tuple], opcional
            Datos de validación como ((x_cgm_val, x_other_val), y_val) (default: None)
        epochs : int, opcional
            Número de épocas (default: 10)
        batch_size : int, opcional
            Tamaño de lote (default: 32)
        callbacks : List, opcional
            Lista de callbacks (default: None)
        verbose : int, opcional
            Nivel de verbosidad (default: 1)
            
        Retorna:
        --------
        Dict
            Historial de entrenamiento
        """
        if verbose > 0:
            print("Entrenando modelo A3C de forma asíncrona...")
        
        # Extraer x_cgm y x_other de la lista
        x_cgm, x_other = x
        
        # Convertir tensores a numpy para el entorno
        x_cgm_np = x_cgm.cpu().numpy() if isinstance(x_cgm, torch.Tensor) else x_cgm
        x_other_np = x_other.cpu().numpy() if isinstance(x_other, torch.Tensor) else x_other
        y_np = y.cpu().numpy() if isinstance(y, torch.Tensor) else y
        
        # Crear función de entorno que puede ser llamada por cada trabajador
        def create_env():
            return self._create_training_environment(x_cgm_np, x_other_np, y_np)
        
        # Entrenar agente A3C de forma asíncrona
        history = self.a3c_agent.train_async(
            env_fn=create_env,
            n_steps=batch_size,
            total_steps=epochs * len(y_np),
            render=False
        )
        
        # Calibrar predictor de dosis
        self._calibrate_dose_predictor(y_np)
        
        if verbose > 0:
            print("Entrenamiento asíncrono completado.")
        
        return history


def create_a2c_model(cgm_shape: Tuple[int, ...], other_features_shape: Tuple[int, ...]) -> DRLModelWrapperPyTorch:
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
    DRLModelWrapperPyTorch
        Modelo A2C envuelto que implementa la interfaz del sistema
    """
    # Definir creador del modelo
    def model_creator(**kwargs) -> nn.Module:
        """
        Crea una instancia del modelo A2C.
        
        Retorna:
        --------
        nn.Module
            Instancia del modelo A2C wrapper
        """
        # Ignorar parámetros que no necesitamos como 'algorithm'
        state_dim = 64
        action_dim = 1
        continuous = True
        
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
            hidden_units=A2C_A3C_CONFIG['hidden_units'],
            seed=CONST_DEFAULT_SEED
        )
        
        # Crear y devolver el wrapper
        return A2CWrapper(
            a2c_agent=a2c_agent,
            cgm_shape=cgm_shape,
            other_features_shape=other_features_shape
        )
    
    # Devolver DRLModelWrapperPyTorch con el creador de modelos
    return DRLModelWrapperPyTorch(model_creator, algorithm='a2c')


def create_a3c_model(cgm_shape: Tuple[int, ...], other_features_shape: Tuple[int, ...]) -> DRLModelWrapperPyTorch:
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
   
    DRLModelWrapperPyTorch
        Modelo A3C envuelto que implementa la interfaz del sistema
    """
    # Definir creador del modelo
    def model_creator(**kwargs) -> nn.Module:
        """
        Crea una instancia del modelo A3C.
        
        Retorna:
        --------
        nn.Module
            Instancia del modelo A3C wrapper
        """
        # Ignorar parámetros que no necesitamos como 'algorithm'
        state_dim = 64
        action_dim = 1
        continuous = True
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
            hidden_units=A2C_A3C_CONFIG['hidden_units'],
            seed=CONST_DEFAULT_SEED
        )
        
        # Crear y devolver el wrapper
        return A3CWrapper(
            a3c_agent=a3c_agent,
            cgm_shape=cgm_shape,
            other_features_shape=other_features_shape
        )
    
    # Devolver DRLModelWrapperPyTorch con el creador de modelos
    return DRLModelWrapperPyTorch(model_creator, algorithm='a3c')


# Funciones para el registro de model creators
def model_creator_a2c() -> Callable[[Tuple[int, ...], Tuple[int, ...]], DRLModelWrapperPyTorch]:
    """
    Retorna una función para crear un modelo A2C compatible con la API del sistema.
    
    Retorna:
    --------
    Callable[[Tuple[int, ...], Tuple[int, ...]], DRLModelWrapperPyTorch]
        Función para crear el modelo con las formas de entrada especificadas
    """
    return create_a2c_model


def model_creator_a3c() -> Callable[[Tuple[int, ...], Tuple[int, ...]], DRLModelWrapperPyTorch]:
    """
    Retorna una función para crear un modelo A3C compatible con la API del sistema.
    
    Retorna:
    --------
    Callable[[Tuple[int, ...], Tuple[int, ...]], DRLModelWrapperPyTorch]
        Función para crear el modelo con las formas de entrada especificadas
    """
    return create_a3c_model