import os
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Normal
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Any, Optional, Union, Callable, Sequence
from torch.utils.data import Dataset, DataLoader, TensorDataset
import time
import threading
from types import SimpleNamespace

PROJECT_ROOT = os.path.abspath(os.getcwd())
sys.path.append(PROJECT_ROOT) 

from config.models_config import PPO_CONFIG
from constants.constants import CONST_DEFAULT_SEED, CONST_DEFAULT_EPOCHS, CONST_DEFAULT_BATCH_SIZE
from custom.DeepReinforcementLearning.drl_pt import DRLModelWrapperPyTorch

# Constantes para uso repetido
CONST_DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CONST_POLICY_LOSS = "policy_loss"
CONST_VALUE_LOSS = "value_loss"
CONST_ENTROPY_LOSS = "entropy_loss"
CONST_TOTAL_LOSS = "total_loss"
CONST_EPISODE_REWARDS = "episode_rewards"
FIGURES_DIR = os.path.join(PROJECT_ROOT, "figures", "pytorch", "ppo")
os.makedirs(FIGURES_DIR, exist_ok=True)

class ActorCriticModel(nn.Module):
    """
    Modelo Actor-Crítico para PPO que divide la arquitectura en redes para
    política (actor) y valor (crítico).
    
    Parámetros:
    -----------
    state_dim : int
        Dimensión del espacio de estados
    action_dim : int
        Dimensión del espacio de acciones
    hidden_units : Optional[List[int]], opcional
        Unidades en capas ocultas (default: None)
    """
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_units: Optional[List[int]] = None
    ) -> None:
        super(ActorCriticModel, self).__init__()
        
        # Valores predeterminados para capas ocultas
        if hidden_units is None:
            hidden_units = PPO_CONFIG['hidden_units']
        
        # Capas compartidas para procesamiento de estados
        self.shared_layers = nn.ModuleList()
        for i, units in enumerate(hidden_units[:2]):
            if i == 0:
                self.shared_layers.append(nn.Linear(state_dim, units))
            else:
                self.shared_layers.append(nn.Linear(hidden_units[i-1], units))
            self.shared_layers.append(nn.ReLU())
        
        # Red del Actor (política)
        self.actor_layers = nn.ModuleList()
        for i, units in enumerate(hidden_units[2:]):
            if i == 0:
                self.actor_layers.append(nn.Linear(hidden_units[1], units))
            else:
                self.actor_layers.append(nn.Linear(hidden_units[i+1], units))
            self.actor_layers.append(nn.ReLU())
        
        # Capa de salida del actor (mu y log_sigma para política gaussiana)
        self.mu = nn.Linear(hidden_units[-1], action_dim)
        self.log_sigma = nn.Linear(hidden_units[-1], action_dim)
        
        # Red del Crítico (valor)
        self.critic_layers = nn.ModuleList()
        for i, units in enumerate(hidden_units[2:]):
            if i == 0:
                self.critic_layers.append(nn.Linear(hidden_units[1], units))
            else:
                self.critic_layers.append(nn.Linear(hidden_units[i+1], units))
            self.critic_layers.append(nn.ReLU())
        
        # Capa de salida del crítico (valor del estado)
        self.value = nn.Linear(hidden_units[-1], 1)
        
        # Enviar modelo al dispositivo adecuado
        self.to(CONST_DEVICE)
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Pasa la entrada por el modelo Actor-Crítico.
        
        Parámetros:
        -----------
        x : torch.Tensor
            Tensor de entrada con los estados
            
        Retorna:
        --------
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor]
            (mu, sigma, value) - Parámetros de la distribución de política y valor estimado
        """
        x = x.to(CONST_DEVICE)
        
        # Capas compartidas
        for layer in self.shared_layers:
            x = layer(x)
        
        # Red del Actor
        actor_x = x
        for layer in self.actor_layers:
            actor_x = layer(actor_x)
        
        mu = self.mu(actor_x)
        log_sigma = self.log_sigma(actor_x)
        sigma = torch.exp(log_sigma)
        
        # Red del Crítico
        critic_x = x
        for layer in self.critic_layers:
            critic_x = layer(critic_x)
        
        value = self.value(critic_x)
        
        return mu, sigma, value
    
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
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(CONST_DEVICE)
            mu, sigma, _ = self(state_tensor)
            
            if deterministic:
                return mu.cpu().numpy()[0]
            
            # Muestrear de la distribución normal
            dist = Normal(mu, sigma)
            action = dist.sample()
            return action.cpu().numpy()[0]
    
    def evaluate_actions(self, states: torch.Tensor, actions: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Evalúa acciones dado un batch de estados, devolviendo log probs, valores y entropía.
        
        Parámetros:
        -----------
        states : torch.Tensor
            Tensor de estados
        actions : torch.Tensor
            Tensor de acciones a evaluar
            
        Retorna:
        --------
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor]
            (log_probs, valores, entropía)
        """
        mu, sigma, values = self(states)
        
        # Crear distribución normal y calcular log prob
        dist = Normal(mu, sigma)
        log_probs = dist.log_prob(actions).sum(dim=-1, keepdim=True)
        entropy = dist.entropy().sum(dim=-1, keepdim=True)
        
        return log_probs, values, entropy
    
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
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(CONST_DEVICE)
            _, _, value = self(state_tensor)
            return value.cpu().item()


class PPO:
    """
    Implementación del algoritmo Proximal Policy Optimization (PPO).
    
    Esta implementación utiliza el clipping de PPO para actualizar la política
    y un estimador de ventaja generalizada (GAE) para mejorar el aprendizaje.
    
    Parámetros:
    -----------
    state_dim : int
        Dimensión del espacio de estados
    action_dim : int
        Dimensión del espacio de acciones
    continuous : bool, opcional
        Si el espacio de acciones es continuo (default: True)
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
    max_grad_norm : float, opcional
        Norma máxima para recorte de gradientes (default: PPO_CONFIG['max_grad_norm'])
    seed : int, opcional
        Semilla para reproducibilidad (default: 42)
    """
    def __init__(
        self, 
        state_dim: int, 
        action_dim: int,
        continuous: bool = True,
        learning_rate: float = PPO_CONFIG['learning_rate'],
        gamma: float = PPO_CONFIG['gamma'],
        epsilon: float = PPO_CONFIG['clip_epsilon'],
        hidden_units: Optional[List[int]] = None,
        entropy_coef: float = PPO_CONFIG['entropy_coef'],
        value_coef: float = PPO_CONFIG['value_coef'],
        max_grad_norm: float = PPO_CONFIG['max_grad_norm'],
        seed: int = CONST_DEFAULT_SEED
    ) -> None:
        # Configurar semillas para reproducibilidad
        torch.manual_seed(seed)
        np.random.seed(seed)
        self.rng = np.random.Generator(np.random.PCG64(seed))
        
        # Parámetros del modelo
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.continuous = continuous
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
            
        # Crear modelo Actor-Crítico
        self.model = ActorCriticModel(state_dim, action_dim, self.hidden_units)
        
        # Optimizador
        self.optimizer = optim.Adam(
            self.model.parameters(), 
            lr=learning_rate,
            weight_decay=1e-5
        )
        
        # Contadores para seguimiento
        self.training_step = 0
        self.global_step = 0
        
        # Inicializar métricas
        self.metrics = {
            CONST_POLICY_LOSS: [],
            CONST_VALUE_LOSS: [],
            CONST_ENTROPY_LOSS: [],
            CONST_TOTAL_LOSS: [],
            CONST_EPISODE_REWARDS: []
        }
    
    def compute_returns_advantages(self, rewards: np.ndarray, values: np.ndarray, 
                                  dones: np.ndarray, next_value: float) -> Tuple[np.ndarray, np.ndarray]:
        """
        Calcula los retornos y ventajas utilizando GAE (Generalized Advantage Estimation).
        
        Parámetros:
        -----------
        rewards : np.ndarray
            Recompensas recibidas
        values : np.ndarray
            Valores estimados para los estados actuales
        dones : np.ndarray
            Indicadores de fin de episodio (1 si terminó, 0 si no)
        next_value : float
            Valor estimado para el estado final
            
        Retorna:
        --------
        Tuple[np.ndarray, np.ndarray]
            (retornos, ventajas)
        """
        # Inicializar arrays para almacenar resultados
        advantages = np.zeros_like(rewards)
        returns = np.zeros_like(rewards)
        
        # Calcular ventajas de atrás hacia adelante usando GAE
        last_gae = 0
        
        for t in reversed(range(len(rewards))):
            # Si es el último paso, usar next_value, si no, usar values[t+1]
            if t == len(rewards) - 1:
                next_val = next_value
                # Para el último paso, verificamos si el episodio terminó
                not_terminal = 1.0 - dones[t]
            else:
                next_val = values[t + 1]
                not_terminal = 1.0 - dones[t]
            
            # Calcular delta y la ventaja GAE
            delta = rewards[t] + self.gamma * next_val * not_terminal - values[t]
            last_gae = delta + self.gamma * 0.95 * not_terminal * last_gae
            advantages[t] = last_gae
        
        # Calcular retornos como ventajas + valores
        returns = advantages + values
        
        # Normalizar ventajas
        advantages = (advantages - np.mean(advantages)) / (np.std(advantages) + 1e-10)
        
        return returns, advantages
    
    def _collect_experience(self, env, state: np.ndarray, n_steps: int, render: bool = False,
                          episode_reward: float = 0, episode_rewards: Optional[List] = None) -> Tuple:
        """
        Recolecta experiencia ejecutando pasos en el entorno.
        
        Parámetros:
        -----------
        env : Any
            Entorno de interacción
        state : np.ndarray
            Estado inicial
        n_steps : int
            Número de pasos a ejecutar
        render : bool, opcional
            Si renderizar el entorno (default: False)
        episode_reward : float, opcional
            Recompensa acumulada del episodio (default: 0)
        episode_rewards : Optional[List], opcional
            Lista para almacenar recompensas de episodios (default: None)
            
        Retorna:
        --------
        Tuple
            Datos de experiencia recolectada
        """
        # Listas para almacenar datos
        states = []
        actions = []
        rewards = []
        dones = []
        values = []
        
        if episode_rewards is None:
            episode_rewards = []
        
        # Recolectar experiencia durante n_steps
        for _ in range(n_steps):
            # Guardar estado actual
            states.append(state)
            
            # Obtener valor estimado y acción a tomar
            value = self.model.get_value(state)
            action = self.model.get_action(state)
            
            # Ejecutar acción en el entorno
            next_state, reward, done, truncated, _ = env.step(action)
            
            # Renderizar si es necesario
            if render:
                env.render()
            
            # Guardar datos de este paso
            values.append(value)
            actions.append(action)
            rewards.append(reward)
            dones.append(float(done))
            
            # Actualizar recompensa acumulada del episodio
            episode_reward += reward
            self.global_step += 1
            
            # Si el episodio terminó, reiniciar el entorno
            if done or truncated:
                # Guardar recompensa del episodio
                episode_rewards.append(episode_reward)
                episode_reward = 0
                
                # Reiniciar entorno
                next_state, _ = env.reset()
            
            # Actualizar estado
            state = next_state
        
        # Obtener valor del último estado
        next_value = self.model.get_value(state)
        
        return states, actions, rewards, dones, values, next_value, state, episode_reward, episode_rewards
    
    def _update_model(self, states: List, actions: List, rewards: List, 
                    dones: List, values: List, next_value: float, 
                    done: bool) -> Tuple[float, float, float]:
        """
        Actualiza el modelo usando los datos recolectados.
        
        Parámetros:
        -----------
        states : List
            Lista de estados
        actions : List
            Lista de acciones
        rewards : List
            Lista de recompensas
        dones : List
            Lista de indicadores de terminación
        values : List
            Lista de valores estimados
        next_value : float
            Valor estimado del último estado
        done : bool
            Si el último estado es terminal
            
        Retorna:
        --------
        Tuple[float, float, float]
            (policy_loss, value_loss, entropy_loss)
        """
        # Convertir listas a arrays numpy
        states_arr = np.array(states)
        actions_arr = np.array(actions)
        rewards_arr = np.array(rewards)
        dones_arr = np.array(dones)
        values_arr = np.array(values)
        
        # Calcular retornos y ventajas
        returns, advantages = self.compute_returns_advantages(
            rewards_arr, values_arr, dones_arr, next_value
        )
        
        # Convertir a tensores PyTorch
        states_tensor = torch.FloatTensor(states_arr).to(CONST_DEVICE)
        actions_tensor = torch.FloatTensor(actions_arr).to(CONST_DEVICE)
        returns_tensor = torch.FloatTensor(returns).unsqueeze(1).to(CONST_DEVICE)
        advantages_tensor = torch.FloatTensor(advantages).unsqueeze(1).to(CONST_DEVICE)
        
        # Actualizar usando el método de train_step
        policy_loss, value_loss, entropy_loss = self.train_step(
            states_tensor, actions_tensor, returns_tensor, advantages_tensor
        )
        
        return policy_loss, value_loss, entropy_loss
    
    def train_step(self, states: torch.Tensor, actions: torch.Tensor, 
                  returns: torch.Tensor, advantages: torch.Tensor) -> Tuple[float, float, float]:
        """
        Ejecuta un paso de entrenamiento PPO.
        
        Parámetros:
        -----------
        states : torch.Tensor
            Tensor con estados
        actions : torch.Tensor
            Tensor con acciones
        returns : torch.Tensor
            Tensor con retornos
        advantages : torch.Tensor
            Tensor con ventajas
            
        Retorna:
        --------
        Tuple[float, float, float]
            (policy_loss, value_loss, entropy_loss)
        """
        # Obtener log probs, valores y entropía actuales
        old_log_probs, _, _ = self.model.evaluate_actions(states, actions)
        old_log_probs = old_log_probs.detach()
        
        # Evaluación de la política actual
        new_log_probs, new_values, entropy = self.model.evaluate_actions(states, actions)
        
        # Radio para PPO
        ratio = torch.exp(new_log_probs - old_log_probs)
        
        # Implementar clipping de PPO
        p1 = ratio * advantages
        p2 = torch.clamp(ratio, 1.0 - self.epsilon, 1.0 + self.epsilon) * advantages
        policy_loss = -torch.min(p1, p2).mean()
        
        # Pérdida del valor (error cuadrático medio)
        value_loss = 0.5 * ((new_values - returns) ** 2).mean()
        
        # Término de entropía para exploración
        entropy_loss = entropy.mean()
        
        # Pérdida total
        total_loss = policy_loss + self.value_coef * value_loss - self.entropy_coef * entropy_loss
        
        # Optimización
        self.optimizer.zero_grad()
        total_loss.backward()
        
        # Recorte de gradientes si es necesario
        if self.max_grad_norm is not None:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
        
        self.optimizer.step()
        self.training_step += 1
        
        # Devolver valores numéricos para métricas
        return policy_loss.item(), value_loss.item(), entropy_loss.item()
    
    def _update_history(self, history: Dict, episode_rewards: List, 
                      epoch: int, epochs: int, policy_loss: float, 
                      value_loss: float) -> List:
        """
        Actualiza el historial con métricas de entrenamiento.
        
        Parámetros:
        -----------
        history : Dict
            Diccionario con historial de métricas
        episode_rewards : List
            Lista de recompensas por episodio
        epoch : int
            Época actual
        epochs : int
            Número total de épocas
        policy_loss : float
            Pérdida de la política
        value_loss : float
            Pérdida del valor
            
        Retorna:
        --------
        List
            Lista actualizada de recompensas por episodio
        """
        # Actualizar historial de métricas
        history.setdefault(CONST_POLICY_LOSS, []).append(policy_loss)
        history.setdefault(CONST_VALUE_LOSS, []).append(value_loss)
        history.setdefault(CONST_EPISODE_REWARDS, []).extend(episode_rewards)
        
        # Mostrar progreso cada 10 épocas o en la última
        if epoch % 10 == 0 or epoch == epochs - 1:
            if episode_rewards:
                avg_reward = sum(episode_rewards) / len(episode_rewards)
                print(f"Época: {epoch+1}/{epochs}, "
                      f"Recompensa promedio: {avg_reward:.4f}, "
                      f"Policy Loss: {policy_loss:.4f}, "
                      f"Value Loss: {value_loss:.4f}")
            else:
                print(f"Época: {epoch+1}/{epochs}, "
                      f"Policy Loss: {policy_loss:.4f}, "
                      f"Value Loss: {value_loss:.4f}")
        
        return episode_rewards

    def train(self, env, n_steps: int = 10, epochs: int = CONST_DEFAULT_EPOCHS, 
             render: bool = False) -> Dict:
        """
        Entrena el modelo PPO en un entorno dado.
        
        Parámetros:
        -----------
        env : Any
            Entorno de entrenamiento
        n_steps : int, opcional
            Número de pasos por época (default: 10)
        epochs : int, opcional
            Número de épocas (default: 1000)
        render : bool, opcional
            Si renderizar el entorno (default: False)
            
        Retorna:
        --------
        Dict
            Historial de entrenamiento con métricas
        """
        history = {}
        episode_rewards = []
        state, _ = env.reset()
        episode_reward = 0
        
        for epoch in range(epochs):
            # Recolectar experiencia
            states, actions, rewards, dones, values, next_value, state, episode_reward, episode_rewards = \
                self._collect_experience(env, state, n_steps, render, episode_reward, episode_rewards)
            
            # Actualizar modelo con la experiencia recolectada
            policy_loss, value_loss, entropy_loss = self._update_model(
                states, actions, rewards, dones, values, next_value, dones[-1]
            )
            
            # Actualizar historial y mostrar progreso
            episode_rewards = self._update_history(
                history, episode_rewards, epoch, epochs, policy_loss, value_loss
            )
            
            # Actualizar métricas internas
            self.metrics[CONST_POLICY_LOSS].append(policy_loss)
            self.metrics[CONST_VALUE_LOSS].append(value_loss)
            self.metrics[CONST_ENTROPY_LOSS].append(entropy_loss)
            self.metrics[CONST_TOTAL_LOSS].append(policy_loss + self.value_coef * value_loss - self.entropy_coef * entropy_loss)
        
        # Calcular recompensa media de los últimos episodios para la historia
        if episode_rewards:
            history['mean_reward'] = sum(episode_rewards[-10:]) / min(len(episode_rewards), 10)
        
        return history
    
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
        
        # Guardar estado del modelo y parámetros de configuración
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'state_dim': self.state_dim,
            'action_dim': self.action_dim,
            'continuous': self.continuous,
            'gamma': self.gamma,
            'epsilon': self.epsilon,
            'entropy_coef': self.entropy_coef,
            'value_coef': self.value_coef,
            'hidden_units': self.hidden_units,
            'seed': self.seed,
            'training_step': self.training_step,
            'global_step': self.global_step
        }, filepath)
    
    def load_model(self, filepath: str) -> None:
        """
        Carga el modelo desde disco.
        
        Parámetros:
        -----------
        filepath : str
            Ruta desde donde cargar el modelo
        """
        # Cargar modelo guardado
        checkpoint = torch.load(filepath, map_location=CONST_DEVICE)
        
        # Restaurar parámetros de configuración
        self.state_dim = checkpoint['state_dim']
        self.action_dim = checkpoint['action_dim']
        self.continuous = checkpoint.get('continuous', True)
        self.gamma = checkpoint['gamma']
        self.epsilon = checkpoint['epsilon']
        self.entropy_coef = checkpoint['entropy_coef']
        self.value_coef = checkpoint['value_coef']
        self.hidden_units = checkpoint['hidden_units']
        self.seed = checkpoint['seed']
        self.training_step = checkpoint.get('training_step', 0)
        self.global_step = checkpoint.get('global_step', 0)
        
        # Recrear modelo si no existe o si las dimensiones no coinciden
        if not hasattr(self, 'model') or self.model.mu.out_features != self.action_dim:
            self.model = ActorCriticModel(self.state_dim, self.action_dim, self.hidden_units)
        
        # Cargar pesos del modelo
        self.model.load_state_dict(checkpoint['model_state_dict'])
        
        # Recrear optimizador y cargar su estado
        if not hasattr(self, 'optimizer'):
            self.optimizer = optim.Adam(
                self.model.parameters(), 
                lr=PPO_CONFIG['learning_rate'],
                weight_decay=1e-5
            )
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])


class PPOWrapper(nn.Module):
    """
    Wrapper para integrar el algoritmo PPO con la interfaz de entrenamiento PyTorch.
    
    Parámetros:
    -----------
    ppo_agent : PPO
        Agente PPO a encapsular
    cgm_shape : Tuple[int, ...]
        Forma de los datos CGM de entrada
    other_features_shape : Tuple[int, ...]
        Forma de otras características de entrada
    """
    def __init__(
        self, 
        ppo_agent: PPO,
        cgm_shape: Tuple[int, ...],
        other_features_shape: Tuple[int, ...]
    ) -> None:
        super(PPOWrapper, self).__init__()
        self.ppo_agent = ppo_agent
        self.cgm_shape = cgm_shape
        self.other_features_shape = other_features_shape
        
        # Capas de codificación
        self._setup_encoders()
        
        # Capa para transformar salidas a dosis
        self.dose_predictor = nn.Linear(1, 1)
        
        # Historiales para seguimiento
        self.history = {'loss': [], 'val_loss': []}
        
        # Mover al dispositivo disponible
        self.to(CONST_DEVICE)
    
    def _setup_encoders(self) -> None:
        """
        Configura las capas de codificación para los datos de entrada.
        """
        # Codificador para datos CGM 
        cnn_out_channels = 32
        self.cgm_encoder = nn.Sequential(
            nn.Conv1d(self.cgm_shape[1], cnn_out_channels, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten()
        )
        
        # Codificador para otras características
        self.other_encoder = nn.Sequential(
            nn.Linear(self.other_features_shape[0], 32),
            nn.ReLU()
        )
        
        # Capa para combinar características
        combined_features = cnn_out_channels + 32
        self.combined_layer = nn.Sequential(
            nn.Linear(combined_features, self.ppo_agent.state_dim),
            nn.ReLU()
        )
    
    def forward(self, cgm_data: torch.Tensor, other_features: torch.Tensor) -> torch.Tensor:
        """
        Realiza el forward pass a través del modelo.
        
        Parámetros:
        -----------
        cgm_data : torch.Tensor
            Datos CGM de entrada
        other_features : torch.Tensor
            Otras características de entrada
            
        Retorna:
        --------
        torch.Tensor
            Dosis predichas
        """
        # Aplicar encoders para obtener estado
        cgm_encoded = self.cgm_encoder(cgm_data.permute(0, 2, 1))
        other_encoded = self.other_encoder(other_features)
        
        # Combinar características
        combined = torch.cat([cgm_encoded, other_encoded], dim=1)
        
        # Obtener estado para PPO
        state = self.combined_layer(combined)
        
        # Usar agente PPO para obtener acción (dosis)
        with torch.no_grad():
            batch_size = state.shape[0]
            actions = torch.zeros(batch_size, 1, device=CONST_DEVICE)
            
            for i in range(batch_size):
                # Usar acción determinista para predicción
                action = self.ppo_agent.model.get_action(state[i].cpu().numpy(), deterministic=True)
                actions[i] = torch.tensor(action, device=CONST_DEVICE)
        
        # Transformar a dosis apropiada
        doses = self.dose_predictor(actions)
        
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
            Número de épocas de entrenamiento (default: 10)
        batch_size : int, opcional
            Tamaño de lote (default: 32)
        callbacks : List, opcional
            Lista de callbacks (no utilizado en esta implementación) (default: None)
        verbose : int, opcional
            Nivel de verbosidad (0=silencioso, 1=progreso) (default: 1)
            
        Retorna:
        --------
        Dict
            Historial de entrenamiento con métricas
        """
        # Desempaquetar datos de entrada
        cgm_data, other_features = x
        
        # Desempaquetar datos de validación si existen
        if validation_data is not None:
            val_x, val_y = validation_data
            val_cgm_data, val_other_features = val_x
        
        # Calibrar el predictor de dosis
        self._calibrate_dose_predictor(y)
        
        # Crear entorno de entrenamiento
        env = self._create_training_environment(cgm_data, other_features, y)
        
        # Entrenar agente PPO
        if verbose:
            print("Entrenando agente PPO...")
        
        self.ppo_agent.train(
            env=env,
            n_steps=batch_size,
            epochs=epochs,
            render=False
        )
        
        # Actualizar historial con métricas del entrenamiento
        self.history['policy_loss'] = self.ppo_agent.metrics.get(CONST_POLICY_LOSS, [])
        self.history['value_loss'] = self.ppo_agent.metrics.get(CONST_VALUE_LOSS, [])
        self.history['episode_rewards'] = self.ppo_agent.metrics.get(CONST_EPISODE_REWARDS, [])
        
        # Para compatibilidad con la interfaz de DL
        self.history['loss'] = self.ppo_agent.metrics.get(CONST_TOTAL_LOSS, [])
        
        # Calcular predicciones en datos de entrenamiento
        # (las predicciones pueden usarse para otras métricas si es necesario)
        _ = self.predict(x)
        
        # Calcular pérdida en datos de validación si existen
        if validation_data is not None:
            val_preds = self.predict([val_cgm_data, val_other_features])
            val_loss = np.mean((val_preds - val_y.cpu().numpy()) ** 2)
            self.history['val_loss'] = [val_loss]
        
        return self.history
    
    def _create_training_environment(
        self, 
        cgm_data: torch.Tensor, 
        other_features: torch.Tensor, 
        targets: torch.Tensor
    ) -> Any:
        """
        Crea un entorno de entrenamiento compatible con RL.
        
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
            Entorno de entrenamiento compatible con RL
        """
        # Convertir tensores a numpy
        cgm_np = cgm_data.cpu().numpy()
        other_np = other_features.cpu().numpy()
        targets_np = targets.cpu().numpy()
        
        # Definir entorno personalizado
        class InsulinDosingEnv:
            """Entorno personalizado para problema de dosificación de insulina."""
            
            def __init__(self, cgm, features, targets, wrapper):
                self.cgm = cgm
                self.features = features
                self.targets = targets
                self.wrapper = wrapper
                self.current_idx = 0
                self.max_idx = len(targets) - 1
                self.rng = np.random.Generator(np.random.PCG64(CONST_DEFAULT_SEED))
                
                # Definir espacios de observación y acción
                self.observation_space = SimpleNamespace(
                    shape=(wrapper.ppo_agent.state_dim,)
                )
                self.action_space = SimpleNamespace(
                    shape=(wrapper.ppo_agent.action_dim,),
                    sample=self._sample_action
                )
            
            def _sample_action(self):
                """Muestrea una acción aleatoria del espacio de acciones."""
                return self.rng.uniform(-1, 1, size=(self.wrapper.ppo_agent.action_dim,))
            
            def reset(self):
                """Reinicia el entorno y devuelve el estado inicial."""
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
                cgm_batch = torch.FloatTensor(self.cgm[self.current_idx:self.current_idx+1]).to(CONST_DEVICE)
                features_batch = torch.FloatTensor(self.features[self.current_idx:self.current_idx+1]).to(CONST_DEVICE)
                
                # Codificar a espacio de estado usando el wrapper
                cgm_encoded = self.wrapper.cgm_encoder(cgm_batch.permute(0, 2, 1))
                other_encoded = self.wrapper.other_encoder(features_batch)
                combined = torch.cat([cgm_encoded, other_encoded], dim=1)
                state = self.wrapper.combined_layer(combined)
                
                return state.cpu().numpy()[0]
        
        # Crear y devolver instancia del entorno
        return InsulinDosingEnv(cgm_np, other_np, targets_np, self)
    
    def _calibrate_dose_predictor(self, y: torch.Tensor) -> None:
        """
        Calibra la capa de predicción de dosis para mapear acciones a dosis apropiadas.
        
        Parámetros:
        -----------
        y : torch.Tensor
            Valores objetivo de dosis
        """
        # Determinar rango de dosis
        min_dose = float(y.min())
        max_dose = float(y.max())
        
        # Configurar capa lineal para mapear de [-1, 1] a [min_dose, max_dose]
        scale = (max_dose - min_dose) / 2.0
        bias = (min_dose + max_dose) / 2.0
        
        with torch.no_grad():
            self.dose_predictor.weight.fill_(scale)
            self.dose_predictor.bias.fill_(bias)
    
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
            Predicciones del modelo
        """
        self.eval()  # Establecer modo evaluación
        
        # Desempaquetar datos de entrada
        cgm_data, other_features = x
        
        # Convertir a tensores si son arrays numpy
        if isinstance(cgm_data, np.ndarray):
            cgm_data = torch.FloatTensor(cgm_data).to(CONST_DEVICE)
        if isinstance(other_features, np.ndarray):
            other_features = torch.FloatTensor(other_features).to(CONST_DEVICE)
        
        with torch.no_grad():
            # Obtener predicciones
            predictions = self(cgm_data, other_features)
            
            # Convertir a numpy para compatibilidad
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
        
        # Guardar agente PPO
        self.ppo_agent.save_model(f"{filepath}_ppo_agent.pt")
        
        # Guardar wrapper completo
        torch.save({
            'wrapper_state_dict': self.state_dict(),
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
        # Cargar agente PPO
        self.ppo_agent.load_model(f"{filepath}_ppo_agent.pt")
        
        # Cargar wrapper
        checkpoint = torch.load(f"{filepath}_wrapper.pt", map_location=CONST_DEVICE)
        self.load_state_dict(checkpoint['wrapper_state_dict'])
        self.history = checkpoint['history']
        self.cgm_shape = checkpoint['cgm_shape']
        self.other_features_shape = checkpoint['other_features_shape']
    
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
            "state_dim": self.ppo_agent.state_dim,
            "action_dim": self.ppo_agent.action_dim,
            "hidden_units": self.ppo_agent.hidden_units,
            "gamma": self.ppo_agent.gamma,
            "epsilon": self.ppo_agent.epsilon,
            "continuous": self.ppo_agent.continuous,
            "entropy_coef": self.ppo_agent.entropy_coef,
            "value_coef": self.ppo_agent.value_coef
        }


def create_ppo_model(cgm_shape: Tuple[int, ...], other_features_shape: Tuple[int, ...]) -> DRLModelWrapperPyTorch:
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
    DRLModelWrapperPyTorch
        Modelo PPO envuelto que implementa la interfaz del sistema
    """
    # Configurar el espacio de estados y acciones
    state_dim = 64  # Dimensión del espacio de estado latente
    action_dim = 1  # Una dimensión para la dosis continua
    
    # Definir creador del modelo
    def model_creator(**kwargs) -> nn.Module:
        # Crear agente PPO
        ppo_agent = PPO(
            state_dim=state_dim,
            action_dim=action_dim,
            continuous=True,
            learning_rate=PPO_CONFIG['learning_rate'],
            gamma=PPO_CONFIG['gamma'],
            epsilon=PPO_CONFIG['clip_epsilon'],
            hidden_units=PPO_CONFIG['hidden_units'],
            entropy_coef=PPO_CONFIG['entropy_coef'],
            value_coef=PPO_CONFIG['value_coef'],
            max_grad_norm=PPO_CONFIG['max_grad_norm'],
            seed=CONST_DEFAULT_SEED
        )
        
        # Crear wrapper PPO
        return PPOWrapper(
            ppo_agent=ppo_agent,
            cgm_shape=cgm_shape,
            other_features_shape=other_features_shape
        )
    
    # Devolver DRLModelWrapperPyTorch con el creador de modelos
    return DRLModelWrapperPyTorch(model_creator, algorithm='ppo')


def model_creator() -> Callable[[Tuple[int, ...], Tuple[int, ...]], DRLModelWrapperPyTorch]:
    """
    Retorna una función para crear un modelo PPO compatible con la API del sistema.
    
    Retorna:
    --------
    Callable[[Tuple[int, ...], Tuple[int, ...]], DRLModelWrapperPyTorch]
        Función para crear el modelo con las formas de entrada especificadas
    """
    return create_ppo_model