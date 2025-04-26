import os
import sys
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from torch.distributions import Categorical, Normal
from typing import Dict, List, Tuple, Optional, Union, Any, Callable

PROJECT_ROOT = os.path.abspath(os.getcwd())
sys.path.append(PROJECT_ROOT) 

from config.models_config import REINFORCE_CONFIG
from custom.printer import print_debug, print_warning
from custom.rl_model_wrapper import RLModelWrapperPyTorch

# Constantes para rutas y mensajes
CONST_FIGURES_DIR = os.path.join(PROJECT_ROOT, "figures", "pytorch", "reinforce_mcpg")
CONST_ENTRENANDO = "Entrenando agente REINFORCE..."
CONST_EPISODIO = "Episodio"
CONST_RECOMPENSA = "Recompensa"
CONST_PROMEDIO = "Promedio"
CONST_POLITICA_LOSS = "Pérdida política"
CONST_ENTROPIA = "Entropía"
CONST_EVALUANDO = "Evaluando modelo..."
CONST_RESULTADOS = "Resultados de evaluación:"
CONST_ERROR_DIMENSION = "Error: Dimensión incorrecta para el estado"
CONST_SALVANDO_MODELO = "Modelo guardado en"
CONST_CARGANDO_MODELO = "Modelo cargado desde"
CONST_BASELINE_LOSS = "Pérdida de baseline"
CONST_CREANDO_ENTORNO = "Creando entorno de entrenamiento..."

# Crear directorio para figuras si no existe
os.makedirs(CONST_FIGURES_DIR, exist_ok=True)

class REINFORCEPolicy(nn.Module):
    """
    Implementación de la política para el algoritmo REINFORCE.
    
    Esta red neuronal parametriza una política estocástica que
    el agente utiliza para seleccionar acciones.
    
    Parámetros:
    -----------
    state_dim : int
        Dimensión del espacio de estados
    action_dim : int
        Dimensión del espacio de acciones
    continuous : bool, opcional
        Si el espacio de acciones es continuo (default: False)
    hidden_sizes : List[int], opcional
        Tamaños de las capas ocultas (default: REINFORCE_CONFIG['hidden_units'])
    activation : str, opcional
        Función de activación para capas ocultas (default: 'relu')
    learning_rate : float, opcional
        Tasa de aprendizaje (default: 0.001)
    log_std_init : float, opcional
        Valor inicial para el logaritmo de la desviación estándar en caso continuo (default: -0.5)
    """
    
    def __init__(
        self, 
        state_dim: int, 
        action_dim: int, 
        continuous: bool = False,
        hidden_sizes: List[int] = REINFORCE_CONFIG['hidden_units'],
        activation: str = 'relu',
        learning_rate: float = 0.001,
        log_std_init: float = -0.5
    ) -> None:
        super(REINFORCEPolicy, self).__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.continuous = continuous
        self.hidden_sizes = hidden_sizes
        self.activation = activation
        self.learning_rate = learning_rate
        
        # Mapeo de strings de activación a funciones
        self.activation_fn = {
            'relu': nn.ReLU(),
            'tanh': nn.Tanh(),
            'sigmoid': nn.Sigmoid(),
            'elu': nn.ELU(),
            'leaky_relu': nn.LeakyReLU(),
            'gelu': nn.GELU(),
        }[activation]
        
        # Construir modelo
        layers = []
        input_dim = state_dim
        
        for size in hidden_sizes:
            layers.append(nn.Linear(input_dim, size))
            layers.append(self.activation_fn)
            input_dim = size
        
        self.shared_layers = nn.Sequential(*layers)
        
        # Capa de salida
        if continuous:
            # Para acciones continuas, predecir la media
            self.mean_layer = nn.Linear(input_dim, action_dim)
            # Parámetro de log_std (aprendible)
            self.log_std = nn.Parameter(torch.ones(action_dim) * log_std_init)
        else:
            # Para acciones discretas, predecir logits
            self.logits_layer = nn.Linear(input_dim, action_dim)
    
    def forward(self, x: torch.Tensor) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Pasa el estado a través de la red para obtener parámetros de distribución.
        
        Parámetros:
        -----------
        x : torch.Tensor
            Tensor del estado
            
        Retorna:
        --------
        Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]
            Si continuo: (media, log_std)
            Si discreto: logits
        """
        # Asegurar que la entrada tenga la forma correcta
        if len(x.shape) == 1:
            x = x.unsqueeze(0)
            
        x = self.shared_layers(x)
        
        if self.continuous:
            mean = self.mean_layer(x)
            return mean, self.log_std.expand_as(mean)
        else:
            logits = self.logits_layer(x)
            return logits
    
    def get_action(self, state: np.ndarray, deterministic: bool = False) -> Union[np.ndarray, int]:
        """
        Selecciona una acción basada en el estado actual.
        
        Parámetros:
        -----------
        state : np.ndarray
            Estado actual
        deterministic : bool, opcional
            Si usar selección determinística (default: False)
            
        Retorna:
        --------
        Union[np.ndarray, int]
            Acción seleccionada (int para discreto, np.ndarray para continuo)
        """
        with torch.no_grad():
            # Convertir a tensor si es array numpy
            if isinstance(state, np.ndarray):
                state_tensor = torch.FloatTensor(state)
            else:
                state_tensor = state
                
            if self.continuous:
                mean, log_std = self.forward(state_tensor)
                
                if deterministic:
                    return mean.cpu().numpy()
                
                std = torch.exp(log_std)
                normal = Normal(mean, std)
                action = normal.sample()
                return action.cpu().numpy()
            else:
                logits = self.forward(state_tensor)
                
                if deterministic:
                    action = torch.argmax(logits, dim=-1)
                    return action.item()
                
                probs = torch.softmax(logits, dim=-1)
                dist = Categorical(probs)
                action = dist.sample()
                return action.item()
    
    def get_log_prob(self, states: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        """
        Calcula log-probabilidades para estados y acciones dados.
        
        Parámetros:
        -----------
        states : torch.Tensor
            Tensor de estados
        actions : torch.Tensor
            Tensor de acciones
            
        Retorna:
        --------
        torch.Tensor
            Log-probabilidades
        """
        if self.continuous:
            mean, log_std = self.forward(states)
            std = torch.exp(log_std)
            normal = Normal(mean, std)
            return normal.log_prob(actions).sum(dim=-1)
        else:
            logits = self.forward(states)
            if len(actions.shape) == 1:
                # Si actions es un vector de índices, convertirlo a one-hot
                actions = torch.nn.functional.one_hot(actions.long(), num_classes=self.action_dim).float()
            
            # Calcular log_softmax y sumar sobre dimensión de acciones
            log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
            return torch.sum(actions * log_probs, dim=-1)
    
    def get_entropy(self, states: torch.Tensor) -> torch.Tensor:
        """
        Calcula la entropía de la política para estados dados.
        
        Parámetros:
        -----------
        states : torch.Tensor
            Tensor de estados
            
        Retorna:
        --------
        torch.Tensor
            Entropía de la política
        """
        if self.continuous:
            # Para distribución normal: 0.5 * log(2 * pi * e * var)
            _, log_std = self.forward(states)
            entropy = 0.5 + 0.5 * torch.log(2 * torch.tensor(np.pi) * torch.exp(log_std)**2)
            return entropy.sum(dim=-1)
        else:
            logits = self.forward(states)
            probs = torch.nn.functional.softmax(logits, dim=-1)
            log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
            entropy = -torch.sum(probs * log_probs, dim=-1)
            return entropy

class REINFORCEValueNetwork(nn.Module):
    """
    Red de valor (baseline) para REINFORCE.
    
    Esta red estima el valor esperado de los estados para reducir
    la varianza del estimador del gradiente de política.
    
    Parámetros:
    -----------
    state_dim : int
        Dimensión del espacio de estados
    hidden_sizes : List[int], opcional
        Tamaños de las capas ocultas (default: REINFORCE_CONFIG['hidden_units'])
    activation : str, opcional
        Función de activación para capas ocultas (default: 'relu')
    learning_rate : float, opcional
        Tasa de aprendizaje (default: 0.001)
    """
    
    def __init__(
        self, 
        state_dim: int, 
        hidden_sizes: List[int] = REINFORCE_CONFIG['hidden_units'],
        activation: str = 'relu',
        learning_rate: float = 0.001
    ) -> None:
        super(REINFORCEValueNetwork, self).__init__()
        self.state_dim = state_dim
        self.hidden_sizes = hidden_sizes
        self.activation = activation
        self.learning_rate = learning_rate
        
        # Mapeo de strings de activación a funciones
        self.activation_fn = {
            'relu': nn.ReLU(),
            'tanh': nn.Tanh(),
            'sigmoid': nn.Sigmoid(),
            'elu': nn.ELU(),
            'leaky_relu': nn.LeakyReLU(),
            'gelu': nn.GELU(),
        }[activation]
        
        # Construir modelo
        layers = []
        input_dim = state_dim
        
        for size in hidden_sizes:
            layers.append(nn.Linear(input_dim, size))
            layers.append(self.activation_fn)
            input_dim = size
        
        layers.append(nn.Linear(input_dim, 1))
        
        self.model = nn.Sequential(*layers)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Pasa el estado a través de la red para obtener el valor estimado.
        
        Parámetros:
        -----------
        x : torch.Tensor
            Tensor del estado
            
        Retorna:
        --------
        torch.Tensor
            Valor estimado del estado
        """
        # Asegurar que la entrada tenga la forma correcta
        if len(x.shape) == 1:
            x = x.unsqueeze(0)
            
        value = self.model(x)
        return value

class REINFORCE(nn.Module):
    """
    Implementación del algoritmo REINFORCE (Monte Carlo Policy Gradient).
    
    Este algoritmo aprende una política parametrizada directamente
    maximizando el retorno esperado utilizando ascenso por gradiente.
    
    Parámetros:
    -----------
    state_dim : int
        Dimensión del espacio de estados
    action_dim : int
        Dimensión del espacio de acciones
    continuous : bool, opcional
        Si el espacio de acciones es continuo (default: False)
    gamma : float, opcional
        Factor de descuento (default: REINFORCE_CONFIG['gamma'])
    policy_lr : float, opcional
        Tasa de aprendizaje para la política (default: REINFORCE_CONFIG['policy_lr'])
    value_lr : float, opcional
        Tasa de aprendizaje para la red de valor (default: REINFORCE_CONFIG['value_lr'])
    use_baseline : bool, opcional
        Si usar una función de valor como baseline (default: REINFORCE_CONFIG['use_baseline'])
    entropy_coef : float, opcional
        Coeficiente para regularización de entropía (default: REINFORCE_CONFIG['entropy_coef'])
    hidden_sizes : List[int], opcional
        Tamaños de las capas ocultas (default: REINFORCE_CONFIG['hidden_units'])
    seed : int, opcional
        Semilla para reproducibilidad (default: 42)
    """
    
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        continuous: bool = False,
        gamma: float = REINFORCE_CONFIG['gamma'],
        policy_lr: float = REINFORCE_CONFIG['policy_lr'],
        value_lr: float = REINFORCE_CONFIG['value_lr'],
        use_baseline: bool = REINFORCE_CONFIG['use_baseline'],
        entropy_coef: float = REINFORCE_CONFIG['entropy_coef'],
        hidden_sizes: List[int] = REINFORCE_CONFIG['hidden_units'],
        seed: int = 42
    ) -> None:
        super(REINFORCE, self).__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.continuous = continuous
        self.gamma = gamma
        self.use_baseline = use_baseline
        self.entropy_coef = entropy_coef
        
        # Fijar semilla para reproducibilidad
        torch.manual_seed(seed)
        np.random.seed(seed)
        
        # Generador de números aleatorios con semilla
        self.rng = np.random.Generator(np.random.PCG64(seed))
        
        # Crear política
        self.policy = REINFORCEPolicy(
            state_dim=state_dim,
            action_dim=action_dim,
            continuous=continuous,
            hidden_sizes=hidden_sizes
        )
        
        # Crear red de valor si se usa baseline
        self.value_network = None
        if use_baseline:
            self.value_network = REINFORCEValueNetwork(
                state_dim=state_dim,
                hidden_sizes=hidden_sizes
            )
        
        # Optimizadores
        self.policy_optimizer = optim.Adam(self.policy.parameters(), lr=policy_lr, weight_decay=REINFORCE_CONFIG.get('weight_decay', 1e-4))
        
        if use_baseline and self.value_network is not None:
            self.value_optimizer = optim.Adam(self.value_network.parameters(), lr=value_lr, weight_decay=REINFORCE_CONFIG.get('weight_decay', 1e-4))
        
        # Métricas
        self.policy_loss_metric = 0.0
        self.entropy_metric = 0.0
        self.returns_metric = 0.0
        self.baseline_loss_metric = 0.0
        
        # Historiales
        self.episode_rewards = []
        self.avg_rewards = []
        self.policy_losses = []
        self.entropy_values = []
    
    def process_batch(self, cgm_data: torch.Tensor, other_data: torch.Tensor) -> torch.Tensor:
        """
        Procesa un lote de datos para predecir dosis de insulina.
        
        Parámetros:
        -----------
        cgm_data : torch.Tensor
            Datos CGM de entrada
        other_data : torch.Tensor
            Otras características de entrada
            
        Retorna:
        --------
        torch.Tensor
            Predicciones de dosis de insulina
        """
        # Procesar entradas
        combined = self._process_inputs(cgm_data, other_data)
        
        # Para el caso continuo y discreto
        if self.continuous:
            # Usar la media directamente
            mean, _ = self.policy(combined)
            # Mapear a rango de dosis (0-20 unidades)
            scaled_mean = torch.sigmoid(mean) * 20.0
            return scaled_mean
        else:
            # Obtener distribución sobre acciones
            logits = self.policy(combined)
            # Convertir a probabilidades
            probs = torch.softmax(logits, dim=-1)
            
            # Vector de valores para cada nivel discreto (0 a 20 unidades)
            device = probs.device
            action_values = torch.linspace(0, 20, self.action_dim, device=device).view(1, -1)
            
            # Calcular valor esperado de forma vectorizada
            batch_size = probs.size(0)
            action_values = action_values.expand(batch_size, -1).unsqueeze(-1)
            expected_value = torch.bmm(probs.unsqueeze(1), action_values).squeeze(1)
            
            return expected_value
    
    def train_policy_step(
        self, 
        states: torch.Tensor, 
        actions: torch.Tensor, 
        returns: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Realiza un paso de entrenamiento para la red de política.
        
        Parámetros:
        -----------
        states : torch.Tensor
            Estados visitados
        actions : torch.Tensor
            Acciones tomadas
        returns : torch.Tensor
            Retornos calculados
            
        Retorna:
        --------
        Tuple[torch.Tensor, torch.Tensor]
            Tupla con (pérdida_política, entropía)
        """
        # Calcular log-probs y entropía
        log_probs = self.policy.get_log_prob(states, actions)
        entropy = self.policy.get_entropy(states)
        mean_entropy = entropy.mean()
        
        # Calcular ventajas usando baseline si corresponde
        if self.use_baseline and self.value_network is not None:
            values = self.value_network(states).squeeze()
            advantages = returns - values.detach()
        else:
            advantages = returns
        
        # Calcular pérdida de política (negativo porque buscamos maximizar)
        policy_loss = -(log_probs * advantages).mean()
        
        # Agregar término de entropía para fomentar exploración
        loss = policy_loss - self.entropy_coef * mean_entropy
        
        # Optimizar política
        self.policy_optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy.parameters(), 1.0)
        self.policy_optimizer.step()
        
        # Actualizar métricas
        self.policy_loss_metric = policy_loss.item()
        self.entropy_metric = mean_entropy.item()
        self.returns_metric = returns.mean().item()
        
        return policy_loss, mean_entropy
    
    def train_baseline_step(self, states: torch.Tensor, returns: torch.Tensor) -> torch.Tensor:
        """
        Realiza un paso de entrenamiento para la red de valor (baseline).
        
        Parámetros:
        -----------
        states : torch.Tensor
            Estados visitados
        returns : torch.Tensor
            Retornos calculados
            
        Retorna:
        --------
        torch.Tensor
            Pérdida del baseline
        """
        if self.value_network is None:
            return torch.tensor(0.0)
        
        # Predecir valores
        values = self.value_network(states).squeeze()
        
        # Calcular pérdida (MSE)
        baseline_loss = torch.mean((values - returns) ** 2)
        
        # Optimizar red de valor
        self.value_optimizer.zero_grad()
        baseline_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.value_network.parameters(), 1.0)
        self.value_optimizer.step()
        
        # Actualizar métrica
        self.baseline_loss_metric = baseline_loss.item()
        
        return baseline_loss
    
    def compute_returns(self, rewards: List[float]) -> np.ndarray:
        """
        Calcula los retornos descontados para cada paso de tiempo.
        
        Parámetros:
        -----------
        rewards : List[float]
            Lista de recompensas recibidas
            
        Retorna:
        --------
        np.ndarray
            Array de retornos descontados
        """
        returns = np.zeros_like(rewards, dtype=np.float32)
        future_return = 0.0
        
        # Calcular retornos desde el final del episodio
        for t in reversed(range(len(rewards))):
            future_return = rewards[t] + self.gamma * future_return
            returns[t] = future_return
        
        # Normalizar retornos para estabilidad
        if len(returns) > 1:
            returns = (returns - np.mean(returns)) / (np.std(returns) + 1e-8)
        
        return returns
    
    def _run_episode(
        self, 
        env: Any, 
        render: bool = False
    ) -> Tuple[List[np.ndarray], List[Union[int, np.ndarray]], List[float], float, int]:
        """
        Ejecuta un episodio completo y recolecta la experiencia.
        
        Parámetros:
        -----------
        env : Any
            Entorno de OpenAI Gym o compatible
        render : bool, opcional
            Si renderizar el entorno durante entrenamiento (default: False)
            
        Retorna:
        --------
        Tuple[List[np.ndarray], List[Union[int, np.ndarray]], List[float], float, int]
            Tupla con (estados, acciones, recompensas, recompensa_total, longitud_episodio)
        """
        if env is None:
            # Crear un episodio sintético si no hay entorno
            return self._create_synthetic_episode()
            
        state, _ = env.reset()
        done = False
        
        # Almacenar datos del episodio
        states = []
        actions = []
        rewards = []
        
        while not done:
            if render:
                env.render()
            
            # Almacenar estado
            states.append(state)
            
            # Seleccionar acción según la política actual
            action = self.policy.get_action(state)
            actions.append(action)
            
            # Ejecutar acción en el entorno
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            
            # Almacenar recompensa
            rewards.append(reward)
            
            # Actualizar estado
            state = next_state
        
        return states, actions, rewards, sum(rewards), len(rewards)
    
    def _create_synthetic_episode(self) -> Tuple[List[np.ndarray], List[Union[int, np.ndarray]], List[float], float, int]:
        """
        Crea un episodio sintético cuando no hay entorno disponible.
        
        Retorna:
        --------
        Tuple[List[np.ndarray], List[Union[int, np.ndarray]], List[float], float, int]
            Tupla con (estados, acciones, recompensas, recompensa_total, longitud_episodio)
        """
        # Crear un episodio sintético corto para entrenamiento simulado
        episode_length = 5
        
        # Generar estados aleatorios usando el generador con semilla
        states = [self.rng.standard_normal(self.state_dim) for _ in range(episode_length)]
        
        # Generar acciones usando la política actual
        actions = []
        for state in states:
            action = self.policy.get_action(state)
            actions.append(action)
        
        # Generar recompensas aleatorias usando el generador con semilla
        rewards = [self.rng.standard_normal() for _ in range(episode_length)]
        total_reward = sum(rewards)
        
        return states, actions, rewards, total_reward, episode_length
    
    def train_rl(
        self, 
        env: Any = None, 
        episodes: int = REINFORCE_CONFIG['episodes'],
        render: bool = False,
        log_interval: int = REINFORCE_CONFIG['log_interval']
    ) -> Dict[str, List[float]]:
        """
        Entrena el agente REINFORCE en el entorno dado o con datos sintéticos.
        
        Parámetros:
        -----------
        env : Any, opcional
            Entorno de OpenAI Gym o compatible (default: None)
        episodes : int, opcional
            Número de episodios de entrenamiento (default: REINFORCE_CONFIG['episodes'])
        render : bool, opcional
            Si renderizar el entorno durante entrenamiento (default: False)
        log_interval : int, opcional
            Cada cuántos episodios mostrar estadísticas (default: REINFORCE_CONFIG['log_interval'])
            
        Retorna:
        --------
        Dict[str, List[float]]
            Diccionario con historiales de entrenamiento
        """
        print(CONST_ENTRENANDO)
        if env is None:
            print("No hay entorno disponible, se usará entrenamiento sintético")
            
        history = {
            'loss': [],
            'val_loss': [],
            'policy_loss': [],
            'entropy': [],
            'episode_reward': [],
            'episode_length': []
        }
        
        for episode in range(episodes):
            # Ejecutar episodio
            states, actions, rewards, episode_reward, episode_length = self._run_episode(env, render)
            
            # Almacenar recompensa del episodio
            self.episode_rewards.append(episode_reward)
            
            # Calcular retornos
            returns = self.compute_returns(rewards)
            
            # Convertir a tensores
            states_tensor = torch.FloatTensor(np.vstack(states))
            
            if self.continuous:
                actions_tensor = torch.FloatTensor(np.vstack(actions))
            else:
                actions_tensor = torch.LongTensor(actions)
                
            returns_tensor = torch.FloatTensor(returns)
            
            # Actualizar red de valor (baseline) si se usa
            if self.use_baseline and self.value_network is not None:
                baseline_loss = self.train_baseline_step(states_tensor, returns_tensor)
                history['val_loss'].append(baseline_loss.item())
            else:
                history['val_loss'].append(0.0)
            
            # Actualizar política
            policy_loss, entropy = self.train_policy_step(states_tensor, actions_tensor, returns_tensor)
            
            # Actualizar historiales
            history['loss'].append(policy_loss.item())
            history['policy_loss'].append(policy_loss.item())
            history['entropy'].append(entropy.item())
            history['episode_reward'].append(episode_reward)
            history['episode_length'].append(episode_length)
            
            # Mostrar progreso periódicamente
            if (episode + 1) % log_interval == 0:
                avg_reward = np.mean(self.episode_rewards[-log_interval:])
                self.avg_rewards.append(avg_reward)
                print(f"{CONST_EPISODIO} {episode + 1}/{episodes}: "
                      f"{CONST_RECOMPENSA} {episode_reward:.2f}, "
                      f"{CONST_PROMEDIO} {avg_reward:.2f}, "
                      f"{CONST_POLITICA_LOSS} {policy_loss.item():.4f}, "
                      f"{CONST_ENTROPIA} {entropy.item():.4f}")
        
        return history
    
    def evaluate(
        self, 
        env: Any, 
        episodes: int = 10, 
        render: bool = False
    ) -> float:
        """
        Evalúa el agente entrenado en el entorno dado.
        
        Parámetros:
        -----------
        env : Any
            Entorno de OpenAI Gym o compatible
        episodes : int, opcional
            Número de episodios de evaluación (default: 10)
        render : bool, opcional
            Si renderizar el entorno durante evaluación (default: False)
            
        Retorna:
        --------
        float
            Recompensa promedio obtenida
        """
        print(CONST_EVALUANDO)
        if env is None:
            return 0.0
        
        total_rewards = []
        
        for episode in range(episodes):
            state, _ = env.reset()
            done = False
            episode_reward = 0
            
            while not done:
                if render:
                    env.render()
                
                # Usar política determinística para evaluación
                action = self.policy.get_action(state, deterministic=True)
                
                # Ejecutar acción
                next_state, reward, terminated, truncated, _ = env.step(action)
                done = terminated or truncated
                
                # Acumular recompensa
                episode_reward += reward
                
                # Actualizar estado
                state = next_state
            
            total_rewards.append(episode_reward)
            print(f"{CONST_EPISODIO} {episode + 1}: {CONST_RECOMPENSA} {episode_reward:.2f}")
        
        avg_reward = np.mean(total_rewards)
        print(f"{CONST_RESULTADOS} {CONST_PROMEDIO} {CONST_RECOMPENSA}: {avg_reward:.2f}")
        
        return avg_reward
    
    def save(self, policy_path: str, value_path: Optional[str] = None) -> None:
        """
        Guarda el modelo en el disco.
        
        Parámetros:
        -----------
        policy_path : str
            Ruta donde guardar la red de política
        value_path : Optional[str], opcional
            Ruta donde guardar la red de valor (default: None)
        """
        torch.save(self.policy.state_dict(), policy_path)
        if self.use_baseline and self.value_network is not None and value_path:
            torch.save(self.value_network.state_dict(), value_path)
        print(f"{CONST_SALVANDO_MODELO} {policy_path}")

    def load(self, policy_path: str, value_path: Optional[str] = None) -> None:
        """
        Carga el modelo desde el disco.
        
        Parámetros:
        -----------
        policy_path : str
            Ruta desde donde cargar la red de política
        value_path : Optional[str], opcional
            Ruta desde donde cargar la red de valor (default: None)
        """
        self.policy.load_state_dict(torch.load(policy_path))
        if self.use_baseline and self.value_network is not None and value_path:
            self.value_network.load_state_dict(torch.load(value_path))
        print(f"{CONST_CARGANDO_MODELO} {policy_path}")

    def _process_inputs(self, x_cgm: Union[np.ndarray, torch.Tensor], x_other: Union[np.ndarray, torch.Tensor]) -> torch.Tensor:
        """
        Procesa las entradas para el modelo.
        
        Parámetros:
        -----------
        x_cgm : Union[np.ndarray, torch.Tensor]
            Datos CGM
        x_other : Union[np.ndarray, torch.Tensor]
            Otras características
            
        Retorna:
        --------
        torch.Tensor
            Tensor combinado para entrada al modelo
        """
        # Convertir a tensores si son arrays numpy
        if isinstance(x_cgm, np.ndarray):
            x_cgm = torch.FloatTensor(x_cgm)
        if isinstance(x_other, np.ndarray):
            x_other = torch.FloatTensor(x_other)
        
        # Reshape si es necesario
        if len(x_cgm.shape) > 2:
            batch_size = x_cgm.shape[0]
            x_cgm = x_cgm.reshape(batch_size, -1)
        if len(x_other.shape) > 2:
            batch_size = x_other.shape[0]
            x_other = x_other.reshape(batch_size, -1)
        
        # Asegurar dimensiones correctas
        if x_cgm.dim() == 1:
            x_cgm = x_cgm.unsqueeze(0)
        if x_other.dim() == 1:
            x_other = x_other.unsqueeze(0)
        
        # Concatenar características
        combined = torch.cat([x_cgm, x_other], dim=1)
        return combined

class REINFORCEModel(nn.Module):
    """
    Implementación de REINFORCE compatible con PyTorch y RLModelWrapperPyTorch.
    
    Esta clase envuelve la implementación de REINFORCE para hacerla compatible
    con el flujo de entrenamiento estándar de PyTorch.
    
    Parámetros:
    -----------
    cgm_shape : Tuple[int, ...]
        Forma de los datos CGM
    other_features_shape : Tuple[int, ...]
        Forma de otras características
    """
    
    def __init__(self, cgm_shape: Tuple[int, ...], other_features_shape: Tuple[int, ...], **kwargs) -> None:
        super(REINFORCEModel, self).__init__()
        
        # Calcular dimensiones
        self.cgm_shape = cgm_shape
        self.other_features_shape = other_features_shape
        cgm_dim = int(np.prod(cgm_shape))
        other_dim = int(np.prod(other_features_shape))
        state_dim = cgm_dim + other_dim
        
        # Crear el agente REINFORCE
        self.reinforce_agent = REINFORCE(
            state_dim=state_dim,
            action_dim=20,  # Niveles discretos de dosis
            continuous=False,  # Usar acciones discretas
            gamma=REINFORCE_CONFIG['gamma'],
            policy_lr=REINFORCE_CONFIG['policy_lr'],
            value_lr=REINFORCE_CONFIG['value_lr'],
            use_baseline=REINFORCE_CONFIG['use_baseline'],
            entropy_coef=REINFORCE_CONFIG['entropy_coef'],
            hidden_sizes=REINFORCE_CONFIG['hidden_units']
        )
        
        # Para crear un entorno de simulación si es necesario
        self.sim_env = None
        
        # Metricas para seguimiento
        self.loss_tracker = 0.0
    
    def forward(self, cgm_data: torch.Tensor, other_features: torch.Tensor) -> torch.Tensor:
        """
        Realiza una pasada forward del modelo.
        
        Parámetros:
        -----------
        cgm_data : torch.Tensor
            Datos CGM
        other_features : torch.Tensor
            Otras características
            
        Retorna:
        --------
        torch.Tensor
            Predicciones de dosis de insulina
        """
        return self.reinforce_agent.process_batch(cgm_data, other_features)
    
    def training_step(self, batch: Tuple[torch.Tensor, torch.Tensor], targets: torch.Tensor) -> torch.Tensor:
        """
        Realiza un paso de entrenamiento con un lote de datos.
        
        Parámetros:
        -----------
        batch : Tuple[torch.Tensor, torch.Tensor]
            Lote de datos de entrada (cgm_data, other_features)
        targets : torch.Tensor
            Valores objetivo (dosis de insulina)
            
        Retorna:
        --------
        torch.Tensor
            Pérdida del lote
        """
        cgm_data, other_features = batch
        
        if self.training:
            # Si estamos en modo entrenamiento, realizar entrenamiento de RL
            # Si es necesario, crear un entorno de simulación
            if self.sim_env is None:
                print(CONST_CREANDO_ENTORNO)
                self.sim_env = self._create_training_environment((cgm_data, other_features), targets)
            
            # Entrenar con un mini-episodio
            history = self.reinforce_agent.train_rl(env=self.sim_env, episodes=1)
            
            # Actualizar métrica de pérdida
            if 'loss' in history and history['loss']:
                self.loss_tracker = history['loss'][-1]
        
        # Realizar predicción normal
        predictions = self(cgm_data, other_features)
        
        # Calcular pérdida MSE
        loss = nn.functional.mse_loss(predictions, targets)
        
        return loss
    
    def _create_training_environment(self, batch: Tuple[torch.Tensor, torch.Tensor], targets: torch.Tensor) -> Any:
        """
        Crea un entorno de entrenamiento para REINFORCE basado en los datos.
        
        Parámetros:
        -----------
        batch : Tuple[torch.Tensor, torch.Tensor]
            Tupla de datos de entrada (cgm_data, other_features)
        targets : torch.Tensor
            Valores objetivo (dosis de insulina)
            
        Retorna:
        --------
        Any
            Entorno de entrenamiento simulado
        """
        cgm_data, other_features = batch
        
        # Convertir a numpy para más fácil manipulación
        cgm_np = cgm_data.detach().cpu().numpy()
        other_np = other_features.detach().cpu().numpy()
        targets_np = targets.detach().cpu().numpy()
        
        # Clase simple de entorno para entrenamiento
        class InsulinEnvironment:
            def __init__(self, cgm, other, targets, model) -> None:
                self.cgm = cgm
                self.other = other
                self.targets = targets
                self.model = model
                self.current_idx = 0
                self.max_steps = 10
                self.step_count = 0
                self.max_idx = len(targets) - 1
                # Usar generador para muestrar aleatorio seguro
                self.rng = np.random.Generator(np.random.PCG64(42))
                
            def reset(self) -> Tuple[np.ndarray, Dict]:
                """Reinicia el entorno y devuelve la observación inicial."""
                self.current_idx = self.rng.integers(0, self.max_idx)
                self.step_count = 0
                combined_state = self._get_state()
                return combined_state, {}
                
            def step(self, action: Union[int, np.ndarray]) -> Tuple[np.ndarray, float, bool, bool, Dict]:
                """Ejecuta un paso en el entorno con la acción dada."""
                # Convertir acción a dosis de insulina
                if isinstance(action, np.ndarray):
                    insulin_dose = action[0]
                else:
                    # Convertir acción discreta a dosis continua
                    insulin_dose = action * (20.0 / (self.model.reinforce_agent.action_dim - 1))
                
                # Calcular recompensa como negativo del error cuadrático
                target_dose = self.targets[self.current_idx]
                error = np.abs(insulin_dose - target_dose)
                reward = -error**2
                
                # Avanzar al siguiente estado
                self.current_idx = (self.current_idx + 1) % self.max_idx
                self.step_count += 1
                
                # Comprobar si el episodio ha terminado
                done = self.step_count >= self.max_steps
                
                # Obtener el nuevo estado
                next_state = self._get_state()
                
                return next_state, reward, done, False, {}
                
            def _get_state(self) -> np.ndarray:
                """Obtiene el estado actual para la política."""
                # Combinar CGM y otras características
                current_cgm = self.cgm[self.current_idx]
                current_other = self.other[self.current_idx]
                
                # Aplanar si es necesario
                if len(current_cgm.shape) > 1:
                    current_cgm = current_cgm.flatten()
                if len(current_other.shape) > 1:
                    current_other = current_other.flatten()
                
                # Concatenar características
                return np.concatenate([current_cgm, current_other])
        
        # Crear y devolver el entorno
        return InsulinEnvironment(cgm_np, other_np, targets_np, self)

def create_reinforce_model(cgm_shape: Tuple[int, ...], other_features_shape: Tuple[int, ...]) -> nn.Module:
    """
    Crea un modelo REINFORCE para las formas de entrada especificadas.
    
    Parámetros:
    -----------
    cgm_shape : Tuple[int, ...]
        Forma de los datos CGM
    other_features_shape : Tuple[int, ...]
        Forma de otras características
        
    Retorna:
    --------
    nn.Module
        Instancia del modelo REINFORCE
    """
    return REINFORCEModel(cgm_shape, other_features_shape)

def create_reinforce_mcpg_model_wrapper(cgm_shape: Tuple[int, ...], other_features_shape: Tuple[int, ...], **kwargs) -> RLModelWrapperPyTorch:
    """
    Crea un wrapper para el modelo REINFORCE compatible con RLModelWrapperPyTorch.
    
    Parámetros:
    -----------
    cgm_shape : Tuple[int, ...]
        Forma de los datos CGM
    other_features_shape : Tuple[int, ...]
        Forma de otras características
    kwargs : dict
        Argumentos adicionales para el modelo
        
    Retorna:
    --------
    RLModelWrapperPyTorch
        Wrapper del modelo REINFORCE
    """
    def model_creator_fn() -> nn.Module:
        return create_reinforce_model(cgm_shape, other_features_shape)
    
    # Crear wrapper para el modelo
    model_wrapper = RLModelWrapperPyTorch(model_cls=model_creator_fn)
    
    return model_wrapper

def model_creator() -> Callable[[Tuple[int, ...], Tuple[int, ...]], RLModelWrapperPyTorch]:
    """
    Retorna una función creadora de modelos REINFORCE compatible con la infraestructura.
    
    Retorna:
    --------
    Callable[[Tuple[int, ...], Tuple[int, ...]], RLModelWrapperPyTorch]
        Función que crea un wrapper REINFORCE para las formas de entrada dadas
    """
    return create_reinforce_mcpg_model_wrapper