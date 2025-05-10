import os
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from typing import Dict, List, Tuple, Optional, Union, Any, Callable

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../.."))
sys.path.append(PROJECT_ROOT)

from config.models_config import MONTE_CARLO_CONFIG
from constants.constants import CONST_DEFAULT_SEED
from custom.rl_model_wrapper import RLModelWrapperPyTorch
from custom.printer import print_warning

# Constante para directorio de figuras
FIGURES_DIR = os.path.join(PROJECT_ROOT, 'figures', 'pytorch', 'monte_carlo')
os.makedirs(FIGURES_DIR, exist_ok=True)

class MonteCarlo:
    """
    Implementación de métodos Monte Carlo para predicción y control en aprendizaje por refuerzo usando PyTorch.
    
    Esta clase proporciona implementaciones de:
    1. Predicción Monte Carlo (first-visit y every-visit) para evaluar políticas
    2. Control Monte Carlo (on-policy y off-policy) para encontrar políticas óptimas
    
    Se incluyen algoritmos como:
    - First-visit MC prediction
    - Every-visit MC prediction
    - Monte Carlo Exploring Starts (MCES)
    - On-policy MC control con epsilon-greedy
    - Off-policy MC control con importance sampling
    """
    
    def __init__(
        self,
        n_states: int,
        n_actions: int,
        gamma: float = MONTE_CARLO_CONFIG['gamma'],
        epsilon_start: float = MONTE_CARLO_CONFIG['epsilon_start'],
        epsilon_end: float = MONTE_CARLO_CONFIG['epsilon_end'],
        epsilon_decay: float = MONTE_CARLO_CONFIG['epsilon_decay'],
        first_visit: bool = MONTE_CARLO_CONFIG['first_visit'],
        evaluation_mode: bool = False,
        seed: int = CONST_DEFAULT_SEED,
        cgm_shape: Optional[Tuple[int, ...]] = None,
        other_features_shape: Optional[Tuple[int, ...]] = None
    ) -> None:
        """
        Inicializa el agente Monte Carlo.

        Parámetros:
        -----------
        n_states : int
            Número de estados discretos.
        n_actions : int
            Número de acciones discretas.
        gamma : float, opcional
            Factor de descuento (default: MONTE_CARLO_CONFIG['gamma']).
        epsilon_start : float, opcional
            Valor inicial de epsilon para exploración (default: MONTE_CARLO_CONFIG['epsilon_start']).
        epsilon_end : float, opcional
            Valor final de epsilon (default: MONTE_CARLO_CONFIG['epsilon_end']).
        epsilon_decay : float, opcional
            Tasa de decaimiento de epsilon (default: MONTE_CARLO_CONFIG['epsilon_decay']).
        first_visit : bool, opcional
            Si usar first-visit MC (True) o every-visit MC (False) (default: MONTE_CARLO_CONFIG['first_visit']).
        evaluation_mode : bool, opcional
            Si el agente está en modo evaluación (sin exploración) (default: False).
        seed : int, opcional
            Semilla para el generador de números aleatorios (default: 42).
        cgm_shape : Optional[Tuple[int, ...]], opcional
            Forma de los datos CGM (puede ser necesario para mapeo estado-observación) (default: None).
        other_features_shape : Optional[Tuple[int, ...]], opcional
            Forma de otras características (puede ser necesario para mapeo estado-observación) (default: None).
        """
        self.n_states = n_states
        self.n_actions = n_actions
        self.gamma = gamma
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.epsilon = epsilon_start
        self.first_visit = first_visit
        self.evaluation_mode = evaluation_mode
        
        # Configurar la semilla para reproducibilidad
        torch.manual_seed(seed)
        self.rng = np.random.Generator(np.random.PCG64(seed))
        
        # Inicializar tablas Q y contadores
        self.q_table = torch.zeros((n_states, n_actions))
        self.returns_sum = torch.zeros((n_states, n_actions))
        self.returns_count = torch.zeros((n_states, n_actions), dtype=torch.long)
        
        # Para modo de evaluación
        self.v_table = torch.zeros(n_states)
        self.state_returns_sum = torch.zeros(n_states)
        self.state_returns_count = torch.zeros(n_states, dtype=torch.long)
        
        # Para off-policy Monte Carlo
        self.c_table = torch.zeros((n_states, n_actions))
        
        # Guardar formas de entrada para mapeo de estados
        self.cgm_shape = cgm_shape
        self.other_features_shape = other_features_shape
        
        # Política (inicialmente equiprobable)
        self.policy = torch.ones((n_states, n_actions)) / n_actions
        
        # Para seguimiento del entrenamiento
        self.episode_rewards = []
        self.episode_lengths = []
        self.policy_changes = []
        self.value_changes = []
        self.epsilon_history = []

    def update_policy(self, state: int) -> bool:
        """
        Actualiza la política para el estado dado basándose en los valores Q actuales.
        
        Parámetros:
        -----------
        state : int
            Estado para el cual actualizar la política
            
        Retorna:
        --------
        bool
            Boolean indicando si la política cambió
        """
        if self.evaluation_mode:
            return False
        
        old_action = torch.argmax(self.policy[state])
        best_action = torch.argmax(self.q_table[state])
        
        # Política epsilon-greedy basada en Q
        self.policy[state] = torch.zeros(self.n_actions)
        
        # Probabilidad pequeña de exploración
        self.policy[state] += self.epsilon / self.n_actions
        
        # Mayor probabilidad para la mejor acción
        self.policy[state][best_action] += (1 - self.epsilon)
        
        return old_action != best_action

    def select_action(self, state: int, explore: bool = True) -> int:
        """
        Selecciona una acción según la política actual, con exploración opcional.
        
        Parámetros:
        -----------
        state : int
            Estado actual
        explore : bool, opcional
            Si es True, usa política epsilon-greedy; si es False, usa política greedy (default: True)
            
        Retorna:
        --------
        int
            La acción seleccionada
        """
        if explore and self.rng.random() < self.epsilon:
            return self.rng.integers(0, self.n_actions)
        else:
            return torch.argmax(self.q_table[state]).item()

    def decay_epsilon(self, episode: Optional[int] = None) -> None:
        """
        Reduce el valor de epsilon según la estrategia de decaimiento.
        
        Parámetros:
        -----------
        episode : Optional[int], opcional
            Número del episodio actual (para decaimientos basados en episodios) (default: None)
        """
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)
        self.epsilon_history.append(self.epsilon)

    def calculate_returns(self, rewards: Union[List[float], np.ndarray, torch.Tensor]) -> torch.Tensor:
        """
        Calcula los retornos descontados para cada paso de tiempo en un episodio.
        
        Parámetros:
        -----------
        rewards : Union[List[float], np.ndarray, torch.Tensor]
            Lista de recompensas recibidas durante el episodio
            
        Retorna:
        --------
        torch.Tensor
            Lista de retornos (G_t) para cada paso de tiempo
        """
        # Convertir a tensor si no lo es ya
        if not isinstance(rewards, torch.Tensor):
            rewards = torch.tensor(rewards, dtype=torch.float32)
            
        returns = torch.zeros_like(rewards)
        g_value = 0.0
        
        # Recorremos las recompensas en orden inverso
        for t in range(len(rewards) - 1, -1, -1):
            g_value = rewards[t] + self.gamma * g_value
            returns[t] = g_value
            
        return returns

    def _map_observation_to_state(self, cgm_obs: np.ndarray, other_obs: np.ndarray) -> int:
        """
        Mapea una observación continua/compleja a un estado discreto.

        Parámetros:
        -----------
        cgm_obs : np.ndarray
            Observación CGM para una muestra.
        other_obs : np.ndarray
            Otras características para una muestra.

        Retorna:
        --------
        int
            El estado discreto correspondiente.
        """
        # Extraer características resumidas
        mean_cgm = np.mean(cgm_obs) if cgm_obs.size > 0 else 0.0
        
        other_feature_summary = 0.0
        if other_obs is not None and other_obs.size > 0 and np.issubdtype(other_obs.dtype, np.number):
            other_feature_summary = np.mean(other_obs) if other_obs.ndim > 0 and other_obs.size > 1 else other_obs.item()

        # Combinar características con ponderación
        combined_feature = 0.7 * mean_cgm + 0.3 * other_feature_summary

        # Discretizar la característica combinada
        min_val, max_val = 50, 250  # Rango esperado aproximado
        state = int(np.floor(((combined_feature - min_val) / (max_val - min_val)) * self.n_states))
        
        # Asegurar que el estado esté dentro de los límites válidos
        return max(0, min(state, self.n_states - 1))

    def run_episode(self, env: Any) -> List[Tuple[int, int, float]]:
        """
        Ejecuta un episodio completo en el entorno y devuelve las transiciones.
        
        Parámetros:
        -----------
        env : Any
            Entorno a ejecutar
            
        Retorna:
        --------
        List[Tuple[int, int, float]]
            Lista de tuplas (estado, acción, recompensa)
        """
        episode = []
        state_obs = env.reset()
        done = False
        
        while not done:
            state = self._map_observation_to_state(state_obs['cgm'], state_obs['other'])
            action = self.select_action(state, explore=True)
            next_state_obs, reward, done, _ = env.step(action)
            episode.append((state, action, reward))
            state_obs = next_state_obs
            
        return episode

    def update_q_table_from_episode(self, episode: List[Tuple[int, int, float]]) -> None:
        """
        Actualiza la tabla Q utilizando un episodio completo.
        
        Parámetros:
        -----------
        episode : List[Tuple[int, int, float]]
            Lista de tuplas (estado, acción, recompensa) que forman el episodio
        """
        g_value = 0.0
        visited_state_action = set()
        
        # Procesar el episodio en orden inverso
        for t in reversed(range(len(episode))):
            state, action, reward = episode[t]
            g_value = self.gamma * g_value + reward
            pair = (state, action)
            
            is_first_occurrence = pair not in visited_state_action
            if self.first_visit:
                visited_state_action.add(pair)
            
            # Actualizar solo si es first-visit o si estamos usando every-visit
            if (self.first_visit and is_first_occurrence) or (not self.first_visit):
                self.returns_count[state, action] += 1
                self.returns_sum[state, action] += g_value
                
                # Actualizar valor Q
                if self.returns_count[state, action] > 0:
                    self.q_table[state, action] = (
                        self.returns_sum[state, action] / 
                        self.returns_count[state, action]
                    )
                
                # Actualizar política si es control
                self.update_policy(state)

class MonteCarloModel(nn.Module):
    def __init__(self, cgm_shape: Tuple[int, ...], other_features_shape: Tuple[int, ...], **kwargs) -> None:
        super().__init__()
        
        # Agregar capas entrenables
        self.cgm_encoder = nn.Sequential(
            nn.Flatten(),
            nn.Linear(np.prod(cgm_shape), 64),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        
        self.other_encoder = nn.Sequential(
            nn.Flatten(),
            nn.Linear(np.prod(other_features_shape), 32),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        
        self.combined_layers = nn.Sequential(
            nn.Linear(64 + 32, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )
        
        # Para algoritmo Monte Carlo
        self.gamma = kwargs.get('gamma', 0.99)
        self.exploration_rate = kwargs.get('exploration_rate', 0.1)
        
        # Registrar tablas como parámetros
        # Haciendo esto aseguramos que haya parámetros para el optimizador
        n_states = kwargs.get('n_states', 1000)
        n_actions = kwargs.get('n_actions', 20)
        self.q_table = nn.Parameter(torch.zeros(n_states, n_actions), requires_grad=True)

    def forward(self, cgm_input: torch.Tensor, other_input: torch.Tensor) -> torch.Tensor:
        """
        Realiza una pasada forward del modelo, combinando la red neuronal y la tabla Q.
        
        Parámetros:
        -----------
        cgm_input : torch.Tensor
            Datos CGM de entrada
        other_input : torch.Tensor
            Otras características de entrada
            
        Retorna:
        --------
        torch.Tensor
            Predicción de dosis de insulina
        """
        # Codificar entradas
        cgm_features = self.cgm_encoder(cgm_input)
        other_features = self.other_encoder(other_input)
        
        # Combinar características
        combined = torch.cat([cgm_features, other_features], dim=1)
        
        # Procesar a través de las capas combinadas
        output = self.combined_layers(combined)
        
        return output

def create_monte_carlo_model_wrapper(cgm_shape: Tuple[int, ...], other_features_shape: Tuple[int, ...], **kwargs) -> RLModelWrapperPyTorch:
    """
    Crea un modelo Monte Carlo envuelto en RLModelWrapperPyTorch.
    
    Parámetros:
    -----------
    cgm_shape : Tuple[int, ...]
        Forma de los datos CGM
    other_features_shape : Tuple[int, ...]
        Forma de otras características
    **kwargs
        Argumentos adicionales para configuración
        
    Retorna:
    --------
    RLModelWrapperPyTorch
        Modelo Monte Carlo envuelto para compatibilidad con el sistema
    """
    # Función creadora del modelo interno - no debe tomar argumentos
    def model_creator_fn():
        return MonteCarloModel(cgm_shape, other_features_shape, **kwargs)
    
    # Crear wrapper con parámetros
    model_wrapper = RLModelWrapperPyTorch(
        model_cls=model_creator_fn,
        n_states=MONTE_CARLO_CONFIG['n_states'],
        n_actions=MONTE_CARLO_CONFIG['n_actions'],
    )
    
    return model_wrapper

def model_creator() -> Callable[[Tuple[int, ...], Tuple[int, ...]], RLModelWrapperPyTorch]:
    """
    Devuelve una función creadora de modelos Monte Carlo compatible con la infraestructura.
    
    Retorna:
    --------
    Callable[[Tuple[int, ...], Tuple[int, ...]], RLModelWrapperPyTorch]
        Función que crea un modelo Monte Carlo envuelto
    """
    return create_monte_carlo_model_wrapper