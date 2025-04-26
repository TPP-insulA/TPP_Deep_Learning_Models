import os
import sys
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import time
import pickle
from typing import Dict, List, Tuple, Optional, Union, Any, Callable
from types import SimpleNamespace

PROJECT_ROOT = os.path.abspath(os.getcwd())
sys.path.append(PROJECT_ROOT) 

from config.models_config import SARSA_CONFIG
from custom.rl_model_wrapper import RLModelWrapperPyTorch

# Constantes para rutas de figuras y etiquetas comunes
FIGURES_DIR = os.path.join(PROJECT_ROOT, "figures", "pytorch", "sarsa")
CONST_EPISODE_LABEL = "Episodio"
CONST_REWARD_LABEL = "Recompensa" 
CONST_STEPS_LABEL = "Pasos"
CONST_EPSILON_LABEL = "Epsilon"
CONST_ORIGINAL_LABEL = "Original"
CONST_SMOOTHED_LABEL = "Suavizado"
CONST_WEIGHT_FILE_SUFFIX = "_weights.pth"
CONST_AGENT_FILE_SUFFIX = "_agent.pkl"

# Crear directorio para figuras si no existe
os.makedirs(FIGURES_DIR, exist_ok=True)

class SARSA:
    """
    Implementación del algoritmo SARSA (State-Action-Reward-State-Action) con PyTorch.
    
    SARSA es un algoritmo de aprendizaje por refuerzo on-policy que actualiza
    los valores Q basándose en la política actual, incluyendo la exploración.
    """
    
    def __init__(
        self, 
        env: Any, 
        config: Optional[Dict] = None,
        seed: int = 42,
        cgm_shape: Optional[Tuple[int, ...]] = None,
        other_features_shape: Optional[Tuple[int, ...]] = None
    ) -> None:
        """
        Inicializa el agente SARSA.
        
        Parámetros:
        -----------
        env : Any
            Entorno de OpenAI Gym o compatible
        config : Optional[Dict], opcional
            Configuración personalizada (default: None)
        seed : int, opcional
            Semilla para reproducibilidad (default: 42)
        cgm_shape : Optional[Tuple[int, ...]], opcional
            Forma de los datos CGM (puede ser necesario para mapeo estado-observación) (default: None)
        other_features_shape : Optional[Tuple[int, ...]], opcional
            Forma de otras características (puede ser necesario para mapeo estado-observación) (default: None)
        """
        self.env = env
        self.config = config or SARSA_CONFIG
        self.cgm_shape = cgm_shape
        self.other_features_shape = other_features_shape
        
        # Inicializar parámetros básicos
        self._init_learning_params()
        self._validate_action_space()
        
        # Configurar generador de números aleatorios
        self.rng = np.random.Generator(np.random.PCG64(seed))
        
        # Configurar espacio de estados y tabla Q
        self.discrete_state_space = hasattr(env.observation_space, 'n')
        
        if self.discrete_state_space:
            self._setup_discrete_state_space()
        else:
            self._setup_continuous_state_space()
        
        # Métricas para seguimiento del entrenamiento
        self.episode_rewards = []
        self.episode_lengths = []
        self.epsilon_history = []
        
        # Mover tabla Q al dispositivo apropiado
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.q_table = self.q_table.to(self.device)
    
    def _init_learning_params(self) -> None:
        """
        Inicializa los parámetros de aprendizaje desde la configuración.
        """
        self.alpha = self.config['learning_rate']
        self.gamma = self.config['gamma']
        self.epsilon = self.config['epsilon_start']
        self.epsilon_min = self.config['epsilon_end']
        self.epsilon_decay = self.config['epsilon_decay']
        self.decay_type = self.config['epsilon_decay_type']
    
    def _validate_action_space(self) -> None:
        """
        Valida que el espacio de acción sea compatible con SARSA.
        
        Genera:
        -------
        ValueError
            Si el espacio de acción no es discreto
        """
        if not hasattr(self.env.action_space, 'n'):
            raise ValueError("SARSA requiere un espacio de acción discreto")
        self.action_space_size = self.env.action_space.n
    
    def _setup_discrete_state_space(self) -> None:
        """
        Configura SARSA para un espacio de estados discreto.
        """
        self.state_space_size = self.env.observation_space.n
        self.q_table = torch.zeros((self.state_space_size, self.action_space_size))
        
        if self.config['optimistic_initialization']:
            self.q_table += self.config['optimistic_initial_value']
    
    def _setup_continuous_state_space(self) -> None:
        """
        Configura SARSA para un espacio de estados continuo con discretización.
        """
        self.state_dim = self.env.observation_space.shape[0]
        self.bins_per_dim = self.config['bins']
        
        # Configurar límites de estado y bins
        self._setup_state_bounds()
        self._create_discretization_bins()
        
        # Crear y configurar tabla Q
        q_shape = tuple([self.bins_per_dim] * self.state_dim + [self.action_space_size])
        self.q_table = torch.zeros(q_shape)
        
        if self.config['optimistic_initialization']:
            self.q_table += self.config['optimistic_initial_value']
    
    def _setup_state_bounds(self) -> None:
        """
        Determina los límites para cada dimensión del espacio de estados.
        """
        if self.config['state_bounds'] is None:
            self.state_bounds = self._get_default_state_bounds()
        else:
            self.state_bounds = self.config['state_bounds']
    
    def _get_default_state_bounds(self) -> List[Tuple[float, float]]:
        """
        Calcula límites predeterminados para cada dimensión del estado.
        
        Retorna:
        --------
        List[Tuple[float, float]]
            Lista de tuplas (min, max) para cada dimensión
        """
        bounds = []
        for i in range(self.state_dim):
            low = self.env.observation_space.low[i]
            high = self.env.observation_space.high[i]
            
            # Manejar valores infinitos
            if low == float("-inf") or low < -1e6:
                low = -10.0
            if high == float("inf") or high > 1e6:
                high = 10.0
                
            bounds.append((low, high))
        return bounds
    
    def _create_discretization_bins(self) -> None:
        """
        Crea bins para discretización del espacio de estados continuo.
        """
        self.discrete_states = []
        for low, high in self.state_bounds:
            self.discrete_states.append(np.linspace(low, high, self.bins_per_dim + 1)[1:-1])
    
    def discretize_state(self, state: np.ndarray) -> Union[int, Tuple[int, ...]]:
        """
        Discretiza un estado continuo.
        
        Parámetros:
        -----------
        state : np.ndarray
            Estado continuo del entorno
            
        Retorna:
        --------
        Union[int, Tuple[int, ...]]
            Índice discretizado o tupla de índices discretizados
        """
        if self.discrete_state_space:
            # Si el espacio de estados ya es discreto, convertir a entero
            if isinstance(state, np.ndarray):
                # Si es un array, tomar el primer elemento como índice
                if len(state.shape) > 0:
                    return int(state[0])
                return int(state)
            elif isinstance(state, torch.Tensor):
                return int(state.item())
            return int(state)
        
        # Para espacio continuo, discretizar cada dimensión
        discrete_state = []
        for i, val in enumerate(state):
            # Limitar val al rango definido
            low, high = self.state_bounds[i]
            val = max(low, min(val, high))
            
            # Encontrar el bin correspondiente
            bins = self.discrete_states[i]
            digitized = np.digitize(val, bins)
            discrete_state.append(digitized)
        
        return tuple(discrete_state)
    
    def get_action(self, state: np.ndarray, explore: bool = True) -> int:
        """
        Selecciona una acción usando la política epsilon-greedy.
        
        Parámetros:
        -----------
        state : np.ndarray
            Estado actual
        explore : bool, opcional
            Si debe explorar (True) o ser greedy (False) (default: True)
            
        Retorna:
        --------
        int
            Acción seleccionada
        """
        discrete_state = self.discretize_state(state)
        
        # Exploración con probabilidad epsilon
        if explore and self.rng.random() < self.epsilon:
            return self.rng.integers(0, self.action_space_size)
        else:
            # Explotación: elegir acción con mayor valor Q
            with torch.no_grad():
                return torch.argmax(self.q_table[discrete_state]).item()
    
    def update_q_value(
        self, 
        state: np.ndarray, 
        action: int, 
        reward: float, 
        next_state: np.ndarray, 
        next_action: int
    ) -> None:
        """
        Actualiza un valor Q usando la regla de actualización SARSA.
        
        Parámetros:
        -----------
        state : np.ndarray
            Estado actual
        action : int
            Acción tomada
        reward : float
            Recompensa recibida
        next_state : np.ndarray
            Siguiente estado
        next_action : int
            Siguiente acción (según política actual)
        """
        discrete_state = self.discretize_state(state)
        discrete_next_state = self.discretize_state(next_state)
        
        # Verificar el tipo de índice y acceder a la tabla Q de manera adecuada
        if isinstance(discrete_state, tuple):
            # Para espacio de estados multidimensional, usamos indexación con tuplas
            current_q = self.q_table[discrete_state][action].item()
            next_q = self.q_table[discrete_next_state][next_action].item()
            
            # Actualizar Q usando la regla SARSA
            new_q = current_q + self.alpha * (reward + self.gamma * next_q - current_q)
            self.q_table[discrete_state][action] = new_q
        else:
            # Para espacio de estados unidimensional, usamos indexación simple
            current_q = self.q_table[discrete_state, action].item()
            next_q = self.q_table[discrete_next_state, next_action].item()
            
            # Actualizar Q usando la regla SARSA
            new_q = current_q + self.alpha * (reward + self.gamma * next_q - current_q)
            self.q_table[discrete_state, action] = new_q
    
    def decay_epsilon(self, episode: Optional[int] = None) -> None:
        """
        Actualiza epsilon según el esquema de decaimiento configurado.
        
        Parámetros:
        -----------
        episode : Optional[int], opcional
            Número de episodio actual (para decaimiento lineal) (default: None)
        """
        if self.decay_type == 'exponential':
            self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
        elif self.decay_type == 'linear':
            # Requiere conocer el número total de episodios para el decaimiento lineal
            if episode is not None and self.config['episodes'] > 0:
                self.epsilon = max(
                    self.epsilon_min,
                    self.epsilon_min + (self.config['epsilon_start'] - self.epsilon_min) * 
                    (1 - episode / self.config['episodes'])
                )
    
    def train(
        self, 
        episodes: Optional[int] = None, 
        max_steps: Optional[int] = None, 
        render: bool = False
    ) -> Dict[str, List[float]]:
        """
        Entrena el agente SARSA.
        
        Parámetros:
        -----------
        episodes : Optional[int], opcional
            Número de episodios de entrenamiento (default: None)
        max_steps : Optional[int], opcional
            Límite de pasos por episodio (default: None)
        render : bool, opcional
            Si renderizar el entorno durante el entrenamiento (default: False)
            
        Retorna:
        --------
        Dict[str, List[float]]
            Diccionario con métricas de entrenamiento
        """
        episodes = episodes or self.config['episodes']
        max_steps = max_steps or self.config['max_steps']
        
        _ = time.time()
        
        # Reiniciar listas para métricas
        self.episode_rewards = []
        self.episode_lengths = []
        self.epsilon_history = []
        
        for episode in range(episodes):
            # Inicializar el episodio
            state, _ = self.env.reset()
            state = np.array(state, dtype=np.float32)
            
            # Seleccionar acción inicial
            action = self.get_action(state)
            
            episode_reward = 0
            steps = 0
            
            # Registrar tiempo de inicio del episodio
            episode_start = time.time()
            
            for _ in range(max_steps):
                # Tomar la acción en el entorno
                next_state, reward, done, truncated, _ = self.env.step(action)
                done = done or truncated
                next_state = np.array(next_state, dtype=np.float32)
                
                # Renderizar si es necesario
                if render:
                    self.env.render()
                
                # Seleccionar la siguiente acción (política actual)
                next_action = self.get_action(next_state)
                
                # Actualizar la tabla Q usando la regla SARSA
                self.update_q_value(state, action, reward, next_state, next_action)
                
                # Actualizar para el próximo paso
                state = next_state
                action = next_action
                
                # Actualizar métricas
                episode_reward += reward
                steps += 1
                
                if done:
                    break
            
            # Actualizar métricas del episodio
            self.episode_rewards.append(episode_reward)
            self.episode_lengths.append(steps)
            self.epsilon_history.append(self.epsilon)
            
            # Decaer epsilon
            self.decay_epsilon(episode)
            
            # Calcular duración del episodio
            episode_duration = time.time() - episode_start
            
            # Mostrar progreso 
            if episode % self.config['log_interval'] == 0 or episode == episodes - 1:
                print(f"Episodio {episode+1}/{episodes} - Recompensa: {episode_reward:.2f}, Pasos: {steps:.2f}, Epsilon: {self.epsilon:.4f}, Tiempo: {episode_duration:.2f}s")
        
        print("Entrenamiento completado!")
        
        # Retornar historial de entrenamiento
        return {
            'rewards': self.episode_rewards,
            'lengths': self.episode_lengths,
            'epsilons': self.epsilon_history
        }
    
    def evaluate(
        self, 
        episodes: int = 10, 
        render: bool = True, 
        verbose: bool = True
    ) -> float:
        """
        Evalúa la política aprendida.
        
        Parámetros:
        -----------
        episodes : int, opcional
            Número de episodios para evaluación (default: 10)
        render : bool, opcional
            Si renderizar el entorno durante evaluación (default: True)
        verbose : bool, opcional
            Si mostrar resultados detallados (default: True)
            
        Retorna:
        --------
        float
            Recompensa promedio obtenida
        """
        rewards = []
        steps = []
        
        for episode in range(episodes):
            state, _ = self.env.reset()
            state = np.array(state, dtype=np.float32)
            episode_reward = 0
            episode_steps = 0
            
            done = False
            while not done:
                action = self.get_action(state, explore=False)
                
                # Ejecutar acción en el entorno
                next_state, reward, done, truncated, _ = self.env.step(action)
                done = done or truncated
                next_state = np.array(next_state, dtype=np.float32)
                
                if render:
                    self.env.render()
                
                # Actualizar estado y métricas
                state = next_state
                episode_reward += reward
                episode_steps += 1
            
            rewards.append(episode_reward)
            steps.append(episode_steps)
            
            if verbose:
                print(f"Episodio de evaluación {episode+1}/{episodes} - Recompensa: {episode_reward:.2f}, Pasos: {episode_steps}")
        
        avg_reward = np.mean(rewards)
        avg_steps = np.mean(steps)
        
        print(f"Evaluación completada - Recompensa Media: {avg_reward:.2f}, Pasos Medios: {avg_steps:.2f}")
        
        return avg_reward
    
    def save(self, filepath: str) -> None:
        """
        Guarda la tabla Q y otra información del modelo.
        
        Parámetros:
        -----------
        filepath : str
            Ruta donde guardar el modelo
        """
        data = {
            'q_table': self.q_table.cpu().numpy(),
            'discrete_states': self.discrete_states if not self.discrete_state_space else None,
            'state_bounds': self.state_bounds if not self.discrete_state_space else None,
            'discrete_state_space': self.discrete_state_space,
            'config': self.config,
            'episode_rewards': self.episode_rewards,
            'episode_lengths': self.episode_lengths,
            'epsilon_history': self.epsilon_history,
            'epsilon': self.epsilon
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(data, f)
        
        print(f"Modelo guardado en {filepath}")
    
    def load(self, filepath: str) -> None:
        """
        Carga la tabla Q y otra información del modelo.
        
        Parámetros:
        -----------
        filepath : str
            Ruta desde donde cargar el modelo
        """
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
        
        self.q_table = torch.tensor(data['q_table']).to(self.device)
        self.discrete_state_space = data['discrete_state_space']
        
        if not self.discrete_state_space:
            self.discrete_states = data['discrete_states']
            self.state_bounds = data['state_bounds']
        
        self.config = data['config']
        self.alpha = self.config['learning_rate']
        self.gamma = self.config['gamma']
        
        # Cargar métricas de entrenamiento si están disponibles
        self.episode_rewards = data.get('episode_rewards', [])
        self.episode_lengths = data.get('episode_lengths', [])
        self.epsilon_history = data.get('epsilon_history', [])
        self.epsilon = data.get('epsilon', self.epsilon)
        
        print(f"Modelo cargado desde {filepath}")
    
    def _map_observation_to_state(self, cgm_obs: np.ndarray, other_obs: np.ndarray) -> int:
        """
        Mapea las observaciones a un estado discreto para la tabla Q.
        
        Parámetros:
        -----------
        cgm_obs : np.ndarray
            Datos CGM de una muestra
        other_obs : np.ndarray
            Otras características de una muestra
            
        Retorna:
        --------
        int
            Estado discreto
        """
        # Extraer características relevantes de CGM
        cgm_flat = cgm_obs.flatten()
        
        cgm_mean = np.mean(cgm_flat) if cgm_flat.size > 0 else 0
        cgm_last = cgm_flat[-1] if cgm_flat.size > 0 else 0
        
        # Calcular pendiente si hay suficientes puntos
        if cgm_flat.size >= 5:
            cgm_slope = (cgm_flat[-1] - cgm_flat[-5]) / 5
        else:
            cgm_slope = 0.0
        
        cgm_std = np.std(cgm_flat) if cgm_flat.size > 0 else 0
        
        # Discretizar características CGM
        cgm_bins = 10  # Número de bins por dimensión
        cgm_mean_bin = min(int(cgm_mean / 300 * cgm_bins), cgm_bins - 1)
        cgm_last_bin = min(int(cgm_last / 300 * cgm_bins), cgm_bins - 1)
        cgm_slope_bin = min(int((cgm_slope + 100) / 200 * cgm_bins), cgm_bins - 1)
        cgm_std_bin = min(int(cgm_std / 50 * cgm_bins), cgm_bins - 1)
        
        # Extraer características relevantes de otras variables
        other_bins = 5  # Menos bins para otras características
        
        # Por defecto, usar los primeros 3 valores o ceros si no hay suficientes
        carb_bin = min(int(other_obs[0] / 100 * other_bins), other_bins - 1) if other_obs.size > 0 else 0
        bg_bin = min(int(other_obs[1] / 300 * other_bins), other_bins - 1) if other_obs.size > 1 else 0
        iob_bin = min(int(other_obs[2] / 10 * other_bins), other_bins - 1) if other_obs.size > 2 else 0
        
        # Combinar bins en un único índice de estado
        state_index = cgm_mean_bin
        state_index = state_index * cgm_bins + cgm_last_bin
        state_index = state_index * cgm_bins + cgm_slope_bin
        state_index = state_index * cgm_bins + cgm_std_bin
        state_index = state_index * other_bins + carb_bin
        state_index = state_index * other_bins + bg_bin
        state_index = state_index * other_bins + iob_bin
        
        return state_index
    
    def _convert_action_to_dose(self, action: int) -> float:
        """
        Convierte una acción discreta a una dosis continua.
        
        Parámetros:
        -----------
        action : int
            Índice de acción discreto
            
        Retorna:
        --------
        float
            Dosis de insulina (valor continuo)
        """
        # Mapear de acción discreta [0, n_actions-1] a dosis [0, 15]
        max_dose = 15.0  # Dosis máxima (unidades)
        return (action / (self.action_space_size - 1)) * max_dose


class SARSAModel(nn.Module):
    """
    Wrapper para el algoritmo SARSA que implementa la interfaz de Module de PyTorch.
    """
    
    def __init__(
        self, 
        sarsa_agent: SARSA,
        cgm_shape: Tuple[int, ...],
        other_features_shape: Tuple[int, ...],
        discretizer: Optional[Any] = None
    ) -> None:
        """
        Inicializa el modelo wrapper para SARSA.
        
        Parámetros:
        -----------
        sarsa_agent : SARSA
            Agente SARSA a utilizar
        cgm_shape : Tuple[int, ...]
            Forma de entrada para datos CGM
        other_features_shape : Tuple[int, ...]
            Forma de entrada para otras características
        discretizer : Optional[Any], opcional
            Función de discretización personalizada (default: None)
        """
        super(SARSAModel, self).__init__()
        self.sarsa_agent = sarsa_agent
        self.cgm_shape = cgm_shape
        self.other_features_shape = other_features_shape
        self.discretizer = discretizer
        
        # Capas para procesar entrada de CGM
        self.cgm_encoder = nn.Sequential(
            nn.Flatten(),
            nn.Linear(np.prod(cgm_shape), 64),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        
        # Capas para procesar otras características
        self.other_encoder = nn.Sequential(
            nn.Flatten(),
            nn.Linear(np.prod(other_features_shape), 32),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        
        # Capa de concatenación (implementada en forward)
        
        # Capa para codificar estado discreto
        state_space_size = sarsa_agent.state_space_size if hasattr(sarsa_agent, 'state_space_size') else 1000
        self.state_encoder = nn.Sequential(
            nn.Linear(64 + 32, state_space_size),
            nn.Softmax(dim=1)
        )
        
        # Capa para convertir acciones discretas a dosis continuas
        self.action_decoder = nn.Linear(1, 1)
        
        # Inicializar los pesos
        self._init_weights()
    
    def _init_weights(self) -> None:
        """
        Inicializa los pesos del decoder para mapear acciones a dosis.
        """
        # Inicialización con pesos que mapean [0, n_actions-1] a [0, 15]
        n_actions = self.sarsa_agent.action_space_size
        max_dose = 15.0
        
        # Mapeo lineal: dose = slope * action + intercept
        slope = max_dose / (n_actions - 1)
        intercept = 0.0
        
        # Establecer pesos y bias
        self.action_decoder.weight.data.fill_(slope)
        self.action_decoder.bias.data.fill_(intercept)
    
    def forward(self, cgm_data: torch.Tensor, other_features: torch.Tensor) -> torch.Tensor:
        """
        Realiza la predicción de dosis de insulina a partir de los datos de entrada.
        
        Parámetros:
        -----------
        cgm_data : torch.Tensor
            Datos CGM de entrada
        other_features : torch.Tensor
            Otras características de entrada
            
        Retorna:
        --------
        torch.Tensor
            Predicciones de dosis de insulina
        """
        batch_size = cgm_data.size(0)
        
        # Codificar estados
        state_encodings = self._encode_states(cgm_data, other_features)
        
        # Lista para guardar acciones
        actions = []
        
        for i in range(batch_size):
            # Extraer codificación individual
            encoding = state_encodings[i]
            
            # Discretizar estado
            discrete_state = self._discretize_state(encoding)
            
            # Obtener mejor acción
            action = torch.argmax(self.sarsa_agent.q_table[discrete_state]).item()
            actions.append(action)
        
        # Convertir a tensor
        actions_tensor = torch.tensor(actions, dtype=torch.float32, device=cgm_data.device).view(-1, 1)
        
        # Decodificar acciones discretas a dosis continuas
        dose_predictions = self.action_decoder(actions_tensor)
        
        return dose_predictions
    
    def _encode_states(self, cgm_data: torch.Tensor, other_features: torch.Tensor) -> torch.Tensor:
        """
        Codifica los datos de entrada en una representación de estado.
        
        Parámetros:
        -----------
        cgm_data : torch.Tensor
            Datos CGM
        other_features : torch.Tensor
            Otras características
            
        Retorna:
        --------
        torch.Tensor
            Estados codificados
        """
        # Procesar datos CGM
        if len(cgm_data.shape) > 2:
            # Si es serie temporal, aplanar
            batch_size = cgm_data.size(0)
            cgm_flattened = cgm_data.reshape(batch_size, -1)
            cgm_encoded = self.cgm_encoder(cgm_flattened)
        else:
            # Si ya es plano
            cgm_encoded = self.cgm_encoder(cgm_data)
        
        # Procesar otras características
        other_encoded = self.other_encoder(other_features)
        
        # Concatenar las características codificadas
        combined = torch.cat([cgm_encoded, other_encoded], dim=1)
        
        # Codificar a un espacio de estados discreto para SARSA
        state_encoded = self.state_encoder(combined)
        
        return state_encoded
    
    def _discretize_state(self, state_encoding: torch.Tensor) -> int:
        """
        Discretiza la codificación de estado para consultar la tabla Q.
        
        Parámetros:
        -----------
        state_encoding : torch.Tensor
            Codificación de estado
            
        Retorna:
        --------
        int
            Índice discretizado para la tabla Q
        """
        # Obtener índice del valor máximo como estado discreto
        discrete_state = torch.argmax(state_encoding).item()
        
        # Convertir a entero para indexación
        return int(discrete_state)
    
    def _calibrate_action_decoder(self, y: np.ndarray) -> None:
        """
        Calibra la capa de decodificación de acciones para mapear acciones discretas a dosis continuas.
        
        Parámetros:
        -----------
        y : np.ndarray
            Valores objetivo (dosis de insulina)
        """
        # Obtener rango de dosis
        min_dose = np.min(y)
        max_dose = np.max(y)
        
        # Determinar espacio de acciones del agente SARSA
        n_actions = self.sarsa_agent.action_space_size
        
        # Calcular pendiente y sesgo para mapeo lineal
        slope = (max_dose - min_dose) / (n_actions - 1)
        intercept = min_dose
        
        # Actualizar pesos
        with torch.no_grad():
            self.action_decoder.weight.data.fill_(slope)
            self.action_decoder.bias.data.fill_(intercept)
    
    def _create_training_environment(
        self, 
        cgm_data: torch.Tensor, 
        other_features: torch.Tensor, 
        targets: np.ndarray
    ) -> Any:
        """
        Crea un entorno de entrenamiento compatible con el agente SARSA.
        
        Parámetros:
        -----------
        cgm_data : torch.Tensor
            Datos CGM
        other_features : torch.Tensor
            Otras características
        targets : np.ndarray
            Valores objetivo (dosis de insulina)
            
        Retorna:
        --------
        Any
            Entorno compatible con Gym para el agente SARSA
        """
        # Convertir tensores a numpy para procesamiento
        cgm_np = cgm_data.detach().cpu().numpy() if hasattr(cgm_data, 'detach') else cgm_data
        other_np = other_features.detach().cpu().numpy() if hasattr(other_features, 'detach') else other_features
        targets_np = targets.flatten()
        
        # Definir clase de entorno
        class InsulinDosingEnv:
            """Entorno personalizado para problema de dosificación de insulina."""
            
            def __init__(self, cgm, other, targets, model):
                self.cgm = cgm
                self.other = other
                self.targets = targets
                self.model = model
                self.current_idx = 0
                self.max_idx = len(targets) - 1
                self.rng = np.random.Generator(np.random.PCG64(42))
                
                # Definir espacio de acción discreto
                self.action_space = SimpleNamespace(
                    n=self.model.sarsa_agent.action_space_size,
                    sample=lambda: self.rng.integers(0, self.model.sarsa_agent.action_space_size)
                )
                
                # Definir espacio de observación
                flat_dim = np.prod(cgm.shape[1:]) + np.prod(other.shape[1:])
                self.observation_space = SimpleNamespace(
                    shape=(flat_dim,),
                    n=self.model.sarsa_agent.state_space_size if hasattr(self.model.sarsa_agent, 'state_space_size') else None
                )
            
            def reset(self):
                """Reinicia el entorno, elige un ejemplo aleatorio."""
                self.current_idx = self.rng.integers(0, self.max_idx)
                state = self._get_state()
                return state, {}
            
            def step(self, action):
                """Ejecuta un paso con la acción dada."""
                # La acción es un índice discreto, obtener la dosis correspondiente
                action_tensor = torch.tensor([[float(action)]], device=self.model.action_decoder.weight.device)
                dose = float(self.model.action_decoder(action_tensor).detach().cpu().numpy()[0, 0])
                
                # Calcular error y recompensa (negativo del error absoluto)
                target = self.targets[self.current_idx]
                error = abs(dose - target)
                reward = -error  # Recompensa es negativo del error
                
                # Avanzar al siguiente ejemplo (o volver al inicio si es el último)
                self.current_idx = (self.current_idx + 1) % (self.max_idx + 1)
                
                # Obtener siguiente estado
                next_state = self._get_state()
                
                # Para este problema, considerar que un episodio termina después de cada paso
                done = True
                truncated = False
                info = {}
                
                return next_state, reward, done, truncated, info
            
            def _get_state(self):
                """Obtiene el estado actual codificado."""
                # Extraer datos actuales
                current_cgm = self.cgm[self.current_idx:self.current_idx+1]
                current_other = self.other[self.current_idx:self.current_idx+1]
                
                # Convertir a tensores PyTorch
                current_cgm_tensor = torch.tensor(current_cgm, dtype=torch.float32, device=self.model.action_decoder.weight.device)
                current_other_tensor = torch.tensor(current_other, dtype=torch.float32, device=self.model.action_decoder.weight.device)
                
                # Codificar estado
                state_encoding = self.model._encode_states(current_cgm_tensor, current_other_tensor)
                
                return state_encoding.detach().cpu().numpy()[0]
        
        return InsulinDosingEnv(cgm_np, other_np, targets_np, self)
    
    def fit(
        self, 
        x: List[torch.Tensor], 
        y: np.ndarray, 
        validation_data: Optional[Tuple] = None, 
        epochs: int = 1,
        batch_size: int = 32,
        callbacks: list = None,
        verbose: int = 0
    ) -> Dict:
        """
        Entrena el modelo utilizando el agente SARSA subyacente.
        
        Parámetros:
        -----------
        x : List[torch.Tensor]
            Lista de tensores de entrada [cgm_data, other_features]
        y : np.ndarray
            Valores objetivo (dosis de insulina)
        validation_data : Optional[Tuple], opcional
            Datos de validación (default: None)
        epochs : int, opcional
            Número de épocas (default: 1)
        batch_size : int, opcional
            Tamaño de lote (default: 32)
        callbacks : list, opcional
            Lista de callbacks (default: None)
        verbose : int, opcional
            Nivel de verbosidad (default: 0)
            
        Retorna:
        --------
        Dict
            Historia de entrenamiento
        """
        # Calibrar el decodificador de acciones para mapear acciones discretas a dosis continuas
        self._calibrate_action_decoder(y)
        
        # Crear entorno de entrenamiento
        train_env = self._create_training_environment(x[0], x[1], y)
        
        # Asignar entorno al agente SARSA
        self.sarsa_agent.env = train_env
        
        # Multiplicar épocas por batch_size para simular épocas completas
        effective_episodes = epochs * (len(y) // batch_size) if batch_size < len(y) else epochs
        
        # Entrenar el agente SARSA
        history = self.sarsa_agent.train(
            episodes=effective_episodes,
            max_steps=1,  # Un paso por episodio es suficiente para este problema
            render=False
        )
        
        # Preparar datos para validación si están disponibles
        if validation_data is not None:
            val_x, val_y = validation_data
            val_env = self._create_training_environment(val_x[0], val_x[1], val_y)
            
            # Guardar el entorno actual
            original_env = self.sarsa_agent.env
            
            # Configurar entorno de validación
            self.sarsa_agent.env = val_env
            
            # Evaluar (sin pasar el entorno como argumento)
            val_reward = self.sarsa_agent.evaluate(
                episodes=len(val_y) // batch_size if batch_size < len(val_y) else 1,
                render=False,
                verbose=verbose > 0
            )
            
            # Restaurar el entorno original
            self.sarsa_agent.env = original_env
            
            if verbose > 0:
                print(f"Recompensa de validación: {val_reward:.4f}")
            
            # Añadir métricas de validación al historial
            history['val_reward'] = val_reward
        
        return history
    
    def get_config(self) -> Dict:
        """
        Obtiene la configuración del modelo para serialización.
        
        Retorna:
        --------
        Dict
            Configuración del modelo
        """
        config = {'cgm_shape': self.cgm_shape, 'other_features_shape': self.other_features_shape}
        return config
    
    def save_weights(self, filepath: str, **kwargs) -> None:
        """
        Guarda los pesos del modelo y el agente SARSA.
        
        Parámetros:
        -----------
        filepath : str
            Ruta donde guardar los pesos
        **kwargs : dict
            Argumentos adicionales para el método de guardado original
        """
        # Guardar pesos de las capas del wrapper
        torch.save(self.state_dict(), filepath + CONST_WEIGHT_FILE_SUFFIX)
        
        # Guardar el agente SARSA
        self.sarsa_agent.save(filepath + CONST_AGENT_FILE_SUFFIX)
    
    def load_weights(self, filepath: str, **kwargs) -> None:
        """
        Carga los pesos del modelo y el agente SARSA.
        
        Parámetros:
        -----------
        filepath : str
            Ruta desde donde cargar los pesos
        **kwargs : dict
            Argumentos adicionales para el método de carga original
        """
        # Cargar pesos de las capas del wrapper
        self.load_state_dict(torch.load(filepath + CONST_WEIGHT_FILE_SUFFIX))
        
        # Cargar el agente SARSA
        self.sarsa_agent.load(filepath + CONST_AGENT_FILE_SUFFIX)


def create_sarsa_agent(cgm_shape: Tuple[int, ...], other_features_shape: Tuple[int, ...], **kwargs) -> SARSA:
    """
    Crea un agente SARSA para el problema de dosificación de insulina.
    
    Parámetros:
    -----------
    cgm_shape : Tuple[int, ...]
        Forma de los datos CGM
    other_features_shape : Tuple[int, ...]
        Forma de otras características
    **kwargs
        Argumentos adicionales para configurar el agente
        
    Retorna:
    --------
    SARSA
        Agente SARSA inicializado
    """
    # Crear entorno ficticio para inicialización
    class TempEnv:
        def __init__(self):
            # Configurar espacios de acción y observación
            # Espacio de acción: 20 niveles discretos (0-15 unidades)
            self.action_space = SimpleNamespace(n=20)
            
            # Espacio de observación: basado en la discretización de características
            cgm_bins = 10
            other_bins = 5
            n_states = cgm_bins**4 * other_bins**3  # 4 características CGM, 3 otras
            self.observation_space = SimpleNamespace(n=n_states)
            
        def reset(self):
            return np.zeros(10), {}
            
        def step(self, _):
            return np.zeros(10), 0, True, False, {}
    
    # Crear entorno temporal
    temp_env = TempEnv()
    
    # Personalizar configuración SARSA
    sarsa_config = SARSA_CONFIG.copy()
    sarsa_config.update({
        'learning_rate': kwargs.get('learning_rate', 0.1),
        'epsilon_start': kwargs.get('epsilon_start', 0.3),
        'epsilon_end': kwargs.get('epsilon_end', 0.01),
        'epsilon_decay': kwargs.get('epsilon_decay', 0.99),
        'gamma': kwargs.get('gamma', 0.95),
        'episodes': kwargs.get('episodes', 1000),
        'max_steps': kwargs.get('max_steps', 100),
        'log_interval': kwargs.get('log_interval', 100)
    })
    
    # Crear agente SARSA
    sarsa_agent = SARSA(
        env=temp_env, 
        config=sarsa_config, 
        seed=kwargs.get('seed', 42),
        cgm_shape=cgm_shape,
        other_features_shape=other_features_shape
    )
    
    return sarsa_agent


def create_sarsa_model(cgm_shape: Tuple[int, ...], other_features_shape: Tuple[int, ...], **kwargs) -> nn.Module:
    """
    Crea un modelo basado en SARSA para predicción de dosis de insulina.
    
    Parámetros:
    -----------
    cgm_shape : Tuple[int, ...]
        Forma de los datos CGM (batch_size, time_steps, features)
    other_features_shape : Tuple[int, ...]
        Forma de otras características (batch_size, n_features)
    **kwargs
        Argumentos adicionales para configuración
        
    Retorna:
    --------
    nn.Module
        Modelo SARSA que implementa la interfaz de PyTorch
    """
    # Crear agente SARSA
    sarsa_agent = create_sarsa_agent(cgm_shape, other_features_shape, **kwargs)
    
    # Crear y devolver el modelo wrapper
    return SARSAModel(
        sarsa_agent=sarsa_agent,
        cgm_shape=cgm_shape,
        other_features_shape=other_features_shape
    )


def model_creator() -> Callable[[Tuple[int, ...], Tuple[int, ...]], RLModelWrapperPyTorch]:
    """
    Devuelve una función para crear un modelo SARSA compatible con la API del sistema.
    
    Retorna:
    --------
    Callable[[Tuple[int, ...], Tuple[int, ...]], RLModelWrapperPyTorch]
        Función para crear el modelo con las formas de entrada especificadas
    """
    def creator_fn(cgm_shape: Tuple[int, ...], other_features_shape: Tuple[int, ...], **kwargs) -> RLModelWrapperPyTorch:
        model = create_sarsa_model(cgm_shape, other_features_shape, **kwargs)
        return RLModelWrapperPyTorch(model_cls=lambda: model)
    
    return creator_fn