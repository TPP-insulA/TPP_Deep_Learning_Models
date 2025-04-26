import os
import sys
import numpy as np
import torch
import matplotlib.pyplot as plt
import time
from typing import Dict, List, Tuple, Optional, Union, Any, Callable
from torch import nn
from types import SimpleNamespace

from custom.rl_model_wrapper import RLModelWrapperPyTorch

PROJECT_ROOT = os.path.abspath(os.getcwd())
sys.path.append(PROJECT_ROOT) 

from config.models_config import QLEARNING_CONFIG

# Constantes para archivos
QTABLE_SUFFIX = "_qtable.npy"
WRAPPER_WEIGHTS_SUFFIX = "_wrapper_weights.pth"

# Constantes para mensajes
CONST_CREANDO_ENTORNO = "Creando entorno de entrenamiento..."
CONST_ENTRENANDO_AGENTE = "Entrenando agente Q-Learning..."
CONST_EVALUANDO = "Evaluando modelo..."
CONST_TRANSFORMANDO_ESTADO = "Transformando estado discreto..."
CONST_ERROR_VALOR = "Error: Valor fuera de rango detectado:"
CONST_FIGURES_DIR = os.path.join(PROJECT_ROOT, "figures", "pytorch", "q_learning")
CONST_NO_ENV_AVAILABLE = "No hay entorno disponible, se usará simulación para entrenamiento"
CONST_MODEL_PATH = "Ruta para guardar modelo:"

class QLearning:
    """
    Implementación de Q-Learning tabular para espacios de estados y acciones discretos.
    
    Este algoritmo aprende una función de valor-acción (Q) a través de experiencias
    recolectadas mediante interacción con el entorno. Utiliza una tabla para almacenar
    los valores Q para cada par estado-acción.
    """
    def __init__(
        self,
        n_states: int,
        n_actions: int,
        learning_rate: float = QLEARNING_CONFIG['learning_rate'],
        gamma: float = QLEARNING_CONFIG['gamma'],
        epsilon_start: float = QLEARNING_CONFIG['epsilon_start'],
        epsilon_end: float = QLEARNING_CONFIG['epsilon_end'],
        epsilon_decay: float = QLEARNING_CONFIG['epsilon_decay'],
        use_decay_schedule: str = QLEARNING_CONFIG['use_decay_schedule'],
        decay_steps: int = QLEARNING_CONFIG['decay_steps'],
        seed: int = 42
    ) -> None:
        """
        Inicializa el agente Q-Learning.
        
        Parámetros:
        -----------
        n_states : int
            Número de estados en el espacio de estados discreto
        n_actions : int
            Número de acciones en el espacio de acciones discreto
        learning_rate : float, opcional
            Tasa de aprendizaje (alpha) (default: QLEARNING_CONFIG['learning_rate'])
        gamma : float, opcional
            Factor de descuento (default: QLEARNING_CONFIG['gamma'])
        epsilon_start : float, opcional
            Valor inicial de epsilon para política epsilon-greedy (default: QLEARNING_CONFIG['epsilon_start'])
        epsilon_end : float, opcional
            Valor final de epsilon para política epsilon-greedy (default: QLEARNING_CONFIG['epsilon_end'])
        epsilon_decay : float, opcional
            Factor de decaimiento para epsilon (default: QLEARNING_CONFIG['epsilon_decay'])
        use_decay_schedule : str, opcional
            Tipo de decaimiento ('linear', 'exponential', o None) (default: QLEARNING_CONFIG['use_decay_schedule'])
        decay_steps : int, opcional
            Número de pasos para decaer epsilon (si se usa decay schedule) (default: QLEARNING_CONFIG['decay_steps'])
        seed : int, opcional
            Semilla para reproducibilidad (default: 42)
        """
        self.n_states = n_states
        self.n_actions = n_actions
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.use_decay_schedule = use_decay_schedule
        self.decay_steps = decay_steps
        
        # Configurar generador de números aleatorios con semilla fija
        self.rng = np.random.Generator(np.random.PCG64(seed))
        
        # Inicializar la tabla Q con valores optimistas o ceros
        if QLEARNING_CONFIG['optimistic_init']:
            self.q_table = torch.ones((n_states, n_actions)) * QLEARNING_CONFIG['optimistic_value']
        else:
            self.q_table = torch.zeros((n_states, n_actions))
        
        # Mover la tabla Q al device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.q_table = self.q_table.to(self.device)
        
        # Contador total de pasos
        self.total_steps = 0
        
        # Métricas
        self.rewards_history = []
    
    def get_action(self, state: int) -> int:
        """
        Selecciona una acción usando la política epsilon-greedy.
        
        Parámetros:
        -----------
        state : int
            Estado actual
            
        Retorna:
        --------
        int
            Acción seleccionada
        """
        if self.rng.random() < self.epsilon:
            # Explorar: acción aleatoria
            return self.rng.integers(0, self.n_actions)
        else:
            # Explotar: mejor acción según la tabla Q
            with torch.no_grad():
                return torch.argmax(self.q_table[state]).item()
    
    def update(self, state: int, action: int, reward: float, next_state: int, done: bool) -> float:
        """
        Actualiza la tabla Q usando la regla de Q-Learning.
        
        Parámetros:
        -----------
        state : int
            Estado actual
        action : int
            Acción tomada
        reward : float
            Recompensa recibida
        next_state : int
            Estado siguiente
        done : bool
            Si el episodio ha terminado
            
        Retorna:
        --------
        float
            TD-error calculado
        """
        # Calcular valor Q objetivo con Q-Learning (off-policy TD control)
        with torch.no_grad():
            if done:
                # Si es estado terminal, no hay recompensa futura
                target_q = reward
            else:
                # Q-Learning: seleccionar máxima acción en estado siguiente (greedy)
                target_q = reward + self.gamma * torch.max(self.q_table[next_state])
            
            # Valor Q actual
            current_q = self.q_table[state, action].item()
            
            # Calcular TD error
            td_error = target_q - current_q
            
            # Actualizar valor Q
            self.q_table[state, action] += self.learning_rate * td_error
        
        return float(td_error)
    
    def update_epsilon(self, step: Optional[int] = None) -> None:
        """
        Actualiza el valor de epsilon según la política de decaimiento.
        
        Parámetros:
        -----------
        step : Optional[int], opcional
            Paso actual para decaimiento programado (default: None)
        """
        if self.use_decay_schedule == 'linear':
            # Decaimiento lineal
            if step is not None:
                fraction = min(1.0, step / self.decay_steps)
                self.epsilon = self.epsilon_end + (self.epsilon_start - self.epsilon_end) * (1 - fraction)
            else:
                self.epsilon = max(self.epsilon_end, self.epsilon - 
                                  (self.epsilon_start - self.epsilon_end) / self.decay_steps)
        elif self.use_decay_schedule == 'exponential':
            # Decaimiento exponencial
            self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)
        # Si no hay decay schedule, epsilon se mantiene constante
    
    def _run_episode(self, env: Any, max_steps: int, render: bool) -> Tuple[float, int]:
        """
        Ejecuta un episodio de entrenamiento.
        
        Parámetros:
        -----------
        env : Any
            Entorno compatible con OpenAI Gym
        max_steps : int
            Máximo número de pasos por episodio
        render : bool
            Si renderizar el entorno durante el entrenamiento
            
        Retorna:
        --------
        Tuple[float, int]
            Recompensa total y número de pasos del episodio
        """
        state, _ = env.reset()
        episode_reward = 0
        steps = 0
        
        for _ in range(max_steps):
            if render:
                env.render()
            
            # Seleccionar y ejecutar acción
            action = self.get_action(state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            
            # Actualizar tabla Q
            self.update(state, action, reward, next_state, done)
            
            # Actualizar contadores y estadísticas
            state = next_state
            episode_reward += reward
            steps += 1
            self.total_steps += 1
            
            # Actualizar epsilon por paso si corresponde
            if self.use_decay_schedule:
                self.update_epsilon(self.total_steps)
            
            if done:
                break
                
        return episode_reward, steps
    
    def _update_history(
        self, 
        history: Dict[str, List[float]], 
        episode_reward: float, 
        steps: int, 
        episode_rewards_window: List[float], 
        log_interval: int
    ) -> float:
        """
        Actualiza el historial de entrenamiento con los resultados del episodio.
        
        Parámetros:
        -----------
        history : Dict[str, List[float]]
            Historial de entrenamiento
        episode_reward : float
            Recompensa del episodio
        steps : int
            Pasos del episodio
        episode_rewards_window : List[float]
            Ventana de recompensas recientes
        log_interval : int
            Intervalo para el cálculo de promedio móvil
            
        Retorna:
        --------
        float
            Recompensa promedio actual
        """
        history['episode_rewards'].append(episode_reward)
        history['episode_lengths'].append(steps)
        history['epsilons'].append(self.epsilon)
        
        # Mantener una ventana de recompensas para promedio móvil
        episode_rewards_window.append(episode_reward)
        if len(episode_rewards_window) > log_interval:
            episode_rewards_window.pop(0)
        
        avg_reward = np.mean(episode_rewards_window)
        history['avg_rewards'].append(avg_reward)
        
        return avg_reward
    
    def train(
        self, 
        env: Any, 
        episodes: int = QLEARNING_CONFIG['episodes'],
        max_steps: int = QLEARNING_CONFIG['max_steps_per_episode'],
        render: bool = False,
        log_interval: int = QLEARNING_CONFIG['log_interval']
    ) -> Dict[str, List[float]]:
        """
        Entrena el agente Q-Learning en el entorno.
        
        Parámetros:
        -----------
        env : Any
            Entorno compatible con OpenAI Gym
        episodes : int, opcional
            Número de episodios para entrenar (default: QLEARNING_CONFIG['episodes'])
        max_steps : int, opcional
            Máximo número de pasos por episodio (default: QLEARNING_CONFIG['max_steps_per_episode'])
        render : bool, opcional
            Si renderizar el entorno durante el entrenamiento (default: False)
        log_interval : int, opcional
            Cada cuántos episodios mostrar estadísticas (default: QLEARNING_CONFIG['log_interval'])
            
        Retorna:
        --------
        Dict[str, List[float]]
            Historia de entrenamiento
        """
        history = {
            'episode_rewards': [],
            'episode_lengths': [],
            'epsilons': [],
            'avg_rewards': []
        }
        
        # Variables para seguimiento
        episode_rewards_window = []
        start_time = time.time()
        
        for episode in range(episodes):
            # Ejecutar episodio
            episode_reward, steps = self._run_episode(env, max_steps, render)
            
            # Actualizar epsilon después de cada episodio si no se hace por paso
            if not self.use_decay_schedule:
                self.update_epsilon()
            
            # Actualizar estadísticas
            avg_reward = self._update_history(history, episode_reward, steps, 
                                            episode_rewards_window, log_interval)
            
            # Mostrar progreso
            if (episode + 1) % log_interval == 0:
                elapsed_time = time.time() - start_time
                print(f"Episodio {episode+1}/{episodes} - "
                      f"Recompensa: {episode_reward:.2f}, "
                      f"Promedio: {avg_reward:.2f}, "
                      f"Epsilon: {self.epsilon:.4f}, "
                      f"Tiempo: {elapsed_time:.2f}s")
                start_time = time.time()
        
        return history

class QlearningModel(nn.Module):
    """
    Modelo Q-Learning compatible con PyTorch.
    
    Combina un agente Q-Learning con una arquitectura PyTorch para permitir
    el uso de datos continuos y la integración con el framework de entrenamiento.
    """
    
    def __init__(
        self, 
        cgm_shape: Tuple[int, ...], 
        other_features_shape: Tuple[int, ...],
        n_states: int = QLEARNING_CONFIG['n_states'],
        n_actions: int = QLEARNING_CONFIG['n_actions']
    ) -> None:
        """
        Inicializa el modelo Q-Learning.
        
        Parámetros:
        -----------
        cgm_shape : Tuple[int, ...]
            Forma de los datos CGM
        other_features_shape : Tuple[int, ...]
            Forma de otras características
        n_states : int, opcional
            Número de estados discretos (default: QLEARNING_CONFIG['n_states'])
        n_actions : int, opcional
            Número de acciones discretas (default: QLEARNING_CONFIG['n_actions'])
        """
        super().__init__()
        
        # Guardar dimensiones de entrada
        self.cgm_shape = cgm_shape
        self.other_features_shape = other_features_shape
        
        # Crear agente Q-Learning
        self.agent = QLearning(
            n_states=n_states,
            n_actions=n_actions,
            learning_rate=QLEARNING_CONFIG['learning_rate'],
            gamma=QLEARNING_CONFIG['gamma'],
            epsilon_start=QLEARNING_CONFIG['epsilon_start'],
            epsilon_end=QLEARNING_CONFIG['epsilon_end'],
            epsilon_decay=QLEARNING_CONFIG['epsilon_decay'],
            use_decay_schedule=QLEARNING_CONFIG['use_decay_schedule'],
            decay_steps=QLEARNING_CONFIG['decay_steps']
        )
        
        # Crear capas para procesar datos continuos
        cgm_size = np.prod(cgm_shape)
        other_size = np.prod(other_features_shape) if other_features_shape else 0
        
        # Encoder para características
        self.encoder = nn.Sequential(
            nn.Flatten(),
            nn.Linear(cgm_size + other_size, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        
        # Cabeza de salida para insulina
        self.head = nn.Linear(32, 1)
        
        # Mezclador para combinar salidas de NN y Q-Learning
        self.alpha = nn.Parameter(torch.tensor([0.5]), requires_grad=True)
        
        # Métricas
        self.loss_tracker = 0.0
        self.mae_metric = 0.0
        self.rmse_metric = 0.0
        
        # Cola para actualizaciones asíncronas del agente
        self.update_queue = []
        self.max_queue_size = 10
        
        # Device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    def _discretize_state(self, cgm_data: torch.Tensor, other_features: torch.Tensor) -> int:
        """
        Convierte datos continuos en un estado discreto para Q-Learning.
        
        Parámetros:
        -----------
        cgm_data : torch.Tensor
            Datos CGM
        other_features : torch.Tensor
            Otras características
            
        Retorna:
        --------
        int
            Índice de estado discreto
        """
        # Extraer características relevantes (se puede modificar según el dominio)
        if len(cgm_data.shape) > 1:
            # Tomar último valor de CGM como principal característica 
            cgm_value = cgm_data[-1].item() if len(cgm_data.shape) == 1 else cgm_data[0, -1].item()
        else:
            cgm_value = cgm_data.item()
            
        # Detectar si los datos ya están normalizados (valores típicamente entre -1 y 1)
        is_normalized = -3.0 <= cgm_value <= 3.0
        
        if is_normalized:
            # Caso 1: Datos ya normalizados (típicamente entre -1 y 1)
            # Ajustar al rango [0, 1] para discretización
            normalized_cgm = (cgm_value + 1) / 2  # Convierte de [-1, 1] a [0, 1]
            normalized_cgm = np.clip(normalized_cgm, 0, 1)  # Garantizar rango [0, 1]
        else:
            # Caso 2: Datos crudos (valores CGM típicos en mg/dL)
            max_cgm = 400.0  # Valor máximo esperado de CGM
            min_cgm = 40.0   # Valor mínimo esperado de CGM
            
            # Normalizar y discretizar
            if cgm_value > max_cgm:
                print(f"{CONST_ERROR_VALOR} CGM={cgm_value}")
                cgm_value = max_cgm
            elif cgm_value < min_cgm:
                print(f"{CONST_ERROR_VALOR} CGM={cgm_value}")
                cgm_value = min_cgm
                
            normalized_cgm = (cgm_value - min_cgm) / (max_cgm - min_cgm)
        
        # Calcular estado discreto
        discrete_state = int(normalized_cgm * (self.agent.n_states - 1))
        return discrete_state
    
    def _action_to_insulin(self, action: int) -> float:
        """
        Convierte una acción discreta en una dosis de insulina continua.
        
        Parámetros:
        -----------
        action : int
            Índice de acción discreta
            
        Retorna:
        --------
        float
            Dosis de insulina correspondiente
        """
        max_insulin = 20.0  # Valor máximo de dosis de insulina
        return (action / (self.agent.n_actions - 1)) * max_insulin
    
    def _insulin_to_action(self, insulin: float) -> int:
        """
        Convierte una dosis de insulina continua en una acción discreta.
        
        Parámetros:
        -----------
        insulin : float
            Dosis de insulina
            
        Retorna:
        --------
        int
            Índice de acción discreta correspondiente
        """
        max_insulin = 20.0  # Valor máximo de dosis de insulina
        normalized = np.clip(insulin / max_insulin, 0, 1)
        return int(round(normalized * (self.agent.n_actions - 1)))
    
    def forward(self, cgm_data: torch.Tensor, other_features: torch.Tensor) -> torch.Tensor:
        """
        Realiza una predicción híbrida combinando Q-Learning y redes neuronales.
        
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
        
        # Parte 1: Predicción del modelo neuronal (diferenciable)
        # Aplanar y concatenar entradas
        cgm_flat = cgm_data.reshape(batch_size, -1)
        other_flat = other_features.reshape(batch_size, -1) if other_features.numel() > 0 else torch.zeros((batch_size, 0), device=self.device)
        
        combined = torch.cat([cgm_flat, other_flat], dim=1)
        
        # Generar predicción usando redes neuronales
        features = self.encoder(combined)
        nn_predictions = self.head(features)
        
        # Parte 2: Predicción Q-learning (no diferenciable)
        with torch.no_grad():
            ql_predictions = torch.zeros((batch_size, 1), device=self.device)
            
            for i in range(batch_size):
                # Discretizar estado
                state = self._discretize_state(cgm_data[i], other_features[i] if other_features.numel() > 0 else None)
                
                # Obtener acción óptima según Q-Learning
                action = self.agent.get_action(state)
                
                # Convertir a dosis de insulina
                insulin_dose = self._action_to_insulin(action)
                ql_predictions[i, 0] = insulin_dose
        
        # Mezclar predicciones usando el parámetro alpha aprendible
        alpha = torch.sigmoid(self.alpha)  # Limitar entre 0 y 1
        combined_predictions = alpha * nn_predictions + (1 - alpha) * ql_predictions
        
        # Programar actualización asíncrona si estamos en entrenamiento
        if self.training:
            self._schedule_update(cgm_data.detach(), other_features.detach(), combined_predictions.detach())
        
        return combined_predictions
    
    def _schedule_update(self, cgm_data: torch.Tensor, other_features: torch.Tensor, 
                        predictions: torch.Tensor) -> None:
        """
        Programa una actualización asíncrona del agente Q-Learning.
        
        Parámetros:
        -----------
        cgm_data : torch.Tensor
            Datos CGM
        other_features : torch.Tensor
            Otras características
        predictions : torch.Tensor
            Predicciones generadas
        """
        # Almacenar en cola para actualización posterior
        self.update_queue.append((
            cgm_data.cpu().numpy(),
            other_features.cpu().numpy() if other_features.numel() > 0 else None,
            predictions.cpu().numpy()
        ))
        
        # Procesar entradas antiguas si la cola es muy grande
        if len(self.update_queue) > self.max_queue_size:
            sample = self.update_queue.pop(0)
            self._process_updates(*sample)
    
    def _process_updates(self, cgm_np: np.ndarray, other_np: np.ndarray, 
                        predictions_np: np.ndarray) -> None:
        """
        Procesa actualizaciones del agente Q-Learning de forma asíncrona.
        
        Parámetros:
        -----------
        cgm_np : np.ndarray
            Datos CGM
        other_np : np.ndarray
            Otras características
        predictions_np : np.ndarray
            Predicciones generadas
        """
        batch_size = cgm_np.shape[0]
        
        for i in range(batch_size):
            # Discretizar estado actual
            state = self._discretize_state(torch.tensor(cgm_np[i]), 
                                         torch.tensor(other_np[i]) if other_np is not None else None)
            
            # Convertir predicción a acción
            action = self._insulin_to_action(predictions_np[i].item())
            
            # Simular transición (en entorno real esto vendría del ambiente)
            reward = 1.0  # Recompensa por defecto
            next_state = min(state + 1, self.agent.n_states - 1)  # Estado siguiente simulado
            done = False
            
            # Actualizar agente
            self.agent.update(state, action, reward, next_state, done)

    def save(self, path: str) -> None:
        """
        Guarda el modelo y la tabla Q.
        
        Parámetros:
        -----------
        path : str
            Ruta base para guardar el modelo
        """
        # Guardar tabla Q
        q_table_path = path + QTABLE_SUFFIX
        np.save(q_table_path, self.agent.q_table.cpu().numpy())
        
        # Guardar pesos del modelo
        torch.save(self.state_dict(), path + WRAPPER_WEIGHTS_SUFFIX)
        
        print(f"{CONST_MODEL_PATH} {path}")
    
    def load(self, path: str) -> None:
        """
        Carga el modelo y la tabla Q.
        
        Parámetros:
        -----------
        path : str
            Ruta base para cargar el modelo
        """
        # Cargar tabla Q
        q_table_path = path + QTABLE_SUFFIX
        if os.path.exists(q_table_path):
            q_table = np.load(q_table_path)
            self.agent.q_table = torch.tensor(q_table, device=self.device)
        
        # Cargar pesos del modelo
        model_path = path + WRAPPER_WEIGHTS_SUFFIX
        if os.path.exists(model_path):
            self.load_state_dict(torch.load(model_path, map_location=self.device))


def create_q_learning_model(cgm_shape: Tuple[int, ...], other_features_shape: Tuple[int, ...]) -> nn.Module:
    """
    Crea una instancia del modelo Q-Learning.
    
    Parámetros:
    -----------
    cgm_shape : Tuple[int, ...]
        Forma de los datos CGM
    other_features_shape : Tuple[int, ...]
        Forma de otras características
        
    Retorna:
    --------
    nn.Module
        Instancia del modelo Q-Learning
    """
    # Cargar configuración
    config = QLEARNING_CONFIG
    
    # Crear modelo Q-Learning
    model = QlearningModel(
        cgm_shape=cgm_shape,
        other_features_shape=other_features_shape,
        n_states=config['n_states'],
        n_actions=config['n_actions']
    )
    
    return model


def create_q_learning_model_wrapper(cgm_shape: Tuple[int, ...], other_features_shape: Tuple[int, ...], **kwargs) -> Any:
    """
    Crea un modelo Q-Learning envuelto en RLModelWrapperPyTorch.
    
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
    Any
        Modelo Q-Learning envuelto para compatibilidad con el sistema
    """
    # Definir función creadora
    def model_creator_fn() -> nn.Module:
        return create_q_learning_model(cgm_shape, other_features_shape)
    
    # Crear wrapper
    model_wrapper = RLModelWrapperPyTorch(model_cls=model_creator_fn)
    
    return model_wrapper


def model_creator() -> Callable[[Tuple[int, ...], Tuple[int, ...]], RLModelWrapperPyTorch]:
    """
    Devuelve una función creadora del modelo Q Learning compatible con la infraestructura.
    
    Retorna:
    --------
    Callable[[Tuple[int, ...], Tuple[int, ...]], RLModelWrapperPyTorch]
        Función que crea un modelo Q Learning envuelto
    """
    return create_q_learning_model_wrapper