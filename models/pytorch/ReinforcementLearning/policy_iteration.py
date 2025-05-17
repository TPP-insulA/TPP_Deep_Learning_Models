import os
import sys
import numpy as np
import torch
import torch.nn as nn
from typing import Dict, List, Tuple, Optional, Union, Any, Callable

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../.."))
sys.path.append(PROJECT_ROOT)

from config.models_config import POLICY_ITERATION_CONFIG
from constants.constants import CONST_DEFAULT_SEED
from custom.ReinforcementLearning.rl_pt import RLModelWrapperPyTorch
from custom.printer import print_warning

# Constantes
FIGURES_DIR = os.path.join(PROJECT_ROOT, 'figures', 'pytorch', 'policy_iteration')
os.makedirs(FIGURES_DIR, exist_ok=True)

# Constantes para cadenas repetidas
CONST_VALUE = "value"
CONST_POLICY = "policy"
CONST_TRANSITIONS = "transitions"
CONST_REWARDS = "rewards"
CONST_MAX_ITER = "max_iterations"
CONST_GAMMA = "gamma"
CONST_THETA = "theta"
CONST_MODEL_NAME = "policy_iteration"

class PolicyIteration:
    """
    Implementación de Policy Iteration para predicción y control en aprendizaje por refuerzo.
    
    Esta clase proporciona algoritmos para encontrar políticas óptimas a través de
    iteración de políticas, alternando entre evaluación de políticas y mejora de políticas.
    """
    
    def __init__(
        self,
        n_states: int,
        n_actions: int,
        gamma: float = POLICY_ITERATION_CONFIG[CONST_GAMMA],
        theta: float = POLICY_ITERATION_CONFIG[CONST_THETA],
        max_iterations: int = POLICY_ITERATION_CONFIG[CONST_MAX_ITER],
        evaluation_mode: bool = False,
        seed: int = CONST_DEFAULT_SEED,
        cgm_shape: Optional[Tuple[int, ...]] = None,
        other_features_shape: Optional[Tuple[int, ...]] = None
    ) -> None:
        """
        Inicializa el agente de Policy Iteration.
        
        Parámetros:
        -----------
        n_states : int
            Número de estados discretos en el entorno.
        n_actions : int
            Número de acciones posibles.
        gamma : float, opcional
            Factor de descuento para recompensas futuras (default: POLICY_ITERATION_CONFIG['gamma']).
        theta : float, opcional
            Umbral de convergencia para evaluación de políticas (default: POLICY_ITERATION_CONFIG['theta']).
        max_iterations : int, opcional
            Número máximo de iteraciones (default: POLICY_ITERATION_CONFIG['max_iterations']).
        evaluation_mode : bool, opcional
            Si el agente está en modo evaluación (default: False).
        seed : int, opcional
            Semilla para reproducibilidad (default: 42).
        cgm_shape : Optional[Tuple[int, ...]], opcional
            Forma de los datos CGM (default: None).
        other_features_shape : Optional[Tuple[int, ...]], opcional
            Forma de otras características (default: None).
        """
        self.n_states = n_states
        self.n_actions = n_actions
        self.gamma = gamma
        self.theta = theta
        self.max_iterations = max_iterations
        self.evaluation_mode = evaluation_mode
        
        # Configurar semilla para reproducibilidad
        torch.manual_seed(seed)
        self.rng = np.random.Generator(np.random.PCG64(seed))
        
        # Inicializar tabla de valores de estado
        self.v_table = torch.zeros(n_states)
        
        # Inicializar política (equiprobable inicialmente)
        self.policy = torch.ones((n_states, n_actions)) / n_actions
        
        # Modelo del entorno (para aprendizaje por modelo)
        self.transitions = {}  # {(s,a): {s': conteo}}
        self.rewards = {}      # {(s,a,s'): [recompensas]}
        
        # Para seguimiento
        self.policy_changes = []
        self.value_changes = []
        
        # Guardar formas para mapeo
        self.cgm_shape = cgm_shape
        self.other_features_shape = other_features_shape
        
        # Contadores para registro de aprendizaje
        self.iterations = 0
        self.updates = 0
    
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
        
        # Asegurar límites válidos
        return max(0, min(state, self.n_states - 1))
    
    def _estimate_transition_model(self, experiences: List[Tuple[int, int, float, int]]) -> None:
        """
        Estima un modelo del entorno basado en experiencias.
        
        Parámetros:
        -----------
        experiences : List[Tuple[int, int, float, int]]
            Lista de tuplas (estado, acción, recompensa, siguiente_estado)
        """
        for state, action, reward, next_state in experiences:
            # Actualizar modelo de transición
            if (state, action) not in self.transitions:
                self.transitions[(state, action)] = {}
            
            if next_state not in self.transitions[(state, action)]:
                self.transitions[(state, action)][next_state] = 0
                
            self.transitions[(state, action)][next_state] += 1
            
            # Actualizar modelo de recompensas
            if (state, action, next_state) not in self.rewards:
                self.rewards[(state, action, next_state)] = []
                
            self.rewards[(state, action, next_state)].append(reward)
    
    def _get_transition_probability(self, state: int, action: int, next_state: int) -> float:
        """
        Obtiene la probabilidad de transición del modelo estimado.
        
        Parámetros:
        -----------
        state : int
            Estado actual
        action : int
            Acción tomada
        next_state : int
            Estado siguiente
            
        Retorna:
        --------
        float
            Probabilidad de transición estimada
        """
        if (state, action) not in self.transitions:
            return 0.0
            
        total_transitions = sum(self.transitions[(state, action)].values())
        
        if next_state not in self.transitions[(state, action)]:
            return 0.0
            
        return self.transitions[(state, action)][next_state] / total_transitions
    
    def _get_expected_reward(self, state: int, action: int, next_state: int) -> float:
        """
        Obtiene la recompensa esperada del modelo estimado.
        
        Parámetros:
        -----------
        state : int
            Estado actual
        action : int
            Acción tomada
        next_state : int
            Estado siguiente
            
        Retorna:
        --------
        float
            Recompensa esperada
        """
        if (state, action, next_state) not in self.rewards:
            return 0.0
            
        return sum(self.rewards[(state, action, next_state)]) / len(self.rewards[(state, action, next_state)])
    
    def _calculate_action_value(self, state: int, action: int) -> float:
        """
        Calcula el valor esperado para una acción en un estado dado.
        
        Parámetros:
        -----------
        state : int
            Estado actual
        action : int
            Acción a evaluar
            
        Retorna:
        --------
        float
            Valor esperado de la acción
        """
        action_value = 0.0
        
        # Si tenemos un modelo del entorno, usarlo
        if (state, action) in self.transitions:
            for next_s in self.transitions[(state, action)]:
                # Probabilidad de transición
                trans_prob = self._get_transition_probability(state, action, next_s)
                
                # Recompensa esperada
                reward = self._get_expected_reward(state, action, next_s)
                
                # Actualizar valor esperado de la acción
                action_value += trans_prob * (reward + self.gamma * self.v_table[next_s].item())
                
        return action_value
    
    def _calculate_state_value(self, state: int) -> float:
        """
        Calcula el valor esperado para un estado según la política actual.
        
        Parámetros:
        -----------
        state : int
            Estado a evaluar
            
        Retorna:
        --------
        float
            Nuevo valor del estado
        """
        new_value = 0.0
        
        # Calcular el valor esperado basado en la política actual
        for a in range(self.n_actions):
            action_prob = self.policy[state, a].item()
            
            # Si la acción tiene probabilidad 0, saltarla
            if action_prob == 0:
                continue
                
            # Calcular y sumar el valor ponderado de la acción
            action_value = self._calculate_action_value(state, a)
            new_value += action_prob * action_value
            
        return new_value
    
    def policy_evaluation(self) -> None:
        """
        Evalúa la política actual, actualizando la función de valor de estado.
        """
        # Iteraciones para evaluación de política
        for _ in range(self.max_iterations):
            delta = 0.0
            
            # Recorrer todos los estados
            for s in range(self.n_states):
                old_value = self.v_table[s].item()
                
                # Calcular nuevo valor del estado
                new_value = self._calculate_state_value(s)
                
                # Actualizar la tabla de valores
                self.v_table[s] = new_value
                
                # Actualizar delta para seguimiento de convergencia
                delta = max(delta, abs(old_value - new_value))
            
            # Verificar convergencia
            if delta < self.theta:
                break
    
    def policy_improvement(self) -> bool:
        """
        Mejora la política basada en la función de valor actual.
        
        Retorna:
        --------
        bool
            True si la política cambió, False si es estable
        """
        policy_stable = True
        
        # Recorrer todos los estados
        for s in range(self.n_states):
            old_action = torch.argmax(self.policy[s]).item()
            
            # Calcular valores de acción para el estado actual
            action_values = torch.zeros(self.n_actions)
            
            for a in range(self.n_actions):
                # Si tenemos un modelo del entorno, usarlo
                if (s, a) in self.transitions:
                    for next_s in self.transitions[(s, a)]:
                        # Probabilidad de transición
                        trans_prob = self._get_transition_probability(s, a, next_s)
                        
                        # Recompensa esperada
                        reward = self._get_expected_reward(s, a, next_s)
                        
                        # Actualizar valor de la acción
                        action_values[a] += trans_prob * (reward + self.gamma * self.v_table[next_s].item())
            
            # Encontrar la mejor acción (o acciones)
            best_action = torch.argmax(action_values).item()
            
            # Política determinista con la mejor acción
            self.policy[s] = torch.zeros(self.n_actions)
            self.policy[s, best_action] = 1.0
            
            # Verificar si la política cambió
            if old_action != best_action:
                policy_stable = False
                
        return policy_stable
    
    def policy_iteration(self, max_iters: int = 100) -> None:
        """
        Ejecuta el algoritmo de iteración de políticas.
        
        Parámetros:
        -----------
        max_iters : int, opcional
            Número máximo de iteraciones de política (default: 100)
        """
        for i in range(max_iters):
            # Evaluar la política actual
            self.policy_evaluation()
            
            # Mejorar la política basada en la evaluación
            policy_stable = self.policy_improvement()
            
            # Si la política es estable, finalizar
            if policy_stable:
                break
        
        self.iterations += i + 1
    
    def update_from_experience(self, experience: List[Tuple[int, int, float, int]]) -> None:
        """
        Actualiza el agente con nuevas experiencias.
        
        Parámetros:
        -----------
        experience : List[Tuple[int, int, float, int]]
            Lista de tuplas (estado, acción, recompensa, siguiente_estado)
        """
        # Actualizar el modelo con las nuevas experiencias
        self._estimate_transition_model(experience)
        
        # Ejecutar política iteración para actualizar valores y política
        self.policy_iteration()
        
        self.updates += 1
    
    def select_action(self, state: int) -> int:
        """
        Selecciona una acción según la política actual.
        
        Parámetros:
        -----------
        state : int
            Estado actual
            
        Retorna:
        --------
        int
            Acción seleccionada
        """
        if self.evaluation_mode:
            # Modo determinista (usar la mejor acción)
            return torch.argmax(self.policy[state]).item()
        else:
            # Modo estocástico (muestrear según la distribución de probabilidad)
            probs = self.policy[state].numpy()
            return self.rng.choice(self.n_actions, p=probs)

class PolicyIterationModel(nn.Module):
    """
    Modelo PyTorch para Policy Iteration.
    
    Encapsula un agente Policy Iteration para utilizarlo con la interfaz de PyTorch.
    Permite transferir conocimiento entre representaciones continuas y discretas.
    """
    
    def __init__(
        self,
        policy_iteration_agent: PolicyIteration,
        cgm_shape: Tuple[int, ...],
        other_features_shape: Tuple[int, ...]
    ) -> None:
        """
        Inicializa el modelo Policy Iteration.
        
        Parámetros:
        -----------
        policy_iteration_agent : PolicyIteration
            Agente Policy Iteration para aprendizaje por refuerzo
        cgm_shape : Tuple[int, ...]
            Forma de los datos CGM
        other_features_shape : Tuple[int, ...]
            Forma de otras características
        """
        super().__init__()
        self.policy_iteration_agent = policy_iteration_agent
        self.cgm_shape = cgm_shape
        self.other_features_shape = other_features_shape
        
        # Métricas
        self.loss_tracker = 0.0
        self.mae_metric = 0.0
        self.rmse_metric = 0.0
        
        # Capas diferenciables para el modelo híbrido
        cgm_size = np.prod(cgm_shape) if isinstance(cgm_shape, tuple) else cgm_shape
        other_size = np.prod(other_features_shape) if isinstance(other_features_shape, tuple) else other_features_shape
        
        # Codificador para procesar entradas
        self.encoder = nn.Sequential(
            nn.Flatten(),
            nn.Linear(cgm_size + other_size, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        
        # Cabeza de política para generar salidas
        self.policy_head = nn.Linear(32, 1)
        
        # Parámetro de mezcla entre el modelo neuronal y el agente RL
        self.alpha = nn.Parameter(torch.tensor([0.5]), requires_grad=True)
        
        # Buffer para actualizaciones asíncronas
        self.update_queue = []
        self.max_queue_size = 10

    def forward(self, cgm_data: torch.Tensor, other_features: torch.Tensor) -> torch.Tensor:
        """
        Realiza una predicción utilizando un enfoque híbrido de RL y redes neuronales.
        
        Parámetros:
        -----------
        cgm_data : torch.Tensor
            Datos CGM de entrada
        other_features : torch.Tensor
            Otras características de entrada
            
        Retorna:
        --------
        torch.Tensor
            Predicciones (dosis de insulina)
        """
        batch_size = cgm_data.size(0)
        
        # 1. Camino diferenciable (red neuronal)
        # Aplanar entradas y concatenar
        cgm_flat = cgm_data.reshape(batch_size, -1)
        other_flat = other_features.reshape(batch_size, -1)
        combined = torch.cat([cgm_flat, other_flat], dim=1)
        
        # Procesar a través del codificador y cabeza de política
        features = self.encoder(combined)
        nn_predictions = self.policy_head(features)
        
        # 2. Camino RL (policy iteration) - no diferenciable pero proporciona señales
        with torch.no_grad():
            rl_predictions = torch.zeros((batch_size, 1), device=cgm_data.device)
            
            # Calcular predicciones del agente RL
            cgm_np = cgm_data.detach().cpu().numpy()
            other_np = other_features.detach().cpu().numpy()
            
            for i in range(batch_size):
                # Mapear observación a estado
                state = self.policy_iteration_agent._map_observation_to_state(
                    cgm_np[i], other_np[i]
                )
                
                # Seleccionar acción (índice de insulina discreta)
                action = self.policy_iteration_agent.select_action(state)
                
                # Convertir acción discreta a dosis continua
                insulin_max = 20.0  # Valor máximo esperado
                insulin_dose = (action / (self.policy_iteration_agent.n_actions - 1)) * insulin_max
                
                rl_predictions[i, 0] = insulin_dose
        
        # Mezclar predicciones usando el parámetro alpha aprendible
        alpha = torch.sigmoid(self.alpha)  # Limitar entre 0 y 1
        combined_predictions = alpha * nn_predictions + (1 - alpha) * rl_predictions.detach()
        
        # Programar actualización asíncrona del agente RL
        if self.training:
            self._schedule_agent_update(cgm_data, other_features, combined_predictions.detach())
        
        return combined_predictions

    def _schedule_agent_update(self, cgm_data: torch.Tensor, other_features: torch.Tensor, 
                             predictions: torch.Tensor) -> None:
        """
        Programa una actualización asíncrona del agente RL.
        
        Parámetros:
        -----------
        cgm_data : torch.Tensor
            Datos CGM de entrada
        other_features : torch.Tensor
            Otras características de entrada
        predictions : torch.Tensor
            Predicciones del modelo para estos datos
        """
        # Almacenar datos para actualización posterior
        self.update_queue.append((
            cgm_data.detach().cpu().numpy(),
            other_features.detach().cpu().numpy(),
            predictions.detach().cpu().numpy()
        ))
        
        # Limitar tamaño de la cola
        if len(self.update_queue) > self.max_queue_size:
            # Procesar la muestra más antigua
            oldest_sample = self.update_queue.pop(0)
            self._update_agent(*oldest_sample)
    
    def _update_agent(self, cgm_np: np.ndarray, other_np: np.ndarray, 
                    predictions_np: np.ndarray) -> None:
        """
        Actualiza el agente RL sin bloquear el flujo de gradientes.
        
        Parámetros:
        -----------
        cgm_np : np.ndarray
            Datos CGM en formato NumPy
        other_np : np.ndarray
            Otras características en formato NumPy
        predictions_np : np.ndarray
            Predicciones en formato NumPy
        """
        # Crear experiencias para el agente
        experiences = []
        batch_size = len(predictions_np)
        max_insulin = 20.0
        
        for i in range(batch_size):
            # Obtener estado actual
            state = self.policy_iteration_agent._map_observation_to_state(
                cgm_np[i], other_np[i]
            )
            
            # Convertir la predicción a acción discreta
            pred_value = predictions_np[i].item() if isinstance(predictions_np[i], np.ndarray) else predictions_np[i]
            norm_pred = np.clip(pred_value / max_insulin, 0, 1)
            action = int(np.round(norm_pred * (self.policy_iteration_agent.n_actions - 1)))
            
            # Simular un estado siguiente (simplificación)
            next_state = min(state + 1, self.policy_iteration_agent.n_states - 1)
            
            # Asignar recompensa (en un entorno real esto vendría del ambiente)
            reward = 1.0  # Recompensa positiva constante para esta demostración
            
            # Añadir a experiencias
            experiences.append((state, action, reward, next_state))
        
        # Actualizar el agente con las experiencias
        if experiences:
            self.policy_iteration_agent.update_from_experience(experiences)

def create_policy_iteration_model(cgm_shape: Tuple[int, ...], other_features_shape: Tuple[int, ...], **kwargs) -> nn.Module:
    """
    Crea un modelo Policy Iteration para predicción de dosis de insulina.
    
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
    nn.Module
        Modelo Policy Iteration basado en PyTorch
    """
    # Configuración del espacio de estados y acciones
    n_states = kwargs.get('n_states', 1000)
    n_actions = kwargs.get('n_actions', 20)
    
    # Crear agente Policy Iteration
    policy_iteration_agent = PolicyIteration(
        n_states=n_states,
        n_actions=n_actions,
        gamma=POLICY_ITERATION_CONFIG[CONST_GAMMA],
        theta=POLICY_ITERATION_CONFIG[CONST_THETA],
        max_iterations=POLICY_ITERATION_CONFIG[CONST_MAX_ITER],
        cgm_shape=cgm_shape,
        other_features_shape=other_features_shape,
        seed=kwargs.get('seed', CONST_DEFAULT_SEED)
    )
    
    # Crear y devolver el modelo wrapper
    return PolicyIterationModel(
        policy_iteration_agent=policy_iteration_agent,
        cgm_shape=cgm_shape,
        other_features_shape=other_features_shape
    )

def create_policy_iteration_model_wrapper(cgm_shape: Tuple[int, ...], other_features_shape: Tuple[int, ...], **kwargs) -> RLModelWrapperPyTorch:
    """
    Crea un modelo Policy Iteration envuelto en RLModelWrapperPyTorch.
    
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
        Modelo Policy Iteration envuelto para compatibilidad con el sistema
    """
    # Definir función creadora que no toma argumentos
    def model_creator_fn():
        return create_policy_iteration_model(cgm_shape, other_features_shape, **kwargs)
    
    # Crear wrapper
    model_wrapper = RLModelWrapperPyTorch(model_cls=model_creator_fn)
    
    return model_wrapper

def model_creator() -> Callable[[Tuple[int, ...], Tuple[int, ...]], RLModelWrapperPyTorch]:
    """
    Devuelve una función creadora de modelos Policy Iteration compatible con la infraestructura.
    
    Retorna:
    --------
    Callable[[Tuple[int, ...], Tuple[int, ...]], RLModelWrapperPyTorch]
        Función que crea un modelo Policy Iteration envuelto
    """
    return create_policy_iteration_model_wrapper