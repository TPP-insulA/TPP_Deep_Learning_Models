"""
Doubly Robust Estimator (DRE)

Estimador que combina el método directo (Direct Method) con el muestreo por importancia (Importance Sampling) para proporcionar una evaluación más robusta de políticas de control de insulina.

1. Direct Method Learning: Aprende un modelo de la función Q para estimar valores, similar a FQE.
2. Importance Sampling: repone las experiencias de la política de comportamiento para estimar el rendimiento de la política objetivo.
3. Behavior Policy Learning: Modela la política que generó los datos de entrenamiento
4. Doubly Robust Estimation: Combina ambos métodos para obtener estimaciones más robustas
5. Bootstrap para Intervalos de Confianza: Calcula intervalos de confianza para las estimaciones

DR = DM + IS(R - Q)
donde:
- DR: Estimación Doubly Robust
- DM: Estimación por Método Directo
- IS: Estimación por Muestreo por Importancia
- R: Recompensas observadas
- Q: Valores Q estimados por la red Q

Si el modelo Q es preciso, la estimación DR es consistente y tiene menor varianza que IS solo; si los pesos de importancia son precisos, la estimación DR es consistente y tiene menor varianza que DM solo.
"""
import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from typing import Dict, Tuple, List, Any, Optional
from tqdm import tqdm

from constants.constants import (
    CONST_DEFAULT_BATCH_SIZE, OFFLINE_GAMMA, CONST_DEFAULT_EPOCHS,
    CONST_CONFIDENCE_LEVEL, CONST_IPS_CLIP
)
from validation.networks import QNetwork

class BehaviorPolicyNetwork(nn.Module):
    """
    Red neuronal para modelar la política de comportamiento que generó los datos.
    
    Parámetros:
    -----------
    cgm_input_dim : tuple
        Dimensiones de entrada para datos CGM
    other_input_dim : tuple
        Dimensiones de entrada para otras características
    action_dim : int
        Dimensión de la acción (dosis de insulina)
    hidden_dim : int, opcional
        Dimensión de las capas ocultas (default: 128)
    """
    
    def __init__(self, cgm_input_dim: tuple, other_input_dim: tuple, 
                action_dim: int = 1, hidden_dim: int = 128):
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
        
        # Capas combinadas para predecir parámetros de distribución
        combined_dim = hidden_dim // 2 + hidden_dim // 2
        self.combined_layer = nn.Sequential(
            nn.Linear(combined_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU()
        )
        
        # Salida para media y desviación estándar (política gaussiana)
        self.mean_layer = nn.Linear(hidden_dim // 2, action_dim)
        self.std_layer = nn.Sequential(
            nn.Linear(hidden_dim // 2, action_dim),
            nn.Softplus()  # Asegura que std sea positiva
        )
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.to(self.device)
    
    def forward(self, x_cgm: torch.Tensor, x_other: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Paso hacia adelante de la red.
        
        Parámetros:
        -----------
        x_cgm : torch.Tensor
            Datos CGM
        x_other : torch.Tensor
            Otras características
            
        Retorna:
        --------
        Tuple[torch.Tensor, torch.Tensor]
            Media y desviación estándar de la distribución de acción
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
        
        # Obtener parámetros de distribución
        mean = self.mean_layer(features)
        std = self.std_layer(features) + 1e-6  # Añadir pequeño epsilon para estabilidad
        
        return mean, std
    
    def log_prob(self, x_cgm: torch.Tensor, x_other: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """
        Calcula el logaritmo de la probabilidad de una acción dada el estado.
        
        Parámetros:
        -----------
        x_cgm : torch.Tensor
            Datos CGM
        x_other : torch.Tensor
            Otras características
        action : torch.Tensor
            Acción para evaluar (dosis de insulina)
            
        Retorna:
        --------
        torch.Tensor
            Logaritmo de la probabilidad
        """
        mean, std = self.forward(x_cgm, x_other)
        
        # Distribución normal
        from torch.distributions import Normal
        dist = Normal(mean, std)
        
        return dist.log_prob(action)


class DoublyRobustEstimator:
    """
    Evaluador de políticas usando el método Doubly Robust.
    
    Parámetros:
    -----------
    cgm_input_dim : tuple
        Dimensiones de entrada para datos CGM
    other_input_dim : tuple
        Dimensiones de entrada para otras características
    hidden_dim : int, opcional
        Dimensión de las capas ocultas (default: 128)
    gamma : float, opcional
        Factor de descuento para recompensas futuras (default: 0.99)
    lr : float, opcional
        Tasa de aprendizaje (default: 0.001)
    """
    
    def __init__(self, cgm_input_dim: tuple, other_input_dim: tuple,
                hidden_dim: int = 128, gamma: float = OFFLINE_GAMMA,
                lr: float = 0.001):
        self.cgm_input_dim = cgm_input_dim
        self.other_input_dim = other_input_dim
        self.hidden_dim = hidden_dim
        self.gamma = gamma
        self.lr = lr
        
        # Inicializar red Q para Direct Method
        self.q_network = QNetwork(
            cgm_input_dim=cgm_input_dim,
            other_input_dim=other_input_dim,
            action_dim=1,  # Dosis de insulina
            hidden_dim=hidden_dim
        )
        
        # Inicializar red de política de comportamiento
        self.behavior_policy = BehaviorPolicyNetwork(
            cgm_input_dim=cgm_input_dim,
            other_input_dim=other_input_dim,
            action_dim=1,
            hidden_dim=hidden_dim
        )
        
        # Optimizadores
        self.q_optimizer = optim.Adam(self.q_network.parameters(), lr=self.lr)
        self.behavior_optimizer = optim.Adam(self.behavior_policy.parameters(), lr=self.lr)
        
        # Criterios de pérdida
        self.mse_criterion = nn.MSELoss()
        self.nll_criterion = nn.GaussianNLLLoss()
        
        # Dispositivo
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.q_network.to(self.device)
        self.behavior_policy.to(self.device)
        
        # Bootstrap para intervalos de confianza
        self.bootstrap_estimates = []
    
    def _generate_rewards(self, x_cgm: np.ndarray, x_other: np.ndarray, 
                        actions: np.ndarray) -> np.ndarray:
        """
        Genera recompensas basadas en el mantenimiento de glucosa en rango.
        
        Parámetros:
        -----------
        x_cgm : np.ndarray
            Datos CGM
        x_other : np.ndarray
            Otras características
        actions : np.ndarray
            Acciones (dosis de insulina)
            
        Retorna:
        --------
        np.ndarray
            Recompensas calculadas
        """
        # Extraer valores de glucosa actuales (último valor de cada serie CGM)
        current_glucose = np.array([x[-1] for x in x_cgm.reshape(-1, self.cgm_input_dim[0])])
        
        # Inicializar recompensas
        rewards = np.zeros_like(actions, dtype=np.float32)
        
        # Asignar recompensas basadas en el rango de glucosa
        # En rango (70-180 mg/dL) - recompensa positiva
        in_range = (current_glucose >= 70.0) & (current_glucose <= 180.0)
        rewards[in_range] = 1.0
        
        # Hipoglucemia (<70 mg/dL) - penalización severa
        hypo = current_glucose < 70.0
        rewards[hypo] = -2.0
        
        # Hiperglucemia (>180 mg/dL) - penalización moderada
        hyper = current_glucose > 180.0
        rewards[hyper] = -1.0
        
        return rewards
    
    def fit(self, x_cgm: np.ndarray, x_other: np.ndarray, y_actions: np.ndarray,
           validation_data: Optional[Tuple] = None,
           batch_size: int = CONST_DEFAULT_BATCH_SIZE,
           epochs: int = CONST_DEFAULT_EPOCHS,
           bootstrap_iterations: int = 20) -> Dict[str, List[float]]:
        """
        Entrena los modelos de Direct Method y Behavior Policy.
        
        Parámetros:
        -----------
        x_cgm : np.ndarray
            Datos CGM de entrenamiento
        x_other : np.ndarray
            Otras características de entrenamiento
        y_actions : np.ndarray
            Acciones (dosis de insulina) reales
        validation_data : Optional[Tuple], opcional
            Datos de validación como ((x_cgm_val, x_other_val), y_val) (default: None)
        batch_size : int, opcional
            Tamaño de lote (default: 32)
        epochs : int, opcional
            Número de épocas (default: 10)
        bootstrap_iterations : int, opcional
            Número de iteraciones bootstrap para intervalos de confianza (default: 20)
            
        Retorna:
        --------
        Dict[str, List[float]]
            Historial de entrenamiento
        """
        # Generar recompensas
        rewards = self._generate_rewards(x_cgm, x_other, y_actions)
        
        # Crear DataLoader
        dataset = TensorDataset(
            torch.FloatTensor(x_cgm),
            torch.FloatTensor(x_other),
            torch.FloatTensor(y_actions).reshape(-1, 1),
            torch.FloatTensor(rewards).reshape(-1, 1)
        )
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        
        # Historial de entrenamiento
        history = {
            'q_loss': [],
            'behavior_loss': [],
            'val_q_loss': [],
            'val_behavior_loss': []
        }
        
        # Entrenamiento principal
        for epoch in range(epochs):
            epoch_q_loss = 0.0
            epoch_behavior_loss = 0.0
            self.q_network.train()
            self.behavior_policy.train()
            
            for batch_cgm, batch_other, batch_actions, batch_rewards in tqdm(dataloader, desc=f"Época {epoch+1}/{epochs}"):
                batch_cgm = batch_cgm.to(self.device)
                batch_other = batch_other.to(self.device)
                batch_actions = batch_actions.to(self.device)
                batch_rewards = batch_rewards.to(self.device)
                
                # Entrenar red Q (Direct Method)
                self.q_optimizer.zero_grad()
                q_values = self.q_network(batch_cgm, batch_other, batch_actions)
                q_loss = self.mse_criterion(q_values, batch_rewards)
                q_loss.backward()
                self.q_optimizer.step()
                
                # Entrenar política de comportamiento
                self.behavior_optimizer.zero_grad()
                mean, std = self.behavior_policy(batch_cgm, batch_other)
                behavior_loss = self.nll_criterion(mean, batch_actions.squeeze(), std.pow(2))
                behavior_loss.backward()
                self.behavior_optimizer.step()
                
                epoch_q_loss += q_loss.item()
                epoch_behavior_loss += behavior_loss.item()
            
            # Registrar pérdida de entrenamiento
            avg_q_loss = epoch_q_loss / len(dataloader)
            avg_behavior_loss = epoch_behavior_loss / len(dataloader)
            history['q_loss'].append(avg_q_loss)
            history['behavior_loss'].append(avg_behavior_loss)
            
            # Validación si hay datos disponibles
            if validation_data:
                val_q_loss, val_behavior_loss = self._validate(validation_data[0][0], validation_data[0][1], validation_data[1])
                history['val_q_loss'].append(val_q_loss)
                history['val_behavior_loss'].append(val_behavior_loss)
                print(f"Época {epoch+1}/{epochs} - Q Loss: {avg_q_loss:.4f} - Behavior Loss: {avg_behavior_loss:.4f} - Val Q Loss: {val_q_loss:.4f} - Val Behavior Loss: {val_behavior_loss:.4f}")
            else:
                print(f"Época {epoch+1}/{epochs} - Q Loss: {avg_q_loss:.4f} - Behavior Loss: {avg_behavior_loss:.4f}")
        
        # Realizar bootstrap para intervalos de confianza
        self._bootstrap(x_cgm, x_other, y_actions, bootstrap_iterations)
        
        return history
    
    def _validate(self, x_cgm_val: np.ndarray, x_other_val: np.ndarray, 
                y_val: np.ndarray) -> Tuple[float, float]:
        """
        Valida los modelos con datos de validación.
        
        Parámetros:
        -----------
        x_cgm_val : np.ndarray
            Datos CGM de validación
        x_other_val : np.ndarray
            Otras características de validación
        y_val : np.ndarray
            Acciones (dosis) de validación
            
        Retorna:
        --------
        Tuple[float, float]
            (pérdida Q, pérdida de comportamiento)
        """
        self.q_network.eval()
        self.behavior_policy.eval()
        
        with torch.no_grad():
            # Convertir a tensores
            x_cgm_tensor = torch.FloatTensor(x_cgm_val).to(self.device)
            x_other_tensor = torch.FloatTensor(x_other_val).to(self.device)
            y_tensor = torch.FloatTensor(y_val).reshape(-1, 1).to(self.device)
            
            # Generar recompensas para validación
            rewards = self._generate_rewards(x_cgm_val, x_other_val, y_val)
            rewards_tensor = torch.FloatTensor(rewards).reshape(-1, 1).to(self.device)
            
            # Calcular pérdida Q
            q_values = self.q_network(x_cgm_tensor, x_other_tensor, y_tensor)
            q_loss = self.mse_criterion(q_values, rewards_tensor).item()
            
            # Calcular pérdida de comportamiento
            mean, std = self.behavior_policy(x_cgm_tensor, x_other_tensor)
            behavior_loss = self.nll_criterion(mean, y_tensor.squeeze(), std.pow(2)).item()
        
        return q_loss, behavior_loss
    
    def _bootstrap(self, x_cgm: np.ndarray, x_other: np.ndarray, actions: np.ndarray, 
                 iterations: int = 20):
        """
        Realiza bootstrap para calcular intervalos de confianza.
        
        Parámetros:
        -----------
        x_cgm : np.ndarray
            Datos CGM
        x_other : np.ndarray
            Otras características
        actions : np.ndarray
            Acciones (dosis)
        iterations : int, opcional
            Número de iteraciones bootstrap (default: 20)
        """
        n_samples = len(x_cgm)
        self.bootstrap_estimates = []
        
        for _ in range(iterations):
            # Muestreo con reemplazo
            indices = np.random.choice(n_samples, n_samples, replace=True)
            x_cgm_bootstrap = x_cgm[indices]
            x_other_bootstrap = x_other[indices]
            actions_bootstrap = actions[indices]
            
            # Calcular estimación DR para esta muestra
            dr_estimate = self._compute_dr_estimate(x_cgm_bootstrap, x_other_bootstrap, actions_bootstrap)
            self.bootstrap_estimates.append(dr_estimate)
    
    def _compute_importance_weights(self, x_cgm: np.ndarray, x_other: np.ndarray, 
                                  actions: np.ndarray, policy) -> np.ndarray:
        """
        Calcula los pesos de importancia entre la política de evaluación y de comportamiento.
        
        Parámetros:
        -----------
        x_cgm : np.ndarray
            Datos CGM
        x_other : np.ndarray
            Otras características
        actions : np.ndarray
            Acciones (dosis)
        policy : object
            Política a evaluar
            
        Retorna:
        --------
        np.ndarray
            Pesos de importancia
        """
        self.behavior_policy.eval()
        
        with torch.no_grad():
            # Calcular probabilidades de comportamiento
            x_cgm_tensor = torch.FloatTensor(x_cgm).to(self.device)
            x_other_tensor = torch.FloatTensor(x_other).to(self.device)
            actions_tensor = torch.FloatTensor(actions).reshape(-1, 1).to(self.device)
            
            behavior_log_probs = self.behavior_policy.log_prob(x_cgm_tensor, x_other_tensor, actions_tensor)
            behavior_probs = torch.exp(behavior_log_probs).cpu().numpy()
            
            # Calcular probabilidades de la política de evaluación
            # Usamos predict para obtener acciones de la política de evaluación
            if hasattr(policy, 'predict'):
                eval_actions = policy.predict(x_cgm, x_other)
            else:
                # Manejar caso especial para el ensamble
                eval_actions = np.array([policy.predict(x_cgm[i:i+1], x_other[i:i+1]) 
                                       for i in range(len(x_cgm))])
            
            # Asumir política gaussiana con sigma fijo para simplicidad
            eval_sigma = 0.1
            eval_probs = np.exp(-0.5 * ((eval_actions - actions) / eval_sigma) ** 2) / (eval_sigma * np.sqrt(2 * np.pi))
            
            # Calcular y recortar pesos de importancia
            weights = np.clip(eval_probs / (behavior_probs + 1e-6), 0, CONST_IPS_CLIP)
            
            return weights
    
    def _compute_dr_estimate(self, x_cgm: np.ndarray, x_other: np.ndarray, 
                           actions: np.ndarray, policy=None) -> float:
        """
        Calcula la estimación Doubly Robust para una política.
        
        Parámetros:
        -----------
        x_cgm : np.ndarray
            Datos CGM
        x_other : np.ndarray
            Otras características
        actions : np.ndarray
            Acciones (dosis)
        policy : object, opcional
            Política a evaluar (default: None)
            
        Retorna:
        --------
        float
            Estimación Doubly Robust
        """
        # Si no se proporciona política, crear un estimador directo simple
        if policy is None:
            return self._compute_direct_method_estimate(x_cgm, x_other, actions)
        
        # Calcular estimación de método directo (Q-values)
        dm_estimate = self._compute_direct_method_estimate(x_cgm, x_other, actions)
        
        # Calcular recompensas reales
        rewards = self._generate_rewards(x_cgm, x_other, actions)
        
        # Calcular pesos de importancia
        importance_weights = self._compute_importance_weights(x_cgm, x_other, actions, policy)
        
        # Calcular Q-values para los pares estado-acción observados
        with torch.no_grad():
            x_cgm_tensor = torch.FloatTensor(x_cgm).to(self.device)
            x_other_tensor = torch.FloatTensor(x_other).to(self.device)
            actions_tensor = torch.FloatTensor(actions).reshape(-1, 1).to(self.device)
            
            q_values = self.q_network(x_cgm_tensor, x_other_tensor, actions_tensor).cpu().numpy().flatten()
        
        # Calcular término de corrección
        correction_term = np.mean(importance_weights * (rewards - q_values))
        
        # Estimación Doubly Robust
        dr_estimate = dm_estimate + correction_term
        
        return float(dr_estimate)
    
    def _compute_direct_method_estimate(self, x_cgm: np.ndarray, x_other: np.ndarray, 
                                     actions: np.ndarray) -> float:
        """
        Calcula la estimación por método directo usando la red Q.
        
        Parámetros:
        -----------
        x_cgm : np.ndarray
            Datos CGM
        x_other : np.ndarray
            Otras características
        actions : np.ndarray
            Acciones (dosis)
            
        Retorna:
        --------
        float
            Estimación por método directo
        """
        self.q_network.eval()
        
        with torch.no_grad():
            x_cgm_tensor = torch.FloatTensor(x_cgm).to(self.device)
            x_other_tensor = torch.FloatTensor(x_other).to(self.device)
            actions_tensor = torch.FloatTensor(actions).reshape(-1, 1).to(self.device)
            
            q_values = self.q_network(x_cgm_tensor, x_other_tensor, actions_tensor)
            mean_q_value = q_values.mean().item()
        
        return mean_q_value
    
    def evaluate_policy(self, policy, x_cgm_test: np.ndarray, x_other_test: np.ndarray, 
                       y_test: np.ndarray, simulator=None) -> Dict[str, float]:
        """
        Evalúa una política usando Doubly Robust Estimation.
        
        Parámetros:
        -----------
        policy : ModelWrapper
            Política (modelo) a evaluar
        x_cgm_test : np.ndarray
            Datos CGM de prueba
        x_other_test : np.ndarray
            Otras características de prueba
        y_test : np.ndarray
            Valores objetivo reales
        simulator : Object, opcional
            Simulador (no usado en DR pero requerido por la interfaz)
            
        Retorna:
        --------
        Dict[str, float]
            Métricas de evaluación
        """
        self.q_network.eval()
        self.behavior_policy.eval()
        
        # Calcular estimación DR
        dr_estimate = self._compute_dr_estimate(x_cgm_test, x_other_test, y_test, policy)
        
        # Calcular también estimaciones DM e IS para comparación
        dm_estimate = self._compute_direct_method_estimate(x_cgm_test, x_other_test, y_test)
        
        # Calcular intervalos de confianza a partir de bootstrap
        bootstrap_estimates = np.array(self.bootstrap_estimates)
        confidence_level = CONST_CONFIDENCE_LEVEL
        lower_idx = int((1 - confidence_level) / 2 * len(bootstrap_estimates))
        upper_idx = int((1 + confidence_level) / 2 * len(bootstrap_estimates))
        sorted_estimates = np.sort(bootstrap_estimates)
        confidence_lower = sorted_estimates[lower_idx]
        confidence_upper = sorted_estimates[upper_idx]
        
        return {
            'estimated_value': float(dr_estimate),
            'confidence_lower': float(confidence_lower),
            'confidence_upper': float(confidence_upper),
            'bootstrap_std': float(np.std(bootstrap_estimates)),
            'direct_method_estimate': float(dm_estimate)
        }


def create_dre_evaluator(cgm_input_dim: tuple, other_input_dim: tuple) -> DoublyRobustEstimator:
    """
    Función para crear un evaluador Doubly Robust.
    
    Parámetros:
    -----------
    cgm_input_dim : tuple
        Dimensiones de entrada para datos CGM
    other_input_dim : tuple
        Dimensiones de entrada para otras características
        
    Retorna:
    --------
    DoublyRobustEstimator
        Instancia del evaluador Doubly Robust
    """
    return DoublyRobustEstimator(cgm_input_dim, other_input_dim)