"""
Fitted Q Evaluation (FQE)

1. Q-Function Learning: FQE ajusta iterativamente una función Q para estimar el valor de las acciones basándose en datos históricos de interacciones con el entorno.
    - Por cada par estado-acción (s,a) en el dataset, se computa un valor objetivo Q(s,a) como la recompensa inmediata más el valor descontado de la acción futura.
    objetivo = recompensa + gamma * Q(s', pi(s'))
    donde pi es la política que se está evaluando y s' es el siguiente estado.
    - Ajusta a un modelo de regresión para predecir estos objetivos a partir de los pares estado-acción.
    - Se repite hasta lograr una convergencia.
2. Policy Evaluation: Una vez que la función Q está "aprendida", el valor de la política se estima como el valor esperado de la función Q bajo la política.
    - Se evalúa la política al calcular el valor esperado de la función Q para las acciones tomadas por la política en los estados observados.
3. Bootstrap para Intervalos de Confianza: FQE utiliza técnicas de bootstrap para calcular intervalos de confianza sobre las estimaciones de valor Q.
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
    CONST_CONFIDENCE_LEVEL, SEVERE_HYPOGLYCEMIA_THRESHOLD,
    HYPOGLYCEMIA_THRESHOLD, HYPERGLYCEMIA_THRESHOLD, SEVERE_HYPERGLYCEMIA_THRESHOLD
)
from config.models_config import EARLY_STOPPING_POLICY
from custom.printer import print_warning
from validation.networks.QNetwork import QNetwork

class FittedQEvaluation:
    """
    Evaluador de políticas usando Fitted Q Evaluation (FQE).
    
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
        
        # Inicializar red Q
        self.q_network = QNetwork(
            cgm_input_dim=cgm_input_dim,
            other_input_dim=other_input_dim,
            action_dim=1,  # Dosis de insulina
            hidden_dim=hidden_dim
        )
        
        # Inicializar red Q objetivo (para estabilidad)
        self.target_q_network = QNetwork(
            cgm_input_dim=cgm_input_dim,
            other_input_dim=other_input_dim,
            action_dim=1,
            hidden_dim=hidden_dim
        )
        self.target_q_network.load_state_dict(self.q_network.state_dict())
        
        # Optimizador
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=self.lr)
        
        # Criterio de pérdida
        self.criterion = nn.MSELoss()
        
        # Dispositivo
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.q_network.to(self.device)
        self.target_q_network.to(self.device)
        
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
        # Inicializar recompensas
        rewards = np.zeros_like(actions, dtype=np.float32)
        
        # Extraer valores de glucosa actuales de manera segura según la forma de los datos
        if len(x_cgm.shape) == 3:  # (samples, time_steps, features)
            current_glucose = x_cgm[:, -1, 0]  # Último paso de tiempo, primera característica
        elif len(x_cgm.shape) == 2:  # (samples, time_steps) o (samples, features)
            if x_cgm.shape[0] == actions.shape[0]:  # Verificar si las muestras coinciden
                if x_cgm.shape[1] > 1:
                    current_glucose = x_cgm[:, -1]  # Último paso de tiempo
                else:
                    current_glucose = x_cgm[:, 0]  # Primera característica
            else:
                raise ValueError(f"Número de muestras no coincide: {x_cgm.shape[0]} vs {actions.shape[0]}")
        else:
            raise ValueError(f"Forma inesperada para x_cgm: {x_cgm.shape}")
        
        # Verificar que las dimensiones coincidan
        if len(current_glucose) != len(rewards):
            raise ValueError(f"Dimensiones no coinciden: {len(current_glucose)} valores de glucosa pero {len(rewards)} recompensas")
        
        # Hipoglucemia severa (<54 mg/dL) - penalización muy severa
        severe_hypo = current_glucose < SEVERE_HYPOGLYCEMIA_THRESHOLD
        rewards[severe_hypo] = -4.0
        
        # Hipoglucemia moderada (54-70 mg/dL) - penalización severa
        hypo = np.logical_and(
            current_glucose >= SEVERE_HYPOGLYCEMIA_THRESHOLD,
            current_glucose < HYPOGLYCEMIA_THRESHOLD
        )
        rewards[hypo] = -2.0
        
        # En rango (70-180 mg/dL) - recompensa positiva
        in_range = np.logical_and(
            current_glucose >= HYPOGLYCEMIA_THRESHOLD, 
            current_glucose <= HYPERGLYCEMIA_THRESHOLD
        )
        rewards[in_range] = 1.0
        
        # Hiperglucemia moderada (180-250 mg/dL) - penalización moderada
        hyper = np.logical_and(
            current_glucose > HYPERGLYCEMIA_THRESHOLD,
            current_glucose <= SEVERE_HYPERGLYCEMIA_THRESHOLD
        )
        rewards[hyper] = -1.0
        
        # Hiperglucemia severa (>250 mg/dL) - penalización severa
        severe_hyper = current_glucose > SEVERE_HYPERGLYCEMIA_THRESHOLD
        rewards[severe_hyper] = -3.0
        
        return rewards
    
    def fit(self, x_cgm: np.ndarray, x_other: np.ndarray, y_actions: np.ndarray,
           validation_data: Optional[Tuple] = None,
           batch_size: int = CONST_DEFAULT_BATCH_SIZE,
           epochs: int = CONST_DEFAULT_EPOCHS,
           bootstrap_iterations: int = 20,
           patience: int = EARLY_STOPPING_POLICY['early_stopping_patience'],
           min_delta: float = EARLY_STOPPING_POLICY['early_stopping_min_delta'],
           restore_best_weights: bool = EARLY_STOPPING_POLICY['early_stopping_restore_best_weights']) -> Dict[str, List[float]]:
        """
        Entrena el modelo FQE con los datos proporcionados.
        
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
        patience : int, opcional
            Número de épocas sin mejora antes de detener el entrenamiento
        min_delta : float, opcional
            Mínima mejora para considerar progreso
        restore_best_weights : bool, opcional
            Si restaurar los mejores pesos al finalizar
        
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
            'loss': [],
            'val_loss': []
        }
        
        # Variables para early stopping
        best_val_loss = float('inf')
        best_weights = None
        early_stop_counter = 0
        
        # Crear barras de progreso
        epoch_progress = tqdm(range(epochs), desc="Entrenamiento", position=0)
        
        # Línea de métricas que se actualizará dinámicamente
        metrics_line = ""
        
        for epoch in epoch_progress:
            epoch_loss = 0.0
            self.q_network.train()
            
            # Barra de progreso para los batches (no mostrará progreso individual)
            batch_progress = tqdm(dataloader, desc=f"Época {epoch+1}/{epochs}", 
                           leave=False, position=1)
            
            for batch_cgm, batch_other, batch_actions, batch_rewards in batch_progress:
                batch_cgm = batch_cgm.to(self.device)
                batch_other = batch_other.to(self.device)
                batch_actions = batch_actions.to(self.device)
                batch_rewards = batch_rewards.to(self.device)
                
                # Forward pass
                q_values = self.q_network(batch_cgm, batch_other, batch_actions)
                
                # Para el estado siguiente, usamos la misma acción (simplificación para FQE)
                with torch.no_grad():
                    target_q_values = batch_rewards + self.gamma * self.target_q_network(
                        batch_cgm, batch_other, batch_actions
                    )
                
                # Calcular pérdida
                loss = self.criterion(q_values, target_q_values)
                
                # Backward pass y optimización
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                
                epoch_loss += loss.item()
            
            # Actualizar red objetivo periódicamente
            if epoch % 5 == 0:
                self.target_q_network.load_state_dict(self.q_network.state_dict())
            
            # Registrar pérdida de entrenamiento
            avg_loss = epoch_loss / len(dataloader)
            history['loss'].append(avg_loss)
            
            # Validación si hay datos disponibles
            val_loss = None
            if validation_data:
                val_loss = self._validate(validation_data[0][0], validation_data[0][1], validation_data[1])
                history['val_loss'].append(val_loss)
                metrics_line = f"Pérdida: {avg_loss:.4f} - Pérdida val: {val_loss:.4f}"
                
                # Verificación para early stopping con datos de validación
                if val_loss < best_val_loss - min_delta:
                    best_val_loss = val_loss
                    if restore_best_weights:
                        best_weights = {k: v.cpu().clone() for k, v in self.q_network.state_dict().items()}
                    early_stop_counter = 0
                else:
                    early_stop_counter += 1
            else:
                # Sin datos de validación, usar pérdida de entrenamiento
                metrics_line = f"Pérdida: {avg_loss:.4f}"
                
                # Verificación para early stopping con pérdida de entrenamiento
                if avg_loss < best_val_loss - min_delta:
                    best_val_loss = avg_loss
                    if restore_best_weights:
                        best_weights = {k: v.cpu().clone() for k, v in self.q_network.state_dict().items()}
                    early_stop_counter = 0
                else:
                    early_stop_counter += 1
            
            # Actualizar descripción de la barra de progreso con las métricas
            epoch_progress.set_description(f"Entrenamiento: {metrics_line}")
            
            # Verificar si se debe activar early stopping
            if early_stop_counter >= patience:
                epoch_progress.write(f"Early stopping en época {epoch+1}")
                break
        
        # Restaurar los mejores pesos si corresponde
        if restore_best_weights and best_weights is not None:
            self.q_network.load_state_dict(best_weights)
            if val_loss is not None:
                epoch_progress.write(f"Restaurados mejores pesos con pérdida de validación: {best_val_loss:.4f}")
            else:
                epoch_progress.write(f"Restaurados mejores pesos con pérdida de entrenamiento: {best_val_loss:.4f}")
        
        # Realizar bootstrap para intervalos de confianza
        self._bootstrap(x_cgm, x_other, y_actions, bootstrap_iterations)
        
        return history
    
    def _validate(self, x_cgm_val: np.ndarray, x_other_val: np.ndarray, 
                y_val: np.ndarray) -> float:
        """
        Valida el modelo con datos de validación.
        
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
        float
            Pérdida de validación
        """
        self.q_network.eval()
        
        with torch.no_grad():
            # Convertir a tensores
            x_cgm_tensor = torch.FloatTensor(x_cgm_val).to(self.device)
            x_other_tensor = torch.FloatTensor(x_other_val).to(self.device)
            y_tensor = torch.FloatTensor(y_val).reshape(-1, 1).to(self.device)
            
            # Generar recompensas para validación
            rewards = self._generate_rewards(x_cgm_val, x_other_val, y_val)
            rewards_tensor = torch.FloatTensor(rewards).reshape(-1, 1).to(self.device)
            
            # Calcular Q-values
            q_values = self.q_network(x_cgm_tensor, x_other_tensor, y_tensor)
            target_q_values = rewards_tensor + self.gamma * self.target_q_network(
                x_cgm_tensor, x_other_tensor, y_tensor
            )
            
            # Calcular pérdida
            val_loss = self.criterion(q_values, target_q_values).item()
        
        return val_loss
    
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
            
            # Evaluar en esta muestra
            with torch.no_grad():
                x_cgm_tensor = torch.FloatTensor(x_cgm_bootstrap).to(self.device)
                x_other_tensor = torch.FloatTensor(x_other_bootstrap).to(self.device)
                actions_tensor = torch.FloatTensor(actions_bootstrap).reshape(-1, 1).to(self.device)
                
                q_values = self.q_network(x_cgm_tensor, x_other_tensor, actions_tensor)
                mean_q_value = q_values.mean().item()
                
                self.bootstrap_estimates.append(mean_q_value)
    
    def evaluate_policy(self, policy, x_cgm_test: np.ndarray, x_other_test: np.ndarray, 
                       y_test: np.ndarray, simulator=None) -> Dict[str, float]:
        """
        Evalúa una política (modelo) usando FQE.
        
        Parámetros:
        -----------
        policy : ModelWrapper or Dict
            Política (modelo) a evaluar o diccionario con historial
        x_cgm_test : np.ndarray
            Datos CGM de prueba
        x_other_test : np.ndarray
            Otras características de prueba
        y_test : np.ndarray
            Valores objetivo reales
        simulator : Object, opcional
            Simulador (no usado en FQE pero requerido por la interfaz)
            
        Retorna:
        --------
        Dict[str, float]
            Métricas de evaluación
        """
        self.q_network.eval()
        
        # Manejar el caso donde policy es un diccionario (historial) en lugar de un modelo
        if isinstance(policy, dict):
            # Si se pasó un diccionario de historial, usamos directamente las predicciones
            if 'predictions' in policy:
                actions = policy['predictions']
            else:
                # Si no hay predicciones, no podemos evaluar
                print_warning("No se pueden evaluar políticas sin método predict o predicciones")
                return {
                    'estimated_value': 0.0,
                    'confidence_lower': 0.0,
                    'confidence_upper': 0.0,
                    'bootstrap_std': 0.0
                }
        else:
            # Obtener acciones de la política (modelo)
            # Primero intentar con predict_with_context si está disponible
            if hasattr(policy, 'predict_with_context'):
                # Crear un array para almacenar las predicciones
                actions = np.zeros(len(x_cgm_test))
                
                # Extraer índice de carbohidratos (asumimos que está en la primera posición)
                carb_intake_idx = 0
                
                # Hacer predicciones individualmente con contexto
                for i in range(len(x_cgm_test)):
                    carb_intake = float(x_other_test[i, carb_intake_idx])
                    actions[i] = policy.predict_with_context(
                        x_cgm_test[i:i+1], 
                        x_other_test[i:i+1],
                        carb_intake=carb_intake
                    )
            elif hasattr(policy, 'predict'):
                actions = policy.predict(x_cgm_test, x_other_test)
            else:
                # Manejar el caso del wrapper del ensamble
                actions = np.array([policy.predict(x_cgm_test[i:i+1], x_other_test[i:i+1]) 
                                   for i in range(len(x_cgm_test))])
        
        # Convertir a tensores
        x_cgm_tensor = torch.FloatTensor(x_cgm_test).to(self.device)
        x_other_tensor = torch.FloatTensor(x_other_test).to(self.device)
        actions_tensor = torch.FloatTensor(actions).reshape(-1, 1).to(self.device)
        
        # Calcular valores Q
        with torch.no_grad():
            q_values = self.q_network(x_cgm_tensor, x_other_tensor, actions_tensor)
            estimated_value = q_values.mean().item()
        
        # Calcular intervalos de confianza
        bootstrap_estimates = np.array(self.bootstrap_estimates)
        confidence_level = CONST_CONFIDENCE_LEVEL
        lower_idx = int((1 - confidence_level) / 2 * len(bootstrap_estimates))
        upper_idx = int((1 + confidence_level) / 2 * len(bootstrap_estimates))
        sorted_estimates = np.sort(bootstrap_estimates)
        confidence_lower = sorted_estimates[lower_idx]
        confidence_upper = sorted_estimates[upper_idx]
        
        return {
            'estimated_value': float(estimated_value),
            'confidence_lower': float(confidence_lower),
            'confidence_upper': float(confidence_upper),
            'bootstrap_std': float(np.std(bootstrap_estimates))
        }


def create_fqe_evaluator(cgm_input_dim: tuple, other_input_dim: tuple):
    """
    Función para crear un evaluador FQE.
    
    Parámetros:
    -----------
    cgm_input_dim : tuple
        Dimensiones de entrada para datos CGM
    other_input_dim : tuple
        Dimensiones de entrada para otras características
        
    Retorna:
    --------
    FittedQEvaluation
        Instancia del evaluador FQE
    """
    return FittedQEvaluation(cgm_input_dim, other_input_dim)