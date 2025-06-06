import torch
import torch.nn as nn
import numpy as np
from typing import Dict, Tuple, List, Optional

class BaseDRLModel(nn.Module):
    """
    Clase base para modelos de aprendizaje por refuerzo profundo.
    
    Parámetros:
    -----------
    state_dim : Tuple[int, ...]
        Dimensiones del estado (CGM + otras características)
    action_dim : int
        Dimensiones de la acción (dosis de insulina)
    hidden_dim : int, opcional
        Dimensiones de las capas ocultas (default: 256)
    lr : float, opcional
        Tasa de aprendizaje (default: 3e-4)
    """
    
    def __init__(self, cgm_input_dim: tuple, other_input_dim: tuple, 
                 action_dim: int = 1, hidden_dim: int = 256, lr: float = 3e-4):
        super().__init__()
        self.cgm_input_dim = cgm_input_dim
        self.other_input_dim = other_input_dim
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
        self.lr = lr
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Encoders para diferentes tipos de entrada
        self.cgm_encoder = self._build_cgm_encoder()
        self.other_encoder = self._build_other_encoder()
        self.combined_layer = self._build_combined_layer()
        
        # Buffer de experiencia
        self.buffer = self._initialize_buffer(10000)  # Tamaño por defecto
        
    def _build_cgm_encoder(self) -> nn.Module:
        """
        Construye el encoder para datos CGM.
        """
        return nn.Sequential(
            nn.Linear(self.cgm_input_dim[0], 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU()
        )
    
    def _build_other_encoder(self) -> nn.Module:
        """
        Construye el encoder para otras características.
        """
        return nn.Sequential(
            nn.Linear(self.other_input_dim[0], 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU()
        )
    
    def _build_combined_layer(self) -> nn.Module:
        """
        Construye la capa que combina las características.
        """
        return nn.Sequential(
            nn.Linear(64 + 32, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU()
        )
    
    def _initialize_buffer(self, capacity: int) -> Dict:
        """
        Inicializa el buffer de experiencia.
        """
        return {
            'states_cgm': [],
            'states_other': [],
            'actions': [],
            'rewards': [],
            'next_states_cgm': [],
            'next_states_other': [],
            'dones': [],
            'capacity': capacity,
            'position': 0,
            'size': 0
        }
    
    def forward(self, x_cgm: torch.Tensor, x_other: torch.Tensor) -> torch.Tensor:
        """
        Paso hacia adelante del modelo.
        """
        cgm_features = self.cgm_encoder(x_cgm)
        other_features = self.other_encoder(x_other)
        combined = torch.cat([cgm_features, other_features], dim=1)
        features = self.combined_layer(combined)
        return features
    
    def predict_with_context(self, x_cgm: np.ndarray, x_other: np.ndarray, **context) -> float:
        """
        Predice dosis de insulina con contexto.
        """
        # Verificar parámetros obligatorios
        if 'glucose' not in context or 'carb_intake' not in context:
            raise ValueError("Los parámetros 'glucose' y 'carb_intake' son obligatorios")
        
        # Convertir a tensores
        x_cgm_t = torch.FloatTensor(x_cgm).to(self.device)
        x_other_t = torch.FloatTensor(x_other).to(self.device)
        
        # Crear tensor de contexto
        glucose = float(context['glucose'])
        carb_intake = float(context['carb_intake'])
        sleep_quality = float(context.get('sleep_quality', 5.0))
        work_intensity = float(context.get('work_intensity', 0.0))
        exercise_intensity = float(context.get('exercise_intensity', 0.0))
        
        # Asegurar dimensiones correctas
        if len(x_cgm_t.shape) == 2:
            x_cgm_t = x_cgm_t.unsqueeze(0)
        if len(x_other_t.shape) == 1:
            x_other_t = x_other_t.unsqueeze(0)
        
        # Implementación específica por algoritmo
        # (Sobrescrita por subclases si es necesario)
        with torch.no_grad():
            action = self(x_cgm_t, x_other_t)
            dose = action.cpu().numpy().flatten()[0]
        
        # Aplicar límites de seguridad
        dose = max(0.0, min(dose, 20.0))
        
        return dose
    
    def update(self, batch: Dict) -> Dict[str, float]:
        """
        Actualiza los parámetros del modelo.
        """
        raise NotImplementedError("Debe ser implementado por las subclases")