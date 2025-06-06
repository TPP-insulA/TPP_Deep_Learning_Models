import torch
import torch.nn as nn
import numpy as np

class QNetwork(nn.Module):
    """
    Red neuronal para aproximar la función Q para FQE.
    
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
        
        # Encoder para acciones
        self.action_encoder = nn.Sequential(
            nn.Linear(action_dim, hidden_dim // 4),
            nn.ReLU()
        )
        
        # Capa combinada
        combined_dim = hidden_dim // 2 + hidden_dim // 2 + hidden_dim // 4
        self.combined_layer = nn.Sequential(
            nn.Linear(combined_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1)  # Valor Q
        )
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.to(self.device)
    
    def forward(self, x_cgm: torch.Tensor, x_other: torch.Tensor, 
               action: torch.Tensor) -> torch.Tensor:
        """
        Paso hacia adelante de la red.
        
        Parámetros:
        -----------
        x_cgm : torch.Tensor
            Datos CGM
        x_other : torch.Tensor
            Otras características
        action : torch.Tensor
            Acción (dosis de insulina)
            
        Retorna:
        --------
        torch.Tensor
            Valor Q estimado
        """
        # Aplanar entradas si es necesario
        if len(x_cgm.shape) > 2:
            x_cgm = x_cgm.reshape(x_cgm.shape[0], -1)
        if len(x_other.shape) > 2:
            x_other = x_other.reshape(x_other.shape[0], -1)
        
        # Codificar cada componente
        cgm_features = self.cgm_encoder(x_cgm)
        other_features = self.other_encoder(x_other)
        action_features = self.action_encoder(action)
        
        # Combinar características
        combined = torch.cat([cgm_features, other_features, action_features], dim=1)
        
        # Obtener valor Q
        q_value = self.combined_layer(combined)
        
        return q_value
