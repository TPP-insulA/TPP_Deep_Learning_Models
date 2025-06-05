import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Dict, Any, Optional, Callable, List, Union

PROJECT_ROOT = os.path.abspath(os.getcwd())
sys.path.append(PROJECT_ROOT) 

from config.models_config import FNN_CONFIG
from custom.DeepLearning.dl_model_wrapper import DLModelWrapper
from custom.DeepLearning.dl_pt import DLModelWrapperPyTorch
from models_old.early_stopping import get_early_stopping_config

# Constantes para cadenas repetidas
CONST_RELU = "relu"
CONST_GELU = "gelu"
CONST_SWISH = "swish"
CONST_SILU = "silu"

def get_activation(x: torch.Tensor, activation_name: str) -> torch.Tensor:
    """
    Aplica la función de activación especificada al tensor de entrada.
    
    Parámetros:
    -----------
    x : torch.Tensor
        Tensor al que aplicar la activación
    activation_name : str
        Nombre de la función de activación
        
    Retorna:
    --------
    torch.Tensor
        Tensor con la activación aplicada
    """
    if activation_name == CONST_RELU:
        return F.relu(x)
    elif activation_name == CONST_GELU:
        return F.gelu(x)
    elif activation_name == CONST_SWISH or activation_name == CONST_SILU:
        return F.silu(x)
    else:
        return F.relu(x)  # Valor por defecto

class ResidualBlock(nn.Module):
    """
    Bloque residual con normalización y dropout.
    
    Parámetros:
    -----------
    in_features : int
        Número de características de entrada
    out_features : int
        Número de características de salida
    dropout_rate : float
        Tasa de dropout para regularización
    use_layer_norm : bool, opcional
        Si usar normalización de capa en lugar de normalización por lotes (default: True)
    activation : str, opcional
        Función de activación a utilizar (default: "gelu")
    """
    
    def __init__(self, 
                 in_features: int, 
                 out_features: int, 
                 dropout_rate: float, 
                 use_layer_norm: bool = True,
                 activation: str = CONST_GELU) -> None:
        super().__init__()
        self.use_layer_norm = use_layer_norm
        self.activation = activation
        
        # Primera capa densa
        self.dense1 = nn.Linear(in_features, out_features)
        
        # Normalización
        if use_layer_norm:
            self.norm1 = nn.LayerNorm(out_features)
            self.norm2 = nn.LayerNorm(out_features)
        else:
            self.norm1 = nn.BatchNorm1d(out_features)
            self.norm2 = nn.BatchNorm1d(out_features)
        
        # Segunda capa densa
        self.dense2 = nn.Linear(out_features, out_features)
        
        # Dropout
        self.dropout1 = nn.Dropout(dropout_rate)
        self.dropout2 = nn.Dropout(dropout_rate)
        
        # Proyección para la conexión residual si es necesario
        self.projection = None
        if in_features != out_features:
            self.projection = nn.Linear(in_features, out_features)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Pasa la entrada a través del bloque residual.
        
        Parámetros:
        -----------
        x : torch.Tensor
            Tensor de entrada
            
        Retorna:
        --------
        torch.Tensor
            Tensor de salida
        """
        # Guardar entrada para la conexión residual
        identity = x
        
        # Primera capa densa con normalización y activación
        x = self.dense1(x)
        
        # Normalización
        if self.use_layer_norm:
            x = self.norm1(x)
        else:
            # Reordenar dimensiones para BatchNorm1d si es necesario
            if x.dim() == 2:
                x = self.norm1(x)
            else:
                batch_size = x.size(0)
                x = x.view(batch_size, -1)
                x = self.norm1(x)
        
        # Activación y dropout
        x = get_activation(x, self.activation)
        x = self.dropout1(x)
        
        # Segunda capa densa con normalización
        x = self.dense2(x)
        
        # Normalización
        if self.use_layer_norm:
            x = self.norm2(x)
        else:
            # Reordenar dimensiones para BatchNorm1d si es necesario
            if x.dim() == 2:
                x = self.norm2(x)
            else:
                batch_size = x.size(0)
                x = x.view(batch_size, -1)
                x = self.norm2(x)
        
        # Proyección para la conexión residual si es necesario
        if self.projection is not None:
            identity = self.projection(identity)
        
        # Suma con la conexión residual
        x = x + identity
        
        # Activación final y dropout
        x = get_activation(x, self.activation)
        x = self.dropout2(x)
        
        return x

class FNNModel(nn.Module):
    """
    Modelo de red neuronal feedforward con bloques residuales.
    
    Parámetros:
    -----------
    cgm_shape : Tuple[int, ...]
        Forma de los datos CGM
    other_features_shape : Tuple[int, ...]
        Forma de otras características
    """
    
    def __init__(self, cgm_shape: Tuple[int, ...], other_features_shape: Tuple[int, ...]) -> None:
        super().__init__()
        
        # Calcular dimensiones de entrada total
        self.cgm_features = np.prod(cgm_shape).item() if cgm_shape else 0
        self.other_features = np.prod(other_features_shape).item() if other_features_shape else 0
        total_input_size = self.cgm_features + self.other_features
        
        # Capas ocultas con bloques residuales
        self.hidden_layers = nn.ModuleList()
        input_size = total_input_size
        
        for i, units in enumerate(FNN_CONFIG['hidden_units']):
            dropout_rate = FNN_CONFIG['dropout_rates'][i] if i < len(FNN_CONFIG['dropout_rates']) else FNN_CONFIG['dropout_rates'][-1]
            self.hidden_layers.append(
                ResidualBlock(
                    in_features=input_size,
                    out_features=units,
                    dropout_rate=dropout_rate,
                    use_layer_norm=FNN_CONFIG['use_layer_norm'],
                    activation=FNN_CONFIG['activation']
                )
            )
            input_size = units
        
        # Capas finales
        self.final_layers = nn.ModuleList()
        for i, units in enumerate(FNN_CONFIG['final_units']):
            self.final_layers.append(nn.Linear(input_size, units))
            self.final_layers.append(nn.LayerNorm(units) if FNN_CONFIG['use_layer_norm'] else nn.BatchNorm1d(units))
            self.final_layers.append(nn.Dropout(FNN_CONFIG['final_dropout_rate']))
            input_size = units
        
        # Capa de salida
        self.output_layer = nn.Linear(input_size, 1)
    
    def forward(self, cgm_input: torch.Tensor, other_input: torch.Tensor) -> torch.Tensor:
        """
        Ejecuta el modelo FNN sobre las entradas.
        
        Parámetros:
        -----------
        cgm_input : torch.Tensor
            Datos de entrada CGM
        other_input : torch.Tensor
            Otras características de entrada
            
        Retorna:
        --------
        torch.Tensor
            Predicciones del modelo
        """
        # Aplanar CGM input
        batch_size = cgm_input.shape[0]
        cgm_flat = cgm_input.view(batch_size, -1)
        
        # Concatenar con otras características
        if self.other_features > 0:
            x = torch.cat([cgm_flat, other_input], dim=1)
        else:
            x = cgm_flat
        
        # Pasar por capas ocultas
        for layer in self.hidden_layers:
            x = layer(x)
        
        # Pasar por capas finales
        for i, layer in enumerate(self.final_layers):
            x = layer(x)
            # Activación solo después de la capa lineal, no después de la normalización o dropout
            if i % 3 == 0:  # Es una capa lineal
                x = get_activation(x, FNN_CONFIG['activation'])
        
        # Capa de salida
        return self.output_layer(x)

def create_fnn_model(cgm_shape: Tuple[int, ...], other_features_shape: Tuple[int, ...]) -> nn.Module:
    """
    Crea un modelo FNN para regresión.
    
    Parámetros:
    -----------
    cgm_shape : Tuple[int, ...]
        Forma de los datos CGM (pasos_temporales, características)
    other_features_shape : Tuple[int, ...]
        Forma de otras características (características)
        
    Retorna:
    --------
    nn.Module
        Modelo FNN inicializado
    """
    return FNNModel(cgm_shape, other_features_shape)

def create_fnn_model_wrapper(cgm_shape: Tuple[int, ...], other_features_shape: Tuple[int, ...]) -> DLModelWrapper:
    """
    Crea un modelo FNN envuelto en DLModelWrapperPyTorch.
    
    Parámetros:
    -----------
    cgm_shape : Tuple[int, ...]
        Forma de los datos CGM (pasos_temporales, características)
    other_features_shape : Tuple[int, ...]
        Forma de otras características (características)
        
    Retorna:
    --------
    DLModelWrapper
        Modelo FNN envuelto en DLModelWrapper para compatibilidad con el sistema
    """
    # Función creadora del modelo
    model_creator_fn = lambda: create_fnn_model(cgm_shape, other_features_shape)
    
    # Crear wrapper
    model_wrapper = DLModelWrapperPyTorch(model_creator=model_creator_fn)
    
    # Configurar early stopping
    es_patience, es_min_delta, es_restore_best = get_early_stopping_config()
    model_wrapper.add_early_stopping(
        patience=es_patience,
        min_delta=es_min_delta,
        restore_best_weights=es_restore_best
    )
    
    return model_wrapper

def model_creator() -> Callable[[Tuple[int, ...], Tuple[int, ...]], DLModelWrapper]:
    """
    Retorna una función para crear un modelo FNN compatible con la API del sistema.
    
    Retorna:
    --------
    Callable[[Tuple[int, ...], Tuple[int, ...]], DLModelWrapper]
        Función para crear el modelo con las formas de entrada especificadas
    """
    return create_fnn_model_wrapper