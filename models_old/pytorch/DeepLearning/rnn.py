import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Dict, Any, Optional, Callable, List, Union

PROJECT_ROOT = os.path.abspath(os.getcwd())
sys.path.append(PROJECT_ROOT) 

from config.models_config_old import RNN_CONFIG
from custom.DeepLearning.dl_model_wrapper import DLModelWrapper
from custom.DeepLearning.dl_pt import DLModelWrapperPyTorch
from models_old.early_stopping import get_early_stopping_config

# Constantes para cadenas repetidas
CONST_RELU = "relu"
CONST_TANH = "tanh"
CONST_SIGMOID = "sigmoid"
CONST_SWISH = "swish"

def get_activation_fn(activation_name: str) -> Callable:
    """
    Obtiene la función de activación según su nombre.
    
    Parámetros:
    -----------
    activation_name : str
        Nombre de la función de activación
        
    Retorna:
    --------
    Callable
        Función de activación
    """
    if activation_name == CONST_RELU:
        return F.relu
    elif activation_name == CONST_TANH:
        return torch.tanh
    elif activation_name == CONST_SIGMOID:
        return torch.sigmoid
    elif activation_name == CONST_SWISH:
        return F.silu
    else:
        return F.relu  # Por defecto


class SimpleRNN(nn.Module):
    """
    Implementación simple de RNN con soporte para bidireccionalidad.
    
    Parámetros:
    -----------
    input_size : int
        Tamaño de la entrada
    hidden_size : int
        Tamaño del estado oculto
    bidirectional : bool, opcional
        Si es True, procesa la secuencia en ambas direcciones (default: False)
    return_sequences : bool, opcional
        Si es True, devuelve secuencias completas (default: False)
    dropout : float, opcional
        Tasa de dropout (default: 0.0)
    activation : str, opcional
        Función de activación (default: "tanh")
    """
    
    def __init__(self, 
                input_size: int, 
                hidden_size: int, 
                bidirectional: bool = False,
                return_sequences: bool = False,
                dropout: float = 0.0,
                activation: str = CONST_TANH) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.bidirectional = bidirectional
        self.return_sequences = return_sequences
        self.activation_name = activation
        
        # Determinar no-linealidad para RNN de PyTorch
        nonlinearity = 'tanh' if activation == CONST_TANH else 'relu'
        
        # Redes recurrentes
        self.rnn_forward = nn.RNN(
            input_size=input_size,
            hidden_size=hidden_size,
            batch_first=True,
            dropout=dropout,
            nonlinearity=nonlinearity
        )
        
        if bidirectional:
            self.rnn_backward = nn.RNN(
                input_size=input_size,
                hidden_size=hidden_size,
                batch_first=True,
                dropout=dropout,
                nonlinearity=nonlinearity
            )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Procesa una secuencia completa.
        
        Parámetros:
        -----------
        x : torch.Tensor
            Tensor de entrada [batch, tiempo, características]
            
        Retorna:
        --------
        torch.Tensor
            Salidas de la capa RNN
        """
        # Procesar RNN hacia adelante
        forward_out, _ = self.rnn_forward(x)
        
        if self.bidirectional:
            # Invertir secuencia para RNN hacia atrás
            x_reversed = torch.flip(x, [1])
            backward_out, _ = self.rnn_backward(x_reversed)
            # Revertir la salida para alinearla con la dirección original
            backward_out = torch.flip(backward_out, [1])
            
            # Concatenar salidas
            output = torch.cat([forward_out, backward_out], dim=-1)
        else:
            output = forward_out
        
        # Devolver toda la secuencia o solo el último paso
        if self.return_sequences:
            return output
        else:
            return output[:, -1, :]


class TimeDistributed(nn.Module):
    """
    Aplica un módulo a cada paso temporal de forma independiente.
    
    Parámetros:
    -----------
    module : nn.Module
        Módulo a aplicar a cada paso temporal
    """
    
    def __init__(self, module: nn.Module) -> None:
        super().__init__()
        self.module = module
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Aplica el módulo a cada paso temporal.
        
        Parámetros:
        -----------
        x : torch.Tensor
            Tensor de entrada [batch, tiempo, características]
            
        Retorna:
        --------
        torch.Tensor
            Tensor procesado [batch, tiempo, características_salida]
        """
        batch_size, time_steps, features = x.size()
        reshaped_x = x.reshape(batch_size * time_steps, features)
        y = self.module(reshaped_x)
        return y.reshape(batch_size, time_steps, -1)


class RNNModel(nn.Module):
    """
    Modelo RNN completo con arquitectura personalizable.
    
    Parámetros:
    -----------
    cgm_shape : Tuple[int, ...]
        Forma de los datos CGM
    other_features_shape : Tuple[int, ...]
        Forma de otras características
    config : Dict, opcional
        Configuración del modelo (default: None, usa RNN_CONFIG)
    """
    
    def __init__(self, 
                cgm_shape: Tuple[int, ...], 
                other_features_shape: Tuple[int, ...],
                config: Optional[Dict] = None) -> None:
        super().__init__()
        self.config = config if config is not None else RNN_CONFIG
        
        # Inicializar las dimensiones del modelo
        self._init_input_dimensions(cgm_shape, other_features_shape)
        
        # Configurar el procesamiento temporal distribuido
        input_size = self._setup_time_distributed()
        
        # Configurar capas RNN
        last_rnn_dim = self._setup_rnn_layers(input_size)
        
        # Configurar capas de salida
        self._setup_output_layers(last_rnn_dim)
    
    def _init_input_dimensions(self, cgm_shape: Tuple[int, ...], other_features_shape: Tuple[int, ...]) -> None:
        """Inicializa las dimensiones de las entradas del modelo."""
        if len(cgm_shape) > 1:
            self.cgm_time_steps, self.cgm_features = cgm_shape
        else:
            self.cgm_time_steps, self.cgm_features = 1, cgm_shape[0]
            
        if len(other_features_shape) > 0:
            self.other_features = other_features_shape[0]
        else:
            self.other_features = 0
    
    def _setup_time_distributed(self) -> int:
        """Configura las capas de procesamiento temporal distribuido si están habilitadas."""
        if self.config['use_time_distributed']:
            self.time_distributed = TimeDistributed(nn.Linear(self.cgm_features, 32))
            self.layer_norm_td = nn.LayerNorm(32, eps=self.config['epsilon'])
            return 32
        else:
            self.time_distributed = None
            return self.cgm_features
    
    def _setup_rnn_layers(self, input_size: int) -> int:
        """Configura las capas RNN y retorna la dimensión de la última capa."""
        self.rnn_layers = nn.ModuleList()
        hidden_units = self.config['hidden_units']
        
        for i, units in enumerate(hidden_units):
            is_last_layer = i == len(hidden_units) - 1
            layer_input_size = self._get_layer_input_size(i, input_size)
            
            # Añadir capa RNN
            self._add_rnn_layer(layer_input_size, units, is_last_layer)
        
        # Calcular dimensión de la última capa RNN
        return hidden_units[-1] * (2 if self.config['bidirectional'] else 1)
    
    def _get_layer_input_size(self, layer_idx: int, first_layer_input_size: int) -> int:
        """Calcula el tamaño de entrada para una capa RNN específica."""
        if layer_idx == 0:
            return first_layer_input_size
        
        prev_units = self.config['hidden_units'][layer_idx - 1]
        factor = 2 if self.config['bidirectional'] else 1
        return prev_units * factor
    
    def _add_rnn_layer(self, input_size: int, units: int, is_last_layer: bool) -> None:
        """Añade una capa RNN y opcionalmente normalización y dropout."""
        # Crear y añadir capa RNN
        rnn_layer = SimpleRNN(
            input_size=input_size,
            hidden_size=units,
            bidirectional=self.config['bidirectional'],
            return_sequences=not is_last_layer,
            dropout=self.config['dropout_rate'],
            activation=self.config['activation']
        )
        self.rnn_layers.append(rnn_layer)
        
        # Añadir normalización para capas intermedias si no es la última
        if not is_last_layer:
            norm_size = units * (2 if self.config['bidirectional'] else 1)
            self.rnn_layers.append(nn.LayerNorm(norm_size, eps=self.config['epsilon']))
            self.rnn_layers.append(nn.Dropout(p=self.config['dropout_rate']))
    
    def _setup_output_layers(self, last_rnn_dim: int) -> None:
        """Configura las capas densas finales y la capa de salida."""
        # Capas densas finales
        self.dense1 = nn.Linear(last_rnn_dim + self.other_features, 64)
        self.layer_norm1 = nn.LayerNorm(64, eps=self.config['epsilon'])
        self.dropout1 = nn.Dropout(p=self.config['dropout_rate'])
        
        self.dense2 = nn.Linear(64, 32)
        
        # Capa de salida
        self.output_layer = nn.Linear(32, 1)
    
    def forward(self, cgm_input: torch.Tensor, other_input: torch.Tensor) -> torch.Tensor:
        """
        Ejecuta el modelo RNN sobre las entradas.
        
        Parámetros:
        -----------
        cgm_input : torch.Tensor
            Datos de entrada CGM [batch, tiempo, características]
        other_input : torch.Tensor
            Otras características [batch, características]
            
        Retorna:
        --------
        torch.Tensor
            Predicciones del modelo
        """
        # Asegurar que la entrada tenga la forma correcta
        if len(cgm_input.shape) < 3:
            cgm_input = cgm_input.unsqueeze(0) if cgm_input.dim() == 1 else cgm_input.unsqueeze(1)
        
        # Aplicar TimeDistributed si está configurado
        x = cgm_input
        if self.time_distributed is not None:
            x = self.time_distributed(x)
            x = F.relu(x)
            x = self.layer_norm_td(x)
        
        # Aplicar capas RNN
        for i, layer in enumerate(self.rnn_layers):
            x = layer(x)
        
        # Combinar con otras características
        x = torch.cat([x, other_input], dim=1)
        
        # Capas densas finales
        x = self.dense1(x)
        x = F.relu(x)
        x = self.layer_norm1(x)
        x = self.dropout1(x)
        
        x = self.dense2(x)
        x = F.relu(x)
        
        # Capa de salida
        x = self.output_layer(x)
        
        return x


def create_rnn_model(cgm_shape: Tuple[int, ...], other_features_shape: Tuple[int, ...]) -> nn.Module:
    """
    Crea un modelo RNN para regresión.
    
    Parámetros:
    -----------
    cgm_shape : Tuple[int, ...]
        Forma de los datos CGM (pasos_temporales, características)
    other_features_shape : Tuple[int, ...]
        Forma de otras características (características)
        
    Retorna:
    --------
    nn.Module
        Modelo RNN inicializado
    """
    return RNNModel(cgm_shape, other_features_shape, RNN_CONFIG)


def create_rnn_model_wrapper(cgm_shape: Tuple[int, ...], other_features_shape: Tuple[int, ...]) -> DLModelWrapper:
    """
    Crea un modelo RNN envuelto en DLModelWrapperPyTorch.
    
    Parámetros:
    -----------
    cgm_shape : Tuple[int, ...]
        Forma de los datos CGM (pasos_temporales, características)
    other_features_shape : Tuple[int, ...]
        Forma de otras características (características)
        
    Retorna:
    --------
    DLModelWrapper
        Modelo RNN envuelto en DLModelWrapper para compatibilidad con el sistema
    """
    # Función creadora del modelo
    model_creator_fn = lambda: create_rnn_model(cgm_shape, other_features_shape)
    
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
    Retorna una función para crear un modelo RNN compatible con la API del sistema.
    
    Retorna:
    --------
    Callable[[Tuple[int, ...], Tuple[int, ...]], DLModelWrapper]
        Función para crear el modelo con las formas de entrada especificadas
    """
    return create_rnn_model_wrapper