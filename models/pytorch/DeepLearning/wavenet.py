import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Callable, Optional, Any, Union

PROJECT_ROOT = os.path.abspath(os.getcwd())
sys.path.append(PROJECT_ROOT)

from config.models_config import WAVENET_CONFIG
from custom.dl_model_wrapper import DLModelWrapper, DLModelWrapperPyTorch
from models.early_stopping import get_early_stopping_config

# Constantes para cadenas repetidas
CONST_RELU = "relu"
CONST_GELU = "gelu"
CONST_ELU = "elu"
CONST_TANH = "tanh" 
CONST_SIGMOID = "sigmoid"
CONST_SAME = "same"
CONST_CAUSAL = "causal"
CONST_VALID = "valid"
CONST_INPUT_CGM = "cgm_input"
CONST_INPUT_OTHER = "other_input"
CONST_OUTPUT = "output"

def get_activation_fn(activation_name: str) -> Callable:
    """
    Obtiene la función de activación correspondiente a un nombre.
    
    Parámetros:
    -----------
    activation_name : str
        Nombre de la función de activación ('relu', 'gelu', 'elu', etc.)
        
    Retorna:
    --------
    Callable
        Función de activación de PyTorch
    """
    if activation_name == CONST_RELU:
        return F.relu
    elif activation_name == CONST_GELU:
        return F.gelu
    elif activation_name == CONST_ELU:
        return F.elu
    elif activation_name == CONST_TANH:
        return torch.tanh
    elif activation_name == CONST_SIGMOID:
        return torch.sigmoid
    else:
        # Por defecto, devolver ReLU
        return F.relu

def causal_padding(x: torch.Tensor, padding_size: int) -> torch.Tensor:
    """
    Aplica padding causal a un tensor para convoluciones dilatadas.
    
    Parámetros:
    -----------
    x : torch.Tensor
        Tensor de entrada con forma (batch, channels, time_steps)
    padding_size : int
        Tamaño del padding a aplicar
        
    Retorna:
    --------
    torch.Tensor
        Tensor con padding causal aplicado
    """
    return F.pad(x, (padding_size, 0))

class WaveNetBlock(nn.Module):
    """
    Bloque básico de WaveNet con activaciones gated y conexiones residuales/skip.
    
    Atributos:
    ----------
    filters : int
        Número de filtros en la capa convolucional
    kernel_size : int
        Tamaño del kernel para las convoluciones
    dilation_rate : int
        Tasa de dilatación para la convolución
    dropout_rate : float
        Tasa de dropout
    residual_scale : float
        Factor de escala para conexión residual
    use_skip_scale : bool
        Si se debe escalar la conexión skip
    """
    
    def __init__(
        self, 
        filters: int, 
        kernel_size: int, 
        dilation_rate: int, 
        dropout_rate: float,
        residual_scale: float = 0.1,
        use_skip_scale: bool = True
    ) -> None:
        """
        Inicializa un bloque WaveNet.
        
        Parámetros:
        -----------
        filters : int
            Número de filtros para las convoluciones
        kernel_size : int
            Tamaño del kernel convolucional
        dilation_rate : int
            Tasa de dilatación para la convolución
        dropout_rate : float
            Tasa de dropout para regularización
        residual_scale : float, opcional
            Factor de escala para la conexión residual (default: 0.1)
        use_skip_scale : bool, opcional
            Si se debe escalar la conexión skip (default: True)
        """
        super().__init__()
        
        self.filters = filters
        self.kernel_size = kernel_size
        self.dilation_rate = dilation_rate
        self.dropout_rate = dropout_rate
        self.residual_scale = residual_scale
        self.use_skip_scale = use_skip_scale
        
        # Convoluciones dilatadas para filter y gate
        self.filter_conv = nn.Conv1d(
            in_channels=filters,
            out_channels=filters,
            kernel_size=kernel_size,
            dilation=dilation_rate,
            bias=True
        )
        
        self.gate_conv = nn.Conv1d(
            in_channels=filters,
            out_channels=filters,
            kernel_size=kernel_size,
            dilation=dilation_rate,
            bias=True
        )
        
        # Normalización por lotes
        self.filter_norm = nn.BatchNorm1d(filters)
        self.gate_norm = nn.BatchNorm1d(filters)
        
        # Dropout
        self.dropout = nn.Dropout(dropout_rate)
        
        # Proyecciones para residual y skip
        self.residual_proj = nn.Conv1d(
            in_channels=filters,
            out_channels=filters,
            kernel_size=1,
            bias=True
        )
        
        self.skip_proj = nn.Conv1d(
            in_channels=filters,
            out_channels=filters,
            kernel_size=1,
            bias=True
        )
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Aplica el bloque WaveNet a la entrada.
        
        Parámetros:
        -----------
        x : torch.Tensor
            Tensor de entrada con forma (batch, channels, time_steps)
            
        Retorna:
        --------
        Tuple[torch.Tensor, torch.Tensor]
            Tupla con (salida_residual, salida_skip)
        """
        # Aplicar padding causal
        padding_size = (self.kernel_size - 1) * self.dilation_rate
        x_padded = causal_padding(x, padding_size)
        
        # Calcular filtro y gate
        filter_out = self.filter_conv(x_padded)
        gate_out = self.gate_conv(x_padded)
        
        filter_out = self.filter_norm(filter_out)
        gate_out = self.gate_norm(gate_out)
        
        # Aplicar activación gated (tanh * sigmoid)
        gated_out = torch.tanh(filter_out) * torch.sigmoid(gate_out)
        gated_out = self.dropout(gated_out)
        
        # Conexión residual
        residual_input_proj = self.residual_proj(x)
        
        # Alinear la dimensión temporal de la entrada proyectada con la salida gated
        # La convolución causal reduce la longitud, así que tomamos la parte final
        target_len = gated_out.size(2)
        residual_input_aligned = residual_input_proj[:, :, -target_len:]
        
        # Aplicar escalado residual y sumar
        residual_out = gated_out * self.residual_scale + residual_input_aligned
        
        # Conexión skip
        skip_out = self.skip_proj(gated_out)
        
        # Escalar skip connection si está habilitado
        if self.use_skip_scale:
            skip_out = skip_out * torch.sqrt(torch.tensor(1.0 - self.residual_scale))
        
        return residual_out, skip_out


class WaveNetModel(nn.Module):
    """
    Modelo WaveNet completo para regresión.
    
    Atributos:
    ----------
    config : Dict
        Configuración del modelo
    cgm_shape : Tuple[int, ...]
        Forma de los datos CGM
    other_features_shape : Tuple[int, ...]
        Forma de otras características
    """
    
    def __init__(self, cgm_shape: Tuple[int, ...], other_features_shape: Tuple[int, ...]) -> None:
        """
        Inicializa el modelo WaveNet.
        
        Parámetros:
        -----------
        cgm_shape : Tuple[int, ...]
            Forma de los datos CGM
        other_features_shape : Tuple[int, ...]
            Forma de otras características
        """
        super().__init__()
        
        # Obtener parámetros de configuración
        self.activation_name = WAVENET_CONFIG['activation']
        self.config_filters = WAVENET_CONFIG['filters']
        self.kernel_size = WAVENET_CONFIG['kernel_size']
        self.dilations = WAVENET_CONFIG['dilations']
        self.dropout_rate = WAVENET_CONFIG['dropout_rate']
        self.use_gating = WAVENET_CONFIG['use_gating']
        self.use_skip_scale = WAVENET_CONFIG['use_skip_scale']
        self.residual_scale = WAVENET_CONFIG['use_residual_scale']
        
        # Obtener función de activación
        self.activation_fn = get_activation_fn(self.activation_name)
        
        # Determinar dimensiones de entrada
        if len(cgm_shape) >= 2:
            self.cgm_steps, self.cgm_features = cgm_shape
        else:
            self.cgm_steps, self.cgm_features = 1, cgm_shape[0]
            
        self.other_features = other_features_shape[0] if len(other_features_shape) > 0 else 0
        
        # Dimensión objetivo para las skip connections (usar el último filtro por defecto)
        self.skip_channels = self.config_filters[-1]
        
        # Proyección inicial
        self.initial_conv = nn.Conv1d(
            in_channels=self.cgm_features,
            out_channels=self.config_filters[0],
            kernel_size=1,
            padding=0
        )
        self.initial_norm = nn.BatchNorm1d(self.config_filters[0])
        
        # Crear bloques WaveNet
        self.wavenet_blocks = nn.ModuleList()
        self.skip_projections = nn.ModuleList()
        
        for i, filters in enumerate(self.config_filters):
            for dilation in self.dilations:
                # Crear bloque WaveNet
                block = WaveNetBlock(
                    filters=filters,
                    kernel_size=self.kernel_size,
                    dilation_rate=dilation,
                    dropout_rate=self.dropout_rate,
                    residual_scale=self.residual_scale,
                    use_skip_scale=self.use_skip_scale
                )
                self.wavenet_blocks.append(block)
                
                # Proyección para skip connection
                skip_proj = nn.Conv1d(
                    in_channels=filters,
                    out_channels=self.skip_channels,
                    kernel_size=1,
                    padding=0
                )
                self.skip_projections.append(skip_proj)
        
        # Capas de post-procesamiento después de sumar las conexiones skip
        self.post_skip_act1 = lambda x: self.activation_fn(x)
        self.post_skip_conv1 = nn.Conv1d(
            in_channels=self.skip_channels,
            out_channels=self.skip_channels,
            kernel_size=1,
            padding=0
        )
        self.post_skip_act2 = lambda x: self.activation_fn(x)
        
        # Pooling para reducir la dimensión temporal
        self.global_pooling = nn.AdaptiveAvgPool1d(1)
        
        # Capas densas finales
        dense_input_size = self.skip_channels + self.other_features
        
        self.final_dense1 = nn.Linear(dense_input_size, 128)
        self.final_norm1 = nn.BatchNorm1d(128)
        self.final_act1 = lambda x: self.activation_fn(x)
        self.final_dropout1 = nn.Dropout(self.dropout_rate)
        
        self.final_dense2 = nn.Linear(128, 64)
        self.final_norm2 = nn.BatchNorm1d(64)
        self.final_act2 = lambda x: self.activation_fn(x)
        self.final_dropout2 = nn.Dropout(self.dropout_rate)
        
        # Capa de salida
        self.output_layer = nn.Linear(64, 1)
    
    def forward(self, cgm_input: torch.Tensor, other_input: torch.Tensor) -> torch.Tensor:
        """
        Realiza la pasada hacia adelante del modelo WaveNet.
        
        Parámetros:
        -----------
        cgm_input : torch.Tensor
            Datos CGM de entrada con forma (batch, time_steps, features)
        other_input : torch.Tensor
            Otras características con forma (batch, features)
            
        Retorna:
        --------
        torch.Tensor
            Predicciones del modelo con forma (batch, 1)
        """
        # Cambiar forma para convolución 1D: (batch, time_steps, features) -> (batch, features, time_steps)
        x = cgm_input.transpose(1, 2)
        
        # Proyección inicial
        x = self.initial_conv(x)
        x = self.initial_norm(x)
        x = self.activation_fn(x)
        
        # Lista para almacenar las salidas de skip connections
        skip_outputs = []
        
        # Aplicar bloques WaveNet
        current_input = x
        for i, (block, skip_proj) in enumerate(zip(self.wavenet_blocks, self.skip_projections)):
            # Aplicar bloque WaveNet
            residual, skip = block(current_input)
            
            # Proyectar skip a la dimensión objetivo
            projected_skip = skip_proj(skip)
            skip_outputs.append(projected_skip)
            
            # Actualizar input para el siguiente bloque
            current_input = residual
        
        # Combinar las salidas de skip
        if skip_outputs:
            combined_skip = torch.stack(skip_outputs).sum(dim=0)
        else:
            # Si no hay bloques, crear un tensor de ceros con la dimensión esperada
            batch_size = x.size(0)
            combined_skip = torch.zeros(
                (batch_size, self.skip_channels, x.size(2)), 
                device=x.device
            )
        
        # Post-procesamiento de las conexiones skip
        post_skip = self.post_skip_act1(combined_skip)
        post_skip = self.post_skip_conv1(post_skip)
        post_skip = self.post_skip_act2(post_skip)
        
        # Pooling global
        pooled_output = self.global_pooling(post_skip).squeeze(2)  # (batch, skip_channels)
        
        # Combinar con otras características
        combined_features = torch.cat([pooled_output, other_input], dim=1)
        
        # Capas densas finales
        x = self.final_dense1(combined_features)
        x = self.final_norm1(x)
        x = self.final_act1(x)
        x = self.final_dropout1(x)
        
        x = self.final_dense2(x)
        x = self.final_norm2(x)
        x = self.final_act2(x)
        x = self.final_dropout2(x)
        
        # Capa de salida
        output = self.output_layer(x)
        
        return output


def create_wavenet_model(cgm_shape: Tuple[int, ...], other_features_shape: Tuple[int, ...]) -> nn.Module:
    """
    Crea una instancia del modelo WaveNet.
    
    Parámetros:
    -----------
    cgm_shape : Tuple[int, ...]
        Forma de los datos CGM (pasos_tiempo, características_cgm)
    other_features_shape : Tuple[int, ...]
        Forma de otras características (características_otras,)
        
    Retorna:
    --------
    nn.Module
        Instancia del modelo WaveNet
    """
    return WaveNetModel(cgm_shape, other_features_shape)


def create_wavenet_model_wrapper(cgm_shape: Tuple[int, ...], other_features_shape: Tuple[int, ...]) -> DLModelWrapper:
    """
    Crea un modelo WaveNet envuelto en DLModelWrapper para compatibilidad con el sistema de entrenamiento.
    
    Parámetros:
    -----------
    cgm_shape : Tuple[int, ...]
        Forma de los datos CGM (pasos_tiempo, características_cgm)
    other_features_shape : Tuple[int, ...]
        Forma de otras características (características_otras,)
        
    Retorna:
    --------
    DLModelWrapper
        Modelo WaveNet envuelto y listo para ser usado por el sistema de entrenamiento
    """
    # Función creadora de modelo
    model_creator = lambda: create_wavenet_model(cgm_shape, other_features_shape)
    
    # Crear wrapper
    model_wrapper = DLModelWrapperPyTorch(model_creator=model_creator)
    
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
    Retorna la función que crea el modelo WaveNet envuelto.
    
    Esta función es la que se importa y usa en params.py.
    
    Retorna:
    --------
    Callable[[Tuple[int, ...], Tuple[int, ...]], DLModelWrapper]
        Función que crea el modelo WaveNet envuelto
    """
    return create_wavenet_model_wrapper