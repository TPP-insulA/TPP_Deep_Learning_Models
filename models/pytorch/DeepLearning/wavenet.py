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
from custom.DeepLearning.dl_model_wrapper import DLModelWrapper
from custom.DeepLearning.dl_pt import DLModelWrapperPyTorch
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
    Bloque residual para WaveNet con conexiones residuales y de salto.
    
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
    
    def __init__(self, residual_channels: int, skip_channels: int, dilation: int, 
                 kernel_size: int = 3, causal: bool = True, use_bias: bool = True,
                 input_channels: Optional[int] = None) -> None:
        """
        Inicializa un bloque residual WaveNet.
        
        Parámetros:
        -----------
        residual_channels : int
            Número de filtros para las convoluciones
        skip_channels : int
            Número de filtros para la convolución de la conexión skip
        dilation : int
            Tasa de dilatación para la convolución
        kernel_size : int, opcional
            Tamaño del kernel convolucional (default: 3)
        causal : bool, opcional
            Si se debe usar padding causal (default: True)
        use_bias : bool, opcional
            Si se debe usar sesgo en las capas convolucionales (default: True)
        input_channels : int, opcional
            Número de canales en la entrada. Si es diferente de residual_channels, 
            se añade una proyección de entrada (default: None)
        """
        super().__init__()
        
        self.residual_channels = residual_channels
        self.skip_channels = skip_channels
        self.dilation = dilation
        self.kernel_size = kernel_size
        self.causal = causal
        
        # Proyección de entrada si es necesario
        self.input_projection = None
        if input_channels is not None and input_channels != residual_channels:
            self.input_projection = nn.Conv1d(
                input_channels, 
                residual_channels, 
                kernel_size=1, 
                bias=use_bias
            )
        
        # Convoluciones dilatadas para filter y gate
        self.filter_conv = nn.Conv1d(
            residual_channels, 
            residual_channels, 
            kernel_size, 
            dilation=dilation, 
            bias=use_bias
        )
        
        self.gate_conv = nn.Conv1d(
            residual_channels, 
            residual_channels, 
            kernel_size, 
            dilation=dilation, 
            bias=use_bias
        )
        
        # Normalización por lotes
        self.filter_norm = nn.BatchNorm1d(residual_channels)
        self.gate_norm = nn.BatchNorm1d(residual_channels)
        
        # Dropout
        self.dropout = nn.Dropout(0.2)
        
        # Proyecciones para residual y skip
        self.residual_proj = nn.Conv1d(
            in_channels=residual_channels,
            out_channels=residual_channels,
            kernel_size=1,
            bias=True
        )
        
        self.skip_proj = nn.Conv1d(
            in_channels=residual_channels,
            out_channels=skip_channels,
            kernel_size=1,
            bias=True
        )
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Procesa la entrada a través del bloque residual.
        
        Parámetros:
        -----------
        x : torch.Tensor
            Tensor de entrada con forma (batch, channels, time_steps)
            
        Retorna:
        --------
        Tuple[torch.Tensor, torch.Tensor]
            Tupla con (salida_residual, salida_skip)
        """
        # Proyectar entrada si es necesario
        if self.input_projection is not None:
            x = self.input_projection(x)
        
        # Guardar la entrada original para la conexión residual
        x_original = x
        
        # Aplicar padding causal si es necesario
        if self.causal:
            padding_size = (self.kernel_size - 1) * self.dilation
            x_padded = causal_padding(x, padding_size)
        else:
            x_padded = x
        
        # Calcular filtro y gate usando la versión con padding
        filter_out = self.filter_conv(x_padded)
        gate_out = self.gate_conv(x_padded)
        
        filter_out = self.filter_norm(filter_out)
        gate_out = self.gate_norm(gate_out)
        
        # Aplicar activación gated (tanh * sigmoid)
        gated_out = torch.tanh(filter_out) * torch.sigmoid(gate_out)
        gated_out = self.dropout(gated_out)
        
        # Conexión residual con la entrada original (sin padding)
        residual_out = self.residual_proj(gated_out) + x_original
        
        # Conexión skip
        skip_out = self.skip_proj(gated_out)
        
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
    
    def __init__(self, cgm_shape: Tuple[int, ...], other_features_shape: Tuple[int, ...], 
             initial_channels: int = 32, residual_channels: int = 32, skip_channels: int = 64) -> None:
        """
        Inicializa el modelo WaveNet.
        
        Parámetros:
        -----------
        cgm_shape : Tuple[int, ...]
            Forma de los datos CGM
        other_features_shape : Tuple[int, ...]
            Forma de otras características
        initial_channels : int, opcional
            Número de canales después de la primera convolución (default: 32)
        residual_channels : int, opcional
            Número de canales en los bloques residuales (default: 32)
        skip_channels : int, opcional
            Número de canales en las conexiones skip (default: 64)
        """
        super().__init__()
        
        # Guardar parámetros de canales
        self.initial_channels = initial_channels
        self.residual_channels = residual_channels  # Añadido
        self.skip_channels = skip_channels          # Reemplaza self.config_filters[-1]
        
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
            out_channels=self.initial_channels,
            kernel_size=1,
            padding=0
        )
        self.initial_norm = nn.BatchNorm1d(self.initial_channels)
        
        # Añadir proyección de canal si es necesario
        self.channel_projection = None
        if self.initial_channels != self.residual_channels:
            self.channel_projection = nn.Conv1d(
                in_channels=self.initial_channels,
                out_channels=self.residual_channels,
                kernel_size=1,
                bias=True
            )
        
        # Crear bloques WaveNet
        self.residual_blocks = nn.ModuleList()  # Cambiar el nombre para que coincida con el forward

        for i, dilation in enumerate(self.dilations):
            # Crear bloque WaveNet con los nuevos parámetros
            block = WaveNetBlock(
                residual_channels=self.residual_channels,
                skip_channels=self.skip_channels,
                dilation=dilation,
                kernel_size=self.kernel_size,
                causal=True,
                use_bias=True,
                # Para el primer bloque, especificar los canales de entrada
                input_channels=self.initial_channels if i == 0 and self.channel_projection is None else None
            )
            self.residual_blocks.append(block)
        
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
        Realiza una pasada forward del modelo WaveNet.
        
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
        # Transponer los datos CGM para que sean compatibles con Conv1D
        # De [batch, time_steps, features] a [batch, features, time_steps]
        if cgm_input.dim() == 3:
            cgm_input = cgm_input.transpose(1, 2)
        
        # Procesar entrada CGM
        x = self.initial_conv(cgm_input)
        
        # Asegurarse de que x tiene el número correcto de canales para los bloques residuales
        if x.size(1) != self.residual_channels:
            x = self.channel_projection(x)  # Añadir esta capa en __init__
        
        # Procesamiento a través de bloques residuales
        skip_connections = []
        for block in self.residual_blocks:
            residual, skip = block(x)
            x = x + residual  # Conexión residual
            skip_connections.append(skip)
        
        # Combinar las salidas de skip
        if skip_connections:
            combined_skip = torch.stack(skip_connections).sum(dim=0)
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
    config = WAVENET_CONFIG
    
    # Extraer parámetros de canales
    initial_channels = config.get('initial_channels', 32)
    residual_channels = config.get('residual_channels', 32)
    skip_channels = config.get('skip_channels', 64)
    
    # Crear el modelo WaveNet con parámetros específicos
    model = WaveNetModel(
        cgm_shape=cgm_shape, 
        other_features_shape=other_features_shape,
        initial_channels=initial_channels,
        residual_channels=residual_channels,
        skip_channels=skip_channels
    )
    
    return model


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