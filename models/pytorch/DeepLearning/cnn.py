import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Dict, Any, Optional, Callable, List

PROJECT_ROOT = os.path.abspath(os.getcwd())
sys.path.append(PROJECT_ROOT) 

from config.models_config import CNN_CONFIG
from custom.DeepLearning.dl_model_wrapper import DLModelWrapper
from custom.DeepLearning.dl_pt import DLModelWrapperPyTorch
from models.early_stopping import get_early_stopping_config

# Constantes para cadenas repetidas
CONST_RELU = "relu"
CONST_GELU = "gelu"
CONST_SWISH = "swish"
CONST_SILU = "silu"
CONST_SAME = "same"

class SqueezeExcitationBlock(nn.Module):
    """
    Bloque Squeeze-and-Excitation como módulo personalizado.
    
    Parámetros:
    -----------
    filters : int
        Número de filtros del bloque
    se_ratio : int
        Factor de reducción para la capa de squeeze
        
    Retorna:
    --------
    torch.Tensor
        Tensor de entrada escalado por los pesos de atención
    """
    def __init__(self, filters: int, se_ratio: int = 16) -> None:
        super().__init__()
        self.filters = filters
        self.se_ratio = se_ratio
        
        # Definir las capas
        self.gap = nn.AdaptiveAvgPool1d(1)
        self.fc1 = nn.Linear(filters, filters // se_ratio)
        self.fc2 = nn.Linear(filters // se_ratio, filters)
    
    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """
        Aplica el mecanismo de Squeeze-and-Excitation a los inputs.
        
        Parámetros:
        -----------
        inputs : torch.Tensor
            Tensor de entrada
            
        Retorna:
        --------
        torch.Tensor
            Tensor procesado con atención de canal
        """
        # Squeeze
        x = self.gap(inputs)
        x = x.view(x.size(0), -1)
        
        # Excitation
        x = F.gelu(self.fc1(x))
        x = torch.sigmoid(self.fc2(x))
        
        # Scale - reshape para broadcasting con inputs
        x = x.unsqueeze(2)
        
        return inputs * x

def get_activation(x: torch.Tensor, activation_name: str) -> torch.Tensor:
    """
    Aplica la función de activación según su nombre.
    
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
    Bloque residual con convoluciones dilatadas y SE.
    
    Parámetros:
    -----------
    in_channels : int
        Número de canales de entrada
    out_channels : int
        Número de canales de salida
    kernel_size : int
        Tamaño del kernel para las convoluciones
    dilation_rate : int
        Tasa de dilatación para las convoluciones
    dropout_rate : float
        Tasa de dropout para regularización
    use_se_block : bool
        Si se debe usar un bloque Squeeze-and-Excitation
    se_ratio : int
        Factor de reducción para el bloque SE
    use_layer_norm : bool
        Si usar normalización de capa en lugar de normalización por lotes
        
    Retorna:
    --------
    torch.Tensor
        Tensor procesado con conexión residual
    """
    def __init__(
        self, 
        in_channels: int, 
        out_channels: int, 
        kernel_size: int, 
        dilation_rate: int = 1, 
        dropout_rate: float = 0.1,
        use_se_block: bool = True, 
        se_ratio: int = 16,
        use_layer_norm: bool = True
    ) -> None:
        super().__init__()
        self.use_layer_norm = use_layer_norm
        self.use_se_block = use_se_block
        
        # Primera convolución con dilatación
        self.conv1 = nn.Conv1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            padding=(kernel_size - 1) // 2 * dilation_rate,
            dilation=dilation_rate
        )
        
        # Normalización
        if use_layer_norm:
            # Solo especificar el número de características para normalizar
            self.norm1 = nn.LayerNorm(out_channels)
            self.norm2 = nn.LayerNorm(out_channels)
        else:
            self.norm1 = nn.BatchNorm1d(out_channels)
            self.norm2 = nn.BatchNorm1d(out_channels)
        
        # Segunda convolución separable para reducir parámetros
        self.depthwise = nn.Conv1d(
            in_channels=out_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            padding=(kernel_size - 1) // 2,
            groups=out_channels
        )
        self.pointwise = nn.Conv1d(
            in_channels=out_channels,
            out_channels=out_channels,
            kernel_size=1
        )
        
        # Normalización
        if use_layer_norm:
            # Solo especificar el número de características para normalizar
            self.norm2 = nn.LayerNorm(out_channels)
        else:
            self.norm2 = nn.BatchNorm1d(out_channels)
        
        # Squeeze-and-Excitation
        if use_se_block:
            self.se = SqueezeExcitationBlock(out_channels, se_ratio)
        
        # Dropout
        self.dropout = nn.Dropout(dropout_rate)
        
        # Conexión residual con proyección si es necesario
        self.projection = None
        if in_channels != out_channels:
            self.projection = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=1),
                nn.BatchNorm1d(out_channels) if not use_layer_norm else nn.Identity()
            )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Aplica las operaciones del bloque residual.
        
        Parámetros:
        -----------
        x : torch.Tensor
            Tensor de entrada
            
        Retorna:
        --------
        torch.Tensor
            Tensor procesado con conexión residual
        """
        # Guardar entrada para la conexión residual
        identity = x
        
        # Primera convolución con dilatación
        x = self.conv1(x)
        
        # Normalización y activación
        if self.use_layer_norm:
            # Permuta para poner los canales al final
            x = x.permute(0, 2, 1)
            x = self.norm1(x)
            x = x.permute(0, 2, 1)
        else:
            x = self.norm1(x)
        
        x = get_activation(x, CNN_CONFIG['activation'])
        
        # Segunda convolución separable
        x = self.depthwise(x)
        x = self.pointwise(x)
        
        # Normalización
        if self.use_layer_norm:
            # Permuta para poner los canales al final
            x = x.permute(0, 2, 1)
            x = self.norm2(x)
            x = x.permute(0, 2, 1)
        else:
            x = self.norm2(x)
        
        # Squeeze-and-Excitation
        if self.use_se_block:
            x = self.se(x)
        
        # Dropout
        x = self.dropout(x)
        
        # Conexión residual con proyección si es necesario
        if self.projection is not None:
            identity = self.projection(identity)
        
        # Sumar con la conexión residual
        x = x + identity
        
        # Activación final
        x = get_activation(x, CNN_CONFIG['activation'])
        
        return x

def apply_normalization(x: torch.Tensor, use_layer_norm: bool = True) -> torch.Tensor:
    """
    Aplica normalización según la configuración.
    
    Parámetros:
    -----------
    x : torch.Tensor
        Tensor de entrada
    use_layer_norm : bool, opcional
        Si usar normalización de capa (default: True)
        
    Retorna:
    --------
    torch.Tensor
        Tensor normalizado
    """
    if use_layer_norm:
        size = x.size()
        x = x.permute(0, 2, 1)
        x = nn.LayerNorm(x.size()[1:])(x)
        x = x.permute(0, 2, 1)
        return x.view(size)
    else:
        return nn.BatchNorm1d(x.size(1))(x)

def apply_conv_block(x: torch.Tensor, in_channels: int, out_channels: int, 
                    kernel_size: int, stride: int = 1) -> Tuple[torch.Tensor, nn.Module]:
    """
    Aplica un bloque de convolución con normalización y activación.
    
    Parámetros:
    -----------
    x : torch.Tensor
        Tensor de entrada
    in_channels : int
        Número de canales de entrada
    out_channels : int
        Número de canales de salida
    kernel_size : int
        Tamaño del kernel para la convolución
    stride : int, opcional
        Paso para la convolución (default: 1)
        
    Retorna:
    --------
    Tuple[torch.Tensor, nn.Module]
        Tensor procesado y módulo convolucional
    """
    conv = nn.Conv1d(
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=kernel_size,
        stride=stride,
        padding=(kernel_size - 1) // 2
    )
    x = conv(x)
    x = apply_normalization(x, CNN_CONFIG['use_layer_norm'])
    x = get_activation(x, CNN_CONFIG['activation'])
    return x, conv

class CNNModel(nn.Module):
    """
    Modelo CNN (Red Neuronal Convolucional) con entrada dual para datos CGM y otras características.
    
    Parámetros:
    -----------
    cgm_shape : Tuple[int, ...]
        Forma de los datos CGM (timesteps, features)
    other_features_shape : Tuple[int, ...]
        Forma de otras características (features)
    """
    def __init__(self, cgm_shape: Tuple[int, ...], other_features_shape: Tuple[int, ...]) -> None:
        super().__init__()
        self.cgm_shape = cgm_shape
        self.other_features_shape = other_features_shape
        
        # Extracción de dimensiones
        _, cgm_features = cgm_shape if len(cgm_shape) >= 2 else (cgm_shape[0], 1)
        other_features = other_features_shape[0] if len(other_features_shape) > 0 else 1
        
        # Proyección inicial
        self.initial_conv = nn.Conv1d(
            in_channels=cgm_features,
            out_channels=CNN_CONFIG['filters'][0],
            kernel_size=1,
            padding=0
        )
        
        # Normalización inicial
        if CNN_CONFIG['use_layer_norm']:
            # Solo especificar el número de características (canales) para normalizar
            self.initial_norm = nn.LayerNorm(CNN_CONFIG['filters'][0])
        else:
            self.initial_norm = nn.BatchNorm1d(CNN_CONFIG['filters'][0])
        
        # Bloques residuales
        self.residual_blocks = nn.ModuleList()
        current_channels = CNN_CONFIG['filters'][0]
        
        for i, filters in enumerate(CNN_CONFIG['filters']):
            # Bloques residuales con diferentes tasas de dilatación
            for dilation_rate in CNN_CONFIG['dilation_rates']:
                self.residual_blocks.append(
                    ResidualBlock(
                        in_channels=current_channels,
                        out_channels=filters,
                        kernel_size=CNN_CONFIG['kernel_size'],
                        dilation_rate=dilation_rate,
                        dropout_rate=CNN_CONFIG['dropout_rate'],
                        use_se_block=CNN_CONFIG['use_se_block'],
                        se_ratio=CNN_CONFIG['se_ratio'],
                        use_layer_norm=CNN_CONFIG['use_layer_norm']
                    )
                )
                current_channels = filters
            
            # Downsampling entre bloques (excepto el último)
            if i < len(CNN_CONFIG['filters']) - 1:
                self.residual_blocks.append(
                    nn.Conv1d(
                        in_channels=current_channels,
                        out_channels=CNN_CONFIG['filters'][i+1],
                        kernel_size=CNN_CONFIG['pool_size'],
                        stride=CNN_CONFIG['pool_size'],
                        padding=(CNN_CONFIG['pool_size'] - 1) // 2
                    )
                )
                current_channels = CNN_CONFIG['filters'][i+1]
        
        # Pooling global
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.max_pool = nn.AdaptiveMaxPool1d(1)
        
        # Tamaño combinado después del pooling
        combined_size = current_channels * 2 + other_features
        
        # MLP final
        self.mlp_layers = nn.Sequential(
            nn.Linear(combined_size, 256),
            nn.GELU(),
            nn.LayerNorm(256) if CNN_CONFIG['use_layer_norm'] else nn.BatchNorm1d(256),
            nn.Dropout(CNN_CONFIG['dropout_rate']),
            
            nn.Linear(256, 256),
            nn.GELU(),
            
            nn.Linear(256, 128),
            nn.GELU(),
            nn.LayerNorm(128) if CNN_CONFIG['use_layer_norm'] else nn.BatchNorm1d(128),
            nn.Dropout(CNN_CONFIG['dropout_rate'] / 2),
            
            nn.Linear(128, 1)
        )
    
    def forward(self, cgm_input: torch.Tensor, other_input: torch.Tensor) -> torch.Tensor:
        """
        Ejecuta el modelo CNN sobre las entradas.
        
        Parámetros:
        -----------
        cgm_input : torch.Tensor
            Datos de entrada CGM con forma (batch, timesteps, features)
        other_input : torch.Tensor
            Otras características de entrada con forma (batch, features)
            
        Retorna:
        --------
        torch.Tensor
            Predicciones del modelo
        """
        # Asegurar que la entrada CGM tenga formato adecuado para Conv1D (batch, channels, length)
        if cgm_input.dim() == 3:
            # (batch, timesteps, features) -> (batch, features, timesteps)
            x = cgm_input.permute(0, 2, 1)
        else:
            # Si solo tiene 2 dimensiones, añadir una dimensión de canal
            x = cgm_input.unsqueeze(1)
        
        # Proyección inicial
        x = self.initial_conv(x)
        
        # Normalización inicial
        if CNN_CONFIG['use_layer_norm']:
            # Permuta para poner los canales al final: [batch, channels, length] -> [batch, length, channels]
            x = x.permute(0, 2, 1)
            # Aplica LayerNorm (normaliza sobre la última dimensión)
            x = self.initial_norm(x)
            # Permuta de vuelta: [batch, length, channels] -> [batch, channels, length]
            x = x.permute(0, 2, 1)
        else:
            x = self.initial_norm(x)
        
        # Activación inicial
        x = get_activation(x, CNN_CONFIG['activation'])
        
        # Aplicar bloques residuales
        for block in self.residual_blocks:
            x = block(x)
        
        # Pooling global
        avg_pool = self.avg_pool(x).view(x.size(0), -1)
        max_pool = self.max_pool(x).view(x.size(0), -1)
        
        # Combinar con otras características
        combined = torch.cat([avg_pool, max_pool, other_input], dim=1)
        
        # MLP final para predicción
        output = self.mlp_layers(combined)
        
        return output

def create_cnn_model(cgm_shape: Tuple[int, ...], other_features_shape: Tuple[int, ...]) -> nn.Module:
    """
    Crea un modelo CNN avanzado con características modernas como bloques residuales,
    convoluciones dilatadas y Squeeze-and-Excitation.
    
    Parámetros:
    -----------
    cgm_shape : Tuple[int, ...]
        Forma de los datos CGM (timesteps, features) sin incluir el batch
    other_features_shape : Tuple[int, ...]
        Forma de otras características (features) sin incluir el batch
        
    Retorna:
    --------
    nn.Module
        Modelo CNN avanzado
    """
    # Validar la forma de entrada
    if len(cgm_shape) < 1:
        raise ValueError(f"La entrada CGM debe tener al menos 1 dimensión. Recibido: {cgm_shape}")
    
    # Crear y retornar el modelo
    return CNNModel(cgm_shape, other_features_shape)

def create_cnn_model_wrapper(cgm_shape: Tuple[int, ...], other_features_shape: Tuple[int, ...]) -> DLModelWrapper:
    """
    Crea un modelo CNN avanzado envuelto en DLModelWrapperPyTorch.
    
    Parámetros:
    -----------
    cgm_shape : Tuple[int, ...]
        Forma de los datos CGM (timesteps, features)
    other_features_shape : Tuple[int, ...]
        Forma de otras características (features)
        
    Retorna:
    --------
    DLModelWrapper
        Modelo CNN avanzado envuelto en DLModelWrapper
    """
    # Crear wrapper con modelo
    model_wrapper = DLModelWrapperPyTorch(
        model_creator=lambda: create_cnn_model(cgm_shape, other_features_shape)
    )
    
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
    Retorna una función para crear un modelo CNN compatible con la API del sistema.
    
    Retorna:
    --------
    Callable[[Tuple[int, ...], Tuple[int, ...]], DLModelWrapper]
        Función para crear el modelo con las formas de entrada especificadas
    """
    return create_cnn_model_wrapper