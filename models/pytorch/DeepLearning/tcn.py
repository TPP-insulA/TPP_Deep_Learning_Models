import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Dict, List, Any, Callable, Optional, Union

PROJECT_ROOT = os.path.abspath(os.getcwd())
sys.path.append(PROJECT_ROOT)

from config.models_config import TCN_CONFIG
from custom.dl_model_wrapper import DLModelWrapper, DLModelWrapperPyTorch
from models.early_stopping import get_early_stopping_config

# Constantes para cadenas repetidas
CONST_RELU = "relu"
CONST_GELU = "gelu"
CONST_SELU = "selu"
CONST_VALID = "valid"
CONST_SAME = "same"
CONST_CAUSAL = "causal"

def get_activation(x: torch.Tensor, activation_name: str) -> torch.Tensor:
    """
    Aplica la función de activación especificada al tensor.
    
    Parámetros:
    -----------
    x : torch.Tensor
        Tensor de entrada
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
    elif activation_name == CONST_SELU:
        return F.selu(x)
    else:
        return F.relu(x)  # Por defecto


class WeightNorm(nn.Module):
    """
    Implementa normalización de pesos para capas convolucionales.
    
    Parámetros:
    -----------
    module : nn.Module
        Módulo al que aplicar la normalización
    dim : int
        Dimensión a lo largo de la cual normalizar
    """
    
    def __init__(self, module: nn.Module, dim: int = 0) -> None:
        super().__init__()
        self.module = module
        self.dim = dim
        
        # Registrar parámetro g para escalar los pesos normalizados
        weight = getattr(self.module, 'weight')
        weight_size = weight.size()
        self.g = nn.Parameter(torch.ones(weight_size[self.dim]))
        
        # Inicializar g con la norma de los pesos
        self._init_g()
    
    def _init_g(self) -> None:
        """Inicializa el parámetro g con la norma de los pesos."""
        weight = getattr(self.module, 'weight')
        with torch.no_grad():
            norm = weight.norm(p=2, dim=tuple(i for i in range(weight.dim()) if i != self.dim), keepdim=True)
            self.g.data = norm.mean(dim=tuple(i for i in range(weight.dim()) if i != self.dim))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Aplica la normalización de pesos y ejecuta el módulo.
        
        Parámetros:
        -----------
        x : torch.Tensor
            Tensor de entrada
            
        Retorna:
        --------
        torch.Tensor
            Resultado del módulo con pesos normalizados
        """
        weight = getattr(self.module, 'weight')
        # Normalizar los pesos
        norm = weight.norm(p=2, dim=tuple(i for i in range(weight.dim()) if i != self.dim), keepdim=True)
        weight_normalized = weight / (norm + 1e-5)
        
        # Escalar por g
        weight_shape = [1] * weight.dim()
        weight_shape[self.dim] = -1
        weight_g = self.g.view(*weight_shape)
        weight_scaled = weight_normalized * weight_g
        
        # Reemplazar temporalmente el peso
        tmp_weight = getattr(self.module, 'weight')
        self.module.weight.data = weight_scaled
        
        # Ejecutar el módulo
        output = self.module(x)
        
        # Restaurar el peso original
        setattr(self.module, 'weight', tmp_weight)
        
        return output


def apply_weight_norm(module: nn.Module) -> nn.Module:
    """
    Aplica normalización de pesos a un módulo.
    
    Parámetros:
    -----------
    module : nn.Module
        Módulo al que aplicar la normalización
        
    Retorna:
    --------
    nn.Module
        Módulo con normalización de pesos aplicada
    """
    return WeightNorm(module)


def causal_padding(x: torch.Tensor, padding_size: int) -> torch.Tensor:
    """
    Aplica padding causal (solo al principio) a un tensor 1D.
    
    Parámetros:
    -----------
    x : torch.Tensor
        Tensor de entrada [batch, channels, length]
    padding_size : int
        Tamaño del padding a aplicar
        
    Retorna:
    --------
    torch.Tensor
        Tensor con padding causal aplicado
    """
    return F.pad(x, (padding_size, 0))


class TCNResidualBlock(nn.Module):
    """
    Bloque residual para TCN con convoluciones dilatadas.
    
    Parámetros:
    -----------
    in_channels : int
        Número de canales de entrada
    out_channels : int
        Número de canales de salida
    kernel_size : int
        Tamaño del kernel convolucional
    dilation_rate : int
        Tasa de dilatación para la convolución
    dropout_rate : float
        Tasa de dropout para regularización
    use_weight_norm : bool
        Si usar normalización de pesos
    use_layer_norm : bool
        Si usar normalización de capa en lugar de batch norm
    use_spatial_dropout : bool
        Si usar dropout espacial
    activation : str
        Nombre de la función de activación
    """
    
    def __init__(
        self, 
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        dilation_rate: int = 1,
        dropout_rate: float = 0.2,
        use_weight_norm: bool = True,
        use_layer_norm: bool = True,
        use_spatial_dropout: bool = True,
        activation: str = CONST_GELU
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.dilation_rate = dilation_rate
        self.dropout_rate = dropout_rate
        self.use_weight_norm = use_weight_norm
        self.use_layer_norm = use_layer_norm
        self.use_spatial_dropout = use_spatial_dropout
        self.activation = activation
        
        # Primera convolución
        self.conv1 = nn.Conv1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            dilation=dilation_rate,
            padding=0  # Aplicaremos padding causal manualmente
        )
        
        # Segunda convolución
        self.conv2 = nn.Conv1d(
            in_channels=out_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            dilation=dilation_rate,
            padding=0  # Aplicaremos padding causal manualmente
        )
        
        # Aplicar normalización de pesos si está habilitada
        if use_weight_norm:
            self.conv1 = apply_weight_norm(self.conv1)
            self.conv2 = apply_weight_norm(self.conv2)
        
        # Normalización
        if use_layer_norm:
            self.norm1 = nn.LayerNorm(out_channels)
            self.norm2 = nn.LayerNorm(out_channels)
        else:
            self.norm1 = nn.BatchNorm1d(out_channels)
            self.norm2 = nn.BatchNorm1d(out_channels)
        
        # Dropout
        if use_spatial_dropout:
            self.dropout1 = nn.Dropout2d(dropout_rate)
            self.dropout2 = nn.Dropout2d(dropout_rate)
        else:
            self.dropout1 = nn.Dropout(dropout_rate)
            self.dropout2 = nn.Dropout(dropout_rate)
        
        # Convolución para la conexión residual si cambian los canales
        self.residual_conv = nn.Conv1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=1,
            padding=0
        ) if in_channels != out_channels else None
        
        if use_weight_norm and self.residual_conv:
            self.residual_conv = apply_weight_norm(self.residual_conv)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Procesa la entrada a través del bloque TCN.
        
        Parámetros:
        -----------
        x : torch.Tensor
            Tensor de entrada [batch, channels, length]
            
        Retorna:
        --------
        torch.Tensor
            Tensor de salida del bloque
        """
        # Guardar entrada original para conexión residual
        residual = x
        
        # Primera parte del bloque
        # Aplicar padding causal manual
        padding_size = (self.kernel_size - 1) * self.dilation_rate
        padded = causal_padding(x, padding_size)
        
        # Primera convolución + activación + normalización + dropout
        out = self.conv1(padded)
        
        # Normalización (ajustamos la forma si es LayerNorm)
        if self.use_layer_norm:
            out = out.transpose(1, 2)  # [batch, length, channels]
            out = self.norm1(out)
            out = out.transpose(1, 2)  # [batch, channels, length]
        else:
            out = self.norm1(out)
        
        # Activación
        out = get_activation(out, self.activation)
        
        # Dropout (ajustamos para spatial dropout)
        if self.use_spatial_dropout:
            # Para spatial dropout, añadimos una dimensión y luego la quitamos
            out = out.unsqueeze(3)  # [batch, channels, length, 1]
            out = self.dropout1(out)
            out = out.squeeze(3)  # [batch, channels, length]
        else:
            out = self.dropout1(out)
        
        # Segunda parte del bloque
        # Aplicar padding causal manual nuevamente
        padded = causal_padding(out, padding_size)
        
        # Segunda convolución + activación + normalización + dropout
        out = self.conv2(padded)
        
        # Normalización (ajustamos la forma si es LayerNorm)
        if self.use_layer_norm:
            out = out.transpose(1, 2)  # [batch, length, channels]
            out = self.norm2(out)
            out = out.transpose(1, 2)  # [batch, channels, length]
        else:
            out = self.norm2(out)
        
        # Activación
        out = get_activation(out, self.activation)
        
        # Dropout (ajustamos para spatial dropout)
        if self.use_spatial_dropout:
            out = out.unsqueeze(3)  # [batch, channels, length, 1]
            out = self.dropout2(out)
            out = out.squeeze(3)  # [batch, channels, length]
        else:
            out = self.dropout2(out)
        
        # Proyectar residual si es necesario
        if self.residual_conv:
            residual = self.residual_conv(residual)
        
        # Asegurar que las dimensiones temporales coincidan
        target_length = out.size(2)
        if residual.size(2) > target_length:
            residual = residual[:, :, -target_length:]
        
        # Conexión residual
        out = out + residual
        
        # Activación final
        out = get_activation(out, self.activation)
        
        return out


class TCNModel(nn.Module):
    """
    Modelo TCN (Temporal Convolutional Network) para series temporales.
    
    Parámetros:
    -----------
    cgm_shape : Tuple[int, ...]
        Forma de los datos CGM (timesteps, features)
    other_features_shape : Tuple[int, ...]
        Forma de otras características (features,)
    config : Dict, opcional
        Configuración del modelo (default: None, usa TCN_CONFIG)
    """
    
    def _initialize_config(self, config: Optional[Dict]) -> None:
        """Inicializa las variables de configuración del modelo."""
        self.config = config if config is not None else TCN_CONFIG
        
        # Obtener parámetros de configuración
        self.filters = self.config['filters']
        self.kernel_size = self.config['kernel_size']
        self.dilations = self.config['dilations']
        self.dropout_rates = self.config['dropout_rate']
        self.use_weight_norm = self.config['use_weight_norm']
        self.use_layer_norm = self.config['use_layer_norm']
        self.use_spatial_dropout = self.config['use_spatial_dropout']
        self.activation = self.config['activation']
        self.epsilon = self.config['epsilon']
    
    def _calculate_dimensions(self, cgm_shape: Tuple[int, ...], other_features_shape: Tuple[int, ...]) -> None:
        """Calcula las dimensiones de entrada para el modelo."""
        self.cgm_timesteps = cgm_shape[0] if len(cgm_shape) > 0 else 1
        self.cgm_features = cgm_shape[1] if len(cgm_shape) > 1 else cgm_shape[0]
        self.other_features = other_features_shape[0] if len(other_features_shape) > 0 else 1
    
    def _create_initial_layers(self) -> None:
        """Crea las capas iniciales de convolución y normalización."""
        # Proyección inicial con Conv1D
        self.initial_conv = nn.Conv1d(
            in_channels=self.cgm_features,
            out_channels=self.filters[0],
            kernel_size=1,
            padding='same'
        )
        
        # Normalización inicial
        if self.use_layer_norm:
            self.initial_norm = nn.LayerNorm(self.filters[0])
        else:
            self.initial_norm = nn.BatchNorm1d(self.filters[0])
        
        # Aplicar normalización de pesos si está habilitada
        if self.use_weight_norm:
            self.initial_conv = apply_weight_norm(self.initial_conv)
    
    def _create_skip_connection(self, in_channels: int) -> nn.Module:
        """Crea una conexión de salto (skip connection)."""
        if in_channels != self.filters[-1]:
            skip_proj = nn.Conv1d(
                in_channels=in_channels,
                out_channels=self.filters[-1],
                kernel_size=1,
                padding='same'
            )
            if self.use_weight_norm:
                skip_proj = apply_weight_norm(skip_proj)
        else:
            skip_proj = nn.Identity()
        
        return skip_proj
    
    def _create_tcn_blocks(self) -> None:
        """Crea los bloques TCN con diferentes tasas de dilatación."""
        self.tcn_blocks = nn.ModuleList()
        self.skip_connections = nn.ModuleList()
        
        current_channels = self.filters[0]
        
        for layer_idx, out_channels in enumerate(self.filters):
            # Bloques dilatados en este nivel
            level_blocks = nn.ModuleList()
            
            for dilation_idx, dilation_rate in enumerate(self.dilations):
                # Crear bloque TCN
                block = TCNResidualBlock(
                    in_channels=current_channels,
                    out_channels=out_channels,
                    kernel_size=self.kernel_size,
                    dilation_rate=dilation_rate,
                    dropout_rate=self.dropout_rates[0],  # Primer valor para bloques TCN
                    use_weight_norm=self.use_weight_norm,
                    use_layer_norm=self.use_layer_norm,
                    use_spatial_dropout=self.use_spatial_dropout,
                    activation=self.activation
                )
                level_blocks.append(block)
                current_channels = out_channels
            
            self.tcn_blocks.append(level_blocks)
            self.skip_connections.append(self._create_skip_connection(out_channels))
    
    def _create_mlp_layers(self) -> None:
        """Crea las capas MLP finales del modelo."""
        # Tamaño combinado después del pooling: features finales + otras características
        combined_size = self.filters[-1] + self.other_features
        
        activation_class = getattr(nn, self.activation.upper()) if hasattr(nn, self.activation.upper()) else nn.GELU
        
        self.mlp_layers = nn.Sequential(
            nn.Linear(combined_size, 128),
            nn.LayerNorm(128) if self.use_layer_norm else nn.BatchNorm1d(128),
            activation_class(),
            nn.Dropout(self.dropout_rates[1]),  # Segundo valor para capas finales
            
            nn.Linear(128, 64),
            nn.LayerNorm(64) if self.use_layer_norm else nn.BatchNorm1d(64),
            activation_class(),
            nn.Dropout(self.dropout_rates[1] / 2),  # Mitad del dropout para la última capa
            
            nn.Linear(64, 1)
        )
    
    def __init__(
        self, 
        cgm_shape: Tuple[int, ...], 
        other_features_shape: Tuple[int, ...],
        config: Optional[Dict] = None
    ) -> None:
        super().__init__()
        
        # Inicializar configuración y dimensiones
        self._initialize_config(config)
        self._calculate_dimensions(cgm_shape, other_features_shape)
        
        # Crear componentes del modelo
        self._create_initial_layers()
        self._create_tcn_blocks()
        
        # Pooling global
        self.global_avg_pool = nn.AdaptiveAvgPool1d(1)
        
        # Crear capas MLP finales
        self._create_mlp_layers()
    
    def forward(self, cgm_input: torch.Tensor, other_input: torch.Tensor) -> torch.Tensor:
        """
        Ejecuta el modelo TCN sobre las entradas.
        
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
        # Preparar la entrada para la convolución 1D
        # TCN espera [batch, channels, length]
        if cgm_input.dim() == 3:
            # Si es [batch, timesteps, features], lo convertimos a [batch, features, timesteps]
            x = cgm_input.permute(0, 2, 1)
        else:
            # Si no tiene 3 dimensiones, añadimos una dimensión de canal
            x = cgm_input.unsqueeze(1)
        
        # Proyección inicial
        x = self.initial_conv(x)
        
        # Normalización inicial
        if self.use_layer_norm:
            # Para layer norm, transponemos para tener [batch, length, channels]
            x_norm = x.transpose(1, 2)
            x_norm = self.initial_norm(x_norm)
            x = x_norm.transpose(1, 2)  # Volvemos a [batch, channels, length]
        else:
            x = self.initial_norm(x)
        
        # Activación inicial
        x = get_activation(x, self.activation)
        
        # Procesar a través de bloques TCN y recopilar skip connections
        skip_outputs = []
        
        for level_idx, level_blocks in enumerate(self.tcn_blocks):
            # Procesar a través de todos los bloques de este nivel
            for block in level_blocks:
                x = block(x)
            
            # Aplicar proyección de skip connection si es necesario y guardar
            skip_out = self.skip_connections[level_idx](x)
            skip_outputs.append(skip_out)
        
        # Combinar skip connections
        if len(skip_outputs) > 1:
            x = sum(skip_outputs)
        else:
            x = skip_outputs[0]
        
        # Activación final antes del pooling
        x = get_activation(x, self.activation)
        
        # Pooling global
        x = self.global_avg_pool(x).squeeze(2)  # [batch, channels]
        
        # Combinar con otras características
        combined = torch.cat([x, other_input], dim=1)
        
        # Pasar por MLP final
        output = self.mlp_layers(combined)
        
        return output


def create_tcn_model(cgm_shape: Tuple[int, ...], other_features_shape: Tuple[int, ...]) -> nn.Module:
    """
    Crea un modelo TCN para regresión.
    
    Parámetros:
    -----------
    cgm_shape : Tuple[int, ...]
        Forma de los datos CGM (timesteps, features)
    other_features_shape : Tuple[int, ...]
        Forma de otras características (features,)
        
    Retorna:
    --------
    nn.Module
        Modelo TCN inicializado
    """
    return TCNModel(cgm_shape, other_features_shape, TCN_CONFIG)


def create_tcn_model_wrapper(cgm_shape: Tuple[int, ...], other_features_shape: Tuple[int, ...]) -> DLModelWrapper:
    """
    Crea un modelo TCN envuelto en DLModelWrapperPyTorch.
    
    Parámetros:
    -----------
    cgm_shape : Tuple[int, ...]
        Forma de los datos CGM (timesteps, features)
    other_features_shape : Tuple[int, ...]
        Forma de otras características (features,)
        
    Retorna:
    --------
    DLModelWrapper
        Modelo TCN envuelto en DLModelWrapper para compatibilidad con el sistema
    """
    # Función creadora del modelo
    model_creator_fn = lambda: create_tcn_model(cgm_shape, other_features_shape)
    
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
    Retorna una función para crear un modelo TCN compatible con la API del sistema.
    
    Retorna:
    --------
    Callable[[Tuple[int, ...], Tuple[int, ...]], DLModelWrapper]
        Función para crear el modelo con las formas de entrada especificadas
    """
    return create_tcn_model_wrapper