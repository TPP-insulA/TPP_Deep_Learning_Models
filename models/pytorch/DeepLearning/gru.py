import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Dict, List, Any, Callable, Optional, Union

PROJECT_ROOT = os.path.abspath(os.getcwd())
sys.path.append(PROJECT_ROOT)

from config.models_config import GRU_CONFIG
from custom.DeepLearning.dl_model_wrapper import DLModelWrapper
from custom.DeepLearning.dl_pt import DLModelWrapperPyTorch
from models.early_stopping import get_early_stopping_config

# Constantes para cadenas repetidas
CONST_GELU = "gelu"
CONST_RELU = "relu"
CONST_TANH = "tanh"
CONST_SIGMOID = "sigmoid"
CONST_DROPOUT = "dropout_rate"
CONST_BIDIRECTIONAL = "use_bidirectional"
CONST_EPSILON = "epsilon"
CONST_RECURRENT_DROPOUT = "recurrent_dropout"

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
    elif activation_name == CONST_TANH:
        return torch.tanh(x)
    elif activation_name == CONST_SIGMOID:
        return torch.sigmoid(x)
    else:
        return F.relu(x)  # Valor predeterminado


class GRULayer(nn.Module):
    """
    Implementación de una capa GRU.
    
    Parámetros:
    -----------
    input_size : int
        Tamaño de la entrada
    hidden_size : int
        Tamaño del estado oculto
    bidirectional : bool, opcional
        Si la capa es bidireccional (default: False)
    dropout : float, opcional
        Tasa de dropout para regularización (default: 0.0)
    recurrent_dropout : float, opcional
        Tasa de dropout para conexiones recurrentes (default: 0.0)
    batch_first : bool, opcional
        Si el primer índice de los datos es el lote (default: True)
    """
    
    def __init__(
        self, 
        input_size: int, 
        hidden_size: int, 
        bidirectional: bool = False, 
        dropout: float = 0.0,
        recurrent_dropout: float = 0.0,
        batch_first: bool = True
    ) -> None:
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.bidirectional = bidirectional
        self.dropout = dropout
        self.recurrent_dropout = recurrent_dropout
        self.batch_first = batch_first
        
        # Usar implementación nativa de GRU por eficiencia
        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=1,
            batch_first=batch_first,
            bidirectional=bidirectional,
            dropout=0.0  # Aplicaremos dropout manualmente
        )
        
        # Dropout separado para aplicar manualmente
        self.dropout_layer = nn.Dropout(dropout) if dropout > 0.0 else None
        
    def forward(self, x: torch.Tensor, h_0: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Procesa la entrada a través de la capa GRU.
        
        Parámetros:
        -----------
        x : torch.Tensor
            Tensor de entrada [batch, seq_len, features] si batch_first=True
        h_0 : Optional[torch.Tensor], opcional
            Estado oculto inicial (default: None)
            
        Retorna:
        --------
        Tuple[torch.Tensor, torch.Tensor]
            Secuencia de salidas y estado oculto final
        """
        # Aplicar la capa GRU
        outputs, h_n = self.gru(x, h_0)
        
        # Aplicar dropout si está configurado
        if self.dropout_layer is not None:
            outputs = self.dropout_layer(outputs)
            
        return outputs, h_n


class AttentionLayer(nn.Module):
    """
    Mecanismo de atención para secuencias.
    
    Parámetros:
    -----------
    hidden_dim : int
        Dimensión del espacio oculto
    attention_dim : int, opcional
        Dimensión interna de la atención (default: None)
    """
    
    def __init__(self, hidden_dim: int, attention_dim: Optional[int] = None) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.attention_dim = attention_dim or hidden_dim
        
        # Proyecciones para query y value
        self.query_proj = nn.Linear(hidden_dim, self.attention_dim)
        self.value_proj = nn.Linear(hidden_dim, self.attention_dim)
        
    def forward(self, query: torch.Tensor, value: torch.Tensor) -> torch.Tensor:
        """
        Calcula y aplica la atención.
        
        Parámetros:
        -----------
        query : torch.Tensor
            Tensor de query para atención [batch, ..., hidden_dim]
        value : torch.Tensor
            Tensor de value para atención [batch, ..., hidden_dim]
            
        Retorna:
        --------
        torch.Tensor
            Salida del mecanismo de atención
        """
        # Proyectar query y value
        q = self.query_proj(query)  # [batch, len_q, attention_dim]
        v = self.value_proj(value)  # [batch, len_v, attention_dim]
        
        # Calcular puntuaciones de atención
        scores = torch.matmul(q, v.transpose(-2, -1))  # [batch, len_q, len_v]
        
        # Normalizar puntuaciones
        attention_weights = F.softmax(scores, dim=-1)  # [batch, len_q, len_v]
        
        # Aplicar atención ponderada
        context = torch.matmul(attention_weights, v)  # [batch, len_q, attention_dim]
        
        return context


def create_gru_attention_block(
    x: torch.Tensor, 
    units: int, 
    bidirectional: bool = True, 
    dropout_rate: float = 0.2, 
    recurrent_dropout: float = 0.1, 
    training: bool = True
) -> torch.Tensor:
    """
    Crea un bloque GRU con mecanismo de atención.
    
    Parámetros:
    -----------
    x : torch.Tensor
        Tensor de entrada [batch, seq_len, features]
    units : int
        Número de unidades GRU
    bidirectional : bool, opcional
        Si usar GRU bidireccional (default: True)
    dropout_rate : float, opcional
        Tasa de dropout para regularización (default: 0.2)
    recurrent_dropout : float, opcional
        Tasa de dropout para conexiones recurrentes (default: 0.1)
    training : bool, opcional
        Si está en modo entrenamiento (default: True)
        
    Retorna:
    --------
    torch.Tensor
        Tensor procesado con GRU y atención
    """
    # Identificar tamaño de entrada
    input_size = x.size(-1)
    
    # GRU layer
    gru_layer = GRULayer(
        input_size=input_size,
        hidden_size=units,
        bidirectional=bidirectional,
        dropout=dropout_rate if training else 0.0,
        recurrent_dropout=recurrent_dropout if training else 0.0,
        batch_first=True
    )
    
    # Ejecutar la capa GRU
    gru_output, _ = gru_layer(x)
    
    # Skip connection con normalización de capa si las dimensiones coinciden
    output_dim = units * (2 if bidirectional else 1)
    if input_size == output_dim:
        # Normalización con skip connection
        skip_connection = x + gru_output
        gru_output = nn.LayerNorm(output_dim, eps=GRU_CONFIG[CONST_EPSILON])(skip_connection)
    else:
        # Solo normalización
        gru_output = nn.LayerNorm(output_dim, eps=GRU_CONFIG[CONST_EPSILON])(gru_output)
    
    # Mecanismo de atención
    attention_layer = AttentionLayer(hidden_dim=output_dim)
    attention_output = attention_layer(gru_output, gru_output)
    
    # Combinar atención con skip connection
    combined = attention_output + gru_output
    combined = nn.LayerNorm(output_dim, eps=GRU_CONFIG[CONST_EPSILON])(combined)
    
    # Dropout final
    return nn.Dropout(dropout_rate)(combined) if training else combined


class GRUModel(nn.Module):
    """
    Modelo GRU con mecanismo de atención para series temporales.
    
    Parámetros:
    -----------
    cgm_shape : Tuple[int, ...]
        Forma de los datos CGM (timesteps, features)
    other_features_shape : Tuple[int, ...]
        Forma de otras características (features,)
    """
    
    def __init__(self, cgm_shape: Tuple[int, ...], other_features_shape: Tuple[int, ...]) -> None:
        super().__init__()
        # Obtener parámetros de configuración
        self.hidden_units = GRU_CONFIG['hidden_units']
        self.use_bidirectional = GRU_CONFIG.get(CONST_BIDIRECTIONAL, True)
        self.dropout_rate = GRU_CONFIG[CONST_DROPOUT]
        self.recurrent_dropout = GRU_CONFIG[CONST_RECURRENT_DROPOUT]
        self.epsilon = GRU_CONFIG[CONST_EPSILON]
        
        # Obtener dimensiones de entrada
        if len(cgm_shape) >= 2:
            self.cgm_timesteps, self.cgm_features = cgm_shape
        else:
            self.cgm_timesteps, self.cgm_features = 1, cgm_shape[0]
            
        if len(other_features_shape) > 0:
            self.other_features = other_features_shape[0]
        else:
            self.other_features = 1
        
        # Proyección inicial
        self.initial_projection = nn.Linear(self.cgm_features, self.hidden_units[0])
        self.initial_norm = nn.LayerNorm(self.hidden_units[0], eps=self.epsilon)
        
        # Bloques GRU con atención
        self.gru_blocks = nn.ModuleList()
        curr_size = self.hidden_units[0]
        
        for units in self.hidden_units:
            # Crear un bloque GRU funcional
            self.gru_blocks.append(
                _GRUBlock(
                    input_size=curr_size,
                    hidden_size=units,
                    bidirectional=self.use_bidirectional,
                    dropout_rate=self.dropout_rate,
                    recurrent_dropout=self.recurrent_dropout,
                    epsilon=self.epsilon
                )
            )
            # Actualizar tamaño para el siguiente bloque
            curr_size = units * (2 if self.use_bidirectional else 1)
        
        # Pooling global
        self.global_avg_pool = nn.AdaptiveAvgPool1d(1)
        
        # Capa para combinar con otras características
        combined_size = curr_size + self.other_features
        
        # MLP Final con skip connections
        self.dense_layers = nn.ModuleList()
        self.layer_norms = nn.ModuleList()
        self.dropouts = nn.ModuleList()
        
        dense_units = [128, 64]
        curr_size = combined_size
        
        for units in dense_units:
            # Skip connection si las dimensiones coinciden
            self.has_skip = curr_size == units
            
            # Capas para el bloque
            self.dense_layers.append(nn.Linear(curr_size, units))
            self.layer_norms.append(nn.LayerNorm(units, eps=self.epsilon))
            self.dropouts.append(nn.Dropout(self.dropout_rate))
            
            # Actualizar tamaño
            curr_size = units
        
        # Capa de salida
        self.output_layer = nn.Linear(curr_size, 1)
    
    def forward(self, x_cgm: torch.Tensor, x_other: torch.Tensor) -> torch.Tensor:
        """
        Procesa datos CGM y otras características para predecir niveles de insulina.
        
        Parámetros:
        -----------
        x_cgm : torch.Tensor
            Datos CGM [batch, seq_len, features]
        x_other : torch.Tensor
            Otras características [batch, features]
            
        Retorna:
        --------
        torch.Tensor
            Predicciones de dosis de insulina
        """
        # Proyección inicial
        x = self.initial_projection(x_cgm)
        x = self.initial_norm(x)
        
        # Aplicar bloques GRU
        for gru_block in self.gru_blocks:
            x = gru_block(x)
        
        # Pooling global
        x = x.transpose(1, 2)  # [batch, channels, seq_len]
        x = self.global_avg_pool(x).squeeze(-1)  # [batch, channels]
        
        # Combinar con otras características
        combined = torch.cat([x, x_other], dim=1)
        
        # Aplicar capas finales con skip connections
        x = combined
        for i, (dense, norm, dropout) in enumerate(zip(self.dense_layers, self.layer_norms, self.dropouts)):
            # Guardar para skip connection
            skip = x if x.size(1) == dense.out_features else None
            
            # Aplicar capa densa
            x = dense(x)
            x = get_activation(x, CONST_RELU)
            x = norm(x)
            x = dropout(x)
            
            # Aplicar skip connection si es posible
            if skip is not None:
                x = x + skip
        
        # Capa de salida
        output = self.output_layer(x)
        
        return output


class _GRUBlock(nn.Module):
    """
    Bloque GRU con atención y normalización.
    
    Parámetros:
    -----------
    input_size : int
        Tamaño de entrada
    hidden_size : int
        Tamaño del estado oculto
    bidirectional : bool
        Si usar GRU bidireccional
    dropout_rate : float
        Tasa de dropout para regularización
    recurrent_dropout : float
        Tasa de dropout para conexiones recurrentes
    epsilon : float
        Valor epsilon para normalización
    """
    
    def __init__(
        self, 
        input_size: int, 
        hidden_size: int, 
        bidirectional: bool,
        dropout_rate: float,
        recurrent_dropout: float,
        epsilon: float
    ) -> None:
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.bidirectional = bidirectional
        self.output_size = hidden_size * (2 if bidirectional else 1)
        
        # Crear componentes del bloque
        self.gru = GRULayer(
            input_size=input_size,
            hidden_size=hidden_size,
            bidirectional=bidirectional,
            dropout=dropout_rate,
            recurrent_dropout=recurrent_dropout
        )
        
        # Layer normalization
        self.layer_norm1 = nn.LayerNorm(self.output_size, eps=epsilon)
        
        # Atención
        self.attention = AttentionLayer(hidden_dim=self.output_size)
        
        # Layer norm después de atención
        self.layer_norm2 = nn.LayerNorm(self.output_size, eps=epsilon)
        
        # Dropout
        self.dropout = nn.Dropout(dropout_rate)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Procesa la entrada a través del bloque GRU.
        
        Parámetros:
        -----------
        x : torch.Tensor
            Tensor de entrada [batch, seq_len, features]
            
        Retorna:
        --------
        torch.Tensor
            Salida procesada
        """
        # Guardar entrada para skip connection
        residual = x if x.size(-1) == self.output_size else None
        
        # Aplicar GRU
        gru_out, _ = self.gru(x)
        
        # Aplicar skip connection y normalización
        if residual is not None:
            gru_out = self.layer_norm1(gru_out + residual)
        else:
            gru_out = self.layer_norm1(gru_out)
        
        # Aplicar atención
        att_out = self.attention(gru_out, gru_out)
        
        # Aplicar segunda skip connection y normalización
        combined = self.layer_norm2(att_out + gru_out)
        
        # Aplicar dropout
        out = self.dropout(combined)
        
        return out


def create_gru_model(cgm_shape: Tuple[int, ...], other_features_shape: Tuple[int, ...]) -> nn.Module:
    """
    Crea un modelo GRU para predicción.
    
    Parámetros:
    -----------
    cgm_shape : Tuple[int, ...]
        Forma de los datos CGM (timesteps, features)
    other_features_shape : Tuple[int, ...]
        Forma de otras características (features,)
        
    Retorna:
    --------
    nn.Module
        Modelo GRU inicializado
    """
    return GRUModel(cgm_shape, other_features_shape)


def create_gru_model_wrapper(cgm_shape: Tuple[int, ...], other_features_shape: Tuple[int, ...]) -> DLModelWrapper:
    """
    Crea un modelo GRU envuelto en DLModelWrapperPyTorch.
    
    Parámetros:
    -----------
    cgm_shape : Tuple[int, ...]
        Forma de los datos CGM (timesteps, features)
    other_features_shape : Tuple[int, ...]
        Forma de otras características (features,)
        
    Retorna:
    --------
    DLModelWrapper
        Modelo GRU envuelto en DLModelWrapper
    """
    # Función creadora del modelo
    model_creator_fn = lambda: create_gru_model(cgm_shape, other_features_shape)
    
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
    Retorna una función para crear un modelo GRU compatible con la API del sistema.
    
    Retorna:
    --------
    Callable[[Tuple[int, ...], Tuple[int, ...]], DLModelWrapper]
        Función para crear el modelo con las formas de entrada especificadas
    """
    return create_gru_model_wrapper