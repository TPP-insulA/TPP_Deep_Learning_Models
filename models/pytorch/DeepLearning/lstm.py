import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Dict, List, Any, Callable, Optional, Union

PROJECT_ROOT = os.path.abspath(os.getcwd())
sys.path.append(PROJECT_ROOT)

from config.models_config import LSTM_CONFIG
from custom.dl_model_wrapper import DLModelWrapper, DLModelWrapperPyTorch
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


class LSTMLayer(nn.Module):
    """
    Implementación de una capa LSTM.
    
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
        
        # Usar implementación nativa de LSTM por eficiencia
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=1,
            batch_first=batch_first,
            bidirectional=bidirectional,
            dropout=0.0  # Aplicaremos dropout manualmente
        )
        
        # Dropout separado para aplicar manualmente
        self.dropout_layer = nn.Dropout(dropout) if dropout > 0.0 else None
        
    def forward(self, x: torch.Tensor, h_0: Optional[Tuple[torch.Tensor, torch.Tensor]] = None) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Procesa la entrada a través de la capa LSTM.
        
        Parámetros:
        -----------
        x : torch.Tensor
            Tensor de entrada [batch, seq_len, features] si batch_first=True
        h_0 : Optional[Tuple[torch.Tensor, torch.Tensor]], opcional
            Estado oculto inicial (h_0, c_0) (default: None)
            
        Retorna:
        --------
        Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]
            Secuencia de salidas y estado oculto final (h_n, c_n)
        """
        # Aplicar la capa LSTM
        outputs, (h_n, c_n) = self.lstm(x, h_0)
        
        # Aplicar dropout si está configurado
        if self.dropout_layer is not None:
            outputs = self.dropout_layer(outputs)
            
        return outputs, (h_n, c_n)


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
        
        # Proyecciones para query, key y value
        self.query_proj = nn.Linear(hidden_dim, self.attention_dim)
        self.key_proj = nn.Linear(hidden_dim, self.attention_dim)
        self.value_proj = nn.Linear(hidden_dim, self.attention_dim)
        
    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor) -> torch.Tensor:
        """
        Calcula y aplica la atención.
        
        Parámetros:
        -----------
        query : torch.Tensor
            Tensor de query para atención [batch, len_q, hidden_dim]
        key : torch.Tensor
            Tensor de key para atención [batch, len_k, hidden_dim]
        value : torch.Tensor
            Tensor de value para atención [batch, len_v, hidden_dim]
            
        Retorna:
        --------
        torch.Tensor
            Salida del mecanismo de atención
        """
        # Proyectar query, key y value
        q = self.query_proj(query)  # [batch, len_q, attention_dim]
        k = self.key_proj(key)      # [batch, len_k, attention_dim]
        v = self.value_proj(value)  # [batch, len_v, attention_dim]
        
        # Calcular puntuaciones de atención
        scores = torch.matmul(q, k.transpose(-2, -1))  # [batch, len_q, len_k]
        
        # Escalar puntuaciones
        scores = scores / (self.attention_dim ** 0.5)
        
        # Normalizar puntuaciones
        attention_weights = F.softmax(scores, dim=-1)  # [batch, len_q, len_k]
        
        # Aplicar atención ponderada
        context = torch.matmul(attention_weights, v)  # [batch, len_q, attention_dim]
        
        return context


class LSTMAttentionBlock(nn.Module):
    """
    Bloque LSTM con atención y normalización.
    
    Parámetros:
    -----------
    input_size : int
        Tamaño de entrada
    hidden_size : int
        Tamaño del estado oculto
    bidirectional : bool
        Si usar LSTM bidireccional
    dropout_rate : float
        Tasa de dropout para regularización
    recurrent_dropout : float
        Tasa de dropout para conexiones recurrentes
    epsilon : float
        Valor epsilon para normalización
    num_heads : int
        Número de cabezas de atención
    """
    
    def __init__(
        self, 
        input_size: int, 
        hidden_size: int, 
        bidirectional: bool,
        dropout_rate: float,
        recurrent_dropout: float,
        epsilon: float,
        num_heads: int
    ) -> None:
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.bidirectional = bidirectional
        self.output_size = hidden_size * (2 if bidirectional else 1)
        
        # Crear componentes del bloque
        self.lstm = LSTMLayer(
            input_size=input_size,
            hidden_size=hidden_size,
            bidirectional=bidirectional,
            dropout=dropout_rate,
            recurrent_dropout=recurrent_dropout
        )
        
        # Layer normalization
        self.layer_norm1 = nn.LayerNorm(self.output_size, eps=epsilon)
        
        # Atención multihead simplificada
        self.attention = nn.MultiheadAttention(
            embed_dim=self.output_size,
            num_heads=num_heads,
            dropout=dropout_rate,
            batch_first=True
        )
        
        # Capa para el mecanismo de gating
        self.gate = nn.Linear(self.output_size, self.output_size)
        
        # Layer norm después de atención
        self.layer_norm2 = nn.LayerNorm(self.output_size, eps=epsilon)
        
        # Dropout
        self.dropout = nn.Dropout(dropout_rate)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Procesa la entrada a través del bloque LSTM con atención.
        
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
        residual1 = x if x.size(-1) == self.output_size else None
        
        # Aplicar LSTM
        lstm_out, _ = self.lstm(x)
        
        # Aplicar skip connection y normalización
        if residual1 is not None:
            lstm_out = self.layer_norm1(lstm_out + residual1)
        else:
            lstm_out = self.layer_norm1(lstm_out)
        
        # Guardar para segunda skip connection
        residual2 = lstm_out
        
        # Aplicar atención multi-cabeza
        attn_out, _ = self.attention(lstm_out, lstm_out, lstm_out)
        
        # Aplicar mecanismo de gating
        gate_values = torch.sigmoid(self.gate(residual2))
        attn_out = attn_out * gate_values
        
        # Aplicar segunda skip connection y normalización
        combined = self.layer_norm2(attn_out + residual2)
        
        # Aplicar dropout
        out = self.dropout(combined)
        
        return out


class BidirectionalLSTMBlock(nn.Module):
    """
    Bloque LSTM bidireccional.
    
    Parámetros:
    -----------
    input_size : int
        Tamaño de entrada
    hidden_size : int
        Tamaño del estado oculto
    dropout_rate : float
        Tasa de dropout para regularización
    recurrent_dropout : float
        Tasa de dropout para conexiones recurrentes
    activation : str
        Nombre de la función de activación
    recurrent_activation : str
        Nombre de la función de activación recurrente
    epsilon : float
        Valor epsilon para normalización
    """
    
    def __init__(
        self, 
        input_size: int, 
        hidden_size: int, 
        dropout_rate: float,
        recurrent_dropout: float,
        activation: str,
        recurrent_activation: str,
        epsilon: float
    ) -> None:
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.dropout_rate = dropout_rate
        self.recurrent_dropout = recurrent_dropout
        self.activation = activation
        self.recurrent_activation = recurrent_activation
        self.epsilon = epsilon
        self.output_size = hidden_size * 2  # Bidireccional siempre duplica la salida
        
        # Crear LSTM bidireccional
        self.lstm = LSTMLayer(
            input_size=input_size,
            hidden_size=hidden_size,
            bidirectional=True,
            dropout=dropout_rate,
            recurrent_dropout=recurrent_dropout,
            batch_first=True
        )
        
        # Layer normalization
        self.layer_norm = nn.LayerNorm(self.output_size, eps=epsilon)
        
        # Dropout
        self.dropout_layer = nn.Dropout(dropout_rate)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Procesa la entrada a través del bloque LSTM bidireccional.
        
        Parámetros:
        -----------
        x : torch.Tensor
            Tensor de entrada [batch, seq_len, features]
            
        Retorna:
        --------
        torch.Tensor
            Salida procesada
        """
        # Aplicar LSTM bidireccional
        outputs, _ = self.lstm(x)
        
        # Aplicar normalización
        outputs = self.layer_norm(outputs)
        
        # Aplicar dropout
        outputs = self.dropout_layer(outputs)
        
        return outputs


class LSTMModel(nn.Module):
    """
    Modelo LSTM avanzado con self-attention y conexiones residuales.
    
    Parámetros:
    -----------
    cgm_shape : Tuple[int, ...]
        Forma de los datos CGM
    other_features_shape : Tuple[int, ...]
        Forma de otras características
    """
    
    def __init__(self, cgm_shape: Tuple[int, ...], other_features_shape: Tuple[int, ...]) -> None:
        super().__init__()
        # Obtener parámetros de configuración
        self.hidden_units = LSTM_CONFIG['hidden_units']
        self.dense_units = LSTM_CONFIG['dense_units']
        self.use_bidirectional = LSTM_CONFIG.get(CONST_BIDIRECTIONAL, True)
        self.dropout_rate = LSTM_CONFIG[CONST_DROPOUT]
        self.recurrent_dropout = LSTM_CONFIG[CONST_RECURRENT_DROPOUT]
        self.epsilon = LSTM_CONFIG[CONST_EPSILON]
        self.activation = LSTM_CONFIG['activation']
        self.recurrent_activation = LSTM_CONFIG['recurrent_activation']
        self.dense_activation = LSTM_CONFIG['dense_activation']
        self.attention_heads = LSTM_CONFIG['attention_heads']
        
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
        
        # Bloques LSTM
        self.lstm_blocks = nn.ModuleList()
        curr_size = self.hidden_units[0]
        
        for i, units in enumerate(self.hidden_units):
            # Opción de bidireccional para primeras capas si está configurado
            if i < len(self.hidden_units)-1 and self.use_bidirectional:
                self.lstm_blocks.append(
                    BidirectionalLSTMBlock(
                        input_size=curr_size,
                        hidden_size=units,
                        dropout_rate=self.dropout_rate,
                        recurrent_dropout=self.recurrent_dropout,
                        activation=self.activation,
                        recurrent_activation=self.recurrent_activation,
                        epsilon=self.epsilon
                    )
                )
                curr_size = units * 2  # Bidireccional duplica la salida
            else:
                # Bloques con atención para capas posteriores
                self.lstm_blocks.append(
                    LSTMAttentionBlock(
                        input_size=curr_size,
                        hidden_size=units,
                        bidirectional=self.use_bidirectional,
                        dropout_rate=self.dropout_rate,
                        recurrent_dropout=self.recurrent_dropout,
                        epsilon=self.epsilon,
                        num_heads=self.attention_heads
                    )
                )
                curr_size = units * (2 if self.use_bidirectional else 1)
        
        # Pooling global
        self.global_avg_pool = nn.AdaptiveAvgPool1d(1)
        self.global_max_pool = nn.AdaptiveMaxPool1d(1)
        
        # Tamaño después de concatenar los pools
        pool_size = curr_size * 2
        
        # Capa para combinar con otras características
        combined_size = pool_size + self.other_features
        
        # Primera capa densa
        self.dense1 = nn.Linear(combined_size, self.dense_units[0])
        self.layer_norm1 = nn.LayerNorm(self.dense_units[0], eps=self.epsilon)
        self.dropout1 = nn.Dropout(self.dropout_rate)
        
        # Segunda capa densa con skip connection si es posible
        self.dense2 = nn.Linear(self.dense_units[0], self.dense_units[1])
        self.layer_norm2 = nn.LayerNorm(self.dense_units[1], eps=self.epsilon)
        self.dropout2 = nn.Dropout(self.dropout_rate * 0.5)  # Menor dropout en capas finales
        
        # Capa de salida
        self.output_layer = nn.Linear(self.dense_units[1], 1)
    
    def forward(self, cgm_input: torch.Tensor, other_input: torch.Tensor) -> torch.Tensor:
        """
        Procesa datos CGM y otras características para predecir niveles de insulina.
        
        Parámetros:
        -----------
        cgm_input : torch.Tensor
            Datos CGM [batch, seq_len, features]
        other_input : torch.Tensor
            Otras características [batch, features]
            
        Retorna:
        --------
        torch.Tensor
            Predicciones de dosis de insulina
        """
        # Proyección inicial
        x = self.initial_projection(cgm_input)
        x = self.initial_norm(x)
        
        # Aplicar bloques LSTM
        for lstm_block in self.lstm_blocks:
            x = lstm_block(x)
        
        # Pooling estadístico (media y máximo)
        x_t = x.transpose(1, 2)  # [batch, channels, seq_len]
        avg_pool = self.global_avg_pool(x_t).squeeze(-1)  # [batch, channels]
        max_pool = self.global_max_pool(x_t).squeeze(-1)  # [batch, channels]
        
        # Concatenar pools
        x = torch.cat([avg_pool, max_pool], dim=1)  # [batch, channels*2]
        
        # Combinar con otras características
        x = torch.cat([x, other_input], dim=1)
        
        # Primera capa densa
        skip = x
        x = self.dense1(x)
        x = get_activation(x, self.dense_activation)
        x = self.layer_norm1(x)
        x = self.dropout1(x)
        
        # Segunda capa densa con residual si es posible
        skip = x  # Nueva skip connection
        x = self.dense2(x)
        x = get_activation(x, self.dense_activation)
        
        # Skip connection si las dimensiones coinciden
        if skip.shape[-1] == x.shape[-1]:
            x = x + skip
            
        x = self.layer_norm2(x)
        x = self.dropout2(x)
        
        # Capa de salida
        output = self.output_layer(x)
        
        return output


def create_lstm_model(cgm_shape: Tuple[int, ...], other_features_shape: Tuple[int, ...]) -> nn.Module:
    """
    Crea un modelo LSTM para predicción.
    
    Parámetros:
    -----------
    cgm_shape : Tuple[int, ...]
        Forma de los datos CGM (timesteps, features)
    other_features_shape : Tuple[int, ...]
        Forma de otras características (features,)
        
    Retorna:
    --------
    nn.Module
        Modelo LSTM inicializado
    """
    return LSTMModel(cgm_shape, other_features_shape)


def create_lstm_model_wrapper(cgm_shape: Tuple[int, ...], other_features_shape: Tuple[int, ...]) -> DLModelWrapper:
    """
    Crea un modelo LSTM envuelto en DLModelWrapperPyTorch.
    
    Parámetros:
    -----------
    cgm_shape : Tuple[int, ...]
        Forma de los datos CGM (timesteps, features)
    other_features_shape : Tuple[int, ...]
        Forma de otras características (features,)
        
    Retorna:
    --------
    DLModelWrapper
        Modelo LSTM envuelto en DLModelWrapper
    """
    # Función creadora del modelo
    model_creator_fn = lambda: create_lstm_model(cgm_shape, other_features_shape)
    
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
    Retorna una función para crear un modelo LSTM compatible con la API del sistema.
    
    Retorna:
    --------
    Callable[[Tuple[int, ...], Tuple[int, ...]], DLModelWrapper]
        Función para crear el modelo con las formas de entrada especificadas
    """
    return create_lstm_model_wrapper