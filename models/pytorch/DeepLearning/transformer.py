import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, Tuple, List, Any, Optional, Callable, Union

PROJECT_ROOT = os.path.abspath(os.getcwd())
sys.path.append(PROJECT_ROOT)

from config.models_config import TRANSFORMER_CONFIG
from custom.dl_model_wrapper import DLModelWrapper, DLModelWrapperPyTorch
from models.early_stopping import get_early_stopping_config

# Constantes para cadenas repetidas
CONST_GELU: str = "gelu"
CONST_RELU: str = "relu"
CONST_SELU: str = "selu"
CONST_SIGMOID: str = "sigmoid"
CONST_TANH: str = "tanh"
CONST_EPSILON: str = "epsilon"
CONST_VALID: str = "valid"
CONST_SAME: str = "same"

def get_activation_fn(activation_name: str) -> Callable:
    """
    Obtiene la función de activación correspondiente a un nombre.
    
    Parámetros:
    -----------
    activation_name : str
        Nombre de la función de activación ('relu', 'gelu', 'selu', etc.).
        
    Retorna:
    --------
    Callable
        La función de activación correspondiente.
    """
    if activation_name == CONST_RELU:
        return F.relu
    elif activation_name == CONST_GELU:
        return F.gelu
    elif activation_name == CONST_SELU:
        return F.selu
    elif activation_name == CONST_SIGMOID:
        return torch.sigmoid
    elif activation_name == CONST_TANH:
        return torch.tanh
    else:
        print(f"Advertencia: Función de activación '{activation_name}' no reconocida. Usando ReLU por defecto.")
        return F.relu

class ActivationLayer(nn.Module):
    """
    Capa de activación personalizada que envuelve cualquier función de activación.
    
    Atributos:
    ----------
    activation_fn : Callable
        Función de activación a aplicar.
    """
    
    def __init__(self, activation_fn: Callable) -> None:
        """
        Inicializa la capa de activación.
        
        Parámetros:
        -----------
        activation_fn : Callable
            Función de activación a aplicar.
        """
        super().__init__()
        self.activation_fn = activation_fn
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Aplica la función de activación a la entrada.
        
        Parámetros:
        -----------
        x : torch.Tensor
            Tensor de entrada.
            
        Retorna:
        --------
        torch.Tensor
            Resultado de aplicar la función de activación.
        """
        return self.activation_fn(x)
class PositionEncoding(nn.Module):
    """
    Codificación posicional para el Transformer (implementación PyTorch).
    
    Atributos:
    ----------
    max_position : int
        Posición máxima a codificar.
    d_model : int
        Dimensión del modelo (profundidad de la codificación).
    """
    
    def __init__(self, max_position: int, d_model: int) -> None:
        """
        Inicializa la capa de codificación posicional.
        
        Parámetros:
        -----------
        max_position : int
            Posición máxima a codificar.
        d_model : int
            Dimensión del modelo.
        """
        super().__init__()
        
        # Crear una matriz de codificación posicional utilizando senos y cosenos
        pe = torch.zeros(max_position, d_model)
        position = torch.arange(0, max_position, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        # Registrar el buffer para que sea parte del estado del modelo
        # pero no un parámetro entrenable
        self.register_buffer('pe', pe.unsqueeze(0))
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Aplica la codificación posicional a las entradas.
        
        Parámetros:
        -----------
        x : torch.Tensor
            Tensor de entrada con forma (batch_size, sequence_length, d_model).
            
        Retorna:
        --------
        torch.Tensor
            Tensor con codificación posicional añadida.
        """
        # Asegurar que sólo se utiliza la parte necesaria de la matriz precalculada
        return x + self.pe[:, :x.size(1)]

class TransformerBlock(nn.Module):
    """
    Bloque Transformer con atención multi-cabeza y red feed-forward.
    
    Atributos:
    ----------
    embed_dim : int
        Dimensión de embedding (igual a d_model).
    num_heads : int
        Número de cabezas de atención.
    ff_dim : int
        Dimensión de la capa oculta en la red feed-forward.
    dropout_rate : float
        Tasa de dropout.
    prenorm : bool
        Indica si se usa pre-normalización (LayerNorm antes de subcapas).
    use_bias : bool
        Indica si las capas lineales usan bias.
    epsilon : float
        Epsilon para LayerNormalization.
    activation_fn : Callable
        Función de activación para la red feed-forward.
    """
    
    def __init__(
        self, 
        embed_dim: int, 
        num_heads: int, 
        ff_dim: int, 
        dropout_rate: float,
        prenorm: bool,
        use_bias: bool,
        epsilon: float,
        activation_fn: Callable
    ) -> None:
        """
        Inicializa el bloque Transformer.
        
        Parámetros:
        -----------
        embed_dim : int
            Dimensión de embedding.
        num_heads : int
            Número de cabezas de atención.
        ff_dim : int
            Dimensión de la capa oculta FFN.
        dropout_rate : float
            Tasa de dropout.
        prenorm : bool
            Si usar pre-normalización.
        use_bias : bool
            Si usar bias en capas.
        epsilon : float
            Epsilon para LayerNorm.
        activation_fn : Callable
            Función de activación FFN.
        """
        super().__init__()
        
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.ff_dim = ff_dim
        self.dropout_rate = dropout_rate
        self.prenorm = prenorm
        self.use_bias = use_bias
        self.epsilon = epsilon
        self.activation_fn = activation_fn
        
        # Mecanismo de atención multi-cabeza
        self.mha = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout_rate,
            bias=use_bias,
            batch_first=True  # Asume entradas de forma (batch, seq_len, features)
        )
        
        # Red Feed-Forward
        self.ff_linear1 = nn.Linear(embed_dim, ff_dim, bias=use_bias)
        self.ff_linear2 = nn.Linear(ff_dim, embed_dim, bias=use_bias)
        
        # Normalización
        self.layernorm1 = nn.LayerNorm(embed_dim, eps=epsilon)
        self.layernorm2 = nn.LayerNorm(embed_dim, eps=epsilon)
        
        # Dropout
        self.dropout1 = nn.Dropout(dropout_rate)
        self.dropout2 = nn.Dropout(dropout_rate)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Aplica el bloque Transformer a la entrada.
        
        Parámetros:
        -----------
        x : torch.Tensor
            Tensor de entrada (batch, seq_len, embed_dim).
            
        Retorna:
        --------
        torch.Tensor
            Tensor procesado por el bloque.
        """
        if self.prenorm:
            # Arquitectura Pre-LN
            
            # 1. Multi-Head Attention con pre-normalización
            x_norm = self.layernorm1(x)
            attn_output, _ = self.mha(x_norm, x_norm, x_norm)
            attn_output = self.dropout1(attn_output)
            # Conexión residual
            x1 = x + attn_output
            
            # 2. Feed-Forward con pre-normalización
            x_norm = self.layernorm2(x1)
            ff_output = self.ff_linear1(x_norm)
            ff_output = self.activation_fn(ff_output)
            ff_output = self.dropout2(ff_output)
            ff_output = self.ff_linear2(ff_output)
            ff_output = self.dropout2(ff_output)
            # Conexión residual
            output = x1 + ff_output
            
        else:
            # Arquitectura Post-LN (original)
            
            # 1. Multi-Head Attention con post-normalización
            attn_output, _ = self.mha(x, x, x)
            attn_output = self.dropout1(attn_output)
            x1 = self.layernorm1(x + attn_output)
            
            # 2. Feed-Forward con post-normalización
            ff_output = self.ff_linear1(x1)
            ff_output = self.activation_fn(ff_output)
            ff_output = self.dropout2(ff_output)
            ff_output = self.ff_linear2(ff_output)
            ff_output = self.dropout2(ff_output)
            # Conexión residual y normalización
            output = self.layernorm2(x1 + ff_output)
            
        return output

class TransformerModel(nn.Module):
    """
    Modelo Transformer para datos CGM y otras características.
    
    Atributos:
    ----------
    config : Dict
        Configuración del modelo (TRANSFORMER_CONFIG).
    cgm_shape : Tuple
        Forma de los datos CGM (pasos_temporales, características_cgm).
    other_features_shape : Tuple
        Forma de otras características (características_otras,).
    """
    
    def __init__(self, cgm_shape: Tuple[int, ...], other_features_shape: Tuple[int, ...], config: Optional[Dict] = None) -> None:
        """
        Inicializa el modelo Transformer.
        
        Parámetros:
        -----------
        cgm_shape : Tuple[int, ...]
            Forma de los datos CGM (pasos_temporales, características_cgm).
        other_features_shape : Tuple[int, ...]
            Forma de otras características (características_otras,).
        config : Optional[Dict], opcional
            Configuración del modelo. Si es None, se usa TRANSFORMER_CONFIG (default: None).
        """
        super().__init__()
        
        # Guardar formas de entrada para referencia
        self.cgm_shape = cgm_shape
        self.other_features_shape = other_features_shape
        
        # Usar la configuración proporcionada o la predeterminada
        self.config = config if config is not None else TRANSFORMER_CONFIG
        
        # Extraer parámetros de configuración
        self.embed_dim = self.config['key_dim'] * self.config['num_heads']
        self.num_heads = self.config['num_heads']
        self.ff_dim = self.config['ff_dim']
        self.num_layers = self.config['num_layers']
        self.dropout_rate = self.config['dropout_rate']
        self.prenorm = self.config['prenorm']
        self.use_bias = self.config['use_bias']
        self.epsilon = self.config[CONST_EPSILON]
        self.activation_name = self.config['activation']
        self.activation_fn = get_activation_fn(self.activation_name)
        
        # Capa de proyección inicial para CGM
        self.input_projection = nn.Linear(cgm_shape[-1], self.embed_dim, bias=self.use_bias)
        
        # Codificación posicional (si se usa)
        self.pos_encoding = None
        if self.config['use_relative_pos']:
            self.pos_encoding = PositionEncoding(
                max_position=self.config['max_position'],
                d_model=self.embed_dim
            )
        
        # Dropout después de la proyección/codificación
        self.input_dropout = nn.Dropout(self.dropout_rate)
        
        # Bloques Transformer
        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(
                embed_dim=self.embed_dim,
                num_heads=self.num_heads,
                ff_dim=self.ff_dim,
                dropout_rate=self.dropout_rate,
                prenorm=self.prenorm,
                use_bias=self.use_bias,
                epsilon=self.epsilon,
                activation_fn=self.activation_fn
            ) for _ in range(self.num_layers)
        ])
        
        # Pooling global
        self.global_avg_pool = nn.AdaptiveAvgPool1d(1)
        self.global_max_pool = nn.AdaptiveMaxPool1d(1)
        
        # Calcular dimensión después del pooling concatenado
        pooled_dim = self.embed_dim * 2  # Concatenación de avg_pool y max_pool
        
        # Calcular dimensión después de concatenar con otras características
        if len(other_features_shape) > 1:
            other_flat_dim = np.prod(other_features_shape)
        else:
            other_flat_dim = other_features_shape[0]
        
        combined_dim = pooled_dim + other_flat_dim
        
        # MLP final
        self.mlp_layers = nn.Sequential(
            nn.Linear(combined_dim, 128, bias=self.use_bias),
            nn.LayerNorm(128, eps=self.epsilon),
            ActivationLayer(self.activation_fn),
            nn.Dropout(self.dropout_rate),
            nn.Linear(128, 64, bias=self.use_bias),
            nn.LayerNorm(64, eps=self.epsilon),
            ActivationLayer(self.activation_fn),
            nn.Dropout(self.dropout_rate),
            nn.Linear(64, 1)  # Capa de salida para regresión
        )
    
    def forward(self, cgm_input: torch.Tensor, other_input: torch.Tensor) -> torch.Tensor:
        """
        Aplica el modelo Transformer a las entradas.
        
        Parámetros:
        -----------
        cgm_input : torch.Tensor
            Datos CGM de entrada (batch, time_steps, cgm_features).
        other_input : torch.Tensor
            Otras características de entrada (batch, other_features).
            
        Retorna:
        --------
        torch.Tensor
            Predicciones del modelo (batch, 1).
        """
        # 1. Proyección y Codificación Posicional
        x = self.input_projection(cgm_input)  # (batch, time_steps, embed_dim)
        
        if self.pos_encoding is not None:
            x = self.pos_encoding(x)
        
        # Dropout después de la proyección/codificación
        x = self.input_dropout(x)
        
        # 2. Bloques Transformer
        for block in self.transformer_blocks:
            x = block(x)  # (batch, time_steps, embed_dim)
        
        # 3. Pooling Global
        # Necesitamos transponer para hacer el pooling en la dimensión temporal
        x_transposed = x.transpose(1, 2)  # (batch, embed_dim, time_steps)
        
        # Aplicar global avg pooling y max pooling
        avg_pool = self.global_avg_pool(x_transposed).squeeze(2)  # (batch, embed_dim)
        max_pool = self.global_max_pool(x_transposed).squeeze(2)  # (batch, embed_dim)
        
        # Concatenar los resultados de pooling
        pooled_output = torch.cat([avg_pool, max_pool], dim=1)  # (batch, 2*embed_dim)
        
        # 4. Combinar con otras características
        # Asegurar que other_input sea 2D (batch, features)
        batch_size = other_input.shape[0]
        if other_input.dim() > 2:
            other_input_flat = other_input.reshape(batch_size, -1)
        else:
            other_input_flat = other_input
        
        # Concatenar con el output del transformer
        combined_features = torch.cat([pooled_output, other_input_flat], dim=1)
        
        # 5. MLP final y capa de salida
        output = self.mlp_layers(combined_features)
        
        return output

def create_transformer_model(cgm_shape: Tuple[int, ...], other_features_shape: Tuple[int, ...]) -> nn.Module:
    """
    Crea una instancia del modelo Transformer.
    
    Parámetros:
    -----------
    cgm_shape : Tuple[int, ...]
        Forma de los datos CGM (pasos_temporales, características_cgm).
    other_features_shape : Tuple[int, ...]
        Forma de otras características (características_otras,).
        
    Retorna:
    --------
    nn.Module
        Instancia del modelo Transformer.
    """
    model = TransformerModel(
        cgm_shape=cgm_shape,
        other_features_shape=other_features_shape,
        config=TRANSFORMER_CONFIG
    )
    return model

def create_transformer_model_wrapper(cgm_shape: Tuple[int, ...], other_features_shape: Tuple[int, ...]) -> DLModelWrapper:
    """
    Crea un modelo Transformer envuelto en DLModelWrapper para compatibilidad con el sistema de entrenamiento.
    
    Parámetros:
    -----------
    cgm_shape : Tuple[int, ...]
        Forma de los datos CGM (pasos_temporales, características_cgm).
    other_features_shape : Tuple[int, ...]
        Forma de otras características (características_otras,).
        
    Retorna:
    --------
    DLModelWrapper
        Modelo Transformer envuelto y listo para ser usado por el sistema de entrenamiento.
    """
    # Definir una función creadora que no toma argumentos
    def model_creator_fn() -> nn.Module:
        return create_transformer_model(cgm_shape, other_features_shape)
    
    # Obtener la configuración del early stopping
    es_patience, es_min_delta, es_restore_best = get_early_stopping_config()
    
    # Crear el wrapper con la función creadora
    model_wrapper = DLModelWrapperPyTorch(model_creator=model_creator_fn)
    
    # Añadir early stopping
    model_wrapper.add_early_stopping(
        patience=es_patience,
        min_delta=es_min_delta,
        restore_best_weights=es_restore_best
    )
    
    return model_wrapper

def model_creator() -> Callable[[Tuple[int, ...], Tuple[int, ...]], DLModelWrapper]:
    """
    Retorna la función (`create_transformer_model_wrapper`) que crea el modelo Transformer envuelto.
    
    Esta función (`model_creator`) es la que se importa y se usa en `params.py`.
    No toma argumentos y devuelve la función que sí los toma (`create_transformer_model_wrapper`).
    
    Retorna:
    --------
    Callable[[Tuple[int, ...], Tuple[int, ...]], DLModelWrapper]
        Función (`create_transformer_model_wrapper`) que, dadas las formas de entrada, crea el modelo Transformer envuelto.
    """
    return create_transformer_model_wrapper