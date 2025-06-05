import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Dict, Any, Optional, Callable, Union, List

PROJECT_ROOT = os.path.abspath(os.getcwd())
sys.path.append(PROJECT_ROOT) 

from config.models_config import ATTENTION_CONFIG, EARLY_STOPPING_POLICY
from custom.DeepLearning.dl_pt import DLModelWrapperPyTorch
from models_old.early_stopping import get_early_stopping_config
from custom.printer import print_debug, print_info

# Constantes para uso repetido
CONST_EMBEDDING = "embedding"
CONST_LAYER_NORM = "layer_norm"
CONST_DENSE = "dense"
CONST_DEVICE = "device"

class RelativePositionEncoding(nn.Module):
    """
    Codificación de posición relativa para mejorar la atención temporal.
    
    Parámetros:
    -----------
    max_position : int
        Posición máxima a codificar
    depth : int
        Profundidad de la codificación
    """
    def __init__(self, max_position: int, depth: int) -> None:
        super().__init__()
        self.max_position = max_position
        self.depth = depth
        self.rel_embeddings = nn.Parameter(torch.randn(2 * max_position - 1, depth) * 0.02)
    
    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """
        Aplica la codificación de posición relativa a las entradas.
        
        Parámetros:
        -----------
        inputs : torch.Tensor
            Tensor de entrada
            
        Retorna:
        --------
        torch.Tensor
            Tensor de codificación de posición
        """
        length = inputs.size(1)
        pos_range = torch.arange(length, device=inputs.device)
        pos_indices = pos_range.unsqueeze(1) - pos_range.unsqueeze(0) + self.max_position - 1
        return self.rel_embeddings[pos_indices]

class AttentionBlock(nn.Module):
    """
    Bloque de atención con soporte para codificación posicional.
    
    Parámetros:
    -----------
    num_heads : int
        Número de cabezas de atención
    key_dim : int
        Dimensión de las claves
    config : Optional[Dict], opcional
        Configuración adicional del bloque (default: None)
    """
    def __init__(self, num_heads: int, key_dim: int, config: Optional[Dict] = None) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.key_dim = key_dim
        self.config = config or ATTENTION_CONFIG
        head_size = self.config['head_size'] if self.config.get('head_size') is not None else key_dim
        
        # Proyecciones para query, key, value
        self.query_projection = nn.Linear(self.config['embed_dim'], num_heads * key_dim)
        self.key_projection = nn.Linear(self.config['embed_dim'], num_heads * key_dim)
        self.value_projection = nn.Linear(self.config['embed_dim'], num_heads * head_size)
        
        # Proyección final
        self.output_projection = nn.Linear(num_heads * head_size, self.config['embed_dim'])
        
        # Compuerta
        self.gate = nn.Linear(self.config['embed_dim'], self.config['embed_dim'])
        
        # Capas feed-forward
        self.ff_layer1 = nn.Linear(self.config['embed_dim'], self.config['ff_dim'])
        self.ff_gate = nn.Linear(self.config['embed_dim'], self.config['ff_dim'])
        self.ff_layer2 = nn.Linear(self.config['ff_dim'], self.config['embed_dim'])
        
        # Normalización
        self.norm1 = nn.LayerNorm(self.config['embed_dim'], eps=1e-6)
        self.norm2 = nn.LayerNorm(self.config['embed_dim'], eps=1e-6)
        
        # Dropout
        self.dropout = nn.Dropout(self.config['dropout_rate'])
        self.head_size = head_size
    
    def forward(self, x: torch.Tensor, pos_encoding: torch.Tensor, training: bool = False) -> torch.Tensor:
        """
        Aplica el mecanismo de atención a la entrada.
        
        Parámetros:
        -----------
        x : torch.Tensor
            Tensor de entrada
        pos_encoding : torch.Tensor
            Codificación posicional
        training : bool, opcional
            Si está en modo entrenamiento (default: False)
            
        Retorna:
        --------
        torch.Tensor
            Resultado del bloque de atención
        """
        # Implementar atención con codificación posicional relativa
        batch_size, seq_len, _ = x.size()
        # residual = x
        
        if self.config['use_relative_attention']:
            # Proyecciones
            query = self.query_projection(x).view(batch_size, seq_len, self.num_heads, self.key_dim)
            key = self.key_projection(x).view(batch_size, seq_len, self.num_heads, self.key_dim)
            value = self.value_projection(x).view(batch_size, seq_len, self.num_heads, self.head_size)
            
            # Reorganizar dimensiones para cálculo de atención
            query = query.permute(0, 2, 1, 3)  # [batch, heads, seq_len, key_dim]
            key = key.permute(0, 2, 3, 1)  # [batch, heads, key_dim, seq_len]
            value = value.permute(0, 2, 1, 3)  # [batch, heads, seq_len, head_size]
            
            # Calcular puntuaciones de atención
            scale = torch.sqrt(torch.tensor(self.key_dim, dtype=torch.float32, device=x.device))
            attention_scores = torch.matmul(query, key) / scale
            
            # Adaptar codificación posicional
            if pos_encoding is not None:
                pos_encoding_shape = pos_encoding.size()
                
                if len(pos_encoding_shape) == 3:
                    # Reducir la dimensión de características tomando la media
                    pos_encoding_reduced = torch.mean(pos_encoding[:seq_len, :seq_len, :], dim=-1)
                    # Expandir para hacer broadcasting con attention_scores
                    pos_encoding_final = pos_encoding_reduced.unsqueeze(0).unsqueeze(1).expand_as(attention_scores)
                    attention_scores = attention_scores + pos_encoding_final
            
            # Aplicar softmax
            attention_probs = F.softmax(attention_scores, dim=-1)
            attention_probs = self.dropout(attention_probs) if training else attention_probs
            
            # Obtener el resultado de la atención
            context = torch.matmul(attention_probs, value)  # [batch, heads, seq_len, head_size]
            context = context.permute(0, 2, 1, 3).contiguous().view(batch_size, seq_len, -1)
            
            # Proyección final
            attention_output = self.output_projection(context)
        else:
            # Implementación regular de multi-head attention
            query = self.query_projection(x).view(batch_size, seq_len, self.num_heads, self.key_dim)
            key = self.key_projection(x).view(batch_size, seq_len, self.num_heads, self.key_dim)
            value = self.value_projection(x).view(batch_size, seq_len, self.num_heads, self.head_size)
            
            query = query.permute(0, 2, 1, 3)
            key = key.permute(0, 2, 3, 1)
            value = value.permute(0, 2, 1, 3)
            
            attention_scores = torch.matmul(query, key) / torch.sqrt(torch.tensor(self.key_dim, dtype=torch.float32, device=x.device))
            attention_probs = F.softmax(attention_scores, dim=-1)
            attention_probs = self.dropout(attention_probs) if training else attention_probs
            
            context = torch.matmul(attention_probs, value)
            context = context.permute(0, 2, 1, 3).contiguous().view(batch_size, seq_len, -1)
            attention_output = self.output_projection(context)
        
        # Mecanismo de compuerta
        gate_values = torch.sigmoid(self.gate(x))
        attention_output = gate_values * attention_output
        
        # Aplicar dropout y primera normalización residual
        attention_output = self.dropout(attention_output) if training else attention_output
        x = self.norm1(x + attention_output)
        
        # Feed-forward con GLU (Gated Linear Unit)
        ff = self.ff_layer1(x)
        ff_gate = torch.sigmoid(self.ff_gate(x))
        ff = ff * ff_gate
        ff = self.ff_layer2(ff)
        ff = self.dropout(ff) if training else ff
        
        # Segunda normalización residual
        return self.norm2(x + ff)


class AttentionModel(nn.Module):
    """
    Modelo basado únicamente en mecanismos de atención.
    
    Parámetros:
    -----------
    config : Dict
        Diccionario con la configuración del modelo
    cgm_shape : Tuple
        Forma de los datos CGM
    other_features_shape : Tuple
        Forma de otras características
    """
    def __init__(self, config: Dict, cgm_shape: Tuple, other_features_shape: Tuple) -> None:
        super().__init__()
        self.config = config
        self.cgm_shape = cgm_shape
        self.other_features_shape = other_features_shape
        
        # Definir dimensiones
        self.dropout_rate = config['dropout_rate']
        
        # Codificación posicional relativa
        self.pos_encoder = RelativePositionEncoding(
            config['max_relative_position'],
            config['key_dim']
        )
        
        # Bloques de atención
        self.attention_blocks = nn.ModuleList([
            AttentionBlock(
                num_heads=config['num_heads'],
                key_dim=config['key_dim'],
                config=config
            )
            for _ in range(config['num_blocks'])
        ])
        
        # Proyección para entrada CGM
        self.cgm_projection = nn.Linear(cgm_shape[-1], config['embed_dim'])
        
        # Proyección para otras características si existen
        if other_features_shape[0] > 0:
            self.other_projection = nn.Linear(other_features_shape[0], config['embed_dim'])
        
        # Capas finales para predicción
        final_layers = []
        input_dim = config['embed_dim'] * 2 if other_features_shape[0] > 0 else config['embed_dim']
        
        for units in config['dense_units']:
            final_layers.append(nn.Linear(input_dim, units))
            final_layers.append(nn.LayerNorm(units, eps=1e-6))
            final_layers.append(nn.Dropout(config['dropout_rate']))
            final_layers.append(self._get_activation_layer(config['activation']))
            input_dim = units
        
        final_layers.append(nn.Linear(input_dim, 1))
        self.final_layers = nn.Sequential(*final_layers)
    
    def _get_activation_layer(self, activation_name: str) -> nn.Module:
        """
        Devuelve la capa de activación según su nombre.
        
        Parámetros:
        -----------
        activation_name : str
            Nombre de la función de activación
            
        Retorna:
        --------
        nn.Module
            Capa de activación correspondiente
        """
        if activation_name == 'relu':
            return nn.ReLU()
        elif activation_name == 'gelu':
            return nn.GELU()
        elif activation_name == 'swish' or activation_name == 'silu':
            return nn.SiLU()
        else:
            return nn.ReLU()  # Valor por defecto
    
    def forward(self, x_cgm: torch.Tensor, x_other: torch.Tensor) -> torch.Tensor:
        """
        Ejecuta el modelo de atención con los datos de entrada.
        
        Parámetros:
        -----------
        x_cgm : torch.Tensor
            Datos CGM con forma (batch_size, seq_length, features)
        x_other : torch.Tensor
            Otras características con forma (batch_size, features)
            
        Retorna:
        --------
        torch.Tensor
            Predicciones del modelo
        """
        # Proyectar entrada CGM
        x = self.cgm_projection(x_cgm)
        
        # Obtener codificación posicional
        pos_encoding = self.pos_encoder(x)
        
        # Aplicar bloques de atención
        for attention_block in self.attention_blocks:
            x = attention_block(x, pos_encoding, self.training)
        
        # Reducción para salida (media temporal)
        x = torch.mean(x, dim=1)
        
        # Procesar características adicionales si existen
        if x_other is not None and x_other.size(-1) > 0 and hasattr(self, 'other_projection'):
            x_other_projected = self.other_projection(x_other)
            x = torch.cat([x, x_other_projected], dim=-1)
        
        # Aplicar capas finales - solo llamar una vez
        x = self.final_layers(x)
        
        # Mantener la forma [batch_size, 1]
        return x


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
    if activation_name == 'relu':
        return F.relu(x)
    elif activation_name == 'gelu':
        return F.gelu(x)
    elif activation_name == 'swish' or activation_name == 'silu':
        return F.silu(x)
    else:
        return F.relu(x)  # Valor por defecto

def create_attention_model(cgm_shape: Tuple[int, ...], other_features_shape: Tuple[int, ...]) -> AttentionModel:
    """
    Crea un modelo basado en mecanismos de atención.
    
    Parámetros:
    -----------
    cgm_shape : Tuple[int, ...]
        Forma de los datos CGM (pasos_temporales, características)
    other_features_shape : Tuple[int, ...]
        Forma de otras características (características)
        
    Retorna:
    --------
    AttentionModel
        Modelo de atención configurado
    """
    return AttentionModel(
        config=ATTENTION_CONFIG,
        cgm_shape=cgm_shape,
        other_features_shape=other_features_shape
    )


def create_attention_model_wrapper(cgm_shape: Tuple[int, ...], other_features_shape: Tuple[int, ...]) -> DLModelWrapperPyTorch:
    """
    Crea un modelo de atención envuelto en DLModelWrapperPyTorch.
    
    Parámetros:
    -----------
    cgm_shape : Tuple[int, ...]
        Forma de los datos CGM (pasos_temporales, características)
    other_features_shape : Tuple[int, ...]
        Forma de otras características (características)
        
    Retorna:
    --------
    DLModelWrapperPyTorch
        Modelo envuelto en DLModelWrapperPyTorch para compatibilidad con el sistema
    """
    # Definir creador del modelo
    def model_creator() -> AttentionModel:
        """
        Crea una instancia del modelo de atención.
        
        Retorna:
        --------
        AttentionModel
            Instancia del modelo de atención
        """
        return create_attention_model(cgm_shape, other_features_shape)
    
    # Crear wrapper
    wrapper = DLModelWrapperPyTorch(model_creator)
    
    # Configurar early stopping si está habilitado
    es_patience, es_min_delta, es_restore_best = get_early_stopping_config()
    if EARLY_STOPPING_POLICY.get('early_stopping', False):
        wrapper.add_early_stopping(
            patience=es_patience,
            min_delta=es_min_delta,
            restore_best_weights=es_restore_best
        )
    
    return wrapper


def model_creator() -> Callable[[Tuple[int, ...], Tuple[int, ...]], DLModelWrapperPyTorch]:
    """
    Retorna una función para crear un modelo de atención compatible con la API del sistema.
    
    Retorna:
    --------
    Callable[[Tuple[int, ...], Tuple[int, ...]], DLModelWrapperPyTorch]
        Función para crear el modelo con las formas de entrada especificadas
    """
    return create_attention_model_wrapper