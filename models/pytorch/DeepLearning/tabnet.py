import os
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Callable, Optional, Any, Union

PROJECT_ROOT = os.path.abspath(os.getcwd())
sys.path.append(PROJECT_ROOT)

from config.models_config import TABNET_CONFIG
from custom.DeepLearning.dl_model_wrapper import DLModelWrapper
from custom.DeepLearning.dl_pt import DLModelWrapperPyTorch
from models.early_stopping import get_early_stopping_config

# Constantes para cadenas repetidas
CONST_TANH = "tanh"
CONST_SIGMOID = "sigmoid"
CONST_FEATURE_MASK = "feature_mask"
CONST_GATED_FEATURE_TRANSFORM = "gated_feature_transform"
CONST_STEP = "step"
CONST_ATTENTION = "attention"
CONST_OUTPUT = "output"

class GatedLinearUnit(nn.Module):
    """
    Implementación de Unidad Lineal con Compuerta (GLU) como capa personalizada.
    
    Parámetros:
    -----------
    input_dim : int
        Dimensión de entrada
    output_dim : int
        Dimensión de salida
    activation : str, opcional
        Activación para la compuerta (default: "sigmoid")
    """
    
    def __init__(self, input_dim: int, output_dim: int, activation: str = CONST_SIGMOID) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.activation = activation
        self.fc = nn.Linear(input_dim, 2 * output_dim)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Aplica la transformación GLU a las entradas.
        
        Parámetros:
        -----------
        x : torch.Tensor
            Tensor de entrada
            
        Retorna:
        --------
        torch.Tensor
            Salida transformada
        """
        x = self.fc(x)
        x_linear, x_gated = torch.split(x, self.output_dim, dim=-1)
        
        # Aplicar activación a la parte de compuerta
        if self.activation == CONST_SIGMOID:
            gate = torch.sigmoid(x_gated)
        else:
            gate = torch.tanh(x_gated)
            
        # Multiplicar la parte lineal con la compuerta
        return x_linear * gate


class GhostBatchNorm(nn.Module):
    """
    Implementación de Normalización por Lotes Fantasma para conjuntos de datos pequeños.
    
    Parámetros:
    -----------
    input_dim : int
        Dimensión de entrada
    virtual_batch_size : int, opcional
        Tamaño de lote virtual para normalización (default: None)
    momentum : float, opcional
        Momentum para actualización de estadísticas (default: 0.9)
    epsilon : float, opcional
        Valor para estabilidad numérica (default: 1e-5)
    """
    
    def __init__(self, input_dim: int, virtual_batch_size: Optional[int] = None, 
                 momentum: float = 0.9, epsilon: float = 1e-5) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.virtual_batch_size = virtual_batch_size
        self.momentum = momentum
        self.epsilon = epsilon
        self.batch_norm = nn.BatchNorm1d(input_dim, momentum=momentum, eps=epsilon)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Aplica normalización por lotes fantasma.
        
        Parámetros:
        -----------
        x : torch.Tensor
            Tensor de entrada
            
        Retorna:
        --------
        torch.Tensor
            Tensor normalizado
        """
        if self.virtual_batch_size is None or self.virtual_batch_size >= x.size(0):
            return self.batch_norm(x)
        
        # Implementación de lotes virtuales
        chunks = x.chunk(int(np.ceil(x.size(0) / self.virtual_batch_size)))
        res = []
        
        for tensor_chunk in chunks:
            res.append(self.batch_norm(tensor_chunk))
            
        return torch.cat(res, dim=0)


def sparsemax(x: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """
    Implementación de Sparsemax como alternativa sparse a softmax.
    
    Parámetros:
    -----------
    x : torch.Tensor
        Tensor de entrada
    dim : int, opcional
        Dimensión sobre la cual aplicar sparsemax (default: -1)
        
    Retorna:
    --------
    torch.Tensor
        Tensor con probabilidades sparsemax
    """
    # Ordenar valores en orden descendente
    orig_size = x.size()
    flat_x = x.view(-1, orig_size[-1])
    
    sorted_x, _ = torch.sort(flat_x, descending=True, dim=1)
    cumsum = torch.cumsum(sorted_x, dim=1)
    
    # Calcular índice de corte (tau)
    indices = torch.arange(1, sorted_x.size(1) + 1).expand_as(sorted_x).to(x.device)
    candidate_tau = (1 + indices * sorted_x - cumsum) / indices
    
    # Obtener último valor donde sorted_x > tau
    valid = sorted_x > candidate_tau
    
    # Sumar para obtener k, asegurando que sea al menos 1
    k = valid.long().sum(dim=1, keepdim=True)
    k = torch.max(k, torch.ones_like(k))  # Asegurar que k sea al menos 1
    
    # Obtener tau y calcular la salida sparsemax
    tau = torch.gather(candidate_tau, 1, k - 1)
    
    # Calcular la salida sparsemax
    result = torch.clamp(x - tau.view(-1, 1), min=0)
    
    return result.view(orig_size)


class FeatureTransformer(nn.Module):
    """
    Transformador de características con atención multi-cabeza.
    
    Parámetros:
    -----------
    input_dim : int
        Dimensión de entrada
    output_dim : int
        Dimensión de salida
    num_heads : int
        Número de cabezas para la atención
    use_bn : bool, opcional
        Si usar normalización por lotes (default: True)
    virtual_batch_size : Optional[int], opcional
        Tamaño de lote virtual (default: None)
    dropout_rate : float, opcional
        Tasa de dropout (default: 0.2)
    """
    
    def __init__(self, input_dim: int, output_dim: int, num_heads: int, 
                 use_bn: bool = True, virtual_batch_size: Optional[int] = None, 
                 dropout_rate: float = 0.2) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.num_heads = num_heads
        self.use_bn = use_bn
        self.virtual_batch_size = virtual_batch_size
        self.dropout_rate = dropout_rate
        
        # Primera GLU
        self.glu1 = GatedLinearUnit(input_dim, output_dim)
        if use_bn:
            self.bn1 = GhostBatchNorm(output_dim, virtual_batch_size)
        
        # Atención multi-cabeza
        head_size = output_dim // num_heads
        self.head_layers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(output_dim, head_size),
                nn.Tanh()
            ) for _ in range(num_heads)
        ])
        
        if use_bn:
            self.bn_attn = GhostBatchNorm(output_dim, virtual_batch_size)
        
        # Segunda GLU
        self.glu2 = GatedLinearUnit(output_dim, output_dim)
        if use_bn:
            self.bn2 = GhostBatchNorm(output_dim, virtual_batch_size)
        
        # Dropout
        self.dropout = nn.Dropout(dropout_rate)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Aplica transformación con atención multi-cabeza.
        
        Parámetros:
        -----------
        x : torch.Tensor
            Tensor de entrada
            
        Retorna:
        --------
        torch.Tensor
            Tensor transformado
        """
        # Primera GLU
        x = self.glu1(x)
        if self.use_bn:
            x = self.bn1(x)
        
        # Procesamiento multi-cabeza
        head_outputs = []
        for head_layer in self.head_layers:
            head_outputs.append(head_layer(x))
        
        # Concatenar cabezas
        x = torch.cat(head_outputs, dim=-1)
        
        if self.use_bn:
            x = self.bn_attn(x)
        
        # Segunda GLU
        x = self.glu2(x)
        if self.use_bn:
            x = self.bn2(x)
        
        # Dropout
        x = self.dropout(x)
        
        return x


class AttentionBlock(nn.Module):
    """
    Bloque de atención para selección de características.
    
    Parámetros:
    -----------
    input_dim : int
        Dimensión de entrada/salida
    use_sparsity : bool, opcional
        Si usar sparsemax (True) o softmax (False) (default: True)
    """
    
    def __init__(self, input_dim: int, use_sparsity: bool = True) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.use_sparsity = use_sparsity
        self.fc = nn.Linear(input_dim, input_dim, bias=False)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Genera máscara de atención.
        
        Parámetros:
        -----------
        x : torch.Tensor
            Tensor de entrada
            
        Retorna:
        --------
        torch.Tensor
            Máscara de atención normalizada
        """
        attn_logits = self.fc(x)
        
        if self.use_sparsity:
            attention_mask = sparsemax(attn_logits, dim=-1)
        else:
            attention_mask = F.softmax(attn_logits, dim=-1)
            
        return attention_mask


class TabNetStep(nn.Module):
    """
    Implementa un paso de decisión en TabNet.
    
    Parámetros:
    -----------
    input_dim : int
        Dimensión de entrada
    output_dim : int
        Dimensión de salida
    feature_dim : int
        Dimensión para el transformador de características
    num_heads : int
        Número de cabezas de atención
    use_bn : bool, opcional
        Si usar normalización por lotes (default: True)
    virtual_batch_size : Optional[int], opcional
        Tamaño de lote virtual (default: None)
    dropout_rate : float, opcional
        Tasa de dropout (default: 0.2)
    use_sparsity : bool, opcional
        Si usar sparsemax para atención (default: True)
    relaxation_factor : float, opcional
        Factor de relajación para atención (default: 1.5)
    sparsity_coeff : float, opcional
        Coeficiente para regularización de esparcidad (default: 1e-4)
    """
    
    def __init__(self, input_dim: int, output_dim: int, feature_dim: int, num_heads: int,
                 use_bn: bool = True, virtual_batch_size: Optional[int] = None, 
                 dropout_rate: float = 0.2, use_sparsity: bool = True,
                 relaxation_factor: float = 1.5, sparsity_coeff: float = 1e-4) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.feature_dim = feature_dim
        self.relaxation_factor = relaxation_factor
        self.sparsity_coeff = sparsity_coeff
        
        # Bloque de atención
        self.attention = AttentionBlock(input_dim, use_sparsity)
        
        # Transformador de características
        self.feature_transformer = FeatureTransformer(
            input_dim, feature_dim, num_heads, use_bn, virtual_batch_size, dropout_rate
        )
        
    def forward(self, features: torch.Tensor, prior_scales: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Ejecuta un paso de decisión de TabNet.
        
        Parámetros:
        -----------
        features : torch.Tensor
            Características de entrada normalizadas
        prior_scales : torch.Tensor
            Escalas de prior de la máscara del paso anterior
            
        Retorna:
        --------
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor]
            - Salida procesada del paso (para la capa final)
            - Máscara de atención generada en este paso
            - Escalas de prior actualizadas para el siguiente paso
        """
        # Estabilizar prior_scales para evitar valores extremos
        prior_scales = torch.clamp(prior_scales, min=1e-8, max=1e8)
        
        # Generar máscara de atención
        attn_raw = self.attention.fc(features)
        
        # Estabilizar attn_raw para evitar valores extremos
        attn_raw = torch.clamp(attn_raw, min=-100.0, max=100.0)
        
        attn_scaled = attn_raw * prior_scales
        
        # Usar try-except para manejar posibles errores en sparsemax
        try:
            if hasattr(self.attention, 'use_sparsity') and self.attention.use_sparsity:
                mask = sparsemax(attn_scaled, dim=-1)
            else:
                mask = F.softmax(attn_scaled, dim=-1)
        except Exception:
            # Fallback seguro si sparsemax falla
            mask = F.softmax(attn_scaled, dim=-1)
        
        # Verificar y corregir cualquier NaN en la máscara
        if torch.isnan(mask).any():
            mask = torch.where(torch.isnan(mask), torch.zeros_like(mask), mask)
            # Renormalizar para asegurar que sume 1 en la dimensión adecuada
            mask_sum = mask.sum(dim=-1, keepdim=True)
            mask_sum = torch.where(mask_sum == 0, torch.ones_like(mask_sum), mask_sum)
            mask = mask / mask_sum
        
        # Actualizar prior_scales con protección contra NaN
        relaxation_term = self.sparsity_coeff * self.relaxation_factor
        relaxation_term = max(0.0, min(relaxation_term, 1.0))
        mask_complement = torch.clamp(1.0 - mask, min=0.0, max=1.0)
        
        new_prior_scales = prior_scales * (mask_complement + relaxation_term)
        
        # Verificar y corregir NaN en prior_scales
        if torch.isnan(new_prior_scales).any():
            new_prior_scales = torch.where(
                torch.isnan(new_prior_scales),
                prior_scales,  # Mantener valores anteriores si hay NaN
                new_prior_scales
            )
        
        # Aplicar máscara a las características con protección
        masked_features = features * mask
        
        # Transformar características enmascaradas
        transformed_features = self.feature_transformer(masked_features)
        
        # Verificar y corregir NaN en las características transformadas
        if torch.isnan(transformed_features).any():
            transformed_features = torch.where(
                torch.isnan(transformed_features),
                torch.zeros_like(transformed_features),
                transformed_features
            )
        
        # Aplicar ReLU para la salida
        step_output = F.relu(transformed_features)
        
        return step_output, mask, new_prior_scales


class TabNetModel(nn.Module):
    """
    Modelo TabNet para regresión.
    
    Parámetros:
    -----------
    cgm_shape : Tuple[int, ...]
        Forma de los datos CGM (timesteps, features)
    other_features_shape : Tuple[int, ...]
        Forma de otras características (features,)
    config : Dict, opcional
        Configuración del modelo (default: None, usa TABNET_CONFIG)
    """
    
    def __init__(self, cgm_shape: Tuple[int, ...], other_features_shape: Tuple[int, ...], 
                 config: Optional[Dict] = None) -> None:
        super().__init__()
        # Inicializar configuración
        self.config = config if config is not None else TABNET_CONFIG
        
        # Obtener parámetros de configuración
        self.feature_dim = self.config.get('feature_dim', 128)
        self.output_dim = self.config.get('output_dim', 64)
        self.num_decision_steps = self.config.get('num_decision_steps', 8)
        self.num_attention_heads = self.config.get('num_attention_heads', 4)
        self.dropout_rate = self.config.get('attention_dropout', 0.2)
        self.feature_dropout = self.config.get('feature_dropout', 0.1)
        self.virtual_batch_size = self.config.get('virtual_batch_size', 128)
        self.sparsity_coeff = self.config.get('sparsity_coefficient', 1e-4)
        self.relaxation_factor = self.config.get('relaxation_factor', 1.5)
        
        # Calcular dimensión de entrada total
        if len(cgm_shape) >= 2:
            self.cgm_features = np.prod(cgm_shape)
        else:
            self.cgm_features = cgm_shape[0]
            
        if len(other_features_shape) > 0:
            self.other_features = other_features_shape[0]
        else:
            self.other_features = 0
            
        self.input_dim = self.cgm_features + self.other_features
        
        # Transformación inicial
        self.initial_transform = nn.Linear(self.input_dim, self.feature_dim)
        self.initial_norm = nn.LayerNorm(self.feature_dim)
        self.feature_dropout_layer = nn.Dropout(self.feature_dropout)
        
        # Pasos de decisión
        self.steps = nn.ModuleList([
            TabNetStep(
                input_dim=self.input_dim,
                output_dim=self.feature_dim,
                feature_dim=self.feature_dim,
                num_heads=self.num_attention_heads,
                use_bn=True,
                virtual_batch_size=self.virtual_batch_size,
                dropout_rate=self.dropout_rate,
                relaxation_factor=self.relaxation_factor,
                sparsity_coeff=self.sparsity_coeff
            ) for _ in range(self.num_decision_steps)
        ])
        
        # Capas finales
        final_units = [self.feature_dim, self.output_dim // 2, self.output_dim]
        self.final_layers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(final_units[i], final_units[i+1]),
                nn.SELU(),
                nn.LayerNorm(final_units[i+1]),
                nn.Dropout(self.dropout_rate) if i < len(final_units) - 2 else nn.Identity()
            ) for i in range(len(final_units) - 1)
        ])
        
        # Capa de salida
        self.output_layer = nn.Linear(self.output_dim, 1)
        
    def forward(self, x_cgm: torch.Tensor, x_other: torch.Tensor) -> torch.Tensor:
        """
        Procesa los datos de entrada a través del modelo TabNet.
        
        Parámetros:
        -----------
        x_cgm : torch.Tensor
            Datos CGM [batch, timesteps, features] o [batch, features]
        x_other : torch.Tensor
            Otras características [batch, features]
            
        Retorna:
        --------
        torch.Tensor
            Predicciones [batch, 1]
        """
        # Aplanar CGM si es necesario
        batch_size = x_cgm.size(0)
        x_cgm_flat = x_cgm.reshape(batch_size, -1)
        
        # Combinar características
        if self.other_features > 0:
            features = torch.cat([x_cgm_flat, x_other], dim=1)
        else:
            features = x_cgm_flat
        
        # Transformación inicial
        x = self.initial_transform(features)
        x = self.initial_norm(x)
        x = self.feature_dropout_layer(x)
        
        # Estado inicial para pasos de decisión
        step_outputs = []
        prior_scales = torch.ones_like(features)
        
        # Aplicar pasos de decisión
        for step in self.steps:
            step_output, _, prior_scales = step(features, prior_scales)
            step_outputs.append(step_output)
        
        # Combinar salidas de los pasos (si hay alguna)
        if step_outputs:
            combined = torch.stack(step_outputs).sum(dim=0)
        else:
            combined = torch.zeros((batch_size, self.output_dim), device=x.device)
        
        # Aplicar capas finales con conexión residual
        skip = combined
        for i, layer in enumerate(self.final_layers):
            combined = layer(combined)
            # Aplicar conexión residual si las dimensiones coinciden
            if i == len(self.final_layers) - 1 and skip.size(-1) == combined.size(-1):
                combined = combined + skip
        
        # Capa de salida
        output = self.output_layer(combined)
        
        return output


def _create_tabnet_model(cgm_shape: Tuple[int, ...], other_features_shape: Tuple[int, ...]) -> nn.Module:
    """
    Crea un modelo TabNet para regresión.
    
    Parámetros:
    -----------
    cgm_shape : Tuple[int, ...]
        Forma de los datos CGM (timesteps, features)
    other_features_shape : Tuple[int, ...]
        Forma de otras características (features,)
        
    Retorna:
    --------
    nn.Module
        Modelo TabNet inicializado
    """
    return TabNetModel(cgm_shape, other_features_shape)


def create_tabnet_model_wrapper(cgm_shape: Tuple[int, ...], other_features_shape: Tuple[int, ...]) -> DLModelWrapper:
    """
    Crea un modelo TabNet envuelto en DLModelWrapperPyTorch.
    
    Parámetros:
    -----------
    cgm_shape : Tuple[int, ...]
        Forma de los datos CGM (timesteps, features)
    other_features_shape : Tuple[int, ...]
        Forma de otras características (features,)
        
    Retorna:
    --------
    DLModelWrapper
        Modelo TabNet envuelto en DLModelWrapper
    """
    # Función creadora del modelo
    model_creator_fn = lambda: _create_tabnet_model(cgm_shape, other_features_shape)
    
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
    Retorna una función para crear un modelo TabNet compatible con la API del sistema.
    
    Retorna:
    --------
    Callable[[Tuple[int, ...], Tuple[int, ...]], DLModelWrapper]
        Función para crear el modelo con las formas de entrada especificadas
    """
    return create_tabnet_model_wrapper