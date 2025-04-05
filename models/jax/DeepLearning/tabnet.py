import os, sys
import jax
import jax.numpy as jnp
import flax.linen as nn
from flax.training import train_state
from typing import List, Tuple, Dict, Any, Optional, Callable, Sequence

import optax

PROJECT_ROOT = os.path.abspath(os.getcwd())
sys.path.append(PROJECT_ROOT) 

from config.models_config import TABNET_CONFIG

class GLU(nn.Module):
    """
    Gated Linear Unit como módulo de Flax.
    
    Parámetros:
    -----------
    units : int
        Número de unidades de salida
    """
    units: int
    
    @nn.compact
    def __call__(self, inputs: jnp.ndarray) -> jnp.ndarray:
        """
        Aplica la capa GLU a las entradas.
        
        Parámetros:
        -----------
        inputs : jnp.ndarray
            Tensor de entrada
            
        Retorna:
        --------
        jnp.ndarray
            Tensor procesado
        """
        x = nn.Dense(self.units * 2)(inputs)
        return x[:, :self.units] * jax.nn.sigmoid(x[:, self.units:])

class MultiHeadFeatureAttention(nn.Module):
    """
    Atención multi-cabeza para características.
    
    Parámetros:
    -----------
    num_heads : int
        Número de cabezas de atención
    key_dim : int
        Dimensión de las claves
    dropout : float
        Tasa de dropout
    """
    num_heads: int
    key_dim: int
    dropout: float = 0.0
    
    @nn.compact
    def __call__(self, inputs: jnp.ndarray, deterministic: bool = False) -> jnp.ndarray:
        """
        Aplica atención multi-cabeza a las entradas.
        
        Parámetros:
        -----------
        inputs : jnp.ndarray
            Tensor de entrada
        deterministic : bool
            Indica si está en modo inferencia
            
        Retorna:
        --------
        jnp.ndarray
            Tensor procesado
        """
        attention_output = nn.MultiHeadAttention(
            num_heads=self.num_heads,
            key_size=self.key_dim,
            dropout_rate=self.dropout,
            deterministic=deterministic
        )(inputs, inputs)
        
        return nn.LayerNorm(epsilon=1e-6)(inputs + attention_output)

class GhostBatchNorm(nn.Module):
    """
    Ghost Batch Normalization para conjuntos de datos pequeños.
    
    Parámetros:
    -----------
    virtual_batch_size : int
        Tamaño del batch virtual
    momentum : float
        Factor de momentum
    epsilon : float
        Valor pequeño para estabilidad numérica
    """
    virtual_batch_size: int
    momentum: float = 0.9
    epsilon: float = 1e-5
    
    @nn.compact
    def __call__(self, x: jnp.ndarray, deterministic: bool = False) -> jnp.ndarray:
        """
        Aplica normalización por lotes fantasma.
        
        Parámetros:
        -----------
        x : jnp.ndarray
            Tensor de entrada
        deterministic : bool
            Indica si está en modo inferencia
            
        Retorna:
        --------
        jnp.ndarray
            Tensor normalizado
        """
        # Parámetros entrenables
        scale = self.param('scale', nn.initializers.ones, (x.shape[-1],))
        bias = self.param('bias', nn.initializers.zeros, (x.shape[-1],))
        
        # Variables de seguimiento de estadísticas
        mean_var = self.variable(
            'batch_stats', 'mean', 
            lambda s: jnp.zeros(s), x.shape[-1]
        )
        var_var = self.variable(
            'batch_stats', 'var', 
            lambda s: jnp.ones(s), x.shape[-1]
        )
        
        if deterministic:
            # Modo inferencia: usar estadísticas acumuladas
            mean = mean_var.value
            var = var_var.value
        else:
            # Modo entrenamiento: normalizar por lotes virtuales
            batch_size = x.shape[0]
            
            if self.virtual_batch_size is None or self.virtual_batch_size >= batch_size:
                # Usar BatchNorm normal si el batch virtual es mayor que el batch real
                mean = jnp.mean(x, axis=0)
                var = jnp.var(x, axis=0)
            else:
                # Dividir en lotes virtuales
                num_virtual_batches = max(batch_size // self.virtual_batch_size, 1)
                x_reshaped = x[:num_virtual_batches * self.virtual_batch_size]
                x_reshaped = x_reshaped.reshape(num_virtual_batches, self.virtual_batch_size, -1)
                
                # Calcular medias y varianzas por batch virtual
                mean = jnp.mean(x_reshaped, axis=1)  # (num_virtual_batches, features)
                var = jnp.var(x_reshaped, axis=1)    # (num_virtual_batches, features)
                
                # Promediar estadísticas entre lotes virtuales
                mean = jnp.mean(mean, axis=0)  # (features,)
                var = jnp.mean(var, axis=0)    # (features,)
                
            # Actualizar estadísticas de seguimiento
            mean_var.value = self.momentum * mean_var.value + (1 - self.momentum) * mean
            var_var.value = self.momentum * var_var.value + (1 - self.momentum) * var
            
        # Normalizar
        return scale * (x - mean) / jnp.sqrt(var + self.epsilon) + bias

class EnhancedFeatureTransformer(nn.Module):
    """
    Transformador de características mejorado con atención y ghost batch norm.
    
    Parámetros:
    -----------
    feature_dim : int
        Dimensión de las características
    num_heads : int
        Número de cabezas de atención
    virtual_batch_size : int
        Tamaño del batch virtual
    dropout_rate : float
        Tasa de dropout
    """
    feature_dim: int
    num_heads: int
    virtual_batch_size: int
    dropout_rate: float = 0.1
    
    @nn.compact
    def __call__(self, inputs: jnp.ndarray, deterministic: bool = False) -> jnp.ndarray:
        """
        Aplica transformación a las características.
        
        Parámetros:
        -----------
        inputs : jnp.ndarray
            Tensor de entrada
        deterministic : bool
            Indica si está en modo inferencia
            
        Retorna:
        --------
        jnp.ndarray
            Tensor transformado
        """
        # GLU layers
        x = GLU(self.feature_dim)(inputs)
        x = GhostBatchNorm(self.virtual_batch_size)(x, deterministic=deterministic)
        x = MultiHeadFeatureAttention(
            num_heads=self.num_heads,
            key_dim=self.feature_dim // self.num_heads,
            dropout=self.dropout_rate
        )(x, deterministic=deterministic)
        
        x = GLU(self.feature_dim)(x)
        x = GhostBatchNorm(self.virtual_batch_size)(x, deterministic=deterministic)
        return nn.Dropout(rate=self.dropout_rate, deterministic=deterministic)(x)

def custom_softmax(x: jnp.ndarray, axis: int = -1) -> jnp.ndarray:
    """
    Implementación de softmax con estabilidad numérica.

    Parámetros:
    -----------
    x : jnp.ndarray
        Tensor de entrada
    axis : int
        Eje de normalización
    
    Retorna:
    --------
    jnp.ndarray
        Tensor normalizado
    """
    exp_x = jnp.exp(x - jnp.max(x, axis=axis, keepdims=True))
    return exp_x / jnp.sum(exp_x, axis=axis, keepdims=True)

def feature_transformer(x: jnp.ndarray, feature_dim: int, batch_momentum: float = 0.98) -> jnp.ndarray:
    """
    Transformador de características.

    Parámetros:
    -----------
    x : jnp.ndarray
        Tensor de entrada
    feature_dim : int
        Dimensión de las características
    batch_momentum : float
        Momento de la normalización por lotes
    
    Retorna:
    --------
    jnp.ndarray
        Tensor transformado
    """
    transform = nn.Dense(feature_dim * 2)(x)
    transform_gated = transform[:, :feature_dim] * jax.nn.sigmoid(transform[:, feature_dim:])
    return nn.BatchNorm(momentum=batch_momentum)(transform_gated)

class TabnetModel(nn.Module):
    """
    Modelo TabNet personalizado con manejo de pérdidas de entropía.
    
    Parámetros:
    -----------
    config : Dict
        Configuración del modelo
    cgm_shape : Tuple
        Forma de los datos CGM
    other_features_shape : Tuple
        Forma de otras características
    """
    config: Dict
    cgm_shape: Tuple
    other_features_shape: Tuple
    
    def setup(self):
        """
        Inicializa los componentes del modelo.
        """
        # Configuración de transformers
        self.transformers = [
            EnhancedFeatureTransformer(
                feature_dim=self.config['feature_dim'],
                num_heads=self.config['num_attention_heads'],
                virtual_batch_size=self.config['virtual_batch_size'],
                dropout_rate=self.config['attention_dropout']
            ) for _ in range(self.config['num_decision_steps'])
        ]
        
        # Capas finales
        self.final_dense1 = nn.Dense(self.config['output_dim'])
        self.final_norm1 = nn.LayerNorm(epsilon=1e-6)
        self.final_dense2 = nn.Dense(self.config['output_dim'] // 2)
        self.final_norm2 = nn.LayerNorm()
        self.final_dense3 = nn.Dense(self.config['output_dim'])
        self.output_layer = nn.Dense(1)
    
    @nn.compact
    def __call__(self, inputs: Tuple[jnp.ndarray, jnp.ndarray], training: bool = True) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """
        Aplica el modelo TabNet a las entradas.
        
        Parámetros:
        -----------
        inputs : Tuple[jnp.ndarray, jnp.ndarray]
            Tupla de (cgm_input, other_input)
        training : bool
            Indica si está en modo entrenamiento
            
        Retorna:
        --------
        Tuple[jnp.ndarray, jnp.ndarray]
            (Predicción, Pérdida de entropía)
        """
        cgm_input, other_input = inputs
        deterministic = not training
        
        # Procesamiento inicial
        x = jnp.reshape(cgm_input, (cgm_input.shape[0], -1))  # Flatten
        x = jnp.concatenate([x, other_input], axis=-1)
        
        # Feature masking en entrenamiento
        if training:
            feature_mask = jax.random.bernoulli(
                self.make_rng('dropout'),
                1.0 - self.config['feature_dropout'],
                shape=x.shape
            )
            x = x * feature_mask
        
        # Pasos de decisión
        step_outputs = []
        entropy_loss = 0.0
        
        for transformer in self.transformers:
            step_output = transformer(x, deterministic=deterministic)
            
            # Feature selection
            attention_mask = nn.Dense(x.shape[-1])(step_output)
            mask = custom_softmax(attention_mask)
            masked_x = x * mask
            
            step_outputs.append(masked_x)
            
            if training:
                # Calcular entropía
                entropy = jnp.mean(jnp.sum(
                    -mask * jnp.log(mask + 1e-15), axis=1
                ))
                entropy_loss += entropy
        
        # Combinar salidas con atención
        combined = jnp.stack(step_outputs, axis=1)
        attention_weights = nn.softmax(nn.Dense(len(step_outputs))(
            jnp.mean(combined, axis=2)
        ))
        x = jnp.sum(
            combined * jnp.expand_dims(attention_weights, -1),
            axis=1
        )
        
        # Calcular pérdida de entropía total
        if training:
            entropy_loss *= self.config['sparsity_coefficient']
        
        # Capas finales con residual
        x = self.final_dense1(x)
        x = nn.selu(x)  # SELU activation
        x = self.final_norm1(x)
        x = nn.Dropout(rate=self.config['attention_dropout'], deterministic=deterministic)(x)
        
        skip = x
        x = self.final_dense2(x)
        x = nn.selu(x)
        x = self.final_norm2(x)
        x = self.final_dense3(x)
        x = nn.selu(x)
        x = x + skip  # Skip connection
        
        output = self.output_layer(x)
        
        return output, entropy_loss

def create_tabnet_model(cgm_shape: tuple, other_features_shape: tuple) -> TabnetModel:
    """
    Crea un modelo TabNet mejorado con JAX/Flax.
    
    Parámetros:
    -----------
    cgm_shape : tuple
        Forma de los datos CGM (muestras, pasos_temporales, características)
    other_features_shape : tuple
        Forma de otras características (muestras, características)
        
    Retorna:
    --------
    tabnet_model
        Modelo TabNet inicializado
    """
    model = TabnetModel(
        config=TABNET_CONFIG,
        cgm_shape=cgm_shape,
        other_features_shape=other_features_shape
    )
    
    return model

def create_train_state(model: TabnetModel, 
                       rng: jax.random.PRNGKey, 
                       learning_rate: float,
                       cgm_shape: Tuple[int, ...],
                       other_features_shape: Tuple[int, ...]) -> train_state.TrainState:
    """
    Crea un estado de entrenamiento para TabNet.
    
    Parámetros:
    -----------
    model : tabnet_model
        Modelo TabNet
    rng : jax.random.PRNGKey
        Clave aleatoria para inicialización
    learning_rate : float
        Tasa de aprendizaje
    cgm_shape : Tuple[int, ...]
        Forma de los datos CGM
    other_features_shape : Tuple[int, ...]
        Forma de otras características
        
    Retorna:
    --------
    train_state.TrainState
        Estado de entrenamiento inicializado
    """
    # Crear claves separadas para parámetros y dropout
    params_rng, dropout_rng = jax.random.split(rng)
    
    # Datos ficticios para inicializar
    dummy_cgm = jnp.ones((1,) + cgm_shape[1:])
    dummy_other = jnp.ones((1, other_features_shape[1]))
    
    # Inicializar el modelo
    variables = model.init(
        {'params': params_rng, 'dropout': dropout_rng},
        (dummy_cgm, dummy_other),
        training=False
    )
    
    # Crear optimizador
    tx = optax.adam(
        learning_rate=learning_rate,
        b1=0.9,
        b2=0.999,
        eps=1e-8
    )
    
    # Crear estado de entrenamiento
    return train_state.TrainState.create(
        apply_fn=model.apply,
        params=variables['params'],
        tx=tx
    )

def apply_model(state: train_state.TrainState, 
                cgm_input: jnp.ndarray, 
                other_input: jnp.ndarray, 
                dropout_rng: Optional[jax.random.PRNGKey] = None,
                training: bool = False) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    Aplica el modelo con manejo de pérdida de entropía.
    
    Parámetros:
    -----------
    state : train_state.TrainState
        Estado actual de entrenamiento
    cgm_input : jnp.ndarray
        Entrada de datos CGM
    other_input : jnp.ndarray
        Entrada de otras características
    dropout_rng : jax.random.PRNGKey, opcional
        Clave para dropout
    training : bool
        Indica si está en modo entrenamiento
        
    Retorna:
    --------
    Tuple[jnp.ndarray, jnp.ndarray]
        (Predicciones, Pérdida de entropía)
    """
    variables = {'params': state.params}
    
    if training and dropout_rng is not None:
        predictions, entropy_loss = state.apply_fn(
            variables,
            (cgm_input, other_input),
            training=True,
            rngs={'dropout': dropout_rng}
        )
    else:
        predictions, entropy_loss = state.apply_fn(
            variables,
            (cgm_input, other_input),
            training=False
        )
    
    return predictions, entropy_loss