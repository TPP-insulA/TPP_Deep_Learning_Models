import os
import sys
import jax
import jax.numpy as jnp
import flax.linen as nn
from flax.training import train_state
import numpy as np
from typing import List, Tuple, Dict, Any, Optional, Callable, Sequence

import optax

from custom.printer import print_warning

# Asegurarse de que el directorio raíz del proyecto esté en el path
# Modificado para buscar correctamente desde la ubicación del archivo actual
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../.."))
sys.path.append(PROJECT_ROOT)

from config.models_config import TABNET_CONFIG
from custom.dl_model_wrapper import DLModelWrapper
from models.early_stopping import get_early_stopping_config # Importar configuración de early stopping

# --- Constantes ---
CONST_TRAINING: str = "training"
CONST_DROPOUT: str = "dropout"
CONST_PARAMS: str = "params"
CONST_BATCH_STATS: str = "batch_stats" # Usado por nn.BatchNorm

class GLU(nn.Module):
    """
    Unidad Lineal con Compuerta (Gated Linear Unit) como módulo de Flax.

    Atributos:
    ----------
    units : int
        Número de unidades de salida.
    """
    units: int

    @nn.compact
    def __call__(self, inputs: jnp.ndarray) -> jnp.ndarray:
        """
        Aplica la capa GLU a las entradas.

        Parámetros:
        -----------
        inputs : jnp.ndarray
            Tensor de entrada.

        Retorna:
        --------
        jnp.ndarray
            Tensor procesado.
        """
        x = nn.Dense(self.units * 2, name="glu_dense")(inputs)
        # Divide la salida en dos partes y aplica la compuerta sigmoide
        return x[..., :self.units] * jax.nn.sigmoid(x[..., self.units:])

class MultiHeadFeatureAttention(nn.Module):
    """
    Atención multi-cabeza para características.

    Atributos:
    ----------
    num_heads : int
        Número de cabezas de atención.
    key_dim : int
        Dimensión de las claves y valores.
    dropout_rate : float
        Tasa de dropout a aplicar después de la atención.
    """
    num_heads: int
    key_dim: int
    dropout_rate: float = 0.0 # Renombrado de 'dropout' a 'dropout_rate' para consistencia

    @nn.compact
    def __call__(self, inputs: jnp.ndarray, training: bool) -> jnp.ndarray:
        """
        Aplica atención multi-cabeza a las entradas.

        Parámetros:
        -----------
        inputs : jnp.ndarray
            Tensor de entrada (batch, seq_len, features).
        training : bool
            Indica si el modelo está en modo entrenamiento (afecta dropout).

        Retorna:
        --------
        jnp.ndarray
            Tensor procesado después de la atención y la conexión residual.
        """
        # Asegurarse de que la entrada sea 3D para MultiHeadAttention (batch, seq_len=1, features)
        if inputs.ndim == 2:
            inputs_reshaped = jnp.expand_dims(inputs, axis=1)
        else:
            inputs_reshaped = inputs

        attention_output = nn.MultiHeadDotProductAttention(
            num_heads=self.num_heads,
            qkv_features=self.key_dim * self.num_heads, # Dimensión total para Q, K, V
            out_features=inputs.shape[-1], # Dimensión de salida igual a la de entrada
            dropout_rate=self.dropout_rate, # Usar el atributo renombrado
            deterministic=not training, # deterministic=True deshabilita dropout
            name="multi_head_attention"
        )(inputs_reshaped, inputs_reshaped) # Atiende sobre sí mismo

        # Eliminar la dimensión de secuencia si se añadió
        if inputs.ndim == 2:
            attention_output = jnp.squeeze(attention_output, axis=1)

        # Conexión residual y normalización de capa
        output = nn.LayerNorm(epsilon=1e-6, name="attention_layernorm")(inputs + attention_output)
        return output

class EnhancedFeatureTransformer(nn.Module):
    """
    Transformador de características mejorado con GLU, BatchNorm y Atención.

    Atributos:
    ----------
    feature_dim : int
        Dimensión de las características de salida de las capas GLU.
    num_heads : int
        Número de cabezas de atención.
    dropout_rate : float
        Tasa de dropout.
    """
    feature_dim: int
    num_heads: int
    dropout_rate: float = 0.1

    @nn.compact
    def __call__(self, inputs: jnp.ndarray, training: bool) -> jnp.ndarray:
        """
        Aplica la transformación a las características de entrada.

        Parámetros:
        -----------
        inputs : jnp.ndarray
            Tensor de entrada.
        training : bool
            Indica si el modelo está en modo entrenamiento (afecta BatchNorm y Dropout).

        Retorna:
        --------
        jnp.ndarray
            Tensor transformado.
        """
        # Primera capa GLU + BatchNorm
        x = GLU(self.feature_dim, name="glu_1")(inputs)
        # Llamada funcional a BatchNorm, use_running_average se pasa aquí
        x = nn.BatchNorm(momentum=0.9, epsilon=1e-5, use_running_average=not training, name="bn_1")(x)

        # Atención Multi-Cabeza
        is_2d = x.ndim == 2
        if is_2d:
            x = jnp.expand_dims(x, axis=1)

        x = MultiHeadFeatureAttention(
            num_heads=self.num_heads,
            key_dim=max(1, self.feature_dim // self.num_heads), # Asegurar key_dim >= 1
            dropout_rate=self.dropout_rate,
            name="attention"
        )(x, training=training)

        if is_2d:
            x = jnp.squeeze(x, axis=1)

        # Segunda capa GLU + BatchNorm + Dropout
        x = GLU(self.feature_dim, name="glu_2")(x)
        # Llamada funcional a BatchNorm
        x = nn.BatchNorm(momentum=0.9, epsilon=1e-5, use_running_average=not training, name="bn_2")(x)
        x = nn.Dropout(rate=self.dropout_rate, deterministic=not training)(x)
        return x

def sparsemax(x: jnp.ndarray, axis: int = -1) -> jnp.ndarray:
    """
    Implementación de Sparsemax con estabilidad numérica.
    Proyecta el vector sobre el simplex de probabilidad, induciendo esparcidad.

    Parámetros:
    -----------
    x : jnp.ndarray
        Tensor de entrada.
    axis : int
        Eje sobre el cual aplicar sparsemax.

    Retorna:
    --------
    jnp.ndarray
        Tensor con probabilidades sparsemax.
    """
    n_features = x.shape[axis]
    x_sorted = jnp.sort(x, axis=axis)[..., ::-1] # Ordenar descendentemente
    cumsum = jnp.cumsum(x_sorted, axis=axis)
    k = jnp.arange(1, n_features + 1)
    # Condición para encontrar el soporte: z_i - tau > 0
    # tau = (cumsum - 1) / k
    condition = (x_sorted - (cumsum - 1) / k) > 0
    # Encontrar el índice k más grande que satisface la condición
    k_max = jnp.sum(condition.astype(jnp.int32), axis=axis, keepdims=True)
    # Obtener el tau correspondiente a k_max
    # Necesitamos gather para seleccionar el valor correcto de cumsum basado en k_max
    # JAX no tiene gather_nd directamente como TF, usamos indexación avanzada
    # Asegurar que los índices sean válidos (>= 0)
    safe_k_max = jnp.maximum(k_max - 1, 0)
    indices = tuple(jnp.indices(k_max.shape[:-1])) + (safe_k_max[..., 0],) # Índices para seleccionar el cumsum correcto
    # Evitar división por cero si k_max es 0 (aunque no debería ocurrir si hay al menos una característica)
    safe_k_max_squeeze = jnp.maximum(k_max.squeeze(axis), 1)
    tau = (cumsum[indices] - 1) / safe_k_max_squeeze # Calcular tau

    # Calcular la salida sparsemax
    output = jnp.maximum(0, x - jnp.expand_dims(tau, axis))
    return output

class AttentiveTransformer(nn.Module):
    """
    Transformador Atento para la selección de características en TabNet.

    Atributos:
    ----------
    output_dim : int
        Dimensión de la salida de la capa Dense que genera la máscara.
    momentum : float
        Momentum para BatchNorm.
    epsilon : float
        Epsilon para BatchNorm.
    sparsity_func : Callable
        Función para inducir esparcidad (e.g., sparsemax o custom_softmax).
    """
    output_dim: int
    momentum: float = 0.9
    epsilon: float = 1e-5
    sparsity_func: Callable = sparsemax # Usar sparsemax por defecto

    @nn.compact
    def __call__(self, inputs: jnp.ndarray, prior_scales: jnp.ndarray, training: bool) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """
        Aplica el transformador atento.

        Parámetros:
        -----------
        inputs : jnp.ndarray
            Características procesadas por el FeatureTransformer anterior.
        prior_scales : jnp.ndarray
            Escalas de prior de la máscara del paso anterior.
        training : bool
            Indica si el modelo está en modo entrenamiento.

        Retorna:
        --------
        Tuple[jnp.ndarray, jnp.ndarray]
            - Máscara de atención generada.
            - Escalas de prior actualizadas para el siguiente paso.
        """
        x = nn.Dense(self.output_dim, name="attn_dense")(inputs)
        # Llamada funcional a BatchNorm
        x = nn.BatchNorm(momentum=self.momentum, epsilon=self.epsilon, use_running_average=not training, name="attn_bn")(x)
        # Multiplicar por las escalas de prior del paso anterior
        x = x * prior_scales
        # Aplicar la función de esparcidad (sparsemax)
        mask = self.sparsity_func(x)
        # Actualizar las escalas de prior para el siguiente paso
        new_prior_scales = prior_scales * (1.0 - mask) # gamma - mask
        return mask, new_prior_scales

class TabNetStep(nn.Module):
    """
    Un paso de decisión en el modelo TabNet.

    Atributos:
    ----------
    feature_dim : int
        Dimensión de las características para el FeatureTransformer.
    output_dim : int
        Dimensión de salida para el AttentiveTransformer (igual a la dimensión de entrada original).
    num_heads : int
        Número de cabezas de atención en el FeatureTransformer.
    dropout_rate : float
        Tasa de dropout en el FeatureTransformer.
    sparsity_func : Callable
        Función de esparcidad para el AttentiveTransformer.
    """
    feature_dim: int
    output_dim: int
    num_heads: int
    dropout_rate: float
    sparsity_func: Callable = sparsemax

    @nn.compact
    def __call__(self, features: jnp.ndarray, prior_scales: jnp.ndarray, training: bool) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        """
        Ejecuta un paso de decisión de TabNet.

        Parámetros:
        -----------
        features : jnp.ndarray
            Características de entrada originales normalizadas.
        prior_scales : jnp.ndarray
            Escalas de prior de la máscara del paso anterior.
        training : bool
            Indica si el modelo está en modo entrenamiento.

        Retorna:
        --------
        Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]
            - Salida procesada del paso (para la capa final).
            - Máscara de atención generada en este paso.
            - Escalas de prior actualizadas para el siguiente paso.
        """
        # 1. Generar máscara de atención usando el AttentiveTransformer
        attn_transformer = AttentiveTransformer(
            output_dim=self.output_dim,
            sparsity_func=self.sparsity_func,
            name="attentive_transformer"
        )
        # La atención se aplica sobre las características originales normalizadas
        mask, new_prior_scales = attn_transformer(features, prior_scales, training)

        # 2. Aplicar máscara a las características originales normalizadas
        masked_features = features * mask

        # 3. Procesar características enmascaradas con el FeatureTransformer
        feature_transformer = EnhancedFeatureTransformer(
            feature_dim=self.feature_dim,
            num_heads=self.num_heads,
            dropout_rate=self.dropout_rate,
            name="feature_transformer"
        )
        processed_features = feature_transformer(masked_features, training)

        # 4. Salida del paso (usualmente ReLU aplicado a una parte)
        # Dividir la salida: una parte para la decisión final, otra para el siguiente paso (no implementado aquí)
        # Aquí usamos toda la salida procesada para la agregación final
        output_for_final_layer = nn.relu(processed_features) # Aplicar ReLU

        return output_for_final_layer, mask, new_prior_scales

class TabNetModel(nn.Module):
    """
    Modelo TabNet implementado con JAX/Flax.

    Atributos:
    ----------
    config : Dict
        Configuración del modelo (ver TABNET_CONFIG).
    cgm_shape : Tuple
        Forma de los datos CGM (ignorada después de flatten).
    other_features_shape : Tuple
        Forma de otras características (ignorada después de flatten).
    """
    config: Dict
    cgm_shape: Tuple
    other_features_shape: Tuple

    def setup(self) -> None:
        """
        Inicializa los componentes del modelo TabNet.
        """
        self.num_decision_steps = self.config['num_decision_steps']
        self.feature_dim = self.config['feature_dim'] # Dimensión para FeatureTransformer
        self.output_dim = self.config['output_dim'] # Dimensión de salida de cada paso (para agregación)
        self.num_heads = self.config['num_attention_heads']
        self.dropout_rate = self.config['attention_dropout'] # Dropout en FeatureTransformer
        self.feature_dropout_rate = self.config['feature_dropout'] # Dropout en entrada
        self.sparsity_coefficient = self.config['sparsity_coefficient']

        # Dimensión de entrada total (calculada en create_tabnet_model y pasada en config)
        input_dim = self.config['input_dim']

        # Crear los pasos de decisión
        self.steps = [
            TabNetStep(
                # Dimensión para el FeatureTransformer dentro del paso
                feature_dim=self.feature_dim + self.output_dim, # Dimensión combinada (decisión + características)
                output_dim=input_dim, # Dimensión de la máscara = dimensión de entrada original
                num_heads=self.num_heads,
                dropout_rate=self.dropout_rate,
                sparsity_func=sparsemax,
                name=f"tabnet_step_{i}"
            ) for i in range(self.num_decision_steps)
        ]

        # Capa final para combinar las salidas de los pasos
        self.final_dense = nn.Dense(1, name="final_output_dense") # Salida de regresión (1 valor)

    @nn.compact
    def __call__(self, cgm_input: jnp.ndarray, other_input: jnp.ndarray, training: bool, return_mask: bool = False) -> jnp.ndarray:
        """
        Aplica el modelo TabNet a las entradas.

        Parámetros:
        -----------
        cgm_input : jnp.ndarray
            Datos de entrada CGM (batch, time_steps, cgm_features).
        other_input : jnp.ndarray
            Otras características de entrada (batch, other_features).
        training : bool
            Indica si el modelo está en modo entrenamiento.
        return_mask : bool
            Indica si se debe devolver la máscara de atención agregada.

        Retorna:
        --------
        jnp.ndarray
            Predicciones del modelo (batch,).
        """
        # 1. Preprocesamiento de entrada
        cgm_flat = jnp.reshape(cgm_input, (cgm_input.shape[0], -1))
        features = jnp.concatenate([cgm_flat, other_input], axis=-1)

        # Verificar dimensión de entrada (opcional, pero bueno para depuración)
        if features.shape[-1] != self.config['input_dim']:
            print_warning(f"Warning: Actual input dim ({features.shape[-1]}) != config input_dim ({self.config['input_dim']}).")
            # Idealmente, esto debería lanzar un error o ser manejado robustamente.

        # Aplicar BatchNorm inicial a las características
        features_norm = nn.BatchNorm(
            momentum=0.9,
            epsilon=1e-5,
            use_running_average=not training,
            name="input_bn"
        )(features)

        # Feature masking (dropout de características) en entrenamiento
        if training and self.feature_dropout_rate > 0:
            dropout_rng = self.make_rng(CONST_DROPOUT)
            keep_prob = 1.0 - self.feature_dropout_rate
            # Generar máscara Bernoulli
            feature_mask = jax.random.bernoulli(dropout_rng, keep_prob, shape=features_norm.shape)
            # Aplicar máscara (inverted dropout)
            features_norm = jnp.where(feature_mask, features_norm / keep_prob, 0)

        # 2. Pasos de decisión de TabNet
        step_outputs_agg = [] # Salidas para la agregación final
        aggregated_mask = jnp.zeros_like(features_norm)
        prior_scales = jnp.ones_like(features_norm) # Inicializar prior scales (gamma)
        total_entropy_loss = 0.0

        # La entrada a cada paso es siempre las características originales normalizadas
        input_features_for_steps = features_norm

        for i, step in enumerate(self.steps):
            # Ejecutar el paso de TabNet
            step_output, mask, prior_scales_next = step(
                input_features_for_steps,
                prior_scales,
                training=training
            )

            # Acumular la salida del paso para la agregación final
            step_outputs_agg.append(step_output)

            # Actualizar prior scales para el siguiente paso
            prior_scales = prior_scales_next

            # Acumular máscara agregada (para análisis o regularización)
            aggregated_mask += mask # Podría necesitar escalarse por (gamma - mask) si se usa diferente

            # Calcular pérdida de entropía para regularización (opcional)
            if self.sparsity_coefficient > 0:
                # Entropía de la máscara del paso actual P(i) = mask
                # Loss = - sum(P(i) * log(P(i) + eps)) promediado sobre batch y características
                entropy_per_sample = -jnp.sum(mask * jnp.log(mask + 1e-10), axis=-1)
                # Entropía promedio sobre el batch
                mean_entropy = jnp.mean(entropy_per_sample)
                total_entropy_loss += mean_entropy

        # 3. Combinar salidas de los pasos
        if not step_outputs_agg:
            # Caso borde: sin pasos de decisión
            final_features = jnp.zeros((features.shape[0], self.output_dim)) # Salida dummy con dim esperada
        else:
            # Sumar las salidas procesadas (con ReLU aplicado) de cada paso
            final_features = jnp.sum(jnp.stack(step_outputs_agg, axis=0), axis=0)

        # 4. Capa final de salida
        output = self.final_dense(final_features)

        # Añadir la pérdida de entropía a las pérdidas del modelo
        # Esto requiere que el framework de entrenamiento lo recoja.
        if self.sparsity_coefficient > 0 and training:
            # Dividir por el número de pasos para obtener la pérdida promedio por paso
            avg_entropy_loss = total_entropy_loss / self.num_decision_steps
            # Multiplicar por el coeficiente de esparcidad
            calculated_entropy_loss = avg_entropy_loss * self.sparsity_coefficient
            # Registrar la pérdida de entropía en el estado del modelo
            entropy_loss_value_to_return = calculated_entropy_loss
            # Registrar la pérdida para que pueda ser añadida a la pérdida principal externamente
            self.sow('losses', 'entropy_loss', calculated_entropy_loss)

        final_output = output.squeeze(-1) # Asegurar forma (batch_size,)
        
        return final_output, aggregated_mask, entropy_loss_value_to_return if return_mask else final_output

def create_tabnet_model(cgm_shape: Tuple[int, ...], other_features_shape: Tuple[int, ...]) -> DLModelWrapper:
    """
    Crea un modelo TabNet con JAX/Flax envuelto en DLModelWrapper.

    Parámetros:
    -----------
    cgm_shape : Tuple[int, ...]
        Forma de los datos CGM (pasos_temporales, características_cgm).
    other_features_shape : Tuple[int, ...]
        Forma de otras características (características_otras,).

    Retorna:
    --------
    DLModelWrapper
        Modelo TabNet inicializado y envuelto en DLModelWrapper.
    """
    # Calcular la dimensión total de entrada aplanada
    input_dim = np.prod(cgm_shape) + np.prod(other_features_shape)

    # Crear una copia local de la configuración para modificarla
    local_tabnet_config = TABNET_CONFIG.copy()
    local_tabnet_config['input_dim'] = int(input_dim) # Asegurar que sea int nativo

    # Función que crea una instancia del modelo con la configuración actualizada
    def model_creator_func() -> TabNetModel:
        return TabNetModel(
            config=local_tabnet_config,
            cgm_shape=cgm_shape,
            other_features_shape=other_features_shape
        )

    # Obtener configuración de early stopping
    es_patience, es_min_delta, es_restore_best = get_early_stopping_config()

    # Envolver el modelo en DLModelWrapper para compatibilidad con el sistema
    # Especificar explícitamente 'jax' como backend
    wrapper = DLModelWrapper(model_creator_func, 'jax')

    # Configurar early stopping
    wrapper.add_early_stopping(
        patience=es_patience,
        min_delta=es_min_delta,
        restore_best_weights=es_restore_best
    )

    return wrapper

def model_creator() -> Callable[[Tuple[int, ...], Tuple[int, ...]], DLModelWrapper]:
    """
    Retorna una función de fábrica para crear un modelo TabNet compatible con la API del sistema.
    Esta función de fábrica toma las formas de entrada y devuelve el modelo envuelto.

    Retorna:
    --------
    Callable[[Tuple[int, ...], Tuple[int, ...]], DLModelWrapper]
        Función que, dadas las formas de entrada, crea el modelo TabNet envuelto.
    """
    return create_tabnet_model