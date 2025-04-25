import time
from typing import Dict, List, Tuple, Callable, Optional, Any, Union
import numpy as np
import jax
import jax.numpy as jnp
from flax.training import train_state
from flax.core.frozen_dict import FrozenDict
import optax
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from custom.model_wrapper import ModelWrapper
from custom.printer import cprint, print_debug, print_info

# Constantes para mensajes de error
CONST_MODEL_INIT_ERROR = "El modelo debe ser inicializado antes de {}"
CONST_DEVICE = "device"

# Clase para modelos TensorFlow
class DLModelWrapperTF(ModelWrapper):
    """
    Wrapper para modelos de deep learning implementados en TensorFlow.
    
    Parámetros:
    -----------
    model_creator : Callable
        Función que crea una instancia del modelo
    """
    
    def __init__(self, model_creator: Callable) -> None:
        """
        Inicializa el wrapper con un creador de modelo TensorFlow.
        
        Parámetros:
        -----------
        model_creator : Callable
            Función que crea una instancia del modelo
        """
        super().__init__()
        self.model_creator = model_creator
        self.model = None
    
    def start(self, x_cgm: np.ndarray, x_other: np.ndarray, y: np.ndarray, 
             rng_key: Any = None) -> Any:
        """
        Inicializa el modelo con los datos de entrada.
        
        Parámetros:
        -----------
        x_cgm : np.ndarray
            Datos CGM de entrada
        x_other : np.ndarray
            Otras características de entrada
        y : np.ndarray
            Valores objetivo
        rng_key : Any, opcional
            Clave para generación aleatoria (default: None)
            
        Retorna:
        --------
        Any
            Estado inicial del modelo o parámetros
        """
        if self.model is None:
            self.model = self.model_creator()
        
        # Compilar si no está compilado
        if not hasattr(self.model, 'compiled_loss'):
            self.model.compile(optimizer='adam', loss='mse')
            
        return self.model
    
    def train(self, x_cgm: np.ndarray, x_other: np.ndarray, y: np.ndarray, 
             validation_data: Optional[Tuple[Tuple[np.ndarray, np.ndarray], np.ndarray]] = None,
             epochs: int = 10, batch_size: int = 32) -> Dict[str, List[float]]:
        """
        Entrena el modelo con los datos proporcionados.
        
        Parámetros:
        -----------
        x_cgm : np.ndarray
            Datos CGM de entrenamiento
        x_other : np.ndarray
            Otras características de entrenamiento
        y : np.ndarray
            Valores objetivo
        validation_data : Optional[Tuple[Tuple[np.ndarray, np.ndarray], np.ndarray]], opcional
            Datos de validación como ((x_cgm_val, x_other_val), y_val) (default: None)
        epochs : int, opcional
            Número de épocas de entrenamiento (default: 10)
        batch_size : int, opcional
            Tamaño de lote (default: 32)
            
        Retorna:
        --------
        Dict[str, List[float]]
            Historial de entrenamiento con métricas
        """
        if self.model is None:
            self.start(x_cgm, x_other, y)
            
        val_data = None
        if validation_data is not None:
            (x_cgm_val, x_other_val), y_val = validation_data
            val_data = ([x_cgm_val, x_other_val], y_val)
            
        history = self.model.fit(
            [x_cgm, x_other], y,
            validation_data=val_data,
            epochs=epochs,
            batch_size=batch_size,
            verbose=1
        )
        
        return history.history
    
    def predict(self, x_cgm: np.ndarray, x_other: np.ndarray) -> np.ndarray:
        """
        Realiza predicciones con el modelo entrenado.
        
        Parámetros:
        -----------
        x_cgm : np.ndarray
            Datos CGM para predicción
        x_other : np.ndarray
            Otras características para predicción
            
        Retorna:
        --------
        np.ndarray
            Predicciones del modelo
        """
        if self.model is None:
            raise ValueError("El modelo debe ser inicializado antes de predecir")
            
        return self.model.predict([x_cgm, x_other])
    
    def evaluate(self, x_cgm: np.ndarray, x_other: np.ndarray, y: np.ndarray) -> Dict[str, float]:
        """
        Evalúa el modelo con datos de prueba.
        
        Parámetros:
        -----------
        x_cgm : np.ndarray
            Datos CGM de prueba
        x_other : np.ndarray
            Otras características de prueba
        y : np.ndarray
            Valores objetivo reales
            
        Retorna:
        --------
        Dict[str, float]
            Diccionario con métricas de evaluación
        """
        if self.model is None:
            raise ValueError("El modelo debe ser inicializado antes de evaluar")
            
        loss = self.model.evaluate([x_cgm, x_other], y, verbose=0)
        preds = self.predict(x_cgm, x_other)
        
        return {
            "loss": float(loss),
            "mae": float(np.mean(np.abs(preds - y))),
            "rmse": float(np.sqrt(np.mean((preds - y) ** 2))),
            "r2": float(1 - np.sum((y - preds) ** 2) / np.sum((y - np.mean(y)) ** 2))
        }
    
    def save(self, path: str) -> None:
        """
        Guarda el modelo en disco.
        
        Parámetros:
        -----------
        path : str
            Ruta donde guardar el modelo
        """
        if self.model is None:
            raise ValueError("El modelo debe ser inicializado antes de guardarlo")
            
        self.model.save(path)
    
    def load(self, path: str) -> None:
        """
        Carga el modelo desde disco.
        
        Parámetros:
        -----------
        path : str
            Ruta desde donde cargar el modelo
        """
        import tensorflow as tf
        self.model = tf.keras.models.load_model(path)


# Clase para modelos JAX
class DLModelWrapperJAX(ModelWrapper):
    """
    Wrapper para modelos de deep learning implementados en JAX/Flax.
    
    Parámetros:
    -----------
    model_creator : Callable
        Función que crea una instancia del modelo
    """
    
    def __init__(self, model_creator: Callable) -> None:
        """
        Inicializa el wrapper con un creador de modelo JAX.
        
        Parámetros:
        -----------
        model_creator : Callable
            Función que crea una instancia del modelo
        """
        super().__init__()
        self.model_creator = model_creator
        self.model_instance = None
        self.params = None
        self.state = None
        self.early_stopping = None
        self.history = {"loss": [], "val_loss": []}
        self.batch_stats = None
    
    def add_early_stopping(self, patience: int = 10, min_delta: float = 0.0, restore_best_weights: bool = True) -> None:
        """
        Agrega early stopping al modelo.

        Parámetros:
        -----------
        patience : int, opcional
            Número de épocas a esperar para detener el entrenamiento (default: 10)
        min_delta : float, opcional
            Mínima mejora requerida para considerar una mejora (default: 0.0)
        restore_best_weights : bool, opcional
            Si restaurar los mejores pesos al finalizar (default: True)
        """
        if hasattr(self.wrapper, 'add_early_stopping'):
            self.wrapper.add_early_stopping(patience, min_delta, restore_best_weights)
        else:
            super().add_early_stopping(patience, min_delta, restore_best_weights)

    def initialize(self, x_cgm: np.ndarray, x_other: np.ndarray, y: np.ndarray, seed: int = 0) -> None:
        """
        Inicializa el modelo JAX con los datos de entrada.
        
        Parámetros:
        -----------
        x_cgm : np.ndarray
            Datos CGM de entrada
        x_other : np.ndarray
            Otras características de entrada
        y : np.ndarray
            Valores objetivo
        seed : int, opcional
            Semilla para la generación de números aleatorios (default: 0)
        """
        # Crea una clave de aleatoriedad
        rng_key = jax.random.PRNGKey(seed)
        
        # Crea una instancia del modelo
        model = self.model_creator()
        
        # Inicializa el modelo
        cgm_sample = jnp.array(x_cgm[0:1])
        other_sample = jnp.array(x_other[0:1])
        
        # Inicializar variables del modelo con colecciones mutables
        variables = model.init(rng_key, cgm_sample, other_sample, training=True)
        
        # Separar params y batch_stats
        params = variables['params']
        batch_stats = variables.get('batch_stats', {})
        
        # Crear estado de entrenamiento
        learning_rate = 0.001
        tx = optax.adam(learning_rate=learning_rate)
        
        state = train_state.TrainState.create(
            apply_fn=model.apply,
            params=params,
            tx=tx
        )
        
        # Guardar estado e instancia del modelo
        self.state = state
        self.params = variables
        self.model_instance = model
        self.batch_stats = batch_stats  # Guardar batch_stats separadamente
    
    def start(self, x_cgm: np.ndarray, x_other: np.ndarray, y: np.ndarray, 
             rng_key: Any = None) -> Any:
        """
        Inicializa el modelo con los datos de entrada.
        
        Parámetros:
        -----------
        x_cgm : np.ndarray
            Datos CGM de entrada
        x_other : np.ndarray
            Otras características de entrada
        y : np.ndarray
            Valores objetivo
        rng_key : Any, opcional
            Clave para generación aleatoria (default: None)
            
        Retorna:
        --------
        Any
            Estado inicial del modelo o parámetros
        """
        # Si no se proporciona una clave, crea una con una semilla predeterminada
        if rng_key is None:
            rng_key = jax.random.PRNGKey(42)
        
        # Inicializa el modelo
        self.initialize(x_cgm, x_other, y, seed=int(rng_key[0]))
        
        return self.params
    
    def _prepare_model_variables(self, params: Union[Dict, FrozenDict], training: bool) -> Tuple[Dict, Union[bool, List[str]]]:
        """
        Prepara las variables del modelo y configuración de mutabilidad.
        
        Parámetros:
        -----------
        params : Union[Dict, FrozenDict]
            Parámetros del modelo.
        training : bool
            Indica si es fase de entrenamiento.
            
        Retorna:
        --------
        Tuple[Dict, Union[bool, List[str]]]
            - Variables del modelo.
            - Configuración de mutabilidad.
        """
        variables = {'params': params}
        mutable_collections: List[str] = []
        
        if training:
            mutable_collections.append('batch_stats')
            mutable_collections.append('losses')
            
        if hasattr(self, 'batch_stats') and self.batch_stats:
            variables['batch_stats'] = self.batch_stats
            
        mutable: Union[bool, List[str]] = mutable_collections if mutable_collections else False
        
        return variables, mutable
    
    def _process_model_outputs(self, outputs: Union[jnp.ndarray, Tuple[jnp.ndarray, Any]], training: bool) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """
        Procesa las salidas del modelo y extrae predicciones y pérdida de entropía.
        
        Parámetros:
        -----------
        outputs : Union[jnp.ndarray, Tuple[jnp.ndarray, Any]]
            Salidas del modelo.
        training : bool
            Indica si es fase de entrenamiento.
            
        Retorna:
        --------
        Tuple[jnp.ndarray, jnp.ndarray]
            - Predicciones.
            - Pérdida de entropía.
        """
        entropy_loss: jnp.ndarray = jnp.array(0.0)
        
        if isinstance(outputs, tuple) and len(outputs) == 2 and training:
            predictions, updated_state = outputs
            entropy_loss = self._extract_entropy_loss(updated_state)
        elif isinstance(outputs, jax.Array):
            predictions = outputs
        else:
            raise ValueError(f"Salida inesperada del modelo.apply: {type(outputs)}")
            
        return predictions, entropy_loss
    
    def _extract_entropy_loss(self, updated_state: Any) -> jnp.ndarray:
        """
        Extrae la pérdida de entropía del estado actualizado.
        
        Parámetros:
        -----------
        updated_state : Any
            Estado actualizado del modelo.
            
        Retorna:
        --------
        jnp.ndarray
            Pérdida de entropía extraída.
        """
        if not updated_state or 'losses' not in updated_state:
            return jnp.array(0.0)
            
        collected_losses = updated_state.get('losses', {})
        sown_entropy_values = collected_losses.get('entropy_loss', None)
        
        # Caso 1: Lista o tupla de valores
        if sown_entropy_values and isinstance(sown_entropy_values, (tuple, list)) and len(sown_entropy_values) > 0:
            potential_loss = sown_entropy_values[0]
            
            if isinstance(potential_loss, jax.Array):
                return potential_loss
                
            try:
                return jnp.array(potential_loss)
            except TypeError:
                cprint(f"Advertencia: No se pudo convertir el valor sembrado '{potential_loss}' a jnp.array.", 'yellow')
                return jnp.array(0.0)
                
        # Caso 2: Valor escalar
        elif sown_entropy_values is not None and jnp.isscalar(sown_entropy_values):
            return jnp.array(sown_entropy_values)
            
        return jnp.array(0.0)
    
    def _calculate_primary_loss(self, predictions: jnp.ndarray, batch_targets: jnp.ndarray) -> jnp.ndarray:
        """
        Calcula la pérdida primaria (MSE).
        
        Parámetros:
        -----------
        predictions : jnp.ndarray
            Predicciones del modelo.
        batch_targets : jnp.ndarray
            Valores objetivo.
            
        Retorna:
        --------
        jnp.ndarray
            Pérdida primaria calculada.
        """
        predictions_flat = predictions.squeeze()
        batch_targets_flat = batch_targets.squeeze()
        return jnp.mean(jnp.square(predictions_flat - batch_targets_flat))
    
    def _create_loss_fn(self) -> Callable:
        """
        Crea una función de pérdida para el entrenamiento.
        
        Retorna:
        --------
        Callable
            Función de pérdida
        """
        def loss_fn(params: Union[Dict, FrozenDict],
                    batch_cgm: jnp.ndarray,
                    batch_other: jnp.ndarray,
                    batch_targets: jnp.ndarray,
                    rng: jax.random.PRNGKey,
                    training: bool = True) -> Tuple[jnp.ndarray, jnp.ndarray]:
            """
            Calcula la pérdida total (primaria + regularización) y las predicciones.
            """
            # Preparar variables del modelo
            variables, mutable = self._prepare_model_variables(params, training)
            
            # Aplicar el modelo
            outputs = self.model_instance.apply(
                variables,
                batch_cgm, batch_other, training=training,
                rngs={'dropout': rng} if training else None,
                mutable=mutable
            )
            
            # Procesar salidas
            predictions, entropy_loss = self._process_model_outputs(outputs, training)
            
            # Calcular pérdida primaria
            primary_loss = self._calculate_primary_loss(predictions, batch_targets)
            
            # Calcular pérdida total
            total_loss = primary_loss + entropy_loss
            
            return total_loss, predictions

        return loss_fn
    
    def _get_batches(self, x_cgm, x_other, y, batch_size):
        """
        Divide los datos en lotes para el entrenamiento.
        
        Parámetros:
        -----------
        x_cgm : np.ndarray
            Datos CGM
        x_other : np.ndarray
            Otras características
        y : np.ndarray
            Valores objetivo
        batch_size : int
            Tamaño de lote
            
        Retorna:
        --------
        Generator
            Generador que devuelve lotes de datos
        """
        # Convertir a jnp.arrays para evitar copias durante el entrenamiento
        x_cgm_arr = jnp.array(x_cgm)
        x_other_arr = jnp.array(x_other)
        y_arr = jnp.array(y)
        
        # Crear índices y dividir en lotes
        num_samples = len(y)
        indices = np.arange(num_samples)
        # Create a random generator with a seed for reproducibility
        rng = np.random.Generator(np.random.PCG64(42))
        rng.shuffle(indices)
        for i in range(0, num_samples, batch_size):
            batch_indices = indices[i:min(i + batch_size, num_samples)]
            batch_cgm = x_cgm_arr[batch_indices]
            batch_other = x_other_arr[batch_indices]
            batch_y = y_arr[batch_indices]
            
            yield batch_cgm, batch_other, batch_y
    
    def _run_training_epoch(self, train_step, x_cgm_arr, x_other_arr, y_arr, batch_size, epoch_rng):
        """
        Ejecuta una época de entrenamiento.
        
        Parámetros:
        -----------
        train_step : Callable
            Función de paso de entrenamiento
        x_cgm_arr : jnp.ndarray
            Datos CGM de entrenamiento
        x_other_arr : jnp.ndarray
            Otras características de entrenamiento
        y_arr : jnp.ndarray
            Valores objetivo
        batch_size : int
            Tamaño de lote
        epoch_rng : jnp.ndarray
            Clave PRNG para la época
            
        Retorna:
        --------
        Tuple[float, train_state.TrainState]
            Pérdida promedio y estado actualizado
        """
        batch_losses = []
        
        for batch_idx, (batch_cgm, batch_other, batch_y) in enumerate(self._get_batches(x_cgm_arr, x_other_arr, y_arr, batch_size)):
            # Generar clave RNG para este lote
            batch_rng = jax.random.fold_in(epoch_rng, batch_idx)
            
            # Ejecutar paso de entrenamiento con batch_stats
            loss, self.state = train_step(self.state, batch_cgm, batch_other, batch_y, batch_rng)
            batch_losses.append(loss)
        
        # Calcular pérdida promedio
        avg_loss = sum(batch_losses) / len(batch_losses) if batch_losses else 0.0
        
        return avg_loss, self.state
    
    def _apply_early_stopping(self, val_loss, avg_loss, do_validation, epoch, epochs):
        """
        Aplica early stopping si está configurado.
        
        Parámetros:
        -----------
        val_loss : Optional[float]
            Pérdida de validación
        avg_loss : float
            Pérdida de entrenamiento
        do_validation : bool
            Si se está realizando validación
        epoch : int
            Época actual
        epochs : int
            Total de épocas
            
        Retorna:
        --------
        bool
            True si se debe detener el entrenamiento, False en caso contrario
        """
        if not self.early_stopping:
            return False
            
        monitor_loss = float(val_loss) if val_loss is not None else avg_loss
        
        if monitor_loss < self.early_stopping['best_loss'] - self.early_stopping['min_delta']:
            # Mejora encontrada
            self.early_stopping['best_loss'] = monitor_loss
            self.early_stopping['wait'] = 0
            
            # Guardar mejores parámetros si se solicita
            if self.early_stopping['restore_best_weights']:
                self.early_stopping['best_params'] = self.state.params
            return False
        else:
            # Sin mejora
            self.early_stopping['wait'] += 1
            if self.early_stopping['wait'] >= self.early_stopping['patience']:
                print(f"\nEarly stopping activado en época {epoch+1}")
                
                # Restaurar mejores parámetros si se solicitó
                if self.early_stopping['restore_best_weights'] and self.early_stopping['best_params'] is not None:
                    self.state = self.state.replace(params=self.early_stopping['best_params'])
                return True
        
        return False
    
    def _prepare_validation_data(self, validation_data):
        """
        Prepara los datos de validación.
        
        Parámetros:
        -----------
        validation_data : Optional[Tuple[Tuple[np.ndarray, np.ndarray], np.ndarray]]
            Datos de validación
            
        Retorna:
        --------
        Tuple[bool, Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray], 
              Optional[jnp.ndarray], Optional[jnp.ndarray], Optional[jnp.ndarray]]
            Tupla con información de validación
        """
        do_validation = validation_data is not None
        x_cgm_val = x_other_val = y_val = None
        x_cgm_val_arr = x_other_val_arr = y_val_arr = None
        
        if do_validation:
            (x_cgm_val, x_other_val), y_val = validation_data
            x_cgm_val_arr = jnp.array(x_cgm_val)
            x_other_val_arr = jnp.array(x_other_val)
            y_val_arr = jnp.array(y_val)
            
        return do_validation, x_cgm_val, x_other_val, y_val, x_cgm_val_arr, x_other_val_arr, y_val_arr
    
    def _compute_validation_metrics(self, val_preds, y_val_arr, do_validation):
        """
        Calcula métricas de validación.
        
        Parámetros:
        -----------
        val_preds : jnp.ndarray
            Predicciones de validación
        y_val_arr : jnp.ndarray
            Valores reales de validación
        do_validation : bool
            Si se realiza validación
            
        Retorna:
        --------
        Tuple[Optional[float], Optional[float], Optional[float], Optional[float]]
            Tupla con métricas de validación (val_loss, mae, rmse, r2)
        """
        if not do_validation:
            return None, None, None, None
            
        val_loss = float(jnp.mean((val_preds - y_val_arr) ** 2))
        mae = float(np.mean(np.abs(val_preds - y_val_arr)))
        rmse = float(np.sqrt(np.mean((val_preds - y_val_arr) ** 2)))
        
        # R²
        ss_res = np.sum((y_val_arr - val_preds) ** 2)
        ss_tot = np.sum((y_val_arr - np.mean(y_val_arr)) ** 2)
        r2 = float(1 - (ss_res / (ss_tot + 1e-8)))
        
        return val_loss, mae, rmse, r2
    
    def _define_train_step(self, loss_fn):
        """
        Define el paso de entrenamiento JIT-compilado.
        
        Parámetros:
        -----------
        loss_fn : Callable
            Función de pérdida
            
        Retorna:
        --------
        Callable
            Función de paso de entrenamiento JIT-compilada
        """
        @jax.jit
        def train_step(state: train_state.TrainState,
                      batch_cgm: jnp.ndarray,
                      batch_other: jnp.ndarray,
                      batch_targets: jnp.ndarray,
                      rng: jax.random.PRNGKey) -> Tuple[jnp.ndarray, train_state.TrainState]:
            """
            Ejecuta un paso de entrenamiento JIT-compilado.
            """
            # Wrapper para la función de pérdida que devuelve (loss, aux_data)
            def loss_wrapper(params: Union[Dict, FrozenDict]) -> Tuple[jnp.ndarray, jnp.ndarray]:
                loss_val, predictions = loss_fn(params, batch_cgm, batch_other, batch_targets, rng, training=True)
                return loss_val, predictions

            # Calcular gradientes
            grad_fn = jax.value_and_grad(loss_wrapper, has_aux=True)
            (loss, _), grads = grad_fn(state.params)

            # Aplicar gradientes
            new_state = state.apply_gradients(grads=grads)

            return loss, new_state
        
        return train_step
    
    def _run_epoch(self, train_step, x_cgm_arr, x_other_arr, y_arr, batch_size, epoch_rng):
        """
        Ejecuta una época de entrenamiento.
        
        Parámetros:
        -----------
        train_step : Callable
            Función de paso de entrenamiento
        x_cgm_arr : jnp.ndarray
            Datos CGM de entrenamiento
        x_other_arr : jnp.ndarray
            Otras características de entrenamiento
        y_arr : jnp.ndarray
            Valores objetivo
        batch_size : int
            Tamaño de lote
        epoch_rng : jax.random.PRNGKey
            Clave RNG para la época
            
        Retorna:
        --------
        Tuple[float, float]
            Pérdida promedio y duración de la época
        """
        batch_losses = []
        epoch_start_time = time.time()
        
        for batch_idx, (batch_cgm, batch_other, batch_y) in enumerate(
                self._get_batches(x_cgm_arr, x_other_arr, y_arr, batch_size)):
            # Generar clave RNG para este lote
            batch_rng = jax.random.fold_in(epoch_rng, batch_idx)
            
            # Ejecutar paso de entrenamiento
            loss_value, new_train_state = train_step(
                self.state, batch_cgm, batch_other, batch_y, batch_rng
            )
            
            # Actualizar estado
            self.state = new_train_state
            if hasattr(new_train_state, 'batch_stats'):
                self.batch_stats = new_train_state.batch_stats
                
            batch_losses.append(float(loss_value))
            
        # Calcular pérdida y duración
        avg_loss = np.mean(batch_losses) if batch_losses else 0.0
        epoch_duration = time.time() - epoch_start_time
        
        return avg_loss, epoch_duration
    
    def _perform_validation(self, x_cgm_val_arr, x_other_val_arr, y_val_arr, do_validation):
        """
        Realiza la validación del modelo.
        
        Parámetros:
        -----------
        x_cgm_val_arr : Optional[jnp.ndarray]
            Datos CGM de validación
        x_other_val_arr : Optional[jnp.ndarray]
            Otras características de validación
        y_val_arr : Optional[jnp.ndarray]
            Valores objetivo de validación
        do_validation : bool
            Indica si se debe realizar validación
            
        Retorna:
        --------
        Tuple[Optional[float], Optional[float], Optional[float], Optional[float]]
            Métricas de validación (val_loss, mae, rmse, r2)
        """
        if not (do_validation and x_cgm_val_arr is not None and x_other_val_arr is not None):
            return None, None, None, None
            
        # Preparar variables para la predicción
        eval_variables = {'params': self.state.params}
        if hasattr(self, 'batch_stats') and self.batch_stats:
            eval_variables['batch_stats'] = self.batch_stats
            
        # Realizar predicción en modo evaluación
        val_preds_jax = self.model_instance.apply(
            eval_variables,
            x_cgm_val_arr, x_other_val_arr, training=False
        )
        
        # Convertir a NumPy
        val_preds_np = np.array(val_preds_jax)
        
        # Calcular métricas
        return self._compute_validation_metrics(
            val_preds_np, np.array(y_val_arr), do_validation
        )
    
    def _update_history(self, avg_loss, val_metrics, do_validation):
        """
        Actualiza el historial de entrenamiento.
        
        Parámetros:
        -----------
        avg_loss : float
            Pérdida promedio de entrenamiento
        val_metrics : Tuple[Optional[float], Optional[float], Optional[float], Optional[float]]
            Métricas de validación (val_loss, mae, rmse, r2)
        do_validation : bool
            Indica si se realizó validación
        """
        val_loss, mae, rmse, r2 = val_metrics
        
        self.history["loss"].append(avg_loss)
        if do_validation:
            self.history["val_loss"].append(val_loss)
            self.history["val_mae"].append(mae)
            self.history["val_rmse"].append(rmse)
            self.history["val_r2"].append(r2)
    
    def _format_progress_message(self, epoch, epochs, avg_loss, epoch_duration, val_metrics):
        """
        Formatea el mensaje de progreso del entrenamiento.
        
        Parámetros:
        -----------
        epoch : int
            Época actual
        epochs : int
            Total de épocas
        avg_loss : float
            Pérdida promedio
        epoch_duration : float
            Duración de la época
        val_metrics : Tuple[Optional[float], Optional[float], Optional[float], Optional[float]]
            Métricas de validación
            
        Retorna:
        --------
        str
            Mensaje formateado
        """
        val_loss, mae, rmse, r2 = val_metrics
        
        val_loss_str = f" | val_loss: {val_loss:.4f}" if val_loss is not None else ""
        metrics_str = ""
        if mae is not None and rmse is not None and r2 is not None:
            metrics_str = f" | MAE: {mae:.4f} | RMSE: {rmse:.4f} | R²: {r2:.4f}"
            
        return f"Época {epoch+1}/{epochs} >> Loss: {avg_loss:.4f}{val_loss_str}{metrics_str} (t: {epoch_duration:.2f}s)"
    
    def train(self, x_cgm: np.ndarray, x_other: np.ndarray, y: np.ndarray, 
             validation_data: Optional[Tuple[Tuple[np.ndarray, np.ndarray], np.ndarray]] = None,
             epochs: int = 10, batch_size: int = 32) -> Dict[str, List[float]]:
        """
        Entrena el modelo con los datos proporcionados.
        
        Parámetros:
        -----------
        x_cgm : np.ndarray
            Datos CGM de entrenamiento
        x_other : np.ndarray
            Otras características de entrenamiento
        y : np.ndarray
            Valores objetivo
        validation_data : Optional[Tuple[Tuple[np.ndarray, np.ndarray], np.ndarray]], opcional
            Datos de validación como ((x_cgm_val, x_other_val), y_val) (default: None)
        epochs : int, opcional
            Número de épocas de entrenamiento (default: 10)
        batch_size : int, opcional
            Tamaño de lote (default: 32)
            
        Retorna:
        --------
        Dict[str, List[float]]
            Historial de entrenamiento con métricas
        """
        # Inicializar modelo si no está inicializado
        if self.state is None:
            self.start(x_cgm, x_other, y)
        
        # Preparar datos
        validation_info = self._prepare_validation_data(validation_data)
        do_validation, _, _, _, x_cgm_val_arr, x_other_val_arr, y_val_arr = validation_info
        
        # Convertir a arrays de JAX
        x_cgm_arr = jnp.array(x_cgm)
        x_other_arr = jnp.array(x_other)
        y_arr = jnp.array(y)
        
        # Configurar entrenamiento
        loss_fn = self._create_loss_fn()
        train_step = self._define_train_step(loss_fn)
        master_rng = jax.random.PRNGKey(0)
        self.history = {"loss": [], "val_loss": [], "val_mae": [], "val_rmse": [], "val_r2": []}
        
        # Bucle de entrenamiento
        for epoch in range(epochs):
            # Generar clave RNG para esta época
            epoch_rng = jax.random.fold_in(master_rng, epoch)
            
            # Ejecutar época de entrenamiento
            avg_loss, epoch_duration = self._run_epoch(
                train_step, x_cgm_arr, x_other_arr, y_arr, batch_size, epoch_rng
            )
            
            # Realizar validación
            val_metrics = self._perform_validation(
                x_cgm_val_arr, x_other_val_arr, y_val_arr, do_validation
            )
            
            # Actualizar historial
            self._update_history(avg_loss, val_metrics, do_validation)
            
            # Imprimir progreso
            progress_msg = self._format_progress_message(
                epoch, epochs, avg_loss, epoch_duration, val_metrics
            )
            cprint(progress_msg, colour='yellow', background='blue', style='bold')
            
            # Early stopping
            if self._apply_early_stopping(val_metrics[0], avg_loss, do_validation, epoch, epochs):
                break
                
        return self.history
    
    def predict(self, x_cgm: np.ndarray, x_other: np.ndarray) -> np.ndarray:
        """
        Realiza predicciones con el modelo entrenado.
        
        Parámetros:
        -----------
        x_cgm : np.ndarray
            Datos CGM para predicción
        x_other : np.ndarray
            Otras características para predicción
            
        Retorna:
        --------
        np.ndarray
            Predicciones del modelo
        """
        if self.state is None:
            raise ValueError("El modelo no ha sido inicializado. Llama a start() primero.")
        
        x_cgm_arr = jnp.array(x_cgm)
        x_other_arr = jnp.array(x_other)
        
        # Preparar variables incluyendo batch_stats
        variables = {'params': self.state.params}
        if hasattr(self, 'batch_stats') and self.batch_stats:
            variables['batch_stats'] = self.batch_stats
        
        # Aplicar el modelo
        predictions = self.model_instance.apply(
            variables,
            x_cgm_arr, x_other_arr, training=False
        )
        
        return np.array(predictions)
    
    def evaluate(self, x_cgm: np.ndarray, x_other: np.ndarray, y: np.ndarray) -> Dict[str, float]:
        """
        Evalúa el modelo con datos de prueba.
        
        Parámetros:
        -----------
        x_cgm : np.ndarray
            Datos CGM de prueba
        x_other : np.ndarray
            Otras características de prueba
        y : np.ndarray
            Valores objetivo reales
            
        Retorna:
        --------
        Dict[str, float]
            Diccionario con métricas de evaluación
        """
        if self.state is None:
            raise ValueError("El modelo debe ser entrenado antes de evaluar")
        
        predictions = self.predict(x_cgm, x_other)
        mse = float(np.mean((predictions - y) ** 2))
        mae = float(np.mean(np.abs(predictions - y)))
        rmse = float(np.sqrt(mse))
        
        # R²
        ss_res = np.sum((y - predictions) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        r2 = float(1 - (ss_res / (ss_tot + 1e-8)))
        
        return {
            "loss": mse,
            "mae": mae,
            "rmse": rmse,
            "r2": r2
        }
    
    def save(self, path: str) -> None:
        """
        Guarda el modelo en disco.
        
        Parámetros:
        -----------
        path : str
            Ruta donde guardar el modelo
        """
        if self.state is None:
            raise ValueError("El modelo debe ser entrenado antes de guardarlo")
        
        import pickle
        with open(path, 'wb') as f:
            pickle.dump({
                'params': self.state.params,
                'history': self.history,
                'early_stopping': self.early_stopping
            }, f)
    
    def load(self, path: str) -> None:
        """
        Carga el modelo desde disco.
        
        Parámetros:
        -----------
        path : str
            Ruta desde donde cargar el modelo
        """
        import pickle
        with open(path, 'rb') as f:
            data = pickle.load(f)
        
        # Crear instancia del modelo
        model = self.model_creator()
        self.model_instance = model
        
        # Recrear estado
        learning_rate = 0.001
        tx = optax.adam(learning_rate=learning_rate)
        
        self.state = train_state.TrainState.create(
            apply_fn=model.apply,
            params=data['params'],
            tx=tx
        )
        
        self.history = data['history']
        self.early_stopping = data['early_stopping']

# Clase para modelos PyTorch
class DLModelWrapperPyTorch(ModelWrapper):
    """
    Wrapper para modelos de deep learning implementados en PyTorch.
    
    Parámetros:
    -----------
    model_creator : Callable
        Función que crea una instancia del modelo
    """
    
    def __init__(self, model_creator: Callable) -> None:
        """
        Inicializa el wrapper con un creador de modelo PyTorch.
        
        Parámetros:
        -----------
        model_creator : Callable
            Función que crea una instancia del modelo
        """
        super().__init__()
        self.model_creator = model_creator
        self.model = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.optimizer = None
        self.criterion = None
        self.scheduler = None
        self.early_stopping_config = {
            'patience': 10,
            'min_delta': 0.0,
            'restore_best_weights': True,
            'best_weights': None,
            'best_val_loss': float('inf'),
            'counter': 0,
            'early_stop': False
        }
        print_info(f"Usando dispositivo: {self.device}")
    
    def add_early_stopping(self, patience: int = 10, min_delta: float = 0.0, restore_best_weights: bool = True) -> None:
        """
        Configura early stopping para el entrenamiento.
        
        Parámetros:
        -----------
        patience : int, opcional
            Número de épocas a esperar para detener el entrenamiento (default: 10)
        min_delta : float, opcional
            Cambio mínimo para considerar mejora (default: 0.0)
        restore_best_weights : bool, opcional
            Si se deben restaurar los mejores pesos al finalizar (default: True)
        """
        self.early_stopping_config['patience'] = patience
        self.early_stopping_config['min_delta'] = min_delta
        self.early_stopping_config['restore_best_weights'] = restore_best_weights
    
    def start(self, x_cgm: np.ndarray, x_other: np.ndarray, y: np.ndarray, 
             rng_key: Any = None) -> Any:
        """
        Inicializa el modelo con los datos de entrada.
        
        Parámetros:
        -----------
        x_cgm : np.ndarray
            Datos CGM de entrada
        x_other : np.ndarray
            Otras características de entrada
        y : np.ndarray
            Valores objetivo
        rng_key : Any, opcional
            Clave para generación aleatoria (no usada en PyTorch) (default: None)
            
        Retorna:
        --------
        Any
            Modelo inicializado
        """
        # Crear modelo
        self.model = self.model_creator()
        
        # Mover modelo al dispositivo
        self.model = self.model.to(self.device)
        
        # Configurar criterio y optimizador por defecto
        self.criterion = nn.MSELoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        
        # Configurar scheduler por defecto
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, 'min', factor=0.5, patience=5, min_lr=1e-6
        )
        
        # Establecer semilla si se proporciona
        if rng_key is not None:
            if isinstance(rng_key, int):
                torch.manual_seed(rng_key)
                np.random.seed(rng_key)
            else:
                # Asumiendo que es un jax.random.PRNGKey
                try:
                    seed_val = int(rng_key[0])
                    torch.manual_seed(seed_val)
                    np.random.seed(seed_val)
                except (TypeError, IndexError):
                    torch.manual_seed(42)  # Valor por defecto
        
        return self.model
    
    def _create_dataloader(self, x_cgm: np.ndarray, x_other: np.ndarray, y: np.ndarray, 
                         batch_size: int, shuffle: bool = True) -> DataLoader:
        """
        Crea un DataLoader de PyTorch para los datos de entrada.
        
        Parámetros:
        -----------
        x_cgm : np.ndarray
            Datos CGM
        x_other : np.ndarray
            Otras características
        y : np.ndarray
            Valores objetivo
        batch_size : int
            Tamaño del lote
        shuffle : bool, opcional
            Si mezclar los datos (default: True)
            
        Retorna:
        --------
        DataLoader
            DataLoader de PyTorch con los datos
        """
        # Convertir a tensores de PyTorch
        x_cgm_tensor = torch.FloatTensor(x_cgm)
        x_other_tensor = torch.FloatTensor(x_other)
        y_tensor = torch.FloatTensor(y.reshape(-1, 1))
        
        # Crear dataset y dataloader
        dataset = TensorDataset(x_cgm_tensor, x_other_tensor, y_tensor)
        return DataLoader(
            dataset, 
            batch_size=batch_size, 
            shuffle=shuffle,
            pin_memory=True
        )
    
    def _check_early_stopping(self, val_loss: float) -> bool:
        """
        Verifica si debe activarse el early stopping.
        
        Parámetros:
        -----------
        val_loss : float
            Pérdida de validación actual
            
        Retorna:
        --------
        bool
            True si debe detenerse el entrenamiento, False en caso contrario
        """
        config = self.early_stopping_config
        
        if val_loss < config['best_val_loss'] - config['min_delta']:
            config['best_val_loss'] = val_loss
            if config['restore_best_weights']:
                config['best_weights'] = {k: v.cpu().clone() for k, v in self.model.state_dict().items()}
            config['counter'] = 0
            return False
        else:
            config['counter'] += 1
            if config['counter'] >= config['patience']:
                config['early_stop'] = True
                if config['restore_best_weights'] and config['best_weights'] is not None:
                    self.model.load_state_dict(config['best_weights'])
                return True
        return False
    
    def train(self, x_cgm: np.ndarray, x_other: np.ndarray, y: np.ndarray, 
             validation_data: Optional[Tuple[Tuple[np.ndarray, np.ndarray], np.ndarray]] = None,
             epochs: int = 10, batch_size: int = 32) -> Dict[str, List[float]]:
        """
        Entrena el modelo con los datos proporcionados.
        
        Parámetros:
        -----------
        x_cgm : np.ndarray
            Datos CGM de entrenamiento
        x_other : np.ndarray
            Otras características de entrenamiento
        y : np.ndarray
            Valores objetivo
        validation_data : Optional[Tuple[Tuple[np.ndarray, np.ndarray], np.ndarray]], opcional
            Datos de validación como ((x_cgm_val, x_other_val), y_val) (default: None)
        epochs : int, opcional
            Número de épocas de entrenamiento (default: 10)
        batch_size : int, opcional
            Tamaño de lote (default: 32)
            
        Retorna:
        --------
        Dict[str, List[float]]
            Historial de entrenamiento con métricas
        """
        if self.model is None:
            self.start(x_cgm, x_other, y)
        
        # Preparar datos de entrenamiento
        train_loader = self._create_dataloader(x_cgm, x_other, y, batch_size)
        
        # Preparar datos de validación si existen
        val_loader = None
        if validation_data is not None:
            (x_cgm_val, x_other_val), y_val = validation_data
            val_loader = self._create_dataloader(x_cgm_val, x_other_val, y_val, batch_size, shuffle=False)
        
        # Historial de entrenamiento
        history = {
            'loss': [],
            'val_loss': [],
            'mae': [],
            'val_mae': [],
            'rmse': [],
            'val_rmse': [],
            'r2': [],
            'val_r2': []
        }
        
        print_info(f"Iniciando entrenamiento para {epochs} épocas...")
        
        # Bucle de entrenamiento
        for epoch in range(epochs):
            epoch_start_time = time.time()
            
            # Modo entrenamiento
            self.model.train()
            train_loss = 0.0
            
            # Entrenar por lotes
            for x_cgm_batch, x_other_batch, y_batch in train_loader:
                # Mover datos al dispositivo
                x_cgm_batch = x_cgm_batch.to(self.device)
                x_other_batch = x_other_batch.to(self.device)
                y_batch = y_batch.to(self.device)
                
                # Limpiar gradientes
                self.optimizer.zero_grad()
                
                # Forward pass
                outputs = self.model(x_cgm_batch, x_other_batch)
                loss = self.criterion(outputs, y_batch)
                
                # Backward pass y optimización
                loss.backward()
                self.optimizer.step()
                
                # Acumular pérdida
                train_loss += loss.item() * x_cgm_batch.size(0)
            
            # Calcular pérdida promedio de entrenamiento
            train_loss = train_loss / len(train_loader.dataset)
            history['loss'].append(train_loss)
            
            # Modo evaluación para validación
            self.model.eval()
            
            # Validación si hay datos disponibles
            if val_loader is not None:
                val_metrics = self._validate(val_loader)
                
                # Actualizar historial con métricas de validación
                for key, value in val_metrics.items():
                    history[f"val_{key}"].append(value)
                
                # Ajustar learning rate basado en pérdida de validación
                self.scheduler.step(val_metrics['loss'])
                
                # Verificar early stopping
                if self._check_early_stopping(val_metrics['loss']):
                    print_info(f"Early stopping activado en época {epoch+1}")
                    break
                
                # Imprimir progreso con métricas de validación
                epoch_time = time.time() - epoch_start_time
                print_info(f"Época {epoch+1}/{epochs} - loss: {train_loss:.4f} - val_loss: {val_metrics['loss']:.4f} - "
                      f"val_mae: {val_metrics['mae']:.4f} - val_rmse: {val_metrics['rmse']:.4f} - "
                      f"val_r2: {val_metrics['r2']:.4f} - tiempo: {epoch_time:.2f}s")
            else:
                # Imprimir progreso sin validación
                epoch_time = time.time() - epoch_start_time
                print_info(f"Época {epoch+1}/{epochs} - loss: {train_loss:.4f} - tiempo: {epoch_time:.2f}s")
        
        return history
    
    def _validate(self, val_loader: DataLoader) -> Dict[str, float]:
        """
        Evalúa el modelo en el conjunto de validación.
        
        Parámetros:
        -----------
        val_loader : DataLoader
            DataLoader con datos de validación
            
        Retorna:
        --------
        Dict[str, float]
            Diccionario con métricas de validación
        """
        val_loss = 0.0
        all_preds = []
        all_targets = []
        
        with torch.no_grad():
            for x_cgm_batch, x_other_batch, y_batch in val_loader:
                # Mover datos al dispositivo
                x_cgm_batch = x_cgm_batch.to(self.device)
                x_other_batch = x_other_batch.to(self.device)
                y_batch = y_batch.to(self.device)
                
                # Forward pass
                outputs = self.model(x_cgm_batch, x_other_batch)
                loss = self.criterion(outputs, y_batch)
                
                # Acumular pérdida
                val_loss += loss.item() * x_cgm_batch.size(0)
                
                # Guardar predicciones y targets para métricas
                all_preds.append(outputs.cpu().numpy())
                all_targets.append(y_batch.cpu().numpy())
        
        # Calcular pérdida promedio
        val_loss = val_loss / len(val_loader.dataset)
        
        # Combinar predicciones y targets
        all_preds = np.vstack(all_preds).flatten()
        all_targets = np.vstack(all_targets).flatten()
        
        # Calcular métricas adicionales
        mae = float(np.mean(np.abs(all_preds - all_targets)))
        rmse = float(np.sqrt(np.mean((all_preds - all_targets) ** 2)))
        
        # Calcular R²
        ss_res = np.sum((all_targets - all_preds) ** 2)
        ss_tot = np.sum((all_targets - np.mean(all_targets)) ** 2)
        r2 = float(1 - (ss_res / (ss_tot + 1e-8)))
        
        return {
            'loss': val_loss,
            'mae': mae,
            'rmse': rmse,
            'r2': r2
        }
    
    def predict(self, x_cgm: np.ndarray, x_other: np.ndarray) -> np.ndarray:
        """
        Realiza predicciones con el modelo entrenado.
        
        Parámetros:
        -----------
        x_cgm : np.ndarray
            Datos CGM para predicción
        x_other : np.ndarray
            Otras características para predicción
            
        Retorna:
        --------
        np.ndarray
            Predicciones del modelo
        """
        if self.model is None:
            raise ValueError(CONST_MODEL_INIT_ERROR.format("predecir"))
        
        # Convertir a tensores
        x_cgm_tensor = torch.FloatTensor(x_cgm).to(self.device)
        x_other_tensor = torch.FloatTensor(x_other).to(self.device)
        
        # Modo evaluación
        self.model.eval()
        
        # Predicciones en lotes para evitar problemas de memoria
        batch_size = 128
        predictions = []
        
        with torch.no_grad():
            for i in range(0, len(x_cgm), batch_size):
                batch_end = min(i + batch_size, len(x_cgm))
                batch_cgm = x_cgm_tensor[i:batch_end]
                batch_other = x_other_tensor[i:batch_end]
                
                outputs = self.model(batch_cgm, batch_other)
                predictions.append(outputs.cpu().numpy())
        
        # Combinar predicciones
        return np.vstack(predictions).flatten()
    
    def evaluate(self, x_cgm: np.ndarray, x_other: np.ndarray, y: np.ndarray) -> Dict[str, float]:
        """
        Evalúa el modelo con datos de prueba.
        
        Parámetros:
        -----------
        x_cgm : np.ndarray
            Datos CGM de prueba
        x_other : np.ndarray
            Otras características de prueba
        y : np.ndarray
            Valores objetivo reales
            
        Retorna:
        --------
        Dict[str, float]
            Diccionario con métricas de evaluación
        """
        if self.model is None:
            raise ValueError(CONST_MODEL_INIT_ERROR.format("evaluar"))
        
        # Crear dataloader para evaluación
        test_loader = self._create_dataloader(x_cgm, x_other, y, batch_size=128, shuffle=False)
        
        # Evaluar el modelo
        metrics = self._validate(test_loader)
        
        return metrics
    
    def save(self, path: str) -> None:
        """
        Guarda el modelo en disco.
        
        Parámetros:
        -----------
        path : str
            Ruta donde guardar el modelo
        """
        if self.model is None:
            raise ValueError(CONST_MODEL_INIT_ERROR.format("guardar"))
        
        # Guardar modelo y configuración de entrenamiento
        save_dict = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict() if self.optimizer else None,
            'early_stopping_config': self.early_stopping_config
        }
        
        torch.save(save_dict, path)
    
    def load(self, path: str) -> None:
        """
        Carga el modelo desde disco.
        
        Parámetros:
        -----------
        path : str
            Ruta desde donde cargar el modelo
        """
        # Crear modelo si no existe
        if self.model is None:
            self.model = self.model_creator()
            self.model = self.model.to(self.device)
        
        # Cargar estado guardado
        checkpoint = torch.load(path, map_location=self.device)
        
        # Cargar estado del modelo
        self.model.load_state_dict(checkpoint['model_state_dict'])
        
        # Cargar estado del optimizador si existe
        if checkpoint['optimizer_state_dict'] and self.optimizer:
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        # Cargar configuración de early stopping si existe
        if 'early_stopping_config' in checkpoint:
            self.early_stopping_config = checkpoint['early_stopping_config']

# Clase principal que selecciona el wrapper adecuado según el framework
class DLModelWrapper(ModelWrapper):
    """
    Wrapper para modelos de deep learning que selecciona el wrapper adecuado según el framework.
    
    Parámetros:
    -----------
    model_creator : Callable
        Función que crea una instancia del modelo
    framework : str
        Framework a utilizar ('jax' o 'tensorflow')
    """
    
    def __init__(self, model_creator: Callable, framework: str = 'jax') -> None:
        """
        Inicializa el wrapper con un creador de modelo.
        
        Parámetros:
        -----------
        model_creator : Callable
            Función que crea una instancia del modelo
        framework : str, opcional
            Framework a utilizar ('jax' o 'tensorflow') (default: 'jax')
        """
        super().__init__()
        
        # Seleccionar wrapper específico según el framework
        if framework.lower() == 'tensorflow':
            self.wrapper = DLModelWrapperTF(model_creator)
        elif framework.lower() == 'pytorch':
            self.wrapper = DLModelWrapperPyTorch(model_creator)
        else:
            self.wrapper = DLModelWrapperJAX(model_creator)
    
    # Delegación de métodos al wrapper específico
    def start(self, x_cgm: np.ndarray, x_other: np.ndarray, y: np.ndarray, 
             rng_key: Any = None) -> Any:
        """
        Inicializa el modelo con los datos de entrada.
        
        Parámetros:
        -----------
        x_cgm : np.ndarray
            Datos CGM de entrada
        x_other : np.ndarray
            Otras características de entrada
        y : np.ndarray
            Valores objetivo
        rng_key : Any, opcional
            Clave para generación aleatoria (default: None)
            
        Retorna:
        --------
        Any
            Estado inicial del modelo o parámetros
        """
        return self.wrapper.start(x_cgm, x_other, y, rng_key)
    
    def train(self, x_cgm: np.ndarray, x_other: np.ndarray, y: np.ndarray, 
             validation_data: Optional[Tuple[Tuple[np.ndarray, np.ndarray], np.ndarray]] = None,
             epochs: int = 10, batch_size: int = 32) -> Dict[str, List[float]]:
        """
        Entrena el modelo con los datos proporcionados.
        
        Parámetros:
        -----------
        x_cgm : np.ndarray
            Datos CGM de entrenamiento
        x_other : np.ndarray
            Otras características de entrenamiento
        y : np.ndarray
            Valores objetivo
        validation_data : Optional[Tuple[Tuple[np.ndarray, np.ndarray], np.ndarray]], opcional
            Datos de validación como ((x_cgm_val, x_other_val), y_val) (default: None)
        epochs : int, opcional
            Número de épocas de entrenamiento (default: 10)
        batch_size : int, opcional
            Tamaño de lote (default: 32)
            
        Retorna:
        --------
        Dict[str, List[float]]
            Historial de entrenamiento con métricas
        """
        return self.wrapper.train(x_cgm, x_other, y, validation_data, epochs, batch_size)
    
    def predict(self, x_cgm: np.ndarray, x_other: np.ndarray) -> np.ndarray:
        """
        Realiza predicciones con el modelo entrenado.
        
        Parámetros:
        -----------
        x_cgm : np.ndarray
            Datos CGM para predicción
        x_other : np.ndarray
            Otras características para predicción
            
        Retorna:
        --------
        np.ndarray
            Predicciones del modelo
        """
        return self.wrapper.predict(x_cgm, x_other)
    
    def evaluate(self, x_cgm: np.ndarray, x_other: np.ndarray, y: np.ndarray) -> Dict[str, float]:
        """
        Evalúa el modelo con datos de prueba.
        
        Parámetros:
        -----------
        x_cgm : np.ndarray
            Datos CGM de prueba
        x_other : np.ndarray
            Otras características de prueba
        y : np.ndarray
            Valores objetivo reales
            
        Retorna:
        --------
        Dict[str, float]
            Diccionario con métricas de evaluación
        """
        return self.wrapper.evaluate(x_cgm, x_other, y)
    
    def save(self, path: str) -> None:
        """
        Guarda el modelo en disco.
        
        Parámetros:
        -----------
        path : str
            Ruta donde guardar el modelo
        """
        self.wrapper.save(path)
    
    def load(self, path: str) -> None:
        """
        Carga el modelo desde disco.
        
        Parámetros:
        -----------
        path : str
            Ruta desde donde cargar el modelo
        """
        self.wrapper.load(path)
    
    # Método para acceder al early stopping en JAX
    def add_early_stopping(self, patience: int = 10, min_delta: float = 0.0, restore_best_weights: bool = True) -> None:
        """
        Añade early stopping al modelo (solo para JAX).
        
        Parámetros:
        -----------
        patience : int, opcional
            Número de épocas a esperar para detener el entrenamiento (default: 10)
        min_delta : float, opcional
            Cambio mínimo considerado como mejora (default: 0.0)
        restore_best_weights : bool, opcional
            Si restaurar los mejores pesos al finalizar (default: True)
        """
        if isinstance(self.wrapper, DLModelWrapperJAX):
            self.wrapper.add_early_stopping(patience, min_delta, restore_best_weights)
        else:
            print("Early stopping solo está disponible para modelos JAX")