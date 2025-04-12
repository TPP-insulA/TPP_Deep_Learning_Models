import os, sys

PROJECT_ROOT = os.path.abspath(os.getcwd())
sys.path.append(PROJECT_ROOT) 

from custom.model_wrapper import ModelWrapper
import flax.linen as nn
import optax
import jax
import jax.numpy as jnp
import numpy as np
from flax.training import train_state
from typing import Dict, List, Tuple, Any, Optional, Callable

class DLModelWrapper(ModelWrapper):
    """
    Wrapper para modelos de aprendizaje profundo.
    
    ...
    """
    
    def __init__(self, model_creator: Callable, early_stopping: Optional[Any] = None):
        """
        Inicializa el wrapper con un creador de modelos.
        
        Parámetros:
        -----------
        model_creator : Callable
            Función que crea una instancia del modelo
        early_stopping : Optional[Any], opcional
            Instancia de callback de early stopping (default: None)
        """
        self.model_creator = model_creator
        self.model = None
        self.params = None
        self.state = None
        self.early_stopping = early_stopping
    
    def start(self, x_cgm: np.ndarray, x_other: np.ndarray, y: np.ndarray, 
                  rng_key: Any = None) -> Any:
        """
        Inicializa los parámetros del modelo utilizando Flax.
        
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
            Parámetros inicializados
        """
        if rng_key is None:
            rng_key = jax.random.PRNGKey(0)
        
        # Determinar formas de entrada
        x_cgm_shape = (1,) + x_cgm.shape[1:]
        x_other_shape = (1,) + x_other.shape[1:]
        
        # Crear e inicializar modelo Flax
        self.model = self.model_creator()
        self.params = self.model.init(rng_key, jnp.ones(x_cgm_shape), jnp.ones(x_other_shape))
        
        # Configurar optimizador
        tx = optax.adam(learning_rate=1e-3)
        self.state = train_state.TrainState.create(
            apply_fn=self.model.apply,
            params=self.params,
            tx=tx
        )
        
        return self.params
    
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
        # Verificar que el modelo esté inicializado
        if self.state is None:
            rng_key = jax.random.PRNGKey(0)
            self.initialize(x_cgm, x_other, y, rng_key)
        
        # Preparar datos de validación si están disponibles
        x_cgm_val, x_other_val, y_val = None, None, None
        if validation_data is not None:
            (x_cgm_val, x_other_val), y_val = validation_data
        
        # Función de pérdida
        def loss_fn(params, x_cgm, x_other, y):
            preds = self.model.apply(params, x_cgm, x_other)
            return jnp.mean((preds - y) ** 2)
        
        # Función de paso de entrenamiento
        @jax.jit
        def train_step(state, x_cgm, x_other, y):
            grad_fn = jax.value_and_grad(loss_fn)
            loss, grads = grad_fn(state.params, x_cgm, x_other, y)
            state = state.apply_gradients(grads=grads)
            return state, loss
        
        # Función de evaluación
        @jax.jit
        def eval_step(params, x_cgm, x_other, y):
            preds = self.model.apply(params, x_cgm, x_other)
            loss = jnp.mean((preds - y) ** 2)
            return loss
        
        # Dividir datos en lotes
        def get_batches(x_cgm, x_other, y, batch_size):
            num_samples = len(y)
            indices = jnp.arange(num_samples)
            indices = jax.random.permutation(jax.random.PRNGKey(0), indices)
            
            for i in range(0, num_samples, batch_size):
                batch_indices = indices[i:min(i + batch_size, num_samples)]
                yield (
                    jnp.take(x_cgm, batch_indices, axis=0),
                    jnp.take(x_other, batch_indices, axis=0),
                    jnp.take(y, batch_indices, axis=0)
                )
        
        # Entrenamiento
        history = {
            "loss": [],
            "val_loss": []
        }
        
        for epoch in range(epochs):
            # Entrenamiento
            epoch_losses = []
            for x_cgm_batch, x_other_batch, y_batch in get_batches(
                    jnp.array(x_cgm), jnp.array(x_other), 
                    jnp.array(y), batch_size):
                self.state, batch_loss = train_step(self.state, x_cgm_batch, x_other_batch, y_batch)
                epoch_losses.append(float(batch_loss))
            
            # Registrar pérdida de entrenamiento
            avg_loss = sum(epoch_losses) / len(epoch_losses)
            history["loss"].append(float(avg_loss))
            
            # Evaluación en validación si hay datos disponibles
            if x_cgm_val is not None and x_other_val is not None and y_val is not None:
                val_loss = eval_step(
                    self.state.params,
                    jnp.array(x_cgm_val),
                    jnp.array(x_other_val),
                    jnp.array(y_val)
                )
                history["val_loss"].append(float(val_loss))
            else:
                history["val_loss"].append(float(0.0))
            
            print(f"Época {epoch+1}/{epochs}, loss: {avg_loss:.4f}, val_loss: {history['val_loss'][-1]:.4f}")
        
        return history
    
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
            raise ValueError("El modelo debe ser inicializado y entrenado antes de predecir")
        
        # Intentar llamar con training=False primero
        try:
            preds = self.model.apply(self.state.params, jnp.array(x_cgm), jnp.array(x_other), training=False)
        except TypeError:
            # Si falla, intentar sin el parámetro training
            preds = self.model.apply(self.state.params, jnp.array(x_cgm), jnp.array(x_other))
        
        return np.array(preds)