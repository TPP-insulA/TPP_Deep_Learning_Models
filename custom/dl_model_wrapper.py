from typing import Dict, List, Tuple, Callable, Optional, Any, Union
import numpy as np
import jax
import jax.numpy as jnp
import flax.linen as nn
from flax.training import train_state
import optax

from custom.model_wrapper import ModelWrapper
from custom.printer import cprint

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
    
    def add_early_stopping(self, patience: int = 10, min_delta: float = 0.0, restore_best_weights: bool = True) -> None:
        """
        Añade early stopping al modelo.
        
        Parámetros:
        -----------
        patience : int, opcional
            Número de épocas a esperar para detener el entrenamiento (default: 10)
        min_delta : float, opcional
            Cambio mínimo considerado como mejora (default: 0.0)
        restore_best_weights : bool, opcional
            Si restaurar los mejores pesos al finalizar (default: True)
        """
        self.early_stopping = {
            'patience': patience,
            'min_delta': min_delta,
            'restore_best_weights': restore_best_weights,
            'best_loss': float('inf'),
            'best_params': None,
            'wait': 0
        }
    
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
        
        # Inicializar variables del modelo
        variables = model.init(rng_key, cgm_sample, other_sample, training=True)
        
        # Crear estado de entrenamiento
        learning_rate = 0.001
        tx = optax.adam(learning_rate=learning_rate)
        
        state = train_state.TrainState.create(
            apply_fn=model.apply,
            params=variables['params'],
            tx=tx
        )
        
        # Guardar estado e instancia del modelo
        self.state = state
        self.params = variables
        self.model_instance = model
    
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
    
    def _create_loss_fn(self) -> Callable:
        """
        Crea una función de pérdida para el entrenamiento.
        
        Retorna:
        --------
        Callable
            Función de pérdida
        """
        def loss_fn(params, batch_cgm, batch_other, batch_targets, rng, training=True):
            predictions = self.model_instance.apply(
                {'params': params},
                batch_cgm, batch_other, training=training,
                rngs={'dropout': rng}
            )
            loss = jnp.mean((predictions - batch_targets) ** 2)
            return loss, predictions
        
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
            batch_rng = jax.random.fold_in(epoch_rng, batch_idx)
            loss, self.state = train_step(self.state, batch_cgm, batch_other, batch_y, batch_rng)
            batch_losses.append(loss)
        
        return float(np.mean(batch_losses)), self.state
    
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
        
        # Datos de validación
        do_validation = validation_data is not None
        x_cgm_val = x_other_val = y_val = None
        
        if do_validation:
            (x_cgm_val, x_other_val), y_val = validation_data
        
        # Convertir a arrays de JAX
        x_cgm_arr = jnp.array(x_cgm)
        x_other_arr = jnp.array(x_other)
        y_arr = jnp.array(y)
        
        if do_validation:
            x_cgm_val_arr = jnp.array(x_cgm_val)
            x_other_val_arr = jnp.array(x_other_val)
            y_val_arr = jnp.array(y_val)
        
        # Crear función de pérdida y paso de entrenamiento
        loss_fn = self._create_loss_fn()
        
        @jax.jit
        def train_step(state, batch_cgm, batch_other, batch_targets, rng):
            """Función para un paso de entrenamiento"""
            def loss_wrapper(params):
                loss, _ = loss_fn(params, batch_cgm, batch_other, batch_targets, rng)
                return loss
            
            # Calcular gradientes y actualizar parámetros
            grad_fn = jax.value_and_grad(loss_wrapper)
            loss, grads = grad_fn(state.params)
            new_state = state.apply_gradients(grads=grads)
            
            return loss, new_state
        
        # Entrenar por épocas
        master_rng = jax.random.PRNGKey(0)
        
        for epoch in range(epochs):
            # Clave PRNG para la época
            epoch_rng = jax.random.fold_in(master_rng, epoch)
            
            # Entrenamiento
            avg_loss, self.state = self._run_training_epoch(
                train_step, x_cgm_arr, x_other_arr, y_arr, batch_size, epoch_rng
            )
            
            # Validación
            val_loss = None
            if do_validation:
                val_preds = self.model_instance.apply(
                    {'params': self.state.params},
                    x_cgm_val_arr, x_other_val_arr, training=False
                )
                val_loss = float(jnp.mean((val_preds - y_val_arr) ** 2))
            
            # Actualizar historial
            self.history["loss"].append(avg_loss)
            if do_validation:
                self.history["val_loss"].append(val_loss)
            
            # Imprimir progreso
            val_str = f" - val_loss: {val_loss:.4f}" if do_validation else ""
            cprint(f"Época {epoch+1}/{epochs} - loss: {avg_loss:.4f}{val_str}", background='blue', colour='yellow', style='bold')
            
            # Early stopping
            if self._apply_early_stopping(val_loss, avg_loss, do_validation, epoch, epochs):
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
            raise ValueError("El modelo debe ser entrenado antes de predecir")
        
        x_cgm_arr = jnp.array(x_cgm)
        x_other_arr = jnp.array(x_other)
        
        predictions = self.model_instance.apply(
            {'params': self.state.params},
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