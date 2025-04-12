from typing import Dict, List, Tuple, Any, Optional, Callable, Union
import numpy as np
import jax
import jax.numpy as jnp
from custom.model_wrapper import ModelWrapper

# Clase para modelos RL con TensorFlow
class RLModelWrapperTF(ModelWrapper):
    """
    Wrapper para modelos de aprendizaje por refuerzo implementados en TensorFlow.
    
    Parámetros:
    -----------
    model_cls : Callable
        Clase del modelo RL a instanciar
    **model_kwargs
        Argumentos para el constructor del modelo
    """
    
    def __init__(self, model_cls: Callable, **model_kwargs) -> None:
        super().__init__()
        self.model_cls = model_cls
        self.model_kwargs = model_kwargs
        self.model = None
    
    def start(self, x_cgm: np.ndarray, x_other: np.ndarray, y: np.ndarray, 
             rng_key: Any = None) -> Any:
        """
        Inicializa el agente RL con las dimensiones del problema.
        
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
            Estado del modelo inicializado
        """
        if self.model is None:
            self.model = self.model_cls(**self.model_kwargs)
        
        # Determinar dimensiones de entrada
        cgm_shape = x_cgm.shape[1:] if x_cgm.ndim > 1 else (1,)
        other_shape = x_other.shape[1:] if x_other.ndim > 1 else (1,)
        
        # Inicializar modelo RL según su interfaz disponible
        if hasattr(self.model, 'setup'):
            self.model.setup(cgm_shape, other_shape)
        elif hasattr(self.model, 'initialize'):
            self.model.initialize(cgm_shape, other_shape)
        
        return self.model
    
    def train(self, x_cgm: np.ndarray, x_other: np.ndarray, y: np.ndarray, 
             validation_data: Optional[Tuple[Tuple[np.ndarray, np.ndarray], np.ndarray]] = None,
             epochs: int = 10, batch_size: int = 32) -> Dict[str, List[float]]:
        """
        Entrena el modelo RL con los datos proporcionados.
        
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
        # Preparar datos de validación y el historial
        x_cgm_val, x_other_val, y_val = self._unpack_validation_data(validation_data)
        history = {"loss": [], "val_loss": []}
        
        # Usar interfaz nativa del modelo si está disponible
        if hasattr(self.model, 'fit'):
            return self._train_with_native_fit(
                x_cgm, x_other, y, 
                (x_cgm_val, x_other_val, y_val) if x_cgm_val is not None else None,
                epochs, batch_size
            )
        
        # Entrenamiento personalizado por épocas
        for _ in range(epochs):
            # Entrenar una época
            epoch_loss = self._train_one_epoch(x_cgm, x_other, y, batch_size)
            history["loss"].append(epoch_loss)
            
            # Validar si hay datos de validación
            if x_cgm_val is not None and y_val is not None:
                val_loss = self._validate_model(x_cgm_val, x_other_val, y_val)
                history["val_loss"].append(val_loss)
        
        return history
    
    def _unpack_validation_data(self, validation_data: Optional[Tuple]) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray]]:
        """
        Desempaqueta los datos de validación si están disponibles.
        
        Parámetros:
        -----------
        validation_data : Optional[Tuple]
            Datos de validación como ((x_cgm_val, x_other_val), y_val)
            
        Retorna:
        --------
        Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray]]
            Tupla con (x_cgm_val, x_other_val, y_val)
        """
        x_cgm_val, x_other_val, y_val = None, None, None
        if validation_data is not None:
            (x_cgm_val, x_other_val), y_val = validation_data
        return x_cgm_val, x_other_val, y_val
    
    def _train_with_native_fit(self, x_cgm: np.ndarray, x_other: np.ndarray, y: np.ndarray, 
                            validation_data: Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]], 
                            epochs: int, batch_size: int) -> Dict[str, List[float]]:
        """
        Entrena utilizando el método fit nativo del modelo.
        
        Parámetros:
        -----------
        x_cgm : np.ndarray
            Datos CGM de entrenamiento
        x_other : np.ndarray
            Otras características de entrenamiento
        y : np.ndarray
            Valores objetivo
        validation_data : Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]]
            Datos de validación como (x_cgm_val, x_other_val, y_val)
        epochs : int
            Número de épocas de entrenamiento
        batch_size : int
            Tamaño de lote
            
        Retorna:
        --------
        Dict[str, List[float]]
            Historial de entrenamiento con métricas
        """
        history = {"loss": [], "val_loss": []}
        
        # Preparar validation_data para el modelo nativo
        val_data = None
        if validation_data is not None:
            x_cgm_val, x_other_val, y_val = validation_data
            val_data = ([x_cgm_val, x_other_val], y_val)
        
        model_history = self.model.fit(
            [x_cgm, x_other], y,
            validation_data=val_data,
            epochs=epochs,
            batch_size=batch_size
        )
        
        # Copiar historial del modelo
        if hasattr(model_history, 'history'):
            for key, values in model_history.history.items():
                history[key] = values
        elif isinstance(model_history, dict):
            for key, values in model_history.items():
                history[key] = values
        
        return history
    
    def _train_one_epoch(self, x_cgm: np.ndarray, x_other: np.ndarray, y: np.ndarray, 
                       batch_size: int) -> float:
        """
        Entrena durante una época completa y devuelve la pérdida promedio.
        
        Parámetros:
        -----------
        x_cgm : np.ndarray
            Datos CGM de entrenamiento
        x_other : np.ndarray
            Otras características de entrenamiento
        y : np.ndarray
            Valores objetivo
        batch_size : int
            Tamaño de lote
            
        Retorna:
        --------
        float
            Pérdida promedio de la época
        """
        epoch_loss = 0.0
        num_batches = 0
        
        # Entrenar por lotes
        for i in range(0, len(y), batch_size):
            batch_end = min(i + batch_size, len(y))
            batch_cgm = x_cgm[i:batch_end]
            batch_other = x_other[i:batch_end]
            batch_y = y[i:batch_end]
            
            batch_loss = self._train_batch(batch_cgm, batch_other, batch_y)
            epoch_loss += batch_loss
            num_batches += 1
        
        # Calcular pérdida promedio
        return epoch_loss / num_batches if num_batches > 0 else 0.0
    
    def _train_batch(self, batch_cgm: np.ndarray, batch_other: np.ndarray, 
                   batch_y: np.ndarray) -> float:
        """
        Entrena con un lote de datos.
        
        Parámetros:
        -----------
        batch_cgm : np.ndarray
            Lote de datos CGM
        batch_other : np.ndarray
            Lote de otras características
        batch_y : np.ndarray
            Lote de valores objetivo
            
        Retorna:
        --------
        float
            Pérdida del lote
        """
        if hasattr(self.model, 'train_batch'):
            return self.model.train_batch(batch_cgm, batch_other, batch_y)
        elif hasattr(self.model, 'update'):
            return self.model.update(batch_cgm, batch_other, batch_y)
        else:
            return self._train_batch_sample_by_sample(batch_cgm, batch_other, batch_y)
    
    def _train_batch_sample_by_sample(self, batch_cgm: np.ndarray, batch_other: np.ndarray, 
                                   batch_y: np.ndarray) -> float:
        """
        Entrena un lote muestra por muestra cuando no hay métodos por lotes.
        
        Parámetros:
        -----------
        batch_cgm : np.ndarray
            Lote de datos CGM
        batch_other : np.ndarray
            Lote de otras características
        batch_y : np.ndarray
            Lote de valores objetivo
            
        Retorna:
        --------
        float
            Pérdida promedio del lote
        """
        total_loss = 0.0
        for j in range(len(batch_y)):
            sample_loss = self._train_single_sample(
                batch_cgm[j:j+1], batch_other[j:j+1], batch_y[j:j+1]
            )
            total_loss += sample_loss
        return total_loss / len(batch_y) if len(batch_y) > 0 else 0.0
    
    def _validate_model(self, x_cgm_val: np.ndarray, x_other_val: np.ndarray, 
                      y_val: np.ndarray) -> float:
        """
        Valida el modelo con datos de validación si están disponibles.
        
        Parámetros:
        -----------
        x_cgm_val : np.ndarray
            Datos CGM de validación
        x_other_val : np.ndarray
            Otras características de validación
        y_val : np.ndarray
            Valores objetivo de validación
            
        Retorna:
        --------
        float
            Pérdida de validación
        """
        if x_cgm_val is not None and y_val is not None:
            preds = self.predict(x_cgm_val, x_other_val)
            return float(np.mean((preds - y_val) ** 2))
        return 0.0
    
    def _train_single_sample(self, x_cgm: np.ndarray, x_other: np.ndarray, y: np.ndarray) -> float:
        """
        Entrena con una sola muestra.
        
        Parámetros:
        -----------
        x_cgm : np.ndarray
            Datos CGM de una muestra
        x_other : np.ndarray
            Otras características de una muestra
        y : np.ndarray
            Valor objetivo de una muestra
            
        Retorna:
        --------
        float
            Pérdida de la muestra
        """
        if hasattr(self.model, 'train_step'):
            return self.model.train_step(x_cgm[0], x_other[0], y[0])
        elif hasattr(self.model, 'update_single'):
            return self.model.update_single(x_cgm[0], x_other[0], y[0])
        else:
            # Implementación genérica si no hay método específico
            return 0.0
    
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
        # Verificar si el modelo soporta predicción por lotes
        if hasattr(self.model, 'predict'):
            return self.model.predict([x_cgm, x_other])
        
        # Predicción muestra por muestra
        predictions = np.zeros((len(x_cgm),))
        for i in range(len(x_cgm)):
            predictions[i] = self._predict_single_sample(
                x_cgm[i:i+1], x_other[i:i+1]
            )
        return predictions
    
    def _predict_single_sample(self, x_cgm: np.ndarray, x_other: np.ndarray) -> float:
        """
        Predice para una sola muestra.
        
        Parámetros:
        -----------
        x_cgm : np.ndarray
            Datos CGM de una muestra
        x_other : np.ndarray
            Otras características de una muestra
            
        Retorna:
        --------
        float
            Predicción para la muestra
        """
        if hasattr(self.model, 'predict_single'):
            return self.model.predict_single(x_cgm[0], x_other[0])
        elif hasattr(self.model, 'get_action'):
            # Si el modelo es un agente RL, usar get_action
            return self.model.get_action(x_cgm[0], x_other[0])
        else:
            # Si no hay método específico
            return 0.0


# Clase para modelos RL con JAX
class RLModelWrapperJAX(ModelWrapper):
    """
    Wrapper para modelos de aprendizaje por refuerzo implementados en JAX.
    
    Parámetros:
    -----------
    model_cls : Callable
        Clase del modelo RL a instanciar
    **model_kwargs
        Argumentos para el constructor del modelo
    """
    
    def __init__(self, model_cls: Callable, **model_kwargs) -> None:
        super().__init__()
        self.model_cls = model_cls
        self.model_kwargs = model_kwargs
        self.model = None
        self.rng_key = jax.random.PRNGKey(0)
        
    def start(self, x_cgm: np.ndarray, x_other: np.ndarray, y: np.ndarray, 
             rng_key: Any = None) -> Any:
        """
        Inicializa el agente RL con las dimensiones del problema.
        
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
            Estado del modelo inicializado
        """
        if rng_key is not None:
            self.rng_key = rng_key
        
        if self.model is None:
            self.model = self.model_cls(**self.model_kwargs)
        
        # Determinar dimensiones de entrada
        cgm_shape = x_cgm.shape[1:] if x_cgm.ndim > 1 else (1,)
        other_shape = x_other.shape[1:] if x_other.ndim > 1 else (1,)
        
        # Inicializar modelo RL según su interfaz disponible
        if hasattr(self.model, 'setup'):
            self.model.setup(cgm_shape, other_shape, self.rng_key)
        elif hasattr(self.model, 'initialize'):
            self.model.initialize(cgm_shape, other_shape, self.rng_key)
        
        return self.model
    
    def train(self, x_cgm: np.ndarray, x_other: np.ndarray, y: np.ndarray, 
              validation_data: Optional[Tuple[Tuple[np.ndarray, np.ndarray], np.ndarray]] = None,
              epochs: int = 10, batch_size: int = 32) -> Dict[str, List[float]]:
        """
        Entrena el modelo RL con los datos proporcionados.
        
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
        # Preparar datos de validación
        x_cgm_val, x_other_val, y_val = self._unpack_validation_data(validation_data)
        
        # Historial a devolver
        history = {"loss": [], "val_loss": []}
        
        # Convertir a arrays de JAX si es necesario
        x_cgm_arr = jnp.array(x_cgm)
        x_other_arr = jnp.array(x_other)
        y_arr = jnp.array(y)
        
        if x_cgm_val is not None:
            x_cgm_val_arr = jnp.array(x_cgm_val)
            x_other_val_arr = jnp.array(x_other_val)
            y_val_arr = jnp.array(y_val)
        
        # Training loop
        for _ in range(epochs):
            # Generar nueva clave para esta época
            self.rng_key, epoch_key = jax.random.split(self.rng_key)
            
            # Entrenar por época
            epoch_loss, model_state = self._train_epoch(
                x_cgm_arr, x_other_arr, y_arr, 
                batch_size, epoch_key
            )
            
            # Actualizar el estado del modelo
            self.model.state = model_state
            
            # Guardar pérdida de entrenamiento
            history["loss"].append(float(epoch_loss))
            
            # Validar si hay datos de validación
            if x_cgm_val is not None:
                # Generar nueva clave para validación
                self.rng_key, val_key = jax.random.split(self.rng_key)
                
                # Evaluar en conjunto de validación
                val_loss = self._validate(
                    x_cgm_val_arr, x_other_val_arr, y_val_arr, 
                    val_key
                )
                
                # Guardar pérdida de validación
                history["val_loss"].append(float(val_loss))
        
        return history
    
    def _unpack_validation_data(self, validation_data: Optional[Tuple]) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray]]:
        """
        Desempaqueta los datos de validación si están disponibles.
        
        Parámetros:
        -----------
        validation_data : Optional[Tuple]
            Datos de validación como ((x_cgm_val, x_other_val), y_val)
            
        Retorna:
        --------
        Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray]]
            Tupla con (x_cgm_val, x_other_val, y_val)
        """
        x_cgm_val, x_other_val, y_val = None, None, None
        if validation_data is not None:
            (x_cgm_val, x_other_val), y_val = validation_data
        return x_cgm_val, x_other_val, y_val
    
    def _train_epoch(self, x_cgm: jnp.ndarray, x_other: jnp.ndarray, y: jnp.ndarray, 
                   batch_size: int, rng_key: jnp.ndarray) -> Tuple[float, Any]:
        """
        Entrena durante una época completa.
        
        Parámetros:
        -----------
        x_cgm : jnp.ndarray
            Datos CGM de entrenamiento
        x_other : jnp.ndarray
            Otras características de entrenamiento
        y : jnp.ndarray
            Valores objetivo
        batch_size : int
            Tamaño de lote
        rng_key : jnp.ndarray
            Clave para generación aleatoria
            
        Retorna:
        --------
        Tuple[float, Any]
            (pérdida_época, estado_modelo)
        """
        # Implementar lógica específica según la interfaz del modelo
        if hasattr(self.model, 'train_epoch'):
            return self.model.train_epoch(x_cgm, x_other, y, batch_size, rng_key)
        
        # Implementación genérica por lotes
        indices = jnp.arange(len(y))
        rng_key, shuffle_key = jax.random.split(rng_key)
        indices = jax.random.permutation(shuffle_key, indices)
        
        total_loss = 0.0
        model_state = self.model.state
        
        for i in range(0, len(y), batch_size):
            batch_idx = indices[i:i+batch_size]
            batch_x_cgm = x_cgm[batch_idx]
            batch_x_other = x_other[batch_idx]
            batch_y = y[batch_idx]
            
            rng_key, batch_key = jax.random.split(rng_key)
            loss, model_state = self._train_batch(
                batch_x_cgm, batch_x_other, batch_y, 
                model_state, batch_key
            )
            
            total_loss += loss
        
        return total_loss / ((len(y) - 1) // batch_size + 1), model_state
    
    def _train_batch(self, x_cgm: jnp.ndarray, x_other: jnp.ndarray, y: jnp.ndarray, 
                   model_state: Any, rng_key: jnp.ndarray) -> Tuple[float, Any]:
        """
        Entrena un lote de datos.
        
        Parámetros:
        -----------
        x_cgm : jnp.ndarray
            Lote de datos CGM
        x_other : jnp.ndarray
            Lote de otras características
        y : jnp.ndarray
            Lote de valores objetivo
        model_state : Any
            Estado actual del modelo
        rng_key : jnp.ndarray
            Clave para generación aleatoria
            
        Retorna:
        --------
        Tuple[float, Any]
            (pérdida_lote, nuevo_estado_modelo)
        """
        if hasattr(self.model, 'train_batch'):
            return self.model.train_batch(x_cgm, x_other, y, model_state, rng_key)
        
        # Implementación genérica
        return 0.0, model_state
    
    def _validate(self, x_cgm_val: jnp.ndarray, x_other_val: jnp.ndarray, 
                y_val: jnp.ndarray, rng_key: jnp.ndarray) -> float:
        """
        Valida el modelo.
        
        Parámetros:
        -----------
        x_cgm_val : jnp.ndarray
            Datos CGM de validación
        x_other_val : jnp.ndarray
            Otras características de validación
        y_val : jnp.ndarray
            Valores objetivo de validación
        rng_key : jnp.ndarray
            Clave para generación aleatoria
            
        Retorna:
        --------
        float
            Pérdida de validación
        """
        if hasattr(self.model, 'evaluate'):
            return self.model.evaluate(x_cgm_val, x_other_val, y_val, rng_key)
        
        # Implementación genérica
        preds = self.predict(x_cgm_val, x_other_val)
        return jnp.mean((preds - y_val) ** 2)
    
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
        # Convertir a arrays de JAX
        x_cgm_arr = jnp.array(x_cgm)
        x_other_arr = jnp.array(x_other)
        
        # Generar clave para inferencia
        self.rng_key, predict_key = jax.random.split(self.rng_key)
        
        # Realizar predicción según la interfaz del modelo
        if hasattr(self.model, 'predict_batch'):
            predictions = self.model.predict_batch(x_cgm_arr, x_other_arr, predict_key)
        else:
            # Predicción muestra por muestra
            predictions = np.zeros(len(x_cgm))
            for i in range(len(x_cgm)):
                self.rng_key, sample_key = jax.random.split(self.rng_key)
                if hasattr(self.model, 'predict_single'):
                    predictions[i] = self.model.predict_single(
                        x_cgm_arr[i], x_other_arr[i], sample_key
                    )
                elif hasattr(self.model, 'get_action'):
                    predictions[i] = self.model.get_action(
                        x_cgm_arr[i], x_other_arr[i], deterministic=True
                    )
        
        return np.array(predictions)


# Clase principal que selecciona el wrapper adecuado según el framework
class RLModelWrapper(ModelWrapper):
    """
    Wrapper para modelos de aprendizaje por refuerzo que selecciona el wrapper adecuado según el framework.
    
    Parámetros:
    -----------
    model_cls : Callable
        Clase del modelo RL a instanciar
    framework : str
        Framework a utilizar ('jax' o 'tensorflow')
    **model_kwargs
        Argumentos para el constructor del modelo
    """
    
    def __init__(self, model_cls: Callable, framework: str = 'jax', **model_kwargs) -> None:
        super().__init__()
        self.framework = framework.lower()
        
        # Seleccionar el wrapper apropiado según el framework
        if self.framework == 'tensorflow' or self.framework == 'tf':
            self.wrapper = RLModelWrapperTF(model_cls, **model_kwargs)
        else:  # default a JAX
            self.wrapper = RLModelWrapperJAX(model_cls, **model_kwargs)
    
    # Delegación de métodos al wrapper específico
    def start(self, x_cgm: np.ndarray, x_other: np.ndarray, y: np.ndarray, 
             rng_key: Any = None) -> Any:
        return self.wrapper.start(x_cgm, x_other, y, rng_key)
    
    def train(self, x_cgm: np.ndarray, x_other: np.ndarray, y: np.ndarray, 
             validation_data: Optional[Tuple[Tuple[np.ndarray, np.ndarray], np.ndarray]] = None,
             epochs: int = 10, batch_size: int = 32) -> Dict[str, List[float]]:
        return self.wrapper.train(x_cgm, x_other, y, validation_data, epochs, batch_size)
    
    def predict(self, x_cgm: np.ndarray, x_other: np.ndarray) -> np.ndarray:
        return self.wrapper.predict(x_cgm, x_other)