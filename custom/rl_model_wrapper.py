from custom.model_wrapper import ModelWrapper
import jax
import jax.numpy as jnp
import numpy as np
from typing import Dict, List, Tuple, Any, Optional, Callable

class RLModelWrapper(ModelWrapper):
    """
    Implementación de ModelWrapper para modelos de aprendizaje por refuerzo.
    """
    
    def __init__(self, model_cls: Callable, **model_kwargs):
        """
        Inicializa un wrapper para modelos de aprendizaje por refuerzo.
        
        Parámetros:
        -----------
        model_cls : Callable
            Clase del modelo RL a instanciar
        **model_kwargs
            Argumentos para el constructor del modelo
        """
        super().__init__()
        self.model = model_cls(**model_kwargs)
    
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
        if rng_key is None:
            rng_key = jax.random.PRNGKey(0)
        
        # Determinar dimensiones de entrada
        cgm_shape = x_cgm.shape[1:] if x_cgm.ndim > 1 else (1,)
        other_shape = x_other.shape[1:] if x_other.ndim > 1 else (1,)
        
        # Inicializar modelo RL según su interfaz disponible
        if hasattr(self.model, 'setup'):
            self.model.setup(
                observation_shape=(cgm_shape, other_shape),
                action_dim=1,  # Para regresión
                rng_key=rng_key
            )
        elif hasattr(self.model, 'initialize'):
            self.model.initialize(x_cgm, x_other, y, rng_key)
        
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
        validation_data_unpacked = self._unpack_validation_data(validation_data)
        history = {"loss": [], "val_loss": []}
        
        # Usar interfaz nativa del modelo si está disponible
        if hasattr(self.model, 'fit'):
            return self._train_with_native_fit(x_cgm, x_other, y, validation_data, epochs, batch_size)
        
        # Entrenamiento personalizado por épocas
        for epoch in range(epochs):
            # Entrenar y actualizar el historial para esta época
            avg_loss = self._train_one_epoch(x_cgm, x_other, y, batch_size)
            history["loss"].append(float(avg_loss))
            
            # Validación si hay datos disponibles
            val_loss = self._validate_model(*validation_data_unpacked)
            history["val_loss"].append(float(val_loss))
            
            print(f"Época {epoch+1}/{epochs}, loss: {avg_loss:.4f}, val_loss: {history['val_loss'][-1]:.4f}")
        
        return history
        
    def _unpack_validation_data(self, validation_data):
        """Desempaqueta los datos de validación si están disponibles."""
        x_cgm_val, x_other_val, y_val = None, None, None
        if validation_data is not None:
            (x_cgm_val, x_other_val), y_val = validation_data
        return x_cgm_val, x_other_val, y_val
    
    def _train_with_native_fit(self, x_cgm, x_other, y, validation_data, epochs, batch_size):
        """Entrena utilizando el método fit nativo del modelo."""
        history = {"loss": [], "val_loss": []}
        model_history = self.model.fit(
            [x_cgm, x_other], y,
            validation_data=validation_data,
            epochs=epochs,
            batch_size=batch_size
        )
        # Copiar historial del modelo
        for key, values in model_history.items():
            history[key] = values
        return history
    
    def _train_one_epoch(self, x_cgm, x_other, y, batch_size):
        """Entrena durante una época completa y devuelve la pérdida promedio."""
        epoch_loss = 0.0
        num_batches = 0
        
        # Entrenar por lotes
        for i in range(0, len(y), batch_size):
            end = min(i + batch_size, len(y))
            batch_loss = self._train_batch(
                x_cgm[i:end], 
                x_other[i:end], 
                y[i:end]
            )
            epoch_loss += batch_loss
            num_batches += 1
        
        # Calcular pérdida promedio
        return epoch_loss / num_batches if num_batches > 0 else 0.0
    
    def _train_batch(self, batch_cgm, batch_other, batch_y):
        """Entrena con un lote de datos."""
        if hasattr(self.model, 'train_batch'):
            return self.model.train_batch(batch_cgm, batch_other, batch_y)
        elif hasattr(self.model, 'update'):
            return self.model.update(batch_cgm, batch_other, batch_y)
        else:
            return self._train_batch_sample_by_sample(batch_cgm, batch_other, batch_y)
    
    def _train_batch_sample_by_sample(self, batch_cgm, batch_other, batch_y):
        """Entrena un lote muestra por muestra cuando no hay métodos por lotes."""
        total_loss = 0.0
        for j in range(len(batch_y)):
            sample_loss = self._train_single_sample(
                batch_cgm[j:j+1], batch_other[j:j+1], batch_y[j:j+1]
            )
            total_loss += sample_loss
        return total_loss / len(batch_y) if len(batch_y) > 0 else 0.0
    
    def _validate_model(self, x_cgm_val, x_other_val, y_val):
        """Valida el modelo con datos de validación si están disponibles."""
        if x_cgm_val is not None and y_val is not None:
            return self.evaluate(x_cgm_val, x_other_val, y_val)
        return 0.0
    
    def _train_single_sample(self, x_cgm: np.ndarray, x_other: np.ndarray, y: np.ndarray) -> float:
        """
        Entrena con una sola muestra, adaptándose a la interfaz disponible.
        
        Parámetros:
        -----------
        x_cgm : np.ndarray
            Datos CGM de una muestra
        x_other : np.ndarray
            Otras características de una muestra
        y : np.ndarray
            Valor objetivo
            
        Retorna:
        --------
        float
            Pérdida de entrenamiento
        """
        loss = 0.0
        if hasattr(self.model, 'train_step'):
            loss = self.model.train_step(x_cgm, x_other, y)
        elif hasattr(self.model, 'update_sample'):
            loss = self.model.update_sample(x_cgm, x_other, y)
        elif hasattr(self.model, 'learn'):
            loss = self.model.learn(x_cgm, x_other, y)
        
        return float(loss)
    
    def predict(self, x_cgm: np.ndarray, x_other: np.ndarray) -> np.ndarray:
        """
        Realiza predicciones con el modelo RL entrenado.
        
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
        if hasattr(self.model, 'predict'):
            return self.model.predict([x_cgm, x_other])
        
        # Si no hay método predict, predecir muestra por muestra
        y_pred = np.zeros((len(x_cgm),))
        for i in range(len(x_cgm)):
            y_pred[i] = self._predict_single_sample(x_cgm[i:i+1], x_other[i:i+1])
        
        return y_pred
    
    def _predict_single_sample(self, x_cgm: np.ndarray, x_other: np.ndarray) -> float:
        """
        Predice para una sola muestra, adaptándose a la interfaz disponible.
        
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
        if hasattr(self.model, 'inference'):
            return self.model.inference(x_cgm, x_other)
        elif hasattr(self.model, '__call__'):
            return self.model(x_cgm, x_other)
        elif hasattr(self.model, 'get_action'):
            return self.model.get_action(x_cgm[0], x_other[0])
        elif hasattr(self.model, 'act'):
            return self.model.act(x_cgm[0], x_other[0])
        
        raise NotImplementedError("No se encontró un método de predicción compatible")