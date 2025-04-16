from typing import Dict, List, Tuple, Any, Optional, Callable, Union
import numpy as np
import jax
import jax.numpy as jnp
from custom.model_wrapper import ModelWrapper
from tqdm.auto import tqdm # Para barra de progreso

# Clase para modelos RL con TensorFlow (sin cambios significativos, añadir docstrings)
class RLModelWrapperTF(ModelWrapper):
    """
    Wrapper para modelos de aprendizaje por refuerzo implementados en TensorFlow.

    Parámetros:
    -----------
    model_cls : Callable
        Clase del modelo RL a instanciar.
    **model_kwargs
        Argumentos para el constructor del modelo.
    """

    def __init__(self, model_cls: Callable, **model_kwargs) -> None:
        super().__init__()
        self.model_cls = model_cls
        self.model_kwargs = model_kwargs
        self.model = None # El modelo se instancia en start
        self.rng = np.random.default_rng(model_kwargs.get('seed', 42)) # Generador aleatorio para mezclar datos

    def _get_input_shapes(self, x_cgm: np.ndarray, x_other: np.ndarray) -> Tuple[Tuple, Tuple]:
        """
        Extrae las formas de los datos de entrada.
        
        Parámetros:
        -----------
        x_cgm : np.ndarray
            Datos CGM de entrada.
        x_other : np.ndarray
            Otras características de entrada.
            
        Retorna:
        --------
        Tuple[Tuple, Tuple]
            Tupla de (cgm_shape, other_shape).
        """
        cgm_shape = x_cgm.shape[1:] if x_cgm.ndim > 1 else (1,)
        other_shape = x_other.shape[1:] if x_other.ndim > 1 else (1,)
        return cgm_shape, other_shape
        
    def _create_model_instance(self, cgm_shape: Tuple, other_shape: Tuple) -> None:
        """
        Crea una instancia del modelo si aún no existe.
        
        Parámetros:
        -----------
        cgm_shape : Tuple
            Forma de los datos CGM.
        other_shape : Tuple
            Forma de las otras características.
        """
        try:
            self.model = self.model_cls(cgm_shape=cgm_shape, other_features_shape=other_shape, **self.model_kwargs)
        except TypeError:
            # Si el modelo no acepta formas en __init__
            self.model = self.model_cls(**self.model_kwargs)
            
    def start(self, x_cgm: np.ndarray, x_other: np.ndarray, y: np.ndarray,
             rng_key: Any = None) -> Any:
        """
        Inicializa el agente RL con las dimensiones del problema.

        Parámetros:
        -----------
        x_cgm : np.ndarray
            Datos CGM de entrada.
        x_other : np.ndarray
            Otras características de entrada.
        y : np.ndarray
            Valores objetivo.
        rng_key : Any, opcional
            Clave para generación aleatoria (no usada en TF wrapper) (default: None).

        Retorna:
        --------
        Any
            Estado del modelo inicializado (la instancia del modelo TF).
        """
        # Obtener formas de entrada
        cgm_shape, other_shape = self._get_input_shapes(x_cgm, x_other)
        
        # Crear modelo si no existe
        if self.model is None:
            self._create_model_instance(cgm_shape, other_shape)

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
            Datos CGM de entrenamiento.
        x_other : np.ndarray
            Otras características de entrenamiento.
        y : np.ndarray
            Valores objetivo.
        validation_data : Optional[Tuple[Tuple[np.ndarray, np.ndarray], np.ndarray]], opcional
            Datos de validación como ((x_cgm_val, x_other_val), y_val) (default: None).
        epochs : int, opcional
            Número de épocas de entrenamiento (default: 10).
        batch_size : int, opcional
            Tamaño de lote (default: 32).

        Retorna:
        --------
        Dict[str, List[float]]
            Historial de entrenamiento con métricas.
        """
        # Preparar datos de validación y el historial
        x_cgm_val, x_other_val, y_val = self._unpack_validation_data(validation_data)
        history = {"loss": [], "val_loss": []}

        # Usar interfaz nativa del modelo si está disponible (ej. Keras fit)
        if hasattr(self.model, 'fit'):
            return self._train_with_native_fit(
                x_cgm, x_other, y,
                (x_cgm_val, x_other_val, y_val) if x_cgm_val is not None else None,
                epochs, batch_size
            )

        # Entrenamiento personalizado por épocas si no hay 'fit'
        print("Iniciando entrenamiento personalizado por épocas (TF)...")
        for epoch in tqdm(range(epochs), desc="Entrenando (TF)"):
            # Entrenar una época
            epoch_loss = self._train_one_epoch(x_cgm, x_other, y, batch_size)
            history["loss"].append(epoch_loss)

            # Validar si hay datos de validación
            val_loss = 0.0
            if x_cgm_val is not None and y_val is not None:
                val_loss = self._validate_model(x_cgm_val, x_other_val, y_val)
                history["val_loss"].append(val_loss)

            print(f"Época {epoch+1}/{epochs} - loss: {epoch_loss:.4f}" + (f" - val_loss: {val_loss:.4f}" if x_cgm_val is not None else ""))


        return history

    def _unpack_validation_data(self, validation_data: Optional[Tuple]) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray]]:
        """
        Desempaqueta los datos de validación si están disponibles.

        Parámetros:
        -----------
        validation_data : Optional[Tuple]
            Datos de validación como ((x_cgm_val, x_other_val), y_val).

        Retorna:
        --------
        Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray]]
            Tupla con (x_cgm_val, x_other_val, y_val).
        """
        x_cgm_val, x_other_val, y_val = None, None, None
        if validation_data is not None:
            (x_cgm_val, x_other_val), y_val = validation_data
        return x_cgm_val, x_other_val, y_val

    def _train_with_native_fit(self, x_cgm: np.ndarray, x_other: np.ndarray, y: np.ndarray,
                            validation_data: Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]],
                            epochs: int, batch_size: int) -> Dict[str, List[float]]:
        """
        Entrena utilizando el método fit nativo del modelo (ej. Keras).

        Parámetros:
        -----------
        x_cgm : np.ndarray
            Datos CGM de entrenamiento.
        x_other : np.ndarray
            Otras características de entrenamiento.
        y : np.ndarray
            Valores objetivo.
        validation_data : Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]]
            Datos de validación como (x_cgm_val, x_other_val, y_val).
        epochs : int
            Número de épocas de entrenamiento.
        batch_size : int
            Tamaño de lote.

        Retorna:
        --------
        Dict[str, List[float]]
            Historial de entrenamiento con métricas.
        """
        history_dict = {"loss": [], "val_loss": []}

        # Preparar validation_data para el modelo nativo
        val_data_keras = None
        if validation_data is not None:
            x_cgm_val, x_other_val, y_val = validation_data
            # Asumiendo que el modelo espera una lista o tupla de entradas
            val_data_keras = ([x_cgm_val, x_other_val], y_val)

        # Asumiendo que el modelo espera una lista o tupla de entradas
        train_data_keras = [x_cgm, x_other]

        print("Iniciando entrenamiento con model.fit (TF)...")
        model_history = self.model.fit(
            train_data_keras, y,
            validation_data=val_data_keras,
            epochs=epochs,
            batch_size=batch_size,
            verbose=1 # Mostrar progreso de Keras
        )

        # Copiar historial del modelo Keras
        if hasattr(model_history, 'history'):
            for key, values in model_history.history.items():
                history_dict[key] = values
        elif isinstance(model_history, dict): # Si fit devuelve un dict directamente
            for key, values in model_history.items():
                history_dict[key] = values

        return history_dict

    def _train_one_epoch(self, x_cgm: np.ndarray, x_other: np.ndarray, y: np.ndarray,
                       batch_size: int) -> float:
        """
        Entrena durante una época completa y devuelve la pérdida promedio.

        Parámetros:
        -----------
        x_cgm : np.ndarray
            Datos CGM de entrenamiento.
        x_other : np.ndarray
            Otras características de entrenamiento.
        y : np.ndarray
            Valores objetivo.
        batch_size : int
            Tamaño de lote.

        Retorna:
        --------
        float
            Pérdida promedio de la época.
        """
        epoch_loss = 0.0
        num_batches = 0
        n_samples = len(y)
        indices = np.arange(n_samples)
        self.rng.shuffle(indices) # Mezclar datos para la época

        # Entrenar por lotes
        for i in range(0, n_samples, batch_size):
            batch_indices = indices[i:min(i + batch_size, n_samples)]
            batch_cgm = x_cgm[batch_indices]
            batch_other = x_other[batch_indices]
            batch_y = y[batch_indices]

            batch_loss = self._train_batch(batch_cgm, batch_other, batch_y)
            epoch_loss += batch_loss
            num_batches += 1

        # Calcular pérdida promedio
        return epoch_loss / num_batches if num_batches > 0 else 0.0

    def _train_batch(self, batch_cgm: np.ndarray, batch_other: np.ndarray,
                   batch_y: np.ndarray) -> float:
        """
        Entrena con un lote de datos. Busca métodos específicos del modelo.

        Parámetros:
        -----------
        batch_cgm : np.ndarray
            Lote de datos CGM.
        batch_other : np.ndarray
            Lote de otras características.
        batch_y : np.ndarray
            Lote de valores objetivo.

        Retorna:
        --------
        float
            Pérdida del lote.
        """
        # Asumiendo que el modelo espera una lista de entradas
        batch_x = [batch_cgm, batch_other]

        if hasattr(self.model, 'train_on_batch'):
            # Interfaz común de Keras
            metrics = self.model.train_on_batch(batch_x, batch_y, return_dict=True)
            return metrics.get('loss', 0.0) # Devolver la pérdida
        elif hasattr(self.model, 'train_step'):
             # Interfaz alternativa
             # Puede requerir adaptar la entrada/salida
             result = self.model.train_step(batch_x, batch_y)
             return result.get('loss', 0.0) if isinstance(result, dict) else float(result)
        elif hasattr(self.model, 'update'):
            # Interfaz común en algunos agentes RL
            result = self.model.update(batch_x, batch_y) # Asumiendo que update acepta lotes
            return result.get('loss', 0.0) if isinstance(result, dict) else float(result)
        else:
            # Fallback: entrenar muestra por muestra si no hay método de lote
            print("Advertencia: No se encontró método de entrenamiento por lotes (train_on_batch, train_step, update). Usando entrenamiento muestra por muestra.")
            return self._train_batch_sample_by_sample(batch_cgm, batch_other, batch_y)

    def _train_batch_sample_by_sample(self, batch_cgm: np.ndarray, batch_other: np.ndarray,
                                   batch_y: np.ndarray) -> float:
        """
        Entrena un lote muestra por muestra cuando no hay métodos por lotes disponibles.

        Parámetros:
        -----------
        batch_cgm : np.ndarray
            Lote de datos CGM.
        batch_other : np.ndarray
            Lote de otras características.
        batch_y : np.ndarray
            Lote de valores objetivo.

        Retorna:
        --------
        float
            Pérdida promedio del lote.
        """
        total_loss = 0.0
        for j in range(len(batch_y)):
            sample_loss = self._train_single_sample(
                batch_cgm[j:j+1], batch_other[j:j+1], batch_y[j:j+1]
            )
            total_loss += sample_loss
        return total_loss / len(batch_y) if len(batch_y) > 0 else 0.0

    def _train_single_sample(self, x_cgm: np.ndarray, x_other: np.ndarray, y: np.ndarray) -> float:
        """
        Entrena con una sola muestra. Busca métodos específicos del modelo.

        Parámetros:
        -----------
        x_cgm : np.ndarray
            Datos CGM de una muestra (con dimensión de lote 1).
        x_other : np.ndarray
            Otras características de una muestra (con dimensión de lote 1).
        y : np.ndarray
            Valor objetivo de una muestra (con dimensión de lote 1).

        Retorna:
        --------
        float
            Pérdida de la muestra.
        """
        sample_x = [x_cgm, x_other] # Mantener formato de lista

        if hasattr(self.model, 'train_on_batch'): # Reutilizar train_on_batch con tamaño 1
            metrics = self.model.train_on_batch(sample_x, y, return_dict=True)
            return metrics.get('loss', 0.0)
        elif hasattr(self.model, 'update_single'):
            # Método específico para una muestra
            result = self.model.update_single(sample_x, y[0]) # Asumiendo que espera y escalar
            return result.get('loss', 0.0) if isinstance(result, dict) else float(result)
        elif hasattr(self.model, 'learn'):
             # Interfaz común en Stable Baselines
             # Requiere adaptación significativa del formato de datos a (obs, action, reward, next_obs, done)
             print("Advertencia: El método 'learn' requiere adaptación de datos no implementada en este wrapper genérico.")
             return 0.0
        else:
            # Implementación genérica si no hay método específico
            print(f"Advertencia: No se encontró método de entrenamiento para una sola muestra en {type(self.model)}. Devolviendo pérdida 0.")
            return 0.0

    def _validate_model(self, x_cgm_val: np.ndarray, x_other_val: np.ndarray,
                      y_val: np.ndarray) -> float:
        """
        Valida el modelo con datos de validación.

        Parámetros:
        -----------
        x_cgm_val : np.ndarray
            Datos CGM de validación.
        x_other_val : np.ndarray
            Otras características de validación.
        y_val : np.ndarray
            Valores objetivo de validación.

        Retorna:
        --------
        float
            Pérdida de validación (MSE por defecto, o usa model.evaluate).
        """
        val_x = [x_cgm_val, x_other_val]

        if hasattr(self.model, 'evaluate'):
            # Usar método evaluate si existe (ej. Keras)
            metrics = self.model.evaluate(val_x, y_val, verbose=0, return_dict=True)
            return metrics.get('loss', 0.0)
        else:
            # Calcular MSE manualmente si no hay evaluate
            preds = self.predict(x_cgm_val, x_other_val)
            if preds.shape != y_val.shape:
                 preds = preds.reshape(y_val.shape) # Intentar ajustar forma
            return float(np.mean((preds - y_val) ** 2))


    def predict(self, x_cgm: np.ndarray, x_other: np.ndarray) -> np.ndarray:
        """
        Realiza predicciones con el modelo entrenado.

        Parámetros:
        -----------
        x_cgm : np.ndarray
            Datos CGM para predicción.
        x_other : np.ndarray
            Otras características para predicción.

        Retorna:
        --------
        np.ndarray
            Predicciones del modelo.
        """
        pred_x = [x_cgm, x_other]

        # Verificar si el modelo soporta predicción por lotes
        if hasattr(self.model, 'predict'):
            # Interfaz Keras/TF común
            return self.model.predict(pred_x)
        elif hasattr(self.model, 'get_action') and callable(getattr(self.model, 'get_action')):
             # Interfaz común de agente RL (puede necesitar adaptación de estado)
             # Esto asume que el estado se puede derivar directamente de x_cgm, x_other
             print("Advertencia: Usando 'get_action' para predicción. Puede requerir adaptación de estado.")
             predictions = []
             for i in range(len(x_cgm)):
                 # Crear estado/observación para get_action (requiere conocimiento específico del modelo)
                 # Simplificación: pasar las características directamente
                 obs = (x_cgm[i], x_other[i])
                 # Asumiendo que get_action toma la observación y devuelve la acción
                 # El argumento 'deterministic=True' es común para evaluación
                 try:
                      action = self.model.get_action(obs, deterministic=True)
                 except TypeError:
                      action = self.model.get_action(obs) # Probar sin deterministic
                 predictions.append(action)
             return np.array(predictions)

        # Fallback: Predicción muestra por muestra si no hay método de lote
        print("Advertencia: No se encontró método de predicción por lotes. Usando predicción muestra por muestra.")
        predictions = np.zeros((len(x_cgm),)) # Asumiendo salida escalar
        for i in range(len(x_cgm)):
            predictions[i] = self._predict_single_sample(
                x_cgm[i:i+1], x_other[i:i+1]
            )
        # Intentar inferir la forma de salida correcta si no es escalar
        if hasattr(self.model, 'output_shape'):
             output_dim = self.model.output_shape[-1]
             if output_dim > 1:
                  predictions = predictions.reshape(-1, output_dim)
        return predictions


    def _predict_single_sample(self, x_cgm: np.ndarray, x_other: np.ndarray) -> Union[float, np.ndarray]:
        """
        Predice para una sola muestra.

        Parámetros:
        -----------
        x_cgm : np.ndarray
            Datos CGM de una muestra (con dimensión de lote 1).
        x_other : np.ndarray
            Otras características de una muestra (con dimensión de lote 1).

        Retorna:
        --------
        Union[float, np.ndarray]
            Predicción para la muestra.
        """
        sample_x = [x_cgm, x_other]

        if hasattr(self.model, 'predict'): # Reutilizar predict con tamaño 1
            return self.model.predict(sample_x)[0] # Devolver la primera (y única) predicción
        elif hasattr(self.model, 'predict_single'):
            return self.model.predict_single(sample_x)
        elif hasattr(self.model, 'get_action'):
            # Adaptación similar a predict, pero para una muestra
            obs = (x_cgm[0], x_other[0]) # Quitar dimensión de lote
            try:
                action = self.model.get_action(obs, deterministic=True)
            except TypeError:
                action = self.model.get_action(obs)
            return action # Devolver la acción directamente
        else:
            # Si no hay método específico
            print(f"Advertencia: No se encontró método de predicción para una sola muestra en {type(self.model)}. Devolviendo 0.")
            return 0.0


# Clase para modelos RL con JAX (Modificada)
class RLModelWrapperJAX(ModelWrapper):
    """
    Wrapper para modelos de aprendizaje por refuerzo implementados en JAX.
    Espera que el modelo subyacente implemente métodos como setup, train_batch, predict_batch, evaluate.

    Parámetros:
    -----------
    agent_creator : Callable[..., Any]
        Función que crea la instancia del agente JAX (ej. create_monte_carlo_agent).
        Debe aceptar cgm_shape y other_features_shape.
    **model_kwargs
        Argumentos adicionales para el creador del agente.
    """

    def __init__(self, agent_creator: Callable[..., Any], **model_kwargs) -> None:
        super().__init__()
        self.agent_creator = agent_creator
        self.model_kwargs = model_kwargs
        self.agent: Optional[Any] = None # Instancia del agente JAX
        self.model_state: Optional[Dict[str, Any]] = None # Estado JAX (params, opt_state, etc.)
        self.base_rng_key = jax.random.PRNGKey(model_kwargs.get('seed', 42)) # Clave base

    def start(self, x_cgm: np.ndarray, x_other: np.ndarray, y: np.ndarray,
             rng_key: Optional[jax.random.PRNGKey] = None) -> Any:
        """
        Inicializa el agente JAX y su estado.

        Parámetros:
        -----------
        x_cgm : np.ndarray
            Datos CGM de entrada (usados para determinar forma).
        x_other : np.ndarray
            Otras características de entrada (usadas para determinar forma).
        y : np.ndarray
            Valores objetivo (no usados directamente en start).
        rng_key : Optional[jax.random.PRNGKey], opcional
            Clave PRNG externa (si se proporciona, reemplaza la clave base) (default: None).

        Retorna:
        --------
        Any
            El estado inicial del modelo JAX.
        """
        if rng_key is not None:
            self.base_rng_key = rng_key
        init_key, self.base_rng_key = jax.random.split(self.base_rng_key) # Usar una subclave para inicializar

        if self.agent is None:
            # Determinar dimensiones de entrada
            cgm_shape = x_cgm.shape[1:] if x_cgm.ndim > 1 else (1,)
            other_shape = x_other.shape[1:] if x_other.ndim > 1 else (1,)

            # Crear la instancia del agente JAX
            self.agent = self.agent_creator(cgm_shape=cgm_shape, other_features_shape=other_shape, **self.model_kwargs)

        # Inicializar el estado del agente JAX usando su método setup
        if hasattr(self.agent, 'setup'):
            self.model_state = self.agent.setup(init_key)
        else:
            # Si no hay setup, el estado podría ser manejado internamente por el agente
            # o necesitar inicialización manual aquí. Asumimos que setup existe.
            raise AttributeError(f"El agente {type(self.agent)} no tiene un método 'setup(rng_key)'.")

        if self.model_state is None:
             raise ValueError("El método 'setup' del agente no devolvió un estado inicial.")

        return self.model_state

    def train(self, x_cgm: np.ndarray, x_other: np.ndarray, y: np.ndarray,
              validation_data: Optional[Tuple[Tuple[np.ndarray, np.ndarray], np.ndarray]] = None,
              epochs: int = 10, batch_size: int = 32) -> Dict[str, List[float]]:
        """
        Entrena el modelo RL/JAX con los datos proporcionados.

        Parámetros:
        -----------
        x_cgm : np.ndarray
            Datos CGM de entrenamiento.
        x_other : np.ndarray
            Otras características de entrenamiento.
        y : np.ndarray
            Valores objetivo.
        validation_data : Optional[Tuple[Tuple[np.ndarray, np.ndarray], np.ndarray]], opcional
            Datos de validación como ((x_cgm_val, x_other_val), y_val) (default: None).
        epochs : int, opcional
            Número de épocas de entrenamiento (default: 10).
        batch_size : int, opcional
            Tamaño de lote (default: 32).

        Retorna:
        --------
        Dict[str, List[float]]
            Historial de entrenamiento con métricas.
        """
        if self.agent is None or self.model_state is None:
            raise RuntimeError("El método 'start' debe ser llamado antes de 'train'.")

        # Preparar datos de validación
        x_cgm_val, x_other_val, y_val = self._unpack_validation_data(validation_data)

        # Historial a devolver
        history = {"loss": [], "val_loss": []}

        # Convertir datos a arrays de JAX
        x_cgm_train_arr = jnp.array(x_cgm)
        x_other_train_arr = jnp.array(x_other)
        y_train_arr = jnp.array(y)

        x_cgm_val_arr, x_other_val_arr, y_val_arr = None, None, None
        if x_cgm_val is not None:
            x_cgm_val_arr = jnp.array(x_cgm_val)
            x_other_val_arr = jnp.array(x_other_val)
            y_val_arr = jnp.array(y_val)

        # Bucle de entrenamiento
        print(f"Iniciando entrenamiento del agente JAX: {type(self.agent).__name__}...")
        for epoch in tqdm(range(epochs), desc="Entrenando (JAX)"):
            # Generar nueva clave para esta época
            epoch_key, self.base_rng_key = jax.random.split(self.base_rng_key)

            # Entrenar por época
            epoch_loss, self.model_state = self._train_epoch(
                x_cgm_train_arr, x_other_train_arr, y_train_arr,
                batch_size, epoch_key
            )

            # Guardar pérdida de entrenamiento
            history["loss"].append(float(epoch_loss))

            # Validar si hay datos de validación
            val_loss = 0.0
            if x_cgm_val_arr is not None:
                # Generar nueva clave para validación
                val_key, self.base_rng_key = jax.random.split(self.base_rng_key)

                # Evaluar en conjunto de validación
                val_loss = self._validate(
                    x_cgm_val_arr, x_other_val_arr, y_val_arr,
                    val_key
                )
                # Guardar pérdida de validación
                history["val_loss"].append(float(val_loss))

            print(f"Época {epoch+1}/{epochs} - loss: {epoch_loss:.4f}" + (f" - val_loss: {val_loss:.4f}" if x_cgm_val_arr is not None else ""))


        return history

    def _unpack_validation_data(self, validation_data: Optional[Tuple]) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray]]:
        """
        Desempaqueta los datos de validación si están disponibles.

        Parámetros:
        -----------
        validation_data : Optional[Tuple]
            Datos de validación como ((x_cgm_val, x_other_val), y_val).

        Retorna:
        --------
        Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray]]
            Tupla con (x_cgm_val, x_other_val, y_val).
        """
        x_cgm_val, x_other_val, y_val = None, None, None
        if validation_data is not None:
            (x_cgm_val, x_other_val), y_val = validation_data
        return x_cgm_val, x_other_val, y_val

    def _train_epoch(self, x_cgm: jnp.ndarray, x_other: jnp.ndarray, y: jnp.ndarray,
                   batch_size: int, rng_key: jax.random.PRNGKey) -> Tuple[float, Dict[str, Any]]:
        """
        Entrena durante una época completa.

        Parámetros:
        -----------
        x_cgm : jnp.ndarray
            Datos CGM de entrenamiento.
        x_other : jnp.ndarray
            Otras características de entrenamiento.
        y : jnp.ndarray
            Valores objetivo.
        batch_size : int
            Tamaño de lote.
        rng_key : jax.random.PRNGKey
            Clave PRNG para la época.

        Retorna:
        --------
        Tuple[float, Dict[str, Any]]
            (pérdida_promedio_época, nuevo_estado_modelo).
        """
        n_samples = len(y)
        n_batches = (n_samples + batch_size - 1) // batch_size
        indices = jnp.arange(n_samples)

        # Mezclar índices para la época
        rng_key, shuffle_key = jax.random.split(rng_key)
        indices = jax.random.permutation(shuffle_key, indices)

        total_loss = 0.0
        current_model_state = self.model_state # Empezar con el estado actual

        # Iterar sobre los lotes
        for i in range(n_batches):
            start_idx = i * batch_size
            end_idx = jnp.minimum(start_idx + batch_size, n_samples)
            batch_idx = indices[start_idx:end_idx]

            batch_x_cgm = x_cgm[batch_idx]
            batch_x_other = x_other[batch_idx]
            batch_y = y[batch_idx]

            # Generar clave para el lote
            rng_key, _ = jax.random.split(rng_key) # Usar subclave para el paso

            # Ejecutar paso de entrenamiento del agente
            if hasattr(self.agent, 'train_batch'):
                 # Pasar la clave PRNG al método train_batch si la necesita
                 # Asumiendo que train_batch toma (estado, xcgm, xother, y, clave_opcional)
                 # y devuelve (nuevo_estado, pérdida_lote)
                 try:
                     current_model_state, batch_loss = self.agent.train_batch(
                         current_model_state, batch_x_cgm, batch_x_other, batch_y #, batch_key # Descomentar si train_batch necesita la clave
                     )
                 except TypeError as e:
                      raise TypeError(f"El método 'train_batch' del agente {type(self.agent)} tiene una firma inesperada: {e}")

                 total_loss += batch_loss
            else:
                 raise AttributeError(f"El agente {type(self.agent)} no tiene un método 'train_batch'.")


        avg_epoch_loss = total_loss / n_batches if n_batches > 0 else 0.0
        return avg_epoch_loss, current_model_state


    def _validate(self, x_cgm_val: jnp.ndarray, x_other_val: jnp.ndarray,
                y_val: jnp.ndarray, rng_key: jax.random.PRNGKey) -> float:
        """
        Valida el modelo usando el método evaluate del agente.

        Parámetros:
        -----------
        x_cgm_val : jnp.ndarray
            Datos CGM de validación.
        x_other_val : jnp.ndarray
            Otras características de validación.
        y_val : jnp.ndarray
            Valores objetivo de validación.
        rng_key : jax.random.PRNGKey
            Clave PRNG para la evaluación (puede no ser usada).

        Retorna:
        --------
        float
            Pérdida de validación.
        """
        if hasattr(self.agent, 'evaluate'):
             # Asumiendo que evaluate toma (estado, xcgm, xother, y, clave_opcional)
             try:
                 val_loss = self.agent.evaluate(
                     self.model_state, x_cgm_val, x_other_val, y_val #, rng_key # Descomentar si evaluate necesita la clave
                 )
                 return float(val_loss)
             except TypeError as e:
                 raise TypeError(f"El método 'evaluate' del agente {type(self.agent)} tiene una firma inesperada: {e}")

        else:
            # Implementación genérica si evaluate no existe
            print("Advertencia: El agente no tiene método 'evaluate'. Calculando MSE manualmente.")
            preds = self.predict(np.array(x_cgm_val), np.array(x_other_val)) # Usar predict del wrapper
            return float(jnp.mean((jnp.array(preds) - y_val) ** 2))


    def predict(self, x_cgm: np.ndarray, x_other: np.ndarray) -> np.ndarray:
        """
        Realiza predicciones con el modelo entrenado usando el método predict_batch del agente.

        Parámetros:
        -----------
        x_cgm : np.ndarray
            Datos CGM para predicción.
        x_other : np.ndarray
            Otras características para predicción.

        Retorna:
        --------
        np.ndarray
            Predicciones del modelo.
        """
        if self.agent is None or self.model_state is None:
            raise RuntimeError("El método 'start' debe ser llamado antes de 'predict'.")

        # Convertir a arrays de JAX
        x_cgm_arr = jnp.array(x_cgm)
        x_other_arr = jnp.array(x_other)

        # Generar clave para inferencia
        _, self.base_rng_key = jax.random.split(self.base_rng_key)

        # Realizar predicción según la interfaz del agente
        if hasattr(self.agent, 'predict_batch'):
             # Asumiendo que predict_batch toma (estado, xcgm, xother, clave_opcional)
             try:
                 predictions_jax = self.agent.predict_batch(
                     self.model_state, x_cgm_arr, x_other_arr #, predict_key # Descomentar si predict_batch necesita clave
                 )
             except TypeError as e:
                 raise TypeError(f"El método 'predict_batch' del agente {type(self.agent)} tiene una firma inesperada: {e}")

        else:
             raise AttributeError(f"El agente {type(self.agent)} no tiene un método 'predict_batch'.")

        # Convertir predicciones JAX a NumPy array para el resto del pipeline
        return np.array(predictions_jax)


# Clase principal que selecciona el wrapper adecuado según el framework
class RLModelWrapper(ModelWrapper):
    """
    Wrapper para modelos de aprendizaje por refuerzo que selecciona el wrapper adecuado según el framework.

    Parámetros:
    -----------
    model_creator_func : Callable
        Función que crea la instancia del agente RL (ej. create_monte_carlo_agent para JAX,
        o la clase del modelo para TF).
    framework : str, opcional
        Framework a utilizar ('jax' o 'tensorflow') (default: 'jax').
    **model_kwargs
        Argumentos adicionales para el creador del agente/modelo.
    """

    def __init__(self, model_creator_func: Callable, framework: str = 'jax', **model_kwargs) -> None:
        super().__init__()
        self.framework = framework.lower()

        # Seleccionar el wrapper apropiado según el framework
        if self.framework == 'tensorflow' or self.framework == 'tf':
            # Para TF, model_creator_func suele ser la clase misma
            self.wrapper = RLModelWrapperTF(model_creator_func, **model_kwargs)
        elif self.framework == 'jax':
             # Para JAX, model_creator_func es la función que crea el agente
             self.wrapper = RLModelWrapperJAX(model_creator_func, **model_kwargs)
        else:
            raise ValueError(f"Framework no soportado: {framework}. Use 'tensorflow' o 'jax'.")

    # Delegación de métodos al wrapper específico
    def start(self, x_cgm: np.ndarray, x_other: np.ndarray, y: np.ndarray,
             rng_key: Any = None) -> Any:
        """ Llama al método start del wrapper específico del framework. """
        return self.wrapper.start(x_cgm, x_other, y, rng_key)

    def train(self, x_cgm: np.ndarray, x_other: np.ndarray, y: np.ndarray,
             validation_data: Optional[Tuple[Tuple[np.ndarray, np.ndarray], np.ndarray]] = None,
             epochs: int = 10, batch_size: int = 32) -> Dict[str, List[float]]:
        """ Llama al método train del wrapper específico del framework. """
        return self.wrapper.train(x_cgm, x_other, y, validation_data, epochs, batch_size)

    def predict(self, x_cgm: np.ndarray, x_other: np.ndarray) -> np.ndarray:
        """ Llama al método predict del wrapper específico del framework. """
        return self.wrapper.predict(x_cgm, x_other)

    def save(self, filepath: str) -> None:
        """ Llama al método save del agente/modelo subyacente si existe. """
        if hasattr(self.wrapper.agent, 'save_state'): # Para JAX
            self.wrapper.agent.save_state(filepath, self.wrapper.model_state)
        elif hasattr(self.wrapper.model, 'save'): # Para TF/Keras
            self.wrapper.model.save(filepath)
        else:
            print(f"Advertencia: El agente/modelo {type(self.wrapper.agent or self.wrapper.model)} no tiene método 'save' o 'save_state'.")

    def load(self, filepath: str) -> None:
        """ Llama al método load del agente/modelo subyacente si existe. """
        if self.framework == 'jax':
            if hasattr(self.wrapper.agent, 'load_state'):
                self.wrapper.model_state = self.wrapper.agent.load_state(filepath)
            else:
                 print(f"Advertencia: El agente JAX {type(self.wrapper.agent)} no tiene método 'load_state'.")
        elif self.framework == 'tensorflow':
             # Cargar modelos TF/Keras requiere más contexto (a menudo se carga fuera del wrapper)
             print(f"Advertencia: La carga de modelos TF/Keras generalmente se maneja externamente. No se cargó {filepath}.")
        else:
             print(f"Advertencia: No se implementó la carga para el framework {self.framework}.")
