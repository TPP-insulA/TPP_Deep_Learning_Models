import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import jax
import jax.numpy as jnp
from flax.training import train_state
from typing import Dict, List, Tuple, Any, Optional, Callable, Union
from tqdm.auto import tqdm
import time

from custom.model_wrapper import ModelWrapper
from custom.printer import print_debug, print_info, print_log

# Constantes para mensajes de error y campos comunes
CONST_MODEL_INIT_ERROR = "El modelo debe ser inicializado antes de {}"
CONST_DEVICE = "device"
CONST_LOSS = "loss"
CONST_VAL_LOSS = "val_loss"

# Clase para modelos RL con TensorFlow
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
            # Intentar pasar las formas si el constructor las acepta
            self.model = self.model_cls(cgm_shape=cgm_shape, other_features_shape=other_shape, **self.model_kwargs)
        except TypeError:
            # Si falla, intentar sin las formas (el modelo podría no necesitarlas en __init__)
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
            # Asumiendo que setup podría necesitar las formas
            self.model.setup(cgm_shape=cgm_shape, other_features_shape=other_shape)
        elif hasattr(self.model, 'initialize'):
            self.model.initialize(cgm_shape=cgm_shape, other_features_shape=other_shape)

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
            val_data_keras = None
            if x_cgm_val is not None:
                val_data_keras = ([x_cgm_val, x_other_val], y_val)
            history = self._train_with_native_fit([x_cgm, x_other], y, val_data_keras, epochs, batch_size)
            return history

        # Entrenamiento personalizado por épocas si no hay 'fit'
        print("Iniciando entrenamiento personalizado por épocas (TF)...")
        for epoch in tqdm(range(epochs), desc="Entrenando (TF)"):
            epoch_loss = self._train_one_epoch(x_cgm, x_other, y, batch_size)
            history["loss"].append(epoch_loss)

            # Validación al final de la época
            if x_cgm_val is not None:
                val_loss = self._validate_model(x_cgm_val, x_other_val, y_val)
                history["val_loss"].append(val_loss)
                print(f"Época {epoch + 1}/{epochs} - Pérdida: {epoch_loss:.4f} - Pérdida Val: {val_loss:.4f}")
            else:
                print(f"Época {epoch + 1}/{epochs} - Pérdida: {epoch_loss:.4f}")

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

    def _train_with_native_fit(self, train_data_keras: Union[List, Tuple], y: np.ndarray,
                            val_data_keras: Optional[Tuple],
                            epochs: int, batch_size: int) -> Dict[str, List[float]]:
        """
        Entrena utilizando el método fit nativo del modelo (ej. Keras).

        Parámetros:
        -----------
        train_data_keras : Union[List, Tuple]
            Datos de entrenamiento en formato esperado por Keras (ej. [x_cgm, x_other]).
        y : np.ndarray
            Valores objetivo.
        val_data_keras : Optional[Tuple]
            Datos de validación en formato Keras (ej. ([x_cgm_val, x_other_val], y_val)).
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

        print("Iniciando entrenamiento con model.fit (TF)...")
        model_history = self.model.fit(
            train_data_keras, y,
            validation_data=val_data_keras,
            epochs=epochs,
            batch_size=batch_size,
            verbose=1 # Mostrar progreso
        )

        # Copiar historial del modelo Keras
        if hasattr(model_history, 'history'):
            history_dict = model_history.history
        elif isinstance(model_history, dict): # Algunos modelos pueden devolver un dict directamente
            history_dict = model_history

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
            batch_indices = indices[i:i + batch_size]
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
            # Interfaz Keras estándar
            loss = self.model.train_on_batch(batch_x, batch_y)
            return float(loss) if isinstance(loss, (float, np.number)) else float(loss[0]) # Keras puede devolver métricas adicionales
        elif hasattr(self.model, 'train_step'):
             # Interfaz personalizada común
             # Asumiendo que train_step devuelve un diccionario de métricas o una pérdida
             result = self.model.train_step((batch_x, batch_y))
             return float(result.get('loss', 0.0)) if isinstance(result, dict) else float(result)
        elif hasattr(self.model, 'update'):
            # Interfaz común en RL
            # Asumiendo que update toma observaciones, acciones, recompensas, etc.
            # Esto requiere adaptar el batch a lo que espera 'update'
            # Placeholder: Simular una pérdida o llamar a un método de entrenamiento por muestra
            return self._train_batch_sample_by_sample(batch_cgm, batch_other, batch_y)
        else:
            # Fallback: Entrenar muestra por muestra si no hay método por lotes
            print("Advertencia: No se encontró método de entrenamiento por lotes (train_on_batch, train_step, update). Entrenando muestra por muestra.")
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
            sample_cgm = batch_cgm[j:j+1] # Mantener dimensión de batch
            sample_other = batch_other[j:j+1]
            sample_y = batch_y[j:j+1]
            total_loss += self._train_single_sample(sample_cgm, sample_other, sample_y)
        return total_loss / len(batch_y) if len(batch_y) > 0 else 0.0

    def _train_single_sample(self, x_cgm: np.ndarray, x_other: np.ndarray, y: np.ndarray) -> float:
        """Entrena con una única muestra."""
        if hasattr(self.model, 'learn_one'): # Interfaz común en algunos agentes RL
            # Asumiendo que learn_one toma estado, acción, recompensa, siguiente_estado...
            # Necesitaríamos más contexto o datos para usar esto correctamente.
            # Placeholder: Devolver 0.0
            return 0.0
        elif hasattr(self.model, 'train_on_batch'): # Usar train_on_batch con tamaño 1
            loss = self.model.train_on_batch([x_cgm, x_other], y)
            return float(loss) if isinstance(loss, (float, np.number)) else float(loss[0])
        else:
            # Si no hay método específico, no se puede entrenar por muestra
            return 0.0


    def _validate_model(self, x_cgm_val: np.ndarray, x_other_val: np.ndarray,
                      y_val: np.ndarray) -> float:
        """Evalúa el modelo en el conjunto de validación."""
        if hasattr(self.model, 'evaluate'):
            loss = self.model.evaluate([x_cgm_val, x_other_val], y_val, verbose=0)
            return float(loss) if isinstance(loss, (float, np.number)) else float(loss[0])
        else:
            # Calcular pérdida manualmente si no hay 'evaluate'
            preds = self.predict(x_cgm_val, x_other_val)
            return float(np.mean((preds - y_val)**2)) # MSE como pérdida por defecto


    def predict(self, x_cgm: np.ndarray, x_other: np.ndarray) -> np.ndarray:
        """Realiza predicciones."""
        if self.model is None:
            raise ValueError("El modelo no ha sido inicializado. Llama a 'start' primero.")

        if hasattr(self.model, 'predict'):
            # Interfaz Keras estándar
            return self.model.predict([x_cgm, x_other])
        elif hasattr(self.model, 'act') or hasattr(self.model, 'select_action'):
            # Interfaz común en RL (actuar de forma determinista)
            actions = []
            for i in range(len(x_cgm)):
                action = self._predict_single_sample(x_cgm[i:i+1], x_other[i:i+1])
                actions.append(action)
            return np.array(actions).reshape(-1, 1) # Asegurar forma correcta
        else:
            raise NotImplementedError("El modelo no tiene un método 'predict', 'act' o 'select_action'.")


    def _predict_single_sample(self, x_cgm: np.ndarray, x_other: np.ndarray) -> Union[float, np.ndarray]:
        """Predice para una única muestra."""
        state = [x_cgm, x_other] # Asumiendo que el estado es una lista
        if hasattr(self.model, 'act'):
            # Asumiendo que act toma el estado y devuelve la acción
            return self.model.act(state, explore=False) # explore=False para predicción determinista
        elif hasattr(self.model, 'select_action'):
            return self.model.select_action(state, deterministic=True) # deterministic=True
        elif hasattr(self.model, 'predict'): # Usar predict con tamaño 1
             return self.model.predict(state)[0]
        else:
             raise NotImplementedError("No se encontró método de predicción individual.")


# Clase para modelos RL con JAX
class RLModelWrapperJAX(ModelWrapper):
    """
    Wrapper para modelos de aprendizaje por refuerzo implementados en JAX.
    Espera que el modelo subyacente implemente métodos como setup, train_batch, predict_batch, evaluate.

    Parámetros:
    -----------
    agent_creator : Callable[..., Any]
        Función que crea la instancia del agente JAX (ej. create_monte_carlo_agent).
        Debe aceptar cgm_shape y other_features_shape.
    cgm_shape : Tuple[int, ...]
        Forma de los datos CGM.
    other_features_shape : Tuple[int, ...]
        Forma de otras características.
    **model_kwargs
        Argumentos adicionales para el creador del agente.
    """

    def __init__(self, agent_creator: Callable[..., Any], cgm_shape: Tuple[int, ...], other_features_shape: Tuple[int, ...], **model_kwargs) -> None:
        super().__init__()
        self.agent_creator = agent_creator
        # Guardar formas explícitamente
        self.cgm_shape = cgm_shape
        self.other_features_shape = other_features_shape
        # Guardar kwargs, asegurándose de no duplicar las formas
        self.model_kwargs = {k: v for k, v in model_kwargs.items() if k not in ['cgm_shape', 'other_features_shape']}
        self.agent = None # El agente se instancia en start
        self.rng = None # Clave JAX principal
        self.history = {"loss": [], "val_loss": []} # Historial de entrenamiento
        self.early_stopping = None
        self.best_agent_state = None # Para guardar el mejor estado si hay early stopping

    def start(self, x_cgm: np.ndarray, x_other: np.ndarray, y: np.ndarray,
             rng_key: Optional[jax.random.PRNGKey] = None) -> Any:
        """
        Inicializa el agente RL JAX.

        Parámetros:
        -----------
        x_cgm : np.ndarray
            Datos CGM de entrada (no usados directamente aquí, formas ya guardadas).
        x_other : np.ndarray
            Otras características de entrada (no usados directamente aquí, formas ya guardadas).
        y : np.ndarray
            Valores objetivo (no usados directamente aquí).
        rng_key : Optional[jax.random.PRNGKey], opcional
            Clave JAX para inicialización (default: None, se creará una).

        Retorna:
        --------
        Any
            El agente JAX inicializado (o su estado).
        """
        if rng_key is None:
            self.rng = jax.random.PRNGKey(int(time.time())) # Usar timestamp si no hay clave
        else:
            self.rng = rng_key

        # Crear instancia del agente usando las formas guardadas y kwargs
        # Pasar las formas explícitamente y el resto de kwargs
        self.agent = self.agent_creator(
            cgm_shape=self.cgm_shape,
            other_features_shape=self.other_features_shape,
            **self.model_kwargs
        )

        # Llamar al método setup del agente si existe
        if hasattr(self.agent, 'setup'):
            setup_rng, self.rng = jax.random.split(self.rng)
            # El método setup debería devolver el estado inicial del agente
            self.agent_state = self.agent.setup(setup_rng)
        else:
            # Si no hay setup, asumimos que el propio agente es el estado
            self.agent_state = self.agent

        return self.agent_state # Devolver el estado inicial

    def add_early_stopping(self, patience: int = 10, min_delta: float = 0.0, restore_best_weights: bool = True) -> None:
        """Configura el early stopping."""
        self.early_stopping = {
            'patience': patience,
            'min_delta': min_delta,
            'restore_best_weights': restore_best_weights,
            'best_loss': float('inf'),
            'best_agent_state': None, # Guardar el estado completo del agente
            'wait': 0
        }

    def train(self, x_cgm: np.ndarray, x_other: np.ndarray, y: np.ndarray,
              validation_data: Optional[Tuple[Tuple[np.ndarray, np.ndarray], np.ndarray]] = None,
              epochs: int = 10, batch_size: int = 32) -> Dict[str, List[float]]:
        """
        Entrena el agente JAX por épocas.

        Parámetros:
        -----------
        x_cgm : np.ndarray
            Datos CGM de entrenamiento.
        x_other : np.ndarray
            Otras características de entrenamiento.
        y : np.ndarray
            Valores objetivo.
        validation_data : Optional[Tuple[Tuple[np.ndarray, np.ndarray], np.ndarray]], opcional
            Datos de validación.
        epochs : int, opcional
            Número de épocas (default: 10).
        batch_size : int, opcional
            Tamaño de lote (default: 32).

        Retorna:
        --------
        Dict[str, List[float]]
            Historial de entrenamiento.
        """
        self._check_agent_initialized()
        
        # Preparar datos
        training_data = self._prepare_training_data(x_cgm, x_other, y)
        validation_info = self._prepare_validation_data(validation_data)
        
        # Entrenar por épocas
        for epoch in tqdm(range(epochs), desc="Entrenando (JAX)"):
            self._run_training_epoch(epoch, epochs, training_data, validation_info, batch_size)
            
            # Verificar early stopping
            if validation_info['do_validation'] and self._should_stop_early():
                break

        return self.history
        
    def _check_agent_initialized(self) -> None:
        """Verifica que el agente esté inicializado."""
        if self.agent is None or self.agent_state is None:
            raise ValueError("El agente no ha sido inicializado. Llama a 'start' primero.")
            
    def _prepare_training_data(self, x_cgm: np.ndarray, x_other: np.ndarray, y: np.ndarray) -> Dict[str, jnp.ndarray]:
        """Prepara los datos de entrenamiento convirtiéndolos a JAX arrays."""
        return {
            'x_cgm': jnp.array(x_cgm),
            'x_other': jnp.array(x_other),
            'y': jnp.array(y)
        }
        
    def _prepare_validation_data(self, validation_data: Optional[Tuple]) -> Dict[str, Any]:
        """Prepara los datos de validación."""
        x_cgm_val, x_other_val, y_val = self._unpack_validation_data(validation_data)
        do_validation = x_cgm_val is not None
        
        return {
            'x_cgm': jnp.array(x_cgm_val) if do_validation else None,
            'x_other': jnp.array(x_other_val) if do_validation else None,
            'y': jnp.array(y_val) if do_validation else None,
            'do_validation': do_validation
        }
        
    def _run_training_epoch(self, epoch: int, epochs: int, training_data: Dict[str, jnp.ndarray], 
                           validation_info: Dict[str, Any], batch_size: int) -> None:
        """Ejecuta una época de entrenamiento."""
        # Entrenar época
        epoch_rng, self.rng = jax.random.split(self.rng)
        avg_loss, metrics = self._train_epoch(
            training_data['x_cgm'], 
            training_data['x_other'], 
            training_data['y'], 
            batch_size, 
            epoch_rng
        )
        self.history["loss"].append(avg_loss)
        
        # Validar si es necesario
        val_loss_str = ""
        if validation_info['do_validation']:
            val_loss = self._run_validation(validation_info)
            val_loss_str = f" - Pérdida Val: {val_loss:.4f}"
            
        # Mostrar progreso
        print(f"Época {epoch + 1}/{epochs} - Pérdida: {avg_loss:.4f}{val_loss_str}")
        print_log(metrics)
    
    def _run_validation(self, validation_info: Dict[str, Any]) -> float:
        """Ejecuta la validación y actualiza el estado del early stopping."""
        val_rng, self.rng = jax.random.split(self.rng)
        val_loss = self._validate(
            validation_info['x_cgm'], 
            validation_info['x_other'], 
            validation_info['y'], 
            val_rng
        )
        self.history["val_loss"].append(val_loss)
        
        # Actualizar estado del early stopping
        if self.early_stopping:
            self._update_early_stopping(val_loss)
            
        return val_loss
        
    def _update_early_stopping(self, val_loss: float) -> None:
        """Actualiza el estado del early stopping."""
        if val_loss < self.early_stopping['best_loss'] - self.early_stopping['min_delta']:
            # Mejoró la pérdida
            self.early_stopping['best_loss'] = val_loss
            self.early_stopping['wait'] = 0
            if self.early_stopping['restore_best_weights']:
                self.best_agent_state = self.agent_state
        else:
            # No mejoró la pérdida
            self.early_stopping['wait'] += 1
            
    def _should_stop_early(self) -> bool:
        """Determina si se debe detener el entrenamiento temprano."""
        if not self.early_stopping:
            return False
            
        if self.early_stopping['wait'] >= self.early_stopping['patience']:
            print("\nEarly stopping activado")
            if self.early_stopping['restore_best_weights'] and self.best_agent_state is not None:
                print("Restaurando el mejor estado del agente.")
                self.agent_state = self.best_agent_state
            return True
        return False

    def _unpack_validation_data(self, validation_data: Optional[Tuple]) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray]]:
        """Desempaqueta los datos de validación."""
        x_cgm_val, x_other_val, y_val = None, None, None
        if validation_data is not None:
            (x_cgm_val, x_other_val), y_val = validation_data
        return x_cgm_val, x_other_val, y_val

    def _train_epoch(self, x_cgm: jnp.ndarray, x_other: jnp.ndarray, y: jnp.ndarray,
                   batch_size: int, rng_key: jax.random.PRNGKey) -> Tuple[float, Dict[str, Any]]:
        """Entrena una época completa."""
        n_samples = len(y)
        steps_per_epoch = int(np.ceil(n_samples / batch_size))
        
        # Mezclar índices para la época
        perm_rng, rng_key = jax.random.split(rng_key)
        indices = jax.random.permutation(perm_rng, n_samples)
        
        total_loss = 0.0
        all_metrics = {}

        # Iterar sobre los lotes
        for i in range(steps_per_epoch):
            batch_indices = indices[i * batch_size:(i + 1) * batch_size]
            batch_cgm = x_cgm[batch_indices]
            batch_other = x_other[batch_indices]
            batch_y = y[batch_indices] # Los targets pueden ser necesarios para calcular recompensas o pérdidas supervisadas

            # Crear lote según lo que espere train_batch del agente
            # Asumiendo que espera (observaciones, targets/acciones)
            # Observaciones: (batch_cgm, batch_other)
            # Targets: batch_y (puede necesitar adaptación)
            # El formato exacto depende de la implementación de train_batch del agente
            batch_data = ((batch_cgm, batch_other), batch_y)

            # Llamar al método train_batch del agente
            step_rng, rng_key = jax.random.split(rng_key)
            # train_batch debe devolver (nuevo_estado_agente, métricas_paso)
            self.agent_state, step_metrics = self.agent.train_batch(self.agent_state, batch_data, step_rng)

            total_loss += step_metrics.get('loss', 0.0)
            for k, v in step_metrics.items():
                if k not in all_metrics: all_metrics[k] = 0.0
                all_metrics[k] += v

        avg_loss = total_loss / steps_per_epoch if steps_per_epoch > 0 else 0.0
        avg_metrics = {k: v / steps_per_epoch for k, v in all_metrics.items()}

        return avg_loss, avg_metrics


    def _validate(self, x_cgm_val: jnp.ndarray, x_other_val: jnp.ndarray,
                y_val: jnp.ndarray, rng_key: jax.random.PRNGKey) -> float:
        """Evalúa el modelo en el conjunto de validación."""
        if hasattr(self.agent, 'evaluate'):
            # evaluate debe devolver un diccionario de métricas, incluyendo 'loss'
            metrics = self.agent.evaluate(self.agent_state, ((x_cgm_val, x_other_val), y_val), rng_key)
            return float(metrics.get('loss', 0.0))
        else:
            # Calcular pérdida manualmente si no hay 'evaluate'
            preds = self.predict(np.array(x_cgm_val), np.array(x_other_val)) # predict espera numpy
            return float(jnp.mean((jnp.array(preds).flatten() - y_val.flatten())**2)) # MSE


    def predict(self, x_cgm: np.ndarray, x_other: np.ndarray) -> np.ndarray:
        """Realiza predicciones usando el agente JAX."""
        if self.agent is None or self.agent_state is None:
            raise ValueError("El agente no ha sido inicializado. Llama a 'start' primero.")

        if not hasattr(self.agent, 'predict_batch'):
             raise NotImplementedError("El agente JAX debe implementar 'predict_batch'.")

        # Usar una clave aleatoria dummy si predict_batch la requiere
        predict_rng, _ = jax.random.split(self.rng)

        # Convertir a JAX arrays para la predicción
        x_cgm_arr = jnp.array(x_cgm)
        x_other_arr = jnp.array(x_other)

        # predict_batch debe tomar (estado_agente, observaciones, clave_rng)
        predictions = self.agent.predict_batch(self.agent_state, (x_cgm_arr, x_other_arr), predict_rng)

        # Convertir predicciones de vuelta a NumPy array
        return np.array(predictions)

# Clase para modelos RL con PyTorch
class RLModelWrapperPyTorch(ModelWrapper):
    """
    Wrapper para modelos de aprendizaje por refuerzo implementados en PyTorch.
    
    Parámetros:
    -----------
    model_cls : Callable
        Clase del modelo RL a instanciar
    **model_kwargs
        Argumentos para el constructor del modelo
    """
    
    def __init__(self, model_cls: Callable, **model_kwargs) -> None:
        """
        Inicializa un wrapper para modelos de aprendizaje por refuerzo en PyTorch.
        
        Parámetros:
        -----------
        model_cls : Callable
            Clase del modelo RL a instanciar
        **model_kwargs
            Argumentos para el constructor del modelo
        """
        super().__init__()
        self.model_cls = model_cls
        self.model_kwargs = model_kwargs
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Crear un modelo dummy con dimensiones mínimas para satisfacer el optimizador
        self.model = self._create_dummy_model()
        self.optimizer = None
        print_info(f"Usando dispositivo: {self.device}")

    def __call__(self, *args, **kwargs):
        """
        Hace que el wrapper sea directamente invocable, delegando al método forward del modelo.
        
        Parámetros:
        -----------
        *args, **kwargs
            Argumentos a pasar al método forward del modelo
                
        Retorna:
        --------
        torch.Tensor
            Resultado del forward pass del modelo interno
        """
        if self.model is None:
            raise ValueError(CONST_MODEL_INIT_ERROR.format("llamar"))
        return self.model(*args, **kwargs)

    def _create_dummy_model(self) -> nn.Module:
        """
        Crea un modelo dummy con parámetros mínimos para inicialización
        """
        try:
            return self.model_cls()
        except Exception:
            class DummyModule(nn.Module):
                def __init__(self):
                    super().__init__()
                    self.dummy = nn.Parameter(torch.zeros(1), requires_grad=True)
                    
                def parameters(self, recurse=True):
                    return iter([self.dummy])
                    
                def forward(self, *args):
                    if len(args) == 2:
                        batch_size = args[0].size(0) if args[0].dim() > 0 else 1
                        return torch.zeros(batch_size, 1, device=args[0].device)
                    return torch.zeros(1)
                
        return DummyModule()

    def _get_input_shapes(self, x_cgm: np.ndarray, x_other: np.ndarray) -> Tuple[Tuple, Tuple]:
        """
        Extrae las formas de los datos de entrada.
        
        Parámetros:
        -----------
        x_cgm : np.ndarray
            Datos CGM de entrada
        x_other : np.ndarray
            Otras características de entrada
            
        Retorna:
        --------
        Tuple[Tuple, Tuple]
            Tupla de (cgm_shape, other_shape)
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
            Forma de los datos CGM
        other_shape : Tuple
            Forma de las otras características
        """
        try:
            # Intentar pasar las formas si el constructor las acepta
            self.model = self.model_cls(cgm_shape=cgm_shape, other_features_shape=other_shape, **self.model_kwargs)
        except TypeError:
            # Si falla, intentar sin las formas
            self.model = self.model_cls(**self.model_kwargs)
        
        # Mover modelo al dispositivo
        self.model = self.model.to(self.device)
        
        # Configurar optimizador si el modelo tiene parámetros entrenables
        if hasattr(self.model, 'parameters'):
            try:
                self.optimizer = optim.Adam(self.model.parameters(), lr=0.001, weight_decay=1e-5)
            except (ValueError, TypeError, RuntimeError) as e:
                print_debug(f"No se pudo crear optimizador automáticamente: {e}")
    
    def to(self, device):
        """
        Mueve el modelo al dispositivo especificado.
        
        Parámetros:
        -----------
        device : torch.device o str
            Dispositivo al que mover el modelo (cpu, cuda, etc.)
            
        Retorna:
        --------
        RLModelWrapperPyTorch
            El wrapper con modelo movido al dispositivo
        """
        if self.model is not None:
            self.model = self.model.to(device)
        self.device = device if isinstance(device, torch.device) else torch.device(device)
        return self
    
    def parameters(self, recurse=True):
        """
        Devuelve un iterador sobre los parámetros del modelo.
        
        Parámetros:
        -----------
        recurse : bool, opcional
            Si incluir parámetros de submodelos recursivamente (default: True)
            
        Retorna:
        --------
        iterator
            Iterador sobre los parámetros entrenables del modelo
        """
        if self.model is not None:
            return self.model.parameters(recurse=recurse)
        else:
            return iter([]) 
    
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
        # Obtener formas de entrada
        cgm_shape, other_shape = self._get_input_shapes(x_cgm, x_other)
        
        # Crear modelo si no existe
        if self.model is None:
            self._create_model_instance(cgm_shape, other_shape)
        
        # Establecer semilla si se proporciona
        if rng_key is not None:
            if isinstance(rng_key, int):
                torch.manual_seed(rng_key)
                self.rng = np.random.Generator(np.random.PCG64(rng_key))
            else:
                # Asumiendo que es un jax.random.PRNGKey o similar
                try:
                    seed_val = int(rng_key[0])
                    torch.manual_seed(seed_val)
                    self.rng = np.random.Generator(np.random.PCG64(seed_val))
                except (TypeError, IndexError):
                    # Usar valor por defecto si falla
                    seed_val = 42
                    torch.manual_seed(seed_val)
                    self.rng = np.random.Generator(np.random.PCG64(seed_val))
        
        # Inicializar modelo RL según su interfaz disponible
        if hasattr(self.model, 'setup'):
            self.model.setup(cgm_shape=cgm_shape, other_features_shape=other_shape)
        elif hasattr(self.model, 'initialize'):
            self.model.initialize(cgm_shape=cgm_shape, other_features_shape=other_shape)
        
        return self.model
    
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
    
    def _train_with_fit(self, x_cgm: np.ndarray, x_other: np.ndarray, y: np.ndarray,
                      validation_data: Optional[Tuple[np.ndarray, np.ndarray]],
                      epochs: int, batch_size: int) -> Dict[str, List[float]]:
        """
        Entrena el modelo usando su método fit nativo si está disponible.
        
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
            Número de épocas
        batch_size : int
            Tamaño de lote
            
        Retorna:
        --------
        Dict[str, List[float]]
            Historial de entrenamiento
        """
        history = {CONST_LOSS: [], CONST_VAL_LOSS: []}
        
        if validation_data:
            x_cgm_val, x_other_val, y_val = validation_data
        else:
            x_cgm_val, x_other_val, y_val = None, None, None
        
        print_info("Iniciando entrenamiento con model.fit (PyTorch)...")
        
        # Convertir datos a tensores
        x_cgm_tensor = torch.FloatTensor(x_cgm).to(self.device)
        x_other_tensor = torch.FloatTensor(x_other).to(self.device)
        y_tensor = torch.FloatTensor(y.reshape(-1, 1)).to(self.device)
        
        if validation_data:
            x_cgm_val_tensor = torch.FloatTensor(x_cgm_val).to(self.device)
            x_other_val_tensor = torch.FloatTensor(x_other_val).to(self.device)
            y_val_tensor = torch.FloatTensor(y_val.reshape(-1, 1)).to(self.device)
            val_data = (x_cgm_val_tensor, x_other_val_tensor, y_val_tensor)
        else:
            val_data = None
        
        # Llamar al método fit del modelo
        fit_history = self.model.fit(
            (x_cgm_tensor, x_other_tensor), y_tensor,
            validation_data=val_data,
            epochs=epochs,
            batch_size=batch_size,
            verbose=1
        )
        
        # Copiar el historial del modelo
        if isinstance(fit_history, dict):
            history = fit_history
        elif hasattr(fit_history, 'history'):
            history = fit_history.history
        
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
        n_samples = len(y)
        indices = np.arange(n_samples)
        
        # Mezclar datos para la época
        self.rng.shuffle(indices)
        
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
        # Convertir a tensores
        batch_cgm_tensor = torch.FloatTensor(batch_cgm).to(self.device)
        batch_other_tensor = torch.FloatTensor(batch_other).to(self.device)
        batch_y_tensor = torch.FloatTensor(batch_y.reshape(-1, 1)).to(self.device)
        
        if hasattr(self.model, 'train_on_batch'):
            # Usar método de entrenamiento por lotes
            loss = self.model.train_on_batch((batch_cgm_tensor, batch_other_tensor), batch_y_tensor)
            return float(loss) if isinstance(loss, (float, int, torch.Tensor)) else float(loss[0])
        
        elif hasattr(self.model, 'train_step'):
            # Usar paso de entrenamiento personalizado
            result = self.model.train_step(((batch_cgm_tensor, batch_other_tensor), batch_y_tensor))
            return float(result[CONST_LOSS]) if isinstance(result, dict) else float(result)
        
        elif hasattr(self.model, 'update'):
            # Interfaz común en RL
            return self._train_batch_generic(batch_cgm_tensor, batch_other_tensor, batch_y_tensor)
        
        else:
            # Fallback: Entrenar muestra por muestra
            print_debug("No se encontró método de entrenamiento por lotes. Entrenando muestra por muestra.")
            return self._train_batch_sample_by_sample(batch_cgm, batch_other, batch_y)
    
    def _train_batch_generic(self, batch_cgm: torch.Tensor, batch_other: torch.Tensor,
                           batch_y: torch.Tensor) -> float:
        """
        Entrena un lote de forma genérica usando optimizador estándar.
        
        Parámetros:
        -----------
        batch_cgm : torch.Tensor
            Lote de datos CGM
        batch_other : torch.Tensor
            Lote de otras características
        batch_y : torch.Tensor
            Lote de valores objetivo
            
        Retorna:
        --------
        float
            Pérdida del lote
        """
        if self.optimizer is None:
            return 0.0
        
        self.model.train()
        self.optimizer.zero_grad()
        
        # Forward pass
        outputs = self.model(batch_cgm, batch_other)
        
        # Calcular pérdida
        criterion = nn.MSELoss()
        loss = criterion(outputs, batch_y)
        
        # Backward pass y optimización
        loss.backward()
        self.optimizer.step()
        
        return loss.item()
    
    def _train_batch_sample_by_sample(self, batch_cgm: np.ndarray, batch_other: np.ndarray,
                                   batch_y: np.ndarray) -> float:
        """
        Entrena un lote muestra por muestra cuando no hay métodos por lotes disponibles.

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
            sample_cgm = batch_cgm[j:j+1] # Mantener dimensión de batch
            sample_other = batch_other[j:j+1]
            sample_y = batch_y[j:j+1]
            total_loss += self._train_single_sample(sample_cgm, sample_other, sample_y)
        return total_loss / len(batch_y) if len(batch_y) > 0 else 0.0
    
    def _train_single_sample(self, x_cgm: np.ndarray, x_other: np.ndarray, y: np.ndarray) -> float:
        """
        Entrena con una única muestra.
        
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
            Pérdida para esta muestra
        """
        # Convertir a tensores
        x_cgm_tensor = torch.FloatTensor(x_cgm).to(self.device)
        x_other_tensor = torch.FloatTensor(x_other).to(self.device)
        y_tensor = torch.FloatTensor(y.reshape(-1, 1)).to(self.device)
        
        if hasattr(self.model, 'learn_one'):
            # Interfaz común en algunos agentes RL
            return self.model.learn_one((x_cgm_tensor, x_other_tensor), y_tensor)
        
        elif hasattr(self.model, 'update_one'):
            return self.model.update_one((x_cgm_tensor, x_other_tensor), y_tensor)
        
        elif self.optimizer is not None:
            # Entrenar con enfoque genérico
            return self._train_batch_generic(x_cgm_tensor, x_other_tensor, y_tensor)
        
        else:
            return 0.0
    
    def _validate_model(self, x_cgm_val: np.ndarray, x_other_val: np.ndarray,
                      y_val: np.ndarray) -> float:
        """
        Evalúa el modelo en el conjunto de validación.
        
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
        # Convertir a tensores
        x_cgm_tensor = torch.FloatTensor(x_cgm_val).to(self.device)
        x_other_tensor = torch.FloatTensor(x_other_val).to(self.device)
        y_tensor = torch.FloatTensor(y_val.reshape(-1, 1)).to(self.device)
        
        if hasattr(self.model, 'evaluate'):
            # Usar método evaluate del modelo
            return self.model.evaluate((x_cgm_tensor, x_other_tensor), y_tensor)
        
        else:
            # Calcular pérdida manualmente
            self.model.eval()
            with torch.no_grad():
                outputs = self.model(x_cgm_tensor, x_other_tensor)
                criterion = nn.MSELoss()
                loss = criterion(outputs, y_tensor)
                return loss.item()
    
    def train(self, mode=True):
        """
        Sets the module in training mode (standard PyTorch method).
        
        Parámetros:
        -----------
        mode : bool, opcional
            Si True, activa el modo de entrenamiento; si False, modo de evaluación (default: True)
            
        Retorna:
        --------
        RLModelWrapperPyTorch
            Self para encadenamiento de llamadas
        """
        if self.model is not None:
            self.model.train(mode)
        return self
    
    def fit(self, x_cgm: np.ndarray, x_other: np.ndarray, y: np.ndarray, 
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
        if self.model is None:
            self.start(x_cgm, x_other, y)
        
        # Preparar datos de validación
        x_cgm_val, x_other_val, y_val = self._unpack_validation_data(validation_data)
        
        # Historial de entrenamiento
        history = {CONST_LOSS: [], CONST_VAL_LOSS: []}
        
        # Usar interfaz nativa del modelo si está disponible
        if hasattr(self.model, 'fit'):
            return self._train_with_fit(
                x_cgm, x_other, y,
                (x_cgm_val, x_other_val, y_val) if x_cgm_val is not None else None,
                epochs, batch_size
            )
        
        # Entrenamiento personalizado por épocas
        print_info("Iniciando entrenamiento personalizado por épocas (PyTorch)...")
        for epoch in tqdm(range(epochs), desc="Entrenando (PyTorch)"):
            # Entrenar época
            epoch_loss = self._train_one_epoch(x_cgm, x_other, y, batch_size)
            history[CONST_LOSS].append(epoch_loss)
            
            # Validación al final de la época
            if x_cgm_val is not None:
                val_loss = self._validate_model(x_cgm_val, x_other_val, y_val)
                history[CONST_VAL_LOSS].append(val_loss)
                print_info(f"Época {epoch + 1}/{epochs} - Pérdida: {epoch_loss:.4f} - Pérdida Val: {val_loss:.4f}")
            else:
                print_info(f"Época {epoch + 1}/{epochs} - Pérdida: {epoch_loss:.4f}")
        
        return history
    
    def predict(self, x_cgm: np.ndarray, x_other: np.ndarray) -> np.ndarray:
        """
        Realiza predicciones usando el modelo RL.
        
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
        
        # Para modelos de RL se pueden requerir predicciones deterministas
        deterministic = True
        
        if hasattr(self.model, 'predict'):
            # Usar método predict del modelo
            x_cgm_tensor = torch.FloatTensor(x_cgm).to(self.device)
            x_other_tensor = torch.FloatTensor(x_other).to(self.device)
            
            with torch.no_grad():
                self.model.eval()
                predictions = self.model.predict((x_cgm_tensor, x_other_tensor))
                
                if isinstance(predictions, torch.Tensor):
                    return predictions.cpu().numpy()
                else:
                    return np.array(predictions)
        
        elif hasattr(self.model, 'act') or hasattr(self.model, 'select_action'):
            # Interfaz común en RL (actuar de forma determinista)
            preds = []
            for i in range(len(x_cgm)):
                action = self._predict_single_sample(x_cgm[i:i+1], x_other[i:i+1], deterministic)
                preds.append(action)
            return np.array(preds).reshape(-1, 1)
        
        else:
            # Evaluación genérica muestra por muestra
            preds = []
            for i in range(len(x_cgm)):
                x_cgm_tensor = torch.FloatTensor(x_cgm[i:i+1]).to(self.device)
                x_other_tensor = torch.FloatTensor(x_other[i:i+1]).to(self.device)
                
                with torch.no_grad():
                    self.model.eval()
                    output = self.model(x_cgm_tensor, x_other_tensor)
                    preds.append(output.cpu().numpy())
            
            return np.vstack(preds).flatten()
    
    def _predict_single_sample(self, x_cgm: np.ndarray, x_other: np.ndarray, deterministic: bool = True) -> Union[float, np.ndarray]:
        """
        Predice para una única muestra usando métodos específicos de RL.
        
        Parámetros:
        -----------
        x_cgm : np.ndarray
            Datos CGM de una muestra
        x_other : np.ndarray
            Otras características de una muestra
        deterministic : bool, opcional
            Si usar comportamiento determinista (para inferencia) (default: True)
            
        Retorna:
        --------
        Union[float, np.ndarray]
            Predicción para la muestra
        """
        # Convertir a tensores
        x_cgm_tensor = torch.FloatTensor(x_cgm).to(self.device)
        x_other_tensor = torch.FloatTensor(x_other).to(self.device)
        state = (x_cgm_tensor, x_other_tensor)
        
        if hasattr(self.model, 'act'):
            # Asumiendo que act toma el estado y devuelve la acción
            action = self.model.act(state, explore=not deterministic)
            if isinstance(action, torch.Tensor):
                return action.cpu().numpy()
            return action
        
        elif hasattr(self.model, 'select_action'):
            action = self.model.select_action(state, deterministic=deterministic)
            if isinstance(action, torch.Tensor):
                return action.cpu().numpy()
            return action
        
        elif hasattr(self.model, 'predict'):
            # Usar predict con tamaño 1
            pred = self.model.predict(state)
            if isinstance(pred, torch.Tensor):
                return pred.cpu().numpy().flatten()[0]
            return pred[0]
        
        else:
            # Usar forward directamente
            with torch.no_grad():
                self.model.eval()
                output = self.model(x_cgm_tensor, x_other_tensor)
                if isinstance(output, torch.Tensor):
                    return output.cpu().numpy().flatten()[0]
                return output[0]
    
    def eval(self):
        """
        Sets the module in evaluation mode (standard PyTorch method).
        
        Retorna:
        --------
        RLModelWrapperPyTorch
            Self para encadenamiento de llamadas
        """
        if self.model is not None:
            self.model.eval()
        return self
    
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
        
        if hasattr(self.model, 'evaluate') and callable(self.model.evaluate):
            # Convertir a tensores
            x_cgm_tensor = torch.FloatTensor(x_cgm).to(self.device)
            x_other_tensor = torch.FloatTensor(x_other).to(self.device)
            y_tensor = torch.FloatTensor(y.reshape(-1, 1)).to(self.device)
            
            # Llamar a evaluate del modelo
            metrics = self.model.evaluate((x_cgm_tensor, x_other_tensor), y_tensor)
            
            # Verificar si devuelve un diccionario o un valor único
            if isinstance(metrics, dict):
                return {k: float(v) if isinstance(v, torch.Tensor) else v for k, v in metrics.items()}
            else:
                loss = float(metrics) if isinstance(metrics, torch.Tensor) else metrics
                return {CONST_LOSS: loss}
        
        # Calcular métricas personalizadas
        preds = self.predict(x_cgm, x_other)
        mse = float(np.mean((preds - y) ** 2))
        mae = float(np.mean(np.abs(preds - y)))
        rmse = float(np.sqrt(mse))
        
        # Calcular R²
        ss_res = np.sum((y - preds) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        r2 = float(1 - (ss_res / (ss_tot + 1e-8)))
        
        return {
            CONST_LOSS: mse,
            "mae": mae,
            "rmse": rmse,
            "r2": r2
        }
    
    def state_dict(self):
        """
        Returns a dictionary containing a whole state of the module.
        
        Retorna:
        --------
        Dict[str, torch.Tensor]
            Estado del modelo (parámetros y buffers)
        """
        if self.model is not None:
            return self.model.state_dict()
        else:
            return {}
            
    def load_state_dict(self, state_dict):
        """
        Copies parameters and buffers from state_dict into this module.
        
        Parámetros:
        -----------
        state_dict : Dict[str, torch.Tensor]
            Estado del modelo a cargar
        """
        if self.model is not None:
            self.model.load_state_dict(state_dict)
        return self

# Clase principal que selecciona el wrapper adecuado según el framework
class RLModelWrapper(ModelWrapper):
    """
    Wrapper para modelos de aprendizaje por refuerzo que selecciona el wrapper adecuado según el framework.

    Parámetros:
    -----------
    model_creator_func : Callable
        Función que crea la instancia del agente RL (ej. create_monte_carlo_agent para JAX,
        o la clase del modelo para TF). Debe aceptar cgm_shape y other_features_shape.
    framework : str, opcional
        Framework a utilizar ('jax' o 'tensorflow') (default: 'jax').
    cgm_shape : Tuple[int, ...]
        Forma de los datos CGM (necesario para JAX wrapper).
    other_features_shape : Tuple[int, ...]
        Forma de otras características (necesario para JAX wrapper).
    **model_kwargs
        Argumentos adicionales para el creador del agente/modelo.
    """

    def __init__(self, model_creator_func: Callable, framework: str = 'jax', cgm_shape: Optional[Tuple[int,...]]=None, other_features_shape: Optional[Tuple[int,...]]=None, **model_kwargs) -> None:
        super().__init__()
        self.framework = framework
        if framework == 'jax':
            if cgm_shape is None or other_features_shape is None:
                 raise ValueError("cgm_shape y other_features_shape son requeridos para el framework JAX.")
            # Pasar formas y kwargs al wrapper JAX
            self.wrapper = RLModelWrapperJAX(model_creator_func, cgm_shape, other_features_shape, **model_kwargs)
        elif framework == 'tensorflow':
            self.wrapper = RLModelWrapperTF(model_creator_func, **model_kwargs)
        elif framework == 'pytorch':
            self.wrapper = RLModelWrapperPyTorch(model_creator_func, **model_kwargs)
        else:
            raise ValueError(f"Framework no soportado: {framework}")

    # Delegación de métodos al wrapper específico
    def start(self, x_cgm: np.ndarray, x_other: np.ndarray, y: np.ndarray,
             rng_key: Any = None) -> Any:
        """Inicializa el modelo/agente."""
        return self.wrapper.start(x_cgm, x_other, y, rng_key)

    def train(self, x_cgm: np.ndarray, x_other: np.ndarray, y: np.ndarray,
             validation_data: Optional[Tuple[Tuple[np.ndarray, np.ndarray], np.ndarray]] = None,
             epochs: int = 10, batch_size: int = 32) -> Dict[str, List[float]]:
        """Entrena el modelo/agente."""
        return self.wrapper.train(x_cgm, x_other, y, validation_data, epochs, batch_size)

    def predict(self, x_cgm: np.ndarray, x_other: np.ndarray) -> np.ndarray:
        """Realiza predicciones."""
        return self.wrapper.predict(x_cgm, x_other)

    def save(self, filepath: str) -> None:
        """Guarda el modelo/agente (si el wrapper lo soporta)."""
        if hasattr(self.wrapper, 'save'):
            self.wrapper.save(filepath)
        else:
            print(f"Advertencia: El guardado no está implementado para el wrapper {type(self.wrapper).__name__}.")

    def load(self, filepath: str) -> None:
        """Carga el modelo/agente (si el wrapper lo soporta)."""
        if hasattr(self.wrapper, 'load'):
            self.wrapper.load(filepath)
        else:
            print(f"Advertencia: La carga no está implementada para el wrapper {type(self.wrapper).__name__}.")

    def add_early_stopping(self, patience: int = 10, min_delta: float = 0.0, restore_best_weights: bool = True) -> None:
        """Configura early stopping (si el wrapper lo soporta)."""
        if hasattr(self.wrapper, 'add_early_stopping'):
            self.wrapper.add_early_stopping(patience, min_delta, restore_best_weights)
        else:
            print(f"Advertencia: Early stopping no está implementado para el wrapper {type(self.wrapper).__name__}.")
