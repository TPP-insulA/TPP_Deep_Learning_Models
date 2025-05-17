import os
import numpy as np
from flax.training.checkpoints import save_checkpoint, restore_checkpoint
import pickle
import tensorflow as tf
from tqdm.auto import tqdm
from typing import Dict, List, Tuple, Any, Optional, Callable, Union

from config.models_config import EARLY_STOPPING_POLICY
from constants.constants import CONST_DEFAULT_SEED, CONST_DEFAULT_EPOCHS, CONST_DEFAULT_BATCH_SIZE
from custom.model_wrapper import ModelWrapper
from custom.printer import print_debug, print_info, print_log, print_success, print_error, print_warning

# Constantes para mensajes de error y campos comunes
CONST_MODEL_INIT_ERROR = "El modelo debe ser inicializado antes de {}"
CONST_DEVICE = "device"
CONST_LOSS = "loss"
CONST_VAL_LOSS = "val_loss"

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
        self.rng = np.random.default_rng(model_kwargs.get('seed', CONST_DEFAULT_SEED)) # Generador aleatorio para mezclar datos

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
             epochs: int = CONST_DEFAULT_EPOCHS, batch_size: int = CONST_DEFAULT_BATCH_SIZE) -> Dict[str, List[float]]:
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
    
    def save(self, path: str) -> None:
        """
        Guarda el modelo en disco.
        
        Parámetros:
        -----------
        path : str
            Ruta donde guardar el modelo
        
        Retorna:
        --------
        None
        """
        if self.model is None:
            raise ValueError(CONST_MODEL_INIT_ERROR.format("guardar"))
        
        if hasattr(self.model, 'save'):
            # Método estándar de Keras
            self.model.save(path)
        elif hasattr(self.model, 'save_weights'):
            # Algunos modelos RL solo pueden guardar pesos
            self.model.save_weights(path)
        else:
            try:
                # Intentar exportar como SavedModel si es posible
                tf.saved_model.save(self.model, path)
            except Exception as e:
                # Si todo falla, intentar guardar con pickle
                with open(f"{path}.pkl", 'wb') as f:
                    pickle.dump({
                        'model_kwargs': self.model_kwargs,
                        'model_weights': self.model.get_weights() if hasattr(self.model, 'get_weights') else None
                    }, f)
                print_warning(f"Modelo guardado como pickle en {path}.pkl debido a: {e}")

    def load(self, path: str) -> None:
        """
        Carga el modelo desde disco.
        
        Parámetros:
        -----------
        path : str
            Ruta desde donde cargar el modelo
        
        Retorna:
        --------
        None
        """
        # Verificar si es un archivo .pkl
        if path.endswith('.pkl'):
            self._load_from_pickle(path)
        # Intentar cargar como modelo Keras estándar
        elif os.path.isdir(path) or path.endswith('.h5') or path.endswith('.keras'):
            self._load_keras_model(path)
        else:
            raise ValueError(f"Formato de archivo no reconocido: {path}")
            
    def _load_from_pickle(self, path: str) -> None:
        """
        Carga el modelo desde un archivo pickle.
        
        Parámetros:
        -----------
        path : str
            Ruta del archivo pickle
        """
        with open(path, 'rb') as f:
            data = pickle.load(f)
            
        # Recrear el modelo si es necesario
        if self.model is None:
            self._create_model_instance(data.get('cgm_shape', (1,)), data.get('other_shape', (1,)))
            
        # Establecer pesos si están disponibles
        if 'model_weights' in data and data['model_weights'] is not None and hasattr(self.model, 'set_weights'):
            self.model.set_weights(data['model_weights'])
            
    def _load_keras_model(self, path: str) -> None:
        """
        Carga un modelo Keras desde un archivo o directorio.
        
        Parámetros:
        -----------
        path : str
            Ruta del archivo o directorio del modelo Keras
        """
        try:
            # Cargar modelo completo
            self.model = tf.keras.models.load_model(path)
        except Exception as e:
            print_error(f"Error al cargar modelo completo: {e}")
            # Intentar cargar solo los pesos
            if self.model is None:
                self._create_model_instance((1,), (1,))
            if hasattr(self.model, 'load_weights'):
                self.model.load_weights(path)
