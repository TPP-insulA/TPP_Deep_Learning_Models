from typing import Dict, List, Tuple, Any, Optional, Callable
import numpy as np
import os
import pickle
import tensorflow as tf

from constants.constants import CONST_DEFAULT_SEED, CONST_DEFAULT_EPOCHS, CONST_DEFAULT_BATCH_SIZE
from config.models_config import EARLY_STOPPING_POLICY
from custom.model_wrapper import ModelWrapper
from custom.printer import print_debug, print_info, print_warning, print_error, print_success

# Constantes para uso repetido
CONST_ACTOR = "actor"
CONST_CRITIC = "critic"
CONST_TARGET = "target"
CONST_PARAMS = "params"
CONST_DEVICE = "device"
CONST_MODEL_INIT_ERROR = "El modelo debe ser inicializado antes de {}"
CONST_LOSS = "loss"
CONST_VAL_LOSS = "val_loss"
CONST_EPSILON = 1e-10


class DRLModelWrapperTF(ModelWrapper):
    """
    Wrapper para modelos de aprendizaje por refuerzo profundo implementados en TensorFlow.
    
    Parámetros:
    -----------
    model_cls : Callable
        Clase del modelo DRL a instanciar
    **model_kwargs
        Argumentos para el constructor del modelo
    """
    
    def __init__(self, model_cls: Callable, **model_kwargs) -> None:
        """
        Inicializa un wrapper para modelos de aprendizaje por refuerzo profundo en TensorFlow.
        
        Parámetros:
        -----------
        model_cls : Callable
            Clase del modelo DRL a instanciar
        **model_kwargs
            Argumentos para el constructor del modelo
        """
        super().__init__()
        self.model_cls = model_cls
        self.model_kwargs = model_kwargs
        self.model = None
        self.buffer = None
        self.algorithm = model_kwargs.get('algorithm', 'generic')
    
    def start(self, x_cgm: np.ndarray, x_other: np.ndarray, y: np.ndarray, 
             rng_key: Any = None) -> Any:
        """
        Inicializa el modelo DRL con los datos de entrada.
        
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
            self.model = self.model_cls(**self.model_kwargs)
        
        # Dimensiones del espacio de estados y acciones
        state_dim = (x_cgm.shape[1:], x_other.shape[1:])
        action_dim = 1  # Para regresión en el caso de dosis de insulina
        
        # Inicializar el modelo según su interfaz disponible
        if hasattr(self.model, 'initialize'):
            self.model.initialize(state_dim, action_dim)
        elif hasattr(self.model, 'setup'):
            self.model.setup(state_dim, action_dim)
        elif hasattr(self.model, 'build'):
            # Crear datos dummy para build
            x_cgm_dummy = np.zeros((1,) + x_cgm.shape[1:])
            x_other_dummy = np.zeros((1,) + x_other.shape[1:])
            self.model.build([x_cgm_dummy.shape, x_other_dummy.shape])
        
        # Inicializar buffer de experiencia si el modelo lo requiere
        if hasattr(self.model, 'init_buffer'):
            self.buffer = self.model.init_buffer()
        
        return self.model
    
    def _unpack_validation_data(self, validation_data: Optional[Tuple]) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray]]:
        """
        Desempaqueta los datos de validación si están disponibles.
        
        Parámetros:
        -----------
        validation_data : Optional[Tuple]
            Datos de validación en formato ((x_cgm_val, x_other_val), y_val)
            
        Retorna:
        --------
        Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray]]
            Tupla con (x_cgm_val, x_other_val, y_val)
        """
        x_cgm_val, x_other_val, y_val = None, None, None
        if validation_data is not None:
            (x_cgm_val, x_other_val), y_val = validation_data
        return x_cgm_val, x_other_val, y_val
    
    def _compute_rewards(self, x_cgm: np.ndarray, x_other: np.ndarray, y: np.ndarray) -> np.ndarray:
        """
        Calcula recompensas a partir de los datos y objetivos.
        
        Parámetros:
        -----------
        x_cgm : np.ndarray
            Datos CGM de entrada
        x_other : np.ndarray
            Otras características de entrada
        y : np.ndarray
            Valores objetivo
            
        Retorna:
        --------
        np.ndarray
            Recompensas calculadas para el entrenamiento RL
        """
        # Si el modelo proporciona una función para calcular recompensas, úsala
        if hasattr(self.model, 'compute_rewards'):
            return self.model.compute_rewards(x_cgm, x_other, y)
        
        # Implementación por defecto: recompensa negativa basada en el error
        predicted = self.predict(x_cgm, x_other)
        error = np.abs(predicted - y)
        # Normalizar error al rango [-1, 0] donde -1 es el peor error y 0 es perfecto
        max_error = np.max(error) if np.max(error) > 0 else 1.0
        rewards = -error / max_error
        return rewards
    
    def _fill_buffer(self, x_cgm: np.ndarray, x_other: np.ndarray, rewards: np.ndarray, y: np.ndarray) -> None:
        """
        Llena el buffer de experiencia con transiciones.
        
        Parámetros:
        -----------
        x_cgm : np.ndarray
            Estados (datos CGM)
        x_other : np.ndarray
            Estados (otras características)
        rewards : np.ndarray
            Recompensas calculadas
        y : np.ndarray
            Acciones objetivo (dosis)
        """
        if self.buffer is None or not hasattr(self.model, 'add_to_buffer'):
            return
        
        for i in range(len(rewards)):
            state = (x_cgm[i], x_other[i])
            action = y[i]
            reward = rewards[i]
            # En un entorno supervisado, el siguiente estado puede ser el mismo
            # y done es siempre True (episodio de un paso)
            next_state = (x_cgm[i], x_other[i])
            done = True
            self.model.add_to_buffer(state, action, reward, next_state, done)
    
    def _update_networks(self, batch_size: int, updates_per_batch: int = 1) -> Dict[str, float]:
        """
        Actualiza las redes del modelo usando experiencias del buffer.
        
        Parámetros:
        -----------
        batch_size : int
            Tamaño del lote para actualizaciones
        updates_per_batch : int
            Número de actualizaciones por lote
            
        Retorna:
        --------
        Dict[str, float]
            Diccionario con métricas de actualización
        """
        metrics = {
            "actor_loss": 0.0,
            "critic_loss": 0.0,
            "q_loss": 0.0,
            "total_loss": 0.0
        }
        
        if not hasattr(self.model, 'update'):
            return metrics
        
        for _ in range(updates_per_batch):
            update_metrics = self.model.update(batch_size)
            if isinstance(update_metrics, dict):
                for key, value in update_metrics.items():
                    if key in metrics:
                        metrics[key] += value
            elif isinstance(update_metrics, (int, float)):
                metrics["total_loss"] += update_metrics
        
        # Promediar las métricas
        for key in metrics:
            metrics[key] /= updates_per_batch
            
        return metrics
    
    def _process_epoch_metrics(self, epoch_metrics: Dict[str, float], 
                               updates_per_epoch: int, history: Dict[str, List[float]]) -> float:
        """
        Procesa y registra las métricas de una época.
        
        Parámetros:
        -----------
        epoch_metrics : Dict[str, float]
            Métricas acumuladas de la época
        updates_per_epoch : int
            Número de actualizaciones por época
        history : Dict[str, List[float]]
            Historial de entrenamiento a actualizar
            
        Retorna:
        --------
        float
            Pérdida total calculada
        """
        # Promediar métricas de la época
        for key in epoch_metrics:
            epoch_metrics[key] /= updates_per_epoch
            if key in history:
                history[key].append(epoch_metrics[key])
        
        # Pérdida total para seguimiento
        total_loss = epoch_metrics["total_loss"]
        if abs(total_loss) < CONST_EPSILON:
            # Si no hay pérdida total (o es muy cercana a cero), usar la suma de otras pérdidas
            total_loss = sum(epoch_metrics[key] for key in ["actor_loss", "critic_loss", "q_loss"])
        
        return total_loss
    
    def _run_epoch_updates(self, batch_size: int, updates_per_epoch: int) -> Dict[str, float]:
        """
        Ejecuta las actualizaciones durante una época.
        
        Parámetros:
        -----------
        batch_size : int
            Tamaño del lote
        updates_per_epoch : int
            Número de actualizaciones por época
            
        Retorna:
        --------
        Dict[str, float]
            Métricas acumuladas de la época
        """
        epoch_metrics = {key: 0.0 for key in ["actor_loss", "critic_loss", "q_loss", "total_loss"]}
        
        for _ in range(updates_per_epoch):
            batch_metrics = self._update_networks(batch_size)
            for key, value in batch_metrics.items():
                epoch_metrics[key] += value
                
        return epoch_metrics
    
    def _validate_model(self, x_cgm_val: np.ndarray, x_other_val: np.ndarray, 
                       y_val: np.ndarray, history: Dict[str, List[float]]) -> None:
        """
        Realiza validación del modelo y registra la pérdida.
        
        Parámetros:
        -----------
        x_cgm_val : np.ndarray
            Datos CGM de validación
        x_other_val : np.ndarray
            Otras características de validación
        y_val : np.ndarray
            Valores objetivo de validación
        history : Dict[str, List[float]]
            Historial donde registrar la pérdida de validación
        """
        if x_cgm_val is not None and y_val is not None:
            val_preds = self.predict(x_cgm_val, x_other_val)
            val_loss = float(np.mean((val_preds - y_val) ** 2))
            history["val_loss"].append(val_loss)
        
    def fit(self, x_cgm: np.ndarray, x_other: np.ndarray, y: np.ndarray, 
           validation_data: Optional[Tuple[Tuple[np.ndarray, np.ndarray], np.ndarray]] = None,
           epochs: int = CONST_DEFAULT_EPOCHS, batch_size: int = CONST_DEFAULT_BATCH_SIZE, verbose: int = 1) -> Dict[str, List[float]]:
        """
        Entrena el modelo DRL con los datos proporcionados.
        
        Parámetros:
        -----------
        x_cgm : np.ndarray
            Datos CGM de entrenamiento
        x_other : np.ndarray
            Otras características de entrenamiento
        y : np.ndarray
            Valores objetivo (acciones)
        validation_data : Optional[Tuple[Tuple[np.ndarray, np.ndarray], np.ndarray]], opcional
            Datos de validación como ((x_cgm_val, x_other_val), y_val) (default: None)
        epochs : int, opcional
            Número de épocas de entrenamiento (default: 10)
        batch_size : int, opcional
            Tamaño de lote (default: 32)
        verbose : int, opcional
            Nivel de verbosidad (0=silencioso, 1=progreso, 2=detallado)
            
        Retorna:
        --------
        Dict[str, List[float]]
            Historial de entrenamiento con métricas
        """
        # Preparar datos de validación
        x_cgm_val, x_other_val, y_val = self._unpack_validation_data(validation_data)
        
        # Historial de entrenamiento
        history = {
            "loss": [],
            "actor_loss": [],
            "critic_loss": [],
            "q_loss": [],
            "val_loss": []
        }
        
        # Si el modelo tiene un método fit nativo, úsalo
        if hasattr(self.model, 'fit'):
            return self._train_with_fit(
                x_cgm, x_other, y,
                (x_cgm_val, x_other_val, y_val) if x_cgm_val is not None else None,
                epochs, batch_size
            )
        
        # Entrenamiento RL personalizado
        for _ in range(epochs):
            # Calcular recompensas y llenar buffer
            rewards = self._compute_rewards(x_cgm, x_other, y)
            self._fill_buffer(x_cgm, x_other, rewards, y)
            
            # Actualizar redes y procesar métricas
            updates_per_epoch = max(1, len(x_cgm) // batch_size)
            epoch_metrics = self._run_epoch_updates(batch_size, updates_per_epoch)
            total_loss = self._process_epoch_metrics(epoch_metrics, updates_per_epoch, history)
            history["loss"].append(total_loss)
            
            # Validación
            self._validate_model(x_cgm_val, x_other_val, y_val, history)
        
        return history
    
    def _train_with_fit(self, x_cgm: np.ndarray, x_other: np.ndarray, y: np.ndarray,
                      validation_data: Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]],
                      epochs: int, batch_size: int) -> Dict[str, List[float]]:
        """
        Entrena el modelo usando su método fit nativo.
        
        Parámetros:
        -----------
        x_cgm : np.ndarray
            Datos CGM de entrenamiento
        x_other : np.ndarray
            Otras características de entrenamiento
        y : np.ndarray
            Valores objetivo (acciones)
        validation_data : Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]]
            Datos de validación
        epochs : int
            Número de épocas
        batch_size : int
            Tamaño de lote
            
        Retorna:
        --------
        Dict[str, List[float]]
            Historial de entrenamiento
        """
        history = {"loss": [], "val_loss": []}
        fit_history = self.model.fit(
            [x_cgm, x_other], y,
            validation_data=validation_data,
            epochs=epochs,
            batch_size=batch_size
        )
        
        # Copiar el historial del modelo
        for key, values in fit_history.history.items():
            history[key] = values
            
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
            Predicciones del modelo (acciones)
        """
        # Si el modelo tiene un método predict directo, úsalo
        if hasattr(self.model, 'predict'):
            return self.model.predict([x_cgm, x_other])
        
        # Si el modelo tiene un método act, úsalo (común en DRL)
        elif hasattr(self.model, 'act') or hasattr(self.model, 'get_action'):
            return self._predict_with_act(x_cgm, x_other)
        
        # Fallback para otros casos
        else:
            # Predicción uno por uno, ya que los modelos DRL suelen estar diseñados para estados individuales
            predictions = np.zeros((len(x_cgm),), dtype=np.float32)
            for i in range(len(x_cgm)):
                state = (x_cgm[i:i+1], x_other[i:i+1])
                if hasattr(self.model, '__call__'):
                    predictions[i] = self.model(state)
                else:
                    # Si no hay forma clara de predecir, devolver ceros
                    pass
            return predictions
    
    def _predict_with_act(self, x_cgm: np.ndarray, x_other: np.ndarray) -> np.ndarray:
        """
        Realiza predicciones usando el método act o get_action del modelo.
        
        Parámetros:
        -----------
        x_cgm : np.ndarray
            Datos CGM para predicción
        x_other : np.ndarray
            Otras características para predicción
            
        Retorna:
        --------
        np.ndarray
            Predicciones del modelo (acciones)
        """
        predictions = np.zeros((len(x_cgm),), dtype=np.float32)
        
        act_method = getattr(self.model, 'act', None) or getattr(self.model, 'get_action', None)
        
        for i in range(len(x_cgm)):
            state = (x_cgm[i:i+1], x_other[i:i+1])
            # Los modelos DRL suelen tener un parámetro deterministic para inferencia
            if 'deterministic' in act_method.__code__.co_varnames:
                predictions[i] = act_method(state, deterministic=True)
            else:
                predictions[i] = act_method(state)
                
        return predictions
    
    def save(self, path: str) -> None:
        """
        Guarda el estado del modelo DRL JAX en disco.

        Parámetros:
        -----------
        path : str
            Ruta donde guardar el modelo
    
        Retorna:
        --------
        None
        """
        # Verificar si el modelo está inicializado
        if self.model is None:
            raise ValueError(CONST_MODEL_INIT_ERROR.format("guardar"))
        
        # Crear directorio si es necesario
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)

        try:
            # Intentar guardar usando el método del modelo subyacente si existe
            if hasattr(self.model, 'save'):
                self.model.save(path)
                print_success(f"Modelo guardado usando método nativo en: {path}")
                return
                    
            # Fallback: guardar con pickle
            print_warning("Guardando con pickle como alternativa.")
            save_data = {
                'params': self.params,
                'state': self.state,
                'states': self.states,
                'model_kwargs': self.model_kwargs,
                'algorithm': self.algorithm,
                'rng_key': self.rng_key
            }
                
            # Asegurar que no haya duplicación de extensión
            save_path = path if path.endswith('.pkl') else f"{path}.pkl"
            
            with open(save_path, 'wb') as f:
                pickle.dump(save_data, f)
            print_success(f"Estado del modelo guardado como pickle en: {save_path}")
                
        except Exception as e:
            print_error(f"Error al guardar modelo: {e}")
            
            # Último recurso - intentar guardar solo información básica
            try:
                basic_info = {
                    'model_kwargs': self.model_kwargs,
                    'algorithm': self.algorithm
                }
                
                fallback_path = f"{path}_info.pkl"
                with open(fallback_path, 'wb') as f:
                    pickle.dump(basic_info, f)
                print_warning(f"No se pudo guardar el modelo completo, pero la información básica fue guardada en: {fallback_path}")
            except Exception as inner_e:
                print_error(f"No se pudo guardar ni siquiera la información básica: {inner_e}")
        
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
        # Verificar si es un archivo pickle
        if path.endswith('.pkl'):
            self._load_from_pickle(path)
        # Verificar si es un archivo de pesos
        elif path.endswith('_weights.h5') or path.endswith('.h5'):
            self._load_weights(path)
        # Intentar cargar como modelo completo
        elif os.path.isdir(path) or path.endswith('.keras'):
            self._load_complete_model(path)
        else:
            raise ValueError(f"Formato de archivo no soportado: {path}")
            
    def _load_from_pickle(self, path: str) -> None:
        """
        Carga el modelo desde un archivo pickle.
        
        Parámetros:
        -----------
        path : str
            Ruta del archivo pickle
        
        Retorna:
        --------
        None
        """
        try:
            with open(path, 'rb') as f:
                data = pickle.load(f)
            
            # Recrear modelo si es necesario
            if self.model is None:
                self._create_model_instance((1,), (1,))
            
            # Establecer pesos si están disponibles
            if 'weights' in data and data['weights'] is not None and hasattr(self.model, 'set_weights'):
                self.model.set_weights(data['weights'])
                print_success(f"Pesos cargados desde {path}")
            else:
                print_warning("No se encontraron pesos en el archivo pickle o el modelo no soporta set_weights")
        except Exception as e:
            print_error(f"Error al cargar desde pickle: {e}")
            
    def _load_weights(self, path: str) -> None:
        """
        Carga los pesos del modelo desde un archivo h5.
        
        Parámetros:
        -----------
        path : str
            Ruta del archivo de pesos
        
        Retorna:
        --------
        None
        """
        try:
            # Asegurar que el modelo existe
            if self.model is None:
                self._create_model_instance((1,), (1,))
                
            # Cargar pesos
            if hasattr(self.model, 'load_weights'):
                self.model.load_weights(path)
                print_success(f"Pesos cargados desde {path}")
            else:
                print_warning("El modelo no tiene método load_weights")
        except Exception as e:
            print_error(f"Error al cargar pesos: {e}")
            
    def _load_complete_model(self, path: str) -> None:
        """
        Carga el modelo completo.
        
        Parámetros:
        -----------
        path : str
            Ruta del modelo completo
        
        Retorna:
        --------
        None
        """
        try:
            self.model = tf.keras.models.load_model(path)
            print_success(f"Modelo cargado desde {path}")
        except Exception as e:
            print_error(f"Error al cargar el modelo completo: {e}")
            print_warning("Intentando cargar como pesos...")
            self._load_weights(path)
