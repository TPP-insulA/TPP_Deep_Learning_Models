import os
import numpy as np
import jax
import jax.numpy as jnp
from flax.training.checkpoints import save_checkpoint, restore_checkpoint
import pickle
import time
from tqdm.auto import tqdm
from typing import Dict, List, Tuple, Any, Optional, Callable

from config.models_config import EARLY_STOPPING_POLICY
from constants.constants import CONST_DEFAULT_SEED, CONST_DEFAULT_EPOCHS, CONST_DEFAULT_BATCH_SIZE
from custom.model_wrapper import ModelWrapper
from custom.printer import print_debug, print_info, print_log, print_success, print_error, print_warning

# Constantes para mensajes de error y campos comunes
CONST_MODEL_INIT_ERROR = "El modelo debe ser inicializado antes de {}"
CONST_DEVICE = "device"
CONST_LOSS = "loss"
CONST_VAL_LOSS = "val_loss"

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
        self.early_stopping_config = {
            'patience': EARLY_STOPPING_POLICY['early_stopping_patience'],
            'min_delta': EARLY_STOPPING_POLICY['early_stopping_min_delta'],
            'restore_best_weights': EARLY_STOPPING_POLICY['early_stopping_restore_best_weights'],
            'best_val_loss': EARLY_STOPPING_POLICY['early_stopping_best_val_loss'],
            'best_loss': EARLY_STOPPING_POLICY['early_stopping_best_loss'],
            'counter': EARLY_STOPPING_POLICY['early_stopping_counter'],
            'best_weights': EARLY_STOPPING_POLICY['early_stopping_best_weights'],
            'wait': 0
        }
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
        # Actualizar la configuración existente de early stopping
        self.early_stopping_config['patience'] = patience
        self.early_stopping_config['min_delta'] = min_delta
        self.early_stopping_config['restore_best_weights'] = restore_best_weights
        self.early_stopping_config['best_loss'] = EARLY_STOPPING_POLICY['early_stopping_best_val_loss'],
        self.early_stopping_config['counter'] = EARLY_STOPPING_POLICY['early_stopping_counter'],
        self.early_stopping_config['best_weights'] = EARLY_STOPPING_POLICY['early_stopping_best_weights']
        self.early_stopping_config['wait'] = 0
        

    def train(self, x_cgm: np.ndarray, x_other: np.ndarray, y: np.ndarray,
              validation_data: Optional[Tuple[Tuple[np.ndarray, np.ndarray], np.ndarray]] = None,
              epochs: int = CONST_DEFAULT_EPOCHS, batch_size: int = CONST_DEFAULT_BATCH_SIZE) -> Dict[str, List[float]]:
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
        """
        Ejecuta validación y actualiza historial.
        
        Parámetros:
        -----------
        validation_info : Dict[str, Any]
            Información de validación como diccionario con claves 'x_cgm', 'x_other', 'y'
            
        Retorna:
        --------
        float
            Pérdida de validación
        """
        x_cgm_val = validation_info.get('x_cgm')
        x_other_val = validation_info.get('x_other')
        y_val = validation_info.get('y')
        val_preds = self.predict(x_cgm_val, x_other_val)
        
        # Asegurar que val_loss sea un float
        val_loss = float(np.mean((val_preds - y_val) ** 2))
        self.history[CONST_VAL_LOSS].append(val_loss)
        
        # Actualizar early stopping con el valor float
        self._update_early_stopping(val_loss)
        
        return val_loss
        
    def _update_early_stopping(self, val_loss: float) -> None:
        """
        Actualiza el estado de early stopping basado en la pérdida de validación.
        
        Parámetros:
        -----------
        val_loss : float
            Pérdida de validación actual
            
        Retorna:
        --------
        None
        """
        # Asegurarse de acceder a la configuración correcta
        if not hasattr(self, 'early_stopping_config'):
            # Si no existe, inicializarla con valores predeterminados
            self.early_stopping_config = {
                'patience': EARLY_STOPPING_POLICY['early_stopping_patience'],
                'min_delta': EARLY_STOPPING_POLICY['early_stopping_min_delta'],
                'restore_best_weights': EARLY_STOPPING_POLICY['early_stopping_restore_best_weights'],
                'best_val_loss': float('inf'),
                'counter': 0,
                'best_weights': None,
                'best_agent_state': None
            }
        
        # Extraer el valor float correcto de val_loss si es una tupla
        if isinstance(val_loss, tuple):
            val_loss = val_loss[0]
        
        if val_loss < (self.early_stopping_config['best_val_loss'] - self.early_stopping_config['min_delta']):
            self.early_stopping_config['best_val_loss'] = val_loss
            self.early_stopping_config['counter'] = 0
            if self.early_stopping_config['restore_best_weights'] and self.agent_state is not None:
                # Guardar copia del mejor estado del agente
                self.early_stopping_config['best_agent_state'] = jax.tree_map(lambda x: x, self.agent_state)
        else:
            self.early_stopping_config['counter'] += 1
            
    def _should_stop_early(self) -> bool:
        """Determina si se debe detener el entrenamiento temprano."""
        if not self.early_stopping_config:
            return False
            
        if self.early_stopping_config['wait'] >= self.early_stopping_config['patience']:
            print("\nEarly stopping activado")
            if self.early_stopping_config['restore_best_weights'] and self.best_agent_state is not None:
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

        # Iterar sobre los lotes con barra de progreso
        batch_iterator = tqdm(
            range(steps_per_epoch), 
            desc="Procesando lotes", 
            leave=False,
            unit="batch"
        )
        
        for i in batch_iterator:
            batch_indices = indices[i * batch_size:(i + 1) * batch_size]
            batch_cgm = x_cgm[batch_indices]
            batch_other = x_other[batch_indices]
            batch_y = y[batch_indices]

            # Crear lote según lo que espere train_batch del agente
            batch_data = ((batch_cgm, batch_other), batch_y)

            # Usar train_batch_vectorized si existe, de lo contrario usar train_batch
            step_rng, rng_key = jax.random.split(rng_key)
            
            if hasattr(self.agent, 'train_batch_vectorized'):
                self.agent_state, step_metrics = self.agent.train_batch_vectorized(self.agent_state, batch_data, step_rng)
            else:
                self.agent_state, step_metrics = self.agent.train_batch(self.agent_state, batch_data, step_rng)

            # Actualizar pérdida total y métricas
            batch_loss = step_metrics.get('loss', 0.0)
            total_loss += batch_loss
            
            # Actualizar descripción de la barra con la pérdida actual
            batch_iterator.set_postfix(loss=f"{batch_loss:.4f}")
            
            # Acumular métricas
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
    
    def save(self, path: str) -> None:
        """
        Guarda el agente JAX en disco, extrayendo solo componentes serializables.
    
        Parámetros:
        -----------
        path : str
            Ruta donde guardar el modelo
    
        Retorna:
        --------
        None
        """
        if self.agent is None or self.agent_state is None:
            raise ValueError(CONST_MODEL_INIT_ERROR.format("guardar"))
        
        # Crear directorio si no existe
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        
        try:
            # Intentar guardar usando métodos directos
            if self._try_save_with_native_method(path):
                return
                
            # Extraer y guardar datos serializables
            self._save_serializable_data(path)
                
        except Exception as e:
            print_error(f"Error al guardar modelo: {e}")
            self._save_fallback_history(path)

    def _try_save_with_native_method(self, path: str) -> bool:
        """
        Intenta guardar el modelo usando métodos nativos.
        
        Parámetros:
        -----------
        path : str
            Ruta donde guardar el modelo
            
        Retorna:
        --------
        bool
            True si se guardó con éxito, False en caso contrario
        """
        # Primera Opción: Utilizar método save nativo del agente
        if hasattr(self.agent, 'save') and callable(self.agent.save):
            self.agent.save(path)
            print_success(f"Modelo guardado usando método nativo en: {path}")
            return True
            
        # Segunda Opción: Caso especial para PolicyIterationWrapper
        if hasattr(self.agent, 'pi_agent') and hasattr(self.agent, 'save'):
            self.agent.save(path)
            print_success(f"Modelo PolicyIteration guardado en: {path}")
            return True
            
        return False
        
    def _extract_serializable_data(self) -> dict:
        """
        Extrae los datos serializables del agente.
        
        Retorna:
        --------
        dict
            Diccionario con datos serializables
        """
        serializable_data = {}
        
        # Extraer parámetros entrenables si existen
        if hasattr(self.agent_state, 'params'):
            serializable_data['params'] = jax.tree_util.tree_map(
                lambda x: np.array(x), self.agent_state.params)
        elif hasattr(self.agent, 'params'):
            serializable_data['params'] = jax.tree_util.tree_map(
                lambda x: np.array(x), self.agent.params)
            
        # Extraer política y función de valor
        if hasattr(self.agent, 'policy'):
            serializable_data['policy'] = np.array(self.agent.policy)
        if hasattr(self.agent, 'v'):
            serializable_data['v'] = np.array(self.agent.v)
            
        # Extraer estados de actor-critic
        for attr in ['actor_params', 'critic_params', 'target_params', 'value_params', 'q_params']:
            if hasattr(self.agent, attr):
                serializable_data[attr] = jax.tree_util.tree_map(
                    lambda x: np.array(x), getattr(self.agent, attr))
                    
        return serializable_data
        
    def _save_serializable_data(self, path: str) -> None:
        """
        Extrae y guarda los datos serializables en un archivo pickle.
        
        Parámetros:
        -----------
        path : str
            Ruta donde guardar el modelo
    
        Retorna:
        --------
        None
        """
        # Extraer componentes serializables comunes
        serializable_data = self._extract_serializable_data()
        
        # Guardar estado y metadatos
        save_data = {
            'serializable_data': serializable_data,
            'cgm_shape': self.cgm_shape,
            'other_features_shape': self.other_features_shape,
            'model_kwargs': self.model_kwargs,
            'history': self.history,
            'algorithm': getattr(self, 'algorithm', 'unknown'),
            'early_stopping_config': self.early_stopping_config
        }
        
        # Verificar si el path ya tiene extensión .pkl
        save_path = path if path.endswith('.pkl') else f"{path}.pkl"
        
        with open(save_path, 'wb') as f:
            pickle.dump(save_data, f)
        print_success(f"Componentes serializables guardados en: {save_path}")
        
    def _save_fallback_history(self, path: str) -> None:
        """
        Guarda solo el historial como último recurso.

        Parámetros:
        -----------
        path : str
            Ruta base donde guardar el historial
    
        Retorna:
        --------
        None
        """
        try:
            base_path = path[:-4] if path.endswith('.pkl') else path
            history_path = f"{base_path}_history.pkl"
        
            with open(history_path, 'wb') as f:
                pickle.dump({'history': self.history}, f)
            print_warning(f"No se pudo guardar el modelo completo, pero el historial fue guardado en: {history_path}")
        except Exception as inner_e:
            print_error(f"No se pudo guardar ni siquiera el historial: {inner_e}")

    def load(self, path: str) -> None:
        """
        Carga el agente JAX desde disco.
        
        Parámetros:
        -----------
        path : str
            Ruta desde donde cargar el modelo
        
        Retorna:
        --------
        None
        """
        # Intentar cargar con checkpoint de Flax primero
        if self._try_load_from_flax_checkpoint(path):
            return
            
        # Si falla, intentar cargar desde pickle
        self._load_from_pickle(path)
            
    def _try_load_from_flax_checkpoint(self, path: str) -> bool:
        """
        Intenta cargar el agente desde un checkpoint de Flax.
        
        Parámetros:
        -----------
        path : str
            Ruta del checkpoint
            
        Retorna:
        --------
        bool
            True si la carga fue exitosa, False en caso contrario
        """
        try:
            self._ensure_agent_is_initialized()
                
            # Cargar estado del checkpoint
            restored_state = restore_checkpoint(path, self.agent_state)
            if restored_state is not None:
                self.agent_state = restored_state
                print_success(f"Agente restaurado desde checkpoint de Flax: {path}")
                return True
        except Exception as e:
            print_error(f"No se pudo cargar desde checkpoint de Flax: {e}")
            
        return False
        
    def _ensure_agent_is_initialized(self) -> None:
        """Asegura que el agente esté inicializado."""
        if self.agent is None:
            self.agent = self.agent_creator(
                cgm_shape=self.cgm_shape,
                other_features_shape=self.other_features_shape,
                **self.model_kwargs
            )
            
            # Crear estado inicial si es necesario
            if not hasattr(self, 'agent_state') or self.agent_state is None:
                if hasattr(self.agent, 'setup'):
                    # Clave aleatoria para inicialización
                    setup_rng = jax.random.PRNGKey(0)
                    self.agent_state = self.agent.setup(setup_rng)
                else:
                    self.agent_state = self.agent
                    
    def _load_from_pickle(self, path: str) -> None:
        """
        Carga el agente desde un archivo pickle.
        
        Parámetros:
        -----------
        path : str
            Ruta del archivo pickle
        """
        try:
            # Verificar si el archivo es .pkl o necesita extensión
            pickle_path = path if path.endswith('.pkl') else f"{path}.pkl"
            with open(pickle_path, 'rb') as f:
                data = pickle.load(f)
                
            # Restaurar datos
            self._restore_agent_data(data)
                
            print_success(f"Agente restaurado desde pickle: {pickle_path}")
        except Exception as e:
            raise ValueError(f"No se pudo cargar el agente desde {path}: {e}")
            
    def _restore_agent_data(self, data: Dict) -> None:
        """
        Restaura los datos del agente desde un diccionario.
        
        Parámetros:
        -----------
        data : Dict
            Diccionario con los datos del agente
        """
        # Restaurar estado y metadatos
        if 'agent_state' in data:
            self.agent_state = data['agent_state']
        if 'history' in data:
            self.history = data['history']
        if 'early_stopping' in data:
            self.early_stopping_config = data['early_stopping']
            
        # Recrear agente si no existe
        if self.agent is None and 'cgm_shape' in data and 'other_features_shape' in data:
            self.cgm_shape = data.get('cgm_shape', self.cgm_shape)
            self.other_features_shape = data.get('other_shape', self.other_features_shape)
            model_kwargs = data.get('model_kwargs', {})
            
            self.agent = self.agent_creator(
                cgm_shape=self.cgm_shape,
                other_features_shape=self.other_features_shape,
                **model_kwargs
            )
