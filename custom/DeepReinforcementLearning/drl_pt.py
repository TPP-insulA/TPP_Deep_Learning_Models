from typing import Dict, List, Tuple, Any, Optional, Callable, Union
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm.auto import tqdm

from constants.constants import CONST_DEFAULT_SEED, CONST_DEFAULT_EPOCHS, CONST_DEFAULT_BATCH_SIZE, CONST_ACTOR, CONST_CRITIC, CONST_TARGET, CONST_PARAMS, CONST_DEVICE, CONST_MODEL_INIT_ERROR, CONST_LOSS, CONST_VAL_LOSS, CONST_EPSILON
from config.models_config import EARLY_STOPPING_POLICY
from custom.model_wrapper import ModelWrapper
from custom.printer import print_debug, print_info, print_warning, print_error, print_success

class DRLModelWrapperPyTorch(ModelWrapper, nn.Module):
    """
    Wrapper para modelos de aprendizaje por refuerzo profundo implementados en PyTorch.
    
    Parámetros:
    -----------
    model_or_cls : Union[Callable, nn.Module]
        Clase del modelo DRL a instanciar o instancia ya creada del modelo
    **model_kwargs
        Argumentos para el constructor del modelo (usado solo si se pasa una clase)
    """
    
    def __init__(self, model_or_cls: Union[Callable, nn.Module], algorithm: str = "generic", **model_kwargs) -> None:
        """
        Inicializa un wrapper para modelos de aprendizaje por refuerzo profundo en PyTorch.
        
        Parámetros:
        -----------
        model_or_cls : Union[Callable, nn.Module]
            Clase del modelo DRL a instanciar o instancia ya creada del modelo
        **model_kwargs
            Argumentos para el constructor del modelo (usado solo si se pasa una clase)
        """
        ModelWrapper.__init__(self)
        nn.Module.__init__(self)
        super().__init__()
        
        # Determinar si es una clase o una instancia
        self.is_class = isinstance(model_or_cls, type) or callable(model_or_cls)
        
        if self.is_class:
            self.model_cls = model_or_cls
            self.model_kwargs = model_kwargs
            # Inicializar el modelo inmediatamente si es posible
            try:
                self.model = self.model_cls(**self.model_kwargs)
            except Exception:
                self.model = None
        else:
            self.model = model_or_cls
            self.model_kwargs = {}
            
        self.buffer = None
        self.algorithm = model_kwargs.get('algorithm', algorithm)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Inicializar el modelo si ya se pasó una instancia
        if not self.is_class and self.model is not None:
            if isinstance(self.model, nn.Module):
                self.model = self.model.to(self.device)
            
        # Crear un generador de numpy para operaciones aleatorias
        self.rng = np.random.Generator(np.random.PCG64(model_kwargs.get('seed', CONST_DEFAULT_SEED)))
        
        # Configuración para early stopping
        self.early_stopping_config = {
            'patience': EARLY_STOPPING_POLICY['early_stopping_patience'],
            'min_delta': EARLY_STOPPING_POLICY['early_stopping_min_delta'],
            'restore_best_weights': EARLY_STOPPING_POLICY['early_stopping_restore_best_weights'],
            'best_val_loss': EARLY_STOPPING_POLICY['early_stopping_best_val_loss'],
            'counter': EARLY_STOPPING_POLICY['early_stopping_counter'],
            'best_weights': EARLY_STOPPING_POLICY['early_stopping_best_weights']
        }
    
    def to(self, device: torch.device) -> 'DRLModelWrapperPyTorch':
        """
        Mueve el modelo al dispositivo especificado.
        
        Parámetros:
        -----------
        device : torch.device
            Dispositivo de destino (CPU/GPU)
            
        Retorna:
        --------
        DRLModelWrapperPyTorch
            Self para permitir encadenamiento
        """
        self.device = device
        if self.model is not None and isinstance(self.model, nn.Module):
                self.model = self.model.to(device)
        return self
    
    def forward(self, x_cgm: torch.Tensor, x_other: torch.Tensor) -> torch.Tensor:
        """
        Implementación requerida del método forward para nn.Module.
        
        Parámetros:
        -----------
        x_cgm : torch.Tensor
            Datos CGM de entrada
        x_other : torch.Tensor
            Otras características de entrada
            
        Retorna:
        --------
        torch.Tensor
            Predicciones del modelo
        """
        # Si el modelo no está inicializado, intentar inicializarlo ahora
        if self.model is None:
            if self.is_class:
                try:
                    self.model = self.model_cls(**self.model_kwargs)
                    if isinstance(self.model, nn.Module):
                        self.model = self.model.to(self.device)
                except Exception as e:
                    print_debug(f"Error al inicializar el modelo: {e}")
                    raise ValueError(CONST_MODEL_INIT_ERROR.format("realizar forward pass"))
            else:
                raise ValueError(CONST_MODEL_INIT_ERROR.format("realizar forward pass"))
            
        # Delegamos el forward al modelo subyacente
        try:
            if (hasattr(self.model, 'forward') and callable(self.model.forward)) or hasattr(self.model, '__call__'):
                return self.model(x_cgm, x_other)
        except Exception as e:
            print_debug(f"Error en forward: {e}")
            
        # Si no podemos usar forward directamente, intentar con predict
        with torch.no_grad():
            try:
                x_cgm_np = x_cgm.cpu().numpy()
                x_other_np = x_other.cpu().numpy()
                predictions_np = self.predict([x_cgm_np, x_other_np])
                return torch.FloatTensor(predictions_np).to(self.device)
            except Exception as e:
                print_debug(f"Error al predecir con el modelo: {e}")
                # Devolver un tensor de ceros como fallback
                return torch.zeros(x_cgm.shape[0], 1, device=self.device)
    
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
        # Si el modelo ya está inicializado y no es una clase, devolver directamente
        if self.model is not None and not self.is_class:
            return self.model

        # Si el modelo no está inicializado pero tenemos una clase, crearlo
        if self.model is None and self.is_class:
            self.model = self.model_cls(**self.model_kwargs)
            
        # Mover modelo al dispositivo adecuado si es un módulo PyTorch
        if isinstance(self.model, nn.Module):
            self.model = self.model.to(self.device)
        
        # Inicialización específica si el modelo lo requiere
        if hasattr(self.model, 'start'):
            return self.model.start(x_cgm, x_other, y, rng_key)
        elif hasattr(self.model, 'initialize'):
            state_dim = (x_cgm.shape[1:], x_other.shape[1:])
            action_dim = 1  # Para regresión en el caso de dosis de insulina
            return self.model.initialize(state_dim, action_dim)
            
        return self.model

    def parameters(self, recurse=True):
        """
        Retorna los parámetros entrenables del modelo.
        Necesario para que optimizadores de PyTorch funcionen correctamente.
        
        Parámetros:
        -----------
        recurse : bool, opcional
            Si True, devuelve parámetros de esta instancia y todos los submódulos
            recursivamente (default: True)
            
        Retorna:
        --------
        Iterator
            Iterador sobre los parámetros del modelo
        """
        # Si el modelo existe y es un módulo PyTorch, delegar a sus parámetros
        if self.model is not None and isinstance(self.model, nn.Module):
            return self.model.parameters(recurse=recurse)
        # De lo contrario, devolver los parámetros de este wrapper (que podría estar vacío)
        return super().parameters(recurse=recurse)
        
    def train(self, mode: bool = True) -> 'DRLModelWrapperPyTorch':
        """
        Establece el módulo en modo entrenamiento.
        
        Parámetros:
        -----------
        mode : bool, opcional
            Si True, establece en modo entrenamiento, si False en modo evaluación (default: True)
            
        Retorna:
        --------
        DRLModelWrapperPyTorch
            Self para permitir encadenamiento
        """
        nn.Module.train(self, mode)
        if self.model is not None and isinstance(self.model, nn.Module):
            self.model.train(mode)
        return self
    
    def eval(self) -> 'DRLModelWrapperPyTorch':
        """
        Establece el módulo en modo evaluación.
        
        Retorna:
        --------
        DRLModelWrapperPyTorch
            Self para permitir encadenamiento
        """
        # Call nn.Module's eval method directly
        nn.Module.eval(self)
        if self.model is not None and isinstance(self.model, nn.Module):
            self.model.eval()
        return self
    
    def _initialize_optimizer(self, learning_rate: float = 1e-3) -> None:
        """
        Inicializa el optimizador si aún no existe.

        Parámetros:
        -----------
        learning_rate : float, opcional
            Tasa de aprendizaje para el optimizador (default: 1e-3)

        Retorna:
        --------
        None
        """
        if not hasattr(self, 'optimizer') or self.optimizer is None:
            if self.model is not None and isinstance(self.model, nn.Module):
                # Usar Adam por defecto si no se especifica otro
                self.optimizer = optim.Adam(self.parameters(), lr=learning_rate, weight_decay=0.01)
            else:
                print_warning("No se pudo inicializar el optimizador: el modelo no es un nn.Module o no está inicializado.")

    def _compute_rewards(self, x_cgm: np.ndarray, x_other: np.ndarray, y: np.ndarray) -> np.ndarray:
        """
        Calcula recompensas a partir de los datos y objetivos. (Adaptado para PyTorch)

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
            with torch.no_grad():
                x_cgm_t = torch.FloatTensor(x_cgm).to(self.device)
                x_other_t = torch.FloatTensor(x_other).to(self.device)
                y_t = torch.FloatTensor(y).to(self.device)
                rewards_t = self.model.compute_rewards(x_cgm_t, x_other_t, y_t)
                return rewards_t.cpu().numpy()

        # Implementación por defecto: recompensa negativa basada en el error
        predicted = self.predict(x_cgm, x_other) # Usa el método predict del wrapper
        error = np.abs(predicted.flatten() - y.flatten())
        # Normalizar error al rango [-1, 0] donde -1 es el peor error y 0 es perfecto
        max_error = np.max(error) if np.max(error) > 0 else 1.0
        rewards = -error / max_error
        return rewards

    def _fill_buffer(self, x_cgm: np.ndarray, x_other: np.ndarray, rewards: np.ndarray, y: np.ndarray) -> None:
        """
        Llena el buffer de experiencia con transiciones. (Adaptado para PyTorch)

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
        if not hasattr(self.model, 'add_to_buffer') or not hasattr(self.model, 'buffer'):
            return

        buffer = getattr(self.model, 'buffer', None)
        if buffer is None:
            return

        for i in range(len(rewards)):
            state = (x_cgm[i], x_other[i])
            action = y[i]
            reward = rewards[i]
            # En un entorno supervisado, el siguiente estado puede ser el mismo
            # y done es siempre True (episodio de un paso)
            next_state = (x_cgm[i], x_other[i])
            done = True
            # El método add_to_buffer del modelo debe manejar la conversión a tensores si es necesario
            self.model.add_to_buffer(buffer, state, action, reward, next_state, done)

    def _update_networks(self, batch_size: int) -> Dict[str, float]:
        """
        Actualiza las redes del modelo usando experiencias del buffer. (Adaptado para PyTorch)

        Parámetros:
        -----------
        batch_size : int
            Tamaño del lote para actualizaciones

        Retorna:
        --------
        Dict[str, float]
            Diccionario con métricas de actualización (pérdidas)
        """
        # Inicializar métricas
        metrics = self._init_update_metrics()
        
        # Verificar si el modelo tiene los métodos necesarios
        if not self._check_model_readiness_for_update():
            return metrics
            
        # Obtener batch del buffer
        batch = self._get_batch_from_buffer(batch_size)
        if batch is None:
            return metrics
            
        # Ejecutar actualización y procesar resultados
        return self._process_update_results(batch, metrics)
        
    def _init_update_metrics(self) -> Dict[str, float]:
        """Inicializa las métricas de actualización con valores por defecto."""
        return {
            "actor_loss": 0.0,
            "critic_loss": 0.0,
            "q_loss": 0.0,
            "total_loss": 0.0
        }
        
    def _check_model_readiness_for_update(self) -> bool:
        """Verifica si el modelo está listo para actualizaciones."""
        if not hasattr(self.model, 'update') or not hasattr(self.model, 'sample_buffer'):
            print_warning("El modelo no implementa 'update' o 'sample_buffer'. No se realizarán actualizaciones.")
            return False
        return True
        
    def _get_batch_from_buffer(self, batch_size: int):
        """Obtiene un batch de experiencias del buffer."""
        buffer = getattr(self.model, 'buffer', None)
        if buffer is None:
            print_warning("El modelo no tiene un atributo 'buffer'. No se puede muestrear.")
            return None
        return self.model.sample_buffer(buffer, batch_size)
        
    def _process_update_results(self, batch, metrics: Dict[str, float]) -> Dict[str, float]:
        """Procesa los resultados de la actualización del modelo."""
        # Preparar para actualización
        self.optimizer.zero_grad()
        update_metrics = self.model.update(batch)
        
        # Manejar diferentes tipos de resultado
        if isinstance(update_metrics, dict) and 'total_loss' in update_metrics:
            self._handle_dict_with_total_loss(update_metrics, metrics)
        elif isinstance(update_metrics, torch.Tensor) and update_metrics.requires_grad:
            self._handle_tensor_loss(update_metrics, metrics)
        elif isinstance(update_metrics, (int, float)):
            metrics["total_loss"] = float(update_metrics)
        elif isinstance(update_metrics, dict):
            self._handle_dict_without_total_loss(update_metrics, metrics)
            
        return metrics
        
    def _handle_dict_with_total_loss(self, update_metrics: Dict, metrics: Dict[str, float]) -> None:
        """Maneja el caso donde update devuelve un diccionario con total_loss."""
        loss = update_metrics['total_loss']
        if isinstance(loss, torch.Tensor) and loss.requires_grad:
            loss.backward()
            self.optimizer.step()
            self._convert_tensor_values_to_float(update_metrics, metrics)
        else:
            self._convert_all_values_to_float(update_metrics, metrics)
            
    def _handle_tensor_loss(self, loss: torch.Tensor, metrics: Dict[str, float]) -> None:
        """Maneja el caso donde update devuelve un tensor como pérdida total."""
        loss.backward()
        self.optimizer.step()
        metrics["total_loss"] = loss.item()
        
    def _handle_dict_without_total_loss(self, update_metrics: Dict, metrics: Dict[str, float]) -> None:
        """Maneja el caso donde update devuelve un diccionario sin total_loss."""
        self._convert_tensor_values_to_float(update_metrics, metrics)
        
    def _convert_tensor_values_to_float(self, source: Dict, target: Dict[str, float]) -> None:
        """Convierte valores tensor a float en las métricas."""
        for key, value in source.items():
            if key in target:
                if isinstance(value, torch.Tensor):
                    target[key] = value.item()
                else:
                    target[key] = float(value)
                    
    def _convert_all_values_to_float(self, source: Dict, target: Dict[str, float]) -> None:
        """Convierte todos los valores a float en las métricas."""
        for key, value in source.items():
            if key in target:
                target[key] = float(value)

    def _run_epoch_updates(self, batch_size: int, updates_per_epoch: int) -> Dict[str, float]:
        """
        Ejecuta las actualizaciones durante una época. (Adaptado para PyTorch)

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
        updates_done = 0
        for _ in range(updates_per_epoch):
            try:
                batch_metrics = self._update_networks(batch_size)
                for key, value in batch_metrics.items():
                    if key in epoch_metrics:
                        epoch_metrics[key] += value
                updates_done += 1
            except Exception as e:
                print_warning(f"Error durante la actualización de redes: {e}. Saltando esta actualización.")
                # Podría ser útil agregar un traceback aquí para depuración
                # import traceback
                # traceback.print_exc()
                continue # Saltar a la siguiente actualización

        # Promediar métricas si se realizaron actualizaciones
        if updates_done > 0:
            for key in epoch_metrics:
                epoch_metrics[key] /= updates_done
        return epoch_metrics

    def _validate_model(self, x_cgm_val: np.ndarray, x_other_val: np.ndarray,
                       y_val: np.ndarray) -> float:
        """
        Realiza validación del modelo y calcula la pérdida (MSE por defecto).

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
            Pérdida de validación calculada
        """
        self.eval() # Poner el modelo en modo evaluación
        with torch.no_grad():
            val_preds = self.predict(x_cgm_val, x_other_val) # Usar el método predict del wrapper
            # Calcular MSE
            val_loss = float(np.mean((val_preds.flatten() - y_val.flatten()) ** 2))
        self.train() # Volver a poner el modelo en modo entrenamiento
        return val_loss

    def _process_epoch_metrics(self, epoch_metrics: Dict[str, float],
                               history: Dict[str, List[float]]) -> float:
        """
        Procesa y registra las métricas de una época en el historial.

        Parámetros:
        -----------
        epoch_metrics : Dict[str, float]
            Métricas promediadas de la época
        history : Dict[str, List[float]]
            Historial de entrenamiento a actualizar

        Retorna:
        --------
        float
            Pérdida total calculada para la época
        """
        # Registrar métricas en el historial
        for key, value in epoch_metrics.items():
            if key in history:
                history[key].append(value)

        # Pérdida total para seguimiento y early stopping
        total_loss = epoch_metrics.get("total_loss", 0.0)

        if abs(total_loss) < CONST_EPSILON:
            # Si no hay pérdida total (o es muy cercana a cero), usar la suma de otras pérdidas si existen
            other_losses = [epoch_metrics.get(key, 0.0) for key in ["actor_loss", "critic_loss", "q_loss"]]
            if any(abs(l) > CONST_EPSILON for l in other_losses):
                 total_loss = sum(other_losses)
            # Si todas las pérdidas son cero, total_loss permanece cero

        history[CONST_LOSS].append(total_loss) # Asegurarse de que 'loss' siempre se registre
        return total_loss

    def _initialize_training(self, x_cgm: np.ndarray, x_other: np.ndarray, y: np.ndarray) -> None:
        """
        Inicializa el modelo y el optimizador para entrenamiento.
        
        Parámetros:
        -----------
        x_cgm : np.ndarray
            Datos CGM de entrada
        x_other : np.ndarray
            Otras características de entrada
        y : np.ndarray
            Valores objetivo
        """
        if self.model is None:
            self.start(x_cgm, x_other, y)
        self._initialize_optimizer()
        
    def _setup_history(self) -> Dict[str, List[float]]:
        """
        Prepara el diccionario para almacenar el historial de entrenamiento.
        
        Retorna:
        --------
        Dict[str, List[float]]
            Diccionario de historial inicializado
        """
        return {
            CONST_LOSS: [],
            "actor_loss": [],
            "critic_loss": [],
            "q_loss": [],
            CONST_VAL_LOSS: []
        }
        
    def _setup_early_stopping(self) -> None:
        """
        Inicializa la configuración para early stopping.
        """
        es_config = self.early_stopping_config
        es_config['best_val_loss'] = float('inf')
        es_config['counter'] = 0
        es_config['best_weights'] = None
    
    def _run_training_epoch(self, epoch: int, x_cgm: np.ndarray, x_other: np.ndarray, 
                           y: np.ndarray, batch_size: int, do_validation: bool,
                           x_cgm_val: Optional[np.ndarray], x_other_val: Optional[np.ndarray],
                           y_val: Optional[np.ndarray], history: Dict[str, List[float]]) -> Tuple[float, Optional[float]]:
        """
        Ejecuta una época completa de entrenamiento.
        
        Retorna:
        --------
        Tuple[float, Optional[float]]
            Tupla con (pérdida de entrenamiento, pérdida de validación)
        """
        self.train()  # Asegurar modo entrenamiento
        
        # Calcular recompensas y llenar buffer
        rewards = self._compute_rewards(x_cgm, x_other, y)
        self._fill_buffer(x_cgm, x_other, rewards, y)
        
        # Ejecutar actualizaciones de redes
        updates_per_epoch = max(1, len(x_cgm) // batch_size)
        epoch_metrics = self._run_epoch_updates(batch_size, updates_per_epoch)
        
        # Procesar métricas
        epoch_loss = self._process_epoch_metrics(epoch_metrics, history)
        
        # Validación
        val_loss = None
        if do_validation:
            val_loss = self._validate_model(x_cgm_val, x_other_val, y_val)
            history[CONST_VAL_LOSS].append(val_loss)
            
        return epoch_loss, val_loss
    
    def _log_epoch_progress(self, epoch: int, epochs: int, epoch_loss: float, 
                           val_loss: Optional[float], history: Dict[str, List[float]]) -> None:
        """
        Registra el progreso de la época actual.
        
        Parámetros:
        -----------
        epoch : int
            Época actual
        epochs : int
            Total de épocas
        epoch_loss : float
            Pérdida de entrenamiento
        val_loss : Optional[float]
            Pérdida de validación
        history : Dict[str, List[float]]
            Historial de entrenamiento
        """
        log_msg = f"Época {epoch + 1}/{epochs} - Pérdida: {epoch_loss:.4f}"
        if val_loss is not None:
            log_msg += f" - Pérdida Val: {val_loss:.4f}"
        
        # Agregar otras métricas si existen
        for metric in ["actor_loss", "critic_loss", "q_loss"]:
            if history[metric] and abs(history[metric][-1]) > CONST_EPSILON:
                log_msg += f" - {metric}: {history[metric][-1]:.4f}"
        
        print_info(log_msg)
    
    def _check_early_stopping(self, epoch_loss: float, val_loss: Optional[float]) -> bool:
        """
        Comprueba si se debe detener el entrenamiento según la política de early stopping.
        
        Parámetros:
        -----------
        epoch_loss : float
            Pérdida de entrenamiento
        val_loss : Optional[float]
            Pérdida de validación
            
        Retorna:
        --------
        bool
            True si se debe detener el entrenamiento, False en caso contrario
        """
        es_config = self.early_stopping_config
        current_val_loss = val_loss if val_loss is not None else epoch_loss
        
        if current_val_loss < es_config['best_val_loss'] - es_config['min_delta']:
            es_config['best_val_loss'] = current_val_loss
            es_config['counter'] = 0
            if es_config['restore_best_weights']:
                es_config['best_weights'] = self.model.state_dict()
            return False
        
        es_config['counter'] += 1
        return es_config['counter'] >= es_config['patience']
    
    def _restore_best_weights(self) -> None:
        """
        Restaura los mejores pesos encontrados durante el entrenamiento.
        """
        es_config = self.early_stopping_config
        if es_config['restore_best_weights'] and es_config['best_weights'] is not None:
            print_info(f"Restaurando mejores pesos con pérdida de validación: {es_config['best_val_loss']:.4f}")
            self.model.load_state_dict(es_config['best_weights'])
    
    def fit(self, x: List[np.ndarray], y: np.ndarray,
           validation_data: Optional[Tuple] = None,
           epochs: int = CONST_DEFAULT_EPOCHS, batch_size: int = CONST_DEFAULT_BATCH_SIZE, verbose: int = 1) -> Dict[str, List[float]]:
        """
        Entrena el modelo DRL con los datos proporcionados usando PyTorch.

        Parámetros:
        -----------
        x : List[np.ndarray]
            Lista con [x_cgm, x_other]
        y : np.ndarray
            Valores objetivo (acciones)
        validation_data : Optional[Tuple], opcional
            Datos de validación como ([x_cgm_val, x_other_val], y_val) (default: None)
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
        x_cgm, x_other = x
        
        # Inicialización del entrenamiento
        self._initialize_training(x_cgm, x_other, y)
        
        # Preparar datos y estructuras
        x_cgm_val, x_other_val, y_val = self._unpack_validation_data(validation_data)
        do_validation = x_cgm_val is not None
        history = self._setup_history()
        self._setup_early_stopping()
        
        # Anunciar inicio del entrenamiento
        if verbose > 0:
            print_info(f"Iniciando entrenamiento DRL ({self.algorithm}) por {epochs} épocas (PyTorch)...")
        
        # Bucle de entrenamiento por épocas
        for epoch in tqdm(range(epochs), desc="Entrenando (PyTorch)", disable=verbose == 0):
            # Ejecutar una época completa
            epoch_loss, val_loss = self._run_training_epoch(
                epoch, x_cgm, x_other, y, batch_size, do_validation,
                x_cgm_val, x_other_val, y_val, history
            )
            
            # Registrar progreso
            if verbose > 0:
                self._log_epoch_progress(epoch, epochs, epoch_loss, val_loss, history)
                
            # Comprobar early stopping
            if self._check_early_stopping(epoch_loss, val_loss):
                print_info(f"Early stopping en época {epoch + 1}")
                break
                
        # Restaurar mejores pesos al finalizar
        self._restore_best_weights()
        
        return history
    
    def predict_with_context(self, x_cgm: np.ndarray, x_other: np.ndarray, **context) -> float:
        """
        Realiza predicciones de dosis con variables contextuales adicionales.
        
        Parámetros:
        -----------
        x_cgm : np.ndarray
            Datos CGM para predicción
        x_other : np.ndarray
            Otras características para predicción
        **context : dict
            Variables contextuales como:
            - carb_intake : float - Ingesta de carbohidratos
            - iob : float - Insulina a bordo
            - objective_glucose : float - Nivel objetivo de glucosa
            - sleep_quality : float - Calidad del sueño
            - work_intensity : float - Intensidad del trabajo
            - exercise_intensity : float - Intensidad del ejercicio
            
        Retorna:
        --------
        float
            Dosis de insulina recomendada en unidades
        """
        # Verificar que el modelo esté inicializado
        if self.model is None:
            if self.is_class:
                try:
                    self.model = self.model_cls(**self.model_kwargs)
                    if isinstance(self.model, nn.Module):
                        self.model = self.model.to(self.device)
                except Exception as e:
                    print_debug(f"Error al inicializar el modelo: {e}")
                    raise ValueError(CONST_MODEL_INIT_ERROR.format("predecir"))
            else:
                raise ValueError(CONST_MODEL_INIT_ERROR.format("predecir"))
        
        # Poner en modo evaluación
        self.eval()
        
        # Verificar si el modelo tiene un método específico para contexto
        if hasattr(self.model, 'predict_with_context'):
            prediction_array = self.model.predict_with_context(x_cgm, x_other, **context)
            return float(prediction_array.flatten()[0]) if prediction_array.size > 0 else 0.0
        
        # Si el modelo tiene predict pero no específico para contexto
        if hasattr(self.model, 'predict'):
            # Intentar pasar el contexto como kwargs
            try:
                prediction_array = self.model.predict([x_cgm, x_other], **context)
                return float(prediction_array.flatten()[0]) if prediction_array.size > 0 else 0.0
            except Exception:
                # Si falla, usar predict normal
                prediction_array = self.model.predict([x_cgm, x_other])
                return float(prediction_array.flatten()[0]) if prediction_array.size > 0 else 0.0
        
        # Implementación estándar si no hay métodos específicos
        with torch.no_grad():
            x_cgm_t = torch.FloatTensor(x_cgm).to(self.device)
            x_other_t = torch.FloatTensor(x_other).to(self.device)
            
            # Crear tensor de contexto si hay variables contextuales
            if context:
                # Extraer valores de contexto (usar 0 como valor por defecto)
                context_values = [
                    context.get('carb_intake', 0.0),
                    context.get('iob', 0.0),
                    context.get('objective_glucose', 0.0),
                    context.get('sleep_quality', 5.0),
                    context.get('work_intensity', 0.0),
                    context.get('exercise_intensity', 0.0)
                ]
                
                # Crear tensor - asegurar que solo procesamos una muestra a la vez
                context_tensor = torch.FloatTensor(context_values).to(self.device).unsqueeze(0)
                
                # Si el modelo tiene un método forward_with_context, usarlo
                if hasattr(self.model, 'forward_with_context'):
                    predictions = self.model.forward_with_context(x_cgm_t, x_other_t, context_tensor)
                else:
                    # Concatenar con otras características como alternativa
                    enhanced_features = torch.cat([x_other_t, context_tensor], dim=1)
                    predictions = self.forward(x_cgm_t, enhanced_features)
            else:
                # Sin contexto, usar forward normal
                predictions = self.forward(x_cgm_t, x_other_t)
            
            # Extraer y devolver el valor único
            return float(predictions.cpu().numpy().flatten()[0])

    def save(self, path: str) -> None:
        """
        Guarda el modelo DRL de PyTorch en disco.

        Parámetros:
        -----------
        path : str
            Ruta donde guardar el modelo.

        Retorna:
        --------
        None
        """
        if self.model is None:
            raise ValueError("El modelo debe estar inicializado antes de guardarlo.")

        save_data = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict() if self.optimizer else None,
            'model_kwargs': self.model_kwargs,
            'rng_state': torch.get_rng_state()
        }
        torch.save(save_data, path)
        print_success(f"Modelo guardado en: {path}")

    def load(self, path: str) -> None:
        """
        Carga el modelo DRL de PyTorch desde disco.

        Parámetros:
        -----------
        path : str
            Ruta desde donde cargar el modelo.

        Retorna:
        --------
        None
        """
        checkpoint = torch.load(path, map_location=self.device)
        if self.model is None:
            self.model = self.model_cls(**self.model_kwargs)
            self.model.to(self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        if 'optimizer_state_dict' in checkpoint and self.optimizer:
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if 'rng_state' in checkpoint:
            torch.set_rng_state(checkpoint['rng_state'])
        print_success(f"Modelo cargado desde: {path}")
