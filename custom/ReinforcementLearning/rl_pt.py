import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
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
            Argumentos a pasar al método forward del modelo interno
                
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
                    seed_val = CONST_DEFAULT_SEED
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
             epochs: int = CONST_DEFAULT_EPOCHS, batch_size: int = CONST_DEFAULT_BATCH_SIZE) -> Dict[str, List[float]]:
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
    
    def save(self, path: str) -> None:
        """
        Guarda el modelo RL de PyTorch en disco.
        
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
        
        # Guardar modelo, optimizador y configuración
        save_dict = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict() if self.optimizer else None,
            'model_cls': self.model_cls.__name__,
            'model_kwargs': self.model_kwargs,
            'device': str(self.device),
            'rng_state': torch.get_rng_state()
        }
        
        # Guardar configuraciones específicas de RL
        if hasattr(self.model, 'get_save_dict'):
            # Algunos modelos RL tienen datos adicionales para guardar
            model_specific_data = self.model.get_save_dict()
            save_dict.update(model_specific_data)
        
        torch.save(save_dict, path)
        print_success(f"Modelo RL guardado en: {path}")

    def load(self, path: str) -> None:
        """
        Carga el modelo RL de PyTorch desde disco.
        
        Parámetros:
        -----------
        path : str
            Ruta desde donde cargar el modelo
        
        Retorna:
        --------
        None
        """
        # Cargar estado guardado
        checkpoint = torch.load(path, map_location=self.device)
        
        # Crear modelo si no existe
        if self.model is None or isinstance(self.model, self._create_dummy_model().__class__):
            # Si tenemos formas guardadas, usarlas para la inicialización
            cgm_shape = checkpoint.get('cgm_shape', (1,))
            other_shape = checkpoint.get('other_shape', (1,))
            self._create_model_instance(cgm_shape, other_shape)
        
        # Cargar estado del modelo
        if 'model_state_dict' in checkpoint:
            self.model.load_state_dict(checkpoint['model_state_dict'])
        
        # Cargar estado del optimizador si existe
        if 'optimizer_state_dict' in checkpoint and checkpoint['optimizer_state_dict'] and self.optimizer:
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        # Restaurar estado RNG si existe
        if 'rng_state' in checkpoint:
            torch.set_rng_state(checkpoint['rng_state'])
        
        # Cargar datos específicos de RL
        if hasattr(self.model, 'set_from_save_dict'):
            # Algunos modelos RL necesitan restaurar estados internos
            model_specific_data = {k: v for k, v in checkpoint.items() 
                                if k not in ['model_state_dict', 'optimizer_state_dict', 
                                            'model_cls', 'model_kwargs', 'device', 'rng_state']}
            self.model.set_from_save_dict(model_specific_data)
        
        # Mover modelo al dispositivo correcto
        self.model = self.model.to(self.device)
        print_success(f"Modelo RL cargado desde: {path}")
