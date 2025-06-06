import os
import numpy as np
from custom.DeepLearning.dl_pt import DLModelWrapperPyTorch
from custom.printer import print_header, print_info, print_debug, print_warning
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, TensorDataset
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold
from joblib import Parallel, delayed
from scipy.optimize import minimize
from typing import Dict, List, Tuple, Callable, Optional, Any, Union
from config.params import DEBUG
from training.common import (
    calculate_metrics, evaluate_clinical_metrics, optimize_ensemble_weights_clinical, get_model_type, enhance_features
)
from constants.constants import (
    CONST_VAL_LOSS, CONST_LOSS, CONST_METRIC_MAE, CONST_METRIC_RMSE, CONST_METRIC_R2,
    CONST_MODELS, CONST_BEST_PREFIX, CONST_LOGS_DIR, CONST_DEFAULT_EPOCHS, 
    CONST_DEFAULT_BATCH_SIZE, CONST_DEFAULT_SEED, CONST_FIGURES_DIR, CONST_MODEL_TYPES, CONST_DURATION_HOURS
)
from tqdm.auto import tqdm

# Usar menos épocas en modo debug
CONST_EPOCHS = 2 if DEBUG else CONST_DEFAULT_EPOCHS

# Configurar dispositivo GPU si está disponible
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class CGMDataset(Dataset):
    """
    Dataset personalizado para datos CGM y otras características.
    
    Parámetros:
    -----------
    x_cgm : np.ndarray
        Datos CGM con forma (muestras, pasos_tiempo, características)
    x_other : np.ndarray
        Otras características con forma (muestras, características)
    y : np.ndarray
        Valores objetivo con forma (muestras,)
    """
    
    def __init__(self, x_cgm: np.ndarray, x_other: np.ndarray, y: np.ndarray) -> None:
        self.x_cgm = torch.FloatTensor(x_cgm)
        self.x_other = torch.FloatTensor(x_other)
        self.y = torch.FloatTensor(y).reshape(-1, 1)
        
    def __len__(self) -> int:
        """
        Obtiene la longitud del dataset.
        
        Retorna:
        --------
        int
            Número de muestras en el dataset
        """
        return len(self.y)
        
    def __getitem__(self, idx: int) -> Tuple[Tuple[torch.Tensor, torch.Tensor], torch.Tensor]:
        """
        Obtiene un ítem específico del dataset.
        
        Parámetros:
        -----------
        idx : int
            Índice del ítem a obtener
            
        Retorna:
        --------
        Tuple[Tuple[torch.Tensor, torch.Tensor], torch.Tensor]
            Tupla con ((x_cgm, x_other), y) para el ítem solicitado
        """
        return ((self.x_cgm[idx], self.x_other[idx]), self.y[idx])


def create_dataloaders(x_cgm: np.ndarray, 
                     x_other: np.ndarray, 
                     y: np.ndarray, 
                     batch_size: int = CONST_DEFAULT_BATCH_SIZE,
                     shuffle: bool = True) -> DataLoader:
    """
    Crea DataLoaders para el entrenamiento PyTorch.
    
    Parámetros:
    -----------
    x_cgm : np.ndarray
        Datos CGM con forma (muestras, pasos_tiempo, características)
    x_other : np.ndarray
        Otras características con forma (muestras, características)
    y : np.ndarray
        Valores objetivo con forma (muestras,)
    batch_size : int, opcional
        Tamaño del batch para entrenamiento (default: 32)
    shuffle : bool, opcional
        Si se deben mezclar los datos (default: True)
        
    Retorna:
    --------
    DataLoader
        DataLoader de PyTorch para el entrenamiento
    """
    dataset = CGMDataset(x_cgm, x_other, y)
    return DataLoader(
        dataset, 
        batch_size=batch_size, 
        shuffle=shuffle,
        num_workers=0,
        pin_memory=torch.cuda.is_available()  # Mejora el rendimiento con GPU
    )


class EarlyStopping:
    """
    Early stopping para prevenir el sobreajuste.
    
    Parámetros:
    -----------
    patience : int, opcional
        Número de épocas a esperar antes de detener (default: 10)
    min_delta : float, opcional
        Cambio mínimo para considerar como mejora (default: 0)
    restore_best_weights : bool, opcional
        Si restaurar los mejores pesos del modelo (default: True)
    """
    def __init__(self, patience: int = 10, min_delta: float = 0, restore_best_weights: bool = True):
        self.patience = patience
        self.min_delta = min_delta
        self.restore_best_weights = restore_best_weights
        self.best_val_loss = float('inf')
        self.best_weights = None
        self.counter = 0
        self.early_stop = False

    def __call__(self, model: nn.Module, val_loss: float) -> bool:
        """
        Verifica si debe activarse el early stopping.
        
        Parámetros:
        -----------
        model : nn.Module
            Modelo PyTorch para guardar sus pesos
        val_loss : float
            Pérdida de validación actual
            
        Retorna:
        --------
        bool
            True si debe detenerse el entrenamiento, False en caso contrario
        """
        if val_loss < self.best_val_loss - self.min_delta:
            self.best_val_loss = val_loss
            if self.restore_best_weights:
                self.best_weights = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
                if self.restore_best_weights and self.best_weights is not None:
                    model.load_state_dict(self.best_weights)
                return True
        return False


def _prepare_model(model: Union[nn.Module, DLModelWrapperPyTorch],
                 x_cgm_train: np.ndarray,
                 x_other_train: np.ndarray,
                 y_train: np.ndarray) -> nn.Module:
    """
    Prepara el modelo para entrenamiento, manejando wrappers si es necesario.
    
    Parámetros:
    -----------
    model : Union[nn.Module, DLModelWrapperPyTorch]
        Modelo a preparar (puede ser un wrapper)
    x_cgm_train : np.ndarray
        Datos CGM de entrenamiento
    x_other_train : np.ndarray
        Otras características de entrenamiento
    y_train : np.ndarray
        Valores objetivo de entrenamiento
        
    Retorna:
    --------
    nn.Module
        Modelo preparado para entrenamiento
    """
    if hasattr(model, 'model') and isinstance(model, DLModelWrapperPyTorch):
        # Si model es un wrapper, inicializar y usar el modelo interno
        model.start(x_cgm_train, x_other_train, y_train)
        actual_model = model.model
    else:
        # Si es un modelo PyTorch normal
        actual_model = model
    
    # Transferir modelo al dispositivo
    return actual_model.to(DEVICE)


def _run_train_epoch(model: nn.Module,
                    train_loader: DataLoader,
                    optimizer: optim.Optimizer,
                    criterion: nn.Module) -> float:
    """
    Ejecuta una época de entrenamiento y devuelve la pérdida promedio.
    
    Parámetros:
    -----------
    model : nn.Module
        Modelo a entrenar
    train_loader : DataLoader
        DataLoader para el conjunto de entrenamiento
    optimizer : optim.Optimizer
        Optimizador para el modelo
    criterion : nn.Module
        Función de pérdida para el entrenamiento
        
    Retorna:
    --------
    float
        Pérdida promedio de la época
    """
    model.train()
    train_loss = 0.0
    
    # Añadir barra de progreso para los batches
    progress_bar = tqdm(enumerate(train_loader), total=len(train_loader), 
                       desc="Batches", leave=False)
    
    for batch_idx, ((x_cgm_batch, x_other_batch), y_batch) in progress_bar:
        # Transferir datos al dispositivo
        x_cgm_batch = x_cgm_batch.to(DEVICE)
        x_other_batch = x_other_batch.to(DEVICE)
        y_batch = y_batch.to(DEVICE)
        
        # Poner gradientes a cero
        optimizer.zero_grad()
        
        # Forward pass
        outputs = model(x_cgm_batch, x_other_batch)
        
        # Asegurar que outputs tiene forma [batch_size, 1]
        if outputs.dim() == 1:
            outputs = outputs.unsqueeze(1)
            
        loss = criterion(outputs, y_batch)
        
        # Backward pass y optimización
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        
        # Acumular pérdida
        train_loss += loss.item()
        
        # Actualizar barra de progreso con la pérdida actual
        if batch_idx % 5 == 0:  # Actualizar cada 5 batches para no sobrecargar
            progress_bar.set_postfix({"batch_loss": f"{loss.item():.4f}"})
    
    return train_loss / len(train_loader)


def _run_validation(model: nn.Module,
                  val_loader: DataLoader,
                  criterion: nn.Module) -> Tuple[float, np.ndarray, np.ndarray]:
    """
    Ejecuta validación y devuelve pérdida, predicciones y valores reales.
    
    Parámetros:
    -----------
    model : nn.Module
        Modelo a validar
    val_loader : DataLoader
        DataLoader para el conjunto de validación
    criterion : nn.Module
        Función de pérdida para la validación
        
    Retorna:
    --------
    Tuple[float, np.ndarray, np.ndarray]
        (pérdida promedio, predicciones, valores reales)
    """
    model.eval()
    val_loss = 0.0
    val_preds = []
    val_targets = []

    with torch.no_grad():
        for (x_cgm_batch, x_other_batch), y_batch in val_loader:
            # Transferir datos al dispositivo
            x_cgm_batch = x_cgm_batch.to(DEVICE)
            x_other_batch = x_other_batch.to(DEVICE)
            y_batch = y_batch.to(DEVICE)
            
            # Forward pass
            outputs = model(x_cgm_batch, x_other_batch)
            
            # Asegurar que outputs tiene forma [batch_size, 1]
            if outputs.dim() == 1:
                outputs = outputs.unsqueeze(1)  # Convertir [batch_size] a [batch_size, 1]
            
            loss = criterion(outputs, y_batch)
            
            # Acumular pérdida y predicciones
            val_loss += loss.item()
            val_preds.append(outputs.cpu().numpy())
            val_targets.append(y_batch.cpu().numpy())

    # Calcular pérdida promedio de validación
    avg_val_loss = val_loss / len(val_loader)
    
    # Preparar arrays para métricas
    val_preds_np = np.vstack([pred.reshape(-1, 1) if pred.ndim == 1 else pred for pred in val_preds]).flatten()
    val_targets_np = np.vstack([targ.reshape(-1, 1) if targ.ndim == 1 else targ for targ in val_targets]).flatten()
    
    return avg_val_loss, val_preds_np, val_targets_np


def _predict_in_batches(model: nn.Module,
                      x_cgm: np.ndarray,
                      x_other: np.ndarray,
                      batch_size: int = 64) -> np.ndarray:
    """
    Realiza predicciones en lotes para evitar problemas de memoria.
    
    Parámetros:
    -----------
    model : nn.Module
        Modelo a usar para predicciones
    x_cgm : np.ndarray
        Datos CGM para predicción
    x_other : np.ndarray
        Otras características para predicción
    batch_size : int, opcional
        Tamaño del batch para predicción (default: 64)
        
    Retorna:
    --------
    np.ndarray
        Predicciones del modelo
    """
    model.eval()
    with torch.no_grad():
        # Convertir datos a tensores
        x_cgm_tensor = torch.FloatTensor(x_cgm).to(DEVICE)
        x_other_tensor = torch.FloatTensor(x_other).to(DEVICE)
        
        preds = []
        
        for i in range(0, len(x_cgm), batch_size):
            end_idx = min(i + batch_size, len(x_cgm))
            batch_cgm = x_cgm_tensor[i:end_idx]
            batch_other = x_other_tensor[i:end_idx]
            
            outputs = model(batch_cgm, batch_other)
            preds.append(outputs.cpu().numpy())
        
        return np.vstack(preds).flatten()


def train_and_evaluate_model(model: Union[nn.Module, DLModelWrapperPyTorch], 
                          model_name: str, 
                          data: Dict[str, Dict[str, np.ndarray]],
                          models_dir: str = CONST_MODELS,
                          training_config: Dict[str, Any] = None) -> Tuple[Dict[str, List[float]], np.ndarray, Dict[str, float]]:
    """
    Entrena y evalúa un modelo con características avanzadas de entrenamiento.
    
    Parámetros:
    -----------
    model : nn.Module
        Modelo a entrenar
    model_name : str
        Nombre del modelo para guardado y registro
    data : Dict[str, Dict[str, np.ndarray]]
        Diccionario con datos de entrenamiento, validación y prueba
    models_dir : str, opcional
        Directorio para guardar modelos (default: "models")
    training_config : Dict[str, Any], opcional
        Configuración de entrenamiento
        
    Retorna:
    --------
    Tuple[Dict[str, List[float]], np.ndarray, Dict[str, float]]
        (history, predictions, metrics)
    """
    # Configuración por defecto
    if training_config is None:
        training_config = {
            'epochs': 100,
            'batch_size': 32,
            'learning_rate': 0.001,
            'patience': 10
        }
    
    # Extraer datos
    x_cgm_train = data['train']['x_cgm']
    x_other_train = data['train']['x_other']
    y_train = data['train']['y']
    
    x_cgm_val = data['val']['x_cgm']
    x_other_val = data['val']['x_other']
    y_val = data['val']['y']
    
    x_cgm_test = data['test']['x_cgm']
    x_other_test = data['test']['x_other']
    y_test = data['test']['y']
    
    # Extraer parámetros de configuración
    epochs = training_config.get('epochs', 100)
    batch_size = training_config.get('batch_size', 32)
    learning_rate = training_config.get('learning_rate', 0.001)
    patience = training_config.get('patience', 10)
    
    # Crear directorios necesarios
    os.makedirs(models_dir, exist_ok=True)
    log_dir = os.path.join(models_dir, CONST_LOGS_DIR, model_name)
    os.makedirs(log_dir, exist_ok=True)
    
    # Preparar el modelo
    actual_model = _prepare_model(model, x_cgm_train, x_other_train, y_train)
    
    # Configurar optimizador y función de pérdida
    optimizer = optim.Adam(actual_model.parameters(), lr=learning_rate, weight_decay=1e-6)
    criterion = nn.MSELoss()
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, 'min', factor=0.5, patience=patience // 2, min_lr=1e-6
    )
    
    # Crear dataloaders
    train_loader = create_dataloaders(x_cgm_train, x_other_train, y_train, batch_size)
    val_loader = create_dataloaders(x_cgm_val, x_other_val, y_val, batch_size, shuffle=False)
    
    # Inicializar early stopping
    early_stopping = EarlyStopping(patience=patience, restore_best_weights=True)
    
    # Historial de entrenamiento
    history = {
        'loss': [],
        'val_loss': [],
        CONST_METRIC_MAE: [],
        f"val_{CONST_METRIC_MAE}": [],
        CONST_METRIC_RMSE: [],
        f"val_{CONST_METRIC_RMSE}": []
    }
    
    # Bucle de entrenamiento
    print(f"\nEntrenando modelo {model_name}...")
    print_info(f"Configuración de entrenamiento: {epochs} épocas, batch size: {batch_size}, lr: {learning_rate}")
    print_info(f"Datos: {len(x_cgm_train)} ejemplos de entrenamiento, {len(x_cgm_val) if x_cgm_val is not None else 0} ejemplos de validación")
    
    progress_bar = tqdm(range(epochs), desc="Entrenamiento")
    for epoch in progress_bar:
        # Ejecutar época de entrenamiento
        avg_train_loss = _run_train_epoch(actual_model, train_loader, optimizer, criterion)
        
        # Fase de validación
        if x_cgm_val is not None:
            avg_val_loss, val_preds_np, val_targets_np = _run_validation(actual_model, val_loader, criterion)
            
            # Métricas de validación
            val_mae = mean_absolute_error(val_targets_np, val_preds_np)
            val_rmse = np.sqrt(mean_squared_error(val_targets_np, val_preds_np))
            
            # Actualizar descripción de la barra de progreso
            progress_bar.set_description(
                f"Época {epoch+1}/{epochs} - loss: {avg_train_loss:.4f}, val_loss: {avg_val_loss:.4f}, val_mae: {val_mae:.4f}"
            )
            
            # Actualizar historial
            history['loss'].append(avg_train_loss)
            history['val_loss'].append(avg_val_loss)
            history['val_mae'].append(val_mae)
            history['val_rmse'].append(val_rmse)
            
            # Comprobar early stopping
            if early_stopping(actual_model, avg_val_loss):
                print_info(f"Early stopping en época {epoch+1}")
                break
            
            # Paso del scheduler basado en pérdida de validación
            scheduler.step(avg_val_loss)
        else:
            # Sin validación, solo mostrar pérdida de entrenamiento
            progress_bar.set_description(f"Época {epoch+1}/{epochs} - loss: {avg_train_loss:.4f}")
            
            # Actualizar historial
            history['loss'].append(avg_train_loss)
    
    # Guardar modelo final
    torch.save(actual_model.state_dict(), os.path.join(models_dir, f'{model_name}.pt'))
    
    # Predicciones en conjunto de prueba
    y_pred = _predict_in_batches(actual_model, x_cgm_test, x_other_test)
    
    # Calcular métricas en datos de prueba
    metrics = calculate_metrics(y_test, y_pred)
    
    return history, y_pred, metrics


def train_model_sequential(model_creator: Callable, 
                         name: str, 
                         input_shapes: Tuple[Tuple[int, ...], Tuple[int, ...]], 
                         x_cgm_train: np.ndarray, 
                         x_other_train: np.ndarray, 
                         y_train: np.ndarray,
                         x_cgm_val: np.ndarray, 
                         x_other_val: np.ndarray, 
                         y_val: np.ndarray,
                         x_cgm_test: np.ndarray, 
                         x_other_test: np.ndarray, 
                         y_test: np.ndarray,
                         models_dir: str = CONST_MODELS) -> Dict[str, Any]:
    """
    Entrena un modelo secuencialmente y devuelve resultados serializables.
    
    Parámetros:
    -----------
    model_creator : Callable
        Función que crea el modelo
    name : str
        Nombre del modelo
    input_shapes : Tuple[Tuple[int, ...], Tuple[int, ...]]
        Formas de las entradas (CGM, otras)
    x_cgm_train : np.ndarray
        Datos CGM de entrenamiento
    x_other_train : np.ndarray
        Otras características de entrenamiento
    y_train : np.ndarray
        Valores objetivo de entrenamiento
    x_cgm_val : np.ndarray
        Datos CGM de validación
    x_other_val : np.ndarray
        Otras características de validación
    y_val : np.ndarray
        Valores objetivo de validación
    x_cgm_test : np.ndarray
        Datos CGM de prueba
    x_other_test : np.ndarray
        Otras características de prueba
    y_test : np.ndarray
        Valores objetivo de prueba
    models_dir : str, opcional
        Directorio para guardar modelos (default: "models")
        
    Retorna:
    --------
    Dict[str, Any]
        Diccionario con nombre del modelo, historial y predicciones
    """
    print(f"\nEntrenando modelo {name}...")
    
    # Identificar tipo de modelo para organización de figuras
    model_type = get_model_type(name)
    figures_path = os.path.join(CONST_FIGURES_DIR, CONST_MODEL_TYPES[model_type], name)
    os.makedirs(figures_path, exist_ok=True)
    
    # Crear modelo
    model = model_creator(input_shapes[0], input_shapes[1])
    
    # Organizar datos en estructura esperada
    data = {
        'train': {'x_cgm': x_cgm_train, 'x_other': x_other_train, 'y': y_train},
        'val': {'x_cgm': x_cgm_val, 'x_other': x_other_val, 'y': y_val},
        'test': {'x_cgm': x_cgm_test, 'x_other': x_other_test, 'y': y_test}
    }
    
    # Configuración por defecto
    training_config = {
        'epochs': CONST_EPOCHS,
        'batch_size': CONST_DEFAULT_BATCH_SIZE
    }
    
    # Entrenar y evaluar modelo
    history, y_pred, metrics = train_and_evaluate_model(
        model=model,
        model_name=name,
        data=data,
        models_dir=models_dir,
        training_config=training_config
    )
    
    # Limpiar memoria
    del model
    torch.cuda.empty_cache() if torch.cuda.is_available() else None
    
    # Devolver solo objetos serializables
    return {
        'name': name,
        'history': history,
        'predictions': y_pred.tolist(),
        'metrics': metrics
    }


def cross_validate_model(create_model_fn: Callable, 
                       x_cgm: np.ndarray, 
                       x_other: np.ndarray, 
                       y: np.ndarray, 
                       n_splits: int = 5, 
                       models_dir: str = CONST_MODELS) -> Tuple[Dict[str, float], Dict[str, float]]:
    """
    Realiza validación cruzada para un modelo.
    
    Parámetros:
    -----------
    create_model_fn : Callable
        Función que crea el modelo
    x_cgm : np.ndarray
        Datos CGM
    x_other : np.ndarray
        Otras características
    y : np.ndarray
        Valores objetivo
    n_splits : int, opcional
        Número de divisiones para validación cruzada (default: 5)
    models_dir : str, opcional
        Directorio para guardar modelos (default: "models")
        
    Retorna:
    --------
    Tuple[Dict[str, float], Dict[str, float]]
        (métricas_promedio, métricas_desviación)
    """
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=CONST_DEFAULT_SEED)
    scores = []
    
    for fold, (train_idx, val_idx) in enumerate(kf.split(x_cgm)):
        print(f"\nEntrenando fold {fold + 1}/{n_splits}")
        
        # Dividir datos
        x_cgm_train_fold = x_cgm[train_idx]
        x_cgm_val_fold = x_cgm[val_idx]
        x_other_train_fold = x_other[train_idx]
        x_other_val_fold = x_other[val_idx]
        y_train_fold = y[train_idx]
        y_val_fold = y[val_idx]
        
        # Crear modelo
        model = create_model_fn()
        
        # Organizar datos en estructura esperada
        data = {
            'train': {'x_cgm': x_cgm_train_fold, 'x_other': x_other_train_fold, 'y': y_train_fold},
            'val': {'x_cgm': x_cgm_val_fold, 'x_other': x_other_val_fold, 'y': y_val_fold},
            'test': {'x_cgm': x_cgm_val_fold, 'x_other': x_other_val_fold, 'y': y_val_fold}
        }
        
        # Entrenar y evaluar modelo
        _, _, metrics = train_and_evaluate_model(
            model=model,
            model_name=f'fold_{fold}',
            data=data,
            models_dir=models_dir
        )
        
        scores.append(metrics)
        
        # Limpiar memoria
        del model
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
    
    # Calcular estadísticas
    mean_scores = {
        metric: np.mean([s[metric] for s in scores])
        for metric in scores[0].keys()
    }
    std_scores = {
        metric: np.std([s[metric] for s in scores])
        for metric in scores[0].keys()
    }
    
    return mean_scores, std_scores


def predict_model(model_path: str, 
                model_creator: Callable, 
                x_cgm: np.ndarray, 
                x_other: np.ndarray, 
                input_shapes: Optional[Tuple[Tuple[int, ...], Tuple[int, ...]]] = None) -> np.ndarray:
    """
    Realiza predicciones con un modelo guardado.
    
    Parámetros:
    -----------
    model_path : str
        Ruta al modelo guardado
    model_creator : Callable
        Función que crea el modelo
    x_cgm : np.ndarray
        Datos CGM para predicción
    x_other : np.ndarray
        Otras características para predicción
    input_shapes : Optional[Tuple[Tuple[int, ...], Tuple[int, ...]]], opcional
        Formas de las entradas. Si es None, se infieren (default: None)
        
    Retorna:
    --------
    np.ndarray
        Predicciones del modelo
    """
    # Determinar formas de entrada si no se proporcionan
    if input_shapes is None:
        input_shapes = ((x_cgm.shape[1:]), (x_other.shape[1:]))
    
    # Crear modelo
    model = model_creator(input_shapes[0], input_shapes[1])
    
    # Cargar pesos guardados
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    model = model.to(DEVICE)
    model.eval()
    
    # Hacer predicciones en lotes para evitar problemas de memoria
    with torch.no_grad():
        x_cgm_tensor = torch.FloatTensor(x_cgm).to(DEVICE)
        x_other_tensor = torch.FloatTensor(x_other).to(DEVICE)
        
        batch_size = 64
        predictions = []
        
        for i in range(0, len(x_cgm), batch_size):
            end_idx = min(i + batch_size, len(x_cgm))
            batch_cgm = x_cgm_tensor[i:end_idx]
            batch_other = x_other_tensor[i:end_idx]
            
            outputs = model(batch_cgm, batch_other)
            predictions.append(outputs.cpu().numpy())
    
    # Limpiar memoria
    del model
    torch.cuda.empty_cache() if torch.cuda.is_available() else None
    
    return np.vstack(predictions).flatten()


def train_multiple_models(model_creators: Dict[str, Callable], 
                        input_shapes: Tuple[Tuple[int, ...], Tuple[int, ...]],
                        x_cgm_train: np.ndarray, 
                        x_other_train: np.ndarray, 
                        y_train: np.ndarray,
                        x_cgm_val: np.ndarray, 
                        x_other_val: np.ndarray, 
                        y_val: np.ndarray,
                        x_cgm_test: np.ndarray, 
                        x_other_test: np.ndarray, 
                        y_test: np.ndarray,
                        models_dir: str = CONST_MODELS) -> Tuple[Dict[str, Dict], Dict[str, np.ndarray], Dict[str, Dict]]:
    """
    Entrena múltiples modelos y recopila sus resultados.
    
    Parámetros:
    -----------
    model_creators : Dict[str, Callable]
        Diccionario de funciones creadoras de modelos indexadas por nombre
    input_shapes : Tuple[Tuple[int, ...], Tuple[int, ...]]
        Formas de las entradas (CGM, otras)
    x_cgm_train : np.ndarray
        Datos CGM de entrenamiento
    x_other_train : np.ndarray
        Otras características de entrenamiento
    y_train : np.ndarray
        Valores objetivo de entrenamiento
    x_cgm_val : np.ndarray
        Datos CGM de validación
    x_other_val : np.ndarray
        Otras características de validación
    y_val : np.ndarray
        Valores objetivo de validación
    x_cgm_test : np.ndarray
        Datos CGM de prueba
    x_other_test : np.ndarray
        Otras características de prueba
    y_test : np.ndarray
        Valores objetivo de prueba
    models_dir : str, opcional
        Directorio para guardar modelos (default: "models")
        
    Retorna:
    --------
    Tuple[Dict[str, Dict], Dict[str, np.ndarray], Dict[str, Dict]]
        (historiales, predicciones, métricas) diccionarios
    """
    models_names = list(model_creators.keys())
    
    model_results = []
    for name in models_names:
        result = train_model_sequential(
            model_creators[name], name, input_shapes,
            x_cgm_train, x_other_train, y_train,
            x_cgm_val, x_other_val, y_val,
            x_cgm_test, x_other_test, y_test,
            models_dir
        )
        model_results.append(result)
    
    
    # Guardar resultados
    histories = {}
    predictions = {}
    metrics = {}
    
    for result in model_results:
        name = result['name']
        histories[name] = result['history']
        predictions[name] = np.array(result['predictions'])
        metrics[name] = result['metrics']
    
    return histories, predictions, metrics


def debug_tensor_info(tensor: torch.Tensor, name: str) -> None:
    """
    Imprime información de depuración sobre un tensor.
    
    Parámetros:
    -----------
    tensor : torch.Tensor
        Tensor a debugguear
    name : str
        Nombre del tensor para identificación
    """
    print_debug(f"{name} - Forma: {tensor.shape}, Tipo: {tensor.dtype}, Dispositivo: {tensor.device}")
    print_debug(f"{name} - Min: {tensor.min().item():.4f}, Max: {tensor.max().item():.4f}, Media: {tensor.mean().item():.4f}")
    if torch.isnan(tensor).any():
        print_warning(f"{name} contiene valores NaN")
    if torch.isinf(tensor).any():
        print_warning(f"{name} contiene valores Inf")