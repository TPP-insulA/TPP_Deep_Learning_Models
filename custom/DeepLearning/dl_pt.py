import time
from typing import Dict, List, Tuple, Callable, Optional, Any
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from constants.constants import CONST_DEFAULT_SEED, CONST_DEFAULT_EPOCHS, CONST_DEFAULT_BATCH_SIZE
from custom.model_wrapper import ModelWrapper
from custom.printer import cprint, print_debug, print_info, print_error

# Constantes para mensajes de error
CONST_MODEL_INIT_ERROR = "El modelo debe ser inicializado antes de {}"
CONST_DEVICE = "device"

class DLModelWrapperPyTorch(ModelWrapper):
    """
    Wrapper para modelos de deep learning implementados en PyTorch.
    
    Parámetros:
    -----------
    model_creator : Callable
        Función que crea una instancia del modelo
    """
    
    def __init__(self, model_creator: Callable) -> None:
        """
        Inicializa el wrapper con un creador de modelo PyTorch.
        
        Parámetros:
        -----------
        model_creator : Callable
            Función que crea una instancia del modelo
        """
        super().__init__()
        self.model_creator = model_creator
        self.model = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.optimizer = None
        self.criterion = None
        self.scheduler = None
        self.early_stopping_config = {
            'patience': 10,
            'min_delta': 0.0,
            'restore_best_weights': True,
            'best_weights': None,
            'best_val_loss': float('inf'),
            'counter': 0,
            'early_stop': False
        }
        print_info(f"Usando dispositivo: {self.device}")
    
    def add_early_stopping(self, patience: int = 10, min_delta: float = 0.0, restore_best_weights: bool = True) -> None:
        """
        Configura early stopping para el entrenamiento.
        
        Parámetros:
        -----------
        patience : int, opcional
            Número de épocas a esperar para detener el entrenamiento (default: 10)
        min_delta : float, opcional
            Cambio mínimo para considerar mejora (default: 0.0)
        restore_best_weights : bool, opcional
            Si se deben restaurar los mejores pesos al finalizar (default: True)
        """
        self.early_stopping_config['patience'] = patience
        self.early_stopping_config['min_delta'] = min_delta
        self.early_stopping_config['restore_best_weights'] = restore_best_weights

    def start(self, x_cgm: np.ndarray, x_other: np.ndarray, y: np.ndarray, 
             rng_key: Any = None) -> Any:
        """
        Inicializa el modelo con los datos de entrada.
        
        Parámetros:
        -----------
        x_cgm : np.ndarray
            Datos CGM de entrada
        x_other : np.ndarray
            Otras características de entrada
        y : np.ndarray
            Valores objetivo
        rng_key : Any, opcional
            Clave para generación aleatoria (no usada en PyTorch) (default: None)
            
        Retorna:
        --------
        Any
            Modelo inicializado
        """
        # Crear modelo
        self.model = self.model_creator()
        
        # Mover modelo al dispositivo
        self.model = self.model.to(self.device)
        
        # Configurar criterio y optimizador por defecto
        self.criterion = nn.MSELoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001, weight_decay=1e-5)
        
        # Configurar scheduler por defecto
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, 'min', factor=0.5, patience=5, min_lr=1e-6
        )
        
        # Establecer semilla si se proporciona
        if rng_key is not None:
            if isinstance(rng_key, int):
                torch.manual_seed(rng_key)
                np.random.seed(rng_key)
            else:
                # Asumiendo que es un jax.random.PRNGKey
                try:
                    seed_val = int(rng_key[0])
                    torch.manual_seed(seed_val)
                    np.random.seed(seed_val)
                except (TypeError, IndexError):
                    torch.manual_seed(CONST_DEFAULT_SEED)  # Valor por defecto
        
        return self.model
    
    def _create_dataloader(self, x_cgm: np.ndarray, x_other: np.ndarray, y: np.ndarray, 
                         batch_size: int, shuffle: bool = True) -> DataLoader:
        """
        Crea un DataLoader de PyTorch para los datos de entrada.
        
        Parámetros:
        -----------
        x_cgm : np.ndarray
            Datos CGM
        x_other : np.ndarray
            Otras características
        y : np.ndarray
            Valores objetivo
        batch_size : int
            Tamaño del lote
        shuffle : bool, opcional
            Si mezclar los datos (default: True)
            
        Retorna:
        --------
        DataLoader
            DataLoader de PyTorch con los datos
        """
        # Convertir a tensores de PyTorch
        x_cgm_tensor = torch.FloatTensor(x_cgm)
        x_other_tensor = torch.FloatTensor(x_other)
        y_tensor = torch.FloatTensor(y.reshape(-1, 1))
        
        # Crear dataset y dataloader
        dataset = TensorDataset(x_cgm_tensor, x_other_tensor, y_tensor)
        return DataLoader(
            dataset, 
            batch_size=batch_size, 
            shuffle=shuffle,
            pin_memory=True,
            num_workers=4
        )
    
    def _check_early_stopping(self, val_loss: float) -> bool:
        """
        Verifica si debe activarse el early stopping.
        
        Parámetros:
        -----------
        val_loss : float
            Pérdida de validación actual
            
        Retorna:
        --------
        bool
            True si debe detenerse el entrenamiento, False en caso contrario
        """
        config = self.early_stopping_config
        
        if val_loss < config['best_val_loss'] - config['min_delta']:
            config['best_val_loss'] = val_loss
            if config['restore_best_weights']:
                config['best_weights'] = {k: v.cpu().clone() for k, v in self.model.state_dict().items()}
            config['counter'] = 0
            return False
        else:
            config['counter'] += 1
            if config['counter'] >= config['patience']:
                config['early_stop'] = True
                if config['restore_best_weights'] and config['best_weights'] is not None:
                    self.model.load_state_dict(config['best_weights'])
                return True
        return False
    
    def train(self, x_cgm: np.ndarray, x_other: np.ndarray, y: np.ndarray, 
             validation_data: Optional[Tuple[Tuple[np.ndarray, np.ndarray], np.ndarray]] = None,
             epochs: int = CONST_DEFAULT_EPOCHS, batch_size: int = CONST_DEFAULT_BATCH_SIZE) -> Dict[str, List[float]]:
        """
        Entrena el modelo con los datos proporcionados.
        
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
        
        # Preparar datos de entrenamiento
        train_loader = self._create_dataloader(x_cgm, x_other, y, batch_size)
        
        # Preparar datos de validación si existen
        val_loader = None
        if validation_data is not None:
            (x_cgm_val, x_other_val), y_val = validation_data
            val_loader = self._create_dataloader(x_cgm_val, x_other_val, y_val, batch_size, shuffle=False)
        
        # Historial de entrenamiento
        history = {
            'loss': [],
            'val_loss': [],
            'mae': [],
            'val_mae': [],
            'rmse': [],
            'val_rmse': [],
            'r2': [],
            'val_r2': []
        }
        
        print_info(f"Iniciando entrenamiento para {epochs} épocas...")
        
        # Bucle de entrenamiento
        for epoch in range(epochs):
            epoch_start_time = time.time()
            
            # Modo entrenamiento
            self.model.train()
            train_loss = 0.0
            
            # Entrenar por lotes
            for x_cgm_batch, x_other_batch, y_batch in train_loader:
                # Mover datos al dispositivo
                x_cgm_batch = x_cgm_batch.to(self.device)
                x_other_batch = x_other_batch.to(self.device)
                y_batch = y_batch.to(self.device)
                
                # Limpiar gradientes
                self.optimizer.zero_grad()
                
                # Forward pass
                outputs = self.model(x_cgm_batch, x_other_batch)
                loss = self.criterion(outputs, y_batch)
                
                # Backward pass y optimización
                loss.backward()
                self.optimizer.step()
                
                # Acumular pérdida
                train_loss += loss.item() * x_cgm_batch.size(0)
            
            # Calcular pérdida promedio de entrenamiento
            train_loss = train_loss / len(train_loader.dataset)
            history['loss'].append(train_loss)
            
            # Modo evaluación para validación
            self.model.eval()
            
            # Validación si hay datos disponibles
            if val_loader is not None:
                val_metrics = self._validate(val_loader)
                
                # Actualizar historial con métricas de validación
                for key, value in val_metrics.items():
                    history[f"val_{key}"].append(value)
                
                # Ajustar learning rate basado en pérdida de validación
                self.scheduler.step(val_metrics['loss'])
                
                # Verificar early stopping
                if self._check_early_stopping(val_metrics['loss']):
                    print_info(f"Early stopping activado en época {epoch+1}")
                    break
                
                # Imprimir progreso con métricas de validación
                epoch_time = time.time() - epoch_start_time
                print_info(f"Época {epoch+1}/{epochs} - loss: {train_loss:.4f} - val_loss: {val_metrics['loss']:.4f} - "
                      f"val_mae: {val_metrics['mae']:.4f} - val_rmse: {val_metrics['rmse']:.4f} - "
                      f"val_r2: {val_metrics['r2']:.4f} - tiempo: {epoch_time:.2f}s")
            else:
                # Imprimir progreso sin validación
                epoch_time = time.time() - epoch_start_time
                print_info(f"Época {epoch+1}/{epochs} - loss: {train_loss:.4f} - tiempo: {epoch_time:.2f}s")
        
        return history
    
    def _validate(self, val_loader: DataLoader) -> Dict[str, float]:
        """
        Evalúa el modelo en el conjunto de validación.
        
        Parámetros:
        -----------
        val_loader : DataLoader
            DataLoader con datos de validación
            
        Retorna:
        --------
        Dict[str, float]
            Diccionario con métricas de validación
        """
        val_loss = 0.0
        all_preds = []
        all_targets = []
        
        with torch.no_grad():
            for x_cgm_batch, x_other_batch, y_batch in val_loader:
                # Mover datos al dispositivo
                x_cgm_batch = x_cgm_batch.to(self.device)
                x_other_batch = x_other_batch.to(self.device)
                y_batch = y_batch.to(self.device)
                
                # Forward pass
                outputs = self.model(x_cgm_batch, x_other_batch)
                loss = self.criterion(outputs, y_batch)
                
                # Acumular pérdida
                val_loss += loss.item() * x_cgm_batch.size(0)
                
                # Guardar predicciones y targets para métricas
                all_preds.append(outputs.cpu().numpy())
                all_targets.append(y_batch.cpu().numpy())
        
        # Calcular pérdida promedio
        val_loss = val_loss / len(val_loader.dataset)
        
        # Combinar predicciones y targets
        all_preds = np.vstack(all_preds).flatten()
        all_targets = np.vstack(all_targets).flatten()
        
        # Calcular métricas adicionales
        mae = float(np.mean(np.abs(all_preds - all_targets)))
        rmse = float(np.sqrt(np.mean((all_preds - all_targets) ** 2)))
        
        # Calcular R²
        ss_res = np.sum((all_targets - all_preds) ** 2)
        ss_tot = np.sum((all_targets - np.mean(all_targets)) ** 2)
        r2 = float(1 - (ss_res / (ss_tot + 1e-8)))
        
        return {
            'loss': val_loss,
            'mae': mae,
            'rmse': rmse,
            'r2': r2
        }
    
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
        if self.model is None:
            raise ValueError(CONST_MODEL_INIT_ERROR.format("predecir"))
        
        # Convertir a tensores
        x_cgm_tensor = torch.FloatTensor(x_cgm).to(self.device)
        x_other_tensor = torch.FloatTensor(x_other).to(self.device)
        
        # Modo evaluación
        self.model.eval()
        
        # Predicciones en lotes para evitar problemas de memoria
        batch_size = 128
        predictions = []
        
        with torch.no_grad():
            for i in range(0, len(x_cgm), batch_size):
                batch_end = min(i + batch_size, len(x_cgm))
                batch_cgm = x_cgm_tensor[i:batch_end]
                batch_other = x_other_tensor[i:batch_end]
                
                outputs = self.model(batch_cgm, batch_other)
                predictions.append(outputs.cpu().numpy())
        
        # Combinar predicciones
        return np.vstack(predictions).flatten()
    
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
        
        # Crear dataloader para evaluación
        test_loader = self._create_dataloader(x_cgm, x_other, y, batch_size=128, shuffle=False)
        
        # Evaluar el modelo
        metrics = self._validate(test_loader)
        
        return metrics
    
    def save(self, path: str) -> None:
        """
        Guarda el modelo en disco.
        
        Parámetros:
        -----------
        path : str
            Ruta donde guardar el modelo
        """
        if self.model is None:
            raise ValueError(CONST_MODEL_INIT_ERROR.format("guardar"))
        
        # Guardar modelo y configuración de entrenamiento
        save_dict = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict() if self.optimizer else None,
            'early_stopping_config': self.early_stopping_config
        }
        
        torch.save(save_dict, path)
    
    def load(self, path: str) -> None:
        """
        Carga el modelo desde disco.
        
        Parámetros:
        -----------
        path : str
            Ruta desde donde cargar el modelo
        """
        # Crear modelo si no existe
        if self.model is None:
            self.model = self.model_creator()
            self.model = self.model.to(self.device)
        
        # Cargar estado guardado
        checkpoint = torch.load(path, map_location=self.device)
        
        # Cargar estado del modelo
        self.model.load_state_dict(checkpoint['model_state_dict'])
        
        # Cargar estado del optimizador si existe
        if checkpoint['optimizer_state_dict'] and self.optimizer:
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        # Cargar configuración de early stopping si existe
        if 'early_stopping_config' in checkpoint:
            self.early_stopping_config = checkpoint['early_stopping_config']
