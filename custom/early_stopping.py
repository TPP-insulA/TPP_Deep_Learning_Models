import torch
import torch.nn as nn

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


class ClinicalEarlyStopping:
    """
    Early stopping basado en métricas clínicas para prevenir el sobreajuste.
    
    Parámetros:
    -----------
    patience : int, opcional
        Número de épocas a esperar antes de detener (default: 10)
    min_delta : float, opcional
        Cambio mínimo para considerar como mejora (default: 0.001)
    restore_best_weights : bool, opcional
        Si restaurar los mejores pesos del modelo (default: True)
    monitor : str, opcional
        Métrica a monitorear ('time_in_range' o 'val_loss') (default: 'time_in_range')
    mode : str, opcional
        'max' para métricas donde valores más altos son mejores, 'min' para lo contrario (default: 'max')
    """
    def __init__(self, patience: int = 10, min_delta: float = 0.001, 
                restore_best_weights: bool = True, monitor: str = 'time_in_range',
                mode: str = 'max'):
        self.patience = patience
        self.min_delta = min_delta
        self.restore_best_weights = restore_best_weights
        self.monitor = monitor
        self.mode = mode
        
        # Inicialización según el modo (maximizar o minimizar)
        if self.mode == 'max':
            self.best_score = float('-inf')
            self.is_better = lambda current, best: current > best + self.min_delta
        else:
            self.best_score = float('inf')
            self.is_better = lambda current, best: current < best - self.min_delta
            
        self.best_weights = None
        self.counter = 0
        self.early_stop = False

    def __call__(self, model: nn.Module, current_score: float) -> bool:
        """
        Verifica si debe activarse el early stopping.
        
        Parámetros:
        -----------
        model : nn.Module
            Modelo PyTorch para guardar sus pesos
        current_score : float
            Valor actual de la métrica monitoreada
            
        Retorna:
        --------
        bool
            True si debe detenerse el entrenamiento, False en caso contrario
        """
        if self.is_better(current_score, self.best_score):
            self.best_score = current_score
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
