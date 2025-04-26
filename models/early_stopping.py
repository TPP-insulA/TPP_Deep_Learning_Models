from typing import Any, Tuple, Union
from config.models_config import EARLY_STOPPING_POLICY

class EarlyStopping:
    """
    Implementa early stopping para detener el entrenamiento cuando una métrica deja de mejorar.
    
    Parámetros:
    -----------
    patience : int
        Número de épocas a esperar antes de detener el entrenamiento
    min_delta : float, opcional
        Cambio mínimo considerado como mejora (default: 0.0)
    restore_best_weights : bool, opcional
        Si restaurar los mejores pesos cuando se detiene (default: True)
    """
    
    def __init__(self, patience: int, min_delta: float = 0.0, restore_best_weights: bool = True) -> None:
        """
        Inicializa el callback de early stopping.
        """
        self.patience = patience
        self.min_delta = min_delta
        self.restore_best_weights = restore_best_weights
        self.best_val_loss = float('inf')
        self.best_params = None
        self.wait = 0
        self.stopped_epoch = 0
        self.should_stop = False
    
    def __call__(self, epoch: int, val_loss: float, params: Any) -> bool:
        """
        Actualiza el estado y determina si el entrenamiento debe detenerse.
        
        Parámetros:
        -----------
        epoch : int
            Época actual
        val_loss : float
            Valor de pérdida de validación de la época actual
        params : Any
            Parámetros del modelo de la época actual
            
        Retorna:
        --------
        bool
            True si el entrenamiento debe detenerse, False en caso contrario
        """
        if val_loss < self.best_val_loss - self.min_delta:
            # Hay mejora
            self.best_val_loss = val_loss
            self.wait = 0
            if self.restore_best_weights:
                self.best_params = params
        else:
            # No hay mejora significativa
            self.wait += 1
            if self.wait >= self.patience:
                self.stopped_epoch = epoch
                self.should_stop = True
                return True
        return False
    
    def get_best_params(self) -> Any:
        """
        Retorna los mejores parámetros encontrados durante el entrenamiento.
        
        Retorna:
        --------
        Any
            Mejores parámetros del modelo si restore_best_weights es True,
            None en caso contrario
        """
        return self.best_params if self.restore_best_weights else None

def get_early_stopping() -> Union[EarlyStopping, None]:
    """
    Crea una instancia de EarlyStopping según la configuración.
    
    Retorna:
    --------
    Union[EarlyStopping, None]
        Instancia de EarlyStopping si está habilitado en la configuración,
        None en caso contrario
    """
    if EARLY_STOPPING_POLICY.get('early_stopping', False):
        return EarlyStopping(
            patience=EARLY_STOPPING_POLICY.get('early_stopping_patience', 10),
            min_delta=EARLY_STOPPING_POLICY.get('early_stopping_min_delta', 0.001),
            restore_best_weights=EARLY_STOPPING_POLICY.get('early_stopping_restore_best', True)
        )
    return None

def get_early_stopping_config() -> Tuple[int, float, bool]:
    """
    Retorna la configuración de EarlyStopping.
    
    Retorna:
    --------
    Tuple[int, float, bool]
        (patience, min_delta, restore_best_weights)
    """
    return (
        EARLY_STOPPING_POLICY.get('early_stopping_patience', 10),
        EARLY_STOPPING_POLICY.get('early_stopping_min_delta', 0.001),
        EARLY_STOPPING_POLICY.get('early_stopping_restore_best', True)
    )