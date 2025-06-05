"""Model configuration parameters"""

###########################################################
###                Global Configuration                 ###
###########################################################
EARLY_STOPPING_POLICY = {
    'early_stopping': True,                         # Activar/desactivar early stopping
    'early_stopping_patience': 10,                  # Épocas a esperar antes de detener
    'early_stopping_min_delta': 0.001,              # Mejora mínima considerada significativa
    'early_stopping_restore_best_weights': True,    # Restaurar mejores pesos al finalizar
    'early_stopping_best_val_loss': float('inf'),   # Mejor pérdida de validación
    'early_stopping_best_loss': float('inf'),       # Mejor pérdida de entrenamiento
    'early_stopping_counter': 0,                    # Contador de épocas sin mejora
    'early_stopping_best_epoch': 0,                 # Época con mejor pérdida de validación
    'early_stopping_best_weights': None,            # Mejores pesos del modelo
}

###########################################################
###         Deep Reinforcement Learning Models          ###
###########################################################
