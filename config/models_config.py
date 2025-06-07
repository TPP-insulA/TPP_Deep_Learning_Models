"""Model configuration parameters"""
from constants.constants import CONST_DEFAULT_SEED

###########################################################
###                Global Configuration                 ###
###########################################################
EARLY_STOPPING_POLICY = {
    'early_stopping': True,                         # Activar/desactivar early stopping
    'early_stopping_patience': 30,                  # Épocas a esperar antes de detener
    'early_stopping_min_delta': 0.0001,              # Mejora mínima considerada significativa
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
DDPG_CONFIG = {
    # Parámetros básicos de la red
    "action_dim": 1,                # Dimensión de la acción (dosis de insulina)
    "hidden_dim": 256,              # Dimensión de las capas ocultas
    
    # Parámetros de aprendizaje
    "actor_lr": 1e-4,               # Tasa de aprendizaje para el actor
    "critic_lr": 1e-3,              # Tasa de aprendizaje para el crítico
    "gamma": 0.99,                  # Factor de descuento para recompensas futuras
    "tau": 0.005,                   # Parámetro de actualización suave para redes objetivo
    
    # Parámetros del buffer de experiencia
    "buffer_size": 100000,          # Capacidad máxima del buffer
    
    # Límites de acción
    "max_action": 20.0,             # Valor máximo de acción (dosis máxima de insulina)
    "min_action": 0.0,              # Valor mínimo de acción (dosis mínima de insulina)
    
    # Parámetros de exploración
    "exploration_noise": 0.1,       # Desviación estándar del ruido de exploración
    
    # Otros parámetros
    "seed": CONST_DEFAULT_SEED                      # Semilla aleatoria para reproducibilidad
}