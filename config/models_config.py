"""Model configuration parameters"""

###########################################################
###                Global Configuration                 ###
###########################################################
EARLY_STOPPING_POLICY = {
    'early_stopping': True,              # Activar/desactivar early stopping
    'early_stopping_patience': 10,       # Épocas a esperar antes de detener
    'early_stopping_min_delta': 0.001,   # Mejora mínima considerada significativa
    'early_stopping_restore_best_weights': True, # Restaurar mejores pesos al finalizar
    'early_stopping_best_val_loss': float('inf'), # Mejor pérdida de validación
    'early_stopping_counter': 0,         # Contador de épocas sin mejora
    'early_stopping_best_epoch': 0,      # Época con mejor pérdida de validación
    'early_stopping_best_weights': None, # Mejores pesos del modelo
}
###########################################################
###                Deep Learning Models                 ###
###########################################################
ATTENTION_CONFIG = {
    # Arquitectura
    'num_heads': 8,               # Número de cabezas de atención
    'key_dim': 64,                # Dimensión de las claves en cada cabeza de atención
    'head_size': 32,              # Tamaño de salida de cada cabeza de atención
    'num_layers': 4,              # Número de capas de atención apiladas
    'ff_dim': 256,                # Dimensión de la capa feed-forward después de la atención
    'num_blocks': 2,              # Número de bloques de atención
    'use_bias': True,             # Si incluir términos de sesgo en las capas
    'use_layer_norm': True,       # Si usar normalización de capa
    'use_pos_encoding': True,     # Si usar codificación posicional
    'embed_dim': 128,             # Dimensión de la codificación de entrada
    'dense_units': [128, 64],     # Unidades en capas densas finales
    
    # Regularización
    'dropout_rate': 0.1,          # Tasa de dropout para regularización
    'layer_dropout': 0.1,         # Tasa de dropout a nivel de capa
    
    # Mecanismo de atención
    'use_relative_attention': True, # Si usar mecanismo de atención relativa
    'max_relative_position': 32,  # Máxima distancia para atención relativa
    'use_mask_future': False,     # Si enmascarar posiciones futuras (para modelos causales)
    
    # Activación
    'activation': 'gelu'          # Función de activación utilizada
}

CNN_CONFIG = {
    # Arquitectura
    'filters': [32, 64, 128, 256], # Número de filtros en cada capa convolucional
    'kernel_size': 3,             # Tamaño del kernel para las convoluciones
    'pool_size': 2,               # Tamaño del pool para capas de pooling
    'dilation_rates': [1, 2, 4],  # Tasas de dilatación para convoluciones dilatadas
    
    # Bloques especiales
    'use_se_block': True,         # Si usar bloques Squeeze-and-Excitation
    'se_ratio': 16,               # Ratio de reducción para bloques SE
    
    # Normalización y regularización
    'use_layer_norm': True,       # Si usar normalización de capa
    'dropout_rate': 0.2,          # Tasa de dropout para regularización
    
    # Activación
    'activation': 'gelu'          # Función de activación utilizada
}

FNN_CONFIG = {
    # Arquitectura
    'hidden_units': [256, 256, 128, 64],  # Unidades en capas ocultas
    'final_units': [32],                  # Unidades en capas finales
    'activation': 'gelu',                 # Función de activación
    'use_layer_norm': True,               # Usar Layer Normalization en lugar de Batch Normalization
    'epsilon': 1e-6,                      # Epsilon para normalización
    
    # Regularización
    'dropout_rates': [0.2, 0.2, 0.3, 0.3],  # Tasas de dropout para capas ocultas
    'final_dropout_rate': 0.2,              # Tasa de dropout para capas finales
    
    # Optimización
    'learning_rate': 1e-3,                # Tasa de aprendizaje inicial
    'beta_1': 0.9,                        # Beta 1 para Adam
    'beta_2': 0.999,                      # Beta 2 para Adam
    'optimizer_epsilon': 1e-7,            # Epsilon para el optimizador
    'batch_size': 64,                     # Tamaño del lote por defecto
    'epochs': 100,                        # Número máximo de épocas
    
    # Ajuste de tasa de aprendizaje
    'lr_reduction_factor': 0.5,           # Factor de reducción para ReduceLROnPlateau
    'lr_patience': 5,                     # Paciencia para ReduceLROnPlateau
    'min_learning_rate': 1e-6,            # Tasa de aprendizaje mínima
    'patience': 10,                       # Paciencia para EarlyStopping
    
    # Tipo de tarea
    'regression': True,                   # Si es regresión (True) o clasificación (False)
    'num_classes': 2                      # Número de clases para clasificación
}

GRU_CONFIG = {
    # Arquitectura
    'hidden_units': [64, 128, 256],  # Unidades en cada capa GRU
    'attention_heads': 4,            # Número de cabezas si se usa atención
    
    # Regularización
    'dropout_rate': 0.3,             # Tasa de dropout para regularización
    'recurrent_dropout': 0.2,        # Tasa de dropout para conexiones recurrentes
    
    # Estabilidad numérica
    'epsilon': 1e-5                  # Epsilon para estabilidad numérica
}

LSTM_CONFIG = {
    # Arquitectura
    'hidden_units': [64, 128, 256],  # Unidades en cada capa LSTM
    'dense_units': [128, 64],        # Unidades en capas densas posteriores
    'attention_heads': 4,            # Número de cabezas si se usa atención
    'use_bidirectional': True,       # Si usar LSTMs bidireccionales
    
    # Activación
    'activation': 'tanh',            # Activación para celdas LSTM
    'recurrent_activation': 'sigmoid', # Activación recurrente para celdas LSTM
    'dense_activation': 'gelu',      # Activación para capas densas
    
    # Regularización
    'dropout_rate': 0.25,            # Tasa de dropout para regularización
    'recurrent_dropout': 0.15,       # Tasa de dropout para conexiones recurrentes
    'use_layer_norm': True,          # Si usar normalización de capa
    
    # Estabilidad numérica
    'epsilon': 1e-6                  # Epsilon para estabilidad numérica
}

RNN_CONFIG = {
    # Arquitectura
    'hidden_units': [64, 32],        # Unidades en cada capa RNN
    'bidirectional': True,           # Si usar RNNs bidireccionales
    'use_time_distributed': True,    # Si aplicar salida a cada paso temporal
    
    # Regularización
    'dropout_rate': 0.2,             # Tasa de dropout para regularización
    'recurrent_dropout': 0.1,        # Tasa de dropout para conexiones recurrentes
    
    # Activación
    'activation': 'relu',            # Función de activación para celdas RNN
    
    # Estabilidad numérica
    'epsilon': 1e-6                  # Epsilon para estabilidad numérica
}

TABNET_CONFIG = {
    # Arquitectura
    'feature_dim': 128,              # Dimensión de la transformación de características
    'output_dim': 64,                # Dimensión de salida
    'num_decision_steps': 8,         # Número de pasos de decisión secuenciales
    'num_attention_heads': 4,         # Número de cabezas de atención
    
    # Regularización
    'sparsity_coefficient': 1e-4,    # Coeficiente para penalizar la falta de sparsity
    'relaxation_factor': 1.5,        # Factor de relajación para atención sparse
    'attention_dropout': 0.2,        # Tasa de dropout para atención
    'feature_dropout': 0.1,          # Tasa de dropout para características
    
    # Lotes y normalización
    'batch_momentum': 0.98,          # Momentum para normalización por lotes
    'virtual_batch_size': 128        # Tamaño de lote virtual para normalización
}

TCN_CONFIG = {
    # Arquitectura
    'filters': [32, 64, 128],        # Filtros en cada capa convolucional temporal
    'kernel_size': 3,                # Tamaño del kernel para convoluciones
    'dilations': [1, 2, 4, 8, 16],   # Tasas de dilatación para el campo receptivo
    
    # Activación
    'activation': 'gelu',            # Función de activación
    
    # Regularización
    'dropout_rate': [0.2, 0.1],      # Tasas de dropout para diferentes capas
    'residual_dropout': 0.1,         # Tasa de dropout para conexiones residuales
    'use_spatial_dropout': True,     # Si usar dropout espacial
    
    # Normalización
    'use_layer_norm': True,          # Si usar normalización de capa
    'use_weight_norm': True,         # Si usar normalización de peso
    'epsilon': 1e-6                  # Epsilon para normalización
}

TRANSFORMER_CONFIG = {
    # Arquitectura
    'num_heads': 8,                  # Número de cabezas de atención
    'key_dim': 64,                   # Dimensión de las claves en mecanismo de atención
    'num_layers': 4,                 # Número de capas de codificador/decodificador
    'ff_dim': 256,                   # Dimensión de la capa feed-forward
    'head_size': 32,                 # Tamaño de salida por cabeza
    
    # Codificación de posición
    'use_relative_pos': True,        # Si usar codificación de posición relativa
    'max_position': 32,              # Máxima distancia para posiciones relativas
    
    # Normalización y regularización
    'dropout_rate': 0.1,             # Tasa de dropout para regularización
    'epsilon': 1e-6,                 # Epsilon para normalización de capa
    'prenorm': True,                 # Si aplicar normalización antes de cada sub-capa
    
    # Configuración de capas
    'activation': 'gelu',            # Función de activación
    'use_bias': True                 # Si incluir términos de sesgo
}

WAVENET_CONFIG = {
    # Arquitectura
    'filters': [32, 64, 128],        # Filtros en cada capa convolucional
    'kernel_size': 3,                # Tamaño del kernel para convoluciones
    'dilations': [1, 2, 4, 8, 16],   # Tasas de dilatación para aumentar campo receptivo
    
    # Mecanismos especiales
    'use_gating': True,              # Si usar mecanismo de compuertas multiplicativas
    'use_skip_scale': True,          # Si escalar conexiones de salto
    'use_residual_scale': 0.1,       # Factor de escala para conexiones residuales
    
    # Regularización
    'dropout_rate': 0.2,             # Tasa de dropout para regularización
    
    # Activación
    'activation': 'elu'              # Función de activación
}

###########################################################
###         Deep Reinforcement Learning Models          ###
###########################################################

A2C_A3C_CONFIG = {
    # Arquitectura
    'hidden_units': [256, 256, 128, 64], # Unidades en cada capa oculta de las redes
    'dropout_rate': 0.1,                 # Tasa de dropout para regularización
    'epsilon': 1e-5,                     # Epsilon para estabilidad numérica en normalizaciones
    
    # Optimización
    'learning_rate': 3e-4,               # Tasa de aprendizaje para actualización de parámetros
    'gamma': 0.99,                       # Factor de descuento para recompensas futuras
    'lambda': 0.95,                      # Factor lambda para cálculo de ventaja generalizada (GAE)
    'entropy_coef': 0.01,                # Coeficiente para el término de entropía en la función de pérdida
    'value_coef': 0.5,                   # Coeficiente para la pérdida de la función de valor
    'max_grad_norm': 0.5                 # Valor máximo para recorte de norma del gradiente
}

DDPG_CONFIG = {
    # Arquitectura
    'actor_hidden_units': [400, 300],    # Unidades en cada capa oculta de la red del actor
    'critic_hidden_units': [400, 300],   # Unidades en cada capa oculta de la red del crítico
    'actor_activation': 'relu',          # Función de activación para la red del actor
    'critic_activation': 'relu',         # Función de activación para la red del crítico
    'dropout_rate': 0.0,                 # Tasa de dropout para regularización
    'epsilon': 1e-6,                     # Epsilon para estabilidad numérica
    
    # Optimización
    'actor_lr': 1e-4,                    # Tasa de aprendizaje para la red del actor
    'critic_lr': 1e-3,                   # Tasa de aprendizaje para la red del crítico
    'gamma': 0.99,                       # Factor de descuento para recompensas futuras
    'tau': 0.001,                        # Factor de suavizado para actualización de redes objetivo
    
    # Buffer de experiencia
    'buffer_capacity': 100000,           # Capacidad máxima del buffer de experiencia
    'batch_size': 64,                    # Tamaño del lote para entrenamiento
    
    # Exploración
    'noise_std': 0.2                     # Desviación estándar del ruido para exploración
}

DQN_CONFIG = {
    # Arquitectura
    'hidden_units': [256, 256, 128, 64], # Unidades en cada capa oculta de la red
    'activation': 'relu',                # Función de activación para las capas
    'dropout_rate': 0.1,                 # Tasa de dropout para regularización
    'epsilon': 1e-5,                     # Epsilon para estabilidad numérica
    'dueling': True,                     # Si usar arquitectura dueling (separación de valor de estado y ventajas)
    'double': True,                      # Si usar double DQN (dos redes para reducir sobreestimación)
    
    # Optimización
    'learning_rate': 3e-4,               # Tasa de aprendizaje
    'gamma': 0.99,                       # Factor de descuento para recompensas futuras
    'target_update_freq': 1000,          # Frecuencia de actualización de la red objetivo
    
    # Exploración
    'epsilon_start': 1.0,                # Valor inicial de epsilon para exploración
    'epsilon_end': 0.01,                 # Valor mínimo de epsilon
    'epsilon_decay': 0.995,              # Factor de decaimiento de epsilon
    
    # Buffer de experiencia
    'buffer_capacity': 100000,           # Capacidad máxima del buffer de experiencia
    'batch_size': 64,                    # Tamaño del lote para entrenamiento
    
    # Experience replay priorizado
    'prioritized': False,                # Si usar experience replay priorizado
    'priority_alpha': 0.6,               # Exponente alpha para determinar prioridad
    'priority_beta': 0.4,                # Exponente beta para corrección de sesgo de importancia
    'priority_beta_increment': 1e-3      # Incremento de beta por paso
}

PPO_CONFIG = {
    # Arquitectura
    'hidden_units': [256, 256, 128, 64], # Unidades en cada capa oculta de las redes
    'dropout_rate': 0.1,                 # Tasa de dropout para regularización
    'epsilon': 1e-5,                     # Epsilon para estabilidad numérica en normalizaciones
    
    # Optimización
    'learning_rate': 3e-4,               # Tasa de aprendizaje para actualización de parámetros
    'gamma': 0.99,                       # Factor de descuento para recompensas futuras
    'clip_epsilon': 0.2,                 # Parámetro epsilon para recorte del ratio de probabilidades
    'entropy_coef': 0.01,                # Coeficiente para el término de entropía en la función de pérdida
    'value_coef': 0.5,                   # Coeficiente para la pérdida de la función de valor
    'max_grad_norm': 0.5,                # Valor máximo para recorte de norma del gradiente
    'lambda': 0.95,                      # Factor lambda para cálculo de ventaja generalizada (GAE)
}

SAC_CONFIG = {
    # Arquitectura
    'actor_hidden_units': [256, 256],    # Unidades en cada capa oculta de la red del actor
    'critic_hidden_units': [256, 256],   # Unidades en cada capa oculta de la red del crítico
    'actor_activation': 'relu',          # Función de activación para la red del actor
    'critic_activation': 'relu',         # Función de activación para la red del crítico
    'dropout_rate': 0.0,                 # Tasa de dropout para regularización
    'epsilon': 1e-6,                     # Epsilon para estabilidad numérica
    
    # Optimización
    'actor_lr': 3e-4,                    # Tasa de aprendizaje para la red del actor
    'critic_lr': 3e-4,                   # Tasa de aprendizaje para la red del crítico
    'alpha_lr': 3e-4,                    # Tasa de aprendizaje para el parámetro de temperatura
    'gamma': 0.99,                       # Factor de descuento para recompensas futuras
    'tau': 0.005,                        # Factor de suavizado para actualización de redes objetivo
    
    # Buffer de experiencia
    'buffer_capacity': 1000000,          # Capacidad máxima del buffer de experiencia
    'batch_size': 256,                   # Tamaño del lote para entrenamiento
    
    # Regularización por entropía
    'initial_alpha': 0.2,                # Valor inicial del parámetro de temperatura (balance exploración-explotación)
    'log_std_min': -20,                  # Límite inferior para el logaritmo de la desviación estándar de la política
    'log_std_max': 2                     # Límite superior para el logaritmo de la desviación estándar de la política
}

TRPO_CONFIG = {
    # Arquitectura
    'hidden_units': [64, 64],            # Unidades en cada capa oculta de las redes
    'use_layer_norm': True,              # Si usar normalización de capas
    'epsilon': 1e-6,                     # Epsilon para estabilidad numérica en normalizaciones
    
    # Parámetros específicos de TRPO
    'delta': 0.01,                       # Límite máximo de divergencia KL para actualizaciones de política
    'gamma': 0.99,                       # Factor de descuento para recompensas futuras
    'lambda': 0.95,                      # Factor lambda para cálculo de ventaja generalizada (GAE)
    'critic_learning_rate': 3e-4,        # Tasa de aprendizaje para la red de valor
    'backtrack_iters': 10,               # Número máximo de iteraciones para búsqueda de línea
    'backtrack_coeff': 0.8,              # Coeficiente de reducción para búsqueda de línea
    'cg_iters': 10,                      # Número de iteraciones para el algoritmo de gradiente conjugado
    'damping': 0.1,                      # Coeficiente de amortiguamiento para estabilidad numérica
    
    # Entrenamiento
    'iterations': 500,                   # Número total de iteraciones de entrenamiento
    'min_steps_per_update': 2048,        # Número mínimo de pasos a recolectar antes de cada actualización
    'value_epochs': 10,                  # Número de épocas para entrenar la función de valor
    'batch_size': 64,                    # Tamaño del lote para entrenamiento
    'evaluate_interval': 10              # Frecuencia para realizar evaluaciones durante entrenamiento
}

###########################################################
###           Reinforcement Learning Models             ###
###########################################################

MONTE_CARLO_CONFIG = {
    # Parámetros generales
    'gamma': 0.99,               # Factor de descuento para recompensas futuras
    'episodes': 1000,            # Número de episodios para entrenamiento
    'max_steps': 1000,           # Pasos máximos por episodio
    
    # Parámetros de exploración
    'epsilon_start': 1.0,        # Epsilon inicial para políticas epsilon-greedy
    'epsilon_end': 0.01,         # Epsilon final mínimo
    'epsilon_decay': 0.995,      # Factor de decaimiento de epsilon por episodio
    
    # Configuración de algoritmo
    'first_visit': True,         # Si True, usa first-visit MC; si False, every-visit MC
    
    # Parámetros de entrenamiento
    'log_interval': 10,          # Cada cuántos episodios mostrar estadísticas
    'batch_size': 10,            # Tamaño de lote para batch MC
    
    # Visualización
    'smoothing_window': 20       # Ventana para suavizado en gráficos
}

POLICY_ITERATION_CONFIG = {
    # Parámetros del algoritmo
    'gamma': 0.99,                # Factor de descuento para recompensas futuras
    'theta': 1e-6,                # Umbral de convergencia para detener iteraciones
    'max_iterations': 100,        # Número máximo de bucles de iteración de política
    'max_iterations_eval': 1000,  # Iteraciones máximas para evaluación de política
    
    # Discretización
    'bins': 10,                  # Número de bins por dimensión para discretización
    'state_bounds': None,        # Límites para cada dimensión del espacio de estados [(min1, max1), (min2, max2), ...]
    'action_bounds': None,       # Límites para cada dimensión del espacio de acciones [(min1, max1), (min2, max2), ...]
    'num_actions': None,        # Número de acciones posibles (si no se especifica, se calcula automáticamente)
    'discretization': 'uniform', # Método de discretización: 'uniform' o 'adaptive'
    
    # Visualización
    'visualization_interval': 5   # Intervalo de iteraciones para mostrar visualizaciones
}

QLEARNING_CONFIG = {
    # Parámetros de aprendizaje
    'learning_rate': 0.1,           # Tasa de aprendizaje (alpha) para actualización de valores Q
    'gamma': 0.99,                  # Factor de descuento para recompensas futuras
    'epsilon_start': 1.0,           # Valor inicial para política epsilon-greedy
    'epsilon_end': 0.01,            # Valor mínimo para epsilon
    'epsilon_decay': 0.995,         # Factor de decaimiento para epsilon
    'decay_steps': 1000,             # Pasos para decaimiento de epsilon
    'use_decay_schedule': True,      # Si usar programación de decaimiento para epsilon
    'decay_schedule': 'exponential', # Tipo de programación de decaimiento: 'exponential' o 'linear'
    'decay_rate': 0.99,             # Tasa de decaimiento para programación exponencial
    
    # Discretización del espacio de estados
    'bins': 10,                     # Número de bins por dimensión para discretización
    'state_bounds': None,           # Límites para cada dimensión del espacio de estados [(min1, max1), (min2, max2), ...]
    
    # Parámetros de entrenamiento
    'episodes': 1000,               # Número de episodios de entrenamiento
    'max_steps_per_episode': 500,               # Máximo de pasos por episodio
    'eval_interval': 100,           # Intervalo de episodios para evaluación
    'eval_episodes': 10,            # Número de episodios para evaluación
    
    # Visualización y registro
    'render_train': False,          # Si renderizar durante entrenamiento
    'render_eval': True,            # Si renderizar durante evaluación
    'log_interval': 50,             # Intervalo para mostrar métricas
    'smoothing_window': 10,          # Ventana para suavizado de métricas
    
    # Inicialización
    'optimistic_init': False, # Si usar inicialización optimista de valores Q
    'optimistic_value': 0.0,    # Valor inicial para inicialización optimista
}

REINFORCE_CONFIG = {
    # Arquitectura de red
    'hidden_units': [128, 64],        # Unidades en cada capa oculta de la red
    'dropout_rate': 0.0,              # Tasa de dropout para regularización
    'epsilon': 1e-6,                  # Epsilon para estabilidad numérica en normalización
    
    # Parámetros de aprendizaje
    'learning_rate': 1e-3,            # Tasa de aprendizaje para actualización de pesos
    'gamma': 0.99,                    # Factor de descuento para recompensas futuras
    'entropy_coef': 0.01,             # Coeficiente para regularización de entropía
    'use_baseline': True,             # Si usar una función de valor de referencia para reducir varianza
    
    # Parámetros de entrenamiento
    'episodes': 1000,                 # Número de episodios para entrenamiento
    'max_steps': 1000,                # Máximo de pasos por episodio
    
    # Visualización y registro
    'log_interval': 10,               # Intervalo para mostrar métricas
    'smoothing_window': 10,            # Ventana para suavizado de gráficos

    'policy_lr': 1e-3,               # Tasa de aprendizaje para la política
    'value_lr': 1e-3,             # Tasa de aprendizaje para la función de valor
    'continuous': False,             # Si la acción es continua (True) o discreta (False)
}

SARSA_CONFIG = {
    # Parámetros de aprendizaje
    'learning_rate': 0.1,              # Tasa de aprendizaje (alpha) para actualización de valores Q
    'gamma': 0.99,                     # Factor de descuento para recompensas futuras
    'epsilon_start': 1.0,              # Valor inicial para política epsilon-greedy
    'epsilon_end': 0.01,               # Valor mínimo para epsilon
    'epsilon_decay': 0.995,            # Factor de decaimiento para epsilon
    'epsilon_decay_type': 'exponential', # Tipo de decaimiento: 'exponential' o 'linear'
    
    # Inicialización
    'optimistic_initialization': False, # Si usar inicialización optimista de valores Q
    'optimistic_initial_value': 0.0,    # Valor inicial para inicialización optimista
    
    # Discretización del espacio de estados
    'bins': 10,                        # Número de bins por dimensión para discretización
    'state_bounds': None,              # Límites para cada dimensión [(min1, max1), (min2, max2), ...]
    
    # Parámetros de entrenamiento
    'episodes': 1000,                  # Número de episodios de entrenamiento
    'max_steps': 500,                  # Máximo de pasos por episodio
    
    # Visualización y registro
    'log_interval': 50,                # Intervalo para mostrar métricas de entrenamiento
    'smoothing_window': 10             # Ventana para suavizado de gráficos
}

VALUE_ITERATION_CONFIG = {
    # Parámetros del algoritmo
    'gamma': 0.99,                # Factor de descuento para recompensas futuras
    'theta': 1e-6,                # Umbral de convergencia para detener iteraciones
    'max_iterations': 1000,       # Número máximo de iteraciones
    
    # Evaluación y visualización
    'eval_episodes': 10,          # Número de episodios para evaluación
    'visualization_interval': 5   # Intervalo de iteraciones para mostrar visualizaciones
}