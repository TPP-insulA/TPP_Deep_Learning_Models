import os, sys
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input, Dense, Dropout, BatchNormalization, 
    LayerNormalization, Concatenate, Activation, Add
)
from typing import Dict, Any, Tuple, Optional, List, Union

PROJECT_ROOT = os.path.abspath(os.getcwd())
sys.path.append(PROJECT_ROOT) 

from config.models_config import FNN_CONFIG

def get_activation(x: tf.Tensor, activation_name: str) -> tf.Tensor:
    """
    Aplica la función de activación según su nombre.
    
    Parámetros:
    -----------
    x : tf.Tensor
        Tensor al que aplicar la activación
    activation_name : str
        Nombre de la función de activación
        
    Retorna:
    --------
    tf.Tensor
        Tensor con la activación aplicada
    """
    if activation_name == 'relu':
        return tf.nn.relu(x)
    elif activation_name == 'gelu':
        return tf.nn.gelu(x)
    elif activation_name == 'swish':
        return tf.nn.swish(x)
    elif activation_name == 'silu':
        return tf.nn.silu(x)
    else:
        return tf.nn.relu(x)  # Valor por defecto

def create_residual_block(x: tf.Tensor, units: int, dropout_rate: float = 0.2, 
                         activation: str = 'relu', use_layer_norm: bool = True) -> tf.Tensor:
    """
    Crea un bloque residual para FNN con normalización y dropout.
    
    Parámetros:
    -----------
    x : tf.Tensor
        Tensor de entrada
    units : int
        Número de unidades en la capa densa
    dropout_rate : float
        Tasa de dropout a aplicar
    activation : str
        Función de activación a utilizar
    use_layer_norm : bool
        Si se debe usar normalización de capa en lugar de normalización por lotes
    
    Retorna:
    --------
    tf.Tensor
        Salida del bloque residual
    """
    # Guarda la entrada para la conexión residual
    skip = x
    
    # Primera capa densa con normalización y activación
    x = Dense(units)(x)
    if use_layer_norm:
        x = LayerNormalization(epsilon=FNN_CONFIG['epsilon'])(x)
    else:
        x = BatchNormalization()(x)
    x = Activation(activation)(x)
    x = Dropout(dropout_rate)(x)
    
    # Segunda capa densa con normalización
    x = Dense(units)(x)
    if use_layer_norm:
        x = LayerNormalization(epsilon=FNN_CONFIG['epsilon'])(x)
    else:
        x = BatchNormalization()(x)
        
    # Proyección para la conexión residual si es necesario
    if skip.shape[-1] != units:
        skip = Dense(units)(skip)
    
    # Conexión residual
    x = Add()([x, skip])
    x = Activation(activation)(x)
    x = Dropout(dropout_rate)(x)
    
    return x

def create_fnn_model(input_shape: tuple, 
                    other_features_shape: Optional[tuple] = None) -> Model:
    """
    Crea un modelo de red neuronal feedforward (FNN) con características modernas.
    
    Parámetros:
    -----------
    input_shape : tuple
        Forma del tensor de entrada principal
    other_features_shape : tuple, opcional
        Forma del tensor de características adicionales
    
    Retorna:
    --------
    Model
        Modelo FNN configurado
    """
    # Entrada principal
    main_input = Input(shape=input_shape)
    
    # Aplanar si es necesario (para entradas multidimensionales)
    if len(input_shape) > 1:
        x = tf.keras.layers.Flatten()(main_input)
    else:
        x = main_input
        
    # Entrada secundaria opcional
    if other_features_shape:
        other_input = Input(shape=other_features_shape)
        x = Concatenate()([x, other_input])
        inputs = [main_input, other_input]
    else:
        inputs = main_input
    
    # Capa inicial
    x = Dense(FNN_CONFIG['hidden_units'][0])(x)
    if FNN_CONFIG['use_layer_norm']:
        x = LayerNormalization(epsilon=FNN_CONFIG['epsilon'])(x)
    else:
        x = BatchNormalization()(x)
    x = Activation(FNN_CONFIG['activation'])(x)
    x = Dropout(FNN_CONFIG['dropout_rates'][0])(x)
    
    # Bloques residuales apilados
    for i, units in enumerate(FNN_CONFIG['hidden_units'][1:]):
        dropout_rate = FNN_CONFIG['dropout_rates'][min(i+1, len(FNN_CONFIG['dropout_rates'])-1)]
        x = create_residual_block(
            x, 
            units, 
            dropout_rate=dropout_rate,
            activation=FNN_CONFIG['activation'],
            use_layer_norm=FNN_CONFIG['use_layer_norm']
        )
    
    # Capas finales con estrechamiento progresivo
    for i, units in enumerate(FNN_CONFIG['final_units']):
        x = Dense(units)(x)
        x = Activation(FNN_CONFIG['activation'])(x)
        if FNN_CONFIG['use_layer_norm']:
            x = LayerNormalization(epsilon=FNN_CONFIG['epsilon'])(x)
        else:
            x = BatchNormalization()(x)
        x = Dropout(FNN_CONFIG['final_dropout_rate'])(x)
    
    # Capa de salida
    if FNN_CONFIG['regression']:
        output = Dense(1)(x)
    else:
        output = Dense(FNN_CONFIG['num_classes'], activation='softmax')(x)
    
    # Crear el modelo
    model = Model(inputs=inputs, outputs=output)
    
    return model

def compile_fnn_model(model: Model, 
                     loss: Optional[Union[str, tf.keras.losses.Loss]] = None, 
                     metrics: Optional[List[Union[str, tf.keras.metrics.Metric]]] = None) -> Model:
    """
    Compila un modelo FNN con la configuración especificada.
    
    Parámetros:
    -----------
    model : Model
        Modelo FNN a compilar
    loss : str o función de pérdida, opcional
        Función de pérdida a utilizar
    metrics : list, opcional
        Lista de métricas para evaluar el modelo
        
    Retorna:
    --------
    Model
        Modelo FNN compilado
    """
    # Valores por defecto para pérdida y métricas
    if loss is None:
        loss = 'mse' if FNN_CONFIG['regression'] else 'sparse_categorical_crossentropy'
    
    if metrics is None:
        metrics = ['mae'] if FNN_CONFIG['regression'] else ['accuracy']
    
    # Compilar el modelo
    model.compile(
        optimizer=tf.keras.optimizers.Adam(
            learning_rate=FNN_CONFIG['learning_rate'],
            beta_1=FNN_CONFIG['beta_1'],
            beta_2=FNN_CONFIG['beta_2'],
            epsilon=FNN_CONFIG['optimizer_epsilon']
        ),
        loss=loss,
        metrics=metrics
    )
    
    return model

def train_fnn_model(model: Model, 
                   x_train: Union[tf.Tensor, Tuple[tf.Tensor, ...], List[tf.Tensor]], 
                   y_train: tf.Tensor, 
                   x_val: Optional[Union[tf.Tensor, Tuple[tf.Tensor, ...], List[tf.Tensor]]] = None, 
                   y_val: Optional[tf.Tensor] = None, 
                   batch_size: Optional[int] = None, 
                   epochs: Optional[int] = None) -> tf.keras.callbacks.History:
    """
    Entrena un modelo FNN con los datos proporcionados.
    
    Parámetros:
    -----------
    model : Model
        Modelo FNN a entrenar
    x_train : array o tuple
        Datos de entrenamiento de entrada
    y_train : array
        Etiquetas de entrenamiento
    x_val : array o tuple, opcional
        Datos de validación de entrada
    y_val : array, opcional
        Etiquetas de validación
    batch_size : int, opcional
        Tamaño del lote para entrenamiento
    epochs : int, opcional
        Número de épocas para entrenamiento
        
    Retorna:
    --------
    History
        Historial del entrenamiento
    """
    # Valores por defecto para lotes y épocas
    if batch_size is None:
        batch_size = FNN_CONFIG['batch_size']
    
    if epochs is None:
        epochs = FNN_CONFIG['epochs']
    
    # Configurar datos de validación
    validation_data = None
    if x_val is not None and y_val is not None:
        validation_data = (x_val, y_val)
    
    # Configurar callbacks
    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            patience=FNN_CONFIG['patience'],
            restore_best_weights=True
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            factor=FNN_CONFIG['lr_reduction_factor'],
            patience=FNN_CONFIG['lr_patience'],
            min_lr=FNN_CONFIG['min_learning_rate']
        )
    ]
    
    # Entrenar el modelo
    history = model.fit(
        x_train, 
        y_train,
        batch_size=batch_size,
        epochs=epochs,
        validation_data=validation_data,
        callbacks=callbacks,
        verbose=1
    )
    
    return history