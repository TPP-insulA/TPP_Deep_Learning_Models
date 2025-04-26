import os
import numpy as np
import pandas as pd
import polars as pl
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Optional, Any, Union, Callable
import seaborn as sns
from matplotlib.figure import Figure
from matplotlib.axes import Axes
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Constantes para etiquetas comunes
CONST_EPISODIO = "Episodio"
CONST_EPOCA = "Época"
CONST_ITERACION = "Iteración"
CONST_PERDIDA = "Pérdida"
CONST_RECOMPENSA = "Recompensa"
CONST_VALOR = "Valor"
CONST_ENTROPIA = "Entropía"
CONST_KL = "KL Divergence"
CONST_EPSILON = "Epsilon"
CONST_POLITICA = "Política"
CONST_SUAVIZADO = "Suavizado"
CONST_ORIGINAL = "Original"
CONST_PASOS = "Pasos"
CONST_VALIDACION = "Validación"
CONST_ENTRENAMIENTO = "Entrenamiento"
CONST_ENSEMBLE = "Ensemble"

# Constantes para nombres de métricas
CONST_METRIC_MAE = "mae"
CONST_METRIC_RMSE = "rmse" 
CONST_METRIC_R2 = "r2"

# Constantes para tipos de gráficos
CONST_TYPE_LOSS = "loss"
CONST_TYPE_REWARD = "reward"
CONST_TYPE_POLICY = "policy"
CONST_TYPE_VALUE = "value"
CONST_TYPE_METRICS = "metrics"
CONST_TYPE_COMPARISON = "comparison"


def smooth_curve(data: np.ndarray, window_size: int) -> np.ndarray:
    """
    Aplica suavizado usando una media móvil.
    
    Parámetros:
    -----------
    data : np.ndarray
        Datos a suavizar
    window_size : int
        Tamaño de la ventana de suavizado
        
    Retorna:
    --------
    np.ndarray
        Datos suavizados
    """
    if len(data) < window_size:
        return data
    
    kernel = np.ones(window_size) / window_size
    return np.convolve(data, kernel, mode='valid')


def create_figure(nrows: int = 1, ncols: int = 1, figsize: Tuple[int, int] = None) -> Tuple[Figure, Union[Axes, np.ndarray]]:
    """
    Crea una figura de matplotlib con configuración estándar.
    
    Parámetros:
    -----------
    nrows : int, opcional
        Número de filas (default: 1)
    ncols : int, opcional
        Número de columnas (default: 1)
    figsize : Tuple[int, int], opcional
        Tamaño de la figura (default: None, se calcula automáticamente)
        
    Retorna:
    --------
    Tuple[Figure, Union[Axes, np.ndarray]]
        Figura y ejes para graficar
    """
    if figsize is None:
        figsize = (ncols * 6, nrows * 4)
        
    return plt.subplots(nrows, ncols, figsize=figsize)


def save_figure(filepath: str, fig: Figure = None, dpi: int = 300) -> None:
    """
    Guarda una figura en un archivo.
    
    Parámetros:
    -----------
    filepath : str
        Ruta donde guardar la figura
    fig : Figure, opcional
        Figura a guardar, si es None usa la figura actual (default: None)
    dpi : int, opcional
        Resolución en puntos por pulgada (default: 300)
    
    Retorna:
    --------
    None
    """
    # Crear directorio si no existe
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    
    if fig is None:
        plt.savefig(filepath, dpi=dpi, bbox_inches='tight')
    else:
        fig.savefig(filepath, dpi=dpi, bbox_inches='tight')


def plot_training_history(history: Dict[str, List[float]], 
                         save_path: Optional[str] = None, 
                         window_size: int = 10,
                         show_plot: bool = True) -> Tuple[Figure, np.ndarray]:
    """
    Visualiza el historial de entrenamiento con múltiples métricas.
    
    Parámetros:
    -----------
    history : Dict[str, List[float]]
        Diccionario con historial de entrenamiento
    save_path : Optional[str], opcional
        Ruta para guardar la figura (default: None)
    window_size : int, opcional
        Tamaño de ventana para suavizado (default: 10)
    show_plot : bool, opcional
        Indica si mostrar el gráfico (default: True)
        
    Retorna:
    --------
    Tuple[Figure, np.ndarray]
        Figura y arreglo de ejes
    """
    # Determinar qué métricas están disponibles
    has_loss = "loss" in history
    has_val_loss = "val_loss" in history
    has_mae = CONST_METRIC_MAE in history
    has_val_mae = f"val_{CONST_METRIC_MAE}" in history
    has_rmse = CONST_METRIC_RMSE in history
    has_val_rmse = f"val_{CONST_METRIC_RMSE}" in history
    
    # Determinar número de subgráficos
    n_plots = sum([has_loss, has_mae, has_rmse])
    
    # Crear figura
    fig, axs = create_figure(nrows=n_plots, ncols=1)
    
    # Convertir a matriz de ejes si solo hay uno
    if n_plots == 1:
        axs = np.array([axs])
    
    plot_idx = 0
    
    # Graficar pérdida
    if has_loss:
        axs[plot_idx].plot(history["loss"], label=CONST_ENTRENAMIENTO, color='blue', alpha=0.5)
        if has_val_loss:
            axs[plot_idx].plot(history["val_loss"], label=CONST_VALIDACION, color='red', alpha=0.5)
        
        axs[plot_idx].set_title(f'{CONST_PERDIDA} durante entrenamiento')
        axs[plot_idx].set_xlabel(CONST_EPOCA)
        axs[plot_idx].set_ylabel(CONST_PERDIDA)
        axs[plot_idx].legend()
        axs[plot_idx].grid(alpha=0.3)
        plot_idx += 1
    
    # Graficar MAE
    if has_mae:
        axs[plot_idx].plot(history[CONST_METRIC_MAE], label=CONST_ENTRENAMIENTO, color='blue', alpha=0.5)
        if has_val_mae:
            axs[plot_idx].plot(history[f"val_{CONST_METRIC_MAE}"], label=CONST_VALIDACION, color='red', alpha=0.5)
        
        axs[plot_idx].set_title('Error Absoluto Medio durante entrenamiento')
        axs[plot_idx].set_xlabel(CONST_EPOCA)
        axs[plot_idx].set_ylabel('MAE')
        axs[plot_idx].legend()
        axs[plot_idx].grid(alpha=0.3)
        plot_idx += 1
    
    # Graficar RMSE
    if has_rmse:
        axs[plot_idx].plot(history[CONST_METRIC_RMSE], label=CONST_ENTRENAMIENTO, color='blue', alpha=0.5)
        if has_val_rmse:
            axs[plot_idx].plot(history[f"val_{CONST_METRIC_RMSE}"], label=CONST_VALIDACION, color='red', alpha=0.5)
        
        axs[plot_idx].set_title('Raíz del Error Cuadrático Medio durante entrenamiento')
        axs[plot_idx].set_xlabel(CONST_EPOCA)
        axs[plot_idx].set_ylabel('RMSE')
        axs[plot_idx].legend()
        axs[plot_idx].grid(alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        save_figure(save_path, fig)
    
    if show_plot:
        plt.show()
    else:
        plt.close(fig)
    
    return fig, axs


def _get_available_metrics(history: Dict[str, List[float]]) -> List[str]:
    """
    Determina qué métricas están disponibles en el historial de entrenamiento.
    
    Parámetros:
    -----------
    history : Dict[str, List[float]]
        Diccionario con historial de entrenamiento de RL
        
    Retorna:
    --------
    List[str]
        Lista con los tipos de métricas disponibles
    """
    metrics = []
    if "episode_rewards" in history or "reward" in history:
        metrics.append("rewards")
    if "episode_lengths" in history or "length" in history:
        metrics.append("lengths")
    if "epsilons" in history or "epsilon_history" in history:
        metrics.append("epsilons")
    if "losses" in history or "loss" in history:
        metrics.append("losses")
    if "value_losses" in history:
        metrics.append("value_losses")
    if "policy_losses" in history:
        metrics.append("policy_losses")
    
    return metrics


def _plot_metric_with_smoothing(ax: Axes, data: List[float], window_size: int, 
                               color: str, title: str, xlabel: str, ylabel: str) -> None:
    """
    Grafica una métrica con su versión suavizada.
    
    Parámetros:
    -----------
    ax : Axes
        Eje donde graficar
    data : List[float]
        Datos a graficar
    window_size : int
        Tamaño de ventana para suavizado
    color : str
        Color para las líneas
    title : str
        Título del gráfico
    xlabel : str
        Etiqueta para eje x
    ylabel : str
        Etiqueta para eje y
    """
    ax.plot(data, alpha=0.3, color=color, label=CONST_ORIGINAL)
    
    if len(data) > window_size:
        smoothed_data = smooth_curve(data, window_size)
        ax.plot(
            range(window_size-1, len(data)),
            smoothed_data,
            color=color,
            label=f'{CONST_SUAVIZADO} (ventana={window_size})'
        )
    
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.legend()
    ax.grid(alpha=0.3)


def _plot_simple_metric(ax: Axes, data: List[float], color: str, 
                       title: str, xlabel: str, ylabel: str) -> None:
    """
    Grafica una métrica simple sin suavizado.
    
    Parámetros:
    -----------
    ax : Axes
        Eje donde graficar
    data : List[float]
        Datos a graficar
    color : str
        Color para la línea
    title : str
        Título del gráfico
    xlabel : str
        Etiqueta para eje x
    ylabel : str
        Etiqueta para eje y
    """
    ax.plot(data, color=color)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(alpha=0.3)


def _get_metric_keys(history: Dict[str, List[float]]) -> Dict[str, str]:
    """
    Obtiene las claves correctas para cada métrica en el historial.
    
    Parámetros:
    -----------
    history : Dict[str, List[float]]
        Diccionario con historial de entrenamiento de RL
        
    Retorna:
    --------
    Dict[str, str]
        Diccionario con mapeo de tipos de métricas a sus claves en el historial
    """
    return {
        "reward": "episode_rewards" if "episode_rewards" in history else "reward",
        "length": "episode_lengths" if "episode_lengths" in history else "length",
        "epsilon": "epsilons" if "epsilons" in history else "epsilon_history",
        "loss": "losses" if "losses" in history else "loss"
    }


def _plot_rl_metrics(axs_flat: np.ndarray, history: Dict[str, List[float]], 
                    metrics: List[str], metric_keys: Dict[str, str], 
                    window_size: int) -> None:
    """
    Grafica las métricas disponibles en el historial de RL.
    
    Parámetros:
    -----------
    axs_flat : np.ndarray
        Array plano de ejes para graficar
    history : Dict[str, List[float]]
        Diccionario con historial de entrenamiento
    metrics : List[str]
        Lista de métricas disponibles
    metric_keys : Dict[str, str]
        Mapeo de tipos de métricas a claves en el historial
    window_size : int
        Tamaño de ventana para suavizado
    """
    plot_idx = 0
    
    # Recompensa
    if "rewards" in metrics:
        _plot_metric_with_smoothing(
            axs_flat[plot_idx], 
            history[metric_keys["reward"]], 
            window_size, 
            'blue', 
            f'{CONST_RECOMPENSA} por {CONST_EPISODIO}', 
            CONST_EPISODIO, 
            CONST_RECOMPENSA
        )
        plot_idx += 1
    
    # Longitud de episodios
    if "lengths" in metrics:
        _plot_metric_with_smoothing(
            axs_flat[plot_idx], 
            history[metric_keys["length"]], 
            window_size, 
            'green', 
            f'Longitud de {CONST_EPISODIO}', 
            CONST_EPISODIO, 
            CONST_PASOS
        )
        plot_idx += 1
    
    # Epsilon (exploración)
    if "epsilons" in metrics:
        _plot_simple_metric(
            axs_flat[plot_idx], 
            history[metric_keys["epsilon"]], 
            'red', 
            f'{CONST_EPSILON} (Exploración)', 
            CONST_EPISODIO, 
            CONST_EPSILON
        )
        plot_idx += 1
    
    # Pérdida general
    if "losses" in metrics:
        _plot_metric_with_smoothing(
            axs_flat[plot_idx], 
            history[metric_keys["loss"]], 
            window_size, 
            'purple', 
            CONST_PERDIDA, 
            CONST_EPISODIO, 
            CONST_PERDIDA
        )
        plot_idx += 1
    
    # Pérdida de valor
    if "value_losses" in metrics:
        xlabel = CONST_ITERACION if "policy_losses" in history else CONST_EPISODIO
        _plot_metric_with_smoothing(
            axs_flat[plot_idx], 
            history["value_losses"], 
            window_size, 
            'orange', 
            f'{CONST_PERDIDA} de {CONST_VALOR}', 
            xlabel, 
            CONST_PERDIDA
        )
        plot_idx += 1
    
    # Pérdida de política
    if "policy_losses" in metrics:
        _plot_metric_with_smoothing(
            axs_flat[plot_idx], 
            history["policy_losses"], 
            window_size, 
            'brown', 
            f'{CONST_PERDIDA} de {CONST_POLITICA}', 
            CONST_ITERACION, 
            CONST_PERDIDA
        )


def plot_rl_training(history: Dict[str, List[float]], 
                    save_path: Optional[str] = None,
                    window_size: int = 10,
                    show_plot: bool = True) -> Tuple[Figure, np.ndarray]:
    """
    Visualiza el historial de entrenamiento específico para aprendizaje por refuerzo.
    
    Parámetros:
    -----------
    history : Dict[str, List[float]]
        Diccionario con historial de entrenamiento de RL
    save_path : Optional[str], opcional
        Ruta para guardar la figura (default: None)
    window_size : int, opcional
        Tamaño de ventana para suavizado (default: 10)
    show_plot : bool, opcional
        Indica si mostrar el gráfico (default: True)
        
    Retorna:
    --------
    Tuple[Figure, np.ndarray]
        Figura y arreglo de ejes
    """
    # Determinar qué métricas están disponibles
    metrics = _get_available_metrics(history)
    
    # Configurar disposición de gráficos
    n_plots = len(metrics)
    n_rows = (n_plots + 1) // 2  # Ceil division
    n_cols = min(2, n_plots)
    
    # Crear figura
    fig, axs = create_figure(nrows=n_rows, ncols=n_cols)
    
    # Convertir a matriz plana para facilitar indexación
    axs_flat = axs.flatten() if isinstance(axs, np.ndarray) else np.array([axs])
    
    # Obtener mapeo de claves de métricas
    metric_keys = _get_metric_keys(history)
    
    # Graficar métricas
    _plot_rl_metrics(axs_flat, history, metrics, metric_keys, window_size)
    
    plt.tight_layout()
    
    # Guardar y/o mostrar la figura
    if save_path:
        save_figure(save_path, fig)
    
    if show_plot:
        plt.show()
    else:
        plt.close(fig)
    
    return fig, axs


def plot_value_function(value_matrix: np.ndarray, 
                       show_values: bool = True, 
                       title: str = "Función de Valor", 
                       save_path: Optional[str] = None,
                       show_plot: bool = True) -> Tuple[Figure, Axes]:
    """
    Visualiza una función de valor como un mapa de calor.
    
    Parámetros:
    -----------
    value_matrix : np.ndarray
        Matriz con valores de la función
    show_values : bool, opcional
        Indica si mostrar los valores numéricos (default: True)
    title : str, opcional
        Título del gráfico (default: "Función de Valor")
    save_path : Optional[str], opcional
        Ruta para guardar la figura (default: None)
    show_plot : bool, opcional
        Indica si mostrar el gráfico (default: True)
        
    Retorna:
    --------
    Tuple[Figure, Axes]
        Figura y eje con el gráfico
    """
    fig, ax = create_figure()
    
    # Crear mapa de calor
    im = ax.imshow(value_matrix, cmap='viridis')
    
    # Agregar barra de color
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label(CONST_VALOR)
    
    # Mostrar valores numéricos si se solicita
    if show_values:
        rows, cols = value_matrix.shape
        for i in range(rows):
            for j in range(cols):
                value = value_matrix[i, j]
                ax.text(j, i, f"{value:.2f}", ha="center", va="center",
                      color="white" if value < np.max(value_matrix)/1.5 else "black")
    
    ax.set_title(title)
    ax.set_xlabel("Columna")
    ax.set_ylabel("Fila")
    
    if save_path:
        save_figure(save_path, fig)
    
    if show_plot:
        plt.show()
    else:
        plt.close(fig)
    
    return fig, ax


def plot_policy(policy_matrix: np.ndarray, 
               action_mapping: Dict[int, str] = None, 
               title: str = "Política", 
               save_path: Optional[str] = None,
               show_plot: bool = True) -> Tuple[Figure, Axes]:
    """
    Visualiza una política como una cuadrícula con flechas o símbolos.
    
    Parámetros:
    -----------
    policy_matrix : np.ndarray
        Matriz con índices de acciones de la política
    action_mapping : Dict[int, str], opcional
        Diccionario que mapea índices de acción a símbolos (default: None)
    title : str, opcional
        Título del gráfico (default: "Política")
    save_path : Optional[str], opcional
        Ruta para guardar la figura (default: None)
    show_plot : bool, opcional
        Indica si mostrar el gráfico (default: True)
        
    Retorna:
    --------
    Tuple[Figure, Axes]
        Figura y eje con el gráfico
    """
    fig, ax = create_figure(figsize=(10, 8))
    
    rows, cols = policy_matrix.shape
    
    # Configurar límites de la cuadrícula
    ax.set_xlim([0, cols])
    ax.set_ylim([0, rows])
    ax.invert_yaxis()  # Origen en esquina superior izquierda
    
    # Dibujar líneas de cuadrícula
    for i in range(rows + 1):
        ax.axhline(i, color='black', alpha=0.3)
    for j in range(cols + 1):
        ax.axvline(j, color='black', alpha=0.3)
    
    # Crear mapeo predeterminado si no se proporciona
    if action_mapping is None:
        action_mapping = {
            0: "←",  # izquierda
            1: "→",  # derecha
            2: "↑",  # arriba
            3: "↓",  # abajo
        }
    
    # Dibujar símbolos o flechas para acciones
    for i in range(rows):
        for j in range(cols):
            action = int(policy_matrix[i, j])
            if action in action_mapping:
                symbol = action_mapping[action]
                ax.text(j + 0.5, i + 0.5, symbol, ha='center', va='center', 
                       fontsize=15, fontweight='bold', color='blue')
    
    ax.set_title(title)
    ax.set_xticks(np.arange(0.5, cols, 1))
    ax.set_yticks(np.arange(0.5, rows, 1))
    ax.set_xticklabels(range(cols))
    ax.set_yticklabels(range(rows))
    ax.grid(False)
    
    if save_path:
        save_figure(save_path, fig)
    
    if show_plot:
        plt.show()
    else:
        plt.close(fig)
    
    return fig, ax


def plot_policy_arrows(policy_matrix: np.ndarray, 
                      title: str = "Política con Flechas", 
                      save_path: Optional[str] = None,
                      show_plot: bool = True) -> Tuple[Figure, Axes]:
    """
    Visualiza una política usando flechas direccionales.
    
    Parámetros:
    -----------
    policy_matrix : np.ndarray
        Matriz con índices de acciones de la política
    title : str, opcional
        Título del gráfico (default: "Política con Flechas")
    save_path : Optional[str], opcional
        Ruta para guardar la figura (default: None)
    show_plot : bool, opcional
        Indica si mostrar el gráfico (default: True)
        
    Retorna:
    --------
    Tuple[Figure, Axes]
        Figura y eje con el gráfico
    """
    fig, ax = create_figure(figsize=(10, 8))
    
    rows, cols = policy_matrix.shape
    
    # Configurar límites de la cuadrícula
    ax.set_xlim([0, cols])
    ax.set_ylim([0, rows])
    ax.invert_yaxis()  # Origen en esquina superior izquierda
    
    # Dibujar líneas de cuadrícula
    for i in range(rows + 1):
        ax.axhline(i, color='black', alpha=0.3)
    for j in range(cols + 1):
        ax.axvline(j, color='black', alpha=0.3)
    
    # Mapeo de acciones a direcciones de flechas
    directions = {
        0: (-0.4, 0),   # izquierda
        1: (0.4, 0),    # derecha
        2: (0, -0.4),   # arriba
        3: (0, 0.4),    # abajo
    }
    
    # Dibujar flechas para acciones
    for i in range(rows):
        for j in range(cols):
            action = int(policy_matrix[i, j])
            if action in directions:
                dx, dy = directions[action]
                ax.arrow(j + 0.5, i + 0.5, dx, dy, head_width=0.1, head_length=0.1, 
                        fc='blue', ec='blue')
    
    ax.set_title(title)
    ax.set_xticks(np.arange(0.5, cols, 1))
    ax.set_yticks(np.arange(0.5, rows, 1))
    ax.set_xticklabels(range(cols))
    ax.set_yticklabels(range(rows))
    ax.grid(False)
    
    if save_path:
        save_figure(save_path, fig)
    
    if show_plot:
        plt.show()
    else:
        plt.close(fig)
    
    return fig, ax


def plot_continuous_value_function(x_range: np.ndarray, 
                                  y_range: np.ndarray, 
                                  values: np.ndarray, 
                                  title: str = "Función de Valor Continua", 
                                  xlabel: str = "Dimensión 1", 
                                  ylabel: str = "Dimensión 2",
                                  save_path: Optional[str] = None,
                                  show_plot: bool = True) -> Tuple[Figure, Axes]:
    """
    Visualiza una función de valor para espacios de estado continuos.
    
    Parámetros:
    -----------
    x_range : np.ndarray
        Valores del eje x
    y_range : np.ndarray
        Valores del eje y
    values : np.ndarray
        Matriz de valores
    title : str, opcional
        Título del gráfico (default: "Función de Valor Continua")
    xlabel : str, opcional
        Etiqueta del eje x (default: "Dimensión 1")
    ylabel : str, opcional
        Etiqueta del eje y (default: "Dimensión 2")
    save_path : Optional[str], opcional
        Ruta para guardar la figura (default: None)
    show_plot : bool, opcional
        Indica si mostrar el gráfico (default: True)
        
    Retorna:
    --------
    Tuple[Figure, Axes]
        Figura y eje con el gráfico
    """
    fig, ax = create_figure()
    
    # Crear mallado
    X, Y = np.meshgrid(x_range, y_range)
    
    # Crear mapa de calor
    value_plot = ax.pcolormesh(X, Y, values, cmap='plasma', shading='auto')
    
    # Agregar barra de color
    plt.colorbar(value_plot, label=CONST_VALOR)
    
    # Configurar etiquetas
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(alpha=0.2)
    
    if save_path:
        save_figure(save_path, fig)
    
    if show_plot:
        plt.show()
    else:
        plt.close(fig)
    
    return fig, ax


def plot_continuous_policy(x_range: np.ndarray, 
                          y_range: np.ndarray, 
                          policy: np.ndarray, 
                          title: str = "Política Continua", 
                          xlabel: str = "Dimensión 1", 
                          ylabel: str = "Dimensión 2",
                          save_path: Optional[str] = None,
                          show_plot: bool = True) -> Tuple[Figure, Axes]:
    """
    Visualiza una política para espacios de estado continuos.
    
    Parámetros:
    -----------
    x_range : np.ndarray
        Valores del eje x
    y_range : np.ndarray
        Valores del eje y
    policy : np.ndarray
        Matriz de índices de acciones
    title : str, opcional
        Título del gráfico (default: "Política Continua")
    xlabel : str, opcional
        Etiqueta del eje x (default: "Dimensión 1")
    ylabel : str, opcional
        Etiqueta del eje y (default: "Dimensión 2")
    save_path : Optional[str], opcional
        Ruta para guardar la figura (default: None)
    show_plot : bool, opcional
        Indica si mostrar el gráfico (default: True)
        
    Retorna:
    --------
    Tuple[Figure, Axes]
        Figura y eje con el gráfico
    """
    fig, ax = create_figure()
    
    # Crear mallado
    X, Y = np.meshgrid(x_range, y_range)
    
    # Crear mapa de calor para política
    policy_plot = ax.pcolormesh(X, Y, policy, cmap='viridis', shading='auto')
    
    # Agregar barra de color
    plt.colorbar(policy_plot, label='Acción')
    
    # Configurar etiquetas
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(alpha=0.2)
    
    if save_path:
        save_figure(save_path, fig)
    
    if show_plot:
        plt.show()
    else:
        plt.close(fig)
    
    return fig, ax


def plot_model_comparison(metrics: Dict[str, Dict[str, float]], 
                         metric_name: str = CONST_METRIC_MAE,
                         title: str = "Comparación de Modelos", 
                         save_path: Optional[str] = None,
                         show_plot: bool = True) -> Tuple[Figure, Axes]:
    """
    Compara métricas entre diferentes modelos.
    
    Parámetros:
    -----------
    metrics : Dict[str, Dict[str, float]]
        Diccionario con nombre de modelos y sus métricas
    metric_name : str, opcional
        Nombre de la métrica a comparar (default: "mae")
    title : str, opcional
        Título del gráfico (default: "Comparación de Modelos")
    save_path : Optional[str], opcional
        Ruta para guardar la figura (default: None)
    show_plot : bool, opcional
        Indica si mostrar el gráfico (default: True)
        
    Retorna:
    --------
    Tuple[Figure, Axes]
        Figura y eje con el gráfico
    """
    fig, ax = create_figure()
    
    # Extraer nombres de modelos y valores de métrica
    models = list(metrics.keys())
    values = [metrics[model][metric_name] for model in models]
    
    # Determinar colores (resaltar el mejor modelo)
    if metric_name in [CONST_METRIC_MAE, CONST_METRIC_RMSE]:  # Métricas donde menor es mejor
        best_idx = np.argmin(values)
    else:  # Métricas donde mayor es mejor (como R²)
        best_idx = np.argmax(values)
    
    colors = ['blue'] * len(models)
    colors[best_idx] = 'red'  # Color diferente para el mejor modelo
    
    # Crear gráfico de barras
    ax.bar(models, values, color=colors)
    
    # Agregar etiquetas con valores
    for i, v in enumerate(values):
        ax.text(i, v + max(values) * 0.01, f"{v:.4f}", ha='center', color='black')
    
    # Agregar etiquetas y título
    ax.set_title(f"{title} - {metric_name.upper()}")
    ax.set_ylabel(metric_name.upper())
    ax.grid(axis='y', alpha=0.3)
    
    # Rotar etiquetas si hay muchos modelos
    if len(models) > 3:
        plt.xticks(rotation=45, ha='right')
    
    plt.tight_layout()
    
    if save_path:
        save_figure(save_path, fig)
    
    if show_plot:
        plt.show()
    else:
        plt.close(fig)
    
    return fig, ax


def plot_predictions_vs_actual(y_true: np.ndarray, 
                              predictions_dict: Dict[str, np.ndarray], 
                              sample_size: int = 100,
                              title: str = "Predicciones vs Valores Reales",
                              save_path: Optional[str] = None,
                              show_plot: bool = True) -> Tuple[Figure, Axes]:
    """
    Visualiza predicciones de diferentes modelos comparadas con valores reales.
    
    Parámetros:
    -----------
    y_true : np.ndarray
        Valores reales
    predictions_dict : Dict[str, np.ndarray]
        Diccionario con nombre de modelos y sus predicciones
    sample_size : int, opcional
        Número de muestras a visualizar (default: 100)
    title : str, opcional
        Título del gráfico (default: "Predicciones vs Valores Reales")
    save_path : Optional[str], opcional
        Ruta para guardar la figura (default: None)
    show_plot : bool, opcional
        Indica si mostrar el gráfico (default: True)
        
    Retorna:
    --------
    Tuple[Figure, Axes]
        Figura y eje con el gráfico
    """
    fig, ax = create_figure(figsize=(14, 7))
    
    # Limitar muestra por claridad visual
    n_samples = min(sample_size, len(y_true))
    indices = np.arange(n_samples)
    
    # Graficar valores reales
    ax.plot(indices, y_true[:n_samples], 'o-', label='Real', color='black', alpha=0.7, markersize=4)
    
    # Graficar cada modelo
    for model_name, pred in predictions_dict.items():
        ax.plot(indices, pred[:n_samples], 'o-', label=model_name, alpha=0.5, markersize=3)
    
    # Configurar etiquetas y leyenda
    ax.set_title(title)
    ax.set_xlabel('Índice de Muestra')
    ax.set_ylabel('Valor')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        save_figure(save_path, fig)
    
    if show_plot:
        plt.show()
    else:
        plt.close(fig)
    
    return fig, ax


def plot_all_metrics(metrics: Dict[str, Dict[str, float]], 
                    save_path: Optional[str] = None,
                    show_plot: bool = True) -> Tuple[Figure, np.ndarray]:
    """
    Visualiza todas las métricas disponibles para comparación de modelos.
    
    Parámetros:
    -----------
    metrics : Dict[str, Dict[str, float]]
        Diccionario con métricas por modelo
    save_path : Optional[str], opcional
        Ruta para guardar la figura (default: None)
    show_plot : bool, opcional
        Indica si mostrar el gráfico (default: True)
        
    Retorna:
    --------
    Tuple[Figure, np.ndarray]
        Figura y arreglo de ejes
    """
    # Determinar métricas disponibles
    all_metrics = set()
    for model_metrics in metrics.values():
        all_metrics.update(model_metrics.keys())
    
    n_metrics = len(all_metrics)
    cols = min(3, n_metrics)
    rows = (n_metrics + cols - 1) // cols  # Ceil division
    
    fig, axs = create_figure(nrows=rows, ncols=cols, figsize=(5*cols, 4*rows))
    
    # Convertir a matriz plana
    axs_flat = axs.flatten() if isinstance(axs, np.ndarray) else np.array([axs])
    
    # Nombres de modelos
    models = list(metrics.keys())
    
    # Crear un gráfico para cada métrica
    for i, metric in enumerate(sorted(all_metrics)):
        # Extraer valores para esta métrica
        values = [metrics[model].get(metric, 0) for model in models]
        
        # Determinar colores
        if metric.lower() in [CONST_METRIC_MAE.lower(), CONST_METRIC_RMSE.lower()]:
            best_idx = np.argmin(values)
        else:
            best_idx = np.argmax(values)
        
        colors = ['blue'] * len(models)
        colors[best_idx] = 'red'
        
        # Crear gráfico de barras
        axs_flat[i].bar(models, values, color=colors)
        
        # Agregar etiquetas con valores
        for j, v in enumerate(values):
            axs_flat[i].text(j, v + max(values) * 0.01, f"{v:.4f}", ha='center', color='black',
                           fontsize=8)
        
        # Agregar título y etiquetas
        axs_flat[i].set_title(f"{metric.upper()}")
        axs_flat[i].set_ylabel(metric.upper())
        axs_flat[i].grid(axis='y', alpha=0.3)
        
        # Rotar etiquetas si hay muchos modelos
        if len(models) > 3:
            plt.setp(axs_flat[i].get_xticklabels(), rotation=45, ha='right')
    
    # Ocultar ejes no utilizados
    for i in range(n_metrics, len(axs_flat)):
        axs_flat[i].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        save_figure(save_path, fig)
    
    if show_plot:
        plt.show()
    else:
        plt.close(fig)
    
    return fig, axs


def plot_ensemble_weights(weights: np.ndarray, 
                         model_names: List[str], 
                         title: str = "Pesos de Ensemble",
                         save_path: Optional[str] = None,
                         show_plot: bool = True) -> Tuple[Figure, Axes]:
    """
    Visualiza los pesos asignados a cada modelo en un ensemble.
    
    Parámetros:
    -----------
    weights : np.ndarray
        Pesos asignados a cada modelo
    model_names : List[str]
        Nombres de los modelos
    title : str, opcional
        Título del gráfico (default: "Pesos de Ensemble")
    save_path : Optional[str], opcional
        Ruta para guardar la figura (default: None)
    show_plot : bool, opcional
        Indica si mostrar el gráfico (default: True)
        
    Retorna:
    --------
    Tuple[Figure, Axes]
        Figura y eje con el gráfico
    """
    fig, ax = create_figure(figsize=(10, 6))
    
    # Crear gráfico de barras
    ax.bar(model_names, weights, color='teal')
    
    # Agregar etiquetas con valores
    for i, v in enumerate(weights):
        ax.text(i, v + 0.01, f"{v:.4f}", ha='center', color='black')
    
    # Configurar etiquetas y título
    ax.set_title(title)
    ax.set_ylabel('Peso')
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Rotar etiquetas si hay muchos modelos
    if len(model_names) > 3:
        plt.xticks(rotation=45, ha='right')
    
    plt.tight_layout()
    
    if save_path:
        save_figure(save_path, fig)
    
    if show_plot:
        plt.show()
    else:
        plt.close(fig)
    
    return fig, ax


def visualize_model_results(history: Dict[str, List[float]], 
                           predictions: np.ndarray, 
                           y_test: np.ndarray,
                           model_name: str,
                           save_dir: str,
                           show_plots: bool = False) -> Dict[str, str]:
    """
    Visualiza y guarda todos los resultados relevantes de un modelo.
    
    Parámetros:
    -----------
    history : Dict[str, List[float]]
        Historial de entrenamiento
    predictions : np.ndarray
        Predicciones del modelo
    y_test : np.ndarray
        Valores reales
    model_name : str
        Nombre del modelo
    save_dir : str
        Directorio donde guardar las visualizaciones
    show_plots : bool, opcional
        Indica si mostrar los gráficos (default: False)
        
    Retorna:
    --------
    Dict[str, str]
        Diccionario con rutas a las figuras generadas
    """
    # Crear directorio para este modelo
    model_dir = os.path.join(save_dir, model_name)
    os.makedirs(model_dir, exist_ok=True)
    
    # Rutas para guardar figuras
    figures = {}
    
    # Visualizar historial de entrenamiento
    if any(k in history for k in ['loss', CONST_METRIC_MAE, CONST_METRIC_RMSE]):
        train_hist_path = os.path.join(model_dir, 'training_history.png')
        plot_training_history(history, save_path=train_hist_path, show_plot=show_plots)
        figures['training_history'] = train_hist_path
    
    # Visualizar historial de RL si aplica
    if any(k in history for k in ['episode_rewards', 'reward', 'epsilons', 'epsilon_history']):
        rl_hist_path = os.path.join(model_dir, 'rl_training.png')
        plot_rl_training(history, save_path=rl_hist_path, show_plot=show_plots)
        figures['rl_training'] = rl_hist_path
    
    # Visualizar predicciones vs real
    preds_path = os.path.join(model_dir, 'predictions.png')
    plot_predictions_vs_actual(
        y_test, 
        {model_name: predictions}, 
        save_path=preds_path,
        show_plot=show_plots
    )
    figures['predictions'] = preds_path
    
    # Visualizar métricas
    mae = float(mean_absolute_error(y_test, predictions))
    rmse = float(np.sqrt(mean_squared_error(y_test, predictions)))
    r2 = float(r2_score(y_test, predictions))
    
    metrics = {model_name: {CONST_METRIC_MAE: mae, CONST_METRIC_RMSE: rmse, CONST_METRIC_R2: r2}}
    
    metrics_path = os.path.join(model_dir, 'metrics.png')
    plot_all_metrics(metrics, save_path=metrics_path, show_plot=show_plots)
    figures['metrics'] = metrics_path
    
    return figures


# Funciones específicas para visualización de análisis exploratorio de datos (EDA)

def plot_distribution(data: np.ndarray, 
                     title: str = "Distribución", 
                     xlabel: str = "Valor", 
                     save_path: Optional[str] = None,
                     show_plot: bool = True) -> Tuple[Figure, Axes]:
    """
    Visualiza la distribución de una variable.
    
    Parámetros:
    -----------
    data : np.ndarray
        Datos a visualizar
    title : str, opcional
        Título del gráfico (default: "Distribución")
    xlabel : str, opcional
        Etiqueta del eje x (default: "Valor")
    save_path : Optional[str], opcional
        Ruta para guardar la figura (default: None)
    show_plot : bool, opcional
        Indica si mostrar el gráfico (default: True)
        
    Retorna:
    --------
    Tuple[Figure, Axes]
        Figura y eje con el gráfico
    """
    fig, ax = create_figure()
    
    # Graficar histograma y KDE
    sns.histplot(data, kde=True, ax=ax)
    
    # Configurar etiquetas
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Frecuencia")
    
    if save_path:
        save_figure(save_path, fig)
    
    if show_plot:
        plt.show()
    else:
        plt.close(fig)
    
    return fig, ax


def plot_cgm_timeseries(cgm_data: np.ndarray, 
                       window_hours: int = 24, 
                       sample_freq_mins: int = 5, 
                       title: str = "Series Temporales CGM",
                       sample_limit: int = 10,
                       save_path: Optional[str] = None,
                       show_plot: bool = True) -> Tuple[Figure, Axes]:
    """
    Visualiza múltiples series temporales de CGM, cada una representando una ventana de tiempo.
    
    Parámetros:
    -----------
    cgm_data : np.ndarray
        Array 3D con forma (muestras, pasos_tiempo, características)
    window_hours : int, opcional
        Horas en cada ventana (default: 24)
    sample_freq_mins : int, opcional
        Minutos entre muestras (default: 5)
    title : str, opcional
        Título del gráfico (default: "Series Temporales CGM")
    sample_limit : int, opcional
        Límite de muestras a visualizar (default: 10)
    save_path : Optional[str], opcional
        Ruta para guardar la figura (default: None)
    show_plot : bool, opcional
        Indica si mostrar el gráfico (default: True)
        
    Retorna:
    --------
    Tuple[Figure, Axes]
        Figura y eje con el gráfico
    """
    fig, ax = create_figure(figsize=(12, 6))
    
    # Crear vector de tiempo (horas)
    steps_per_hour = 60 // sample_freq_mins
    time_hours = np.arange(0, window_hours, 1/steps_per_hour)
    
    # Extraer y graficar muestras
    n_samples = min(sample_limit, cgm_data.shape[0])
    
    # Seleccionar muestras uniformemente distribuidas
    sample_indices = np.linspace(0, cgm_data.shape[0] - 1, n_samples, dtype=int)
    
    for i in sample_indices:
        # Extraer datos CGM para esta muestra y aplanarlos si es necesario
        if cgm_data.ndim == 3:
            sample_cgm = cgm_data[i, :, 0]
        else:
            sample_cgm = cgm_data[i, :]
            
        ax.plot(time_hours[:len(sample_cgm)], sample_cgm, alpha=0.7)
    
    # Configurar etiquetas
    ax.set_title(title)
    ax.set_xlabel("Tiempo (horas)")
    ax.set_ylabel("Nivel de Glucosa (mg/dL)")
    ax.grid(True, alpha=0.3)
    
    # Agregar líneas de referencia para rangos normales
    ax.axhline(y=70, color='r', linestyle='-', alpha=0.4, label="Hipoglucemia")
    ax.axhline(y=180, color='orange', linestyle='-', alpha=0.4, label="Hiperglucemia")
    ax.axhspan(70, 180, alpha=0.1, color='green', label="Rango Normal")
    
    ax.legend()
    
    if save_path:
        save_figure(save_path, fig)
    
    if show_plot:
        plt.show()
    else:
        plt.close(fig)
    
    return fig, ax


def plot_correlation_matrix(df: Union[pd.DataFrame, pl.DataFrame], 
                           title: str = "Matriz de Correlación", 
                           save_path: Optional[str] = None,
                           show_plot: bool = True) -> Tuple[Figure, Axes]:
    """
    Visualiza una matriz de correlación entre variables.
    
    Parámetros:
    -----------
    df : Union[pd.DataFrame, pl.DataFrame]
        DataFrame con las variables a correlacionar (pandas o polars)
    title : str, opcional
        Título del gráfico (default: "Matriz de Correlación")
    save_path : Optional[str], opcional
        Ruta para guardar la figura (default: None)
    show_plot : bool, opcional
        Indica si mostrar el gráfico (default: True)
        
    Retorna:
    --------
    Tuple[Figure, Axes]
        Figura y eje con el gráfico
    """
    # Verificar si es DataFrame de polars y convertirlo a pandas si es necesario
    if isinstance(df, pl.DataFrame):
        corr_matrix = df.corr().to_pandas()
    else:
        corr_matrix = df.corr()
    
    # Determinar tamaño de figura en base a número de variables
    n_vars = len(corr_matrix)
    figsize = (max(8, n_vars*0.7), max(6, n_vars*0.7))
    
    fig, ax = create_figure(figsize=figsize)
    
    # Crear mapa de calor
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
    cmap = sns.diverging_palette(230, 20, as_cmap=True)
    
    sns.heatmap(corr_matrix, mask=mask, cmap=cmap, vmax=1, vmin=-1, center=0,
               annot=True, fmt=".2f", square=True, linewidths=.5, ax=ax)
    
    ax.set_title(title)
    
    plt.tight_layout()
    
    if save_path:
        save_figure(save_path, fig)
    
    if show_plot:
        plt.show()
    else:
        plt.close(fig)
    
    return fig, ax


def plot_feature_importance(feature_names: List[str], 
                           importances: np.ndarray, 
                           title: str = "Importancia de Características", 
                           save_path: Optional[str] = None,
                           show_plot: bool = True) -> Tuple[Figure, Axes]:
    """
    Visualiza la importancia de características para un modelo.
    
    Parámetros:
    -----------
    feature_names : List[str]
        Lista con nombres de características
    importances : np.ndarray
        Array con valores de importancia
    title : str, opcional
        Título del gráfico (default: "Importancia de Características")
    save_path : Optional[str], opcional
        Ruta para guardar la figura (default: None)
    show_plot : bool, opcional
        Indica si mostrar el gráfico (default: True)
        
    Retorna:
    --------
    Tuple[Figure, Axes]
        Figura y eje con el gráfico
    """
    # Ordenar características por importancia
    indices = np.argsort(importances)[::-1]
    sorted_names = [feature_names[i] for i in indices]
    sorted_importances = importances[indices]
    
    fig, ax = create_figure(figsize=(10, max(6, len(feature_names)*0.3)))
    
    # Crear gráfico de barras horizontales
    ax.barh(range(len(sorted_names)), sorted_importances, color='skyblue')
    
    # Configurar etiquetas
    ax.set_yticks(range(len(sorted_names)))
    ax.set_yticklabels(sorted_names)
    ax.set_title(title)
    ax.set_xlabel('Importancia')
    ax.grid(axis='x', alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        save_figure(save_path, fig)
    
    if show_plot:
        plt.show()
    else:
        plt.close(fig)
    
    return fig, ax

def _plot_loss_comparison(histories: Dict[str, Dict[str, List[float]]], 
                          save_path: str, 
                          show_plot: bool) -> str:
    """
    Visualiza y guarda una comparación de pérdidas durante entrenamiento.
    
    Parámetros:
    -----------
    histories : Dict[str, Dict[str, List[float]]]
        Diccionario con historiales de entrenamiento por modelo
    save_path : str
        Ruta donde guardar la figura
    show_plot : bool
        Indica si mostrar el gráfico
        
    Retorna:
    --------
    str
        Ruta del archivo guardado
    """
    fig, ax = create_figure(figsize=(14, 8))
    
    for model_name, history in histories.items():
        if 'loss' in history:
            ax.plot(history['loss'], label=f'{model_name} ({CONST_ENTRENAMIENTO})')
            if 'val_loss' in history:
                ax.plot(history['val_loss'], linestyle='--', 
                      label=f'{model_name} ({CONST_VALIDACION})')
    
    ax.set_title('Comparación de Pérdida Durante Entrenamiento')
    ax.set_xlabel(CONST_EPOCA)
    ax.set_ylabel(CONST_PERDIDA)
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    
    save_figure(save_path, fig)
    
    if not show_plot:
        plt.close(fig)
        
    return save_path


def _plot_metric_comparisons(metrics_all: Dict[str, Dict[str, float]], 
                            save_dir: str, 
                            show_plot: bool) -> Dict[str, str]:
    """
    Visualiza y guarda comparaciones de diferentes métricas.
    
    Parámetros:
    -----------
    metrics_all : Dict[str, Dict[str, float]]
        Diccionario con métricas de modelos
    save_dir : str
        Directorio donde guardar las figuras
    show_plot : bool
        Indica si mostrar los gráficos
        
    Retorna:
    --------
    Dict[str, str]
        Diccionario con rutas de los archivos guardados
    """
    metric_paths = {}
    
    for metric_name in [CONST_METRIC_MAE, CONST_METRIC_RMSE, CONST_METRIC_R2]:
        if not all(metric_name in m for m in metrics_all.values()):
            continue
            
        fig, _ = plot_model_comparison(
            metrics_all, 
            metric_name=metric_name, 
            title=f"Comparación de {metric_name.upper()}", 
            show_plot=False
        )
        
        metric_path = os.path.join(save_dir, f'{metric_name}_comparison.png')
        save_figure(metric_path, fig)
        metric_paths[metric_name] = metric_path
        
        if not show_plot:
            plt.close(fig)
            
    return metric_paths


def _plot_ensemble_figures(y_true: np.ndarray,
                          ensemble_predictions: np.ndarray,
                          ensemble_metrics: Dict[str, float],
                          ensemble_weights: Optional[np.ndarray] = None,
                          model_names: Optional[List[str]] = None,
                          save_dir: str = "",
                          sample_size: int = 100,
                          show_plot: bool = False) -> Dict[str, str]:
    """
    Visualiza y guarda figuras específicas para el ensemble.
    
    Parámetros:
    -----------
    y_true : np.ndarray
        Valores reales
    ensemble_predictions : np.ndarray
        Predicciones del ensemble
    ensemble_metrics : Dict[str, float]
        Métricas del ensemble
    ensemble_weights : Optional[np.ndarray], opcional
        Pesos del ensemble (default: None)
    model_names : Optional[List[str]], opcional
        Nombres de los modelos (default: None)
    save_dir : str, opcional
        Directorio donde guardar figuras (default: "")
    sample_size : int, opcional
        Número de muestras (default: 100)
    show_plot : bool, opcional
        Indica si mostrar gráficos (default: False)
        
    Retorna:
    --------
    Dict[str, str]
        Diccionario con rutas de los archivos
    """
    figure_paths = {}
    
    # Crear un gráfico de predicciones para el ensemble
    fig, _ = plot_predictions_vs_actual(
        y_true,
        {CONST_ENSEMBLE: ensemble_predictions},
        sample_size=sample_size,
        title=f"Predicciones del {CONST_ENSEMBLE}",
        show_plot=False
    )
    
    preds_path = os.path.join(save_dir, 'predictions.png')
    save_figure(preds_path, fig)
    figure_paths['predictions'] = preds_path
    
    if not show_plot:
        plt.close(fig)
    
    # Crear visualización de métricas para el ensemble
    metrics_ensemble = {CONST_ENSEMBLE: ensemble_metrics}
    fig, _ = plot_all_metrics(
        metrics_ensemble,
        show_plot=False
    )
    
    metrics_path = os.path.join(save_dir, 'metrics.png')
    save_figure(metrics_path, fig)
    figure_paths['metrics'] = metrics_path
    
    if not show_plot:
        plt.close(fig)
    
    # Si hay pesos del ensemble, guardarlos también
    if ensemble_weights is not None and model_names is not None:
        fig, _ = plot_ensemble_weights(
            ensemble_weights, 
            model_names,
            title="Pesos del Ensemble",
            show_plot=False
        )
        
        weights_path = os.path.join(save_dir, 'weights.png')
        save_figure(weights_path, fig)
        figure_paths['weights'] = weights_path
        
        if not show_plot:
            plt.close(fig)
            
    return figure_paths


def plot_model_evaluation_summary(histories: Dict[str, Dict[str, List[float]]], 
                                 predictions: Dict[str, np.ndarray],
                                 y_true: np.ndarray,
                                 metrics: Dict[str, Dict[str, float]],
                                 ensemble_predictions: Optional[np.ndarray] = None,
                                 ensemble_weights: Optional[np.ndarray] = None,
                                 ensemble_metrics: Optional[Dict[str, float]] = None,
                                 save_dir: str = "figures",
                                 sample_size: int = 100,
                                 show_plots: bool = False) -> Dict[str, Dict[str, str]]:
    """
    Genera y guarda un conjunto completo de visualizaciones para todos los modelos y ensemble.
    
    Parámetros:
    -----------
    histories : Dict[str, Dict[str, List[float]]]
        Diccionario con historiales de entrenamiento por modelo
    predictions : Dict[str, np.ndarray]
        Diccionario con predicciones por modelo
    y_true : np.ndarray
        Valores reales para comparar con las predicciones
    metrics : Dict[str, Dict[str, float]]
        Diccionario con métricas de rendimiento por modelo
    ensemble_predictions : Optional[np.ndarray], opcional
        Predicciones del modelo ensemble (default: None)
    ensemble_weights : Optional[np.ndarray], opcional
        Pesos optimizados para el ensemble (default: None)
    ensemble_metrics : Optional[Dict[str, float]], opcional
        Métricas del modelo ensemble (default: None)
    save_dir : str, opcional
        Directorio base para guardar las visualizaciones (default: "figures")
    sample_size : int, opcional
        Número de muestras a visualizar en gráficos de predicción (default: 100)
    show_plots : bool, opcional
        Indica si mostrar los gráficos en pantalla (default: False)
        
    Retorna:
    --------
    Dict[str, Dict[str, str]]
        Diccionario con rutas a los archivos generados por modelo
    """
    # Constantes para rutas
    CONST_INDIVIDUAL_DIR = "individual_models"
    CONST_COMPARATIVE_DIR = "comparative"
    
    # Crear estructura de directorios
    model_dir = os.path.join(save_dir, CONST_INDIVIDUAL_DIR)
    comparative_dir = os.path.join(save_dir, CONST_COMPARATIVE_DIR)
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(comparative_dir, exist_ok=True)
    
    # Diccionario para almacenar rutas de figuras
    figure_paths = {
        "comparative": {},
        "individual": {}
    }
    
    # 1. Gráficos comparativos
    
    # 1.1 Comparación de pérdidas durante entrenamiento
    loss_path = _plot_loss_comparison(
        histories, 
        os.path.join(comparative_dir, 'loss_comparison.png'),
        show_plots
    )
    figure_paths["comparative"]["loss"] = loss_path
    
    # 1.2 Comparación de métricas (MAE, RMSE, R2)
    metrics_all = metrics.copy()
    if ensemble_metrics:
        metrics_all[CONST_ENSEMBLE] = ensemble_metrics
    
    metric_paths = _plot_metric_comparisons(metrics_all, comparative_dir, show_plots)
    figure_paths["comparative"].update(metric_paths)
    
    # 1.3 Gráfico comparativo de todas las predicciones
    all_preds = predictions.copy()
    if ensemble_predictions is not None:
        all_preds[CONST_ENSEMBLE] = ensemble_predictions
    
    fig, _ = plot_predictions_vs_actual(
        y_true, 
        all_preds, 
        sample_size=sample_size,
        title="Comparación de Predicciones",
        show_plot=False
    )
    
    preds_path = os.path.join(comparative_dir, 'predictions_comparison.png')
    save_figure(preds_path, fig)
    figure_paths["comparative"]["predictions"] = preds_path
    if not show_plots:
        plt.close(fig)
    
    # 1.4 Pesos del ensemble (si se proporcionan)
    if ensemble_weights is not None and len(ensemble_weights) == len(predictions):
        fig, _ = plot_ensemble_weights(
            ensemble_weights, 
            list(predictions.keys()),
            title="Pesos del Ensemble",
            show_plot=False
        )
        
        weights_path = os.path.join(comparative_dir, 'ensemble_weights.png')
        save_figure(weights_path, fig)
        figure_paths["comparative"]["ensemble_weights"] = weights_path
        if not show_plots:
            plt.close(fig)
    
    # 1.5 Gráfico de todas las métricas
    fig, _ = plot_all_metrics(
        metrics_all,
        show_plot=False
    )
    
    all_metrics_path = os.path.join(comparative_dir, 'all_metrics.png')
    save_figure(all_metrics_path, fig)
    figure_paths["comparative"]["all_metrics"] = all_metrics_path
    if not show_plots:
        plt.close(fig)
    
    # 2. Gráficos por modelo individual
    figure_paths["individual"] = {}
    
    for model_name in histories.keys():
        model_figures = visualize_model_results(
            history=histories[model_name],
            predictions=predictions[model_name],
            y_test=y_true,
            model_name=model_name,
            save_dir=model_dir,
            show_plots=show_plots
        )
        figure_paths["individual"][model_name] = model_figures
    
    # 3. Gráficos del ensemble (si se proporcionan los datos necesarios)
    if ensemble_predictions is not None and ensemble_metrics is not None:
        ensemble_dir = os.path.join(model_dir, CONST_ENSEMBLE)
        os.makedirs(ensemble_dir, exist_ok=True)
        
        ensemble_figures = _plot_ensemble_figures(
            y_true=y_true,
            ensemble_predictions=ensemble_predictions,
            ensemble_metrics=ensemble_metrics,
            ensemble_weights=ensemble_weights,
            model_names=list(predictions.keys()) if ensemble_weights is not None else None,
            save_dir=ensemble_dir,
            sample_size=sample_size,
            show_plot=show_plots
        )
        
        figure_paths["individual"][CONST_ENSEMBLE] = ensemble_figures
    
    return figure_paths