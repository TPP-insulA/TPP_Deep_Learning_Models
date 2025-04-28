import os
from typing import Dict, Optional
from custom.printer import cprint
from constants.constants import CONST_FRAMEWORKS, CONST_MODELS_NAMES, HEADERS_BACKGROUND, MODELS_BACKGROUND, ENSEMBLE_BACKGROUND

def create_report(model_figures: Dict[str, Dict[str, str]], 
                  ensemble_metrics: Dict[str, float],
                  framework: str,
                  project_root: str,
                  figures_dir: str,
                  metrics: Dict[str, Dict[str, float]]) -> str:
    """
    Crea un reporte en formato Typst con los resultados de entrenamiento.
    
    Parámetros:
    -----------
    model_figures : Dict[str, Dict[str, str]]
        Diccionario con rutas a figuras por modelo
    ensemble_metrics : Dict[str, float]
        Métricas del modelo ensemble
    framework : str
        Framework utilizado (tensorflow o jax)
    project_root : str
        Ruta base del proyecto
    figures_dir : str
        Directorio donde se guardan las figuras
    metrics : Dict[str, Dict[str, float]]
        Métricas de todos los modelos
        
    Retorna:
    --------
    str
        Ruta al archivo Typst generado
    """
    # # Fecha actual para el reporte
    # from datetime import datetime
    # current_date = datetime.now().strftime("%Y-%m-%d")
    
    # Crear directorio docs si no existe
    docs_dir = os.path.join(project_root, "docs")
    os.makedirs(docs_dir, exist_ok=True)
    
    def get_min_max(data: Dict[str, Dict[str, float]], metric: str) -> tuple[float, float]:
        """
        Obtiene el mínimo y máximo valor de una métrica dada en los modelos.
        
        Parámetros:
        -----------
        data : Dict[str, Dict[str, float]]
            Diccionario con métricas de los modelos
        metric : str
            Nombre de la métrica a evaluar (ej. 'mae', 'rmse', 'r2')
        
        Retorna:
        --------
        tuple[float, float]
            Mínimo y máximo valor de la métrica
        """
        values = [model_metric[metric] for model_metric in data.values()]
        return min(values), max(values)

    def color_for_value(value: float, min_val: float, max_val: float, better_is_lower: bool = False) -> str:
        """
        Genera un color RGB como cadena para Typst.
        
        Parámetros:
        -----------
        value : float
            Valor a evaluar
        min_val : float
            Valor mínimo de la métrica
        max_val : float
            Valor máximo de la métrica
        better_is_lower : bool
            Indica si un valor menor es mejor (default: False)
        
        Retorna:
        --------
        str
            Color en formato RGB para Typst
        """
        if min_val == max_val:
            t = 0.5
        else:
            t = (value - min_val) / (max_val - min_val)
        if better_is_lower:
            r = int(t * 255)
            g = int(255 * (1 - t))
            b = 0
        else:
            r = int(255 * (1 - t))
            g = int(t * 255)
            b = 0
        return f"rgb({r}, {g}, {b})"
    
    mae_min, mae_max = get_min_max(metrics, "mae")
    rmse_min, rmse_max = get_min_max(metrics, "rmse")
    r2_min, r2_max = get_min_max(metrics, "r2")
    
    # Inicio del documento Typst
    typst_content = f"""
#set page(
  margin: 2cm,
  numbering: "1 de 1",
)

#set text(font: "New Computer Modern")
#set heading(numbering: "1.")
#show heading: set block(above: 1.4em, below: 1em)

#set table(
  fill: (x, y) =>
    if y == 0 {{
      rgb("{HEADERS_BACKGROUND}").lighten(40%)
    }} else if x == 0 {{
      rgb("{MODELS_BACKGROUND}")
    }},
  align: right,
)

#align(center)[
  #text(17pt)[*Resultados de Entrenamiento de Modelos*]
  #v(0.5em)
  #text(13pt)[#underline[Framework]: *{CONST_FRAMEWORKS[framework]}*]
]

= Resumen de Resultados

== Métricas de Rendimiento

#figure(
  table(
    columns: 4,
    align: center + horizon,
    [*Modelo*], [*MAE*], [*RMSE*], [*R²*],
"""

    # Agregar filas para cada modelo
    for model_name, model_metric in metrics.items():
        mae_color = color_for_value(model_metric["mae"], mae_min, mae_max, better_is_lower=True)
        rmse_color = color_for_value(model_metric["rmse"], rmse_min, rmse_max, better_is_lower=True)
        r2_color = color_for_value(model_metric["r2"], r2_min, r2_max)
        typst_content += f"""
    [*{CONST_MODELS_NAMES[model_name]}*], table.cell(fill: {mae_color}.lighten(37%), [{model_metric["mae"]:.4f}]), table.cell(fill: {rmse_color}.lighten(37%), [{model_metric["rmse"]:.4f}]), table.cell(fill: {r2_color}.lighten(37%),  [{model_metric["r2"]:.4f}]),"""
    
    # Agregar fila del ensemble
    typst_content += f"""
    table.cell(fill: rgb("{ENSEMBLE_BACKGROUND}"), [*Ensemble*]), table.cell(fill: rgb("{ENSEMBLE_BACKGROUND}"), [{ensemble_metrics["mae"]:.4f}]), table.cell(fill: rgb("{ENSEMBLE_BACKGROUND}"), [{ensemble_metrics["rmse"]:.4f}]), table.cell(fill: rgb("{ENSEMBLE_BACKGROUND}"), [{ensemble_metrics["r2"]:.4f}]),
  ),
  caption: [Comparación de métricas entre modelos],
)

= Resultados por Modelo

"""

    # Agregar secciones para cada modelo
    for model_name, figures in model_figures.items():
        # Obtener rutas relativas para las imágenes
        training_history = figures.get('training_history', '')
        predictions = figures.get('predictions', '')
        metrics_fig = figures.get('metrics', '')
        
        # Convertir rutas absolutas a relativas desde la ubicación del documento
        if training_history:
            training_history_rel = os.path.relpath(training_history, docs_dir)
        if predictions:
            predictions_rel = os.path.relpath(predictions, docs_dir)
        if metrics_fig:
            _ = os.path.relpath(metrics_fig, docs_dir)
        
        typst_content += f"""
== Modelo: {CONST_MODELS_NAMES[model_name]}

=== Métricas
- MAE: {metrics[model_name]["mae"]:.4f}
- RMSE: {metrics[model_name]["rmse"]:.4f}
- R²: {metrics[model_name]["r2"]:.4f}

=== Historial de Entrenamiento
#figure(
  image("{training_history_rel}", width: 71%),
  caption: [Historial de entrenamiento para {model_name}],
)

=== Predicciones
#figure(
  image("{predictions_rel}", width: 71%),
  caption: [Predicciones vs valores reales para {model_name}],
)

"""
    
    # Agregar sección para el ensemble
    typst_content += f"""
== Modelo Ensemble

=== Métricas
- MAE: {ensemble_metrics["mae"]:.4f}
- RMSE: {ensemble_metrics["rmse"]:.4f}
- R²: {ensemble_metrics["r2"]:.4f}

=== Pesos del Ensemble
#figure(
  image("{os.path.relpath(os.path.join(figures_dir, 'ensemble_weights.png'), docs_dir)}", width: 71%),
  caption: [Pesos optimizados para cada modelo en el ensemble],
)

= Conclusiones

El framework {framework.upper()} fue utilizado para entrenar {len(model_figures)} modelos diferentes. 
El modelo ensemble logró un MAE de {ensemble_metrics["mae"]:.4f}, un RMSE de {ensemble_metrics["rmse"]:.4f} 
y un coeficiente R² de {ensemble_metrics["r2"]:.4f}.

"""
    
    # Guardar el archivo Typst
    typst_path = os.path.join(docs_dir, f"models_results_{framework}.typ")
    with open(typst_path, 'w') as f:
        f.write(typst_content)
    
    return typst_path

def render_to_pdf(typst_path: str) -> Optional[str]:
    """
    Renderiza un archivo Typst a PDF si Typst está instalado.
    
    Parámetros:
    -----------
    typst_path : str
        Ruta al archivo Typst
        
    Retorna:
    --------
    Optional[str]
        Ruta al PDF generado o None si falló
    """
    try:
        import subprocess
        pdf_path = typst_path.replace('.typ', '.pdf')
        _ = subprocess.run(['typst', 'compile', typst_path, pdf_path], 
                              check=True, capture_output=True, text=True)
        return pdf_path
    except Exception as e:
        cprint(f"No se pudo renderizar el PDF: {e}", 'yellow')
        cprint("Para renderizar manualmente, ejecute: typst compile docs/models_results.typ", 'yellow')
        return None