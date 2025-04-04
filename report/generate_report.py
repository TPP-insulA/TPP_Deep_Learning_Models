import os
from typing import Dict, Optional
from custom.printer import cprint

def create_report(model_figures: Dict[str, Dict[str, str]], 
                       ensemble_metrics: Dict[str, float],
                       framework: str,
                       project_root,
                       figures_dir,
                       metrics) -> str:
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
        
    Retorna:
    --------
    str
        Ruta al archivo Typst generado
    """
    # Fecha actual para el reporte
    from datetime import datetime
    current_date = datetime.now().strftime("%Y-%m-%d")
    
    # Crear directorio docs si no existe
    docs_dir = os.path.join(project_root, "docs")
    os.makedirs(docs_dir, exist_ok=True)
    
    # Inicio del documento Typst
    typst_content = f"""
#import "@preview/tablex:0.0.5": tablex, cellx, rowspanx, colspanx

#set page(
  margin: 2.5cm,
  numbering: "1",
)

#set text(font: "New Computer Modern")
#show heading: set block(above: 1.4em, below: 1em)

#align(center)[
  #text(17pt)[*Resultados de Entrenamiento de Modelos*]
  #v(0.5em)
  #text(13pt)[Framework: {framework.upper()}]
  #v(0.5em)
  #text(11pt)[Fecha: {current_date}]
]

= Resumen de Resultados

== Métricas de Rendimiento

#figure(
  tablex(
    columns: (auto, auto, auto, auto),
    inset: 10pt,
    align: horizon,
    [], [*MAE*], [*RMSE*], [*R²*],
"""

    # Agregar filas para cada modelo
    for model_name, model_metric in metrics.items():
        typst_content += f"""
    [*{model_name}*], [{model_metric["mae"]:.4f}], [{model_metric["rmse"]:.4f}], [{model_metric["r2"]:.4f}],"""
    
    # Agregar fila del ensemble
    typst_content += f"""
    [*Ensemble*], [{ensemble_metrics["mae"]:.4f}], [{ensemble_metrics["rmse"]:.4f}], [{ensemble_metrics["r2"]:.4f}],
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
== Modelo: {model_name}

=== Métricas
- MAE: {metrics[model_name]["mae"]:.4f}
- RMSE: {metrics[model_name]["rmse"]:.4f}
- R²: {metrics[model_name]["r2"]:.4f}

=== Historial de Entrenamiento
#figure(
  image("{training_history_rel}", width: 80%),
  caption: [Historial de entrenamiento para {model_name}],
)

=== Predicciones
#figure(
  image("{predictions_rel}", width: 80%),
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
  image("{os.path.relpath(os.path.join(figures_dir, 'ensemble_weights.png'), docs_dir)}", width: 80%),
  caption: [Pesos optimizados para cada modelo en el ensemble],
)

= Conclusiones

El framework {framework.upper()} fue utilizado para entrenar {len(model_figures)} modelos diferentes. 
El modelo ensemble logró un MAE de {ensemble_metrics["mae"]:.4f}, un RMSE de {ensemble_metrics["rmse"]:.4f} 
y un coeficiente R² de {ensemble_metrics["r2"]:.4f}.

"""
    
    # Guardar el archivo Typst
    typst_path = os.path.join(docs_dir, "models_results.typ")
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