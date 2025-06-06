import os
from typing import Any, Dict, Optional
from datetime import datetime
from custom.printer import cprint
from constants.constants import CONST_FRAMEWORKS, CONST_MODELS_NAMES, HEADERS_BACKGROUND, MODELS_BACKGROUND, ENSEMBLE_BACKGROUND

def create_report(data: Dict[str, Any], output_path: str, figures_dir: str) -> str:
    """
    Crea un reporte en formato Typst con los resultados de entrenamiento, priorizando métricas clínicas.
    
    Parámetros:
    -----------
    data : Dict[str, Any]
        Diccionario con datos para el reporte incluyendo:
        - title: Título del informe
        - framework: Framework utilizado
        - models: Información de modelos entrenados
        - metrics: Métricas de regresión
        - clinical_metrics: Métricas clínicas
        - offline_evaluation: Resultados de evaluación offline (opcional)
        - figures: Rutas a las figuras generadas
    output_path : str
        Ruta donde se guardará el reporte
    figures_dir : str
        Directorio donde se encuentran las figuras
        
    Retorna:
    --------
    str
        Ruta al archivo Typst generado
    """
    # Obtener fecha actual para el informe
    current_date = datetime.now().strftime("%d/%m/%Y")
    
    # Crear directorio para el informe si no existe
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Obtener métricas clínicas
    clinical_metrics = data.get('clinical_metrics', {})
    
    # Función para generar color según el valor clínico
    def color_for_clinical(value: float, metric: str) -> str:
        """
        Genera color para métricas clínicas.
        
        Parámetros:
        -----------
        value : float
            Valor de la métrica
        metric : str
            Nombre de la métrica
            
        Retorna:
        --------
        str
            Color en formato RGB para Typst
        """
        if metric == 'time_in_range':
            # Verde más intenso para valores más altos (mejor)
            intensity = min(100, int(value * 2.55))
            return f"rgb(0, {intensity}, 0)"
        elif metric == 'time_below_range':
            # Rojo más intenso para valores más altos (peor)
            intensity = min(100, int(value * 2.55))
            return f"rgb({intensity}, 0, 0)"
        elif metric == 'time_above_range':
            # Naranja más intenso para valores más altos (peor)
            intensity = min(100, int(value * 2.55))
            return f"rgb({intensity}, {intensity//2}, 0)"
        return "black"
    
    # Inicio del documento Typst
    typst_content = f"""
#set page(
  margin: 2cm,
  numbering: "1 de 1",
)

#set text(font: "New Computer Modern", lang: "es")
#set heading(numbering: "1.")
#show heading: set block(above: 1.4em, below: 1em)

#set table(
  fill: (x, y) => {{
    if y == 0 {{
      rgb("{HEADERS_BACKGROUND}").lighten(40%)
    }} else if x == 0 {{
      rgb("{MODELS_BACKGROUND}")
    }}
  }},
  align: center,
)

#align(center)[
  #text(17pt)[*{data['title']}*]
  #v(0.5em)
  #text(13pt)[Fecha: *{data.get('date', current_date)}*]
  #v(0.5em)
  #text(13pt)[Framework: *{data['framework'].upper()}*]
]

= Métricas Clínicas y Resultados de Evaluación

La eficacia de los modelos de dosificación de insulina se evalúa principalmente por su capacidad para mantener los niveles de glucosa dentro del rango objetivo (70-180 mg/dL).

== Tiempo en Rango y Control Glucémico

#figure(
  table(
    columns: 4,
    align: center + horizon,
    [*Modelo*], [*Tiempo en Rango (%)*], [*Tiempo Bajo Rango (%)*], [*Tiempo Sobre Rango (%)*],
"""

    # Agregar filas para cada modelo (métricas clínicas)
    models = list(clinical_metrics.keys())
    for model_name in models:
        model_metrics = clinical_metrics[model_name]
        tir = model_metrics.get('time_in_range', 0)
        tbr = model_metrics.get('time_below_range', 0)
        tar = model_metrics.get('time_above_range', 0)
        
        tir_color = color_for_clinical(tir, 'time_in_range')
        tbr_color = color_for_clinical(tbr, 'time_below_range')
        tar_color = color_for_clinical(tar, 'time_above_range')
        
        model_display_name = CONST_MODELS_NAMES.get(model_name, model_name)
        
        typst_content += f"""
    [*{model_display_name}*], 
    table.cell(fill: {tir_color}.lighten(80%), [{tir:.2f}%]), 
    table.cell(fill: {tbr_color}.lighten(80%), [{tbr:.2f}%]), 
    table.cell(fill: {tar_color}.lighten(80%), [{tar:.2f}%]),"""
    
    typst_content += f"""
  ),
  caption: [Comparación de métricas clínicas entre modelos],
)

Las métricas clave son:
- *Tiempo en Rango (TIR)*: Porcentaje de tiempo que los niveles de glucosa permanecen entre 70-180 mg/dL.
- *Tiempo Bajo Rango (TBR)*: Porcentaje de tiempo con glucosa <70 mg/dL (hipoglucemia).
- *Tiempo Sobre Rango (TAR)*: Porcentaje de tiempo con glucosa >180 mg/dL (hiperglucemia).

#figure(
  image("{os.path.relpath(os.path.join(figures_dir, 'clinical_metrics.png'), os.path.dirname(output_path))}", width: 85%),
  caption: [Visualización de métricas clínicas entre modelos],
)
"""

    # Agregar sección de evaluación offline si existe
    if 'offline_evaluation' in data:
        offline_results = data['offline_evaluation']
        typst_content += """
== Evaluación con Métodos Offline RL

La evaluación fuera de política (off-policy) permite estimar el rendimiento esperado de los modelos sin necesidad de probarlos directamente en pacientes.

#figure(
  table(
    columns: 4,
    align: center + horizon,
    [*Modelo*], [*Valor Estimado*], [*Límite Inferior*], [*Límite Superior*],
"""
        
        # Obtener primero los evaluadores disponibles
        for model_name in offline_results:
            evaluators = list(offline_results[model_name].keys())
            break
        
        # Para cada evaluador, mostrar resultados de todos los modelos
        for evaluator in evaluators:
            # Agregar subtítulo del evaluador
            typst_content += f"""
  ),
  caption: [Resultados de evaluación con {evaluator}],
)

=== Evaluación con {evaluator}

#figure(
  table(
    columns: 4,
    align: center + horizon,
    [*Modelo*], [*Valor Estimado*], [*Límite Inferior*], [*Límite Superior*],
"""
            
            # Agregar filas para cada modelo
            for model_name in offline_results:
                if evaluator in offline_results[model_name]:
                    results = offline_results[model_name][evaluator]
                    estimated = results.get('estimated_value', 0)
                    lower = results.get('confidence_lower', 0)
                    upper = results.get('confidence_upper', 0)
                    
                    model_display_name = CONST_MODELS_NAMES.get(model_name, model_name)
                    
                    typst_content += f"""
    [*{model_display_name}*], [{estimated:.4f}], [{lower:.4f}], [{upper:.4f}],"""
            
            # Agregar figura si existe
            if 'offline' in data.get('figures', {}):
                for fig_path in data['figures']['offline']:
                    if evaluator in fig_path:
                        fig_rel_path = os.path.relpath(os.path.join(figures_dir, fig_path), os.path.dirname(output_path))
                        typst_content += f"""
  ),
  caption: [Resultados de evaluación con {evaluator}],
)

#figure(
  image("{fig_rel_path}", width: 80%),
  caption: [Visualización de evaluación con {evaluator}],
)
"""
                        break
                else:
                    typst_content += """
  ),
  caption: [Resultados de evaluación offline],
)
"""
            else:
                typst_content += """
  ),
  caption: [Resultados de evaluación offline],
)
"""

    # Agregar sección de métricas de regresión (secundaria)
    regression_metrics = data.get('metrics', {})
    
    typst_content += """

= Métricas de Regresión (Secundarias)

Aunque las métricas clínicas son más relevantes para evaluar el impacto real de los modelos en pacientes, las métricas de regresión proporcionan información sobre la precisión de las predicciones respecto a los datos de entrenamiento.

"""

    # Función para obtener min/max de métricas
    def get_min_max(metrics_dict: Dict[str, Dict[str, float]], metric: str) -> tuple:
        values = [model_metric.get(metric, 0) for model_metric in metrics_dict.values()]
        return min(values), max(values)
    
    # Función para color según valor
    def color_for_value(value: float, min_val: float, max_val: float, better_is_lower: bool = False) -> str:
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
    
    # Obtener min/max para colores
    mae_min, mae_max = get_min_max(regression_metrics, "mae")
    rmse_min, rmse_max = get_min_max(regression_metrics, "rmse")
    r2_min, r2_max = get_min_max(regression_metrics, "r2")
    
    # Métricas de regresión en tabla
    typst_content += f"""
#figure(
  table(
    columns: 4,
    align: center + horizon,
    [*Modelo*], [*MAE*], [*RMSE*], [*R²*],
"""

    # Agregar filas para cada modelo (métricas de regresión)
    for model_name in regression_metrics.keys():
        model_metric = regression_metrics[model_name]
        mae_color = color_for_value(model_metric.get("mae", 0), mae_min, mae_max, better_is_lower=True)
        rmse_color = color_for_value(model_metric.get("rmse", 0), rmse_min, rmse_max, better_is_lower=True)
        r2_color = color_for_value(model_metric.get("r2", 0), r2_min, r2_max)
        
        model_display_name = CONST_MODELS_NAMES.get(model_name, model_name)
        
        typst_content += f"""
    [*{model_display_name}*], 
    table.cell(fill: {mae_color}.lighten(80%), [{model_metric.get("mae", 0):.4f}]), 
    table.cell(fill: {rmse_color}.lighten(80%), [{model_metric.get("rmse", 0):.4f}]), 
    table.cell(fill: {r2_color}.lighten(80%), [{model_metric.get("r2", 0):.4f}]),"""
    
    typst_content += f"""
  ),
  caption: [Comparación de métricas de regresión entre modelos],
)

#figure(
  image("{os.path.relpath(os.path.join(figures_dir, 'regression_metrics.png'), os.path.dirname(output_path))}", width: 85%),
  caption: [Visualización de métricas de regresión],
)
"""

    # Agregar sección para cada modelo
    typst_content += """
= Análisis por Modelo

A continuación se presentan los detalles de entrenamiento y evaluación para cada modelo.
"""

    for model_name in models:
        model_history_fig = f"{model_name}_training.png"
        
        typst_content += f"""
== Modelo: {CONST_MODELS_NAMES.get(model_name, model_name)}

=== Métricas Clínicas
- Tiempo en Rango: {clinical_metrics[model_name].get('time_in_range', 0):.2f}%
- Tiempo Bajo Rango: {clinical_metrics[model_name].get('time_below_range', 0):.2f}%
- Tiempo Sobre Rango: {clinical_metrics[model_name].get('time_above_range', 0):.2f}%

=== Métricas de Regresión
- MAE: {regression_metrics.get(model_name, {}).get('mae', 0):.4f}
- RMSE: {regression_metrics.get(model_name, {}).get('rmse', 0):.4f}
- R²: {regression_metrics.get(model_name, {}).get('r2', 0):.4f}

=== Historial de Entrenamiento
#figure(
  image("{os.path.relpath(os.path.join(figures_dir, model_history_fig), os.path.dirname(output_path))}", width: 80%),
  caption: [Historial de entrenamiento para {CONST_MODELS_NAMES.get(model_name, model_name)}],
)
"""

    # Agregar conclusiones    
    typst_content += f"""
= Conclusiones

Se han evaluado varios modelos de aprendizaje por refuerzo profundo para la predicción de dosis de insulina.

== Mejor Rendimiento Clínico

"""
    # Identificar el mejor modelo por tiempo en rango
    best_model = max(clinical_metrics.keys(), key=lambda m: clinical_metrics[m].get('time_in_range', 0))
    best_tir = clinical_metrics[best_model].get('time_in_range', 0)
    
    typst_content += f"""
El modelo con mejor rendimiento clínico es *{CONST_MODELS_NAMES.get(best_model, best_model)}* con un tiempo en rango de {best_tir:.2f}%.

== Consideraciones Adicionales

Las métricas clínicas deben ser el principal criterio para seleccionar modelos en el contexto de dosificación de insulina, ya que están directamente relacionadas con el objetivo final de mantener niveles de glucosa en el rango seguro.

"""

    # Guardar el archivo Typst
    with open(output_path, 'w') as f:
        f.write(typst_content)
    
    return output_path

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