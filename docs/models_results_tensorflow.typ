
#set page(
  margin: 2cm,
  numbering: "1 de 1",
)

#set text(font: "New Computer Modern")
#set heading(numbering: "1.")
#show heading: set block(above: 1.4em, below: 1em)

#set table(
  fill: (x, y) =>
    if y == 0 {
      rgb("#e7da27").lighten(40%)
    } else if x == 0 {
      rgb("#e6e6e6")
    },
  align: right,
)

#align(center)[
  #text(17pt)[*Resultados de Entrenamiento de Modelos*]
  #v(0.5em)
  #text(13pt)[#underline[Framework]: *TensorFlow*]
]

= Resumen de Resultados

== Métricas de Rendimiento

#figure(
  table(
    columns: 4,
    align: center + horizon,
    [*Modelo*], [*MAE*], [*RMSE*], [*R²*],

    [*Reinforce Monte Carlo Policy Gradient*], table.cell(fill: rgb(0, 255, 0).lighten(37%), [1.2921]), table.cell(fill: rgb(0, 255, 0).lighten(37%), [1.4912]), table.cell(fill: rgb(0, 255, 0).lighten(37%),  [-0.4554]),
    [*Soft Actor-Critic*], table.cell(fill: rgb(255, 0, 0).lighten(37%), [6.8640]), table.cell(fill: rgb(255, 0, 0).lighten(37%), [8.2453]), table.cell(fill: rgb(255, 0, 0).lighten(37%),  [-43.4945]),
    [*Trust Region Policy Optimization*], table.cell(fill: rgb(16, 238, 0).lighten(37%), [1.6608]), table.cell(fill: rgb(15, 239, 0).lighten(37%), [1.9010]), table.cell(fill: rgb(5, 249, 0).lighten(37%),  [-1.3651]),
    table.cell(fill: rgb("#ffe0b2"), [*Ensemble*]), table.cell(fill: rgb("#ffe0b2"), [1.2742]), table.cell(fill: rgb("#ffe0b2"), [1.4684]), table.cell(fill: rgb("#ffe0b2"), [-0.4112]),
  ),
  caption: [Comparación de métricas entre modelos],
)

= Resultados por Modelo


== Modelo: Reinforce Monte Carlo Policy Gradient

=== Métricas
- MAE: 1.2921
- RMSE: 1.4912
- R²: -0.4554

=== Historial de Entrenamiento
#figure(
  image("../figures/various_models/tensorflow/individual_models/tf_reinforce_mcpg/training_history.png", width: 71%),
  caption: [Historial de entrenamiento para tf_reinforce_mcpg],
)

=== Predicciones
#figure(
  image("../figures/various_models/tensorflow/individual_models/tf_reinforce_mcpg/predictions.png", width: 71%),
  caption: [Predicciones vs valores reales para tf_reinforce_mcpg],
)


== Modelo: Soft Actor-Critic

=== Métricas
- MAE: 6.8640
- RMSE: 8.2453
- R²: -43.4945

=== Historial de Entrenamiento
#figure(
  image("../figures/various_models/tensorflow/individual_models/tf_sac/training_history.png", width: 71%),
  caption: [Historial de entrenamiento para tf_sac],
)

=== Predicciones
#figure(
  image("../figures/various_models/tensorflow/individual_models/tf_sac/predictions.png", width: 71%),
  caption: [Predicciones vs valores reales para tf_sac],
)


== Modelo: Trust Region Policy Optimization

=== Métricas
- MAE: 1.6608
- RMSE: 1.9010
- R²: -1.3651

=== Historial de Entrenamiento
#figure(
  image("../figures/various_models/tensorflow/individual_models/tf_trpo/training_history.png", width: 71%),
  caption: [Historial de entrenamiento para tf_trpo],
)

=== Predicciones
#figure(
  image("../figures/various_models/tensorflow/individual_models/tf_trpo/predictions.png", width: 71%),
  caption: [Predicciones vs valores reales para tf_trpo],
)


== Modelo Ensemble

=== Métricas
- MAE: 1.2742
- RMSE: 1.4684
- R²: -0.4112

=== Pesos del Ensemble
#figure(
  image("../figures/various_models/tensorflow/ensemble_weights.png", width: 71%),
  caption: [Pesos optimizados para cada modelo en el ensemble],
)

= Conclusiones

El framework TENSORFLOW fue utilizado para entrenar 3 modelos diferentes. 
El modelo ensemble logró un MAE de 1.2742, un RMSE de 1.4684 
y un coeficiente R² de -0.4112.

