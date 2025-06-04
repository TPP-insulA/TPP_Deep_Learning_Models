
#set page(
  margin: 2cm,
  numbering: "1 de 1",
)

#set text(font: "New Computer Modern", lang: "es")
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
  #text(13pt)[#underline[Framework]: *PyTorch*]
]

= Resumen de Resultados

== Métricas de Rendimiento

#figure(
  table(
    columns: 4,
    align: center + horizon,
    [*Modelo*], [*MAE*], [*RMSE*], [*R²*],

    [*Advantage Actor-Critic*], table.cell(fill: rgb(103, 151, 0).lighten(37%), [0.6121]), table.cell(fill: rgb(98, 156, 0).lighten(37%), [0.8011]), table.cell(fill: rgb(84, 170, 0).lighten(37%),  [0.0663]),
    [*Asynchronous Advantage Actor-Critic*], table.cell(fill: rgb(168, 86, 0).lighten(37%), [0.6483]), table.cell(fill: rgb(114, 140, 0).lighten(37%), [0.8256]), table.cell(fill: rgb(99, 155, 0).lighten(37%),  [0.0083]),
    [*Deep Deterministic Policy Gradient*], table.cell(fill: rgb(0, 255, 0).lighten(37%), [0.5542]), table.cell(fill: rgb(0, 255, 0).lighten(37%), [0.6482]), table.cell(fill: rgb(0, 255, 0).lighten(37%),  [0.3887]),
    [*Deep Q-Network*], table.cell(fill: rgb(255, 0, 0).lighten(37%), [0.6964]), table.cell(fill: rgb(255, 0, 0).lighten(37%), [1.0435]), table.cell(fill: rgb(255, 0, 0).lighten(37%),  [-0.5843]),
    [*Proximal Policy Optimization*], table.cell(fill: rgb(178, 76, 0).lighten(37%), [0.6536]), table.cell(fill: rgb(117, 137, 0).lighten(37%), [0.8298]), table.cell(fill: rgb(102, 152, 0).lighten(37%),  [-0.0017]),
    [*Soft Actor-Critic*], table.cell(fill: rgb(132, 122, 0).lighten(37%), [0.6283]), table.cell(fill: rgb(94, 160, 0).lighten(37%), [0.7949]), table.cell(fill: rgb(80, 174, 0).lighten(37%),  [0.0806]),
    [*Trust Region Policy Optimization*], table.cell(fill: rgb(141, 113, 0).lighten(37%), [0.6330]), table.cell(fill: rgb(103, 151, 0).lighten(37%), [0.8083]), table.cell(fill: rgb(88, 166, 0).lighten(37%),  [0.0495]),
    table.cell(fill: rgb("#ffe0b2"), [*Ensemble*]), table.cell(fill: rgb("#ffe0b2"), [0.5539]), table.cell(fill: rgb("#ffe0b2"), [0.6467]), table.cell(fill: rgb("#ffe0b2"), [0.3914]),
  ),
  caption: [Comparación de métricas entre modelos],
)

= Resultados por Modelo


== Modelo: Advantage Actor-Critic

=== Métricas
- MAE: 0.6121
- RMSE: 0.8011
- R²: 0.0663

=== Historial de Entrenamiento
#figure(
  image("../figures/various_models/pytorch/individual_models/pt_a2c/training_history.png", width: 71%),
  caption: [Historial de entrenamiento para Advantage Actor-Critic.],
)

=== Predicciones
#figure(
  image("../figures/various_models/pytorch/individual_models/pt_a2c/predictions.png", width: 71%),
  caption: [Predicciones vs valores reales para Advantage Actor-Critic.],
)


== Modelo: Asynchronous Advantage Actor-Critic

=== Métricas
- MAE: 0.6483
- RMSE: 0.8256
- R²: 0.0083

=== Historial de Entrenamiento
#figure(
  image("../figures/various_models/pytorch/individual_models/pt_a3c/training_history.png", width: 71%),
  caption: [Historial de entrenamiento para Asynchronous Advantage Actor-Critic.],
)

=== Predicciones
#figure(
  image("../figures/various_models/pytorch/individual_models/pt_a3c/predictions.png", width: 71%),
  caption: [Predicciones vs valores reales para Asynchronous Advantage Actor-Critic.],
)


== Modelo: Deep Deterministic Policy Gradient

=== Métricas
- MAE: 0.5542
- RMSE: 0.6482
- R²: 0.3887

=== Historial de Entrenamiento
#figure(
  image("../figures/various_models/pytorch/individual_models/pt_ddpg/training_history.png", width: 71%),
  caption: [Historial de entrenamiento para Deep Deterministic Policy Gradient.],
)

=== Predicciones
#figure(
  image("../figures/various_models/pytorch/individual_models/pt_ddpg/predictions.png", width: 71%),
  caption: [Predicciones vs valores reales para Deep Deterministic Policy Gradient.],
)


== Modelo: Deep Q-Network

=== Métricas
- MAE: 0.6964
- RMSE: 1.0435
- R²: -0.5843

=== Historial de Entrenamiento
#figure(
  image("../figures/various_models/pytorch/individual_models/pt_dqn/training_history.png", width: 71%),
  caption: [Historial de entrenamiento para Deep Q-Network.],
)

=== Predicciones
#figure(
  image("../figures/various_models/pytorch/individual_models/pt_dqn/predictions.png", width: 71%),
  caption: [Predicciones vs valores reales para Deep Q-Network.],
)


== Modelo: Proximal Policy Optimization

=== Métricas
- MAE: 0.6536
- RMSE: 0.8298
- R²: -0.0017

=== Historial de Entrenamiento
#figure(
  image("../figures/various_models/pytorch/individual_models/pt_ppo/training_history.png", width: 71%),
  caption: [Historial de entrenamiento para Proximal Policy Optimization.],
)

=== Predicciones
#figure(
  image("../figures/various_models/pytorch/individual_models/pt_ppo/predictions.png", width: 71%),
  caption: [Predicciones vs valores reales para Proximal Policy Optimization.],
)


== Modelo: Soft Actor-Critic

=== Métricas
- MAE: 0.6283
- RMSE: 0.7949
- R²: 0.0806

=== Historial de Entrenamiento
#figure(
  image("../figures/various_models/pytorch/individual_models/pt_sac/training_history.png", width: 71%),
  caption: [Historial de entrenamiento para Soft Actor-Critic.],
)

=== Predicciones
#figure(
  image("../figures/various_models/pytorch/individual_models/pt_sac/predictions.png", width: 71%),
  caption: [Predicciones vs valores reales para Soft Actor-Critic.],
)


== Modelo: Trust Region Policy Optimization

=== Métricas
- MAE: 0.6330
- RMSE: 0.8083
- R²: 0.0495

=== Historial de Entrenamiento
#figure(
  image("../figures/various_models/pytorch/individual_models/pt_trpo/training_history.png", width: 71%),
  caption: [Historial de entrenamiento para Trust Region Policy Optimization.],
)

=== Predicciones
#figure(
  image("../figures/various_models/pytorch/individual_models/pt_trpo/predictions.png", width: 71%),
  caption: [Predicciones vs valores reales para Trust Region Policy Optimization.],
)


== Modelo Ensemble

=== Métricas
- MAE: 0.5539
- RMSE: 0.6467
- R²: 0.3914

=== Pesos del Ensemble
#figure(
  image("../figures/various_models/pytorch/ensemble_weights.png", width: 71%),
  caption: [Pesos optimizados para cada modelo en el ensemble],
)

= Conclusiones

El framework PYTORCH fue utilizado para entrenar 7 modelos diferentes. 
El modelo ensemble logró un MAE de 0.5539, un RMSE de 0.6467 
y un coeficiente R² de 0.3914.

