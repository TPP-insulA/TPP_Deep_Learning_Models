
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
  #text(13pt)[#underline[Framework]: *PyTorch*]
]

= Resumen de Resultados

== Métricas de Rendimiento

#figure(
  table(
    columns: 4,
    align: center + horizon,
    [*Modelo*], [*MAE*], [*RMSE*], [*R²*],

    [*Advantage Actor-Critic*], table.cell(fill: rgb(102, 152, 0).lighten(37%), [0.8495]), table.cell(fill: rgb(145, 109, 0).lighten(37%), [1.2773]), table.cell(fill: rgb(135, 119, 0).lighten(37%),  [0.0146]),
    [*Asynchronous Advantage Actor-Critic*], table.cell(fill: rgb(157, 97, 0).lighten(37%), [0.8927]), table.cell(fill: rgb(153, 101, 0).lighten(37%), [1.2897]), table.cell(fill: rgb(144, 110, 0).lighten(37%),  [-0.0046]),
    [*Deep Deterministic Policy Gradient*], table.cell(fill: rgb(0, 255, 0).lighten(37%), [0.7695]), table.cell(fill: rgb(0, 255, 0).lighten(37%), [1.0591]), table.cell(fill: rgb(0, 255, 0).lighten(37%),  [0.3225]),
    [*Deep Q-Network*], table.cell(fill: rgb(231, 23, 0).lighten(37%), [0.9509]), table.cell(fill: rgb(255, 0, 0).lighten(37%), [1.4422]), table.cell(fill: rgb(255, 0, 0).lighten(37%),  [-0.2562]),
    [*Proximal Policy Optimization*], table.cell(fill: rgb(196, 58, 0).lighten(37%), [0.9234]), table.cell(fill: rgb(151, 103, 0).lighten(37%), [1.2869]), table.cell(fill: rgb(142, 112, 0).lighten(37%),  [-0.0002]),
    [*Soft Actor-Critic*], table.cell(fill: rgb(255, 0, 0).lighten(37%), [0.9690]), table.cell(fill: rgb(157, 97, 0).lighten(37%), [1.2952]), table.cell(fill: rgb(147, 107, 0).lighten(37%),  [-0.0132]),
    [*Trust Region Policy Optimization*], table.cell(fill: rgb(101, 153, 0).lighten(37%), [0.8485]), table.cell(fill: rgb(137, 117, 0).lighten(37%), [1.2651]), table.cell(fill: rgb(127, 127, 0).lighten(37%),  [0.0333]),
    table.cell(fill: rgb("#ffe0b2"), [*Ensemble*]), table.cell(fill: rgb("#ffe0b2"), [0.7695]), table.cell(fill: rgb("#ffe0b2"), [1.0591]), table.cell(fill: rgb("#ffe0b2"), [0.3225]),
  ),
  caption: [Comparación de métricas entre modelos],
)

= Resultados por Modelo


== Modelo: Advantage Actor-Critic

=== Métricas
- MAE: 0.8495
- RMSE: 1.2773
- R²: 0.0146

=== Historial de Entrenamiento
#figure(
  image("../figures/various_models/pytorch/individual_models/pt_a2c/training_history.png", width: 71%),
  caption: [Historial de entrenamiento para pt_a2c],
)

=== Predicciones
#figure(
  image("../figures/various_models/pytorch/individual_models/pt_a2c/predictions.png", width: 71%),
  caption: [Predicciones vs valores reales para pt_a2c],
)


== Modelo: Asynchronous Advantage Actor-Critic

=== Métricas
- MAE: 0.8927
- RMSE: 1.2897
- R²: -0.0046

=== Historial de Entrenamiento
#figure(
  image("../figures/various_models/pytorch/individual_models/pt_a3c/training_history.png", width: 71%),
  caption: [Historial de entrenamiento para pt_a3c],
)

=== Predicciones
#figure(
  image("../figures/various_models/pytorch/individual_models/pt_a3c/predictions.png", width: 71%),
  caption: [Predicciones vs valores reales para pt_a3c],
)


== Modelo: Deep Deterministic Policy Gradient

=== Métricas
- MAE: 0.7695
- RMSE: 1.0591
- R²: 0.3225

=== Historial de Entrenamiento
#figure(
  image("../figures/various_models/pytorch/individual_models/pt_ddpg/training_history.png", width: 71%),
  caption: [Historial de entrenamiento para pt_ddpg],
)

=== Predicciones
#figure(
  image("../figures/various_models/pytorch/individual_models/pt_ddpg/predictions.png", width: 71%),
  caption: [Predicciones vs valores reales para pt_ddpg],
)


== Modelo: Deep Q-Network

=== Métricas
- MAE: 0.9509
- RMSE: 1.4422
- R²: -0.2562

=== Historial de Entrenamiento
#figure(
  image("../figures/various_models/pytorch/individual_models/pt_dqn/training_history.png", width: 71%),
  caption: [Historial de entrenamiento para pt_dqn],
)

=== Predicciones
#figure(
  image("../figures/various_models/pytorch/individual_models/pt_dqn/predictions.png", width: 71%),
  caption: [Predicciones vs valores reales para pt_dqn],
)


== Modelo: Proximal Policy Optimization

=== Métricas
- MAE: 0.9234
- RMSE: 1.2869
- R²: -0.0002

=== Historial de Entrenamiento
#figure(
  image("../figures/various_models/pytorch/individual_models/pt_ppo/training_history.png", width: 71%),
  caption: [Historial de entrenamiento para pt_ppo],
)

=== Predicciones
#figure(
  image("../figures/various_models/pytorch/individual_models/pt_ppo/predictions.png", width: 71%),
  caption: [Predicciones vs valores reales para pt_ppo],
)


== Modelo: Soft Actor-Critic

=== Métricas
- MAE: 0.9690
- RMSE: 1.2952
- R²: -0.0132

=== Historial de Entrenamiento
#figure(
  image("../figures/various_models/pytorch/individual_models/pt_sac/training_history.png", width: 71%),
  caption: [Historial de entrenamiento para pt_sac],
)

=== Predicciones
#figure(
  image("../figures/various_models/pytorch/individual_models/pt_sac/predictions.png", width: 71%),
  caption: [Predicciones vs valores reales para pt_sac],
)


== Modelo: Trust Region Policy Optimization

=== Métricas
- MAE: 0.8485
- RMSE: 1.2651
- R²: 0.0333

=== Historial de Entrenamiento
#figure(
  image("../figures/various_models/pytorch/individual_models/pt_trpo/training_history.png", width: 71%),
  caption: [Historial de entrenamiento para pt_trpo],
)

=== Predicciones
#figure(
  image("../figures/various_models/pytorch/individual_models/pt_trpo/predictions.png", width: 71%),
  caption: [Predicciones vs valores reales para pt_trpo],
)


== Modelo Ensemble

=== Métricas
- MAE: 0.7695
- RMSE: 1.0591
- R²: 0.3225

=== Pesos del Ensemble
#figure(
  image("../figures/various_models/pytorch/ensemble_weights.png", width: 71%),
  caption: [Pesos optimizados para cada modelo en el ensemble],
)

= Conclusiones

El framework PYTORCH fue utilizado para entrenar 7 modelos diferentes. 
El modelo ensemble logró un MAE de 0.7695, un RMSE de 1.0591 
y un coeficiente R² de 0.3225.

