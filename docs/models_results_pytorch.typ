
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

    [*Advantage Actor-Critic*], table.cell(fill: rgb(190, 64, 0).lighten(37%), [0.8074]), table.cell(fill: rgb(87, 167, 0).lighten(37%), [0.9621]), table.cell(fill: rgb(77, 177, 0).lighten(37%),  [-0.0940]),
    [*Asynchronous Advantage Actor-Critic*], table.cell(fill: rgb(185, 69, 0).lighten(37%), [0.8030]), table.cell(fill: rgb(82, 172, 0).lighten(37%), [0.9563]), table.cell(fill: rgb(73, 181, 0).lighten(37%),  [-0.0809]),
    [*Deep Deterministic Policy Gradient*], table.cell(fill: rgb(0, 255, 0).lighten(37%), [0.6386]), table.cell(fill: rgb(0, 255, 0).lighten(37%), [0.8430]), table.cell(fill: rgb(0, 255, 0).lighten(37%),  [0.1602]),
    [*Deep Q-Network*], table.cell(fill: rgb(255, 0, 0).lighten(37%), [0.8640]), table.cell(fill: rgb(255, 0, 0).lighten(37%), [1.1920]), table.cell(fill: rgb(255, 0, 0).lighten(37%),  [-0.6792]),
    [*Proximal Policy Optimization*], table.cell(fill: rgb(187, 67, 0).lighten(37%), [0.8045]), table.cell(fill: rgb(83, 171, 0).lighten(37%), [0.9577]), table.cell(fill: rgb(74, 180, 0).lighten(37%),  [-0.0840]),
    [*Soft Actor-Critic*], table.cell(fill: rgb(184, 70, 0).lighten(37%), [0.8014]), table.cell(fill: rgb(80, 174, 0).lighten(37%), [0.9537]), table.cell(fill: rgb(71, 183, 0).lighten(37%),  [-0.0750]),
    [*Trust Region Policy Optimization*], table.cell(fill: rgb(175, 79, 0).lighten(37%), [0.7941]), table.cell(fill: rgb(76, 178, 0).lighten(37%), [0.9471]), table.cell(fill: rgb(66, 188, 0).lighten(37%),  [-0.0600]),
    table.cell(fill: rgb("#ffe0b2"), [*Ensemble*]), table.cell(fill: rgb("#ffe0b2"), [0.6470]), table.cell(fill: rgb("#ffe0b2"), [0.8393]), table.cell(fill: rgb("#ffe0b2"), [0.1675]),
  ),
  caption: [Comparación de métricas entre modelos],
)

= Resultados por Modelo


== Modelo: Advantage Actor-Critic

=== Métricas
- MAE: 0.8074
- RMSE: 0.9621
- R²: -0.0940

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
- MAE: 0.8030
- RMSE: 0.9563
- R²: -0.0809

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
- MAE: 0.6386
- RMSE: 0.8430
- R²: 0.1602

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
- MAE: 0.8640
- RMSE: 1.1920
- R²: -0.6792

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
- MAE: 0.8045
- RMSE: 0.9577
- R²: -0.0840

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
- MAE: 0.8014
- RMSE: 0.9537
- R²: -0.0750

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
- MAE: 0.7941
- RMSE: 0.9471
- R²: -0.0600

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
- MAE: 0.6470
- RMSE: 0.8393
- R²: 0.1675

=== Pesos del Ensemble
#figure(
  image("../figures/various_models/pytorch/ensemble_weights.png", width: 71%),
  caption: [Pesos optimizados para cada modelo en el ensemble],
)

= Conclusiones

El framework PYTORCH fue utilizado para entrenar 7 modelos diferentes. 
El modelo ensemble logró un MAE de 0.6470, un RMSE de 0.8393 
y un coeficiente R² de 0.1675.

