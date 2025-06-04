
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

    [*Advantage Actor-Critic*], table.cell(fill: rgb(29, 225, 0).lighten(37%), [0.7891]), table.cell(fill: rgb(112, 142, 0).lighten(37%), [1.2125]), table.cell(fill: rgb(104, 150, 0).lighten(37%),  [0.1121]),
    [*Asynchronous Advantage Actor-Critic*], table.cell(fill: rgb(126, 128, 0).lighten(37%), [0.8532]), table.cell(fill: rgb(160, 94, 0).lighten(37%), [1.2756]), table.cell(fill: rgb(151, 103, 0).lighten(37%),  [0.0172]),
    [*Deep Deterministic Policy Gradient*], table.cell(fill: rgb(0, 255, 0).lighten(37%), [0.7695]), table.cell(fill: rgb(0, 255, 0).lighten(37%), [1.0614]), table.cell(fill: rgb(0, 255, 0).lighten(37%),  [0.3197]),
    [*Deep Q-Network*], table.cell(fill: rgb(223, 31, 0).lighten(37%), [0.9169]), table.cell(fill: rgb(255, 0, 0).lighten(37%), [1.4025]), table.cell(fill: rgb(255, 0, 0).lighten(37%),  [-0.1880]),
    [*Proximal Policy Optimization*], table.cell(fill: rgb(196, 58, 0).lighten(37%), [0.8993]), table.cell(fill: rgb(171, 83, 0).lighten(37%), [1.2907]), table.cell(fill: rgb(163, 91, 0).lighten(37%),  [-0.0060]),
    [*Soft Actor-Critic*], table.cell(fill: rgb(255, 0, 0).lighten(37%), [0.9376]), table.cell(fill: rgb(164, 90, 0).lighten(37%), [1.2815]), table.cell(fill: rgb(156, 98, 0).lighten(37%),  [0.0082]),
    [*Trust Region Policy Optimization*], table.cell(fill: rgb(104, 150, 0).lighten(37%), [0.8387]), table.cell(fill: rgb(153, 101, 0).lighten(37%), [1.2663]), table.cell(fill: rgb(144, 110, 0).lighten(37%),  [0.0315]),
    table.cell(fill: rgb("#ffe0b2"), [*Ensemble*]), table.cell(fill: rgb("#ffe0b2"), [0.7646]), table.cell(fill: rgb("#ffe0b2"), [1.0609]), table.cell(fill: rgb("#ffe0b2"), [0.3202]),
  ),
  caption: [Comparación de métricas entre modelos],
)

= Resultados por Modelo


== Modelo: Advantage Actor-Critic

=== Métricas
- MAE: 0.7891
- RMSE: 1.2125
- R²: 0.1121

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
- MAE: 0.8532
- RMSE: 1.2756
- R²: 0.0172

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
- MAE: 0.7695
- RMSE: 1.0614
- R²: 0.3197

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
- MAE: 0.9169
- RMSE: 1.4025
- R²: -0.1880

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
- MAE: 0.8993
- RMSE: 1.2907
- R²: -0.0060

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
- MAE: 0.9376
- RMSE: 1.2815
- R²: 0.0082

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
- MAE: 0.8387
- RMSE: 1.2663
- R²: 0.0315

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
- MAE: 0.7646
- RMSE: 1.0609
- R²: 0.3202

=== Pesos del Ensemble
#figure(
  image("../figures/various_models/pytorch/ensemble_weights.png", width: 71%),
  caption: [Pesos optimizados para cada modelo en el ensemble],
)

= Conclusiones

El framework PYTORCH fue utilizado para entrenar 7 modelos diferentes. 
El modelo ensemble logró un MAE de 0.7646, un RMSE de 1.0609 
y un coeficiente R² de 0.3202.

