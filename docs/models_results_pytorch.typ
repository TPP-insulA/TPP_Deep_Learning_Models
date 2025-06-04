
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

    [*Advantage Actor-Critic*], table.cell(fill: rgb(151, 103, 0).lighten(37%), [0.8886]), table.cell(fill: rgb(172, 82, 0).lighten(37%), [1.2868]), table.cell(fill: rgb(164, 90, 0).lighten(37%),  [-0.0000]),
    [*Asynchronous Advantage Actor-Critic*], table.cell(fill: rgb(137, 117, 0).lighten(37%), [0.8778]), table.cell(fill: rgb(166, 88, 0).lighten(37%), [1.2781]), table.cell(fill: rgb(158, 96, 0).lighten(37%),  [0.0135]),
    [*Deep Deterministic Policy Gradient*], table.cell(fill: rgb(0, 255, 0).lighten(37%), [0.7702]), table.cell(fill: rgb(0, 255, 0).lighten(37%), [1.0566]), table.cell(fill: rgb(0, 255, 0).lighten(37%),  [0.3257]),
    [*Deep Q-Network*], table.cell(fill: rgb(202, 52, 0).lighten(37%), [0.9284]), table.cell(fill: rgb(255, 0, 0).lighten(37%), [1.3965]), table.cell(fill: rgb(255, 0, 0).lighten(37%),  [-0.1777]),
    [*Proximal Policy Optimization*], table.cell(fill: rgb(196, 58, 0).lighten(37%), [0.9234]), table.cell(fill: rgb(172, 82, 0).lighten(37%), [1.2869]), table.cell(fill: rgb(165, 89, 0).lighten(37%),  [-0.0002]),
    [*Soft Actor-Critic*], table.cell(fill: rgb(255, 0, 0).lighten(37%), [0.9690]), table.cell(fill: rgb(179, 75, 0).lighten(37%), [1.2952]), table.cell(fill: rgb(171, 83, 0).lighten(37%),  [-0.0132]),
    [*Trust Region Policy Optimization*], table.cell(fill: rgb(70, 184, 0).lighten(37%), [0.8255]), table.cell(fill: rgb(147, 107, 0).lighten(37%), [1.2535]), table.cell(fill: rgb(139, 115, 0).lighten(37%),  [0.0510]),
    table.cell(fill: rgb("#ffe0b2"), [*Ensemble*]), table.cell(fill: rgb("#ffe0b2"), [0.7702]), table.cell(fill: rgb("#ffe0b2"), [1.0566]), table.cell(fill: rgb("#ffe0b2"), [0.3257]),
  ),
  caption: [Comparación de métricas entre modelos],
)

= Resultados por Modelo


== Modelo: Advantage Actor-Critic

=== Métricas
- MAE: 0.8886
- RMSE: 1.2868
- R²: -0.0000

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
- MAE: 0.8778
- RMSE: 1.2781
- R²: 0.0135

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
- MAE: 0.7702
- RMSE: 1.0566
- R²: 0.3257

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
- MAE: 0.9284
- RMSE: 1.3965
- R²: -0.1777

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
- MAE: 0.8255
- RMSE: 1.2535
- R²: 0.0510

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
- MAE: 0.7702
- RMSE: 1.0566
- R²: 0.3257

=== Pesos del Ensemble
#figure(
  image("../figures/various_models/pytorch/ensemble_weights.png", width: 71%),
  caption: [Pesos optimizados para cada modelo en el ensemble],
)

= Conclusiones

El framework PYTORCH fue utilizado para entrenar 7 modelos diferentes. 
El modelo ensemble logró un MAE de 0.7702, un RMSE de 1.0566 
y un coeficiente R² de 0.3257.

