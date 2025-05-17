
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

    [*Trust Region Policy Optimization*], table.cell(fill: rgb(11, 243, 0).lighten(37%), [1.5972]), table.cell(fill: rgb(9, 245, 0).lighten(37%), [1.8432]), table.cell(fill: rgb(3, 251, 0).lighten(37%),  [-3.0150]),
    [*Advantage Actor-Critic*], table.cell(fill: rgb(14, 240, 0).lighten(37%), [1.6517]), table.cell(fill: rgb(15, 239, 0).lighten(37%), [1.9790]), table.cell(fill: rgb(6, 248, 0).lighten(37%),  [-3.6284]),
    [*Asynchronous Advantage Actor-Critic*], table.cell(fill: rgb(5, 249, 0).lighten(37%), [1.4887]), table.cell(fill: rgb(6, 248, 0).lighten(37%), [1.7557]), table.cell(fill: rgb(2, 252, 0).lighten(37%),  [-2.6431]),
    [*Deep Deterministic Policy Gradient*], table.cell(fill: rgb(25, 229, 0).lighten(37%), [1.8676]), table.cell(fill: rgb(56, 198, 0).lighten(37%), [2.9604]), table.cell(fill: rgb(28, 226, 0).lighten(37%),  [-9.3571]),
    [*Deep Q-Network*], table.cell(fill: rgb(3, 251, 0).lighten(37%), [1.4572]), table.cell(fill: rgb(4, 250, 0).lighten(37%), [1.7162]), table.cell(fill: rgb(1, 253, 0).lighten(37%),  [-2.4810]),
    [*Proximal Policy Optimization*], table.cell(fill: rgb(23, 231, 0).lighten(37%), [1.8231]), table.cell(fill: rgb(27, 227, 0).lighten(37%), [2.2551]), table.cell(fill: rgb(11, 243, 0).lighten(37%),  [-5.0103]),
    [*Soft Actor-Critic*], table.cell(fill: rgb(255, 0, 0).lighten(37%), [6.1716]), table.cell(fill: rgb(255, 0, 0).lighten(37%), [7.6508]), table.cell(fill: rgb(255, 0, 0).lighten(37%),  [-68.1776]),
    [*Trust Region Policy Optimization*], table.cell(fill: rgb(0, 255, 0).lighten(37%), [1.3836]), table.cell(fill: rgb(0, 255, 0).lighten(37%), [1.6120]), table.cell(fill: rgb(0, 255, 0).lighten(37%),  [-2.0709]),
    table.cell(fill: rgb("#ffe0b2"), [*Ensemble*]), table.cell(fill: rgb("#ffe0b2"), [0.7973]), table.cell(fill: rgb("#ffe0b2"), [0.9758]), table.cell(fill: rgb("#ffe0b2"), [-0.1253]),
  ),
  caption: [Comparación de métricas entre modelos],
)

= Resultados por Modelo


== Modelo: Trust Region Policy Optimization

=== Métricas
- MAE: 1.5972
- RMSE: 1.8432
- R²: -3.0150

=== Historial de Entrenamiento
#figure(
  image("../figures/various_models/tensorflow/individual_models/tf_value_iteration/training_history.png", width: 71%),
  caption: [Historial de entrenamiento para tf_value_iteration],
)

=== Predicciones
#figure(
  image("../figures/various_models/tensorflow/individual_models/tf_value_iteration/predictions.png", width: 71%),
  caption: [Predicciones vs valores reales para tf_value_iteration],
)


== Modelo: Advantage Actor-Critic

=== Métricas
- MAE: 1.6517
- RMSE: 1.9790
- R²: -3.6284

=== Historial de Entrenamiento
#figure(
  image("../figures/various_models/tensorflow/individual_models/tf_a2c/training_history.png", width: 71%),
  caption: [Historial de entrenamiento para tf_a2c],
)

=== Predicciones
#figure(
  image("../figures/various_models/tensorflow/individual_models/tf_a2c/predictions.png", width: 71%),
  caption: [Predicciones vs valores reales para tf_a2c],
)


== Modelo: Asynchronous Advantage Actor-Critic

=== Métricas
- MAE: 1.4887
- RMSE: 1.7557
- R²: -2.6431

=== Historial de Entrenamiento
#figure(
  image("../figures/various_models/tensorflow/individual_models/tf_a3c/training_history.png", width: 71%),
  caption: [Historial de entrenamiento para tf_a3c],
)

=== Predicciones
#figure(
  image("../figures/various_models/tensorflow/individual_models/tf_a3c/predictions.png", width: 71%),
  caption: [Predicciones vs valores reales para tf_a3c],
)


== Modelo: Deep Deterministic Policy Gradient

=== Métricas
- MAE: 1.8676
- RMSE: 2.9604
- R²: -9.3571

=== Historial de Entrenamiento
#figure(
  image("../figures/various_models/tensorflow/individual_models/tf_ddpg/training_history.png", width: 71%),
  caption: [Historial de entrenamiento para tf_ddpg],
)

=== Predicciones
#figure(
  image("../figures/various_models/tensorflow/individual_models/tf_ddpg/predictions.png", width: 71%),
  caption: [Predicciones vs valores reales para tf_ddpg],
)


== Modelo: Deep Q-Network

=== Métricas
- MAE: 1.4572
- RMSE: 1.7162
- R²: -2.4810

=== Historial de Entrenamiento
#figure(
  image("../figures/various_models/tensorflow/individual_models/tf_dqn/training_history.png", width: 71%),
  caption: [Historial de entrenamiento para tf_dqn],
)

=== Predicciones
#figure(
  image("../figures/various_models/tensorflow/individual_models/tf_dqn/predictions.png", width: 71%),
  caption: [Predicciones vs valores reales para tf_dqn],
)


== Modelo: Proximal Policy Optimization

=== Métricas
- MAE: 1.8231
- RMSE: 2.2551
- R²: -5.0103

=== Historial de Entrenamiento
#figure(
  image("../figures/various_models/tensorflow/individual_models/tf_ppo/training_history.png", width: 71%),
  caption: [Historial de entrenamiento para tf_ppo],
)

=== Predicciones
#figure(
  image("../figures/various_models/tensorflow/individual_models/tf_ppo/predictions.png", width: 71%),
  caption: [Predicciones vs valores reales para tf_ppo],
)


== Modelo: Soft Actor-Critic

=== Métricas
- MAE: 6.1716
- RMSE: 7.6508
- R²: -68.1776

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
- MAE: 1.3836
- RMSE: 1.6120
- R²: -2.0709

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
- MAE: 0.7973
- RMSE: 0.9758
- R²: -0.1253

=== Pesos del Ensemble
#figure(
  image("../figures/various_models/tensorflow/ensemble_weights.png", width: 71%),
  caption: [Pesos optimizados para cada modelo en el ensemble],
)

= Conclusiones

El framework TENSORFLOW fue utilizado para entrenar 8 modelos diferentes. 
El modelo ensemble logró un MAE de 0.7973, un RMSE de 0.9758 
y un coeficiente R² de -0.1253.

