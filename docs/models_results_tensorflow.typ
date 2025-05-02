
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

    [*Advantage Actor-Critic*], table.cell(fill: rgb(160, 94, 0).lighten(37%), [4.7057]), table.cell(fill: rgb(130, 124, 0).lighten(37%), [4.8282]), table.cell(fill: rgb(83, 171, 0).lighten(37%),  [-14.2565]),
    [*Asynchronous Advantage Actor-Critic*], table.cell(fill: rgb(17, 237, 0).lighten(37%), [1.4403]), table.cell(fill: rgb(23, 231, 0).lighten(37%), [1.8834]), table.cell(fill: rgb(7, 247, 0).lighten(37%),  [-1.3216]),
    [*Deep Deterministic Policy Gradient*], table.cell(fill: rgb(0, 255, 0).lighten(37%), [1.0401]), table.cell(fill: rgb(0, 255, 0).lighten(37%), [1.2363]), table.cell(fill: rgb(0, 255, 0).lighten(37%),  [-0.0003]),
    [*Deep Q-Network*], table.cell(fill: rgb(31, 223, 0).lighten(37%), [1.7604]), table.cell(fill: rgb(32, 222, 0).lighten(37%), [2.1347]), table.cell(fill: rgb(11, 243, 0).lighten(37%),  [-1.9824]),
    [*Proximal Policy Optimization*], table.cell(fill: rgb(13, 241, 0).lighten(37%), [1.3416]), table.cell(fill: rgb(15, 239, 0).lighten(37%), [1.6525]), table.cell(fill: rgb(4, 250, 0).lighten(37%),  [-0.7873]),
    [*Soft Actor-Critic*], table.cell(fill: rgb(255, 0, 0).lighten(37%), [6.8640]), table.cell(fill: rgb(255, 0, 0).lighten(37%), [8.2453]), table.cell(fill: rgb(255, 0, 0).lighten(37%),  [-43.4945]),
    [*Trust Region Policy Optimization*], table.cell(fill: rgb(27, 227, 0).lighten(37%), [1.6608]), table.cell(fill: rgb(24, 230, 0).lighten(37%), [1.9010]), table.cell(fill: rgb(8, 246, 0).lighten(37%),  [-1.3651]),
    table.cell(fill: rgb("#ffe0b2"), [*Ensemble*]), table.cell(fill: rgb("#ffe0b2"), [1.0366]), table.cell(fill: rgb("#ffe0b2"), [1.2359]), table.cell(fill: rgb("#ffe0b2"), [0.0003]),
  ),
  caption: [Comparación de métricas entre modelos],
)

= Resultados por Modelo


== Modelo: Advantage Actor-Critic

=== Métricas
- MAE: 4.7057
- RMSE: 4.8282
- R²: -14.2565

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
- MAE: 1.4403
- RMSE: 1.8834
- R²: -1.3216

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
- MAE: 1.0401
- RMSE: 1.2363
- R²: -0.0003

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
- MAE: 1.7604
- RMSE: 2.1347
- R²: -1.9824

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
- MAE: 1.3416
- RMSE: 1.6525
- R²: -0.7873

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
- MAE: 1.0366
- RMSE: 1.2359
- R²: 0.0003

=== Pesos del Ensemble
#figure(
  image("../figures/various_models/tensorflow/ensemble_weights.png", width: 71%),
  caption: [Pesos optimizados para cada modelo en el ensemble],
)

= Conclusiones

El framework TENSORFLOW fue utilizado para entrenar 7 modelos diferentes. 
El modelo ensemble logró un MAE de 1.0366, un RMSE de 1.2359 
y un coeficiente R² de 0.0003.

