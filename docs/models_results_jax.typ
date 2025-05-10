
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
  #text(13pt)[#underline[Framework]: *JAX*]
]

= Resumen de Resultados

== Métricas de Rendimiento

#figure(
  table(
    columns: 4,
    align: center + horizon,
    [*Modelo*], [*MAE*], [*RMSE*], [*R²*],

    [*Advantage Actor-Critic*], table.cell(fill: rgb(127, 127, 0).lighten(37%), [1.0401]), table.cell(fill: rgb(127, 127, 0).lighten(37%), [1.2363]), table.cell(fill: rgb(127, 127, 0).lighten(37%),  [-0.0003]),
    [*Asynchronous Advantage Actor-Critic*], table.cell(fill: rgb(127, 127, 0).lighten(37%), [1.0401]), table.cell(fill: rgb(127, 127, 0).lighten(37%), [1.2363]), table.cell(fill: rgb(127, 127, 0).lighten(37%),  [-0.0003]),
    [*Deep Deterministic Policy Gradient*], table.cell(fill: rgb(127, 127, 0).lighten(37%), [1.0401]), table.cell(fill: rgb(127, 127, 0).lighten(37%), [1.2363]), table.cell(fill: rgb(127, 127, 0).lighten(37%),  [-0.0003]),
    [*Deep Q-Network*], table.cell(fill: rgb(127, 127, 0).lighten(37%), [1.0401]), table.cell(fill: rgb(127, 127, 0).lighten(37%), [1.2363]), table.cell(fill: rgb(127, 127, 0).lighten(37%),  [-0.0003]),
    [*Proximal Policy Optimization*], table.cell(fill: rgb(127, 127, 0).lighten(37%), [1.0401]), table.cell(fill: rgb(127, 127, 0).lighten(37%), [1.2363]), table.cell(fill: rgb(127, 127, 0).lighten(37%),  [-0.0003]),
    [*Soft Actor-Critic*], table.cell(fill: rgb(127, 127, 0).lighten(37%), [1.0401]), table.cell(fill: rgb(127, 127, 0).lighten(37%), [1.2363]), table.cell(fill: rgb(127, 127, 0).lighten(37%),  [-0.0003]),
    [*Trust Region Policy Optimization*], table.cell(fill: rgb(127, 127, 0).lighten(37%), [1.0401]), table.cell(fill: rgb(127, 127, 0).lighten(37%), [1.2363]), table.cell(fill: rgb(127, 127, 0).lighten(37%),  [-0.0003]),
    table.cell(fill: rgb("#ffe0b2"), [*Ensemble*]), table.cell(fill: rgb("#ffe0b2"), [1.0401]), table.cell(fill: rgb("#ffe0b2"), [1.2363]), table.cell(fill: rgb("#ffe0b2"), [-0.0003]),
  ),
  caption: [Comparación de métricas entre modelos],
)

= Resultados por Modelo


== Modelo: Advantage Actor-Critic

=== Métricas
- MAE: 1.0401
- RMSE: 1.2363
- R²: -0.0003

=== Historial de Entrenamiento
#figure(
  image("../figures/various_models/jax/individual_models/jax_a2c/training_history.png", width: 71%),
  caption: [Historial de entrenamiento para jax_a2c],
)

=== Predicciones
#figure(
  image("../figures/various_models/jax/individual_models/jax_a2c/predictions.png", width: 71%),
  caption: [Predicciones vs valores reales para jax_a2c],
)


== Modelo: Asynchronous Advantage Actor-Critic

=== Métricas
- MAE: 1.0401
- RMSE: 1.2363
- R²: -0.0003

=== Historial de Entrenamiento
#figure(
  image("../figures/various_models/jax/individual_models/jax_a3c/training_history.png", width: 71%),
  caption: [Historial de entrenamiento para jax_a3c],
)

=== Predicciones
#figure(
  image("../figures/various_models/jax/individual_models/jax_a3c/predictions.png", width: 71%),
  caption: [Predicciones vs valores reales para jax_a3c],
)


== Modelo: Deep Deterministic Policy Gradient

=== Métricas
- MAE: 1.0401
- RMSE: 1.2363
- R²: -0.0003

=== Historial de Entrenamiento
#figure(
  image("../figures/various_models/jax/individual_models/jax_ddpg/training_history.png", width: 71%),
  caption: [Historial de entrenamiento para jax_ddpg],
)

=== Predicciones
#figure(
  image("../figures/various_models/jax/individual_models/jax_ddpg/predictions.png", width: 71%),
  caption: [Predicciones vs valores reales para jax_ddpg],
)


== Modelo: Deep Q-Network

=== Métricas
- MAE: 1.0401
- RMSE: 1.2363
- R²: -0.0003

=== Historial de Entrenamiento
#figure(
  image("../figures/various_models/jax/individual_models/jax_dqn/training_history.png", width: 71%),
  caption: [Historial de entrenamiento para jax_dqn],
)

=== Predicciones
#figure(
  image("../figures/various_models/jax/individual_models/jax_dqn/predictions.png", width: 71%),
  caption: [Predicciones vs valores reales para jax_dqn],
)


== Modelo: Proximal Policy Optimization

=== Métricas
- MAE: 1.0401
- RMSE: 1.2363
- R²: -0.0003

=== Historial de Entrenamiento
#figure(
  image("../figures/various_models/jax/individual_models/jax_ppo/training_history.png", width: 71%),
  caption: [Historial de entrenamiento para jax_ppo],
)

=== Predicciones
#figure(
  image("../figures/various_models/jax/individual_models/jax_ppo/predictions.png", width: 71%),
  caption: [Predicciones vs valores reales para jax_ppo],
)


== Modelo: Soft Actor-Critic

=== Métricas
- MAE: 1.0401
- RMSE: 1.2363
- R²: -0.0003

=== Historial de Entrenamiento
#figure(
  image("../figures/various_models/jax/individual_models/jax_sac/training_history.png", width: 71%),
  caption: [Historial de entrenamiento para jax_sac],
)

=== Predicciones
#figure(
  image("../figures/various_models/jax/individual_models/jax_sac/predictions.png", width: 71%),
  caption: [Predicciones vs valores reales para jax_sac],
)


== Modelo: Trust Region Policy Optimization

=== Métricas
- MAE: 1.0401
- RMSE: 1.2363
- R²: -0.0003

=== Historial de Entrenamiento
#figure(
  image("../figures/various_models/jax/individual_models/jax_trpo/training_history.png", width: 71%),
  caption: [Historial de entrenamiento para jax_trpo],
)

=== Predicciones
#figure(
  image("../figures/various_models/jax/individual_models/jax_trpo/predictions.png", width: 71%),
  caption: [Predicciones vs valores reales para jax_trpo],
)


== Modelo Ensemble

=== Métricas
- MAE: 1.0401
- RMSE: 1.2363
- R²: -0.0003

=== Pesos del Ensemble
#figure(
  image("../figures/various_models/jax/ensemble_weights.png", width: 71%),
  caption: [Pesos optimizados para cada modelo en el ensemble],
)

= Conclusiones

El framework JAX fue utilizado para entrenar 7 modelos diferentes. 
El modelo ensemble logró un MAE de 1.0401, un RMSE de 1.2363 
y un coeficiente R² de -0.0003.

