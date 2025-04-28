
#set page(
  margin: 2cm,
  numbering: "1 de 1",
)

#set text(font: "New Computer Modern")
#set heading(numbering: "1.")
#show heading: set block(above: 1.4em, below: 1em)

#align(center)[
  #text(17pt)[*Resultados de Entrenamiento de Modelos*]
  #v(0.5em)
  #text(13pt)[#underline[Framework]: *PyTorch*]
]

#set table(
  fill: (x, y) =>
    if x == 0 {
      rgb("#247fff").lighten(40%)
    } else if y == 0 {
      rgb("#e7da27").lighten(40%)
    },
  align: right,
)



= Resumen de Resultados

== Métricas de Rendimiento

#figure(
  table(
    columns: 4,
    align: center + horizon,
    [], [*MAE*], [*RMSE*], [*R²*],

    [*Attention Only*], [0.2113], [0.3427], [0.9232],
    [*Convolutional Neural Network*], [0.2032], [0.3258], [0.9305],
    [*Feed Forward Neural Network*], [0.2054], [0.3418], [0.9235],
    [*Gated Recurrent Unit*], [0.2037], [0.3450], [0.9221],
    [*Long Short Term Memory*], [0.2210], [0.3661], [0.9123],
    [*Recurrent Neural Network*], [0.2011], [0.3404], [0.9242],
    [*TabNet*], [0.2349], [0.3636], [0.9135],
    [*Temporal Convolutional Network*], [0.2092], [0.3480], [0.9207],
    [*Transformer*], [0.9152], [1.1675], [0.1079],
    [*WaveNet*], [0.2385], [0.3905], [0.9002],
    [*Monte Carlo Methods*], [0.2603], [0.3947], [0.8981],
    [*Policy Iteration*], [0.2303], [0.3657], [0.9125],
    [*Q Learning*], [0.2520], [0.3882], [0.9014],
    [*Reinforce Monte Carlo Policy Gradient*], [0.5814], [0.8105], [0.5701],
    [*State-Action-Reward-State-Action*], [1.0408], [1.2362], [-0.0001],
    [*Value Iteration*], [1.0401], [1.2363], [-0.0003],
    [*Advantage Actor-Critic*], [1.0402], [1.2352], [0.0014],
    [*Asynchronous Advantage Actor-Critic*], [1.0058], [1.2148], [0.0342],
    [*Deep Deterministic Policy Gradient*], [0.6180], [0.8264], [0.5530],
    [*Deep Q-Network*], [1.0657], [1.3798], [-0.2461],
    [*Proximal Policy Optimization*], [1.0521], [1.2391], [-0.0048],
    [*Soft Actor-Critic*], [1.0386], [1.2354], [0.0011],
    [*Trust Region Policy Optimization*], [9.7705], [10.9991], [-78.1776],
    [*Ensemble*], [0.1942], [0.3217], [0.9323],
  ),
  caption: [Comparación de métricas entre modelos],
)

= Resultados por Modelo


== Modelo: pt_attention_only

=== Métricas
- MAE: 0.2113
- RMSE: 0.3427
- R²: 0.9232

=== Historial de Entrenamiento
#figure(
  image("../figures/various_models/pytorch/individual_models/pt_attention_only/training_history.png", width: 71%),
  caption: [Historial de entrenamiento para pt_attention_only],
)

=== Predicciones
#figure(
  image("../figures/various_models/pytorch/individual_models/pt_attention_only/predictions.png", width: 71%),
  caption: [Predicciones vs valores reales para pt_attention_only],
)


== Modelo: pt_cnn

=== Métricas
- MAE: 0.2032
- RMSE: 0.3258
- R²: 0.9305

=== Historial de Entrenamiento
#figure(
  image("../figures/various_models/pytorch/individual_models/pt_cnn/training_history.png", width: 71%),
  caption: [Historial de entrenamiento para pt_cnn],
)

=== Predicciones
#figure(
  image("../figures/various_models/pytorch/individual_models/pt_cnn/predictions.png", width: 71%),
  caption: [Predicciones vs valores reales para pt_cnn],
)


== Modelo: pt_fnn

=== Métricas
- MAE: 0.2054
- RMSE: 0.3418
- R²: 0.9235

=== Historial de Entrenamiento
#figure(
  image("../figures/various_models/pytorch/individual_models/pt_fnn/training_history.png", width: 71%),
  caption: [Historial de entrenamiento para pt_fnn],
)

=== Predicciones
#figure(
  image("../figures/various_models/pytorch/individual_models/pt_fnn/predictions.png", width: 71%),
  caption: [Predicciones vs valores reales para pt_fnn],
)


== Modelo: pt_gru

=== Métricas
- MAE: 0.2037
- RMSE: 0.3450
- R²: 0.9221

=== Historial de Entrenamiento
#figure(
  image("../figures/various_models/pytorch/individual_models/pt_gru/training_history.png", width: 71%),
  caption: [Historial de entrenamiento para pt_gru],
)

=== Predicciones
#figure(
  image("../figures/various_models/pytorch/individual_models/pt_gru/predictions.png", width: 71%),
  caption: [Predicciones vs valores reales para pt_gru],
)


== Modelo: pt_lstm

=== Métricas
- MAE: 0.2210
- RMSE: 0.3661
- R²: 0.9123

=== Historial de Entrenamiento
#figure(
  image("../figures/various_models/pytorch/individual_models/pt_lstm/training_history.png", width: 71%),
  caption: [Historial de entrenamiento para pt_lstm],
)

=== Predicciones
#figure(
  image("../figures/various_models/pytorch/individual_models/pt_lstm/predictions.png", width: 71%),
  caption: [Predicciones vs valores reales para pt_lstm],
)


== Modelo: pt_rnn

=== Métricas
- MAE: 0.2011
- RMSE: 0.3404
- R²: 0.9242

=== Historial de Entrenamiento
#figure(
  image("../figures/various_models/pytorch/individual_models/pt_rnn/training_history.png", width: 71%),
  caption: [Historial de entrenamiento para pt_rnn],
)

=== Predicciones
#figure(
  image("../figures/various_models/pytorch/individual_models/pt_rnn/predictions.png", width: 71%),
  caption: [Predicciones vs valores reales para pt_rnn],
)


== Modelo: pt_tabnet

=== Métricas
- MAE: 0.2349
- RMSE: 0.3636
- R²: 0.9135

=== Historial de Entrenamiento
#figure(
  image("../figures/various_models/pytorch/individual_models/pt_tabnet/training_history.png", width: 71%),
  caption: [Historial de entrenamiento para pt_tabnet],
)

=== Predicciones
#figure(
  image("../figures/various_models/pytorch/individual_models/pt_tabnet/predictions.png", width: 71%),
  caption: [Predicciones vs valores reales para pt_tabnet],
)


== Modelo: pt_tcn

=== Métricas
- MAE: 0.2092
- RMSE: 0.3480
- R²: 0.9207

=== Historial de Entrenamiento
#figure(
  image("../figures/various_models/pytorch/individual_models/pt_tcn/training_history.png", width: 71%),
  caption: [Historial de entrenamiento para pt_tcn],
)

=== Predicciones
#figure(
  image("../figures/various_models/pytorch/individual_models/pt_tcn/predictions.png", width: 71%),
  caption: [Predicciones vs valores reales para pt_tcn],
)


== Modelo: pt_transformer

=== Métricas
- MAE: 0.9152
- RMSE: 1.1675
- R²: 0.1079

=== Historial de Entrenamiento
#figure(
  image("../figures/various_models/pytorch/individual_models/pt_transformer/training_history.png", width: 71%),
  caption: [Historial de entrenamiento para pt_transformer],
)

=== Predicciones
#figure(
  image("../figures/various_models/pytorch/individual_models/pt_transformer/predictions.png", width: 71%),
  caption: [Predicciones vs valores reales para pt_transformer],
)


== Modelo: pt_wavenet

=== Métricas
- MAE: 0.2385
- RMSE: 0.3905
- R²: 0.9002

=== Historial de Entrenamiento
#figure(
  image("../figures/various_models/pytorch/individual_models/pt_wavenet/training_history.png", width: 71%),
  caption: [Historial de entrenamiento para pt_wavenet],
)

=== Predicciones
#figure(
  image("../figures/various_models/pytorch/individual_models/pt_wavenet/predictions.png", width: 71%),
  caption: [Predicciones vs valores reales para pt_wavenet],
)


== Modelo: pt_monte_carlo

=== Métricas
- MAE: 0.2603
- RMSE: 0.3947
- R²: 0.8981

=== Historial de Entrenamiento
#figure(
  image("../figures/various_models/pytorch/individual_models/pt_monte_carlo/training_history.png", width: 71%),
  caption: [Historial de entrenamiento para pt_monte_carlo],
)

=== Predicciones
#figure(
  image("../figures/various_models/pytorch/individual_models/pt_monte_carlo/predictions.png", width: 71%),
  caption: [Predicciones vs valores reales para pt_monte_carlo],
)


== Modelo: pt_policy_iteration

=== Métricas
- MAE: 0.2303
- RMSE: 0.3657
- R²: 0.9125

=== Historial de Entrenamiento
#figure(
  image("../figures/various_models/pytorch/individual_models/pt_policy_iteration/training_history.png", width: 71%),
  caption: [Historial de entrenamiento para pt_policy_iteration],
)

=== Predicciones
#figure(
  image("../figures/various_models/pytorch/individual_models/pt_policy_iteration/predictions.png", width: 71%),
  caption: [Predicciones vs valores reales para pt_policy_iteration],
)


== Modelo: pt_q_learning

=== Métricas
- MAE: 0.2520
- RMSE: 0.3882
- R²: 0.9014

=== Historial de Entrenamiento
#figure(
  image("../figures/various_models/pytorch/individual_models/pt_q_learning/training_history.png", width: 71%),
  caption: [Historial de entrenamiento para pt_q_learning],
)

=== Predicciones
#figure(
  image("../figures/various_models/pytorch/individual_models/pt_q_learning/predictions.png", width: 71%),
  caption: [Predicciones vs valores reales para pt_q_learning],
)


== Modelo: pt_reinforce_mcpg

=== Métricas
- MAE: 0.5814
- RMSE: 0.8105
- R²: 0.5701

=== Historial de Entrenamiento
#figure(
  image("../figures/various_models/pytorch/individual_models/pt_reinforce_mcpg/training_history.png", width: 71%),
  caption: [Historial de entrenamiento para pt_reinforce_mcpg],
)

=== Predicciones
#figure(
  image("../figures/various_models/pytorch/individual_models/pt_reinforce_mcpg/predictions.png", width: 71%),
  caption: [Predicciones vs valores reales para pt_reinforce_mcpg],
)


== Modelo: pt_sarsa

=== Métricas
- MAE: 1.0408
- RMSE: 1.2362
- R²: -0.0001

=== Historial de Entrenamiento
#figure(
  image("../figures/various_models/pytorch/individual_models/pt_sarsa/training_history.png", width: 71%),
  caption: [Historial de entrenamiento para pt_sarsa],
)

=== Predicciones
#figure(
  image("../figures/various_models/pytorch/individual_models/pt_sarsa/predictions.png", width: 71%),
  caption: [Predicciones vs valores reales para pt_sarsa],
)


== Modelo: pt_value_iteration

=== Métricas
- MAE: 1.0401
- RMSE: 1.2363
- R²: -0.0003

=== Historial de Entrenamiento
#figure(
  image("../figures/various_models/pytorch/individual_models/pt_value_iteration/training_history.png", width: 71%),
  caption: [Historial de entrenamiento para pt_value_iteration],
)

=== Predicciones
#figure(
  image("../figures/various_models/pytorch/individual_models/pt_value_iteration/predictions.png", width: 71%),
  caption: [Predicciones vs valores reales para pt_value_iteration],
)


== Modelo: pt_a2c

=== Métricas
- MAE: 1.0402
- RMSE: 1.2352
- R²: 0.0014

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


== Modelo: pt_a3c

=== Métricas
- MAE: 1.0058
- RMSE: 1.2148
- R²: 0.0342

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


== Modelo: pt_ddpg

=== Métricas
- MAE: 0.6180
- RMSE: 0.8264
- R²: 0.5530

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


== Modelo: pt_dqn

=== Métricas
- MAE: 1.0657
- RMSE: 1.3798
- R²: -0.2461

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


== Modelo: pt_ppo

=== Métricas
- MAE: 1.0521
- RMSE: 1.2391
- R²: -0.0048

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


== Modelo: pt_sac

=== Métricas
- MAE: 1.0386
- RMSE: 1.2354
- R²: 0.0011

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


== Modelo: pt_trpo

=== Métricas
- MAE: 9.7705
- RMSE: 10.9991
- R²: -78.1776

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
- MAE: 0.1942
- RMSE: 0.3217
- R²: 0.9323

=== Pesos del Ensemble
#figure(
  image("../figures/various_models/pytorch/ensemble_weights.png", width: 71%),
  caption: [Pesos optimizados para cada modelo en el ensemble],
)

= Conclusiones

El framework PYTORCH fue utilizado para entrenar 23 modelos diferentes. 
El modelo ensemble logró un MAE de 0.1942, un RMSE de 0.3217 
y un coeficiente R² de 0.9323.

