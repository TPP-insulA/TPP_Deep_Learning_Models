
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

    [*Attention Only*], table.cell(fill: rgb(5, 249, 0).lighten(37%), [0.1914]), table.cell(fill: rgb(1, 253, 0).lighten(37%), [0.3092]), table.cell(fill: rgb(0, 254, 0).lighten(37%),  [0.8870]),
    [*Convolutional Neural Network*], table.cell(fill: rgb(0, 255, 0).lighten(37%), [0.1773]), table.cell(fill: rgb(3, 251, 0).lighten(37%), [0.3143]), table.cell(fill: rgb(1, 253, 0).lighten(37%),  [0.8833]),
    [*Feed Forward Neural Network*], table.cell(fill: rgb(11, 243, 0).lighten(37%), [0.2076]), table.cell(fill: rgb(11, 243, 0).lighten(37%), [0.3434]), table.cell(fill: rgb(5, 249, 0).lighten(37%),  [0.8606]),
    [*Gated Recurrent Unit*], table.cell(fill: rgb(14, 240, 0).lighten(37%), [0.2162]), table.cell(fill: rgb(8, 246, 0).lighten(37%), [0.3317]), table.cell(fill: rgb(3, 251, 0).lighten(37%),  [0.8699]),
    [*Long Short Term Memory*], table.cell(fill: rgb(23, 231, 0).lighten(37%), [0.2400]), table.cell(fill: rgb(12, 242, 0).lighten(37%), [0.3451]), table.cell(fill: rgb(5, 249, 0).lighten(37%),  [0.8593]),
    [*Recurrent Neural Network*], table.cell(fill: rgb(2, 252, 0).lighten(37%), [0.1846]), table.cell(fill: rgb(0, 255, 0).lighten(37%), [0.3027]), table.cell(fill: rgb(0, 255, 0).lighten(37%),  [0.8917]),
    [*TabNet*], table.cell(fill: rgb(43, 211, 0).lighten(37%), [0.2948]), table.cell(fill: rgb(30, 224, 0).lighten(37%), [0.4104]), table.cell(fill: rgb(14, 240, 0).lighten(37%),  [0.8009]),
    [*Temporal Convolutional Network*], table.cell(fill: rgb(9, 245, 0).lighten(37%), [0.2018]), table.cell(fill: rgb(3, 251, 0).lighten(37%), [0.3142]), table.cell(fill: rgb(1, 253, 0).lighten(37%),  [0.8833]),
    [*Transformer*], table.cell(fill: rgb(223, 31, 0).lighten(37%), [0.7782]), table.cell(fill: rgb(183, 71, 0).lighten(37%), [0.9426]), table.cell(fill: rgb(152, 102, 0).lighten(37%),  [-0.0501]),
    [*WaveNet*], table.cell(fill: rgb(16, 238, 0).lighten(37%), [0.2218]), table.cell(fill: rgb(9, 245, 0).lighten(37%), [0.3353]), table.cell(fill: rgb(3, 251, 0).lighten(37%),  [0.8671]),
    [*Monte Carlo Methods*], table.cell(fill: rgb(29, 225, 0).lighten(37%), [0.2570]), table.cell(fill: rgb(13, 241, 0).lighten(37%), [0.3513]), table.cell(fill: rgb(6, 248, 0).lighten(37%),  [0.8542]),
    [*Policy Iteration*], table.cell(fill: rgb(6, 248, 0).lighten(37%), [0.1948]), table.cell(fill: rgb(2, 252, 0).lighten(37%), [0.3126]), table.cell(fill: rgb(1, 253, 0).lighten(37%),  [0.8845]),
    [*Q Learning*], table.cell(fill: rgb(12, 242, 0).lighten(37%), [0.2119]), table.cell(fill: rgb(8, 246, 0).lighten(37%), [0.3322]), table.cell(fill: rgb(3, 251, 0).lighten(37%),  [0.8696]),
    [*Reinforce Monte Carlo Policy Gradient*], table.cell(fill: rgb(154, 100, 0).lighten(37%), [0.5943]), table.cell(fill: rgb(150, 104, 0).lighten(37%), [0.8265]), table.cell(fill: rgb(113, 141, 0).lighten(37%),  [0.1927]),
    [*State-Action-Reward-State-Action*], table.cell(fill: rgb(232, 22, 0).lighten(37%), [0.8023]), table.cell(fill: rgb(187, 67, 0).lighten(37%), [0.9550]), table.cell(fill: rgb(157, 97, 0).lighten(37%),  [-0.0778]),
    [*Trust Region Policy Optimization*], table.cell(fill: rgb(233, 21, 0).lighten(37%), [0.8066]), table.cell(fill: rgb(188, 66, 0).lighten(37%), [0.9602]), table.cell(fill: rgb(159, 95, 0).lighten(37%),  [-0.0896]),
    [*Advantage Actor-Critic*], table.cell(fill: rgb(231, 23, 0).lighten(37%), [0.8016]), table.cell(fill: rgb(186, 68, 0).lighten(37%), [0.9545]), table.cell(fill: rgb(157, 97, 0).lighten(37%),  [-0.0767]),
    [*Asynchronous Advantage Actor-Critic*], table.cell(fill: rgb(241, 13, 0).lighten(37%), [0.8273]), table.cell(fill: rgb(194, 60, 0).lighten(37%), [0.9823]), table.cell(fill: rgb(167, 87, 0).lighten(37%),  [-0.1404]),
    [*Deep Deterministic Policy Gradient*], table.cell(fill: rgb(171, 83, 0).lighten(37%), [0.6386]), table.cell(fill: rgb(154, 100, 0).lighten(37%), [0.8430]), table.cell(fill: rgb(118, 136, 0).lighten(37%),  [0.1602]),
    [*Deep Q-Network*], table.cell(fill: rgb(255, 0, 0).lighten(37%), [0.8640]), table.cell(fill: rgb(255, 0, 0).lighten(37%), [1.1920]), table.cell(fill: rgb(255, 0, 0).lighten(37%),  [-0.6792]),
    [*Proximal Policy Optimization*], table.cell(fill: rgb(232, 22, 0).lighten(37%), [0.8045]), table.cell(fill: rgb(187, 67, 0).lighten(37%), [0.9577]), table.cell(fill: rgb(158, 96, 0).lighten(37%),  [-0.0840]),
    [*Soft Actor-Critic*], table.cell(fill: rgb(221, 33, 0).lighten(37%), [0.7725]), table.cell(fill: rgb(178, 76, 0).lighten(37%), [0.9238]), table.cell(fill: rgb(146, 108, 0).lighten(37%),  [-0.0086]),
    [*Trust Region Policy Optimization*], table.cell(fill: rgb(229, 25, 0).lighten(37%), [0.7941]), table.cell(fill: rgb(184, 70, 0).lighten(37%), [0.9471]), table.cell(fill: rgb(154, 100, 0).lighten(37%),  [-0.0600]),
    table.cell(fill: rgb("#ffe0b2"), [*Ensemble*]), table.cell(fill: rgb("#ffe0b2"), [0.1738]), table.cell(fill: rgb("#ffe0b2"), [0.2901]), table.cell(fill: rgb("#ffe0b2"), [0.9006]),
  ),
  caption: [Comparación de métricas entre modelos],
)

= Resultados por Modelo


== Modelo: Attention Only

=== Métricas
- MAE: 0.1914
- RMSE: 0.3092
- R²: 0.8870

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


== Modelo: Convolutional Neural Network

=== Métricas
- MAE: 0.1773
- RMSE: 0.3143
- R²: 0.8833

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


== Modelo: Feed Forward Neural Network

=== Métricas
- MAE: 0.2076
- RMSE: 0.3434
- R²: 0.8606

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


== Modelo: Gated Recurrent Unit

=== Métricas
- MAE: 0.2162
- RMSE: 0.3317
- R²: 0.8699

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


== Modelo: Long Short Term Memory

=== Métricas
- MAE: 0.2400
- RMSE: 0.3451
- R²: 0.8593

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


== Modelo: Recurrent Neural Network

=== Métricas
- MAE: 0.1846
- RMSE: 0.3027
- R²: 0.8917

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


== Modelo: TabNet

=== Métricas
- MAE: 0.2948
- RMSE: 0.4104
- R²: 0.8009

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


== Modelo: Temporal Convolutional Network

=== Métricas
- MAE: 0.2018
- RMSE: 0.3142
- R²: 0.8833

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


== Modelo: Transformer

=== Métricas
- MAE: 0.7782
- RMSE: 0.9426
- R²: -0.0501

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


== Modelo: WaveNet

=== Métricas
- MAE: 0.2218
- RMSE: 0.3353
- R²: 0.8671

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


== Modelo: Monte Carlo Methods

=== Métricas
- MAE: 0.2570
- RMSE: 0.3513
- R²: 0.8542

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


== Modelo: Policy Iteration

=== Métricas
- MAE: 0.1948
- RMSE: 0.3126
- R²: 0.8845

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


== Modelo: Q Learning

=== Métricas
- MAE: 0.2119
- RMSE: 0.3322
- R²: 0.8696

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


== Modelo: Reinforce Monte Carlo Policy Gradient

=== Métricas
- MAE: 0.5943
- RMSE: 0.8265
- R²: 0.1927

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


== Modelo: State-Action-Reward-State-Action

=== Métricas
- MAE: 0.8023
- RMSE: 0.9550
- R²: -0.0778

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


== Modelo: Trust Region Policy Optimization

=== Métricas
- MAE: 0.8066
- RMSE: 0.9602
- R²: -0.0896

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


== Modelo: Advantage Actor-Critic

=== Métricas
- MAE: 0.8016
- RMSE: 0.9545
- R²: -0.0767

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
- MAE: 0.8273
- RMSE: 0.9823
- R²: -0.1404

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
- MAE: 0.7725
- RMSE: 0.9238
- R²: -0.0086

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
- MAE: 0.1738
- RMSE: 0.2901
- R²: 0.9006

=== Pesos del Ensemble
#figure(
  image("../figures/various_models/pytorch/ensemble_weights.png", width: 71%),
  caption: [Pesos optimizados para cada modelo en el ensemble],
)

= Conclusiones

El framework PYTORCH fue utilizado para entrenar 23 modelos diferentes. 
El modelo ensemble logró un MAE de 0.1738, un RMSE de 0.2901 
y un coeficiente R² de 0.9006.

