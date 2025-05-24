
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

    [*Attention Only*], table.cell(fill: rgb(20, 234, 0).lighten(37%), [0.4894]), table.cell(fill: rgb(14, 240, 0).lighten(37%), [0.6637]), table.cell(fill: rgb(9, 245, 0).lighten(37%),  [0.4851]),
    [*Convolutional Neural Network*], table.cell(fill: rgb(33, 221, 0).lighten(37%), [0.5241]), table.cell(fill: rgb(29, 225, 0).lighten(37%), [0.7129]), table.cell(fill: rgb(19, 235, 0).lighten(37%),  [0.4060]),
    [*Feed Forward Neural Network*], table.cell(fill: rgb(7, 247, 0).lighten(37%), [0.4569]), table.cell(fill: rgb(6, 248, 0).lighten(37%), [0.6377]), table.cell(fill: rgb(4, 250, 0).lighten(37%),  [0.5248]),
    [*Gated Recurrent Unit*], table.cell(fill: rgb(4, 250, 0).lighten(37%), [0.4508]), table.cell(fill: rgb(0, 254, 0).lighten(37%), [0.6169]), table.cell(fill: rgb(0, 254, 0).lighten(37%),  [0.5553]),
    [*Long Short Term Memory*], table.cell(fill: rgb(20, 234, 0).lighten(37%), [0.4905]), table.cell(fill: rgb(15, 239, 0).lighten(37%), [0.6657]), table.cell(fill: rgb(9, 245, 0).lighten(37%),  [0.4822]),
    [*Recurrent Neural Network*], table.cell(fill: rgb(0, 255, 0).lighten(37%), [0.4386]), table.cell(fill: rgb(0, 255, 0).lighten(37%), [0.6148]), table.cell(fill: rgb(0, 255, 0).lighten(37%),  [0.5583]),
    [*TabNet*], table.cell(fill: rgb(25, 229, 0).lighten(37%), [0.5037]), table.cell(fill: rgb(26, 228, 0).lighten(37%), [0.7022]), table.cell(fill: rgb(17, 237, 0).lighten(37%),  [0.4237]),
    [*Temporal Convolutional Network*], table.cell(fill: rgb(18, 236, 0).lighten(37%), [0.4851]), table.cell(fill: rgb(13, 241, 0).lighten(37%), [0.6578]), table.cell(fill: rgb(8, 246, 0).lighten(37%),  [0.4944]),
    [*Transformer*], table.cell(fill: rgb(161, 93, 0).lighten(37%), [0.8469]), table.cell(fill: rgb(123, 131, 0).lighten(37%), [1.0200]), table.cell(fill: rgb(98, 156, 0).lighten(37%),  [-0.2160]),
    [*WaveNet*], table.cell(fill: rgb(15, 239, 0).lighten(37%), [0.4783]), table.cell(fill: rgb(9, 245, 0).lighten(37%), [0.6471]), table.cell(fill: rgb(6, 248, 0).lighten(37%),  [0.5107]),
    [*Monte Carlo Methods*], table.cell(fill: rgb(20, 234, 0).lighten(37%), [0.4896]), table.cell(fill: rgb(16, 238, 0).lighten(37%), [0.6701]), table.cell(fill: rgb(10, 244, 0).lighten(37%),  [0.4752]),
    [*Policy Iteration*], table.cell(fill: rgb(2, 252, 0).lighten(37%), [0.4456]), table.cell(fill: rgb(3, 251, 0).lighten(37%), [0.6256]), table.cell(fill: rgb(1, 253, 0).lighten(37%),  [0.5426]),
    [*Q Learning*], table.cell(fill: rgb(21, 233, 0).lighten(37%), [0.4919]), table.cell(fill: rgb(15, 239, 0).lighten(37%), [0.6662]), table.cell(fill: rgb(9, 245, 0).lighten(37%),  [0.4813]),
    [*Reinforce Monte Carlo Policy Gradient*], table.cell(fill: rgb(195, 59, 0).lighten(37%), [0.9310]), table.cell(fill: rgb(141, 113, 0).lighten(37%), [1.0780]), table.cell(fill: rgb(116, 138, 0).lighten(37%),  [-0.3580]),
    [*State-Action-Reward-State-Action*], table.cell(fill: rgb(218, 36, 0).lighten(37%), [0.9894]), table.cell(fill: rgb(159, 95, 0).lighten(37%), [1.1384]), table.cell(fill: rgb(135, 119, 0).lighten(37%),  [-0.5146]),
    [*Trust Region Policy Optimization*], table.cell(fill: rgb(221, 33, 0).lighten(37%), [0.9986]), table.cell(fill: rgb(162, 92, 0).lighten(37%), [1.1472]), table.cell(fill: rgb(138, 116, 0).lighten(37%),  [-0.5380]),
    [*Advantage Actor-Critic*], table.cell(fill: rgb(224, 30, 0).lighten(37%), [1.0059]), table.cell(fill: rgb(163, 91, 0).lighten(37%), [1.1507]), table.cell(fill: rgb(140, 114, 0).lighten(37%),  [-0.5475]),
    [*Asynchronous Advantage Actor-Critic*], table.cell(fill: rgb(214, 40, 0).lighten(37%), [0.9808]), table.cell(fill: rgb(156, 98, 0).lighten(37%), [1.1285]), table.cell(fill: rgb(132, 122, 0).lighten(37%),  [-0.4883]),
    [*Deep Deterministic Policy Gradient*], table.cell(fill: rgb(204, 50, 0).lighten(37%), [0.9536]), table.cell(fill: rgb(145, 109, 0).lighten(37%), [1.0925]), table.cell(fill: rgb(120, 134, 0).lighten(37%),  [-0.3949]),
    [*Deep Q-Network*], table.cell(fill: rgb(255, 0, 0).lighten(37%), [1.0821]), table.cell(fill: rgb(255, 0, 0).lighten(37%), [1.4495]), table.cell(fill: rgb(255, 0, 0).lighten(37%),  [-1.4554]),
    [*Proximal Policy Optimization*], table.cell(fill: rgb(219, 35, 0).lighten(37%), [0.9927]), table.cell(fill: rgb(160, 94, 0).lighten(37%), [1.1415]), table.cell(fill: rgb(136, 118, 0).lighten(37%),  [-0.5229]),
    [*Soft Actor-Critic*], table.cell(fill: rgb(198, 56, 0).lighten(37%), [0.9389]), table.cell(fill: rgb(142, 112, 0).lighten(37%), [1.0811]), table.cell(fill: rgb(117, 137, 0).lighten(37%),  [-0.3660]),
    [*Trust Region Policy Optimization*], table.cell(fill: rgb(224, 30, 0).lighten(37%), [1.0055]), table.cell(fill: rgb(164, 90, 0).lighten(37%), [1.1546]), table.cell(fill: rgb(141, 113, 0).lighten(37%),  [-0.5579]),
    table.cell(fill: rgb("#ffe0b2"), [*Ensemble*]), table.cell(fill: rgb("#ffe0b2"), [0.4254]), table.cell(fill: rgb("#ffe0b2"), [0.5840]), table.cell(fill: rgb("#ffe0b2"), [0.6014]),
  ),
  caption: [Comparación de métricas entre modelos],
)

= Resultados por Modelo


== Modelo: Attention Only

=== Métricas
- MAE: 0.4894
- RMSE: 0.6637
- R²: 0.4851

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
- MAE: 0.5241
- RMSE: 0.7129
- R²: 0.4060

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
- MAE: 0.4569
- RMSE: 0.6377
- R²: 0.5248

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
- MAE: 0.4508
- RMSE: 0.6169
- R²: 0.5553

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
- MAE: 0.4905
- RMSE: 0.6657
- R²: 0.4822

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
- MAE: 0.4386
- RMSE: 0.6148
- R²: 0.5583

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
- MAE: 0.5037
- RMSE: 0.7022
- R²: 0.4237

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
- MAE: 0.4851
- RMSE: 0.6578
- R²: 0.4944

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
- MAE: 0.8469
- RMSE: 1.0200
- R²: -0.2160

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
- MAE: 0.4783
- RMSE: 0.6471
- R²: 0.5107

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
- MAE: 0.4896
- RMSE: 0.6701
- R²: 0.4752

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
- MAE: 0.4456
- RMSE: 0.6256
- R²: 0.5426

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
- MAE: 0.4919
- RMSE: 0.6662
- R²: 0.4813

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
- MAE: 0.9310
- RMSE: 1.0780
- R²: -0.3580

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
- MAE: 0.9894
- RMSE: 1.1384
- R²: -0.5146

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
- MAE: 0.9986
- RMSE: 1.1472
- R²: -0.5380

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
- MAE: 1.0059
- RMSE: 1.1507
- R²: -0.5475

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
- MAE: 0.9808
- RMSE: 1.1285
- R²: -0.4883

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
- MAE: 0.9536
- RMSE: 1.0925
- R²: -0.3949

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
- MAE: 1.0821
- RMSE: 1.4495
- R²: -1.4554

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
- MAE: 0.9927
- RMSE: 1.1415
- R²: -0.5229

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
- MAE: 0.9389
- RMSE: 1.0811
- R²: -0.3660

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
- MAE: 1.0055
- RMSE: 1.1546
- R²: -0.5579

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
- MAE: 0.4254
- RMSE: 0.5840
- R²: 0.6014

=== Pesos del Ensemble
#figure(
  image("../figures/various_models/pytorch/ensemble_weights.png", width: 71%),
  caption: [Pesos optimizados para cada modelo en el ensemble],
)

= Conclusiones

El framework PYTORCH fue utilizado para entrenar 23 modelos diferentes. 
El modelo ensemble logró un MAE de 0.4254, un RMSE de 0.5840 
y un coeficiente R² de 0.6014.

