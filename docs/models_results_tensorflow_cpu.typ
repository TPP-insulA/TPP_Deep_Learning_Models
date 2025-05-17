
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

    [*Attention Only*], table.cell(fill: rgb(4, 250, 0).lighten(37%), [0.3416]), table.cell(fill: rgb(4, 250, 0).lighten(37%), [0.4854]), table.cell(fill: rgb(0, 254, 0).lighten(37%),  [0.8458]),
    [*Convolutional Neural Network*], table.cell(fill: rgb(0, 254, 0).lighten(37%), [0.1977]), table.cell(fill: rgb(0, 254, 0).lighten(37%), [0.3170]), table.cell(fill: rgb(0, 254, 0).lighten(37%),  [0.9342]),
    [*Feed Forward Neural Network*], table.cell(fill: rgb(0, 254, 0).lighten(37%), [0.2006]), table.cell(fill: rgb(0, 254, 0).lighten(37%), [0.3321]), table.cell(fill: rgb(0, 254, 0).lighten(37%),  [0.9278]),
    [*Gated Recurrent Unit*], table.cell(fill: rgb(0, 254, 0).lighten(37%), [0.1960]), table.cell(fill: rgb(0, 254, 0).lighten(37%), [0.3178]), table.cell(fill: rgb(0, 254, 0).lighten(37%),  [0.9339]),
    [*Long Short Term Memory*], table.cell(fill: rgb(0, 254, 0).lighten(37%), [0.2006]), table.cell(fill: rgb(0, 254, 0).lighten(37%), [0.3291]), table.cell(fill: rgb(0, 254, 0).lighten(37%),  [0.9291]),
    [*Recurrent Neural Network*], table.cell(fill: rgb(3, 251, 0).lighten(37%), [0.2969]), table.cell(fill: rgb(2, 252, 0).lighten(37%), [0.4106]), table.cell(fill: rgb(0, 254, 0).lighten(37%),  [0.8897]),
    [*TabNet*], table.cell(fill: rgb(0, 255, 0).lighten(37%), [0.1810]), table.cell(fill: rgb(0, 255, 0).lighten(37%), [0.3033]), table.cell(fill: rgb(0, 255, 0).lighten(37%),  [0.9398]),
    [*Temporal Convolutional Network*], table.cell(fill: rgb(0, 254, 0).lighten(37%), [0.2153]), table.cell(fill: rgb(0, 254, 0).lighten(37%), [0.3391]), table.cell(fill: rgb(0, 254, 0).lighten(37%),  [0.9247]),
    [*Transformer*], table.cell(fill: rgb(21, 233, 0).lighten(37%), [0.9863]), table.cell(fill: rgb(23, 231, 0).lighten(37%), [1.2027]), table.cell(fill: rgb(3, 251, 0).lighten(37%),  [0.0533]),
    [*WaveNet*], table.cell(fill: rgb(4, 250, 0).lighten(37%), [0.3486]), table.cell(fill: rgb(4, 250, 0).lighten(37%), [0.4726]), table.cell(fill: rgb(0, 254, 0).lighten(37%),  [0.8538]),
    [*Monte Carlo Methods*], table.cell(fill: rgb(255, 0, 0).lighten(37%), [9.7722]), table.cell(fill: rgb(255, 0, 0).lighten(37%), [10.0824]), table.cell(fill: rgb(255, 0, 0).lighten(37%),  [-65.5297]),
    [*Policy Iteration*], table.cell(fill: rgb(44, 210, 0).lighten(37%), [1.8643]), table.cell(fill: rgb(50, 204, 0).lighten(37%), [2.2311]), table.cell(fill: rgb(12, 242, 0).lighten(37%),  [-2.2578]),
    [*Q Learning*], table.cell(fill: rgb(55, 199, 0).lighten(37%), [2.2520]), table.cell(fill: rgb(61, 193, 0).lighten(37%), [2.6721]), table.cell(fill: rgb(17, 237, 0).lighten(37%),  [-3.6730]),
    [*Reinforce Monte Carlo Policy Gradient*], table.cell(fill: rgb(25, 229, 0).lighten(37%), [1.1309]), table.cell(fill: rgb(27, 227, 0).lighten(37%), [1.3436]), table.cell(fill: rgb(4, 250, 0).lighten(37%),  [-0.1815]),
    [*State-Action-Reward-State-Action*], table.cell(fill: rgb(46, 208, 0).lighten(37%), [1.9390]), table.cell(fill: rgb(52, 202, 0).lighten(37%), [2.2997]), table.cell(fill: rgb(13, 241, 0).lighten(37%),  [-2.4614]),
    [*Trust Region Policy Optimization*], table.cell(fill: rgb(46, 208, 0).lighten(37%), [1.9406]), table.cell(fill: rgb(52, 202, 0).lighten(37%), [2.3008]), table.cell(fill: rgb(13, 241, 0).lighten(37%),  [-2.4646]),
    [*Advantage Actor-Critic*], table.cell(fill: rgb(90, 164, 0).lighten(37%), [3.5762]), table.cell(fill: rgb(88, 166, 0).lighten(37%), [3.7040]), table.cell(fill: rgb(34, 220, 0).lighten(37%),  [-7.9789]),
    [*Asynchronous Advantage Actor-Critic*], table.cell(fill: rgb(39, 215, 0).lighten(37%), [1.6548]), table.cell(fill: rgb(44, 210, 0).lighten(37%), [2.0038]), table.cell(fill: rgb(9, 245, 0).lighten(37%),  [-1.6279]),
    [*Deep Deterministic Policy Gradient*], table.cell(fill: rgb(22, 232, 0).lighten(37%), [1.0401]), table.cell(fill: rgb(24, 230, 0).lighten(37%), [1.2363]), table.cell(fill: rgb(3, 251, 0).lighten(37%),  [-0.0003]),
    [*Deep Q-Network*], table.cell(fill: rgb(41, 213, 0).lighten(37%), [1.7604]), table.cell(fill: rgb(47, 207, 0).lighten(37%), [2.1347]), table.cell(fill: rgb(11, 243, 0).lighten(37%),  [-1.9824]),
    [*Proximal Policy Optimization*], table.cell(fill: rgb(30, 224, 0).lighten(37%), [1.3416]), table.cell(fill: rgb(35, 219, 0).lighten(37%), [1.6525]), table.cell(fill: rgb(6, 248, 0).lighten(37%),  [-0.7873]),
    [*Soft Actor-Critic*], table.cell(fill: rgb(177, 77, 0).lighten(37%), [6.8640]), table.cell(fill: rgb(207, 47, 0).lighten(37%), [8.2453]), table.cell(fill: rgb(170, 84, 0).lighten(37%),  [-43.4945]),
    [*Trust Region Policy Optimization*], table.cell(fill: rgb(39, 215, 0).lighten(37%), [1.6608]), table.cell(fill: rgb(41, 213, 0).lighten(37%), [1.9010]), table.cell(fill: rgb(8, 246, 0).lighten(37%),  [-1.3651]),
    table.cell(fill: rgb("#ffe0b2"), [*Ensemble*]), table.cell(fill: rgb("#ffe0b2"), [0.1752]), table.cell(fill: rgb("#ffe0b2"), [0.2922]), table.cell(fill: rgb("#ffe0b2"), [0.9441]),
  ),
  caption: [Comparación de métricas entre modelos],
)

// = Resultados por Modelo


// == Modelo: Attention Only

// === Métricas
// - MAE: 0.3416
// - RMSE: 0.4854
// - R²: 0.8458

// === Historial de Entrenamiento
// #figure(
//   image("../figures/various_models/tensorflow/individual_models/tf_attention_only/training_history.png", width: 71%),
//   caption: [Historial de entrenamiento para tf_attention_only],
// )

// === Predicciones
// #figure(
//   image("../figures/various_models/tensorflow/individual_models/tf_attention_only/predictions.png", width: 71%),
//   caption: [Predicciones vs valores reales para tf_attention_only],
// )


// == Modelo: Convolutional Neural Network

// === Métricas
// - MAE: 0.1977
// - RMSE: 0.3170
// - R²: 0.9342

// === Historial de Entrenamiento
// #figure(
//   image("../figures/various_models/tensorflow/individual_models/tf_cnn/training_history.png", width: 71%),
//   caption: [Historial de entrenamiento para tf_cnn],
// )

// === Predicciones
// #figure(
//   image("../figures/various_models/tensorflow/individual_models/tf_cnn/predictions.png", width: 71%),
//   caption: [Predicciones vs valores reales para tf_cnn],
// )


// == Modelo: Feed Forward Neural Network

// === Métricas
// - MAE: 0.2006
// - RMSE: 0.3321
// - R²: 0.9278

// === Historial de Entrenamiento
// #figure(
//   image("../figures/various_models/tensorflow/individual_models/tf_fnn/training_history.png", width: 71%),
//   caption: [Historial de entrenamiento para tf_fnn],
// )

// === Predicciones
// #figure(
//   image("../figures/various_models/tensorflow/individual_models/tf_fnn/predictions.png", width: 71%),
//   caption: [Predicciones vs valores reales para tf_fnn],
// )


// == Modelo: Gated Recurrent Unit

// === Métricas
// - MAE: 0.1960
// - RMSE: 0.3178
// - R²: 0.9339

// === Historial de Entrenamiento
// #figure(
//   image("../figures/various_models/tensorflow/individual_models/tf_gru/training_history.png", width: 71%),
//   caption: [Historial de entrenamiento para tf_gru],
// )

// === Predicciones
// #figure(
//   image("../figures/various_models/tensorflow/individual_models/tf_gru/predictions.png", width: 71%),
//   caption: [Predicciones vs valores reales para tf_gru],
// )


// == Modelo: Long Short Term Memory

// === Métricas
// - MAE: 0.2006
// - RMSE: 0.3291
// - R²: 0.9291

// === Historial de Entrenamiento
// #figure(
//   image("../figures/various_models/tensorflow/individual_models/tf_lstm/training_history.png", width: 71%),
//   caption: [Historial de entrenamiento para tf_lstm],
// )

// === Predicciones
// #figure(
//   image("../figures/various_models/tensorflow/individual_models/tf_lstm/predictions.png", width: 71%),
//   caption: [Predicciones vs valores reales para tf_lstm],
// )


// == Modelo: Recurrent Neural Network

// === Métricas
// - MAE: 0.2969
// - RMSE: 0.4106
// - R²: 0.8897

// === Historial de Entrenamiento
// #figure(
//   image("../figures/various_models/tensorflow/individual_models/tf_rnn/training_history.png", width: 71%),
//   caption: [Historial de entrenamiento para tf_rnn],
// )

// === Predicciones
// #figure(
//   image("../figures/various_models/tensorflow/individual_models/tf_rnn/predictions.png", width: 71%),
//   caption: [Predicciones vs valores reales para tf_rnn],
// )


// == Modelo: TabNet

// === Métricas
// - MAE: 0.1810
// - RMSE: 0.3033
// - R²: 0.9398

// === Historial de Entrenamiento
// #figure(
//   image("../figures/various_models/tensorflow/individual_models/tf_tabnet/training_history.png", width: 71%),
//   caption: [Historial de entrenamiento para tf_tabnet],
// )

// === Predicciones
// #figure(
//   image("../figures/various_models/tensorflow/individual_models/tf_tabnet/predictions.png", width: 71%),
//   caption: [Predicciones vs valores reales para tf_tabnet],
// )


// == Modelo: Temporal Convolutional Network

// === Métricas
// - MAE: 0.2153
// - RMSE: 0.3391
// - R²: 0.9247

// === Historial de Entrenamiento
// #figure(
//   image("../figures/various_models/tensorflow/individual_models/tf_tcn/training_history.png", width: 71%),
//   caption: [Historial de entrenamiento para tf_tcn],
// )

// === Predicciones
// #figure(
//   image("../figures/various_models/tensorflow/individual_models/tf_tcn/predictions.png", width: 71%),
//   caption: [Predicciones vs valores reales para tf_tcn],
// )


// == Modelo: Transformer

// === Métricas
// - MAE: 0.9863
// - RMSE: 1.2027
// - R²: 0.0533

// === Historial de Entrenamiento
// #figure(
//   image("../figures/various_models/tensorflow/individual_models/tf_transformer/training_history.png", width: 71%),
//   caption: [Historial de entrenamiento para tf_transformer],
// )

// === Predicciones
// #figure(
//   image("../figures/various_models/tensorflow/individual_models/tf_transformer/predictions.png", width: 71%),
//   caption: [Predicciones vs valores reales para tf_transformer],
// )


// == Modelo: WaveNet

// === Métricas
// - MAE: 0.3486
// - RMSE: 0.4726
// - R²: 0.8538

// === Historial de Entrenamiento
// #figure(
//   image("../figures/various_models/tensorflow/individual_models/tf_wavenet/training_history.png", width: 71%),
//   caption: [Historial de entrenamiento para tf_wavenet],
// )

// === Predicciones
// #figure(
//   image("../figures/various_models/tensorflow/individual_models/tf_wavenet/predictions.png", width: 71%),
//   caption: [Predicciones vs valores reales para tf_wavenet],
// )


// == Modelo: Monte Carlo Methods

// === Métricas
// - MAE: 9.7722
// - RMSE: 10.0824
// - R²: -65.5297

// === Historial de Entrenamiento
// #figure(
//   image("../figures/various_models/tensorflow/individual_models/tf_monte_carlo/training_history.png", width: 71%),
//   caption: [Historial de entrenamiento para tf_monte_carlo],
// )

// === Predicciones
// #figure(
//   image("../figures/various_models/tensorflow/individual_models/tf_monte_carlo/predictions.png", width: 71%),
//   caption: [Predicciones vs valores reales para tf_monte_carlo],
// )


// == Modelo: Policy Iteration

// === Métricas
// - MAE: 1.8643
// - RMSE: 2.2311
// - R²: -2.2578

// === Historial de Entrenamiento
// #figure(
//   image("../figures/various_models/tensorflow/individual_models/tf_policy_iteration/training_history.png", width: 71%),
//   caption: [Historial de entrenamiento para tf_policy_iteration],
// )

// === Predicciones
// #figure(
//   image("../figures/various_models/tensorflow/individual_models/tf_policy_iteration/predictions.png", width: 71%),
//   caption: [Predicciones vs valores reales para tf_policy_iteration],
// )


// == Modelo: Q Learning

// === Métricas
// - MAE: 2.2520
// - RMSE: 2.6721
// - R²: -3.6730

// === Historial de Entrenamiento
// #figure(
//   image("../figures/various_models/tensorflow/individual_models/tf_q_learning/training_history.png", width: 71%),
//   caption: [Historial de entrenamiento para tf_q_learning],
// )

// === Predicciones
// #figure(
//   image("../figures/various_models/tensorflow/individual_models/tf_q_learning/predictions.png", width: 71%),
//   caption: [Predicciones vs valores reales para tf_q_learning],
// )


// == Modelo: Reinforce Monte Carlo Policy Gradient

// === Métricas
// - MAE: 1.1309
// - RMSE: 1.3436
// - R²: -0.1815

// === Historial de Entrenamiento
// #figure(
//   image("../figures/various_models/tensorflow/individual_models/tf_reinforce_mcpg/training_history.png", width: 71%),
//   caption: [Historial de entrenamiento para tf_reinforce_mcpg],
// )

// === Predicciones
// #figure(
//   image("../figures/various_models/tensorflow/individual_models/tf_reinforce_mcpg/predictions.png", width: 71%),
//   caption: [Predicciones vs valores reales para tf_reinforce_mcpg],
// )


// == Modelo: State-Action-Reward-State-Action

// === Métricas
// - MAE: 1.9390
// - RMSE: 2.2997
// - R²: -2.4614

// === Historial de Entrenamiento
// #figure(
//   image("../figures/various_models/tensorflow/individual_models/tf_sarsa/training_history.png", width: 71%),
//   caption: [Historial de entrenamiento para tf_sarsa],
// )

// === Predicciones
// #figure(
//   image("../figures/various_models/tensorflow/individual_models/tf_sarsa/predictions.png", width: 71%),
//   caption: [Predicciones vs valores reales para tf_sarsa],
// )


// == Modelo: Trust Region Policy Optimization

// === Métricas
// - MAE: 1.9406
// - RMSE: 2.3008
// - R²: -2.4646

// === Historial de Entrenamiento
// #figure(
//   image("../figures/various_models/tensorflow/individual_models/tf_value_iteration/training_history.png", width: 71%),
//   caption: [Historial de entrenamiento para tf_value_iteration],
// )

// === Predicciones
// #figure(
//   image("../figures/various_models/tensorflow/individual_models/tf_value_iteration/predictions.png", width: 71%),
//   caption: [Predicciones vs valores reales para tf_value_iteration],
// )


// == Modelo: Advantage Actor-Critic

// === Métricas
// - MAE: 3.5762
// - RMSE: 3.7040
// - R²: -7.9789

// === Historial de Entrenamiento
// #figure(
//   image("../figures/various_models/tensorflow/individual_models/tf_a2c/training_history.png", width: 71%),
//   caption: [Historial de entrenamiento para tf_a2c],
// )

// === Predicciones
// #figure(
//   image("../figures/various_models/tensorflow/individual_models/tf_a2c/predictions.png", width: 71%),
//   caption: [Predicciones vs valores reales para tf_a2c],
// )


// == Modelo: Asynchronous Advantage Actor-Critic

// === Métricas
// - MAE: 1.6548
// - RMSE: 2.0038
// - R²: -1.6279

// === Historial de Entrenamiento
// #figure(
//   image("../figures/various_models/tensorflow/individual_models/tf_a3c/training_history.png", width: 71%),
//   caption: [Historial de entrenamiento para tf_a3c],
// )

// === Predicciones
// #figure(
//   image("../figures/various_models/tensorflow/individual_models/tf_a3c/predictions.png", width: 71%),
//   caption: [Predicciones vs valores reales para tf_a3c],
// )


// == Modelo: Deep Deterministic Policy Gradient

// === Métricas
// - MAE: 1.0401
// - RMSE: 1.2363
// - R²: -0.0003

// === Historial de Entrenamiento
// #figure(
//   image("../figures/various_models/tensorflow/individual_models/tf_ddpg/training_history.png", width: 71%),
//   caption: [Historial de entrenamiento para tf_ddpg],
// )

// === Predicciones
// #figure(
//   image("../figures/various_models/tensorflow/individual_models/tf_ddpg/predictions.png", width: 71%),
//   caption: [Predicciones vs valores reales para tf_ddpg],
// )


// == Modelo: Deep Q-Network

// === Métricas
// - MAE: 1.7604
// - RMSE: 2.1347
// - R²: -1.9824

// === Historial de Entrenamiento
// #figure(
//   image("../figures/various_models/tensorflow/individual_models/tf_dqn/training_history.png", width: 71%),
//   caption: [Historial de entrenamiento para tf_dqn],
// )

// === Predicciones
// #figure(
//   image("../figures/various_models/tensorflow/individual_models/tf_dqn/predictions.png", width: 71%),
//   caption: [Predicciones vs valores reales para tf_dqn],
// )


// == Modelo: Proximal Policy Optimization

// === Métricas
// - MAE: 1.3416
// - RMSE: 1.6525
// - R²: -0.7873

// === Historial de Entrenamiento
// #figure(
//   image("../figures/various_models/tensorflow/individual_models/tf_ppo/training_history.png", width: 71%),
//   caption: [Historial de entrenamiento para tf_ppo],
// )

// === Predicciones
// #figure(
//   image("../figures/various_models/tensorflow/individual_models/tf_ppo/predictions.png", width: 71%),
//   caption: [Predicciones vs valores reales para tf_ppo],
// )


// == Modelo: Soft Actor-Critic

// === Métricas
// - MAE: 6.8640
// - RMSE: 8.2453
// - R²: -43.4945

// === Historial de Entrenamiento
// #figure(
//   image("../figures/various_models/tensorflow/individual_models/tf_sac/training_history.png", width: 71%),
//   caption: [Historial de entrenamiento para tf_sac],
// )

// === Predicciones
// #figure(
//   image("../figures/various_models/tensorflow/individual_models/tf_sac/predictions.png", width: 71%),
//   caption: [Predicciones vs valores reales para tf_sac],
// )


// == Modelo: Trust Region Policy Optimization

// === Métricas
// - MAE: 1.6608
// - RMSE: 1.9010
// - R²: -1.3651

// === Historial de Entrenamiento
// #figure(
//   image("../figures/various_models/tensorflow/individual_models/tf_trpo/training_history.png", width: 71%),
//   caption: [Historial de entrenamiento para tf_trpo],
// )

// === Predicciones
// #figure(
//   image("../figures/various_models/tensorflow/individual_models/tf_trpo/predictions.png", width: 71%),
//   caption: [Predicciones vs valores reales para tf_trpo],
// )


// == Modelo Ensemble

// === Métricas
// - MAE: 0.1752
// - RMSE: 0.2922
// - R²: 0.9441

// === Pesos del Ensemble
// #figure(
//   image("../figures/various_models/tensorflow/ensemble_weights.png", width: 71%),
//   caption: [Pesos optimizados para cada modelo en el ensemble],
// )

// = Conclusiones

// El framework TENSORFLOW fue utilizado para entrenar 23 modelos diferentes. 
// El modelo ensemble logró un MAE de 0.1752, un RMSE de 0.2922 
// y un coeficiente R² de 0.9441.

