CONST_VAL_LOSS = "val_loss"
CONST_LOSS = "loss"
CONST_METRIC_MAE = "mae"
CONST_METRIC_RMSE = "rmse"
CONST_METRIC_R2 = "r2"
CONST_MODELS = "models"
CONST_BEST_PREFIX = "best_"
CONST_LOGS_DIR = "logs"
CONST_DEFAULT_EPOCHS = 100
CONST_DEFAULT_BATCH_SIZE = 32
CONST_DEFAULT_SEED = 42
CONST_FIGURES_DIR = "figures"
CONST_MODEL_TYPES = {
    "dl": "deep_learning",
    "drl": "deep_reinforcement_learning",
    "rl": "reinforcement_learning"
}
CONST_FRAMEWORKS = {
    "tensorflow": "TensorFlow",
    "jax": "JAX",
    "pytorch": "PyTorch"
}
# Modelos de Aprendizaje Profundo
ATT_ONLY = "Attention Only"
CNN = "Convolutional Neural Network"
FNN = "Feed Forward Neural Network"
GRU = "Gated Recurrent Unit"
LSTM = "Long Short Term Memory"
RNN = "Recurrent Neural Network"
TABNET = "TabNet"
TCN = "Temporal Convolutional Network"
TRANSFORMER = "Transformer"
WAVENET = "WaveNet"
# Modelos de Aprendizaje por Refuerzo
MONTE_CARLO = "Monte Carlo Methods"
POLICY_ITERATION = "Policy Iteration"
Q_LEARNING = "Q Learning"
REINFORCE_MCPG = "Reinforce Monte Carlo Policy Gradient"
SARSA = "State-Action-Reward-State-Action"
VALUE_ITERATION = "Value Iteration"
# Modelos de Aprendizaje por Refuerzo Profundo
A2C = "Advantage Actor-Critic"
A3C = "Asynchronous Advantage Actor-Critic"
DDPG = "Deep Deterministic Policy Gradient"
DQN = "Deep Q-Network"
PPO = "Proximal Policy Optimization"
SAC = "Soft Actor-Critic"
TRPO = "Trust Region Policy Optimization"
# Mapa de Nombres de Modelos
CONST_MODELS_NAMES = {
    # Modelos de Aprendizaje Profundo
    ## TensorFlow
    "tf_attention_only": ATT_ONLY,
    "tf_cnn": CNN,
    "tf_fnn": FNN,
    "tf_gru": GRU,
    "tf_lstm": LSTM,
    "tf_rnn": RNN,
    "tf_tabnet": TABNET,
    "tf_tcn": TCN,
    "tf_transformer": TRANSFORMER,
    "tf_wavenet": WAVENET,
    ## JAX
    "jax_attention_only": ATT_ONLY,
    "jax_cnn": CNN,
    "jax_fnn": FNN,
    "jax_gru": GRU,
    "jax_lstm": LSTM,
    "jax_rnn": RNN,
    "jax_tabnet": TABNET,
    "jax_tcn": TCN,
    "jax_transformer": TRANSFORMER,
    "jax_wavenet": WAVENET,
    ## PyTorch
    "pt_attention_only": ATT_ONLY,
    "pt_cnn": CNN,
    "pt_fnn": FNN,
    "pt_gru": GRU,
    "pt_lstm": LSTM,
    "pt_rnn": RNN,
    "pt_tabnet": TABNET,
    "pt_tcn": TCN,
    "pt_transformer": TRANSFORMER,
    "pt_wavenet": WAVENET,
    # Modelos de Aprendizaje por Refuerzo
    ## TensorFlow
    "tf_monte_carlo": MONTE_CARLO,
    "tf_policy_iteration": POLICY_ITERATION,
    "tf_q_learning": Q_LEARNING,
    "tf_reinforce_mcpg": REINFORCE_MCPG,
    "tf_sarsa": SARSA,
    "tf_value_iteration": TRPO,
    ## JAX
    "jax_monte_carlo": MONTE_CARLO,
    "jax_policy_iteration": POLICY_ITERATION,
    "jax_q_learning": Q_LEARNING,
    "jax_reinforce_mcpg": REINFORCE_MCPG,
    "jax_sarsa": SARSA,
    "jax_value_iteration": TRPO,
    ## PyTorch
    "pt_monte_carlo": MONTE_CARLO,
    "pt_policy_iteration": POLICY_ITERATION,
    "pt_q_learning": Q_LEARNING,
    "pt_reinforce_mcpg": REINFORCE_MCPG,
    "pt_sarsa": SARSA,
    "pt_value_iteration": TRPO,
    # Modelos de Aprendizaje por Refuerzo Profundo
    ## TensorFlow
    "tf_a2c": A2C,
    "tf_a3c": A3C,
    "tf_ddpg": DDPG,
    "tf_dqn": DQN,
    "tf_ppo": PPO,
    "tf_sac": SAC,
    "tf_trpo": TRPO,
    ## JAX
    "jax_a2c": A2C,
    "jax_a3c": A3C,
    "jax_ddpg": DDPG,
    "jax_dqn": DQN,
    "jax_ppo": PPO,
    "jax_sac": SAC,
    "jax_trpo": TRPO,
    ## PyTorch
    "pt_a2c": A2C,
    "pt_a3c": A3C,
    "pt_ddpg": DDPG,
    "pt_dqn": DQN,
    "pt_ppo": PPO,
    "pt_sac": SAC,
    "pt_trpo": TRPO,
}
# Reporte
HEADERS_BACKGROUND = "#e7da27"
MODELS_BACKGROUND = "#e6e6e6"
ENSEMBLE_BACKGROUND = "#ffe0b2"