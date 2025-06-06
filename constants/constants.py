# Metricas
CONST_VAL_LOSS = "val_loss"
CONST_LOSS = "loss"
CONST_METRIC_MAE = "mae"
CONST_METRIC_RMSE = "rmse"
CONST_METRIC_R2 = "r2"
# Constantes Generales
CONST_MODELS = "models"
CONST_BEST_PREFIX = "best_"
CONST_LOGS_DIR = "logs"
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
# Toggleable
CONST_DEFAULT_EPOCHS = 100
CONST_DEFAULT_BATCH_SIZE = 32
CONST_DEFAULT_SEED = 7
CONST_DURATION_HOURS = 24
CONST_EPSILON = 1e-10

# Constantes de Procesamiento
DATE_FORMAT: str = "%d-%m-%Y %H:%M:%S"
TIMESTAMP_COL: str = "Timestamp"
SUBJECT_ID_COL: str = "SubjectID"
GLUCOSE_COL: str = "value"
BOLUS_COL: str = "bolus"
MEAL_COL: str = "meal_carbs"
BASAL_COL: str = "basal_rate"
TEMP_BASAL_COL: str = "temp_basal_rate"

# Constantes de los Modelos
## Valores de Glucosa
LOWER_BOUND_NORMAL_GLUCOSE_RANGE: float = 70.0
UPPER_BOUND_NORMAL_GLUCOSE_RANGE: float = 180.0
TARGET_GLUCOSE: float = 100.0
POSITIVE_REWARD: float = 1.0
MILD_PENALTY_REWARD: float = -0.5
SEVERE_PENALTY_REWARD: float = -1.0
## Constantes de texto
CONST_ACTOR = "actor"
CONST_ACTOR_LOSS = "actor_loss"
CONST_CRITIC = "critic"
CONST_CRITIC_LOSS = "critic_loss"
CONST_TARGET = "target"
CONST_PARAMS = "params"
CONST_DEVICE = "device"
CONST_MODEL_INIT_ERROR = "El modelo debe ser inicializado antes de {}"
CONST_DROPOUT = "dropout"
CONST_PARAMS = "params"
CONST_POLICY_LOSS = "policy_loss"
CONST_VALUE_LOSS = "value_loss"
CONST_ENTROPY_LOSS = "entropy_loss"
CONST_TOTAL_LOSS = "total_loss"
CONST_EPISODE_REWARDS = "episode_rewards"
CONST_Q_VALUE = "q_value"
CONST_TRAINING = "training"
CONST_ALPHA_LOSS = "alpha_loss"
CONST_TOTAL_LOSS = "total_loss"
CONST_ENTROPY = "entropy"
CONST_CGM_ENCODER = "cgm_encoder"
CONST_OTHER_ENCODER = "other_encoder"
CONST_COMBINED_LAYER = "combined_layer"

# Funciones de Activación
CONST_RELU = "relu"
CONST_TANH = "tanh"
CONST_LEAKY_RELU = "leaky_relu"
CONST_GELU = "gelu"

# Constantes para aprendizaje por refuerzo offline
OFFLINE_RL_REWARD_SCALE = 1.0
OFFLINE_GAMMA = 0.99  # Factor de descuento para recompensas futuras
OFFLINE_LAMBDA = 0.95  # Factor para GAE (Generalized Advantage Estimation)

# Rangos de glucosa para función de recompensa
SEVERE_HYPOGLYCEMIA_THRESHOLD = 54.0  # mg/dL
HYPOGLYCEMIA_THRESHOLD = 70.0  # mg/dL 
HYPERGLYCEMIA_THRESHOLD = 180.0  # mg/dL
SEVERE_HYPERGLYCEMIA_THRESHOLD = 250.0  # mg/dL
TARGET_GLUCOSE = 100.0  # mg/dL - Objetivo ideal

# Constantes para evaluación fuera de política
CONST_IPS_CLIP = 10.0  # Clipping para importance sampling
CONST_CQL_ALPHA = 1.0  # Parámetro para Conservative Q-Learning
CONST_CONFIDENCE_LEVEL = 0.95  # Nivel de confianza para intervalos

# ===============================

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