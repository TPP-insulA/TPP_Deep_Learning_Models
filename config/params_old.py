# Modelos
## Modelos TensorFlow
### Modelos de Aprendizaje Profundo
from datetime import timedelta
from typing import Union
from models_old.tensorflow.DeepLearning.attention_only import create_attention_model as tf_create_attention_model
from models_old.tensorflow.DeepLearning.cnn import create_cnn_model as tf_create_cnn_model
from models_old.tensorflow.DeepLearning.fnn import create_fnn_model as tf_create_fnn_model
from models_old.tensorflow.DeepLearning.gru import create_gru_model as tf_create_gru_model
from models_old.tensorflow.DeepLearning.lstm import create_lstm_model as tf_create_lstm_model
from models_old.tensorflow.DeepLearning.rnn import create_rnn_model as tf_create_rnn_model
from models_old.tensorflow.DeepLearning.tabnet import create_tabnet_model as tf_create_tabnet_model
from models_old.tensorflow.DeepLearning.tcn import create_tcn_model as tf_create_tcn_model
from models_old.tensorflow.DeepLearning.transformer import create_transformer_model as tf_create_transformer_model
from models_old.tensorflow.DeepLearning.wavenet import create_wavenet_model as tf_create_wavenet_model

### Modelos de Aprendizaje por Refuerzo
from models_old.tensorflow.ReinforcementLearning.monte_carlo_methods import create_monte_carlo_model as tf_create_monte_carlo_model
from models_old.tensorflow.ReinforcementLearning.policy_iteration import create_policy_iteration_model as tf_create_policy_iteration_model
from models_old.tensorflow.ReinforcementLearning.q_learning import create_q_learning_model as tf_create_q_learning_model
from models_old.tensorflow.ReinforcementLearning.reinforce_mcpg import create_reinforce_mcpg_model as tf_create_reinforce_mcpg_model
from models_old.tensorflow.ReinforcementLearning.sarsa import create_sarsa_model as tf_create_sarsa_model
from models_old.tensorflow.ReinforcementLearning.value_iteration import create_value_iteration_model as tf_create_value_iteration_model

### Modelos de Aprendizaje por Refuerzo Profundo
from models_old.tensorflow.DeepReinforcementLearning.a2c_a3c import create_a2c_model as tf_create_a2c_model, create_a3c_model as tf_create_a3c_model
from models_old.tensorflow.DeepReinforcementLearning.ddpg import create_ddpg_model as tf_create_ddpg_model
from models_old.tensorflow.DeepReinforcementLearning.dqn import create_dqn_model as tf_create_dqn_model
from models_old.tensorflow.DeepReinforcementLearning.ppo import create_ppo_model as tf_create_ppo_model
from models_old.tensorflow.DeepReinforcementLearning.sac import create_sac_model as tf_create_sac_model
from models_old.tensorflow.DeepReinforcementLearning.trpo import create_trpo_model as tf_create_trpo_model

## Modelos JAX
### Modelos de Aprendizaje Profundo
from models_old.jax.DeepLearning.attention_only import model_creator as jax_create_attention_model
from models_old.jax.DeepLearning.cnn import model_creator as jax_create_cnn_model
from models_old.jax.DeepLearning.fnn import model_creator as jax_create_fnn_model
from models_old.jax.DeepLearning.gru import model_creator as jax_create_gru_model
from models_old.jax.DeepLearning.lstm import model_creator as jax_create_lstm_model
from models_old.jax.DeepLearning.rnn import model_creator as jax_create_rnn_model
from models_old.jax.DeepLearning.tabnet import model_creator as jax_create_tabnet_model
from models_old.jax.DeepLearning.tcn import model_creator as jax_create_tcn_model
from models_old.jax.DeepLearning.transformer import model_creator as jax_create_transformer_model
from models_old.jax.DeepLearning.wavenet import model_creator as jax_create_wavenet_model

### Modelos de Aprendizaje por Refuerzo
from models_old.jax.ReinforcementLearning.monte_carlo_methods import create_monte_carlo_model as jax_monte_carlo_creator
from models_old.jax.ReinforcementLearning.policy_iteration import model_creator as jax_policy_iteration_creator
from models_old.jax.ReinforcementLearning.q_learning import model_creator as jax_q_learning_creator
from models_old.jax.ReinforcementLearning.reinforce_mcgp import model_creator as jax_reinforce_mcgp_creator
from models_old.jax.ReinforcementLearning.sarsa import model_creator as jax_sarsa_creator
from models_old.jax.ReinforcementLearning.value_iteration import model_creator as jax_value_iteration_creator

### Modelos de Aprendizaje por Refuerzo Profundo
from models_old.jax.DeepReinforcementLearning.a2c_a3c import model_creator_a2c as jax_a2c_creator, model_creator_a3c as jax_a3c_creator
from models_old.jax.DeepReinforcementLearning.ddpg import model_creator as jax_ddpg_creator
from models_old.jax.DeepReinforcementLearning.dqn import model_creator as jax_dqn_creator
from models_old.jax.DeepReinforcementLearning.ppo import model_creator as jax_ppo_creator
from models_old.jax.DeepReinforcementLearning.sac import model_creator as jax_sac_creator
from models_old.jax.DeepReinforcementLearning.trpo import model_creator as jax_trpo_creator


## Modelos PyTorch
### Modelos de Aprendizaje Profundo
from models_old.pytorch.DeepLearning.attention_only import model_creator as pt_create_attention_model
from models_old.pytorch.DeepLearning.cnn import model_creator as pt_create_cnn_model
from models_old.pytorch.DeepLearning.fnn import model_creator as pt_create_fnn_model
from models_old.pytorch.DeepLearning.gru import model_creator as pt_create_gru_model
from models_old.pytorch.DeepLearning.lstm import model_creator as pt_create_lstm_model
from models_old.pytorch.DeepLearning.rnn import model_creator as pt_create_rnn_model
from models_old.pytorch.DeepLearning.tabnet import model_creator as pt_create_tabnet_model
from models_old.pytorch.DeepLearning.tcn import model_creator as pt_create_tcn_model
from models_old.pytorch.DeepLearning.transformer import model_creator as pt_create_transformer_model
from models_old.pytorch.DeepLearning.wavenet import model_creator as pt_create_wavenet_model

### Modelos de Aprendizaje por Refuerzo
from models_old.pytorch.ReinforcementLearning.monte_carlo_methods import model_creator as pt_monte_carlo_creator
from models_old.pytorch.ReinforcementLearning.policy_iteration import model_creator as pt_policy_iteration_creator
from models_old.pytorch.ReinforcementLearning.q_learning import model_creator as pt_q_learning_creator
from models_old.pytorch.ReinforcementLearning.reinforce_mcgp import model_creator as pt_reinforce_mcgp_creator
from models_old.pytorch.ReinforcementLearning.sarsa import model_creator as pt_sarsa_creator
from models_old.pytorch.ReinforcementLearning.value_iteration import model_creator as pt_value_iteration_creator

### Modelos de Aprendizaje por Refuerzo Profundo
from models_old.pytorch.DeepReinforcementLearning.a2c_a3c import model_creator_offline_a2c as pt_a2c_creator, model_creator_a3c as pt_a3c_creator
from models_old.pytorch.DeepReinforcementLearning.ddpg import model_creator as pt_ddpg_creator
from models_old.pytorch.DeepReinforcementLearning.dqn import model_creator as pt_dqn_creator
from models_old.pytorch.DeepReinforcementLearning.ppo import model_creator as pt_ppo_creator
from models_old.pytorch.DeepReinforcementLearning.sac import model_creator as pt_sac_creator
from models_old.pytorch.DeepReinforcementLearning.trpo import model_creator as pt_trpo_creator

# Modo de Ejecución
DEBUG = False
FRAMEWORK_OP = 2
PROCESSING_OP = 1

# Configuración de procesamiento
## Framework a utilizar durante la ejecución. Puede ser con TensorFlow o JAX.
## Opciones: "tensorflow", "jax", "pytorch"
FRAMEWORK_OPTIONS = ["tensorflow", "jax", "pytorch"]
FRAMEWORK = FRAMEWORK_OPTIONS[FRAMEWORK_OP] 
## Procesamiento de datos. Puede ser con pandas o polars.
## Opciones: "pandas", "polars"
PROCESSING_OPTIONS = ["pandas", "polars"]
PROCESSING = PROCESSING_OPTIONS[PROCESSING_OP]

# Configuración de Procesamiento
CONFIG_PROCESSING: dict[str, Union[int, float, str]] = {
    "batch_size": 128,
    "window_hours": 2,
    "window_steps": 24,  # 5-min steps in 2 hours
    "window_size": 12,
    "extended_window_size": 288,
    "insulin_lifetime_hours": 4.0,
    "cap_normal": 30,
    "cap_bg": 300,
    "cap_iob": 5,
    "cap_carb": 150,
    "min_carbs": 0,
    "max_carbs": 150,
    "min_bg": 40,
    "max_bg": 400,
    "min_insulin": 0,
    "max_insulin": 30,
    "min_icr": 5,
    "max_icr": 20,
    "min_isf": 10,
    "max_isf": 100,
    "timezone": "UTC",
    "max_work_intensity": 10,
    "max_sleep_quality": 10,
    "max_activity_intensity": 10,
    "low_dose_threshold": 7.0,
    "min_cgm_points": 12,
    "alignment_tolerance_minutes": 15,
    "random_seed": 42,
    "carb_effect_factor": 5,
    "insulin_effect_factor": 50,
    "glucose_min": 40,
    "glucose_max": 400,
    "bolus_max": 20.0,
    "bolus_min": 0.1,
    "partial_window_threshold": 6,
    "event_tolerance": timedelta(minutes=15),
    "basal_estimation_hours": (0, 24),
    "basal_estimation_factor": 0.5,
    "hypoglycemia_threshold": 70,
    "hyperglycemia_threshold": 180,
    "tir_lower": 70,
    "tir_upper": 180,
    "simulation_steps": 72,
    "hypo_risk_threshold": 70,
    "hyper_risk_threshold": 180,
    "significant_meal_threshold": 20,
}


## Modelos TensorFlow disponibles.
TF_MODELS = {
    # TensorFlow
    ## Modelos de Aprendizaje Profundo
    "tf_attention_only": tf_create_attention_model,
    "tf_cnn": tf_create_cnn_model,
    "tf_fnn": tf_create_fnn_model,
    "tf_gru": tf_create_gru_model,
    "tf_lstm": tf_create_lstm_model,
    "tf_rnn": tf_create_rnn_model,
    "tf_tabnet": tf_create_tabnet_model,
    "tf_tcn": tf_create_tcn_model,
    "tf_transformer": tf_create_transformer_model,
    "tf_wavenet": tf_create_wavenet_model,
    ## Modelos de Aprendizaje por Refuerzo
    "tf_monte_carlo": tf_create_monte_carlo_model,
    "tf_policy_iteration": tf_create_policy_iteration_model,
    "tf_q_learning": tf_create_q_learning_model,
    "tf_reinforce_mcpg": tf_create_reinforce_mcpg_model,
    "tf_sarsa": tf_create_sarsa_model,
    "tf_value_iteration": tf_create_value_iteration_model,
    ## Modelos de Aprendizaje por Refuerzo Profundo
    "tf_a2c": tf_create_a2c_model,
    "tf_a3c": tf_create_a3c_model,
    "tf_ddpg": tf_create_ddpg_model,
    "tf_dqn": tf_create_dqn_model,
    "tf_ppo": tf_create_ppo_model,
    "tf_sac": tf_create_sac_model,
    "tf_trpo": tf_create_trpo_model,
}

## Modelos JAX disponibles.
JAX_MODELS = {
    # JAX
    ## Modelos de Aprendizaje Profundo
    "jax_attention_only": jax_create_attention_model,
    "jax_cnn": jax_create_cnn_model,
    "jax_fnn": jax_create_fnn_model,
    "jax_gru": jax_create_gru_model,
    "jax_lstm": jax_create_lstm_model,
    "jax_rnn": jax_create_rnn_model,
    "jax_tabnet": jax_create_tabnet_model,
    "jax_tcn": jax_create_tcn_model,
    "jax_transformer": jax_create_transformer_model,
    "jax_wavenet": jax_create_wavenet_model,
    ## Modelos de Aprendizaje por Refuerzo
    "jax_monte_carlo": jax_monte_carlo_creator,
    "jax_policy_iteration": jax_policy_iteration_creator,
    "jax_q_learning": jax_q_learning_creator,
    "jax_reinforce_mcpg": jax_reinforce_mcgp_creator,
    "jax_sarsa": jax_sarsa_creator,
    "jax_value_iteration": jax_value_iteration_creator,
    ## Modelos de Aprendizaje por Refuerzo Profundo
    "jax_a2c": jax_a2c_creator,
    "jax_a3c": jax_a3c_creator,
    "jax_ddpg": jax_ddpg_creator,
    "jax_dqn": jax_dqn_creator,
    "jax_ppo": jax_ppo_creator,
    "jax_sac": jax_sac_creator,
    "jax_trpo": jax_trpo_creator,
}

## Modelos PyTorch disponibles.
PT_MODELS = {
    # PyTorch
    ## Modelos de Aprendizaje Profundo
    "pt_attention_only": pt_create_attention_model,
    "pt_cnn": pt_create_cnn_model,
    "pt_fnn": pt_create_fnn_model,
    "pt_gru": pt_create_gru_model,
    "pt_lstm": pt_create_lstm_model,
    "pt_rnn": pt_create_rnn_model,
    "pt_tabnet": pt_create_tabnet_model,
    "pt_tcn": pt_create_tcn_model,
    "pt_transformer": pt_create_transformer_model,
    "pt_wavenet": pt_create_wavenet_model,
    ## Modelos de Aprendizaje por Refuerzo
    "pt_monte_carlo": pt_monte_carlo_creator,
    "pt_policy_iteration": pt_policy_iteration_creator,
    "pt_q_learning": pt_q_learning_creator,
    "pt_reinforce_mcpg": pt_reinforce_mcgp_creator,
    "pt_sarsa": pt_sarsa_creator,
    "pt_value_iteration": pt_value_iteration_creator,
    ## Modelos de Aprendizaje por Refuerzo Profundo
    "pt_a2c": pt_a2c_creator,
    "pt_a3c": pt_a3c_creator,
    "pt_ddpg": pt_ddpg_creator,
    "pt_dqn": pt_dqn_creator,
    "pt_ppo": pt_ppo_creator,
    "pt_sac": pt_sac_creator,
    "pt_trpo": pt_trpo_creator,
}

# Modelos TensorFlow a utilizar
USE_TF_MODELS = {
    ## Modelos de Aprendizaje Profundo
    "tf_attention_only": False,
    "tf_cnn": False,
    "tf_fnn": False,
    "tf_gru": False,
    "tf_lstm": False,
    "tf_rnn": False,
    "tf_tabnet": False,
    "tf_tcn": False,
    "tf_transformer": False,
    "tf_wavenet": False,
    ## Modelos de Aprendizaje por Refuerzo
    "tf_monte_carlo": False,
    "tf_policy_iteration": False,
    "tf_q_learning": False,
    "tf_reinforce_mcpg": False,
    "tf_sarsa": True,
    "tf_value_iteration": False,
    ## Modelos de Aprendizaje por Refuerzo Profundo
    "tf_a2c": False,
    "tf_a3c": False,
    "tf_ddpg": False,
    "tf_dqn": False,
    "tf_ppo": False,
    "tf_sac": False,
    "tf_trpo": False,
}

# Modelos JAX a utilizar
USE_JAX_MODELS = {
    ## Modelos de Aprendizaje Profundo
    "jax_attention_only": False,
    "jax_cnn": False,
    "jax_fnn": False,
    "jax_gru": False,
    "jax_lstm": False,
    "jax_rnn": False,
    "jax_tabnet": False,
    "jax_tcn": False,
    "jax_transformer": False,
    "jax_wavenet": False,
    ## Modelos de Aprendizaje por Refuerzo
    "jax_monte_carlo": False,
    "jax_policy_iteration": False,
    "jax_q_learning": False,
    "jax_reinforce_mcpg": False,
    "jax_sarsa": False,
    "jax_value_iteration": False,
    ## Modelos de Aprendizaje por Refuerzo Profundo
    "jax_a2c": True,
    "jax_a3c": True,
    "jax_ddpg": True,
    "jax_dqn": True,
    "jax_ppo": True,
    "jax_sac": True,
    "jax_trpo": True,
}

# Modelos Pytorch a utilizar
USE_PT_MODELS = {
    ## Modelos de Aprendizaje Profundo
    "pt_attention_only": False,
    "pt_cnn": False,
    "pt_fnn": False,
    "pt_gru": False,
    "pt_lstm": False,
    "pt_rnn": False,
    "pt_tabnet": False,
    "pt_tcn": False,
    "pt_transformer": False,
    "pt_wavenet": False,
    ## Modelos de Aprendizaje por Refuerzo
    "pt_monte_carlo": False,
    "pt_policy_iteration": False,
    "pt_q_learning": False,
    "pt_reinforce_mcpg": False,
    "pt_sarsa": False,
    "pt_value_iteration": False,
    ## Modelos de Aprendizaje por Refuerzo Profundo
    "pt_a2c": True,
    "pt_a3c": True,
    "pt_ddpg": False,
    "pt_dqn": False,
    "pt_ppo": False,
    "pt_sac": False,
    "pt_trpo": False,
}