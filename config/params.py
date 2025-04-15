# Modelos
## Modelos TensorFlow
### Modelos de Aprendizaje Profundo
from models.tensorflow.DeepLearning.attention_only import create_attention_model as tf_create_attention_model
from models.tensorflow.DeepLearning.cnn import create_cnn_model as tf_create_cnn_model
from models.tensorflow.DeepLearning.fnn import create_fnn_model as tf_create_fnn_model
from models.tensorflow.DeepLearning.gru import create_gru_model as tf_create_gru_model
from models.tensorflow.DeepLearning.lstm import create_lstm_model as tf_create_lstm_model
from models.tensorflow.DeepLearning.rnn import create_rnn_model as tf_create_rnn_model
from models.tensorflow.DeepLearning.tabnet import create_tabnet_model as tf_create_tabnet_model
from models.tensorflow.DeepLearning.tcn import create_tcn_model as tf_create_tcn_model
from models.tensorflow.DeepLearning.transformer import create_transformer_model as tf_create_transformer_model
from models.tensorflow.DeepLearning.wavenet import create_wavenet_model as tf_create_wavenet_model

### Modelos de Aprendizaje por Refuerzo
from models.tensorflow.ReinforcementLearning.monte_carlo_methods import create_monte_carlo_model as tf_create_monte_carlo_model
from models.tensorflow.ReinforcementLearning.policy_iteration import create_policy_iteration_model as tf_create_policy_iteration_model
from models.tensorflow.ReinforcementLearning.q_learning import create_q_learning_model as tf_create_q_learning_model
from models.tensorflow.ReinforcementLearning.reinforce_mcpg import create_reinforce_mcpg_model as tf_create_reinforce_mcpg_model
from models.tensorflow.ReinforcementLearning.sarsa import create_sarsa_model as tf_create_sarsa_model
from models.tensorflow.ReinforcementLearning.value_iteration import create_value_iteration_model as tf_create_value_iteration_model

### Modelos de Aprendizaje por Refuerzo Profundo
from models.tensorflow.DeepReinforcementLearning.a2c_a3c import create_a2c_model as tf_create_a2c_model, create_a3c_model as tf_create_a3c_model
from models.tensorflow.DeepReinforcementLearning.ddpg import create_ddpg_model as tf_create_ddpg_model
from models.tensorflow.DeepReinforcementLearning.dqn import create_dqn_model as tf_create_dqn_model
from models.tensorflow.DeepReinforcementLearning.ppo import create_ppo_model as tf_create_ppo_model
from models.tensorflow.DeepReinforcementLearning.sac import create_sac_model as tf_create_sac_model
from models.tensorflow.DeepReinforcementLearning.trpo import create_trpo_model as tf_create_trpo_model

## Modelos JAX
### Modelos de Aprendizaje Profundo
from models.jax.DeepLearning.attention_only import model_creator as jax_create_attention_model
from models.jax.DeepLearning.cnn import model_creator as jax_create_cnn_model
from models.jax.DeepLearning.fnn import model_creator as jax_create_fnn_model
from models.jax.DeepLearning.gru import model_creator as jax_create_gru_model
from models.jax.DeepLearning.lstm import model_creator as jax_create_lstm_model
from models.jax.DeepLearning.rnn import model_creator as jax_create_rnn_model
from models.jax.DeepLearning.tabnet import model_creator as jax_create_tabnet_model
from models.jax.DeepLearning.tcn import model_creator as jax_create_tcn_model
from models.jax.DeepLearning.transformer import model_creator as jax_create_transformer_model
from models.jax.DeepLearning.wavenet import model_creator as jax_create_wavenet_model

### Modelos de Aprendizaje por Refuerzo
from models.jax.ReinforcementLearning.monte_carlo_methods import model_creator as jax_monte_carlo_creator
from models.jax.ReinforcementLearning.policy_iteration import model_creator as jax_policy_iteration_creator
from models.jax.ReinforcementLearning.q_learning import model_creator as jax_q_learning_creator
from models.jax.ReinforcementLearning.reinforce_mcgp import model_creator as jax_reinforce_mcgp_creator
from models.jax.ReinforcementLearning.sarsa import model_creator as jax_sarsa_creator
from models.jax.ReinforcementLearning.value_iteration import model_creator as jax_value_iteration_creator

### Modelos de Aprendizaje por Refuerzo Profundo
from models.jax.DeepReinforcementLearning.a2c_a3c import model_creator_a2c as jax_a2c_creator, model_creator_a3c as jax_a3c_creator
from models.jax.DeepReinforcementLearning.ddpg import model_creator as jax_ddpg_creator
from models.jax.DeepReinforcementLearning.dqn import model_creator as jax_dqn_creator
from models.jax.DeepReinforcementLearning.ppo import model_creator as jax_ppo_creator
from models.jax.DeepReinforcementLearning.sac import model_creator as jax_sac_creator
from models.jax.DeepReinforcementLearning.trpo import model_creator as jax_trpo_creator

# Modo de Ejecución
DEBUG = True 

# Configuración de procesamiento
## Framework a utilizar durante la ejecución. Puede ser con TensorFlow o JAX.
## Opciones: "tensorflow", "jax"
FRAMEWORK = "tensorflow"
## Procesamiento de datos. Puede ser con pandas o polars.
## Opciones: "pandas", "polars"
PROCESSING = "polars"
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

# Modelos TensorFlow a utilizar
USE_TF_MODELS = {
    ## Modelos de Aprendizaje Profundo
    "tf_attention_only": True,
    "tf_cnn": True,
    "tf_fnn": True,
    "tf_gru": True,
    "tf_lstm": True,
    "tf_rnn": True,
    "tf_tabnet": True,
    "tf_tcn": False,
    "tf_transformer": False,
    "tf_wavenet": False,
    ## Modelos de Aprendizaje por Refuerzo
    "tf_monte_carlo": False,
    "tf_policy_iteration": False,
    "tf_q_learning": False,
    "tf_reinforce_mcpg": False,
    "tf_sarsa": False,
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
    "jax_attention_only": True,
    "jax_cnn": True,
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
    "jax_a2c": False,
    "jax_a3c": False,
    "jax_ddpg": False,
    "jax_dqn": False,
    "jax_ppo": False,
    "jax_sac": False,
    "jax_trpo": False,
}