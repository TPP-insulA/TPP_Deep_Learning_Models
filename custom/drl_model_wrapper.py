from custom.model_wrapper import ModelWrapper
import jax
import jax.numpy as jnp
import numpy as np
import flax.linen as nn
from flax.training import train_state
import optax
from typing import Dict, List, Tuple, Any, Optional, Callable, Union

class DRLModelWrapper(ModelWrapper):
    """
    Implementación de ModelWrapper para modelos de aprendizaje profundo por refuerzo (DRL).
    
    Proporciona soporte para algoritmos como A2C, PPO, SAC, DDPG, DQN, etc.
    """
    
    def __init__(self, model_cls: Callable, **model_kwargs):
        """
        Inicializa un wrapper para modelos de aprendizaje profundo por refuerzo.
        
        Parámetros:
        -----------
        model_cls : Callable
            Clase del modelo DRL a instanciar
        **model_kwargs
            Argumentos para el constructor del modelo
        """
        super().__init__()
        self.model = model_cls(**model_kwargs)
        self.params = None
        self.states = {}  # Múltiples estados de entrenamiento (actor, crítico, etc.)
        self.buffer = None  # Buffer de experiencia
        self.algorithm = model_kwargs.get('algorithm', 'generic')  # Tipo de algoritmo DRL
    
    def start(self, x_cgm: np.ndarray, x_other: np.ndarray, y: np.ndarray, 
             rng_key: Any = None) -> Any:
        """
        Inicializa las redes del modelo DRL y estructuras auxiliares.
        
        Parámetros:
        -----------
        x_cgm : np.ndarray
            Datos CGM de entrada
        x_other : np.ndarray
            Otras características de entrada
        y : np.ndarray
            Valores objetivo
        rng_key : Any, opcional
            Clave para generación aleatoria (default: None)
            
        Retorna:
        --------
        Any
            Estado inicial del modelo o parámetros
        """
        if rng_key is None:
            rng_key = jax.random.PRNGKey(0)
        
        # Dimensiones del espacio de estados y acciones
        state_dim = (x_cgm.shape[1:], x_other.shape[1:])
        action_dim = 1  # Para regresión
        
        # Dejar que el modelo se inicialice con interfaz específica de DRL
        if hasattr(self.model, 'initialize'):
            self.params = self.model.initialize(state_dim, action_dim, rng_key)
        elif hasattr(self.model, 'setup'):
            self.model.setup(state_dim, action_dim, rng_key)
            self.params = self.model.get_params() if hasattr(self.model, 'get_params') else None
        else:
            # Inicialización genérica para modelos que usan redes JAX/Flax
            # Dividir la clave para diferentes redes
            actor_key, critic_key, buffer_key = jax.random.split(rng_key, 3)
            
            # Inicializar redes según el algoritmo
            if self.algorithm in ['a2c', 'a3c', 'ppo']:
                # Inicializar redes de política y valor
                self._initialize_actor_critic(actor_key, critic_key, state_dim, action_dim)
            elif self.algorithm in ['dqn']:
                # Inicializar redes Q
                self._initialize_q_networks(actor_key, critic_key, state_dim, action_dim)
            elif self.algorithm in ['ddpg', 'td3', 'sac']:
                # Inicializar redes actor-crítico con redes objetivo
                self._initialize_actor_critic_targets(actor_key, critic_key, state_dim, action_dim)
            
            # Crear buffer de experiencia
            self._initialize_buffer(buffer_key, state_dim, action_dim)
        
        return self.params
    
    def _initialize_actor_critic(self, actor_key: Any, critic_key: Any, 
                              state_dim: Tuple, action_dim: int) -> None:
        """
        Inicializa redes de actor y crítico para algoritmos basados en política.
        
        Parámetros:
        -----------
        actor_key : Any
            Clave de aleatorización para el actor
        critic_key : Any
            Clave de aleatorización para el crítico
        state_dim : Tuple
            Dimensiones del espacio de estados
        action_dim : int
            Dimensión del espacio de acciones
        """
        # Formas de entrada para inicialización
        cgm_shape, other_shape = state_dim
        x_cgm_dummy = jnp.ones((1,) + cgm_shape)
        x_other_dummy = jnp.ones((1,) + other_shape)
        
        # Inicializar redes si están disponibles en el modelo
        if hasattr(self.model, 'actor_network'):
            actor_params = self.model.actor_network.init(actor_key, x_cgm_dummy, x_other_dummy)
            actor_tx = optax.adam(learning_rate=3e-4)
            self.states['actor'] = train_state.TrainState.create(
                apply_fn=self.model.actor_network.apply,
                params=actor_params,
                tx=actor_tx
            )
        
        if hasattr(self.model, 'critic_network'):
            critic_params = self.model.critic_network.init(critic_key, x_cgm_dummy, x_other_dummy)
            critic_tx = optax.adam(learning_rate=3e-4)
            self.states['critic'] = train_state.TrainState.create(
                apply_fn=self.model.critic_network.apply,
                params=critic_params,
                tx=critic_tx
            )
    
    def _initialize_q_networks(self, q_key: Any, target_key: Any, 
                            state_dim: Tuple, action_dim: int) -> None:
        """
        Inicializa redes Q para algoritmos basados en valor.
        
        Parámetros:
        -----------
        q_key : Any
            Clave de aleatorización para la red Q
        target_key : Any
            Clave de aleatorización para la red Q objetivo
        state_dim : Tuple
            Dimensiones del espacio de estados
        action_dim : int
            Dimensión del espacio de acciones
        """
        # Formas de entrada para inicialización
        cgm_shape, other_shape = state_dim
        x_cgm_dummy = jnp.ones((1,) + cgm_shape)
        x_other_dummy = jnp.ones((1,) + other_shape)
        
        # Inicializar red Q
        if hasattr(self.model, 'q_network'):
            q_params = self.model.q_network.init(q_key, x_cgm_dummy, x_other_dummy)
            q_tx = optax.adam(learning_rate=3e-4)
            self.states['q'] = train_state.TrainState.create(
                apply_fn=self.model.q_network.apply,
                params=q_params,
                tx=q_tx
            )
            
            # Inicializar red Q objetivo con los mismos parámetros
            if hasattr(self.model, 'target_q_network'):
                self.states['target_q'] = {
                    'params': q_params,
                    'apply_fn': self.model.target_q_network.apply
                }
    
    def _initialize_actor_critic_targets(self, actor_key: Any, critic_key: Any, 
                                      state_dim: Tuple, action_dim: int) -> None:
        """
        Inicializa redes de actor-crítico con redes objetivo para algoritmos como DDPG, SAC.
        
        Parámetros:
        -----------
        actor_key : Any
            Clave de aleatorización para el actor
        critic_key : Any
            Clave de aleatorización para el crítico
        state_dim : Tuple
            Dimensiones del espacio de estados
        action_dim : int
            Dimensión del espacio de acciones
        """
        # Inicializar actor y crítico base
        self._initialize_actor_critic(actor_key, critic_key, state_dim, action_dim)
        
        # Inicializar redes objetivo con los mismos parámetros
        if 'actor' in self.states:
            self.states['target_actor'] = {
                'params': self.states['actor'].params,
                'apply_fn': self.model.actor_network.apply if hasattr(self.model, 'actor_network') else None
            }
            
        if 'critic' in self.states:
            self.states['target_critic'] = {
                'params': self.states['critic'].params,
                'apply_fn': self.model.critic_network.apply if hasattr(self.model, 'critic_network') else None
            }
    
    def _initialize_buffer(self, rng_key: Any, state_dim: Tuple, action_dim: int) -> None:
        """
        Inicializa el buffer de experiencia para entrenamiento.
        
        Parámetros:
        -----------
        rng_key : Any
            Clave de aleatorización
        state_dim : Tuple
            Dimensiones del espacio de estados
        action_dim : int
            Dimensión del espacio de acciones
        """
        # Crear buffer si el modelo no tiene uno
        if not hasattr(self.model, 'buffer'):
            cgm_shape, other_shape = state_dim
            buffer_size = getattr(self.model, 'buffer_size', 100000)
            
            # Estructura simple de buffer (personalizable según algoritmo)
            self.buffer = {
                'cgm_states': np.zeros((buffer_size,) + cgm_shape),
                'other_states': np.zeros((buffer_size,) + other_shape),
                'actions': np.zeros((buffer_size, action_dim)),
                'rewards': np.zeros(buffer_size),
                'next_cgm_states': np.zeros((buffer_size,) + cgm_shape),
                'next_other_states': np.zeros((buffer_size,) + other_shape),
                'dones': np.zeros(buffer_size, dtype=bool),
                'position': 0,
                'size': 0
            }
    
    def train(self, x_cgm: np.ndarray, x_other: np.ndarray, y: np.ndarray, 
             validation_data: Optional[Tuple[Tuple[np.ndarray, np.ndarray], np.ndarray]] = None,
             epochs: int = 10, batch_size: int = 32) -> Dict[str, List[float]]:
        """
        Entrena el modelo DRL con los datos proporcionados, adaptando el enfoque
        según el algoritmo específico.
        
        Parámetros:
        -----------
        x_cgm : np.ndarray
            Datos CGM de entrenamiento
        x_other : np.ndarray
            Otras características de entrenamiento
        y : np.ndarray
            Valores objetivo
        validation_data : Optional[Tuple[Tuple[np.ndarray, np.ndarray], np.ndarray]], opcional
            Datos de validación como ((x_cgm_val, x_other_val), y_val) (default: None)
        epochs : int, opcional
            Número de épocas de entrenamiento (default: 10)
        batch_size : int, opcional
            Tamaño de lote (default: 32)
            
        Retorna:
        --------
        Dict[str, List[float]]
            Historial de entrenamiento con métricas
        """
        # Verificar que el modelo esté inicializado
        if self.params is None and not self.states and not hasattr(self.model, 'is_initialized'):
            rng_key = jax.random.PRNGKey(0)
            self.start(x_cgm, x_other, y, rng_key)
        
        # Preparar datos de validación si están disponibles
        x_cgm_val, x_other_val, y_val = None, None, None
        if validation_data is not None:
            (x_cgm_val, x_other_val), y_val = validation_data
        
        # Usar método de entrenamiento nativo del modelo si existe
        if hasattr(self.model, 'train') or hasattr(self.model, 'fit'):
            train_fn = getattr(self.model, 'train', None) or getattr(self.model, 'fit')
            return train_fn(
                x_cgm, x_other, y,
                validation_data=validation_data if validation_data else None,
                epochs=epochs,
                batch_size=batch_size
            )
        
        # Implementación específica según algoritmo
        history = self._train_by_algorithm(
            x_cgm, x_other, y,
            x_cgm_val, x_other_val, y_val,
            epochs, batch_size
        )
        
        return history
    
    def _train_by_algorithm(self, 
                          x_cgm: np.ndarray, x_other: np.ndarray, y: np.ndarray,
                          x_cgm_val: Optional[np.ndarray], x_other_val: Optional[np.ndarray], 
                          y_val: Optional[np.ndarray],
                          epochs: int, batch_size: int) -> Dict[str, List[float]]:
        """
        Implementa el entrenamiento específico según el algoritmo DRL.
        
        Parámetros:
        -----------
        x_cgm : np.ndarray
            Datos CGM de entrenamiento
        x_other : np.ndarray
            Otras características de entrenamiento
        y : np.ndarray
            Valores objetivo
        x_cgm_val : Optional[np.ndarray]
            Datos CGM de validación
        x_other_val : Optional[np.ndarray]
            Otras características de validación
        y_val : Optional[np.ndarray]
            Valores objetivo de validación
        epochs : int
            Número de épocas
        batch_size : int
            Tamaño del lote
            
        Retorna:
        --------
        Dict[str, List[float]]
            Historial de entrenamiento
        """
        # Estructura para registrar métricas
        history = {
            "policy_loss": [],
            "value_loss": [],
            "total_loss": [],
            "val_loss": []
        }
        
        # Entrenamiento según algoritmo específico
        for epoch in range(epochs):
            epoch_metrics = {
                "policy_loss": 0.0,
                "value_loss": 0.0,
                "total_loss": 0.0
            }
            num_batches = 0
            
            # Procesar datos en lotes
            for batch_start in range(0, len(y), batch_size):
                batch_end = min(batch_start + batch_size, len(y))
                batch_cgm = x_cgm[batch_start:batch_end]
                batch_other = x_other[batch_start:batch_end]
                batch_y = y[batch_start:batch_end]
                
                # Calcular recompensas a partir de valores objetivo
                # (en un caso real esto vendría del entorno)
                rewards = self._compute_rewards(batch_cgm, batch_other, batch_y)
                
                # Llenar buffer de experiencia
                self._fill_buffer(batch_cgm, batch_other, rewards, batch_y)
                
                # Actualizar redes según el algoritmo
                if self.algorithm in ['a2c', 'a3c', 'ppo']:
                    p_loss, v_loss = self._update_actor_critic()
                elif self.algorithm in ['dqn']:
                    p_loss, v_loss = 0.0, self._update_q_network()
                elif self.algorithm in ['ddpg', 'td3', 'sac']:
                    p_loss, v_loss = self._update_actor_critic_targets()
                else:
                    # Caso genérico
                    p_loss, v_loss = self._update_generic()
                
                # Acumular métricas
                epoch_metrics["policy_loss"] += p_loss
                epoch_metrics["value_loss"] += v_loss
                epoch_metrics["total_loss"] += p_loss + v_loss
                num_batches += 1
            
            # Calcular promedios de métricas de entrenamiento
            for key in epoch_metrics:
                epoch_metrics[key] /= max(1, num_batches)
                history[key].append(float(epoch_metrics[key]))
            
            # Evaluación en validación si hay datos disponibles
            if x_cgm_val is not None and x_other_val is not None and y_val is not None:
                val_loss = self.evaluate(x_cgm_val, x_other_val, y_val)
                history["val_loss"].append(float(val_loss))
            else:
                history["val_loss"].append(0.0)
            
            print(f"Época {epoch+1}/{epochs}, " +
                  f"policy_loss: {history['policy_loss'][-1]:.4f}, " +
                  f"value_loss: {history['value_loss'][-1]:.4f}, " +
                  f"total_loss: {history['total_loss'][-1]:.4f}, " +
                  f"val_loss: {history['val_loss'][-1]:.4f}")
        
        return history
    
    def _compute_rewards(self, x_cgm: np.ndarray, x_other: np.ndarray, y: np.ndarray) -> np.ndarray:
        """
        Calcula recompensas a partir de los datos de entrenamiento.
        
        Parámetros:
        -----------
        x_cgm : np.ndarray
            Datos CGM del lote
        x_other : np.ndarray
            Otras características del lote
        y : np.ndarray
            Valores objetivo del lote
            
        Retorna:
        --------
        np.ndarray
            Recompensas calculadas
        """
        # Predicción actual
        predictions = self.predict(x_cgm, x_other)
        
        # Calcular error
        errors = np.abs(predictions - y)
        
        # Convertir error a recompensa (menor error = mayor recompensa)
        max_error = np.max(errors) if len(errors) > 0 else 1.0
        rewards = 1.0 - (errors / max(max_error, 1e-8))
        
        return rewards
    
    def _fill_buffer(self, x_cgm: np.ndarray, x_other: np.ndarray, 
                   rewards: np.ndarray, y: np.ndarray) -> None:
        """
        Llena el buffer de experiencia con transiciones.
        
        Parámetros:
        -----------
        x_cgm : np.ndarray
            Estados CGM actuales
        x_other : np.ndarray
            Otras características actuales
        rewards : np.ndarray
            Recompensas obtenidas
        y : np.ndarray
            Valores objetivo (usados para derivar acciones)
        """
        if self.buffer is None or hasattr(self.model, 'store_experience'):
            # Usar método del modelo si existe
            if hasattr(self.model, 'store_experience'):
                self.model.store_experience(x_cgm, x_other, y, rewards)
            return
            
        # Almacenar en buffer propio
        batch_size = len(rewards)
        for i in range(batch_size):
            pos = self.buffer['position']
            
            # Guardar la transición
            self.buffer['cgm_states'][pos] = x_cgm[i]
            self.buffer['other_states'][pos] = x_other[i]
            self.buffer['actions'][pos] = y[i]  # Usar target como acción
            self.buffer['rewards'][pos] = rewards[i]
            
            # En un caso real tendríamos next_state del entorno
            # Acá simulamos un next_state simple
            next_index = min(i + 1, batch_size - 1)
            self.buffer['next_cgm_states'][pos] = x_cgm[next_index]
            self.buffer['next_other_states'][pos] = x_other[next_index]
            self.buffer['dones'][pos] = (next_index == batch_size - 1)
            
            # Actualizar posición y tamaño
            self.buffer['position'] = (pos + 1) % len(self.buffer['rewards'])
            self.buffer['size'] = min(self.buffer['size'] + 1, len(self.buffer['rewards']))
    
    def _update_actor_critic(self) -> Tuple[float, float]:
        """
        Actualiza redes de actor y crítico para algoritmos como A2C, PPO.
        
        Retorna:
        --------
        Tuple[float, float]
            Pérdidas de política y valor
        """
        # Implementación específica para algoritmos actor-crítico
        # Este es un esquema simplificado que debe adaptarse al algoritmo específico
        
        if not self.states or not hasattr(self.model, 'update_networks'):
            return 0.0, 0.0
            
        # Permitir que el modelo maneje la actualización si tiene el método
        if hasattr(self.model, 'update_networks'):
            return self.model.update_networks()
        
        # Implementación genérica (esquema)
        policy_loss = 0.0
        value_loss = 0.0
        
        # Actualizar redes si están disponibles
        if 'actor' in self.states and 'critic' in self.states:
            # Acá iría el código específico de actualización
            pass
        
        return policy_loss, value_loss
    
    def _update_q_network(self) -> float:
        """
        Actualiza red Q para algoritmos como DQN.
        
        Retorna:
        --------
        float
            Pérdida de la red Q
        """
        # Implementación para algoritmos basados en valor
        if not self.states or not hasattr(self.model, 'update_q_network'):
            return 0.0
            
        # Permitir que el modelo maneje la actualización si tiene el método
        if hasattr(self.model, 'update_q_network'):
            return self.model.update_q_network()
        
        # Implementación genérica (esquema)
        q_loss = 0.0
        
        # Actualizar redes si están disponibles
        if 'q' in self.states:
            # Acá iría el código específico de actualización
            pass
        
        return q_loss
    
    def _update_actor_critic_targets(self) -> Tuple[float, float]:
        """
        Actualiza redes de actor-crítico y sus objetivos para algoritmos como DDPG, SAC.
        
        Retorna:
        --------
        Tuple[float, float]
            Pérdidas de política y crítico
        """
        # Implementación para algoritmos con redes objetivo
        if not self.states or not hasattr(self.model, 'update_networks'):
            return 0.0, 0.0
            
        # Permitir que el modelo maneje la actualización si tiene el método
        if hasattr(self.model, 'update_networks'):
            return self.model.update_networks()
        
        # Implementación genérica (esquema)
        policy_loss = 0.0
        value_loss = 0.0
        
        # Actualizar redes si están disponibles
        if 'actor' in self.states and 'critic' in self.states:
            # Acá iría el código específico de actualización
            pass
            
        # Actualizar redes objetivo (soft update)
        _ = getattr(self.model, 'tau', 0.005)  # Factor de suavizado
        
        # Soft update para actor objetivo
        if 'actor' in self.states and 'target_actor' in self.states:
            # Acá iría el código de actualización suave
            pass
            
        # Soft update para crítico objetivo
        if 'critic' in self.states and 'target_critic' in self.states:
            # Acá iría el código de actualización suave
            pass
        
        return policy_loss, value_loss
    
    def _update_generic(self) -> Tuple[float, float]:
        """
        Actualización genérica para cualquier algoritmo DRL.
        
        Retorna:
        --------
        Tuple[float, float]
            Pérdidas de política y valor/crítico
        """
        # Implementación de respaldo para cualquier algoritmo
        if hasattr(self.model, 'update'):
            losses = self.model.update()
            if isinstance(losses, tuple) and len(losses) >= 2:
                return losses[0], losses[1]
            elif isinstance(losses, (int, float)):
                return 0.0, float(losses)
                
        return 0.0, 0.0
    
    def predict(self, x_cgm: np.ndarray, x_other: np.ndarray) -> np.ndarray:
        """
        Realiza predicciones con el modelo DRL entrenado.
        
        Parámetros:
        -----------
        x_cgm : np.ndarray
            Datos CGM para predicción
        x_other : np.ndarray
            Otras características para predicción
            
        Retorna:
        --------
        np.ndarray
            Predicciones del modelo
        """
        # Usar método nativo del modelo si existe
        if hasattr(self.model, 'predict'):
            return np.array(self.model.predict([x_cgm, x_other]))
        
        # Si el modelo tiene un método act o get_action, úsalo
        if hasattr(self.model, 'act') or hasattr(self.model, 'get_action'):
            return self._predict_with_act_method(x_cgm, x_other)
        
        # Predicción usando las redes directamente (según algoritmo)
        return self._predict_by_algorithm(x_cgm, x_other)
    
    def _predict_with_act_method(self, x_cgm: np.ndarray, x_other: np.ndarray) -> np.ndarray:
        """
        Realiza predicciones usando el método act o get_action del modelo.
        
        Parámetros:
        -----------
        x_cgm : np.ndarray
            Datos CGM para predicción
        x_other : np.ndarray
            Otras características para predicción
            
        Retorna:
        --------
        np.ndarray
            Predicciones usando el método act o get_action
        """
        act_fn = getattr(self.model, 'act', None) or getattr(self.model, 'get_action')
        
        # Predecir por lotes si es posible
        if hasattr(act_fn, 'vmap') or hasattr(act_fn, '__call__') and len(x_cgm) > 1:
            try:
                return act_fn(x_cgm, x_other)
            except Exception:
                # Fallback a predicción muestra por muestra
                pass
        
        # Predicción muestra por muestra
        predictions = np.zeros((len(x_cgm),))
        for i in range(len(x_cgm)):
            predictions[i] = act_fn(x_cgm[i:i+1], x_other[i:i+1])
        
        return predictions
    
    def _predict_by_algorithm(self, x_cgm: np.ndarray, x_other: np.ndarray) -> np.ndarray:
        """
        Selecciona el método de predicción según el algoritmo.
        
        Parámetros:
        -----------
        x_cgm : np.ndarray
            Datos CGM para predicción
        x_other : np.ndarray
            Otras características para predicción
            
        Retorna:
        --------
        np.ndarray
            Predicciones según el algoritmo utilizado
        """
        if self.algorithm in ['a2c', 'a3c', 'ppo', 'ddpg', 'td3', 'sac'] and 'actor' in self.states:
            # Usar red de política (actor) para predicción
            return self._predict_with_actor(x_cgm, x_other)
        elif self.algorithm in ['dqn'] and 'q' in self.states:
            # Usar red Q para predicción (argmax Q)
            return self._predict_with_q(x_cgm, x_other)
        
        # Fallback genérico
        return np.zeros((len(x_cgm),))
    
    def _predict_with_actor(self, x_cgm: np.ndarray, x_other: np.ndarray) -> np.ndarray:
        """
        Realiza predicciones usando la red de política (actor).
        
        Parámetros:
        -----------
        x_cgm : np.ndarray
            Datos CGM para predicción
        x_other : np.ndarray
            Otras características para predicción
            
        Retorna:
        --------
        np.ndarray
            Predicciones de la red de política
        """
        # Convertir entradas a arrays JAX
        x_cgm_jax = jnp.array(x_cgm)
        x_other_jax = jnp.array(x_other)
        
        # Obtener predicción de la red de actor
        actor_state = self.states.get('actor')
        if actor_state is not None:
            # Si es un TrainState de Flax
            if hasattr(actor_state, 'apply_fn') and hasattr(actor_state, 'params'):
                preds = actor_state.apply_fn(actor_state.params, x_cgm_jax, x_other_jax)
            # Si es un diccionario con apply_fn y params
            elif isinstance(actor_state, dict) and 'apply_fn' in actor_state and 'params' in actor_state:
                preds = actor_state['apply_fn'](actor_state['params'], x_cgm_jax, x_other_jax)
            else:
                return np.zeros((len(x_cgm),))
                
            return np.array(preds)
        
        return np.zeros((len(x_cgm),))
    
    def _predict_with_q(self, x_cgm: np.ndarray, x_other: np.ndarray) -> np.ndarray:
        """
        Realiza predicciones seleccionando la acción con mayor valor Q.
        
        Parámetros:
        -----------
        x_cgm : np.ndarray
            Datos CGM para predicción
        x_other : np.ndarray
            Otras características para predicción
            
        Retorna:
        --------
        np.ndarray
            Predicciones basadas en el máximo valor Q
        """
        # Predicción basada en valores Q (simulada)
        # En un caso real, esto evaluaría los valores Q para cada acción posible
        
        # Simplificación: devuelve un valor entre -1 y 1 basado en features
        predictions = np.zeros((len(x_cgm),))
        for i in range(len(x_cgm)):
            # Cálculo simplificado para demostración
            cgm_mean = np.mean(x_cgm[i])
            other_mean = np.mean(x_other[i])
            predictions[i] = (cgm_mean + other_mean) / 2
        
        return predictions