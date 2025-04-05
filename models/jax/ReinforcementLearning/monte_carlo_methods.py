import os, sys
import jax
import jax.numpy as jnp
from jax import jit, grad, random, vmap
import numpy as np
import matplotlib.pyplot as plt
import time
from typing import Dict, List, Tuple, Optional, Union, Any, Callable
import pickle
from functools import partial

PROJECT_ROOT = os.path.abspath(os.getcwd())
sys.path.append(PROJECT_ROOT)

from config.models_config import MONTE_CARLO_CONFIG


class MonteCarlo:
    """
    Implementación de métodos Monte Carlo para predicción y control en aprendizaje por refuerzo usando JAX.
    
    Esta clase proporciona implementaciones de:
    1. Predicción Monte Carlo (first-visit y every-visit) para evaluar políticas
    2. Control Monte Carlo (on-policy y off-policy) para encontrar políticas óptimas
    
    Se incluyen algoritmos como:
    - First-visit MC prediction
    - Every-visit MC prediction
    - Monte Carlo Exploring Starts (MCES)
    - On-policy MC control con epsilon-greedy
    - Off-policy MC control con importance sampling
    """
    
    def __init__(
        self,
        n_states: int,
        n_actions: int,
        gamma: float = MONTE_CARLO_CONFIG['gamma'],
        epsilon_start: float = MONTE_CARLO_CONFIG['epsilon_start'],
        epsilon_end: float = MONTE_CARLO_CONFIG['epsilon_end'],
        epsilon_decay: float = MONTE_CARLO_CONFIG['epsilon_decay'],
        first_visit: bool = MONTE_CARLO_CONFIG['first_visit'],
        evaluation_mode: bool = False,
        seed: int = 42
    ) -> None:
        """
        Inicializa el agente de Monte Carlo con JAX.
        
        Parámetros:
        -----------
        n_states : int
            Número de estados en el entorno
        n_actions : int
            Número de acciones en el entorno
        gamma : float, opcional
            Factor de descuento para recompensas futuras (default: MONTE_CARLO_CONFIG['gamma'])
        epsilon_start : float, opcional
            Valor inicial de epsilon para políticas epsilon-greedy (default: MONTE_CARLO_CONFIG['epsilon_start'])
        epsilon_end : float, opcional
            Valor mínimo de epsilon (default: MONTE_CARLO_CONFIG['epsilon_end'])
        epsilon_decay : float, opcional
            Factor de decaimiento de epsilon (default: MONTE_CARLO_CONFIG['epsilon_decay'])
        first_visit : bool, opcional
            Si True, usa first-visit MC, sino every-visit MC (default: MONTE_CARLO_CONFIG['first_visit'])
        evaluation_mode : bool, opcional
            Si True, inicializa en modo evaluación de política (sin control) (default: False)
        seed : int, opcional
            Semilla para generación de números aleatorios (default: 42)
        """
        # Inicializar clave para generador de números aleatorios de JAX
        self.key = jax.random.PRNGKey(seed)
        
        self.n_states = n_states
        self.n_actions = n_actions
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.first_visit = first_visit
        self.evaluation_mode = evaluation_mode
        
        # Inicializar tablas de valor de acción (Q) y política usando arrays de JAX
        self.q_table = jnp.zeros((n_states, n_actions))
        
        # Para modo de evaluación, la política es fija (proporcionada externamente)
        # Para control, comenzamos con una política epsilon-greedy derivada de Q
        self.policy = jnp.ones((n_states, n_actions)) / n_actions  # Inicialmente equiprobable
        
        # Contadores para calcular promedios incrementales
        # Usamos arrays de NumPy para estos contadores porque se actualizarán in-place
        self.returns_sum = np.zeros((n_states, n_actions))
        self.returns_count = np.zeros((n_states, n_actions))
        
        # Para evaluación de política (valor de estado)
        self.v_table = jnp.zeros(n_states)
        self.state_returns_sum = np.zeros(n_states)
        self.state_returns_count = np.zeros(n_states)
        
        # Para off-policy Monte Carlo
        self.c_table = np.zeros((n_states, n_actions))  # Pesos acumulativos para importance sampling
        
        # Métricas (usamos listas de Python ya que JAX no admite mutabilidad)
        self.episode_rewards = []
        self.episode_lengths = []
        self.policy_changes = []
        self.value_changes = []
        self.epsilon_history = []
    
    def reset_counters(self) -> None:
        """
        Reinicia los contadores de retornos para un nuevo entrenamiento.
        """
        self.returns_sum = np.zeros((self.n_states, self.n_actions))
        self.returns_count = np.zeros((self.n_states, self.n_actions))
        self.state_returns_sum = np.zeros(self.n_states)
        self.state_returns_count = np.zeros(self.n_states)
        self.c_table = np.zeros((self.n_states, self.n_actions))
    
    def get_action(self, state: int, explore: bool = True) -> int:
        """
        Selecciona una acción según la política actual, con exploración opcional.
        
        Parámetros:
        -----------
        state : int
            Estado actual
        explore : bool, opcional
            Si es True, usa política epsilon-greedy; si es False, usa política greedy (default: True)
            
        Retorna:
        --------
        int
            La acción seleccionada
        """
        # Generar nueva clave para mantener la aleatoriedad
        self.key, subkey = jax.random.split(self.key)
        
        if explore and jax.random.uniform(subkey) < self.epsilon:
            # Exploración: acción aleatoria
            self.key, subkey = jax.random.split(self.key)
            return int(jax.random.randint(subkey, (), 0, self.n_actions))
        
        # Explotación: mejor acción según la política actual
        if self.evaluation_mode:
            # En modo evaluación, muestreamos de la distribución de política
            self.key, subkey = jax.random.split(self.key)
            return int(jax.random.choice(subkey, self.n_actions, p=np.array(self.policy[state])))
        # En modo control, elegimos la acción greedy (máximo Q)
        return int(jnp.argmax(self.q_table[state]))
    
    def update_policy(self, state: int) -> bool:
        """
        Actualiza la política para el estado dado basándose en los valores Q actuales.
        
        Parámetros:
        -----------
        state : int
            Estado para el cual actualizar la política
            
        Retorna:
        --------
        bool
            Boolean indicando si la política cambió
        """
        if self.evaluation_mode:
            # En modo evaluación, no se actualiza la política
            return False
        
        old_action = int(jnp.argmax(self.policy[state]))
        best_action = int(jnp.argmax(self.q_table[state]))
        
        # Política epsilon-greedy basada en Q
        new_policy = np.zeros(self.n_actions)
        
        # Probabilidad pequeña de exploración
        new_policy += self.epsilon / self.n_actions
        
        # Mayor probabilidad para la mejor acción
        new_policy[best_action] += (1 - self.epsilon)
        
        # Actualizar la política (como np.array para permitir mutabilidad)
        self.policy = self.policy.at[state].set(new_policy)
        
        return old_action != best_action
    
    def decay_epsilon(self, episode: Optional[int] = None) -> None:
        """
        Reduce el valor de epsilon según la estrategia de decaimiento.
        
        Parámetros:
        -----------
        episode : Optional[int], opcional
            Número del episodio actual (para decaimientos basados en episodios) (default: None)
        """
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)
        self.epsilon_history.append(self.epsilon)
    
    @staticmethod
    @jit
    def _calculate_returns_jit(rewards: jnp.ndarray, gamma: float) -> jnp.ndarray:
        """
        Función JIT-compilada para calcular los retornos descontados.
        
        Parámetros:
        -----------
        rewards : jnp.ndarray
            Array de recompensas
        gamma : float
            Factor de descuento
            
        Retorna:
        --------
        jnp.ndarray
            Array de retornos descontados
        """
        n = len(rewards)
        returns = jnp.zeros(n)
        
        def body_fun(i, val):
            returns, G = val
            G = rewards[n - 1 - i] + gamma * G
            returns = returns.at[n - 1 - i].set(G)
            return returns, G
        
        returns, _ = jax.lax.fori_loop(0, n, body_fun, (returns, 0.0))
        return returns
    
    def calculate_returns(self, rewards: List[float]) -> jnp.ndarray:
        """
        Calcula los retornos descontados para cada paso de tiempo en un episodio.
        
        Parámetros:
        -----------
        rewards : List[float]
            Lista de recompensas recibidas durante el episodio
            
        Retorna:
        --------
        jnp.ndarray
            Array de retornos (G_t) para cada paso de tiempo
        """
        rewards_array = jnp.array(rewards)
        return self._calculate_returns_jit(rewards_array, self.gamma)
    
    def monte_carlo_prediction(self, episodes: List[Tuple[List[int], List[int], List[float]]]) -> jnp.ndarray:
        """
        Realiza predicción Monte Carlo (evaluación de política) usando episodios proporcionados.
        
        Parámetros:
        -----------
        episodes : List[Tuple[List[int], List[int], List[float]]]
            Lista de episodios, cada uno como una tupla de (estados, acciones, recompensas)
        
        Retorna:
        --------
        jnp.ndarray
            v_table actualizada (función de valor de estado)
        """
        old_v = self.v_table
        
        for states, actions, rewards in episodes:
            returns = self.calculate_returns(rewards)
            
            # Procesar cada paso en el episodio
            visited_state_steps = set()
            
            for t in range(len(states)):
                state = states[t]
                
                # Para first-visit MC, solo consideramos la primera visita a cada estado
                if self.first_visit and state in visited_state_steps:
                    continue
                    
                visited_state_steps.add(state)
                
                # Actualizar el conteo y la suma de retornos para este estado
                self.state_returns_sum[state] += float(returns[t])
                self.state_returns_count[state] += 1
                
                # Actualizar la función de valor usando promedio incremental
                if self.state_returns_count[state] > 0:
                    self.v_table = self.v_table.at[state].set(
                        self.state_returns_sum[state] / self.state_returns_count[state]
                    )
        
        # Calcular cambio en la función de valor
        value_change = float(jnp.mean(jnp.abs(self.v_table - old_v)))
        self.value_changes.append(value_change)
        
        return self.v_table
    
    def _run_episode(self, env: Any, max_steps: int, render: bool = False) -> Tuple[List[int], List[int], List[float]]:
        """
        Ejecuta un episodio completo en el entorno.
        
        Parámetros:
        -----------
        env : Any
            Entorno donde ejecutar el episodio
        max_steps : int
            Número máximo de pasos
        render : bool
            Si renderizar o no el entorno
            
        Retorna:
        --------
        Tuple[List[int], List[int], List[float]]
            Estados, acciones y recompensas del episodio
        """
        state, _ = env.reset()
        episode_states = []
        episode_actions = []
        episode_rewards = []
        
        for _ in range(max_steps):
            if render:
                env.render()
            
            # Seleccionar acción según política epsilon-greedy
            action = self.get_action(state, explore=True)
            
            # Dar un paso en el entorno
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            
            # Guardar transición
            episode_states.append(state)
            episode_actions.append(action)
            episode_rewards.append(reward)
            
            # Actualizar estado
            state = next_state
            
            if done:
                break
                
        return episode_states, episode_actions, episode_rewards
    
    def _update_q_values(self, episode_states: List[int], episode_actions: List[int], returns: jnp.ndarray) -> bool:
        """
        Actualiza los valores Q y la política para un episodio completo.
        
        Parámetros:
        -----------
        episode_states : List[int]
            Estados visitados en el episodio
        episode_actions : List[int]
            Acciones tomadas en el episodio
        returns : jnp.ndarray
            Retornos calculados para el episodio
            
        Retorna:
        --------
        bool
            Si la política cambió en algún estado
        """
        policy_changed = False
        visited_state_action_pairs = set()
        
        for t in range(len(episode_states)):
            state = episode_states[t]
            action = episode_actions[t]
            
            # Para first-visit MC, solo considerar primera visita a cada par estado-acción
            state_action = (state, action)
            if self.first_visit and state_action in visited_state_action_pairs:
                continue
            
            visited_state_action_pairs.add(state_action)
            
            # Actualizar conteos y sumas para este par estado-acción
            self.returns_sum[state, action] += float(returns[t])
            self.returns_count[state, action] += 1
            
            # Actualizar valor Q usando promedio incremental
            if self.returns_count[state, action] > 0:
                self.q_table = self.q_table.at[state, action].set(
                    self.returns_sum[state, action] / self.returns_count[state, action]
                )
                
                # Actualizar política basada en nuevo valor Q
                if self.update_policy(state):
                    policy_changed = True
                    
        return policy_changed
    
    def monte_carlo_control_on_policy(
        self, 
        env: Any, 
        episodes: int = MONTE_CARLO_CONFIG['episodes'], 
        max_steps: int = MONTE_CARLO_CONFIG['max_steps'],
        render: bool = False
    ) -> Dict[str, List[float]]:
        """
        Implementa control Monte Carlo on-policy con epsilon-greedy.
        
        Parámetros:
        -----------
        env : Any
            Entorno de OpenAI Gym o similar
        episodes : int, opcional
            Número de episodios a ejecutar (default: MONTE_CARLO_CONFIG['episodes'])
        max_steps : int, opcional
            Número máximo de pasos por episodio (default: MONTE_CARLO_CONFIG['max_steps'])
        render : bool, opcional
            Si renderizar o no el entorno (default: False)
            
        Retorna:
        --------
        Dict[str, List[float]]
            Historia del entrenamiento
        """
        start_time = time.time()
        
        for episode in range(episodes):
            # Ejecutar un episodio completo
            episode_states, episode_actions, episode_rewards = self._run_episode(env, max_steps, render)
            
            # Calcular retornos para el episodio
            returns = self.calculate_returns(episode_rewards)
            
            # Actualizar función de valor de acción (Q) y política
            policy_changed = self._update_q_values(episode_states, episode_actions, returns)
            
            # Registrar métricas del episodio
            self.episode_rewards.append(sum(episode_rewards))
            self.episode_lengths.append(len(episode_rewards))
            self.policy_changes.append(1 if policy_changed else 0)
            
            # Decaer epsilon
            self.decay_epsilon(episode)
            
            # Mostrar progreso periódicamente
            if (episode + 1) % MONTE_CARLO_CONFIG['log_interval'] == 0 or episode == 0:
                self._log_progress(episode, episodes, start_time)
        
        # Crear historial de entrenamiento
        return {
            'episode_rewards': self.episode_rewards,
            'episode_lengths': self.episode_lengths,
            'policy_changes': self.policy_changes,
            'value_changes': self.value_changes,
            'epsilon_history': self.epsilon_history,
            'training_time': time.time() - start_time
        }
    
    def _execute_off_policy_episode(
        self,
        env: Any,
        behavior_epsilon: float,
        max_steps: int,
        render: bool = False
    ) -> Tuple[List[int], List[int], List[float], List[float]]:
        """
        Ejecuta un episodio completo usando la política de comportamiento off-policy.
        
        Parámetros:
        -----------
        env : Any
            Entorno donde ejecutar el episodio
        behavior_epsilon : float
            Epsilon para la política de comportamiento
        max_steps : int
            Número máximo de pasos por episodio
        render : bool, opcional
            Si renderizar o no el entorno (default: False)
            
        Retorna:
        --------
        Tuple[List[int], List[int], List[float], List[float]]
            Estados, acciones, recompensas y probabilidades de comportamiento
        """
        state, _ = env.reset()
        episode_states = []
        episode_actions = []
        episode_rewards = []
        episode_behavior_probs = []  # Probabilidades bajo política de comportamiento
        
        for _ in range(max_steps):
            if render:
                env.render()
            
            # Seleccionar acción usando política de comportamiento (más exploratoria)
            self.key, subkey = jax.random.split(self.key)
            if jax.random.uniform(subkey) < behavior_epsilon:
                self.key, subkey = jax.random.split(self.key)
                action = int(jax.random.randint(subkey, (), 0, self.n_actions))
                behavior_prob = behavior_epsilon / self.n_actions
            else:
                action = int(jnp.argmax(self.q_table[state]))
                behavior_prob = 1 - behavior_epsilon + (behavior_epsilon / self.n_actions)
            
            # Dar un paso en el entorno
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            
            # Guardar transición
            episode_states.append(state)
            episode_actions.append(action)
            episode_rewards.append(reward)
            episode_behavior_probs.append(behavior_prob)
            
            # Actualizar estado
            state = next_state
            
            if done:
                break
                
        return episode_states, episode_actions, episode_rewards, episode_behavior_probs
    
    def _update_off_policy_q_values(
        self,
        episode_states: List[int],
        episode_actions: List[int],
        episode_rewards: List[float],
        episode_behavior_probs: List[float]
    ) -> None:
        """
        Actualiza los valores Q usando importance sampling para off-policy.
        
        Parámetros:
        -----------
        episode_states : List[int]
            Estados visitados en el episodio
        episode_actions : List[int]
            Acciones tomadas en el episodio
        episode_rewards : List[float]
            Recompensas recibidas en el episodio
        episode_behavior_probs : List[float]
            Probabilidades bajo política de comportamiento
        """
        # Calcular retornos para el episodio
        _ = self.calculate_returns(episode_rewards)
        
        # Calcular ratios de importancia y actualizar Q
        G = 0.0
        W = 1.0  # Peso de importancia inicial
        
        # Recorremos el episodio en orden inverso (importante para OFF-policy MC)
        for t in range(len(episode_states) - 1, -1, -1):
            state = episode_states[t]
            action = episode_actions[t]
            reward = episode_rewards[t]
            
            # Actualizar retorno acumulado
            G = reward + self.gamma * G
            
            # Actualizar contador de visitas para este par estado-acción
            self.c_table[state, action] += W
            
            # Actualizar función Q usando importance sampling
            current_q = float(self.q_table[state, action])
            new_q = current_q + (W / self.c_table[state, action]) * (G - current_q)
            self.q_table = self.q_table.at[state, action].set(new_q)
            
            # Actualizar política target (greedy respecto a Q)
            self.update_policy(state)
            
            # Obtener probabilidad bajo policy target (greedy)
            target_policy_prob = 1.0 if action == int(jnp.argmax(self.q_table[state])) else 0.0
            
            # Actualizar ratio de importancia
            if target_policy_prob < 1e-10:  # Use small threshold instead of exact 0.0
                # Si la acción no sería elegida por la política target,
                # terminamos el procesamiento de este episodio
                break
            
            W *= target_policy_prob / episode_behavior_probs[t]
    
    def monte_carlo_control_off_policy(
        self, 
        env: Any, 
        episodes: int = MONTE_CARLO_CONFIG['episodes'], 
        max_steps: int = MONTE_CARLO_CONFIG['max_steps'],
        render: bool = False
    ) -> Dict[str, List[float]]:
        """
        Implementa control Monte Carlo off-policy con importance sampling.
        
        Parámetros:
        -----------
        env : Any
            Entorno de OpenAI Gym o similar
        episodes : int, opcional
            Número de episodios a ejecutar (default: MONTE_CARLO_CONFIG['episodes'])
        max_steps : int, opcional
            Número máximo de pasos por episodio (default: MONTE_CARLO_CONFIG['max_steps'])
        render : bool, opcional
            Si renderizar o no el entorno (default: False)
            
        Retorna:
        --------
        Dict[str, List[float]]
            Historia del entrenamiento
        """
        start_time = time.time()
        
        # Política de comportamiento (behavior policy) - más exploratoria
        behavior_epsilon = max(0.1, self.epsilon)
        
        for episode in range(episodes):
            # Ejecutar un episodio completo usando política de comportamiento
            episode_data = self._execute_off_policy_episode(env, behavior_epsilon, max_steps, render)
            episode_states, episode_actions, episode_rewards, episode_behavior_probs = episode_data
            
            # Actualizar valores Q y política usando importance sampling
            self._update_off_policy_q_values(episode_states, episode_actions, episode_rewards, episode_behavior_probs)
            
            # Registrar métricas del episodio
            self.episode_rewards.append(sum(episode_rewards))
            self.episode_lengths.append(len(episode_rewards))
            
            # Decaer epsilon
            self.decay_epsilon(episode)
            
            # Mostrar progreso periódicamente
            if (episode + 1) % MONTE_CARLO_CONFIG['log_interval'] == 0 or episode == 0:
                avg_reward = np.mean(self.episode_rewards[-MONTE_CARLO_CONFIG['log_interval']:])
                elapsed_time = time.time() - start_time
                
                print(f"Episodio {episode+1}/{episodes} - Recompensa promedio: {avg_reward:.2f}, "
                      f"Tiempo: {elapsed_time:.2f}s")
        
        # Crear historial de entrenamiento
        history = {
            'episode_rewards': self.episode_rewards,
            'episode_lengths': self.episode_lengths,
            'epsilon_history': self.epsilon_history,
            'training_time': time.time() - start_time
        }
        
        return history
    
    def _start_exploring_episode(self, env: Any) -> Tuple[int, List[int], List[int], List[float]]:
        """
        Inicializa un episodio con exploring starts.
        
        Parámetros:
        -----------
        env : Any
            Entorno donde iniciar el episodio
            
        Retorna:
        --------
        Tuple[int, List[int], List[int], List[float]]
            Estado actual, estados visitados, acciones tomadas, recompensas recibidas
        """
        # Iniciar con un estado aleatorio si es posible
        if hasattr(env, 'set_state'):
            self.key, subkey = jax.random.split(self.key)
            random_state = int(jax.random.randint(subkey, (), 0, self.n_states))
            env.set_state(random_state)
            state = random_state
        else:
            # Si no podemos establecer el estado, iniciamos normalmente
            state, _ = env.reset()
            
        # Seleccionar una primera acción aleatoria para exploring starts
        self.key, subkey = jax.random.split(self.key)
        action = int(jax.random.randint(subkey, (), 0, self.n_actions))
        
        # Ejecutar la primera acción
        next_state, reward, terminated, truncated, _ = env.step(action)
        
        # Inicializar listas para la trayectoria
        episode_states = [state]
        episode_actions = [action]
        episode_rewards = [reward]
        
        return next_state, terminated or truncated, episode_states, episode_actions, episode_rewards
    
    def _continue_episode(
        self, 
        env: Any, 
        state: int, 
        done: bool, 
        steps: int,
        max_steps: int,
        episode_states: List[int], 
        episode_actions: List[int], 
        episode_rewards: List[float],
        render: bool = False
    ) -> Tuple[List[int], List[int], List[float]]:
        """
        Continúa un episodio ya iniciado hasta su finalización.
        
        Parámetros:
        -----------
        env : Any
            Entorno donde continuar el episodio
        state : int
            Estado actual
        done : bool
            Flag indicando si el episodio ya terminó
        steps : int
            Contador de pasos actuales
        max_steps : int
            Máximo número de pasos permitidos
        episode_states : List[int]
            Estados ya visitados
        episode_actions : List[int]
            Acciones ya tomadas
        episode_rewards : List[float]
            Recompensas ya recibidas
        render : bool, opcional
            Si renderizar o no el entorno (default: False)
            
        Retorna:
        --------
        Tuple[List[int], List[int], List[float]]
            Estados visitados, acciones tomadas, recompensas recibidas actualizados
        """
        current_state = state
        current_done = done
        current_steps = steps
        
        while not current_done and current_steps < max_steps:
            if render:
                env.render()
            
            # Seleccionar acción según política actual (sin exploración adicional)
            action = self.get_action(current_state, explore=False)
            
            # Dar un paso en el entorno
            next_state, reward, terminated, truncated, _ = env.step(action)
            current_done = terminated or truncated
            
            # Guardar transición
            episode_states.append(current_state)
            episode_actions.append(action)
            episode_rewards.append(reward)
            
            # Actualizar estado
            current_state = next_state
            current_steps += 1
            
            if current_done:
                break
                
        return episode_states, episode_actions, episode_rewards
    
    def _update_policy_deterministic(self, state: int) -> bool:
        """
        Actualiza la política de forma determinística (para MCES).
        
        Parámetros:
        -----------
        state : int
            Estado para actualizar política
            
        Retorna:
        --------
        bool
            Si la política cambió o no
        """
        old_action = int(jnp.argmax(self.policy[state]))
        best_action = int(jnp.argmax(self.q_table[state]))
        
        if old_action != best_action:
            # En MCES, la política es totalmente greedy (determinística)
            new_policy = np.zeros(self.n_actions)
            new_policy[best_action] = 1.0
            self.policy = self.policy.at[state].set(new_policy)
            return True
        
        return False
    
    def monte_carlo_exploring_starts(
        self, 
        env: Any, 
        episodes: int = MONTE_CARLO_CONFIG['episodes'], 
        max_steps: int = MONTE_CARLO_CONFIG['max_steps'], 
        render: bool = False
    ) -> Dict[str, List[float]]:
        """
        Implementa control Monte Carlo con exploring starts (MCES).
        
        Nota: Este método solo funciona para entornos que permiten establecer el estado inicial.
        
        Parámetros:
        -----------
        env : Any
            Entorno de OpenAI Gym o similar con soporte para establecer estado
        episodes : int, opcional
            Número de episodios a ejecutar (default: MONTE_CARLO_CONFIG['episodes'])
        max_steps : int, opcional
            Número máximo de pasos por episodio (default: MONTE_CARLO_CONFIG['max_steps'])
        render : bool, opcional
            Si renderizar o no el entorno (default: False)
            
        Retorna:
        --------
        Dict[str, List[float]]
            Historia del entrenamiento
        """
        start_time = time.time()
        
        # Verificar si el entorno soporta establecer estados
        if not hasattr(env, 'set_state'):
            print("Advertencia: Este entorno no parece soportar 'set_state'. El método MCES puede no funcionar correctamente.")
        
        for episode in range(episodes):
            # Iniciar episodio con exploring starts
            state, done, episode_states, episode_actions, episode_rewards = self._start_exploring_episode(env)
            
            # Continuar el episodio hasta que termine
            episode_states, episode_actions, episode_rewards = self._continue_episode(
                env, state, done, 1, max_steps, episode_states, episode_actions, episode_rewards, render
            )
            
            # Calcular retornos para el episodio
            returns = self.calculate_returns(episode_rewards)
            
            # Actualizar función Q y política
            policy_changed = self._update_episode_q_values_mces(episode_states, episode_actions, returns)
            
            # Registrar métricas del episodio
            self.episode_rewards.append(sum(episode_rewards))
            self.episode_lengths.append(len(episode_rewards))
            self.policy_changes.append(1 if policy_changed else 0)
            
            # Mostrar progreso periódicamente
            if (episode + 1) % MONTE_CARLO_CONFIG['log_interval'] == 0 or episode == 0:
                self._log_progress(episode, episodes, start_time)
        
        # Crear y retornar historial de entrenamiento
        return {
            'episode_rewards': self.episode_rewards,
            'episode_lengths': self.episode_lengths,
            'policy_changes': self.policy_changes,
            'training_time': time.time() - start_time
        }
        
    def _update_episode_q_values_mces(
        self,
        episode_states: List[int],
        episode_actions: List[int],
        returns: jnp.ndarray
    ) -> bool:
        """
        Actualiza los valores Q y la política para un episodio completo en MCES.
        
        Parámetros:
        -----------
        episode_states : List[int]
            Estados visitados en el episodio
        episode_actions : List[int]
            Acciones tomadas en el episodio
        returns : jnp.ndarray
            Retornos calculados para el episodio
            
        Retorna:
        --------
        bool
            Si la política cambió en algún estado
        """
        policy_changed = False
        visited_state_action_pairs = set()
        
        for t in range(len(episode_states)):
            state = episode_states[t]
            action = episode_actions[t]
            
            # Para first-visit MC, solo considerar primera visita a cada par estado-acción
            state_action = (state, action)
            if self.first_visit and state_action in visited_state_action_pairs:
                continue
            
            visited_state_action_pairs.add(state_action)
            
            # Actualizar conteos y sumas para este par estado-acción
            self.returns_sum[state, action] += float(returns[t])
            self.returns_count[state, action] += 1
            
            # Actualizar valor Q usando promedio incremental
            if self.returns_count[state, action] > 0:
                self.q_table = self.q_table.at[state, action].set(
                    self.returns_sum[state, action] / self.returns_count[state, action]
                )
                
                # Actualizar política (determinística para MCES)
                if self._update_policy_deterministic(state):
                    policy_changed = True
                    
        return policy_changed
        
    def _log_progress(self, episode: int, episodes: int, start_time: float) -> None:
        """
        Registra y muestra el progreso del entrenamiento.
        
        Parámetros:
        -----------
        episode : int
            Episodio actual
        episodes : int
            Total de episodios
        start_time : float
            Tiempo de inicio del entrenamiento
        """
        avg_reward = np.mean(self.episode_rewards[-MONTE_CARLO_CONFIG['log_interval']:])
        elapsed_time = time.time() - start_time
        
        print(f"Episodio {episode+1}/{episodes} - Recompensa promedio: {avg_reward:.2f}, "
              f"Tiempo: {elapsed_time:.2f}s")
    
    def evaluate(
        self, 
        env: Any, 
        episodes: int = 10, 
        max_steps: int = 1000, 
        render: bool = False
    ) -> float:
        """
        Evalúa la política actual en el entorno.
        
        Parámetros:
        -----------
        env : Any
            Entorno para evaluar
        episodes : int, opcional
            Número de episodios para la evaluación (default: 10)
        max_steps : int, opcional
            Máximo número de pasos por episodio (default: 1000)
        render : bool, opcional
            Si mostrar o no la visualización del entorno (default: False)
            
        Retorna:
        --------
        float
            Recompensa promedio por episodio
        """
        total_rewards = []
        
        for episode in range(episodes):
            state, _ = env.reset()
            total_reward = 0
            steps = 0
            done = False
            
            while not done and steps < max_steps:
                if render:
                    env.render()
                
                # Seleccionar acción según la política actual, sin exploración
                action = self.get_action(state, explore=False)
                
                # Ejecutar acción
                next_state, reward, terminated, truncated, _ = env.step(action)
                done = terminated or truncated
                
                # Actualizar
                total_reward += reward
                state = next_state
                steps += 1
            
            total_rewards.append(total_reward)
            print(f"Episodio {episode+1}: Recompensa = {total_reward}, Pasos = {steps}")
        
        avg_reward = np.mean(total_rewards)
        print(f"Evaluación: Recompensa promedio = {avg_reward:.2f}")
        
        return avg_reward
    
    def save(self, filepath: str) -> None:
        """
        Guarda el modelo (tablas Q, política, etc.) en un archivo.
        
        Parámetros:
        -----------
        filepath : str
            Ruta donde guardar el modelo
        """
        data = {
            'q_table': np.array(self.q_table),
            'policy': np.array(self.policy),
            'v_table': np.array(self.v_table),
            'n_states': self.n_states,
            'n_actions': self.n_actions,
            'gamma': self.gamma,
            'epsilon': self.epsilon,
            'first_visit': self.first_visit,
            'evaluation_mode': self.evaluation_mode
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(data, f)
        
        print(f"Modelo guardado en {filepath}")
    
    def load(self, filepath: str) -> None:
        """
        Carga el modelo desde un archivo.
        
        Parámetros:
        -----------
        filepath : str
            Ruta desde donde cargar el modelo
        """
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
        
        self.q_table = jnp.array(data['q_table'])
        self.policy = jnp.array(data['policy'])
        self.v_table = jnp.array(data['v_table'])
        self.n_states = data['n_states']
        self.n_actions = data['n_actions']
        self.gamma = data['gamma']
        self.epsilon = data['epsilon']
        self.first_visit = data['first_visit']
        self.evaluation_mode = data['evaluation_mode']
        
        print(f"Modelo cargado desde {filepath}")
    
    def _get_grid_position(self, env: Any, state: int, grid_shape: Tuple[int, int]) -> Tuple[int, int]:
        """
        Obtiene la posición en la cuadrícula para un estado dado.
        
        Parámetros:
        -----------
        env : Any
            Entorno con estructura de cuadrícula
        state : int
            Estado para el cual obtener la posición
        grid_shape : Tuple[int, int]
            Forma de la cuadrícula (filas, columnas)
            
        Retorna:
        --------
        Tuple[int, int]
            Posición del estado en la cuadrícula (i, j)
        """
        if hasattr(env, 'state_mapping'):
            # Convertir índice de estado a posición en cuadrícula
            return env.state_mapping(state)
        else:
            # Asumir orden row-major
            return state // grid_shape[1], state % grid_shape[1]
    
    def _draw_grid_lines(self, ax: plt.Axes, grid_shape: Tuple[int, int]) -> None:
        """
        Dibuja las líneas de la cuadrícula.
        
        Parámetros:
        -----------
        ax : plt.Axes
            Ejes donde dibujar las líneas
        grid_shape : Tuple[int, int]
            Forma de la cuadrícula (filas, columnas)
        """
        # Dibujar líneas de cuadrícula
        for i in range(grid_shape[1] + 1):
            ax.axvline(i, color='black', linestyle='-')
        for j in range(grid_shape[0] + 1):
            ax.axhline(j, color='black', linestyle='-')
    
    def _draw_policy_arrows(self, ax: plt.Axes, env: Any, grid_shape: Tuple[int, int]) -> None:
        """
        Dibuja flechas para mostrar la política en cada estado.
        
        Parámetros:
        -----------
        ax : plt.Axes
            Ejes donde dibujar las flechas
        env : Any
            Entorno con estructura de cuadrícula
        grid_shape : Tuple[int, int]
            Forma de la cuadrícula (filas, columnas)
        """
        # Direcciones de flechas (pueden variar según el entorno)
        directions = {
            0: (0, -0.4),  # Izquierda
            1: (0, 0.4),   # Derecha
            2: (-0.4, 0),  # Abajo
            3: (0.4, 0)    # Arriba
        }
        
        for s in range(self.n_states):
            # Evitar estados terminales
            if hasattr(env, 'is_terminal') and env.is_terminal(s):
                continue
                
            # Obtener posición en la cuadrícula
            i, j = self._get_grid_position(env, s, grid_shape)
            
            # Determinar acción a mostrar basada en el modo
            action = int(jnp.argmax(self.policy[s] if self.evaluation_mode else self.q_table[s]))
            
            # Dibujar flecha para la acción
            if action in directions:
                dx, dy = directions[action]
                ax.arrow(j + 0.5, grid_shape[0] - i - 0.5, dx, dy, 
                         head_width=0.1, head_length=0.1, fc='blue', ec='blue')
    
    def _draw_state_values(self, ax: plt.Axes, env: Any, grid_shape: Tuple[int, int]) -> None:
        """
        Muestra valores de estado en la cuadrícula.
        
        Parámetros:
        -----------
        ax : plt.Axes
            Ejes donde mostrar los valores
        env : Any
            Entorno con estructura de cuadrícula
        grid_shape : Tuple[int, int]
            Forma de la cuadrícula (filas, columnas)
        """
        for s in range(self.n_states):
            # Obtener posición en la cuadrícula
            i, j = self._get_grid_position(env, s, grid_shape)
            
            # Determinar valor a mostrar basado en el modo
            value = float(self.v_table[s] if self.evaluation_mode else jnp.max(self.q_table[s]))
            
            # Mostrar valor
            ax.text(j + 0.5, grid_shape[0] - i - 0.5, f"{value:.2f}", 
                   ha='center', va='center', color='red', fontsize=9)
    
    def visualize_policy(self, env: Any, title: str = "Política") -> None:
        """
        Visualiza la política actual para entornos tipo cuadrícula.
        
        Parámetros:
        -----------
        env : Any
            Entorno con estructura de cuadrícula
        title : str, opcional
            Título para la visualización (default: "Política")
        """
        if not hasattr(env, 'shape'):
            print("El entorno no tiene estructura de cuadrícula para visualización")
            return
        
        grid_shape = env.shape
        _, ax = plt.subplots(figsize=(8, 8))
        
        # Configurar límites de la cuadrícula
        ax.set_xlim([0, grid_shape[1]])
        ax.set_ylim([0, grid_shape[0]])
        
        # Dibujar componentes de la visualización
        self._draw_grid_lines(ax, grid_shape)
        self._draw_policy_arrows(ax, env, grid_shape)
        self._draw_state_values(ax, env, grid_shape)
        
        ax.set_title(title)
        plt.tight_layout()
        plt.show()

    def _create_value_grid(self, env: Any, grid_shape: Tuple[int, int]) -> np.ndarray:
        """
        Crea una matriz con los valores de estado para visualización.
        
        Parámetros:
        -----------
        env : Any
            Entorno con estructura de cuadrícula
        grid_shape : Tuple[int, int]
            Forma de la cuadrícula
            
        Retorna:
        --------
        np.ndarray
            Matriz con valores de estado
        """
        value_grid = np.zeros(grid_shape)
        
        # Llenar matriz con valores
        for s in range(self.n_states):
            if hasattr(env, 'state_mapping'):
                i, j = env.state_mapping(s)
            else:
                i, j = s // grid_shape[1], s % grid_shape[1]
                
            if self.evaluation_mode:
                value_grid[i, j] = float(self.v_table[s])
            else:
                value_grid[i, j] = float(jnp.max(self.q_table[s]))
                
        return value_grid
    
    def _draw_value_labels(self, ax: plt.Axes, value_grid: np.ndarray, grid_shape: Tuple[int, int]) -> None:
        """
        Dibuja las etiquetas de valor en cada celda.
        
        Parámetros:
        -----------
        ax : plt.Axes
            Ejes donde dibujar las etiquetas
        value_grid : np.ndarray
            Matriz con valores de estado
        grid_shape : Tuple[int, int]
            Forma de la cuadrícula
        """
        max_value = np.max(value_grid)
        threshold = max_value / 1.5
        
        for i in range(grid_shape[0]):
            for j in range(grid_shape[1]):
                text_color = 'white' if value_grid[i, j] < threshold else 'black'
                ax.text(j, i, f"{value_grid[i, j]:.2f}", ha='center', va='center',
                        color=text_color, fontsize=9)
    
    def visualize_value_function(self, env: Any, title: str = "Función de Valor") -> None:
        """
        Visualiza la función de valor para entornos tipo cuadrícula.
        
        Parámetros:
        -----------
        env : Any
            Entorno con estructura de cuadrícula
        title : str, opcional
            Título para la visualización (default: "Función de Valor")
        """
        if not hasattr(env, 'shape'):
            print("El entorno no tiene estructura de cuadrícula para visualización")
            return
        
        grid_shape = env.shape
        
        # Crear matriz de valores
        value_grid = self._create_value_grid(env, grid_shape)
        
        # Configurar visualización
        _, ax = plt.subplots(figsize=(10, 8))
        im = ax.imshow(value_grid, cmap='viridis')
        plt.colorbar(im, ax=ax, label='Valor')
        
        # Dibujar etiquetas de valor
        self._draw_value_labels(ax, value_grid, grid_shape)
        
        ax.set_title(title)
        plt.tight_layout()
        plt.show()
    
    def visualize_training(self, history: Optional[Dict[str, List[float]]] = None) -> None:
        """
        Visualiza métricas de entrenamiento.
        
        Parámetros:
        -----------
        history : Optional[Dict[str, List[float]]], opcional
            Diccionario con historial de entrenamiento (default: None)
        """
        if history is None:
            # Si no se proporciona historia, usar datos internos
            history = {
                'episode_rewards': self.episode_rewards,
                'episode_lengths': self.episode_lengths,
                'policy_changes': self.policy_changes,
                'value_changes': self.value_changes if len(self.value_changes) > 0 else None,
                'epsilon_history': self.epsilon_history
            }
        
        # Configuración de la figura
        n_plots = 3
        if 'value_changes' in history and history['value_changes']:
            n_plots += 1
        if 'policy_changes' in history and history['policy_changes']:
            n_plots += 1
            
        _, axs = plt.subplots(n_plots, 1, figsize=(12, 3*n_plots))
        
        plot_idx = 0
        
        # Gráfico de recompensas
        axs[plot_idx].plot(history['episode_rewards'])
        axs[plot_idx].set_title('Recompensas por Episodio')
        axs[plot_idx].set_xlabel('Episodio')
        axs[plot_idx].set_ylabel('Recompensa Total')
        axs[plot_idx].grid(True)
        
        # Suavizar curva para mejor visualización
        window_size = min(len(history['episode_rewards']) // 10 + 1, 100)
        if window_size > 1:
            smoothed = np.convolve(history['episode_rewards'], 
                                  np.ones(window_size)/window_size, mode='valid')
            axs[plot_idx].plot(range(window_size-1, len(history['episode_rewards'])), 
                              smoothed, 'r-', linewidth=2, label=f'Suavizado (ventana={window_size})')
            axs[plot_idx].legend()
        
        plot_idx += 1
        
        # Gráfico de longitud de episodios
        axs[plot_idx].plot(history['episode_lengths'])
        axs[plot_idx].set_title('Longitud de Episodios')
        axs[plot_idx].set_xlabel('Episodio')
        axs[plot_idx].set_ylabel('Pasos')
        axs[plot_idx].grid(True)
        
        plot_idx += 1
        
        # Gráfico de epsilon (si existe)
        if 'epsilon_history' in history and history['epsilon_history']:
            axs[plot_idx].plot(history['epsilon_history'])
            axs[plot_idx].set_title('Epsilon (Exploración)')
            axs[plot_idx].set_xlabel('Episodio')
            axs[plot_idx].set_ylabel('Epsilon')
            axs[plot_idx].grid(True)
            
            plot_idx += 1
        
        # Gráfico de cambios en la política (si existe)
        if 'policy_changes' in history and history['policy_changes']:
            axs[plot_idx].plot(history['policy_changes'])
            axs[plot_idx].set_title('Cambios en la Política')
            axs[plot_idx].set_xlabel('Episodio')
            axs[plot_idx].set_ylabel('Cambio (0=No, 1=Sí)')
            axs[plot_idx].grid(True)
            
            plot_idx += 1
        
        # Gráfico de cambios en valores (si existe)
        if 'value_changes' in history and history['value_changes']:
            axs[plot_idx].plot(history['value_changes'])
            axs[plot_idx].set_title('Cambios en Valores')
            axs[plot_idx].set_xlabel('Actualización')
            axs[plot_idx].set_ylabel('Cambio Promedio')
            axs[plot_idx].set_yscale('log')  # Escala logarítmica para ver mejor la convergencia
            axs[plot_idx].grid(True)
        
        plt.tight_layout()
        plt.show()
    
    def train(
        self, 
        env: Any, 
        method: str = 'on_policy', 
        episodes: Optional[int] = None, 
        max_steps: Optional[int] = None,
        render: bool = False
    ) -> Dict[str, List[float]]:
        """
        Método principal para entrenar el agente con el algoritmo Monte Carlo seleccionado.
        
        Parámetros:
        -----------
        env : Any
            Entorno para entrenar
        method : str, opcional
            Método de entrenamiento ('on_policy', 'off_policy', 'exploring_starts') (default: 'on_policy')
        episodes : Optional[int], opcional
            Número de episodios (si None, usa valor de configuración) (default: None)
        max_steps : Optional[int], opcional
            Pasos máximos por episodio (si None, usa valor de configuración) (default: None)
        render : bool, opcional
            Si mostrar o no la visualización del entorno (default: False)
            
        Retorna:
        --------
        Dict[str, List[float]]
            Historia de entrenamiento
        """
        if episodes is None:
            episodes = MONTE_CARLO_CONFIG['episodes']
        
        if max_steps is None:
            max_steps = MONTE_CARLO_CONFIG['max_steps']
        
        # Resetear contadores para nuevo entrenamiento
        self.reset_counters()
        
        # Seleccionar método de entrenamiento
        if method == 'on_policy':
            return self.monte_carlo_control_on_policy(env, episodes, max_steps, render)
        elif method == 'off_policy':
            return self.monte_carlo_control_off_policy(env, episodes, max_steps, render)
        elif method == 'exploring_starts':
            return self.monte_carlo_exploring_starts(env, episodes, max_steps, render)
        else:
            raise ValueError(f"Método desconocido: {method}. Use 'on_policy', 'off_policy' o 'exploring_starts'")
    
    def visualize_action_values(self, state: int, title: Optional[str] = None) -> None:
        """
        Visualiza los valores Q para todas las acciones en un estado específico.
        
        Parámetros:
        -----------
        state : int
            Estado para visualizar valores de acción
        title : Optional[str], opcional
            Título opcional para el gráfico (default: None)
        """
        if not title:
            title = f"Valores Q para el Estado {state}"
        
        actions = np.arange(self.n_actions)
        values = np.array(self.q_table[state])
        
        plt.figure(figsize=(10, 6))
        plt.bar(actions, values, color='skyblue')
        plt.axhline(y=0, color='k', linestyle='-', alpha=0.3)
        
        # Añadir valores encima de cada barra
        for i, v in enumerate(values):
            plt.text(i, v + 0.05, f"{v:.2f}", ha='center')
        
        # Resaltar la mejor acción
        best_action = int(jnp.argmax(self.q_table[state]))
        plt.bar(best_action, values[best_action], color='green', label='Mejor Acción')
        
        plt.xlabel('Acciones')
        plt.ylabel('Valor Q')
        plt.title(title)
        plt.xticks(actions)
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.show()
    
    def compare_visits(
        self, 
        env: Any, 
        episodes: int = 100, 
        max_steps: int = 1000
    ) -> Dict[str, jnp.ndarray]:
        """
        Compara first-visit y every-visit Monte Carlo para la evaluación de política.
        
        Parámetros:
        -----------
        env : Any
            Entorno para evaluar
        episodes : int, opcional
            Número de episodios para la comparación (default: 100)
        max_steps : int, opcional
            Pasos máximos por episodio (default: 1000)
            
        Retorna:
        --------
        Dict[str, jnp.ndarray]
            Diccionario con resultados de la comparación
        """
        print("Comparando first-visit vs every-visit Monte Carlo...")
        
        # Guardar configuración original
        original_first_visit = self.first_visit
        
        # Crear agentes para comparar
        first_visit_agent = MonteCarlo(
            self.n_states, 
            self.n_actions,
            gamma=self.gamma,
            first_visit=True,
            evaluation_mode=True
        )
        
        every_visit_agent = MonteCarlo(
            self.n_states, 
            self.n_actions,
            gamma=self.gamma,
            first_visit=False,
            evaluation_mode=True
        )
        
        # Establecer la misma política para ambos agentes
        first_visit_agent.policy = jnp.array(self.policy)
        every_visit_agent.policy = jnp.array(self.policy)
        
        # Recopilar episodios
        collected_episodes = []
        for _ in range(episodes):
            state, _ = env.reset()
            states = []
            actions = []
            rewards = []
            done = False
            step = 0
            
            while not done and step < max_steps:
                # Elegir acción según la política actual
                action = self.get_action(state)
                
                # Ejecutar acción
                next_state, reward, terminated, truncated, _ = env.step(action)
                done = terminated or truncated
                
                # Guardar transición
                states.append(state)
                actions.append(action)
                rewards.append(reward)
                
                # Actualizar
                state = next_state
                step += 1
            
            collected_episodes.append((states, actions, rewards))
        
        # Evaluar política con ambos métodos
        first_v = first_visit_agent.monte_carlo_prediction(collected_episodes)
        every_v = every_visit_agent.monte_carlo_prediction(collected_episodes)
        
        # Calcular diferencias
        diff = jnp.abs(first_v - every_v)
        mean_diff = float(jnp.mean(diff))
        max_diff = float(jnp.max(diff))
        
        # Visualizar diferencias
        plt.figure(figsize=(12, 6))
        
        plt.subplot(1, 3, 1)
        plt.plot(np.array(first_v), label='First-visit')
        plt.plot(np.array(every_v), label='Every-visit')
        plt.xlabel('Estado')
        plt.ylabel('Valor')
        plt.title('Comparación de Valores')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.subplot(1, 3, 2)
        plt.plot(np.array(diff))
        plt.xlabel('Estado')
        plt.ylabel('Diferencia Absoluta')
        plt.title(f'Diferencia (Media: {mean_diff:.4f})')
        plt.grid(True, alpha=0.3)
        
        plt.subplot(1, 3, 3)
        plt.hist(np.array(diff), bins=20)
        plt.xlabel('Diferencia')
        plt.ylabel('Frecuencia')
        plt.title('Histograma de Diferencias')
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
        
        # Restaurar configuración original
        self.first_visit = original_first_visit
        
        return {
            'first_visit': first_v,
            'every_visit': every_v,
            'mean_diff': mean_diff,
            'max_diff': max_diff
        }
    
    def _init_weighted_sampling(self) -> float:
        """
        Inicializa las estructuras para weighted importance sampling y devuelve 
        el epsilon de la política de comportamiento.
        
        Retorna:
        --------
        float
            Epsilon para la política de comportamiento
        """
        # Inicializar matrices para weighted importance sampling
        self.q_table = jnp.zeros((self.n_states, self.n_actions))  # Valores Q
        self.c_table = np.zeros((self.n_states, self.n_actions))  # Pesos acumulados
        
        # Política de comportamiento (behavior policy) - más exploratoria
        return max(0.1, self.epsilon)
    
    def _select_behavior_action(self, state: int, behavior_epsilon: float) -> Tuple[int, float]:
        """
        Selecciona una acción según la política de comportamiento.
        
        Parámetros:
        -----------
        state : int
            Estado actual
        behavior_epsilon : float
            Epsilon para la política de comportamiento
            
        Retorna:
        --------
        Tuple[int, float]
            Acción seleccionada y su probabilidad bajo la política de comportamiento
        """
        self.key, subkey = jax.random.split(self.key)
        
        if jax.random.uniform(subkey) < behavior_epsilon:
            self.key, subkey = jax.random.split(self.key)
            action = int(jax.random.randint(subkey, (), 0, self.n_actions))
            behavior_prob = behavior_epsilon / self.n_actions
        else:
            action = int(jnp.argmax(self.q_table[state]))
            behavior_prob = 1 - behavior_epsilon + (behavior_epsilon / self.n_actions)
            
        return action, behavior_prob
    
    def _run_behavior_episode(
        self, 
        env: Any, 
        behavior_epsilon: float, 
        max_steps: int, 
        render: bool
    ) -> Tuple[List[int], List[int], List[float], List[float]]:
        """
        Ejecuta un episodio usando la política de comportamiento.
        
        Parámetros:
        -----------
        env : Any
            Entorno a utilizar
        behavior_epsilon : float
            Epsilon para la política de comportamiento
        max_steps : int
            Número máximo de pasos
        render : bool
            Si renderizar o no el entorno
            
        Retorna:
        --------
        Tuple[List[int], List[int], List[float], List[float]]
            Estados, acciones, recompensas y probabilidades de comportamiento
        """
        state, _ = env.reset()
        episode_states = []
        episode_actions = []
        episode_rewards = []
        episode_behavior_probs = []
        
        for _ in range(max_steps):
            if render:
                env.render()
            
            # Seleccionar acción usando política de comportamiento
            action, behavior_prob = self._select_behavior_action(state, behavior_epsilon)
            
            # Dar un paso en el entorno
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            
            # Guardar transición
            episode_states.append(state)
            episode_actions.append(action)
            episode_rewards.append(reward)
            episode_behavior_probs.append(behavior_prob)
            
            # Actualizar estado
            state = next_state
            
            if done:
                break
                
        return episode_states, episode_actions, episode_rewards, episode_behavior_probs
    
    def _update_weighted_values(
        self, 
        episode_states: List[int], 
        episode_actions: List[int], 
        episode_rewards: List[float], 
        episode_behavior_probs: List[float]
    ) -> None:
        """
        Actualiza los valores Q usando weighted importance sampling.
        
        Parámetros:
        -----------
        episode_states : List[int]
            Estados visitados
        episode_actions : List[int]
            Acciones tomadas
        episode_rewards : List[float]
            Recompensas recibidas
        episode_behavior_probs : List[float]
            Probabilidades bajo la política de comportamiento
        """
        # Calcular retornos
        _ = self.calculate_returns(episode_rewards)
        
        # Procesar el episodio en orden inverso
        G = 0.0
        W = 1.0  # Peso de importancia inicial
        
        for t in range(len(episode_states) - 1, -1, -1):
            state = episode_states[t]
            action = episode_actions[t]
            reward = episode_rewards[t]
            
            # Actualizar retorno acumulado
            G = reward + self.gamma * G
            
            # Incrementar el contador de visitas con el peso de importancia
            self.c_table[state, action] += W
            
            # Actualizar valor Q usando weighted importance sampling
            if self.c_table[state, action] > 0:
                current_q = float(self.q_table[state, action])
                new_q = current_q + (W / self.c_table[state, action]) * (G - current_q)
                self.q_table = self.q_table.at[state, action].set(new_q)
            
            # Actualizar política (greedy respecto a Q)
            new_policy = np.zeros(self.n_actions)
            best_action = int(jnp.argmax(self.q_table[state]))
            new_policy[best_action] = 1.0
            self.policy = self.policy.at[state].set(new_policy)
            
            # Si la acción no habría sido tomada por la política target, detenemos la actualización
            if action != best_action:
                break
            
            # Actualizar ratio de importancia
            target_prob = 1.0  # Política greedy
            W *= target_prob / episode_behavior_probs[t]
    
    def weighted_importance_sampling(
        self, 
        env: Any, 
        episodes: int = MONTE_CARLO_CONFIG['episodes'], 
        max_steps: int = MONTE_CARLO_CONFIG['max_steps'], 
        render: bool = False
    ) -> Dict[str, List[float]]:
        """
        Implementa control Monte Carlo off-policy con weighted importance sampling.
        Este método tiende a ser más estable que el importance sampling ordinario.
        
        Parámetros:
        -----------
        env : Any
            Entorno de OpenAI Gym o similar
        episodes : int, opcional
            Número de episodios a ejecutar (default: MONTE_CARLO_CONFIG['episodes'])
        max_steps : int, opcional
            Número máximo de pasos por episodio (default: MONTE_CARLO_CONFIG['max_steps'])
        render : bool, opcional
            Si renderizar o no el entorno (default: False)
            
        Retorna:
        --------
        Dict[str, List[float]]
            Historia del entrenamiento
        """
        start_time = time.time()
        
        # Inicializar valores para weighted importance sampling
        behavior_epsilon = self._init_weighted_sampling()
        
        for episode in range(episodes):
            # Ejecutar un episodio completo usando política de comportamiento
            episode_data = self._run_behavior_episode(env, behavior_epsilon, max_steps, render)
            episode_states, episode_actions, episode_rewards, episode_behavior_probs = episode_data
            
            # Actualizar valores usando importance sampling
            self._update_weighted_values(episode_states, episode_actions, episode_rewards, episode_behavior_probs)
            
            # Registrar métricas del episodio
            self.episode_rewards.append(sum(episode_rewards))
            self.episode_lengths.append(len(episode_rewards))
            
            # Mostrar progreso periódicamente
            if (episode + 1) % MONTE_CARLO_CONFIG['log_interval'] == 0 or episode == 0:
                avg_reward = np.mean(self.episode_rewards[-MONTE_CARLO_CONFIG['log_interval']:])
                elapsed_time = time.time() - start_time
                
                print(f"Episodio {episode+1}/{episodes} - Recompensa promedio: {avg_reward:.2f}, "
                      f"Tiempo: {elapsed_time:.2f}s")
        
        # Crear historial de entrenamiento
        history = {
            'episode_rewards': self.episode_rewards,
            'episode_lengths': self.episode_lengths,
            'training_time': time.time() - start_time
        }
        
        return history
    
    def _run_episode_incremental(
        self,
        env: Any,
        max_steps: int,
        render: bool = False
    ) -> List[Tuple[int, int, float]]:
        """
        Ejecuta un episodio completo y retorna un buffer de experiencias.
        
        Parámetros:
        -----------
        env : Any
            Entorno donde ejecutar el episodio
        max_steps : int
            Número máximo de pasos a ejecutar
        render : bool, opcional
            Si renderizar o no el entorno (default: False)
            
        Retorna:
        --------
        List[Tuple[int, int, float]]
            Buffer con transiciones (estado, acción, recompensa)
        """
        state, _ = env.reset()
        done = False
        step = 0
        
        # Inicializar buffer para el episodio actual
        episode_buffer = []
        
        # Ejecutar un episodio completo
        while not done and step < max_steps:
            if render:
                env.render()
            
            # Seleccionar acción según política epsilon-greedy
            action = self.get_action(state, explore=True)
            
            # Dar un paso en el entorno
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            
            # Guardar transición en buffer
            episode_buffer.append((state, action, reward))
            
            # Actualizar estado
            state = next_state
            step += 1
            
        return episode_buffer
    
    def _is_first_visit(self, episode_buffer: List[Tuple[int, int, float]], t: int) -> bool:
        """
        Verifica si es la primera visita a un par estado-acción.
        
        Parámetros:
        -----------
        episode_buffer : List[Tuple[int, int, float]]
            Buffer de experiencias del episodio
        t : int
            Índice actual en el buffer
            
        Retorna:
        --------
        bool
            True si es la primera visita, False en caso contrario
        """
        if not self.first_visit:
            return True
            
        state, action, _ = episode_buffer[t]
        
        # Buscar ocurrencias previas del mismo par estado-acción
        for i in range(t):
            if episode_buffer[i][0] == state and episode_buffer[i][1] == action:
                return False
                
        return True
    
    def _update_q_incrementally(
        self, 
        state: int, 
        action: int, 
        G: float
    ) -> None:
        """
        Actualiza el valor Q para un par estado-acción de forma incremental.
        
        Parámetros:
        -----------
        state : int
            Estado a actualizar
        action : int
            Acción a actualizar
        G : float
            Retorno acumulado
        """
        # Actualizar contador
        self.returns_count[state, action] += 1
        
        # Actualización incremental: Q(s,a) = Q(s,a) + (1/N) * (G - Q(s,a))
        alpha = 1.0 / self.returns_count[state, action]
        current_q = float(self.q_table[state, action])
        self.q_table = self.q_table.at[state, action].set(current_q + alpha * (G - current_q))
        
        # Actualizar política
        self.update_policy(state)
    
    def _process_episode_incrementally(self, episode_buffer: List[Tuple[int, int, float]]) -> None:
        """
        Procesa un episodio completo y actualiza valores Q de forma incremental.
        
        Parámetros:
        -----------
        episode_buffer : List[Tuple[int, int, float]]
            Buffer con transiciones (estado, acción, recompensa)
        """
        G = 0
        # Recorrer el episodio en orden inverso
        for t in range(len(episode_buffer) - 1, -1, -1):
            state, action, reward = episode_buffer[t]
            
            # Actualizar retorno acumulado
            G = reward + self.gamma * G
            
            # Solo actualizar si es primera visita (cuando corresponda)
            if self._is_first_visit(episode_buffer, t):
                self._update_q_incrementally(state, action, G)
    
    def incremental_monte_carlo(
        self, 
        env: Any, 
        episodes: int = MONTE_CARLO_CONFIG['episodes'], 
        max_steps: int = MONTE_CARLO_CONFIG['max_steps'],
        render: bool = False
    ) -> Dict[str, List[float]]:
        """
        Implementa una versión incremental de Monte Carlo control que actualiza
        los valores Q después de cada paso en lugar de al final del episodio.
        
        Parámetros:
        -----------
        env : Any
            Entorno de OpenAI Gym o similar
        episodes : int, opcional
            Número de episodios a ejecutar (default: MONTE_CARLO_CONFIG['episodes'])
        max_steps : int, opcional
            Número máximo de pasos por episodio (default: MONTE_CARLO_CONFIG['max_steps'])
        render : bool, opcional
            Si renderizar o no el entorno (default: False)
            
        Retorna:
        --------
        Dict[str, List[float]]
            Historia del entrenamiento
        """
        start_time = time.time()
        
        for episode in range(episodes):
            # Ejecutar un episodio completo
            episode_buffer = self._run_episode_incremental(env, max_steps, render)
            
            # Procesar el episodio y actualizar valores Q
            self._process_episode_incrementally(episode_buffer)
            
            # Registrar métricas del episodio
            self.episode_rewards.append(sum(r for _, _, r in episode_buffer))
            self.episode_lengths.append(len(episode_buffer))
            
            # Decaer epsilon
            self.decay_epsilon(episode)
            
            # Mostrar progreso periódicamente
            if (episode + 1) % MONTE_CARLO_CONFIG['log_interval'] == 0 or episode == 0:
                self._log_progress(episode, episodes, start_time)
        
        # Crear historial de entrenamiento
        return self._create_training_history(start_time)
    
    def _collect_single_episode(
        self,
        env: Any,
        max_steps: int,
        render: bool = False
    ) -> Tuple[List[int], List[int], List[float]]:
        """
        Recopila un episodio completo siguiendo la política actual.
        
        Parámetros:
        -----------
        env : Any
            Entorno en el que recopilar el episodio
        max_steps : int
            Número máximo de pasos a ejecutar
        render : bool, opcional
            Si renderizar o no el entorno (default: False)
            
        Retorna:
        --------
        Tuple[List[int], List[int], List[float]]
            Tupla de (estados, acciones, recompensas) para el episodio
        """
        state, _ = env.reset()
        episode_states = []
        episode_actions = []
        episode_rewards = []
        done = False
        step = 0
        
        while not done and step < max_steps:
            if render:
                env.render()
            
            # Seleccionar acción según política actual
            action = self.get_action(state, explore=True)
            
            # Dar un paso en el entorno
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            
            # Guardar transición
            episode_states.append(state)
            episode_actions.append(action)
            episode_rewards.append(reward)
            
            # Actualizar estado
            state = next_state
            step += 1
            
            if done:
                break
        
        return episode_states, episode_actions, episode_rewards
    
    def _collect_batch_episodes(
        self,
        env: Any,
        batch_size: int,
        max_steps: int,
        render: bool = False
    ) -> Tuple[List[Tuple[List[int], List[int], List[float]]], float, int]:
        """
        Recopila un lote de episodios.
        
        Parámetros:
        -----------
        env : Any
            Entorno en el que recopilar los episodios
        batch_size : int
            Número de episodios a recopilar
        max_steps : int
            Número máximo de pasos por episodio
        render : bool, opcional
            Si renderizar o no el entorno (default: False)
            
        Retorna:
        --------
        Tuple[List[Tuple[List[int], List[int], List[float]]], float, int]
            Tupla de (episodios, recompensa_total, pasos_total)
        """
        batch_episodes = []
        batch_rewards_sum = 0
        batch_steps_sum = 0
        
        for _ in range(batch_size):
            # Recopilar un episodio
            episode_states, episode_actions, episode_rewards = self._collect_single_episode(
                env, max_steps, render
            )
            
            # Guardar episodio completo
            batch_episodes.append((episode_states, episode_actions, episode_rewards))
            batch_rewards_sum += sum(episode_rewards)
            batch_steps_sum += len(episode_rewards)
        
        return batch_episodes, batch_rewards_sum, batch_steps_sum
    
    def _update_q_from_episode(
        self,
        states: List[int],
        actions: List[int],
        returns: jnp.ndarray
    ) -> None:
        """
        Actualiza los valores Q a partir de un episodio.
        
        Parámetros:
        -----------
        states : List[int]
            Lista de estados visitados
        actions : List[int]
            Lista de acciones tomadas
        returns : jnp.ndarray
            Array de retornos calculados
        """
        visited_state_action_pairs = set()
        
        for t in range(len(states)):
            state = states[t]
            action = actions[t]
            
            # Para first-visit MC, solo considerar primera visita a cada par estado-acción
            state_action = (state, action)
            if self.first_visit and state_action in visited_state_action_pairs:
                continue
            
            visited_state_action_pairs.add(state_action)
            
            # Actualizar conteos y sumas para este par estado-acción
            self.returns_sum[state, action] += float(returns[t])
            self.returns_count[state, action] += 1
            
            # Actualizar valor Q usando promedio incremental
            if self.returns_count[state, action] > 0:
                self.q_table = self.q_table.at[state, action].set(
                    self.returns_sum[state, action] / self.returns_count[state, action]
                )
    
    def _update_policy_for_all_states(self) -> bool:
        """
        Actualiza la política para todos los estados basada en los valores Q actuales.
        
        Retorna:
        --------
        bool
            Boolean indicando si la política cambió en algún estado
        """
        policy_changed = False
        
        for s in range(self.n_states):
            if self.update_policy(s):
                policy_changed = True
        
        return policy_changed
    
    def _create_training_history(self, start_time: float) -> Dict[str, List[float]]:
        """
        Crea un diccionario con el historial de entrenamiento.
        
        Parámetros:
        -----------
        start_time : float
            Tiempo de inicio del entrenamiento
            
        Retorna:
        --------
        Dict[str, List[float]]
            Historia del entrenamiento
        """
        return {
            'episode_rewards': self.episode_rewards,
            'episode_lengths': self.episode_lengths,
            'policy_changes': self.policy_changes,
            'epsilon_history': self.epsilon_history,
            'training_time': time.time() - start_time
        }
    
    def batch_monte_carlo(
        self, 
        env: Any, 
        batch_size: int = 10, 
        iterations: int = MONTE_CARLO_CONFIG['episodes'] // 10, 
        max_steps: int = MONTE_CARLO_CONFIG['max_steps'], 
        render: bool = False
    ) -> Dict[str, List[float]]:
        """
        Implementación de Monte Carlo por lotes, donde los valores Q son actualizados
        después de recopilar múltiples episodios.
        
        Parámetros:
        -----------
        env : Any
            Entorno de OpenAI Gym o similar
        batch_size : int, opcional
            Número de episodios por lote (default: 10)
        iterations : int, opcional
            Número de iteraciones (lotes) a ejecutar (default: MONTE_CARLO_CONFIG['episodes'] // 10)
        max_steps : int, opcional
            Número máximo de pasos por episodio (default: MONTE_CARLO_CONFIG['max_steps'])
        render : bool, opcional
            Si renderizar o no el entorno (default: False)
            
        Retorna:
        --------
        Dict[str, List[float]]
            Historia del entrenamiento
        """
        start_time = time.time()
        
        for iteration in range(iterations):
            # Recopilar un lote de episodios
            batch_episodes, batch_rewards_sum, batch_steps_sum = self._collect_batch_episodes(
                env, batch_size, max_steps, render
            )
            
            # Procesar todos los episodios del lote
            for states, actions, rewards in batch_episodes:
                # Calcular retornos
                returns = self.calculate_returns(rewards)
                
                # Actualizar valores Q
                self._update_q_from_episode(states, actions, returns)
            
            # Actualizar política para todos los estados basada en nuevos valores Q
            policy_changed = self._update_policy_for_all_states()
            
            # Registrar métricas del lote
            self.episode_rewards.append(batch_rewards_sum / batch_size)
            self.episode_lengths.append(batch_steps_sum / batch_size)
            self.policy_changes.append(1 if policy_changed else 0)
            
            # Decaer epsilon
            self.decay_epsilon(iteration * batch_size)
            
            # Mostrar progreso
            avg_reward = self.episode_rewards[-1]
            elapsed_time = time.time() - start_time
            
            print(f"Iteración {iteration+1}/{iterations} - Recompensa promedio: {avg_reward:.2f}, "
                f"Epsilon: {self.epsilon:.4f}, Tiempo: {elapsed_time:.2f}s")
        
        # Crear historial de entrenamiento
        return self._create_training_history(start_time)

    def _plot_weight_distribution(self) -> None:
        """
        Visualiza la distribución de pesos de importance sampling.
        """
        plt.subplot(2, 2, 1)
        weights = self.c_table.flatten()
        weights = weights[weights > 0]  # Solo pesos positivos
        plt.hist(weights, bins=50)
        plt.title('Distribución de Pesos de Importance Sampling')
        plt.xlabel('Peso')
        plt.ylabel('Frecuencia')
        plt.grid(True, alpha=0.3)
    
    def _plot_weight_vs_q_values(self) -> None:
        """
        Visualiza la relación entre pesos y valores Q.
        """
        plt.subplot(2, 2, 2)
        x = []
        y = []
        for s in range(self.n_states):
            for a in range(self.n_actions):
                if self.c_table[s, a] > 0:
                    x.append(self.c_table[s, a])
                    y.append(float(self.q_table[s, a]))
        
        plt.scatter(x, y, alpha=0.5)
        plt.title('Relación entre Pesos y Valores Q')
        plt.xlabel('Peso (C)')
        plt.ylabel('Valor Q')
        plt.xscale('log')  # Escala logarítmica para mejor visualización
        plt.grid(True, alpha=0.3)
    
    def _get_target_policy(self) -> np.ndarray:
        """
        Crea una política target greedy basada en los valores Q.
        
        Returns:
        --------
        np.ndarray:
            La política target determinística
        """
        target_policy = np.zeros((self.n_states, self.n_actions))
        for s in range(self.n_states):
            best_action = int(jnp.argmax(self.q_table[s]))
            target_policy = target_policy.at[s, best_action].set(1.0)
        return target_policy
    
    def _collect_episode_weights(
        self,
        env: Any,
        max_steps: int,
        behavior_epsilon: float,
        target_policy: np.ndarray
    ) -> List[float]:
        """
        Recopila los pesos de importance sampling para un episodio.
        
        Parameters:
        -----------
        env : Any
            Entorno para recopilar datos
        max_steps : int
            Pasos máximos para el episodio
        behavior_epsilon : float
            Epsilon para la política de comportamiento
        target_policy : np.ndarray
            Política target precomputada
            
        Returns:
        --------
        List[float]:
            Lista de pesos de importancia para el episodio
        """
        state, _ = env.reset()
        trajectory = []
        weights = []
        W = 1.0
        
        for _ in range(max_steps):
            # Seleccionar acción usando política de comportamiento
            self.key, subkey = jax.random.split(self.key)
            if jax.random.uniform(subkey) < behavior_epsilon:
                self.key, subkey = jax.random.split(self.key)
                action = int(jax.random.randint(subkey, (), 0, self.n_actions))
                behavior_prob = behavior_epsilon / self.n_actions
            else:
                action = int(jnp.argmax(self.q_table[state]))
                behavior_prob = 1 - behavior_epsilon + (behavior_epsilon / self.n_actions)
            
            # Ejecutar acción
            next_state, _, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            
            # Calcular probabilidad bajo política target
            target_prob = target_policy[state, action]
            
            # Actualizar peso de importancia
            if behavior_prob > 0:
                W *= target_prob / behavior_prob
            
            # Guardar datos
            trajectory.append((state, action))
            weights.append(W)
            
            # Actualizar estado
            state = next_state
            
            if done or target_prob == 0:
                break
        
        return weights
    
    def _plot_weight_evolution(self, importance_weights: List[List[float]], episodes: int) -> None:
        """
        Visualiza la evolución de pesos a lo largo de episodios.
        
        Parameters:
        -----------
        importance_weights : List[List[float]]
            Lista de listas de pesos para cada episodio
        episodes : int
            Número de episodios
        """
        plt.subplot(2, 2, 3)
        for i, weights in enumerate(importance_weights):
            plt.plot(weights, label=f'Episodio {i+1}' if i < 10 else None)
        
        plt.title('Evolución de Pesos de Importancia')
        plt.xlabel('Paso')
        plt.ylabel('Peso')
        plt.yscale('log')  # Escala logarítmica para mejor visualización
        plt.grid(True, alpha=0.3)
        if episodes <= 10:
            plt.legend()
    
    def _plot_weight_statistics(self, importance_weights: List[List[float]]) -> None:
        """
        Visualiza estadísticas de pesos entre episodios.
        
        Parameters:
        -----------
        importance_weights : List[List[float]]
            Lista de listas de pesos para cada episodio
        """
        plt.subplot(2, 2, 4)
        max_len = max(len(w) for w in importance_weights)
        padded_weights = []
        
        # Rellenar con NaN para tener longitudes iguales
        for w in importance_weights:
            padded = w + [float('nan')] * (max_len - len(w))
            padded_weights.append(padded)
        
        weights_array = np.array(padded_weights)
        mean_weights = np.nanmean(weights_array, axis=0)
        std_weights = np.nanstd(weights_array, axis=0)
        
        steps = np.arange(max_len)
        plt.plot(steps, mean_weights, 'b-', label='Media')
        plt.fill_between(steps, mean_weights - std_weights, mean_weights + std_weights, 
                        color='b', alpha=0.2, label='Desviación Estándar')
        
        plt.title('Estadísticas de Pesos por Paso')
        plt.xlabel('Paso')
        plt.ylabel('Peso')
        plt.yscale('log')
        plt.grid(True, alpha=0.3)
        plt.legend()
    
    def visualize_importance_weights(
        self, 
        env: Any, 
        episodes: int = 10, 
        max_steps: int = 100
    ) -> None:
        """
        Visualiza los pesos de importance sampling de Monte Carlo off-policy.
        
        Parámetros:
        -----------
        env : Any
            Entorno para recopilar datos
        episodes : int, opcional
            Número de episodios para visualizar (default: 10)
        max_steps : int, opcional
            Pasos máximos por episodio (default: 100)
        """
        # Asegurarse que tenemos pesos de importance sampling acumulados
        if np.sum(self.c_table) == 0:
            print("No hay pesos de importance sampling para visualizar. Ejecute monte_carlo_control_off_policy primero.")
            return
        
        plt.figure(figsize=(15, 10))
        
        # Visualizar distribución y relación de pesos
        self._plot_weight_distribution()
        self._plot_weight_vs_q_values()
        
        # Crear política target
        target_policy = self._get_target_policy()
        
        # Recopilar pesos de importance sampling para múltiples episodios
        behavior_epsilon = 0.1  # Política de comportamiento más exploratoria
        importance_weights = []
        
        for _ in range(episodes):
            weights = self._collect_episode_weights(
                env, max_steps, behavior_epsilon, target_policy
            )
            importance_weights.append(weights)
        
        # Visualizar evolución y estadísticas de pesos
        self._plot_weight_evolution(importance_weights, episodes)
        self._plot_weight_statistics(importance_weights)
        
        plt.tight_layout()
        plt.show()

    def plot_convergence_comparison(
        self, 
        env: Any, 
        methods: List[str] = ['on_policy', 'weighted', 'batch'], 
        episodes: int = 1000
    ) -> Dict[str, Dict[str, List[float]]]:
        """
        Compara la convergencia de diferentes métodos Monte Carlo en un mismo gráfico.
        
        Parámetros:
        -----------
        env : Any
            Entorno para la comparación
        methods : List[str], opcional
            Lista de métodos a comparar (default: ['on_policy', 'weighted', 'batch'])
        episodes : int, opcional
            Número de episodios para cada método (default: 1000)
            
        Retorna:
        --------
        Dict[str, Dict[str, List[float]]]
            Diccionario con historiales de entrenamiento
        """
        method_map = {
            'on_policy': (self.monte_carlo_control_on_policy, 'On-Policy MC', 'blue'),
            'off_policy': (self.monte_carlo_control_off_policy, 'Off-Policy MC', 'red'),
            'exploring_starts': (self.monte_carlo_exploring_starts, 'MCES', 'green'),
            'weighted': (self.weighted_importance_sampling, 'Weighted IS', 'purple'),
            'incremental': (self.incremental_monte_carlo, 'Incremental MC', 'orange'),
            'batch': (self.batch_monte_carlo, 'Batch MC', 'brown')
        }
        
        all_histories = {}
        
        plt.figure(figsize=(12, 6))
        
        for method_name in methods:
            if method_name not in method_map:
                print(f"Método desconocido: {method_name}")
                continue
                
            method_func, label, color = method_map[method_name]
            
            # Reiniciar el agente
            self.reset_counters()
            self.episode_rewards = []
            self.episode_lengths = []
            self.policy_changes = []
            self.value_changes = []
            self.epsilon_history = []
            self.epsilon = self.epsilon_start
            
            # Entrenar con este método
            print(f"\nEntrenando con método: {label}")
            if method_name == 'batch':
                # Batch necesita parámetros especiales
                batch_size = 10
                iterations = episodes // batch_size
                history = method_func(env, batch_size=batch_size, iterations=iterations, max_steps=MONTE_CARLO_CONFIG['max_steps'])
            else:
                history = method_func(env, episodes=episodes, max_steps=MONTE_CARLO_CONFIG['max_steps'])
            
            # Aplicar suavizado para mejor visualización
            window_size = min(len(history['episode_rewards']) // 10 + 1, 100)
            if window_size > 1 and len(history['episode_rewards']) > window_size:
                smoothed_rewards = np.convolve(history['episode_rewards'], 
                                            np.ones(window_size)/window_size, 
                                            mode='valid')
                x = range(window_size-1, len(history['episode_rewards']))
            else:
                smoothed_rewards = history['episode_rewards']
                x = range(len(smoothed_rewards))
            
            # Graficar resultados
            plt.plot(x, smoothed_rewards, color=color, label=f"{label}")
            
            # Guardar historia
            all_histories[method_name] = history
        
        plt.title('Comparación de Convergencia de Métodos Monte Carlo')
        plt.xlabel('Episodio')
        plt.ylabel('Recompensa Media (suavizada)')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()
        
        return all_histories

class MonteCarloWrapper:
    """
    Wrapper para hacer que MonteCarlo sea compatible con la interfaz de modelos de aprendizaje profundo.
    """
    def __init__(
        self,
        mc_agent: MonteCarlo,
        input_shape: Tuple[int, ...],
        other_features_shape: Tuple[int, ...],
    ) -> None:
        """
        Inicializa el wrapper para MonteCarlo.
        
        Parámetros:
        -----------
        mc_agent : MonteCarlo
            Instancia de agente MonteCarlo
        input_shape : Tuple[int, ...]
            Forma de entrada principal (CGM)
        other_features_shape : Tuple[int, ...]
            Forma de características adicionales
        """
        self.mc_agent = mc_agent
        self.input_shape = input_shape
        self.other_features_shape = other_features_shape
    
    def __call__(self, inputs: list) -> np.ndarray:
        """
        Implementa interfaz de llamada para compatibilidad con pipeline de predicción.
        
        Parámetros:
        -----------
        inputs : list
            Lista de tensores de entrada [cgm_data, other_features]
            
        Retorna:
        --------
        np.ndarray
            Predicciones basadas en política actual
        """
        # Extraer características
        cgm_data, other_features = inputs
        _ = cgm_data.shape[0]
        
        # Transformar datos a estados (simplificado)
        states = np.argmax(other_features, axis=1) % self.mc_agent.n_states
        
        # Obtener acciones según la política actual
        actions = np.array([self.mc_agent.get_action(int(state), explore=False) for state in states])
        
        # Devolver acciones como predicciones
        # Esto es una simplificación - en un caso real habría que mapear acciones a valores de dosis
        return actions.reshape(-1, 1).astype(np.float32)
    
    def predict(self, inputs: list) -> np.ndarray:
        """
        Método de predicción para compatibilidad con API de Keras.
        
        Parámetros:
        -----------
        inputs : list
            Lista de tensores de entrada [cgm_data, other_features]
            
        Retorna:
        --------
        np.ndarray
            Predicciones basadas en política actual
        """
        return self(inputs)
    
    def fit(
        self, 
        x: list, 
        y: np.ndarray, 
        validation_data: Optional[Tuple] = None, 
        epochs: int = 1,
        batch_size: int = 32,
        callbacks: list = None,
        verbose: int = 0
    ) -> Dict:
        """
        Simula interfaz de entrenamiento compatible con Keras.
        
        Parámetros:
        -----------
        x : list
            Lista de tensores de entrada [cgm_data, other_features]
        y : np.ndarray
            Valores objetivo
        validation_data : Optional[Tuple], opcional
            Datos de validación (default: None)
        epochs : int, opcional
            Número de épocas (default: 1)
        batch_size : int, opcional
            Tamaño de lote (default: 32)
        callbacks : list, opcional
            Lista de callbacks (default: None)
        verbose : int, opcional
            Nivel de verbosidad (default: 0)
            
        Retorna:
        --------
        Dict
            Historia de entrenamiento simulada
        """
        # Simular history para compatibilidad
        history = {
            'loss': [0.5],
            'val_loss': [0.5] if validation_data is not None else None
        }
        
        # En un caso real, aquí se implementaría la lógica para adaptar
        # los datos al formato que necesita MonteCarlo y llamar a sus métodos de entrenamiento
        
        return {'history': history}
    
    def save(self, filepath: str) -> None:
        """
        Guarda el agente Monte Carlo.
        
        Parámetros:
        -----------
        filepath : str
            Ruta donde guardar el modelo
        """
        self.mc_agent.save(filepath)
    
    def get_config(self) -> Dict:
        """
        Devuelve la configuración para serialización.
        
        Retorna:
        --------
        Dict
            Configuración del modelo
        """
        return {
            "input_shape": self.input_shape,
            "other_features_shape": self.other_features_shape
        }


def create_monte_carlo_model(cgm_shape: Tuple[int, ...], other_features_shape: Tuple[int, ...]) -> MonteCarloWrapper:
    """
    Crea un modelo de MonteCarlo compatible con la interfaz de modelos de aprendizaje profundo.
    
    Parámetros:
    -----------
    cgm_shape : Tuple[int, ...]
        Forma de los datos CGM (batch_size, time_steps, features)
    other_features_shape : Tuple[int, ...]
        Forma de otras características (batch_size, n_features)
        
    Retorna:
    --------
    MonteCarloWrapper
        Wrapper de MonteCarlo que implementa la interfaz compatible
    """
    # Configurar el tamaño del espacio de estados y acciones
    # Esto es una simplificación - en un caso real habría que definirlo según los datos
    n_states = 100  # Por ejemplo: discretización del espacio de estados
    n_actions = 20   # Por ejemplo: niveles discretos de dosis de insulina
    
    # Crear agente MonteCarlo
    mc_agent = MonteCarlo(
        n_states=n_states,
        n_actions=n_actions,
        gamma=MONTE_CARLO_CONFIG['gamma'],
        epsilon_start=MONTE_CARLO_CONFIG['epsilon_start'],
        epsilon_end=MONTE_CARLO_CONFIG['epsilon_end'],
        epsilon_decay=MONTE_CARLO_CONFIG['epsilon_decay'],
        first_visit=MONTE_CARLO_CONFIG['first_visit']
    )
    
    # Crear y devolver wrapper
    return MonteCarloWrapper(mc_agent, cgm_shape, other_features_shape)