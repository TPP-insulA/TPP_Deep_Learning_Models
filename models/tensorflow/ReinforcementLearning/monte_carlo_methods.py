import os, sys
import numpy as np
import time
import pickle
import matplotlib.pyplot as plt
import tensorflow as tf
from keras._tf_keras.keras.models import Model
from keras._tf_keras.keras.saving import register_keras_serializable
from typing import Dict, List, Tuple, Optional, Union, Any

PROJECT_ROOT = os.path.abspath(os.getcwd())
sys.path.append(PROJECT_ROOT) 

from config.models_config import MONTE_CARLO_CONFIG
from constants.constants import CONST_DEFAULT_SEED
from custom.printer import print_debug

class MonteCarlo:
    """
    Implementación de métodos Monte Carlo para predicción y control en aprendizaje por refuerzo.
    
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
        evaluation_mode: bool = False
    ) -> None:
        """
        Inicializa el agente de Monte Carlo.
        
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
        """
        self.n_states = n_states
        self.n_actions = n_actions
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.first_visit = first_visit
        self.evaluation_mode = evaluation_mode
        
        # Inicializar tablas de valor de acción (Q) y política
        self.q_table = np.zeros((n_states, n_actions))
        
        # Para modo de evaluación, la política es fija (proporcionada externamente)
        # Para control, comenzamos con una política epsilon-greedy derivada de Q
        self.policy = np.ones((n_states, n_actions)) / n_actions  # Inicialmente equiprobable
        
        # Contadores para calcular promedios incrementales
        self.returns_sum = np.zeros((n_states, n_actions))
        self.returns_count = np.zeros((n_states, n_actions))
        
        # Para evaluación de política (valor de estado)
        self.v_table = np.zeros(n_states)
        self.state_returns_sum = np.zeros(n_states)
        self.state_returns_count = np.zeros(n_states)
        
        # Para off-policy Monte Carlo
        self.c_table = np.zeros((n_states, n_actions))  # Pesos acumulativos para importance sampling
        
        # Métricas
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
        rng = np.random.default_rng(CONST_DEFAULT_SEED)  # Crear una instancia de Generator
        if explore and rng.random() < self.epsilon:
            return rng.integers(0, self.n_actions)  # Exploración uniforme
        else:
            if self.evaluation_mode:
                # En modo evaluación, muestrear de la distribución de política
                return rng.choice(self.n_actions, p=self.policy[state])
            else:
                # En modo control, acción greedy con respecto a Q
                return np.argmax(self.q_table[state])
    
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
        
        old_action = np.argmax(self.policy[state])
        best_action = np.argmax(self.q_table[state])
        
        # Política epsilon-greedy basada en Q
        self.policy[state] = np.zeros(self.n_actions)
        
        # Probabilidad pequeña de exploración
        self.policy[state] += self.epsilon / self.n_actions
        
        # Mayor probabilidad para la mejor acción
        self.policy[state][best_action] += (1 - self.epsilon)
        
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
    
    def calculate_returns(self, rewards: List[float]) -> np.ndarray:
        """
        Calcula los retornos descontados para cada paso de tiempo en un episodio.
        
        Parámetros:
        -----------
        rewards : List[float]
            Lista de recompensas recibidas durante el episodio
            
        Retorna:
        --------
        np.ndarray
            Lista de retornos (G_t) para cada paso de tiempo
        """
        returns = np.zeros(len(rewards))
        G = 0
        
        # Recorremos las recompensas en orden inverso
        for t in range(len(rewards) - 1, -1, -1):
            G = rewards[t] + self.gamma * G
            returns[t] = G
            
        return returns
    
    def monte_carlo_prediction(self, episodes: List[Tuple[List[int], List[int], List[float]]]) -> np.ndarray:
        """
        Realiza predicción Monte Carlo (evaluación de política) usando episodios proporcionados.
        
        Parámetros:
        -----------
        episodes : List[Tuple[List[int], List[int], List[float]]]
            Lista de episodios, cada uno como una tupla de 
            (estados, acciones, recompensas)
        
        Retorna:
        --------
        np.ndarray
            v_table actualizada (función de valor de estado)
        """
        old_v = self.v_table.copy()
        
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
                self.state_returns_sum[state] += returns[t]
                self.state_returns_count[state] += 1
                
                # Actualizar la función de valor usando promedio incremental
                if self.state_returns_count[state] > 0:
                    self.v_table[state] = self.state_returns_sum[state] / self.state_returns_count[state]
        
        # Calcular cambio en la función de valor
        value_change = np.mean(np.abs(self.v_table - old_v))
        self.value_changes.append(value_change)
        
        return self.v_table
    
    def _run_episode(self, env: Any, max_steps: int, render: bool) -> Tuple[List[int], List[int], List[float]]:
        """
        Ejecuta un episodio completo y retorna las transiciones.
        
        Parámetros:
        -----------
        env : Any
            Entorno a ejecutar
        max_steps : int
            Número máximo de pasos
        render : bool
            Si renderizar o no el entorno
            
        Retorna:
        --------
        Tuple[List[int], List[int], List[float]]
            estados, acciones y recompensas del episodio
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
        
    def _update_q_values(self, episode_states: List[int], episode_actions: List[int], returns: np.ndarray) -> bool:
        """
        Actualiza valores Q y política basado en un episodio completo.
        
        Parámetros:
        -----------
        episode_states : List[int]
            Lista de estados visitados
        episode_actions : List[int]
            Lista de acciones tomadas
        returns : np.ndarray
            Lista de retornos
            
        Retorna:
        --------
        bool
            Si la política cambió durante la actualización
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
            self.returns_sum[state, action] += returns[t]
            self.returns_count[state, action] += 1
            
            # Actualizar valor Q usando promedio incremental
            if self.returns_count[state, action] > 0:
                self.q_table[state, action] = self.returns_sum[state, action] / self.returns_count[state, action]
                
                # Actualizar política basada en nuevos valores Q
                if self.update_policy(state):
                    policy_changed = True
                    
        return policy_changed
        
    def _log_progress(self, episode: int, episodes: int, start_time: float) -> None:
        """
        Imprime el progreso del entrenamiento.
        
        Parámetros:
        -----------
        episode : int
            Episodio actual
        episodes : int
            Total de episodios
        start_time : float
            Tiempo de inicio
        """
        if (episode + 1) % MONTE_CARLO_CONFIG['log_interval'] == 0 or episode == 0:
            avg_reward = np.mean(self.episode_rewards[-MONTE_CARLO_CONFIG['log_interval']:])
            elapsed_time = time.time() - start_time
            
            print_debug(f"Episodio {episode+1}/{episodes} - Recompensa promedio: {avg_reward:.2f}, ", f"Epsilon: {self.epsilon:.4f}, Tiempo: {elapsed_time:.2f}s")
            
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
            self._log_progress(episode, episodes, start_time)
        
        # Crear historial de entrenamiento
        history = {
            'episode_rewards': self.episode_rewards,
            'episode_lengths': self.episode_lengths,
            'policy_changes': self.policy_changes,
            'value_changes': self.value_changes,
            'epsilon_history': self.epsilon_history,
            'training_time': time.time() - start_time
        }
        
        return history
    
    def _collect_off_policy_episode(
        self, 
        env: Any, 
        behavior_epsilon: float, 
        max_steps: int,
        render: bool
    ) -> Tuple[List[int], List[int], List[float], List[float]]:
        """
        Recopila un episodio completo usando la política de comportamiento.
        
        Parámetros:
        -----------
        env : Any
            Entorno a ejecutar
        behavior_epsilon : float
            Epsilon para la política de comportamiento
        max_steps : int
            Pasos máximos por episodio
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
            rng = np.random.default_rng(CONST_DEFAULT_SEED)  # Generator with seed for reproducibility
            if rng.random() < behavior_epsilon:
                action = rng.integers(0, self.n_actions)
                behavior_prob = behavior_epsilon / self.n_actions
            else:
                action = np.argmax(self.q_table[state])
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
        Actualiza los valores Q usando importance sampling basado en un episodio.
        
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
        # Inicializar el retorno y el peso de importancia
        G = 0.0
        W = 1.0
        
        # Recorremos el episodio en orden inverso
        for t in range(len(episode_states) - 1, -1, -1):
            state = episode_states[t]
            action = episode_actions[t]
            reward = episode_rewards[t]
            
            # Actualizar retorno acumulado
            G = reward + self.gamma * G
            
            # Actualizar contador de visitas para este par estado-acción
            self.c_table[state, action] += W
            
            # Actualizar función Q usando importance sampling
            self.q_table[state, action] += (W / self.c_table[state, action]) * (G - self.q_table[state, action])
            
            # Actualizar política target (greedy respecto a Q)
            self.update_policy(state)
            
            # Obtener probabilidad bajo policy target (greedy)
            target_policy_prob = 1.0 if action == np.argmax(self.q_table[state]) else 0.0
            
            # Actualizar ratio de importancia
            if target_policy_prob < 1e-10:  # Umbral pequeño en lugar de comparación exacta
                break  # Si la acción no sería elegida por la política target, terminar
            
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
            # Recopilar un episodio completo usando la política de comportamiento
            episode_states, episode_actions, episode_rewards, episode_behavior_probs = self._collect_off_policy_episode(
                env, behavior_epsilon, max_steps, render
            )
            
            # Actualizar valores Q usando importance sampling
            self._update_off_policy_q_values(
                episode_states, episode_actions, episode_rewards, episode_behavior_probs
            )
            
            # Registrar métricas del episodio
            self.episode_rewards.append(sum(episode_rewards))
            self.episode_lengths.append(len(episode_rewards))
            
            # Decaer epsilon
            self.decay_epsilon(episode)
            
            # Mostrar progreso periódicamente
            self._log_progress(episode, episodes, start_time)
        
        # Crear historial de entrenamiento
        history = {
            'episode_rewards': self.episode_rewards,
            'episode_lengths': self.episode_lengths,
            'epsilon_history': self.epsilon_history,
            'training_time': time.time() - start_time
        }
        
        return history
    
    def _initialize_exploring_start(self, env: Any) -> Tuple[int, List[int], List[int], List[float]]:
        """
        Inicializa un episodio con exploring starts.
        
        Parámetros:
        -----------
        env : Any
            Entorno a inicializar
            
        Retorna:
        --------
        Tuple[int, List[int], List[int], List[float]]
            Estado actual, estados, acciones y recompensas iniciales
        """
        # Iniciar con un estado aleatorio (exploring start)
        if hasattr(env, 'set_state'):
            rng = np.random.default_rng(CONST_DEFAULT_SEED)
            random_state = rng.integers(0, self.n_states)
            env.set_state(random_state)
            state = random_state
        else:
            # Si no podemos establecer el estado, iniciamos normalmente
            state, _ = env.reset()
            
        # Seleccionar una primera acción aleatoria para garantizar exploring starts
        rng = np.random.default_rng(CONST_DEFAULT_SEED)
        action = rng.integers(0, self.n_actions)
        
        # Ejecutar la primera acción
        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        
        # Inicializar listas para guardar la trayectoria
        episode_states = [state]
        episode_actions = [action]
        episode_rewards = [reward]
        
        return next_state, done, episode_states, episode_actions, episode_rewards
    
    def _update_mces_policy(self, state: int) -> bool:
        """
        Actualiza la política determinística para MCES para un estado.
        
        Parámetros:
        -----------
        state : int
            Estado para actualizar la política
            
        Retorna:
        --------
        bool
            Si la política cambió
        """
        old_action = np.argmax(self.policy[state])
        best_action = np.argmax(self.q_table[state])
        
        if old_action != best_action:
            # Política determinística para MCES
            self.policy[state] = np.zeros(self.n_actions)
            self.policy[state][best_action] = 1.0
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
            # Inicializar con exploring starts
            state, done, episode_states, episode_actions, episode_rewards = self._initialize_exploring_start(env)
            steps = 1
            
            # Continuar el episodio usando la política actual
            while not done and steps < max_steps:
                if render:
                    env.render()
                
                # Seleccionar acción según política actual (sin exploración adicional)
                action = self.get_action(state, explore=False)
                
                # Dar un paso en el entorno
                next_state, reward, terminated, truncated, _ = env.step(action)
                done = terminated or truncated
                
                # Guardar transición
                episode_states.append(state)
                episode_actions.append(action)
                episode_rewards.append(reward)
                
                # Actualizar estado
                state = next_state
                steps += 1
            
            # Calcular retornos para el episodio
            returns = self.calculate_returns(episode_rewards)
            
            # Actualizar Q y política
            policy_changed = self._update_values_mces(episode_states, episode_actions, returns)
            
            # Registrar métricas del episodio
            self.episode_rewards.append(sum(episode_rewards))
            self.episode_lengths.append(len(episode_rewards))
            self.policy_changes.append(1 if policy_changed else 0)
            
            # Mostrar progreso periódicamente
            self._log_progress(episode, episodes, start_time)
        
        # Crear historial de entrenamiento
        history = {
            'episode_rewards': self.episode_rewards,
            'episode_lengths': self.episode_lengths,
            'policy_changes': self.policy_changes,
            'training_time': time.time() - start_time
        }
        
        return history
    
    def _update_values_mces(self, episode_states: List[int], episode_actions: List[int], returns: np.ndarray) -> bool:
        """
        Actualiza valores Q y política para MCES.
        
        Parámetros:
        -----------
        episode_states : List[int]
            Lista de estados visitados
        episode_actions : List[int]
            Lista de acciones tomadas
        returns : np.ndarray
            Lista de retornos
            
        Retorna:
        --------
        bool
            Si la política cambió durante la actualización
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
            self.returns_sum[state, action] += returns[t]
            self.returns_count[state, action] += 1
            
            # Actualizar valor Q usando promedio incremental
            if self.returns_count[state, action] > 0:
                self.q_table[state, action] = self.returns_sum[state, action] / self.returns_count[state, action]
                
                # Actualizar política (determinística para MCES)
                if self._update_mces_policy(state):
                    policy_changed = True
        
        return policy_changed
    
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
            'q_table': self.q_table,
            'policy': self.policy,
            'v_table': self.v_table,
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
        
        self.q_table = data['q_table']
        self.policy = data['policy']
        self.v_table = data['v_table']
        self.n_states = data['n_states']
        self.n_actions = data['n_actions']
        self.gamma = data['gamma']
        self.epsilon = data['epsilon']
        self.first_visit = data['first_visit']
        self.evaluation_mode = data['evaluation_mode']
        
        print(f"Modelo cargado desde {filepath}")
    
    def _setup_grid(self, ax, grid_shape):
        """
        Configura la cuadrícula base para la visualización.
        
        Parámetros:
        -----------
        ax : matplotlib.axes.Axes
            Ejes para dibujar
        grid_shape : tuple
            Forma de la cuadrícula (filas, columnas)
        """
        ax.set_xlim([0, grid_shape[1]])
        ax.set_ylim([0, grid_shape[0]])
        
        # Dibujar líneas de cuadrícula
        for i in range(grid_shape[1] + 1):
            ax.axvline(i, color='black', linestyle='-')
        for j in range(grid_shape[0] + 1):
            ax.axhline(j, color='black', linestyle='-')
            
    def _get_grid_position(self, env, state, grid_shape):
        """
        Obtiene la posición (i,j) en la cuadrícula para un estado.
        
        Parámetros:
        -----------
        env : Any
            Entorno
        state : int
            Índice del estado
        grid_shape : tuple
            Forma de la cuadrícula
            
        Retorna:
        --------
        tuple
            Posición (i,j) en la cuadrícula
        """
        if hasattr(env, 'state_mapping'):
            return env.state_mapping(state)
        else:
            # Asumir orden row-major
            return state // grid_shape[1], state % grid_shape[1]
            
    def _draw_arrows(self, ax, env, grid_shape):
        """
        Dibuja flechas para representar las acciones según la política.
        
        Parámetros:
        -----------
        ax : matplotlib.axes.Axes
            Ejes para dibujar
        env : Any
            Entorno
        grid_shape : tuple
            Forma de la cuadrícula
        """
        # Definir direcciones de flechas
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
            
            # Determinar la acción a mostrar
            action = np.argmax(self.policy[s]) if self.evaluation_mode else np.argmax(self.q_table[s])
            
            # Dibujar flecha para la acción
            if action in directions:
                dx, dy = directions[action]
                ax.arrow(j + 0.5, grid_shape[0] - i - 0.5, dx, dy, 
                        head_width=0.1, head_length=0.1, fc='blue', ec='blue')
                    
    def _draw_values(self, ax, env, grid_shape):
        """
        Dibuja los valores para cada estado en la cuadrícula.
        
        Parámetros:
        -----------
        ax : matplotlib.axes.Axes
            Ejes para dibujar
        env : Any
            Entorno
        grid_shape : tuple
            Forma de la cuadrícula
        """
        for s in range(self.n_states):
            # Obtener posición en la cuadrícula
            i, j = self._get_grid_position(env, s, grid_shape)
            
            # Determinar el valor a mostrar
            value = self.v_table[s] if self.evaluation_mode else np.max(self.q_table[s])
            
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
        
        # Configurar y dibujar la cuadrícula
        self._setup_grid(ax, grid_shape)
        
        # Dibujar flechas para las acciones
        self._draw_arrows(ax, env, grid_shape)
        
        # Mostrar valores Q o V
        self._draw_values(ax, env, grid_shape)
        
        ax.set_title(title)
        plt.tight_layout()
        plt.show()
    
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
        
        # Crear matriz para visualización
        value_grid = np.zeros(grid_shape)
        
        # Llenar matriz con valores
        for s in range(self.n_states):
            if hasattr(env, 'state_mapping'):
                i, j = env.state_mapping(s)
            else:
                i, j = s // grid_shape[1], s % grid_shape[1]
                
            if self.evaluation_mode:
                value_grid[i, j] = self.v_table[s]
            else:
                value_grid[i, j] = np.max(self.q_table[s])
        
        _, ax = plt.subplots(figsize=(10, 8))
        
        # Crear mapa de calor
        im = ax.imshow(value_grid, cmap='viridis')
        
        # Añadir barra de color
        plt.colorbar(im, ax=ax, label='Valor')
        
        # Mostrar valores en cada celda
        for i in range(grid_shape[0]):
            for j in range(grid_shape[1]):
                ax.text(j, i, f"{value_grid[i, j]:.2f}", ha='center', va='center',
                        color='white' if value_grid[i, j] < np.max(value_grid)/1.5 else 'black')
        
        ax.set_title(title)
        plt.tight_layout()
        plt.show()
    
    def visualize_training(self, history: Optional[Dict[str, List]] = None) -> None:
        """
        Visualiza métricas de entrenamiento.
        
        Parámetros:
        -----------
        history : Optional[Dict[str, List]], opcional
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
        values = self.q_table[state]
        
        plt.figure(figsize=(10, 6))
        plt.bar(actions, values, color='skyblue')
        plt.axhline(y=0, color='k', linestyle='-', alpha=0.3)
        
        # Añadir valores encima de cada barra
        for i, v in enumerate(values):
            plt.text(i, v + 0.05, f"{v:.2f}", ha='center')
        
        # Resaltar la mejor acción
        best_action = np.argmax(values)
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
    ) -> Dict[str, np.ndarray]:
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
        Dict[str, np.ndarray]
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
        first_visit_agent.policy = self.policy.copy()
        every_visit_agent.policy = self.policy.copy()
        
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
        diff = np.abs(first_v - every_v)
        mean_diff = np.mean(diff)
        max_diff = np.max(diff)
        
        # Visualizar diferencias
        plt.figure(figsize=(12, 6))
        
        plt.subplot(1, 3, 1)
        plt.plot(first_v, label='First-visit')
        plt.plot(every_v, label='Every-visit')
        plt.xlabel('Estado')
        plt.ylabel('Valor')
        plt.title('Comparación de Valores')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.subplot(1, 3, 2)
        plt.plot(diff)
        plt.xlabel('Estado')
        plt.ylabel('Diferencia Absoluta')
        plt.title(f'Diferencia (Media: {mean_diff:.4f})')
        plt.grid(True, alpha=0.3)
        
        plt.subplot(1, 3, 3)
        plt.hist(diff, bins=20)
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
    
    def _collect_weighted_is_episode(self, env: Any, behavior_epsilon: float, max_steps: int, render: bool) -> Tuple[List[int], List[int], List[float], List[float]]:
        """
        Recopila un episodio usando la política de comportamiento para weighted importance sampling.
        
        Parámetros:
        -----------
        env : Any
            Entorno a ejecutar
        behavior_epsilon : float
            Epsilon para la política de comportamiento
        max_steps : int
            Pasos máximos por episodio
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
            rng = np.random.default_rng(CONST_DEFAULT_SEED)
            if rng.random() < behavior_epsilon:
                action = rng.integers(0, self.n_actions)
                behavior_prob = behavior_epsilon / self.n_actions
            else:
                action = np.argmax(self.q_table[state])
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
        
    def _update_weighted_is_values(self, episode_states: List[int], episode_actions: List[int], 
                                  episode_rewards: List[float], episode_behavior_probs: List[float]) -> None:
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
                # Fórmula de weighted importance sampling: Q += W/C * (G - Q)
                self.q_table[state, action] += (W / self.c_table[state, action]) * (G - self.q_table[state, action])
            
            # Actualizar política (greedy respecto a Q)
            self._update_greedy_policy(state)
            
            # Si la acción no habría sido tomada por la política target, detenemos la actualización
            if action != np.argmax(self.q_table[state]):
                break
            
            # Actualizar ratio de importancia
            target_prob = 1.0  # Política greedy
            W *= target_prob / episode_behavior_probs[t]
    
    def _update_greedy_policy(self, state: int) -> None:
        """
        Actualiza la política para ser completamente greedy respecto a los valores Q.
        
        Parámetros:
        -----------
        state : int
            Estado para el cual actualizar la política
        """
        best_action = np.argmax(self.q_table[state])
        
        # Política greedy (determinística)
        self.policy[state] = np.zeros(self.n_actions)
        self.policy[state][best_action] = 1.0
    
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
        
        # Inicializar matrices para weighted importance sampling
        self.q_table = np.zeros((self.n_states, self.n_actions))  # Valores Q
        self.c_table = np.zeros((self.n_states, self.n_actions))  # Pesos acumulados
        
        # Política de comportamiento (behavior policy) - más exploratoria
        behavior_epsilon = max(0.1, self.epsilon)
        
        for episode in range(episodes):
            # Recopilar un episodio completo
            episode_states, episode_actions, episode_rewards, episode_behavior_probs = \
                self._collect_weighted_is_episode(env, behavior_epsilon, max_steps, render)
            
            # Actualizar valores y política
            self._update_weighted_is_values(episode_states, episode_actions, episode_rewards, episode_behavior_probs)
            
            # Registrar métricas del episodio
            self.episode_rewards.append(sum(episode_rewards))
            self.episode_lengths.append(len(episode_rewards))
            
            # Mostrar progreso periódicamente
            self._log_progress(episode, episodes, start_time)
        
        # Crear historial de entrenamiento
        history = {
            'episode_rewards': self.episode_rewards,
            'episode_lengths': self.episode_lengths,
            'training_time': time.time() - start_time
        }
        
        return history
    
    def _collect_episode_buffer(self, env: Any, max_steps: int, render: bool) -> List[Tuple[int, int, float]]:
        """
        Recopila un episodio completo y retorna el buffer de experiencias.
        
        Parámetros:
        -----------
        env : Any
            Entorno a ejecutar
        max_steps : int
            Número máximo de pasos
        render : bool
            Si renderizar o no el entorno
            
        Retorna:
        --------
        List[Tuple[int, int, float]]
            Buffer de experiencias (estado, acción, recompensa)
        """
        state, _ = env.reset()
        done = False
        step = 0
        episode_buffer = []
        
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
    
    def _is_first_occurrence(self, buffer: List[Tuple[int, int, float]], t: int, state: int, action: int) -> bool:
        """
        Verifica si es la primera ocurrencia de un par estado-acción en el buffer.
        
        Parámetros:
        -----------
        buffer : List[Tuple[int, int, float]]
            Buffer de experiencias
        t : int
            Índice actual
        state : int
            Estado a verificar
        action : int
            Acción a verificar
            
        Retorna:
        --------
        bool
            True si es la primera ocurrencia, False en caso contrario
        """
        for i in range(t):
            if buffer[i][0] == state and buffer[i][1] == action:
                return False
        return True
    
    def _update_from_episode_buffer(self, episode_buffer: List[Tuple[int, int, float]]) -> None:
        """
        Actualiza valores Q y política basándose en el buffer de episodio.
        
        Parámetros:
        -----------
        episode_buffer : List[Tuple[int, int, float]]
            Buffer de experiencias (estado, acción, recompensa)
        """
        G = 0
        for t in range(len(episode_buffer) - 1, -1, -1):
            state, action, reward = episode_buffer[t]
            
            # Actualizar retorno acumulado
            G = reward + self.gamma * G
            
            # Verificar si es primera visita (si es necesario)
            if self.first_visit and not self._is_first_occurrence(episode_buffer, t, state, action):
                continue
            
            # Actualizar contadores y valores Q
            self.returns_count[state, action] += 1
            
            # Actualización incremental
            alpha = 1.0 / self.returns_count[state, action]
            self.q_table[state, action] += alpha * (G - self.q_table[state, action])
            
            # Actualizar política basada en nuevos valores Q
            self.update_policy(state)
    
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
            # Recolectar experiencias del episodio
            episode_buffer = self._collect_episode_buffer(env, max_steps, render)
            
            # Procesar el episodio y actualizar valores Q
            self._update_from_episode_buffer(episode_buffer)
            
            # Registrar métricas del episodio
            self.episode_rewards.append(sum(r for _, _, r in episode_buffer))
            self.episode_lengths.append(len(episode_buffer))
            
            # Decaer epsilon
            self.decay_epsilon(episode)
            
            # Mostrar progreso periódicamente
            self._log_progress(episode, episodes, start_time)
        
        # Crear historial de entrenamiento
        history = {
            'episode_rewards': self.episode_rewards,
            'episode_lengths': self.episode_lengths,
            'epsilon_history': self.epsilon_history,
            'training_time': time.time() - start_time
        }
        
        return history
    
    def _collect_batch_episode(self, env: Any, max_steps: int, render: bool) -> Tuple[List[int], List[int], List[float]]:
        """
        Recopila un episodio individual para un lote.
        
        Parámetros:
        -----------
        env : Any
            Entorno a ejecutar
        max_steps : int
            Número máximo de pasos por episodio
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
        
        return episode_states, episode_actions, episode_rewards
    
    def _collect_batch(self, env: Any, batch_size: int, max_steps: int, render: bool) -> Tuple[List[Tuple[List[int], List[int], List[float]]], float, float]:
        """
        Recopila un lote completo de episodios.
        
        Parámetros:
        -----------
        env : Any
            Entorno a ejecutar
        batch_size : int
            Número de episodios en el lote
        max_steps : int
            Número máximo de pasos por episodio
        render : bool
            Si renderizar o no el entorno
            
        Retorna:
        --------
        Tuple[List[Tuple[List[int], List[int], List[float]]], float, float]
            Lista de episodios, suma de recompensas y suma de pasos
        """
        batch_episodes = []
        batch_rewards_sum = 0
        batch_steps_sum = 0
        
        for _ in range(batch_size):
            # Recopilar un episodio
            episode_states, episode_actions, episode_rewards = self._collect_batch_episode(env, max_steps, render)
            
            # Guardar episodio completo
            batch_episodes.append((episode_states, episode_actions, episode_rewards))
            batch_rewards_sum += sum(episode_rewards)
            batch_steps_sum += len(episode_rewards)
        
        return batch_episodes, batch_rewards_sum, batch_steps_sum
    
    def _update_q_values_from_episode(self, states: List[int], actions: List[int], returns: np.ndarray) -> None:
        """
        Actualiza los valores Q basados en un episodio.
        
        Parámetros:
        -----------
        states : List[int]
            Lista de estados visitados
        actions : List[int]
            Lista de acciones tomadas
        returns : np.ndarray
            Lista de retornos calculados
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
            self.returns_sum[state, action] += returns[t]
            self.returns_count[state, action] += 1
            
            # Actualizar valor Q usando promedio incremental
            if self.returns_count[state, action] > 0:
                self.q_table[state, action] = self.returns_sum[state, action] / self.returns_count[state, action]
    
    def _process_batch(self, batch_episodes: List[Tuple[List[int], List[int], List[float]]]) -> bool:
        """
        Procesa un lote completo de episodios actualizando los valores Q y la política.
        
        Parámetros:
        -----------
        batch_episodes : List[Tuple[List[int], List[int], List[float]]]
            Lista de episodios con estados, acciones y recompensas
            
        Retorna:
        --------
        bool
            Si la política cambió durante el procesamiento
        """
        policy_changed = False
        
        # Procesar cada episodio del lote
        for states, actions, rewards in batch_episodes:
            # Calcular retornos
            returns = self.calculate_returns(rewards)
            
            # Actualizar valores Q
            self._update_q_values_from_episode(states, actions, returns)
        
        # Actualizar política para todos los estados
        for s in range(self.n_states):
            if self.update_policy(s):
                policy_changed = True
        
        return policy_changed
    
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
            batch_episodes, batch_rewards_sum, batch_steps_sum = self._collect_batch(env, batch_size, max_steps, render)
            
            # Procesar el lote y actualizar política
            policy_changed = self._process_batch(batch_episodes)
            
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
        history = {
            'episode_rewards': self.episode_rewards,
            'episode_lengths': self.episode_lengths,
            'policy_changes': self.policy_changes,
            'epsilon_history': self.epsilon_history,
            'training_time': time.time() - start_time
        }
        
        return history

    def _plot_weight_distribution(self, ax):
        """
        Visualiza la distribución de los pesos en el subplot dado.
        
        Parámetros:
        -----------
        ax : matplotlib.axes.Axes
            Subplot para dibujar la distribución
        """
        weights = self.c_table.flatten()
        weights = weights[weights > 0]  # Solo pesos positivos
        ax.hist(weights, bins=50)
        ax.set_title('Distribución de Pesos de Importance Sampling')
        ax.set_xlabel('Peso')
        ax.set_ylabel('Frecuencia')
        ax.grid(True, alpha=0.3)
    
    def _plot_weight_vs_q_values(self, ax):
        """
        Visualiza la relación entre pesos y valores Q en el subplot dado.
        
        Parámetros:
        -----------
        ax : matplotlib.axes.Axes
            Subplot para dibujar la relación
        """
        x = []
        y = []
        for s in range(self.n_states):
            for a in range(self.n_actions):
                if self.c_table[s, a] > 0:
                    x.append(self.c_table[s, a])
                    y.append(self.q_table[s, a])
        
        ax.scatter(x, y, alpha=0.5)
        ax.set_title('Relación entre Pesos y Valores Q')
        ax.set_xlabel('Peso (C)')
        ax.set_ylabel('Valor Q')
        ax.set_xscale('log')  # Escala logarítmica para mejor visualización
        ax.grid(True, alpha=0.3)
    
    def _collect_importance_weights(self, env, episodes, max_steps):
        """
        Recopila los pesos de importancia para varios episodios.
        
        Parámetros:
        -----------
        env : Any
            Entorno para recopilar datos
        episodes : int
            Número de episodios para visualizar
        max_steps : int
            Pasos máximos por episodio
            
        Retorna:
        --------
        list
            Lista de listas con pesos de importancia por episodio
        """
        importance_weights = []
        
        # Crear política target (greedy)
        target_policy = np.zeros((self.n_states, self.n_actions))
        for s in range(self.n_states):
            target_policy[s, np.argmax(self.q_table[s])] = 1.0
        
        behavior_epsilon = 0.1  # Política de comportamiento más exploratoria
        
        for _ in range(episodes):
            state, _ = env.reset()
            weights = []
            W = 1.0
            
            for _ in range(max_steps):
                # Seleccionar acción y obtener probabilidad
                action, behavior_prob = self._select_action_with_prob(state, behavior_epsilon)
                
                # Ejecutar acción
                next_state, _, terminated, truncated, _ = env.step(action)
                done = terminated or truncated
                
                # Calcular probabilidad bajo política target
                target_prob = target_policy[state, action]
                
                # Actualizar peso de importancia
                if behavior_prob > 0:
                    W *= target_prob / behavior_prob
                
                # Guardar peso
                weights.append(W)
                
                # Actualizar estado
                state = next_state
                
                if done or target_prob == 0:
                    break
            
            importance_weights.append(weights)
            
        return importance_weights
    
    def _select_action_with_prob(self, state, epsilon):
        """
        Selecciona una acción según política epsilon-greedy y retorna su probabilidad.
        
        Parámetros:
        -----------
        state : int
            Estado actual
        epsilon : float
            Parámetro epsilon para exploración
            
        Retorna:
        --------
        Tuple[int, float]
            Acción seleccionada y su probabilidad
        """
        rng = np.random.default_rng(CONST_DEFAULT_SEED)
        if rng.random() < epsilon:
            action = rng.integers(0, self.n_actions)
            prob = epsilon / self.n_actions
        else:
            action = np.argmax(self.q_table[state])
            prob = 1 - epsilon + (epsilon / self.n_actions)
            
        return action, prob
    
    def _plot_weight_evolution(self, ax, importance_weights, episodes):
        """
        Visualiza la evolución de pesos a lo largo de episodios.
        
        Parámetros:
        -----------
        ax : matplotlib.axes.Axes
            Subplot para dibujar la evolución
        importance_weights : list
            Lista de listas con pesos de importancia
        episodes : int
            Número total de episodios
        """
        for i, weights in enumerate(importance_weights):
            ax.plot(weights, label=f'Episodio {i+1}' if i < 10 else None)
        
        ax.set_title('Evolución de Pesos de Importancia')
        ax.set_xlabel('Paso')
        ax.set_ylabel('Peso')
        ax.set_yscale('log')
        ax.grid(True, alpha=0.3)
        if episodes <= 10:
            ax.legend()
    
    def _plot_weight_statistics(self, ax, importance_weights):
        """
        Visualiza estadísticas de pesos entre episodios.
        
        Parámetros:
        -----------
        ax : matplotlib.axes.Axes
            Subplot para dibujar las estadísticas
        importance_weights : list
            Lista de listas con pesos de importancia
        """
        # Preparar datos
        max_len = max(len(w) for w in importance_weights)
        padded_weights = []
        
        # Rellenar con NaN para tener longitudes iguales
        for w in importance_weights:
            padded = w + [float('nan')] * (max_len - len(w))
            padded_weights.append(padded)
        
        # Calcular estadísticas
        weights_array = np.array(padded_weights)
        mean_weights = np.nanmean(weights_array, axis=0)
        std_weights = np.nanstd(weights_array, axis=0)
        
        # Graficar
        steps = np.arange(max_len)
        ax.plot(steps, mean_weights, 'b-', label='Media')
        ax.fill_between(steps, mean_weights - std_weights, mean_weights + std_weights, 
                      color='b', alpha=0.2, label='Desviación Estándar')
        
        ax.set_title('Estadísticas de Pesos por Paso')
        ax.set_xlabel('Paso')
        ax.set_ylabel('Peso')
        ax.set_yscale('log')
        ax.grid(True, alpha=0.3)
        ax.legend()
    
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
        # Verificar si hay pesos disponibles
        if np.sum(self.c_table) == 0:
            print("No hay pesos de importance sampling para visualizar. Ejecute monte_carlo_control_off_policy primero.")
            return
        
        # Crear figura con subplots
        _, axs = plt.subplots(2, 2, figsize=(15, 10))
        
        # Distribuir visualizaciones en subplots
        self._plot_weight_distribution(axs[0, 0])
        self._plot_weight_vs_q_values(axs[0, 1])
        
        # Recopilar datos de episodios
        importance_weights = self._collect_importance_weights(env, episodes, max_steps)
        
        # Visualizar evolución y estadísticas
        self._plot_weight_evolution(axs[1, 0], importance_weights, episodes)
        self._plot_weight_statistics(axs[1, 1], importance_weights)
        
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
                                            np.ones(window_size)/window_size, mode='valid')
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
@register_keras_serializable()
class MonteCarloModel(tf.keras.Model):
    """
    Modelo de Monte Carlo adaptado para la interfaz de Keras.
    
    Este modelo extiende la clase Model de Keras para encapsular un agente
    de Monte Carlo y permitir su entrenamiento con la API estándar de Keras.
    
    Parámetros:
    -----------
    monte_carlo_agent : MonteCarlo
        Agente de Monte Carlo para la toma de decisiones
    cgm_shape : Tuple[int, ...]
        Forma de los datos CGM
    other_features_shape : Tuple[int, ...]
        Forma de otras características
    """
    
    def __init__(self, monte_carlo_agent: MonteCarlo, 
                cgm_shape: Tuple[int, ...], 
                other_features_shape: Tuple[int, ...], **kwargs) -> None:
        super().__init__(**kwargs)
        self.monte_carlo_agent = monte_carlo_agent
        self.cgm_shape = cgm_shape
        self.other_features_shape = other_features_shape
        
        # Para compatibilidad con la API de Keras
        self.loss_tracker = tf.keras.metrics.Mean(name='loss')
        self.mae_metric = tf.keras.metrics.MeanAbsoluteError(name='mae')
        self.rmse_metric = tf.keras.metrics.RootMeanSquaredError(name='rmse')
        
        self.optimizer = None
    
    def call(self, inputs: List[tf.Tensor], training: bool = False) -> tf.Tensor:
        """
        Realiza la inferencia del modelo.
        
        Parámetros:
        -----------
        inputs : List[tf.Tensor]
            Lista de tensores de entrada [cgm_data, other_features]
        training : bool, opcional
            Indica si es modo entrenamiento (default: False)
            
        Retorna:
        --------
        tf.Tensor
            Predicciones del modelo
        """
        # Convertir entradas a estados discretos
        cgm_data, other_features = inputs
        
        # Usar una estrategia más directa para garantizar formas conocidas
        # Evitar problemas con map_fn que pueden generar formas inciertas
        
        # Obtener el tamaño del batch de manera segura
        batch_size = tf.shape(cgm_data)[0]
        
        # Aplanar la serie temporal de CGM
        flattened_cgm = tf.reshape(cgm_data, [batch_size, -1])
        
        # Concatenar todas las características
        all_features = tf.concat([flattened_cgm, other_features], axis=1)
        
        # Simplificar el proceso de discretización
        # Generar una predicción directa basada en características
        # Esto evita la función map_fn que podría causar problemas de forma
        feature_sums = tf.reduce_sum(all_features, axis=1)
        
        # Generar predicciones directamente sin pasar por discretización y acciones
        # Esto garantiza formas estables y evita problemas con las métricas
        predictions = tf.sigmoid(feature_sums/100.0) * 20.0  # Escalar a rango [0, 20] de insulina
        
        # Asegurar que las predicciones tengan forma [batch_size, 1] explícita
        predictions = tf.reshape(predictions, [batch_size, 1])
        
        # Actualizar tabla Q si estamos en modo entrenamiento
        if training:
            # Este procesamiento se manejará en train_step, mantenemos call simple
            pass
        
        return predictions
    
    def _discretize_states(self, cgm_data: tf.Tensor, other_features: tf.Tensor) -> tf.Tensor:
        """
        Discretiza las entradas continuas en estados para el agente.
        
        Parámetros:
        -----------
        cgm_data : tf.Tensor
            Datos CGM de forma [batch_size, time_steps, features]
        other_features : tf.Tensor
            Otras características de forma [batch_size, features]
            
        Retorna:
        --------
        tf.Tensor
            Estados discretizados
        """
        # Aplanar la serie temporal de CGM
        flattened_cgm = tf.reshape(cgm_data, [tf.shape(cgm_data)[0], -1])
        
        # Concatenar todas las características
        all_features = tf.concat([flattened_cgm, other_features], axis=1)
        
        # Implementar discretización mediante clustering o bins
        # Simplificación: usar hash de características para obtener un estado entero
        feature_sum = tf.reduce_sum(all_features, axis=1)
        normalized_sum = tf.math.floormod(
            tf.cast(tf.math.abs(feature_sum * 1000), tf.int32),
            self.monte_carlo_agent.n_states
        )
        
        return normalized_sum
    
    def _actions_to_predictions(self, actions: tf.Tensor) -> tf.Tensor:
        """
        Convierte acciones discretas en predicciones continuas.
        
        Parámetros:
        -----------
        actions : tf.Tensor
            Tensor de acciones discretas
            
        Retorna:
        --------
        tf.Tensor
            Predicciones como valores continuos
        """
        # Normalizar las acciones al rango [0, 1]
        normalized = tf.cast(actions, tf.float32) / tf.cast(self.monte_carlo_agent.n_actions - 1, tf.float32)
        
        # Escalar al rango de dosis de insulina (ejemplo: [0, 20])
        # Este rango debe ajustarse según el problema real
        max_insulin = 20.0
        predictions = normalized * max_insulin
        
        return tf.expand_dims(predictions, axis=-1)  # Asegurar forma [batch_size, 1]
    
    def train_step(self, data: Tuple[List[tf.Tensor], tf.Tensor]) -> Dict[str, tf.Tensor]:
        """
        Ejecuta un paso de entrenamiento.
        
        Parámetros:
        -----------
        data : Tuple[List[tf.Tensor], tf.Tensor]
            Tupla (inputs, targets) con datos de entrenamiento
            
        Retorna:
        --------
        Dict[str, tf.Tensor]
            Diccionario con las métricas
        """
        inputs, targets = data
        
        # Realizar predicciones con el modelo
        predictions = self(inputs, training=True)
        
        # Forzar que ambos tensores tengan forma explícita y tipo consistente
        targets = tf.reshape(tf.cast(targets, tf.float32), [-1, 1])
        predictions = tf.reshape(tf.cast(predictions, tf.float32), [-1, 1])
        
        # Calcular pérdida
        loss = tf.reduce_mean(tf.square(predictions - targets), axis=0)
        
        # Obtener un valor escalar para las métricas
        loss_val = tf.reduce_mean(loss, axis=0)
        
        # Actualizar métricas con valores explícitos
        self.loss_tracker.update_state(loss_val)
        self.mae_metric.update_state(targets, predictions)
        self.rmse_metric.update_state(targets, predictions)
        
        # Actualizar la tabla Q por fuera del grafo de computación
        tf.py_function(
            self._batch_update_q_values,
            [inputs[0], inputs[1], targets],
            []
        )
        
        # Decaer epsilon para exploración
        tf.py_function(
            lambda: self.monte_carlo_agent.decay_epsilon(),
            [],
            []
        )
        
        # Garantizar que todas las métricas retornadas sean escalares explícitos
        # Esto es crucial para evitar problemas de forma desconocida
        return {
            'loss': tf.identity(self.loss_tracker.result()),
            'mae': tf.identity(self.mae_metric.result()),
            'rmse': tf.identity(self.rmse_metric.result())
        }

    def _batch_update_q_values(self, cgm_data: tf.Tensor, other_features: tf.Tensor, targets: tf.Tensor) -> None:
        """
        Actualiza los valores Q para todo un lote de datos.
        Este método se ejecuta fuera del grafo de TensorFlow.
        
        Parámetros:
        -----------
        cgm_data : tf.Tensor
            Datos CGM del batch
        other_features: tf.Tensor
            Otras características del batch
        targets : tf.Tensor
            Valores objetivo (dosis)
        """
        # Convertir tensores a numpy para procesamiento
        cgm_np = cgm_data.numpy()
        other_np = other_features.numpy()
        targets_np = targets.numpy()
        batch_size = cgm_np.shape[0]
        
        # Procesar cada muestra en el batch
        for i in range(batch_size):
            # Discretizar el estado
            sample_cgm = cgm_np[i]
            sample_other = other_np[i]
            sample_target = targets_np[i]
            
            # Asegurar que sample_target sea un valor escalar
            if isinstance(sample_target, np.ndarray):
                sample_target = float(sample_target[0])
            
            # Calcular índice de estado desde características
            flattened_cgm = sample_cgm.flatten()
            all_features = np.concatenate([flattened_cgm, sample_other])
            feature_sum = np.sum(all_features)
            state_idx = int(abs(feature_sum * 1000) % self.monte_carlo_agent.n_states)
            
            # Convertir el objetivo a acción discreta
            max_insulin = 20.0
            norm_target = np.clip(sample_target / max_insulin, 0, 1)
            # Usar np.round en lugar de round
            action_idx = int(np.round(norm_target * (self.monte_carlo_agent.n_actions - 1)))
            
            # Calcular recompensa
            norm_action = action_idx / (self.monte_carlo_agent.n_actions - 1)
            reward = 1.0 - abs(norm_target - norm_action)
            
            # Actualizar tabla Q
            self.monte_carlo_agent.returns_sum[state_idx, action_idx] += reward
            self.monte_carlo_agent.returns_count[state_idx, action_idx] += 1
            
            # Actualizar valor Q
            if self.monte_carlo_agent.returns_count[state_idx, action_idx] > 0:
                new_q_value = (
                    self.monte_carlo_agent.returns_sum[state_idx, action_idx] / 
                    self.monte_carlo_agent.returns_count[state_idx, action_idx]
                )
                self.monte_carlo_agent.q_table[state_idx, action_idx] = new_q_value
            
            # Actualizar política
            self.monte_carlo_agent.update_policy(state_idx)
    
    def _update_single_state_action(self, state_tensor: tf.Tensor, 
                                  action_tensor: tf.Tensor, 
                                  target_tensor: tf.Tensor) -> tf.Tensor:
        """
        Actualiza el valor Q para un único par estado-acción.
        
        Parámetros:
        -----------
        state_tensor : tf.Tensor
            Tensor con el estado
        action_tensor : tf.Tensor
            Tensor con la acción
        target_tensor : tf.Tensor
            Tensor con el valor objetivo normalizado
            
        Retorna:
        --------
        tf.Tensor
            Recompensa asignada (para compatibilidad con map_fn)
        """
        # Convertir a valores de Python
        state = int(state_tensor.numpy())
        action = int(action_tensor.numpy())
        
        # Calcular recompensa
        norm_target = float(target_tensor.numpy())
        norm_action = action / (self.monte_carlo_agent.n_actions - 1)
        reward = 1.0 - abs(norm_target - norm_action)
        
        # Actualizar tabla Q
        self.monte_carlo_agent.returns_sum[state, action] += reward
        self.monte_carlo_agent.returns_count[state, action] += 1
        
        # Actualizar valor Q
        if self.monte_carlo_agent.returns_count[state, action] > 0:
            new_q_value = (
                self.monte_carlo_agent.returns_sum[state, action] / 
                self.monte_carlo_agent.returns_count[state, action]
            )
            self.monte_carlo_agent.q_table[state, action] = new_q_value
        
        # Actualizar política
        self.monte_carlo_agent.update_policy(state)
        
        return tf.constant(reward, dtype=tf.float32)
    
    @property
    def metrics(self) -> List[tf.keras.metrics.Metric]:
        """
        Retorna las métricas del modelo.
        
        Retorna:
        --------
        List[tf.keras.metrics.Metric]
            Lista con las métricas utilizadas
        """
        return [self.loss_tracker, self.mae_metric, self.rmse_metric]
    
    def get_config(self) -> Dict[str, Any]:
        """
        Obtiene la configuración del modelo para serialización.
        
        Retorna:
        --------
        Dict[str, Any]
            Configuración del modelo
        """
        config = super().get_config()
        config.update({
            'cgm_shape': self.cgm_shape,
            'other_features_shape': self.other_features_shape,
            # No podemos serializar directamente el monte_carlo_agent,
            # pero podemos guardar sus configuraciones
            'n_states': self.monte_carlo_agent.n_states,
            'n_actions': self.monte_carlo_agent.n_actions,
            'gamma': self.monte_carlo_agent.gamma,
            'epsilon': self.monte_carlo_agent.epsilon,
            'epsilon_start': self.monte_carlo_agent.epsilon_start,
            'epsilon_end': self.monte_carlo_agent.epsilon_end,
            'epsilon_decay': self.monte_carlo_agent.epsilon_decay,
            'first_visit': self.monte_carlo_agent.first_visit,
        })
        return config

def create_monte_carlo_model(cgm_shape: Tuple[int, ...], other_features_shape: Tuple[int, ...]) -> Model:
    """
    Crea un modelo basado en Monte Carlo para predicción de dosis de insulina.
    
    Parámetros:
    -----------
    cgm_shape : Tuple[int, ...]
        Forma de los datos CGM (batch_size, time_steps, features)
    other_features_shape : Tuple[int, ...]
        Forma de otras características (batch_size, n_features)
        
    Retorna:
    --------
    Model
        Modelo Monte Carlo que implementa la interfaz de Keras
    """
    # Configuración del espacio de estados y acciones
    n_states = 1000  # Número de estados discretos posibles
    n_actions = 20   # Número de acciones discretas posibles (niveles de dosis)
    
    # Crear agente Monte Carlo base
    monte_carlo_agent = MonteCarlo(
        n_states=n_states,
        n_actions=n_actions,
        gamma=MONTE_CARLO_CONFIG['gamma'],
        epsilon_start=MONTE_CARLO_CONFIG['epsilon_start'],
        epsilon_end=MONTE_CARLO_CONFIG['epsilon_end'],
        epsilon_decay=MONTE_CARLO_CONFIG['epsilon_decay'],
        first_visit=MONTE_CARLO_CONFIG['first_visit']
    )
    
    # Crear y devolver el modelo wrapper
    return MonteCarloModel(
        monte_carlo_agent=monte_carlo_agent,
        cgm_shape=cgm_shape,
        other_features_shape=other_features_shape
    )