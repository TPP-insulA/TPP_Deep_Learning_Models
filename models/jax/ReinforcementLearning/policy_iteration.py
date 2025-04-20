import os, sys
import jax
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
import time
from typing import Dict, List, Tuple, Optional, Union, Any, Callable
from tqdm import tqdm
from functools import partial
import joblib # For saving/loading numpy arrays efficiently

PROJECT_ROOT = os.path.abspath(os.getcwd())
sys.path.append(PROJECT_ROOT) 

from config.models_config import POLICY_ITERATION_CONFIG
from custom.model_wrapper import ModelWrapper # Import base class
from custom.printer import print_info, print_warning # For better logging
from constants.constants import CONST_LOSS, CONST_VAL_LOSS # Importar constantes comunes

# Constantes para rutas de figuras y mensajes recurrentes
FIGURES_DIR = os.path.join(PROJECT_ROOT, "figures", "jax", "policy_iteration")
CONST_ITERATION_LABEL = "Iteración"
CONST_REWARD_LABEL = "Recompensa"
CONST_VALUE_LABEL = "Valor"
CONST_TIME_LABEL = "Tiempo (segundos)"
CONST_POLICY_LABEL = "Política"
CONST_EINSUM_PATTERN = 'san,n->sa'  # Patrón para calcular valores esperados
TRANSITION_PROB = 1.0 # Probabilidad de transición (simplificada)

# Crear directorio para figuras si no existe
os.makedirs(FIGURES_DIR, exist_ok=True)


class PolicyIteration:
    """
    Implementación del algoritmo de Iteración de Política usando JAX.

    La Iteración de Política alterna entre Evaluación de Política (calcular la función
    de valor para la política actual) y Mejora de Política (hacer la política codiciosa
    respecto a la función de valor actual).
    """

    def __init__(
        self,
        n_states: int,
        n_actions: int,
        gamma: float = POLICY_ITERATION_CONFIG['gamma'],
        theta: float = POLICY_ITERATION_CONFIG['theta'],
        max_iterations: int = POLICY_ITERATION_CONFIG['max_iterations'],
        max_iterations_eval: int = POLICY_ITERATION_CONFIG['max_iterations_eval'],
        seed: int = POLICY_ITERATION_CONFIG.get('seed', 42)
    ) -> None:
        """
        Inicializa el agente de Iteración de Política.

        Parámetros:
        -----------
        n_states : int
            Número de estados en el entorno.
        n_actions : int
            Número de acciones en el entorno.
        gamma : float, opcional
            Factor de descuento (default: POLICY_ITERATION_CONFIG['gamma']).
        theta : float, opcional
            Umbral para convergencia (default: POLICY_ITERATION_CONFIG['theta']).
        max_iterations : int, opcional
            Número máximo de iteraciones de iteración de política
            (default: POLICY_ITERATION_CONFIG['max_iterations']).
        max_iterations_eval : int, opcional
            Número máximo de iteraciones para evaluación de política
            (default: POLICY_ITERATION_CONFIG['max_iterations_eval']).
        seed : int, opcional
            Semilla para reproducibilidad (default: POLICY_ITERATION_CONFIG.get('seed', 42)).
        """
        self.n_states = n_states
        self.n_actions = n_actions
        self.gamma = gamma
        self.theta = theta
        self.max_iterations = max_iterations
        self.max_iterations_eval = max_iterations_eval

        # Configurar claves para aleatorización
        self.key = jax.random.key(seed)
        self.rng = np.random.Generator(np.random.PCG64(seed)) # Usar Generator

        # Inicializar función de valor y política
        self.v = jnp.zeros(n_states)
        # Inicializar política aleatoria uniforme
        self.policy = jnp.ones((n_states, n_actions)) / n_actions

        # Para métricas
        self.policy_changes = []
        self.value_changes = []
        self.policy_iteration_times = []
        self.eval_iteration_counts = []

        # Compilar funciones clave para mejorar rendimiento
        self._init_jitted_functions()

    def _init_jitted_functions(self) -> None:
        """
        Inicializa funciones JIT-compiladas para mejorar el rendimiento.
        """
        # Definimos las funciones que serán compiladas con JIT
        self._jit_policy_evaluation_step = jax.jit(self._policy_evaluation_step)
        self._jit_calculate_state_values = jax.jit(self._calculate_state_values)
        self._jit_policy_improvement = jax.jit(self._policy_improvement)

    def _extract_matrices_from_P_R(
        self,
        P: Dict[int, Dict[int, List[Tuple[float, int, float, bool]]]],
        R: Dict[int, Dict[int, float]]
    ) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        """
        Extrae matrices JAX de transición, recompensa y terminales a partir de los diccionarios P y R.

        Parámetros:
        -----------
        P : Dict[int, Dict[int, List[Tuple[float, int, float, bool]]]]
            Diccionario de transiciones {estado: {accion: [(prob, prox_estado, recompensa, done)]}}.
        R : Dict[int, Dict[int, float]]
            Diccionario de recompensas {estado: {accion: recompensa_promedio}}.

        Retorna:
        --------
        Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]
            Matrices JAX: (probabilidades, recompensas, terminales).
            Shapes: (S, A, S), (S, A), (S, A).
        """
        # Crear matrices NumPy para llenar
        transition_probs_np = np.zeros((self.n_states, self.n_actions, self.n_states))
        rewards_np = np.zeros((self.n_states, self.n_actions))
        terminals_np = np.zeros((self.n_states, self.n_actions), dtype=bool)

        # Llenar matrices desde P
        for s in range(self.n_states):
            if s in P:
                for a in range(self.n_actions):
                    if a in P[s]:
                        transitions = P[s][a]
                        if transitions:
                            # Asumimos una sola transición posible por (s, a) en el MDP simplificado
                            prob, next_s, _, done = transitions[0]
                            # Asegurarse que next_s está dentro de los límites
                            if 0 <= next_s < self.n_states:
                                transition_probs_np[s, a, next_s] = prob
                            terminals_np[s, a] = done
                        # else: Si no hay transiciones, la probabilidad es 0 (ya inicializado)
            # else: Si un estado no está en P, se asume que no es alcanzable o es terminal.

        # Llenar matriz de recompensas desde R
        for s in range(self.n_states):
            if s in R:
                for a in range(self.n_actions):
                    if a in R[s]:
                        rewards_np[s, a] = R[s][a]
            # else: Si un estado no está en R, la recompensa es 0 (ya inicializado)

        # Convertir a JAX arrays
        transition_probs = jnp.array(transition_probs_np)
        rewards = jnp.array(rewards_np)
        terminals = jnp.array(terminals_np)

        return transition_probs, rewards, terminals

    # jit
    def _calculate_state_values(
        self,
        policy: jnp.ndarray,
        v: jnp.ndarray,
        transition_probs: jnp.ndarray,
        rewards: jnp.ndarray,
        terminals: jnp.ndarray
    ) -> jnp.ndarray:
        """
        Calcula los valores esperados de estado para todos los estados bajo la política dada. (JIT)

        Parámetros:
        -----------
        policy : jnp.ndarray
            Política actual (probabilidad de cada acción en cada estado). Shape: (S, A).
        v : jnp.ndarray
            Función de valor actual. Shape: (S,).
        transition_probs : jnp.ndarray
            Matriz de probabilidades de transición P(s'|s, a). Shape: (S, A, S).
        rewards : jnp.ndarray
            Matriz de recompensas R(s, a). Shape: (S, A).
        terminals : jnp.ndarray
            Matriz de indicadores de terminal T(s, a). Shape: (S, A).

        Retorna:
        --------
        jnp.ndarray
            Nuevos valores de estado esperados. Shape: (S,).
        """
        # Valor esperado de tomar acción 'a' en estado 's':
        # q(s, a) = R(s, a) + gamma * sum_{s'} P(s'|s, a) * V(s') * (1 - T(s, a))
        # Usamos einsum para la suma ponderada sobre s'
        expected_future_values = jnp.einsum('sak,k->sa', transition_probs, v, optimize='optimal') # Ajustado patrón einsum
        q_sa = rewards + self.gamma * expected_future_values * (1 - terminals.astype(jnp.float32))

        # Valor esperado del estado 's' bajo la política 'policy':
        # V(s) = sum_a policy(a|s) * q(s, a)
        v_new = jnp.einsum('sa,sa->s', policy, q_sa, optimize='optimal')

        return v_new

    # jit
    def _policy_evaluation_step(
        self,
        v: jnp.ndarray,
        policy: jnp.ndarray,
        transition_probs: jnp.ndarray,
        rewards: jnp.ndarray,
        terminals: jnp.ndarray
    ) -> Tuple[jnp.ndarray, float]:
        """
        Realiza un paso de barrido de evaluación de política. (JIT)

        Parámetros:
        -----------
        v : jnp.ndarray
            Función de valor actual.
        policy : jnp.ndarray
            Política actual a evaluar.
        transition_probs : jnp.ndarray
            Matriz de probabilidades de transición P(s'|s, a).
        rewards : jnp.ndarray
            Matriz de recompensas R(s, a).
        terminals : jnp.ndarray
            Matriz de indicadores de terminal T(s, a).

        Retorna:
        --------
        Tuple[jnp.ndarray, float]
            (Nueva función de valor, cambio máximo en la función de valor).
        """
        # Usar la función JIT interna si está disponible
        v_new = self._jit_calculate_state_values(policy, v, transition_probs, rewards, terminals)
        delta = jnp.max(jnp.abs(v_new - v))
        return v_new, delta

    def policy_evaluation(
        self,
        transition_probs: jnp.ndarray, # Cambiado de env
        rewards: jnp.ndarray,         # Añadido
        terminals: jnp.ndarray,       # Añadido
        policy: jnp.ndarray,
        use_jit: bool = True
    ) -> Tuple[jnp.ndarray, int]:
        """
        Evalúa una política dada calculando su función de valor V.

        Parámetros:
        -----------
        transition_probs : jnp.ndarray
            Matriz de probabilidades de transición P(s'|s, a).
        rewards : jnp.ndarray
            Matriz de recompensas R(s, a).
        terminals : jnp.ndarray
            Matriz de indicadores de terminal T(s, a).
        policy : jnp.ndarray
            Política a evaluar.
        use_jit : bool, opcional
            Si usar la versión JIT compilada (default: True).

        Retorna:
        --------
        Tuple[jnp.ndarray, int]
            (Función de valor V para la política, número de iteraciones realizadas).
        """
        v = jnp.zeros(self.n_states) # Empezar desde cero
        # Las matrices ya se pasan como argumento
        eval_step_fn = self._jit_policy_evaluation_step if use_jit else self._policy_evaluation_step
        iterations = 0

        for i in range(self.max_iterations_eval):
            iterations = i + 1
            v_new, delta = eval_step_fn(v, policy, transition_probs, rewards, terminals)
            v = v_new
            if delta < self.theta:
                break # Convergencia
        self.eval_iteration_counts.append(iterations)
        return v, iterations

    # jit
    def _policy_improvement(
        self,
        v: jnp.ndarray,
        transition_probs: jnp.ndarray,
        rewards: jnp.ndarray,
        terminals: jnp.ndarray
    ) -> Tuple[jnp.ndarray, bool]:
        """
        Mejora la política haciéndola codiciosa respecto a la función de valor V. (JIT)

        Parámetros:
        -----------
        v : jnp.ndarray
            Función de valor actual.
        transition_probs: jnp.ndarray
            Matriz de probabilidades de transición P(s'|s, a).
        rewards: jnp.ndarray
            Matriz de recompensas R(s, a).
        terminals: jnp.ndarray
            Matriz de indicadores de terminal T(s, a).

        Retorna:
        --------
        Tuple[jnp.ndarray, bool]
            (Nueva política determinista codiciosa, indicador de estabilidad).
        """
        # Calcular valores Q(s, a)
        expected_future_values = jnp.einsum('sak,k->sa', transition_probs, v, optimize='optimal') # Ajustado patrón einsum
        q_sa = rewards + self.gamma * expected_future_values * (1 - terminals.astype(jnp.float32))

        # Encontrar la mejor acción para cada estado
        best_actions = jnp.argmax(q_sa, axis=1)

        # Crear la nueva política determinista (one-hot)
        new_policy = jax.nn.one_hot(best_actions, num_classes=self.n_actions)

        return new_policy

    def train(
        self,
        P: Dict[int, Dict[int, List[Tuple[float, int, float, bool]]]], # Cambiado de env
        R: Dict[int, Dict[int, float]],                               # Añadido
        use_jit: bool = True
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """
        Ejecuta el algoritmo de Iteración de Política completo usando P y R precalculados.

        Parámetros:
        -----------
        P : Dict
            Diccionario de transiciones precalculado.
        R : Dict
            Diccionario de recompensas precalculado.
        use_jit : bool, opcional
            Si usar las versiones JIT compiladas (default: True).

        Retorna:
        --------
        Tuple[jnp.ndarray, jnp.ndarray]
            (Política óptima encontrada, Función de valor óptima encontrada).
        """
        policy_stable = False
        start_time = time.time()

        # Extraer matrices JAX de los diccionarios P y R
        transition_probs, rewards, terminals = self._extract_matrices_from_P_R(P, R)

        improve_fn = self._jit_policy_improvement if use_jit else self._policy_improvement

        for i in range(self.max_iterations):
            iter_start_time = time.time()
            # 1. Evaluación de Política
            self.v, eval_iters = self.policy_evaluation(
                transition_probs, rewards, terminals, self.policy, use_jit
            )
            self.value_changes.append(jnp.max(jnp.abs(self.v))) # Registrar cambio o valor máximo

            # 2. Mejora de Política
            old_policy = self.policy.copy()
            new_policy = improve_fn(self.v, transition_probs, rewards, terminals)
            self.policy = new_policy

            # Comprobar estabilidad
            if jnp.array_equal(new_policy, old_policy):
                policy_stable = True
                self.policy_changes.append(0) # No hubo cambios
                iter_time = time.time() - iter_start_time
                self.policy_iteration_times.append(iter_time)
                print_info(f"Iteración {i+1}: Política estable encontrada en {iter_time:.4f}s. Evaluación tomó {eval_iters} iteraciones.")
                break
            else:
                changes = jnp.sum(jnp.argmax(new_policy, axis=1) != jnp.argmax(old_policy, axis=1))
                self.policy_changes.append(int(changes)) # Registrar número de cambios
                iter_time = time.time() - iter_start_time
                self.policy_iteration_times.append(iter_time)
                print_info(f"Iteración {i+1}: Política mejorada en {iter_time:.4f}s ({changes} cambios). Evaluación tomó {eval_iters} iteraciones.")

        total_time = time.time() - start_time
        if not policy_stable:
            print_warning(f"Iteración de Política no convergió en {self.max_iterations} iteraciones.")
        print_info(f"Iteración de Política finalizada en {total_time:.2f} segundos.")

        return self.policy, self.v

    def get_action(self, state: int) -> int:
        """
        Obtiene la acción óptima para un estado dado según la política actual.

        Parámetros:
        -----------
        state : int
            Estado actual.

        Retorna:
        --------
        int
            Acción óptima.
        """
        if not (0 <= state < self.n_states):
             raise ValueError(f"Estado {state} fuera de rango [0, {self.n_states-1}]")
        # La política es determinista después de la mejora, así que argmax es suficiente
        return int(jnp.argmax(self.policy[state]))

    def plot_metrics(self, save_plots: bool = True) -> None:
        """
        Genera gráficos de las métricas de entrenamiento.

        Parámetros:
        -----------
        save_plots : bool, opcional
            Si guardar los gráficos en archivos (default: True).
        """
        iterations = range(1, len(self.policy_changes) + 1)

        plt.style.use('seaborn-v0_8-whitegrid')
        fig, axs = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle("Métricas de Entrenamiento - Iteración de Política (JAX)", fontsize=16)

        # Cambios en la política
        axs[0, 0].plot(iterations, self.policy_changes, marker='o', linestyle='-', color='b')
        axs[0, 0].set_title("Cambios en la Política por Iteración")
        axs[0, 0].set_xlabel(CONST_ITERATION_LABEL)
        axs[0, 0].set_ylabel("Número de Estados con Política Cambiada")
        axs[0, 0].grid(True)

        # Cambios en la función de valor (usando el valor máximo como proxy)
        axs[0, 1].plot(iterations, self.value_changes, marker='s', linestyle='--', color='r')
        axs[0, 1].set_title("Magnitud Máxima de la Función de Valor por Iteración")
        axs[0, 1].set_xlabel(CONST_ITERATION_LABEL)
        axs[0, 1].set_ylabel("Máximo |V(s)|")
        axs[0, 1].grid(True)

        # Tiempo por iteración de política
        axs[1, 0].plot(iterations, self.policy_iteration_times, marker='^', linestyle='-.', color='g')
        axs[1, 0].set_title("Tiempo por Iteración de Política")
        axs[1, 0].set_xlabel(CONST_ITERATION_LABEL)
        axs[1, 0].set_ylabel(CONST_TIME_LABEL)
        axs[1, 0].grid(True)

        # Iteraciones de evaluación de política
        axs[1, 1].plot(iterations, self.eval_iteration_counts, marker='d', linestyle=':', color='purple')
        axs[1, 1].set_title("Iteraciones por Evaluación de Política")
        axs[1, 1].set_xlabel(CONST_ITERATION_LABEL)
        axs[1, 1].set_ylabel("Número de Iteraciones de Evaluación")
        axs[1, 1].grid(True)

        plt.tight_layout(rect=[0, 0.03, 1, 0.95]) # Ajustar para título

        if save_plots:
            plot_path = os.path.join(FIGURES_DIR, "policy_iteration_metrics.png")
            plt.savefig(plot_path, dpi=300)
            print_info(f"Gráfico de métricas guardado en: {plot_path}")
        else:
            plt.show()
        plt.close(fig)


class PolicyIterationWrapper(ModelWrapper):
    """
    Wrapper para el agente PolicyIteration compatible con la interfaz ModelWrapper.
    Este wrapper maneja la discretización, construcción del MDP y llamada al agente PI.
    """
    def __init__(
        self,
        bins: int = POLICY_ITERATION_CONFIG['bins'],
        state_bounds: Optional[List[Tuple[float, float]]] = POLICY_ITERATION_CONFIG['state_bounds'],
        action_bins: Optional[int] = POLICY_ITERATION_CONFIG.get('num_actions'),
        action_bounds: Optional[List[Tuple[float, float]]] = POLICY_ITERATION_CONFIG['action_bounds'],
        n_states: Optional[int] = None,
        **pi_kwargs
    ) -> None:
        """
        Inicializa el wrapper para el algoritmo de Iteración de Política.
        
        Parámetros:
        -----------
        bins : int
            Número de bins para discretización (default: POLICY_ITERATION_CONFIG['bins']).
        state_bounds : Optional[List[Tuple[float, float]]]
            Límites para el espacio de estado antes de discretizar (default: POLICY_ITERATION_CONFIG['state_bounds']).
            Se espera una lista de tuplas, e.g., [(cgm_min, cgm_max), (time_min, time_max), ...].
        action_bins : Optional[int]
            Número de acciones discretas. Si es None, se usa 10 por defecto (default: POLICY_ITERATION_CONFIG.get('num_actions')).
        action_bounds : Optional[List[Tuple[float, float]]]
            Límites para el espacio de acción continuo antes de discretizar (default: POLICY_ITERATION_CONFIG['action_bounds']).
            Se espera una lista de tuplas, e.g., [(dose_min, dose_max)].
        n_states : Optional[int]
            Número total de estados discretos. Si es None, se calcula a partir de `bins` y `state_bounds` (default: None).
        **pi_kwargs
            Argumentos adicionales para el agente PolicyIteration.
        """
        super().__init__()
        print_info("Inicializando PolicyIterationWrapper...")

        # Guardar configuración de discretización y aplicar default para action_bins si es None
        self.bins = bins
        self.state_bounds = state_bounds if state_bounds else [(0, 400), (0, 24*60), (0, 10)]  # Defaults si no se proveen
        self.action_bins = action_bins if action_bins is not None else 10
        self.action_bounds = action_bounds if action_bounds is not None else [(0, 15)]  # Default si no se proveen
        self.pi_kwargs = pi_kwargs  # Guardar kwargs para el agente PI

        # Validar state_bounds
        if not isinstance(self.state_bounds, list) or not all(isinstance(b, tuple) and len(b) == 2 for b in self.state_bounds):
            raise ValueError("state_bounds debe ser una lista de tuplas (min, max).")

        # Calcular número de estados si no se proporciona
        self._n_states = n_states if n_states is not None else self._calculate_max_states()
        # Usar la misma convención para accciones (_n_actions en lugar de n_actions)
        self._n_actions = self.action_bins
        
        # Configurar dimensiones específicas para discretización
        self.cgm_bins = bins
        self.time_bins = bins
        self.insulin_bins = bins
        self._calculated_max_states = self.cgm_bins * self.time_bins * self.insulin_bins
        
        # Configurar rangos y límites
        self.cgm_range = self.state_bounds[0]
        self.time_range = (0, 24*60)  # minutos en un día
        self.insulin_range = self.state_bounds[2] if len(self.state_bounds) > 2 else (0, 10)
        self.max_dose = self.action_bounds[0][1] if self.action_bounds else 15
        
        # Precalcular bordes para discretización
        self._cgm_edges = np.linspace(self.cgm_range[0], self.cgm_range[1], self.cgm_bins + 1)
        self._time_edges = np.linspace(0, 24, self.time_bins + 1)
        self._insulin_edges = np.linspace(self.insulin_range[0], self.insulin_range[1], self.insulin_bins + 1)
        self._action_doses = np.linspace(0, self.max_dose, self._n_actions)
        
        # Inicializar historial y flag de entrenamiento
        self.history = {CONST_LOSS: [], CONST_VAL_LOSS: []}
        self.is_trained = False
        
        # Inicializar pi_agent a None (esta era la línea faltante)
        self.pi_agent = None
        
        print_info(f"  Número de estados discretos: {self._n_states}")
        print_info(f"  Número de acciones discretas: {self._n_actions}")

    def _calculate_max_states(self) -> int:
        """Calcula el número máximo de estados basado en los bins."""
        # Asume que cada dimensión definida en state_bounds se discretiza con self.bins
        num_dims = len(self.state_bounds)
        return self.bins ** num_dims

    def start(self, x_cgm: np.ndarray, x_other: np.ndarray, y: np.ndarray,
             rng_key: Optional[jax.random.PRNGKey] = None) -> Any:
        """
        Inicializa el agente PolicyIteration.

        Parámetros:
        -----------
        x_cgm : np.ndarray
            Datos CGM de entrada (no usados directamente aquí, pero parte de la interfaz).
        x_other : np.ndarray
            Otras características de entrada (no usados directamente aquí).
        y : np.ndarray
            Valores objetivo (no usados directamente aquí).
        rng_key : Optional[jax.random.PRNGKey], opcional
            Clave JAX RNG (no usada directamente por PI, usa su propia semilla).

        Retorna:
        --------
        Any
            La instancia del agente PolicyIteration inicializado.
        """
        if self.pi_agent is None:
            print_info(f"Inicializando agente PolicyIteration con {self._n_states} estados y {self._n_actions} acciones.")
            # Filtrar kwargs para que coincidan con los parámetros de PolicyIteration.__init__
            accepted_args = ['gamma', 'theta', 'max_iterations', 'max_iterations_eval', 'seed']
            filtered_pi_kwargs = {k: v for k, v in self.pi_kwargs.items() if k in accepted_args}

            self.pi_agent = PolicyIteration(
                n_states=self._n_states,
                n_actions=self._n_actions,
                **filtered_pi_kwargs # Pasar solo argumentos aceptados
            )
        return self.pi_agent

    def __call__(self, x_cgm: np.ndarray, x_other: np.ndarray) -> np.ndarray:
        """
        Permite llamar al wrapper como una función para predicción.
        
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
        return self.predict(x_cgm, x_other)

    def predict(self, x_cgm: np.ndarray, x_other: np.ndarray) -> np.ndarray:
        """
        Realiza predicciones utilizando el modelo de Policy Iteration entrenado.
        
        Parámetros:
        -----------
        x_cgm : np.ndarray
            Datos CGM para predicción
        x_other : np.ndarray
            Otras características para predicción
            
        Retorna:
        --------
        np.ndarray
            Predicciones de dosis de insulina
        """
        if self.pi_agent is None or not self.is_trained:
            print_warning("El modelo no ha sido entrenado o inicializado correctamente.")
            return np.zeros((x_cgm.shape[0], 1))
            
        n_samples = x_cgm.shape[0]
        predictions = np.zeros(n_samples)

        for i in range(n_samples):
            # Asegurarse de pasar x_other[i] si existe y es necesario
            current_x_other = x_other[i] if x_other.shape[1] > 0 else None
            state = self._discretize_state(x_cgm[i], current_x_other)
            
            # Obtener acción según la política entrenada
            action = self.pi_agent.get_action(state)
            
            # Convertir acción a dosis de insulina
            predictions[i] = self._convert_action_to_dose(action)

        return predictions.reshape(-1, 1)  # Devolver como columna

    def _discretize_state(self, cgm_sample: np.ndarray, other_sample: Optional[np.ndarray]) -> int:
        """
        Discretiza una muestra de entrada en un índice de estado.
        
        Parámetros:
        -----------
        cgm_sample : np.ndarray
            Muestra de datos CGM
        other_sample : Optional[np.ndarray]
            Muestra de otras características (hora, insulina activa)
            
        Retorna:
        --------
        int
            Índice discreto del estado
        """
        # Extraer características relevantes
        # Asumimos que el último valor de CGM es representativo
        last_cgm = cgm_sample[-1, 0] if cgm_sample.ndim == 2 else cgm_sample[0]
        
        # Extraer hora e insulina activa de other_sample si está disponible
        time_of_day = other_sample[0] if other_sample is not None and len(other_sample) > 0 else 12.0
        active_insulin = other_sample[1] if other_sample is not None and len(other_sample) > 1 else 0.0
        
        # Discretizar cada característica
        cgm_bin = np.digitize(last_cgm, self._cgm_edges[1:]) - 1
        time_bin = np.digitize(time_of_day, self._time_edges[1:]) - 1
        insulin_bin = np.digitize(active_insulin, self._insulin_edges[1:]) - 1
        
        # Asegurar que los bins estén dentro de los límites [0, n_bins-1]
        cgm_bin = np.clip(cgm_bin, 0, self.cgm_bins - 1)
        time_bin = np.clip(time_bin, 0, self.time_bins - 1)
        insulin_bin = np.clip(insulin_bin, 0, self.insulin_bins - 1)
        
        # Combinar bins en un índice de estado único
        state_index = np.ravel_multi_index(
            (cgm_bin, time_bin, insulin_bin),
            (self.cgm_bins, self.time_bins, self.insulin_bins)
        )
        
        # Ajustar si n_states es menor que el máximo calculable
        if self._n_states < self._calculated_max_states:
            state_index = state_index % self._n_states
        
        return int(state_index)

    def _convert_action_to_dose(self, action: int) -> float:
        """
        Convierte una acción discreta a una dosis continua.
        
        Parámetros:
        -----------
        action : int
            Índice de la acción a convertir
            
        Retorna:
        --------
        float
            Dosis de insulina
        """
        # Asegurar que la acción esté en rango válido
        action = np.clip(action, 0, self._n_actions - 1)
        
        # Convertir acción a dosis usando el array precalculado
        return self._action_doses[action]

    def _convert_dose_to_action(self, dose: float) -> int:
        """Convierte una dosis continua al índice de acción discreto más cercano."""
        # Encuentra el índice de la dosis en _action_doses que está más cerca de la dosis dada
        action = int(np.argmin(np.abs(self._action_doses - dose)))
        return action

    def _build_mdp_model(
        self,
        x_cgm: np.ndarray,
        x_other: np.ndarray,
        y: np.ndarray,
        verbose: int = 1
    ) -> Tuple[Dict[int, Dict[int, List[Tuple[float, int, float, bool]]]], Dict[int, Dict[int, float]]]:
        """
        Construye el modelo MDP simplificado (P, R) a partir de los datos.

        Parámetros:
        -----------
        x_cgm : np.ndarray
            Datos CGM de entrenamiento.
        x_other : np.ndarray
            Otras características de entrenamiento.
        y : np.ndarray
            Valores objetivo (dosis reales).
        verbose : int, opcional
            Nivel de verbosidad (default: 1).

        Retorna:
        --------
        Tuple[Dict[int, Dict[int, List[Tuple[float, int, float, bool]]]], Dict[int, Dict[int, float]]]
            - P: Diccionario de transiciones {estado: {accion: [(prob, prox_estado, recompensa, done)]}}.
                 En este MDP simplificado, prox_estado es el mismo estado y done es True.
            - R: Diccionario de recompensas {estado: {accion: recompensa_promedio}}.
        """
        if self.pi_agent is None:
            raise RuntimeError("Agente PolicyIteration no inicializado.")

        n_samples = x_cgm.shape[0]
        n_states = self._n_states
        n_actions = self._n_actions

        # Diccionarios para almacenar recompensas y conteos por (estado, acción)
        # Usamos defaultdict para simplificar la adición de recompensas
        from collections import defaultdict
        state_action_rewards: Dict[Tuple[int, int], List[float]] = defaultdict(list)

        if verbose > 0:
            print_info("Discretizando estados y calculando recompensas...")

        # 1. Discretizar estados y calcular recompensas para cada muestra
        # No necesitamos guardar discretized_states aquí
        for i in tqdm(range(n_samples), desc="Discretizando Estados", disable=verbose == 0):
             # Asegurarse de pasar x_other[i] si existe y es necesario
            current_x_other = x_other[i] if x_other.shape[1] > 0 else None
            state = self._discretize_state(x_cgm[i], current_x_other)
            target_dose = y[i, 0] if y.ndim > 1 else y[i] # Asegurar que y sea escalar

            # Calcular recompensa para cada acción posible en este estado
            for action in range(n_actions):
                predicted_dose = self._convert_action_to_dose(action)
                # Recompensa: negativa del error absoluto (más cerca de 0 es mejor)
                # Podríamos usar otras funciones de recompensa, e.g., cuadrática
                reward = -abs(predicted_dose - target_dose)

                pair = (state, action)
                state_action_rewards[pair].append(reward)

        # 2. Calcular recompensas promedio y construir P y R
        # Usar defaultdict para P y R también simplifica
        P: Dict[int, Dict[int, List[Tuple[float, int, float, bool]]]] = defaultdict(dict)
        R: Dict[int, Dict[int, float]] = defaultdict(dict)

        if verbose > 0:
            print_info("Calculando recompensas promedio y construyendo P y R...")

        # Recompensa por defecto si un par (s, a) no se observó en los datos
        # Podría ser la recompensa mínima observada, o una penalización grande
        all_rewards = [r for rewards_list in state_action_rewards.values() for r in rewards_list]
        default_reward = min(all_rewards) if all_rewards else -self.max_dose * 2 # Penalización grande si no hay datos

        # Iterar sobre los pares (estado, acción) observados
        observed_pairs = list(state_action_rewards.keys())
        for s, a in tqdm(observed_pairs, desc="Construyendo MDP", disable=verbose == 0):
            rewards = state_action_rewards[(s, a)]
            avg_reward = np.mean(rewards)

            # En este MDP simplificado, la acción lleva a una recompensa inmediata
            # y el episodio termina (done=True). El estado no cambia realmente (next_s = s).
            # P[estado][accion] = [(probabilidad, proximo_estado, recompensa, done)]
            # La recompensa en P no se usa directamente por PI clásico si R se proporciona.
            P[s][a] = [(TRANSITION_PROB, s, avg_reward, True)]
            R[s][a] = avg_reward

        # Opcional: Llenar pares (s, a) no observados con recompensa por defecto
        # Esto asegura que P y R cubran todos los estados/acciones posibles,
        # aunque puede no ser estrictamente necesario si PI maneja estados/acciones faltantes.
        # for s in range(n_states):
        #     for a in range(n_actions):
        #         if a not in P[s]: # Si la acción 'a' no fue vista para el estado 's'
        #             P[s][a] = [(TRANSITION_PROB, s, default_reward, True)]
        #             R[s][a] = default_reward

        # Convertir defaultdicts a dicts normales si es necesario por la interfaz externa
        return dict(P), dict(R)


    def train(
        self,
        x_cgm: np.ndarray,
        x_other: np.ndarray,
        y: np.ndarray,
        validation_data: Optional[Tuple[Tuple[np.ndarray, np.ndarray], np.ndarray]] = None,
        epochs: int = 1, # Policy Iteration no usa épocas tradicionalmente
        batch_size: int = 32, # No aplica a PI clásico
        callbacks: Optional[List] = None,
        verbose: int = 1
    ) -> Dict[str, List[float]]:
        """
        Entrena el modelo de Policy Iteration en los datos proporcionados.
        
        Parámetros:
        -----------
        x_cgm : np.ndarray
            Datos CGM de entrenamiento
        x_other : np.ndarray
            Otras características de entrenamiento
        y : np.ndarray
            Valores objetivo (dosis de insulina)
        validation_data : Optional[Tuple[Tuple[np.ndarray, np.ndarray], np.ndarray]], opcional
            Datos de validación como ((x_cgm_val, x_other_val), y_val) (default: None)
        epochs : int, opcional
            Número de épocas (no usado directamente por PI) (default: 1)
        batch_size : int, opcional
            Tamaño de lote (no usado directamente por PI) (default: 32)
        callbacks : Optional[List], opcional
            Lista de callbacks (no implementado) (default: None)
        verbose : int, opcional
            Nivel de verbosidad (0: silencioso, 1: información) (default: 1)
            
        Retorna:
        --------
        Dict[str, List[float]]
            Historial de métricas durante el entrenamiento
        """
        # Iniciar el agente si no se ha hecho
        if self.pi_agent is None:
            self.start(x_cgm, x_other, y)
        
        # Construir el modelo MDP a partir de los datos
        P, R = self._build_mdp_model(x_cgm, x_other, y, verbose)
        
        if verbose:
            print_info(f"Modelo MDP construido con {len(P)} estados y {self._n_actions} acciones.")
            print_info(f"Iniciando entrenamiento de Iteración de Política...")
        
        # Entrenamiento principal - Iteración de Política
        policy, values = self.pi_agent.train(P, R)
        self.is_trained = True
        
        # Evaluar en los datos de entrenamiento
        train_preds = self.predict(x_cgm, x_other)  # CORREGIDO: separar argumentos
        train_loss = np.mean((train_preds.flatten() - y.flatten()) ** 2)
        self.history[CONST_LOSS].append(train_loss)
        
        if verbose:
            print_info(f"Pérdida (MSE) en entrenamiento: {train_loss:.4f}")
        
        # Validación si se proporcionan datos
        if validation_data is not None:
            (x_cgm_val, x_other_val), y_val = validation_data
            val_preds = self.predict(x_cgm_val, x_other_val)  # CORREGIDO: separar argumentos
            val_loss = np.mean((val_preds.flatten() - y_val.flatten()) ** 2)
            self.history[CONST_VAL_LOSS].append(val_loss)
            
            if verbose:
                print_info(f"Pérdida (MSE) en validación: {val_loss:.4f}")
        
        # Opcional: Graficar métricas internas si existen
        if hasattr(self.pi_agent, 'plot_metrics'):
            self.pi_agent.plot_metrics(save_plots=True)
        
        return self.history

    def evaluate(self, x_cgm: np.ndarray, x_other: np.ndarray, y: np.ndarray) -> Dict[str, float]:
        """
        Evalúa el rendimiento del modelo entrenado en datos de prueba.

        Parámetros:
        -----------
        x_cgm : np.ndarray
            Datos CGM de prueba.
        x_other : np.ndarray
            Otras características de prueba.
        y : np.ndarray
            Valores objetivo reales.

        Retorna:
        --------
        Dict[str, float]
            Diccionario con métricas de evaluación (MSE, MAE, RMSE, R2).
        """
        if not self.is_trained:
            raise RuntimeError("El modelo debe ser entrenado antes de evaluar.")

        predictions = self.predict([x_cgm, x_other])
        y_true = y.flatten()
        y_pred = predictions.flatten()

        mse = np.mean((y_true - y_pred)**2)
        mae = np.mean(np.abs(y_true - y_pred))
        rmse = np.sqrt(mse)
        # Calcular R2 score
        ss_res = np.sum((y_true - y_pred)**2)
        ss_tot = np.sum((y_true - np.mean(y_true))**2)
        r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0 # Evitar división por cero

        return {
            "mse": float(mse),
            "mae": float(mae),
            "rmse": float(rmse),
            "r2": float(r2)
        }

    def save(self, filepath: str) -> None:
        """
        Guarda el estado del agente PolicyIteration entrenado.

        Parámetros:
        -----------
        filepath : str
            Ruta base para guardar los archivos del modelo (sin extensión).
        """
        if not self.is_trained or self.pi_agent is None:
            print_warning("Intento de guardar un modelo PolicyIteration no entrenado.")
            return

        # Guardar la política (policy) y la función de valor (v)
        model_data = {
            'policy': np.array(self.pi_agent.policy), # Convertir a NumPy para guardar
            'v': np.array(self.pi_agent.v),
            'config': { # Guardar configuración de discretización y PI
                'cgm_bins': self.cgm_bins, 'cgm_range': self.cgm_range,
                'time_bins': self.time_bins, 'insulin_bins': self.insulin_bins,
                'insulin_range': self.insulin_range, 'action_bins': self.action_bins,
                'max_dose': self.max_dose, 'n_states': self._n_states,
                'pi_config': self.pi_kwargs
            },
            'history': self.history
        }
        # Usar joblib para guardar eficientemente arrays NumPy
        save_path = f"{filepath}_pi_agent.joblib"
        joblib.dump(model_data, save_path)
        print_info(f"Modelo PolicyIteration guardado en {save_path}")

    def load(self, filepath: str) -> None:
        """
        Carga el estado de un agente PolicyIteration entrenado.

        Parámetros:
        -----------
        filepath : str
            Ruta base desde donde cargar los archivos del modelo (sin extensión).
        """
        load_path = f"{filepath}_pi_agent.joblib"
        try:
            model_data = joblib.load(load_path)

            # Restaurar configuración
            config = model_data.get('config', {})
            self.cgm_bins = config.get('cgm_bins', self.cgm_bins)
            self.cgm_range = tuple(config.get('cgm_range', self.cgm_range))
            self.time_bins = config.get('time_bins', self.time_bins)
            self.insulin_bins = config.get('insulin_bins', self.insulin_bins)
            self.insulin_range = tuple(config.get('insulin_range', self.insulin_range))
            self.action_bins = config.get('action_bins', self.action_bins)
            self.max_dose = config.get('max_dose', self.max_dose)
            self._n_states = config.get('n_states', self._n_states)
            self._n_actions = self.action_bins
            self.pi_kwargs = config.get('pi_config', self.pi_kwargs)

            # Recalcular dependencias de configuración
            self._calculated_max_states = self.cgm_bins * self.time_bins * self.insulin_bins
            self._cgm_edges = np.linspace(self.cgm_range[0], self.cgm_range[1], self.cgm_bins + 1)
            self._time_edges = np.linspace(0, 24, self.time_bins + 1)
            self._insulin_edges = np.linspace(self.insulin_range[0], self.insulin_range[1], self.insulin_bins + 1)
            self._action_doses = np.linspace(0, self.max_dose, self.action_bins)


            # Recrear el agente PI con la configuración cargada
            accepted_args = ['gamma', 'theta', 'max_iterations', 'max_iterations_eval', 'seed']
            filtered_pi_kwargs = {k: v for k, v in self.pi_kwargs.items() if k in accepted_args}
            self.pi_agent = PolicyIteration(
                n_states=self._n_states,
                n_actions=self._n_actions,
                **filtered_pi_kwargs
            )

            # Restaurar estado del agente
            self.pi_agent.policy = jnp.array(model_data['policy'])
            self.pi_agent.v = jnp.array(model_data['v'])
            self.history = model_data.get('history', self.history)
            self.is_trained = True
            print_info(f"Modelo PolicyIteration cargado desde {load_path}")

        except FileNotFoundError:
            raise FileNotFoundError(f"No se encontró el archivo del modelo PolicyIteration en {load_path}")
        except Exception as e:
            raise IOError(f"Error al cargar el modelo PolicyIteration desde {load_path}: {e}")


def create_policy_iteration_model(cgm_shape: Tuple[int, ...], other_features_shape: Tuple[int, ...], **kwargs) -> PolicyIterationWrapper:
    """
    Función de fábrica para crear un PolicyIterationWrapper.

    Parámetros:
    -----------
    cgm_shape : Tuple[int, ...]
        Forma de los datos CGM (pasos_temporales, características). No usado directamente por PI, pero parte de la API.
    other_features_shape : Tuple[int, ...]
        Forma de otras características (características). No usado directamente por PI, pero parte de la API.
    **kwargs
        Argumentos para pasar al constructor de PolicyIterationWrapper
        (incluyendo configuración de discretización y parámetros de PI).

    Retorna:
    --------
    PolicyIterationWrapper
        Instancia del wrapper de Policy Iteration.
    """
    # Separar kwargs para el wrapper y para PolicyIteration
    wrapper_args_names = ['bins', 'state_bounds', 'action_bins', 'action_bounds', 'n_states']
    pi_args_names = ['gamma', 'theta', 'max_iterations', 'max_iterations_eval', 'seed']

    # Crear diccionario con todos los argumentos, priorizando los específicos del wrapper si hay colisión (poco probable)
    all_wrapper_kwargs = {k: v for k, v in kwargs.items()}

    # Extraer los argumentos específicos del agente PI para logging, pero se pasarán todos a través de **kwargs
    pi_kwargs_for_logging = {k: v for k, v in kwargs.items() if k in pi_args_names}
    wrapper_kwargs_for_logging = {k: v for k, v in kwargs.items() if k in wrapper_args_names}


    # cgm_shape y other_features_shape no se usan directamente en el __init__ actual,
    # pero se aceptan para cumplir con la API. Podrían ser útiles si la discretización
    # dependiera de las dimensiones exactas.
    print_info(f"Creando PolicyIterationWrapper con formas: CGM={cgm_shape}, Other={other_features_shape}")
    # Loggear los argumentos que se pasarán al wrapper (incluye los del agente)
    print_info(f"  Wrapper args (incluye PI args): {all_wrapper_kwargs}")
    # Loggear específicamente los que se espera que use el agente PI interno
    print_info(f"  Expected PI Agent args: {pi_kwargs_for_logging}")


    # Pasar todos los kwargs al wrapper. El __init__ del wrapper separará los que necesita
    # y pasará el resto (recogidos por **pi_kwargs) al agente PolicyIteration.
    return PolicyIterationWrapper(**all_wrapper_kwargs)

def model_creator() -> Callable[..., PolicyIterationWrapper]:
    """
    Retorna la función de fábrica `create_policy_iteration_model`.
    Compatible con la API esperada por `run.py`.

    Retorna:
    --------
    Callable[..., PolicyIterationWrapper]
        La función `create_policy_iteration_model`.
    """
    # Esta función simplemente devuelve la función de creación real.
    # run.py llamará a esta función devuelta con cgm_shape, other_features_shape y los kwargs necesarios.
    return create_policy_iteration_model