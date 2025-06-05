import os
import sys
import numpy as np
import torch
import torch.nn as nn
import time
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Optional, Union, Any, Callable
from types import SimpleNamespace

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../.."))
sys.path.append(PROJECT_ROOT)

from config.models_config_old import VALUE_ITERATION_CONFIG, EARLY_STOPPING_POLICY
from constants.constants import CONST_DEFAULT_SEED, CONST_DEFAULT_EPOCHS, CONST_DEFAULT_BATCH_SIZE
from custom.ReinforcementLearning.rl_pt import RLModelWrapperPyTorch
from custom.printer import print_success, print_warning, print_error, print_info

# Constantes para rutas y mensajes comunes
CONST_FIGURES_DIR = os.path.join(PROJECT_ROOT, "figures", "pytorch", "value_iteration")
CONST_PESO_ARCHIVO_SUFIJO = "_weights.pth"
CONST_AGENTE_ARCHIVO_SUFIJO = "_agent.pkl"
CONST_ITERACION = "Iteración"
CONST_VALOR = "Valor"
CONST_DELTA = "Delta"
CONST_TIEMPO = "Tiempo (segundos)"
CONST_POLITICA = "Política"
CONST_PROBABILIDAD = 1.0

# Crear directorio para figuras si no existe
os.makedirs(CONST_FIGURES_DIR, exist_ok=True)

class ValueIteration:
    """
    Implementación del algoritmo de Iteración de Valor (Value Iteration) con PyTorch.
    
    La Iteración de Valor es un método de programación dinámica que encuentra la política
    óptima calculando directamente la función de valor óptima utilizando la ecuación de
    optimalidad de Bellman, sin mantener explícitamente una política durante el proceso.
    """
    
    def __init__(
        self,
        n_states: int,
        n_actions: int,
        gamma: float = VALUE_ITERATION_CONFIG['gamma'],
        theta: float = VALUE_ITERATION_CONFIG['theta'],
        max_iterations: int = VALUE_ITERATION_CONFIG['max_iterations']
    ) -> None:
        """
        Inicializa el agente de Iteración de Valor.
        
        Parámetros:
        -----------
        n_states : int
            Número de estados en el entorno
        n_actions : int
            Número de acciones en el entorno
        gamma : float, opcional
            Factor de descuento (default: VALUE_ITERATION_CONFIG['gamma'])
        theta : float, opcional
            Umbral para convergencia (default: VALUE_ITERATION_CONFIG['theta'])
        max_iterations : int, opcional
            Número máximo de iteraciones (default: VALUE_ITERATION_CONFIG['max_iterations'])
        """
        self.n_states = n_states
        self.n_actions = n_actions
        self.gamma = gamma
        self.theta = theta
        self.max_iterations = max_iterations
        
        # Establecer dispositivo para cálculos en GPU si está disponible
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Inicializar función de valor y política
        self.v_table = torch.zeros(n_states, device=self.device)
        self.policy = torch.zeros((n_states, n_actions), device=self.device)
        
        # Para seguimiento de métricas
        self.value_changes = []
        self.iteration_times = []
        
        # Configurar generador de números aleatorios
        self.rng = np.random.Generator(np.random.PCG64(CONST_DEFAULT_SEED))
    
    def _calculate_action_values(self, state: int, transitions: Dict[int, Dict[int, List]]) -> torch.Tensor:
        """
        Calcula los valores Q para todas las acciones en un estado dado.
        
        Parámetros:
        -----------
        state : int
            Estado para el cual calcular los valores
        transitions : Dict[int, Dict[int, List]]
            Diccionario con las transiciones del entorno
            
        Retorna:
        --------
        torch.Tensor
            Tensor con los valores Q para cada acción
        """
        action_values = torch.zeros(self.n_actions, device=self.device)
        
        for a in range(self.n_actions):
            if state in transitions and a in transitions[state]:
                for prob, next_s, reward, done in transitions[state][a]:
                    # Aplicar ecuación de Bellman
                    not_done = 0.0 if done else 1.0
                    action_values[a] += prob * (reward + self.gamma * self.v_table[next_s] * not_done)
        
        return action_values
    
    def value_update(self, env: Any) -> float:
        """
        Actualiza la función de valor usando la ecuación de Bellman.
        
        Parámetros:
        -----------
        env : Any
            Entorno con las dinámicas de transición
            
        Retorna:
        --------
        float
            Delta máximo (cambio máximo en la función de valor)
        """
        delta = 0.0
        
        for s in range(self.n_states):
            if hasattr(env, 'P') and s in env.P:
                v_old = self.v_table[s].item()
                
                # Calcular valores para cada acción y seleccionar máximo
                action_values = self._calculate_action_values(s, env.P)
                max_value = torch.max(action_values).item()
                
                # Actualizar valor del estado
                self.v_table[s] = max_value
                
                # Actualizar delta
                delta = max(delta, abs(v_old - max_value))
        
        return delta
    
    def extract_policy(self, env: Any) -> torch.Tensor:
        """
        Extrae la política óptima a partir de la función de valor.
        
        Parámetros:
        -----------
        env : Any
            Entorno con las dinámicas de transición
            
        Retorna:
        --------
        torch.Tensor
            Política óptima (determinística)
        """
        policy = torch.zeros((self.n_states, self.n_actions), device=self.device)
        
        for s in range(self.n_states):
            if hasattr(env, 'P') and s in env.P:
                # Calcular valores para cada acción
                action_values = self._calculate_action_values(s, env.P)
                
                # Encontrar la(s) mejor(es) acción(es)
                best_actions = torch.where(action_values == torch.max(action_values))[0]
                
                # Resolver empates de forma determinista
                if len(best_actions) > 1:
                    # Usar una función hash para resolver empates de manera determinista pero variable
                    hash_value = hash(str(s)) % len(best_actions)
                    best_action = best_actions[hash_value].item()
                else:
                    best_action = best_actions[0].item()
                
                # Asignar probabilidad 1 a la mejor acción
                policy[s, best_action] = 1.0
        
        return policy
    
    def _check_early_stopping(self, delta: float, best_delta: float, no_improvement_count: int, 
                             patience: int, min_delta: float) -> Tuple[float, int, bool]:
        """
        Verifica si se debe activar el early stopping.
        
        Parámetros:
        -----------
        delta : float
            Delta actual (cambio en la función de valor)
        best_delta : float
            Mejor delta encontrado hasta ahora
        no_improvement_count : int
            Contador de iteraciones sin mejora
        patience : int
            Número máximo de iteraciones sin mejora antes de parar
        min_delta : float
            Cambio mínimo considerado como mejora
            
        Retorna:
        --------
        Tuple[float, int, bool]
            Nueva mejor delta, nuevo contador sin mejora, bandera de parada
        """
        should_stop = False
        new_best_delta = best_delta
        new_count = no_improvement_count
        
        if delta < best_delta - min_delta:
            new_best_delta = delta
            new_count = 0
        else:
            new_count += 1
            
        if new_count >= patience:
            should_stop = True
            
        return new_best_delta, new_count, should_stop
    
    def _check_convergence(self, delta: float, start_time: float) -> bool:
        """
        Verifica si el algoritmo ha convergido o si se ha excedido el tiempo.
        
        Parámetros:
        -----------
        delta : float
            Cambio en la función de valor
        start_time : float
            Tiempo de inicio del entrenamiento
            
        Retorna:
        --------
        bool
            True si debe detenerse, False en caso contrario
        """
        # Verificar convergencia
        if delta < self.theta:
            print_success(f"¡Convergencia alcanzada! Delta={delta:.6f}")
            return True
        
        # Límite de tiempo (30 segundos)
        if time.time() - start_time > 30:
            print_warning("Tiempo límite excedido. Deteniendo entrenamiento.")
            return True
            
        return False
    
    def train(self, env: Any) -> Dict[str, List]:
        """
        Entrena al agente usando iteración de valor.
        
        Parámetros:
        -----------
        env : Any
            Entorno con las dinámicas de transición
            
        Retorna:
        --------
        Dict[str, List]
            Diccionario con historial de entrenamiento
        """
        print("Iniciando iteración de valor con espacio de estados reducido...")
        
        iterations = 0
        start_time = time.time()
        self.value_changes = []
        self.iteration_times = []
        
        # Configurar early stopping
        patience = EARLY_STOPPING_POLICY.get('early_stopping_patience', 5)
        min_delta = EARLY_STOPPING_POLICY.get('early_stopping_min_delta', 0.001)
        use_early_stopping = EARLY_STOPPING_POLICY.get('early_stopping', True)
        
        # Contador para early stopping
        no_improvement_count = 0
        best_delta = float('inf')
        
        # Limitar número de iteraciones para entrenamientos iniciales
        actual_max_iterations = min(self.max_iterations, 50)
        print(f"Limitando a {actual_max_iterations} iteraciones máximas")
        
        for i in range(actual_max_iterations):
            iteration_start = time.time()
            
            # Actualizar función de valor
            delta = self.value_update(env)
            
            # Registrar cambio de valor y tiempo
            self.value_changes.append(float(delta))
            iteration_time = time.time() - iteration_start
            self.iteration_times.append(iteration_time)
            
            iterations = i + 1
            
            # Mostrar progreso
            if i % 10 == 0 or i < 5:
                print(f"Iteración {iterations}: Delta = {delta:.6f}, Tiempo = {iteration_time:.2f} segundos")
            
            # Verificar early stopping
            if use_early_stopping:
                best_delta, no_improvement_count, should_stop = self._check_early_stopping(
                    delta, best_delta, no_improvement_count, patience, min_delta)
                
                if should_stop:
                    print_success(f"¡Early stopping activado después de {iterations} iteraciones!")
                    break
            
            # Verificar convergencia o tiempo límite
            if self._check_convergence(delta, start_time):
                break
                
        # Extraer política óptima
        self.policy = self.extract_policy(env)
        
        total_time = time.time() - start_time
        print(f"Iteración de valor completada en {iterations} iteraciones, {total_time:.2f} segundos")
        
        return {
            'iterations': iterations,
            'value_changes': self.value_changes,
            'iteration_times': self.iteration_times,
            'total_time': total_time
        }
    
    def get_action(self, state: int) -> int:
        """
        Devuelve la mejor acción para un estado según la política actual.
        
        Parámetros:
        -----------
        state : int
            Estado actual
            
        Retorna:
        --------
        int
            Mejor acción
        """
        return torch.argmax(self.policy[state]).item()
    
    def get_value(self, state: int) -> float:
        """
        Devuelve el valor de un estado según la función de valor actual.
        
        Parámetros:
        -----------
        state : int
            Estado para obtener su valor
            
        Retorna:
        --------
        float
            Valor del estado
        """
        return self.v_table[state].item()
    
    def evaluate(self, env: Any, max_steps: int = 100, episodes: int = 10) -> float:
        """
        Evalúa la política actual en el entorno.
        
        Parámetros:
        -----------
        env : Any
            Entorno para evaluar
        max_steps : int, opcional
            Pasos máximos por episodio (default: 100)
        episodes : int, opcional
            Número de episodios para evaluar (default: 10)
            
        Retorna:
        --------
        float
            Recompensa promedio en los episodios
        """
        total_rewards = []
        episode_lengths = []
        
        for ep in range(episodes):
            state, _ = env.reset()
            done = False
            total_reward = 0.0
            steps = 0
            
            while not done and steps < max_steps:
                action = self.get_action(state)
                next_state, reward, terminated, truncated, _ = env.step(action)
                done = terminated or truncated
                
                total_reward += reward
                state = next_state
                steps += 1
            
            total_rewards.append(total_reward)
            episode_lengths.append(steps)
            print(f"Episodio {ep+1}: Recompensa = {total_reward:.2f}, Pasos = {steps}")
        
        avg_reward = np.mean(total_rewards)
        avg_length = np.mean(episode_lengths)
        print(f"Evaluación: recompensa media en {episodes} episodios = {avg_reward:.2f}, "
              f"longitud media = {avg_length:.2f}")
        
        return avg_reward
    
    def save(self, filepath: str) -> None:
        """
        Guarda la política y función de valor en un archivo.
        
        Parámetros:
        -----------
        filepath : str
            Ruta donde guardar el modelo
        """
        data = {
            'policy': self.policy.cpu().numpy(),
            'v_table': self.v_table.cpu().numpy(),
            'n_states': self.n_states,
            'n_actions': self.n_actions,
            'gamma': self.gamma
        }
        
        torch.save(data, filepath)
        print(f"Modelo guardado en {filepath}")
    
    def load(self, filepath: str) -> None:
        """
        Carga la política y función de valor desde un archivo.
        
        Parámetros:
        -----------
        filepath : str
            Ruta desde donde cargar el modelo
        """
        data = torch.load(filepath, map_location=self.device)
        
        self.policy = torch.tensor(data['policy'], device=self.device)
        self.v_table = torch.tensor(data['v_table'], device=self.device)
        self.n_states = data['n_states']
        self.n_actions = data['n_actions']
        self.gamma = data['gamma']
        
        print(f"Modelo cargado desde {filepath}")
    
    def _setup_grid_plot(self, ax: plt.Axes, grid_shape: Tuple[int, int]) -> None:
        """
        Configura el gráfico base para visualización de cuadrícula.
        
        Parámetros:
        -----------
        ax : plt.Axes
            Ejes para dibujar
        grid_shape : Tuple[int, int]
            Forma de la cuadrícula (filas, columnas)
        """
        # Configurar límites
        ax.set_xlim([0, grid_shape[1]])
        ax.set_ylim([0, grid_shape[0]])
        
        # Dibujar líneas de cuadrícula
        for i in range(grid_shape[1] + 1):
            ax.axvline(i, color='black', linestyle='-')
        for j in range(grid_shape[0] + 1):
            ax.axhline(j, color='black', linestyle='-')
    
    def _draw_policy_arrows(self, ax: plt.Axes, env: Any, grid_shape: Tuple[int, int], directions: Dict[int, Tuple[float, float]]) -> None:
        """
        Dibuja flechas representando la política en la cuadrícula.
        
        Parámetros:
        -----------
        ax : plt.Axes
            Ejes para dibujar
        env : Any
            Entorno con estructura de cuadrícula
        grid_shape : Tuple[int, int]
            Forma de la cuadrícula (filas, columnas)
        directions : Dict[int, Tuple[float, float]]
            Diccionario que mapea acciones a direcciones para flechas
        """
        for s in range(self.n_states):
            # Obtener coordenadas del estado
            if hasattr(env, 'state_mapping'):
                i, j = env.state_mapping(s)
            else:
                # Asumir ordenamiento row-major
                i, j = s // grid_shape[1], s % grid_shape[1]
            
            # Mostrar valores de estado
            value = self.get_value(s)
            ax.text(j + 0.5, grid_shape[0] - i - 0.5, f"{value:.2f}", 
                   ha='center', va='center', color='red', fontsize=9)
            
            # Verificar si es estado terminal
            if s in env.P:
                is_terminal = all(len(env.P[s][a]) > 0 and env.P[s][a][0][3] for a in range(self.n_actions))
                if is_terminal:
                    continue
                
                action = self.get_action(s)
                
                if action in directions:
                    dx, dy = directions[action]
                    ax.arrow(j + 0.5, grid_shape[0] - i - 0.5, dx, dy, 
                             head_width=0.1, head_length=0.1, fc='blue', ec='blue')
    
    def visualize_policy(self, env: Any, title: str = "Política Óptima") -> None:
        """
        Visualiza la política para entornos de tipo cuadrícula.
        
        Parámetros:
        -----------
        env : Any
            Entorno con estructura de cuadrícula
        title : str, opcional
            Título para la visualización (default: "Política Óptima")
        """
        if not hasattr(env, 'shape'):
            print_error("El entorno no tiene estructura de cuadrícula para visualización")
            return
            
        grid_shape = env.shape
        _, ax = plt.subplots(figsize=(8, 8))
        
        # Configurar la cuadrícula base
        self._setup_grid_plot(ax, grid_shape)
        
        # Definir direcciones para flechas
        directions = {
            0: (0, -0.4),  # Izquierda
            1: (0, 0.4),   # Derecha
            2: (-0.4, 0),  # Abajo
            3: (0.4, 0)    # Arriba
        }
        
        # Dibujar flechas para acciones
        self._draw_policy_arrows(ax, env, grid_shape, directions)
        
        ax.set_title(title)
        plt.tight_layout()
        
        # Guardar figura
        plt.savefig(os.path.join(CONST_FIGURES_DIR, f"politica_{title.lower().replace(' ', '_')}.png"), dpi=300)
        plt.show()
    
    def visualize_value_function(self, env: Any, title: str = "Función de Valor") -> None:
        """
        Visualiza la función de valor para entornos de tipo cuadrícula.
        
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
                # Asumir ordenamiento row-major
                i, j = s // grid_shape[1], s % grid_shape[1]
            
            value_grid[i, j] = self.get_value(s)
        
        _, ax = plt.subplots(figsize=(10, 8))
        
        # Crear mapa de calor
        im = ax.imshow(value_grid, cmap='viridis')
        
        # Añadir barra de color
        plt.colorbar(im, ax=ax, label=CONST_VALOR)
        
        # Mostrar valores en cada celda
        for i in range(grid_shape[0]):
            for j in range(grid_shape[1]):
                ax.text(j, i, f"{value_grid[i, j]:.2f}", ha='center', va='center',
                        color='white' if value_grid[i, j] < np.max(value_grid)/1.5 else 'black')
        
        ax.set_title(title)
        plt.tight_layout()
        
        # Guardar figura
        plt.savefig(os.path.join(CONST_FIGURES_DIR, f"funcion_valor_{title.lower().replace(' ', '_')}.png"), dpi=300)
        plt.show()
    
    def visualize_training(self, history: Dict[str, List]) -> None:
        """
        Visualiza las métricas de entrenamiento.
        
        Parámetros:
        -----------
        history : Dict[str, List]
            Diccionario con historial de entrenamiento
        """
        _, axs = plt.subplots(2, 1, figsize=(12, 8))
        
        # Gráfico de cambios en la función de valor (delta)
        axs[0].plot(range(1, len(history['value_changes']) + 1), 
                   history['value_changes'])
        axs[0].set_title(f'Cambios en la Función de {CONST_VALOR} ({CONST_DELTA})')
        axs[0].set_xlabel(CONST_ITERACION)
        axs[0].set_ylabel(CONST_DELTA)
        axs[0].set_yscale('log')  # Escala logarítmica para ver mejor la convergencia
        axs[0].grid(True)
        
        # Gráfico de tiempos de iteración
        axs[1].plot(range(1, len(history['iteration_times']) + 1), 
                   history['iteration_times'])
        axs[1].set_title(f'Tiempos de {CONST_ITERACION}')
        axs[1].set_xlabel(CONST_ITERACION)
        axs[1].set_ylabel(CONST_TIEMPO)
        axs[1].grid(True)
        
        plt.tight_layout()
        
        # Guardar figura
        plt.savefig(os.path.join(CONST_FIGURES_DIR, "entrenamiento_resumen.png"), dpi=300)
        plt.show()


class ValueIterationModel(nn.Module):
    """
    Modelo PyTorch para Value Iteration.
    
    Encapsula un agente Value Iteration para utilizarlo con la interfaz de PyTorch.
    """
    
    def __init__(
        self, 
        value_iteration_agent: ValueIteration,
        cgm_shape: Tuple[int, ...],
        other_features_shape: Tuple[int, ...],
        discretizer: Optional[Any] = None
    ) -> None:
        """
        Inicializa el modelo de Value Iteration.
        
        Parámetros:
        -----------
        value_iteration_agent : ValueIteration
            Agente de Value Iteration para el algoritmo
        cgm_shape : Tuple[int, ...]
            Forma de los datos CGM
        other_features_shape : Tuple[int, ...]
            Forma de otras características
        discretizer : Optional[Any], opcional
            Discretizador de estados personalizado (default: None)
        """
        super(ValueIterationModel, self).__init__()
        self.value_iteration_agent = value_iteration_agent
        self.cgm_shape = cgm_shape
        self.other_features_shape = other_features_shape
        self.discretizer = discretizer
        
        # Registrar device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Capas para procesamiento de entrada
        self.cgm_encoder = nn.Sequential(
            nn.Flatten(),
            nn.Linear(np.prod(cgm_shape), 64),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        
        self.other_encoder = nn.Sequential(
            nn.Flatten(),
            nn.Linear(np.prod(other_features_shape), 32),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        
        # Capa para generar representación de estado combinada
        self.combined_encoder = nn.Sequential(
            nn.Linear(64 + 32, 64),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        
        # Capa para predecir dosis directamente desde la representación de estado
        # Esta se usará en modo de entrenamiento diferenciable
        self.policy_decoder = nn.Linear(64, 1)
        
        # Capa para convertir acciones discretas a dosis continuas
        # Esta se usará en modo de inferencia no diferenciable
        self.action_decoder = nn.Linear(1, 1)
        
        # Inicializar los pesos de los decoders
        self._init_decoders()
        
        # Para crear entorno sintético para entrenamiento
        self.env = None
        
        # Métricas para seguimiento
        self.training_metrics = {}
    
    def _init_decoders(self) -> None:
        """
        Inicializa los pesos de los decoders.
        """
        # Inicializar decoder de política (para entrenamiento diferenciable)
        nn.init.xavier_uniform_(self.policy_decoder.weight)
        nn.init.zeros_(self.policy_decoder.bias)
        
        # Inicializar decoder de acción (para inferencia no diferenciable)
        n_actions = self.value_iteration_agent.n_actions
        max_dose = 15.0
        
        # Configurar la capa para mapear [0, n_actions-1] a [0, max_dose]
        with torch.no_grad():
            self.action_decoder.weight.fill_(max_dose / (n_actions - 1))
            self.action_decoder.bias.fill_(0.0)
    
    def _encode_states(self, cgm_data: torch.Tensor, other_features: torch.Tensor) -> torch.Tensor:
        """
        Codifica las entradas en una representación de estados para Value Iteration.
        
        Parámetros:
        -----------
        cgm_data : torch.Tensor
            Datos CGM de entrada
        other_features : torch.Tensor
            Otras características de entrada
            
        Retorna:
        --------
        torch.Tensor
            Representación de estados
        """
        # Procesar mediante encoders
        batch_size = cgm_data.size(0)
        cgm_encoded = self.cgm_encoder(cgm_data.reshape(batch_size, -1))
        other_encoded = self.other_encoder(other_features.reshape(batch_size, -1))
        
        # Concatenar características
        combined = torch.cat([cgm_encoded, other_encoded], dim=1)
        
        # Generar representación final
        return self.combined_encoder(combined)
    
    def forward(self, cgm_data: torch.Tensor, other_features: torch.Tensor) -> torch.Tensor:
        """
        Realiza una predicción usando el modelo Value Iteration.
        
        Parámetros:
        -----------
        cgm_data : torch.Tensor
            Datos CGM de entrada
        other_features : torch.Tensor
            Otras características de entrada
            
        Retorna:
        --------
        torch.Tensor
            Predicciones de dosis de insulina
        """
        batch_size = cgm_data.size(0)
        
        # Implementación diferenciable para entrenamiento
        if self.training:
            # Codificar entradas con la red neuronal
            cgm_encoded = self.cgm_encoder(cgm_data.reshape(batch_size, -1))
            other_encoded = self.other_encoder(other_features.reshape(batch_size, -1))
            
            # Concatenar características
            combined = torch.cat([cgm_encoded, other_encoded], dim=1)
            
            # Generar representación final diferenciable
            encoded_states = self.combined_encoder(combined)
            
            # Usar el policy_decoder para generar dosis directamente
            # Esta es la ruta diferenciable para entrenamiento
            predictions = self.policy_decoder(encoded_states)
            
            return predictions
        
        # Implementación basada en política para inferencia
        else:
            # Inicializar tensor de resultados
            predictions = torch.zeros(batch_size, 1, device=self.device)
            
            # Convertir a array NumPy para procesamiento (seguro porque estamos en modo eval)
            cgm_np = cgm_data.detach().cpu().numpy()
            other_np = other_features.detach().cpu().numpy()
            
            # Procesar cada muestra
            for i in range(batch_size):
                # Discretizar estado
                state_idx = self._discretize_state(cgm_np[i], other_np[i])
                
                # Obtener acción según la política actual
                action = self.value_iteration_agent.get_action(state_idx)
                
                # Convertir a tensor para usar el decoder
                action_tensor = torch.tensor([[float(action)]], device=self.device)
                
                # Decodificar a dosis continua
                with torch.no_grad():  # Explícitamente sin gradientes para inferencia
                    dose = self.action_decoder(action_tensor).item()
                    predictions[i, 0] = dose
            
            return predictions
    
    def _discretize_state(self, cgm_data: np.ndarray, other_features: np.ndarray) -> int:
        """
        Discretiza un estado continuo para Value Iteration.
        
        Parámetros:
        -----------
        cgm_data : np.ndarray
            Datos CGM para una muestra
        other_features : np.ndarray
            Otras características para una muestra
            
        Retorna:
        --------
        int
            Índice de estado discreto
        """
        if self.discretizer is not None:
            return self.discretizer(cgm_data, other_features)
        
        # Implementación por defecto
        cgm_flat = cgm_data.flatten()
        
        # Extraer características clave
        cgm_mean = np.mean(cgm_flat) if cgm_flat.size > 0 else 0.0
        cgm_last = cgm_flat[-1] if cgm_flat.size > 0 else 0.0
        cgm_trend = (cgm_flat[-1] - cgm_flat[0]) if cgm_flat.size > 1 else 0.0
        
        # Normalizar valores
        max_cgm = 400.0
        cgm_mean_norm = min(1.0, max(0.0, cgm_mean / max_cgm))
        cgm_last_norm = min(1.0, max(0.0, cgm_last / max_cgm))
        cgm_trend_norm = min(1.0, max(0.0, (cgm_trend + 100) / 200))
        
        # Discretizar usando bins
        n_bins = int(np.ceil(np.cbrt(self.value_iteration_agent.n_states)))
        
        mean_bin = min(int(cgm_mean_norm * n_bins), n_bins - 1)
        last_bin = min(int(cgm_last_norm * n_bins), n_bins - 1)
        trend_bin = min(int(cgm_trend_norm * n_bins), n_bins - 1)
        
        # Combinar en un solo índice
        state_idx = mean_bin * n_bins * n_bins + last_bin * n_bins + trend_bin
        
        # Asegurar que no exceda el número de estados
        return min(state_idx, self.value_iteration_agent.n_states - 1)
    
    def _update_action_decoder(self, y: np.ndarray) -> None:
        """
        Actualiza los pesos del decoder para mapear acciones a dosis basado en datos.
        
        Parámetros:
        -----------
        y : np.ndarray
            Valores objetivo (dosis de insulina)
        """
        min_dose = np.min(y)
        max_dose = np.max(y)
        dose_range = max_dose - min_dose
        
        n_actions = self.value_iteration_agent.n_actions
        
        # Actualizar pesos para mapear [0, n_actions-1] a [min_dose, max_dose]
        with torch.no_grad():
            self.action_decoder.weight.fill_(dose_range / (n_actions - 1))
            self.action_decoder.bias.fill_(min_dose)
            
            # También actualizar el policy_decoder para inicializar en un rango similar
            nn.init.normal_(self.policy_decoder.weight, mean=0.0, std=0.01)
            nn.init.constant_(self.policy_decoder.bias, min_dose + dose_range/2)
    
    def _create_training_environment(
        self, 
        cgm_data: np.ndarray, 
        other_features: np.ndarray, 
        targets: np.ndarray
    ) -> Any:
        """
        Crea un entorno de entrenamiento para Value Iteration.
        
        Parámetros:
        -----------
        cgm_data : np.ndarray
            Datos CGM
        other_features : np.ndarray
            Otras características
        targets : np.ndarray
            Valores objetivo (dosis de insulina)
            
        Retorna:
        --------
        Any
            Entorno compatible con Value Iteration
        """
        print("Creando entorno de entrenamiento para Value Iteration...")
        
        # Limitar el número de muestras para procesamiento más rápido
        max_samples = min(500, len(targets))
        
        if len(targets) > max_samples:
            # Muestreo aleatorio
            rng = np.random.Generator(np.random.PCG64(CONST_DEFAULT_SEED))
            indices = rng.choice(len(targets), max_samples, replace=False)
            cgm_data = cgm_data[indices]
            other_features = other_features[indices]
            targets = targets[indices]
        
        class InsulinDosingEnv:
            """Entorno personalizado para problema de dosificación de insulina."""
            
            def __init__(self, cgm: np.ndarray, features: np.ndarray, targets: np.ndarray, 
                         model: ValueIterationModel) -> None:
                """
                Inicializa el entorno de dosificación de insulina.
                
                Parámetros:
                -----------
                cgm : np.ndarray
                    Datos CGM
                features : np.ndarray
                    Otras características
                targets : np.ndarray
                    Dosis objetivo
                model : ValueIterationModel
                    Modelo Value Iteration
                """
                self.cgm = cgm
                self.features = features
                self.targets = targets
                self.model = model
                self.current_idx = 0
                self.max_idx = len(targets) - 1
                self.rng = np.random.Generator(np.random.PCG64(CONST_DEFAULT_SEED))
                
                # Preparar entorno
                self.P = {}
                self.prepare_environment()
                
                # Para visualización
                self.shape = (10, 10)  # Grid virtual para visualización
                
            def prepare_environment(self) -> None:
                """Prepara el entorno con transiciones para cada par estado-acción."""
                n_states = self.model.value_iteration_agent.n_states
                n_actions = self.model.value_iteration_agent.n_actions
                
                print(f"Preparando entorno con {n_states} estados y {n_actions} acciones...")
                
                # Caracterizar el espacio de estados
                states = set()
                state_to_indices = {}
                
                # Mapear cada muestra a un estado
                for i in range(len(self.cgm)):
                    state = self.model._discretize_state(self.cgm[i], self.features[i])
                    states.add(state)
                    if state not in state_to_indices:
                        state_to_indices[state] = []
                    state_to_indices[state].append(i)
                
                # Construir modelo de transición
                for state in states:
                    self.P[state] = {}
                    for action in range(n_actions):
                        self.P[state][action] = []
                        
                        # Calcular recompensa promedio para cada acción en cada estado
                        rewards = []
                        for idx in state_to_indices.get(state, []):
                            target = self.targets[idx]
                            dose = self._action_to_dose(action)
                            reward = -abs(dose - target)  # Recompensa negativa por error
                            rewards.append(reward)
                        
                        # Si no hay muestras para este estado, usar recompensa por defecto
                        if not rewards:
                            avg_reward = -1.0
                        else:
                            avg_reward = np.mean(rewards)
                        
                        # Para Value Iteration, asumimos estado terminal después de cada acción
                        next_state = state  # Transición a sí mismo
                        self.P[state][action].append((CONST_PROBABILIDAD, next_state, avg_reward, True))
                
                print(f"Modelo de transición construido con {len(states)} estados únicos")
            
            def _discretize_state(self, cgm: np.ndarray, features: np.ndarray) -> int:
                """
                Discretiza estado usando el modelo.
                
                Parámetros:
                -----------
                cgm : np.ndarray
                    Datos CGM
                features : np.ndarray
                    Otras características
                    
                Retorna:
                --------
                int
                    Estado discreto
                """
                return self.model._discretize_state(cgm, features)
            
            def _action_to_dose(self, action: int) -> float:
                """
                Convierte acción discreta a dosis continua.
                
                Parámetros:
                -----------
                action : int
                    Índice de acción
                    
                Retorna:
                --------
                float
                    Dosis de insulina correspondiente
                """
                action_tensor = torch.tensor([[float(action)]], device=self.model.device)
                return self.model.action_decoder(action_tensor).item()
            
            def reset(self) -> Tuple[int, Dict]:
                """
                Reinicia el entorno seleccionando un ejemplo aleatorio.
                
                Retorna:
                --------
                Tuple[int, Dict]
                    Estado inicial y diccionario con info adicional
                """
                self.current_idx = self.rng.integers(0, self.max_idx + 1)
                state = self._discretize_state(self.cgm[self.current_idx], self.features[self.current_idx])
                return state, {}
            
            def step(self, action: int) -> Tuple[int, float, bool, bool, Dict]:
                """
                Ejecuta un paso en el entorno con la acción dada.
                
                Parámetros:
                -----------
                action : int
                    Acción a ejecutar
                    
                Retorna:
                --------
                Tuple[int, float, bool, bool, Dict]
                    Siguiente estado, recompensa, terminado, truncado, info adicional
                """
                # Convertir acción a dosis
                dose = self._action_to_dose(action)
                
                # Calcular recompensa
                target = self.targets[self.current_idx]
                reward = -abs(dose - target)
                
                # Avanzar al siguiente ejemplo
                self.current_idx = (self.current_idx + 1) % (self.max_idx + 1)
                
                # Obtener siguiente estado
                next_state = self._discretize_state(self.cgm[self.current_idx], self.features[self.current_idx])
                
                # Para este problema, consideramos episodios de un solo paso
                done = True
                truncated = False
                
                return next_state, reward, done, truncated, {}
                
            def state_mapping(self, state: int) -> Tuple[int, int]:
                """
                Mapea un estado a coordenadas para visualización.
                
                Parámetros:
                -----------
                state : int
                    Estado a mapear
                    
                Retorna:
                --------
                Tuple[int, int]
                    Coordenadas (i, j) para visualización
                """
                grid_size = 10  # Para la visualización
                return state // grid_size, state % grid_size
        
        return InsulinDosingEnv(cgm_data, other_features, targets, self)
    
    def fit(
        self, 
        x: List[torch.Tensor], 
        y: np.ndarray, 
        validation_data: Optional[Tuple] = None, 
        epochs: int = CONST_DEFAULT_EPOCHS, 
        batch_size: int = CONST_DEFAULT_BATCH_SIZE, 
        verbose: int = 1, 
        **kwargs
    ) -> Dict[str, List[float]]:
        """
        Entrena el modelo con los datos proporcionados.
        
        Parámetros:
        -----------
        x : List[torch.Tensor]
            Lista con [cgm_data, other_features]
        y : np.ndarray
            Valores objetivo (dosis de insulina)
        validation_data : Optional[Tuple], opcional
            Datos de validación (default: None)
        epochs : int, opcional
            Número de épocas (default: 10)
        batch_size : int, opcional
            Tamaño de lote (default: 32)
        verbose : int, opcional
            Nivel de verbosidad (default: 1)
        **kwargs
            Argumentos adicionales
            
        Retorna:
        --------
        Dict[str, List[float]]
            Historial de entrenamiento
        """
        # Extraer datos
        cgm_data, other_features = x
        
        # Actualizar decoder para mapear acciones a dosis
        self._update_action_decoder(y)
        
        # Convertir tensores a numpy para procesamiento
        cgm_np = cgm_data.detach().cpu().numpy()
        other_np = other_features.detach().cpu().numpy()
        
        # Crear entorno para el agente de Value Iteration si no existe
        if self.env is None:
            self.env = self._create_training_environment(cgm_np, other_np, y)
        
        # Entrenar el agente de Value Iteration (enfoque no diferenciable)
        if verbose > 0:
            print_info("Entrenando modelo de Value Iteration...")
        
        # Multiplicar el máximo de iteraciones por épocas para simular épocas múltiples
        self.value_iteration_agent.max_iterations *= max(1, epochs)
        history = self.value_iteration_agent.train(self.env)
        
        # Restaurar máximo de iteraciones
        self.value_iteration_agent.max_iterations //= max(1, epochs)
        
        # Activar modo de entrenamiento para la parte diferenciable
        self.train()
        
        # Crear optimizador para la parte diferenciable
        optimizer = torch.optim.Adam(self.parameters(), lr=0.001, weight_decay=1e-5)
        criterion = torch.nn.MSELoss()
        
        # Historial de entrenamiento de la parte diferenciable
        nn_history = {'loss': []}
        
        # Crear tensor de objetivos
        y_tensor = torch.tensor(y, dtype=torch.float32, device=self.device)
        
        # Crear dataset y dataloader
        dataset = torch.utils.data.TensorDataset(
            cgm_data, 
            other_features, 
            y_tensor.view(-1, 1)
        )
        dataloader = torch.utils.data.DataLoader(
            dataset, 
            batch_size=batch_size, 
            shuffle=True,
            num_workers=4
        )
        
        # Entrenar la parte diferenciable
        for epoch in range(epochs):
            epoch_loss = 0.0
            batch_count = 0
            
            for batch_cgm, batch_other, batch_y in dataloader:
                # Forward pass
                outputs = self.forward(batch_cgm, batch_other)
                loss = criterion(outputs, batch_y)
                
                # Backward y optimize
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                epoch_loss += loss.item()
                batch_count += 1
            
            # Registrar pérdida promedio
            avg_loss = epoch_loss / batch_count if batch_count > 0 else 0.0
            nn_history['loss'].append(avg_loss)
            
            if verbose > 0:
                print(f"Época {epoch+1}/{epochs} - Pérdida: {avg_loss:.4f}")
        
        # Combinar historiales
        combined_history = {
            'loss': nn_history['loss'],
            'value_changes': history.get('value_changes', []),
            'iteration_times': history.get('iteration_times', [])
        }
        
        # Actualizar métricas de entrenamiento
        self.training_metrics = history
        
        return combined_history
    
    def evaluate(
        self, 
        x: List[torch.Tensor], 
        y: np.ndarray, 
        batch_size: int = CONST_DEFAULT_BATCH_SIZE, 
        verbose: int = 1, 
        **kwargs
    ) -> float:
        """
        Evalúa el modelo con los datos proporcionados.
        
        Parámetros:
        -----------
        x : List[torch.Tensor]
            Lista con [cgm_data, other_features]
        y : np.ndarray
            Valores objetivo (dosis de insulina)
        batch_size : int, opcional
            Tamaño de lote (default: 32)
        verbose : int, opcional
            Nivel de verbosidad (default: 1)
        **kwargs
            Argumentos adicionales
            
        Retorna:
        --------
        float
            Error cuadrático medio
        """
        # Realizar predicciones
        cgm_data, other_features = x
        predictions = self.forward(cgm_data, other_features).detach().cpu().numpy()
        
        # Calcular error cuadrático medio
        mse = np.mean((predictions - y) ** 2)
        
        if verbose > 0:
            print(f"Evaluación - MSE: {mse:.4f}")
        
        return float(mse)
    def predict(
        self, 
        x: List[torch.Tensor], 
        batch_size: int = CONST_DEFAULT_BATCH_SIZE, 
        verbose: int = 0, 
        **kwargs
    ) -> np.ndarray:
        """
        Realiza predicciones con el modelo.
        
        Parámetros:
        -----------
        x : List[torch.Tensor]
            Lista con [cgm_data, other_features]
        batch_size : int, opcional
            Tamaño de lote (default: 32)
        verbose : int, opcional
            Nivel de verbosidad (default: 0)
        **kwargs
            Argumentos adicionales
            
        Retorna:
        --------
        np.ndarray
            Predicciones de dosis de insulina
        """
        # Establecer modo evaluación explícitamente
        self.eval()
        
        with torch.no_grad():
            cgm_data, other_features = x
            predictions = self.forward(cgm_data, other_features)
            return predictions.cpu().numpy()
    
    def get_config(self) -> Dict[str, Any]:
        """
        Obtiene la configuración del modelo.
        
        Retorna:
        --------
        Dict[str, Any]
            Configuración del modelo
        """
        return {
            'cgm_shape': self.cgm_shape,
            'other_features_shape': self.other_features_shape,
            'n_states': self.value_iteration_agent.n_states,
            'n_actions': self.value_iteration_agent.n_actions,
            'gamma': self.value_iteration_agent.gamma,
            'theta': self.value_iteration_agent.theta
        }


def create_value_iteration_model(cgm_shape: Tuple[int, ...], other_features_shape: Tuple[int, ...], **kwargs) -> nn.Module:
    """
    Crea un modelo basado en Value Iteration para predicción de dosis de insulina.
    
    Parámetros:
    -----------
    cgm_shape : Tuple[int, ...]
        Forma de los datos CGM
    other_features_shape : Tuple[int, ...]
        Forma de otras características
    **kwargs
        Argumentos adicionales para configuración
        
    Retorna:
    --------
    nn.Module
        Modelo Value Iteration basado en PyTorch
    """
    # Configuración del espacio de estados y acciones
    n_states = kwargs.get('n_states', 1000)
    n_actions = kwargs.get('n_actions', 20)
    
    # Crear agente Value Iteration
    value_iteration_agent = ValueIteration(
        n_states=n_states,
        n_actions=n_actions,
        gamma=VALUE_ITERATION_CONFIG['gamma'],
        theta=VALUE_ITERATION_CONFIG['theta'],
        max_iterations=VALUE_ITERATION_CONFIG['max_iterations']
    )
    
    # Crear y devolver el modelo wrapper
    return ValueIterationModel(
        value_iteration_agent=value_iteration_agent,
        cgm_shape=cgm_shape,
        other_features_shape=other_features_shape
    )


def create_value_iteration_model_wrapper(cgm_shape: Tuple[int, ...], other_features_shape: Tuple[int, ...], **kwargs) -> RLModelWrapperPyTorch:
    """
    Crea un modelo Value Iteration envuelto en RLModelWrapperPyTorch.
    
    Parámetros:
    -----------
    cgm_shape : Tuple[int, ...]
        Forma de los datos CGM
    other_features_shape : Tuple[int, ...]
        Forma de otras características
    **kwargs
        Argumentos adicionales para configuración
        
    Retorna:
    --------
    RLModelWrapperPyTorch
        Modelo Value Iteration envuelto para compatibilidad con el sistema
    """
    # Definir función creadora que no toma argumentos
    def model_creator_fn() -> nn.Module:
        return create_value_iteration_model(cgm_shape, other_features_shape, **kwargs)
    
    # Crear wrapper
    model_wrapper = RLModelWrapperPyTorch(model_cls=model_creator_fn)
    
    return model_wrapper


def model_creator() -> Callable[[Tuple[int, ...], Tuple[int, ...]], RLModelWrapperPyTorch]:
    """
    Devuelve una función creadora de modelos Value Iteration compatible con la infraestructura.
    
    Retorna:
    --------
    Callable[[Tuple[int, ...], Tuple[int, ...]], RLModelWrapperPyTorch]
        Función que crea un modelo Value Iteration envuelto
    """
    return create_value_iteration_model_wrapper