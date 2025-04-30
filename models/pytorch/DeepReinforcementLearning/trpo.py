import os
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from typing import Dict, List, Tuple, Any, Optional, Union, Callable
from collections import deque
from types import SimpleNamespace

PROJECT_ROOT = os.path.abspath(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))))
sys.path.append(PROJECT_ROOT)

from config.models_config import TRPO_CONFIG
from constants.constants import CONST_DEFAULT_SEED
from custom.drl_model_wrapper import DRLModelWrapperPyTorch

# Constantes para uso repetido
CONST_DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CONST_FIGURES_DIR = os.path.join(PROJECT_ROOT, "figures", "pytorch", "trpo")
os.makedirs(CONST_FIGURES_DIR, exist_ok=True)

# Constantes para nombres de capas del encoder
CONST_CGM_ENCODER = "cgm_encoder"
CONST_OTHER_ENCODER = "other_encoder"
CONST_COMBINED_LAYER = "combined_layer"


class ActorNetwork(nn.Module):
    """
    Red del Actor para TRPO que produce parámetros para distribuciones de política.
    
    Parámetros:
    -----------
    state_dim : int
        Dimensión del espacio de estados
    action_dim : int
        Dimensión del espacio de acciones
    hidden_units : List[int]
        Unidades en cada capa oculta
    continuous : bool
        Si el espacio de acciones es continuo o discreto
    """
    def __init__(
        self, 
        state_dim: int, 
        action_dim: int,
        hidden_units: List[int],
        continuous: bool = True
    ) -> None:
        super(ActorNetwork, self).__init__()
        
        self.continuous = continuous
        self.state_dim = state_dim
        self.action_dim = action_dim
        
        # Crear capas ocultas
        layers = []
        input_dim = state_dim
        
        for units in hidden_units:
            layers.append(nn.Linear(input_dim, units))
            layers.append(nn.Tanh())
            input_dim = units
        
        self.hidden_layers = nn.Sequential(*layers)
        
        # Capas de salida según el tipo de política (continua o discreta)
        if continuous:
            # Para políticas continuas (gaussianas)
            self.mu = nn.Linear(hidden_units[-1], action_dim)
            self.log_std = nn.Parameter(torch.zeros(action_dim))
        else:
            # Para políticas discretas (categóricas)
            self.logits = nn.Linear(hidden_units[-1], action_dim)
    
    def forward(self, x: torch.Tensor) -> Union[Tuple[torch.Tensor, torch.Tensor], torch.Tensor]:
        """
        Pasa la entrada por la red del actor.
        
        Parámetros:
        -----------
        x : torch.Tensor
            Tensor con estados
            
        Retorna:
        --------
        Union[Tuple[torch.Tensor, torch.Tensor], torch.Tensor]
            (mu, std) para políticas continuas o logits para políticas discretas
        """
        x = self.hidden_layers(x)
        
        if self.continuous:
            mu = self.mu(x)
            std = torch.exp(self.log_std)
            return mu, std
        else:
            logits = self.logits(x)
            return logits


class CriticNetwork(nn.Module):
    """
    Red del Crítico para TRPO que estima valores de estado.
    
    Parámetros:
    -----------
    state_dim : int
        Dimensión del espacio de estados
    hidden_units : List[int]
        Unidades en cada capa oculta
    """
    def __init__(
        self, 
        state_dim: int,
        hidden_units: List[int]
    ) -> None:
        super(CriticNetwork, self).__init__()
        
        # Crear capas ocultas
        layers = []
        input_dim = state_dim
        
        for units in hidden_units:
            layers.append(nn.Linear(input_dim, units))
            layers.append(nn.Tanh())
            input_dim = units
        
        layers.append(nn.Linear(input_dim, 1))
        
        self.model = nn.Sequential(*layers)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Pasa la entrada por la red del crítico.
        
        Parámetros:
        -----------
        x : torch.Tensor
            Tensor con estados
            
        Retorna:
        --------
        torch.Tensor
            Valores estimados para los estados
        """
        return self.model(x)


class TRPO:
    """
    Implementación del algoritmo Trust Region Policy Optimization (TRPO).
    
    Parámetros:
    -----------
    state_dim : int
        Dimensión del espacio de estados
    action_dim : int
        Dimensión del espacio de acciones
    continuous : bool, opcional
        Si el espacio de acciones es continuo o discreto (default: True)
    gamma : float, opcional
        Factor de descuento (default: TRPO_CONFIG['gamma'])
    delta : float, opcional
        Límite de divergencia KL (default: TRPO_CONFIG['delta'])
    hidden_units : Optional[List[int]], opcional
        Unidades en capas ocultas (default: None)
    backtrack_iters : int, opcional
        Iteraciones para búsqueda de línea (default: TRPO_CONFIG['backtrack_iters'])
    backtrack_coeff : float, opcional
        Coeficiente de reducción para búsqueda de línea (default: TRPO_CONFIG['backtrack_coeff'])
    cg_iters : int, opcional
        Iteraciones para gradiente conjugado (default: TRPO_CONFIG['cg_iters'])
    damping : float, opcional
        Término de regularización para estabilidad numérica (default: TRPO_CONFIG['damping'])
    seed : int, opcional
        Semilla para reproducibilidad (default: 42)
    """
    def __init__(
        self, 
        state_dim: int, 
        action_dim: int,
        continuous: bool = True,
        gamma: float = TRPO_CONFIG['gamma'],
        delta: float = TRPO_CONFIG['delta'],
        hidden_units: Optional[List[int]] = None,
        backtrack_iters: int = TRPO_CONFIG['backtrack_iters'],
        backtrack_coeff: float = TRPO_CONFIG['backtrack_coeff'],
        cg_iters: int = TRPO_CONFIG['cg_iters'],
        damping: float = TRPO_CONFIG['damping'],
        seed: int = CONST_DEFAULT_SEED
    ) -> None:
        # Configurar semillas para reproducibilidad
        torch.manual_seed(seed)
        np.random.seed(seed)
        self.rng = np.random.Generator(np.random.PCG64(seed))
        
        # Parámetros del modelo
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.continuous = continuous
        self.gamma = gamma
        self.delta = delta
        self.backtrack_iters = backtrack_iters
        self.backtrack_coeff = backtrack_coeff
        self.cg_iters = cg_iters
        self.damping = damping
        
        # Valores predeterminados para capas ocultas
        if hidden_units is None:
            self.hidden_units = [64, 64]
        else:
            self.hidden_units = hidden_units
        
        # Crear redes de actor y crítico
        self.actor = ActorNetwork(
            state_dim=state_dim,
            action_dim=action_dim,
            hidden_units=self.hidden_units,
            continuous=continuous
        ).to(CONST_DEVICE)
        
        self.critic = CriticNetwork(
            state_dim=state_dim,
            hidden_units=self.hidden_units
        ).to(CONST_DEVICE)
        
        # Optimizador para el crítico
        self.critic_optimizer = optim.Adam(
            self.critic.parameters(),
            lr=TRPO_CONFIG.get('critic_learning_rate', 3e-4),
            weight_decay=TRPO_CONFIG.get('critic_weight_decay', 1e-4)
        )
        
        # Historial de entrenamiento
        self.history = {
            'policy_losses': [],
            'value_losses': [],
            'kl_divergences': [],
            'entropies': [],
            'mean_episode_rewards': []
        }
    
    def get_action(self, state: np.ndarray, deterministic: bool = False) -> np.ndarray:
        """
        Obtiene una acción basada en el estado actual.
        
        Parámetros:
        -----------
        state : np.ndarray
            Estado actual del entorno
        deterministic : bool, opcional
            Si es True, devuelve la acción con máxima probabilidad (default: False)
        
        Retorna:
        --------
        np.ndarray
            Una acción muestreada de la distribución de política
        """
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).to(CONST_DEVICE)
            
            if self.continuous:
                mu, std = self.actor(state_tensor)
                if deterministic:
                    action = mu.cpu().numpy()
                else:
                    normal = torch.distributions.Normal(mu, std)
                    action = normal.sample().cpu().numpy()
            else:
                logits = self.actor(state_tensor)
                if deterministic:
                    action = torch.argmax(logits).item()
                else:
                    probs = F.softmax(logits, dim=-1)
                    action = torch.multinomial(probs, 1).item()
                
                # Convertir acción discreta a array para formato consistente
                action = np.array([action])
            
            return action
    
    def get_log_prob(self, states: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        """
        Calcula el logaritmo de probabilidad de acciones bajo la política actual.
        
        Parámetros:
        -----------
        states : torch.Tensor
            Estados observados
        actions : torch.Tensor
            Acciones tomadas
            
        Retorna:
        --------
        torch.Tensor
            Logaritmo de probabilidad de las acciones
        """
        if self.continuous:
            mu, std = self.actor(states)
            normal = torch.distributions.Normal(mu, std)
            log_probs = normal.log_prob(actions)
            return log_probs.sum(dim=1, keepdim=True)
        else:
            logits = self.actor(states)
            dist = torch.distributions.Categorical(logits=logits)
            return dist.log_prob(actions.squeeze().long()).unsqueeze(1)
    
    def get_kl(self, states: torch.Tensor) -> torch.Tensor:
        """
        Computa la divergencia KL entre la política actual y una referencia.
        
        Parámetros:
        -----------
        states : torch.Tensor
            Estados para los que calcular la divergencia KL
            
        Retorna:
        --------
        torch.Tensor
            Divergencia KL media
        """
        if self.continuous:
            mu, std = self.actor(states)
            mu_old = mu.detach()
            std_old = std.detach()
            
            kl = torch.log(std/std_old) + (std_old.pow(2) + (mu_old - mu).pow(2)) / (2.0 * std.pow(2)) - 0.5
            return kl.sum(1, keepdim=True).mean()
        else:
            logits = self.actor(states)
            logits_old = logits.detach()
            
            old_policy = F.softmax(logits_old, dim=-1)
            log_old_policy = F.log_softmax(logits_old, dim=-1)
            log_policy = F.log_softmax(logits, dim=-1)
            
            kl = (old_policy * (log_old_policy - log_policy)).sum(-1, keepdim=True)
            return kl.mean()
    
    def get_entropy(self, states: torch.Tensor) -> torch.Tensor:
        """
        Calcula la entropía de la política para los estados dados.
        
        Parámetros:
        -----------
        states : torch.Tensor
            Estados para los que calcular la entropía
            
        Retorna:
        --------
        torch.Tensor
            Entropía media de la política
        """
        if self.continuous:
            _, std = self.actor(states)
            entropy = 0.5 + 0.5 * np.log(2 * np.pi) + torch.log(std)
            return entropy.sum(dim=-1, keepdim=True).mean()
        else:
            logits = self.actor(states)
            policy = F.softmax(logits, dim=-1)
            log_policy = F.log_softmax(logits, dim=-1)
            entropy = -(policy * log_policy).sum(-1, keepdim=True)
            return entropy.mean()
    
    def compute_gae(
        self, 
        rewards: List[float], 
        values: List[float], 
        next_values: List[float], 
        dones: List[bool], 
        gamma: float = TRPO_CONFIG['gamma'],
        lam: float = TRPO_CONFIG['lambda']
    ) -> Tuple[List[float], List[float]]:
        """
        Calcula el Estimador de Ventaja Generalizada (GAE).
        
        Parámetros:
        -----------
        rewards : List[float]
            Recompensas recibidas
        values : List[float]
            Valores estimados para los estados actuales
        next_values : List[float]
            Valores estimados para los estados siguientes
        dones : List[bool]
            Indicadores de fin de episodio
        gamma : float, opcional
            Factor de descuento (default: TRPO_CONFIG['gamma'])
        lam : float, opcional
            Factor lambda para GAE (default: TRPO_CONFIG['lambda'])
            
        Retorna:
        --------
        Tuple[List[float], List[float]]
            (ventajas, retornos)
        """
        # Inicializar arrays para almacenar resultados
        advantages = np.zeros_like(rewards)
        returns = np.zeros_like(rewards)
        
        # Calcular ventajas de atrás hacia adelante usando GAE
        gae = 0
        for t in reversed(range(len(rewards))):
            if t == len(rewards) - 1:
                next_value = next_values[t]
            else:
                next_value = values[t + 1]
            
            # Delta temporal
            delta = rewards[t] + gamma * next_value * (1 - dones[t]) - values[t]
            
            # Actualizar GAE
            gae = delta + gamma * lam * (1 - dones[t]) * gae
            
            # Almacenar ventaja
            advantages[t] = gae
        
        # Calcular retornos como ventajas + valores
        returns = advantages + values
        
        # Normalizar ventajas para reducir varianza
        advantages = (advantages - np.mean(advantages)) / (np.std(advantages) + 1e-8)
        
        return advantages, returns
    
    def conjugate_gradient(
        self, 
        states: torch.Tensor, 
        b: torch.Tensor, 
        nsteps: int = 10, 
        residual_tol: float = 1e-10
    ) -> torch.Tensor:
        """
        Resuelve el sistema lineal Ax = b usando gradiente conjugado.
        
        Parámetros:
        -----------
        states : torch.Tensor
            Estados para los que calcular el gradiente conjugado
        b : torch.Tensor
            Vector de gradientes
        nsteps : int, opcional
            Número máximo de iteraciones (default: 10)
        residual_tol : float, opcional
            Tolerancia para convergencia (default: 1e-10)
            
        Retorna:
        --------
        torch.Tensor
            Solución aproximada del sistema Ax = b
        """
        x = torch.zeros_like(b)
        r = b.clone()
        p = b.clone()
        rdotr = torch.dot(r, r)
        
        for _ in range(nsteps):
            Ap = self.fisher_vector_product(states, p)
            alpha = rdotr / (torch.dot(p, Ap) + 1e-8)
            
            x += alpha * p
            r -= alpha * Ap
            
            new_rdotr = torch.dot(r, r)
            beta = new_rdotr / (rdotr + 1e-8)
            p = r + beta * p
            rdotr = new_rdotr
            
            if rdotr < residual_tol:
                break
                
        return x
    
    def fisher_vector_product(self, states: torch.Tensor, p: torch.Tensor) -> torch.Tensor:
        """
        Calcula el producto Fisher-vector para el método de gradiente conjugado.
        
        Parámetros:
        -----------
        states : torch.Tensor
            Estados para los que calcular el producto Fisher-vector
        p : torch.Tensor
            Vector para multiplicar con la matriz Fisher
            
        Retorna:
        --------
        torch.Tensor
            Producto Fisher-vector
        """
        # Calcular KL divergence
        kl = self.get_kl(states)
        
        # Obtener gradientes de KL con respecto a parámetros del actor
        grads = torch.autograd.grad(kl, self.actor.parameters(), create_graph=True)
        flat_grad_kl = torch.cat([grad.view(-1) for grad in grads])
        
        # Producto con el vector p
        kl_p = (flat_grad_kl * p).sum()
        
        # Gradiente del producto
        grads_p = torch.autograd.grad(kl_p, self.actor.parameters(), retain_graph=True)
        flat_grad_grad_kl = torch.cat([grad.contiguous().view(-1) for grad in grads_p])
        
        return flat_grad_grad_kl + self.damping * p
    
    def get_flat_params(self) -> torch.Tensor:
        """
        Obtiene los parámetros de la red del actor en un vector plano.
        
        Retorna:
        --------
        torch.Tensor
            Vector con los parámetros
        """
        return torch.cat([param.data.view(-1) for param in self.actor.parameters()])
    
    def set_flat_params(self, flat_params: torch.Tensor) -> None:
        """
        Establece los parámetros de la red del actor desde un vector plano.
        
        Parámetros:
        -----------
        flat_params : torch.Tensor
            Vector con los nuevos parámetros
        """
        prev_ind = 0
        for param in self.actor.parameters():
            flat_size = param.numel()
            param.data.copy_(flat_params[prev_ind:prev_ind + flat_size].view(param.size()))
            prev_ind += flat_size
    
    def update_policy(
        self, 
        states: torch.Tensor, 
        actions: torch.Tensor, 
        advantages: torch.Tensor
    ) -> Dict[str, float]:
        """
        Actualiza la política utilizando el método TRPO.
        
        Parámetros:
        -----------
        states : torch.Tensor
            Estados
        actions : torch.Tensor
            Acciones
        advantages : torch.Tensor
            Ventajas
            
        Retorna:
        --------
        Dict[str, float]
            Métricas de actualización
        """
        # Calcular probabilidades de la política actual
        log_probs_old = self.get_log_prob(states, actions).detach()
        
        # Calcular función objetivo
        def surrogate():
            log_probs = self.get_log_prob(states, actions)
            ratio = torch.exp(log_probs - log_probs_old)
            surr = (ratio * advantages).mean()
            return surr
        
        # Calcular gradiente de la función objetivo
        loss = surrogate()
        grads = torch.autograd.grad(loss, self.actor.parameters())
        loss_grad = torch.cat([grad.view(-1) for grad in grads]).detach()
        
        # Calcular dirección de actualización usando gradiente conjugado
        step_dir = self.conjugate_gradient(states, loss_grad, nsteps=self.cg_iters)
        
        # Calcular el tamaño del paso
        fvp = self.fisher_vector_product(states, step_dir)
        shs = 0.5 * (step_dir * fvp).sum()
        lm = torch.sqrt(2 * self.delta / (shs + 1e-8))
        fullstep = step_dir * lm
        
        # Guardar parámetros antiguos
        old_params = self.get_flat_params()
        
        # Realizar búsqueda de línea para encontrar los mejores parámetros
        # expected_improve = (loss_grad * fullstep).sum()
        
        # Búsqueda de línea con backtracking
        for i in range(self.backtrack_iters):
            # Probar nuevos parámetros
            new_params = old_params + fullstep * self.backtrack_coeff**i
            self.set_flat_params(new_params)
            
            # Evaluar la mejora
            new_loss = surrogate()
            actual_improve = new_loss - loss
            
            # Si la mejora es suficiente, terminar la búsqueda
            if actual_improve > 0:
                break
                
            # Si es la última iteración, volver a los parámetros antiguos
            if i == self.backtrack_iters - 1:
                self.set_flat_params(old_params)
        
        # Calcular métricas para seguimiento
        kl = self.get_kl(states).item()
        entropy = self.get_entropy(states).item()
        
        return {
            'policy_loss': -loss.item(),
            'kl': kl,
            'entropy': entropy
        }
    
    def update_value(
        self, 
        states: torch.Tensor, 
        returns: torch.Tensor, 
        epochs: int = TRPO_CONFIG['value_epochs'],
        batch_size: int = TRPO_CONFIG['batch_size']
    ) -> Dict[str, float]:
        """
        Actualiza la función de valor.
        
        Parámetros:
        -----------
        states : torch.Tensor
            Estados
        returns : torch.Tensor
            Retornos objetivo
        epochs : int, opcional
            Número de épocas para entrenar (default: TRPO_CONFIG['value_epochs'])
        batch_size : int, opcional
            Tamaño de lote (default: TRPO_CONFIG['batch_size'])
            
        Retorna:
        --------
        Dict[str, float]
            Métricas de actualización
        """
        dataset = torch.utils.data.TensorDataset(states, returns)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4)
        
        value_losses = []
        
        for _ in range(epochs):
            for state_batch, return_batch in dataloader:
                # Predicción y cálculo de pérdida
                value_pred = self.critic(state_batch)
                value_loss = F.mse_loss(value_pred, return_batch)
                
                # Actualización
                self.critic_optimizer.zero_grad()
                value_loss.backward()
                self.critic_optimizer.step()
                
                value_losses.append(value_loss.item())
        
        return {
            'value_loss': np.mean(value_losses)
        }
    
    def collect_trajectories(
        self, 
        env: Any, 
        min_steps: int = TRPO_CONFIG['min_steps_per_update'],
        render: bool = False
    ) -> Dict[str, np.ndarray]:
        """
        Recolecta trayectorias (experiencias) interactuando con el entorno.
        
        Parámetros:
        -----------
        env : Any
            Entorno para interactuar
        min_steps : int, opcional
            Número mínimo de pasos a recolectar (default: TRPO_CONFIG['min_steps_per_update'])
        render : bool, opcional
            Si renderizar el entorno (default: False)
            
        Retorna:
        --------
        Dict[str, np.ndarray]
            Diccionario con experiencias recolectadas
        """
        states, actions, rewards, next_states, dones = [], [], [], [], []
        episode_rewards = []
        
        current_episode_reward = 0
        state, _ = env.reset()
        
        steps_taken = 0
        while steps_taken < min_steps:
            if render:
                env.render()
            
            # Seleccionar acción
            action = self.get_action(state)
            
            # Ejecutar acción en el entorno
            next_state, reward, done, _, _ = env.step(action)
            
            # Guardar experiencia
            states.append(state)
            actions.append(action)
            rewards.append(reward)
            next_states.append(next_state)
            dones.append(float(done))
            
            # Actualizar estado y recompensa acumulada
            state = next_state
            current_episode_reward += reward
            steps_taken += 1
            
            # Manejar fin de episodio
            if done:
                episode_rewards.append(current_episode_reward)
                current_episode_reward = 0
                state, _ = env.reset()
        
        # Calcular valores para cada estado usando el crítico
        with torch.no_grad():
            states_tensor = torch.FloatTensor(np.array(states)).to(CONST_DEVICE)
            next_states_tensor = torch.FloatTensor(np.array(next_states)).to(CONST_DEVICE)
            
            values = self.critic(states_tensor).cpu().numpy().flatten()
            next_values = self.critic(next_states_tensor).cpu().numpy().flatten()
        
        # Calcular ventajas y retornos
        advantages, returns = self.compute_gae(
            rewards=np.array(rewards),
            values=values,
            next_values=next_values,
            dones=np.array(dones),
            gamma=self.gamma,
            lam=TRPO_CONFIG['lambda']
        )
        
        # Retornar experiencias
        return {
            'states': np.array(states),
            'actions': np.array(actions),
            'returns': returns,
            'advantages': advantages,
            'episode_rewards': np.array(episode_rewards) if episode_rewards else np.array([0])
        }
    
    def train(
        self, 
        env: Any, 
        iterations: int = TRPO_CONFIG['iterations'],
        min_steps_per_update: int = TRPO_CONFIG['min_steps_per_update'],
        render: bool = False,
        evaluate_interval: int = TRPO_CONFIG['evaluate_interval']
    ) -> Dict[str, List[float]]:
        """
        Entrena el agente TRPO en el entorno proporcionado.
        
        Parámetros:
        -----------
        env : Any
            Entorno para entrenamiento
        iterations : int, opcional
            Número de iteraciones de entrenamiento (default: TRPO_CONFIG['iterations'])
        min_steps_per_update : int, opcional
            Número mínimo de pasos por cada actualización (default: TRPO_CONFIG['min_steps_per_update'])
        render : bool, opcional
            Si renderizar durante el entrenamiento (default: False)
        evaluate_interval : int, opcional
            Intervalo para evaluación (default: TRPO_CONFIG['evaluate_interval'])
            
        Retorna:
        --------
        Dict[str, List[float]]
            Historial de entrenamiento
        """
        policy_losses = []
        value_losses = []
        kls = []
        entropies = []
        mean_rewards = []
        
        for i in range(iterations):
            # Recolectar trayectorias
            trajectories = self.collect_trajectories(env, min_steps_per_update, render)
            
            # Convertir a tensores
            states = torch.FloatTensor(trajectories['states']).to(CONST_DEVICE)
            actions = torch.FloatTensor(trajectories['actions']).to(CONST_DEVICE)
            returns = torch.FloatTensor(trajectories['returns']).to(CONST_DEVICE).unsqueeze(1)
            advantages = torch.FloatTensor(trajectories['advantages']).to(CONST_DEVICE).unsqueeze(1)
            
            # Actualizar política
            policy_update = self.update_policy(states, actions, advantages)
            
            # Actualizar función de valor
            value_update = self.update_value(states, returns)
            
            # Registrar métricas
            policy_losses.append(policy_update['policy_loss'])
            value_losses.append(value_update['value_loss'])
            kls.append(policy_update['kl'])
            entropies.append(policy_update['entropy'])
            mean_rewards.append(float(np.mean(trajectories['episode_rewards'])))
            
            # Imprimir progreso
            if (i + 1) % evaluate_interval == 0:
                print(f"Iter {i+1}/{iterations} | "
                      f"Mean Reward: {mean_rewards[-1]:.2f} | "
                      f"Policy Loss: {policy_losses[-1]:.4f} | "
                      f"Value Loss: {value_losses[-1]:.4f} | "
                      f"KL: {kls[-1]:.4f} | "
                      f"Entropy: {entropies[-1]:.4f}")
        
        # Actualizar historial
        self.history['policy_losses'] = policy_losses
        self.history['value_losses'] = value_losses
        self.history['kl_divergences'] = kls
        self.history['entropies'] = entropies
        self.history['mean_episode_rewards'] = mean_rewards
        
        return self.history
    
    def evaluate(
        self, 
        env: Any, 
        episodes: int = 10, 
        render: bool = False
    ) -> float:
        """
        Evalúa el agente en el entorno.
        
        Parámetros:
        -----------
        env : Any
            Entorno para evaluación
        episodes : int, opcional
            Número de episodios para evaluación (default: 10)
        render : bool, opcional
            Si renderizar durante la evaluación (default: False)
            
        Retorna:
        --------
        float
            Recompensa media por episodio
        """
        episode_rewards = []
        
        for _ in range(episodes):
            state, _ = env.reset()
            done = False
            total_reward = 0
            
            while not done:
                if render:
                    env.render()
                
                # Seleccionar acción determinística
                action = self.get_action(state, deterministic=True)
                
                # Ejecutar acción en el entorno
                next_state, reward, done, _, _ = env.step(action)
                
                # Actualizar estado y recompensa
                state = next_state
                total_reward += reward
            
            episode_rewards.append(total_reward)
        
        mean_reward = np.mean(episode_rewards)
        print(f"Evaluación: {mean_reward:.2f} recompensa media en {episodes} episodios")
        
        return mean_reward
    
    def save_model(self, actor_path: str, critic_path: str) -> None:
        """
        Guarda los modelos del actor y crítico.
        
        Parámetros:
        -----------
        actor_path : str
            Ruta para guardar el modelo del actor
        critic_path : str
            Ruta para guardar el modelo del crítico
        """
        torch.save(self.actor.state_dict(), actor_path)
        torch.save(self.critic.state_dict(), critic_path)
        print(f"Modelos guardados en {actor_path} y {critic_path}")
    
    def load_model(self, actor_path: str, critic_path: str) -> None:
        """
        Carga los modelos del actor y crítico.
        
        Parámetros:
        -----------
        actor_path : str
            Ruta desde donde cargar el modelo del actor
        critic_path : str
            Ruta desde donde cargar el modelo del crítico
        """
        self.actor.load_state_dict(torch.load(actor_path, map_location=CONST_DEVICE))
        self.critic.load_state_dict(torch.load(critic_path, map_location=CONST_DEVICE))
        print(f"Modelos cargados desde {actor_path} y {critic_path}")


class TRPOWrapper(nn.Module):
    """
    Wrapper para el algoritmo TRPO que implementa la interfaz compatible con modelos de aprendizaje profundo.
    
    Parámetros:
    -----------
    trpo_agent : TRPO
        Agente TRPO
    cgm_shape : Tuple[int, ...]
        Forma de los datos CGM
    other_features_shape : Tuple[int, ...]
        Forma de otras características
    """
    def __init__(
        self, 
        trpo_agent: TRPO,
        cgm_shape: Tuple[int, ...],
        other_features_shape: Tuple[int, ...],
    ) -> None:
        super(TRPOWrapper, self).__init__()
        self.trpo_agent = trpo_agent
        self.cgm_shape = cgm_shape
        self.other_features_shape = other_features_shape
        
        # Setup para encoders y decoders
        self._setup_encoders()
        
        # Variables para seguimiento de entrenamiento
        self.history = {
            'loss': [],
            'val_loss': []
        }
    
    def _setup_encoders(self) -> None:
        """
        Configura los encoders para CGM y otras características.
        """
        # Calcular dimensiones aplanadas
        cgm_flat_dim = np.prod(self.cgm_shape)
        other_flat_dim = np.prod(self.other_features_shape)
        
        # Encoder para datos CGM
        self.cgm_encoder = nn.Sequential(
            nn.Flatten(),
            nn.Linear(cgm_flat_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 32),
            nn.ReLU()
        )
        
        # Encoder para otras características
        self.other_encoder = nn.Sequential(
            nn.Flatten(),
            nn.Linear(other_flat_dim, 16),
            nn.ReLU(),
            nn.Linear(16, 16),
            nn.ReLU()
        )
        
        # Capa combinada
        self.combined_layer = nn.Sequential(
            nn.Linear(32 + 16, 64),
            nn.ReLU(),
            nn.Linear(64, self.trpo_agent.state_dim)
        )
        
        # Capa de acción para entrenamiento
        self.action_head = nn.Sequential(
            nn.Linear(self.trpo_agent.state_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1)  # Salida escalar para dosis de insulina
        )
    
    def forward(self, x_cgm: torch.Tensor, x_other: torch.Tensor) -> torch.Tensor:
        """
        Pasa las entradas por el modelo para obtener predicciones.
        
        Parámetros:
        -----------
        x_cgm : torch.Tensor
            Tensor con datos CGM
        x_other : torch.Tensor
            Tensor con otras características
            
        Retorna:
        --------
        torch.Tensor
            Predicciones de dosis de insulina
        """
        # Codificar entradas
        cgm_encoded = self.cgm_encoder(x_cgm)
        other_encoded = self.other_encoder(x_other)
        combined = torch.cat([cgm_encoded, other_encoded], dim=1)
        state = self.combined_layer(combined)
        
        if self.training:
            # Durante entrenamiento: usar capas diferenciables
            return self.action_head(state)
        else:
            # Durante evaluación: usar el agente TRPO completo (no diferenciable)
            batch_size = state.shape[0]
            actions = torch.zeros((batch_size, 1), device=state.device)
            
            for i in range(batch_size):
                s = state[i].cpu().numpy()  # No usar detach() aquí
                action = self.trpo_agent.get_action(s, deterministic=True)
                if self.trpo_agent.continuous:
                    actions[i] = torch.tensor(action, device=state.device)
                else:
                    action_scaled = action[0] / (self.trpo_agent.action_dim - 1) * 15.0
                    actions[i] = torch.tensor(action_scaled, device=state.device)
            
            return actions

    def predict(self, x_cgm: torch.Tensor, x_other: torch.Tensor) -> np.ndarray:
        """
        Realiza predicciones usando el modelo TRPO entrenado.
        
        Parámetros:
        -----------
        x_cgm : torch.Tensor
            Tensor con datos CGM
        x_other : torch.Tensor
            Tensor con otras características
            
        Retorna:
        --------
        np.ndarray
            Predicciones de dosis de insulina
        """
        # Si la entrada es numpy arrays, convertir a tensores
        if not isinstance(x_cgm, torch.Tensor):
            x_cgm = torch.FloatTensor(x_cgm)
        if not isinstance(x_other, torch.Tensor):
            x_other = torch.FloatTensor(x_other)
        
        # Mover a dispositivo
        x_cgm = x_cgm.to(CONST_DEVICE)
        x_other = x_other.to(CONST_DEVICE)
        
        # Hacer predicción usando forward
        with torch.no_grad():
            predictions = self.forward(x_cgm, x_other)
        
        # Convertir a numpy array
        return predictions.cpu().numpy()

    def _prepare_tensor(self, x: Any) -> torch.Tensor:
        """
        Convierte a tensor y mueve al dispositivo adecuado.
        
        Parámetros:
        -----------
        x : Any
            Datos para convertir
            
        Retorna:
        --------
        torch.Tensor
            Tensor en el dispositivo correcto
        """
        if not isinstance(x, torch.Tensor):
            x = torch.FloatTensor(x)
        return x.to(CONST_DEVICE)
    
    def _prepare_validation_data(self, validation_data: Tuple) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Prepara los datos de validación.
        
        Parámetros:
        -----------
        validation_data : Tuple
            Datos de validación ((x_cgm_val, x_other_val), y_val)
            
        Retorna:
        --------
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor]
            Datos de validación preparados
        """
        (val_cgm, val_other), val_y = validation_data
        val_cgm = self._prepare_tensor(val_cgm)
        val_other = self._prepare_tensor(val_other)
        val_y = self._prepare_tensor(val_y)
        return val_cgm, val_other, val_y
        
    def _compute_loss(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Calcula la pérdida MSE.
        
        Parámetros:
        -----------
        predictions : torch.Tensor
            Predicciones del modelo
        targets : torch.Tensor
            Valores objetivo
            
        Retorna:
        --------
        torch.Tensor
            Valor de pérdida
        """
        return torch.mean((predictions.flatten() - targets) ** 2)
        
    def _log_epoch_results(self, epoch: int, epochs: int, train_loss: torch.Tensor, val_loss: Optional[torch.Tensor] = None):
        """
        Registra los resultados de la época en consola.
        
        Parámetros:
        -----------
        epoch : int
            Época actual
        epochs : int
            Total de épocas
        train_loss : torch.Tensor
            Pérdida de entrenamiento
        val_loss : Optional[torch.Tensor]
            Pérdida de validación
        """
        print(f"Época {epoch+1}/{epochs}")
        if val_loss is not None:
            print(f"train_loss: {train_loss.item():.4f}, val_loss: {val_loss.item():.4f}")
        else:
            print(f"train_loss: {train_loss.item():.4f}")

    def fit(
        self, 
        x_cgm: torch.Tensor, 
        x_other: torch.Tensor, 
        y: torch.Tensor, 
        validation_data: Optional[Tuple] = None, 
        epochs: int = 1,
        batch_size: int = 32,
        callbacks: List = None,
        verbose: int = 0
    ) -> Dict:
        """
        Entrena el modelo TRPO con los datos proporcionados.
        
        Parámetros:
        -----------
        x_cgm : torch.Tensor
            Tensor con datos CGM
        x_other : torch.Tensor
            Tensor con otras características
        y : torch.Tensor
            Valores objetivo (dosis de insulina)
        validation_data : Optional[Tuple]
            Datos de validación ((x_cgm_val, x_other_val), y_val)
        epochs : int, opcional
            Número de épocas (default: 1)
        batch_size : int, opcional
            Tamaño del lote (default: 32)
        callbacks : List, opcional
            Lista de callbacks (no utilizado, para compatibilidad)
        verbose : int, opcional
            Nivel de verbosidad (default: 0)
            
        Retorna:
        --------
        Dict
            Historial de entrenamiento
        """
        # Preparar datos de entrada
        x_cgm = self._prepare_tensor(x_cgm)
        x_other = self._prepare_tensor(x_other)
        y = self._prepare_tensor(y)
        
        # Preparar validación si existe
        val_data = None
        if validation_data:
            val_data = self._prepare_validation_data(validation_data)
        
        # Crear entorno de entrenamiento
        env = self._create_training_environment(x_cgm, x_other, y)
        
        # Entrenar por épocas
        for epoch in range(epochs):
            # Entrenar usando el algoritmo TRPO
            self.trpo_agent.train(
                env=env, iterations=1, min_steps_per_update=len(y),
                render=False, evaluate_interval=1
            )
            
            # Evaluar modelo
            train_preds = self.forward(x_cgm, x_other)
            train_loss = self._compute_loss(train_preds, y)
            self.history['loss'].append(float(train_loss.item()))
            
            # Evaluar en validación si hay datos
            val_loss = None
            if val_data:
                val_cgm, val_other, val_y = val_data
                with torch.no_grad():
                    val_preds = self.forward(val_cgm, val_other)
                    val_loss = self._compute_loss(val_preds, val_y)
                    self.history['val_loss'].append(float(val_loss.item()))
            
            # Mostrar progreso
            if verbose > 0:
                self._log_epoch_results(epoch, epochs, train_loss, val_loss)
        
        return self.history

    def evaluate(
        self, 
        x_cgm: torch.Tensor, 
        x_other: torch.Tensor, 
        y: torch.Tensor, 
        batch_size: int = 32,
        verbose: int = 0
    ) -> Dict[str, float]:
        """
        Evalúa el rendimiento del modelo en datos de prueba.
        
        Parámetros:
        -----------
        x_cgm : torch.Tensor
            Tensor con datos CGM
        x_other : torch.Tensor
            Tensor con otras características
        y : torch.Tensor
            Valores objetivo (dosis de insulina)
        batch_size : int, opcional
            Tamaño del lote (default: 32)
        verbose : int, opcional
            Nivel de verbosidad (default: 0)
            
        Retorna:
        --------
        Dict[str, float]
            Métricas de evaluación
        """
        # Convertir a tensores si es necesario
        if not isinstance(x_cgm, torch.Tensor):
            x_cgm = torch.FloatTensor(x_cgm)
        if not isinstance(x_other, torch.Tensor):
            x_other = torch.FloatTensor(x_other)
        if not isinstance(y, torch.Tensor):
            y = torch.FloatTensor(y)
        
        # Mover a dispositivo
        x_cgm = x_cgm.to(CONST_DEVICE)
        x_other = x_other.to(CONST_DEVICE)
        y = y.to(CONST_DEVICE)
        
        # Hacer predicciones
        with torch.no_grad():
            y_pred = self.forward(x_cgm, x_other)
        
        # Calcular métricas
        mse = torch.mean((y_pred.flatten() - y) ** 2).item()
        mae = torch.mean(torch.abs(y_pred.flatten() - y)).item()
        
        # Mostrar resultados si es verbose
        if verbose > 0:
            print(f"Evaluación - MSE: {mse:.4f}, MAE: {mae:.4f}")
        
        return {
            'mse': mse,
            'mae': mae
        }

    # También necesitamos actualizar el método _create_training_environment
    def _create_training_environment(
        self, 
        cgm_data: torch.Tensor, 
        other_features: torch.Tensor, 
        targets: torch.Tensor
    ) -> Any:
        """
        Crea un entorno de entrenamiento personalizado para RL.
        
        Parámetros:
        -----------
        cgm_data : torch.Tensor
            Datos CGM
        other_features : torch.Tensor
            Otras características
        targets : torch.Tensor
            Valores objetivo (dosis de insulina)
            
        Retorna:
        --------
        Any
            Entorno compatible con interfaz OpenAI Gym
        """
        # Convertir tensores a numpy para procesamiento
        cgm_np = cgm_data.cpu().numpy()
        other_np = other_features.cpu().numpy()
        targets_np = targets.cpu().numpy()
        
        class InsulinDosingEnv:
            """Entorno personalizado para problema de dosificación de insulina."""
            
            def __init__(self, cgm, features, targets, model_wrapper):
                self.cgm = cgm
                self.features = features
                self.targets = targets
                self.model = model_wrapper
                self.rng = np.random.Generator(np.random.PCG64(CONST_DEFAULT_SEED))
                self.current_idx = 0
                self.max_idx = len(targets) - 1
                
                # Para compatibilidad con RL
                self.observation_space = SimpleNamespace(
                    shape=(model_wrapper.trpo_agent.state_dim,),
                    low=np.full((model_wrapper.trpo_agent.state_dim,), -10.0),
                    high=np.full((model_wrapper.trpo_agent.state_dim,), 10.0)
                )
                
                if model_wrapper.trpo_agent.continuous:
                    self.action_space = SimpleNamespace(
                        shape=(model_wrapper.trpo_agent.action_dim,),
                        low=np.zeros(model_wrapper.trpo_agent.action_dim),
                        high=np.ones(model_wrapper.trpo_agent.action_dim) * 15.0,  # Max 15 unidades
                        sample=self._sample_action
                    )
                else:
                    self.action_space = SimpleNamespace(
                        n=model_wrapper.trpo_agent.action_dim,
                        sample=lambda: self.rng.integers(0, model_wrapper.trpo_agent.action_dim)
                    )
                
                self.render_mode = None
            
            def _sample_action(self) -> np.ndarray:
                """Muestrea una acción aleatoria del espacio continuo."""
                return self.rng.uniform(
                    self.action_space.low,
                    self.action_space.high
                )
            
            def reset(self) -> Tuple[np.ndarray, Dict]:
                """Reinicia el entorno eligiendo un ejemplo aleatorio."""
                self.current_idx = self.rng.integers(0, self.max_idx)
                state = self._get_state()
                return state, {}
            
            def step(self, action: Union[np.ndarray, int]) -> Tuple[np.ndarray, float, bool, bool, Dict]:
                """Ejecuta un paso con la acción dada."""
                # Convertir acción a dosis según el tipo de espacio
                if self.model.trpo_agent.continuous:
                    dose = float(action[0])  # Ya está en escala correcta
                else:
                    # Para acciones discretas, mapear a rango de dosis
                    dose = action / (self.model.trpo_agent.action_dim - 1) * 15.0
                
                # Calcular recompensa como negativo del error absoluto
                target = self.targets[self.current_idx]
                reward = -abs(dose - target)
                
                # Avanzar al siguiente ejemplo
                self.current_idx = (self.current_idx + 1) % self.max_idx
                
                # Obtener próximo estado
                next_state = self._get_state()
                
                # Episodio siempre termina después de una acción
                done = True
                truncated = False
                
                return next_state, reward, done, truncated, {}
            
            def _get_state(self) -> np.ndarray:
                """Obtiene el estado codificado para el ejemplo actual."""
                # Obtener datos actuales
                cgm_batch = self.cgm[self.current_idx:self.current_idx+1]
                features_batch = self.features[self.current_idx:self.current_idx+1]
                
                # Codificar a estado
                with torch.no_grad():
                    cgm_tensor = torch.FloatTensor(cgm_batch).to(CONST_DEVICE)
                    other_tensor = torch.FloatTensor(features_batch).to(CONST_DEVICE)
                    
                    cgm_encoded = self.model.cgm_encoder(cgm_tensor)
                    other_encoded = self.model.other_encoder(other_tensor)
                    
                    combined = torch.cat([cgm_encoded, other_encoded], dim=1)
                    state = self.model.combined_layer(combined)
                
                return state.cpu().numpy()[0]
            
            def render(self) -> None:
                """Renderización dummy del entorno (no implementada)."""
                pass
        
        return InsulinDosingEnv(cgm_np, other_np, targets_np, self)
    

def create_trpo_model(cgm_shape: Tuple[int, ...], other_features_shape: Tuple[int, ...]) -> DRLModelWrapperPyTorch:
    """
    Crea un modelo basado en TRPO (Trust Region Policy Optimization) para predicción de dosis de insulina.
    
    Parámetros:
    -----------
    cgm_shape : Tuple[int, ...]
        Forma de los datos CGM (batch_size, time_steps, features)
    other_features_shape : Tuple[int, ...]
        Forma de otras características (batch_size, n_features)
        
    Retorna:
    --------
    DRLModelWrapperPyTorch
        Modelo TRPO que implementa la interfaz del sistema
    """
    # Configurar dimensiones
    state_dim = 64  # Dimensión del espacio de estado latente
    action_dim = 20  # Acciones discretas para representar dosis [0-15]
    
    # Definir si usar espacio continuo o discreto para las acciones
    continuous = TRPO_CONFIG.get('continuous', False)
    
    def model_creator(**kwargs) -> nn.Module:
        trpo_agent = TRPO(
            state_dim=state_dim,
            action_dim=action_dim,
            continuous=continuous,
            gamma=TRPO_CONFIG['gamma'],
            delta=TRPO_CONFIG['delta'],
            hidden_units=TRPO_CONFIG['hidden_units'],
            backtrack_iters=TRPO_CONFIG['backtrack_iters'],
            backtrack_coeff=TRPO_CONFIG['backtrack_coeff'],
            cg_iters=TRPO_CONFIG['cg_iters'],
            damping=TRPO_CONFIG['damping'],
            seed=TRPO_CONFIG.get('seed', CONST_DEFAULT_SEED)
        )
        
        wrapper = TRPOWrapper(
            trpo_agent=trpo_agent,
            cgm_shape=cgm_shape,
            other_features_shape=other_features_shape
        )
        
        return wrapper
    
    # Crear y devolver el wrapper para el sistema
    return DRLModelWrapperPyTorch(model_creator, algorithm='trpo')


def model_creator() -> Callable[[Tuple[int, ...], Tuple[int, ...]], DRLModelWrapperPyTorch]:
    """
    Retorna una función para crear un modelo TRPO compatible con la API del sistema.
    
    Retorna:
    --------
    Callable[[Tuple[int, ...], Tuple[int, ...]], DRLModelWrapperPyTorch]
        Función para crear el modelo con las formas de entrada especificadas
    """
    return create_trpo_model