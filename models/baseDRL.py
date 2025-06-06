import torch
import torch.nn as nn
import numpy as np
from typing import Dict, Tuple, List, Optional

class BaseDRLModel(nn.Module):
    """
    Clase base para modelos de aprendizaje por refuerzo profundo.
    
    Parámetros:
    -----------
    state_dim : Tuple[int, ...]
        Dimensiones del estado (CGM + otras características)
    action_dim : int
        Dimensiones de la acción (dosis de insulina)
    hidden_dim : int, opcional
        Dimensiones de las capas ocultas (default: 256)
    lr : float, opcional
        Tasa de aprendizaje (default: 3e-4)
    """
    
    def __init__(self, cgm_input_dim: tuple, other_input_dim: tuple, 
                 action_dim: int = 1, hidden_dim: int = 256, lr: float = 3e-4):
        super().__init__()
        self.cgm_input_dim = cgm_input_dim
        self.other_input_dim = other_input_dim
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
        self.lr = lr
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Encoders para diferentes tipos de entrada
        self.cgm_encoder = self._build_cgm_encoder()
        self.other_encoder = self._build_other_encoder()
        self.combined_layer = self._build_combined_layer()
        
        # Buffer de experiencia
        self.buffer = self._initialize_buffer(10000)  # Tamaño por defecto
        
    def _build_cgm_encoder(self) -> nn.Module:
        """
        Construye el encoder para datos CGM.
        """
        return nn.Sequential(
            nn.Linear(self.cgm_input_dim[0], 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU()
        )
    
    def _build_other_encoder(self) -> nn.Module:
        """
        Construye el encoder para otras características.
        """
        return nn.Sequential(
            nn.Linear(self.other_input_dim[0], 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU()
        )
    
    def _build_combined_layer(self) -> nn.Module:
        """
        Construye la capa que combina las características.
        """
        return nn.Sequential(
            nn.Linear(64 + 32, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU()
        )
    
    def _initialize_buffer(self, capacity: int) -> Dict:
        """
        Inicializa el buffer de experiencia.
        """
        return {
            'states_cgm': [],
            'states_other': [],
            'actions': [],
            'rewards': [],
            'next_states_cgm': [],
            'next_states_other': [],
            'dones': [],
            'capacity': capacity,
            'position': 0,
            'size': 0
        }
    
    def forward(self, x_cgm: torch.Tensor, x_other: torch.Tensor) -> torch.Tensor:
        """
        Paso hacia adelante del modelo.
        """
        cgm_features = self.cgm_encoder(x_cgm)
        other_features = self.other_encoder(x_other)
        combined = torch.cat([cgm_features, other_features], dim=1)
        features = self.combined_layer(combined)
        return features
    
    def predict_with_context(self, x_cgm: np.ndarray, x_other: np.ndarray, **context) -> float:
        """
        Predice dosis de insulina con contexto.
        
        Parámetros:
        -----------
        x_cgm : np.ndarray
            Datos CGM para predicción (array con mediciones de glucosa recientes)
        x_other : np.ndarray
            Otras características para predicción
        **context : dict
            Variables contextuales como:
            - carb_intake : float - Ingesta de carbohidratos (obligatorio)
            - sleep_quality : int - Calidad del sueño (opcional, 0-10)
            - work_intensity : int - Intensidad del trabajo (opcional, 0-10)
            - exercise_intensity : int - Intensidad del ejercicio (opcional, 0-10)
            
        Retorna:
        --------
        float
            Dosis de insulina recomendada en unidades
        """
        # Verificar parámetros obligatorios
        if 'carb_intake' not in context:
            raise ValueError("El parámetro 'carb_intake' es obligatorio")
        
        # Convertir a tensores
        x_cgm_t = torch.FloatTensor(x_cgm).to(self.device)
        x_other_t = torch.FloatTensor(x_other).to(self.device)
        
        # Obtener valores de contexto
        carb_intake = float(context['carb_intake'])
        sleep_quality = float(context.get('sleep_quality', 5.0))
        work_intensity = float(context.get('work_intensity', 0.0))
        exercise_intensity = float(context.get('exercise_intensity', 0.0))
        
        # Extraer valor de glucosa actual del array x_cgm (último valor)
        current_glucose = float(x_cgm[-1, -1] if len(x_cgm.shape) > 1 else x_cgm[-1])
        
        # Asegurar dimensiones correctas
        if len(x_cgm_t.shape) == 2:
            x_cgm_t = x_cgm_t.unsqueeze(0)
        if len(x_other_t.shape) == 1:
            x_other_t = x_other_t.unsqueeze(0)
        
        # Implementación específica por algoritmo
        with torch.no_grad():
            # Crear tensor de contexto
            context_tensor = torch.FloatTensor([
                current_glucose, carb_intake, sleep_quality,
                work_intensity, exercise_intensity
            ]).to(self.device).unsqueeze(0)
            
            # Combinar con x_other
            enhanced_x_other = torch.cat([x_other_t, context_tensor], dim=1)
            
            # Usar el modelo para predecir
            features = self(x_cgm_t, enhanced_x_other)
            action = self.actor(features) if hasattr(self, 'actor') else features
            dose = action.cpu().numpy().flatten()[0]
        
        # Aplicar límites de seguridad
        dose = max(0.0, min(dose, 20.0))
        
        return float(dose)
    
    def evaluate_with_context(self, x_cgm: np.ndarray, x_other: np.ndarray, 
                         carb_intake: np.ndarray,
                         true_doses: np.ndarray = None,
                         sleep_quality: np.ndarray = None,
                         work_intensity: np.ndarray = None,
                         exercise_intensity: np.ndarray = None,
                         simulator = None) -> Dict[str, float]:
        """
        Evalúa el rendimiento del modelo con métricas clínicas usando contexto adicional.
        
        Parámetros:
        -----------
        x_cgm : np.ndarray
            Datos CGM para evaluación (array con mediciones de glucosa recientes)
        x_other : np.ndarray
            Otras características para evaluación
        carb_intake : np.ndarray
            Ingesta de carbohidratos en gramos para cada instancia (obligatorio)
        true_doses : np.ndarray, opcional
            Dosis reales de insulina para comparación
        sleep_quality : np.ndarray, opcional
            Calidad del sueño para cada instancia (escala de 0-10)
        work_intensity : np.ndarray, opcional
            Intensidad del trabajo para cada instancia (escala de 0-10)
        exercise_intensity : np.ndarray, opcional
            Intensidad del ejercicio para cada instancia (escala de 0-10)
        simulator : object, opcional
            Simulador de glucosa para evaluar las métricas clínicas
            
        Retorna:
        --------
        Dict[str, float]
            Diccionario con métricas de evaluación clínica
        """
        # Validar que las dimensiones coincidan
        n_samples = len(x_cgm)
        if len(carb_intake) != n_samples:
            raise ValueError("La cantidad de valores de ingesta de carbohidratos debe coincidir con la cantidad de muestras")
        
        # Inicializar resultados
        predictions = []
        time_in_range = []
        time_below_range = []
        time_above_range = []
        clinical_metrics = {}
        
        # Realizar predicciones para cada muestra
        for i in range(n_samples):
            # Extraer valores de contexto
            context = {
                'carb_intake': float(carb_intake[i])
            }
            
            # Agregar parámetros opcionales si están disponibles
            if sleep_quality is not None:
                context['sleep_quality'] = float(sleep_quality[i])
            if work_intensity is not None:
                context['work_intensity'] = float(work_intensity[i])
            if exercise_intensity is not None:
                context['exercise_intensity'] = float(exercise_intensity[i])
            
            # Predecir dosis
            dose = self.predict_with_context(x_cgm[i:i+1], x_other[i:i+1], **context)
            predictions.append(dose)
            
            # Evaluar con simulador si está disponible
            if simulator is not None:
                # Extraer glucosa inicial del CGM
                initial_glucose = float(x_cgm[i, -1] if len(x_cgm[i].shape) > 0 else x_cgm[i])
                
                # Simular trayectoria de glucosa
                trajectory = simulator.simulate(
                    initial_glucose=initial_glucose,
                    insulin_doses=[dose],
                    carb_intake=[float(carb_intake[i])],
                    duration_hours=6  # Duración estándar para evaluación
                )
                
                # Calcular métricas clínicas
                in_range = np.logical_and(trajectory >= 70.0, trajectory <= 180.0)
                below_range = trajectory < 70.0
                above_range = trajectory > 180.0
                
                time_in_range.append(100.0 * np.mean(in_range))
                time_below_range.append(100.0 * np.mean(below_range))
                time_above_range.append(100.0 * np.mean(above_range))
        
        # Calcular métricas clínicas agregadas
        if simulator is not None:
            clinical_metrics = {
                'time_in_range': float(np.mean(time_in_range)),
                'time_below_range': float(np.mean(time_below_range)),
                'time_above_range': float(np.mean(time_above_range))
            }
        
        # Calcular métricas de error si se proporcionan las dosis reales
        if true_doses is not None:
            mae = np.mean(np.abs(np.array(predictions) - true_doses))
            rmse = np.sqrt(np.mean(np.square(np.array(predictions) - true_doses)))
            clinical_metrics.update({
                'mae': float(mae),
                'rmse': float(rmse)
            })
        
        return clinical_metrics
    
    def update(self, batch: Dict) -> Dict[str, float]:
        """
        Actualiza los parámetros del modelo.
        """
        raise NotImplementedError("Debe ser implementado por las subclases")