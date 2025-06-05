import numpy as np
from typing import Tuple, List, Dict, Any, Optional

class GlucoseSimulator:
    """
    Simulador de dinámica de glucosa para validar dosis de insulina.
    
    Implementa un modelo fisiológico simplificado para predecir el efecto
    de la insulina y los carbohidratos en la glucosa sanguínea.
    
    Parámetros:
    -----------
    insulin_sensitivity : float, opcional
        Factor de sensibilidad a la insulina (mg/dL por unidad) (default: 50)
    carb_ratio : float, opcional
        Ratio insulina:carbohidratos (gramos por unidad) (default: 10)
    basal_glucose_impact : float, opcional
        Aumento de glucosa basal por hora sin insulina (default: 20)
    insulin_duration_hours : float, opcional
        Duración de acción de la insulina en horas (default: 4)
    """
    def __init__(
        self,
        insulin_sensitivity: float = 50,
        carb_ratio: float = 10,
        basal_glucose_impact: float = 20,
        insulin_duration_hours: float = 4
    ) -> None:
        self.insulin_sensitivity = insulin_sensitivity  # mg/dL por unidad
        self.carb_ratio = carb_ratio  # gramos por unidad
        self.basal_glucose_impact = basal_glucose_impact  # mg/dL por hora
        self.insulin_duration_hours = insulin_duration_hours  # horas
        
        # Parámetros internos para modelado
        self.insulin_decay = np.log(2) / (insulin_duration_hours / 2)  # Vida media
        
    def predict_glucose_trajectory(
        self,
        initial_glucose: float,
        insulin_doses: List[float],
        carb_intakes: List[float],
        timestamps: List[float],
        prediction_horizon: int = 12
    ) -> np.ndarray:
        """
        Predice la trayectoria de glucosa basada en dosis de insulina y carbohidratos.
        
        Parámetros:
        -----------
        initial_glucose : float
            Glucosa inicial en mg/dL
        insulin_doses : List[float]
            Lista de dosis de insulina en unidades
        carb_intakes : List[float]
            Lista de ingestas de carbohidratos en gramos
        timestamps : List[float]
            Tiempos relativos en horas (0 = inicio)
        prediction_horizon : int, opcional
            Horas a predecir después del último evento (default: 12)
            
        Retorna:
        --------
        np.ndarray
            Trayectoria de glucosa predicha cada 5 minutos
        """
        # Tiempo total en horas
        total_duration = max(timestamps) + prediction_horizon
        
        # Puntos de tiempo para predicción (cada 5 minutos)
        time_points = np.arange(0, total_duration, 5/60)
        
        # Inicializar trayectoria de glucosa
        glucose_trajectory = np.zeros_like(time_points)
        glucose_trajectory[0] = initial_glucose
        
        # Para cada punto de tiempo
        for i in range(1, len(time_points)):
            t = time_points[i]
            dt = time_points[i] - time_points[i-1]  # Diferencia de tiempo en horas
            
            # Efecto de insulina activa (IOB)
            insulin_effect = 0
            for dose, dose_time in zip(insulin_doses, timestamps):
                if t > dose_time:
                    # Modelo de acción de insulina: efecto máximo a las 2 horas, luego decae
                    time_since_dose = t - dose_time
                    if time_since_dose < self.insulin_duration_hours:
                        if time_since_dose < 2:
                            # Fase creciente (0-2 horas)
                            effect_fraction = time_since_dose / 2
                        else:
                            # Fase decreciente (2-4 horas)
                            effect_fraction = 1 - (time_since_dose - 2) / (self.insulin_duration_hours - 2)
                        
                        # Efecto de la insulina en mg/dL
                        effect = dose * self.insulin_sensitivity * effect_fraction * dt
                        insulin_effect += effect
            
            # Efecto de carbohidratos activos (COB)
            carb_effect = 0
            for carbs, carb_time in zip(carb_intakes, timestamps):
                if t > carb_time:
                    # Modelo de absorción de carbohidratos: efecto máximo a la hora, dura 3 horas
                    time_since_intake = t - carb_time
                    if time_since_intake < 3:
                        if time_since_intake < 1:
                            # Fase creciente (0-1 hora)
                            effect_fraction = time_since_intake
                        else:
                            # Fase decreciente (1-3 horas)
                            effect_fraction = 1 - (time_since_intake - 1) / 2
                        
                        # Conversión carbohidratos a glucosa (mg/dL)
                        # Aproximadamente 1g de carbohidratos eleva 5 mg/dL para un adulto promedio
                        effect = carbs * 5 * effect_fraction * dt
                        carb_effect += effect
            
            # Efecto de producción de glucosa basal
            basal_effect = self.basal_glucose_impact * dt
            
            # Actualizar nivel de glucosa
            glucose_trajectory[i] = glucose_trajectory[i-1] + carb_effect - insulin_effect + basal_effect
            
            # Limitar valores mínimos (no puede ser menor a 40 mg/dL fisiológicamente)
            glucose_trajectory[i] = max(40, glucose_trajectory[i])
        
        return glucose_trajectory