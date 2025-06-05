import numpy as np
import pandas as pd
import polars as pl
from typing import Dict, List, Any, Callable, Optional, Tuple, Union
import matplotlib.pyplot as plt
import os

from validation.simulator import GlucoseSimulator
from validation.metrics import evaluate_glucose_control
from constants.constants import LOWER_BOUND_NORMAL_GLUCOSE_RANGE, UPPER_BOUND_NORMAL_GLUCOSE_RANGE

def validate_dosing_model_pl(
    model: Any,
    test_data: pl.DataFrame,
    patient_params: Dict[str, Dict[str, float]],
    output_dir: str = "validation_results",
    visualize: bool = True
) -> Dict[str, Dict[str, float]]:
    """
    Valida un modelo de dosificación de insulina basado en el impacto en los niveles de glucosa.
    
    Parámetros:
    -----------
    model : Any
        Modelo entrenado que implementa predict_with_context
    test_data : pl.DataFrame
        Datos de prueba con columnas para CGM, carbohidratos, etc. (polars DataFrame)
    patient_params : Dict[str, Dict[str, float]]
        Parámetros específicos por paciente (sensibilidad, ratio, etc.)
    output_dir : str, opcional
        Directorio para guardar resultados (default: "validation_results")
    visualize : bool, opcional
        Si generar visualizaciones (default: True)
        
    Retorna:
    --------
    Dict[str, Dict[str, float]]
        Métricas de control glucémico por paciente
    """
    results = {}
    patient_ids = test_data["subject_id"].unique().to_list()
    
    for patient_id in patient_ids:
        print(f"Validando paciente {patient_id}...")
        
        # Filtrar datos para este paciente
        patient_data_pl = test_data.filter(pl.col("subject_id") == patient_id)
        
        # Ordenar datos por timestamp
        patient_data_pl = patient_data_pl.sort(by="timestamp")
        
        # Obtener parámetros específicos del paciente o usar valores por defecto
        if patient_id in patient_params:
            params = patient_params[patient_id]
        else:
            params = {
                "insulin_sensitivity": 50,
                "carb_ratio": 10,
                "basal_glucose_impact": 20,
                "insulin_duration_hours": 4
            }
        
        # Crear simulador específico para este paciente
        simulator = GlucoseSimulator(**params)
        
        # Para cada punto de inicio en los datos
        patient_metrics_list = []
        
        # Convertir a pandas solo para esta sección o usar operaciones nativas de polars
        patient_data = patient_data_pl.to_pandas()
        
        for i in range(0, len(patient_data) - 48, 48):  # Bloques de 4 horas (48 muestras de 5 min)
            # Extraer ventana de datos para predecir
            window = patient_data.iloc[i:i+24]  # 2 horas para predecir
            
            # Obtener datos CGM y otros valores necesarios
            current_cgm = window["glucose"].values
            carb_intake = window["carb_intake"].max() if "carb_intake" in window.columns else 0
            iob = window["insulin_on_board"].values[-1] if "insulin_on_board" in window.columns else 0
            
            # Completar el procesamiento de la ventana y predicción igual que antes...
            # Preparar datos para predicción
            x_cgm = np.expand_dims(current_cgm, axis=0)  # Forma [1, time_steps]
            x_other = np.array([[carb_intake, iob]])  # Forma [1, features]
            
            # Predecir dosis con el modelo
            context = {
                "carb_intake": carb_intake,
                "iob": iob,
                "objective_glucose": 100.0
            }
            predicted_dose = model.predict_with_context(x_cgm, x_other, **context)
            
            # Datos para simulación
            initial_glucose = current_cgm[-1]
            insulin_doses = [predicted_dose]
            carb_intakes = [carb_intake]
            timestamps = [0]  # Tiempo relativo en horas
            
            # Simular efecto de la dosis en glucosa
            glucose_trajectory = simulator.predict_glucose_trajectory(
                initial_glucose, 
                insulin_doses,
                carb_intakes,
                timestamps,
                prediction_horizon=4  # 4 horas
            )
            
            # Evaluar métricas
            metrics = evaluate_glucose_control(glucose_trajectory)
            patient_metrics_list.append(metrics)
            
            # Visualizar si se solicita
            if visualize and i % 240 == 0:  # Visualizar cada 20 horas
                plt.figure(figsize=(10, 6))
                time_hours = np.arange(0, len(glucose_trajectory) * 5 / 60, 5 / 60)
                
                plt.plot(time_hours, glucose_trajectory, 'b-', label='Glucosa Predicha')
                plt.axhline(y=70, color='r', linestyle='--', label='Límite Inferior (70 mg/dL)')
                plt.axhline(y=180, color='r', linestyle='--', label='Límite Superior (180 mg/dL)')
                plt.axvline(x=0, color='g', linestyle='--', label=f'Dosis: {predicted_dose:.2f}U')
                
                plt.title(f'Paciente {patient_id} - Simulación de Glucosa')
                plt.xlabel('Tiempo (horas)')
                plt.ylabel('Glucosa (mg/dL)')
                plt.legend()
                plt.grid(True)
                
                # Añadir métricas al gráfico
                txt = (f"TIR: {metrics['time_in_range']:.1f}%\n"
                        f"Hypo: {metrics['time_below_range']:.1f}%\n"
                        f"Hyper: {metrics['time_above_range']:.1f}%")
                plt.text(0.02, 0.02, txt, transform=plt.gca().transAxes, 
                        bbox=dict(facecolor='white', alpha=0.8))
                
                plt.savefig(os.path.join(output_dir, f'patient_{patient_id}_sim_{i}.png'))
                plt.close()
        
        # Agregar métricas de este paciente
        if patient_metrics_list:
            # Calcular promedios de todas las métricas
            avg_metrics = {}
            for key in patient_metrics_list[0].keys():
                avg_metrics[key] = np.mean([m[key] for m in patient_metrics_list])
            
            results[patient_id] = avg_metrics

def validate_dosing_model_pd(
    model: Any,
    test_data: pd.DataFrame,
    patient_params: Dict[str, Dict[str, float]],
    output_dir: str = "validation_results",
    visualize: bool = True
) -> Dict[str, Dict[str, float]]:
    """
    Valida un modelo de dosificación de insulina basado en el impacto en los niveles de glucosa.
    
    Parámetros:
    -----------
    model : Any
        Modelo entrenado que implementa predict_with_context
    test_data : pd.DataFrame
        Datos de prueba con columnas para CGM, carbohidratos, etc. (pandas DataFrame)
    patient_params : Dict[str, Dict[str, float]]
        Parámetros específicos por paciente (sensibilidad, ratio, etc.)
    output_dir : str, opcional
        Directorio para guardar resultados (default: "validation_results")
    visualize : bool, opcional
        Si generar visualizaciones (default: True)
        
    Retorna:
    --------
    Dict[str, Dict[str, float]]
        Métricas de control glucémico por paciente
    """
    results = {}
    for patient_id, patient_data in test_data.groupby("subject_id"):
        print(f"Validando paciente {patient_id}...")
        
        # Obtener parámetros específicos del paciente o usar valores por defecto
        if patient_id in patient_params:
            params = patient_params[patient_id]
        else:
            params = {
                "insulin_sensitivity": 50,
                "carb_ratio": 10,
                "basal_glucose_impact": 20,
                "insulin_duration_hours": 4
            }
        
        # Crear simulador específico para este paciente
        simulator = GlucoseSimulator(**params)
        
        # Ordenar datos por timestamp
        patient_data = patient_data.sort_values("timestamp")
        
        # Para cada punto de inicio en los datos
        patient_metrics_list = []
        
        for i in range(0, len(patient_data) - 48, 48):  # Bloques de 4 horas (48 muestras de 5 min)
            # Extraer ventana de datos para predecir
            window = patient_data.iloc[i:i+24]  # 2 horas para predecir
            
            # Obtener datos CGM y otros valores necesarios
            current_cgm = window["glucose"].values
            carb_intake = window["carb_intake"].max() if "carb_intake" in window.columns else 0
            iob = window["insulin_on_board"].values[-1] if "insulin_on_board" in window.columns else 0
            
            # Preparar datos para predicción
            x_cgm = np.expand_dims(current_cgm, axis=0)  # Forma [1, time_steps]
            x_other = np.array([[carb_intake, iob]])  # Forma [1, features]
            
            # Predecir dosis con el modelo
            context = {
                "carb_intake": carb_intake,
                "iob": iob,
                "objective_glucose": 100.0
            }
            predicted_dose = model.predict_with_context(x_cgm, x_other, **context)
            
            # Datos para simulación
            initial_glucose = current_cgm[-1]
            insulin_doses = [predicted_dose]
            carb_intakes = [carb_intake]
            timestamps = [0]  # Tiempo relativo en horas
            
            # Simular efecto de la dosis en glucosa
            glucose_trajectory = simulator.predict_glucose_trajectory(
                initial_glucose, 
                insulin_doses,
                carb_intakes,
                timestamps,
                prediction_horizon=4  # 4 horas
            )
            
            # Evaluar métricas
            metrics = evaluate_glucose_control(glucose_trajectory)
            patient_metrics_list.append(metrics)
            
            # Visualizar si se solicita
            if visualize and i % 240 == 0:  # Visualizar cada 20 horas
                plt.figure(figsize=(10, 6))
                time_hours = np.arange(0, len(glucose_trajectory) * 5 / 60, 5 / 60)
                
                plt.plot(time_hours, glucose_trajectory, 'b-', label='Glucosa Predicha')
                plt.axhline(y=70, color='r', linestyle='--', label='Límite Inferior (70 mg/dL)')
                plt.axhline(y=180, color='r', linestyle='--', label='Límite Superior (180 mg/dL)')
                plt.axvline(x=0, color='g', linestyle='--', label=f'Dosis: {predicted_dose:.2f}U')
                
                plt.title(f'Paciente {patient_id} - Simulación de Glucosa')
                plt.xlabel('Tiempo (horas)')
                plt.ylabel('Glucosa (mg/dL)')
                plt.legend()
                plt.grid(True)
                
                # Añadir métricas al gráfico
                txt = (f"TIR: {metrics['time_in_range']:.1f}%\n"
                        f"Hypo: {metrics['time_below_range']:.1f}%\n"
                        f"Hyper: {metrics['time_above_range']:.1f}%")
                plt.text(0.02, 0.02, txt, transform=plt.gca().transAxes, 
                        bbox=dict(facecolor='white', alpha=0.8))
                
                plt.savefig(os.path.join(output_dir, f'patient_{patient_id}_sim_{i}.png'))
                plt.close()
        
        # Agregar métricas de este paciente
        if patient_metrics_list:
            # Calcular promedios de todas las métricas
            avg_metrics = {}
            for key in patient_metrics_list[0].keys():
                avg_metrics[key] = np.mean([m[key] for m in patient_metrics_list])
            
            results[patient_id] = avg_metrics
    return results

def validate_dosing_model(
    model: Any,
    test_data: Union[pd.DataFrame, pl.DataFrame],
    patient_params: Dict[str, Dict[str, float]],
    output_dir: str = "validation_results",
    visualize: bool = True
) -> Dict[str, Dict[str, float]]:
    """
    Valida un modelo de dosificación de insulina basado en el impacto en los niveles de glucosa.
    
    Parámetros:
    -----------
    model : Any
        Modelo entrenado que implementa predict_with_context
    test_data : Union[pd.DataFrame, pl.DataFrame]
        Datos de prueba con columnas para CGM, carbohidratos, etc. (pandas o polars)
    patient_params : Dict[str, Dict[str, float]]
        Parámetros específicos por paciente (sensibilidad, ratio, etc.)
    output_dir : str, opcional
        Directorio para guardar resultados (default: "validation_results")
    visualize : bool, opcional
        Si generar visualizaciones (default: True)
        
    Retorna:
    --------
    Dict[str, Dict[str, float]]
        Métricas de control glucémico por paciente
    """
    os.makedirs(output_dir, exist_ok=True)
    
    results = {}
    
    # Determinar si estamos trabajando con polars o pandas
    is_polars = isinstance(test_data, pl.DataFrame)

    results = validate_dosing_model_pl(model, test_data, patient_params, output_dir, visualize) if is_polars else validate_dosing_model_pd(model, test_data.to_pandas(), patient_params, output_dir, visualize)
   
    # Calcular y guardar métricas globales
    global_metrics = {}
    for metric in results[list(results.keys())[0]].keys():
        global_metrics[metric] = np.mean([r[metric] for r in results.values()])
    
    results["global"] = global_metrics
    
    # Guardar resultados en CSV
    results_df = pd.DataFrame.from_dict(results, orient='index')
    results_df.to_csv(os.path.join(output_dir, "validation_metrics.csv"))
    
    # Generar gráfico comparativo de tiempo en rango
    if visualize:
        plt.figure(figsize=(12, 6))
        
        # Extraer TIR para cada paciente
        patient_ids = [pid for pid in results.keys() if pid != "global"]
        tir_values = [results[pid]["time_in_range"] for pid in patient_ids]
        
        # Añadir TIR global
        patient_ids.append("Global")
        tir_values.append(results["global"]["time_in_range"])
        
        # Crear gráfico
        bars = plt.bar(patient_ids, tir_values, color='skyblue')
        bars[-1].set_color('navy')  # Destacar la barra global
        
        plt.axhline(y=70, color='r', linestyle='--', label='Objetivo Mínimo (70%)')
        plt.title('Tiempo en Rango por Paciente')
        plt.xlabel('ID de Paciente')
        plt.ylabel('Tiempo en Rango (%)')
        plt.ylim(0, 100)
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.tight_layout()
        
        plt.savefig(os.path.join(output_dir, "time_in_range_comparison.png"))
        plt.close()
    
    return results

def validate_model_with_simulator(model: Any, test_data: Dict[str, np.array], simulator: GlucoseSimulator) -> Dict[str, float]:
    """
    Valida la precisión de un modelo de dosificación de insulina simulando el impacto en los niveles de glucosa.
    
    Parámetros:
    -----------
    model : Any
        Modelo entrenado que implementa predict_with_context
    test_data : Dict[str, np.ndarray]
        Datos de prueba con claves 'x_cgm' y 'x_other' (numpy arrays)
    simulator : GlucoseSimulator
        Simulador de dinámica de glucosa para validar dosis de insulina
        
    Retorna:
    --------
    Dict[str, float]
        Métricas de control glucémico como tiempo en rango, eventos de hipoglucemia e hiperglucemia
    """
    time_in_range_percentages = []
    
    for i in range(len(test_data['x_cgm'])):
        # Obtener glucosa inicial y otros datos contextuales
        initial_glucose = test_data['x_cgm'][i][-1][-1]  # última lectura de glucosa
        carb_intake = test_data['x_other'][i][0]  # Assuming carb intake is first feature
        
        # Get model's predicted dose
        predicted_dose = model.predict(test_data['x_cgm'][i:i+1], test_data['x_other'][i:i+1])[0]
        
        # Simulate glucose trajectory for next 6 hours with 5-min intervals
        glucose_trajectory = simulator.predict_glucose_trajectory(
            initial_glucose=initial_glucose,
            insulin_doses=[predicted_dose],
            carb_intakes=[carb_intake],
            timestamps=[0],  # Dose given at time 0
            prediction_horizon=6  # Simulate 6 hours ahead
        )
        
        # Calculate time in range
        in_range = np.logical_and(
            glucose_trajectory >= LOWER_BOUND_NORMAL_GLUCOSE_RANGE,
            glucose_trajectory <= UPPER_BOUND_NORMAL_GLUCOSE_RANGE
        )
        time_in_range = np.mean(in_range) * 100  # Percentage
        time_in_range_percentages.append(time_in_range)
    
    return {
        'mean_time_in_range': np.mean(time_in_range_percentages),
        'hypo_events': np.sum(glucose_trajectory < LOWER_BOUND_NORMAL_GLUCOSE_RANGE),
        'hyper_events': np.sum(glucose_trajectory > UPPER_BOUND_NORMAL_GLUCOSE_RANGE)
    }