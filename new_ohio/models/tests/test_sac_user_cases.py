import numpy as np
import polars as pl
from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import DummyVecEnv
from new_ohio.processing.ohio_polars import simulate_glucose
from new_ohio.models.tests.user_cases import USER_CASES

# Color codes for terminal output
class Colors:
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

MODEL_PATH = 'new_ohio/models/sac_ohiot1dm.zip'

def build_synthetic_df(case):
    # Calcular características estadísticas de las glucosas
    glucoses = np.array(case['glucoses'])
    cgm_mean_24h = np.mean(glucoses)
    cgm_std_24h = np.std(glucoses)
    cgm_median_24h = np.median(glucoses)
    cgm_range_24h = np.max(glucoses) - np.min(glucoses)
    
    # Calcular TIR y otros porcentajes
    tir_lower, tir_upper = 70, 180  # Valores estándar
    time_in_range_24h = np.mean((glucoses >= tir_lower) & (glucoses <= tir_upper))
    hypo_percentage_24h = np.mean(glucoses < tir_lower)
    hyper_percentage_24h = np.mean(glucoses > tir_upper)
    
    # Calcular CV (coeficiente de variación)
    cv_24h = (cgm_std_24h / cgm_mean_24h) if cgm_mean_24h > 0 else 0
    
    # Calcular MAGE (Mean Amplitude of Glucose Excursions)
    glucose_diff = np.diff(glucoses)
    mage_24h = np.mean(np.abs(glucose_diff)) if len(glucose_diff) > 0 else 0
    
    # Calcular tendencia de glucosa
    glucose_trend_24h = np.polyfit(np.arange(len(glucoses)), glucoses, 1)[0]
    
    # Contar episodios de hipo/hiper
    hypo_episodes_24h = np.sum(np.diff((glucoses < tir_lower).astype(int)) == 1)
    hyper_episodes_24h = np.sum(np.diff((glucoses > tir_upper).astype(int)) == 1)
    
    # Calcular características de tiempo
    hour = 12  # Asumimos mediodía para simplificar
    hour_sin = np.sin(2 * np.pi * hour / 24)
    hour_cos = np.cos(2 * np.pi * hour / 24)
    
    # Asegurar que los valores estén en rangos razonables
    cgm_mean_24h = np.clip(cgm_mean_24h, 40, 400)
    cgm_std_24h = np.clip(cgm_std_24h, 0, 100)
    cgm_median_24h = np.clip(cgm_median_24h, 40, 400)
    cgm_range_24h = np.clip(cgm_range_24h, 0, 360)
    time_in_range_24h = np.clip(time_in_range_24h, 0, 1)
    hypo_percentage_24h = np.clip(hypo_percentage_24h, 0, 1)
    hyper_percentage_24h = np.clip(hyper_percentage_24h, 0, 1)
    cv_24h = np.clip(cv_24h, 0, 1)
    mage_24h = np.clip(mage_24h, 0, 100)
    glucose_trend_24h = np.clip(glucose_trend_24h, -50, 50)
    hypo_episodes_24h = np.clip(hypo_episodes_24h, 0, 10)
    hyper_episodes_24h = np.clip(hyper_episodes_24h, 0, 10)
    
    base = {
        'Timestamp': [0],
        'SubjectID': ['test'],
        'bolus': [0.0],
        'hour_sin': [hour_sin],
        'hour_cos': [hour_cos],
        'cgm_mean_24h': [cgm_mean_24h],
        'cgm_std_24h': [cgm_std_24h],
        'time_in_range_24h': [time_in_range_24h],
        'hypo_percentage_24h': [hypo_percentage_24h],
        'hyper_percentage_24h': [hyper_percentage_24h],
        'cv_24h': [cv_24h],
        'mage_24h': [mage_24h],
        'glucose_trend_24h': [glucose_trend_24h],
        'cgm_range_24h': [cgm_range_24h],
        'cgm_median_24h': [cgm_median_24h],
        'hypo_episodes_24h': [hypo_episodes_24h],
        'hyper_episodes_24h': [hyper_episodes_24h]
    }
    return pl.DataFrame(base)

def test_user_cases():
    # Import here to avoid circular dependency
    from new_ohio.models.sac import OhioT1DMEnv
    
    print(f"\n{Colors.HEADER}{Colors.BOLD}Evaluando casos de usuario tipo app con el modelo SAC entrenado:{Colors.ENDC}\n")
    model = SAC.load(MODEL_PATH)
    
    # Cargar estadísticas de normalización
    state_mean, state_std, feature_columns = OhioT1DMEnv.load_normalization_stats('new_ohio/models/normalization_stats.json')
    
    # Imprimir estadísticas de normalización una sola vez
    print(f"{Colors.BLUE}{Colors.BOLD}Estadísticas de normalización:{Colors.ENDC}")
    print(f"{Colors.BLUE}{'='*50}{Colors.ENDC}")
    for i, col in enumerate(feature_columns):
        print(f"{Colors.BLUE}  {col:<20} mean={state_mean[i]:>8.2f}, std={state_std[i]:>8.2f}{Colors.ENDC}")
    print(f"{Colors.BLUE}{'='*50}{Colors.ENDC}\n")
    
    for case in USER_CASES:
        df = build_synthetic_df(case)
        # Usar las estadísticas de normalización guardadas
        env = DummyVecEnv([lambda: OhioT1DMEnv(df, state_mean, state_std)])
        obs = env.reset()
        action, _ = model.predict(obs, deterministic=True)
        
        # Manejar tanto la versión antigua (4 valores) como la nueva (5 valores) de Gym/Gymnasium
        step_result = env.step(action)
        if len(step_result) == 4:  # Versión antigua de Gym
            obs, reward, done, info = step_result
            terminated, truncated = done, False
        else:  # Nueva versión de Gymnasium
            obs, reward, terminated, truncated, info = step_result
        
        print(f"{Colors.BLUE}{Colors.BOLD}Caso: {case['name']}{Colors.ENDC}")
        print(f"{Colors.GREEN}  Glucosas recientes: {case['glucoses']}{Colors.ENDC}")
        print(f"{Colors.YELLOW}  Carbohidratos: {case['carbs']} g{Colors.ENDC}")
        print(f"{Colors.RED}  Dosis predicha: {action[0][0]:.2f} U{Colors.ENDC}")
        # Extraer métricas simuladas relevantes
        next_glucose = simulate_glucose(
            cgm_values=case['glucoses'],
            bolus=float(action[0][0]),
            carbs=case['carbs'],
            basal_rate=case.get('basal_rate', 0.0),
            exercise_intensity=case.get('exercise_intensity', 0.0)
        )
        print(f"{Colors.GREEN}  TIR 2h simulado: {next_glucose.get('simulated_tir_2h', 0.0)*100:.1f}%{Colors.ENDC}")
        print(f"{Colors.YELLOW}  Glucosa a 2h simulada: {next_glucose.get('simulated_glucose_2h', 0.0):.1f} mg/dL{Colors.ENDC}")
        print(f"{Colors.RED}  Recompensa: {reward[0]:.2f}{Colors.ENDC}\n")

if __name__ == "__main__":
    test_user_cases() 