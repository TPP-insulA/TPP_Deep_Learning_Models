# 🧾 Configuraciones de Preprocesamiento de Datos y Modelos PPO

Este documento registra las configuraciones utilizadas para el preprocesamiento de datos y el entrenamiento de modelos PPO.  
Cualquier modificación en el modelo o en el preprocesamiento debe estar acompañada por un nuevo ID y ser registrada aquí.

> **Objetivo**: entrenar agentes PPO capaces de predecir la dosis adecuada de insulina bolus, maximizando el control glucémico postprandial en pacientes con diabetes.

## 📁 Estructura de Archivos

- El archivo `config.py` debe reflejar los IDs actuales de modelo (`MODEL_ID`) y preprocesamiento (`PREPROCESSING_ID`).
- Carpeta `/data/params`: contiene archivos `.json` con los parámetros de estandarización por ID.  
  Ejemplo: `state_standardization_params_0.json` corresponde al preprocesamiento ID 0.
- Carpeta `/data/processed`: contiene datasets `train_all_<ID>.parquet`, `val_all_<ID>.parquet`, `test_all_<ID>.parquet` por preprocesamiento.
- Carpeta `/data/results`: guarda las predicciones generadas por los modelos.  
  Ejemplo: `ppo_predictions_val_0_0.csv` representa modelo ID 0 y preprocesamiento ID 0.

## 🧪 Preprocesamiento de Datos

### ID = 0

**Variables utilizadas como entrada:**

- 24 valores de CGM previos al momento del bolo (2 horas antes, 1 dato cada 5 minutos)
- `carbInput`
- `insulinCarbRatio`
- `bgInput`
- `insulinOnBoard`
- `targetBloodGlucose`

**Configuraciones utilizadas:**

```python
WINDOW_PREV_HOURS = 2
WINDOW_POST_HOURS = 2
IOB_WINDOW_HOURS = 4
SAMPLES_PER_HOUR = 12

PREV_SAMPLES = 24  # 2 horas * 12
POST_SAMPLES = 24
```

La función de cálculo de IOB utilizada fue la de ID = 0.

## 🧠 Modelos PPO Entrenados

Todos los modelos comparten la siguiente estructura:

```python
self.state_cols = [
    *[f"mg/dl_prev_{i+1}" for i in range(PREV_SAMPLES)],
    "carbInput", "insulinCarbRatio", "bgInput", "insulinOnBoard", "targetBloodGlucose"
]
self.action_col = "normal"
self.post_cols = [f"mg/dl_post_{i+1}" for i in range(POST_SAMPLES)]
```

---

## 🧩 Modelos PPO por ID

Cada modelo se entrena con una combinación específica de pasos (`total_timesteps`), hiperparámetros y función de recompensa.

### ID = 0

```python
model.learn(total_timesteps=1000000)
```

**Recompensa:**

```python
if NORMAL_LOW <= avg_mgdl <= NORMAL_HIGH:
    return 1.0 if abs(pred_dose - real_dose) < 1.0 else -0.5
elif avg_mgdl > NORMAL_HIGH:
    return 0.5 if pred_dose > real_dose else -1.0
else:
    return 0.5 if pred_dose < real_dose else -1.0
```

### ID = 1

```python
model.learn(total_timesteps=1000000)
```

**Recompensa:**

```python
DOSE_THRESHOLD = real_dose * 0.10
abs_diff = abs(pred_dose - real_dose)

if NORMAL_LOW <= avg_mgdl <= NORMAL_HIGH:
    return 1.0 if abs_diff < DOSE_THRESHOLD else -0.5
elif avg_mgdl > NORMAL_HIGH:
    if pred_dose > real_dose: return 1.0
    elif abs_diff < DOSE_THRESHOLD: return -1.0
    else: return 2.0
else:
    if pred_dose < real_dose: return 1.0
    elif abs_diff < DOSE_THRESHOLD: return -1.0
    else: return 2.0
```

### ID = 2

```python
model.learn(total_timesteps=1000000)
```

**Recompensa:** igual a ID = 1, pero con penalización más severa ante errores.

```python
DOSE_THRESHOLD = real_dose * 0.10
abs_diff = abs(pred_dose - real_dose)

if NORMAL_LOW <= avg_mgdl <= NORMAL_HIGH:
    return 1.0 if abs_diff < DOSE_THRESHOLD else -0.5
elif avg_mgdl > NORMAL_HIGH:
    if pred_dose > real_dose: return 1.0
    elif abs_diff < DOSE_THRESHOLD: return -1.0
    else: return -2.0
else:
    if pred_dose < real_dose: return 1.0
    elif abs_diff < DOSE_THRESHOLD: return -1.0
    else: return -2.0
```

### ID = 3

```python
model.learn(total_timesteps=100000)
```

**Recompensa:** igual a ID = 2.

### ID = 4

```python
model.learn(total_timesteps=1000000)
```

**Recompensa:** busca acentuar el castigo o premio según severidad de la condición.

```python
DOSE_THRESHOLD = real_dose * 0.10
abs_diff = abs(pred_dose - real_dose)

if NORMAL_LOW <= avg_mgdl <= NORMAL_HIGH:
    return 2.0 if abs_diff < DOSE_THRESHOLD else -2.0
elif avg_mgdl > NORMAL_HIGH:
    if pred_dose > real_dose: return 2.0
    elif abs_diff < DOSE_THRESHOLD: return -2.0
    else: return -3.0
else:
    if pred_dose < real_dose: return 4.0
    elif abs_diff < DOSE_THRESHOLD: return -2.0
    else: return -4.0
```

### ID = 5

```python
model.learn(total_timesteps=300000)
```

**Recompensa:** basada en el error relativo, penalizando más cuanto mayor sea la diferencia entre predicción y realidad.

```python
abs_diff = abs(pred_dose - real_dose)

if NORMAL_LOW <= avg_mgdl <= NORMAL_HIGH:
    return 1.0 - (abs_diff / (real_dose + 1e-5))
elif avg_mgdl > NORMAL_HIGH:
    if pred_dose >= real_dose:
        return 1.0 - (abs_diff / (real_dose + 1e-5))
    else:
        return - (abs_diff / (real_dose + 1e-5))
else:
    if pred_dose < real_dose:
        return 1.0 - (abs_diff / (real_dose + 1e-5))
    else:
        return -2.0
```

### ID = 6

```python
model = PPO(
    "MlpPolicy",
    env,
    verbose=1,
    learning_rate=1e-4,  # más conservador
    n_steps=2048,
    batch_size=256,      # batch más grande
    n_epochs=10,
    gamma=0.99,
    gae_lambda=0.95,
    clip_range=0.2,
    device="cpu"
)
```

```python
model.learn(total_timesteps=500000)
```

**Recompensa:** igual a la del modelo ID = 5.

### ID = 7

Igual que ID = 6 pero con más pasos de entrenamiento.

```python
model.learn(total_timesteps=1000000)
```

**Recompensa:** igual a la del modelo ID = 5.

### ID = 8

Como el ID = 7, pero cambia el espacio de acción a discreto:

```python
self.action_space = Discrete(41)  # 41 acciones: 0.0, 0.5, ..., 20.0
pred_dose = float(action.item()) * 0.5
```

```python
model.learn(total_timesteps=1000000)
```

**Recompensa:** igual a la del modelo ID = 5.

### ID = 9

Extiende la recompensa del ID = 5, con penalización explícita por hipoglucemia e hiperglucemia severas y bonificación por buena glucemia postprandial:

```python
reward = base_reward(...)  # misma lógica que ID 5

if mgdl_post[-1] < 70:
    reward -= 1.0  # hipoglucemia severa
elif mgdl_post[-1] > 300:
    reward -= 0.5  # hiperglucemia grave
elif 70 <= mgdl_post[-1] <= 180:
    reward += 0.5  # buen control postprandial
```

```python
model.learn(total_timesteps=1000000)
```

## 📈 Resultados

Los archivos de predicción generados por los modelos se guardan en:

```
/data/results/ppo_predictions_val_<PREPROCESSING_ID>_<MODEL_ID>.csv
```

**Ejemplos:**

- `ppo_predictions_val_0_0.csv` → modelo ID = 0, preprocesamiento ID = 0
- `ppo_predictions_val_0_9.csv` → modelo ID = 9, preprocesamiento ID = 0

## 🧩 Resumen de Modelos

| Modelo ID | Timesteps | LR   | Batch | Acción          | Recompensa                          |
| --------- | --------- | ---- | ----- | --------------- | ----------------------------------- |
| 0         | 1000000   | 3e-4 | 64    | Continua        | Rango + dosis fija ±1.0             |
| 1         | 1000000   | 3e-4 | 64    | Continua        | ±10% real con ajustes en hipo/hiper |
| 2         | 1000000   | 3e-4 | 64    | Continua        | Penalización severa (-2.0)          |
| 3         | 100000    | 3e-4 | 64    | Continua        | Igual a ID=2                        |
| 4         | 1000000   | 3e-4 | 64    | Continua        | Premios más agresivos               |
| 5         | 300000    | 3e-4 | 64    | Continua        | Error relativo                      |
| 6         | 500000    | 1e-4 | 256   | Continua        | Igual a ID=5                        |
| 7         | 1000000   | 1e-4 | 256   | Continua        | Igual a ID=5                        |
| 8         | 1000000   | 1e-4 | 256   | Discreta (0–20) | Igual a ID=5                        |
| 9         | 1000000   | 1e-4 | 256   | Discreta (0–20) | ID=5 + penalización por extremos    |

---

Este archivo debe mantenerse actualizado con cada nuevo modelo entrenado o cambio en el preprocesamiento.

> **Nota:**  
> - **MAE** = *Mean Absolute Error* (error absoluto medio)  
> - **RMSE** = *Root Mean Squared Error* (raíz del error cuadrático medio)  
> - **% Similares (±10%)** = porcentaje de predicciones cuya diferencia con la dosis real es menor o igual al 10%