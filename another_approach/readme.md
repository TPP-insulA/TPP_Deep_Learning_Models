# Cofiguraciones de preprocesamiento de datos y de modelos entrenados

En config modificar ID de modelo y ID de preprocesamiento cuando se hagan cambios con uno o con el otro.
Luego documentar aca datos utiles sobre los cambios.

En la carpeta /data se encuentran datos sobre los modelos y el preprocesamiento.

En /data/params se encontraran archivos json con los parametros de las estandarizaciones (incluir otros archivos con otros parametros?).
Por ejemplo, la estandarizacion que se utilizo para el preprocesamiento con ID 0 tiene sus datos en "state_standardization_params_0.json".

En /data/processed se encuentran 3 archivos con los datasets de train, val y test por cada preprocesamiento utilizado.
Por ejemplo, para el ID 0 de preprocesamiento tenemos "train_all_0.parquet", "test_all_0.parquet" y "val_all_0.parquet".

## Preprocesamiento de datos

### ID = 0:

Las columnas que se utilizaron para predecir normal fueron:
- los 24 datos previos de CGM al momento de inyectarse el bolo de insulina, tomando 2 horas antes y asumiendo que hay un dato de CGM cada 5 minutos.
- carbInput
- insulinCarbRatio
- bgInput
- insulinOnBoard
- targetBloodGlucose

Para el calculo de iob se utilizo la funcion con ID 0 de calculo de iob.

Parametros de CONFIG:

WINDOW_PREV_HOURS = 2  # Ventana previa de 2 horas (parametrizable)
WINDOW_POST_HOURS = 2  # Ventana posterior de 2 horas (parametrizable)
IOB_WINDOW_HOURS = 4   # Ventana de 4 horas para insulinOnBoard
SAMPLES_PER_HOUR = 12  # 1 dato cada 5 min = 12 datos por hora
PREV_SAMPLES = WINDOW_PREV_HOURS * SAMPLES_PER_HOUR  # 24 datos previos
POST_SAMPLES = WINDOW_POST_HOURS * SAMPLES_PER_HOUR  # 24 datos posteriores

## Modelos entrenados

### ID = 0:

Hiperparametros:

model = PPO(
        "MlpPolicy",
        env,
        verbose=1,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        device="cpu"
    )

Pasos:

model.learn(total_timesteps=500000)

Funcion de recompensa:

def _calculate_reward(self, pred_dose, real_dose, mgdl_post):
        avg_mgdl = np.mean(mgdl_post)
        
        NORMAL_LOW = 70
        NORMAL_HIGH = 180
        DOSE_THRESHOLD = 1.0
        
        if NORMAL_LOW <= avg_mgdl <= NORMAL_HIGH:
            if abs(pred_dose - real_dose) < DOSE_THRESHOLD:
                return 1.0
            else:
                return -0.5
        elif avg_mgdl > NORMAL_HIGH:
            if pred_dose > real_dose:
                return 0.5
            else:
                return -1.0
        else:
            if pred_dose < real_dose:
                return 0.5
            else:
                return -1.0


## Resultados:

### ID de modelo 0 y ID de preprocesamiento 0:

Predicciones para dataset de validacion guardado en ppo_predictions_val_0_0.csv.

(TODO: Incluir aca graficos, tablas o datos que muestren como le fue al modelo, y asi poder comparar con otras combinaciones de modelos y preprocesamientos).

