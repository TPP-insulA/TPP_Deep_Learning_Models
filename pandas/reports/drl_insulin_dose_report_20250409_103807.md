# Experiment Report: drl_insulin_dose_report
Generated: 2025-04-09 10:38:07

# Data Processing
```python
Starting data processing and preparation...
Metadata after cleaning:
shape: (5, 6)
┌────────────┬────────────────┬────────┬──────┬───────┬──────────────────┐
│ subject_id ┆ age_normalized ┆ gender ┆ race ┆ hbA1c ┆ hbA1c_normalized │
│ ---        ┆ ---            ┆ ---    ┆ ---  ┆ ---   ┆ ---              │
│ i64        ┆ f64            ┆ f64    ┆ f64  ┆ f64   ┆ f64              │
╞════════════╪════════════════╪════════╪══════╪═══════╪══════════════════╡
│ 0          ┆ 0.018182       ┆ 1.0    ┆ 1.0  ┆ 6.3   ┆ 0.2875           │
│ 1          ┆ 0.036364       ┆ 0.0    ┆ 1.0  ┆ 9.9   ┆ 0.7375           │
│ 2          ┆ 0.0            ┆ 0.0    ┆ 1.0  ┆ 8.3   ┆ 0.5375           │
│ 3          ┆ 0.127273       ┆ 0.0    ┆ 1.0  ┆ 7.2   ┆ 0.4              │
│ 4          ┆ 0.163636       ┆ 1.0    ┆ 1.0  ┆ 8.6   ┆ 0.575            │
└────────────┴────────────────┴────────┴──────┴───────┴──────────────────┘

Found Subject files (54):
Subject1.xlsx
Subject10.xlsx
Subject11.xlsx
Subject12.xlsx
Subject13.xlsx
Subject14.xlsx
Subject15.xlsx
Subject16.xlsx
Subject17.xlsx
Subject18.xlsx
Subject19.xlsx
Subject2.xlsx
Subject20.xlsx
Subject21.xlsx
Subject22.xlsx
Subject23.xlsx
Subject24.xlsx
Subject25.xlsx
Subject26.xlsx
Subject27.xlsx
Subject28.xlsx
Subject29.xlsx
Subject3.xlsx
Subject30.xlsx
Subject31.xlsx
Subject32.xlsx
Subject33.xlsx
Subject34.xlsx
Subject35.xlsx
Subject36.xlsx
Subject37.xlsx
Subject38.xlsx
Subject39.xlsx
Subject4.xlsx
Subject40.xlsx
Subject41.xlsx
Subject42.xlsx
Subject43.xlsx
Subject44.xlsx
Subject45.xlsx
Subject46.xlsx
Subject47.xlsx
Subject48.xlsx
Subject49.xlsx
Subject5.xlsx
Subject50.xlsx
Subject51.xlsx
Subject52.xlsx
Subject53.xlsx
Subject54.xlsx
Subject6.xlsx
Subject7.xlsx
Subject8.xlsx
Subject9.xlsx
Muestra de datos procesados combinados:
shape: (5, 15)
┌────────────┬───────────────────┬────────┬───────────┬───┬────────────────┬────────┬──────────────────┬──────────────────────┐
│ subject_id ┆ cgm_window        ┆ normal ┆ carbInput ┆ … ┆ age_normalized ┆ gender ┆ hbA1c_normalized ┆ hbA1c_bg_interaction │
│ ---        ┆ ---               ┆ ---    ┆ ---       ┆   ┆ ---            ┆ ---    ┆ ---              ┆ ---                  │
│ i64        ┆ object            ┆ f64    ┆ f64       ┆   ┆ f64            ┆ f64    ┆ f64              ┆ f64                  │
╞════════════╪═══════════════════╪════════╪═══════════╪═══╪════════════════╪════════╪══════════════════╪══════════════════════╡
│ 21         ┆ [210.8174087      ┆ 1.5    ┆ 20.0      ┆ … ┆ 0.072727       ┆ 0.0    ┆ 0.35             ┆ 77.982459            │
│            ┆ 198.82779304 196… ┆        ┆           ┆   ┆                ┆        ┆                  ┆                      │
│ 21         ┆ [162.85894606     ┆ 2.0    ┆ 20.0      ┆ … ┆ 0.072727       ┆ 0.0    ┆ 0.35             ┆ 91.270949            │
│            ┆ 167.85461925 176… ┆        ┆           ┆   ┆                ┆        ┆                  ┆                      │
│ 21         ┆ [256.77760207     ┆ 2.0    ┆ 20.0      ┆ … ┆ 0.072727       ┆ 0.0    ┆ 0.35             ┆ 108.755805           │
│            ┆ 257.7767367  260… ┆        ┆           ┆   ┆                ┆        ┆                  ┆                      │
│ 21         ┆ [320.72221892     ┆ 3.0    ┆ 20.0      ┆ … ┆ 0.072727       ┆ 0.0    ┆ 0.35             ┆ 86.375189            │
│            ┆ 319.72308428 317… ┆        ┆           ┆   ┆                ┆        ┆                  ┆                      │
│ 21         ┆ [113.90134878     ┆ 1.0    ┆ 20.0      ┆ … ┆ 0.072727       ┆ 0.0    ┆ 0.35             ┆ 31.123044            │
│            ┆ 112.90221414 111… ┆        ┆           ┆   ┆                ┆        ┆                  ┆                      │
└────────────┴───────────────────┴────────┴───────────┴───┴────────────────┴────────┴──────────────────┴──────────────────────┘
Total de muestras: 8649
Schema de df_processed:
Schema([('subject_id', Int64), ('cgm_window', Object), ('normal', Float64), ('carbInput', Float64), ('bgInput', Float64), ('insulinOnBoard', Float64), ('insulinCarbRatio', Float64), ('insulinSensitivityFactor', Float64), ('hour_of_day', Float64), ('time_since_last_bolus', Float64), ('cgm_trend', Float64), ('age_normalized', Float64), ('gender', Float64), ('hbA1c_normalized', Float64), ('hbA1c_bg_interaction', Float64)])
Verificación de NaN en df_final:
shape: (1, 38)
┌───────┬───────┬───────┬───────┬───┬────────────────┬────────┬──────────────────┬──────────────────────┐
│ cgm_0 ┆ cgm_1 ┆ cgm_2 ┆ cgm_3 ┆ … ┆ age_normalized ┆ gender ┆ hbA1c_normalized ┆ hbA1c_bg_interaction │
│ ---   ┆ ---   ┆ ---   ┆ ---   ┆   ┆ ---            ┆ ---    ┆ ---              ┆ ---                  │
│ u32   ┆ u32   ┆ u32   ┆ u32   ┆   ┆ u32            ┆ u32    ┆ u32              ┆ u32                  │
╞═══════╪═══════╪═══════╪═══════╪═══╪════════════════╪════════╪══════════════════╪══════════════════════╡
│ 0     ┆ 0     ┆ 0     ┆ 0     ┆ … ┆ 0              ┆ 0      ┆ 0                ┆ 0                    │
└───────┴───────┴───────┴───────┴───┴────────────────┴────────┴──────────────────┴──────────────────────┘
Preprocesamiento completo en 43.21 segundos
```
# Cross-Validation and Model Training
```python
Fold 1 - Post-split Train y: mean = 1.44, std = 0.60
Fold 1 - Post-split Val y: mean = 1.30, std = 0.48
Fold 1 - Post-split Test y: mean = 0.99, std = 0.62
Fold 1 - Entrenamiento CGM: (2125, 24, 1), Validación CGM: (540, 24, 1), Prueba CGM: (5984, 24, 1)
Fold 1 - Entrenamiento Otros: (2125, 10), Validación Otros: (540, 10), Prueba Otros: (5984, 10)
Fold 1 - Sujetos de prueba: [21 23 24 31 39 43 46 47 48]
Fold 2 - Post-split Train y: mean = 0.98, std = 0.62
Fold 2 - Post-split Val y: mean = 1.15, std = 0.48
Fold 2 - Post-split Test y: mean = 1.41, std = 0.58
Fold 2 - Entrenamiento CGM: (5812, 24, 1), Validación CGM: (172, 24, 1), Prueba CGM: (2665, 24, 1)
Fold 2 - Entrenamiento Otros: (5812, 10), Validación Otros: (172, 10), Prueba Otros: (2665, 10)
Fold 2 - Sujetos de prueba: [29 30 32 36 40 41 45 49]
División de datos completa en 0.07 segundos

Training Fold 1...
```
# Training - Fold 1
```python
Using cpu device
Wrapping the env with a `Monitor` wrapper
Wrapping the env in a DummyVecEnv.
---------------------------------
| rollout/           |          |
|    ep_len_mean     | 1        |
|    ep_rew_mean     | -13.2    |
| time/              |          |
|    fps             | 337      |
|    iterations      | 1        |
|    time_elapsed    | 6        |
|    total_timesteps | 2048     |
---------------------------------
-----------------------------------------
| rollout/                |             |
|    ep_len_mean          | 1           |
|    ep_rew_mean          | -13         |
| time/                   |             |
|    fps                  | 254         |
|    iterations           | 2           |
|    time_elapsed         | 16          |
|    total_timesteps      | 4096        |
| train/                  |             |
|    approx_kl            | 0.026487628 |
|    clip_fraction        | 0.163       |
|    clip_range           | 0.2         |
|    entropy_loss         | -1.38       |
|    explained_variance   | -0.177      |
|    learning_rate        | 0.0003      |
|    loss                 | 45.9        |
|    n_updates            | 10          |
|    policy_gradient_loss | -0.0508     |
|    std                  | 0.933       |
|    value_loss           | 144         |
-----------------------------------------
Eval num_timesteps=5000, episode_reward=-11.87 +/- 0.27
Episode length: 1.00 +/- 0.00
----------------------------------------
| eval/                   |            |
|    mean_ep_length       | 1          |
|    mean_reward          | -11.9      |
| time/                   |            |
|    total_timesteps      | 5000       |
| train/                  |            |
|    approx_kl            | 0.02619838 |
|    clip_fraction        | 0.118      |
|    clip_range           | 0.2        |
|    entropy_loss         | -1.31      |
|    explained_variance   | -1.53      |
|    learning_rate        | 0.0003     |
|    loss                 | 0.587      |
|    n_updates            | 20         |
|    policy_gradient_loss | -0.0261    |
|    std                  | 0.841      |
|    value_loss           | 25.3       |
----------------------------------------
New best mean reward!
---------------------------------
| rollout/           |          |
|    ep_len_mean     | 1        |
|    ep_rew_mean     | -13      |
| time/              |          |
|    fps             | 236      |
|    iterations      | 3        |
|    time_elapsed    | 26       |
|    total_timesteps | 6144     |
---------------------------------
-----------------------------------------
| rollout/                |             |
|    ep_len_mean          | 1           |
|    ep_rew_mean          | -12.8       |
| time/                   |             |
|    fps                  | 229         |
|    iterations           | 4           |
|    time_elapsed         | 35          |
|    total_timesteps      | 8192        |
| train/                  |             |
|    approx_kl            | 0.034907863 |
|    clip_fraction        | 0.134       |
|    clip_range           | 0.2         |
|    entropy_loss         | -1.15       |
|    explained_variance   | -0.31       |
|    learning_rate        | 0.0003      |
|    loss                 | 0.272       |
|    n_updates            | 30          |
|    policy_gradient_loss | -0.0666     |
|    std                  | 0.711       |
|    value_loss           | 1.03        |
-----------------------------------------
Eval num_timesteps=10000, episode_reward=-11.88 +/- 0.16
Episode length: 1.00 +/- 0.00
-----------------------------------------
| eval/                   |             |
|    mean_ep_length       | 1           |
|    mean_reward          | -11.9       |
| time/                   |             |
|    total_timesteps      | 10000       |
| train/                  |             |
|    approx_kl            | 0.026584547 |
|    clip_fraction        | 0.104       |
|    clip_range           | 0.2         |
|    entropy_loss         | -1          |
|    explained_variance   | -0.107      |
|    learning_rate        | 0.0003      |
|    loss                 | 0.193       |
|    n_updates            | 40          |
|    policy_gradient_loss | -0.0602     |
|    std                  | 0.619       |
|    value_loss           | 0.855       |
-----------------------------------------
---------------------------------
| rollout/           |          |
|    ep_len_mean     | 1        |
|    ep_rew_mean     | -12.5    |
| time/              |          |
|    fps             | 222      |
|    iterations      | 5        |
|    time_elapsed    | 46       |
|    total_timesteps | 10240    |
---------------------------------
-----------------------------------------
| rollout/                |             |
|    ep_len_mean          | 1           |
|    ep_rew_mean          | -12.6       |
| time/                   |             |
|    fps                  | 218         |
|    iterations           | 6           |
|    time_elapsed         | 56          |
|    total_timesteps      | 12288       |
| train/                  |             |
|    approx_kl            | 0.028358348 |
|    clip_fraction        | 0.116       |
|    clip_range           | 0.2         |
|    entropy_loss         | -0.87       |
|    explained_variance   | -0.0609     |
|    learning_rate        | 0.0003      |
|    loss                 | 0.224       |
|    n_updates            | 50          |
|    policy_gradient_loss | -0.0625     |
|    std                  | 0.546       |
|    value_loss           | 0.762       |
-----------------------------------------
-----------------------------------------
| rollout/                |             |
|    ep_len_mean          | 1           |
|    ep_rew_mean          | -12.4       |
| time/                   |             |
|    fps                  | 219         |
|    iterations           | 7           |
|    time_elapsed         | 65          |
|    total_timesteps      | 14336       |
| train/                  |             |
|    approx_kl            | 0.021376777 |
|    clip_fraction        | 0.0973      |
|    clip_range           | 0.2         |
|    entropy_loss         | -0.754      |
|    explained_variance   | -0.0654     |
|    learning_rate        | 0.0003      |
|    loss                 | 0.159       |
|    n_updates            | 60          |
|    policy_gradient_loss | -0.0566     |
|    std                  | 0.488       |
|    value_loss           | 0.707       |
-----------------------------------------
Eval num_timesteps=15000, episode_reward=-11.90 +/- 0.26
Episode length: 1.00 +/- 0.00
-----------------------------------------
| eval/                   |             |
|    mean_ep_length       | 1           |
|    mean_reward          | -11.9       |
| time/                   |             |
|    total_timesteps      | 15000       |
| train/                  |             |
|    approx_kl            | 0.020908475 |
|    clip_fraction        | 0.0982      |
|    clip_range           | 0.2         |
|    entropy_loss         | -0.641      |
|    explained_variance   | -0.0956     |
|    learning_rate        | 0.0003      |
|    loss                 | 0.194       |
|    n_updates            | 70          |
|    policy_gradient_loss | -0.0575     |
|    std                  | 0.437       |
|    value_loss           | 0.584       |
-----------------------------------------
---------------------------------
| rollout/           |          |
|    ep_len_mean     | 1        |
|    ep_rew_mean     | -12.5    |
| time/              |          |
|    fps             | 219      |
|    iterations      | 8        |
|    time_elapsed    | 74       |
|    total_timesteps | 16384    |
---------------------------------
-----------------------------------------
| rollout/                |             |
|    ep_len_mean          | 1           |
|    ep_rew_mean          | -12.3       |
| time/                   |             |
|    fps                  | 218         |
|    iterations           | 9           |
|    time_elapsed         | 84          |
|    total_timesteps      | 18432       |
| train/                  |             |
|    approx_kl            | 0.018647341 |
|    clip_fraction        | 0.0991      |
|    clip_range           | 0.2         |
|    entropy_loss         | -0.534      |
|    explained_variance   | -0.075      |
|    learning_rate        | 0.0003      |
|    loss                 | 0.11        |
|    n_updates            | 80          |
|    policy_gradient_loss | -0.0572     |
|    std                  | 0.392       |
|    value_loss           | 0.506       |
-----------------------------------------
Eval num_timesteps=20000, episode_reward=-12.15 +/- 0.41
Episode length: 1.00 +/- 0.00
-----------------------------------------
| eval/                   |             |
|    mean_ep_length       | 1           |
|    mean_reward          | -12.2       |
| time/                   |             |
|    total_timesteps      | 20000       |
| train/                  |             |
|    approx_kl            | 0.018082246 |
|    clip_fraction        | 0.0956      |
|    clip_range           | 0.2         |
|    entropy_loss         | -0.427      |
|    explained_variance   | -0.0299     |
|    learning_rate        | 0.0003      |
|    loss                 | 0.107       |
|    n_updates            | 90          |
|    policy_gradient_loss | -0.0557     |
|    std                  | 0.353       |
|    value_loss           | 0.432       |
-----------------------------------------
---------------------------------
| rollout/           |          |
|    ep_len_mean     | 1        |
|    ep_rew_mean     | -12.3    |
| time/              |          |
|    fps             | 216      |
|    iterations      | 10       |
|    time_elapsed    | 94       |
|    total_timesteps | 20480    |
---------------------------------
```
