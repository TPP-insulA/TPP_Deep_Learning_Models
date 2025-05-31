# Experiment Report: drl_insulin_dose_report
Generated: 2025-04-09 10:00:41

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
┌────────────┬─────────────────────────────────┬────────┬───────────┬───┬────────────────┬────────┬──────────────────┬──────────────────────┐
│ subject_id ┆ cgm_window                      ┆ normal ┆ carbInput ┆ … ┆ age_normalized ┆ gender ┆ hbA1c_normalized ┆ hbA1c_bg_interaction │
│ ---        ┆ ---                             ┆ ---    ┆ ---       ┆   ┆ ---            ┆ ---    ┆ ---              ┆ ---                  │
│ i64        ┆ object                          ┆ f64    ┆ f64       ┆   ┆ f64            ┆ f64    ┆ f64              ┆ f64                  │
╞════════════╪═════════════════════════════════╪════════╪═══════════╪═══╪════════════════╪════════╪══════════════════╪══════════════════════╡
│ 21         ┆ [210.8174087  198.82779304 196… ┆ 1.5    ┆ 20.0      ┆ … ┆ 0.072727       ┆ 0.0    ┆ 0.35             ┆ 77.982459            │
│ 21         ┆ [162.85894606 167.85461925 176… ┆ 2.0    ┆ 20.0      ┆ … ┆ 0.072727       ┆ 0.0    ┆ 0.35             ┆ 91.270949            │
│ 21         ┆ [256.77760207 257.7767367  260… ┆ 2.0    ┆ 20.0      ┆ … ┆ 0.072727       ┆ 0.0    ┆ 0.35             ┆ 108.755805           │
│ 21         ┆ [320.72221892 319.72308428 317… ┆ 3.0    ┆ 20.0      ┆ … ┆ 0.072727       ┆ 0.0    ┆ 0.35             ┆ 86.375189            │
│ 21         ┆ [113.90134878 112.90221414 111… ┆ 1.0    ┆ 20.0      ┆ … ┆ 0.072727       ┆ 0.0    ┆ 0.35             ┆ 31.123044            │
└────────────┴─────────────────────────────────┴────────┴───────────┴───┴────────────────┴────────┴──────────────────┴──────────────────────┘
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
Preprocesamiento completo en 78.94 segundos
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
|    ep_rew_mean     | -12.5    |
| time/              |          |
|    fps             | 335      |
|    iterations      | 1        |
|    time_elapsed    | 6        |
|    total_timesteps | 2048     |
---------------------------------
-----------------------------------------
| rollout/                |             |
|    ep_len_mean          | 1           |
|    ep_rew_mean          | -12.4       |
| time/                   |             |
|    fps                  | 257         |
|    iterations           | 2           |
|    time_elapsed         | 15          |
|    total_timesteps      | 4096        |
| train/                  |             |
|    approx_kl            | 0.021775644 |
|    clip_fraction        | 0.138       |
|    clip_range           | 0.2         |
|    entropy_loss         | -1.39       |
|    explained_variance   | -0.271      |
|    learning_rate        | 0.0003      |
|    loss                 | 36.9        |
|    n_updates            | 10          |
|    policy_gradient_loss | -0.0405     |
|    std                  | 0.946       |
|    value_loss           | 127         |
-----------------------------------------
Eval num_timesteps=5000, episode_reward=-11.82 +/- 0.33
Episode length: 1.00 +/- 0.00
-----------------------------------------
| eval/                   |             |
|    mean_ep_length       | 1           |
|    mean_reward          | -11.8       |
| time/                   |             |
|    total_timesteps      | 5000        |
| train/                  |             |
|    approx_kl            | 0.017263584 |
|    clip_fraction        | 0.115       |
|    clip_range           | 0.2         |
|    entropy_loss         | -1.33       |
|    explained_variance   | -2.06       |
|    learning_rate        | 0.0003      |
|    loss                 | 0.197       |
|    n_updates            | 20          |
|    policy_gradient_loss | -0.0235     |
|    std                  | 0.867       |
|    value_loss           | 18.5        |
-----------------------------------------
New best mean reward!
---------------------------------
| rollout/           |          |
|    ep_len_mean     | 1        |
|    ep_rew_mean     | -12.2    |
| time/              |          |
|    fps             | 237      |
|    iterations      | 3        |
|    time_elapsed    | 25       |
|    total_timesteps | 6144     |
---------------------------------
-----------------------------------------
| rollout/                |             |
|    ep_len_mean          | 1           |
|    ep_rew_mean          | -12.3       |
| time/                   |             |
|    fps                  | 231         |
|    iterations           | 4           |
|    time_elapsed         | 35          |
|    total_timesteps      | 8192        |
| train/                  |             |
|    approx_kl            | 0.033123553 |
|    clip_fraction        | 0.194       |
|    clip_range           | 0.2         |
|    entropy_loss         | -1.2        |
|    explained_variance   | -0.531      |
|    learning_rate        | 0.0003      |
|    loss                 | 0.0412      |
|    n_updates            | 30          |
|    policy_gradient_loss | -0.0502     |
|    std                  | 0.764       |
|    value_loss           | 0.514       |
-----------------------------------------
Eval num_timesteps=10000, episode_reward=-11.83 +/- 0.13
Episode length: 1.00 +/- 0.00
-----------------------------------------
| eval/                   |             |
|    mean_ep_length       | 1           |
|    mean_reward          | -11.8       |
| time/                   |             |
|    total_timesteps      | 10000       |
| train/                  |             |
|    approx_kl            | 0.035328038 |
|    clip_fraction        | 0.183       |
|    clip_range           | 0.2         |
|    entropy_loss         | -1.08       |
|    explained_variance   | -0.0997     |
|    learning_rate        | 0.0003      |
|    loss                 | 0.107       |
|    n_updates            | 40          |
|    policy_gradient_loss | -0.0562     |
|    std                  | 0.679       |
|    value_loss           | 0.424       |
-----------------------------------------
---------------------------------
| rollout/           |          |
|    ep_len_mean     | 1        |
|    ep_rew_mean     | -12      |
| time/              |          |
|    fps             | 223      |
|    iterations      | 5        |
|    time_elapsed    | 45       |
|    total_timesteps | 10240    |
---------------------------------
-----------------------------------------
| rollout/                |             |
|    ep_len_mean          | 1           |
|    ep_rew_mean          | -12         |
| time/                   |             |
|    fps                  | 221         |
|    iterations           | 6           |
|    time_elapsed         | 55          |
|    total_timesteps      | 12288       |
| train/                  |             |
|    approx_kl            | 0.028428929 |
|    clip_fraction        | 0.181       |
|    clip_range           | 0.2         |
|    entropy_loss         | -0.974      |
|    explained_variance   | -0.0748     |
|    learning_rate        | 0.0003      |
|    loss                 | -0.00567    |
|    n_updates            | 50          |
|    policy_gradient_loss | -0.053      |
|    std                  | 0.614       |
|    value_loss           | 0.308       |
-----------------------------------------
-----------------------------------------
| rollout/                |             |
|    ep_len_mean          | 1           |
|    ep_rew_mean          | -12         |
| time/                   |             |
|    fps                  | 219         |
|    iterations           | 7           |
|    time_elapsed         | 65          |
|    total_timesteps      | 14336       |
| train/                  |             |
|    approx_kl            | 0.027167233 |
|    clip_fraction        | 0.172       |
|    clip_range           | 0.2         |
|    entropy_loss         | -0.874      |
|    explained_variance   | -0.00437    |
|    learning_rate        | 0.0003      |
|    loss                 | 0.00555     |
|    n_updates            | 60          |
|    policy_gradient_loss | -0.0516     |
|    std                  | 0.554       |
|    value_loss           | 0.259       |
-----------------------------------------
Eval num_timesteps=15000, episode_reward=-11.83 +/- 0.20
Episode length: 1.00 +/- 0.00
-----------------------------------------
| eval/                   |             |
|    mean_ep_length       | 1           |
|    mean_reward          | -11.8       |
| time/                   |             |
|    total_timesteps      | 15000       |
| train/                  |             |
|    approx_kl            | 0.030184895 |
|    clip_fraction        | 0.162       |
|    clip_range           | 0.2         |
|    entropy_loss         | -0.77       |
|    explained_variance   | 0.066       |
|    learning_rate        | 0.0003      |
|    loss                 | 0.00627     |
|    n_updates            | 70          |
|    policy_gradient_loss | -0.0543     |
|    std                  | 0.499       |
|    value_loss           | 0.187       |
-----------------------------------------
---------------------------------
| rollout/           |          |
|    ep_len_mean     | 1        |
|    ep_rew_mean     | -11.9    |
| time/              |          |
|    fps             | 217      |
|    iterations      | 8        |
|    time_elapsed    | 75       |
|    total_timesteps | 16384    |
---------------------------------
-----------------------------------------
| rollout/                |             |
|    ep_len_mean          | 1           |
|    ep_rew_mean          | -11.9       |
| time/                   |             |
|    fps                  | 220         |
|    iterations           | 9           |
|    time_elapsed         | 83          |
|    total_timesteps      | 18432       |
| train/                  |             |
|    approx_kl            | 0.025597999 |
|    clip_fraction        | 0.164       |
|    clip_range           | 0.2         |
|    entropy_loss         | -0.672      |
|    explained_variance   | -0.0491     |
|    learning_rate        | 0.0003      |
|    loss                 | -0.0261     |
|    n_updates            | 80          |
|    policy_gradient_loss | -0.0513     |
|    std                  | 0.453       |
|    value_loss           | 0.163       |
-----------------------------------------
Eval num_timesteps=20000, episode_reward=-11.87 +/- 0.19
Episode length: 1.00 +/- 0.00
-----------------------------------------
| eval/                   |             |
|    mean_ep_length       | 1           |
|    mean_reward          | -11.9       |
| time/                   |             |
|    total_timesteps      | 20000       |
| train/                  |             |
|    approx_kl            | 0.025984466 |
|    clip_fraction        | 0.164       |
|    clip_range           | 0.2         |
|    entropy_loss         | -0.585      |
|    explained_variance   | 0.0839      |
|    learning_rate        | 0.0003      |
|    loss                 | 0.00286     |
|    n_updates            | 90          |
|    policy_gradient_loss | -0.0435     |
|    std                  | 0.418       |
|    value_loss           | 0.145       |
-----------------------------------------
---------------------------------
| rollout/           |          |
|    ep_len_mean     | 1        |
|    ep_rew_mean     | -11.8    |
| time/              |          |
|    fps             | 222      |
|    iterations      | 10       |
|    time_elapsed    | 92       |
|    total_timesteps | 20480    |
---------------------------------
-----------------------------------------
| rollout/                |             |
|    ep_len_mean          | 1           |
|    ep_rew_mean          | -11.8       |
| time/                   |             |
|    fps                  | 225         |
|    iterations           | 11          |
|    time_elapsed         | 100         |
|    total_timesteps      | 22528       |
| train/                  |             |
|    approx_kl            | 0.025636528 |
|    clip_fraction        | 0.156       |
|    clip_range           | 0.2         |
|    entropy_loss         | -0.504      |
|    explained_variance   | -0.0305     |
|    learning_rate        | 0.0003      |
|    loss                 | -0.0248     |
|    n_updates            | 100         |
|    policy_gradient_loss | -0.0441     |
|    std                  | 0.386       |
|    value_loss           | 0.128       |
-----------------------------------------
-----------------------------------------
| rollout/                |             |
|    ep_len_mean          | 1           |
|    ep_rew_mean          | -11.7       |
| time/                   |             |
|    fps                  | 223         |
|    iterations           | 12          |
|    time_elapsed         | 109         |
|    total_timesteps      | 24576       |
| train/                  |             |
|    approx_kl            | 0.026811697 |
|    clip_fraction        | 0.16        |
|    clip_range           | 0.2         |
|    entropy_loss         | -0.426      |
|    explained_variance   | -0.0278     |
|    learning_rate        | 0.0003      |
|    loss                 | -0.0334     |
|    n_updates            | 110         |
|    policy_gradient_loss | -0.0417     |
|    std                  | 0.357       |
|    value_loss           | 0.0998      |
-----------------------------------------
Eval num_timesteps=25000, episode_reward=-11.74 +/- 0.13
Episode length: 1.00 +/- 0.00
-----------------------------------------
| eval/                   |             |
|    mean_ep_length       | 1           |
|    mean_reward          | -11.7       |
| time/                   |             |
|    total_timesteps      | 25000       |
| train/                  |             |
|    approx_kl            | 0.025069095 |
|    clip_fraction        | 0.178       |
|    clip_range           | 0.2         |
|    entropy_loss         | -0.351      |
|    explained_variance   | 0.0265      |
|    learning_rate        | 0.0003      |
|    loss                 | -0.0505     |
|    n_updates            | 120         |
|    policy_gradient_loss | -0.0348     |
|    std                  | 0.332       |
|    value_loss           | 0.0927      |
-----------------------------------------
New best mean reward!
---------------------------------
| rollout/           |          |
|    ep_len_mean     | 1        |
|    ep_rew_mean     | -11.8    |
| time/              |          |
|    fps             | 223      |
|    iterations      | 13       |
|    time_elapsed    | 118      |
|    total_timesteps | 26624    |
---------------------------------
-----------------------------------------
| rollout/                |             |
|    ep_len_mean          | 1           |
|    ep_rew_mean          | -11.7       |
| time/                   |             |
|    fps                  | 222         |
|    iterations           | 14          |
|    time_elapsed         | 128         |
|    total_timesteps      | 28672       |
| train/                  |             |
|    approx_kl            | 0.019820957 |
|    clip_fraction        | 0.167       |
|    clip_range           | 0.2         |
|    entropy_loss         | -0.277      |
|    explained_variance   | -0.0661     |
|    learning_rate        | 0.0003      |
|    loss                 | -0.0325     |
|    n_updates            | 130         |
|    policy_gradient_loss | -0.035      |
|    std                  | 0.308       |
|    value_loss           | 0.0722      |
-----------------------------------------
Eval num_timesteps=30000, episode_reward=-11.89 +/- 0.15
Episode length: 1.00 +/- 0.00
-----------------------------------------
| eval/                   |             |
|    mean_ep_length       | 1           |
|    mean_reward          | -11.9       |
| time/                   |             |
|    total_timesteps      | 30000       |
| train/                  |             |
|    approx_kl            | 0.022972563 |
|    clip_fraction        | 0.177       |
|    clip_range           | 0.2         |
|    entropy_loss         | -0.211      |
|    explained_variance   | 0.0208      |
|    learning_rate        | 0.0003      |
|    loss                 | -0.0195     |
|    n_updates            | 140         |
|    policy_gradient_loss | -0.0304     |
|    std                  | 0.29        |
|    value_loss           | 0.0806      |
-----------------------------------------
---------------------------------
| rollout/           |          |
|    ep_len_mean     | 1        |
|    ep_rew_mean     | -11.7    |
| time/              |          |
|    fps             | 223      |
|    iterations      | 15       |
|    time_elapsed    | 137      |
|    total_timesteps | 30720    |
---------------------------------
-----------------------------------------
| rollout/                |             |
|    ep_len_mean          | 1           |
|    ep_rew_mean          | -11.7       |
| time/                   |             |
|    fps                  | 223         |
|    iterations           | 16          |
|    time_elapsed         | 146         |
|    total_timesteps      | 32768       |
| train/                  |             |
|    approx_kl            | 0.027427837 |
|    clip_fraction        | 0.176       |
|    clip_range           | 0.2         |
|    entropy_loss         | -0.148      |
|    explained_variance   | 0.0559      |
|    learning_rate        | 0.0003      |
|    loss                 | -0.0412     |
|    n_updates            | 150         |
|    policy_gradient_loss | -0.0303     |
|    std                  | 0.272       |
|    value_loss           | 0.0667      |
-----------------------------------------
-----------------------------------------
| rollout/                |             |
|    ep_len_mean          | 1           |
|    ep_rew_mean          | -11.7       |
| time/                   |             |
|    fps                  | 222         |
|    iterations           | 17          |
|    time_elapsed         | 156         |
|    total_timesteps      | 34816       |
| train/                  |             |
|    approx_kl            | 0.027870648 |
|    clip_fraction        | 0.185       |
|    clip_range           | 0.2         |
|    entropy_loss         | -0.0821     |
|    explained_variance   | 0.0285      |
|    learning_rate        | 0.0003      |
|    loss                 | -0.0211     |
|    n_updates            | 160         |
|    policy_gradient_loss | -0.0232     |
|    std                  | 0.254       |
|    value_loss           | 0.0594      |
-----------------------------------------
Eval num_timesteps=35000, episode_reward=-11.97 +/- 0.05
Episode length: 1.00 +/- 0.00
-----------------------------------------
| eval/                   |             |
|    mean_ep_length       | 1           |
|    mean_reward          | -12         |
| time/                   |             |
|    total_timesteps      | 35000       |
| train/                  |             |
|    approx_kl            | 0.025245862 |
|    clip_fraction        | 0.209       |
|    clip_range           | 0.2         |
|    entropy_loss         | -0.0177     |
|    explained_variance   | 0.0434      |
|    learning_rate        | 0.0003      |
|    loss                 | -0.0141     |
|    n_updates            | 170         |
|    policy_gradient_loss | -0.0254     |
|    std                  | 0.239       |
|    value_loss           | 0.055       |
-----------------------------------------
---------------------------------
| rollout/           |          |
|    ep_len_mean     | 1        |
|    ep_rew_mean     | -11.7    |
| time/              |          |
|    fps             | 220      |
|    iterations      | 18       |
|    time_elapsed    | 166      |
|    total_timesteps | 36864    |
---------------------------------
-----------------------------------------
| rollout/                |             |
|    ep_len_mean          | 1           |
|    ep_rew_mean          | -11.7       |
| time/                   |             |
|    fps                  | 221         |
|    iterations           | 19          |
|    time_elapsed         | 175         |
|    total_timesteps      | 38912       |
| train/                  |             |
|    approx_kl            | 0.026756274 |
|    clip_fraction        | 0.205       |
|    clip_range           | 0.2         |
|    entropy_loss         | 0.0417      |
|    explained_variance   | 0.00854     |
|    learning_rate        | 0.0003      |
|    loss                 | -0.0348     |
|    n_updates            | 180         |
|    policy_gradient_loss | -0.0287     |
|    std                  | 0.225       |
|    value_loss           | 0.0476      |
-----------------------------------------
Eval num_timesteps=40000, episode_reward=-11.72 +/- 0.15
Episode length: 1.00 +/- 0.00
-----------------------------------------
| eval/                   |             |
|    mean_ep_length       | 1           |
|    mean_reward          | -11.7       |
| time/                   |             |
|    total_timesteps      | 40000       |
| train/                  |             |
|    approx_kl            | 0.026684554 |
|    clip_fraction        | 0.205       |
|    clip_range           | 0.2         |
|    entropy_loss         | 0.101       |
|    explained_variance   | 0.021       |
|    learning_rate        | 0.0003      |
|    loss                 | -0.0108     |
|    n_updates            | 190         |
|    policy_gradient_loss | -0.0251     |
|    std                  | 0.213       |
|    value_loss           | 0.0491      |
-----------------------------------------
New best mean reward!
---------------------------------
| rollout/           |          |
|    ep_len_mean     | 1        |
|    ep_rew_mean     | -11.7    |
| time/              |          |
|    fps             | 220      |
|    iterations      | 20       |
|    time_elapsed    | 185      |
|    total_timesteps | 40960    |
---------------------------------
-----------------------------------------
| rollout/                |             |
|    ep_len_mean          | 1           |
|    ep_rew_mean          | -11.7       |
| time/                   |             |
|    fps                  | 219         |
|    iterations           | 21          |
|    time_elapsed         | 195         |
|    total_timesteps      | 43008       |
| train/                  |             |
|    approx_kl            | 0.026379144 |
|    clip_fraction        | 0.19        |
|    clip_range           | 0.2         |
|    entropy_loss         | 0.157       |
|    explained_variance   | -0.0113     |
|    learning_rate        | 0.0003      |
|    loss                 | -0.0274     |
|    n_updates            | 200         |
|    policy_gradient_loss | -0.0183     |
|    std                  | 0.202       |
|    value_loss           | 0.0422      |
-----------------------------------------
Eval num_timesteps=45000, episode_reward=-11.83 +/- 0.13
Episode length: 1.00 +/- 0.00
-----------------------------------------
| eval/                   |             |
|    mean_ep_length       | 1           |
|    mean_reward          | -11.8       |
| time/                   |             |
|    total_timesteps      | 45000       |
| train/                  |             |
|    approx_kl            | 0.037123844 |
|    clip_fraction        | 0.223       |
|    clip_range           | 0.2         |
|    entropy_loss         | 0.212       |
|    explained_variance   | 0.012       |
|    learning_rate        | 0.0003      |
|    loss                 | 0.000312    |
|    n_updates            | 210         |
|    policy_gradient_loss | -0.0214     |
|    std                  | 0.19        |
|    value_loss           | 0.0401      |
-----------------------------------------
---------------------------------
| rollout/           |          |
|    ep_len_mean     | 1        |
|    ep_rew_mean     | -11.7    |
| time/              |          |
|    fps             | 222      |
|    iterations      | 22       |
|    time_elapsed    | 202      |
|    total_timesteps | 45056    |
---------------------------------
-----------------------------------------
| rollout/                |             |
|    ep_len_mean          | 1           |
|    ep_rew_mean          | -11.6       |
| time/                   |             |
|    fps                  | 223         |
|    iterations           | 23          |
|    time_elapsed         | 210         |
|    total_timesteps      | 47104       |
| train/                  |             |
|    approx_kl            | 0.040115353 |
|    clip_fraction        | 0.263       |
|    clip_range           | 0.2         |
|    entropy_loss         | 0.267       |
|    explained_variance   | 0.0378      |
|    learning_rate        | 0.0003      |
|    loss                 | 0.00819     |
|    n_updates            | 220         |
|    policy_gradient_loss | -0.0188     |
|    std                  | 0.181       |
|    value_loss           | 0.0422      |
-----------------------------------------
----------------------------------------
| rollout/                |            |
|    ep_len_mean          | 1          |
|    ep_rew_mean          | -11.6      |
| time/                   |            |
|    fps                  | 224        |
|    iterations           | 24         |
|    time_elapsed         | 219        |
|    total_timesteps      | 49152      |
| train/                  |            |
|    approx_kl            | 0.03508485 |
|    clip_fraction        | 0.233      |
|    clip_range           | 0.2        |
|    entropy_loss         | 0.32       |
|    explained_variance   | 0.0335     |
|    learning_rate        | 0.0003     |
|    loss                 | 0.0232     |
|    n_updates            | 230        |
|    policy_gradient_loss | -0.0161    |
|    std                  | 0.171      |
|    value_loss           | 0.0381     |
----------------------------------------
Eval num_timesteps=50000, episode_reward=-11.71 +/- 0.24
Episode length: 1.00 +/- 0.00
----------------------------------------
| eval/                   |            |
|    mean_ep_length       | 1          |
|    mean_reward          | -11.7      |
| time/                   |            |
|    total_timesteps      | 50000      |
| train/                  |            |
|    approx_kl            | 0.03850361 |
|    clip_fraction        | 0.261      |
|    clip_range           | 0.2        |
|    entropy_loss         | 0.372      |
|    explained_variance   | 0.0497     |
|    learning_rate        | 0.0003     |
|    loss                 | 0.000677   |
|    n_updates            | 240        |
|    policy_gradient_loss | -0.00965   |
|    std                  | 0.163      |
|    value_loss           | 0.033      |
----------------------------------------
New best mean reward!
---------------------------------
| rollout/           |          |
|    ep_len_mean     | 1        |
|    ep_rew_mean     | -11.6    |
| time/              |          |
|    fps             | 224      |
|    iterations      | 25       |
|    time_elapsed    | 227      |
|    total_timesteps | 51200    |
---------------------------------
-----------------------------------------
| rollout/                |             |
|    ep_len_mean          | 1           |
|    ep_rew_mean          | -11.6       |
| time/                   |             |
|    fps                  | 226         |
|    iterations           | 26          |
|    time_elapsed         | 235         |
|    total_timesteps      | 53248       |
| train/                  |             |
|    approx_kl            | 0.037971668 |
|    clip_fraction        | 0.265       |
|    clip_range           | 0.2         |
|    entropy_loss         | 0.416       |
|    explained_variance   | 0.0347      |
|    learning_rate        | 0.0003      |
|    loss                 | 0.0491      |
|    n_updates            | 250         |
|    policy_gradient_loss | -0.00244    |
|    std                  | 0.156       |
|    value_loss           | 0.0355      |
-----------------------------------------
Eval num_timesteps=55000, episode_reward=-12.10 +/- 0.16
Episode length: 1.00 +/- 0.00
-----------------------------------------
| eval/                   |             |
|    mean_ep_length       | 1           |
|    mean_reward          | -12.1       |
| time/                   |             |
|    total_timesteps      | 55000       |
| train/                  |             |
|    approx_kl            | 0.043809008 |
|    clip_fraction        | 0.269       |
|    clip_range           | 0.2         |
|    entropy_loss         | 0.461       |
|    explained_variance   | -0.0191     |
|    learning_rate        | 0.0003      |
|    loss                 | 0.0164      |
|    n_updates            | 260         |
|    policy_gradient_loss | -0.0167     |
|    std                  | 0.149       |
|    value_loss           | 0.0323      |
-----------------------------------------
---------------------------------
| rollout/           |          |
|    ep_len_mean     | 1        |
|    ep_rew_mean     | -11.6    |
| time/              |          |
|    fps             | 226      |
|    iterations      | 27       |
|    time_elapsed    | 244      |
|    total_timesteps | 55296    |
---------------------------------
-----------------------------------------
| rollout/                |             |
|    ep_len_mean          | 1           |
|    ep_rew_mean          | -11.6       |
| time/                   |             |
|    fps                  | 226         |
|    iterations           | 28          |
|    time_elapsed         | 253         |
|    total_timesteps      | 57344       |
| train/                  |             |
|    approx_kl            | 0.037664436 |
|    clip_fraction        | 0.265       |
|    clip_range           | 0.2         |
|    entropy_loss         | 0.506       |
|    explained_variance   | 0.102       |
|    learning_rate        | 0.0003      |
|    loss                 | 0.0172      |
|    n_updates            | 270         |
|    policy_gradient_loss | -0.0091     |
|    std                  | 0.143       |
|    value_loss           | 0.0382      |
-----------------------------------------
----------------------------------------
| rollout/                |            |
|    ep_len_mean          | 1          |
|    ep_rew_mean          | -11.6      |
| time/                   |            |
|    fps                  | 226        |
|    iterations           | 29         |
|    time_elapsed         | 262        |
|    total_timesteps      | 59392      |
| train/                  |            |
|    approx_kl            | 0.04370948 |
|    clip_fraction        | 0.281      |
|    clip_range           | 0.2        |
|    entropy_loss         | 0.558      |
|    explained_variance   | 0.076      |
|    learning_rate        | 0.0003     |
|    loss                 | -0.0135    |
|    n_updates            | 280        |
|    policy_gradient_loss | -0.013     |
|    std                  | 0.134      |
|    value_loss           | 0.0288     |
----------------------------------------
Eval num_timesteps=60000, episode_reward=-12.07 +/- 0.27
Episode length: 1.00 +/- 0.00
-----------------------------------------
| eval/                   |             |
|    mean_ep_length       | 1           |
|    mean_reward          | -12.1       |
| time/                   |             |
|    total_timesteps      | 60000       |
| train/                  |             |
|    approx_kl            | 0.047695287 |
|    clip_fraction        | 0.293       |
|    clip_range           | 0.2         |
|    entropy_loss         | 0.609       |
|    explained_variance   | 0.116       |
|    learning_rate        | 0.0003      |
|    loss                 | 0.0498      |
|    n_updates            | 290         |
|    policy_gradient_loss | -0.00799    |
|    std                  | 0.129       |
|    value_loss           | 0.03        |
-----------------------------------------
---------------------------------
| rollout/           |          |
|    ep_len_mean     | 1        |
|    ep_rew_mean     | -11.6    |
| time/              |          |
|    fps             | 226      |
|    iterations      | 30       |
|    time_elapsed    | 271      |
|    total_timesteps | 61440    |
---------------------------------
----------------------------------------
| rollout/                |            |
|    ep_len_mean          | 1          |
|    ep_rew_mean          | -11.6      |
| time/                   |            |
|    fps                  | 226        |
|    iterations           | 31         |
|    time_elapsed         | 280        |
|    total_timesteps      | 63488      |
| train/                  |            |
|    approx_kl            | 0.04989786 |
|    clip_fraction        | 0.28       |
|    clip_range           | 0.2        |
|    entropy_loss         | 0.656      |
|    explained_variance   | 0.107      |
|    learning_rate        | 0.0003     |
|    loss                 | 0.0873     |
|    n_updates            | 300        |
|    policy_gradient_loss | -0.0119    |
|    std                  | 0.123      |
|    value_loss           | 0.0232     |
----------------------------------------
Eval num_timesteps=65000, episode_reward=-11.95 +/- 0.24
Episode length: 1.00 +/- 0.00
-----------------------------------------
| eval/                   |             |
|    mean_ep_length       | 1           |
|    mean_reward          | -11.9       |
| time/                   |             |
|    total_timesteps      | 65000       |
| train/                  |             |
|    approx_kl            | 0.056034043 |
|    clip_fraction        | 0.323       |
|    clip_range           | 0.2         |
|    entropy_loss         | 0.707       |
|    explained_variance   | 0.101       |
|    learning_rate        | 0.0003      |
|    loss                 | 0.0825      |
|    n_updates            | 310         |
|    policy_gradient_loss | -0.0114     |
|    std                  | 0.117       |
|    value_loss           | 0.0273      |
-----------------------------------------
---------------------------------
| rollout/           |          |
|    ep_len_mean     | 1        |
|    ep_rew_mean     | -11.6    |
| time/              |          |
|    fps             | 226      |
|    iterations      | 32       |
|    time_elapsed    | 289      |
|    total_timesteps | 65536    |
---------------------------------
----------------------------------------
| rollout/                |            |
|    ep_len_mean          | 1          |
|    ep_rew_mean          | -11.6      |
| time/                   |            |
|    fps                  | 225        |
|    iterations           | 33         |
|    time_elapsed         | 299        |
|    total_timesteps      | 67584      |
| train/                  |            |
|    approx_kl            | 0.06759174 |
|    clip_fraction        | 0.33       |
|    clip_range           | 0.2        |
|    entropy_loss         | 0.756      |
|    explained_variance   | 0.148      |
|    learning_rate        | 0.0003     |
|    loss                 | 0.0575     |
|    n_updates            | 320        |
|    policy_gradient_loss | -0.00401   |
|    std                  | 0.111      |
|    value_loss           | 0.0281     |
----------------------------------------
----------------------------------------
| rollout/                |            |
|    ep_len_mean          | 1          |
|    ep_rew_mean          | -11.6      |
| time/                   |            |
|    fps                  | 225        |
|    iterations           | 34         |
|    time_elapsed         | 309        |
|    total_timesteps      | 69632      |
| train/                  |            |
|    approx_kl            | 0.06343221 |
|    clip_fraction        | 0.344      |
|    clip_range           | 0.2        |
|    entropy_loss         | 0.793      |
|    explained_variance   | 0.133      |
|    learning_rate        | 0.0003     |
|    loss                 | 0.0305     |
|    n_updates            | 330        |
|    policy_gradient_loss | 0.00763    |
|    std                  | 0.108      |
|    value_loss           | 0.029      |
----------------------------------------
Eval num_timesteps=70000, episode_reward=-12.22 +/- 0.17
Episode length: 1.00 +/- 0.00
-----------------------------------------
| eval/                   |             |
|    mean_ep_length       | 1           |
|    mean_reward          | -12.2       |
| time/                   |             |
|    total_timesteps      | 70000       |
| train/                  |             |
|    approx_kl            | 0.075107574 |
|    clip_fraction        | 0.348       |
|    clip_range           | 0.2         |
|    entropy_loss         | 0.823       |
|    explained_variance   | 0.13        |
|    learning_rate        | 0.0003      |
|    loss                 | 0.029       |
|    n_updates            | 340         |
|    policy_gradient_loss | 0.00168     |
|    std                  | 0.105       |
|    value_loss           | 0.0222      |
-----------------------------------------
---------------------------------
| rollout/           |          |
|    ep_len_mean     | 1        |
|    ep_rew_mean     | -11.6    |
| time/              |          |
|    fps             | 224      |
|    iterations      | 35       |
|    time_elapsed    | 318      |
|    total_timesteps | 71680    |
---------------------------------
----------------------------------------
| rollout/                |            |
|    ep_len_mean          | 1          |
|    ep_rew_mean          | -11.6      |
| time/                   |            |
|    fps                  | 224        |
|    iterations           | 36         |
|    time_elapsed         | 328        |
|    total_timesteps      | 73728      |
| train/                  |            |
|    approx_kl            | 0.06401284 |
|    clip_fraction        | 0.352      |
|    clip_range           | 0.2        |
|    entropy_loss         | 0.853      |
|    explained_variance   | 0.0774     |
|    learning_rate        | 0.0003     |
|    loss                 | 0.0277     |
|    n_updates            | 350        |
|    policy_gradient_loss | 0.00994    |
|    std                  | 0.102      |
|    value_loss           | 0.0237     |
----------------------------------------
Eval num_timesteps=75000, episode_reward=-12.05 +/- 0.45
Episode length: 1.00 +/- 0.00
----------------------------------------
| eval/                   |            |
|    mean_ep_length       | 1          |
|    mean_reward          | -12        |
| time/                   |            |
|    total_timesteps      | 75000      |
| train/                  |            |
|    approx_kl            | 0.07045666 |
|    clip_fraction        | 0.357      |
|    clip_range           | 0.2        |
|    entropy_loss         | 0.877      |
|    explained_variance   | 0.185      |
|    learning_rate        | 0.0003     |
|    loss                 | 0.0248     |
|    n_updates            | 360        |
|    policy_gradient_loss | 0.000995   |
|    std                  | 0.0999     |
|    value_loss           | 0.0242     |
----------------------------------------
---------------------------------
| rollout/           |          |
|    ep_len_mean     | 1        |
|    ep_rew_mean     | -11.5    |
| time/              |          |
|    fps             | 224      |
|    iterations      | 37       |
|    time_elapsed    | 337      |
|    total_timesteps | 75776    |
---------------------------------
---------------------------------------
| rollout/                |           |
|    ep_len_mean          | 1         |
|    ep_rew_mean          | -11.6     |
| time/                   |           |
|    fps                  | 224       |
|    iterations           | 38        |
|    time_elapsed         | 346       |
|    total_timesteps      | 77824     |
| train/                  |           |
|    approx_kl            | 0.0893822 |
|    clip_fraction        | 0.368     |
|    clip_range           | 0.2       |
|    entropy_loss         | 0.904     |
|    explained_variance   | 0.188     |
|    learning_rate        | 0.0003    |
|    loss                 | 0.0185    |
|    n_updates            | 370       |
|    policy_gradient_loss | -0.00534  |
|    std                  | 0.0962    |
|    value_loss           | 0.0278    |
---------------------------------------
----------------------------------------
| rollout/                |            |
|    ep_len_mean          | 1          |
|    ep_rew_mean          | -11.5      |
| time/                   |            |
|    fps                  | 224        |
|    iterations           | 39         |
|    time_elapsed         | 356        |
|    total_timesteps      | 79872      |
| train/                  |            |
|    approx_kl            | 0.06300467 |
|    clip_fraction        | 0.387      |
|    clip_range           | 0.2        |
|    entropy_loss         | 0.936      |
|    explained_variance   | 0.194      |
|    learning_rate        | 0.0003     |
|    loss                 | 0.162      |
|    n_updates            | 380        |
|    policy_gradient_loss | 0.0109     |
|    std                  | 0.0934     |
|    value_loss           | 0.0208     |
----------------------------------------
Eval num_timesteps=80000, episode_reward=-12.11 +/- 0.22
Episode length: 1.00 +/- 0.00
----------------------------------------
| eval/                   |            |
|    mean_ep_length       | 1          |
|    mean_reward          | -12.1      |
| time/                   |            |
|    total_timesteps      | 80000      |
| train/                  |            |
|    approx_kl            | 0.13685896 |
|    clip_fraction        | 0.393      |
|    clip_range           | 0.2        |
|    entropy_loss         | 0.96       |
|    explained_variance   | 0.136      |
|    learning_rate        | 0.0003     |
|    loss                 | 0.0584     |
|    n_updates            | 390        |
|    policy_gradient_loss | -0.0137    |
|    std                  | 0.0918     |
|    value_loss           | 0.0178     |
----------------------------------------
---------------------------------
| rollout/           |          |
|    ep_len_mean     | 1        |
|    ep_rew_mean     | -11.6    |
| time/              |          |
|    fps             | 224      |
|    iterations      | 40       |
|    time_elapsed    | 365      |
|    total_timesteps | 81920    |
---------------------------------
----------------------------------------
| rollout/                |            |
|    ep_len_mean          | 1          |
|    ep_rew_mean          | -11.6      |
| time/                   |            |
|    fps                  | 224        |
|    iterations           | 41         |
|    time_elapsed         | 373        |
|    total_timesteps      | 83968      |
| train/                  |            |
|    approx_kl            | 0.08964573 |
|    clip_fraction        | 0.382      |
|    clip_range           | 0.2        |
|    entropy_loss         | 0.972      |
|    explained_variance   | 0.209      |
|    learning_rate        | 0.0003     |
|    loss                 | 0.0411     |
|    n_updates            | 400        |
|    policy_gradient_loss | 0.00474    |
|    std                  | 0.0911     |
|    value_loss           | 0.0211     |
----------------------------------------
Eval num_timesteps=85000, episode_reward=-12.31 +/- 0.45
Episode length: 1.00 +/- 0.00
-----------------------------------------
| eval/                   |             |
|    mean_ep_length       | 1           |
|    mean_reward          | -12.3       |
| time/                   |             |
|    total_timesteps      | 85000       |
| train/                  |             |
|    approx_kl            | 0.111175135 |
|    clip_fraction        | 0.365       |
|    clip_range           | 0.2         |
|    entropy_loss         | 1           |
|    explained_variance   | 0.196       |
|    learning_rate        | 0.0003      |
|    loss                 | 0.0141      |
|    n_updates            | 410         |
|    policy_gradient_loss | -0.00113    |
|    std                  | 0.0872      |
|    value_loss           | 0.0186      |
-----------------------------------------
---------------------------------
| rollout/           |          |
|    ep_len_mean     | 1        |
|    ep_rew_mean     | -11.5    |
| time/              |          |
|    fps             | 224      |
|    iterations      | 42       |
|    time_elapsed    | 382      |
|    total_timesteps | 86016    |
---------------------------------
----------------------------------------
| rollout/                |            |
|    ep_len_mean          | 1          |
|    ep_rew_mean          | -11.6      |
| time/                   |            |
|    fps                  | 224        |
|    iterations           | 43         |
|    time_elapsed         | 392        |
|    total_timesteps      | 88064      |
| train/                  |            |
|    approx_kl            | 0.07206388 |
|    clip_fraction        | 0.371      |
|    clip_range           | 0.2        |
|    entropy_loss         | 1.03       |
|    explained_variance   | 0.201      |
|    learning_rate        | 0.0003     |
|    loss                 | 0.0925     |
|    n_updates            | 420        |
|    policy_gradient_loss | 0.00431    |
|    std                  | 0.0852     |
|    value_loss           | 0.022      |
----------------------------------------
Eval num_timesteps=90000, episode_reward=-12.05 +/- 0.49
Episode length: 1.00 +/- 0.00
----------------------------------------
| eval/                   |            |
|    mean_ep_length       | 1          |
|    mean_reward          | -12.1      |
| time/                   |            |
|    total_timesteps      | 90000      |
| train/                  |            |
|    approx_kl            | 0.07676299 |
|    clip_fraction        | 0.395      |
|    clip_range           | 0.2        |
|    entropy_loss         | 1.05       |
|    explained_variance   | 0.173      |
|    learning_rate        | 0.0003     |
|    loss                 | 0.00553    |
|    n_updates            | 430        |
|    policy_gradient_loss | 0.000285   |
|    std                  | 0.0836     |
|    value_loss           | 0.0215     |
----------------------------------------
---------------------------------
| rollout/           |          |
|    ep_len_mean     | 1        |
|    ep_rew_mean     | -11.6    |
| time/              |          |
|    fps             | 223      |
|    iterations      | 44       |
|    time_elapsed    | 402      |
|    total_timesteps | 90112    |
---------------------------------
----------------------------------------
| rollout/                |            |
|    ep_len_mean          | 1          |
|    ep_rew_mean          | -11.6      |
| time/                   |            |
|    fps                  | 223        |
|    iterations           | 45         |
|    time_elapsed         | 412        |
|    total_timesteps      | 92160      |
| train/                  |            |
|    approx_kl            | 0.08538673 |
|    clip_fraction        | 0.416      |
|    clip_range           | 0.2        |
|    entropy_loss         | 1.07       |
|    explained_variance   | 0.22       |
|    learning_rate        | 0.0003     |
|    loss                 | 0.0579     |
|    n_updates            | 440        |
|    policy_gradient_loss | 0.00392    |
|    std                  | 0.0823     |
|    value_loss           | 0.0228     |
----------------------------------------
-----------------------------------------
| rollout/                |             |
|    ep_len_mean          | 1           |
|    ep_rew_mean          | -11.6       |
| time/                   |             |
|    fps                  | 223         |
|    iterations           | 46          |
|    time_elapsed         | 422         |
|    total_timesteps      | 94208       |
| train/                  |             |
|    approx_kl            | 0.089385495 |
|    clip_fraction        | 0.373       |
|    clip_range           | 0.2         |
|    entropy_loss         | 1.09        |
|    explained_variance   | 0.284       |
|    learning_rate        | 0.0003      |
|    loss                 | 0.00928     |
|    n_updates            | 450         |
|    policy_gradient_loss | 0.00563     |
|    std                  | 0.0803      |
|    value_loss           | 0.019       |
-----------------------------------------
Eval num_timesteps=95000, episode_reward=-12.15 +/- 0.32
Episode length: 1.00 +/- 0.00
-----------------------------------------
| eval/                   |             |
|    mean_ep_length       | 1           |
|    mean_reward          | -12.2       |
| time/                   |             |
|    total_timesteps      | 95000       |
| train/                  |             |
|    approx_kl            | 0.088851064 |
|    clip_fraction        | 0.393       |
|    clip_range           | 0.2         |
|    entropy_loss         | 1.12        |
|    explained_variance   | 0.218       |
|    learning_rate        | 0.0003      |
|    loss                 | 0.0185      |
|    n_updates            | 460         |
|    policy_gradient_loss | 0.00108     |
|    std                  | 0.0783      |
|    value_loss           | 0.0174      |
-----------------------------------------
---------------------------------
| rollout/           |          |
|    ep_len_mean     | 1        |
|    ep_rew_mean     | -11.6    |
| time/              |          |
|    fps             | 222      |
|    iterations      | 47       |
|    time_elapsed    | 431      |
|    total_timesteps | 96256    |
---------------------------------
-----------------------------------------
| rollout/                |             |
|    ep_len_mean          | 1           |
|    ep_rew_mean          | -11.5       |
| time/                   |             |
|    fps                  | 223         |
|    iterations           | 48          |
|    time_elapsed         | 440         |
|    total_timesteps      | 98304       |
| train/                  |             |
|    approx_kl            | 0.094063796 |
|    clip_fraction        | 0.421       |
|    clip_range           | 0.2         |
|    entropy_loss         | 1.15        |
|    explained_variance   | 0.228       |
|    learning_rate        | 0.0003      |
|    loss                 | 0.0156      |
|    n_updates            | 470         |
|    policy_gradient_loss | 0.0239      |
|    std                  | 0.0753      |
|    value_loss           | 0.0181      |
-----------------------------------------
Eval num_timesteps=100000, episode_reward=-12.29 +/- 0.14
Episode length: 1.00 +/- 0.00
-----------------------------------------
| eval/                   |             |
|    mean_ep_length       | 1           |
|    mean_reward          | -12.3       |
| time/                   |             |
|    total_timesteps      | 100000      |
| train/                  |             |
|    approx_kl            | 0.086265236 |
|    clip_fraction        | 0.431       |
|    clip_range           | 0.2         |
|    entropy_loss         | 1.18        |
|    explained_variance   | 0.251       |
|    learning_rate        | 0.0003      |
|    loss                 | 0.0661      |
|    n_updates            | 480         |
|    policy_gradient_loss | 0.0133      |
|    std                  | 0.0733      |
|    value_loss           | 0.0155      |
-----------------------------------------
---------------------------------
| rollout/           |          |
|    ep_len_mean     | 1        |
|    ep_rew_mean     | -11.6    |
| time/              |          |
|    fps             | 223      |
|    iterations      | 49       |
|    time_elapsed    | 449      |
|    total_timesteps | 100352   |
---------------------------------
Predicción basada en reglas completa en 0.00 segundos
Fold 1 - PPO Test - MAE: 0.32, RMSE: 0.44, R²: 0.51
Fold 1 - PPO Train - MAE: 0.08, RMSE: 0.14, R²: 0.94
Fold 1 - PPO Val - MAE: 0.48, RMSE: 0.55, R²: -0.29
Fold 1 - Rules Test - MAE: 0.99, RMSE: 1.17, R²: -2.52
```
# Feature Importance - Fold 1
