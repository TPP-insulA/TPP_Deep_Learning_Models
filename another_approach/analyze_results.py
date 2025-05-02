# %% CELL: Imports and Configuration
import polars as pl
import matplotlib.pyplot as plt
import seaborn as sns
from config import CONFIG, MODEL_ID, PREPROCESSSING_ID
import os
import numpy as np

# Configuración
DOSE_THRESHOLD_PERCENT = 0.10  # 10% threshold for similar dose
folder = CONFIG["processed_data_path"]
model = "ppo"
dataset = "val"
CSV_PATH = os.path.join(folder, f"{model}_predictions_{dataset}_{PREPROCESSSING_ID}_{MODEL_ID}.csv")
print("CSV_PATH:", CSV_PATH)
# Configurar estilo de gráficos
plt.style.use('seaborn-v0_8')
# %matplotlib inline

# %% CELL: Load and Initial Processing
df = pl.read_csv(CSV_PATH)

# Calcular promedio de CGM post-bolo
post_cols = [f"mg/dl_post_{i+1}" for i in range(24)]
df = df.with_columns(
    pl.mean_horizontal(post_cols).alias("avg_mgdl_post")
)

print("Primeras 5 filas con avg_mgdl_post:")
print(df.head(5)[["avg_mgdl_post", "pred_dose", "real_dose"]])

# %% CELL: Glucose Category Classification
# Clasificación usando promedio
df = df.with_columns(
    pl.when(pl.col("avg_mgdl_post") > 180)
    .then(pl.lit("Hiperglucemia"))
    .when(pl.col("avg_mgdl_post") < 70)
    .then(pl.lit("Hipoglucemia"))
    .otherwise(pl.lit("Rango normal"))
    .alias("glucose_category_avg")
)

# Clasificación usando último valor
df = df.with_columns(
    pl.when(pl.col("mg/dl_post_24") > 180)
    .then(pl.lit("Hiperglucemia"))
    .when(pl.col("mg/dl_post_24") < 70)
    .then(pl.lit("Hipoglucemia"))
    .otherwise(pl.lit("Rango normal"))
    .alias("glucose_category_last")
)

# %% CELL: Plot Glucose Categories
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
sns.countplot(data=df.to_pandas(), x="glucose_category_avg", 
              order=["Hipoglucemia", "Rango normal", "Hiperglucemia"], ax=ax1)
ax1.set_title("Distribución usando promedio CGM")
ax1.set_xlabel("Categoría de glucosa")
ax1.set_ylabel("Número de casos")

sns.countplot(data=df.to_pandas(), x="glucose_category_last", 
              order=["Hipoglucemia", "Rango normal", "Hiperglucemia"], ax=ax2)
ax2.set_title("Distribución usando último valor CGM")
ax2.set_xlabel("Categoría de glucosa")
ax2.set_ylabel("Número de casos")
plt.tight_layout()
plt.show()

# %% CELL: Glucose Buckets
def create_glucose_buckets(col_name, new_col_name):
    return pl.when(pl.col(col_name) < 70).then(pl.lit("<70"))\
            .when(pl.col(col_name) < 100).then(pl.lit("70-100"))\
            .when(pl.col(col_name) < 130).then(pl.lit("100-130"))\
            .when(pl.col(col_name) < 170).then(pl.lit("130-170"))\
            .when(pl.col(col_name) < 200).then(pl.lit("170-200"))\
            .when(pl.col(col_name) < 230).then(pl.lit("200-230"))\
            .when(pl.col(col_name) < 260).then(pl.lit("230-260"))\
            .otherwise(pl.lit("260+")).alias(new_col_name)

df = df.with_columns([
    create_glucose_buckets("avg_mgdl_post", "glucose_bucket_avg"),
    create_glucose_buckets("mg/dl_post_24", "glucose_bucket_last")
])

# %% CELL: Dose Comparison
df = df.with_columns(
    pl.when(((pl.col("pred_dose") - pl.col("real_dose")).abs() / pl.col("real_dose")) <= DOSE_THRESHOLD_PERCENT)
    .then(pl.lit("Similar"))
    .when(pl.col("pred_dose") > pl.col("real_dose") * (1 + DOSE_THRESHOLD_PERCENT))
    .then(pl.lit("Mayor"))
    .otherwise(pl.lit("Menor"))
    .alias("dose_comparison")
)

# %% CELL: Plot Dose Comparison - Average CGM
summary_avg = df.group_by(["glucose_category_avg", "dose_comparison"]).agg(
    count=pl.col("dose_comparison").count()
).pivot(
    index="glucose_category_avg",
    columns="dose_comparison",
    values="count",
    aggregate_function="sum"
).fill_null(0)

print("Resumen de predicciones por categoría (promedio CGM):")
print(summary_avg)

summary_melted_avg = summary_avg.to_pandas().melt(id_vars="glucose_category_avg", 
                                               value_vars=["Mayor", "Menor", "Similar"],
                                               var_name="dose_comparison", 
                                               value_name="count")
plt.figure(figsize=(10, 6))
sns.barplot(data=summary_melted_avg, 
            x="glucose_category_avg", 
            y="count", 
            hue="dose_comparison",
            order=["Hipoglucemia", "Rango normal", "Hiperglucemia"],
            hue_order=["Mayor", "Menor", "Similar"])
plt.title("Predicciones de dosis por categoría (promedio CGM)")
plt.xlabel("Categoría de glucosa")
plt.ylabel("Número de casos")
plt.legend(title="Comparación de dosis")
plt.show()

# %% CELL: Plot Dose Comparison - Last CGM
summary_last = df.group_by(["glucose_category_last", "dose_comparison"]).agg(
    count=pl.col("dose_comparison").count()
).pivot(
    index="glucose_category_last",
    columns="dose_comparison",
    values="count",
    aggregate_function="sum"
).fill_null(0)

print("Resumen de predicciones por categoría (último CGM):")
print(summary_last)

summary_melted_last = summary_last.to_pandas().melt(id_vars="glucose_category_last", 
                                                  value_vars=["Mayor", "Menor", "Similar"],
                                                  var_name="dose_comparison", 
                                                  value_name="count")
plt.figure(figsize=(10, 6))
sns.barplot(data=summary_melted_last, 
            x="glucose_category_last", 
            y="count", 
            hue="dose_comparison",
            order=["Hipoglucemia", "Rango normal", "Hiperglucemia"],
            hue_order=["Mayor", "Menor", "Similar"])
plt.title("Predicciones de dosis por categoría (último CGM)")
plt.xlabel("Categoría de glucosa")
plt.ylabel("Número de casos")
plt.legend(title="Comparación de dosis")
plt.show()

# %% CELL: Plot Dose Comparison by Buckets - Average CGM
summary_bucket_avg = df.group_by(["glucose_bucket_avg", "dose_comparison"]).agg(
    count=pl.col("dose_comparison").count()
).pivot(
    index="glucose_bucket_avg",
    columns="dose_comparison",
    values="count",
    aggregate_function="sum"
).fill_null(0)

print("Resumen de predicciones por bucket (promedio CGM):")
print(summary_bucket_avg)

summary_melted_bucket_avg = summary_bucket_avg.to_pandas().melt(id_vars="glucose_bucket_avg", 
                                                             value_vars=["Mayor", "Menor", "Similar"],
                                                             var_name="dose_comparison", 
                                                             value_name="count")
plt.figure(figsize=(12, 6))
sns.barplot(data=summary_melted_bucket_avg, 
            x="glucose_bucket_avg", 
            y="count", 
            hue="dose_comparison",
            order=["<70", "70-100", "100-130", "130-170", "170-200", "200-230", "230-260", "260+"],
            hue_order=["Mayor", "Menor", "Similar"])
plt.title("Predicciones de dosis por bucket (promedio CGM)")
plt.xlabel("Rango de glucosa (mg/dL)")
plt.ylabel("Número de casos")
plt.xticks(rotation=45)
plt.legend(title="Comparación de dosis")
plt.tight_layout()
plt.show()

# %% CELL: Plot Dose Comparison by Buckets - Last CGM
summary_bucket_last = df.group_by(["glucose_bucket_last", "dose_comparison"]).agg(
    count=pl.col("dose_comparison").count()
).pivot(
    index="glucose_bucket_last",
    columns="dose_comparison",
    values="count",
    aggregate_function="sum"
).fill_null(0)

print("Resumen de predicciones por bucket (último CGM):")
print(summary_bucket_last)

summary_melted_bucket_last = summary_bucket_last.to_pandas().melt(id_vars="glucose_bucket_last", 
                                                               value_vars=["Mayor", "Menor", "Similar"],
                                                               var_name="dose_comparison", 
                                                               value_name="count")
plt.figure(figsize=(12, 6))
sns.barplot(data=summary_melted_bucket_last, 
            x="glucose_bucket_last", 
            y="count", 
            hue="dose_comparison",
            order=["<70", "70-100", "100-130", "130-170", "170-200", "200-230", "230-260", "260+"],
            hue_order=["Mayor", "Menor", "Similar"])
plt.title("Predicciones de dosis por bucket (último CGM)")
plt.xlabel("Rango de glucosa (mg/dL)")
plt.ylabel("Número de casos")
plt.xticks(rotation=45)
plt.legend(title="Comparación de dosis")
plt.tight_layout()
plt.show()

# %% CELL: Percentage Difference Analysis
df = df.with_columns(
    ((pl.col("pred_dose") - pl.col("real_dose")) / pl.col("real_dose") * 100).alias("dose_diff_percent")
)

def create_percent_buckets():
    return pl.when(pl.col("dose_diff_percent") < -20).then(pl.lit("<-20%"))\
            .when(pl.col("dose_diff_percent") < -10).then(pl.lit("-20% a -10%"))\
            .when(pl.col("dose_diff_percent") <= 10).then(pl.lit("-10% a 10%"))\
            .when(pl.col("dose_diff_percent") <= 20).then(pl.lit("10% a 20%"))\
            .otherwise(pl.lit(">20%")).alias("dose_diff_bucket")

df = df.with_columns(create_percent_buckets())

# %% CELL: Plot Percentage Differences - Average CGM
for category in ["Hipoglucemia", "Rango normal", "Hiperglucemia"]:
    plt.figure(figsize=(10, 6))
    sns.countplot(data=df.filter(pl.col("glucose_category_avg") == category).to_pandas(),
                  x="dose_diff_bucket",
                  order=["<-20%", "-20% a -10%", "-10% a 10%", "10% a 20%", ">20%"])
    plt.title(f"Diferencias porcentuales - {category} (promedio CGM)")
    plt.xlabel("Diferencia porcentual")
    plt.ylabel("Número de casos")
    plt.xticks(rotation=45)
    plt.show()

# %% CELL: Plot Percentage Differences - Last CGM
for category in ["Hipoglucemia", "Rango normal", "Hiperglucemia"]:
    plt.figure(figsize=(10, 6))
    sns.countplot(data=df.filter(pl.col("glucose_category_last") == category).to_pandas(),
                  x="dose_diff_bucket",
                  order=["<-20%", "-20% a -10%", "-10% a 10%", "10% a 20%", ">20%"])
    plt.title(f"Diferencias porcentuales - {category} (último CGM)")
    plt.xlabel("Diferencia porcentual")
    plt.ylabel("Número de casos")
    plt.xticks(rotation=45)
    plt.show()

# %% CELL: Scatter Plot Comparison
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
sns.scatterplot(data=df.to_pandas(), 
                x="real_dose", 
                y="pred_dose", 
                hue="glucose_category_avg",
                style="glucose_category_avg",
                hue_order=["Hipoglucemia", "Rango normal", "Hiperglucemia"],
                ax=ax1)
ax1.plot([0, 20], [0, 20], 'r--', label="Línea ideal")
ax1.set_title("Predicciones vs. Dosis reales (promedio CGM)")
ax1.set_xlabel("Dosis real (unidades)")
ax1.set_ylabel("Dosis predicha (unidades)")
ax1.legend(title="Categoría")

sns.scatterplot(data=df.to_pandas(), 
                x="real_dose", 
                y="pred_dose", 
                hue="glucose_category_last",
                style="glucose_category_last",
                hue_order=["Hipoglucemia", "Rango normal", "Hiperglucemia"],
                ax=ax2)
ax2.plot([0, 20], [0, 20], 'r--', label="Línea ideal")
ax2.set_title("Predicciones vs. Dosis reales (último CGM)")
ax2.set_xlabel("Dosis real (unidades)")
ax2.set_ylabel("Dosis predicha (unidades)")
ax2.legend(title="Categoría")
plt.tight_layout()
plt.show()

# %% CELL: Additional Analysis - Correlation
# Correlation heatmap between variables
plt.figure(figsize=(10, 8))
correlation_matrix = df.select(["avg_mgdl_post", "mg/dl_post_24", "pred_dose", "real_dose", "reward"]).to_pandas().corr()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0)
plt.title("Matriz de correlación entre variables principales")
plt.tight_layout()
plt.show()

# %% CELL: Statistics
print("Estadísticas del error porcentual de dosis:")
print(df["dose_diff_percent"].describe())

print("Promedio de glucosa por bucket (promedio CGM):")
print(df.group_by("glucose_bucket_avg").agg(pl.col("avg_mgdl_post").mean()))

print("Promedio de glucosa por bucket (último CGM):")
print(df.group_by("glucose_bucket_last").agg(pl.col("mg/dl_post_24").mean()))

print("Recompensa promedio por categoría (promedio CGM):")
print(df.group_by("glucose_category_avg").agg(pl.col("reward").mean()))

print("Recompensa promedio por categoría (último CGM):")
print(df.group_by("glucose_category_last").agg(pl.col("reward").mean()))
# %%
