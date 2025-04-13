# %% CELL: Imports
# Importar librerías
import polars as pl
import matplotlib.pyplot as plt
import seaborn as sns
from config import CONFIG
import os

# Configuración
DOSE_THRESHOLD = 1.0  # Umbral para considerar dosis "similar"
folder = CONFIG["processed_data_path"]
model = "ppo"
dataset = "val"
CSV_PATH = os.path.join(folder, f"{model}_predictions_{dataset}.csv")

# Configurar estilo de gráficos
plt.style.use('seaborn')  # Estilo limpio para los gráficos
# %matplotlib inline
# %% CELL: Load Data

df = pl.read_csv(CSV_PATH)

# Calcular promedio de CGM post-bolo
post_cols = [f"mg/dl_post_{i+1}" for i in range(24)]
df = df.with_columns(
    pl.mean_horizontal(post_cols).alias("avg_mgdl_post")
)

# Mostrar primeras filas
print("Primeras 5 filas con avg_mgdl_post:")
print(df.head(5)[["avg_mgdl_post", "pred_dose", "real_dose"]])

# %% CELL: Classify cases

# Clasificar en hiperglucemia, hipoglucemia, rango normal
df = df.with_columns(
    pl.when(pl.col("avg_mgdl_post") > 180)
    .then(pl.lit("Hiperglucemia"))
    .when(pl.col("avg_mgdl_post") < 70)
    .then(pl.lit("Hipoglucemia"))
    .otherwise(pl.lit("Rango normal"))
    .alias("glucose_category")
)

# Contar casos por categoría
category_counts = df["glucose_category"].value_counts()
print("Distribución de categorías de glucosa:")
print(category_counts)

# Gráfico de distribución
plt.figure(figsize=(8, 6))
sns.countplot(data=df.to_pandas(), x="glucose_category", order=["Hipoglucemia", "Rango normal", "Hiperglucemia"])
plt.title("Distribución de casos por categoría de glucosa")
plt.xlabel("Categoría de glucosa")
plt.ylabel("Número de casos")
plt.show()

# %% CELL: Compare doses

# Determinar si pred_dose es mayor, menor o similar a real_dose
df = df.with_columns(
    pl.when((pl.col("pred_dose") - pl.col("real_dose")).abs() <= DOSE_THRESHOLD)
    .then(pl.lit("Similar"))
    .when(pl.col("pred_dose") > pl.col("real_dose") + DOSE_THRESHOLD)
    .then(pl.lit("Mayor"))
    .otherwise(pl.lit("Menor"))
    .alias("dose_comparison")
)

# Resumen por categoría y comparación
summary = df.group_by(["glucose_category", "dose_comparison"]).agg(
    count=pl.col("dose_comparison").count()
).pivot(
    index="glucose_category",
    columns="dose_comparison",
    values="count",
    aggregate_function="sum"
).fill_null(0)

print("Resumen de predicciones por categoría de glucosa:")
print(summary)

# Gráfico de barras apiladas
summary_melted = summary.to_pandas().melt(id_vars="glucose_category", 
                                         value_vars=["Mayor", "Menor", "Similar"],
                                         var_name="dose_comparison", 
                                         value_name="count")
plt.figure(figsize=(10, 6))
sns.barplot(data=summary_melted, 
            x="glucose_category", 
            y="count", 
            hue="dose_comparison",
            order=["Hipoglucemia", "Rango normal", "Hiperglucemia"],
            hue_order=["Mayor", "Menor", "Similar"])
plt.title("Predicciones de dosis por categoría de glucosa")
plt.xlabel("Categoría de glucosa")
plt.ylabel("Número de casos")
plt.legend(title="Comparación de dosis")
plt.show()

# %% CELL: pred dose vs real dose

# Scatter plot de pred_dose vs real_dose
plt.figure(figsize=(12, 8))
sns.scatterplot(data=df.to_pandas(), 
                x="real_dose", 
                y="pred_dose", 
                hue="glucose_category", 
                style="glucose_category",
                hue_order=["Hipoglucemia", "Rango normal", "Hiperglucemia"])
plt.plot([0, 20], [0, 20], 'r--', label="Línea ideal")  # Línea y=x
plt.title("Predicciones de dosis vs. Dosis reales por categoría")
plt.xlabel("Dosis real (unidades)")
plt.ylabel("Dosis predicha (unidades)")
plt.legend(title="Categoría de glucosa")
plt.show()

# %% CELL: Extra data

# Estadísticas de error de dosis
df = df.with_columns(
    (pl.col("pred_dose") - pl.col("real_dose")).abs().alias("dose_error")
)
print("Estadísticas del error absoluto de dosis:")
print(df["dose_error"].describe())

# Promedio de glucosa por categoría
print("Promedio de glucosa post-bolo por categoría:")
print(df.group_by("glucose_category").agg(pl.col("avg_mgdl_post").mean()))

# Recompensa por categoría
print("Recompensa promedio por categoría:")
print(df.group_by("glucose_category").agg(pl.col("reward").mean()))

# %%
