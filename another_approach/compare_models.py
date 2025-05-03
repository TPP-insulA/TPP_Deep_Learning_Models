import pandas as pd
import os
from config import CONFIG, PREPROCESSSING_ID

MODEL_IDS = [4, 5, 6, 7, 8, 9, 10, 11, 12]


def analyze_model(model_id):
    folder = CONFIG["processed_data_path"]
    path = os.path.join(folder, f"ppo_predictions_val_{PREPROCESSSING_ID}_{model_id}.csv")

    print(f"\n🔍 Analyzing model {model_id}...")
    print(f"📂 File path: {path}")

    if not os.path.exists(path):
        print(f"⚠️ File not found for MODEL_ID = {model_id}")
        return None

    df = pd.read_csv(path)

    # Errores clásicos
    df["perc_error"] = (df["pred_dose"] - df["real_dose"]) / (df["real_dose"] + 1e-5)
    df["abs_error"] = (df["pred_dose"] - df["real_dose"]).abs()

    mae = df["abs_error"].mean()
    rmse = ((df["pred_dose"] - df["real_dose"]) ** 2).mean() ** 0.5
    corr = df["pred_dose"].corr(df["real_dose"])
    pct_similar = (df["perc_error"].abs() < 0.1).mean() * 100
    pct_within_1u = (df["abs_error"] <= 1.0).mean() * 100
    pct_overdose = (df["perc_error"] > 0.2).mean() * 100
    pct_underdose = (df["perc_error"] < -0.2).mean() * 100

    # Evaluación clínica (según glucemia post)
    post_col = "mg/dl_post_24" if "mg/dl_post_24" in df.columns else None
    if not post_col:
        post_cols = [col for col in df.columns if col.startswith("mg/dl_post_")]
        df["avg_mgdl_post"] = df[post_cols].mean(axis=1)
        post_col = "avg_mgdl_post"

    df["correction_hyper"] = ((df[post_col] > 180) & (df["pred_dose"] > df["real_dose"])).astype(int)
    df["correction_hypo"] = ((df[post_col] < 70) & (df["pred_dose"] < df["real_dose"])).astype(int)

    total_hyper = (df[post_col] > 180).sum()
    total_hypo = (df[post_col] < 70).sum()
    pct_corr_hyper = df["correction_hyper"].sum() / total_hyper * 100 if total_hyper > 0 else None
    pct_corr_hypo = df["correction_hypo"].sum() / total_hypo * 100 if total_hypo > 0 else None

    return {
        "MODEL_ID": model_id,
        "MAE": mae,
        "RMSE": rmse,
        "Correlation (pred/real)": corr,
        "% Similar (±10%)": pct_similar,
        "% within 1U": pct_within_1u,
        "% overdose (>+20%)": pct_overdose,
        "% underdose (<-20%)": pct_underdose,
        "% corr hyper (>180)": pct_corr_hyper,
        "% corr hypo (<70)": pct_corr_hypo,
    }


# Run model comparisons
results = []
for mid in MODEL_IDS:
    res = analyze_model(mid)
    if res:
        results.append(res)

# Si no hay datos, salir
if not results:
    print("❌ No se encontraron predicciones para ningún modelo.")
    exit()

df = pd.DataFrame(results).round(3)

# 🏥 Evaluación clínica: modelo que mejor corrige hiperglucemia e hipoglucemia
df["avg_correction_score"] = df[["% corr hyper (>180)", "% corr hypo (<70)"]].mean(axis=1)

best_correction_model = df.loc[df["avg_correction_score"].idxmax(), "MODEL_ID"]
# Mostrar solo métricas clínicas (criterio Pedro)
print("\n🩺 Comparación de modelos según corrección clínica:\n")
print(df[["MODEL_ID", "% corr hyper (>180)", "% corr hypo (<70)", "avg_correction_score"]]
      .sort_values(by="avg_correction_score", ascending=False)
      .to_string(index=False))

print(f"\n✅ Según criterios clínicos (corrección en hipo/hiper), el mejor modelo es el {int(best_correction_model)}.")