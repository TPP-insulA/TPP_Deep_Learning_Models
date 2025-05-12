import pandas as pd
import os
from config import CONFIG, PREPROCESSSING_ID, MODEL_ID

pre_ids = list(range(PREPROCESSSING_ID + 1))
model_ids = list(range(MODEL_ID + 1))

def bucket_analysis(df, category):
    """
    Devuelve proporciones (%) de errores por bucket para una categoría de glucemia.
    """
    df_cat = df[df["glucose_category"] == category].copy()
    df_cat["error_percent"] = 100 * (df_cat["pred_dose"] - df_cat["real_dose"]) / (df_cat["real_dose"] + 1e-5)

    def bucket(e):
        if e < -20: return "<-20%"
        elif e < -10: return "-20% a -10%"
        elif e <= 10: return "-10% a 10%"
        elif e <= 20: return "10% a 20%"
        else: return ">20%"

    df_cat["bucket"] = df_cat["error_percent"].apply(bucket)
    return df_cat["bucket"].value_counts(normalize=True).sort_index() * 100


def analyze_model(pre_id, model_id):
    folder = CONFIG["processed_data_path"]
    path = os.path.join(folder, f"ppo_predictions_val_{pre_id}_{model_id}.csv")

    if not os.path.exists(path):
        print(f"❌ No se encontró: {path}")
        return None

    print(f"\n🔍 Evaluando PRE={pre_id} - MODEL={model_id}")
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

    # Evaluación clínica
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

    # Clasificación de categoría de glucosa
    df["glucose_category"] = pd.cut(
        df[post_col],
        bins=[-float("inf"), 70, 180, float("inf")],
        labels=["Hypo", "Normal", "Hyper"]
    )

    # Análisis por buckets
    buckets_hyper = bucket_analysis(df, "Hyper")
    buckets_hypo = bucket_analysis(df, "Hypo")
    buckets_normal = bucket_analysis(df, "Normal")

    return {
        "PRE_ID": pre_id,
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
        # Buckets por categoría
        **{f"[Hyper] {k}": round(v, 2) for k, v in buckets_hyper.items()},
        **{f"[Normal] {k}": round(v, 2) for k, v in buckets_normal.items()},
        **{f"[Hypo] {k}": round(v, 2) for k, v in buckets_hypo.items()},
    }


# Evaluación total
results = []
for pre_id in pre_ids:
    for model_id in model_ids:
        res = analyze_model(pre_id, model_id)
        if res:
            results.append(res)

# Tabla final
df = pd.DataFrame(results)
df["avg_correction_score"] = df[["% corr hyper (>180)", "% corr hypo (<70)"]].mean(axis=1)

df_sorted = df.sort_values("avg_correction_score", ascending=False)

# Mostrar resumen
print("\n🩺 Comparación clínica + buckets de error (%):\n")
print(df_sorted[[
    "PRE_ID", "MODEL_ID",
    "% corr hyper (>180)", "% corr hypo (<70)", "avg_correction_score",
    "[Hyper] <-20%", "[Hyper] >20%",
    "[Hypo] >20%",
    "[Normal] >20%"
]].to_string(index=False))

top = df_sorted.iloc[0]
print(f"\n✅ Mejor modelo clínico: PRE={int(top.PRE_ID)} - MODEL={int(top.MODEL_ID)} (score={top.avg_correction_score:.2f})")
