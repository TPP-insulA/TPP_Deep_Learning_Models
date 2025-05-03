import pandas as pd
import os
from config import CONFIG, PREPROCESSSING_ID

MODEL_IDS = [4, 5, 6, 7, 8, 9]


def analyze_model(model_id):
    folder = CONFIG["processed_data_path"]
    path = os.path.join(folder, f"ppo_predictions_val_{PREPROCESSSING_ID}_{model_id}.csv")

    print(f"\n🔍 Analyzing model {model_id}...")
    print(f"📂 File path: {path}")

    if not os.path.exists(path):
        print(f"⚠️ File not found for MODEL_ID = {model_id}")
        return None

    df = pd.read_csv(path)

    df["perc_error"] = (df["pred_dose"] - df["real_dose"]) / (df["real_dose"] + 1e-5)

    mae = (df["pred_dose"] - df["real_dose"]).abs().mean()
    rmse = ((df["pred_dose"] - df["real_dose"]) ** 2).mean() ** 0.5
    corr = df["pred_dose"].corr(df["real_dose"])
    pct_similar = (df["perc_error"].abs() < 0.1).mean() * 100
    count = len(df)

    return {
        "MODEL_ID": model_id,
        "MAE": mae,
        "RMSE": rmse,
        "Correlation (pred/real)": corr,
        "% Similar (±10%)": pct_similar,
        "Samples": count,
    }


# Run model comparisons
results = [analyze_model(mid) for mid in MODEL_IDS if analyze_model(mid)]
df = pd.DataFrame(results)
df = df.round(3)


# Print full comparison
print("\n📊 PPO Model Comparison:\n")
print(df.to_string(index=False))


# Determine best model per metric
def best_model(metric, higher_is_better):
    return df.loc[df[metric].idxmax() if higher_is_better else df[metric].idxmin(), "MODEL_ID"]


summary = {
    "MAE": best_model("MAE", higher_is_better=False),
    "RMSE": best_model("RMSE", higher_is_better=False),
    "Correlation (pred/real)": best_model("Correlation (pred/real)", higher_is_better=True),
    "% Similar (±10%)": best_model("% Similar (±10%)", higher_is_better=True),
}

print("\n🏆 Best Model per Metric:")
for metric, model_id in summary.items():
    print(f"✔️ {metric}: Model {model_id}")

print("\n📌 Summary:")
if summary.values().__len__() == 1:
    print(f"Model {summary['MAE']} outperformed in all metrics.")
else:
    print("Models performed differently depending on the metric.")
