import polars as pl
import glob

# Columnas mínimas requeridas por el entorno
REQUIRED_COLUMNS = [
    *[f"cgm_{i}" for i in range(24)],
    "cgm_trend", "cgm_std",
    "bolus_log1p", "carb_input_log1p", "meal_carbs_log1p", "insulin_on_board_log1p",
    "hour_of_day", "meal_time_diff_hours", "has_meal", "meals_in_window"
]

def validate_parquet_columns(parquet_path: str):
    df = pl.read_parquet(parquet_path)
    cols = set(df.columns)
    missing = [col for col in REQUIRED_COLUMNS if col not in cols]

    if missing:
        print(f"❌ {parquet_path} — FALTAN columnas: {missing}")
    else:
        print(f"✅ {parquet_path} — OK ({len(df)} filas)")

def main():
    for path in glob.glob("new_ohio/processed_data/*/*.parquet"):
        validate_parquet_columns(path)

if __name__ == "__main__":
    main()
