import os
import numpy as np
import polars as pl
from pathlib import Path
from itertools import product
import logging
from typing import List, Dict, Any
from sklearn.model_selection import KFold
from models import ppo

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def get_data_dirs(base_dir: str) -> List[str]:
    """Get all subject data directories for cross-validation."""
    dirs = [str(p) for p in Path(base_dir).iterdir() if p.is_dir()]
    return sorted(dirs)

def run_grid_search_cv(
    data_dirs: List[str],
    output_dir: str,
    tensorboard_log: str,
    total_timesteps: int = 200_000,
    n_envs: int = 1,
    k_folds: int = 5
) -> Dict[str, Any]:
    """
    Run grid search with k-fold cross-validation for PPO hyperparameters.
    """
    # Hyperparameter grid
    learning_rates = [1e-5, 5e-5, 1e-4]
    batch_sizes = [64, 128, 256]
    net_archs = [[256, 256, 128], [512, 256]]
    grid = list(product(learning_rates, batch_sizes, net_archs))

    results = []
    kf = KFold(n_splits=k_folds, shuffle=True, random_state=42)
    data_dirs = np.array(data_dirs)

    for lr, batch_size, net_arch in grid:
        fold_metrics = []
        logging.info(f"Evaluating: lr={lr}, batch_size={batch_size}, net_arch={net_arch}")
        for fold, (train_idx, val_idx) in enumerate(kf.split(data_dirs)):
            train_dirs = data_dirs[train_idx].tolist()
            val_dirs = data_dirs[val_idx].tolist()
            fold_out = os.path.join(output_dir, f"fold_{fold+1}_lr{lr}_bs{batch_size}_arch{'-'.join(map(str, net_arch))}")
            tb_log = os.path.join(tensorboard_log, f"fold_{fold+1}_lr{lr}_bs{batch_size}_arch{'-'.join(map(str, net_arch))}")
            Path(fold_out).mkdir(parents=True, exist_ok=True)
            Path(tb_log).mkdir(parents=True, exist_ok=True)
            # Train
            model = ppo.train_with_hyperparameters(
                train_dirs=train_dirs,
                output_dir=fold_out,
                tensorboard_log=tb_log,
                total_timesteps=total_timesteps,
                n_envs=n_envs,
                learning_rate=lr,
                batch_size=batch_size,
                net_arch=net_arch
            )
            # Evaluate
            val_metrics = []
            for val_dir in val_dirs:
                val_file = f"{val_dir}/processed_{Path(val_dir).name}.parquet"
                if not Path(val_file).exists():
                    continue
                res = ppo.predict_from_preprocessed(
                    model_path=f"{fold_out}/ppo_ohio_final",
                    preprocessed_data_path=val_file
                )
                metrics = ppo.evaluate_metrics(res['predictions'], res['true_values'])
                val_metrics.append(metrics)
            # Aggregate fold metrics
            if val_metrics:
                avg_mae = np.mean([m['mae'] for m in val_metrics])
                avg_safe = np.mean([m['safe_rate'] for m in val_metrics])
                fold_metrics.append({'mae': avg_mae, 'safe_rate': avg_safe})
            else:
                fold_metrics.append({'mae': np.nan, 'safe_rate': np.nan})
        # Aggregate across folds
        maes = [m['mae'] for m in fold_metrics if not np.isnan(m['mae'])]
        safes = [m['safe_rate'] for m in fold_metrics if not np.isnan(m['safe_rate'])]
        mean_mae = np.mean(maes) if maes else np.nan
        mean_safe = np.mean(safes) if safes else np.nan
        results.append({
            'learning_rate': lr,
            'batch_size': batch_size,
            'net_arch': net_arch,
            'mean_mae': mean_mae,
            'mean_safe_rate': mean_safe,
            'folds': fold_metrics
        })
        logging.info(f"Grid: lr={lr}, bs={batch_size}, arch={net_arch} | MAE={mean_mae:.4f}, Safe={mean_safe:.2f}%")
    # Select best by MAE, then safe rate
    best = min(results, key=lambda x: (x['mean_mae'], -x['mean_safe_rate']))
    logging.info(f"Best config: lr={best['learning_rate']}, bs={best['batch_size']}, arch={best['net_arch']} | MAE={best['mean_mae']:.4f}, Safe={best['mean_safe_rate']:.2f}%")
    return {'results': results, 'best': best}

if __name__ == "__main__":
    # Example usage
    base_dir = "new_ohio/processed_data/train"
    output_dir = "new_ohio/models/gridsearch"
    tensorboard_log = "new_ohio/models/runs/ppo_gridsearch"
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    Path(tensorboard_log).mkdir(parents=True, exist_ok=True)
    data_dirs = get_data_dirs(base_dir)
    summary = run_grid_search_cv(
        data_dirs=data_dirs,
        output_dir=output_dir,
        tensorboard_log=tensorboard_log,
        total_timesteps=200_000,
        n_envs=1,
        k_folds=5
    )
    # Save summary
    import json
    with open(os.path.join(output_dir, "gridsearch_summary.json"), "w") as f:
        json.dump(summary, f, indent=2, default=str)
    print("Grid search and cross-validation complete. Best config:")
    print(summary['best'])
