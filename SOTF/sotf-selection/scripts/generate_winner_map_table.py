"""Generate CSV from SOTF predictor winner map data.

This script extracts which predictor wins (lowest mean best-survivor gap) for each
model zoo and target mode, then writes to a CSV file.
"""

from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path
from statistics import mean

REPO_ROOT = Path(__file__).resolve().parents[2]
BATCH_ROOT = REPO_ROOT / "sotf-selection" / "outputs" / "batch_experiments"
TABLE_DIR = (
    REPO_ROOT
    / "tex_master_thesis"
    / "Thesis"
    / "sections"
    / "experiments"
    / "tables"
    / "survival_of_the_fittest"
)

TARGET_MODES = ["final_value", "horizon_value"]
TARGET_LABELS = {
    "final_value": "Final value",
    "horizon_value": "Horizon value",
}

DATASET_ORDER = [1, 2, 3, 8, 9, 10, 14, 5, 7]
DATASET_DISPLAY = {
    1: "CIFAR 10 CNN Uniform",
    2: "CIFAR 10 CNN Hyp. Rand.",
    3: "CIFAR 10 CNN Hyp. Fixed",
    5: "CIFAR 10 ResNet 18 Width",
    7: "CIFAR 10 ViT",
    8: "STL 10 CNN Uniform",
    9: "STL 10 CNN Hyp. Rand.",
    10: "STL 10 CNN Hyp. Fixed",
    14: "TinyImageNet ResNet 18 Phase",
}

EXPERIMENTS = [
    "lc_extrapolation_baseline",
    "lcnet_baseline",
    "naive_baseline",
    "online_xgboost_perf_only",
    "weight_statistics_online_xgboost",
    "weight_statistics_online_xgboost_with_perf",
    "xgboost_perf_only",
    "weight_statistics_xgboost",
    "weight_statistics_xgboost_with_perf",
    "mlp_perf_only",
    "latent_averaged_representations_xgboost",
    "latent_averaged_representations_xgboost_with_perf",
    "latent_averaged_representations_mlp",
    "latent_averaged_representations_mlp_with_perf",
    "latent_window_representations_lstm",
    "latent_window_representations_lstm_with_perf",
    "latent_window_representations_online_lstm",
    "latent_window_representations_online_lstm_with_perf",
    "latent_tokens_representations_transformer",
    "latent_tokens_representations_transformer_with_perf",
    "latent_tokens_representations_online_transformer",
    "latent_tokens_representations_online_transformer_with_perf",
]

EXPERIMENT_LABELS = {
    "lc_extrapolation_baseline": "LC Extrap.",
    "lcnet_baseline": "LCNet",
    "naive_baseline": "Naive",
    "online_xgboost_perf_only": "Online XGB",
    "weight_statistics_online_xgboost": "Online XGB Wt.",
    "weight_statistics_online_xgboost_with_perf": "Online XGB + Wt.Stats",
    "xgboost_perf_only": "Offline XGB",
    "weight_statistics_xgboost": "Offline XGB Wt.",
    "weight_statistics_xgboost_with_perf": "Offline XGB + Wt.Stats",
    "mlp_perf_only": "Offline MLP",
    "latent_averaged_representations_xgboost": "Avg. Latent XGB",
    "latent_averaged_representations_xgboost_with_perf": "Avg. Latent XGB + Perf.",
    "latent_averaged_representations_mlp": "Avg. Latent MLP",
    "latent_averaged_representations_mlp_with_perf": "Avg. Latent MLP + Perf.",
    "latent_window_representations_lstm": "Latent Window LSTM",
    "latent_window_representations_lstm_with_perf": "Latent Window LSTM + Perf.",
    "latent_window_representations_online_lstm": "Online Latent Window LSTM",
    "latent_window_representations_online_lstm_with_perf": "Online Latent Window LSTM + Perf.",
    "latent_tokens_representations_transformer": "Latent Token Transformer",
    "latent_tokens_representations_transformer_with_perf": "Latent Token Transformer + Perf.",
    "latent_tokens_representations_online_transformer": "Online Latent Token Transformer",
    "latent_tokens_representations_online_transformer_with_perf": "Online Latent Token Transformer + Perf.",
}


def load_runs() -> list[dict]:
    """Load all SOTF simulation results from batch outputs."""
    rows: list[dict] = []
    for summary_path in BATCH_ROOT.rglob("simulation_summary.json"):
        manifest_path = summary_path.parent / "run_manifest.json"
        if not manifest_path.exists():
            continue
        try:
            summary = json.loads(summary_path.read_text())
            manifest = json.loads(manifest_path.read_text())
        except json.JSONDecodeError:
            continue

        experiment = manifest.get("experiment_name")
        if experiment not in EXPERIMENTS:
            continue

        oracle = summary.get("oracle_best_final_target")
        survivor = summary.get("best_survivor_final_target")
        if oracle is None or survivor is None:
            continue

        rows.append(
            {
                "dataset_id": manifest.get("dataset_id"),
                "target_mode": manifest.get("target_mode"),
                "experiment": experiment,
                "gap": float(oracle) - float(survivor),
            }
        )
    return rows


def best_by_dataset(rows: list[dict]) -> dict[tuple[int, str], tuple[float, str]]:
    """Find the best (lowest gap) predictor for each dataset and target mode."""
    grouped: dict[tuple[int, str, str], list[float]] = defaultdict(list)
    for row in rows:
        grouped[(row["dataset_id"], row["target_mode"], row["experiment"])].append(row["gap"])

    result: dict[tuple[int, str], tuple[float, str]] = {}
    for dataset_id in DATASET_ORDER:
        for target_mode in TARGET_MODES:
            candidates = []
            for experiment in EXPERIMENTS:
                vals = grouped.get((dataset_id, target_mode, experiment), [])
                if vals:
                    candidates.append((mean(vals), experiment))
            if candidates:
                best_gap, best_experiment = min(candidates)
                result[(dataset_id, target_mode)] = (best_gap, best_experiment)
    return result


def write_winner_map_csv(winners: dict[tuple[int, str], tuple[float, str]]) -> None:
    """Write CSV with the winner map data."""
    TABLE_DIR.mkdir(parents=True, exist_ok=True)

    output_path = TABLE_DIR / "sotf_predictor_winner_map.csv"

    with open(output_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Model Zoo", "Final Value Predictor", "Final Value Gap", "Horizon Value Predictor", "Horizon Value Gap"])

        for dataset_id in DATASET_ORDER:
            dataset_name = DATASET_DISPLAY[dataset_id]
            final_key = (dataset_id, "final_value")
            horizon_key = (dataset_id, "horizon_value")

            final_pred = "---"
            final_gap = "---"
            if final_key in winners:
                gap, experiment = winners[final_key]
                label = EXPERIMENT_LABELS[experiment]
                final_pred = label
                final_gap = f"{gap:.4f}"

            horizon_pred = "---"
            horizon_gap = "---"
            if horizon_key in winners:
                gap, experiment = winners[horizon_key]
                label = EXPERIMENT_LABELS[experiment]
                horizon_pred = label
                horizon_gap = f"{gap:.4f}"

            writer.writerow([dataset_name, final_pred, final_gap, horizon_pred, horizon_gap])

    print(f"Wrote {output_path}")


def main() -> None:
    rows = load_runs()
    print(f"Loaded {len(rows)} runs.")
    winners = best_by_dataset(rows)
    print(f"Found winners for {len(winners)} dataset-mode combinations.")
    write_winner_map_csv(winners)


if __name__ == "__main__":
    main()
