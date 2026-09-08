"""Create thesis-ready aggregate SOTF predictor summaries.

The script reads completed SOTF simulation outputs and generates:
  1. A compact LaTeX table with aggregate predictor performance.
  2. A dataset-by-target-mode winner map showing which predictor has the
     lowest mean best-survivor gap for each model zoo.
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from statistics import mean, median

import matplotlib
import matplotlib.pyplot as plt

matplotlib.use("Agg")

REPO_ROOT = Path(__file__).resolve().parents[2]
BATCH_ROOT = REPO_ROOT / "sotf-selection" / "outputs" / "batch_experiments"
FIG_DIR = (
    REPO_ROOT
    / "tex_master_thesis"
    / "Thesis"
    / "sections"
    / "experiments"
    / "figures"
    / "survival_of_the_fittest"
)
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
WINNER_MAP_LABELS = {
    "weight_statistics_online_xgboost_with_perf": "Online XGB\n+ Wt.Stats",
    "weight_statistics_xgboost_with_perf": "Offline XGB\n+ Wt.Stats",
    "latent_window_representations_online_lstm": "Online Latent Window\nLSTM",
    "latent_window_representations_online_lstm_with_perf": "Online Latent Window\nLSTM + Perf.",
    "latent_tokens_representations_transformer_with_perf": "Latent Token\nTransformer + Perf.",
    "latent_tokens_representations_online_transformer": "Online Latent Token\nTransformer",
    "latent_tokens_representations_online_transformer_with_perf": "Online Latent Token\nTransformer + Perf.",
}
EXPERIMENT_COLORS = {
    "lc_extrapolation_baseline": "#DD8452",
    "lcnet_baseline": "#E5A06E",
    "naive_baseline": "#4C72B0",
    "online_xgboost_perf_only": "#C44E52",
    "weight_statistics_online_xgboost": "#B07AA1",
    "weight_statistics_online_xgboost_with_perf": "#8172B2",
    "xgboost_perf_only": "#55A868",
    "weight_statistics_xgboost": "#7CBF95",
    "weight_statistics_xgboost_with_perf": "#64B5CD",
    "mlp_perf_only": "#937860",
    "latent_averaged_representations_xgboost": "#8CD17D",
    "latent_averaged_representations_xgboost_with_perf": "#499894",
    "latent_averaged_representations_mlp": "#D4A6C8",
    "latent_averaged_representations_mlp_with_perf": "#B279A2",
    "latent_window_representations_lstm": "#F28E2B",
    "latent_window_representations_lstm_with_perf": "#E15759",
    "latent_window_representations_online_lstm": "#FFBE7D",
    "latent_window_representations_online_lstm_with_perf": "#D37295",
    "latent_tokens_representations_transformer": "#9C755F",
    "latent_tokens_representations_transformer_with_perf": "#BAB0AC",
    "latent_tokens_representations_online_transformer": "#59A14F",
    "latent_tokens_representations_online_transformer_with_perf": "#EDC948",
}


def load_runs() -> list[dict]:
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
                "recall": summary.get("oracle_top_k_recall"),
            }
        )
    return rows


def summarize(rows: list[dict]) -> dict[tuple[str, str], dict]:
    grouped: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for row in rows:
        grouped[(row["target_mode"], row["experiment"])].append(row)

    summary: dict[tuple[str, str], dict] = {}
    for key, group in grouped.items():
        gaps = [row["gap"] for row in group]
        recalls = [row["recall"] for row in group if row["recall"] is not None]
        summary[key] = {
            "mean_gap": mean(gaps),
            "median_gap": median(gaps),
            "mean_recall": mean(recalls) if recalls else None,
            "datasets": len({row["dataset_id"] for row in group}),
            "runs": len(group),
        }
    return summary


def _fmt(value: float | None, *, bold: bool = False) -> str:
    if value is None:
        text = "n/a"
    else:
        text = f"{value:.4f}"
    return rf"\textbf{{{text}}}" if bold else text


def write_latex_table(summary: dict[tuple[str, str], dict]) -> None:
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    best_gap_by_mode = {}
    for target_mode in TARGET_MODES:
        vals = [
            stats["mean_gap"]
            for (mode, _), stats in summary.items()
            if mode == target_mode
        ]
        best_gap_by_mode[target_mode] = min(vals)

    lines = [
        r"\begin{table}[H]",
        r"    \centering",
        r"    \scriptsize",
        r"    \caption[SOTF aggregate predictor ranking]{\textbf{SOTF Aggregate Predictor Ranking.}",
        r"    Mean best survivor gap and mean oracle top K recall are averaged over all completed runs for each predictor and target mode.",
        r"    Lower gap and higher recall are better. The number of covered model zoos is shown separately for final value and horizon value mode.}",
        r"    \label{tab:sotf-aggregate-predictor-ranking}",
        r"    \renewcommand{\arraystretch}{0.92}",
        r"    \begin{tabularx}{\textwidth}{@{}>{\raggedright\arraybackslash}X c c c c@{}}",
        r"        \hline",
        r"        \textbf{Predictor} & \textbf{Zoos F/H} & \textbf{Mean gap F} & \textbf{Mean gap H} & \textbf{Mean recall F/H} \\",
        r"        \hline",
    ]

    for experiment in EXPERIMENTS:
        final_stats = summary.get(("final_value", experiment))
        horizon_stats = summary.get(("horizon_value", experiment))
        final_gap = final_stats["mean_gap"] if final_stats else None
        horizon_gap = horizon_stats["mean_gap"] if horizon_stats else None
        final_recall = final_stats["mean_recall"] if final_stats else None
        horizon_recall = horizon_stats["mean_recall"] if horizon_stats else None
        coverage = (
            f"{final_stats['datasets'] if final_stats else 0}/"
            f"{horizon_stats['datasets'] if horizon_stats else 0}"
        )
        recall_text = f"{final_recall:.3f}/{horizon_recall:.3f}" if final_recall is not None and horizon_recall is not None else "n/a"
        lines.append(
            "        "
            + " & ".join(
                [
                    EXPERIMENT_LABELS[experiment],
                    coverage,
                    _fmt(final_gap, bold=final_gap == best_gap_by_mode["final_value"]),
                    _fmt(horizon_gap, bold=horizon_gap == best_gap_by_mode["horizon_value"]),
                    recall_text,
                ]
            )
            + r" \\"
        )

    lines.extend(
        [
            r"        \hline",
            r"    \end{tabularx}",
            r"    \renewcommand{\arraystretch}{1.0}",
            r"\end{table}",
            "",
        ]
    )
    (TABLE_DIR / "sotf_aggregate_predictor_ranking.tex").write_text("\n".join(lines))


def best_by_dataset(rows: list[dict]) -> dict[tuple[int, str], list[tuple[float, str]]]:
    grouped: dict[tuple[int, str, str], list[float]] = defaultdict(list)
    for row in rows:
        grouped[(row["dataset_id"], row["target_mode"], row["experiment"])].append(row["gap"])

    result: dict[tuple[int, str], list[tuple[float, str]]] = {}
    for dataset_id in DATASET_ORDER:
        for target_mode in TARGET_MODES:
            candidates = []
            for experiment in EXPERIMENTS:
                vals = grouped.get((dataset_id, target_mode, experiment), [])
                if vals:
                    candidates.append((mean(vals), experiment))
            result[(dataset_id, target_mode)] = sorted(candidates)
    return result


def plot_winner_map(winners: dict[tuple[int, str], list[tuple[float, str]]]) -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    plt.rcParams["font.family"] = "Times New Roman"
    plt.rcParams["font.size"] = 16

    fig, ax = plt.subplots(figsize=(11.8, 14.5))
    ax.set_xlim(0, 2)
    ax.set_ylim(0, len(DATASET_ORDER))
    ax.invert_yaxis()
    ax.set_xticks([0.5, 1.5])
    ax.set_xticklabels([TARGET_LABELS[mode] for mode in TARGET_MODES], fontsize=24)
    ax.set_yticks([idx + 0.5 for idx in range(len(DATASET_ORDER))])
    ax.set_yticklabels([DATASET_DISPLAY[dataset_id] for dataset_id in DATASET_ORDER], fontsize=20)
    ax.tick_params(axis="both", length=0)

    for idx, dataset_id in enumerate(DATASET_ORDER):
        for col, target_mode in enumerate(TARGET_MODES):
            candidates = winners.get((dataset_id, target_mode), [])
            if not candidates:
                color = "#f2f2f2"
                text = "no data"
            else:
                gap, experiment = candidates[0]
                color = EXPERIMENT_COLORS[experiment]
                label = WINNER_MAP_LABELS.get(experiment, EXPERIMENT_LABELS[experiment])
                suffix = "\n(single predictor)" if len(candidates) == 1 else ""
                text = f"{label}\nmean gap {gap:.4f}{suffix}"
            rect = plt.Rectangle((col, idx), 1, 1, facecolor=color, alpha=0.78, edgecolor="white", linewidth=2)
            ax.add_patch(rect)
            ax.text(col + 0.5, idx + 0.5, text, ha="center", va="center", fontsize=20, color="black")

    ax.set_title("Best Predictor per Model Zoo and Target Mode", fontsize=28, pad=18)
    for spine in ax.spines.values():
        spine.set_visible(False)

    winning_experiments = sorted(
        {
            candidates[0][1]
            for candidates in winners.values()
            if candidates
        },
        key=lambda experiment: EXPERIMENTS.index(experiment),
    )
    legend_handles = [
        plt.Rectangle((0, 0), 1, 1, facecolor=EXPERIMENT_COLORS[experiment], alpha=0.78)
        for experiment in winning_experiments
    ]
    ax.legend(
        legend_handles,
        [EXPERIMENT_LABELS[experiment] for experiment in winning_experiments],
        ncol=2,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.08),
        frameon=False,
        fontsize=20,
    )
    fig.tight_layout()
    for ext in (".pdf", ".png"):
        fig.savefig(FIG_DIR / f"sotf_predictor_winner_map{ext}", bbox_inches="tight", dpi=250)
    plt.close(fig)


def main() -> None:
    rows = load_runs()
    print(f"Loaded {len(rows)} runs.")
    summary = summarize(rows)
    write_latex_table(summary)
    winners = best_by_dataset(rows)
    plot_winner_map(winners)
    print(f"Wrote {TABLE_DIR / 'sotf_aggregate_predictor_ranking.tex'}")
    print(f"Wrote {FIG_DIR / 'sotf_predictor_winner_map.pdf'}")


if __name__ == "__main__":
    main()
