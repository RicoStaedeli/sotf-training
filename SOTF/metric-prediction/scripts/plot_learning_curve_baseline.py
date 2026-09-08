#!/usr/bin/env python3
"""Plot learning-curve extrapolation baseline results for the thesis."""

from __future__ import annotations

import argparse
import json
import textwrap
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[2]
BASE_FONT_SIZE = 16
FONT_FAMILY = "Times New Roman"
TARGET_LABELS = {"final_value": "Final value", "horizon_value": "Horizon 5", "horizon": "Horizon 5"}
TARGET_ORDER = ["Final value", "Horizon 5"]
DATASET_ORDER = [
    "cifar10_cnn_large_uniform",
    "cifar10_cnn_large_hyp_random_seeds",
    "cifar10_cnn_large_hyp_fixed_seeds",
    "stl10_cnn_large_uniform",
    "stl10_cnn_large_hyp_random_seeds",
    "stl10_cnn_large_hyp_fixed_seeds",
    "tinyimagenet_resnet18_phase_transitions",
    "cifar10_resnet18_width_2-256",
    "cifar10_vit",
]
DATASET_ORDER_INDEX = {dataset: index for index, dataset in enumerate(DATASET_ORDER)}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render thesis plots for learning-curve extrapolation baselines.")
    parser.add_argument(
        "--baseline-root",
        type=Path,
        default=REPO_ROOT / "metric-prediction/outputs/baselines/learning_curve_extrapolation",
        help="Root containing baseline metrics.",
    )
    parser.add_argument(
        "--model-zoo-info",
        type=Path,
        default=REPO_ROOT / "learning-dynamics/configs/model_zoo_informations.json",
        help="JSON file containing model-zoo plot names.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT / "metric-prediction/outputs/metric_prediction/learning_curve_baseline",
        help="Directory where plot and CSV artifacts are written.",
    )
    parser.add_argument(
        "--tex-figure-dir",
        type=Path,
        default=REPO_ROOT / "tex_master_thesis/Thesis/sections/experiments/figures/metric_prediction",
        help="Thesis figure directory where the PDF is copied.",
    )
    parser.add_argument(
        "--artifact-prefix",
        default="learning_curve_extrapolation_baseline",
        help="Prefix for generated CSV and PDF artifacts.",
    )
    return parser.parse_args()


def load_plot_names(path: Path) -> dict[int, str]:
    records = json.loads(path.read_text(encoding="utf-8"))
    out: dict[int, str] = {}
    for item in records:
        zoo_name = str(item["zoo_name"])
        dataset_num = int(zoo_name.split("_", 1)[0])
        out[dataset_num] = str(item["plot_name"])
    return out


def load_baseline_rows(root: Path, plot_names: dict[int, str]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for metrics_path in sorted(root.rglob("metrics.json")):
        parsed = parse_baseline_path(metrics_path, root)
        if parsed is None:
            continue
        target_dir, dataset_num, run_name = parsed
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        target_mode = TARGET_LABELS.get(str(metrics.get("target_mode", target_dir)), str(metrics.get("target_mode", target_dir)))
        rows.append(
            {
                "dataset_num": dataset_num,
                "dataset": plot_names.get(dataset_num, f"DS {dataset_num}"),
                "target_mode": target_mode,
                "run_name": run_name,
                "r2": float(metrics.get("r2", np.nan)),
                "rmse": float(metrics.get("rmse", np.nan)),
                "mae": float(metrics.get("mae", np.nan)),
                "fallback_fraction": float(metrics.get("fallback_fraction", np.nan)),
                "n_validation": int(metrics.get("n_validation", 0) or 0),
                "source_path": metrics_path.as_posix(),
            }
        )
    if not rows:
        raise FileNotFoundError(f"No baseline metrics.json files found under {root}")

    frame = pd.DataFrame(rows)
    frame["dataset_order"] = frame["dataset"].map(DATASET_ORDER_INDEX).fillna(len(DATASET_ORDER))
    frame["target_order"] = frame["target_mode"].map({target: idx for idx, target in enumerate(TARGET_ORDER)})
    return frame.sort_values(["dataset_order", "target_order", "dataset_num"]).reset_index(drop=True)


def parse_baseline_path(metrics_path: Path, root: Path) -> tuple[str, int, str] | None:
    try:
        parts = metrics_path.relative_to(root).parts
    except ValueError:
        return None
    if len(parts) != 6 or parts[2] != "results" or parts[4] != "artifacts":
        return None
    target_dir, dataset_dir, _, run_name, _, filename = parts
    if filename != "metrics.json" or not dataset_dir.startswith("ds"):
        return None
    try:
        dataset_num = int(dataset_dir.removeprefix("ds"))
    except ValueError:
        return None
    return target_dir, dataset_num, run_name


def configure_plot_style() -> None:
    plt.rcParams.update(
        {
            "font.family": FONT_FAMILY,
            "font.serif": [FONT_FAMILY],
            "font.size": BASE_FONT_SIZE,
            "axes.titlesize": BASE_FONT_SIZE + 1,
            "axes.labelsize": BASE_FONT_SIZE,
            "xtick.labelsize": BASE_FONT_SIZE - 1,
            "ytick.labelsize": BASE_FONT_SIZE - 1,
            "legend.fontsize": BASE_FONT_SIZE,
            "mathtext.fontset": "custom",
            "mathtext.rm": FONT_FAMILY,
            "mathtext.it": f"{FONT_FAMILY}:italic",
            "mathtext.bf": f"{FONT_FAMILY}:bold",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def render_baseline_plot(data: pd.DataFrame, output_path: Path) -> None:
    datasets = (
        data[["dataset_num", "dataset", "dataset_order"]]
        .drop_duplicates()
        .sort_values(["dataset_order", "dataset_num"])
        .reset_index(drop=True)
    )
    dataset_labels = [
        "\n".join(textwrap.wrap(str(row.dataset), width=26, break_long_words=False))
        for row in datasets.itertuples(index=False)
    ]
    y = np.arange(len(datasets))
    bar_height = 0.34
    baseline_color = "#7a7a7a"
    target_hatch = {"Final value": "", "Horizon 5": "///"}
    target_edge = {"Final value": "#222222", "Horizon 5": "#ffffff"}
    offsets = {"Final value": -bar_height / 2, "Horizon 5": bar_height / 2}

    fig, axes = plt.subplots(1, 2, figsize=(12.8, 6.8), sharey=True)
    for ax, metric, title in [
        (axes[0], "rmse", "Validation RMSE"),
        (axes[1], "r2", r"Validation $R^2$"),
    ]:
        for target_mode in TARGET_ORDER:
            values = []
            for dataset_num in datasets["dataset_num"]:
                row = data[(data["dataset_num"] == dataset_num) & (data["target_mode"] == target_mode)]
                if row.empty:
                    values.append(np.nan)
                else:
                    value = float(row[metric].iloc[0])
                    values.append(value)
            ax.barh(
                y + offsets[target_mode],
                values,
                bar_height,
                label=target_mode,
                color=baseline_color,
                edgecolor=target_edge[target_mode],
                linewidth=0.7,
                hatch=target_hatch[target_mode],
            )

        if metric == "r2":
            ax.axvline(0, color="#333333", linewidth=0.8)
            ax.set_xlim(0.0, 1.12)
        else:
            ax.set_xlim(left=0.0)
        ax.set_title(title)
        ax.set_xlabel("")
        ax.grid(axis="x", color="#d0d0d0", linewidth=0.6, alpha=0.8)
        ax.set_axisbelow(True)

    axes[0].set_yticks(y)
    axes[0].set_yticklabels(dataset_labels)
    axes[0].invert_yaxis()
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 1.02), ncol=2, frameon=False)
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.95), w_pad=2.0)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def render_heatmap(data: pd.DataFrame, output_path: Path) -> None:
    datasets = (
        data[["dataset_num", "dataset", "dataset_order"]]
        .drop_duplicates()
        .sort_values(["dataset_order", "dataset_num"])
        .reset_index(drop=True)
    )
    dataset_labels = [
        "\n".join(textwrap.wrap(str(row.dataset), width=16, break_long_words=False))
        for row in datasets.itertuples(index=False)
    ]
    r2_grid = np.full((len(TARGET_ORDER), len(datasets)), np.nan)
    rmse_grid = np.full((len(TARGET_ORDER), len(datasets)), np.nan)
    for row in data.itertuples(index=False):
        if row.target_mode not in TARGET_ORDER:
            continue
        x_idx = int(datasets.index[datasets["dataset_num"] == row.dataset_num][0])
        y_idx = TARGET_ORDER.index(row.target_mode)
        r2_grid[y_idx, x_idx] = float(row.r2)
        rmse_grid[y_idx, x_idx] = float(row.rmse)

    fig, ax = plt.subplots(figsize=(15.5, 4.2))
    norm = TwoSlopeNorm(vmin=min(-0.35, float(np.nanmin(r2_grid))), vcenter=0.0, vmax=1.0)
    image = ax.imshow(np.ma.masked_invalid(r2_grid), cmap="RdYlGn", norm=norm, aspect="auto")
    ax.set_yticks(np.arange(len(TARGET_ORDER)))
    ax.set_yticklabels(TARGET_ORDER)
    ax.set_xticks(np.arange(len(datasets)))
    ax.set_xticklabels(dataset_labels, rotation=35, ha="right")
    ax.set_xticks(np.arange(-0.5, len(datasets), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(TARGET_ORDER), 1), minor=True)
    ax.grid(which="minor", color="#ffffff", linewidth=0.8)
    ax.tick_params(which="minor", bottom=False, left=False)
    for y_idx in range(len(TARGET_ORDER)):
        for x_idx in range(len(datasets)):
            r2_value = r2_grid[y_idx, x_idx]
            rmse_value = rmse_grid[y_idx, x_idx]
            if not np.isfinite(r2_value) or not np.isfinite(rmse_value):
                continue
            color = "#ffffff" if r2_value >= 0.65 else "#111111"
            ax.text(x_idx, y_idx, f"{r2_value:.2f}\n{rmse_value:.3f}", ha="center", va="center", fontsize=BASE_FONT_SIZE - 4, color=color)
    colorbar = fig.colorbar(image, ax=ax, fraction=0.03, pad=0.02)
    colorbar.set_label(r"Validation $R^2$")
    ax.set_xlabel("Dataset (cell text: $R^2$ / RMSE)")
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.tex_figure_dir.mkdir(parents=True, exist_ok=True)
    configure_plot_style()

    plot_names = load_plot_names(args.model_zoo_info)
    data = load_baseline_rows(args.baseline_root, plot_names)
    data.drop(columns=["dataset_order", "target_order"], errors="ignore").to_csv(
        args.output_dir / f"{args.artifact_prefix}_metrics.csv",
        index=False,
    )
    summary = (
        data.groupby("target_mode", as_index=False)
        .agg(
            n_datasets=("dataset_num", "nunique"),
            mean_r2=("r2", "mean"),
            median_r2=("r2", "median"),
            mean_rmse=("rmse", "mean"),
            median_rmse=("rmse", "median"),
            mean_fallback_fraction=("fallback_fraction", "mean"),
        )
    )
    summary.to_csv(args.output_dir / f"{args.artifact_prefix}_summary.csv", index=False)

    pdf_path = args.output_dir / f"{args.artifact_prefix}_performance.pdf"
    heatmap_path = args.output_dir / f"{args.artifact_prefix}_heatmap.pdf"
    render_baseline_plot(data, pdf_path)
    render_heatmap(data, heatmap_path)

    tex_pdf_path = args.tex_figure_dir / pdf_path.name
    tex_heatmap_path = args.tex_figure_dir / heatmap_path.name
    tex_pdf_path.write_bytes(pdf_path.read_bytes())
    tex_heatmap_path.write_bytes(heatmap_path.read_bytes())
    print(f"Wrote baseline plot artifacts to {args.output_dir}")
    print(f"Copied thesis figures to {args.tex_figure_dir}")


if __name__ == "__main__":
    main()
