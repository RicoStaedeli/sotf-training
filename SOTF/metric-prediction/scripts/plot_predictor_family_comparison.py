#!/usr/bin/env python3
"""Summarize metric-forecasting predictor families for the thesis."""

from __future__ import annotations

import argparse
import json
import re
import textwrap
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from matplotlib.patches import Rectangle
import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[2]
BASE_FONT_SIZE = 18
FONT_FAMILY = "Times New Roman"
TARGET_LABELS = {"final_value": "Final value", "horizon": "Horizon 5", "horizon_value": "Horizon 5"}
TARGET_ORDER = ["Final value", "Horizon 5"]
ABLATION_ORDER = ["Performance only", "Representation only", "Representation + performance"]
ABLATION_COLORS = {
    "Performance only": "#7a7a7a",
    "Representation only": "#3d74a3",
    "Representation + performance": "#d47c3c",
}
FAMILY_ORDER = [
    "Naive current value",
    "LCNet baseline",
    "Learning-curve extrapolation baseline",
    "Weight statistics + XGBoost",
    "Weight statistics + MLP",
    "Averaged latent + Ridge",
    "Averaged latent + XGBoost",
    "Averaged latent + MLP",
    "Latent windows + LSTM",
    "Latent tokens + Transformer",
]
FAMILY_COLORS = {
    "Naive current value": "#7a7a7a",
    "LCNet baseline": "#a0a0a0",
    "Learning-curve extrapolation baseline": "#c0c0c0",
    "Weight statistics + XGBoost": "#2364aa",
    "Weight statistics + MLP": "#3da35d",
    "Averaged latent + Ridge": "#e0a458",
    "Averaged latent + XGBoost": "#5b8e7d",
    "Averaged latent + MLP": "#9b5de5",
    "Latent windows + LSTM": "#d1495b",
    "Latent tokens + Transformer": "#2f4858",
}
FAMILY_ABBREVIATIONS = {
    "Naive current value": "Naive",
    "LCNet baseline": "baseline LCNet",
    "Learning-curve extrapolation baseline": "baseline LC extr.",
    "Weight statistics + XGBoost": "Wt. stat. + XGBoost",
    "Weight statistics + MLP": "Wt. stat. + MLP",
    "Averaged latent + Ridge": "Lat. aver. + Ridge",
    "Averaged latent + XGBoost": "Lat. aver. + XGBoost",
    "Averaged latent + MLP": "Lat. aver. + MLP",
    "Latent windows + LSTM": "Lat. win. + LSTM",
    "Latent tokens + Transformer": "Lat. tok. + Transformer",
}
SECTION_FAMILY_PLOTS = {
    "averaged_latent_ridge_by_dataset": ["Averaged latent + Ridge"],
    "weight_statistics_xgboost_by_dataset": ["Weight statistics + XGBoost"],
    "averaged_latent_xgboost_by_dataset": ["Averaged latent + XGBoost"],
    "weight_statistics_mlp_by_dataset": ["Weight statistics + MLP"],
    "averaged_latent_mlp_by_dataset": ["Averaged latent + MLP"],
    "latent_windows_lstm_by_dataset": ["Latent windows + LSTM"],
    "latent_tokens_transformer_by_dataset": ["Latent tokens + Transformer"],
}
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
    parser = argparse.ArgumentParser(
        description="Create compact thesis plots comparing metric-prediction families.",
    )
    parser.add_argument(
        "--forecasting-root",
        type=Path,
        default=REPO_ROOT / "metric-prediction/outputs/forecasting",
        help="Root containing saved forecasting metrics.",
    )
    parser.add_argument(
        "--naive-metrics",
        type=Path,
        default=(
            REPO_ROOT
            / "metric-prediction/outputs/metric_prediction/naive_current_value_baseline"
            / "naive_current_value_baseline_metrics.csv"
        ),
        help="CSV produced by plot_naive_current_value_baseline.py.",
    )
    parser.add_argument(
        "--lcnet-metrics",
        type=Path,
        default=(
            REPO_ROOT
            / "metric-prediction/outputs/metric_prediction/lcnet_baseline"
            / "lcnet_baseline_metrics.csv"
        ),
        help="CSV produced by plot_learning_curve_baseline.py for the LCNet baseline.",
    )
    parser.add_argument(
        "--learning-curve-metrics",
        type=Path,
        default=(
            REPO_ROOT
            / "metric-prediction/outputs/metric_prediction/learning_curve_baseline"
            / "learning_curve_baseline_metrics.csv"
        ),
        help="CSV produced by plot_learning_curve_baseline.py for the parametric extrapolation baseline.",
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
        default=REPO_ROOT / "metric-prediction/outputs/metric_prediction/predictor_family_comparison",
        help="Directory where the generated PDF and CSV files are written.",
    )
    parser.add_argument(
        "--horizon",
        type=int,
        default=5,
        help="Horizon to include for horizon-value forecasting plots.",
    )
    return parser.parse_args()


def load_plot_names(path: Path) -> dict[int, str]:
    with path.open(encoding="utf-8") as handle:
        records = json.load(handle)

    plot_names: dict[int, str] = {}
    for item in records:
        zoo_name = str(item["zoo_name"])
        dataset_num = int(zoo_name.split("_", 1)[0])
        plot_names[dataset_num] = str(item["plot_name"])
    return plot_names


def sort_datasets(datasets: pd.DataFrame) -> pd.DataFrame:
    result = datasets.copy()
    result["dataset_order"] = result["dataset"].map(DATASET_ORDER_INDEX).fillna(len(DATASET_ORDER))
    return result.sort_values(["dataset_order", "dataset_num", "dataset"]).drop(columns=["dataset_order"]).reset_index(drop=True)


def load_metric_rows(
    forecasting_root: Path,
    plot_names: dict[int, str],
    *,
    horizon: int,
    include_perf_only: bool,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for metrics_path in sorted(forecasting_root.rglob("metrics.json")):
        parsed = parse_metric_path(metrics_path, forecasting_root)
        if parsed is None:
            continue
        representation, target_dir, model_name, dataset_num, run_name = parsed
        family = family_label(representation, model_name)
        if family is None or (run_is_perf_only(run_name) and not include_perf_only):
            continue

        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        if target_dir == "horizon" and int(metrics.get("horizon", horizon)) != horizon:
            continue
        target_label = TARGET_LABELS.get(str(metrics.get("target_mode", target_dir)), TARGET_LABELS.get(target_dir, target_dir))
        rows.append(
            {
                "dataset_num": dataset_num,
                "dataset": plot_names.get(dataset_num, f"DS {dataset_num}"),
                "target_mode": target_label,
                "representation": representation,
                "model": model_name,
                "family": family,
                "ablation": ablation_label(run_name),
                "run_name": run_name,
                "horizon": int(metrics.get("horizon", 0) or 0),
                "r2": float(metrics.get("r2", np.nan)),
                "rmse": float(metrics.get("rmse", np.nan)),
                "mae": float(metrics.get("mae", np.nan)),
                "n_validation": int(metrics.get("n_validation", 0) or 0),
                "source_path": metrics_path.as_posix(),
            }
        )

    if not rows:
        raise FileNotFoundError(f"No learned predictor metrics found under {forecasting_root}")
    return pd.DataFrame(rows)


def parse_metric_path(metrics_path: Path, forecasting_root: Path) -> tuple[str, str, str, int, str] | None:
    try:
        parts = metrics_path.relative_to(forecasting_root).parts
    except ValueError:
        return None
    if len(parts) < 6 or parts[4] != "results":
        return None

    representation, target_dir, model_name, dataset_dir, _, run_name = parts[:6]
    match = re.fullmatch(r"ds(\d+)", dataset_dir)
    if not match:
        return None
    return representation, target_dir, model_name, int(match.group(1)), run_name


def family_label(representation: str, model_name: str) -> str | None:
    labels = {
        ("weight_statistics", "xgboost"): "Weight statistics + XGBoost",
        ("weight_statistics", "mlp"): "Weight statistics + MLP",
        ("latent_averaged", "ridge"): "Averaged latent + Ridge",
        ("latent_averaged", "xgboost"): "Averaged latent + XGBoost",
        ("latent_averaged", "mlp"): "Averaged latent + MLP",
        ("latent_windows", "lstm"): "Latent windows + LSTM",
        ("latent_tokens", "transformer"): "Latent tokens + Transformer",
    }
    return labels.get((representation, model_name))


def run_is_perf_only(run_name: str) -> bool:
    return "perf_only" in run_name.lower()


def ablation_label(run_name: str) -> str:
    normalized = run_name.lower()
    if "perf_only" in normalized:
        return "Performance only"
    if "with_perf" in normalized:
        return "Representation + performance"
    return "Representation only"


def select_best_configurations(runs: pd.DataFrame) -> pd.DataFrame:
    """Select exactly one configuration per dataset, target mode, and family.

    The thesis-level mean RMSE reduction is computed only from this selected
    table, never from the full set of hyperparameter/configuration runs.
    Lower validation RMSE is primary; higher R2 is only a deterministic
    tie-breaker.
    """
    selected = (
        runs.sort_values(["dataset_num", "target_mode", "family", "rmse", "r2"], ascending=[True, True, True, True, False])
        .drop_duplicates(["dataset_num", "target_mode", "family"], keep="first")
        .sort_values(["dataset_num", "target_mode", "family"])
        .reset_index(drop=True)
    )
    assert_unique_selected_configurations(selected)
    return selected


def select_best_ablation_configurations(runs: pd.DataFrame) -> pd.DataFrame:
    selected = (
        runs.sort_values(
            ["dataset_num", "target_mode", "family", "ablation", "rmse", "r2"],
            ascending=[True, True, True, True, True, False],
        )
        .drop_duplicates(["dataset_num", "target_mode", "family", "ablation"], keep="first")
        .sort_values(["dataset_num", "target_mode", "family", "ablation"])
        .reset_index(drop=True)
    )
    duplicate_mask = selected.duplicated(["dataset_num", "target_mode", "family", "ablation"], keep=False)
    if duplicate_mask.any():
        duplicates = selected.loc[
            duplicate_mask,
            ["dataset_num", "target_mode", "family", "ablation", "run_name"],
        ]
        raise RuntimeError(
            "Ablation summary must use exactly one selected configuration per "
            "dataset, target mode, family, and ablation setting. Duplicate selections:\n"
            f"{duplicates.to_string(index=False)}"
        )
    return selected


def assert_unique_selected_configurations(selected: pd.DataFrame) -> None:
    duplicate_mask = selected.duplicated(["dataset_num", "target_mode", "family"], keep=False)
    if duplicate_mask.any():
        duplicates = selected.loc[duplicate_mask, ["dataset_num", "target_mode", "family", "run_name"]]
        raise RuntimeError(
            "Predictor-family summary must use exactly one selected configuration "
            "per dataset, target mode, and family. Duplicate selections:\n"
            f"{duplicates.to_string(index=False)}"
        )


def load_naive_rows(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(
            f"Missing naive baseline CSV: {path}. Run metric-prediction/scripts/plot_naive_current_value_baseline.py first."
        )
    naive = pd.read_csv(path)
    naive["family"] = "Naive current value"
    naive["run_name"] = "current_value"
    naive["representation"] = "current_value"
    naive["model"] = "naive"
    naive["mae"] = naive.get("mae", np.nan)
    naive["n_validation"] = naive.get("n", 0)
    naive["source_path"] = naive.get("source_path", "")
    return naive[
        [
            "dataset_num",
            "dataset",
            "target_mode",
            "representation",
            "model",
            "family",
            "run_name",
            "r2",
            "rmse",
            "mae",
            "n_validation",
            "source_path",
        ]
    ]


def load_baseline_metric_rows(path: Path, *, family: str) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing baseline metrics CSV: {path}")
    baseline = pd.read_csv(path)
    baseline["family"] = family
    baseline["run_name"] = baseline.get("run_name", family.lower().replace(" ", "_"))
    baseline["representation"] = "learning_curve"
    baseline["model"] = family.lower().replace(" ", "_")
    baseline["mae"] = baseline.get("mae", np.nan)
    baseline["n_validation"] = baseline.get("n_validation", baseline.get("n", 0))
    baseline["source_path"] = baseline.get("source_path", "")
    return baseline[
        [
            "dataset_num",
            "dataset",
            "target_mode",
            "representation",
            "model",
            "family",
            "run_name",
            "r2",
            "rmse",
            "mae",
            "n_validation",
            "source_path",
        ]
    ]


def add_naive_comparison(selected_configurations: pd.DataFrame, naive: pd.DataFrame) -> pd.DataFrame:
    assert_unique_selected_configurations(selected_configurations)
    baseline = naive.rename(columns={"r2": "naive_r2", "rmse": "naive_rmse"})[
        ["dataset_num", "target_mode", "naive_r2", "naive_rmse"]
    ]
    compared = selected_configurations.merge(baseline, on=["dataset_num", "target_mode"], how="left")
    compared["delta_r2_vs_naive"] = compared["r2"] - compared["naive_r2"]
    compared["rmse_reduction_vs_naive"] = compared["naive_rmse"] - compared["rmse"]
    compared["rmse_reduction_pct_vs_naive"] = 100.0 * compared["rmse_reduction_vs_naive"] / compared["naive_rmse"]
    return compared


def add_learning_curve_baseline_comparisons(
    selected_configurations: pd.DataFrame,
    *,
    lcnet: pd.DataFrame,
    extrapolation: pd.DataFrame,
) -> pd.DataFrame:
    compared = selected_configurations.copy()
    for prefix, baseline in [
        ("lcnet", lcnet),
        ("learning_curve", extrapolation),
    ]:
        baseline_cols = baseline.rename(
            columns={
                "r2": f"{prefix}_r2",
                "rmse": f"{prefix}_rmse",
            }
        )[["dataset_num", "target_mode", f"{prefix}_r2", f"{prefix}_rmse"]]
        compared = compared.merge(baseline_cols, on=["dataset_num", "target_mode"], how="left")
        compared[f"delta_r2_vs_{prefix}"] = compared["r2"] - compared[f"{prefix}_r2"]
        compared[f"rmse_reduction_vs_{prefix}"] = compared[f"{prefix}_rmse"] - compared["rmse"]
        compared[f"rmse_reduction_pct_vs_{prefix}"] = (
            100.0 * compared[f"rmse_reduction_vs_{prefix}"] / compared[f"{prefix}_rmse"]
        )
    compared["representation_group"] = compared["family"].map(representation_group)
    return compared


def add_naive_comparison_for_ablation(selected: pd.DataFrame, naive: pd.DataFrame) -> pd.DataFrame:
    baseline = naive.rename(columns={"r2": "naive_r2", "rmse": "naive_rmse"})[
        ["dataset_num", "target_mode", "naive_r2", "naive_rmse"]
    ]
    compared = selected.merge(baseline, on=["dataset_num", "target_mode"], how="left")
    compared["delta_r2_vs_naive"] = compared["r2"] - compared["naive_r2"]
    compared["rmse_reduction_vs_naive"] = compared["naive_rmse"] - compared["rmse"]
    compared["rmse_reduction_pct_vs_naive"] = 100.0 * compared["rmse_reduction_vs_naive"] / compared["naive_rmse"]
    return compared


def build_summary(compared: pd.DataFrame) -> pd.DataFrame:
    assert_unique_selected_configurations(compared)
    usable = compared[np.isfinite(compared["rmse_reduction_pct_vs_naive"])].copy()
    summary = (
        usable.groupby(["family", "target_mode"], as_index=False)
        .agg(
            n_datasets=("dataset_num", "nunique"),
            mean_r2=("r2", "mean"),
            mean_rmse=("rmse", "mean"),
            mean_delta_r2_vs_naive=("delta_r2_vs_naive", "mean"),
            mean_rmse_reduction_pct_vs_naive=("rmse_reduction_pct_vs_naive", "mean"),
            median_rmse_reduction_pct_vs_naive=("rmse_reduction_pct_vs_naive", "median"),
        )
    )
    return sort_by_orders(summary)


def build_ablation_summary(compared: pd.DataFrame) -> pd.DataFrame:
    usable = compared[np.isfinite(compared["rmse_reduction_pct_vs_naive"])].copy()
    summary = (
        usable.groupby(["family", "target_mode", "ablation"], as_index=False)
        .agg(
            n_datasets=("dataset_num", "nunique"),
            mean_r2=("r2", "mean"),
            mean_rmse=("rmse", "mean"),
            median_r2=("r2", "median"),
            median_rmse=("rmse", "median"),
            mean_delta_r2_vs_naive=("delta_r2_vs_naive", "mean"),
            mean_rmse_reduction_pct_vs_naive=("rmse_reduction_pct_vs_naive", "mean"),
            median_rmse_reduction_pct_vs_naive=("rmse_reduction_pct_vs_naive", "median"),
        )
    )
    result = summary.copy()
    result["family"] = pd.Categorical(result["family"], categories=FAMILY_ORDER, ordered=True)
    result["target_mode"] = pd.Categorical(result["target_mode"], categories=TARGET_ORDER, ordered=True)
    result["ablation"] = pd.Categorical(result["ablation"], categories=ABLATION_ORDER, ordered=True)
    return result.sort_values(["family", "target_mode", "ablation"]).reset_index(drop=True)


def build_winners(compared: pd.DataFrame, naive: pd.DataFrame) -> pd.DataFrame:
    candidates = pd.concat(
        [
            compared[
                [
                    "dataset_num",
                    "dataset",
                    "target_mode",
                    "family",
                    "run_name",
                    "r2",
                    "rmse",
                    "source_path",
                ]
            ],
            naive[
                [
                    "dataset_num",
                    "dataset",
                    "target_mode",
                    "family",
                    "run_name",
                    "r2",
                    "rmse",
                    "source_path",
                ]
            ],
        ],
        ignore_index=True,
    )
    winners = (
        candidates.sort_values(["dataset_num", "target_mode", "rmse", "r2"], ascending=[True, True, True, False])
        .drop_duplicates(["dataset_num", "target_mode"], keep="first")
        .sort_values(["dataset_num", "target_mode"])
        .reset_index(drop=True)
    )
    return winners


def representation_group(family: str) -> str:
    if str(family).startswith("Weight statistics"):
        return "Weight statistics"
    if "latent" in str(family).lower() or "Transformer" in str(family):
        return "SANE representation"
    return "Other"


def select_best_representation_groups(compared: pd.DataFrame) -> pd.DataFrame:
    eligible = compared[compared["representation_group"].isin(["Weight statistics", "SANE representation"])].copy()
    selected = (
        eligible.sort_values(
            ["dataset_num", "target_mode", "representation_group", "rmse", "r2"],
            ascending=[True, True, True, True, False],
        )
        .drop_duplicates(["dataset_num", "target_mode", "representation_group"], keep="first")
        .sort_values(["dataset_num", "target_mode", "representation_group"])
        .reset_index(drop=True)
    )
    return selected


def sort_by_orders(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    result["family"] = pd.Categorical(result["family"], categories=FAMILY_ORDER, ordered=True)
    result["target_mode"] = pd.Categorical(result["target_mode"], categories=TARGET_ORDER, ordered=True)
    return result.sort_values(["family", "target_mode"]).reset_index(drop=True)


def _text_color_for_bg(rgba: tuple[float, float, float, float]) -> str:
    r, g, b = rgba[:3]

    def _lin(c: float) -> float:
        return c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4

    lum = 0.2126 * _lin(r) + 0.7152 * _lin(g) + 0.0722 * _lin(b)
    return "#111111" if lum > 0.179 else "#ffffff"


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


def render_plot(selected: pd.DataFrame, output_path: Path, baselines: pd.DataFrame | None = None, *, target_mode_filter: str | None = None) -> None:
    if baselines is not None:
        selected = pd.concat([baselines, selected], ignore_index=True)

    target_modes = [t for t in TARGET_ORDER if target_mode_filter is None or t == target_mode_filter]
    all_families = [family for family in FAMILY_ORDER if family in set(selected["family"])]
    datasets = sort_datasets(selected[["dataset_num", "dataset"]].drop_duplicates())
    dataset_labels = [
        "\n".join(textwrap.wrap(str(row.dataset), width=16, break_long_words=False))
        for row in datasets.itertuples(index=False)
    ]

    norm = TwoSlopeNorm(vmin=-0.3, vcenter=0.0, vmax=1.0)
    n_rows = len(target_modes)
    fig_height = 7.5 if n_rows == 1 else 13.5
    fig, axes = plt.subplots(n_rows, 1, figsize=(11.0, fig_height), sharex=True, constrained_layout=True)
    if n_rows == 1:
        axes = [axes]
    image = None

    for ax, target_mode in zip(axes, target_modes):
        target_data = selected[selected["target_mode"] == target_mode]
        r2_grid = np.full((len(all_families), len(datasets)), np.nan)
        rmse_grid = np.full((len(all_families), len(datasets)), np.nan)
        for y_idx, family in enumerate(all_families):
            for x_idx, dataset_num in enumerate(datasets["dataset_num"]):
                row = target_data[
                    (target_data["family"] == family)
                    & (target_data["dataset_num"] == dataset_num)
                ]
                if row.empty:
                    continue
                r2_grid[y_idx, x_idx] = float(row["r2"].iloc[0])
                rmse_grid[y_idx, x_idx] = float(row["rmse"].iloc[0])

        masked_r2 = np.ma.masked_invalid(r2_grid)
        cmap_obj = plt.get_cmap("RdYlGn")
        image = ax.imshow(masked_r2, cmap=cmap_obj, norm=norm, aspect="auto")
        ax.set_title(target_mode)
        baseline_families_set = {"Naive current value", "LCNet baseline", "Learning-curve extrapolation baseline"}
        ax.set_yticks(np.arange(len(all_families)))
        ax.set_yticklabels([FAMILY_ABBREVIATIONS.get(family, family) for family in all_families])
        ax.set_xticks(np.arange(len(datasets)))
        ax.set_xticklabels(dataset_labels, rotation=35, ha="right")
        ax.set_xlim(-0.5, len(datasets) - 0.5)
        ax.set_ylim(len(all_families) - 0.5, -0.5)
        ax.set_xticks(np.arange(-0.5, len(datasets), 1), minor=True)
        ax.set_yticks(np.arange(-0.5, len(all_families), 1), minor=True)
        ax.grid(which="minor", color="#ffffff", linewidth=0.8)
        ax.tick_params(which="minor", bottom=False, left=False)

        best_rmse_per_dataset = target_data.groupby("dataset_num")["rmse"].min()
        best_families = set()
        for _, row in target_data.iterrows():
            best_rmse = best_rmse_per_dataset.get(row["dataset_num"])
            if np.isfinite(row["rmse"]) and np.isclose(row["rmse"], best_rmse, rtol=1e-10):
                best_families.add((row["dataset_num"], row["family"]))

        for y_idx, family in enumerate(all_families):
            for x_idx, dataset_num in enumerate(datasets["dataset_num"]):
                r2_value = r2_grid[y_idx, x_idx]
                rmse_value = rmse_grid[y_idx, x_idx]
                if not np.isfinite(r2_value) or not np.isfinite(rmse_value):
                    continue
                text_color = _text_color_for_bg(cmap_obj(norm(r2_value)))
                ax.text(
                    x_idx,
                    y_idx,
                    f"{r2_value:.2f}\n{rmse_value:.3f}",
                    ha="center",
                    va="center",
                    fontsize=BASE_FONT_SIZE - 4,
                    color=text_color,
                )
                if (dataset_num, family) in best_families:
                    ax.add_patch(
                        Rectangle(
                            (x_idx - 0.5, y_idx - 0.5),
                            1.0,
                            1.0,
                            fill=False,
                            edgecolor="#ff6b35",
                            linewidth=3.2,
                        )
                    )

    if image is not None:
        colorbar = fig.colorbar(image, ax=axes, fraction=0.024, pad=0.02, shrink=0.92)
        colorbar.set_label(r"Validation $R^2$")
    axes[-1].set_xlabel("Dataset (cell text: $R^2$ / RMSE; outline: lowest RMSE)")
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def render_rmse_plot(selected: pd.DataFrame, output_path: Path, baselines: pd.DataFrame | None = None, *, target_mode_filter: str | None = None) -> None:
    if baselines is not None:
        selected = pd.concat([baselines, selected], ignore_index=True)

    target_modes = [t for t in TARGET_ORDER if target_mode_filter is None or t == target_mode_filter]
    all_families = [family for family in FAMILY_ORDER if family in set(selected["family"])]
    datasets = sort_datasets(selected[["dataset_num", "dataset"]].drop_duplicates())
    dataset_labels = [
        "\n".join(textwrap.wrap(str(row.dataset), width=16, break_long_words=False))
        for row in datasets.itertuples(index=False)
    ]

    n_rows = len(target_modes)
    fig_height = 7.5 if n_rows == 1 else 13.5
    fig, axes = plt.subplots(n_rows, 1, figsize=(11.0, fig_height), sharex=True, constrained_layout=True)
    if n_rows == 1:
        axes = [axes]
    image = None

    for ax, target_mode in zip(axes, target_modes):
        target_data = selected[selected["target_mode"] == target_mode]
        r2_grid = np.full((len(all_families), len(datasets)), np.nan)
        rmse_grid = np.full((len(all_families), len(datasets)), np.nan)
        for y_idx, family in enumerate(all_families):
            for x_idx, dataset_num in enumerate(datasets["dataset_num"]):
                row = target_data[
                    (target_data["family"] == family)
                    & (target_data["dataset_num"] == dataset_num)
                ]
                if row.empty:
                    continue
                r2_grid[y_idx, x_idx] = float(row["r2"].iloc[0])
                rmse_grid[y_idx, x_idx] = float(row["rmse"].iloc[0])

        masked_rmse = np.ma.masked_invalid(rmse_grid)
        rmse_max = float(np.nanmax(masked_rmse))
        norm = plt.Normalize(vmin=0.0, vmax=rmse_max)
        cmap_obj = plt.get_cmap("RdYlGn_r")
        image = ax.imshow(masked_rmse, cmap=cmap_obj, norm=norm, aspect="auto")
        ax.set_title(target_mode)
        ax.set_yticks(np.arange(len(all_families)))
        ax.set_yticklabels([FAMILY_ABBREVIATIONS.get(family, family) for family in all_families])
        ax.set_xticks(np.arange(len(datasets)))
        ax.set_xticklabels(dataset_labels, rotation=35, ha="right")
        ax.set_xlim(-0.5, len(datasets) - 0.5)
        ax.set_ylim(len(all_families) - 0.5, -0.5)
        ax.set_xticks(np.arange(-0.5, len(datasets), 1), minor=True)
        ax.set_yticks(np.arange(-0.5, len(all_families), 1), minor=True)
        ax.grid(which="minor", color="#ffffff", linewidth=0.8)
        ax.tick_params(which="minor", bottom=False, left=False)

        best_rmse_per_dataset = target_data.groupby("dataset_num")["rmse"].min()
        best_families = set()
        for _, row in target_data.iterrows():
            best_rmse = best_rmse_per_dataset.get(row["dataset_num"])
            if np.isfinite(row["rmse"]) and np.isclose(row["rmse"], best_rmse, rtol=1e-10):
                best_families.add((row["dataset_num"], row["family"]))

        for y_idx, family in enumerate(all_families):
            for x_idx, dataset_num in enumerate(datasets["dataset_num"]):
                r2_value = r2_grid[y_idx, x_idx]
                rmse_value = rmse_grid[y_idx, x_idx]
                if not np.isfinite(r2_value) or not np.isfinite(rmse_value):
                    continue
                text_color = _text_color_for_bg(cmap_obj(norm(rmse_value)))
                ax.text(
                    x_idx,
                    y_idx,
                    f"{r2_value:.2f}\n{rmse_value:.3f}",
                    ha="center",
                    va="center",
                    fontsize=BASE_FONT_SIZE - 4,
                    color=text_color,
                )
                if (dataset_num, family) in best_families:
                    ax.add_patch(
                        Rectangle(
                            (x_idx - 0.5, y_idx - 0.5),
                            1.0,
                            1.0,
                            fill=False,
                            edgecolor="#ff6b35",
                            linewidth=3.2,
                        )
                    )

    if image is not None:
        colorbar = fig.colorbar(image, ax=axes, fraction=0.024, pad=0.02, shrink=0.92)
        colorbar.set_label("Validation RMSE")
    axes[-1].set_xlabel("Dataset (cell text: $R^2$ / RMSE; outline: lowest RMSE)")
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def render_section_family_plots(compared: pd.DataFrame, output_dir: Path) -> None:
    for file_stem, families in SECTION_FAMILY_PLOTS.items():
        section_data = compared[compared["family"].isin(families)].copy()
        if section_data.empty:
            continue
        render_family_dataset_plot(
            section_data,
            output_dir / f"predictor_{file_stem}.pdf",
        )


def render_family_dataset_plot(data: pd.DataFrame, output_path: Path) -> None:
    families = [family for family in FAMILY_ORDER if family in set(data["family"])]
    datasets = sort_datasets(data[["dataset_num", "dataset"]].drop_duplicates())
    dataset_labels = [
        "\n".join(textwrap.wrap(str(row.dataset), width=28, break_long_words=False))
        for row in datasets.itertuples(index=False)
    ]
    series = [(family, target_mode) for family in families for target_mode in TARGET_ORDER]
    fig_height = max(4.6, 0.52 * len(datasets) + 1.3)
    fig_width = 10.8 if len(families) == 1 else 13.6
    fig, axes = plt.subplots(1, 2, figsize=(fig_width, fig_height), sharey=True)

    y = np.arange(len(datasets))
    bar_height = min(0.28, 0.78 / max(len(series), 1))
    offsets = (np.arange(len(series)) - (len(series) - 1) / 2.0) * bar_height
    target_hatch = {"Final value": "", "Horizon 5": "///"}
    target_edge = {"Final value": "#222222", "Horizon 5": "#ffffff"}

    metric_specs = [
        ("rmse", "RMSE", "Validation RMSE"),
        ("r2", r"$R^2$", r"Validation $R^2$"),
    ]
    for ax, (metric, xlabel, title) in zip(axes, metric_specs):
        for offset, (family, target_mode) in zip(offsets, series):
            values = []
            for dataset_num in datasets["dataset_num"]:
                row = data[
                    (data["dataset_num"] == dataset_num)
                    & (data["family"] == family)
                    & (data["target_mode"] == target_mode)
                ]
                values.append(float(row[metric].iloc[0]) if not row.empty else np.nan)
            label = target_mode if len(families) == 1 else f"{family}: {target_mode}"
            ax.barh(
                y + offset,
                values,
                bar_height,
                label=label,
                color=FAMILY_COLORS[family],
                edgecolor=target_edge[target_mode],
                linewidth=0.7,
                hatch=target_hatch[target_mode],
            )
        if metric == "r2":
            ax.axvline(0, color="#333333", linewidth=0.8)
            min_r2 = float(np.nanmin(data["r2"]))
            r2_left = min(-0.05, min_r2 - 0.05) if min_r2 < 0 else 0.0
            ax.set_xlim(left=r2_left, right=1.02)
        else:
            ax.set_xlim(left=0.0)
        ax.set_title(title)
        ax.set_xlabel(xlabel)
        ax.grid(axis="x", color="#d0d0d0", linewidth=0.6, alpha=0.8)
        ax.set_axisbelow(True)

    axes[0].set_yticks(y)
    axes[0].set_yticklabels(dataset_labels)
    axes[0].invert_yaxis()

    handles, labels = axes[0].get_legend_handles_labels()
    legend_columns = 2 if len(families) == 1 else min(4, len(labels))
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 1.02), ncol=legend_columns, frameon=False)
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.95), w_pad=2.0)
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def render_ablation_plot(summary: pd.DataFrame, output_path: Path, *, title: str | None = None) -> None:
    families = [family for family in FAMILY_ORDER if family != "Naive current value" and family in set(summary["family"].astype(str))]
    y = np.arange(len(families))
    bar_height = 0.22
    offsets = {
        "Performance only": -bar_height,
        "Representation only": 0.0,
        "Representation + performance": bar_height,
    }
    fig, axes = plt.subplots(1, 2, figsize=(14.0, 6.8), sharey=True)
    x_min, x_max = 0.0, 1.0
    axis_min, axis_max = 0.0, 1.16

    for ax, target_mode in zip(axes, TARGET_ORDER):
        target_data = summary[summary["target_mode"].astype(str) == target_mode]
        for ablation in ABLATION_ORDER:
            values = []
            for family in families:
                if family == "Latent tokens + Transformer" and ablation == "Performance only":
                    values.append(np.nan)
                    continue
                row = target_data[
                    (target_data["family"].astype(str) == family)
                    & (target_data["ablation"].astype(str) == ablation)
                ]
                r2_value = (
                    float(row["median_r2"].iloc[0])
                    if not row.empty
                    else np.nan
                )
                values.append(float(np.clip(r2_value, x_min, x_max)) if np.isfinite(r2_value) else np.nan)
            ax.barh(
                y + offsets[ablation],
                values,
                bar_height,
                label=ablation,
                color=ABLATION_COLORS[ablation],
                edgecolor="#222222",
                linewidth=0.5,
            )
        ax.set_title(target_mode)
        ax.set_xlim(axis_min, axis_max)
        ax.set_xlabel(r"Median validation $R^2$")
        ax.grid(axis="x", color="#d0d0d0", linewidth=0.6, alpha=0.8)
        ax.set_axisbelow(True)

    axes[0].set_yticks(y)
    axes[0].set_yticklabels(["\n".join(textwrap.wrap(family, width=23, break_long_words=False)) for family in families])
    axes[0].invert_yaxis()
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 1.03), ncol=3, frameon=False)
    top = 0.82 if title else 0.88
    if title:
        fig.suptitle(title, fontsize=BASE_FONT_SIZE + 2, y=0.97)
    fig.subplots_adjust(left=0.24, right=0.98, top=top, bottom=0.12, wspace=0.16)
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def render_ablation_rmse_plot(summary: pd.DataFrame, output_path: Path, *, title: str | None = None) -> None:
    families = [family for family in FAMILY_ORDER if family != "Naive current value" and family in set(summary["family"].astype(str))]
    y = np.arange(len(families))
    bar_height = 0.22
    offsets = {
        "Performance only": -bar_height,
        "Representation only": 0.0,
        "Representation + performance": bar_height,
    }
    excluded = summary[
        (summary["family"].astype(str) == "Latent tokens + Transformer")
        & (summary["ablation"].astype(str) == "Performance only")
    ].index
    finite_rmse = summary.drop(index=excluded)["median_rmse"].dropna().to_numpy(dtype="float64")
    finite_rmse = finite_rmse[np.isfinite(finite_rmse)]
    x_cap = float(np.max(finite_rmse)) * 1.12 if finite_rmse.size else 0.1

    fig, axes = plt.subplots(1, 2, figsize=(14.0, 6.8), sharey=True)

    for ax, target_mode in zip(axes, TARGET_ORDER):
        target_data = summary[summary["target_mode"].astype(str) == target_mode]
        for ablation in ABLATION_ORDER:
            raw_values = []
            display_values = []
            for family in families:
                if family == "Latent tokens + Transformer" and ablation == "Performance only":
                    raw_values.append(np.nan)
                    display_values.append(np.nan)
                    continue
                row = target_data[
                    (target_data["family"].astype(str) == family)
                    & (target_data["ablation"].astype(str) == ablation)
                ]
                rmse_value = float(row["median_rmse"].iloc[0]) if not row.empty else np.nan
                raw_values.append(rmse_value)
                display_values.append(min(rmse_value, x_cap) if np.isfinite(rmse_value) else np.nan)
            bars = ax.barh(
                y + offsets[ablation],
                display_values,
                bar_height,
                label=ablation,
                color=ABLATION_COLORS[ablation],
                edgecolor="#222222",
                linewidth=0.5,
            )
            for bar, raw, display in zip(bars, raw_values, display_values):
                if np.isfinite(raw) and raw > x_cap:
                    ax.text(
                        x_cap - (x_cap * 0.02),
                        bar.get_y() + bar.get_height() / 2,
                        f"{raw:.3f}",
                        ha="right",
                        va="center",
                        fontsize=BASE_FONT_SIZE - 3,
                        color="#ffffff",
                    )
        ax.set_title(target_mode)
        ax.set_xlim(0.0, x_cap)
        ax.set_xlabel("Median validation RMSE")
        ax.grid(axis="x", color="#d0d0d0", linewidth=0.6, alpha=0.8)
        ax.set_axisbelow(True)

    axes[0].set_yticks(y)
    axes[0].set_yticklabels(["\n".join(textwrap.wrap(family, width=23, break_long_words=False)) for family in families])
    axes[0].invert_yaxis()
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 1.03), ncol=3, frameon=False)
    top = 0.82 if title else 0.88
    if title:
        fig.suptitle(title, fontsize=BASE_FONT_SIZE + 2, y=0.97)
    fig.subplots_adjust(left=0.24, right=0.98, top=top, bottom=0.12, wspace=0.16)
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def render_ablation_plots_per_zoo(ablation_compared: pd.DataFrame, output_dir: Path) -> None:
    zoo_dir = output_dir / "ablation_per_zoo"
    zoo_dir.mkdir(parents=True, exist_ok=True)
    for _, group in ablation_compared.groupby("dataset_num"):
        dataset_name = str(group["dataset"].iloc[0])
        summary = build_ablation_summary(group)
        if summary.empty:
            continue
        safe_name = re.sub(r"[^a-zA-Z0-9]+", "_", dataset_name).strip("_").lower()
        render_ablation_plot(
            summary,
            zoo_dir / f"predictor_feature_ablation_{safe_name}.pdf",
            title=dataset_name,
        )
        render_ablation_rmse_plot(
            summary,
            zoo_dir / f"predictor_feature_ablation_rmse_{safe_name}.pdf",
            title=dataset_name,
        )


def render_representation_baseline_improvement_plot(data: pd.DataFrame, output_path: Path) -> None:
    datasets = sort_datasets(data[["dataset_num", "dataset"]].drop_duplicates())
    dataset_labels = [
        "\n".join(textwrap.wrap(str(row.dataset), width=28, break_long_words=False))
        for row in datasets.itertuples(index=False)
    ]
    groups = ["Weight statistics", "SANE representation"]
    colors = {
        "Weight statistics": "#2364aa",
        "SANE representation": "#d1495b",
    }
    series = [(group, target_mode) for group in groups for target_mode in TARGET_ORDER]
    target_hatch = {"Final value": "", "Horizon 5": "///"}
    target_edge = {"Final value": "#222222", "Horizon 5": "#ffffff"}
    font_size = BASE_FONT_SIZE + 2
    group_spacing = 2.2
    y = np.arange(len(datasets)) * group_spacing
    bar_height = min(0.50, 2.0 / max(len(series), 1))
    offsets = (np.arange(len(series)) - (len(series) - 1) / 2.0) * bar_height
    fig, ax = plt.subplots(figsize=(12.8, 11.5))

    finite_values = data["rmse_reduction_pct_vs_lcnet"].to_numpy(dtype="float64")
    finite_values = finite_values[np.isfinite(finite_values)]
    clipped_values = np.maximum(finite_values, 0.0)
    max_value = float(np.max(clipped_values)) if clipped_values.size else 10.0
    x_left = 0.0
    x_right = max(10.0, max_value + 12.0)

    for offset, (group, target_mode) in zip(offsets, series):
        target_data = data[data["target_mode"] == target_mode]
        values = []
        labels = []
        for dataset_num in datasets["dataset_num"]:
            row = target_data[
                (target_data["dataset_num"] == dataset_num)
                & (target_data["representation_group"] == group)
            ]
            if row.empty:
                values.append(np.nan)
                labels.append("")
            else:
                values.append(max(0.0, float(row["rmse_reduction_pct_vs_lcnet"].iloc[0])))
                labels.append(short_family_label(str(row["family"].iloc[0])))
        bars = ax.barh(
            y + offset,
            values,
            bar_height,
            label=f"{group}: {target_mode}",
            color=colors[group],
            edgecolor=target_edge[target_mode],
            linewidth=0.7,
            hatch=target_hatch[target_mode],
        )
        for bar, value, label in zip(bars, values, labels):
            if not np.isfinite(value) or value < 2.0:
                continue
            ax.text(
                value + 1.4,
                bar.get_y() + bar.get_height() * 0.65,
                label,
                va="center",
                ha="left",
                fontsize=font_size - 3,
            )

    ax.axvline(0, color="#333333", linewidth=0.9)
    ax.set_xlim(x_left, x_right)
    ax.set_xlabel("RMSE reduction vs LCNet baseline (%)", fontsize=font_size)
    ax.grid(axis="x", color="#d0d0d0", linewidth=0.6, alpha=0.8)
    ax.set_axisbelow(True)
    ax.tick_params(axis="x", labelsize=font_size - 1)
    ax.set_yticks(y)
    ax.set_yticklabels(dataset_labels, fontsize=font_size - 1)
    ax.invert_yaxis()
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.56, 1.02),
        ncol=2,
        frameon=False,
        fontsize=font_size - 1,
    )
    fig.subplots_adjust(left=0.27, right=0.99, top=0.88, bottom=0.1)
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def render_lcnet_improvement_table(data: pd.DataFrame, output_path: Path) -> None:
    families = [family for family in FAMILY_ORDER if family != "Naive current value" and family in set(data["family"])]
    headers = {
        "Weight statistics + XGBoost": "WS-XGB",
        "Weight statistics + MLP": "WS-MLP",
        "Averaged latent + Ridge": "Avg-Ridge",
        "Averaged latent + XGBoost": "Avg-XGB",
        "Averaged latent + MLP": "Avg-MLP",
        "Latent windows + LSTM": "LW-LSTM",
        "Latent tokens + Transformer": "LT-Trans.",
    }
    lines = [
        "\\begin{table}[H]",
        "\\centering",
        "\\caption[Predictor improvements over LCNet baseline]{\\textbf{Predictor Improvements over LCNet Baseline.} Validation RMSE reduction in percent relative to LCNet. Positive values indicate lower RMSE than LCNet, while negative values indicate worse performance. The best predictor family in each row is shown in bold.}",
        "\\label{tab:predictor_improvement_vs_lcnet}",
        "\\begingroup",
        "\\setlength{\\tabcolsep}{3pt}",
    ]
    for target_mode in TARGET_ORDER:
        target_rows = data[data["target_mode"] == target_mode].copy()
        pivot = target_rows.pivot_table(
            index="dataset",
            columns="family",
            values="rmse_reduction_pct_vs_lcnet",
            aggfunc="first",
        )
        pivot = pivot.reindex(DATASET_ORDER)
        pivot = pivot[[family for family in families if family in pivot.columns]]
        if target_mode != TARGET_ORDER[0]:
            lines.append("\\par\\vspace{1.8em}\\noindent")
        lines.extend(
            [
                "\\begin{minipage}{\\textwidth}",
                f"\\noindent\\textbf{{{target_mode}.}}\\\\[0.45em]",
                "\\resizebox{\\textwidth}{!}{%",
                "\\begin{tabular}{@{}l" + "r" * len(families) + "@{}}",
                "\\hline",
                "\\textbf{Dataset} & "
                + " & ".join(f"\\textbf{{{headers.get(family, latex_escape(family))}}}" for family in families)
                + " \\\\",
                "\\hline",
            ]
        )
        for dataset, values in pivot.iterrows():
            finite = values.dropna()
            best_value = float(finite.max()) if not finite.empty else np.nan
            cells = []
            for family in families:
                value = values.get(family, np.nan)
                if not np.isfinite(value):
                    cells.append("--")
                    continue
                formatted = f"{value:.1f}"
                if np.isfinite(best_value) and np.isclose(float(value), best_value) and best_value > 0:
                    formatted = f"\\textbf{{{formatted}}}"
                cells.append(formatted)
            lines.append(f"{latex_escape(str(dataset))} & " + " & ".join(cells) + " \\\\")
        lines.extend(
            [
                "\\hline",
                "\\end{tabular}%",
                "}",
                "\\end{minipage}",
            ]
        )
    lines.extend(["\\endgroup", "\\end{table}", ""])
    output_path.write_text("\n".join(lines), encoding="utf-8")


def latex_escape(text: str) -> str:
    replacements = {
        "\\": "\\textbackslash{}",
        "&": "\\&",
        "%": "\\%",
        "$": "\\$",
        "#": "\\#",
        "_": "\\_",
        "{": "\\{",
        "}": "\\}",
        "~": "\\textasciitilde{}",
        "^": "\\textasciicircum{}",
    }
    return "".join(replacements.get(char, char) for char in text)


def short_family_label(family: str) -> str:
    labels = {
        "Weight statistics + XGBoost": "XGB (Wt. stats)",
        "Weight statistics + MLP": "MLP (Wt. stats)",
        "Averaged latent + Ridge": "Ridge (Avg. latent)",
        "Averaged latent + XGBoost": "XGB (Avg. latent)",
        "Averaged latent + MLP": "MLP (Avg. latent)",
        "Latent windows + LSTM": "LSTM (Lat. windows)",
        "Latent tokens + Transformer": "Transformer (Lat. tokens)",
    }
    return labels.get(family, "")


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    configure_plot_style()

    plot_names = load_plot_names(args.model_zoo_info)
    runs = load_metric_rows(
        args.forecasting_root,
        plot_names,
        horizon=args.horizon,
        include_perf_only=False,
    )
    ablation_runs = load_metric_rows(
        args.forecasting_root,
        plot_names,
        horizon=args.horizon,
        include_perf_only=True,
    )
    selected_configurations = select_best_configurations(runs)
    naive = load_naive_rows(args.naive_metrics)
    lcnet = load_baseline_metric_rows(args.lcnet_metrics, family="LCNet baseline")
    learning_curve_extrapolation = load_baseline_metric_rows(
        args.learning_curve_metrics,
        family="Learning-curve extrapolation baseline",
    )
    compared = add_naive_comparison(selected_configurations, naive)
    baseline_compared = add_learning_curve_baseline_comparisons(
        selected_configurations,
        lcnet=lcnet,
        extrapolation=learning_curve_extrapolation,
    )
    representation_baseline_compared = select_best_representation_groups(baseline_compared)
    summary = build_summary(compared)
    winners = build_winners(compared, naive)
    selected_ablation_configurations = select_best_ablation_configurations(ablation_runs)
    ablation_compared = add_naive_comparison_for_ablation(selected_ablation_configurations, naive)
    ablation_summary = build_ablation_summary(ablation_compared)

    selected_configurations.to_csv(args.output_dir / "predictor_family_best_runs.csv", index=False)
    compared.to_csv(args.output_dir / "predictor_family_best_runs_vs_naive.csv", index=False)
    baseline_compared.to_csv(args.output_dir / "predictor_family_best_runs_vs_learning_curve_baselines.csv", index=False)
    representation_baseline_compared.to_csv(
        args.output_dir / "predictor_representation_best_runs_vs_lcnet.csv",
        index=False,
    )
    summary.to_csv(args.output_dir / "predictor_family_summary.csv", index=False)
    winners.to_csv(args.output_dir / "predictor_family_winners.csv", index=False)
    selected_ablation_configurations.to_csv(args.output_dir / "predictor_feature_ablation_best_runs.csv", index=False)
    ablation_compared.to_csv(args.output_dir / "predictor_feature_ablation_best_runs_vs_naive.csv", index=False)
    ablation_summary.to_csv(args.output_dir / "predictor_feature_ablation_summary.csv", index=False)

    baseline_families = pd.concat([naive, lcnet, learning_curve_extrapolation], ignore_index=True)
    render_plot(selected_configurations, args.output_dir / "predictor_family_comparison.pdf", baselines=baseline_families)
    render_plot(selected_configurations, args.output_dir / "predictor_family_comparison_final_value.pdf", baselines=baseline_families, target_mode_filter="Final value")
    render_plot(selected_configurations, args.output_dir / "predictor_family_comparison_horizon5.pdf", baselines=baseline_families, target_mode_filter="Horizon 5")
    render_rmse_plot(selected_configurations, args.output_dir / "predictor_family_comparison_rmse.pdf", baselines=baseline_families)
    render_rmse_plot(selected_configurations, args.output_dir / "predictor_family_comparison_rmse_final_value.pdf", baselines=baseline_families, target_mode_filter="Final value")
    render_rmse_plot(selected_configurations, args.output_dir / "predictor_family_comparison_rmse_horizon5.pdf", baselines=baseline_families, target_mode_filter="Horizon 5")
    render_section_family_plots(compared, args.output_dir)
    render_ablation_plot(ablation_summary, args.output_dir / "predictor_feature_ablation.pdf")
    render_ablation_rmse_plot(ablation_summary, args.output_dir / "predictor_feature_ablation_rmse.pdf")
    render_ablation_plots_per_zoo(ablation_compared, args.output_dir)
    render_representation_baseline_improvement_plot(
        representation_baseline_compared,
        args.output_dir / "predictor_representation_improvement_vs_lcnet.pdf",
    )
    render_lcnet_improvement_table(
        baseline_compared,
        args.output_dir / "predictor_improvement_vs_lcnet_table.tex",
    )

    print(f"Wrote summary artifacts to {args.output_dir}")


if __name__ == "__main__":
    main()
