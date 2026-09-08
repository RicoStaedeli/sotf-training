"""
Plot LC Baseline Comparison for SOTF thesis experiments.

Generates two PDFs:
  1. lc_baselines_vs_naive.pdf  — Naive vs LC baselines (all datasets)
  2. predictors_vs_lc_baselines.pdf — Key predictors vs LC baselines (datasets with all data)
"""

import json
import re
from pathlib import Path
from collections import defaultdict

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# ---------------------------------------------------------------------------
# Style
# ---------------------------------------------------------------------------
plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["font.size"] = 16
plt.rcParams["axes.titlesize"] = 18
plt.rcParams["axes.labelsize"] = 16
plt.rcParams["xtick.labelsize"] = 15
plt.rcParams["ytick.labelsize"] = 15
plt.rcParams["legend.fontsize"] = 15

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
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
FIG_DIR.mkdir(parents=True, exist_ok=True)

DATASET_DISPLAY = {
    1: "C10 CNN Uni.",
    2: "C10 CNN Hyp.R",
    3: "C10 CNN Hyp.F",
    5: "C10 ResNet-18",
    7: "C10 ViT",
    8: "STL CNN Uni.",
    9: "STL CNN Hyp.R",
    10: "STL CNN Hyp.F",
    14: "TinyImg. Phase",
}

DATASET_ORDER = [1, 2, 3, 8, 9, 10, 14, 5, 7]

TARGET_MODES = ["final_value", "horizon_value"]

# Experiments to load from lc_baselines directories
LC_EXPERIMENTS = {"lc_extrapolation_baseline", "lcnet_baseline"}

# Experiments to load from standard ds directories
STANDARD_EXPERIMENTS = {
    "naive_baseline",
    "online_xgboost_perf_only",
    "weight_statistics_online_xgboost_with_perf",
}

# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def parse_sweep_dir(sweep_dir_name: str):
    """Extract start_epoch and survival_rate from a sweep dir name like se05_sr10_i5."""
    m = re.match(r"se(\d+)_sr(\d+)_i(\d+)", sweep_dir_name)
    if not m:
        return None, None
    return int(m.group(1)), int(m.group(2))


def load_run(exp_dir: Path) -> dict | None:
    """Load simulation_summary + run_manifest from an experiment leaf dir."""
    summary_path = exp_dir / "simulation_summary.json"
    manifest_path = exp_dir / "run_manifest.json"
    if not summary_path.exists() or not manifest_path.exists():
        return None
    try:
        summary = json.loads(summary_path.read_text())
        manifest = json.loads(manifest_path.read_text())
    except Exception as e:
        print(f"  WARNING: could not read {exp_dir}: {e}")
        return None

    oracle = summary.get("oracle_best_final_target")
    survivor = summary.get("best_survivor_final_target")
    if oracle is None or survivor is None:
        return None

    return {
        "dataset_id": manifest.get("dataset_id"),
        "dataset_name": manifest.get("dataset_name", ""),
        "experiment_name": manifest.get("experiment_name"),
        "predictor": manifest.get("predictor"),
        "target_mode": manifest.get("target_mode"),
        "start_epoch": manifest.get("selection", {}).get("start_epoch"),
        "survival_rate": manifest.get("selection", {}).get("survival_rate"),
        "oracle_best": oracle,
        "best_survivor": survivor,
        "gap": oracle - survivor,
        "retained_oracle_best": summary.get("retained_oracle_best", False),
        "oracle_top_k_recall": summary.get("oracle_top_k_recall"),
    }


def discover_experiment_dirs(batch_subdir: Path, wanted_experiments: set) -> list[Path]:
    """Find all experiment leaf dirs under a batch subdir matching wanted_experiments."""
    found = []
    if not batch_subdir.exists():
        return found
    # Structure: batch_subdir / sweep_dir / results / predictor / test_acc / target_mode / dataset_dir / experiment_name
    for sweep_dir in batch_subdir.iterdir():
        if not sweep_dir.is_dir():
            continue
        results_dir = sweep_dir / "results"
        if not results_dir.exists():
            continue
        for predictor_dir in results_dir.iterdir():
            if not predictor_dir.is_dir():
                continue
            test_acc_dir = predictor_dir / "test_acc"
            if not test_acc_dir.exists():
                continue
            for tm_dir in test_acc_dir.iterdir():
                if not tm_dir.is_dir():
                    continue
                for dataset_dir in tm_dir.iterdir():
                    if not dataset_dir.is_dir():
                        continue
                    for exp_dir in dataset_dir.iterdir():
                        if exp_dir.is_dir() and exp_dir.name in wanted_experiments:
                            found.append(exp_dir)
    return found


def load_all_data() -> list[dict]:
    """Load all relevant runs from both target_modes and both ds/lc_baselines dirs."""
    all_records = []

    for tm in TARGET_MODES:
        tm_root = BATCH_ROOT / tm
        if not tm_root.exists():
            print(f"WARNING: {tm_root} does not exist")
            continue

        for batch_subdir in sorted(tm_root.iterdir()):
            if not batch_subdir.is_dir():
                continue
            name = batch_subdir.name

            is_lc = name.startswith("lc_baselines_ds")
            is_ds = re.match(r"^ds\d+$", name) is not None

            if not (is_lc or is_ds):
                continue

            wanted = LC_EXPERIMENTS if is_lc else STANDARD_EXPERIMENTS
            exp_dirs = discover_experiment_dirs(batch_subdir, wanted)

            for exp_dir in exp_dirs:
                record = load_run(exp_dir)
                if record is not None:
                    all_records.append(record)

    print(f"Loaded {len(all_records)} runs total.")
    return all_records


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------

def aggregate(records: list[dict], key_fields: list[str]) -> dict:
    """Group records by key_fields and compute mean gap."""
    groups = defaultdict(list)
    for r in records:
        key = tuple(r.get(f) for f in key_fields)
        groups[key].append(r["gap"])

    result = {}
    for key, gaps in groups.items():
        result[key] = {"mean_gap": np.mean(gaps), "n": len(gaps)}
    return result


# ---------------------------------------------------------------------------
# Plot helpers
# ---------------------------------------------------------------------------

PALETTE = {
    "naive_baseline": "#4C72B0",
    "lc_extrapolation_baseline": "#DD8452",
    "lcnet_baseline": "#55A868",
    "online_xgboost_perf_only": "#C44E52",
    "weight_statistics_online_xgboost_with_perf": "#8172B2",
}

LABELS = {
    "naive_baseline": "Naive",
    "lc_extrapolation_baseline": "LC Extrap.",
    "lcnet_baseline": "LCNet",
    "online_xgboost_perf_only": "XGB Perf-Only",
    "weight_statistics_online_xgboost_with_perf": "XGB + Wt. Stats",
}

TM_TITLES = {
    "final_value": "Final-Value Mode",
    "horizon_value": "Horizon-Value Mode",
}


def build_bar_chart(
    ax,
    records: list[dict],
    experiment_names: list[str],
    dataset_order: list[int],
    target_mode: str,
    title: str,
):
    """Draw a grouped bar chart on ax."""
    # Filter to this target_mode
    tm_records = [r for r in records if r["target_mode"] == target_mode]

    # Aggregate: mean gap per (dataset_id, experiment_name)
    agg = defaultdict(list)
    for r in tm_records:
        if r["experiment_name"] in experiment_names:
            agg[(r["dataset_id"], r["experiment_name"])].append(r["gap"])

    # Determine which datasets actually have data
    available_datasets = sorted(
        {ds for ds in dataset_order if any(k[0] == ds for k in agg)},
        key=lambda x: dataset_order.index(x) if x in dataset_order else 999,
    )

    n_ds = len(available_datasets)
    n_exp = len(experiment_names)
    bar_width = 0.8 / n_exp
    x = np.arange(n_ds)

    for i, exp_name in enumerate(experiment_names):
        gaps = []
        for ds_id in available_datasets:
            vals = agg.get((ds_id, exp_name), [])
            gaps.append(np.mean(vals) if vals else np.nan)

        offset = (i - n_exp / 2 + 0.5) * bar_width
        bars = ax.bar(
            x + offset,
            gaps,
            width=bar_width * 0.9,
            color=PALETTE.get(exp_name, "#888888"),
            label=LABELS.get(exp_name, exp_name),
            zorder=3,
        )

    ax.set_xticks(x)
    ax.set_xticklabels(
        [DATASET_DISPLAY.get(ds, str(ds)) for ds in available_datasets],
        rotation=30,
        ha="right",
    )
    ax.set_ylabel("Mean Best-Survivor Gap (↓ better)")
    ax.set_title(title)
    ax.yaxis.grid(True, linestyle="--", alpha=0.6, zorder=0)
    ax.set_axisbelow(True)
    ax.set_ylim(bottom=0)


# ---------------------------------------------------------------------------
# Plot 1: Naive vs LC baselines
# ---------------------------------------------------------------------------

def plot_naive_vs_lc(records: list[dict], out_stem: str):
    experiments = ["naive_baseline", "lc_extrapolation_baseline", "lcnet_baseline"]

    fig, axes = plt.subplots(2, 1, figsize=(14, 9), sharey=False)
    fig.suptitle(
        r"Naive vs.\ LC Baselines: Mean Best-Survivor Gap by Dataset",
        fontsize=18,
    )

    for ax, tm in zip(axes, TARGET_MODES):
        build_bar_chart(
            ax,
            records,
            experiments,
            DATASET_ORDER,
            tm,
            title=TM_TITLES[tm],
        )

    # Shared legend below
    handles = [
        mpatches.Patch(color=PALETTE[e], label=LABELS[e]) for e in experiments
    ]
    handles = [
        mpatches.Patch(color=PALETTE[e], label=LABELS[e]) for e in experiments
    ]
    fig.legend(
        handles=handles,
        loc="lower center",
        ncol=len(experiments),
        bbox_to_anchor=(0.5, -0.06),
        frameon=True,
    )

    fig.tight_layout()
    fig.subplots_adjust(bottom=0.14)
    for suffix in [".pdf", ".png"]:
        out_path = FIG_DIR / (out_stem + suffix)
        fig.savefig(out_path, bbox_inches="tight", dpi=200)
        print(f"Saved: {out_path}")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Plot 2: Key predictors vs LC baselines (datasets where ALL experiments present)
# ---------------------------------------------------------------------------

def plot_predictors_vs_lc(records: list[dict], out_stem: str):
    experiments = [
        "naive_baseline",
        "lc_extrapolation_baseline",
        "lcnet_baseline",
        "online_xgboost_perf_only",
        "weight_statistics_online_xgboost_with_perf",
    ]

    fig, axes = plt.subplots(2, 1, figsize=(14, 9), sharey=False)
    fig.suptitle(
        r"Key Predictors vs.\ LC Baselines: Mean Best-Survivor Gap by Dataset",
        fontsize=18,
    )

    for ax, tm in zip(axes, TARGET_MODES):
        tm_records = [r for r in records if r["target_mode"] == tm]

        # Aggregate per (ds_id, experiment)
        agg = defaultdict(list)
        for r in tm_records:
            if r["experiment_name"] in experiments:
                agg[(r["dataset_id"], r["experiment_name"])].append(r["gap"])

        # Only keep datasets that have data for ALL experiments
        valid_datasets = [
            ds for ds in DATASET_ORDER
            if all(len(agg.get((ds, e), [])) > 0 for e in experiments)
        ]

        if not valid_datasets:
            ax.text(
                0.5, 0.5, "No datasets with all experiments",
                ha="center", va="center", transform=ax.transAxes,
            )
            ax.set_title(TM_TITLES[tm])
            continue

        n_ds = len(valid_datasets)
        n_exp = len(experiments)
        bar_width = 0.8 / n_exp
        x = np.arange(n_ds)

        for i, exp_name in enumerate(experiments):
            gaps = [np.mean(agg.get((ds, exp_name), [np.nan])) for ds in valid_datasets]
            offset = (i - n_exp / 2 + 0.5) * bar_width
            ax.bar(
                x + offset,
                gaps,
                width=bar_width * 0.9,
                color=PALETTE.get(exp_name, "#888888"),
                label=LABELS.get(exp_name, exp_name),
                zorder=3,
            )

        ax.set_xticks(x)
        ax.set_xticklabels(
            [DATASET_DISPLAY.get(ds, str(ds)) for ds in valid_datasets],
            rotation=30,
            ha="right",
        )
        ax.set_ylabel("Mean Best-Survivor Gap (↓ better)")
        ax.set_title(TM_TITLES[tm])
        ax.yaxis.grid(True, linestyle="--", alpha=0.6, zorder=0)
        ax.set_axisbelow(True)
        ax.set_ylim(bottom=0)

    handles = [
        mpatches.Patch(color=PALETTE[e], label=LABELS[e]) for e in experiments
    ]
    fig.legend(
        handles=handles,
        loc="lower center",
        ncol=len(experiments),
        bbox_to_anchor=(0.5, -0.06),
        frameon=True,
    )

    fig.tight_layout()
    fig.subplots_adjust(bottom=0.14)
    for suffix in [".pdf", ".png"]:
        out_path = FIG_DIR / (out_stem + suffix)
        fig.savefig(out_path, bbox_inches="tight", dpi=200)
        print(f"Saved: {out_path}")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Diagnostics
# ---------------------------------------------------------------------------

def print_coverage(records: list[dict]):
    print("\n--- Coverage summary ---")
    for tm in TARGET_MODES:
        tm_r = [r for r in records if r["target_mode"] == tm]
        exps = sorted({r["experiment_name"] for r in tm_r})
        ds_ids = sorted({r["dataset_id"] for r in tm_r if r["dataset_id"] is not None})
        print(f"\n{tm}:")
        print(f"  Datasets: {ds_ids}")
        print(f"  Experiments: {exps}")
        for exp in exps:
            ds_with = sorted({r["dataset_id"] for r in tm_r if r["experiment_name"] == exp})
            print(f"    {exp}: datasets {ds_with}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    records = load_all_data()
    print_coverage(records)

    plot_naive_vs_lc(records, "lc_baselines_vs_naive")
    plot_predictors_vs_lc(records, "predictors_vs_lc_baselines")

    print("\nDone.")
