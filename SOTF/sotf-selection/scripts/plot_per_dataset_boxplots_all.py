"""
Per-dataset box plots comparing ALL predictors discovered in the output directory.

One PDF per dataset (ds_boxplots_all_{N}.pdf), 2×2 subplots:
  Row 1: Best-Survivor Gap      (lower is better)
  Row 2: Oracle Top-K Recall    (higher is better)
  Col 1: Final-Value Mode
  Col 2: Horizon-Value Mode

Each box = distribution of values across all (start_epoch, SR) conditions
for that dataset. One box per predictor. Automatically discovers and includes
all experiments found in the batch outputs.
"""

import json
import re
from pathlib import Path
from collections import defaultdict

import numpy as np
import matplotlib
import matplotlib.pyplot as plt

matplotlib.use("Agg")

plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["font.size"] = 14
plt.rcParams["axes.titlesize"] = 14
plt.rcParams["axes.labelsize"] = 13
plt.rcParams["xtick.labelsize"] = 12
plt.rcParams["ytick.labelsize"] = 12
plt.rcParams["legend.fontsize"] = 12

# ---------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parents[2]
BATCH_ROOT = REPO_ROOT / "sotf-selection" / "outputs" / "batch_experiments"
FIG_DIR = (
    REPO_ROOT / "sotf-selection" / "outputs"
    / "evaluation" / "boxplots_per_dataset_all"
)
FIG_DIR.mkdir(parents=True, exist_ok=True)

DATASET_ORDER = [1, 2, 3, 8, 9, 10, 14, 5, 7]
DATASET_DISPLAY = {
    1:  "CIFAR-10 CNN Uniform",       2:  "CIFAR-10 CNN Hyp. Rand.",
    3:  "CIFAR-10 CNN Hyp. Fixed",    5:  "CIFAR-10 ResNet-18 Width",
    7:  "CIFAR-10 ViT",               8:  "STL-10 CNN Uniform",
    9:  "STL-10 CNN Hyp. Rand.",      10: "STL-10 CNN Hyp. Fixed",
    14: "TinyImageNet Phase",
}

# Predefined labels and colors for known experiments
PREDEFINED_EXP_LABEL = {
    "lc_extrapolation_baseline":                  "LC Extrap.",
    "lcnet_baseline":                             "LCNet",
    "naive_baseline":                             "Naive",
    "online_xgboost_perf_only":                   "Online XGB",
    "weight_statistics_online_xgboost":           "Online XGB+Wt.",
    "weight_statistics_online_xgboost_with_perf": "Online XGB+Wt.+P",
    "xgboost_perf_only":                          "Offline XGB",
    "weight_statistics_xgboost":                  "Offline XGB+Wt.",
    "weight_statistics_xgboost_with_perf":        "Offline XGB+Wt.+P",
    "mlp_perf_only":                              "Offline MLP",
    "latent_averaged_representations_xgboost":    "AvgLat XGB",
    "latent_averaged_representations_xgboost_with_perf":
                                                    "AvgLat XGB+P",
    "latent_averaged_representations_mlp":        "AvgLat MLP",
    "latent_averaged_representations_mlp_with_perf":
                                                    "AvgLat MLP+P",
    "latent_window_representations_lstm":         "WinLat LSTM",
    "latent_window_representations_lstm_with_perf":
                                                    "WinLat LSTM+P",
    "latent_window_representations_online_lstm":  "Onl. WinLat LSTM",
    "latent_window_representations_online_lstm_with_perf":
                                                    "Onl. WinLat LSTM+P",
    "latent_tokens_representations_transformer":  "TokLat Transf.",
    "latent_tokens_representations_transformer_with_perf":
                                                    "TokLat Transf.+P",
    "latent_tokens_representations_online_transformer":
                                                    "Onl. TokLat Transf.",
    "latent_tokens_representations_online_transformer_with_perf":
                                                    "Onl. TokLat Transf.+P",
}

PREDEFINED_EXP_COLOR = {
    "lc_extrapolation_baseline":                  "#DD8452",
    "naive_baseline":                             "#4C72B0",
    "online_xgboost_perf_only":                   "#C44E52",
    "weight_statistics_online_xgboost_with_perf": "#8172B2",
    "xgboost_perf_only":                          "#55A868",
    "weight_statistics_xgboost_with_perf":        "#64B5CD",
    "mlp_perf_only":                              "#937860",
}

TARGET_MODES = ["final_value", "horizon_value"]
TM_LABEL     = {"final_value": "Final-Value Mode", "horizon_value": "Horizon-Value Mode"}

# Color palette for auto-generated colors
AUTO_COLORS = [
    "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
    "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf",
    "#aec7e8", "#ffbb78", "#98df8a", "#ff9896", "#c5b0d5",
    "#c49c94", "#f7b6d2", "#c7c7c7", "#dbbd22", "#9edae5",
]


def _generate_color(index: int) -> str:
    """Generate a color for an experiment, cycling through palette."""
    return AUTO_COLORS[index % len(AUTO_COLORS)]


def load_runs() -> tuple[list[dict], set[str]]:
    """Load all runs from batch outputs. Returns (rows, all_experiments)."""
    rows = []
    all_experiments = set()

    for tm_dir in BATCH_ROOT.iterdir():
        if not tm_dir.is_dir():
            continue
        for batch in sorted(tm_dir.iterdir()):
            if not batch.is_dir():
                continue
            name = batch.name
            if not (name.startswith("lc_baselines_ds") or re.match(r"^ds\d+$", name)):
                continue
            for s_path in batch.rglob("simulation_summary.json"):
                m_path = s_path.parent / "run_manifest.json"
                if not m_path.exists():
                    continue
                try:
                    s = json.loads(s_path.read_text())
                    m = json.loads(m_path.read_text())
                except Exception:
                    continue
                exp = m.get("experiment_name")
                if exp is None:
                    continue
                all_experiments.add(exp)
                oracle   = s.get("oracle_best_final_target")
                survivor = s.get("best_survivor_final_target")
                recall   = s.get("oracle_top_k_recall")
                if oracle is None or survivor is None:
                    continue
                rows.append({
                    "ds":     m.get("dataset_id"),
                    "exp":    exp,
                    "tm":     m.get("target_mode"),
                    "se":     m.get("selection", {}).get("start_epoch"),
                    "sr":     m.get("selection", {}).get("survival_rate"),
                    "gap":    oracle - survivor,
                    "recall": recall,
                })
    print(f"Loaded {len(rows)} runs from {len(all_experiments)} unique experiments.")
    return rows, all_experiments


def _abbreviate_label(name: str) -> str:
    """Generate short abbreviation from long experiment name."""
    # Remove common suffixes
    name = re.sub(r'_(with_perf|perf_only|online|offline)$', '', name)

    # Split by underscore and take first 2-3 chars of each word
    parts = name.split('_')
    abbrev = '.'.join(p[:3] for p in parts if p)  # First 3 chars of each part

    # Limit total length
    if len(abbrev) > 30:
        abbrev = '.'.join(p[0] for p in parts if p)  # Fall back to first letter only

    return abbrev


def build_exp_dicts(all_experiments: set[str]) -> tuple[dict, dict]:
    """Build label and color dicts for all experiments, using predefined or generated values."""
    exp_label = {}
    exp_color = {}
    sorted_exps = sorted(all_experiments)

    for idx, exp in enumerate(sorted_exps):
        # Use predefined if available, else auto-generate abbreviation
        if exp in PREDEFINED_EXP_LABEL:
            exp_label[exp] = PREDEFINED_EXP_LABEL[exp]
        else:
            exp_label[exp] = _abbreviate_label(exp)
        exp_color[exp] = PREDEFINED_EXP_COLOR.get(exp, _generate_color(idx))

    return exp_label, exp_color


# ---------------------------------------------------------------------------
# Per-dataset box plot
# ---------------------------------------------------------------------------

def _sorted_exps(ds_rows: list[dict], tm: str, all_experiments: set[str]) -> list[str]:
    """Sort experiments by median gap ascending (best = lowest gap first)."""
    tm_rows = [r for r in ds_rows if r["tm"] == tm]
    vals: dict[str, list[float]] = defaultdict(list)
    for r in tm_rows:
        vals[r["exp"]].append(r["gap"])
    present = [e for e in sorted(all_experiments) if e in vals]
    return sorted(present, key=lambda e: np.median(vals[e]))


def _draw_boxplot(ax, ds_rows: list[dict], metric: str, tm: str,
                  exps_ordered: list[str], exp_label: dict, exp_color: dict) -> None:
    """Horizontal box plot for one (metric, target_mode) panel."""
    tm_rows = [r for r in ds_rows if r["tm"] == tm]

    data, colors, positions = [], [], []
    for pos, exp in enumerate(exps_ordered):
        vals = [r[metric] for r in tm_rows if r["exp"] == exp and r[metric] is not None]
        if not vals:
            continue
        data.append(vals)
        colors.append(exp_color[exp])
        positions.append(pos)

    if not data:
        ax.text(0.5, 0.5, "no data", ha="center", va="center",
                transform=ax.transAxes, color="grey")
        return

    bp = ax.boxplot(
        data,
        positions=positions,
        vert=False,
        widths=0.55,
        patch_artist=True,
        medianprops=dict(color="black", linewidth=2),
        whiskerprops=dict(linewidth=1.2),
        capprops=dict(linewidth=1.2),
        flierprops=dict(marker="o", markersize=3, linestyle="none",
                        markerfacecolor="grey", alpha=0.5),
    )
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.8)

    ax.set_yticks(range(len(exps_ordered)))
    ax.set_yticklabels([exp_label[e] for e in exps_ordered], fontsize=12)
    ax.xaxis.grid(True, linestyle="--", alpha=0.5)
    ax.set_axisbelow(True)

    if metric == "gap":
        ax.set_xlim(left=0)
    else:
        ax.set_xlim(0, 1.05)


def plot_dataset(ds_id: int, rows: list[dict], all_experiments: set[str],
                exp_label: dict, exp_color: dict) -> None:
    ds_rows = [r for r in rows if r["ds"] == ds_id]
    if not ds_rows:
        print(f"  ds{ds_id}: no data, skipping.")
        return

    # Sort order fixed by final_value gap; reuse for horizon_value so rows align
    exps_fv = _sorted_exps(ds_rows, "final_value", all_experiments) or _sorted_exps(ds_rows, "horizon_value", all_experiments)
    exps_hv = _sorted_exps(ds_rows, "horizon_value", all_experiments) or exps_fv

    fig, axes = plt.subplots(2, 2, figsize=(18, 11), sharey=False)
    for col, (tm, exps_ordered) in enumerate(
        [("final_value", exps_fv), ("horizon_value", exps_hv)]
    ):
        for row, (metric, xlabel) in enumerate(
            [("gap", "Best-Survivor Gap (↓ better)"),
             ("recall", "Oracle Top-K Recall (↑ better)")]
        ):
            ax = axes[row][col]
            _draw_boxplot(ax, ds_rows, metric, tm, exps_ordered, exp_label, exp_color)

            ax.set_xlabel(xlabel, fontsize=13)
            if row == 0:
                ax.set_title(TM_LABEL[tm], fontsize=14)

    fig.subplots_adjust(left=0.38, right=0.98, wspace=0.55, top=0.93)

    # Center title relative to the plot area
    left, right = 0.38, 0.98
    center_x = (left + right) / 2
    fig.suptitle(
        f"SOTF Results — {DATASET_DISPLAY.get(ds_id, f'ds{ds_id}')} (All Predictors)",
        fontsize=17,
        x=center_x,
        ha="center",
    )

    stem = f"ds_boxplots_all_{ds_id}"
    print(f"  ds{ds_id}: ", end="")
    for ext in (".pdf", ".png"):
        path = FIG_DIR / (stem + ext)
        fig.savefig(path, bbox_inches="tight", dpi=200)
        print(f"{path.name} ", end="")
    print()
    plt.close(fig)


# ---------------------------------------------------------------------------

if __name__ == "__main__":
    rows, all_experiments = load_runs()
    print(f"\nDiscovered experiments: {sorted(all_experiments)}\n")

    exp_label, exp_color = build_exp_dicts(all_experiments)

    print("=== Per-dataset box plots (all predictors) ===")
    for ds_id in DATASET_ORDER:
        plot_dataset(ds_id, rows, all_experiments, exp_label, exp_color)
    print("\nDone.")
