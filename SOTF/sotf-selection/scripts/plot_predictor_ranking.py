"""
Two clean comparison figures:

1. lc_ranking_bar.pdf
   Horizontal bar chart: mean gap per predictor, sorted best→worst.
   Error bars = IQR (25th–75th percentile).
   Two panels: Final-Value (left) and Horizon-Value (right).

2. lc_winrate_matrix.pdf
   N×N win-rate matrix: cell[i,j] = % of (dataset, start_epoch, SR) conditions
   where predictor i beats predictor j (lower gap = better).
   Two panels: Final-Value and Horizon-Value.
"""

import json
import re
from pathlib import Path
from collections import defaultdict
from itertools import combinations

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

matplotlib.use("Agg")

plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["font.size"] = 14
plt.rcParams["axes.titlesize"] = 15
plt.rcParams["axes.labelsize"] = 13
plt.rcParams["xtick.labelsize"] = 12
plt.rcParams["ytick.labelsize"] = 12
plt.rcParams["legend.fontsize"] = 12

# ---------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parents[2]
BATCH_ROOT = REPO_ROOT / "sotf-selection" / "outputs" / "batch_experiments"
FIG_DIR = (
    REPO_ROOT / "sotf-selection" / "outputs"
    / "evaluation" / "predictor_ranking"
)
FIG_DIR.mkdir(parents=True, exist_ok=True)

DATASET_ORDER = [1, 2, 3, 5, 7, 8, 9, 10, 14]
SURVIVAL_RATES = [0.1, 0.25, 0.5, 0.75, 0.9]
TARGET_MODES = ["final_value", "horizon_value"]
TM_LABEL = {"final_value": "Final-Value Mode", "horizon_value": "Horizon-Value Mode"}

ALL_EXPERIMENTS = [
    "lc_extrapolation_baseline",
    "naive_baseline",
    "online_xgboost_perf_only",
    "weight_statistics_online_xgboost_with_perf",
    "xgboost_perf_only",
    "weight_statistics_xgboost_with_perf",
    "mlp_perf_only",
]

EXP_LABEL = {
    "lc_extrapolation_baseline":                  "LC Extrap. (baseline)",
    "naive_baseline":                             "Naive",
    "online_xgboost_perf_only":                   "Online XGB (perf only)",
    "weight_statistics_online_xgboost_with_perf": "Online XGB + Wt.Stats",
    "xgboost_perf_only":                          "Offline XGB (perf only)",
    "weight_statistics_xgboost_with_perf":        "Offline XGB + Wt.Stats",
    "mlp_perf_only":                              "Offline MLP (perf only)",
}

# Colour by predictor family
EXP_COLOR = {
    "lc_extrapolation_baseline":                  "#DD8452",
    "naive_baseline":                             "#4C72B0",
    "online_xgboost_perf_only":                   "#C44E52",
    "weight_statistics_online_xgboost_with_perf": "#8172B2",
    "xgboost_perf_only":                          "#55A868",
    "weight_statistics_xgboost_with_perf":        "#64B5CD",
    "mlp_perf_only":                              "#937860",
}


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_runs() -> list[dict]:
    rows = []
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
                if exp not in ALL_EXPERIMENTS:
                    continue
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
    print(f"Loaded {len(rows)} runs.")
    return rows


# ---------------------------------------------------------------------------
# Plot 1: 2×2 box plots  (rows=metrics, cols=modes), sorted by median gap
# ---------------------------------------------------------------------------

METRICS = [
    ("gap",    "Best-Survivor Gap (↓ better)"),
    ("recall", "Oracle Top-K Recall (↑ better)"),
]


def _sort_order_for_metric(metric: str, rows: list[dict], tm: str) -> list[str]:
    """Return experiments sorted best→worst by median for this metric/mode."""
    tm_rows = [r for r in rows if r["tm"] == tm]
    vals: dict[str, list[float]] = defaultdict(list)
    for r in tm_rows:
        if r[metric] is not None:
            vals[r["exp"]].append(r[metric])
    # gap: lower is better → sort ascending
    # recall: higher is better → sort descending
    reverse = (metric == "recall")
    return sorted(vals, key=lambda e: np.median(vals[e]), reverse=reverse)


def plot_ranking_bars(rows: list[dict]) -> None:
    fig, axes = plt.subplots(
        2, 2, figsize=(14, 10), sharey=False,
    )
    fig.suptitle("Predictor Comparison across All Datasets, Start Epochs and Survival Rates", fontsize=14)

    for col, tm in enumerate(TARGET_MODES):
        tm_rows = [r for r in rows if r["tm"] == tm]

        # Use gap sort order for both rows so predictors appear in the same vertical order
        exps_sorted = _sort_order_for_metric("gap", rows, tm)

        for row, (metric, ylabel) in enumerate(METRICS):
            ax = axes[row][col]

            data, colors, positions = [], [], []
            for pos, exp in enumerate(exps_sorted):
                vals = [r[metric] for r in tm_rows
                        if r["exp"] == exp and r[metric] is not None]
                if not vals:
                    continue
                data.append(vals)
                colors.append(EXP_COLOR.get(exp, "#888888"))
                positions.append(pos)

            if not data:
                ax.set_title(TM_LABEL[tm])
                continue

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

            ax.set_yticks(range(len(exps_sorted)))
            ax.set_yticklabels(
                [EXP_LABEL.get(e, e) for e in exps_sorted],
                fontsize=11,
            )
            ax.set_xlabel(ylabel, fontsize=11)
            ax.xaxis.grid(True, linestyle="--", alpha=0.5)
            ax.set_axisbelow(True)

            if metric == "gap":
                ax.set_xlim(left=0)
            else:
                ax.set_xlim(0, 1.05)

            # Column title only on top row
            if row == 0:
                ax.set_title(TM_LABEL[tm], fontsize=13)

    fig.tight_layout()
    _save(fig, "lc_ranking_bar")


# ---------------------------------------------------------------------------
# Plot 2: win-rate matrix
# ---------------------------------------------------------------------------

def plot_winrate_matrix(rows: list[dict]) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    fig.suptitle(
        "Pairwise Win Rate: row beats column (% of conditions, lower gap = win)",
        fontsize=14,
    )

    for ax, tm in zip(axes, TARGET_MODES):
        tm_rows = [r for r in rows if r["tm"] == tm]

        # Build index: (ds, se, sr, exp) -> mean gap
        idx: dict[tuple, float] = {}
        acc: dict[tuple, list] = defaultdict(list)
        for r in tm_rows:
            acc[(r["ds"], r["se"], r["sr"], r["exp"])].append(r["gap"])
        for key, vals in acc.items():
            idx[key] = float(np.mean(vals))

        # All conditions (ds, se, sr) where ≥2 experiments have data
        conditions = set()
        for (ds, se, sr, exp) in idx:
            conditions.add((ds, se, sr))

        # Experiments present in this mode
        exps_in_mode = sorted(
            {exp for (_, _, _, exp) in idx},
            key=lambda e: np.mean([idx[(ds, se, sr, e)]
                                   for ds, se, sr in conditions
                                   if (ds, se, sr, e) in idx]),
        )
        n = len(exps_in_mode)
        win_matrix = np.full((n, n), np.nan)

        for i, exp_i in enumerate(exps_in_mode):
            for j, exp_j in enumerate(exps_in_mode):
                if i == j:
                    win_matrix[i, j] = np.nan
                    continue
                wins = 0
                total = 0
                for cond in conditions:
                    ds, se, sr = cond
                    gap_i = idx.get((ds, se, sr, exp_i))
                    gap_j = idx.get((ds, se, sr, exp_j))
                    if gap_i is not None and gap_j is not None:
                        total += 1
                        if gap_i < gap_j:
                            wins += 1
                win_matrix[i, j] = (wins / total * 100) if total > 0 else np.nan

        labels = [EXP_LABEL.get(e, e) for e in exps_in_mode]

        im = ax.imshow(win_matrix, cmap="RdYlGn", vmin=0, vmax=100, aspect="auto")

        ax.set_xticks(range(n))
        ax.set_xticklabels(labels, rotation=35, ha="right", fontsize=10)
        ax.set_yticks(range(n))
        ax.set_yticklabels(labels, fontsize=10)
        ax.set_title(TM_LABEL[tm], fontsize=14)
        ax.set_xlabel("Column predictor (loser)", fontsize=11)
        ax.set_ylabel("Row predictor (winner)", fontsize=11)

        # Annotate cells
        for i in range(n):
            for j in range(n):
                val = win_matrix[i, j]
                if not np.isnan(val):
                    text_color = "black"
                    ax.text(j, i, f"{val:.0f}%",
                            ha="center", va="center",
                            fontsize=9, color=text_color, fontweight="bold")
                else:
                    ax.text(j, i, "—",
                            ha="center", va="center", fontsize=10, color="grey")

        # Diagonal styling
        for k in range(n):
            ax.add_patch(plt.Rectangle(
                (k - 0.5, k - 0.5), 1, 1,
                fill=True, color="#dddddd", zorder=2,
            ))
            ax.text(k, k, "—", ha="center", va="center",
                    fontsize=10, color="grey", zorder=3)

        cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label("Win rate (%)", fontsize=11)
        cbar.set_ticks([0, 25, 50, 75, 100])

    fig.tight_layout()
    _save(fig, "lc_winrate_matrix")


# ---------------------------------------------------------------------------

def _save(fig, stem: str) -> None:
    for ext in (".pdf", ".png"):
        path = FIG_DIR / (stem + ext)
        fig.savefig(path, bbox_inches="tight", dpi=200)
        print(f"  Saved: {path}")
    plt.close(fig)


# ---------------------------------------------------------------------------

if __name__ == "__main__":
    rows = load_runs()

    print("\n=== Plot 1: Ranked bar chart ===")
    plot_ranking_bars(rows)

    print("\n=== Plot 2: Win-rate matrix ===")
    plot_winrate_matrix(rows)

    print("\nDone.")
