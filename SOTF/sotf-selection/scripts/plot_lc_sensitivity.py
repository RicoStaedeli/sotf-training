"""
Sensitivity plots comparing predictors against the best LC baseline (LC Extrapolation).

Option A: Per-dataset line plots  (lc_sensitivity_ds{N}.pdf)
  Rows    = target mode (Final-Value / Horizon-Value)
  Columns = start epoch
  X       = survival rate, Y = best-survivor gap
  Lines   = one per predictor (only predictors that have data for that cell)

Option B: Delta heatmap vs LC Extrap.  (lc_delta_vs_lc_extrap.pdf)
  Horizon-value only — full overlap between LC Extrap. and all other predictors.
  One panel per comparison predictor.
  Rows    = dataset × start epoch
  Columns = survival rate
  Color   = predictor_gap - lc_extrap_gap  (green = predictor beats LC Extrap.)
"""

import json
import re
from pathlib import Path
from collections import defaultdict

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import TwoSlopeNorm

matplotlib.use("Agg")

plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["font.size"] = 13
plt.rcParams["axes.titlesize"] = 13
plt.rcParams["axes.labelsize"] = 12
plt.rcParams["xtick.labelsize"] = 11
plt.rcParams["ytick.labelsize"] = 11
plt.rcParams["legend.fontsize"] = 10

# ---------------------------------------------------------------------------
# Paths / constants
# ---------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parents[2]
BATCH_ROOT = REPO_ROOT / "sotf-selection" / "outputs" / "batch_experiments"
FIG_DIR = (
    REPO_ROOT / "sotf-selection" / "outputs"
    / "evaluation" / "lc_sensitivity"
)
FIG_DIR.mkdir(parents=True, exist_ok=True)

DATASET_DISPLAY = {
    1: "CIFAR-10 CNN Uniform",       2: "CIFAR-10 CNN Hyp. Rand.",
    3: "CIFAR-10 CNN Hyp. Fixed",    5: "CIFAR-10 ResNet-18 Width",
    7: "CIFAR-10 ViT",               8: "STL-10 CNN Uniform",
    9: "STL-10 CNN Hyp. Rand.",     10: "STL-10 CNN Hyp. Fixed",
   14: "TinyImageNet Phase",
}
DATASET_ORDER = [1, 2, 3, 8, 9, 10, 14, 5, 7]
SURVIVAL_RATES = [0.1, 0.25, 0.5, 0.75, 0.9]
SR_LABELS = ["0.10", "0.25", "0.50", "0.75", "0.90"]

# Best LC baseline chosen from overall performance analysis
LC_BASELINE = "lc_extrapolation_baseline"

# All experiments to load
ALL_EXPERIMENTS = {
    "lc_extrapolation_baseline",
    "naive_baseline",
    "online_xgboost_perf_only",
    "weight_statistics_online_xgboost_with_perf",
    "xgboost_perf_only",
    "weight_statistics_xgboost_with_perf",
    "mlp_perf_only",
}

# Display names and colours
EXP_LABEL = {
    "lc_extrapolation_baseline":            "LC Extrap. (baseline)",
    "naive_baseline":                       "Naive",
    "online_xgboost_perf_only":             "Online XGB (perf only)",
    "weight_statistics_online_xgboost_with_perf": "Online XGB + Wt.Stats",
    "xgboost_perf_only":                    "Offline XGB (perf only)",
    "weight_statistics_xgboost_with_perf":  "Offline XGB + Wt.Stats",
    "mlp_perf_only":                        "Offline MLP (perf only)",
}
EXP_COLOR = {
    "lc_extrapolation_baseline":            "#DD8452",   # orange — reference
    "naive_baseline":                       "#4C72B0",   # blue
    "online_xgboost_perf_only":             "#C44E52",   # red
    "weight_statistics_online_xgboost_with_perf": "#8172B2",  # purple
    "xgboost_perf_only":                    "#55A868",   # green
    "weight_statistics_xgboost_with_perf":  "#64B5CD",   # teal
    "mlp_perf_only":                        "#937860",   # brown
}
# LC Extrap. drawn dashed so it's clearly the reference line
EXP_LINESTYLE = {exp: "-" for exp in ALL_EXPERIMENTS}
EXP_LINESTYLE["lc_extrapolation_baseline"] = "--"
EXP_LINEWIDTH = {exp: 1.5 for exp in ALL_EXPERIMENTS}
EXP_LINEWIDTH["lc_extrapolation_baseline"] = 2.2

TM_LABEL = {"final_value": "Final-Value Mode", "horizon_value": "Horizon-Value Mode"}

# Comparison predictors for the delta heatmap (Option B)
COMPARE_EXPERIMENTS = [
    "naive_baseline",
    "online_xgboost_perf_only",
    "weight_statistics_online_xgboost_with_perf",
    "xgboost_perf_only",
    "weight_statistics_xgboost_with_perf",
    "mlp_perf_only",
]


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_all_runs() -> list[dict]:
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
                oracle = s.get("oracle_best_final_target")
                survivor = s.get("best_survivor_final_target")
                if oracle is None or survivor is None:
                    continue
                rows.append({
                    "ds":  m.get("dataset_id"),
                    "exp": exp,
                    "tm":  m.get("target_mode"),
                    "se":  m.get("selection", {}).get("start_epoch"),
                    "sr":  m.get("selection", {}).get("survival_rate"),
                    "gap": oracle - survivor,
                })
    print(f"Loaded {len(rows)} runs.")
    return rows


def build_index(rows: list[dict]) -> dict:
    """Mean gap per (ds, tm, exp, se, sr)."""
    acc = defaultdict(list)
    for r in rows:
        acc[(r["ds"], r["tm"], r["exp"], r["se"], r["sr"])].append(r["gap"])
    return {k: float(np.mean(v)) for k, v in acc.items()}


def _save(fig, stem: str) -> None:
    for ext in (".pdf", ".png"):
        path = FIG_DIR / (stem + ext)
        fig.savefig(path, bbox_inches="tight", dpi=200)
        print(f"  Saved: {path}")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Option A: per-dataset line plots
# ---------------------------------------------------------------------------

def plot_dataset_sensitivity(ds_id: int, rows: list[dict]) -> None:
    """
    Per-dataset figure.
    Rows = target modes (2).  Columns = sorted union of all start epochs for this dataset.
    In each panel: x=survival_rate, y=gap, one line per predictor (only those with data).
    LC Extrap. drawn as dashed reference in every panel where it has data.
    """
    ds_rows = [r for r in rows if r["ds"] == ds_id]
    if not ds_rows:
        return

    target_modes = ["final_value", "horizon_value"]

    # Collect all start epochs per target_mode for this dataset
    se_by_tm: dict[str, list[int]] = {}
    for tm in target_modes:
        ses = sorted({r["se"] for r in ds_rows if r["tm"] == tm and r["se"] is not None})
        se_by_tm[tm] = ses

    n_cols = max(len(v) for v in se_by_tm.values()) if se_by_tm else 1
    n_rows = len(target_modes)

    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(4 * n_cols, 3.8 * n_rows),
        sharey=False, sharex=True,
        squeeze=False,
    )
    fig.suptitle(
        f"SOTF Sensitivity — {DATASET_DISPLAY.get(ds_id, f'ds{ds_id}')}",
        fontsize=14, y=1.01,
    )

    plotted_exps: set[str] = set()

    for row_idx, tm in enumerate(target_modes):
        ses = se_by_tm.get(tm, [])
        for col_idx in range(n_cols):
            ax = axes[row_idx][col_idx]

            if col_idx >= len(ses):
                ax.set_visible(False)
                continue

            se = ses[col_idx]
            se_rows = [r for r in ds_rows if r["tm"] == tm and r["se"] == se]
            exps_present = sorted({r["exp"] for r in se_rows})

            # Draw LC Extrap. first (always dashed reference if present)
            exp_order = (
                [LC_BASELINE] + [e for e in exps_present if e != LC_BASELINE]
            )

            for exp in exp_order:
                if exp not in exps_present:
                    continue
                gaps = []
                for sr in SURVIVAL_RATES:
                    vals = [r["gap"] for r in se_rows if r["exp"] == exp
                            and abs(r["sr"] - sr) < 1e-6]
                    gaps.append(np.mean(vals) if vals else np.nan)

                ax.plot(
                    SURVIVAL_RATES, gaps,
                    color=EXP_COLOR.get(exp, "#888888"),
                    linestyle=EXP_LINESTYLE.get(exp, "-"),
                    linewidth=EXP_LINEWIDTH.get(exp, 1.5),
                    marker="o", markersize=4,
                    label=EXP_LABEL.get(exp, exp),
                )
                plotted_exps.add(exp)

            ax.set_ylim(bottom=0)
            ax.set_xticks(SURVIVAL_RATES)
            ax.set_xticklabels(SR_LABELS, fontsize=10)
            ax.yaxis.grid(True, linestyle="--", alpha=0.45)
            ax.set_axisbelow(True)

            # Titles and axis labels
            if row_idx == 0:
                ax.set_title(f"se = {se}", fontsize=12)
            if col_idx == 0:
                ax.set_ylabel(f"{TM_LABEL[tm]}\n\nGap (↓ better)", fontsize=11)
            if row_idx == n_rows - 1:
                ax.set_xlabel("Survival Rate", fontsize=11)

    # Shared legend below — only for experiments that actually appear
    legend_exps = [LC_BASELINE] + [e for e in COMPARE_EXPERIMENTS if e in plotted_exps]
    handles = [
        plt.Line2D(
            [0], [0],
            color=EXP_COLOR[e],
            linestyle=EXP_LINESTYLE[e],
            linewidth=EXP_LINEWIDTH[e],
            marker="o", markersize=4,
            label=EXP_LABEL[e],
        )
        for e in legend_exps if e in EXP_COLOR
    ]
    if handles:
        fig.legend(
            handles=handles,
            loc="lower center",
            ncol=min(len(handles), 4),
            bbox_to_anchor=(0.5, -0.06),
            frameon=True,
            fontsize=10,
        )

    fig.tight_layout()
    fig.subplots_adjust(bottom=0.18)
    stem = f"lc_sensitivity_ds{ds_id}"
    print(f"ds{ds_id}:")
    _save(fig, stem)


def plot_all_datasets(rows: list[dict]) -> None:
    for ds_id in DATASET_ORDER:
        plot_dataset_sensitivity(ds_id, rows)


# ---------------------------------------------------------------------------
# Option B: delta heatmap vs LC Extrap. (horizon-value only)
# ---------------------------------------------------------------------------

def plot_delta_heatmap(rows: list[dict]) -> None:
    """
    One panel per comparison predictor.
    Rows = dataset × start_epoch (horizon_value, shared with LC Extrap.).
    Cols = survival rate.
    Color = predictor_gap - lc_extrap_gap  (negative/green = predictor beats LC Extrap.)
    """
    tm = "horizon_value"
    tm_rows = [r for r in rows if r["tm"] == tm]
    idx = build_index(tm_rows)

    # Collect (ds, se) pairs where LC Extrap. has data
    lc_pairs: list[tuple[int, int]] = []
    for ds_id in DATASET_ORDER:
        lc_ses = sorted({
            r["se"] for r in tm_rows
            if r["ds"] == ds_id and r["exp"] == LC_BASELINE and r["se"] is not None
        })
        for se in lc_ses:
            lc_pairs.append((ds_id, se))

    n_pairs = len(lc_pairs)
    n_sr = len(SURVIVAL_RATES)
    n_comp = len(COMPARE_EXPERIMENTS)

    # Build delta matrices
    matrices: dict[str, np.ndarray] = {}
    abs_max = 0.0
    for exp in COMPARE_EXPERIMENTS:
        mat = np.full((n_pairs, n_sr), np.nan)
        for ri, (ds_id, se) in enumerate(lc_pairs):
            for ci, sr in enumerate(SURVIVAL_RATES):
                lc_gap  = idx.get((ds_id, tm, LC_BASELINE, se, sr))
                exp_gap = idx.get((ds_id, tm, exp, se, sr))
                if lc_gap is not None and exp_gap is not None:
                    mat[ri, ci] = exp_gap - lc_gap
        matrices[exp] = mat
        local = np.nanmax(np.abs(mat))
        if not np.isnan(local):
            abs_max = max(abs_max, local)

    if abs_max == 0:
        abs_max = 0.01
    norm = TwoSlopeNorm(vmin=-abs_max, vcenter=0, vmax=abs_max)

    row_labels = [f"ds{ds}  se={se}" for ds, se in lc_pairs]

    # Separator positions (between datasets)
    separators: list[float] = []
    prev_ds = None
    for ri, (ds_id, _) in enumerate(lc_pairs):
        if prev_ds is not None and ds_id != prev_ds:
            separators.append(ri - 0.5)
        prev_ds = ds_id

    fig_h = max(5, 0.42 * n_pairs + 2.5)
    fig, axes = plt.subplots(
        1, n_comp,
        figsize=(3.2 * n_comp, fig_h),
        sharey=True,
    )

    im = None
    for col_idx, exp in enumerate(COMPARE_EXPERIMENTS):
        ax = axes[col_idx]
        mat = matrices[exp]
        im = ax.imshow(mat, cmap="RdYlGn_r", norm=norm, aspect="auto")

        ax.set_xticks(range(n_sr))
        ax.set_xticklabels(SR_LABELS, fontsize=9, rotation=45, ha="right")
        ax.set_xlabel("Surv. Rate", fontsize=10)
        ax.set_title(EXP_LABEL[exp], fontsize=11, wrap=True)

        for sep in separators:
            ax.axhline(sep, color="black", linewidth=1.0)

        # Cell annotations
        for ri in range(n_pairs):
            for ci in range(n_sr):
                val = mat[ri, ci]
                if not np.isnan(val):
                    ax.text(ci, ri, f"{val:+.3f}", ha="center", va="center",
                            fontsize=7, color="black")

    axes[0].set_yticks(range(n_pairs))
    axes[0].set_yticklabels(row_labels, fontsize=9)

    if im is not None:
        cbar = fig.colorbar(im, ax=axes, fraction=0.015, pad=0.02)
        cbar.set_label(
            "Δ Gap = Predictor − LC Extrap.\n"
            "Negative (green) = predictor beats LC Extrap.",
            fontsize=10,
        )

    fig.suptitle(
        "Predictors vs. Best LC Baseline (LC Extrap.) — Gap Delta\n"
        "Horizon-Value Mode  ·  Negative = predictor beats LC Extrap.",
        fontsize=12, y=1.01,
    )
    fig.tight_layout()
    stem = "lc_delta_vs_lc_extrap"
    print("Delta heatmap vs LC Extrap.:")
    _save(fig, stem)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    rows = load_all_runs()

    print("\n=== Option A: per-dataset sensitivity ===")
    plot_all_datasets(rows)

    print("\n=== Option B: delta heatmap vs LC Extrap. ===")
    plot_delta_heatmap(rows)

    print("\nDone.")
