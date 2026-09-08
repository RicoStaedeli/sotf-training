"""Compute and report mean best-survivor gap reductions relative to all baselines.

Aggregates completed SOTF simulation outputs over all nine model zoos, all
start-epoch/survival-rate configurations, and the fixed decision interval of
five epochs.  For each target mode the script finds the best (lowest mean gap)
forecast-based predictor and reports the absolute and percentage reduction
relative to three reference baselines:

  - Naive (current observed accuracy)
  - LCNet
  - LC Extrapolation

Run from the repo root:
    uv run python sotf-selection/scripts/compute_gap_reductions.py
    uv run python sotf-selection/scripts/compute_gap_reductions.py --offline-only

The offline-only mode retains the three baselines for comparison but excludes
online predictors from the best-predictor search and detailed tables. Output is
printed to stdout and written to a JSON file.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from statistics import mean

REPO_ROOT = Path(__file__).resolve().parents[2]
BATCH_ROOT = REPO_ROOT / "sotf-selection" / "outputs" / "batch_experiments"

TARGET_MODES = ["final_value", "horizon_value"]

DATASET_ORDER = [1, 2, 3, 8, 9, 10, 14, 5, 7]
DATASET_DISPLAY = {
    1: "CIFAR-10 CNN Uniform",
    2: "CIFAR-10 CNN Hyp. Rand.",
    3: "CIFAR-10 CNN Hyp. Fixed",
    5: "CIFAR-10 ResNet-18 Width",
    7: "CIFAR-10 ViT",
    8: "STL-10 CNN Uniform",
    9: "STL-10 CNN Hyp. Rand.",
    10: "STL-10 CNN Hyp. Fixed",
    14: "TinyImageNet ResNet-18 Phase",
}

BASELINES = [
    "naive_baseline",
    "lcnet_baseline",
    "lc_extrapolation_baseline",
]

BASELINE_LABELS = {
    "naive_baseline": "Naive",
    "lcnet_baseline": "LCNet",
    "lc_extrapolation_baseline": "LC Extrapolation",
}

# All experiments tracked — order matches the thesis tables
ALL_EXPERIMENTS = [
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

OFFLINE_PREDICTORS = [
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
    "latent_tokens_representations_transformer",
    "latent_tokens_representations_transformer_with_perf",
]

ALL_PREDICTORS = [
    experiment for experiment in ALL_EXPERIMENTS if experiment not in BASELINES
]

EXPERIMENT_LABELS = {
    "lc_extrapolation_baseline": "LC Extrapolation (baseline)",
    "lcnet_baseline": "LCNet (baseline)",
    "naive_baseline": "Naive (baseline)",
    "online_xgboost_perf_only": "Online XGB (perf only)",
    "weight_statistics_online_xgboost": "Online XGB + Wt.Stats",
    "weight_statistics_online_xgboost_with_perf": "Online XGB + Wt.Stats + Perf",
    "xgboost_perf_only": "Offline XGB (perf only)",
    "weight_statistics_xgboost": "Offline XGB + Wt.Stats",
    "weight_statistics_xgboost_with_perf": "Offline XGB + Wt.Stats + Perf",
    "mlp_perf_only": "Offline MLP (perf only)",
    "latent_averaged_representations_xgboost": "Avg-Latent XGBoost",
    "latent_averaged_representations_xgboost_with_perf": "Avg-Latent XGBoost + Perf",
    "latent_averaged_representations_mlp": "Avg-Latent MLP",
    "latent_averaged_representations_mlp_with_perf": "Avg-Latent MLP + Perf",
    "latent_window_representations_lstm": "Latent-Window LSTM",
    "latent_window_representations_lstm_with_perf": "Latent-Window LSTM + Perf",
    "latent_window_representations_online_lstm": "Online Latent-Window LSTM",
    "latent_window_representations_online_lstm_with_perf": "Online Latent-Window LSTM + Perf",
    "latent_tokens_representations_transformer": "Latent-Token Transformer",
    "latent_tokens_representations_transformer_with_perf": "Latent-Token Transformer + Perf",
    "latent_tokens_representations_online_transformer": "Online Latent-Token Transformer",
    "latent_tokens_representations_online_transformer_with_perf": "Online Latent-Token Transformer + Perf",
}


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Compute aggregate best-survivor gap reductions.",
    )
    parser.add_argument(
        "--offline-only",
        action="store_true",
        help=(
            "consider only offline predictors while retaining the naive, LCNet, "
            "and LC-extrapolation baselines"
        ),
    )
    return parser.parse_args()


def load_runs(included_experiments: set[str] | None = None) -> list[dict]:
    """Walk batch output tree and return one record per completed simulation."""
    if included_experiments is None:
        included_experiments = set(ALL_EXPERIMENTS)

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
        if experiment not in included_experiments:
            continue

        oracle = summary.get("oracle_best_final_target")
        survivor = summary.get("best_survivor_final_target")
        if oracle is None or survivor is None:
            continue

        rows.append({
            "dataset_id": manifest.get("dataset_id"),
            "target_mode": manifest.get("target_mode"),
            "experiment": experiment,
            "gap": float(oracle) - float(survivor),
        })
    return rows


def compute_mean_gaps(rows: list[dict]) -> dict[tuple[str, str], float]:
    """Return mean gap per (target_mode, experiment) across all loaded runs."""
    acc: dict[tuple[str, str], list[float]] = defaultdict(list)
    for row in rows:
        acc[(row["target_mode"], row["experiment"])].append(row["gap"])
    return {key: mean(vals) for key, vals in acc.items()}


def compute_mean_gaps_per_zoo(rows: list[dict]) -> dict[tuple[str, int, str], float]:
    """Return mean gap per (target_mode, dataset_id, experiment)."""
    acc: dict[tuple[str, int, str], list[float]] = defaultdict(list)
    for row in rows:
        ds = row["dataset_id"]
        if ds is None:
            continue
        acc[(row["target_mode"], int(ds), row["experiment"])].append(row["gap"])
    return {key: mean(vals) for key, vals in acc.items()}


def pct_reduction(reference: float, improved: float) -> float:
    """Percentage by which `improved` is lower than `reference`."""
    return (reference - improved) / reference * 100.0


def find_best_predictor_for_zoo(
    zoo_gaps: dict[tuple[str, int, str], float],
    target_mode: str,
    dataset_id: int,
    predictor_experiments: set[str] | None = None,
) -> tuple[str, float] | tuple[None, None]:
    """Return (experiment, mean_gap) for the best non-baseline predictor on one zoo."""
    if predictor_experiments is None:
        predictor_experiments = set(ALL_PREDICTORS)

    candidates = {
        exp: gap
        for (tm, ds, exp), gap in zoo_gaps.items()
        if tm == target_mode and ds == dataset_id and exp in predictor_experiments
    }
    if not candidates:
        return None, None
    best_exp = min(candidates, key=candidates.__getitem__)
    return best_exp, candidates[best_exp]


def find_best_predictor(
    mean_gaps: dict[tuple[str, str], float],
    target_mode: str,
    predictor_experiments: set[str] | None = None,
) -> tuple[str, float]:
    """Return (experiment_name, mean_gap) for the lowest-gap non-baseline predictor."""
    if predictor_experiments is None:
        predictor_experiments = set(ALL_PREDICTORS)

    non_baseline = {
        exp: gap
        for (tm, exp), gap in mean_gaps.items()
        if tm == target_mode and exp in predictor_experiments
    }
    if not non_baseline:
        print(f"WARNING: no non-baseline predictor data found for {target_mode}")
        sys.exit(1)
    best_exp = min(non_baseline, key=non_baseline.__getitem__)
    return best_exp, non_baseline[best_exp]


def report(
    mean_gaps: dict[tuple[str, str], float],
    predictor_experiments: set[str] | None = None,
    experiment_order: list[str] | None = None,
) -> dict:
    """Print and return the reduction table."""
    if predictor_experiments is None:
        predictor_experiments = set(ALL_PREDICTORS)
    if experiment_order is None:
        experiment_order = ALL_EXPERIMENTS

    results = {}

    print("=" * 72)
    print("SOTF  Mean Best-Survivor Gap — Reduction vs Baselines")
    print("=" * 72)

    for tm in TARGET_MODES:
        label = "Final-Value" if tm == "final_value" else "Horizon-Value"
        print(f"\n{'─' * 72}")
        print(f"  Target mode: {label}")
        print(f"{'─' * 72}")

        # Baseline gaps
        print("\n  Baselines:")
        baseline_gaps: dict[str, float] = {}
        for exp in BASELINES:
            gap = mean_gaps.get((tm, exp))
            if gap is None:
                print(f"    {BASELINE_LABELS[exp]:30s}  NO DATA")
            else:
                baseline_gaps[exp] = gap
                print(f"    {BASELINE_LABELS[exp]:30s}  {gap:.4f}")

        # Best predictor
        best_exp, best_gap = find_best_predictor(
            mean_gaps,
            tm,
            predictor_experiments,
        )
        print(f"\n  Best predictor: {EXPERIMENT_LABELS[best_exp]}")
        print(f"    Mean gap: {best_gap:.4f}")

        # Reductions
        print("\n  Reductions relative to baselines:")
        reductions: dict[str, dict] = {}
        for bname, blabel in BASELINE_LABELS.items():
            if bname not in baseline_gaps:
                print(f"    vs {blabel:30s}  NO DATA")
                continue
            ref = baseline_gaps[bname]
            pct = pct_reduction(ref, best_gap)
            abs_red = ref - best_gap
            print(f"    vs {blabel:30s}  {ref:.4f} → {best_gap:.4f}  "
                  f"(↓ {abs_red:.4f}, {pct:.1f}%)")
            reductions[bname] = {
                "baseline_mean_gap": ref,
                "best_mean_gap": best_gap,
                "absolute_reduction": abs_red,
                "pct_reduction": pct,
            }

        results[tm] = {
            "best_predictor": best_exp,
            "best_predictor_label": EXPERIMENT_LABELS[best_exp],
            "best_mean_gap": best_gap,
            "baselines": {
                bname: baseline_gaps[bname]
                for bname in BASELINES
                if bname in baseline_gaps
            },
            "reductions": reductions,
        }

        # Reductions for every included predictor
        print("\n  Reductions for all predictors:")
        print(
            f"    {'Predictor':50s}  {'Mean gap':>9}  {'vs Naive':>10}  "
            f"{'vs LCNet':>10}  {'vs LC Ext.':>11}"
        )
        print(f"    {'─' * 96}")

        predictor_results: dict[str, dict] = {}
        for exp in experiment_order:
            if exp not in predictor_experiments:
                continue
            gap = mean_gaps.get((tm, exp))
            if gap is None:
                continue

            predictor_reductions: dict[str, dict] = {}
            formatted_percentages: dict[str, str] = {}
            for baseline_name in BASELINES:
                reference = baseline_gaps.get(baseline_name)
                if reference is None or reference == 0.0:
                    formatted_percentages[baseline_name] = "n/a"
                    continue
                absolute_reduction = reference - gap
                percentage_reduction = pct_reduction(reference, gap)
                formatted_percentages[baseline_name] = f"{percentage_reduction:+.1f}%"
                predictor_reductions[baseline_name] = {
                    "baseline_mean_gap": reference,
                    "predictor_mean_gap": gap,
                    "absolute_reduction": absolute_reduction,
                    "pct_reduction": percentage_reduction,
                }

            print(
                f"    {EXPERIMENT_LABELS[exp]:50s}  {gap:9.4f}  "
                f"{formatted_percentages.get('naive_baseline', 'n/a'):>10}  "
                f"{formatted_percentages.get('lcnet_baseline', 'n/a'):>10}  "
                f"{formatted_percentages.get('lc_extrapolation_baseline', 'n/a'):>11}"
            )
            predictor_results[exp] = {
                "predictor_label": EXPERIMENT_LABELS[exp],
                "mean_gap": gap,
                "reductions": predictor_reductions,
            }

        results[tm]["predictors"] = predictor_results

    # All-predictor table for traceability
    print(f"\n{'─' * 72}")
    print("  Full mean-gap table (all experiments with data)")
    print(f"{'─' * 72}")
    for tm in TARGET_MODES:
        label = "Final-Value" if tm == "final_value" else "Horizon-Value"
        print(f"\n  {label}:")
        tm_gaps = {exp: g for (t, exp), g in mean_gaps.items() if t == tm}
        for exp in experiment_order:
            if exp in tm_gaps:
                marker = "  [baseline]" if exp in BASELINES else ""
                print(f"    {EXPERIMENT_LABELS[exp]:50s}  {tm_gaps[exp]:.4f}{marker}")

    return results


def report_per_zoo(
    zoo_gaps: dict[tuple[str, int, str], float],
    predictor_experiments: set[str] | None = None,
    experiment_order: list[str] | None = None,
) -> dict:
    """Print and return mean gap per zoo for each baseline + best predictor."""
    if predictor_experiments is None:
        predictor_experiments = set(ALL_PREDICTORS)
    if experiment_order is None:
        experiment_order = ALL_EXPERIMENTS

    results: dict[str, dict] = {}

    def _fmt_gap(v: float | None) -> str:
        return f"{v:.4f}" if v is not None else "  n/a "

    def _fmt_pct(ref: float | None, best: float | None) -> str:
        if ref is None or best is None or ref == 0.0:
            return "  n/a "
        return f"{pct_reduction(ref, best):+.1f}%"

    for tm in TARGET_MODES:
        label = "Final-Value" if tm == "final_value" else "Horizon-Value"
        print(f"\n{'═' * 105}")
        print(f"  Per-Zoo Mean Best-Survivor Gap  [{label}]")
        print(f"{'═' * 105}")

        col_w = 30
        header = (
            f"  {'Zoo':{col_w}s}"
            f"  {'Naive':>7}"
            f"  {'LCNet':>7}"
            f"  {'LC Ext.':>7}"
            f"  {'Best':>7}"
            f"  {'%Naive':>8}"
            f"  {'%LCNet':>8}"
            f"  {'%LC Ext.':>9}"
            f"  Best predictor"
        )
        print(header)
        print(f"  {'─' * 103}")

        zoo_results: dict[int, dict] = {}
        seen: set[int] = set()
        ordered_ids = DATASET_ORDER + [
            ds for (tm2, ds, _) in zoo_gaps if tm2 == tm and ds not in DATASET_ORDER
        ]
        for ds_id in ordered_ids:
            if ds_id in seen:
                continue
            seen.add(ds_id)

            baseline_vals: dict[str, float | None] = {
                bname: zoo_gaps.get((tm, ds_id, bname)) for bname in BASELINES
            }
            best_exp, best_gap = find_best_predictor_for_zoo(
                zoo_gaps,
                tm,
                ds_id,
                predictor_experiments,
            )

            if all(v is None for v in baseline_vals.values()) and best_gap is None:
                continue

            zoo_name = DATASET_DISPLAY.get(ds_id, f"Dataset {ds_id}")
            naive_gap = baseline_vals["naive_baseline"]
            lcnet_gap = baseline_vals["lcnet_baseline"]
            lcext_gap = baseline_vals["lc_extrapolation_baseline"]
            best_label = EXPERIMENT_LABELS.get(best_exp, best_exp) if best_exp else "n/a"

            row = (
                f"  {zoo_name:{col_w}s}"
                f"  {_fmt_gap(naive_gap):>7}"
                f"  {_fmt_gap(lcnet_gap):>7}"
                f"  {_fmt_gap(lcext_gap):>7}"
                f"  {_fmt_gap(best_gap):>7}"
                f"  {_fmt_pct(naive_gap, best_gap):>8}"
                f"  {_fmt_pct(lcnet_gap, best_gap):>8}"
                f"  {_fmt_pct(lcext_gap, best_gap):>9}"
                f"  {best_label}"
            )
            print(row)

            predictor_results: dict[str, dict] = {}
            for exp in experiment_order:
                if exp not in predictor_experiments:
                    continue
                predictor_gap = zoo_gaps.get((tm, ds_id, exp))
                if predictor_gap is None:
                    continue

                predictor_reductions: dict[str, dict] = {}
                for baseline_name, reference in baseline_vals.items():
                    if reference is None or reference == 0.0:
                        continue
                    absolute_reduction = reference - predictor_gap
                    predictor_reductions[baseline_name] = {
                        "baseline_mean_gap": reference,
                        "predictor_mean_gap": predictor_gap,
                        "absolute_reduction": absolute_reduction,
                        "pct_reduction": pct_reduction(reference, predictor_gap),
                    }

                predictor_results[exp] = {
                    "predictor_label": EXPERIMENT_LABELS[exp],
                    "mean_gap": predictor_gap,
                    "reductions": predictor_reductions,
                }

            zoo_results[ds_id] = {
                "zoo_name": zoo_name,
                "naive_gap": naive_gap,
                "lcnet_gap": lcnet_gap,
                "lc_extrapolation_gap": lcext_gap,
                "best_predictor": best_exp,
                "best_predictor_label": best_label,
                "best_gap": best_gap,
                "pct_reduction_vs_naive": pct_reduction(naive_gap, best_gap) if naive_gap and best_gap is not None else None,
                "pct_reduction_vs_lcnet": pct_reduction(lcnet_gap, best_gap) if lcnet_gap and best_gap is not None else None,
                "pct_reduction_vs_lc_extrapolation": pct_reduction(lcext_gap, best_gap) if lcext_gap and best_gap is not None else None,
                "predictors": predictor_results,
            }

        print(f"\n  Detailed predictor reductions by zoo [{label}]")
        for zoo_result in zoo_results.values():
            print(f"\n  {zoo_result['zoo_name']}")
            print(
                f"    {'Predictor':50s}  {'Mean gap':>9}  {'vs Naive':>10}  "
                f"{'vs LCNet':>10}  {'vs LC Ext.':>11}"
            )
            print(f"    {'─' * 96}")
            for predictor_result in zoo_result["predictors"].values():
                reductions = predictor_result["reductions"]

                def _predictor_pct(baseline_name: str) -> str:
                    reduction = reductions.get(baseline_name)
                    if reduction is None:
                        return "n/a"
                    return f"{reduction['pct_reduction']:+.1f}%"

                print(
                    f"    {predictor_result['predictor_label']:50s}  "
                    f"{predictor_result['mean_gap']:9.4f}  "
                    f"{_predictor_pct('naive_baseline'):>10}  "
                    f"{_predictor_pct('lcnet_baseline'):>10}  "
                    f"{_predictor_pct('lc_extrapolation_baseline'):>11}"
                )

        results[tm] = zoo_results
    return results


def main() -> None:
    args = parse_args()

    if not BATCH_ROOT.exists():
        print(f"ERROR: batch output directory not found:\n  {BATCH_ROOT}", file=sys.stderr)
        sys.exit(1)

    predictor_experiments = set(
        OFFLINE_PREDICTORS if args.offline_only else ALL_PREDICTORS
    )
    experiment_order = [
        experiment
        for experiment in ALL_EXPERIMENTS
        if experiment in BASELINES or experiment in predictor_experiments
    ]

    rows = load_runs(set(experiment_order))
    if not rows:
        print("ERROR: no simulation runs loaded — check BATCH_ROOT path.", file=sys.stderr)
        sys.exit(1)
    filter_label = "offline predictors only" if args.offline_only else "all predictors"
    print(f"Loaded {len(rows)} simulation runs ({filter_label}; baselines retained).\n")

    mean_gaps = compute_mean_gaps(rows)
    zoo_gaps = compute_mean_gaps_per_zoo(rows)

    results = report(mean_gaps, predictor_experiments, experiment_order)
    zoo_results = report_per_zoo(
        zoo_gaps,
        predictor_experiments,
        experiment_order,
    )

    # Persist for citation / reproducibility audit
    out = {
        "filters": {"offline_only": args.offline_only},
        "aggregate": results,
        "per_zoo": zoo_results,
    }
    output_name = (
        "gap_reduction_summary_offline_only.json"
        if args.offline_only
        else "gap_reduction_summary.json"
    )
    out_path = REPO_ROOT / "sotf-selection" / "outputs" / output_name
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2))
    print(f"\nResults written to: {out_path.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
