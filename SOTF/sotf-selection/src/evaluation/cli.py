from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

import pandas as pd
from omegaconf import OmegaConf

from .analysis import (
    build_predictor_comparison,
    build_predictor_summary,
    evaluate_runs,
    filter_runs,
    load_completed_runs,
    load_runs_from_explicit_paths,
    normalize_predictor_name,
    parse_experiment_path_specs,
    resolve_results_root,
    select_latest_runs,
)
from .plots import (
    render_best_survivor_regret_comparison,
    render_best_survivor_regret_comparison_by_experiment,
    render_population_accuracy_boxplots,
    render_population_accuracy_comparison_grid,
)
from .reporting import print_precision_recall_summary, write_summary_report
from .shared import (
    build_experiment_subtitle,
    build_filter_summary,
    build_model_zoo_subtitle,
    sanitize_file_stem,
)
from .tracking import create_evaluation_context, finalize_evaluation


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate survival-of-the-fittest selection runs from one or more experiment paths.",
    )
    parser.add_argument(
        "experiment_root",
        nargs="?",
        type=Path,
        default=None,
        help="Experiment root or results directory to evaluate. Optional when --experiment-path is used.",
    )
    parser.add_argument(
        "--experiment-path",
        action="append",
        default=None,
        help=(
            "Explicit predictor/path pair in the format `predictor=/path/to/experiment_or_results`. "
            "Repeat to compare predictors from different directories."
        ),
    )
    parser.add_argument(
        "--baseline-root",
        type=Path,
        default=None,
        help="Optional baseline experiment root or results directory to include for comparison (legacy mode).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help=(
            "Directory where plots and CSV files will be written. "
            "Defaults to sotf-selection/outputs/evaluation/<name>_<evaluation-id> when the source is under outputs/."
        ),
    )
    parser.add_argument(
        "--predictor",
        action="append",
        default=None,
        help="Predictor(s) to include, for example `forecast_model` or `naive`. Repeat the flag to compare multiple predictors.",
    )
    parser.add_argument("--target-column", default=None, help="Filter to a specific target column, for example `test_acc`.")
    parser.add_argument("--target-mode", default=None, help="Filter to a specific sequence target mode, for example `horizon_value`.")
    parser.add_argument("--horizon", type=int, default=None, help="Filter to a specific forecast horizon.")
    parser.add_argument("--start-epoch", type=int, default=None, help="Filter to a specific selection start epoch.")
    parser.add_argument("--interval", type=int, default=None, help="Filter to a specific selection interval.")
    parser.add_argument("--survival-rate", type=float, default=None, help="Filter to a specific survival rate.")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    explicit_paths = parse_experiment_path_specs(args.experiment_path)
    if explicit_paths and (args.experiment_root is not None or args.baseline_root is not None):
        raise SystemExit(
            "Use either explicit `--experiment-path predictor=path` entries OR positional `experiment_root`/`--baseline-root`, not both."
        )

    if explicit_paths:
        completed_runs = load_runs_from_explicit_paths(explicit_paths)
        primary_root = explicit_paths[0][1]
        source_roots = unique_paths([path for _, path in explicit_paths])
        experiment_root_label = "explicit_paths"
        baseline_root_label = None
        experiment_paths_label = [f"{predictor}={path}" for predictor, path in explicit_paths]
    else:
        if args.experiment_root is None:
            raise SystemExit("Provide `experiment_root` or at least one `--experiment-path predictor=path` entry.")
        experiment_root = args.experiment_root.resolve()
        completed_runs = load_completed_runs(resolve_results_root(experiment_root))
        if completed_runs.empty:
            raise SystemExit(f"No completed SOTF runs found under {experiment_root}")

        source_roots = [experiment_root]
        if args.baseline_root is not None:
            baseline_root = args.baseline_root.resolve()
            baseline_runs = load_completed_runs(resolve_results_root(baseline_root))
            if baseline_runs.empty:
                raise SystemExit(f"No completed SOTF runs found under baseline root {baseline_root}")
            completed_runs = pd.concat([completed_runs, baseline_runs], ignore_index=True, sort=False)
            baseline_root_label = str(baseline_root)
            source_roots.append(baseline_root)
        else:
            baseline_root_label = None
        primary_root = experiment_root
        experiment_root_label = str(experiment_root)
        experiment_paths_label = None

    predictor_filters = [normalize_predictor_name(value) for value in args.predictor] if args.predictor else None
    filters = build_filter_summary(
        experiment_root=experiment_root_label,
        baseline_root=baseline_root_label,
        experiment_paths=experiment_paths_label,
        predictors=predictor_filters,
        target_column=args.target_column,
        target_mode=args.target_mode,
        horizon=args.horizon,
        start_epoch=args.start_epoch,
        interval=args.interval,
        survival_rate=args.survival_rate,
    )
    detected_modes = _detected_target_modes(completed_runs, explicit_target_mode=args.target_mode)
    context = create_evaluation_context(
        name=build_evaluation_name(primary_root=primary_root, explicit_paths=explicit_paths, filters=filters, detected_target_modes=detected_modes),
        source_roots=unique_paths(source_roots),
        requested_output_dir=args.output_dir,
        filters=filters,
    )
    remove_obsolete_plots(context.output_dir)

    filtered_runs = filter_runs(
        completed_runs,
        predictors=predictor_filters,
        target_column=args.target_column,
        target_mode=args.target_mode,
        horizon=args.horizon,
        start_epoch=args.start_epoch,
        interval=args.interval,
        survival_rate=args.survival_rate,
    )
    if filtered_runs.empty:
        raise SystemExit("No completed SOTF runs matched the requested metadata filters.")

    experiment_order = load_batch_experiment_order(source_roots)
    filtered_runs = attach_experiment_order(filtered_runs, experiment_order)

    latest_runs = select_latest_runs(filtered_runs)
    latest_runs = sort_latest_runs(latest_runs)

    round_metrics, dataset_summary = evaluate_runs(latest_runs)
    round_metrics.to_csv(context.output_dir / "round_metrics.csv", index=False)
    predictor_summary = build_predictor_summary(dataset_summary)
    predictor_comparison = build_predictor_comparison(dataset_summary)

    dataset_subtitle = build_experiment_subtitle(dataset_summary)
    dataset_secondary_subtitle = build_model_zoo_subtitle(dataset_summary)
    png_dir = context.output_dir / "png"
    render_best_survivor_regret_comparison_by_experiment(
        dataset_summary,
        png_dir / "11_best_survivor_gap_by_experiment.png",
        subtitle=dataset_subtitle,
        secondary_subtitle=dataset_secondary_subtitle,
    )

    if not predictor_comparison.empty:
        render_best_survivor_regret_comparison(
            predictor_comparison,
            png_dir / "11_best_survivor_gap_comparison.png",
            subtitle=dataset_subtitle,
            secondary_subtitle=dataset_secondary_subtitle,
        )

    render_population_accuracy_boxplots(
        latest_runs,
        png_dir / "12_population_accuracy_boxplots.png",
        subtitle=dataset_subtitle,
        secondary_subtitle=dataset_secondary_subtitle,
    )

    render_population_accuracy_comparison_grid(
        latest_runs,
        png_dir / "13_population_accuracy_comparison_grid.png",
        subtitle=dataset_subtitle,
        secondary_subtitle=dataset_secondary_subtitle,
    )

    write_summary_report(
        dataset_summary=dataset_summary,
        predictor_summary=predictor_summary,
        predictor_comparison=predictor_comparison,
        round_metrics=round_metrics,
        output_path=context.output_dir / "summary.md",
        filters=filters,
        evaluation_id=context.evaluation_id,
    )

    finalize_evaluation(context)
    print_precision_recall_summary(predictor_summary)
    print(f"Evaluation ID: {context.evaluation_id}")
    print(f"Wrote SOTF evaluation outputs to {context.output_dir}")
    return 0


def remove_obsolete_plots(output_dir: Path) -> None:
    obsolete_paths = [output_dir / "10_predictor_comparison.png"]
    obsolete_paths.extend(output_dir.glob("04_*_best_survivor_gap.png"))
    for path in obsolete_paths:
        if path.exists():
            path.unlink()


def build_evaluation_name(
    *,
    primary_root: Path,
    explicit_paths: list[tuple[str, Path]],
    filters: dict[str, object],
    detected_target_modes: list[str] | None = None,
) -> str:
    parts = ["sotf", primary_root.parent.name, primary_root.name]
    if explicit_paths:
        parts.extend(predictor for predictor, _ in explicit_paths)
    for key in ("predictors", "target_column", "target_mode", "horizon", "start_epoch", "interval", "survival_rate"):
        value = filters.get(key)
        if key == "target_mode" and value in {None, "all", "none"} and detected_target_modes:
            parts.extend(detected_target_modes)
            continue
        if isinstance(value, list):
            parts.extend(str(item) for item in value)
            continue
        if value in {None, "all", "none"}:
            continue
        parts.append(str(value))
    return "_".join(sanitize_file_stem(part) for part in parts if str(part).strip())


def load_batch_experiment_order(source_roots: Sequence[Path]) -> dict[str, int]:
    """Load experiment ordering from copied batch YAML files near the evaluated roots."""
    order: dict[str, int] = {}
    for config_path in batch_config_candidate_paths(source_roots):
        for experiment_name in experiment_names_from_batch_config(config_path):
            order.setdefault(experiment_name, len(order))
    return order


def batch_config_candidate_paths(source_roots: Sequence[Path]) -> list[Path]:
    candidates: list[Path] = []
    seen: set[str] = set()
    for source_root in source_roots:
        resolved = source_root.resolve()
        roots_to_check = [resolved, *resolved.parents]
        for root in roots_to_check:
            batch_files_dir = root / "batch_files"
            if not batch_files_dir.is_dir():
                continue
            for config_path in sorted((*batch_files_dir.glob("*.yaml"), *batch_files_dir.glob("*.yml"))):
                key = str(config_path.resolve())
                if key in seen:
                    continue
                seen.add(key)
                candidates.append(config_path)
            break
    return candidates


def experiment_names_from_batch_config(config_path: Path) -> list[str]:
    try:
        raw_config = OmegaConf.to_container(OmegaConf.load(config_path), resolve=True)
    except Exception:
        return []
    plan_config = raw_config
    if isinstance(raw_config, dict) and isinstance(raw_config.get("experiments"), dict):
        plan_config = raw_config["experiments"]
    if not isinstance(plan_config, dict):
        return []
    runs = plan_config.get("runs")
    if not isinstance(runs, list):
        return []

    names: list[str] = []
    for run_config in runs:
        if not isinstance(run_config, dict):
            continue
        experiment_name = str(run_config.get("experiment_name", "")).strip()
        if experiment_name:
            names.append(experiment_name)
    return names


def attach_experiment_order(frame: pd.DataFrame, experiment_order: dict[str, int]) -> pd.DataFrame:
    if frame.empty or not experiment_order or "experiment_name" not in frame.columns:
        return frame.copy()
    ordered = frame.copy()
    ordered["experiment_order"] = ordered["experiment_name"].astype(str).map(experiment_order)
    return ordered


def sort_latest_runs(latest_runs: pd.DataFrame) -> pd.DataFrame:
    if latest_runs.empty:
        return latest_runs.copy()
    sort_columns = ["predictor", "dataset_number", "experiment_name", "run_started_at"]
    if "experiment_order" in latest_runs.columns:
        sort_columns = ["experiment_order", *sort_columns]
    return latest_runs.sort_values(sort_columns, kind="stable").reset_index(drop=True)


def _detected_target_modes(runs: pd.DataFrame, *, explicit_target_mode: str | None) -> list[str] | None:
    """Return sorted unique target_mode values from runs when no explicit filter was given."""
    if explicit_target_mode is not None or runs.empty or "target_mode" not in runs.columns:
        return None
    modes = sorted(runs["target_mode"].dropna().unique().tolist())
    return modes if modes else None


def unique_paths(paths: list[Path]) -> list[Path]:
    unique: list[Path] = []
    seen: set[str] = set()
    for path in paths:
        resolved = path.resolve()
        key = str(resolved)
        if key in seen:
            continue
        seen.add(key)
        unique.append(resolved)
    return unique
