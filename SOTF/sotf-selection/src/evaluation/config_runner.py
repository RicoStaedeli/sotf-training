from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import pandas as pd
from omegaconf import OmegaConf

from .analysis import (
    build_predictor_comparison,
    build_predictor_summary,
    evaluate_runs,
    filter_runs,
    load_completed_runs,
    normalize_predictor_name,
    resolve_results_root,
    select_latest_runs,
)
from .cli import (
    _detected_target_modes,
    attach_experiment_order,
    build_evaluation_name,
    load_batch_experiment_order,
    remove_obsolete_plots,
    sort_latest_runs,
    unique_paths,
)
from .plots import (
    render_best_survivor_regret_comparison,
    render_best_survivor_regret_comparison_by_experiment,
    render_cross_dataset_best_survivor_regret_bars,
    render_cross_dataset_gap_dominance_matrix,
    render_cross_dataset_gap_parameter_map,
    render_cross_dataset_gap_robustness_scorecard,
    render_cross_dataset_win_heatmap,
    render_cross_dataset_win_matrix,
    render_population_accuracy_boxplots,
    render_population_accuracy_comparison_grid,
    render_sotf_combined_sensitivity,
    render_sotf_compute_savings_sensitivity,
    render_sotf_efficiency_ratio,
    render_sotf_parameter_sensitivity,
    render_sotf_parameter_sensitivity_delta,
    render_sotf_parameter_sensitivity_delta_with_compute,
)
from .reporting import print_precision_recall_summary, write_summary_report
from .shared import (
    build_experiment_subtitle,
    build_filter_summary,
    build_model_zoo_subtitle,
)
from .tracking import create_evaluation_context, finalize_evaluation

log = logging.getLogger(__name__)


def run_from_config(config_path: Path) -> int:
    """Load a YAML evaluation config and generate the requested plots.

    The config must specify ``batches_dir``: a directory whose immediate
    subdirectories are individual batch outputs.  The evaluation runs once
    per discovered batch, sharing the same plot config and filters.

    Args:
        config_path: Absolute path to the YAML config file.

    Returns:
        Exit code (0 on success).
    """
    raw: Any = OmegaConf.to_container(OmegaConf.load(config_path), resolve=True)
    if not isinstance(raw, dict):
        raise SystemExit(f"Config {config_path} must be a YAML mapping.")

    batches_dir = _resolve_path(raw.get("batches_dir"), config_path)
    if not batches_dir or not batches_dir.exists():
        raise SystemExit(f"batches_dir does not exist: {raw.get('batches_dir')!r}")

    batch_roots = _discover_batches(batches_dir)

    extra_raw = raw.get("extra_batches_dirs") or []
    if isinstance(extra_raw, str):
        extra_raw = [extra_raw]
    for extra_entry in extra_raw:
        extra_dir = _resolve_path(extra_entry, config_path)
        if not extra_dir or not extra_dir.exists():
            log.warning("extra_batches_dirs entry does not exist — skipping: %r", extra_entry)
            continue
        batch_roots.extend(_discover_batches(extra_dir))

    if not batch_roots:
        raise SystemExit(f"No batch subdirectories with completed runs found under {batches_dir}")

    log.info("Found %d batch(es) under %s", len(batch_roots), batches_dir)

    failed: list[tuple[Path, str]] = []
    all_summaries: list[pd.DataFrame] = []
    for batch_root in batch_roots:
        print(f"\n── Evaluating batch: {batch_root.name} ──")
        try:
            summary = _run_single_batch(batch_root, raw, config_path)
            if summary is not None and not summary.empty:
                all_summaries.append(summary)
        except SystemExit as exc:
            msg = str(exc)
            log.error("Batch %s failed: %s", batch_root.name, msg)
            failed.append((batch_root, msg))

    if all_summaries:
        cross_batch_summary = pd.concat(all_summaries, ignore_index=True, sort=False)
        plots_raw: dict = raw.get("plots") or {}
        _render_parameter_sensitivity(plots_raw, cross_batch_summary, batches_dir)

    if failed:
        print(f"\n{len(failed)} batch(es) failed:")
        for path, msg in failed:
            print(f"  {path.name}: {msg}")
        return 1
    return 0


# ---------------------------------------------------------------------------
# Single-batch evaluation
# ---------------------------------------------------------------------------

def _run_single_batch(batch_root: Path, raw: dict, config_path: Path) -> pd.DataFrame | None:
    filters_raw: dict = raw.get("filters") or {}
    predictor_filters = _normalised_list(filters_raw.get("predictor"))
    target_column = _str_or_none(filters_raw.get("target_column"))
    target_mode = _str_or_none(filters_raw.get("target_mode"))
    horizon = _int_or_none(filters_raw.get("horizon"))
    start_epoch = _int_or_none(filters_raw.get("start_epoch"))
    interval = _int_or_none(filters_raw.get("interval"))
    survival_rate = _float_or_none(filters_raw.get("survival_rate"))

    completed_runs = load_completed_runs(resolve_results_root(batch_root))
    if completed_runs.empty:
        raise SystemExit(f"No completed runs found under {batch_root}")

    filters = build_filter_summary(
        experiment_root=str(batch_root),
        baseline_root=None,
        experiment_paths=None,
        predictors=predictor_filters,
        target_column=target_column,
        target_mode=target_mode,
        horizon=horizon,
        start_epoch=start_epoch,
        interval=interval,
        survival_rate=survival_rate,
    )

    requested_output_dir = _resolve_path(raw.get("output_dir"), config_path)
    detected_modes = _detected_target_modes(completed_runs, explicit_target_mode=target_mode)
    context = create_evaluation_context(
        name=build_evaluation_name(primary_root=batch_root, explicit_paths=[], filters=filters, detected_target_modes=detected_modes),
        source_roots=unique_paths([batch_root]),
        requested_output_dir=requested_output_dir,
        filters=filters,
    )
    remove_obsolete_plots(context.output_dir)

    filtered_runs = filter_runs(
        completed_runs,
        predictors=predictor_filters,
        target_column=target_column,
        target_mode=target_mode,
        horizon=horizon,
        start_epoch=start_epoch,
        interval=interval,
        survival_rate=survival_rate,
    )
    if filtered_runs.empty:
        raise SystemExit("No runs matched the requested filters.")

    experiment_order = load_batch_experiment_order([batch_root])
    filtered_runs = attach_experiment_order(filtered_runs, experiment_order)
    latest_runs = sort_latest_runs(select_latest_runs(filtered_runs))

    round_metrics, dataset_summary = evaluate_runs(latest_runs)
    round_metrics.to_csv(context.output_dir / "round_metrics.csv", index=False)
    predictor_summary = build_predictor_summary(dataset_summary)
    predictor_comparison = build_predictor_comparison(dataset_summary)

    subtitle = build_experiment_subtitle(dataset_summary)
    secondary_subtitle = build_model_zoo_subtitle(dataset_summary)
    png_dir = context.output_dir / "png"
    plots_raw: dict = raw.get("plots") or {}

    _render_best_survivor_gap_by_experiment(plots_raw, dataset_summary, png_dir, subtitle, secondary_subtitle)
    _render_best_survivor_gap_comparison(plots_raw, predictor_comparison, png_dir, subtitle, secondary_subtitle)
    _render_population_accuracy_boxplots(plots_raw, latest_runs, png_dir, subtitle, secondary_subtitle)
    _render_population_accuracy_comparison_grid(plots_raw, latest_runs, png_dir, subtitle, secondary_subtitle)

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
    print(f"Wrote outputs to {context.output_dir}")
    return dataset_summary


def _render_parameter_sensitivity(
    plots_raw: dict,
    cross_batch_summary: pd.DataFrame,
    batches_dir: Path,
) -> None:
    abs_cfg = _plot_cfg(plots_raw, "parameter_sensitivity")
    delta_cfg = _plot_cfg(plots_raw, "parameter_sensitivity_delta")
    compute_cfg = _plot_cfg(plots_raw, "compute_savings_sensitivity")
    delta_compute_cfg = _plot_cfg(plots_raw, "parameter_sensitivity_delta_with_compute")
    efficiency_cfg = _plot_cfg(plots_raw, "efficiency_ratio")
    combined_cfg = _plot_cfg(plots_raw, "combined_sensitivity")

    any_enabled = any(_enabled(c) for c in [
        abs_cfg, delta_cfg, compute_cfg, delta_compute_cfg, efficiency_cfg, combined_cfg,
    ])
    if not any_enabled:
        return

    subtitle = build_experiment_subtitle(cross_batch_summary)
    secondary_subtitle = ""  # Dataset name is already appended to subtitle for cross-batch plots

    modes = _detected_target_modes(cross_batch_summary, explicit_target_mode=None)
    mode_tag = "_".join(modes) if modes else batches_dir.parent.name
    context = create_evaluation_context(
        name=f"cross_batch_{batches_dir.name}_{mode_tag}",
        source_roots=unique_paths([batches_dir]),
        requested_output_dir=None,
        filters={},
    )
    png_dir = context.output_dir / "png"
    paths: list = []

    if _enabled(abs_cfg):
        metrics_cfg = abs_cfg.get("metrics")
        metrics: list[str] | None = [str(m) for m in metrics_cfg] if metrics_cfg else None
        baseline_predictor = _str_or_none(abs_cfg.get("baseline_predictor")) or "online_lc_extrapolation"
        paths.extend(render_sotf_parameter_sensitivity(
            cross_batch_summary, png_dir,
            metrics=metrics, subtitle=subtitle, secondary_subtitle=secondary_subtitle,
            baseline_predictor=baseline_predictor,
        ))

    if _enabled(delta_cfg):
        metrics_cfg = delta_cfg.get("metrics")
        metrics = [str(m) for m in metrics_cfg] if metrics_cfg else None
        paths.extend(render_sotf_parameter_sensitivity_delta(
            cross_batch_summary, png_dir,
            metrics=metrics, subtitle=subtitle, secondary_subtitle=secondary_subtitle,
        ))

    if _enabled(compute_cfg):
        paths.extend(render_sotf_compute_savings_sensitivity(
            cross_batch_summary, png_dir,
            subtitle=subtitle, secondary_subtitle=secondary_subtitle,
        ))

    if _enabled(delta_compute_cfg):
        metrics_cfg = delta_compute_cfg.get("metrics")
        metrics = [str(m) for m in metrics_cfg] if metrics_cfg else None
        show_title = bool(delta_compute_cfg.get("show_title", True))
        paths.extend(render_sotf_parameter_sensitivity_delta_with_compute(
            cross_batch_summary, png_dir,
            metrics=metrics, subtitle=subtitle, secondary_subtitle=secondary_subtitle,
            show_title=show_title,
        ))

    if _enabled(efficiency_cfg):
        paths.extend(render_sotf_efficiency_ratio(
            cross_batch_summary, png_dir,
            subtitle=subtitle, secondary_subtitle=secondary_subtitle,
        ))

    if _enabled(combined_cfg):
        metrics_cfg = combined_cfg.get("metrics")
        metrics = [str(m) for m in metrics_cfg] if metrics_cfg else None
        paths.extend(render_sotf_combined_sensitivity(
            cross_batch_summary, png_dir,
            metrics=metrics,
        ))

    if paths:
        finalize_evaluation(context)
        print(f"\n── Cross-batch parameter sensitivity ──")
        print(f"Wrote {len(paths)} sensitivity plot(s) to {context.output_dir}")


def _discover_batches(batches_dir: Path) -> list[Path]:
    """Return sorted subdirectories of batches_dir that contain completed runs."""
    found: list[Path] = []
    for subdir in sorted(batches_dir.iterdir()):
        if not subdir.is_dir():
            continue
        results_root = resolve_results_root(subdir)
        if any(results_root.rglob("simulation_summary.json")):
            found.append(subdir)
    return found


# ---------------------------------------------------------------------------
# Per-plot render helpers
# ---------------------------------------------------------------------------

def _render_best_survivor_gap_by_experiment(
    plots_raw: dict,
    dataset_summary: pd.DataFrame,
    png_dir: Path,
    subtitle: str,
    secondary_subtitle: str,
) -> None:
    cfg = _plot_cfg(plots_raw, "best_survivor_gap_by_experiment")
    if not _enabled(cfg):
        return
    render_best_survivor_regret_comparison_by_experiment(
        dataset_summary,
        png_dir / "11_best_survivor_gap_by_experiment.png",
        subtitle=subtitle,
        secondary_subtitle=secondary_subtitle,
    )


def _render_best_survivor_gap_comparison(
    plots_raw: dict,
    predictor_comparison: pd.DataFrame,
    png_dir: Path,
    subtitle: str,
    secondary_subtitle: str,
) -> None:
    cfg = _plot_cfg(plots_raw, "best_survivor_gap_comparison")
    if not _enabled(cfg) or predictor_comparison.empty:
        return
    render_best_survivor_regret_comparison(
        predictor_comparison,
        png_dir / "11_best_survivor_gap_comparison.png",
        subtitle=subtitle,
        secondary_subtitle=secondary_subtitle,
    )


def _render_population_accuracy_boxplots(
    plots_raw: dict,
    latest_runs: pd.DataFrame,
    png_dir: Path,
    subtitle: str,
    secondary_subtitle: str,
) -> None:
    cfg = _plot_cfg(plots_raw, "population_accuracy_boxplots")
    if not _enabled(cfg):
        return
    runs = _filter_by_experiments(latest_runs, _experiment_list(cfg))
    if runs.empty:
        return
    render_population_accuracy_boxplots(
        runs,
        png_dir / "12_population_accuracy_boxplots.png",
        subtitle=subtitle,
        secondary_subtitle=secondary_subtitle,
    )


def _render_population_accuracy_comparison_grid(
    plots_raw: dict,
    latest_runs: pd.DataFrame,
    png_dir: Path,
    subtitle: str,
    secondary_subtitle: str,
) -> None:
    cfg = _plot_cfg(plots_raw, "population_accuracy_comparison_grid")
    if not _enabled(cfg):
        return
    runs = _filter_by_experiments(latest_runs, _experiment_list(cfg))
    if runs.empty:
        return
    render_population_accuracy_comparison_grid(
        runs,
        png_dir / "13_population_accuracy_comparison_grid.png",
        subtitle=subtitle,
        secondary_subtitle=secondary_subtitle,
    )


# ---------------------------------------------------------------------------
# Config helpers
# ---------------------------------------------------------------------------

def _plot_cfg(plots_raw: dict, name: str) -> dict:
    value = plots_raw.get(name)
    if value is None:
        return {"enabled": True}
    if isinstance(value, bool):
        return {"enabled": value}
    return dict(value)


def _enabled(cfg: dict) -> bool:
    return bool(cfg.get("enabled", True))


def _experiment_list(cfg: dict) -> list[str] | None:
    raw = cfg.get("experiments")
    if not raw:
        return None
    return [str(e) for e in raw]


def _filter_by_experiments(runs: pd.DataFrame, experiments: list[str] | None) -> pd.DataFrame:
    if experiments is None or "experiment_name" not in runs.columns:
        return runs
    mask = runs["experiment_name"].astype(str).isin(experiments)
    if not mask.any():
        log.warning("experiments filter matched no runs; check experiment names in config")
    return runs[mask].reset_index(drop=True)


def _resolve_path(value: Any, config_path: Path) -> Path | None:
    if not value:
        return None
    p = Path(str(value))
    if p.is_absolute():
        return p
    cwd_candidate = Path.cwd() / p
    if cwd_candidate.exists():
        return cwd_candidate.resolve()
    return (config_path.parent / p).resolve()


def _normalised_list(value: Any) -> list[str] | None:
    if value is None:
        return None
    items = [value] if isinstance(value, str) else list(value)
    return [normalize_predictor_name(v) for v in items] if items else None


def _str_or_none(value: Any) -> str | None:
    return str(value) if value is not None else None


def _int_or_none(value: Any) -> int | None:
    try:
        return int(value) if value is not None else None
    except (TypeError, ValueError):
        return None


def _float_or_none(value: Any) -> float | None:
    try:
        return float(value) if value is not None else None
    except (TypeError, ValueError):
        return None


# ---------------------------------------------------------------------------
# Multi-dataset cross-comparison runner
# ---------------------------------------------------------------------------

def run_from_multi_dataset_config(config_path: Path) -> int:
    """Load a multi-dataset YAML evaluation config and produce cross-dataset win plots.

    The config must contain a ``datasets`` list, each entry with:
      - ``name``            — short label for the dataset
      - ``batches_dir``     — parent directory of batch-output subdirectories
      - ``start_epoch_map`` — mapping from concrete start_epoch → "early"/"mid"/"end"

    Optional top-level keys:
      - ``filters``         — same filter fields as the single-dataset config
      - ``win_metric``      — column used to decide outperformance (default: best_survivor_regret)
      - ``output_dir``      — override output directory
      - ``plots``           — enable/disable individual plots

    Args:
        config_path: Absolute path to the YAML config file.

    Returns:
        Exit code (0 on success, 1 if any dataset failed).
    """
    raw: Any = OmegaConf.to_container(OmegaConf.load(config_path), resolve=True)
    if not isinstance(raw, dict):
        raise SystemExit(f"Config {config_path} must be a YAML mapping.")

    datasets_raw: list = raw.get("datasets") or []
    if not datasets_raw:
        raise SystemExit("Multi-dataset config must contain a non-empty 'datasets' list.")

    filters_raw: dict = raw.get("filters") or {}
    predictor_filters = _normalised_list(filters_raw.get("predictor"))
    target_column = _str_or_none(filters_raw.get("target_column"))
    target_mode = _str_or_none(filters_raw.get("target_mode"))
    horizon = _int_or_none(filters_raw.get("horizon"))
    start_epoch = _int_or_none(filters_raw.get("start_epoch"))
    interval = _int_or_none(filters_raw.get("interval"))
    survival_rate = _float_or_none(filters_raw.get("survival_rate"))

    win_metric = str(raw.get("win_metric") or "best_survivor_regret")
    plots_raw: dict = raw.get("plots") or {}
    requested_output_dir = _resolve_path(raw.get("output_dir"), config_path)

    all_frames: list[pd.DataFrame] = []
    failed: list[tuple[str, str]] = []
    n_datasets_loaded = 0

    for ds_entry in datasets_raw:
        if not isinstance(ds_entry, dict):
            log.warning("Skipping malformed dataset entry: %r", ds_entry)
            continue
        ds_name = str(ds_entry.get("name", "unnamed"))
        batches_dir = _resolve_path(ds_entry.get("batches_dir"), config_path)
        if not batches_dir or not batches_dir.exists():
            log.error("Dataset '%s': batches_dir not found: %r", ds_name, ds_entry.get("batches_dir"))
            failed.append((ds_name, f"batches_dir not found: {ds_entry.get('batches_dir')!r}"))
            continue

        raw_epoch_map = ds_entry.get("start_epoch_map") or {}
        start_epoch_map: dict[int, str] = {int(k): str(v) for k, v in raw_epoch_map.items()}

        print(f"\n── Loading dataset: {ds_name} from {batches_dir} ──")
        batch_roots = _discover_batches(batches_dir)

        extra_dirs_raw = ds_entry.get("extra_batches_dirs") or []
        if isinstance(extra_dirs_raw, str):
            extra_dirs_raw = [extra_dirs_raw]
        for extra_entry in extra_dirs_raw:
            extra_dir = _resolve_path(extra_entry, config_path)
            if not extra_dir or not extra_dir.exists():
                log.warning("Dataset '%s': extra_batches_dirs entry does not exist — skipping: %r", ds_name, extra_entry)
                continue
            extra_roots = _discover_batches(extra_dir)
            print(f"   + extra batches from {extra_dir.name}: {len(extra_roots)} batch(es)")
            batch_roots.extend(extra_roots)

        if not batch_roots:
            log.warning("Dataset '%s': no batch subdirectories with completed runs found.", ds_name)
            continue

        dataset_loaded = False
        for batch_root in batch_roots:
            completed_runs = load_completed_runs(resolve_results_root(batch_root))
            if completed_runs.empty:
                continue
            filtered_runs = filter_runs(
                completed_runs,
                predictors=predictor_filters,
                target_column=target_column,
                target_mode=target_mode,
                horizon=horizon,
                start_epoch=start_epoch,
                interval=interval,
                survival_rate=survival_rate,
            )
            if filtered_runs.empty:
                continue
            latest_runs = sort_latest_runs(select_latest_runs(filtered_runs))
            if latest_runs.empty:
                continue
            try:
                _, ds_summary = evaluate_runs(latest_runs)
            except Exception as exc:
                log.warning("Dataset '%s' batch '%s': evaluate_runs failed: %s", ds_name, batch_root.name, exc)
                continue
            if ds_summary.empty:
                continue
            ds_summary = ds_summary.copy()
            ds_summary["dataset_name"] = ds_name
            ds_summary["start_epoch_category"] = ds_summary["selection_start_epoch"].apply(
                lambda e: start_epoch_map.get(int(e), f"epoch_{int(e)}")
            )
            all_frames.append(ds_summary)
            dataset_loaded = True

        if dataset_loaded:
            n_datasets_loaded += 1

    if not all_frames:
        print("\nNo data loaded from any dataset — check batches_dir paths and filters.")
        return 1

    pooled_df = pd.concat(all_frames, ignore_index=True, sort=False)
    print(f"\nPooled {len(pooled_df)} experiment rows from {n_datasets_loaded} dataset(s).")

    # Build output context
    context = create_evaluation_context(
        name=f"cross_dataset_{config_path.stem}",
        source_roots=unique_paths([
            Path(str(ds.get("batches_dir", ".")))
            for ds in datasets_raw
            if isinstance(ds, dict) and ds.get("batches_dir")
        ]),
        requested_output_dir=requested_output_dir,
        filters={},
    )
    png_dir = context.output_dir / "png"
    png_dir.mkdir(parents=True, exist_ok=True)

    output_paths: list[Path] = []

    if _enabled(_plot_cfg(plots_raw, "cross_dataset_win_heatmap")):
        output_paths.extend(
            render_cross_dataset_win_heatmap(
                pooled_df,
                png_dir,
                win_metric=win_metric,
                n_datasets=n_datasets_loaded,
                subtitle=f"{n_datasets_loaded} datasets | metric: {win_metric}",
            )
        )

    if _enabled(_plot_cfg(plots_raw, "cross_dataset_win_matrix")):
        output_paths.extend(
            render_cross_dataset_win_matrix(
                pooled_df,
                png_dir,
                win_metric=win_metric,
                n_datasets=n_datasets_loaded,
                subtitle=f"{n_datasets_loaded} datasets",
            )
        )

    if _enabled(_plot_cfg(plots_raw, "cross_dataset_best_survivor_gap_bars")):
        output_paths.extend(
            render_cross_dataset_best_survivor_regret_bars(
                pooled_df,
                png_dir,
                subtitle=f"{n_datasets_loaded} datasets",
            )
        )

    dominance_cfg = _plot_cfg(plots_raw, "cross_dataset_gap_dominance_matrix")
    if _enabled(dominance_cfg):
        output_paths.extend(
            render_cross_dataset_gap_dominance_matrix(
                pooled_df,
                png_dir,
                top_n=int(dominance_cfg.get("top_n", 28)),
                subtitle=f"{n_datasets_loaded} datasets",
            )
        )

    scorecard_cfg = _plot_cfg(plots_raw, "cross_dataset_gap_robustness_scorecard")
    if _enabled(scorecard_cfg):
        output_paths.extend(
            render_cross_dataset_gap_robustness_scorecard(
                pooled_df,
                png_dir,
                top_n=int(scorecard_cfg.get("top_n", 34)),
                subtitle=f"{n_datasets_loaded} datasets",
            )
        )

    parameter_map_cfg = _plot_cfg(plots_raw, "cross_dataset_gap_parameter_map")
    if _enabled(parameter_map_cfg):
        output_paths.extend(
            render_cross_dataset_gap_parameter_map(
                pooled_df,
                png_dir,
                top_n=int(parameter_map_cfg.get("top_n", 18)),
                subtitle=f"{n_datasets_loaded} datasets",
            )
        )

    finalize_evaluation(context)
    print(f"\nWrote {len(output_paths)} cross-dataset plot(s) to {context.output_dir}")

    if failed:
        print(f"\n{len(failed)} dataset(s) failed to load:")
        for name, msg in failed:
            print(f"  {name}: {msg}")
        return 1
    return 0
