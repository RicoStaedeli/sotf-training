from __future__ import annotations

from contextlib import ExitStack
import logging
from pathlib import Path
import re
from typing import Any, Dict, List
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import pandas as pd
import yaml

from ..io.registry import load_dataset_registry
from ..io.jsonl import load_dataset_long_df
from ..analyses.registry import ANALYSIS_REGISTRY
from ..analyses.result_store import (
    dataset_output_dir,
    iter_dataset_table_views,
    match_dataset_name_from_key,
)
from .pdf import append_pdf_page, write_single_pdf
from .png import write_png

log = logging.getLogger(__name__)

plt.rcParams["font.family"] = "Times New Roman"

_STREAMING_ANALYSES = {"weight_analysis", "weight_rank_correspondence"}
_PER_DATASET_OUTPUT_ANALYSES = {"weight_analysis", "weight_rank_correspondence", "weight_early_signal"}
_DATASET_ONLY_PNG_ANALYSES = {"weight_analysis", "weight_rank_correspondence", "weight_early_signal"}


def _slugify(value: Any) -> str:
    return re.sub(r"[^A-Za-z0-9]+", "_", str(value)).strip("_").lower()


def _resolve_output_pdf_dir(cfg: Dict[str, Any]) -> Path:
    output_pdf_dir = cfg.get("output_pdf_dir")
    if output_pdf_dir:
        return Path(output_pdf_dir)

    legacy_output_pdf = cfg.get("output_pdf")
    if legacy_output_pdf:
        return Path(legacy_output_pdf).parent

    return Path("outputs/reports/PDF")


def _build_analysis_pdf_stem(analysis_cfg: Dict[str, Any], used_stems: set[str]) -> str:
    parts = [_slugify(analysis_cfg.get("name", "analysis"))]
    for key in ("tag", "metric", "rank_metric", "functional_metric", "metric_family", "scope", "dataset_split"):
        value = analysis_cfg.get(key)
        if value is not None:
            slug = _slugify(value)
            if slug:
                parts.append(slug)

    stem = "_".join(part for part in parts if part) or "analysis"
    candidate = stem
    suffix = 2
    while candidate in used_stems:
        candidate = f"{stem}_{suffix:02d}"
        suffix += 1

    used_stems.add(candidate)
    return candidate


def _normalize_metric_columns(df: pd.DataFrame, metric_aliases: Dict[str, List[str]]) -> pd.DataFrame:
    """
    Rename dataset-specific metric columns to canonical names before concatenation.

    metric_aliases example:
      {"validation_acc": ["validation_acc", "val_acc"], "validation_loss": ["validation_loss", "val_loss"]}
    """
    if df is None or df.empty or not metric_aliases:
        return df

    df = df.copy()

    for canonical, aliases in metric_aliases.items():
        if canonical in df.columns:
            continue

        # find first alias present in the DF
        found = None
        for a in aliases:
            if a in df.columns:
                found = a
                break

        if found is None:
            continue

        # if both exist somehow, coalesce instead of overwriting
        if canonical in df.columns:
            df[canonical] = df[canonical].combine_first(df[found])
            df = df.drop(columns=[found])
        else:
            df = df.rename(columns={found: canonical})

    return df


def build_report(config_path: str | Path, datasets_registry_path: str | Path | None = None) -> Path:
    config_path = Path(config_path)
    with config_path.open("r", encoding="utf-8") as f:
        cfg: Dict[str, Any] = yaml.safe_load(f)

    dataset_names: List[str] = cfg["datasets"]
    cache_dir = cfg.get("cache_dir", "outputs/cache")
    out_tables_dir = Path(cfg.get("output_tables_dir", "outputs/reports/tables"))

    epoch_col = cfg.get("epoch_col", "training_iteration")
    model_col = cfg.get("model_col", "model_name")
    output_pdf_dir = _resolve_output_pdf_dir(cfg)
    output_png_dir = Path(cfg.get("output_png_dir", "outputs/reports/PNG/"))
    png_dpi = int(cfg.get("png_dpi", 200))
    metric_aliases = cfg.get("metric_aliases", {}) or {}

    registry: Dict[str, Any] = {}
    if datasets_registry_path is not None:
        registry_path = Path(datasets_registry_path)
        if registry_path.exists():
            registry = load_dataset_registry(registry_path)
        else:
            log.warning("Datasets registry not found at %s; proceeding with cache only.", registry_path)

    dfs = []

    for ds in dataset_names:
        spec = registry.get(ds)
        root_dir = spec.root_dir if spec is not None else None
        df = load_dataset_long_df(dataset_name=ds, root_dir=root_dir, cache_dir=cache_dir)
        if not df.empty:
            df = _normalize_metric_columns(df, metric_aliases)
            dfs.append(df)

    if not dfs:
        raise RuntimeError("All datasets loaded empty; check paths and format.")

    df_all = pd.concat(dfs, axis=0, ignore_index=True)
    output_pdf_dir.mkdir(parents=True, exist_ok=True)
    used_pdf_stems: set[str] = set()

    for a in cfg.get("analyses", []):
        analysis_name = a["name"]
        runner = ANALYSIS_REGISTRY.get(analysis_name)
        if runner is None:
            raise ValueError(f"Unknown analysis: {analysis_name}. Available: {sorted(ANALYSIS_REGISTRY)}")

        output_png_dir_analysis = Path(output_png_dir / analysis_name)
        output_png_dir_analysis.mkdir(parents=True, exist_ok=True)

        out_tables_dir_analysis = Path(out_tables_dir / analysis_name)
        out_tables_dir_analysis.mkdir(parents=True, exist_ok=True)

        analysis_pdf_name = _build_analysis_pdf_stem(a, used_pdf_stems)
        analysis_pdf_path = output_pdf_dir / f"{analysis_pdf_name}.pdf"
        per_dataset_outputs_enabled = analysis_name in _PER_DATASET_OUTPUT_ANALYSES
        save_dataset_png_only = analysis_name in _DATASET_ONLY_PNG_ANALYSES

        with PdfPages(analysis_pdf_path) as pdf, ExitStack() as dataset_pdf_stack:
            dataset_pdf_pages: Dict[str, PdfPages] = {}
            if per_dataset_outputs_enabled:
                for dataset_name in dataset_names:
                    dataset_pdf_path = (
                        dataset_output_dir(output_pdf_dir / analysis_name, dataset_name)
                        / f"{analysis_pdf_name}.pdf"
                    )
                    dataset_pdf_path.parent.mkdir(parents=True, exist_ok=True)
                    dataset_pdf_pages[str(dataset_name)] = dataset_pdf_stack.enter_context(PdfPages(dataset_pdf_path))

            a_cfg = dict(a)
            a_cfg.setdefault("epoch_col", epoch_col)
            a_cfg.setdefault("source_epoch_col", epoch_col)
            a_cfg.setdefault("model_col", model_col)
            a_cfg["output_png_dir"] = output_png_dir_analysis
            a_cfg["output_tables_dir"] = out_tables_dir_analysis
            a_cfg["pdf_pages"] = pdf
            a_cfg["save_dataset_png_only"] = save_dataset_png_only
            a_cfg.setdefault("write_outputs_immediately", analysis_name in _STREAMING_ANALYSES)
            a_cfg["metric_aliases"] = metric_aliases
            a_cfg["dataset_names"] = list(dataset_names)
            if per_dataset_outputs_enabled:
                a_cfg["dataset_output_names"] = list(dataset_names)
                a_cfg["dataset_output_png_dir"] = output_png_dir_analysis
                a_cfg["dataset_output_tables_dir"] = out_tables_dir_analysis
                a_cfg["dataset_output_pdf_pages"] = dataset_pdf_pages

            log.info("Running analysis: %s -> %s", analysis_name, analysis_pdf_path)
            res = runner(df_all, a_cfg, registry)

            for item in res.figures:
                if not isinstance(item, dict) or len(item) != 1:
                    raise ValueError("Each figure entry must be a dict with exactly one {name: Figure}.")
                fig_name, fig = next(iter(item.items()))
                dataset_name = (
                    match_dataset_name_from_key(str(fig_name), dataset_names)
                    if per_dataset_outputs_enabled
                    else None
                )

                append_pdf_page(pdf, fig)
                if not save_dataset_png_only or dataset_name is None:
                    write_png(output_png_dir_analysis, fig, str(fig_name), dpi=png_dpi)
                    write_single_pdf(output_png_dir_analysis, fig, str(fig_name))
                if per_dataset_outputs_enabled:
                    if dataset_name is not None:
                        append_pdf_page(dataset_pdf_pages[dataset_name], fig)
                        ds_out_dir = dataset_output_dir(output_png_dir_analysis, dataset_name)
                        write_png(ds_out_dir, fig, str(fig_name), dpi=png_dpi)
                        write_single_pdf(ds_out_dir, fig, str(fig_name))

                plt.close(fig)
                log.info(f"Wrote figure: {fig_name}")

            for k, t in res.tables.items():
                if isinstance(t, pd.DataFrame) and not t.empty:
                    t.to_parquet(out_tables_dir_analysis / f"{k}.parquet", index=False)
                    log.info(f"Saved table to: {out_tables_dir_analysis}/{k}.parquet")
                    if per_dataset_outputs_enabled:
                        for dataset_name, subset in iter_dataset_table_views(str(k), t, dataset_names):
                            dataset_table_path = dataset_output_dir(out_tables_dir_analysis, dataset_name) / f"{k}.parquet"
                            dataset_table_path.parent.mkdir(parents=True, exist_ok=True)
                            subset.to_parquet(dataset_table_path, index=False)

    return output_pdf_dir
