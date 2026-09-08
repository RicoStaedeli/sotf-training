from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd
from tqdm import tqdm

log = logging.getLogger(__name__)


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    """Read a JSONL file (one JSON object per line)."""
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as e:
                raise ValueError(f"Invalid JSON on {path} line {line_no}: {e}") from e
    return rows


def build_long_dataframe(root_dir: str | Path) -> pd.DataFrame:
    """
    Builds a dataframe with one row per JSON line, plus:
      - dataset_name, model_name inferred from path parts
      - file_path for traceability
      - config flattened into columns (sep='.')
    Assumes: /<dataset_name>/<model_name>/result.json
    """
    root = Path(root_dir).expanduser().resolve()
    root_exists = root.exists()

    log.info("Dataset root: %s | exists=%s", root, root_exists)

    # If root doesn't exist, return empty df with a consistent schema
    if not root_exists:
        log.warning("Root dir does not exist: %s", root)
        return pd.DataFrame(columns=["root_dir", "root_dir_exists"])

    records: List[Dict[str, Any]] = []

    result_files = []
    for subdir in root.iterdir():  # only direct children
        if subdir.is_dir():
            p = subdir / "result.json"  # root/subdir/result.json
            if p.is_file():
                result_files.append(p)
    log.info("Found %d result files", len(result_files))

    for result_path in tqdm(result_files, desc="Reading result.json files", unit="file"):
        model_name_raw = result_path.parent.name
        dataset_name = result_path.parent.parent.name
        model_name = model_name_raw.split("::", 1)[0]

        rows = read_jsonl(result_path)

        # OPTIONAL: show per-file line progress (comment out if too noisy/slow)
        for row in tqdm(rows, desc=f"{result_path.name}", unit="row", leave=False):
            row = dict(row)
            row["dataset_name"] = dataset_name
            row["model_name"] = model_name
            row["model_name_full"] = model_name_raw
            row["file_path"] = str(result_path)
            records.append(row)

    return pd.json_normalize(records, sep=".") if records else pd.DataFrame()


def load_dataset_long_df(
    *,
    dataset_name: str,
    root_dir: str | Path | None = None,
    cache_dir: str | Path = "outputs/cache",
    force_rebuild: bool = False,
) -> pd.DataFrame:
    """
    Load one dataset into a long dataframe.
    Uses parquet cache at: <cache_dir>/long__<dataset_name>.parquet
    root_dir is only required when the cache does not yet exist.
    """
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = cache_dir / f"long__{dataset_name}.parquet"

    if cache_path.exists() and not force_rebuild:
        log.info("Loading cached parquet: %s", cache_path)
        df = pd.read_parquet(cache_path)
        # Ensure dataset_name is correct even if path-derived name differs
        if "dataset_name" in df.columns:
            df["dataset_name"] = dataset_name
        else:
            df.insert(0, "dataset_name", dataset_name)
        return df

    if root_dir is None:
        log.warning("No cache found for '%s' and root_dir was not provided; skipping.", dataset_name)
        return pd.DataFrame()

    log.info("Building long dataframe from: %s", root_dir)
    df = build_long_dataframe(root_dir)

    if df.empty:
        log.warning("Dataset %s produced empty dataframe.", dataset_name)
        return df

    if "dataset_name" in df.columns:
        df["dataset_name"] = dataset_name
    else:
        df.insert(0, "dataset_name", dataset_name)

    log.info("Writing parquet cache: %s", cache_path)
    df.to_parquet(cache_path, index=False)
    return df