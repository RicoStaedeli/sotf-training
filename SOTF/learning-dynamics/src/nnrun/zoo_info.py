from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

_ZOO_INFO_PATH = Path(__file__).resolve().parents[2] / "configs" / "model_zoo_informations.json"


@lru_cache(maxsize=None)
def _load_plot_name_lookup() -> dict[str, str]:
    try:
        data = json.loads(_ZOO_INFO_PATH.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return {
        entry["zoo_name"]: entry["plot_name"]
        for entry in data
        if isinstance(entry, dict) and "zoo_name" in entry and "plot_name" in entry
    }


def zoo_plot_name(dataset_name: str) -> str:
    """Return the plot_name for a dataset_name (zoo_name), falling back to dataset_name."""
    return _load_plot_name_lookup().get(dataset_name, dataset_name)
