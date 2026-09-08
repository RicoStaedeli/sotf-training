from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Any

import yaml


@dataclass(frozen=True)
class DatasetSpec:
    name: str
    root_dir: Path
    dstype: str


def load_dataset_registry(path: str | Path) -> Dict[str, DatasetSpec]:
    path = Path(path)
    with path.open("r", encoding="utf-8") as f:
        cfg: Dict[str, Any] = yaml.safe_load(f)

    datasets = cfg.get("datasets", {})
    out: Dict[str, DatasetSpec] = {}
    for name, spec in datasets.items():
        root_dir = Path(spec["root_dir"])
        dstype = spec["dstype"]
        out[name] = DatasetSpec(name=name, root_dir=root_dir, dstype=dstype)
    return out