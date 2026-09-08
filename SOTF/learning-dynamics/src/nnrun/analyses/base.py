from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Protocol

import pandas as pd
from matplotlib.figure import Figure


@dataclass
class AnalysisResult:
    tables: Dict[str, pd.DataFrame]
    figures: List[Dict[str, Figure]]


class Analysis(Protocol):
    name: str

    def run(self, df: pd.DataFrame, config: Dict[str, Any]) -> AnalysisResult: ...