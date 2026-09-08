from __future__ import annotations

import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
SOTF_SRC = PROJECT_ROOT / "sotf-selection" / "src"
METRIC_PREDICTION_SRC = PROJECT_ROOT / "metric-prediction" / "src"

if str(SOTF_SRC) not in sys.path:
    sys.path.insert(0, str(SOTF_SRC))
if str(METRIC_PREDICTION_SRC) not in sys.path:
    sys.path.insert(1, str(METRIC_PREDICTION_SRC))

from evaluation.cli import main


if __name__ == "__main__":
    raise SystemExit(main())
