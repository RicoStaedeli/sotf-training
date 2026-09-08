import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SOTF_SRC = PROJECT_ROOT / "sotf-selection" / "src"
METRIC_PREDICTION_SRC = PROJECT_ROOT / "metric-prediction" / "src"

sys.path.insert(0, str(SOTF_SRC))
sys.path.insert(1, str(METRIC_PREDICTION_SRC))

from pipeline.cli import main


if __name__ == "__main__":
    main()
