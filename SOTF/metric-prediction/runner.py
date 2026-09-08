from pathlib import Path
import sys

# Prefer the source tree package over this wrapper module when run as a script.
SRC_DIR = Path(__file__).resolve().parent / "src"
sys.path.insert(0, str(SRC_DIR))

from runner.orchestrator import main


if __name__ == "__main__":
    main()
