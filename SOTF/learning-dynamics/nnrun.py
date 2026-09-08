#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path


_THIS_DIR = Path(__file__).resolve().parent
_SRC_DIR = _THIS_DIR / "src"
_THIS_DIR_STR = str(_THIS_DIR)
_SRC_DIR_STR = str(_SRC_DIR)

# Ensure imports resolve against the package under src/ instead of this wrapper
# file, which is also named "nnrun.py".
sys.path = [_SRC_DIR_STR] + [
    entry
    for entry in sys.path
    if Path(entry or ".").resolve() != _THIS_DIR
]
sys.modules.pop("nnrun", None)

from nnrun.cli import main


if __name__ == "__main__":
    main()
