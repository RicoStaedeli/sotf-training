"""Compatibility shim for relocated script modules."""

from __future__ import annotations

from pathlib import Path

_PACKAGE_DIR = Path(__file__).resolve().parent
_RELOCATED_DIR = _PACKAGE_DIR.parent.parent / "scripts"
_RELOCATED_DIR_STR = str(_RELOCATED_DIR)

if _RELOCATED_DIR_STR not in __path__:
    __path__.append(_RELOCATED_DIR_STR)
