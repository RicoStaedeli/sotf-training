from __future__ import annotations

from pathlib import Path

from matplotlib.figure import Figure


def write_png(out_dir: str | Path, fig: Figure, name: str, dpi: int = 200) -> Path:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{name}.png"
    fig.savefig(path, dpi=dpi, bbox_inches="tight", facecolor="white")
    return path