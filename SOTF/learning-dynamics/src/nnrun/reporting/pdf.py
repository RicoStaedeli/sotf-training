from __future__ import annotations

from pathlib import Path
from typing import Iterable

from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.figure import Figure


def write_pdf(path: str | Path, figures: Iterable[Figure]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with PdfPages(path) as pdf:
        for fig in figures:
            pdf.savefig(fig, bbox_inches="tight")

def append_pdf_page(pdf: PdfPages, fig: Figure) -> None:
    pdf.savefig(fig, bbox_inches="tight")


def write_single_pdf(out_dir: str | Path, fig: Figure, name: str) -> Path:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{name}.pdf"
    with PdfPages(path) as pdf:
        pdf.savefig(fig, bbox_inches="tight")
    return path