from __future__ import annotations

import logging
from pathlib import Path
from typing import Iterable, Optional, Sequence

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.figure import Figure

from nnrun.reporting.pdf import append_pdf_page
from nnrun.reporting.png import write_png

log = logging.getLogger(__name__)


def _dataset_path_component(dataset_name: str) -> str:
    return str(dataset_name).replace("/", "_").replace("\\", "_")


def match_dataset_name_from_key(key: str, dataset_names: Sequence[str]) -> Optional[str]:
    key = str(key)
    for dataset_name in sorted((str(name) for name in dataset_names), key=len, reverse=True):
        if key == dataset_name or key.startswith(f"{dataset_name}_"):
            return dataset_name
    return None


def iter_dataset_table_views(
    key: str,
    value: pd.DataFrame,
    dataset_names: Sequence[str],
) -> Iterable[tuple[str, pd.DataFrame]]:
    dataset_name = match_dataset_name_from_key(key, dataset_names)
    if dataset_name is not None:
        yield dataset_name, value
        return

    if "dataset_name" not in value.columns:
        return

    dataset_series = value["dataset_name"]
    dataset_values = dataset_series.dropna().astype(str)
    if dataset_values.empty:
        return

    dataset_name_set = {str(name) for name in dataset_names}
    dataset_series_str = dataset_series.astype("string")
    for dataset_name in sorted(dataset_name_set.intersection(dataset_values.unique().tolist())):
        subset = value[dataset_series_str.eq(dataset_name).fillna(False)]
        if not subset.empty:
            yield dataset_name, subset


def dataset_output_dir(root_dir: str | Path, dataset_name: str) -> Path:
    return Path(root_dir) / _dataset_path_component(dataset_name)


class TableStore(dict[str, pd.DataFrame]):
    def __init__(
        self,
        *,
        out_dir: Optional[str | Path] = None,
        dataset_out_dir: Optional[str | Path] = None,
        dataset_names: Optional[Sequence[str]] = None,
        keep_in_memory: bool = True,
    ) -> None:
        super().__init__()
        self.out_dir = Path(out_dir) if out_dir is not None else None
        self.dataset_out_dir = Path(dataset_out_dir) if dataset_out_dir is not None else None
        self.dataset_names = [str(name) for name in (dataset_names or [])]
        self.keep_in_memory = keep_in_memory
        if self.out_dir is not None:
            self.out_dir.mkdir(parents=True, exist_ok=True)
        if self.dataset_out_dir is not None:
            self.dataset_out_dir.mkdir(parents=True, exist_ok=True)

    def __setitem__(self, key: str, value: pd.DataFrame) -> None:
        if self.out_dir is not None and isinstance(value, pd.DataFrame) and not value.empty:
            path = self.out_dir / f"{key}.parquet"
            value.to_parquet(path, index=False)
            # log.info("Saved table to: %s", path)
        if (
            self.dataset_out_dir is not None
            and self.dataset_names
            and isinstance(value, pd.DataFrame)
            and not value.empty
        ):
            for dataset_name, subset in iter_dataset_table_views(key, value, self.dataset_names):
                path = dataset_output_dir(self.dataset_out_dir, dataset_name) / f"{key}.parquet"
                path.parent.mkdir(parents=True, exist_ok=True)
                subset.to_parquet(path, index=False)

        if self.keep_in_memory:
            super().__setitem__(key, value)


class FigureStore(list[dict[str, Figure]]):
    def __init__(
        self,
        *,
        out_dir: Optional[str | Path] = None,
        pdf_pages: Optional[PdfPages] = None,
        dataset_out_dir: Optional[str | Path] = None,
        dataset_pdf_pages: Optional[dict[str, PdfPages]] = None,
        dataset_names: Optional[Sequence[str]] = None,
        write_root_outputs: bool = True,
        dpi: int = 200,
        keep_in_memory: bool = True,
        close_after_write: bool = True,
    ) -> None:
        super().__init__()
        self.out_dir = Path(out_dir) if out_dir is not None else None
        self.pdf_pages = pdf_pages
        self.dataset_out_dir = Path(dataset_out_dir) if dataset_out_dir is not None else None
        self.dataset_pdf_pages = {str(k): v for k, v in (dataset_pdf_pages or {}).items()}
        self.dataset_names = [str(name) for name in (dataset_names or [])]
        self.write_root_outputs = bool(write_root_outputs)
        self.dpi = dpi
        self.keep_in_memory = keep_in_memory
        self.close_after_write = close_after_write
        if self.out_dir is not None:
            self.out_dir.mkdir(parents=True, exist_ok=True)
        if self.dataset_out_dir is not None:
            self.dataset_out_dir.mkdir(parents=True, exist_ok=True)

    def append(self, item: dict[str, Figure]) -> None:
        if not isinstance(item, dict) or len(item) != 1:
            raise ValueError("Each figure entry must be a dict with exactly one {name: Figure}.")

        fig_name, fig = next(iter(item.items()))
        dataset_name = match_dataset_name_from_key(str(fig_name), self.dataset_names)
        write_root_outputs = self.write_root_outputs or dataset_name is None

        if write_root_outputs and self.pdf_pages is not None:
            append_pdf_page(self.pdf_pages, fig)
        if write_root_outputs and self.out_dir is not None:
            write_png(self.out_dir, fig, str(fig_name), dpi=self.dpi)
        if dataset_name is not None:
            dataset_pdf = self.dataset_pdf_pages.get(dataset_name)
            if dataset_pdf is not None:
                append_pdf_page(dataset_pdf, fig)
            if self.dataset_out_dir is not None:
                write_png(dataset_output_dir(self.dataset_out_dir, dataset_name), fig, str(fig_name), dpi=self.dpi)
        #if self.pdf_pages is not None or self.out_dir is not None:
            # log.info("Wrote figure: %s", fig_name)

        if self.keep_in_memory:
            super().append(item)
            return

        if self.close_after_write:
            plt.close(fig)


def make_result_stores(
    config: dict,
    *,
    default_png_dpi: int = 200,
) -> tuple[TableStore, FigureStore]:
    direct_write = bool(config.get("write_outputs_immediately", False)) or any(
        config.get(key) is not None
        for key in ("output_tables_dir", "output_png_dir", "pdf_pages")
    )

    return (
        TableStore(
            out_dir=config.get("output_tables_dir") if direct_write else None,
            dataset_out_dir=config.get("dataset_output_tables_dir") if direct_write else None,
            dataset_names=config.get("dataset_output_names", []),
            keep_in_memory=not direct_write,
        ),
        FigureStore(
            out_dir=config.get("output_png_dir") if direct_write else None,
            pdf_pages=config.get("pdf_pages") if direct_write else None,
            dataset_out_dir=config.get("dataset_output_png_dir") if direct_write else None,
            dataset_pdf_pages=config.get("dataset_output_pdf_pages", {}),
            dataset_names=config.get("dataset_output_names", []),
            write_root_outputs=not bool(config.get("save_dataset_png_only", False)),
            dpi=int(config.get("png_dpi", default_png_dpi)),
            keep_in_memory=not direct_write,
            close_after_write=direct_write,
        ),
    )
