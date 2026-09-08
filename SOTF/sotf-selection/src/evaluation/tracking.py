from __future__ import annotations

import json
import os
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import matplotlib
import matplotlib.image as mpimg

matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["font.family"] = "Times New Roman"

from .shared import sanitize_file_stem


README_HEADER = [
    "Evaluation ID",
    "Created",
    "Name",
    "Output",
    "Sources",
]


@dataclass(frozen=True)
class EvaluationContext:
    evaluation_id: str
    created_at: str
    name: str
    output_root: Path
    output_dir: Path
    source_roots: list[Path]
    filters: dict[str, Any]


def create_evaluation_context(
    *,
    name: str,
    source_roots: list[Path],
    requested_output_dir: Path | None,
    filters: dict[str, Any],
) -> EvaluationContext:
    evaluation_id = generate_evaluation_id()
    created_at = datetime.now().isoformat(timespec="seconds")
    default_root = resolve_default_evaluation_root(source_roots[0])
    output_root = (requested_output_dir.parent if requested_output_dir is not None else default_root).resolve()
    default_dir = output_root / f"{sanitize_file_stem(name)}_{evaluation_id}"
    output_dir = (requested_output_dir or default_dir).resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    return EvaluationContext(
        evaluation_id=evaluation_id,
        created_at=created_at,
        name=name,
        output_root=output_root,
        output_dir=output_dir,
        source_roots=source_roots,
        filters=filters,
    )


def resolve_default_evaluation_root(source_root: Path) -> Path:
    resolved = source_root.resolve()
    for candidate in (resolved, *resolved.parents):
        if candidate.name == "outputs":
            return candidate / "evaluation"
    return resolved / "evaluation"


def finalize_evaluation(context: EvaluationContext) -> None:
    stamp_generated_plots(context.output_dir, context.evaluation_id)
    write_evaluation_metadata(context)
    append_evaluation_index(context)


def generate_evaluation_id() -> str:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    return f"eval_{timestamp}_{os.getpid()}"


def stamp_generated_plots(output_dir: Path, evaluation_id: str) -> None:
    for image_path in sorted(output_dir.rglob("*.png")):
        stamp_plot(image_path, evaluation_id)


def stamp_plot(image_path: Path, evaluation_id: str) -> None:
    try:
        image = mpimg.imread(image_path)
    except Exception as exc:  # pragma: no cover - defensive for local artifacts
        print(f"Skipping plot stamp for {image_path}: {exc}")
        return

    height, width = image.shape[:2]
    dpi = 100
    label_strip_px = 24
    bottom_frac = label_strip_px / (height + label_strip_px)
    fig, ax = plt.subplots(figsize=(width / dpi, (height + label_strip_px) / dpi), dpi=dpi)
    ax.imshow(image)
    ax.axis("off")
    font_size = max(8, min(14, width / 120))
    fig.subplots_adjust(left=0, right=1, bottom=bottom_frac, top=1)
    fig.text(
        0.995,
        bottom_frac / 2,
        f"Evaluation ID: {evaluation_id}",
        ha="right",
        va="center",
        fontsize=font_size,
        color="black",
        bbox={"facecolor": "white", "edgecolor": "black", "alpha": 0.82, "pad": 3},
    )
    fig.savefig(image_path, dpi=dpi)
    plt.close(fig)


def write_evaluation_metadata(context: EvaluationContext) -> None:
    payload = {
        "evaluation_id": context.evaluation_id,
        "created_at": context.created_at,
        "name": context.name,
        "output_dir": str(context.output_dir),
        "output_root": str(context.output_root),
        "source_roots": [str(path) for path in context.source_roots],
        "filters": context.filters,
    }
    context.output_dir.joinpath("evaluation_metadata.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True),
        encoding="utf-8",
    )


def append_evaluation_index(context: EvaluationContext) -> None:
    readme_path = context.output_root / "README.md"
    if not readme_path.exists():
        readme_path.write_text(
            "\n".join(
                [
                    "# Evaluation Runs",
                    "",
                    "| " + " | ".join(README_HEADER) + " |",
                    "| " + " | ".join("---" for _ in README_HEADER) + " |",
                ]
            )
            + "\n",
            encoding="utf-8",
        )

    row = [
        f"`{context.evaluation_id}`",
        context.created_at,
        context.name,
        markdown_link(context.output_dir.name, context.output_dir, root=context.output_root),
        "<br>".join(markdown_link(path.name, path, root=context.output_root) for path in context.source_roots),
    ]
    with readme_path.open("a", encoding="utf-8") as handle:
        handle.write("| " + " | ".join(escape_markdown_table_cell(value) for value in row) + " |\n")


def markdown_link(label: str, target: Path, *, root: Path) -> str:
    try:
        relative = target.resolve().relative_to(root.resolve())
    except ValueError:
        relative = Path(os.path.relpath(target.resolve(), root.resolve()))
    return f"[{label}]({relative.as_posix()})"


def escape_markdown_table_cell(value: Any) -> str:
    return str(value).replace("|", "\\|").replace("\n", " ")
