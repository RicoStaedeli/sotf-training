from __future__ import annotations

import argparse
import json
import os
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Sequence

import matplotlib
import matplotlib.image as mpimg
from omegaconf import OmegaConf

matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["font.family"] = "Times New Roman"

from evaluation import dataset_overview
from evaluation import forecasting
from evaluation import prediction


DATE_DIR_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")
FULL_TIMESTAMP_FORMATS = ("%Y-%m-%d_%H-%M-%S-%f", "%Y-%m-%d-%H-%M-%S-%f")
EVALUATION_OUTPUT_ROOT = Path("metric-prediction") / "outputs" / "evaluation"
README_HEADER = [
    "Evaluation ID",
    "Created",
    "Type",
    "Name",
    "Output",
    "Batch Results",
    "Config",
]


@dataclass(frozen=True)
class EvaluationContext:
    evaluation_id: str
    created_at: str
    evaluation_type: str
    name: str
    output_root: Path
    output_dir: Path
    batch_roots: list[Path]
    config_path: Path | None = None


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse args.

    Args:
        argv (Sequence[str] | None): Sequence[str] | None value used by the function.

    Returns:
        argparse.Namespace: Result of the function.
    """
    parser = argparse.ArgumentParser(
        description=(
            "Unified entry point for metric-prediction evaluations. Pass one or more "
            "batch/output directories to run the normal evaluator across all of them."
        )
    )
    parser.add_argument(
        "targets",
        nargs="*",
        type=Path,
        help=(
            "One or more batch/output directories to evaluate together. "
            "Optional for --kind dataset-overview (defaults to metric-prediction/outputs/forecasting)."
        ),
    )
    parser.add_argument(
        "--kind",
        choices=["auto", "prediction", "forecasting", "dataset-overview"],
        default="auto",
        help="Batch-local evaluator to use. Defaults to auto-detection.",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default=None,
        help="Dataset name filter for dataset-overview (e.g. ds14). Auto-detected when only one dataset is present.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory where evaluation artifacts are written.",
    )
    parser.add_argument(
        "--batch-file",
        dest="batch_files",
        type=Path,
        action="append",
        default=None,
        help="Batch config passed through to the prediction evaluator.",
    )
    parser.add_argument(
        "--extra-outputs-root",
        dest="extra_outputs_roots",
        type=Path,
        action="append",
        default=None,
        help="Additional outputs root passed through to the prediction evaluator.",
    )
    parser.add_argument(
        "--top-n",
        type=int,
        default=20,
        help="Top-N value passed through to the prediction evaluator.",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    """Main.

    Args:
        argv (Sequence[str] | None): Sequence[str] | None value used by the function.

    Returns:
        int: Result of the function.
    """
    args = parse_args(argv)
    targets = [target.resolve() for target in args.targets]
    kind = args.kind if args.kind != "auto" else infer_shared_evaluation_kind(targets)

    if kind == "dataset-overview":
        batch_roots = [resolve_batch_root_for_evaluation(t) for t in targets]
        context = create_evaluation_context(
            evaluation_type=kind,
            name=evaluation_name_from_batch_roots(batch_roots) if batch_roots else "forecasting_all",
            batch_roots=batch_roots,
            config_path=single_or_none(discover_candidate_batch_files(targets)),
            requested_output_dir=args.output_dir,
        )
        backend_args = ["--output-dir", str(context.output_dir)]
        if targets:
            backend_args = [str(targets[0])] + backend_args
            for extra_target in targets[1:]:
                backend_args.extend(["--extra-results-root", str(extra_target)])
        if args.dataset:
            backend_args.extend(["--dataset", args.dataset])
        exit_code = dataset_overview.main(backend_args)
        finalize_evaluation(context)
        return exit_code

    if not targets:
        raise SystemExit("At least one target directory is required.")

    batch_roots = [resolve_batch_root_for_evaluation(target) for target in targets]
    context = create_evaluation_context(
        evaluation_type=kind,
        name=evaluation_name_from_batch_roots(batch_roots),
        batch_roots=batch_roots,
        config_path=single_or_none(discover_candidate_batch_files(targets)),
        requested_output_dir=args.output_dir,
    )

    if kind == "forecasting":
        backend_args = [str(targets[0]), "--output-dir", str(context.output_dir)]
        for extra_target in targets[1:]:
            backend_args.extend(["--extra-results-root", str(extra_target)])
        exit_code = forecasting.main(backend_args)
        finalize_evaluation(context)
        return exit_code

    backend_args = ["--outputs-root", str(targets[0]), "--top-n", str(args.top_n)]
    backend_args.extend(["--output-dir", str(context.output_dir)])
    batch_files = [path.resolve() for path in (args.batch_files or discover_candidate_batch_files(targets))]
    for batch_file in batch_files:
        backend_args.extend(["--batch-file", str(batch_file)])
    for extra_root in [*targets[1:], *(args.extra_outputs_roots or [])]:
        backend_args.extend(["--extra-outputs-root", str(extra_root)])
    exit_code = prediction.main(backend_args)
    finalize_evaluation(context)
    return exit_code


def infer_batch_evaluation_kind(target: Path) -> str:
    """Infer batch evaluation kind.

    Args:
        target (Path): Path value used by the function.

    Returns:
        str: Resolved string value.
    """
    configured_kind = evaluation_kind_from_batch_config(target)
    if configured_kind in {"prediction", "forecasting"}:
        return configured_kind

    target_text = str(target).lower()
    if "forecast" in target_text or has_forecasting_artifacts(target):
        return "forecasting"
    return "prediction"


def infer_shared_evaluation_kind(targets: list[Path]) -> str:
    """Infer shared evaluation kind.

    Args:
        targets (list[Path]): list[Path] value used by the function.

    Returns:
        str: Resolved string value.
    """
    resolved_kinds = [(target, infer_batch_evaluation_kind(target)) for target in targets]
    unique_kinds = {kind for _, kind in resolved_kinds}
    if len(unique_kinds) == 1:
        return resolved_kinds[0][1]

    details = ", ".join(f"{target.name}: {kind}" for target, kind in resolved_kinds)
    raise SystemExit(
        "All provided batch/output directories must resolve to the same evaluation kind. "
        f"Found: {details}"
    )


def evaluation_kind_from_batch_config(target: Path) -> str | None:
    """Evaluation kind from batch config.

    Args:
        target (Path): Path value used by the function.

    Returns:
        str | None: Result of the function.
    """
    for batch_file in candidate_batch_files(target):
        try:
            config = load_yaml_mapping(batch_file)
        except Exception:
            continue
        evaluation = config.get("evaluation")
        if isinstance(evaluation, dict):
            kind = evaluation.get("kind") or evaluation.get("recipe")
            if kind is not None:
                return normalize_kind(str(kind))
        mode = config.get("common_overrides", {}).get("mode") if isinstance(config.get("common_overrides"), dict) else None
        if isinstance(mode, str):
            return normalize_kind(mode)
    return None


def candidate_batch_files(target: Path) -> list[Path]:
    """Candidate batch files.

    Args:
        target (Path): Path value used by the function.

    Returns:
        list[Path]: Result of the function.
    """
    if target.is_file() and target.suffix.lower() in {".yaml", ".yml"}:
        return [target]
    roots = [target, target / "batch_files"]
    return sorted(
        path
        for root in roots
        if root.is_dir()
        for pattern in ("*.yaml", "*.yml")
        for path in root.glob(pattern)
    )


def discover_candidate_batch_files(targets: list[Path]) -> list[Path]:
    """Discover candidate batch files.

    Args:
        targets (list[Path]): list[Path] value used by the function.

    Returns:
        list[Path]: Result of the function.
    """
    return unique_paths([path.resolve() for target in targets for path in candidate_batch_files(target)])


def normalize_kind(value: str) -> str:
    """Normalize kind.

    Args:
        value (str): Value to process.

    Returns:
        str: Resolved string value.
    """
    normalized = value.lower().replace("-", "_")
    if normalized in {"forecast", "forecasting", "latent_forecast"}:
        return "forecasting"
    if normalized in {"latent", "latent_sweep", "latent_mlp", "latent_prediction"}:
        return "prediction"
    if normalized in {"prediction", "static_prediction"}:
        return "prediction"
    if normalized in {"dataset_overview", "ds_overview"}:
        return "dataset-overview"
    return normalized


def has_forecasting_artifacts(target: Path) -> bool:
    """Has forecasting artifacts.

    Args:
        target (Path): Path value used by the function.

    Returns:
        bool: Result of the function.
    """
    search_root = target / "results" if (target / "results").is_dir() else target
    for metrics_path in search_root.rglob("metrics.json"):
        artifacts_dir = metrics_path.parent
        if (artifacts_dir / "validation_predictions.parquet").exists():
            return True
        config = load_json_if_exists(artifacts_dir / "resolved_config.json")
        if nested_get(config, "sequence", "target_mode") is not None:
            return True
    return False


def create_evaluation_context(
    *,
    evaluation_type: str,
    name: str,
    batch_roots: list[Path],
    config_path: Path | None,
    requested_output_dir: Path | None,
) -> EvaluationContext:
    """Create evaluation context.

    Args:
        evaluation_type (str): str value used by the function.
        name (str): Name used by the function.
        batch_roots (list[Path]): list[Path] value used by the function.
        config_path (Path | None): Path | None value used by the function.
        requested_output_dir (Path | None): Path | None value used by the function.

    Returns:
        EvaluationContext: Result of the function.
    """
    evaluation_id = generate_evaluation_id()
    created_at = datetime.now().isoformat(timespec="seconds")
    output_root = (EVALUATION_OUTPUT_ROOT / evaluation_type).resolve()
    default_dir = output_root / f"{sanitize_file_name(name)}_{evaluation_id}"
    output_dir = (requested_output_dir or default_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    output_root.mkdir(parents=True, exist_ok=True)
    return EvaluationContext(
        evaluation_id=evaluation_id,
        created_at=created_at,
        evaluation_type=evaluation_type,
        name=name,
        output_root=output_root,
        output_dir=output_dir,
        batch_roots=batch_roots,
        config_path=config_path,
    )


def finalize_evaluation(context: EvaluationContext) -> None:
    """Finalize evaluation.

    Args:
        context (EvaluationContext): EvaluationContext value used by the function.

    Returns:
        None: None.
    """
    stamp_generated_plots(context.output_dir, context.evaluation_id)
    write_evaluation_metadata(context)
    append_evaluation_index(context)


def generate_evaluation_id() -> str:
    """Generate evaluation id.

    Returns:
        str: Resolved string value.
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    return f"eval_{timestamp}_{os.getpid()}"


def stamp_generated_plots(output_dir: Path, evaluation_id: str) -> None:
    """Stamp generated plots.

    Args:
        output_dir (Path): Output directory for artifacts.
        evaluation_id (str): str value used by the function.

    Returns:
        None: None.
    """
    for image_path in sorted(output_dir.rglob("*.png")):
        stamp_plot(image_path, evaluation_id)


def stamp_plot(image_path: Path, evaluation_id: str) -> None:
    """Stamp plot.

    Args:
        image_path (Path): Path value used by the function.
        evaluation_id (str): str value used by the function.

    Returns:
        None: None.
    """
    try:
        image = mpimg.imread(image_path)
    except Exception as exc:  # pragma: no cover - defensive reporting for local artifacts
        print(f"Skipping plot stamp for {image_path}: {exc}")
        return

    height, width = image.shape[:2]
    dpi = 150
    label_strip_px = 30
    bottom_frac = label_strip_px / (height + label_strip_px)
    fig, ax = plt.subplots(figsize=(width / dpi, (height + label_strip_px) / dpi), dpi=dpi)
    ax.imshow(image, aspect="auto", interpolation="nearest")
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
    """Write evaluation metadata.

    Args:
        context (EvaluationContext): EvaluationContext value used by the function.

    Returns:
        None: None.
    """
    payload = {
        "evaluation_id": context.evaluation_id,
        "created_at": context.created_at,
        "type": context.evaluation_type,
        "name": context.name,
        "output_dir": str(context.output_dir),
        "batch_roots": [str(path) for path in context.batch_roots],
        "config_path": None if context.config_path is None else str(context.config_path),
    }
    context.output_dir.joinpath("evaluation_metadata.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True),
        encoding="utf-8",
    )


def append_evaluation_index(context: EvaluationContext) -> None:
    """Append evaluation index.

    Args:
        context (EvaluationContext): EvaluationContext value used by the function.

    Returns:
        None: None.
    """
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
        context.evaluation_type,
        context.name,
        markdown_link(context.output_dir.name, context.output_dir, root=context.output_root),
        "<br>".join(markdown_link(path.name, path, root=context.output_root) for path in context.batch_roots),
        "" if context.config_path is None else markdown_link(context.config_path.name, context.config_path, root=context.output_root),
    ]
    with readme_path.open("a", encoding="utf-8") as handle:
        handle.write("| " + " | ".join(escape_markdown_table_cell(value) for value in row) + " |\n")


def markdown_link(label: str, target: Path, *, root: Path) -> str:
    """Markdown link.

    Args:
        label (str): str value used by the function.
        target (Path): Path value used by the function.
        root (Path): Path value used by the function.

    Returns:
        str: Resolved string value.
    """
    try:
        relative = target.resolve().relative_to(root.resolve())
    except ValueError:
        relative = Path(os.path.relpath(target.resolve(), root.resolve()))
    return f"[{label}]({relative.as_posix()})"


def escape_markdown_table_cell(value: Any) -> str:
    """Escape markdown table cell.

    Args:
        value (Any): Value to process.

    Returns:
        str: Resolved string value.
    """
    return str(value).replace("|", "\\|").replace("\n", " ")


def resolve_batch_root_for_evaluation(target: Path) -> Path:
    """Resolve batch root for evaluation.

    Args:
        target (Path): Path value used by the function.

    Returns:
        Path: Resolved path.
    """
    resolved = target.resolve()
    if resolved.name == "results":
        return resolved.parent
    for candidate in (resolved, *resolved.parents):
        if (candidate / "batch_files").is_dir():
            return candidate
        if candidate.parent.name == "batch_experiments":
            return candidate
    return resolved


def unique_paths(paths: list[Path]) -> list[Path]:
    """Unique paths.

    Args:
        paths (list[Path]): Paths to process.

    Returns:
        list[Path]: Result of the function.
    """
    unique: list[Path] = []
    seen: set[str] = set()
    for path in paths:
        key = str(path)
        if key in seen:
            continue
        seen.add(key)
        unique.append(path)
    return unique


def evaluation_name_from_batch_roots(batch_roots: list[Path]) -> str:
    """Evaluation name from batch roots.

    Args:
        batch_roots (list[Path]): list[Path] value used by the function.

    Returns:
        str: Resolved string value.
    """
    if not batch_roots:
        return "evaluation"
    return "_".join(sanitize_file_name(path.name) for path in batch_roots)


def single_or_none(paths: list[Path]) -> Path | None:
    """Single or none.

    Args:
        paths (list[Path]): Paths to process.

    Returns:
        Path | None: Result of the function.
    """
    if len(paths) == 1:
        return paths[0]
    return None


def load_yaml_mapping(path: Path) -> dict[str, Any]:
    """Load yaml mapping.

    Args:
        path (Path): Path to process.

    Returns:
        dict[str, Any]: Result of the function.
    """
    raw = OmegaConf.to_container(OmegaConf.load(path), resolve=True)
    if not isinstance(raw, dict):
        raise ValueError(f"{path} does not contain a mapping.")
    return raw


def load_json_if_exists(path: Path) -> dict[str, Any]:
    """Load json if exists.

    Args:
        path (Path): Path to process.

    Returns:
        dict[str, Any]: Result of the function.
    """
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def nested_get(mapping: dict[str, Any], *keys: str, default: Any = None) -> Any:
    """Nested get.

    Args:
        mapping (dict[str, Any]): dict[str, Any] value used by the function.
        *keys (str): str value used by the function.
        default (Any): Value used by the function.

    Returns:
        Any: Result of the function.
    """
    current: Any = mapping
    for key in keys:
        if not isinstance(current, dict) or key not in current:
            return default
        current = current[key]
    return current


def sanitize_file_name(value: str) -> str:
    """Sanitize file name.

    Args:
        value (str): Value to process.

    Returns:
        str: Resolved string value.
    """
    sanitized = re.sub(r"[^A-Za-z0-9_.-]+", "_", value.strip())
    return sanitized or "evaluation"


if __name__ == "__main__":
    raise SystemExit(main())
