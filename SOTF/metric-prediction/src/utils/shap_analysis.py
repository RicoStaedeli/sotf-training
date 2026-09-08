from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from utils.artifacts import write_frame

log = logging.getLogger(__name__)


def write_shap_feature_contributions(
    model: Any,
    X: pd.DataFrame,
    artifacts_dir: Path,
    *,
    max_samples: int,
    max_display: int,
    random_seed: int,
) -> None:
    """Write shap feature contributions.

    Args:
        model (Any): Model to process.
        X (pd.DataFrame): pd.DataFrame value used by the function.
        artifacts_dir (Path): Directory for artifacts.
        max_samples (int): int value used by the function.
        max_display (int): int value used by the function.
        random_seed (int): Random seed used by the function.

    Returns:
        None: None.
    """
    if X.empty:
        log.warning("Skipping SHAP artifact generation because the validation feature matrix is empty")
        return

    try:
        import shap
    except Exception as exc:
        raise RuntimeError(
            "Failed to import SHAP. Install project dependencies again so the `shap` package is available."
        ) from exc

    shap_input = _sample_frame(X, max_samples=max_samples, random_seed=random_seed)
    log.info("Generating SHAP feature contributions from %d validation rows", len(shap_input))

    explainer = _build_shap_explainer(shap, model=model, shap_input=shap_input)
    explanation = _compute_explanation(shap, explainer=explainer, shap_input=shap_input)
    shap_matrix = _coerce_shap_matrix(explanation.values)
    if shap_matrix.shape != shap_input.shape:
        raise ValueError(
            "Unexpected SHAP output shape. "
            f"Expected {shap_input.shape}, received {shap_matrix.shape}."
        )

    contribution_frame = pd.DataFrame(
        {
            "feature": list(shap_input.columns),
            "mean_abs_shap": np.abs(shap_matrix).mean(axis=0),
            "mean_shap": shap_matrix.mean(axis=0),
            "max_abs_shap": np.abs(shap_matrix).max(axis=0),
        }
    ).sort_values("mean_abs_shap", ascending=False, ignore_index=True)

    write_frame(artifacts_dir / "shap_feature_contributions.parquet", contribution_frame)
    _write_summary_plot(
        contribution_frame=contribution_frame,
        output_path=artifacts_dir / "shap_feature_contributions.png",
        max_display=max_display,
    )
    _write_beeswarm_plot(
        explanation=explanation,
        output_path=artifacts_dir / "shap_beeswarm.png",
        max_display=max_display,
    )
    _write_heatmap_plot(
        explanation=explanation,
        output_path=artifacts_dir / "shap_heatmap.png",
        max_display=max_display,
    )


def _sample_frame(frame: pd.DataFrame, *, max_samples: int, random_seed: int) -> pd.DataFrame:
    """Sample frame.

    Args:
        frame (pd.DataFrame): Data frame to process.
        max_samples (int): int value used by the function.
        random_seed (int): Random seed used by the function.

    Returns:
        pd.DataFrame: Computed data frame.
    """
    if max_samples <= 0 or len(frame) <= max_samples:
        return frame
    return frame.sample(n=max_samples, random_state=random_seed)


def _build_shap_explainer(shap: Any, *, model: Any, shap_input: pd.DataFrame) -> Any:
    """Build shap explainer.

    Args:
        shap (Any): Value used by the function.
        model (Any): Model to process.
        shap_input (pd.DataFrame): pd.DataFrame value used by the function.

    Returns:
        Any: Result of the function.
    """
    shap_model = _resolve_shap_model(model)
    try:
        return shap.Explainer(shap_model, shap_input)
    except TypeError:
        log.info("Falling back to prediction-function SHAP explainer for wrapped model %s", type(model).__name__)
        return shap.Explainer(_predict_callable(model), shap_input)


def _compute_explanation(shap: Any, *, explainer: Any, shap_input: pd.DataFrame) -> Any:
    """Compute explanation.

    Args:
        shap (Any): Value used by the function.
        explainer (Any): Value used by the function.
        shap_input (pd.DataFrame): pd.DataFrame value used by the function.

    Returns:
        Any: Result of the function.
    """
    try:
        return explainer(shap_input)
    except Exception as exc:
        if not _is_additivity_error(shap, exc):
            raise
        log.warning(
            "SHAP additivity check failed for %s; retrying with check_additivity=False",
            type(explainer).__name__,
        )
        return explainer(shap_input, check_additivity=False)


def _is_additivity_error(shap: Any, exc: Exception) -> bool:
    """Is additivity error.

    Args:
        shap (Any): Value used by the function.
        exc (Exception): Exception value used by the function.

    Returns:
        bool: Result of the function.
    """
    shap_exception = getattr(getattr(shap, "utils", None), "_exceptions", None)
    explainer_error = getattr(shap_exception, "ExplainerError", None)
    if explainer_error is not None and isinstance(exc, explainer_error):
        return "Additivity check failed" in str(exc)
    return False


def _resolve_shap_model(model: Any) -> Any:
    """Resolve shap model.

    Args:
        model (Any): Model to process.

    Returns:
        Any: Result of the function.
    """
    inner_model = getattr(model, "_model", None)
    if inner_model is not None:
        return inner_model
    return model


def _predict_callable(model: Any) -> Any:
    """Predict callable.

    Args:
        model (Any): Model to process.

    Returns:
        Any: Result of the function.
    """
    predict = getattr(model, "predict", None)
    if predict is None:
        raise TypeError(f"Model of type {type(model).__name__} does not expose a predict method for SHAP fallback.")
    return predict


def _coerce_shap_matrix(values: Any) -> np.ndarray:
    """Coerce shap matrix.

    Args:
        values (Any): Values to process.

    Returns:
        np.ndarray: Result of the function.
    """
    matrix = np.asarray(values)
    if matrix.ndim == 1:
        return matrix.reshape(-1, 1)
    if matrix.ndim == 2:
        return matrix
    if matrix.ndim == 3 and matrix.shape[-1] == 1:
        return matrix[:, :, 0]
    raise ValueError(f"Unsupported SHAP value tensor shape: {matrix.shape}")


def _write_summary_plot(
    *,
    contribution_frame: pd.DataFrame,
    output_path: Path,
    max_display: int,
) -> None:
    """Write summary plot.

    Args:
        contribution_frame (pd.DataFrame): pd.DataFrame value used by the function.
        output_path (Path): Path value used by the function.
        max_display (int): int value used by the function.

    Returns:
        None: None.
    """
    import matplotlib.pyplot as plt

    plt.rcParams["font.family"] = "Times New Roman"

    plot_frame = contribution_frame.head(max_display).iloc[::-1]
    height = max(4.0, 0.45 * len(plot_frame) + 1.5)
    figure, axis = plt.subplots(figsize=(10, height))
    axis.barh(plot_frame["feature"], plot_frame["mean_abs_shap"], color="#1f77b4")
    axis.set_xlabel("mean(|SHAP value|)")
    axis.set_ylabel("feature")
    axis.set_title("SHAP Feature Contributions")
    figure.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(figure)
    log.info("Wrote SHAP plot artifact: %s", output_path)


def _write_beeswarm_plot(
    *,
    explanation: Any,
    output_path: Path,
    max_display: int,
) -> None:
    """Write beeswarm plot.

    Args:
        explanation (Any): Value used by the function.
        output_path (Path): Path value used by the function.
        max_display (int): int value used by the function.

    Returns:
        None: None.
    """
    import matplotlib.pyplot as plt

    plt.rcParams["font.family"] = "Times New Roman"
    import shap

    shap.plots.beeswarm(
        explanation,
        max_display=max_display,
        show=False,
        plot_size="auto",
    )
    figure = plt.gcf()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(figure)
    log.info("Wrote SHAP plot artifact: %s", output_path)


def _write_heatmap_plot(
    *,
    explanation: Any,
    output_path: Path,
    max_display: int,
) -> None:
    """Write heatmap plot.

    Args:
        explanation (Any): Value used by the function.
        output_path (Path): Path value used by the function.
        max_display (int): int value used by the function.

    Returns:
        None: None.
    """
    import matplotlib.pyplot as plt

    plt.rcParams["font.family"] = "Times New Roman"
    import shap

    shap.plots.heatmap(
        explanation,
        max_display=max_display,
        show=False,
        plot_width=12,
    )
    figure = plt.gcf()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(figure)
    log.info("Wrote SHAP plot artifact: %s", output_path)
