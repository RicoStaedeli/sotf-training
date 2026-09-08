from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor, as_completed
from collections import OrderedDict
from dataclasses import dataclass, field
from multiprocessing import get_context
from typing import Any, Dict, Iterator, List, Optional, Sequence, Tuple
from pathlib import Path
import os
import re
import math

import numpy as np
import pandas as pd
import torch
import tqdm
import logging

from nnrun.utils.dataset_base_optimized import ModelDatasetBase

log = logging.getLogger(__name__)

# -----------------------------------------------------------------------------
# Dataset loading
# -----------------------------------------------------------------------------
def load_dataset_from_path(
    path: Path,
    dstype: str,
    epoch_lst: Optional[Sequence[int]] = None,
    preload_properties: bool = False,
    validate_checkpoints: bool = False,
) -> Dict[str, ModelDatasetBase]:
    if epoch_lst is None:
        epoch_lst = [5, 15, 25]

    result_key_list = [
        "test_acc",
        "train_acc",
        "test_loss",
        "train_loss",
        "training_iteration",
        "ggap",
    ]

    if dstype == "Seed":
        config_key_list = []
    elif dstype == "ViT":
        config_key_list = [
            "optim::optimizer",
            "model::dropout",
            "optim::lr",
            "optim::wd",
        ]
    else:
        config_key_list = [
            "model::nlin",
            "model::init_type",
            "optim::optimizer",
            "model::dropout",
            "optim::lr",
            "optim::wd",
        ]

    property_keys = {
        "result_keys": result_key_list,
        "config_keys": config_key_list,
    }

    path_zoo_root = [path.absolute()]

    common_kwargs = dict(
        root=path_zoo_root,
        layer_lst=None,
        epoch_lst=epoch_lst,
        use_bias=True,
        ds_split=None,
        max_samples=None,
        weight_threshold=5,
        filter_function=None,
        property_keys=property_keys,
        num_threads=6,
        verbosity=0,
        shuffle_path=True,
        preload_properties=preload_properties,
        validate_checkpoints=validate_checkpoints,
    )

    full = ModelDatasetBase(train_val_test="full", **common_kwargs)
    return {"full": full}


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------
_PARAM_GROUP_CACHE: Dict[Tuple[str, ...], List[Dict[str, Any]]] = {}
_QUANTILE_Q = torch.tensor([0.0, 0.25, 0.5, 0.75, 1.0], dtype=torch.float32)

# Safe threshold for torch.quantile on 1D flattened tensors.
# If exceeded, fall back to numpy.
_TORCH_QUANTILE_MAX_NUMEL = 50_000_000

_NAN_MATRIX_STATS = {
    "fro_norm": np.nan,
    "mean_abs": np.nan,
    "max_abs": np.nan,
    "rms": np.nan,
    "n_params": np.nan,
    "n_rows": np.nan,
    "n_cols": np.nan,
    "aspect_ratio": np.nan,
    "row_l2_mean": np.nan,
    "row_l2_max": np.nan,
    "col_l2_mean": np.nan,
    "col_l2_max": np.nan,
}

_NAN_DISTRIBUTION_STATS = {
    "mean": np.nan,
    "variance": np.nan,
    "percentile_0": np.nan,
    "percentile_25": np.nan,
    "percentile_50": np.nan,
    "percentile_75": np.nan,
    "percentile_100": np.nan,
    "range_width": np.nan,
}

_NAN_SHARPNESS_STATS = {
    "sharp_proxy_maxabs_over_l2": np.nan,
    "sharp_proxy_maxabs_over_meanabs": np.nan,
    "sharp_proxy_rms_over_meanabs": np.nan,
    "sharp_proxy_row_imbalance": np.nan,
    "sharp_proxy_col_imbalance": np.nan,
}


def sanitize_layer_name(name: str) -> str:
    name = name.replace(".", "_")
    name = re.sub(r"[^a-zA-Z0-9_]+", "_", name)
    name = re.sub(r"_+", "_", name).strip("_")
    return name


def infer_tensor_type(param_name: str, tensor: torch.Tensor) -> str:
    name = param_name.lower()

    if "embedding" in name:
        return "embedding"
    if "cls_token" in name or "token" in name:
        return "token"
    if "norm" in name:
        return "norm"
    if "bias" in name and tensor.ndim == 1:
        return "bias"

    if tensor.ndim == 4:
        return "conv2d"
    if tensor.ndim == 3:
        return "conv1d_or_seq"
    if tensor.ndim == 2:
        return "linear"
    if tensor.ndim == 1:
        return "vector"
    if tensor.ndim == 0:
        return "scalar"
    return f"{tensor.ndim}d"


def infer_parameter_groups_from_checkpoint(
    checkpoint: Dict[str, torch.Tensor],
) -> List[Dict[str, Any]]:
    key_sig = tuple(checkpoint.keys())
    cached = _PARAM_GROUP_CACHE.get(key_sig)
    if cached is not None:
        return cached

    grouped: Dict[str, Dict[str, Any]] = {}
    standalone: List[Dict[str, Any]] = []
    seen_order = 0

    for key, tensor in checkpoint.items():
        if not torch.is_tensor(tensor):
            continue

        if key.endswith(".weight"):
            base = key[:-7]
            if base not in grouped:
                grouped[base] = {
                    "base_name": base,
                    "weight_key": None,
                    "bias_key": None,
                    "order": seen_order,
                }
                seen_order += 1
            grouped[base]["weight_key"] = key

        elif key.endswith(".bias"):
            base = key[:-5]
            if base not in grouped:
                grouped[base] = {
                    "base_name": base,
                    "weight_key": None,
                    "bias_key": None,
                    "order": seen_order,
                }
                seen_order += 1
            grouped[base]["bias_key"] = key

        else:
            standalone.append(
                {
                    "base_name": key,
                    "direct_key": key,
                    "order": seen_order,
                    "tensor": tensor,
                }
            )
            seen_order += 1

    out: List[Dict[str, Any]] = []

    for _, item in sorted(grouped.items(), key=lambda x: x[1]["order"]):
        weight_key = item["weight_key"]
        bias_key = item["bias_key"]
        ref_key = weight_key if weight_key is not None else bias_key
        ref_tensor = checkpoint[ref_key]
        base_name = item["base_name"]

        out.append(
            {
                "order": item["order"],
                "base_name": base_name,
                "safe_name": sanitize_layer_name(base_name),
                "layer_type": infer_tensor_type(base_name, ref_tensor),
                "weight_key": weight_key,
                "bias_key": bias_key,
                "direct_key": None,
            }
        )

    for item in sorted(standalone, key=lambda x: x["order"]):
        base_name = item["base_name"]
        tensor = item["tensor"]
        out.append(
            {
                "order": item["order"],
                "base_name": base_name,
                "safe_name": sanitize_layer_name(base_name),
                "layer_type": infer_tensor_type(base_name, tensor),
                "weight_key": None,
                "bias_key": None,
                "direct_key": item["direct_key"],
            }
        )

    out.sort(key=lambda x: x["order"])
    _PARAM_GROUP_CACHE[key_sig] = out
    return out


def reshape_tensor_to_matrix(tensor: torch.Tensor) -> torch.Tensor:
    x = tensor.detach().float()
    if x.ndim == 2:
        return x
    if x.ndim >= 3:
        return x.reshape(x.shape[0], -1)
    if x.ndim == 1:
        return x.reshape(1, -1)
    if x.ndim == 0:
        return x.reshape(1, 1)
    return x.reshape(x.shape[0], -1)


def flatten_float(tensor: torch.Tensor) -> torch.Tensor:
    return tensor.detach().float().reshape(-1)


def resolve_checkpoint_path(path: str | Path, epoch: int) -> Path:
    base_path = Path(path)

    p1 = base_path / f"checkpoint_{epoch}" / "checkpoints"
    if p1.exists():
        return p1

    p2 = base_path / f"checkpoint_{epoch:06d}" / "checkpoints"
    if p2.exists():
        return p2

    raise FileNotFoundError(
        f"Could not find checkpoint for epoch {epoch} in {base_path}. "
        f"Tried {p1} and {p2}"
    )


def vector_norm_triplet_from_flat(x: torch.Tensor) -> Tuple[float, float, float]:
    if x.numel() == 0 or not torch.isfinite(x).all():
        return np.nan, np.nan, np.nan
    l1 = float(torch.linalg.vector_norm(x, ord=1).item())
    l2 = float(torch.linalg.vector_norm(x, ord=2).item())
    linf = float(torch.linalg.vector_norm(x, ord=float("inf")).item())
    return l1, l2, linf


def vector_norm_triplet(x: torch.Tensor) -> Tuple[float, float, float]:
    return vector_norm_triplet_from_flat(flatten_float(x))


def combine_norm_triplets(
    a: Optional[Tuple[float, float, float]],
    b: Optional[Tuple[float, float, float]],
) -> Optional[Tuple[float, float, float]]:
    if a is None:
        return b
    if b is None:
        return a

    a_l1, a_l2, a_linf = a
    b_l1, b_l2, b_linf = b
    return (
        float(a_l1 + b_l1),
        float(math.sqrt(a_l2 * a_l2 + b_l2 * b_l2)),
        float(max(a_linf, b_linf)),
    )


def flatten_checkpoint_model_tensor(
    checkpoint: Dict[str, torch.Tensor],
    param_groups: List[Dict[str, Any]],
    use_bias: bool = True,
) -> torch.Tensor:
    parts: List[torch.Tensor] = []

    for group in param_groups:
        direct_key = group["direct_key"]
        weight_key = group["weight_key"]
        bias_key = group["bias_key"]

        if direct_key is not None:
            parts.append(flatten_float(checkpoint[direct_key]))
            continue

        if weight_key is not None:
            parts.append(flatten_float(checkpoint[weight_key]))
        if use_bias and bias_key is not None:
            parts.append(flatten_float(checkpoint[bias_key]))

    if not parts:
        return torch.empty(0, dtype=torch.float32)

    return torch.cat(parts, dim=0)


def cheap_matrix_stats_for_weight_tensor(
    tensor: torch.Tensor,
    eps: float = 1e-12,
) -> Dict[str, float]:
    W = reshape_tensor_to_matrix(tensor)

    if W.numel() == 0 or not torch.isfinite(W).all():
        return dict(_NAN_MATRIX_STATS)

    absW = W.abs()
    row_l2 = torch.linalg.vector_norm(W, ord=2, dim=1)
    col_l2 = torch.linalg.vector_norm(W, ord=2, dim=0)

    n_rows, n_cols = W.shape
    n_params = W.numel()

    return {
        "fro_norm": float(torch.linalg.matrix_norm(W, ord="fro").item()),
        "mean_abs": float(absW.mean().item()),
        "max_abs": float(absW.max().item()),
        "rms": float(torch.sqrt((W * W).mean()).item()),
        "n_params": float(n_params),
        "n_rows": float(n_rows),
        "n_cols": float(n_cols),
        "aspect_ratio": float(max(n_rows, n_cols) / (min(n_rows, n_cols) + eps)),
        "row_l2_mean": float(row_l2.mean().item()),
        "row_l2_max": float(row_l2.max().item()),
        "col_l2_mean": float(col_l2.mean().item()),
        "col_l2_max": float(col_l2.max().item()),
    }


def distribution_stats_for_flat_tensor(
    x: torch.Tensor,
    compute_percentiles: bool = True,
) -> Dict[str, float]:
    if x.numel() == 0 or not torch.isfinite(x).all():
        return dict(_NAN_DISTRIBUTION_STATS)

    mean = float(x.mean().item())
    variance = float(x.var(unbiased=False).item())

    if compute_percentiles:
        if x.numel() <= _TORCH_QUANTILE_MAX_NUMEL:
            try:
                q = torch.quantile(x, _QUANTILE_Q.to(device=x.device, dtype=x.dtype))
                q0, q25, q50, q75, q100 = [float(v.item()) for v in q]
            except RuntimeError:
                x_np = x.detach().cpu().numpy()
                q0, q25, q50, q75, q100 = map(
                    float, np.quantile(x_np, [0.0, 0.25, 0.5, 0.75, 1.0])
                )
        else:
            x_np = x.detach().cpu().numpy()
            q0, q25, q50, q75, q100 = map(
                float, np.quantile(x_np, [0.0, 0.25, 0.5, 0.75, 1.0])
            )
        range_width = float(q100 - q0)
    else:
        q0 = q25 = q50 = q75 = q100 = np.nan
        range_width = np.nan

    return {
        "mean": mean,
        "variance": variance,
        "percentile_0": q0,
        "percentile_25": q25,
        "percentile_50": q50,
        "percentile_75": q75,
        "percentile_100": q100,
        "range_width": range_width,
    }


def distribution_stats_for_tensor(
    tensor: torch.Tensor,
    compute_percentiles: bool = True,
) -> Dict[str, float]:
    return distribution_stats_for_flat_tensor(
        flatten_float(tensor),
        compute_percentiles=compute_percentiles,
    )


def sharpness_proxies_for_flat_weight(
    flat: torch.Tensor,
    matrix_stats: Dict[str, float],
    eps: float = 1e-12,
) -> Dict[str, float]:
    if flat.numel() == 0 or not torch.isfinite(flat).all():
        return dict(_NAN_SHARPNESS_STATS)

    l2 = float(torch.linalg.vector_norm(flat, ord=2).item())
    max_abs = matrix_stats["max_abs"]
    mean_abs = matrix_stats["mean_abs"]
    rms = matrix_stats["rms"]
    row_l2_mean = matrix_stats["row_l2_mean"]
    row_l2_max = matrix_stats["row_l2_max"]
    col_l2_mean = matrix_stats["col_l2_mean"]
    col_l2_max = matrix_stats["col_l2_max"]

    return {
        "sharp_proxy_maxabs_over_l2": float(max_abs / (l2 + eps)),
        "sharp_proxy_maxabs_over_meanabs": float(max_abs / (mean_abs + eps)),
        "sharp_proxy_rms_over_meanabs": float(rms / (mean_abs + eps)),
        "sharp_proxy_row_imbalance": float(row_l2_max / (row_l2_mean + eps)),
        "sharp_proxy_col_imbalance": float(col_l2_max / (col_l2_mean + eps)),
    }


def sharpness_proxies_for_weight_tensor(
    tensor: torch.Tensor,
    matrix_stats: Optional[Dict[str, float]] = None,
    eps: float = 1e-12,
) -> Dict[str, float]:
    flat = flatten_float(tensor)
    if matrix_stats is None:
        matrix_stats = cheap_matrix_stats_for_weight_tensor(tensor, eps=eps)
    return sharpness_proxies_for_flat_weight(flat, matrix_stats, eps=eps)


def safe_cosine_similarity(
    x: torch.Tensor,
    y: torch.Tensor,
    eps: float = 1e-12,
) -> float:
    nx = torch.linalg.vector_norm(x, ord=2)
    ny = torch.linalg.vector_norm(y, ord=2)
    denom = float((nx * ny).item())
    if denom < eps:
        return 0.0
    return float(torch.dot(x, y).item() / (denom + eps))


def build_base_meta(
    dataset,
    i: int,
    sample: Dict[str, Any],
    property_cols: Sequence[str],
) -> Dict[str, Any]:
    path = sample["path"]
    epoch = sample["epoch"] if "epoch" in sample else sample["epoch_in"]

    return {
        "sample_idx": int(i),
        "model_id": Path(path).name,
        "path": str(path),
        "epoch": int(epoch),
        "_property_values": tuple(
            dataset.properties[key][i] for key in property_cols
        ) if dataset.properties is not None else tuple(),
    }


def append_distance_metrics(
    builder: "MetricTableBuilder",
    base_prefix: Tuple[Any, ...],
    theta_t: torch.Tensor,
    theta0: torch.Tensor,
    theta0_norm: float,
    prev_theta: Optional[torch.Tensor],
    cumulative_path: float,
) -> float:
    dist_to_init = float(torch.linalg.vector_norm(theta_t - theta0, ord=2).item())
    rel_dist_to_init = float(dist_to_init / (theta0_norm + 1e-12))
    cos_to_init = safe_cosine_similarity(theta_t, theta0)

    if prev_theta is None:
        dist_to_prev = 0.0
        cos_to_prev = 1.0
    else:
        dist_to_prev = float(torch.linalg.vector_norm(theta_t - prev_theta, ord=2).item())
        cos_to_prev = safe_cosine_similarity(theta_t, prev_theta)
        cumulative_path += dist_to_prev

    straightness = float(dist_to_init / (cumulative_path + 1e-12)) if cumulative_path > 0 else 1.0

    builder.append_from_prefix(base_prefix, -1, "__model__", "__model__", "model", "distance", "distance_to_init_l2", dist_to_init, scope="default")
    builder.append_from_prefix(base_prefix, -1, "__model__", "__model__", "model", "distance", "relative_distance_to_init_l2", rel_dist_to_init, scope="default")
    builder.append_from_prefix(base_prefix, -1, "__model__", "__model__", "model", "distance", "distance_to_prev_l2", dist_to_prev, scope="default")
    builder.append_from_prefix(base_prefix, -1, "__model__", "__model__", "model", "distance", "cosine_to_init", cos_to_init, scope="default")
    builder.append_from_prefix(base_prefix, -1, "__model__", "__model__", "model", "distance", "cosine_to_prev", cos_to_prev, scope="default")
    builder.append_from_prefix(base_prefix, -1, "__model__", "__model__", "model", "distance", "cumulative_path_length_l2", cumulative_path, scope="default")
    builder.append_from_prefix(base_prefix, -1, "__model__", "__model__", "model", "distance", "straightness_ratio", straightness, scope="default")

    return cumulative_path


# -----------------------------------------------------------------------------
# Streaming aggregators
# -----------------------------------------------------------------------------
@dataclass(slots=True)
class RunningScalarAgg:
    total: float = 0.0
    count: int = 0
    max_value: float = -float("inf")

    def add(self, value: float) -> None:
        if not np.isfinite(value):
            return
        v = float(value)
        self.total += v
        self.count += 1
        if v > self.max_value:
            self.max_value = v

    def mean(self) -> float:
        return float(self.total / self.count) if self.count > 0 else np.nan

    def max(self) -> float:
        return float(self.max_value) if self.count > 0 else np.nan


@dataclass(slots=True)
class RunningNormAgg:
    l1_sum: float = 0.0
    l2_sq_sum: float = 0.0
    linf_max: float = 0.0

    def add_l1(self, value: float) -> None:
        if np.isfinite(value):
            self.l1_sum += float(value)

    def add_l2(self, value: float) -> None:
        if np.isfinite(value):
            v = float(value)
            self.l2_sq_sum += v * v

    def add_linf(self, value: float) -> None:
        if np.isfinite(value):
            v = float(value)
            if v > self.linf_max:
                self.linf_max = v

    def value_for(self, norm_name: str) -> float:
        if norm_name == "l1":
            return float(self.l1_sum)
        if norm_name == "l2":
            return float(math.sqrt(self.l2_sq_sum))
        if norm_name == "linf":
            return float(self.linf_max)
        raise ValueError(f"Unsupported norm '{norm_name}'.")


# -----------------------------------------------------------------------------
# Columnar row builder
# -----------------------------------------------------------------------------
@dataclass(slots=True)
class MetricTableBuilder:
    property_cols: List[str]
    columns: List[str] = field(init=False)
    rows: List[Tuple[Any, ...]] = field(default_factory=list)

    FIXED_FRONT = [
        "sample_idx",
        "model_id",
        "path",
        "epoch",
        "layer_order",
        "layer_name",
        "layer_safe_name",
        "layer_type",
        "metric_family",
        "metric",
        "scope",
        "value",
    ]

    def __init__(self, property_cols: Sequence[str]) -> None:
        prop_cols = list(property_cols)
        self.property_cols = prop_cols
        self.columns = self.FIXED_FRONT + prop_cols
        self.rows = []

    @staticmethod
    def make_base_prefix(base_meta: Dict[str, Any]) -> Tuple[Any, ...]:
        return (
            base_meta["sample_idx"],
            base_meta["model_id"],
            base_meta["path"],
            base_meta["epoch"],
            *base_meta["_property_values"],
        )

    def append_from_prefix(
        self,
        base_prefix: Tuple[Any, ...],
        layer_order: int,
        layer_name: str,
        layer_safe_name: str,
        layer_type: str,
        metric_family: str,
        metric: str,
        value: float,
        scope: str = "default",
    ) -> None:
        sample_idx, model_id, path, epoch, *property_values = base_prefix
        self.rows.append(
            (
                sample_idx,
                model_id,
                path,
                epoch,
                int(layer_order),
                layer_name,
                layer_safe_name,
                layer_type,
                metric_family,
                metric,
                scope,
                float(value),
                *property_values,
            )
        )

    def append(
        self,
        base_meta: Dict[str, Any],
        layer_order: int,
        layer_name: str,
        layer_safe_name: str,
        layer_type: str,
        metric_family: str,
        metric: str,
        value: float,
        scope: str = "default",
    ) -> None:
        self.append_from_prefix(
            self.make_base_prefix(base_meta),
            layer_order,
            layer_name,
            layer_safe_name,
            layer_type,
            metric_family,
            metric,
            value,
            scope=scope,
        )

    def append_norm_triplet_from_prefix(
        self,
        base_prefix: Tuple[Any, ...],
        layer_order: int,
        layer_name: str,
        layer_safe_name: str,
        layer_type: str,
        scope: str,
        norm_triplet: Tuple[float, float, float],
        model_norm_accumulator: Dict[Tuple[str, str], RunningNormAgg],
    ) -> None:
        l1, l2, linf = norm_triplet

        self.append_from_prefix(
            base_prefix, layer_order, layer_name, layer_safe_name,
            layer_type, "norm", "l1", l1, scope=scope
        )
        model_norm_accumulator.setdefault((scope, "l1"), RunningNormAgg()).add_l1(l1)

        self.append_from_prefix(
            base_prefix, layer_order, layer_name, layer_safe_name,
            layer_type, "norm", "l2", l2, scope=scope
        )
        model_norm_accumulator.setdefault((scope, "l2"), RunningNormAgg()).add_l2(l2)

        self.append_from_prefix(
            base_prefix, layer_order, layer_name, layer_safe_name,
            layer_type, "norm", "linf", linf, scope=scope
        )
        model_norm_accumulator.setdefault((scope, "linf"), RunningNormAgg()).add_linf(linf)

    def to_dataframe(self) -> pd.DataFrame:
        return pd.DataFrame.from_records(self.rows, columns=self.columns)


def _build_path_jobs(
    dataset,
    property_cols: Sequence[str],
) -> List[Dict[str, Any]]:
    jobs_by_path: "OrderedDict[str, Dict[str, Any]]" = OrderedDict()

    for i, sample in enumerate(dataset.samples):
        path = str(sample["path"])
        epoch = int(sample["epoch"] if "epoch" in sample else sample["epoch_in"])
        job = jobs_by_path.setdefault(path, {"path": path, "samples": []})
        job["samples"].append(
            {
                "sample_idx": int(i),
                "model_id": Path(path).name,
                "path": path,
                "epoch": epoch,
                "property_values": tuple(
                    dataset.properties[key][i] for key in property_cols
                ) if dataset.properties is not None else tuple(),
            }
        )

    return list(jobs_by_path.values())


def _rows_to_dataframe(
    rows: List[Tuple[Any, ...]],
    property_cols: Sequence[str],
) -> pd.DataFrame:
    columns = MetricTableBuilder(property_cols=property_cols).columns
    return pd.DataFrame.from_records(rows, columns=columns)


def _compute_metric_rows_for_path_job(job: Dict[str, Any]) -> List[Tuple[Any, ...]]:
    property_cols = list(job["property_cols"])
    builder = MetricTableBuilder(property_cols=property_cols)

    use_bias = bool(job["use_bias"])
    compute_percentiles = bool(job["compute_percentiles"])
    compute_full_model_theta = bool(job["compute_full_model_theta"])

    theta0: Optional[torch.Tensor] = None
    theta0_norm = 0.0
    prev_theta: Optional[torch.Tensor] = None
    prev_epoch: Optional[int] = None
    cumulative_path = 0.0

    for sample in job["samples"]:
        path = sample["path"]
        epoch = int(sample["epoch"])
        checkpoint = torch.load(
            resolve_checkpoint_path(path, epoch),
            map_location="cpu",
            weights_only=True,
        )

        base_prefix = (
            sample["sample_idx"],
            sample["model_id"],
            path,
            epoch,
            *sample["property_values"],
        )
        param_groups = infer_parameter_groups_from_checkpoint(checkpoint)

        model_norm_accumulator: Dict[Tuple[str, str], RunningNormAgg] = {}
        model_matrix_values: Dict[str, RunningScalarAgg] = {}
        model_sharp_values: Dict[str, RunningScalarAgg] = {}
        theta_parts: Optional[List[torch.Tensor]] = [] if compute_full_model_theta else None

        for group in param_groups:
            layer_order = int(group["order"])
            layer_name = group["base_name"]
            layer_safe_name = group["safe_name"]
            layer_type = group["layer_type"]
            weight_key = group["weight_key"]
            bias_key = group["bias_key"]
            direct_key = group["direct_key"]

            if direct_key is not None:
                tensor = checkpoint[direct_key]
                flat = flatten_float(tensor)
                if theta_parts is not None:
                    theta_parts.append(flat)

                norm_triplet = vector_norm_triplet_from_flat(flat)
                builder.append_norm_triplet_from_prefix(
                    base_prefix,
                    layer_order,
                    layer_name,
                    layer_safe_name,
                    layer_type,
                    "combined",
                    norm_triplet,
                    model_norm_accumulator,
                )
                builder.append_norm_triplet_from_prefix(
                    base_prefix,
                    layer_order,
                    layer_name,
                    layer_safe_name,
                    layer_type,
                    "weight_only",
                    norm_triplet,
                    model_norm_accumulator,
                )

                matrix_stats = cheap_matrix_stats_for_weight_tensor(tensor)
                for metric_name, value in matrix_stats.items():
                    builder.append_from_prefix(
                        base_prefix,
                        layer_order,
                        layer_name,
                        layer_safe_name,
                        layer_type,
                        "matrix",
                        metric_name,
                        value,
                        scope="default",
                    )
                    model_matrix_values.setdefault(metric_name, RunningScalarAgg()).add(value)

                distribution_stats = distribution_stats_for_flat_tensor(
                    flat,
                    compute_percentiles=compute_percentiles,
                )
                for metric_name, value in distribution_stats.items():
                    builder.append_from_prefix(
                        base_prefix,
                        layer_order,
                        layer_name,
                        layer_safe_name,
                        layer_type,
                        "distribution",
                        metric_name,
                        value,
                        scope="weight_only",
                    )

                sharpness_stats = sharpness_proxies_for_flat_weight(flat, matrix_stats=matrix_stats)
                for metric_name, value in sharpness_stats.items():
                    builder.append_from_prefix(
                        base_prefix,
                        layer_order,
                        layer_name,
                        layer_safe_name,
                        layer_type,
                        "sharpness",
                        metric_name,
                        value,
                        scope="default",
                    )
                    model_sharp_values.setdefault(metric_name, RunningScalarAgg()).add(value)

                continue

            w_norms = None
            b_norms = None

            if weight_key is not None:
                weight_tensor = checkpoint[weight_key]
                w = flatten_float(weight_tensor)
                if theta_parts is not None:
                    theta_parts.append(w)

                w_norms = vector_norm_triplet_from_flat(w)
                builder.append_norm_triplet_from_prefix(
                    base_prefix,
                    layer_order,
                    layer_name,
                    layer_safe_name,
                    layer_type,
                    "weight_only",
                    w_norms,
                    model_norm_accumulator,
                )

                matrix_stats = cheap_matrix_stats_for_weight_tensor(weight_tensor)
                for metric_name, value in matrix_stats.items():
                    builder.append_from_prefix(
                        base_prefix,
                        layer_order,
                        layer_name,
                        layer_safe_name,
                        layer_type,
                        "matrix",
                        metric_name,
                        value,
                        scope="default",
                    )
                    model_matrix_values.setdefault(metric_name, RunningScalarAgg()).add(value)

                distribution_stats = distribution_stats_for_flat_tensor(
                    w,
                    compute_percentiles=compute_percentiles,
                )
                for metric_name, value in distribution_stats.items():
                    builder.append_from_prefix(
                        base_prefix,
                        layer_order,
                        layer_name,
                        layer_safe_name,
                        layer_type,
                        "distribution",
                        metric_name,
                        value,
                        scope="weight_only",
                    )

                sharpness_stats = sharpness_proxies_for_flat_weight(w, matrix_stats=matrix_stats)
                for metric_name, value in sharpness_stats.items():
                    builder.append_from_prefix(
                        base_prefix,
                        layer_order,
                        layer_name,
                        layer_safe_name,
                        layer_type,
                        "sharpness",
                        metric_name,
                        value,
                        scope="default",
                    )
                    model_sharp_values.setdefault(metric_name, RunningScalarAgg()).add(value)

            if use_bias and bias_key is not None:
                bias_tensor = checkpoint[bias_key]
                b = flatten_float(bias_tensor)
                if theta_parts is not None:
                    theta_parts.append(b)

                b_norms = vector_norm_triplet_from_flat(b)
                builder.append_norm_triplet_from_prefix(
                    base_prefix,
                    layer_order,
                    layer_name,
                    layer_safe_name,
                    layer_type,
                    "bias_only",
                    b_norms,
                    model_norm_accumulator,
                )

                distribution_stats = distribution_stats_for_flat_tensor(
                    b,
                    compute_percentiles=compute_percentiles,
                )
                for metric_name, value in distribution_stats.items():
                    builder.append_from_prefix(
                        base_prefix,
                        layer_order,
                        layer_name,
                        layer_safe_name,
                        layer_type,
                        "distribution",
                        metric_name,
                        value,
                        scope="bias_only",
                    )

            combined_norms = combine_norm_triplets(w_norms, b_norms)
            if combined_norms is not None:
                builder.append_norm_triplet_from_prefix(
                    base_prefix,
                    layer_order,
                    layer_name,
                    layer_safe_name,
                    layer_type,
                    "combined",
                    combined_norms,
                    model_norm_accumulator,
                )

        for (scope, norm_name), agg in model_norm_accumulator.items():
            builder.append_from_prefix(
                base_prefix,
                -1,
                "__model__",
                "__model__",
                "model",
                "norm",
                norm_name,
                agg.value_for(norm_name),
                scope=scope,
            )

        for metric_name, agg in model_matrix_values.items():
            builder.append_from_prefix(
                base_prefix,
                -1,
                "__model__",
                "__model__",
                "model",
                "matrix",
                f"{metric_name}_mean",
                agg.mean(),
                scope="default",
            )
            builder.append_from_prefix(
                base_prefix,
                -1,
                "__model__",
                "__model__",
                "model",
                "matrix",
                f"{metric_name}_max",
                agg.max(),
                scope="default",
            )

        for metric_name, agg in model_sharp_values.items():
            builder.append_from_prefix(
                base_prefix,
                -1,
                "__model__",
                "__model__",
                "model",
                "sharpness",
                f"{metric_name}_mean",
                agg.mean(),
                scope="default",
            )
            builder.append_from_prefix(
                base_prefix,
                -1,
                "__model__",
                "__model__",
                "model",
                "sharpness",
                f"{metric_name}_max",
                agg.max(),
                scope="default",
            )

        if compute_full_model_theta:
            theta = (
                torch.cat(theta_parts, dim=0)
                if theta_parts
                else torch.empty(0, dtype=torch.float32)
            )

            model_distribution_stats = distribution_stats_for_flat_tensor(
                theta,
                compute_percentiles=False,
            )
            for metric_name, value in model_distribution_stats.items():
                builder.append_from_prefix(
                    base_prefix,
                    -1,
                    "__model__",
                    "__model__",
                    "model",
                    "distribution",
                    metric_name,
                    value,
                    scope="full_model",
                )

            if prev_epoch is not None and epoch < prev_epoch:
                raise ValueError(
                    f"Checkpoint epochs for path {path} are not in non-decreasing order: "
                    f"got epoch {epoch} after {prev_epoch}."
                )

            if theta0 is None:
                theta0 = theta
                theta0_norm = float(torch.linalg.vector_norm(theta, ord=2).item())
                prev_theta = None
                cumulative_path = 0.0

            cumulative_path = append_distance_metrics(
                builder=builder,
                base_prefix=base_prefix,
                theta_t=theta,
                theta0=theta0,
                theta0_norm=theta0_norm,
                prev_theta=prev_theta,
                cumulative_path=cumulative_path,
            )

            prev_theta = theta
            prev_epoch = epoch

        del checkpoint

    return builder.rows


def iter_metric_frames_for_dataset(
    dataset,
    show_progress: bool = True,
    compute_percentiles: bool = True,
    compute_full_model_theta: bool = True,
    num_workers: int = 1,
) -> Iterator[pd.DataFrame]:
    property_cols = list(dataset.properties.keys()) if dataset.properties is not None else []
    jobs = _build_path_jobs(dataset, property_cols=property_cols)
    if not jobs:
        yield _rows_to_dataframe([], property_cols)
        return

    for job in jobs:
        job["property_cols"] = property_cols
        job["use_bias"] = bool(getattr(dataset, "use_bias", True))
        job["compute_percentiles"] = bool(compute_percentiles)
        job["compute_full_model_theta"] = bool(compute_full_model_theta)

    if num_workers <= 1 or len(jobs) <= 1:
        iterator = jobs
        if show_progress:
            iterator = tqdm.tqdm(
                jobs,
                total=len(jobs),
                desc="Computing per-path metrics",
            )
        for job in iterator:
            rows = _compute_metric_rows_for_path_job(job)
            if rows:
                yield _rows_to_dataframe(rows, property_cols)
        return

    max_workers = min(int(num_workers), len(jobs), os.cpu_count() or 1)
    ctx = get_context("spawn")
    with ProcessPoolExecutor(max_workers=max_workers, mp_context=ctx) as executor:
        futures = [executor.submit(_compute_metric_rows_for_path_job, job) for job in jobs]
        iterator = as_completed(futures)
        if show_progress:
            iterator = tqdm.tqdm(
                iterator,
                total=len(futures),
                desc="Computing per-path metrics",
            )
        for future in iterator:
            rows = future.result()
            if rows:
                yield _rows_to_dataframe(rows, property_cols)


def write_metrics_parquet_for_dataset(
    root_dir: Path,
    dstype: str,
    output_path: Path,
    epoch_lst: Optional[Sequence[int]] = None,
    dataset_split: str = "train",
    preload_properties: bool = True,
    validate_checkpoints: bool = False,
    show_progress: bool = True,
    compute_percentiles: bool = True,
    compute_full_model_theta: bool = True,
    num_workers: int = 1,
    chunk_rows: int = 1_000_000,
) -> None:
    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
    except ImportError as exc:
        raise ImportError(
            "pyarrow is required for chunked weight-statistics parquet writing."
        ) from exc

    ds_to_analyze = load_dataset_from_path(
        path=root_dir,
        dstype=dstype,
        epoch_lst=epoch_lst,
        preload_properties=preload_properties,
        validate_checkpoints=validate_checkpoints,
    )

    if dataset_split not in ds_to_analyze:
        raise KeyError(
            f"dataset_split='{dataset_split}' invalid. "
            f"Expected one of {list(ds_to_analyze.keys())}."
        )

    dataset = ds_to_analyze[dataset_split]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = output_path.with_suffix(output_path.suffix + ".tmp")
    if tmp_path.exists():
        tmp_path.unlink()

    writer = None
    writer_schema = None
    buffered_frames: List[pd.DataFrame] = []
    buffered_rows = 0

    def flush_buffer() -> None:
        nonlocal writer, writer_schema, buffered_frames, buffered_rows
        if not buffered_frames:
            return

        chunk_df = (
            buffered_frames[0]
            if len(buffered_frames) == 1
            else pd.concat(buffered_frames, ignore_index=True)
        )
        table = pa.Table.from_pandas(
            chunk_df,
            schema=writer_schema,
            preserve_index=False,
        ) if writer_schema is not None else pa.Table.from_pandas(
            chunk_df,
            preserve_index=False,
        )

        if writer is None:
            writer_schema = table.schema
            writer = pq.ParquetWriter(tmp_path, writer_schema, compression="snappy")

        writer.write_table(table)
        buffered_frames = []
        buffered_rows = 0

    try:
        for frame in iter_metric_frames_for_dataset(
            dataset,
            show_progress=show_progress,
            compute_percentiles=compute_percentiles,
            compute_full_model_theta=compute_full_model_theta,
            num_workers=num_workers,
        ):
            if frame.empty:
                continue
            buffered_frames.append(frame)
            buffered_rows += len(frame)
            if buffered_rows >= int(chunk_rows):
                flush_buffer()

        flush_buffer()
    finally:
        if writer is not None:
            writer.close()

    if not tmp_path.exists():
        property_cols = list(dataset.properties.keys()) if dataset.properties is not None else []
        _rows_to_dataframe([], property_cols).to_parquet(tmp_path, index=False)

    tmp_path.replace(output_path)


# -----------------------------------------------------------------------------
# Unified computation
# -----------------------------------------------------------------------------
def compute_all_metrics_long(
    dataset,
    show_progress: bool = True,
    compute_percentiles: bool = True,
    compute_full_model_theta: bool = True,
    num_workers: int = 1,
) -> pd.DataFrame:
    frames = list(
        iter_metric_frames_for_dataset(
            dataset,
            show_progress=show_progress,
            compute_percentiles=compute_percentiles,
            compute_full_model_theta=compute_full_model_theta,
            num_workers=num_workers,
        )
    )
    if not frames:
        property_cols = list(dataset.properties.keys()) if dataset.properties is not None else []
        return _rows_to_dataframe([], property_cols)
    if len(frames) == 1:
        return frames[0]
    return pd.concat(frames, ignore_index=True)


# -----------------------------------------------------------------------------
# Convenience function for external callers
# -----------------------------------------------------------------------------
def compute_metrics_df_long_for_dataset(
    root_dir: Path,
    dstype: str,
    epoch_lst: Optional[Sequence[int]] = None,
    dataset_split: str = "train",
    preload_properties: bool = True,
    validate_checkpoints: bool = False,
    show_progress: bool = True,
    compute_percentiles: bool = True,
    compute_full_model_theta: bool = True,
    num_workers: int = 1,
) -> pd.DataFrame:
    ds_to_analyze = load_dataset_from_path(
        path=root_dir,
        dstype=dstype,
        epoch_lst=epoch_lst,
        preload_properties=preload_properties,
        validate_checkpoints=validate_checkpoints,
    )

    if dataset_split not in ds_to_analyze:
        raise KeyError(
            f"dataset_split='{dataset_split}' invalid. "
            f"Expected one of {list(ds_to_analyze.keys())}."
        )

    dataset = ds_to_analyze[dataset_split]
    return compute_all_metrics_long(
        dataset,
        show_progress=show_progress,
        compute_percentiles=compute_percentiles,
        compute_full_model_theta=compute_full_model_theta,
        num_workers=num_workers,
    )
