from __future__ import annotations

import numpy as np
import pandas as pd
import logging


log = logging.getLogger(__name__)


def rank_per_epoch(mat: pd.DataFrame, higher_is_better: bool = True, tie_method: str = "average") -> pd.DataFrame:
    """Rank 1 = best. For loss metrics, set higher_is_better=False."""
    return mat.rank(axis=0, ascending=not higher_is_better, method=tie_method)


def rank_per_epoch_metric(
    mat: pd.DataFrame,
    metric_key: str,
    higher_is_better: dict[str, bool] | None = None,
) -> pd.DataFrame:
    """Rank 1 = best (acc high is good; loss low is good)."""
    if higher_is_better is None:
        log.error("higher_is_better not set for rank_per_epoch_metric")

    # if unknown metric, default to higher is better
    hib = higher_is_better.get(metric_key, True)
    asc = not hib  # loss => asc True
    return mat.rank(axis=0, ascending=asc, method="average")


def rank_change_table(
    mat: pd.DataFrame,
    rank_mat: pd.DataFrame | None = None,
    smoothening_threshold: float | None = None,
    tau: float | None = None,
) -> pd.DataFrame:
    """
    Computes, per consecutive epoch pair (e0->e1):
      - models_changed_rank: count of models whose rank changed without smoothening
      - models_changed_rank_with_smoothening:
        rank-changed AND |metric change|>=smoothening_threshold (optional)
    Rank convention: rank_mat should be "1=best". If not given, ranks are computed (higher is better).
    """
    if smoothening_threshold is None and tau is not None:
        smoothening_threshold = tau

    if rank_mat is None:
        rank_mat = mat.rank(axis=0, ascending=False, method="average")

    epochs = list(mat.columns)
    if len(epochs) < 2:
        return pd.DataFrame()

    rows = []
    for e0, e1 in zip(epochs[:-1], epochs[1:]):
        r0 = rank_mat[e0]
        r1 = rank_mat[e1]
        rank_changed = (r0 != r1)

        row = {
            "from_epoch": e0,
            "to_epoch": e1,
            "models_changed_rank": int(rank_changed.sum()),
            "models_moved_up": int(((r1 - r0) < 0).sum()),
            "models_moved_down": int(((r1 - r0) > 0).sum()),
        }

        if smoothening_threshold is not None:
            a0 = mat[e0]
            a1 = mat[e1]
            big_move = (a1 - a0).abs() >= smoothening_threshold

            row.update({
                "models_changed_rank_with_smoothening": int((rank_changed & big_move).sum()),
                "models_moved_up_with_smoothening": int((((r1 - r0) < 0) & big_move).sum()),
                "models_moved_down_with_smoothening": int((((r1 - r0) > 0) & big_move).sum()),
                "smoothening_threshold_rank_change": float(smoothening_threshold),
            })

        rows.append(row)

    return pd.DataFrame(rows)


def count_pairwise_flips(acc_t, acc_t1, smoothening_threshold=None, tau=None) -> int:
    if smoothening_threshold is None and tau is not None:
        smoothening_threshold = tau

    a0 = np.asarray(acc_t, dtype=float)
    a1 = np.asarray(acc_t1, dtype=float)
    N = a0.shape[0]

    d0 = a0[:, None] - a0[None, :]
    d1 = a1[:, None] - a1[None, :]

    sign0 = np.sign(d0)
    sign1 = np.sign(d1)

    flipped = (sign0 * sign1) < 0
    mask = np.triu(np.ones((N, N), dtype=bool), k=1)

    if smoothening_threshold is None:
        return int(np.sum(flipped & mask))

    separated = (np.abs(d0) >= smoothening_threshold) & (np.abs(d1) >= smoothening_threshold)
    return int(np.sum(flipped & separated & mask))


def pairwise_flip_table(
    mat: pd.DataFrame,
    smoothening_threshold: float = 0.03,
    tau: float | None = None,
) -> pd.DataFrame:
    if tau is not None:
        smoothening_threshold = tau

    epochs = list(mat.columns)
    if len(epochs) < 2:
        return pd.DataFrame()

    rows = []
    for e0, e1 in zip(epochs[:-1], epochs[1:]):
        a0 = mat[e0].to_numpy()
        a1 = mat[e1].to_numpy()

        flips_without_smoothening = count_pairwise_flips(a0, a1, smoothening_threshold=None)
        flips_with_smoothening = count_pairwise_flips(a0, a1, smoothening_threshold=smoothening_threshold)

        rows.append({
            "from_epoch": e0,
            "to_epoch": e1,
            "pairwise_flips_without_smoothening": flips_without_smoothening,
            "pairwise_flips_with_smoothening": flips_with_smoothening,
            "smoothening_threshold_pairwise_flip": smoothening_threshold,
        })

    return pd.DataFrame(rows)
