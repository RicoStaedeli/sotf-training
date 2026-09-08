from __future__ import annotations

import pandas as pd

from nnrun.analyses.delta_to_best import _select_models_for_delta_plot


def test_select_models_for_delta_plot_picks_best_middle_and_worst_ranks() -> None:
    ranked = pd.DataFrame(
        {
            "model_name": [f"model_{i}" for i in range(1, 13)],
            "final_rank": list(range(1, 13)),
        }
    )

    selected = _select_models_for_delta_plot(
        ranked,
        selection_mode="rank_bands",
        best_n=3,
        middle_n=3,
        worst_n=3,
    )

    assert selected["final_rank"].tolist() == [1, 2, 3, 5, 6, 7, 10, 11, 12]
    assert selected["selection_group"].tolist() == [
        "best",
        "best",
        "best",
        "middle",
        "middle",
        "middle",
        "worst",
        "worst",
        "worst",
    ]


def test_select_models_for_delta_plot_deduplicates_small_rank_sets() -> None:
    ranked = pd.DataFrame(
        {
            "model_name": [f"model_{i}" for i in range(1, 6)],
            "final_rank": list(range(1, 6)),
        }
    )

    selected = _select_models_for_delta_plot(
        ranked,
        selection_mode="rank_bands",
        best_n=3,
        middle_n=3,
        worst_n=3,
    )

    assert selected["final_rank"].tolist() == [1, 2, 3, 4, 5]
    assert selected["selection_group"].tolist() == ["best", "best", "best", "middle", "worst"]


def test_select_models_for_delta_plot_marks_top_n_selection_group() -> None:
    ranked = pd.DataFrame(
        {
            "model_name": [f"model_{i}" for i in range(1, 7)],
            "final_rank": list(range(1, 7)),
        }
    )

    selected = _select_models_for_delta_plot(
        ranked,
        selection_mode="top_n",
        top_n=4,
    )

    assert selected["final_rank"].tolist() == [1, 2, 3, 4]
    assert selected["selection_group"].tolist() == ["top", "top", "top", "top"]
