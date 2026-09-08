from __future__ import annotations

import matplotlib.pyplot as plt
import pandas as pd

from nnrun.analyses.acc_curves import _natural_sort_key, run_acc_curves


def test_acc_curves_overview_uses_natural_dataset_order() -> None:
    rows = []
    for dataset_name in [
        "10_stl10_large_hyp_fixed_seeds",
        "1_cifar10_uniform_large",
        "2_cifar10_large_hyp_random_seeds",
    ]:
        for model_name in ["2_model", "1_model"]:
            for epoch in [1, 2]:
                rows.append(
                    {
                        "dataset_name": dataset_name,
                        "model_name": model_name,
                        "training_iteration": epoch,
                        "test_acc": 0.5 + 0.01 * epoch,
                    }
                )
    df = pd.DataFrame(rows)

    result = run_acc_curves(
        df,
        {
            "metric": "test_acc",
            "overlay_models": False,
            "boxplot_curves": True,
            "boxplot_overview": {
                "enabled": True,
                "single_page": {"enabled": True, "ncols": 2},
                "paged": {"enabled": False},
            },
        },
        registry={},
    )

    try:
        figure_names = [next(iter(item.keys())) for item in result.figures]
        assert figure_names[:3] == [
            "1_cifar10_uniform_large_acc_curves_boxplot_by_epoch_test_acc",
            "2_cifar10_large_hyp_random_seeds_acc_curves_boxplot_by_epoch_test_acc",
            "10_stl10_large_hyp_fixed_seeds_acc_curves_boxplot_by_epoch_test_acc",
        ]

        overview_fig = next(iter(result.figures[-1].values()))
        panel_titles = [ax.get_title() for ax in overview_fig.axes if ax.get_title()]
        assert panel_titles == [
            "1_cifar10_uniform_large",
            "2_cifar10_large_hyp_random_seeds",
            "10_stl10_large_hyp_fixed_seeds",
        ]
    finally:
        for item in result.figures:
            plt.close(next(iter(item.values())))


def test_natural_sort_key_orders_numbered_model_names() -> None:
    model_names = ["1_model", "10_model", "11_model", "12_model", "2_model", "3_model"]

    assert sorted(model_names, key=_natural_sort_key) == [
        "1_model",
        "2_model",
        "3_model",
        "10_model",
        "11_model",
        "12_model",
    ]
