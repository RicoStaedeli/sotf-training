from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd

from evaluation.plots import (
    build_best_survivor_regret_by_experiment,
    plt,
    render_best_survivor_regret_comparison_by_experiment,
)


class BestSurvivorGapByExperimentTests(unittest.TestCase):
    def test_builds_single_experiment_summary_with_mean_regret_per_predictor(self) -> None:
        dataset_summary = pd.DataFrame(
            [
                {
                    "experiment_name": "exp_alpha",
                    "predictor": "forecast_model",
                    "best_survivor_regret": 0.10,
                },
                {
                    "experiment_name": "exp_alpha",
                    "predictor": "forecast_model",
                    "best_survivor_regret": 0.30,
                },
                {
                    "experiment_name": "exp_alpha",
                    "predictor": "naive",
                    "best_survivor_regret": 0.40,
                },
                {
                    "experiment_name": "exp_alpha",
                    "predictor": "naive",
                    "best_survivor_regret": 0.20,
                },
            ]
        )

        aggregated, predictor_order = build_best_survivor_regret_by_experiment(dataset_summary)

        self.assertEqual(predictor_order, ["forecast_model", "naive"])
        self.assertEqual(
            aggregated.loc[:, ["experiment_name", "predictor", "mean_best_survivor_regret", "dataset_count"]].to_dict("records"),
            [
                {
                    "experiment_name": "exp_alpha",
                    "predictor": "forecast_model",
                    "mean_best_survivor_regret": 0.2,
                    "dataset_count": 2,
                },
                {
                    "experiment_name": "exp_alpha",
                    "predictor": "naive",
                    "mean_best_survivor_regret": 0.30000000000000004,
                    "dataset_count": 2,
                },
            ],
        )

    def test_renders_single_plot_across_all_experiments(self) -> None:
        dataset_summary = pd.DataFrame(
            [
                {
                    "experiment_name": "exp_alpha",
                    "predictor": "forecast_model",
                    "dataset_number": 1,
                    "dataset_label": "dataset_1",
                    "best_survivor_regret": 0.10,
                },
                {
                    "experiment_name": "exp_alpha",
                    "predictor": "naive",
                    "dataset_number": 1,
                    "dataset_label": "dataset_1",
                    "best_survivor_regret": 0.25,
                },
                {
                    "experiment_name": "exp_beta",
                    "predictor": "forecast_model",
                    "dataset_number": 2,
                    "dataset_label": "dataset_2",
                    "best_survivor_regret": 0.15,
                },
                {
                    "experiment_name": "exp_beta",
                    "predictor": "naive",
                    "dataset_number": 2,
                    "dataset_label": "dataset_2",
                    "best_survivor_regret": 0.30,
                },
                {
                    "experiment_name": "exp_single_predictor",
                    "predictor": "forecast_model",
                    "dataset_number": 3,
                    "dataset_label": "dataset_3",
                    "best_survivor_regret": 0.05,
                },
            ]
        )

        with tempfile.TemporaryDirectory() as tmp_dir:
            output_path = render_best_survivor_regret_comparison_by_experiment(
                dataset_summary,
                Path(tmp_dir) / "11_best_survivor_gap_by_experiment.png",
                subtitle="test subtitle",
            )

            self.assertIsNotNone(output_path)
            assert output_path is not None
            self.assertEqual(output_path.name, "11_best_survivor_gap_by_experiment.png")
            self.assertTrue(output_path.exists(), str(output_path))

    def test_baseline_uses_distinct_color_and_category_legend(self) -> None:
        # Each experiment maps to one predictor; naive_baseline is the baseline experiment.
        # The plot should colour bars by category (baseline / online / offline) and render
        # a legend that shows only the categories present in the data.
        dataset_summary = pd.DataFrame(
            [
                {
                    "experiment_name": "naive_baseline",
                    "predictor": "naive",
                    "dataset_number": 1,
                    "dataset_label": "dataset_1",
                    "best_survivor_regret": 0.25,
                },
                {
                    "experiment_name": "forecast_experiment",
                    "predictor": "forecast_model",
                    "dataset_number": 1,
                    "dataset_label": "dataset_1",
                    "best_survivor_regret": 0.10,
                },
            ]
        )
        fig = unittest.mock.MagicMock()
        axis = unittest.mock.MagicMock()

        with (
            patch.object(plt, "subplots", return_value=(fig, axis)),
            patch.object(plt, "close"),
        ):
            output_path = render_best_survivor_regret_comparison_by_experiment(
                dataset_summary,
                Path("/tmp/11_best_survivor_gap_by_experiment.png"),
                subtitle="test subtitle",
            )

        self.assertIsNotNone(output_path)
        # A category legend should now be rendered (baseline + offline categories are present)
        axis.legend.assert_called_once()
        legend_call_args = axis.legend.call_args
        legend_labels = legend_call_args.args[1] if legend_call_args.args else []
        self.assertIn("Baseline", legend_labels, "legend should include Baseline category")
        self.assertIn("Offline", legend_labels, "legend should include Offline category")
        self.assertNotIn("Online", legend_labels, "Online category should not appear when only naive/forecast_model are present")

        self.assertEqual(fig.tight_layout.call_args.kwargs["rect"][2], 1)
        colors_used = axis.barh.call_args.kwargs["color"]
        naive_color = "#a8dadc"
        offline_color = "#1d3557"
        self.assertTrue(any(c == naive_color for c in colors_used), "baseline bar should use the naive colour")
        self.assertTrue(any(c == offline_color for c in colors_used), "offline bar should use the offline colour")

    def test_sorts_baseline_first_then_batch_experiment_order(self) -> None:
        dataset_summary = pd.DataFrame(
            [
                {
                    "experiment_name": "baseline",
                    "predictor": "naive",
                    "experiment_order": 0,
                    "dataset_number": 1,
                    "dataset_label": "dataset_1",
                    "best_survivor_regret": 0.40,
                },
                {
                    "experiment_name": "best_method",
                    "predictor": "forecast_model",
                    "experiment_order": 1,
                    "dataset_number": 1,
                    "dataset_label": "dataset_1",
                    "best_survivor_regret": 0.05,
                },
                {
                    "experiment_name": "middle_method",
                    "predictor": "xgboost",
                    "experiment_order": 2,
                    "dataset_number": 1,
                    "dataset_label": "dataset_1",
                    "best_survivor_regret": 0.20,
                },
                {
                    "experiment_name": "worst_method",
                    "predictor": "online_lstm",
                    "experiment_order": 3,
                    "dataset_number": 1,
                    "dataset_label": "dataset_1",
                    "best_survivor_regret": 0.35,
                },
            ]
        )
        fig = unittest.mock.MagicMock()
        axis = unittest.mock.MagicMock()

        with (
            patch.object(plt, "subplots", return_value=(fig, axis)),
            patch.object(plt, "close"),
        ):
            render_best_survivor_regret_comparison_by_experiment(
                dataset_summary,
                Path("/tmp/11_best_survivor_gap_by_experiment.png"),
                subtitle="test subtitle",
            )

        self.assertEqual(
            axis.set_yticks.call_args.args[1],
            ["baseline", "best_method", "middle_method", "worst_method"],
        )


if __name__ == "__main__":
    unittest.main()
