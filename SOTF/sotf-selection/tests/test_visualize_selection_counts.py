from __future__ import annotations

import unittest

import pandas as pd

from evaluation import selection_counts as VISUALIZE


class VisualizeSelectionCountsTests(unittest.TestCase):
    def test_selection_count_rows_use_cumulative_killed_models(self) -> None:
        """Verify that selection count rows use cumulative killed models.

        Returns:
            None: None.
        """
        run = {
            "run_dir": "/tmp/sotf-run",
            "experiment_name": "forecast_model_exp5_ds1",
            "predictor": "forecast_model",
            "dataset_number": 1,
            "dataset_label": "1_cifar10_uniform_large",
            "dataset_slug": "1_cifar10_uniform_large",
            "target_column": "test_acc",
            "target_mode": "horizon_value",
            "horizon": 5,
            "selection_start_epoch": 10,
            "selection_interval": 10,
            "selection_survival_rate": 0.5,
            "initial_population": 100,
        }
        round_summary = pd.DataFrame(
            {
                "round_index": [1, 2, 3],
                "decision_epoch": [10.0, 20.0, 30.0],
                "population_before": [100, 50, 25],
                "population_after": [50, 25, 13],
            }
        )

        rows = VISUALIZE.selection_count_rows(run=run, round_summary=round_summary)

        self.assertEqual([row["decision_epoch"] for row in rows], [0.0, 10.0, 20.0, 30.0])
        self.assertEqual([row["kept_models"] for row in rows], [100, 50, 25, 13])
        self.assertEqual([row["killed_models"] for row in rows], [0, 50, 75, 87])
        self.assertEqual([row["total_models"] for row in rows], [100, 100, 100, 100])
        self.assertEqual([row["killed_this_round"] for row in rows], [0, 50, 25, 12])

    def test_summarize_round_details_returns_one_row_per_round(self) -> None:
        """Verify that summarize round details returns one row per round.

        Returns:
            None: None.
        """
        round_details = pd.DataFrame(
            {
                "round_index": [1, 1, 2, 2],
                "decision_epoch": [10.0, 10.0, 20.0, 20.0],
                "population_before": [10, 10, 5, 5],
                "population_after": [5, 5, 3, 3],
            }
        )

        summary = VISUALIZE.summarize_round_details(round_details)

        self.assertEqual(summary.to_dict("records"), [
            {"round_index": 1, "decision_epoch": 10.0, "population_before": 10, "population_after": 5, "kept_models": 5},
            {"round_index": 2, "decision_epoch": 20.0, "population_before": 5, "population_after": 3, "kept_models": 3},
        ])


if __name__ == "__main__":
    unittest.main()
