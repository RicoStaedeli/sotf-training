from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import pandas as pd

from evaluation.plots import (
    build_cross_dataset_best_survivor_regret_bars,
    render_cross_dataset_gap_dominance_matrix,
    render_cross_dataset_gap_parameter_map,
    render_cross_dataset_gap_robustness_scorecard,
    render_cross_dataset_best_survivor_regret_bars,
)


class CrossDatasetBestSurvivorGapBarsTests(unittest.TestCase):
    def test_builds_dataset_lanes_grouped_by_start_category_and_survival_rate(self) -> None:
        pooled = pd.DataFrame(
            [
                {
                    "dataset_name": "ds2",
                    "experiment_name": "forecast_model",
                    "predictor": "forecast_model",
                    "selection_survival_rate": 0.1,
                    "selection_start_epoch": 5,
                    "start_epoch_category": "early",
                    "best_survivor_regret": 0.20,
                },
                {
                    "dataset_name": "ds2",
                    "experiment_name": "forecast_model",
                    "predictor": "forecast_model",
                    "selection_survival_rate": 0.1,
                    "selection_start_epoch": 5,
                    "start_epoch_category": "early",
                    "best_survivor_regret": 0.10,
                },
                {
                    "dataset_name": "ds1",
                    "experiment_name": "forecast_model",
                    "predictor": "forecast_model",
                    "selection_survival_rate": 0.25,
                    "selection_start_epoch": 10,
                    "start_epoch_category": "mid",
                    "best_survivor_regret": 0.30,
                },
                {
                    "dataset_name": "ds10",
                    "experiment_name": "forecast_model",
                    "predictor": "forecast_model",
                    "selection_survival_rate": 0.25,
                    "selection_start_epoch": 25,
                    "start_epoch_category": "end",
                    "best_survivor_regret": 0.40,
                },
            ]
        )

        aggregated, dataset_order, categories, survival_rates = build_cross_dataset_best_survivor_regret_bars(pooled)

        self.assertEqual(dataset_order, ["ds1", "ds2", "ds10"])
        self.assertEqual(categories, ["early", "mid", "end"])
        self.assertEqual(survival_rates, [0.1, 0.25])
        row = aggregated[
            (aggregated["dataset_label"] == "ds2")
            & (aggregated["selection_survival_rate"] == 0.1)
            & (aggregated["start_epoch_category"] == "early")
        ].iloc[0]
        self.assertAlmostEqual(row["mean_best_survivor_regret"], 0.15)
        self.assertEqual(int(row["run_count"]), 2)

    def test_derives_start_mid_end_when_category_column_is_missing(self) -> None:
        pooled = pd.DataFrame(
            [
                {
                    "dataset_name": "ds1",
                    "experiment_name": "forecast_model",
                    "predictor": "forecast_model",
                    "selection_survival_rate": 0.1,
                    "selection_start_epoch": epoch,
                    "best_survivor_regret": regret,
                }
                for epoch, regret in [(5, 0.3), (10, 0.2), (25, 0.1)]
            ]
        )

        aggregated, _, categories, _ = build_cross_dataset_best_survivor_regret_bars(pooled)

        self.assertEqual(categories, ["early", "mid", "end"])
        values_by_category = dict(
            zip(
                aggregated["start_epoch_category"],
                aggregated["mean_best_survivor_regret"],
                strict=True,
            )
        )
        self.assertAlmostEqual(values_by_category["early"], 0.3)
        self.assertAlmostEqual(values_by_category["mid"], 0.2)
        self.assertAlmostEqual(values_by_category["end"], 0.1)

    def test_renders_one_plot_per_experiment_and_overview_pdf(self) -> None:
        rows = []
        for experiment_name in ["forecast_model", "online_xgboost"]:
            for dataset_name in ["ds1", "ds2"]:
                for survival_rate in [0.1, 0.25]:
                    for category, epoch, regret in [
                        ("early", 5, 0.30),
                        ("mid", 10, 0.20),
                        ("end", 25, 0.10),
                    ]:
                        rows.append(
                            {
                                "dataset_name": dataset_name,
                                "experiment_name": experiment_name,
                                "predictor": "forecast_model",
                                "selection_survival_rate": survival_rate,
                                "selection_start_epoch": epoch,
                                "start_epoch_category": category,
                                "best_survivor_regret": regret,
                            }
                        )
        pooled = pd.DataFrame(rows)

        with tempfile.TemporaryDirectory() as tmp_dir:
            output_dir = Path(tmp_dir) / "png"
            paths = render_cross_dataset_best_survivor_regret_bars(
                pooled,
                output_dir,
                subtitle="test subtitle",
            )

            self.assertTrue((output_dir / "32_cross_dataset_best_survivor_gap_forecast_model.png").exists())
            self.assertTrue((output_dir / "32_cross_dataset_best_survivor_gap_online_xgboost.png").exists())
            self.assertTrue((Path(tmp_dir) / "pdf" / "32_cross_dataset_best_survivor_gap_overview.pdf").exists())
            self.assertEqual(len(paths), 3)

    def test_renders_compact_cross_dataset_gap_views(self) -> None:
        rows = []
        for dataset_name, naive_gap, candidate_gap in [
            ("ds1", 0.08, 0.04),
            ("ds2", 0.10, 0.06),
            ("ds3", 0.05, 0.07),
        ]:
            rows.extend(
                [
                    {
                        "dataset_name": dataset_name,
                        "experiment_name": "naive_baseline",
                        "predictor": "naive",
                        "selection_survival_rate": 0.1,
                        "selection_start_epoch": 5,
                        "start_epoch_category": "early",
                        "best_survivor_regret": naive_gap,
                    },
                    {
                        "dataset_name": dataset_name,
                        "experiment_name": "weight_statistics_xgboost_with_perf",
                        "predictor": "forecast_model",
                        "selection_survival_rate": 0.1,
                        "selection_start_epoch": 5,
                        "start_epoch_category": "early",
                        "best_survivor_regret": candidate_gap,
                    },
                ]
            )
        pooled = pd.DataFrame(rows)

        with tempfile.TemporaryDirectory() as tmp_dir:
            output_dir = Path(tmp_dir) / "png"
            dominance_paths = render_cross_dataset_gap_dominance_matrix(pooled, output_dir)
            scorecard_paths = render_cross_dataset_gap_robustness_scorecard(pooled, output_dir)
            parameter_paths = render_cross_dataset_gap_parameter_map(pooled, output_dir)

            self.assertEqual([path.name for path in dominance_paths], ["33_cross_dataset_gap_dominance_matrix.png"])
            self.assertEqual([path.name for path in scorecard_paths], ["34_cross_dataset_gap_robustness_scorecard.png"])
            self.assertEqual([path.name for path in parameter_paths], ["35_cross_dataset_gap_parameter_robustness_map.png"])
            self.assertTrue((Path(tmp_dir) / "cross_dataset_gap_dominance_summary.csv").exists())


if __name__ == "__main__":
    unittest.main()
