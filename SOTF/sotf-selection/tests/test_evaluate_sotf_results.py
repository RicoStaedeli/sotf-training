from __future__ import annotations

from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from matplotlib.colors import Normalize
import numpy as np
import pandas as pd

SRC_ROOT = Path(__file__).resolve().parents[1] / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from evaluation import analysis as ANALYSIS  # noqa: E402
from evaluation import cli as CLI  # noqa: E402
from evaluation import plots as PLOTS  # noqa: E402
from evaluation import shared as SHARED  # noqa: E402
from evaluation import tracking as TRACKING  # noqa: E402


EVALUATE = SimpleNamespace(
    build_experiment_subtitle=SHARED.build_experiment_subtitle,
    build_model_zoo_subtitle=SHARED.build_model_zoo_subtitle,
    build_predictor_comparison=ANALYSIS.build_predictor_comparison,
    build_predictor_summary=ANALYSIS.build_predictor_summary,
    build_round_metric_table=SHARED.build_round_metric_table,
    compute_final_epoch_oracle_labels=ANALYSIS.compute_final_epoch_oracle_labels,
    compute_prediction_metric=SHARED.compute_prediction_metric,
    compute_round_metrics=ANALYSIS.compute_round_metrics,
    compute_run_roc_curve=ANALYSIS.compute_run_roc_curve,
    parse_experiment_path_specs=ANALYSIS.parse_experiment_path_specs,
    plt=PLOTS.plt,
    regret_against_oracle=SHARED.regret_against_oracle,
    remove_obsolete_plots=CLI.remove_obsolete_plots,
    render_best_survivor_regret_comparison=PLOTS.render_best_survivor_regret_comparison,
    render_dataset_small_multiples=PLOTS.render_dataset_small_multiples,
    render_metric_comparison_overview=PLOTS.render_metric_comparison_overview,
    render_prediction_quality_heatmaps=PLOTS.render_prediction_quality_heatmaps,
    render_predictor_lineplot=PLOTS.render_predictor_lineplot,
    render_roc_curve_comparison_by_dataset=PLOTS.render_roc_curve_comparison_by_dataset,
    render_roc_curve_overview=PLOTS.render_roc_curve_overview,
    render_roc_curves_by_dataset=PLOTS.render_roc_curves_by_dataset,
    render_sotf_parameter_sensitivity=PLOTS.render_sotf_parameter_sensitivity,
    render_sotf_parameter_sensitivity_delta=PLOTS.render_sotf_parameter_sensitivity_delta,
    shared_remaining_population_trace=SHARED.shared_remaining_population_trace,
)


class EvaluateSotfResultsTests(unittest.TestCase):
    def test_create_evaluation_context_uses_versioned_output_dir(self) -> None:
        """Verify that create evaluation context uses a versioned output directory.

        Returns:
            None: None.
        """
        with tempfile.TemporaryDirectory() as tmp_dir:
            source_root = Path(tmp_dir) / "runs"
            source_root.mkdir()

            context = TRACKING.create_evaluation_context(
                name="SOTF DS1 Overview",
                source_roots=[source_root],
                requested_output_dir=None,
                filters={"predictors": "all"},
            )

            self.assertTrue(context.output_dir.exists())
            self.assertEqual(context.output_root, (source_root / "evaluation").resolve())
            self.assertIn(context.evaluation_id, context.output_dir.name)
            self.assertTrue(context.output_dir.name.startswith("sotf_ds1_overview_"))

    def test_regret_against_oracle_for_maximize_metric_is_oracle_minus_survivor(self) -> None:
        """Verify that regret against oracle for maximize metric is oracle minus survivor.

        Returns:
            None: None.
        """
        regret = EVALUATE.regret_against_oracle(metric_goal="maximize", oracle_value=0.80, survivor_value=0.79)
        self.assertAlmostEqual(regret, 0.01)

    def test_regret_against_oracle_for_minimize_metric_is_survivor_minus_oracle(self) -> None:
        """Verify that regret against oracle for minimize metric is survivor minus oracle.

        Returns:
            None: None.
        """
        regret = EVALUATE.regret_against_oracle(metric_goal="minimize", oracle_value=0.20, survivor_value=0.21)
        self.assertAlmostEqual(regret, 0.01)

    def test_remove_obsolete_plots_deletes_old_plot_04_and_10_outputs(self) -> None:
        """Verify that remove obsolete plots deletes old plot 04 and 10 outputs.

        Returns:
            None: None.
        """
        with tempfile.TemporaryDirectory() as tmp_dir:
            output_dir = Path(tmp_dir)
            old_plot_04 = output_dir / "04_forecast_model_best_survivor_gap.png"
            old_plot_10 = output_dir / "10_predictor_comparison.png"
            old_plot_04.write_text("stale", encoding="utf-8")
            old_plot_10.write_text("stale", encoding="utf-8")

            EVALUATE.remove_obsolete_plots(output_dir)

            self.assertFalse(old_plot_04.exists())
            self.assertFalse(old_plot_10.exists())

    def test_build_experiment_subtitle_uses_metadata_columns(self) -> None:
        """Verify that build experiment subtitle uses metadata columns.

        Returns:
            None: None.
        """
        frame = pd.DataFrame(
            {
                "target_mode": ["horizon_value", "horizon_value"],
                "horizon": [5, 5],
                "selection_start_epoch": [10, 10],
                "selection_interval": [10, 10],
                "selection_survival_rate": [0.25, 0.25],
            }
        )

        subtitle = EVALUATE.build_experiment_subtitle(frame)

        self.assertEqual(
            subtitle,
            "Forecast horizon: 5 | Start epoch: 10 | Decision interval: 10 | Survival rate: 0.25",
        )

    def test_build_experiment_subtitle_uses_final_value_label(self) -> None:
        """Verify that build experiment subtitle uses a final-value label.

        Returns:
            None: None.
        """
        frame = pd.DataFrame(
            {
                "target_mode": ["final_value", "final_value"],
                "horizon": [5, 5],
                "selection_start_epoch": [10, 10],
                "selection_interval": [10, 10],
                "selection_survival_rate": [0.25, 0.25],
            }
        )

        subtitle = EVALUATE.build_experiment_subtitle(frame)

        self.assertEqual(
            subtitle,
            "Final value prediction | Start epoch: 10 | Decision interval: 10 | Survival rate: 0.25",
        )

    def test_render_sotf_parameter_sensitivity_writes_overview_pdf(self) -> None:
        """Verify that parameter sensitivity writes a multi-experiment overview PDF.

        Returns:
            None: None.
        """
        rows = []
        for experiment_name, predictor, offset in [
            ("forecast_a", "forecast_model", 0.00),
            ("forecast_b", "forecast_model", 0.01),
        ]:
            for start_epoch in [5, 10]:
                for survival_rate in [0.5, 0.75]:
                    rows.append(
                        {
                            "experiment_name": experiment_name,
                            "predictor": predictor,
                            "selection_start_epoch": start_epoch,
                            "selection_interval": 5,
                            "selection_survival_rate": survival_rate,
                            "best_survivor_regret": offset + start_epoch / 1000 + survival_rate / 100,
                        }
                    )
        summary = pd.DataFrame(rows)

        with tempfile.TemporaryDirectory() as tmp_dir:
            output_dir = Path(tmp_dir) / "png"
            paths = EVALUATE.render_sotf_parameter_sensitivity(
                summary,
                output_dir,
                metrics=["best_survivor_regret"],
                subtitle="subtitle",
            )

            overview_path = Path(tmp_dir) / "pdf" / "20_parameter_sensitivity_overview.pdf"
            self.assertTrue((output_dir / "20_parameter_sensitivity_forecast_a.png").exists())
            self.assertTrue((output_dir / "20_parameter_sensitivity_forecast_b.png").exists())
            self.assertTrue(overview_path.exists())
            self.assertIn(overview_path, paths)

    def test_render_sotf_parameter_sensitivity_delta_writes_overview_pdf(self) -> None:
        """Verify that delta sensitivity writes a multi-experiment overview PDF.

        Returns:
            None: None.
        """
        rows = []
        for experiment_name, predictor, offset in [
            ("naive_baseline", "naive", 0.00),
            ("xgboost_model", "xgboost", -0.01),
            ("forecast_model", "forecast_model", -0.02),
        ]:
            for start_epoch in [5, 10]:
                for survival_rate in [0.5, 0.75]:
                    rows.append(
                        {
                            "experiment_name": experiment_name,
                            "predictor": predictor,
                            "selection_start_epoch": start_epoch,
                            "selection_interval": 5,
                            "selection_survival_rate": survival_rate,
                            "best_survivor_regret": offset + start_epoch / 1000 + survival_rate / 100,
                        }
                    )
        summary = pd.DataFrame(rows)

        with tempfile.TemporaryDirectory() as tmp_dir:
            output_dir = Path(tmp_dir) / "png"
            paths = EVALUATE.render_sotf_parameter_sensitivity_delta(
                summary,
                output_dir,
                metrics=["best_survivor_regret"],
                subtitle="subtitle",
            )

            overview_path = Path(tmp_dir) / "pdf" / "21_parameter_sensitivity_delta_overview.pdf"
            self.assertFalse((output_dir / "21_parameter_sensitivity_delta_naive_baseline.png").exists())
            self.assertTrue((output_dir / "21_parameter_sensitivity_delta_xgboost_model.png").exists())
            self.assertTrue((output_dir / "21_parameter_sensitivity_delta_forecast_model.png").exists())
            self.assertTrue(overview_path.exists())
            self.assertIn(overview_path, paths)

    def test_build_model_zoo_subtitle_reads_metadata_from_json(self) -> None:
        """Verify that build model-zoo subtitle resolves metadata for known datasets.

        Returns:
            None: None.
        """
        frame = pd.DataFrame(
            {
                "dataset_number": [2, 12],
                "dataset_slug": [
                    "2_cifar10_large_hyp_random_seeds",
                    "12_phase_resnet18_tinyimagenet",
                ],
                "dataset_label": [
                    "2_cifar10_large_hyp_random_seeds",
                    "12_phase_resnet18_tinyimagenet",
                ],
            }
        )

        with tempfile.TemporaryDirectory() as tmp_dir:
            info_path = Path(tmp_dir) / "model_zoo_informations.json"
            info_path.write_text(
                (
                    "["
                    "{\"zoo_name\": \"2_cifar10_large_hyp_random_seeds\", \"model_type\": \"CNN3\", \"dataset\": \"cifar10\", \"Initialization\": \"Hyperparameter Search\"},"
                    "{\"zoo_name\": \"12_phase_resnet18_tinyimagenet\", \"model_type\": \"ResNet18\", \"dataset\": \"tinyimagenet\", \"Initialization\": \"Phase Transitions\"}"
                    "]"
                ),
                encoding="utf-8",
            )

            subtitle = EVALUATE.build_model_zoo_subtitle(frame, info_path=info_path)

        self.assertEqual(
            subtitle,
            "Model zoo info: 2_cifar10_large_hyp_random_seeds: model=CNN3, data=cifar10, init=Hyperparameter Search; 12_phase_resnet18_tinyimagenet: model=ResNet18, data=tinyimagenet, init=Phase Transitions",
        )

    def test_heatmap_annotation_color_switches_for_readability(self) -> None:
        """Verify that heatmap annotation text switches on dark cell colours.

        Returns:
            None: None.
        """
        cmap = PLOTS.plt.get_cmap("viridis")
        norm = Normalize(vmin=0.0, vmax=1.0)

        self.assertEqual(PLOTS._heatmap_annotation_color(0.0, cmap=cmap, norm=norm), "white")
        self.assertEqual(PLOTS._heatmap_annotation_color(1.0, cmap=cmap, norm=norm), "black")

    def test_parse_experiment_path_specs_supports_predictor_path_pairs(self) -> None:
        """Verify that parse experiment path specs supports predictor path pairs.

        Returns:
            None: None.
        """
        specs = EVALUATE.parse_experiment_path_specs(
            ["forecast=/tmp/forecast_runs", "naive=/tmp/naive_runs", "xgboost=/tmp/xgb_runs"]
        )

        self.assertEqual(specs[0][0], "forecast_model")
        self.assertEqual(specs[1][0], "naive")
        self.assertEqual(specs[2][0], "xgboost")
        self.assertEqual(specs[0][1], Path("/tmp/forecast_runs").resolve())

    def test_parse_experiment_path_specs_rejects_duplicate_predictors(self) -> None:
        """Verify that parse experiment path specs rejects duplicate predictors.

        Returns:
            None: None.
        """
        with self.assertRaises(SystemExit):
            EVALUATE.parse_experiment_path_specs(
                ["forecast_model=/tmp/forecast_a", "forecast=/tmp/forecast_b"]
            )

    def test_load_batch_experiment_order_reads_copied_batch_yaml(self) -> None:
        """Verify that evaluation recovers experiment order from batch_files YAML."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            experiment_root = Path(tmp_dir) / "example_batch"
            batch_files = experiment_root / "batch_files"
            batch_files.mkdir(parents=True)
            (batch_files / "example.yaml").write_text(
                "\n".join(
                    [
                        "batch_name: example",
                        "runs:",
                        "  - experiment_name: naive_baseline",
                        "  - experiment_name: xgboost_weight_statistics",
                        "  - experiment_name: online_lstm_latent_window_representations",
                    ]
                ),
                encoding="utf-8",
            )

            order = CLI.load_batch_experiment_order([experiment_root / "results"])

        self.assertEqual(
            order,
            {
                "naive_baseline": 0,
                "xgboost_weight_statistics": 1,
                "online_lstm_latent_window_representations": 2,
            },
        )

    def test_shared_remaining_population_trace_returns_single_line_when_identical(self) -> None:
        """Verify that shared remaining population trace returns single line when identical.

        Returns:
            None: None.
        """
        dataset_df = pd.DataFrame(
            {
                "predictor": ["forecast_model", "naive", "forecast_model", "naive"],
                "decision_epoch": [10, 10, 20, 20],
                "policy_keep_count": [50, 50, 25, 25],
            }
        )

        trace, warning = EVALUATE.shared_remaining_population_trace(dataset_df)

        self.assertIsNone(warning)
        self.assertEqual(trace.to_dict("records"), [{"decision_epoch": 10, "policy_keep_count": 50}, {"decision_epoch": 20, "policy_keep_count": 25}])

    def test_shared_remaining_population_trace_warns_when_predictors_disagree(self) -> None:
        """Verify that shared remaining population trace warns when predictors disagree.

        Returns:
            None: None.
        """
        dataset_df = pd.DataFrame(
            {
                "predictor": ["forecast_model", "naive", "forecast_model", "naive"],
                "decision_epoch": [10, 10, 20, 20],
                "policy_keep_count": [50, 50, 25, 24],
            }
        )

        trace, warning = EVALUATE.shared_remaining_population_trace(dataset_df)

        self.assertEqual(trace.to_dict("records"), [{"decision_epoch": 10, "policy_keep_count": 50}, {"decision_epoch": 20, "policy_keep_count": 25}])
        self.assertEqual(warning, "Warning: remaining model counts differ between predictors")

    def test_render_metric_comparison_overview_uses_four_columns(self) -> None:
        """Verify that render metric comparison overview uses four columns.

        Returns:
            None: None.
        """
        round_metrics = pd.DataFrame(
            {
                "predictor": ["forecast_model", "naive"] * 12,
                "dataset_label": [f"DS{i}" for i in range(1, 13) for _ in range(2)],
                "dataset_number": [i for i in range(1, 13) for _ in range(2)],
                "round_index": [1] * 24,
                "decision_epoch": [10] * 24,
                "decision_accuracy": [0.8] * 24,
                "policy_keep_count": [50] * 24,
            }
        )
        dataset_summary = pd.DataFrame(
            {
                "predictor": ["forecast_model", "naive"] * 12,
                "dataset_label": [f"DS{i}" for i in range(1, 13) for _ in range(2)],
                "dataset_number": [i for i in range(1, 13) for _ in range(2)],
            }
        )
        axes = np.array([[object() for _ in range(4)] for _ in range(3)], dtype=object)
        fig = unittest.mock.MagicMock()

        with (
            patch.object(EVALUATE.plt, "subplots", return_value=(fig, axes)) as subplots_mock,
            patch.object(PLOTS, "render_predictor_lineplot"),
            patch.object(EVALUATE.plt, "close"),
        ):
            EVALUATE.render_metric_comparison_overview(
                round_metrics=round_metrics,
                dataset_summary=dataset_summary,
                metric_column="decision_accuracy",
                metric_label="Decision Accuracy",
                y_limits=(0.0, 1.05),
                output_path=Path("/tmp/decision_accuracy_comparison_overview.png"),
                subtitle="test subtitle",
            )

        subplots_mock.assert_called_once()
        self.assertEqual(subplots_mock.call_args.args[:2], (3, 4))

    def test_render_metric_comparison_overview_labels_remaining_models_on_right_edge(self) -> None:
        """Verify that render metric comparison overview labels remaining models on right edge.

        Returns:
            None: None.
        """
        round_metrics = pd.DataFrame(
            {
                "predictor": ["forecast_model", "naive"] * 5,
                "dataset_label": [f"DS{i}" for i in range(1, 6) for _ in range(2)],
                "dataset_number": [i for i in range(1, 6) for _ in range(2)],
                "round_index": [1] * 10,
                "decision_epoch": [10] * 10,
                "decision_accuracy": [0.8] * 10,
                "policy_keep_count": [50] * 10,
            }
        )
        dataset_summary = pd.DataFrame(
            {
                "predictor": ["forecast_model", "naive"] * 5,
                "dataset_label": [f"DS{i}" for i in range(1, 6) for _ in range(2)],
                "dataset_number": [i for i in range(1, 6) for _ in range(2)],
            }
        )
        axes = np.empty((2, 4), dtype=object)
        for row_index in range(2):
            for column_index in range(4):
                axes[row_index, column_index] = unittest.mock.MagicMock()
        fig = unittest.mock.MagicMock()

        with (
            patch.object(EVALUATE.plt, "subplots", return_value=(fig, axes)),
            patch.object(PLOTS, "render_predictor_lineplot") as lineplot_mock,
            patch.object(EVALUATE.plt, "close"),
        ):
            EVALUATE.render_metric_comparison_overview(
                round_metrics=round_metrics,
                dataset_summary=dataset_summary,
                metric_column="decision_accuracy",
                metric_label="Decision Accuracy",
                y_limits=(0.0, 1.05),
                output_path=Path("/tmp/decision_accuracy_comparison_overview.png"),
                subtitle="test subtitle",
            )

        metric_label_flags = [call.kwargs["show_metric_axis_label"] for call in lineplot_mock.call_args_list]
        remaining_label_flags = [call.kwargs["show_remaining_axis_label"] for call in lineplot_mock.call_args_list]
        self.assertEqual(metric_label_flags, [True, False, False, False, True])
        self.assertEqual(remaining_label_flags, [False, False, False, True, True])

    def test_render_metric_comparison_overview_supports_xgboost_vs_naive(self) -> None:
        """Verify that render metric comparison overview supports xgboost vs naive.

        Returns:
            None: None.
        """
        round_metrics = pd.DataFrame(
            {
                "predictor": ["xgboost", "naive"] * 4,
                "dataset_label": [f"DS{i}" for i in range(1, 5) for _ in range(2)],
                "dataset_number": [i for i in range(1, 5) for _ in range(2)],
                "round_index": [1] * 8,
                "decision_epoch": [15] * 8,
                "decision_accuracy": [0.8] * 8,
                "policy_keep_count": [50] * 8,
            }
        )
        dataset_summary = pd.DataFrame(
            {
                "predictor": ["xgboost", "naive"] * 4,
                "dataset_label": [f"DS{i}" for i in range(1, 5) for _ in range(2)],
                "dataset_number": [i for i in range(1, 5) for _ in range(2)],
            }
        )
        axes = np.array([[object() for _ in range(4)]], dtype=object)
        fig = unittest.mock.MagicMock()

        with (
            patch.object(EVALUATE.plt, "subplots", return_value=(fig, axes)),
            patch.object(PLOTS, "render_predictor_lineplot"),
            patch.object(EVALUATE.plt, "close"),
        ):
            EVALUATE.render_metric_comparison_overview(
                round_metrics=round_metrics,
                dataset_summary=dataset_summary,
                metric_column="decision_accuracy",
                metric_label="Decision Accuracy",
                y_limits=(0.0, 1.05),
                output_path=Path("/tmp/decision_accuracy_comparison_overview.png"),
                subtitle="test subtitle",
            )

        self.assertEqual(fig.savefig.call_count, 2)
        saved_paths = [call.args[0] for call in fig.savefig.call_args_list]
        self.assertEqual(saved_paths[0], Path("/tmp/decision_accuracy_comparison_overview.png"))
        self.assertEqual(saved_paths[1], Path("/tmp/pdf/decision_accuracy_comparison_overview.pdf"))

    def test_render_predictor_lineplot_draws_remaining_population_drop_at_decision_epoch(self) -> None:
        """Verify that render predictor lineplot draws remaining population drop at decision epoch.

        Returns:
            None: None.
        """
        axis = unittest.mock.MagicMock()
        remaining_axis = unittest.mock.MagicMock()
        axis.twinx.return_value = remaining_axis
        dataset_df = pd.DataFrame(
            {
                "predictor": ["forecast_model", "naive", "forecast_model", "naive"],
                "decision_epoch": [10, 10, 20, 20],
                "decision_accuracy": [0.8, 0.7, 0.85, 0.75],
                "policy_keep_count": [50, 50, 25, 25],
            }
        )

        EVALUATE.render_predictor_lineplot(
            axis=axis,
            dataset_df=dataset_df,
            metric_column="decision_accuracy",
            metric_label="Decision Accuracy",
            y_limits=(0.0, 1.05),
            title="DS1",
            show_legend=False,
            subtitle="subtitle",
            show_metric_axis_label=True,
            show_remaining_axis_label=True,
        )

        self.assertEqual(remaining_axis.step.call_args.kwargs["where"], "post")
        axis.set_title.assert_called_once_with("DS1", pad=26)
        self.assertEqual(axis.text.call_args_list[0].args[1], 1.005)

    def test_render_predictor_lineplot_supports_three_predictors(self) -> None:
        """Verify that render predictor lineplot supports three predictors.

        Returns:
            None: None.
        """
        axis = unittest.mock.MagicMock()
        remaining_axis = unittest.mock.MagicMock()
        axis.twinx.return_value = remaining_axis
        dataset_df = pd.DataFrame(
            {
                "predictor": [
                    "forecast_model",
                    "xgboost",
                    "naive",
                    "forecast_model",
                    "xgboost",
                    "naive",
                ],
                "decision_epoch": [10, 10, 10, 20, 20, 20],
                "decision_accuracy": [0.82, 0.78, 0.72, 0.86, 0.8, 0.74],
                "policy_keep_count": [50, 50, 50, 25, 25, 25],
            }
        )

        EVALUATE.render_predictor_lineplot(
            axis=axis,
            dataset_df=dataset_df,
            metric_column="decision_accuracy",
            metric_label="Decision Accuracy",
            y_limits=(0.0, 1.05),
            title="DS1",
            show_legend=False,
            subtitle="subtitle",
            show_metric_axis_label=True,
            show_remaining_axis_label=True,
        )

        self.assertEqual(axis.plot.call_count, 3)

    def test_render_dataset_small_multiples_draws_remaining_population_drop_at_decision_epoch(self) -> None:
        """Verify that render dataset small multiples draws remaining population drop at decision epoch.

        Returns:
            None: None.
        """
        round_metrics = pd.DataFrame(
            {
                "dataset_label": ["DS1", "DS1"],
                "round_index": [1, 2],
                "decision_epoch": [10, 20],
                "decision_accuracy": [0.8, 0.9],
                "false_kill_rate": [0.1, 0.2],
                "false_keep_rate": [0.05, 0.1],
                "policy_keep_count": [50, 25],
            }
        )
        dataset_summary = pd.DataFrame(
            {
                "dataset_label": ["DS1"],
                "dataset_number": [1],
                "oracle_top_k_recall": [0.9],
                "best_survivor_full_rank": [3],
            }
        )
        axis = unittest.mock.MagicMock()
        remaining_axis = unittest.mock.MagicMock()
        axis.twinx.return_value = remaining_axis
        fig = unittest.mock.MagicMock()
        axes = np.empty((1, 3), dtype=object)
        axes[0, 0] = axis
        axes[0, 1] = unittest.mock.MagicMock()
        axes[0, 2] = unittest.mock.MagicMock()

        with (
            patch.object(EVALUATE.plt, "subplots", return_value=(fig, axes)),
            patch.object(EVALUATE.plt, "close"),
        ):
            EVALUATE.render_dataset_small_multiples(
                round_metrics,
                dataset_summary,
                Path("/tmp/per_dataset_round_quality.png"),
                title="SOTF",
                subtitle="subtitle",
            )

        self.assertEqual(remaining_axis.step.call_args.kwargs["where"], "post")

    def test_render_best_survivor_regret_comparison_writes_plot(self) -> None:
        """Verify that render best survivor regret comparison writes plot.

        Returns:
            None: None.
        """
        comparison_df = pd.DataFrame(
            {
                "dataset_label": ["DS1", "DS2"],
                "dataset_number": [1, 2],
                "comparison_predictor": ["xgboost", "xgboost"],
                "baseline_predictor": ["naive", "naive"],
                "comparison_best_survivor_regret": [0.01, 0.02],
                "baseline_best_survivor_regret": [0.03, 0.04],
            }
        )

        with tempfile.TemporaryDirectory() as tmp_dir:
            output_path = Path(tmp_dir) / "best_survivor_gap.png"
            EVALUATE.render_best_survivor_regret_comparison(
                comparison_df,
                output_path,
                subtitle="subtitle",
            )

            self.assertTrue(output_path.exists())

    def test_render_best_survivor_regret_comparison_uses_dataset_on_y_axis(self) -> None:
        """Verify that render best survivor regret comparison uses dataset on y axis.

        Returns:
            None: None.
        """
        comparison_df = pd.DataFrame(
            {
                "dataset_label": ["DS1", "DS2"],
                "dataset_number": [1, 2],
                "comparison_predictor": ["xgboost", "xgboost"],
                "baseline_predictor": ["naive", "naive"],
                "comparison_best_survivor_regret": [0.01, 0.02],
                "baseline_best_survivor_regret": [0.03, 0.04],
            }
        )
        fig = unittest.mock.MagicMock()
        axis = unittest.mock.MagicMock()

        with (
            patch.object(EVALUATE.plt, "subplots", return_value=(fig, axis)),
            patch.object(EVALUATE.plt, "close"),
        ):
            EVALUATE.render_best_survivor_regret_comparison(
                comparison_df,
                Path("/tmp/best_survivor_gap.png"),
                subtitle="subtitle",
            )

        self.assertEqual(axis.barh.call_count, 2)
        axis.set_xlabel.assert_called_once_with("Gap to true best")
        axis.set_ylabel.assert_called_once_with("Dataset")
        axis.set_yticks.assert_called_once()
        axis.invert_yaxis.assert_called_once_with()
        axis.legend.assert_called_once_with(frameon=False, loc="lower right")
        self.assertEqual(fig.tight_layout.call_args.kwargs["rect"][2], 1)
        self.assertAlmostEqual(axis.set_ylim.call_args.args[0], 3.6)
        self.assertAlmostEqual(axis.set_ylim.call_args.args[1], -0.8)

    def test_compute_prediction_metric_supports_r2_and_rmse(self) -> None:
        """Verify that compute prediction metric supports r2 and rmse.

        Returns:
            None: None.
        """
        y_true = pd.Series([1.0, 2.0, 3.0])
        y_pred = pd.Series([1.0, 2.0, 2.0])

        r2 = EVALUATE.compute_prediction_metric(y_true, y_pred, metric_name="r2")
        rmse = EVALUATE.compute_prediction_metric(y_true, y_pred, metric_name="rmse")

        self.assertAlmostEqual(r2, 0.5)
        self.assertAlmostEqual(rmse, np.sqrt(1.0 / 3.0))

    def test_compute_prediction_metric_returns_nan_for_r2_with_single_row(self) -> None:
        """Verify that compute prediction metric returns nan for r2 with single row.

        Returns:
            None: None.
        """
        r2 = EVALUATE.compute_prediction_metric(
            pd.Series([1.0]),
            pd.Series([1.0]),
            metric_name="r2",
        )

        self.assertTrue(np.isnan(r2))

    def test_compute_round_metrics_includes_keep_precision_recall_and_roc_auc(self) -> None:
        """Verify that compute round metrics includes keep precision recall and roc auc.

        Returns:
            None: None.
        """
        run = {
            "experiment_name": "exp",
            "predictor": "xgboost",
            "dataset_number": 1,
            "dataset_label": "DS1",
            "dataset_slug": "ds1",
            "target_column": "test_acc",
            "target_mode": "horizon_value",
            "horizon": 5,
            "selection_start_epoch": 10,
            "selection_interval": 10,
            "selection_survival_rate": 0.5,
            "metric_goal": "maximize",
        }
        round_details = pd.DataFrame(
            {
                "round_index": [1, 1, 1, 1],
                "decision_epoch": [10, 10, 10, 10],
                "selected": [True, True, False, False],
                "oracle_selected": [True, False, True, False],
                "oracle_best_model": [True, False, False, False],
                "predicted_rank": [1, 2, 3, 4],
                "actual_future_rank": [1, 3, 2, 4],
                "predicted_future_target": [0.9, 0.8, 0.7, 0.1],
                "actual_future_target": [1.0, 0.2, 0.9, 0.0],
            }
        )

        metrics = EVALUATE.compute_round_metrics(run=run, round_details=round_details)
        row = metrics.iloc[0]

        self.assertAlmostEqual(float(row["keep_precision"]), 0.5)
        self.assertAlmostEqual(float(row["keep_recall"]), 0.5)
        self.assertAlmostEqual(float(row["kill_recall"]), 0.5)
        self.assertAlmostEqual(float(row["roc_auc"]), 0.75)

    def test_compute_run_roc_curve_uses_minimize_goal_direction(self) -> None:
        """Verify that compute run roc curve uses minimize goal direction.

        Returns:
            None: None.
        """
        run = {
            "experiment_name": "exp",
            "predictor": "xgboost",
            "dataset_number": 1,
            "dataset_label": "DS1",
            "dataset_slug": "ds1",
            "target_column": "test_loss",
            "target_mode": "horizon_value",
            "horizon": 5,
            "selection_start_epoch": 10,
            "selection_interval": 10,
            "selection_survival_rate": 0.5,
            "metric_goal": "minimize",
        }
        round_details = pd.DataFrame(
            {
                "oracle_selected": [True, True, False, False],
                "predicted_future_target": [0.10, 0.20, 0.80, 0.90],
            }
        )

        curve = EVALUATE.compute_run_roc_curve(run=run, round_details=round_details)

        self.assertFalse(curve.empty)
        self.assertAlmostEqual(float(curve["roc_auc"].iloc[0]), 1.0)

    def test_compute_final_epoch_oracle_labels_selects_top_final_models_per_round(self) -> None:
        """Verify that compute final epoch oracle labels selects top final models per round.

        Returns:
            None: None.
        """
        round_details = pd.DataFrame(
            {
                "round_index": [1, 1, 1, 1, 2, 2],
                "model_id": ["a", "b", "c", "d", "a", "b"],
                "selected": [True, True, False, False, True, False],
                "predicted_future_target": [0.9, 0.8, 0.7, 0.1, 0.5, 0.4],
                "final_target": [0.2, 0.9, 0.8, 0.1, 0.3, 0.7],
            }
        )

        labels = EVALUATE.compute_final_epoch_oracle_labels(round_details, metric_goal="maximize")

        self.assertEqual(labels.astype(bool).tolist(), [False, True, True, False, False, True])

    def test_compute_run_roc_curve_supports_final_epoch_oracle_labels(self) -> None:
        """Verify that compute run roc curve supports final epoch oracle labels.

        Returns:
            None: None.
        """
        run = {
            "experiment_name": "exp",
            "predictor": "xgboost",
            "dataset_number": 1,
            "dataset_label": "DS1",
            "dataset_slug": "ds1",
            "target_column": "test_acc",
            "target_mode": "horizon_value",
            "horizon": 5,
            "selection_start_epoch": 10,
            "selection_interval": 10,
            "selection_survival_rate": 0.5,
            "metric_goal": "maximize",
        }
        round_details = pd.DataFrame(
            {
                "round_index": [1, 1, 1, 1],
                "model_id": ["a", "b", "c", "d"],
                "selected": [True, True, False, False],
                "oracle_selected": [True, False, True, False],
                "predicted_future_target": [0.95, 0.90, 0.30, 0.20],
                "actual_future_target": [0.9, 0.2, 0.8, 0.1],
                "final_target": [0.9, 0.8, 0.1, 0.0],
            }
        )

        horizon_curve = EVALUATE.compute_run_roc_curve(run=run, round_details=round_details)
        final_curve = EVALUATE.compute_run_roc_curve(
            run=run,
            round_details=round_details,
            oracle_label_source="final_epoch",
        )

        self.assertAlmostEqual(float(horizon_curve["roc_auc"].iloc[0]), 0.75)
        self.assertAlmostEqual(float(final_curve["roc_auc"].iloc[0]), 1.0)

    def test_build_round_metric_table_sorts_datasets_and_rounds(self) -> None:
        """Verify that build round metric table sorts datasets and rounds.

        Returns:
            None: None.
        """
        round_metrics = pd.DataFrame(
            {
                "dataset_label": ["DS2", "DS1", "DS2", "DS1"],
                "dataset_number": [2, 1, 2, 1],
                "decision_epoch": [20, 10, 10, 20],
                "round_index": [2, 1, 1, 2],
                "prediction_r2": [0.4, 0.9, 0.5, 0.8],
            }
        )

        table = EVALUATE.build_round_metric_table(round_metrics, metric_column="prediction_r2")

        self.assertEqual(list(table.index), ["DS1", "DS2"])
        self.assertEqual(list(table.columns), [10, 20])
        self.assertAlmostEqual(float(table.loc["DS1", 10]), 0.9)
        self.assertAlmostEqual(float(table.loc["DS2", 20]), 0.4)

    def test_render_prediction_quality_heatmaps_writes_plot(self) -> None:
        """Verify that render prediction quality heatmaps writes plot.

        Returns:
            None: None.
        """
        round_metrics = pd.DataFrame(
            {
                "dataset_label": ["DS1", "DS1", "DS2", "DS2"],
                "dataset_number": [1, 1, 2, 2],
                "round_index": [1, 2, 1, 2],
                "prediction_r2": [0.8, 0.7, 0.6, 0.5],
                "prediction_rmse": [0.10, 0.12, 0.20, 0.22],
            }
        )

        with tempfile.TemporaryDirectory() as tmp_dir:
            output_path = Path(tmp_dir) / "prediction_quality.png"
            EVALUATE.render_prediction_quality_heatmaps(
                round_metrics,
                output_path,
                title="Prediction Quality",
                subtitle="subtitle",
            )

            self.assertTrue(output_path.exists())

    def test_render_roc_curve_overview_writes_plot(self) -> None:
        """Verify that render roc curve overview writes plot.

        Returns:
            None: None.
        """
        roc_curve_points = pd.DataFrame(
            {
                "predictor": ["xgboost", "xgboost", "xgboost"],
                "dataset_label": ["DS1", "DS1", "DS1"],
                "dataset_number": [1, 1, 1],
                "positive_count": [2, 2, 2],
                "negative_count": [2, 2, 2],
                "roc_auc": [1.0, 1.0, 1.0],
                "fpr": [0.0, 0.0, 1.0],
                "tpr": [0.0, 1.0, 1.0],
            }
        )

        with tempfile.TemporaryDirectory() as tmp_dir:
            output_path = Path(tmp_dir) / "roc_curves.png"
            EVALUATE.render_roc_curve_overview(
                roc_curve_points,
                output_path,
                title="ROC",
                subtitle="subtitle",
            )

            self.assertTrue(output_path.exists())

    def test_render_roc_curves_by_dataset_writes_one_plot_per_dataset(self) -> None:
        """Verify that render roc curves by dataset writes one plot per dataset.

        Returns:
            None: None.
        """
        roc_curve_points = pd.DataFrame(
            {
                "predictor": ["xgboost"] * 6,
                "dataset_label": ["DS1", "DS1", "DS1", "DS2", "DS2", "DS2"],
                "dataset_number": [1, 1, 1, 2, 2, 2],
                "positive_count": [2] * 6,
                "negative_count": [2] * 6,
                "roc_auc": [1.0] * 6,
                "fpr": [0.0, 0.0, 1.0, 0.0, 0.5, 1.0],
                "tpr": [0.0, 1.0, 1.0, 0.0, 1.0, 1.0],
            }
        )

        with tempfile.TemporaryDirectory() as tmp_dir:
            output_dir = Path(tmp_dir)
            EVALUATE.render_roc_curves_by_dataset(
                roc_curve_points,
                output_dir,
                title_prefix="ROC",
                subtitle="subtitle",
            )

            self.assertTrue((output_dir / "ds1_roc_curve.png").exists())
            self.assertTrue((output_dir / "ds2_roc_curve.png").exists())

    def test_render_roc_curve_comparison_by_dataset_writes_shared_dataset_plots(self) -> None:
        """Verify that render roc curve comparison by dataset writes shared dataset plots.

        Returns:
            None: None.
        """
        roc_curve_points = pd.DataFrame(
            {
                "predictor": ["xgboost", "xgboost", "xgboost", "naive", "naive", "naive"],
                "dataset_label": ["DS1"] * 6,
                "dataset_number": [1] * 6,
                "positive_count": [2] * 6,
                "negative_count": [2] * 6,
                "roc_auc": [1.0, 1.0, 1.0, 0.75, 0.75, 0.75],
                "fpr": [0.0, 0.0, 1.0, 0.0, 0.5, 1.0],
                "tpr": [0.0, 1.0, 1.0, 0.0, 1.0, 1.0],
            }
        )
        dataset_summary = pd.DataFrame(
            {
                "predictor": ["xgboost", "naive"],
                "dataset_label": ["DS1", "DS1"],
                "dataset_number": [1, 1],
            }
        )

        with tempfile.TemporaryDirectory() as tmp_dir:
            output_dir = Path(tmp_dir)
            EVALUATE.render_roc_curve_comparison_by_dataset(
                roc_curve_points=roc_curve_points,
                dataset_summary=dataset_summary,
                output_dir=output_dir,
                subtitle="subtitle",
            )

            self.assertTrue((output_dir / "ds1_roc_curve_comparison.png").exists())

    def test_build_predictor_summary_includes_prediction_quality_columns(self) -> None:
        """Verify that build predictor summary includes prediction quality columns.

        Returns:
            None: None.
        """
        dataset_summary = pd.DataFrame(
            {
                "predictor": ["xgboost", "xgboost", "naive"],
                "oracle_top_k_recall": [0.8, 0.9, 0.7],
                "decision_accuracy_weighted": [0.75, 0.85, 0.65],
                "keep_precision_weighted": [0.76, 0.86, 0.66],
                "keep_recall_weighted": [0.77, 0.87, 0.67],
                "kill_precision_weighted": [0.78, 0.88, 0.68],
                "kill_recall_weighted": [0.79, 0.89, 0.69],
                "false_kill_rate_weighted": [0.10, 0.05, 0.20],
                "roc_auc_weighted": [0.70, 0.80, np.nan],
                "prediction_r2_weighted": [0.60, 0.70, np.nan],
                "prediction_rmse_weighted": [0.11, 0.09, np.nan],
                "retained_oracle_best": [True, True, False],
                "best_survivor_regret": [0.02, 0.01, 0.04],
                "best_survivor_rank_percentile": [0.9, 0.95, 0.7],
            }
        )

        summary = EVALUATE.build_predictor_summary(dataset_summary)
        xgboost_row = summary.loc[summary["predictor"] == "xgboost"].iloc[0]

        self.assertIn("mean_prediction_r2_weighted", summary.columns)
        self.assertIn("mean_prediction_rmse_weighted", summary.columns)
        self.assertIn("mean_keep_precision_weighted", summary.columns)
        self.assertIn("mean_keep_recall_weighted", summary.columns)
        self.assertIn("mean_roc_auc_weighted", summary.columns)
        self.assertAlmostEqual(float(xgboost_row["mean_prediction_r2_weighted"]), 0.65)
        self.assertAlmostEqual(float(xgboost_row["mean_prediction_rmse_weighted"]), 0.10)
        self.assertAlmostEqual(float(xgboost_row["mean_keep_precision_weighted"]), 0.81)
        self.assertAlmostEqual(float(xgboost_row["mean_keep_recall_weighted"]), 0.82)
        self.assertAlmostEqual(float(xgboost_row["mean_roc_auc_weighted"]), 0.75)

    def test_build_predictor_comparison_supports_xgboost_vs_naive(self) -> None:
        """Verify that build predictor comparison supports xgboost vs naive.

        Returns:
            None: None.
        """
        dataset_summary = pd.DataFrame(
            {
                "predictor": ["xgboost", "naive", "xgboost", "naive"],
                "dataset_number": [1, 1, 2, 2],
                "dataset_label": ["DS1", "DS1", "DS2", "DS2"],
                "dataset_slug": ["ds1", "ds1", "ds2", "ds2"],
                "target_column": ["test_acc"] * 4,
                "target_mode": ["horizon_value"] * 4,
                "horizon": [5] * 4,
                "selection_start_epoch": [15] * 4,
                "selection_interval": [5] * 4,
                "selection_survival_rate": [0.5] * 4,
                "oracle_top_k_recall": [0.8, 0.7, 0.9, 0.6],
                "decision_accuracy_weighted": [0.75, 0.65, 0.85, 0.55],
                "false_kill_rate_weighted": [0.10, 0.20, 0.05, 0.25],
                "best_survivor_regret": [0.01, 0.03, 0.02, 0.04],
                "best_survivor_rank_percentile": [0.9, 0.7, 0.95, 0.6],
            }
        )

        comparison = EVALUATE.build_predictor_comparison(dataset_summary)

        self.assertEqual(list(comparison["comparison_predictor"].unique()), ["xgboost"])
        self.assertEqual(list(comparison["baseline_predictor"].unique()), ["naive"])
        self.assertEqual(list(comparison["dataset_label"]), ["DS1", "DS2"])

    def test_build_predictor_comparison_supports_multiple_predictors_vs_naive(self) -> None:
        """Verify that build predictor comparison supports multiple predictors vs naive.

        Returns:
            None: None.
        """
        dataset_summary = pd.DataFrame(
            {
                "predictor": [
                    "forecast_model",
                    "xgboost",
                    "naive",
                    "forecast_model",
                    "xgboost",
                    "naive",
                ],
                "dataset_number": [1, 1, 1, 2, 2, 2],
                "dataset_label": ["DS1", "DS1", "DS1", "DS2", "DS2", "DS2"],
                "dataset_slug": ["ds1", "ds1", "ds1", "ds2", "ds2", "ds2"],
                "target_column": ["test_acc"] * 6,
                "target_mode": ["horizon_value"] * 6,
                "horizon": [5] * 6,
                "selection_start_epoch": [15] * 6,
                "selection_interval": [5] * 6,
                "selection_survival_rate": [0.5] * 6,
                "oracle_top_k_recall": [0.9, 0.8, 0.7, 0.85, 0.75, 0.65],
                "decision_accuracy_weighted": [0.82, 0.78, 0.70, 0.80, 0.74, 0.66],
                "false_kill_rate_weighted": [0.08, 0.12, 0.20, 0.10, 0.14, 0.22],
                "best_survivor_regret": [0.01, 0.02, 0.03, 0.015, 0.025, 0.035],
                "best_survivor_rank_percentile": [0.95, 0.9, 0.85, 0.94, 0.89, 0.84],
            }
        )

        comparison = EVALUATE.build_predictor_comparison(dataset_summary)

        self.assertEqual(set(comparison["comparison_predictor"]), {"forecast_model", "xgboost"})
        self.assertEqual(set(comparison["baseline_predictor"]), {"naive"})
        self.assertEqual(len(comparison), 4)
        self.assertIn("comparison_oracle_top_k_recall", comparison.columns)
        self.assertIn("baseline_oracle_top_k_recall", comparison.columns)
        self.assertIn("delta_oracle_top_k_recall", comparison.columns)


if __name__ == "__main__":
    unittest.main()
