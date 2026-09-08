from __future__ import annotations

import unittest
from pathlib import Path

from evaluation.tracking import resolve_default_evaluation_root


class EvaluationTrackingTests(unittest.TestCase):
    def test_resolve_default_evaluation_root_uses_top_level_outputs_dir(self) -> None:
        source_root = Path("/tmp/project/sotf-selection/outputs/batch_experiments/exp_15_ds12_train_split/results")
        resolved = resolve_default_evaluation_root(source_root)
        self.assertEqual(
            resolved,
            Path("/tmp/project/sotf-selection/outputs/evaluation").resolve(),
        )

    def test_resolve_default_evaluation_root_falls_back_when_no_outputs_dir_exists(self) -> None:
        source_root = Path("/tmp/project/custom_runs/exp_15")
        resolved = resolve_default_evaluation_root(source_root)
        self.assertEqual(
            resolved,
            Path("/tmp/project/custom_runs/exp_15/evaluation").resolve(),
        )


if __name__ == "__main__":
    unittest.main()
