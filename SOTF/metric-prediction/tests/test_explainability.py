from __future__ import annotations

import unittest
from types import SimpleNamespace

import pandas as pd

from utils.shap_analysis import _compute_explanation


class _FakeExplainerError(Exception):
    pass


class _FailOnceExplainer:
    def __init__(self, result: object) -> None:
        self._result = result
        self.calls: list[dict[str, object]] = []

    def __call__(self, shap_input: pd.DataFrame, check_additivity: bool = True) -> object:
        self.calls.append(
            {
                "shape": shap_input.shape,
                "check_additivity": check_additivity,
            }
        )
        if check_additivity:
            raise _FakeExplainerError("Additivity check failed in TreeExplainer!")
        return self._result


class ComputeExplanationTests(unittest.TestCase):
    def test_retries_with_additivity_disabled_after_shap_additivity_failure(self) -> None:
        shap_module = SimpleNamespace(
            utils=SimpleNamespace(
                _exceptions=SimpleNamespace(ExplainerError=_FakeExplainerError),
            )
        )
        explainer = _FailOnceExplainer(result={"ok": True})
        shap_input = pd.DataFrame({"feature_a": [1.0, 2.0]})

        explanation = _compute_explanation(shap_module, explainer=explainer, shap_input=shap_input)

        self.assertEqual(explanation, {"ok": True})
        self.assertEqual(
            explainer.calls,
            [
                {"shape": (2, 1), "check_additivity": True},
                {"shape": (2, 1), "check_additivity": False},
            ],
        )


if __name__ == "__main__":
    unittest.main()
