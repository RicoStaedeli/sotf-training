"""Tests for LatentHierarchicalTransformerRegressor.

Covers:
  - Config parsing (valid params, alias support, validation errors)
  - Positional encoding (even and odd d_model)
  - Network forward pass (flat mode, hierarchical mode, padding correctness)
  - fit / predict round-trip (same-length and variable-length sequences,
    with and without scalar features)
  - Pickle serialisation (getstate / setstate)
  - Factory registration (all accepted name aliases)
  - Integration with the latent forecasting pipeline detection helper
"""
from __future__ import annotations

import pickle
import unittest

import numpy as np
import pandas as pd
import torch

# Import models and data modules first to avoid the circular import between
# forecasting.dataset ↔ data.dataset_splitter that exists in this codebase.
from models.factory import create_regressor
from models.latent_hierarchical_transformer_regressor import (
    LatentHierarchicalTransformerConfig,
    LatentHierarchicalTransformerNetwork,
    LatentHierarchicalTransformerRegressor,
    _SinusoidalPositionalEncoding,
    _parse_config,
)
from runner.pipelines.latent_forecasting import _is_latent_window_sequence_model
from forecasting.dataset import WINDOW_SEQUENCE_COLUMN


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_sequences(
    n: int = 20,
    min_len: int = 3,
    max_len: int = 7,
    embed_dim: int = 8,
    seed: int = 0,
) -> list[np.ndarray]:
    rng = np.random.default_rng(seed)
    lengths = rng.integers(min_len, max_len + 1, size=n)
    return [rng.standard_normal((int(l), embed_dim)).astype(np.float32) for l in lengths]


def _make_dataframe(
    sequences: list[np.ndarray],
    n_scalar: int = 0,
    seed: int = 1,
) -> tuple[pd.DataFrame, pd.Series]:
    rng = np.random.default_rng(seed)
    data: dict[str, object] = {WINDOW_SEQUENCE_COLUMN: sequences}
    for i in range(n_scalar):
        data[f"scalar_{i}"] = rng.random(len(sequences)).astype(np.float32)
    X = pd.DataFrame(data)
    y = pd.Series(rng.random(len(sequences)), dtype=np.float64)
    return X, y


def _tiny_params(**overrides: object) -> dict[str, object]:
    """Return minimal config params that train fast on CPU."""
    base: dict[str, object] = dict(
        d_model=16,
        nhead=2,
        num_local_layers=1,
        num_global_layers=1,
        dim_feedforward=32,
        dropout=0.0,
        head_hidden_layer_sizes=[8],
        batch_size=8,
        max_epochs=2,
        patience=2,
        early_stopping=False,
        validation_fraction=0.2,
        random_state=42,
        device="cpu",
        scale_features=False,
        scale_target=False,
    )
    base.update(overrides)
    return base


# ---------------------------------------------------------------------------
# Config parsing
# ---------------------------------------------------------------------------

class TestConfigParsing(unittest.TestCase):

    def test_defaults_parse_cleanly(self) -> None:
        cfg = _parse_config({})
        self.assertIsInstance(cfg, LatentHierarchicalTransformerConfig)
        self.assertEqual(cfg.d_model, 128)
        self.assertEqual(cfg.nhead, 4)
        self.assertEqual(cfg.local_window_size, 1)

    def test_learning_rate_init_alias(self) -> None:
        cfg = _parse_config({"learning_rate_init": 5e-4})
        self.assertAlmostEqual(cfg.learning_rate, 5e-4)

    def test_alpha_alias_for_weight_decay(self) -> None:
        cfg = _parse_config({"alpha": 0.01})
        self.assertAlmostEqual(cfg.weight_decay, 0.01)

    def test_max_iter_alias_for_max_epochs(self) -> None:
        cfg = _parse_config({"max_iter": 123})
        self.assertEqual(cfg.max_epochs, 123)

    def test_n_iter_no_change_alias_for_patience(self) -> None:
        cfg = _parse_config({"n_iter_no_change": 7})
        self.assertEqual(cfg.patience, 7)

    def test_head_hidden_layer_sizes_scalar_allowed(self) -> None:
        cfg = _parse_config({"head_hidden_layer_sizes": 64})
        self.assertEqual(cfg.head_hidden_layer_sizes, (64,))

    def test_d_model_not_divisible_by_nhead_raises(self) -> None:
        with self.assertRaises(ValueError):
            _parse_config({"d_model": 10, "nhead": 3})

    def test_local_window_size_zero_raises(self) -> None:
        with self.assertRaises(ValueError):
            _parse_config({"local_window_size": 0})

    def test_unknown_param_raises(self) -> None:
        with self.assertRaises(ValueError):
            _parse_config({"totally_unknown_param": 1})

    def test_solver_adam_ok(self) -> None:
        cfg = _parse_config({"solver": "adam"})
        self.assertIsNotNone(cfg)

    def test_solver_non_adam_raises(self) -> None:
        with self.assertRaises(ValueError):
            _parse_config({"solver": "sgd"})


# ---------------------------------------------------------------------------
# Positional encoding
# ---------------------------------------------------------------------------

class TestPositionalEncoding(unittest.TestCase):

    def test_even_d_model(self) -> None:
        pe = _SinusoidalPositionalEncoding(d_model=16)
        x = torch.zeros(2, 5, 16)
        out = pe(x)
        self.assertEqual(out.shape, (2, 5, 16))
        # PE should change the values
        self.assertFalse(torch.all(out == 0).item())

    def test_odd_d_model(self) -> None:
        pe = _SinusoidalPositionalEncoding(d_model=15)
        x = torch.zeros(1, 4, 15)
        out = pe(x)
        self.assertEqual(out.shape, (1, 4, 15))
        self.assertTrue(torch.isfinite(out).all().item())

    def test_longer_sequence_than_default_max_len_raises(self) -> None:
        pe = _SinusoidalPositionalEncoding(d_model=8, max_len=10)
        x = torch.zeros(1, 11, 8)
        with self.assertRaises(Exception):
            pe(x)


# ---------------------------------------------------------------------------
# Network forward pass
# ---------------------------------------------------------------------------

class TestNetworkForward(unittest.TestCase):

    def _make_net(self, local_window_size: int = 1, use_pos_enc: bool = True):
        return LatentHierarchicalTransformerNetwork(
            input_dim=8,
            scalar_dim=2,
            d_model=16,
            nhead=2,
            num_local_layers=1,
            num_global_layers=1,
            local_window_size=local_window_size,
            dim_feedforward=32,
            dropout=0.0,
            head_hidden_layer_sizes=[8],
            activation="gelu",
            use_positional_encoding=use_pos_enc,
        )

    def _batch(self, lengths: list[int], embed_dim: int = 8, scalar_dim: int = 2):
        batch = len(lengths)
        max_len = max(lengths)
        rng = torch.Generator().manual_seed(0)
        seqs = torch.randn(batch, max_len, embed_dim, generator=rng)
        # Zero-pad
        for i, l in enumerate(lengths):
            if l < max_len:
                seqs[i, l:] = 0.0
        scalars = torch.randn(batch, scalar_dim, generator=rng)
        lengths_t = torch.tensor(lengths, dtype=torch.long)
        return {"sequences": seqs, "lengths": lengths_t, "scalar_features": scalars}

    def test_flat_mode_same_length(self) -> None:
        net = self._make_net(local_window_size=1)
        net.eval()
        batch = self._batch([5, 5, 5])
        with torch.no_grad():
            out = net(batch)
        self.assertEqual(out.shape, (3, 1))
        self.assertTrue(torch.isfinite(out).all().item())

    def test_flat_mode_variable_length(self) -> None:
        net = self._make_net(local_window_size=1)
        net.eval()
        batch = self._batch([3, 5, 4])
        with torch.no_grad():
            out = net(batch)
        self.assertEqual(out.shape, (3, 1))
        self.assertTrue(torch.isfinite(out).all().item())

    def test_hierarchical_mode_even_divisor(self) -> None:
        # seq_len=6, local_window_size=3 → 2 windows per sequence
        net = self._make_net(local_window_size=3)
        net.eval()
        batch = self._batch([6, 6, 6])
        with torch.no_grad():
            out = net(batch)
        self.assertEqual(out.shape, (3, 1))
        self.assertTrue(torch.isfinite(out).all().item())

    def test_hierarchical_mode_non_divisor_length(self) -> None:
        # seq_len=7 with local_window_size=3 → pads to 9, giving 3 windows
        net = self._make_net(local_window_size=3)
        net.eval()
        batch = self._batch([7, 5, 6])
        with torch.no_grad():
            out = net(batch)
        self.assertEqual(out.shape, (3, 1))
        self.assertTrue(torch.isfinite(out).all().item())

    def test_window_size_larger_than_seq_len(self) -> None:
        # local_window_size > all sequence lengths → 1 window per sequence
        net = self._make_net(local_window_size=20)
        net.eval()
        batch = self._batch([3, 4, 5])
        with torch.no_grad():
            out = net(batch)
        self.assertEqual(out.shape, (3, 1))
        self.assertTrue(torch.isfinite(out).all().item())

    def test_no_positional_encoding(self) -> None:
        net = self._make_net(local_window_size=2, use_pos_enc=False)
        net.eval()
        batch = self._batch([4, 6])
        with torch.no_grad():
            out = net(batch)
        self.assertEqual(out.shape, (2, 1))

    def test_scalar_dim_zero(self) -> None:
        net = LatentHierarchicalTransformerNetwork(
            input_dim=8, scalar_dim=0,
            d_model=16, nhead=2,
            num_local_layers=1, num_global_layers=1,
            local_window_size=1, dim_feedforward=32,
            dropout=0.0, head_hidden_layer_sizes=[8],
            activation="gelu", use_positional_encoding=True,
        )
        net.eval()
        batch = {
            "sequences": torch.randn(2, 4, 8),
            "lengths": torch.tensor([4, 3]),
            "scalar_features": torch.zeros(2, 0),
        }
        with torch.no_grad():
            out = net(batch)
        self.assertEqual(out.shape, (2, 1))

    def test_padding_does_not_affect_unpadded_prediction(self) -> None:
        """Prediction for a sample must be the same whether it is padded or not."""
        net = self._make_net(local_window_size=1)
        net.eval()
        rng = torch.Generator().manual_seed(7)
        seq = torch.randn(1, 3, 8, generator=rng)
        scalar = torch.randn(1, 2, generator=rng)

        # Batch of 1, no padding
        batch_clean = {
            "sequences": seq,
            "lengths": torch.tensor([3]),
            "scalar_features": scalar,
        }
        # Same sequence zero-padded to length 6
        padded_seq = torch.zeros(1, 6, 8)
        padded_seq[:, :3, :] = seq
        batch_padded = {
            "sequences": padded_seq,
            "lengths": torch.tensor([3]),
            "scalar_features": scalar,
        }
        with torch.no_grad():
            out_clean = net(batch_clean)
            out_padded = net(batch_padded)

        self.assertTrue(torch.allclose(out_clean, out_padded, atol=1e-5))


# ---------------------------------------------------------------------------
# fit / predict
# ---------------------------------------------------------------------------

class TestFitPredict(unittest.TestCase):

    def _model(self, **overrides):
        return LatentHierarchicalTransformerRegressor(**_tiny_params(**overrides))

    def test_fit_predict_same_length_no_scalar(self) -> None:
        seqs = _make_sequences(n=20, min_len=4, max_len=4, embed_dim=6)
        X, y = _make_dataframe(seqs, n_scalar=0)
        model = self._model()
        model.fit(X, y)
        preds = model.predict(X.iloc[:5])
        self.assertEqual(preds.shape, (5,))
        self.assertEqual(preds.dtype, np.float64)
        self.assertTrue(np.isfinite(preds).all())

    def test_fit_predict_variable_length_with_scalars(self) -> None:
        seqs = _make_sequences(n=24, min_len=2, max_len=6, embed_dim=8)
        X, y = _make_dataframe(seqs, n_scalar=2)
        model = self._model()
        model.fit(X, y)
        preds = model.predict(X)
        self.assertEqual(preds.shape, (24,))
        self.assertTrue(np.isfinite(preds).all())

    def test_hierarchical_mode_fit_predict(self) -> None:
        seqs = _make_sequences(n=20, min_len=4, max_len=10, embed_dim=8)
        X, y = _make_dataframe(seqs, n_scalar=1)
        model = self._model(local_window_size=3, num_local_layers=1)
        model.fit(X, y)
        preds = model.predict(X.iloc[:4])
        self.assertEqual(preds.shape, (4,))
        self.assertTrue(np.isfinite(preds).all())

    def test_requires_at_least_two_samples(self) -> None:
        seqs = _make_sequences(n=1, embed_dim=4)
        X, y = _make_dataframe(seqs)
        model = self._model()
        with self.assertRaises(ValueError):
            model.fit(X, y)

    def test_predict_before_fit_raises(self) -> None:
        model = LatentHierarchicalTransformerRegressor(**_tiny_params())
        seqs = _make_sequences(n=4, embed_dim=4)
        X, _ = _make_dataframe(seqs)
        with self.assertRaises(RuntimeError):
            model.predict(X)

    def test_predict_wrong_embedding_dim_raises(self) -> None:
        seqs_train = _make_sequences(n=20, embed_dim=8)
        X_train, y_train = _make_dataframe(seqs_train)
        model = self._model()
        model.fit(X_train, y_train)

        seqs_wrong = _make_sequences(n=4, embed_dim=16)
        X_wrong, _ = _make_dataframe(seqs_wrong)
        with self.assertRaises(ValueError):
            model.predict(X_wrong)

    def test_predict_wrong_scalar_count_raises(self) -> None:
        seqs = _make_sequences(n=20, embed_dim=8)
        X_train, y_train = _make_dataframe(seqs, n_scalar=2)
        model = self._model()
        model.fit(X_train, y_train)

        X_wrong, _ = _make_dataframe(seqs, n_scalar=3)
        with self.assertRaises(ValueError):
            model.predict(X_wrong)

    def test_training_history_recorded(self) -> None:
        seqs = _make_sequences(n=20, embed_dim=6)
        X, y = _make_dataframe(seqs)
        model = self._model(max_epochs=3, early_stopping=False)
        model.fit(X, y)
        self.assertGreater(len(model.training_history_), 0)
        self.assertIn("train_loss", model.training_history_[0])


# ---------------------------------------------------------------------------
# Pickle round-trip
# ---------------------------------------------------------------------------

class TestPickleSerialisation(unittest.TestCase):

    def test_pickle_roundtrip_preserves_predictions(self) -> None:
        seqs = _make_sequences(n=20, embed_dim=8)
        X, y = _make_dataframe(seqs, n_scalar=1)
        model = LatentHierarchicalTransformerRegressor(**_tiny_params())
        model.fit(X, y)

        preds_before = model.predict(X.iloc[:5])
        blob = pickle.dumps(model)
        restored = pickle.loads(blob)
        preds_after = restored.predict(X.iloc[:5])

        np.testing.assert_allclose(preds_before, preds_after, atol=1e-5)

    def test_pickle_roundtrip_hierarchical(self) -> None:
        seqs = _make_sequences(n=20, min_len=4, max_len=9, embed_dim=8)
        X, y = _make_dataframe(seqs)
        model = LatentHierarchicalTransformerRegressor(
            **_tiny_params(local_window_size=3, num_local_layers=1)
        )
        model.fit(X, y)
        blob = pickle.dumps(model)
        restored = pickle.loads(blob)
        preds = restored.predict(X.iloc[:3])
        self.assertTrue(np.isfinite(preds).all())


# ---------------------------------------------------------------------------
# Factory registration
# ---------------------------------------------------------------------------

class TestFactoryRegistration(unittest.TestCase):

    _ACCEPTED_NAMES = [
        "latent_hierarchical_transformer",
        "hierarchical_transformer",
        "latent_window_transformer",
        "latent_windows_transformer",
    ]

    def _base_cfg(self, name: str) -> dict:
        return {"name": name, "params": _tiny_params()}

    def test_all_aliases_build_correct_type(self) -> None:
        for name in self._ACCEPTED_NAMES:
            with self.subTest(name=name):
                model = create_regressor(self._base_cfg(name))
                self.assertIsInstance(model, LatentHierarchicalTransformerRegressor)

    def test_all_aliases_detected_as_window_sequence_model(self) -> None:
        for name in self._ACCEPTED_NAMES:
            with self.subTest(name=name):
                self.assertTrue(
                    _is_latent_window_sequence_model({"name": name}),
                    msg=f"'{name}' should be detected as a window-sequence model",
                )

    def test_lstm_still_detected_as_window_sequence_model(self) -> None:
        for name in ["latent_window_lstm", "lstm_latent_sequences", "latent_windows_lstm"]:
            with self.subTest(name=name):
                self.assertTrue(_is_latent_window_sequence_model({"name": name}))

    def test_unknown_model_not_detected(self) -> None:
        self.assertFalse(_is_latent_window_sequence_model({"name": "ridge"}))
        self.assertFalse(_is_latent_window_sequence_model({"name": "xgboost"}))


if __name__ == "__main__":
    unittest.main()
