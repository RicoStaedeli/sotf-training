"""Tests for the PerTokenKDESampler class."""

import pytest
import torch

from sane.downstream.generative_task import PerTokenKDESampler


def test_kde_sampler_fit():
    """Test that PerTokenKDESampler can be fitted with embeddings."""
    n_models = 20
    n_tokens = 5
    latent_dim = 8
    
    embeddings = [torch.randn(n_tokens, latent_dim) for _ in range(n_models)]
    
    sampler = PerTokenKDESampler(bandwidth=0.5)
    sampler.fit(embeddings)
    
    assert sampler.fitted, "Sampler should be marked as fitted after fit()"
    assert sampler.n_tokens == n_tokens
    assert sampler.latent_dim == latent_dim
    assert len(sampler.kdes) == n_tokens, f"Should have {n_tokens} KDEs, got {len(sampler.kdes)}"


def test_kde_sampler_sample_shape():
    """Test that sample output has correct shape."""
    n_models = 20
    n_tokens = 5
    latent_dim = 8
    num_samples = 10
    
    embeddings = [torch.randn(n_tokens, latent_dim) for _ in range(n_models)]
    
    sampler = PerTokenKDESampler(bandwidth=0.5)
    sampler.fit(embeddings)
    samples = sampler.sample(num_samples)
    
    expected_shape = (num_samples, n_tokens, latent_dim)
    assert samples.shape == expected_shape, f"Expected shape {expected_shape}, got {samples.shape}"


def test_kde_sampler_not_fitted_raises():
    """Test that sampling before fitting raises an error."""
    sampler = PerTokenKDESampler(bandwidth=0.5)
    
    with pytest.raises(RuntimeError, match="Must call fit"):
        sampler.sample(10)


def test_kde_sampler_get_stats():
    """Test that get_stats returns correct information."""
    n_tokens = 5
    latent_dim = 8
    bandwidth = 0.3
    
    embeddings = [torch.randn(n_tokens, latent_dim) for _ in range(20)]
    
    sampler = PerTokenKDESampler(bandwidth=bandwidth)
    sampler.fit(embeddings)
    stats = sampler.get_stats()
    
    assert stats['mode'] == 'per_token'
    assert stats['n_kdes'] == n_tokens
    assert stats['dimensions_per_kde'] == latent_dim
    assert stats['bandwidth'] == bandwidth
    assert stats['n_tokens'] == n_tokens


def test_kde_sampler_filters_nan_embeddings():
    """Test that NaN embeddings are filtered out during fitting."""
    n_tokens = 5
    latent_dim = 8
    
    valid_embeddings = [torch.randn(n_tokens, latent_dim) for _ in range(10)]
    nan_embedding = torch.full((n_tokens, latent_dim), float('nan'))
    
    embeddings = valid_embeddings + [nan_embedding]
    
    sampler = PerTokenKDESampler(bandwidth=0.5)
    sampler.fit(embeddings)  # Should not raise
    
    assert sampler.fitted


def test_kde_sampler_all_nan_raises():
    """Test that all-NaN embeddings raise an error."""
    n_tokens = 5
    latent_dim = 8
    
    nan_embeddings = [torch.full((n_tokens, latent_dim), float('nan')) for _ in range(5)]
    
    sampler = PerTokenKDESampler(bandwidth=0.5)
    
    with pytest.raises(ValueError, match="All embeddings contained NaN"):
        sampler.fit(nan_embeddings)


def test_kde_sampler_inconsistent_shapes_raises():
    """Test that inconsistent embedding shapes raise an error."""
    embeddings = [
        torch.randn(5, 8),
        torch.randn(5, 8),
        torch.randn(6, 8),  # Different n_tokens
    ]
    
    sampler = PerTokenKDESampler(bandwidth=0.5)
    
    with pytest.raises(ValueError, match="same shape"):
        sampler.fit(embeddings)
