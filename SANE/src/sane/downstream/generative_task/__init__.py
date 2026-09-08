"""Sampling utilities for generative downstream tasks."""

from .kde_samplers import PerTokenKDESampler
from .anchor_utils import encode_anchors, get_arch_signature, group_by_architecture
from .decode_utils import decode_embeddings, sample_and_decode, condition_bn_models
from .eval_utils import evaluate_model

from .reporting_utils import extract_minimal_metrics, _save_sampled_models, save_accuracy_plot, save_plots_for_results

__all__ = [
    # KDE Samplers
    'PerTokenKDESampler',
    
    # Anchor utilities
    'encode_anchors',
    'get_arch_signature',
    'group_by_architecture',
    
    # Decode, conditioning, and evaluation utilities
    'decode_embeddings',
    'sample_and_decode',
    'condition_bn_models',
    'evaluate_model',
    
    # Reporting
    'extract_minimal_metrics',
    '_save_sampled_models',
    'save_accuracy_plot',
    'save_plots_for_results',
]
