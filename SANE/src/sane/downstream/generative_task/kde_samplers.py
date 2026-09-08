"""
KDE Sampler for per-token sampling strategy.

Implements per-token KDE sampling where each token position has its own
independent KDE distribution.
"""

from typing import List
import numpy as np
import torch
from sklearn.neighbors import KernelDensity
from ray import logger


class PerTokenKDESampler:
    """
    Per-token KDE sampler - separate KDE per token position.
    
    How it works:
        1. Fit separate KDE for each token position
        2. Each KDE models [latent_dim]-dimensional distribution
        3. Sample each token independently
        4. Combine to form [N_tokens, latent_dim] vector
    """
    
    def __init__(self, bandwidth: float = 0.2):
        """
        Args:
            bandwidth: KDE bandwidth parameter
        """
        self.bandwidth = bandwidth
        self.fitted = False
        self.kdes = []
        self.n_tokens = None
        self.latent_dim = None
    
    def fit(self, embeddings_list: List[torch.Tensor]) -> None:
        """Fit separate KDE for each token position."""
        # Filter NaN embeddings
        valid_embeddings = [z for z in embeddings_list if not torch.isnan(z).any()]
        
        if len(valid_embeddings) == 0:
            raise ValueError("All embeddings contained NaN!")
        
        # Verify all have same shape
        shapes = [z.shape for z in valid_embeddings]
        if len(set(shapes)) > 1:
            raise ValueError(f"All models must have same shape for per-token KDE. Found: {set(shapes)}")
        
        self.n_tokens = valid_embeddings[0].shape[0]
        self.latent_dim = valid_embeddings[0].shape[1]
        
        # Fit KDE for each token position
        self.kdes = []
        for token_idx in range(self.n_tokens):
            # Extract embeddings for this token across all models
            token_embeddings = np.stack([z[token_idx].numpy() for z in valid_embeddings])
            
            kde = KernelDensity(kernel='gaussian', bandwidth=self.bandwidth)
            kde.fit(token_embeddings)
            self.kdes.append(kde)
            
        self.fitted = True
        logger.info(f"Fitted {len(self.kdes)} per-token KDEs (bandwidth={self.bandwidth})")
    
    def sample(self, num_samples: int) -> torch.Tensor:
        """Sample new embeddings using per-token KDEs."""
        if not self.fitted:
            raise RuntimeError("Must call fit() before sample()")
        
        # Sample all embeddings for each token at once, then concatenate along token dim
        z_sample_list = []
        for kde in self.kdes:
            samples = kde.sample(num_samples, random_state=42)
            z_sample_list.append(torch.tensor(samples, dtype=torch.float32).unsqueeze(1))
        
        # Concatenate: list of [num_samples, 1, latent_dim] -> [num_samples, n_tokens, latent_dim]
        z_sample = torch.cat(z_sample_list, dim=1)
        
        return z_sample
    
    def get_stats(self) -> dict:
        """Get statistics about the fitted KDE."""
        return {
            'mode': 'per_token',
            'n_kdes': len(self.kdes),
            'dimensions_per_kde': self.latent_dim,
            'bandwidth': self.bandwidth,
            'n_tokens': self.n_tokens,
            'total_parameters': len(self.kdes) * self.latent_dim}
