"""Tests for decode utilities: evaluate_model and condition_bn_models."""

import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from sane.downstream.generative_task import evaluate_model, condition_bn_models


class SimpleCNN(nn.Module):
    """Simple CNN for testing."""
    
    def __init__(self, num_classes=10):
        super().__init__()
        self.conv = nn.Conv2d(3, 8, 3, padding=1)
        self.bn = nn.BatchNorm2d(8)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(8, num_classes)
    
    def forward(self, x):
        x = self.pool(torch.relu(self.bn(self.conv(x))))
        x = x.flatten(1)
        return self.fc(x)


def test_evaluate_model_perfect_accuracy():
    """Test evaluate_model with a model that should achieve ~100% on deterministic data."""
    model = SimpleCNN(num_classes=2)
    
    # Create deterministic test data where class 0 = zeros, class 1 = ones
    x0 = torch.zeros(10, 3, 8, 8)
    x1 = torch.ones(10, 3, 8, 8) 
    x = torch.cat([x0, x1], dim=0)
    y = torch.cat([torch.zeros(10), torch.ones(10)]).long()
    
    # Train briefly to learn the pattern
    model.train()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    for _ in range(50):
        optimizer.zero_grad()
        loss = nn.CrossEntropyLoss()(model(x), y)
        loss.backward()
        optimizer.step()
    
    loader = DataLoader(TensorDataset(x, y), batch_size=5)
    acc = evaluate_model(model.state_dict(), model, loader, device=torch.device('cpu'))
    
    # Should get high accuracy on this simple task
    assert acc > 80.0, f"Expected > 80% accuracy, got {acc:.1f}%"


def test_evaluate_model_random_guess():
    """Test evaluate_model with an untrained model achieves ~50% on binary task."""
    torch.manual_seed(42)
    model = SimpleCNN(num_classes=2)
    
    x = torch.randn(100, 3, 8, 8)
    y = torch.randint(0, 2, (100,))
    
    loader = DataLoader(TensorDataset(x, y), batch_size=10)
    acc = evaluate_model(model.state_dict(), model, loader, device=torch.device('cpu'))
    
    # Random model should be around 50% on binary task
    assert 20.0 < acc < 80.0, f"Expected ~50% accuracy for random model, got {acc:.1f}%"


def test_evaluate_model_empty_loader():
    """Test evaluate_model returns 0 for empty loader."""
    model = SimpleCNN(num_classes=2)
    loader = DataLoader(TensorDataset(torch.empty(0, 3, 8, 8), torch.empty(0).long()), batch_size=1)
    
    acc = evaluate_model(model.state_dict(), model, loader, device=torch.device('cpu'))
    assert acc == 0.0, "Empty loader should return 0% accuracy"


def test_condition_bn_models_updates_bn_stats():
    """Test that BN conditioning updates running statistics."""
    model = SimpleCNN(num_classes=2)
    original_state = model.state_dict()
    
    # Get original BN stats
    original_mean = original_state['bn.running_mean'].clone()
    original_var = original_state['bn.running_var'].clone()
    
    # Create data with different distribution
    x = torch.randn(50, 3, 8, 8) * 5 + 10  # Different mean/var
    y = torch.randint(0, 2, (50,))
    dataloader = DataLoader(TensorDataset(x, y), batch_size=10)
    
    conditioned = condition_bn_models(
        state_dicts=[original_state],
        model_template=model,
        dataloader=dataloader,
        device='cpu',
        iterations=5,
        verbose=False
    )
    
    # Check BN stats changed
    new_mean = conditioned[0]['bn.running_mean']
    new_var = conditioned[0]['bn.running_var']
    
    assert not torch.allclose(original_mean, new_mean), "BN running_mean should change after conditioning"
    assert not torch.allclose(original_var, new_var), "BN running_var should change after conditioning"


def test_condition_bn_models_multiple_models():
    """Test conditioning multiple models."""
    model = SimpleCNN()
    
    x = torch.randn(20, 3, 8, 8)
    y = torch.randint(0, 10, (20,))
    dataloader = DataLoader(TensorDataset(x, y), batch_size=5)
    
    state_dicts = [model.state_dict() for _ in range(3)]
    
    conditioned = condition_bn_models(
        state_dicts=state_dicts,
        model_template=model,
        dataloader=dataloader,
        device='cpu',
        iterations=2,
        verbose=False
    )
    
    assert len(conditioned) == 3, "Should return same number of models"
