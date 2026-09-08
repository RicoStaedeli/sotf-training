import copy
from typing import Union

import torch
from torch.utils.data import DataLoader
from torch.amp import autocast

def evaluate_model(
    state_dict: dict,
    template_model: torch.nn.Module,
    test_loader: DataLoader,
    device: Union[str, torch.device]) -> float:
    """
    Evaluate a single model's classification accuracy.
    
    Args:
        state_dict: Model parameters to evaluate
        template_model: Template model with correct architecture
        test_loader: DataLoader for test dataset
        device: Device to run evaluation on
        
    Returns:
        Accuracy as percentage 
    """
    model = copy.deepcopy(template_model)
    model.load_state_dict(state_dict, strict=True)
    model.to(device)
    model.eval()
    
    correct = 0
    total = 0
    use_amp = device.type == 'cuda' if isinstance(device, torch.device) else 'cuda' in str(device)
    
    with torch.inference_mode():
        for x, y in test_loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            with autocast(enabled=use_amp, device_type='cuda' if use_amp else 'cpu'):
                pred = model(x).argmax(1)
            
            total += y.size(0)
            correct += (pred == y).sum().item()
    
    return 100.0 * correct / total if total > 0 else 0.0

