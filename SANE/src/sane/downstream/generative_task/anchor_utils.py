import json
import torch
from typing import Tuple, Any, List
from torch.amp import autocast
from ray import logger

from sane.data.datasets import ZooDataset
from sane.data.datasets.windowed_dataset import WindowedDataset
from sane.model.autoencoder import SANEAutoEncoder


def encode_anchors(
    sane_ae: SANEAutoEncoder,
    tokens_dataset: WindowedDataset,
    zoo_dataset: ZooDataset,
    batch_size: int = 64,
    verbose: bool = True) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, List[Any]]:
    """
    Encode anchor models through SANE encoder.
    
    Args:
        sane_ae: SANE autoencoder model
        tokens_dataset: Dataset of tokenized models 
        zoo_dataset: Zoo dataset for getting checkpoint info
        batch_size: Batch size for encoding
        verbose: Whether to print progress
        
    Returns:
        Tuple of (embeddings, anchor_mask, anchor_positions, checkpoints)
        where checkpoints[i] is the checkpoint corresponding to embeddings[i].
    """
    device = next(sane_ae.parameters()).device
    sane_ae.eval()
    
    embeddings_list = []
    anchor_tokens = None
    anchor_mask = None  
    anchor_positions = None
    
    use_amp = device.type == 'cuda'
    
    with torch.no_grad():
        for i in range(0, len(tokens_dataset), batch_size):
            batch_indices = range(i, min(i + batch_size, len(tokens_dataset)))
            
            tokens_list, mask_list, pos_list = [], [], []
            for idx in batch_indices:
                item = tokens_dataset[idx]
                tokens, mask, pos = item
                # Handle augmentation dimension if present
                if tokens.dim() == 3:
                    tokens, mask, pos = tokens[0], mask[0], pos[0]
                tokens_list.append(tokens)
                mask_list.append(mask)
                pos_list.append(pos)
            
            tokens = torch.stack(tokens_list).to(device, non_blocking=True)
            mask = torch.stack(mask_list).to(device, non_blocking=True)
            pos = torch.stack(pos_list).to(device, non_blocking=True)
            
            with autocast(enabled=use_amp, device_type=device.type):
                embeddings = sane_ae.encoder(tokens, pos)
            
            embeddings_list.append(embeddings.cpu())
            
            # Store first batch tokenization for anchor template
            if anchor_tokens is None:
                anchor_tokens = tokens[0:1].cpu()
                anchor_mask = mask[0:1].cpu()
                anchor_positions = pos[0:1].cpu()
    
    embeddings = torch.cat(embeddings_list, dim=0)
    
    # Collect all checkpoints for per-model decoding
    checkpoints = []
    for i in range(len(zoo_dataset)):
        result = zoo_dataset[i]
        checkpoints.append(result[0] if isinstance(result, list) else result)
    
    if verbose:
        logger.info(f"  Encoded {len(embeddings)} models, embeddings shape: {embeddings.shape}")
    
    return embeddings, anchor_mask, anchor_positions, checkpoints


def get_arch_signature(checkpoint) -> tuple:
    """Get an architecture signature from a checkpoint's parameter shapes and (if available) model config.
    
    The signature captures both the state_dict structure (keys + shapes) and architectural hyperparameters like activation function and dropout rate.
    """
    model = checkpoint.model if hasattr(checkpoint, 'model') else checkpoint
    sd = model.state_dict() if hasattr(model, 'state_dict') else model
    shape_sig = tuple((k, tuple(v.shape)) for k, v in sorted(sd.items()))
    
    # Include model config if available (catches activation) (meant for our model zoos with varying hyper params)
    _ARCH_KEYS = {'model::nlin',}
    config_sig = ""
    if hasattr(checkpoint, 'metadata') and checkpoint.metadata:
        model_config = checkpoint.metadata.get('model_config', {})
        if model_config:
            arch_keys = {k: v for k, v in sorted(model_config.items())
                         if k in _ARCH_KEYS}
            if arch_keys:
                config_sig = json.dumps(arch_keys, sort_keys=True)
    
    return (shape_sig, config_sig)


def group_by_architecture(checkpoints: List[Any], indices: List[int] = None) -> dict:
    """Groups checkpoints by architecture signature.
    
    Args:
        checkpoints: List of checkpoint objects.
        indices: Optional list of original indices corresponding to each
            checkpoint. If None, uses 0..len(checkpoints)-1.
    
    Returns:
        Dict mapping arch_signature -> {
            'indices': list of indices into the original list,
            'template': first checkpoint with this architecture (for decoding/eval)}
    """
    if indices is None:
        indices = list(range(len(checkpoints)))
    
    groups = {}
    for i, ckpt in zip(indices, checkpoints):
        sig = get_arch_signature(ckpt)
        if sig not in groups:
            groups[sig] = {'indices': [], 'template': ckpt}
        groups[sig]['indices'].append(i)
    
    return groups