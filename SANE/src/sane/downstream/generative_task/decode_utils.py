import copy
from typing import List, Dict, Optional, Union

from ray import logger
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from torch.amp import autocast

from sane.model.autoencoder import SANEAutoEncoder
from sane.data.tokenizers.base import SANETokenizer

def condition_bn_models(
    state_dicts: List[Dict],
    model_template: nn.Module,
    dataloader: DataLoader,
    device: Union[str, torch.device] = 'cuda',
    iterations: int = 10,
    verbose: bool = True) -> List[Dict]:
    """
    generated resnets with batchnorm layers struggle with running_mean / running_var values, which often produce nan values
    as a fix, we perform #iterations __forward__ passes through the model to condition the batchnorm statistics
    no gradients are computed, no weights updated.
    
    Args:
        state_dicts: List of model state dicts
        model_template: Template model (for architecture)
        dataloader: Dataloader for conditioning
        device: Device to run on
        iterations: Number of forward pass iterations per model
        verbose: Print progress
    
    Returns:
        List of updated state_dicts with conditioned BN statistics
    """
    conditioned_models = []
    
    for i, state_dict in enumerate(state_dicts):
        if verbose:
            logger.info(f"   Conditioning model {i+1}/{len(state_dicts)}...")
        
        model = copy.deepcopy(model_template)
        model.load_state_dict(state_dict)
        model.to(device)
        model.train()  # BN layers update running stats in train mode
        
        with torch.no_grad():
            for idx, batch in enumerate(dataloader):
                if idx >= iterations:
                    break
                
                if isinstance(batch, (list, tuple)) and len(batch) >= 2:
                    images, _ = batch
                else:
                    images = batch
                
                images = images.to(device)
                _ = model(images)
        
        model.eval()
        conditioned_models.append(model.cpu().state_dict())
    
    return conditioned_models

def decode_embeddings(
    sane_ae: SANEAutoEncoder,
    sampled_embeddings: torch.Tensor,
    anchor_checkpoint,
    anchor_mask: torch.Tensor,
    anchor_positions: torch.Tensor,
    tokenizer: SANETokenizer,
    batch_size: int = 64,
    verbose: bool = True,
    enable_bn_conditioning: bool = False,
    bn_dataloader: Optional[DataLoader] = None,
    bn_iterations: int = 200,
    device: Optional[str] = None) -> List[Dict[str, torch.Tensor]]:
    """
    Decode embeddings back to model state_dicts using the SANE decoder,
    with optional BatchNorm conditioning.
    
    Args:
        sane_ae: SANE autoencoder
        sampled_embeddings: Tensor of embeddings [N, seq_len, latent_dim]
        anchor_checkpoint: Template checkpoint for architecture
        anchor_mask: Mask template [1, seq_len]
        anchor_positions: Position template [1, seq_len]
        tokenizer: Tokenizer for reconstruction
        batch_size: Batch size for decoding (increased for speed)
        verbose: Print progress
        enable_bn_conditioning: Enable BN conditioning after decoding
        bn_dataloader: DataLoader for BN conditioning (required if enable_bn_conditioning=True)
        bn_iterations: Number of forward passes for BN conditioning
        device: Device for BN conditioning (if None, uses model device)
        
    Returns:
        List of reconstructed state_dicts
    """
    device = next(sane_ae.parameters()).device
    sane_ae.eval()

    # Create DataLoader for batch processing
    emb_dataset = TensorDataset(sampled_embeddings)
    emb_loader = DataLoader(emb_dataset, batch_size=batch_size, shuffle=False, pin_memory=True)
    
    all_tokens = []
    use_amp = device.type == 'cuda'
    
    with torch.inference_mode():
        for (embeddings_batch,) in emb_loader:
            embeddings_batch = embeddings_batch.to(device, non_blocking=True)
            batch_size_actual = embeddings_batch.size(0)
            
            # Expand templates to match batch size and move to device
            pos_batch = anchor_positions.expand(batch_size_actual, -1, -1).to(device)  # 3D: [batch, seq_len, 3]
            
            # Decode with AMP
            with autocast(enabled=use_amp, device_type=device.type):
                decoded_tokens = sane_ae.decoder(embeddings_batch, pos_batch)
            
            all_tokens.append(decoded_tokens.cpu())
    
    # Concatenate all batches
    all_tokens = torch.cat(all_tokens, dim=0)
    
    # Reconstruct state dicts
    state_dicts = []
    template_model = anchor_checkpoint.model if hasattr(anchor_checkpoint, 'model') else anchor_checkpoint
    template_state_dict = template_model.state_dict() if hasattr(template_model, 'state_dict') else template_model
    
    for i in range(len(all_tokens)):
        tokens = all_tokens[i]
        mask = anchor_mask[0]
        
        state_dict = tokenizer.detokenize(
            tokens,  
            mask=mask,
            position=anchor_positions[0],
            reference_statedict=template_state_dict)
        state_dicts.append(state_dict)

    if verbose:
        logger.info(f"  Decoded {len(state_dicts)} models from embeddings")

    if enable_bn_conditioning:
        if bn_dataloader is None:
            raise ValueError("bn_dataloader must be provided when enable_bn_conditioning=True")
        
        if verbose:
            logger.info(f"  Conditioning BatchNorm statistics...")
        
        template_model = anchor_checkpoint.model if hasattr(anchor_checkpoint, 'model') else anchor_checkpoint
        
        if device is None:
            device = next(sane_ae.parameters()).device
        
        state_dicts = condition_bn_models(
            state_dicts=state_dicts,
            model_template=template_model,
            dataloader=bn_dataloader,
            device=device,
            iterations=bn_iterations,
            verbose=verbose)

    return state_dicts


def sample_and_decode(
    sane_ae: SANEAutoEncoder,
    kde_sampler,
    num_samples: int,
    anchor_checkpoint,
    anchor_mask: torch.Tensor,
    anchor_positions: torch.Tensor,
    tokenizer: SANETokenizer,
    batch_size: int = 64,
    verbose: bool = True,
    enable_bn_conditioning: bool = False,
    bn_dataloader: Optional[DataLoader] = None,
    bn_iterations: int = 200,
    device: Optional[str] = None) -> tuple:
    """
    Sample from KDE and decode to model parameters with optional BN conditioning.
    
    Args:
        sane_ae: SANE autoencoder
        kde_sampler: Fitted KDE sampler
        num_samples: Number of models to sample
        anchor_checkpoint: Template checkpoint
        anchor_mask: Mask template
        anchor_positions: Position template
        tokenizer: Tokenizer for reconstruction
        batch_size: Batch size for decoding 
        verbose: Print progress
        enable_bn_conditioning: Enable BN conditioning after decoding
        bn_dataloader: DataLoader for BN conditioning (required if enable_bn_conditioning=True)
        bn_iterations: Number of forward passes for BN conditioning
        device: Device for BN conditioning (if None, uses 'cuda' if available)
        
    Returns:
        Tuple of (state_dicts, template_model)
    """
    if verbose:
        logger.info(f"  Sampling {num_samples} models from KDE...")
    
    sampled_embeddings = kde_sampler.sample(
        num_samples=num_samples)
    
    state_dicts = decode_embeddings(
        sane_ae=sane_ae,
        sampled_embeddings=sampled_embeddings,
        anchor_checkpoint=anchor_checkpoint,
        anchor_mask=anchor_mask,
        anchor_positions=anchor_positions,
        tokenizer=tokenizer,
        batch_size=batch_size,
        verbose=verbose,
        enable_bn_conditioning=enable_bn_conditioning,
        bn_dataloader=bn_dataloader,
        bn_iterations=bn_iterations,
        device=device)
    
    template_model = anchor_checkpoint.model if hasattr(anchor_checkpoint, 'model') else anchor_checkpoint
    
    return state_dicts, template_model
