import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

""" LOSS FUNCTIONS """
def pitch_level_loss(
        logits_step: torch.Tensor, 
        y_step: torch.Tensor, 
        mask: torch.Tensor, 
        pos_weight: bool = False
) -> torch.Tensor:
    """ 
    Compute pitch-level loss given ground truth. Valid for binary or smoothed (e.g., sigmoid) outcome labels.

    Args:
        logits_step (torch.Tensor): Logits from the model of shape [B, T].
        y_step (torch.Tensor): Ground truth labels of shape [B, T]. Should be in [0, 1].
        mask (torch.Tensor): Mask indicating valid time steps of shape [B, T].
        pos_weight (bool, optional): Whether or not to use weights for positive class in BCE loss. Default is False.
    """
    if pos_weight is None:
        # setup weights
        pos = y_step[mask].sum()
        neg = mask.sum() - pos
        pos_weight = neg / pos.clamp(min=1.0)
        
        return F.binary_cross_entropy_with_logits(logits_step[mask], y_step[mask])
    
    return F.binary_cross_entropy_with_logits(logits_step[mask], y_step[mask], pos_weight=pos_weight)

@torch.no_grad()
def compute_pos_weight(
    train_loader: DataLoader, 
    device: torch.device = torch.device('cpu')
) -> torch.Tensor:
    pos = 0.0
    tot = 0.0
    
    # iterate through training data to compute positive and total counts
    for x, y, m, L in train_loader:
        y, m = y.to(device), m.to(device)
        pos += y[m].sum().item()
        tot += m.sum().item()
    
    # compute positive weight
    neg = max(tot - pos, 1.0)
    pos = max(pos, 1.0)
    
    return torch.tensor(neg / pos, device=device, dtype=torch.float32)
