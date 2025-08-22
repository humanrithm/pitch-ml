import torch

def compute_masked_scalers(
        x: torch.Tensor, 
        mask: torch.Tensor
) -> tuple[float, float]:
    # x: [B,T,K], mask: [B,T]
    m = mask.unsqueeze(-1)                  # [B,T,1]
    num = m.sum(dim=(0,1)).clamp(min=1)     # [K]
    
    # compute mean, var, std
    mean = (x * m).sum(dim=(0,1)) / num
    var  = ((x - mean) * m).pow(2).sum(dim=(0,1)) / num
    std  = var.sqrt().clamp(min=1e-6)
    
    return mean, std

# normalize all splits in-place (or create new tensors)
def apply_scalers(
        x: torch.Tensor,
        mean: float, 
        std: float
) -> torch.Tensor: 
    return (x - mean) / std
