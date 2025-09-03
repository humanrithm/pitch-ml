import torch
import torch.nn as nn

# CNN block for local patterns
class CNNBlock(nn.Module):
    """ 
    Depthwise-separable 1D convolutional block over time with residual. 

    Args:
        num_channels (int): Number of input channels.
        kernel (int): Size of the convolutional kernel. Defaults to 7.
        dropout (float): Dropout rate. Defaults to 0.1.
    
    Returns:
        None
    
    **Note**: CNN expects tensor with shape [B, C, T]. 
    """
    def __init__(
            self, 
            num_channels: int, 
            kernel: int = 7, 
            dropout: float = 0.1
    ) -> None:
        super().__init__()
        pad = kernel // 2
        
        # depthwise convolution
        self.dw = nn.Conv1d(num_channels, num_channels, kernel_size=kernel, padding=pad, groups=num_channels)
        self.pw = nn.Conv1d(num_channels, num_channels, kernel_size=1)
        
        # batch normalization, activation, and dropout
        self.bn = nn.BatchNorm1d(num_channels)
        self.act = nn.GELU()
        self.drop = nn.Dropout(dropout)

    def forward(
            self, 
            x: torch.Tensor
    ):  # x: [B,C,T]
        residual = x
        
        # apply layers
        x = self.dw(x)
        x = self.pw(x)
        
        # batch normalization, activation, and dropout
        x = self.bn(x)
        x = self.act(x)
        x = self.drop(x)
        
        return x + residual