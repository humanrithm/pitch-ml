import torch
import torch.nn as nn
from .cnn_block import CNNBlock
from .loss_functions import masked_softmax
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

class CNNbiLSTM(nn.Module):
    """
    CNN + BiLSTM model for time series data with per-pitch head.

    Args:
        k_in (int): Number of input features (K).
        stem (int): Number of channels in the stem layer. Defaults to 64.
        c (int): Number of channels after projection. Defaults to 96.
        kernel (int): Size of the convolutional kernel. Defaults to 7.
        lstm_hidden (int): Hidden size for the LSTM layer. Defaults to 128.
        dropout (float): Dropout rate. Defaults to 0.1.
        bidir (bool): Whether to use a bidirectional LSTM. Defaults to True.

    Returns:
        None

    **Note**: CNN expects tensor with shape [B, K, T]. 
    """
    def __init__(
            self, 
            k_in: int, 
            stem: int = 64, 
            c: int = 96, 
            kernel: int = 7, 
            lstm_hidden: int = 128, 
            dropout: float = 0.1, 
            bidir: bool = True
    ) -> None:
        super().__init__()
        
        # 1x1 stem to mix K features into 'stem' channels
        self.stem = nn.Sequential(
            nn.Conv1d(k_in, stem, kernel_size=1),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        
        # progressive temporal convs
        self.conv1 = CNNBlock(stem, kernel=kernel, dropout=dropout)
        self.proj1 = nn.Conv1d(stem, c, kernel_size=1)                    # project to C channels
        self.conv2 = CNNBlock(c, kernel=kernel, dropout=dropout)

        # BiLSTM over time
        self.lstm = nn.LSTM(input_size=c, hidden_size=lstm_hidden, batch_first=True, bidirectional=bidir)
        hdim = lstm_hidden * (2 if bidir else 1)

        # per-pitch head
        self.head_step = nn.Sequential(
            nn.LayerNorm(hdim),
            nn.Linear(hdim, 1)
        )

        # attention scorer for pooling + sequence head
        self.attn_scorer = nn.Linear(hdim, 1)
        self.head_seq    = nn.Linear(hdim, 1)

    def forward(
            self, 
            x: torch.Tensor, 
            lengths: torch.Tensor,
            mask: torch.Tensor
    ) -> None:
        # CNN expects [B,K,T]
        x = x.transpose(1, 2)                  # [B,K,T]
        x = self.stem(x)                       # [B,stem,T]
        x = self.conv1(x)                      # [B,stem,T]
        x = self.proj1(x)                      # [B,C,T]
        x = self.conv2(x)                      # [B,C,T]
        x = x.transpose(1, 2)                  # [B,T,C] for LSTM

        # pack to ignore padding inside LSTM
        packed = pack_padded_sequence(x, lengths.cpu(), batch_first=True, enforce_sorted=False)
        packed_out, _ = self.lstm(packed)
        out, _ = pad_packed_sequence(packed_out, batch_first=True, total_length=x.size(1))    # [B,T,H]

        # apply head to get logits for each pitch
        logits_step = self.head_step(out).squeeze(-1)  # [B,T]

        # attention pooling (mask‑aware)
        scores = self.attn_scorer(out).squeeze(-1)     # [B,T]
        attn   = masked_softmax(scores, mask, dim=1)   # [B,T]
        ctx    = (attn.unsqueeze(-1) * out).sum(dim=1) # [B,H]
        logit_seq = self.head_seq(ctx).squeeze(-1)     # [B]
        
        return logits_step, logit_seq, attn 
