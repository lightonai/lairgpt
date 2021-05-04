import torch
import torch.nn as nn
from typing import Optional


class LearnedPositionalEncoding(nn.Module):
    """Applies absolute positional encoding with learned weights"""

    def __init__(self, d_model, dropout=0.1, max_seq_len=1024):
        super().__init__()
        self.wpe = nn.Embedding(max_seq_len, d_model)
        position_ids = torch.arange(0, max_seq_len, dtype=torch.long).unsqueeze(0)
        self.register_buffer("position_ids", position_ids)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.FloatTensor) -> torch.FloatTensor:  # N x L x D
        x = x + self.wpe(self.position_ids[:, : x.size(1)])
        return self.dropout(x)


class DecoderLayer(nn.Module):
    """Transformer block containing the self-attention module and the feedfoward module."""

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        d_feedforward: int,
        dropout: float,
    ):
        super(DecoderLayer, self).__init__()
        self.self_attention = nn.MultiheadAttention(d_model, n_heads, dropout)
        self.attn_norm = nn.LayerNorm(d_model)
        self.attn_dropout = nn.Dropout(dropout)

        self.mlp = nn.Sequential(
            nn.Linear(d_model, d_feedforward, bias=True),
            nn.GELU(),
            nn.Linear(d_feedforward, d_model, bias=True),
        )
        self.mlp_norm = nn.LayerNorm(d_model)
        self.mlp_dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.FloatTensor,
        attn_mask: Optional[torch.BoolTensor] = None,
        padding_mask: Optional[torch.BoolTensor] = None,
    ) -> torch.FloatTensor:
        y = self.attn_norm(x)
        y = y.transpose(1, 0).contiguous()
        y, attn_weights = self.self_attention(
            y, y, y, attn_mask=attn_mask, key_padding_mask=padding_mask
        )
        y = y.transpose(1, 0).contiguous()
        x = x + self.attn_dropout(y)

        y = self.mlp_norm(x)
        y = self.mlp(y)
        x = x + self.mlp_dropout(y)
        return x
