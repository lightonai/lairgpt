import torch
import torch.nn as nn
from typing import Optional
from .blocks.blocks import LearnedPositionalEncoding, DecoderLayer


class LairGPT(nn.Module):
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        n_layers: int,
        vocab_size: int,
        dropout: float = 0.1,
        max_seq_len: int = 1024,
        add_bias=False,
    ):
        super(LairGPT, self).__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.positional_encoder = LearnedPositionalEncoding(d_model, max_seq_len=max_seq_len)
        self.layers = nn.ModuleList(
            [
                DecoderLayer(d_model, n_heads, d_feedforward=4 * d_model, dropout=dropout)
                for _ in range(n_layers)
            ]
        )
        self.final_norm = nn.LayerNorm(d_model)
        self.projector = nn.Linear(d_model, vocab_size, bias=add_bias)

    def forward(
        self,
        input_ids: torch.LongTensor,
        attn_mask: Optional[torch.BoolTensor] = None,
        padding_mask: Optional[torch.BoolTensor] = None,
    ) -> torch.FloatTensor:
        x = self.embedding(input_ids)
        x = self.positional_encoder(x)
        for layer in self.layers:
            x = layer(x, attn_mask=attn_mask, padding_mask=padding_mask)

        x = self.final_norm(x)
        logits = self.projector(x)
        return logits
