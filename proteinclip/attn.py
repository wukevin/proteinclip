import numpy as np

import torch
from torch import nn
from torch.nn import functional as F

from esm.modules import TransformerLayer, ESM1bLayerNorm


class ESMTransformerAggregation(nn.Module):
    """Transformer to aggregate sequence of embeddings into a single token."""

    # Heavily inspired by https://github.com/facebookresearch/esm/blob/main/esm/model/esm2.py

    def __init__(self, embed_dim: int = 128, num_heads: int = 4, num_layers: int = 1):
        super().__init__()

        self.layers = nn.ModuleList(
            TransformerLayer(
                embed_dim=embed_dim,
                ffn_embed_dim=embed_dim * 4,
                attention_heads=num_heads,
                add_bias_kv=False,
                use_esm1b_layer_norm=True,
                use_rotary_embeddings=True,
            )
            for _ in range(num_layers)
        )
        self.emb_layer_norm_after = ESM1bLayerNorm(embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        # Input is B, N, E
        padding_mask = torch.eq(x, 0.0).all(dim=-1)

        x = x * (1 - padding_mask.unsqueeze(-1).type_as(x))

        # Transformer expects B, N, E -> N, B, E
        x = x.transpose(0, 1)

        for l in self.layers:
            x, _attn = l(x, self_attn_padding_mask=padding_mask)

        x = self.emb_layer_norm_after(x)
        x = x.transpose(0, 1)  # (T, B, E) => (B, T, E)

        return x[:, 0]  # Return the first token


class SingleTokenAttention(nn.Module):
    """Attention mechanism to aggregate sequence of embeddings into a single token."""

    def __init__(self, embed_dim: int = 128, num_heads: int = 4):
        super().__init__()

        self.attn = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            batch_first=True,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        # Input is expected to be B, N, E

        # Attention masking is detected as positions that are all zero
        key_mask = torch.eq(x[:, 1:], 0.0).all(dim=-1)  # (batch, N - 1)
        # Should have shape (batch * heads, 1, N - 1)
        attn_mask = key_mask.unsqueeze(1).repeat_interleave(self.attn.num_heads, dim=0)

        # Extract first token as query
        # Remainder of sequence is key and value
        q = x[:, 0].unsqueeze(1)  # B, 1, E
        k = x[:, 1:]  # B, N-1, E
        v = x[:, 1:]  # B, N-1, E
        attn_out, _attn_weights = self.attn(
            q, k, v, key_padding_mask=key_mask, attn_mask=attn_mask
        )
        assert attn_out.shape == (q.shape[0], 1, q.shape[-1])
        return attn_out.squeeze(1)


if __name__ == "__main__":
    # Toy run to test memory
    b, n, e = 64, 5801, 1280
    dev = torch.device("cuda:0")

    x = torch.rand((b, n, e), device=dev)
    print(x.shape)

    net = SingleTokenAttention(embed_dim=e, num_heads=4).to(dev)
    y = net.forward(x)
    print(y.shape)
