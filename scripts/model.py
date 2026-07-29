"""MLP Autoencoder for anomaly detection.

Architecture (design §):
    encoder: input_dim → 96 → 48 → code_dim (32)
    decoder: code_dim → 48 → 96 → input_dim
Each hidden layer is Linear + ReLU + Dropout(0.1); the decoder's final Linear is bare
(no ReLU) so reconstructions can span the standardized ℝ.
"""
from __future__ import annotations

import torch
import torch.nn as nn


def _mlp_block(in_dim: int, out_dim: int, dropout: float) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(in_dim, out_dim),
        nn.ReLU(),
        nn.Dropout(dropout),
    )


class Autoencoder(nn.Module):
    """Deep MLP autoencoder.

    Args:
        input_dim: dimensionality of the flattened window (default 160 = 20 lookback × 8 feats).
        hidden:    encoder hidden widths, high → low (decoder mirrors).
        code_dim:  bottleneck dimension.
        dropout:   dropout probability on each hidden ReLU block.
    """

    def __init__(
        self,
        input_dim: int = 160,
        hidden: tuple[int, ...] = (96, 48),
        code_dim: int = 32,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.code_dim = code_dim

        enc_layers: list[nn.Module] = []
        prev = input_dim
        for h in hidden:
            enc_layers.append(_mlp_block(prev, h, dropout))
            prev = h
        enc_layers.append(_mlp_block(prev, code_dim, dropout))
        self.encoder = nn.Sequential(*enc_layers)

        dec_layers: list[nn.Module] = []
        prev = code_dim
        for h in reversed(hidden):
            dec_layers.append(_mlp_block(prev, h, dropout))
            prev = h
        dec_layers.append(nn.Linear(prev, input_dim))
        self.decoder = nn.Sequential(*dec_layers)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decode(self.encode(x))


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
