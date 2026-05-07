"""1-D temporal CNN baseline.

Stack of 1-D convolutions with dilation (TCN-lite) over the input window,
followed by global-average pooling and a linear head. Faster than RNNs,
captures local temporal patterns; classical baseline for time-series
benchmarks (Bai-Kolter-Koltun 2018 "An Empirical Evaluation of Generic
Convolutional and Recurrent Networks for Sequence Modeling").

Conforms to:
    forward(x_enc, x_mark_enc=None) -> [B, pred_len]

Hyperparameters via configs:
    enc_in        : input feature dim
    pred_len      : forecast horizon
    d_model       : channel count per conv layer (default 64)
    e_layers      : number of conv blocks (default 3)
    kernel_size   : conv kernel (default 3)
    dropout       : dropout in each block (default 0.1)
"""
from __future__ import annotations

import torch
import torch.nn as nn


class _ConvBlock(nn.Module):
    """Causal 1-D conv → LayerNorm → GELU → Dropout. Dilation grows
    geometrically across blocks to expand the receptive field without
    pooling away the temporal dimension."""

    def __init__(self, c_in: int, c_out: int, kernel_size: int,
                 dilation: int, dropout: float):
        super().__init__()
        # Causal padding: (kernel-1)*dilation on the left, 0 on the right.
        self.pad = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(c_in, c_out, kernel_size,
                              padding=0, dilation=dilation)
        self.norm = nn.LayerNorm(c_out)
        self.act = nn.GELU()
        self.drop = nn.Dropout(dropout)
        self.proj = nn.Linear(c_in, c_out) if c_in != c_out else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, C_in, T]
        residual = x.transpose(1, 2)               # [B, T, C_in]
        residual = self.proj(residual).transpose(1, 2)
        # Left-pad causally.
        x_pad = nn.functional.pad(x, (self.pad, 0))
        h = self.conv(x_pad)                       # [B, C_out, T]
        h = self.norm(h.transpose(1, 2)).transpose(1, 2)
        h = self.act(h)
        h = self.drop(h)
        return h + residual


class Model(nn.Module):
    def __init__(self, configs):
        super().__init__()
        def _g(k, default): return configs.get(k, default) if isinstance(configs, dict) else getattr(configs, k, default)
        self.enc_in = int(_g("enc_in", 6))
        self.pred_len = int(_g("pred_len", 5))
        self.d_model = int(_g("d_model", 64))
        self.e_layers = int(_g("e_layers", 3))
        self.kernel_size = int(_g("kernel_size", 3))
        self.dropout_p = float(_g("dropout", 0.1))

        self.input_proj = nn.Conv1d(self.enc_in, self.d_model, kernel_size=1)
        blocks = []
        for i in range(self.e_layers):
            blocks.append(_ConvBlock(
                c_in=self.d_model, c_out=self.d_model,
                kernel_size=self.kernel_size,
                dilation=2 ** i,                   # 1, 2, 4, ...
                dropout=self.dropout_p,
            ))
        self.blocks = nn.ModuleList(blocks)
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.LayerNorm(self.d_model),
            nn.Dropout(self.dropout_p),
            nn.Linear(self.d_model, self.pred_len),
        )

    def forward(self, x_enc: torch.Tensor, x_mark_enc=None):
        # x_enc: [B, T, F] -> [B, F, T]
        x = x_enc.transpose(1, 2)
        x = self.input_proj(x)                     # [B, d_model, T]
        for blk in self.blocks:
            x = blk(x)
        pred = self.head(x)                        # [B, pred_len]
        return pred
