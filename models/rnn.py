"""Vanilla Elman RNN baseline.

The simplest recurrent model — single-layer or stacked tanh RNN. Included
to anchor the "simple vs complex" axis of the benchmark: any improvement
PatchTST/GCFormer/TFT show over this baseline is the price of complexity.

Conforms to:
    forward(x_enc, x_mark_enc=None) -> [B, pred_len]

Hyperparameters via configs:
    enc_in        : input feature dim
    pred_len      : forecast horizon
    d_model       : hidden size (default 64; deliberately smaller than LSTM
                    since vanilla RNN suffers from gradient pathology at depth)
    e_layers      : layers (default 1; vanilla RNN above 2 layers is fragile)
    dropout       : dropout (default 0.1)
"""
from __future__ import annotations

import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(self, configs):
        super().__init__()
        def _g(k, default): return configs.get(k, default) if isinstance(configs, dict) else getattr(configs, k, default)
        self.enc_in = int(_g("enc_in", 6))
        self.pred_len = int(_g("pred_len", 5))
        self.d_model = int(_g("d_model", 64))
        self.e_layers = int(_g("e_layers", 1))
        self.dropout_p = float(_g("dropout", 0.1))

        self.input_proj = nn.Linear(self.enc_in, self.d_model)
        self.rnn = nn.RNN(
            input_size=self.d_model,
            hidden_size=self.d_model,
            num_layers=self.e_layers,
            nonlinearity="tanh",
            dropout=self.dropout_p if self.e_layers > 1 else 0.0,
            batch_first=True,
        )
        self.head = nn.Sequential(
            nn.LayerNorm(self.d_model),
            nn.Dropout(self.dropout_p),
            nn.Linear(self.d_model, self.pred_len),
        )

    def forward(self, x_enc: torch.Tensor, x_mark_enc=None):
        # x_enc: [B, seq_len, enc_in]
        h = self.input_proj(x_enc)
        out, _ = self.rnn(h)                       # [B, T, d_model]
        last = out[:, -1, :]                       # [B, d_model]
        pred = self.head(last)                     # [B, pred_len]
        return pred
