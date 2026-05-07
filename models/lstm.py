"""LSTM baseline — simple recurrent forecaster.

Conforms to the model interface used elsewhere in the repo:
    forward(x_enc, x_mark_enc=None) -> [B, pred_len]

Two-layer LSTM over the input window, followed by a linear projection
from the final hidden state to a `pred_len`-dim output. The output is
in *target space* (log-return if data_loader uses log-return targets,
scaled price if it uses price targets) — no internal normalisation
beyond what the model's own LayerNorm does.

Hyperparameters configurable via the `configs` namespace passed in:
    enc_in        : input feature dimension (default len(FEATURES) = 6)
    pred_len      : forecast horizon H
    d_model       : hidden size (default 128)
    e_layers      : number of stacked LSTM layers (default 2)
    dropout       : LSTM dropout (default 0.1)
    seq_len       : input window length (default config.SEQ_LEN)
"""
from __future__ import annotations

import torch
import torch.nn as nn


class Model(nn.Module):
    """Bidirectional? No — we deliberately use unidirectional to match the
    causality of forecasting. For classification or seq2seq tasks one would
    flip this."""

    def __init__(self, configs):
        super().__init__()
        # Configs is a dict (matches DLinear/PatchTST/iTransformer convention).
        # Bug fix 2026-05-07: previously used getattr() which silently fell
        # to defaults because dicts don't expose keys as attributes -- causing
        # pred_len to default to 5 regardless of horizon. Caused [B,5] vs
        # [B,H] target shape mismatch for H!=5 jobs.
        def _g(k, default): return configs.get(k, default) if isinstance(configs, dict) else getattr(configs, k, default)
        self.enc_in = int(_g("enc_in", 6))
        self.pred_len = int(_g("pred_len", 5))
        self.d_model = int(_g("d_model", 128))
        self.e_layers = int(_g("e_layers", 2))
        self.dropout_p = float(_g("dropout", 0.1))

        self.input_proj = nn.Linear(self.enc_in, self.d_model)
        self.lstm = nn.LSTM(
            input_size=self.d_model,
            hidden_size=self.d_model,
            num_layers=self.e_layers,
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
        h = self.input_proj(x_enc)                 # [B, T, d_model]
        out, (hT, _) = self.lstm(h)                # out: [B, T, d_model]
        last = out[:, -1, :]                       # [B, d_model] — final timestep
        pred = self.head(last)                     # [B, pred_len]
        return pred
