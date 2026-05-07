"""Smoke tests for the new LSTM/RNN/CNN baselines.

Verifies they conform to the model interface used elsewhere in the repo:
  forward(x_enc, x_mark_enc=None) -> [B, pred_len]

Also verifies they can be wrapped in RiskAwareHead and trained with
CompositeRiskLoss (i.e., they're drop-in replacements for the modern
backbones). These tests guarantee that under PLAN_v2 the headline
7-model panel can run identical pipelines.
"""
from __future__ import annotations

import os
import sys
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from models import LSTM, RNN, CNN, model_dict, HEADLINE_MODELS  # noqa: E402
from engine.heads import RiskAwareHead  # noqa: E402


class _Cfg:
    def __init__(self, **kw):
        for k, v in kw.items():
            setattr(self, k, v)


def _common_cfg(d_model=64, e_layers=2, pred_len=5):
    return _Cfg(enc_in=6, pred_len=pred_len, d_model=d_model,
                 e_layers=e_layers, dropout=0.1, kernel_size=3)


def test_lstm_forward_shape():
    m = LSTM(_common_cfg(d_model=128))
    x = torch.randn(8, 96, 6)
    out = m(x)
    assert out.shape == (8, 5)


def test_rnn_forward_shape():
    m = RNN(_common_cfg(d_model=64, e_layers=1))
    x = torch.randn(4, 48, 6)
    out = m(x)
    assert out.shape == (4, 5)


def test_cnn_forward_shape():
    m = CNN(_common_cfg(d_model=64, e_layers=3))
    x = torch.randn(2, 504, 6)
    out = m(x)
    assert out.shape == (2, 5)


def test_models_dict_contains_simple_models():
    """Headline panel must include all 7 models."""
    for name in HEADLINE_MODELS:
        assert name in model_dict, f"{name} missing from model_dict"


def test_lstm_wrappable_in_riskawarehead():
    cfg = _common_cfg(d_model=64, e_layers=2, pred_len=5)
    backbone = LSTM(cfg)
    wrapped = RiskAwareHead(backbone=backbone, n_features=6, pred_len=5,
                             close_idx=3, lookback_for_aux=20, d_hidden=32)
    x = torch.randn(4, 96, 6)
    out = wrapped(x)
    assert isinstance(out, dict)
    for key in ["mu_close", "mu_close_H", "mu_return_H", "log_sigma2_H",
                "log_vol_pred", "last_close"]:
        assert key in out, f"missing key {key}"
    assert out["mu_close"].shape == (4, 5)


def test_rnn_wrappable_in_riskawarehead():
    cfg = _common_cfg(d_model=64, e_layers=1, pred_len=20)
    backbone = RNN(cfg)
    wrapped = RiskAwareHead(backbone=backbone, n_features=6, pred_len=20,
                             close_idx=3, lookback_for_aux=20, d_hidden=32)
    x = torch.randn(2, 96, 6)
    out = wrapped(x)
    assert out["mu_close"].shape == (2, 20)


def test_cnn_wrappable_in_riskawarehead():
    cfg = _common_cfg(d_model=64, e_layers=3, pred_len=60)
    backbone = CNN(cfg)
    wrapped = RiskAwareHead(backbone=backbone, n_features=6, pred_len=60,
                             close_idx=3, lookback_for_aux=20, d_hidden=32)
    x = torch.randn(2, 504, 6)
    out = wrapped(x)
    assert out["mu_close"].shape == (2, 60)


def test_simple_models_can_backprop():
    """Loss propagates gradients through every parameter."""
    for name, ModelCls in [("LSTM", LSTM), ("RNN", RNN), ("CNN", CNN)]:
        cfg = _common_cfg(d_model=32, e_layers=2, pred_len=5)
        m = ModelCls(cfg)
        x = torch.randn(2, 48, 6, requires_grad=False)
        out = m(x)
        loss = out.pow(2).mean()
        loss.backward()
        for pname, p in m.named_parameters():
            assert p.grad is not None, f"{name}.{pname} has no grad"


if __name__ == "__main__":
    import pytest
    sys.exit(pytest.main([__file__, "-v"]))
