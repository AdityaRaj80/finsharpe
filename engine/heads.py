"""Output heads for the dual-arm benchmark.

Two head wrappers are exported:

  * `MSEReturnHead` — light wrapper for the MSE arm. Maps the backbone's
    [B, pred_len] z-scored close prediction to a SCALAR predicted log-
    return via a learned linear projection. Trained with plain MSE on
    the real log-return target. Replaces the previous "MSE arm with no
    wrapper" path, which forced eval_v2 to either un-z-score with per-
    stock (mu, sd) (injecting an anchor-z bias) or use z-delta ranking
    (injecting an inverse-volatility bias). With this wrapper, both arms
    output a comparable real-log-return scalar — the eval pipeline ranks
    them identically (Jury 2 fix B1+B2, 2026-05-08).

  * `RiskAwareHead` — Track-B wrapper. Adds three auxiliary heads to the
    backbone:
        - `sigma_head` predicts log-variance of the H-step ahead return,
        - `vol_head`   predicts the forward log realized volatility,
        - `return_head` projects the [pred_len] z-prediction to a scalar
          predicted log-return (same projection structure as
          `MSEReturnHead`, so eval reads `mu_return_H` from BOTH arms).

The wrappers expose the standard `forward(x_enc, x_mark_enc=None)`
signature used elsewhere in the codebase. `MSEReturnHead.forward` returns
a [B] tensor; `RiskAwareHead.forward` returns a *dict* of named outputs —
`engine/trainer.py` recognises which is which and dispatches to either
plain `nn.MSELoss` (MSE arm) or `CompositeRiskLoss` (Track B arm).

Init scale is now horizon-aware (Jury 2 fix N2): for H-step log-returns
the typical scale is σ·√H where σ is the per-step return σ ≈ 0.01-0.02.
The previous H-agnostic init w[-1]=0.01 underestimated this by ~√H, so
the loss was dominated by miscalibration for the first many epochs.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn

from config import CLOSE_IDX


def _init_return_head(linear: nn.Linear, pred_len: int) -> None:
    """Initialise a `Linear(pred_len, 1)` return-projection head.

    Strategy: weight-equally over the last `min(5, pred_len)` z-steps with a
    per-step scale calibrated so that, on a typical z-pred trajectory of
    O(1), the output is in the expected log-return scale ≈ σ·√H ≈ 0.01·√H.
    A 5-step average is used (instead of last-step only) to smooth the
    one-step noise of the backbone's final position.

    Jury 2 fix N2 (2026-05-08): replaces the previous H-agnostic init
    `w[-1] = 0.01` which severely under-estimated the H-step return scale
    (target |y_logret| at H=60 is ~0.05, not 0.005), making L_NLL/L_MSE_R
    dominated by miscalibration for the first many epochs.
    """
    with torch.no_grad():
        scale = 0.01 * math.sqrt(max(1, int(pred_len)))   # ~σ·√H
        n_avg = min(5, pred_len)
        w = torch.zeros(1, pred_len)
        w[0, -n_avg:] = scale / n_avg                      # equal-weight avg of last n_avg
        linear.weight.copy_(w)
        linear.bias.zero_()


class MSEReturnHead(nn.Module):
    """Lightweight wrapper for the MSE arm.

    Wraps the backbone so its [B, pred_len] z-scored close prediction is
    projected to a single scalar predicted H-step log-return via a learned
    linear layer. This is the SAME projection structure used by
    `RiskAwareHead.return_head`, so the MSE and Track-B arms produce
    apples-to-apples log-return predictions and the downstream evaluator
    can rank them identically.

    Args
    ----
        backbone: any nn.Module producing a price prediction tensor of
            shape [B, pred_len] (or a tuple whose first element is that
            tensor — handled identically to existing trainer.py logic).
        pred_len: forecast horizon H (in trading days).
    """

    def __init__(self, backbone: nn.Module, pred_len: int):
        super().__init__()
        self.backbone = backbone
        self.pred_len = int(pred_len)
        self.return_head = nn.Linear(self.pred_len, 1)
        _init_return_head(self.return_head, self.pred_len)

    def forward(self, x_enc: torch.Tensor, x_mark_enc=None):
        out = self.backbone(x_enc, x_mark_enc)
        if isinstance(out, tuple):
            mu_close = out[0]
        else:
            mu_close = out
        if mu_close.ndim == 1:
            mu_close = mu_close.unsqueeze(1)
        return self.return_head(mu_close).squeeze(-1)   # [B] log-return


class RiskAwareHead(nn.Module):
    """Backbone wrapper that adds heteroscedastic σ + forward-vol heads.

    Args:
        backbone: any nn.Module producing a price prediction tensor of
            shape [B, pred_len] (or a tuple whose first element is that
            tensor — handled identically to existing trainer.py logic).
        n_features: number of input feature channels (FNSPID = 6).
        pred_len: forecast horizon H.
        close_idx: index of the Close column in the input feature order.
        lookback_for_aux: number of trailing input rows the σ/vol heads
            see (independent of backbone's own context_len).
        d_hidden: hidden width of the auxiliary MLPs.
    """

    def __init__(
        self,
        backbone: nn.Module,
        n_features: int,
        pred_len: int,
        close_idx: int = CLOSE_IDX,
        lookback_for_aux: int = 20,
        d_hidden: int = 64,
    ):
        super().__init__()
        self.backbone = backbone
        self.pred_len = int(pred_len)
        self.close_idx = int(close_idx)
        self.lookback = int(lookback_for_aux)

        in_dim = self.lookback * int(n_features)

        # Sigma head — outputs log_sigma2 of the H-step LOG-RETURN.
        # Jury 2 fix IMP2 (2026-05-08): initialise the FINAL Linear's bias
        # so log_sigma2 starts at log(σ_typical²) where σ_typical = 0.01·√H
        # (the typical H-step return SD). Without this, log_sigma2 starts
        # at 0 → σ = 1 (huge), which combined with α_pos=10 makes the tanh
        # argument trivially small at init and the loss flat.
        self.sigma_head = nn.Sequential(
            nn.Linear(in_dim, d_hidden),
            nn.GELU(),
            nn.Linear(d_hidden, 1),
        )
        with torch.no_grad():
            sigma_typ = 0.01 * math.sqrt(max(1, self.pred_len))
            log_var_init = 2.0 * math.log(max(sigma_typ, 1e-6))
            self.sigma_head[-1].weight.zero_()
            self.sigma_head[-1].bias.fill_(log_var_init)

        # Vol head — outputs predicted forward log realized vol.
        # Same horizon-aware bias init (so the Z-score-space log-vol starts
        # near the typical input-window log-std).
        self.vol_head = nn.Sequential(
            nn.Linear(in_dim, d_hidden),
            nn.GELU(),
            nn.Linear(d_hidden, 1),
        )
        with torch.no_grad():
            # Vol target is log-std of z-score-space close diffs ~ log(0.5)
            # for a stock that fluctuates by half a sd per day in z-space.
            self.vol_head[-1].weight.zero_()
            self.vol_head[-1].bias.fill_(math.log(0.5))

        # EMA-tracked tau buffers (Jury 2 fix CR1+CR2, 2026-05-08).
        # These buffers live in RiskAwareHead — a CHILD of the wrapped
        # model — so:
        #   (a) `model.eval()` propagates here and stops EMA updates.
        #   (b) `model.state_dict()` saves them with the checkpoint.
        #   (c) `eval_v2` can read them at inference time if it wants
        #       to apply gate-aware reranking (currently it does not;
        #       the gate is a training-time regulariser only).
        # Bias correction (Adam-style) is applied at READ time via the
        # `_tau_ema_step` counter so values are unbiased even with
        # short training runs.
        self.register_buffer("tau_sigma_ema", torch.zeros(1))
        self.register_buffer("tau_vol_ema",   torch.zeros(1))
        self.register_buffer("_tau_ema_step", torch.zeros(1, dtype=torch.long))
        self._tau_ema_decay = 0.95   # Jury 2 fix IMP1 (was 0.99)

        # Return head (Jury 2 fix E4 / P2, 2026-05-07): maps backbone's
        # [pred_len] z-scored close-prediction to a scalar predicted
        # H-step LOG-RETURN. Learned linear projection. Without this,
        # mu_return_H lives in z-score-delta space which is not
        # comparable to the externally-provided y_logret in real return
        # space; L_NLL and L_MSE_R would have mismatched units.
        self.return_head = nn.Linear(self.pred_len, 1)
        # Horizon-aware init (Jury 2 fix N2, 2026-05-08): scale ≈ σ·√H so
        # the head's output is calibrated to the actual H-step log-return
        # magnitude from the start. Previous H-agnostic w[-1]=0.01 was
        # ~√H too small at H=60 and made the loss dominated by
        # miscalibration for the first many epochs.
        _init_return_head(self.return_head, self.pred_len)

    def forward(self, x_enc: torch.Tensor, x_mark_enc=None):
        """Forward pass.

        Args:
            x_enc: [B, seq_len, n_features] input window.
            x_mark_enc: time markers (unused here; passed through to backbone).

        Returns:
            dict with keys:
                mu_close       [B, pred_len]  — full price prediction (back-compat)
                mu_close_H     [B]            — H-step ahead price prediction
                mu_return_H    [B]            — H-step ahead return (derived)
                log_sigma2_H   [B]            — log-variance of H-step return
                log_vol_pred   [B]            — forward log realized vol
                last_close     [B]            — last observed close (for inverse scaling)
        """
        backbone_out = self.backbone(x_enc, x_mark_enc)
        if isinstance(backbone_out, tuple):
            mu_close = backbone_out[0]
        else:
            mu_close = backbone_out

        if mu_close.ndim == 1:
            # Edge case: some legacy backbones output [B] for pred_len=1.
            mu_close = mu_close.unsqueeze(1)
        # Final-step prediction
        mu_close_H = mu_close[:, -1]                                     # [B]

        last_close = x_enc[:, -1, self.close_idx]                        # [B] z-scored anchor

        # H-step predicted log-return (Jury 2 fix E4 / P2):
        # We map the backbone's [pred_len] z-scored close prediction to a
        # SCALAR LOG-RETURN via a learned linear projection (return_head).
        # The projection is initialised to extract a small last-step delta
        # (reasonable inductive bias); end-to-end training under
        # CompositeRiskLoss + true y_logret target shapes it correctly.
        # Output is in REAL LOG-RETURN units, comparable to the externally-
        # provided y_logret from the data loader.
        mu_return_H = self.return_head(mu_close).squeeze(-1)             # [B] log-return

        # Auxiliary heads on last `lookback` rows
        last_window = x_enc[:, -self.lookback:, :]                       # [B, L, F]
        flat = last_window.reshape(last_window.shape[0], -1)             # [B, L*F]
        log_sigma2_H = self.sigma_head(flat).squeeze(-1)                 # [B]
        log_vol_pred = self.vol_head(flat).squeeze(-1)                   # [B]

        return {
            "mu_close": mu_close,
            "mu_close_H": mu_close_H,
            "mu_return_H": mu_return_H,
            "log_sigma2_H": log_sigma2_H,
            "log_vol_pred": log_vol_pred,
            "last_close": last_close,
        }

    @torch.no_grad()
    def update_tau_ema(self, sigma: torch.Tensor, log_vol_pred: torch.Tensor) -> None:
        """Update the gate-threshold EMA buffers (Adam-style).

        Called by `CompositeRiskLoss.forward` during training. The update is
        a no-op when `self.training` is False (so val/test passes do not
        contaminate the buffers — Jury 2 fix CR1).

        Buffers are zero-initialised. Each call applies
        `ema_t = d·ema_{t-1} + (1-d)·x_t`. Bias correction (in `get_tau`)
        rescales by `1/(1-d^t)` so the corrected output is unbiased even
        at small `t`.
        """
        if not self.training:
            return
        d = float(self._tau_ema_decay)
        s_med = sigma.detach().median()
        v_med = log_vol_pred.detach().median()
        # Always Adam-style EMA from zero init (no special-case t=0).
        self.tau_sigma_ema.mul_(d).add_(s_med, alpha=1.0 - d)
        self.tau_vol_ema.mul_(d).add_(v_med, alpha=1.0 - d)
        self._tau_ema_step += 1

    def get_tau(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Return (tau_sigma, tau_vol) with Adam-style bias correction.

        For the first ~30 epochs, the raw EMA underestimates the long-run
        mean; the correction `1/(1-d^t)` exactly rescales for that bias.
        Outputs are unbiased for any t >= 1.

        If the buffers were never updated (e.g. eval-only inference on a
        legacy checkpoint), returns the raw zeros — the caller can fall
        back to per-batch median in that case.
        """
        t = int(self._tau_ema_step.item())
        if t == 0:
            return self.tau_sigma_ema, self.tau_vol_ema
        d = float(self._tau_ema_decay)
        bias_corr = 1.0 / (1.0 - d ** t)
        return self.tau_sigma_ema * bias_corr, self.tau_vol_ema * bias_corr
