import os
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW

from engine.early_stopping import EarlyStopping, adjust_learning_rate
from engine.evaluator import evaluate
from engine.losses import CompositeRiskLoss


def _compute_log_vol_target(batch_x: torch.Tensor,
                            close_idx: int,
                            lookback: int = 20,
                            eps: float = 1e-3) -> torch.Tensor:
    """Realised log-vol target derived from the INPUT WINDOW close stride.

    We supervise the vol head with a *trailing* realised-vol estimator — the
    log-std of the last `lookback` daily log-returns within the input
    window. This is well-defined regardless of `target_mode`, scale-aware,
    and a more meaningful supervision signal than `log|y_logret|` (which is
    just a noisy single-sample magnitude).

    Jury 2 fix N3 (2026-05-08): replaces the previous implementation that
    fell back to `log|single-step return|` whenever batch_y was 1-D
    (which happens for `target_mode='log_return'`, the canonical mode) —
    that implementation supervised the vol head on noise.

    Args
    ----
        batch_x    : [B, seq_len, F] z-scored input window.
        close_idx  : index of the Close column.
        lookback   : number of trailing rows to use for the vol estimate.
                     Default 20 ≈ 1 month, matching the typical realised-vol
                     horizon used by HIST / MASTER.
        eps        : numerical floor on vol before log.

    Returns
    -------
        log_vol_target : [B]   log-std of the last `lookback` daily
                                z-score-space close differences.
    """
    L = min(lookback + 1, batch_x.shape[1])
    if L < 3:
        return torch.zeros(batch_x.shape[0], device=batch_x.device)
    closes = batch_x[:, -L:, close_idx]                          # [B, L]
    diffs = closes[:, 1:] - closes[:, :-1]                       # [B, L-1]
    std = diffs.std(dim=1, unbiased=False)
    return torch.log(std + eps)


class Trainer:
    def __init__(self, args, model, device):
        self.args = args
        self.model = model
        self.device = device
        # Close-column index in the input feature order (used by the vol-
        # target estimator).
        try:
            from config import CLOSE_IDX as _CLOSE_IDX
        except Exception:
            _CLOSE_IDX = 3
        self.close_idx = int(_CLOSE_IDX)
        # Risk-aware mode: model is wrapped in RiskAwareHead (returns dict).
        # Detect either via explicit args flag (preferred) or via duck-typing
        # for back-compat with callers that don't set the flag.
        self.use_risk_head = bool(getattr(args, 'use_risk_head', False))
        if self.use_risk_head:
            use_xs = bool(getattr(args, 'use_xs_sharpe', False))
            xs_K = int(getattr(args, 'xs_n_subgroups', 32))
            self.criterion = CompositeRiskLoss(
                use_xs_sharpe=use_xs, xs_n_subgroups=xs_K
            ).to(device)
            # Bind the criterion to the RiskAwareHead so it can read the
            # head's EMA tau buffers (Jury 2 fix CR1+CR2). The head is
            # the wrapper around the backbone; the criterion uses the
            # head's `update_tau_ema` and `get_tau` methods inside its
            # forward.
            from engine.heads import RiskAwareHead
            head = self.model
            if not isinstance(head, RiskAwareHead):
                # Defensive: criterion only meaningful when model is the head.
                raise TypeError("use_risk_head=True requires the model to be "
                                "wrapped in RiskAwareHead before constructing "
                                "Trainer.")
            self.criterion.attach_head(head)
            xs_msg = f" (xs-portfolio loss, K={xs_K})" if use_xs else " (per-sample Sharpe)"
            print(f"[Trainer] CompositeRiskLoss active{xs_msg}. "
                  f"Schedule: phase1<{self.criterion._phase1_end} (MSE-only) "
                  f"-> phase2<{self.criterion._phase2_end} (alpha=0.3) "
                  f"-> phase3 (alpha=0.7).")
        else:
            # MSE arm now trains on REAL log-return target (Jury 2 fix B1+B2).
            # The model is wrapped in MSEReturnHead, which outputs [B] log-
            # return; the criterion is MSE against y_logret. This makes the
            # MSE arm apples-to-apples with Track-B's mu_return_H output.
            self.criterion = nn.MSELoss()
        self.optimizer = AdamW(self.model.parameters(), lr=args.lr)

    # ────────────────────────────────────────────────────────────── helpers
    def _forward_loss(self, batch_x: torch.Tensor, batch_y: torch.Tensor,
                       y_logret: torch.Tensor | None = None):
        """Single forward + loss eval, branching on training mode.

        Args:
            batch_x  : [B, seq_len, F] z-scored features
            batch_y  : training-mode-specific target (z-scored close window
                       for 'scaled_price' mode, scalar log-return for
                       'log_return' mode)
            y_logret : [B] TRUE log-return scalar (always provided by v2
                       data loader). Used by Track-B's composite loss for
                       Sharpe / NLL computation in actual return space
                       (Jury 2 fix item E4).

        Returns
        -------
            loss        : scalar tensor (backward-passable)
            parts       : dict of named float components (or {} for plain MSE)
        """
        outputs = self.model(batch_x, None)
        if self.use_risk_head:
            assert isinstance(outputs, dict), (
                "use_risk_head=True but model output is not a dict. "
                "Wrap the backbone in engine.heads.RiskAwareHead."
            )
            # Jury 2 fix N3 (2026-05-08): vol target is now derived from
            # the input window's close stride (well-defined for any
            # target_mode), not from `log|y_logret|` (a noisy single
            # sample) or from a price window which is unavailable in
            # `target_mode='log_return'`.
            log_vol_target = _compute_log_vol_target(batch_x, self.close_idx)
            loss, parts = self.criterion(outputs, batch_y, log_vol_target,
                                          y_logret=y_logret)
            return loss, parts
        # MSE arm: model is wrapped in MSEReturnHead, output is [B] log-
        # return scalar. Criterion is MSE against y_logret (the TRUE
        # H-step log-return in real price space). Apples-to-apples with
        # Track-B's mu_return_H output.
        if isinstance(outputs, tuple):
            if self.args.model_name == 'AdaPatch':
                # AdaPatch retains its dual-loss training (pred + reconstruction);
                # we keep the legacy z-scored close target for it.
                pred, orig, dec = outputs
                loss_pred = self.criterion(pred, batch_y)
                loss_rec = self.criterion(dec, orig)
                loss = (self.args.adapatch_alpha * loss_pred
                        + (1 - self.args.adapatch_alpha) * loss_rec)
                return loss, {}
            else:
                outputs = outputs[0]
        # outputs should be [B] log-return scalar from MSEReturnHead.
        if outputs.dim() > 1:
            outputs = outputs.squeeze(-1)
        target = y_logret if y_logret is not None else (
            batch_y if batch_y.dim() == 1 else batch_y[:, -1]
        )
        loss = self.criterion(outputs, target)
        return loss, {}

    def train_epoch(self, train_loader):
        self.model.train()
        train_loss = []
        accum_parts: dict[str, list] = {}

        for batch in train_loader:
            # v2 loader returns (X, y_main, y_logret); legacy 2-tuple support
            # for back-compat with old caches / tests.
            if len(batch) == 3:
                batch_x, batch_y, y_logret = batch
                y_logret = y_logret.float().to(self.device)
            else:
                batch_x, batch_y = batch
                y_logret = None
            self.optimizer.zero_grad()
            batch_x = batch_x.float().to(self.device)
            batch_y = batch_y.float().to(self.device)

            if self.args.use_amp:
                device_type = 'cuda' if self.device.type == 'cuda' else 'cpu'
                # bf16 only -- A100/H100/H200 all support it natively. fp16
                # fallback removed (Jury 2 fix D1: dead code under campaign).
                if not (self.device.type == 'cuda'
                         and torch.cuda.get_device_capability(self.device)[0] >= 8):
                    raise RuntimeError(
                        "use_amp requires an Ampere+ GPU (A100/H100/H200). "
                        "Older GPUs would need fp16+GradScaler which is not "
                        "wired up in v2 (would silently underflow gradients).")
                with torch.autocast(device_type=device_type, dtype=torch.bfloat16):
                    loss, parts = self._forward_loss(batch_x, batch_y, y_logret)
                loss.backward()
            else:
                loss, parts = self._forward_loss(batch_x, batch_y, y_logret)
                loss.backward()

            # Gradient clipping ALWAYS applied (Jury 2 fix D2: was previously
            # skipped under AMP path -- the entire campaign was unclamped).
            nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()

            train_loss.append(loss.item())
            for k, v in parts.items():
                accum_parts.setdefault(k, []).append(v)

        avg_parts = {k: sum(vs) / max(len(vs), 1) for k, vs in accum_parts.items()}
        return sum(train_loss) / max(len(train_loss), 1), avg_parts

    @staticmethod
    def _format_parts(parts: dict) -> str:
        """Compact one-line summary of CompositeRiskLoss components."""
        if not parts:
            return ""
        keys = ["L_MSE_R", "L_NLL", "L_VOL", "L_SR_gated", "L_GATE_BCE",
                "alpha", "gamma", "gate_mean", "sigma_mean", "position_mean_abs"]
        bits = []
        for k in keys:
            if k in parts:
                v = parts[k]
                bits.append(f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}")
        return " | " + " ".join(bits)

    # ─────────────────────────────────────────────── val rank-IC ──
    @torch.no_grad()
    def _compute_val_rank_ic(self, val_loader, val_anchor_dates) -> float:
        """Compute mean-per-day Spearman rank-IC on the val split.

        Args:
            val_loader: DataLoader producing (X, y_main, y_logret) batches in
                        the same order as `val_anchor_dates`.
            val_anchor_dates: int64 array of anchor calendar dates (ns
                              since epoch), aligned with the loader's
                              sample order.

        Returns:
            mean_ic : float in [-1, 1]. Returns 0.0 if no valid days.

        Used for early-stopping decisions (Jury 2 fix F14, 2026-05-08).
        """
        import numpy as np
        from scipy.stats import spearmanr
        n_total = len(val_anchor_dates)
        preds = np.zeros(n_total, dtype=np.float64)
        targs = np.zeros(n_total, dtype=np.float64)
        self.model.eval()
        cursor = 0
        for batch in val_loader:
            if len(batch) == 3:
                X, _y_main, y_logret = batch
            else:
                X, y_logret = batch
            X = X.float().to(self.device)
            B = X.shape[0]
            out = self.model(X, None)
            if isinstance(out, dict):
                p = out["mu_return_H"].detach().float().cpu().numpy()
            elif isinstance(out, tuple):
                p = out[0].detach().float().cpu().numpy()
                if p.ndim > 1:
                    p = p[:, -1]
            else:
                p = out.detach().float().cpu().numpy()
                if p.ndim > 1:
                    p = p.squeeze(-1)
            preds[cursor:cursor + B] = p
            targs[cursor:cursor + B] = y_logret.detach().float().cpu().numpy()
            cursor += B
        if cursor != n_total:
            # Loader didn't iterate everything (possible with drop_last);
            # truncate to what we actually saw.
            preds = preds[:cursor]; targs = targs[:cursor]
            anchor_dates = val_anchor_dates[:cursor]
        else:
            anchor_dates = val_anchor_dates
        # Group by anchor date and compute Spearman per date.
        unique_dates, inv = np.unique(anchor_dates, return_inverse=True)
        ic_per_day = []
        for k in range(len(unique_dates)):
            m = inv == k
            if m.sum() < 5:
                continue
            p_k = preds[m]; t_k = targs[m]
            if np.std(p_k) < 1e-12 or np.std(t_k) < 1e-12:
                continue
            rho, _ = spearmanr(p_k, t_k)
            if not np.isnan(rho):
                ic_per_day.append(float(rho))
        return float(np.mean(ic_per_day)) if ic_per_day else 0.0

    # ────────────────────────────────────────────────────────────── global
    def train_global(self, train_loader, val_loader, test_loader, save_path,
                      data_loader_obj=None):
        """Train the model with val rank-IC early stopping.

        Jury 2 fix F14 (2026-05-08): EVERY arm now uses val-rank-IC for
        early stopping (Track-B included). Previously Track-B disabled
        early stopping because val MSE on z-scored close was meaningless
        for it; rank-IC is meaningful for both arms (it directly measures
        cross-sectional ranking quality, which is what the eval pipeline
        scores). Patience is interpreted as "stop if rank-IC hasn't
        improved for `patience` epochs."

        Args:
            data_loader_obj : the UnifiedDataLoader instance. Required —
                              we need its `val_anchor_date` to group
                              predictions by calendar date for IC.
                              For backward compat, accepts None and
                              falls back to legacy val-MSE early stop.
        """
        use_ic_es = data_loader_obj is not None and hasattr(data_loader_obj, "val_anchor_date")
        if use_ic_es:
            mode = "max"
            print(f"[Trainer] Early stopping on val rank-IC "
                  f"(patience={self.args.patience}, mode={mode}).")
        else:
            mode = "min"
            print(f"[Trainer] Early stopping on val MSE (legacy fallback; "
                  f"data_loader_obj not provided).")
        early_stopping = EarlyStopping(patience=self.args.patience,
                                        verbose=True, mode=mode)

        # For Track-B we still keep a "best val MSE" diagnostic checkpoint.
        best_val_path = None
        if self.use_risk_head:
            best_val_path = save_path.replace(".pth", "_bestval.pth")
            best_val_loss = float("inf")

        for epoch in range(self.args.epochs):
            t1 = time.time()
            if self.use_risk_head:
                self.criterion.step_epoch(epoch)
            train_loss, parts = self.train_epoch(train_loader)

            # Diagnostic val MSE (lower is better; logged either way).
            eval_crit = nn.MSELoss() if self.use_risk_head else self.criterion
            val_metrics = evaluate(self.model, val_loader, self.device, eval_crit)
            val_loss = val_metrics["loss"]

            test_metrics = evaluate(self.model, test_loader, self.device, eval_crit)
            test_loss = test_metrics["loss"]

            # Val rank-IC (higher is better; the headline early-stop metric).
            if use_ic_es:
                val_ic = self._compute_val_rank_ic(
                    val_loader, data_loader_obj.val_anchor_date)
            else:
                val_ic = float("nan")

            print(f"Epoch: {epoch+1} | Train Loss: {train_loss:.5f} "
                  f"Val MSE: {val_loss:.5f} Test MSE: {test_loss:.5f} "
                  f"Val IC: {val_ic:+.4f} | "
                  f"Time: {time.time()-t1:.2f}s"
                  f"{self._format_parts(parts)}")

            metric_for_es = val_ic if use_ic_es else val_loss
            early_stopping(metric_for_es, self.model, save_path)
            if early_stopping.early_stop:
                print("Early stopping")
                break

            # Track-B diagnostic: parallel best-val-MSE checkpoint.
            if best_val_path is not None and val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(self.model.state_dict(), best_val_path)
                print(f"  [diagnostic] new best val MSE {val_loss:.6f} -> "
                      f"{os.path.basename(best_val_path)}")

            adjust_learning_rate(self.optimizer, epoch + 1, self.args)

        # Load back the best (highest val IC, or lowest val MSE in fallback)
        # checkpoint that EarlyStopping saved.
        self.model.load_state_dict(torch.load(save_path))
        return self.model

    # ────────────────────────────────────────────────────────────── sequential
    def train_sequential(self, data_loader_obj, val_loader, test_loader, save_path):
        """data_loader_obj is the UnifiedDataLoader instance. We call its
        `iter_train_loaders()` method ONCE PER ROUND to get a fresh generator,
        so each round re-iterates the full stock pool with only one stock's
        DataLoader resident in memory at a time. Scientifically equivalent to
        the previous list-based approach (same data, same order, same model
        state propagation) but bounded memory footprint.
        """
        # No early stopping to mimic catastrophic forgetting effect if present
        best_val = float('inf')
        epochs_per_stock = getattr(self.args, 'epochs_per_stock', 10)
        total_stocks = len(data_loader_obj.train_stocks)
        import gc

        for r in range(self.args.rounds):
            t1 = time.time()
            if self.use_risk_head:
                # In sequential mode each "round" is the analogue of one epoch
                # over the stock universe; advance the schedule per round.
                self.criterion.step_epoch(r)
            print(f"Starting Round {r+1}/{self.args.rounds} over ~{total_stocks} stocks "
                  f"({epochs_per_stock} epochs/stock)")

            train_losses = []
            last_parts: dict = {}
            # Fresh generator each round (generators are single-use)
            for idx, stock_loader in enumerate(data_loader_obj.iter_train_loaders()):
                stock_t = time.time()
                stock_losses = []
                for ep in range(epochs_per_stock):
                    loss, parts = self.train_epoch(stock_loader)
                    stock_losses.append(loss)
                    if parts:
                        last_parts = parts
                avg_loss = sum(stock_losses) / len(stock_losses)
                train_losses.append(avg_loss)
                if idx % 50 == 0:
                    print(f"  Stock {idx}/{total_stocks} | "
                          f"Avg Loss: {avg_loss:.5f} | Time: {time.time()-stock_t:.1f}s")
                # Explicit cleanup so the underlying dataset/sequences free
                # their numpy arrays before the next stock is built.
                del stock_loader
                gc.collect()

            train_loss = sum(train_losses) / len(train_losses) if train_losses else float('nan')

            eval_crit = nn.MSELoss() if self.use_risk_head else self.criterion
            val_metrics = evaluate(self.model, val_loader, self.device, eval_crit)
            val_loss = val_metrics["loss"]

            print(f"Round: {r+1} | Train Loss: {train_loss:.5f} Vali Loss: {val_loss:.5f} | "
                  f"Time: {time.time()-t1:.2f}s"
                  f"{self._format_parts(last_parts)}")

            if val_loss < best_val:
                best_val = val_loss
                torch.save(self.model.state_dict(), save_path)
                print(f"  Saved best model at round {r+1}")

            adjust_learning_rate(self.optimizer, r + 1, self.args)

        self.model.load_state_dict(torch.load(save_path))
        return self.model
