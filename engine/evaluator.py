import torch
import numpy as np
from utils.metrics import metric

def evaluate(model, test_loader, device, criterion=None,
             close_min=None, close_max=None):
    """Run model over test_loader and compute regression metrics.

    Scaled-space metrics (MSE/MAE/RMSE/R²) are always returned. If both
    ``close_min`` and ``close_max`` arrays are supplied (each of shape [N_samples],
    aligned with the loader's iteration order — only valid when the loader has
    shuffle=False), the function ALSO computes dollar-space metrics by
    inverse-transforming Close predictions/targets per-sample using the
    MinMaxScaler formula:

        x_orig = x_scaled * (close_max - close_min) + close_min

    R² is scale-invariant so it is the same in both spaces; we still report it
    once. Dollar-space MSE/MAE are interpretable as USD² and USD respectively.
    """
    model.eval()
    preds = []
    trues = []
    total_loss = 0

    if criterion is None:
        criterion = torch.nn.MSELoss()

    with torch.no_grad():
        for batch in test_loader:
            # v2 loader returns (X, y_main, y_logret); legacy 2-tuple support
            if len(batch) == 3:
                batch_x, batch_y, y_logret = batch
                y_logret_d = y_logret.float().to(device)
            else:
                batch_x, batch_y = batch
                y_logret_d = None
            batch_x = batch_x.float().to(device)
            batch_y = batch_y.float().to(device)

            # Forward pass + per-arm scoring (Jury 2 fix B1+B2, 2026-05-08).
            # Three cases:
            #   * RiskAwareHead   -> dict; score mu_return_H against y_logret.
            #   * MSEReturnHead   -> [B] log-return; score against y_logret.
            #   * AdaPatch / legacy backbone -> [B, pred_len] z-scored close
            #     window; score against batch_y (the legacy z-scored target).
            outputs = model(batch_x, None)
            if isinstance(outputs, dict):
                pred = outputs["mu_return_H"]
                target = y_logret_d if y_logret_d is not None else batch_y
            elif isinstance(outputs, tuple):
                # AdaPatch path: still scoring on z-scored close (the model's
                # native target, used for its reconstruction co-loss).
                pred = outputs[0]
                target = batch_y
            elif outputs.dim() == 1 or (outputs.dim() == 2 and outputs.shape[1] == 1):
                # MSEReturnHead: [B] or [B, 1] log-return scalar.
                pred = outputs.squeeze(-1) if outputs.dim() == 2 else outputs
                target = y_logret_d if y_logret_d is not None else batch_y
            else:
                # Legacy backbone: [B, pred_len] z-scored close window.
                pred = outputs
                target = batch_y

            loss = criterion(pred, target)
            total_loss += loss.item()

            preds.append(pred.detach().cpu().numpy())
            trues.append(target.detach().cpu().numpy())

    # Guard against empty loader (e.g. n_val=0 at H=120 after embargo —
    # can happen on 1-year folds with long horizons). Return NaN metrics
    # so trainer can keep going without crashing; early stopping then
    # never receives an improvement signal and falls back to last-epoch.
    if not preds:
        return {
            "loss": float("nan"),
            "mse": float("nan"), "mae": float("nan"),
            "rmse": float("nan"), "r2": float("nan"),
        }
    preds = np.concatenate(preds, axis=0)
    trues = np.concatenate(trues, axis=0)

    mae, mse, rmse, r2 = metric(preds, trues)

    result = {
        "loss": total_loss / max(len(test_loader), 1),
        "mse": mse,
        "mae": mae,
        "rmse": rmse,
        "r2": r2,
    }

    # Dollar-space metrics if per-sample inverse-scale arrays were supplied
    if close_min is not None and close_max is not None:
        N = preds.shape[0]
        if len(close_min) != N or len(close_max) != N:
            print(f"[evaluator] WARN: close_min/max len mismatch ({len(close_min)} vs N={N}) "
                  f"— skipping dollar-space metrics.")
        else:
            cmin = np.asarray(close_min, dtype=np.float64).reshape(-1, 1)  # [N, 1] broadcasts over pred_len
            cmax = np.asarray(close_max, dtype=np.float64).reshape(-1, 1)
            scale = cmax - cmin
            preds_usd  = preds.astype(np.float64)  * scale + cmin
            trues_usd  = trues.astype(np.float64)  * scale + cmin
            err_usd = preds_usd - trues_usd
            mse_usd  = float(np.mean(err_usd ** 2))
            mae_usd  = float(np.mean(np.abs(err_usd)))
            rmse_usd = float(np.sqrt(mse_usd))
            result["mse_usd"] = mse_usd
            result["mae_usd"] = mae_usd
            result["rmse_usd"] = rmse_usd

    return result
