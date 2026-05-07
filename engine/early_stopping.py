import math

import numpy as np
import torch


class EarlyStopping:
    """Early-stopping monitor with `mode={'min','max'}` support.

    Jury 2 fix F14 (2026-05-08): added `mode='max'` so we can early-stop
    on validation rank-IC (higher is better) instead of validation MSE
    on z-scored close (lower is better — but a noisy proxy for the
    cross-sectional ranking quality we actually care about).
    """

    def __init__(self, patience: int = 7, verbose: bool = False,
                 delta: float = 0.0, mode: str = "min"):
        if mode not in ("min", "max"):
            raise ValueError(f"mode must be 'min' or 'max'; got {mode!r}")
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.inf if mode == "min" else -np.inf
        self.delta = delta
        self.mode = mode

    def __call__(self, val_metric, model, path):
        # Convert to "higher = better" internal score regardless of mode.
        score = -val_metric if self.mode == "min" else val_metric
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_metric, model, path)
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.verbose:
                print(f"EarlyStopping counter: {self.counter} out of {self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_metric, model, path)
            self.counter = 0

    def save_checkpoint(self, val_metric, model, path):
        if self.verbose:
            arrow = "decreased" if self.mode == "min" else "increased"
            print(f"Validation metric {arrow} ({self.val_loss_min:.6f} --> "
                  f"{val_metric:.6f}). Saving model ...")
        torch.save(model.state_dict(), path)
        self.val_loss_min = val_metric


def adjust_learning_rate(optimizer, epoch: int, args):
    """Anneal LR per `args.lradj`.

    Schedules:
      type1  : LR *= 0.5 each epoch (legacy; aggressive).
      type3  : LR constant for 3 epochs, then LR *= 0.9 each epoch (legacy).
      cosine : Linear warmup for `args.warmup_epochs` (default 5) from
               `args.lr * 0.1` to `args.lr`, then cosine decay to
               `args.lr_min` (default args.lr / 100) over the remaining
               epochs (`args.epochs - warmup_epochs`).
               Recommended for transformer training (Jury 2 fix F15).
      none / unrecognised : no change.
    """
    epoch = int(epoch)
    if args.lradj == "type1":
        lr = args.lr * (0.5 ** ((epoch - 1) // 1))
        _set_lr(optimizer, lr)
    elif args.lradj == "type3":
        lr = args.lr if epoch < 3 else args.lr * (0.9 ** ((epoch - 3) // 1))
        _set_lr(optimizer, lr)
    elif args.lradj == "cosine":
        warmup = int(getattr(args, "warmup_epochs", 5))
        lr_min = float(getattr(args, "lr_min", args.lr / 100.0))
        if epoch <= warmup:
            # Linear warmup from lr*0.1 to lr over `warmup` epochs.
            t = max(epoch, 1) / max(warmup, 1)
            lr = args.lr * (0.1 + 0.9 * t)
        else:
            # Cosine decay from lr to lr_min over the remaining epochs.
            t = (epoch - warmup) / max(args.epochs - warmup, 1)
            t = min(max(t, 0.0), 1.0)
            lr = lr_min + 0.5 * (args.lr - lr_min) * (1.0 + math.cos(math.pi * t))
        _set_lr(optimizer, lr)
    # else: do nothing


def _set_lr(optimizer, lr: float):
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr
    print(f"Updating learning rate to {lr:.6g}")
