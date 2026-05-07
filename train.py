"""Training driver for the v2 finsharpe campaign.

Conventions (PLAN_v2):
  * Universe: 300 stocks from `data/universe_main.csv` (leakage-free).
  * Calendar: walk-forward fold passed via --fold (F1..F4).
  * Target:   log-return over H trading days (not MinMax-scaled price).
  * Same-stocks calendar-only split.
  * Per-stock z-score fitted on TRAIN window only (no leakage).
  * Adj-Close-adjusted OHLC + log1p Volume.
  * Purged walk-forward + embargo (default embargo = horizon).

Two arms (per cell):
  * --use_risk_head NOT set : MSE-only baseline (plain nn.MSELoss on log-return).
  * --use_risk_head set     : Track B (CompositeRiskLoss with phase schedule;
                              + RiskAwareHead wrapper -> sigma + vol + gate heads).

Usage:
    python train.py --model PatchTST --horizon 5 --fold F4
    python train.py --model GCFormer --horizon 20 --fold F4 --use_risk_head
    python train.py --model LSTM --horizon 60 --fold F1 --use_risk_head --use_xs_sharpe
"""
from __future__ import annotations

import argparse
import json
import os
import random
import time

import numpy as np
import pandas as pd
import torch

from config import (FEATURES, CLOSE_IDX, MODEL_SAVE_DIR, RESULTS_DIR,
                     SEQ_LEN, HORIZONS, WALK_FORWARD_FOLDS,
                     PATCHTST_CONFIG, TFT_CONFIG, ADAPATCH_CONFIG,
                     GCFORMER_CONFIG, ITRANSFORMER_CONFIG,
                     VANILLA_TRANSFORMER_CONFIG, DLINEAR_CONFIG,
                     LSTM_CONFIG, RNN_CONFIG, CNN_CONFIG)
from data_loader import UnifiedDataLoader
from engine.evaluator import evaluate
from engine.heads import RiskAwareHead
from engine.trainer import Trainer
from models import model_dict


def get_config_for_model(model_name: str, horizon: int) -> dict:
    base = {
        "pred_len": horizon,
        "context_len": SEQ_LEN,
        "enc_in": len(FEATURES),
    }
    cfg_map = {
        "PatchTST": PATCHTST_CONFIG, "TFT": TFT_CONFIG,
        "AdaPatch": ADAPATCH_CONFIG, "GCFormer": GCFORMER_CONFIG,
        "iTransformer": ITRANSFORMER_CONFIG,
        "VanillaTransformer": VANILLA_TRANSFORMER_CONFIG,
        "DLinear": DLINEAR_CONFIG,
        "LSTM": LSTM_CONFIG, "RNN": RNN_CONFIG, "CNN": CNN_CONFIG,
    }
    if model_name in cfg_map:
        base.update(cfg_map[model_name])
    return base


def set_seed(seed: int):
    """Deterministic-ish seeding (per Jury 1 J2). PyTorch determinism for
    CUDA isn't perfectly reproducible across hardware, but seeding all RNGs
    + setting cudnn deterministic gives same-hardware reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def main():
    p = argparse.ArgumentParser(description="finsharpe v2 training driver")

    # Required cell-identification args
    p.add_argument("--model", required=True,
                   choices=list(model_dict.keys()))
    p.add_argument("--horizon", type=int, required=True, choices=HORIZONS)
    p.add_argument("--fold", type=str, default="F4",
                   choices=[f["name"] for f in WALK_FORWARD_FOLDS],
                   help="Walk-forward CV fold to train on (default F4 headline).")

    # Training mode + arm
    p.add_argument("--method", default="global", choices=["global", "sequential"])
    p.add_argument("--use_risk_head", action="store_true",
                   help="Track-B arm: wrap backbone in RiskAwareHead and train "
                        "with CompositeRiskLoss (phase schedule). "
                        "Without this flag = MSE-only baseline arm.")
    p.add_argument("--use_xs_sharpe", action="store_true",
                   help="Track-B v2: differentiable cross-sectional Sharpe loss.")
    p.add_argument("--xs_n_subgroups", type=int, default=32)

    # Hyperparameters
    p.add_argument("--batch_size", type=int, default=128)
    p.add_argument("--epochs", type=int, default=60)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--lradj", type=str, default="type3")
    p.add_argument("--patience", type=int, default=10)
    p.add_argument("--use_amp", action="store_true")
    p.add_argument("--seed", type=int, default=2026)

    # Risk-head architecture
    p.add_argument("--risk_head_lookback", type=int, default=20)
    p.add_argument("--risk_head_d_hidden", type=int, default=64)

    # Optional fine-tune init
    p.add_argument("--init_from", type=str, default=None,
                   help="Optional pre-trained backbone checkpoint.")

    # Data layer
    p.add_argument("--max_stocks", type=int, default=None,
                   help="Cap n stocks (smoke testing).")
    p.add_argument("--num_workers", type=int, default=2)
    p.add_argument("--no_purge", action="store_true",
                   help="Disable purged walk-forward (debug only -- introduces leakage).")
    p.add_argument("--embargo_days", type=int, default=None,
                   help="Embargo days at split boundaries. Default = horizon.")

    # Sequential-mode config
    p.add_argument("--rounds", type=int, default=3)
    p.add_argument("--epochs_per_stock", type=int, default=20)
    p.add_argument("--adapatch_alpha", type=float, default=0.5)

    args = p.parse_args()
    args.model_name = args.model

    # Seed RNGs
    set_seed(args.seed)

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        torch.set_float32_matmul_precision("high")  # TF32 on Ampere+
    args.device = str(device)
    print(f"[v2] device={device} | model={args.model} | H={args.horizon} | "
          f"fold={args.fold} | arm={'riskhead' if args.use_risk_head else 'mse'} | "
          f"batch_size={args.batch_size}")

    # Build data loaders
    print("[v2] building UnifiedDataLoader ...")
    t0 = time.time()
    loader = UnifiedDataLoader(
        seq_len=SEQ_LEN, horizon=args.horizon,
        batch_size=args.batch_size, fold=args.fold,
        max_stocks=args.max_stocks, num_workers=args.num_workers,
        purge=not args.no_purge,
        embargo_days=args.embargo_days,
    )
    print(f"[v2] loader built in {time.time()-t0:.1f}s -> {loader!r}")
    train_loader = loader.get_train_loader(shuffle=True)
    val_loader = loader.get_val_loader()
    test_loader = loader.get_test_loader()

    # Build backbone
    print(f"[v2] init {args.model} (H={args.horizon}, enc_in={len(FEATURES)})")
    configs = get_config_for_model(args.model, args.horizon)
    backbone = model_dict[args.model](configs)

    # Optional pre-trained init
    if args.init_from is not None:
        print(f"[v2] init_from={args.init_from}")
        ckpt = torch.load(args.init_from, map_location="cpu")
        if any(k.startswith("backbone.") for k in ckpt.keys()):
            ckpt = {k.split("backbone.", 1)[1]: v for k, v in ckpt.items()
                    if k.startswith("backbone.")}
        missing, unexpected = backbone.load_state_dict(ckpt, strict=False)
        if missing:    print(f"  missing: {len(missing)}")
        if unexpected: print(f"  unexpected: {len(unexpected)}")

    # Wrap or not
    if args.use_risk_head:
        print(f"[v2] wrapping in RiskAwareHead "
              f"(lookback={args.risk_head_lookback}, d_hidden={args.risk_head_d_hidden})")
        model = RiskAwareHead(
            backbone=backbone, n_features=len(FEATURES),
            pred_len=args.horizon, close_idx=CLOSE_IDX,
            lookback_for_aux=args.risk_head_lookback,
            d_hidden=args.risk_head_d_hidden,
        ).to(device)
    else:
        model = backbone.to(device)

    trainer = Trainer(args, model, device)

    # Save path: encode (model, horizon, fold, arm) so different cells don't collide
    arm = "riskhead" if args.use_risk_head else "mse"
    if args.use_xs_sharpe:
        arm = "riskhead_xs"
    save_name = f"{args.model}_{args.method}_H{args.horizon}_{args.fold}_{arm}.pth"
    save_path = os.path.join(MODEL_SAVE_DIR, save_name)

    # Train
    print(f"[v2] starting training ({args.method} mode, {args.epochs} epochs)")
    t0 = time.time()
    if args.method == "global":
        model = trainer.train_global(train_loader, val_loader, test_loader, save_path)
    else:
        # Sequential mode operates per-stock; not the headline path. Kept
        # for the seqvsglob side-finding ablation only.
        model = trainer.train_sequential(loader, val_loader, test_loader, save_path)
    train_secs = time.time() - t0
    print(f"[v2] training done in {train_secs/60:.1f} min")

    # Final test eval
    print("[v2] evaluating on test set ...")
    model.load_state_dict(torch.load(save_path, map_location=device))
    test_metrics = evaluate(model, test_loader, device)

    # Result CSV row -- one row per cell, append-safe.
    res_path = os.path.join(RESULTS_DIR, f"results_{args.fold}_{arm}.csv")
    row = {
        "model": args.model,
        "horizon": args.horizon,
        "fold": args.fold,
        "arm": arm,
        "batch_size": args.batch_size,
        "seed": args.seed,
        "n_train": int(loader.sample_table_train.shape[0]),
        "n_val":   int(loader.sample_table_val.shape[0]),
        "n_test":  int(loader.sample_table_test.shape[0]),
        "n_stocks_used": len(loader.universe) - len(loader.skipped),
        "train_minutes": train_secs / 60,
        "test_loss": float(test_metrics.get("loss", float("nan"))),
        "test_mse":  float(test_metrics.get("mse", float("nan"))),
        "test_mae":  float(test_metrics.get("mae", float("nan"))),
        "ckpt_path": save_path,
    }
    df = pd.DataFrame([row])
    if os.path.exists(res_path):
        df.to_csv(res_path, mode="a", header=False, index=False)
    else:
        df.to_csv(res_path, index=False)
    print(f"[v2] result row appended -> {res_path}")
    print(json.dumps(row, indent=2))


if __name__ == "__main__":
    main()
