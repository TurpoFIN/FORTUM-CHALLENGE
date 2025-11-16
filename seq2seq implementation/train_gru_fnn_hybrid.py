"""
Train GRU+FNN hybrid model for 48-hour forecasting.

This hybrid model:
- Uses GRU encoder for temporal pattern extraction
- Uses FNN decoder for direct multi-step prediction (faster than autoregressive)
"""

import argparse
import os
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split

from datasets.hourly_window import HourlySeq2SeqDataset
from models.gru_fnn_hybrid import GRUFNNHybrid


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train GRU+FNN hybrid for 48h hourly forecast (GPU-ready).")
    p.add_argument("--csv", type=str, required=True, help="Path to formatted features CSV (semicolon separated).")
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--window-size", type=int, default=168)
    p.add_argument("--horizon", type=int, default=48)
    p.add_argument("--hidden-size", type=int, default=128)
    p.add_argument("--layers", type=int, default=2)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--decoder-dims", type=int, nargs="+", default=[256, 128], help="FNN decoder hidden dimensions")
    p.add_argument("--val-split", type=float, default=0.1, help="Fraction for validation split.")
    p.add_argument("--test-split", type=float, default=0.1, help="Fraction for test split.")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--save-dir", type=str, default="artifacts/models")
    p.add_argument("--early-stopping-patience", type=int, default=10, help="Stop if val loss doesn't improve for N epochs (0 = disabled)")
    return p.parse_args()


def collate_fn(batch):
    enc_x, dec_known, y, group_idx = zip(*batch)
    enc_x = torch.stack(enc_x, dim=0)
    dec_known = torch.stack(dec_known, dim=0)
    y = torch.stack(y, dim=0)
    group_idx = torch.stack(group_idx, dim=0)
    return enc_x, dec_known, y, group_idx


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[device] using {device}")

    best_val = float("inf")
    patience_counter = 0

    df = pd.read_csv(args.csv, sep=";")
    required_cols = [
        "measured_at",
        "group_id",
        "scaled_consumption",
        "scaled_price",
        "hour_sin",
        "hour_cos",
        "day_of_week_sin",
        "day_of_week_cos",
        "month_sin",
        "month_cos",
        "is_holiday",
        "is_weekend",
    ]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}. Please run prepare_csv_features.py first.")

    dataset = HourlySeq2SeqDataset(
        df=df,
        window_size=args.window_size,
        horizon=args.horizon,
        min_samples_per_group=1,
    )

    # Split dataset
    total = len(dataset)
    test_size = int(total * args.test_split)
    val_size = int(total * args.val_split)
    train_size = total - val_size - test_size
    train_ds, val_ds, test_ds = random_split(dataset, [train_size, val_size, test_size])

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn, num_workers=args.num_workers)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn, num_workers=args.num_workers) if val_size > 0 else None
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn, num_workers=args.num_workers) if test_size > 0 else None

    # Build model
    enc_input_size = 1 + 1 + len(dataset.time_cols) + len(dataset.event_cols) + len(dataset.extra_enc_cols)
    dec_covariate_size = 1 + len(dataset.time_cols) + len(dataset.event_cols) + len(dataset.extra_dec_cols)

    model = GRUFNNHybrid(
        enc_input_size=enc_input_size,
        dec_covariate_size=dec_covariate_size,
        hidden_size=args.hidden_size,
        num_layers=args.layers,
        dropout=args.dropout,
        horizon=args.horizon,
        decoder_hidden_dims=args.decoder_dims,
        num_groups=dataset.num_groups,
    ).to(device)

    print(f"[model] GRU+FNN hybrid: enc_input={enc_input_size}, dec_cov={dec_covariate_size}, hidden={args.hidden_size}, layers={args.layers}, groups={dataset.num_groups}")

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=5, verbose=True)

    Path(args.save_dir).mkdir(parents=True, exist_ok=True)
    ckpt_path = Path(args.save_dir) / "gru_fnn_hybrid.pt"

    # Training loop
    for epoch in range(1, args.epochs + 1):
        model.train()
        s = 0.0
        n = 0
        for enc_x, dec_known, y, gidx in train_loader:
            enc_x = enc_x.to(device, non_blocking=True)
            dec_known = dec_known.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            gidx = gidx.to(device, non_blocking=True)

            optimizer.zero_grad()
            preds = model(enc_x, dec_known, gidx)
            loss = criterion(preds, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            s += loss.item() * enc_x.size(0)
            n += enc_x.size(0)
        train_loss = s / max(n, 1)

        val_loss = None
        if val_loader is not None:
            model.eval()
            s = 0.0
            n = 0
            with torch.no_grad():
                for enc_x, dec_known, y, gidx in val_loader:
                    enc_x = enc_x.to(device, non_blocking=True)
                    dec_known = dec_known.to(device, non_blocking=True)
                    y = y.to(device, non_blocking=True)
                    gidx = gidx.to(device, non_blocking=True)
                    preds = model(enc_x, dec_known, gidx)
                    loss = criterion(preds, y)
                    s += loss.item() * enc_x.size(0)
                    n += enc_x.size(0)
            val_loss = s / max(n, 1)
            scheduler.step(val_loss)

        print(f"[epoch {epoch:03d}] train_mse={train_loss:.6f}" + (f"  val_mse={val_loss:.6f}" if val_loss is not None else ""))

        if val_loss is not None:
            if val_loss < best_val:
                best_val = val_loss
                torch.save({"model_state_dict": model.state_dict(), "config": vars(args)}, ckpt_path)
                print(f"[ckpt] saved best to {ckpt_path} (val_mse={best_val:.6f})")
                patience_counter = 0
            else:
                patience_counter += 1
                if args.early_stopping_patience > 0 and patience_counter >= args.early_stopping_patience:
                    print(f"[early_stop] validation loss didn't improve for {args.early_stopping_patience} epochs. Stopping.")
                    break

    # Final test evaluation
    if test_loader is not None:
        model.eval()
        s = 0.0
        n = 0
        with torch.no_grad():
            for enc_x, dec_known, y, gidx in test_loader:
                enc_x = enc_x.to(device, non_blocking=True)
                dec_known = dec_known.to(device, non_blocking=True)
                y = y.to(device, non_blocking=True)
                gidx = gidx.to(device, non_blocking=True)
                preds = model(enc_x, dec_known, gidx)
                loss = criterion(preds, y)
                s += loss.item() * enc_x.size(0)
                n += enc_x.size(0)
        test_mse = s / max(n, 1)
        print(f"[test] mse={test_mse:.6f}")


if __name__ == "__main__":
    main()

