import argparse
import os
import random
import time
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split

from datasets.hourly_window import HourlySeq2SeqDataset
from models.seq2seq_lstm import Seq2Seq


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def parse_args() -> argparse.Namespace:
    print("\n" + "=" * 80)
    print("TRAINING SCRIPT VERSION: 2024-11-15-v2 with ETA tracking ENABLED")
    print("=" * 80 + "\n")
    
    p = argparse.ArgumentParser(description="Train Seq2Seq LSTM for 48h hourly forecast (GPU-ready).")
    p.add_argument("--csv", type=str, required=True, help="Path to formatted features CSV (semicolon separated).")
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--window-size", type=int, default=168)
    p.add_argument("--horizon", type=int, default=48)
    p.add_argument("--hidden-size", type=int, default=128)
    p.add_argument("--layers", type=int, default=2)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--teacher-forcing", type=float, default=0.5)
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
    # Ensure required scaled columns exist (created by prepare_csv_features.py)
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
        extra_enc_cols=[c for c in df.columns if c.startswith("consumption_lag_")],  # include lag feats if present
    )

    total_len = len(dataset)
    if total_len == 0:
        raise ValueError("No samples were generated. Check your data/parameters.")

    test_len = int(total_len * args.test_split)
    val_len = int(total_len * args.val_split)
    train_len = total_len - val_len - test_len

    train_set, val_set, test_set = random_split(dataset, [train_len, val_len, test_len], generator=torch.Generator().manual_seed(args.seed))

    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, collate_fn=collate_fn, pin_memory=True)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, collate_fn=collate_fn, pin_memory=True) if val_len > 0 else None
    test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, collate_fn=collate_fn, pin_memory=True) if test_len > 0 else None

    model = Seq2Seq(
        enc_input_size=len(dataset.enc_cols),
        dec_known_size=len(dataset.dec_cols),
        hidden_size=args.hidden_size,
        num_layers=args.layers,
        dropout=args.dropout,
        use_group_embedding=True,
        num_groups=dataset.num_groups,
        group_emb_dim=16,
        encoder_bidirectional=False,
    ).to(device)
    
    # Enable multi-GPU training if available
    if torch.cuda.device_count() > 1:
        print(f"[multi-gpu] Using {torch.cuda.device_count()} GPUs!")
        model = nn.DataParallel(model)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.MSELoss()

    best_val = float("inf")
    os.makedirs(args.save_dir, exist_ok=True)
    ckpt_path = Path(args.save_dir) / "seq2seq_hourly.pt"

    # ETA tracking
    epoch_times = []
    training_start_time = time.time()
    print("\n" + "=" * 80)
    print("STARTING TRAINING WITH ETA TRACKING ENABLED")
    print("=" * 80)

    for epoch in range(1, args.epochs + 1):
        epoch_start_time = time.time()
        print(f"\n[DEBUG] Starting epoch {epoch}/{args.epochs} at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        model.train()
        train_loss = 0.0
        total_batches = len(train_loader)
        samples_seen = 0
        
        for batch_idx, (enc_x, dec_known, y, gidx) in enumerate(train_loader, 1):
            batch_start = time.time()
            enc_x = enc_x.to(device, non_blocking=True)
            dec_known = dec_known.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            gidx = gidx.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            preds = model(enc_x, dec_known, y, gidx, teacher_forcing_ratio=args.teacher_forcing)
            loss = criterion(preds, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            batch_samples = enc_x.size(0)
            train_loss += loss.item() * batch_samples
            samples_seen += batch_samples
            
            # Progress update every 10 batches or at key milestones
            if batch_idx % 10 == 0 or batch_idx == 1 or batch_idx == total_batches:
                batch_time = time.time() - batch_start
                avg_batch_time = (time.time() - epoch_start_time) / batch_idx
                eta_epoch = avg_batch_time * (total_batches - batch_idx)
                progress_pct = (batch_idx / total_batches) * 100
                current_avg_loss = train_loss / samples_seen
                
                print(f"  [Train] Batch {batch_idx}/{total_batches} ({progress_pct:.1f}%) | "
                      f"loss={loss.item():.6f} | avg={current_avg_loss:.6f} | "
                      f"ETA: {int(eta_epoch)}s ({eta_epoch/60:.1f}m)")

        train_loss /= train_len

        val_loss = None
        if val_loader is not None:
            model.eval()
            print(f"\n  [Validation] Starting validation...")
            val_start = time.time()
            s = 0.0
            n = 0
            total_val_batches = len(val_loader)
            with torch.no_grad():
                for val_batch_idx, (enc_x, dec_known, y, gidx) in enumerate(val_loader, 1):
                    enc_x = enc_x.to(device, non_blocking=True)
                    dec_known = dec_known.to(device, non_blocking=True)
                    y = y.to(device, non_blocking=True)
                    gidx = gidx.to(device, non_blocking=True)
                    preds = model(enc_x, dec_known, y, gidx, teacher_forcing_ratio=0.0)
                    loss = criterion(preds, y)
                    s += loss.item() * enc_x.size(0)
                    n += enc_x.size(0)
                    
                    # Progress update every 5 batches for validation
                    if val_batch_idx % 5 == 0 or val_batch_idx == 1 or val_batch_idx == total_val_batches:
                        progress_pct = (val_batch_idx / total_val_batches) * 100
                        print(f"  [Val] Batch {val_batch_idx}/{total_val_batches} ({progress_pct:.1f}%)")
                        
            val_loss = s / max(n, 1)
            val_time = time.time() - val_start
            print(f"  [Validation] Completed in {val_time:.1f}s ({val_time/60:.1f}m)")

        # Calculate ETA
        epoch_time = time.time() - epoch_start_time
        epoch_times.append(epoch_time)
        
        # Calculate average time per epoch (use last 5 epochs if available)
        recent_times = epoch_times[-5:] if len(epoch_times) > 5 else epoch_times
        avg_epoch_time = sum(recent_times) / len(recent_times)
        
        # Estimate remaining time
        remaining_epochs = args.epochs - epoch
        eta_seconds = avg_epoch_time * remaining_epochs
        eta_str = str(timedelta(seconds=int(eta_seconds)))
        
        # Total elapsed time
        elapsed_seconds = time.time() - training_start_time
        elapsed_str = str(timedelta(seconds=int(elapsed_seconds)))
        
        # Print epoch info with ETA
        print("\n" + "-" * 80)
        epoch_info = f"[epoch {epoch:03d}/{args.epochs}] train_mse={train_loss:.6f}"
        if val_loss is not None:
            epoch_info += f"  val_mse={val_loss:.6f}"
        epoch_info += f"\n  Epoch time: {epoch_time:.1f}s ({epoch_time/60:.1f} min)"
        epoch_info += f"\n  Total elapsed: {elapsed_str}"
        epoch_info += f"\n  ETA remaining: {eta_str}"
        epoch_info += f"\n  Avg epoch time: {avg_epoch_time:.1f}s ({avg_epoch_time/60:.1f} min)"
        print(epoch_info)
        print("-" * 80)

        if val_loss is not None:
            if val_loss < best_val:
                best_val = val_loss
                # Save underlying model if using DataParallel
                model_to_save = model.module if isinstance(model, nn.DataParallel) else model
                torch.save({"model_state_dict": model_to_save.state_dict(), "config": vars(args)}, ckpt_path)
                print(f"  [CHECKPOINT] New best model saved! val_mse={best_val:.6f}")
                patience_counter = 0
            else:
                patience_counter += 1
                print(f"  [WARNING] No improvement for {patience_counter} epoch(s) (patience={args.early_stopping_patience})")
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
                preds = model(enc_x, dec_known, y, gidx, teacher_forcing_ratio=0.0)
                loss = criterion(preds, y)
                s += loss.item() * enc_x.size(0)
                n += enc_x.size(0)
        test_mse = s / max(n, 1)
        print(f"[test] mse={test_mse:.6f}")


if __name__ == "__main__":
    main()


