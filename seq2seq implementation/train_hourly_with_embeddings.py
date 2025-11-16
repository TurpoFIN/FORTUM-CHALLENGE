"""
Training script for Seq2Seq with categorical embeddings.

This script uses learned embeddings for categorical features instead of one-hot encoding,
which is more efficient and captures semantic relationships.

Usage:
    python "seq2seq implementation/train_hourly_with_embeddings.py" \
        --csv formatted_features.csv \
        --epochs 40 \
        --batch-size 1024 \
        --hidden-size 384 \
        --layers 3 \
        --save-dir artifacts/models_embeddings
"""

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
from torch.utils.data import DataLoader, Subset
from torch.amp import autocast, GradScaler
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler

from datasets.hourly_window_with_embeddings import HourlySeq2SeqDatasetWithEmbeddings
from models.seq2seq_lstm_with_embeddings import Seq2SeqWithEmbeddings


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def parse_args() -> argparse.Namespace:
    print("\n" + "=" * 80)
    print("TRAINING SCRIPT VERSION: 2024-11-15-EMBEDDINGS with ETA tracking")
    print("Using LEARNED EMBEDDINGS for categorical features (more efficient!)")
    print("=" * 80 + "\n")
    
    p = argparse.ArgumentParser(description="Train Seq2Seq LSTM with categorical embeddings.")
    p.add_argument("--csv", type=str, required=True, help="Path to formatted features CSV.")
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--window-size", type=int, default=168)
    p.add_argument("--horizon", type=int, default=48)
    p.add_argument("--hidden-size", type=int, default=128)
    p.add_argument("--layers", type=int, default=2)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--teacher-forcing", type=float, default=0.5)
    p.add_argument("--val-split", type=float, default=0.1)
    p.add_argument("--test-split", type=float, default=0.1)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--save-dir", type=str, default="artifacts/models_embeddings")
    p.add_argument("--early-stopping-patience", type=int, default=10)
    p.add_argument("--group-emb-dim", type=int, default=16, help="Dimension of group embeddings")
    # Rolling-origin validation options
    p.add_argument("--rolling-folds", type=int, default=0, help="If >0, use rolling-origin validation with this many folds")
    p.add_argument("--rolling-horizon", type=int, default=None, help="Horizon (in hours) for each fold (defaults to --horizon)")
    p.add_argument("--rolling-stride", type=int, default=None, help="Stride (in hours) between folds (defaults to --horizon)")
    # Performance options
    p.add_argument("--no-amp", action="store_true", help="Disable automatic mixed precision")
    p.add_argument("--prefetch-factor", type=int, default=4, help="DataLoader prefetch factor per worker (only if num_workers>0)")
    return p.parse_args()


def collate_fn_with_categorical(batch):
    """Custom collate function that handles categorical indices."""
    enc_x, dec_known, y, group_idx, categorical_indices = zip(*batch)
    
    # Stack regular tensors
    enc_x = torch.stack(enc_x)
    dec_known = torch.stack(dec_known)
    y = torch.stack(y)
    group_idx = torch.stack(group_idx)
    
    # Combine categorical indices
    combined_categorical = {}
    if categorical_indices and len(categorical_indices[0]) > 0:
        # Get all categorical feature names
        feat_names = categorical_indices[0].keys()
        for feat_name in feat_names:
            combined_categorical[feat_name] = torch.stack([item[feat_name] for item in categorical_indices])
    
    return enc_x, dec_known, y, group_idx, combined_categorical


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    # DDP detection (torchrun)
    ddp_enabled = "LOCAL_RANK" in os.environ
    local_rank = int(os.environ.get("LOCAL_RANK", "0")) if ddp_enabled else 0
    world_size = int(os.environ.get("WORLD_SIZE", "1")) if ddp_enabled else 1

    if ddp_enabled:
        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend="nccl")
        device = torch.device(f"cuda:{local_rank}")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    torch.backends.cudnn.benchmark = True
    if (not ddp_enabled) or local_rank == 0:
        print(f"[device] using {device}  ddp={'ON' if ddp_enabled else 'OFF'} world_size={world_size}")
    use_amp = (not args.no_amp) and (device.type == "cuda")
    if (not ddp_enabled) or local_rank == 0:
        print(f"[perf] AMP={'ON' if use_amp else 'OFF'}  cudnn.benchmark={'ON' if torch.backends.cudnn.benchmark else 'OFF'}")

    # Load data
    df = pd.read_csv(args.csv, sep=";")
    if (not ddp_enabled) or local_rank == 0:
        print(f"[data] Loaded {len(df)} rows, {len(df.columns)} columns")
    
    # Define which categorical features to embed
    # These columns should exist in the CSV as non-one-hot-encoded values
    categorical_cols = {
        'province': 'province',
        'customer_type': 'customer_type',
        'price_type': 'price_type',
        'consumption_level': 'consumption_level',
    }
    
    # Check which categorical columns are available
    available_categorical = {k: v for k, v in categorical_cols.items() if v in df.columns}
    if not available_categorical:
        print("[warn] No categorical columns found! Make sure your CSV has non-one-hot encoded categorical columns.")
        print(f"[warn] Looking for: {list(categorical_cols.values())}")
        print(f"[warn] Available columns: {list(df.columns[:20])}...")
    else:
        print(f"[categorical] Using embeddings for: {list(available_categorical.keys())}")
    
    # Detect lag features
    lag_cols = [col for col in df.columns if 'lag' in col.lower()]
    
    # Create dataset
    dataset = HourlySeq2SeqDatasetWithEmbeddings(
        df=df,
        window_size=args.window_size,
        horizon=args.horizon,
        extra_enc_cols=lag_cols,
        categorical_cols=available_categorical,
    )
    
    if (not ddp_enabled) or local_rank == 0:
        print(f"[dataset] {len(dataset)} samples from {dataset.num_groups} groups")
    
    # Get categorical config for model
    categorical_config = dataset.get_categorical_config()
    if categorical_config:
        print(f"[embeddings] Categorical feature configuration:")
        total_one_hot = 0
        total_embedded = 0
        for feat_name, (num_cats, emb_dim) in categorical_config.items():
            print(f"  {feat_name}: {num_cats} categories → {emb_dim}-dim embedding (vs {num_cats} one-hot)")
            total_one_hot += num_cats
            total_embedded += emb_dim
        print(f"[embeddings] Total: {total_one_hot} one-hot features → {total_embedded} embedded features")
        print(f"[embeddings] Reduction: {100 * (1 - total_embedded/total_one_hot):.1f}%")
    
    # Split dataset by TIME (temporal split for time series)
    # This ensures train < val < test chronologically (no data leakage)
    if (not ddp_enabled) or local_rank == 0:
        print(f"[split] Using TEMPORAL split (train < val < test by time)")
    
    # Extract timestamps for each sample to sort by time
    sample_timestamps = []
    for idx in range(len(dataset)):
        grp_frame_idx, start, end = dataset.samples[idx]
        gdf = dataset.group_frames[grp_frame_idx]
        # Get the start timestamp of the encoder window
        if 'measured_at' in gdf.columns:
            sample_timestamp = pd.to_datetime(gdf.iloc[start]['measured_at'])
        else:
            # Fallback: use index if timestamp column not available
            sample_timestamp = pd.to_datetime(gdf.index[start]) if hasattr(gdf.index, 'start') else pd.Timestamp('2000-01-01')
        sample_timestamps.append((idx, sample_timestamp))
    
    # Sort by timestamp
    sample_timestamps.sort(key=lambda x: x[1])
    sorted_indices = [idx for idx, _ in sample_timestamps]
    
    # Split by time (not randomly)
    total_len = len(dataset)
    train_len = int(total_len * (1 - args.val_split - args.test_split))
    val_len = int(total_len * args.val_split)
    test_len = total_len - train_len - val_len
    
    train_indices = sorted_indices[:train_len]
    val_indices = sorted_indices[train_len:train_len + val_len]
    test_indices = sorted_indices[train_len + val_len:]
    
    # Create subset datasets
    train_ds = Subset(dataset, train_indices)
    val_ds = Subset(dataset, val_indices)
    test_ds = Subset(dataset, test_indices)
    
    # Print time ranges
    if len(train_indices) > 0:
        train_start_idx = train_indices[0]
        train_end_idx = train_indices[-1]
        train_start_ts = sample_timestamps[0][1]
        train_end_ts = sample_timestamps[train_len - 1][1]
        if (not ddp_enabled) or local_rank == 0:
            print(f"[split] Train: {len(train_ds)} samples ({train_start_ts} to {train_end_ts})")
    
    if len(val_indices) > 0:
        val_start_ts = sample_timestamps[train_len][1]
        val_end_ts = sample_timestamps[train_len + val_len - 1][1]
        if (not ddp_enabled) or local_rank == 0:
            print(f"[split] Val: {len(val_ds)} samples ({val_start_ts} to {val_end_ts})")
    
    if len(test_indices) > 0:
        test_start_ts = sample_timestamps[train_len + val_len][1]
        test_end_ts = sample_timestamps[-1][1]
        if (not ddp_enabled) or local_rank == 0:
            print(f"[split] Test: {len(test_ds)} samples ({test_start_ts} to {test_end_ts})")
    
    # Create dataloaders (with persistent workers/prefetch)
    common_loader_kwargs = dict(
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        collate_fn=collate_fn_with_categorical,
        pin_memory=True if torch.cuda.is_available() else False,
        persistent_workers=True if args.num_workers and args.num_workers > 0 else False,
    )
    if args.num_workers and args.num_workers > 0 and args.prefetch_factor and args.prefetch_factor > 0:
        common_loader_kwargs["prefetch_factor"] = args.prefetch_factor

    # Samplers for DDP
    train_sampler = DistributedSampler(train_ds, num_replicas=world_size, rank=local_rank, shuffle=True, drop_last=False) if ddp_enabled else None
    val_sampler = DistributedSampler(val_ds, num_replicas=world_size, rank=local_rank, shuffle=False, drop_last=False) if ddp_enabled and val_len > 0 and args.rolling_folds <= 0 else None

    train_loader = DataLoader(
        train_ds,
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        **common_loader_kwargs,
    )
    val_loader = None
    if args.rolling_folds <= 0 and val_len > 0:
        val_loader = DataLoader(
            val_ds,
            shuffle=False if val_sampler is None else False,
            sampler=val_sampler,
            **common_loader_kwargs,
        )

    # Precompute structures for rolling-origin validation (timestamp-indexed)
    rolling_cfg = None
    if args.rolling_folds and args.rolling_folds > 0:
        # Build per-sample timestamps vector aligned to dataset indices
        sample_ts_all = [ts for _, ts in sample_timestamps]  # same order as dataset index
        # Unique hourly timestamps in ascending order
        unique_ts = sorted(set(sample_ts_all))
        rolling_h = int(args.rolling_horizon or args.horizon)
        rolling_stride = int(args.rolling_stride or args.horizon)

        # Determine fold starts (by timestamp index). Start from the latest possible window.
        starts: list[int] = []
        cursor = len(unique_ts) - rolling_h
        for _ in range(args.rolling_folds):
            if cursor < 0:
                break
            starts.append(cursor)
            cursor -= rolling_stride
        # Reverse to go from older to newer folds
        starts = list(reversed(starts))
        if (not ddp_enabled) or local_rank == 0:
            print(f"[rolling] folds={len(starts)} horizon={rolling_h}h stride={rolling_stride}h")
            if starts:
                first_start_ts = unique_ts[starts[0]]
                last_end_ts = unique_ts[starts[-1] + rolling_h - 1]
                print(f"[rolling] coverage: {first_start_ts} → {last_end_ts}")
        rolling_cfg = {
            "unique_ts": unique_ts,
            "starts": starts,
            "h": rolling_h,
        }
    
    # Determine feature sizes (continuous features only)
    sample_enc, sample_dec, _, _, _ = dataset[0]
    enc_input_size = sample_enc.shape[1]
    dec_known_size = sample_dec.shape[1]
    
    if (not ddp_enabled) or local_rank == 0:
        print(f"[model] Encoder input size (continuous): {enc_input_size}")
        print(f"[model] Decoder input size (continuous): {dec_known_size}")
    
    # Create model
    model = Seq2SeqWithEmbeddings(
        enc_input_size=enc_input_size,
        dec_known_size=dec_known_size,
        hidden_size=args.hidden_size,
        num_layers=args.layers,
        dropout=args.dropout,
        use_group_embedding=True,
        num_groups=dataset.num_groups,
        group_emb_dim=args.group_emb_dim,
        categorical_config=categorical_config if categorical_config else None,
    ).to(device)

    if ddp_enabled:
        model = DDP(model, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=False)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    if (not ddp_enabled) or local_rank == 0:
        print(f"[model] Total parameters: {total_params:,} ({trainable_params:,} trainable)")
    
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.MSELoss()
    scaler = GradScaler('cuda', enabled=use_amp)
    
    best_val = float("inf")
    os.makedirs(args.save_dir, exist_ok=True)
    ckpt_path = Path(args.save_dir) / "seq2seq_hourly_embeddings.pt"
    
    # ETA tracking
    epoch_times = []
    training_start_time = time.time()
    if (not ddp_enabled) or local_rank == 0:
        print("\n" + "=" * 80)
        print("STARTING TRAINING WITH CATEGORICAL EMBEDDINGS")
        print("=" * 80)
    
    patience_counter = 0
    
    for epoch in range(1, args.epochs + 1):
        epoch_start_time = time.time()
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)
        if (not ddp_enabled) or local_rank == 0:
            print(f"\n[DEBUG] Starting epoch {epoch}/{args.epochs} at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        model.train()
        train_loss = 0.0
        total_batches = len(train_loader)
        samples_seen = 0
        
        for batch_idx, (enc_x, dec_known, y, gidx, cat_indices) in enumerate(train_loader, 1):
            enc_x = enc_x.to(device, non_blocking=True)
            dec_known = dec_known.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            gidx = gidx.to(device, non_blocking=True)
            
            # Move categorical indices to device
            cat_indices_device = {k: v.to(device, non_blocking=True) for k, v in cat_indices.items()}
            
            optimizer.zero_grad(set_to_none=True)
            with autocast('cuda', enabled=use_amp):
                preds = model(
                    enc_x, 
                    dec_known, 
                    y, 
                    gidx, 
                    categorical_indices=cat_indices_device if cat_indices_device else None,
                    teacher_forcing_ratio=args.teacher_forcing
                )
                loss = criterion(preds, y)
            scaler.scale(loss).backward()
            # Unscale before clipping to avoid overflow
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            
            batch_samples = enc_x.size(0)
            train_loss += loss.item() * batch_samples
            samples_seen += batch_samples
            
            # Progress update
            if ((not ddp_enabled) or local_rank == 0) and (batch_idx % 10 == 0 or batch_idx == 1 or batch_idx == total_batches):
                avg_batch_time = (time.time() - epoch_start_time) / batch_idx
                eta_epoch = avg_batch_time * (total_batches - batch_idx)
                progress_pct = (batch_idx / total_batches) * 100
                current_avg_loss = train_loss / samples_seen
                
                print(f"  [Train] Batch {batch_idx}/{total_batches} ({progress_pct:.1f}%) | "
                      f"loss={loss.item():.6f} | avg={current_avg_loss:.6f} | "
                      f"ETA: {int(eta_epoch)}s ({eta_epoch/60:.1f}m)")
        
        train_loss /= len(train_loader.dataset)
        
        # Validation
        val_loss = None
        if val_loader is not None or rolling_cfg is not None:
            model.eval()
            if rolling_cfg is None:
                # Single holdout validation (legacy)
                if (not ddp_enabled) or local_rank == 0:
                    print(f"\n  [Validation] Starting validation...")
                val_start = time.time()
                s = 0.0
                n = 0
                total_val_batches = len(val_loader)
                with torch.no_grad():
                    for val_batch_idx, (enc_x, dec_known, y, gidx, cat_indices) in enumerate(val_loader, 1):
                        enc_x = enc_x.to(device, non_blocking=True)
                        dec_known = dec_known.to(device, non_blocking=True)
                        y = y.to(device, non_blocking=True)
                        gidx = gidx.to(device, non_blocking=True)
                        cat_indices_device = {k: v.to(device, non_blocking=True) for k, v in cat_indices.items()}
                        
                        with autocast('cuda', enabled=use_amp):
                            preds = model(
                                enc_x, 
                                dec_known, 
                                y, 
                                gidx, 
                                categorical_indices=cat_indices_device if cat_indices_device else None,
                                teacher_forcing_ratio=0.0
                            )
                        loss = criterion(preds, y)
                        s += loss.item() * enc_x.size(0)
                        n += enc_x.size(0)
                        
                        if ((not ddp_enabled) or local_rank == 0) and (val_batch_idx % 5 == 0 or val_batch_idx == 1 or val_batch_idx == total_val_batches):
                            progress_pct = (val_batch_idx / total_val_batches) * 100
                            print(f"  [Val] Batch {val_batch_idx}/{total_val_batches} ({progress_pct:.1f}%)")
                # Reduce across ranks
                if ddp_enabled:
                    t = torch.tensor([s, n], dtype=torch.float64, device=device)
                    dist.all_reduce(t, op=dist.ReduceOp.SUM)
                    s, n = t.tolist()
                val_loss = s / max(n, 1)
                val_time = time.time() - val_start
                if (not ddp_enabled) or local_rank == 0:
                    print(f"  [Validation] Completed in {val_time:.1f}s ({val_time/60:.1f}m)")
            else:
                # Rolling-origin validation over multiple folds
                if (not ddp_enabled) or local_rank == 0:
                    print(f"\n  [Rolling-Validation] Evaluating {len(rolling_cfg['starts'])} folds...")
                rv_start = time.time()
                unique_ts = rolling_cfg["unique_ts"]
                starts = rolling_cfg["starts"]
                h = rolling_cfg["h"]

                # Map timestamp to indices for quick lookup
                ts_to_indices = {}
                for idx_s, ts in enumerate(sample_ts_all):
                    ts_to_indices.setdefault(ts, []).append(idx_s)

                fold_losses = []
                with torch.no_grad():
                    for fold_idx, start_pos in enumerate(starts, 1):
                        start_ts = unique_ts[start_pos]
                        end_ts = unique_ts[start_pos + h - 1]
                        # Build validation indices for this fold
                        val_idx_fold = []
                        for pos in range(start_pos, start_pos + h):
                            ts = unique_ts[pos]
                            if ts in ts_to_indices:
                                val_idx_fold.extend(ts_to_indices[ts])
                        # Train indices: all samples strictly before start_ts
                        # Note: We do not retrain per fold here; we only evaluate validation windows
                        if not val_idx_fold:
                            continue
                        val_subset = Subset(dataset, val_idx_fold)
                        fold_loader_kwargs = dict(**common_loader_kwargs)
                        # For fold eval, reduce workers slightly to avoid contention
                        fold_loader_kwargs["num_workers"] = max(0, args.num_workers // 2)
                        val_loader_fold = DataLoader(val_subset, shuffle=False, **fold_loader_kwargs)
                        s = 0.0
                        n = 0
                        for enc_x, dec_known, y, gidx, cat_indices in val_loader_fold:
                            enc_x = enc_x.to(device, non_blocking=True)
                            dec_known = dec_known.to(device, non_blocking=True)
                            y = y.to(device, non_blocking=True)
                            gidx = gidx.to(device, non_blocking=True)
                            cat_indices_device = {k: v.to(device, non_blocking=True) for k, v in cat_indices.items()}
                            with autocast('cuda', enabled=use_amp):
                                preds = model(
                                    enc_x,
                                    dec_known,
                                    y,
                                    gidx,
                                    categorical_indices=cat_indices_device if cat_indices_device else None,
                                    teacher_forcing_ratio=0.0
                                )
                            loss = criterion(preds, y)
                            s += loss.item() * enc_x.size(0)
                            n += enc_x.size(0)
                        # Reduce across ranks
                        if ddp_enabled:
                            t = torch.tensor([s, n], dtype=torch.float64, device=device)
                            dist.all_reduce(t, op=dist.ReduceOp.SUM)
                            s, n = t.tolist()
                        fold_mse = s / max(n, 1)
                        fold_losses.append(fold_mse)
                        if (not ddp_enabled) or local_rank == 0:
                            print(f"    [Fold {fold_idx}/{len(starts)}] {start_ts} → {end_ts}  val_mse={fold_mse:.6f}")
                # Average fold losses
                val_loss = float(np.mean(fold_losses)) if fold_losses else None
                rv_time = time.time() - rv_start
                if (not ddp_enabled) or local_rank == 0:
                    if val_loss is not None:
                        print(f"  [Rolling-Validation] avg_val_mse={val_loss:.6f} over {len(fold_losses)} folds in {rv_time:.1f}s")
                    else:
                        print(f"  [Rolling-Validation] No folds evaluated")
        
        # Calculate ETA
        epoch_time = time.time() - epoch_start_time
        epoch_times.append(epoch_time)
        recent_times = epoch_times[-5:] if len(epoch_times) > 5 else epoch_times
        avg_epoch_time = sum(recent_times) / len(recent_times)
        remaining_epochs = args.epochs - epoch
        eta_seconds = avg_epoch_time * remaining_epochs
        eta_str = str(timedelta(seconds=int(eta_seconds)))
        elapsed_seconds = time.time() - training_start_time
        elapsed_str = str(timedelta(seconds=int(elapsed_seconds)))
        
        # Print epoch summary
        if (not ddp_enabled) or local_rank == 0:
            print("\n" + "-" * 80)
        epoch_info = f"[epoch {epoch:03d}/{args.epochs}] train_mse={train_loss:.6f}"
        if val_loss is not None:
            epoch_info += f"  val_mse={val_loss:.6f}"
        epoch_info += f"\n  Epoch time: {epoch_time:.1f}s ({epoch_time/60:.1f} min)"
        epoch_info += f"\n  Total elapsed: {elapsed_str}"
        epoch_info += f"\n  ETA remaining: {eta_str}"
        epoch_info += f"\n  Avg epoch time: {avg_epoch_time:.1f}s ({avg_epoch_time/60:.1f} min)"
        if (not ddp_enabled) or local_rank == 0:
            print(epoch_info)
            print("-" * 80)
        
        # Save best model
        if val_loss is not None:
            if val_loss < best_val:
                best_val = val_loss
                if (not ddp_enabled) or local_rank == 0:
                    model_to_save = model.module if isinstance(model, DDP) else model
                    torch.save({
                        "model_state_dict": model_to_save.state_dict(),
                        "config": vars(args),
                        "categorical_config": categorical_config,
                        "categorical_encoders": dataset.categorical_encoders,
                    }, ckpt_path)
                    print(f"  [CHECKPOINT] New best model saved! val_mse={best_val:.6f}")
                patience_counter = 0
            else:
                patience_counter += 1
                if (not ddp_enabled) or local_rank == 0:
                    print(f"  [WARNING] No improvement for {patience_counter} epoch(s) (patience={args.early_stopping_patience})")
                if args.early_stopping_patience > 0 and patience_counter >= args.early_stopping_patience:
                    if (not ddp_enabled) or local_rank == 0:
                        print(f"[early_stop] Stopping early after {epoch} epochs")
                    break
    
    if (not ddp_enabled) or local_rank == 0:
        print(f"\n[done] Training completed! Best val_mse={best_val:.6f}")
        print(f"[saved] Model saved to {ckpt_path}")

    if ddp_enabled:
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()

