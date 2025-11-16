"""
Train Seq2Seq model for 12-month ahead predictions using monthly aggregated data.

This is adapted from train_hourly_with_embeddings.py for monthly resolution.

Key differences:
- Window size: 24 months (2 years)
- Horizon: 12 months (1 year)
- Smaller batch size due to limited data
- Higher dropout for regularization
- Optional transfer learning from hourly model

Usage:
    python "seq2seq implementation/train_monthly_with_embeddings.py" \
        --csv formatted_features_monthly.csv \
        --epochs 200 \
        --batch-size 64 \
        --window-size 24 \
        --horizon 12 \
        --hidden-size 256 \
        --layers 2 \
        --dropout 0.45 \
        --group-emb-dim 32 \
        --save-dir artifacts/models_monthly
"""

import argparse
import random
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau

from datasets.hourly_window_with_embeddings import HourlySeq2SeqDatasetWithEmbeddings
from models.seq2seq_lstm_with_embeddings import Seq2SeqWithEmbeddings


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train monthly Seq2Seq model with embeddings")
    
    # Data
    p.add_argument("--csv", type=str, required=True, help="Path to monthly features CSV")
    p.add_argument("--save-dir", type=str, default="artifacts/models_monthly", help="Directory to save models")
    
    # Model architecture
    p.add_argument("--window-size", type=int, default=24, help="Encoder window size (months)")
    p.add_argument("--horizon", type=int, default=12, help="Decoder forecast horizon (months)")
    p.add_argument("--hidden-size", type=int, default=256, help="LSTM hidden size")
    p.add_argument("--layers", type=int, default=2, help="Number of LSTM layers")
    p.add_argument("--dropout", type=float, default=0.45, help="Dropout rate (higher for monthly)")
    p.add_argument("--group-emb-dim", type=int, default=32, help="Group embedding dimension")
    
    # Training
    p.add_argument("--epochs", type=int, default=200, help="Training epochs")
    p.add_argument("--batch-size", type=int, default=64, help="Batch size")
    p.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    p.add_argument("--weight-decay", type=float, default=1e-4, help="Weight decay")
    p.add_argument("--teacher-forcing", type=float, default=0.5, help="Teacher forcing ratio")
    p.add_argument("--grad-clip", type=float, default=1.0, help="Gradient clipping")
    p.add_argument("--early-stopping-patience", type=int, default=30, help="Early stopping patience")
    
    # Transfer learning
    p.add_argument("--pretrained-model", type=str, default=None, help="Path to pretrained hourly model for transfer learning")
    p.add_argument("--freeze-embeddings", action="store_true", help="Freeze group embeddings during training")
    
    # System
    p.add_argument("--num-workers", type=int, default=4, help="DataLoader workers")
    p.add_argument("--seed", type=int, default=42, help="Random seed")
    
    return p.parse_args()


def load_pretrained_weights(model: nn.Module, pretrained_path: str, freeze_embeddings: bool = False):
    """Load weights from pretrained hourly model."""
    print(f"[transfer] Loading pretrained weights from {pretrained_path}")
    
    checkpoint = torch.load(pretrained_path, map_location='cpu')
    pretrained_state = checkpoint['model_state_dict']
    model_state = model.state_dict()
    
    # Transfer compatible layers
    transferred = []
    for name, param in pretrained_state.items():
        if name in model_state and model_state[name].shape == param.shape:
            model_state[name] = param
            transferred.append(name)
    
    model.load_state_dict(model_state)
    print(f"[transfer] Transferred {len(transferred)} layers")
    
    if freeze_embeddings:
        print("[transfer] Freezing group embeddings")
        if hasattr(model.encoder, 'group_embedding'):
            for param in model.encoder.group_embedding.parameters():
                param.requires_grad = False


def collate_fn(batch):
    """Collate batch with optional categorical indices."""
    enc_x, dec_x, dec_y, group_ids = zip(*[(b[0], b[1], b[2], b[3]) for b in batch])
    
    enc_x = torch.stack(enc_x)
    dec_x = torch.stack(dec_x)
    dec_y = torch.stack(dec_y)
    group_ids = torch.stack(group_ids)
    
    # Check if batch has categorical indices
    if len(batch[0]) > 4 and batch[0][4] is not None:
        cat_indices_list = [b[4] for b in batch]
        # Stack each categorical feature
        cat_indices = {}
        for key in cat_indices_list[0].keys():
            cat_indices[key] = torch.stack([ci[key] for ci in cat_indices_list])
        return enc_x, dec_x, dec_y, group_ids, cat_indices
    else:
        return enc_x, dec_x, dec_y, group_ids, None


def train_epoch(model, dataloader, optimizer, criterion, device, teacher_forcing_ratio, grad_clip):
    model.train()
    total_loss = 0.0
    
    for batch in dataloader:
        if len(batch) == 5:
            enc_x, dec_x, dec_y, group_ids, cat_indices = batch
        else:
            enc_x, dec_x, dec_y, group_ids = batch
            cat_indices = None
        
        enc_x = enc_x.to(device)
        dec_x = dec_x.to(device)
        dec_y = dec_y.to(device)
        group_ids = group_ids.to(device)
        
        if cat_indices is not None:
            cat_indices = {k: v.to(device) for k, v in cat_indices.items()}
        
        optimizer.zero_grad()
        
        predictions = model(
            enc_x,
            dec_x,
            dec_y,
            group_ids=group_ids,
            categorical_indices=cat_indices,
            teacher_forcing_ratio=teacher_forcing_ratio,
        )
        
        loss = criterion(predictions, dec_y)
        loss.backward()
        
        if grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        
        optimizer.step()
        total_loss += loss.item()
    
    return total_loss / len(dataloader)


def validate(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0.0
    
    with torch.no_grad():
        for batch in dataloader:
            if len(batch) == 5:
                enc_x, dec_x, dec_y, group_ids, cat_indices = batch
            else:
                enc_x, dec_x, dec_y, group_ids = batch
                cat_indices = None
            
            enc_x = enc_x.to(device)
            dec_x = dec_x.to(device)
            dec_y = dec_y.to(device)
            group_ids = group_ids.to(device)
            
            if cat_indices is not None:
                cat_indices = {k: v.to(device) for k, v in cat_indices.items()}
            
            predictions = model(
                enc_x,
                dec_x,
                dec_y,
                group_ids=group_ids,
                categorical_indices=cat_indices,
                teacher_forcing_ratio=0.0,
            )
            
            loss = criterion(predictions, dec_y)
            total_loss += loss.item()
    
    return total_loss / len(dataloader)


def main():
    args = parse_args()
    set_seed(args.seed)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[info] Using device: {device}")
    
    # Load data
    print(f"[load] Loading monthly data from {args.csv}...")
    df = pd.read_csv(args.csv, sep=';', parse_dates=['measured_at'])
    
    print(f"[info] Data shape: {df.shape}")
    print(f"[info] Groups: {df['group_id'].nunique()}")
    print(f"[info] Date range: {df['measured_at'].min()} to {df['measured_at'].max()}")
    
    # Create dataset - note: categorical_cols can be None for auto-detection from group metadata
    dataset = HourlySeq2SeqDatasetWithEmbeddings(
        df=df,
        window_size=args.window_size,
        horizon=args.horizon,
        group_col='group_id',
        consumption_col='scaled_consumption',
        price_col='scaled_price',
        min_samples_per_group=1,
        categorical_cols=None,  # Will auto-detect from group metadata if available
    )
    
    print(f"[info] Dataset size: {len(dataset)} samples")
    
    # Split into train/val (80/20)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size], generator=torch.Generator().manual_seed(args.seed)
    )
    
    print(f"[info] Train: {len(train_dataset)}, Val: {len(val_dataset)}")
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
    )
    
    # Get model dimensions
    sample = dataset[0]
    enc_input_size = sample[0].shape[1]
    dec_covariate_size = sample[1].shape[1]
    num_groups = df['group_id'].nunique()
    
    print(f"[info] Encoder input size: {enc_input_size}")
    print(f"[info] Decoder covariate size: {dec_covariate_size}")
    print(f"[info] Number of groups: {num_groups}")
    
    # Create model
    # Note: categorical_encoders from dataset maps feature_name -> {category: index}
    # Convert to categorical_config format: Dict[str, Tuple[int, int]] = (num_categories, embedding_dim)
    categorical_config = {}
    if hasattr(dataset, 'categorical_encoders') and dataset.categorical_encoders:
        for feat_name, encoder in dataset.categorical_encoders.items():
            num_categories = len(encoder)
            embedding_dim = min(8, (num_categories + 1) // 2)  # Heuristic: ~half of categories, max 8
            categorical_config[feat_name] = (num_categories, embedding_dim)
        print(f"[info] Categorical features: {categorical_config}")
    
    model = Seq2SeqWithEmbeddings(
        enc_input_size=enc_input_size,
        dec_known_size=dec_covariate_size,
        hidden_size=args.hidden_size,
        num_layers=args.layers,
        dropout=args.dropout,
        use_group_embedding=True,
        num_groups=num_groups,
        group_emb_dim=args.group_emb_dim,
        encoder_bidirectional=False,
        categorical_config=categorical_config if categorical_config else None,
    ).to(device)
    
    # Load pretrained weights if specified
    if args.pretrained_model:
        load_pretrained_weights(model, args.pretrained_model, args.freeze_embeddings)
    
    print(f"[info] Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Optimizer and scheduler
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10, verbose=True)
    criterion = nn.MSELoss()
    
    # Training loop
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    best_val_loss = float('inf')
    patience_counter = 0
    
    print(f"\n[train] Starting training for {args.epochs} epochs...")
    
    for epoch in range(1, args.epochs + 1):
        train_loss = train_epoch(
            model, train_loader, optimizer, criterion, device,
            args.teacher_forcing, args.grad_clip
        )
        
        val_loss = validate(model, val_loader, criterion, device)
        
        scheduler.step(val_loss)
        
        print(f"Epoch {epoch:3d}/{args.epochs} | Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f}")
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                'config': {
                    'model_kwargs': {
                        'enc_input_size': enc_input_size,
                        'dec_known_size': dec_covariate_size,
                        'hidden_size': args.hidden_size,
                        'num_layers': args.layers,
                        'dropout': args.dropout,
                        'use_group_embedding': True,
                        'num_groups': num_groups,
                        'group_emb_dim': args.group_emb_dim,
                        'encoder_bidirectional': False,
                        'categorical_config': categorical_config if categorical_config else None,
                    },
                    'group_to_idx': dataset.group_to_idx,
                    'categorical_encoders': dataset.categorical_encoders if hasattr(dataset, 'categorical_encoders') else {},
                },
            }
            
            save_path = save_dir / "seq2seq_monthly_embeddings.pt"
            torch.save(checkpoint, save_path)
            print(f"[save] Best model saved to {save_path}")
        else:
            patience_counter += 1
        
        # Early stopping
        if patience_counter >= args.early_stopping_patience:
            print(f"\n[early stop] No improvement for {args.early_stopping_patience} epochs")
            break
    
    print(f"\n[done] Training complete!")
    print(f"[info] Best validation loss: {best_val_loss:.6f}")


if __name__ == "__main__":
    main()

