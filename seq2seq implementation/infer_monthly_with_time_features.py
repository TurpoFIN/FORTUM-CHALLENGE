"""
Fixed monthly inference that provides REAL future time features to the decoder.

The decoder needs to know WHICH months it's predicting for (Oct 2024 - Sep 2025)
so it can use seasonal patterns.

Usage:
    python "seq2seq implementation/infer_monthly_with_time_features.py" \
        --model artifacts/models_monthly_a100/seq2seq_monthly_embeddings.pt \
        --csv formatted_features_monthly.csv \
        --out predictions_monthly_with_time.csv
"""

import argparse
from pathlib import Path
import pandas as pd
import torch
import numpy as np

from models.seq2seq_lstm_with_embeddings import Seq2SeqWithEmbeddings


# Features that the model was trained on (in order)
TRAINED_FEATURES = [
    'scaled_consumption',
    'scaled_price',
    'hour_sin',
    'hour_cos',
    'day_of_week_sin',
    'day_of_week_cos',
    'month_sin',
    'month_cos',
    'is_holiday',
    'is_weekend',
]

# Decoder features (exclude consumption, add time/events)
DECODER_FEATURES = [
    'scaled_price',  # Unknown future prices - will use last known
    'hour_sin',
    'hour_cos',
    'day_of_week_sin',
    'day_of_week_cos',
    'month_sin',
    'month_cos',
    'is_holiday',
    'is_weekend',
]


def create_future_time_features(start_date='2024-10-01', n_months=12):
    """Create time features for future months."""
    dates = pd.date_range(start=start_date, periods=n_months, freq='MS')
    
    features = []
    for date in dates:
        # Hour features (use midnight = 0)
        hour_sin = np.sin(2 * np.pi * 0 / 24)
        hour_cos = np.cos(2 * np.pi * 0 / 24)
        
        # Day of week features (first of month)
        day_of_week = date.dayofweek
        dow_sin = np.sin(2 * np.pi * day_of_week / 7)
        dow_cos = np.cos(2 * np.pi * day_of_week / 7)
        
        # Month features
        month = date.month - 1  # 0-11
        month_sin = np.sin(2 * np.pi * month / 12)
        month_cos = np.cos(2 * np.pi * month / 12)
        
        # Events (simplified - no future holiday info)
        is_holiday = 0
        is_weekend = 1 if day_of_week >= 5 else 0
        
        features.append([
            hour_sin, hour_cos,
            dow_sin, dow_cos,
            month_sin, month_cos,
            is_holiday, is_weekend
        ])
    
    return np.array(features)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", type=str, required=True, help="Path to trained model")
    p.add_argument("--csv", type=str, required=True, help="Path to monthly features CSV")
    p.add_argument("--out", type=str, required=True, help="Output CSV path")
    p.add_argument("--window-size", type=int, default=24, help="Window size (months)")
    p.add_argument("--horizon", type=int, default=12, help="Forecast horizon (months)")
    return p.parse_args()


def main():
    args = parse_args()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[info] Using device: {device}")
    
    # Load model
    print(f"[load] Loading model from {args.model}...")
    checkpoint = torch.load(args.model, map_location=device)
    config = checkpoint['config']
    
    model = Seq2SeqWithEmbeddings(**config['model_kwargs'])
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    
    enc_input_size = config['model_kwargs']['enc_input_size']
    dec_known_size = config['model_kwargs']['dec_known_size']
    num_groups = config['model_kwargs']['num_groups']
    
    print(f"[info] Model configuration:")
    print(f"  Encoder input size: {enc_input_size}")
    print(f"  Decoder known size: {dec_known_size}")
    print(f"  Number of groups: {num_groups}")
    
    # Load data
    print(f"[load] Loading data from {args.csv}...")
    df = pd.read_csv(args.csv, sep=';', parse_dates=['measured_at'])
    df = df.sort_values(['group_id', 'measured_at']).reset_index(drop=True)
    
    print(f"[info] CSV shape: {df.shape}")
    print(f"[info] Date range: {df['measured_at'].min()} to {df['measured_at'].max()}")
    
    # Create future time features
    print(f"\n[time] Creating future time features for 12 months...")
    future_time_features = create_future_time_features('2024-10-01', args.horizon)
    print(f"[time] Future time features shape: {future_time_features.shape}")
    print(f"[time] Sample (Oct 2024): {future_time_features[0]}")
    
    # Get group mapping
    group_to_idx = config['group_to_idx']
    
    # Generate predictions
    predictions_by_group = {}
    skipped_groups = []
    
    print(f"\n[generate] Generating predictions for {len(group_to_idx)} groups...")
    
    for group_id in sorted(group_to_idx.keys()):
        gdf = df[df['group_id'] == group_id].copy()
        
        if len(gdf) < args.window_size:
            skipped_groups.append((group_id, len(gdf)))
            continue
        
        # Use last window_size months as encoder input
        window_df = gdf.iloc[-args.window_size:]
        
        # Extract encoder features
        enc_features = window_df[TRAINED_FEATURES].values
        
        if pd.isna(enc_features).any():
            skipped_groups.append((group_id, 'NaN'))
            continue
        
        enc_x = torch.tensor(enc_features, dtype=torch.float32).unsqueeze(0).to(device)
        
        # Create decoder known features
        # Use last known price + future time features
        last_price = window_df['scaled_price'].iloc[-1]
        
        # Build decoder features: [price, time_features...]
        dec_features = np.zeros((args.horizon, dec_known_size))
        
        # Fill in prices (use last known price for all future months)
        dec_features[:, 0] = last_price
        
        # Fill in time features (columns 1-8)
        dec_features[:, 1:9] = future_time_features
        
        dec_known = torch.tensor(dec_features, dtype=torch.float32).unsqueeze(0).to(device)
        
        # Dummy target
        dummy_y = torch.zeros(1, args.horizon, 1, dtype=torch.float32).to(device)
        
        # Group index
        group_idx = torch.tensor([group_to_idx[group_id]], dtype=torch.long).to(device)
        
        # Generate predictions
        with torch.no_grad():
            preds = model(
                enc_x,
                dec_known,
                dummy_y,
                group_ids=group_idx,
                categorical_indices=None,
                teacher_forcing_ratio=0.0,
            )
        
        # Store predictions
        preds_np = preds.squeeze().cpu().numpy()
        predictions_by_group[group_id] = preds_np.tolist()
        
        if len(predictions_by_group) % 20 == 0:
            print(f"  Processed {len(predictions_by_group)} groups...")
    
    print(f"\n[save] Generated predictions for {len(predictions_by_group)} groups")
    
    if skipped_groups:
        print(f"[warn] Skipped {len(skipped_groups)} groups (insufficient data)")
    
    # Create wide-format DataFrame
    forecast_dates = pd.date_range(start='2024-10-01', periods=args.horizon, freq='MS')
    
    wide_data = {'measured_at': forecast_dates}
    for group_id in sorted(predictions_by_group.keys()):
        wide_data[group_id] = predictions_by_group[group_id]
    
    wide_df = pd.DataFrame(wide_data)
    wide_df['measured_at'] = wide_df['measured_at'].dt.strftime('%Y-%m-%dT%H:%M:%S.000Z')
    
    # Save
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    wide_df.to_csv(out_path, sep=';', index=False)
    
    print(f"[done] Predictions saved to {out_path}")
    print(f"[info] Format: {len(wide_df)} months × {len(predictions_by_group)} groups")
    print(f"[info] Date range: {wide_df['measured_at'].iloc[0]} to {wide_df['measured_at'].iloc[-1]}")
    
    # Show sample with variation check
    if len(predictions_by_group) > 0:
        for gid in [28, 29, 30]:
            if gid in predictions_by_group:
                preds = predictions_by_group[gid]
                print(f"\n[sample] Group {gid}:")
                print(f"  Oct-Dec: {preds[:3]}")
                print(f"  Range: {min(preds):.1f} to {max(preds):.1f}")
                print(f"  Std Dev: {np.std(preds):.1f}")


if __name__ == "__main__":
    main()

