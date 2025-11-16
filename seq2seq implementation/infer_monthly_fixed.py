"""
Fixed monthly inference that uses ONLY the features the model was trained on.

The model expects exactly these 10 features:
1. scaled_consumption
2. scaled_price  
3. hour_sin, hour_cos
4. day_of_week_sin, day_of_week_cos
5. month_sin, month_cos
6. is_holiday, is_weekend

Usage:
    python "seq2seq implementation/infer_monthly_fixed.py" \
        --model artifacts/models_monthly_a100/seq2seq_monthly_embeddings.pt \
        --csv formatted_features_monthly.csv \
        --out predictions_monthly_junction.csv
"""

import argparse
from pathlib import Path
import pandas as pd
import torch

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
    
    # Verify feature count
    if enc_input_size != len(TRAINED_FEATURES):
        print(f"[warn] Model expects {enc_input_size} features but TRAINED_FEATURES has {len(TRAINED_FEATURES)}")
        print(f"[warn] Will use first {enc_input_size} features from TRAINED_FEATURES")
        features_to_use = TRAINED_FEATURES[:enc_input_size]
    else:
        features_to_use = TRAINED_FEATURES
    
    print(f"[info] Using features: {features_to_use}")
    
    # Load data
    print(f"[load] Loading data from {args.csv}...")
    df = pd.read_csv(args.csv, sep=';', parse_dates=['measured_at'])
    df = df.sort_values(['group_id', 'measured_at']).reset_index(drop=True)
    
    print(f"[info] CSV shape: {df.shape}")
    print(f"[info] Groups in CSV: {df['group_id'].nunique()}")
    print(f"[info] Date range: {df['measured_at'].min()} to {df['measured_at'].max()}")
    
    # Check for missing features
    missing_features = [f for f in features_to_use if f not in df.columns]
    if missing_features:
        print(f"[error] Missing required features in CSV: {missing_features}")
        print(f"[error] Available columns: {list(df.columns)}")
        return
    
    # Get group mapping
    group_to_idx = config['group_to_idx']
    print(f"[info] Model knows {len(group_to_idx)} groups")
    
    # Generate predictions
    predictions_by_group = {}
    skipped_groups = []
    
    print(f"\n[generate] Generating predictions...")
    
    for group_id in sorted(group_to_idx.keys()):
        gdf = df[df['group_id'] == group_id].copy()
        
        if len(gdf) < args.window_size:
            skipped_groups.append((group_id, len(gdf)))
            continue
        
        # Use last window_size months as encoder input
        window_df = gdf.iloc[-args.window_size:]
        
        # Extract only the features the model expects
        enc_features = window_df[features_to_use].values  # [window_size, num_features]
        
        # Check for NaN values
        if pd.isna(enc_features).any():
            print(f"[warn] Group {group_id}: Found NaN values in features, skipping")
            skipped_groups.append((group_id, 'NaN'))
            continue
        
        # Prepare tensors
        enc_x = torch.tensor(enc_features, dtype=torch.float32).unsqueeze(0).to(device)  # [1, window_size, num_features]
        
        # Decoder known features (zeros for future months - we don't know future prices/events)
        dec_known = torch.zeros(1, args.horizon, dec_known_size, dtype=torch.float32).to(device)
        
        # Dummy target (not used during inference)
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
        preds_np = preds.squeeze().cpu().numpy()  # [horizon]
        predictions_by_group[group_id] = preds_np.tolist()
        
        if len(predictions_by_group) % 20 == 0:
            print(f"  Processed {len(predictions_by_group)} groups...")
    
    print(f"\n[save] Generated predictions for {len(predictions_by_group)} groups")
    
    if skipped_groups:
        print(f"[warn] Skipped {len(skipped_groups)} groups:")
        for gid, reason in skipped_groups[:10]:
            print(f"  Group {gid}: {reason}")
        if len(skipped_groups) > 10:
            print(f"  ... and {len(skipped_groups) - 10} more")
    
    # Create wide-format DataFrame for Junction submission
    # Start from October 2024
    forecast_dates = pd.date_range(start='2024-10-01', periods=args.horizon, freq='MS')
    
    wide_data = {'measured_at': forecast_dates}
    for group_id in sorted(predictions_by_group.keys()):
        wide_data[group_id] = predictions_by_group[group_id]
    
    wide_df = pd.DataFrame(wide_data)
    
    # Format dates as required by Junction
    wide_df['measured_at'] = wide_df['measured_at'].dt.strftime('%Y-%m-%dT%H:%M:%S.000Z')
    
    # Save
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    wide_df.to_csv(out_path, sep=';', index=False)
    
    print(f"[done] Predictions saved to {out_path}")
    print(f"[info] Format: {len(wide_df)} months × {len(predictions_by_group)} groups")
    print(f"[info] Date range: {wide_df['measured_at'].iloc[0]} to {wide_df['measured_at'].iloc[-1]}")
    
    # Show sample predictions
    if len(predictions_by_group) > 0:
        sample_group = list(predictions_by_group.keys())[0]
        sample_preds = predictions_by_group[sample_group]
        print(f"\n[sample] Group {sample_group} predictions (first 3 months):")
        for i, val in enumerate(sample_preds[:3]):
            print(f"  Month {i+1}: {val:.2f}")


if __name__ == "__main__":
    main()
