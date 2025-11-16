"""
Generate 12-month predictions in Junction challenge format (wide CSV).

Usage:
    python "seq2seq implementation/infer_monthly_junction_format.py" \
        --model artifacts/models_monthly/seq2seq_monthly_embeddings.pt \
        --csv formatted_features_monthly.csv \
        --out predictions_monthly_junction.csv \
        --window-size 24 \
        --horizon 12
"""

import argparse
from pathlib import Path
from typing import Dict

import pandas as pd
import torch

try:
    from tqdm import tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False
    print("[info] tqdm not installed, progress bar disabled")

from models.seq2seq_lstm_with_embeddings import Seq2SeqWithEmbeddings


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate 12-month predictions in Junction format.")
    p.add_argument("--model", type=str, required=True, help="Path to trained model (.pt file)")
    p.add_argument("--csv", type=str, required=True, help="Path to monthly features CSV")
    p.add_argument("--out", type=str, required=True, help="Output CSV path (wide format)")
    p.add_argument("--window-size", type=int, default=24, help="Encoder window size (months)")
    p.add_argument("--horizon", type=int, default=12, help="Forecast horizon (months)")
    return p.parse_args()


def load_model_and_config(model_path: Path, device: torch.device):
    """Load model checkpoint and configuration."""
    checkpoint = torch.load(model_path, map_location=device)
    
    config = checkpoint['config']
    model = Seq2SeqWithEmbeddings(**config['model_kwargs'])
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    
    return model, config


def main() -> None:
    args = parse_args()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[info] Using device: {device}")
    
    # Load model
    model_path = Path(args.model)
    print(f"[load] Loading model from {model_path}...")
    model, config = load_model_and_config(model_path, device)
    
    # Load data
    csv_path = Path(args.csv)
    print(f"[load] Loading data from {csv_path}...")
    df = pd.read_csv(csv_path, sep=';', parse_dates=['measured_at'])
    df = df.sort_values(['group_id', 'measured_at']).reset_index(drop=True)
    
    window_size = args.window_size
    horizon = args.horizon
    
    # Get feature columns
    feature_cols = [c for c in df.columns if c not in ['measured_at', 'group_id']]
    cons_col = 'scaled_consumption'
    
    if cons_col not in df.columns:
        raise ValueError(f"Column '{cons_col}' not found in CSV")
    
    # Get group to index mapping
    group_to_idx = config.get('group_to_idx', {})
    if not group_to_idx:
        # Build from data
        unique_groups = sorted(df['group_id'].unique())
        group_to_idx = {g: i for i, g in enumerate(unique_groups)}
        print(f"[info] Built group_to_idx from data: {len(group_to_idx)} groups")
    
    # Storage for predictions (group_id -> list of 12 values)
    predictions_by_group = {}
    
    # Generate predictions for each group
    groups = sorted(df['group_id'].unique())
    iterator = tqdm(groups, desc="Generating predictions") if HAS_TQDM else groups
    
    for group_id in iterator:
        if group_id not in group_to_idx:
            print(f"[warn] Group {group_id} not in training data, skipping")
            continue
        
        gdf = df[df['group_id'] == group_id].copy()
        
        if len(gdf) < window_size:
            print(f"[warn] Group {group_id} has only {len(gdf)} months, need {window_size}, skipping")
            continue
        
        # Use last window_size months as input
        window_df = gdf.iloc[-window_size:].copy()
        
        # Prepare encoder input [1, window_size, enc_input_size]
        # Match the exact feature set from training
        enc_feature_cols = [c for c in feature_cols if c in window_df.columns]
        enc_features = window_df[enc_feature_cols].values
        enc_x = torch.tensor(enc_features, dtype=torch.float32).unsqueeze(0).to(device)
        
        # Debug: Check feature dimensions
        if enc_x.shape[2] != config['model_kwargs']['enc_input_size']:
            print(f"[warn] Group {group_id}: Feature mismatch!")
            print(f"  Expected: {config['model_kwargs']['enc_input_size']} features")
            print(f"  Got: {enc_x.shape[2]} features from {len(enc_feature_cols)} columns")
            print(f"  Columns: {enc_feature_cols}")
            continue
        
        # For monthly predictions, decoder known features
        dec_covariate_size = config['model_kwargs']['dec_known_size']
        
        # Build decoder known features (simplified for monthly - use zeros or copy time features)
        dec_known = torch.zeros(1, horizon, dec_covariate_size, dtype=torch.float32).to(device)
        
        # Dummy target (not used in inference)
        dummy_y = torch.zeros(1, horizon, 1, dtype=torch.float32).to(device)
        
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
            )  # [1, horizon, 1]
        
        # Extract predictions
        preds_np = preds.squeeze().cpu().numpy()  # [horizon]
        
        # Store predictions for this group
        predictions_by_group[group_id] = preds_np.tolist()
    
    print(f"\n[save] Generated predictions for {len(predictions_by_group)} groups")
    
    # Create wide-format DataFrame
    # Determine forecast start date (first month after last data point)
    last_date = df['measured_at'].max()
    forecast_start = last_date + pd.DateOffset(months=1)
    
    # Generate 12 monthly timestamps
    forecast_dates = pd.date_range(start=forecast_start, periods=horizon, freq='MS')
    
    # Create wide format: rows = months, columns = groups
    wide_data = {'measured_at': forecast_dates}
    
    for group_id in sorted(predictions_by_group.keys()):
        wide_data[group_id] = predictions_by_group[group_id]
    
    wide_df = pd.DataFrame(wide_data)
    
    # Format measured_at as ISO with Z suffix (Junction format)
    wide_df['measured_at'] = wide_df['measured_at'].dt.strftime('%Y-%m-%dT%H:%M:%S.000Z')
    
    # Save
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    wide_df.to_csv(out_path, sep=';', index=False)
    
    print(f"[done] Predictions saved to {out_path}")
    print(f"[info] Format: {len(wide_df)} months × {len(predictions_by_group)} groups")
    print(f"[info] Date range: {forecast_dates[0].strftime('%Y-%m-%d')} to {forecast_dates[-1].strftime('%Y-%m-%d')}")


if __name__ == "__main__":
    main()

