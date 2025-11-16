"""
Simple monthly inference that matches training feature extraction exactly.

Usage:
    python "seq2seq implementation/infer_monthly_simple.py" \
        --model artifacts/models_monthly_a100/seq2seq_monthly_embeddings.pt \
        --csv formatted_features_monthly.csv \
        --out predictions_monthly_junction.csv
"""

import argparse
from pathlib import Path
import pandas as pd
import torch

from models.seq2seq_lstm_with_embeddings import Seq2SeqWithEmbeddings


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", type=str, required=True)
    p.add_argument("--csv", type=str, required=True)
    p.add_argument("--out", type=str, required=True)
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
    
    print(f"[info] Model expects:")
    print(f"  Encoder input: {config['model_kwargs']['enc_input_size']} features")
    print(f"  Decoder input: {config['model_kwargs']['dec_known_size']} features")
    print(f"  Groups: {config['model_kwargs']['num_groups']}")
    
    # Load data
    print(f"[load] Loading data from {args.csv}...")
    df = pd.read_csv(args.csv, sep=';', parse_dates=['measured_at'])
    df = df.sort_values(['group_id', 'measured_at']).reset_index(drop=True)
    
    print(f"[info] CSV has {len(df.columns)} columns: {list(df.columns)}")
    
    # Get window size from model or use default
    window_size = 24
    horizon = 12
    
    group_to_idx = config['group_to_idx']
    
    # Identify numeric feature columns (exclude metadata)
    exclude_cols = ['measured_at', 'group_id', 'year_month']
    numeric_cols = [c for c in df.columns if c not in exclude_cols and pd.api.types.is_numeric_dtype(df[c])]
    
    print(f"[info] Using {len(numeric_cols)} numeric features: {numeric_cols[:5]}...")
    
    # Extract feature dimensions from first group
    sample_group = df[df['group_id'] == sorted(df['group_id'].unique())[0]]
    if len(sample_group) >= window_size:
        sample_features = sample_group[numeric_cols].iloc[:window_size].values
        print(f"[info] Sample features shape: {sample_features.shape}")
    
    predictions_by_group = {}
    
    print(f"\n[generate] Generating predictions for {len(group_to_idx)} groups...")
    
    for group_id in sorted(group_to_idx.keys()):
        gdf = df[df['group_id'] == group_id].copy()
        
        if len(gdf) < window_size:
            print(f"[skip] Group {group_id}: only {len(gdf)} months (need {window_size})")
            continue
        
        # Use last window_size months
        window_df = gdf.iloc[-window_size:]
        
        # Extract numeric features only
        enc_features = window_df[numeric_cols].values  # [window_size, num_features]
        enc_x = torch.tensor(enc_features, dtype=torch.float32).unsqueeze(0).to(device)  # [1, window_size, num_features]
        
        # Check dimensions
        if enc_x.shape[2] != config['model_kwargs']['enc_input_size']:
            print(f"[error] Group {group_id}: Feature count mismatch!")
            print(f"  Expected: {config['model_kwargs']['enc_input_size']}")
            print(f"  Got: {enc_x.shape[2]}")
            print(f"  Features: {numeric_cols}")
            # Try to fix by padding or trimming
            if enc_x.shape[2] > config['model_kwargs']['enc_input_size']:
                enc_x = enc_x[:, :, :config['model_kwargs']['enc_input_size']]
                print(f"  [fix] Trimmed to {enc_x.shape[2]} features")
            else:
                print(f"  [skip] Cannot fix - skipping group")
                continue
        
        # Decoder known features (zeros for simplicity)
        dec_known_size = config['model_kwargs']['dec_known_size']
        dec_known = torch.zeros(1, horizon, dec_known_size, dtype=torch.float32).to(device)
        
        # Dummy target
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
            )
        
        # Store predictions
        preds_np = preds.squeeze().cpu().numpy()  # [horizon]
        predictions_by_group[group_id] = preds_np.tolist()
        
        if len(predictions_by_group) % 20 == 0:
            print(f"  Processed {len(predictions_by_group)} groups...")
    
    print(f"\n[save] Generated predictions for {len(predictions_by_group)} groups")
    
    # Create wide-format DataFrame
    last_date = df['measured_at'].max()
    forecast_start = last_date + pd.DateOffset(months=1)
    forecast_dates = pd.date_range(start=forecast_start, periods=horizon, freq='MS')
    
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


if __name__ == "__main__":
    main()

