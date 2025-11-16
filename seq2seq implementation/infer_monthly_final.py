"""
Final working monthly inference - filters out categorical text columns properly.
"""

import argparse
from pathlib import Path
import pandas as pd
import torch
import numpy as np
from models.seq2seq_lstm_with_embeddings import Seq2SeqWithEmbeddings


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--csv", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load model
    checkpoint = torch.load(args.model, map_location=device, weights_only=False)
    model = Seq2SeqWithEmbeddings(**checkpoint['config']['model_kwargs'])
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device).eval()
    
    enc_size = checkpoint['config']['model_kwargs']['enc_input_size']
    dec_size = checkpoint['config']['model_kwargs']['dec_known_size']
    
    print(f"Model expects: enc={enc_size}, dec={dec_size}")
    
    # Load data
    df = pd.read_csv(args.csv, sep=';', parse_dates=['measured_at'])
    df = df.sort_values(['group_id', 'measured_at'])
    
    # Filter to ONLY truly numeric columns (exclude text categories)
    numeric_cols = []
    for col in df.columns:
        if col in ['measured_at', 'group_id']:
            continue
        # Check if actually numeric values (not text)
        if pd.api.types.is_numeric_dtype(df[col]):
            # Also check if values are actually numbers, not strings
            sample = df[col].dropna().iloc[0] if len(df[col].dropna()) > 0 else None
            if sample is not None and isinstance(sample, (int, float, np.number)):
                numeric_cols.append(col)
    
    print(f"CSV has {len(numeric_cols)} truly numeric columns")
    print(f"Columns: {numeric_cols}")
    
    # Use first N columns for encoder and decoder
    enc_cols = numeric_cols[:enc_size]
    # Decoder: remove consumption column
    dec_cols = [c for c in numeric_cols if 'consumption' not in c.lower()][:dec_size]
    
    print(f"Encoder cols ({len(enc_cols)}): {enc_cols}")
    print(f"Decoder cols ({len(dec_cols)}): {dec_cols}")
    
    window_size = 24
    horizon = 12
    group_to_idx = checkpoint['config']['group_to_idx']
    
    all_predictions = {}
    
    for group_id in sorted(group_to_idx.keys()):
        gdf = df[df['group_id'] == group_id]
        if len(gdf) < window_size:
            print(f"Skip group {group_id}: only {len(gdf)} months")
            continue
        
        # Last window
        window = gdf.iloc[-window_size:]
        
        # Encoder input
        enc_data = window[enc_cols].values.astype(np.float32)
        enc_x = torch.from_numpy(enc_data).unsqueeze(0).to(device)
        
        # Decoder input (repeat last known values)
        last_known = window.iloc[-1][dec_cols].values.astype(np.float32)
        dec_data = np.tile(last_known, (horizon, 1)).astype(np.float32)
        dec_known = torch.from_numpy(dec_data).unsqueeze(0).to(device)
        
        # Dummy target
        dummy_y = torch.zeros(1, horizon, 1, dtype=torch.float32, device=device)
        
        # Group ID
        group_idx = torch.tensor([group_to_idx[group_id]], dtype=torch.long, device=device)
        
        # Predict
        with torch.no_grad():
            preds = model(enc_x, dec_known, dummy_y, group_ids=group_idx, 
                         categorical_indices=None, teacher_forcing_ratio=0.0)
        
        # Extract predictions
        preds_np = preds.squeeze(0).squeeze(-1).cpu().numpy()  # [horizon]
        
        # Check for NaN/Inf
        if np.any(np.isnan(preds_np)) or np.any(np.isinf(preds_np)):
            print(f"WARNING: Group {group_id} has NaN/Inf predictions")
            preds_np = np.nan_to_num(preds_np, nan=0.0, posinf=0.0, neginf=0.0)
        
        all_predictions[group_id] = preds_np.tolist()
        
        if len(all_predictions) % 20 == 0:
            print(f"Processed {len(all_predictions)} groups")
    
    print(f"\nGenerated {len(all_predictions)} predictions")
    
    # Verify predictions are not empty
    sample_pred = list(all_predictions.values())[0] if all_predictions else []
    print(f"Sample prediction (first group): {sample_pred}")
    
    # Create output using concat (faster)
    last_date = df['measured_at'].max()
    dates = pd.date_range(start=last_date + pd.DateOffset(months=1), periods=horizon, freq='MS')
    
    output_data = {'measured_at': dates.strftime('%Y-%m-%dT%H:%M:%S.000Z')}
    
    # Add all group predictions
    for gid in sorted(all_predictions.keys()):
        output_data[gid] = all_predictions[gid]
    
    output = pd.DataFrame(output_data)
    
    print(f"Output shape: {output.shape}")
    print(f"Sample row:\n{output.iloc[0]}")
    
    output.to_csv(args.out, sep=';', index=False)
    print(f"\nSaved to {args.out}")


if __name__ == "__main__":
    main()

