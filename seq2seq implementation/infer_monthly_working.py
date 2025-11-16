"""
Working monthly inference - no BS.
"""

import argparse
from pathlib import Path
import pandas as pd
import torch
from models.seq2seq_lstm_with_embeddings import Seq2SeqWithEmbeddings


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--csv", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load model
    checkpoint = torch.load(args.model, map_location=device)
    model = Seq2SeqWithEmbeddings(**checkpoint['config']['model_kwargs'])
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device).eval()
    
    enc_size = checkpoint['config']['model_kwargs']['enc_input_size']
    dec_size = checkpoint['config']['model_kwargs']['dec_known_size']
    
    print(f"Model expects: enc={enc_size}, dec={dec_size}")
    
    # Load data
    df = pd.read_csv(args.csv, sep=';', parse_dates=['measured_at'])
    df = df.sort_values(['group_id', 'measured_at'])
    
    # Get numeric columns only
    numeric_cols = [c for c in df.columns if c not in ['measured_at', 'group_id'] and pd.api.types.is_numeric_dtype(df[c])]
    
    print(f"CSV has {len(numeric_cols)} numeric columns")
    print(f"Columns: {numeric_cols}")
    
    # Encoder uses: consumption + (enc_size-1) other features
    # Decoder uses: price + (dec_size-1) other features (no consumption)
    
    # Simple approach: use first N columns
    enc_cols = numeric_cols[:enc_size]
    dec_cols = [c for c in numeric_cols if c != 'scaled_consumption'][:dec_size]
    
    print(f"Using encoder cols: {enc_cols}")
    print(f"Using decoder cols: {dec_cols}")
    
    window_size = 24
    horizon = 12
    group_to_idx = checkpoint['config']['group_to_idx']
    
    predictions = {}
    
    for group_id in sorted(group_to_idx.keys()):
        gdf = df[df['group_id'] == group_id]
        if len(gdf) < window_size:
            continue
        
        # Last window
        window = gdf.iloc[-window_size:]
        
        # Encoder input
        enc_x = torch.tensor(window[enc_cols].values, dtype=torch.float32).unsqueeze(0).to(device)
        
        # Decoder input (repeat last known values)
        last_known = window.iloc[-1][dec_cols].values
        dec_known = torch.tensor([last_known] * horizon, dtype=torch.float32).unsqueeze(0).to(device)
        
        # Dummy target
        dummy_y = torch.zeros(1, horizon, 1, dtype=torch.float32).to(device)
        
        # Group ID
        group_idx = torch.tensor([group_to_idx[group_id]], dtype=torch.long).to(device)
        
        # Predict
        with torch.no_grad():
            preds = model(enc_x, dec_known, dummy_y, group_ids=group_idx, categorical_indices=None, teacher_forcing_ratio=0.0)
        
        predictions[group_id] = preds.squeeze().cpu().numpy().tolist()
        
        if len(predictions) % 20 == 0:
            print(f"Processed {len(predictions)} groups")
    
    print(f"\nGenerated {len(predictions)} predictions")
    
    # Create output
    last_date = df['measured_at'].max()
    dates = pd.date_range(start=last_date + pd.DateOffset(months=1), periods=horizon, freq='MS')
    
    output = pd.DataFrame({'measured_at': dates})
    for gid in sorted(predictions.keys()):
        output[gid] = predictions[gid]
    
    output['measured_at'] = output['measured_at'].dt.strftime('%Y-%m-%dT%H:%M:%S.000Z')
    output.to_csv(args.out, sep=';', index=False)
    
    print(f"Saved to {args.out}")
    print(f"Shape: {output.shape}")


if __name__ == "__main__":
    main()

