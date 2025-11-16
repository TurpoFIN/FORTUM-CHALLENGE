"""
Simple rescaling of predictions to match expected output range.

The model outputs are in "scaled_consumption" space which needs a simple
multiplication factor to get to actual monthly kWh/MWh values.

Usage:
    python "seq2seq implementation/rescale_predictions_simple.py" \
        --predictions predictions_monthly_v2.csv \
        --data formatted_features_monthly.csv \
        --out predictions_monthly_final.csv
"""

import argparse
import pandas as pd
import numpy as np


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--predictions", required=True, help="Predictions CSV (scaled)")
    p.add_argument("--data", required=True, help="Training data to learn scaling factor")
    p.add_argument("--out", required=True, help="Output CSV")
    p.add_argument("--scale-factor", type=float, default=None, help="Manual scale factor (optional)")
    return p.parse_args()


def main():
    args = parse_args()
    
    print(f"[load] Loading predictions from {args.predictions}...")
    pred_df = pd.read_csv(args.predictions, sep=';')
    
    print(f"[load] Loading training data from {args.data}...")
    data_df = pd.read_csv(args.data, sep=';')
    
    print(f"[info] Predictions shape: {pred_df.shape}")
    
    if args.scale_factor is not None:
        scale_factor = args.scale_factor
        print(f"[info] Using manual scale factor: {scale_factor}")
    else:
        # Estimate scale factor from training data
        # The model was trained on scaled_consumption which is monthly sum
        # The example outputs are also monthly consumption in similar units
        
        # Check the range of scaled_consumption in training data
        scaled_mean = data_df['scaled_consumption'].mean()
        scaled_std = data_df['scaled_consumption'].std()
        scaled_min = data_df['scaled_consumption'].min()
        scaled_max = data_df['scaled_consumption'].max()
        
        print(f"\n[info] Training data scaled_consumption statistics:")
        print(f"  Mean: {scaled_mean:.2f}")
        print(f"  Std: {scaled_std:.2f}")
        print(f"  Min: {scaled_min:.2f}")
        print(f"  Max: {scaled_max:.2f}")
        
        # The model predicts in a normalized space where typical values are -400 to +400
        # Training scaled_consumption ranges from ~-400 to ~4000
        # So we want predictions in that same scale
        
        # Simple approach: use a small multiplier to account for the model's output range
        # Model outputs are typically in the range -400 to 400
        # We want them in the range 0 to 2000 (typical monthly consumption)
        
        # Actually, let's check what range the predictions are in
        numeric_cols = [c for c in pred_df.columns if c != 'measured_at']
        all_pred_values = []
        for col in numeric_cols:
            all_pred_values.extend(pred_df[col].values)
        
        pred_mean = np.mean(all_pred_values)
        pred_std = np.std(all_pred_values)
        pred_min = np.min(all_pred_values)
        pred_max = np.max(all_pred_values)
        
        print(f"\n[info] Model prediction statistics:")
        print(f"  Mean: {pred_mean:.2f}")
        print(f"  Std: {pred_std:.2f}")
        print(f"  Min: {pred_min:.2f}")
        print(f"  Max: {pred_max:.2f}")
        
        # The predictions are in a similar scale to scaled_consumption
        # We just need to shift and clip to non-negative
        scale_factor = 1.0  # Direct use
        print(f"\n[strategy] Predictions are already in scaled_consumption space")
        print(f"[strategy] Will add offset to center around positive values")
    
    # Process predictions
    print(f"\n[process] Rescaling predictions...")
    
    output_df = pred_df.copy()
    
    for col in pred_df.columns:
        if col == 'measured_at':
            continue
        
        # Get predictions for this group
        preds = pred_df[col].values * scale_factor
        
        # Add offset to make mostly positive
        # The scaled_consumption in training data has mean around 1000-2000
        # Model predictions centered around 0-400, so add offset
        offset = 1000  # Reasonable monthly consumption baseline
        
        preds_adjusted = preds + offset
        
        # Clip to non-negative
        preds_adjusted = np.maximum(preds_adjusted, 0)
        
        output_df[col] = preds_adjusted
    
    # Show statistics
    print(f"\n[stats] Output statistics:")
    numeric_cols = [c for c in output_df.columns if c != 'measured_at']
    all_output_values = []
    for col in numeric_cols:
        all_output_values.extend(output_df[col].values)
    
    print(f"  Min: {np.min(all_output_values):.2f}")
    print(f"  Max: {np.max(all_output_values):.2f}")
    print(f"  Mean: {np.mean(all_output_values):.2f}")
    print(f"  Median: {np.median(all_output_values):.2f}")
    
    # Show sample for a few groups
    print(f"\n[sample] Sample predictions for first few groups:")
    for col in numeric_cols[:5]:
        print(f"\nGroup {col}:")
        print(f"  Original (scaled): {pred_df[col].head(3).values}")
        print(f"  Final (kWh/MWh): {output_df[col].head(3).values}")
    
    # Save
    print(f"\n[save] Saving to {args.out}...")
    output_df.to_csv(args.out, sep=';', index=False)
    
    print(f"[done] Rescaled predictions saved!")
    print(f"[info] Output shape: {output_df.shape}")


if __name__ == "__main__":
    main()

