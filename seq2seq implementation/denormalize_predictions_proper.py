"""
Properly denormalize predictions by learning the actual scaling relationship from data.

The scaled_consumption column doesn't use standard z-score normalization.
We need to learn the actual relationship between scaled_consumption and consumption.

Usage:
    python "seq2seq implementation/denormalize_predictions_proper.py" \
        --predictions predictions_monthly_v2.csv \
        --data formatted_features_monthly.csv \
        --out predictions_monthly_junction_fixed.csv
"""

import argparse
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--predictions", required=True, help="Predictions CSV (scaled)")
    p.add_argument("--data", required=True, help="Original data CSV to learn scaling relationship")
    p.add_argument("--out", required=True, help="Output denormalized CSV")
    return p.parse_args()


def main():
    args = parse_args()
    
    print(f"[load] Loading predictions from {args.predictions}...")
    pred_df = pd.read_csv(args.predictions, sep=';')
    
    print(f"[load] Loading original data from {args.data}...")
    data_df = pd.read_csv(args.data, sep=';')
    
    print(f"[info] Predictions shape: {pred_df.shape}")
    print(f"[info] Data shape: {data_df.shape}")
    
    # Learn the scaling relationship per group
    group_scalers = {}
    
    print("\n[learn] Learning scaling relationship for each group...")
    
    for group_id in data_df['group_id'].unique():
        group_data = data_df[data_df['group_id'] == group_id]
        
        # Get both scaled and actual consumption
        scaled = group_data['scaled_consumption'].values
        actual = group_data['consumption'].values
        
        # Remove NaN values
        valid_mask = ~np.isnan(scaled) & ~np.isnan(actual)
        scaled_valid = scaled[valid_mask]
        actual_valid = actual[valid_mask]
        
        if len(scaled_valid) < 2:
            print(f"[warn] Group {group_id}: Insufficient data, will use mean fallback")
            group_scalers[group_id] = {
                'type': 'mean',
                'mean': actual_valid.mean() if len(actual_valid) > 0 else 0
            }
            continue
        
        # Fit linear regression: actual = a * scaled + b
        scaler = LinearRegression()
        scaler.fit(scaled_valid.reshape(-1, 1), actual_valid)
        
        # Store the scaler
        coef = scaler.coef_[0]
        intercept = scaler.intercept_
        r2 = scaler.score(scaled_valid.reshape(-1, 1), actual_valid)
        
        group_scalers[group_id] = {
            'type': 'linear',
            'coef': coef,
            'intercept': intercept,
            'r2': r2
        }
        
        if group_id in [28, 29, 30, 36]:  # Sample groups
            print(f"[learn] Group {group_id}: actual = {coef:.6f} * scaled + {intercept:.6f} (R²={r2:.4f})")
    
    print(f"\n[info] Learned scaling for {len(group_scalers)} groups")
    
    # Apply denormalization
    print("\n[denorm] Denormalizing predictions...")
    
    denormed_df = pred_df.copy()
    
    for col in pred_df.columns:
        if col == 'measured_at':
            continue
        
        try:
            group_id = int(col)
            
            if group_id not in group_scalers:
                print(f"[warn] Group {group_id} not in training data, using 0")
                denormed_df[col] = 0.0
                continue
            
            scaler_info = group_scalers[group_id]
            scaled_preds = pred_df[col].values.astype(float)
            
            if scaler_info['type'] == 'linear':
                # Apply: actual = coef * scaled + intercept
                denormed_df[col] = scaled_preds * scaler_info['coef'] + scaler_info['intercept']
            else:  # mean fallback
                denormed_df[col] = scaler_info['mean']
            
            # Clip to non-negative (consumption can't be negative)
            denormed_df[col] = denormed_df[col].clip(lower=0)
            
        except (ValueError, KeyError) as e:
            print(f"[warn] Could not process column {col}: {e}")
            continue
    
    # Show sample
    print("\n[sample] Denormalization results for first few groups:")
    sample_groups = [c for c in pred_df.columns if c != 'measured_at'][:5]
    for g in sample_groups:
        print(f"\nGroup {g}:")
        print(f"  Scaled (first 3): {pred_df[g].head(3).values}")
        print(f"  Actual (first 3): {denormed_df[g].head(3).values}")
    
    # Statistics
    print(f"\n[stats] Denormalized predictions:")
    numeric_cols = [c for c in denormed_df.columns if c != 'measured_at']
    all_values = []
    for col in numeric_cols:
        all_values.extend(denormed_df[col].dropna().values)
    
    if all_values:
        print(f"  Min: {np.min(all_values):.2f}")
        print(f"  Max: {np.max(all_values):.2f}")
        print(f"  Mean: {np.mean(all_values):.2f}")
        print(f"  Median: {np.median(all_values):.2f}")
        
        # Count non-zero predictions
        non_zero = sum(1 for v in all_values if v > 0.01)
        print(f"  Non-zero predictions: {non_zero}/{len(all_values)} ({100*non_zero/len(all_values):.1f}%)")
    
    # Save
    print(f"\n[save] Saving to {args.out}...")
    denormed_df.to_csv(args.out, sep=';', index=False)
    
    print(f"[done] Denormalized predictions saved!")
    print(f"[info] Shape: {denormed_df.shape}")


if __name__ == "__main__":
    main()

