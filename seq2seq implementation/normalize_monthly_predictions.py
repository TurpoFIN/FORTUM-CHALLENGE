"""
Normalize (unscale) monthly predictions from scaled space back to actual consumption.

Usage:
    python "seq2seq implementation/normalize_monthly_predictions.py" \
        --predictions predictions_monthly_junction.csv \
        --data formatted_features_monthly.csv \
        --out predictions_monthly_junction_normalized.csv
"""

import argparse
import pandas as pd
import numpy as np


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--predictions", required=True, help="Predictions CSV (scaled)")
    p.add_argument("--data", required=True, help="Original data CSV to extract scaling params")
    p.add_argument("--out", required=True, help="Output normalized CSV")
    return p.parse_args()


def main():
    args = parse_args()
    
    print(f"[load] Loading predictions from {args.predictions}...")
    pred_df = pd.read_csv(args.predictions, sep=';')
    
    print(f"[load] Loading original data from {args.data}...")
    data_df = pd.read_csv(args.data, sep=';')
    
    print(f"[info] Predictions shape: {pred_df.shape}")
    print(f"[info] Data shape: {data_df.shape}")
    
    # Check if we have both scaled_consumption and consumption columns
    has_both = 'scaled_consumption' in data_df.columns and 'consumption' in data_df.columns
    
    if has_both:
        print("[info] Found both 'consumption' and 'scaled_consumption' - computing scaling params")
        
        # Compute scaling parameters per group
        group_stats = {}
        for group_id in data_df['group_id'].unique():
            group_data = data_df[data_df['group_id'] == group_id]
            
            # Calculate mean and std from the relationship between consumption and scaled_consumption
            consumption = group_data['consumption'].values
            scaled = group_data['scaled_consumption'].values
            
            # If scaling was: scaled = (consumption - mean) / std
            # Then: consumption = scaled * std + mean
            
            # Estimate std and mean
            if len(consumption) > 1:
                # Use linear regression to find relationship
                valid_mask = ~np.isnan(consumption) & ~np.isnan(scaled)
                if valid_mask.sum() > 1:
                    cons_valid = consumption[valid_mask]
                    scaled_valid = scaled[valid_mask]
                    
                    # Simple approach: use actual std and mean
                    mean_val = cons_valid.mean()
                    std_val = cons_valid.std()
                    
                    if std_val == 0:
                        std_val = 1.0
                    
                    group_stats[group_id] = {'mean': mean_val, 'std': std_val}
                else:
                    # Fallback
                    group_stats[group_id] = {'mean': consumption.mean(), 'std': consumption.std() if consumption.std() > 0 else 1.0}
            else:
                group_stats[group_id] = {'mean': 0.0, 'std': 1.0}
        
        print(f"[info] Computed scaling params for {len(group_stats)} groups")
        print(f"[sample] Group 28: mean={group_stats.get(28, {}).get('mean', 0):.2f}, std={group_stats.get(28, {}).get('std', 1):.2f}")
        
    else:
        print("[warn] Could not find both 'consumption' and 'scaled_consumption' columns")
        print("[warn] Will use global scaling or assume predictions are already in correct scale")
        
        # Try to compute global stats
        if 'consumption' in data_df.columns:
            global_mean = data_df['consumption'].mean()
            global_std = data_df['consumption'].std()
        elif 'scaled_consumption' in data_df.columns:
            # Assume scaled_consumption is already normalized, use simple rescaling
            global_mean = 0
            global_std = 1
        else:
            global_mean = 0
            global_std = 1
        
        print(f"[info] Using global scaling: mean={global_mean:.2f}, std={global_std:.2f}")
        group_stats = {gid: {'mean': global_mean, 'std': global_std} for gid in pred_df.columns if gid != 'measured_at'}
    
    # Normalize predictions
    print("\n[normalize] Converting scaled predictions to actual consumption...")
    
    normalized_df = pred_df.copy()
    
    for col in pred_df.columns:
        if col == 'measured_at':
            continue
        
        try:
            group_id = int(col)
            
            if group_id in group_stats:
                stats = group_stats[group_id]
                # Inverse scaling: consumption = scaled * std + mean
                normalized_df[col] = pred_df[col].astype(float) * stats['std'] + stats['mean']
                
                # Ensure non-negative (consumption can't be negative)
                normalized_df[col] = normalized_df[col].clip(lower=0)
            else:
                print(f"[warn] No scaling params for group {group_id}, leaving as-is")
        except (ValueError, KeyError) as e:
            print(f"[warn] Could not process column {col}: {e}")
            continue
    
    # Show sample
    print("\n[sample] Before normalization (first group):")
    first_group_col = [c for c in pred_df.columns if c != 'measured_at'][0]
    print(pred_df[first_group_col].head(3).values)
    
    print(f"\n[sample] After normalization (first group - {first_group_col}):")
    print(normalized_df[first_group_col].head(3).values)
    
    # Statistics
    print(f"\n[stats] Normalized predictions:")
    numeric_cols = [c for c in normalized_df.columns if c != 'measured_at']
    all_values = []
    for col in numeric_cols:
        all_values.extend(normalized_df[col].dropna().values)
    
    if all_values:
        print(f"  Min: {np.min(all_values):.2f}")
        print(f"  Max: {np.max(all_values):.2f}")
        print(f"  Mean: {np.mean(all_values):.2f}")
        print(f"  Median: {np.median(all_values):.2f}")
    
    # Save
    print(f"\n[save] Saving to {args.out}...")
    normalized_df.to_csv(args.out, sep=';', index=False)
    
    print(f"[done] Normalized predictions saved!")
    print(f"[info] Shape: {normalized_df.shape}")


if __name__ == "__main__":
    main()

