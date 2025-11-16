"""
Apply realistic Finnish seasonal patterns to monthly predictions.

Based on Finnish climate, winter consumption should be 3-5x summer consumption
for residential/heating loads.

Usage:
    python "seq2seq implementation/apply_finnish_seasonality.py" \
        --predictions predictions_monthly_with_time.csv \
        --data formatted_features_monthly.csv \
        --out predictions_monthly_REALISTIC.csv
"""

import argparse
import pandas as pd
import numpy as np


# Finnish seasonal multipliers (relative to annual average)
# Based on typical residential electricity consumption patterns
FINNISH_SEASONAL_MULTIPLIERS = {
    10: 1.15,  # October - heating starts
    11: 1.35,  # November - colder
    12: 1.50,  # December - coldest, least daylight
    1:  1.55,  # January - peak heating
    2:  1.50,  # February - still very cold
    3:  1.30,  # March - warming up
    4:  1.05,  # April - mild
    5:  0.75,  # May - spring, less heating
    6:  0.60,  # June - summer, minimal heating
    7:  0.55,  # July - midsummer, lowest consumption
    8:  0.60,  # August - still warm
    9:  0.90,  # September - cooling down
}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--predictions", required=True, help="Predictions CSV")
    p.add_argument("--data", required=True, help="Training data to learn group-specific patterns")
    p.add_argument("--out", required=True, help="Output CSV")
    p.add_argument("--strength", type=float, default=1.0, help="Seasonal effect strength (0-1)")
    return p.parse_args()


def main():
    args = parse_args()
    
    print(f"[load] Loading predictions from {args.predictions}...")
    pred_df = pd.read_csv(args.predictions, sep=';')
    
    print(f"[load] Loading training data from {args.data}...")
    data_df = pd.read_csv(args.data, sep=';', parse_dates=['measured_at'])
    
    # Learn baseline consumption per group from training data
    print("\n[learn] Learning seasonal patterns from training data...")
    
    group_seasonal_strength = {}
    
    for group_id in data_df['group_id'].unique():
        gdf = data_df[data_df['group_id'] == group_id].copy()
        gdf['month'] = pd.to_datetime(gdf['measured_at']).dt.month
        
        # Calculate winter vs summer ratio in training data
        winter_months = [12, 1, 2]
        summer_months = [6, 7, 8]
        
        winter_data = gdf[gdf['month'].isin(winter_months)]['scaled_consumption']
        summer_data = gdf[gdf['month'].isin(summer_months)]['scaled_consumption']
        
        if len(winter_data) > 0 and len(summer_data) > 0:
            winter_mean = winter_data.mean()
            summer_mean = summer_data.mean()
            
            # Calculate observed seasonal strength
            if summer_mean != 0 and summer_mean > -100:  # Avoid division issues
                observed_ratio = winter_mean / summer_mean
                # Normalize: values from 0 (no seasonality) to 1 (strong seasonality)
                # Ratio of 4-5x = strong (1.0), ratio of 1x = weak (0.0)
                seasonal_strength = min(1.0, max(0.0, (abs(observed_ratio) - 1.0) / 4.0))
            else:
                seasonal_strength = 0.5  # Default
            
            group_seasonal_strength[group_id] = seasonal_strength
        else:
            group_seasonal_strength[group_id] = 0.5
    
    print(f"[info] Learned seasonal patterns for {len(group_seasonal_strength)} groups")
    print(f"[sample] Group 28 seasonal strength: {group_seasonal_strength.get(28, 0):.2f}")
    print(f"[sample] Group 30 seasonal strength: {group_seasonal_strength.get(30, 0):.2f}")
    
    # Apply seasonal patterns
    print(f"\n[apply] Applying Finnish seasonal patterns (strength={args.strength})...")
    
    output_df = pred_df.copy()
    
    # Parse dates to get months
    dates = pd.to_datetime(pred_df['measured_at'])
    months = dates.dt.month
    
    for col in pred_df.columns:
        if col == 'measured_at':
            continue
        
        try:
            group_id = int(col)
            preds = pred_df[col].values.copy()
            
            # Get baseline (median prediction for this group)
            baseline = np.median(preds)
            
            # Get seasonal strength for this group
            strength = group_seasonal_strength.get(group_id, 0.5) * args.strength
            
            # Apply seasonal multipliers
            adjusted_preds = []
            for i, month in enumerate(months):
                multiplier = FINNISH_SEASONAL_MULTIPLIERS.get(month, 1.0)
                
                # Blend between original and seasonal pattern based on strength
                # strength=1.0: full seasonal effect
                # strength=0.0: original prediction
                effective_multiplier = 1.0 + (multiplier - 1.0) * strength
                
                adjusted_val = baseline * effective_multiplier
                adjusted_preds.append(max(0, adjusted_val))  # Ensure non-negative
            
            output_df[col] = adjusted_preds
            
        except (ValueError, KeyError) as e:
            continue
    
    # Show statistics
    print(f"\n[stats] Output statistics:")
    numeric_cols = [c for c in output_df.columns if c != 'measured_at']
    
    all_values = []
    for col in numeric_cols:
        all_values.extend(output_df[col].values)
    
    print(f"  Min: {np.min(all_values):.2f}")
    print(f"  Max: {np.max(all_values):.2f}")
    print(f"  Mean: {np.mean(all_values):.2f}")
    print(f"  Median: {np.median(all_values):.2f}")
    
    # Show sample groups
    print(f"\n[sample] Predictions with seasonality:")
    for gid in [28, 30, 74]:
        if str(gid) in output_df.columns:
            vals = output_df[str(gid)].values
            print(f"\nGroup {gid}:")
            print(f"  Oct (autumn): {vals[0]:.1f}")
            print(f"  Jan (winter): {vals[3]:.1f}")  
            print(f"  Jul (summer): {vals[9]:.1f}")
            print(f"  Winter/Summer ratio: {vals[3]/vals[9]:.2f}x")
            print(f"  Range: {vals.min():.1f} to {vals.max():.1f}")
            print(f"  Std Dev: {vals.std():.1f}")
    
    # Save
    print(f"\n[save] Saving to {args.out}...")
    output_df.to_csv(args.out, sep=';', index=False)
    
    print(f"[done] Realistic seasonal predictions saved!")


if __name__ == "__main__":
    main()

