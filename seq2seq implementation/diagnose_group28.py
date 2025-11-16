"""Diagnose why group 28 performs poorly."""
import pandas as pd
import numpy as np

df = pd.read_csv('formatted_features.csv', sep=';')
pred = pd.read_csv('predictions.csv', sep=';')

print("=== GROUP STATISTICS ===")
for g in sorted(df['group_id'].unique()):
    gdf = df[df['group_id'] == g]
    print(f"\nGroup {g}:")
    print(f"  Rows: {len(gdf)}")
    print(f"  Consumption - mean: {gdf['consumption'].mean():.2f}, std: {gdf['consumption'].std():.2f}")
    print(f"  Consumption - min: {gdf['consumption'].min():.2f}, max: {gdf['consumption'].max():.2f}")
    print(f"  Scaled consumption - mean: {gdf['scaled_consumption'].mean():.3f}, std: {gdf['scaled_consumption'].std():.3f}")

print("\n=== GROUP 28 SPECIFIC ANALYSIS ===")
g28 = df[df['group_id'] == 28]
pred28 = pred[pred['group_id'] == 28]
merged = pred28.merge(g28[['measured_at', 'group_id', 'consumption']], on=['measured_at', 'group_id'], how='inner')

if len(merged) > 0:
    print(f"Group 28 predictions vs actual:")
    print(f"  Actual mean: {merged['consumption'].mean():.2f}")
    print(f"  Predicted mean: {merged['predicted_consumption'].mean():.2f}")
    print(f"  Mean error: {(merged['consumption'] - merged['predicted_consumption']).mean():.2f}")
    print(f"  Error std: {(merged['consumption'] - merged['predicted_consumption']).std():.2f}")
    print(f"\n  Sample comparison (first 10):")
    print(merged[['measured_at', 'consumption', 'predicted_consumption']].head(10))

