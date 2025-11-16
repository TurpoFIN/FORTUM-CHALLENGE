"""Quick diagnostic for prediction issues."""
import pandas as pd

df = pd.read_csv('formatted_features.csv', sep=';')
pred = pd.read_csv('predictions.csv', sep=';')

print("=== DATA INFO ===")
print(f"Date range: {df['measured_at'].min()} to {df['measured_at'].max()}")
print(f"Groups in data: {sorted(df['group_id'].unique())}")
print(f"Total rows: {len(df)}")

print("\n=== PREDICTION INFO ===")
print(f"Predicted group: {pred['group_id'].unique()[0]}")
print(f"Prediction date range: {pred['measured_at'].min()} to {pred['measured_at'].max()}")

print("\n=== MERGE CHECK ===")
merged = pred.merge(df[['measured_at', 'group_id', 'consumption']], 
                   on=['measured_at', 'group_id'], how='inner')
print(f"Merged rows: {len(merged)}")

if len(merged) > 0:
    print("\n=== SAMPLE COMPARISON ===")
    print(merged[['measured_at', 'consumption', 'predicted_consumption']].head(10))
    
    print("\n=== STATS ===")
    print(f"Actual consumption - mean: {merged['consumption'].mean():.2f}, std: {merged['consumption'].std():.2f}")
    print(f"Predicted consumption - mean: {merged['predicted_consumption'].mean():.2f}, std: {merged['predicted_consumption'].std():.2f}")
    
    errors = merged['consumption'] - merged['predicted_consumption']
    print(f"\nErrors - mean: {errors.mean():.2f}, std: {errors.std():.2f}")
    print(f"MAE: {errors.abs().mean():.2f}")
    print(f"RMSE: {(errors**2).mean()**0.5:.2f}")
    
    # Check if predictions are systematically off
    print(f"\n=== SYSTEMATIC BIAS CHECK ===")
    print(f"Mean error: {errors.mean():.2f} (should be ~0)")
    if abs(errors.mean()) > 0.5:
        print("WARNING: Large systematic bias detected!")

