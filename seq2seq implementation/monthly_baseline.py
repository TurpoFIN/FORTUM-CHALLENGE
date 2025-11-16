"""
Simple baseline: Use average of last 12 months for each group.
This will actually work unlike the broken model.
"""

import pandas as pd
import numpy as np

# Load data
df = pd.read_csv('formatted_features_monthly.csv', sep=';', parse_dates=['measured_at'])
df = df.sort_values(['group_id', 'measured_at'])

# Get last 12 months for each group and average them
predictions = {}

for group_id in sorted(df['group_id'].unique()):
    gdf = df[df['group_id'] == group_id]
    
    # Use last 12 months average
    if len(gdf) >= 12:
        last_12 = gdf.tail(12)['consumption'].values
        avg = last_12.mean()
        
        # Use average for all 12 future months
        predictions[group_id] = [avg] * 12
    else:
        # Fallback: overall average
        predictions[group_id] = [gdf['consumption'].mean()] * 12

print(f"Generated {len(predictions)} predictions")

# Create output
last_date = df['measured_at'].max()
dates = pd.date_range(start=last_date + pd.DateOffset(months=1), periods=12, freq='MS')

output = pd.DataFrame({'measured_at': dates.strftime('%Y-%m-%dT%H:%M:%S.000Z')})

for gid in sorted(predictions.keys()):
    output[gid] = predictions[gid]

# Save
output.to_csv('predictions_monthly_baseline.csv', sep=';', index=False)

print("Saved to predictions_monthly_baseline.csv")
print(f"Sample prediction (group 28): {predictions[28][0]:.2f}")
print(f"Shape: {output.shape}")

