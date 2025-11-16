"""Quick utility to inspect CSV structure and identify issues."""

import pandas as pd
import sys

if len(sys.argv) < 2:
    print("Usage: python inspect_csv.py <csv_file>")
    sys.exit(1)

csv_file = sys.argv[1]

print(f"[inspect] Loading {csv_file}...")

# Try different separators
for sep in [';', ',']:
    try:
        df = pd.read_csv(csv_file, sep=sep, nrows=10)
        print(f"\n✓ Successfully loaded with separator: '{sep}'")
        print(f"\nShape: {df.shape}")
        print(f"\nColumns ({len(df.columns)}):")
        for i, col in enumerate(df.columns):
            print(f"  {i+1}. {col}")
        
        print(f"\nFirst 3 rows:")
        print(df.head(3))
        
        print(f"\nData types:")
        print(df.dtypes)
        
        print(f"\nChecking for embedded headers:")
        for col in df.columns:
            if col in df[col].values:
                print(f"  ⚠️ WARNING: Column name '{col}' found in data!")
        
        break
    except Exception as e:
        print(f"✗ Failed with separator '{sep}': {e}")

