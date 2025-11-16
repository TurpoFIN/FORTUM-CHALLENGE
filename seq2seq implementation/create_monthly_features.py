"""
Aggregate hourly formatted_features.csv to monthly resolution for 1-year predictions.

Usage:
    python "seq2seq implementation/create_monthly_features.py" \
        --input formatted_features.csv \
        --output formatted_features_monthly.csv
"""

import argparse
from pathlib import Path

import pandas as pd
import numpy as np


def parse_args():
    p = argparse.ArgumentParser(description="Aggregate hourly data to monthly")
    p.add_argument("--input", type=str, default="formatted_features.csv", help="Input hourly CSV")
    p.add_argument("--output", type=str, default="formatted_features_monthly.csv", help="Output monthly CSV")
    return p.parse_args()


def aggregate_to_monthly(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate hourly data to monthly.
    
    Key decisions:
    - Consumption: SUM (total monthly consumption)
    - Price (numeric): MEAN (average monthly price)
    - Categorical features: FIRST (static per group - region, price_type, customer_type, etc.)
    - Time features: Use first of month values
    - Event features: SUM (count of events in month)
    """
    
    # Parse dates
    df['measured_at'] = pd.to_datetime(df['measured_at'])
    
    # Create month period for grouping
    df['year_month'] = df['measured_at'].dt.to_period('M')
    
    # Identify feature types
    consumption_cols = ['scaled_consumption']
    
    # Identify categorical vs numeric columns
    categorical_cols = []
    numeric_cols = []
    
    for col in df.columns:
        if col in ['measured_at', 'group_id', 'year_month']:
            continue
        
        # Check if column is numeric or categorical
        if pd.api.types.is_numeric_dtype(df[col]):
            numeric_cols.append(col)
        else:
            categorical_cols.append(col)
    
    # Also check for object dtype that might contain numbers as strings
    for col in df.columns:
        if col in ['measured_at', 'group_id', 'year_month']:
            continue
        if df[col].dtype == 'object':
            # Try to identify if it's actually numeric stored as string
            try:
                sample = df[col].dropna().iloc[0] if len(df[col].dropna()) > 0 else None
                if sample is not None and isinstance(sample, str):
                    # If it contains text like "Spot Price", it's categorical
                    if any(word in str(sample) for word in ['Price', 'Private', 'Enterprise', 'Finland', 'High', 'Medium', 'Low']):
                        if col not in categorical_cols:
                            categorical_cols.append(col)
                    else:
                        # Might be numeric as string
                        if col not in numeric_cols and col not in categorical_cols:
                            numeric_cols.append(col)
            except:
                pass
    
    # Now identify specific feature types among numeric columns
    price_cols = [c for c in numeric_cols if 'price' in c.lower() and c not in consumption_cols]
    time_cols = [c for c in numeric_cols if any(x in c for x in ['sin', 'cos', 'month', 'day', 'hour', 'week'])]
    event_cols = [c for c in numeric_cols if 'event' in c.lower() or 'holiday' in c.lower()]
    
    # Remaining numeric columns
    other_numeric = [c for c in numeric_cols if c not in consumption_cols + price_cols + time_cols + event_cols]
    
    # Aggregation rules
    agg_dict = {}
    
    # Consumption: SUM (total monthly)
    for col in consumption_cols:
        if col in df.columns:
            agg_dict[col] = 'sum'
    
    # Price: MEAN (average monthly price) - only for numeric price columns
    for col in price_cols:
        if col in df.columns:
            agg_dict[col] = 'mean'
    
    # Time features: FIRST (use month start values)
    for col in time_cols:
        if col in df.columns:
            agg_dict[col] = 'first'
    
    # Event features: SUM (count of events in month)
    for col in event_cols:
        if col in df.columns:
            agg_dict[col] = 'sum'
    
    # Other numeric: MEAN (average)
    for col in other_numeric:
        if col in df.columns:
            agg_dict[col] = 'mean'
    
    # Categorical features: FIRST (static per group)
    for col in categorical_cols:
        if col in df.columns:
            agg_dict[col] = 'first'
    
    # Group by group_id and month
    print("[info] Aggregating to monthly...")
    print(f"[info] Total columns to aggregate: {len(agg_dict)}")
    print(f"[info] Categorical columns ({len(categorical_cols)}): {categorical_cols[:5]}...")
    print(f"[info] Numeric columns ({len(numeric_cols)}): {numeric_cols[:5]}...")
    print(f"[info] Consumption (sum): {consumption_cols}")
    print(f"[info] Prices (mean): {price_cols[:3]}...")
    
    try:
        monthly_df = df.groupby(['group_id', 'year_month']).agg(agg_dict).reset_index()
    except Exception as e:
        print(f"\n[error] Aggregation failed: {e}")
        print(f"\n[debug] Aggregation rules:")
        for col, agg in list(agg_dict.items())[:10]:
            if col in df.columns:
                print(f"  {col} ({df[col].dtype}): {agg} | sample: {repr(df[col].iloc[0])}")
        raise
    
    # Convert period back to timestamp (first of month)
    monthly_df['measured_at'] = monthly_df['year_month'].dt.to_timestamp()
    monthly_df = monthly_df.drop(columns=['year_month'])
    
    # Sort by group and time
    monthly_df = monthly_df.sort_values(['group_id', 'measured_at']).reset_index(drop=True)
    
    return monthly_df


def main():
    args = parse_args()
    
    input_path = Path(args.input)
    output_path = Path(args.output)
    
    print(f"[load] Reading {input_path}...")
    
    # Try reading with different separators
    try:
        df = pd.read_csv(input_path, sep=';')
        print(f"[info] Loaded with ';' separator")
    except:
        df = pd.read_csv(input_path, sep=',')
        print(f"[info] Loaded with ',' separator")
    
    # Check for and remove any duplicate header rows in data
    initial_shape = df.shape
    if 'group_id' in df['group_id'].values:
        print(f"[warn] Found header row in data, removing...")
        df = df[df['group_id'] != 'group_id']
        print(f"[info] Removed {initial_shape[0] - df.shape[0]} header rows")
    
    # Convert numeric columns to numeric type (coerce errors to NaN)
    numeric_cols = [c for c in df.columns if c not in ['measured_at', 'group_id']]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # Drop any rows with NaN in critical columns
    critical_cols = ['measured_at', 'group_id', 'scaled_consumption']
    before_drop = len(df)
    df = df.dropna(subset=critical_cols)
    after_drop = len(df)
    if before_drop != after_drop:
        print(f"[info] Dropped {before_drop - after_drop} rows with missing critical values")
    
    # Ensure group_id is integer
    df['group_id'] = df['group_id'].astype(int)
    
    print(f"[info] Input shape: {df.shape}")
    print(f"[info] Groups: {df['group_id'].nunique()}")
    print(f"[info] Time range: {df['measured_at'].min()} to {df['measured_at'].max()}")
    
    # Aggregate
    monthly_df = aggregate_to_monthly(df)
    
    print(f"\n[info] Output shape: {monthly_df.shape}")
    print(f"[info] Groups: {monthly_df['group_id'].nunique()}")
    print(f"[info] Time range: {monthly_df['measured_at'].min()} to {monthly_df['measured_at'].max()}")
    print(f"[info] Months per group: {monthly_df.groupby('group_id').size().describe()}")
    
    # Check for groups with too few months
    months_per_group = monthly_df.groupby('group_id').size()
    insufficient = months_per_group[months_per_group < 24]
    if len(insufficient) > 0:
        print(f"\n[warn] {len(insufficient)} groups have <24 months of data:")
        print(insufficient)
    
    # Save
    output_path.parent.mkdir(parents=True, exist_ok=True)
    monthly_df.to_csv(output_path, sep=';', index=False)
    print(f"\n[done] Saved to {output_path}")


if __name__ == "__main__":
    main()

