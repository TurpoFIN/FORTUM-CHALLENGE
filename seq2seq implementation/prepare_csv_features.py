"""
Prepare a CSV with model-ready features for Seq2Seq training/inference.

What it does:
- Parses timestamp column
- Merges group metadata (province, region, customer type, etc.) if provided
- Adds cyclical time features (hour/day-of-week/month sin/cos)
- Adds event features (is_holiday, is_weekend, is_business_day)
- Adds lagged consumption features (e.g., t-24, t-48, t-168)
- Scales numeric features (e.g., consumption, price, weather columns)
- Saves a new CSV with the engineered features

Usage examples:
  py "seq2seq implementation/prepare_csv_features.py" input.csv output.csv ^
    --groups-file "challenge data/groups.md" ^
    --encode-categoricals ^
    --timestamp-col measured_at ^
    --consumption-col consumption ^
    --price-col price ^
    --weather-cols temperature wind_speed ^
    --lags 24 48 168 ^
    --scaler standard ^
    --drop-na

  # Fit scalers on train period only and save them for reuse
  py "seq2seq implementation/prepare_csv_features.py" input.csv output.csv ^
    --groups-file "challenge data/groups.md" ^
    --encode-categoricals ^
    --train-end 2024-09-30 ^
    --save-scalers-dir artifacts/scalers
"""

import argparse
from pathlib import Path
from typing import List, Optional, Dict, Any

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from joblib import dump
import holidays


def load_group_metadata(groups_file: Path) -> pd.DataFrame:
    """
    Load and parse groups metadata from groups.md file.
    
    The group_label format is: Region | Province | Municipality | Customer Type | Price Type | Consumption Level
    
    Returns a DataFrame with columns:
    - group_id: int
    - region: str
    - province: str
    - municipality: str
    - customer_type: str
    - price_type: str
    - consumption_level: str
    """
    print(f"[groups] Loading group metadata from {groups_file}")
    
    # Read the tab-separated file - handle potential encoding issues
    df = pd.read_csv(groups_file, sep="\t", encoding='utf-8', on_bad_lines='skip')
    
    # Clean column names (remove trailing spaces)
    df.columns = df.columns.str.strip()
    
    # Check if columns are detected
    if 'group_label' not in df.columns:
        # Try to manually parse if automatic parsing failed
        print("[groups] Auto-parse failed, trying manual parsing...")
        with open(groups_file, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        
        # Skip header, parse data lines
        data = []
        for line in lines[1:]:  # Skip header
            line = line.strip()
            if not line or line == '':
                continue
            parts = line.split('\t')
            if len(parts) >= 2:
                try:
                    group_id = int(parts[0])
                    group_label = parts[1]
                    data.append({'group_id': group_id, 'group_label': group_label})
                except (ValueError, IndexError):
                    continue
        
        df = pd.DataFrame(data)
    
    # Remove any incomplete rows
    df = df.dropna(subset=['group_label'])
    
    # Strip any leading/trailing whitespace from group_label
    df['group_label'] = df['group_label'].str.strip()
    
    # Parse the group_label column - escape the pipe character as it's a regex special char
    parsed = df['group_label'].str.split(' \\| ', expand=True, regex=True)
    
    # Ensure we have 6 columns (Region | Province | Municipality | Customer Type | Price Type | Consumption Level)
    if parsed.shape[1] >= 6:
        df['region'] = parsed[0].str.strip()
        df['province'] = parsed[1].str.strip()
        df['municipality'] = parsed[2].str.strip()
        df['customer_type'] = parsed[3].str.strip()
        df['price_type'] = parsed[4].str.strip()
        df['consumption_level'] = parsed[5].str.strip()
    else:
        print(f"[warn] Unexpected group_label format. Expected 6 parts, got {parsed.shape[1]}")
        print(f"[warn] Sample row: {df['group_label'].iloc[0] if len(df) > 0 else 'N/A'}")
        df['region'] = parsed[0].str.strip() if parsed.shape[1] > 0 else None
        df['province'] = parsed[1].str.strip() if parsed.shape[1] > 1 else None
        df['municipality'] = parsed[2].str.strip() if parsed.shape[1] > 2 else None
        df['customer_type'] = parsed[3].str.strip() if parsed.shape[1] > 3 else None
        df['price_type'] = parsed[4].str.strip() if parsed.shape[1] > 4 else None
        df['consumption_level'] = parsed[5].str.strip() if parsed.shape[1] > 5 else None
    
    # Keep only the parsed columns
    result = df[['group_id', 'region', 'province', 'municipality', 'customer_type', 'price_type', 'consumption_level']]
    
    print(f"[groups] Loaded {len(result)} group metadata entries")
    print(f"[groups] Unique provinces: {result['province'].nunique()}")
    print(f"[groups] Sample provinces: {result['province'].unique()[:5].tolist()}")
    
    return result


def add_time_cyclical_features(df: pd.DataFrame, ts_col: str) -> pd.DataFrame:
    ts = df[ts_col]
    hours = ts.dt.hour.astype(float)
    dows = ts.dt.dayofweek.astype(float)  # Monday=0
    months = ts.dt.month.astype(float)

    df["hour_sin"] = np.sin(2 * np.pi * hours / 24.0)
    df["hour_cos"] = np.cos(2 * np.pi * hours / 24.0)
    df["day_of_week_sin"] = np.sin(2 * np.pi * dows / 7.0)
    df["day_of_week_cos"] = np.cos(2 * np.pi * dows / 7.0)
    df["month_sin"] = np.sin(2 * np.pi * months / 12.0)
    df["month_cos"] = np.cos(2 * np.pi * months / 12.0)
    return df


def add_event_features(
    df: pd.DataFrame,
    ts_col: str,
    country: str = "Finland",
    years: Optional[List[int]] = None,
) -> pd.DataFrame:
    if years is None:
        years = list(range(int(df[ts_col].dt.year.min()), int(df[ts_col].dt.year.max()) + 1))
    if country == "Finland":
        cal = holidays.Finland(years=years)
    else:
        cal = holidays.country_holidays(country, years=years)
    holiday_dates = set(cal.keys())

    df["is_holiday"] = df[ts_col].dt.date.isin(holiday_dates).astype(int)
    df["is_weekend"] = df[ts_col].dt.dayofweek.isin([5, 6]).astype(int)
    df["is_business_day"] = ((df["is_holiday"] == 0) & (df["is_weekend"] == 0)).astype(int)
    return df


def add_lag_features(
    df: pd.DataFrame,
    ts_col: str,
    value_col: str,
    lags: List[int],
    group_cols: Optional[List[str]] = None,
) -> pd.DataFrame:
    if group_cols:
        df = df.sort_values(group_cols + [ts_col]).copy()
        group_obj = df.groupby(group_cols, group_keys=False)
        for lag in lags:
            df[f"{value_col}_lag_{lag}h"] = group_obj[value_col].shift(lag)
    else:
        df = df.sort_values(ts_col).copy()
        for lag in lags:
            df[f"{value_col}_lag_{lag}h"] = df[value_col].shift(lag)
    return df


def encode_categorical_features(df: pd.DataFrame, categorical_cols: List[str]) -> pd.DataFrame:
    """
    One-hot encode categorical features for model input.
    
    Args:
        df: DataFrame with categorical columns
        categorical_cols: List of column names to encode
    
    Returns:
        DataFrame with one-hot encoded columns added
    """
    print(f"[encode] One-hot encoding categorical features: {categorical_cols}")
    
    for col in categorical_cols:
        if col not in df.columns:
            print(f"[warn] Categorical column '{col}' not found; skipping.")
            continue
        
        # Create one-hot encoded columns with clean names
        dummies = pd.get_dummies(df[col], prefix=col, prefix_sep='_')
        
        # Convert to float for consistency with other features
        dummies = dummies.astype(float)
        
        # Add to dataframe
        df = pd.concat([df, dummies], axis=1)
        
        print(f"[encode]   {col}: {len(dummies.columns)} categories -> {list(dummies.columns)[:3]}...")
    
    return df


def build_scalers(
    scaler_type: str,
    cols: List[str],
    save_dir: Optional[Path] = None,
    prefix: str = "",
) -> Dict[str, Any]:
    scalers: Dict[str, Any] = {}
    for col in cols:
        if scaler_type == "standard":
            scalers[col] = StandardScaler()
        elif scaler_type == "minmax":
            scalers[col] = MinMaxScaler()
        else:
            raise ValueError(f"Unknown scaler type: {scaler_type}")
        if save_dir:
            save_dir.mkdir(parents=True, exist_ok=True)
    return scalers


def fit_transform_scalers(
    df_train: pd.DataFrame,
    df_all: pd.DataFrame,
    scalers: Dict[str, Any],
    group_col: Optional[str] = None,
) -> pd.DataFrame:
    df_out = df_all.copy()
    
    if group_col and group_col in df_all.columns:
        # Per-group scaling
        print(f"[scale] Using per-group scaling (group_col='{group_col}').")
        for col, scaler_template in scalers.items():
            if col not in df_all.columns:
                print(f"[warn] Column '{col}' not found; skipping scaling.")
                continue
            
            # Create scaler per group
            group_scalers = {}
            for group_id in df_train[group_col].unique():
                group_train = df_train[df_train[group_col] == group_id]
                if len(group_train) == 0:
                    continue
                
                # Create new scaler instance for this group
                if isinstance(scaler_template, StandardScaler):
                    group_scalers[group_id] = StandardScaler()
                elif isinstance(scaler_template, MinMaxScaler):
                    group_scalers[group_id] = MinMaxScaler()
                else:
                    raise ValueError(f"Unknown scaler type: {type(scaler_template)}")
                
                values_train = group_train[[col]].astype(float).values
                group_scalers[group_id].fit(values_train)
            
            # Transform each group separately, maintaining original order
            scaled_series = pd.Series(index=df_all.index, dtype=float)
            for group_id in df_all[group_col].unique():
                group_mask = df_all[group_col] == group_id
                group_data = df_all[group_mask]
                if group_id in group_scalers:
                    values = group_data[[col]].astype(float).values
                    scaled = group_scalers[group_id].transform(values).flatten()
                else:
                    # Fallback: use global scaler if group not in training
                    print(f"[warn] Group {group_id} not in training; using global scaler.")
                    if col not in scalers:
                        scalers[col] = scaler_template
                        values_train = df_train[[col]].astype(float).values
                        scalers[col].fit(values_train)
                    values = group_data[[col]].astype(float).values
                    scaled = scalers[col].transform(values).flatten()
                scaled_series.loc[group_mask] = scaled
            
            df_out[f"scaled_{col}"] = scaled_series.values
    else:
        # Global scaling (original behavior)
        for col, scaler in scalers.items():
            if col not in df_all.columns:
                print(f"[warn] Column '{col}' not found; skipping scaling.")
                continue
            values_train = df_train[[col]].astype(float).values
            scaler.fit(values_train)
            values_all = df_all[[col]].astype(float).values
            df_out[f"scaled_{col}"] = scaler.transform(values_all).astype(float)
    
    return df_out


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Prepare CSV with engineered features for Seq2Seq.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("input_file", type=str, help="Path to input CSV (semicolon-separated expected).")
    p.add_argument("output_file", type=str, help="Path to output CSV.")
    p.add_argument("--groups-file", type=str, default=None, help="Path to groups.md file to add province and other group metadata.")
    p.add_argument(
        "--encode-categoricals",
        action="store_true",
        help="One-hot encode categorical group features (province, region, customer_type, price_type, municipality, consumption_level).",
    )
    p.add_argument("--timestamp-col", type=str, default="measured_at", help="Timestamp column name.")
    p.add_argument("--consumption-col", type=str, default="consumption", help="Consumption column name.")
    p.add_argument("--price-col", type=str, default=None, help="Price column name, if available.")
    p.add_argument(
        "--weather-cols",
        type=str,
        nargs="*",
        default=[],
        help="Weather columns to include (e.g., temperature wind_speed).",
    )
    p.add_argument("--group-cols", type=str, nargs="*", default=[], help="Optional group columns for per-group lags.")
    p.add_argument("--per-group-scaling", action="store_true", help="Scale numeric features per group (recommended if groups have very different scales).")
    p.add_argument("--lags", type=int, nargs="*", default=[24, 48, 168], help="Lag hours for consumption.")
    p.add_argument("--country", type=str, default="Finland", help="Country for holiday calendar.")
    p.add_argument("--years", type=int, nargs="*", default=None, help="Years for holiday calendar (auto if omitted).")
    p.add_argument(
        "--scaler",
        type=str,
        choices=["standard", "minmax"],
        default="standard",
        help="Scaler type for numeric columns.",
    )
    p.add_argument(
        "--scale-cols",
        type=str,
        nargs="*",
        default=[],
        help="Explicit columns to scale (in addition to consumption/price/weather).",
    )
    p.add_argument(
        "--train-end",
        type=str,
        default=None,
        help="ISO date/time; scalers fit on rows <= train_end (train-only fit).",
    )
    p.add_argument("--save-scalers-dir", type=str, default=None, help="Directory to save fitted scalers (joblib).")
    p.add_argument(
        "--drop-na",
        action="store_true",
        help="Drop rows with NA introduced by lags or missing data.",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()

    in_path = Path(args.input_file)
    out_path = Path(args.output_file)
    if not in_path.exists():
        raise FileNotFoundError(f"Input file not found: {in_path}")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"[read] {in_path}")
    df = pd.read_csv(in_path, sep=";")

    # Load and merge group metadata if provided
    if args.groups_file:
        groups_path = Path(args.groups_file)
        if not groups_path.exists():
            raise FileNotFoundError(f"Groups file not found: {groups_path}")
        
        group_metadata = load_group_metadata(groups_path)
        
        # Check if group_id exists in the dataframe
        if 'group_id' in df.columns:
            print(f"[groups] Merging group metadata on 'group_id'")
            df = df.merge(group_metadata, on='group_id', how='left')
            
            # Report any missing group metadata
            missing_groups = df[df['province'].isna()]['group_id'].unique()
            if len(missing_groups) > 0:
                print(f"[warn] {len(missing_groups)} group_id(s) have no metadata: {missing_groups[:5]}...")
        else:
            print("[warn] 'group_id' column not found in data; cannot merge group metadata.")

    # Parse timestamp
    ts_col = args.timestamp_col
    if ts_col not in df.columns:
        raise ValueError(f"Timestamp column '{ts_col}' not found. Available: {list(df.columns)}")
    df[ts_col] = pd.to_datetime(df[ts_col])

    # Sort for deterministic operations (and within groups if provided)
    sort_cols = (args.group_cols or []) + [ts_col]
    df = df.sort_values(sort_cols).reset_index(drop=True)

    # Add cyclical time features
    print("[feat] Adding cyclical time features (hour/day-of-week/month).")
    df = add_time_cyclical_features(df, ts_col)

    # Add event features
    print("[feat] Adding event features (is_holiday, is_weekend, is_business_day).")
    df = add_event_features(df, ts_col, country=args.country, years=args.years)

    # Add lagged consumption features
    cons_col = args.consumption_col
    if cons_col in df.columns:
        print(f"[feat] Adding lag features for '{cons_col}': {args.lags}.")
        df = add_lag_features(df, ts_col, cons_col, args.lags, args.group_cols or None)
    else:
        print(f"[warn] Consumption column '{cons_col}' not found; skipping lag features.")

    # One-hot encode categorical features if requested
    if args.encode_categoricals:
        categorical_cols = ['region', 'province', 'municipality', 'customer_type', 'price_type', 'consumption_level']
        # Only encode columns that exist in the dataframe
        existing_categoricals = [col for col in categorical_cols if col in df.columns]
        if existing_categoricals:
            df = encode_categorical_features(df, existing_categoricals)
        else:
            print("[warn] --encode-categoricals specified but no categorical columns found.")
            print("       Make sure you've provided --groups-file to add categorical metadata.")

    # Decide which numeric columns to scale
    scale_cols: List[str] = []
    if cons_col in df.columns:
        scale_cols.append(cons_col)
    if args.price_col and args.price_col in df.columns:
        scale_cols.append(args.price_col)
    for wcol in args.weather_cols:
        if wcol in df.columns:
            scale_cols.append(wcol)
        else:
            print(f"[warn] Weather column '{wcol}' not found; skipping.")
    for scol in args.scale_cols:
        if scol in df.columns and scol not in scale_cols:
            scale_cols.append(scol)
        elif scol not in df.columns:
            print(f"[warn] Scale column '{scol}' not found; skipping.")

    # Fit scalers on train-only subset if provided
    if args.train_end:
        train_end_ts = pd.to_datetime(args.train_end)
        df_train = df[df[ts_col] <= train_end_ts]
        if df_train.empty:
            raise ValueError(f"No rows <= train_end '{args.train_end}' for scaler fitting.")
        print(f"[scale] Train-only fit on <= {train_end_ts} (rows: {len(df_train)}).")
    else:
        df_train = df
        print(f"[scale] Fit on all rows (no train_end provided).")

    scalers = build_scalers(
        scaler_type=args.scaler,
        cols=scale_cols,
        save_dir=Path(args.save_scalers_dir) if args.save_scalers_dir else None,
    )
    
    # Determine group column for per-group scaling
    group_col_for_scaling = None
    if args.per_group_scaling:
        if args.group_cols and len(args.group_cols) > 0:
            # Use first group column for scaling
            group_col_for_scaling = args.group_cols[0]
            print(f"[scale] Per-group scaling enabled (using '{group_col_for_scaling}').")
        else:
            print("[warn] --per-group-scaling requested but no --group-cols provided; using global scaling.")
    
    df = fit_transform_scalers(df_train, df, scalers, group_col=group_col_for_scaling)

    # Persist scalers if requested
    if args.save_scalers_dir:
        save_dir = Path(args.save_scalers_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        for col, scaler in scalers.items():
            dump(scaler, save_dir / f"{col}_{args.scaler}_scaler.joblib")
        print(f"[save] Scalers saved to {save_dir}")

    # Optionally drop rows with NA (due to lags or missing values)
    if args.drop_na:
        before = len(df)
        df = df.dropna().reset_index(drop=True)
        print(f"[clean] Dropped {before - len(df)} rows with NA (now {len(df)} rows).")

    # Save output
    print(f"[write] {out_path}")
    df.to_csv(out_path, sep=";", index=False)
    print("[done] Feature preparation completed.")


if __name__ == "__main__":
    main()


