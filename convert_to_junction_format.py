"""
Convert predictions CSV from long format to Junction format (wide format).

Input format: measured_at;group_id;predicted_consumption;predicted_consumption_scaled
Output format: measured_at;28;29;30;... (group IDs as columns)

Usage:
    python convert_to_junction_format.py --input predictions.csv --output predictions_junction.csv
    python convert_to_junction_format.py --input predictions.csv --output predictions_junction.csv --use-scaled
"""

import argparse
from pathlib import Path

import pandas as pd


def convert_to_junction_format(
    input_path: Path,
    output_path: Path,
    use_scaled: bool = False,
    value_col: str = None,
) -> None:
    """
    Convert predictions from long format to Junction wide format.
    
    Args:
        input_path: Path to input predictions CSV (long format)
        output_path: Path to output CSV (Junction format)
        use_scaled: If True, use predicted_consumption_scaled, else predicted_consumption
        value_col: Explicit column name to use (overrides use_scaled)
    """
    print(f"[read] Reading predictions from {input_path}")
    df = pd.read_csv(input_path, sep=";")
    
    # Determine which value column to use
    if value_col:
        if value_col not in df.columns:
            raise ValueError(f"Column '{value_col}' not found. Available: {list(df.columns)}")
        value_column = value_col
    elif use_scaled:
        if "predicted_consumption_scaled" not in df.columns:
            raise ValueError("predicted_consumption_scaled column not found. Use --use-scaled=false or specify --value-col")
        value_column = "predicted_consumption_scaled"
    else:
        if "predicted_consumption" not in df.columns:
            raise ValueError("predicted_consumption column not found. Use --use-scaled or specify --value-col")
        value_column = "predicted_consumption"
    
    print(f"[convert] Using column: {value_column}")
    
    # Ensure measured_at is datetime
    df["measured_at"] = pd.to_datetime(df["measured_at"])
    
    # Pivot: measured_at as index, group_id as columns, values from value_column
    print(f"[pivot] Pivoting {len(df)} rows...")
    pivoted = df.pivot_table(
        index="measured_at",
        columns="group_id",
        values=value_column,
        aggfunc="first"  # In case of duplicates
    )
    
    # Reset index to make measured_at a column
    pivoted = pivoted.reset_index()
    
    # Format timestamp to match Junction format (ISO with .000Z)
    pivoted["measured_at"] = pivoted["measured_at"].dt.strftime("%Y-%m-%dT%H:%M:%S.000Z")
    
    # Sort columns: measured_at first, then group IDs sorted
    group_cols = sorted([col for col in pivoted.columns if col != "measured_at"])
    pivoted = pivoted[["measured_at"] + group_cols]
    
    print(f"[write] Writing to {output_path}")
    print(f"[info] Shape: {len(pivoted)} rows × {len(pivoted.columns)} columns")
    print(f"[info] Groups: {len(group_cols)}")
    print(f"[info] Time range: {pivoted['measured_at'].min()} to {pivoted['measured_at'].max()}")
    
    # Write with semicolon separator and comma decimal (European format)
    pivoted.to_csv(output_path, sep=";", index=False, decimal=",")
    
    print(f"[done] Conversion complete!")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Convert predictions CSV to Junction format (wide format with group IDs as columns).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--input",
        type=str,
        required=True,
        help="Input predictions CSV file (long format: measured_at;group_id;predicted_consumption;...)",
    )
    p.add_argument(
        "--output",
        type=str,
        required=True,
        help="Output CSV file (Junction format: measured_at;28;29;30;...)",
    )
    p.add_argument(
        "--use-scaled",
        action="store_true",
        help="Use predicted_consumption_scaled instead of predicted_consumption",
    )
    p.add_argument(
        "--value-col",
        type=str,
        default=None,
        help="Explicit column name to use for values (overrides --use-scaled)",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    input_path = Path(args.input)
    output_path = Path(args.output)
    
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    convert_to_junction_format(
        input_path=input_path,
        output_path=output_path,
        use_scaled=args.use_scaled,
        value_col=args.value_col,
    )


if __name__ == "__main__":
    main()

