"""
Script to add is_holiday and is_weekend binary flags to a CSV file.

Usage:
    python add_event_features.py input.csv output.csv
    
    Or with custom timestamp column:
    python add_event_features.py input.csv output.csv --timestamp-col measured_at
"""

import argparse
import pandas as pd
import holidays
from pathlib import Path


def add_event_features(
    input_file: str,
    output_file: str,
    timestamp_col: str = "measured_at",
    country: str = "Finland",
    years: list = None
):
    """
    Add is_holiday and is_weekend binary flags to a CSV file.
    
    Args:
        input_file: Path to input CSV file
        output_file: Path to output CSV file
        timestamp_col: Name of the timestamp column
        country: Country for holiday calendar (default: Finland)
        years: List of years to include in holiday calendar. If None, auto-detect from data.
    """
    # Read the CSV file
    print(f"Reading CSV file: {input_file}")
    df = pd.read_csv(input_file, sep=';')
    
    # Check if timestamp column exists
    if timestamp_col not in df.columns:
        raise ValueError(f"Timestamp column '{timestamp_col}' not found in CSV. Available columns: {df.columns.tolist()}")
    
    # Parse timestamp column
    print(f"Parsing timestamp column: {timestamp_col}")
    df[timestamp_col] = pd.to_datetime(df[timestamp_col])
    
    # Auto-detect years if not provided
    if years is None:
        min_year = df[timestamp_col].dt.year.min()
        max_year = df[timestamp_col].dt.year.max()
        years = list(range(min_year, max_year + 1))
        print(f"Auto-detected years: {years}")
    
    # Create holiday calendar
    print(f"Creating {country} holiday calendar for years {years}")
    if country == "Finland":
        holiday_calendar = holidays.Finland(years=years)
    else:
        # Support for other countries if needed
        holiday_calendar = holidays.country_holidays(country, years=years)
    
    # Get holiday dates as a set for fast lookup
    holiday_dates = set(holiday_calendar.keys())
    print(f"Found {len(holiday_dates)} holidays in the calendar")
    
    # Add is_holiday flag
    print("Adding is_holiday flag...")
    df['is_holiday'] = df[timestamp_col].dt.date.isin(holiday_dates).astype(int)
    
    # Add is_weekend flag (Saturday=5, Sunday=6)
    print("Adding is_weekend flag...")
    df['is_weekend'] = df[timestamp_col].dt.dayofweek.isin([5, 6]).astype(int)
    
    # Optional: Add is_business_day (not holiday and not weekend)
    print("Adding is_business_day flag...")
    df['is_business_day'] = ((df['is_holiday'] == 0) & (df['is_weekend'] == 0)).astype(int)
    
    # Check if columns already exist (to avoid duplicates)
    existing_cols = ['is_holiday', 'is_weekend', 'is_business_day']
    for col in existing_cols:
        if col in df.columns:
            print(f"Warning: Column '{col}' already exists. Overwriting...")
    
    # Save the updated CSV
    print(f"Saving updated CSV to: {output_file}")
    df.to_csv(output_file, sep=';', index=False)
    
    # Print summary statistics
    print("\nSummary:")
    print(f"  Total rows: {len(df)}")
    print(f"  Holidays: {df['is_holiday'].sum()} ({df['is_holiday'].sum() / len(df) * 100:.2f}%)")
    print(f"  Weekends: {df['is_weekend'].sum()} ({df['is_weekend'].sum() / len(df) * 100:.2f}%)")
    print(f"  Business days: {df['is_business_day'].sum()} ({df['is_business_day'].sum() / len(df) * 100:.2f}%)")
    print(f"\nDone! Event features added successfully.")


def main():
    parser = argparse.ArgumentParser(
        description="Add is_holiday and is_weekend binary flags to a CSV file",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python add_event_features.py input.csv output.csv
  python add_event_features.py input.csv output.csv --timestamp-col measured_at
  python add_event_features.py input.csv output.csv --years 2020 2021 2022 2023 2024 2025
        """
    )
    
    parser.add_argument(
        'input_file',
        type=str,
        help='Path to input CSV file'
    )
    
    parser.add_argument(
        'output_file',
        type=str,
        help='Path to output CSV file'
    )
    
    parser.add_argument(
        '--timestamp-col',
        type=str,
        default='measured_at',
        help='Name of the timestamp column (default: measured_at)'
    )
    
    parser.add_argument(
        '--country',
        type=str,
        default='Finland',
        help='Country for holiday calendar (default: Finland)'
    )
    
    parser.add_argument(
        '--years',
        type=int,
        nargs='+',
        default=None,
        help='Years to include in holiday calendar (default: auto-detect from data)'
    )
    
    args = parser.parse_args()
    
    # Validate input file exists
    if not Path(args.input_file).exists():
        raise FileNotFoundError(f"Input file not found: {args.input_file}")
    
    # Create output directory if it doesn't exist
    output_path = Path(args.output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Run the function
    add_event_features(
        input_file=args.input_file,
        output_file=args.output_file,
        timestamp_col=args.timestamp_col,
        country=args.country,
        years=args.years
    )


if __name__ == "__main__":
    main()

