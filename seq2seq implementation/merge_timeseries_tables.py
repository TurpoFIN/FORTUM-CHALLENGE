"""
Merge separate consumption and price tables into a single time-aligned CSV.

Supports CSV or Excel inputs, custom sheet names, group keys, and join types.
Normalizes timestamp, optional rounding, and writes a unified semicolon-separated CSV.

Examples:
  # CSV → CSV (semicolon-separated)
  py "seq2seq implementation/merge_timeseries_tables.py" ^
    --cons-file "data/consumption.csv" --cons-ts measured_at --cons-col consumption ^
    --price-file "data/price.csv" --price-ts measured_at --price-col price ^
    --group-cols region segment product ^
    --out "data/merged.csv" --join inner --round-to H

  # Excel sheets → CSV
  py "seq2seq implementation/merge_timeseries_tables.py" ^
    --cons-file "training.xlsx" --cons-sheet Consumption --cons-ts measured_at --cons-col consumption ^
    --price-file "training.xlsx" --price-sheet Prices --price-ts measured_at --price-col price ^
    --group-cols region segment ^
    --out "merged.csv"
"""

import argparse
from pathlib import Path
from typing import List, Optional, Sequence, Union

import pandas as pd


def read_table(
    path: Path,
    sheet: Optional[str],
    sep: Optional[str],
    decimal: Optional[str],
) -> pd.DataFrame:
    ext = path.suffix.lower()
    if ext in [".xlsx", ".xls"]:
        if sheet is None:
            df = pd.read_excel(path)
        else:
            df = pd.read_excel(path, sheet_name=sheet)
    else:
        # default to semicolon-separated if not specified
        df = pd.read_csv(path, sep=sep or ";", decimal=decimal or None)
    return df


def normalize_ts(df: pd.DataFrame, ts_col: str, round_to: Optional[str]) -> pd.DataFrame:
    if ts_col not in df.columns:
        raise ValueError(f"Timestamp column '{ts_col}' not found. Available: {list(df.columns)}")
    df = df.copy()
    df[ts_col] = pd.to_datetime(df[ts_col])
    if round_to:
        df[ts_col] = df[ts_col].dt.round(round_to)
    return df


def _resolve_numeric_col(df: pd.DataFrame, col: str) -> Union[str, int]:
    if col in df.columns:
        return col
    if isinstance(col, str) and col.isdigit():
        col_int = int(col)
        if col_int in df.columns:
            return col_int
    return col


def merge_tables(
    cons_df: pd.DataFrame,
    price_df: pd.DataFrame,
    cons_ts: str,
    price_ts: str,
    cons_col: str,
    price_col: str,
    group_cols: List[str],
    join: str,
) -> pd.DataFrame:
    # Rename for consistent keys
    cons_df = cons_df.rename(columns={cons_ts: "_ts_"}).copy()
    price_df = price_df.rename(columns={price_ts: "_ts_"}).copy()

    # Resolve possibly numeric column names passed as strings (e.g., "459")
    cons_col_resolved = _resolve_numeric_col(cons_df, cons_col)
    price_col_resolved = _resolve_numeric_col(price_df, price_col)

    # Select minimal necessary columns to avoid accidental col collisions
    cons_keep = ["_ts_", cons_col_resolved] + [c for c in group_cols if c in cons_df.columns]
    price_keep = ["_ts_", price_col_resolved] + [c for c in group_cols if c in price_df.columns]
    cons_df = cons_df[cons_keep]
    price_df = price_df[price_keep]

    # If group cols differ across tables, merge on intersection only and warn for misses
    on_cols = ["_ts_"] + [c for c in group_cols if c in cons_df.columns and c in price_df.columns]

    merged = pd.merge(cons_df, price_df, on=on_cols, how=join, validate="m:m")
    # Rename back
    merged = merged.rename(columns={"_ts_": "measured_at", cons_col_resolved: "consumption", price_col_resolved: "price"})
    # Sort for determinism
    sort_cols = on_cols.copy()
    sort_cols[0] = "measured_at"
    merged = merged.sort_values(sort_cols).reset_index(drop=True)
    return merged


def melt_consumption_wide(
    cons_df: pd.DataFrame,
    cons_ts: str,
    group_col_name: str = "group_id",
    include_cols: Optional[Sequence[Union[str, int]]] = None,
) -> pd.DataFrame:
    """
    Convert a wide consumption sheet (timestamp + many group columns) to long:
    columns: measured_at, group_id, consumption
    """
    df = cons_df.rename(columns={cons_ts: "_ts_"}).copy()
    id_vars = ["_ts_"]
    # Determine value columns: all except timestamp and any non-numeric metadata
    value_cols: List[Union[str, int]] = []
    for c in df.columns:
        if c in id_vars:
            continue
        # Keep numeric-like or explicitly requested cols
        keep = False
        if include_cols is not None and c in include_cols:
            keep = True
        # If header is numeric (int) or a string digit
        if isinstance(c, int) or (isinstance(c, str) and c.isdigit()):
            keep = True if include_cols is None else keep
        if keep:
            value_cols.append(c)
    if not value_cols:
        raise ValueError("No consumption value columns detected to melt.")
    long_df = df.melt(id_vars=id_vars, value_vars=value_cols, var_name=group_col_name, value_name="consumption")
    # Coerce group ids that are string digits into int
    if long_df[group_col_name].dtype == object:
        with pd.option_context("mode.use_inf_as_na", True):
            numeric_mask = long_df[group_col_name].astype(str).str.fullmatch(r"\d+")
        long_df.loc[numeric_mask, group_col_name] = long_df.loc[numeric_mask, group_col_name].astype(int)
    long_df = long_df.rename(columns={"_ts_": "measured_at"})
    return long_df


def merge_tables_wide(
    cons_df: pd.DataFrame,
    price_df: pd.DataFrame,
    cons_ts: str,
    price_ts: str,
    price_col: str,
    join: str,
    group_col_name: str = "group_id",
    include_cols: Optional[Sequence[Union[str, int]]] = None,
) -> pd.DataFrame:
    # Melt wide consumption to long
    cons_long = melt_consumption_wide(cons_df, cons_ts=cons_ts, group_col_name=group_col_name, include_cols=include_cols)
    # Prepare price
    price_df = price_df.rename(columns={price_ts: "measured_at"}).copy()
    price_col_resolved = _resolve_numeric_col(price_df, price_col)
    price_keep = ["measured_at", price_col_resolved]
    price_df = price_df[price_keep]
    price_df = price_df.rename(columns={price_col_resolved: "price"})
    # Merge on timestamp only (price is global across groups)
    merged = pd.merge(cons_long, price_df, on=["measured_at"], how=join, validate="m:m")
    merged = merged.sort_values(["measured_at", group_col_name]).reset_index(drop=True)
    return merged


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Merge consumption and price tables into one time-aligned CSV.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--cons-file", type=str, required=True, help="Path to consumption table (CSV/Excel).")
    p.add_argument("--cons-sheet", type=str, default=None, help="Excel sheet name for consumption table.")
    p.add_argument("--cons-sep", type=str, default=None, help="CSV separator for consumption table.")
    p.add_argument("--cons-decimal", type=str, default=None, help="CSV decimal for consumption table (e.g., ',').")
    p.add_argument("--cons-ts", type=str, required=True, help="Timestamp column in consumption table.")
    p.add_argument("--cons-col", type=str, required=False, default=None, help="Consumption value column (required unless --cons-wide).")

    p.add_argument("--price-file", type=str, required=True, help="Path to price table (CSV/Excel).")
    p.add_argument("--price-sheet", type=str, default=None, help="Excel sheet name for price table.")
    p.add_argument("--price-sep", type=str, default=None, help="CSV separator for price table.")
    p.add_argument("--price-decimal", type=str, default=None, help="CSV decimal for price table (e.g., ',').")
    p.add_argument("--price-ts", type=str, required=True, help="Timestamp column in price table.")
    p.add_argument("--price-col", type=str, required=True, help="Price value column.")

    p.add_argument("--group-cols", type=str, nargs="*", default=[], help="Shared grouping keys (e.g., region segment).")
    p.add_argument("--cons-wide", action="store_true", help="Treat consumption sheet as wide (many group columns).")
    p.add_argument("--group-col-name", type=str, default="group_id", help="Output column name for melted group id.")
    p.add_argument("--cons-include-cols", type=str, nargs="*", default=None, help="Subset of consumption columns to include (e.g., 459 682).")
    p.add_argument("--round-to", type=str, default="h", help="Round timestamps to this freq (e.g., h, 30min).")
    p.add_argument("--join", type=str, default="inner", choices=["inner", "left", "right", "outer"], help="Join type.")
    p.add_argument("--out", type=str, required=True, help="Output CSV path (semicolon-separated).")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    cons_path = Path(args.cons_file)
    price_path = Path(args.price_file)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if not cons_path.exists():
        raise FileNotFoundError(f"Consumption file not found: {cons_path}")
    if not price_path.exists():
        raise FileNotFoundError(f"Price file not found: {price_path}")

    print(f"[read] consumption: {cons_path}")
    cons_df = read_table(cons_path, args.cons_sheet, args.cons_sep, args.cons_decimal)
    print(f"[read] price: {price_path}")
    price_df = read_table(price_path, args.price_sheet, args.price_sep, args.price_decimal)

    cons_df = normalize_ts(cons_df, args.cons_ts, args.round_to)
    price_df = normalize_ts(price_df, args.price_ts, args.round_to)

    if args.cons_wide:
        print(f"[merge] wide consumption: melting all group columns into '{args.group_col_name}' | join={args.join}")
        include_cols = None
        if args.cons_include_cols is not None:
            # resolve numeric strings into ints where applicable
            include_cols = []
            for c in args.cons_include_cols:
                include_cols.append(int(c) if c.isdigit() else c)
        merged = merge_tables_wide(
            cons_df=cons_df,
            price_df=price_df,
            cons_ts=args.cons_ts,
            price_ts=args.price_ts,
            price_col=args.price_col,
            join=args.join,
            group_col_name=args.group_col_name,
            include_cols=include_cols,
        )
    else:
        if not args.cons_col:
            raise ValueError("--cons-col is required when not using --cons-wide.")
        print(f"[merge] keys: timestamp + {args.group_cols} | join={args.join}")
        merged = merge_tables(
            cons_df=cons_df,
            price_df=price_df,
            cons_ts=args.cons_ts,
            price_ts=args.price_ts,
            cons_col=args.cons_col,
            price_col=args.price_col,
            group_cols=args.group_cols,
            join=args.join,
        )

    print(f"[write] {out_path}")
    merged.to_csv(out_path, sep=";", index=False)
    print("[done] Merge complete.")


if __name__ == "__main__":
    main()


