"""
Create a smaller Excel workbook from the full training file by slicing a date range
and (optionally) subsetting consumption groups.

Default sheet names:
- training_prices (columns: measured_at, eur_per_mwh, ...)
- training_consumption (columns: measured_at, <many group_id columns>)
- groups (metadata; optional pass-through or filtered to included groups)

Usage examples:
  py "seq2seq implementation/slice_training_excel.py" ^
    --in-xlsx "challenge data/20251111_JUNCTION_training.xlsx" ^
    --out-xlsx "challenge data/20251111_JUNCTION_training_small.xlsx" ^
    --start "2021-01-01" --end "2021-02-15"

  # Keep only a few group IDs
  py "seq2seq implementation/slice_training_excel.py" ^
    --in-xlsx "challenge data/20251111_JUNCTION_training.xlsx" ^
    --out-xlsx "challenge data/20251111_JUNCTION_training_small.xlsx" ^
    --start "2021-01-01" --end "2021-02-15" ^
    --groups 225 459 682
"""

import argparse
from pathlib import Path
from typing import List, Optional, Sequence, Union

import pandas as pd


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Slice training workbook by date range and optional group subset.")
    p.add_argument("--in-xlsx", type=str, required=True, help="Path to input training .xlsx")
    p.add_argument("--out-xlsx", type=str, required=True, help="Path to output sliced .xlsx")
    p.add_argument("--start", type=str, required=True, help="Inclusive start datetime (e.g., 2021-01-01)")
    p.add_argument("--end", type=str, required=True, help="Inclusive end datetime (e.g., 2021-02-15)")
    p.add_argument("--prices-sheet", type=str, default="training_prices")
    p.add_argument("--cons-sheet", type=str, default="training_consumption")
    p.add_argument("--groups-sheet", type=str, default="groups")
    p.add_argument("--groups", type=str, nargs="*", default=None, help="Subset of consumption group columns to keep (e.g., 225 459 682)")
    p.add_argument("--max-groups", type=int, default=None, help="Keep first N group columns if --groups not provided")
    return p.parse_args()


def _to_dt(s: pd.Series) -> pd.Series:
    # Preserve tz if present; round-trip compatible with previous pipeline
    dt = pd.to_datetime(s, utc=True, errors="coerce")
    if dt.isna().any():
        # try without forcing UTC if original was naive
        dt2 = pd.to_datetime(s, errors="coerce")
        if dt2.notna().sum() > dt.notna().sum():
            return dt2
    return dt


def _resolve_group_cols(cols: Sequence[Union[str, int]], pick: Optional[List[str]], max_n: Optional[int]) -> List[Union[str, int]]:
    value_cols: List[Union[str, int]] = []
    measured = {"measured_at"}
    if pick:
        # keep provided if exist (int-like strings to int where applicable)
        normalized: List[Union[str, int]] = []
        for g in pick:
            normalized.append(int(g) if isinstance(g, str) and g.isdigit() else g)
        for c in cols:
            if c in measured:
                continue
            if c in normalized:
                value_cols.append(c)
    else:
        # auto-detect numeric-like columns
        for c in cols:
            if c in measured:
                continue
            if isinstance(c, int) or (isinstance(c, str) and c.isdigit()):
                value_cols.append(c)
        if max_n is not None:
            value_cols = value_cols[:max_n]
    if not value_cols:
        raise ValueError("No consumption group columns selected. Use --groups or adjust --max-groups.")
    return value_cols


def main() -> None:
    args = parse_args()
    in_path = Path(args.in_xlsx)
    out_path = Path(args.out_xlsx)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"[read] {in_path}")
    with pd.ExcelFile(in_path) as xls:
        prices = pd.read_excel(xls, sheet_name=args.prices_sheet)
        cons = pd.read_excel(xls, sheet_name=args.cons_sheet)
        groups = pd.read_excel(xls, sheet_name=args.groups_sheet) if args.groups_sheet in xls.sheet_names else None

    # Parse and filter by datetime
    prices["measured_at"] = _to_dt(prices["measured_at"])
    cons["measured_at"] = _to_dt(cons["measured_at"])

    # Make start/end timezone-aware to match the data
    # Parse start/end dates
    start = pd.to_datetime(args.start)
    end = pd.to_datetime(args.end)
    
    # Get timezone from data (if any)
    data_tz = None
    if hasattr(prices["measured_at"].dtype, 'tz') and prices["measured_at"].dtype.tz is not None:
        data_tz = prices["measured_at"].dtype.tz
    elif prices["measured_at"].iloc[0].tz is not None:
        data_tz = prices["measured_at"].iloc[0].tz
    
    # Make start/end timezone-aware to match data
    if data_tz is not None:
        if start.tz is None:
            start = start.tz_localize(data_tz)
        else:
            start = start.tz_convert(data_tz)
        if end.tz is None:
            end = end.tz_localize(data_tz)
        else:
            end = end.tz_convert(data_tz)
    else:
        # Data is timezone-naive, ensure start/end are also naive
        if start.tz is not None:
            start = start.tz_localize(None)
        if end.tz is not None:
            end = end.tz_localize(None)
    
    # Ensure start is at beginning of day (00:00:00)
    start = start.normalize()
    # Ensure end is at end of day (23:59:59.999999) - add 1 day then subtract 1 microsecond
    end = end.normalize() + pd.Timedelta(days=1) - pd.Timedelta(microseconds=1)

    print(f"[filter] Date range: {start} to {end}")
    print(f"[filter] Data range: {prices['measured_at'].min()} to {prices['measured_at'].max()}")
    
    # Filter: >= start (inclusive) and < end (exclusive, but we set end to end of day)
    # Actually, use <= end since we set end to end of day
    prices_small = prices.loc[(prices["measured_at"] >= start) & (prices["measured_at"] <= end)].copy()
    cons_small = cons.loc[(cons["measured_at"] >= start) & (cons["measured_at"] <= end)].copy()
    
    print(f"[filter] Filtered prices: {len(prices_small)} rows (from {len(prices)} total)")
    print(f"[filter] Filtered consumption: {len(cons_small)} rows (from {len(cons)} total)")

    # Subset consumption columns
    group_cols = _resolve_group_cols(cons_small.columns, args.groups, args.max_groups)
    cons_small = cons_small[["measured_at"] + group_cols]

    # Excel cannot store timezone-aware datetimes; drop tz to write
    if getattr(prices_small["measured_at"].dtype, "tz", None) is not None:
        prices_small["measured_at"] = prices_small["measured_at"].dt.tz_localize(None)
    if getattr(cons_small["measured_at"].dtype, "tz", None) is not None:
        cons_small["measured_at"] = cons_small["measured_at"].dt.tz_localize(None)

    # Optionally subset groups sheet to included group ids if present
    if groups is not None and "group_id" in groups.columns and args.groups:
        normalized = [int(g) if isinstance(g, str) and g.isdigit() else g for g in args.groups]
        groups_small = groups[groups["group_id"].isin(normalized)].copy()
    else:
        groups_small = groups

    print(f"[write] {out_path}")
    with pd.ExcelWriter(out_path, engine="openpyxl") as writer:
        prices_small.to_excel(writer, sheet_name=args.prices_sheet, index=False)
        cons_small.to_excel(writer, sheet_name=args.cons_sheet, index=False)
        if groups_small is not None:
            groups_small.to_excel(writer, sheet_name=args.groups_sheet, index=False)
    print("[done] Sliced workbook created.")


if __name__ == "__main__":
    main()


