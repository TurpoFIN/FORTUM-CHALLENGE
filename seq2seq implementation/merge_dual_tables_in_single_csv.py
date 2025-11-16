"""
Merge two stacked tables contained in a single CSV into one time-aligned table.

This handles CSV files where Excel displays two separate "tables" one below another
in the same sheet/file (e.g., a consumption table followed by a price table),
each with its own header row (often repeating the timestamp column name).

Approach:
- Detect the second header row by searching for a header token (default: 'measured_at').
- Parse the top section as Table A and the bottom section as Table B.
- Normalize timestamps (with optional rounding) and merge on timestamp + optional group keys.

Examples:
  py "seq2seq implementation/merge_dual_tables_in_single_csv.py" ^
    --file "challenge data/20251111_JUNCTION_example_hourly.csv" ^
    --cons-ts measured_at --cons-col consumption ^
    --price-ts measured_at --price-col price ^
    --group-cols region segment ^
    --out merged.csv

  # If your header token differs (e.g., 'timestamp'):
  py "seq2seq implementation/merge_dual_tables_in_single_csv.py" ^
    --file input.csv --header-token timestamp --cons-ts timestamp --price-ts timestamp ^
    --cons-col consumption --price-col price --out merged.csv
"""

import argparse
from io import StringIO
from pathlib import Path
from typing import List, Optional

import pandas as pd


def find_section_headers(lines: List[str], header_token: str) -> List[int]:
    token_lower = header_token.strip().lower()
    idxs = []
    for i, line in enumerate(lines):
        # Basic detection: non-empty line whose first cell matches token
        first_cell = line.split(";", 1)[0].strip().strip('"').lower()
        if first_cell == token_lower:
            idxs.append(i)
    return idxs


def read_section(lines: List[str], start_idx: int, end_idx: Optional[int], sep: str, decimal: Optional[str]) -> pd.DataFrame:
    buf = StringIO("".join(lines[start_idx:end_idx]))
    return pd.read_csv(buf, sep=sep, decimal=decimal or None)


def normalize_ts(df: pd.DataFrame, ts_col: str, round_to: Optional[str]) -> pd.DataFrame:
    if ts_col not in df.columns:
        raise ValueError(f"Timestamp column '{ts_col}' not found. Available: {list(df.columns)}")
    df = df.copy()
    df[ts_col] = pd.to_datetime(df[ts_col])
    if round_to:
        df[ts_col] = df[ts_col].dt.round(round_to)
    return df


def merge_tables(
    top_df: pd.DataFrame,
    bot_df: pd.DataFrame,
    cons_ts: str,
    price_ts: str,
    cons_col: str,
    price_col: str,
    group_cols: List[str],
    join: str,
) -> pd.DataFrame:
    # Standardize timestamp column name for joining
    top_df = top_df.rename(columns={cons_ts: "_ts_"}).copy()
    bot_df = bot_df.rename(columns={price_ts: "_ts_"}).copy()

    # Keep only needed columns
    top_keep = ["_ts_", cons_col] + [c for c in group_cols if c in top_df.columns]
    bot_keep = ["_ts_", price_col] + [c for c in group_cols if c in bot_df.columns]
    top_df = top_df[top_keep]
    bot_df = bot_df[bot_keep]

    # Join keys: timestamp plus intersection of group cols
    on_cols = ["_ts_"] + [c for c in group_cols if c in top_df.columns and c in bot_df.columns]
    merged = pd.merge(top_df, bot_df, on=on_cols, how=join, validate="m:m")
    merged = merged.rename(columns={"_ts_": "measured_at", cons_col: "consumption", price_col: "price"})
    sort_cols = on_cols.copy()
    sort_cols[0] = "measured_at"
    merged = merged.sort_values(sort_cols).reset_index(drop=True)
    return merged


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Merge two stacked tables in a single CSV into one time-aligned CSV.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--file", type=str, required=True, help="Path to the input CSV (semicolon-separated expected).")
    p.add_argument("--sep", type=str, default=";", help="CSV separator.")
    p.add_argument("--decimal", type=str, default=None, help="CSV decimal (e.g., ',').")
    p.add_argument("--header-token", type=str, default="measured_at", help="Token to detect header rows.")
    p.add_argument("--second-header-row", type=int, default=None, help="Explicit 0-based line index for second header.")
    p.add_argument("--cons-ts", type=str, required=True, help="Timestamp column name in the top table (consumption).")
    p.add_argument("--cons-col", type=str, required=True, help="Consumption value column name in the top table.")
    p.add_argument("--price-ts", type=str, required=True, help="Timestamp column name in the bottom table (price).")
    p.add_argument("--price-col", type=str, required=True, help="Price value column name in the bottom table.")
    p.add_argument("--group-cols", type=str, nargs="*", default=[], help="Optional shared grouping keys.")
    p.add_argument("--round-to", type=str, default="H", help="Round timestamps to this freq (e.g., H, 30min).")
    p.add_argument("--join", type=str, default="inner", choices=["inner", "left", "right", "outer"], help="Join type.")
    p.add_argument("--out", type=str, required=True, help="Output CSV path (semicolon-separated).")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    in_path = Path(args.file)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if not in_path.exists():
        raise FileNotFoundError(f"Input file not found: {in_path}")

    print(f"[read] {in_path}")
    text = in_path.read_text(encoding="utf-8")
    lines = text.splitlines(keepends=True)

    # Detect header rows
    header_idxs = find_section_headers(lines, args.header_token) if args.second_header_row is None else []
    if args.second_header_row is not None:
        header_idxs = [header_idxs[0]] if header_idxs else []
        header_idxs.append(args.second_header_row)
    if len(header_idxs) < 2:
        raise ValueError(
            "Could not detect two header rows. "
            "Provide --second-header-row or check --header-token matches your header's first cell."
        )
    top_start = header_idxs[0]
    bot_start = header_idxs[1]

    # Read sections
    print(f"[parse] section A: lines {top_start}..{bot_start - 1}")
    top_df = read_section(lines, top_start, bot_start, sep=args.sep, decimal=args.decimal)
    print(f"[parse] section B: lines {bot_start}..end")
    bot_df = read_section(lines, bot_start, None, sep=args.sep, decimal=args.decimal)

    # Normalize timestamps
    top_df = normalize_ts(top_df, args.cons_ts, args.round_to)
    bot_df = normalize_ts(bot_df, args.price_ts, args.round_to)

    # Merge
    print(f"[merge] keys: timestamp + {args.group_cols} | join={args.join}")
    merged = merge_tables(
        top_df, bot_df, args.cons_ts, args.price_ts, args.cons_col, args.price_col, args.group_cols, args.join
    )

    print(f"[write] {out_path}")
    merged.to_csv(out_path, sep=";", index=False)
    print("[done] Merge complete.")


if __name__ == "__main__":
    main()


