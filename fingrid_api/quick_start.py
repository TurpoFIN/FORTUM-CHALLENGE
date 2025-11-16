import argparse
import datetime as dt
from typing import Optional

from .api_client import FingridApiClient
from .datasets import (
    VAR_ID_FI_EE,
    VAR_ID_FI_NO4,
    VAR_ID_FI_SE1,
    VAR_ID_FI_SE3,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Quick fetch for Fingrid commercial transfers")
    p.add_argument("--variable", type=int, required=True, help="Dataset ID (e.g., 55 FI-EE phys, 57 FI-NO4 phys, 60 FI-SE1 phys, 61 FI-SE3 phys)")
    p.add_argument("--hours", type=int, default=48, help="Lookback window in hours (default: 48)")
    p.add_argument("--start", type=str, help="ISO start time (UTC). If set, --end must be set.")
    p.add_argument("--end", type=str, help="ISO end time (UTC)")
    p.add_argument("--tz", type=str, default="UTC", help="Timezone for output index (default: UTC)")
    p.add_argument("--max-pages", type=int, default=None, help="Maximum pages to fetch (default: all)")
    p.add_argument("--per-page", type=int, default=1000, help="Items per page (default: 1000)")
    p.add_argument("--csv", type=str, help="Optional CSV path to write")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    client = FingridApiClient()

    if args.start and args.end:
        start = dt.datetime.fromisoformat(args.start.replace("Z", "+00:00"))
        end = dt.datetime.fromisoformat(args.end.replace("Z", "+00:00"))
    else:
        end = dt.datetime.now(dt.timezone.utc).replace(microsecond=0)
        start = end - dt.timedelta(hours=args.hours)

    df = client.fetch_dataframe(
        args.variable, 
        start, 
        end, 
        tz=args.tz,
        max_pages=args.max_pages,
        per_page=args.per_page
    )
    print(df.head())
    print(f"\nRows: {len(df)}  |  From: {df.index.min()}  To: {df.index.max() if len(df) else 'N/A'}")

    if args.csv:
        df.to_csv(args.csv, index=True)
        print(f"Saved CSV: {args.csv}")


if __name__ == "__main__":
    main()


