"""
Evaluate predictions against ground truth.

Usage:
  py "seq2seq implementation/eval_predictions.py" --pred predictions.csv --truth formatted_features.csv

Optional:
  --per-group           # prints per-group metrics
  --out metrics.csv     # saves per-group metrics to CSV
"""

import argparse
from pathlib import Path
import numpy as np
import pandas as pd


def mae(y, yhat):
    return np.mean(np.abs(y - yhat))

def rmse(y, yhat):
    return np.sqrt(np.mean((y - yhat) ** 2))

def mape(y, yhat, eps=1e-8):
    denom = np.maximum(np.abs(y), eps)
    return np.mean(np.abs((y - yhat) / denom)) * 100.0


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Compare predictions with ground truth and compute metrics.")
    p.add_argument("--pred", type=str, required=True, help="Path to predictions CSV (from infer_hourly.py)")
    p.add_argument("--truth", type=str, required=True, help="Path to formatted_features.csv")
    p.add_argument("--per-group", action="store_true", help="Print per-group metrics")
    p.add_argument("--out", type=str, default=None, help="Optional path to save per-group metrics CSV")
    p.add_argument("--unscaled-only", action="store_true", help="Report only metrics in original (unscaled) units")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    pred_path = Path(args.pred)
    truth_path = Path(args.truth)
    if not pred_path.exists():
        raise FileNotFoundError(f"Pred file not found: {pred_path}")
    if not truth_path.exists():
        raise FileNotFoundError(f"Truth file not found: {truth_path}")

    pred = pd.read_csv(pred_path, sep=";")
    truth = pd.read_csv(truth_path, sep=";")
    # Ensure datetime for join
    pred["measured_at"] = pd.to_datetime(pred["measured_at"])
    truth["measured_at"] = pd.to_datetime(truth["measured_at"])

    # Keep necessary columns
    truth_keep = ["measured_at", "group_id", "consumption"]
    if "scaled_consumption" in truth.columns:
        truth_keep.append("scaled_consumption")
    truth = truth[truth_keep]

    df = pd.merge(
        pred,
        truth,
        on=["measured_at", "group_id"],
        how="inner",
        validate="m:m",
    )
    if df.empty:
        raise ValueError("No overlapping rows between predictions and truth. Check timestamps and groups.")

    # If requested, report ONLY unscaled metrics (approximate inverse scaling)
    if args.unscaled_only and "predicted_consumption_scaled" in df.columns and "scaled_consumption" in df.columns:
        print("[info] Reporting ONLY unscaled metrics (approx inverse scaling)")
        grp = df.groupby("group_id")
        mean_s = grp["scaled_consumption"].transform("mean")
        std_s = grp["scaled_consumption"].transform("std").replace(0, np.nan)
        mean_y = grp["consumption"].transform("mean")
        std_y = grp["consumption"].transform("std")

        yshat = df["predicted_consumption_scaled"].to_numpy()
        a = (std_y / std_s).fillna(1.0)
        b = mean_y - a * mean_s
        yhat_unscaled = a.to_numpy() * yshat + b.to_numpy()
        y_unscaled = df["consumption"].to_numpy()

        print(f"Overall MAE:  {mae(y_unscaled, yhat_unscaled):.6f}")
        print(f"Overall RMSE: {rmse(y_unscaled, yhat_unscaled):.6f}")
        print(f"Overall MAPE: {mape(y_unscaled, yhat_unscaled):.2f}%")

        if args.per_group or args.out:
            rows = []
            for g, gdf in df.groupby("group_id"):
                yhat_s = gdf["predicted_consumption_scaled"].to_numpy()
                ms = gdf["scaled_consumption"].mean()
                ss = gdf["scaled_consumption"].std()
                my = gdf["consumption"].mean()
                sy = gdf["consumption"].std()
                a_g = (sy / ss) if (ss and ss > 0) else 1.0
                b_g = my - a_g * ms
                arr_y_unscaled = gdf["consumption"].to_numpy()
                arr_yhat_unscaled = a_g * yhat_s + b_g
                rows.append({
                    "group_id": g,
                    "MAE": mae(arr_y_unscaled, arr_yhat_unscaled),
                    "RMSE": rmse(arr_y_unscaled, arr_yhat_unscaled),
                    "MAPE_percent": mape(arr_y_unscaled, arr_yhat_unscaled),
                    "count": len(gdf),
                })
            per_group_df = pd.DataFrame(rows).sort_values("RMSE")
            if args.per_group:
                print("\nPer-group (top 10 by lowest RMSE):")
                print(per_group_df.head(10).to_string(index=False))
            if args.out:
                out_path = Path(args.out)
                per_group_df.to_csv(out_path, sep=";", index=False)
                print(f"[save] per-group metrics saved to {out_path}")
        return

    # Handle both column name formats (scaled vs unscaled)
    if "predicted_consumption_scaled" in df.columns:
        # Comparing scaled predictions with scaled truth
        yhat = df["predicted_consumption_scaled"].to_numpy()
        if "scaled_consumption" in df.columns:
            y = df["scaled_consumption"].to_numpy()
            print("[info] Comparing scaled predictions vs scaled truth")
        else:
            raise ValueError("Predictions are scaled but truth doesn't have 'scaled_consumption' column")
    elif "predicted_consumption" in df.columns:
        # Comparing unscaled predictions with unscaled truth
        yhat = df["predicted_consumption"].to_numpy()
        y = df["consumption"].to_numpy()
        print("[info] Comparing unscaled predictions vs unscaled truth")
    else:
        raise ValueError("Predictions CSV must have 'predicted_consumption' or 'predicted_consumption_scaled' column")

    overall_mae = mae(y, yhat)
    overall_rmse = rmse(y, yhat)
    overall_mape = mape(y, yhat)

    print(f"Overall MAE:  {overall_mae:.6f}")
    print(f"Overall RMSE: {overall_rmse:.6f}")
    print(f"Overall MAPE: {overall_mape:.2f}%")

    # Also compute scaled-space MSE if available
    if "predicted_consumption_scaled" in df.columns and "scaled_consumption" in df.columns:
        ys = df["scaled_consumption"].to_numpy()
        yshat = df["predicted_consumption_scaled"].to_numpy()
        scaled_mse = np.mean((ys - yshat) ** 2)
        print(f"Scaled-space MSE: {scaled_mse:.6f}")

        # Additionally compute MAPE in ORIGINAL units by approximating inverse scaling per group
        # Use linear mapping: y ≈ a*ys + b, where a = std_y/std_ys, b = mean_y - a*mean_ys
        # Compute per-group stats on the merged truth
        grp = df.groupby("group_id")
        mean_s = grp["scaled_consumption"].transform("mean")
        std_s = grp["scaled_consumption"].transform("std").replace(0, np.nan)
        mean_y = grp["consumption"].transform("mean")
        std_y = grp["consumption"].transform("std")

        a = (std_y / std_s).fillna(1.0)
        b = mean_y - a * mean_s

        yhat_unscaled = a.to_numpy() * yshat + b.to_numpy()
        y_unscaled = df["consumption"].to_numpy()

        unscaled_mape = mape(y_unscaled, yhat_unscaled)
        print(f"Unscaled-space MAPE (approx inverse scaling): {unscaled_mape:.2f}%")

    if args.per_group or args.out:
        rows = []
        for g, gdf in df.groupby("group_id"):
            # Use same column names as overall evaluation
            if "predicted_consumption_scaled" in gdf.columns:
                arr_y = gdf["scaled_consumption"].to_numpy()
                arr_yhat = gdf["predicted_consumption_scaled"].to_numpy()

                # per-group inverse scaling approximation for MAPE in original units
                ms = gdf["scaled_consumption"].mean()
                ss = gdf["scaled_consumption"].std()
                my = gdf["consumption"].mean()
                sy = gdf["consumption"].std()
                if ss and ss > 0:
                    a_g = sy / ss
                else:
                    a_g = 1.0
                b_g = my - a_g * ms
                arr_y_unscaled = gdf["consumption"].to_numpy()
                arr_yhat_unscaled = a_g * arr_yhat + b_g

                mape_value = mape(arr_y_unscaled, arr_yhat_unscaled)
            else:
                arr_y = gdf["consumption"].to_numpy()
                arr_yhat = gdf["predicted_consumption"].to_numpy()
                mape_value = mape(arr_y, arr_yhat)
            rows.append({
                "group_id": g,
                "MAE": mae(arr_y, arr_yhat),
                "RMSE": rmse(arr_y, arr_yhat),
                "MAPE_percent": mape_value,
                "count": len(gdf),
            })
        per_group_df = pd.DataFrame(rows).sort_values("RMSE")
        if args.per_group:
            print("\nPer-group (top 10 by lowest RMSE):")
            print(per_group_df.head(10).to_string(index=False))
        if args.out:
            out_path = Path(args.out)
            per_group_df.to_csv(out_path, sep=";", index=False)
            print(f"[save] per-group metrics saved to {out_path}")


if __name__ == "__main__":
    main()


