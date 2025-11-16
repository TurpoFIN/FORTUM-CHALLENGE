"""
Generate 48-hour forecasts for all groups using a trained Seq2Seq model.

Usage:
    py "seq2seq implementation/infer_hourly.py" \
        --model artifacts/models/seq2seq_hourly.pt \
        --csv formatted_features.csv \
        --out predictions.csv
"""

import argparse
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from datasets.hourly_window import HourlySeq2SeqDataset
from models.seq2seq_lstm import Seq2Seq


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate 48-hour forecasts from trained model.")
    p.add_argument("--model", type=str, required=True, help="Path to trained model checkpoint (.pt)")
    p.add_argument("--csv", type=str, required=True, help="Path to formatted features CSV")
    p.add_argument("--out", type=str, required=True, help="Output CSV path for predictions")
    p.add_argument("--window-size", type=int, default=168, help="Encoder window size (must match training)")
    p.add_argument("--horizon", type=int, default=48, help="Forecast horizon (must match training)")
    p.add_argument("--batch-size", type=int, default=32, help="Batch size for inference")
    p.add_argument("--start-at", type=str, default=None, help="Optional ISO start timestamp for forecast (e.g., 2021-02-14 00:00:00). If unset, uses last_time+1h per group.")
    p.add_argument("--offset-hours", type=int, default=0, help="If >0, start at (last_time - offset_hours + 1h) so predictions can align within history (e.g., use 48).")
    return p.parse_args()


def build_decoder_known_covariates(
    gdf: pd.DataFrame,
    start_time: pd.Timestamp,
    horizon: int,
    time_cols: List[str],
    event_cols: List[str],
    price_col: str,
    extra_dec_cols: Optional[List[str]] = None,
) -> torch.Tensor:
    """
    Build decoder known covariates for future timestamps.
    
    Args:
        df: Full dataframe (for price lookup if available)
        start_time: Start timestamp for forecast (t+1)
        horizon: Number of hours to forecast
        time_cols: List of cyclical time feature column names
        event_cols: List of event feature column names
        price_col: Price column name
        extra_dec_cols: Optional extra decoder columns
    
    Returns:
        Tensor of shape [horizon, F_dec] with known future covariates
    """
    import holidays
    
    # Generate future timestamps
    future_times = pd.date_range(start=start_time, periods=horizon, freq="H")
    
    # Build features for each future timestamp
    features_list = []
    # Pre-index group dataframe by timestamp for fast lookups
    gdf_idx = gdf.set_index("measured_at") if "measured_at" in gdf.columns else gdf

    for ts in future_times:
        feat = []
        
        # Prefer true covariates if the timestamp exists in the group data
        row = None
        if isinstance(gdf_idx.index, pd.DatetimeIndex):
            if ts in gdf_idx.index:
                row = gdf_idx.loc[ts]
                # If duplicated index, take last
                if isinstance(row, pd.DataFrame):
                    row = row.iloc[-1]

        if row is not None:
            # Use true time/event if present, otherwise compute
            # Time features
            if all(col in row.index for col in ["hour_sin", "hour_cos", "day_of_week_sin", "day_of_week_cos", "month_sin", "month_cos"]):
                feat.extend([row["hour_sin"], row["hour_cos"], row["day_of_week_sin"], row["day_of_week_cos"], row["month_sin"], row["month_cos"]])
            else:
                hour = ts.hour
                dow = ts.dayofweek
                month = ts.month
                feat.extend([
                    np.sin(2 * np.pi * hour / 24),
                    np.cos(2 * np.pi * hour / 24),
                    np.sin(2 * np.pi * dow / 7),
                    np.cos(2 * np.pi * dow / 7),
                    np.sin(2 * np.pi * month / 12),
                    np.cos(2 * np.pi * month / 12),
                ])
            # Event features
            if all(col in row.index for col in ["is_holiday", "is_weekend"]):
                feat.extend([row["is_holiday"], row["is_weekend"]])
            else:
                fi_holidays = holidays.Finland(years=ts.year)
                is_holiday = 1 if ts.date() in fi_holidays else 0
                is_weekend = 1 if ts.dayofweek >= 5 else 0
                feat.extend([is_holiday, is_weekend])
            # Price (scaled) true if present, else fallback
            if price_col in row.index and not pd.isna(row[price_col]):
                price_val = row[price_col]
            else:
                price_val = gdf[price_col].iloc[-1] if price_col in gdf.columns and len(gdf) > 0 else 0.0
            feat.append(price_val)
            # Extra decoder cols (use row if present)
            if extra_dec_cols:
                for col in extra_dec_cols:
                    val = row[col] if (col in row.index and not pd.isna(row[col])) else (gdf[col].iloc[-1] if col in gdf.columns and len(gdf) > 0 else 0.0)
                    feat.append(val)
        else:
            # Fallback: compute time/event; price from last known
            hour = ts.hour
            dow = ts.dayofweek
            month = ts.month
            feat.extend([
                np.sin(2 * np.pi * hour / 24),
                np.cos(2 * np.pi * hour / 24),
                np.sin(2 * np.pi * dow / 7),
                np.cos(2 * np.pi * dow / 7),
                np.sin(2 * np.pi * month / 12),
                np.cos(2 * np.pi * month / 12),
            ])
            fi_holidays = holidays.Finland(years=ts.year)
            is_holiday = 1 if ts.date() in fi_holidays else 0
            is_weekend = 1 if dow >= 5 else 0
            feat.extend([is_holiday, is_weekend])
            price_val = gdf[price_col].iloc[-1] if price_col in gdf.columns and len(gdf) > 0 else 0.0
            feat.append(price_val)
            if extra_dec_cols:
                for col in extra_dec_cols:
                    val = gdf[col].iloc[-1] if col in gdf.columns and len(gdf) > 0 else 0.0
                    feat.append(val)
        
        features_list.append(feat)
    
    return torch.tensor(features_list, dtype=torch.float32)


def main() -> None:
    args = parse_args()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[device] using {device}")
    
    # Load checkpoint
    print(f"[load] loading model from {args.model}")
    ckpt = torch.load(args.model, map_location=device)
    config = ckpt.get("config", {})
    
    # Override with command-line args if provided
    window_size = args.window_size
    horizon = args.horizon
    
    # Load data
    print(f"[load] loading data from {args.csv}")
    df = pd.read_csv(args.csv, sep=";")
    df["measured_at"] = pd.to_datetime(df["measured_at"])
    df = df.sort_values(["group_id", "measured_at"]).reset_index(drop=True)
    
    # Get consumption scaling stats (for inverse transform)
    # Check if per-group scaling was used (scaled values have ~mean=0, ~std=1 per group)
    consumption_col = "consumption" if "consumption" in df.columns else "scaled_consumption"
    per_group_scaling = False
    group_scaling_stats = {}
    
    if "consumption" in df.columns and "group_id" in df.columns:
        # Check if per-group scaling was used by looking at scaled_consumption stats
        if "scaled_consumption" in df.columns:
            # Check each group's scaled mean/std
            for gid in df["group_id"].unique():
                gdf = df[df["group_id"] == gid]
                scaled_mean = gdf["scaled_consumption"].mean()
                scaled_std = gdf["scaled_consumption"].std()
                # If each group has ~mean=0, ~std=1, then per-group scaling was used
                if abs(scaled_mean) < 0.1 and abs(scaled_std - 1.0) < 0.1:
                    per_group_scaling = True
                    group_scaling_stats[gid] = {
                        "mean": gdf["consumption"].mean(),
                        "std": gdf["consumption"].std(),
                    }
        
        if per_group_scaling:
            print("[info] Detected per-group scaling, will use per-group inverse transform")
        else:
            # Global scaling
            cons_mean = df["consumption"].mean()
            cons_std = df["consumption"].std()
    else:
        # If only scaled, assume it was standardized (mean=0, std=1)
        cons_mean = 0.0
        cons_std = 1.0
        print("[warn] 'consumption' column not found, predictions will be in scaled units")
    
    # Reconstruct model
    hidden_size = config.get("hidden_size", 128)
    num_layers = config.get("layers", 2)
    dropout = config.get("dropout", 0.1)
    
    # Determine feature sizes from dataset (match training: include lag features if present)
    lag_cols = [c for c in df.columns if c.startswith("consumption_lag_")]
    dataset = HourlySeq2SeqDataset(
        df=df,
        window_size=window_size,
        horizon=horizon,
        extra_enc_cols=lag_cols,
        extra_dec_cols=[],
        min_samples_per_group=1,
    )
    
    # Build encoder input size
    enc_input_size = 1 + 1 + len(dataset.time_cols) + len(dataset.event_cols) + len(dataset.extra_enc_cols)  # consumption + price + time + events + extra
    dec_input_size = 1 + len(dataset.time_cols) + len(dataset.event_cols) + len(dataset.extra_dec_cols)  # price + time + events + extra (no consumption)
    
    model = Seq2Seq(
        enc_input_size=enc_input_size,
        dec_known_size=dec_input_size,
        hidden_size=hidden_size,
        num_layers=num_layers,
        dropout=dropout,
        num_groups=dataset.num_groups,
    ).to(device)
    
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    print(f"[model] loaded: hidden={hidden_size}, layers={num_layers}, groups={dataset.num_groups}")
    
    # Generate predictions for each group
    all_predictions = []
    
    for group_id in sorted(df["group_id"].unique()):
        gdf = df[df["group_id"] == group_id].sort_values("measured_at").reset_index(drop=True)
        
        if len(gdf) < window_size:
            print(f"[skip] group {group_id}: insufficient data ({len(gdf)} < {window_size})")
            continue
        
        # Get latest window
        last_time = gdf["measured_at"].iloc[-1]
        if args.start_at is not None:
            start_forecast = pd.to_datetime(args.start_at)
            # Find encoder window ending just before start_forecast
            before_forecast = gdf[gdf["measured_at"] < start_forecast]
            if len(before_forecast) < window_size:
                print(f"[skip] group {group_id}: insufficient data before start_forecast")
                continue
            latest_window = before_forecast.iloc[-window_size:].copy()
        elif args.offset_hours > 0:
            start_forecast = last_time - pd.Timedelta(hours=args.offset_hours) + pd.Timedelta(hours=1)
            # Find encoder window ending just before start_forecast
            before_forecast = gdf[gdf["measured_at"] < start_forecast]
            if len(before_forecast) < window_size:
                print(f"[skip] group {group_id}: insufficient data before start_forecast")
                continue
            latest_window = before_forecast.iloc[-window_size:].copy()
        else:
            start_forecast = last_time + pd.Timedelta(hours=1)
            latest_window = gdf.iloc[-window_size:].copy()
        
        # Build encoder input [1, window_size, enc_input_size]
        enc_features = []
        for _, row in latest_window.iterrows():
            feat = [
                row["scaled_consumption"],
                row["scaled_price"],
            ]
            feat.extend([row[col] for col in dataset.time_cols if col in row.index])
            feat.extend([row[col] for col in dataset.event_cols if col in row.index])
            feat.extend([row[col] for col in dataset.extra_enc_cols if col in row.index])
            enc_features.append(feat)
        
        enc_x = torch.tensor([enc_features], dtype=torch.float32).to(device)
        
        # Build decoder known covariates [1, horizon, dec_input_size]
        dec_known = build_decoder_known_covariates(
            gdf=gdf,
            start_time=start_forecast,
            horizon=horizon,
            time_cols=dataset.time_cols,
            event_cols=dataset.event_cols,
            price_col="scaled_price",
            extra_dec_cols=dataset.extra_dec_cols,
        ).unsqueeze(0).to(device)
        
        # Get group index
        group_idx = torch.tensor([dataset.group_to_idx[group_id]], dtype=torch.long).to(device)
        
        # Generate predictions
        with torch.no_grad():
            # Dummy y for shape (not used in eval mode)
            dummy_y = torch.zeros(1, horizon, 1, dtype=torch.float32).to(device)
            preds = model(enc_x, dec_known, dummy_y, group_idx, teacher_forcing_ratio=0.0)
            preds = preds.cpu().numpy()[0, :, 0]  # [horizon]
        
        # Inverse scale predictions (use per-group stats if available)
        if per_group_scaling and group_id in group_scaling_stats:
            g_mean = group_scaling_stats[group_id]["mean"]
            g_std = group_scaling_stats[group_id]["std"]
            preds_unscaled = preds * g_std + g_mean
        else:
            preds_unscaled = preds * cons_std + cons_mean
        
        # Create output rows
        forecast_times = pd.date_range(start=start_forecast, periods=horizon, freq="H")
        for i, ts in enumerate(forecast_times):
            all_predictions.append({
                "measured_at": ts,
                "group_id": group_id,
                "predicted_consumption": preds_unscaled[i],
                "predicted_consumption_scaled": preds[i],
            })
        
        print(f"[predict] group {group_id}: forecast from {start_forecast} (48 hours)")
    
    # Save predictions
    pred_df = pd.DataFrame(all_predictions)
    pred_df.to_csv(args.out, sep=";", index=False)
    print(f"[save] predictions saved to {args.out} ({len(pred_df)} rows)")


if __name__ == "__main__":
    main()

