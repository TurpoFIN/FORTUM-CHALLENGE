"""
Normalize scaled predictions to original units using per-group mean/std from truth.

Usage (Windows PowerShell one-liner):
  py normalize_predictions.py --pred predictions_embeddings_eval.csv --truth formatted_features.csv --out predictions_normalized.csv
"""

import argparse
from pathlib import Path

import pandas as pd
import numpy as np


def normalize_predictions_csv(pred_path: str, truth_path: str, out_path: str) -> None:
	# Read files
	pred = pd.read_csv(pred_path, sep=";")
	truth = pd.read_csv(truth_path, sep=";")

	# Basic checks
	required_pred_cols = {"measured_at", "group_id", "predicted_consumption_scaled"}
	missing_pred = required_pred_cols - set(pred.columns)
	if missing_pred:
		raise ValueError(f"Predictions missing columns: {sorted(missing_pred)}")

	required_truth_cols = {"group_id", "consumption", "scaled_consumption"}
	missing_truth = required_truth_cols - set(truth.columns)
	if missing_truth:
		raise ValueError(f"Truth missing columns: {sorted(missing_truth)} (needed for inverse scaling)")

	# Compute per-group stats on truth (use all rows for stability)
	stats = (
		truth.groupby("group_id")
		.agg(
			mean_s=("scaled_consumption", "mean"),
			std_s=("scaled_consumption", "std"),
			mean_y=("consumption", "mean"),
			std_y=("consumption", "std"),
		)
		.reset_index()
	)

	# Join stats to predictions by group_id
	df = pred.merge(stats, on="group_id", how="left", validate="m:1")

	# Handle groups without stats
	if df[["mean_s", "std_s", "mean_y", "std_y"]].isna().any().any():
		missing_groups = df.loc[df[["mean_s", "std_s", "mean_y", "std_y"]].isna().any(axis=1), "group_id"].unique()
		raise ValueError(f"No stats found in truth for groups: {sorted(map(int, missing_groups))}")

	# Compute linear inverse scaling per group: y ≈ a*ys + b, where a = std_y/std_s, b = mean_y - a*mean_s
	# Guard against std_s == 0
	std_s_safe = df["std_s"].replace(0, np.nan)
	a = (df["std_y"] / std_s_safe).fillna(1.0)
	b = df["mean_y"] - a * df["mean_s"]

	df["predicted_consumption"] = a.values * df["predicted_consumption_scaled"].values + b.values

	# Select and write output
	out_cols = ["measured_at", "group_id", "predicted_consumption", "predicted_consumption_scaled"]
	df[out_cols].to_csv(out_path, sep=";", index=False)
	print(f"[done] Wrote normalized predictions to {out_path}")


def parse_args() -> argparse.Namespace:
	p = argparse.ArgumentParser(description="Normalize scaled predictions to original units per group.")
	p.add_argument("--pred", type=str, required=True, help="Path to predictions CSV with 'predicted_consumption_scaled'")
	p.add_argument("--truth", type=str, required=True, help="Path to formatted_features.csv (must contain 'consumption' and 'scaled_consumption')")
	p.add_argument("--out", type=str, required=True, help="Path to write normalized predictions CSV")
	return p.parse_args()


def main() -> None:
	args = parse_args()
	normalize_predictions_csv(args.pred, args.truth, args.out)


if __name__ == "__main__":
	main()


