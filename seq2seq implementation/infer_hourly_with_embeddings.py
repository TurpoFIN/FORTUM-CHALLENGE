"""
Inference script for Seq2Seq model with categorical embeddings.

Generates predictions using a trained model that uses learned embeddings
instead of one-hot encoding.

Usage:
    python "seq2seq implementation/infer_hourly_with_embeddings.py" \
        --model artifacts/models_embeddings/seq2seq_hourly_embeddings.pt \
        --csv formatted_features.csv \
        --out predictions_embeddings.csv \
        --window-size 168 \
        --horizon 48
"""

import argparse
from pathlib import Path
from typing import Dict

import pandas as pd
import torch

try:
    from tqdm import tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False
    print("[info] tqdm not installed, progress bar disabled")

from models.seq2seq_lstm_with_embeddings import Seq2SeqWithEmbeddings


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate predictions using embedding-based Seq2Seq model.")
    p.add_argument("--model", type=str, required=True, help="Path to trained model (.pt file)")
    p.add_argument("--csv", type=str, required=True, help="Path to formatted features CSV")
    p.add_argument("--out", type=str, required=True, help="Output CSV path for predictions")
    p.add_argument("--window-size", type=int, default=168, help="Encoder window size (must match training)")
    p.add_argument("--horizon", type=int, default=48, help="Forecast horizon (must match training)")
    p.add_argument("--offset-hours", type=int, default=0, help="Offset from end of data (0 = predict from latest)")
    p.add_argument("--batch-size", type=int, default=32, help="Batch size for inference")
    return p.parse_args()


def load_model_and_config(model_path: Path, device: torch.device):
    """Load model checkpoint and configuration."""
    print(f"[load] Loading model from {model_path}")
    checkpoint = torch.load(model_path, map_location=device)
    
    config = checkpoint.get("config", {})
    categorical_config = checkpoint.get("categorical_config", {})
    categorical_encoders = checkpoint.get("categorical_encoders", {})
    
    print(f"[config] Model configuration:")
    print(f"  Hidden size: {config.get('hidden_size', 'N/A')}")
    print(f"  Layers: {config.get('layers', 'N/A')}")
    print(f"  Group embedding dim: {config.get('group_emb_dim', 16)}")
    
    if categorical_config:
        print(f"[categorical] Using embeddings for:")
        for feat_name, (num_cats, emb_dim) in categorical_config.items():
            print(f"  {feat_name}: {num_cats} categories → {emb_dim}-dim embedding")
    
    return checkpoint, config, categorical_config, categorical_encoders


def prepare_categorical_indices(df_row: pd.Series, categorical_encoders: Dict) -> Dict[str, int]:
    """Convert categorical values to indices for embedding lookup."""
    cat_indices = {}
    for feat_name, encoder in categorical_encoders.items():
        # Map from feature name to column name
        # Assuming feature name matches column name
        if feat_name in df_row.index:
            cat_value = df_row[feat_name]
            cat_idx = encoder.get(cat_value, 0)  # Default to 0 if unknown
            cat_indices[feat_name] = cat_idx
    return cat_indices


def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[device] using {device}")
    
    # Load model
    model_path = Path(args.model)
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")
    
    checkpoint, config, categorical_config, categorical_encoders = load_model_and_config(model_path, device)
    
    # Load data
    print(f"[data] Loading {args.csv}")
    df = pd.read_csv(args.csv, sep=";")
    df["measured_at"] = pd.to_datetime(df["measured_at"])
    print(f"[data] Loaded {len(df)} rows")
    
    # Define feature columns (continuous only)
    time_cols = ["hour_sin", "hour_cos", "day_of_week_sin", "day_of_week_cos", "month_sin", "month_cos"]
    event_cols = ["is_holiday", "is_weekend"]
    lag_cols = [col for col in df.columns if 'lag' in col.lower()]
    
    enc_cols = ["scaled_consumption", "scaled_price"] + time_cols + event_cols + lag_cols
    dec_cols = ["scaled_price"] + time_cols + event_cols
    
    # Verify all required columns exist
    missing_cols = [col for col in enc_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing columns in CSV: {missing_cols}")
    
    # Build group mapping
    groups = sorted(df["group_id"].unique().tolist(), key=lambda x: str(x))
    group_to_idx = {g: i for i, g in enumerate(groups)}
    num_groups = len(group_to_idx)
    
    print(f"[data] Found {num_groups} groups")
    
    # Reconstruct model
    sample_enc_size = len(enc_cols)
    sample_dec_size = len(dec_cols)
    
    model = Seq2SeqWithEmbeddings(
        enc_input_size=sample_enc_size,
        dec_known_size=sample_dec_size,
        hidden_size=config.get("hidden_size", 128),
        num_layers=config.get("layers", 2),
        dropout=config.get("dropout", 0.1),
        use_group_embedding=True,
        num_groups=num_groups,
        group_emb_dim=config.get("group_emb_dim", 16),
        categorical_config=categorical_config if categorical_config else None,
    ).to(device)
    
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    
    print(f"[model] Model loaded successfully")
    
    # Generate predictions
    predictions = []
    
    print(f"[predict] Generating predictions for {len(groups)} groups...")
    
    with torch.no_grad():
        group_iter = tqdm(groups, desc="Generating predictions") if HAS_TQDM else groups
        for idx, group_id in enumerate(group_iter):
            if not HAS_TQDM and idx % 10 == 0:
                print(f"  Processing group {idx+1}/{len(groups)} (ID: {group_id})")
            gdf = df[df["group_id"] == group_id].sort_values("measured_at").reset_index(drop=True)
            
            if len(gdf) < args.window_size:
                print(f"[warn] Group {group_id} has only {len(gdf)} rows (need {args.window_size}), skipping")
                continue
            
            # Determine start index
            if args.offset_hours > 0:
                start_idx = max(0, len(gdf) - args.window_size - args.offset_hours)
            else:
                start_idx = len(gdf) - args.window_size
            
            if start_idx < 0:
                continue
            
            # Extract encoder window
            enc_slice = gdf.iloc[start_idx : start_idx + args.window_size]
            
            # Get categorical indices (static for this group)
            cat_indices_dict = {}
            if categorical_encoders:
                first_row = enc_slice.iloc[0]
                for feat_name, encoder in categorical_encoders.items():
                    # Try to find the column
                    if feat_name in first_row.index:
                        cat_value = first_row[feat_name]
                        cat_idx = encoder.get(cat_value, 0)
                        cat_indices_dict[feat_name] = torch.tensor(cat_idx, dtype=torch.long).to(device)
            
            # Prepare decoder known features (future time/event features from end of encoder window)
            last_timestamp = enc_slice["measured_at"].iloc[-1]
            future_timestamps = pd.date_range(
                start=last_timestamp + pd.Timedelta(hours=1),
                periods=args.horizon,
                freq="H"
            )
            
            # Create future features
            dec_data = []
            for ts in future_timestamps:
                row = {
                    "measured_at": ts,
                    "hour_sin": torch.sin(torch.tensor(2 * 3.14159 * ts.hour / 24.0)),
                    "hour_cos": torch.cos(torch.tensor(2 * 3.14159 * ts.hour / 24.0)),
                    "day_of_week_sin": torch.sin(torch.tensor(2 * 3.14159 * ts.dayofweek / 7.0)),
                    "day_of_week_cos": torch.cos(torch.tensor(2 * 3.14159 * ts.dayofweek / 7.0)),
                    "month_sin": torch.sin(torch.tensor(2 * 3.14159 * ts.month / 12.0)),
                    "month_cos": torch.cos(torch.tensor(2 * 3.14159 * ts.month / 12.0)),
                    "is_holiday": 0,  # Would need holiday calendar
                    "is_weekend": 1 if ts.dayofweek >= 5 else 0,
                }
                
                # Use last known price (or 0 if unknown)
                row["scaled_price"] = enc_slice["scaled_price"].iloc[-1] if "scaled_price" in enc_slice.columns else 0.0
                
                dec_data.append([row[col] for col in dec_cols])
            
            # Prepare tensors
            enc_x = torch.tensor(enc_slice[enc_cols].values, dtype=torch.float32).unsqueeze(0).to(device)  # [1, N, F_enc]
            dec_known = torch.tensor(dec_data, dtype=torch.float32).unsqueeze(0).to(device)  # [1, H, F_dec]
            dummy_y = torch.zeros(1, args.horizon, 1, dtype=torch.float32).to(device)  # Not used in inference
            group_idx = torch.tensor([group_to_idx[group_id]], dtype=torch.long).to(device)  # [1]
            
            # Prepare categorical indices for batch (add batch dimension)
            cat_indices_batch = {k: v.unsqueeze(0) for k, v in cat_indices_dict.items()} if cat_indices_dict else {}
            
            # Generate predictions
            preds = model(
                enc_x,
                dec_known,
                dummy_y,
                group_ids=group_idx,
                categorical_indices=cat_indices_batch if cat_indices_batch else None,
                teacher_forcing_ratio=0.0,  # Pure inference
            )  # [1, H, 1]
            
            # Convert to dataframe
            preds_np = preds.squeeze().cpu().numpy()  # [H]
            
            for i, ts in enumerate(future_timestamps):
                predictions.append({
                    "measured_at": ts,
                    "group_id": group_id,
                    "predicted_consumption_scaled": preds_np[i],
                })
    
    # Save predictions
    print(f"[save] Generated {len(predictions)} predictions")
    pred_df = pd.DataFrame(predictions)
    
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pred_df.to_csv(out_path, sep=";", index=False)
    
    print(f"[done] Predictions saved to {out_path}")
    print(f"[info] Time range: {pred_df['measured_at'].min()} to {pred_df['measured_at'].max()}")
    print(f"[info] Groups: {pred_df['group_id'].nunique()}")


if __name__ == "__main__":
    main()

