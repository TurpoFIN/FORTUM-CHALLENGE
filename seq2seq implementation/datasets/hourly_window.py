from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


class HourlySeq2SeqDataset(Dataset):
    """
    Builds encoder/decoder windows from a long dataframe with columns at least:
      - measured_at (datetime-like)
      - group_id (int or str)
      - scaled_consumption (float)  -> target and autoregressive encoder feature
      - scaled_price (float)        -> known covariate (enc+dec)
      - hour_sin, hour_cos, day_of_week_sin, day_of_week_cos, month_sin, month_cos
      - is_holiday, is_weekend      -> known covariates
      - Optional: one-hot encoded categorical features (region_*, province_*, customer_type_*, etc.)

    For each group, creates samples with:
      - enc_x: [N, F_enc]  (includes scaled_consumption + categorical features)
      - dec_known: [H, F_dec] (no future scaled_consumption, includes categorical features)
      - y: [H, 1] scaled future consumption
      - group_idx: int id for embedding
      
    If auto_detect_categorical=True (default), automatically includes one-hot encoded
    categorical columns (e.g., province_Lapland, customer_type_Private) as static features.
    """

    def __init__(
        self,
        df: pd.DataFrame,
        window_size: int = 168,
        horizon: int = 48,
        time_cols: Optional[List[str]] = None,
        event_cols: Optional[List[str]] = None,
        extra_enc_cols: Optional[List[str]] = None,
        extra_dec_cols: Optional[List[str]] = None,
        group_col: str = "group_id",
        consumption_col: str = "scaled_consumption",
        price_col: str = "scaled_price",
        min_samples_per_group: int = 1,
        auto_detect_categorical: bool = True,
    ) -> None:
        df = df.copy()
        df["measured_at"] = pd.to_datetime(df["measured_at"])
        self.window_size = window_size
        self.horizon = horizon
        self.group_col = group_col
        self.consumption_col = consumption_col
        self.price_col = price_col

        self.time_cols = time_cols or [
            "hour_sin",
            "hour_cos",
            "day_of_week_sin",
            "day_of_week_cos",
            "month_sin",
            "month_cos",
        ]
        self.event_cols = event_cols or ["is_holiday", "is_weekend"]

        self.extra_enc_cols = extra_enc_cols or []  # e.g., lag features
        self.extra_dec_cols = extra_dec_cols or []  # usually empty unless you want special covariates
        
        # Auto-detect categorical one-hot encoded columns if requested
        if auto_detect_categorical:
            categorical_prefixes = ['region_', 'province_', 'municipality_', 'customer_type_', 'price_type_', 'consumption_level_']
            detected_categorical_cols = [
                col for col in df.columns 
                if any(col.startswith(prefix) for prefix in categorical_prefixes)
            ]
            if detected_categorical_cols:
                print(f"[dataset] Auto-detected {len(detected_categorical_cols)} categorical feature columns")
                print(f"[dataset]   Examples: {detected_categorical_cols[:5]}")
                # Add to both encoder and decoder (these are static features)
                self.extra_enc_cols = list(set(self.extra_enc_cols + detected_categorical_cols))
                self.extra_dec_cols = list(set(self.extra_dec_cols + detected_categorical_cols))

        # group index mapping for embedding
        groups = df[group_col].dropna().unique().tolist()
        self.group_to_idx: Dict = {g: i for i, g in enumerate(sorted(groups, key=lambda x: str(x)))}
        self.num_groups = len(self.group_to_idx)

        # Pre-build index of samples
        self.samples: List[Tuple[int, int, int]] = []  # list of (group_idx, start_idx_in_group, end_idx_in_group)
        self.group_frames: List[pd.DataFrame] = []
        self.group_ids_by_index: List[int] = []

        for g in sorted(groups, key=lambda x: str(x)):
            gdf = df[df[group_col] == g].sort_values("measured_at").reset_index(drop=True)
            if len(gdf) < (window_size + horizon + min_samples_per_group):
                continue
            start_max = len(gdf) - (window_size + horizon)
            if start_max < 0:
                continue
            self.group_frames.append(gdf)
            gidx = self.group_to_idx[g]
            for start in range(0, start_max + 1):
                self.samples.append((len(self.group_frames) - 1, start, start + window_size + horizon))
                self.group_ids_by_index.append(gidx)

        # Define column layouts
        self.enc_cols = [consumption_col, price_col] + self.time_cols + self.event_cols + self.extra_enc_cols
        # decoder known excludes future consumption
        self.dec_cols = [price_col] + self.time_cols + self.event_cols + self.extra_dec_cols

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        grp_frame_idx, start, end = self.samples[idx]
        gdf = self.group_frames[grp_frame_idx]
        gidx = self.group_ids_by_index[idx]

        enc_slice = gdf.iloc[start : start + self.window_size]
        dec_slice = gdf.iloc[start + self.window_size : start + self.window_size + self.horizon]

        enc_x = torch.tensor(enc_slice[self.enc_cols].values, dtype=torch.float32)  # [N, F_enc]
        dec_known = torch.tensor(dec_slice[self.dec_cols].values, dtype=torch.float32)  # [H, F_dec]
        y = torch.tensor(dec_slice[[self.consumption_col]].values, dtype=torch.float32)  # [H, 1]
        group_idx = torch.tensor(gidx, dtype=torch.long)

        return enc_x, dec_known, y, group_idx


