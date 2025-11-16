"""
Dataset for Seq2Seq with categorical embeddings.

Instead of one-hot encoding, this dataset provides categorical indices
that will be embedded by the model.
"""

from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


class HourlySeq2SeqDatasetWithEmbeddings(Dataset):
    """
    Builds encoder/decoder windows with categorical indices for embedding.
    
    Categorical features (province, customer_type, etc.) are passed as integer indices
    instead of one-hot vectors, allowing the model to learn dense embeddings.
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
        categorical_cols: Optional[Dict[str, str]] = None,  # e.g., {'province': 'province', 'customer_type': 'customer_type'}
    ) -> None:
        """
        Args:
            categorical_cols: Dict mapping feature name to column name in dataframe
                Example: {'province': 'province', 'customer_type': 'customer_type'}
                These columns should contain string/int category values, NOT one-hot encoded!
        """
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
        self.extra_enc_cols = extra_enc_cols or []
        self.extra_dec_cols = extra_dec_cols or []
        
        # Build categorical encoders
        self.categorical_cols = categorical_cols or {}
        self.categorical_encoders = {}  # Maps feature name -> {category_value: index}
        
        for feat_name, col_name in self.categorical_cols.items():
            if col_name in df.columns:
                unique_cats = sorted(df[col_name].dropna().unique().tolist(), key=str)
                self.categorical_encoders[feat_name] = {cat: idx for idx, cat in enumerate(unique_cats)}
                print(f"[dataset] Categorical feature '{feat_name}': {len(unique_cats)} categories")
            else:
                print(f"[warn] Categorical column '{col_name}' not found in dataframe")

        # group index mapping for embedding
        groups = df[group_col].dropna().unique().tolist()
        self.group_to_idx: Dict = {g: i for i, g in enumerate(sorted(groups, key=lambda x: str(x)))}
        self.num_groups = len(self.group_to_idx)

        # Pre-build index of samples
        self.samples: List[Tuple[int, int, int]] = []
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

        # Define column layouts (continuous features only, no categorical one-hot)
        self.enc_cols = [consumption_col, price_col] + self.time_cols + self.event_cols + self.extra_enc_cols
        self.dec_cols = [price_col] + self.time_cols + self.event_cols + self.extra_dec_cols

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        grp_frame_idx, start, end = self.samples[idx]
        gdf = self.group_frames[grp_frame_idx]
        gidx = self.group_ids_by_index[idx]

        enc_slice = gdf.iloc[start : start + self.window_size]
        dec_slice = gdf.iloc[start + self.window_size : start + self.window_size + self.horizon]

        # Continuous features
        enc_x = torch.tensor(enc_slice[self.enc_cols].values, dtype=torch.float32)
        dec_known = torch.tensor(dec_slice[self.dec_cols].values, dtype=torch.float32)
        y = torch.tensor(dec_slice[[self.consumption_col]].values, dtype=torch.float32)
        group_idx = torch.tensor(gidx, dtype=torch.long)

        # Categorical indices (static per sample)
        categorical_indices = {}
        for feat_name, col_name in self.categorical_cols.items():
            if col_name in gdf.columns and feat_name in self.categorical_encoders:
                # Get the category value (should be same across all rows for this group)
                cat_value = enc_slice[col_name].iloc[0]
                cat_idx = self.categorical_encoders[feat_name].get(cat_value, 0)  # Default to 0 if unknown
                categorical_indices[feat_name] = torch.tensor(cat_idx, dtype=torch.long)
        
        return enc_x, dec_known, y, group_idx, categorical_indices

    def get_categorical_config(self) -> Dict[str, Tuple[int, int]]:
        """
        Returns categorical configuration for model initialization.
        Maps feature name to (num_categories, recommended_embedding_dim)
        """
        config = {}
        for feat_name, encoder in self.categorical_encoders.items():
            num_cats = len(encoder)
            # Rule of thumb: embedding_dim = min(50, (num_cats + 1) // 2)
            emb_dim = min(50, max(2, (num_cats + 1) // 2))
            config[feat_name] = (num_cats, emb_dim)
        return config

