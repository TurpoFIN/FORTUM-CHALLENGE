"""
Hybrid GRU + FNN model for 48-hour forecasting.

Architecture:
- GRU Encoder: Processes 168-hour historical window to extract temporal features
- FNN Decoder: Takes encoder output + known future covariates, predicts all 48 hours at once

Benefits:
- Faster inference (single forward pass, no autoregressive loop)
- Captures temporal patterns via GRU
- Can leverage all known future covariates simultaneously
"""

import torch
import torch.nn as nn
from typing import Optional


class GRUEncoder(nn.Module):
    """GRU encoder for processing historical sequence."""
    
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_layers: int = 2,
        dropout: float = 0.1,
        bidirectional: bool = False,
    ) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        
        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            batch_first=True,
            bidirectional=bidirectional,
        )
    
    def forward(self, x: torch.Tensor, hidden: Optional[torch.Tensor] = None) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: [B, N, F] input sequence
            hidden: Optional initial hidden state
        
        Returns:
            outputs: [B, N, D] GRU outputs
            hidden: [L*(2 if bi), B, H] final hidden state
        """
        outputs, hidden = self.gru(x, hidden)
        
        if self.bidirectional:
            # Concatenate forward and backward hidden states
            num_directions = 2
            hidden = hidden.view(self.num_layers, num_directions, x.size(0), self.hidden_size)
            # Take last layer's forward and backward states
            hidden = torch.cat([hidden[-1, 0, :, :], hidden[-1, 1, :, :]], dim=-1)  # [B, 2H]
        else:
            # Take last layer's hidden state
            hidden = hidden[-1]  # [B, H]
        
        return outputs, hidden


class FNNDecoder(nn.Module):
    """Feedforward decoder that predicts all 48 hours at once."""
    
    def __init__(
        self,
        encoder_hidden_size: int,
        dec_covariate_size: int,
        horizon: int = 48,
        hidden_dims: list[int] = [256, 128],
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.horizon = horizon
        
        # Input: encoder hidden + known covariates for each future step
        # We'll flatten all known covariates for all 48 steps
        input_size = encoder_hidden_size + (dec_covariate_size * horizon)
        
        layers = []
        prev_dim = input_size
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            prev_dim = hidden_dim
        
        # Output: 48 predictions
        layers.append(nn.Linear(prev_dim, horizon))
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, encoder_hidden: torch.Tensor, dec_covariates: torch.Tensor) -> torch.Tensor:
        """
        Args:
            encoder_hidden: [B, H] encoder's final hidden state
            dec_covariates: [B, horizon, F_dec] known future covariates
        
        Returns:
            predictions: [B, horizon] predicted consumption for all 48 hours
        """
        B = encoder_hidden.size(0)
        
        # Flatten decoder covariates: [B, horizon, F_dec] -> [B, horizon * F_dec]
        dec_flat = dec_covariates.view(B, -1)
        
        # Concatenate encoder hidden with flattened covariates
        combined = torch.cat([encoder_hidden, dec_flat], dim=1)  # [B, H + horizon*F_dec]
        
        # Predict all 48 hours at once
        predictions = self.network(combined)  # [B, horizon]
        
        # Reshape to [B, horizon, 1] for consistency with seq2seq interface
        return predictions.unsqueeze(-1)


class GRUFNNHybrid(nn.Module):
    """Hybrid GRU encoder + FNN decoder model."""
    
    def __init__(
        self,
        enc_input_size: int,
        dec_covariate_size: int,
        hidden_size: int = 128,
        num_layers: int = 2,
        dropout: float = 0.1,
        horizon: int = 48,
        decoder_hidden_dims: list[int] = [256, 128],
        num_groups: int = 1,
        group_embedding_dim: int = 16,
    ) -> None:
        super().__init__()
        self.horizon = horizon
        
        # Group embedding (optional)
        if num_groups > 1:
            self.group_embedding = nn.Embedding(num_groups, group_embedding_dim)
            encoder_hidden_size = hidden_size + group_embedding_dim
        else:
            self.group_embedding = None
            encoder_hidden_size = hidden_size
        
        # GRU Encoder
        self.encoder = GRUEncoder(
            input_size=enc_input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
            bidirectional=False,
        )
        
        # FNN Decoder
        self.decoder = FNNDecoder(
            encoder_hidden_size=encoder_hidden_size,
            dec_covariate_size=dec_covariate_size,
            horizon=horizon,
            hidden_dims=decoder_hidden_dims,
            dropout=dropout,
        )
    
    def forward(
        self,
        enc_x: torch.Tensor,
        dec_covariates: torch.Tensor,
        group_idx: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            enc_x: [B, N, F_enc] encoder input (historical sequence)
            dec_covariates: [B, horizon, F_dec] known future covariates
            group_idx: [B] optional group indices for embedding
        
        Returns:
            predictions: [B, horizon, 1] predicted consumption
        """
        # Encode historical sequence
        _, encoder_hidden = self.encoder(enc_x)  # encoder_hidden: [B, H]
        
        # Add group embedding if provided
        if self.group_embedding is not None and group_idx is not None:
            group_emb = self.group_embedding(group_idx)  # [B, E]
            encoder_hidden = torch.cat([encoder_hidden, group_emb], dim=-1)  # [B, H+E]
        
        # Decode to predictions
        predictions = self.decoder(encoder_hidden, dec_covariates)  # [B, horizon, 1]
        
        return predictions

