"""
Enhanced Seq2Seq LSTM with categorical embeddings.

Instead of one-hot encoding provinces/regions/customer_types (70+ features),
this model learns dense embeddings (10-20 features total) which is more efficient
and captures semantic relationships between categories.
"""

import torch
import torch.nn as nn
from typing import Optional, Tuple, Dict


class CategoricalEmbedder(nn.Module):
    """
    Learns embeddings for multiple categorical features and concatenates them.
    """
    def __init__(self, categorical_config: Dict[str, Tuple[int, int]]):
        """
        Args:
            categorical_config: Dict mapping feature name to (num_categories, embedding_dim)
                Example: {
                    'province': (13, 4),
                    'customer_type': (2, 2),
                    'price_type': (5, 3),
                    'consumption_level': (3, 2)
                }
        """
        super().__init__()
        self.embeddings = nn.ModuleDict()
        self.embedding_dims = {}
        
        for feat_name, (num_cats, emb_dim) in categorical_config.items():
            self.embeddings[feat_name] = nn.Embedding(num_cats, emb_dim)
            self.embedding_dims[feat_name] = emb_dim
        
        self.total_dim = sum(self.embedding_dims.values())
        
    def forward(self, categorical_indices: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Args:
            categorical_indices: Dict of tensors with shape [B] or [B, T] (for sequences)
                Each tensor contains indices for that categorical feature
        
        Returns:
            Concatenated embeddings of shape [B, total_dim] or [B, T, total_dim]
        """
        embedded = []
        for feat_name, emb_layer in self.embeddings.items():
            if feat_name in categorical_indices:
                embedded.append(emb_layer(categorical_indices[feat_name]))
        
        if not embedded:
            # No categorical features provided
            return None
        
        return torch.cat(embedded, dim=-1)


class Encoder(nn.Module):
    def __init__(
        self,
        input_size: int,  # Size of continuous features only
        hidden_size: int,
        num_layers: int = 2,
        dropout: float = 0.1,
        bidirectional: bool = False,
        categorical_config: Optional[Dict[str, Tuple[int, int]]] = None,
    ) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        
        # Categorical embedder
        self.cat_embedder = None
        if categorical_config:
            self.cat_embedder = CategoricalEmbedder(categorical_config)
            input_size += self.cat_embedder.total_dim
        
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            batch_first=True,
            bidirectional=bidirectional,
        )

    def forward(
        self, 
        x: torch.Tensor,  # [B, N, F_continuous]
        categorical_indices: Optional[Dict[str, torch.Tensor]] = None,
        hidden: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        # Embed categorical features if provided
        if self.cat_embedder is not None and categorical_indices is not None:
            cat_emb = self.cat_embedder(categorical_indices)  # [B, N, E_total] or [B, E_total]
            # Expand to sequence length if needed
            if cat_emb.dim() == 2:  # [B, E_total] - static features
                cat_emb = cat_emb.unsqueeze(1).expand(-1, x.size(1), -1)  # [B, N, E_total]
            x = torch.cat([x, cat_emb], dim=-1)  # [B, N, F_continuous + E_total]
        
        outputs, (h, c) = self.lstm(x, hidden)
        
        if self.bidirectional:
            num_directions = 2
            h = h.view(self.lstm.num_layers, num_directions, x.size(0), self.hidden_size)
            c = c.view(self.lstm.num_layers, num_directions, x.size(0), self.hidden_size)
            h = torch.cat([h[:, 0, :, :], h[:, 1, :, :]], dim=-1)
            c = torch.cat([c[:, 0, :, :], c[:, 1, :, :]], dim=-1)
        
        return outputs, (h, c)


class Decoder(nn.Module):
    def __init__(
        self,
        dec_input_size: int,  # Size of continuous decoder features only
        hidden_size: int,
        out_size: int = 1,
        num_layers: int = 2,
        dropout: float = 0.1,
        use_group_embedding: bool = True,
        num_groups: int = 0,
        group_emb_dim: int = 16,
        encoder_bidirectional: bool = False,
        categorical_config: Optional[Dict[str, Tuple[int, int]]] = None,
    ) -> None:
        super().__init__()
        self.use_group_embedding = use_group_embedding and num_groups > 0
        self.group_emb = nn.Embedding(num_groups, group_emb_dim) if self.use_group_embedding else None
        self.hidden_size = hidden_size * (2 if encoder_bidirectional else 1)

        # Categorical embedder
        self.cat_embedder = None
        if categorical_config:
            self.cat_embedder = CategoricalEmbedder(categorical_config)
            dec_input_size += self.cat_embedder.total_dim

        lstm_input_size = dec_input_size + 1  # previous y + known decoder covariates + categorical embeddings
        if self.use_group_embedding:
            lstm_input_size += group_emb_dim

        self.lstm = nn.LSTM(
            input_size=lstm_input_size,
            hidden_size=self.hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            batch_first=True,
        )
        self.proj = nn.Linear(self.hidden_size, out_size)

    def forward(
        self,
        dec_known: torch.Tensor,  # [B, H, F_dec_continuous]
        target_y: torch.Tensor,   # [B, H, 1]
        enc_hidden: Tuple[torch.Tensor, torch.Tensor],
        group_ids: Optional[torch.Tensor] = None,  # [B]
        categorical_indices: Optional[Dict[str, torch.Tensor]] = None,
        teacher_forcing_ratio: float = 0.5,
        last_enc_consumption: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        batch_size, horizon, _ = dec_known.size()
        device = dec_known.device

        # Embed categorical features once (they're static)
        cat_emb = None
        if self.cat_embedder is not None and categorical_indices is not None:
            cat_emb = self.cat_embedder(categorical_indices)  # [B, E_total]
            if cat_emb.dim() == 3:  # Already has time dimension, take first timestep
                cat_emb = cat_emb[:, 0, :]

        outputs = []
        h, c = enc_hidden

        # Initialize with last encoder consumption
        if last_enc_consumption is not None:
            prev_y = last_enc_consumption  # [B, 1]
        else:
            prev_y = target_y[:, 0, :]

        if self.use_group_embedding and group_ids is not None:
            group_vec = self.group_emb(group_ids)  # [B, E_group]
        else:
            group_vec = None

        for t in range(horizon):
            known_t = dec_known[:, t, :]  # [B, F_dec_continuous]
            
            # Concatenate: prev_y + continuous features + categorical embeddings + group embedding
            inputs_to_concat = [prev_y, known_t]
            if cat_emb is not None:
                inputs_to_concat.append(cat_emb)
            if self.use_group_embedding and group_vec is not None:
                inputs_to_concat.append(group_vec)
            
            dec_in = torch.cat(inputs_to_concat, dim=-1).unsqueeze(1)  # [B, 1, F_total]

            out, (h, c) = self.lstm(dec_in, (h, c))
            step_pred = self.proj(out[:, -1, :]).unsqueeze(-1)  # [B, 1, 1]
            outputs.append(step_pred)

            use_teacher = torch.rand(1, device=device).item() < teacher_forcing_ratio
            prev_y = target_y[:, t, :] if use_teacher else step_pred.squeeze(-1)

        return torch.cat(outputs, dim=1)  # [B, H, 1]


class Seq2SeqWithEmbeddings(nn.Module):
    """
    Seq2Seq model with learned embeddings for categorical features.
    
    More efficient than one-hot encoding:
    - province (13 categories): 13 binary features → 4-dim embedding
    - customer_type (2): 2 binary → 2-dim embedding
    - price_type (5): 5 binary → 3-dim embedding  
    - consumption_level (3): 3 binary → 2-dim embedding
    
    Total: 23 binary features → 11 dense features (52% reduction!)
    """
    def __init__(
        self,
        enc_input_size: int,  # Size of continuous encoder features
        dec_known_size: int,  # Size of continuous decoder features
        hidden_size: int = 128,
        num_layers: int = 2,
        dropout: float = 0.1,
        use_group_embedding: bool = True,
        num_groups: int = 0,
        group_emb_dim: int = 16,
        encoder_bidirectional: bool = False,
        categorical_config: Optional[Dict[str, Tuple[int, int]]] = None,
    ) -> None:
        """
        Args:
            categorical_config: Dict mapping categorical feature names to (num_categories, embedding_dim)
                If None, falls back to standard behavior (treats all features as continuous)
        """
        super().__init__()
        
        self.categorical_config = categorical_config
        
        self.encoder = Encoder(
            input_size=enc_input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
            bidirectional=encoder_bidirectional,
            categorical_config=categorical_config,
        )
        self.decoder = Decoder(
            dec_input_size=dec_known_size,
            hidden_size=hidden_size,
            out_size=1,
            num_layers=num_layers,
            dropout=dropout,
            use_group_embedding=use_group_embedding,
            num_groups=num_groups,
            group_emb_dim=group_emb_dim,
            encoder_bidirectional=encoder_bidirectional,
            categorical_config=categorical_config,
        )

    def forward(
        self,
        enc_x: torch.Tensor,       # [B, N, F_enc_continuous]
        dec_known: torch.Tensor,   # [B, H, F_dec_continuous]
        target_y: torch.Tensor,    # [B, H, 1]
        group_ids: Optional[torch.Tensor] = None,  # [B]
        categorical_indices: Optional[Dict[str, torch.Tensor]] = None,  # Dict of [B] tensors
        teacher_forcing_ratio: float = 0.5,
    ) -> torch.Tensor:
        _, (h, c) = self.encoder(enc_x, categorical_indices=categorical_indices)
        
        # Extract last encoder consumption
        last_enc_consumption = enc_x[:, -1:, 0:1].contiguous().squeeze(1)  # [B, 1]
        
        preds = self.decoder(
            dec_known=dec_known,
            target_y=target_y,
            enc_hidden=(h, c),
            group_ids=group_ids,
            categorical_indices=categorical_indices,
            teacher_forcing_ratio=teacher_forcing_ratio,
            last_enc_consumption=last_enc_consumption,
        )
        return preds

