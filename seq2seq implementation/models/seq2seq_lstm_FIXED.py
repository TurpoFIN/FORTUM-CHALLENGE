import math
from typing import Optional, Tuple

import torch
import torch.nn as nn


class Encoder(nn.Module):
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
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            batch_first=True,
            bidirectional=bidirectional,
        )

    def forward(self, x: torch.Tensor, hidden: Optional[Tuple[torch.Tensor, torch.Tensor]] = None) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        # x: [B, N, F]
        outputs, (h, c) = self.lstm(x, hidden)  # outputs: [B, N, D], h/c: [L*(2 if bi), B, H]
        if self.bidirectional:
            # concatenate last forward and backward states
            # reshape to [num_layers, num_directions, batch, hidden]
            num_directions = 2
            h = h.view(self.lstm.num_layers, num_directions, x.size(0), self.hidden_size)
            c = c.view(self.lstm.num_layers, num_directions, x.size(0), self.hidden_size)
            h = torch.cat([h[:, 0, :, :], h[:, 1, :, :]], dim=-1)  # [L, B, 2H]
            c = torch.cat([c[:, 0, :, :], c[:, 1, :, :]], dim=-1)  # [L, B, 2H]
        return outputs, (h, c)


class Decoder(nn.Module):
    def __init__(
        self,
        dec_input_size: int,
        hidden_size: int,
        out_size: int = 1,
        num_layers: int = 2,
        dropout: float = 0.1,
        use_group_embedding: bool = True,
        num_groups: int = 0,
        group_emb_dim: int = 16,
        encoder_bidirectional: bool = False,
    ) -> None:
        super().__init__()
        self.use_group_embedding = use_group_embedding and num_groups > 0
        self.group_emb = nn.Embedding(num_groups, group_emb_dim) if self.use_group_embedding else None
        self.hidden_size = hidden_size * (2 if encoder_bidirectional else 1)

        lstm_input_size = dec_input_size + 1  # previous y + known decoder covariates
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
        dec_known: torch.Tensor,  # [B, H, F_dec]
        target_y: torch.Tensor,   # [B, H, 1] (for teacher forcing)
        enc_hidden: Tuple[torch.Tensor, torch.Tensor],
        group_ids: Optional[torch.Tensor] = None,  # [B]
        teacher_forcing_ratio: float = 0.5,
        last_enc_consumption: Optional[torch.Tensor] = None,  # [B, 1] - last encoder consumption
    ) -> torch.Tensor:
        batch_size, horizon, _ = dec_known.size()
        device = dec_known.device

        outputs = []
        h, c = enc_hidden  # shapes [L, B, H_enc]

        # FIXED: Initialize with last encoder consumption for training-inference consistency
        # During training: use last encoder value (proper autoregressive conditioning)
        # During inference: use last encoder value (same distribution as training)
        if last_enc_consumption is not None:
            prev_y = last_enc_consumption  # [B, 1]
        else:
            # Fallback to old behavior if not provided (for backwards compatibility)
            prev_y = target_y[:, 0, :]  # [B, 1]

        if self.use_group_embedding and group_ids is not None:
            group_vec = self.group_emb(group_ids)  # [B, E]
        else:
            group_vec = None

        for t in range(horizon):
            known_t = dec_known[:, t, :]  # [B, F_dec]
            if self.use_group_embedding and group_vec is not None:
                dec_in = torch.cat([prev_y, known_t, group_vec], dim=-1).unsqueeze(1)  # [B,1,F]
            else:
                dec_in = torch.cat([prev_y, known_t], dim=-1).unsqueeze(1)

            out, (h, c) = self.lstm(dec_in, (h, c))  # out: [B,1,H]
            step_pred = self.proj(out[:, -1, :]).unsqueeze(-1)  # [B,1,1]
            outputs.append(step_pred)

            use_teacher = torch.rand(1, device=device).item() < teacher_forcing_ratio
            prev_y = target_y[:, t, :] if use_teacher else step_pred.squeeze(-1)  # [B,1] or [B,1]

        return torch.cat(outputs, dim=1)  # [B, H, 1]


class Seq2Seq(nn.Module):
    def __init__(
        self,
        enc_input_size: int,
        dec_known_size: int,
        hidden_size: int = 128,
        num_layers: int = 2,
        dropout: float = 0.1,
        use_group_embedding: bool = True,
        num_groups: int = 0,
        group_emb_dim: int = 16,
        encoder_bidirectional: bool = False,
    ) -> None:
        super().__init__()
        self.encoder = Encoder(
            input_size=enc_input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
            bidirectional=encoder_bidirectional,
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
        )

    def forward(
        self,
        enc_x: torch.Tensor,       # [B, N, F_enc]
        dec_known: torch.Tensor,   # [B, H, F_dec]
        target_y: torch.Tensor,    # [B, H, 1]
        group_ids: Optional[torch.Tensor] = None,  # [B]
        teacher_forcing_ratio: float = 0.5,
    ) -> torch.Tensor:
        _, (h, c) = self.encoder(enc_x)  # h,c: [L, B, H_enc]
        
        # FIXED: Extract last encoder consumption (first feature at last timestep)
        # Assuming consumption is the first feature in enc_x
        last_enc_consumption = enc_x[:, -1, 0:1]  # [B, 1]
        
        preds = self.decoder(
            dec_known=dec_known,
            target_y=target_y,
            enc_hidden=(h, c),
            group_ids=group_ids,
            teacher_forcing_ratio=teacher_forcing_ratio,
            last_enc_consumption=last_enc_consumption,
        )
        return preds


