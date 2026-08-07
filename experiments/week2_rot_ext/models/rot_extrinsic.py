"""BiLSTM regressor for static mounting extrinsic R_SB (6D → SO(3))."""
from __future__ import annotations

import torch
import torch.nn as nn

from dataset import r6d_to_rotation_matrix


class RotExtrinsicNet(nn.Module):
    """
    Input:
      x:    [B, T, feat_dim]  (acc+ori, feat_dim=12)
      slot: [B] long in {0..n_slots-1}
    Output:
      r6d: [B, 6]
      R:   [B, 3, 3]
    """

    def __init__(
        self,
        feat_dim: int = 12,
        n_slots: int = 4,
        n_hidden: int = 128,
        n_lstm_layers: int = 2,
        bidirectional: bool = True,
        dropout: float = 0.3,
        use_slot_onehot: bool = True,
    ):
        super().__init__()
        self.use_slot_onehot = use_slot_onehot
        self.n_slots = n_slots
        in_dim = feat_dim + (n_slots if use_slot_onehot else 0)

        self.fc_in = nn.Sequential(
            nn.Linear(in_dim, n_hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )
        self.lstm = nn.LSTM(
            input_size=n_hidden,
            hidden_size=n_hidden,
            num_layers=n_lstm_layers,
            batch_first=True,
            bidirectional=bidirectional,
            dropout=dropout if n_lstm_layers > 1 else 0.0,
        )
        feat_out = n_hidden * (2 if bidirectional else 1)
        self.head = nn.Linear(feat_out, 6)

    def _augment_slot(self, x: torch.Tensor, slot: torch.Tensor) -> torch.Tensor:
        if not self.use_slot_onehot:
            return x
        b, t, _ = x.shape
        onehot = torch.zeros(b, self.n_slots, device=x.device, dtype=x.dtype)
        onehot.scatter_(1, slot.view(-1, 1), 1.0)
        onehot = onehot.unsqueeze(1).expand(-1, t, -1)
        return torch.cat([x, onehot], dim=-1)

    def forward(self, x: torch.Tensor, slot: torch.Tensor):
        x = self._augment_slot(x, slot)
        h = self.fc_in(x)
        seq, _ = self.lstm(h)
        feat = seq.mean(dim=1)
        r6d = self.head(feat)
        r = r6d_to_rotation_matrix(r6d)
        return r6d, r
