"""Week2 dual-head R_SB regressor, reused by Week3 step 2.

Architecture and checkpoint are identical to
experiments/week2_rot_ext/models/rot_extrinsic_dual.py.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from dataset import r6d_to_rotation_matrix


class RotExtrinsicDualNet(nn.Module):
    """
    Input:
      x:          [B, T, 24]  watch(12)+phone(12)
      slot_watch: [B] in {0,1} (LW/RW absolute)
      slot_phone: [B] in {2,3} (LP/RP absolute)
    Output:
      (r6d_w, R_w), (r6d_p, R_p)
    """

    def __init__(
        self,
        feat_dim: int = 24,
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
        in_dim = feat_dim + (2 * n_slots if use_slot_onehot else 0)

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
        self.head_watch = nn.Linear(feat_out, 6)
        self.head_phone = nn.Linear(feat_out, 6)

    def _augment_slots(
        self,
        x: torch.Tensor,
        slot_watch: torch.Tensor,
        slot_phone: torch.Tensor,
    ) -> torch.Tensor:
        if not self.use_slot_onehot:
            return x
        b, t, _ = x.shape
        oh_w = torch.zeros(b, self.n_slots, device=x.device, dtype=x.dtype)
        oh_p = torch.zeros(b, self.n_slots, device=x.device, dtype=x.dtype)
        oh_w.scatter_(1, slot_watch.view(-1, 1), 1.0)
        oh_p.scatter_(1, slot_phone.view(-1, 1), 1.0)
        cond = torch.cat([oh_w, oh_p], dim=-1).unsqueeze(1).expand(-1, t, -1)
        return torch.cat([x, cond], dim=-1)

    def forward(
        self,
        x: torch.Tensor,
        slot_watch: torch.Tensor,
        slot_phone: torch.Tensor,
    ):
        x = self._augment_slots(x, slot_watch, slot_phone)
        h = self.fc_in(x)
        seq, _ = self.lstm(h)
        feat = seq.mean(dim=1)
        r6d_w = self.head_watch(feat)
        r6d_p = self.head_phone(feat)
        return (r6d_w, r6d_to_rotation_matrix(r6d_w)), (
            r6d_p,
            r6d_to_rotation_matrix(r6d_p),
        )
