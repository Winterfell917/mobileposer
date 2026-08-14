"""BiLSTM dual-head classifier for watch/phone side prediction."""
from __future__ import annotations

import torch
import torch.nn as nn


class PosClassifier(nn.Module):
    """
    Input:  [B, T, 24]  watch/phone a_M(3)+R_MS(9)
    Output: watch_logits [B, 2], phone_logits [B, 2]
    """

    def __init__(
        self,
        n_input: int = 12,
        n_hidden: int = 128,
        n_lstm_layers: int = 2,
        bidirectional: bool = True,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.fc_in = nn.Sequential(
            nn.Linear(n_input, n_hidden),
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
        feat_dim = n_hidden * (2 if bidirectional else 1)
        self.head_watch = nn.Linear(feat_dim, 2)
        self.head_phone = nn.Linear(feat_dim, 2)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        h = self.fc_in(x)
        seq, _ = self.lstm(h)
        # mean pool over time (online-friendly alternative: last frame)
        return seq.mean(dim=1)

    def forward(self, x: torch.Tensor):
        feat = self.encode(x)
        return self.head_watch(feat), self.head_phone(feat)
