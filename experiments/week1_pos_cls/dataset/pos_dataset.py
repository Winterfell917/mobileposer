"""PyTorch Dataset / DataLoader for Week1 position classification."""
from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional, Tuple

import torch
from torch.utils.data import DataLoader, Dataset


class PosClsDataset(Dataset):
    def __init__(
        self,
        pt_path: str | Path,
        norm_stats: Optional[Dict[str, torch.Tensor]] = None,
        normalize: bool = True,
    ):
        data = torch.load(pt_path, map_location="cpu")
        self.x = data["x"].float()  # [N, W, 12]
        self.y_watch = data["y_watch"].long()
        self.y_phone = data["y_phone"].long()
        self.meta = data.get("meta", [None] * len(self.x))

        self.normalize = normalize
        self.mean = None
        self.std = None
        if normalize:
            if norm_stats is None:
                raise ValueError("norm_stats required when normalize=True")
            self.mean = norm_stats["mean"].float().view(1, 1, -1)
            self.std = norm_stats["std"].float().view(1, 1, -1)

    def __len__(self) -> int:
        return self.x.shape[0]

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        x = self.x[idx]
        if self.normalize:
            x = (x - self.mean.squeeze(0)) / self.std.squeeze(0)
        return x, self.y_watch[idx], self.y_phone[idx]


def load_norm_stats(path: str | Path) -> Dict[str, torch.Tensor]:
    return torch.load(path, map_location="cpu")


def make_loader(
    pt_path: str | Path,
    norm_stats_path: str | Path,
    batch_size: int,
    shuffle: bool,
    num_workers: int = 4,
) -> DataLoader:
    stats = load_norm_stats(norm_stats_path)
    ds = PosClsDataset(pt_path, norm_stats=stats, normalize=True)
    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
    )
