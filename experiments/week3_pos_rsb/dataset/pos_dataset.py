"""Online Week3 step-1 position datasets with window-constant R_BS.

Windows are NOT materialized on disk. Each window has its own
deterministic R_BS pair (constant inside the 90-frame window,
independent across windows of the same sequence).
Input features: a_M(3)+R_MS(9) per device → [T, 24].
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
from torch.utils.data import DataLoader, Dataset

from dataset import (
    apply_mount_offset,
    combo_to_indices,
    make_rms_features,
    offset_euler_bounds,
    resolve_path,
    sample_random_offsets,
)


def amass_seed_offset(
    seq_i: int,
    y_watch: int,
    y_phone: int,
    base: int,
    start: int = 0,
) -> int:
    """Deterministic seed. `start` makes R_BS independent per window."""
    return (
        int(base)
        + 10007 * int(seq_i)
        + 17 * int(y_watch)
        + 3 * int(y_phone)
        + 100003 * int(start)
    )


def _inject_pair(
    acc: torch.Tensor,
    ori: torch.Tensor,
    start: int,
    window_len: int,
    y_watch: int,
    y_phone: int,
    offset_range_deg: float,
    acc_scale: float,
    seed: int,
    lo_deg: float = 0.0,
    hi_deg: float | None = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return (x[T,24], R_BS_watch[3,3], R_BS_phone[3,3]). Seed is window-level."""
    w_idx, p_idx = combo_to_indices(int(y_watch), int(y_phone))
    gen = torch.Generator().manual_seed(int(seed))
    r_w = sample_random_offsets(
        1, offset_range_deg, generator=gen, lo_deg=lo_deg, hi_deg=hi_deg
    )[0]
    r_p = sample_random_offsets(
        1, offset_range_deg, generator=gen, lo_deg=lo_deg, hi_deg=hi_deg
    )[0]
    sl = slice(int(start), int(start) + int(window_len))
    acc_w, ori_w = apply_mount_offset(acc[sl, w_idx], ori[sl, w_idx], r_w)
    acc_p, ori_p = apply_mount_offset(acc[sl, p_idx], ori[sl, p_idx], r_p)
    x = make_rms_features(acc_w, ori_w, acc_p, ori_p, acc_scale)
    return x, r_w, r_p


def _inject_window(
    acc: torch.Tensor,
    ori: torch.Tensor,
    start: int,
    window_len: int,
    y_watch: int,
    y_phone: int,
    offset_range_deg: float,
    acc_scale: float,
    seed: int,
    lo_deg: float = 0.0,
    hi_deg: float | None = None,
) -> torch.Tensor:
    x, _, _ = _inject_pair(
        acc, ori, start, window_len, y_watch, y_phone,
        offset_range_deg, acc_scale, seed, lo_deg, hi_deg,
    )
    return x


class AmassRmsPosDataset(Dataset):
    """AMASS sequences + window index; inject R_BS in __getitem__."""

    def __init__(
        self,
        sequences: List[Tuple[torch.Tensor, torch.Tensor]],
        index: torch.Tensor,
        *,
        window_len: int,
        acc_scale: float,
        offset_range_deg: float,
        seed: int,
        mean: Optional[torch.Tensor] = None,
        std: Optional[torch.Tensor] = None,
        lo_deg: float = 0.0,
        hi_deg: float | None = None,
    ):
        self.sequences = sequences
        self.index = index.long()
        self.window_len = int(window_len)
        self.acc_scale = float(acc_scale)
        self.offset_range_deg = float(offset_range_deg)
        self.offset_lo_deg = float(lo_deg)
        self.offset_hi_deg = None if hi_deg is None else float(hi_deg)
        self.seed = int(seed)
        self.mean = mean.float().view(1, -1) if mean is not None else None
        self.std = std.float().view(1, -1).clamp_min(1e-6) if std is not None else None

    def __len__(self) -> int:
        return int(self.index.shape[0])

    def __getitem__(self, i: int):
        seq_i, yw, yp, start = self.index[i].tolist()
        acc, ori = self.sequences[seq_i]
        seed = amass_seed_offset(seq_i, yw, yp, self.seed, start)
        x = _inject_window(
            acc,
            ori,
            start,
            self.window_len,
            yw,
            yp,
            self.offset_range_deg,
            self.acc_scale,
            seed,
            self.offset_lo_deg,
            self.offset_hi_deg,
        )
        if self.mean is not None:
            x = (x - self.mean) / self.std
        return x, torch.tensor(yw, dtype=torch.long), torch.tensor(yp, dtype=torch.long)


class ImuposerRmsPosDataset(Dataset):
    """IMUPoser recorded stream treated as R_MB; inject window-level R_BS."""

    def __init__(
        self,
        sequences: List[Tuple[torch.Tensor, torch.Tensor]],
        index: torch.Tensor,
        *,
        window_len: int,
        acc_scale: float,
        offset_range_deg: float,
        seed: int,
        y_watch: int,
        y_phone: int,
        mean: torch.Tensor,
        std: torch.Tensor,
        lo_deg: float = 0.0,
        hi_deg: float | None = None,
    ):
        self.sequences = sequences
        self.index = index.long()  # [N, 2] seq, start
        self.window_len = int(window_len)
        self.acc_scale = float(acc_scale)
        self.offset_range_deg = float(offset_range_deg)
        self.offset_lo_deg = float(lo_deg)
        self.offset_hi_deg = None if hi_deg is None else float(hi_deg)
        self.seed = int(seed)
        self.y_watch = int(y_watch)
        self.y_phone = int(y_phone)
        self.mean = mean.float().view(1, -1)
        self.std = std.float().view(1, -1).clamp_min(1e-6)

    def __len__(self) -> int:
        return int(self.index.shape[0])

    def __getitem__(self, i: int):
        seq_i, start = self.index[i].tolist()
        acc, ori = self.sequences[seq_i]
        seed = amass_seed_offset(seq_i, self.y_watch, self.y_phone, self.seed + 1009, start)
        x = _inject_window(
            acc,
            ori,
            start,
            self.window_len,
            self.y_watch,
            self.y_phone,
            self.offset_range_deg,
            self.acc_scale,
            seed,
            self.offset_lo_deg,
            self.offset_hi_deg,
        )
        x = (x - self.mean) / self.std
        return (
            x,
            torch.tensor(self.y_watch, dtype=torch.long),
            torch.tensor(self.y_phone, dtype=torch.long),
        )


class AmassRmsCascadeDataset(Dataset):
    """Raw a_M+R_MS windows + GT R_BS for Week3 step-2 cascade eval.

    Does not normalize: caller applies pos-net stats and Week2 dual stats.
    """

    def __init__(
        self,
        sequences: List[Tuple[torch.Tensor, torch.Tensor]],
        index: torch.Tensor,
        *,
        window_len: int,
        acc_scale: float,
        offset_range_deg: float,
        seed: int,
        lo_deg: float = 0.0,
        hi_deg: float | None = None,
    ):
        self.sequences = sequences
        self.index = index.long()
        self.window_len = int(window_len)
        self.acc_scale = float(acc_scale)
        self.offset_range_deg = float(offset_range_deg)
        self.offset_lo_deg = float(lo_deg)
        self.offset_hi_deg = None if hi_deg is None else float(hi_deg)
        self.seed = int(seed)

    def __len__(self) -> int:
        return int(self.index.shape[0])

    def __getitem__(self, i: int):
        seq_i, yw, yp, start = self.index[i].tolist()
        acc, ori = self.sequences[seq_i]
        seed = amass_seed_offset(seq_i, yw, yp, self.seed, start)
        x, r_w, r_p = _inject_pair(
            acc, ori, start, self.window_len, yw, yp,
            self.offset_range_deg, self.acc_scale, seed,
            self.offset_lo_deg, self.offset_hi_deg,
        )
        return (
            x,
            torch.tensor(yw, dtype=torch.long),
            torch.tensor(yp, dtype=torch.long),
            r_w,
            r_p,
            torch.tensor(seq_i, dtype=torch.long),
            torch.tensor(int(start), dtype=torch.long),
        )


class ImuposerRmsCascadeDataset(Dataset):
    """IMUPoser cascade windows: recorded stream as R_MB, window-level R_BS."""

    def __init__(
        self,
        sequences: List[Tuple[torch.Tensor, torch.Tensor]],
        index: torch.Tensor,
        *,
        window_len: int,
        acc_scale: float,
        offset_range_deg: float,
        seed: int,
        y_watch: int,
        y_phone: int,
        lo_deg: float = 0.0,
        hi_deg: float | None = None,
    ):
        self.sequences = sequences
        self.index = index.long()
        self.window_len = int(window_len)
        self.acc_scale = float(acc_scale)
        self.offset_range_deg = float(offset_range_deg)
        self.offset_lo_deg = float(lo_deg)
        self.offset_hi_deg = None if hi_deg is None else float(hi_deg)
        self.seed = int(seed)
        self.y_watch = int(y_watch)
        self.y_phone = int(y_phone)

    def __len__(self) -> int:
        return int(self.index.shape[0])

    def __getitem__(self, i: int):
        seq_i, start = self.index[i].tolist()
        acc, ori = self.sequences[seq_i]
        seed = amass_seed_offset(seq_i, self.y_watch, self.y_phone, self.seed + 1009, start)
        x, r_w, r_p = _inject_pair(
            acc, ori, start, self.window_len, self.y_watch, self.y_phone,
            self.offset_range_deg, self.acc_scale, seed,
            self.offset_lo_deg, self.offset_hi_deg,
        )
        return (
            x,
            torch.tensor(self.y_watch, dtype=torch.long),
            torch.tensor(self.y_phone, dtype=torch.long),
            r_w,
            r_p,
            torch.tensor(seq_i, dtype=torch.long),
            torch.tensor(int(start), dtype=torch.long),
        )


def load_amass_sequences(
    processed_dir: Path, subsets: List[str]
) -> Tuple[List[Tuple[torch.Tensor, torch.Tensor]], List[str]]:
    seqs = []
    names = []
    for name in subsets:
        path = processed_dir / f"{name}.pt"
        if not path.exists():
            print(f"[warn] missing subset: {path}")
            continue
        data = torch.load(path, map_location="cpu")
        for acc, ori in zip(data["acc"], data["ori"]):
            if acc.shape[1] < 4 or ori.shape[1] < 4:
                continue
            seqs.append((acc[:, :4].float(), ori[:, :4].float()))
            names.append(name)
    return seqs, names


def load_imuposer_sequences(path: Path) -> List[Tuple[torch.Tensor, torch.Tensor]]:
    data = torch.load(path, map_location="cpu")
    seqs = []
    for acc, ori in zip(data["acc"], data["ori"]):
        if acc.shape[1] < 4 or ori.shape[1] < 4:
            continue
        seqs.append((acc[:, :4].float(), ori[:, :4].float()))
    return seqs


def split_val_ids(n_seq: int, val_ratio: float, seed: int) -> set:
    g = torch.Generator().manual_seed(seed)
    perm = torch.randperm(n_seq, generator=g).tolist()
    n_val = max(1, int(round(n_seq * val_ratio))) if n_seq > 1 else 0
    return set(perm[:n_val])


def build_amass_index(
    sequences: List[Tuple[torch.Tensor, torch.Tensor]],
    window_len: int,
    stride: int,
    val_ids: set,
) -> Tuple[torch.Tensor, torch.Tensor]:
    train_rows, val_rows = [], []
    for seq_i, (acc, _) in enumerate(sequences):
        t_len = acc.shape[0]
        if t_len < window_len:
            continue
        bucket = val_rows if seq_i in val_ids else train_rows
        for y_watch in (0, 1):
            for y_phone in (0, 1):
                for start in range(0, t_len - window_len + 1, stride):
                    bucket.append((seq_i, y_watch, y_phone, start))
    def _t(rows):
        if not rows:
            return torch.zeros(0, 4, dtype=torch.long)
        return torch.tensor(rows, dtype=torch.long)
    return _t(train_rows), _t(val_rows)


def build_imuposer_index(
    sequences: List[Tuple[torch.Tensor, torch.Tensor]],
    window_len: int,
    stride: int,
) -> torch.Tensor:
    rows = []
    for seq_i, (acc, _) in enumerate(sequences):
        t_len = acc.shape[0]
        if t_len < window_len:
            continue
        for start in range(0, t_len - window_len + 1, stride):
            rows.append((seq_i, start))
    if not rows:
        return torch.zeros(0, 2, dtype=torch.long)
    return torch.tensor(rows, dtype=torch.long)


def compute_norm_stats(
    sequences,
    index: torch.Tensor,
    window_len: int,
    acc_scale: float,
    offset_range_deg: float,
    seed: int,
    max_windows: Optional[int] = None,
    lo_deg: float = 0.0,
    hi_deg: float | None = None,
) -> Dict[str, torch.Tensor]:
    """Streaming mean/std over train windows (optionally subsampled)."""
    n = index.shape[0]
    if n == 0:
        return {"mean": torch.zeros(24), "std": torch.ones(24)}
    if max_windows is not None and n > max_windows:
        g = torch.Generator().manual_seed(seed + 3)
        pick = torch.randperm(n, generator=g)[:max_windows]
        index = index[pick]
        n = index.shape[0]

    sum_x = torch.zeros(24)
    sum_x2 = torch.zeros(24)
    count = 0
    for i in range(n):
        seq_i, yw, yp, start = index[i].tolist()
        acc, ori = sequences[seq_i]
        seed_i = amass_seed_offset(seq_i, yw, yp, seed, start)
        x = _inject_window(
            acc, ori, start, window_len, yw, yp, offset_range_deg, acc_scale, seed_i,
            lo_deg, hi_deg,
        )
        sum_x += x.sum(dim=0)
        sum_x2 += (x * x).sum(dim=0)
        count += x.shape[0]
        if (i + 1) % 20000 == 0:
            print(f"  norm_stats {i + 1}/{n}")
    mean = sum_x / max(count, 1)
    var = (sum_x2 / max(count, 1) - mean * mean).clamp_min(1e-8)
    return {"mean": mean, "std": var.sqrt()}


def offset_kwargs_from_cfg(cfg: dict) -> dict:
    lo_deg, hi_deg = offset_euler_bounds(cfg["data"])
    return {
        "offset_range_deg": cfg["data"]["offset_range_deg"],
        "lo_deg": lo_deg,
        "hi_deg": hi_deg,
    }


def make_amass_rms_loaders(cfg: dict, stats: Dict[str, torch.Tensor], pack: dict):
    sequences = pack["sequences"]
    lo_deg, hi_deg = offset_euler_bounds(cfg["data"])
    common = dict(
        window_len=cfg["data"]["window_len"],
        acc_scale=cfg["data"]["acc_scale"],
        offset_range_deg=cfg["data"]["offset_range_deg"],
        seed=cfg["experiment"]["seed"],
        mean=stats["mean"],
        std=stats["std"],
        lo_deg=lo_deg,
        hi_deg=hi_deg,
    )
    train_ds = AmassRmsPosDataset(
        sequences,
        pack["train_index"],
        **common,
    )
    val_ds = AmassRmsPosDataset(
        sequences,
        pack["val_index"],
        **common,
    )
    train_loader = DataLoader(
        train_ds,
        batch_size=cfg["train"]["batch_size"],
        shuffle=True,
        num_workers=cfg["train"]["num_workers"],
        pin_memory=True,
        drop_last=False,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=cfg["train"]["batch_size"],
        shuffle=False,
        num_workers=cfg["train"]["num_workers"],
        pin_memory=True,
        drop_last=False,
    )
    return train_loader, val_loader


def load_amass_rms_pack(cfg: dict) -> dict:
    """Load sequences + indices saved by build_amass_pos_rms.py."""
    out_dir = resolve_path(cfg["data"]["out_dir"])
    idx_pt = out_dir / "amass_index.pt"
    if not idx_pt.exists():
        raise FileNotFoundError(
            f"Missing {idx_pt}. Run dataset/build_amass.py first."
        )
    payload = torch.load(idx_pt, map_location="cpu")
    processed_dir = resolve_path(cfg["data"]["processed_amass_dir"])
    sequences, _ = load_amass_sequences(processed_dir, payload["subsets"])
    if len(sequences) != int(payload["n_sequences"]):
        raise RuntimeError(
            f"AMASS sequence count changed: {len(sequences)} vs {payload['n_sequences']}"
        )
    return {
        "sequences": sequences,
        "train_index": payload["train_index"],
        "val_index": payload["val_index"],
    }
