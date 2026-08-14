"""Shared helpers for Week3 step-1: position from a_M + R_MS (unknown R_BS)."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
import yaml

LW, RW, LP, RP = 0, 1, 2, 3
SLOT_NAMES = ("LW", "RW", "LP", "RP")


def load_config(path: str | Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def project_root() -> Path:
    return Path(__file__).resolve().parents[3]


def resolve_path(p: str | Path) -> Path:
    path = Path(p)
    if path.is_absolute():
        return path
    return project_root() / path


def set_seed(seed: int) -> None:
    import random

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def combo_to_indices(y_watch: int, y_phone: int) -> Tuple[int, int]:
    watch_idx = LW if y_watch == 0 else RW
    phone_idx = LP if y_phone == 0 else RP
    return watch_idx, phone_idx


def _euler_xyz_to_matrix(euler: torch.Tensor) -> torch.Tensor:
    cx, cy, cz = torch.cos(euler[:, 0]), torch.cos(euler[:, 1]), torch.cos(euler[:, 2])
    sx, sy, sz = torch.sin(euler[:, 0]), torch.sin(euler[:, 1]), torch.sin(euler[:, 2])
    r = euler.new_zeros(euler.shape[0], 3, 3)
    r[:, 0, 0] = cy * cz
    r[:, 0, 1] = -cy * sz
    r[:, 0, 2] = sy
    r[:, 1, 0] = sx * sy * cz + cx * sz
    r[:, 1, 1] = -sx * sy * sz + cx * cz
    r[:, 1, 2] = -sx * cy
    r[:, 2, 0] = -cx * sy * cz + sx * sz
    r[:, 2, 1] = cx * sy * sz + sx * cz
    r[:, 2, 2] = cx * cy
    return r


def sample_random_offsets(
    n: int,
    offset_range_deg: float,
    generator: torch.Generator | None = None,
    device: torch.device | None = None,
) -> torch.Tensor:
    lo = -offset_range_deg * np.pi / 180.0
    hi = offset_range_deg * np.pi / 180.0
    euler = torch.empty(n, 3, device=device)
    euler.uniform_(lo, hi, generator=generator)
    return _euler_xyz_to_matrix(euler)


def apply_mount_offset(
    acc: torch.Tensor,
    ori: torch.Tensor,
    r_bs: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """R_MS = R_MB @ R_BS; a_M unchanged."""
    return acc.clone(), ori @ r_bs


def make_rms_features(
    watch_acc: torch.Tensor,
    watch_ori: torch.Tensor,
    phone_acc: torch.Tensor,
    phone_ori: torch.Tensor,
    acc_scale: float,
) -> torch.Tensor:
    """[T, 24] = watch a_M(3)+R_MS(9) + phone a_M(3)+R_MS(9)."""
    return torch.cat(
        [
            watch_acc / acc_scale,
            watch_ori.reshape(watch_ori.shape[0], 9),
            phone_acc / acc_scale,
            phone_ori.reshape(phone_ori.shape[0], 9),
        ],
        dim=-1,
    )


def inject_device_pair(
    acc: torch.Tensor,
    ori: torch.Tensor,
    watch_idx: int,
    phone_idx: int,
    offset_range_deg: float,
    generator: torch.Generator,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    r_w = sample_random_offsets(1, offset_range_deg, generator=generator)[0]
    r_p = sample_random_offsets(1, offset_range_deg, generator=generator)[0]
    acc_w, ori_w = apply_mount_offset(acc[:, watch_idx], ori[:, watch_idx], r_w)
    acc_p, ori_p = apply_mount_offset(acc[:, phone_idx], ori[:, phone_idx], r_p)
    return acc_w, ori_w, acc_p, ori_p, r_w, r_p


def load_norm_stats(path: str | Path) -> Dict[str, torch.Tensor]:
    return torch.load(path, map_location="cpu")
