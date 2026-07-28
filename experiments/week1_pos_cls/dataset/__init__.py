"""Shared helpers for Week1 position-classification datasets."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import yaml


# MobilePoser IMU order: LW, RW, LP, RP, head, pelvis
LW, RW, LP, RP = 0, 1, 2, 3


def load_config(path: str | Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def project_root() -> Path:
    """Repo root that contains setup.py / mobileposer/."""
    return Path(__file__).resolve().parents[3]


def resolve_path(p: str | Path) -> Path:
    path = Path(p)
    if path.is_absolute():
        return path
    return project_root() / path


def rotation_to_gyro(ori: torch.Tensor, fps: float = 30.0) -> torch.Tensor:
    """
    Estimate angular velocity (rad/s) from consecutive rotation matrices.

    Args:
        ori: [T, 3, 3]
    Returns:
        gyro: [T, 3]
    """
    t = ori.shape[0]
    if t < 2:
        return torch.zeros(t, 3, dtype=ori.dtype)

    r0 = ori[:-1]
    r1 = ori[1:]
    # R_rel = R_t^T @ R_{t+1}
    r_rel = torch.matmul(r0.transpose(-1, -2), r1)
    # vee(log(R)) approx from skew-symmetric part
    skew = 0.5 * (r_rel - r_rel.transpose(-1, -2))
    omega = torch.stack([skew[:, 2, 1], skew[:, 0, 2], skew[:, 1, 0]], dim=-1) * fps
    gyro = torch.cat([omega[:1], omega], dim=0)
    return gyro


def make_window_features(
    acc: torch.Tensor,
    ori: torch.Tensor,
    watch_idx: int,
    phone_idx: int,
    fps: float,
    acc_scale: float,
) -> torch.Tensor:
    """
    Build [T, 12] features:
      watch_acc(3), watch_gyro(3), phone_acc(3), phone_gyro(3)
    """
    w_acc = acc[:, watch_idx] / acc_scale
    p_acc = acc[:, phone_idx] / acc_scale
    w_gyro = rotation_to_gyro(ori[:, watch_idx], fps=fps)
    p_gyro = rotation_to_gyro(ori[:, phone_idx], fps=fps)
    return torch.cat([w_acc, w_gyro, p_acc, p_gyro], dim=-1)


def slide_windows(
    x: torch.Tensor,
    window_len: int,
    stride: int,
) -> List[torch.Tensor]:
    """x: [T, C] -> list of [W, C]."""
    t = x.shape[0]
    if t < window_len:
        return []
    outs = []
    for start in range(0, t - window_len + 1, stride):
        outs.append(x[start : start + window_len].clone())
    return outs


def combo_to_indices(y_watch: int, y_phone: int) -> Tuple[int, int]:
    """y_watch/y_phone in {0,1} -> absolute IMU slot indices."""
    watch_idx = LW if y_watch == 0 else RW
    phone_idx = LP if y_phone == 0 else RP
    return watch_idx, phone_idx


def set_seed(seed: int) -> None:
    import random

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
