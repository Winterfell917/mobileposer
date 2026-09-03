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


def sides_to_abs_slots(y_watch, y_phone):
    """Week1/3 side labels {0,1}×{0,1} → Week2 absolute slots {0,1}×{2,3}."""
    return y_watch, y_phone + 2


def slide_windows(
    x: torch.Tensor,
    window_len: int,
    stride: int,
) -> List[Tuple[int, torch.Tensor]]:
    """x: [T, C] -> list of (start, [W, C])."""
    t = x.shape[0]
    if t < window_len:
        return []
    outs = []
    for start in range(0, t - window_len + 1, stride):
        outs.append((start, x[start : start + window_len].clone()))
    return outs


def rotation_matrix_to_r6d(r: torch.Tensor) -> torch.Tensor:
    """r: [..., 3, 3] -> [..., 6] (first two columns)."""
    return torch.cat([r[..., :, 0], r[..., :, 1]], dim=-1)


def r6d_to_rotation_matrix(r6d: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    a1 = r6d[..., 0:3]
    a2 = r6d[..., 3:6]
    b1 = torch.nn.functional.normalize(a1, dim=-1, eps=eps)
    b2 = a2 - (b1 * a2).sum(dim=-1, keepdim=True) * b1
    b2 = torch.nn.functional.normalize(b2, dim=-1, eps=eps)
    b3 = torch.cross(b1, b2, dim=-1)
    return torch.stack([b1, b2, b3], dim=-1)


def rotation_cos(r1: torch.Tensor, r2: torch.Tensor) -> torch.Tensor:
    rel = r1.transpose(-1, -2) @ r2
    return ((rel.diagonal(dim1=-2, dim2=-1).sum(-1) - 1.0) * 0.5).clamp(
        -1.0 + 1e-7, 1.0 - 1e-7
    )


def geodesic_angle_deg(r1: torch.Tensor, r2: torch.Tensor) -> torch.Tensor:
    return torch.acos(rotation_cos(r1, r2)) * (180.0 / np.pi)


def calibrate_with_rsb(
    acc_obs: torch.Tensor,
    ori_obs: torch.Tensor,
    r_sb: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """R_cal = R_obs @ R_SB^T; a_cal = a_obs."""
    return acc_obs.clone(), ori_obs @ r_sb.transpose(0, 1)


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


def offset_euler_bounds(data_cfg: Dict[str, Any]) -> Tuple[float, float]:
    """XYZ Euler per-axis Uniform[lo, hi] degrees from config.

    Default is ``[0, offset_range_deg]``. Set ``offset_euler_lo_deg`` /
    ``offset_euler_hi_deg`` to override (legacy ±45: lo=-45, hi=45).
    """
    rng = float(data_cfg["offset_range_deg"])
    lo = data_cfg.get("offset_euler_lo_deg")
    hi = data_cfg.get("offset_euler_hi_deg")
    return (
        float(0.0 if lo is None else lo),
        float(rng if hi is None else hi),
    )


def sample_random_offsets(
    n: int,
    offset_range_deg: float,
    generator: torch.Generator | None = None,
    device: torch.device | None = None,
    lo_deg: float | None = None,
    hi_deg: float | None = None,
) -> torch.Tensor:
    """XYZ Euler, each axis Uniform[lo_deg, hi_deg] degrees → [n, 3, 3].

    Default interval is ``[0, offset_range_deg]`` (not ±range).
    """
    if lo_deg is None:
        lo_deg = 0.0
    if hi_deg is None:
        hi_deg = float(offset_range_deg)
    lo = float(lo_deg) * np.pi / 180.0
    hi = float(hi_deg) * np.pi / 180.0
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
    lo_deg: float | None = None,
    hi_deg: float | None = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    r_w = sample_random_offsets(
        1, offset_range_deg, generator=generator, lo_deg=lo_deg, hi_deg=hi_deg
    )[0]
    r_p = sample_random_offsets(
        1, offset_range_deg, generator=generator, lo_deg=lo_deg, hi_deg=hi_deg
    )[0]
    acc_w, ori_w = apply_mount_offset(acc[:, watch_idx], ori[:, watch_idx], r_w)
    acc_p, ori_p = apply_mount_offset(acc[:, phone_idx], ori[:, phone_idx], r_p)
    return acc_w, ori_w, acc_p, ori_p, r_w, r_p


def load_norm_stats(path: str | Path) -> Dict[str, torch.Tensor]:
    return torch.load(path, map_location="cpu")
