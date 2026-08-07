"""Shared helpers for Week2 R_SB (mounting extrinsic) estimation."""
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
    """r: [..., 3, 3] -> [..., 6] (first two columns, each column contiguous)."""
    # Avoid reshape on [..., 3, 2] which would interleave rows.
    return torch.cat([r[..., :, 0], r[..., :, 1]], dim=-1)


def r6d_to_rotation_matrix(r6d: torch.Tensor) -> torch.Tensor:
    """
    Zhou et al. 6D -> SO(3).
    r6d: [..., 6] -> [..., 3, 3]
    """
    a1 = r6d[..., 0:3]
    a2 = r6d[..., 3:6]
    b1 = torch.nn.functional.normalize(a1, dim=-1)
    b2 = a2 - (b1 * a2).sum(dim=-1, keepdim=True) * b1
    b2 = torch.nn.functional.normalize(b2, dim=-1)
    b3 = torch.cross(b1, b2, dim=-1)
    return torch.stack([b1, b2, b3], dim=-1)


def geodesic_angle_deg(r1: torch.Tensor, r2: torch.Tensor) -> torch.Tensor:
    """
    Geodesic angle between rotations (degrees).
    r1, r2: [..., 3, 3] -> [...]
    """
    rel = r1.transpose(-1, -2) @ r2
    cos = ((rel.diagonal(dim1=-2, dim2=-1).sum(-1) - 1.0) * 0.5).clamp(-1.0, 1.0)
    return torch.acos(cos) * (180.0 / np.pi)


def _euler_xyz_to_matrix(euler: torch.Tensor) -> torch.Tensor:
    """euler: [N, 3] radians (XYZ intrinsic / R = Rz@Ry@Rx approx via sequential)."""
    # Use ZYX extrinsic == XYZ intrinsic common convention: R = Rx @ Ry @ Rz
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
    """
    Sample n random mount rotations (SO(3)), TIC-like:
    Euler angles ~ Uniform([-range, range]) on each axis.
    Returns: [n, 3, 3]
    """
    lo = -offset_range_deg * np.pi / 180.0
    hi = offset_range_deg * np.pi / 180.0
    euler = torch.empty(n, 3, device=device)
    euler.uniform_(lo, hi, generator=generator)
    return _euler_xyz_to_matrix(euler)


def apply_mount_offset(
    acc: torch.Tensor,
    ori: torch.Tensor,
    r_sb: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Apply static mounting extrinsic R_SB (same convention as TIC offset, drift=I):

        R_obs = R_bone @ R_SB
        a_obs = R_SB^T @ a_bone   (acc as column vectors)

    Args:
        acc: [T, 3]
        ori: [T, 3, 3]
        r_sb: [3, 3]
    Returns:
        acc_obs [T, 3], ori_obs [T, 3, 3]
    """
    # ori_obs[t] = ori[t] @ R_SB
    ori_obs = ori @ r_sb
    # a_obs[t] = R_SB^T @ a[t]
    acc_obs = (r_sb.transpose(0, 1) @ acc.unsqueeze(-1)).squeeze(-1)
    return acc_obs, ori_obs


def calibrate_with_rsb(
    acc_obs: torch.Tensor,
    ori_obs: torch.Tensor,
    r_sb: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Inverse of apply_mount_offset:

        R_cal = R_obs @ R_SB^T
        a_cal = R_SB @ a_obs
    """
    ori_cal = ori_obs @ r_sb.transpose(0, 1)
    acc_cal = (r_sb @ acc_obs.unsqueeze(-1)).squeeze(-1)
    return acc_cal, ori_cal


def make_device_features(
    acc: torch.Tensor,
    ori: torch.Tensor,
    acc_scale: float,
) -> torch.Tensor:
    """
    Build [T, 12] = acc(3)/scale + ori_flat(9) row-major.
    acc: [T, 3], ori: [T, 3, 3]
    """
    a = acc / acc_scale
    o = ori.reshape(ori.shape[0], 9)
    return torch.cat([a, o], dim=-1)


def rotation_to_gyro(ori: torch.Tensor, fps: float = 30.0) -> torch.Tensor:
    """Estimate angular velocity (rad/s) from consecutive rotation matrices. ori: [T,3,3]."""
    t = ori.shape[0]
    if t < 2:
        return torch.zeros(t, 3, dtype=ori.dtype)
    r0 = ori[:-1]
    r1 = ori[1:]
    r_rel = torch.matmul(r0.transpose(-1, -2), r1)
    skew = 0.5 * (r_rel - r_rel.transpose(-1, -2))
    omega = torch.stack([skew[:, 2, 1], skew[:, 0, 2], skew[:, 1, 0]], dim=-1) * fps
    return torch.cat([omega[:1], omega], dim=0)


def make_week1_dual_features(
    watch_acc: torch.Tensor,
    watch_ori: torch.Tensor,
    phone_acc: torch.Tensor,
    phone_ori: torch.Tensor,
    fps: float,
    acc_scale: float,
) -> torch.Tensor:
    """Week1-style [T,12]: watch_acc, watch_gyro, phone_acc, phone_gyro."""
    w_acc = watch_acc / acc_scale
    p_acc = phone_acc / acc_scale
    w_gyro = rotation_to_gyro(watch_ori, fps=fps)
    p_gyro = rotation_to_gyro(phone_ori, fps=fps)
    return torch.cat([w_acc, w_gyro, p_acc, p_gyro], dim=-1)


def slot_one_hot(slot: int, n_slots: int = 4) -> torch.Tensor:
    v = torch.zeros(n_slots, dtype=torch.float32)
    v[slot] = 1.0
    return v


def combo_to_indices(y_watch: int, y_phone: int) -> Tuple[int, int]:
    """y_watch/y_phone in {0,1} -> absolute IMU slot indices."""
    watch_idx = LW if y_watch == 0 else RW
    phone_idx = LP if y_phone == 0 else RP
    return watch_idx, phone_idx
