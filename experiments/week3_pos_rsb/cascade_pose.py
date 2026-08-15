"""Shared Week3 step-3 cascade: Pred slot → R_SB → R_MB → MobilePoser.

Convention (acc unchanged):
    R_MS = R_MB @ R_BS
    R_MB = R_MS @ R_BS^T     (network output is the injected right-multiply)
    a_M  = a_obs
"""
from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import torch

from dataset import (
    apply_mount_offset,
    calibrate_with_rsb,
    combo_to_indices,
    make_rms_features,
    r6d_to_rotation_matrix,
    rotation_matrix_to_r6d,
    sample_random_offsets,
    sides_to_abs_slots,
    slide_windows,
)

HEAD = 4
METRIC_NAMES = [
    "sip_deg",
    "angular_deg",
    "masked_angular_deg",
    "positional_cm",
    "masked_positional_cm",
    "mesh_cm",
    "jitter",
    "distance_cm",
]


def _norm(x: torch.Tensor, mean: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
    return (x.to(device=mean.device, dtype=mean.dtype) - mean) / std


def mean_rotation(rots: Sequence[torch.Tensor]) -> torch.Tensor:
    r6 = torch.stack([rotation_matrix_to_r6d(r) for r in rots], dim=0).mean(dim=0)
    return r6d_to_rotation_matrix(r6)


def apply_slot_offset(acc_all, ori_all, slot, r_bs):
    a, o = apply_mount_offset(acc_all[:, slot], ori_all[:, slot], r_bs)
    acc_all = acc_all.clone()
    ori_all = ori_all.clone()
    acc_all[:, slot] = a
    ori_all[:, slot] = o
    return acc_all, ori_all


def calib_slot(acc_all, ori_all, slot, r_sb):
    a, o = calibrate_with_rsb(acc_all[:, slot], ori_all[:, slot], r_sb)
    acc_all = acc_all.clone()
    ori_all = ori_all.clone()
    acc_all[:, slot] = a
    ori_all[:, slot] = o
    return acc_all, ori_all


def pack_mobileposer_imu(
    acc: torch.Tensor,
    ori: torch.Tensor,
    combo: List[int],
    acc_scale: float,
) -> torch.Tensor:
    """[T, n, 3] / [T, n, 3, 3] → [T, 60] with unused slots zeroed."""
    t_len = acc.shape[0]
    a = torch.zeros(t_len, 5, 3, dtype=acc.dtype)
    r = torch.zeros(t_len, 5, 3, 3, dtype=ori.dtype)
    for s in combo:
        a[:, s] = acc[:, s] / acc_scale
        r[:, s] = ori[:, s]
    return torch.cat([a.flatten(1), r.flatten(1)], dim=1)


def window_features(
    acc_w: torch.Tensor,
    ori_w: torch.Tensor,
    acc_p: torch.Tensor,
    ori_p: torch.Tensor,
    window_len: int,
    stride: int,
    acc_scale: float,
) -> Optional[torch.Tensor]:
    feats = []
    for start, _ in slide_windows(acc_w, window_len, stride):
        sl = slice(start, start + window_len)
        feats.append(make_rms_features(acc_w[sl], ori_w[sl], acc_p[sl], ori_p[sl], acc_scale))
    if not feats:
        return None
    return torch.stack(feats, dim=0)


@torch.no_grad()
def predict_sides(
    pos_model,
    x: torch.Tensor,
    pos_mean: torch.Tensor,
    pos_std: torch.Tensor,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor]:
    xn = _norm(x, pos_mean, pos_std).to(device)
    lw, lp = pos_model(xn)
    return lw.argmax(dim=-1).cpu(), lp.argmax(dim=-1).cpu()


@torch.no_grad()
def predict_rsb_windows(
    rot_model,
    x: torch.Tensor,
    slot_w: torch.Tensor,
    slot_p: torch.Tensor,
    rot_mean: torch.Tensor,
    rot_std: torch.Tensor,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor]:
    xn = _norm(x, rot_mean, rot_std).to(device)
    sw = slot_w.to(device)
    sp = slot_p.to(device)
    (_, rw), (_, rp) = rot_model(xn, sw, sp)
    return rw.cpu(), rp.cpu()


def majority_side(pred: torch.Tensor) -> int:
    vals = [int(v) for v in pred.tolist()]
    return Counter(vals).most_common(1)[0][0]


def estimate_rsb_pair(
    rot_model,
    x: torch.Tensor,
    y_watch: int,
    y_phone: int,
    rot_mean: torch.Tensor,
    rot_std: torch.Tensor,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor]:
    n = x.shape[0]
    sw, sp = sides_to_abs_slots(
        torch.full((n,), int(y_watch), dtype=torch.long),
        torch.full((n,), int(y_phone), dtype=torch.long),
    )
    rw, rp = predict_rsb_windows(rot_model, x, sw, sp, rot_mean, rot_std, device)
    return mean_rotation([rw[i] for i in range(n)]), mean_rotation([rp[i] for i in range(n)])


def inject_combo(
    acc_all: torch.Tensor,
    ori_all: torch.Tensor,
    w_idx: int,
    p_idx: int,
    offset_range: float,
    seed: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    gen = torch.Generator().manual_seed(int(seed))
    r_w = sample_random_offsets(1, offset_range, generator=gen)[0]
    r_p = sample_random_offsets(1, offset_range, generator=gen)[0]
    acc_obs, ori_obs = apply_slot_offset(acc_all, ori_all, w_idx, r_w)
    acc_obs, ori_obs = apply_slot_offset(acc_obs, ori_obs, p_idx, r_p)
    return acc_obs, ori_obs, r_w, r_p


def cascade_calibrated(
    acc_all: torch.Tensor,
    ori_all: torch.Tensor,
    *,
    w_idx: int,
    p_idx: int,
    y_watch: int,
    y_phone: int,
    seq_i: int,
    offset_range: float,
    seed: int,
    window_len: int,
    stride: int,
    acc_scale: float,
    pos_model,
    rot_model,
    pos_mean: torch.Tensor,
    pos_std: torch.Tensor,
    rot_mean: torch.Tensor,
    rot_std: torch.Tensor,
    device: torch.device,
) -> Optional[Dict[str, Tuple[torch.Tensor, torch.Tensor]]]:
    """Return acc/ori for none / pred_seq / gt_slot / oracle, plus hats."""
    acc_obs, ori_obs, r_w, r_p = inject_combo(
        acc_all, ori_all, w_idx, p_idx, offset_range, seed
    )
    x = window_features(
        acc_obs[:, w_idx],
        ori_obs[:, w_idx],
        acc_obs[:, p_idx],
        ori_obs[:, p_idx],
        window_len,
        stride,
        acc_scale,
    )
    if x is None:
        return None
    pw, pp = predict_sides(pos_model, x, pos_mean, pos_std, device)
    yw_hat, yp_hat = majority_side(pw), majority_side(pp)
    r_w_pred, r_p_pred = estimate_rsb_pair(
        rot_model, x, yw_hat, yp_hat, rot_mean, rot_std, device
    )
    r_w_gt, r_p_gt = estimate_rsb_pair(
        rot_model, x, y_watch, y_phone, rot_mean, rot_std, device
    )
    acc_pred, ori_pred = calib_slot(acc_obs, ori_obs, w_idx, r_w_pred)
    acc_pred, ori_pred = calib_slot(acc_pred, ori_pred, p_idx, r_p_pred)
    acc_gt, ori_gt = calib_slot(acc_obs, ori_obs, w_idx, r_w_gt)
    acc_gt, ori_gt = calib_slot(acc_gt, ori_gt, p_idx, r_p_gt)
    acc_or, ori_or = calib_slot(acc_obs, ori_obs, w_idx, r_w)
    acc_or, ori_or = calib_slot(acc_or, ori_or, p_idx, r_p)
    return {
        "none": (acc_obs, ori_obs),
        "pred_seq": (acc_pred, ori_pred),
        "gt_slot": (acc_gt, ori_gt),
        "oracle": (acc_or, ori_or),
        "meta": {
            "y_watch": y_watch,
            "y_phone": y_phone,
            "pred_watch": yw_hat,
            "pred_phone": yp_hat,
            "joint_ok": int(yw_hat == y_watch and yp_hat == y_phone),
            "seq_i": seq_i,
        },
    }


@torch.no_grad()
def run_pose(
    pose_net,
    imu_60: torch.Tensor,
    pose_gt: torch.Tensor,
    tran_gt: torch.Tensor,
    evaluator,
    future: int = 5,
) -> torch.Tensor:
    pose_net.reset()
    device = next(pose_net.parameters()).device
    x = imu_60.to(device)
    pose_t = pose_gt.to(device)
    tran_t = tran_gt.to(device)
    padded = torch.cat([x, x[-1:].repeat(future, 1)], dim=0)
    outs = [pose_net.forward_online(f) for f in padded]
    pose_p, _, tran_p, _ = [torch.stack(v)[future:] for v in zip(*outs)]
    return evaluator.eval(pose_p, pose_t, tran_p=tran_p, tran_t=tran_t)


@torch.no_grad()
def predict_pose_sequence(
    pose_net,
    imu_60: torch.Tensor,
    future: int = 5,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Return (pose_p [T,24,3,3], tran_p [T,3]) on CPU."""
    pose_net.reset()
    device = next(pose_net.parameters()).device
    x = imu_60.to(device)
    padded = torch.cat([x, x[-1:].repeat(future, 1)], dim=0)
    outs = [pose_net.forward_online(f) for f in padded]
    pose_p, _, tran_p, _ = [torch.stack(v)[future:] for v in zip(*outs)]
    return pose_p.detach().cpu(), tran_p.detach().cpu()


def _err_dict(stack: torch.Tensor) -> Dict[str, Dict[str, float]]:
    mean = stack.mean(dim=0)
    out = {}
    for i, name in enumerate(METRIC_NAMES):
        out[name] = {"mean": float(mean[i, 0]), "std": float(mean[i, 1])}
    return out


def combo_name(y_watch: int, y_phone: int) -> str:
    w_idx, p_idx = combo_to_indices(y_watch, y_phone)
    names = ("lw", "rw", "lp", "rp")
    return f"{names[w_idx]}_{names[p_idx]}_h"


def collect_amass_pose(
    processed_dir: Path,
    subsets: List[str],
    max_seqs: int,
    window_len: int,
) -> List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]]:
    seqs = []
    for name in subsets:
        path = processed_dir / f"{name}.pt"
        if not path.exists():
            continue
        data = torch.load(path, map_location="cpu")
        for acc, ori, pose, tran in zip(
            data["acc"], data["ori"], data["pose"], data["tran"]
        ):
            if acc.shape[0] < window_len or acc.shape[1] < 5:
                continue
            seqs.append((acc, ori, pose, tran))
            if len(seqs) >= max_seqs:
                return seqs
    return seqs


def collect_imuposer_pose(
    path: Path, max_seqs: int, window_len: int
) -> List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]]:
    data = torch.load(path, map_location="cpu")
    seqs = []
    n = min(len(data["acc"]), max_seqs)
    for i in range(n):
        acc, ori, pose, tran = (
            data["acc"][i],
            data["ori"][i],
            data["pose"][i],
            data["tran"][i],
        )
        if acc.shape[0] < window_len or acc.shape[1] < 5:
            continue
        seqs.append((acc, ori, pose, tran))
    return seqs


def load_imuposer_pose_one(
    path: Path, seq_i: int, window_len: int
) -> Optional[Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]]:
    data = torch.load(path, map_location="cpu")
    if seq_i < 0 or seq_i >= len(data["acc"]):
        return None
    acc, ori, pose, tran = (
        data["acc"][seq_i],
        data["ori"][seq_i],
        data["pose"][seq_i],
        data["tran"][seq_i],
    )
    if acc.shape[0] < window_len or acc.shape[1] < 5:
        return None
    return acc, ori, pose, tran
