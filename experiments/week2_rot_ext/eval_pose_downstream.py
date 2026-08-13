#!/usr/bin/env python3
"""
Week2 R_SB → bone-aligned IMU (R_MB, a_M) → MobilePoser pose.

Convention (acc unchanged):
    R_obs = R_MB @ R_SB
    R_MB  = R_obs @ R_SB^T
    a_M   = a_obs

Enumerates 4 watch×phone combos (LW/RW × LP/RP), each packed as
[watch, phone, Head] for MobilePoser. Head is left clean (no inject).

Usage (repo root; needs Week2 ckpt + MobilePoser weights.pth):
  python experiments/week2_rot_ext/eval_pose_downstream.py \\
      --config experiments/week2_rot_ext/configs/default.yaml
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
from tqdm import tqdm

_EXP_DIR = Path(__file__).resolve().parent
_REPO = _EXP_DIR.parents[1]
_MP_DIR = _REPO / "mobileposer"
if str(_EXP_DIR) not in sys.path:
    sys.path.insert(0, str(_EXP_DIR))

from dataset import (  # noqa: E402  week2
    SLOT_NAMES,
    apply_mount_offset,
    calibrate_with_rsb,
    combo_to_indices,
    load_config,
    make_device_features,
    r6d_to_rotation_matrix,
    resolve_path,
    rotation_matrix_to_r6d,
    sample_random_offsets,
    set_seed,
    slide_windows,
)
from models import RotExtrinsicDualNet, RotExtrinsicNet  # noqa: E402

# MobilePoser also ships a top-level `models` package — swap after Week2 imports.
sys.modules.pop("models", None)
if str(_MP_DIR) not in sys.path:
    sys.path.insert(0, str(_MP_DIR))

from eval_mocap import PoseEvaluator  # noqa: E402
from utils.model_utils import load_model  # noqa: E402

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
    return (x - mean) / std


def _mean_rotation(rots: List[torch.Tensor]) -> torch.Tensor:
    r6 = torch.stack([rotation_matrix_to_r6d(r) for r in rots], dim=0).mean(dim=0)
    return r6d_to_rotation_matrix(r6)


@torch.no_grad()
def predict_rsb_window(
    model: RotExtrinsicNet,
    acc: torch.Tensor,
    ori: torch.Tensor,
    slot: int,
    acc_scale: float,
    mean: torch.Tensor,
    std: torch.Tensor,
    device: torch.device,
) -> torch.Tensor:
    feat = make_device_features(acc, ori, acc_scale)
    x = _norm(feat, mean, std).unsqueeze(0).to(device)
    slot_t = torch.tensor([slot], device=device)
    _, r = model(x, slot_t)
    return r[0].cpu()


@torch.no_grad()
def predict_rsb_dual_sequence(
    model: RotExtrinsicDualNet,
    acc_w: torch.Tensor,
    ori_w: torch.Tensor,
    acc_p: torch.Tensor,
    ori_p: torch.Tensor,
    slot_w: int,
    slot_p: int,
    window_len: int,
    stride: int,
    acc_scale: float,
    mean: torch.Tensor,
    std: torch.Tensor,
    device: torch.device,
) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
    t_len = acc_w.shape[0]
    if t_len < window_len:
        return None, None
    pred_w, pred_p = [], []
    for start, _ in slide_windows(acc_w, window_len, stride):
        fw = make_device_features(
            acc_w[start : start + window_len],
            ori_w[start : start + window_len],
            acc_scale,
        )
        fp = make_device_features(
            acc_p[start : start + window_len],
            ori_p[start : start + window_len],
            acc_scale,
        )
        x = _norm(torch.cat([fw, fp], dim=-1), mean, std).unsqueeze(0).to(device)
        sw = torch.tensor([slot_w], device=device)
        sp = torch.tensor([slot_p], device=device)
        (_, rw), (_, rp) = model(x, sw, sp)
        pred_w.append(rw[0].cpu())
        pred_p.append(rp[0].cpu())
    if not pred_w:
        return None, None
    return _mean_rotation(pred_w), _mean_rotation(pred_p)


@torch.no_grad()
def predict_rsb_sequence(
    model: RotExtrinsicNet,
    acc: torch.Tensor,
    ori: torch.Tensor,
    slot: int,
    window_len: int,
    stride: int,
    acc_scale: float,
    mean: torch.Tensor,
    std: torch.Tensor,
    device: torch.device,
) -> Optional[torch.Tensor]:
    """Average window-wise R_SB (6D mean) over a sequence. Static mount."""
    t_len = acc.shape[0]
    if t_len < window_len:
        return None
    preds = []
    for start, _ in slide_windows(acc, window_len, stride):
        preds.append(
            predict_rsb_window(
                model,
                acc[start : start + window_len],
                ori[start : start + window_len],
                slot,
                acc_scale,
                mean,
                std,
                device,
            )
        )
    return _mean_rotation(preds) if preds else None


def pack_mobileposer_imu(
    acc: torch.Tensor,
    ori: torch.Tensor,
    combo: List[int],
    acc_scale: float,
) -> torch.Tensor:
    """[T, n, 3] / [T, n, 3, 3] → [T, 60] with unused slots zeroed (MobilePoser combo)."""
    t_len = acc.shape[0]
    a = torch.zeros(t_len, 5, 3, dtype=acc.dtype)
    r = torch.zeros(t_len, 5, 3, 3, dtype=ori.dtype)
    for s in combo:
        a[:, s] = acc[:, s] / acc_scale
        r[:, s] = ori[:, s]
    return torch.cat([a.flatten(1), r.flatten(1)], dim=1)


def apply_slot_offset(acc_all, ori_all, slot, r_sb):
    a, o = apply_mount_offset(acc_all[:, slot], ori_all[:, slot], r_sb)
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


@torch.no_grad()
def run_pose(
    pose_net,
    imu_60: torch.Tensor,
    pose_gt: torch.Tensor,
    tran_gt: torch.Tensor,
    evaluator: PoseEvaluator,
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


def _err_dict(stack: torch.Tensor) -> Dict[str, Dict[str, float]]:
    mean = stack.mean(dim=0)
    out = {}
    for i, name in enumerate(METRIC_NAMES):
        out[name] = {
            "mean": float(mean[i, 0]),
            "std": float(mean[i, 1]),
        }
    return out


def _combo_name(watch_idx: int, phone_idx: int) -> str:
    return f"{SLOT_NAMES[watch_idx].lower()}_{SLOT_NAMES[phone_idx].lower()}_h"


def _pose_combos(cfg: Dict[str, Any]) -> List[Tuple[str, int, int]]:
    """(name, watch_slot, phone_slot); Head is always appended at pack time."""
    raw = cfg["eval"].get("pose_combos") or cfg["eval"]["downstream_combos"]
    out = []
    for pair in raw:
        w, p = combo_to_indices(int(pair[0]), int(pair[1]))
        out.append((_combo_name(w, p), w, p))
    return out


def eval_source(
    seqs: List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]],
    *,
    rot_net,
    pose_net,
    use_dual: bool,
    window_len: int,
    stride: int,
    acc_scale: float,
    offset_range: float,
    combos: List[Tuple[str, int, int]],
    w2_mean: torch.Tensor,
    w2_std: torch.Tensor,
    device: torch.device,
    gen: torch.Generator,
    desc: str,
) -> Dict[str, object]:
    evaluator = PoseEvaluator()
    pooled = {k: [] for k in ("none", "learned", "oracle")}
    per_combo: Dict[str, Dict[str, List[torch.Tensor]]] = {
        name: {k: [] for k in ("none", "learned", "oracle")} for name, _, _ in combos
    }
    n_used = 0

    for acc_all, ori_all, pose_gt, tran_gt in tqdm(seqs, desc=desc):
        if acc_all.shape[0] < window_len or acc_all.shape[1] < 5:
            continue
        acc_all = acc_all[:, :5].float()
        ori_all = ori_all[:, :5].float()
        pose_gt = pose_gt.float()
        tran_gt = tran_gt.float()
        seq_ok = False

        for cname, w_idx, p_idx in combos:
            r_w = sample_random_offsets(1, offset_range, generator=gen)[0]
            r_p = sample_random_offsets(1, offset_range, generator=gen)[0]
            acc_obs, ori_obs = apply_slot_offset(acc_all, ori_all, w_idx, r_w)
            acc_obs, ori_obs = apply_slot_offset(acc_obs, ori_obs, p_idx, r_p)
            slots = [w_idx, p_idx, HEAD]

            if use_dual:
                r_w_hat, r_p_hat = predict_rsb_dual_sequence(
                    rot_net,
                    acc_obs[:, w_idx],
                    ori_obs[:, w_idx],
                    acc_obs[:, p_idx],
                    ori_obs[:, p_idx],
                    w_idx,
                    p_idx,
                    window_len,
                    stride,
                    acc_scale,
                    w2_mean,
                    w2_std,
                    device,
                )
            else:
                r_w_hat = predict_rsb_sequence(
                    rot_net,
                    acc_obs[:, w_idx],
                    ori_obs[:, w_idx],
                    w_idx,
                    window_len,
                    stride,
                    acc_scale,
                    w2_mean,
                    w2_std,
                    device,
                )
                r_p_hat = predict_rsb_sequence(
                    rot_net,
                    acc_obs[:, p_idx],
                    ori_obs[:, p_idx],
                    p_idx,
                    window_len,
                    stride,
                    acc_scale,
                    w2_mean,
                    w2_std,
                    device,
                )
            if r_w_hat is None or r_p_hat is None:
                continue

            acc_l, ori_l = calib_slot(acc_obs, ori_obs, w_idx, r_w_hat)
            acc_l, ori_l = calib_slot(acc_l, ori_l, p_idx, r_p_hat)
            acc_o, ori_o = calib_slot(acc_obs, ori_obs, w_idx, r_w)
            acc_o, ori_o = calib_slot(acc_o, ori_o, p_idx, r_p)

            packed = {
                "none": pack_mobileposer_imu(acc_obs, ori_obs, slots, acc_scale),
                "learned": pack_mobileposer_imu(acc_l, ori_l, slots, acc_scale),
                "oracle": pack_mobileposer_imu(acc_o, ori_o, slots, acc_scale),
            }
            for name, imu in packed.items():
                err = run_pose(pose_net, imu, pose_gt, tran_gt, evaluator).cpu()
                per_combo[cname][name].append(err)
                pooled[name].append(err)
            seq_ok = True
        if seq_ok:
            n_used += 1

    def _summarize(parts: Dict[str, List[torch.Tensor]]) -> Dict[str, object]:
        results = {}
        for name, errs in parts.items():
            results[name] = _err_dict(torch.stack(errs, dim=0)) if errs else None
        return results

    return {
        "n_sequences_used": n_used,
        "combos": [c[0] for c in combos],
        "results": _summarize(pooled),
        "per_combo": {
            name: _summarize(parts) for name, parts in per_combo.items()
        },
    }


def _collect_amass(
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


def _collect_imuposer(
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        default="experiments/week2_rot_ext/configs/default.yaml",
    )
    parser.add_argument("--dual", action="store_true")
    parser.add_argument("--skip-amass", action="store_true")
    parser.add_argument("--skip-imuposer", action="store_true")
    parser.add_argument("--max-seqs", type=int, default=None)
    parser.add_argument(
        "--mobileposer-checkpoint",
        type=str,
        default=None,
        help="Override MobilePoser weights.pth",
    )
    args = parser.parse_args()

    cfg = load_config(resolve_path(args.config))
    set_seed(cfg["experiment"]["seed"])
    device = torch.device(
        cfg["train"]["device"] if torch.cuda.is_available() else "cpu"
    )

    use_dual = bool(args.dual)
    w2_ckpt = resolve_path(
        "experiments/week2_rot_ext/outputs/checkpoints/best_rot_err_dual.pt"
        if use_dual
        else cfg["eval"]["checkpoint"]
    )
    stats_pt = resolve_path(cfg["data"]["out_dir"]) / (
        "norm_stats_dual.pt" if use_dual else "norm_stats.pt"
    )
    mp_ckpt = resolve_path(
        args.mobileposer_checkpoint
        or cfg["eval"].get("mobileposer_checkpoint", "checkpoints/weights.pth")
    )
    for p in (w2_ckpt, stats_pt, mp_ckpt):
        if not p.exists():
            raise FileNotFoundError(
                f"Missing {p}. Week2 needs a trained ckpt; "
                "MobilePoser needs checkpoints/weights.pth "
                "(see repo README: Box pretrained weights)."
            )

    stats = torch.load(stats_pt, map_location="cpu")
    w2_mean, w2_std = stats["mean"].float(), stats["std"].float().clamp_min(1e-6)

    if use_dual:
        rot_net = RotExtrinsicDualNet(
            feat_dim=24,
            n_slots=cfg["data"]["n_slots"],
            n_hidden=cfg["model"]["n_hidden"],
            n_lstm_layers=cfg["model"]["n_lstm_layers"],
            bidirectional=cfg["model"]["bidirectional"],
            dropout=cfg["model"]["dropout"],
            use_slot_onehot=cfg["model"]["use_slot_onehot"],
        ).to(device)
    else:
        rot_net = RotExtrinsicNet(
            feat_dim=cfg["data"]["feat_dim"],
            n_slots=cfg["data"]["n_slots"],
            n_hidden=cfg["model"]["n_hidden"],
            n_lstm_layers=cfg["model"]["n_lstm_layers"],
            bidirectional=cfg["model"]["bidirectional"],
            dropout=cfg["model"]["dropout"],
            use_slot_onehot=cfg["model"]["use_slot_onehot"],
        ).to(device)
    rot_net.load_state_dict(torch.load(w2_ckpt, map_location=device)["model"])
    rot_net.eval()

    pose_net = load_model(str(mp_ckpt))
    pose_net.to(device)
    pose_net.eval()

    window_len = int(cfg["data"]["window_len"])
    stride = int(cfg["data"]["test_stride"])
    acc_scale = float(cfg["data"]["acc_scale"])
    offset_range = float(cfg["data"]["offset_range_deg"])
    combos = _pose_combos(cfg)
    max_seqs = int(args.max_seqs or cfg["eval"].get("pose_max_seqs", 12))
    gen = torch.Generator().manual_seed(cfg["experiment"]["seed"] + 21)

    out = {
        "protocol": (
            "Inject R_SB on watch+phone for all 4 combos (LW/RW × LP/RP); "
            "Head stays clean. Calib to R_MB, a_M → MobilePoser [watch, phone, Head]. "
            "None / Learned / Oracle."
        ),
        "week2_mode": "dual_joint" if use_dual else "single_device_x2",
        "combos": [c[0] for c in combos],
        "note": (
            "Official MobilePoser weights were trained with combo mask lw_rp_h; "
            "other combos are a domain shift (slots still valid IMU, unused channels zero)."
        ),
        "offset_range_deg": offset_range,
        "mobileposer_checkpoint": str(mp_ckpt),
        "datasets": {},
    }

    if not args.skip_amass:
        amass_dir = resolve_path(cfg["data"]["processed_amass_dir"])
        seqs = _collect_amass(
            amass_dir, cfg["data"]["amass_subsets"], max_seqs, window_len
        )
        print(f"[amass] sequences={len(seqs)} combos={[c[0] for c in combos]}")
        out["datasets"]["amass"] = eval_source(
            seqs,
            rot_net=rot_net,
            pose_net=pose_net,
            use_dual=use_dual,
            window_len=window_len,
            stride=stride,
            acc_scale=acc_scale,
            offset_range=offset_range,
            combos=combos,
            w2_mean=w2_mean,
            w2_std=w2_std,
            device=device,
            gen=gen,
            desc="pose AMASS",
        )

    if not args.skip_imuposer:
        imu_path = resolve_path(cfg["data"]["processed_imuposer_file"])
        seqs = _collect_imuposer(imu_path, max_seqs, window_len)
        print(f"[imuposer] sequences={len(seqs)} combos={[c[0] for c in combos]}")
        out["datasets"]["imuposer"] = eval_source(
            seqs,
            rot_net=rot_net,
            pose_net=pose_net,
            use_dual=use_dual,
            window_len=window_len,
            stride=stride,
            acc_scale=acc_scale,
            offset_range=offset_range,
            combos=combos,
            w2_mean=w2_mean,
            w2_std=w2_std,
            device=device,
            gen=gen,
            desc="pose IMUPoser",
        )

    log_dir = resolve_path("experiments/week2_rot_ext/outputs/logs")
    log_dir.mkdir(parents=True, exist_ok=True)
    out_path = log_dir / (
        "metrics_pose_downstream_dual.json"
        if use_dual
        else "metrics_pose_downstream.json"
    )
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print(json.dumps(out, indent=2))
    print(f"saved {out_path}")


if __name__ == "__main__":
    main()
