#!/usr/bin/env python3
"""
Week3 step 3: cascade R_SB → R_MB → MobilePoser pose.

    R_MB = R_MS @ R̂_SB^T ,  a_M unchanged

Conditions (lower is better):
  none      — feed uncalibrated R_MS (R_MS → pose baseline); GT slots
  pred_seq  — per-segment majority slot → Week2 dual R_SB → calib on
              true watch/phone, then pack those streams into predicted
              MobilePoser slots (90-frame segments)
  gt_slot   — oracle position into Week2; GT slots
  oracle    — perfect R_SB; GT slots

Usage (repo root; needs MobilePoser checkpoints/weights.pth):
  python experiments/week3_pos_rsb/eval_step3.py \
      --config experiments/week3_pos_rsb/configs/default.yaml
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import torch
from tqdm import tqdm

_EXP_DIR = Path(__file__).resolve().parent
_REPO = _EXP_DIR.parents[1]
_MP_DIR = _REPO / "mobileposer"
if str(_EXP_DIR) not in sys.path:
    sys.path.insert(0, str(_EXP_DIR))

from cascade_pose import (  # noqa: E402
    cascade_calibrated,
    collect_amass_pose,
    collect_imuposer_pose,
    combo_name,
    pack_condition_imu,
    run_pose,
    _err_dict,
)
from dataset import load_config, load_norm_stats, offset_euler_bounds, resolve_path, set_seed  # noqa: E402
from dataset.pos_dataset import amass_seed_offset  # noqa: E402
from models import PosClassifier, RotExtrinsicDualNet  # noqa: E402

sys.modules.pop("models", None)
if str(_MP_DIR) not in sys.path:
    sys.path.insert(0, str(_MP_DIR))


def _pose_combos(cfg: Dict[str, Any]) -> List[Tuple[str, int, int]]:
    raw = (cfg.get("step3") or {}).get("pose_combos") or [[0, 0], [0, 1], [1, 0], [1, 1]]
    out = []
    for pair in raw:
        yw, yp = int(pair[0]), int(pair[1])
        out.append((combo_name(yw, yp), yw, yp))
    return out


def eval_source(
    seqs: List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]],
    *,
    pos_model,
    rot_model,
    pose_net,
    evaluator,
    window_len: int,
    stride: int,
    acc_scale: float,
    offset_range: float,
    seed: int,
    seed_extra: int,
    combos: List[Tuple[str, int, int]],
    pos_mean: torch.Tensor,
    pos_std: torch.Tensor,
    rot_mean: torch.Tensor,
    rot_std: torch.Tensor,
    device: torch.device,
    desc: str,
    lo_deg: float = 0.0,
    hi_deg: float | None = None,
) -> Dict[str, Any]:
    keys = ("none", "pred_seq", "gt_slot", "oracle")
    pooled = {k: [] for k in keys}
    per_combo: Dict[str, Dict[str, List[torch.Tensor]]] = {
        name: {k: [] for k in keys} for name, _, _ in combos
    }
    n_used = 0
    n_joint_ok = 0
    n_combo_runs = 0
    pred_ok: List[torch.Tensor] = []
    pred_bad: List[torch.Tensor] = []

    for seq_i, (acc_all, ori_all, pose_gt, tran_gt) in enumerate(tqdm(seqs, desc=desc)):
        if acc_all.shape[0] < window_len or acc_all.shape[1] < 5:
            continue
        acc_all = acc_all[:, :5].float()
        ori_all = ori_all[:, :5].float()
        pose_gt = pose_gt.float()
        tran_gt = tran_gt.float()
        seq_ok = False
        for cname, yw, yp in combos:
            w_idx = 0 if yw == 0 else 1
            p_idx = 2 if yp == 0 else 3
            pack = cascade_calibrated(
                acc_all,
                ori_all,
                w_idx=w_idx,
                p_idx=p_idx,
                y_watch=yw,
                y_phone=yp,
                seq_i=seq_i,
                offset_range=offset_range,
                seed=amass_seed_offset(seq_i, yw, yp, seed + seed_extra),
                window_len=window_len,
                stride=stride,
                acc_scale=acc_scale,
                pos_model=pos_model,
                rot_model=rot_model,
                pos_mean=pos_mean,
                pos_std=pos_std,
                rot_mean=rot_mean,
                rot_std=rot_std,
                device=device,
                lo_deg=lo_deg,
                hi_deg=hi_deg,
            )
            if pack is None:
                continue
            n_combo_runs += 1
            joint_ok = int(pack["meta"]["joint_ok"])
            n_joint_ok += joint_ok
            for name in keys:
                acc_c, ori_c = pack[name]
                imu = pack_condition_imu(
                    acc_c,
                    ori_c,
                    name=name,
                    w_idx=w_idx,
                    p_idx=p_idx,
                    acc_scale=acc_scale,
                    dst_watch=pack["dst_watch"],
                    dst_phone=pack["dst_phone"],
                )
                err = run_pose(pose_net, imu, pose_gt, tran_gt, evaluator).cpu()
                per_combo[cname][name].append(err)
                pooled[name].append(err)
                if name == "pred_seq":
                    (pred_ok if joint_ok else pred_bad).append(err)
            seq_ok = True
        if seq_ok:
            n_used += 1

    def _summarize(parts: Dict[str, List[torch.Tensor]]) -> Dict[str, Any]:
        return {
            name: (_err_dict(torch.stack(errs, dim=0)) if errs else None)
            for name, errs in parts.items()
        }

    return {
        "n_sequences_used": n_used,
        "n_combo_runs": n_combo_runs,
        "seq_joint_ok_rate": (n_joint_ok / n_combo_runs) if n_combo_runs else float("nan"),
        "n_pred_joint_ok": len(pred_ok),
        "n_pred_joint_bad": len(pred_bad),
        "combos": [c[0] for c in combos],
        "results": _summarize(pooled),
        "pred_seq_joint_ok": _err_dict(torch.stack(pred_ok, dim=0)) if pred_ok else None,
        "pred_seq_joint_bad": _err_dict(torch.stack(pred_bad, dim=0)) if pred_bad else None,
        "per_combo": {name: _summarize(parts) for name, parts in per_combo.items()},
    }


def _print_block(title: str, block: Dict[str, Any]) -> None:
    print(f"\n======== {title} n_seq={block['n_sequences_used']} ========")
    print(
        f"  combo_runs={block['n_combo_runs']}  "
        f"seq-joint-ok={block['seq_joint_ok_rate']:.3f}"
    )
    print(f"  {'cond':<10} {'pos_cm':>8} {'ang_deg':>8} {'sip_deg':>8} {'mesh_cm':>8}")
    for name in ("none", "pred_seq", "gt_slot", "oracle"):
        row = block["results"].get(name)
        if not row:
            continue
        print(
            f"  {name:<10} {row['positional_cm']['mean']:8.2f} "
            f"{row['angular_deg']['mean']:8.2f} "
            f"{row['sip_deg']['mean']:8.2f} "
            f"{row['mesh_cm']['mean']:8.2f}"
        )
    ok_row = block.get("pred_seq_joint_ok")
    bad_row = block.get("pred_seq_joint_bad")
    if ok_row:
        print(
            f"  pred|OK n={block.get('n_pred_joint_ok', 0):<3} "
            f"{ok_row['positional_cm']['mean']:8.2f} "
            f"{ok_row['angular_deg']['mean']:8.2f}"
        )
    if bad_row:
        print(
            f"  pred|BAD n={block.get('n_pred_joint_bad', 0):<2} "
            f"{bad_row['positional_cm']['mean']:8.2f} "
            f"{bad_row['angular_deg']['mean']:8.2f}"
        )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        default="experiments/week3_pos_rsb/configs/default.yaml",
    )
    parser.add_argument("--pos-checkpoint", type=str, default=None)
    parser.add_argument("--rot-checkpoint", type=str, default=None)
    parser.add_argument("--mobileposer-checkpoint", type=str, default=None)
    parser.add_argument("--skip-amass", action="store_true")
    parser.add_argument("--skip-imuposer", action="store_true")
    parser.add_argument("--max-seqs", type=int, default=None)
    parser.add_argument("--device", type=str, default=None)
    args = parser.parse_args()

    cfg = load_config(resolve_path(args.config))
    set_seed(cfg["experiment"]["seed"])
    device_name = args.device or (
        cfg["train"]["device"] if torch.cuda.is_available() else "cpu"
    )
    device = torch.device(device_name)

    # MobilePoser helpers bind model_config.device at import; keep them aligned.
    sys.path.insert(0, str(_MP_DIR))
    import config as mp_cfg  # noqa: WPS433

    mp_cfg.model_config.device = device
    from eval_mocap import PoseEvaluator  # noqa: E402, WPS433
    from utils.model_utils import load_model  # noqa: E402, WPS433

    step2 = cfg.get("step2") or {}
    step3 = cfg.get("step3") or {}
    pos_ckpt = resolve_path(args.pos_checkpoint or cfg["eval"]["checkpoint"])
    rot_ckpt = resolve_path(
        args.rot_checkpoint
        or step2.get("week2_dual_checkpoint")
        or "experiments/week2_rot_ext/outputs/checkpoints/best_rot_err_dual.pt"
    )
    mp_ckpt = resolve_path(
        args.mobileposer_checkpoint
        or step3.get("mobileposer_checkpoint")
        or "checkpoints/weights.pth"
    )
    pos_stats_pt = resolve_path(cfg["data"]["out_dir"]) / "norm_stats.pt"
    rot_stats_pt = resolve_path(
        step2.get("week2_dual_norm_stats")
        or "experiments/week2_rot_ext/outputs/data/norm_stats_dual.pt"
    )
    for p in (pos_ckpt, rot_ckpt, mp_ckpt, pos_stats_pt, rot_stats_pt):
        if not p.exists():
            raise FileNotFoundError(p)

    pos_stats = load_norm_stats(pos_stats_pt)
    rot_stats = load_norm_stats(rot_stats_pt)
    pos_mean = pos_stats["mean"].float().view(1, 1, -1).to(device)
    pos_std = pos_stats["std"].float().view(1, 1, -1).clamp_min(1e-6).to(device)
    rot_mean = rot_stats["mean"].float().view(1, 1, -1).to(device)
    rot_std = rot_stats["std"].float().view(1, 1, -1).clamp_min(1e-6).to(device)

    pos_model = PosClassifier(
        n_input=cfg["data"]["input_dim"],
        n_hidden=cfg["model"]["n_hidden"],
        n_lstm_layers=cfg["model"]["n_lstm_layers"],
        bidirectional=cfg["model"]["bidirectional"],
        dropout=0.0,
    ).to(device)
    pos_payload = torch.load(pos_ckpt, map_location=device)
    pos_model.load_state_dict(pos_payload["model"])
    pos_model.eval()

    rot_model = RotExtrinsicDualNet(
        feat_dim=24,
        n_slots=4,
        n_hidden=cfg["model"]["n_hidden"],
        n_lstm_layers=cfg["model"]["n_lstm_layers"],
        bidirectional=cfg["model"]["bidirectional"],
        dropout=0.0,
        use_slot_onehot=True,
    ).to(device)
    rot_payload = torch.load(rot_ckpt, map_location=device)
    rot_model.load_state_dict(rot_payload["model"])
    rot_model.eval()

    pose_net = load_model(str(mp_ckpt))
    pose_net.to(device)
    pose_net.eval()
    evaluator = PoseEvaluator()

    window_len = int(cfg["data"]["window_len"])
    stride = int(cfg["data"]["test_stride"])
    acc_scale = float(cfg["data"]["acc_scale"])
    offset_range = float(cfg["data"]["offset_range_deg"])
    lo_deg, hi_deg = offset_euler_bounds(cfg["data"])
    combos = _pose_combos(cfg)
    max_seqs = int(args.max_seqs or step3.get("pose_max_seqs", 12))
    seed = int(cfg["experiment"]["seed"])

    print(
        f"pos_ckpt={pos_ckpt} epoch={pos_payload.get('epoch')} | "
        f"rot_ckpt={rot_ckpt} epoch={rot_payload.get('epoch')} | "
        f"mp={mp_ckpt} device={device} max_seqs={max_seqs}"
    )

    out: Dict[str, Any] = {
        "protocol": (
            "Cascade step1 per-segment Pred slot → Week2 dual R_SB → "
            "R_MB=R_MS@R_SB^T (calib on true watch/phone channels). "
            "Pred-seq packs those streams into predicted MobilePoser slots; "
            "None/GT-slot/Oracle pack anatomical GT slots. "
            "R_BS window-constant (piecewise). "
            "XYZ Euler per axis Uniform[lo, hi] deg (not ±range)."
        ),
        "pos_checkpoint": str(pos_ckpt),
        "rot_checkpoint": str(rot_ckpt),
        "mobileposer_checkpoint": str(mp_ckpt),
        "offset_range_deg": offset_range,
        "offset_euler_lo_deg": lo_deg,
        "offset_euler_hi_deg": hi_deg,
        "window_len": window_len,
        "max_seqs": max_seqs,
        "combos": [c[0] for c in combos],
        "note": (
            "Pred-seq uses predicted pose-net channels (true IMU remapped). "
            "Official MobilePoser weights were trained with combo mask lw_rp_h; "
            "other combos are a domain shift. Sequence count matches Week2 pose "
            "downstream (default 12) so GT-slot is comparable to Week2 dual Learned. "
            "Old IMUPoser Pred-seq 7.21 cm used GT-slot packing and must not be mixed."
        ),
        "datasets": {},
    }

    if not args.skip_amass:
        seqs = collect_amass_pose(
            resolve_path(cfg["data"]["processed_amass_dir"]),
            cfg["data"]["amass_subsets"],
            max_seqs,
            window_len,
        )
        print(f"[amass] sequences={len(seqs)} combos={[c[0] for c in combos]}")
        out["datasets"]["amass"] = eval_source(
            seqs,
            pos_model=pos_model,
            rot_model=rot_model,
            pose_net=pose_net,
            evaluator=evaluator,
            window_len=window_len,
            stride=stride,
            acc_scale=acc_scale,
            offset_range=offset_range,
            seed=seed,
            seed_extra=0,
            combos=combos,
            pos_mean=pos_mean,
            pos_std=pos_std,
            rot_mean=rot_mean,
            rot_std=rot_std,
            device=device,
            desc="pose AMASS",
            lo_deg=lo_deg,
            hi_deg=hi_deg,
        )
        _print_block("AMASS", out["datasets"]["amass"])

    if not args.skip_imuposer:
        seqs = collect_imuposer_pose(
            resolve_path(cfg["data"]["processed_imuposer_file"]),
            max_seqs,
            window_len,
        )
        print(f"[imuposer] sequences={len(seqs)} combos={[c[0] for c in combos]}")
        out["datasets"]["imuposer"] = eval_source(
            seqs,
            pos_model=pos_model,
            rot_model=rot_model,
            pose_net=pose_net,
            evaluator=evaluator,
            window_len=window_len,
            stride=stride,
            acc_scale=acc_scale,
            offset_range=offset_range,
            seed=seed,
            seed_extra=1009,
            combos=combos,
            pos_mean=pos_mean,
            pos_std=pos_std,
            rot_mean=rot_mean,
            rot_std=rot_std,
            device=device,
            desc="pose IMUPoser",
            lo_deg=lo_deg,
            hi_deg=hi_deg,
        )
        _print_block("IMUPoser", out["datasets"]["imuposer"])

    log_dir = resolve_path(cfg["eval"]["log_dir"])
    log_dir.mkdir(parents=True, exist_ok=True)
    out_path = log_dir / "metrics_step3.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print(f"\nsaved {out_path}")

    print("\n=== Step3 Main (positional_cm / angular_deg) ===")
    header = f"{'Split':<10} {'None':>16} {'Pred-seq':>16} {'GT-slot':>16} {'Oracle':>16}"
    print(header)
    print("-" * len(header))
    for tag, lab in (("amass", "AMASS"), ("imuposer", "IMUPoser")):
        if tag not in out["datasets"]:
            continue
        r = out["datasets"][tag]["results"]

        def _cell(name: str) -> str:
            row = r[name]
            return f"{row['positional_cm']['mean']:.2f}/{row['angular_deg']['mean']:.2f}"

        print(
            f"{lab:<10} {_cell('none'):>16} {_cell('pred_seq'):>16} "
            f"{_cell('gt_slot'):>16} {_cell('oracle'):>16}"
        )


if __name__ == "__main__":
    main()
