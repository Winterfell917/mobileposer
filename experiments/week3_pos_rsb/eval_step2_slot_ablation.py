#!/usr/bin/env python3
"""Q4 eval-only slot ablations on frozen Week2 dual.

Does not retrain. Same windows / injection as eval_step2.
Conditions (R_SB geodesic deg, lower better):
  none     — identity
  gt_slot  — oracle position one-hot (4.0)
  pred_seq — step-1 majority slot (already in eval_step2)
  random   — 4.2 per-window random watch{0,1} phone{2,3}
  swap     — 4.3 LW↔RW and LP↔RP
  zero     — 4.4 all-zero one-hot (unseen code)

Usage (repo root):
  python experiments/week3_pos_rsb/eval_step2_slot_ablation.py --device cuda:3 \\
    --log-dir experiments/week3_pos_rsb/outputs/logs/ablation_q4
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
from tqdm import tqdm

_EXP_DIR = Path(__file__).resolve().parent
if str(_EXP_DIR) not in sys.path:
    sys.path.insert(0, str(_EXP_DIR))

from dataset import (  # noqa: E402
    geodesic_angle_deg,
    load_config,
    load_norm_stats,
    resolve_path,
    set_seed,
    sides_to_abs_slots,
)
from dataset.pos_dataset import (  # noqa: E402
    AmassRmsCascadeDataset,
    ImuposerRmsCascadeDataset,
    build_imuposer_index,
    load_amass_rms_pack,
    load_imuposer_sequences,
    offset_kwargs_from_cfg,
)
from eval_step2 import (  # noqa: E402
    _pair,
    collect_split,
    combo_label,
    combo_tag,
    make_loader,
    majority_slots,
    rot_with_slots,
)
from models import PosClassifier, RotExtrinsicDualNet  # noqa: E402


@torch.no_grad()
def dual_forward_cond(
    rot_model,
    x: torch.Tensor,
    cond: torch.Tensor,
):
    """cond: [B, 8] watch one-hot(4)+phone one-hot(4)."""
    b, t, _ = x.shape
    cond_t = cond.unsqueeze(1).expand(-1, t, -1)
    h = rot_model.fc_in(torch.cat([x, cond_t], dim=-1))
    seq, _ = rot_model.lstm(h)
    feat = seq.mean(dim=1)
    r6d_w = rot_model.head_watch(feat)
    r6d_p = rot_model.head_phone(feat)
    from dataset import r6d_to_rotation_matrix

    return r6d_to_rotation_matrix(r6d_w), r6d_to_rotation_matrix(r6d_p)


@torch.no_grad()
def rot_with_cond(
    rot_model,
    x_rot: torch.Tensor,
    cond: torch.Tensor,
    rw: torch.Tensor,
    rp: torch.Tensor,
    device,
    batch_size: int,
) -> Tuple[np.ndarray, np.ndarray]:
    ew, ep = [], []
    n = x_rot.shape[0]
    for i in range(0, n, batch_size):
        xb = x_rot[i : i + batch_size].to(device)
        cb = cond[i : i + batch_size].to(device)
        rwb = rw[i : i + batch_size].to(device)
        rpb = rp[i : i + batch_size].to(device)
        hw, hp = dual_forward_cond(rot_model, xb, cb)
        ew.append(geodesic_angle_deg(hw, rwb).cpu().numpy())
        ep.append(geodesic_angle_deg(hp, rpb).cpu().numpy())
    return np.concatenate(ew), np.concatenate(ep)


def ablate_pack(
    pack: Dict[str, Any],
    rot_model,
    device,
    batch_size: int,
    seed: int,
) -> Dict[str, Any]:
    n = int(len(pack["yw"]))
    yw = torch.from_numpy(pack["yw"].astype(np.int64))
    yp = torch.from_numpy(pack["yp"].astype(np.int64))
    sw_gt, sp_gt = sides_to_abs_slots(yw, yp)

    g = torch.Generator().manual_seed(int(seed) + 404)
    sw_rand = torch.randint(0, 2, (n,), generator=g)
    sp_rand = torch.randint(2, 4, (n,), generator=g)

    sw_swap = 1 - yw
    sp_swap = 2 + (1 - yp)

    cond_zero = torch.zeros(n, 8, dtype=torch.float32)

    mw, mp = majority_slots(
        pack["seq"], pack["yw"], pack["yp"], pack["pw"], pack["pp"]
    )
    sw_pred = torch.from_numpy(mw.astype(np.int64))
    sp_pred = torch.from_numpy(mp.astype(np.int64)) + 2

    modes = {
        "gt_slot": (sw_gt, sp_gt),
        "pred_seq": (sw_pred, sp_pred),
        "random": (sw_rand, sp_rand),
        "swap": (sw_swap, sp_swap),
    }
    out_rsb: Dict[str, Any] = {
        "none": _pair(pack["none_w"], pack["none_p"]),
    }
    for name, (sw, sp) in modes.items():
        ew, ep = rot_with_slots(
            rot_model, pack["x_rot"], sw, sp, pack["rw"], pack["rp"], device, batch_size
        )
        out_rsb[name] = _pair(ew, ep)

    ew, ep = rot_with_cond(
        rot_model, pack["x_rot"], cond_zero, pack["rw"], pack["rp"], device, batch_size
    )
    out_rsb["zero"] = _pair(ew, ep)

    pos = {
        "n": n,
        "watch_acc": float((pack["yw"] == pack["pw"]).mean()),
        "phone_acc": float((pack["yp"] == pack["pp"]).mean()),
        "joint_acc": float(
            ((pack["yw"] == pack["pw"]) & (pack["yp"] == pack["pp"])).mean()
        ),
    }
    return {"pos": pos, "rsb_deg": out_rsb}


def print_block(title: str, m: Dict[str, Any]) -> None:
    pos = m["pos"]
    rsb = m["rsb_deg"]
    print(f"\n======== {title} n={pos['n']} ========")
    print(
        f"  pos  watch={pos['watch_acc']:.4f} phone={pos['phone_acc']:.4f} "
        f"joint={pos['joint_acc']:.4f}"
    )
    for name in ("none", "gt_slot", "pred_seq", "random", "swap", "zero"):
        row = rsb[name]["mean"]
        print(
            f"  R_SB {name:<10} mean={row['mean']:.2f}°  "
            f"watch={rsb[name]['watch']['mean']:.2f}°  "
            f"phone={rsb[name]['phone']['mean']:.2f}°"
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
    parser.add_argument("--split", choices=["val", "test", "both"], default="both")
    parser.add_argument("--combos", type=str, default="lw_lp,lw_rp,rw_lp,rw_rp")
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--log-dir", type=str, default=None)
    args = parser.parse_args()

    cfg = load_config(resolve_path(args.config))
    set_seed(cfg["experiment"]["seed"])
    device_name = args.device or (
        cfg["train"]["device"] if torch.cuda.is_available() else "cpu"
    )
    device = torch.device(device_name)
    step2 = cfg.get("step2") or {}
    pos_ckpt = resolve_path(args.pos_checkpoint or cfg["eval"]["checkpoint"])
    rot_ckpt = resolve_path(
        args.rot_checkpoint
        or step2.get("week2_dual_checkpoint")
        or "experiments/week2_rot_ext/outputs/checkpoints/best_rot_err_dual.pt"
    )
    pos_stats_pt = resolve_path(cfg["data"]["out_dir"]) / "norm_stats.pt"
    rot_stats_pt = resolve_path(
        step2.get("week2_dual_norm_stats")
        or "experiments/week2_rot_ext/outputs/data/norm_stats_dual.pt"
    )
    for p in (pos_ckpt, rot_ckpt, pos_stats_pt, rot_stats_pt):
        if not p.exists():
            raise FileNotFoundError(p)

    print(
        "Q4 slot ablation (eval only). 180° convention: XYZ Euler per axis "
        f"Uniform[{cfg['data'].get('offset_euler_lo_deg', 0)},"
        f"{cfg['data'].get('offset_euler_hi_deg', 180)}] "
        "(NOT axis-angle magnitude)."
    )

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
    print(
        f"pos_ckpt={pos_ckpt} epoch={pos_payload.get('epoch')} | "
        f"rot_ckpt={rot_ckpt} epoch={rot_payload.get('epoch')} | device={device}"
    )

    batch_size = int(cfg["eval"]["batch_size"])
    num_workers = int(cfg["train"]["num_workers"])
    log_dir = resolve_path(
        args.log_dir or "experiments/week3_pos_rsb/outputs/logs/ablation_q4"
    )
    log_dir.mkdir(parents=True, exist_ok=True)

    seed = int(cfg["experiment"]["seed"])
    out: Dict[str, Any] = {
        "protocol": (
            "Frozen dual; IMU features unchanged; only slot one-hot changes. "
            "XYZ Euler per axis Uniform[lo, hi]. random=4.2; swap=4.3; zero=4.4."
        ),
        "pos_checkpoint": str(pos_ckpt),
        "rot_checkpoint": str(rot_ckpt),
        "offset_euler_lo_deg": cfg["data"].get("offset_euler_lo_deg", 0.0),
        "offset_euler_hi_deg": cfg["data"].get("offset_euler_hi_deg", 180.0),
        "note": (
            "Suggestion 2 already trained at Euler-per-axis [0,180], not "
            "axis-angle magnitude 180. Do not rerun unless advisor wants the other 180."
        ),
    }

    def run_ds(ds) -> Dict[str, Any]:
        loader = make_loader(ds, batch_size, num_workers)
        pack = collect_split(
            loader, pos_model, rot_model, pos_mean, pos_std, rot_mean, rot_std, device
        )
        return ablate_pack(pack, rot_model, device, batch_size, seed)

    if args.split in ("val", "both"):
        pack_amass = load_amass_rms_pack(cfg)
        ds = AmassRmsCascadeDataset(
            pack_amass["sequences"],
            pack_amass["val_index"],
            window_len=cfg["data"]["window_len"],
            acc_scale=cfg["data"]["acc_scale"],
            **offset_kwargs_from_cfg(cfg),
            seed=seed,
        )
        print(f"[AMASS val] windows={len(ds)}")
        metrics = run_ds(ds)
        metrics["split"] = "AMASS Val"
        out["val"] = metrics
        print_block("AMASS Val", metrics)

    if args.split in ("test", "both"):
        imu_seqs = load_imuposer_sequences(
            resolve_path(cfg["data"]["processed_imuposer_file"])
        )
        imu_index = build_imuposer_index(
            imu_seqs, cfg["data"]["window_len"], cfg["data"]["test_stride"]
        )
        out["test"] = {}
        for tok in args.combos.split(","):
            tok = tok.strip().lower().replace("+", "_").replace("-", "_")
            w, p = tok.split("_")
            yw, yp = (0 if w == "lw" else 1), (0 if p == "lp" else 1)
            tag = combo_tag(yw, yp)
            label = combo_label(yw, yp)
            ds = ImuposerRmsCascadeDataset(
                imu_seqs,
                imu_index,
                window_len=cfg["data"]["window_len"],
                acc_scale=cfg["data"]["acc_scale"],
                **offset_kwargs_from_cfg(cfg),
                seed=seed,
                y_watch=yw,
                y_phone=yp,
            )
            print(f"[IMUPoser {label}] windows={len(ds)}")
            metrics = run_ds(ds)
            metrics["split"] = f"IMUPoser {label}"
            metrics["combo"] = label
            out["test"][tag] = metrics
            print_block(f"IMUPoser {label}", metrics)

    out_path = log_dir / "metrics_slot_ablation.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print(f"\nsaved {out_path}")

    print("\n=== Q4 slot ablation (R_SB mean °) ===")
    header = (
        f"{'Split':<18} {'None':>8} {'GT-slot':>8} {'PredSeq':>8} "
        f"{'Random':>8} {'Swap':>8} {'Zero':>8}"
    )
    print(header)
    print("-" * len(header))
    rows: List[Tuple[str, Dict[str, Any]]] = []
    if "val" in out:
        rows.append(("AMASS Val", out["val"]))
    for tag in ("lw_lp", "lw_rp", "rw_lp", "rw_rp"):
        if tag in out.get("test", {}):
            rows.append((out["test"][tag]["split"], out["test"][tag]))
    for name, m in rows:
        r = m["rsb_deg"]
        print(
            f"{name:<18} {r['none']['mean']['mean']:8.2f} "
            f"{r['gt_slot']['mean']['mean']:8.2f} "
            f"{r['pred_seq']['mean']['mean']:8.2f} "
            f"{r['random']['mean']['mean']:8.2f} "
            f"{r['swap']['mean']['mean']:8.2f} "
            f"{r['zero']['mean']['mean']:8.2f}"
        )


if __name__ == "__main__":
    main()
