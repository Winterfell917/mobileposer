#!/usr/bin/env python3
"""
Week3 step 2: cascade position (step 1) → R_SB (Week2 dual).

Reuse trained Week2 RotExtrinsicDualNet. Slots come from:
  - Pred window: step-1 classifier per window
  - Pred seq:    majority vote of step-1 over the sequence
  - GT slot:     oracle position (Week2 upper bound under this protocol)
  - None:        identity R_SB

Protocol matches step 1: R_MS = R_MB @ R_BS (window-constant), a_M unchanged.

Usage (repo root):
  python experiments/week3_pos_rsb/eval_step2.py \
      --config experiments/week3_pos_rsb/configs/default.yaml
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader
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
)
from models import PosClassifier, RotExtrinsicDualNet  # noqa: E402


def combo_tag(watch_side: int, phone_side: int) -> str:
    return f"{'lw' if watch_side == 0 else 'rw'}_{'lp' if phone_side == 0 else 'rp'}"


def combo_label(watch_side: int, phone_side: int) -> str:
    return f"{'LW' if watch_side == 0 else 'RW'}+{'LP' if phone_side == 0 else 'RP'}"


def _summ(xs: np.ndarray) -> Dict[str, float]:
    if xs.size == 0:
        return {"n": 0, "mean": float("nan"), "median": float("nan")}
    return {
        "n": int(xs.size),
        "mean": float(xs.mean()),
        "median": float(np.median(xs)),
    }


def _pair(w: np.ndarray, p: np.ndarray) -> Dict[str, Any]:
    m = 0.5 * (w + p)
    return {"watch": _summ(w), "phone": _summ(p), "mean": _summ(m)}


def _normalize(x: torch.Tensor, mean: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
    return (x - mean) / std


def majority_slots(
    seq: np.ndarray,
    yw: np.ndarray,
    yp: np.ndarray,
    pw: np.ndarray,
    pp: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    buckets: Dict[Tuple[int, int, int], List[int]] = defaultdict(list)
    for i in range(len(seq)):
        buckets[(int(seq[i]), int(yw[i]), int(yp[i]))].append(i)
    mw = np.array(pw, copy=True)
    mp = np.array(pp, copy=True)
    for idxs in buckets.values():
        preds = [int(pw[i]) * 2 + int(pp[i]) for i in idxs]
        voted = max(set(preds), key=preds.count)
        vw, vp = divmod(voted, 2)
        for i in idxs:
            mw[i] = vw
            mp[i] = vp
    return mw, mp


@torch.no_grad()
def collect_split(
    loader,
    pos_model,
    rot_model,
    pos_mean,
    pos_std,
    rot_mean,
    rot_std,
    device,
) -> Dict[str, np.ndarray]:
    pos_model.eval()
    rot_model.eval()
    eye = torch.eye(3, device=device)
    bags = defaultdict(list)
    for x, yw, yp, rw, rp, seq, start in tqdm(loader, desc="cascade"):
        x = x.to(device)
        yw = yw.to(device)
        yp = yp.to(device)
        rw = rw.to(device)
        rp = rp.to(device)
        x_pos = _normalize(x, pos_mean, pos_std)
        x_rot = _normalize(x, rot_mean, rot_std)
        lw, lp = pos_model(x_pos)
        pw = lw.argmax(dim=-1)
        pp = lp.argmax(dim=-1)
        sw_gt, sp_gt = sides_to_abs_slots(yw, yp)
        sw_pr, sp_pr = sides_to_abs_slots(pw, pp)

        (_, hat_gt_w), (_, hat_gt_p) = rot_model(x_rot, sw_gt, sp_gt)
        (_, hat_pr_w), (_, hat_pr_p) = rot_model(x_rot, sw_pr, sp_pr)

        none_w = geodesic_angle_deg(eye.expand_as(rw), rw)
        none_p = geodesic_angle_deg(eye.expand_as(rp), rp)
        bags["yw"].append(yw.cpu().numpy())
        bags["yp"].append(yp.cpu().numpy())
        bags["pw"].append(pw.cpu().numpy())
        bags["pp"].append(pp.cpu().numpy())
        bags["seq"].append(seq.numpy())
        bags["start"].append(start.numpy())
        bags["none_w"].append(none_w.cpu().numpy())
        bags["none_p"].append(none_p.cpu().numpy())
        bags["gt_w"].append(geodesic_angle_deg(hat_gt_w, rw).cpu().numpy())
        bags["gt_p"].append(geodesic_angle_deg(hat_gt_p, rp).cpu().numpy())
        bags["pred_w"].append(geodesic_angle_deg(hat_pr_w, rw).cpu().numpy())
        bags["pred_p"].append(geodesic_angle_deg(hat_pr_p, rp).cpu().numpy())
        bags["x_rot"].append(x_rot.cpu())
        bags["rw"].append(rw.cpu())
        bags["rp"].append(rp.cpu())
    out = {}
    for k, vs in bags.items():
        if k in ("x_rot", "rw", "rp"):
            out[k] = torch.cat(vs, dim=0)
        else:
            out[k] = np.concatenate(vs)
    return out


@torch.no_grad()
def rot_with_slots(
    rot_model,
    x_rot: torch.Tensor,
    sw: torch.Tensor,
    sp: torch.Tensor,
    rw: torch.Tensor,
    rp: torch.Tensor,
    device,
    batch_size: int,
) -> Tuple[np.ndarray, np.ndarray]:
    rot_model.eval()
    ew, ep = [], []
    n = x_rot.shape[0]
    for i in range(0, n, batch_size):
        xb = x_rot[i : i + batch_size].to(device)
        swb = sw[i : i + batch_size].to(device)
        spb = sp[i : i + batch_size].to(device)
        rwb = rw[i : i + batch_size].to(device)
        rpb = rp[i : i + batch_size].to(device)
        (_, hw), (_, hp) = rot_model(xb, swb, spb)
        ew.append(geodesic_angle_deg(hw, rwb).cpu().numpy())
        ep.append(geodesic_angle_deg(hp, rpb).cpu().numpy())
    return np.concatenate(ew), np.concatenate(ep)


def summarize_pack(pack: Dict[str, Any], seq_w: np.ndarray, seq_p: np.ndarray) -> Dict[str, Any]:
    yw, yp, pw, pp = pack["yw"], pack["yp"], pack["pw"], pack["pp"]
    n = int(len(yw))
    w_ok = yw == pw
    p_ok = yp == pp
    both = w_ok & p_ok
    pos = {
        "n": n,
        "watch_acc": float(w_ok.mean()) if n else float("nan"),
        "phone_acc": float(p_ok.mean()) if n else float("nan"),
        "joint_acc": float(both.mean()) if n else float("nan"),
    }
    rsb = {
        "none": _pair(pack["none_w"], pack["none_p"]),
        "pred_window": _pair(pack["pred_w"], pack["pred_p"]),
        "pred_seq": _pair(seq_w, seq_p),
        "gt_slot": _pair(pack["gt_w"], pack["gt_p"]),
        "oracle": {"watch": {"mean": 0.0, "median": 0.0, "n": n}, "phone": {"mean": 0.0, "median": 0.0, "n": n}, "mean": {"mean": 0.0, "median": 0.0, "n": n}},
    }
    stratified = {
        "joint_correct": {
            "n": int(both.sum()),
            "pred_window": _pair(pack["pred_w"][both], pack["pred_p"][both]) if both.any() else _pair(np.zeros(0), np.zeros(0)),
            "gt_slot": _pair(pack["gt_w"][both], pack["gt_p"][both]) if both.any() else _pair(np.zeros(0), np.zeros(0)),
        },
        "joint_wrong": {
            "n": int((~both).sum()),
            "pred_window": _pair(pack["pred_w"][~both], pack["pred_p"][~both]) if (~both).any() else _pair(np.zeros(0), np.zeros(0)),
            "gt_slot": _pair(pack["gt_w"][~both], pack["gt_p"][~both]) if (~both).any() else _pair(np.zeros(0), np.zeros(0)),
        },
        "watch_correct": {
            "n": int(w_ok.sum()),
            "pred_window_watch": _summ(pack["pred_w"][w_ok]) if w_ok.any() else _summ(np.zeros(0)),
        },
        "phone_correct": {
            "n": int(p_ok.sum()),
            "pred_window_phone": _summ(pack["pred_p"][p_ok]) if p_ok.any() else _summ(np.zeros(0)),
        },
    }
    per_combo: Dict[str, Any] = {}
    for yw_i in (0, 1):
        for yp_i in (0, 1):
            m = (yw == yw_i) & (yp == yp_i)
            if not m.any():
                continue
            per_combo[combo_label(yw_i, yp_i)] = {
                "n": int(m.sum()),
                "joint_acc": float(both[m].mean()),
                "pred_window": _pair(pack["pred_w"][m], pack["pred_p"][m]),
                "pred_seq": _pair(seq_w[m], seq_p[m]),
                "gt_slot": _pair(pack["gt_w"][m], pack["gt_p"][m]),
                "none": _pair(pack["none_w"][m], pack["none_p"][m]),
            }
    return {"pos": pos, "rsb_deg": rsb, "stratified": stratified, "per_combo": per_combo}


def make_loader(ds, batch_size: int, num_workers: int) -> DataLoader:
    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )


def run_dataset(
    ds,
    pos_model,
    rot_model,
    pos_mean,
    pos_std,
    rot_mean,
    rot_std,
    device,
    batch_size: int,
    num_workers: int,
) -> Dict[str, Any]:
    loader = make_loader(ds, batch_size, num_workers)
    pack = collect_split(
        loader, pos_model, rot_model, pos_mean, pos_std, rot_mean, rot_std, device
    )
    mw, mp = majority_slots(pack["seq"], pack["yw"], pack["yp"], pack["pw"], pack["pp"])
    sw = torch.from_numpy(mw.astype(np.int64))
    sp = torch.from_numpy(mp.astype(np.int64)) + 2
    seq_w, seq_p = rot_with_slots(
        rot_model, pack["x_rot"], sw, sp, pack["rw"], pack["rp"], device, batch_size
    )
    metrics = summarize_pack(pack, seq_w, seq_p)
    # drop bulky tensors from return
    return metrics


def print_block(title: str, m: Dict[str, Any]) -> None:
    pos = m["pos"]
    rsb = m["rsb_deg"]
    print(f"\n======== {title} n={pos['n']} ========")
    print(
        f"  pos  watch={pos['watch_acc']:.4f} phone={pos['phone_acc']:.4f} "
        f"joint={pos['joint_acc']:.4f}"
    )
    for name in ("none", "pred_window", "pred_seq", "gt_slot"):
        row = rsb[name]["mean"]
        print(
            f"  R_SB {name:<12} mean={row['mean']:.2f}°  "
            f"watch={rsb[name]['watch']['mean']:.2f}°  "
            f"phone={rsb[name]['phone']['mean']:.2f}°"
        )
    sc = m["stratified"]["joint_correct"]
    sw = m["stratified"]["joint_wrong"]
    if sc["n"]:
        print(
            f"  when joint OK  (n={sc['n']}) pred={sc['pred_window']['mean']['mean']:.2f}°  "
            f"gt_slot={sc['gt_slot']['mean']['mean']:.2f}°"
        )
    if sw["n"]:
        print(
            f"  when joint BAD (n={sw['n']}) pred={sw['pred_window']['mean']['mean']:.2f}°  "
            f"gt_slot={sw['gt_slot']['mean']['mean']:.2f}°"
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
    args = parser.parse_args()

    cfg = load_config(resolve_path(args.config))
    set_seed(cfg["experiment"]["seed"])
    device = torch.device(
        cfg["train"]["device"] if torch.cuda.is_available() else "cpu"
    )
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
        f"rot_ckpt={rot_ckpt} epoch={rot_payload.get('epoch')}"
    )

    batch_size = int(cfg["eval"]["batch_size"])
    num_workers = int(cfg["train"]["num_workers"])
    log_dir = resolve_path(cfg["eval"]["log_dir"])
    log_dir.mkdir(parents=True, exist_ok=True)

    out: Dict[str, Any] = {
        "protocol": (
            "Cascade: step1 Pred slot → Week2 dual R_SB. "
            "R_MS=R_MB@R_BS window-constant, a_M unchanged. "
            "None=I; pred_window=step1 per window; pred_seq=majority slot; "
            "gt_slot=oracle position into Week2; oracle R_SB=0°."
        ),
        "pos_checkpoint": str(pos_ckpt),
        "rot_checkpoint": str(rot_ckpt),
        "offset_range_deg": cfg["data"]["offset_range_deg"],
        "window_len": cfg["data"]["window_len"],
    }

    if args.split in ("val", "both"):
        pack = load_amass_rms_pack(cfg)
        ds = AmassRmsCascadeDataset(
            pack["sequences"],
            pack["val_index"],
            window_len=cfg["data"]["window_len"],
            acc_scale=cfg["data"]["acc_scale"],
            offset_range_deg=cfg["data"]["offset_range_deg"],
            seed=cfg["experiment"]["seed"],
        )
        print(f"[AMASS val] windows={len(ds)}")
        metrics = run_dataset(
            ds, pos_model, rot_model, pos_mean, pos_std, rot_mean, rot_std,
            device, batch_size, num_workers,
        )
        metrics["split"] = "AMASS Val"
        out["val"] = metrics
        print_block("AMASS Val", metrics)
        with open(log_dir / "metrics_step2_val.json", "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2)

    if args.split in ("test", "both"):
        imu_seqs = load_imuposer_sequences(
            resolve_path(cfg["data"]["processed_imuposer_file"])
        )
        imu_index = build_imuposer_index(
            imu_seqs, cfg["data"]["window_len"], cfg["data"]["test_stride"]
        )
        combo_list = []
        for tok in args.combos.split(","):
            tok = tok.strip().lower().replace("+", "_").replace("-", "_")
            w, p = tok.split("_")
            combo_list.append((0 if w == "lw" else 1, 0 if p == "lp" else 1))
        out["test"] = {}
        for yw, yp in combo_list:
            tag = combo_tag(yw, yp)
            label = combo_label(yw, yp)
            ds = ImuposerRmsCascadeDataset(
                imu_seqs,
                imu_index,
                window_len=cfg["data"]["window_len"],
                acc_scale=cfg["data"]["acc_scale"],
                offset_range_deg=cfg["data"]["offset_range_deg"],
                seed=cfg["experiment"]["seed"],
                y_watch=yw,
                y_phone=yp,
            )
            print(f"[IMUPoser {label}] windows={len(ds)}")
            metrics = run_dataset(
                ds, pos_model, rot_model, pos_mean, pos_std, rot_mean, rot_std,
                device, batch_size, num_workers,
            )
            metrics["split"] = f"IMUPoser {label}"
            metrics["combo"] = label
            out["test"][tag] = metrics
            print_block(f"IMUPoser {label}", metrics)
            with open(log_dir / f"metrics_step2_test_{tag}.json", "w", encoding="utf-8") as f:
                json.dump(metrics, f, indent=2)

    out_path = log_dir / "metrics_step2.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print(f"\nsaved {out_path}")

    print("\n=== Step2 Main (R_SB mean °) ===")
    header = f"{'Split':<18} {'None':>8} {'PredWin':>8} {'PredSeq':>8} {'GT-slot':>8}"
    print(header)
    print("-" * len(header))
    rows = []
    if "val" in out:
        rows.append(("AMASS Val", out["val"]))
    for tag in ("lw_lp", "lw_rp", "rw_lp", "rw_rp"):
        if tag in out.get("test", {}):
            rows.append((out["test"][tag]["split"], out["test"][tag]))
    for name, m in rows:
        r = m["rsb_deg"]
        print(
            f"{name:<18} {r['none']['mean']['mean']:8.2f} "
            f"{r['pred_window']['mean']['mean']:8.2f} "
            f"{r['pred_seq']['mean']['mean']:8.2f} "
            f"{r['gt_slot']['mean']['mean']:8.2f}"
        )


if __name__ == "__main__":
    main()
