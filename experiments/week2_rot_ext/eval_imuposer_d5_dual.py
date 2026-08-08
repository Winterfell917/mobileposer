#!/usr/bin/env python3
"""
D5 evaluation on IMUPoser for the dual-device (watch+phone) Week2 model.

Same inject protocol as eval_imuposer_d5.py, but:
  - enumerate watch/phone slot combos
  - independently inject R_SB on both devices
  - one forward of RotExtrinsicDualNet recovers both extrinsics

Metrics:
  D5-A: geodesic(R_hat, R_SB_gt) for watch / phone / mean (°)
  D5-B: after calib, mean geodesic(ori_cal, ori_clean); None / Learned / Oracle

Usage (repo root, needs trained dual checkpoint):
  python experiments/week2_rot_ext/eval_imuposer_d5_dual.py \\
      --config experiments/week2_rot_ext/configs/default.yaml
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List

import torch
from tqdm import tqdm

_EXP_DIR = Path(__file__).resolve().parent
if str(_EXP_DIR) not in sys.path:
    sys.path.insert(0, str(_EXP_DIR))

from dataset import (  # noqa: E402
    SLOT_NAMES,
    apply_mount_offset,
    calibrate_with_rsb,
    combo_to_indices,
    geodesic_angle_deg,
    load_config,
    make_device_features,
    resolve_path,
    sample_random_offsets,
    set_seed,
)
from models import RotExtrinsicDualNet  # noqa: E402


def _normalize_feat(x: torch.Tensor, mean: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
    return (x - mean) / std


def _summ(xs: List[float]) -> Dict[str, float]:
    if not xs:
        return {"n": 0, "mean": float("nan"), "median": float("nan")}
    t = torch.tensor(xs, dtype=torch.float32)
    return {
        "n": int(t.numel()),
        "mean": float(t.mean()),
        "median": float(t.median()),
    }


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        default="experiments/week2_rot_ext/configs/default.yaml",
    )
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--max-seqs", type=int, default=None)
    args = parser.parse_args()

    cfg = load_config(resolve_path(args.config))
    set_seed(cfg["experiment"]["seed"])
    gen = torch.Generator().manual_seed(cfg["experiment"]["seed"] + 9)

    ckpt_path = resolve_path(
        args.checkpoint
        or "experiments/week2_rot_ext/outputs/checkpoints/best_rot_err_dual.pt"
    )
    stats_path = resolve_path(cfg["data"]["out_dir"]) / "norm_stats_dual.pt"
    src = resolve_path(cfg["data"]["processed_imuposer_file"])
    for p in (ckpt_path, stats_path, src):
        if not p.exists():
            raise FileNotFoundError(p)

    device = torch.device(
        cfg["train"]["device"] if torch.cuda.is_available() else "cpu"
    )
    stats = torch.load(stats_path, map_location="cpu")
    mean = stats["mean"].float().view(1, -1)
    std = stats["std"].float().view(1, -1)

    model = RotExtrinsicDualNet(
        feat_dim=24,
        n_slots=cfg["data"]["n_slots"],
        n_hidden=cfg["model"]["n_hidden"],
        n_lstm_layers=cfg["model"]["n_lstm_layers"],
        bidirectional=cfg["model"]["bidirectional"],
        dropout=cfg["model"]["dropout"],
        use_slot_onehot=cfg["model"]["use_slot_onehot"],
    ).to(device)
    payload = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(payload["model"])
    model.eval()

    data = torch.load(src, map_location="cpu")
    accs, oris = data["acc"], data["ori"]
    max_seqs = args.max_seqs or int(cfg["eval"].get("imuposer_max_seqs", 40))
    combos = [tuple(c) for c in cfg["eval"]["downstream_combos"]]
    window_len = cfg["data"]["window_len"]
    stride = cfg["data"]["test_stride"]
    acc_scale = float(cfg["data"]["acc_scale"])
    offset_range = float(cfg["data"]["offset_range_deg"])

    a_watch: List[float] = []
    a_phone: List[float] = []
    a_mean: List[float] = []
    per_combo_a: Dict[str, Dict[str, List[float]]] = {}
    per_slot_a: Dict[int, List[float]] = {0: [], 1: [], 2: [], 3: []}

    b_none_w: List[float] = []
    b_none_p: List[float] = []
    b_learned_w: List[float] = []
    b_learned_p: List[float] = []
    b_oracle_w: List[float] = []
    b_oracle_p: List[float] = []

    n_seq = min(len(accs), max_seqs)
    for seq_i in tqdm(range(n_seq), desc="D5 IMUPoser dual"):
        acc_all = accs[seq_i].float()
        ori_all = oris[seq_i].float()
        if acc_all.shape[1] < 4 or ori_all.shape[1] < 4:
            continue
        t_len = acc_all.shape[0]
        if t_len < window_len:
            continue

        for y_watch, y_phone in combos:
            w_idx, p_idx = combo_to_indices(y_watch, y_phone)
            combo_key = f"{SLOT_NAMES[w_idx]}+{SLOT_NAMES[p_idx]}"
            bucket = per_combo_a.setdefault(combo_key, {"w": [], "p": [], "mean": []})

            for start in range(0, t_len - window_len + 1, stride):
                aw = acc_all[start : start + window_len, w_idx]
                ow = ori_all[start : start + window_len, w_idx]
                ap = acc_all[start : start + window_len, p_idx]
                op = ori_all[start : start + window_len, p_idx]

                r_w = sample_random_offsets(1, offset_range, generator=gen)[0]
                r_p = sample_random_offsets(1, offset_range, generator=gen)[0]
                aw_o, ow_o = apply_mount_offset(aw, ow, r_w)
                ap_o, op_o = apply_mount_offset(ap, op, r_p)

                fw = make_device_features(aw_o, ow_o, acc_scale)
                fp = make_device_features(ap_o, op_o, acc_scale)
                feat = torch.cat([fw, fp], dim=-1)
                x = _normalize_feat(feat, mean, std).unsqueeze(0).to(device)
                sw = torch.tensor([w_idx], device=device)
                sp = torch.tensor([p_idx], device=device)
                (_, r_w_hat), (_, r_p_hat) = model(x, sw, sp)
                r_w_hat = r_w_hat[0].cpu()
                r_p_hat = r_p_hat[0].cpu()

                err_w = float(
                    geodesic_angle_deg(r_w_hat.unsqueeze(0), r_w.unsqueeze(0))[0]
                )
                err_p = float(
                    geodesic_angle_deg(r_p_hat.unsqueeze(0), r_p.unsqueeze(0))[0]
                )
                err_m = 0.5 * (err_w + err_p)

                a_watch.append(err_w)
                a_phone.append(err_p)
                a_mean.append(err_m)
                bucket["w"].append(err_w)
                bucket["p"].append(err_p)
                bucket["mean"].append(err_m)
                per_slot_a[w_idx].append(err_w)
                per_slot_a[p_idx].append(err_p)

                # D5-B: ori proxy (pooled over both devices via separate lists)
                b_none_w.append(float(geodesic_angle_deg(ow_o, ow).mean()))
                b_none_p.append(float(geodesic_angle_deg(op_o, op).mean()))

                _, ow_l = calibrate_with_rsb(aw_o, ow_o, r_w_hat)
                _, op_l = calibrate_with_rsb(ap_o, op_o, r_p_hat)
                b_learned_w.append(float(geodesic_angle_deg(ow_l, ow).mean()))
                b_learned_p.append(float(geodesic_angle_deg(op_l, op).mean()))

                _, ow_or = calibrate_with_rsb(aw_o, ow_o, r_w)
                _, op_or = calibrate_with_rsb(ap_o, op_o, r_p)
                b_oracle_w.append(float(geodesic_angle_deg(ow_or, ow).mean()))
                b_oracle_p.append(float(geodesic_angle_deg(op_or, op).mean()))

    def _b_pair(w_list: List[float], p_list: List[float]) -> Dict:
        pooled = w_list + p_list
        return {
            "watch": _summ(w_list),
            "phone": _summ(p_list),
            "pooled": _summ(pooled),
        }

    metrics = {
        "protocol": (
            "Inject independent R_SB onto watch+phone IMUPoser streams; "
            "dual model recovers both; A=extrinsic error; "
            "B=ori error after calib vs clean reference."
        ),
        "week2_mode": "dual_joint",
        "n_sequences_used": n_seq,
        "offset_range_deg": offset_range,
        "combos": [
            f"{SLOT_NAMES[wi]}+{SLOT_NAMES[pi]}"
            for wi, pi in (combo_to_indices(a, b) for a, b in combos)
        ],
        "D5_A_rsb_error_deg": {
            "watch": _summ(a_watch),
            "phone": _summ(a_phone),
            "mean": _summ(a_mean),
        },
        "D5_A_per_slot": {
            SLOT_NAMES[s]: _summ(per_slot_a[s])
            for s in (0, 1, 2, 3)
            if per_slot_a[s]
        },
        "D5_A_per_combo": {
            k: {
                "n": len(v["mean"]),
                "watch_mean": float(torch.tensor(v["w"]).mean()) if v["w"] else float("nan"),
                "phone_mean": float(torch.tensor(v["p"]).mean()) if v["p"] else float("nan"),
                "mean": float(torch.tensor(v["mean"]).mean()) if v["mean"] else float("nan"),
            }
            for k, v in per_combo_a.items()
        },
        "D5_B_ori_proxy_deg": {
            "none": _b_pair(b_none_w, b_none_p),
            "learned": _b_pair(b_learned_w, b_learned_p),
            "oracle": _b_pair(b_oracle_w, b_oracle_p),
        },
        "note": (
            "Native R_SB is not in imuposer_full.pt; inject-GT is the practical "
            "dual D5-A. Combos match eval.downstream_combos (same as D6)."
        ),
    }

    log_dir = resolve_path("experiments/week2_rot_ext/outputs/logs")
    log_dir.mkdir(parents=True, exist_ok=True)
    out_path = log_dir / "metrics_imuposer_d5_dual.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    print(json.dumps(metrics, indent=2))
    print(f"saved {out_path}")


if __name__ == "__main__":
    main()
