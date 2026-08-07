#!/usr/bin/env python3
"""
D5 evaluation on IMUPoser: A + B together.

IMUPoser processed streams are typically already calibrated and do NOT store a
native R_SB label. Protocol:

  1) Treat recorded (acc, ori) as the clean reference.
  2) Inject a known random mount R_SB (same convention as training).
  3) Ask the Week2 network to recover R_SB.

Metrics:
  D5-A: geodesic(R_hat, R_SB_gt) in degrees  (true extrinsic error)
  D5-B: after calib, mean geodesic(ori_cal, ori_clean) over the window
        (+ None baseline: geodesic(ori_obs, ori_clean))

Usage (repo root, needs trained checkpoint):
  python experiments/week2_rot_ext/eval_imuposer_d5.py \\
      --config experiments/week2_rot_ext/configs/default.yaml
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
from tqdm import tqdm

_EXP_DIR = Path(__file__).resolve().parent
if str(_EXP_DIR) not in sys.path:
    sys.path.insert(0, str(_EXP_DIR))

from dataset import (  # noqa: E402
    SLOT_NAMES,
    apply_mount_offset,
    calibrate_with_rsb,
    geodesic_angle_deg,
    load_config,
    make_device_features,
    resolve_path,
    sample_random_offsets,
    set_seed,
)
from models import RotExtrinsicNet  # noqa: E402


def _normalize_feat(x: torch.Tensor, mean: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
    return (x - mean) / std


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
    gen = torch.Generator().manual_seed(cfg["experiment"]["seed"] + 7)

    ckpt_path = resolve_path(args.checkpoint or cfg["eval"]["checkpoint"])
    stats_path = resolve_path(cfg["data"]["out_dir"]) / "norm_stats.pt"
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

    model = RotExtrinsicNet(
        feat_dim=cfg["data"]["feat_dim"],
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
    slots = list(cfg["eval"].get("imuposer_slots", [0, 1, 2, 3]))
    window_len = cfg["data"]["window_len"]
    stride = cfg["data"]["test_stride"]
    acc_scale = cfg["data"]["acc_scale"]
    offset_range = float(cfg["data"]["offset_range_deg"])

    # collectors
    a_err = []  # D5-A
    b_none = []
    b_learned = []
    b_oracle = []
    per_slot_a = {s: [] for s in slots}

    n_seq = min(len(accs), max_seqs)
    for seq_i in tqdm(range(n_seq), desc="D5 IMUPoser"):
        acc_all = accs[seq_i].float()
        ori_all = oris[seq_i].float()
        for slot in slots:
            if acc_all.shape[1] <= slot:
                continue
            acc_clean = acc_all[:, slot]
            ori_clean = ori_all[:, slot]
            t_len = acc_clean.shape[0]
            if t_len < window_len:
                continue

            for start in range(0, t_len - window_len + 1, stride):
                r_sb = sample_random_offsets(1, offset_range, generator=gen)[0]
                acc_w = acc_clean[start : start + window_len]
                ori_w = ori_clean[start : start + window_len]
                acc_obs, ori_obs = apply_mount_offset(acc_w, ori_w, r_sb)

                feat = make_device_features(acc_obs, ori_obs, acc_scale)
                x = _normalize_feat(feat, mean, std).unsqueeze(0).to(device)
                slot_t = torch.tensor([slot], device=device)
                _, r_hat = model(x, slot_t)
                r_hat = r_hat[0].cpu()

                # D5-A
                err_a = float(geodesic_angle_deg(r_hat.unsqueeze(0), r_sb.unsqueeze(0))[0])
                a_err.append(err_a)
                per_slot_a[slot].append(err_a)

                # D5-B proxies
                # None: no calib
                b_none.append(
                    float(geodesic_angle_deg(ori_obs, ori_w).mean())
                )
                # Learned
                acc_l, ori_l = calibrate_with_rsb(acc_obs, ori_obs, r_hat)
                b_learned.append(float(geodesic_angle_deg(ori_l, ori_w).mean()))
                # Oracle
                acc_o, ori_o = calibrate_with_rsb(acc_obs, ori_obs, r_sb)
                b_oracle.append(float(geodesic_angle_deg(ori_o, ori_w).mean()))

    def _summ(xs):
        if not xs:
            return {"n": 0, "mean": float("nan"), "median": float("nan")}
        t = torch.tensor(xs, dtype=torch.float32)
        return {
            "n": int(t.numel()),
            "mean": float(t.mean()),
            "median": float(t.median()),
        }

    metrics = {
        "protocol": (
            "Inject known R_SB onto calibrated IMUPoser streams; "
            "A=extrinsic error; B=ori error after calib vs clean reference."
        ),
        "n_sequences_used": n_seq,
        "offset_range_deg": offset_range,
        "D5_A_rsb_error_deg": _summ(a_err),
        "D5_A_per_slot": {
            SLOT_NAMES[s]: _summ(per_slot_a[s]) for s in slots if per_slot_a[s]
        },
        "D5_B_ori_proxy_deg": {
            "none": _summ(b_none),
            "learned": _summ(b_learned),
            "oracle": _summ(b_oracle),
        },
        "note": (
            "Native R_SB is not in imuposer_full.pt; inject-GT is the "
            "practical D5-A. D5-B uses the same windows as an ori-alignment proxy."
        ),
    }

    log_dir = resolve_path("experiments/week2_rot_ext/outputs/logs")
    log_dir.mkdir(parents=True, exist_ok=True)
    out_path = log_dir / "metrics_imuposer_d5.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    print(json.dumps(metrics, indent=2))
    print(f"saved {out_path}")


if __name__ == "__main__":
    main()
