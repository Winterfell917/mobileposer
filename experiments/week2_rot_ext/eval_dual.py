#!/usr/bin/env python3
"""Evaluate dual-device R_SB model on AMASS dual val set."""
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
    geodesic_angle_deg,
    load_config,
    resolve_path,
    set_seed,
)
from dataset.rot_dataset import make_dual_loader  # noqa: E402
from models import RotExtrinsicDualNet  # noqa: E402


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        default="experiments/week2_rot_ext/configs/default.yaml",
    )
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--split", type=str, default="val", choices=["val", "train"])
    args = parser.parse_args()
    cfg = load_config(resolve_path(args.config))
    set_seed(cfg["experiment"]["seed"])

    data_dir = resolve_path(cfg["data"]["out_dir"])
    pt = data_dir / f"amass_dual_{args.split}.pt"
    stats = data_dir / "norm_stats_dual.pt"
    ckpt = resolve_path(
        args.checkpoint
        or "experiments/week2_rot_ext/outputs/checkpoints/best_rot_err_dual.pt"
    )
    for p in (pt, stats, ckpt):
        if not p.exists():
            raise FileNotFoundError(p)

    device = torch.device(
        cfg["train"]["device"] if torch.cuda.is_available() else "cpu"
    )
    loader = make_dual_loader(
        pt,
        stats,
        batch_size=cfg["eval"]["batch_size"],
        shuffle=False,
        num_workers=cfg["train"]["num_workers"],
    )

    model = RotExtrinsicDualNet(
        feat_dim=24,
        n_slots=cfg["data"]["n_slots"],
        n_hidden=cfg["model"]["n_hidden"],
        n_lstm_layers=cfg["model"]["n_lstm_layers"],
        bidirectional=cfg["model"]["bidirectional"],
        dropout=cfg["model"]["dropout"],
        use_slot_onehot=cfg["model"]["use_slot_onehot"],
    ).to(device)
    model.load_state_dict(torch.load(ckpt, map_location=device)["model"])
    model.eval()

    errs_w, errs_p = [], []
    per_combo = {}
    eye = torch.eye(3, device=device)
    none_w = none_p = 0.0
    n_none = 0

    for x, sw, sp, rw, rp, _, _ in tqdm(loader, desc="eval_dual"):
        x = x.to(device)
        sw, sp = sw.to(device), sp.to(device)
        rw, rp = rw.to(device), rp.to(device)
        (_, prw), (_, prp) = model(x, sw, sp)
        dw = geodesic_angle_deg(prw, rw)
        dp = geodesic_angle_deg(prp, rp)
        errs_w.append(dw.cpu())
        errs_p.append(dp.cpu())
        none_w += geodesic_angle_deg(eye.expand_as(rw), rw).sum().item()
        none_p += geodesic_angle_deg(eye.expand_as(rp), rp).sum().item()
        n_none += x.size(0)
        for i in range(x.size(0)):
            key = f"{SLOT_NAMES[int(sw[i])]}+{SLOT_NAMES[int(sp[i])]}"
            per_combo.setdefault(key, {"w": [], "p": []})
            per_combo[key]["w"].append(float(dw[i]))
            per_combo[key]["p"].append(float(dp[i]))

    ew = torch.cat(errs_w)
    ep = torch.cat(errs_p)
    metrics = {
        "n": int(ew.numel()),
        "watch_rot_err_deg_mean": float(ew.mean()),
        "watch_rot_err_deg_median": float(ew.median()),
        "phone_rot_err_deg_mean": float(ep.mean()),
        "phone_rot_err_deg_median": float(ep.median()),
        "mean_rot_err_deg": float(0.5 * (ew.mean() + ep.mean())),
        "contrast": {
            "none_watch_deg": none_w / max(n_none, 1),
            "none_phone_deg": none_p / max(n_none, 1),
            "oracle_deg": 0.0,
        },
        "per_combo": {},
    }
    for k, v in per_combo.items():
        tw = torch.tensor(v["w"])
        tp = torch.tensor(v["p"])
        metrics["per_combo"][k] = {
            "n": len(v["w"]),
            "watch_mean": float(tw.mean()),
            "phone_mean": float(tp.mean()),
        }

    log_dir = resolve_path("experiments/week2_rot_ext/outputs/logs")
    log_dir.mkdir(parents=True, exist_ok=True)
    out = log_dir / f"metrics_dual_{args.split}.json"
    with open(out, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    print(json.dumps(metrics, indent=2))
    print(f"saved {out}")


if __name__ == "__main__":
    main()
