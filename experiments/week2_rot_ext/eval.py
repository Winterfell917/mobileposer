#!/usr/bin/env python3
"""
Evaluate Week2 R_SB estimator on AMASS val (and optional held-out pt).

Reports:
  - mean geodesic error (°) overall and per slot
  - None / Oracle / Learned sanity on a small resampled batch
    (None = identity assumption; Oracle = GT R_SB; Learned = network)
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
    geodesic_angle_deg,
    load_config,
    resolve_path,
    set_seed,
)
from dataset.rot_dataset import make_loader  # noqa: E402
from models import RotExtrinsicNet  # noqa: E402


@torch.no_grad()
def eval_split(model, loader, device):
    model.eval()
    errs = []
    per_slot = {s: [] for s in range(4)}
    for x, slot, r_sb, _ in tqdm(loader, desc="eval"):
        x = x.to(device)
        slot = slot.to(device)
        r_sb = r_sb.to(device)
        _, pred_r = model(x, slot)
        deg = geodesic_angle_deg(pred_r, r_sb)
        errs.append(deg.cpu())
        for s in slot.unique().tolist():
            per_slot[s].append(deg[slot == s].cpu())
    all_err = torch.cat(errs) if errs else torch.zeros(0)
    out = {
        "n": int(all_err.numel()),
        "rot_err_deg_mean": float(all_err.mean()) if all_err.numel() else float("nan"),
        "rot_err_deg_median": float(all_err.median()) if all_err.numel() else float("nan"),
        "per_slot": {},
    }
    for s, parts in per_slot.items():
        if not parts:
            continue
        e = torch.cat(parts)
        out["per_slot"][SLOT_NAMES[s]] = {
            "n": int(e.numel()),
            "mean": float(e.mean()),
            "median": float(e.median()),
        }
    return out


@torch.no_grad()
def contrast_none_oracle_learned(model, loader, device, max_batches: int = 5):
    """
    Compare extrinsic error under three hypotheses on the same windows:
      None:    assume R_SB = I
      Oracle:  use GT R_SB  (error = 0 by definition)
      Learned: network prediction
    """
    model.eval()
    none_errs, learned_errs = [], []
    n = 0
    for bi, (x, slot, r_sb, _) in enumerate(loader):
        if bi >= max_batches:
            break
        x = x.to(device)
        slot = slot.to(device)
        r_sb = r_sb.to(device)
        eye = torch.eye(3, device=device).expand(r_sb.shape[0], 3, 3)
        _, pred_r = model(x, slot)
        none_errs.append(geodesic_angle_deg(eye, r_sb).cpu())
        learned_errs.append(geodesic_angle_deg(pred_r, r_sb).cpu())
        n += x.size(0)
    none = torch.cat(none_errs) if none_errs else torch.zeros(0)
    learned = torch.cat(learned_errs) if learned_errs else torch.zeros(0)
    return {
        "n_windows": n,
        "none_assume_I_deg_mean": float(none.mean()) if none.numel() else float("nan"),
        "oracle_deg_mean": 0.0,
        "learned_deg_mean": float(learned.mean()) if learned.numel() else float("nan"),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        default="experiments/week2_rot_ext/configs/default.yaml",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="val",
        choices=["val", "train"],
        help="Which AMASS pt to evaluate (synthetic has GT R_SB).",
    )
    parser.add_argument("--checkpoint", type=str, default=None)
    args = parser.parse_args()
    cfg = load_config(resolve_path(args.config))
    set_seed(cfg["experiment"]["seed"])

    data_dir = resolve_path(cfg["data"]["out_dir"])
    pt = data_dir / f"amass_{args.split}.pt"
    stats_pt = data_dir / "norm_stats.pt"
    ckpt_path = resolve_path(args.checkpoint or cfg["eval"]["checkpoint"])
    if not pt.exists():
        raise FileNotFoundError(f"Missing {pt}")
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Missing checkpoint {ckpt_path}")

    device = torch.device(
        cfg["train"]["device"] if torch.cuda.is_available() else "cpu"
    )
    loader = make_loader(
        pt,
        stats_pt,
        batch_size=cfg["eval"]["batch_size"],
        shuffle=False,
        num_workers=cfg["train"]["num_workers"],
    )

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

    metrics = eval_split(model, loader, device)
    contrast = contrast_none_oracle_learned(model, loader, device)
    metrics["contrast"] = contrast

    log_dir = resolve_path("experiments/week2_rot_ext/outputs/logs")
    log_dir.mkdir(parents=True, exist_ok=True)
    out_path = log_dir / f"metrics_{args.split}.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    print(json.dumps(metrics, indent=2))
    print(f"saved {out_path}")


if __name__ == "__main__":
    main()
