#!/usr/bin/env python3
"""
Build AMASS index + norm stats for position classification under unknown R_BS.

Does NOT write expanded windows (disk-friendly). Training injects
R_MS = R_MB @ R_BS online; a_M unchanged; R_BS constant per window.

Usage (repo root):
  python experiments/week3_pos_rsb/dataset/build_amass.py \\
      --config experiments/week3_pos_rsb/configs/default.yaml
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

_EXP_DIR = Path(__file__).resolve().parents[1]
if str(_EXP_DIR) not in sys.path:
    sys.path.insert(0, str(_EXP_DIR))

from dataset import (  # noqa: E402
    feat_dim_for_mode,
    load_config,
    offset_euler_bounds,
    resolve_path,
    set_seed,
)
from dataset.pos_dataset import (  # noqa: E402
    build_amass_index,
    compute_norm_stats,
    load_amass_sequences,
    split_val_ids,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        default="experiments/week3_pos_rsb/configs/default.yaml",
    )
    parser.add_argument(
        "--feat-mode",
        choices=["full", "acc", "ori"],
        default="full",
        help="full=24d a_M+R_MS; acc=6d; ori=18d",
    )
    parser.add_argument(
        "--stats-out",
        type=str,
        default=None,
        help="write only norm_stats.pt here (keep live amass_index.pt)",
    )
    parser.add_argument(
        "--skip-index",
        action="store_true",
        help="do not rewrite amass_index.pt",
    )
    parser.add_argument(
        "--yaw-align",
        action="store_true",
        help="suggestion 1: sequence-level yaw-only before inject (stats only)",
    )
    args = parser.parse_args()
    cfg = load_config(resolve_path(args.config))
    set_seed(cfg["experiment"]["seed"])

    processed_dir = resolve_path(cfg["data"]["processed_amass_dir"])
    out_dir = resolve_path(cfg["data"]["out_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    subsets = cfg["data"].get("amass_subsets") or []

    print("loading AMASS sequences (acc/ori only)...")
    sequences, _ = load_amass_sequences(processed_dir, subsets)
    val_ids = split_val_ids(
        len(sequences), cfg["data"]["val_ratio"], cfg["experiment"]["seed"]
    )
    train_index, val_index = build_amass_index(
        sequences,
        cfg["data"]["window_len"],
        cfg["data"]["train_stride"],
        val_ids,
    )
    print(
        f"sequences={len(sequences)} val_seqs={len(val_ids)} "
        f"train_windows={train_index.shape[0]} val_windows={val_index.shape[0]}"
    )
    print(
        f"computing norm stats on train windows (online inject, "
        f"feat_mode={args.feat_mode} yaw_align={args.yaw_align})..."
    )
    lo_deg, hi_deg = offset_euler_bounds(cfg["data"])
    stats = compute_norm_stats(
        sequences,
        train_index,
        window_len=cfg["data"]["window_len"],
        acc_scale=cfg["data"]["acc_scale"],
        offset_range_deg=cfg["data"]["offset_range_deg"],
        seed=cfg["experiment"]["seed"],
        lo_deg=lo_deg,
        hi_deg=hi_deg,
        feat_mode=args.feat_mode,
        yaw_align=args.yaw_align,
    )
    if not args.skip_index:
        torch.save(
            {
                "n_sequences": len(sequences),
                "subsets": subsets,
                "train_index": train_index,
                "val_index": val_index,
                "val_ids": sorted(val_ids),
                "offset_euler_lo_deg": lo_deg,
                "offset_euler_hi_deg": hi_deg,
                "protocol": (
                    "R_MS=R_MB@R_BS, a_M unchanged, R_BS unknown, "
                    "window-constant (independent across windows), "
                    f"XYZ Euler per axis Uniform[{lo_deg}, {hi_deg}] deg"
                ),
            },
            out_dir / "amass_index.pt",
        )
        print(f"saved {out_dir / 'amass_index.pt'}")
    stats_path = (
        resolve_path(args.stats_out) if args.stats_out else (out_dir / "norm_stats.pt")
    )
    stats_path.parent.mkdir(parents=True, exist_ok=True)
    dim = feat_dim_for_mode(args.feat_mode)
    torch.save(
        {
            **stats,
            "protocol": (
                f"feat_mode={args.feat_mode} dim={dim}, unknown window-level R_BS, "
                f"XYZ Euler per axis Uniform[{lo_deg}, {hi_deg}] deg, "
                f"yaw_align={args.yaw_align}"
            ),
            "input_dim": dim,
            "feat_mode": args.feat_mode,
            "yaw_align": bool(args.yaw_align),
            "offset_euler_lo_deg": lo_deg,
            "offset_euler_hi_deg": hi_deg,
        },
        stats_path,
    )
    print(f"saved {stats_path}")


if __name__ == "__main__":
    main()
