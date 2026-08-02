#!/usr/bin/env python3
"""
Build IMUPoser test samples for Week1 position classification.

IMPORTANT:
  Real recordings have a fixed wearing config. This script CANNOT invent GT labels.
  You must provide --watch-side and --phone-side for the subset you evaluate,
  or supply a label map JSON.

Usage:
  python experiments/week1_pos_cls/dataset/build_imuposer_pos_cls.py \\
      --config experiments/week1_pos_cls/configs/default.yaml \\
      --watch-side 0 --phone-side 1
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional

import torch
from tqdm import tqdm

_EXP_DIR = Path(__file__).resolve().parents[1]
if str(_EXP_DIR) not in sys.path:
    sys.path.insert(0, str(_EXP_DIR))

from dataset import (  # noqa: E402
    combo_to_indices,
    load_config,
    make_window_features,
    resolve_path,
    set_seed,
    slide_windows,
)


def _load_labels(path: Optional[Path]) -> Optional[Dict[str, Dict[str, int]]]:
    if path is None:
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        default="experiments/week1_pos_cls/configs/default.yaml",
    )
    parser.add_argument("--watch-side", type=int, choices=[0, 1], default=None,
                        help="0=LW, 1=RW (global default if no label map)")
    parser.add_argument("--phone-side", type=int, choices=[0, 1], default=None,
                        help="0=LP, 1=RP (global default if no label map)")
    parser.add_argument(
        "--label-map",
        type=str,
        default=None,
        help="Optional JSON: {seq_key: {watch:0/1, phone:0/1}}",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output filename under data.out_dir "
        "(default: imuposer_test_{lw|rw}_{lp|rp}.pt)",
    )
    args = parser.parse_args()
    cfg = load_config(resolve_path(args.config))
    set_seed(cfg["experiment"]["seed"])

    src = resolve_path(cfg["data"]["processed_imuposer_file"])
    out_dir = resolve_path(cfg["data"]["out_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    if not src.exists():
        raise FileNotFoundError(
            f"IMUPoser processed file not found: {src}. "
            "Run data_process_mocap.py --dataset imuposer and update config."
        )

    label_map = _load_labels(Path(args.label_map) if args.label_map else None)
    if label_map is None and (args.watch_side is None or args.phone_side is None):
        raise SystemExit(
            "Provide --watch-side/--phone-side OR --label-map. "
            "Do not guess IMUPoser wearing labels."
        )

    data = torch.load(src, map_location="cpu")
    accs, oris = data["acc"], data["ori"]

    xs, yw, yp, metas = [], [], [], []
    for i, (acc, ori) in enumerate(tqdm(list(zip(accs, oris)), desc="build_imuposer")):
        key = str(i)
        if label_map is not None:
            if key not in label_map:
                continue
            y_watch = int(label_map[key]["watch"])
            y_phone = int(label_map[key]["phone"])
        else:
            y_watch, y_phone = args.watch_side, args.phone_side

        if acc.shape[1] < 4 or ori.shape[1] < 4:
            continue
        acc = acc[:, :4].float()
        ori = ori[:, :4].float()
        w_idx, p_idx = combo_to_indices(y_watch, y_phone)
        feat = make_window_features(
            acc,
            ori,
            w_idx,
            p_idx,
            fps=cfg["data"]["fps"],
            acc_scale=cfg["data"]["acc_scale"],
        )
        wins = slide_windows(feat, cfg["data"]["window_len"], cfg["data"]["test_stride"])
        for j, win in enumerate(wins):
            xs.append(win)
            yw.append(y_watch)
            yp.append(y_phone)
            metas.append(
                {
                    "seq": i,
                    "start": j * cfg["data"]["test_stride"],
                    "y_watch": y_watch,
                    "y_phone": y_phone,
                    "watch_imu": w_idx,
                    "phone_imu": p_idx,
                }
            )

    if not xs:
        raise RuntimeError("No IMUPoser windows produced. Check labels / sequence lengths.")

    out = {
        "x": torch.stack(xs, dim=0).float(),
        "y_watch": torch.tensor(yw, dtype=torch.long),
        "y_phone": torch.tensor(yp, dtype=torch.long),
        "meta": metas,
    }
    if args.output:
        out_name = args.output
    elif args.watch_side is not None and args.phone_side is not None:
        w_name = "lw" if args.watch_side == 0 else "rw"
        p_name = "lp" if args.phone_side == 0 else "rp"
        out_name = f"imuposer_test_{w_name}_{p_name}.pt"
    else:
        out_name = "imuposer_test.pt"
    out_path = out_dir / out_name
    torch.save(out, out_path)
    print(f"test windows: {out['x'].shape}")
    print(f"combo: watch={yw[0]} phone={yp[0]} -> {out_path}")


if __name__ == "__main__":
    main()
