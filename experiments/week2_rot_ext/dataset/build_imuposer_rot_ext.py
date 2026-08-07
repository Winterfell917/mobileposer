#!/usr/bin/env python3
"""
Optional: export IMUPoser single-device windows (no native R_SB GT).

Primary D5 evaluation uses eval_imuposer_d5.py (inject-GT protocol).
This builder only dumps calibrated streams for inspection / future use.

Usage:
  python experiments/week2_rot_ext/dataset/build_imuposer_rot_ext.py --all-slots
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
from tqdm import tqdm

_EXP_DIR = Path(__file__).resolve().parents[1]
if str(_EXP_DIR) not in sys.path:
    sys.path.insert(0, str(_EXP_DIR))

from dataset import (  # noqa: E402
    SLOT_NAMES,
    load_config,
    make_device_features,
    resolve_path,
    rotation_matrix_to_r6d,
    set_seed,
    slide_windows,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        default="experiments/week2_rot_ext/configs/default.yaml",
    )
    parser.add_argument("--slot", type=int, default=0)
    parser.add_argument("--all-slots", action="store_true")
    args = parser.parse_args()
    cfg = load_config(resolve_path(args.config))
    set_seed(cfg["experiment"]["seed"])

    src = resolve_path(cfg["data"]["processed_imuposer_file"])
    out_dir = resolve_path(cfg["data"]["out_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    if not src.exists():
        raise FileNotFoundError(src)

    data = torch.load(src, map_location="cpu")
    accs, oris = data["acc"], data["ori"]
    slots = list(range(4)) if args.all_slots else [args.slot]
    window_len = cfg["data"]["window_len"]
    stride = cfg["data"]["test_stride"]
    acc_scale = cfg["data"]["acc_scale"]
    eye = torch.eye(3)

    for slot in slots:
        xs, slots_l, rs, r6s, meta = [], [], [], [], []
        for seq_i, (acc_all, ori_all) in enumerate(
            tqdm(list(zip(accs, oris)), desc=f"slot {SLOT_NAMES[slot]}")
        ):
            if acc_all.shape[1] <= slot:
                continue
            feat = make_device_features(
                acc_all[:, slot].float(), ori_all[:, slot].float(), acc_scale
            )
            for start, win in slide_windows(feat, window_len, stride):
                xs.append(win)
                slots_l.append(slot)
                rs.append(eye.clone())
                r6s.append(rotation_matrix_to_r6d(eye))
                meta.append(
                    {
                        "seq": seq_i,
                        "slot": slot,
                        "slot_name": SLOT_NAMES[slot],
                        "start": start,
                        "note": "calibrated_stream_r_sb_I_placeholder; use eval_imuposer_d5.py",
                    }
                )
        if not xs:
            continue
        out = {
            "x": torch.stack(xs).float(),
            "slot": torch.tensor(slots_l, dtype=torch.long),
            "r_sb": torch.stack(rs).float(),
            "r_sb_6d": torch.stack(r6s).float(),
            "meta": meta,
        }
        path = out_dir / f"imuposer_slot_{SLOT_NAMES[slot].lower()}.pt"
        torch.save(out, path)
        print(f"saved {path}: {out['x'].shape}")


if __name__ == "__main__":
    main()
