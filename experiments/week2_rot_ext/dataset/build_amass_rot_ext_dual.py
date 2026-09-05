#!/usr/bin/env python3
"""
Build AMASS train/val windows for Week2 *dual-device* R_SB joint training.

Each sample:
  - watch + phone IMU features [T, 24] = watch(12)+phone(12)
  - slots (absolute): watch in {0,1}, phone in {2,3}
  - independent R_SB for watch and phone (GT)

Enumerates 4 placement combos (LW/RW × LP/RP), same as Week1.

Usage (repo root):
  python experiments/week2_rot_ext/dataset/build_amass_rot_ext_dual.py \\
      --config experiments/week2_rot_ext/configs/default.yaml
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, List

import torch
from tqdm import tqdm

_EXP_DIR = Path(__file__).resolve().parents[1]
if str(_EXP_DIR) not in sys.path:
    sys.path.insert(0, str(_EXP_DIR))

from dataset import (  # noqa: E402
    SLOT_NAMES,
    apply_mount_offset,
    combo_to_indices,
    load_config,
    make_device_features,
    offset_euler_bounds,
    resolve_path,
    rotation_matrix_to_r6d,
    sample_random_offsets,
    set_seed,
)


def _list_amass_files(processed_dir: Path, subsets: List[str]) -> List[Path]:
    if subsets:
        files = []
        for name in subsets:
            p = processed_dir / f"{name}.pt"
            if p.exists():
                files.append(p)
            else:
                print(f"[warn] missing subset: {p}")
        return files
    return sorted(processed_dir.glob("*.pt"))


def _split_by_sequence(n_seq: int, val_ratio: float, seed: int):
    g = torch.Generator().manual_seed(seed)
    perm = torch.randperm(n_seq, generator=g).tolist()
    n_val = max(1, int(round(n_seq * val_ratio))) if n_seq > 1 else 0
    return set(perm[:n_val])


def process_file(
    path: Path,
    window_len: int,
    stride: int,
    acc_scale: float,
    offset_range_deg: float,
    offset_per: str,
    val_ids: set,
    gen: torch.Generator,
    lo_deg: float = 0.0,
    hi_deg: float | None = None,
) -> Dict[str, Dict[str, list]]:
    data = torch.load(path, map_location="cpu")
    accs, oris = data["acc"], data["ori"]

    keys = (
        "x",
        "slot_watch",
        "slot_phone",
        "r_sb_watch",
        "r_sb_phone",
        "r_sb_watch_6d",
        "r_sb_phone_6d",
        "meta",
    )
    buckets = {
        "train": {k: [] for k in keys},
        "val": {k: [] for k in keys},
    }

    for local_i, (acc_all, ori_all) in enumerate(zip(accs, oris)):
        if acc_all.shape[1] < 4 or ori_all.shape[1] < 4:
            continue
        acc_all = acc_all[:, :4].float()
        ori_all = ori_all[:, :4].float()
        split = "val" if local_i in val_ids else "train"
        t_len = acc_all.shape[0]
        if t_len < window_len:
            continue

        for y_watch in (0, 1):
            for y_phone in (0, 1):
                w_idx, p_idx = combo_to_indices(y_watch, y_phone)
                aw_full = acc_all[:, w_idx]
                ow_full = ori_all[:, w_idx]
                ap_full = acc_all[:, p_idx]
                op_full = ori_all[:, p_idx]

                if offset_per == "sequence":
                    r_w = sample_random_offsets(
                        1, offset_range_deg, generator=gen, lo_deg=lo_deg, hi_deg=hi_deg
                    )[0]
                    r_p = sample_random_offsets(
                        1, offset_range_deg, generator=gen, lo_deg=lo_deg, hi_deg=hi_deg
                    )[0]
                    aw_o, ow_o = apply_mount_offset(aw_full, ow_full, r_w)
                    ap_o, op_o = apply_mount_offset(ap_full, op_full, r_p)
                    feat_w = make_device_features(aw_o, ow_o, acc_scale)
                    feat_p = make_device_features(ap_o, op_o, acc_scale)
                    feat = torch.cat([feat_w, feat_p], dim=-1)  # [T,24]
                    for start in range(0, t_len - window_len + 1, stride):
                        win = feat[start : start + window_len].clone()
                        buckets[split]["x"].append(win)
                        buckets[split]["slot_watch"].append(w_idx)
                        buckets[split]["slot_phone"].append(p_idx)
                        buckets[split]["r_sb_watch"].append(r_w.clone())
                        buckets[split]["r_sb_phone"].append(r_p.clone())
                        buckets[split]["r_sb_watch_6d"].append(rotation_matrix_to_r6d(r_w))
                        buckets[split]["r_sb_phone_6d"].append(rotation_matrix_to_r6d(r_p))
                        buckets[split]["meta"].append(
                            {
                                "source": path.name,
                                "seq_local": local_i,
                                "y_watch": y_watch,
                                "y_phone": y_phone,
                                "watch_imu": w_idx,
                                "phone_imu": p_idx,
                                "start": start,
                                "offset_per": offset_per,
                            }
                        )
                else:
                    for start in range(0, t_len - window_len + 1, stride):
                        r_w = sample_random_offsets(
                        1, offset_range_deg, generator=gen, lo_deg=lo_deg, hi_deg=hi_deg
                    )[0]
                        r_p = sample_random_offsets(
                        1, offset_range_deg, generator=gen, lo_deg=lo_deg, hi_deg=hi_deg
                    )[0]
                        aw = aw_full[start : start + window_len]
                        ow = ow_full[start : start + window_len]
                        ap = ap_full[start : start + window_len]
                        op = op_full[start : start + window_len]
                        aw_o, ow_o = apply_mount_offset(aw, ow, r_w)
                        ap_o, op_o = apply_mount_offset(ap, op, r_p)
                        feat_w = make_device_features(aw_o, ow_o, acc_scale)
                        feat_p = make_device_features(ap_o, op_o, acc_scale)
                        win = torch.cat([feat_w, feat_p], dim=-1)
                        buckets[split]["x"].append(win)
                        buckets[split]["slot_watch"].append(w_idx)
                        buckets[split]["slot_phone"].append(p_idx)
                        buckets[split]["r_sb_watch"].append(r_w.clone())
                        buckets[split]["r_sb_phone"].append(r_p.clone())
                        buckets[split]["r_sb_watch_6d"].append(rotation_matrix_to_r6d(r_w))
                        buckets[split]["r_sb_phone_6d"].append(rotation_matrix_to_r6d(r_p))
                        buckets[split]["meta"].append(
                            {
                                "source": path.name,
                                "seq_local": local_i,
                                "y_watch": y_watch,
                                "y_phone": y_phone,
                                "watch_imu": w_idx,
                                "phone_imu": p_idx,
                                "start": start,
                                "offset_per": "window",
                            }
                        )
    return buckets


def _stack_bucket(bucket: Dict[str, list]) -> Dict[str, torch.Tensor | list]:
    if not bucket["x"]:
        return {
            "x": torch.zeros(0, 1, 24),
            "slot_watch": torch.zeros(0, dtype=torch.long),
            "slot_phone": torch.zeros(0, dtype=torch.long),
            "r_sb_watch": torch.zeros(0, 3, 3),
            "r_sb_phone": torch.zeros(0, 3, 3),
            "r_sb_watch_6d": torch.zeros(0, 6),
            "r_sb_phone_6d": torch.zeros(0, 6),
            "meta": [],
        }
    return {
        "x": torch.stack(bucket["x"], dim=0).float(),
        "slot_watch": torch.tensor(bucket["slot_watch"], dtype=torch.long),
        "slot_phone": torch.tensor(bucket["slot_phone"], dtype=torch.long),
        "r_sb_watch": torch.stack(bucket["r_sb_watch"], dim=0).float(),
        "r_sb_phone": torch.stack(bucket["r_sb_phone"], dim=0).float(),
        "r_sb_watch_6d": torch.stack(bucket["r_sb_watch_6d"], dim=0).float(),
        "r_sb_phone_6d": torch.stack(bucket["r_sb_phone_6d"], dim=0).float(),
        "meta": bucket["meta"],
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        default="experiments/week2_rot_ext/configs/default.yaml",
    )
    args = parser.parse_args()
    cfg = load_config(resolve_path(args.config))
    set_seed(cfg["experiment"]["seed"])
    gen = torch.Generator().manual_seed(cfg["experiment"]["seed"] + 100)

    processed_dir = resolve_path(cfg["data"]["processed_amass_dir"])
    out_dir = resolve_path(cfg["data"]["out_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    files = _list_amass_files(processed_dir, cfg["data"].get("amass_subsets") or [])
    if not files:
        raise FileNotFoundError(f"No AMASS .pt under {processed_dir}")

    seq_counts = [len(torch.load(f, map_location="cpu")["acc"]) for f in files]
    total_seq = sum(seq_counts)
    val_ids_global = _split_by_sequence(
        total_seq, cfg["data"]["val_ratio"], cfg["experiment"]["seed"]
    )
    lo_deg, hi_deg = offset_euler_bounds(cfg["data"])
    print(
        f"[dual] AMASS files={len(files)}, sequences={total_seq}, "
        f"val_seqs={len(val_ids_global)}, "
        f"euler=[{lo_deg},{hi_deg}] offset_per={cfg['data']['offset_per']}"
    )

    keys = (
        "x",
        "slot_watch",
        "slot_phone",
        "r_sb_watch",
        "r_sb_phone",
        "r_sb_watch_6d",
        "r_sb_phone_6d",
        "meta",
    )
    train_b = {k: [] for k in keys}
    val_b = {k: [] for k in keys}

    offset = 0
    for f, n_seq in tqdm(list(zip(files, seq_counts)), desc="build_amass_dual"):
        local_val = {
            i - offset for i in range(offset, offset + n_seq) if i in val_ids_global
        }
        part = process_file(
            f,
            window_len=cfg["data"]["window_len"],
            stride=cfg["data"]["train_stride"],
            acc_scale=cfg["data"]["acc_scale"],
            offset_range_deg=float(cfg["data"]["offset_range_deg"]),
            offset_per=str(cfg["data"]["offset_per"]),
            val_ids=local_val,
            gen=gen,
            lo_deg=lo_deg,
            hi_deg=hi_deg,
        )
        for k in keys:
            train_b[k].extend(part["train"][k])
            val_b[k].extend(part["val"][k])
        offset += n_seq

    train = _stack_bucket(train_b)
    val = _stack_bucket(val_b)

    # Normalize acc channels of watch (0:3) and phone (12:15)
    if train["x"].numel() > 0:
        flat = train["x"].reshape(-1, train["x"].shape[-1])
        mean = torch.zeros(flat.shape[-1])
        std = torch.ones(flat.shape[-1])
        for sl in (slice(0, 3), slice(12, 15)):
            mean[sl] = flat[:, sl].mean(dim=0)
            std[sl] = flat[:, sl].std(dim=0).clamp_min(1e-6)
    else:
        mean = torch.zeros(24)
        std = torch.ones(24)

    torch.save(train, out_dir / "amass_dual_train.pt")
    torch.save(val, out_dir / "amass_dual_val.pt")
    torch.save(
        {
            "mean": mean,
            "std": std,
            "acc_channels": [0, 1, 2, 12, 13, 14],
            "feat_dim": 24,
            "mode": "dual",
            "offset_range_deg": cfg["data"]["offset_range_deg"],
            "offset_euler_lo_deg": lo_deg,
            "offset_euler_hi_deg": hi_deg,
            "convention": "R_obs = R_bone @ R_SB; a_obs = a_bone",
        },
        out_dir / "norm_stats_dual.pt",
    )

    print(f"train windows: {train['x'].shape}")
    print(f"val windows:   {val['x'].shape}")
    if train["slot_watch"].numel() > 0:
        for yw, yp in ((0, 0), (0, 1), (1, 0), (1, 1)):
            w_idx, p_idx = combo_to_indices(yw, yp)
            m = (train["slot_watch"] == w_idx) & (train["slot_phone"] == p_idx)
            print(
                f"  train {SLOT_NAMES[w_idx]}+{SLOT_NAMES[p_idx]}: {int(m.sum())}"
            )
    print(f"saved dual data to {out_dir}")


if __name__ == "__main__":
    main()
