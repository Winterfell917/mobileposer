#!/usr/bin/env python3
"""
Build AMASS train/val windows for Week2 single-device R_SB estimation.

For each sequence and each configured slot:
  - take bone-aligned IMU (acc, ori)
  - sample random R_SB (offset_per=window|sequence)
  - apply mount offset on ori only (acc unchanged) → observed IMU
  - cut sliding windows; label = R_SB (matrix + 6D)

Usage (repo root):
  python experiments/week2_rot_ext/dataset/build_amass_rot_ext.py \\
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
    load_config,
    make_device_features,
    offset_euler_bounds,
    resolve_path,
    rotation_matrix_to_r6d,
    sample_random_offsets,
    set_seed,
    slide_windows,
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
    train_slots: List[int],
    val_ids: set,
    gen: torch.Generator,
    lo_deg: float = 0.0,
    hi_deg: float | None = None,
) -> Dict[str, Dict[str, list]]:
    data = torch.load(path, map_location="cpu")
    accs, oris = data["acc"], data["ori"]
    assert len(accs) == len(oris)

    buckets = {
        "train": {
            "x": [],
            "slot": [],
            "r_sb": [],
            "r_sb_6d": [],
            "meta": [],
        },
        "val": {
            "x": [],
            "slot": [],
            "r_sb": [],
            "r_sb_6d": [],
            "meta": [],
        },
    }

    for local_i, (acc_all, ori_all) in enumerate(zip(accs, oris)):
        if acc_all.shape[1] < 4 or ori_all.shape[1] < 4:
            continue
        acc_all = acc_all[:, :4].float()
        ori_all = ori_all[:, :4].float()
        split = "val" if local_i in val_ids else "train"

        for slot in train_slots:
            acc_bone = acc_all[:, slot]  # [T, 3]
            ori_bone = ori_all[:, slot]  # [T, 3, 3]
            t_len = acc_bone.shape[0]
            if t_len < window_len:
                continue

            r_seq = None
            if offset_per == "sequence":
                r_seq = sample_random_offsets(
                    1, offset_range_deg, generator=gen, lo_deg=lo_deg, hi_deg=hi_deg
                )[0]

            # Build full-length observed stream if sequence-level offset
            if r_seq is not None:
                acc_obs, ori_obs = apply_mount_offset(acc_bone, ori_bone, r_seq)
                feat = make_device_features(acc_obs, ori_obs, acc_scale)
                for start, win in slide_windows(feat, window_len, stride):
                    buckets[split]["x"].append(win)
                    buckets[split]["slot"].append(slot)
                    buckets[split]["r_sb"].append(r_seq.clone())
                    buckets[split]["r_sb_6d"].append(rotation_matrix_to_r6d(r_seq))
                    buckets[split]["meta"].append(
                        {
                            "source": path.name,
                            "seq_local": local_i,
                            "slot": slot,
                            "slot_name": SLOT_NAMES[slot],
                            "start": start,
                            "offset_per": offset_per,
                        }
                    )
            else:
                # per-window offset
                for start in range(0, t_len - window_len + 1, stride):
                    r_sb = sample_random_offsets(
                        1, offset_range_deg, generator=gen, lo_deg=lo_deg, hi_deg=hi_deg
                    )[0]
                    acc_w = acc_bone[start : start + window_len]
                    ori_w = ori_bone[start : start + window_len]
                    acc_obs, ori_obs = apply_mount_offset(acc_w, ori_w, r_sb)
                    feat = make_device_features(acc_obs, ori_obs, acc_scale)
                    buckets[split]["x"].append(feat)
                    buckets[split]["slot"].append(slot)
                    buckets[split]["r_sb"].append(r_sb.clone())
                    buckets[split]["r_sb_6d"].append(rotation_matrix_to_r6d(r_sb))
                    buckets[split]["meta"].append(
                        {
                            "source": path.name,
                            "seq_local": local_i,
                            "slot": slot,
                            "slot_name": SLOT_NAMES[slot],
                            "start": start,
                            "offset_per": offset_per,
                        }
                    )
    return buckets


def _stack_bucket(bucket: Dict[str, list]) -> Dict[str, torch.Tensor | list]:
    if not bucket["x"]:
        return {
            "x": torch.zeros(0, 1, 12),
            "slot": torch.zeros(0, dtype=torch.long),
            "r_sb": torch.zeros(0, 3, 3),
            "r_sb_6d": torch.zeros(0, 6),
            "meta": [],
        }
    return {
        "x": torch.stack(bucket["x"], dim=0).float(),
        "slot": torch.tensor(bucket["slot"], dtype=torch.long),
        "r_sb": torch.stack(bucket["r_sb"], dim=0).float(),
        "r_sb_6d": torch.stack(bucket["r_sb_6d"], dim=0).float(),
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
    gen = torch.Generator().manual_seed(cfg["experiment"]["seed"])

    processed_dir = resolve_path(cfg["data"]["processed_amass_dir"])
    out_dir = resolve_path(cfg["data"]["out_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    files = _list_amass_files(processed_dir, cfg["data"].get("amass_subsets") or [])
    if not files:
        raise FileNotFoundError(
            f"No AMASS .pt under {processed_dir}. "
            "Run mobileposer data_process_mocap.py --dataset amass first."
        )

    seq_counts = []
    for f in files:
        d = torch.load(f, map_location="cpu")
        seq_counts.append(len(d["acc"]))
    total_seq = sum(seq_counts)
    # global val ids then remap per file
    val_ids_global = _split_by_sequence(
        total_seq, cfg["data"]["val_ratio"], cfg["experiment"]["seed"]
    )
    print(
        f"AMASS files={len(files)}, sequences={total_seq}, "
        f"val_seqs={len(val_ids_global)}, "
        f"offset_range=[{cfg['data'].get('offset_euler_lo_deg', 0)},"
        f"{cfg['data'].get('offset_euler_hi_deg', cfg['data']['offset_range_deg'])}]deg, "
        f"offset_per={cfg['data']['offset_per']}"
    )

    train_b = {"x": [], "slot": [], "r_sb": [], "r_sb_6d": [], "meta": []}
    val_b = {"x": [], "slot": [], "r_sb": [], "r_sb_6d": [], "meta": []}

    offset = 0
    train_slots = list(cfg["data"]["train_slots"])
    for f, n_seq in tqdm(list(zip(files, seq_counts)), desc="build_amass_rot_ext"):
        local_val = {i - offset for i in range(offset, offset + n_seq) if i in val_ids_global}
        part = process_file(
            f,
            window_len=cfg["data"]["window_len"],
            stride=cfg["data"]["train_stride"],
            acc_scale=cfg["data"]["acc_scale"],
            offset_range_deg=float(cfg["data"]["offset_range_deg"]),
            offset_per=str(cfg["data"]["offset_per"]),
            train_slots=train_slots,
            val_ids=local_val,
            gen=gen,
            lo_deg=offset_euler_bounds(cfg["data"])[0],
            hi_deg=offset_euler_bounds(cfg["data"])[1],
        )
        for k in train_b:
            train_b[k].extend(part["train"][k])
            val_b[k].extend(part["val"][k])
        offset += n_seq

    train = _stack_bucket(train_b)
    val = _stack_bucket(val_b)

    # Normalize only acc channels (0:3); leave ori as-is (mean0/std1)
    if train["x"].numel() > 0:
        flat = train["x"].reshape(-1, train["x"].shape[-1])
        mean = torch.zeros(flat.shape[-1])
        std = torch.ones(flat.shape[-1])
        mean[:3] = flat[:, :3].mean(dim=0)
        std[:3] = flat[:, :3].std(dim=0).clamp_min(1e-6)
    else:
        mean = torch.zeros(12)
        std = torch.ones(12)

    torch.save(train, out_dir / "amass_train.pt")
    torch.save(val, out_dir / "amass_val.pt")
    torch.save(
        {
            "mean": mean,
            "std": std,
            "acc_channels": [0, 1, 2],
            "feat_dim": 12,
            "offset_range_deg": cfg["data"]["offset_range_deg"],
            "offset_euler_lo_deg": cfg["data"].get("offset_euler_lo_deg", 0.0),
            "offset_euler_hi_deg": cfg["data"].get(
                "offset_euler_hi_deg", cfg["data"]["offset_range_deg"]
            ),
            "convention": "R_obs = R_bone @ R_SB; a_obs = R_SB^T @ a_bone",
        },
        out_dir / "norm_stats.pt",
    )

    print(f"train windows: {train['x'].shape}")
    print(f"val windows:   {val['x'].shape}")
    if train["slot"].numel() > 0:
        for s in sorted(set(train["slot"].tolist())):
            n = int((train["slot"] == s).sum())
            print(f"  train slot {s} ({SLOT_NAMES[s]}): {n}")
    print(f"saved to {out_dir}")


if __name__ == "__main__":
    main()
