#!/usr/bin/env python3
"""
Build AMASS training/validation samples for Week1 position classification.

For each AMASS sequence, enumerate 4 placement combos:
  watch: LW(0)/RW(1), phone: LP(0)/RP(1)
and cut sliding windows of features [W, 12].

Usage (from repo root):
  python experiments/week1_pos_cls/dataset/build_amass_pos_cls.py \\
      --config experiments/week1_pos_cls/configs/default.yaml
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, List

import torch
from tqdm import tqdm

# Allow importing sibling helpers when run as a script
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
    val_ids = set(perm[:n_val])
    return val_ids


def process_file(
    path: Path,
    window_len: int,
    stride: int,
    fps: float,
    acc_scale: float,
    val_ids: set,
    seq_offset: int,
) -> Dict[str, Dict[str, list]]:
    data = torch.load(path, map_location="cpu")
    accs, oris = data["acc"], data["ori"]
    assert len(accs) == len(oris)

    buckets = {
        "train": {"x": [], "y_watch": [], "y_phone": [], "meta": []},
        "val": {"x": [], "y_watch": [], "y_phone": [], "meta": []},
    }

    for local_i, (acc, ori) in enumerate(zip(accs, oris)):
        # AMASS synth usually has 6 IMUs; keep at least 4
        if acc.shape[1] < 4 or ori.shape[1] < 4:
            continue
        acc = acc[:, :4].float()
        ori = ori[:, :4].float()
        split = "val" if (seq_offset + local_i) in val_ids else "train"

        for y_watch in (0, 1):
            for y_phone in (0, 1):
                w_idx, p_idx = combo_to_indices(y_watch, y_phone)
                feat = make_window_features(acc, ori, w_idx, p_idx, fps, acc_scale)
                for start, win in enumerate(
                    slide_windows(feat, window_len, stride)
                ):
                    # recover start frame from enumerate of windows
                    frame0 = start * stride
                    buckets[split]["x"].append(win)
                    buckets[split]["y_watch"].append(y_watch)
                    buckets[split]["y_phone"].append(y_phone)
                    buckets[split]["meta"].append(
                        {
                            "source": path.name,
                            "seq_local": local_i,
                            "y_watch": y_watch,
                            "y_phone": y_phone,
                            "watch_imu": w_idx,
                            "phone_imu": p_idx,
                            "start": frame0,
                        }
                    )
    return buckets


def _stack_bucket(bucket: Dict[str, list]) -> Dict[str, torch.Tensor | list]:
    if not bucket["x"]:
        return {
            "x": torch.zeros(0, 1, 12),
            "y_watch": torch.zeros(0, dtype=torch.long),
            "y_phone": torch.zeros(0, dtype=torch.long),
            "meta": [],
        }
    return {
        "x": torch.stack(bucket["x"], dim=0).float(),
        "y_watch": torch.tensor(bucket["y_watch"], dtype=torch.long),
        "y_phone": torch.tensor(bucket["y_phone"], dtype=torch.long),
        "meta": bucket["meta"],
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        default="experiments/week1_pos_cls/configs/default.yaml",
    )
    args = parser.parse_args()
    cfg = load_config(resolve_path(args.config))
    set_seed(cfg["experiment"]["seed"])

    processed_dir = resolve_path(cfg["data"]["processed_amass_dir"])
    out_dir = resolve_path(cfg["data"]["out_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    files = _list_amass_files(processed_dir, cfg["data"].get("amass_subsets") or [])
    if not files:
        raise FileNotFoundError(
            f"No AMASS .pt found under {processed_dir}. "
            "Run mobileposer data_process_mocap.py --dataset amass first, "
            "and fix data.processed_amass_dir in the config."
        )

    # Count sequences for split
    seq_counts = []
    for f in files:
        d = torch.load(f, map_location="cpu")
        seq_counts.append(len(d["acc"]))
    total_seq = sum(seq_counts)
    val_ids = _split_by_sequence(total_seq, cfg["data"]["val_ratio"], cfg["experiment"]["seed"])
    print(f"AMASS files={len(files)}, sequences={total_seq}, val_seqs={len(val_ids)}")

    train_b = {"x": [], "y_watch": [], "y_phone": [], "meta": []}
    val_b = {"x": [], "y_watch": [], "y_phone": [], "meta": []}

    offset = 0
    for f, n_seq in tqdm(list(zip(files, seq_counts)), desc="build_amass"):
        # rebuild val_ids relative to global sequence index
        local_val = {i for i in range(offset, offset + n_seq) if i in val_ids}
        # remap to local indices expected by process_file
        local_val_ids = {i - offset for i in local_val}
        part = process_file(
            f,
            window_len=cfg["data"]["window_len"],
            stride=cfg["data"]["train_stride"],
            fps=cfg["data"]["fps"],
            acc_scale=cfg["data"]["acc_scale"],
            val_ids=local_val_ids,
            seq_offset=0,
        )
        for k in train_b:
            train_b[k].extend(part["train"][k])
            val_b[k].extend(part["val"][k])
        offset += n_seq

    train = _stack_bucket(train_b)
    val = _stack_bucket(val_b)

    # Normalization stats from train only (acc+gyro channels)
    if train["x"].numel() > 0:
        flat = train["x"].reshape(-1, train["x"].shape[-1])
        mean = flat.mean(dim=0)
        std = flat.std(dim=0).clamp_min(1e-6)
    else:
        mean = torch.zeros(12)
        std = torch.ones(12)

    torch.save(train, out_dir / "amass_train.pt")
    torch.save(val, out_dir / "amass_val.pt")
    torch.save({"mean": mean, "std": std}, out_dir / "norm_stats.pt")

    print(f"train windows: {train['x'].shape}")
    print(f"val windows:   {val['x'].shape}")
    print(f"saved to {out_dir}")


if __name__ == "__main__":
    main()
