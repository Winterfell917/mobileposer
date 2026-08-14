#!/usr/bin/env python3
"""
Evaluate position classifier trained with unknown R_BS (a_M + R_MS input).

AMASS val and IMUPoser test both reconstruct R_MS = R_MB @ R_BS online
(same protocol as training). IMUPoser recordings are treated as R_MB.

Usage (repo root):
  python experiments/week3_pos_rsb/eval.py \\
      --config experiments/week3_pos_rsb/configs/default.yaml
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

_EXP_DIR = Path(__file__).resolve().parent
if str(_EXP_DIR) not in sys.path:
    sys.path.insert(0, str(_EXP_DIR))

from dataset import (  # noqa: E402
    combo_to_indices,
    load_config,
    load_norm_stats,
    resolve_path,
    set_seed,
)
from dataset.pos_dataset import (  # noqa: E402
    AmassRmsPosDataset,
    ImuposerRmsPosDataset,
    build_imuposer_index,
    load_amass_rms_pack,
    load_imuposer_sequences,
)
from models import PosClassifier  # noqa: E402


def _acc(y: np.ndarray, p: np.ndarray) -> float:
    if len(y) == 0:
        return float("nan")
    return float((y == p).mean())


@torch.no_grad()
def run_loader(model, loader, device) -> Dict[str, np.ndarray]:
    model.eval()
    yw, yp, pw, pp = [], [], [], []
    for x, y_w, y_p in tqdm(loader, desc="eval"):
        x = x.to(device)
        lw, lp = model(x)
        yw.append(y_w.numpy())
        yp.append(y_p.numpy())
        pw.append(lw.argmax(dim=-1).cpu().numpy())
        pp.append(lp.argmax(dim=-1).cpu().numpy())
    return {
        "yw": np.concatenate(yw) if yw else np.zeros(0, dtype=np.int64),
        "yp": np.concatenate(yp) if yp else np.zeros(0, dtype=np.int64),
        "pw": np.concatenate(pw) if pw else np.zeros(0, dtype=np.int64),
        "pp": np.concatenate(pp) if pp else np.zeros(0, dtype=np.int64),
    }


def summarize(pred: Dict[str, np.ndarray]) -> Dict[str, Any]:
    yw, yp, pw, pp = pred["yw"], pred["yp"], pred["pw"], pred["pp"]
    n = int(len(yw))
    return {
        "n": n,
        "watch_acc": _acc(yw, pw),
        "phone_acc": _acc(yp, pp),
        "joint_acc": float(((yw == pw) & (yp == pp)).mean()) if n else float("nan"),
    }


def seq_majority_amass(index: torch.Tensor, pred: Dict[str, np.ndarray]) -> Dict[str, Any]:
    buckets: Dict[Tuple[int, int, int], List[int]] = defaultdict(list)
    for i, row in enumerate(index.tolist()):
        seq_i, yw, yp, _ = row
        buckets[(seq_i, yw, yp)].append(i)
    return _majority(buckets, pred)


def seq_majority_imuposer(index: torch.Tensor, pred: Dict[str, np.ndarray]) -> Dict[str, Any]:
    buckets: Dict[Tuple[int], List[int]] = defaultdict(list)
    for i, row in enumerate(index.tolist()):
        buckets[(row[0],)].append(i)
    return _majority(buckets, pred)


def _majority(buckets, pred) -> Dict[str, Any]:
    yw, yp, pw, pp = pred["yw"], pred["yp"], pred["pw"], pred["pp"]
    correct = 0
    for idxs in buckets.values():
        gts = [int(yw[i]) * 2 + int(yp[i]) for i in idxs]
        preds = [int(pw[i]) * 2 + int(pp[i]) for i in idxs]
        gt = max(set(gts), key=gts.count)
        pr = max(set(preds), key=preds.count)
        correct += int(pr == gt)
    n_seq = len(buckets)
    return {"seq_joint_acc": correct / max(n_seq, 1), "n_seq": n_seq}


def combo_tag(watch_side: int, phone_side: int) -> str:
    return f"{'lw' if watch_side == 0 else 'rw'}_{'lp' if phone_side == 0 else 'rp'}"


def combo_label(watch_side: int, phone_side: int) -> str:
    return f"{'LW' if watch_side == 0 else 'RW'}+{'LP' if phone_side == 0 else 'RP'}"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        default="experiments/week3_pos_rsb/configs/default.yaml",
    )
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--split", choices=["val", "test", "both"], default="both")
    parser.add_argument("--combos", type=str, default="lw_lp,lw_rp,rw_lp,rw_rp")
    args = parser.parse_args()

    cfg = load_config(resolve_path(args.config))
    set_seed(cfg["experiment"]["seed"])
    device = torch.device(
        cfg["train"]["device"] if torch.cuda.is_available() else "cpu"
    )
    stats_pt = resolve_path(cfg["data"]["out_dir"]) / "norm_stats.pt"
    ckpt_path = resolve_path(args.checkpoint or cfg["eval"]["checkpoint"])
    for p in (stats_pt, ckpt_path):
        if not p.exists():
            raise FileNotFoundError(p)
    stats = load_norm_stats(stats_pt)

    model = PosClassifier(
        n_input=cfg["data"]["input_dim"],
        n_hidden=cfg["model"]["n_hidden"],
        n_lstm_layers=cfg["model"]["n_lstm_layers"],
        bidirectional=cfg["model"]["bidirectional"],
        dropout=cfg["model"]["dropout"],
    ).to(device)
    model.load_state_dict(torch.load(ckpt_path, map_location=device)["model"])
    model.eval()

    log_dir = resolve_path(cfg["eval"]["log_dir"])
    log_dir.mkdir(parents=True, exist_ok=True)
    batch_size = int(cfg["eval"]["batch_size"])
    num_workers = int(cfg["train"]["num_workers"])
    out: Dict[str, Any] = {
        "protocol": (
            "Input a_M+R_MS; R_MS=R_MB@R_BS independent per device; "
            "a_M unchanged; R_BS unknown / sequence-constant"
        )
    }
    rows = []

    if args.split in ("val", "both"):
        pack = load_amass_rms_pack(cfg)
        ds = AmassRmsPosDataset(
            pack["sequences"],
            pack["val_index"],
            window_len=cfg["data"]["window_len"],
            acc_scale=cfg["data"]["acc_scale"],
            offset_range_deg=cfg["data"]["offset_range_deg"],
            seed=cfg["experiment"]["seed"],
            mean=stats["mean"],
            std=stats["std"],
        )
        loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)
        print(f"[AMASS val] windows={len(ds)}")
        pred = run_loader(model, loader, device)
        metrics = summarize(pred)
        metrics.update(seq_majority_amass(pack["val_index"], pred))
        metrics["split"] = "AMASS Val"
        out["val"] = metrics
        rows.append(metrics)
        print(
            f"  watch={metrics['watch_acc']:.4f} phone={metrics['phone_acc']:.4f} "
            f"joint={metrics['joint_acc']:.4f} seq_joint={metrics['seq_joint_acc']:.4f}"
        )

    if args.split in ("test", "both"):
        imu_seqs = load_imuposer_sequences(
            resolve_path(cfg["data"]["processed_imuposer_file"])
        )
        imu_index = build_imuposer_index(
            imu_seqs, cfg["data"]["window_len"], cfg["data"]["test_stride"]
        )
        combo_list = []
        for tok in args.combos.split(","):
            tok = tok.strip().lower().replace("+", "_")
            w, p = tok.split("_")
            combo_list.append((0 if w == "lw" else 1, 0 if p == "lp" else 1))
        out["test"] = {}
        for yw, yp in combo_list:
            tag = combo_tag(yw, yp)
            ds = ImuposerRmsPosDataset(
                imu_seqs,
                imu_index,
                window_len=cfg["data"]["window_len"],
                acc_scale=cfg["data"]["acc_scale"],
                offset_range_deg=cfg["data"]["offset_range_deg"],
                seed=cfg["experiment"]["seed"],
                y_watch=yw,
                y_phone=yp,
                mean=stats["mean"],
                std=stats["std"],
            )
            loader = DataLoader(
                ds, batch_size=batch_size, shuffle=False, num_workers=num_workers
            )
            print(f"[IMUPoser {combo_label(yw, yp)}] windows={len(ds)} slots={combo_to_indices(yw, yp)}")
            pred = run_loader(model, loader, device)
            metrics = summarize(pred)
            metrics.update(seq_majority_imuposer(imu_index, pred))
            metrics["split"] = f"IMUPoser {combo_label(yw, yp)}"
            metrics["combo"] = combo_label(yw, yp)
            out["test"][tag] = metrics
            rows.append(metrics)
            print(
                f"  watch={metrics['watch_acc']:.4f} phone={metrics['phone_acc']:.4f} "
                f"joint={metrics['joint_acc']:.4f} seq_joint={metrics['seq_joint_acc']:.4f}"
            )

    print("\n=== Main ===")
    print(f"{'Split':<18} {'Watch':>8} {'Phone':>8} {'Joint':>8} {'SeqJoint':>8}")
    for m in rows:
        print(
            f"{m['split']:<18} {m['watch_acc']:8.4f} {m['phone_acc']:8.4f} "
            f"{m['joint_acc']:8.4f} {m.get('seq_joint_acc', float('nan')):8.4f}"
        )
    out_path = log_dir / "metrics_step1.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print(f"saved {out_path}")


if __name__ == "__main__":
    main()
