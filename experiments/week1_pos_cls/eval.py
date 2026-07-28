#!/usr/bin/env python3
"""Evaluate Week1 position classifier on AMASS val / IMUPoser test."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

_EXP_DIR = Path(__file__).resolve().parent
if str(_EXP_DIR) not in sys.path:
    sys.path.insert(0, str(_EXP_DIR))

from dataset import load_config, resolve_path, set_seed  # noqa: E402
from dataset.pos_dataset import PosClsDataset, load_norm_stats  # noqa: E402
from models import PosClassifier  # noqa: E402
from torch.utils.data import DataLoader


def confusion_2(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    cm = np.zeros((2, 2), dtype=np.int64)
    for t, p in zip(y_true, y_pred):
        cm[int(t), int(p)] += 1
    return cm


def confusion_4(yw: np.ndarray, yp: np.ndarray, pw: np.ndarray, pp: np.ndarray) -> np.ndarray:
    """Joint 4-class combo: id = watch*2 + phone."""
    cm = np.zeros((4, 4), dtype=np.int64)
    yt = yw * 2 + yp
    yp_ = pw * 2 + pp
    for t, p in zip(yt, yp_):
        cm[int(t), int(p)] += 1
    return cm


@torch.no_grad()
def run_eval(model, loader, device):
    model.eval()
    all_yw, all_yp, all_pw, all_pp = [], [], [], []
    ce = nn.CrossEntropyLoss(reduction="sum")
    loss_sum = 0.0
    n = 0
    for x, yw, yp in tqdm(loader, desc="eval"):
        x = x.to(device)
        yw = yw.to(device)
        yp = yp.to(device)
        lw, lp = model(x)
        loss_sum += (ce(lw, yw) + ce(lp, yp)).item()
        pw = lw.argmax(dim=-1)
        pp = lp.argmax(dim=-1)
        all_yw.append(yw.cpu().numpy())
        all_yp.append(yp.cpu().numpy())
        all_pw.append(pw.cpu().numpy())
        all_pp.append(pp.cpu().numpy())
        n += x.size(0)

    yw = np.concatenate(all_yw)
    yp = np.concatenate(all_yp)
    pw = np.concatenate(all_pw)
    pp = np.concatenate(all_pp)
    watch_acc = float((yw == pw).mean())
    phone_acc = float((yp == pp).mean())
    joint_acc = float(((yw == pw) & (yp == pp)).mean())
    return {
        "n": int(n),
        "loss": float(loss_sum / max(n, 1)),
        "watch_acc": watch_acc,
        "phone_acc": phone_acc,
        "joint_acc": joint_acc,
        "cm_watch": confusion_2(yw, pw).tolist(),
        "cm_phone": confusion_2(yp, pp).tolist(),
        "cm_joint4": confusion_4(yw, yp, pw, pp).tolist(),
        "y_watch": yw,
        "y_phone": yp,
        "p_watch": pw,
        "p_phone": pp,
    }


def sequence_majority_vote(meta, yw, yp, pw, pp):
    """If meta has 'seq', aggregate by majority vote per sequence."""
    if not meta or meta[0] is None or "seq" not in meta[0]:
        return None
    from collections import defaultdict

    buckets = defaultdict(list)
    for i, m in enumerate(meta):
        buckets[m["seq"]].append(i)

    correct = 0
    total = 0
    for _, idxs in buckets.items():
        # majority on predicted combo
        preds = [int(pw[i]) * 2 + int(pp[i]) for i in idxs]
        gts = [int(yw[i]) * 2 + int(yp[i]) for i in idxs]
        # GT should be constant per seq
        gt = max(set(gts), key=gts.count)
        pred = max(set(preds), key=preds.count)
        correct += int(pred == gt)
        total += 1
    return {"seq_joint_acc": correct / max(total, 1), "n_seq": total}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        default="experiments/week1_pos_cls/configs/default.yaml",
    )
    parser.add_argument(
        "--split",
        type=str,
        choices=["val", "test"],
        default="test",
        help="val=amass_val.pt, test=imuposer_test.pt",
    )
    parser.add_argument("--checkpoint", type=str, default=None)
    args = parser.parse_args()

    cfg = load_config(resolve_path(args.config))
    set_seed(cfg["experiment"]["seed"])
    data_dir = resolve_path(cfg["data"]["out_dir"])
    stats_pt = data_dir / "norm_stats.pt"
    pt = data_dir / ("amass_val.pt" if args.split == "val" else "imuposer_test.pt")
    ckpt_path = resolve_path(args.checkpoint or cfg["eval"]["checkpoint"])

    for p in (pt, stats_pt, ckpt_path):
        if not p.exists():
            raise FileNotFoundError(p)

    device = torch.device(
        cfg["train"]["device"] if torch.cuda.is_available() else "cpu"
    )
    stats = load_norm_stats(stats_pt)
    ds = PosClsDataset(pt, norm_stats=stats, normalize=True)
    loader = DataLoader(
        ds,
        batch_size=cfg["eval"]["batch_size"],
        shuffle=False,
        num_workers=cfg["train"]["num_workers"],
    )

    ckpt = torch.load(ckpt_path, map_location=device)
    model = PosClassifier(
        n_input=cfg["data"]["input_dim"],
        n_hidden=cfg["model"]["n_hidden"],
        n_lstm_layers=cfg["model"]["n_lstm_layers"],
        bidirectional=cfg["model"]["bidirectional"],
        dropout=cfg["model"]["dropout"],
    ).to(device)
    model.load_state_dict(ckpt["model"])

    metrics = run_eval(model, loader, device)
    seq_metric = sequence_majority_vote(
        ds.meta,
        metrics.pop("y_watch"),
        metrics.pop("y_phone"),
        metrics.pop("p_watch"),
        metrics.pop("p_phone"),
    )
    if seq_metric:
        metrics.update(seq_metric)

    out_path = resolve_path("experiments/week1_pos_cls/outputs/logs") / f"metrics_{args.split}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    print(json.dumps(metrics, indent=2))
    print(f"saved {out_path}")


if __name__ == "__main__":
    main()
