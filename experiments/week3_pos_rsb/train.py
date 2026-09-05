#!/usr/bin/env python3
"""Train Week3 step-1: position classifier from a_M + R_MS (unknown R_BS)."""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import torch
import torch.nn as nn
from tqdm import tqdm

_EXP_DIR = Path(__file__).resolve().parent
if str(_EXP_DIR) not in sys.path:
    sys.path.insert(0, str(_EXP_DIR))

from dataset import load_config, load_norm_stats, resolve_path, set_seed  # noqa: E402
from dataset.pos_dataset import load_amass_rms_pack, make_amass_rms_loaders  # noqa: E402
from models import PosClassifier  # noqa: E402


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    n = 0
    correct_w = correct_p = correct_j = 0
    total_loss = 0.0
    ce = nn.CrossEntropyLoss()
    for x, yw, yp in loader:
        x = x.to(device)
        yw = yw.to(device)
        yp = yp.to(device)
        lw, lp = model(x)
        loss = ce(lw, yw) + ce(lp, yp)
        total_loss += loss.item() * x.size(0)
        pw = lw.argmax(dim=-1)
        pp = lp.argmax(dim=-1)
        correct_w += (pw == yw).sum().item()
        correct_p += (pp == yp).sum().item()
        correct_j += ((pw == yw) & (pp == yp)).sum().item()
        n += x.size(0)
    return {
        "loss": total_loss / max(n, 1),
        "watch_acc": correct_w / max(n, 1),
        "phone_acc": correct_p / max(n, 1),
        "joint_acc": correct_j / max(n, 1),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        default="experiments/week3_pos_rsb/configs/default.yaml",
    )
    parser.add_argument(
        "--yaw-align",
        action="store_true",
        help="suggestion 1: sequence-level yaw-only left-multiply before inject",
    )
    args = parser.parse_args()
    cfg = load_config(resolve_path(args.config))
    set_seed(cfg["experiment"]["seed"])

    data_dir = resolve_path(cfg["data"]["out_dir"])
    stats_pt = data_dir / "norm_stats.pt"
    idx_pt = data_dir / "amass_index.pt"
    for p in (idx_pt, stats_pt):
        if not p.exists():
            raise FileNotFoundError(
                f"Missing {p}. Run dataset/build_amass.py first."
            )

    ckpt_dir = resolve_path(cfg["train"]["ckpt_dir"])
    log_dir = resolve_path(cfg["train"]["log_dir"])
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(
        cfg["train"]["device"] if torch.cuda.is_available() else "cpu"
    )
    stats = load_norm_stats(stats_pt)
    pack = load_amass_rms_pack(cfg)
    train_loader, val_loader = make_amass_rms_loaders(
        cfg, stats, pack, yaw_align=bool(args.yaw_align)
    )
    print(
        f"[week3 step1] train_windows={len(train_loader.dataset)} "
        f"val_windows={len(val_loader.dataset)} "
        f"input=a_M+R_MS (24)  R_BS unknown / window-constant "
        f"yaw_align={bool(args.yaw_align)}"
    )

    model = PosClassifier(
        n_input=cfg["data"]["input_dim"],
        n_hidden=cfg["model"]["n_hidden"],
        n_lstm_layers=cfg["model"]["n_lstm_layers"],
        bidirectional=cfg["model"]["bidirectional"],
        dropout=cfg["model"]["dropout"],
    ).to(device)

    opt = torch.optim.AdamW(
        model.parameters(),
        lr=cfg["train"]["lr"],
        weight_decay=cfg["train"]["weight_decay"],
    )
    ce = nn.CrossEntropyLoss()

    log_path = log_dir / "train_log.csv"
    resume = ckpt_dir / "last.pt"
    start_epoch = 1
    best_joint = -1.0
    if resume.exists():
        payload = torch.load(resume, map_location=device)
        model.load_state_dict(payload["model"])
        start_epoch = int(payload.get("epoch", 0)) + 1
        best_joint = float(payload.get("metrics", {}).get("joint_acc", -1.0))
        print(f"resume from {resume} epoch={start_epoch - 1} best_joint={best_joint:.4f}")

    log_mode = "a" if start_epoch > 1 and log_path.exists() else "w"
    with open(log_path, log_mode, newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if log_mode == "w":
            writer.writerow(
                ["epoch", "train_loss", "val_loss", "watch_acc", "phone_acc", "joint_acc"]
            )

        for epoch in range(start_epoch, cfg["train"]["epochs"] + 1):
            model.train()
            running = 0.0
            n = 0
            pbar = tqdm(train_loader, desc=f"epoch {epoch}")
            for step, (x, yw, yp) in enumerate(pbar, start=1):
                x = x.to(device)
                yw = yw.to(device)
                yp = yp.to(device)
                opt.zero_grad(set_to_none=True)
                lw, lp = model(x)
                loss = ce(lw, yw) + ce(lp, yp)
                loss.backward()
                opt.step()
                running += loss.item() * x.size(0)
                n += x.size(0)
                if step % cfg["train"]["log_every"] == 0:
                    pbar.set_postfix(loss=running / max(n, 1))

            train_loss = running / max(n, 1)
            metrics = evaluate(model, val_loader, device)
            writer.writerow(
                [
                    epoch,
                    f"{train_loss:.6f}",
                    f"{metrics['loss']:.6f}",
                    f"{metrics['watch_acc']:.4f}",
                    f"{metrics['phone_acc']:.4f}",
                    f"{metrics['joint_acc']:.4f}",
                ]
            )
            f.flush()
            print(
                f"[epoch {epoch}] train_loss={train_loss:.4f} "
                f"val_loss={metrics['loss']:.4f} "
                f"watch={metrics['watch_acc']:.3f} "
                f"phone={metrics['phone_acc']:.3f} "
                f"joint={metrics['joint_acc']:.3f}"
            )

            torch.save(
                {"epoch": epoch, "model": model.state_dict(), "cfg": cfg, "metrics": metrics},
                ckpt_dir / "last.pt",
            )
            if metrics["joint_acc"] > best_joint:
                best_joint = metrics["joint_acc"]
                torch.save(
                    {
                        "epoch": epoch,
                        "model": model.state_dict(),
                        "cfg": cfg,
                        "metrics": metrics,
                    },
                    ckpt_dir / "best_joint_acc.pt",
                )
                print(f"  -> saved best_joint_acc.pt ({best_joint:.4f})")

    print(f"done. best joint_acc={best_joint:.4f}")
    print(f"log: {log_path}")


if __name__ == "__main__":
    main()
