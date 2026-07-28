#!/usr/bin/env python3
"""Train Week1 dual-head position classifier."""
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

from dataset import load_config, resolve_path, set_seed  # noqa: E402
from dataset.pos_dataset import make_loader  # noqa: E402
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
        default="experiments/week1_pos_cls/configs/default.yaml",
    )
    args = parser.parse_args()
    cfg = load_config(resolve_path(args.config))
    set_seed(cfg["experiment"]["seed"])

    data_dir = resolve_path(cfg["data"]["out_dir"])
    train_pt = data_dir / "amass_train.pt"
    val_pt = data_dir / "amass_val.pt"
    stats_pt = data_dir / "norm_stats.pt"
    for p in (train_pt, val_pt, stats_pt):
        if not p.exists():
            raise FileNotFoundError(
                f"Missing {p}. Run dataset/build_amass_pos_cls.py first."
            )

    ckpt_dir = resolve_path("experiments/week1_pos_cls/outputs/checkpoints")
    log_dir = resolve_path("experiments/week1_pos_cls/outputs/logs")
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(
        cfg["train"]["device"] if torch.cuda.is_available() else "cpu"
    )
    train_loader = make_loader(
        train_pt,
        stats_pt,
        batch_size=cfg["train"]["batch_size"],
        shuffle=True,
        num_workers=cfg["train"]["num_workers"],
    )
    val_loader = make_loader(
        val_pt,
        stats_pt,
        batch_size=cfg["train"]["batch_size"],
        shuffle=False,
        num_workers=cfg["train"]["num_workers"],
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
    with open(log_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            ["epoch", "train_loss", "val_loss", "watch_acc", "phone_acc", "joint_acc"]
        )

        best_joint = -1.0
        for epoch in range(1, cfg["train"]["epochs"] + 1):
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
