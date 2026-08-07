#!/usr/bin/env python3
"""Train Week2 single-device R_SB regressor."""
from __future__ import annotations

import argparse
import csv
import math
import sys
from pathlib import Path

import torch
import torch.nn as nn
from tqdm import tqdm

_EXP_DIR = Path(__file__).resolve().parent
if str(_EXP_DIR) not in sys.path:
    sys.path.insert(0, str(_EXP_DIR))

from dataset import (  # noqa: E402
    geodesic_angle_deg,
    load_config,
    resolve_path,
    rot_chordal_loss,
    set_seed,
)
from dataset.rot_dataset import make_loader  # noqa: E402
from models import RotExtrinsicNet  # noqa: E402


def compute_loss(pred_6d, pred_r, r_sb, r6d, r6d_weight: float = 0.1):
    """Chordal SO(3) loss + light 6D MSE (no acos)."""
    return rot_chordal_loss(pred_r, r_sb) + r6d_weight * nn.functional.mse_loss(
        pred_6d, r6d
    )


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    n = 0
    total_loss = 0.0
    total_deg = 0.0
    per_slot_deg = {}
    per_slot_n = {}
    for x, slot, r_sb, r6d in loader:
        x = x.to(device)
        slot = slot.to(device)
        r_sb = r_sb.to(device)
        r6d = r6d.to(device)
        pred_6d, pred_r = model(x, slot)
        loss = compute_loss(pred_6d, pred_r, r_sb, r6d)
        if not torch.isfinite(loss):
            continue
        deg = geodesic_angle_deg(pred_r, r_sb)
        total_loss += loss.item() * x.size(0)
        total_deg += deg.sum().item()
        n += x.size(0)
        for s in slot.unique().tolist():
            m = slot == s
            per_slot_deg[s] = per_slot_deg.get(s, 0.0) + deg[m].sum().item()
            per_slot_n[s] = per_slot_n.get(s, 0) + int(m.sum().item())
    metrics = {
        "loss": total_loss / max(n, 1),
        "rot_err_deg": total_deg / max(n, 1),
    }
    for s, v in per_slot_deg.items():
        metrics[f"rot_err_deg_slot{s}"] = v / max(per_slot_n[s], 1)
    return metrics


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

    data_dir = resolve_path(cfg["data"]["out_dir"])
    train_pt = data_dir / "amass_train.pt"
    val_pt = data_dir / "amass_val.pt"
    stats_pt = data_dir / "norm_stats.pt"
    for p in (train_pt, val_pt, stats_pt):
        if not p.exists():
            raise FileNotFoundError(
                f"Missing {p}. Run dataset/build_amass_rot_ext.py first."
            )

    ckpt_dir = resolve_path("experiments/week2_rot_ext/outputs/checkpoints")
    log_dir = resolve_path("experiments/week2_rot_ext/outputs/logs")
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

    model = RotExtrinsicNet(
        feat_dim=cfg["data"]["feat_dim"],
        n_slots=cfg["data"]["n_slots"],
        n_hidden=cfg["model"]["n_hidden"],
        n_lstm_layers=cfg["model"]["n_lstm_layers"],
        bidirectional=cfg["model"]["bidirectional"],
        dropout=cfg["model"]["dropout"],
        use_slot_onehot=cfg["model"]["use_slot_onehot"],
    ).to(device)

    opt = torch.optim.AdamW(
        model.parameters(),
        lr=cfg["train"]["lr"],
        weight_decay=cfg["train"]["weight_decay"],
    )
    grad_clip = float(cfg["train"].get("grad_clip_norm", 1.0))
    r6d_weight = float(cfg["train"].get("r6d_loss_weight", 0.1))

    log_path = log_dir / "train_log.csv"
    with open(log_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "train_loss", "val_loss", "val_rot_err_deg"])

        best_err = float("inf")
        nan_epochs = 0
        for epoch in range(1, cfg["train"]["epochs"] + 1):
            model.train()
            running = 0.0
            n = 0
            skipped = 0
            pbar = tqdm(train_loader, desc=f"epoch {epoch}")
            for step, (x, slot, r_sb, r6d) in enumerate(pbar, start=1):
                x = x.to(device)
                slot = slot.to(device)
                r_sb = r_sb.to(device)
                r6d = r6d.to(device)
                opt.zero_grad(set_to_none=True)
                pred_6d, pred_r = model(x, slot)
                loss = compute_loss(pred_6d, pred_r, r_sb, r6d, r6d_weight=r6d_weight)
                if not torch.isfinite(loss):
                    skipped += 1
                    continue
                loss.backward()
                if grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                # Skip update if any grad is non-finite
                grads_ok = True
                for p in model.parameters():
                    if p.grad is not None and not torch.isfinite(p.grad).all():
                        grads_ok = False
                        break
                if not grads_ok:
                    skipped += 1
                    opt.zero_grad(set_to_none=True)
                    continue
                opt.step()
                running += loss.item() * x.size(0)
                n += x.size(0)
                if step % cfg["train"]["log_every"] == 0:
                    pbar.set_postfix(
                        loss=running / max(n, 1), skipped=skipped
                    )

            train_loss = running / max(n, 1) if n > 0 else float("nan")
            metrics = evaluate(model, val_loader, device)
            writer.writerow(
                [
                    epoch,
                    f"{train_loss:.6f}",
                    f"{metrics['loss']:.6f}",
                    f"{metrics['rot_err_deg']:.4f}",
                ]
            )
            f.flush()
            print(
                f"[epoch {epoch}] train_loss={train_loss:.4f} "
                f"val_loss={metrics['loss']:.4f} "
                f"val_rot_err={metrics['rot_err_deg']:.2f}° "
                f"skipped_batches={skipped}"
            )

            if not math.isfinite(metrics["rot_err_deg"]) or not math.isfinite(
                train_loss
            ):
                nan_epochs += 1
                print(
                    f"  [warn] non-finite metrics (nan_epochs={nan_epochs}); "
                    "not updating best checkpoint"
                )
                if nan_epochs >= 3:
                    print(
                        "  [stop] too many NaN epochs; keep best_rot_err.pt and exit"
                    )
                    break
                continue
            nan_epochs = 0

            payload = {
                "epoch": epoch,
                "model": model.state_dict(),
                "cfg": cfg,
                "metrics": metrics,
            }
            torch.save(payload, ckpt_dir / "last.pt")
            if metrics["rot_err_deg"] < best_err:
                best_err = metrics["rot_err_deg"]
                torch.save(payload, ckpt_dir / "best_rot_err.pt")
                print(f"  -> saved best_rot_err.pt ({best_err:.2f}°)")

    print(f"done. best val rot_err={best_err:.2f}°")
    print(f"log: {log_path}")


if __name__ == "__main__":
    main()
