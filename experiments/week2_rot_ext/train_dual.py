#!/usr/bin/env python3
"""Train Week2 dual-device R_SB joint regressor."""
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
from dataset.rot_dataset import make_dual_loader  # noqa: E402
from models import RotExtrinsicDualNet  # noqa: E402


def compute_dual_loss(
    pred_6d_w,
    pred_r_w,
    pred_6d_p,
    pred_r_p,
    r_w,
    r_p,
    r6d_w,
    r6d_p,
    r6d_weight: float = 0.1,
):
    loss_w = rot_chordal_loss(pred_r_w, r_w) + r6d_weight * nn.functional.mse_loss(
        pred_6d_w, r6d_w
    )
    loss_p = rot_chordal_loss(pred_r_p, r_p) + r6d_weight * nn.functional.mse_loss(
        pred_6d_p, r6d_p
    )
    return loss_w + loss_p, loss_w.detach(), loss_p.detach()


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    n = 0
    total_loss = 0.0
    sum_w = sum_p = 0.0
    for batch in loader:
        x, sw, sp, rw, rp, r6w, r6p = batch
        x = x.to(device)
        sw, sp = sw.to(device), sp.to(device)
        rw, rp = rw.to(device), rp.to(device)
        r6w, r6p = r6w.to(device), r6p.to(device)
        (p6w, prw), (p6p, prp) = model(x, sw, sp)
        loss, _, _ = compute_dual_loss(p6w, prw, p6p, prp, rw, rp, r6w, r6p)
        if not torch.isfinite(loss):
            continue
        deg_w = geodesic_angle_deg(prw, rw)
        deg_p = geodesic_angle_deg(prp, rp)
        total_loss += loss.item() * x.size(0)
        sum_w += deg_w.sum().item()
        sum_p += deg_p.sum().item()
        n += x.size(0)
    return {
        "loss": total_loss / max(n, 1),
        "rot_err_deg_watch": sum_w / max(n, 1),
        "rot_err_deg_phone": sum_p / max(n, 1),
        "rot_err_deg_mean": 0.5 * (sum_w + sum_p) / max(n, 1),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        default="experiments/week2_rot_ext/configs/default.yaml",
    )
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Resume from last_dual.pt (or a dual checkpoint). Appends train_log_dual.csv.",
    )
    args = parser.parse_args()
    cfg = load_config(resolve_path(args.config))
    set_seed(cfg["experiment"]["seed"])

    data_dir = resolve_path(cfg["data"]["out_dir"])
    train_pt = data_dir / "amass_dual_train.pt"
    val_pt = data_dir / "amass_dual_val.pt"
    stats_pt = data_dir / "norm_stats_dual.pt"
    for p in (train_pt, val_pt, stats_pt):
        if not p.exists():
            raise FileNotFoundError(
                f"Missing {p}. Run dataset/build_amass_rot_ext_dual.py first."
            )

    ckpt_dir = resolve_path("experiments/week2_rot_ext/outputs/checkpoints")
    log_dir = resolve_path("experiments/week2_rot_ext/outputs/logs")
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(
        cfg["train"]["device"] if torch.cuda.is_available() else "cpu"
    )
    train_loader = make_dual_loader(
        train_pt,
        stats_pt,
        batch_size=cfg["train"]["batch_size"],
        shuffle=True,
        num_workers=cfg["train"]["num_workers"],
    )
    val_loader = make_dual_loader(
        val_pt,
        stats_pt,
        batch_size=cfg["train"]["batch_size"],
        shuffle=False,
        num_workers=cfg["train"]["num_workers"],
    )

    model = RotExtrinsicDualNet(
        feat_dim=24,
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

    log_path = log_dir / "train_log_dual.csv"
    best_path = ckpt_dir / "best_rot_err_dual.pt"
    last_path = ckpt_dir / "last_dual.pt"

    start_epoch = 1
    best_err = float("inf")
    if args.resume:
        resume_path = resolve_path(args.resume)
        ckpt = torch.load(resume_path, map_location=device)
        model.load_state_dict(ckpt["model"])
        start_epoch = int(ckpt.get("epoch", 0)) + 1
        if best_path.exists():
            best_ck = torch.load(best_path, map_location="cpu")
            best_err = float(
                best_ck.get("metrics", {}).get("rot_err_deg_mean", float("inf"))
            )
        print(
            f"resume {resume_path.name} epoch={ckpt.get('epoch')} "
            f"-> start {start_epoch}, best_err={best_err:.2f}°"
        )

    log_mode = "a" if args.resume and log_path.exists() else "w"
    with open(log_path, log_mode, newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if log_mode == "w":
            writer.writerow(
                [
                    "epoch",
                    "train_loss",
                    "val_loss",
                    "val_rot_err_watch",
                    "val_rot_err_phone",
                    "val_rot_err_mean",
                ]
            )

        nan_epochs = 0
        for epoch in range(start_epoch, cfg["train"]["epochs"] + 1):
            model.train()
            running = 0.0
            n = 0
            skipped = 0
            pbar = tqdm(train_loader, desc=f"dual epoch {epoch}")
            for step, batch in enumerate(pbar, start=1):
                x, sw, sp, rw, rp, r6w, r6p = batch
                x = x.to(device)
                sw, sp = sw.to(device), sp.to(device)
                rw, rp = rw.to(device), rp.to(device)
                r6w, r6p = r6w.to(device), r6p.to(device)
                opt.zero_grad(set_to_none=True)
                (p6w, prw), (p6p, prp) = model(x, sw, sp)
                loss, _, _ = compute_dual_loss(
                    p6w, prw, p6p, prp, rw, rp, r6w, r6p, r6d_weight=r6d_weight
                )
                if not torch.isfinite(loss):
                    skipped += 1
                    continue
                loss.backward()
                if grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                grads_ok = all(
                    p.grad is None or torch.isfinite(p.grad).all()
                    for p in model.parameters()
                )
                if not grads_ok:
                    skipped += 1
                    opt.zero_grad(set_to_none=True)
                    continue
                opt.step()
                running += loss.item() * x.size(0)
                n += x.size(0)
                if step % cfg["train"]["log_every"] == 0:
                    pbar.set_postfix(loss=running / max(n, 1), skipped=skipped)

            train_loss = running / max(n, 1) if n > 0 else float("nan")
            metrics = evaluate(model, val_loader, device)
            writer.writerow(
                [
                    epoch,
                    f"{train_loss:.6f}",
                    f"{metrics['loss']:.6f}",
                    f"{metrics['rot_err_deg_watch']:.4f}",
                    f"{metrics['rot_err_deg_phone']:.4f}",
                    f"{metrics['rot_err_deg_mean']:.4f}",
                ]
            )
            f.flush()
            print(
                f"[dual epoch {epoch}] train={train_loss:.4f} "
                f"val_loss={metrics['loss']:.4f} "
                f"watch={metrics['rot_err_deg_watch']:.2f}° "
                f"phone={metrics['rot_err_deg_phone']:.2f}° "
                f"mean={metrics['rot_err_deg_mean']:.2f}° "
                f"skipped={skipped}"
            )

            if not math.isfinite(metrics["rot_err_deg_mean"]) or not math.isfinite(
                train_loss
            ):
                nan_epochs += 1
                if nan_epochs >= 3:
                    print("[stop] too many NaN epochs")
                    break
                continue
            nan_epochs = 0

            payload = {
                "epoch": epoch,
                "model": model.state_dict(),
                "cfg": cfg,
                "metrics": metrics,
                "mode": "dual",
            }
            torch.save(payload, last_path)
            if metrics["rot_err_deg_mean"] < best_err:
                best_err = metrics["rot_err_deg_mean"]
                torch.save(payload, best_path)
                print(f"  -> saved {best_path.name} ({best_err:.2f}°)")

    print(f"done. best mean rot_err={best_err:.2f}°")
    print(f"log: {log_path}")


if __name__ == "__main__":
    main()
