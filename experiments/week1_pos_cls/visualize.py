#!/usr/bin/env python3
"""Visualize confusion matrices and a few prediction timelines."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader

_EXP_DIR = Path(__file__).resolve().parent
if str(_EXP_DIR) not in sys.path:
    sys.path.insert(0, str(_EXP_DIR))

from dataset import load_config, resolve_path, set_seed  # noqa: E402
from dataset.pos_dataset import PosClsDataset, load_norm_stats  # noqa: E402
from models import PosClassifier  # noqa: E402


COMBO_NAMES = ["LW+LP", "LW+RP", "RW+LP", "RW+RP"]


def plot_cm(cm: np.ndarray, labels, title: str, out: Path):
    fig, ax = plt.subplots(figsize=(4.5, 4))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels)
    ax.set_yticklabels(labels)
    ax.set_xlabel("Pred")
    ax.set_ylabel("GT")
    ax.set_title(title)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center")
    fig.colorbar(im, ax=ax, fraction=0.046)
    fig.tight_layout()
    fig.savefig(out, dpi=160)
    plt.close(fig)


@torch.no_grad()
def collect_preds(model, loader, device):
    model.eval()
    yw_all, yp_all, pw_all, pp_all, x_all = [], [], [], [], []
    for x, yw, yp in loader:
        x = x.to(device)
        lw, lp = model(x)
        yw_all.append(yw.numpy())
        yp_all.append(yp.numpy())
        pw_all.append(lw.argmax(-1).cpu().numpy())
        pp_all.append(lp.argmax(-1).cpu().numpy())
        x_all.append(x.cpu().numpy())
    return (
        np.concatenate(yw_all),
        np.concatenate(yp_all),
        np.concatenate(pw_all),
        np.concatenate(pp_all),
        np.concatenate(x_all),
    )


def plot_timeline(x_win, yw, yp, pw, pp, out: Path):
    """x_win: [T, 12] one window; also supports list of consecutive windows."""
    # mean |acc| for watch / phone over the window
    t = np.arange(x_win.shape[0])
    watch_acc_norm = np.linalg.norm(x_win[:, 0:3], axis=-1)
    phone_acc_norm = np.linalg.norm(x_win[:, 6:9], axis=-1)

    fig, axes = plt.subplots(3, 1, figsize=(10, 6), sharex=True)
    axes[0].plot(t, watch_acc_norm, label="watch |acc|")
    axes[0].plot(t, phone_acc_norm, label="phone |acc|")
    axes[0].legend(loc="upper right")
    axes[0].set_ylabel("norm")
    axes[0].set_title("IMU magnitude")

    axes[1].axhline(yw, color="g", linestyle="--", label="GT watch")
    axes[1].axhline(yp + 2, color="b", linestyle="--", label="GT phone(+2)")
    axes[1].set_yticks([0, 1, 2, 3])
    axes[1].set_yticklabels(["W0", "W1", "P0", "P1"])
    axes[1].legend(loc="upper right")
    axes[1].set_title("Ground truth")

    axes[2].axhline(pw, color="g", label="Pred watch")
    axes[2].axhline(pp + 2, color="b", label="Pred phone(+2)")
    ok = (pw == yw) and (pp == yp)
    axes[2].set_title("Prediction  " + ("OK" if ok else "WRONG"))
    axes[2].set_yticks([0, 1, 2, 3])
    axes[2].set_yticklabels(["W0", "W1", "P0", "P1"])
    axes[2].legend(loc="upper right")
    axes[2].set_xlabel("frame in window")
    fig.tight_layout()
    fig.savefig(out, dpi=160)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        default="experiments/week1_pos_cls/configs/default.yaml",
    )
    parser.add_argument("--split", type=str, choices=["val", "test"], default="test")
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--metrics-json", type=str, default=None,
                        help="If set, only plot confusion from this file")
    args = parser.parse_args()

    cfg = load_config(resolve_path(args.config))
    set_seed(cfg["experiment"]["seed"])
    fig_dir = resolve_path("experiments/week1_pos_cls/outputs/figures")
    fig_dir.mkdir(parents=True, exist_ok=True)

    # Prefer metrics json if provided / exists
    metrics_path = resolve_path(
        args.metrics_json
        or f"experiments/week1_pos_cls/outputs/logs/metrics_{args.split}.json"
    )
    if metrics_path.exists():
        with open(metrics_path, "r", encoding="utf-8") as f:
            metrics = json.load(f)
        plot_cm(
            np.array(metrics["cm_watch"]),
            ["LW", "RW"],
            f"Watch CM ({args.split})",
            fig_dir / f"cm_watch_{args.split}.png",
        )
        plot_cm(
            np.array(metrics["cm_phone"]),
            ["LP", "RP"],
            f"Phone CM ({args.split})",
            fig_dir / f"cm_phone_{args.split}.png",
        )
        plot_cm(
            np.array(metrics["cm_joint4"]),
            COMBO_NAMES,
            f"Joint-4 CM ({args.split})",
            fig_dir / f"cm_joint4_{args.split}.png",
        )
        print(f"saved confusion matrices to {fig_dir}")

    # Timeline examples from model predictions
    data_dir = resolve_path(cfg["data"]["out_dir"])
    pt = data_dir / ("amass_val.pt" if args.split == "val" else "imuposer_test.pt")
    stats_pt = data_dir / "norm_stats.pt"
    ckpt_path = resolve_path(args.checkpoint or cfg["eval"]["checkpoint"])
    if not (pt.exists() and stats_pt.exists() and ckpt_path.exists()):
        print("skip timelines (missing data/checkpoint)")
        return

    device = torch.device(
        cfg["train"]["device"] if torch.cuda.is_available() else "cpu"
    )
    stats = load_norm_stats(stats_pt)
    ds = PosClsDataset(pt, norm_stats=stats, normalize=True)
    loader = DataLoader(ds, batch_size=cfg["eval"]["batch_size"], shuffle=False)

    ckpt = torch.load(ckpt_path, map_location=device)
    model = PosClassifier(
        n_input=cfg["data"]["input_dim"],
        n_hidden=cfg["model"]["n_hidden"],
        n_lstm_layers=cfg["model"]["n_lstm_layers"],
        bidirectional=cfg["model"]["bidirectional"],
        dropout=0.0,
    ).to(device)
    model.load_state_dict(ckpt["model"])

    yw, yp, pw, pp, xs = collect_preds(model, loader, device)
    # pick a few correct and incorrect windows
    joint_ok = (yw == pw) & (yp == pp)
    idxs = []
    ok_ids = np.where(joint_ok)[0][: cfg["visualize"]["num_sequences"]]
    bad_ids = np.where(~joint_ok)[0][: cfg["visualize"]["num_sequences"]]
    idxs.extend(ok_ids.tolist())
    idxs.extend(bad_ids.tolist())
    if not idxs:
        idxs = list(range(min(3, len(xs))))

    for k, i in enumerate(idxs):
        # xs are normalized; for magnitude plot reload raw if needed — normalized is fine for shape
        plot_timeline(
            xs[i],
            int(yw[i]),
            int(yp[i]),
            int(pw[i]),
            int(pp[i]),
            fig_dir / f"timeline_{args.split}_{k}.png",
        )
    print(f"saved {len(idxs)} timelines to {fig_dir}")


if __name__ == "__main__":
    main()
