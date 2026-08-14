#!/usr/bin/env python3
"""Bar charts for Week3 step-2 cascade R_SB errors (from metrics_step2.json)."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

_EXP_DIR = Path(__file__).resolve().parent
if str(_EXP_DIR) not in sys.path:
    sys.path.insert(0, str(_EXP_DIR))

from dataset import load_config, resolve_path  # noqa: E402


KEYS = ["none", "pred_window", "pred_seq", "gt_slot"]
LABELS = ["None", "Pred-win", "Pred-seq", "GT-slot"]
COLORS = ["#9e9e9e", "#d62728", "#ff7f0e", "#2ca02c"]


def _mean_row(block: dict) -> list:
    return [block[k]["mean"]["mean"] for k in KEYS]


def plot_grouped(rows: list, title: str, out: Path):
    names = [r[0] for r in rows]
    data = np.array([r[1] for r in rows], dtype=np.float64)
    x = np.arange(len(names))
    width = 0.18
    fig, ax = plt.subplots(figsize=(10.5, 4.2))
    for i, (lab, col) in enumerate(zip(LABELS, COLORS)):
        ax.bar(x + (i - 1.5) * width, data[:, i], width, label=lab, color=col)
        for xi, v in zip(x + (i - 1.5) * width, data[:, i]):
            ax.text(xi, v + 0.4, f"{v:.1f}", ha="center", va="bottom", fontsize=7)
    ax.set_xticks(x)
    ax.set_xticklabels(names)
    ax.set_ylabel("R_SB geodesic error (°)")
    ax.set_title(title)
    ax.legend(frameon=True, ncol=4, fontsize=9)
    ax.set_ylim(0, max(50.0, float(np.nanmax(data)) * 1.18))
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    fig.savefig(out, dpi=160)
    plt.close(fig)
    print(f"saved {out}")


def plot_watch_phone(m: dict, title: str, out: Path):
    rsb = m["rsb_deg"]
    watch = [rsb[k]["watch"]["mean"] for k in KEYS]
    phone = [rsb[k]["phone"]["mean"] for k in KEYS]
    x = np.arange(len(KEYS))
    fig, ax = plt.subplots(figsize=(7.2, 3.8))
    ax.bar(x - 0.18, watch, 0.36, label="Watch", color="#1f77b4")
    ax.bar(x + 0.18, phone, 0.36, label="Phone", color="#d62728")
    ax.set_xticks(x)
    ax.set_xticklabels(LABELS)
    ax.set_ylabel("mean °")
    ax.set_title(title)
    ax.legend(frameon=True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    fig.savefig(out, dpi=160)
    plt.close(fig)
    print(f"saved {out}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="experiments/week3_pos_rsb/configs/default.yaml")
    parser.add_argument("--metrics", type=str, default=None)
    args = parser.parse_args()
    cfg = load_config(resolve_path(args.config))
    metrics_path = resolve_path(
        args.metrics or "experiments/week3_pos_rsb/outputs/logs/metrics_step2.json"
    )
    fig_dir = resolve_path("experiments/week3_pos_rsb/outputs/figures")
    fig_dir.mkdir(parents=True, exist_ok=True)
    data = json.loads(metrics_path.read_text(encoding="utf-8"))

    rows = []
    if "val" in data:
        rows.append(("AMASS Val", _mean_row(data["val"]["rsb_deg"])))
        plot_watch_phone(data["val"], "Week3 step2 AMASS Val (watch / phone °)", fig_dir / "step2_amass_watch_phone.png")
    for tag, lab in [
        ("lw_lp", "LW+LP"),
        ("lw_rp", "LW+RP"),
        ("rw_lp", "RW+LP"),
        ("rw_rp", "RW+RP"),
    ]:
        if tag in data.get("test", {}):
            rows.append((f"IMU {lab}", _mean_row(data["test"][tag]["rsb_deg"])))
    if rows:
        plot_grouped(rows, "Week3 step2 cascade: R_SB error (°)", fig_dir / "step2_rsb_main.png")

    # stratified OK vs BAD on AMASS
    if "val" in data:
        sc = data["val"]["stratified"]["joint_correct"]["pred_window"]["mean"]["mean"]
        sw = data["val"]["stratified"]["joint_wrong"]["pred_window"]["mean"]["mean"]
        gt_ok = data["val"]["stratified"]["joint_correct"]["gt_slot"]["mean"]["mean"]
        gt_bad = data["val"]["stratified"]["joint_wrong"]["gt_slot"]["mean"]["mean"]
        fig, ax = plt.subplots(figsize=(6.2, 3.6))
        labs = ["Joint OK\nPred-slot", "Joint OK\nGT-slot", "Joint BAD\nPred-slot", "Joint BAD\nGT-slot"]
        vals = [sc, gt_ok, sw, gt_bad]
        cols = ["#ff7f0e", "#2ca02c", "#d62728", "#98df8a"]
        ax.bar(labs, vals, color=cols)
        for i, v in enumerate(vals):
            ax.text(i, v + 0.4, f"{v:.1f}", ha="center", fontsize=9)
        ax.set_ylabel("mean R_SB error (°)")
        ax.set_title("AMASS Val: slot correctness vs R_SB error")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        fig.tight_layout()
        fig.savefig(fig_dir / "step2_amass_stratified.png", dpi=160)
        plt.close(fig)
        print(f"saved {fig_dir / 'step2_amass_stratified.png'}")
    print(f"done. figures in {fig_dir}")


if __name__ == "__main__":
    main()
