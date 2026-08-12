#!/usr/bin/env python3
"""Bar charts for Week2 advisor PPT (from metrics_summary.json)."""
from __future__ import annotations

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

from dataset import resolve_path  # noqa: E402


def _style(ax):
    ax.set_facecolor("white")
    for sp in ax.spines.values():
        sp.set_color("black")
        sp.set_linewidth(1.0)
    ax.tick_params(direction="out", length=4, labelsize=10)
    ax.grid(False)


def plot_extrinsic_bars(summary: dict, out: Path):
    """None / Learned / Oracle for AMASS + D5, single vs dual."""
    want = {
        ("single", "AMASS val (R_SB °)"): "AMASS\nSingle",
        ("dual", "AMASS val (R_SB °)"): "AMASS\nDual",
        ("single", "D5 IMUPoser (R_SB °)"): "D5 Real\nSingle",
        ("dual", "D5 IMUPoser (R_SB °)"): "D5 Real\nDual",
    }
    rows = {(r["track"], r["eval"]): r for r in summary["main_table"]}
    labels, none_v, learn_v, ora_v = [], [], [], []
    for key, lab in want.items():
        r = rows[key]
        labels.append(lab)
        none_v.append(r["none"])
        learn_v.append(r["learned"])
        ora_v.append(r["oracle"])

    x = np.arange(len(labels))
    w = 0.25
    fig, ax = plt.subplots(figsize=(9.2, 4.2))
    fig.patch.set_facecolor("white")
    _style(ax)
    ax.bar(x - w, none_v, w, label="None", color="#7f7f7f")
    ax.bar(x, learn_v, w, label="Learned", color="#1f77b4")
    ax.bar(x + w, ora_v, w, label="GT / Oracle", color="#2ca02c")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Rotation Error (deg)", fontsize=12)
    ax.set_title("Week2 Extrinsic Error: None / Learned / Oracle", fontsize=12)
    ax.legend(frameon=True, edgecolor="black", fontsize=9)
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def plot_d6_bars(summary: dict, out: Path):
    want = {
        ("single", "D6 AMASS Joint Acc"): "AMASS\nSingle",
        ("dual", "D6 AMASS Joint Acc"): "AMASS\nDual",
        ("single", "D6 IMUPoser Joint Acc"): "IMUPoser\nSingle",
        ("dual", "D6 IMUPoser Joint Acc"): "IMUPoser\nDual",
    }
    rows = {(r["track"], r["eval"]): r for r in summary["main_table"]}
    labels, none_v, learn_v, ora_v = [], [], [], []
    for key, lab in want.items():
        r = rows[key]
        labels.append(lab)
        none_v.append(r["none"])
        learn_v.append(r["learned"])
        ora_v.append(r["oracle"])

    x = np.arange(len(labels))
    w = 0.25
    fig, ax = plt.subplots(figsize=(9.2, 4.2))
    fig.patch.set_facecolor("white")
    _style(ax)
    ax.bar(x - w, none_v, w, label="None", color="#7f7f7f")
    ax.bar(x, learn_v, w, label="Learned", color="#1f77b4")
    ax.bar(x + w, ora_v, w, label="GT / Oracle", color="#2ca02c")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Week1 Joint Acc", fontsize=12)
    ax.set_ylim(0, 1.05)
    ax.set_title("Week2 Downstream D6: Position Classification Recovery", fontsize=12)
    ax.legend(frameon=True, edgecolor="black", fontsize=9)
    fig.tight_layout()
    fig.savefig(out, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def plot_single_vs_dual(summary: dict, out: Path):
    """Learned only: single vs dual on 4 evals (normalize extrinsic to 0-1 for joint scale? No - two panels)."""
    comps = summary["single_vs_dual"]
    # panel1 extrinsic, panel2 joint
    ext = [c for c in comps if c["lower_better"]]
    jnt = [c for c in comps if not c["lower_better"]]

    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.0))
    fig.patch.set_facecolor("white")
    for ax, group, ylabel, title in (
        (axes[0], ext, "Error (deg)", "Extrinsic (lower better)"),
        (axes[1], jnt, "Joint Acc", "Downstream D6 (higher better)"),
    ):
        _style(ax)
        labs = [c["eval"].replace(" (R_SB °)", "").replace(" Joint Acc", "") for c in group]
        s = [c["single_learned"] for c in group]
        d = [c["dual_learned"] for c in group]
        x = np.arange(len(labs))
        w = 0.35
        ax.bar(x - w / 2, s, w, label="Single", color="#ff7f0e")
        ax.bar(x + w / 2, d, w, label="Dual", color="#1f77b4")
        ax.set_xticks(x)
        ax.set_xticklabels(labs, fontsize=9)
        ax.set_ylabel(ylabel, fontsize=11)
        ax.set_title(title, fontsize=11)
        ax.legend(frameon=True, edgecolor="black", fontsize=9)
    fig.suptitle("Single-device vs Dual-device (Learned)", fontsize=12, y=1.02)
    fig.tight_layout()
    fig.savefig(out, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main():
    log = resolve_path("experiments/week2_rot_ext/outputs/logs/metrics_summary.json")
    if not log.exists():
        raise FileNotFoundError(f"Run summarize_metrics.py first: missing {log}")
    summary = json.loads(log.read_text(encoding="utf-8"))
    fig_dir = resolve_path("experiments/week2_rot_ext/outputs/figures")
    plot_extrinsic_bars(summary, fig_dir / "ppt_extrinsic_none_learned_oracle.png")
    plot_d6_bars(summary, fig_dir / "ppt_d6_joint_none_learned_oracle.png")
    plot_single_vs_dual(summary, fig_dir / "ppt_single_vs_dual_learned.png")
    print("saved:")
    print(" ", fig_dir / "ppt_extrinsic_none_learned_oracle.png")
    print(" ", fig_dir / "ppt_d6_joint_none_learned_oracle.png")
    print(" ", fig_dir / "ppt_single_vs_dual_learned.png")


if __name__ == "__main__":
    main()
