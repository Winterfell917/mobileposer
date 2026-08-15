#!/usr/bin/env python3
"""Bar charts + GT / R_MS→pose / cascade mesh for Week3 step 3."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.patches import Patch
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

_EXP_DIR = Path(__file__).resolve().parent
_REPO = _EXP_DIR.parents[1]
_MP_DIR = _REPO / "mobileposer"
if str(_EXP_DIR) not in sys.path:
    sys.path.insert(0, str(_EXP_DIR))

from cascade_pose import (  # noqa: E402
    HEAD,
    cascade_calibrated,
    load_imuposer_pose_one,
    pack_mobileposer_imu,
    predict_pose_sequence,
)
from dataset import load_config, load_norm_stats, resolve_path, set_seed  # noqa: E402
from dataset.pos_dataset import amass_seed_offset  # noqa: E402
from models import PosClassifier, RotExtrinsicDualNet  # noqa: E402

KEYS = ["none", "pred_seq", "gt_slot", "oracle"]
LABELS = ["None\n(R_MS→pose)", "Pred-seq", "GT-slot", "Oracle"]
COLORS = ["#9e9e9e", "#ff7f0e", "#2ca02c", "#1f77b4"]
SMPL_PARENTS = [
    -1, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 9, 12, 13, 14, 16, 17, 18, 19, 20, 21
]


def parse_combo(s: str) -> Tuple[int, int]:
    tok = s.strip().lower().replace("+", "_").replace("-", "_")
    w, p = tok.split("_")
    return (0 if w == "lw" else 1), (0 if p == "lp" else 1)


def _metric(block: dict, key: str, name: str) -> float:
    return float(block["results"][key][name]["mean"])


def plot_grouped(data: Dict[str, dict], metric: str, ylabel: str, title: str, out: Path):
    names = [k for k in ("amass", "imuposer") if k in data]
    labels = ["AMASS" if n == "amass" else "IMUPoser" for n in names]
    vals = np.array([[_metric(data[n], k, metric) for k in KEYS] for n in names])
    x = np.arange(len(names))
    width = 0.18
    fig, ax = plt.subplots(figsize=(9.2, 4.0))
    for i, (lab, col) in enumerate(zip(LABELS, COLORS)):
        ax.bar(x + (i - 1.5) * width, vals[:, i], width, label=lab.replace("\n", " "), color=col)
        for xi, v in zip(x + (i - 1.5) * width, vals[:, i]):
            ax.text(xi, v + 0.15, f"{v:.1f}", ha="center", va="bottom", fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend(frameon=True, ncol=4, fontsize=8)
    ax.set_ylim(0, float(np.nanmax(vals)) * 1.22)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    fig.savefig(out, dpi=160)
    plt.close(fig)
    print(f"saved {out}")


def plot_per_combo(block: dict, metric: str, ylabel: str, title: str, out: Path):
    combos = list(block.get("per_combo") or {})
    if not combos:
        return
    vals = np.array(
        [[float(block["per_combo"][c][k][metric]["mean"]) for k in KEYS] for c in combos]
    )
    x = np.arange(len(combos))
    width = 0.18
    fig, ax = plt.subplots(figsize=(10.5, 4.0))
    for i, (lab, col) in enumerate(zip(LABELS, COLORS)):
        ax.bar(x + (i - 1.5) * width, vals[:, i], width, label=lab.replace("\n", " "), color=col)
        for xi, v in zip(x + (i - 1.5) * width, vals[:, i]):
            ax.text(xi, v + 0.12, f"{v:.1f}", ha="center", va="bottom", fontsize=7)
    ax.set_xticks(x)
    ax.set_xticklabels(combos)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend(frameon=True, ncol=4, fontsize=8)
    ax.set_ylim(0, float(np.nanmax(vals)) * 1.22)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    fig.savefig(out, dpi=160)
    plt.close(fig)
    print(f"saved {out}")


def _set_axes_equal(ax, pts: np.ndarray):
    mins = pts.min(axis=0)
    maxs = pts.max(axis=0)
    center = (mins + maxs) / 2
    radius = 0.5 * np.max(maxs - mins) + 1e-3
    ax.set_xlim(center[0] - radius, center[0] + radius)
    ax.set_ylim(center[1] - radius, center[1] + radius)
    ax.set_zlim(center[2] - radius, center[2] + radius)


def draw_pose_mesh(ax, verts, faces, joints, color, title: str, face_stride: int = 3):
    tri = verts[faces[::face_stride]]
    mesh = Poly3DCollection(tri, alpha=0.14, linewidths=0.05)
    mesh.set_facecolor((*color, 0.28))
    mesh.set_edgecolor((*color, 0.12))
    ax.add_collection3d(mesh)
    for j, p in enumerate(SMPL_PARENTS):
        if p < 0:
            continue
        a, b = joints[j], joints[p]
        ax.plot([a[0], b[0]], [a[1], b[1]], [a[2], b[2]], color=color, linewidth=1.1, alpha=0.85)
    ax.view_init(elev=15, azim=70)
    ax.set_title(title, fontsize=9)
    ax.set_axis_off()
    _set_axes_equal(ax, verts)


def fk_mesh(body_model, pose: torch.Tensor, tran: torch.Tensor):
    pose_c = pose.float().view(-1, 24, 3, 3)
    tran_c = tran.float().view(-1, 3)
    tran_c = tran_c - tran_c[:1]
    verts, joints = [], []
    chunk = 64
    with torch.no_grad():
        for i in range(0, pose_c.shape[0], chunk):
            _, joint, vert = body_model.forward_kinematics(
                pose_c[i : i + chunk],
                tran=tran_c[i : i + chunk],
                calc_mesh=True,
            )
            verts.append(vert.cpu().numpy())
            joints.append(joint.cpu().numpy())
    return np.concatenate(verts, axis=0), np.concatenate(joints, axis=0)


def render_compare(
    body_model,
    packs: Dict[str, Tuple[np.ndarray, np.ndarray]],
    out: Path,
    n_frames: int = 4,
    title: str = "",
):
    keys = ["gt", "none", "pred_seq"]
    labels = ["GT pose", "None (R_MS→pose)", "Pred-seq cascade"]
    colors = [(0.25, 0.55, 0.30), (0.65, 0.20, 0.20), (0.90, 0.50, 0.10)]
    t_len = packs["gt"][0].shape[0]
    n_show = min(n_frames, t_len)
    frame_ids = np.linspace(0, t_len - 1, n_show, dtype=np.int64)
    faces = np.asarray(body_model.face)
    fig = plt.figure(figsize=(3.3 * n_show, 9.2))
    for row, (key, lab, col) in enumerate(zip(keys, labels, colors)):
        verts, joints = packs[key]
        for col_i, fidx in enumerate(frame_ids):
            ax = fig.add_subplot(3, n_show, row * n_show + col_i + 1, projection="3d")
            draw_pose_mesh(
                ax,
                verts[fidx],
                faces,
                joints[fidx],
                col,
                title=f"{lab}\nt={int(fidx)}" if col_i == 0 else f"t={int(fidx)}",
            )
    if title:
        fig.suptitle(title, fontsize=12)
    fig.legend(
        handles=[
            Patch(facecolor=c, label=l) for c, l in zip(colors, labels)
        ],
        loc="lower center",
        ncol=3,
        fontsize=9,
    )
    fig.tight_layout(rect=[0, 0.04, 1, 0.96])
    fig.savefig(out, dpi=140)
    plt.close(fig)
    print(f"saved {out}")


def load_cascade_models(cfg, device):
    step2 = cfg.get("step2") or {}
    pos_ckpt = resolve_path(cfg["eval"]["checkpoint"])
    rot_ckpt = resolve_path(step2["week2_dual_checkpoint"])
    pos_stats = load_norm_stats(resolve_path(cfg["data"]["out_dir"]) / "norm_stats.pt")
    rot_stats = load_norm_stats(resolve_path(step2["week2_dual_norm_stats"]))
    pos_mean = pos_stats["mean"].float().view(1, 1, -1).to(device)
    pos_std = pos_stats["std"].float().view(1, 1, -1).clamp_min(1e-6).to(device)
    rot_mean = rot_stats["mean"].float().view(1, 1, -1).to(device)
    rot_std = rot_stats["std"].float().view(1, 1, -1).clamp_min(1e-6).to(device)
    pos_model = PosClassifier(
        n_input=cfg["data"]["input_dim"],
        n_hidden=cfg["model"]["n_hidden"],
        n_lstm_layers=cfg["model"]["n_lstm_layers"],
        bidirectional=cfg["model"]["bidirectional"],
        dropout=0.0,
    ).to(device)
    pos_model.load_state_dict(torch.load(pos_ckpt, map_location=device)["model"])
    pos_model.eval()
    rot_model = RotExtrinsicDualNet(
        feat_dim=24,
        n_slots=4,
        n_hidden=cfg["model"]["n_hidden"],
        n_lstm_layers=cfg["model"]["n_lstm_layers"],
        bidirectional=cfg["model"]["bidirectional"],
        dropout=0.0,
        use_slot_onehot=True,
    ).to(device)
    rot_model.load_state_dict(torch.load(rot_ckpt, map_location=device)["model"])
    rot_model.eval()
    return pos_model, rot_model, pos_mean, pos_std, rot_mean, rot_std


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="experiments/week3_pos_rsb/configs/default.yaml")
    parser.add_argument("--metrics", type=str, default=None)
    parser.add_argument("--combo", type=str, default="lw_rp")
    parser.add_argument("--seq-ids", type=str, default="13,8")
    parser.add_argument("--skip-mesh", action="store_true")
    parser.add_argument("--max-mesh-frames", type=int, default=4)
    args = parser.parse_args()

    cfg = load_config(resolve_path(args.config))
    set_seed(cfg["experiment"]["seed"])
    fig_dir = resolve_path("experiments/week3_pos_rsb/outputs/figures")
    fig_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = resolve_path(
        args.metrics or "experiments/week3_pos_rsb/outputs/logs/metrics_step3.json"
    )
    if metrics_path.exists():
        data = json.loads(metrics_path.read_text(encoding="utf-8"))
        ds = data.get("datasets") or {}
        if ds:
            plot_grouped(
                ds, "positional_cm", "positional error (cm)",
                "Week3 step3: pose position error",
                fig_dir / "step3_pose_pos.png",
            )
            plot_grouped(
                ds, "angular_deg", "angular error (°)",
                "Week3 step3: pose angular error",
                fig_dir / "step3_pose_ang.png",
            )
            if "imuposer" in ds:
                plot_per_combo(
                    ds["imuposer"], "positional_cm", "positional error (cm)",
                    "Week3 step3 IMUPoser per combo (cm)",
                    fig_dir / "step3_imuposer_pos_combo.png",
                )
    else:
        print(f"[warn] missing {metrics_path}; skip bar charts")

    if args.skip_mesh:
        print("done (charts only).")
        return

    device = torch.device(cfg["train"]["device"] if torch.cuda.is_available() else "cpu")
    sys.modules.pop("models", None)
    if str(_MP_DIR) not in sys.path:
        sys.path.insert(0, str(_MP_DIR))
    import config as mp_cfg  # noqa: WPS433

    mp_cfg.model_config.device = device
    from articulate.model import ParametricModel  # noqa: WPS433
    from config import paths  # noqa: WPS433
    from utils.model_utils import load_model  # noqa: WPS433

    pos_model, rot_model, pos_mean, pos_std, rot_mean, rot_std = load_cascade_models(cfg, device)
    mp_ckpt = resolve_path((cfg.get("step3") or {}).get("mobileposer_checkpoint", "checkpoints/weights.pth"))
    pose_net = load_model(str(mp_ckpt))
    pose_net.to(device)
    pose_net.eval()
    body_model = ParametricModel(paths.smpl_file)

    yw, yp = parse_combo(args.combo)
    w_idx, p_idx = (0 if yw == 0 else 1), (2 if yp == 0 else 3)
    imu_path = resolve_path(cfg["data"]["processed_imuposer_file"])
    window_len = int(cfg["data"]["window_len"])
    stride = int(cfg["data"]["test_stride"])
    acc_scale = float(cfg["data"]["acc_scale"])
    offset_range = float(cfg["data"]["offset_range_deg"])
    seed = int(cfg["experiment"]["seed"])
    seq_ids = [int(x) for x in args.seq_ids.split(",") if x.strip() != ""]

    for sid in seq_ids:
        item = load_imuposer_pose_one(imu_path, sid, window_len)
        if item is None:
            print(f"[warn] seq {sid} missing or too short")
            continue
        acc, ori, pose_gt, tran_gt = item
        acc = acc[:, :5].float()
        ori = ori[:, :5].float()
        pack = cascade_calibrated(
            acc,
            ori,
            w_idx=w_idx,
            p_idx=p_idx,
            y_watch=yw,
            y_phone=yp,
            seq_i=sid,
            offset_range=offset_range,
            seed=amass_seed_offset(sid, yw, yp, seed + 1009),
            window_len=window_len,
            stride=stride,
            acc_scale=acc_scale,
            pos_model=pos_model,
            rot_model=rot_model,
            pos_mean=pos_mean,
            pos_std=pos_std,
            rot_mean=rot_mean,
            rot_std=rot_std,
            device=device,
        )
        if pack is None:
            print(f"[warn] seq {sid} cascade failed")
            continue
        slots = [w_idx, p_idx, HEAD]
        meshes = {}
        meshes["gt"] = fk_mesh(body_model, pose_gt.float(), tran_gt.float())
        for key in ("none", "pred_seq"):
            acc_c, ori_c = pack[key]
            imu = pack_mobileposer_imu(acc_c, ori_c, slots, acc_scale)
            pose_p, tran_p = predict_pose_sequence(pose_net, imu)
            meshes[key] = fk_mesh(body_model, pose_p, tran_p)
        meta = pack["meta"]
        title = (
            f"IMUPoser seq{sid:03d} {args.combo}  "
            f"slot Pred=({meta['pred_watch']},{meta['pred_phone']}) "
            f"GT=({yw},{yp})"
        )
        render_compare(
            body_model,
            meshes,
            fig_dir / f"step3_mesh_seq{sid:03d}_{args.combo}.png",
            n_frames=args.max_mesh_frames,
            title=title,
        )
    print(f"done. figures in {fig_dir}")


if __name__ == "__main__":
    main()
