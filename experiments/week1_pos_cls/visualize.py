#!/usr/bin/env python3
"""
Qualitative visualization for Week1 position classification.

Outputs under experiments/week1_pos_cls/outputs/figures/:
  - confusion matrices
  - sequence-level GT/Pred timelines (true over-time judgment)
  - SMPL mesh frames with wrist/pocket markers (GT outline + Pred fill)
  - optional GIF of mesh over time

Usage (repo root):
  python experiments/week1_pos_cls/visualize.py --split test --combo lw_rp
  python experiments/week1_pos_cls/visualize.py --split test --combo lw_rp --seq-ids 110,12
  # export GT/Pred timelines for ALL sequences (sorted by seq-id)
  python experiments/week1_pos_cls/visualize.py --split test --combo lw_rp --all-sequences
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.patches import Patch
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from torch.utils.data import DataLoader
from tqdm import tqdm

_EXP_DIR = Path(__file__).resolve().parent
_REPO = _EXP_DIR.parents[1]
_MOBILE = _REPO / "mobileposer"
if str(_EXP_DIR) not in sys.path:
    sys.path.insert(0, str(_EXP_DIR))

from dataset import load_config, resolve_path, set_seed  # noqa: E402
from dataset.pos_dataset import PosClsDataset, load_norm_stats  # noqa: E402
from models import PosClassifier  # noqa: E402

# mobileposer path only needed later for SMPL ParametricModel
if str(_MOBILE) not in sys.path:
    sys.path.append(str(_MOBILE))

# SMPL joints used as visual anchors (wrist/pocket)
# 20/21 = wrists; 1/2 = hips (pocket region)
JOINT_LW, JOINT_RW, JOINT_LP, JOINT_RP = 20, 21, 1, 2
SMPL_PARENTS = [
    -1, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 9, 12, 13, 14, 16, 17, 18, 19, 20, 21
]
COMBO_NAMES = ["LW+LP", "LW+RP", "RW+LP", "RW+RP"]


def combo_tag(watch_side: int, phone_side: int) -> str:
    return f"{'lw' if watch_side == 0 else 'rw'}_{'lp' if phone_side == 0 else 'rp'}"


def parse_combo(s: str) -> Tuple[int, int]:
    tok = s.strip().lower().replace("+", "_").replace("-", "_")
    w, p = tok.split("_")
    return (0 if w == "lw" else 1), (0 if p == "lp" else 1)


def resolve_test_pt(data_dir: Path, watch_side: int, phone_side: int) -> Path:
    named = data_dir / f"imuposer_test_{combo_tag(watch_side, phone_side)}.pt"
    if named.exists():
        return named
    legacy = data_dir / "imuposer_test.pt"
    if legacy.exists():
        return legacy
    raise FileNotFoundError(named)


def list_imuposer_motions(raw_dir: Path) -> List[Dict[str, str]]:
    subjects = {f"P{i}" for i in range(1, 11)}
    items = []
    for pid_path in sorted(raw_dir.iterdir()):
        if pid_path.name not in subjects:
            continue
        for fpath in sorted(pid_path.iterdir()):
            parts = fpath.name.split(".")
            motion = parts[1].strip() if len(parts) >= 3 else fpath.stem
            items.append({"pid": pid_path.name, "motion": motion, "file": fpath.name})
    return items


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
def predict_all(model, loader, device):
    model.eval()
    yw_all, yp_all, pw_all, pp_all = [], [], [], []
    for x, yw, yp in tqdm(loader, desc="predict"):
        x = x.to(device)
        lw, lp = model(x)
        yw_all.append(yw.numpy())
        yp_all.append(yp.numpy())
        pw_all.append(lw.argmax(-1).cpu().numpy())
        pp_all.append(lp.argmax(-1).cpu().numpy())
    return (
        np.concatenate(yw_all),
        np.concatenate(yp_all),
        np.concatenate(pw_all),
        np.concatenate(pp_all),
    )


def group_by_sequence(meta: List[dict], yw, yp, pw, pp) -> Dict[int, dict]:
    buckets: Dict[int, List[int]] = defaultdict(list)
    key_map = {}
    next_id = 0
    for i, m in enumerate(meta):
        if m is None:
            continue
        if "seq" in m:
            sid = int(m["seq"])
        else:
            # AMASS: (source, seq_local, combo) as one instance
            raw = (m.get("source"), m.get("seq_local"), m.get("y_watch"), m.get("y_phone"))
            if raw not in key_map:
                key_map[raw] = next_id
                next_id += 1
            sid = key_map[raw]
        buckets[sid].append(i)
    out = {}
    for seq, idxs in buckets.items():
        idxs = sorted(idxs, key=lambda i: int(meta[i].get("start", 0)))
        starts = np.array([int(meta[i].get("start", 0)) for i in idxs], dtype=np.int64)
        out[seq] = {
            "idxs": idxs,
            "starts": starts,
            "yw": np.array([yw[i] for i in idxs], dtype=np.int64),
            "yp": np.array([yp[i] for i in idxs], dtype=np.int64),
            "pw": np.array([pw[i] for i in idxs], dtype=np.int64),
            "pp": np.array([pp[i] for i in idxs], dtype=np.int64),
        }
    return out


def expand_to_frames(
    starts: np.ndarray,
    values: np.ndarray,
    seq_len: int,
    stride: int,
    window_len: int,
) -> np.ndarray:
    """Assign each window prediction to frames [start, start+stride)."""
    out = np.full(seq_len, -1, dtype=np.int64)
    for st, v in zip(starts.tolist(), values.tolist()):
        end = min(seq_len, st + stride)
        if end <= st:
            end = min(seq_len, st + 1)
        out[st:end] = int(v)
    # fill trailing frames covered by last window
    if len(starts):
        last = int(starts[-1])
        out[last : min(seq_len, last + window_len)] = int(values[-1])
    # forward-fill leading gaps
    last_v = -1
    for t in range(seq_len):
        if out[t] >= 0:
            last_v = out[t]
        elif last_v >= 0:
            out[t] = last_v
    return out


def plot_sequence_timeline(
    seq_id: int,
    pack: dict,
    seq_len: int,
    stride: int,
    window_len: int,
    title: str,
    out: Path,
):
    """
    Single-panel timeline in paper style:
      x = Frame Number
      y = position id: 0 LW, 1 RW, 2 LP, 3 RP
      curves: GT / Pred for watch & phone
    """
    t = np.arange(seq_len)
    yw_f = expand_to_frames(pack["starts"], pack["yw"], seq_len, stride, window_len)
    yp_f = expand_to_frames(pack["starts"], pack["yp"], seq_len, stride, window_len)
    pw_f = expand_to_frames(pack["starts"], pack["pw"], seq_len, stride, window_len)
    pp_f = expand_to_frames(pack["starts"], pack["pp"], seq_len, stride, window_len)

    # unify to 0/1/2/3 axis (phone -> 2/3)
    gt_watch = yw_f.astype(float)
    pred_watch = pw_f.astype(float)
    gt_phone = yp_f.astype(float) + 2.0
    pred_phone = pp_f.astype(float) + 2.0

    joint_ok = (pw_f == yw_f) & (pp_f == yp_f) & (yw_f >= 0)
    acc = float(joint_ok.mean()) if len(joint_ok) else 0.0

    fig, ax = plt.subplots(figsize=(10, 3.8))
    # Paper-style: white bg, no grid; GT solid / Pred dashed
    ax.set_facecolor("white")
    fig.patch.set_facecolor("white")
    kw = dict(linewidth=1.8, drawstyle="steps-post", zorder=3)
    ax.plot(t, gt_watch, color="#1f77b4", linestyle="-", label="GT Watch", **kw)
    ax.plot(t, pred_watch, color="#1f77b4", linestyle="--", label="Pred Watch", **kw)
    ax.plot(t, gt_phone, color="#d62728", linestyle="-", label="GT Phone", **kw)
    ax.plot(t, pred_phone, color="#d62728", linestyle="--", label="Pred Phone", **kw)

    ax.set_xlabel("Frame Number", fontsize=12)
    ax.set_ylabel("Device Position", fontsize=12)
    ax.set_yticks([0, 1, 2, 3])
    ax.set_yticklabels(
        [
            "0 Left Hand",
            "1 Right Hand",
            "2 Left Pocket",
            "3 Right Pocket",
        ]
    )
    ax.set_ylim(-0.4, 3.4)
    ax.set_xlim(0, max(seq_len - 1, 1))
    ax.grid(False)
    for spine in ax.spines.values():
        spine.set_color("black")
        spine.set_linewidth(1.0)
    ax.tick_params(direction="out", length=4, labelsize=10)
    ax.legend(loc="upper right", frameon=True, fontsize=9, edgecolor="black")
    ax.set_title(f"{title}\nseq={seq_id}  joint-acc={acc:.3f}", fontsize=11)
    fig.tight_layout()
    fig.savefig(out, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def pick_sequences(groups: Dict[int, dict], n: int) -> List[int]:
    """Pick one good + one bad sequence (then fill if n>2)."""
    scored = []
    for sid, pack in groups.items():
        if len(pack["idxs"]) < 8:
            continue
        ok = float(((pack["pw"] == pack["yw"]) & (pack["pp"] == pack["yp"])).mean())
        scored.append((sid, ok, len(pack["idxs"])))
    if not scored:
        return list(groups.keys())[:n]
    scored.sort(key=lambda x: (x[1], -x[2]))  # acc asc, prefer longer
    bad = scored[0][0]
    good = scored[-1][0]
    out = [good, bad] if good != bad else [good]
    # extra mid ones if requested
    if n > 2:
        mid = [s[0] for s in scored[1:-1]]
        for s in mid:
            if s not in out:
                out.append(s)
            if len(out) >= n:
                break
    return out[:n]


def _set_axes_equal(ax, pts: np.ndarray):
    mins = pts.min(axis=0)
    maxs = pts.max(axis=0)
    center = (mins + maxs) / 2
    radius = 0.5 * np.max(maxs - mins) + 1e-3
    ax.set_xlim(center[0] - radius, center[0] + radius)
    ax.set_ylim(center[1] - radius, center[1] + radius)
    ax.set_zlim(center[2] - radius, center[2] + radius)


def draw_mesh_frame(
    ax,
    verts: np.ndarray,
    faces: np.ndarray,
    joints: np.ndarray,
    yw: int,
    yp: int,
    pw: int,
    pp: int,
    face_stride: int = 3,
    title: str = "",
):
    """
    verts: [V,3], joints: [24,3]
    GT = hollow marker at true device joint; Pred = filled marker (green ok / red wrong)
    """
    # light mesh
    tri = verts[faces[::face_stride]]
    mesh = Poly3DCollection(tri, alpha=0.12, linewidths=0.05)
    mesh.set_facecolor((0.7, 0.75, 0.85, 0.25))
    mesh.set_edgecolor((0.5, 0.5, 0.55, 0.1))
    ax.add_collection3d(mesh)

    # skeleton
    for j, p in enumerate(SMPL_PARENTS):
        if p < 0:
            continue
        a, b = joints[j], joints[p]
        ax.plot([a[0], b[0]], [a[1], b[1]], [a[2], b[2]], color="gray", linewidth=1.0, alpha=0.7)

    watch_gt = JOINT_LW if yw == 0 else JOINT_RW
    phone_gt = JOINT_LP if yp == 0 else JOINT_RP
    watch_pr = JOINT_LW if pw == 0 else JOINT_RW
    phone_pr = JOINT_LP if pp == 0 else JOINT_RP

    # GT hollow
    for jid, color, label in [
        (watch_gt, "green", "GT watch"),
        (phone_gt, "blue", "GT phone"),
    ]:
        p = joints[jid]
        ax.scatter([p[0]], [p[1]], [p[2]], s=120, facecolors="none", edgecolors=color, linewidths=2.5, label=label)

    # Pred filled
    w_ok = pw == yw
    p_ok = pp == yp
    ax.scatter(
        [joints[watch_pr][0]],
        [joints[watch_pr][1]],
        [joints[watch_pr][2]],
        s=70,
        c=("limegreen" if w_ok else "red"),
        marker="o",
        label="Pred watch",
        depthshade=False,
    )
    ax.scatter(
        [joints[phone_pr][0]],
        [joints[phone_pr][1]],
        [joints[phone_pr][2]],
        s=70,
        c=("dodgerblue" if p_ok else "magenta"),
        marker="s",
        label="Pred phone",
        depthshade=False,
    )

    ax.view_init(elev=15, azim=70)
    ax.set_title(title, fontsize=9)
    ax.set_axis_off()
    _set_axes_equal(ax, verts)


def render_sequence_mesh(
    body_model,
    pose: torch.Tensor,
    tran: torch.Tensor,
    pack: dict,
    stride: int,
    window_len: int,
    out_png: Path,
    out_gif: Optional[Path] = None,
    max_frames: int = 24,
    fps: int = 10,
):
    """Render a strip of mesh frames + optional GIF with position markers."""
    device = next(body_model.parameters()).device if hasattr(body_model, "parameters") else "cpu"
    # ParametricModel may not be nn.Module; place tensors on cpu/cuda consistently
    try:
        # articulate ParametricModel stores device in construction
        pass
    except Exception:
        pass

    T = pose.shape[0]
    yw_f = expand_to_frames(pack["starts"], pack["yw"], T, stride, window_len)
    yp_f = expand_to_frames(pack["starts"], pack["yp"], T, stride, window_len)
    pw_f = expand_to_frames(pack["starts"], pack["pw"], T, stride, window_len)
    pp_f = expand_to_frames(pack["starts"], pack["pp"], T, stride, window_len)

    # subsample frames
    n_show = min(max_frames, T)
    frame_ids = np.linspace(0, T - 1, n_show, dtype=np.int64)

    # FK in chunks
    pose_c = pose.float()
    tran_c = tran.float()
    # root-relative translation for stable camera
    tran_c = tran_c - tran_c[:1]

    verts_all = []
    joints_all = []
    chunk = 64
    with torch.no_grad():
        for i in range(0, T, chunk):
            grot, joint, vert = body_model.forward_kinematics(
                pose_c[i : i + chunk],
                tran=tran_c[i : i + chunk],
                calc_mesh=True,
            )
            verts_all.append(vert.cpu().numpy())
            joints_all.append(joint.cpu().numpy())
    verts_all = np.concatenate(verts_all, axis=0)
    joints_all = np.concatenate(joints_all, axis=0)
    faces = np.asarray(body_model.face)

    # montage: up to 8 panels
    n_panel = min(8, n_show)
    panel_ids = np.linspace(0, n_show - 1, n_panel, dtype=np.int64)
    fig = plt.figure(figsize=(3.2 * n_panel, 4.2))
    for pi, local_i in enumerate(panel_ids):
        fidx = int(frame_ids[local_i])
        ax = fig.add_subplot(1, n_panel, pi + 1, projection="3d")
        ok = (pw_f[fidx] == yw_f[fidx]) and (pp_f[fidx] == yp_f[fidx])
        draw_mesh_frame(
            ax,
            verts_all[fidx],
            faces,
            joints_all[fidx],
            int(yw_f[fidx]),
            int(yp_f[fidx]),
            int(pw_f[fidx]),
            int(pp_f[fidx]),
            title=f"t={fidx}\n{'OK' if ok else 'WRONG'}",
        )
    legend_elems = [
        Patch(facecolor="none", edgecolor="green", label="GT watch"),
        Patch(facecolor="none", edgecolor="blue", label="GT phone"),
        Patch(facecolor="limegreen", label="Pred watch OK"),
        Patch(facecolor="red", label="Pred watch wrong"),
        Patch(facecolor="dodgerblue", label="Pred phone OK"),
        Patch(facecolor="magenta", label="Pred phone wrong"),
    ]
    fig.legend(handles=legend_elems, loc="upper center", ncol=3, fontsize=8)
    fig.tight_layout(rect=[0, 0, 1, 0.90])
    fig.savefig(out_png, dpi=140)
    plt.close(fig)

    if out_gif is not None:
        import imageio

        frames = []
        for local_i, fidx in enumerate(tqdm(frame_ids, desc="gif frames")):
            fig = plt.figure(figsize=(5, 6))
            ax = fig.add_subplot(111, projection="3d")
            ok = (pw_f[fidx] == yw_f[fidx]) and (pp_f[fidx] == yp_f[fidx])
            draw_mesh_frame(
                ax,
                verts_all[fidx],
                faces,
                joints_all[fidx],
                int(yw_f[fidx]),
                int(yp_f[fidx]),
                int(pw_f[fidx]),
                int(pp_f[fidx]),
                title=f"frame {fidx}  {'OK' if ok else 'WRONG'}",
            )
            fig.tight_layout()
            fig.canvas.draw()
            # Agg buffer (compatible across matplotlib versions)
            buf = np.asarray(fig.canvas.buffer_rgba())[:, :, :3].copy()
            frames.append(buf)
            plt.close(fig)
        imageio.mimsave(out_gif, frames, fps=fps)
        print(f"saved gif {out_gif}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="experiments/week1_pos_cls/configs/default.yaml")
    parser.add_argument("--split", type=str, choices=["val", "test"], default="test")
    parser.add_argument("--combo", type=str, default="lw_rp", help="IMUPoser combo, e.g. lw_rp")
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--seq-ids", type=str, default=None, help="Comma-separated sequence ids")
    parser.add_argument(
        "--all-sequences",
        action="store_true",
        help="Export timelines for ALL sequences in ascending seq-id order",
    )
    parser.add_argument(
        "--timeline-only",
        action="store_true",
        help="Only save timelines (skip mesh). Implied by --all-sequences unless --mesh is set",
    )
    parser.add_argument(
        "--mesh",
        action="store_true",
        help="Also render mesh montages (default on for few seqs; off for --all-sequences)",
    )
    parser.add_argument("--num-sequences", type=int, default=None)
    parser.add_argument("--gif", action="store_true", help="Also export mesh GIF per sequence")
    parser.add_argument("--max-mesh-frames", type=int, default=24)
    parser.add_argument("--metrics-json", type=str, default=None)
    args = parser.parse_args()

    cfg = load_config(resolve_path(args.config))
    set_seed(cfg["experiment"]["seed"])

    watch_side, phone_side = parse_combo(args.combo)
    tag = combo_tag(watch_side, phone_side)
    n_seq_vis = args.num_sequences if args.num_sequences is not None else 2

    if args.all_sequences:
        fig_dir = resolve_path("experiments/week1_pos_cls/outputs/figures") / f"timelines_all_{args.split}_{tag}"
    else:
        fig_dir = resolve_path("experiments/week1_pos_cls/outputs/figures")
    fig_dir.mkdir(parents=True, exist_ok=True)

    do_mesh = False if args.timeline_only else True
    if args.all_sequences and not args.mesh:
        do_mesh = False
    if args.mesh:
        do_mesh = True
    if args.timeline_only:
        do_mesh = False

    # ---- confusion matrices ----
    metrics_path = resolve_path(
        args.metrics_json
        or (
            f"experiments/week1_pos_cls/outputs/logs/metrics_test_{tag}.json"
            if args.split == "test"
            else "experiments/week1_pos_cls/outputs/logs/metrics_val.json"
        )
    )
    cm_dir = resolve_path("experiments/week1_pos_cls/outputs/figures")
    cm_dir.mkdir(parents=True, exist_ok=True)
    if metrics_path.exists() and not args.all_sequences:
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        plot_cm(np.array(metrics["cm_watch"]), ["LW", "RW"], f"Watch CM ({args.split}/{tag})", cm_dir / f"cm_watch_{args.split}_{tag}.png")
        plot_cm(np.array(metrics["cm_phone"]), ["LP", "RP"], f"Phone CM ({args.split}/{tag})", cm_dir / f"cm_phone_{args.split}_{tag}.png")
        plot_cm(np.array(metrics["cm_joint4"]), COMBO_NAMES, f"Joint-4 CM ({args.split}/{tag})", cm_dir / f"cm_joint4_{args.split}_{tag}.png")
        print(f"saved confusion matrices -> {cm_dir}")

    # ---- load data / model ----
    data_dir = resolve_path(cfg["data"]["out_dir"])
    if args.split == "val":
        pt = data_dir / "amass_val.pt"
        pose_src = None
    else:
        pt = resolve_test_pt(data_dir, watch_side, phone_side)
        pose_src = resolve_path(cfg["data"]["processed_imuposer_file"])

    stats_pt = data_dir / "norm_stats.pt"
    ckpt_path = resolve_path(args.checkpoint or cfg["eval"]["checkpoint"])
    for p in (pt, stats_pt, ckpt_path):
        if not p.exists():
            raise FileNotFoundError(p)

    device = torch.device(cfg["train"]["device"] if torch.cuda.is_available() else "cpu")
    stats = load_norm_stats(stats_pt)
    ds = PosClsDataset(pt, norm_stats=stats, normalize=True)
    loader = DataLoader(ds, batch_size=cfg["eval"]["batch_size"], shuffle=False, num_workers=0)

    ckpt = torch.load(ckpt_path, map_location=device)
    model = PosClassifier(
        n_input=cfg["data"]["input_dim"],
        n_hidden=cfg["model"]["n_hidden"],
        n_lstm_layers=cfg["model"]["n_lstm_layers"],
        bidirectional=cfg["model"]["bidirectional"],
        dropout=0.0,
    ).to(device)
    model.load_state_dict(ckpt["model"])

    yw, yp, pw, pp = predict_all(model, loader, device)
    groups = group_by_sequence(ds.meta, yw, yp, pw, pp)

    if args.all_sequences:
        seq_ids = sorted(groups.keys())
    elif args.seq_ids:
        seq_ids = [int(x) for x in args.seq_ids.split(",") if x.strip() != ""]
    else:
        seq_ids = pick_sequences(groups, n_seq_vis)

    motion_map = []
    raw_dir = resolve_path("data/raw/IMUPoser")
    if raw_dir.exists():
        motion_map = list_imuposer_motions(raw_dir)

    stride = int(cfg["data"]["test_stride"] if args.split == "test" else cfg["data"]["train_stride"])
    window_len = int(cfg["data"]["window_len"])

    pose_data = None
    body_model = None
    if do_mesh and args.split == "test" and pose_src is not None and pose_src.exists():
        pose_data = torch.load(pose_src, map_location="cpu")
        from articulate.model import ParametricModel  # noqa: WPS433
        from config import paths  # noqa: WPS433

        body_model = ParametricModel(paths.smpl_file)

    index_rows = []
    for sid in seq_ids:
        if sid not in groups:
            print(f"[warn] seq {sid} not in predictions, skip")
            continue
        pack = groups[sid]
        motion = ""
        pid = ""
        motion_name = ""
        if sid < len(motion_map):
            pid = motion_map[sid]["pid"]
            motion_name = motion_map[sid]["motion"]
            motion = f"{pid}/{motion_name}"

        if pose_data is not None and sid < len(pose_data["pose"]):
            seq_len = int(pose_data["pose"][sid].shape[0])
        else:
            seq_len = int(pack["starts"][-1] + window_len)

        ok = float(((pack["pw"] == pack["yw"]) & (pack["pp"] == pack["yp"])).mean())
        quality = "good" if ok >= 0.9 else ("bad" if ok < 0.5 else "mid")
        title = f"{args.split} {tag}  {motion}  [{quality}]".strip()
        # zero-padded seq id so files sort in seq order
        tl_path = fig_dir / f"timeline_seq{sid:03d}_{quality}_{args.split}_{tag}.png"
        plot_sequence_timeline(sid, pack, seq_len, stride, window_len, title, tl_path)
        print(f"saved timeline {tl_path} (joint-acc={ok:.3f})")
        index_rows.append(
            {
                "seq": sid,
                "pid": pid,
                "motion": motion_name,
                "joint_acc": round(ok, 4),
                "quality": quality,
                "n_windows": len(pack["idxs"]),
                "seq_len": seq_len,
                "timeline": str(tl_path.name),
            }
        )

        if body_model is not None and pose_data is not None and sid < len(pose_data["pose"]):
            mesh_png = fig_dir / f"mesh_seq{sid:03d}_{args.split}_{tag}.png"
            mesh_gif = fig_dir / f"mesh_seq{sid:03d}_{args.split}_{tag}.gif" if args.gif else None
            render_sequence_mesh(
                body_model,
                pose_data["pose"][sid],
                pose_data["tran"][sid],
                pack,
                stride,
                window_len,
                mesh_png,
                out_gif=mesh_gif,
                max_frames=args.max_mesh_frames,
                fps=max(1, int(cfg["data"]["fps"] // 3)),
            )
            print(f"saved mesh montage {mesh_png}")

    if index_rows:
        index_path = fig_dir / f"timeline_index_{args.split}_{tag}.json"
        index_path.write_text(json.dumps(index_rows, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"saved index {index_path} ({len(index_rows)} sequences)")

    print(f"done. figures in {fig_dir}")


if __name__ == "__main__":
    main()
