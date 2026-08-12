#!/usr/bin/env python3
"""
Qualitative visualization for Week2 R_SB extrinsic (D5-B style timelines).

Paper-style single panel (aligned with Week1 visualize.py):
  x = Frame Number
  y = Error Angle (°)
  curves:
    未预测       — no calib: geodesic(ori_obs, ori_clean)
    预测校准后   — Learned:  geodesic(ori_cal(R̂), ori_clean)
    GT           — Oracle:   geodesic(ori_cal(R_SB), ori_clean)  (~0°)

Protocol (IMUPoser test, same as eval_imuposer_d5*):
  treat recorded stream as clean → inject one R_SB per sequence → recover / calib.

Usage (repo root):
  # single-device, auto pick good/bad sequences for slot RP (one panel)
  python experiments/week2_rot_ext/visualize.py --slot 3

  # single-device ×2: same seq, predict watch then phone, stack two panels
  python experiments/week2_rot_ext/visualize.py --pair --combo lw_rp

  # dual-device joint model (one forward, two heads)
  python experiments/week2_rot_ext/visualize.py --dual --combo lw_rp

  # specific sequences
  python experiments/week2_rot_ext/visualize.py --slot 2 --seq-ids 0,12,110

  # export all sequences under a cap
  python experiments/week2_rot_ext/visualize.py --slot 3 --all-sequences --max-seqs 40
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from tqdm import tqdm

_EXP_DIR = Path(__file__).resolve().parent
_REPO = _EXP_DIR.parents[1]
if str(_EXP_DIR) not in sys.path:
    sys.path.insert(0, str(_EXP_DIR))

from dataset import (  # noqa: E402
    SLOT_NAMES,
    apply_mount_offset,
    combo_to_indices,
    geodesic_angle_deg,
    load_config,
    make_device_features,
    resolve_path,
    sample_random_offsets,
    set_seed,
)
from models import RotExtrinsicDualNet, RotExtrinsicNet  # noqa: E402

# Week1-like palette; legend = 未预测 / 预测校准后 / GT (English for font reliability)
COLOR_NONE = "#7f7f7f"
COLOR_LEARNED = "#1f77b4"
COLOR_GT = "#2ca02c"
LABEL_NONE = "None"
LABEL_LEARNED = "Learned"
LABEL_GT = "GT"


def parse_combo(s: str) -> Tuple[int, int]:
    tok = s.strip().lower().replace("+", "_").replace("-", "_")
    w, p = tok.split("_")
    return (0 if w == "lw" else 1), (0 if p == "lp" else 1)


def combo_tag(watch_side: int, phone_side: int) -> str:
    return f"{'lw' if watch_side == 0 else 'rw'}_{'lp' if phone_side == 0 else 'rp'}"


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


def expand_rot_to_frames(
    starts: Sequence[int],
    rots: torch.Tensor,
    seq_len: int,
    stride: int,
    window_len: int,
) -> torch.Tensor:
    """
    Expand per-window R_hat [N,3,3] to per-frame [T,3,3]
    (same coverage rule as Week1 expand_to_frames).
    """
    out = torch.zeros(seq_len, 3, 3, dtype=rots.dtype)
    out[:] = torch.eye(3, dtype=rots.dtype)
    assigned = torch.zeros(seq_len, dtype=torch.bool)
    for st, r in zip(starts, rots):
        end = min(seq_len, int(st) + stride)
        if end <= st:
            end = min(seq_len, int(st) + 1)
        out[st:end] = r
        assigned[st:end] = True
    if len(starts):
        last = int(starts[-1])
        out[last : min(seq_len, last + window_len)] = rots[-1]
        assigned[last : min(seq_len, last + window_len)] = True
    last_r = None
    for t in range(seq_len):
        if assigned[t]:
            last_r = out[t]
        elif last_r is not None:
            out[t] = last_r
            assigned[t] = True
    return out


def plot_error_timeline(
    frames: np.ndarray,
    err_none: np.ndarray,
    err_learned: np.ndarray,
    err_gt: np.ndarray,
    title: str,
    out: Path,
    ylabel: str = "Error Angle (deg)",
):
    """Week1-style white single panel: three error curves vs Frame Number."""
    fig, ax = plt.subplots(figsize=(10, 3.8))
    ax.set_facecolor("white")
    fig.patch.set_facecolor("white")
    kw = dict(linewidth=1.8, zorder=3)
    ax.plot(frames, err_none, color=COLOR_NONE, linestyle="-", label=LABEL_NONE, **kw)
    ax.plot(
        frames,
        err_learned,
        color=COLOR_LEARNED,
        linestyle="-",
        label=LABEL_LEARNED,
        **kw,
    )
    ax.plot(frames, err_gt, color=COLOR_GT, linestyle="-", label=LABEL_GT, **kw)

    ax.set_xlabel("Frame Number", fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_xlim(0, max(int(frames[-1]) if len(frames) else 1, 1))
    ymax = float(
        max(
            err_none.max() if len(err_none) else 0,
            err_learned.max() if len(err_learned) else 0,
            err_gt.max() if len(err_gt) else 0,
            1.0,
        )
    )
    ax.set_ylim(0, ymax * 1.15)
    ax.grid(False)
    for spine in ax.spines.values():
        spine.set_color("black")
        spine.set_linewidth(1.0)
    ax.tick_params(direction="out", length=4, labelsize=10)
    ax.set_title(title, fontsize=11)
    ax.legend(loc="upper right", frameon=True, fontsize=9, edgecolor="black")
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def plot_dual_error_timeline(
    frames: np.ndarray,
    watch: Tuple[np.ndarray, np.ndarray, np.ndarray],
    phone: Tuple[np.ndarray, np.ndarray, np.ndarray],
    title: str,
    out: Path,
):
    """Two stacked panels (watch / phone), same three curves."""
    fig, axes = plt.subplots(2, 1, figsize=(10, 6.2), sharex=True)
    fig.patch.set_facecolor("white")
    labels = (LABEL_NONE, LABEL_LEARNED, LABEL_GT)
    colors = (COLOR_NONE, COLOR_LEARNED, COLOR_GT)
    panel_names = ("Watch", "Phone")
    series = (watch, phone)
    for ax, name, (e0, e1, e2) in zip(axes, panel_names, series):
        ax.set_facecolor("white")
        for e, lab, c in zip((e0, e1, e2), labels, colors):
            ax.plot(frames, e, color=c, linestyle="-", linewidth=1.8, label=lab)
        ax.set_ylabel(f"{name} Error (deg)", fontsize=11)
        ymax = float(max(e0.max(), e1.max(), e2.max(), 1.0))
        ax.set_ylim(0, ymax * 1.15)
        ax.grid(False)
        for spine in ax.spines.values():
            spine.set_color("black")
            spine.set_linewidth(1.0)
        ax.tick_params(direction="out", length=4, labelsize=10)
        ax.legend(loc="upper right", frameon=True, fontsize=8, edgecolor="black")
        mean_l = float(e1.mean())
        ax.set_title(f"{name}  mean(Learned)={mean_l:.2f} deg", fontsize=10)
    axes[-1].set_xlabel("Frame Number", fontsize=12)
    axes[-1].set_xlim(0, max(int(frames[-1]) if len(frames) else 1, 1))
    fig.suptitle(title, fontsize=11, y=1.02)
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def _normalize_feat(x: torch.Tensor, mean: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
    return (x - mean) / std


@torch.no_grad()
def predict_rsb_windows_single(
    model: RotExtrinsicNet,
    acc_obs: torch.Tensor,
    ori_obs: torch.Tensor,
    slot: int,
    window_len: int,
    stride: int,
    acc_scale: float,
    mean: torch.Tensor,
    std: torch.Tensor,
    device: torch.device,
) -> Tuple[List[int], torch.Tensor]:
    starts: List[int] = []
    hats: List[torch.Tensor] = []
    t_len = acc_obs.shape[0]
    for start in range(0, t_len - window_len + 1, stride):
        acc_w = acc_obs[start : start + window_len]
        ori_w = ori_obs[start : start + window_len]
        feat = make_device_features(acc_w, ori_w, acc_scale)
        x = _normalize_feat(feat, mean, std).unsqueeze(0).to(device)
        slot_t = torch.tensor([slot], device=device)
        _, r_hat = model(x, slot_t)
        starts.append(start)
        hats.append(r_hat[0].cpu())
    if not hats:
        return [], torch.zeros(0, 3, 3)
    return starts, torch.stack(hats, dim=0)


@torch.no_grad()
def predict_rsb_windows_dual(
    model: RotExtrinsicDualNet,
    aw: torch.Tensor,
    ow: torch.Tensor,
    ap: torch.Tensor,
    op: torch.Tensor,
    watch_slot: int,
    phone_slot: int,
    window_len: int,
    stride: int,
    acc_scale: float,
    mean: torch.Tensor,
    std: torch.Tensor,
    device: torch.device,
) -> Tuple[List[int], torch.Tensor, torch.Tensor]:
    starts: List[int] = []
    hw: List[torch.Tensor] = []
    hp: List[torch.Tensor] = []
    t_len = aw.shape[0]
    sw = torch.tensor([watch_slot], device=device)
    sp = torch.tensor([phone_slot], device=device)
    for start in range(0, t_len - window_len + 1, stride):
        fw = make_device_features(
            aw[start : start + window_len], ow[start : start + window_len], acc_scale
        )
        fp = make_device_features(
            ap[start : start + window_len], op[start : start + window_len], acc_scale
        )
        feat = torch.cat([fw, fp], dim=-1)
        x = _normalize_feat(feat, mean, std).unsqueeze(0).to(device)
        ( _, r_w), (_, r_p) = model(x, sw, sp)
        starts.append(start)
        hw.append(r_w[0].cpu())
        hp.append(r_p[0].cpu())
    if not hw:
        return [], torch.zeros(0, 3, 3), torch.zeros(0, 3, 3)
    return starts, torch.stack(hw, 0), torch.stack(hp, 0)


def frame_ori_errors(
    ori_obs: torch.Tensor,
    ori_clean: torch.Tensor,
    r_hat_f: torch.Tensor,
    r_gt: torch.Tensor,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Per-frame geodesic ori error under None / Learned / GT calib."""
    # None
    err_none = geodesic_angle_deg(ori_obs, ori_clean).numpy()
    # Learned: ori_cal[t] = ori_obs[t] @ R_hat[t]^T
    ori_l = torch.matmul(ori_obs, r_hat_f.transpose(-1, -2))
    err_learned = geodesic_angle_deg(ori_l, ori_clean).numpy()
    # GT / Oracle: ori_cal = ori_obs @ R_SB^T
    ori_o = torch.matmul(ori_obs, r_gt.transpose(0, 1))
    err_gt = geodesic_angle_deg(ori_o, ori_clean).numpy()
    return err_none, err_learned, err_gt


def mean_rsb_err_deg(hats: torch.Tensor, r_gt: torch.Tensor) -> float:
    """Mean geodesic(R_hat_window, R_SB) over windows."""
    if hats.numel() == 0:
        return float("nan")
    return float(geodesic_angle_deg(hats, r_gt.expand_as(hats)).mean())


def quality_tag(mean_learned: float, good_th: float, bad_th: float) -> str:
    if mean_learned <= good_th:
        return "good"
    if mean_learned >= bad_th:
        return "bad"
    return "mid"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        default="experiments/week2_rot_ext/configs/default.yaml",
    )
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument(
        "--slot",
        type=int,
        default=3,
        choices=[0, 1, 2, 3],
        help="Single-device IMU slot for one-panel mode (default RP=3)",
    )
    parser.add_argument(
        "--pair",
        action="store_true",
        help=(
            "Single-device model ×2: run watch then phone on the same sequence "
            "and stack two panels (uses --combo). Mutually exclusive with --dual."
        ),
    )
    parser.add_argument(
        "--dual",
        action="store_true",
        help="Use dual-device joint model; requires --combo",
    )
    parser.add_argument(
        "--combo",
        type=str,
        default="lw_rp",
        help="Watch+phone combo for --dual / --pair, e.g. lw_rp / rw_lp",
    )
    parser.add_argument("--seq-ids", type=str, default=None, help="Comma-separated seq ids")
    parser.add_argument(
        "--all-sequences",
        action="store_true",
        help="Export all sequences (up to --max-seqs)",
    )
    parser.add_argument("--max-seqs", type=int, default=None)
    parser.add_argument(
        "--n-pick",
        type=int,
        default=2,
        help="Auto-pick this many sequences (good+bad) when --seq-ids omitted",
    )
    parser.add_argument("--offset-seed", type=int, default=None)
    args = parser.parse_args()
    if args.dual and args.pair:
        raise SystemExit("Use either --dual or --pair, not both.")

    cfg = load_config(resolve_path(args.config))
    set_seed(cfg["experiment"]["seed"])
    device = torch.device(
        cfg["train"]["device"] if torch.cuda.is_available() else "cpu"
    )

    src = resolve_path(cfg["data"]["processed_imuposer_file"])
    if not src.exists():
        raise FileNotFoundError(src)
    data = torch.load(src, map_location="cpu")
    accs, oris = data["acc"], data["ori"]
    max_seqs = args.max_seqs or int(cfg["eval"].get("imuposer_max_seqs", 40))
    n_seq = min(len(accs), max_seqs)
    window_len = int(cfg["data"]["window_len"])
    stride = int(cfg["data"]["test_stride"])
    acc_scale = float(cfg["data"]["acc_scale"])
    offset_range = float(cfg["data"]["offset_range_deg"])
    # dual / pair share D5-dual-like seed offset; single-slot keeps legacy +7
    offset_seed = (
        args.offset_seed
        if args.offset_seed is not None
        else int(cfg["experiment"]["seed"]) + (9 if (args.dual or args.pair) else 7)
    )

    raw_imuposer = resolve_path("data/raw/IMUPoser")
    motions = list_imuposer_motions(raw_imuposer) if raw_imuposer.exists() else []

    w_idx = p_idx = -1
    if args.dual or args.pair:
        watch_side, phone_side = parse_combo(args.combo)
        w_idx, p_idx = combo_to_indices(watch_side, phone_side)

    if args.dual:
        tag = f"dual_{combo_tag(watch_side, phone_side)}"
        ckpt_path = resolve_path(
            args.checkpoint
            or "experiments/week2_rot_ext/outputs/checkpoints/best_rot_err_dual.pt"
        )
        stats_path = resolve_path(cfg["data"]["out_dir"]) / "norm_stats_dual.pt"
        model = RotExtrinsicDualNet(
            feat_dim=24,
            n_slots=cfg["data"]["n_slots"],
            n_hidden=cfg["model"]["n_hidden"],
            n_lstm_layers=cfg["model"]["n_lstm_layers"],
            bidirectional=cfg["model"]["bidirectional"],
            dropout=0.0,
            use_slot_onehot=cfg["model"]["use_slot_onehot"],
        ).to(device)
    elif args.pair:
        # Same single-device weights; run watch then phone (D6 single_device_x2 style)
        tag = f"single_pair_{combo_tag(watch_side, phone_side)}"
        ckpt_path = resolve_path(args.checkpoint or cfg["eval"]["checkpoint"])
        stats_path = resolve_path(cfg["data"]["out_dir"]) / "norm_stats.pt"
        model = RotExtrinsicNet(
            feat_dim=cfg["data"]["feat_dim"],
            n_slots=cfg["data"]["n_slots"],
            n_hidden=cfg["model"]["n_hidden"],
            n_lstm_layers=cfg["model"]["n_lstm_layers"],
            bidirectional=cfg["model"]["bidirectional"],
            dropout=0.0,
            use_slot_onehot=cfg["model"]["use_slot_onehot"],
        ).to(device)
    else:
        tag = f"slot_{SLOT_NAMES[args.slot].lower()}"
        ckpt_path = resolve_path(args.checkpoint or cfg["eval"]["checkpoint"])
        stats_path = resolve_path(cfg["data"]["out_dir"]) / "norm_stats.pt"
        model = RotExtrinsicNet(
            feat_dim=cfg["data"]["feat_dim"],
            n_slots=cfg["data"]["n_slots"],
            n_hidden=cfg["model"]["n_hidden"],
            n_lstm_layers=cfg["model"]["n_lstm_layers"],
            bidirectional=cfg["model"]["bidirectional"],
            dropout=0.0,
            use_slot_onehot=cfg["model"]["use_slot_onehot"],
        ).to(device)

    for p in (ckpt_path, stats_path):
        if not p.exists():
            raise FileNotFoundError(p)
    stats = torch.load(stats_path, map_location="cpu")
    mean = stats["mean"].float().view(1, -1)
    std = stats["std"].float().view(1, -1)
    payload = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(payload["model"])
    model.eval()

    fig_dir = resolve_path("experiments/week2_rot_ext/outputs/figures")
    if args.all_sequences:
        fig_dir = fig_dir / f"timelines_all_imuposer_{tag}"
    fig_dir.mkdir(parents=True, exist_ok=True)

    # Score sequences first (for auto pick)
    scored: List[Tuple[int, float, int]] = []
    cache: Dict[int, dict] = {}

    seq_iter = range(n_seq)
    for seq_i in tqdm(seq_iter, desc="score/predict"):
        acc_all = accs[seq_i].float()
        ori_all = oris[seq_i].float()
        if acc_all.shape[0] < window_len:
            continue

        gen = torch.Generator().manual_seed(offset_seed + seq_i * 17)

        if args.dual:
            if acc_all.shape[1] <= max(w_idx, p_idx):
                continue
            aw_c, ow_c = acc_all[:, w_idx], ori_all[:, w_idx]
            ap_c, op_c = acc_all[:, p_idx], ori_all[:, p_idx]
            r_w = sample_random_offsets(1, offset_range, generator=gen)[0]
            r_p = sample_random_offsets(1, offset_range, generator=gen)[0]
            aw_o, ow_o = apply_mount_offset(aw_c, ow_c, r_w)
            ap_o, op_o = apply_mount_offset(ap_c, op_c, r_p)
            starts, hats_w, hats_p = predict_rsb_windows_dual(
                model,
                aw_o,
                ow_o,
                ap_o,
                op_o,
                w_idx,
                p_idx,
                window_len,
                stride,
                acc_scale,
                mean,
                std,
                device,
            )
            if not starts:
                continue
            T = aw_c.shape[0]
            r_hat_w_f = expand_rot_to_frames(starts, hats_w, T, stride, window_len)
            r_hat_p_f = expand_rot_to_frames(starts, hats_p, T, stride, window_len)
            ew = frame_ori_errors(ow_o, ow_c, r_hat_w_f, r_w)
            ep = frame_ori_errors(op_o, op_c, r_hat_p_f, r_p)
            mean_l = float(0.5 * (ew[1].mean() + ep[1].mean()))
            cache[seq_i] = {
                "frames": np.arange(T),
                "watch": ew,
                "phone": ep,
                "mean_learned": mean_l,
                "rsb_err_w": mean_rsb_err_deg(hats_w, r_w),
                "rsb_err_p": mean_rsb_err_deg(hats_p, r_p),
                "mode": "dual",
            }
        elif args.pair:
            # Single-device weights ×2: independent inject + predict per device
            if acc_all.shape[1] <= max(w_idx, p_idx):
                continue
            aw_c, ow_c = acc_all[:, w_idx], ori_all[:, w_idx]
            ap_c, op_c = acc_all[:, p_idx], ori_all[:, p_idx]
            r_w = sample_random_offsets(1, offset_range, generator=gen)[0]
            r_p = sample_random_offsets(1, offset_range, generator=gen)[0]
            aw_o, ow_o = apply_mount_offset(aw_c, ow_c, r_w)
            ap_o, op_o = apply_mount_offset(ap_c, op_c, r_p)
            starts_w, hats_w = predict_rsb_windows_single(
                model,
                aw_o,
                ow_o,
                w_idx,
                window_len,
                stride,
                acc_scale,
                mean,
                std,
                device,
            )
            starts_p, hats_p = predict_rsb_windows_single(
                model,
                ap_o,
                op_o,
                p_idx,
                window_len,
                stride,
                acc_scale,
                mean,
                std,
                device,
            )
            if not starts_w or not starts_p:
                continue
            T = aw_c.shape[0]
            r_hat_w_f = expand_rot_to_frames(starts_w, hats_w, T, stride, window_len)
            r_hat_p_f = expand_rot_to_frames(starts_p, hats_p, T, stride, window_len)
            ew = frame_ori_errors(ow_o, ow_c, r_hat_w_f, r_w)
            ep = frame_ori_errors(op_o, op_c, r_hat_p_f, r_p)
            mean_l = float(0.5 * (ew[1].mean() + ep[1].mean()))
            cache[seq_i] = {
                "frames": np.arange(T),
                "watch": ew,
                "phone": ep,
                "mean_learned": mean_l,
                "rsb_err_w": mean_rsb_err_deg(hats_w, r_w),
                "rsb_err_p": mean_rsb_err_deg(hats_p, r_p),
                "mode": "pair",
            }
        else:
            slot = args.slot
            if acc_all.shape[1] <= slot:
                continue
            acc_c, ori_c = acc_all[:, slot], ori_all[:, slot]
            r_sb = sample_random_offsets(1, offset_range, generator=gen)[0]
            acc_o, ori_o = apply_mount_offset(acc_c, ori_c, r_sb)
            starts, hats = predict_rsb_windows_single(
                model,
                acc_o,
                ori_o,
                slot,
                window_len,
                stride,
                acc_scale,
                mean,
                std,
                device,
            )
            if not starts:
                continue
            T = acc_c.shape[0]
            r_hat_f = expand_rot_to_frames(starts, hats, T, stride, window_len)
            e_none, e_learned, e_gt = frame_ori_errors(ori_o, ori_c, r_hat_f, r_sb)
            mean_l = float(e_learned.mean())
            cache[seq_i] = {
                "frames": np.arange(T),
                "none": e_none,
                "learned": e_learned,
                "gt": e_gt,
                "mean_learned": mean_l,
                "mean_none": float(e_none.mean()),
                "mean_gt": float(e_gt.mean()),
                "rsb_err": mean_rsb_err_deg(hats, r_sb),
                "mode": "single",
            }
        scored.append((seq_i, mean_l, int(acc_all.shape[0])))

    if not scored:
        raise RuntimeError("No sequences long enough to visualize.")

    # Select sequences
    if args.seq_ids:
        selected = [int(x) for x in args.seq_ids.split(",") if x.strip() != ""]
    elif args.all_sequences:
        selected = sorted(cache.keys())
    else:
        scored_sorted = sorted(scored, key=lambda x: (x[1], -x[2]))
        bad = scored_sorted[0][0]
        good = scored_sorted[-1][0]
        selected = [good, bad] if good != bad else [good]
        # fill mids if n_pick > 2
        for sid, _, _ in scored_sorted:
            if len(selected) >= args.n_pick:
                break
            if sid not in selected:
                selected.append(sid)

    means = [cache[s]["mean_learned"] for s in cache]
    good_th = float(np.percentile(means, 25))
    bad_th = float(np.percentile(means, 75))

    index = []
    for sid in selected:
        if sid not in cache:
            print(f"[skip] seq {sid} not in scored cache (check --max-seqs)")
            continue
        pack = cache[sid]
        q = quality_tag(pack["mean_learned"], good_th, bad_th)
        if sid < len(motions):
            meta = f"{motions[sid]['pid']}/{motions[sid]['motion']}"
        else:
            meta = f"seq{sid}"

        if args.dual or args.pair:
            mode_label = "dual" if args.dual else "single×2"
            title = (
                f"Week2 D5-B  {tag}  ({mode_label})  {meta}\n"
                f"seq={sid}  mean(Learned)={pack['mean_learned']:.2f}deg  "
                f"R_SB err W/P~{pack['rsb_err_w']:.1f}/{pack['rsb_err_p']:.1f}deg"
            )
            out = fig_dir / f"timeline_seq{sid:03d}_{q}_imuposer_{tag}.png"
            plot_dual_error_timeline(
                pack["frames"], pack["watch"], pack["phone"], title, out
            )
        else:
            title = (
                f"Week2 D5-B  {SLOT_NAMES[args.slot]}  {meta}\n"
                f"seq={sid}  none={pack['mean_none']:.2f}deg  "
                f"learned={pack['mean_learned']:.2f}deg  gt={pack['mean_gt']:.2f}deg  "
                f"R_SB err~{pack['rsb_err']:.1f}deg"
            )
            out = fig_dir / f"timeline_seq{sid:03d}_{q}_imuposer_{tag}.png"
            plot_error_timeline(
                pack["frames"],
                pack["none"],
                pack["learned"],
                pack["gt"],
                title,
                out,
            )
        print(f"saved {out}  mean_learned={pack['mean_learned']:.3f}°")
        index.append(
            {
                "seq_id": sid,
                "quality": q,
                "meta": meta,
                "mean_learned_deg": pack["mean_learned"],
                "figure": out.name,
            }
        )

    idx_path = fig_dir / f"timeline_index_imuposer_{tag}.json"
    with open(idx_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "tag": tag,
                "dual": bool(args.dual),
                "pair_single_x2": bool(args.pair),
                "n_scored": len(cache),
                "good_th_deg": good_th,
                "bad_th_deg": bad_th,
                "offset_seed": offset_seed,
                "items": index,
            },
            f,
            indent=2,
            ensure_ascii=False,
        )
    print(f"saved index {idx_path}")


if __name__ == "__main__":
    main()
