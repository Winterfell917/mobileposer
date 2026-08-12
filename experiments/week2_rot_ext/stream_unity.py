#!/usr/bin/env python3
"""
Stream IMUPoser GT pose/tran to Unity MotionViewer, with Week2 error HUD.

No position markers. Same inject protocol / seq-id / fps as visualize.py timelines,
so you can scrub the timeline figure while watching the motion.

HUD (MotionViewer has no text API):
  1) Terminal live line: frame + None/Learned/GT (deg)  ← always on
  2) Optional OpenCV window (--hud-window; needs local X11 DISPLAY)
  3) Optional 3D error bars beside the character (--error-bars)

Unity Online (same as Week1):
  Server Ip = 127.0.0.1 , Port = 8989 , Connect On Load = True
  Start this script first → Unity Play → Enter (unless --auto-start)

Usage (repo root):
  # single sequence
  python experiments/week2_rot_ext/stream_unity.py --slot 3 --seq-id 0 --once --error-bars

  # playlist: play all (or a range) after ONE Unity connection
  python experiments/week2_rot_ext/stream_unity.py --slot 3 --all-seqs --auto-start --error-bars
  python experiments/week2_rot_ext/stream_unity.py --slot 3 --all-seqs --seq-start 0 --seq-end 20 --gap 0.5
  python experiments/week2_rot_ext/stream_unity.py --dual --combo lw_rp --all-seqs --loop-playlist
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import torch

try:
    import cv2
except Exception:  # pragma: no cover
    cv2 = None

_EXP_DIR = Path(__file__).resolve().parent
_REPO = _EXP_DIR.parents[1]
_MOBILE = _REPO / "mobileposer"
if str(_EXP_DIR) not in sys.path:
    sys.path.insert(0, str(_EXP_DIR))
if str(_MOBILE) not in sys.path:
    sys.path.append(str(_MOBILE))

from dataset import (  # noqa: E402
    SLOT_NAMES,
    apply_mount_offset,
    combo_to_indices,
    load_config,
    resolve_path,
    sample_random_offsets,
    set_seed,
)
from models import RotExtrinsicDualNet, RotExtrinsicNet  # noqa: E402
from visualize import (  # noqa: E402
    expand_rot_to_frames,
    frame_ori_errors,
    list_imuposer_motions,
    parse_combo,
    predict_rsb_windows_dual,
    predict_rsb_windows_single,
)

from articulate.utils.unity import MotionViewer  # noqa: E402

COLOR_NONE = (0.5, 0.5, 0.5)
COLOR_LEARNED = (0.12, 0.47, 0.71)
COLOR_GT = (0.17, 0.63, 0.17)


def _normalize_feat(x: torch.Tensor, mean: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
    return (x - mean) / std


@torch.no_grad()
def prepare_single_errors(
    model,
    acc_all: torch.Tensor,
    ori_all: torch.Tensor,
    slot: int,
    window_len: int,
    stride: int,
    acc_scale: float,
    mean: torch.Tensor,
    std: torch.Tensor,
    offset_range: float,
    gen: torch.Generator,
    device: torch.device,
) -> Dict[str, np.ndarray]:
    acc_c, ori_c = acc_all[:, slot].float(), ori_all[:, slot].float()
    r_sb = sample_random_offsets(1, offset_range, generator=gen)[0]
    acc_o, ori_o = apply_mount_offset(acc_c, ori_c, r_sb)
    starts, hats = predict_rsb_windows_single(
        model, acc_o, ori_o, slot, window_len, stride, acc_scale, mean, std, device
    )
    T = acc_c.shape[0]
    if not starts:
        z = np.zeros(T, dtype=np.float32)
        return {"none": z, "learned": z, "gt": z, "rsb_err": float("nan")}
    r_hat_f = expand_rot_to_frames(starts, hats, T, stride, window_len)
    e0, e1, e2 = frame_ori_errors(ori_o, ori_c, r_hat_f, r_sb)
    from visualize import mean_rsb_err_deg

    return {
        "none": e0.astype(np.float32),
        "learned": e1.astype(np.float32),
        "gt": e2.astype(np.float32),
        "rsb_err": mean_rsb_err_deg(hats, r_sb),
    }


@torch.no_grad()
def prepare_dual_errors(
    model,
    acc_all: torch.Tensor,
    ori_all: torch.Tensor,
    w_idx: int,
    p_idx: int,
    window_len: int,
    stride: int,
    acc_scale: float,
    mean: torch.Tensor,
    std: torch.Tensor,
    offset_range: float,
    gen: torch.Generator,
    device: torch.device,
) -> Dict[str, np.ndarray]:
    aw_c, ow_c = acc_all[:, w_idx].float(), ori_all[:, w_idx].float()
    ap_c, op_c = acc_all[:, p_idx].float(), ori_all[:, p_idx].float()
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
    T = aw_c.shape[0]
    if not starts:
        z = np.zeros(T, dtype=np.float32)
        return {
            "none": z,
            "learned": z,
            "gt": z,
            "none_w": z,
            "learned_w": z,
            "gt_w": z,
            "none_p": z,
            "learned_p": z,
            "gt_p": z,
            "rsb_err_w": float("nan"),
            "rsb_err_p": float("nan"),
        }
    r_hat_w_f = expand_rot_to_frames(starts, hats_w, T, stride, window_len)
    r_hat_p_f = expand_rot_to_frames(starts, hats_p, T, stride, window_len)
    ew = frame_ori_errors(ow_o, ow_c, r_hat_w_f, r_w)
    ep = frame_ori_errors(op_o, op_c, r_hat_p_f, r_p)
    from visualize import mean_rsb_err_deg

    return {
        "none": (0.5 * (ew[0] + ep[0])).astype(np.float32),
        "learned": (0.5 * (ew[1] + ep[1])).astype(np.float32),
        "gt": (0.5 * (ew[2] + ep[2])).astype(np.float32),
        "none_w": ew[0].astype(np.float32),
        "learned_w": ew[1].astype(np.float32),
        "gt_w": ew[2].astype(np.float32),
        "none_p": ep[0].astype(np.float32),
        "learned_p": ep[1].astype(np.float32),
        "gt_p": ep[2].astype(np.float32),
        "rsb_err_w": mean_rsb_err_deg(hats_w, r_w),
        "rsb_err_p": mean_rsb_err_deg(hats_p, r_p),
    }


def _draw_error_bars(
    viewer: MotionViewer,
    root: np.ndarray,
    none_deg: float,
    learned_deg: float,
    gt_deg: float,
    scale: float = 0.02,
):
    """Vertical bars next to character: height ∝ error degrees."""
    base = np.array(root, dtype=np.float64) + np.array([0.55, 0.05, 0.0])
    spacing = 0.12
    for i, (val, col) in enumerate(
        (
            (none_deg, COLOR_NONE),
            (learned_deg, COLOR_LEARNED),
            (gt_deg, COLOR_GT),
        )
    ):
        p0 = base + np.array([i * spacing, 0.0, 0.0])
        p1 = p0 + np.array([0.0, max(0.02, float(val) * scale), 0.0])
        viewer.draw_line(p0, p1, color=col, width=0.03, render=False)


def _has_display() -> bool:
    """OpenCV/Qt GUI needs a working DISPLAY (fails on many SSH / headless setups)."""
    if sys.platform.startswith("linux") and not os.environ.get("DISPLAY"):
        return False
    return cv2 is not None


def _hud_image(
    title: str,
    fidx: int,
    t_len: int,
    none_deg: float,
    learned_deg: float,
    gt_deg: float,
    extra: str = "",
) -> np.ndarray:
    img = np.ones((220, 520, 3), dtype=np.uint8) * 245
    lines = [
        title[:60],
        f"Frame  {fidx:4d} / {t_len - 1}",
        f"None     {none_deg:6.2f} deg",
        f"Learned  {learned_deg:6.2f} deg",
        f"GT       {gt_deg:6.2f} deg",
    ]
    if extra:
        lines.append(extra[:70])
    y = 28
    for i, line in enumerate(lines):
        color = (20, 20, 20)
        if i == 2:
            color = (80, 80, 80)
        elif i == 3:
            color = (180, 100, 40)
        elif i == 4:
            color = (40, 140, 40)
        cv2.putText(img, line, (16, y), cv2.FONT_HERSHEY_SIMPLEX, 0.65, color, 2, cv2.LINE_AA)
        y += 34
    return img


def _stream(
    viewer: MotionViewer,
    pose: torch.Tensor,
    tran: torch.Tensor,
    errs: Dict[str, np.ndarray],
    fps: float,
    t0: int,
    t1: int,
    label: str,
    dual: bool,
    hud_window: bool,
    error_bars: bool,
):
    win_name = "Week2 Error HUD"
    hud_ok = False
    if hud_window:
        try:
            cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)
            cv2.resizeWindow(win_name, 520, 220)
            hud_ok = True
        except Exception as e:
            print(f"[warn] HUD window unavailable ({e}); terminal HUD only")
            hud_ok = False

    for i, fidx in enumerate(range(t0, t1)):
        t_wall = time.time()
        viewer.clear_all(render=False)
        viewer.update(pose[fidx], tran[fidx], index=0, render=False)

        none_d = float(errs["none"][fidx])
        learn_d = float(errs["learned"][fidx])
        gt_d = float(errs["gt"][fidx])

        if error_bars:
            root = tran[fidx].detach().cpu().numpy().reshape(3)
            _draw_error_bars(viewer, root, none_d, learn_d, gt_d)

        viewer.render()

        if dual:
            extra = (
                f"W L={errs['learned_w'][fidx]:.1f}  "
                f"P L={errs['learned_p'][fidx]:.1f}"
            )
        else:
            extra = f"R_SB err~{errs.get('rsb_err', float('nan')):.1f}deg"

        hud_line = (
            f"\r{label} | frame={fidx:4d}/{t1 - 1} "
            f"None={none_d:5.1f} Learned={learn_d:5.1f} GT={gt_d:5.1f} {extra}   "
        )
        print(hud_line, end="", flush=True)

        if hud_ok:
            try:
                img = _hud_image(label, fidx, pose.shape[0], none_d, learn_d, gt_d, extra)
                cv2.imshow(win_name, img)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    print("\nHUD window quit")
                    break
            except Exception as e:
                print(f"\n[warn] HUD window failed ({e}); continuing without it")
                hud_ok = False

        time.sleep(max(0.0, (1.0 / fps) - (time.time() - t_wall)))

    print()
    if hud_ok:
        try:
            cv2.destroyWindow(win_name)
        except Exception:
            pass


def main():
    parser = argparse.ArgumentParser(description="Week2 Unity GT stream + error HUD")
    parser.add_argument(
        "--config",
        type=str,
        default="experiments/week2_rot_ext/configs/default.yaml",
    )
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--slot", type=int, default=3, choices=[0, 1, 2, 3])
    parser.add_argument("--dual", action="store_true")
    parser.add_argument("--combo", type=str, default="lw_rp")
    parser.add_argument(
        "--seq-id",
        type=int,
        default=None,
        help="Single sequence id (omit with --all-seqs)",
    )
    parser.add_argument(
        "--all-seqs",
        action="store_true",
        help="After one Unity connect, play sequences from --seq-start to --seq-end",
    )
    parser.add_argument("--seq-start", type=int, default=0, help="First seq id for --all-seqs")
    parser.add_argument(
        "--seq-end",
        type=int,
        default=None,
        help="End seq id exclusive for --all-seqs (default: all)",
    )
    parser.add_argument(
        "--gap",
        type=float,
        default=0.5,
        help="Seconds to pause between sequences in --all-seqs mode",
    )
    parser.add_argument(
        "--loop-playlist",
        action="store_true",
        help="With --all-seqs: after last seq, restart from first until Ctrl+C",
    )
    parser.add_argument("--port", type=int, default=8989)
    parser.add_argument("--ip", type=str, default="127.0.0.1")
    parser.add_argument("--fps", type=float, default=None)
    parser.add_argument("--start", type=int, default=0, help="Start frame within each sequence")
    parser.add_argument("--end", type=int, default=None, help="End frame exclusive within each sequence")
    parser.add_argument(
        "--once",
        action="store_true",
        help="Single --seq-id: play once then exit. --all-seqs already plays each once.",
    )
    parser.add_argument("--auto-start", action="store_true")
    parser.add_argument("--no-tran", action="store_true")
    parser.add_argument("--offset-seed", type=int, default=None)
    parser.add_argument(
        "--hud-window",
        action="store_true",
        help="Force OpenCV HUD window (needs local X11 DISPLAY)",
    )
    parser.add_argument(
        "--no-hud-window",
        action="store_true",
        help="Disable OpenCV HUD window (terminal line only)",
    )
    parser.add_argument(
        "--error-bars",
        action="store_true",
        help="Draw None/Learned/GT vertical bars beside character in Unity",
    )
    args = parser.parse_args()

    if not args.all_seqs and args.seq_id is None:
        parser.error("provide --seq-id or --all-seqs")
    if args.all_seqs and args.seq_id is not None:
        print("[warn] --all-seqs set; ignoring --seq-id")

    cfg = load_config(resolve_path(args.config))
    set_seed(cfg["experiment"]["seed"])
    fps = float(args.fps if args.fps is not None else cfg["data"]["fps"])
    window_len = int(cfg["data"]["window_len"])
    stride = int(cfg["data"]["test_stride"])
    acc_scale = float(cfg["data"]["acc_scale"])
    offset_range = float(cfg["data"]["offset_range_deg"])
    offset_seed = (
        args.offset_seed
        if args.offset_seed is not None
        else int(cfg["experiment"]["seed"]) + (9 if args.dual else 7)
    )

    src = resolve_path(cfg["data"]["processed_imuposer_file"])
    if not src.exists():
        raise FileNotFoundError(src)
    data = torch.load(src, map_location="cpu")
    n_total = len(data["pose"])

    if args.all_seqs:
        s0 = max(0, int(args.seq_start))
        s1 = int(args.seq_end) if args.seq_end is not None else n_total
        s1 = min(s1, n_total)
        seq_ids = list(range(s0, s1))
        if not seq_ids:
            raise ValueError(f"empty seq range [{s0}, {s1})")
    else:
        sid0 = int(args.seq_id)
        if sid0 < 0 or sid0 >= n_total:
            raise IndexError(f"seq-id {sid0} out of range [0, {n_total})")
        seq_ids = [sid0]

    motions = []
    raw_dir = resolve_path("data/raw/IMUPoser")
    if raw_dir.exists():
        motions = list_imuposer_motions(raw_dir)

    device = torch.device(
        cfg["train"]["device"] if torch.cuda.is_available() else "cpu"
    )

    w_idx = p_idx = None
    if args.dual:
        watch_side, phone_side = parse_combo(args.combo)
        w_idx, p_idx = combo_to_indices(watch_side, phone_side)
        tag = f"dual_{args.combo}"
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

    def _prep_seq(sid: int):
        motion = (
            f"{motions[sid]['pid']}/{motions[sid]['motion']}"
            if sid < len(motions)
            else f"seq{sid}"
        )
        print(f"Preparing Week2 errors for seq={sid}/{seq_ids[-1]} ({motion}) [{tag}] ...")
        gen = torch.Generator().manual_seed(offset_seed + sid * 17)
        acc_all, ori_all = data["acc"][sid], data["ori"][sid]
        if args.dual:
            errs = prepare_dual_errors(
                model,
                acc_all,
                ori_all,
                w_idx,
                p_idx,
                window_len,
                stride,
                acc_scale,
                mean,
                std,
                offset_range,
                gen,
                device,
            )
            print(
                f"  mean Learned={float(errs['learned'].mean()):.2f}deg  "
                f"R_SB err W/P~{errs['rsb_err_w']:.1f}/{errs['rsb_err_p']:.1f}deg"
            )
        else:
            errs = prepare_single_errors(
                model,
                acc_all,
                ori_all,
                args.slot,
                window_len,
                stride,
                acc_scale,
                mean,
                std,
                offset_range,
                gen,
                device,
            )
            print(
                f"  mean None/Learned/GT="
                f"{errs['none'].mean():.2f}/{errs['learned'].mean():.2f}/{errs['gt'].mean():.2f}deg  "
                f"R_SB err~{errs['rsb_err']:.1f}deg"
            )
        pose = data["pose"][sid].view(-1, 24, 3, 3).float()
        tran = data["tran"][sid].view(-1, 3).float()
        if args.no_tran:
            tran = torch.zeros_like(tran)
        t0 = max(0, int(args.start))
        t1 = int(args.end) if args.end is not None else pose.shape[0]
        t1 = min(t1, pose.shape[0])
        label = f"seq={sid}/{seq_ids[-1]} {motion} [{tag}]"
        return pose, tran, errs, label, t0, t1

    MotionViewer.ip = args.ip
    MotionViewer.port = int(args.port)

    hud_window = bool(args.hud_window) and not bool(args.no_hud_window)
    if hud_window and not _has_display():
        print("[warn] --hud-window requested but no DISPLAY; falling back to terminal HUD")
        hud_window = False

    print("=" * 60)
    print("Week2 Unity Online streaming (GT pose + error HUD)")
    print(f"  bind     {args.ip}:{args.port}")
    print(f"  tag      {tag}")
    if args.all_seqs:
        print(f"  playlist seq [{seq_ids[0]}, {seq_ids[-1]}]  ({len(seq_ids)} sequences)")
        print(f"  mode     each once, then {'loop playlist' if args.loop_playlist else 'exit'}")
        print(f"  gap      {args.gap}s between sequences")
    else:
        print(f"  seq      {seq_ids[0]}")
        print(f"  mode     {'once' if args.once else 'loop single seq'}")
    print(f"  fps={fps}  offset_seed={offset_seed}")
    print(f"  HUD window={hud_window}  error_bars={args.error_bars}")
    print("  Tip: open matching timeline PNG beside Unity; same seq-id/fps.")
    print("=" * 60)

    viewer_name = f"w2_{tag}_playlist" if args.all_seqs else f"w2_{tag}_seq{seq_ids[0]}"
    with MotionViewer(1, overlap=False, names=[viewer_name], fps=fps) as viewer:
        print(f"Unity connected from {viewer.conn.getpeername()}")
        if not args.auto_start:
            try:
                input(">>> Unity Game view ready, then Enter to START...")
            except EOFError:
                print("no stdin; starting in 3s...")
                time.sleep(3.0)

        print("Streaming... (Ctrl+C stop; HUD window press q)")
        try:
            while True:
                for sid in seq_ids:
                    pose, tran, errs, label, t0, t1 = _prep_seq(sid)
                    if t1 <= t0:
                        print(f"[warn] skip seq {sid}: empty frame range")
                        continue
                    print(f">>> playing {label}  frames=[{t0},{t1})")
                    _stream(
                        viewer,
                        pose,
                        tran,
                        errs,
                        fps,
                        t0,
                        t1,
                        label,
                        args.dual,
                        hud_window,
                        args.error_bars,
                    )
                    if args.all_seqs and args.gap > 0 and sid != seq_ids[-1]:
                        time.sleep(args.gap)

                if args.all_seqs:
                    if args.loop_playlist:
                        print("playlist done → restart from first seq")
                        continue
                    print("playlist finished (all sequences once).")
                    break
                if args.once:
                    print("single pass done.")
                    break
                print("seq done → replay (Ctrl+C to stop)")
        except KeyboardInterrupt:
            print("\nstopped by user")
        except (BrokenPipeError, ConnectionResetError, OSError) as e:
            print(f"\nUnity disconnected: {e}")

    if hud_window and cv2 is not None:
        try:
            cv2.destroyAllWindows()
        except Exception:
            pass
    print("done.")


if __name__ == "__main__":
    main()
