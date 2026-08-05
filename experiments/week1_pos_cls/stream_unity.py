#!/usr/bin/env python3
"""
Stream IMUPoser GT pose (+ optional watch/phone markers) to Unity MotionViewer (Online).

Matches your Unity Client settings:
  Server Ip = 127.0.0.1
  Port      = 8989
  Connect On Load = True

Startup order (important):
  1) Run this script first — it binds and waits for Unity.
  2) In Unity: enable Hierarchy → Online, disable Offline if needed.
  3) Press Play. Unity Client connects to 127.0.0.1:8989.

Usage (repo root):
  # single sequence
  python experiments/week1_pos_cls/stream_unity.py --combo lw_rp --seq-id 110

  # play ALL sequences after ONE Unity connection (seq 0 → last)
  python experiments/week1_pos_cls/stream_unity.py --combo lw_rp --all-seqs --auto-start

  # subsequence range
  python experiments/week1_pos_cls/stream_unity.py --combo lw_rp --all-seqs --seq-start 0 --seq-end 20
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader

_EXP_DIR = Path(__file__).resolve().parent
_REPO = _EXP_DIR.parents[1]
_MOBILE = _REPO / "mobileposer"
if str(_EXP_DIR) not in sys.path:
    sys.path.insert(0, str(_EXP_DIR))
if str(_MOBILE) not in sys.path:
    sys.path.append(str(_MOBILE))

from dataset import load_config, resolve_path, set_seed  # noqa: E402
from dataset.pos_dataset import PosClsDataset, load_norm_stats  # noqa: E402
from models import PosClassifier  # noqa: E402
from visualize import (  # noqa: E402
    JOINT_LP,
    JOINT_RP,
    expand_to_frames,
    group_by_sequence,
    list_imuposer_motions,
    parse_combo,
    predict_all,
    resolve_test_pt,
)

from articulate.model import ParametricModel  # noqa: E402
from articulate.utils.unity import MotionViewer  # noqa: E402
from config import paths  # noqa: E402


def _watch_hand_joint(side: int) -> int:
    return 22 if int(side) == 0 else 23


def _watch_wrist_joint(side: int) -> int:
    return 20 if int(side) == 0 else 21


def _phone_joint(side: int) -> int:
    return JOINT_LP if int(side) == 0 else JOINT_RP


def _unit(v: np.ndarray) -> np.ndarray:
    n = float(np.linalg.norm(v))
    return v / n if n > 1e-8 else np.array([0.0, 1.0, 0.0], dtype=np.float64)


def _marker_world_pos(joints: np.ndarray, kind: str, side: int, offset: float) -> np.ndarray:
    if kind == "watch":
        j = joints[_watch_hand_joint(side)]
        parent = joints[_watch_wrist_joint(side)]
        return j + _unit(j - parent) * offset
    j = joints[_phone_joint(side)]
    lateral = j - joints[0]
    lateral[1] = 0.0
    return j + _unit(lateral) * offset


def _marker_color_ok(ok: bool):
    return (0.1, 0.95, 0.2) if ok else (1.0, 0.12, 0.12)


def _draw_device_markers(
    viewer: MotionViewer,
    joints: np.ndarray,
    yw: int,
    yp: int,
    pw: int,
    pp: int,
    offset: float,
    radius: float,
):
    w_ok = int(pw) == int(yw)
    p_ok = int(pp) == int(yp)
    viewer.draw_point(
        _marker_world_pos(joints, "watch", pw, offset),
        color=_marker_color_ok(w_ok),
        radius=radius,
        render=False,
    )
    viewer.draw_point(
        _marker_world_pos(joints, "phone", pp, offset),
        color=_marker_color_ok(p_ok),
        radius=radius,
        render=False,
    )
    if not w_ok:
        viewer.draw_point(
            _marker_world_pos(joints, "watch", yw, offset * 0.85),
            color=(0.25, 0.45, 1.0),
            radius=radius * 0.65,
            render=False,
        )
    if not p_ok:
        viewer.draw_point(
            _marker_world_pos(joints, "phone", yp, offset * 0.85),
            color=(0.2, 0.95, 1.0),
            radius=radius * 0.65,
            render=False,
        )


def _load_all_pred_groups(cfg, combo: str, device: torch.device) -> Dict[int, dict]:
    watch_side, phone_side = parse_combo(combo)
    data_dir = resolve_path(cfg["data"]["out_dir"])
    pt = resolve_test_pt(data_dir, watch_side, phone_side)
    stats = load_norm_stats(data_dir / "norm_stats.pt")
    ds = PosClsDataset(pt, norm_stats=stats, normalize=True)
    loader = DataLoader(ds, batch_size=cfg["eval"]["batch_size"], shuffle=False, num_workers=0)

    ckpt_path = resolve_path(cfg["eval"]["checkpoint"])
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
    return group_by_sequence(ds.meta, yw, yp, pw, pp)


def _frame_labels(pack: Optional[dict], seq_len: int, stride: int, window_len: int):
    if pack is None:
        return None, None, None, None
    yw_f = expand_to_frames(pack["starts"], pack["yw"], seq_len, stride, window_len)
    yp_f = expand_to_frames(pack["starts"], pack["yp"], seq_len, stride, window_len)
    pw_f = expand_to_frames(pack["starts"], pack["pw"], seq_len, stride, window_len)
    pp_f = expand_to_frames(pack["starts"], pack["pp"], seq_len, stride, window_len)
    return yw_f, yp_f, pw_f, pp_f


def _stream_one_sequence(
    viewer: MotionViewer,
    pose: torch.Tensor,
    tran: torch.Tensor,
    fps: float,
    t0: int,
    t1: int,
    use_markers: bool,
    body_model,
    yw_f,
    yp_f,
    pw_f,
    pp_f,
    marker_offset: float,
    marker_radius: float,
    label: str,
):
    n_frames = t1 - t0
    for i, fidx in enumerate(range(t0, t1)):
        t_wall = time.time()
        viewer.clear_all(render=False)
        viewer.update(pose[fidx], tran[fidx], index=0, render=False)

        if use_markers and body_model is not None and yw_f is not None:
            yw = int(yw_f[fidx]) if yw_f[fidx] >= 0 else 0
            yp = int(yp_f[fidx]) if yp_f[fidx] >= 0 else 0
            pw = int(pw_f[fidx]) if pw_f[fidx] >= 0 else yw
            pp = int(pp_f[fidx]) if pp_f[fidx] >= 0 else yp
            fk_out = body_model.forward_kinematics(
                pose[fidx : fidx + 1],
                tran=tran[fidx : fidx + 1],
                calc_mesh=False,
            )
            joints = fk_out[1][0].detach().cpu().numpy()
            _draw_device_markers(
                viewer,
                joints,
                yw,
                yp,
                pw,
                pp,
                offset=marker_offset,
                radius=marker_radius,
            )

        viewer.render()
        if i == 0 or (i + 1) % 60 == 0 or i + 1 == n_frames:
            print(f"  {label}  frame {fidx} ({i + 1}/{n_frames})")
        time.sleep(max(0.0, (1.0 / fps) - (time.time() - t_wall)))


def main():
    parser = argparse.ArgumentParser(description="Stream pose to Unity MotionViewer (Online)")
    parser.add_argument("--config", type=str, default="experiments/week1_pos_cls/configs/default.yaml")
    parser.add_argument("--combo", type=str, default="lw_rp")
    parser.add_argument("--seq-id", type=int, default=None, help="Single sequence id (omit with --all-seqs)")
    parser.add_argument(
        "--all-seqs",
        action="store_true",
        help="After one Unity connect, play sequences from --seq-start to --seq-end (default: all)",
    )
    parser.add_argument("--seq-start", type=int, default=0, help="First seq id when using --all-seqs")
    parser.add_argument("--seq-end", type=int, default=None, help="End seq id exclusive (default: all)")
    parser.add_argument("--port", type=int, default=8989, help="Must match Unity Client Port")
    parser.add_argument("--ip", type=str, default="127.0.0.1")
    parser.add_argument("--fps", type=float, default=None, help="Playback fps (default: data fps)")
    parser.add_argument("--start", type=int, default=0, help="Start frame within each sequence")
    parser.add_argument("--end", type=int, default=None, help="End frame exclusive within each sequence")
    parser.add_argument(
        "--once",
        action="store_true",
        help="For single --seq-id: play once then exit. For --all-seqs: implied (each seq once).",
    )
    parser.add_argument(
        "--loop-playlist",
        action="store_true",
        help="With --all-seqs: after last seq, restart from first until Ctrl+C",
    )
    parser.add_argument(
        "--gap",
        type=float,
        default=0.5,
        help="Seconds to pause between sequences in --all-seqs mode",
    )
    parser.add_argument("--auto-start", action="store_true", help="Skip Enter prompt after connect")
    parser.add_argument("--no-markers", action="store_true")
    parser.add_argument("--marker-offset", type=float, default=0.12)
    parser.add_argument("--marker-radius", type=float, default=0.08)
    parser.add_argument("--no-tran", action="store_true", help="Zero root translation (in-place)")
    args = parser.parse_args()

    if not args.all_seqs and args.seq_id is None:
        parser.error("provide --seq-id or --all-seqs")
    if args.all_seqs and args.seq_id is not None:
        print("[warn] --all-seqs set; ignoring --seq-id")

    use_markers = not args.no_markers
    cfg = load_config(resolve_path(args.config))
    set_seed(cfg["experiment"]["seed"])
    fps = float(args.fps if args.fps is not None else cfg["data"]["fps"])
    stride = int(cfg["data"]["test_stride"])
    window_len = int(cfg["data"]["window_len"])

    pose_src = resolve_path(cfg["data"]["processed_imuposer_file"])
    if not pose_src.exists():
        raise FileNotFoundError(pose_src)
    pose_data = torch.load(pose_src, map_location="cpu")
    n_total = len(pose_data["pose"])

    if args.all_seqs:
        s0 = max(0, int(args.seq_start))
        s1 = int(args.seq_end) if args.seq_end is not None else n_total
        s1 = min(s1, n_total)
        seq_ids = list(range(s0, s1))
        if not seq_ids:
            raise ValueError(f"empty seq range [{s0}, {s1})")
    else:
        if args.seq_id < 0 or args.seq_id >= n_total:
            raise IndexError(f"seq-id {args.seq_id} out of range [0, {n_total})")
        seq_ids = [int(args.seq_id)]

    motion_map = []
    raw_dir = resolve_path("data/raw/IMUPoser")
    if raw_dir.exists():
        motion_map = list_imuposer_motions(raw_dir)

    device = torch.device(cfg["train"]["device"] if torch.cuda.is_available() else "cpu")
    groups = None
    body_model = None
    if use_markers:
        print("Loading position classifier predictions for markers...")
        groups = _load_all_pred_groups(cfg, args.combo, device)
        body_model = ParametricModel(paths.smpl_file)
        print(f"  predictions ready for {len(groups)} sequences")

    MotionViewer.ip = args.ip
    MotionViewer.port = int(args.port)

    print("=" * 60)
    print("Unity Online streaming")
    print(f"  bind       {args.ip}:{args.port}")
    print(f"  combo      {args.combo}")
    if args.all_seqs:
        print(f"  playlist   seq [{seq_ids[0]}, {seq_ids[-1]}]  ({len(seq_ids)} sequences)")
        print(f"  mode       play each once, then {'loop playlist' if args.loop_playlist else 'exit'}")
    else:
        print(f"  seq        {seq_ids[0]}")
        print(f"  mode       {'once' if args.once else 'loop single seq'}")
    print(f"  fps={fps}  markers={use_markers}  gap={args.gap}s")
    print("Startup: run this first → Unity Play (Connect On Load) → Enter (unless --auto-start)")
    print("=" * 60)

    name = f"{args.combo}_playlist" if args.all_seqs else f"seq{seq_ids[0]}_{args.combo}"
    with MotionViewer(1, overlap=False, names=[name], fps=fps) as viewer:
        print(f"Unity connected from {viewer.conn.getpeername()}")
        if not args.auto_start:
            try:
                input(">>> Switch to Unity Game view, then press Enter here to START streaming...")
            except EOFError:
                print("no stdin; starting in 3s...")
                time.sleep(3.0)

        print("Streaming... (Ctrl+C to stop)")
        try:
            while True:
                for sid in seq_ids:
                    pose = pose_data["pose"][sid].view(-1, 24, 3, 3)
                    tran = pose_data["tran"][sid].view(-1, 3)
                    if args.no_tran:
                        tran = torch.zeros_like(tran)
                    t0 = max(0, int(args.start))
                    t1 = int(args.end) if args.end is not None else pose.shape[0]
                    t1 = min(t1, pose.shape[0])
                    if t1 <= t0:
                        print(f"[warn] skip seq {sid}: empty frame range")
                        continue

                    motion = ""
                    if sid < len(motion_map):
                        motion = f"{motion_map[sid]['pid']}/{motion_map[sid]['motion']}"
                    pack = groups.get(sid) if groups else None
                    yw_f, yp_f, pw_f, pp_f = _frame_labels(pack, pose.shape[0], stride, window_len)
                    j_acc = ""
                    if pack is not None:
                        ok = float(((pack["pw"] == pack["yw"]) & (pack["pp"] == pack["yp"])).mean())
                        j_acc = f" joint-acc={ok:.3f}"

                    label = f"seq={sid}/{seq_ids[-1]} {motion}{j_acc}".strip()
                    print(f">>> playing {label}  frames=[{t0},{t1})")
                    _stream_one_sequence(
                        viewer,
                        pose,
                        tran,
                        fps,
                        t0,
                        t1,
                        use_markers,
                        body_model,
                        yw_f,
                        yp_f,
                        pw_f,
                        pp_f,
                        float(args.marker_offset),
                        float(args.marker_radius),
                        label,
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

    print("done.")


if __name__ == "__main__":
    main()
