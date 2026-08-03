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
  # markers ON by default (green=correct, red=wrong)
  python experiments/week1_pos_cls/stream_unity.py --combo lw_rp --seq-id 102
  python experiments/week1_pos_cls/stream_unity.py --combo lw_rp --seq-id 12

  # play once / no markers
  python experiments/week1_pos_cls/stream_unity.py --combo lw_rp --seq-id 102 --once
  python experiments/week1_pos_cls/stream_unity.py --combo lw_rp --seq-id 102 --no-markers
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

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
    parse_combo,
    predict_all,
    resolve_test_pt,
)

from articulate.model import ParametricModel  # noqa: E402
from articulate.utils.unity import MotionViewer  # noqa: E402
from config import paths  # noqa: E402


def _watch_hand_joint(side: int) -> int:
    """SMPL left/right hand (more distal / easier to see than wrist)."""
    return 22 if int(side) == 0 else 23


def _watch_wrist_joint(side: int) -> int:
    return 20 if int(side) == 0 else 21


def _phone_joint(side: int) -> int:
    return JOINT_LP if int(side) == 0 else JOINT_RP


def _unit(v: np.ndarray) -> np.ndarray:
    n = float(np.linalg.norm(v))
    return v / n if n > 1e-8 else np.array([0.0, 1.0, 0.0], dtype=np.float64)


def _marker_world_pos(joints: np.ndarray, kind: str, side: int, offset: float) -> np.ndarray:
    """
    Place marker OUTSIDE the mesh: joint center + outward offset.

    kind='watch': push past hand along wrist→hand
    kind='phone': push sideways from pelvis→hip (horizontal)
    """
    if kind == "watch":
        j = joints[_watch_hand_joint(side)]
        parent = joints[_watch_wrist_joint(side)]
        return j + _unit(j - parent) * offset
    j = joints[_phone_joint(side)]
    lateral = j - joints[0]
    lateral[1] = 0.0
    return j + _unit(lateral) * offset


def _load_preds_for_seq(cfg, combo: str, seq_id: int, device: torch.device):
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
    groups = group_by_sequence(ds.meta, yw, yp, pw, pp)
    if seq_id not in groups:
        raise KeyError(f"seq {seq_id} not found in predictions for combo={combo}")
    return groups[seq_id]


def _marker_color_ok(ok: bool):
    """Green = correct prediction, red = wrong."""
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
    """
    World-space spheres offset outside the body (avoid burying at joint centers).
    Pred: green/red; GT: blue/cyan only when Pred wrong.
    """
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


def main():
    parser = argparse.ArgumentParser(description="Stream pose to Unity MotionViewer (Online)")
    parser.add_argument("--config", type=str, default="experiments/week1_pos_cls/configs/default.yaml")
    parser.add_argument("--combo", type=str, default="lw_rp")
    parser.add_argument("--seq-id", type=int, required=True, help="IMUPoser sequence id, e.g. 102 / 12")
    parser.add_argument("--port", type=int, default=8989, help="Must match Unity Client Port")
    parser.add_argument("--ip", type=str, default="127.0.0.1")
    parser.add_argument("--fps", type=float, default=None, help="Playback fps (default: data fps)")
    parser.add_argument("--start", type=int, default=0, help="Start frame")
    parser.add_argument("--end", type=int, default=None, help="End frame (exclusive)")
    parser.add_argument(
        "--once",
        action="store_true",
        help="Play the sequence once then exit (default: loop until Ctrl+C)",
    )
    parser.add_argument(
        "--auto-start",
        action="store_true",
        help="Start streaming immediately after Unity connects (skip Enter prompt)",
    )
    parser.add_argument(
        "--no-markers",
        action="store_true",
        help="Disable watch/phone correctness markers (enabled by default)",
    )
    parser.add_argument(
        "--marker-offset",
        type=float,
        default=0.12,
        help="Meters to push spheres OUTSIDE the body along limb/lateral (default 0.12)",
    )
    parser.add_argument(
        "--marker-radius",
        type=float,
        default=0.08,
        help="Sphere radius in meters (default 0.08)",
    )
    parser.add_argument("--no-tran", action="store_true", help="Zero root translation (in-place)")
    args = parser.parse_args()
    do_loop = not args.once
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
    if args.seq_id < 0 or args.seq_id >= len(pose_data["pose"]):
        raise IndexError(f"seq-id {args.seq_id} out of range [0, {len(pose_data['pose'])})")

    pose = pose_data["pose"][args.seq_id]  # [T, 24, 3, 3] or flat
    tran = pose_data["tran"][args.seq_id]
    pose = pose.view(-1, 24, 3, 3)
    tran = tran.view(-1, 3)
    if args.no_tran:
        tran = torch.zeros_like(tran)

    t0 = max(0, int(args.start))
    t1 = int(args.end) if args.end is not None else pose.shape[0]
    t1 = min(t1, pose.shape[0])
    if t1 <= t0:
        raise ValueError(f"empty frame range [{t0}, {t1})")

    yw_f = yp_f = pw_f = pp_f = None
    body_model = None
    device = torch.device(cfg["train"]["device"] if torch.cuda.is_available() else "cpu")
    if use_markers:
        print("Loading position classifier predictions for markers...")
        pack = _load_preds_for_seq(cfg, args.combo, args.seq_id, device)
        yw_f = expand_to_frames(pack["starts"], pack["yw"], pose.shape[0], stride, window_len)
        yp_f = expand_to_frames(pack["starts"], pack["yp"], pose.shape[0], stride, window_len)
        pw_f = expand_to_frames(pack["starts"], pack["pw"], pose.shape[0], stride, window_len)
        pp_f = expand_to_frames(pack["starts"], pack["pp"], pose.shape[0], stride, window_len)
        w_acc = float((pw_f == yw_f).mean()) if len(yw_f) else 0.0
        p_acc = float((pp_f == yp_f).mean()) if len(yp_f) else 0.0
        j_acc = float(((pw_f == yw_f) & (pp_f == yp_f)).mean()) if len(yw_f) else 0.0
        print(f"  frame-level acc  watch={w_acc:.3f}  phone={p_acc:.3f}  joint={j_acc:.3f}")
        body_model = ParametricModel(paths.smpl_file)

    MotionViewer.ip = args.ip
    MotionViewer.port = int(args.port)

    n_frames = t1 - t0
    est_sec = n_frames / fps
    print("=" * 60)
    print("Unity Online streaming")
    print(f"  bind       {args.ip}:{args.port}")
    print(f"  combo/seq  {args.combo} / {args.seq_id}")
    print(f"  frames     [{t0}, {t1}) = {n_frames} frames (~{est_sec:.1f}s/pass)")
    print(f"  fps={fps}  loop={do_loop}  markers={use_markers}")
    if use_markers:
        print(
            f"  markers: world spheres offset={args.marker_offset}m radius={args.marker_radius}m "
            "(green=OK, red=WRONG; blue/cyan=GT if wrong)"
        )
        print("  tip: if still buried, raise --marker-offset 0.18 --marker-radius 0.10")
        print("  Unity: keep Online → Points enabled")
    print("Startup:")
    print("  1) Keep this process running (waiting for Unity)...")
    print("  2) Unity: enable Hierarchy 'Online', disable 'Offline'")
    print("  3) Client: 127.0.0.1 / 8989 / Connect On Load")
    print("  4) Press Play; if body is cropped, switch Game view camera to Front/Side")
    print("  5) After connect, press Enter here to start playback (unless --auto-start)")
    print("=" * 60)

    name = f"seq{args.seq_id}_{args.combo}"
    with MotionViewer(1, overlap=False, names=[name], fps=fps) as viewer:
        print(f"Unity connected from {viewer.conn.getpeername()}")
        if not args.auto_start:
            try:
                input(">>> Switch to Unity Game view, then press Enter here to START streaming...")
            except EOFError:
                print("no stdin; starting in 3s...")
                time.sleep(3.0)

        print("Streaming... (Ctrl+C to stop)")
        pass_id = 0
        try:
            while True:
                pass_id += 1
                for i, fidx in enumerate(range(t0, t1)):
                    t_wall = time.time()
                    viewer.clear_all(render=False)
                    viewer.update(pose[fidx], tran[fidx], index=0, render=False)

                    if use_markers and body_model is not None:
                        yw = int(yw_f[fidx]) if yw_f[fidx] >= 0 else 0
                        yp = int(yp_f[fidx]) if yp_f[fidx] >= 0 else 0
                        pw = int(pw_f[fidx]) if pw_f[fidx] >= 0 else yw
                        pp = int(pp_f[fidx]) if pp_f[fidx] >= 0 else yp
                        fk_out = body_model.forward_kinematics(
                            pose[fidx : fidx + 1],
                            tran=tran[fidx : fidx + 1],
                            calc_mesh=False,
                        )
                        # calc_mesh=False → (grot, joint); True → (grot, joint, vert)
                        joints_t = fk_out[1]
                        joints = joints_t[0].detach().cpu().numpy()
                        _draw_device_markers(
                            viewer,
                            joints,
                            yw,
                            yp,
                            pw,
                            pp,
                            offset=float(args.marker_offset),
                            radius=float(args.marker_radius),
                        )

                    viewer.render()
                    if i == 0 or (i + 1) % 30 == 0 or i + 1 == n_frames:
                        extra = ""
                        if use_markers:
                            w_ok = int(pw_f[fidx]) == int(yw_f[fidx])
                            p_ok = int(pp_f[fidx]) == int(yp_f[fidx])
                            extra = f"  watch={'OK' if w_ok else 'WRONG'} phone={'OK' if p_ok else 'WRONG'}"
                        print(f"  pass={pass_id}  frame {fidx} ({i + 1}/{n_frames}){extra}")
                    time.sleep(max(0.0, (1.0 / fps) - (time.time() - t_wall)))
                if not do_loop:
                    print("single pass done (re-run without --once to loop).")
                    break
                print(f"pass {pass_id} done → replay (Ctrl+C to stop)")
        except KeyboardInterrupt:
            print("\nstopped by user")
        except (BrokenPipeError, ConnectionResetError, OSError) as e:
            print(f"\nUnity disconnected: {e}")

    print("done.")


if __name__ == "__main__":
    main()
