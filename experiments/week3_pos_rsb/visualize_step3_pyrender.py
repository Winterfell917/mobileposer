#!/usr/bin/env python3
"""Week3 step-3 Pyrender comparison: GT / Pred-seq / Baseline in one frame.

Renders IMUPoser sequences with the same offscreen Pyrender path as
render_pyrender_test.py. Default layout is side-by-side:

    GT pose | Pred-seq (pos → R_SB → R_MB → MobilePoser) | Baseline (R_MS → pose)

All panels share one camera (from the GT mesh). Works headless via EGL.

Usage (repo root, conda env mobileposer):
  python experiments/week3_pos_rsb/visualize_step3_pyrender.py --seq-ids 13 --combo lw_rp
  python experiments/week3_pos_rsb/visualize_step3_pyrender.py --seq-ids 13 --layout overlay

Outputs under experiments/week3_pos_rsb/outputs/figures/step3_pyrender/
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

_EXP_DIR = Path(__file__).resolve().parent
_REPO = _EXP_DIR.parents[1]
_MP_DIR = _REPO / "mobileposer"
if str(_EXP_DIR) not in sys.path:
    sys.path.insert(0, str(_EXP_DIR))

from cascade_pose import (  # noqa: E402
    cascade_calibrated,
    load_imuposer_pose_one,
    pack_condition_imu,
    predict_pose_sequence,
)
from dataset import load_config, offset_euler_bounds, resolve_path, set_seed  # noqa: E402
from dataset.pos_dataset import amass_seed_offset  # noqa: E402
from pyrender_smpl import (  # noqa: E402
    PyrenderSMPL,
    camera_from_verts,
    compose_row,
    contact_sheet,
    save_png,
    write_mp4,
)
from visualize_step3 import fk_mesh, load_cascade_models, parse_combo  # noqa: E402

PANEL_SPEC = {
    "gt": {
        "label": "GT pose",
        "rgba": (0.42, 0.68, 0.50, 1.0),
        "rgb": (50, 130, 80),
    },
    "pred_seq": {
        "label": "Pred-seq cascade",
        "rgba": (0.90, 0.55, 0.18, 1.0),
        "rgb": (200, 110, 20),
    },
    "none": {
        "label": "Baseline (R_MS->pose)",
        "rgba": (0.75, 0.30, 0.30, 1.0),
        "rgb": (170, 50, 50),
    },
}


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


def parse_panels(s: str, include_none: bool) -> List[str]:
    keys = [t.strip() for t in s.split(",") if t.strip()]
    if include_none and "none" not in keys:
        keys = keys + ["none"]
    unknown = [k for k in keys if k not in PANEL_SPEC]
    if unknown:
        raise SystemExit(f"unknown --panels {unknown}; choose from {list(PANEL_SPEC)}")
    if not keys:
        raise SystemExit("--panels is empty")
    return keys


def header_lines(sid: int, combo: str, motion: str, meta: dict, t: Optional[int] = None) -> List[str]:
    slot = (
        f"slot Pred=({meta['pred_watch']},{meta['pred_phone']})  "
        f"GT=({meta['y_watch']},{meta['y_phone']})  "
        f"joint={'ok' if meta['joint_ok'] else 'miss'}"
    )
    left = f"IMUPoser seq{sid:03d}  {combo}"
    if motion:
        left = f"{left}  {motion}"
    if t is not None:
        left = f"{left}  t={t}"
    return [left, slot]


def render_side(
    renderer: PyrenderSMPL,
    meshes: Dict[str, Tuple[np.ndarray, np.ndarray]],
    keys: Sequence[str],
    t: int,
    cam_pose: np.ndarray,
    center: np.ndarray,
    extent: float,
    header: Sequence[str],
) -> np.ndarray:
    panels = []
    labels = []
    rgbs = []
    for key in keys:
        spec = PANEL_SPEC[key]
        img = renderer.render(
            meshes[key][0][t],
            rgba=spec["rgba"],
            cam_pose=cam_pose,
            center=center,
            extent=extent,
        )
        panels.append(img)
        labels.append(spec["label"])
        rgbs.append(spec["rgb"])
    return compose_row(panels, labels, rgbs, header)


def render_overlay(
    renderer: PyrenderSMPL,
    meshes: Dict[str, Tuple[np.ndarray, np.ndarray]],
    keys: Sequence[str],
    t: int,
    cam_pose: np.ndarray,
    center: np.ndarray,
    extent: float,
    header: Sequence[str],
) -> np.ndarray:
    items = [(meshes[k][0][t], PANEL_SPEC[k]["rgba"]) for k in keys]
    img = renderer.render_many(items, cam_pose=cam_pose, center=center, extent=extent)
    label = " + ".join(PANEL_SPEC[k]["label"] for k in keys)
    return compose_row([img], [label], [(40, 40, 40)], header)


def infer_meshes(
    body_model,
    pose_net,
    item,
    pack,
    keys: Sequence[str],
    w_idx: int,
    p_idx: int,
    acc_scale: float,
) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
    acc, ori, pose_gt, tran_gt = item
    meshes: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
    if "gt" in keys:
        meshes["gt"] = fk_mesh(body_model, pose_gt.float(), tran_gt.float())
    for key in keys:
        if key == "gt":
            continue
        acc_c, ori_c = pack[key]
        imu = pack_condition_imu(
            acc_c,
            ori_c,
            name=key,
            w_idx=w_idx,
            p_idx=p_idx,
            acc_scale=acc_scale,
            dst_watch=pack["dst_watch"],
            dst_phone=pack["dst_phone"],
        )
        pose_p, tran_p = predict_pose_sequence(pose_net, imu)
        meshes[key] = fk_mesh(body_model, pose_p, tran_p)
    t_min = min(v.shape[0] for v, _ in meshes.values())
    for key in list(meshes):
        verts, joints = meshes[key]
        meshes[key] = (verts[:t_min], joints[:t_min])
    return meshes


def main():
    parser = argparse.ArgumentParser(description="Week3 step-3 Pyrender GT vs Pred-seq")
    parser.add_argument("--config", type=str, default="experiments/week3_pos_rsb/configs/default.yaml")
    parser.add_argument("--combo", type=str, default="lw_rp")
    parser.add_argument("--seq-ids", type=str, default="13,8")
    parser.add_argument("--panels", type=str, default="gt,pred_seq,none", help="gt,pred_seq,none")
    parser.add_argument("--include-none", action="store_true", help="Append Baseline if not already in --panels")
    parser.add_argument("--layout", type=str, default="side", choices=("side", "overlay"))
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument("--width", type=int, default=640, help="Width of each panel")
    parser.add_argument("--height", type=int, default=720, help="Height of each panel")
    parser.add_argument("--n-frames", type=int, default=4, help="Contact-sheet stills")
    parser.add_argument("--max-frames", type=int, default=0, help="Cap video length; 0 = full sequence")
    parser.add_argument("--no-video", action="store_true")
    parser.add_argument("--no-stills", action="store_true")
    args = parser.parse_args()

    os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
    try:
        import pyrender  # noqa: F401
        import trimesh  # noqa: F401
    except ImportError as exc:
        raise SystemExit(
            "Missing dependency. In the mobileposer env run:\n"
            "  pip install pyrender trimesh\n"
            f"Original error: {exc}"
        ) from exc

    keys = parse_panels(args.panels, bool(args.include_none))
    cfg = load_config(resolve_path(args.config))
    set_seed(cfg["experiment"]["seed"])
    out_dir = resolve_path("experiments/week3_pos_rsb/outputs/figures/step3_pyrender")
    out_dir.mkdir(parents=True, exist_ok=True)

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
    faces = np.asarray(body_model.face)

    yw, yp = parse_combo(args.combo)
    w_idx, p_idx = (0 if yw == 0 else 1), (2 if yp == 0 else 3)
    imu_path = resolve_path(cfg["data"]["processed_imuposer_file"])
    window_len = int(cfg["data"]["window_len"])
    stride = int(cfg["data"]["test_stride"])
    acc_scale = float(cfg["data"]["acc_scale"])
    offset_range = float(cfg["data"]["offset_range_deg"])
    lo_deg, hi_deg = offset_euler_bounds(cfg["data"])
    seed = int(cfg["experiment"]["seed"])
    seq_ids = [int(x) for x in args.seq_ids.split(",") if x.strip() != ""]

    motion_map = []
    raw_dir = resolve_path("data/raw/IMUPoser")
    if raw_dir.exists():
        motion_map = list_imuposer_motions(raw_dir)

    panel_w = int(args.width)
    if args.layout == "overlay":
        panel_w = min(960, max(int(args.width), 720))
    renderer = PyrenderSMPL(faces, width=panel_w, height=int(args.height))

    try:
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
                lo_deg=lo_deg,
                hi_deg=hi_deg,
            )
            if pack is None:
                print(f"[warn] seq {sid} cascade failed")
                continue

            print(f"FK + pose net  seq={sid} T={pose_gt.shape[0]}  panels={keys}  layout={args.layout}")
            meshes = infer_meshes(
                body_model,
                pose_net,
                (acc, ori, pose_gt, tran_gt),
                pack,
                keys,
                w_idx,
                p_idx,
                acc_scale,
            )
            t_len = meshes[keys[0]][0].shape[0]
            n_vid = t_len if int(args.max_frames) <= 0 else min(t_len, int(args.max_frames))
            cam_src = meshes["gt"][0][:n_vid] if "gt" in meshes else meshes[keys[0]][0][:n_vid]
            cam_pose, center, extent = camera_from_verts(cam_src)

            motion = ""
            if sid < len(motion_map):
                motion = f"{motion_map[sid]['pid']}/{motion_map[sid]['motion']}"
            meta = pack["meta"]
            tag = f"seq{sid:03d}_{args.combo}_{args.layout}"
            render_fn = render_overlay if args.layout == "overlay" else render_side

            if not args.no_stills:
                n_show = min(int(args.n_frames), n_vid)
                frame_ids = np.linspace(0, n_vid - 1, n_show, dtype=np.int64)
                rows = [
                    render_fn(
                        renderer,
                        meshes,
                        keys,
                        int(fidx),
                        cam_pose,
                        center,
                        extent,
                        header_lines(sid, args.combo, motion, meta, t=int(fidx)),
                    )
                    for fidx in frame_ids
                ]
                sheet = contact_sheet(rows)
                png = out_dir / f"step3_pyrender_{tag}.png"
                save_png(sheet, png)
                print(f"saved {png}")

            if not args.no_video:
                mp4 = out_dir / f"step3_pyrender_{tag}.mp4"

                def _frames():
                    for t in range(n_vid):
                        yield render_fn(
                            renderer,
                            meshes,
                            keys,
                            t,
                            cam_pose,
                            center,
                            extent,
                            header_lines(sid, args.combo, motion, meta, t=t),
                        )

                print(f"encoding {n_vid} frames -> {mp4}")
                write_mp4(_frames(), mp4, args.fps)
                print(f"saved {mp4}")
    finally:
        renderer.close()
    print(f"done. figures in {out_dir}")


if __name__ == "__main__":
    main()
