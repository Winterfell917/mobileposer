"""Offscreen Pyrender helpers for SMPL comparison videos (EGL / headless)."""
from __future__ import annotations

from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple

import numpy as np


def look_at(eye, target, up=(0.0, 1.0, 0.0)) -> np.ndarray:
    eye = np.asarray(eye, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    up = np.asarray(up, dtype=np.float64)
    z = eye - target
    z /= np.linalg.norm(z) + 1e-8
    x = np.cross(up, z)
    x /= np.linalg.norm(x) + 1e-8
    y = np.cross(z, x)
    pose = np.eye(4, dtype=np.float64)
    pose[:3, 0] = x
    pose[:3, 1] = y
    pose[:3, 2] = z
    pose[:3, 3] = eye
    return pose


def camera_from_verts(*vert_sets: np.ndarray) -> Tuple[np.ndarray, np.ndarray, float]:
    """Shared camera so GT / pred stay in the same frame."""
    pts = np.concatenate([np.asarray(v).reshape(-1, 3) for v in vert_sets], axis=0)
    center = pts.mean(axis=0)
    extent = float(np.max(np.linalg.norm(pts - center, axis=1)) + 1e-6)
    eye = center + np.array([0.0, 0.25 * extent, 3.2 * extent], dtype=np.float64)
    return look_at(eye, center), center, extent


class PyrenderSMPL:
    """Reuse one OffscreenRenderer; rebuild the scene per call."""

    def __init__(self, faces: np.ndarray, width: int = 640, height: int = 720):
        import pyrender

        self.faces = np.asarray(faces, dtype=np.int32)
        self.pyrender = pyrender
        self.renderer = pyrender.OffscreenRenderer(width, height)
        self.camera = pyrender.PerspectiveCamera(yfov=np.pi / 4.0, aspectRatio=width / height)
        self.width = int(width)
        self.height = int(height)

    def close(self) -> None:
        self.renderer.delete()

    def _mesh(self, verts: np.ndarray, rgba) -> object:
        import trimesh

        mesh = trimesh.Trimesh(vertices=verts, faces=self.faces, process=False)
        color = np.asarray(rgba, dtype=np.float32)
        if color.size == 3:
            color = np.append(color, 1.0)
        mesh.visual.vertex_colors = (color * 255).astype(np.uint8)
        return self.pyrender.Mesh.from_trimesh(mesh, smooth=True)

    def _lights(self, scene, cam_pose: np.ndarray, center: np.ndarray, extent: float) -> None:
        scene.add(self.pyrender.DirectionalLight(color=np.ones(3), intensity=3.5), pose=cam_pose)
        fill_pose = look_at(center + np.array([-2.0 * extent, 1.5 * extent, 0.5 * extent]), center)
        scene.add(self.pyrender.DirectionalLight(color=np.ones(3), intensity=1.2), pose=fill_pose)

    def render(
        self,
        verts: np.ndarray,
        rgba=(0.75, 0.55, 0.40, 1.0),
        cam_pose: Optional[np.ndarray] = None,
        center: Optional[np.ndarray] = None,
        extent: Optional[float] = None,
    ) -> np.ndarray:
        return self.render_many([(verts, rgba)], cam_pose=cam_pose, center=center, extent=extent)

    def render_many(
        self,
        items: Sequence[Tuple[np.ndarray, Sequence[float]]],
        cam_pose: Optional[np.ndarray] = None,
        center: Optional[np.ndarray] = None,
        extent: Optional[float] = None,
    ) -> np.ndarray:
        if cam_pose is None or center is None or extent is None:
            cam_pose, center, extent = camera_from_verts(*(v for v, _ in items))
        scene = self.pyrender.Scene(bg_color=[1.0, 1.0, 1.0, 1.0], ambient_light=[0.4, 0.4, 0.4])
        for verts, rgba in items:
            scene.add(self._mesh(verts, rgba))
        scene.add(self.camera, pose=cam_pose)
        self._lights(scene, cam_pose, center, float(extent))
        color_img, _ = self.renderer.render(scene)
        return color_img


def _put_text(img: np.ndarray, text: str, org: Tuple[int, int], scale: float, color, thick: int = 2) -> None:
    import cv2

    cv2.putText(
        img,
        text,
        org,
        cv2.FONT_HERSHEY_SIMPLEX,
        scale,
        color,
        thick,
        cv2.LINE_AA,
    )


def label_panel(img: np.ndarray, text: str, rgb=(40, 40, 40)) -> np.ndarray:
    import cv2

    out = img.copy()
    bar_h = 36
    cv2.rectangle(out, (0, 0), (out.shape[1], bar_h), (255, 255, 255), -1)
    _put_text(out, text, (12, 26), 0.72, (int(rgb[2]), int(rgb[1]), int(rgb[0])), 2)
    return out


def banner(width: int, lines: Sequence[str], height: int = 52) -> np.ndarray:
    import cv2

    bar = np.full((height, width, 3), 255, dtype=np.uint8)
    y = 22
    for line in lines:
        _put_text(bar, line, (12, y), 0.55, (40, 40, 40), 1)
        y += 22
    cv2.line(bar, (0, height - 1), (width - 1, height - 1), (210, 210, 210), 1)
    return bar


def hstack(imgs: Sequence[np.ndarray], gap: int = 10) -> np.ndarray:
    h = max(im.shape[0] for im in imgs)
    parts = []
    for i, im in enumerate(imgs):
        if im.shape[0] != h:
            pad = np.full((h - im.shape[0], im.shape[1], 3), 255, dtype=np.uint8)
            im = np.vstack([im, pad])
        parts.append(im)
        if i + 1 < len(imgs) and gap > 0:
            parts.append(np.full((h, gap, 3), 255, dtype=np.uint8))
    return np.hstack(parts)


def compose_row(
    panels: Sequence[np.ndarray],
    labels: Sequence[str],
    label_rgbs: Sequence[Tuple[int, int, int]],
    header_lines: Sequence[str],
) -> np.ndarray:
    labeled = [label_panel(im, lab, rgb) for im, lab, rgb in zip(panels, labels, label_rgbs)]
    row = hstack(labeled)
    top = banner(row.shape[1], header_lines)
    return np.vstack([top, row])


def pad_even(img: np.ndarray) -> np.ndarray:
    h, w = img.shape[:2]
    ph, pw = h % 2, w % 2
    if ph == 0 and pw == 0:
        return img
    return np.pad(img, ((0, ph), (0, pw), (0, 0)), mode="constant", constant_values=255)


def write_mp4(frames: Iterable[np.ndarray], path: Path, fps: float) -> None:
    import cv2

    first = None
    writer = None
    n = 0
    try:
        for frame in frames:
            frame = pad_even(frame)
            if writer is None:
                first = frame
                h, w = frame.shape[:2]
                writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), float(fps), (w, h))
                if not writer.isOpened():
                    raise SystemExit(f"OpenCV could not open VideoWriter for {path}")
            writer.write(frame[:, :, ::-1].copy())
            n += 1
            if n % 50 == 0:
                print(f"  encoded {n}")
    finally:
        if writer is not None:
            writer.release()
    if first is None:
        raise SystemExit("no frames to write")
    print(f"  encoded {n}/{n}")


def save_png(img: np.ndarray, path: Path) -> None:
    from PIL import Image

    Image.fromarray(img).save(path)


def contact_sheet(rows: List[np.ndarray], gap: int = 12) -> np.ndarray:
    w = max(r.shape[1] for r in rows)
    parts = []
    for i, row in enumerate(rows):
        if row.shape[1] != w:
            pad = np.full((row.shape[0], w - row.shape[1], 3), 255, dtype=np.uint8)
            row = np.hstack([row, pad])
        parts.append(row)
        if i + 1 < len(rows) and gap > 0:
            parts.append(np.full((gap, w, 3), 255, dtype=np.uint8))
    return np.vstack(parts)
