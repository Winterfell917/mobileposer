"""
Diagnose whether IMUPoser recorded IMU global == SMPL pose world (DIP-aligned).

For each sequence (pose DIP-aligned → FK bone ori R_bone):
  H_old: use ori_raw          (preprocess: align pose only)
  H_new: use R_align @ ori_raw (preprocess: align pose + IMU)
  H_opt: use W* @ ori_raw      (W* = best constant left-multiply, Procrustes)

If IMU shares pose's native world, H_new should win (R_rel≈R_SB stable).
If IMU is already near DIP while pose needed R_align, H_old should win.
If H_opt ≪ both and W* ≉ I and W* ≉ R_align → third unknown world.

Run from repo root:
  conda run -n mobileposer python experiments/week1_pos_cls/scripts/diagnose_imuposer_frame.py
"""
from __future__ import annotations

import json
import pickle
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "mobileposer"))

from articulate import math  # noqa: E402
from articulate.model import ParametricModel  # noqa: E402
from config import paths  # noqa: E402

R_ALIGN = torch.tensor(
    [[-1.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, 1.0, 0.0]], dtype=torch.float32
)
# IMUPoser slots 0..4 ↔ bone globals from ji_mask[:5]
JI_MASK = torch.tensor([18, 19, 1, 2, 15])  # LW,RW,LP,RP,Head (elbows for wrists)
SLOT_NAMES = ["LW", "RW", "LP", "RP", "Head"]
MAX_SEQS = 40
MAX_FRAMES = 400
STRIDE = 2


def geodesic_deg(R_a: torch.Tensor, R_b: torch.Tensor) -> torch.Tensor:
    """Angle (deg) between rotations. R_*: [..., 3, 3]."""
    R = R_a.transpose(-1, -2) @ R_b
    cos = ((R[..., 0, 0] + R[..., 1, 1] + R[..., 2, 2]) - 1.0) * 0.5
    cos = cos.clamp(-1.0 + 1e-6, 1.0 - 1e-6)
    return torch.rad2deg(torch.acos(cos))


def rot_procrustes_left(R_src: torch.Tensor, R_tgt: torch.Tensor) -> torch.Tensor:
    """
    Best W (3x3 rot) s.t. W @ R_src ≈ R_tgt in Frobenius sense.
    Stack columns as 3-vectors: solve Wahba / Kabsch on paired columns.
    R_src/tgt: [N, 3, 3]
    """
    # Map each column of R_src to corresponding column of R_tgt
    X = R_src.reshape(-1, 3).T  # 3 x 3N
    Y = R_tgt.reshape(-1, 3).T
    M = Y @ X.T
    U, _, Vh = torch.linalg.svd(M)
    W = U @ Vh
    if torch.det(W) < 0:
        U = U.clone()
        U[:, -1] *= -1
        W = U @ Vh
    return W


def r_rel_stability_deg(R_bone: torch.Tensor, R_imu: torch.Tensor) -> dict:
    """
    R_rel = R_bone^T @ R_imu. If same world + constant R_SB, R_rel is stable.
    Report mean geodesic to median R_rel, plus raw bone-vs-imu geodesic.
    """
    R_rel = R_bone.transpose(-1, -2) @ R_imu  # [T,3,3]
    # chordal mean ≈ normalize average matrix
    M = R_rel.mean(0)
    U, _, Vh = torch.linalg.svd(M)
    R_med = U @ Vh
    if torch.det(R_med) < 0:
        U = U.clone()
        U[:, -1] *= -1
        R_med = U @ Vh
    stab = geodesic_deg(R_rel, R_med.expand_as(R_rel))
    raw = geodesic_deg(R_bone, R_imu)
    return {
        "rel_stab_mean_deg": float(stab.mean()),
        "rel_stab_med_deg": float(stab.median()),
        "raw_geodesic_mean_deg": float(raw.mean()),
        "raw_geodesic_med_deg": float(raw.median()),
    }


def acc_err(a_imu: torch.Tensor, a_syn: torch.Tensor) -> dict:
    err = (a_imu - a_syn).norm(dim=-1)
    return {
        "acc_l2_mean": float(err.mean()),
        "acc_l2_med": float(err.median()),
        "acc_corr": float(
            torch.nn.functional.cosine_similarity(
                a_imu.reshape(-1, 3), a_syn.reshape(-1, 3), dim=-1
            ).mean()
        ),
    }


def syn_acc(v: torch.Tensor, smooth_n: int = 4) -> torch.Tensor:
    mid = smooth_n // 2
    scale = 30.0**2
    acc = torch.stack(
        [(v[i] + v[i + 2] - 2 * v[i + 1]) * scale for i in range(0, v.shape[0] - 2)]
    )
    acc = torch.cat((torch.zeros_like(acc[:1]), acc, torch.zeros_like(acc[:1])))
    if mid != 0:
        acc[smooth_n:-smooth_n] = torch.stack(
            [
                (v[i] + v[i + smooth_n * 2] - 2 * v[i + smooth_n])
                * scale
                / smooth_n**2
                for i in range(0, v.shape[0] - smooth_n * 2)
            ]
        )
    return acc


def load_sequences(max_seqs: int):
    body = ParametricModel(paths.smpl_file)
    seqs = []
    for pid_path in sorted(paths.raw_imuposer.iterdir()):
        if not pid_path.is_dir() or not pid_path.name.startswith("P"):
            continue
        for fpath in sorted(pid_path.iterdir()):
            with open(fpath, "rb") as f:
                fdata = pickle.load(f)
            acc = fdata["imu"][:, : 5 * 3].view(-1, 5, 3).float()
            ori = fdata["imu"][:, 5 * 3 :].view(-1, 5, 3, 3).float()
            pose = math.axis_angle_to_rotation_matrix(fdata["pose"]).view(-1, 24, 3, 3)
            tran = fdata["trans"].to(torch.float32)
            if acc.shape[0] < 60:
                continue
            # DIP-align pose/tran only (canonical for FK target)
            pose = pose.clone()
            pose[:, 0] = R_ALIGN @ pose[:, 0]
            tran = tran @ R_ALIGN
            with torch.no_grad():
                grot, _, vert = body.forward_kinematics(
                    pose=pose, tran=tran, calc_mesh=True
                )
            R_bone = grot[:, JI_MASK].float()
            vacc = syn_acc(vert[:, torch.tensor([1961, 5424, 876, 4362, 411])])
            # subsample frames
            T = min(acc.shape[0], R_bone.shape[0], vacc.shape[0])
            idx = torch.arange(0, T, STRIDE)[:MAX_FRAMES]
            seqs.append(
                {
                    "name": f"{pid_path.name}/{fpath.name}",
                    "acc_raw": acc[idx],
                    "ori_raw": ori[idx],
                    "R_bone": R_bone[idx],
                    "vacc": vacc[idx],
                }
            )
            if len(seqs) >= max_seqs:
                return seqs
    return seqs


def pool_hypotheses(seqs):
    # Collect all frames all slots for W* fit (use Head+pockets; wrists are elbow proxy)
    fit_slots = [2, 3, 4]  # LP, RP, Head — better than elbow-proxy wrists
    R_src, R_tgt = [], []
    for s in seqs:
        for k in fit_slots:
            R_src.append(s["ori_raw"][:, k])
            R_tgt.append(s["R_bone"][:, k])
    R_src = torch.cat(R_src, 0)
    R_tgt = torch.cat(R_tgt, 0)
    W_star = rot_procrustes_left(R_src, R_tgt)

    def summarize(name, transform):
        per_slot = {n: [] for n in SLOT_NAMES}
        acc_pool = {n: [] for n in SLOT_NAMES}
        for s in seqs:
            ori = transform(s["ori_raw"])
            acc = transform_acc(s["acc_raw"], transform)
            for k, n in enumerate(SLOT_NAMES):
                per_slot[n].append(r_rel_stability_deg(s["R_bone"][:, k], ori[:, k]))
                acc_pool[n].append(acc_err(acc[:, k], s["vacc"][:, k]))
        out = {"hypothesis": name}
        for n in SLOT_NAMES:
            stab = torch.tensor([d["rel_stab_mean_deg"] for d in per_slot[n]])
            raw = torch.tensor([d["raw_geodesic_mean_deg"] for d in per_slot[n]])
            al2 = torch.tensor([d["acc_l2_mean"] for d in acc_pool[n]])
            ac = torch.tensor([d["acc_corr"] for d in acc_pool[n]])
            out[n] = {
                "rel_stab_mean_deg": float(stab.mean()),
                "raw_geo_mean_deg": float(raw.mean()),
                "acc_l2_mean": float(al2.mean()),
                "acc_corr": float(ac.mean()),
            }
        # focus slots average
        focus = ["LP", "RP", "Head"]
        out["focus_avg"] = {
            k: float(sum(out[n][k] for n in focus) / len(focus))
            for k in ["rel_stab_mean_deg", "raw_geo_mean_deg", "acc_l2_mean", "acc_corr"]
        }
        return out

    def transform_acc(acc, rot_fn):
        # rot_fn expects [T,5,3,3]-like; for acc apply same left multiply if rot_fn is left W
        # We pass lambdas that work on ori; mirror for acc below in callers
        return acc

    # redefine with proper acc transforms
    results = []

    def hyp(name, W):
        per_slot = {n: [] for n in SLOT_NAMES}
        acc_pool = {n: [] for n in SLOT_NAMES}
        for s in seqs:
            ori = torch.einsum("ij,tajk->taik", W, s["ori_raw"])
            acc = torch.einsum("ij,taj->tai", W, s["acc_raw"])
            for k, n in enumerate(SLOT_NAMES):
                per_slot[n].append(r_rel_stability_deg(s["R_bone"][:, k], ori[:, k]))
                acc_pool[n].append(acc_err(acc[:, k], s["vacc"][:, k]))
        out = {"hypothesis": name, "W": W.tolist()}
        for n in SLOT_NAMES:
            stab = torch.tensor([d["rel_stab_mean_deg"] for d in per_slot[n]])
            raw = torch.tensor([d["raw_geodesic_mean_deg"] for d in per_slot[n]])
            al2 = torch.tensor([d["acc_l2_mean"] for d in acc_pool[n]])
            ac = torch.tensor([d["acc_corr"] for d in acc_pool[n]])
            out[n] = {
                "rel_stab_mean_deg": float(stab.mean()),
                "raw_geo_mean_deg": float(raw.mean()),
                "acc_l2_mean": float(al2.mean()),
                "acc_corr": float(ac.mean()),
            }
        focus = ["LP", "RP", "Head"]
        out["focus_avg"] = {
            k: float(sum(out[n][k] for n in focus) / len(focus))
            for k in ["rel_stab_mean_deg", "raw_geo_mean_deg", "acc_l2_mean", "acc_corr"]
        }
        return out

    I = torch.eye(3)
    results.append(hyp("H_old: W=I (pose-align only)", I))
    results.append(hyp("H_new: W=R_align (pose+IMU)", R_ALIGN))
    results.append(hyp("H_opt: W=W* (Procrustes)", W_star))

    # compare W* to I and R_align
    def fro_to(A, B):
        return float(torch.norm(A - B))

    meta = {
        "n_seqs": len(seqs),
        "fit_slots": [SLOT_NAMES[i] for i in fit_slots],
        "W_star": W_star.tolist(),
        "||W*-I||_F": fro_to(W_star, I),
        "||W*-R_align||_F": fro_to(W_star, R_ALIGN),
        "geodesic(W*,I)_deg": float(geodesic_deg(W_star, I)),
        "geodesic(W*,R_align)_deg": float(geodesic_deg(W_star, R_ALIGN)),
        "det(W*)": float(torch.det(W_star)),
    }
    return meta, results


def verdict(meta, results):
    by = {r["hypothesis"]: r["focus_avg"]["rel_stab_mean_deg"] for r in results}
    best = min(by, key=by.get)
    lines = [
        f"Best by R_rel stability (LP/RP/Head): {best}",
        f"  H_old stab={by[[k for k in by if k.startswith('H_old')][0]]:.2f}°",
        f"  H_new stab={by[[k for k in by if k.startswith('H_new')][0]]:.2f}°",
        f"  H_opt stab={by[[k for k in by if k.startswith('H_opt')][0]]:.2f}°",
        f"W* vs I: {meta['geodesic(W*,I)_deg']:.1f}°, vs R_align: {meta['geodesic(W*,R_align)_deg']:.1f}°",
    ]
    # interpretation
    g_i = meta["geodesic(W*,I)_deg"]
    g_a = meta["geodesic(W*,R_align)_deg"]
    old = by[[k for k in by if k.startswith("H_old")][0]]
    new = by[[k for k in by if k.startswith("H_new")][0]]
    if new + 5 < old and g_a < 25:
        lines.append(
            "INTERPRET: Supports SAME native world as pose → should apply R_align to IMU."
        )
    elif old + 5 < new and g_i < 25:
        lines.append(
            "INTERPRET: Supports IMU already near DIP → do NOT apply R_align to IMU."
        )
    elif min(g_i, g_a) > 35:
        lines.append(
            "INTERPRET: W* far from both I and R_align → IMU global ≠ pose world; "
            "neither 'align IMU' nor 'leave IMU' is fully right; use W* or recalibrate."
        )
    else:
        lines.append(
            "INTERPRET: Mixed/weak; check per-slot table and acc_corr; "
            "Week1 should A/B eval H_old vs H_new."
        )
    return "\n".join(lines)


def main():
    print("Loading raw IMUPoser + FK (this may take a few minutes)...")
    seqs = load_sequences(MAX_SEQS)
    print(f"Loaded {len(seqs)} sequences")
    meta, results = pool_hypotheses(seqs)
    text = verdict(meta, results)
    print(text)
    print("\n=== focus_avg (LP/RP/Head) ===")
    for r in results:
        print(r["hypothesis"], json.dumps(r["focus_avg"], indent=None))
    print("\n=== per-slot rel_stab_mean_deg ===")
    for r in results:
        row = {n: round(r[n]["rel_stab_mean_deg"], 2) for n in SLOT_NAMES}
        print(r["hypothesis"][:28], row)

    out_dir = ROOT / "experiments/week1_pos_cls/outputs/logs"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "diagnose_imuposer_frame.json"
    payload = {"meta": meta, "results": results, "verdict": text}
    # drop bulky W from each hyp except meta
    out_path.write_text(json.dumps(payload, indent=2))
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
