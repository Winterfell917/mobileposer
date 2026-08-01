#!/usr/bin/env python3
"""
Evaluate Week1 position classifier (spec 6.1 / 6.2).

Reports:
  - window-level Watch / Phone / Joint accuracy
  - sequence-level Joint accuracy (majority vote)
  - confusion matrices
  - IMUPoser accuracy by motion type
  - window-length ablation: 30 / 60 / 90 / 150
  - init_done latency (K consecutive frames with stable prediction)
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

_EXP_DIR = Path(__file__).resolve().parent
if str(_EXP_DIR) not in sys.path:
    sys.path.insert(0, str(_EXP_DIR))

from dataset import (  # noqa: E402
    combo_to_indices,
    load_config,
    make_window_features,
    resolve_path,
    set_seed,
    slide_windows,
)
from dataset.pos_dataset import PosClsDataset, load_norm_stats  # noqa: E402
from models import PosClassifier  # noqa: E402

COMBO_NAMES = ["LW+LP", "LW+RP", "RW+LP", "RW+RP"]
ABLATION_WINDOW_LENS = [30, 60, 90, 150]


def confusion_2(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    cm = np.zeros((2, 2), dtype=np.int64)
    for t, p in zip(y_true, y_pred):
        cm[int(t), int(p)] += 1
    return cm


def confusion_4(yw: np.ndarray, yp: np.ndarray, pw: np.ndarray, pp: np.ndarray) -> np.ndarray:
    cm = np.zeros((4, 4), dtype=np.int64)
    yt = yw * 2 + yp
    yp_ = pw * 2 + pp
    for t, p in zip(yt, yp_):
        cm[int(t), int(p)] += 1
    return cm


def _acc(y: np.ndarray, p: np.ndarray) -> float:
    if len(y) == 0:
        return float("nan")
    return float((y == p).mean())


@torch.no_grad()
def predict_batches(
    model: nn.Module,
    x: torch.Tensor,
    device: torch.device,
    batch_size: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """x: [N, W, C] -> (p_watch, p_phone)."""
    model.eval()
    pws, pps = [], []
    for i in range(0, x.shape[0], batch_size):
        xb = x[i : i + batch_size].to(device)
        lw, lp = model(xb)
        pws.append(lw.argmax(dim=-1).cpu())
        pps.append(lp.argmax(dim=-1).cpu())
    if not pws:
        return np.zeros(0, dtype=np.int64), np.zeros(0, dtype=np.int64)
    return torch.cat(pws).numpy(), torch.cat(pps).numpy()


@torch.no_grad()
def run_eval_loader(model, loader, device) -> Dict[str, Any]:
    model.eval()
    all_yw, all_yp, all_pw, all_pp = [], [], [], []
    ce = nn.CrossEntropyLoss(reduction="sum")
    loss_sum = 0.0
    n = 0
    for x, yw, yp in tqdm(loader, desc="eval"):
        x = x.to(device)
        yw = yw.to(device)
        yp = yp.to(device)
        lw, lp = model(x)
        loss_sum += (ce(lw, yw) + ce(lp, yp)).item()
        all_yw.append(yw.cpu().numpy())
        all_yp.append(yp.cpu().numpy())
        all_pw.append(lw.argmax(dim=-1).cpu().numpy())
        all_pp.append(lp.argmax(dim=-1).cpu().numpy())
        n += x.size(0)

    yw = np.concatenate(all_yw) if all_yw else np.zeros(0, dtype=np.int64)
    yp = np.concatenate(all_yp) if all_yp else np.zeros(0, dtype=np.int64)
    pw = np.concatenate(all_pw) if all_pw else np.zeros(0, dtype=np.int64)
    pp = np.concatenate(all_pp) if all_pp else np.zeros(0, dtype=np.int64)
    return {
        "n": int(n),
        "loss": float(loss_sum / max(n, 1)),
        "watch_acc": _acc(yw, pw),
        "phone_acc": _acc(yp, pp),
        "joint_acc": float(((yw == pw) & (yp == pp)).mean()) if n else float("nan"),
        "cm_watch": confusion_2(yw, pw).tolist(),
        "cm_phone": confusion_2(yp, pp).tolist(),
        "cm_joint4": confusion_4(yw, yp, pw, pp).tolist(),
        "y_watch": yw,
        "y_phone": yp,
        "p_watch": pw,
        "p_phone": pp,
    }


def seq_key_from_meta(m: dict, split: str) -> Tuple:
    if split == "test" or "seq" in m:
        return ("seq", m["seq"], m.get("y_watch"), m.get("y_phone"))
    # AMASS: one sequence instance = source + local idx + wearing combo
    return (
        "amass",
        m.get("source"),
        m.get("seq_local"),
        m.get("y_watch"),
        m.get("y_phone"),
    )


def sequence_majority_vote(
    meta: List[dict],
    yw: np.ndarray,
    yp: np.ndarray,
    pw: np.ndarray,
    pp: np.ndarray,
    split: str,
) -> Optional[Dict[str, Any]]:
    if not meta or meta[0] is None:
        return None
    buckets: Dict[Tuple, List[int]] = defaultdict(list)
    for i, m in enumerate(meta):
        if m is None:
            continue
        buckets[seq_key_from_meta(m, split)].append(i)
    if not buckets:
        return None

    correct = 0
    for idxs in buckets.values():
        preds = [int(pw[i]) * 2 + int(pp[i]) for i in idxs]
        gts = [int(yw[i]) * 2 + int(yp[i]) for i in idxs]
        gt = max(set(gts), key=gts.count)
        pred = max(set(preds), key=preds.count)
        correct += int(pred == gt)
    return {"seq_joint_acc": correct / len(buckets), "n_seq": len(buckets)}


def list_imuposer_motions(raw_dir: Path) -> List[Dict[str, str]]:
    """Match data_process_mocap.process_imuposer iteration order."""
    subjects = {f"P{i}" for i in range(1, 11)}
    items = []
    for pid_path in sorted(raw_dir.iterdir()):
        if pid_path.name not in subjects:
            continue
        for fpath in sorted(pid_path.iterdir()):
            # "1. ArmRaises.pkl" -> ArmRaises
            parts = fpath.name.split(".")
            motion = parts[1].strip() if len(parts) >= 3 else fpath.stem
            items.append({"pid": pid_path.name, "motion": motion, "file": fpath.name})
    return items


def motion_type_breakdown(
    meta: List[dict],
    yw: np.ndarray,
    yp: np.ndarray,
    pw: np.ndarray,
    pp: np.ndarray,
    motion_map: List[Dict[str, str]],
) -> Dict[str, Any]:
    by_motion: Dict[str, List[int]] = defaultdict(list)
    for i, m in enumerate(meta):
        seq = int(m["seq"])
        if seq < 0 or seq >= len(motion_map):
            name = "unknown"
        else:
            name = motion_map[seq]["motion"]
        by_motion[name].append(i)

    out = {}
    for name, idxs in sorted(by_motion.items(), key=lambda kv: -len(kv[1])):
        ii = np.asarray(idxs, dtype=np.int64)
        out[name] = {
            "n": int(len(ii)),
            "watch_acc": _acc(yw[ii], pw[ii]),
            "phone_acc": _acc(yp[ii], pp[ii]),
            "joint_acc": float(((yw[ii] == pw[ii]) & (yp[ii] == pp[ii])).mean()),
        }
    return out


def init_done_stats(
    meta: List[dict],
    yw: np.ndarray,
    yp: np.ndarray,
    pw: np.ndarray,
    pp: np.ndarray,
    split: str,
    fps: float,
    k_frames: int = 30,
) -> Dict[str, Any]:
    """
    Spec 6.2: along each sequence timeline, assign each window prediction to
    frames [start, start+stride). When the last K frames share one combo,
    init_done=True; record latency.
    """
    buckets: Dict[Tuple, List[int]] = defaultdict(list)
    for i, m in enumerate(meta):
        if m is None:
            continue
        buckets[seq_key_from_meta(m, split)].append(i)

    latencies_s = []
    latencies_f = []
    init_ok = 0
    init_correct = 0
    never = 0

    for idxs in buckets.values():
        idxs = sorted(idxs, key=lambda i: int(meta[i].get("start", 0)))
        starts = [int(meta[i].get("start", 0)) for i in idxs]
        if len(starts) >= 2:
            stride = max(1, starts[1] - starts[0])
        else:
            stride = 15

        # GT combo (constant per seq instance)
        gt = int(yw[idxs[0]]) * 2 + int(yp[idxs[0]])
        # Build frame-wise prediction up to last window coverage
        last_end = starts[-1] + stride
        frame_pred = np.full(last_end, -1, dtype=np.int64)
        for i, st in zip(idxs, starts):
            pred = int(pw[i]) * 2 + int(pp[i])
            frame_pred[st : st + stride] = pred

        done_at = None
        stable_pred = None
        for t in range(k_frames - 1, last_end):
            seg = frame_pred[t - k_frames + 1 : t + 1]
            if (seg >= 0).all() and (seg == seg[0]).all():
                done_at = t + 1  # frames elapsed
                stable_pred = int(seg[0])
                break

        if done_at is None:
            never += 1
            continue
        init_ok += 1
        latencies_f.append(done_at)
        latencies_s.append(done_at / fps)
        init_correct += int(stable_pred == gt)

    n_seq = len(buckets)
    return {
        "k_frames": k_frames,
        "n_seq": n_seq,
        "init_rate": init_ok / max(n_seq, 1),
        "init_correct_rate": init_correct / max(init_ok, 1) if init_ok else 0.0,
        "never_init": never,
        "latency_frames_mean": float(np.mean(latencies_f)) if latencies_f else None,
        "latency_frames_median": float(np.median(latencies_f)) if latencies_f else None,
        "latency_sec_mean": float(np.mean(latencies_s)) if latencies_s else None,
        "latency_sec_median": float(np.median(latencies_s)) if latencies_s else None,
    }


def normalize_x(x: torch.Tensor, mean: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
    return (x - mean.view(1, 1, -1)) / std.view(1, 1, -1)


def build_imuposer_windows(
    cfg: dict,
    window_len: int,
    watch_side: int,
    phone_side: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, List[dict]]:
    src = resolve_path(cfg["data"]["processed_imuposer_file"])
    data = torch.load(src, map_location="cpu")
    w_idx, p_idx = combo_to_indices(watch_side, phone_side)
    xs, yw, yp, metas = [], [], [], []
    for i, (acc, ori) in enumerate(zip(data["acc"], data["ori"])):
        if acc.shape[1] < 4 or ori.shape[1] < 4:
            continue
        acc = acc[:, :4].float()
        ori = ori[:, :4].float()
        feat = make_window_features(
            acc, ori, w_idx, p_idx, cfg["data"]["fps"], cfg["data"]["acc_scale"]
        )
        wins = slide_windows(feat, window_len, cfg["data"]["test_stride"])
        for j, win in enumerate(wins):
            xs.append(win)
            yw.append(watch_side)
            yp.append(phone_side)
            metas.append(
                {
                    "seq": i,
                    "start": j * cfg["data"]["test_stride"],
                    "y_watch": watch_side,
                    "y_phone": phone_side,
                }
            )
    if not xs:
        return (
            torch.zeros(0, window_len, 12),
            torch.zeros(0, dtype=torch.long),
            torch.zeros(0, dtype=torch.long),
            [],
        )
    return (
        torch.stack(xs).float(),
        torch.tensor(yw, dtype=torch.long),
        torch.tensor(yp, dtype=torch.long),
        metas,
    )


def _amass_val_ids(processed_dir: Path, subsets: List[str], val_ratio: float, seed: int):
    files = []
    for name in subsets:
        p = processed_dir / f"{name}.pt"
        if p.exists():
            files.append(p)
    seq_counts = []
    for f in files:
        d = torch.load(f, map_location="cpu")
        seq_counts.append(len(d["acc"]))
    total = sum(seq_counts)
    g = torch.Generator().manual_seed(seed)
    perm = torch.randperm(total, generator=g).tolist()
    n_val = max(1, int(round(total * val_ratio))) if total > 1 else 0
    return files, seq_counts, set(perm[:n_val])


def build_amass_val_windows(
    cfg: dict,
    window_len: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, List[dict]]:
    processed_dir = resolve_path(cfg["data"]["processed_amass_dir"])
    subsets = cfg["data"].get("amass_subsets") or []
    files, seq_counts, val_ids = _amass_val_ids(
        processed_dir,
        subsets,
        cfg["data"]["val_ratio"],
        cfg["experiment"]["seed"],
    )
    xs, yw, yp, metas = [], [], [], []
    offset = 0
    stride = cfg["data"]["train_stride"]
    for f, n_seq in zip(files, seq_counts):
        data = torch.load(f, map_location="cpu")
        for local_i, (acc, ori) in enumerate(zip(data["acc"], data["ori"])):
            global_i = offset + local_i
            if global_i not in val_ids:
                continue
            if acc.shape[1] < 4 or ori.shape[1] < 4:
                continue
            acc = acc[:, :4].float()
            ori = ori[:, :4].float()
            for y_watch in (0, 1):
                for y_phone in (0, 1):
                    w_idx, p_idx = combo_to_indices(y_watch, y_phone)
                    feat = make_window_features(
                        acc,
                        ori,
                        w_idx,
                        p_idx,
                        cfg["data"]["fps"],
                        cfg["data"]["acc_scale"],
                    )
                    wins = slide_windows(feat, window_len, stride)
                    for j, win in enumerate(wins):
                        xs.append(win)
                        yw.append(y_watch)
                        yp.append(y_phone)
                        metas.append(
                            {
                                "source": f.name,
                                "seq_local": local_i,
                                "start": j * stride,
                                "y_watch": y_watch,
                                "y_phone": y_phone,
                            }
                        )
        offset += n_seq
    if not xs:
        return (
            torch.zeros(0, window_len, 12),
            torch.zeros(0, dtype=torch.long),
            torch.zeros(0, dtype=torch.long),
            [],
        )
    return (
        torch.stack(xs).float(),
        torch.tensor(yw, dtype=torch.long),
        torch.tensor(yp, dtype=torch.long),
        metas,
    )


def eval_built_windows(
    model,
    x: torch.Tensor,
    yw: torch.Tensor,
    yp: torch.Tensor,
    meta: List[dict],
    mean: torch.Tensor,
    std: torch.Tensor,
    device: torch.device,
    batch_size: int,
    split: str,
    fps: float,
    k_frames: int,
) -> Dict[str, Any]:
    if x.numel() == 0:
        return {"n": 0, "watch_acc": None, "phone_acc": None, "joint_acc": None}
    xn = normalize_x(x, mean, std)
    pw, pp = predict_batches(model, xn, device, batch_size)
    yw_np, yp_np = yw.numpy(), yp.numpy()
    metrics = {
        "n": int(x.shape[0]),
        "watch_acc": _acc(yw_np, pw),
        "phone_acc": _acc(yp_np, pp),
        "joint_acc": float(((yw_np == pw) & (yp_np == pp)).mean()),
        "cm_watch": confusion_2(yw_np, pw).tolist(),
        "cm_phone": confusion_2(yp_np, pp).tolist(),
        "cm_joint4": confusion_4(yw_np, yp_np, pw, pp).tolist(),
    }
    seq_m = sequence_majority_vote(meta, yw_np, yp_np, pw, pp, split)
    if seq_m:
        metrics.update(seq_m)
    metrics["init_done"] = init_done_stats(
        meta, yw_np, yp_np, pw, pp, split, fps=fps, k_frames=k_frames
    )
    return metrics


def load_model(cfg: dict, ckpt_path: Path, device: torch.device) -> nn.Module:
    ckpt = torch.load(ckpt_path, map_location=device)
    model = PosClassifier(
        n_input=cfg["data"]["input_dim"],
        n_hidden=cfg["model"]["n_hidden"],
        n_lstm_layers=cfg["model"]["n_lstm_layers"],
        bidirectional=cfg["model"]["bidirectional"],
        dropout=cfg["model"]["dropout"],
    ).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    return model


def print_main_table(rows: List[Dict[str, Any]]) -> None:
    header = (
        f"{'Split':<16} {'Watch Acc':>10} {'Phone Acc':>10} "
        f"{'Joint Acc':>10} {'Seq Joint':>10}"
    )
    print("\n=== Main Result Table ===")
    print(header)
    print("-" * len(header))
    for r in rows:
        print(
            f"{r['split']:<16} "
            f"{r['watch_acc']:>10.4f} "
            f"{r['phone_acc']:>10.4f} "
            f"{r['joint_acc']:>10.4f} "
            f"{r.get('seq_joint_acc', float('nan')):>10.4f}"
        )


def evaluate_split(
    split: str,
    cfg: dict,
    model: nn.Module,
    device: torch.device,
    stats: dict,
    watch_side: int,
    phone_side: int,
    k_frames: int,
    do_ablation: bool,
) -> Dict[str, Any]:
    data_dir = resolve_path(cfg["data"]["out_dir"])
    pt = data_dir / ("amass_val.pt" if split == "val" else "imuposer_test.pt")
    if not pt.exists():
        raise FileNotFoundError(pt)

    ds = PosClsDataset(pt, norm_stats=stats, normalize=True)
    loader = DataLoader(
        ds,
        batch_size=cfg["eval"]["batch_size"],
        shuffle=False,
        num_workers=cfg["train"]["num_workers"],
    )
    metrics = run_eval_loader(model, loader, device)
    yw = metrics.pop("y_watch")
    yp = metrics.pop("y_phone")
    pw = metrics.pop("p_watch")
    pp = metrics.pop("p_phone")

    seq_m = sequence_majority_vote(ds.meta, yw, yp, pw, pp, split)
    if seq_m:
        metrics.update(seq_m)

    metrics["init_done"] = init_done_stats(
        ds.meta,
        yw,
        yp,
        pw,
        pp,
        split,
        fps=cfg["data"]["fps"],
        k_frames=k_frames,
    )

    if split == "test":
        raw_dir = resolve_path("data/raw/IMUPoser")
        if raw_dir.exists():
            motion_map = list_imuposer_motions(raw_dir)
            metrics["by_motion"] = motion_type_breakdown(
                ds.meta, yw, yp, pw, pp, motion_map
            )

    if do_ablation:
        mean, std = stats["mean"].float(), stats["std"].float()
        ablation = {}
        for wlen in ABLATION_WINDOW_LENS:
            print(f"[{split}] window ablation len={wlen} ...")
            if split == "test":
                x, yww, ypp, meta = build_imuposer_windows(
                    cfg, wlen, watch_side, phone_side
                )
            else:
                x, yww, ypp, meta = build_amass_val_windows(cfg, wlen)
            ablation[str(wlen)] = eval_built_windows(
                model,
                x,
                yww,
                ypp,
                meta,
                mean,
                std,
                device,
                cfg["eval"]["batch_size"],
                split,
                cfg["data"]["fps"],
                k_frames,
            )
        metrics["window_ablation"] = ablation

    metrics["split"] = "AMASS Val" if split == "val" else "IMUPoser Test"
    metrics["combo"] = (
        f"watch={watch_side}({ 'LW' if watch_side == 0 else 'RW' }), "
        f"phone={phone_side}({ 'LP' if phone_side == 0 else 'RP' })"
    )
    return metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        default="experiments/week1_pos_cls/configs/default.yaml",
    )
    parser.add_argument(
        "--split",
        type=str,
        choices=["val", "test", "both"],
        default="both",
        help="val=AMASS val, test=IMUPoser, both=report main table",
    )
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--watch-side", type=int, choices=[0, 1], default=0)
    parser.add_argument("--phone-side", type=int, choices=[0, 1], default=1)
    parser.add_argument("--k-frames", type=int, default=30, help="init_done stability K")
    parser.add_argument(
        "--no-ablation",
        action="store_true",
        help="skip window-length ablation (faster)",
    )
    args = parser.parse_args()

    cfg = load_config(resolve_path(args.config))
    set_seed(cfg["experiment"]["seed"])
    data_dir = resolve_path(cfg["data"]["out_dir"])
    stats_pt = data_dir / "norm_stats.pt"
    ckpt_path = resolve_path(args.checkpoint or cfg["eval"]["checkpoint"])
    for p in (stats_pt, ckpt_path):
        if not p.exists():
            raise FileNotFoundError(p)

    device = torch.device(
        cfg["train"]["device"] if torch.cuda.is_available() else "cpu"
    )
    stats = load_norm_stats(stats_pt)
    model = load_model(cfg, ckpt_path, device)

    splits = ["val", "test"] if args.split == "both" else [args.split]
    all_metrics = {}
    table_rows = []
    log_dir = resolve_path("experiments/week1_pos_cls/outputs/logs")
    log_dir.mkdir(parents=True, exist_ok=True)

    for split in splits:
        print(f"\n======== Evaluating split={split} ========")
        metrics = evaluate_split(
            split=split,
            cfg=cfg,
            model=model,
            device=device,
            stats=stats,
            watch_side=args.watch_side,
            phone_side=args.phone_side,
            k_frames=args.k_frames,
            do_ablation=not args.no_ablation,
        )
        # serializable copy (drop huge arrays already removed)
        all_metrics[split] = metrics
        table_rows.append(
            {
                "split": metrics["split"],
                "watch_acc": metrics["watch_acc"],
                "phone_acc": metrics["phone_acc"],
                "joint_acc": metrics["joint_acc"],
                "seq_joint_acc": metrics.get("seq_joint_acc", float("nan")),
            }
        )
        out_path = log_dir / f"metrics_{split}.json"
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2)
        print(f"saved {out_path}")

        print(
            f"[{metrics['split']}] "
            f"watch={metrics['watch_acc']:.4f} "
            f"phone={metrics['phone_acc']:.4f} "
            f"joint={metrics['joint_acc']:.4f} "
            f"seq_joint={metrics.get('seq_joint_acc', float('nan')):.4f}"
        )
        init = metrics["init_done"]
        print(
            f"  init_done: rate={init['init_rate']:.4f} "
            f"correct={init['init_correct_rate']:.4f} "
            f"latency_s(mean/med)="
            f"{init['latency_sec_mean']}/{init['latency_sec_median']}"
        )
        if "by_motion" in metrics:
            print("  by_motion (top joint_acc):")
            ranked = sorted(
                metrics["by_motion"].items(),
                key=lambda kv: kv[1]["joint_acc"],
                reverse=True,
            )
            for name, m in ranked[:8]:
                print(
                    f"    {name:<22} n={m['n']:<5} "
                    f"joint={m['joint_acc']:.3f} "
                    f"w={m['watch_acc']:.3f} p={m['phone_acc']:.3f}"
                )
        if "window_ablation" in metrics:
            print("  window_ablation joint_acc:")
            for wlen, m in metrics["window_ablation"].items():
                print(
                    f"    W={wlen:<3} n={m['n']:<7} "
                    f"joint={m['joint_acc']:.4f} "
                    f"seq={m.get('seq_joint_acc', float('nan')):.4f}"
                )

    print_main_table(table_rows)
    summary_path = log_dir / "metrics_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump({"table": table_rows, "details": all_metrics}, f, indent=2)
    print(f"\nsaved summary {summary_path}")


if __name__ == "__main__":
    main()
