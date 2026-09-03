#!/usr/bin/env python3
"""
Evaluate Week3 step-1 position classifier (unknown R_BS).

AMASS val / IMUPoser test reconstruct R_MS = R_MB @ R_BS online
(same protocol as training). IMUPoser recordings are treated as R_MB.

Reports (aligned with Week1):
  - window-level Watch / Phone / Joint accuracy
  - sequence-level Joint accuracy (majority vote)
  - confusion matrices
  - IMUPoser accuracy by motion type
  - window-length ablation: 30 / 60 / 90 / 150
  - init_done latency (K consecutive frames with stable prediction)

Usage (repo root):
  python experiments/week3_pos_rsb/eval.py \
      --config experiments/week3_pos_rsb/configs/default.yaml
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
from torch.utils.data import DataLoader
from tqdm import tqdm

_EXP_DIR = Path(__file__).resolve().parent
if str(_EXP_DIR) not in sys.path:
    sys.path.insert(0, str(_EXP_DIR))

from dataset import (  # noqa: E402
    combo_to_indices,
    load_config,
    load_norm_stats,
    resolve_path,
    set_seed,
)
from dataset.pos_dataset import (  # noqa: E402
    AmassRmsPosDataset,
    ImuposerRmsPosDataset,
    build_amass_index,
    build_imuposer_index,
    load_amass_rms_pack,
    load_imuposer_sequences,
    offset_kwargs_from_cfg,
)
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


def combo_tag(watch_side: int, phone_side: int) -> str:
    return f"{'lw' if watch_side == 0 else 'rw'}_{'lp' if phone_side == 0 else 'rp'}"


def combo_label(watch_side: int, phone_side: int) -> str:
    return f"{'LW' if watch_side == 0 else 'RW'}+{'LP' if phone_side == 0 else 'RP'}"


def amass_meta_from_index(index: torch.Tensor) -> List[dict]:
    out = []
    for seq_i, yw, yp, start in index.tolist():
        out.append(
            {
                "seq": int(seq_i),
                "y_watch": int(yw),
                "y_phone": int(yp),
                "start": int(start),
            }
        )
    return out


def imu_meta_from_index(index: torch.Tensor, y_watch: int, y_phone: int) -> List[dict]:
    out = []
    for seq_i, start in index.tolist():
        out.append(
            {
                "seq": int(seq_i),
                "start": int(start),
                "y_watch": int(y_watch),
                "y_phone": int(y_phone),
            }
        )
    return out


def seq_key_from_meta(m: dict, split: str) -> Tuple:
    if split == "test":
        return ("seq", m["seq"], m.get("y_watch"), m.get("y_phone"))
    return ("amass", m["seq"], m.get("y_watch"), m.get("y_phone"))


@torch.no_grad()
def run_loader(model, loader, device) -> Dict[str, np.ndarray]:
    model.eval()
    yw, yp, pw, pp = [], [], [], []
    ce = nn.CrossEntropyLoss(reduction="sum")
    loss_sum = 0.0
    n = 0
    for x, y_w, y_p in tqdm(loader, desc="eval"):
        x = x.to(device)
        y_w_d = y_w.to(device)
        y_p_d = y_p.to(device)
        lw, lp = model(x)
        loss_sum += (ce(lw, y_w_d) + ce(lp, y_p_d)).item()
        yw.append(y_w.numpy())
        yp.append(y_p.numpy())
        pw.append(lw.argmax(dim=-1).cpu().numpy())
        pp.append(lp.argmax(dim=-1).cpu().numpy())
        n += x.size(0)
    return {
        "yw": np.concatenate(yw) if yw else np.zeros(0, dtype=np.int64),
        "yp": np.concatenate(yp) if yp else np.zeros(0, dtype=np.int64),
        "pw": np.concatenate(pw) if pw else np.zeros(0, dtype=np.int64),
        "pp": np.concatenate(pp) if pp else np.zeros(0, dtype=np.int64),
        "n": int(n),
        "loss": float(loss_sum / max(n, 1)),
    }


def metrics_from_pred(pred: Dict[str, np.ndarray]) -> Dict[str, Any]:
    yw, yp, pw, pp = pred["yw"], pred["yp"], pred["pw"], pred["pp"]
    n = int(pred.get("n", len(yw)))
    out = {
        "n": n,
        "watch_acc": _acc(yw, pw),
        "phone_acc": _acc(yp, pp),
        "joint_acc": float(((yw == pw) & (yp == pp)).mean()) if n else float("nan"),
        "cm_watch": confusion_2(yw, pw).tolist(),
        "cm_phone": confusion_2(yp, pp).tolist(),
        "cm_joint4": confusion_4(yw, yp, pw, pp).tolist(),
    }
    if "loss" in pred:
        out["loss"] = float(pred["loss"])
    return out


def sequence_majority_vote(
    meta: List[dict],
    yw: np.ndarray,
    yp: np.ndarray,
    pw: np.ndarray,
    pp: np.ndarray,
    split: str,
) -> Dict[str, Any]:
    buckets: Dict[Tuple, List[int]] = defaultdict(list)
    for i, m in enumerate(meta):
        buckets[seq_key_from_meta(m, split)].append(i)
    correct = 0
    for idxs in buckets.values():
        gts = [int(yw[i]) * 2 + int(yp[i]) for i in idxs]
        preds = [int(pw[i]) * 2 + int(pp[i]) for i in idxs]
        gt = max(set(gts), key=gts.count)
        pr = max(set(preds), key=preds.count)
        correct += int(pr == gt)
    n_seq = len(buckets)
    return {"seq_joint_acc": correct / max(n_seq, 1), "n_seq": n_seq}


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
        name = motion_map[seq]["motion"] if 0 <= seq < len(motion_map) else "unknown"
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
    buckets: Dict[Tuple, List[int]] = defaultdict(list)
    for i, m in enumerate(meta):
        buckets[seq_key_from_meta(m, split)].append(i)

    latencies_s, latencies_f = [], []
    init_ok = init_correct = never = 0
    for idxs in buckets.values():
        idxs = sorted(idxs, key=lambda i: int(meta[i].get("start", 0)))
        starts = [int(meta[i].get("start", 0)) for i in idxs]
        stride = max(1, starts[1] - starts[0]) if len(starts) >= 2 else 15
        gt = int(yw[idxs[0]]) * 2 + int(yp[idxs[0]])
        last_end = starts[-1] + stride
        frame_pred = np.full(last_end, -1, dtype=np.int64)
        for i, st in zip(idxs, starts):
            frame_pred[st : st + stride] = int(pw[i]) * 2 + int(pp[i])
        done_at = None
        stable_pred = None
        for t in range(k_frames - 1, last_end):
            seg = frame_pred[t - k_frames + 1 : t + 1]
            if (seg >= 0).all() and (seg == seg[0]).all():
                done_at = t + 1
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


def attach_seq_stats(
    metrics: Dict[str, Any],
    meta: List[dict],
    pred: Dict[str, np.ndarray],
    split: str,
    fps: float,
    k_frames: int,
) -> Dict[str, Any]:
    yw, yp, pw, pp = pred["yw"], pred["yp"], pred["pw"], pred["pp"]
    metrics.update(sequence_majority_vote(meta, yw, yp, pw, pp, split))
    metrics["init_done"] = init_done_stats(
        meta, yw, yp, pw, pp, split, fps=fps, k_frames=k_frames
    )
    return metrics


def cheap_combo_balance(index: torch.Tensor) -> Dict[str, Any]:
    cnt = Counter()
    for row in index.tolist():
        yw, yp = int(row[1]), int(row[2])
        cnt[combo_label(yw, yp)] += 1
    n = max(int(index.shape[0]), 1)
    return {k: {"n": v, "frac": v / n} for k, v in sorted(cnt.items())}


def make_loader(ds, batch_size: int, num_workers: int) -> DataLoader:
    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )


def eval_amass_windows(
    model,
    sequences,
    index: torch.Tensor,
    cfg: dict,
    stats: dict,
    device,
    batch_size: int,
    num_workers: int,
    window_len: int,
    k_frames: int,
) -> Dict[str, Any]:
    ds = AmassRmsPosDataset(
        sequences,
        index,
        window_len=window_len,
        acc_scale=cfg["data"]["acc_scale"],
        **offset_kwargs_from_cfg(cfg),
        seed=cfg["experiment"]["seed"],
        mean=stats["mean"],
        std=stats["std"],
    )
    pred = run_loader(model, make_loader(ds, batch_size, num_workers), device)
    metrics = metrics_from_pred(pred)
    meta = amass_meta_from_index(index)
    attach_seq_stats(metrics, meta, pred, "val", cfg["data"]["fps"], k_frames)
    return metrics, pred, meta


def eval_imuposer_windows(
    model,
    sequences,
    index: torch.Tensor,
    cfg: dict,
    stats: dict,
    device,
    batch_size: int,
    num_workers: int,
    window_len: int,
    k_frames: int,
    y_watch: int,
    y_phone: int,
) -> Dict[str, Any]:
    ds = ImuposerRmsPosDataset(
        sequences,
        index,
        window_len=window_len,
        acc_scale=cfg["data"]["acc_scale"],
        **offset_kwargs_from_cfg(cfg),
        seed=cfg["experiment"]["seed"],
        y_watch=y_watch,
        y_phone=y_phone,
        mean=stats["mean"],
        std=stats["std"],
    )
    pred = run_loader(model, make_loader(ds, batch_size, num_workers), device)
    metrics = metrics_from_pred(pred)
    meta = imu_meta_from_index(index, y_watch, y_phone)
    attach_seq_stats(metrics, meta, pred, "test", cfg["data"]["fps"], k_frames)
    return metrics, pred, meta


def window_ablation_amass(
    model, pack, val_ids, cfg, stats, device, batch_size, num_workers, k_frames
) -> Dict[str, Any]:
    out = {}
    stride = int(cfg["data"]["train_stride"])
    for wlen in ABLATION_WINDOW_LENS:
        print(f"[val] window ablation len={wlen} ...")
        _, val_index = build_amass_index(pack["sequences"], wlen, stride, val_ids)
        metrics, _, _ = eval_amass_windows(
            model,
            pack["sequences"],
            val_index,
            cfg,
            stats,
            device,
            batch_size,
            num_workers,
            wlen,
            k_frames,
        )
        out[str(wlen)] = metrics
    return out


def window_ablation_imuposer(
    model,
    sequences,
    cfg,
    stats,
    device,
    batch_size,
    num_workers,
    k_frames,
    y_watch,
    y_phone,
) -> Dict[str, Any]:
    out = {}
    stride = int(cfg["data"]["test_stride"])
    for wlen in ABLATION_WINDOW_LENS:
        print(f"[test {combo_label(y_watch, y_phone)}] window ablation len={wlen} ...")
        index = build_imuposer_index(sequences, wlen, stride)
        metrics, _, _ = eval_imuposer_windows(
            model,
            sequences,
            index,
            cfg,
            stats,
            device,
            batch_size,
            num_workers,
            wlen,
            k_frames,
            y_watch,
            y_phone,
        )
        out[str(wlen)] = metrics
    return out


def _print_split_details(metrics: Dict[str, Any]) -> None:
    print(
        f"[{metrics['split']}] "
        f"watch={metrics['watch_acc']:.4f} "
        f"phone={metrics['phone_acc']:.4f} "
        f"joint={metrics['joint_acc']:.4f} "
        f"seq_joint={metrics.get('seq_joint_acc', float('nan')):.4f}"
    )
    init = metrics.get("init_done")
    if init:
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


def print_main_table(rows: List[Dict[str, Any]]) -> None:
    header = (
        f"{'Split':<18} {'Combo':<8} {'Watch Acc':>10} {'Phone Acc':>10} "
        f"{'Joint Acc':>10} {'Seq Joint':>10}"
    )
    print("\n=== Main Result Table ===")
    print(header)
    print("-" * len(header))
    for r in rows:
        print(
            f"{r['split']:<18} "
            f"{str(r.get('combo') or '-'):<8} "
            f"{r['watch_acc']:>10.4f} "
            f"{r['phone_acc']:>10.4f} "
            f"{r['joint_acc']:>10.4f} "
            f"{r.get('seq_joint_acc', float('nan')):>10.4f}"
        )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        default="experiments/week3_pos_rsb/configs/default.yaml",
    )
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--split", choices=["val", "test", "both"], default="both")
    parser.add_argument("--combos", type=str, default="lw_lp,lw_rp,rw_lp,rw_rp")
    parser.add_argument("--k-frames", type=int, default=30)
    parser.add_argument(
        "--no-ablation",
        action="store_true",
        help="skip window-length ablation (faster)",
    )
    args = parser.parse_args()

    cfg = load_config(resolve_path(args.config))
    set_seed(cfg["experiment"]["seed"])
    device = torch.device(
        cfg["train"]["device"] if torch.cuda.is_available() else "cpu"
    )
    stats_pt = resolve_path(cfg["data"]["out_dir"]) / "norm_stats.pt"
    idx_pt = resolve_path(cfg["data"]["out_dir"]) / "amass_index.pt"
    ckpt_path = resolve_path(args.checkpoint or cfg["eval"]["checkpoint"])
    for p in (stats_pt, ckpt_path):
        if not p.exists():
            raise FileNotFoundError(p)
    stats = load_norm_stats(stats_pt)

    model = PosClassifier(
        n_input=cfg["data"]["input_dim"],
        n_hidden=cfg["model"]["n_hidden"],
        n_lstm_layers=cfg["model"]["n_lstm_layers"],
        bidirectional=cfg["model"]["bidirectional"],
        dropout=0.0,
    ).to(device)
    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    print(
        f"checkpoint={ckpt_path} epoch={ckpt.get('epoch')} "
        f"train_joint={((ckpt.get('metrics') or {}).get('joint_acc'))}"
    )

    log_dir = resolve_path(cfg["eval"]["log_dir"])
    log_dir.mkdir(parents=True, exist_ok=True)
    batch_size = int(cfg["eval"]["batch_size"])
    num_workers = int(cfg["train"]["num_workers"])
    k_frames = int(args.k_frames)
    do_ablation = not args.no_ablation

    protocol = (
        "Input a_M+R_MS; R_MS=R_MB@R_BS independent per device; "
        "a_M unchanged; R_BS unknown / window-constant"
    )
    all_metrics: Dict[str, Any] = {"protocol": protocol}
    table_rows: List[Dict[str, Any]] = []

    pack = None
    val_ids = None
    if args.split in ("val", "both") or do_ablation:
        pack = load_amass_rms_pack(cfg)
        if idx_pt.exists():
            payload = torch.load(idx_pt, map_location="cpu")
            val_ids = set(int(x) for x in payload.get("val_ids", []))

    if args.split in ("val", "both"):
        print(f"\n======== Evaluating split=val combo=4-mix ========")
        metrics, pred, meta = eval_amass_windows(
            model,
            pack["sequences"],
            pack["val_index"],
            cfg,
            stats,
            device,
            batch_size,
            num_workers,
            int(cfg["data"]["window_len"]),
            k_frames,
        )
        metrics["split"] = "AMASS Val"
        metrics["combo"] = "4 组合混合"
        metrics["cheap_checks"] = {
            "amass_train_combo": cheap_combo_balance(pack["train_index"]),
            "amass_val_combo": cheap_combo_balance(pack["val_index"]),
        }
        if do_ablation and val_ids is not None:
            metrics["window_ablation"] = window_ablation_amass(
                model,
                pack,
                val_ids,
                cfg,
                stats,
                device,
                batch_size,
                num_workers,
                k_frames,
            )
        all_metrics["val"] = metrics
        table_rows.append(
            {
                "split": metrics["split"],
                "combo": metrics["combo"],
                "watch_acc": metrics["watch_acc"],
                "phone_acc": metrics["phone_acc"],
                "joint_acc": metrics["joint_acc"],
                "seq_joint_acc": metrics.get("seq_joint_acc", float("nan")),
            }
        )
        out_path = log_dir / "metrics_val.json"
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2)
        print(f"saved {out_path}")
        _print_split_details(metrics)

    if args.split in ("test", "both"):
        imu_seqs = load_imuposer_sequences(
            resolve_path(cfg["data"]["processed_imuposer_file"])
        )
        imu_index = build_imuposer_index(
            imu_seqs, cfg["data"]["window_len"], cfg["data"]["test_stride"]
        )
        combo_list = []
        for tok in args.combos.split(","):
            tok = tok.strip().lower().replace("+", "_").replace("-", "_")
            w, p = tok.split("_")
            combo_list.append((0 if w == "lw" else 1, 0 if p == "lp" else 1))
        motion_map = []
        raw_dir = resolve_path("data/raw/IMUPoser")
        if raw_dir.exists():
            motion_map = list_imuposer_motions(raw_dir)
        all_metrics["test"] = {}
        for yw, yp in combo_list:
            tag = combo_tag(yw, yp)
            label = combo_label(yw, yp)
            print(f"\n======== Evaluating split=test combo={label} ========")
            print(f"  windows={imu_index.shape[0]} slots={combo_to_indices(yw, yp)}")
            metrics, pred, meta = eval_imuposer_windows(
                model,
                imu_seqs,
                imu_index,
                cfg,
                stats,
                device,
                batch_size,
                num_workers,
                int(cfg["data"]["window_len"]),
                k_frames,
                yw,
                yp,
            )
            metrics["split"] = f"IMUPoser {label}"
            metrics["combo"] = label
            metrics["combo_tag"] = tag
            if motion_map:
                metrics["by_motion"] = motion_type_breakdown(
                    meta, pred["yw"], pred["yp"], pred["pw"], pred["pp"], motion_map
                )
            if do_ablation:
                metrics["window_ablation"] = window_ablation_imuposer(
                    model,
                    imu_seqs,
                    cfg,
                    stats,
                    device,
                    batch_size,
                    num_workers,
                    k_frames,
                    yw,
                    yp,
                )
            all_metrics["test"][tag] = metrics
            table_rows.append(
                {
                    "split": metrics["split"],
                    "combo": metrics["combo"],
                    "watch_acc": metrics["watch_acc"],
                    "phone_acc": metrics["phone_acc"],
                    "joint_acc": metrics["joint_acc"],
                    "seq_joint_acc": metrics.get("seq_joint_acc", float("nan")),
                }
            )
            out_path = log_dir / f"metrics_test_{tag}.json"
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(metrics, f, indent=2)
            if tag == "lw_rp":
                legacy = log_dir / "metrics_test.json"
                with open(legacy, "w", encoding="utf-8") as f:
                    json.dump(metrics, f, indent=2)
                print(f"saved {out_path} (and {legacy})")
            else:
                print(f"saved {out_path}")
            _print_split_details(metrics)

    print_main_table(table_rows)
    summary = {"protocol": protocol, "table": table_rows, "details": all_metrics}
    summary_path = log_dir / "metrics_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    step1_path = log_dir / "metrics_step1.json"
    with open(step1_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"\nsaved summary {summary_path}")
    print(f"saved {step1_path}")


if __name__ == "__main__":
    main()
