#!/usr/bin/env python3
"""
D6-A: After R_SB calibration, does Week1 position classification recover?

On AMASS (bone-aligned IMU):
  For each watch/phone combo window:
    - sample independent R_SB for watch & phone
    - corrupt IMU
    - build Week1 features under three settings:
        None:    no calib
        Learned: Week2 net predicts each device R_SB, then calib
        Oracle:  GT R_SB calib
    - run Week1 PosClassifier; report Watch/Phone/Joint accuracy

Usage (repo root; needs Week1 + Week2 checkpoints):
  python experiments/week2_rot_ext/eval_downstream_week1.py \\
      --config experiments/week2_rot_ext/configs/default.yaml
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import torch
from tqdm import tqdm

_EXP_DIR = Path(__file__).resolve().parent
_WEEK1_DIR = _EXP_DIR.parent / "week1_pos_cls"
if str(_EXP_DIR) not in sys.path:
    sys.path.insert(0, str(_EXP_DIR))

from dataset import (  # noqa: E402  week2
    apply_mount_offset,
    calibrate_with_rsb,
    combo_to_indices,
    load_config,
    make_device_features,
    make_week1_dual_features,
    resolve_path,
    sample_random_offsets,
    set_seed,
)
from models import RotExtrinsicNet  # noqa: E402

# Avoid clashing with week2's `models` package name: load Week1 classifier by path.
import importlib.util

_w1_spec = importlib.util.spec_from_file_location(
    "week1_pos_classifier",
    _WEEK1_DIR / "models" / "pos_classifier.py",
)
_w1_mod = importlib.util.module_from_spec(_w1_spec)
assert _w1_spec.loader is not None
_w1_spec.loader.exec_module(_w1_mod)
PosClassifier = _w1_mod.PosClassifier


def _list_amass_files(processed_dir: Path, subsets: List[str]) -> List[Path]:
    if subsets:
        files = []
        for name in subsets:
            p = processed_dir / f"{name}.pt"
            if p.exists():
                files.append(p)
            else:
                print(f"[warn] missing subset: {p}")
        return files
    return sorted(processed_dir.glob("*.pt"))


def _norm(x: torch.Tensor, mean: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
    return (x - mean) / std


@torch.no_grad()
def predict_rsb(
    model: RotExtrinsicNet,
    acc: torch.Tensor,
    ori: torch.Tensor,
    slot: int,
    acc_scale: float,
    mean: torch.Tensor,
    std: torch.Tensor,
    device: torch.device,
) -> torch.Tensor:
    feat = make_device_features(acc, ori, acc_scale)
    x = _norm(feat, mean, std).unsqueeze(0).to(device)
    slot_t = torch.tensor([slot], device=device)
    _, r = model(x, slot_t)
    return r[0].cpu()


@torch.no_grad()
def classify_week1(
    clf: PosClassifier,
    feat: torch.Tensor,
    mean: torch.Tensor,
    std: torch.Tensor,
    device: torch.device,
) -> Tuple[int, int]:
    x = _norm(feat, mean, std).unsqueeze(0).to(device)
    lw, lp = clf(x)
    return int(lw.argmax(-1).item()), int(lp.argmax(-1).item())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        default="experiments/week2_rot_ext/configs/default.yaml",
    )
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--max-seqs", type=int, default=None)
    args = parser.parse_args()

    cfg = load_config(resolve_path(args.config))
    set_seed(cfg["experiment"]["seed"])
    gen = torch.Generator().manual_seed(cfg["experiment"]["seed"] + 11)

    w2_ckpt = resolve_path(args.checkpoint or cfg["eval"]["checkpoint"])
    w2_stats = resolve_path(cfg["data"]["out_dir"]) / "norm_stats.pt"
    w1_ckpt = resolve_path(cfg["eval"]["week1_checkpoint"])
    w1_stats = resolve_path(cfg["eval"]["week1_norm_stats"])
    for p in (w2_ckpt, w2_stats, w1_ckpt, w1_stats):
        if not p.exists():
            raise FileNotFoundError(
                f"Missing {p}. Train Week1/Week2 and build datasets first."
            )

    device = torch.device(
        cfg["train"]["device"] if torch.cuda.is_available() else "cpu"
    )

    w2_mean = torch.load(w2_stats, map_location="cpu")["mean"].float().view(1, -1)
    w2_std = torch.load(w2_stats, map_location="cpu")["std"].float().view(1, -1)
    w1_st = torch.load(w1_stats, map_location="cpu")
    w1_mean = w1_st["mean"].float().view(1, -1)
    w1_std = w1_st["std"].float().view(1, -1)

    rot_net = RotExtrinsicNet(
        feat_dim=cfg["data"]["feat_dim"],
        n_slots=cfg["data"]["n_slots"],
        n_hidden=cfg["model"]["n_hidden"],
        n_lstm_layers=cfg["model"]["n_lstm_layers"],
        bidirectional=cfg["model"]["bidirectional"],
        dropout=cfg["model"]["dropout"],
        use_slot_onehot=cfg["model"]["use_slot_onehot"],
    ).to(device)
    rot_net.load_state_dict(torch.load(w2_ckpt, map_location=device)["model"])
    rot_net.eval()

    # Week1 cfg for architecture
    w1_cfg = load_config(resolve_path(cfg["eval"]["week1_config"]))
    clf = PosClassifier(
        n_input=w1_cfg["data"]["input_dim"],
        n_hidden=w1_cfg["model"]["n_hidden"],
        n_lstm_layers=w1_cfg["model"]["n_lstm_layers"],
        bidirectional=w1_cfg["model"]["bidirectional"],
        dropout=w1_cfg["model"]["dropout"],
    ).to(device)
    clf.load_state_dict(torch.load(w1_ckpt, map_location=device)["model"])
    clf.eval()

    processed_dir = resolve_path(cfg["data"]["processed_amass_dir"])
    files = _list_amass_files(processed_dir, cfg["data"].get("amass_subsets") or [])
    if not files:
        raise FileNotFoundError(f"No AMASS under {processed_dir}")

    window_len = cfg["data"]["window_len"]
    stride = cfg["data"]["train_stride"]
    fps = float(cfg["data"]["fps"])
    acc_scale = float(cfg["data"]["acc_scale"])
    offset_range = float(cfg["data"]["offset_range_deg"])
    combos = [tuple(c) for c in cfg["eval"]["downstream_combos"]]
    max_seqs = args.max_seqs or int(cfg["eval"].get("downstream_max_seqs", 30))

    tallies: Dict[str, Dict[str, int]] = {
        k: {"n": 0, "watch": 0, "phone": 0, "joint": 0}
        for k in ("none", "learned", "oracle")
    }

    seq_seen = 0
    for fpath in files:
        if seq_seen >= max_seqs:
            break
        data = torch.load(fpath, map_location="cpu")
        for acc_all, ori_all in zip(data["acc"], data["ori"]):
            if seq_seen >= max_seqs:
                break
            if acc_all.shape[1] < 4 or ori_all.shape[1] < 4:
                continue
            acc_all = acc_all[:, :4].float()
            ori_all = ori_all[:, :4].float()
            t_len = acc_all.shape[0]
            if t_len < window_len:
                continue
            seq_seen += 1

            for y_watch, y_phone in combos:
                w_idx, p_idx = combo_to_indices(y_watch, y_phone)
                for start in range(0, t_len - window_len + 1, stride):
                    aw = acc_all[start : start + window_len, w_idx]
                    ow = ori_all[start : start + window_len, w_idx]
                    ap = acc_all[start : start + window_len, p_idx]
                    op = ori_all[start : start + window_len, p_idx]

                    r_w = sample_random_offsets(1, offset_range, generator=gen)[0]
                    r_p = sample_random_offsets(1, offset_range, generator=gen)[0]
                    aw_o, ow_o = apply_mount_offset(aw, ow, r_w)
                    ap_o, op_o = apply_mount_offset(ap, op, r_p)

                    # None
                    feat_none = make_week1_dual_features(
                        aw_o, ow_o, ap_o, op_o, fps, acc_scale
                    )
                    # Oracle
                    aw_or, ow_or = calibrate_with_rsb(aw_o, ow_o, r_w)
                    ap_or, op_or = calibrate_with_rsb(ap_o, op_o, r_p)
                    feat_oracle = make_week1_dual_features(
                        aw_or, ow_or, ap_or, op_or, fps, acc_scale
                    )
                    # Learned
                    r_w_hat = predict_rsb(
                        rot_net, aw_o, ow_o, w_idx, acc_scale, w2_mean, w2_std, device
                    )
                    r_p_hat = predict_rsb(
                        rot_net, ap_o, op_o, p_idx, acc_scale, w2_mean, w2_std, device
                    )
                    aw_l, ow_l = calibrate_with_rsb(aw_o, ow_o, r_w_hat)
                    ap_l, op_l = calibrate_with_rsb(ap_o, op_o, r_p_hat)
                    feat_learned = make_week1_dual_features(
                        aw_l, ow_l, ap_l, op_l, fps, acc_scale
                    )

                    for name, feat in (
                        ("none", feat_none),
                        ("learned", feat_learned),
                        ("oracle", feat_oracle),
                    ):
                        pw, pp = classify_week1(clf, feat, w1_mean, w1_std, device)
                        tallies[name]["n"] += 1
                        tallies[name]["watch"] += int(pw == y_watch)
                        tallies[name]["phone"] += int(pp == y_phone)
                        tallies[name]["joint"] += int(pw == y_watch and pp == y_phone)

    def _acc(t):
        n = max(t["n"], 1)
        return {
            "n": t["n"],
            "watch_acc": t["watch"] / n,
            "phone_acc": t["phone"] / n,
            "joint_acc": t["joint"] / n,
        }

    metrics = {
        "protocol": "AMASS + injected dual-device R_SB → calib → Week1 PosClassifier",
        "n_sequences_used": seq_seen,
        "offset_range_deg": offset_range,
        "results": {k: _acc(v) for k, v in tallies.items()},
        "expectation": "none << learned <= oracle (joint_acc)",
    }

    log_dir = resolve_path("experiments/week2_rot_ext/outputs/logs")
    log_dir.mkdir(parents=True, exist_ok=True)
    out_path = log_dir / "metrics_downstream_week1.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    print(json.dumps(metrics, indent=2))
    print(f"saved {out_path}")


if __name__ == "__main__":
    main()
