#!/usr/bin/env python3
"""
D6-A: After R_SB calibration, does Week1 position classification recover?

Evaluates on two sources (same inject / None / Learned / Oracle protocol):
  1) AMASS (synthetic bone-aligned IMU)
  2) IMUPoser (real recorded streams treated as clean reference; inject R_SB)

For each watch/phone combo window:
  - sample independent R_SB for watch & phone
  - corrupt IMU
  - build Week1 features under:
      None:    no calib
      Learned: Week2 predicts R_SB, then calib
      Oracle:  GT R_SB calib
  - run Week1 PosClassifier; report Watch/Phone/Joint accuracy

Usage (repo root; needs Week1 + Week2 checkpoints):
  # single-device Week2 ×2
  python experiments/week2_rot_ext/eval_downstream_week1.py \\
      --config experiments/week2_rot_ext/configs/default.yaml

  # dual-joint Week2
  python experiments/week2_rot_ext/eval_downstream_week1.py --dual
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
from models import RotExtrinsicDualNet, RotExtrinsicNet  # noqa: E402

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


def _acc(t: Dict[str, int]) -> Dict[str, float]:
    n = max(t["n"], 1)
    return {
        "n": t["n"],
        "watch_acc": t["watch"] / n,
        "phone_acc": t["phone"] / n,
        "joint_acc": t["joint"] / n,
    }


def _empty_tallies() -> Dict[str, Dict[str, int]]:
    return {
        k: {"n": 0, "watch": 0, "phone": 0, "joint": 0}
        for k in ("none", "learned", "oracle")
    }


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
def predict_rsb_dual(
    model: RotExtrinsicDualNet,
    aw: torch.Tensor,
    ow: torch.Tensor,
    ap: torch.Tensor,
    op: torch.Tensor,
    slot_w: int,
    slot_p: int,
    acc_scale: float,
    mean: torch.Tensor,
    std: torch.Tensor,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor]:
    fw = make_device_features(aw, ow, acc_scale)
    fp = make_device_features(ap, op, acc_scale)
    feat = torch.cat([fw, fp], dim=-1)
    x = _norm(feat, mean, std).unsqueeze(0).to(device)
    sw = torch.tensor([slot_w], device=device)
    sp = torch.tensor([slot_p], device=device)
    (_, rw), (_, rp) = model(x, sw, sp)
    return rw[0].cpu(), rp[0].cpu()


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


@torch.no_grad()
def eval_sequences(
    seq_acc_ori: List[Tuple[torch.Tensor, torch.Tensor]],
    *,
    rot_net: torch.nn.Module,
    clf: PosClassifier,
    use_dual: bool,
    combos: List[Tuple[int, int]],
    window_len: int,
    stride: int,
    fps: float,
    acc_scale: float,
    offset_range: float,
    gen: torch.Generator,
    w2_mean: torch.Tensor,
    w2_std: torch.Tensor,
    w1_mean: torch.Tensor,
    w1_std: torch.Tensor,
    device: torch.device,
    desc: str,
) -> Tuple[Dict[str, Dict[str, int]], int]:
    """
    seq_acc_ori: list of (acc [T,4+,3], ori [T,4+,3,3]) with at least 4 slots.
    Returns tallies and number of sequences used.
    """
    tallies = _empty_tallies()
    n_used = 0

    for acc_all, ori_all in tqdm(seq_acc_ori, desc=desc):
        if acc_all.shape[1] < 4 or ori_all.shape[1] < 4:
            continue
        acc_all = acc_all[:, :4].float()
        ori_all = ori_all[:, :4].float()
        t_len = acc_all.shape[0]
        if t_len < window_len:
            continue
        n_used += 1

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

                feat_none = make_week1_dual_features(
                    aw_o, ow_o, ap_o, op_o, fps, acc_scale
                )
                aw_or, ow_or = calibrate_with_rsb(aw_o, ow_o, r_w)
                ap_or, op_or = calibrate_with_rsb(ap_o, op_o, r_p)
                feat_oracle = make_week1_dual_features(
                    aw_or, ow_or, ap_or, op_or, fps, acc_scale
                )

                if use_dual:
                    r_w_hat, r_p_hat = predict_rsb_dual(
                        rot_net,
                        aw_o,
                        ow_o,
                        ap_o,
                        op_o,
                        w_idx,
                        p_idx,
                        acc_scale,
                        w2_mean,
                        w2_std,
                        device,
                    )
                else:
                    r_w_hat = predict_rsb(
                        rot_net,
                        aw_o,
                        ow_o,
                        w_idx,
                        acc_scale,
                        w2_mean,
                        w2_std,
                        device,
                    )
                    r_p_hat = predict_rsb(
                        rot_net,
                        ap_o,
                        op_o,
                        p_idx,
                        acc_scale,
                        w2_mean,
                        w2_std,
                        device,
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

    return tallies, n_used


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        default="experiments/week2_rot_ext/configs/default.yaml",
    )
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument(
        "--max-seqs",
        type=int,
        default=None,
        help="Override AMASS sequence cap (default: eval.downstream_max_seqs)",
    )
    parser.add_argument(
        "--imuposer-max-seqs",
        type=int,
        default=None,
        help="Override IMUPoser sequence cap (default: eval.imuposer_max_seqs)",
    )
    parser.add_argument(
        "--dual",
        action="store_true",
        help="Use jointly-trained dual-device Week2 model (best_rot_err_dual.pt)",
    )
    parser.add_argument(
        "--skip-amass",
        action="store_true",
        help="Only run IMUPoser downstream eval",
    )
    parser.add_argument(
        "--skip-imuposer",
        action="store_true",
        help="Only run AMASS downstream eval",
    )
    args = parser.parse_args()

    cfg = load_config(resolve_path(args.config))
    set_seed(cfg["experiment"]["seed"])

    use_dual = bool(args.dual)
    if use_dual:
        default_ckpt = (
            "experiments/week2_rot_ext/outputs/checkpoints/best_rot_err_dual.pt"
        )
        w2_ckpt = resolve_path(args.checkpoint or default_ckpt)
        w2_stats = resolve_path(cfg["data"]["out_dir"]) / "norm_stats_dual.pt"
    else:
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

    if use_dual:
        rot_net = RotExtrinsicDualNet(
            feat_dim=24,
            n_slots=cfg["data"]["n_slots"],
            n_hidden=cfg["model"]["n_hidden"],
            n_lstm_layers=cfg["model"]["n_lstm_layers"],
            bidirectional=cfg["model"]["bidirectional"],
            dropout=cfg["model"]["dropout"],
            use_slot_onehot=cfg["model"]["use_slot_onehot"],
        ).to(device)
    else:
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

    window_len = cfg["data"]["window_len"]
    fps = float(cfg["data"]["fps"])
    acc_scale = float(cfg["data"]["acc_scale"])
    offset_range = float(cfg["data"]["offset_range_deg"])
    combos = [tuple(c) for c in cfg["eval"]["downstream_combos"]]

    common_kw = dict(
        rot_net=rot_net,
        clf=clf,
        use_dual=use_dual,
        combos=combos,
        window_len=window_len,
        fps=fps,
        acc_scale=acc_scale,
        offset_range=offset_range,
        w2_mean=w2_mean,
        w2_std=w2_std,
        w1_mean=w1_mean,
        w1_std=w1_std,
        device=device,
    )

    metrics: Dict = {
        "protocol": (
            "Inject independent watch/phone R_SB → calib → Week1 PosClassifier "
            "(AMASS synthetic + IMUPoser real streams as clean reference)."
        ),
        "week2_mode": "dual_joint" if use_dual else "single_device_x2",
        "offset_range_deg": offset_range,
        "expectation": "none << learned <= oracle (joint_acc)",
        "datasets": {},
    }

    # --- AMASS ---
    if not args.skip_amass:
        gen_amass = torch.Generator().manual_seed(cfg["experiment"]["seed"] + 11)
        processed_dir = resolve_path(cfg["data"]["processed_amass_dir"])
        files = _list_amass_files(processed_dir, cfg["data"].get("amass_subsets") or [])
        if not files:
            raise FileNotFoundError(f"No AMASS under {processed_dir}")
        max_seqs = args.max_seqs or int(cfg["eval"].get("downstream_max_seqs", 30))
        stride = int(cfg["data"]["train_stride"])

        seqs: List[Tuple[torch.Tensor, torch.Tensor]] = []
        for fpath in files:
            if len(seqs) >= max_seqs:
                break
            data = torch.load(fpath, map_location="cpu")
            for acc_all, ori_all in zip(data["acc"], data["ori"]):
                if len(seqs) >= max_seqs:
                    break
                if acc_all.shape[1] < 4 or ori_all.shape[1] < 4:
                    continue
                if acc_all.shape[0] < window_len:
                    continue
                seqs.append((acc_all, ori_all))

        tallies, n_used = eval_sequences(
            seqs,
            stride=stride,
            gen=gen_amass,
            desc="D6 AMASS",
            **common_kw,
        )
        metrics["datasets"]["amass"] = {
            "source": "AMASS bone-aligned IMU + injected R_SB",
            "n_sequences_used": n_used,
            "stride": stride,
            "results": {k: _acc(v) for k, v in tallies.items()},
        }

    # --- IMUPoser (real streams as clean reference) ---
    if not args.skip_imuposer:
        gen_imu = torch.Generator().manual_seed(cfg["experiment"]["seed"] + 13)
        src = resolve_path(cfg["data"]["processed_imuposer_file"])
        if not src.exists():
            raise FileNotFoundError(src)
        max_seqs_imu = args.imuposer_max_seqs or int(
            cfg["eval"].get("imuposer_max_seqs", 40)
        )
        stride_imu = int(cfg["data"]["test_stride"])

        data = torch.load(src, map_location="cpu")
        n_take = min(len(data["acc"]), max_seqs_imu)
        seqs_imu = [
            (data["acc"][i], data["ori"][i]) for i in range(n_take)
        ]

        tallies_imu, n_used_imu = eval_sequences(
            seqs_imu,
            stride=stride_imu,
            gen=gen_imu,
            desc="D6 IMUPoser",
            **common_kw,
        )
        metrics["datasets"]["imuposer"] = {
            "source": (
                "IMUPoser recorded streams as clean reference; "
                "inject known R_SB (same as D5); then Week1"
            ),
            "n_sequences_used": n_used_imu,
            "stride": stride_imu,
            "results": {k: _acc(v) for k, v in tallies_imu.items()},
        }

    log_dir = resolve_path("experiments/week2_rot_ext/outputs/logs")
    log_dir.mkdir(parents=True, exist_ok=True)
    out_path = log_dir / (
        "metrics_downstream_week1_dual.json"
        if use_dual
        else "metrics_downstream_week1.json"
    )
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    print(json.dumps(metrics, indent=2))
    print(f"saved {out_path}")


if __name__ == "__main__":
    main()
