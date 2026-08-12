#!/usr/bin/env python3
"""
Aggregate Week2 quantitative logs into one summary (like Week1 metrics_summary).

Reads (if present):
  outputs/logs/metrics_val.json
  outputs/logs/metrics_dual_val.json
  outputs/logs/metrics_imuposer_d5.json
  outputs/logs/metrics_imuposer_d5_dual.json
  outputs/logs/metrics_downstream_week1.json
  outputs/logs/metrics_downstream_week1_dual.json

Writes:
  outputs/logs/metrics_summary.json

Usage (repo root):
  python experiments/week2_rot_ext/summarize_metrics.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict, Optional

_EXP_DIR = Path(__file__).resolve().parent
if str(_EXP_DIR) not in sys.path:
    sys.path.insert(0, str(_EXP_DIR))

from dataset import load_config, resolve_path  # noqa: E402


def _load(path: Path) -> Optional[Dict[str, Any]]:
    if not path.exists():
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _get(d: Any, *keys, default=None):
    cur = d
    for k in keys:
        if not isinstance(cur, dict) or k not in cur:
            return default
        cur = cur[k]
    return cur


def _round(x, nd=3):
    if x is None:
        return None
    try:
        return round(float(x), nd)
    except (TypeError, ValueError):
        return None


def build_summary(log_dir: Path) -> Dict[str, Any]:
    val = _load(log_dir / "metrics_val.json")
    dval = _load(log_dir / "metrics_dual_val.json")
    d5 = _load(log_dir / "metrics_imuposer_d5.json")
    d5d = _load(log_dir / "metrics_imuposer_d5_dual.json")
    d6 = _load(log_dir / "metrics_downstream_week1.json")
    d6d = _load(log_dir / "metrics_downstream_week1_dual.json")

    # Headline rows for a single "综合主表"
    table = []

    def add_row(**kwargs):
        table.append(kwargs)

    if val:
        add_row(
            track="single",
            eval="AMASS val (R_SB °)",
            none=_round(_get(val, "contrast", "none_assume_I_deg_mean"), 2),
            learned=_round(_get(val, "rot_err_deg_mean"), 2),
            oracle=0.0,
            note="geo(R_hat, R_SB); contrast.none is subset",
        )
    if dval:
        add_row(
            track="dual",
            eval="AMASS val (R_SB °)",
            none=_round(
                (
                    float(_get(dval, "contrast", "none_watch_deg", default=0))
                    + float(_get(dval, "contrast", "none_phone_deg", default=0))
                )
                / 2.0,
                2,
            ),
            learned=_round(_get(dval, "mean_rot_err_deg"), 2),
            oracle=0.0,
            note="mean of watch/phone",
        )
    if d5:
        add_row(
            track="single",
            eval="D5 IMUPoser (R_SB °)",
            none=_round(_get(d5, "D5_B_ori_proxy_deg", "none", "mean"), 2),
            learned=_round(_get(d5, "D5_A_rsb_error_deg", "mean"), 2),
            oracle=_round(_get(d5, "D5_B_ori_proxy_deg", "oracle", "mean"), 2),
            note="A=learned extrinsic; none/oracle from B ori proxy",
        )
    if d5d:
        add_row(
            track="dual",
            eval="D5 IMUPoser (R_SB °)",
            none=_round(_get(d5d, "D5_B_ori_proxy_deg", "none", "pooled", "mean"), 2),
            learned=_round(_get(d5d, "D5_A_rsb_error_deg", "mean", "mean"), 2),
            oracle=_round(_get(d5d, "D5_B_ori_proxy_deg", "oracle", "pooled", "mean"), 2),
            note="mean of watch/phone",
        )
    if d6:
        for ds_name, label in (
            ("amass", "D6 AMASS Joint Acc"),
            ("imuposer", "D6 IMUPoser Joint Acc"),
        ):
            r = _get(d6, "datasets", ds_name, "results")
            if not r:
                continue
            add_row(
                track="single",
                eval=label,
                none=_round(_get(r, "none", "joint_acc"), 3),
                learned=_round(_get(r, "learned", "joint_acc"), 3),
                oracle=_round(_get(r, "oracle", "joint_acc"), 3),
                note="Week1 after calib",
            )
    if d6d:
        for ds_name, label in (
            ("amass", "D6 AMASS Joint Acc"),
            ("imuposer", "D6 IMUPoser Joint Acc"),
        ):
            r = _get(d6d, "datasets", ds_name, "results")
            if not r:
                continue
            add_row(
                track="dual",
                eval=label,
                none=_round(_get(r, "none", "joint_acc"), 3),
                learned=_round(_get(r, "learned", "joint_acc"), 3),
                oracle=_round(_get(r, "oracle", "joint_acc"), 3),
                note="Week1 after dual calib",
            )

    # Compact comparison: dual vs single on shared evals
    compare = []
    by_key = {}
    for row in table:
        by_key[(row["track"], row["eval"])] = row
    for eval_name in (
        "AMASS val (R_SB °)",
        "D5 IMUPoser (R_SB °)",
        "D6 AMASS Joint Acc",
        "D6 IMUPoser Joint Acc",
    ):
        s = by_key.get(("single", eval_name))
        d = by_key.get(("dual", eval_name))
        if not s or not d:
            continue
        lower_better = "R_SB" in eval_name or "°" in eval_name
        s_l, d_l = s["learned"], d["learned"]
        if s_l is None or d_l is None:
            winner = "n/a"
        elif abs(s_l - d_l) < 1e-6:
            winner = "tie"
        elif lower_better:
            winner = "dual" if d_l < s_l else "single"
        else:
            winner = "dual" if d_l > s_l else "single"
        compare.append(
            {
                "eval": eval_name,
                "single_learned": s_l,
                "dual_learned": d_l,
                "winner": winner,
                "lower_better": lower_better,
            }
        )

    summary = {
        "title": "Week2 Rot Extrinsic — consolidated quantitative summary",
        "sources": {
            "single_amass": "metrics_val.json",
            "dual_amass": "metrics_dual_val.json",
            "single_d5": "metrics_imuposer_d5.json",
            "dual_d5": "metrics_imuposer_d5_dual.json",
            "single_d6": "metrics_downstream_week1.json",
            "dual_d6": "metrics_downstream_week1_dual.json",
        },
        "main_table": table,
        "single_vs_dual": compare,
        "details": {
            "single_amass_per_slot": _get(val, "per_slot") if val else None,
            "dual_amass": {
                "watch_mean": _get(dval, "watch_rot_err_deg_mean"),
                "phone_mean": _get(dval, "phone_rot_err_deg_mean"),
                "mean": _get(dval, "mean_rot_err_deg"),
            }
            if dval
            else None,
            "single_d5_per_slot": _get(d5, "D5_A_per_slot") if d5 else None,
            "dual_d5_per_slot": _get(d5d, "D5_A_per_slot") if d5d else None,
            "single_d6": _get(d6, "datasets") if d6 else None,
            "dual_d6": _get(d6d, "datasets") if d6d else None,
        },
    }
    return summary


def print_table(summary: Dict[str, Any]) -> None:
    rows = summary["main_table"]
    print("=== Week2 综合主表 (None / Learned / Oracle) ===")
    print(f"{'Track':<8} {'Eval':<28} {'None':>8} {'Learned':>8} {'Oracle':>8}")
    print("-" * 68)
    for r in rows:
        print(
            f"{r['track']:<8} {r['eval']:<28} "
            f"{_fmt(r['none']):>8} {_fmt(r['learned']):>8} {_fmt(r['oracle']):>8}"
        )
    print()
    print("=== 单设备 vs 双设备（Learned）===")
    print(f"{'Eval':<28} {'Single':>8} {'Dual':>8} {'Winner':>8}")
    print("-" * 56)
    for c in summary["single_vs_dual"]:
        print(
            f"{c['eval']:<28} {_fmt(c['single_learned']):>8} "
            f"{_fmt(c['dual_learned']):>8} {c['winner']:>8}"
        )


def _fmt(x) -> str:
    if x is None:
        return "n/a"
    if isinstance(x, float) and abs(x) < 1.5:
        return f"{x:.3f}"
    if isinstance(x, float):
        return f"{x:.2f}"
    return str(x)


def main():
    cfg = load_config(resolve_path("experiments/week2_rot_ext/configs/default.yaml"))
    log_dir = resolve_path("experiments/week2_rot_ext/outputs/logs")
    log_dir.mkdir(parents=True, exist_ok=True)
    summary = build_summary(log_dir)
    out = log_dir / "metrics_summary.json"
    with open(out, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print_table(summary)
    print(f"\nsaved {out}")
    _ = cfg  # config reserved for future path overrides


if __name__ == "__main__":
    main()
