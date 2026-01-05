import os
import numpy as np
import torch
from argparse import ArgumentParser
from tqdm import tqdm
from pathlib import Path

import articulate as art
from articulate.model import ParametricModel

from config import *
from helpers import *
from constants import MODULES
from utils.model_utils import load_model
from data_mocap import PoseDataset
from models import MobilePoserNet
from model_tic import *

vi_mask = torch.tensor([1961, 5424, 876, 4362, 411, 3021])
ji_mask = torch.tensor([18, 19, 1, 2, 15, 0])
body_model = ParametricModel(paths.smpl_file, device=model_config.device)

# ============================================================
# Subject utils
# ============================================================

def build_subject_ranges(subject_num):
    ranges = []
    start = 0
    for n in subject_num:
        ranges.append((start, start + n))
        start += n
    return ranges


def get_subject_and_local_idx(idx, subject_ranges):
    for sid, (s, e) in enumerate(subject_ranges):
        if s <= idx < e:
            return sid, idx - s
    raise ValueError(f"Index {idx} not in subject ranges")


# ============================================================
# Pose Evaluator
# ============================================================

class PoseEvaluator:
    def __init__(self):
        self._eval_fn = art.FullMotionEvaluator(
            paths.smpl_file,
            joint_mask=torch.tensor([2, 5, 16, 20]),
            fps=datasets.fps,
        )

    def eval(self, pose_p, pose_t, tran_p=None, tran_t=None):
        pose_p = pose_p.clone().view(-1, 24, 3, 3)
        pose_t = pose_t.clone().view(-1, 24, 3, 3)
        tran_p = tran_p.clone().view(-1, 3)
        tran_t = tran_t.clone().view(-1, 3)

        pose_p[:, joint_set.ignored] = torch.eye(3, device=pose_p.device)
        pose_t[:, joint_set.ignored] = torch.eye(3, device=pose_t.device)

        errs = self._eval_fn(pose_p, pose_t, tran_p=tran_p, tran_t=tran_t)

        return torch.stack([
            errs[9],           # SIP
            errs[3],           # Angular
            errs[9],           # Masked Angular
            errs[0] * 100,     # Positional
            errs[7] * 100,     # Masked Positional
            errs[1] * 100,     # Mesh
            errs[4] / 100,     # Jitter
            errs[6],           # Distance
        ])

    @staticmethod
    def print(errors):
        names = [
            'SIP Error (deg)',
            'Angular Error (deg)',
            'Masked Angular Error (deg)',
            'Positional Error (cm)',
            'Masked Positional Error (cm)',
            'Mesh Error (cm)',
            'Jitter Error (100m/s^3)',
            'Distance Error (cm)',
        ]
        for i, n in enumerate(names):
            print(f"{n}: {errors[i,0]:.2f} (+/- {errors[i,1]:.2f})")

    @staticmethod
    def print_single(errors, file=None):
        names = [
            'Angular Error (deg)',
            'Mesh Error (cm)',
        ]
        max_len = max(len(n) for n in names)
        outs = []
        for i, n in enumerate([
            'SIP Error (deg)',
            'Angular Error (deg)',
            'Masked Angular Error (deg)',
            'Positional Error (cm)',
            'Masked Positional Error (cm)',
            'Mesh Error (cm)',
            'Jitter Error (100m/s^3)',
            'Distance Error (cm)',
        ]):
            if n in names:
                outs.append(f"{n:<{max_len}}: {errors[i,0]:.2f}")
        print(" | ".join(outs), file=file)


# ============================================================
# Acc synthesis
# ============================================================

TARGET_FPS = 30

def _syn_acc(v, smooth_n=4):
    mid = smooth_n // 2
    scale = TARGET_FPS ** 2
    acc = torch.stack([(v[i] + v[i+2] - 2*v[i+1]) * scale for i in range(len(v)-2)])
    acc = torch.cat([torch.zeros_like(acc[:1]), acc, torch.zeros_like(acc[:1])])
    if mid != 0:
        acc[smooth_n:-smooth_n] = torch.stack([
            (v[i] + v[i+smooth_n*2] - 2*v[i+smooth_n]) * scale / smooth_n**2
            for i in range(len(v) - smooth_n*2)
        ])
    return acc


# ============================================================
# Main evaluation
# ============================================================

@torch.no_grad()
def evaluate_pose(
    model,
    dataset,
    num_future_frame=5,
    evaluate_tran=False,
    calibrator=None,
    use_cali=False,
    cali_type=None,
    save_dir=None,
    gt_input=False,
    subject_num=None,
):
    device = model_config.device
    xs, ys = zip(*[(imu.to(device), (pose.to(device), tran)) for imu, pose, joint, tran in dataset])
    
    evaluator = PoseEvaluator()
    model.eval()

    if subject_num is not None:
        subject_ranges = build_subject_ranges(subject_num)
    else:
        subject_ranges = None

    online_errs = []

    for idx, (x, y) in enumerate(tqdm(zip(xs, ys), total=len(xs))):
        model.reset()

        x = x.to(device)
        pose_t, tran_t = y
        pose_t = pose_t.to(device)
        tran_t = tran_t.to(device)

        pose_t = art.math.r6d_to_rotation_matrix(pose_t).view(-1, 24, 3, 3)

        acc = x[:, :5*3] * amass.acc_scale
        rot = x[:, 5*3:]
        acc = acc.view(-1, 5, 3)
        rot = rot.view(-1, 5, 3, 3)

        if use_cali:
            calibrator.reset()
            acc_raw = acc[:, [0,3,4]]
            rot_raw = rot[:, [0,3,4]]

            if cali_type == "tic":
                rot_cali, acc_cali, _, _ = calibrator.run(rot_raw, acc_raw)
            else:
                rot_cali, acc_cali, _, _ = calibrator.run_per_frame(rot_raw, acc_raw)

            acc[:, [0,3,4]] = acc_cali
            rot[:, [0,3,4]] = rot_cali

        if gt_input:
            # convert to device
            pose_t = pose_t.to(device)
            tran_t = tran_t.to(device)
            grot, joint, vert = body_model.forward_kinematics(pose=pose_t, tran=tran_t, calc_mesh=True)
            # vacc = _syn_acc(vert[:, vi_mask])
            vrot = grot[:, ji_mask]
            
            rot_gt = vrot[:, [0, 3, 4]]
            
            rot[:, [0, 3, 4]] = rot_gt

        acc = acc / amass.acc_scale
        
        x = torch.cat([acc.flatten(1), rot.flatten(1)], dim=-1)

        outs = [
            model.forward_online(f)
            for f in torch.cat([x, x[-1].repeat(num_future_frame, 1)])
        ]
        pose_p, _, tran_p, _ = [
            torch.stack(v)[num_future_frame:]
            for v in zip(*outs)
        ]

        err = evaluator.eval(pose_p, pose_t, tran_p=tran_p, tran_t=tran_t)
        online_errs.append(err)

        # ================= save per subject =================
        if save_dir:
            if subject_ranges is not None:
                sid, lid = get_subject_and_local_idx(idx, subject_ranges)
                out_dir = save_dir / f"subject_{sid:02d}"
                out_dir.mkdir(parents=True, exist_ok=True)
                out_path = out_dir / f"{lid:02d}.pt"
            else:
                out_path = save_dir / f"{idx:02d}.pt"

            torch.save(
                {
                    "pose_t": pose_t.cpu(),
                    "pose_p": pose_p.cpu(),
                },
                out_path
            )

    # ================= print summary =================
    print("============== online ================")
    evaluator.print(torch.stack(online_errs).mean(dim=0))

    # ================= log per seq =================
    if save_dir:
        log_path = save_dir / "log.txt"
        with open(log_path, "w") as f:
            for i, e in enumerate(online_errs):
                if subject_ranges is not None:
                    sid, lid = get_subject_and_local_idx(i, subject_ranges)
                    f.write(f"[Subject {sid:02d} | Seq {lid:03d}] ")
                else:
                    f.write(f"[Seq {i:04d}] ")
                evaluator.print_single(e, file=f)


# ============================================================
# Entry
# ============================================================

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--model", type=str, default="./data/checkpoint/mocap/mobileposer/base_model.pth")
    parser.add_argument("--dataset", type=str, default="xposer")
    parser.add_argument("--use_cali", action="store_true")
    parser.add_argument("--cali", type=str, default="tic")
    parser.add_argument("--gt_input", action="store_true")
    args = parser.parse_args()

    model = load_model(args.model)

    if args.cali == "tic":
        from TicOperator import *
        tic = TIC(stack=3, n_input=imu_num * (3 + 3 * 3), n_output=imu_num * 6)
        tic.restore("./data/checkpoint/calibrator/TIC_egoHead/TIC_20.pth")
        net = tic.to(model_config.device).eval()
        calibrator = TicOperator(TIC_network=net, imu_num=imu_num, data_frame_rate=30)
    else:
        from TicOperator_ours import *
        lstm = LSTMIC(n_input=imu_num * (3 + 3 * 3), n_output=imu_num * 6)
        lstm.restore("./data/checkpoint/calibrator/Ours_SynData_MODA/2.pth")
        net = lstm.to(model_config.device).eval()
        calibrator = TicOperator(TIC_network=net, imu_num=imu_num, data_frame_rate=30)

    dataset = PoseDataset(fold="test", evaluate=args.dataset)

    if args.use_cali:
        model_name = f"mobileposer_cali_{args.cali}"
    elif args.gt_input:
        model_name = "mobileposer_gtrot"
    else:
        model_name = "mobileposer"

    save_dir = Path("data") / "eval" / args.dataset / model_name
    save_dir.mkdir(parents=True, exist_ok=True)

    evaluate_pose(
        model,
        dataset,
        calibrator=calibrator,
        use_cali=args.use_cali,
        cali_type=args.cali,
        save_dir=save_dir,
        gt_input=args.gt_input,
        subject_num=imuposer_dataset.subject_num,
    )