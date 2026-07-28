import os
import torch
import articulate as art
import config
import numpy as np
import matplotlib.pyplot as plt

from tqdm import tqdm
from enum import Enum
import importlib

from evaluation_functions import *
from model_tic import *


# ============================================================
# 1. Calibration mode definition
# ============================================================

class CalibMode(str, Enum):
    NONE = "none"
    TIC = "tic"
    OURS = "ours"


GLOBAL_MODE = "global"   # <<<<<< 新增：全局未对齐模式


# ============================================================
# 2. Calibration configuration table
# ============================================================

CALIB_CONFIG = {
    CalibMode.TIC: {
        "operator_module": "TicOperator",
        "operator_class": "TicOperator",
        "model_class": TIC,
        "checkpoint": "./data/checkpoint/calibrator/TIC_egoHead/TIC_20.pth",
        "operator_kwargs": {"device": None},
        "run_type": "batch",
    },
    CalibMode.OURS: {
        "operator_module": "TicOperator_ours",
        "operator_class": "TicOperator",
        "model_class": LSTMIC,
        "checkpoint": "./data/checkpoint/calibrator/Ours_SynData_TIC/34.pth",
        "operator_kwargs": {"data_frame_rate": 30},
        "run_type": "per_frame",
    },
}


# ============================================================
# 3. Evaluator
# ============================================================

class TICModelEvaluator:
    def __init__(self, device="cuda:0"):
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")

        data_dir = config.paths.eval_dir
        dataset_name = "imuposer_full.pt"
        self.data = torch.load(os.path.join(data_dir, dataset_name))

        self.body_model = art.ParametricModel(config.paths.smpl_file)

        self.ego_id = -1
        print(f"[INFO] Ego ID: {self.ego_id}")

        self._model_cache = {}
        self._operator_cache = {}

    # ========================================================
    # operator + model (lazy load)
    # ========================================================
    def _get_operator(self, mode: CalibMode):
        if mode in self._operator_cache:
            return self._operator_cache[mode]

        cfg = CALIB_CONFIG[mode]

        model = cfg["model_class"](
            n_input=config.imu_num * (3 + 3 * 3),
            n_output=config.imu_num * 6,
        )
        model.restore(cfg["checkpoint"])
        model = model.to(self.device).eval()

        module = importlib.import_module(cfg["operator_module"])
        OperatorCls = getattr(module, cfg["operator_class"])

        op_kwargs = dict(cfg.get("operator_kwargs", {}))
        if "device" in op_kwargs:
            op_kwargs["device"] = self.device

        operator = OperatorCls(
            TIC_network=model,
            imu_num=config.imu_num,
            **op_kwargs,
        )

        self._model_cache[mode] = model
        self._operator_cache[mode] = operator
        return operator

    # ========================================================
    # unified calibration
    # ========================================================
    def apply_calibration(self, acc, ori, mode: CalibMode):
        if mode == CalibMode.NONE:
            return ori

        acc = acc.to(self.device)
        ori = ori.to(self.device)

        operator = self._get_operator(mode)
        run_type = CALIB_CONFIG[mode]["run_type"]

        if run_type == "batch":
            ori_fix, _, _, _ = operator.run(ori, acc, trigger_t=1)
        elif run_type == "per_frame":
            ori_fix, _, _, _ = operator.run_per_frame(ori, acc)
        else:
            raise ValueError(run_type)

        return ori_fix.cpu()

    # ========================================================
    # helper
    # ========================================================
    @staticmethod
    def _get_subject_and_local_idx(idx, subject_ranges):
        for sid, (s, e) in enumerate(subject_ranges):
            if s <= idx < e:
                return sid, idx - s
        raise ValueError(idx)

    # ========================================================
    # main evaluation
    # ========================================================
    def evaluate_imuposer(
        self,
        calib_modes,
        subject_num=None,
        save_dir=None,
        save_img=False,
    ):
        # ---------- subject ranges ----------
        if subject_num is not None:
            subject_ranges = []
            start = 0
            for n in subject_num:
                subject_ranges.append((start, start + n))
                start += n
            num_subjects = len(subject_num)
        else:
            subject_ranges = None

        # ---------- results ----------
        all_modes = [GLOBAL_MODE] + list(calib_modes)

        if subject_ranges is None:
            results = {mode: [] for mode in all_modes}
        else:
            results = {
                mode: {sid: [] for sid in range(num_subjects)}
                for mode in all_modes
            }

        oris = self.data["ori"]
        accs = self.data["acc"]
        poses = self.data["pose"]

        for idx in tqdm(range(len(oris))):
            if subject_ranges is not None:
                subject_id, local_seq_id = self._get_subject_and_local_idx(
                    idx, subject_ranges
                )
            else:
                subject_id, local_seq_id = None, idx

            ori = oris[idx][:, [0, 3, 4]]
            acc = accs[idx][:, [0, 3, 4]]
            pose = poses[idx]

            pose = self.body_model.forward_kinematics(
                pose, calc_mesh=False
            )[0].view(-1, 24, 3, 3)
            gt_bone = pose[:, [18, 2, 15]]

            # ================= GLOBAL (no yaw alignment) =================
            err_glb = angle_diff(
                ori,
                gt_bone,
                imu_num=3,
                print_result=False,
            )

            if subject_ranges is None:
                results[GLOBAL_MODE].append(err_glb)
            else:
                results[GLOBAL_MODE][subject_id].append(err_glb)

            # ================= save Euler error =================
            if save_img and save_dir is not None:
                R_err = ori.transpose(-2, -1).matmul(gt_bone)
                euler_err = (art.math.rotation_matrix_to_euler_angle(R_err, seq="YZX") * 180 / np.pi).view(-1, 3, 3)
                if subject_ranges is not None:
                    seq_dir = os.path.join(save_dir, f"subject_{subject_id:02d}", f"seq_{local_seq_id:03d}", )
                else:
                    seq_dir = os.path.join(save_dir, f"seq_{idx:04d}")
                os.makedirs(seq_dir, exist_ok=True)
                T = euler_err.shape[0]
                x = torch.arange(T).cpu().numpy()
                for sensor_id in range(3):
                    y = euler_err[:, sensor_id].cpu().numpy()
                    plt.figure(figsize=(6, 3)) 
                    plt.plot(x, y[:, 0], label="yaw") 
                    plt.plot(x, y[:, 1], label="roll") 
                    plt.plot(x, y[:, 2], label="pitch") 
                    plt.ylim(-60, 60) 
                    plt.xlabel("frame") 
                    plt.ylabel("euler error (deg)") 
                    plt.title( f"Sub {subject_id:02d} | Seq {local_seq_id:03d} | IMU {sensor_id}" if subject_id is not None else f"Seq {idx:04d} | IMU {sensor_id}" ) 
                    plt.legend(loc="upper right") 
                    plt.tight_layout() 
                    plt.savefig( os.path.join(seq_dir, f"imu_{sensor_id:02d}.png"), dpi=200, ) 
                    plt.close()

            # ================= EGO-YAW ALIGNED =================
            gt_yaw = get_ego_yaw(gt_bone, ego_idx=self.ego_id)
            gt_bone_ego = gt_yaw.transpose(-2, -1).matmul(gt_bone)

            for mode in calib_modes:
                ori_calib = self.apply_calibration(acc, ori, mode)
                imu_yaw = get_ego_yaw(ori_calib, ego_idx=self.ego_id)
                imu_bone_ego = imu_yaw.transpose(-2, -1).matmul(ori_calib)

                err = angle_diff(
                    imu_bone_ego,
                    gt_bone_ego,
                    imu_num=3,
                    print_result=False,
                )

                if subject_ranges is None:
                    results[mode].append(err)
                else:
                    results[mode][subject_id].append(err)

        # ================= print =================
        print("\n========== Evaluation Results ==========")

        for mode in all_modes:
            print(f"\n[{mode}]")

            if subject_ranges is None:
                errs = torch.cat(results[mode], dim=0)
                v, m = errs.mean(dim=0).tolist(), errs.mean().item()
                print(f"  Overall: [{v[0]:.2f}, {v[1]:.2f}, {v[2]:.2f}] | {m:.2f}")
            else:
                all_errs = []
                for sid in range(num_subjects):
                    errs = torch.cat(results[mode][sid], dim=0)
                    all_errs.append(errs)
                    v, m = errs.mean(dim=0).tolist(), errs.mean().item()
                    print(
                        f"  Subject {sid:02d}: "
                        f"[{v[0]:.2f}, {v[1]:.2f}, {v[2]:.2f}] | {m:.2f}"
                    )

                errs = torch.cat(all_errs, dim=0)
                v, m = errs.mean(dim=0).tolist(), errs.mean().item()
                print(f"  Overall: [{v[0]:.2f}, {v[1]:.2f}, {v[2]:.2f}] | {m:.2f}")


# ============================================================
# 7. Main
# ============================================================

def main():
    evaluator = TICModelEvaluator()
    evaluator.evaluate_imuposer(
        calib_modes=[CalibMode.NONE, CalibMode.OURS],
        subject_num=config.imuposer_dataset.subject_num,
        save_dir="data/rotation/full",
        save_img=True,
    )


if __name__ == "__main__":
    main()
