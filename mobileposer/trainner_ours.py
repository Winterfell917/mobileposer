import config
from Aplus.runner import *
from articulate.evaluator import (
    RotationErrorEvaluator,
    PerJointRotationErrorEvaluator,
    PerJointAccErrorEvaluator,
)
from articulate.math.angular import *
from tqdm import tqdm
from Aplus.data.process import *
from torch.utils.tensorboard import SummaryWriter
import torch
import torch.nn as nn


class TicTrainner(BaseTrainer):
    def __init__(
        self,
        model: nn.Module,
        train_data,
        optimizer,
        batch_size,
        val_data=None,
        log_dir="data/log",
    ):
        self.model = model
        self.optimizer = optimizer
        self.MSE = nn.MSELoss()

        self.train_data = train_data
        self.val_data = val_data
        self.batch_size = batch_size
        self.epoch = 0

        # ================= TensorBoard =================
        self.writer = SummaryWriter(log_dir=log_dir)

        rep = RotationRepresentation.ROTATION_MATRIX
        self.rot_err_evaluator = RotationErrorEvaluator(rep=rep)
        self.per_joint_rot_err_evaluator = PerJointRotationErrorEvaluator(rep=rep)

        self.imu_num = config.imu_num
        self.checkpoint = None
        self.init_logged = False

    # ============================================================
    # Validation (with per-IMU + init baseline)
    # ============================================================
    @torch.no_grad()
    def evaluate(self, data):
        from simulations_ours import imu_offset_simulation_realdata

        device = self.get_model_device()

        avg_meter_loss = DataMeter()
        avg_meter_angle_offset = DataMeter()

        self.model.eval()

        data_loader = DataLoader(dataset=data, batch_size=self.batch_size, shuffle=False, drop_last=False,)
        for data in tqdm(data_loader, desc="[Val]"):
            
            rot, acc, rot_gt, acc_gt = data
            rot = rot.to(device)
            acc = acc.to(device)
            rot_gt = rot_gt.to(device)
            acc_gt = acc_gt.to(device)

            # ---- simulate offset (real data) ----
            rot, acc, offset = imu_offset_simulation_realdata(rot, acc, rot_gt, acc_gt, imu_num=config.imu_num)

            rot, acc, offset = rot.flatten(2), acc.flatten(2), offset.flatten(2)
            acc /= 30

            x = torch.cat([acc, rot], dim=-1)
            offset_hat = self.model(x)

            loss = self.MSE(offset_hat, offset)

            # ---- to rotation matrices ----
            offset = r6d_to_rotation_matrix(offset.reshape(-1, 6)).reshape(-1, self.imu_num, 3, 3)
            offset_hat = r6d_to_rotation_matrix(offset_hat.reshape(-1, 6)).reshape(-1, self.imu_num, 3, 3)

            # ---- predicted offset error ----
            ang_err_offset = self.per_joint_rot_err_evaluator(p=offset_hat, t=offset, joint_num=self.imu_num,).cpu()

            if not self.init_logged:
                # ---- initial (zero-offset) baseline ----
                avg_meter_angle_init = DataMeter()
                identity = torch.eye(3, device=offset.device)
                offset_zero = identity.view(1, 1, 3, 3).repeat(offset.shape[0], self.imu_num, 1, 1)
                ang_err_init = self.per_joint_rot_err_evaluator(p=offset_zero, t=offset, joint_num=self.imu_num,).cpu()
                avg_meter_angle_init.update(ang_err_init, n_sample=len(offset))
            else:
                avg_meter_angle_init = None

            # ---- meters ----
            avg_meter_loss.update(loss.item(), n_sample=len(x))
            avg_meter_angle_offset.update(ang_err_offset, n_sample=len(offset))
            

        return (
            avg_meter_loss.get_avg(),
            avg_meter_angle_offset.get_avg(),  # per-IMU
            avg_meter_angle_init.get_avg() if avg_meter_angle_init else None
        )

    # ============================================================
    # Training loop
    # ============================================================
    def run(self, epoch, data_shuffle=True, evaluator=None, noise_sigma=None):
        from simulations_ours import imu_offset_simulation, simulation_MODA

        device = self.get_model_device()

        avg_meter_loss = DataMeter()
        avg_meter_angle_offset = DataMeter()

        for e in range(epoch):
            data_loader = DataLoader(dataset=self.train_data, batch_size=self.batch_size, shuffle=data_shuffle, drop_last=False,)

            avg_meter_loss.reset()
            avg_meter_angle_offset.reset()
            self.model.train()
            # idx = 0
            for data in tqdm(data_loader, desc=f"[Train][Epoch {self.epoch}]"):
                # if idx > 2:
                #     break
                # idx += 1
                rot, acc = data
                rot = rot.to(device)
                acc = acc.to(device)

                # rot, acc, offset = imu_offset_simulation(
                #     imu_rot=rot,
                #     imu_acc=acc,
                #     imu_num=config.imu_num,
                #     offset_range=45,
                #     random_global_yaw=True,
                # )
                rot, acc, offset = simulation_MODA(imu_rot=rot, imu_acc=acc, imu_num=config.imu_num, random_global_yaw=True)

                rot, acc, offset = rot.flatten(2), acc.flatten(2), offset.flatten(2)
                acc /= 30

                x = torch.cat([acc, rot], dim=-1)

                self.optimizer[0].zero_grad()
                offset_hat = self.model(x)
                
                loss = self.MSE(offset_hat, offset)
                loss.backward()
                self.optimizer[0].step()

                # ---- rotation error (train) ----
                offset = r6d_to_rotation_matrix(offset.reshape(-1, 6)).reshape(-1, self.imu_num, 3, 3)

                offset_hat_rm = r6d_to_rotation_matrix(offset_hat.detach().reshape(-1, 6)).reshape(-1, self.imu_num, 3, 3)

                ang_err_offset = self.per_joint_rot_err_evaluator(p=offset_hat_rm, t=offset, joint_num=self.imu_num,).cpu()

                avg_meter_loss.update(loss.item(), n_sample=len(x))
                avg_meter_angle_offset.update(ang_err_offset, n_sample=len(offset))

            # ================= Train TensorBoard =================
            loss_train = avg_meter_loss.get_avg()
            err_offset_train = avg_meter_angle_offset.get_avg()

            self.writer.add_scalar("Train/Loss", loss_train, self.epoch)
            self.writer.add_scalar("Train/OffsetError_Mean",err_offset_train.mean().item(),self.epoch,)

            # ================= Val TensorBoard =================
            if self.val_data is not None:
                
                loss_val, err_offset_val, err_init_val = self.evaluate(self.val_data)
                if not self.init_logged:
                    self.writer.add_scalar("Val/Init_OffsetError_Mean", err_init_val.mean().item(), 0,)
                    
                self.writer.add_scalar("Val/Loss", loss_val, self.epoch)
                self.writer.add_scalar("Val/OffsetError_Mean", err_offset_val.mean().item(), self.epoch,)
                
                # ---- per-IMU logging ----
                for j in range(self.imu_num):
                    if not self.init_logged:
                        self.writer.add_scalar(f"Val/Init_OffsetError_IMU{j}", err_init_val[j].item(),0,)
                    self.writer.add_scalar(f"Val/OffsetError_IMU{j}", err_offset_val[j].item(),self.epoch,)
                
                self.init_logged = True

            self.epoch += 1
