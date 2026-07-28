import torch
from articulate.math.angular import *
import numpy as np
from Aplus.data.process import add_gaussian_noise
import config
import os
import articulate as art

@torch.no_grad()
def imu_drift_offset_simulation(imu_rot, imu_acc, imu_num=6, ego_imu_id=-1, drift_range=60,
                 offset_range=45, random_global_yaw=True, global_yaw_only=False, acc_noise=0.025):
    """
    Simulates drift and offset effects on IMU (Inertial Measurement Unit) data.

    This function applies simulated drift and offset transformations to the rotation and acceleration data
    from multiple IMUs. It applies random drift and offset, and optionally applies random global yaw rotation.

    Parameters:
    ----------
    imu_rot : torch.Tensor
        A tensor of shape (batch_size, seq_len, imu_num, 3, 3) representing the well-calibrated IMU rotations
        (bone orientation measurements in SMPL frame) sequences in the batch.

    imu_acc : torch.Tensor
        A tensor of shape (batch_size, seq_len, imu_num, 3, 1) representing the acceleration data (in SMPL frame)
        for each IMU in the batch.

    imu_num : int, optional
        The number of IMUs in the simulation (default is 6).

    ego_imu_id : int, optional
        The ID of the ego-yaw IMU (default is -1).

    drift_range : float, optional
        The maximum range of random drift to apply to the IMU data, in degrees (default is 60 degrees).

    offset_range : float, optional
        The maximum range of random offset to apply to the IMU data, in degrees (default is 45 degrees).

    random_global_yaw : bool, optional
        If True, a random global yaw rotation will be applied to the IMU data (default is True).

    global_yaw_only : bool, optional
        If True, only global yaw will be applied without drift or offset (default is False).

    acc_noise : float, optional
        The standard deviation of the Gaussian noise to be added to the acceleration data (default is 0.025).

    Returns:
    -------
    imu_acc : torch.Tensor
        The modified acceleration data after applying drift, offset, and noise.

    imu_rot : torch.Tensor
        The modified rotation data after applying drift, offset, and any global yaw transformations.

    drift : torch.Tensor
        A tensor representing the simulated drift applied to the IMU data, shaped as (batch_size, imu_num, imu_num).

    offset : torch.Tensor
        A tensor representing the simulated offset applied to the IMU data, shaped as (batch_size, imu_num, imu_num).

    Notes:
    -----
    - The added drift matrices are in ego-yaw coordinate system.
    """

    batch_size = imu_rot.shape[0]
    seq_len = imu_rot.shape[1]
    drift_range = (drift_range / 180) * torch.pi
    non_yaw_drift_range = (20 / 180) * torch.pi
    offset_range = (offset_range / 180) * torch.pi
    GA = torch.FloatTensor([[0, -9.80665, 0]])

    # acc noise
    imu_acc = add_gaussian_noise(imu_acc, sigma=acc_noise)

    if global_yaw_only:
        # no drift
        drift = config.unit_r6d.reshape(1, 1, 6).repeat(batch_size, imu_num, 1).to(imu_rot.device)
        offset = drift.clone()
    else:
        drift = torch.zeros(batch_size, imu_num, 3).to(imu_rot.device)
        offset = torch.zeros(batch_size, imu_num, 3).to(imu_rot.device)
        
        # random drift
        drift[:, :, 0] = drift[:, :, 0].uniform_(-drift_range, drift_range)
        drift[:, :, [1, 2]] = drift[:, :, [1, 2]].uniform_(-non_yaw_drift_range, non_yaw_drift_range)
        drift[:, ego_imu_id, [0]] *= 0

        # random offset
        offset = offset.uniform_(-offset_range, offset_range)

        # random scaling
        if True:
            scale_mask = torch.zeros(batch_size, 1, 1).uniform_(0, 1).to(drift.device)
            drift *= scale_mask
            offset *= scale_mask
        
        drift = euler_angle_to_rotation_matrix(drift, seq='YZX').reshape(batch_size, imu_num, 3, 3)
        offset = euler_angle_to_rotation_matrix(offset).reshape(batch_size, imu_num, 3, 3)

    # random global yaw
    if random_global_yaw:
        global_yaw_rot = torch.zeros(batch_size, 1, 3).to(imu_rot.device)
        global_yaw_rot[:, :, 1] = global_yaw_rot[:, :, 1].uniform_(-np.pi, np.pi)
        global_yaw_rot = euler_angle_to_rotation_matrix(global_yaw_rot).reshape(batch_size, 1, 3, 3)

        global_yaw_rot = global_yaw_rot.unsqueeze(1).repeat(1, seq_len, imu_num, 1, 1)
        imu_rot = global_yaw_rot.matmul(imu_rot)

        if imu_acc is not None:
            imu_acc = global_yaw_rot.matmul(imu_acc)

    if global_yaw_only == False:
        # adding drift & offset
        drift = drift.unsqueeze(1).repeat(1, seq_len, 1, 1, 1)
        offset = offset.unsqueeze(1).repeat(1, seq_len, 1, 1, 1)
        imu_rot = drift.matmul(imu_rot).matmul(offset)
        # GA Leakage simulation
        if imu_acc is not None:
            GA = GA.to(imu_acc.device)
            GA = GA.reshape(1, 1, 1, 3, 1)
            GA = GA.repeat(imu_acc.shape[0], imu_acc.shape[1], imu_acc.shape[2], 1, 1)

            # [method 1 & 2 are equal]
            # method 1 (According to Hardware Level Acceleration formulation in the supp mat)
            imu_acc = drift.matmul(imu_acc) + (torch.eye(3, device=imu_acc.device) - drift).matmul(GA)

            # method 2
            # imu_acc -= GA
            # imu_acc = drift.matmul(imu_acc)
            # imu_acc += GA

    drift = rotation_matrix_to_r6d(drift[:, 0].reshape(-1, 3, 3)).reshape(batch_size, imu_num, 6)
    offset = rotation_matrix_to_r6d(offset[:, 0].reshape(-1, 3, 3)).reshape(batch_size, imu_num, 6)

    return imu_rot, imu_acc, drift, offset

def imu_offset_simulation(imu_rot, imu_acc, imu_num=2, acc_noise=0.025,
                          offset_range=45,
                          random_global_yaw=True):
    batch_size = imu_rot.shape[0]
    seq_len = imu_rot.shape[1]
    
    offset_range = (offset_range / 180) * torch.pi
    
    # add acc noise
    imu_acc = add_gaussian_noise(imu_acc, sigma=acc_noise)
    
    # random offset
    offset = torch.zeros(batch_size, imu_num, 3).to(imu_rot.device)
    offset = offset.uniform_(-offset_range, offset_range)
    
    # random scaling
    scale_mask = torch.zeros(batch_size, 1, 1).uniform_(0, 1).to(offset.device)
    offset *= scale_mask
    
    offset = euler_angle_to_rotation_matrix(offset, seq='YZX').reshape(batch_size, imu_num, 3, 3)
    
    # random global yaw
    if random_global_yaw:
        global_yaw_rot = torch.zeros(batch_size, 1, 3).to(imu_rot.device)
        global_yaw_rot[:, :, 1] = global_yaw_rot[:, :, 1].uniform_(-np.pi, np.pi)
        global_yaw_rot = euler_angle_to_rotation_matrix(global_yaw_rot).reshape(batch_size, 1, 3, 3)

        global_yaw_rot = global_yaw_rot.unsqueeze(1).repeat(1, seq_len, imu_num, 1, 1)
        imu_rot = global_yaw_rot.matmul(imu_rot)

        if imu_acc is not None:
            imu_acc = global_yaw_rot.matmul(imu_acc)    

    # adding offset
    offset = offset.unsqueeze(1).repeat(1, seq_len, 1, 1, 1)
    imu_rot = imu_rot.matmul(offset)

    offset = rotation_matrix_to_r6d(offset).reshape(batch_size, seq_len, imu_num, 6)
    
    return imu_rot, imu_acc, offset

def imu_offset_simulation_realdata(rot, acc, rot_gt, acc_gt, imu_num=3, acc_noise=0.025):
    batch_size, seq_len = rot.shape[0], rot.shape[1]
    
    acc = add_gaussian_noise(acc, sigma=acc_noise)
    
    # calculate offset
    # delta_R = R^T * R_gt
    # rot: [batch_size, seq_len, imu_num, 3, 3], rot_gt: [batch_size, seq_len, imu_num, 3, 3]
    offset_mat = torch.matmul(rot.transpose(-1, -2), rot_gt)
    offset = rotation_matrix_to_r6d(offset_mat.reshape(-1, 3, 3)).reshape(batch_size, seq_len, imu_num, 6)
    
    return rot, acc, offset

def simulation_MODA(imu_rot, imu_acc, imu_num=3, acc_noise=0.025, random_global_yaw=True):
    device = imu_rot.device
    B, T = imu_rot.shape[0], imu_rot.shape[1]
    
    imu_acc = add_gaussian_noise(imu_acc, sigma=acc_noise)

    # TODO: add MODA specific offset simulation here
    sigma = np.pi / 32
    delta_euler = torch.randn(B, T, imu_num, 3, device=device) * sigma
    
    delta_rot = euler_angle_to_rotation_matrix(delta_euler.reshape(-1, 3), seq="YZX",).reshape(B, T, imu_num, 3, 3)
    
    # cumulative product along time
    offset_rot = torch.zeros_like(delta_rot)
    offset_rot[:, 0] = delta_rot[:, 0]
    for t in range(1, T):
        offset_rot[:, t] = offset_rot[:, t - 1].matmul(delta_rot[:, t])
    
    # random global yaw
    if random_global_yaw:
        global_yaw_rot = torch.zeros(B, 1, 3).to(imu_rot.device)
        global_yaw_rot[:, :, 1] = global_yaw_rot[:, :, 1].uniform_(-np.pi, np.pi)
        global_yaw_rot = euler_angle_to_rotation_matrix(global_yaw_rot).reshape(B, 1, 3, 3)

        global_yaw_rot = global_yaw_rot.unsqueeze(1).repeat(1, T, imu_num, 1, 1)
        imu_rot = global_yaw_rot.matmul(imu_rot)

        if imu_acc is not None:
            imu_acc = global_yaw_rot.matmul(imu_acc)
    
    imu_rot = imu_rot.matmul(offset_rot)
    offset = rotation_matrix_to_r6d(offset_rot).reshape(B, T, imu_num, 6)

    return imu_rot, imu_acc, offset

def simulation_ours(imu_rot, imu_acc, imu_num=3, acc_noise=0.025, random_global_yaw=True):
    device = imu_rot.device
    B, T = imu_rot.shape[0], imu_rot.shape[1]
    
    imu_acc = add_gaussian_noise(imu_acc, sigma=acc_noise)
    
    # TODO: add pose-aware offset simulation

    # 1) compute rotation angle to identity (geodesic distance on SO(3))
    trace = imu_rot[..., 0, 0] + imu_rot[..., 1, 1] + imu_rot[..., 2, 2]
    cos_theta = (trace - 1) / 2
    cos_theta = torch.clamp(cos_theta, -1 + 1e-6, 1 - 1e-6)
    theta = torch.acos(cos_theta)        # [B, T, imu_num], in [0, pi]

    # 2) pose-aware offset magnitude (proportional to deviation)
    max_offset_rad = 60 / 180 * np.pi    # max rotation error, e.g. 20°
    offset_scale = theta / np.pi * max_offset_rad
    offset_scale = offset_scale.unsqueeze(-1)  # [B, T, imu_num, 1]

    # 3) temporal periodic drift (sinusoidal)
    t = torch.linspace(0, 1, T, device=device).view(1, T, 1, 1)

    freq = torch.empty(B, 1, imu_num, 1, device=device).uniform_(0.5, 2.0)
    phase = torch.empty(B, 1, imu_num, 1, device=device).uniform_(0, 2 * np.pi)

    sin_wave = torch.sin(2 * np.pi * freq * t + phase)  # [B, T, imu_num, 1]

    # 4) random rotation axis per IMU
    axis = torch.randn(B, 1, imu_num, 3, device=device)
    axis = axis / (axis.norm(dim=-1, keepdim=True) + 1e-8)

    # 5) final pose-aware offset in axis-angle form
    offset_axis_angle = axis * offset_scale * sin_wave  # [B, T, imu_num, 3]

    # 6) convert to rotation matrix and apply
    offset_rot = euler_angle_to_rotation_matrix(offset_axis_angle)

    # random global yaw
    if random_global_yaw:
        global_yaw_rot = torch.zeros(B, 1, 3).to(imu_rot.device)
        global_yaw_rot[:, :, 1] = global_yaw_rot[:, :, 1].uniform_(-np.pi, np.pi)
        global_yaw_rot = euler_angle_to_rotation_matrix(global_yaw_rot).reshape(B, 1, 3, 3)

        global_yaw_rot = global_yaw_rot.unsqueeze(1).repeat(1, T, imu_num, 1, 1)
        imu_rot = global_yaw_rot.matmul(imu_rot)

        if imu_acc is not None:
            imu_acc = global_yaw_rot.matmul(imu_acc)
    
    imu_rot = imu_rot.matmul(offset_rot)
    offset = rotation_matrix_to_r6d(offset_rot).reshape(B, T, imu_num, 6)
    
    return imu_rot, imu_acc, offset

def load_data():
    from config import paths
    data_dir = paths.eval_dir
    dataset_name = "imuposer_full.pt"
    data = torch.load(os.path.join(data_dir, dataset_name))
    
    return data

def vis_simulation(sim_type="tic", T=128, imu_num=3, num_runs=3, save_dir="data/vis_offset", device="cuda",):
    import matplotlib.pyplot as plt

    os.makedirs(save_dir, exist_ok=True)
    
    data = load_data()

    for run_id in range(num_runs):
        run_dir = os.path.join(save_dir, sim_type, f"run_{run_id:02d}")
        os.makedirs(run_dir, exist_ok=True)
        # ------------------------------------------------
        # 1. fake IMU input
        # ------------------------------------------------
        B = 1

        imu_rot = torch.eye(3, device=device).view(1, 1, 1, 3, 3).repeat(B, T, imu_num, 1, 1)

        imu_acc = torch.zeros(B, T, imu_num, 3, 1, device=device)

        # ------------------------------------------------
        # 2. run simulation
        # ------------------------------------------------
        if sim_type == "tic":
            _, _, offset = imu_offset_simulation(imu_rot, imu_acc, imu_num=imu_num, random_global_yaw=False,)
            title_prefix = "TIC (static)"

        elif sim_type == "moda":
            _, _, offset = simulation_MODA(imu_rot, imu_acc, imu_num=imu_num, random_global_yaw=False,)
            title_prefix = "MODA (random walk)"
        else:
            raise ValueError(sim_type)

        # ------------------------------------------------
        # 3. r6d → Euler
        # ------------------------------------------------
        offset = offset[0]  # [T, imu_num, 6]

        offset_R = r6d_to_rotation_matrix(offset.reshape(-1, 6)).reshape(T, imu_num, 3, 3)

        euler = art.math.rotation_matrix_to_euler_angle(offset_R, seq="YZX") * 180 / np.pi  # [T, imu_num, 3]
        euler = euler.reshape(T, imu_num, 3).cpu().numpy()
        x = np.arange(T)

        # ------------------------------------------------
        # 4. plot
        # ------------------------------------------------
        for imu_id in range(imu_num):
            plt.figure(figsize=(7, 3))
            plt.plot(x, euler[:, imu_id, 0], label="yaw (Y)")
            plt.plot(x, euler[:, imu_id, 1], label="roll (Z)")
            plt.plot(x, euler[:, imu_id, 2], label="pitch (X)")
            plt.ylim(-60, 60) 
            plt.xlabel("Frame")
            plt.ylabel("Angle (deg)")
            plt.title(f"{title_prefix} | Run {run_id} | IMU {imu_id}")
            plt.legend(loc="upper right") 
            plt.grid(True)
            plt.tight_layout()

            out_path = os.path.join(
                run_dir,
                f"imu_{imu_id:02d}.png",
            )
            plt.savefig(out_path, dpi=200)
            plt.close()

            print(f"[Saved] {out_path}")

if __name__ == "__main__":
    vis_simulation(sim_type="tic")