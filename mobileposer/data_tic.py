import random
import Aplus.tools.smpl_light
import config
from Aplus.data import *
from config import amass, paths
import os
from articulate.math import axis_angle_to_rotation_matrix, rotation_matrix_to_r6d, axis_angle_to_quaternion, euler_angle_to_rotation_matrix, rotation_matrix_to_euler_angle, quaternion_to_rotation_matrix
from tqdm import tqdm
from Aplus.tools.annotations import timing
import articulate as art

combo = amass.combos['lw_rp_h']

def amass_read_seg(path, min_len=128, step=2, read_rate=1, combo=None):
    data = torch.load(path)
    selected_data = []
    seg_info = []
    for slice in data:
        # print(slice.shape)
        if len(slice) < min_len*step:
            continue
        else:
            selected_data.append(slice[::step])
            seg_info.append(len(slice[::step]))

    if read_rate < 1:
        seq_num = int(len(selected_data) * read_rate)
        selected_data = selected_data[:seq_num]
        seg_info = seg_info[:seq_num]
    data = torch.cat(selected_data, dim=0)
    # print(acc_t.shape)
    # index_info = seg_info_2_index_info(seg_info)
    # print(index_info)
    # find_seg_index(index_info, 2200)
    
    # select combo in data
    if combo is not None:
        data = data[:, combo]
    
    return data, seg_info

def dip_read_seg(data, min_len=256, step=2):
    selected_data = []
    seg_info = []
    for slice in data:
        # print(slice.shape)
        if len(slice) < min_len*step:
            continue
        else:
            selected_data.append(slice[::step])
            seg_info.append(len(slice[::step]))
    data = torch.cat(selected_data, dim=0)
    return data, seg_info

def seg_info_2_index_info(seg_info):
    index_info = [0]
    for v in seg_info:
        index_info.append(index_info[-1] + v)
    return index_info

def find_seg_index(index_info, data_index, n_seg=0):

    seq_index = -1
    if n_seg != 0:
        seq_index = n_seg - 1

    for v in index_info[n_seg:]:
        if v <= data_index:
            seq_index += 1
        else:
            break

    inner_index = data_index - index_info[seq_index]
    return seq_index, inner_index

class IMUData_Syn(BaseDataset):
    def __init__(self, rot: torch.Tensor, acc, seg_info, head_acc=None, seq_len=256):
        self.rot = rot
        self.acc = acc
        self.head_acc = head_acc
        self.data_len = len(rot) - len(seg_info) * (seq_len - 1)
        self.amass_data_len = 1834274
        # self.amass_data_len = 2330402
        self.seg_info = seg_info
        self.seq_len = seq_len
        self.head_acc = head_acc
        self.index_info = seg_info_2_index_info(seg_info)

        data_seq_info = (np.array(seg_info) - (seq_len - 1)).tolist()
        self.data_index_info = seg_info_2_index_info(data_seq_info)

        self.indexer = [i for i in range(self.data_len)]
        self.mapping = [-1]
        print('processing [Data Index - Seq Begin Index] mapping...')
        n_seg = 0
        for i in tqdm(range(self.data_len)):
            self.mapping.append(self.data_index_2_seq_begin(i, n_seg))
            # print(self.mapping)
            if self.mapping[-2] + 1 != self.mapping[-1]:
                # print(self.mapping)
                n_seg += 1
                # print(n_seg)
        self.mapping = self.mapping[1:]
        print('Done')

    def __len__(self):
        return self.data_len

    def __getitem__(self, index):
        i = (index % self.data_len)
        i = self.indexer[i]
        index_begin = self.mapping[i]
        _rot, _acc = self.rot[index_begin:index_begin + self.seq_len], self.acc[index_begin:index_begin + self.seq_len]
        if self.head_acc is not None and i < self.amass_data_len:
            head_acc_idx = random.randint(0, 13)
            head_acc = self.head_acc[index_begin:index_begin + self.seq_len, head_acc_idx]
            _acc[:, 2] = head_acc # change: only contain three devices, the third device is head acc

        return _rot, _acc

    @staticmethod
    def merge(data_dict_1, data_dict_2):
        data_dict_1['imu_rot'] = torch.cat([data_dict_1['imu_rot'], data_dict_2['imu_rot']], dim=0)
        data_dict_1['imu_acc'] = torch.cat([data_dict_1['imu_acc'], data_dict_2['imu_acc']], dim=0)
        data_dict_1['seg_info'] = data_dict_1['seg_info'] + data_dict_2['seg_info']

        return data_dict_1
    def data_index_2_seq_begin(self, data_index, n_seg):
        seg_index, inner_index = find_seg_index(index_info=self.data_index_info, data_index=data_index, n_seg=n_seg)
        # print(seg_index, inner_index)
        index_begin = self.index_info[seg_index] + inner_index
        return index_begin

    @staticmethod
    @timing
    def load_data(folder_path: str, use_elbow_angle=False, pose_type='r6d', step=2, read_rate=1.0) -> dict:
        """
        Load data from files. Rewrite this function to realize data loading for your project. We suggest
        transform data to [torch.Tensor] and saving to a dict as the return value.
        Args:
            path: Path of data files
            s3_type: ['r6d', 'axis_angle']

        Returns:
            Dict of datas.
        """

        rot, seg_info = amass_read_seg(os.path.join(folder_path, 'vrot.pt'), min_len=256,step=step, read_rate=read_rate, combo=combo)
        acc, _ = amass_read_seg(os.path.join(folder_path, 'vacc.pt'), min_len=256, step=step, read_rate=read_rate, combo=combo)
        head_acc, _ = amass_read_seg(os.path.join(folder_path, 'vacc_head14.pt'), min_len=256, step=step, read_rate=read_rate)

        rot = rot.reshape(-1, config.imu_num, 3, 3)
        acc = torch.clamp(acc, min=-90, max=90).reshape(-1, config.imu_num, 3)
        head_acc = torch.clamp(head_acc, min=-90, max=90).unsqueeze(-1)
        acc = acc.unsqueeze(-1)

        return {'imu_rot': rot,
                'imu_acc': acc,
                'head_acc': head_acc,
                'seg_info': seg_info}
        
    @staticmethod
    @timing
    def load_realdata(folder_path: str, use_elbow_angle=False, pose_type='r6d', step=2, read_rate=1.0) -> dict:
        """
        Load data from files. Rewrite this function to realize data loading for your project. We suggest
        transform data to [torch.Tensor] and saving to a dict as the return value.
        Args:
            path: Path of data files
            s3_type: ['r6d', 'axis_angle']

        Returns:
            Dict of datas.
        """

        rot, seg_info = amass_read_seg(os.path.join(folder_path, 'rot.pt'), min_len=256, step=step, read_rate=read_rate, combo=combo)
        acc, _ = amass_read_seg(os.path.join(folder_path, 'acc.pt'), min_len=256, step=step, read_rate=read_rate, combo=combo)
        
        rot_gt, _ = amass_read_seg(os.path.join(folder_path, 'rot_gt.pt'), min_len=256, step=step, read_rate=read_rate, combo=combo)
        acc_gt, _ = amass_read_seg(os.path.join(folder_path, 'acc_gt.pt'), min_len=256, step=step, read_rate=read_rate, combo=combo)

        rot = rot.reshape(-1, config.imu_num, 3, 3)
        rot_gt = rot_gt.reshape(-1, config.imu_num, 3, 3)
        
        acc = torch.clamp(acc, min=-90, max=90).reshape(-1, config.imu_num, 3)
        acc = acc.unsqueeze(-1)
        acc_gt = torch.clamp(acc_gt, min=-90, max=90).reshape(-1, config.imu_num, 3)
        acc_gt = acc_gt.unsqueeze(-1)

        return {'rot': rot,
                'acc': acc,
                'rot_gt': rot_gt,
                'acc_gt': acc_gt,
                'seg_info': seg_info}