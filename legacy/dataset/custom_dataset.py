# tensorboard --logdir D:\2026\code\TransHandR\TransHandR\checkpoint\logs\experiment_20260129_194546

import numpy as np
import copy
from dataset.skeleton import Skeleton #相对导入
from dataset.mocap_dataset import MocapDataset

import h5py
import torch.utils.data as data
from dataset.utils import deterministic_random
from dataset.generators import ChunkedGenerator

linkhand_skeleton = Skeleton(
    parents=[
        -1,  # 0 wrist
        0, 1, 2, 3,          # thumb: mcp->pip->dip->tip
        0, 5, 6, 7, 8,         # index
        0, 10, 11, 12, 13,       # middle
        0, 15, 16, 17, 18,     # ring
        0, 20, 21, 22, 23        # little
    ],
    joints_left=[],
    joints_right=list(range(25)),
    hand_type='right'
)

class linkhand_onehand_dataset(MocapDataset):
    def __init__(self, path, data_hand_type='right', scaling_factor=1.0, opt=None, remove_static_joints=True):
        super().__init__(fps=50, skeleton=linkhand_skeleton)
        # 训练和测试集划分暂未使用
        self.train_list = []
        self.test_list = []
        # 设置相机参数为空（动捕数据不需要相机）
        self._cameras = {}
        # 加载数据（假设h5文件格式与Human36m类似）
        # 如果h5文件格式不同，需要相应调整
        self._data = {}
        self.scaling_factor = scaling_factor
        self.hand_type = data_hand_type
        with h5py.File(path, 'r') as f:
        #     根据实际h5文件结构调整数据读取方式
        #     数据放在h5文件按库名分库存储
        #     例如库名有‘测试-ceshi’
        #     数据分键值防止,格式为: frames, l_glove_pos, r_glove_pos, 其中
        #     frames_ids: (num_frames,1)
        #     l_glove_pos: (num_frames,25,3)
        #     r_glove_pos: (num_frames,25,3)
            data = {}
            for subject in f.keys():
                data[subject] = {}
                
                r_positions = f[subject]['r_glove_pos'][:]  # 读取3D位置数据
                #如果有索引数据，则读取索引数据  
                # frames_ids = f[subject]['frame_ids'][:]
                # 如果没有这个键值，则生成默认索引
                try :
                    frames_ids = f[subject]['frame_ids'][:]
                except KeyError:
                    print("没有索引数据，生成默认索引")
                    frames_ids = np.arange(r_positions.shape[0]).reshape(-1, 1)                  
                data[subject] = {
                    'frame_ids': frames_ids,
                    'r_positions': r_positions,
                    'l_positions': [],  # 空列表，因为不需要左手数据
                    'cameras': [],  # 空列表，因为不需要相机
                }
                self._data[subject] = data[subject]
        
    def check_alignment(self):
        #检查数据中的索引长度是否对齐
        for subject in self._data.keys():
            f_ids_len = len(self._data[subject]['frame_ids'])
            r_pos_len = self._data[subject]['r_positions'].shape[0]
            if f_ids_len != r_pos_len:
                raise ValueError(f"Data length mismatch for subject {subject}: frames_ids length {f_ids_len} != r_positions length {r_pos_len}")
            else:
                print(f"数据长度对齐成功")
                print(f"数据长度: {f_ids_len}")
                return True
    def all_keys(self):
        '''
        输出所有key,返回一个列表,包含所有key
        '''
        keys = []
        for subject in self._data.keys():
           print(f"数据集包含的subject: {subject}")
           keys.append(subject)
        return keys

    def supports_semi_supervised(self):
        return True
    def fetch(self,subjects, action_filter=None, subset=1, parse_3d_poses=True , downsample=1):
        out_poses_3d = []
        keypoints = self._data
        
        for subject in subjects:
        # for subject in self._data.keys():
            print(f"正在处理数据: {subject}")
            if subject not in keypoints: continue
            out_poses_3d.append(self._data[subject]['r_positions']*self.scaling_factor)
        if len(out_poses_3d) == 0:
            out_poses_3d = None
            print("没有3D数据")
        else:
            #输出数据形状
            #数组列表 每个元素是一个数组
            size = [len(out_poses_3d), out_poses_3d[0].shape[0],out_poses_3d[0].shape[1], out_poses_3d[0].shape[2]]
            print(f"fetch数据格式: {size}")
        return out_poses_3d

    def frames_sum(self):
        total_frames = 0
        for subject in self._data.keys():
            total_frames += self._data[subject]['r_positions'].shape[0]
        return total_frames
        

# 测试代码
if __name__ == '__main__':
    dataset_path = 'D:\\2026\\code\\TransHandR\\TransHandR\\dataset\\data\\h5\\sign_glove.h5'
    hand_type = 'right'
    dataset = linkhand_onehand_dataset(dataset_path, data_hand_type=hand_type)
    #数据筛选器
    action_filter = None
    # subjects_test = ['测试-ceshi']
    # subjects_train = ['训练-train']
    keys = dataset.all_keys()
    # subjects_train = ['测试-ceshi']
    # subjects_train = ['S000001_P0000']
    subjects_train = keys
    #输出数据集数据
    # print(dataset.all_keys())
    poses_valid = dataset.fetch(subjects_train, action_filter)
    print(poses_valid.__len__())
    print(dataset.frames_sum())