"""观测数据记录系统 - 记录机器人状态、相机流、物体元数据"""

import time
import numpy as np
import h5py
from collections import defaultdict


class SensorRecorder:
    """记录每次遥操作episode的所有观测数据"""

    def __init__(self, save_dir: str = './data/'):
        import os
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)
        self.reset()

    def reset(self):
        self.episode_data = {
            'joint_positions': [],
            'joint_velocities': [],
            'object_poses': [],         # {obj_name: [(pos, orn), ...]}
            'camera_rgb': [],           # 第三人称相机图像
            'head_camera_rgb': [],      # 第一人称相机图像
            'timestamps': [],
        }
        self.obj_pose_history = defaultdict(list)
        self.success = False
        self.completion_time = 0.0
        self.start_time = time.time()
        self.episode_id = 0

    def record_step(self, sample):
        if sample.joint_positions is not None:
            self.episode_data["joint_positions"].append(sample.joint_positions)
            self.episode_data["joint_velocities"].append(sample.joint_velocities)
        for name, pose in sample.object_poses.items(): self.obj_pose_history[name].append(pose)
        for name, camera in sample.cameras.items(): self.episode_data[name].append(camera)
        self.episode_data["timestamps"].append(sample.elapsed)


    def save_episode(self, episode_id: int = None):
        """保存当前episode到HDF5文件"""
        if episode_id is None:
            episode_id = self.episode_id
        
        filepath = f"{self.save_dir}/ep_{episode_id:04d}.h5"
        with h5py.File(filepath, 'w') as f:
            # 存储关节数据
            if self.episode_data['joint_positions']:
                f.create_dataset('joint_positions',
                                 data=np.array(self.episode_data['joint_positions']))
                f.create_dataset('joint_velocities',
                                 data=np.array(self.episode_data['joint_velocities']))

            # 存储时间戳
            if self.episode_data['timestamps']:
                f.create_dataset('timestamps',
                                 data=np.array(self.episode_data['timestamps']))

            # 存储物体轨迹
            obj_group = f.create_group('object_trajectories')
            for name, trajectory in self.obj_pose_history.items():
                positions = np.array([t[0] for t in trajectory])
                orientations = np.array([t[1] for t in trajectory])
                grp = obj_group.create_group(name)
                grp.create_dataset('positions', data=positions)
                grp.create_dataset('orientations', data=orientations)

            # 存储相机图像摘要（完整RGB太大，只存首尾帧）
            for cam_key in ['camera_rgb', 'head_camera_rgb']:
                if self.episode_data[cam_key]:
                    cam_group = f.create_group(cam_key)
                    first = self.episode_data[cam_key][0]
                    last = self.episode_data[cam_key][-1]
                    cam_group.create_dataset('first_rgb',
                                             data=np.array(first['rgb']))
                    cam_group.create_dataset('last_rgb',
                                             data=np.array(last['rgb']))

            # 元数据
            f.attrs['episode_id'] = episode_id
            f.attrs['success'] = self.success
            f.attrs['completion_time'] = self.completion_time
            f.attrs['num_steps'] = len(self.episode_data['timestamps'])

        print(f"Episode {episode_id} saved to {filepath}")

    def mark_success(self, completion_time: float):
        """标记任务成功"""
        self.success = True
        self.completion_time = completion_time
