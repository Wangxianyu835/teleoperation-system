"""观测数据记录系统 - 记录机器人状态、相机流、物体元数据"""

import time
import numpy as np
import h5py
import pybullet as p
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

    def record_step(self, robot_id: int, task_objects: dict,
                    client_id: int, elapsed: float):
        """记录一步数据"""
        # 关节状态
        num_joints = p.getNumJoints(robot_id, physicsClientId=client_id)
        if num_joints > 0:
            joint_states = p.getJointStates(robot_id, range(num_joints),
                                            physicsClientId=client_id)
            self.episode_data['joint_positions'].append(
                [s[0] for s in joint_states])
            self.episode_data['joint_velocities'].append(
                [s[1] for s in joint_states])

        # 物体6D pose
        for name, obj_id in task_objects.items():
            pos, orn = p.getBasePositionAndOrientation(
                obj_id, physicsClientId=client_id)
            self.obj_pose_history[name].append((pos, orn))

        # 第三人称相机（320x240）
        self._record_camera('camera_rgb', client_id,
                            eye=[1.5, 0, 1.8],
                            target=[0.5, 0, 0.8])

        # 头部第一人称相机
        self._record_camera('head_camera_rgb', client_id,
                            eye=[0, 0, 1.5],
                            target=[0.5, 0, 0.8])

        self.episode_data['timestamps'].append(elapsed)

    def _record_camera(self, key: str, client_id: int,
                       eye: list, target: list):
        """记录相机图像"""
        width, height = 320, 240
        view_matrix = p.computeViewMatrix(
            cameraEyePosition=eye,
            cameraTargetPosition=target,
            cameraUpVector=[0, 0, 1],
            physicsClientId=client_id,
        )
        proj_matrix = p.computeProjectionMatrixFOV(
            fov=60, aspect=width / height,
            nearVal=0.1, farVal=10.0,
            physicsClientId=client_id,
        )
        try:
            img = p.getCameraImage(width, height, view_matrix, proj_matrix,
                                   physicsClientId=client_id)
            self.episode_data[key].append({
                'rgb': img[2],
                'depth': img[3],
                'seg': img[4],
            })
        except Exception:
            pass

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