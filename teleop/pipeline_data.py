"""
数据接口定义 — 三人协作的数据格式

完整流程:
  1. 王健杰: 摄像头采集 -> CameraFrame (RGB数组)
  2. 肖奕阳: 重定向算法 -> RobotJointCommand (关节角度)
  3. 王宪雨: 仿真环境接收 RobotJointCommand -> 驱动 H1-2

每个模块独立开发, 通过标准 Numpy 数组传递数据
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional


# ============================================================
# 接口1: 摄像头 -> 重定向
# 王健杰采集的 RGB 帧, 传给肖奕阳
# ============================================================
@dataclass
class CameraFrame:
    """摄像头采集的一帧数据"""
    rgb: np.ndarray          # (480, 640, 3) RGB图像, uint8
    timestamp: float         # 时间戳 (秒)
    frame_id: int            # 帧序号

    def save(self, filepath: str):
        """保存为 .npy 文件供后续读取"""
        np.savez(filepath, rgb=self.rgb, timestamp=self.timestamp,
                 frame_id=self.frame_id)

    @classmethod
    def load(cls, filepath: str) -> 'CameraFrame':
        """从 .npy 文件加载"""
        data = np.load(filepath, allow_pickle=True)
        return cls(rgb=data['rgb'], timestamp=float(data['timestamp']),
                   frame_id=int(data['frame_id']))


# ============================================================
# 接口2: 重定向 -> 仿真
# 肖奕阳输出的 H1-2 关节角度, 传给王宪雨
# ============================================================
@dataclass
class RobotJointCommand:
    """
    重定向算法输出的全部机器人关节角度

    数据格式是标准 Python dict, 便于跨模块传递:
    {
        'left_arm':  [j1, j2, ..., j7],     # 左臂7个关节 (弧度)
        'right_arm': [j1, j2, ..., j7],     # 右臂7个关节
        'left_hand': {joint_name: angle},     # 左手手指
        'right_hand': {joint_name: angle},    # 右手手指
    }

    H1-2 关节名映射 (论文原版):
      左/右臂: shoulder_pitch_joint, shoulder_roll_joint, shoulder_yaw_joint,
               elbow_pitch_joint, elbow_roll_joint,
               wrist_pitch_joint, wrist_yaw_joint
      左/右手: thumb_proximal_yaw_joint, thumb_proximal_pitch_joint,
               thumb_intermediate_joint, thumb_distal_joint,
               index_proximal_joint, index_intermediate_joint,
               middle_proximal_joint, middle_intermediate_joint,
               ring_proximal_joint, ring_intermediate_joint,
               pinky_proximal_joint, pinky_intermediate_joint
    """
    left_arm: np.ndarray = field(default_factory=lambda: np.zeros(7))
    right_arm: np.ndarray = field(default_factory=lambda: np.zeros(7))
    left_hand: Dict[str, float] = field(default_factory=dict)
    right_hand: Dict[str, float] = field(default_factory=dict)
    timestamp: float = 0.0

    def to_dict(self) -> dict:
        return {
            'left_arm': self.left_arm,
            'right_arm': self.right_arm,
            'left_hand': self.left_hand,
            'right_hand': self.right_hand,
            'timestamp': self.timestamp,
        }

    def save(self, filepath: str):
        """保存为 .npz 文件供仿真读取"""
        np.savez(filepath,
                 left_arm=self.left_arm,
                 right_arm=self.right_arm,
                 timestamp=self.timestamp)

    @classmethod
    def load(cls, filepath: str) -> 'RobotJointCommand':
        """从 .npz 文件加载"""
        data = np.load(filepath, allow_pickle=True)
        return cls(
            left_arm=data['left_arm'],
            right_arm=data['right_arm'],
            timestamp=float(data['timestamp']),
        )

    @classmethod
    def mock_demo(cls, t: float) -> 'RobotJointCommand':
        """生成模拟的关节指令 (用于演示和测试, 等重定向算法完成后替换)"""
        a = 0.3 * np.sin(t * 0.5)
        left_arm = np.array([
            -0.3 + a,      # shoulder_pitch
            0.5,           # shoulder_roll
            0.0,           # shoulder_yaw
            1.2 + a * 0.5, # elbow_pitch
            0.0,           # elbow_roll
            0.0,           # wrist_pitch
            0.0,           # wrist_yaw
        ])
        right_arm = np.array([
            -0.3 - a,      # shoulder_pitch
            -0.5,          # shoulder_roll
            0.0,           # shoulder_yaw
            1.2 - a * 0.5, # elbow_pitch
            0.0,           # elbow_roll
            0.0,           # wrist_pitch
            0.0,           # wrist_yaw
        ])
        return cls(left_arm=left_arm, right_arm=right_arm, timestamp=t)


# ============================================================
# 接口3: 仿真 -> 数据记录
# 王宪雨记录每次遥操作 episode 的完整数据
# ============================================================
@dataclass
class TeleopEpisode:
    """
    一次完整遥操作的数据记录

    等价于论文 Table 2 中的一行测试数据
    """
    task_name: str
    robot_type: str
    success: bool
    completion_time: float
    num_steps: int
    camera_frames: List[CameraFrame] = field(default_factory=list)
    joint_commands: List[RobotJointCommand] = field(default_factory=list)