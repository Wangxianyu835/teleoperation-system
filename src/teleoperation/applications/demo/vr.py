"""VR 遥操作接口 - Apple Vision Pro 手腕追踪 -> 机器人关节角度

论文 Section 3.2.3:
  Vision Pro 追踪 -> OpenXR 坐标转换 -> 机器人骨盆坐标系
  -> PINK 逆运动学 -> 双臂关节角度
"""

import numpy as np
import pybullet as p


class VRInterface:
    """VR 遥操作接口：接收手腕/头部位姿，输出机器人关节角度"""

    def __init__(self, physics_client_id: int):
        self.client = physics_client_id

        # 坐标系变换参数
        # 操作者骨盆 -> 机器人 pelvis 的缩放因子
        self.body_scale = 1.0

        # 手动设置 VR 数据（无真实设备时使用）
        self._mock_head_pos = np.array([0.0, 0.0, 1.6])
        self._mock_head_orn = np.array([0.0, 0.0, 0.0, 1.0])

        # 默认手腕位置（机器人坐标系下，相对于 pelvis）
        self._left_wrist_pos = np.array([-0.25, 0.3, 1.1])
        self._right_wrist_pos = np.array([0.25, 0.3, 1.1])

    def receive_vr_data(self):
        """
        接收 Vision Pro 的 VR 追踪数据
        实际使用时要替换为:
          - Apple Vision Pro API (ARKit hand tracking)
          - 或 OpenXR 标准接口

        Returns:
            dict: {
                'head_pos', 'head_quat',
                'left_wrist_pos', 'left_wrist_quat',
                'right_wrist_pos', 'right_wrist_quat',
                'left_finger_joints': (5, 4, 3),  # 手指关键点
                'right_finger_joints': (5, 4, 3),
            }
        """
        # TODO: 替换为真实 Vision Pro 数据
        return {
            'head_pos': self._mock_head_pos,
            'head_quat': self._mock_head_orn,
            'left_wrist_pos': self._left_wrist_pos,
            'left_wrist_quat': np.array([0.0, 0.0, 0.0, 1.0]),
            'right_wrist_pos': self._right_wrist_pos,
            'right_wrist_quat': np.array([0.0, 0.0, 0.0, 1.0]),
        }

    def transform_to_robot_frame(self, vr_data: dict,
                                  robot_pelvis_pos: np.ndarray) -> dict:
        """
        将 VR 坐标系数据转换到机器人骨盆坐标系
        Section 3.2.3: 手腕相对于头部 -> 相对于骨盆
        """
        head_pos = vr_data['head_pos']
        scale = self.body_scale

        # 手腕相对于头部的偏移
        l_wrist_rel = vr_data['left_wrist_pos'] - head_pos
        r_wrist_rel = vr_data['right_wrist_pos'] - head_pos

        # 缩放并偏移到机器人骨盆坐标系
        transformed = {
            'left_wrist_target': robot_pelvis_pos + l_wrist_rel * scale,
            'right_wrist_target': robot_pelvis_pos + r_wrist_rel * scale,
        }
        return transformed

    def solve_ik_for_arms(self, left_wrist_target: np.ndarray,
                           right_wrist_target: np.ndarray,
                           robot_id: int,
                           left_arm_indices: list,
                           right_arm_indices: list,
                           left_ee_index: int,
                           right_ee_index: int) -> dict:
        """
        用逆运动学求解双臂关节角度
        (当前用 PyBullet IK，Isaac Sim 中用 PINK)

        Args:
            left_arm_indices: 左臂关节整数索引 [j1, j2, ...] (不是名字!)
            right_arm_indices: 右臂关节整数索引
        """
        result = {}

        left_joints = p.calculateInverseKinematics(
            robot_id, left_ee_index,
            targetPosition=left_wrist_target.tolist(),
            physicsClientId=self.client
        )
        result['left_joints'] = [left_joints[i] for i in left_arm_indices]

        right_joints = p.calculateInverseKinematics(
            robot_id, right_ee_index,
            targetPosition=right_wrist_target.tolist(),
            physicsClientId=self.client
        )
        result['right_joints'] = [right_joints[i] for i in right_arm_indices]

        return result

    def process_frame(self, robot_id: int,
                      left_arm_indices: list, right_arm_indices: list,
                      left_ee_idx: int, right_ee_idx: int,
                      pelvis_pos: np.ndarray) -> dict:
        """
        处理一帧 VR 数据，输出完整的机器人关节指令

        Returns:
            dict: {'left_arm', 'right_arm', 'vr_data'}
        """
        # 1. 接收 VR 数据
        vr_data = self.receive_vr_data()

        # 2. 坐标转换
        targets = self.transform_to_robot_frame(vr_data, pelvis_pos)

        # 3. IK 求解
        joint_solution = self.solve_ik_for_arms(
            targets['left_wrist_target'],
            targets['right_wrist_target'],
            robot_id,
            left_arm_indices,
            right_arm_indices,
            left_ee_idx,
            right_ee_idx,
        )

        return {
            'left_arm': joint_solution['left_joints'],
            'right_arm': joint_solution['right_joints'],
            'vr_data': vr_data,
        }
