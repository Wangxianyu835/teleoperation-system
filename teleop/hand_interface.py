"""手部遥操作接口 - 简化的手指开合动画

不再用 SDK 的电机映射算法（那是给真实电机用的）
直接用弧度值驱动 LinkerHand l7 的 17 个 UDP 关节
"""

import numpy as np


class HandInterface:
    """手指开合控制 - 纯弧度值，无 SDK 映射"""

    def __init__(self, mode: str = 'simulator'):
        self.mode = mode
        self._sim_t = 0.0

    def get_both_hands(self) -> dict:
        """返回双手 17 关节的弧度值"""
        # 缓慢正弦波: grasp = 0 (张开) → 1 (握紧)
        self._sim_t += 0.003  # 非常慢
        grasp = 0.4 + 0.3 * np.sin(self._sim_t * 0.8)  # 0.1 → 0.7

        joints = {
            # 拇指: 5 关节, 值范围 0 (张开) ~ 1.5 (弯曲)
            'thumb_cmc_roll':  0.0,
            'thumb_cmc_yaw':   0.3 * grasp,
            'thumb_cmc_pitch': 0.6 * grasp,
            'thumb_mcp':       0.5 * grasp,
            'thumb_ip':        0.4 * grasp,
            # 食指: 3 关节
            'index_mcp_pitch': 0.6 * grasp,
            'index_pip':       0.5 * grasp,
            'index_dip':       0.3 * grasp,
            # 中指: 3 关节
            'middle_mcp_pitch': 0.7 * grasp,
            'middle_pip':       0.5 * grasp,
            'middle_dip':       0.4 * grasp,
            # 无名指: 3 关节
            'ring_mcp_pitch':   0.6 * grasp,
            'ring_pip':         0.5 * grasp,
            'ring_dip':         0.3 * grasp,
            # 小指: 3 关节
            'pinky_mcp_pitch':  0.5 * grasp,
            'pinky_pip':        0.4 * grasp,
            'pinky_dip':        0.3 * grasp,
        }

        return {'left': joints, 'right': joints}

    def close(self):
        pass