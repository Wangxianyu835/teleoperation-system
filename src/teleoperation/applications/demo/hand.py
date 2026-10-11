"""手部遥操作接口 - 简化的手指开合动画

不再用 SDK 的电机映射算法（那是给真实电机用的）
直接用弧度值驱动 LinkerHand **L21** 的 17 个关节

★ 2026-09-12 型号统一（队友确认）：
    本文件原先用的是 **l7** 的关节名（`index_dip/middle_dip/ring_dip/pinky_dip`），
    但 **L21 没有 `*_dip` 关节**，且四指各有 1 个 `*_mcp_roll`（侧摆）——
    结果只能命中 13/17 个关节。现已统一为 L21。
    关节名与顺序以 `robots/l21.py` 的 `MAPPING_18` 为准。

L21 的 17 个可动关节（URDF 顺序）
---------------------------------
    index_mcp_roll, index_mcp_pitch, index_pip
    middle_mcp_roll, middle_mcp_pitch, middle_pip
    ring_mcp_roll,   ring_mcp_pitch,   ring_pip
    pinky_mcp_roll,  pinky_mcp_pitch,  pinky_pip
    thumb_cmc_roll, thumb_cmc_yaw, thumb_cmc_pitch, thumb_mcp, thumb_ip
"""

import numpy as np

# L21 的 17 个可动关节（URDF 顺序）—— 供本模块与其它代码统一引用
L21_JOINTS = (
    'index_mcp_roll', 'index_mcp_pitch', 'index_pip',
    'middle_mcp_roll', 'middle_mcp_pitch', 'middle_pip',
    'ring_mcp_roll', 'ring_mcp_pitch', 'ring_pip',
    'pinky_mcp_roll', 'pinky_mcp_pitch', 'pinky_pip',
    'thumb_cmc_roll', 'thumb_cmc_yaw', 'thumb_cmc_pitch',
    'thumb_mcp', 'thumb_ip',
)


class HandInterface:
    """手指开合控制 - 纯弧度值，无 SDK 映射（目标手：LinkerHand L21）"""

    def __init__(self, mode: str = 'simulator'):
        self.mode = mode
        self._sim_t = 0.0

    def get_both_hands(self) -> dict:
        """返回双手 17 关节的弧度值（L21 关节名，与 L21_JOINTS 一致）"""
        # 缓慢正弦波: grasp = 0 (张开) -> 1 (握紧)
        self._sim_t += 0.003  # 非常慢
        grasp = 0.4 + 0.3 * np.sin(self._sim_t * 0.8)  # 0.1 -> 0.7

        joints = {
            # 拇指: 5 关节
            'thumb_cmc_roll':  0.0,
            'thumb_cmc_yaw':   0.3 * grasp,
            'thumb_cmc_pitch': 0.6 * grasp,
            'thumb_mcp':       0.5 * grasp,
            'thumb_ip':        0.4 * grasp,
            # 食指: 3 关节（L21 = mcp_roll / mcp_pitch / pip，无 dip）
            'index_mcp_roll':  0.0,
            'index_mcp_pitch': 0.6 * grasp,
            'index_pip':       0.5 * grasp,
            # 中指: 3 关节
            'middle_mcp_roll':  0.0,
            'middle_mcp_pitch': 0.7 * grasp,
            'middle_pip':       0.5 * grasp,
            # 无名指: 3 关节
            'ring_mcp_roll':   0.0,
            'ring_mcp_pitch':  0.6 * grasp,
            'ring_pip':        0.5 * grasp,
            # 小指: 3 关节
            'pinky_mcp_roll':  0.0,
            'pinky_mcp_pitch': 0.5 * grasp,
            'pinky_pip':       0.4 * grasp,
        }

        return {'left': joints, 'right': joints}

    def apply_to_robot(self, robot_id, joints: dict, client: int = 0,
                       force: float = 20.0) -> int:
        """把 {关节名: 目标角度} 应用到 pybullet 机器人上（按【名字】匹配）

        ★ 2026-09-12 新增：`teleop_pipeline.py` 一直在调用本方法，
          但 `HandInterface` 里根本没有它（潜在 AttributeError）。

        Args:
            robot_id: pybullet body id
            joints: {'index_mcp_roll': 0.1, ...}（L21 关节名）
            client: physicsClientId
            force: 关节电机力矩上限

        Returns:
            int: 实际应用成功的关节数

        注意：本方法按名字匹配。**如果是 H1-2 / GR1-T2 / G1 自带的原装手**
        （关节名形如 `L_index_proximal_joint`），L21 名字一个也对不上，
        会返回 0 —— 那种情况请改用「把 L21 挂到机器人腕部」的方案：
            applications/replay/replay_hand_on_robot.py
        """
        import pybullet as p

        name2idx = {}
        for i in range(p.getNumJoints(robot_id, physicsClientId=client)):
            info = p.getJointInfo(robot_id, i, physicsClientId=client)
            nm = info[1].decode() if isinstance(info[1], bytes) else info[1]
            name2idx[nm] = i

        applied = 0
        for jname, val in joints.items():
            j = name2idx.get(jname)
            if j is None:
                continue
            p.setJointMotorControl2(
                robot_id, j, p.POSITION_CONTROL,
                targetPosition=float(val), force=force,
                physicsClientId=client,
            )
            applied += 1
        return applied

    def close(self):
        pass
