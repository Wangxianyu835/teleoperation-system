
import os
from teleoperation.paths import ROBOTS_ROOT
import hashlib
from pathlib import Path

DEFAULT_ROBOTS_ROOT = str(ROBOTS_ROOT)
TRON2A_ARM_JOINTS = {
    "left": (
        "proximal_pitch_L_Joint", "proximal_roll_L_Joint", "proximal_yaw_L_Joint",
        "elbow_L_Joint", "wrist_yaw_L_Joint", "wrist_pitch_L_Joint", "wrist_roll_L_Joint",
    ),
    "right": (
        "proximal_pitch_R_Joint", "proximal_roll_R_Joint", "proximal_yaw_R_Joint",
        "elbow_R_Joint", "wrist_yaw_R_Joint", "wrist_pitch_R_Joint", "wrist_roll_R_Joint",
    ),
}
TRON2A_EE_LINKS = {"left": "grasper_L_Link", "right": "grasper_R_Link"}


ROBOT_SPECS = {
    'h1_2': {
        'urdf': os.path.join('h1_2', 'h1_2.urdf'),
        'base_z': 1.00,
        'desc': 'Unitree H1-2 (55关节, 双臂7DoF, 每手12关节)',
    },
    'gr1_t2': {
        'urdf': os.path.join('gr1', 'urdf', 'robot.urdf'),
        'base_z': 0.85,
        'desc': 'Fourier GR1-T2 (70关节, 双臂7DoF, 每手11关节)',
    },
    'g1': {
        'urdf': os.path.join('g1', 'g1_29dof_with_hand_lock_waist.urdf'),
        'base_z': 0.80,
        'desc': 'Unitree G1 (53关节, 双臂7DoF, 每手7关节)',
    },
}

# 判定「手臂关节」的关键词（三种机器人通用）
_ARM_KEYWORDS = ('shoulder', 'elbow', 'wrist')

# 手臂初始姿态（只对名字匹配的关节生效，其他机器人自动跳过）
ARM_NEUTRAL = {
    'left_shoulder_pitch_joint': -0.3,
    'left_elbow_pitch_joint': 1.2,
    'left_elbow_joint': 1.2,
    'right_shoulder_pitch_joint': -0.3,
    'right_elbow_pitch_joint': 1.2,
    'right_elbow_joint': 1.2,
}


def _is_arm_joint(name, side):
    """是否属于指定侧的手臂关节（含 shoulder / elbow / wrist）"""
    return name.startswith(side + '_') and any(k in name for k in _ARM_KEYWORDS)


def _is_hand_joint(name, side):
    """是否属于指定侧的灵巧手关节"""
    if side == 'left':
        return name.startswith('L_') or name.startswith('left_hand_')
    return name.startswith('R_') or name.startswith('right_hand_')


def arm_neutral_positions(robot_id, urdf_path=None):
    """Return model-specific references shared by scene loading and PINK.

    GR1's elbow direction was measured against this exact official asset.
    A modified GR1 model needs its own verified reference declaration.
    """
    values = dict(ARM_NEUTRAL)
    if robot_id == 'gr1_t2':
        path = Path(urdf_path or (Path(DEFAULT_ROBOTS_ROOT) / ROBOT_SPECS[robot_id]['urdf']))
        if hashlib.sha256(path.read_bytes()).hexdigest() != '5686e0189a9277c5fa01e67e5b0048581e1c7719fb37063f13e53e5a1d123ae0':
            raise ValueError('Neutral reference requires the verified GR1 URDF; verify custom elbow geometry first')
        for side in ('left', 'right'):
            values[side + '_elbow_pitch_joint'] = -1.2
    return values
