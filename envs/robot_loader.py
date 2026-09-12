"""加载 TeleOpBench 论文原版模型"""
import os, pybullet as p, numpy as np

H1_DIR = os.path.normpath(os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    '..', 'robots', 'from_teleopbench', 'h1_2'))
H1_URDF = os.path.join(H1_DIR, 'h1_2.urdf')

H1_ARM_NEUTRAL = {
    'left_shoulder_pitch_joint': -0.3,
    'left_elbow_pitch_joint': 1.2,
    'right_shoulder_pitch_joint': -0.3,
    'right_elbow_pitch_joint': 1.2,
}

class RobotLoader:
    def __init__(self, client):
        self.client = client
        self.robot_id = None
        self.all_joints = {}

    def load_robot(self):
        print(f"  加载 TeleOpBench 官方 H1-2 (55关节人形)...")
        p.setAdditionalSearchPath(H1_DIR, physicsClientId=self.client)
        self.robot_id = p.loadURDF(H1_URDF, [0, 0, 1.0],
                                   p.getQuaternionFromEuler([0, 0, 0]),
                                   useFixedBase=True,
                                   physicsClientId=self.client)
        for i in range(p.getNumJoints(self.robot_id, physicsClientId=self.client)):
            info = p.getJointInfo(self.robot_id, i, physicsClientId=self.client)
            if info[2] != p.JOINT_FIXED:
                name = info[1].decode() if isinstance(info[1], bytes) else info[1]
                self.all_joints[name] = i
                if name in H1_ARM_NEUTRAL:
                    p.resetJointState(self.robot_id, i, H1_ARM_NEUTRAL[name],
                                      physicsClientId=self.client)
        return {'robot': self.robot_id, 'left': self.robot_id, 'right': self.robot_id}

    def set_arm_pose(self, side, angles):
        prefix = 'left_' if side == 'left' else 'right_'
        for k, v in angles.items():
            jname = prefix + k
            if jname in self.all_joints:
                p.setJointMotorControl2(self.robot_id, self.all_joints[jname],
                                         p.POSITION_CONTROL, targetPosition=v,
                                         force=200, physicsClientId=self.client)