import os
import gym
from gym import spaces
from gym.utils import seeding
import pybullet as p
import pybullet_data
import numpy as np
data_path = pybullet_data.getDataPath()
# # 'yumi'  'linker'  'shadow' 'svhhand'
hand_brand = 'linker'  
# hand_brand = 'svhhand'
# hand_brand = 'shadow'
# hand_brand = 'inspire'
if hand_brand == 'linker':
    robot_path = os.path.join('D:\\2026\\code\\TransHandR\\TransHandR\\dataset\\robot\\l21_right\\linkerhand_l21_right.urdf')
    tip_joints = ['index_tip', 'middle_tip', 'ring_tip', 'pinky_tip', 'thumb_tip']
elif hand_brand == 'shadow':
    robot_path = os.path.join('D:\\2026\\code\\TransHandR\\TransHandR\\dataset\\robot\\shadow_hand\\shadow_hand_right.urdf')
    tip_joints = ['THtip', 'FFtip', 'MFtip', 'RFtip', 'LFtip']
elif hand_brand == 'svhhand':
    robot_path = os.path.join('D:\\2026\\code\\TransHandR\\TransHandR\\dataset\\robot\\schunk_hand\\schunk_svh_hand_right.urdf')
elif hand_brand == 'inspire':
    robot_path = os.path.join('D:\\2026\\code\\TransHandR\\TransHandR\\dataset\\robot\\inspire_URDF\\urdf\\R_inspire.urdf')
class YumiEnv(gym.Env):
    """docstring for YumiEnv"""
    def __init__(self):
        super(YumiEnv, self).__init__()
        p.connect(p.GUI)
        p.resetDebugVisualizerCamera(cameraDistance=2.5, cameraYaw=90, cameraPitch=-20, cameraTargetPosition=[0,0,0.06])
        self.step_counter = 0
        if hand_brand == 'linker':
            self.joints = [
            'hand_base_link',
            'index_mcp_roll',
            'index_mcp_pitch',
            'index_pip',
            'middle_mcp_roll',
            'middle_mcp_pitch',
            'middle_pip',
            'ring_mcp_roll',
            'ring_mcp_pitch',
            'ring_pip',
            'pinky_mcp_roll',
            'pinky_mcp_pitch',
            'pinky_pip',
            'thumb_cmc_roll',
            'thumb_cmc_yaw',
            'thumb_cmc_pitch',
            'thumb_mcp',
            'thumb_ip',

            'index_tip',
            'middle_tip',
            'ring_tip',
            'pinky_tip',
            'thumb_tip'

                ]
        elif hand_brand == 'shadow':
            self.joints = [
                # 腕部关节
                'WRJ1', #0
                # 拇指关节
                'THJ5', 'THJ4', 'THJ3', 'THJ2', 'THJ1', #1-5
                # 食指关节
                'FFJ4', 'FFJ3', 'FFJ2', 'FFJ1', #6-9
                # 中指关节
                'MFJ4', 'MFJ3', 'MFJ2', 'MFJ1', #10-13
                # 无名指关节
                'RFJ4', 'RFJ3', 'RFJ2', 'RFJ1', #14-17
                # 小指关节
                'LFJ5', 'LFJ4', 'LFJ3', 'LFJ2', 'LFJ1', #18-22
                # 指尖固定关节
                'THtip', 'FFtip', 'MFtip', 'RFtip', 'LFtip' #23-27
            ]
        elif hand_brand == 'svhhand':
            self.joints = [
            'right_hand_f4',  #0
            # 手腕（虚拟关节）
            # 拇指关节
            'right_hand_Thumb_Opposition', #1
            'right_hand_Thumb_Flexion',
            'right_hand_j3',
            'right_hand_j4',
            # 食指关节
            'right_hand_index_spread', #5
            'right_hand_Index_Finger_Proximal', #6
            'right_hand_Index_Finger_Distal',
            'right_hand_j14',
            # 中指关节
            'right_hand_middle_spread_dummy', #9
            'right_hand_Middle_Finger_Proximal',
            'right_hand_Middle_Finger_Distal',
            'right_hand_j15',
            # 无名指和尾指掌面可动关节
            'right_hand_j5', # 13
            # 无名指关节
            'right_hand_ring_spread', #14
            'right_hand_Ring_Finger',
            'right_hand_j12',
            'right_hand_j16',

            # 小指关节
            'right_hand_Finger_Spread', #18
            'right_hand_Pinky',
            'right_hand_j13',
            'right_hand_j17', #21
            # 指尖关节 固定关节
            'thtip_joint', #22
            'fftip_joint',
            'mftip_joint',
            'rftip_joint',
            'lftip_joint' #26
        ]
        elif hand_brand == 'inspire':
            self.joints = [
            'R_base_link_joint',  # 0
            # 拇指关节
            'R_thumb_proximal_yaw_joint',  # 1
            'R_thumb_proximal_pitch_joint',  # 2
            'R_thumb_intermediate_joint',  # 3
            'R_thumb_distal_joint',  # 4
            # 食指关节
            'R_index_proximal_joint',  # 5
            'R_index_intermediate_joint',  # 6
            # 中指关节
            'R_middle_proximal_joint',  # 7
            'R_middle_intermediate_joint',  # 8
            # 无名指关节
            'R_ring_proximal_joint',  # 9
            'R_ring_intermediate_joint',  # 10
            # 小指关节
            'R_pinky_proximal_joint',  # 11
            'R_pinky_intermediate_joint'  # 12
        ]
        self.action_space = spaces.Box(np.array([-1]*len(self.joints)), np.array([1]*len(self.joints)))
        self.observation_space = spaces.Box(np.array([-1]*len(self.joints)), np.array([1]*len(self.joints)))

    def step(self, action, custom_reward=None):
        p.configureDebugVisualizer(p.COV_ENABLE_SINGLE_STEP_RENDERING)
        # print(self.joint2Index["yumi_joint_1_l"])
        # print(len(action))
        # print(self.joints)
        p.setJointMotorControlArray(self.yumiUid, [self.joint2Index[joint] for joint in self.joints], p.POSITION_CONTROL, action)

        p.stepSimulation()

        jointStates = {}
        for joint in self.joints:
            jointStates[joint] = p.getJointState(self.yumiUid, self.joint2Index[joint]) + p.getLinkState(self.yumiUid, self.joint2Index[joint])
        # recover color
        for joint, index in self.joint2Index.items():
            if joint in self.jointColor and joint != 'world_joint':
                p.changeVisualShape(self.yumiUid, index, rgbaColor=self.jointColor[joint])
        
        collision = False
        # 只检查指尖关节的碰撞
        for joint in tip_joints:
            if joint in self.joint2Index:  # 确保关节存在
                contact_points = p.getContactPoints(bodyA=self.yumiUid, linkIndexA=self.joint2Index[joint])
                if len(contact_points) > 0:
                    collision = True
                    # 如果需要可视化碰撞，可以保留这部分
                    for contact in contact_points:
                        p.changeVisualShape(self.yumiUid, contact[3], rgbaColor=[1,0,0,1])
                        p.changeVisualShape(self.yumiUid, contact[4], rgbaColor=[1,0,0,1])
                    break  # 发现指尖碰撞就退出
        #         for contact in p.getContactPoints(bodyA=self.yumiUid, linkIndexA=self.joint2Index[joint]):
        #             print("Collision Occurred in Joint {} & Joint {}!!!".format(contact[3], contact[4]))
        #             p.changeVisualShape(self.yumiUid, contact[3], rgbaColor=[1,0,0,1])
        #             p.changeVisualShape(self.yumiUid, contact[4], rgbaColor=[1,0,0,1])
        
        self.step_counter += 1

        if custom_reward is None:
            # default reward
            reward = 0
            done = False
        else:
            # custom reward
            reward, done = custom_reward(jointStates=jointStates, collision=collision, step_counter=self.step_counter)

        info = {'collision': collision}
        observation = [jointStates[joint][0] for joint in self.joints]
        return observation, reward, done, info

    def reset(self):
        k=0
        p.resetSimulation()
        self.step_counter = 0
        self.yumiUid = p.loadURDF(robot_path, [0,0,0.05],useFixedBase=True, flags=p.URDF_USE_SELF_COLLISION+p.URDF_USE_SELF_COLLISION_EXCLUDE_ALL_PARENTS)
        floor = p.loadURDF(pybullet_data.getDataPath() + '/plane.urdf', [0, 0, 0], p.getQuaternionFromEuler([0, 0, 0]))
        # self.tableUid = p.loadURDF(os.path.join(pybullet_data.getDataPath(),
        #     "table/table.urdf"), basePosition=[0,0,-0.65])
        p.setGravity(0,0,-10)
        p.setPhysicsEngineParameter(numSolverIterations=150)
        p.setTimeStep(1./240.)
        self.joint2Index = {} # jointIndex map to jointName
        for i in range(p.getNumJoints(self.yumiUid)):
            self.joint2Index[p.getJointInfo(self.yumiUid, i)[1].decode('utf-8')] = i
        # print(self.joint2Index)
        self.jointColor = {} # jointName map to jointColor

        # print(p.getVisualShapeData(self.yumiUid))
        for data in p.getVisualShapeData(self.yumiUid):
            k=k+1
            # print(p.getJointInfo(self.yumiUid, 0)[1].decode('utf-8'))
            if(k>=2):
                self.jointColor[p.getJointInfo(self.yumiUid, data[1])[1].decode('utf-8')] = data[7]

        # recover color
        for joint, index in self.joint2Index.items():
            if joint in self.jointColor and joint != 'world_joint':
                p.changeVisualShape(self.yumiUid, index, rgbaColor=self.jointColor[joint])


    def render(self, mode='human'):
        view_matrix = p.computeViewMatrixFromYawPitchRoll(cameraTargetPosition=[0.0,0.0,0.05],
                                                          distance=2.2,
                                                          yaw=90,
                                                          pitch=0,
                                                          roll=0,
                                                          upAxisIndex=2)
        proj_matrix = p.computeProjectionMatrixFOV(fov=60,
                                                   aspect=float(960)/720,
                                                   nearVal=0.1,
                                                   farVal=100.0)
        (_, _, px, _, _) = p.getCameraImage(width=960,
                                            height=720,
                                            viewMatrix=view_matrix,
                                            projectionMatrix=proj_matrix,
                                            renderer=p.ER_BULLET_HARDWARE_OPENGL)

        rgb_array = np.array(px, dtype=np.uint8)
        rgb_array = np.reshape(rgb_array, (720,960,4))

        rgb_array = rgb_array[:, :, :3]
        return rgb_array

    def close(self):
        p.disconnect()