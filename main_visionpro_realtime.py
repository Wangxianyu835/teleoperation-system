'''
多进程版本：将Vision Pro手部跟踪数据转换为虚拟环境和真实机器手驱动
使用四个进程：数据获取进程、重定向进程、虚拟环境进程、真实机器手进程
'''
# 添加项目根目录到 Python 路径
import os
import sys

import multiprocessing
import gym, yumi_gym
import pybullet as p
import numpy as np
import h5py
import time
import math
import yaml

import can
import torch
import torch.nn.functional as F
# import keyboard
from avp_stream import VisionProStreamer

from LinkerHand.linker_hand_api import LinkerHandApi
from LinkerHand.utils.load_write_yaml import LoadWriteYaml
from LinkerHand.utils.color_msg import ColorMsg

from model.model_poseformer import PoseTransformer
from model.angle2real import create_hand_kinematics

data_tpye = 'visionpro'
# data_tpye = 'slahmr'
#### 手型配置选择
hand_brand = 'linker'  
# # 'yumi'  'linker'  'shadow' 'svhhand'
# hand_brand = 'svhhand'
# hand_brand = 'shadow'

if data_tpye == 'visionpro':
    # scaling_factor = 1.0/0.061
    scaling_factor = 1.0
        # 记录原始数据的特定关节索引
    # 顺序是 大拇指 , 食指 , 中指 , 无名指 , 小指
    TIP_dic = [4, 9, 14, 19, 24] # 指尖
    DIP_dic = [3, 8, 13, 18, 23] # 远端  
    PIP_dic = [2, 7, 12, 17, 22] # 近端
    MCP_dic = [1, 6, 11, 16, 21] # 掌指
    PALM_dic = [1,5, 10, 15, 20] # 手掌根部
    source_dic = {'TIP_dic':TIP_dic, 'DIP_dic':DIP_dic, 'PIP_dic':PIP_dic, 'MCP_dic':MCP_dic, 'PALM_dic':PALM_dic}
    num_joints = 25
    hand_connections = [
        # 手掌连接
        (0, 1), (1, 2), (2, 3), (3, 4),  # 拇指
        (0, 5), (5, 6), (6, 7), (7, 8), (8, 9), # 食指
        (0, 10), (10, 11), (11, 12), (12, 13), (13, 14), # 中指
        (0, 15), (15, 16), (16, 17), (17, 18), (18, 19), # 无名指
        (0, 20), (20, 21), (21, 22), (22, 23), (23, 24) # 小指
    ]
    keypoints ='glove_data_aligned'
elif data_tpye == 'slahmr':
    # 大拇指  食指   中指   无名指   小指
    TIP_dic = [15, 3, 6, 12, 9] # 指尖
    DIP_dic = None
    PIP_dic = [14, 2, 5, 11, 8] # 近端
    MCP_dic = [13, 1, 4, 10, 7] # 掌指端
    PALM_dic = None # 手掌根部
    source_dic = {'TIP_dic':TIP_dic, 'DIP_dic':DIP_dic, 'PIP_dic':PIP_dic, 'MCP_dic':MCP_dic, 'PALM_dic':PALM_dic}
    hand_connections = [
        (0,13), (13,14), (14,15), # 大拇指
        (0, 1), (1, 2),  (2, 3), # 食指
        (0, 4), (4, 5),  (5, 6), # 中指
        (0, 10), (10, 11), (11, 12), # 无名指
        (0, 7), (7, 8),  (8, 9) # 小指
    ]
    num_joints = 16
    scaling_factor = 1.0/0.0647
    keypoints ='sign_glove_aligned'
    # 训练集键列表
    train_keys = [
        'S000018_P0004',
        'S000059_P0008',
        'S000042_P0004',
        'S000099_P0000',
        'S000099_P0004',
        'S000068_P0000',
        'S000098_P0004',
        'S000024_P0000',
        'S000117_P0000',
        'S000114_P0008',
        'S000085_P0008',
        'S000123_P0004',
        'S000108_P0000',
        'S000043_P0004',
        'S000078_P0004',
        'S000117_P0008',
        'S000109_P0000',
        'S000071_P0000',
        'S000054_P0008',
        'S000033_P0000',
        'S000102_P0008',
        'S000090_P0000',
        'S000024_P0004',
        'S000023_P0000',
        'S000119_P0008',
        'S000006_P0004',
        'S000015_P0004',
        'S000104_P0000',
        'S000056_P0008',
        'S000016_P0000',
        'S000058_P0000',
        'S000036_P0008',
        'S000065_P0008',
        'S000103_P0008',
        'S000097_P0000',
        'S000005_P0000',
        'S000120_P0000',
        'S000035_P0008',
        'S000061_P0004',
        'S000061_P0000',
        'S000016_P0004',
        'S000063_P0008',
        'S000114_P0004',
        'S000014_P0004',
        'S000040_P0008',
        'S000070_P0004',
        'S000046_P0008',
        'S000082_P0004',
        'S000047_P0008',
        'S000074_P0000',
        'S000010_P0008',
        'S000073_P0008',
        'S000109_P0008',
        'S000116_P0000',
        'S000058_P0008',
        'S000019_P0004',
        'S000086_P0000',
        'S000069_P0008',
        'S000091_P0004',
        'S000020_P0008',
        'S000012_P0004',
        'S000040_P0000',
        'S000084_P0004',
        'S000006_P0008',
        'S000115_P0004',
        'S000035_P0000',
        'S000068_P0004',
        'S000078_P0008',
        'S000085_P0004',
        'S000039_P0004',
        'S000014_P0000',
        'S000051_P0004',
        'S000090_P0004',
        'S000057_P0008',
        'S000081_P0000',
        'S000045_P0000',
        'S000028_P0004',
        'S000049_P0000',
        'S000094_P0000',
        'S000021_P0000',
        'S000083_P0004',
        'S000044_P0000',
        'S000116_P0004',
        'S000047_P0004',
        'S000093_P0004',
        'S000064_P0000',
        'S000063_P0004',
        'S000042_P0000',
        'S000084_P0008',
        'S000057_P0004',
        'S000108_P0008',
        'S000118_P0008',
        'S000119_P0000',
        'S000081_P0004',
        'S000070_P0008',
        'S000074_P0008',
        'S000037_P0004',
        'S000001_P0008',
        'S000040_P0004',
        'S000118_P0000',
        'S000069_P0004',
        'S000089_P0004',
        'S000048_P0008',
        'S000056_P0004',
        'S000069_P0000',
        'S000082_P0000',
        'S000100_P0004',
        'S000039_P0008',
        'S000060_P0004',
        'S000018_P0000',
        'S000100_P0008',
        'S000089_P0000',
        'S000110_P0004',
        'S000063_P0000',
        'S000077_P0004',
        'S000004_P0008',
        'S000059_P0004',
        'S000028_P0008',
        'S000072_P0008',
        'S000082_P0008',
        'S000067_P0004',
        'S000034_P0008',
        'S000075_P0004',
        'S000044_P0004',
        'S000058_P0004',
        'S000114_P0000',
        'S000098_P0000',
        'S000018_P0008',
        'S000008_P0004',
        'S000020_P0000',
        'S000017_P0008',
        'S000116_P0008',
        'S000075_P0000',
        'S000026_P0004',
        'S000048_P0004',
        'S000113_P0008',
        'S000084_P0000',
        'S000022_P0008',
        'S000101_P0008',
        'S000083_P0000',
        'S000067_P0000',
        'S000016_P0008',
        'S000120_P0004',
        'S000068_P0008',
        'S000088_P0004',
        'S000001_P0000',
        'S000109_P0004',
        'S000051_P0000',
        'S000097_P0004',
        'S000095_P0000',
        'S000049_P0004',
        'S000020_P0004',
        'S000033_P0004',
        'S000072_P0000',
        'S000088_P0000',
        'S000096_P0000',
        'S000008_P0000',
        'S000121_P0008',
        'S000108_P0004',
        'S000017_P0000',
        'S000083_P0008',
        'S000004_P0004',
        'S000062_P0000',
        'S000019_P0000',
        'S000077_P0008',
        'S000076_P0004',
        'S000079_P0004',
        'S000112_P0008',
        'S000067_P0008',
        'S000033_P0008',
        'S000066_P0008',
        'S000064_P0004',
        'S000119_P0004',
        'S000102_P0000',
        'S000037_P0008',
        'S000103_P0000',
        'S000034_P0004',
        'S000023_P0008',
        'S000121_P0000',
        'S000074_P0004',
        'S000022_P0000',
        'S000062_P0008',
        'S000073_P0004',
        'S000015_P0008',
        'S000113_P0004',
        'S000100_P0000',
        'S000043_P0000',
        'S000021_P0008',
        'S000112_P0004',
        'S000024_P0008',
        'S000048_P0000'
    ]
    # 测试集键列表
    test_keys = [
        'S000102_P0004',
        'S000111_P0000',
        'S000076_P0000',
        'S000023_P0004',
        'S000070_P0000',
        'S000087_P0004',
        'S000093_P0000',
        'S000004_P0000',
        'S000055_P0000',
        'S000027_P0004',
        'S000066_P0000',
        'S000001_P0004',
        'S000103_P0004',
        'S000054_P0000',
        'S000025_P0008',
        'S000077_P0000',
        'S000042_P0008',
        'S000095_P0004',
        'S000019_P0008',
        'S000111_P0004',
        'S000005_P0008',
        'S000079_P0008',
        'S000091_P0000',
        'S000113_P0000',
        'S000057_P0000',
        'S000047_P0000',
        'S000043_P0008',
        'S000064_P0008',
        'S000026_P0008',
        'S000046_P0004',
        'S000006_P0000',
        'S000010_P0000',
        'S000032_P0008',
        'S000115_P0000',
        'S000065_P0004',
        'S000012_P0008',
        'S000078_P0000',
        'S000039_P0000',
        'S000120_P0008',
        'S000066_P0004',
        'S000110_P0008',
        'S000104_P0004',
        'S000118_P0004',
        'S000034_P0000',
        'S000027_P0008',
        'S000055_P0008',
        'S000026_P0000',
        'S000111_P0008',
        'S000081_P0008',
        'S000112_P0000',
        'S000087_P0000',
        'S000010_P0004',
        'S000096_P0004',
        'S000086_P0004',
        'S000101_P0004',
        'S000079_P0000',
        'S000032_P0004',
        'S000101_P0000',
        'S000025_P0004',
        'S000046_P0000',
        'S000110_P0000',
        'S000115_P0008',
        'S000015_P0000',
        'S000031_P0008',
        'S000094_P0004',
        'S000075_P0008',
        'S000087_P0008',
        'S000117_P0004',
        'S000071_P0008',
        'S000065_P0000',
        'S000072_P0004',
        'S000049_P0008',
        'S000045_P0008',
        'S000099_P0008',
        'S000061_P0008',
        'S000123_P0008',
        'S000014_P0008',
        'S000086_P0008',
        'S000038_P0008',
        'S000055_P0004',
        'S000123_P0000',
        'S000051_P0008'
    ]

if hand_brand == 'yumi':
    hand_cfg = {
        'joints_name': [
            'yumi_link_7_r_joint',
            'Link1',
            'Link11',
            'R_ring_tip_joint',

            'Link2',
            'Link22',
            'R_middle_tip_joint',

            'Link3',
            'Link33',
            'R_index_tip_joint',

            'Link4',
            'Link44',
            'R_pinky_tip_joint',

            'Link5',
            'Link51',
            'Link52',
            'Link53',
            'R_thumb_tip_joint',
        ],
        'edges': [
            ['yumi_link_7_r_joint', 'Link1'],
            ['Link1', 'Link11'],
            ['Link11', 'R_ring_tip_joint'],
            ['yumi_link_7_r_joint', 'Link2'],
            ['Link2', 'Link22'],
            ['Link22', 'R_middle_tip_joint'],
            ['yumi_link_7_r_joint', 'Link3'],
            ['Link3', 'Link33'],
            ['Link33', 'R_index_tip_joint'],
            ['yumi_link_7_r_joint', 'Link4'],
            ['Link4', 'Link44'],
            ['Link44', 'R_pinky_tip_joint'],
            ['yumi_link_7_r_joint', 'Link5'],
            ['Link5', 'Link51'],
            ['Link51', 'Link52'],
            ['Link52', 'Link53'],
            ['Link53', 'R_thumb_tip_joint'],
        ],
        'root_name': 'yumi_link_7_r_joint',
        'end_effectors': [
            'R_index_tip_joint',
            'R_middle_tip_joint',
            'R_ring_tip_joint',
            'R_pinky_tip_joint',
            'R_thumb_tip_joint',
        ],
        # 'end_effectors': [
        #     'Link11',
        #     'Link22',
        #     'Link33',
        #     'Link44',
        #     'Link53',
        # ],
        'elbows': [
            'Link1',
            'Link2',
            'Link3',
            'Link4',
            'Link5',
        ],
    }
    urdf_file = "D:\\2026\\code\\TransHandR\\dataset\\robot\\ur3\\robot(ur3).urdf"

elif hand_brand == 'linker':
    urdf_file = "D:\\2026\\code\\TransHandR\\TransHandR\\dataset\\robot\\l21_right\\linkerhand_l21_right.urdf"
    excluded_pairs=[(1, 2), (4, 5), (7, 8), (10, 11), (14, 15)]
    TIP_dic_rb_gym = [22, 4, 8 , 12, 16]
    # 记录机器手的特定关节索引
    TIP_dic_rb = [22, 18, 19 , 20, 21]
    DIP_dic_rb = [17, 3, 6, 9, 12]
    PIP_dic_rb = [15, 2, 5, 8, 11]
    MCP_dic_rb = [14, 1, 4, 7, 10]
    rb_dic = {'TIP_dic':TIP_dic_rb, 'DIP_dic':DIP_dic_rb, 'PIP_dic':PIP_dic_rb, 'MCP_dic':MCP_dic_rb}
    # 关节映射字典 实机使用的关节索引
    joint_map = {0: 15, 1: 2, 2: 5, 3: 8, 4: 11, 5: 14, 6: 1, 7: 4, 8: 7, 9: 10, 10: 13, 
                11: 0, 12: 0, 13: 0, 14: 0, 15: 16, 16: 0, 17: 0, 18: 0, 19: 0,
                20: 17, 21: 3, 22: 6, 23: 9, 24: 12}
    # scaling_factor_rb = 1.0/0.064
    scaling_factor_rb = 1.0
    out_num_joint = 18
    hand_cfg = {
        'joints_name': [
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
    ],
    'edges': [
        ['hand_base_link', 'index_mcp_roll'],
        ['index_mcp_roll', 'index_mcp_pitch'],
        ['index_mcp_pitch', 'index_pip'],
        ['hand_base_link', 'middle_mcp_roll'],
        ['middle_mcp_roll', 'middle_mcp_pitch'],
        ['middle_mcp_pitch', 'middle_pip'],
        ['hand_base_link', 'ring_mcp_roll'],
        ['ring_mcp_roll', 'ring_mcp_pitch'],
        ['ring_mcp_pitch', 'ring_pip'],
        ['hand_base_link', 'pinky_mcp_roll'],
        ['pinky_mcp_roll', 'pinky_mcp_pitch'],
        ['pinky_mcp_pitch', 'pinky_pip'],
        ['hand_base_link', 'thumb_cmc_roll'],
        ['thumb_cmc_roll', 'thumb_cmc_yaw'],
        ['thumb_cmc_yaw', 'thumb_cmc_pitch'],
        ['thumb_cmc_pitch', 'thumb_mcp'],
        ['thumb_mcp', 'thumb_ip'],

        ['index_pip', 'index_tip'],
        ['middle_pip', 'middle_tip'],
        ['ring_pip', 'ring_tip'],
        ['pinky_pip', 'pinky_tip'],
        ['thumb_ip', 'thumb_tip']
    ],
    'root_name': 'hand_base_link',
    'end_effectors': [
        'index_pip',
        'middle_pip',
        'ring_pip',
        'pinky_pip',
        'thumb_ip'
    ],
    'elbows': [
        'index_mcp_pitch',
        'middle_mcp_pitch',
        'ring_mcp_pitch',
        'pinky_mcp_pitch',
        'thumb_mcp'
    ]
    }
    robot_connections = [
                # 手基座到各指根
                [0, 1], [0, 4], [0, 7], [0, 10], [0, 13],
                # 食指
                [1, 2], [2, 3], [3, 18],
                # 中指
                [4, 5], [5, 6], [6, 19],
                # 无名指
                [7, 8], [8, 9], [9, 20],
                # 小指
                [10, 11], [11, 12], [12, 21],
                # 大拇指
                [13, 14], [14, 15], [15, 16], [16, 17], [17, 22]
            ]
    correction_matrix = None
    # 关节角度限制 (弧度)
    angle_limit_rob = [
        [0.0, 0.0],           # hand_base_link (固定关节，无限制或设为0)
        [-0.18, 0.18],       # index_mcp_roll
        [0.0, 1.57],         # index_mcp_pitch
        [0.0, 1.57],         # index_pip
        [-0.18, 0.18],       # middle_mcp_roll
        [0.0, 1.57],         # middle_mcp_pitch
        [0.0, 1.57],         # middle_pip
        [-0.18, 0.18],       # ring_mcp_roll
        [0.0, 1.57],         # ring_mcp_pitch
        [0.0, 1.57],         # ring_pip
        [-0.18, 0.18],       # pinky_mcp_roll
        [0.0, 1.57],         # pinky_mcp_pitch
        [0.0, 1.57],         # pinky_pip
        [-0.6, 0.6],         # thumb_cmc_roll
        [0.0, 1.6],          # thumb_cmc_yaw
        [0.0, 1.0],          # thumb_cmc_pitch
        [0.0, 1.57],         # thumb_mcp
        [0.0, 1.57]         # thumb_ip
    ]
   
elif hand_brand == 'shadow':
    excluded_pairs=[(3, 4), (7, 8), (11, 12), (15, 16), (20, 21)]
    urdf_file = "D:\\2026\\code\\TransHandR\\TransHandR\\dataset\\robot\\shadow_hand\\shadow_hand_right.urdf"
    hand_cfg = {
        'joints_name': [
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
        ],
        'edges': [
            # 拇指链路
            ['WRJ1', 'THJ5'], ['THJ5', 'THJ4'], ['THJ4', 'THJ3'], 
            ['THJ3', 'THJ2'], ['THJ2', 'THJ1'], ['THJ1', 'THtip'],
            # 食指链路
            ['WRJ1', 'FFJ4'], ['FFJ4', 'FFJ3'], ['FFJ3', 'FFJ2'], ['FFJ2', 'FFJ1'], ['FFJ1', 'FFtip'],
            # 中指链路
            ['WRJ1', 'MFJ4'], ['MFJ4', 'MFJ3'], ['MFJ3', 'MFJ2'], ['MFJ2', 'MFJ1'], ['MFJ1', 'MFtip'],
            # 无名指链路
            ['WRJ1', 'RFJ4'], ['RFJ4', 'RFJ3'], ['RFJ3', 'RFJ2'], ['RFJ2', 'RFJ1'], ['RFJ1', 'RFtip'],
            # 小指链路
            ['WRJ1', 'LFJ5'], ['LFJ5', 'LFJ4'], ['LFJ4', 'LFJ3'], 
            ['LFJ3', 'LFJ2'], ['LFJ2', 'LFJ1'], ['LFJ1', 'LFtip']
        ],
        'root_name': 'WRJ1',  # 从腕关节开始
        'end_effectors': [
            'THtip', 'FFtip', 'MFtip', 'RFtip', 'LFTip'  # 各指末端关节
        ],
        'elbows': [
            'THJ3', 'FFJ2', 'MFJ2', 'RFJ2', 'LFJ2'  # 各指中间关节作为"肘部"
        ],
    }
    robot_connections = [
        [0, 1], [0, 6], [0, 10], [0, 14], [0, 18],  # 手腕到各指根
        # 拇指
        [1, 2], [2, 3], [3, 4], [4, 5], [5, 23],
        # 食指
        [6, 7], [7, 8], [8, 9], [9, 24],
        # 中指
        [10, 11], [11, 12], [12, 25],
        # 无名指
        [14, 15], [15, 16], [16, 26],
        # 小指
        [18, 19], [19, 20], [20, 21], [21, 22], [22, 27]
    ]
    correction_matrix = torch.tensor([[0, -1, 0],
                                      [1, 0, 0],
                                      [0, 0, 1]], dtype=torch.float32)
    # 记录机器手的特定关节索引
    TIP_dic_rb = [23, 24, 25, 26, 27]  # 对应各指末端
    DIP_dic_rb = [5, 9, 13, 17, 22]   # 对应远端关节
    PIP_dic_rb = [4, 8, 12, 16, 21]   # 对应近端关节
    MCP_dic_rb = [3, 7, 11, 15, 20]   # 对应掌指关节
    
    rb_dic = {'TIP_dic':TIP_dic_rb, 'DIP_dic':DIP_dic_rb, 'PIP_dic':PIP_dic_rb, 'MCP_dic':MCP_dic_rb}
    
    out_num_joint = 23 # 23个活动关节（不含指尖关节）
    # Shadow Hand 的关节角度限制
    angle_limit_rob = [
        [-0.6981, 0.4886],    # WRJ1 (lower=-0.698131700798, upper=0.488692190558)
        # 拇指关节限制
        [-1.0471, 1.0471],    # THJ5 (lower=-1.0471975512, upper=1.0471975512)
        [0.0, 1.2217],        # THJ4 (lower=0.0, upper=1.2217304764)
        [-0.2094, 0.2094],    # THJ3 (lower=-0.209439510239, upper=0.209439510239)
        [-0.6981, 0.6981],    # THJ2 (lower=-0.698131700798, upper=0.698131700798)
        [-0.2617, 1.5707],    # THJ1 (lower=-0.261799387799, upper=1.57079632679)
        # 食指关节限制
        [-0.3490, 0.3490],    # FFJ4 (lower=-0.349065850399, upper=0.349065850399)
        [-0.2617, 1.5707],    # FFJ3 (lower=-0.261799387799, upper=1.57079632679)
        [0.0, 1.5707],        # FFJ2 (lower=0.0, upper=1.57079632679)
        [0.0, 1.5707],        # FFJ1 (lower=0.0, upper=1.57079632679)
        # 中指关节限制
        [-0.3490, 0.3490],    # MFJ4 (lower=-0.349065850399, upper=0.349065850399)
        [-0.2617, 1.5707],    # MFJ3 (lower=-0.261799387799, upper=1.57079632679)
        [0.0, 1.5707],        # MFJ2 (lower=0.0, upper=1.57079632679)
        [0.0, 1.5707],        # MFJ1 (lower=0.0, upper=1.57079632679)
        # 无名指关节限制
        [-0.3490, 0.3490],    # RFJ4 (lower=-0.349065850399, upper=0.349065850399)
        [-0.2617, 1.5707],    # RFJ3 (lower=-0.261799387799, upper=1.57079632679)
        [0.0, 1.5707],        # RFJ2 (lower=0.0, upper=1.57079632679)
        [0.0, 1.5707],        # RFJ1 (lower=0.0, upper=1.57079632679)
        # 小指关节限制
        [0.0, 0.7853],         # LFJ5 (lower=0.0, upper=0.785398163397)
        [-0.3490, 0.3490],    # LFJ4 (lower=-0.349065850399, upper=0.349065850399)
        [-0.2617, 1.5707],    # LFJ3 (lower=-0.261799387799, upper=1.57079632679)
        [0.0, 1.5707],        # LFJ2 (lower=0.0, upper=1.57079632679)
        [0.0, 1.5707]        # LFJ1 (lower=0.0, upper=1.57079632679)
    ]
    scaling_factor_rb = 1.0/0.0659

elif hand_brand == 'svhhand':
    excluded_pairs=[(2, 3), (6, 7), (10, 11), (14, 15), (18, 19)]
    urdf_file = "D:\\2026\\code\\TransHandR\\TransHandR\\dataset\\robot\\schunk_hand\\schunk_svh_hand_right.urdf"
    hand_cfg = {
        'joints_name': [
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

        ],
        'edges': [
            # 拇指链路
            ['right_hand_f4', 'right_hand_Thumb_Opposition'],
            ['right_hand_Thumb_Opposition', 'right_hand_Thumb_Flexion'],
            ['right_hand_Thumb_Flexion', 'right_hand_j3'],
            ['right_hand_j3', 'right_hand_j4'],
            ['right_hand_j4', 'thtip_joint'],
            # 食指链路
            ['right_hand_f4', 'right_hand_index_spread'],
            ['right_hand_index_spread', 'right_hand_Index_Finger_Proximal'],
            ['right_hand_Index_Finger_Proximal', 'right_hand_Index_Finger_Distal'],
            ['right_hand_Index_Finger_Distal', 'right_hand_j14'],
            ['right_hand_j14', 'fftip_joint'],
            # 中指链路
            ['right_hand_f4', 'right_hand_middle_spread_dummy'],
            ['right_hand_middle_spread_dummy', 'right_hand_Middle_Finger_Proximal'],
            ['right_hand_Middle_Finger_Proximal', 'right_hand_Middle_Finger_Distal'],
            ['right_hand_Middle_Finger_Distal', 'right_hand_j15'],
            ['right_hand_j15', 'mftip_joint'],
            # 无名指和尾指掌面可动关节
            ['right_hand_f4', 'right_hand_j5'],
            # 无名指链路
            ['right_hand_j5', 'right_hand_ring_spread'],
            ['right_hand_ring_spread', 'right_hand_Ring_Finger'],
            ['right_hand_Ring_Finger', 'right_hand_j12'],
            ['right_hand_j12', 'right_hand_j16'],
            ['right_hand_j16', 'rftip_joint'],
            # 小指链路
            ['right_hand_j5', 'right_hand_Finger_Spread'],
            ['right_hand_Finger_Spread', 'right_hand_Pinky'],
            ['right_hand_Pinky', 'right_hand_j13'],
            ['right_hand_j13', 'right_hand_j17'],
            ['right_hand_j17', 'lftip_joint']
        ],
        'root_name': 'right_hand_f4',  # 抽象的指根位置
        'end_effectors': [
            'thtip_joint', 'fftip_joint', 'mftip_joint', 'rftip_joint', 'lftip_joint'  # 各指末端关节
        ],
        'elbows': [
            'right_hand_j3', 'right_hand_Index_Finger_Distal', 'right_hand_Middle_Finger_Distal',
            'right_hand_j12', 'right_hand_j13'  # 各指中间关节
        ]
    }
    robot_connections = [ [0, 1], [0, 5], [0, 9], [0, 13], [13,14],[13, 18],  # 手腕到各指根
    [1,2], [2,3], [3,4], [4,22],          # 拇指
    [5,6], [6,7], [7,8], [8,23],          # 食指
    [9,10], [10,11], [11,12], [12,24],    # 中指
    [14,15], [15,16], [16,17], [17,25],    # 无名指
    [18,19], [19,20], [20,21], [21,26]     # 小指
    ]
    # 记录机器手的特定关节索引
    TIP_dic_rb = [21, 22, 23, 24, 25]  # 对应各指末端
    DIP_dic_rb = [4, 8, 12, 16, 20]   # 对应远端关节
    PIP_dic_rb = [3, 7, 11, 15, 19]   # 对应近端关节
    MCP_dic_rb = [2, 6, 10, 14, 18]   # 对应掌指关节
    
    rb_dic = {'TIP_dic':TIP_dic_rb, 'DIP_dic':DIP_dic_rb, 'PIP_dic':PIP_dic_rb, 'MCP_dic':MCP_dic_rb}
    
    # # 关节映射字典 实机使用的关节索引
    # joint_map = {0: 15, 1: 2, 2: 5, 3: 8, 4: 11, 5: 14, 6: 1, 7: 4, 8: 7, 9: 10, 10: 13, 
    #             11: 0, 12: 0, 13: 0, 14: 0, 15: 16, 16: 0, 17: 0, 18: 0, 19: 0,
    #             20: 17, 21: 3, 22: 6, 23: 9, 24: 12}
    correction_matrix = torch.tensor([[0, 1, 0],
                                      [-1, 0, 0],
                                      [0, 0, 1]], dtype=torch.float32)
    scaling_factor_rb = 1.0/0.0687
    out_num_joint =22  # 22个活动关节（不含指尖关节）
    # SVH Hand 的关节角度限制 (弧度)
    angle_limit_rob = [
    # 拇指关节
    [0.0 , 0.0],            # right_hand_f4 - 手腕(固定)
    [0.0, 0.9879],         # right_hand_Thumb_Opposition - 拇指对掌
    [0.0, 0.9704],         # right_hand_Thumb_Flexion - 拇指弯曲
    [0.0, 0.98506],        # right_hand_j3 - 拇指联动1
    [0.0, 1.406],          # right_hand_j4 - 拇指联动2
    # 食指关节
    [0.0, 0.28833],        # right_hand_index_spread - 食指展开
    [0.0, 0.79849],        # right_hand_Index_Finger_Proximal - 食指近节弯曲
    [0.0, 1.334],          # right_hand_Index_Finger_Distal - 食指远节弯曲
    [0.0, 1.394],          # right_hand_j14 - 食指尖联动
    
    # 中指关节
    [0.0, 0.0],            # right_hand_middle_spread_dummy - 中指展开(固定)
    [0.0, 0.79849],        # right_hand_Middle_Finger_Proximal - 中指近节弯曲
    [0.0, 1.334],          # right_hand_Middle_Finger_Distal - 中指远节弯曲
    [0.0, 1.334],          # right_hand_j15 - 中指尖联动
    # 基础关节
    [0.0, 0.98786],        # right_hand_j5 - 掌骨间联动
    # 无名指关节
    [0.0, 0.28833],        # right_hand_ring_spread - 无名指展开
    [0.0, 0.98175],        # right_hand_Ring_Finger - 无名指弯曲
    [0.0, 1.334],          # right_hand_j12 - 无名指联动1
    [0.0, 1.395],          # right_hand_j16 - 无名指尖联动
    
    # 小指关节
    [0.0, 0.5829],         # right_hand_Finger_Spread - 小指展开
    [0.0, 0.98175],        # right_hand_Pinky - 小指弯曲
    [0.0, 1.334],          # right_hand_j13 - 小指联动1
    [0.0, 1.3971],         # right_hand_j17 - 小指尖联动
]

elif hand_brand == 'allegro_hand':
    urdf_file = "D:\\2026\\code\\TransHandR\\TransHandR\\dataset\\robot\\allegro_hand\\allegro_hand_right_glb.urdf"
    hand_cfg = {
        'joints_name': [
            'hand_base_joint', #0
            # 拇指关节
            'joint_12.0', 'joint_13.0', 'joint_14.0', 'joint_15.0','joint_15.0_tip', #1-5
            # 食指关节
            'joint_0.0', 'joint_1.0', 'joint_2.0', 'joint_3.0', 'joint_3.0_tip', # 6-10
            # 中指关节
            'joint_4.0', 'joint_5.0', 'joint_6.0', 'joint_7.0','joint_7.0_tip', # 11-15
            # 无名指关节
            'joint_8.0', 'joint_9.0', 'joint_10.0', 'joint_11.0','joint_11.0_tip' # 16-20
        ],
        'edges': [
            # 拇指链路
            ['hand_base_joint', 'joint_12.0'], ['joint_12.0', 'joint_13.0'],
            ['joint_13.0', 'joint_14.0'], ['joint_14.0', 'joint_15.0'],['joint_15.0', 'joint_15.0_tip'],
            # 食指链路
            ['hand_base_joint', 'joint_0.0'], ['joint_0.0', 'joint_1.0'],
            ['joint_1.0', 'joint_2.0'], ['joint_2.0', 'joint_3.0'],['joint_3.0', 'joint_3.0_tip'],
            # 中指链路
            ['hand_base_joint', 'joint_4.0'], ['joint_4.0', 'joint_5.0'],
            ['joint_5.0', 'joint_6.0'], ['joint_6.0', 'joint_7.0'],['joint_7.0', 'joint_7.0_tip'],
            # 无名指链路
            ['hand_base_joint', 'joint_8.0'], ['joint_8.0', 'joint_9.0'],
            ['joint_9.0', 'joint_10.0'], ['joint_10.0', 'joint_11.0'],['joint_11.0', 'joint_11.0_tip'],
        ],
        'root_name': 'hand_base_joint',
        'end_effectors': [
            'joint_15.0_tip', 'joint_3.0_tip', 'joint_7.0_tip', 'joint_11.0_tip'
        ],
        'elbows': [
            'joint_13.0', 'joint_1.0', 'joint_5.0', 'joint_9.0'
        ],
    }
    robot_connections = [
        # 手基座到各指根
        [0, 1], [0, 6], [0, 11], [0, 16],
        # 拇指
        [1, 2], [2, 3], [3, 4], [4, 5],
        # 食指
        [6, 7], [7, 8], [8, 9], [9, 10],
        # 中指
        [11, 12], [12, 13], [13, 14], [14, 15],
        # 无名指
        [16, 17], [17, 18], [18, 19], [19, 20],
    ]
    correction_matrix = None
    # 记录机器手的特定关节索引
    TIP_dic_rb = [5,10, 15, 20]  # 拇指尖, 食指尖, 中指尖, 无名指尖
    DIP_dic_rb = [4, 9, 14, 19]  # 拇指远端, 食指远端, 中指远端, 无名指远端
    PIP_dic_rb = [3, 8, 13, 18 ]   # 拇指近端, 食指近端, 中指近端, 无名指近端
    MCP_dic_rb = [2, 7, 12, 17]   # 拇指掌指, 食指掌指, 中指掌指, 无名指掌指
    PALM_dic_rb = [1, 6, 11, 16]  # 手掌根部
    
    rb_dic = {'TIP_dic':TIP_dic_rb, 'DIP_dic':DIP_dic_rb, 'PIP_dic':PIP_dic_rb, 'MCP_dic':MCP_dic_rb, 'PALM_dic':PALM_dic_rb}
    
    scaling_factor_rb = 1.0/0.064
    out_num_joint = 16  # 16个活动关节
    # Allegro Hand 的关节角度限制 (弧度)
    angle_limit_rob = [
        [0.0, 0.0],           # hand_base_joint (固定关节)
        # 拇指关节限制
        [0.263, 1.396],       # joint_12.0 (thumb abduction)
        [-0.105, 1.163],      # joint_13.0 (thumb flexion)
        [-0.189, 1.644],      # joint_14.0 (thumb proximal)
        [-0.162, 1.719],      # joint_15.0 (thumb distal)
        # 食指关节限制
        [-0.47, 0.47],        # joint_0.0 (index abduction)
        [-0.196, 1.61],       # joint_1.0 (index proximal)
        [-0.174, 1.709],      # joint_2.0 (index intermediate)
        [-0.227, 1.618],      # joint_3.0 (index distal)
        # 中指关节限制
        [-0.47, 0.47],        # joint_4.0 (middle abduction)
        [-0.196, 1.61],       # joint_5.0 (middle proximal)
        [-0.174, 1.709],      # joint_6.0 (middle intermediate)
        [-0.227, 1.618],      # joint_7.0 (middle distal)
        # 无名指关节限制
        [-0.47, 0.47],        # joint_8.0 (ring abduction)
        [-0.196, 1.61],       # joint_9.0 (ring proximal)
        [-0.174, 1.709],      # joint_10.0 (ring intermediate)
        [-0.227, 1.618],      # joint_11.0 (ring distal)
    ]

'''
模型参数
'''
receptive_field = 3  # 感受野
#输出输入点数已在上面定义中给出
in_chans = 3        # 输入通道数
embed_dim_ratio = 32 # 嵌入维度比率
spatial_depth = 6    # 空间Transformer层数
temporal_depth = 4   # 时序Transformer层数
spatial_mlp_ratio = 4.# 空间MLP比例
temporal_mlp_ratio = 1.# 时序MLP比例
num_heads = 8
qkv_bias = True     # QKV偏置
qk_scale = None     # QK缩放
drop_path_rate = 0.1
# 关节角度限制 (弧度)
angle_limit_rob = [
    [0.0, 0.0],           # hand_base_link (固定关节，无限制或设为0)
    [-0.18, 0.18],       # index_mcp_roll
    [0.0, 1.57],         # index_mcp_pitch
    [0.0, 1.57],         # index_pip
    [-0.18, 0.18],       # middle_mcp_roll
    [0.0, 1.57],         # middle_mcp_pitch
    [0.0, 1.57],         # middle_pip
    [-0.18, 0.18],       # ring_mcp_roll
    [0.0, 1.57],         # ring_mcp_pitch
    [0.0, 1.57],         # ring_pip
    [-0.18, 0.18],       # pinky_mcp_roll
    [0.0, 1.57],         # pinky_mcp_pitch
    [0.0, 1.57],         # pinky_pip
    [-0.6, 0.6],         # thumb_cmc_roll
    [0.0, 1.6],          # thumb_cmc_yaw
    [0.0, 1.0],          # thumb_cmc_pitch
    [0.0, 1.57],         # thumb_mcp
    [0.0, 1.57]         # thumb_ip
]

# 关节映射字典
joint_map = {0: 15, 
             1: 2, 
             2: 5, 
             3: 8, 
             4: 11, 
             5: 14, 
             6: 1, 
             7: 4, 
             8: 7, 
             9: 10, 
             10: 13, 
            11: 0, 12: 0, 13: 0, 14: 0, 
            15: 16, 
            16: 0, 17: 0, 18: 0, 19: 0,
            20: 17, 
            21: 3, 
            22: 6, 
            23: 9, 
            24: 12}

def trans2realworld(angle):
    '''
    将虚拟角度转换为真实角度,且检查是否超限,输入为弧度,下限,上限
    '''
    # 18个关节
    angle_real = angle.copy()
    # 先归一化至0-255 按照关节角度限制angle_limit_rob进行归一化
    for i in range(len(angle_real)):
        low, high = angle_limit_rob[i]
        # 归一化到0-1
        norm_angle = (angle_real[i] - low) / (high - low) if high > low else 0.0
        # 归一化到0-255
        angle_real[i] = int(norm_angle * 255)
    # 再进行重排顺序 按照joint_map进行重排
    angle_mapped = [0] * 25
    for drive_idx, joint_idx in joint_map.items():
        angle_mapped[drive_idx] = angle_real[joint_idx]
        # 所有角度要从0-255转为255-0
        angle_mapped[drive_idx] = 255 - angle_mapped[drive_idx]
    #不需要反转故再反转一次
    angle_mapped[0] = 255 - angle_mapped[0]
    angle_mapped[6] = 255 - angle_mapped[6]
    angle_mapped[7] = 255 - angle_mapped[7]
    angle_mapped[8] = 255 - angle_mapped[8]
    angle_mapped[9] = 255 - angle_mapped[9]
    #输出要求是整数列表
    angle_mapped = [unit(int(a)) for a in angle_mapped]
    return angle_mapped

def unit(num):
    #限制在0-255
    return 0 if num < 0 else 255 if num > 255 else num

class HandController:
    def __init__(self, left_positions=None):
        self.yaml = LoadWriteYaml()
        # 加载左手配置文件
        self.left_setting = self.yaml.load_setting_yaml(config="setting")
        self.hands = {}  # 存储左手的配置和API
        self._init_hands()
        if self.hands:
            self._set_default_speeds()
        self.init_positions = {
            "left": self._get_default_positions("left", left_positions)
        }

    def _test_can_connection(self, can_channel, bitrate=1000000):
        """测试 CAN 连接是否可用"""
        try:
            ColorMsg(msg=f"测试 CAN 通道 {can_channel}...", color="yellow")
            bus = can.interface.Bus(
                channel=can_channel,
                bustype='pcan',
                bitrate=bitrate
            )
            test_msg = can.Message(arbitration_id=0x123, data=[0x01], is_extended_id=False)
            bus.send(test_msg)
            time.sleep(0.1)
            bus.shutdown()
            ColorMsg(msg=f"CAN 通道 {can_channel} 连接成功", color="green")
            return True
        except Exception as e:
            ColorMsg(msg=f"CAN 通道 {can_channel} 连接失败: {e}", color="red")
            return False

    def _init_hands(self):
        # 初始化左手
        hand_type = "left"
        setting = self.left_setting
        hand_config = setting['LINKER_HAND']['LEFT_HAND']
        if hand_config.get('EXISTS', False):
            hand_joint = hand_config['JOINT']
            can_channel = hand_config.get('CAN_CHANNEL', 'PCAN_USBBUS1')
            bitrate = hand_config.get('BITRATE', 1000000)

            if not self._test_can_connection(can_channel, bitrate):
                ColorMsg(msg=f"左手 CAN 通道不可用，跳过初始化", color="red")
                return

            try:
                ColorMsg(msg=f"初始化 左手 LinkerHandApi...", color="yellow")
                api = LinkerHandApi(
                    hand_type=hand_type,
                    hand_joint=hand_joint,
                    can=can_channel
                )

                if not hasattr(api.hand, 'bus') or api.hand.bus is None:
                    ColorMsg(msg=f"{hand_type} bus 未正确初始化，正在修复...", color="yellow")
                    api.hand.bus = can.interface.Bus(
                        channel=can_channel,
                        bustype='pcan',
                        bitrate=bitrate,
                        can_filters=[{"can_id": api.hand.can_id, "can_mask": 0x7FF}]
                    )

                version = api.get_embedded_version()
                if version is None or len(version) == 0:
                    ColorMsg(msg=f"左手 硬件版本未识别，可能设备未响应",
                             color="red")
                    return

                self.hands[hand_type] = {
                    "joint": hand_joint,
                    "api": api,
                    "bus": api.hand.bus,
                    "channel": can_channel
                }
                ColorMsg(
                    msg=f"初始化左手成功！关节类型: {hand_joint}, CAN通道: {can_channel}, 版本: {version}",
                    color="green")

            except Exception as e:
                ColorMsg(msg=f"初始化左手 LinkerHandApi 失败: {e}",
                         color="red")
                ColorMsg(
                    msg=f"详细建议：1. 确认 PCAN 驱动已安装；2. 使用 PCAN-View 测试 {can_channel}；3. 检查设备连接；4. 验证 YAML 中的 CAN_CHANNEL 配置。",
                    color="yellow")
                return
        else:
            print("左手未启用")

        if not self.hands:
            ColorMsg(msg="警告：左手初始化失败，请检查硬件和配置！", color="red")
        else:
            ColorMsg(msg=f"成功初始化左手", color="green")

    def _set_default_speeds(self):
        speed_map = {
            "L7": [180, 250, 250, 250, 250, 250, 250],
            "L10": [180, 250, 250, 250, 250],
            "L20": [120, 180, 180, 180, 180],
            "L21": [60, 220, 220, 220, 220],
            "L25": [60, 250, 250, 250, 250]
        }
        for hand_type, hand_info in self.hands.items():
            speed = speed_map.get(hand_info["joint"], [180, 250, 250, 250, 250])
            ColorMsg(msg=f"设置左手速度: {speed}", color="green")
            try:
                hand_info["api"].set_speed(speed)
                ColorMsg(msg=f"左手速度设置成功", color="green")
            except Exception as e:
                ColorMsg(msg=f"设置左手速度失败: {e}", color="red")

    def _get_default_positions(self, hand_type, positions):
        if hand_type not in self.hands:
            return []
        pos_map = {
            "L7": [250] * 7,
            "L10": [255] * 10,
            "L20": [255, 255, 255, 255, 255, 255, 10, 100, 180, 240, 245, 255, 255, 255, 255, 255, 255, 255, 255, 255],
            "L21": [96, 255, 255, 255, 255, 150, 114, 151, 189, 255, 180, 255, 255, 255, 255, 255, 255, 255, 255, 255,
                    255, 255, 255, 255, 255],
            "L25": [96, 255, 255, 255, 255, 150, 114, 151, 189, 255, 180, 255, 255, 255, 255, 255, 255, 255, 255, 255,
                    255, 255, 255, 255, 255]
        }
        return positions if positions else pos_map.get(self.hands[hand_type]["joint"], [255] * 10)

    def control_hand(self, left_positions=None):
        if not self.hands:
            ColorMsg(msg="无可用手部，无法执行控制", color="red")
            return

        for hand_type, hand_info in self.hands.items():
            positions = left_positions 

            if not positions:
                positions = self.init_positions.get(hand_type, [])

            if not positions:
                ColorMsg(msg=f"左手 无有效位置数据，跳过控制", color="yellow")
                continue

            expected_len = len(self.init_positions.get(hand_type, []))
            if expected_len > 0 and len(positions) != expected_len:
                ColorMsg(
                    msg=f"错误: 左手控制信号长度 {len(positions)} 不匹配关节数量 {expected_len}",
                    color="red")
                continue

            ColorMsg(
                msg=f"执行左手控制信号: 前{5}个位置值 [{', '.join(map(str, positions[:5]))}]...",
                color="green")
            try:
                hand_info["api"].finger_move(pose=positions)
                ColorMsg(msg=f"左手控制执行成功", color="green")
            except Exception as e:
                ColorMsg(msg=f"控制左手失败: {e}", color="red")
                continue

    def close(self):
        for hand_type, hand_info in self.hands.items():
            if "bus" in hand_info and hand_info["bus"]:
                try:
                    hand_info["bus"].shutdown()
                    print(f"关闭左手 CAN 总线")
                except Exception as e:
                    ColorMsg(msg=f"关闭左手 CAN 总线失败: {e}", color="red")

def vision_pro_data_process(shared_dict, stop_event):
    """
    Vision Pro数据获取进程
    """
    print("Vision Pro数据获取进程启动")
    
    avp_ip = "192.168.43.20"  # Vision Pro IP (shown in the app)
    s = VisionProStreamer(ip=avp_ip)
    
    # 存储最近3帧数据
    recent_frames = []
    t_sum = 0
    t_average = 0
    f = 0 
    while not stop_event.is_set():
        t_start = time.time()
        r = s.get_latest()
        
        # 提取右手手指跟踪数据
        # right_fingers = r['right_fingers']
        right_fingers = r['left_fingers']
        
        # 存储处理后的坐标
        coordinates = []
        
        # 遍历所有关节
        for i in range(len(right_fingers)):
            # 获取4x4变换矩阵
            transform_matrix = right_fingers[i]
            
            # 应用坐标变换
            # x = -transform_matrix[1][3]
            # y = transform_matrix[2][3]
            # z = -transform_matrix[0][3]  #右手

            x = transform_matrix[1][3]
            y = -transform_matrix[2][3]
            z = transform_matrix[0][3] # 左手
            # 添加到坐标列表
            coordinates.append([x, y, z])
        
        # 转换为numpy数组以便处理
        coordinates = np.array(coordinates)
        
        # 获取手腕位置（0号点）
        wrist_pos = coordinates[0]
        
        # 将所有点相对于手腕位置进行变换（减去0号点）
        relative_coordinates = coordinates - wrist_pos
        
        # 添加到最近帧列表
        recent_frames.append(relative_coordinates)
        
        # 保持最多3帧
        if len(recent_frames) > 3:
            recent_frames.pop(0)
        
        # 如果已经有3帧数据，则发送给重定向进程
        if len(recent_frames) == 3:
            # 转换为期望的格式并发送
            shared_dict['vision_pro_data'] = np.stack(recent_frames, axis=1)  # (25, 3, 3) -> (25, 3, 3)
        t_stop = time.time()
        t_sum += t_stop - t_start
        f += 1
        t_average = t_sum/f
        if f%100 == 0:
            print(f"Vision Pro数据获取完成，平均处理时间: {t_average}")
        time.sleep(0.001)  # 控制数据获取频率
    
    print("Vision Pro数据获取进程结束")

def redirection_process(shared_dict, stop_event, model_path=None):
    """
    重定向进程：接收三帧人手数据，输出一帧机器手关节角度
    """
    print("重定向进程启动")
    
    if model_path is None:
        # model_path = r"D:\\2026\\code\\TransHandR\\TransHandR\\checkpoint\\models\\visionpro\\linker\\model_final.pth"
        model_path = r"D:\\2026\\code\\TransHandR\\TransHandR\\checkpoint\\models\\thumb2\\linker\\model_final.pth"
    
    
    # 定义一些必要的字典用于测试
    rb_dic = {
        'TIP_dic': [22, 18, 19, 20, 21],  # 机器人手的指尖索引
        'DIP_dic': [17, 3, 6, 9, 12],
        'PIP_dic': [15, 2, 5, 8, 11],
        'MCP_dic': [14, 1, 4, 7, 10]
    }
    source_dic = {
        'TIP_dic': [4, 9, 14, 19, 24],  # 人体手的指尖索引
        'DIP_dic': [3, 8, 13, 18, 23],
        'PIP_dic': [2, 7, 12, 17, 22],
        'MCP_dic': [1, 6, 11, 16, 21]
    }
    try:
        # # 加载模型
        # 在 redirection_process 函数中，替换原来的模型初始化代码：
        model = PoseTransformer(
            num_frame=receptive_field,
            in_num_joints=num_joints, 
            in_chans=3, 
            out_num_joint=out_num_joint, 
            out_chans=1, 
            embed_dim_ratio=embed_dim_ratio,
            spatial_depth=spatial_depth,     
            temporal_depth=temporal_depth,    
            spatial_mlp_ratio=spatial_mlp_ratio, 
            temporal_mlp_ratio=temporal_mlp_ratio, 
            num_heads=num_heads, 
            qkv_bias=qkv_bias, 
            qk_scale=None,
            drop_path_rate=drop_path_rate,
            angle_limit_rad=angle_limit_rob
        )
        
        # 如果有可用的GPU，将模型移到GPU上
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = model.to(device)
        hand_fk = create_hand_kinematics(urdf_file, hand_cfg, device,scale_factor=scaling_factor_rb)

        # 加载模型权重
        print(f"Loading model from: {model_path}")
        # 使用 strict=False 忽略不匹配的键
        checkpoint = torch.load(model_path)
        model.load_state_dict(checkpoint['model_pos'], strict=False)
        print("Model loaded successfully!")
        print(f"Model is on device: {next(model.parameters()).device}")
        
        model.eval()
        
        print("重定向进程初始化完成")
        f = 0 
        t_sum = 0
        t_average = 0
        while not stop_event.is_set():
            # 检查是否有新的Vision Pro数据
            vision_data = shared_dict.get('vision_pro_data', None)
            t1 = time.time()
            if vision_data is not None:
                # 转换数据格式以适应模型输入
                # vision_data shape: (25, 3, 3) -> (1, 3, 25, 3)
                reshaped_data = np.transpose(vision_data, (1, 0, 2))  # (3, 25, 3)
                reshaped_data = np.expand_dims(reshaped_data, axis=0)  # (1, 3, 25, 3)
                # 数据乘上缩放因子
                reshaped_data *= scaling_factor
                # 转换为tensor并移动到设备
                input_tensor = torch.from_numpy(reshaped_data).float().to(device)
                
                with torch.no_grad():
                    # 模型推理
                    output = model(input_tensor)
                    
                    # 提取结果并转换为numpy数组
                    result = output.cpu().numpy()[0]  # 移除批次维度
                    
                    # 将结果发送到下一个进程
                    shared_dict['robot_angles'] = result
                    
                    # print(f"重定向进程处理完成，输出形状: {result.shape}")
            t2 = time.time()
            f += 1
            t_sum += t2-t1
            t_average = t_sum/f
            if f%100 == 0:
                # print(f"重定向进程处理完成，输出形状: {result.shape}")
                print(f"处理时间: {t_average}")
                print(t2-t1)
            time.sleep(0.0001)  # 控制处理频率
    
    except Exception as e:
        print(f"重定向进程出错: {str(e)}")
    
    print("重定向进程结束")

def virtual_env_process(shared_dict, stop_event, v_rate=1):
    """
    虚拟环境进程
    """
    # 初始化虚拟环境
    env = gym.make('yumi-v0')
    observation = env.reset()
    
    camera_distance = 2
    camera_yaw = 90
    camera_pitch = -10
    camera_roll = 0
    camera_target_position = [0, 0, 0.05]
    paused = False
    
    print("虚拟环境进程启动")
    t_sum = 0
    t_average = 0
    f = 0
    while not stop_event.is_set():
        t_start = time.time()
        env.render()
        
        # 尝试获取机器人关节角度数据
        robot_angles = shared_dict.get('robot_angles', None)
        if robot_angles is not None:
            # 将数据转换为虚拟环境所需的格式
            for i in range(2):
                R_robot_angle = np.concatenate((robot_angles, np.zeros((5,)))).tolist()
                action = R_robot_angle
                # 检查键盘事件
                keys = p.getKeyboardEvents()
                for k, v in keys.items():
                    if v & p.KEY_WAS_TRIGGERED:
                        if k == ord('w'):
                            camera_distance -= 0.3
                        elif k == ord('s'):
                            camera_distance += 0.3
                        elif k == ord('a'):
                            camera_yaw -= 10
                        elif k == ord('d'):
                            camera_yaw += 10
                        elif k == ord('q'):
                            camera_pitch -= 10
                        elif k == ord('e'):
                            camera_pitch += 10
                        elif k == ord(' '):
                            paused = not paused
                            print('切换暂停')
                # 如果处于暂停状态，则跳过仿真步骤
                if paused:
                    time.sleep(0.02)  # 保持短暂延迟以减少CPU占用
                p.resetDebugVisualizerCamera(cameraDistance=camera_distance,
                                            cameraYaw=camera_yaw,
                                            cameraPitch=camera_pitch,
                                            cameraTargetPosition=camera_target_position)
                
                observation, reward, done, info = env.step(action)
                t_end = time.time()
                f += 1
                t_sum += t_end-t_start
                t_average = t_sum/f
                if f%100 == 0:
                    print(f"虚拟环境进程处理完成，平均处理时间: {t_average}")
                time.sleep(0.02 * v_rate)
    
    env.close()
    print("虚拟环境进程结束")

def real_hand_process(shared_dict, stop_event, v_rate=1):
    """
    真实机器手进程
    """
    print("真实机器手进程启动")
    
    # 初始化手部控制器
    initial_positions = [255] * 25  # 默认位置
    controller = HandController(left_positions=initial_positions)
    
    while not stop_event.is_set():
        # 尝试获取机器人关节角度数据
        robot_angles = shared_dict.get('robot_angles', None)
        if robot_angles is not None:
            # 将数据转换为真实机器手所需的格式
            R_robot_angle = trans2realworld(robot_angles)
            controller.control_hand(left_positions=R_robot_angle)
        
        time.sleep(0.02 * v_rate)
    
    controller.close()
    print("真实机器手进程结束")

if __name__ == '__main__':
    # 设置项目根目录
    current_dir = os.path.dirname(os.path.abspath(__file__))
    sys.path.append(os.path.abspath(os.path.join(current_dir, "../..")))
    
    # 创建共享字典
    manager = multiprocessing.Manager()
    shared_dict = manager.dict()
    
    # 创建停止事件
    stop_event = multiprocessing.Event()
    v_rate = 1  # 可根据需要调整速度比例
    
    try:
        # 启动四个进程
        vision_process = multiprocessing.Process(target=vision_pro_data_process, args=(shared_dict, stop_event))
        redirection_process_instance = multiprocessing.Process(target=redirection_process, args=(shared_dict, stop_event))
        virtual_env_process_instance = multiprocessing.Process(target=virtual_env_process, args=(shared_dict, stop_event, v_rate))
        real_hand_process_instance = multiprocessing.Process(target=real_hand_process, args=(shared_dict, stop_event, v_rate))
        
        # 启动进程
        vision_process.start()
        redirection_process_instance.start()
        virtual_env_process_instance.start()
        real_hand_process_instance.start()
        
        print("所有进程已启动，按 Ctrl+C 或 'z' 键停止...")
        
        # 等待用户中断
        while True:
            # if keyboard.is_pressed('z'):
                # print("检测到按键 'z'，正在停止程序...")
                # break
            time.sleep(0.1)
    
    except KeyboardInterrupt:
        print("用户中断程序")
    
    finally:
        # 设置停止事件，通知所有进程退出
        stop_event.set()
        
        # 等待进程结束
        processes = [vision_process, redirection_process_instance, virtual_env_process_instance, real_hand_process_instance]
        # processes = [vision_process, redirection_process_instance, real_hand_process_instance]
        # processes = [vision_process, redirection_process_instance, virtual_env_process_instance]
        for proc in processes:
            if proc and proc.is_alive():
                proc.join(timeout=2)
                if proc.is_alive():
                    proc.terminate()
        
        print("程序已安全退出")