'''
关节配置文件
'''
import numpy as np
import torch
"""
碰撞忽略点记得调整
"""
data_tpye = 'visionpro'
# data_tpye = 'slahmr'
#### 手型配置选择
hand_brand = 'linker'  
# # 'yumi'  'linker'  'shadow' 'svhhand' 'inspire'
# hand_brand = 'svhhand'
# hand_brand = 'shadow'
# hand_brand = 'inspire'

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
    urdf_file = "D:\\2026\\code\\mytrans\\dataset\\robot\\ur3\\robot(ur3).urdf"

elif hand_brand == 'linker':
    urdf_file = "D:\\2026\\code\\mytrans\\dataset\\robot\\l21_right\\linkerhand_l21_right.urdf"
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
    urdf_file = "D:\\2026\\code\\mytrans\\dataset\\robot\\shadow_hand\\shadow_hand_right.urdf"
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
    scaling_factor_rb = 4.0/5.0

elif hand_brand == 'svhhand':
    excluded_pairs=[(1, 2), (5, 6), (9, 10), (13, 14), (14, 15),(15, 16)]
    urdf_file = "D:\\2026\\code\\mytrans\\dataset\\robot\\schunk_hand\\schunk_svh_hand_right.urdf"
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
    TIP_dic_rb = [22, 23, 24, 25, 26]  # 对应各指末端
    DIP_dic_rb = [4, 8, 12, 19, 21]   # 对应远端关节
    PIP_dic_rb = [3, 7, 11, 18, 20]   # 对应近端关节
    MCP_dic_rb = [2, 6, 10, 17, 19]   # 对应掌指关节
    
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
    urdf_file = "D:\\2026\\code\\mytrans\\dataset\\robot\\allegro_hand\\allegro_hand_right_glb.urdf"
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

elif hand_brand == 'inspire':
    excluded_pairs=[(1, 2), (4, 5), (7, 8), (10, 11), (14, 15)]  # 根据linker手的配置设定
    urdf_file = "D:\\2026\\code\\mytrans\\dataset\\robot\\inspire_URDF\\urdf\\R_inspire.urdf"
    
    # Inspire手的关节配置
    hand_cfg = {
        'joints_name': [
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
            'R_pinky_intermediate_joint',  # 12
            # 指尖
            'R_thumb_tip_joint', # 13
            'R_index_tip_joint', # 14
            'R_middle_tip_joint', # 15
            'R_ring_tip_joint', # 16
            'R_pinky_tip_joint' # 17
        ],
        'edges': [
            # 拇指链路
            ['R_base_link_joint', 'R_thumb_proximal_yaw_joint'],
            ['R_thumb_proximal_yaw_joint', 'R_thumb_proximal_pitch_joint'],
            ['R_thumb_proximal_pitch_joint', 'R_thumb_intermediate_joint'],
            ['R_thumb_intermediate_joint', 'R_thumb_distal_joint'],
            # 食指链路
            ['R_base_link_joint', 'R_index_proximal_joint'],
            ['R_index_proximal_joint', 'R_index_intermediate_joint'],
            # 中指链路
            ['R_base_link_joint', 'R_middle_proximal_joint'],
            ['R_middle_proximal_joint', 'R_middle_intermediate_joint'],
            # 无名指链路
            ['R_base_link_joint', 'R_ring_proximal_joint'],
            ['R_ring_proximal_joint', 'R_ring_intermediate_joint'],
            # 小指链路
            ['R_base_link_joint', 'R_pinky_proximal_joint'],
            ['R_pinky_proximal_joint', 'R_pinky_intermediate_joint'],
            # 指尖
            ['R_thumb_distal_joint', 'R_thumb_tip_joint'],
            ['R_index_intermediate_joint', 'R_index_tip_joint'],
            ['R_middle_intermediate_joint', 'R_middle_tip_joint'],
            ['R_ring_intermediate_joint', 'R_ring_tip_joint'],
            ['R_pinky_intermediate_joint', 'R_pinky_tip_joint']
        ],
        'root_name': 'R_base_link_joint',
        'end_effectors': [
            'R_thumb_tip_joint', 'R_index_tip_joint', 'R_middle_tip_joint', 'R_ring_tip_joint', 'R_pinky_tip_joint'
        ],
        'elbows': [
            'R_thumb_proximal_pitch_joint',
            'R_thumb_intermediate_joint',
            'R_index_proximal_joint',
            'R_middle_proximal_joint',
            'R_ring_proximal_joint',
            'R_pinky_proximal_joint'
        ]
    }
    
    robot_connections = [
        # 手基座到各指根
        [0, 1],  # 拇指根部
        [1, 2], [2, 3], [3, 4],  # 拇指链
        [0, 5], [5, 6],  # 食指链
        [0, 7], [7, 8],  # 中指链
        [0, 9], [9, 10],  # 无名指链
        [0, 11], [11, 12],  # 小指链
        # 指尖
        [4, 13], [6, 14], [8, 15], [10, 16], [12, 17]
    ]
    
    # 记录机器手的特定关节索引
    TIP_dic_rb = [13, 14, 15, 16, 17]  # 拇指尖, 食指尖, 中指尖, 无名指尖, 小指尖
    DIP_dic_rb = [3, 6, 8, 10, 12]  # 拇指远端, 食指远端, 中指远端, 无名指远端, 小指远端
    PIP_dic_rb = [2, 5, 7, 9, 11]   # 拇指近端, 食指近端, 中指近端, 无名指近端, 小指近端
    MCP_dic_rb = [1, 5, 7, 9, 11]   # 拇指掌指, 食指掌指, 中指掌指, 无名指掌指, 小指掌指
    
    rb_dic = {'TIP_dic':TIP_dic_rb, 'DIP_dic':DIP_dic_rb, 'PIP_dic':PIP_dic_rb, 'MCP_dic':MCP_dic_rb}
    
    scaling_factor_rb = 1  # 使用类似linker手的比例因子
    out_num_joint = 13  # 13个活动关节
    
    # Inspire Hand 的关节角度限制 (弧度)
    angle_limit_rob = [
        [0.0, 0.0],           # R_base_link_joint (固定关节)
        # 拇指关节限制
        [0.0, 1.308],         # R_thumb_proximal_yaw_joint
        [0.0, 0.6],           # R_thumb_proximal_pitch_joint
        [0.0, 0.8],           # R_thumb_intermediate_joint
        [0.0, 0.4],           # R_thumb_distal_joint
        # 食指关节限制
        [0.0, 1.47],          # R_index_proximal_joint
        [-0.04545, 1.56],     # R_index_intermediate_joint
        # 中指关节限制
        [0.0, 1.47],          # R_middle_proximal_joint
        [-0.04545, 1.56],     # R_middle_intermediate_joint
        # 无名指关节限制
        [0.0, 1.47],          # R_ring_proximal_joint
        [-0.04545, 1.56],     # R_ring_intermediate_joint
        # 小指关节限制
        [0.0, 1.47],          # R_pinky_proximal_joint
        [-0.04545, 1.56],     # R_pinky_intermediate_joint
    ]
    
    correction_matrix = torch.tensor([[-1, 0, 0],
                                      [0, 1 , 0],
                                      [0, 0, -1]], dtype=torch.float32)
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

'''
损失函数权重
'''
#thumb
# loss_weight = [500, 500, 10, 100, 10 , 500] # 侧摆损失，指尖损失，碰撞损失，大拇指损失，指尖距离损失, 大拇指二损失
#thumb2
# loss_weight = [500, 500, 10, 1000, 10, 500]
#thumb3
loss_weight = [500, 500, 10, 500, 10, 500]

# loss_weight = [0, 50, 10, 0, 10] # 侧摆损失，指尖损失，碰撞损失，大拇指损失，指尖距离损失
# loss_weight = [500, 0, 10, 0, 50]

col_threshold = 0.010