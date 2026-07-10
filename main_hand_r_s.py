'''
多进程版本：将H5数据转换为虚拟环境和真实机器手驱动
使用三个进程：数据读取进程、虚拟环境进程、真实机器手进程
'''
import multiprocessing
import gym, yumi_gym
import pybullet as p
import numpy as np
import h5py
import time
import math
import yaml
import os
import sys
import argparse
import can
from LinkerHand.linker_hand_api import LinkerHandApi
from LinkerHand.utils.load_write_yaml import LoadWriteYaml
from LinkerHand.utils.color_msg import ColorMsg
import keyboard

# 关节角度限制 (弧度) - 从main_h5_realtime.py复制
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

# 关节映射字典 - 从main_h5_realtime.py复制
joint_map = {0: 15, 1: 2, 2: 5, 3: 8, 4: 11, 5: 14, 6: 1, 7: 4, 8: 7, 9: 10, 10: 13, 
            11: 0, 12: 0, 13: 0, 14: 0, 15: 16, 16: 0, 17: 0, 18: 0, 19: 0,
            20: 17, 21: 3, 22: 6, 23: 9, 24: 12}

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
        # 大拇指的角度要从0-255转为255-0
    angle_mapped[0] = 255 - angle_mapped[0]
    angle_mapped[5] = 255 - angle_mapped[5]
    angle_mapped[10] = 255 - angle_mapped[10]
    angle_mapped[15] = 255 - angle_mapped[15]
    angle_mapped[20] = 255 - angle_mapped[20]
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
    
    while not stop_event.is_set():
        env.render()
        
        # 尝试获取共享数据
        current_frame = shared_dict.get('current_frame', None)
        if current_frame is not None:
            # 将数据转换为虚拟环境所需的格式
            R_robot_angle = np.concatenate((current_frame, np.zeros((5,)))).tolist()
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
    
    last_frame_index = -1
    
    while not stop_event.is_set():
        # 尝试获取共享数据
        current_frame = shared_dict.get('current_frame', None)
        if current_frame is not None:
            # 将数据转换为真实机器手所需的格式
            R_robot_angle = trans2realworld(current_frame)
            controller.control_hand(left_positions=R_robot_angle)
        
        time.sleep(0.02 * v_rate)
    
    controller.close()
    print("真实机器手进程结束")

def data_reader_process(h5_file_path, shared_dict, stop_event):
    """
    数据读取进程
    """
    print("数据读取进程启动")
    
    # 读取H5数据
    h5_file = h5py.File(h5_file_path, 'r')
    r_glove_angles = h5_file['outputs'][:]
    r_glove_angle_np = np.array(r_glove_angles)
    print('数据格式', r_glove_angle_np.shape)
    total_frames = r_glove_angle_np.shape[0]
    h5_file.close()
    
    frame_index = 0
    
    while not stop_event.is_set():
        # 更新共享字典中的当前帧数据
        shared_dict['current_frame'] = r_glove_angle_np[frame_index, 0, :]
        shared_dict['frame_index'] = frame_index
        
        print(f'当前帧数：{frame_index}')
        
        # 循环播放数据
        frame_index = (frame_index + 1) % total_frames
        
        # 等待一定时间后再更新下一帧
        time.sleep(0.05)  # 可根据需要调整帧率
    
    print("数据读取进程结束")

if __name__ == '__main__':
    # 设置项目根目录
    current_dir = os.path.dirname(os.path.abspath(__file__))
    sys.path.append(os.path.abspath(os.path.join(current_dir, "../..")))
    
    # H5文件路径
    h5_file_path = 'D:\\2026\\code\\TransHandR\\TransHandR\\model_outputs.h5'
    
    # 创建共享字典
    manager = multiprocessing.Manager()
    shared_dict = manager.dict()
    
    # 创建停止事件
    stop_event = multiprocessing.Event()
    v_rate = 10  # 可根据需要调整速度比例
    try:
        # 启动三个进程
        data_process = multiprocessing.Process(target=data_reader_process, args=(h5_file_path, shared_dict, stop_event))
        virtual_env_process_instance = multiprocessing.Process(target=virtual_env_process, args=(shared_dict, stop_event, v_rate))
        real_hand_process_instance = multiprocessing.Process(target=real_hand_process, args=(shared_dict, stop_event, v_rate))
        
        # 启动进程
        data_process.start()
        virtual_env_process_instance.start()
        real_hand_process_instance.start()
        
        print("所有进程已启动，按 Ctrl+C 停止...")
        
        # 等待用户中断
        while True:
            if keyboard.is_pressed('z'):
                print("检测到按键 'z'，正在停止程序...")
                break
            time.sleep(0.1)
    
    except KeyboardInterrupt:
        print("用户中断程序")
    
    finally:
        # 设置停止事件，通知所有进程退出
        stop_event.set()
        
        # 等待进程结束
        if 'data_process' in locals():
            data_process.join(timeout=2)
            if data_process.is_alive():
                data_process.terminate()
                
        if 'virtual_env_process_instance' in locals():
            virtual_env_process_instance.join(timeout=2)
            if virtual_env_process_instance.is_alive():
                virtual_env_process_instance.terminate()
                
        if 'real_hand_process_instance' in locals():
            real_hand_process_instance.join(timeout=2)
            if real_hand_process_instance.is_alive():
                real_hand_process_instance.terminate()
        
        print("程序已安全退出")