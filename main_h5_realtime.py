'''
调用模型将h5数据转换成虚拟环境机器手驱动
'''
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
# 关节映射字典 25个驱动器对应18个关节，即0-24驱动器对应0-17关节。其中11-14驱动器为空，16-19驱动器为空
# 剩下的一一对应
joint_map = {0: 15, 1: 2, 2: 5, 3: 8, 4: 11, 5: 14, 6: 1, 7: 4, 8: 7, 9: 10, 10: 13, 
            11: 0, 12: 0, 13: 0, 14: 0, 15: 16, 16: 0, 17: 0, 18: 0, 19: 0,
            20: 17, 21: 3, 22: 6, 23: 9, 24: 12}
def trans2realworld(angle):
    '''
    要将虚拟角度转换为真实角度,且检查是否超限,输入为弧度,下限,上限
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
    # print('角度',angle)
    # print('转后角度',angle_real)
    # print('转后角度',angle_mapped)
    return angle_mapped

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
def unit(num):
    #限制在0-255
    return 0 if num < 0 else 255 if num > 255 else num

if __name__ == '__main__':
    # 设置项目根目录
    current_dir = os.path.dirname(os.path.abspath(__file__))
    sys.path.append(os.path.abspath(os.path.join(current_dir, "../..")))
    #读取h5数据
    h5_file_path = "D:\\2026\\code\\TransHandR\\TransHandR\\output\\h5\\linker\\linker_output.h5"
    h5_file = h5py.File(h5_file_path, 'r')
    r_glove_angles = h5_file['outputs'][:]
    r_glove_angle_np = np.array(r_glove_angles)
    print('数据格式',r_glove_angle_np.shape)
    total_frames = r_glove_angle_np.shape[0]
    h5_file.close()
    # 初始化参数
    stop = False
    v_rate = 1
    print('左手大拇指角度全部需要取0-255的反')
    # 初始化手部控制器
    controller = HandController(
            left_positions=trans2realworld(r_glove_angle_np[0,:])
            # left_positions=trans2realworld(r_glove_angle_np[0,0,:])
        )
    try:
        while not stop:
            for t in range(total_frames):
                R_robot_angle = trans2realworld(r_glove_angle_np[t,:])
                # R_robot_angle = trans2realworld(r_glove_angle_np[t,0,:])
                # 输出为list
                controller.control_hand(
                    left_positions=R_robot_angle
                )
                time.sleep(0.02*v_rate)
                print('当前帧数：',t)
                if keyboard.is_pressed('z'):
                    print(R_robot_angle)
                    break
                time.sleep(0.033*v_rate)
                # break
            # break
    except KeyboardInterrupt:
        ColorMsg(msg="用户中断", color="yellow")
    finally:
        if controller:
            controller.close()


