'''
调用模型将h5数据转换成虚拟环境机器手驱动
'''
import gymnasium as gym
import yumi_gym
import pybullet as p
import numpy as np
import h5py
import time
import math

def trans2realworld(angle):
    '''
    要将虚拟角度转换为真实角度,且检查是否超限,输入为弧度,下限,上限
    '''
    #给angle最后再补五个零以对齐实际机器手
    # print('角度',angle)
    angle_real = angle
    angle_real = np.concatenate((angle, np.zeros((5,))))
    # print('补零后角度',angle_real)
    return angle_real
# D:\2026\code\TransHandR\TransHandR\model_outputs.h5
#读取h5数据
# h5_file_path = 'D:\\2026\\code\\TransHandR\\TransHandR\\output\\h5\\slahmr\\linker\\linker_slahmr_output.h5'
hand_brand = 'linker'
# hand_brand = 'svhhand'
# hand_brand = 'shadow'
# hand_brand = 'inspire'
# ab_experiment_name = 'thumb'
# ab_experiment_name = 'thumb2'
ab_experiment_name = 'thumb3'
# h5_file_path = f'D:\\2026\\code\\TransHandR\\TransHandR\\output\\h5\\{hand_brand}\\{hand_brand}_output.h5'
h5_file_path = f'D:\\2026\\code\\TransHandR\\TransHandR\\output\\h5\\{ab_experiment_name}\\{hand_brand}\\{hand_brand}_output.h5'
# h5_file_path = f'D:\\2026\\code\\TransHandR\\TransHandR\\output\\h5\\{hand_brand}\\{hand_brand}_output1.h5'
# h5_file_path = 'D:\\2026\\code\\TransHandR\\TransHandR\\output\\comparing\\linker_visionpro10000_t.h5'
# h5_file_path = 'D:\\2026\\code\\test_other\\Testhand_retargeing\\hand_pose_retargeting-main\\data\\dexterous_hand_angles_linker.h5'
# h5_file_path = 'D:\\2026\\code\\TransHandR\\TransHandR\\dataset\\data\\h5'
h5_file = h5py.File(h5_file_path, 'r')
r_glove_angles = h5_file['outputs'][:]
r_glove_angle_np = np.array(r_glove_angles)
print('数据格式',r_glove_angle_np.shape)
total_frames = r_glove_angle_np.shape[0]
h5_file.close()
# 初始化虚拟环境
env = gym.make('yumi-v0')
observation, info = env.reset()
camera_distance = 2
camera_yaw = 90
camera_pitch = -10
camera_roll = 0
camera_target_position = [0, 0, 0.05]
paused = False
v_rate = 1
stop = False
while not stop:
    env.render()
    for t in range(total_frames):
        for i in range(2):
            # R_robot_angle = trans2realworld(r_glove_angle_np[t,0,:]).tolist()
            R_robot_angle = trans2realworld(r_glove_angle_np[t,:]).tolist()
            action = R_robot_angle
            # print('执行动作：',len(action))
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
                        print ('切换暂停')
        #     # 如果处于暂停状态，则跳过仿真步骤
            if paused:
                time.sleep(0.02)  # 保持短暂延迟以减少CPU占用
            p.resetDebugVisualizerCamera(cameraDistance=camera_distance,
                                            cameraYaw=camera_yaw,
                                            cameraPitch=camera_pitch,
                                            cameraTargetPosition=camera_target_position)
            observation, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            time.sleep(0.02*v_rate)
        print('当前帧数：',t)


