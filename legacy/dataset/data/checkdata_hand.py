'''
数据位于dataset/data/h5目录下
该脚本用于检查数据的hand部分数据
'''

import h5py
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

def load_h5_data(file_path, key):
    """
    加载H5文件中的数据
    """
    h5_file = h5py.File(file_path, 'r')
    data_group = h5_file[key]
    return data_group, h5_file

def get_hand_positions(data_group, frame):
    """
    获取指定帧的左右手位置
    """
    l_hand_pos = data_group['l_glove_pos'][frame]
    r_hand_pos = data_group['r_glove_pos'][frame]
    return l_hand_pos, r_hand_pos

def plot_hands_with_numbers(ax, l_hand_pos, r_hand_pos):
    """
    绘制左右手位置并标注序号
    """
    # 清除当前轴
    ax.cla()
    
    # 设置3D视图范围
    ax.set_xlim([-0.1, 0.1])
    ax.set_ylim([-0.1, 0.1])
    ax.set_zlim([-0.1, 0.1])
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    
    # 绘制左手位置并标注序号
    for i, point in enumerate(l_hand_pos):
        ax.scatter(point[0], point[1], point[2], c='blue', marker='o', s=20)
        ax.text(point[0], point[1], point[2], f'{i}', fontsize=8, color='black')
    
    # 绘制右手位置并标注序号
    for i, point in enumerate(r_hand_pos):
        ax.scatter(point[0], point[1], point[2], c='red', marker='o', s=20)
        ax.text(point[0], point[1], point[2], f'{i}', fontsize=8, color='black')
    
    # 添加图例
    ax.legend(['Left Hand Points', 'Right Hand Points'])
    # ax.legend(['Right Hand Points'])
    # ax.legend(['Left Hand Points'])


def animate(frame, data_group, ax):
    """
    动画更新函数
    """
    # 获取当前帧的手部位置
    l_hand_pos, r_hand_pos = get_hand_positions(data_group, frame)
    
    # 绘制手部位置并标注序号
    plot_hands_with_numbers(ax, l_hand_pos, r_hand_pos)
    
    # 设置标题显示当前帧
    ax.set_title(f'Hand Positions with Numbers - Frame: {frame}')

def visualize_hands(h5_file_path, data_key):
    """
    主可视化函数
    """
    # 加载数据
    data_group, h5_file = load_h5_data(h5_file_path, data_key)
    
    # 获取总帧数
    sample_data = data_group['l_glove_pos']
    frame_num = sample_data.shape[0]
    
    # 创建3D图形
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # 创建动画
    ani = FuncAnimation(fig, animate, frames=frame_num, 
                        fargs=(data_group, ax), 
                        interval=100, repeat=True)
    
    # 显示图形
    plt.show()
    
    # 关闭文件
    h5_file.close()

if __name__ == '__main__':
    # 文件路径和键值配置
    h5_file_path = 'D:\\2026\\code\\TransHandR\\TransHandR\\dataset\\data\\h5\\glove_data.h5'
    data_key = '测试-ceshi'
    
    # 运行动画可视化
    visualize_hands(h5_file_path, data_key)