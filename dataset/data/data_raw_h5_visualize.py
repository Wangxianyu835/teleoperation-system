import h5py
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import time
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
# 检查网络输入h5文件结构
def explore_h5_structure(file_path):
    """
    探索H5文件的结构
    """
    def print_structure(name, obj):
        print(f"{name}: {type(obj)}")
        if hasattr(obj, 'shape'):
            print(f"  Shape: {obj.shape}")
        if hasattr(obj, 'dtype'):
            print(f"  Dtype: {obj.dtype}")
        if hasattr(obj, 'keys'):
            for key in obj.keys():
                print(f"  Key: {key}")

    with h5py.File(file_path, 'r') as f:
        print("H5文件结构:")
        f.visititems(print_structure)

def load_and_visualize_h5_with_connections(h5_file_path, subject_key="测试-ceshi", connections=None):
    """
    读取H5文件，提取指定主题的第一帧数据并可视化（带连接关系）
    
    参数:
        h5_file_path: H5文件路径
        subject_key: 主题键名，默认为"测试-ceshi"
        connections: 关节连接关系列表，格式为 [(joint1_idx, joint2_idx), ...]
                    如果为None，将使用默认连接（相邻关节相连）
    """
    with h5py.File(h5_file_path, 'r') as f:
        print(f"文件中的所有键: {list(f.keys())}")
        
        # 检查指定的主题键是否存在
        if subject_key not in f:
            print(f"键 '{subject_key}' 不存在，可用的键: {list(f.keys())}")
            return
        
        # 获取主题数据
        subject_data = f[subject_key]
        print(f"主题 '{subject_key}' 包含的键: {list(subject_data.keys())}")
        
        # 查找包含3D数据的数据集
        dataset_name = None
        for key in subject_data.keys():
            sub_data = subject_data[key]
            if isinstance(sub_data, h5py.Dataset):
                if len(sub_data.shape) >= 3 and sub_data.shape[-1] == 3:  # 3D数据格式 [帧数, 关节数, 3]
                    dataset_name = key
                    break
        dataset_name = 'r_glove_pos'
        print(dataset_name)
        if dataset_name is None:
            # 如果没找到标准格式，尝试查找任何数据集
            for key in subject_data.keys():
                sub_data = subject_data[key]
                if isinstance(sub_data, h5py.Dataset):
                    dataset_name = key
                    break
        
        if dataset_name is None:
            print(f"在主题 '{subject_key}' 中未找到数据集")
            return
        
        print(f"选择数据集: {dataset_name}")
        data = subject_data[dataset_name]
        print(f"数据形状: {data.shape}")
        
        # 检查数据维度并提取第一帧
        if len(data.shape) == 3:  # [帧数, 关节数, 3]
            if data.shape[0] > 0:
                first_frame = data[0]  # [关节数, 3]
            else:
                print("数据集中没有帧数据")
                return
        elif len(data.shape) == 2:  # [关节数, 3] - 可能只有一帧
            first_frame = data
        else:
            print(f"意外的数据形状: {data.shape}")
            return
        
        print(f"第一帧数据形状: {first_frame.shape}")
        print(f"第一帧数据:\n{first_frame[:]}")
        
        # 如果没有提供连接关系，使用默认连接
        if connections is None:
            # 默认：相邻关节相连
            connections = [(i, i+1) for i in range(len(first_frame)-1)]
        
        visualize_3d_joints_with_connections(first_frame, connections, 
                                          title=f"第一帧 - {subject_key}/{dataset_name}")

def visualize_3d_joints_with_connections(joints, connections, title="3D Joints with Connections"):
    """
    可视化3D关节数据（带连接线）
    
    参数:
        joints: 关节数据，形状为 [关节数量, 3]
        connections: 连接关系列表
        title: 图形标题
    """
    fig = plt.figure(figsize=(12, 9))
    ax = fig.add_subplot(111, projection='3d')
    
    x = joints[:, 0]
    y = joints[:, 1] 
    z = joints[:, 2]
    
    # 绘制关节点
    ax.scatter(x, y, z, c='red', s=100, alpha=0.8, label='Joints')
    
    # 绘制连接线
    for conn in connections:
        joint1_idx, joint2_idx = conn
        if joint1_idx < len(joints) and joint2_idx < len(joints):
            ax.plot([x[joint1_idx], x[joint2_idx]], 
                   [y[joint1_idx], y[joint2_idx]], 
                   [z[joint1_idx], z[joint2_idx]], 'b-', alpha=0.6, linewidth=2)
    
    # 添加关节编号
    for i in range(min(len(joints), 25)):  # 只标记前25个关节，避免图表过于拥挤
        ax.text(x[i], y[i], z[i], f'{i}', fontsize=12, color='black', 
                bbox=dict(boxstyle="round,pad=0.1", facecolor="white", alpha=0.7))
    
    ax.set_xlabel('X')
    ax.set_ylabel('Y') 
    ax.set_zlabel('Z')
    ax.set_title(title)
    
    # 设置相等的坐标轴比例
    max_range = np.array([x.max()-x.min(), y.max()-y.min(), z.max()-z.min()]).max() / 2.0
    mid_x = (x.max()+x.min()) * 0.5
    mid_y = (y.max()+y.min()) * 0.5
    mid_z = (z.max()+z.min()) * 0.5
    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range) 
    ax.set_zlim(mid_z - max_range, mid_z + max_range)
    
    plt.legend()
    plt.tight_layout()
    plt.show()

def animate_3d_joints_with_connections(h5_file_path, subject_key="测试-ceshi", dataset_name='r_glove_pos', connections=None, interval=200):
    """
    动态循环播放H5文件中的多帧3D关节数据
    
    参数: 
        h5_file_path: H5文件路径
        subject_key: 主题键名
        dataset_name: 数据集名称
        connections: 连接关系列表
        interval: 帧之间的间隔时间（毫秒）
    """
    import matplotlib.animation as animation
    
    with h5py.File(h5_file_path, 'r') as f:
        if subject_key not in f:
            print(f"键 '{subject_key}' 不存在，可用的键: {list(f.keys())}")
            return
        
        subject_data = f[subject_key]
        
        if dataset_name not in subject_data:
            print(f"数据集 '{dataset_name}' 不存在，可用的数据集: {list(subject_data.keys())}")
            return
        
        data = subject_data[dataset_name]
        
        if len(data.shape) != 3:
            print(f"数据必须是3维格式 [帧数, 关节数, 3]，当前形状: {data.shape}")
            return
        
        num_frames = data.shape[0]
        num_joints = data.shape[1]
        
        print(f"总帧数: {num_frames}, 关节数: {num_joints}")
        
        if connections is None:
            connections = [(i, i+1) for i in range(num_joints-1)]
        
        # 初始化图形
        fig = plt.figure(figsize=(12, 9))
        ax = fig.add_subplot(111, projection='3d')
        
        # 获取整个数据的时间范围，用于设置坐标轴
        all_x = data[:, :, 0].flatten()
        all_y = data[:, :, 1].flatten()
        all_z = data[:, :, 2].flatten()
        
        max_range = np.array([all_x.max()-all_x.min(), all_y.max()-all_y.min(), all_z.max()-all_z.min()]).max() / 2.0
        mid_x = (all_x.max()+all_x.min()) * 0.5
        mid_y = (all_y.max()+all_y.min()) * 0.5
        mid_z = (all_z.max()+all_z.min()) * 0.5
        ax.set_xlim(mid_x - max_range, mid_x + max_range)
        ax.set_ylim(mid_y - max_range, mid_y + max_range)
        ax.set_zlim(mid_z - max_range, mid_z + max_range)
        
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        
        # 创建线条和散点对象
        lines = []
        for conn in connections:
            line, = ax.plot([], [], [], 'b-', alpha=0.6, linewidth=2)
            lines.append(line)
        
        scatter = ax.scatter([], [], [], c='red', s=100, alpha=0.8)
        
        frame_text = ax.text2D(0.02, 0.95, "", transform=ax.transAxes, fontsize=12)
        
        def update_frame(frame_idx):
            current_frame = data[frame_idx]
            
            x = current_frame[:, 0]
            y = current_frame[:, 1]
            z = current_frame[:, 2]
            
            # 更新散点
            scatter._offsets3d = (x, y, z)
            
            # 更新连接线
            for i, conn in enumerate(connections):
                joint1_idx, joint2_idx = conn
                if joint1_idx < len(current_frame) and joint2_idx < len(current_frame):
                    lines[i].set_data([x[joint1_idx], x[joint2_idx]], 
                                     [y[joint1_idx], y[joint2_idx]])
                    lines[i].set_3d_properties([z[joint1_idx], z[joint2_idx]])
            
            # 更新帧数显示
            frame_text.set_text(f'Frame: {frame_idx + 1}/{num_frames}')
            
            # # 添加关节编号
            # for j in range(min(len(current_frame), 25)):
            #     ax.texts[j]._position3d = (x[j], y[j], z[j])
            
            return lines + [scatter, frame_text]
        
        # 创建动画
        ani = animation.FuncAnimation(
            fig, 
            update_frame, 
            frames=num_frames,
            interval=interval,
            blit=False,
            repeat=True  # 循环播放
        )
        time.sleep(0.001)
        plt.title(f"动态播放 - {subject_key}/{dataset_name}")
          # 确保动画对象被创建
        # 添加关节编号文本对象
        # for i in range(min(num_joints, 25)):
        #     ax.text(0, 0, 0, f'{i}', fontsize=12, color='black',
        #            bbox=dict(boxstyle="round,pad=0.1", facecolor="white", alpha=0.7))
        
        plt.show()
        
        return ani  # 返回动画对象，防止被垃圾回收
    
# 使用示例
if __name__ == "__main__":
    # 替换为您的H5文件路径
    # h5_file_path = "D:\\2026\\code\\TransHandR\\TransHandR\\dataset\\data\\h5\\sign_glove_aligned.h5"  # 根据您的实际路径修改
    h5_file_path = "D:\\2026\\code\\TransHandR\\TransHandR\\dataset\\data\\h5\\glove_data_aligned.h5"  # 根据您的实际路径修改
    # h5_file_path = "D:\\2026\\code\\TransHandR\\TransHandR\\dataset\\data\\h5\\test_glove_data0204_aligned.h5"  # 根据您的实际路径修改
    # 定义手部关节连接关系（25个关节的示例）
    hand_connections = [
        # 手掌连接
        (0, 1), (1, 2), (2, 3), (3, 4),  # 拇指
        (0, 5), (5, 6), (6, 7), (7, 8), (8, 9), # 食指
        (0, 10), (10, 11), (11, 12), (12, 13), (13, 14), # 中指
        (0, 15), (15, 16), (16, 17), (17, 18), (18, 19), # 无名指
        (0, 20), (20, 21), (21, 22), (22, 23), (23, 24) # 小指
    ]
    
    # hand_connections = [
    #     (0,13), (13,14), (14,15), # 大拇指
    #     (0, 1), (1, 2),  (2, 3), # 食指
    #     (0, 4), (4, 5),  (5, 6), # 中指
    #     (0, 10), (10, 11), (11, 12), # 无名指
    #     (0, 7), (7, 8),  (8, 9) # 小指
    # ]
    try:
        # 探索文件结构
        # print("探索H5文件结构...")
        # explore_h5_structure(h5_file_path)
        # print("\n" + "="*50 + "\n")
        
        # 加载并可视化指定主题的数据
        load_and_visualize_h5_with_connections(h5_file_path, subject_key="测试-ceshi", connections=hand_connections)
        
        # 或者使用自定义连接关系
        # load_and_visualize_h5_with_connections(h5_file_path, subject_key="S000001_P0000", connections=hand_connections)
        # animate_3d_joints_with_connections(h5_file_path, subject_key="S000001_P0000", connections=hand_connections, interval=50)
        # 动态播放多帧数据
        animate_3d_joints_with_connections(h5_file_path, subject_key="测试-ceshi", connections=hand_connections, interval=25)
    except FileNotFoundError:
        print(f"文件 {h5_file_path} 未找到")
    except Exception as e:
        print(f"读取文件时出错: {e}")