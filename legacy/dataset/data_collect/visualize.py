# r['right_fingers']: np.ndarray 
#   # shape (25,4,4) / measured from right wrist frame 
# r['left_fingers']: np.ndarray 
#   # shape (25,4,4) / measured from left wrist 
# r['right_wrist']: np.ndarray 
#   # shape (1,4,4) / measured from ground frame
# r['left_wrist']: np.ndarray 
# #   shape (1,4,4) / measured from ground frame
import time
import cv2
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

#坐标系搬移
def translate_3d_points(xs,ys,zs,hand_side, offset_x=0, offset_y=0, offset_z=0):
    #右手的话对调z和x
    if hand_side == 'right_hand_data':
        xs,ys,zs = zs,ys,-xs
        return xs,ys,zs
    elif hand_side == 'left_hand_data':
        xs,ys,zs = -zs,-ys,xs
        return xs,ys,zs
def hand_3d_visualizer(shared_dict, hand_side, width=800, height=600, 
                      scale=300, offset_x=0, offset_y=0, invert_y=True):
    """
    单独进程中的3D手部可视化函数
    
    Args:
        shared_dict: 共享字典，包含手部数据
        hand_side: 手部标识 ('left' 或 'right')
        width: 窗口宽度
        height: 窗口高度
        scale: 缩放系数
        offset_x: X轴偏移像素
        offset_y: Y轴偏移像素
        invert_y: 是否反转Y轴（视觉效果）
    """
    
    # 创建可视化窗口
    window_name = f"{hand_side.capitalize()} Hand 3D Visualization"
    cv2.namedWindow(window_name, cv2.WINDOW_AUTOSIZE)
    
    while True:
        try:
            # 检查共享字典中是否有数据
            if hand_side in shared_dict and shared_dict[hand_side] is not None:
                # 获取手部数据
                hand_data = shared_dict[hand_side]
                
                # 创建黑色背景帧
                frame = np.zeros((height, width, 3), dtype=np.uint8)
                frame[:] = [20, 20, 20]
                
                # 绘制坐标系（考虑偏移）
                center_x, center_y = width // 2 + offset_x, height // 2 + offset_y
                cv2.line(frame, (center_x, center_y), (center_x + 100, center_y), (0, 0, 255), 2)  # X轴
                cv2.line(frame, (center_x, center_y), (center_x, center_y - 100), (0, 255, 0), 2)  # Y轴
                cv2.putText(frame, "X", (center_x + 110, center_y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                cv2.putText(frame, "Y", (center_x, center_y - 110), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                
                # 提取手指关节位置并绘制
                if 'fingers' in hand_data and hand_data['fingers'] is not None:
                    fingers = hand_data['fingers']
                    
                    # 绘制所有手指关节点
                    for i in range(fingers.shape[0]):
                        raw_x = float(fingers[i, 0, 3])
                        raw_y = float(fingers[i, 1, 3])
                        
                        x = int(center_x + raw_x * scale)
                        # 根据invert_y参数决定Y轴方向
                        y = int(center_y + (-raw_y if invert_y else raw_y) * scale)
                        
                        # 根据深度调整颜色
                        depth = float(fingers[i, 2, 3])
                        color_intensity = max(50, min(255, int(150 + depth * 100)))
                        cv2.circle(frame, (x, y), 5, (color_intensity, 150, 255-color_intensity), -1)
                
                # 绘制手腕位置（如果有）
                if 'wrist' in hand_data and hand_data['wrist'] is not None:
                    wrist = hand_data['wrist']
                    raw_wx = float(wrist[0, 0, 3])
                    raw_wy = float(wrist[0, 1, 3])
                    
                    wx = int(center_x + raw_wx * scale)
                    wy = int(center_y + (-raw_wy if invert_y else raw_wy) * scale)
                    
                    # 根据捏合状态改变颜色
                    pinch_state = hand_data.get('pinch', 0)
                    color = (0, 255, 0) if pinch_state < 0.02 else (0, 100, 255)
                    cv2.circle(frame, (wx, wy), 15, color, -1)
                    cv2.putText(frame, hand_side[0].upper(), (wx-8, wy+5), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                
                # 显示捏合信息
                if 'pinch' in hand_data:
                    cv2.putText(frame, f"Pinch: {hand_data['pinch']:.3f}", (20, 30),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                
                cv2.imshow(window_name, frame)
            
            else:
                # 没有数据时显示等待信息
                frame = np.zeros((height, width, 3), dtype=np.uint8)
                frame[:] = [20, 20, 20]
                cv2.putText(frame, f"Waiting for {hand_side} hand data...", 
                           (width//2 - 200, height//2),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
                cv2.imshow(window_name, frame)
            
            # 检查按键事件，允许退出
            if cv2.waitKey(30) & 0xFF == ord('q'):
                break
                
        except Exception as e:
            print(f"Visualization error for {hand_side} hand: {e}")
            time.sleep(0.1)
    
    cv2.destroyWindow(window_name)

def hand_3d_visualizer_plot(shared_dict, hand_side, figsize=(10, 8)):
    """
    使用matplotlib进行3D手部可视化函数
    
    Args:
        shared_dict: 共享字典，包含手部数据
        hand_side: 手部标识 ('left' 或 'right')
        figsize: 图形大小
    """
    
    # 设置图形和3D轴
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, projection='3d')
    fig.canvas.manager.set_window_title(f'{hand_side.capitalize()} Hand 3D Visualization')
    
    # 设置坐标轴标签
    ax.set_xlabel('X axis')
    ax.set_ylabel('Y axis')
    ax.set_zlabel('Z axis')
    
    # 设置坐标轴范围
    ax.set_xlim([-0.5, 0.5])
    ax.set_ylim([-0.5, 0.5])
    ax.set_zlim([-0.5, 0.5])
    
    # 设置标题
    ax.set_title(f'{hand_side.capitalize()} Hand 3D Positions')
    
    # 初始化绘图元素
    finger_points = ax.scatter([], [], [], c='blue', s=50)
    wrist_point = ax.scatter([], [], [], c='red', s=200)
    
    plt.ion()  # 开启交互模式
    plt.show()
    
    try:
        while True:
            # 检查共享字典中是否有数据
            if hand_side in shared_dict and shared_dict[hand_side] is not None:
                # 获取手部数据
                hand_data = shared_dict[hand_side]
                
                # 清空之前的绘图
                ax.clear()
                
                # 重新设置坐标轴
                ax.set_xlabel('X axis')
                ax.set_ylabel('Y axis')
                ax.set_zlabel('Z axis')
                ax.set_xlim([-0.5, 0.5])
                ax.set_ylim([-0.5, 0.5])
                ax.set_zlim([-0.5, 0.5])
                ax.set_title(f'{hand_side.capitalize()} Hand 3D Positions')
                
                # 绘制坐标系参考线
                ax.plot([0, 0.3], [0, 0], [0, 0], 'r-', linewidth=2, label='X axis')
                ax.plot([0, 0], [0, 0.3], [0, 0], 'g-', linewidth=2, label='Y axis')
                ax.plot([0, 0], [0, 0], [0, 0.3], 'b-', linewidth=2, label='Z axis')
                
                # 提取手指关节位置并绘制
                if 'fingers' in hand_data and hand_data['fingers'] is not None:
                    fingers = hand_data['fingers']
                    
                    # 提取所有手指关节点的坐标
                    xs = [float(fingers[i, 0, 3]) for i in range(fingers.shape[0])]
                    ys = [float(fingers[i, 1, 3]) for i in range(fingers.shape[0])]
                    zs = [float(fingers[i, 2, 3]) for i in range(fingers.shape[0])]
                    
                    # 根据捏合状态设置颜色
                    pinch_state = hand_data.get('pinch', 0)
                    colors = ['green' if pinch_state < 0.02 else 'lightcoral'] * len(xs)
                    
                    # 绘制手指关节点
                    ax.scatter(xs, ys, zs, c=colors, s=50, alpha=0.7)
                    
                    # 可选：连接手指关节形成骨架
                    # 绘制拇指
                    thumb_indices = list(range(0, 5))  # 假设前5个点是拇指
                    if len(thumb_indices) <= len(xs):
                        thumb_xs = [xs[i] for i in thumb_indices]
                        thumb_ys = [ys[i] for i in thumb_indices]
                        thumb_zs = [zs[i] for i in thumb_indices]
                        ax.plot(thumb_xs, thumb_ys, thumb_zs, 'yellow', linewidth=2)
                    
                    # 绘制食指
                    index_indices = list(range(5, 9))  # 假设接下来4个点是食指
                    if len(index_indices) <= len(xs):
                        index_xs = [xs[i] for i in index_indices]
                        index_ys = [ys[i] for i in index_indices]
                        index_zs = [zs[i] for i in index_indices]
                        ax.plot(index_xs, index_ys, index_zs, 'orange', linewidth=2)
                
                # 绘制手腕位置（如果有）
                if 'wrist' in hand_data and hand_data['wrist'] is not None:
                    wrist = hand_data['wrist']
                    wx = float(wrist[0, 0, 3])
                    wy = float(wrist[0, 1, 3])
                    wz = float(wrist[0, 2, 3])
                    
                    # 根据捏合状态改变颜色
                    pinch_state = hand_data.get('pinch', 0)
                    wrist_color = 'green' if pinch_state < 0.02 else 'red'
                    
                    ax.scatter(wx, wy, wz, c=wrist_color, s=200, marker='o')
                    ax.text(wx, wy, wz, f'{hand_side[0].upper()} Wrist', fontsize=9)
                
                # 添加图例
                ax.legend()
                
            else:
                # 没有数据时清空图形并显示等待信息
                ax.clear()
                ax.text(0, 0, 0, f'Waiting for {hand_side} hand data...', 
                       fontsize=12, ha='center', va='center')
                ax.set_xlim([-1, 1])
                ax.set_ylim([-1, 1])
                ax.set_zlim([-1, 1])
            
            # 更新图形
            plt.draw()
            plt.pause(0.01)  # 短暂停留以允许图形更新
            
            # 检查是否关闭了窗口
            if not plt.fignum_exists(fig.number):
                break
                
    except Exception as e:
        print(f"3D Visualization error for {hand_side} hand: {e}")
    finally:
        plt.close(fig)

def enhanced_hand_3d_visualizer_with_skeleton(shared_dict, hand_side, figsize=(12, 9)):
    """
    增强版3D手部可视化函数，包含完整的手部骨架连接
    
    Args:
        shared_dict: 共享字典，包含手部数据
        hand_side: 手部标识 ('left' 或 'right')
        figsize: 图形大小
    """
    
    # 设置图形和3D轴
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, projection='3d')
    fig.canvas.manager.set_window_title(f'{hand_side.capitalize()} Hand Enhanced 3D Visualization with Skeleton')
    
    plt.ion()
    plt.show()
    # 设置坐标轴样式
    # ax.set_xlabel('X (meters)', fontsize=10)
    # ax.set_ylabel('Y (meters)', fontsize=10)
    # ax.set_zlabel('Z (meters)', fontsize=10)
    ax.set_title(f'{hand_side.capitalize()} Hand 3D Visualization\n(Press close button to exit)', 
                fontsize=12, pad=20)
    # 自动调整坐标轴范围
    margin = 0.2
    ax.set_xlim([-0.3 - margin, 0.3 + margin])
    ax.set_ylim([-0.3 - margin, 0.3 + margin])
    ax.set_zlim([-0.3 - margin, 0.3 + margin])
    ax.set_xlabel('X')
    ax.set_ylabel('Y') 
    ax.set_zlabel('Z')
    try:
        while True:
            # 检查共享字典中是否有数据
            if hand_side in shared_dict and shared_dict[hand_side] is not None:
                # 获取手部数据
                hand_data = shared_dict[hand_side]
                # 清空图形
                ax.clear()
                ax.set_xlabel('X')
                ax.set_ylabel('Y') 
                ax.set_zlabel('Z')

                # 提取手指关节位置并绘制
                if 'fingers' in hand_data and hand_data['fingers'] is not None:
                    fingers = hand_data['fingers']
                    # 提取坐标
                    xs = np.array([float(fingers[i, 0, 3]) for i in range(fingers.shape[0])])
                    ys = np.array([float(fingers[i, 1, 3]) for i in range(fingers.shape[0])])
                    zs = np.array([float(fingers[i, 2, 3]) for i in range(fingers.shape[0])])
                    xs, ys, zs = translate_3d_points(xs, ys, zs, hand_side)
                    # 根据深度设置颜色映射
                    colors = zs  # 使用Z坐标作为颜色映射
                    
                    # 绘制手指关节点
                    scatter = ax.scatter(xs, ys, zs, c=colors, cmap='viridis', s=60, alpha=0.8)
                    
                    # 定义完整的手部骨架连接（基于提供的图像）
                    # 连接关系：每个点连接到其父节点
                    skeleton_connections = [
                        # 拇指
                        [0, 1], [1, 2], [2, 3], [3, 4],
                        # 食指
                        [0, 5], [5, 6], [6, 7], [7, 8],[8, 9],
                        # 中指
                        [0,10 ],  [10, 11], [11, 12], [12, 13], [13, 14],
                        # 无名指
                        [0, 15],  [15, 16], [16, 17], [17, 18], [18, 19],
                        # 小指
                        [0, 20], [20, 21], [21, 22], [22, 23], [23, 24],
                    ]
                    
                    # 绘制骨架连线
                    for connection in skeleton_connections:
                        if connection[0] < len(xs) and connection[1] < len(xs):
                            # 不同手指使用不同颜色
                            finger_colors = ['yellow', 'orange', 'cyan', 'magenta', 'lightcoral']
                            
                            # 确定是哪个手指
                            finger_index = -1
                            if connection[0] == 0:  # 腕部连接
                                if connection[1] in [1, 2, 3, 4]:  # 拇指
                                    finger_index = 0
                                elif connection[1] in [5, 6, 7, 8, 9]:  # 食指
                                    finger_index = 1
                                elif connection[1] in [10, 11, 12, 13, 14]:  # 中指
                                    finger_index = 2
                                elif connection[1] in [15, 16, 17, 18, 19]:  # 无名指
                                    finger_index = 3
                                elif connection[1] in [20, 21, 22, 23, 24]:  # 小指
                                    finger_index = 4
                            # 设置线条颜色
                            line_color = finger_colors[finger_index] if finger_index >= 0 else 'gray'
                            
                            ax.plot([xs[connection[0]], xs[connection[1]]],
                                   [ys[connection[0]], ys[connection[1]]],
                                   [zs[connection[0]], zs[connection[1]]],
                                   color=line_color, alpha=0.7, linewidth=2)
                
                # 绘制手腕位置（如果有）
                # if 'wrist' in hand_data and hand_data['wrist'] is not None:
                #     wrist = hand_data['wrist']
                #     wx = float(wrist[0, 0, 3])
                #     wy = float(wrist[0, 1, 3])
                #     wz = float(wrist[0, 2, 3])
                    
                #     # 根据捏合状态改变标记
                #     pinch_state = hand_data.get('pinch', 0)
                #     wrist_marker = 'o' if pinch_state < 0.02 else '^'
                #     wrist_color = 'green' if pinch_state < 0.02 else 'red'
                    
                #     ax.scatter(wx, wy, wz, c=wrist_color, s=300, marker=wrist_marker, alpha=0.9)
                #     # ax.text(wx, wy, wz + 0.05, f'{hand_side[0].upper()}-Wrist\nPinch:{pinch_state:.3f}', 
                #     #        fontsize=8, ha='center')
                
                # 添加颜色条（如果有点被绘制）
                # if 'fingers' in hand_data and hand_data['fingers'] is not None:
                    # plt.colorbar(scatter, ax=ax, shrink=0.5, aspect=5, label='Depth (Z-axis)')
                
            else:
                # 没有数据时显示等待信息
                # ax.clear()
                # ax.text(0, 0, 0, f'Waiting for {hand_side} hand data...\nPlease ensure VisionPro is connected.', 
                #        fontsize=12, ha='center', va='center')
                # ax.set_xlim([-1, 1])
                # ax.set_ylim([-1, 1])
                # ax.set_zlim([-1, 1])
                ax.set_title('Waiting for Data')
            
            # 更新图形
            plt.draw()
            plt.pause(0.05)  # 控制刷新率
            
            # # 检查是否关闭了窗口
            # if shared_dict.get('data_process_running', False):
            #     break
                
    except Exception as e:
        print(f"Enhanced 3D Visualization error for {hand_side} hand: {e}")
    finally:
        plt.close(fig)

