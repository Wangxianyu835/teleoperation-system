# data_collect_visionpro.py
import multiprocessing as mp
import numpy as np
import cv2
import time
import h5py
from avp_stream import VisionProStreamer
from visualize import hand_3d_visualizer_plot, hand_3d_visualizer, enhanced_hand_3d_visualizer_with_skeleton
import threading
def data_collector_process(avp_ip, shared_dict):
    """
    数据收集进程
    
    Args:
        avp_ip: VisionPro设备IP地址
        shared_dict: 共享字典用于存储手部数据
    """
    # 创建VisionPro流媒体对象
    streamer = VisionProStreamer(ip=avp_ip)
    
    print(f"Connecting to VisionPro at {avp_ip}...")
    
    try:
        frame_count = 1  # 初始化帧标签
        while shared_dict.get('data_process_running', True):
            # 获取最新数据
            latest = streamer.get_latest()
            
            if latest is not None:
                # 处理左手数据
                left_fingers = latest.get('left_fingers')
                right_fingers = latest.get('right_fingers')
                
                # 检查左右手数据是否都存在
                if left_fingers is not None and right_fingers is not None:
                    # 提取坐标数据（只取位置信息，忽略旋转）
                    left_coords = left_fingers[:, :3, 3]  # (25, 3) - x, y, z coordinates
                    right_coords = right_fingers[:, :3, 3]  # (25, 3) - x, y, z coordinates
                    
                    # 更新共享字典中的保存数据（标记为需要保存）
                    shared_dict['save_data'] = {
                        'left_coords': left_coords,
                        'right_coords': right_coords,
                        'frame_id': frame_count,
                        'need_save': True,
                        'timestamp': time.time()
                    }
                    
                    frame_count += 1  # 增加帧标签
                else:
                    # 标记为不需要保存
                    shared_dict['save_data'] = {
                        'need_save': False
                    }
                
                # 更新共享字典（用于左手可视化）
                if left_fingers is not None:
                    left_data = {
                        'fingers': left_fingers,
                        'wrist': latest.get('left_wrist'),
                        'pinch': latest.get('left_pinch_distance', 0)
                    }
                    shared_dict['left_hand_data'] = left_data
                else:
                    shared_dict['left_hand_data'] = None
                
                # 更新共享字典（用于右手可视化）
                if right_fingers is not None:
                    right_data = {
                        'fingers': right_fingers,
                        'wrist': latest.get('right_wrist'),
                        'pinch': latest.get('right_pinch_distance', 0)
                    }
                    shared_dict['right_hand_data'] = right_data
                else:
                    shared_dict['right_hand_data'] = None
            
            # 控制数据采集频率
            time.sleep(1/120.)  # 120 FPS
            
    except KeyboardInterrupt:
        print("Data collection stopped.")
        # 标记数据收集进程结束
        shared_dict['data_process_running'] = False
    except Exception as e:
        print(f"Data collection error: {e}")
        # 标记数据收集进程结束
        shared_dict['data_process_running'] = False

def data_saver_process(shared_dict, output_file='hand_data_0204.h5'):
    """
    数据保存进程，将数据保存到HDF5文件中
    
    Args:
        shared_dict: 共享字典用于获取数据
        output_file: 输出文件名
    """
    # 创建HDF5文件
    with h5py.File(output_file, 'w') as f:
        # 创建数据集
        left_dataset = f.create_dataset('left_hand', (0, 25, 3), maxshape=(None, 25, 3), dtype=np.float32)
        right_dataset = f.create_dataset('right_hand', (0, 25, 3), maxshape=(None, 25, 3), dtype=np.float32)
        frame_dataset = f.create_dataset('frame_ids', (0,), maxshape=(None,), dtype=np.int32)
        
        try:
            saved_frames = 0
            last_timestamp = 0
            
            while shared_dict.get('data_process_running', True):
                # 检查是否有需要保存的数据
                if 'save_data' in shared_dict and shared_dict['save_data'] is not None:
                    save_data = shared_dict['save_data']
                    
                    # 检查是否需要保存且是新数据
                    if (save_data.get('need_save', False) and 
                        save_data.get('timestamp', 0) > last_timestamp):
                        
                        # 添加数据到HDF5文件
                        left_dataset.resize(left_dataset.shape[0] + 1, axis=0)
                        left_dataset[-1] = save_data['left_coords']
                        
                        right_dataset.resize(right_dataset.shape[0] + 1, axis=0)
                        right_dataset[-1] = save_data['right_coords']
                        
                        frame_dataset.resize(frame_dataset.shape[0] + 1, axis=0)
                        frame_dataset[-1] = save_data['frame_id']
                        
                        # 更新最后处理的时间戳
                        last_timestamp = save_data['timestamp']
                        saved_frames += 1
                        
                        # 打印进度
                        if saved_frames % 100 == 0:
                            print(f"Saved {saved_frames} frames to {output_file}")
                
                #1200帧后停止
                if saved_frames >= 2400:
                    break
                # 控制检查频率
                time.sleep(1/120.)  # 120 FPS
                
            print(f"Data saving completed. Total frames saved: {saved_frames}")
                    
        except Exception as e:
            print(f"Data saving error: {e}")

def key_listener(shared_dict):
    while shared_dict.get('data_process_running', True):
        user_input = input()
        if user_input.strip().lower() == 'q':
            print("Received 'q' input. Stopping data collection.")
            shared_dict['data_process_running'] = False
            break

def main():
    # VisionPro IP地址
    avp_ip = "192.168.43.20"
    
    # 创建管理器和共享字典
    manager = mp.Manager()
    shared_dict = manager.dict()
    
    # 初始化共享字典（使用分离的键值）
    shared_dict['left_hand_data'] = None
    shared_dict['right_hand_data'] = None
    shared_dict['save_data'] = None
    shared_dict['data_process_running'] = True  # 标记数据收集进程是否运行
    
    # 创建进程
    data_process = mp.Process(target=data_collector_process, args=(avp_ip, shared_dict))
    
    # 创建独立的可视化进程，分别使用不同的键值
    left_viz_process = mp.Process(target=enhanced_hand_3d_visualizer_with_skeleton, 
                                 args=(shared_dict, 'left_hand_data'))  # 使用left_hand_data键
    right_viz_process = mp.Process(target=enhanced_hand_3d_visualizer_with_skeleton, 
                                  args=(shared_dict, 'right_hand_data'))  # 使用right_hand_data键
    
    # 创建数据保存进程
    saver_process = mp.Process(target=data_saver_process, args=(shared_dict, 'hand_data_0204.h5'))
    # 在 main 函数中创建并启动监听线程
    key_thread = threading.Thread(target=key_listener, args=(shared_dict,))
    key_thread.daemon = True
    key_thread.start()
    # 启动所有进程
    data_process.start()
    left_viz_process.start()
    right_viz_process.start()
    saver_process.start()
    
    print("Processes started:")
    print("- Data collection process")
    print("- Left hand visualization process")
    print("- Right hand visualization process")
    print("- Data saving process")
    print("Press Ctrl+C to stop all processes")
    print("In visualization windows, press 'q' to close them")
    
    try:
        # 等待所有进程完成
        data_process.join()
        left_viz_process.join()
        right_viz_process.join()
        saver_process.join()
        
    except KeyboardInterrupt:
        print("\nTerminating processes...")
        data_process.terminate()
        left_viz_process.terminate()
        right_viz_process.terminate()
        saver_process.terminate()
        
        # 等待进程结束
        data_process.join()
        left_viz_process.join()
        right_viz_process.join()
        saver_process.join()
        
        print("All processes terminated.")

if __name__ == "__main__":
    main()