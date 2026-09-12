# visualize_separate.py
import multiprocessing as mp
import numpy as np
# import cv2
import time
from avp_stream import VisionProStreamer
from visualize import hand_3d_visualizer_plot,hand_3d_visualizer,enhanced_hand_3d_visualizer_with_skeleton


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
        while True:
            # 获取最新数据
            latest = streamer.get_latest()
            
            if latest is not None:
                # 处理左手数据
                if latest.get('left_fingers') is not None:
                    left_data = {
                        'fingers': latest['left_fingers'],
                        'wrist': latest.get('left_wrist'),
                        'pinch': latest.get('left_pinch_distance', 0)
                    }
                    shared_dict['left'] = left_data
                else:
                    shared_dict['left'] = None
                
                # 处理右手数据
                if latest.get('right_fingers') is not None:
                    right_data = {
                        'fingers': latest['right_fingers'],
                        'wrist': latest.get('right_wrist'),
                        'pinch': latest.get('right_pinch_distance', 0)
                    }
                    shared_dict['right'] = right_data
                else:
                    shared_dict['right'] = None
            
            # 控制数据采集频率
            time.sleep(1/30.)  # 30 FPS
            
    except KeyboardInterrupt:
        print("Data collection stopped.")
    except Exception as e:
        print(f"Data collection error: {e}")


def main():
    # VisionPro IP地址
    avp_ip = "192.168.43.20"
    
    # 创建管理器和共享字典
    manager = mp.Manager()
    shared_dict = manager.dict()
    
    # 初始化共享字典
    shared_dict['left'] = None
    shared_dict['right'] = None
    
    # 创建进程
    data_process = mp.Process(target=data_collector_process, args=(avp_ip, shared_dict))
    # 替换原来的进程创建方式
    left_viz_process = mp.Process(target=enhanced_hand_3d_visualizer_with_skeleton, args=(shared_dict, 'left'))
    right_viz_process = mp.Process(target=enhanced_hand_3d_visualizer_with_skeleton, args=(shared_dict, 'right'))
    # # 调整缩放和视角
    # left_viz_process = mp.Process(
    #     target=hand_3d_visualizer, 
    #     args=(shared_dict, 'left'), 
    #     kwargs={'scale': 400, 'offset_x': 50, 'offset_y': -30}
    # )

    # right_viz_process = mp.Process(
    #     target=hand_3d_visualizer, 
    #     args=(shared_dict, 'right'), 
    #     kwargs={'scale': 400, 'offset_x': -50, 'offset_y': -30}
    # )
    # 启动所有进程
    data_process.start()
    left_viz_process.start()
    right_viz_process.start()
    
    print("Processes started:")
    print("- Data collection process")
    print("- Left hand visualization process")
    print("- Right hand visualization process")
    print("Press Ctrl+C to stop all processes")
    print("In visualization windows, press 'q' to close them")
    
    try:
        # 等待所有进程完成
        data_process.join()
        left_viz_process.join()
        right_viz_process.join()
        
    except KeyboardInterrupt:
        print("\nTerminating processes...")
        data_process.terminate()
        left_viz_process.terminate()
        right_viz_process.terminate()
        
        # 等待进程结束
        data_process.join()
        left_viz_process.join()
        right_viz_process.join()
        
        print("All processes terminated.")

if __name__ == "__main__":
    main()