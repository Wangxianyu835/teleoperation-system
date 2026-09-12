"""摄像头采集模块 - 联想电脑 RGB 摄像头

论文 Section 3.2.1 (Vision-based teleoperation):
  单目 RGB 相机 -> MediaPipe 人体姿态 -> SMPLer-X -> PINK IK -> 机器人关节

本模块负责第一步: RGB 图像采集 + 保存帧供后续分析
重定向部分(MediaPipe/SMPLer-X/IK)由同学负责
"""

import cv2
import numpy as np
import time


class CameraInterface:
    """联想电脑摄像头 -> RGB 帧采集"""

    def __init__(self, camera_id: int = 0):
        self.camera_id = camera_id
        self.cap = None
        
    def start(self):
        """开启摄像头"""
        self.cap = cv2.VideoCapture(self.camera_id)
        if not self.cap.isOpened():
            raise RuntimeError(f"无法打开摄像头 {self.camera_id}")
        w = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        print(f"[CameraInterface] 摄像头已开启 ({w}x{h})")

    def stop(self):
        """关闭摄像头"""
        if self.cap:
            self.cap.release()
        cv2.destroyAllWindows()
        print("[CameraInterface] 摄像头已关闭")

    def capture_frame(self):
        """捕获一帧 RGB 图像
        
        Returns:
            (rgb_frame, bgr_frame) 或 (None, None)
            rgb_frame: RGB 格式 (480, 640, 3), 供同学后续处理
        """
        if not self.cap or not self.cap.isOpened():
            return None, None
        
        ret, bgr = self.cap.read()
        if not ret:
            return None, None
        
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        return rgb, bgr
    
    def show_preview(self, bgr_frame, window_name="Camera"):
        """显示摄像头预览窗口"""
        cv2.imshow(window_name, bgr_frame)
        return cv2.waitKey(1) & 0xFF