"""
MediaPipe 摄像头适配器，用于低成本手部跟踪实验。

该模块使用 MediaPipe 的手部关键点检测模型，从摄像头读取视频流，
逐帧检测左右手的关键点，并交给共享 Hand Core 处理 canonicalization、
三帧窗口和标准输入载荷（payload）。
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np

# 导入手部关键点缓冲区以及数据来源常量（MEDIAPIPE_APPROX_SOURCE）
from retargeting.hand_core import CanonicalHandProcessor
from retargeting.tracking import (
    DEFAULT_MAX_CENTER_DISPLACEMENT,
    DEFAULT_MAX_SHAPE_RMSE,
    MEDIAPIPE_APPROX_SOURCE,
)


class MediaPipeCameraAdapter:
    """
    从网络摄像头读取数据，使用 MediaPipe 提取手部关键点，
    并输出规范化的重定向输入载荷（供后续模型使用）。
    """

    def __init__(
        self,
        model_asset_path: str | Path = "hand_landmarker.task",
        camera_index: int = 0,
        width: int = 640,
        height: int = 480,
        fps: int = 30,
        scale_factor: float = 1.0,
        min_confidence: float = 0.5,
        max_center_displacement: float = DEFAULT_MAX_CENTER_DISPLACEMENT,
        max_shape_rmse: float = DEFAULT_MAX_SHAPE_RMSE,
    ):
        """
        初始化摄像头适配器。

        参数:
            model_asset_path: MediaPipe 手部关键点模型文件路径（.task）
            camera_index: 摄像头设备索引（默认 0）
            width: 期望的视频帧宽度
            height: 期望的视频帧高度
            fps: 期望的帧率
            scale_factor: 手部关键点的缩放系数（用于归一化）
            min_confidence: 检测和跟踪的最小置信度阈值
        """
        # 验证模型文件是否存在
        self.model_asset_path = Path(model_asset_path)
        if not self.model_asset_path.exists():
            raise FileNotFoundError(
                f"MediaPipe hand model not found: {self.model_asset_path}"
            )

        # 延迟导入 OpenCV 和 MediaPipe，避免不必要的依赖加载
        import cv2
        import mediapipe as mp

        self._cv2 = cv2
        self._mp = mp

        # 初始化手部数据缓冲区（用于累积多帧，构建时序窗口）
        self._hand_core = CanonicalHandProcessor(
            scale_factor=scale_factor,
            max_center_displacement=max_center_displacement,
            max_shape_rmse=max_shape_rmse,
        )

        # 记录启动时间，用于生成相对时间戳
        self._start_time = time.time()

        # --- 配置 MediaPipe HandLandmarker ---
        base_options = mp.tasks.BaseOptions(model_asset_path=str(self.model_asset_path))
        options = mp.tasks.vision.HandLandmarkerOptions(
            base_options=base_options,
            running_mode=mp.tasks.vision.RunningMode.VIDEO,  # 视频模式（逐帧处理）
            num_hands=2,                                     # 最多检测双手
            min_hand_detection_confidence=min_confidence,
            min_hand_presence_confidence=min_confidence,
            min_tracking_confidence=min_confidence,
        )
        self._landmarker = mp.tasks.vision.HandLandmarker.create_from_options(options)

        # --- 初始化摄像头 ---
        self._cap = cv2.VideoCapture(camera_index)
        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self._cap.set(cv2.CAP_PROP_FPS, fps)

    def next_input(self) -> dict | None:
        """
        从摄像头读取下一帧，检测手部关键点，更新缓冲区，
        如果缓冲区积累了足够帧数，则返回一个重定向输入载荷字典。

        返回:
            dict: 包含左右手窗口、时间戳、来源和元数据的载荷；
                  如果读取帧失败或缓冲区未就绪，返回 None。
        """
        # 1. 读取一帧图像
        success, frame = self._cap.read()
        if not success:
            return None

        # 2. 转换为 RGB 颜色空间（MediaPipe 要求）
        frame_rgb = self._cv2.cvtColor(frame, self._cv2.COLOR_BGR2RGB)

        # 3. 包装为 MediaPipe Image 对象
        mp_image = self._mp.Image(
            image_format=self._mp.ImageFormat.SRGB,
            data=frame_rgb,
        )

        # 4. 调用 MediaPipe 手部检测（视频模式下需要提供时间戳毫秒值）
        timestamp_ms = int(time.time() * 1000)
        result = self._landmarker.detect_for_video(mp_image, timestamp_ms)

        # 5. 解析检测结果，提取左右手的关键点（21 个 x,y,z 坐标）
        detections = []
        if result.hand_landmarks:
            for index, hand_landmarks in enumerate(result.hand_landmarks):
                # 获取该手是左手还是右手
                hand_type = result.handedness[index][0].category_name.lower()
                # 提取关键点坐标（归一化的 x, y, z）
                keypoints = np.asarray(
                    [[landmark.x, landmark.y, landmark.z] for landmark in hand_landmarks],
                    dtype=np.float32,
                )
                if hand_type in ("left", "right"):
                    detections.append((hand_type, keypoints))

        # 6. Shared hand core performs identity continuity, canonicalization,
        # wrist-relative normalization, and three-frame windowing.
        return self._hand_core.update_detections(
            detections=detections,
            timestamp=round(time.time() - self._start_time, 3),  # 相对时间（秒）
            source=MEDIAPIPE_APPROX_SOURCE,                     # 标识数据来源
            metadata={"camera_index": int(self._cap.get(self._cv2.CAP_PROP_POS_FRAMES))},  # 帧索引
        )

    def release(self) -> None:
        """
        释放摄像头资源和 MediaPipe landmarker 资源。
        应在适配器不再使用时调用。
        """
        if self._cap is not None:
            self._cap.release()
        if self._landmarker is not None:
            self._landmarker.close()

    # --- 上下文管理器支持（with 语句） ---
    def __enter__(self) -> "MediaPipeCameraAdapter":
        """进入上下文时返回自身实例。"""
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        """退出上下文时自动释放资源。"""
        self.release()
