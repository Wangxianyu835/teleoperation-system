"""MediaPipe camera adapter for low-cost hand-tracking experiments."""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np

from input_adapters.hand_keypoints import HandWindowBuffer, MEDIAPIPE_APPROX_SOURCE


class MediaPipeCameraAdapter:
    """Read a webcam with MediaPipe and emit canonical retargeting payloads."""

    def __init__(
        self,
        model_asset_path: str | Path = "hand_landmarker.task",
        camera_index: int = 0,
        width: int = 640,
        height: int = 480,
        fps: int = 30,
        scale_factor: float = 1.0,
        min_confidence: float = 0.5,
    ):
        self.model_asset_path = Path(model_asset_path)
        if not self.model_asset_path.exists():
            raise FileNotFoundError(
                f"MediaPipe hand model not found: {self.model_asset_path}"
            )

        import cv2
        import mediapipe as mp

        self._cv2 = cv2
        self._mp = mp
        self._buffer = HandWindowBuffer(scale_factor=scale_factor)
        self._start_time = time.time()

        base_options = mp.tasks.BaseOptions(model_asset_path=str(self.model_asset_path))
        options = mp.tasks.vision.HandLandmarkerOptions(
            base_options=base_options,
            running_mode=mp.tasks.vision.RunningMode.VIDEO,
            num_hands=2,
            min_hand_detection_confidence=min_confidence,
            min_hand_presence_confidence=min_confidence,
            min_tracking_confidence=min_confidence,
        )
        self._landmarker = mp.tasks.vision.HandLandmarker.create_from_options(options)

        self._cap = cv2.VideoCapture(camera_index)
        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self._cap.set(cv2.CAP_PROP_FPS, fps)

    def next_input(self) -> dict | None:
        success, frame = self._cap.read()
        if not success:
            return None

        frame_rgb = self._cv2.cvtColor(frame, self._cv2.COLOR_BGR2RGB)
        mp_image = self._mp.Image(
            image_format=self._mp.ImageFormat.SRGB,
            data=frame_rgb,
        )
        timestamp_ms = int(time.time() * 1000)
        result = self._landmarker.detect_for_video(mp_image, timestamp_ms)

        left_hand = None
        right_hand = None
        if result.hand_landmarks:
            for index, hand_landmarks in enumerate(result.hand_landmarks):
                hand_type = result.handedness[index][0].category_name.lower()
                keypoints = np.asarray(
                    [[landmark.x, landmark.y, landmark.z] for landmark in hand_landmarks],
                    dtype=np.float32,
                )
                if hand_type == "left":
                    left_hand = keypoints
                elif hand_type == "right":
                    right_hand = keypoints

        return self._buffer.update(
            left_hand=left_hand,
            right_hand=right_hand,
            timestamp=round(time.time() - self._start_time, 3),
            source=MEDIAPIPE_APPROX_SOURCE,
            metadata={"camera_index": int(self._cap.get(self._cv2.CAP_PROP_POS_FRAMES))},
        )

    def release(self) -> None:
        if self._cap is not None:
            self._cap.release()
        if self._landmarker is not None:
            self._landmarker.close()

    def __enter__(self) -> "MediaPipeCameraAdapter":
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        self.release()
