"""Raw camera acquisition composed with explicit palm-local window processing."""

from __future__ import annotations

import argparse
from pathlib import Path
import time

import numpy as np

from teleoperation.contracts.coordinates import palm_local_metadata
from teleoperation.contracts.hand_control import valid_joint_result
from teleoperation.inputs.mediapipe import MediaPipeCameraInput
from teleoperation.contracts.constants import DEFAULT_MAX_CENTER_DISPLACEMENT, DEFAULT_MAX_SHAPE_RMSE





SIDES = ("left", "right")


class MediaPipeCameraWorkflow:
    """Compose raw camera capture with palm-local realtime processing.

    next_input() retains the optional-window payload and relative seconds
    timestamp. Capture errors propagate after clearing history.
    Scale and confidence retain the raw-input defaults.
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
        landmark_space: str = "world",
        input_profile: str | None = None,
    ):
        if scale_factor != 1.0:
            raise ValueError("MediaPipe palm-local realtime requires scale_factor=1.0")
        if min_confidence != 0.5:
            raise ValueError("MediaPipeCameraInput uses fixed min_confidence=0.5")
        self.model_asset_path = Path(model_asset_path)
        self.landmark_space = landmark_space
        self.input_profile = input_profile
        from teleoperation.contracts.coordinates import UNIFIED_HAND_PROFILE
        from teleoperation.applications.hand_processing import UnifiedHandPipeline
        self.pipeline = (UnifiedHandPipeline() if input_profile == UNIFIED_HAND_PROFILE
                         else MediaPipePalmLocalPipeline(landmark_space, input_profile=input_profile))
        self.processor = (self.pipeline.camera if input_profile == UNIFIED_HAND_PROFILE
                          else self.pipeline.processor)
        self.last_raw_frame = None
        self._input = MediaPipeCameraInput(
            model_asset_path=model_asset_path, camera_index=camera_index,
            width=width, height=height, fps=fps,
            invalid_hand_as_missing=True,
        )
        self._start_time = time.time()

    def next_window(self):
        try:
            raw = (self._input.next_observation(include_world=True) if self.landmark_space == "world"
                   else self._input.next_observation())
            raw_frame = {**raw.hands, "timestamp": raw.timestamp, "metadata": dict(raw.metadata)}
            self.last_raw_frame = raw_frame
            frame = {
                **raw_frame,
                "timestamp": round(time.time() - self._start_time, 3),
                "metadata": {
                    **raw_frame.get("metadata", {}),
                    "raw_timestamp_ms": raw_frame["timestamp"],
                    "timestamp_unit": "relative_seconds",
                },
            }
            if raw.world_hands is not None:
                frame.update({"world_" + side: raw.world_hands[side] for side in SIDES})
            from teleoperation.contracts.coordinates import UNIFIED_HAND_PROFILE
            if self.input_profile == UNIFIED_HAND_PROFILE:
                from dataclasses import replace
                return self.pipeline.process_window(replace(
                    raw, timestamp=frame['timestamp'], metadata=frame['metadata']))
            return self.pipeline.process_window(frame)
        except BaseException:
            self.last_raw_frame = None
            self.release()
            raise

    def next_input(self) -> dict | None:
        window = self.next_window()
        return None if window is None else window.to_payload()

    def release(self) -> None:
        self.pipeline.reset()
        self._input.close()

    def __enter__(self) -> "MediaPipeCameraWorkflow":
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        self.release()


from teleoperation.applications.hand_processing import MediaPipePalmLocalPipeline
