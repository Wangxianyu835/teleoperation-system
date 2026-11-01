"""Explicit application composition of acquisition, preprocessing and buffering."""
from teleoperation.contracts.observations import RawHandFrame
from teleoperation.contracts.hand import CanonicalHandFrame, HandWindow
from teleoperation.retargeting.hand.temporal import TemporalBuffer
from teleoperation.contracts.constants import DEFAULT_MAX_CENTER_DISPLACEMENT, DEFAULT_MAX_SHAPE_RMSE
from teleoperation.contracts.coordinates import (
    COORDINATE_ALIGNMENT, PALM_LOCAL_COORDINATE_ALIGNMENT,
    validate_coordinate_alignment,
    PALM_LENGTH_PROFILE,
)
from teleoperation.retargeting.hand.processing import MediaPipePalmLocalProcessor
import numpy as np


SIDES = ("left", "right")


class MediaPipePalmLocalPipeline:
    """Shared hand/dual composition; alignment always precedes buffering."""

    def __init__(self, landmark_space="normalized", *, input_profile=None):
        if input_profile not in (None, PALM_LENGTH_PROFILE):
            raise ValueError("Unknown camera hand input profile")
        if input_profile == PALM_LENGTH_PROFILE and landmark_space != "world":
            raise ValueError("Palm-length normalization requires world metre input")
        self.input_profile = input_profile
        self.processor = MediaPipePalmLocalProcessor(landmark_space)
        self.buffer = TemporalBuffer()

    @property
    def current_hands(self): return self.processor.current_hands
    @property
    def valid_streak(self): return self.processor.valid_streak
    @property
    def invalid_reasons(self): return self.processor.invalid_reasons

    def process_observation(self, frame):
        try:
            raw = frame if isinstance(frame, RawHandFrame) else RawHandFrame(
                {side: frame[side] for side in SIDES}, frame.get("timestamp"),
                "mediapipe_approx", frame.get("metadata", {}),
                world_hands=({side: frame.get(f"world_{side}") for side in SIDES}
                             if "world_left" in frame and "world_right" in frame else None),
            )
            if "coordinate_alignment" in raw.metadata or raw.metadata.get("coordinate_frame") == "l21":
                raise ValueError("MediaPipe palm-local preprocessing requires raw input; refusing to align already declared coordinates")
            canonical = self.processor.process_frame(raw)
            if self.input_profile == PALM_LENGTH_PROFILE:
                from teleoperation.retargeting.hand.normalization import validate_metric_hand_frame
                canonical = validate_metric_hand_frame(canonical)
            window = self.buffer.append(canonical)
            if window is not None and self.input_profile == PALM_LENGTH_PROFILE:
                from teleoperation.retargeting.hand.normalization import normalize_hand_window
                ready = {side:window.hands[side] is not None for side in SIDES}
                window = normalize_hand_window(window)
                for side in SIDES:
                    if ready[side] and window.hands[side] is None:
                        self.buffer.reset_side(side)
                if all(points is None for points in window.hands.values()):
                    window = None
            return window, canonical
        except (TypeError, ValueError, KeyError):
            self.reset()
            raise

    def process_window(self, frame):
        return self.process_observation(frame)[0]

    def process_frame(self, frame):
        window = self.process_window(frame)
        return None if window is None else window.to_payload()

    def reset(self):
        self.processor.reset(); self.buffer.reset()
