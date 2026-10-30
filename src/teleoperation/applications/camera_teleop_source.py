"""One camera capture shared by MediaPipe Hands and the SMPLer-X worker."""
from __future__ import annotations

import time

import numpy as np

from teleoperation.contracts.body import BodyFrameId
from teleoperation.contracts.observations import (
    CameraFrame,
    RawHandFrame,
    RawTeleopObservation,
    UpperBodyObservation,
)
from teleoperation.inputs.body_tracking import BodyTracker


class CameraTeleopSource:
    def __init__(self, camera, hands_processor, smplerx_client=None, *,
                 session_id="camera", arms_enabled=None, tracker=None):
        self.camera = camera
        self.hands_processor = hands_processor
        self.smplerx_client = smplerx_client
        self.session_id = session_id
        self.arms_enabled = bool(smplerx_client is not None) if arms_enabled is None else bool(arms_enabled)
        if self.arms_enabled and smplerx_client is None:
            raise ValueError("arms_enabled requires an SMPLer-X client")
        self.tracker = tracker or BodyTracker()
        self.last_frame: CameraFrame | None = None
        self.last_result = None
        self.last_tracked = None
        self.last_fault = None
        from .body_source_metrics import BodySourceMetrics
        self.metrics = BodySourceMetrics()

    def next_teleop_observation(self) -> RawTeleopObservation:
        frame = self.camera.capture_sample()
        if not isinstance(frame, CameraFrame):
            raise TypeError("camera.capture_sample() must return CameraFrame")
        if np.asarray(frame.rgb).ndim != 3 or np.asarray(frame.rgb).shape[2] != 3:
            raise ValueError("camera frame must contain an RGB image")
        self.last_frame = frame
        hands = self.hands_processor.process_sample(frame)
        if not isinstance(hands, RawHandFrame):
            raise TypeError("hands_processor.process_sample() must return RawHandFrame")
        if not self.arms_enabled:
            body = UpperBodyObservation(
                {"left": None, "right": None}, {"left": False, "right": False},
                frame.timestamp, "smplx_image_normalized", "normalized",
            )
            return RawTeleopObservation(hands, body, None)
        frame_id = BodyFrameId(self.session_id, frame.frame_id, frame.timestamp)
        result = None
        if self.last_fault is None:
            try:
                self.smplerx_client.submit(frame_id, frame.rgb)
                result = self.smplerx_client.poll()
            except (RuntimeError, ValueError, OSError) as error:
                self.last_fault = str(error)
        self.last_result = result
        tracked = self.tracker.update(result, now_s=frame.timestamp)
        self.last_tracked = tracked
        self.metrics.update(tracked.body, frame.timestamp)
        return RawTeleopObservation(hands, tracked.body, tracked.wrists)

    def release(self):
        close_worker = getattr(self.smplerx_client, "close", None)
        try:
            if close_worker is not None:
                close_worker()
        finally:
            close = getattr(self.camera, "release", None)
            if close is not None:
                close()
