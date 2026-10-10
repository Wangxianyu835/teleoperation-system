"""Explicit application composition of acquisition, preprocessing and buffering."""
from teleoperation.contracts.observations import RawHandFrame
from teleoperation.contracts.hand import CanonicalHandFrame, HandWindow
from teleoperation.retargeting.hand.processing import CanonicalHandProcessor
from teleoperation.retargeting.hand.temporal import TemporalBuffer
from teleoperation.contracts.constants import DEFAULT_MAX_CENTER_DISPLACEMENT, DEFAULT_MAX_SHAPE_RMSE
from teleoperation.inputs.replay import NpyReplayInput
from teleoperation.contracts.coordinates import (
    COORDINATE_ALIGNMENT, PALM_LOCAL_COORDINATE_ALIGNMENT,
    validate_coordinate_alignment,
)
from teleoperation.retargeting.hand.processing import MediaPipePalmLocalProcessor
import numpy as np


SIDES = ("left", "right")


class MediaPipePalmLocalPipeline:
    """Shared hand/dual composition; alignment always precedes buffering."""

    def __init__(self):
        self.processor = MediaPipePalmLocalProcessor()
        self.buffer = TemporalBuffer()

    @property
    def current_hands(self): return self.processor.current_hands
    @property
    def valid_streak(self): return self.processor.valid_streak
    @property
    def invalid_reasons(self): return self.processor.invalid_reasons

    def process_observation(self, frame):
        raw = frame if isinstance(frame, RawHandFrame) else RawHandFrame(
            {side: frame[side] for side in SIDES}, frame.get("timestamp"),
            "mediapipe_approx", frame.get("metadata", {}),
        )
        if "coordinate_alignment" in raw.metadata or raw.metadata.get("coordinate_frame") == "l21":
            self.reset()
            raise ValueError("MediaPipe palm-local preprocessing requires raw input; refusing to align already declared coordinates")
        canonical = self.processor.process_frame(raw)
        return self.buffer.append(canonical), canonical

    def process_window(self, frame):
        return self.process_observation(frame)[0]

    def process_frame(self, frame):
        window = self.process_window(frame)
        return None if window is None else window.to_payload()

    def reset(self):
        self.processor.reset(); self.buffer.reset()


def hand_coordinate_alignment(preprocessing):
    modes = {"legacy": COORDINATE_ALIGNMENT, "palm-local": PALM_LOCAL_COORDINATE_ALIGNMENT}
    if preprocessing not in modes:
        raise ValueError(f"Unknown hand preprocessing mode: {preprocessing!r}")
    return modes[preprocessing]


class DualHandPipeline:
    """Validate input space before choosing the existing hand algorithms.

    Aligned landmarks declare coordinate_frame=l21 and coordinate_alignment.
    Unaligned MediaPipe declares source_landmark_space=mediapipe_normalized;
    Vision Pro declares source=visionpro and transforms25 geometry. Other
    unaligned spaces are rejected. source_landmark_space on aligned data is
    provenance, as in H5, and never requests another alignment.
    """

    def __init__(self, preprocessing="legacy"):
        self.coordinate_alignment = hand_coordinate_alignment(preprocessing)
        self.pipeline = (MediaPipePalmLocalPipeline() if preprocessing == "palm-local"
                         else CanonicalHandPipeline())
        self.buffer = self.pipeline.buffer

    def process_frame(self, frame):
        try:
            return self._process_frame(frame)
        except (TypeError, ValueError, KeyError):
            # A rejected contract must not leave a usable previous window.
            self.reset()
            raise

    def _process_frame(self, frame):
        if not isinstance(frame, RawHandFrame):
            raise TypeError("Expected RawHandFrame")
        metadata = frame.metadata
        if "coordinate_alignment" in metadata:
            alignment = validate_coordinate_alignment(metadata["coordinate_alignment"], "Dual hand input")
            if alignment != self.coordinate_alignment:
                raise ValueError(f"Hand preprocessing coordinate alignment mismatch: input={alignment!r}; preprocessing={self.coordinate_alignment!r}")
            if metadata.get("coordinate_frame") != "l21":
                raise ValueError("Aligned dual hand input must declare coordinate_frame='l21'")
            frame.validate(strict=False)
            if alignment == PALM_LOCAL_COORDINATE_ALIGNMENT:
                # Already aligned: no recentering, topology, tracking or rotation.
                hands = {}
                reasons = {}
                for side in SIDES:
                    points = frame.hands[side]
                    if points is not None and np.shape(points) != (25, 3):
                        raise ValueError("Aligned palm-local input must have shape (25,3)")
                    with np.errstate(over="ignore", invalid="ignore"):
                        points = None if points is None else np.asarray(points, dtype=np.float32)
                    if points is None or not np.isfinite(points).all() or not np.any(points):
                        points = None
                        reasons[side] = "missing or invalid aligned hand"
                    hands[side] = points
                canonical = CanonicalHandFrame(hands, frame.timestamp, frame.source, {
                    **metadata, "identity_tracking": False,
                    "hand_valid": {side: hands[side] is not None for side in SIDES},
                    "invalid_reasons": reasons,
                })
                return self.buffer.append(canonical), canonical
            if any(frame.geometry_encoding.get(side) == "transforms25" for side in SIDES):
                raise ValueError("Aligned input must contain landmarks, not transforms25")
        elif metadata.get("coordinate_frame") == "l21":
            raise ValueError("Aligned dual hand input must declare coordinate_alignment")
        elif self.coordinate_alignment == PALM_LOCAL_COORDINATE_ALIGNMENT:
            if (frame.source != "mediapipe_approx"
                    or metadata.get("source_landmark_space") != "mediapipe_normalized"
                    or any(frame.geometry_encoding.get(side, "landmarks21") not in ("landmarks", "landmarks21") for side in SIDES)):
                raise ValueError("Palm-local preprocessing requires explicit MediaPipe normalized landmarks21")
            return self.pipeline.process_observation(frame)
        elif not (frame.source == "visionpro"
                  and "transforms25" in frame.geometry_encoding.values()
                  and all(frame.hands[side] is None or frame.geometry_encoding.get(side) == "transforms25" for side in SIDES)):
            raise ValueError("Unknown legacy input space: declare aligned l21 coordinates or Vision Pro transforms25")

        # Preserve the legacy canonical processor, including identity tracking.
        window, canonical = self.pipeline.process_frame(frame)
        canonical = CanonicalHandFrame(canonical.hands, canonical.timestamp, canonical.source, {
            **canonical.metadata, "coordinate_frame": "l21",
            "coordinate_alignment": self.coordinate_alignment,
        })
        window = None if window is None else HandWindow(window.hands, window.timestamp, window.source, canonical.metadata)
        return window, canonical

    def reset(self):
        self.pipeline.reset()


class CanonicalHandPipeline:
    def __init__(self, scale_factor=1.0, receptive_field=3, track_identity=True, max_center_displacement=DEFAULT_MAX_CENTER_DISPLACEMENT, max_shape_rmse=DEFAULT_MAX_SHAPE_RMSE):
        self.processor = CanonicalHandProcessor(scale_factor, track_identity, max_center_displacement, max_shape_rmse)
        self.buffer = TemporalBuffer(receptive_field)
        self.track_identity = track_identity

    def process_frame(self, frame):
        canonical = self.processor.process_frame(frame)
        return self.buffer.append(canonical), canonical

    def process(self, left_hand=None, right_hand=None, timestamp=None, source="unknown", metadata=None):
        window, canonical = self.process_frame(RawHandFrame({"left": left_hand, "right": right_hand}, timestamp, source, metadata or {}))
        return (None if window is None else window.to_payload(), {side: value for side, value in canonical.hands.items() if value is not None})

    def update(self, *args, **kwargs): return self.process(*args, **kwargs)[0]

    def process_detections(self, detections, timestamp=None, source="unknown", metadata=None):
        canonical = self.processor.process_detections(detections, timestamp, source, metadata)
        return self.buffer.append(canonical), canonical

    def update_detections(self, *args, **kwargs):
        window, _ = self.process_detections(*args, **kwargs)
        return None if window is None else window.to_payload()

    def reset(self):
        self.processor.reset(); self.buffer.reset()

    def reset_side(self, side): self.buffer.reset_side(side)


class NpyReplayWorkflow:
    def __init__(self, path, scale_factor=1.0):
        self.input = NpyReplayInput(path)
        self.pipeline = CanonicalHandPipeline(scale_factor=scale_factor, track_identity=False)

    def next_input(self):
        while True:
            try: frame = self.input.next_observation()
            except StopIteration: return None
            window, _ = self.pipeline.process_frame(frame)
            if window is not None: return window.to_payload()

    def __iter__(self):
        while True:
            payload = self.next_input()
            if payload is None: break
            yield payload

    def close(self):
        self.input.close(); self.pipeline.reset()
