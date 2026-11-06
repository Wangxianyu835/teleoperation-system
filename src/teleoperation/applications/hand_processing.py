"""Explicit application composition of acquisition, preprocessing and buffering."""
from teleoperation.contracts.observations import RawHandFrame
from teleoperation.contracts.hand import CanonicalHandFrame
from teleoperation.retargeting.hand.temporal import TemporalBuffer
from teleoperation.contracts.coordinates import PALM_LENGTH_PROFILE
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


class UnifiedHandPipeline:
    """Parallel sensor adapters -> one metric hand contract -> one window rule.

    Sensor adapters only extract topology/axes. Both NN and dex consume the
    resulting dimensionless window without knowing which device produced it.
    TransHandR's original checkpoint preprocessing remains a separate explicit
    compatibility route; it is never relabelled as this contract.
    """

    def __init__(self):
        from teleoperation.inputs.hand_adapters import VisionProMetricAdapter
        self.camera = MediaPipePalmLocalProcessor('world')
        self.visionpro = VisionProMetricAdapter()
        self.buffer = TemporalBuffer()
        self._source = None
        self.current_hands = {s: None for s in SIDES}
        self.invalid_reasons = {s: None for s in SIDES}

    def process_observation(self, raw):
        from teleoperation.contracts.coordinates import unified_metric_metadata
        from teleoperation.retargeting.hand.normalization import validate_metric_hand_frame, normalize_hand_window
        try:
            if not isinstance(raw, RawHandFrame):
                raw = RawHandFrame({s: raw[s] for s in SIDES}, raw.get('timestamp'),
                    'mediapipe_approx', raw.get('metadata', {}),
                    world_hands={s: raw.get('world_' + s) for s in SIDES})
            if raw.timestamp is None or not np.isfinite(raw.timestamp):
                raise ValueError('Hand input requires finite seconds')
            if 'coordinate_alignment' in raw.metadata:
                raise ValueError('Unified pipeline requires raw sensor observations')
            if raw.source not in ('visionpro', 'mediapipe', 'mediapipe_hands', 'mediapipe_approx'):
                raise ValueError('Unsupported hand input source')
            device = 'visionpro' if raw.source == 'visionpro' else 'camera'
            if self._source != device:
                self.reset()
                self._source = device
            adapted = (self.visionpro.adapt(raw) if device == 'visionpro'
                       else self.camera.process_frame(raw))
            canonical = CanonicalHandFrame(adapted.hands, raw.timestamp, raw.source, {
                **unified_metric_metadata(),
                'observation_source': raw.source,
                'sensor_metadata': dict(raw.metadata),
                'sensor_adapter': 'visionpro_anatomical_palm_v1' if device == 'visionpro' else 'mediapipe_world_palm_v1',
                'invalid_reasons': dict(adapted.invalid_reasons),
            })
            canonical = validate_metric_hand_frame(canonical)
            self.current_hands, self.invalid_reasons = canonical.hands, canonical.invalid_reasons
            window = self.buffer.append(canonical)
            if window is not None:
                window = normalize_hand_window(window)
                if all(x is None for x in window.hands.values()):
                    window = None
            return window, canonical
        except (TypeError, ValueError, KeyError):
            self.reset()
            raise

    def process_window(self, raw):
        return self.process_observation(raw)[0]

    def reset(self):
        self.camera.reset()
        self.buffer.reset()
        self._source = None
        self.current_hands = {s: None for s in SIDES}
        self.invalid_reasons = {s: None for s in SIDES}
