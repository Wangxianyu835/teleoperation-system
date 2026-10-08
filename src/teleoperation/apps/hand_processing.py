"""Explicit application composition of acquisition, preprocessing and buffering."""
from teleoperation.contracts.observations import RawHandFrame
from teleoperation.contracts.hand import CanonicalHandFrame, HandWindow
from teleoperation.retargeting.hand.processing import CanonicalHandProcessor
from teleoperation.retargeting.hand.temporal import TemporalBuffer
from teleoperation.contracts.constants import DEFAULT_MAX_CENTER_DISPLACEMENT, DEFAULT_MAX_SHAPE_RMSE
from teleoperation.inputs.replay import NpyReplayInput


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
