"""Vision Pro application composition; inputs return only original matrices."""
from teleoperation.inputs.visionpro import VisionProInput
from teleoperation.retargeting.hand.processing import CanonicalHandProcessor
from teleoperation.retargeting.hand.temporal import TemporalBuffer


class VisionProWorkflow:
    def __init__(self, ip, side="left", scale_factor=1.0, streamer=None):
        self.input = VisionProInput(ip, side, streamer)
        self.processor = CanonicalHandProcessor(scale_factor=scale_factor, track_identity=False, discard_untrackable=False)
        self.buffer = TemporalBuffer(reset_on_missing=False)

    def next_input(self):
        raw = self.input.next_observation()
        if raw is None: return None
        frame = self.processor.process_frame(raw)
        window = self.buffer.append(frame)
        return None if window is None else window.to_payload()

    def close(self): self.input.close(); self.buffer.reset(); self.processor.reset()
