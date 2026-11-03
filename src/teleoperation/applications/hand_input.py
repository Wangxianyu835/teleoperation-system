"""Vision Pro acquisition into the same contract used by RGB hand control."""
import time

from teleoperation.applications.hand_processing import UnifiedHandPipeline
from teleoperation.contracts.observations import RawHandFrame


class VisionProHandWorkflow:
    def __init__(self, *, input_h5=None, ip=None, hand='both'):
        from teleoperation.inputs.visionpro import VisionProInput, VisionProReplayInput
        self.pipeline = UnifiedHandPipeline()
        self.source = (VisionProReplayInput(input_h5, hand) if input_h5 is not None
                       else VisionProInput(ip, hand))
        self.last_observation = None
        self.timestamp = None

    def next_window(self):
        raw = self.source.next_observation()
        if raw is None:
            raw = RawHandFrame({'left': None, 'right': None}, time.time(), 'visionpro')
        stamp = float(raw.timestamp)
        if self.timestamp is not None and stamp <= self.timestamp:
            self.pipeline.reset()
            raise ValueError('Vision Pro timestamps must strictly increase')
        self.timestamp = stamp
        self.last_observation = raw
        return self.pipeline.process_window(raw)

    def close(self):
        self.pipeline.reset()
        self.source.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()
