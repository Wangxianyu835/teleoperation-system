"""Vision Pro acquisition composed with dedicated TransHandR preprocessing."""
from teleoperation.inputs.visionpro import VisionProInput
from teleoperation.applications.visionpro_processing import TransHandRVisionProPipeline


class VisionProWorkflow:
    def __init__(self, ip, side='left', scale_factor=1.0, streamer=None):
        self.input = VisionProInput(ip, side, streamer)
        self.pipeline = TransHandRVisionProPipeline(scale_factor)

    def next_input(self):
        window = self.pipeline.process_window(self.input.next_observation())
        return None if window is None else window.to_payload()

    def close(self):
        try:
            self.input.close()
        finally:
            self.pipeline.reset()
