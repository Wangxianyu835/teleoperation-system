"""Replay recorded SMPL-X observations with their original sample timeline."""
from __future__ import annotations
from teleoperation.data.body_recording import load_body_recording, replay_observation


class BodyReplaySource:
    def __init__(self, path):
        self.recording = load_body_recording(path)
        self.index = 0
        self.frames = len(self.recording.samples)
        self.exhausted = False
        self.last_observation = None
        self.last_time_s = None
        self.frame_dt = None
        from .body_source_metrics import BodySourceMetrics
        self.metrics = BodySourceMetrics()

    def next_teleop_observation(self):
        if self.index >= self.frames:
            self.exhausted = True
            raise StopIteration
        row = self.recording.samples[self.index]
        self.index += 1
        stamp = float(row['capture_time_s'])
        if self.last_time_s is not None:
            if stamp <= self.last_time_s:
                raise ValueError('Body replay camera capture times must increase')
            self.frame_dt = stamp - self.last_time_s
        self.last_time_s = stamp
        self.last_observation = replay_observation(row)
        self.metrics.update(self.last_observation.upper_body, stamp)
        return self.last_observation

    def release(self):
        return None
