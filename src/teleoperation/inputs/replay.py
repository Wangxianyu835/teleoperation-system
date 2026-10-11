"""Read every recorded frame without normalization or buffering."""
import numpy as np
from pathlib import Path
from teleoperation.contracts.observations import RawHandFrame
from teleoperation.data.npy import read_recording, read_frame


class NpyReplayInput:
    def __init__(self, path):
        self.path = Path(path)
        self._frames = read_recording(path)
        self._index = 0

    def next_observation(self):
        if self._frames is None or self._index >= len(self._frames): raise StopIteration
        frame = read_frame(self._frames[self._index])
        hands = {side: frame.get(side + "_hand") for side in ("left", "right")}
        source = "mediapipe_approx" if any(points is not None and np.shape(points) == (21, 3) for points in hands.values()) else "npy_replay"
        metadata = {"path": str(self.path), "frame_id": frame.get("frame_id", self._index), "adapter": "npy_replay"}
        self._index += 1
        return RawHandFrame(hands, frame.get("timestamp"), source, metadata)

    def __iter__(self): return self
    def __next__(self): return self.next_observation()
    def close(self): self._frames = None
