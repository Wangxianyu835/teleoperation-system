"""Buffer canonical frames only. No topology or coordinate conversion."""
from collections import deque
import numpy as np
from teleoperation.contracts.hand import CanonicalHandFrame, HandWindow


class TemporalBuffer:
    def __init__(self, receptive_field=3, reset_on_missing=True):
        self.receptive_field = int(receptive_field)
        if self.receptive_field < 1: raise ValueError("receptive_field must be positive")
        self.reset_on_missing = reset_on_missing
        self._buffers = {side: deque(maxlen=self.receptive_field) for side in ("left", "right")}

    def append(self, frame: CanonicalHandFrame):
        if not isinstance(frame, CanonicalHandFrame): raise TypeError("TemporalBuffer requires CanonicalHandFrame")
        for side, points in frame.hands.items():
            if points is None:
                if self.reset_on_missing: self.reset_side(side)
            else:
                if np.shape(points) != (25, 3): raise ValueError("Canonical frame must have shape (25,3)")
                self._buffers[side].append(np.asarray(points, dtype=np.float32))
        hands = {side: self._window_or_none(side) for side in ("left", "right")}
        if all(value is None for value in hands.values()): return None
        return HandWindow(hands, frame.timestamp, frame.source, frame.metadata)

    def _window_or_none(self, side):
        buffer = self._buffers[side]
        if len(buffer) != self.receptive_field: return None
        return np.stack(tuple(buffer), axis=0).astype(np.float32, copy=False)

    def reset(self):
        for buffer in self._buffers.values(): buffer.clear()

    def reset_side(self, side):
        if side not in self._buffers: raise ValueError(f"Invalid hand side: {side}")
        self._buffers[side].clear()
