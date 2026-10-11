"""Expose original Vision Pro matrices. Coordinate conversion belongs downstream."""
import time
import numpy as np
from teleoperation.contracts.observations import RawHandFrame


class VisionProInput:
    def __init__(self, ip, side="left", streamer=None):
        if side not in ("left", "right"): raise ValueError(f"Invalid Vision Pro hand side: {side}")
        self.side = side
        if streamer is None:
            from avp_stream import VisionProStreamer
            streamer = VisionProStreamer(ip=ip)
        self._streamer = streamer

    def next_observation(self):
        if self._streamer is None: raise RuntimeError("Vision Pro input is closed")
        latest = self._streamer.get_latest()
        key = self.side + "_fingers"
        if key not in latest: return None
        hands = {"left": None, "right": None}
        hands[self.side] = np.asarray(latest[key])
        return RawHandFrame(hands, time.time(), "visionpro", {"side": self.side}, {self.side: "transforms25"})

    def close(self):
        stream, self._streamer = self._streamer, None
        if stream is not None and hasattr(stream, "close"): stream.close()
