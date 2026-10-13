"""Expose original Vision Pro matrices. Coordinate conversion belongs downstream."""
import time
import numpy as np
from teleoperation.contracts.observations import RawHandFrame


class VisionProInput:
    def __init__(self, ip, side="left", streamer=None):
        if side not in ("left", "right", "both"): raise ValueError(f"Invalid Vision Pro hand side: {side}")
        self.side = side
        if streamer is None:
            from avp_stream import VisionProStreamer
            streamer = VisionProStreamer(ip=ip)
        self._streamer = streamer

    def next_observation(self):
        if self._streamer is None: raise RuntimeError("Vision Pro input is closed")
        latest = self._streamer.get_latest()
        if latest is None: return None
        sides = ("left", "right") if self.side == "both" else (self.side,)
        hands = {"left": None, "right": None}
        for side in sides:
            value = latest.get(side + "_fingers")
            hands[side] = None if value is None else np.asarray(value)
        if all(value is None for value in hands.values()): return None
        return RawHandFrame(hands, time.time(), "visionpro", {"side": self.side},
                            {side: "transforms25" for side in sides})

    def close(self):
        stream, self._streamer = self._streamer, None
        if stream is not None and hasattr(stream, "close"): stream.close()


class VisionProReplayInput:
    """Replay the raw transforms in the existing Vision Pro capture H5 format."""

    def __init__(self, path, side="left"):
        import h5py
        if side not in ("left", "right", "both"):
            raise ValueError(f"Invalid Vision Pro hand side: {side}")
        self.side = side
        self._file = h5py.File(path, "r")
        try:
            if self._file.attrs.get("source") != "visionpro":
                raise ValueError("VR replay requires a source=visionpro recording")
            self.frames = len(self._file["timestamps"])
            self.sides = ("left", "right") if side == "both" else (side,)
            for name in self.sides:
                dataset = self._file[name + "_hand_transforms"]
                if dataset.shape != (self.frames, 25, 4, 4):
                    raise ValueError(f"Invalid raw VR transform shape: {dataset.shape}")
            self.index = 0
        except Exception:
            self._file.close()
            raise

    def next_observation(self):
        if self._file is None:
            raise RuntimeError("Vision Pro replay is closed")
        if self.index >= self.frames:
            raise StopIteration
        index = self.index
        self.index += 1
        hands = {"left": None, "right": None}
        for side in self.sides:
            valid = self._file.get(side + "_valid")
            if valid is None or bool(valid[index]):
                hands[side] = self._file[side + "_hand_transforms"][index]
        return RawHandFrame(hands, float(self._file["timestamps"][index]), "visionpro",
                            {"side": self.side, "frame_id": index},
                            {side: "transforms25" for side in self.sides})

    def close(self):
        handle, self._file = self._file, None
        if handle is not None:
            handle.close()
