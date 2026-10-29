"""Expose original Vision Pro matrices. Coordinate conversion belongs downstream."""
import time
import numpy as np
from teleoperation.contracts.observations import RawHandFrame
from teleoperation.inputs.wrist import vr_wrist


class VisionProInput:
    def __init__(self, ip, side="left", streamer=None):
        if side not in ("left", "right", "both"): raise ValueError(f"Invalid Vision Pro hand side: {side}")
        self.side = side
        if streamer is None:
            from avp_stream import VisionProStreamer
            streamer = VisionProStreamer(ip=ip)
        self._streamer = streamer
        if getattr(streamer, "origin", "avp") != "avp":
            raise ValueError("Wrist input requires avp_stream origin='avp'")
        self._last_payload = None
        self._last_payload_at = time.monotonic()
        self._sample_timestamp = None
        self._side_values = {}
        self._side_changed = {}

    def next_observation(self):
        if self._streamer is None: raise RuntimeError("Vision Pro input is closed")
        latest = self._streamer.get_latest()
        if latest is None: return None
        sides = ("left", "right") if self.side == "both" else (self.side,)
        hands = {"left": None, "right": None}
        timestamp = time.time()
        now = time.monotonic()
        payload = getattr(latest, "_raw", latest)
        if payload is not self._last_payload:
            self._last_payload, self._last_payload_at = payload, now
            self._sample_timestamp = timestamp
        fresh = now - self._last_payload_at < .5
        wrists = {}
        for side in sides:
            value = latest.get(side + "_fingers")
            hands[side] = None if value is None else np.asarray(value)
            matrix = latest.get(side + "_wrist")
            signature = (None if matrix is None else np.asarray(matrix).tobytes(),
                         None if value is None else np.asarray(value).tobytes())
            if signature != self._side_values.get(side):
                self._side_values[side] = signature
                self._side_changed[side] = now
            tracked = (fresh and now - self._side_changed.get(side, now) < 1.
                       and bool(latest.get(side + "_is_tracked", True)))
            wrists[side] = vr_wrist(matrix, self._sample_timestamp, tracked=tracked)
        return RawHandFrame(hands, timestamp, "visionpro", {"side": self.side},
                            {side: "transforms25" for side in sides}, wrists=wrists)

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
            if self._file.attrs.get("wrist_reference_frame", "visionpro_sdk_zup") != "visionpro_sdk_zup":
                raise ValueError("VR wrist recording must use the SDK Z-up reference")
            self.frames = len(self._file["timestamps"])
            self.sides = ("left", "right") if side == "both" else (side,)
            for name in self.sides:
                dataset = self._file[name + "_hand_transforms"]
                if dataset.shape != (self.frames, 25, 4, 4):
                    raise ValueError(f"Invalid raw VR transform shape: {dataset.shape}")
                wrist = self._file.get(name + "_wrist_transform")
                if wrist is not None and wrist.shape != (self.frames, 4, 4):
                    raise ValueError(f"Invalid VR wrist shape: {wrist.shape}")
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
        wrists = {}
        timestamp = float(self._file["timestamps"][index])
        for side in self.sides:
            valid = self._file.get(side + "_valid")
            if valid is None or bool(valid[index]):
                hands[side] = self._file[side + "_hand_transforms"][index]
            wrist = self._file.get(side + "_wrist_transform")
            # Legacy captures only have the hand mask, including frozen-tail
            # rejection. Use it conservatively until a separate wrist mask exists.
            wrist_valid = self._file.get(side + "_wrist_valid", valid)
            wrists[side] = vr_wrist(None if wrist is None else wrist[index], timestamp,
                                    tracked=True if wrist_valid is None else bool(wrist_valid[index]))
        return RawHandFrame(hands, timestamp, "visionpro",
                            {"side": self.side, "frame_id": index},
                            {side: "transforms25" for side in self.sides}, wrists=wrists)

    def close(self):
        handle, self._file = self._file, None
        if handle is not None:
            handle.close()
