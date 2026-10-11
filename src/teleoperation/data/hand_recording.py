# 类 RawHandH5Writer[append/finish/set_metadata…]（封装数据/流程）；链路：应用/训练→文件读写→观测/命令数据
"""Incremental storage of raw MediaPipe two-hand recordings."""

from pathlib import Path

import h5py
import numpy as np
from teleoperation.contracts.coordinates import landmark_source, WORLD_LANDMARK_SPACE


class RawHandH5Writer:
    """Keep every frame, representing an undetected hand by 21 zero points."""

    def __init__(self, path, metadata=None, *, landmark_space="normalized"):
        self.landmark_space = landmark_space
        processing_space = landmark_source(landmark_space)
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.count = 0
        self.hand_counts = {"left": 0, "right": 0}
        self._last_timestamp = None
        # Exclusive creation protects existing recordings.
        self._file = h5py.File(self.path, "x")
        try:
            self._file.attrs.update(metadata or {})
            self._file.attrs.update({
                "source": "mediapipe",
                "source_landmark_space": "mediapipe_normalized",
                "coordinate_frame": "mediapipe_normalized",
                "hand_processing_landmark_space": processing_space,
                "hand_world_landmark_space": WORLD_LANDMARK_SPACE,
                "world_length_unit": "m",
                "hand_world_origin": "hand_geometric_center",
                "timestamp_unit": "unix_seconds",
                "missing_hand_policy": "zeros",
                "recording_status": "recording",
                "recorded_frames": 0,
            })
            for side in self.hand_counts:
                self._file.create_dataset(
                    f"{side}_hand_keypoints", shape=(0, 21, 3),
                    maxshape=(None, 21, 3), dtype="float32",
                    chunks=(64, 21, 3), compression="gzip",
                )
                world = self._file.create_dataset(
                    f"{side}_world_landmarks", shape=(0, 21, 3),
                    maxshape=(None, 21, 3), dtype="float32",
                    chunks=(64, 21, 3), compression="gzip",
                )
                world.attrs.update({"source_landmark_space": WORLD_LANDMARK_SPACE,
                                    "length_unit": "m", "origin": "hand_geometric_center"})
                self._file[f"{side}_hand_keypoints"].attrs["source_landmark_space"] = "mediapipe_normalized"
            for name, dtype in (("frame_ids", "int64"), ("timestamps", "float64")):
                self._file.create_dataset(
                    name, shape=(0,), maxshape=(None,), dtype=dtype, chunks=(64,),
                )
        except BaseException:
            self._file.close()
            raise

    def append(self, hands, timestamp, *, world_hands=None):
        if self.landmark_space == "world" and world_hands is None:
            raise ValueError("World recording requires world_hands; normalized fallback is forbidden")
        timestamp = float(timestamp)
        if not np.isfinite(timestamp):
            raise ValueError("Recording timestamp must be finite")
        if self._last_timestamp is not None and timestamp <= self._last_timestamp:
            raise ValueError("Recording timestamps must increase")
        points, present, world_points = {}, {}, {}
        for side in self.hand_counts:
            value = hands[side]
            present[side] = value is not None
            points[side] = (np.zeros((21, 3), dtype=np.float32) if value is None
                            else np.asarray(value, dtype=np.float32))
            if points[side].shape != (21, 3) or not np.isfinite(points[side]).all():
                raise ValueError(f"{side} hand must contain finite (21, 3) landmarks")
            value = None if world_hands is None else world_hands[side]
            world_points[side] = (np.zeros((21, 3), dtype=np.float32) if value is None
                                  else np.asarray(value, dtype=np.float32))
            if world_points[side].shape != (21, 3):
                raise ValueError(f"{side} world hand must have shape (21, 3)")
            if not np.isfinite(world_points[side]).all():
                world_points[side] = np.zeros((21, 3), dtype=np.float32)
            if self.landmark_space == "world":
                present[side] = bool(np.any(world_points[side]))

        for dataset in self._file.values():
            dataset.resize(self.count + 1, axis=0)
        for side in self.hand_counts:
            self._file[f"{side}_hand_keypoints"][self.count] = points[side]
            self._file[f"{side}_world_landmarks"][self.count] = world_points[side]
        self._file["frame_ids"][self.count] = self.count
        self._file["timestamps"][self.count] = timestamp
        self.count += 1
        self._last_timestamp = timestamp
        for side in self.hand_counts:
            self.hand_counts[side] += int(present[side])
        if self.count % 30 == 0:
            self._file.flush()

    def finish(self, reason):
        self._file.attrs["recording_status"] = reason

    def set_metadata(self, name, value):
        self._file.attrs[name] = value

    def close(self):
        if self._file is None:
            return
        try:
            # Drop any partially written last frame after an exception/interrupt.
            for dataset in self._file.values():
                dataset.resize(self.count, axis=0)
            self._file.attrs["recorded_frames"] = self.count
            for side, count in self.hand_counts.items():
                self._file.attrs[f"{side}_detected_frames"] = count
            self._file.flush()
        finally:
            self._file.close()
            self._file = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        try:
            if exc_type is not None:
                self.finish("interrupted" if exc_type is KeyboardInterrupt else "error")
        finally:
            self.close()
