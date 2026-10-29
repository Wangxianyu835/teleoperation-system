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
                for suffix, shape, dtype in (("wrist_orientation_xyzw", (4,), "float64"),
                                             ("wrist_orientation_valid", (), "bool")):
                    self._file.create_dataset(f"{side}_{suffix}", shape=(0, *shape),
                                              maxshape=(None, *shape), dtype=dtype)
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
            from teleoperation.inputs.wrist import camera_wrists, CAMERA_REFERENCE, CAMERA_LOCAL
            wrist = camera_wrists(world_points, timestamp)[side]
            self._file[f"{side}_wrist_orientation_xyzw"][self.count] = wrist.orientation_xyzw if wrist.orientation_valid else np.full(4, np.nan)
            self._file[f"{side}_wrist_orientation_valid"][self.count] = wrist.orientation_valid
            self._file.attrs.update(wrist_observation_version=1, wrist_reference_frame=CAMERA_REFERENCE, wrist_local_frame=CAMERA_LOCAL)
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


def iter_dex_observations(path, source):
    """Read declared raw camera/VR recordings; normalize declared units only."""
    from teleoperation.contracts.observations import RawHandFrame
    if source not in ("camera", "visionpro"):
        raise ValueError("Unknown dex recording source")
    with h5py.File(path, "r") as stream:
        expected = "mediapipe" if source == "camera" else "visionpro"
        if stream.attrs.get("source") != expected:
            raise ValueError("Recording source does not match requested dex adapter")
        unit = stream.attrs.get("timestamp_unit")
        factors = {"unix_seconds": 1., "relative_seconds": 1., "seconds": 1., "unix_ms": .001}
        if unit not in factors:
            raise ValueError("Recording must explicitly declare timestamp units")
        timestamps = np.asarray(stream["timestamps"], dtype=float) * factors[unit]
        if not np.isfinite(timestamps).all() or np.any(np.diff(timestamps) <= 0):
            raise ValueError("Recording timestamps must be finite and strictly increasing")
        sides = ("left", "right")
        if source == "camera":
            if stream.attrs.get("world_length_unit") != "m" or stream.attrs.get("hand_world_landmark_space") != "mediapipe_world_meters":
                raise ValueError("Dex camera replay requires declared world metre landmarks")
            datasets = {s: stream[s + "_world_landmarks"] for s in sides}
            shape = (len(timestamps), 21, 3)
            metadata = {"world_landmark_space": "mediapipe_world_meters", "world_length_unit": "m", "timestamp_unit": "seconds"}
        else:
            if stream.attrs.get("wrist_reference_frame", "visionpro_sdk_zup") != "visionpro_sdk_zup":
                raise ValueError("VR wrist recording must use the SDK Z-up reference")
            if stream.attrs.get("length_unit", stream.attrs.get("units")) != "m":
                raise ValueError("Dex VR replay requires metre transforms")
            datasets = {s: stream[s + "_hand_transforms"] for s in sides}
            shape = (len(timestamps), 25, 4, 4)
            metadata = {"timestamp_unit": "seconds"}
        if any(dataset.shape != shape for dataset in datasets.values()):
            raise ValueError("Raw dex recording topology or frame count mismatch")
        for index, timestamp in enumerate(timestamps):
            from teleoperation.inputs.wrist import camera_wrists, vr_wrist
            hands = {}
            for side in sides:
                value = np.asarray(datasets[side][index])
                valid = stream.get(side + "_valid")
                hands[side] = value if (valid is None or bool(valid[index])) and np.any(value) else None
            wrists = (camera_wrists(hands, timestamp) if source == "camera" else {
                s: vr_wrist(stream[s + "_wrist_transform"][index]
                            if s + "_wrist_transform" in stream else None, timestamp,
                            tracked=bool(stream[s + "_wrist_valid"][index])
                            if s + "_wrist_valid" in stream else
                            (bool(stream[s + "_valid"][index]) if s + "_valid" in stream else True)) for s in sides})
            if source == "camera" and "wrist_observation_version" in stream.attrs:
                from teleoperation.inputs.wrist import recorded_camera_wrist
                for side in sides:
                    qkey = side + "_wrist_orientation_xyzw"
                    if qkey in stream:
                        valid = stream.get(side + "_wrist_orientation_valid")
                        wrists[side] = recorded_camera_wrist(
                            stream[qkey][index], timestamp,
                            stream.attrs["wrist_reference_frame"], stream.attrs["wrist_local_frame"],
                            valid is None or bool(valid[index]))
            yield RawHandFrame(hands if source == "visionpro" else {s: None for s in sides},
                timestamp, expected, {**metadata, "frame_id": index},
                {s: "transforms25" for s in sides} if source == "visionpro" else {},
                world_hands=hands if source == "camera" else None, wrists=wrists)


class LegacyNativeResultWriter:
    """Exclusive native result H5; raw solutions and accepted commands stay separate."""

    def __init__(self, path, models, metadata, *, specs):
        import json
        self.stream = h5py.File(path, "x")
        self.models, self.count, self.datasets, self.last_time = models, 0, [], None
        self.specs = specs
        try:
            self.stream.attrs.update({"schema": "dex_native_results_v1", "timestamp_unit": "seconds",
                "angle_unit": "rad", "runtime_mesh_collision_guard": False, "metadata_json": json.dumps(metadata)})
            def dataset(parent, name, tail=(), dtype="float64"):
                result = parent.create_dataset(name, shape=(0, *tail), maxshape=(None, *tail),
                    chunks=(64, *tail), dtype=dtype)
                self.datasets.append(result)
                return result
            dataset(self.stream, "timestamps")
            dataset(self.stream, "is_calibration", dtype="bool")
            self.groups = {}
            for key, model in models.items():
                group = self.stream.create_group("/".join(key))
                group.attrs.update({"joint_names": json.dumps(model.joint_names), "config_hash": model.config_hash,
                    "independent_names": json.dumps(model.independent_names)})
                self.groups[key] = group
                for name in ("raw_angles", "command_angles"):
                    dataset(group, name, (len(model.joint_names),))
                dataset(group, "target_vectors", (len(model.fingers), 3))
                dataset(group, "valid", dtype="bool")
                dataset(group, "solve_ms")
                dataset(group, "error", dtype=h5py.string_dtype("utf-8"))
        except BaseException:
            self.stream.close()
            raise

    def append(self, timestamp, results, commands, targets, is_calibration=False):
        if not np.isfinite(timestamp) or (self.last_time is not None and timestamp <= self.last_time):
            raise ValueError("Native result timestamps must increase")
        if set(results) != set(self.models) or set(commands) != set(self.models):
            raise ValueError("Native frame must cover configured robot sides")
        rows = {}
        for key, model in self.models.items():
            result = results[key]
            spec = self.specs[key]
            if result.joint_names != spec.independent_joint_names or result.spec_hash != spec.spec_hash:
                raise ValueError("Native result config differs from H5 header")
            accepted = commands[key]
            if accepted.key != key or accepted.spec_hash != spec.spec_hash or accepted.joint_names != spec.joint_names:
                raise ValueError("Native command config differs from H5 header")
            command = accepted.positions
            if not np.isfinite(command).all():
                raise ValueError("Native commands must remain finite on missing frames")
            raw = None if not result.valid else model.expand(result.positions)
            rows[key] = {"raw_angles": np.array([raw[n] for n in model.joint_names]) if raw is not None else np.full(len(model.joint_names), np.nan),
                "command_angles": command, "valid": result.valid, "solve_ms": result.solve_ms,
                "error": result.error or "", "target_vectors": targets[key] if targets[key] is not None else np.full((len(model.fingers), 3), np.nan)}
        self._store(timestamp, rows, is_calibration)

    def _store(self, timestamp, rows, is_calibration):
        for dataset in self.datasets:
            dataset.resize(self.count + 1, axis=0)
        self.stream["timestamps"][self.count] = timestamp
        self.stream["is_calibration"][self.count] = is_calibration
        for key, values in rows.items():
            for name, value in values.items():
                self.groups[key][name][self.count] = value
        self.count += 1
        self.last_time = timestamp

    def close(self):
        if self.stream is not None:
            for dataset in self.datasets:
                dataset.resize(self.count, axis=0)
            self.stream.attrs["frames"] = self.count
            self.stream.close()
            self.stream = None

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()


class LegacyComparisonWriter(LegacyNativeResultWriter):
    """Lossy historical mapping arrays, explicitly not replayable commands."""
    def __init__(self, path, models, metadata):
        super().__init__(path, models, metadata, specs={})
        self.stream.attrs["schema"] = "dex_native_comparison_v1"

    def append(self, timestamp, diagnostics, commands, targets, is_calibration=False):
        if not np.isfinite(timestamp) or (self.last_time is not None and timestamp <= self.last_time):
            raise ValueError("Comparison timestamps must increase")
        if set(diagnostics) != set(self.models) or set(commands) != set(self.models):
            raise ValueError("Comparison rows must cover the recorded robot hands")
        rows = {}
        for key, model in self.models.items():
            row = diagnostics[key]
            count = len(model.joint_names)
            raw = np.asarray(row["raw_angles"], dtype=float)
            command = np.array([commands[key][n] for n in model.joint_names], dtype=float)
            if raw.shape != (count,) or not np.isfinite(command).all() or (row["valid"] and not np.isfinite(raw).all()):
                raise ValueError("Malformed legacy comparison arrays")
            rows[key] = {**row, "command_angles":command,
                         "target_vectors": targets[key] if targets[key] is not None else np.full((len(model.fingers),3), np.nan)}
        self._store(timestamp, rows, is_calibration)


def iter_legacy_native_commands(path, models, *, specs):
    import json
    from teleoperation.contracts.hand_control import HandJointCommand
    with h5py.File(path, "r") as stream:
        if stream.attrs.get("schema") != "dex_native_results_v1" or stream.attrs.get("timestamp_unit") != "seconds":
            raise ValueError("Not a native result recording")
        timestamps = np.asarray(stream["timestamps"])
        if not np.isfinite(timestamps).all() or np.any(np.diff(timestamps) <= 0):
            raise ValueError("Invalid native result timebase")
        for key, model in models.items():
            group = stream["/".join(key)]
            if tuple(json.loads(group.attrs["joint_names"])) != model.joint_names or group.attrs["config_hash"] != model.config_hash:
                raise ValueError("Replay model/config does not match exported recording")
            if group["command_angles"].shape != (len(timestamps), len(model.joint_names)):
                raise ValueError("Replay command shape mismatch")
        for index, timestamp in enumerate(timestamps):
            results = {}
            for key, model in models.items():
                values = np.asarray(stream["/".join(key)]["command_angles"][index])
                named = dict(zip(model.joint_names, values))
                expanded = model.expand([named[n] for n in model.independent_names])
                if any(abs(expanded[n] - named[n]) > 1e-6 for n in model.joint_names):
                    raise ValueError("Replay commands violate native coupling")
                spec = specs[key]
                results[key] = HandJointCommand(*key, spec.spec_hash, model.joint_names, values, float(timestamp))
            yield float(timestamp), results
