from __future__ import annotations
from pathlib import Path
import h5py
import numpy as np
from teleoperation.contracts.coordinates import COORDINATE_FRAME, validate_coordinate_alignment
from teleoperation.contracts.constants import HAND_SIDES, HAND_ANGLE_DIM
_SIDE_KEYS = {"left": "left_hand_keypoints", "right": "right_hand_keypoints"}
def _read_coordinate_alignment(source) -> str:
    frame = source.attrs.get("coordinate_frame")
    if isinstance(frame, bytes):
        frame = frame.decode("utf-8")
    if not isinstance(frame, str) or frame != COORDINATE_FRAME:
        raise ValueError(
            "H5 input must be pre-aligned and declare "
            f"coordinate_frame={COORDINATE_FRAME!r}; got {frame!r}. "
            "Run the align_h5 command first."
        )
    return validate_coordinate_alignment(source.attrs.get("coordinate_alignment"), "H5")


def read_coordinate_alignment(path: str | Path) -> str:
    """Read and validate the H5 root's explicit L21 coordinate contract."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"H5 input was not found: {path}")
    with h5py.File(path, "r") as source:
        return _read_coordinate_alignment(source)


def load_twohand_h5(
    path: str | Path,
    require_aligned: bool = True,
) -> tuple[np.ndarray, np.ndarray, dict[str, np.ndarray]]:
    """Load and validate the supported two-hand H5 layouts."""
    h5_path = Path(path)
    if not h5_path.is_file():
        raise FileNotFoundError(f"H5 input was not found: {h5_path}")

    with h5py.File(h5_path, "r") as h5_file:
        if require_aligned:
            _read_coordinate_alignment(h5_file)
        if all(key in h5_file for key in _SIDE_KEYS.values()):
            left = np.asarray(h5_file[_SIDE_KEYS["left"]][:])
            right = np.asarray(h5_file[_SIDE_KEYS["right"]][:])
            frame_ids = _read_optional_vector(
                h5_file,
                "frame_ids",
                left.shape[0],
                dtype=np.int64,
            )
            timestamps = _read_optional_vector(
                h5_file,
                "timestamps",
                left.shape[0],
                dtype=np.float64,
            )
        else:
            groups = [
                key
                for key in h5_file.keys()
                if isinstance(h5_file[key], h5py.Group)
                and "l_glove_pos" in h5_file[key]
                and "r_glove_pos" in h5_file[key]
            ]
            if len(groups) != 1:
                raise ValueError(
                    "Expected root datasets "
                    "'left_hand_keypoints'/'right_hand_keypoints' or "
                    "exactly one group with 'l_glove_pos'/'r_glove_pos'"
                )
            group = h5_file[groups[0]]
            left = np.asarray(group["l_glove_pos"][:])
            right = np.asarray(group["r_glove_pos"][:])
            frame_ids = _read_optional_vector(
                group,
                "frame_ids",
                left.shape[0],
                dtype=np.int64,
            )
            timestamps = _read_optional_vector(
                group,
                "timestamps",
                left.shape[0],
                dtype=np.float64,
            )

    _validate_hand_array(left, "left")
    _validate_hand_array(right, "right")
    if left.shape[0] != right.shape[0]:
        raise ValueError(
            "Left/right frame counts do not match: "
            f"{left.shape[0]} != {right.shape[0]}"
        )
    if frame_ids.shape[0] != left.shape[0]:
        raise ValueError("frame_ids length does not match hand data")
    if timestamps.shape[0] != left.shape[0]:
        raise ValueError("timestamps length does not match hand data")
    if not np.isfinite(frame_ids).all() or not np.isfinite(timestamps).all():
        raise ValueError("frame_ids and timestamps must be finite")

    return (
        frame_ids,
        timestamps,
        {"left": left.astype(np.float32, copy=False), "right": right.astype(np.float32, copy=False)},
    )


def _read_optional_vector(
    container,
    name: str,
    length: int,
    dtype,
) -> np.ndarray:
    if name not in container:
        return np.arange(length, dtype=dtype)
    values = np.asarray(container[name][:], dtype=dtype).reshape(-1)
    if values.shape[0] != length:
        raise ValueError(f"{name} length does not match hand data")
    return values


def _validate_hand_array(points: np.ndarray, side: str) -> None:
    if points.ndim != 3 or points.shape[2] != 3:
        raise ValueError(
            f"{side} hand data must have shape (frames, joints, 3), "
            f"got {points.shape}"
        )
    if points.shape[1] not in (21, 25):
        raise ValueError(
            f"{side} hand data must contain 21 or 25 points, "
            f"got {points.shape[1]}"
        )
    if not np.issubdtype(points.dtype, np.number):
        raise TypeError(f"{side} hand data must be numeric")

from teleoperation.contracts.hand import AngleFrame
from collections.abc import Iterator
def iter_angle_h5(path: str | Path) -> Iterator[AngleFrame]:
    with h5py.File(Path(path), "r") as h5_file:
        required = ("frame_ids", "timestamps") + tuple(
            f"{side}_{suffix}"
            for side in HAND_SIDES
            for suffix in ("angles", "valid")
        )
        missing = [name for name in required if name not in h5_file]
        if missing:
            raise ValueError(f"Angle H5 is missing datasets: {', '.join(missing)}")

        frame_ids = np.asarray(h5_file["frame_ids"][:]).reshape(-1)
        timestamps = np.asarray(h5_file["timestamps"][:], dtype=np.float64).reshape(-1)
        angles = {
            side: np.asarray(h5_file[f"{side}_angles"][:], dtype=np.float32)
            for side in HAND_SIDES
        }
        valid = {
            side: np.asarray(h5_file[f"{side}_valid"][:], dtype=bool).reshape(-1)
            for side in HAND_SIDES
        }

    frame_count = frame_ids.shape[0]
    for side in HAND_SIDES:
        if angles[side].shape != (frame_count, HAND_ANGLE_DIM):
            raise ValueError(f"{side}_angles must have shape ({frame_count}, 18)")
        if valid[side].shape != (frame_count,):
            raise ValueError(f"{side}_valid must have shape ({frame_count},)")
        if not np.isfinite(angles[side]).all():
            raise ValueError(f"{side}_angles contains NaN or Inf")
    if timestamps.shape != (frame_count,):
        raise ValueError("timestamps length does not match frame_ids")

    for index in range(frame_count):
        yield AngleFrame(int(frame_ids[index]), float(timestamps[index]), angles["left"][index].copy(), angles["right"][index].copy(), bool(valid["left"][index]), bool(valid["right"][index]))


def _validate_coordinate_metadata(source) -> None:
    values = {}
    for name in ("coordinate_frame", "coordinate_alignment"):
        value = source.attrs.get(name)
        if value is not None and not isinstance(value, (str, bytes)):
            raise ValueError(f"{name} must be a scalar string")
        values[name] = value.decode("utf-8") if isinstance(value, bytes) else value
    if values["coordinate_frame"] == COORDINATE_FRAME:
        raise ValueError(f"Input is already marked as {COORDINATE_FRAME!r}")
    if values["coordinate_alignment"] is not None:
        raise ValueError(
            "Input already declares coordinate_alignment; refusing to apply "
            "alignment twice or interpret an unknown alignment"
        )
    _validate_source_landmark_space(source)


def _validate_source_landmark_space(container) -> None:
    value = container.attrs.get("source_landmark_space")
    if isinstance(value, bytes):
        value = value.decode("utf-8")
    if value is not None and (not isinstance(value, str) or value != "mediapipe_normalized"):
        raise ValueError(
            "source_landmark_space must be 'mediapipe_normalized' or undeclared; "
            f"got {value!r}"
        )


def read_alignment_input(path):
    with h5py.File(path, "r") as source:
        _validate_coordinate_metadata(source)
        frame_ids, _, raw = load_twohand_h5(path, require_aligned=False)
        if frame_ids.size == 0: raise ValueError("Input H5 contains no frames")
        if "left_hand_keypoints" not in source:
            group = next(source[key] for key in source if isinstance(source[key], h5py.Group) and "l_glove_pos" in source[key] and "r_glove_pos" in source[key])
            _validate_source_landmark_space(group)
        return frame_ids, raw


def write_aligned_copy(input_path, temporary_path, aligned, metadata):
    with h5py.File(input_path, "r") as source, h5py.File(temporary_path, "w") as target:
        for key in source: source.copy(key, target)
        target.attrs.update(source.attrs)
        if "left_hand_keypoints" in source and "right_hand_keypoints" in source:
            container = target
            names = ("left_hand_keypoints", "right_hand_keypoints")
        else:
            name = next(key for key in source if isinstance(source[key], h5py.Group) and "l_glove_pos" in source[key] and "r_glove_pos" in source[key])
            container = target[name]
            names = ("l_glove_pos", "r_glove_pos")
        for side, name in zip(("left", "right"), names):
            attributes = dict(container[name].attrs)
            del container[name]
            dataset = container.create_dataset(name, data=aligned[side], compression="gzip")
            dataset.attrs.update(attributes)
        target.attrs.update(metadata)
        if "source_file" not in target.attrs: target.attrs["source_file"] = str(input_path)
        target.flush()


def read_angle_arrays(path):
    path = Path(path)
    if not path.is_file(): raise FileNotFoundError(f"Angle H5 was not found: {path}")
    required = ("frame_ids", "timestamps") + tuple(f"{side}_{suffix}" for side in HAND_SIDES for suffix in ("angles", "valid"))
    with h5py.File(path, "r") as file:
        missing = [name for name in required if name not in file]
        if missing: raise ValueError(f"Angle H5 is missing datasets: {', '.join(missing)}")
        return {name: np.asarray(file[name][:]) for name in required}, dict(file.attrs)


def read_native_angles(path):
    with h5py.File(path, "r") as file:
        data = {"left_angles": np.asarray(file["left_angles"][:], dtype=np.float64), "right_angles": np.asarray(file["right_angles"][:], dtype=np.float64), "timestamps": np.asarray(file["timestamps"][:], dtype=np.float64)}
        for name in ("left_valid", "right_valid"):
            data[name] = np.asarray(file[name][:]).astype(bool) if name in file else np.ones(len(data["timestamps"]), bool)
        return data


def write_hand_angles(path, frame_ids, timestamps, angles, valid, metadata):
    with h5py.File(path, "w") as file:
        file.create_dataset("frame_ids", data=frame_ids)
        file.create_dataset("timestamps", data=timestamps)
        for side in HAND_SIDES:
            file.create_dataset(f"{side}_angles", data=angles[side])
            file.create_dataset(f"{side}_valid", data=valid[side])
        file.attrs.update(metadata)


def read_attributes(path):
    with h5py.File(path, "r") as handle: return dict(handle.attrs)


def read_pipeline_angles(path):
    """Diagnostic schema: timestamps may be absent, as in the original tool."""
    with h5py.File(path, "r") as handle:
        left = np.asarray(handle["left_angles"][:], dtype=np.float64)
        right = np.asarray(handle["right_angles"][:], dtype=np.float64)
        left_valid = np.asarray(handle["left_valid"][:]).astype(bool) if "left_valid" in handle else np.ones(len(left), bool)
        right_valid = np.asarray(handle["right_valid"][:]).astype(bool) if "right_valid" in handle else np.ones(len(right), bool)
        timestamps = np.asarray(handle["timestamps"][:], dtype=np.float64) if "timestamps" in handle else np.arange(len(left)) / 30.0
        return list(handle.keys()), left, right, left_valid, right_valid, timestamps, dict(handle.attrs)


def read_required_right_angles(path):
    with h5py.File(path, "r") as handle:
        return np.asarray(handle["right_angles"][:], dtype=np.float64), np.asarray(handle["right_valid"][:]).astype(bool)
