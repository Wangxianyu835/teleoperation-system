"""Combined normalized and metric hand/arm capture persistence."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Mapping

import h5py
import numpy as np

from .transaction import atomic_output


FORMAT_VERSION = "hand_body_capture_h5_v1"
POINT_DATASETS = (
    ("left_hand_keypoints", (21, 3)),
    ("right_hand_keypoints", (21, 3)),
    ("left_world_landmarks", (21, 3)),
    ("right_world_landmarks", (21, 3)),
    ("left_arm_keypoints", (3, 3)),
    ("right_arm_keypoints", (3, 3)),
    ("left_arm_world_keypoints", (3, 3)),
    ("right_arm_world_keypoints", (3, 3)),
)
VECTOR_DATASETS = (
    "frame_ids",
    "timestamps",
    "left_valid",
    "right_valid",
    "left_arm_valid",
    "right_arm_valid",
    "pose_visibility",
)
DATASETS = (*[name for name, _ in POINT_DATASETS], *VECTOR_DATASETS)


def _stack_points(frames: list[Mapping], key: str, shape: tuple[int, int]) -> tuple[np.ndarray, np.ndarray]:
    values = np.full((len(frames), *shape), np.nan, dtype=np.float32)
    valid = np.zeros(len(frames), dtype=bool)
    for index, frame in enumerate(frames):
        value = frame.get(key)
        if value is None:
            continue
        array = np.asarray(value, dtype=np.float32)
        if array.shape != shape:
            raise ValueError(f"{key} frame {index} must have shape {shape}")
        values[index] = array
        valid[index] = True
    return values, valid


def _stack_vector(frames: list[Mapping], key: str, width: int, default) -> tuple[np.ndarray, np.ndarray]:
    values = np.full((len(frames), width), default, dtype=np.float32)
    present = np.zeros(len(frames), dtype=bool)
    for index, frame in enumerate(frames):
        value = frame.get(key)
        if value is None:
            continue
        array = np.asarray(value, dtype=np.float32).reshape(-1)
        if array.shape != (width,):
            raise ValueError(f"{key} frame {index} must have shape ({width},)")
        if not np.isfinite(array).all():
            raise ValueError(f"{key} frame {index} must be finite")
        values[index] = array
        present[index] = True
    return values, present


def write_hand_body_recording(
    path: str | Path,
    frames: Iterable[Mapping],
    *,
    metadata: Mapping | None = None,
) -> Path:
    """Write one capture containing both project and metric coordinate spaces."""
    rows = list(frames)
    if not rows:
        raise ValueError("Capture requires at least one frame")

    frame_ids = np.asarray(
        [frame.get("frame_id", index) for index, frame in enumerate(rows)],
        dtype=np.int64,
    )
    timestamps = np.asarray([frame["timestamp"] for frame in rows], dtype=np.float64)
    if not np.isfinite(timestamps).all():
        raise ValueError("Capture timestamps must be finite")

    point_data = {}
    presence = {}
    for name, shape in POINT_DATASETS:
        point_data[name], presence[name] = _stack_points(rows, name, shape)
    hand_finite = {
        name: np.isfinite(point_data[name]).all(axis=(1, 2))
        for name in ("left_hand_keypoints", "right_hand_keypoints",
                     "left_world_landmarks", "right_world_landmarks")
    }
    left_valid = (
        presence["left_hand_keypoints"]
        & presence["left_world_landmarks"]
        & hand_finite["left_hand_keypoints"]
        & hand_finite["left_world_landmarks"]
    )
    right_valid = (
        presence["right_hand_keypoints"]
        & presence["right_world_landmarks"]
        & hand_finite["right_hand_keypoints"]
        & hand_finite["right_world_landmarks"]
    )
    left_arm_valid, left_arm_present = _stack_vector(rows, "left_arm_valid", 1, False)
    right_arm_valid, right_arm_present = _stack_vector(rows, "right_arm_valid", 1, False)
    visibility, _ = _stack_vector(rows, "pose_visibility", 6, 0.0)
    left_arm_finite = (
        np.isfinite(point_data["left_arm_keypoints"]).all(axis=(1, 2))
        & np.isfinite(point_data["left_arm_world_keypoints"]).all(axis=(1, 2))
    )
    right_arm_finite = (
        np.isfinite(point_data["right_arm_keypoints"]).all(axis=(1, 2))
        & np.isfinite(point_data["right_arm_world_keypoints"]).all(axis=(1, 2))
    )

    first_metadata = rows[0].get("metadata", {})
    attributes = {
        "dataset_format_version": FORMAT_VERSION,
        "source_landmark_space": "mediapipe_normalized",
        "hand_landmark_space": "mediapipe_normalized",
        "hand_world_landmark_space": "mediapipe_world_meters",
        "hand_world_origin": "hand_geometric_center",
        "arm_source_space": "mediapipe_pose_normalized",
        "arm_world_space": "mediapipe_pose_world_meters",
        "arm_world_origin": "hip_midpoint",
        "length_unit": "m",
        "frame_count": len(rows),
        "image_width": int(first_metadata.get("image_width", 0)),
        "image_height": int(first_metadata.get("image_height", 0)),
        "created_utc": datetime.now(timezone.utc).isoformat(),
        **({} if metadata is None else {key: str(value) for key, value in metadata.items()}),
    }

    destination = Path(path)
    with atomic_output(destination) as temporary:
        with h5py.File(temporary, "w") as h5_file:
            h5_file.create_dataset("frame_ids", data=frame_ids)
            h5_file.create_dataset("timestamps", data=timestamps)
            for name, values in point_data.items():
                h5_file.create_dataset(name, data=values)
            h5_file.create_dataset("left_valid", data=left_valid)
            h5_file.create_dataset("right_valid", data=right_valid)
            h5_file.create_dataset(
                "left_arm_valid",
                data=(
                    left_arm_valid[:, 0].astype(bool)
                    & left_arm_present
                    & left_arm_finite
                ),
            )
            h5_file.create_dataset(
                "right_arm_valid",
                data=(
                    right_arm_valid[:, 0].astype(bool)
                    & right_arm_present
                    & right_arm_finite
                ),
            )
            h5_file.create_dataset("pose_visibility", data=visibility)
            h5_file.attrs.update(attributes)
    return destination


def read_hand_body_recording(path: str | Path) -> tuple[dict[str, np.ndarray], dict]:
    """Read one combined hand/body capture."""
    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(f"Capture was not found: {source}")
    with h5py.File(source, "r") as h5_file:
        missing = [name for name in DATASETS if name not in h5_file]
        if missing:
            raise ValueError(f"Capture is missing datasets: {', '.join(missing)}")
        arrays = {name: np.asarray(h5_file[name][:]) for name in DATASETS}
        attributes = dict(h5_file.attrs)
    count = arrays["timestamps"].shape[0]
    for name, shape in POINT_DATASETS:
        if arrays[name].shape != (count, *shape):
            raise ValueError(f"Dataset {name} must have shape ({count}, {shape[0]}, {shape[1]})")
    for name in VECTOR_DATASETS:
        if arrays[name].shape[0] != count:
            raise ValueError(f"Dataset {name} does not match timestamps length")
    return arrays, attributes
