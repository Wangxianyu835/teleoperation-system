"""Combined hand capture persistence.

Arm recordings use :mod:`body_recording` and are never embedded in this hand
file.  This keeps the finger data format useful without preserving the old
MediaPipe Pose arm reader.
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Mapping

import h5py
import numpy as np

from .transaction import atomic_output


POINT_DATASETS = (
    ("left_hand_keypoints", (21, 3)),
    ("right_hand_keypoints", (21, 3)),
    ("left_world_landmarks", (21, 3)),
    ("right_world_landmarks", (21, 3)),
)
VECTOR_DATASETS = (
    "frame_ids",
    "timestamps",
    "left_valid",
    "right_valid",
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
    for index, frame in enumerate(rows):
        legacy = [key for key in frame if key.startswith("left_arm") or key.startswith("right_arm")
                  or key in {"pose_visibility", "pose_keypoints", "pose_world_keypoints"}]
        if legacy:
            raise ValueError(f"SMPL-X body recording is separate; legacy arm fields rejected in frame {index}: {legacy}")

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
    first_metadata = rows[0].get("metadata", {})
    if any("mediapipe_pose" in str(value).lower() or "pose_model" in str(value).lower()
           for value in first_metadata.values()):
        raise ValueError("Legacy MediaPipe Pose arm metadata is not accepted")
    attributes = {
        "dataset_format_version": "hand_capture_h5_v2",
        "source_landmark_space": "mediapipe_normalized",
        "hand_processing_landmark_space": "mediapipe_world_meters",
        "hand_landmark_space": "mediapipe_normalized",
        "hand_world_landmark_space": "mediapipe_world_meters",
        "hand_world_origin": "hand_geometric_center",
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
            h5_file.attrs.update(attributes)
    return destination


def read_hand_body_recording(path: str | Path) -> tuple[dict[str, np.ndarray], dict]:
    """Read one combined hand/body capture."""
    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(f"Capture was not found: {source}")
    with h5py.File(source, "r") as h5_file:
        legacy = [name for name in h5_file if name.startswith("left_arm") or name.startswith("right_arm")
                  or name.startswith("pose_")]
        legacy += [key for key, value in h5_file.attrs.items()
                   if "mediapipe_pose" in str(value).lower() or "pose_model" in str(value).lower()]
        if legacy:
            raise ValueError("Legacy MediaPipe Pose arm recording is not accepted")
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
