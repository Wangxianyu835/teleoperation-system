"""Persistence for MediaPipe metre-based world landmark captures."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Mapping

import numpy as np
import h5py

from .transaction import atomic_output


FORMAT_VERSION = "world_landmarks_h5_v1"
DATASETS = (
    "frame_ids",
    "timestamps",
    "left_world_landmarks",
    "right_world_landmarks",
    "left_valid",
    "right_valid",
)


def _side_arrays(frames: list[Mapping], key: str) -> tuple[np.ndarray, np.ndarray]:
    points = np.full((len(frames), 21, 3), np.nan, dtype=np.float32)
    valid = np.zeros(len(frames), dtype=bool)
    for index, frame in enumerate(frames):
        value = frame.get(key)
        if value is None:
            continue
        array = np.asarray(value, dtype=np.float32)
        if array.shape != (21, 3):
            raise ValueError(f"{key} frame {index} must have shape (21, 3)")
        if not np.isfinite(array).all():
            raise ValueError(f"{key} frame {index} must be finite")
        points[index] = array
        valid[index] = True
    return points, valid


def write_world_landmark_recording(
    path: str | Path,
    frames: Iterable[Mapping],
    *,
    metadata: Mapping | None = None,
) -> Path:
    """Write captured world landmarks to an atomic HDF5 recording."""
    rows = list(frames)
    if not rows:
        raise ValueError("World landmark recording requires at least one frame")

    frame_ids = np.asarray(
        [frame.get("frame_id", index) for index, frame in enumerate(rows)],
        dtype=np.int64,
    )
    timestamps = np.asarray(
        [frame["timestamp"] for frame in rows], dtype=np.float64
    )
    if not np.isfinite(timestamps).all():
        raise ValueError("World landmark timestamps must be finite")
    left, left_valid = _side_arrays(rows, "world_left")
    right, right_valid = _side_arrays(rows, "world_right")

    destination = Path(path)
    attributes = {
        "dataset_format_version": FORMAT_VERSION,
        "source_landmark_space": "mediapipe_world_meters",
        "length_unit": "m",
        "world_origin": "hand_geometric_center",
        "frame_count": len(rows),
        "created_utc": datetime.now(timezone.utc).isoformat(),
        **({} if metadata is None else {key: str(value) for key, value in metadata.items()}),
    }
    with atomic_output(destination) as temporary:
        with h5py.File(temporary, "w") as h5_file:
            h5_file.create_dataset("frame_ids", data=frame_ids)
            h5_file.create_dataset("timestamps", data=timestamps)
            h5_file.create_dataset("left_world_landmarks", data=left)
            h5_file.create_dataset("right_world_landmarks", data=right)
            h5_file.create_dataset("left_valid", data=left_valid)
            h5_file.create_dataset("right_valid", data=right_valid)
            h5_file.attrs.update(attributes)
    return destination


def read_world_landmark_recording(path: str | Path) -> tuple[dict[str, np.ndarray], dict]:
    """Read and validate a world landmark HDF5 recording."""
    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(f"World landmark recording was not found: {source}")
    with h5py.File(source, "r") as h5_file:
        missing = [name for name in DATASETS if name not in h5_file]
        if missing:
            raise ValueError(
                f"World landmark recording is missing datasets: {', '.join(missing)}"
            )
        arrays = {name: np.asarray(h5_file[name][:]) for name in DATASETS}
        attributes = dict(h5_file.attrs)
    count = arrays["timestamps"].shape[0]
    for name in DATASETS:
        if arrays[name].shape[0] != count:
            raise ValueError(f"Dataset {name} does not match timestamps length")
    for name in ("left_world_landmarks", "right_world_landmarks"):
        if arrays[name].shape != (count, 21, 3):
            raise ValueError(f"Dataset {name} must have shape ({count}, 21, 3)")
    return arrays, attributes
