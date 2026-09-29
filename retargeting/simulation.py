"""Stable simulator-facing adapters for exported angle H5 files."""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

import h5py
import numpy as np

from retargeting.contracts import HAND_SIDES


@dataclass(frozen=True)
class AngleFrame:
    frame_id: int
    timestamp: float
    left: np.ndarray
    right: np.ndarray
    left_valid: bool
    right_valid: bool


def angle18_to_dofs(angle: np.ndarray) -> np.ndarray:
    value = _angle18(angle)
    return value[1:].copy()


def angle18_to_nodes(angle: np.ndarray) -> np.ndarray:
    value = _angle18(angle)
    return np.concatenate((value, np.zeros(5, dtype=np.float32)))


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
        if angles[side].shape != (frame_count, 18):
            raise ValueError(f"{side}_angles must have shape ({frame_count}, 18)")
        if valid[side].shape != (frame_count,):
            raise ValueError(f"{side}_valid must have shape ({frame_count},)")
        if not np.isfinite(angles[side]).all():
            raise ValueError(f"{side}_angles contains NaN or Inf")
    if timestamps.shape != (frame_count,):
        raise ValueError("timestamps length does not match frame_ids")

    previous = {side: np.zeros(18, dtype=np.float32) for side in HAND_SIDES}
    for index in range(frame_count):
        for side in HAND_SIDES:
            if valid[side][index]:
                previous[side] = angles[side][index].copy()
        yield AngleFrame(
            frame_id=int(frame_ids[index]),
            timestamp=float(timestamps[index]),
            left=previous["left"].copy(),
            right=previous["right"].copy(),
            left_valid=bool(valid["left"][index]),
            right_valid=bool(valid["right"][index]),
        )


def _angle18(angle: np.ndarray) -> np.ndarray:
    value = np.asarray(angle, dtype=np.float32).reshape(-1)
    if value.shape != (18,):
        raise ValueError(f"Expected 18 hand angles, got {value.shape}")
    if not np.isfinite(value).all():
        raise ValueError("Hand angles contain NaN or Inf")
    return value
