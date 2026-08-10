"""Hand keypoint conversion helpers for retargeting input adapters."""

from __future__ import annotations

from collections import deque
from typing import Deque

import numpy as np

from config.retarget_io import (
    HAND_COORDS,
    HAND_KEYPOINTS,
    HAND_SIDES,
    RECEPTIVE_FIELD,
    build_retarget_input,
)

MEDIAPIPE_HAND_KEYPOINTS = 21
VISIONPRO_SOURCE = "visionpro"
MEDIAPIPE_APPROX_SOURCE = "mediapipe_approx"
NPY_REPLAY_SOURCE = "npy_replay"


def wrist_relative(points: np.ndarray, scale_factor: float = 1.0) -> np.ndarray:
    """Return points relative to wrist index 0."""
    points = np.asarray(points, dtype=np.float32)
    if points.ndim != 2 or points.shape[1] != HAND_COORDS:
        raise ValueError(f"Hand points must have shape (joints, {HAND_COORDS})")
    return (points - points[0:1]) * float(scale_factor)


def ensure_hand25(points: np.ndarray, scale_factor: float = 1.0) -> np.ndarray:
    """Convert supported hand point layouts to the canonical 25x3 layout."""
    points = np.asarray(points, dtype=np.float32)
    if points.shape == (HAND_KEYPOINTS, HAND_COORDS):
        return wrist_relative(points, scale_factor=scale_factor)
    if points.shape == (MEDIAPIPE_HAND_KEYPOINTS, HAND_COORDS):
        return mediapipe21_to_hand25(points, scale_factor=scale_factor)
    raise ValueError(
        "Hand points must have shape "
        f"({HAND_KEYPOINTS}, {HAND_COORDS}) or "
        f"({MEDIAPIPE_HAND_KEYPOINTS}, {HAND_COORDS}), got {points.shape}"
    )


def mediapipe21_to_hand25(points: np.ndarray, scale_factor: float = 1.0) -> np.ndarray:
    """Map MediaPipe's 21 hand landmarks into the project's 25 point topology.

    The project topology adds one palm-root point before each non-thumb finger.
    MediaPipe does not provide these directly, so the first version uses the
    midpoint between wrist and each finger MCP.
    """
    mp_points = np.asarray(points, dtype=np.float32)
    if mp_points.shape != (MEDIAPIPE_HAND_KEYPOINTS, HAND_COORDS):
        raise ValueError(
            f"MediaPipe hand must have shape "
            f"({MEDIAPIPE_HAND_KEYPOINTS}, {HAND_COORDS}), got {mp_points.shape}"
        )

    out = np.zeros((HAND_KEYPOINTS, HAND_COORDS), dtype=np.float32)
    out[0] = mp_points[0]

    out[1:5] = mp_points[1:5]
    _copy_finger_with_palm_root(out, mp_points, out_start=5, mp_start=5)
    _copy_finger_with_palm_root(out, mp_points, out_start=10, mp_start=9)
    _copy_finger_with_palm_root(out, mp_points, out_start=15, mp_start=13)
    _copy_finger_with_palm_root(out, mp_points, out_start=20, mp_start=17)

    return wrist_relative(out, scale_factor=scale_factor)


def _copy_finger_with_palm_root(
    out: np.ndarray,
    mp_points: np.ndarray,
    out_start: int,
    mp_start: int,
) -> None:
    out[out_start] = 0.5 * (mp_points[0] + mp_points[mp_start])
    out[out_start + 1 : out_start + 5] = mp_points[mp_start : mp_start + 4]


class HandWindowBuffer:
    """Collect per-frame hand points and emit canonical three-frame windows."""

    def __init__(self, scale_factor: float = 1.0):
        self.scale_factor = scale_factor
        self._buffers: dict[str, Deque[np.ndarray]] = {
            side: deque(maxlen=RECEPTIVE_FIELD) for side in HAND_SIDES
        }

    def update(
        self,
        left_hand: np.ndarray | None = None,
        right_hand: np.ndarray | None = None,
        timestamp: float | None = None,
        source: str = "unknown",
        metadata: dict | None = None,
    ) -> dict | None:
        for side, points in (("left", left_hand), ("right", right_hand)):
            if points is not None:
                self._buffers[side].append(
                    ensure_hand25(points, scale_factor=self.scale_factor)
                )

        left_window = self._window_or_none("left")
        right_window = self._window_or_none("right")
        if left_window is None and right_window is None:
            return None

        return build_retarget_input(
            source=source,
            timestamp=timestamp,
            left_hand=left_window,
            right_hand=right_window,
            metadata=metadata,
        )

    def _window_or_none(self, side: str) -> np.ndarray | None:
        buffer = self._buffers[side]
        if len(buffer) != RECEPTIVE_FIELD:
            return None
        return np.stack(tuple(buffer), axis=0).astype(np.float32, copy=False)
