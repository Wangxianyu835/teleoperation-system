"""Deterministic synthetic hand frames for hand-core regression tests."""

from __future__ import annotations

import numpy as np


def fixed_mediapipe_hand() -> np.ndarray:
    """Return one deterministic MediaPipe-layout hand with shape (21, 3)."""
    return np.arange(21 * 3, dtype=np.float32).reshape(21, 3) / 100.0


def continuous_three_frames() -> np.ndarray:
    """Return three nearby frames; frame two has a small translation."""
    hand = fixed_mediapipe_hand()
    translations = np.asarray(
        ([0.0, 0.0, 0.0], [0.01, 0.0, 0.0], [0.02, 0.0, 0.0]),
        dtype=np.float32,
    )
    return hand[None, :, :] + translations[:, None, :]


def left_hand() -> np.ndarray:
    return fixed_mediapipe_hand()


def right_hand() -> np.ndarray:
    return fixed_mediapipe_hand() + np.asarray([0.4, 0.0, 0.0], dtype=np.float32)


def identity_jump() -> np.ndarray:
    return fixed_mediapipe_hand() + np.asarray([1.0, 0.0, 0.0], dtype=np.float32)


def nan_frame() -> np.ndarray:
    frame = fixed_mediapipe_hand()
    frame[7, 1] = np.nan
    return frame


def zero_frame() -> np.ndarray:
    return np.zeros((21, 3), dtype=np.float32)


def missing_left() -> None:
    return None


def missing_right() -> None:
    return None
