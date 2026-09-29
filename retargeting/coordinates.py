"""Fixed source-hand to LinkerHand L21 coordinate alignment."""

from __future__ import annotations

import numpy as np


COORDINATE_FRAME = "l21"
COORDINATE_ALIGNMENT = "source_to_l21_xyz"

# For row-vector points, p_aligned = p_source @ MATRIX.T.
# This encodes x'=-y, y'=z, z'=-x.
SOURCE_TO_L21_MATRIX = np.asarray(
    (
        (0.0, -1.0, 0.0),
        (0.0, 0.0, 1.0),
        (-1.0, 0.0, 0.0),
    ),
    dtype=np.float32,
)


def align_source_hand_coordinates(points: np.ndarray) -> np.ndarray:
    """Map source-hand coordinates into the fixed L21 frame.

    Supports individual hands, temporal windows, and batches. The input is
    never modified in place.
    """
    values = np.asarray(points, dtype=np.float32)
    if values.ndim < 2 or values.shape[-1] != 3:
        raise ValueError(
            "Hand coordinates must have at least two dimensions and shape "
            f"(..., 3), got {values.shape}"
        )
    if not np.isfinite(values).all():
        raise ValueError("Hand coordinates must be finite")
    aligned = np.matmul(values, SOURCE_TO_L21_MATRIX.T)
    if not np.isfinite(aligned).all():
        raise ValueError("Aligned hand coordinates contain NaN or Inf")
    return aligned.astype(np.float32, copy=False)
