"""Coordinate normalization modes for offline two-hand experiments."""

from __future__ import annotations

from typing import Literal

import torch
import numpy as np


LEFT_COORDINATE_MODES = ("none", "mirror_x")
LeftCoordinateMode = Literal["none", "mirror_x"]


def validate_left_coordinate_mode(mode: str) -> str:
    """Validate and return a supported left-hand coordinate mode."""
    if mode not in LEFT_COORDINATE_MODES:
        choices = ", ".join(LEFT_COORDINATE_MODES)
        raise ValueError(
            f"Unsupported left coordinate mode {mode!r}; choose one of {choices}"
        )
    return mode


def transform_hand_coordinates(
    points: np.ndarray,
    side: str,
    mode: str = "none",
) -> np.ndarray:
    """Transform hand coordinates for the shared canonical model frame."""
    validate_left_coordinate_mode(mode)
    if side not in ("left", "right"):
        raise ValueError(f"Unsupported hand side: {side}")

    result = np.asarray(points, dtype=np.float32).copy()
    if side == "left" and mode == "mirror_x":
        result[..., 0] *= -1.0
    return result


def transform_fk_positions(
    positions: torch.Tensor,
    side: str,
    mode: str = "none",
) -> torch.Tensor:
    """Transform FK positions into the same frame as model targets."""
    validate_left_coordinate_mode(mode)
    if side not in ("left", "right"):
        raise ValueError(f"Unsupported hand side: {side}")

    if side == "left" and mode == "mirror_x":
        result = positions.clone()
        result[..., 0] *= -1.0
        return result
    return positions
