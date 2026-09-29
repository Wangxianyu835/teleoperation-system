"""Shared input/output contracts for realtime and offline retargeting."""

from __future__ import annotations

from typing import Any

import numpy as np


HAND_SIDES = ("left", "right")
ARM_SIDES = ("left", "right")

ACTION_ORDER = (
    "left_arm",
    "left_hand",
    "right_arm",
    "right_hand",
)

RECEPTIVE_FIELD = 3
HAND_KEYPOINTS = 25
HAND_COORDS = 3
INPUT_KEY = "retarget_input"
LEGACY_VISIONPRO_KEY = "vision_pro_data"


def validate_hand_input(
    hand_data: np.ndarray | None,
    side: str,
    allow_missing: bool = True,
) -> bool:
    """Validate one hand input window with shape (frames, 25, 3)."""
    if side not in HAND_SIDES:
        raise ValueError(f"Invalid hand side: {side}")

    if hand_data is None:
        if allow_missing:
            return True
        raise ValueError(f"{side}_hand is missing")

    if not isinstance(hand_data, np.ndarray):
        raise TypeError(f"{side}_hand must be np.ndarray")

    expected_shape = (RECEPTIVE_FIELD, HAND_KEYPOINTS, HAND_COORDS)
    if hand_data.shape != expected_shape:
        raise ValueError(
            f"{side}_hand shape must be {expected_shape}, got {hand_data.shape}"
        )

    if not np.issubdtype(hand_data.dtype, np.number):
        raise TypeError(f"{side}_hand must contain numeric values")
    if not np.isfinite(hand_data).all():
        raise ValueError(f"{side}_hand contains NaN or infinite values")
    return True


def validate_retarget_input(
    retarget_input: dict[str, Any],
    allow_missing_hands: bool = True,
) -> bool:
    """Validate the shared realtime/offline retargeting input structure."""
    if not isinstance(retarget_input, dict):
        raise TypeError("retarget_input must be a dict")

    if "hands" not in retarget_input:
        raise KeyError("retarget_input must contain 'hands'")
    if not isinstance(retarget_input["hands"], dict):
        raise TypeError("retarget_input['hands'] must be a dict")

    for side in HAND_SIDES:
        if side not in retarget_input["hands"]:
            raise KeyError(f"retarget_input['hands'] missing '{side}'")
        validate_hand_input(
            retarget_input["hands"][side],
            side,
            allow_missing=allow_missing_hands,
        )

    if "arms" in retarget_input:
        if not isinstance(retarget_input["arms"], dict):
            raise TypeError("retarget_input['arms'] must be a dict")
        for side in ARM_SIDES:
            if side not in retarget_input["arms"]:
                raise KeyError(f"retarget_input['arms'] missing '{side}'")

    return True


def build_retarget_input(
    source: str,
    timestamp: float | None,
    left_hand: np.ndarray | None = None,
    right_hand: np.ndarray | None = None,
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build the canonical payload shared by all input sources."""
    payload: dict[str, Any] = {
        "source": source,
        "timestamp": timestamp,
        "hands": {
            "left": left_hand,
            "right": right_hand,
        },
        "arms": {
            "left": None,
            "right": None,
        },
    }
    if metadata:
        payload["metadata"] = dict(metadata)
    validate_retarget_input(payload)
    return payload


def legacy_visionpro_to_window(vision_data: np.ndarray) -> np.ndarray:
    """Convert the old Vision Pro shared shape (25, 3, 3) to (3, 25, 3)."""
    data = np.asarray(vision_data, dtype=np.float32)
    if data.shape == (RECEPTIVE_FIELD, HAND_KEYPOINTS, HAND_COORDS):
        return data
    legacy_shape = (HAND_KEYPOINTS, RECEPTIVE_FIELD, HAND_COORDS)
    if data.shape != legacy_shape:
        raise ValueError(
            f"Legacy Vision Pro data must have shape {legacy_shape}, got {data.shape}"
        )
    return np.transpose(data, (1, 0, 2)).copy()


def select_hand_window(
    retarget_input: dict[str, Any],
    preferred_side: str = "left",
) -> tuple[str, np.ndarray] | None:
    """Select one complete hand window for the current single-hand model."""
    if preferred_side not in HAND_SIDES:
        raise ValueError(f"Invalid preferred side: {preferred_side}")

    validate_retarget_input(retarget_input)
    sides = (preferred_side,) + tuple(
        side for side in HAND_SIDES if side != preferred_side
    )
    for side in sides:
        hand = retarget_input["hands"][side]
        if hand is not None:
            return side, hand
    return None


def build_action(
    left_arm: np.ndarray | None,
    left_hand: np.ndarray | None,
    right_arm: np.ndarray | None,
    right_hand: np.ndarray | None,
) -> np.ndarray:
    """Build the simulator action in left_arm, left_hand, right_arm, right_hand order."""
    parts = (left_arm, left_hand, right_arm, right_hand)
    valid_parts = [np.asarray(part, dtype=np.float32).reshape(-1) for part in parts if part is not None]

    if not valid_parts:
        raise ValueError("No valid action parts provided")

    return np.concatenate(valid_parts, axis=0)


def build_retarget_output(
    left_hand: np.ndarray,
    right_hand: np.ndarray,
    left_arm: np.ndarray | None = None,
    right_arm: np.ndarray | None = None,
    timestamp: float | None = None,
) -> dict[str, Any]:
    """Create the standard output payload sent to simulation or hardware adapters."""
    action = build_action(left_arm, left_hand, right_arm, right_hand)
    return {
        "timestamp": timestamp,
        "hands": {
            "left": left_hand,
            "right": right_hand,
        },
        "arms": {
            "left": left_arm,
            "right": right_arm,
        },
        "action": action,
    }
