"""Shared input/output contracts for realtime and offline retargeting."""

from __future__ import annotations

from typing import Any

import numpy as np

from retargeting.contracts import (
    HAND_SIDES, RECEPTIVE_FIELD, HAND_KEYPOINTS, HAND_COORDS,
    validate_hand_input as _validate_hand_input,
    legacy_visionpro_to_window,
    build_retarget_input as _build_hand_input,
)


ARM_SIDES = ("left", "right")

ACTION_ORDER = (
    "left_arm",
    "left_hand",
    "right_arm",
    "right_hand",
)

INPUT_KEY = "retarget_input"
LEGACY_VISIONPRO_KEY = "vision_pro_data"


def validate_hand_input(
    hand_data: np.ndarray | None,
    side: str,
    allow_missing: bool = True,
) -> bool:
    """Delegate hand shape/finite validation to the canonical subsystem."""
    return _validate_hand_input(hand_data, side, allow_missing=allow_missing)


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
    """Use the canonical hand builder; application arm fields remain optional."""
    return _build_hand_input(source, timestamp, left_hand, right_hand, metadata=metadata)


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
