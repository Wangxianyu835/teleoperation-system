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

RECEPTIVE_FIELD = 3 #time window length for retargeting model input
HAND_KEYPOINTS = 25 #number of keypoints for each hand
HAND_COORDS = 3 #维度
ARM_KEYPOINTS = 3
INPUT_KEY = "retarget_input"
LEGACY_VISIONPRO_KEY = "vision_pro_data"


def validate_hand_input(
    hand_data: np.ndarray | None,
    side: str,
    allow_missing: bool = True,
) -> bool:
    """to check one hand input window with shape (frames, 25, 3). it doesn't check the arms data"""
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


def validate_arm_input(
    arm_data: np.ndarray | None,
    side: str,
    allow_missing: bool = True,
) -> bool:
    """Validate one arm observation with shoulder/elbow/wrist keypoints."""
    if side not in ARM_SIDES:
        raise ValueError(f"Invalid arm side: {side}")

    if arm_data is None:
        if allow_missing:
            return True
        raise ValueError(f"{side}_arm is missing")

    value = np.asarray(arm_data, dtype=np.float32)
    expected_shape = (ARM_KEYPOINTS, HAND_COORDS)
    if value.shape != expected_shape:
        raise ValueError(f"{side}_arm shape must be {expected_shape}, got {value.shape}")
    if not np.issubdtype(value.dtype, np.number):
        raise TypeError(f"{side}_arm must contain numeric values")
    if not np.isfinite(value).all():
        raise ValueError(f"{side}_arm contains NaN or infinite values")
    return True


def validate_retarget_input(
    retarget_input: dict[str, Any],
    allow_missing_hands: bool = True,
    allow_missing_arms: bool = True,
    require_arms: bool = False,
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

    if "arms" not in retarget_input:
        if require_arms:
            raise KeyError("retarget_input must contain 'arms'")
    else:
        if not isinstance(retarget_input["arms"], dict):
            raise TypeError("retarget_input['arms'] must be a dict")
        for side in ARM_SIDES:
            if side not in retarget_input["arms"]:
                raise KeyError(f"retarget_input['arms'] missing '{side}'")
            validate_arm_input(
                retarget_input["arms"][side],
                side,
                allow_missing=allow_missing_arms,
            )

    return True


def build_retarget_input(
    source: str,
    timestamp: float | None,
    left_hand: np.ndarray | None = None,
    right_hand: np.ndarray | None = None,
    left_arm: np.ndarray | None = None,
    right_arm: np.ndarray | None = None,
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
            "left": left_arm,
            "right": right_arm,
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
    """Build the fixed 48D simulator action in canonical limb order."""
    parts = (("left_arm", left_arm, 7), ("left_hand", left_hand, 17), ("right_arm", right_arm, 7), ("right_hand", right_hand, 17))
    values = []
    for name, part, size in parts:
        if part is None:
            raise ValueError(f"{name} is required for a fixed RobotCommand")
        value = np.asarray(part, dtype=np.float32).reshape(-1)
        if value.shape != (size,) or not np.isfinite(value).all():
            raise ValueError(f"{name} must be finite with shape ({size},)")
        values.append(value)
    return np.concatenate(values, axis=0)


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
