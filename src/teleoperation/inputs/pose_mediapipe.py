"""MediaPipe Pose -> shoulder/elbow/wrist observations for data collection."""

from __future__ import annotations

import numpy as np


POSE_ARM_INDICES = (11, 12, 13, 14, 15, 16)
POSE_ARM_ORDER = (
    "left_shoulder",
    "right_shoulder",
    "left_elbow",
    "right_elbow",
    "left_wrist",
    "right_wrist",
)
DEFAULT_POSE_VISIBILITY = 0.2


def _point(landmark) -> tuple[float, float, float]:
    return float(landmark.x), float(landmark.y), float(landmark.z)


def _visibility(landmark) -> float:
    return float(getattr(landmark, "visibility", 1.0))


def _empty_pose() -> dict:
    return {
        "pose_keypoints": np.full((6, 3), np.nan, dtype=np.float32),
        "pose_world_keypoints": np.full((6, 3), np.nan, dtype=np.float32),
        "pose_visibility": np.zeros(6, dtype=np.float32),
        "left_arm_keypoints": np.full((3, 3), np.nan, dtype=np.float32),
        "right_arm_keypoints": np.full((3, 3), np.nan, dtype=np.float32),
        "left_arm_world_keypoints": np.full((3, 3), np.nan, dtype=np.float32),
        "right_arm_world_keypoints": np.full((3, 3), np.nan, dtype=np.float32),
        "left_arm_valid": False,
        "right_arm_valid": False,
        "metadata": {
            "arm_source_space": "mediapipe_pose_normalized",
            "arm_world_space": "mediapipe_pose_world_meters",
            "arm_world_origin": "hip_midpoint",
            "arm_length_unit": "m",
        },
    }


def parse_pose_result(
    result,
    *,
    min_visibility: float = DEFAULT_POSE_VISIBILITY,
) -> dict:
    """Extract six arm points from one MediaPipe PoseLandmarker result."""
    output = _empty_pose()
    landmarks = getattr(result, "pose_landmarks", None)
    if not landmarks:
        return output

    full = landmarks[0]
    if len(full) <= max(POSE_ARM_INDICES):
        raise ValueError("MediaPipe pose result does not contain shoulder/elbow/wrist")

    pose = np.asarray(
        [_point(full[index]) for index in POSE_ARM_INDICES],
        dtype=np.float32,
    )
    visibility = np.asarray(
        [_visibility(full[index]) for index in POSE_ARM_INDICES],
        dtype=np.float32,
    )
    if not np.isfinite(pose).all() or not np.isfinite(visibility).all():
        raise ValueError("MediaPipe pose result contains non-finite values")

    world_landmarks = getattr(result, "pose_world_landmarks", None)
    world = np.full((6, 3), np.nan, dtype=np.float32)
    if world_landmarks:
        full_world = world_landmarks[0]
        if len(full_world) > max(POSE_ARM_INDICES):
            world = np.asarray(
                [_point(full_world[index]) for index in POSE_ARM_INDICES],
                dtype=np.float32,
            )
            if not np.isfinite(world).all():
                raise ValueError("MediaPipe pose world result contains non-finite values")

    output["pose_keypoints"] = pose
    output["pose_world_keypoints"] = world
    output["pose_visibility"] = visibility
    output["left_arm_keypoints"] = pose[[0, 2, 4]]
    output["right_arm_keypoints"] = pose[[1, 3, 5]]
    output["left_arm_world_keypoints"] = world[[0, 2, 4]]
    output["right_arm_world_keypoints"] = world[[1, 3, 5]]
    output["left_arm_valid"] = _side_valid(
        pose[[0, 2, 4]], world[[0, 2, 4]], visibility[[0, 2, 4]], min_visibility
    )
    output["right_arm_valid"] = _side_valid(
        pose[[1, 3, 5]], world[[1, 3, 5]], visibility[[1, 3, 5]], min_visibility
    )
    return output


def _side_valid(
    normalized: np.ndarray,
    world: np.ndarray,
    visibility: np.ndarray,
    min_visibility: float,
) -> bool:
    return bool(
        np.isfinite(normalized).all()
        and np.isfinite(world).all()
        and np.isfinite(visibility).all()
        and np.all(visibility >= float(min_visibility))
    )
