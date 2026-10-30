"""Convert SMPLer-X worker arrays into the project's body and wrist contracts."""
from __future__ import annotations

from typing import Mapping

import numpy as np
from scipy.spatial.transform import Rotation

from teleoperation.contracts.body import SmplerXResult
from teleoperation.contracts.observations import UpperBodyObservation
from teleoperation.contracts.wrist import WristObservation


def _rotation(value: np.ndarray, name: str) -> np.ndarray:
    matrix = np.asarray(value, dtype=float)
    if matrix.shape != (3, 3) or not np.isfinite(matrix).all():
        raise ValueError(f"{name} must have shape (3,3) and finite values")
    if not np.allclose(matrix.T @ matrix, np.eye(3), atol=1e-5) or not np.isclose(np.linalg.det(matrix), 1.0, atol=1e-5):
        raise ValueError(f"{name} must be a proper rotation")
    return matrix


def fk_global_rotations(local_rotations: np.ndarray, parents: list[int] | tuple[int, ...]) -> np.ndarray:
    """Compose local joint rotations through a parent index array."""
    local = np.asarray(local_rotations, dtype=float)
    if local.ndim != 3 or local.shape[1:] != (3, 3) or len(parents) != local.shape[0]:
        raise ValueError("local_rotations and parents have incompatible shapes")
    global_rotations = np.empty_like(local)
    for index, parent in enumerate(parents):
        _rotation(local[index], f"local_rotations[{index}]")
        if parent < -1 or parent >= index:
            raise ValueError(f"invalid parent index {parent} for joint {index}")
        global_rotations[index] = local[index] if parent == -1 else global_rotations[parent] @ local[index]
        _rotation(global_rotations[index], f"global_rotations[{index}]")
    return global_rotations


def _indices(result: SmplerXResult) -> Mapping[str, Mapping[str, int]]:
    metadata = dict(result.metadata or {})
    values = metadata.get("joint_indices")
    if not isinstance(values, Mapping):
        raise ValueError("SMPL-X result metadata must contain joint_indices")
    for side in ("left", "right"):
        if not isinstance(values.get(side), Mapping):
            raise ValueError(f"joint_indices missing {side}")
        for joint in ("shoulder", "elbow", "wrist"):
            if joint not in values[side]:
                raise ValueError(f"joint_indices missing {side}.{joint}")
    return values


def smplerx_observations(result: SmplerXResult) -> tuple[UpperBodyObservation, dict[str, WristObservation]]:
    """Build body positions and wrist orientations from one worker result."""
    joints = np.asarray(result.arrays["joints_root_m"], dtype=float)
    rotations = np.asarray(result.arrays["global_rotations"], dtype=float)
    image_joints = np.asarray(result.arrays.get("image_joints", joints), dtype=float)
    if joints.ndim != 2 or joints.shape[1] != 3 or not np.isfinite(joints).all():
        raise ValueError("joints_root_m must have shape (J,3) and finite values")
    if rotations.shape != (len(joints), 3, 3):
        raise ValueError("global_rotations must have shape (J,3,3)")
    if image_joints.shape != joints.shape or not np.isfinite(image_joints).all():
        raise ValueError("image_joints must have shape (J,3) and finite values")
    index_map = _indices(result)
    arms: dict[str, np.ndarray | None] = {}
    world_arms: dict[str, np.ndarray | None] = {}
    valid: dict[str, bool] = {}
    wrists: dict[str, WristObservation] = {}
    for side in ("left", "right"):
        indices = [int(index_map[side][joint]) for joint in ("shoulder", "elbow", "wrist")]
        if any(index < 0 or index >= len(joints) for index in indices):
            raise ValueError(f"{side} joint index is outside result arrays")
        points = joints[indices]
        projected = image_joints[indices]
        rotation = _rotation(rotations[indices[2]], f"{side} wrist rotation")
        side_valid = bool(result.valid.get(side, False)) and np.isfinite(points).all() and np.isfinite(projected).all()
        valid[side] = side_valid
        arms[side] = projected if side_valid else None
        world_arms[side] = points if side_valid else None
        wrists[side] = WristObservation(
            Rotation.from_matrix(rotation).as_quat() if side_valid else None,
            "smplx_camera_axes", "smplx_wrist_local", "smplerx",
            result.frame_id.capture_time_s, points[2].copy() if side_valid else None,
            orientation_valid=side_valid, position_valid=side_valid,
            reason="" if side_valid else result.reason or "invalid SMPL-X side",
        )
    body = UpperBodyObservation(
        arms=arms, valid=valid, timestamp=result.frame_id.capture_time_s,
        coordinate_frame="smplx_image_normalized", units="normalized",
        world_arms=world_arms,
        metadata={
            "source": "smplerx",
            "model_id": result.model_id,
            "model_hash": result.model_hash,
            "source_coordinate_frame": "smplx_camera_root_relative_m",
            "source_units": "m",
            "sample_sequence": result.frame_id.sequence,
        },
    )
    return body, wrists
