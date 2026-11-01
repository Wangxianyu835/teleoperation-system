import numpy as np
from teleoperation.contracts.constants import HAND_ANGLE_DIM, FK_FIXED_TIPS


def l21_joint_result(side, angles, timestamp):
    """Convert a historical 18D inference value to the common native result."""
    from teleoperation.contracts.hand_control import HandJointResult
    from teleoperation.robots.hand_spec import load_hand_spec

    spec = load_hand_spec("l21", side)
    if angles is not None:
        angles = np.asarray(angles)
        if angles.shape != (18,) or not np.isfinite(angles).all():
            raise ValueError("Historical L21 inference requires 18 finite values")
    return HandJointResult("l21", side, spec.spec_hash, spec.independent_joint_names,
                           None if angles is None else angles[1:], timestamp,
                           valid=angles is not None,
                           error="missing input or unfinished calibration" if angles is None else None)


def l21_legacy_values(result):
    """Serialize common L21 independent coordinates into historical 18D layout."""
    if result.hand_model != "l21" or result.positions is None or result.positions.shape != (17,):
        raise ValueError("Historical L21 output requires a valid L21 joint result")
    return np.concatenate((np.zeros(1, dtype=result.positions.dtype), result.positions))


def legacy_l21_dofs(row):
    """Read historical 18D storage/FK layout without reducing float64 precision."""
    values = np.asarray(row)
    if values.shape != (18,) or not np.isfinite(values).all():
        raise ValueError("Historical L21 layout must contain 18 finite values")
    return values[1:].copy()


def canonical_to_native_angles(angles, side: str):
    """Map canonical right-angle semantics to the shipped L21 URDFs.

    Zero-pose axes transformed as axial vectors (det(K) * axis @ K) agree
    between sides except four MCP roll axes. Their left-native signs reverse.
    Works on (...,18) or padded (...,23) NumPy/Torch values without mutation.
    No offsets: geometry differences are retained in the two native FKs.
    """
    if side not in ("left", "right"):
        raise ValueError(f"Unknown hand side: {side!r}")
    if angles.shape[-1] not in (HAND_ANGLE_DIM - 1, HAND_ANGLE_DIM, HAND_ANGLE_DIM + FK_FIXED_TIPS):
        raise ValueError("Expected (...,17), (...,18) or (...,23) hand angles")
    signs = np.ones(angles.shape[-1], dtype=np.float32)
    if side == "left":
        indices = (0, 3, 6, 9) if angles.shape[-1] == HAND_ANGLE_DIM - 1 else (1, 4, 7, 10)
        signs[list(indices)] = -1
    multiplier = angles.new_tensor(signs) if hasattr(angles, "new_tensor") else signs
    return angles * multiplier


def canonical_to_native_target_angles(angles, canonical_spec, target_spec):
    """Convert one right-canonical target output to a target side's native axes."""
    if canonical_spec.hand_model != target_spec.hand_model or canonical_spec.side != "right":
        raise ValueError("Canonical and target specs must describe one model and use the right canonical spec")
    if target_spec.dof != canonical_spec.dof:
        raise ValueError("Canonical and target specs must have matching output dimensions")
    if target_spec.side == "right":
        return angles
    if target_spec.hand_model == "l21":
        return canonical_to_native_angles(angles, "left")

    right_bounds = canonical_spec.bounds
    left_bounds = target_spec.bounds
    signs = np.ones(target_spec.dof, dtype=np.float32)

    def direction(bounds):
        low, high = bounds
        if low >= -1e-6 and high > 1e-6:
            return 1
        if high <= 1e-6 and low < -1e-6:
            return -1
        return 0

    for index, (right, left) in enumerate(zip(right_bounds, left_bounds)):
        if direction(right) and direction(left) and direction(right) != direction(left):
            signs[index] = -1
    multiplier = angles.new_tensor(signs) if hasattr(angles, "new_tensor") else signs
    return angles * multiplier
def angle18_to_dofs(angle: np.ndarray) -> np.ndarray:
    value = _angle18(angle)
    return value[1:].copy()


def angle18_to_nodes(angle: np.ndarray) -> np.ndarray:
    value = _angle18(angle)
    return np.concatenate((value, np.zeros(FK_FIXED_TIPS, dtype=np.float32)))


def _angle18(angle: np.ndarray) -> np.ndarray:
    value = np.asarray(angle, dtype=np.float32).reshape(-1)
    if value.shape != (HAND_ANGLE_DIM,):
        raise ValueError(f"Expected 18 hand angles, got {value.shape}")
    if not np.isfinite(value).all():
        raise ValueError("Hand angles contain NaN or Inf")
    return value
