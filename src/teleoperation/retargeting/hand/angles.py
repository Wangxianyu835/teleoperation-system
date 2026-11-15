import numpy as np
from teleoperation.contracts.constants import HAND_ANGLE_DIM, FK_FIXED_TIPS








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
