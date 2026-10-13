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
    if angles.shape[-1] not in (HAND_ANGLE_DIM, HAND_ANGLE_DIM + FK_FIXED_TIPS):
        raise ValueError("Expected (...,18) or (...,23) hand angles")
    signs = np.ones(angles.shape[-1], dtype=np.float32)
    if side == "left": signs[[1, 4, 7, 10]] = -1
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
