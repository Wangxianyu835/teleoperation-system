import numpy as np
from teleoperation.contracts.constants import HAND_ANGLE_DIM, FK_FIXED_TIPS
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
