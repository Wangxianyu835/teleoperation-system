"""Distinct arm observations, target poses and inverse-kinematics results."""
from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class ArmTargetPose:
    side: str
    pose: np.ndarray
    valid: bool = True
    robot: str = "tron2a"
    coordinate_frame: str = "base_Link"

    def __post_init__(self):
        if self.side not in ("left", "right"):
            raise ValueError(f"Invalid arm side: {self.side}")
        if np.shape(self.pose) != (7,) or not np.isfinite(self.pose).all():
            raise ValueError("ArmTargetPose must be finite xyz + xyzw with shape (7,)")


@dataclass(frozen=True)
class ArmIKResult:
    success: bool
    q: np.ndarray
    position_error: float
    orientation_error: float
    side: str
    robot: str = "tron2a"

    def __post_init__(self):
        if self.side not in ("left", "right") or np.shape(self.q) != (7,) or not np.isfinite(self.q).all():
            raise ValueError("ArmIKResult requires a known side and finite q[7]")
        if self.success and not np.isfinite([self.position_error, self.orientation_error]).all():
            raise ValueError("Successful IK errors must be finite")
        if np.isnan([self.position_error, self.orientation_error]).any():
            raise ValueError("IK diagnostics must not contain NaN")
