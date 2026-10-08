from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation
ARM_DOF = 7
from .calibration import ArmCalibration
from teleoperation.contracts.observations import UpperBodyObservation
from teleoperation.contracts.arm import ArmTargetPose
class ArmRetargeter:
    """Map shoulder/elbow/wrist observations to calibrated TRON2A EE targets."""

    def __init__(self, calibration: ArmCalibration):
        self.calibration = calibration
        self._previous: dict[str, np.ndarray] = {
            side: calibration.robot_neutral_pose[side].astype(np.float32).copy()
            for side in ("left", "right")
        }

    def update(self, observation: UpperBodyObservation, side: str) -> ArmTargetPose:
        if not isinstance(observation, UpperBodyObservation): raise TypeError("Expected UpperBodyObservation")
        if side not in ("left", "right"):
            raise ValueError(f"Invalid arm side: {side}")
        keypoints, valid = observation.arms[side], observation.valid[side]
        points = np.asarray(keypoints, dtype=np.float64)
        if not valid or points.shape != (3, 3) or not np.isfinite(points).all():
            return ArmTargetPose(side, self._previous[side].copy(), False)
        orientation = _forearm_rotation(points)
        if orientation is None:
            return ArmTargetPose(side, self._previous[side].copy(), False)
        wrist = points[2]
        position = self.calibration.robot_neutral_pose[side][:3] + self.calibration.translation + self.calibration.rotation @ (self.calibration.scale * (wrist - self.calibration.human_neutral_wrist[side]))
        position = np.clip(position, self.calibration.workspace_min, self.calibration.workspace_max)
        tool_rotation = Rotation.from_quat(self.calibration.tool_offset_quat_xyzw[side]).as_matrix()
        output_rotation = self.calibration.rotation @ orientation @ tool_rotation
        pose = np.concatenate((position, Rotation.from_matrix(output_rotation).as_quat())).astype(np.float32)
        self._previous[side] = pose.copy()
        return ArmTargetPose(side, pose, True)


def _forearm_rotation(points: np.ndarray) -> np.ndarray | None:
    upper = points[1] - points[0]
    forearm = points[2] - points[1]
    if np.linalg.norm(upper) < 1e-6 or np.linalg.norm(forearm) < 1e-6:
        return None
    x = forearm / np.linalg.norm(forearm)
    z = np.cross(upper, forearm)
    if np.linalg.norm(z) < 1e-6:
        return None
    z /= np.linalg.norm(z)
    y = np.cross(z, x)
    return np.column_stack((x, y, z))
