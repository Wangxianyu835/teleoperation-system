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

    def __init__(self, calibration: ArmCalibration, *, input_mode="legacy", orientation_mode=None):
        self.calibration = calibration
        if input_mode not in ("legacy", "camera_world"):
            raise ValueError("Unknown arm input mode")
        self.input_mode = input_mode
        self.orientation_mode = orientation_mode or ("fixed" if input_mode == "camera_world" else "forearm")
        if self.orientation_mode not in ("fixed", "forearm"):
            raise ValueError("Unknown arm orientation mode")
        if input_mode == "camera_world":
            expected = dict(coordinate_source="mediapipe_pose_world_meters", units="m",
                            human_reference="each_side_own_shoulder", robot_reference_frame="base_Link",
                            point_order=["shoulder", "elbow", "wrist"])
            if any(calibration.metadata.get(key) != value for key, value in expected.items()):
                raise ValueError("Camera calibration source, units or shoulder reference mismatch")
        self.orientation_held = {side: False for side in ("left", "right")}
        self._previous: dict[str, np.ndarray] = {
            side: calibration.robot_neutral_pose[side].astype(np.float32).copy()
            for side in ("left", "right")
        }
        self._forearm_previous = {
            side: calibration.rotation.T @ Rotation.from_quat(calibration.robot_neutral_pose[side][3:]).as_matrix()
            @ Rotation.from_quat(calibration.tool_offset_quat_xyzw[side]).as_matrix().T
            for side in ("left", "right")
        }

    def update(self, observation: UpperBodyObservation, side: str) -> ArmTargetPose:
        if not isinstance(observation, UpperBodyObservation): raise TypeError("Expected UpperBodyObservation")
        if side not in ("left", "right"):
            raise ValueError(f"Invalid arm side: {side}")
        if self.input_mode == "camera_world":
            if observation.coordinate_frame != "mediapipe_pose_normalized" or observation.units != "normalized":
                raise ValueError("Expected combined MediaPipe Pose normalized/world observation")
            keypoints = None if observation.world_arms is None else observation.world_arms[side]
        else:
            keypoints = observation.arms[side]
        valid = observation.valid[side]
        points = np.asarray(keypoints, dtype=np.float64)
        if not valid or points.shape != (3, 3) or not np.isfinite(points).all():
            return ArmTargetPose(side, self._previous[side].copy(), False)
        orientation = _forearm_rotation(points, min_bend_sine=0.05 if self.input_mode == "camera_world" else 0.) if self.orientation_mode == "forearm" else None
        self.orientation_held[side] = self.orientation_mode == "forearm" and orientation is None
        if self.input_mode == "legacy" and self.orientation_held[side]:
            return ArmTargetPose(side, self._previous[side].copy(), False)
        wrist = points[2] - points[0] if self.input_mode == "camera_world" else points[2]
        position = self.calibration.robot_neutral_pose[side][:3] + self.calibration.translation + self.calibration.rotation @ (self.calibration.scale * (wrist - self.calibration.human_neutral_wrist[side]))
        if self.input_mode == "legacy":
            position = np.clip(position, self.calibration.workspace_min, self.calibration.workspace_max)
        if self.orientation_mode == "fixed":
            quaternion = self.calibration.robot_neutral_pose[side][3:]
        elif orientation is None:
            quaternion = self._previous[side][3:]
        else:
            if self.input_mode == "camera_world":
                # The elbow plane is ambiguous across straight extension.
                # Choose its sign continuously rather than introducing a roll flip.
                if np.dot(orientation[:, 2], self._forearm_previous[side][:, 2]) < 0:
                    orientation[:, 1:] *= -1
                self._forearm_previous[side] = orientation.copy()
            tool_rotation = Rotation.from_quat(self.calibration.tool_offset_quat_xyzw[side]).as_matrix()
            output_rotation = self.calibration.rotation @ orientation @ tool_rotation
            quaternion = Rotation.from_matrix(output_rotation).as_quat()
            if np.dot(quaternion, self._previous[side][3:]) < 0:
                quaternion = -quaternion
        pose = np.concatenate((position, quaternion)).astype(np.float32)
        self._previous[side] = pose.copy()
        return ArmTargetPose(side, pose, True)


def _forearm_rotation(points: np.ndarray, *, min_bend_sine=0.) -> np.ndarray | None:
    upper = points[1] - points[0]
    forearm = points[2] - points[1]
    if np.linalg.norm(upper) < 1e-6 or np.linalg.norm(forearm) < 1e-6:
        return None
    x = forearm / np.linalg.norm(forearm)
    z = np.cross(upper, forearm)
    if np.linalg.norm(z) < max(1e-6, min_bend_sine * np.linalg.norm(upper) * np.linalg.norm(forearm)):
        return None
    z /= np.linalg.norm(z)
    y = np.cross(z, x)
    return np.column_stack((x, y, z))
