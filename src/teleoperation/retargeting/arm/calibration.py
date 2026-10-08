from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation
ARM_DOF = 7
from teleoperation.data.calibration import read_calibration
@dataclass(frozen=True)
class ArmCalibration:
    rotation: np.ndarray
    translation: np.ndarray
    scale: np.ndarray
    human_neutral_wrist: dict[str, np.ndarray]
    robot_neutral_pose: dict[str, np.ndarray]
    tool_offset_quat_xyzw: dict[str, np.ndarray]
    safe_q: dict[str, np.ndarray]
    workspace_min: np.ndarray
    workspace_max: np.ndarray
    max_acceleration: float
    tracking_timeout: float
    hand_mount_pose: dict[str, np.ndarray]

    @classmethod
    def load(cls, path: str | Path) -> "ArmCalibration":
        data = read_calibration(path)
        def vector(name, size):
            value = np.asarray(data[name], dtype=np.float64)
            if value.shape != (size,):
                raise ValueError(f"calibration.{name} must have shape ({size},)")
            return value
        rotation = np.asarray(data["rotation"], dtype=np.float64)
        if rotation.shape != (3, 3) or not np.isfinite(rotation).all():
            raise ValueError("calibration.rotation must have shape (3, 3)")
        scale = np.asarray(data["scale"], dtype=np.float64)
        if scale.ndim == 0:
            scale = np.full(3, float(scale))
        if scale.shape != (3,) or np.any(scale <= 0):
            raise ValueError("calibration.scale must be positive scalar or shape (3,)")
        sides = ("left", "right")
        return cls(
            rotation=rotation, translation=vector("translation", 3), scale=scale,
            human_neutral_wrist={side: np.asarray(data["human_neutral_wrist"][side], dtype=np.float64) for side in sides},
            robot_neutral_pose={side: np.asarray(data["robot_neutral_pose"][side], dtype=np.float64) for side in sides},
            tool_offset_quat_xyzw={side: np.asarray(data["tool_offset_quat_xyzw"][side], dtype=np.float64) for side in sides},
            safe_q={side: np.asarray(data["safe_q"][side], dtype=np.float64) for side in sides},
            workspace_min=vector("workspace_min", 3), workspace_max=vector("workspace_max", 3),
            max_acceleration=float(data.get("max_acceleration", 20.0)),
            tracking_timeout=float(data.get("tracking_timeout", 0.5)),
            hand_mount_pose={side: np.asarray(data.get("hand_mount_pose", {}).get(side, [0, 0, 0, 0, 0, 0, 1]), dtype=np.float64) for side in sides},
        )
