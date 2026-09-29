"""TRON2A DACH arm kinematics, geometric retargeting, and IK."""

from __future__ import annotations

import json
import xml.etree.ElementTree as XmlET
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import numpy as np
from roboticstoolbox import ERobot, ET, ETS
from scipy.spatial.transform import Rotation


ARM_DOF = 7
ARM_KEYPOINT_ORDER = ("shoulder", "elbow", "wrist")
TRON2A_ARM_JOINTS = {
    "left": (
        "proximal_pitch_L_Joint", "proximal_roll_L_Joint", "proximal_yaw_L_Joint",
        "elbow_L_Joint", "wrist_yaw_L_Joint", "wrist_pitch_L_Joint", "wrist_roll_L_Joint",
    ),
    "right": (
        "proximal_pitch_R_Joint", "proximal_roll_R_Joint", "proximal_yaw_R_Joint",
        "elbow_R_Joint", "wrist_yaw_R_Joint", "wrist_pitch_R_Joint", "wrist_roll_R_Joint",
    ),
}
TRON2A_EE_LINKS = {"left": "grasper_L_Link", "right": "grasper_R_Link"}


@dataclass(frozen=True)
class ArmSpecification:
    side: str
    joint_names: tuple[str, ...]
    ee_link: str
    lower: np.ndarray
    upper: np.ndarray
    velocity: np.ndarray
    robot: ERobot


@dataclass(frozen=True)
class IKResult:
    success: bool
    q: np.ndarray
    position_error: float
    orientation_error: float


class ArmIK(Protocol):
    def solve(self, target_pose: np.ndarray, q_previous: np.ndarray, q_reference: np.ndarray) -> IKResult:
        """Solve one 7-DOF arm target while preserving continuity."""


def load_arm_specification(urdf_path: str | Path, side: str) -> ArmSpecification:
    """Build an exact 7-DOF ETS chain from ``base_Link`` to the grasper link."""
    if side not in TRON2A_ARM_JOINTS:
        raise ValueError(f"Invalid arm side: {side}")
    path = Path(urdf_path)
    if not path.is_file():
        raise FileNotFoundError(f"TRON2A URDF was not found: {path}")

    root = XmlET.parse(path).getroot()
    joints = {joint.attrib["name"]: joint for joint in root.findall("joint")}
    chain = _joint_path(joints, "base_Link", TRON2A_EE_LINKS[side])
    expected = TRON2A_ARM_JOINTS[side]
    movable = tuple(joint.attrib["name"] for joint in chain if joint.attrib["type"] != "fixed")
    if movable != expected:
        raise ValueError(f"Unexpected {side} arm chain: {movable}")

    ets = []
    lower, upper, velocity = [], [], []
    for joint in chain:
        ets.extend(_joint_ets(joint))
        if joint.attrib["type"] != "fixed":
            limit = joint.find("limit")
            if limit is None:
                raise ValueError(f"Joint has no limits: {joint.attrib['name']}")
            lower.append(float(limit.attrib["lower"]))
            upper.append(float(limit.attrib["upper"]))
            velocity.append(float(limit.attrib["velocity"]))

    robot = ERobot(ETS(ets), name=f"tron2a_{side}_arm")
    robot.qlim = np.vstack((lower, upper))
    return ArmSpecification(
        side=side,
        joint_names=expected,
        ee_link=TRON2A_EE_LINKS[side],
        lower=np.asarray(lower, dtype=np.float64),
        upper=np.asarray(upper, dtype=np.float64),
        velocity=np.asarray(velocity, dtype=np.float64),
        robot=robot,
    )


class RoboticsToolboxArmIK:
    """Bounded, continuity-biased numerical IK for one TRON2A arm."""

    def __init__(self, specification: ArmSpecification, *, position_tolerance=0.01, orientation_tolerance=0.15):
        self.specification = specification
        self.position_tolerance = position_tolerance
        self.orientation_tolerance = orientation_tolerance

    def solve(self, target_pose: np.ndarray, q_previous: np.ndarray, q_reference: np.ndarray) -> IKResult:
        target = _pose_matrix(target_pose)
        previous = _joint_vector(q_previous, "q_previous")
        reference = _joint_vector(q_reference, "q_reference")
        previous_error = _pose_error(target, self.specification.robot.fkine(previous).A)
        if previous_error[0] <= self.position_tolerance and previous_error[1] <= self.orientation_tolerance:
            return IKResult(True, previous.astype(np.float32), *previous_error)
        candidates = (np.clip(previous, self.specification.lower, self.specification.upper), np.clip(reference, self.specification.lower, self.specification.upper), 0.5 * (self.specification.lower + self.specification.upper))
        results: list[IKResult] = []
        for seed in candidates:
            solution = self.specification.robot.ikine_LM(
                target, q0=seed, ilimit=100, slimit=4, tol=1e-6,
                joint_limits=True,
            )
            if not solution.success or not np.isfinite(solution.q).all():
                continue
            q = np.asarray(solution.q, dtype=np.float64)
            if q.shape != (ARM_DOF,) or np.any(q < self.specification.lower - 1e-6) or np.any(q > self.specification.upper + 1e-6):
                continue
            position_error, orientation_error = _pose_error(target, self.specification.robot.fkine(q).A)
            if position_error <= self.position_tolerance and orientation_error <= self.orientation_tolerance:
                results.append(IKResult(True, q.astype(np.float32), position_error, orientation_error))
        if not results:
            return IKResult(False, previous.astype(np.float32), float("inf"), float("inf"))
        return min(results, key=lambda item: float(np.linalg.norm(item.q - previous)))


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
        data = json.loads(Path(path).read_text(encoding="utf-8"))
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


class ArmRetargeter:
    """Map shoulder/elbow/wrist observations to calibrated TRON2A EE targets."""

    def __init__(self, calibration: ArmCalibration):
        self.calibration = calibration
        self._previous: dict[str, np.ndarray] = {
            side: calibration.robot_neutral_pose[side].astype(np.float32).copy()
            for side in ("left", "right")
        }

    def update(self, side: str, keypoints: np.ndarray, valid: bool) -> tuple[np.ndarray, bool]:
        if side not in ("left", "right"):
            raise ValueError(f"Invalid arm side: {side}")
        points = np.asarray(keypoints, dtype=np.float64)
        if not valid or points.shape != (3, 3) or not np.isfinite(points).all():
            return self._previous[side].copy(), False
        orientation = _forearm_rotation(points)
        if orientation is None:
            return self._previous[side].copy(), False
        wrist = points[2]
        position = self.calibration.robot_neutral_pose[side][:3] + self.calibration.translation + self.calibration.rotation @ (self.calibration.scale * (wrist - self.calibration.human_neutral_wrist[side]))
        position = np.clip(position, self.calibration.workspace_min, self.calibration.workspace_max)
        tool_rotation = Rotation.from_quat(self.calibration.tool_offset_quat_xyzw[side]).as_matrix()
        output_rotation = self.calibration.rotation @ orientation @ tool_rotation
        pose = np.concatenate((position, Rotation.from_matrix(output_rotation).as_quat())).astype(np.float32)
        self._previous[side] = pose.copy()
        return pose, True


class ArmSafetyController:
    """Apply joint limits, velocity/acceleration limits, timeout, and safe pose."""

    def __init__(self, specification: ArmSpecification, safe_q: np.ndarray, max_acceleration: float, tracking_timeout: float):
        self.specification = specification
        self.safe_q = np.clip(_joint_vector(safe_q, "safe_q"), specification.lower, specification.upper).astype(np.float32)
        self.max_acceleration = float(max_acceleration)
        self.tracking_timeout = float(tracking_timeout)
        self.q = self.safe_q.copy()
        self.velocity = np.zeros(ARM_DOF, dtype=np.float32)
        self._last_timestamp: float | None = None
        self._last_valid_timestamp: float | None = None

    def update(self, target_q: np.ndarray | None, valid: bool, timestamp: float) -> tuple[np.ndarray, bool]:
        timestamp = float(timestamp)
        dt = 1.0 / 30.0 if self._last_timestamp is None else max(timestamp - self._last_timestamp, 1.0 / 240.0)
        self._last_timestamp = timestamp
        if valid:
            desired = np.clip(_joint_vector(target_q, "target_q"), self.specification.lower, self.specification.upper)
            self._last_valid_timestamp = timestamp
        elif self._last_valid_timestamp is not None and timestamp - self._last_valid_timestamp >= self.tracking_timeout:
            desired = self.safe_q
        else:
            desired = self.q
        target_velocity = np.clip((desired - self.q) / dt, -self.specification.velocity, self.specification.velocity)
        maximum_velocity_change = self.max_acceleration * dt
        self.velocity += np.clip(target_velocity - self.velocity, -maximum_velocity_change, maximum_velocity_change)
        self.q = np.clip(self.q + self.velocity * dt, self.specification.lower, self.specification.upper).astype(np.float32)
        return self.q.copy(), bool(valid)


def _joint_path(joints: dict[str, XmlET.Element], start_link: str, end_link: str) -> list[XmlET.Element]:
    by_parent: dict[str, list[XmlET.Element]] = {}
    for joint in joints.values():
        by_parent.setdefault(joint.find("parent").attrib["link"], []).append(joint)
    queue = [(start_link, [])]
    while queue:
        link, path = queue.pop(0)
        if link == end_link:
            return path
        for joint in by_parent.get(link, []):
            queue.append((joint.find("child").attrib["link"], path + [joint]))
    raise ValueError(f"No URDF path from {start_link} to {end_link}")


def _joint_ets(joint: XmlET.Element) -> list[ET]:
    origin = joint.find("origin")
    xyz = np.fromstring((origin.attrib.get("xyz", "0 0 0") if origin is not None else "0 0 0"), sep=" ")
    rpy = np.fromstring((origin.attrib.get("rpy", "0 0 0") if origin is not None else "0 0 0"), sep=" ")
    transform = np.eye(4)
    transform[:3, :3] = Rotation.from_euler("xyz", rpy).as_matrix()
    transform[:3, 3] = xyz
    result = [ET.SE3(transform)]
    if joint.attrib["type"] == "fixed":
        return result
    axis = np.fromstring(joint.find("axis").attrib["xyz"], sep=" ")
    axis /= np.linalg.norm(axis)
    align = _align_z(axis)
    result.extend((ET.SE3(align), ET.Rz(), ET.SE3(np.linalg.inv(align))))
    return result


def _align_z(axis: np.ndarray) -> np.ndarray:
    z = np.asarray(axis, dtype=np.float64)
    candidate = np.array([1.0, 0.0, 0.0]) if abs(z[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    x = np.cross(candidate, z)
    x /= np.linalg.norm(x)
    y = np.cross(z, x)
    matrix = np.eye(4)
    matrix[:3, :3] = np.column_stack((x, y, z))
    return matrix


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


def _pose_matrix(pose: np.ndarray) -> np.ndarray:
    value = np.asarray(pose, dtype=np.float64)
    if value.shape == (4, 4):
        return value
    if value.shape != (7,) or not np.isfinite(value).all():
        raise ValueError("Target pose must have shape (7,) as xyz + xyzw")
    matrix = np.eye(4)
    matrix[:3, :3] = Rotation.from_quat(value[3:]).as_matrix()
    matrix[:3, 3] = value[:3]
    return matrix


def _pose_error(target: np.ndarray, actual: np.ndarray) -> tuple[float, float]:
    position = float(np.linalg.norm(target[:3, 3] - actual[:3, 3]))
    delta = target[:3, :3].T @ actual[:3, :3]
    orientation = float(np.arccos(np.clip((np.trace(delta) - 1.0) * 0.5, -1.0, 1.0)))
    return position, orientation


def _joint_vector(value: np.ndarray, name: str) -> np.ndarray:
    vector = np.asarray(value, dtype=np.float64)
    if vector.shape != (ARM_DOF,) or not np.isfinite(vector).all():
        raise ValueError(f"{name} must have shape ({ARM_DOF},)")
    return vector
