from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation
ARM_DOF = 7
import xml.etree.ElementTree as XmlET
from typing import Protocol
from roboticstoolbox import ERobot, ET, ETS
from teleoperation.contracts.arm import ArmIKResult, ArmTargetPose
from teleoperation.robots.specification import TRON2A_ARM_JOINTS, TRON2A_EE_LINKS
@dataclass(frozen=True)
class ArmSpecification:
    side: str
    joint_names: tuple[str, ...]
    ee_link: str
    lower: np.ndarray
    upper: np.ndarray
    velocity: np.ndarray
    robot: ERobot


class ArmIK(Protocol):
    def solve(self, target_pose: ArmTargetPose, q_previous: np.ndarray, q_reference: np.ndarray) -> ArmIKResult:
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

    def solve(self, target_pose: ArmTargetPose, q_previous: np.ndarray, q_reference: np.ndarray) -> ArmIKResult:
        if not isinstance(target_pose, ArmTargetPose): raise TypeError("IK requires ArmTargetPose")
        if target_pose.robot != "tron2a" or target_pose.side != self.specification.side: raise ValueError("IK target robot/side mismatch")
        target = _pose_matrix(target_pose.pose)
        previous = _joint_vector(q_previous, "q_previous")
        reference = _joint_vector(q_reference, "q_reference")
        previous_error = _pose_error(target, self.specification.robot.fkine(previous).A)
        if previous_error[0] <= self.position_tolerance and previous_error[1] <= self.orientation_tolerance:
            return ArmIKResult(True, previous.astype(np.float32), *previous_error, side=self.specification.side)
        candidates = (np.clip(previous, self.specification.lower, self.specification.upper), np.clip(reference, self.specification.lower, self.specification.upper), 0.5 * (self.specification.lower + self.specification.upper))
        results: list[ArmIKResult] = []
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
                results.append(ArmIKResult(True, q.astype(np.float32), position_error, orientation_error, side=self.specification.side))
        if not results:
            return ArmIKResult(False, previous.astype(np.float32), float("inf"), float("inf"), side=self.specification.side)
        return min(results, key=lambda item: float(np.linalg.norm(item.q - previous)))


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


def target_pose_from_matrix(matrix, side):
    value = np.asarray(matrix)
    if value.shape != (4, 4) or not np.isfinite(value).all(): raise ValueError("Target matrix must be finite (4,4)")
    return ArmTargetPose(side, np.concatenate((value[:3, 3], Rotation.from_matrix(value[:3, :3]).as_quat())))
