"""PyBullet scene and controller for fixed 48D TRON2A/L21 commands."""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np

from teleoperation.robots.specification import TRON2A_ARM_JOINTS, TRON2A_EE_LINKS
from teleoperation.contracts.commands import Tron2AL21Command
from teleoperation.retargeting.hand.config import L21


class Tron2ADualArmWorld:
    """Own the PyBullet bodies and resolve all controlled joints by name."""

    def __init__(self, urdf_path: str | Path, *, hand_mount_pose: dict[str, np.ndarray] | None = None, gui: bool = True, time_step: float = 1.0 / 240.0):
        import pybullet as pybullet

        self.p = pybullet
        self.client = pybullet.connect(pybullet.GUI if gui else pybullet.DIRECT)
        self.p.setTimeStep(time_step, physicsClientId=self.client)
        self.p.setGravity(0, 0, -9.81, physicsClientId=self.client)
        self._temporary = tempfile.TemporaryDirectory(prefix="tron2a_pybullet_")
        prepared_urdf = _prepare_urdf(urdf_path, Path(self._temporary.name))
        self.robot = self.p.loadURDF(str(prepared_urdf), useFixedBase=True, physicsClientId=self.client)
        self.joint_indices, self.link_indices = _joint_and_link_maps(self.p, self.robot, self.client)
        self.arm_indices = {
            side: _require_indices(self.joint_indices, TRON2A_ARM_JOINTS[side])
            for side in ("left", "right")
        }
        self.ee_indices = {side: _require_link(self.link_indices, TRON2A_EE_LINKS[side]) for side in ("left", "right")}
        self.hand_mount_pose = hand_mount_pose or {}
        self.hand_bodies, self.hand_indices = self._attach_l21_hands()

    def _attach_l21_hands(self):
        hand_bodies, hand_indices = {}, {}
        for side, urdf in (("left", L21.left_urdf), ("right", L21.right_urdf)):
            mount = np.asarray(self.hand_mount_pose.get(side, [0, 0, 0, 0, 0, 0, 1]), dtype=np.float32)
            if mount.shape != (7,):
                raise ValueError(f"{side} hand mount pose must have shape (7,)")
            body = self.p.loadURDF(str(urdf), useFixedBase=False, physicsClientId=self.client)
            self.p.createConstraint(
                self.robot, self.ee_indices[side], body, -1, self.p.JOINT_FIXED,
                [0, 0, 0],
                mount[:3].tolist(),
                [0, 0, 0],
                parentFrameOrientation=mount[3:].tolist(),
                childFrameOrientation=[0, 0, 0, 1],
                physicsClientId=self.client,
            )
            joint_map, _ = _joint_and_link_maps(self.p, body, self.client)
            hand_names = tuple(L21.hand_kinematics_config()["joints_name"][1:18])
            hand_bodies[side] = body
            hand_indices[side] = _require_indices(joint_map, hand_names)
        return hand_bodies, hand_indices

    def apply(self, command: Tron2AL21Command) -> None:
        if not isinstance(command, Tron2AL21Command):
            raise TypeError("TRON2A execution requires Tron2AL21Command")
        for side in ("left", "right"):
            self.p.setJointMotorControlArray(
                self.robot, self.arm_indices[side], self.p.POSITION_CONTROL,
                targetPositions=getattr(command, f"{side}_arm").tolist(),
                physicsClientId=self.client,
            )
            self.p.setJointMotorControlArray(
                self.hand_bodies[side], self.hand_indices[side], self.p.POSITION_CONTROL,
                targetPositions=getattr(command, f"{side}_hand").tolist(),
                physicsClientId=self.client,
            )

    def step(self) -> None:
        self.p.stepSimulation(physicsClientId=self.client)

    def close(self) -> None:
        if self.p.isConnected(self.client):
            self.p.disconnect(physicsClientId=self.client)
        self._temporary.cleanup()


def _prepare_urdf(urdf_path: str | Path, directory: Path) -> Path:
    source = Path(urdf_path).resolve()
    mesh_directory = source.parent.parent / "meshes"
    if not source.is_file() or not mesh_directory.is_dir():
        raise FileNotFoundError(f"TRON2A URDF or mesh directory was not found: {source}")
    text = source.read_text(encoding="utf-8")
    text = text.replace("package://robot_description/tron2/DACH_TRON2A/meshes/", mesh_directory.as_posix() + "/")
    target = directory / "robot_pybullet.urdf"
    target.write_text(text, encoding="utf-8")
    return target


def _joint_and_link_maps(pybullet, body: int, client: int):
    joints, links = {}, {"base_Link": -1}
    for index in range(pybullet.getNumJoints(body, physicsClientId=client)):
        info = pybullet.getJointInfo(body, index, physicsClientId=client)
        joints[info[1].decode("utf-8")] = index
        links[info[12].decode("utf-8")] = index
    return joints, links


def _require_indices(mapping: dict[str, int], names: tuple[str, ...]) -> list[int]:
    missing = [name for name in names if name not in mapping]
    if missing:
        raise ValueError(f"URDF is missing controlled joints: {', '.join(missing)}")
    return [mapping[name] for name in names]


def _require_link(mapping: dict[str, int], name: str) -> int:
    if name not in mapping:
        raise ValueError(f"URDF is missing end-effector link: {name}")
    return mapping[name]
