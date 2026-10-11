"""PyBullet scene and controller for fixed 48D TRON2A/L21 commands."""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np

from teleoperation.robots.specification import TRON2A_ARM_JOINTS, TRON2A_EE_LINKS
from teleoperation.contracts.commands import Tron2AL21Command
from .urdf_loader import load_urdf


class Tron2ADualArmWorld:
    """Own the PyBullet bodies and resolve all controlled joints by name."""

    def __init__(self, urdf_path: str | Path, *, hand_mount_pose=None, gui=True,
                 time_step=1.0 / 240.0, display="full", attach_hands=True, safe_q=None,
                 hand_reference_only=False):
        import pybullet as pybullet
        self.p = pybullet
        self.client = -1
        self._temporary = tempfile.TemporaryDirectory(prefix="tron2a_pybullet_")
        self.time_step = float(time_step)
        if not np.isfinite(time_step) or time_step <= 0:
            self._temporary.cleanup()
            raise ValueError("Simulation time step must be finite and positive")
        try:
            prepared = _prepare_urdf(urdf_path, Path(self._temporary.name), display=display)
            self.client = self.p.connect(self.p.GUI if gui else self.p.DIRECT)
            if self.client < 0:
                raise RuntimeError("PyBullet connection failed")
            self.p.setTimeStep(time_step, physicsClientId=self.client)
            self.p.setGravity(0, 0, -9.81, physicsClientId=self.client)
            self.robot = self.p.loadURDF(str(prepared), useFixedBase=True,
                                        flags=self.p.URDF_USE_INERTIA_FROM_FILE, physicsClientId=self.client)
            self.joint_indices, self.link_indices = _joint_and_link_maps(self.p, self.robot, self.client)
            self.arm_indices = {side: _require_indices(self.joint_indices, TRON2A_ARM_JOINTS[side]) for side in ("left", "right")}
            self.ee_indices = {side: _require_link(self.link_indices, TRON2A_EE_LINKS[side]) for side in ("left", "right")}
            self.joint_info = {index: self.p.getJointInfo(self.robot, index, physicsClientId=self.client)
                               for index in self.joint_indices.values()}
            controlled = set(sum(self.arm_indices.values(), []))
            self.fixed_reference = {}
            for index, info in self.joint_info.items():
                if info[2] == self.p.JOINT_FIXED:
                    continue
                value = float(np.clip(0., info[8], info[9])) if info[8] <= info[9] else 0.
                if index not in controlled:
                    self.fixed_reference[index] = value
                self.p.resetJointState(self.robot, index, value, physicsClientId=self.client)
            if display == "simplified":
                # Preview only: primitive proxies cannot represent real contact.
                for index in range(-1, self.p.getNumJoints(self.robot, physicsClientId=self.client)):
                    self.p.setCollisionFilterGroupMask(self.robot, index, 0, 0, physicsClientId=self.client)
            initial = safe_q or {side: np.zeros(7) for side in ("left", "right")}
            for side, indices in self.arm_indices.items():
                q = self._checked_arm(side, initial[side])
                for index, angle in zip(indices, q):
                    self.p.resetJointState(self.robot, index, float(angle), physicsClientId=self.client)
            self.apply_arms(initial)
            self._hold_other_joints()
            self.hand_mount_pose = hand_mount_pose or {}
            self.hand_reference_only = hand_reference_only
            self.hand_bodies, self.hand_indices = self._attach_l21_hands() if attach_hands else ({}, {})
            if hand_reference_only:
                self._position_reference_hands()
            if gui:
                self.p.resetDebugVisualizerCamera(2.4, 50, -20, [0, 0, 0], physicsClientId=self.client)
        except BaseException:
            self.close()
            raise

    def _checked_arm(self, side, value):
        q = np.asarray(value, dtype=float)
        if q.shape != (7,) or not np.isfinite(q).all():
            raise ValueError(f"{side} arm command must be finite q[7]")
        for index, angle in zip(self.arm_indices[side], q):
            info = self.joint_info[index]
            if not info[8] - 1e-7 <= angle <= info[9] + 1e-7:
                raise ValueError(f"{side} arm command exceeds URDF limits")
        return q

    def apply_arms(self, joints):
        checked = {side: self._checked_arm(side, joints[side]) for side in ("left", "right")}
        for side, q in checked.items():
            for index, angle in zip(self.arm_indices[side], q):
                info = self.joint_info[index]
                self.p.setJointMotorControl2(self.robot, index, self.p.POSITION_CONTROL,
                    targetPosition=float(angle), force=max(float(info[10]), 1.),
                    maxVelocity=max(float(info[11]), 0.01), positionGain=0.3, velocityGain=1.,
                    physicsClientId=self.client)

    def _hold_other_joints(self):
        for index, value in self.fixed_reference.items():
            self.p.setJointMotorControl2(self.robot, index, self.p.POSITION_CONTROL,
                targetPosition=value, force=5000., positionGain=0.5, velocityGain=1., physicsClientId=self.client)

    def actual_state(self):
        arms = {}
        for side, indices in self.arm_indices.items():
            states = self.p.getJointStates(self.robot, indices, physicsClientId=self.client)
            ee = self.p.getLinkState(self.robot, self.ee_indices[side], computeForwardKinematics=True, physicsClientId=self.client)
            arms[side] = dict(q=np.array([s[0] for s in states]), velocity=np.array([s[1] for s in states]),
                              pose=np.array((*ee[4], *ee[5])))
        drift = {self.joint_info[index][1].decode(): float(self.p.getJointState(self.robot, index, physicsClientId=self.client)[0] - value)
                 for index, value in self.fixed_reference.items()}
        return dict(arms=arms, noncontrolled_joint_drift=drift)

    def is_connected(self):
        return self.client >= 0 and bool(self.p.isConnected(self.client))

    def _attach_l21_hands(self):
        from teleoperation.retargeting.hand.config import L21
        hand_bodies, hand_indices = {}, {}
        for side, urdf in (("left", L21.left_urdf), ("right", L21.right_urdf)):
            mount = np.asarray(self.hand_mount_pose.get(side, [0, 0, 0, 0, 0, 0, 1]), dtype=np.float32)
            if mount.shape != (7,):
                raise ValueError(f"{side} hand mount pose must have shape (7,)")
            body = load_urdf(urdf, useFixedBase=self.hand_reference_only, physicsClientId=self.client)
            ee = self.p.getLinkState(self.robot, self.ee_indices[side], computeForwardKinematics=True, physicsClientId=self.client)
            position, orientation = self.p.multiplyTransforms(ee[4], ee[5], mount[:3].tolist(), mount[3:].tolist())
            self.p.resetBasePositionAndOrientation(body, position, orientation, physicsClientId=self.client)
            if not self.hand_reference_only:
                self.p.createConstraint(
                    self.robot, self.ee_indices[side], body, -1, self.p.JOINT_FIXED,
                    [0, 0, 0], mount[:3].tolist(), [0, 0, 0],
                    parentFrameOrientation=mount[3:].tolist(), childFrameOrientation=[0, 0, 0, 1],
                    physicsClientId=self.client,
                )
            else:
                for index in range(-1, self.p.getNumJoints(body, physicsClientId=self.client)):
                    self.p.setCollisionFilterGroupMask(body, index, 0, 0, physicsClientId=self.client)
            joint_map, _ = _joint_and_link_maps(self.p, body, self.client)
            hand_names = tuple(L21.hand_kinematics_config()["joints_name"][1:18])
            hand_bodies[side] = body
            hand_indices[side] = _require_indices(joint_map, hand_names)
            self.p.setJointMotorControlArray(body, hand_indices[side], self.p.POSITION_CONTROL, targetPositions=[0.] * 17, physicsClientId=self.client)
        return hand_bodies, hand_indices

    def _position_reference_hands(self):
        for side, body in self.hand_bodies.items():
            mount = np.asarray(self.hand_mount_pose.get(side, [0, 0, 0, 0, 0, 0, 1]))
            ee = self.p.getLinkState(self.robot, self.ee_indices[side], computeForwardKinematics=True, physicsClientId=self.client)
            link_pose = self.p.multiplyTransforms(ee[4], ee[5], mount[:3], mount[3:])
            inertia = self.p.getDynamicsInfo(body, -1, physicsClientId=self.client)
            position, orientation = self.p.multiplyTransforms(*link_pose, inertia[3], inertia[4])
            self.p.resetBasePositionAndOrientation(body, position, orientation, physicsClientId=self.client)
            for index in self.hand_indices[side]:
                self.p.resetJointState(body, index, 0., targetVelocity=0., physicsClientId=self.client)

    def apply(self, command: Tron2AL21Command) -> None:
        if not isinstance(command, Tron2AL21Command):
            raise TypeError("TRON2A execution requires Tron2AL21Command")
        self.apply_arms({side: getattr(command, f"{side}_arm") for side in ("left", "right")})
        for side, body in self.hand_bodies.items():
            self.p.setJointMotorControlArray(body, self.hand_indices[side], self.p.POSITION_CONTROL,
                targetPositions=getattr(command, f"{side}_hand").tolist(), physicsClientId=self.client)

    def step(self) -> None:
        self.p.stepSimulation(physicsClientId=self.client)
        if self.hand_reference_only:
            self._position_reference_hands()

    def close(self) -> None:
        try:
            if self.is_connected():
                self.p.disconnect(physicsClientId=self.client)
        finally:
            self._temporary.cleanup()


def _prepare_urdf(urdf_path: str | Path, directory: Path, *, display="full") -> Path:
    import xml.etree.ElementTree as ET
    from scipy.spatial.transform import Rotation
    if display not in ("full", "simplified"):
        raise ValueError("Display must be full or simplified")
    source = Path(urdf_path).resolve()
    if not source.is_file():
        raise FileNotFoundError(f"TRON2A URDF was not found: {source}")
    tree = ET.parse(source)
    root = tree.getroot()
    missing = []
    for mesh in root.findall(".//mesh"):
        filename = mesh.attrib["filename"]
        if filename.startswith("package://"):
            suffix = filename.split("/meshes/", 1)
            path = source.parent.parent / "meshes" / suffix[1] if len(suffix) == 2 else source.parent / filename[10:]
        else:
            path = Path(filename)
            if not path.is_absolute():
                path = source.parent / path
        if display == "full":
            if not path.is_file():
                missing.append(str(path))
            mesh.set("filename", path.resolve().as_posix())
    if missing:
        raise FileNotFoundError("TRON2A mesh resources missing; install them or explicitly use --display simplified: " + "; ".join(sorted(set(missing))))
    if display == "simplified":
        for link in root.findall("link"):
            # Replace mesh proxies only; keep all original joints and inertias.
            for element in link.findall("visual") + link.findall("collision"):
                geometry = element.find("geometry")
                if geometry is not None and geometry.find("mesh") is not None:
                    geometry.remove(geometry.find("mesh"))
                    ET.SubElement(geometry, "sphere", radius="0.035")
            # Rods between link origin and child joint origins make the chain visible.
            for joint in root.findall("joint"):
                if joint.find("parent").get("link") != link.get("name"):
                    continue
                origin = joint.find("origin")
                xyz = np.fromstring(origin.get("xyz", "0 0 0") if origin is not None else "0 0 0", sep=" ")
                length = np.linalg.norm(xyz)
                if length < 0.02:
                    continue
                visual = ET.SubElement(link, "visual")
                rotation, _ = Rotation.align_vectors([xyz / length], [[0, 0, 1]])
                ET.SubElement(visual, "origin", xyz=" ".join(map(str, xyz / 2)), rpy=" ".join(map(str, rotation.as_euler("xyz"))))
                ET.SubElement(ET.SubElement(visual, "geometry"), "cylinder", radius="0.015", length=str(length))
                material = ET.SubElement(visual, "material", name="preview")
                ET.SubElement(material, "color", rgba="0.3 0.65 0.85 1")
    target = directory / "robot_pybullet.urdf"
    tree.write(target, encoding="utf-8", xml_declaration=True)
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
