"""URDF hand subtree metadata and reference-pose transforms (NumPy only)."""
from dataclasses import dataclass
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np


@dataclass(frozen=True)
class HandJoint:
    name: str
    kind: str
    parent: str
    child: str
    origin: np.ndarray
    axis: np.ndarray


def _origin(node):
    values = np.zeros(6)
    if node is not None:
        values[:3] = np.fromstring(node.get("xyz", "0 0 0"), sep=" ")
        values[3:] = np.fromstring(node.get("rpy", "0 0 0"), sep=" ")
    if not np.isfinite(values).all():
        raise ValueError("Nonfinite URDF hand joint origin")
    roll, pitch, yaw = values[3:]
    cr, sr, cp, sp, cy, sy = np.cos(roll), np.sin(roll), np.cos(pitch), np.sin(pitch), np.cos(yaw), np.sin(yaw)
    matrix = np.eye(4)
    matrix[:3, :3] = [[cy*cp, cy*sp*sr-sy*cr, cy*sp*cr+sy*sr],
                      [sy*cp, sy*sp*sr+cy*cr, sy*sp*cr-cy*sr], [-sp, cp*sr, cp*cr]]
    matrix[:3, 3] = values[:3]
    return matrix


def read_hand_tree(path: Path, root_link: str) -> tuple[HandJoint, ...]:
    tree = ET.parse(path).getroot()
    if root_link not in {link.get("name") for link in tree.findall("link")}:
        raise ValueError(f"Missing hand root frame: {root_link}")
    links, ordered = {root_link}, []
    remaining = list(tree.findall("joint"))
    while True:
        selected = [j for j in remaining if j.find("parent").get("link") in links]
        if not selected:
            break
        for joint in selected:
            remaining.remove(joint)
            kind = joint.get("type")
            if kind not in ("fixed", "revolute", "continuous"):
                raise ValueError(f"Unsupported angular hand joint: {joint.get('name')}/{kind}")
            child = joint.find("child").get("link")
            if child in links:
                raise ValueError("Cyclic or multiply-parented hand subtree")
            axis_node = joint.find("axis")
            axis = np.fromstring("1 0 0" if axis_node is None else axis_node.get("xyz", "1 0 0"), sep=" ")
            if axis.shape != (3,) or not np.isfinite(axis).all() or (kind != "fixed" and np.linalg.norm(axis) == 0):
                raise ValueError("Invalid hand joint axis")
            if kind != "fixed":
                axis /= np.linalg.norm(axis)
            ordered.append(HandJoint(joint.get("name"), kind, joint.find("parent").get("link"),
                                     child, _origin(joint.find("origin")), axis))
            links.add(child)
    return tuple(ordered)


def reference_positions(tree, root_link, named_positions, frame_names):
    transforms = {root_link: np.eye(4)}
    for joint in tree:
        motion = np.eye(4)
        if joint.kind != "fixed":
            angle = named_positions[joint.name]
            x, y, z = joint.axis
            cross = np.array([[0., -z, y], [z, 0., -x], [-y, x, 0.]])
            motion[:3, :3] = np.eye(3) + np.sin(angle)*cross + (1-np.cos(angle))*(cross @ cross)
        transforms[joint.child] = transforms[joint.parent] @ joint.origin @ motion
    if any(frame not in transforms for frame in frame_names):
        raise ValueError("Task/reference frame is outside the target hand subtree")
    return np.array([transforms[name][:3, 3] for name in frame_names])
