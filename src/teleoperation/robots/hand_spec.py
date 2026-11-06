"""Target hand joint semantics, effective limits and asset identity.

    This module reads robot descriptions, not human landmarks. Independent
    coordinates and expanded movable joints are distinct spaces. Retargeting,
    training and execution use the same specification for a selected target.
"""
from dataclasses import dataclass, field
import hashlib
import json
from pathlib import Path
import re
import xml.etree.ElementTree as ET

import numpy as np

from teleoperation.paths import L21_ROOT
from .l21 import L21_JOINT_ORDER


from teleoperation.contracts.constants import HAND_MODELS


@dataclass(frozen=True)
class HandSpec:
    hand_model: str
    side: str
    urdf_path: Path
    root_link: str
    joint_names: tuple[str, ...]
    independent_joint_names: tuple[str, ...]
    joint_limits: tuple[tuple[float, float], ...]
    joint_velocities: tuple[float, ...]
    mimic_rules: tuple[tuple[str, str, float, float], ...] = ()
    tip_frames: tuple[str, ...] = ()
    human_tip_indices: tuple[int, ...] = ()
    palm_frames: tuple[str, ...] = ()
    source_asset_hash: str = ""
    _relations: tuple = field(init=False, repr=False)
    _bounds: tuple = field(init=False, repr=False)
    _speeds: tuple = field(init=False, repr=False)
    spec_hash: str = field(init=False)
    _palm_length: float | None = field(init=False, repr=False, default=None)
    _native_to_canonical: tuple | None = field(init=False, repr=False, default=None)

    def __post_init__(self):
        if not isinstance(self.hand_model, str) or not re.fullmatch(r"[a-z0-9][a-z0-9_-]*", self.hand_model):
            raise ValueError("Hand model identifiers must be safe directory components")
        if self.side not in ("left", "right") or not self.root_link:
            raise ValueError("Hand specification requires a target side and root link")
        for name in ("joint_names", "independent_joint_names", "joint_velocities", "tip_frames",
                     "human_tip_indices", "palm_frames"):
            object.__setattr__(self, name, tuple(getattr(self, name)))
        object.__setattr__(self, "joint_limits", tuple(tuple(pair) for pair in self.joint_limits))
        object.__setattr__(self, "mimic_rules", tuple(tuple(rule) for rule in self.mimic_rules))
        object.__setattr__(self, "urdf_path", Path(self.urdf_path).resolve())
        names, independent = self.joint_names, self.independent_joint_names
        for sequence in (names, independent):
            if not sequence or len(set(sequence)) != len(sequence) or any(not isinstance(n, str) or not n for n in sequence):
                raise ValueError("Hand joint names must be nonempty and unique")
        if not set(independent) <= set(names):
            raise ValueError("Independent coordinates must name movable hand joints")
        limits = np.asarray(self.joint_limits, dtype=float)
        speeds = np.asarray(self.joint_velocities, dtype=float)
        if limits.shape != (len(names), 2) or not np.isfinite(limits).all() or np.any(limits[:, 0] > limits[:, 1]):
            raise ValueError("Every movable joint requires finite ordered position limits")
        if speeds.shape != (len(names),) or not np.isfinite(speeds).all() or np.any(speeds <= 0):
            raise ValueError("Every movable joint requires a positive finite speed limit")
        if len(self.tip_frames) != len(self.human_tip_indices) or len(set(self.tip_frames)) != len(self.tip_frames):
            raise ValueError("Task frames must have unique matching human tip indices")
        if any(not isinstance(i, (int, np.integer)) or not 0 <= i < 25 for i in self.human_tip_indices):
            raise ValueError("Human tip indices must address the common hand25 topology")
        rules = {}
        for rule in self.mimic_rules:
            if len(rule) != 4:
                raise ValueError("Mimic rules require follower, source, multiplier and offset")
            follower, source, multiplier, offset = rule
            if follower in rules or follower not in names or source not in names or follower in independent:
                raise ValueError("Mimic rules must name distinct dependent joints and existing sources")
            if not np.isfinite([multiplier, offset]).all():
                raise ValueError("Mimic coefficients must be finite")
            rules[follower] = (source, float(multiplier), float(offset))
        if set(rules) != set(names) - set(independent):
            raise ValueError("Every dependent joint requires exactly one mimic rule")

        def resolve(name, visiting=()):
            if name in visiting:
                raise ValueError("Cyclic mimic joint relations")
            if name in independent:
                return name, 1., 0.
            source, multiplier, offset = rules[name]
            driver, a, b = resolve(source, (*visiting, name))
            return driver, multiplier * a, multiplier * b + offset

        relations = tuple(resolve(name) for name in names)
        bounds = np.array([limits[names.index(n)] for n in independent])
        effective_speeds = np.array([speeds[names.index(n)] for n in independent])
        for index, (driver, multiplier, offset) in enumerate(relations):
            driver_index = independent.index(driver)
            low, high = limits[index]
            if multiplier == 0:
                if not low <= offset <= high:
                    raise ValueError("A constant mimic joint is outside its limits")
                continue
            projected = sorted(((low - offset) / multiplier, (high - offset) / multiplier))
            bounds[driver_index, 0] = max(bounds[driver_index, 0], projected[0])
            bounds[driver_index, 1] = min(bounds[driver_index, 1], projected[1])
            effective_speeds[driver_index] = min(effective_speeds[driver_index], speeds[index] / abs(multiplier))
        if np.any(bounds[:, 0] > bounds[:, 1]):
            raise ValueError("Mimic joint position constraints are infeasible")
        object.__setattr__(self, "_relations", relations)
        object.__setattr__(self, "_bounds", tuple(map(tuple, bounds)))
        object.__setattr__(self, "_speeds", tuple(effective_speeds))
        identity = {name: getattr(self, name) for name in (
            "hand_model", "side", "root_link", "joint_names", "independent_joint_names",
            "joint_limits", "joint_velocities", "mimic_rules", "tip_frames",
            "human_tip_indices", "palm_frames", "source_asset_hash",
        )}
        encoded = json.dumps(identity, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
        object.__setattr__(self, "spec_hash", hashlib.sha256(encoded + b"\0" + self.urdf_path.read_bytes()).hexdigest())
        if self.palm_frames:
            if len(self.palm_frames) != 3 or len(set(self.palm_frames)) != 3 or self.palm_frames[0] != self.root_link:
                raise ValueError("Palm reference must name root, index MCP and middle MCP frames")
            from .hand_kinematics import read_hand_tree, reference_positions
            from teleoperation.retargeting.hand.coordinates import build_palm_basis, OPERATOR_TO_CANONICAL

            tree = read_hand_tree(self.urdf_path, self.root_link)
            positions = reference_positions(tree, self.root_link, self.expand(self.q_reference), self.palm_frames)
            reference_positions(tree, self.root_link, self.expand(self.q_reference), self.tip_frames)
            length = float(np.linalg.norm(positions[2] - positions[0]))
            if length <= 1e-6:
                raise ValueError("Target reference palm length must exceed 1e-6 metres")
            transform = build_palm_basis(positions, (0, 1, 2)) @ OPERATOR_TO_CANONICAL[self.side]
            object.__setattr__(self, "_palm_length", length)
            object.__setattr__(self, "_native_to_canonical", tuple(map(tuple, transform)))

    @property
    def dof(self) -> int:
        return len(self.independent_joint_names)

    @property
    def key(self):
        return self.hand_model, self.side

    @property
    def bounds(self) -> np.ndarray:
        return np.array(self._bounds, dtype=float)

    @property
    def velocity_limits(self) -> np.ndarray:
        return np.array(self._speeds, dtype=float)

    @property
    def q_reference(self) -> np.ndarray:
        return np.clip(np.zeros(self.dof), *self.bounds.T)

    @property
    def reference_palm_length_m(self) -> float:
        if self._palm_length is None:
            raise ValueError("Target hand has no declared palm reference geometry")
        return self._palm_length

    @property
    def native_to_canonical(self) -> np.ndarray:
        if self._native_to_canonical is None:
            raise ValueError("Target hand has no declared palm reference geometry")
        return np.array(self._native_to_canonical, dtype=float)

    def expand(self, positions) -> dict[str, float]:
        values = np.asarray(positions, dtype=float)
        if values.shape != (self.dof,) or not np.isfinite(values).all():
            raise ValueError("Independent positions must be finite and match the target DOF")
        bounds = self.bounds
        if np.any(values < bounds[:, 0] - 1e-6) or np.any(values > bounds[:, 1] + 1e-6):
            raise ValueError("Independent positions violate effective joint limits")
        drivers = dict(zip(self.independent_joint_names, np.clip(values, *bounds.T)))
        return {name: float(drivers[driver] * a + b)
                for name, (driver, a, b) in zip(self.joint_names, self._relations)}


def load_hand_spec(hand_model: str, side: str) -> HandSpec:
    """Read a registered target without loading an optimizer, network or device."""
    if hand_model not in HAND_MODELS or side not in ("left", "right"):
        raise ValueError(f"Unknown hand model/side: {hand_model}/{side}")
    if hand_model == "l21":
        path = L21_ROOT / side / f"linkerhand_l21_{side}.urdf"
        tree = ET.parse(path).getroot()
        joints = {j.get("name"): j for j in tree.findall("joint")}
        names = tuple(L21_JOINT_ORDER)
        limits = tuple((float(joints[n].find("limit").get("lower")),
                        float(joints[n].find("limit").get("upper"))) for n in names)
        speeds = tuple(float(joints[n].find("limit").get("velocity")) for n in names)
        palm = ("hand_base_link", joints["index_mcp_roll"].find("child").get("link"),
                joints["middle_mcp_roll"].find("child").get("link"))
        return HandSpec(hand_model, side, path, "hand_base_link", names, names, limits, speeds,
                        tip_frames=("thumb_tip", "index_tip", "middle_tip", "ring_tip", "pinky_tip"),
                        human_tip_indices=(4, 9, 14, 19, 24), palm_frames=palm)
    from .native_hand import NativeHandModel, PREFIX

    model = NativeHandModel(hand_model, side)
    tree = ET.parse(model.urdf_path).getroot()
    joints = {j.get("name"): j for j in tree.findall("joint")}
    rules = tuple((name, *model.relations[name]) for name in model.joint_names
                  if name not in model.independent_names)
    prefix = PREFIX[hand_model][side]
    suffix = "_0_link" if hand_model == "g1" else "_proximal"
    return HandSpec(
        hand_model, side, model.urdf_path, model.root_link, model.joint_names,
        model.independent_names, tuple(model.limits[n] for n in model.joint_names),
        tuple(float(joints[n].find("limit").get("velocity")) for n in model.joint_names),
        mimic_rules=rules, tip_frames=tuple(model.tip_frames),
        human_tip_indices=tuple(int(i) for i in model.human_tip_indices),
        palm_frames=(model.root_link, prefix + "index" + suffix, prefix + "middle" + suffix),
        source_asset_hash=model.config_hash,
    )
