"""Shared normalized human targets and differentiable native-hand kinematics."""
from dataclasses import dataclass

import numpy as np
import torch
from torch import nn

from teleoperation.contracts.hand import HandWindow
from teleoperation.contracts.coordinates import validate_palm_length_contract
from teleoperation.robots.hand_spec import HandSpec
from teleoperation.robots.hand_kinematics import read_hand_tree


@dataclass(frozen=True)
class HandTargets:
    vectors: np.ndarray | None
    positions_native_m: np.ndarray | None
    timestamp: float | None


def human_target_vectors(points, spec):
    """Normalized latest human frames (...,25,3) in the native root basis."""
    values = np.asarray(points)
    if values.shape[-2:] != (25, 3) or not np.isfinite(values).all():
        raise ValueError("Targets require finite hand25 frames")
    return (values[..., list(spec.human_tip_indices), :] - values[..., :1, :]) @ spec.native_to_canonical.T


def build_hand_targets(window: HandWindow, spec: HandSpec) -> HandTargets:
    if not isinstance(window, HandWindow):
        raise TypeError("Target construction requires a normalized HandWindow")
    validate_palm_length_contract(window.metadata, "Hand targets")
    points = window.hands[spec.side]
    if points is None:
        return HandTargets(None, None, window.timestamp)
    if points.shape != (3, 25, 3) or not np.isfinite(points).all():
        raise ValueError("Hand targets require a finite three-frame normalized window")
    vectors = human_target_vectors(points[-1], spec)
    return HandTargets(vectors, vectors * spec.reference_palm_length_m, window.timestamp)


class DifferentiableHandModel(nn.Module):
    """Native root-frame positions with gradients through independent coordinates.

    Fixed joints contribute origin transforms; mimic coordinates remain affine
    functions of the independent joints. No float/NumPy conversion of q occurs.
    """

    def __init__(self, spec: HandSpec):
        super().__init__()
        self.spec = spec
        self.tree = read_hand_tree(spec.urdf_path, spec.root_link)
        self._indices = {name: i for i, name in enumerate(spec.independent_joint_names)}
        self._relations = dict(zip(spec.joint_names, spec._relations))
        self._frames = {spec.root_link, *(joint.child for joint in self.tree)}
        active = {joint.name for joint in self.tree if joint.kind != "fixed"}
        if active != set(spec.joint_names):
            raise ValueError("Hand specification does not match its movable URDF subtree")

    def positions(self, q: torch.Tensor, frame_names: tuple[str, ...]) -> torch.Tensor:
        if q.ndim != 2 or q.shape[1] != self.spec.dof or not torch.isfinite(q).all():
            raise ValueError("FK requires finite batched independent coordinates")
        if any(name not in self._frames for name in frame_names):
            raise ValueError("FK requested a frame outside the target hand")
        batch = q.shape[0]
        identity = torch.eye(3, dtype=q.dtype, device=q.device)
        transforms = {self.spec.root_link: (identity.expand(batch, 3, 3), q.new_zeros((batch, 3)))}
        for joint in self.tree:
            origin = torch.as_tensor(joint.origin, dtype=q.dtype, device=q.device)
            parent_r, parent_t = transforms[joint.parent]
            rotation = origin[:3, :3].expand(batch, 3, 3)
            translation = parent_t + (parent_r @ origin[:3, 3].reshape(1, 3, 1)).squeeze(-1)
            if joint.kind != "fixed":
                driver, multiplier, offset = self._relations[joint.name]
                angle = q[:, self._indices[driver]] * multiplier + offset
                x, y, z = joint.axis
                cross = torch.tensor([[0., -z, y], [z, 0., -x], [-y, x, 0.]], dtype=q.dtype, device=q.device)
                motion = identity + torch.sin(angle)[:, None, None] * cross + (1-torch.cos(angle))[:, None, None] * (cross @ cross)
                rotation = rotation @ motion
            transforms[joint.child] = parent_r @ rotation, translation
        return torch.stack([transforms[name][1] for name in frame_names], dim=1)

    def vectors(self, q: torch.Tensor) -> torch.Tensor:
        frames = (self.spec.root_link, *self.spec.tip_frames)
        points = self.positions(q, frames)
        return (points[:, 1:] - points[:, :1]) / self.spec.reference_palm_length_m
