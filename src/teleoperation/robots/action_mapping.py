"""Robot-specific action encoding, with no cross-robot arm conversion."""
from dataclasses import dataclass
from typing import Mapping
import numpy as np
from teleoperation.contracts.commands import Tron2AL21Command
from .partial import PartialLimbVector


from teleoperation.contracts.commands import H1JointTargets, GR1JointTargets, G1JointTargets


def validate_native_action(action, action_dim):
    if isinstance(action, PartialLimbVector): raise TypeError("PartialLimbVector is not a complete robot action")
    if not isinstance(action, np.ndarray): raise TypeError("Native action must be np.ndarray")
    if action.shape != (action_dim,): raise ValueError(f"Native action must have shape ({action_dim},), got {action.shape}")
    if not np.isfinite(action).all(): raise ValueError("Native action must be finite")
    return action


class _NativeActionAdapter:
    def __init__(self, joint_names):
        self.joint_names = tuple(joint_names)
        if len(self.joint_names) != self.dimension: raise ValueError("Robot layout dimension mismatch")
        if len(set(self.joint_names)) != self.dimension: raise ValueError("Duplicate native joints")

    def encode(self, targets):
        if not isinstance(targets, self.target_type): raise TypeError(f"Expected {self.target_type.__name__}; no cross-robot arm mapping is available")
        missing = set(self.joint_names) - targets.joints.keys()
        if missing: raise ValueError(f"Missing native joints: {sorted(missing)}")
        return validate_native_action(np.asarray([targets.joints[name] for name in self.joint_names]), self.dimension)


class H1ActionAdapter(_NativeActionAdapter):
    target_type, dimension = H1JointTargets, 38

class GR1ActionAdapter(_NativeActionAdapter):
    target_type, dimension = GR1JointTargets, 36

class G1ActionAdapter(_NativeActionAdapter):
    target_type, dimension = G1JointTargets, 28

def pack_tron2a_l21_action(
    left_arm: np.ndarray | None,
    left_hand: np.ndarray | None,
    right_arm: np.ndarray | None,
    right_hand: np.ndarray | None,
) -> np.ndarray:
    """Build the fixed 48D simulator action in canonical limb order."""
    parts = (("left_arm", left_arm, 7), ("left_hand", left_hand, 17), ("right_arm", right_arm, 7), ("right_hand", right_hand, 17))
    values = []
    for name, part, size in parts:
        if part is None:
            raise ValueError(f"{name} is required for a fixed Tron2AL21Command")
        value = np.asarray(part, dtype=np.float32).reshape(-1)
        if value.shape != (size,) or not np.isfinite(value).all():
            raise ValueError(f"{name} must be finite with shape ({size},)")
        values.append(value)
    return np.concatenate(values, axis=0)


class Tron2AL21ActionAdapter:
    def encode(self, command: Tron2AL21Command):
        if not isinstance(command, Tron2AL21Command):
            raise TypeError("Expected Tron2AL21Command")
        return pack_tron2a_l21_action(command.left_arm, command.left_hand, command.right_arm, command.right_hand)
