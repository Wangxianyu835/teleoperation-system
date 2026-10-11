"""Partial diagnostic vectors never constitute executable robot actions."""
from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class PartialLimbVector:
    values: np.ndarray
    limbs: tuple[str, ...]


def pack_partial_limb_vector(left_arm, left_hand, right_arm, right_hand):
    parts = [(name, value) for name, value in zip(("left_arm", "left_hand", "right_arm", "right_hand"), (left_arm, left_hand, right_arm, right_hand)) if value is not None]
    if not parts: raise ValueError("No valid action parts provided")
    values = np.concatenate([np.asarray(value, dtype=np.float32).reshape(-1) for _, value in parts])
    return PartialLimbVector(values, tuple(name for name, _ in parts))
