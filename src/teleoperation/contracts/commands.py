"""Fixed dual-arm, dual-hand command contract."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import numpy as np


@dataclass(frozen=True)
class Tron2AL21Command:
    timestamp: float
    left_arm: np.ndarray
    left_hand: np.ndarray
    right_arm: np.ndarray
    right_hand: np.ndarray
    left_arm_valid: bool
    left_hand_valid: bool
    right_arm_valid: bool
    right_hand_valid: bool

    def __post_init__(self) -> None:
        for name, size in (("left_arm", 7), ("right_arm", 7), ("left_hand", 17), ("right_hand", 17)):
            value = np.asarray(getattr(self, name), dtype=np.float32)
            if value.shape != (size,) or not np.isfinite(value).all():
                raise ValueError(f"{name} must be finite with shape ({size},)")
            object.__setattr__(self, name, value)

    def flatten(self) -> np.ndarray:
        return np.concatenate((self.left_arm, self.left_hand, self.right_arm, self.right_hand)).astype(np.float32)


@dataclass(frozen=True)
class H1JointTargets:
    joints: Mapping[str, float]

@dataclass(frozen=True)
class GR1JointTargets:
    joints: Mapping[str, float]

@dataclass(frozen=True)
class G1JointTargets:
    joints: Mapping[str, float]
