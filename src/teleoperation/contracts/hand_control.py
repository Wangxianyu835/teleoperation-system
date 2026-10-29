"""Robot-specific independent joint solutions shared by retargeting methods."""
from dataclasses import dataclass
import re

import numpy as np


HandKey = tuple[str, str]


def valid_joint_result(results, hand_model, side):
    """Select one valid target result; missing/invalid solutions remain absent."""
    if results is None:
        return None
    result = results.get((hand_model, side))
    return result if result is not None and result.valid else None


@dataclass(frozen=True)
class HandJointResult:
    """One target hand's ordered independent coordinates, in radians.

    Values are a solution, not a filtered command. An invalid solution can omit
    positions; its ordered names and specification still identify the target.
    """

    hand_model: str
    side: str
    spec_hash: str
    joint_names: tuple[str, ...]
    positions: np.ndarray | None
    timestamp: float
    valid: bool = True
    error: str | None = None
    solve_ms: float = 0.

    def __post_init__(self):
        if not isinstance(self.hand_model, str) or not re.fullmatch(r"[a-z0-9][a-z0-9_-]*", self.hand_model):
            raise ValueError("A hand result requires a registered-style hand model identifier")
        if self.side not in ("left", "right"):
            raise ValueError("A hand result requires left or right target side")
        if not isinstance(self.spec_hash, str) or not re.fullmatch(r"[0-9a-f]{64}", self.spec_hash):
            raise ValueError("A hand result requires the target specification SHA256")
        names = tuple(self.joint_names)
        if not names or any(not isinstance(name, str) or not name for name in names) or len(set(names)) != len(names):
            raise ValueError("Independent joint names must be nonempty and unique")
        if self.timestamp is None or not np.isfinite(self.timestamp):
            raise ValueError("A hand result timestamp must be finite seconds")
        if not np.isfinite(self.solve_ms) or self.solve_ms < 0:
            raise ValueError("Solve time must be finite nonnegative milliseconds")
        if not isinstance(self.valid, (bool, np.bool_)):
            raise ValueError("A hand result validity must be boolean")
        if self.valid and self.positions is None:
            raise ValueError("A valid hand result requires independent joint positions")
        if self.positions is not None:
            positions = np.array(self.positions, dtype=np.float64, copy=True)
            if positions.shape != (len(names),) or not np.isfinite(positions).all():
                raise ValueError("Hand positions must be a finite vector matching the ordered independent joints")
            positions.setflags(write=False)
            object.__setattr__(self, "positions", positions)
        object.__setattr__(self, "joint_names", names)

    @property
    def key(self) -> HandKey:
        return self.hand_model, self.side

    def named_positions(self):
        return {} if self.positions is None else dict(zip(self.joint_names, self.positions))


@dataclass(frozen=True)
class HandJointCommand:
    """Accepted, expanded native joint positions ready for a named executor."""
    hand_model: str
    side: str
    spec_hash: str
    joint_names: tuple[str, ...]
    positions: np.ndarray
    timestamp: float

    def __post_init__(self):
        checked = HandJointResult(self.hand_model, self.side, self.spec_hash,
                                  self.joint_names, self.positions, self.timestamp)
        object.__setattr__(self, "joint_names", checked.joint_names)
        object.__setattr__(self, "positions", checked.positions)

    @property
    def key(self) -> HandKey:
        return self.hand_model, self.side

    def named_positions(self):
        return dict(zip(self.joint_names, self.positions))
