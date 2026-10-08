"""A simulator sample ready for persistence; no simulator handles."""
from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class SimulationSample:
    elapsed: float
    joint_positions: list | None
    joint_velocities: list | None
    object_poses: dict
    cameras: dict[str, Any] = field(default_factory=dict)
