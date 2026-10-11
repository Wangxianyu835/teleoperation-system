# 类 SimulationSample（封装数据/流程）；链路：输入/存储边界→应用/算法数据契约
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
