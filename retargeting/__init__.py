"""LinkerHand L21 two-hand retargeting package."""

from retargeting.config import DEFAULT_CHECKPOINT, JOINT_NAMES, L21
from retargeting.retargeter import (
    DexRetargetingRetargeter,
    HandCommand,
    HandRetargeter,
    PoseTransformerRetargeter,
)

__all__ = [
    "DEFAULT_CHECKPOINT",
    "JOINT_NAMES",
    "L21",
    "angle18_to_dofs",
    "angle18_to_nodes",
    "iter_angle_h5",
    "DexRetargetingRetargeter",
    "HandCommand",
    "HandRetargeter",
    "PoseTransformerRetargeter",
]


def __getattr__(name):
    if name in {"angle18_to_dofs", "angle18_to_nodes", "iter_angle_h5"}:
        from retargeting import simulation

        return getattr(simulation, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
