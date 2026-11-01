"""Explicit 18D result → 17DOF → native mapping application boundary."""
import numpy as np
from teleoperation.retargeting.hand.angles import legacy_l21_dofs
from teleoperation.robots.native_hand import map_dofs


def map_frame(row, mapping, limit_mode="clamp"):
    dofs = legacy_l21_dofs(np.asarray(row, dtype=np.float64))
    return map_dofs(dofs, mapping, limit_mode)
