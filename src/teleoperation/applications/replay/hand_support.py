"""Explicit 18D result → 17DOF → native mapping application boundary."""
import numpy as np
from teleoperation.contracts.hand import L21HandAngles
from teleoperation.robots.native_hand import map_dofs


def map_frame(row, mapping, limit_mode="clamp", pinch=None,
              robot_type=None, side="right"):
    dofs = L21HandAngles(np.asarray(row, dtype=np.float64)).native_mapping_dofs()
    return map_dofs(dofs.values, mapping, limit_mode, pinch=pinch,
                    robot_type=robot_type, side=side)
