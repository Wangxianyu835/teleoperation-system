"""Visualize LinkerHand L21 FK nodes from the configured URDF assets."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import torch

from teleoperation.paths import PROJECT_ROOT

from teleoperation.retargeting.hand.kinematics import create_hand_kinematics
from teleoperation.retargeting.hand.config import JOINT_EDGES, JOINT_NAMES, L21


TIP_NAMES = ("thumb_tip", "index_tip", "middle_tip", "ring_tip", "pinky_tip")
SIDE_COLORS = {
    "left": "#2563eb",
    "right": "#dc2626",
}


def main(args=None) -> int:
    if args is None:
        raise TypeError("Use teleoperation.cli.main() to parse command arguments")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    sides = ("left", "right") if args.side == "both" else (args.side,)
    outputs = []
    for side in sides:
        path = args.output_dir / f"l21_{side}_fk_{args.pose}.png"
        _plot_side(_fk_positions(side, args.pose), side, args.pose, args.label_names, path)
        outputs.append(path)

    if len(sides) == 2:
        combined = args.output_dir / f"l21_both_fk_{args.pose}.png"
        _plot_both({side: _fk_positions(side, args.pose) for side in ("left", "right")}, args.pose, args.label_names, combined)
        outputs.append(combined)

    for path in outputs:
        print(path)
    return 0


def _fk_positions(side: str, pose: str) -> np.ndarray:
    urdf = L21.left_urdf if side == "left" else L21.right_urdf
    fk = create_hand_kinematics(
        urdf,
        L21.hand_kinematics_config(),
        device="cpu",
        scale_factor=L21.training.robot_scale,
    )
    angles = _angles_for_pose(pose)
    _, _, positions = fk.forward(angles)
    return positions[0].detach().cpu().numpy()


from teleoperation.tools.l21_fk import _draw_hand, _finish_axes, _plot_both, _plot_side, _set_equal_axes

from teleoperation.retargeting.hand.fk_demo import _angles_for_pose
