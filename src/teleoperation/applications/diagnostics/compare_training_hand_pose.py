"""Compare before/after checkpoint FK outputs against hand keypoint targets."""

from __future__ import annotations

import argparse

from pathlib import Path
import sys

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

from teleoperation.paths import PROJECT_ROOT

from teleoperation.retargeting.hand.kinematics import create_hand_kinematics
from teleoperation.retargeting.hand.device import resolve_device as _resolve_device
from teleoperation.retargeting.hand.config import (
    JOINT_EDGES,
    JOINT_NAMES,
    L21,
)
from teleoperation.learning.dataset import HAND_SIDES, TwoHandH5Dataset
from teleoperation.retargeting.hand.predictor import create_twohand_retargeter
from teleoperation.applications.hand_checkpoints import checkpoint_options, checkpoint_description


HAND25_CONNECTIONS = (
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8), (8, 9),
    (0, 10), (10, 11), (11, 12), (12, 13), (13, 14),
    (0, 15), (15, 16), (16, 17), (17, 18), (18, 19),
    (0, 20), (20, 21), (21, 22), (22, 23), (23, 24),
)
TIP_NAMES = ("thumb_tip", "index_tip", "middle_tip", "ring_tip", "pinky_tip")
SIDE_COLORS = {"left": "#2563eb", "right": "#dc2626"}


def main(args=None) -> int:
    if args is None:
        raise TypeError("Use teleoperation.cli.main() to parse command arguments")

    device = _resolve_device(args.device)
    dataset = TwoHandH5Dataset(
        args.input,
        receptive_field=L21.model.receptive_field,
        scale_factor=L21.training.source_scale,
    )
    sample = _sample_for_frame(dataset, args.frame)

    fks = _create_fks(device)
    checkpoints = (
        ("Before", checkpoint_options(args, prefix="before")),
        ("After", checkpoint_options(args, prefix="after")),
    )
    predictions = {
        title: _predict_checkpoint(selection["checkpoint_path"], sample, fks, device,
                                   dataset.coordinate_alignment,
                                   left_checkpoint=selection["left_checkpoint"],
                                   right_checkpoint=selection["right_checkpoint"])
        for title, selection in checkpoints
    }
    for title, selection in checkpoints:
        print(f"{title}: {checkpoint_description(selection)}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    _plot_comparison(sample, predictions, args.frame, args.output)
    print(args.output)
    return 0


def _sample_for_frame(dataset: TwoHandH5Dataset, frame: int) -> dict:
    by_frame = {sample["frame_index"]: sample for sample in dataset.samples}
    if frame not in by_frame:
        raise ValueError(
            f"Frame {frame} is not available as a training sample. "
            f"Try frame >= {L21.model.receptive_field - 1} or a valid hand frame."
        )
    sample = by_frame[frame]
    if not any(sample[f"{side}_valid"] for side in HAND_SIDES):
        raise ValueError(f"Frame {frame} has no valid hand target")
    return sample


def _create_fks(device: torch.device) -> dict:
    cfg = L21.hand_kinematics_config()
    return {
        "left": create_hand_kinematics(
            L21.left_urdf,
            cfg,
            device=device,
            scale_factor=L21.training.robot_scale,
        ),
        "right": create_hand_kinematics(
            L21.right_urdf,
            cfg,
            device=device,
            scale_factor=L21.training.robot_scale,
        ),
    }


def _predict_checkpoint(
    checkpoint: Path | str | None,
    sample: dict,
    fks: dict,
    device: torch.device,
    coordinate_alignment: str,
    *,
    left_checkpoint: Path | str | None = None,
    right_checkpoint: Path | str | None = None,
) -> dict:
    retargeter = create_twohand_retargeter(
        L21.model_kwargs(), device,
        checkpoint_path=None if checkpoint is None else str(checkpoint),
        left_checkpoint=None if left_checkpoint is None else str(left_checkpoint),
        right_checkpoint=None if right_checkpoint is None else str(right_checkpoint),
        expected_coordinate_alignment=coordinate_alignment,
    ).eval()

    result = {}
    with torch.no_grad():
        for side in HAND_SIDES:
            if not sample[f"{side}_valid"]:
                result[side] = None
                continue
            hand_input = torch.from_numpy(sample[f"{side}_input"][None].astype(np.float32)).to(device)
            angles18 = retargeter.model_for_side(side)(hand_input)
            angles23 = F.pad(angles18, (0, 5, 0, 0), "constant", 0)
            _, _, robot = fks[side].forward(angles23)
            result[side] = robot[0].detach().cpu().numpy()
    return result


from teleoperation.tools.training_hand_pose import _apply_limits, _draw_robot, _draw_target, _equal_limits, _plot_comparison
