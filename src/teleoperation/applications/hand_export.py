"""Run shared or dedicated hand models on an offline H5 recording."""

from __future__ import annotations

import argparse

from pathlib import Path

import h5py
import numpy as np
import torch

from teleoperation.retargeting.hand.device import resolve_device as _resolve_device
from teleoperation.retargeting.hand.config import ANGLE_LIMITS, L21, RUNTIME
from teleoperation.paths import DEFAULT_CHECKPOINT, DEFAULT_WORLD_REALTIME_CHECKPOINT
from teleoperation.learning.dataset import (
    HAND_SIDES,
    TwoHandH5ChunkedGenerator,
    TwoHandH5Dataset,
)
from teleoperation.contracts.constants import DEFAULT_MAX_CENTER_DISPLACEMENT, DEFAULT_MAX_SHAPE_RMSE
from teleoperation.retargeting.hand.predictor import create_twohand_retargeter
from .hand_checkpoints import checkpoint_options, checkpoint_metadata, checkpoint_description


def run(args: argparse.Namespace) -> int:
    device = _resolve_device(args.device)
    dataset = TwoHandH5Dataset(
        args.input,
        receptive_field=L21.model.receptive_field,
        scale_factor=args.scale_factor,
        reset_on_gaps=True,
        track_identity=not args.disable_identity_tracking,
        max_center_displacement=args.max_center_displacement,
        max_shape_rmse=args.max_shape_rmse,
    )
    coordinate_alignment = dataset.coordinate_alignment
    if dataset.is_world and L21.training.robot_scale != 1.0:
        raise ValueError("World export requires robot_scale=1.0 (URDF metres)")
    checkpoints = checkpoint_options(args, default=(DEFAULT_WORLD_REALTIME_CHECKPOINT
                                                   if dataset.is_world else DEFAULT_CHECKPOINT))
    retargeter = create_twohand_retargeter(
        model_kwargs=_model_kwargs(),
        device=device,
        **checkpoints,
        expected_coordinate_alignment=coordinate_alignment,
    )
    retargeter.eval()

    frame_count = dataset.frame_count
    generator = TwoHandH5ChunkedGenerator(dataset, batch_size=args.batch_size, shuffle=False)
    angles, valid = predict_batches(retargeter, generator.next_epoch(), frame_count, device)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    metadata = {
        "input_file": str(args.input), **checkpoint_metadata(checkpoints),
        "output_shape": (L21.model.output_joints,), "coordinate_frame": dataset.coordinate_frame,
        "coordinate_alignment": coordinate_alignment, "identity_tracking": dataset.track_identity,
        "max_center_displacement": args.max_center_displacement, "max_shape_rmse": args.max_shape_rmse,
        "invalid_angle_policy": "hold_previous",
    }
    if dataset.source_landmark_space is not None: metadata["source_landmark_space"] = dataset.source_landmark_space
    if dataset.is_world:
        metadata.update({"length_unit": "m", "units": "m", "source_scale": 1.0})
    if dataset.is_canonical:
        metadata.update({"canonical_hand": "right", "angle_convention": "l21_native_angles_v1"})
    write_hand_angles(output_path, dataset.frame_ids, dataset.timestamps, angles, valid, metadata)

    print(f"device={device}")
    print(f"input={args.input}")
    print(checkpoint_description(checkpoints))
    print(f"coordinate_alignment={coordinate_alignment}")
    print(f"output={output_path}")
    print(f"frames={frame_count}")
    for side in HAND_SIDES:
        print(f"{side}_predictions={int(valid[side].sum())}")
        print(f"{side}_shape={angles[side].shape}")
    return 0


def _model_kwargs() -> dict:
    return L21.model_kwargs()


from teleoperation.retargeting.hand.exporting import predict_batches
from teleoperation.data.hand_h5 import write_hand_angles
