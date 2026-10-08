"""Train one shared left/right retargeting model from an H5 recording."""

from __future__ import annotations

import argparse

import logging
import random
import csv
from datetime import datetime
from pathlib import Path
from time import perf_counter

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

try:
    from tensorboardX import SummaryWriter
except ModuleNotFoundError:
    try:
        from torch.utils.tensorboard import SummaryWriter
    except ModuleNotFoundError:
        class SummaryWriter:
            def __init__(self, *args, **kwargs):
                pass

            def add_scalar(self, *args, **kwargs):
                pass

            def close(self):
                pass

from teleoperation.retargeting.hand.device import resolve_device as _resolve_device
from teleoperation.retargeting.hand.config import ANGLE_LIMITS, RUNTIME, EXCLUDED_COLLISION_PAIRS, L21, ROBOT_JOINTS, SOURCE_JOINTS
from teleoperation.paths import DEFAULT_WARMSTART_CHECKPOINT, DEFAULT_CHECKPOINT_ROOT
from teleoperation.contracts.coordinates import COORDINATE_FRAME, validate_coordinate_alignment
from teleoperation.retargeting.hand.predictor import _require_coordinate_alignment
from teleoperation.learning.dataset import HAND_SIDES, TwoHandH5ChunkedGenerator, TwoHandH5Dataset
from teleoperation.data.hand_h5 import load_twohand_h5
from teleoperation.retargeting.hand.kinematics import create_hand_kinematics
from teleoperation.learning.losses import CollisionLoss, RegLoss, hand_loss
from teleoperation.retargeting.hand.transformer import PoseTransformer


receptive_field = L21.model.receptive_field
scaling_factor = L21.training.source_scale
scaling_factor_rb = L21.training.robot_scale
hand_brand = "linker"
hand_cfg = L21.hand_kinematics_config()
left_urdf_file = L21.left_urdf
right_urdf_file = L21.right_urdf
source_dic = SOURCE_JOINTS
rb_dic = ROBOT_JOINTS
excluded_pairs = EXCLUDED_COLLISION_PAIRS
angle_limit_rob = ANGLE_LIMITS
loss_weight = L21.training.loss_weights
col_threshold = L21.training.collision_threshold
correction_matrix = None


LOSS_NAMES = (
    "total",
    "vec",
    "pos",
    "collision",
    "thumb",
    "tip_distance",
    "thumb2",
)


def _run_epoch(
    model,
    generator,
    device,
    pos_loss,
    vec_loss,
    collision_losses,
    reg_criterion,
    hand_fks,
    optimizer,
    model_parameters,
    writer,
    global_step,
    logger,
    training,
):
    totals = {name: 0.0 for name in LOSS_NAMES}
    side_totals = {
        side: {name: 0.0 for name in LOSS_NAMES}
        for side in HAND_SIDES
    }
    side_counts = {side: 0 for side in HAND_SIDES}
    total_count = 0
    batch_count = 0
    gradient_total = 0.0

    for batch_index, batch in enumerate(generator.next_epoch(), start=1):
        if training:
            optimizer.zero_grad(set_to_none=True)

        losses = {}
        masks = {}
        for side in HAND_SIDES:
            masks[side] = torch.as_tensor(
                batch[f"{side}_valid"],
                dtype=torch.bool,
                device=device,
            )
            if masks[side].any():
                predicted_angle = F.pad(
                    model(_tensor(batch[f"{side}_input"], device)),
                    (0, 5, 0, 0),
                    "constant",
                    0,
                )
                losses[side] = _masked_hand_loss(
                    predicted_angle,
                    _tensor(batch[f"{side}_target"], device),
                    masks[side],
                    pos_loss,
                    vec_loss,
                    collision_losses[side],
                    reg_criterion,
                    hand_fks[side],
                    logger,
                    hand_side=side,
                )
            else:
                zero = next(model.parameters()).sum() * 0.0
                losses[side] = (zero,) * len(LOSS_NAMES)

        active_sides = [side for side in HAND_SIDES if masks[side].any()]
        if not active_sides:
            continue
        loss_total = sum(losses[side][0] for side in active_sides)
        loss_total = loss_total / len(active_sides)
        context = f"batch={batch_index}, global_step={global_step}"
        if not torch.isfinite(loss_total).all():
            raise FloatingPointError(f"Non-finite total loss at {context}")

        if training:
            loss_total.backward()
            try:
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    model_parameters,
                    max_norm=L21.training.gradient_clip_norm,
                    error_if_nonfinite=True,
                )
            except RuntimeError as error:
                raise RuntimeError(f"Gradient clipping failed at {context}: {error}") from error
            optimizer.step()
            gradient_total += float(grad_norm)
            writer.add_scalar(
                "Batch/gradient_norm",
                float(grad_norm),
                global_step,
            )

        batch_count += 1
        global_step += 1
        batch_valid_count = 0
        for side in HAND_SIDES:
            count = int(masks[side].sum().item())
            side_counts[side] += count
            batch_valid_count += count
            if count == 0:
                continue
            batch_loss_values = losses[side]
            for name, value in zip(LOSS_NAMES, batch_loss_values):
                numeric_value = float(value.detach().item())
                side_totals[side][name] += numeric_value * count
                totals[name] += numeric_value * count
            writer.add_scalar(
                f"Batch/{side}_valid",
                count,
                global_step,
            )
        total_count += batch_valid_count
        writer.add_scalar(
            "Batch/total",
            float(loss_total.detach().item()),
            global_step,
        )

    divisor = max(total_count, 1)
    stats = {name: totals[name] / divisor for name in LOSS_NAMES}
    stats["gradient_norm"] = gradient_total / max(batch_count, 1)
    for side in HAND_SIDES:
        side_divisor = max(side_counts[side], 1)
        for name in LOSS_NAMES:
            stats[f"{side}_{name}"] = (
                side_totals[side][name] / side_divisor
            )
        stats[f"{side}_valid"] = side_counts[side]
    return stats, global_step


def _masked_hand_loss(
    predicted_angle,
    target_3d,
    mask,
    pos_loss,
    vec_loss,
    col_loss,
    reg_criterion,
    hand_fk,
    logger,
    hand_side,
):
    return hand_loss(
        predicted_angle[mask],
        target_3d[mask],
        rb_dic,
        source_dic,
        pos_loss,
        vec_loss,
        col_loss,
        reg_criterion,
        visualizer=None,
        hand_fk_model=hand_fk,
        logger=logger,
        loss_weight=loss_weight,
        hand_side=hand_side,
    )


def _tensor(array, device):
    return torch.from_numpy(np.asarray(array, dtype=np.float32)).to(device)


def _split_frame_ranges(frame_count: int, val_ratio: float, receptive_field: int):
    if not 0.0 < val_ratio < 1.0:
        raise ValueError("val_ratio must be between 0 and 1")
    train_end = int(frame_count * (1.0 - val_ratio))
    boundary_gap = max(receptive_field - 1, 0)
    val_start = train_end + boundary_gap
    if train_end < receptive_field:
        raise ValueError("Training split is shorter than receptive_field")
    if val_start >= frame_count:
        raise ValueError("Validation split is empty after boundary gap")
    return train_end, val_start
