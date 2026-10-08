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


def _metrics_fieldnames() -> list[str]:
    return [
        "epoch",
        "train_total",
        "train_left_total",
        "train_right_total",
        "val_total",
        "val_left_total",
        "val_right_total",
        "learning_rate",
        "gradient_norm",
        "left_valid",
        "right_valid",
    ]


def _epoch_metrics_row(epoch, train_stats, val_stats, learning_rate) -> dict:
    return {
        "epoch": epoch,
        "train_total": train_stats["total"],
        "train_left_total": train_stats["left_total"],
        "train_right_total": train_stats["right_total"],
        "val_total": val_stats["total"],
        "val_left_total": val_stats["left_total"],
        "val_right_total": val_stats["right_total"],
        "learning_rate": learning_rate,
        "gradient_norm": train_stats["gradient_norm"],
        "left_valid": train_stats["left_valid"],
        "right_valid": train_stats["right_valid"],
    }


def _write_epoch_stats(writer, prefix, stats, epoch):
    for name in LOSS_NAMES:
        tag_name = "total" if name == "total" else name
        writer.add_scalar(f"{prefix}/{tag_name}", stats[name], epoch)
    if prefix == "Train":
        writer.add_scalar("Train/gradient_norm", stats["gradient_norm"], epoch)
    for side in HAND_SIDES:
        for name in LOSS_NAMES:
            tag_name = "total" if name == "total" else name
            writer.add_scalar(
                f"{prefix}/{side}_{tag_name}",
                stats[f"{side}_{name}"],
                epoch,
            )
        writer.add_scalar(
            f"{prefix}/{side}_valid",
            stats[f"{side}_valid"],
            epoch,
        )
