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


def run(args: argparse.Namespace) -> int:
    _validate_args(args)
    _set_seed(args.seed)

    device = _resolve_device(args.device)
    checkpoint_dir = Path(args.checkpoint_root)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    run_name = _resolve_run_name(args.run_name)
    logger = _setup_logging(
        checkpoint_dir / "logs" / "twohand_h5" / hand_brand / run_name / "training.log"
    )
    start_time = perf_counter()

    h5_path = Path(args.input)
    frame_count = load_twohand_h5(h5_path)[0].shape[0]
    train_end, val_start = _split_frame_ranges(
        frame_count,
        args.val_ratio,
        receptive_field,
    )

    train_dataset = TwoHandH5Dataset(
        h5_path,
        receptive_field=receptive_field,
        scale_factor=scaling_factor,
        frame_start=0,
        frame_end=train_end,
        reset_on_gaps=True,
    )
    val_dataset = TwoHandH5Dataset(
        h5_path,
        receptive_field=receptive_field,
        scale_factor=scaling_factor,
        frame_start=val_start,
        frame_end=frame_count,
        reset_on_gaps=True,
    )
    coordinate_alignment = train_dataset.coordinate_alignment
    if val_dataset.coordinate_alignment != coordinate_alignment:
        raise ValueError(
            "Training/validation coordinate alignment mismatch: "
            f"training={coordinate_alignment!r}; validation={val_dataset.coordinate_alignment!r}"
        )
    if len(train_dataset) == 0:
        raise ValueError("Training split contains no complete hand windows")
    if len(val_dataset) == 0:
        raise ValueError("Validation split contains no complete hand windows")

    print(f"device={device}")
    print(f"h5_path={h5_path}")
    print(f"raw_frames={frame_count}")
    print(f"train_raw_range=[0, {train_end})")
    print(f"validation_raw_range=[{val_start}, {frame_count})")
    print(f"train_windows={len(train_dataset)}")
    print(f"validation_windows={len(val_dataset)}")
    print(f"coordinate_alignment={coordinate_alignment}")
    print(f"run_name={run_name}")
    print(f"train_side_counts={train_dataset.side_counts()}")
    print(f"validation_side_counts={val_dataset.side_counts()}")
    logger.info(
        "Loaded H5=%s raw_frames=%s train_windows=%s val_windows=%s coordinate_alignment=%s",
        h5_path,
        frame_count,
        len(train_dataset),
        len(val_dataset),
        coordinate_alignment,
    )

    train_generator = TwoHandH5ChunkedGenerator(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        random_seed=args.seed,
    )
    val_generator = TwoHandH5ChunkedGenerator(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        random_seed=args.seed,
    )

    model = _create_pose_model().to(device)
    if args.init_checkpoint is not None:
        _load_checkpoint(model, args.init_checkpoint, device,
                         expected_coordinate_alignment=coordinate_alignment)
        print(f"initialized_from={args.init_checkpoint}")

    model_parameters = list(model.parameters())
    optimizer = optim.AdamW(
        model_parameters,
        lr=args.learning_rate,
        weight_decay=L21.training.weight_decay,
        eps=L21.training.optimizer_eps,
    )
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=L21.training.scheduler_factor,
        patience=L21.training.scheduler_patience,
        min_lr=L21.training.minimum_learning_rate,
    )

    pos_loss = nn.MSELoss()
    vec_loss = None
    collision_losses = {
        side: _create_collision_loss(side)
        for side in HAND_SIDES
    }
    reg_criterion = RegLoss()
    hand_fks = _create_hand_fks(device)

    output_dir = (
        checkpoint_dir
        / "models"
        / "twohand_h5"
        / hand_brand
        / run_name
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    log_dir = (
        checkpoint_dir
        / "logs"
        / "twohand_h5"
        / hand_brand
        / run_name
    )
    writer = SummaryWriter(log_dir=log_dir)
    metrics_path = output_dir / "epoch_metrics.csv"
    metrics_file = metrics_path.open("w", newline="", encoding="utf-8")
    metrics_writer = csv.DictWriter(
        metrics_file,
        fieldnames=_metrics_fieldnames(),
    )
    metrics_writer.writeheader()

    best_val = float("inf")
    best_epoch = 0
    epochs_without_improvement = 0
    global_step = 0
    final_epoch = 0

    try:
        for epoch in range(1, args.epochs + 1):
            final_epoch = epoch
            model.train()
            train_stats, global_step = _run_epoch(
                model=model,
                generator=train_generator,
                device=device,
                pos_loss=pos_loss,
                vec_loss=vec_loss,
                collision_losses=collision_losses,
                reg_criterion=reg_criterion,
                hand_fks=hand_fks,
                optimizer=optimizer,
                model_parameters=model_parameters,
                writer=writer,
                global_step=global_step,
                logger=logger,
                training=True,
            )

            model.eval()
            with torch.no_grad():
                val_stats, _ = _run_epoch(
                    model=model,
                    generator=val_generator,
                    device=device,
                    pos_loss=pos_loss,
                    vec_loss=vec_loss,
                    collision_losses=collision_losses,
                    reg_criterion=reg_criterion,
                    hand_fks=hand_fks,
                    optimizer=None,
                    model_parameters=model_parameters,
                    writer=writer,
                    global_step=global_step,
                    logger=logger,
                    training=False,
                )

            val_total = val_stats["total"]
            scheduler.step(val_total)
            learning_rate = optimizer.param_groups[0]["lr"]
            writer.add_scalar("Train/learning_rate", learning_rate, epoch)
            _write_epoch_stats(writer, "Train", train_stats, epoch)
            _write_epoch_stats(writer, "Val", val_stats, epoch)

            is_improved = val_total < best_val
            if is_improved:
                best_val = val_total
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1

            if is_improved:
                best_epoch = epoch

            metrics_writer.writerow(
                _epoch_metrics_row(
                    epoch=epoch,
                    train_stats=train_stats,
                    val_stats=val_stats,
                    learning_rate=learning_rate,
                )
            )
            metrics_file.flush()

            checkpoint_kwargs = {
                "model": model,
                "optimizer": optimizer,
                "scheduler": scheduler,
                "epoch": epoch,
                "best_val": best_val,
                "best_epoch": best_epoch,
                "h5_path": h5_path,
                "train_end": train_end,
                "val_start": val_start,
                "run_name": run_name,
                "init_checkpoint": args.init_checkpoint,
                "coordinate_alignment": coordinate_alignment,
                "source_landmark_space": train_dataset.source_landmark_space,
            }
            _save_checkpoint(
                output_dir / "model_last.pth",
                **checkpoint_kwargs,
            )
            if is_improved:
                _save_checkpoint(
                    output_dir / "model_best.pth",
                    **checkpoint_kwargs,
                )

            print(
                f"epoch={epoch} "
                f"train_total={train_stats['total']:.6f} "
                f"val_total={val_total:.6f} "
                f"lr={learning_rate:.8g}"
            )
            logger.info(
                "epoch=%s train_total=%.6f val_total=%.6f lr=%.8g",
                epoch,
                train_stats["total"],
                val_total,
                learning_rate,
            )

            if epochs_without_improvement >= args.early_stopping_patience:
                print(
                    f"early_stopping=1 patience={args.early_stopping_patience}"
                )
                break
    finally:
        writer.close()
        metrics_file.close()

    print(f"saved_last={output_dir / 'model_last.pth'}")
    print(f"saved_best={output_dir / 'model_best.pth'}")
    print(f"final_epoch={final_epoch}")
    print(f"training_time={perf_counter() - start_time:.3f}s")
    return 0


def _resolve_run_name(run_name: str | None) -> str:
    if run_name is None:
        return f"experiment_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    value = str(run_name).strip()
    if not value:
        raise ValueError("run_name must not be empty")
    if value in {".", ".."} or Path(value).name != value:
        raise ValueError("run_name must be a single directory name")
    return value


def _create_pose_model():
    return PoseTransformer(**L21.model_kwargs())


def _create_hand_fks(device):
    urdf_files = _resolve_hand_urdf_paths(left_urdf_file, right_urdf_file)
    return {
        "left": create_hand_kinematics(
            urdf_files["left"],
            hand_cfg,
            device,
            scale_factor=scaling_factor_rb,
            axis_correction_matrix=correction_matrix,
        ),
        "right": create_hand_kinematics(
            urdf_files["right"],
            hand_cfg,
            device,
            scale_factor=scaling_factor_rb,
            axis_correction_matrix=correction_matrix,
        ),
    }


def _save_checkpoint(
    path,
    model,
    optimizer,
    scheduler,
    epoch,
    best_val,
    h5_path,
    train_end,
    val_start,
    best_epoch,
    run_name,
    init_checkpoint,
    coordinate_alignment,
    source_landmark_space=None,
):
    coordinate_alignment = validate_coordinate_alignment(coordinate_alignment, "Checkpoint save")
    write_checkpoint(
        {
            "epoch": epoch,
            "model_pos": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "best_val": best_val,
            "best_epoch": best_epoch,
            "h5_path": str(h5_path),
            "train_end": train_end,
            "val_start": val_start,
            "hand_sides": HAND_SIDES,
            "coordinate_frame": COORDINATE_FRAME,
            "coordinate_alignment": coordinate_alignment,
            "source_landmark_space": source_landmark_space,
            "run_name": run_name,
            "init_checkpoint": (
                str(init_checkpoint) if init_checkpoint is not None else None
            ),
        },
        path,
    )


def _load_checkpoint(model, checkpoint_path: Path, device, expected_coordinate_alignment):
    if not checkpoint_path.is_file():
        raise FileNotFoundError(
            f"Initialization checkpoint was not found: {checkpoint_path}"
        )
    checkpoint = read_checkpoint(checkpoint_path, device)
    _require_coordinate_alignment(checkpoint, str(checkpoint_path), expected_coordinate_alignment)
    state_dict = (
        checkpoint["model_pos"]
        if isinstance(checkpoint, dict) and "model_pos" in checkpoint
        else checkpoint
    )
    model.load_state_dict(state_dict, strict=True)


def _create_collision_loss(side: str) -> CollisionLoss:
    return CollisionLoss(
        threshold=col_threshold,
        rb_dic=rb_dic,
        excluded_points=[0],
        excluded_pairs=excluded_pairs,
        hand_type=side,
    )


def _require_urdf(urdf_file: str | Path, side: str) -> str:
    path = Path(urdf_file)
    if not path.is_file():
        raise FileNotFoundError(
            f"{side.capitalize()} hand URDF was not found: {path}"
        )
    return str(path)


def _resolve_hand_urdf_paths(left_urdf, right_urdf):
    return {
        "left": _require_urdf(left_urdf, "left"),
        "right": _require_urdf(right_urdf, "right"),
    }


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _validate_args(args) -> None:
    if args.batch_size < 1:
        raise ValueError("batch_size must be positive")
    if args.epochs < 1:
        raise ValueError("epochs must be positive")
    if args.early_stopping_patience < 1:
        raise ValueError("early_stopping_patience must be positive")


def _setup_logging(path: Path) -> logging.Logger:
    path.parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("teleoperation.learning.trainer")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    logger.addHandler(logging.FileHandler(path, mode="w", encoding="utf-8"))
    return logger


from teleoperation.learning.trainer import _run_epoch, _masked_hand_loss, _tensor, _split_frame_ranges
from teleoperation.learning.reporting import _metrics_fieldnames, _epoch_metrics_row, _write_epoch_stats
from teleoperation.data.checkpoint import read_checkpoint, write_checkpoint
