"""Run one shared two-hand retargeting model on an offline H5 recording."""

from __future__ import annotations

import argparse
from pathlib import Path

import h5py
import numpy as np
import torch

from retargeting.config import ANGLE_LIMITS, DEFAULT_CHECKPOINT, L21
from retargeting.data import (
    HAND_SIDES,
    TwoHandH5ChunkedGenerator,
    TwoHandH5Dataset,
)
from retargeting.coordinates import validate_left_coordinate_mode
from retargeting.tracking import (
    DEFAULT_MAX_CENTER_DISPLACEMENT,
    DEFAULT_MAX_SHAPE_RMSE,
)
from retargeting.model import create_twohand_retargeter


def run(args: argparse.Namespace) -> int:
    device = _resolve_device(args.device)
    left_coordinate_mode = _resolve_coordinate_mode(
        args.left_coordinate_mode,
        args.checkpoint,
    )

    dataset = TwoHandH5Dataset(
        args.input,
        receptive_field=L21.model.receptive_field,
        scale_factor=args.scale_factor,
        reset_on_gaps=True,
        left_coordinate_mode=left_coordinate_mode,
        track_identity=not args.disable_identity_tracking,
        max_center_displacement=args.max_center_displacement,
        max_shape_rmse=args.max_shape_rmse,
    )
    retargeter = create_twohand_retargeter(
        model_kwargs=_model_kwargs(),
        device=device,
        checkpoint_path=str(args.checkpoint),
    )
    retargeter.eval()

    frame_count = dataset.frame_count
    angles = {
        side: np.zeros((frame_count, L21.model.output_joints), dtype=np.float32)
        for side in HAND_SIDES
    }
    valid = {
        side: np.zeros(frame_count, dtype=bool)
        for side in HAND_SIDES
    }
    angle_limits = np.asarray(ANGLE_LIMITS, dtype=np.float32)
    if angle_limits.shape != (L21.model.output_joints, 2):
        raise ValueError(f"Unexpected angle limit shape: {angle_limits.shape}")
    generator = TwoHandH5ChunkedGenerator(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
    )

    with torch.no_grad():
        for batch in generator.next_epoch():
            frame_indices = batch["frame_index"]
            for side in HAND_SIDES:
                side_mask = batch[f"{side}_valid"]
                if not side_mask.any():
                    continue
                hand_input = torch.from_numpy(
                    batch[f"{side}_input"][side_mask].astype(np.float32)
                ).to(device)
                prediction = retargeter.model(hand_input).detach().cpu().numpy()
                if prediction.shape != (
                    int(side_mask.sum()),
                    L21.model.output_joints,
                ):
                    raise ValueError(
                        f"Unexpected {side} model output shape: {prediction.shape}"
                    )
                selected_indices = frame_indices[side_mask]
                valid_rows = _valid_angle_rows(prediction, angle_limits)
                accepted_indices = selected_indices[valid_rows]
                angles[side][accepted_indices] = prediction[valid_rows]
                valid[side][accepted_indices] = True

    for side in HAND_SIDES:
        _hold_last_valid_angles(angles[side], valid[side])

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(output_path, "w") as h5_file:
        h5_file.create_dataset("frame_ids", data=dataset.frame_ids)
        h5_file.create_dataset("timestamps", data=dataset.timestamps)
        for side in HAND_SIDES:
            h5_file.create_dataset(f"{side}_angles", data=angles[side])
            h5_file.create_dataset(f"{side}_valid", data=valid[side])
        h5_file.attrs["input_file"] = str(args.input)
        h5_file.attrs["checkpoint"] = str(args.checkpoint)
        h5_file.attrs["output_shape"] = (L21.model.output_joints,)
        h5_file.attrs["left_coordinate_mode"] = left_coordinate_mode
        h5_file.attrs["identity_tracking"] = not args.disable_identity_tracking
        h5_file.attrs["max_center_displacement"] = args.max_center_displacement
        h5_file.attrs["max_shape_rmse"] = args.max_shape_rmse
        h5_file.attrs["invalid_angle_policy"] = "hold_previous"

    print(f"device={device}")
    print(f"input={args.input}")
    print(f"checkpoint={args.checkpoint}")
    print(f"left_coordinate_mode={left_coordinate_mode}")
    print(f"output={output_path}")
    print(f"frames={frame_count}")
    for side in HAND_SIDES:
        print(f"{side}_predictions={int(valid[side].sum())}")
        print(f"{side}_shape={angles[side].shape}")
    return 0


def _model_kwargs() -> dict:
    return L21.model_kwargs()


def _valid_angle_rows(
    angles: np.ndarray,
    limits: np.ndarray,
    tolerance: float = 1e-4,
) -> np.ndarray:
    finite = np.isfinite(angles).all(axis=1)
    within_lower = (angles >= limits[:, 0] - tolerance).all(axis=1)
    within_upper = (angles <= limits[:, 1] + tolerance).all(axis=1)
    return finite & within_lower & within_upper


def _hold_last_valid_angles(angles: np.ndarray, valid: np.ndarray) -> None:
    last_valid = None
    for frame_index in range(angles.shape[0]):
        if valid[frame_index]:
            last_valid = angles[frame_index].copy()
        elif last_valid is not None:
            angles[frame_index] = last_valid


def configure_parser(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("input/visual_hand_data_20260912_112108.h5"),
    )
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("output/twohand_angles.h5"),
    )
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument(
        "--scale-factor",
        type=float,
        default=L21.training.source_scale,
    )
    parser.add_argument(
        "--disable-identity-tracking",
        action="store_true",
        help="trust recorded left/right labels without temporal reassignment",
    )
    parser.add_argument(
        "--max-center-displacement",
        type=float,
        default=DEFAULT_MAX_CENTER_DISPLACEMENT,
        help="maximum normalized palm-center displacement between frames",
    )
    parser.add_argument(
        "--max-shape-rmse",
        type=float,
        default=DEFAULT_MAX_SHAPE_RMSE,
        help="maximum wrist-relative landmark RMSE between frames",
    )
    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "cuda"),
        default="auto",
    )
    parser.add_argument(
        "--left-coordinate-mode",
        choices=("auto", "none", "mirror_x"),
        default="auto",
        help="left-hand coordinate mode; auto reads the checkpoint metadata",
    )
    parser.set_defaults(handler=run)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Export LinkerHand L21 angles from a two-hand H5 recording"
    )
    configure_parser(parser)
    return run(parser.parse_args())


def _resolve_coordinate_mode(requested: str, checkpoint_path: Path) -> str:
    checkpoint_mode = _checkpoint_coordinate_mode(checkpoint_path)
    if requested == "auto":
        return checkpoint_mode
    explicit_mode = validate_left_coordinate_mode(requested)
    if explicit_mode != checkpoint_mode:
        print(
            "warning=left_coordinate_mode_mismatch "
            f"checkpoint={checkpoint_mode} requested={explicit_mode}"
        )
    return explicit_mode


def _checkpoint_coordinate_mode(checkpoint_path: Path) -> str:
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Checkpoint was not found: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    mode = (
        checkpoint.get("left_coordinate_mode")
        if isinstance(checkpoint, dict)
        else None
    )
    if mode is None:
        print(
            "warning=checkpoint_missing_left_coordinate_mode "
            "using=none"
        )
        return "none"
    return validate_left_coordinate_mode(mode)


def _resolve_device(requested: str) -> torch.device:
    if requested == "auto":
        requested = "cuda" if torch.cuda.is_available() else "cpu"
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    return torch.device(requested)


if __name__ == "__main__":
    raise SystemExit(main())
