from __future__ import annotations
import torch
import torch.nn as nn
from teleoperation.data.checkpoint import read_checkpoint
from teleoperation.contracts.coordinates import COORDINATE_ALIGNMENT, validate_coordinate_alignment
def load_hand_checkpoint(
    model: nn.Module,
    checkpoint_path: str,
    device: torch.device | str,
    side: str | None = None,
    strict: bool = True,
    expected_coordinate_alignment: str | None = None,
) -> nn.Module:
    """Load a checkpoint saved by main_train.py into one hand model."""
    checkpoint = read_checkpoint(checkpoint_path, device)
    _require_coordinate_alignment(checkpoint, checkpoint_path, expected_coordinate_alignment)
    state_dict = _state_dict_for_side(checkpoint, side)
    model.load_state_dict(state_dict, strict=strict)
    return model


def _require_coordinate_alignment(
    checkpoint: object, checkpoint_path: str,
    expected_coordinate_alignment: str | None = None,
) -> None:
    # Old realtime/legacy callers retain their original fixed-coordinate guard.
    expected = validate_coordinate_alignment(
        COORDINATE_ALIGNMENT if expected_coordinate_alignment is None else expected_coordinate_alignment,
        "Expected input",
    )
    alignment = validate_coordinate_alignment((
        checkpoint.get("coordinate_alignment")
        if isinstance(checkpoint, dict)
        else None
    ), f"Checkpoint {checkpoint_path}")
    if alignment != expected:
        raise ValueError(
            f"Checkpoint coordinate alignment mismatch ({checkpoint_path}): "
            f"input={expected!r}; checkpoint={alignment!r}."
        )


def _state_dict_for_side(checkpoint: dict, side: str | None) -> dict:
    if "model_pos" in checkpoint:
        return checkpoint["model_pos"]

    if side is not None:
        side_key = f"{side}_model_pos"
        if side_key in checkpoint:
            return checkpoint[side_key]

    if "left_model_pos" in checkpoint or "right_model_pos" in checkpoint:
        raise ValueError("Two-hand checkpoints require side='left' or side='right'")

    return checkpoint
