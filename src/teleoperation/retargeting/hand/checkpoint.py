from __future__ import annotations
import torch
import torch.nn as nn
from teleoperation.data.checkpoint import read_checkpoint
from teleoperation.contracts.coordinates import COORDINATE_ALIGNMENT, validate_coordinate_alignment
from teleoperation.contracts.constants import HAND_SIDES
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
    hand_side = _require_hand_side(checkpoint, side, checkpoint_path)
    state_dict = _state_dict_for_side(checkpoint, side)
    model.load_state_dict(state_dict, strict=strict)
    # Runtime-only contract, not a parameter or a state_dict/metadata rewrite.
    model.coordinate_alignment = (COORDINATE_ALIGNMENT if expected_coordinate_alignment is None
                                  else validate_coordinate_alignment(expected_coordinate_alignment, "Expected input"))
    model.hand_side = hand_side
    return model


def _require_hand_side(checkpoint: dict, side: str | None, checkpoint_path: str) -> str | None:
    """Reject a dedicated checkpoint routed to the wrong hand or shared model."""
    if side is not None and side not in HAND_SIDES:
        raise ValueError(f"Unknown hand side: {side!r}")
    declared = checkpoint.get("hand_side")
    sides = checkpoint.get("hand_sides")
    if declared is not None and declared not in (*HAND_SIDES, "shared"):
        raise ValueError(f"Invalid hand_side in checkpoint {checkpoint_path}: {declared!r}")
    if sides is not None:
        if not isinstance(sides, (list, tuple)) or not sides or any(s not in HAND_SIDES for s in sides) or len(set(sides)) != len(sides):
            raise ValueError(f"Invalid hand_sides in checkpoint {checkpoint_path}: {sides!r}")
        expected_sides = HAND_SIDES if declared == "shared" else (declared,)
        if declared is not None and set(sides) != set(expected_sides):
            raise ValueError(f"Inconsistent hand_side/hand_sides in checkpoint {checkpoint_path}")
        if declared is None and len(sides) == 1:
            declared = sides[0]
    if declared in HAND_SIDES and declared != side:
        destination = "shared left/right model" if side is None else f"{side} model"
        raise ValueError(f"Checkpoint hand side mismatch ({checkpoint_path}): checkpoint={declared!r}; destination={destination}")
    return declared


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
