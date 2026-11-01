from __future__ import annotations
import torch
import torch.nn as nn
from teleoperation.data.checkpoint import read_checkpoint
from teleoperation.contracts.coordinates import (
    COORDINATE_ALIGNMENT, validate_coordinate_alignment,
    validate_landmark_contract,
    WORLD_PALM_LOCAL_COORDINATE_ALIGNMENT, CANONICAL_ANGLE_CONVENTION,
)
from teleoperation.contracts.constants import HAND_SIDES


def _reference_geometry(spec):
    return {'root_link':spec.root_link,'palm_frames':list(spec.palm_frames),
            'tip_frames':list(spec.tip_frames),'human_tip_indices':list(spec.human_tip_indices),
            'reference_pose':'clipped_zero_within_effective_bounds'}


def joint_checkpoint_payload(model, spec):
    from teleoperation.contracts.coordinates import palm_length_metadata
    return {
        "schema": "hand_joint_nn_v1", "hand_model": spec.hand_model,
        "side": spec.side, "spec_hash": spec.spec_hash,
        "joint_names": list(spec.independent_joint_names),
        "output_units": "rad", "output_coordinates": "independent_native",
        "reference_geometry": _reference_geometry(spec),
        **palm_length_metadata(),
        "model_kwargs": dict(model.model_kwargs), "state_dict": model.state_dict(),
    }


def shared_joint_checkpoint_payload(model, spec):
    """Serialize one canonical right-hand model for both runtime sides."""
    from teleoperation.contracts.coordinates import palm_length_metadata
    if spec.side != "right":
        raise ValueError("Shared joint checkpoints require the right canonical target spec")
    return {
        "schema": "hand_joint_nn_v1", "hand_model": spec.hand_model,
        "side": "shared", "hand_sides": ["left", "right"],
        "spec_hash": spec.spec_hash,
        "joint_names": list(spec.independent_joint_names),
        "output_units": "rad", "output_coordinates": "canonical_right",
        "training_robot": "right", "angle_convention": CANONICAL_ANGLE_CONVENTION,
        "reference_geometry": _reference_geometry(spec),
        **palm_length_metadata(),
        "model_kwargs": dict(model.model_kwargs), "state_dict": model.state_dict(),
    }


def save_joint_checkpoint(path, payload):
    from teleoperation.data.transaction import atomic_output
    with atomic_output(path) as temporary:
        torch.save(payload, temporary)


def load_joint_checkpoint(path, spec, device="cpu"):
    from .predictor import build_joint_model
    from teleoperation.contracts.coordinates import validate_palm_length_contract
    payload = torch.load(path, map_location=device, weights_only=True)
    expected = {
        "schema": "hand_joint_nn_v1", "hand_model": spec.hand_model,
        "side": spec.side, "spec_hash": spec.spec_hash,
        "joint_names": list(spec.independent_joint_names),
        "output_units": "rad", "output_coordinates": "independent_native",
        "reference_geometry": _reference_geometry(spec),
    }
    if not isinstance(payload, dict):
        raise ValueError("Joint checkpoint must contain a metadata dictionary")
    for key, value in expected.items():
        if payload.get(key) != value:
            raise ValueError(f"Checkpoint {path}: incompatible {key}")
    validate_palm_length_contract(payload, f"Checkpoint {path}")
    kwargs = payload.get("model_kwargs")
    if not isinstance(kwargs, dict):
        raise ValueError("Joint checkpoint requires explicit model_kwargs")
    model = build_joint_model(spec, kwargs).to(device)
    if model.model_kwargs != kwargs:
        raise ValueError("Checkpoint architecture/output limits do not match target spec")
    expected_state = model.state_dict()
    for name in ('angle_min','angle_max','angle_range','angle_center'):
        key = 'angle_clamper.'+name
        actual = payload['state_dict'].get(key)
        if not isinstance(actual,torch.Tensor) or not torch.equal(actual,expected_state[key]):
            raise ValueError('Checkpoint output limits differ from target specification')
    model.load_state_dict(payload["state_dict"], strict=True)
    return model.eval(), payload


def load_shared_joint_checkpoint(path, spec, device="cpu"):
    """Load a shared canonical checkpoint using the right-hand target spec."""
    from .predictor import build_joint_model
    from teleoperation.contracts.coordinates import validate_palm_length_contract
    if spec.side != "right":
        raise ValueError("Shared joint checkpoints require the right canonical target spec")
    payload = torch.load(path, map_location=device, weights_only=True)
    expected = {
        "schema": "hand_joint_nn_v1", "hand_model": spec.hand_model,
        "side": "shared", "hand_sides": ["left", "right"],
        "spec_hash": spec.spec_hash,
        "joint_names": list(spec.independent_joint_names),
        "output_units": "rad", "output_coordinates": "canonical_right",
        "training_robot": "right", "angle_convention": CANONICAL_ANGLE_CONVENTION,
        "reference_geometry": _reference_geometry(spec),
    }
    if not isinstance(payload, dict):
        raise ValueError("Shared joint checkpoint must contain a metadata dictionary")
    for key, value in expected.items():
        if payload.get(key) != value:
            raise ValueError(f"Checkpoint {path}: incompatible shared {key}")
    validate_palm_length_contract(payload, f"Checkpoint {path}")
    kwargs = payload.get("model_kwargs")
    if not isinstance(kwargs, dict):
        raise ValueError("Shared joint checkpoint requires explicit model_kwargs")
    model = build_joint_model(spec, kwargs).to(device)
    if model.model_kwargs != kwargs:
        raise ValueError("Checkpoint architecture/output limits do not match target spec")
    expected_state = model.state_dict()
    for name in ('angle_min','angle_max','angle_range','angle_center'):
        key = 'angle_clamper.' + name
        actual = payload['state_dict'].get(key)
        if not isinstance(actual, torch.Tensor) or not torch.equal(actual, expected_state[key]):
            raise ValueError('Shared checkpoint output limits differ from target specification')
    model.load_state_dict(payload["state_dict"], strict=True)
    return model.eval(), payload
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
    validate_landmark_contract(alignment, checkpoint, f"Checkpoint {checkpoint_path}")
    if alignment == WORLD_PALM_LOCAL_COORDINATE_ALIGNMENT:
        if (checkpoint.get("hand_side") != "shared" or checkpoint.get("angle_convention") != CANONICAL_ANGLE_CONVENTION
                or checkpoint.get("training_robot") != "right"):
            raise ValueError("Canonical checkpoint requires hand_side='shared', training_robot='right' and the canonical angle_convention")


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
