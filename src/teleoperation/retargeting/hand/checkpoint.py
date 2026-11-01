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
