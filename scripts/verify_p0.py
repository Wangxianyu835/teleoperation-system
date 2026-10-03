"""Read-only checkpoint inference and bilateral FK/loss backward audit.

No optimizer or training epoch is created. Existing checkpoint and input bytes
are hashed before and after the audit. Results are emitted as JSON on stdout.
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import io
import json
from pathlib import Path
import sys

import numpy as np
import torch
import torch.nn.functional as F

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from model.kinematics import create_hand_kinematics
from model.losses import CollisionLoss, hand_loss
from retargeting.config import ANGLE_LIMITS, EXCLUDED_COLLISION_PAIRS, L21, ROBOT_JOINTS, SOURCE_JOINTS
from retargeting.data import TwoHandH5Dataset
from retargeting.inference import _valid_angle_rows
from retargeting.model import create_twohand_retargeter
from retargeting.training import LOSS_NAMES


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def verify(input_path: Path, checkpoint_path: Path, sample_count: int = 3) -> dict:
    if sample_count < 1:
        raise ValueError("sample_count must be positive")
    before = {str(path): sha256(path) for path in (input_path, checkpoint_path)}
    dataset = TwoHandH5Dataset(input_path, scale_factor=L21.training.source_scale)
    retargeter = create_twohand_retargeter(L21.model_kwargs(), "cpu", str(checkpoint_path)).eval()
    report = {
        "checkpoint": str(checkpoint_path), "input": str(input_path),
        "checkpoint_loaded": True, "raw_frames": dataset.frame_count,
        "valid_window_counts": dataset.side_counts(), "device": "cpu",
        "training_performed": False, "optimizer_created": False,
        "physical_scale_status": "PHYSICAL SCALE NOT VERIFIED",
        "historical_total_loss_comparable": False, "sides": {},
    }
    for side in ("left", "right"):
        valid = [sample for sample in dataset.samples if sample[f"{side}_valid"]]
        if not valid:
            raise ValueError(f"No valid {side} hand windows")
        selected = [valid[i] for i in np.linspace(0, len(valid) - 1, min(sample_count, len(valid)), dtype=int)]
        inputs = torch.from_numpy(np.stack([sample[f"{side}_input"] for sample in selected]))
        targets = torch.from_numpy(np.stack([sample[f"{side}_target"] for sample in selected]))
        retargeter.zero_grad(set_to_none=True)
        prediction = retargeter.model(inputs)
        if prediction.shape != (len(selected), 18):
            raise RuntimeError(f"Unexpected {side} prediction shape: {prediction.shape}")
        values = prediction.detach().numpy()
        if not _valid_angle_rows(values, np.asarray(ANGLE_LIMITS)).all():
            raise RuntimeError(f"Non-finite or out-of-limit {side} angles")
        with contextlib.redirect_stdout(io.StringIO()):
            fk = create_hand_kinematics(
                getattr(L21, f"{side}_urdf"), L21.hand_kinematics_config(), "cpu",
                scale_factor=L21.training.robot_scale,
            )
        padded = F.pad(prediction, (0, 5))
        positions = fk.forward(padded)[2]
        if not torch.isfinite(positions).all():
            raise RuntimeError(f"Non-finite {side} FK positions")
        collision = CollisionLoss(
            L21.training.collision_threshold, ROBOT_JOINTS,
            excluded_points=[0], excluded_pairs=EXCLUDED_COLLISION_PAIRS,
            hand_type=side,
        )
        losses = hand_loss(
            padded, targets, ROBOT_JOINTS, SOURCE_JOINTS,
            torch.nn.MSELoss(), None, collision,
            hand_fk_model=fk, loss_weight=L21.training.loss_weights, hand_side=side,
        )
        if not all(torch.isfinite(loss).all() for loss in losses):
            raise RuntimeError(f"Non-finite {side} loss")
        losses[0].backward()
        gradients = [parameter.grad for parameter in retargeter.parameters() if parameter.grad is not None]
        if not gradients or not all(torch.isfinite(gradient).all() for gradient in gradients):
            raise RuntimeError(f"Missing or non-finite {side} model gradients")
        report["sides"][side] = {
            "frame_indices": [sample["frame_index"] for sample in selected],
            "input_shape": list(inputs.shape), "output_shape": list(prediction.shape),
            "fk_shape": list(positions.shape), "nonfinite_angles": 0,
            "joint_limit_violations": 0, "fixed_root_zero": bool((values[:, 0] == 0).all()),
            "finite_fk": True, "finite_losses": True, "finite_sampled_gradients": True,
            "parameter_gradients_checked": len(gradients),
            "max_abs_gradient": max(float(gradient.abs().max()) for gradient in gradients),
            "weighted_loss_components": {name: float(loss.detach()) for name, loss in zip(LOSS_NAMES, losses)},
        }
    after = {str(path): sha256(path) for path in (input_path, checkpoint_path)}
    if before != after:
        raise RuntimeError("Checkpoint or input bytes changed during the audit")
    report["sha256"] = after
    report["protected_files_unchanged"] = True
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--sample-count", type=int, default=3)
    args = parser.parse_args()
    torch.set_num_threads(2)
    print(json.dumps(verify(args.input, args.checkpoint, args.sample_count), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
