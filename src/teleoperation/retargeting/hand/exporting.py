"""Batch inference algorithms, independent of Dataset and H5."""
import numpy as np
import torch
from .config import ANGLE_LIMITS
from teleoperation.contracts.constants import HAND_SIDES


def predict_batches(retargeter, batches, frame_count, device):
    angles = {
        side: np.zeros((frame_count, 18), dtype=np.float32)
        for side in HAND_SIDES
    }
    valid = {
        side: np.zeros(frame_count, dtype=bool)
        for side in HAND_SIDES
    }
    angle_limits = np.asarray(ANGLE_LIMITS, dtype=np.float32)
    if angle_limits.shape != (18, 2):
        raise ValueError(f"Unexpected angle limit shape: {angle_limits.shape}")

    with torch.no_grad():
        for batch in batches:
            frame_indices = batch["frame_index"]
            for side in HAND_SIDES:
                side_mask = batch[f"{side}_valid"]
                if not side_mask.any():
                    continue
                hand_input = torch.from_numpy(
                    batch[f"{side}_input"][side_mask].astype(np.float32)
                ).to(device)
                prediction = retargeter.model_for_side(side)(hand_input).detach().cpu().numpy()
                if prediction.shape != (
                    int(side_mask.sum()),
                    18,
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


    return angles, valid

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
