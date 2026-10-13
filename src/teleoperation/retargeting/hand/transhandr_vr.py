"""Strict upstream checkpoint loading and compatible typed VR model results."""
from pathlib import Path

import numpy as np
import torch

from teleoperation.contracts.coordinates import COORDINATE_ALIGNMENT
from teleoperation.contracts.hand import HandWindow, HandRetargetResult, L21HandAngles
from teleoperation.data.checkpoint import read_checkpoint
from teleoperation.retargeting.hand.config import L21
from teleoperation.retargeting.hand.interface import HandRetargeter
from teleoperation.retargeting.hand.transformer import PoseTransformer


def load_transhandr_vr_model(checkpoint, device="cpu"):
    """An explicit upstream profile, without rewriting or guessing camera metadata."""
    checkpoint = Path(checkpoint)
    if not checkpoint.is_file():
        raise FileNotFoundError(f"TransHandR VR checkpoint not found: {checkpoint}")
    payload = read_checkpoint(str(checkpoint), device)
    if not isinstance(payload, dict) or "model_pos" not in payload:
        raise ValueError("TransHandR checkpoint must contain model_pos")
    alignment = payload.get("coordinate_alignment")
    if alignment is not None and alignment != COORDINATE_ALIGNMENT:
        raise ValueError(f"VR checkpoint alignment mismatch: {alignment!r}; expected {COORDINATE_ALIGNMENT!r}")
    if payload.get("hand_side") is not None or payload.get("angle_convention") is not None:
        raise ValueError("Use an upstream TransHandR VR checkpoint, not camera/shared-side weights")
    model = PoseTransformer(**L21.model_kwargs()).to(device)
    # Unlike upstream strict=False, prevent silently leaving random parameters.
    # The current model strictly loads the actual upstream thumb3 checkpoint.
    model.load_state_dict(payload["model_pos"], strict=True)
    return model.eval()


class TransHandRVRRetargeter(HandRetargeter):
    """The upstream input tensor and output angles, wrapped in the public interface.

The original worker takes left_fingers and drives the right L21 URDF model.
Source and destination are explicit so those two meanings cannot be confused.
"""

    def __init__(self, model, device="cpu", source_side="left", target_side="right"):
        if source_side not in ("left", "right") or target_side not in ("left", "right"):
            raise ValueError("VR source and target hand must be left or right")
        self.model, self.device = model, device
        self.source_side, self.target_side = source_side, target_side

    def reset(self):
        pass  # Upstream inference is stateless; its window belongs to the producer.

    @torch.no_grad()
    def retarget(self, window):
        if not isinstance(window, HandWindow) or window.source != "visionpro":
            raise TypeError("VR retargeter requires a Vision Pro HandWindow")
        if window.metadata.get("coordinate_alignment") != COORDINATE_ALIGNMENT:
            raise ValueError("VR window coordinate alignment mismatch")
        if window.metadata.get("vr_preprocessing") != "transhandr_axis_wrist_three_frame_v1":
            raise ValueError("VR retargeter requires the explicit upstream preprocessing profile")
        hands = {"left": None, "right": None}
        data = window.hands[self.source_side]
        if data is not None:
            if data.shape != (3, 25, 3) or not np.isfinite(data).all():
                raise ValueError("VR model input must be a finite (3,25,3) window")
            input_tensor = torch.from_numpy(data).float().unsqueeze(0).to(self.device)
            result = self.model(input_tensor).detach().cpu().numpy()[0]
            if result.shape != (18,) or not np.isfinite(result).all():
                raise ValueError("VR model must output 18 finite angles")
            hands[self.target_side] = L21HandAngles(result)
        return HandRetargetResult(left_angles=hands["left"], right_angles=hands["right"],
                                  left_valid=hands["left"] is not None, right_valid=hands["right"] is not None,
                                  timestamp=window.timestamp)
