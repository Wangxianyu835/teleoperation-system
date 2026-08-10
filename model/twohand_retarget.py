"""Two-hand retargeting wrapper built on the existing PoseTransformer."""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
import torch.nn as nn

from config.retarget_io import HAND_SIDES, validate_retarget_input
from model.model_poseformer import PoseTransformer


def build_hand_model(model_kwargs: dict[str, Any]) -> PoseTransformer:
    """Create one hand PoseTransformer with the existing model implementation."""
    return PoseTransformer(**model_kwargs)


def load_hand_checkpoint(
    model: nn.Module,
    checkpoint_path: str,
    device: torch.device | str,
    strict: bool = True,
) -> nn.Module:
    """Load a checkpoint saved by main_train.py into one hand model."""
    checkpoint = torch.load(checkpoint_path, map_location=device)
    state_dict = checkpoint["model_pos"] if "model_pos" in checkpoint else checkpoint
    model.load_state_dict(state_dict, strict=strict)
    return model


class TwoHandRetargeter(nn.Module):
    """Run separate left and right hand retargeting models.

    The wrapper deliberately keeps both sides explicit. This makes it easy to
    use separate left/right checkpoints later, while still allowing the first
    version to reuse the same checkpoint for both sides.
    """

    def __init__(self, left_model: nn.Module, right_model: nn.Module):
        super().__init__()
        self.models = nn.ModuleDict(
            {
                "left": left_model,
                "right": right_model,
            }
        )

    def forward(
        self,
        left_hand: torch.Tensor,
        right_hand: torch.Tensor,
    ) -> dict[str, dict[str, torch.Tensor]]:
        """Retarget batched hand tensors shaped (batch, frames, 25, 3)."""
        return {
            "hands": {
                "left": self.models["left"](left_hand),
                "right": self.models["right"](right_hand),
            }
        }

    @torch.no_grad()
    def predict(self, retarget_input: dict[str, Any], device: torch.device | str) -> dict[str, Any]:
        """Retarget one realtime input payload using numpy arrays.

        Expected hand input shape is (3, 25, 3) for each side. Returned hand
        angles are numpy arrays with shape (hand_dof,).
        """
        validate_retarget_input(retarget_input, allow_missing_hands=False)
        self.eval()

        outputs = {"hands": {}}
        for side in HAND_SIDES:
            hand_tensor = _to_single_batch_tensor(retarget_input["hands"][side], device)
            hand_output = self.models[side](hand_tensor)
            outputs["hands"][side] = hand_output.detach().cpu().numpy()[0]

        return outputs


def create_twohand_retargeter(
    model_kwargs: dict[str, Any],
    device: torch.device | str,
    left_checkpoint: str | None = None,
    right_checkpoint: str | None = None,
    strict: bool = True,
) -> TwoHandRetargeter:
    """Create a two-hand wrapper and optionally load side-specific checkpoints."""
    left_model = build_hand_model(model_kwargs).to(device)
    right_model = build_hand_model(model_kwargs).to(device)

    if left_checkpoint is not None:
        load_hand_checkpoint(left_model, left_checkpoint, device, strict=strict)
    if right_checkpoint is not None:
        load_hand_checkpoint(right_model, right_checkpoint, device, strict=strict)

    return TwoHandRetargeter(left_model=left_model, right_model=right_model).to(device)


def _to_single_batch_tensor(hand_data: np.ndarray, device: torch.device | str) -> torch.Tensor:
    return torch.from_numpy(hand_data.astype(np.float32)).unsqueeze(0).to(device)
