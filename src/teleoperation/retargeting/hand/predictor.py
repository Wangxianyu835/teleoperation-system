"""Two-hand retargeting wrapper built on the existing PoseTransformer."""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
import torch.nn as nn

from teleoperation.contracts.validation import HAND_SIDES, validate_retarget_input
from teleoperation.contracts.coordinates import COORDINATE_ALIGNMENT, validate_coordinate_alignment


def build_hand_model(model_kwargs: dict[str, Any]) -> nn.Module:
    """Create one hand PoseTransformer with the existing model implementation."""
    from teleoperation.retargeting.hand.transformer import PoseTransformer

    return PoseTransformer(**model_kwargs)


class TwoHandRetargeter(nn.Module):
    """Run one shared retargeting model for either hand side."""

    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model

    def forward(
        self,
        left_hand: torch.Tensor | None = None,
        right_hand: torch.Tensor | None = None,
    ) -> dict[str, dict[str, torch.Tensor]]:
        """Retarget whichever batched hand tensors are present."""
        outputs = {}
        if left_hand is not None:
            outputs["left"] = self.model(left_hand)
        if right_hand is not None:
            outputs["right"] = self.model(right_hand)
        return {"hands": outputs}

    @torch.no_grad()
    def predict(self, retarget_input: dict[str, Any], device: torch.device | str) -> dict[str, Any]:
        """Retarget one realtime input payload using numpy arrays.

        Expected hand input shape is (3, 25, 3) for each side. Returned hand
        angles are numpy arrays with shape (hand_dof,).
        """
        validate_retarget_input(retarget_input, allow_missing_hands=True)
        self.eval()

        outputs = {"hands": {}}
        for side in HAND_SIDES:
            hand_data = retarget_input["hands"][side]
            if hand_data is None:
                outputs["hands"][side] = None
                continue
            hand_tensor = _to_single_batch_tensor(hand_data, device)
            hand_output = self.model(hand_tensor)
            outputs["hands"][side] = hand_output.detach().cpu().numpy()[0]

        return outputs


def create_twohand_retargeter(
    model_kwargs: dict[str, Any],
    device: torch.device | str,
    checkpoint_path: str | None = None,
    left_checkpoint: str | None = None,
    right_checkpoint: str | None = None,
    strict: bool = True,
    expected_coordinate_alignment: str | None = None,
) -> TwoHandRetargeter:
    """Create a shared two-hand model and optionally load its checkpoint."""
    model = build_hand_model(model_kwargs).to(device)

    if checkpoint_path is not None:
        load_hand_checkpoint(model, checkpoint_path, device, strict=strict,
                             expected_coordinate_alignment=expected_coordinate_alignment)
    elif left_checkpoint is not None or right_checkpoint is not None:
        selected_checkpoint = left_checkpoint or right_checkpoint
        load_hand_checkpoint(model, selected_checkpoint, device, strict=strict,
                             expected_coordinate_alignment=expected_coordinate_alignment)

    return TwoHandRetargeter(model=model).to(device)


def _to_single_batch_tensor(hand_data: np.ndarray, device: torch.device | str) -> torch.Tensor:
    return torch.from_numpy(hand_data.astype(np.float32)).unsqueeze(0).to(device)


from .checkpoint import load_hand_checkpoint, _require_coordinate_alignment, _state_dict_for_side
