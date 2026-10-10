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
    """Route each hand to a shared model or its own independently trained model."""

    def __init__(self, model: nn.Module | None = None, expected_coordinate_alignment: str | None = None,
                 *, left_model: nn.Module | None = None, right_model: nn.Module | None = None):
        super().__init__()
        independent = left_model is not None or right_model is not None
        if independent and (model is not None or left_model is None or right_model is None):
            raise ValueError("Provide a shared model or both left_model and right_model")
        if not independent and model is None:
            raise ValueError("A shared model or both hand models are required")
        if independent:
            # Detect both reused modules and shared parameter storage.
            left_storage = {(p.device, p.untyped_storage().data_ptr()) for p in left_model.parameters() if p.numel()}
            right_storage = {(p.device, p.untyped_storage().data_ptr()) for p in right_model.parameters() if p.numel()}
            if left_model is right_model or left_storage & right_storage:
                raise ValueError("Independent hand models must not share parameters")
        self.model = model
        self.left_model = left_model
        self.right_model = right_model
        expected = (None if expected_coordinate_alignment is None else
                    validate_coordinate_alignment(expected_coordinate_alignment, "Expected input"))
        alignments = set()
        for side, hand_model in ((("left", left_model), ("right", right_model)) if independent else ((None, model),)):
            _require_hand_side({"hand_side": getattr(hand_model, "hand_side", None)}, side, "Loaded model")
            loaded = getattr(hand_model, "coordinate_alignment", None)
            if loaded is not None:
                loaded = validate_coordinate_alignment(loaded, "Loaded model")
                alignments.add(loaded)
                if expected is not None and loaded != expected:
                    raise ValueError(f"Model coordinate alignment mismatch: input={expected!r}; model={loaded!r}")
        if len(alignments) > 1:
            raise ValueError("Left/right model coordinate alignment mismatch")
        self.coordinate_alignment = expected if expected is not None else next(iter(alignments), None)

    def model_for_side(self, side: str) -> nn.Module:
        if side not in HAND_SIDES:
            raise ValueError(f"Unknown hand side: {side!r}")
        return self.model if self.model is not None else getattr(self, f"{side}_model")

    def forward(
        self,
        left_hand: torch.Tensor | None = None,
        right_hand: torch.Tensor | None = None,
    ) -> dict[str, dict[str, torch.Tensor]]:
        """Retarget whichever batched hand tensors are present."""
        outputs = {}
        if left_hand is not None:
            outputs["left"] = self.model_for_side("left")(left_hand)
        if right_hand is not None:
            outputs["right"] = self.model_for_side("right")(right_hand)
        return {"hands": outputs}

    @torch.no_grad()
    def predict(self, retarget_input: dict[str, Any], device: torch.device | str) -> dict[str, Any]:
        """Retarget one realtime input payload using numpy arrays.

        Expected hand input shape is (3, 25, 3) for each side. Returned hand
        angles are numpy arrays with shape (hand_dof,).
        """
        validate_retarget_input(retarget_input, allow_missing_hands=True)
        if self.coordinate_alignment is not None:
            alignment = validate_coordinate_alignment(
                retarget_input.get("metadata", {}).get("coordinate_alignment"), "HandWindow",
            )
            if alignment != self.coordinate_alignment:
                raise ValueError(f"HandWindow coordinate alignment mismatch: input={alignment!r}; model={self.coordinate_alignment!r}")
        self.eval()

        outputs = {"hands": {}}
        for side in HAND_SIDES:
            hand_data = retarget_input["hands"][side]
            if hand_data is None:
                outputs["hands"][side] = None
                continue
            hand_tensor = _to_single_batch_tensor(hand_data, device)
            hand_output = self.model_for_side(side)(hand_tensor)
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
    """Create a shared model or independently load a complete checkpoint pair."""
    from teleoperation.contracts.checkpoints import validate_checkpoint_selection

    validate_checkpoint_selection(checkpoint_path, left_checkpoint, right_checkpoint)
    if left_checkpoint is not None:
        models = {}
        for side, path in (("left", left_checkpoint), ("right", right_checkpoint)):
            hand_model = build_hand_model(model_kwargs).to(device)
            load_hand_checkpoint(hand_model, path, device, side=side, strict=strict,
                                 expected_coordinate_alignment=expected_coordinate_alignment)
            models[side] = hand_model
        return TwoHandRetargeter(left_model=models["left"], right_model=models["right"],
                                 expected_coordinate_alignment=expected_coordinate_alignment).to(device)
    model = build_hand_model(model_kwargs).to(device)

    if checkpoint_path is not None:
        load_hand_checkpoint(model, checkpoint_path, device, strict=strict,
                             expected_coordinate_alignment=expected_coordinate_alignment)

    return TwoHandRetargeter(model=model, expected_coordinate_alignment=expected_coordinate_alignment).to(device)


def _to_single_batch_tensor(hand_data: np.ndarray, device: torch.device | str) -> torch.Tensor:
    return torch.from_numpy(hand_data.astype(np.float32)).unsqueeze(0).to(device)


from .checkpoint import load_hand_checkpoint, _require_coordinate_alignment, _state_dict_for_side, _require_hand_side
