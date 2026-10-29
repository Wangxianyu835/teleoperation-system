"""Target-native independent joint results for every hand retargeting method."""

from __future__ import annotations

from abc import ABC, abstractmethod
from time import perf_counter
from typing import Any

import numpy as np

from teleoperation.contracts.hand import HandWindow
from teleoperation.contracts.hand_control import HandKey, HandJointResult
from .angles import l21_joint_result


class HandRetargeter(ABC):
    """Common boundary between declared hand windows and target joint results."""

    @abstractmethod
    def reset(self) -> None:
        """Reset retargeter state."""

    @abstractmethod
    def retarget(self, hand_window: HandWindow) -> dict[HandKey, HandJointResult]:
        """Solve each configured target, reporting invalid targets explicitly."""


class PoseTransformerRetargeter(HandRetargeter):
    """Adapt the existing :class:`TwoHandRetargeter` to ``HandRetargeter``.

    The wrapped retargeter owns the existing PoseTransformer invocation.  This
    adapter converts historical model output to named native L21 independent
    coordinates. It preserves the checkpoint's declared input and side rules.
    """

    def __init__(
        self,
        retargeter: Any | None = None,
        device: Any = "cpu",
        *,
        model: Any | None = None,
        two_hand_retargeter: Any | None = None,
    ) -> None:
        supplied = [
            value
            for value in (retargeter, model, two_hand_retargeter)
            if value is not None
        ]
        if len(supplied) != 1:
            raise TypeError(
                "provide exactly one of retargeter, model, or two_hand_retargeter"
            )
        retargeter = supplied[0]
        # Accept a raw PoseTransformer as a convenience while keeping the
        # official path centered on the existing TwoHandRetargeter wrapper.
        if not hasattr(retargeter, "predict"):
            from teleoperation.retargeting.hand.predictor import TwoHandRetargeter

            retargeter = TwoHandRetargeter(retargeter)
        self._retargeter = retargeter
        self.device = device

    @property
    def retargeter(self) -> Any:
        """Return the wrapped existing retargeter for inspection/integration."""

        return self._retargeter

    def reset(self) -> None:
        """Reset the adapter.

        The current PoseTransformer path is stateless, so reset intentionally
        performs no model mutation and preserves its existing behavior.
        """

    def retarget(self, hand_window: HandWindow) -> dict[HandKey, HandJointResult]:
        if not isinstance(hand_window, HandWindow):
            raise TypeError("hand_window must be a HandWindow")

        output = self._retargeter.predict(
            hand_window.to_payload(),
            device=self.device,
        )
        hands = output["hands"]
        return {("l21", side): l21_joint_result(side, hands.get(side), hand_window.timestamp)
                for side in ("left", "right")}


class JointNNRetargeter(HandRetargeter):
    """Target-specific models returning native independent coordinates directly."""

    def __init__(self, specs, models, device="cpu"):
        if not specs or set(specs) != set(models):
            raise ValueError("Provide a model for every target spec")
        self.specs, self.models, self.device = dict(specs), dict(models), device
        for key, spec in specs.items():
            if key != spec.key:
                raise ValueError("Target key/spec mismatch")
            self.models[key].to(device).eval()

    def reset(self):
        """Inference is stateless; windows are owned by the producer."""

    def retarget(self, window):
        import torch
        from teleoperation.contracts.coordinates import validate_palm_length_contract
        if not isinstance(window, HandWindow):
            raise TypeError("NN retargeting requires HandWindow")
        validate_palm_length_contract(window.metadata, "Joint NN")
        results = {}
        with torch.no_grad():
            for key, spec in self.specs.items():
                points = window.hands[spec.side]
                values, error, elapsed = None, None, 0.
                if points is not None:
                    if points.shape != (3,25,3) or not np.isfinite(points).all():
                        raise ValueError("Joint NN requires a finite normalized (3,25,3) window")
                    started = perf_counter()
                    tensor = torch.as_tensor(np.array(points), dtype=torch.float32, device=self.device).unsqueeze(0)
                    values = self.models[key](tensor).detach().cpu().numpy()[0]
                    elapsed = (perf_counter()-started)*1000
                    if values.shape != (spec.dof,) or not np.isfinite(values).all():
                        values, error = None, "invalid model output"
                else:
                    error = "missing/invalid input"
                results[key] = HandJointResult(*key,spec.spec_hash,spec.independent_joint_names,
                                               values,window.timestamp,values is not None,error,elapsed)
        return results


__all__ = ["HandRetargeter", "PoseTransformerRetargeter", "JointNNRetargeter"]
