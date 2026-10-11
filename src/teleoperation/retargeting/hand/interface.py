"""Unified hand retargeting interface and model adapters.

The interface stops at the L21 model result.  Robot-specific
transport, action packing, and simulator integration remain outside this
module.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

import numpy as np

from teleoperation.contracts.hand import HandWindow, HandRetargetResult, L21HandAngles
from teleoperation.retargeting.hand.config import HAND_ANGLE_DIM


class HandRetargeter(ABC):
    """Common boundary between canonical hand windows and L21 model results."""

    @abstractmethod
    def reset(self) -> None:
        """Reset retargeter state."""

    @abstractmethod
    def retarget(self, hand_window: HandWindow) -> HandRetargetResult:
        """Convert one canonical hand window into an L21 model result."""


class PoseTransformerRetargeter(HandRetargeter):
    """Adapt the existing :class:`TwoHandRetargeter` to ``HandRetargeter``.

    The wrapped retargeter owns the existing PoseTransformer invocation.  This
    adapter only translates the typed ``HandWindow``/``HandRetargetResult`` boundary;
    it does not alter model inputs, outputs, or numerical behavior.
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

    def retarget(self, hand_window: HandWindow) -> HandRetargetResult:
        if not isinstance(hand_window, HandWindow):
            raise TypeError("hand_window must be a HandWindow")

        output = self._retargeter.predict(
            hand_window.to_payload(),
            device=self.device,
        )
        hands = output["hands"]
        left_angles = _normalize_angles(hands.get("left"), "left")
        right_angles = _normalize_angles(hands.get("right"), "right")
        return HandRetargetResult(
            left_angles=left_angles,
            right_angles=right_angles,
            left_valid=left_angles is not None,
            right_valid=right_angles is not None,
            timestamp=hand_window.timestamp,
        )


class DexRetargetingRetargeter(HandRetargeter):
    """Placeholder for a future Dex-Retargeting implementation."""

    def reset(self) -> None:
        """Reset future algorithm state (currently there is no state)."""

    def retarget(self, hand_window: HandWindow) -> HandRetargetResult:
        raise NotImplementedError(
            "DexRetargetingRetargeter is reserved for a future implementation"
        )


def _normalize_angles(
    angles: np.ndarray | None,
    side: str,
) -> L21HandAngles | None:
    if angles is None:
        return None
    value = np.asarray(angles)
    if value.shape != (HAND_ANGLE_DIM,):
        raise ValueError(
            f"{side} retargeted angles must have shape (18,), got {value.shape}"
        )
    return L21HandAngles(value)


__all__ = [
    "DexRetargetingRetargeter",
    "HandRetargetResult",
    "HandRetargeter",
    "PoseTransformerRetargeter",
]
