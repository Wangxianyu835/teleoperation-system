"""Unified hand retargeting interface and model adapters.

The interface stops at the canonical L21 hand command.  Robot-specific
transport, action packing, and simulator integration remain outside this
module.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

import numpy as np

from retargeting.hand_core import HandWindow
from retargeting.config import HAND_ANGLE_DIM


@dataclass(frozen=True)
class HandCommand:
    """One retargeted L21 command for the available hand sides."""

    left_angles: np.ndarray | None = None
    right_angles: np.ndarray | None = None
    left_valid: bool = False
    right_valid: bool = False
    timestamp: float | None = None


class HandRetargeter(ABC):
    """Common boundary between canonical hand windows and L21 commands."""

    @abstractmethod
    def reset(self) -> None:
        """Reset retargeter state."""

    @abstractmethod
    def retarget(self, hand_window: HandWindow) -> HandCommand:
        """Convert one canonical hand window into an L21 command."""


class PoseTransformerRetargeter(HandRetargeter):
    """Adapt the existing :class:`TwoHandRetargeter` to ``HandRetargeter``.

    The wrapped retargeter owns the existing PoseTransformer invocation.  This
    adapter only translates the typed ``HandWindow``/``HandCommand`` boundary;
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
            from retargeting.model import TwoHandRetargeter

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

    def retarget(self, hand_window: HandWindow) -> HandCommand:
        if not isinstance(hand_window, HandWindow):
            raise TypeError("hand_window must be a HandWindow")

        output = self._retargeter.predict(
            hand_window.to_payload(),
            device=self.device,
        )
        hands = output["hands"]
        left_angles = _normalize_angles(hands.get("left"), "left")
        right_angles = _normalize_angles(hands.get("right"), "right")
        return HandCommand(
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

    def retarget(self, hand_window: HandWindow) -> HandCommand:
        raise NotImplementedError(
            "DexRetargetingRetargeter is reserved for a future implementation"
        )


def _normalize_angles(
    angles: np.ndarray | None,
    side: str,
) -> np.ndarray | None:
    if angles is None:
        return None
    value = np.asarray(angles)
    if value.shape != (HAND_ANGLE_DIM,):
        raise ValueError(
            f"{side} retargeted angles must have shape (18,), got {value.shape}"
        )
    return value


__all__ = [
    "DexRetargetingRetargeter",
    "HandCommand",
    "HandRetargeter",
    "PoseTransformerRetargeter",
]
