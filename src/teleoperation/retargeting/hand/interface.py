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
from time import perf_counter
from teleoperation.contracts.hand import HandWindow
from teleoperation.contracts.hand_control import HandKey, HandJointResult
from .angles import canonical_to_native_target_angles


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
    "SharedJointNNRetargeter",
]


class SharedJointNNRetargeter(HandRetargeter):
    """Run one canonical target model and adapt its output to both sides."""

    def __init__(self, specs, model, device="cpu", *, canonical_spec=None):
        import torch

        if not specs:
            raise ValueError("At least one target spec is required")
        self.specs = dict(specs)
        self.device = device
        models = {spec.hand_model for spec in self.specs.values()}
        if len(models) != 1:
            raise ValueError("A shared joint model can serve one target hand model")
        target = next(iter(models))
        right_key = (target, "right")
        self.canonical_spec = canonical_spec or self.specs.get(right_key)
        if self.canonical_spec is None:
            raise ValueError("A shared joint model requires the right canonical target spec")
        if self.canonical_spec.hand_model != target or self.canonical_spec.side != "right":
            raise ValueError("Shared canonical spec must be the right spec for the target hand model")
        for key, spec in self.specs.items():
            if key != spec.key or spec.dof != self.canonical_spec.dof:
                raise ValueError("Shared target specs must have matching native output dimensions")
        self.model = model.to(device).eval()
        # eval() 选择推理行为；梯度是否记录由 retarget() 的 no_grad() 控制。

    def reset(self):
        """Inference is stateless; windows are owned by the producer."""

    def retarget(self, window):
        import torch
        from teleoperation.contracts.native_coordinates import validate_palm_length_contract

        if not isinstance(window, HandWindow):
            raise TypeError("NN retargeting requires HandWindow")
        validate_palm_length_contract(window.metadata, "Shared joint NN")
        results = {}
        # 模型不保存历史窗口；三帧历史由上游 pipeline 管理。左右手共用权重，
        # 但循环中分别推理，并非把两只手自动组成一个 batch。
        with torch.no_grad():
            for key, spec in self.specs.items():
                points = window.hands[spec.side]
                values, error, elapsed = None, None, 0.
                if points is not None:
                    if points.shape != (3, 25, 3) or not np.isfinite(points).all():
                        raise ValueError("Shared joint NN requires a finite normalized (3,25,3) window")
                    started = perf_counter()
                    # as_tensor 转为模型张量，unsqueeze(0) 加 batch 维：
                    # (3,25,3) → (1,3,25,3)；model(...) 经 PyTorch 调用 forward。
                    tensor = torch.as_tensor(
                        np.array(points), dtype=torch.float32, device=self.device,
                    ).unsqueeze(0)
                    values = self.model(tensor).detach()[0]
                    # 模型返回 (1,D)，[0] 取出 D 个角；随后转换目标左右手约定。
                    values = canonical_to_native_target_angles(
                        values, self.canonical_spec, spec,
                    )
                    values = values.detach().cpu().numpy()
                    elapsed = (perf_counter() - started) * 1000
                    if values.shape != (spec.dof,) or not np.isfinite(values).all():
                        values, error = None, "invalid shared model output"
                else:
                    error = "missing/invalid input"
                results[key] = HandJointResult(
                    *key, spec.spec_hash, spec.independent_joint_names,
                    values, window.timestamp, values is not None, error, elapsed,
                )
        return results
