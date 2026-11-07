"""Target-native independent joint results for every hand retargeting method."""

from __future__ import annotations

from abc import ABC, abstractmethod
from time import perf_counter

import numpy as np

from teleoperation.contracts.hand import HandWindow
from teleoperation.contracts.hand_control import HandKey, HandJointResult
from .angles import canonical_to_native_target_angles


class HandRetargeter(ABC):
    """Common boundary between declared hand windows and target joint results."""

    @abstractmethod
    def reset(self) -> None:
        """Reset retargeter state."""

    @abstractmethod
    def retarget(self, hand_window: HandWindow) -> dict[HandKey, HandJointResult]:
        """Solve each configured target, reporting invalid targets explicitly."""


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
        from teleoperation.contracts.coordinates import validate_palm_length_contract

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


__all__ = [
    "HandRetargeter",
    "SharedJointNNRetargeter",
]
