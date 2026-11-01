"""Target-native independent joint results for every hand retargeting method."""

from __future__ import annotations

from abc import ABC, abstractmethod
from time import perf_counter
from typing import Any

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

    def reset(self):
        """Inference is stateless; windows are owned by the producer."""

    def retarget(self, window):
        import torch
        from teleoperation.contracts.coordinates import validate_palm_length_contract

        if not isinstance(window, HandWindow):
            raise TypeError("NN retargeting requires HandWindow")
        validate_palm_length_contract(window.metadata, "Shared joint NN")
        results = {}
        with torch.no_grad():
            for key, spec in self.specs.items():
                points = window.hands[spec.side]
                values, error, elapsed = None, None, 0.
                if points is not None:
                    if points.shape != (3, 25, 3) or not np.isfinite(points).all():
                        raise ValueError("Shared joint NN requires a finite normalized (3,25,3) window")
                    started = perf_counter()
                    tensor = torch.as_tensor(
                        np.array(points), dtype=torch.float32, device=self.device,
                    ).unsqueeze(0)
                    values = self.model(tensor).detach()[0]
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
    "HandRetargeter", "JointNNRetargeter",
    "SharedJointNNRetargeter",
]
