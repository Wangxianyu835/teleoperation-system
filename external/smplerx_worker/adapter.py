"""Lazy boundary for the official SMPLer-X adapter.

Importing this module must not import torch, OpenMMLab, or SMPL-X.  The
model-specific implementation is loaded only by the isolated worker after
asset checks pass.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class SmplerXAssets:
    source_root: Path
    checkpoint: Path
    body_model_dir: Path
    model: str = "s32"

    def validate(self) -> None:
        if self.model != "s32":
            raise ValueError(f"unsupported SMPLer-X model: {self.model!r}")
        for name, path in (("source_root", self.source_root), ("checkpoint", self.checkpoint),
                           ("body_model_dir", self.body_model_dir)):
            if not path.exists():
                raise FileNotFoundError(f"SMPLer-X {name} not found: {path}")


class SmplerXAdapter:
    """Load and run a fixed upstream adapter inside the worker process."""

    def __init__(self, assets: SmplerXAssets):
        self.assets = assets
        self._backend: Any | None = None

    def load(self) -> None:
        self.assets.validate()
        raise RuntimeError(
            "The official SMPLer-X S32 backend is not installed. "
            "Install it in the isolated worker environment and implement "
            "the upstream adapter before running model smoke tests."
        )

    def infer(self, rgb):
        if self._backend is None:
            raise RuntimeError("SMPLer-X adapter is not loaded")
        return self._backend.infer(rgb)
