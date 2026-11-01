"""Versioned body inference contracts shared by the camera process and worker."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping
import math

import numpy as np


@dataclass(frozen=True, order=True)
class BodyFrameId:
    session_id: str
    sequence: int
    capture_time_s: float

    def __post_init__(self):
        if not self.session_id:
            raise ValueError("session_id must not be empty")
        if int(self.sequence) != self.sequence or self.sequence < 0:
            raise ValueError("sequence must be a non-negative integer")
        if not math.isfinite(float(self.capture_time_s)):
            raise ValueError("capture_time_s must be finite")


@dataclass(frozen=True)
class SmplerXResult:
    frame_id: BodyFrameId
    completed_time_s: float
    model_id: str
    model_hash: str
    arrays: Mapping[str, np.ndarray]
    valid: Mapping[str, bool]
    reason: str = ""
    metadata: Mapping[str, Any] | None = None

    def __post_init__(self):
        if not math.isfinite(float(self.completed_time_s)):
            raise ValueError("completed_time_s must be finite")
        if not self.model_id or not self.model_hash:
            raise ValueError("model_id and model_hash are required")
        for name, value in self.arrays.items():
            array = np.asarray(value)
            if not np.issubdtype(array.dtype, np.number):
                raise ValueError(f"result array {name!r} must be numeric")
            if not np.isfinite(array).all():
                raise ValueError(f"result array {name!r} contains NaN or Inf")
        for side in self.valid:
            if side not in ("left", "right"):
                raise ValueError(f"unknown body side: {side!r}")
