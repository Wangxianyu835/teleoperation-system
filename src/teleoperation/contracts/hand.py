"""Hand processing values, deliberately separate from actuator commands."""
from dataclasses import dataclass, field
from typing import Any, Mapping
import numpy as np
from .validation import build_retarget_input


class _HandMetadataView:
    @property
    def coordinate_frame(self): return str(self.metadata.get("coordinate_frame", "unknown"))
    @property
    def units(self): return str(self.metadata.get("units", "unknown"))


@dataclass(frozen=True)
class CanonicalHandFrame(_HandMetadataView):
    hands: dict[str, np.ndarray | None]
    timestamp: float | None = None
    source: str = "unknown"
    metadata: Mapping[str, Any] = field(default_factory=dict)

    @property
    def valid(self): return {side: self.hands[side] is not None for side in ("left", "right")}
    @property
    def invalid_reasons(self):
        declared = self.metadata.get("invalid_reasons", {})
        return {side: declared.get(side, None if self.valid[side] else "missing or untrackable") for side in ("left", "right")}

    def __post_init__(self):
        for side in ("left", "right"):
            points = self.hands[side]
            if points is not None and np.shape(points) != (25, 3):
                raise ValueError(f"{side} canonical hand must have shape (25,3)")


@dataclass(frozen=True)
class HandWindow(_HandMetadataView):
    hands: dict[str, np.ndarray | None]
    timestamp: float | None = None
    source: str = "unknown"
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        for side in ("left", "right"):
            points = self.hands[side]
            if points is not None and (np.ndim(points) != 3 or np.shape(points)[1:] != (25, 3)):
                raise ValueError(f"{side} hand window must have shape (frames,25,3)")

    def to_payload(self):
        return build_retarget_input(self.source, self.timestamp, self.hands["left"], self.hands["right"], metadata=dict(self.metadata))


@dataclass(frozen=True)
class L21HandAngles:
    values: np.ndarray

    def __post_init__(self):
        if not isinstance(self.values, np.ndarray) or self.values.shape != (18,):
            raise ValueError("L21HandAngles must have shape (18,)")

    def native_mapping_dofs(self):
        """Remove only the placeholder; retain float64 replay precision."""
        return L21HandDOFs(self.values[1:].copy())


@dataclass(frozen=True)
class L21HandDOFs:
    values: np.ndarray

    def __post_init__(self):
        if not isinstance(self.values, np.ndarray) or self.values.shape != (17,):
            raise ValueError("L21HandDOFs must have shape (17,)")
        if not np.isfinite(self.values).all():
            raise ValueError("L21HandDOFs must be finite")


@dataclass(frozen=True)
class HandRetargetResult:
    left_angles: L21HandAngles | None = None
    right_angles: L21HandAngles | None = None
    left_valid: bool = False
    right_valid: bool = False
    timestamp: float | None = None


    def __post_init__(self):
        for side in ("left", "right"):
            angles = getattr(self, side + "_angles")
            if angles is not None and not isinstance(angles, L21HandAngles):
                raise TypeError(f"{side} model result must be L21HandAngles")
            if getattr(self, side + "_valid") and angles is None:
                raise ValueError(f"{side} valid result requires angles")


@dataclass(frozen=True)
class AngleFrame:
    frame_id: int
    timestamp: float
    left: np.ndarray
    right: np.ndarray
    left_valid: bool
    right_valid: bool
