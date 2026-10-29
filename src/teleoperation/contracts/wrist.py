"""Source-space wrist observations, independent of robot and hand-model axes."""
from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class WristObservation:
    """R_reference_local as xyzw; position is optional and in reference_frame.

    A hand-model reflection is NOT a wrist rotation. Invalid observations carry
    no invented identity pose. Timestamps use the enclosing frame's timebase.
    """
    orientation_xyzw: np.ndarray | None
    reference_frame: str
    local_frame: str
    source: str
    timestamp: float | None = None
    position: np.ndarray | None = None
    orientation_valid: bool = False
    position_valid: bool = False
    reason: str = ""

    def __post_init__(self):
        if not self.reference_frame or not self.local_frame:
            raise ValueError("Wrist coordinate frames must be explicit")
        if self.orientation_valid:
            q = np.asarray(self.orientation_xyzw, dtype=float)
            if q.shape != (4,) or not np.isfinite(q).all() or not np.isclose(np.linalg.norm(q), 1., atol=1e-5):
                raise ValueError("Wrist orientation must be a finite unit xyzw quaternion")
        if self.position_valid:
            p = np.asarray(self.position, dtype=float)
            if p.shape != (3,) or not np.isfinite(p).all():
                raise ValueError("Wrist position must be finite xyz in metres")
