from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation
ARM_DOF = 7
from .ik import ArmSpecification, _joint_vector
class ArmSafetyController:
    """Apply joint limits, velocity/acceleration limits, timeout, and safe pose."""

    def __init__(self, specification: ArmSpecification, safe_q: np.ndarray, max_acceleration: float, tracking_timeout: float):
        self.specification = specification
        self.safe_q = np.clip(_joint_vector(safe_q, "safe_q"), specification.lower, specification.upper).astype(np.float32)
        self.max_acceleration = float(max_acceleration)
        self.tracking_timeout = float(tracking_timeout)
        self.q = self.safe_q.copy()
        self.velocity = np.zeros(ARM_DOF, dtype=np.float32)
        self._last_timestamp: float | None = None
        self._last_valid_timestamp: float | None = None

    def update(self, target_q: np.ndarray | None, valid: bool, timestamp: float) -> tuple[np.ndarray, bool]:
        timestamp = float(timestamp)
        dt = 1.0 / 30.0 if self._last_timestamp is None else max(timestamp - self._last_timestamp, 1.0 / 240.0)
        self._last_timestamp = timestamp
        if valid:
            desired = np.clip(_joint_vector(target_q, "target_q"), self.specification.lower, self.specification.upper)
            self._last_valid_timestamp = timestamp
        elif self._last_valid_timestamp is not None and timestamp - self._last_valid_timestamp >= self.tracking_timeout:
            desired = self.safe_q
        else:
            desired = self.q
        target_velocity = np.clip((desired - self.q) / dt, -self.specification.velocity, self.specification.velocity)
        maximum_velocity_change = self.max_acceleration * dt
        self.velocity += np.clip(target_velocity - self.velocity, -maximum_velocity_change, maximum_velocity_change)
        self.q = np.clip(self.q + self.velocity * dt, self.specification.lower, self.specification.upper).astype(np.float32)
        return self.q.copy(), bool(valid)
