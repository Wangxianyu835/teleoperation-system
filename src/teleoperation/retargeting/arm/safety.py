from __future__ import annotations
import numpy as np
from .ik import ArmSpecification, _joint_vector


class ArmSafetyController:
    """Bounded position/velocity/acceleration, including loss and bad clocks."""

    def __init__(self, specification: ArmSpecification, safe_q: np.ndarray, max_acceleration: float, tracking_timeout: float):
        self.specification = specification
        self.safe_q = np.clip(_joint_vector(safe_q, "safe_q"), specification.lower, specification.upper)
        self.max_acceleration = float(max_acceleration)
        self.tracking_timeout = float(tracking_timeout)
        if not np.isfinite([max_acceleration, tracking_timeout]).all() or min(max_acceleration, tracking_timeout) <= 0:
            raise ValueError("Acceleration and timeout must be finite and positive")
        self.q = self.safe_q.copy()
        self.velocity = np.zeros(7)
        self._last_timestamp = self._last_valid_timestamp = self._start_timestamp = None
        self.status = "initial_hold"

    def update(self, target_q: np.ndarray | None, valid: bool, timestamp: float) -> tuple[np.ndarray, bool]:
        timestamp = float(timestamp)
        if (not np.isfinite(timestamp) or timestamp < 0 or
                self._last_timestamp is not None and timestamp <= self._last_timestamp):
            self.status = "time_invalid"
            return self.q.copy(), False
        if self._last_timestamp is None:
            self._start_timestamp = self._last_timestamp = timestamp
            if valid:
                _joint_vector(target_q, "target_q")
                self._last_valid_timestamp = timestamp
            self.status = "initial_hold"
            return self.q.copy(), bool(valid)
        elapsed = timestamp - self._last_timestamp
        self._last_timestamp = timestamp
        # An overdue frame must not command an accumulated large displacement.
        # Integrate at most 50 ms, but use real elapsed time for the loss deadline.
        gap = elapsed > 0.25
        valid = bool(valid and not gap)
        dt = min(elapsed, 0.05)
        if valid:
            desired = np.clip(_joint_vector(target_q, "target_q"), self.specification.lower, self.specification.upper)
            self._last_valid_timestamp = timestamp
            self.status = "tracking"
        elif timestamp - (self._last_valid_timestamp if self._last_valid_timestamp is not None else self._start_timestamp) >= self.tracking_timeout:
            desired = self.safe_q
            self.status = "timeout_return"
        else:
            desired = None
            self.status = "loss_braking" if self._last_valid_timestamp is not None else "initial_hold"
        if gap:
            self.status = "time_gap_" + self.status
        # Small integration steps and a stopping-distance envelope prevent limit
        # clipping from silently violating acceleration bounds during braking.
        steps = max(1, int(np.ceil(dt * 240)))
        h = dt / steps
        a = self.max_acceleration
        for _ in range(steps):
            error = np.zeros(7) if desired is None else desired - self.q
            requested = np.sign(error) * np.minimum(self.specification.velocity, np.sqrt(2 * a * np.abs(error)))
            previous = self.velocity.copy()
            candidate = previous + np.clip(requested - previous, -a * h, a * h)
            toward_upper = (previous > 0) | ((previous == 0) & (requested >= 0))
            distance = np.where(toward_upper, self.specification.upper - self.q,
                                self.q - self.specification.lower)
            braking = distance <= previous ** 2 / (2 * a) + 2 * np.abs(previous) * h + 2 * a * h ** 2
            candidate = np.where(braking, np.sign(previous) * np.maximum(0., np.abs(previous) - a * h), candidate)
            movement = (previous + candidate) * h / 2
            # If braking reaches zero inside this substep, integrate only to
            # the stop event, then hold; stretching it to h would overshoot.
            stopping = braking & (np.abs(previous) < a * h)
            movement = np.where(stopping, np.sign(previous) * previous ** 2 / (2 * a), movement)
            self.velocity = candidate
            self.q += movement
        return self.q.copy(), valid
