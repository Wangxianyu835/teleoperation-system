"""Common arm control step for live camera input and deterministic replay."""
from __future__ import annotations


class CameraArmPipeline:
    def __init__(self, arm_controller):
        self.arm_controller = arm_controller
        self.last_control_seconds = None
        self.last_result = None

    def update(self, tracked, *, control_seconds: float):
        if self.last_control_seconds is not None and control_seconds <= self.last_control_seconds:
            raise ValueError("control_seconds must increase monotonically")
        self.last_control_seconds = float(control_seconds)
        update = getattr(self.arm_controller, "update")
        self.last_result = update(tracked.body, self.last_control_seconds, tracked.wrists)
        return self.last_result

    def step(self, dt_s: float):
        if dt_s <= 0:
            raise ValueError("dt_s must be positive")
        step = getattr(self.arm_controller, "step", None)
        if step is not None:
            return step(float(dt_s))
        return self.last_result
