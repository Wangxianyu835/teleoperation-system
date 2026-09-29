"""Reusable realtime observation-to-RobotCommand service."""

from __future__ import annotations

import numpy as np

from retargeting.arm import ArmCalibration, ArmRetargeter, ArmSafetyController, RoboticsToolboxArmIK, load_arm_specification
from retargeting.command import RobotCommand
from retargeting.contracts import validate_retarget_input
from retargeting.simulation import angle18_to_dofs


class DualTeleoperator:
    """Combine the existing hand model and TRON2A arm IK for one payload."""

    def __init__(self, hand_retargeter, device, calibration_path, urdf_path):
        self.hand_retargeter = hand_retargeter
        self.device = device
        calibration = ArmCalibration.load(calibration_path)
        self.arm_retargeter = ArmRetargeter(calibration)
        self.solvers = {side: RoboticsToolboxArmIK(load_arm_specification(urdf_path, side)) for side in ("left", "right")}
        self.safe_q = {side: calibration.safe_q[side].astype(np.float32) for side in ("left", "right")}
        self.controllers = {
            side: ArmSafetyController(self.solvers[side].specification, self.safe_q[side], calibration.max_acceleration, calibration.tracking_timeout)
            for side in ("left", "right")
        }
        self.hand_q = {side: np.zeros(17, dtype=np.float32) for side in ("left", "right")}
        self.last_timestamp = None

    def update(self, payload: dict) -> RobotCommand:
        validate_retarget_input(payload, require_arms=True)
        timestamp = float(payload.get("timestamp") or 0.0)
        self.last_timestamp = timestamp
        hand_output = self.hand_retargeter.predict(payload, self.device)["hands"]
        arm_valid, hand_valid = {}, {}
        for side in ("left", "right"):
            pose, observed = self.arm_retargeter.update(side, payload["arms"][side], payload["arms"][side] is not None)
            result = self.solvers[side].solve(pose, self.controllers[side].q, self.safe_q[side]) if observed else None
            arm_valid[side] = result is not None and result.success
            self.controllers[side].update(result.q if arm_valid[side] else None, arm_valid[side], timestamp)
            predicted = hand_output[side]
            hand_valid[side] = predicted is not None
            if hand_valid[side]:
                self.hand_q[side] = angle18_to_dofs(predicted)
        return RobotCommand(timestamp, self.controllers["left"].q, self.hand_q["left"], self.controllers["right"].q, self.hand_q["right"], arm_valid["left"], hand_valid["left"], arm_valid["right"], hand_valid["right"])
