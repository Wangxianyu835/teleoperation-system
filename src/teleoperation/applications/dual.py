"""Reusable realtime observation-to-Tron2AL21Command service."""

from __future__ import annotations

import numpy as np

from teleoperation.retargeting.arm.calibration import ArmCalibration
from teleoperation.retargeting.arm.retargeter import ArmRetargeter
from teleoperation.retargeting.arm.safety import ArmSafetyController
from teleoperation.retargeting.arm.ik import RoboticsToolboxArmIK, load_arm_specification
from teleoperation.contracts.commands import Tron2AL21Command
from teleoperation.contracts.validation import validate_retarget_input
from teleoperation.retargeting.hand.angles import angle18_to_dofs


class DualTeleoperator:
    """Combine the existing hand model and TRON2A arm IK for one payload."""

    def __init__(self, hand_retargeter, device, calibration_path, urdf_path, *, hand_preprocessing="legacy"):
        expected = hand_coordinate_alignment(hand_preprocessing)
        actual = validate_coordinate_alignment(getattr(hand_retargeter, "coordinate_alignment", None), "Dual hand model")
        if actual != expected:
            raise ValueError(f"Dual hand model coordinate alignment mismatch: preprocessing={expected!r}; model={actual!r}")
        self.hand_retargeter = PoseTransformerRetargeter(hand_retargeter, device)
        self.hand_pipeline = DualHandPipeline(hand_preprocessing)
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

    def update(self, observation: RawTeleopObservation | dict) -> Tron2AL21Command:
        try:
            observation = self._raw_observation(observation)
            observation.upper_body.validate(strict=True)
        except (TypeError, ValueError, KeyError):
            self.hand_pipeline.reset()
            raise
        timestamp = float(observation.upper_body.timestamp or 0.0)
        self.last_timestamp = timestamp
        window, _ = self.hand_pipeline.process_frame(observation.hands)
        result = None if window is None else self.hand_retargeter.retarget(window)
        hand_output = {side: None if result is None else getattr(result, side + "_angles") for side in ("left", "right")}
        arm_valid, hand_valid = {}, {}
        for side in ("left", "right"):
            target = self.arm_retargeter.update(observation.upper_body, side)
            pose, observed = target.pose, target.valid
            result = self.solvers[side].solve(target, self.controllers[side].q, self.safe_q[side]) if observed else None
            arm_valid[side] = result is not None and result.success
            self.controllers[side].update(result.q if arm_valid[side] else None, arm_valid[side], timestamp)
            predicted = hand_output[side]
            hand_valid[side] = predicted is not None
            if hand_valid[side]:
                self.hand_q[side] = L21HandDOFs(angle18_to_dofs(predicted.values)).values
        return Tron2AL21Command(timestamp, self.controllers["left"].q, self.hand_q["left"], self.controllers["right"].q, self.hand_q["right"], arm_valid["left"], hand_valid["left"], arm_valid["right"], hand_valid["right"])

    @staticmethod
    def _raw_observation(observation):
        # Factory dictionaries are converted explicitly at the application boundary.
        if isinstance(observation, dict):
            hands = observation.get("hands", {})
            if any(value is not None and getattr(value, "ndim", None) not in (2, 3) for value in hands.values()):
                raise ValueError("dual adapter must return raw hand landmarks, not HandWindow")
            timestamp = observation.get("timestamp")
            arms = observation["arms"]
            observation = RawTeleopObservation(
                RawHandFrame(hands, timestamp, observation.get("source", "unknown"), observation.get("metadata", {}), observation.get("geometry_encoding", {})),
                UpperBodyObservation(arms, {side: arms[side] is not None for side in ("left", "right")}, timestamp),
            )
        if not isinstance(observation, RawTeleopObservation):
            raise TypeError("dual input must be RawTeleopObservation")
        for side in ("left", "right"):
            value = observation.hands.hands[side]
            if value is not None and np.ndim(value) == 3 and (
                observation.hands.geometry_encoding.get(side) != "transforms25"
                or np.shape(value) != (25, 4, 4)
            ):
                raise ValueError("dual adapter must return raw hand landmarks, not HandWindow")
        return observation

    def close(self):
        self.hand_pipeline.reset()
        self.hand_retargeter.reset()


from teleoperation.contracts.observations import UpperBodyObservation, RawHandFrame, RawTeleopObservation
from teleoperation.contracts.hand import L21HandDOFs
from teleoperation.retargeting.hand.interface import PoseTransformerRetargeter
from teleoperation.applications.hand_processing import DualHandPipeline, hand_coordinate_alignment
from teleoperation.contracts.coordinates import validate_coordinate_alignment
