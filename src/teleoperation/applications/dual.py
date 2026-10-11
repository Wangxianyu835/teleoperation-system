"""Reusable realtime observation-to-Tron2AL21Command service."""

from __future__ import annotations

import numpy as np

from teleoperation.retargeting.arm.calibration import ArmCalibration
from teleoperation.retargeting.arm.retargeter import ArmRetargeter
from teleoperation.retargeting.arm.safety import ArmSafetyController
from teleoperation.retargeting.arm.ik import RoboticsToolboxArmIK, load_arm_specification, _pose_matrix, _pose_error
from teleoperation.contracts.commands import Tron2AL21Command
from teleoperation.contracts.validation import validate_retarget_input


class Tron2ADualArmController:
    """Shared independent arm mapping, checked IK and bounded control."""

    def __init__(self, calibration_path, urdf_path, *, input_mode="legacy", orientation_mode=None):
        self.calibration = calibration = (calibration_path if isinstance(calibration_path, ArmCalibration)
                                          else ArmCalibration.load(calibration_path))
        self.arm_retargeter = ArmRetargeter(calibration, input_mode=input_mode, orientation_mode=orientation_mode)
        self.solvers = {side: RoboticsToolboxArmIK(load_arm_specification(urdf_path, side)) for side in ("left", "right")}
        self.safe_q = {side: calibration.safe_q[side].astype(np.float64) for side in ("left", "right")}
        self.controllers = {}
        for side, solver in self.solvers.items():
            spec = solver.specification
            if np.any(self.safe_q[side] < spec.lower) or np.any(self.safe_q[side] > spec.upper):
                raise ValueError(f"{side} safe reference violates URDF joint limits")
            if input_mode == "camera_world":
                errors = _pose_error(_pose_matrix(calibration.robot_neutral_pose[side]), spec.robot.fkine(self.safe_q[side]).A)
                if errors[0] > 1e-6 or errors[1] > 1e-6:
                    raise ValueError(f"{side} calibration reference does not match this URDF FK")
            self.controllers[side] = ArmSafetyController(spec, self.safe_q[side], calibration.max_acceleration, calibration.tracking_timeout)
        self.diagnostics = {}

    def update_side(self, body, side, timestamp):
        target = self.arm_retargeter.update(body, side)
        solver, safety = self.solvers[side], self.controllers[side]
        in_workspace = bool(np.all(target.pose[:3] >= self.calibration.workspace_min)
                            and np.all(target.pose[:3] <= self.calibration.workspace_max))
        workspace_rejected = self.arm_retargeter.input_mode == "camera_world" and target.valid and not in_workspace
        result = solver.solve(target, safety.q, self.safe_q[side]) if target.valid and not workspace_rejected else None
        success = False
        position_error = orientation_error = None
        if result is not None and result.success:
            spec = solver.specification
            q = np.asarray(result.q)
            if (q.shape == (7,) and np.isfinite(q).all()
                    and np.all(q >= spec.lower) and np.all(q <= spec.upper)):
                position_error, orientation_error = _pose_error(_pose_matrix(target.pose), spec.robot.fkine(q).A)
                success = position_error <= solver.position_tolerance and orientation_error <= solver.orientation_tolerance
        safety.update(result.q if success else None, success, timestamp)
        self.diagnostics[side] = dict(
            observation_valid=bool(target.valid), ik_attempted=result is not None, ik_success=bool(success),
            ik_position_error_m=position_error, ik_orientation_error_rad=orientation_error,
            target_pose=target.pose.tolist(), orientation_held=self.arm_retargeter.orientation_held[side],
            workspace_rejected=bool(workspace_rejected),
            status=safety.status, control_q=safety.q.tolist(), control_velocity=safety.velocity.tolist(),
        )
        return bool(success)

    def update(self, body, timestamp):
        return {side: self.update_side(body, side, timestamp) for side in ("left", "right")}


class DualTeleoperator:
    """Combine the existing hand model and TRON2A arm IK for one payload."""

    def __init__(self, hand_retargeter, device, calibration_path, urdf_path, *, hand_preprocessing="legacy"):
        from teleoperation.retargeting.hand.interface import PoseTransformerRetargeter
        from teleoperation.applications.hand_processing import DualHandPipeline, hand_coordinate_alignment
        from teleoperation.contracts.coordinates import validate_coordinate_alignment
        expected = hand_coordinate_alignment(hand_preprocessing)
        actual = validate_coordinate_alignment(getattr(hand_retargeter, "coordinate_alignment", None), "Dual hand model")
        if actual != expected:
            raise ValueError(f"Dual hand model coordinate alignment mismatch: preprocessing={expected!r}; model={actual!r}")
        self.hand_retargeter = PoseTransformerRetargeter(hand_retargeter, device)
        self.hand_pipeline = DualHandPipeline(hand_preprocessing)
        self.device = device
        self.arms = Tron2ADualArmController(calibration_path, urdf_path)
        for name in ("arm_retargeter", "solvers", "safe_q", "controllers"):
            setattr(self, name, getattr(self.arms, name))
        self.hand_q = {side: np.zeros(17, dtype=np.float32) for side in ("left", "right")}
        self.last_timestamp = None

    def update(self, observation: RawTeleopObservation | dict) -> Tron2AL21Command:
        from teleoperation.retargeting.hand.angles import angle18_to_dofs
        from teleoperation.contracts.hand import L21HandDOFs
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
            arm_valid[side] = self.arms.update_side(observation.upper_body, side, timestamp)
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
