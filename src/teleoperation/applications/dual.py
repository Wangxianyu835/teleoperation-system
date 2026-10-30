"""Reusable realtime observation-to-Tron2AL21Command service."""

from __future__ import annotations

import numpy as np
from dataclasses import replace
from numbers import Real
import time

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
            if input_mode == "smplerx":
                errors = _pose_error(_pose_matrix(calibration.robot_neutral_pose[side]), spec.robot.fkine(self.safe_q[side]).A)
                if errors[0] > 1e-6 or errors[1] > 1e-6:
                    raise ValueError(f"{side} calibration reference does not match this URDF FK")
            self.controllers[side] = ArmSafetyController(spec, self.safe_q[side], calibration.max_acceleration, calibration.tracking_timeout)
        self.diagnostics = {}
        from teleoperation.retargeting.arm.wrist_orientation import WristTracker
        self.wrist_trackers = {s: WristTracker(calibration.rotation) for s in self.solvers}
        self.wrist_initial = {}
        self.wrist_generations = {}
        self.wrist_times = {}

    def update_side(self, body, side, timestamp, wrist=None):
        target = self.arm_retargeter.update(body, side)
        solver, safety = self.solvers[side], self.controllers[side]
        from scipy.spatial.transform import Rotation
        tracker = self.wrist_trackers[side]
        tracker.update(wrist, timestamp - self.wrist_times.get(side, timestamp - 1/30))
        self.wrist_times[side] = timestamp
        if tracker.neutral is not None:
            if self.wrist_generations.get(side) != tracker.generation:
                self.wrist_initial[side] = Rotation.from_matrix(solver.specification.robot.fkine(safety.q).A[:3, :3])
                self.wrist_generations[side] = tracker.generation
            pose = target.pose.copy()
            pose[3:] = (tracker.delta * self.wrist_initial[side]).as_quat()
            target = replace(target, pose=pose)
        in_workspace = bool(np.all(target.pose[:3] >= self.calibration.workspace_min)
                            and np.all(target.pose[:3] <= self.calibration.workspace_max))
        workspace_rejected = self.arm_retargeter.input_mode == "smplerx" and target.valid and not in_workspace
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
            wrist=tracker.summary(),
            observation_valid=bool(target.valid), ik_attempted=result is not None, ik_success=bool(success),
            ik_position_error_m=position_error, ik_orientation_error_rad=orientation_error,
            target_pose=target.pose.tolist(), orientation_held=self.arm_retargeter.orientation_held[side],
            workspace_rejected=bool(workspace_rejected),
            status=safety.status, control_q=safety.q.tolist(), control_velocity=safety.velocity.tolist(),
        )
        return bool(success)

    def update(self, body, timestamp, wrists=None):
        return {side: self.update_side(body, side, timestamp, (wrists or {}).get(side)) for side in ("left", "right")}


class DualTeleoperator:
    """Combine the existing hand model and TRON2A arm IK for one payload."""

    def __init__(self, hand_retargeter, device, calibration_path, urdf_path, *, hand_preprocessing="legacy",
                 input_mode="legacy", orientation_mode=None):
        from teleoperation.retargeting.hand.interface import PoseTransformerRetargeter
        from teleoperation.applications.hand_processing import DualHandPipeline, MediaPipePalmLocalPipeline, hand_coordinate_alignment
        from teleoperation.contracts.coordinates import validate_coordinate_alignment, WORLD_PALM_LOCAL_COORDINATE_ALIGNMENT
        if input_mode not in ("legacy", "smplerx"):
            raise ValueError("Unknown dual input mode")
        if input_mode == "smplerx" and hand_preprocessing != "legacy":
            raise ValueError(f"{input_mode} selects world hand preprocessing; do not set hand_preprocessing")
        self.input_mode = input_mode
        camera_world_hands = input_mode == "smplerx"
        expected = (WORLD_PALM_LOCAL_COORDINATE_ALIGNMENT if camera_world_hands
                    else hand_coordinate_alignment(hand_preprocessing))
        actual = validate_coordinate_alignment(getattr(hand_retargeter, "coordinate_alignment", None), "Dual hand model")
        if actual != expected:
            raise ValueError(f"Dual hand model coordinate alignment mismatch: preprocessing={expected!r}; model={actual!r}")
        self.hand_retargeter = PoseTransformerRetargeter(hand_retargeter, device)
        self.hand_pipeline = (MediaPipePalmLocalPipeline("world") if camera_world_hands
                              else DualHandPipeline(hand_preprocessing))
        self.device = device
        self.arms = Tron2ADualArmController(calibration_path, urdf_path, input_mode=input_mode,
                                          orientation_mode=orientation_mode)
        for name in ("arm_retargeter", "solvers", "safe_q", "controllers"):
            setattr(self, name, getattr(self.arms, name))
        self.hand_q = {side: np.zeros(17, dtype=np.float32) for side in ("left", "right")}
        self.last_timestamp = None
        self.last_observation = None
        self.hand_diagnostics = {}
        self.hand_processing_seconds = 0.

    def update(self, observation: RawTeleopObservation | dict, *, control_seconds=None) -> Tron2AL21Command:
        from teleoperation.contracts.hand_control import valid_joint_result
        try:
            observation = self._raw_observation(observation)
            # Camera per-side nonfinite/lost Pose is handled by the existing
            # mapper and safety controller, independently of finger inference.
            if self.input_mode == 'legacy':
                observation.upper_body.validate(strict=True)
            if self.input_mode == "smplerx" or control_seconds is not None:
                if (isinstance(control_seconds, bool) or not isinstance(control_seconds, Real)
                        or not np.isfinite(control_seconds) or control_seconds < 0
                        or (self.last_timestamp is not None and control_seconds <= self.last_timestamp)):
                    raise ValueError("control_seconds must be explicit, finite, non-negative and strictly increasing")
        except (TypeError, ValueError, KeyError):
            self.hand_pipeline.reset()
            raise
        timestamp = float(observation.upper_body.timestamp or 0.0) if control_seconds is None else float(control_seconds)
        self.last_timestamp = timestamp
        self.last_observation = observation
        started = time.perf_counter()
        if self.input_mode == "smplerx":
            hands = replace(observation.hands, timestamp=timestamp, metadata={
                **observation.hands.metadata, "capture_time_s": observation.hands.timestamp,
                "timestamp_unit": "relative_seconds",
            })
            window, canonical = self.hand_pipeline.process_observation(hands)
        else:
            window, canonical = self.hand_pipeline.process_frame(observation.hands)
        result = None if window is None else self.hand_retargeter.retarget(window)
        hand_output = {side: valid_joint_result(result, "l21", side) for side in ("left", "right")}
        self.hand_processing_seconds = time.perf_counter() - started
        arm_valid, hand_valid = {}, {}
        for side in ("left", "right"):
            wrist_source = observation.wrists if self.input_mode == "smplerx" else observation.hands.wrists
            arm_valid[side] = self.arms.update_side(observation.upper_body, side, timestamp, (wrist_source or {}).get(side))
            predicted = hand_output[side]
            hand_valid[side] = predicted is not None
            if hand_valid[side]:
                self.hand_q[side] = predicted.positions.astype(np.float32)
            if self.input_mode == "smplerx":
                self.hand_diagnostics[side] = dict(
                    detected=observation.hands.world_hands[side] is not None,
                    normalized_detected=observation.hands.hands[side] is not None,
                    aligned_valid=canonical.hands[side] is not None,
                    valid_streak=self.hand_pipeline.valid_streak[side],
                    window_ready=window is not None and window.hands[side] is not None,
                    inference_valid=hand_valid[side], invalid_reason=self.hand_pipeline.invalid_reasons[side],
                    target_q=self.hand_q[side].tolist(),
                )
        return Tron2AL21Command(timestamp, self.controllers["left"].q.copy(), self.hand_q["left"].copy(), self.controllers["right"].q.copy(), self.hand_q["right"].copy(), arm_valid["left"], hand_valid["left"], arm_valid["right"], hand_valid["right"])

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
        self.last_timestamp = None
        self.last_observation = None
        self.hand_diagnostics = {}


from teleoperation.contracts.observations import UpperBodyObservation, RawHandFrame, RawTeleopObservation
