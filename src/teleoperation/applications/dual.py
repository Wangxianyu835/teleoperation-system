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
from teleoperation.contracts.observations import RawTeleopObservation


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


def load_camera_hand_retargeter(checkpoint, device="cpu"):
    """Load only the canonical shared native L21 camera checkpoint."""
    from teleoperation.robots.hand_spec import load_hand_spec
    from teleoperation.retargeting.hand.checkpoint import load_shared_joint_checkpoint
    from teleoperation.retargeting.hand.interface import SharedJointNNRetargeter
    specs = {("l21", side): load_hand_spec("l21", side) for side in ("left", "right")}
    model, _ = load_shared_joint_checkpoint(checkpoint, specs[("l21", "right")], device)
    return SharedJointNNRetargeter(specs, model, device)


class DualTeleoperator:
    """Calibrated SMPL-X arms and canonical camera native L21 commands."""

    def __init__(self, hand_retargeter, device, calibration_path, urdf_path, *, orientation_mode=None):
        from teleoperation.applications.hand_processing import MediaPipePalmLocalPipeline
        from teleoperation.contracts.coordinates import PALM_LENGTH_PROFILE
        from teleoperation.retargeting.hand.joint_commands import JointCommandPipeline
        from teleoperation.robots.hand_spec import load_hand_spec
        self.specs = {("l21", side): load_hand_spec("l21", side) for side in ("left", "right")}
        if getattr(hand_retargeter, "specs", None) != self.specs:
            raise ValueError("Dual hand retargeter requires both native L21 specifications")
        self.hand_retargeter = hand_retargeter
        self.hand_pipeline = MediaPipePalmLocalPipeline("world", input_profile=PALM_LENGTH_PROFILE)
        self.hand_commands = JointCommandPipeline(self.specs)
        self.device = device
        self.arms = Tron2ADualArmController(calibration_path, urdf_path, input_mode="smplerx",
                                          orientation_mode=orientation_mode)
        for name in ("arm_retargeter", "solvers", "safe_q", "controllers"):
            setattr(self, name, getattr(self.arms, name))
        self.hand_q = {side: self.specs[("l21", side)].q_reference.astype(np.float32) for side in ("left", "right")}
        self.last_timestamp = None
        self.last_observation = None
        self.hand_diagnostics = {}
        self.hand_processing_seconds = 0.

    def update(self, observation: RawTeleopObservation, *, control_seconds) -> Tron2AL21Command:
        from teleoperation.contracts.hand_control import valid_joint_result
        try:
            if not isinstance(observation, RawTeleopObservation):
                raise TypeError("dual input must be RawTeleopObservation")
            if (isinstance(control_seconds, bool) or not isinstance(control_seconds, Real)
                    or not np.isfinite(control_seconds) or control_seconds < 0
                    or (self.last_timestamp is not None and control_seconds <= self.last_timestamp)):
                raise ValueError("control_seconds must be explicit, finite, non-negative and strictly increasing")
        except (TypeError, ValueError):
            self.hand_pipeline.reset()
            raise
        timestamp = float(control_seconds)
        started = time.perf_counter()
        hands = replace(observation.hands, timestamp=timestamp, metadata={
            **observation.hands.metadata, "capture_time_s": observation.hands.timestamp,
            "timestamp_unit": "relative_seconds",
        })
        window, canonical = self.hand_pipeline.process_observation(hands)
        result = None if window is None else self.hand_retargeter.retarget(window)
        if result is not None:
            if set(result) != set(self.specs) or any(value.key != key or value.timestamp != timestamp for key, value in result.items()):
                raise ValueError("Dual hand results must match selected target identities and control time")
        hand_output = {side: valid_joint_result(result, "l21", side) for side in ("left", "right")}
        self.last_timestamp = timestamp
        self.last_observation = observation
        arm_valid, hand_valid = {}, {}
        for side in ("left", "right"):
            key = ("l21", side)
            predicted = hand_output[side]
            accepted = (self.hand_commands.missing(key, timestamp) if predicted is None
                        else self.hand_commands.update(predicted, timestamp))
            named = accepted.named_positions()
            self.hand_q[side] = np.array([named[n] for n in self.specs[key].independent_joint_names], dtype=np.float32)
            hand_valid[side] = predicted is not None
            self.hand_diagnostics[side] = dict(
                detected=observation.hands.world_hands[side] is not None,
                normalized_detected=observation.hands.hands[side] is not None,
                aligned_valid=canonical.hands[side] is not None,
                valid_streak=self.hand_pipeline.valid_streak[side],
                window_ready=window is not None and window.hands[side] is not None,
                inference_valid=hand_valid[side], invalid_reason=self.hand_pipeline.invalid_reasons[side],
                target_q=self.hand_q[side].tolist(),
            )
        self.hand_processing_seconds = time.perf_counter() - started
        for side in ("left", "right"):
            arm_valid[side] = self.arms.update_side(observation.upper_body, side, timestamp, (observation.wrists or {}).get(side))
        return Tron2AL21Command(timestamp, self.controllers["left"].q.copy(), self.hand_q["left"].copy(), self.controllers["right"].q.copy(), self.hand_q["right"].copy(), arm_valid["left"], hand_valid["left"], arm_valid["right"], hand_valid["right"])


    def close(self):
        from teleoperation.retargeting.hand.joint_commands import JointCommandPipeline
        self.hand_pipeline.reset()
        self.hand_retargeter.reset()
        self.hand_commands = JointCommandPipeline(self.specs)
        self.hand_q = {side: self.specs[("l21", side)].q_reference.astype(np.float32) for side in ("left", "right")}
        self.last_timestamp = None
        self.last_observation = None
        self.hand_diagnostics = {}
