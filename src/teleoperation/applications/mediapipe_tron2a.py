"""MediaPipe camera neutral calibration and TRON2A arms/hands simulation."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import time

import numpy as np

from teleoperation.contracts.observations import RawTeleopObservation
from teleoperation.data.calibration import write_calibration
from teleoperation.inputs.mediapipe import MediaPipeCameraInput
from teleoperation.retargeting.arm.calibration import (
    ArmCalibration, CameraCalibrationConfig, NeutralPoseCollector, tron2a_reference,
)


class RelativeControlClock:
    """Independent monotonic relative seconds; raw Unix ms stays untouched."""
    def __init__(self):
        self._origin = time.monotonic()
        self._last = None

    def next_seconds(self):
        value = max(0.0, time.monotonic() - self._origin)
        if self._last is not None:
            value = max(value, np.nextafter(self._last, np.inf))
        self._last = float(value)
        return self._last


@dataclass(frozen=True)
class TimedCameraObservation:
    raw: RawTeleopObservation
    control_relative_seconds: float


def _configuration(args):
    config = CameraCalibrationConfig(
        rotation=np.asarray(args.rotation, dtype=float).reshape(3, 3),
        scale=np.asarray(args.scale, dtype=float),
        translation=np.asarray(args.translation, dtype=float),
        safe_q={side: np.asarray(getattr(args, f"safe_q_{side}"), dtype=float)
                for side in ("left", "right")},
        workspace_min=np.asarray(args.workspace_min, dtype=float),
        workspace_max=np.asarray(args.workspace_max, dtype=float),
        sample_count=args.samples, min_visibility=args.pose_min_visibility,
        stability_tolerance=args.stability_tolerance,
        min_elbow_bend_degrees=args.min_elbow_bend,
        max_elbow_bend_degrees=args.max_elbow_bend,
        max_sample_gap=args.max_sample_gap,
        max_acceleration=args.max_acceleration, tracking_timeout=args.tracking_timeout,
    )
    for name in ("width", "height", "fps", "max_frames"):
        if getattr(args, name) <= 0:
            raise ValueError(f"--{name.replace('_', '-')} must be positive")
    if args.max_frames < args.samples:
        raise ValueError("--max-frames must be at least --samples")
    if not np.isfinite(args.timeout) or args.timeout <= 0:
        raise ValueError("--timeout must be finite and positive")
    if args.camera_index < 0:
        raise ValueError("--camera-index must be non-negative")
    output = Path(args.output)
    if output.suffix.lower() != ".json":
        raise ValueError("--output must end with .json")
    if output.exists():
        raise FileExistsError(f"Calibration already exists: {output}; choose a new output")
    for name in ("model_asset_path", "pose_model_asset_path"):
        if not Path(getattr(args, name)).is_file():
            raise FileNotFoundError(f"{name} not found: {getattr(args, name)}")
    # Validate URDF, safe joint bounds and FK before opening any camera.
    tron2a_reference(config, args.urdf)
    return config


def _preview(camera, timed, collector, window):
    import cv2
    from .world_landmark_capture import HAND_CONNECTIONS, _draw_landmarks

    raw = timed.raw
    canvas = camera.last_bgr.copy()
    for side, color in (("left", (0, 255, 0)), ("right", (255, 180, 0))):
        _draw_landmarks(canvas, raw.hands.hands[side], HAND_CONNECTIONS, color, f"{side}-hand")
        _draw_landmarks(canvas, raw.upper_body.arms[side], ((0, 1), (1, 2)), color,
                        f"{side}-arm", show_index=True)
    status = " ".join(
        f"{side}: hand={'OK' if raw.hands.hands[side] is not None else '--'} "
        f"arm={'OK' if raw.upper_body.valid[side] else '--'}"
        for side in ("left", "right")
    )
    for row, text in enumerate((status, collector.status,
                               f"relative seconds={timed.control_relative_seconds:.3f}; q/Esc cancels")):
        cv2.putText(canvas, text, (10, 23 + row * 25), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (255, 255, 255), 1)
    cv2.imshow(window, canvas)
    key = cv2.waitKey(1) & 0xFF
    return key in (ord("q"), 27) or cv2.getWindowProperty(window, cv2.WND_PROP_VISIBLE) < 1


def run(args):
    """Check -> capture -> collect -> calculate -> persist -> close resources."""
    camera = None
    preview_open = False
    window = "TRON2A neutral calibration (camera only)"
    try:
        config = _configuration(args)
        collector = NeutralPoseCollector(config)
        print("Hold both arms neutral with elbows slightly bent. "
              "Rotation/scale are configured; this does not calibrate camera extrinsics.", flush=True)
        camera = MediaPipeCameraInput(
            model_asset_path=args.model_asset_path, camera_index=args.camera_index,
            width=args.width, height=args.height, fps=args.fps,
            invalid_hand_as_missing=True, pose_model_asset_path=args.pose_model_asset_path,
            pose_min_visibility=config.min_visibility,
        )
        clock = RelativeControlClock()
        for _ in range(args.max_frames):
            seconds = clock.next_seconds()
            if seconds >= args.timeout:
                raise RuntimeError(f"Calibration timed out: {collector.status}")
            raw = camera.next_teleop_observation()
            timed = TimedCameraObservation(raw, clock.next_seconds())
            if timed.control_relative_seconds >= args.timeout:
                raise RuntimeError(f"Calibration timed out: {collector.status}")
            collector.add(timed.raw, timed.control_relative_seconds)
            if args.preview:
                preview_open = True
                if _preview(camera, timed, collector, window):
                    print("Calibration cancelled; no file saved.")
                    return 130
            if collector.ready:
                calibration = collector.calibration(args.urdf)
                data = calibration.to_dict()
                data["metadata"].update({
                    "camera_index": args.camera_index,
                    "hand_model_asset_path": str(Path(args.model_asset_path).resolve()),
                    "pose_model_asset_path": str(Path(args.pose_model_asset_path).resolve()),
                })
                # Validate the exact serializable result before publishing it.
                ArmCalibration.from_dict(data)
                destination = write_calibration(args.output, data)
                try:
                    ArmCalibration.load(destination)
                except BaseException:
                    # This command created the file exclusively. Do not leave
                    # an apparent success if read-back validation fails.
                    destination.unlink(missing_ok=True)
                    raise
                print(f"saved_calibration={destination} samples={len(collector.samples)}")
                return 0
        raise RuntimeError(f"No stable bilateral neutral window: {collector.status}")
    except KeyboardInterrupt:
        print("Calibration cancelled; no file saved.")
        return 130
    except (OSError, ValueError, RuntimeError, ImportError) as error:
        print(f"Calibration failed: {error}")
        return 1
    finally:
        try:
            if camera is not None:
                camera.release()
        finally:
            if preview_open:
                import cv2
                try:
                    cv2.destroyWindow(window)
                except cv2.error:
                    # Closing the window manually may already destroy it.
                    pass


class CameraRunReport:
    """Aggregate diagnostics without creating a trajectory/replay format."""
    def __init__(self, args):
        self.data = dict(mode="camera_arms_only_simulation", display=args.display,
                         calibration=str(Path(args.calibration).resolve()), urdf=str(Path(args.urdf).resolve()),
                         frames=0, exit_reason="initializing", real_camera_verified=False,
                         contact_validation=False, simulation_step_seconds=1/240,
                         simulation_steps_per_frame=args.simulation_steps,
                         latency_definition="wall clock: capture start through command simulation/state read; excludes preview",
                         latency={}, sides={side: dict(valid_observations=0, invalid_observations=0,
                         ik_success=0, ik_failure=0, workspace_rejections=0, statuses={}, orientation_held=0, metrics={},
                         control_constraint_violations=0, actual_position_limit_violations=0,
                         actual_velocity_limit_violations=0, actual_acceleration_peak_rad_s2=0.,
                         tracking_within_0_02_m=0, tracking_samples=0)
                         for side in ("left", "right")}, noncontrolled_drift_max_rad=0.)
        self._previous = {}
        self._hand_previous = {side: dict(detected=None, inference_valid=False, awaiting_recovery=False)
                               for side in ("left", "right")}
        if getattr(args, "with_hands", False):
            self.data["mode"] = "camera_dual_arms_hands_simulation"
            self.data["hands"] = {side: dict(detected_frames=0, aligned_valid_frames=0,
                window_ready_frames=0, inference_valid_frames=0, window_wait_frames=0,
                missing_or_invalid_frames=0, detection_losses=0, detection_recoveries=0,
                inference_recoveries=0) for side in ("left", "right")}

    @staticmethod
    def metric(metrics, name, value):
        if value is None or not np.isfinite(value):
            return
        item = metrics.setdefault(name, dict(count=0, sum=0., max=0., min=float(value)))
        item["count"] += 1
        item["sum"] += float(value)
        item["max"] = max(item["max"], float(value))
        item["min"] = min(item["min"], float(value))
        item["mean"] = item["sum"] / item["count"]

    def add(self, raw, seconds, controller, actual, durations):
        from teleoperation.retargeting.arm.ik import _pose_matrix, _pose_error
        self.data["frames"] += 1
        for name, value in durations.items():
            self.metric(self.data["latency"], name, value * 1000)
        for side, diag in controller.diagnostics.items():
            item = self.data["sides"][side]
            item["valid_observations" if diag["observation_valid"] else "invalid_observations"] += 1
            if diag["ik_attempted"]:
                item["ik_success" if diag["ik_success"] else "ik_failure"] += 1
            item["workspace_rejections"] += int(diag["workspace_rejected"])
            item["orientation_held"] += int(diag["orientation_held"])
            item["statuses"][diag["status"]] = item["statuses"].get(diag["status"], 0) + 1
            metrics = item["metrics"]
            for key in ("ik_position_error_m", "ik_orientation_error_rad"):
                self.metric(metrics, key, diag[key])
            safety = controller.controllers[side]
            spec = controller.solvers[side].specification
            state = actual["arms"][side]
            controlled_fk = spec.robot.fkine(safety.q).A
            limited_errors = _pose_error(controlled_fk, _pose_matrix(state["pose"]))
            self.metric(metrics, "actual_vs_control_position_m", limited_errors[0])
            self.metric(metrics, "actual_vs_control_orientation_rad", limited_errors[1])
            if diag["observation_valid"]:
                errors = _pose_error(_pose_matrix(diag["target_pose"]), _pose_matrix(state["pose"]))
                for name, value in zip(("actual_vs_target_position_m", "actual_vs_target_orientation_rad"), errors):
                    self.metric(metrics, name, value)
                target_control = _pose_error(_pose_matrix(diag["target_pose"]), controlled_fk)
                self.metric(metrics, "control_vs_target_position_m", target_control[0])
                self.metric(metrics, "control_vs_target_orientation_rad", target_control[1])
                item["tracking_samples"] += 1
                item["tracking_within_0_02_m"] += int(errors[0] <= 0.02)
            violation = bool(np.any(safety.q < spec.lower - 1e-7) or np.any(safety.q > spec.upper + 1e-7)
                             or np.any(np.abs(safety.velocity) > spec.velocity + 1e-7))
            previous = self._previous.get(side)
            if previous is not None:
                dt = seconds - previous[0]
                acceleration = np.abs(safety.velocity - previous[1]) / dt
                self.metric(metrics, "control_acceleration_peak_rad_s2", np.max(acceleration))
                violation |= bool(np.any(acceleration > safety.max_acceleration + 1e-6))
                actual_acceleration = np.max(np.abs(state["velocity"] - previous[2])) / (self.data["simulation_steps_per_frame"] / 240)
                item["actual_acceleration_peak_rad_s2"] = max(item["actual_acceleration_peak_rad_s2"], float(actual_acceleration))
            self._previous[side] = (seconds, safety.velocity.copy(), state["velocity"].copy())
            item["control_constraint_violations"] += int(violation)
            item["actual_position_limit_violations"] += int(np.any(state["q"] < spec.lower - 1e-5) or np.any(state["q"] > spec.upper + 1e-5))
            item["actual_velocity_limit_violations"] += int(np.any(np.abs(state["velocity"]) > spec.velocity + 1e-5))
            self.metric(metrics, "control_velocity_peak_rad_s", np.max(np.abs(safety.velocity)))
            self.metric(metrics, "actual_joint_vs_control_max_rad", np.max(np.abs(state["q"] - safety.q)))
            item["last"] = dict(diag, actual_q=state["q"].tolist(), actual_ee_pose=state["pose"].tolist(),
                                control_fk_pose=controlled_fk.tolist(), control_relative_seconds=seconds,
                                detection_unix_ms=raw.upper_body.timestamp)
        drift = max((abs(value) for value in actual["noncontrolled_joint_drift"].values()), default=0.)
        self.data["noncontrolled_drift_max_rad"] = max(self.data["noncontrolled_drift_max_rad"], drift)

    def add_hands(self, teleoperator):
        for side, diag in teleoperator.hand_diagnostics.items():
            item = self.data["hands"][side]
            previous = self._hand_previous[side]
            for flag, count in (("detected", "detected_frames"), ("aligned_valid", "aligned_valid_frames"),
                                ("window_ready", "window_ready_frames"), ("inference_valid", "inference_valid_frames")):
                item[count] += int(diag[flag])
            item["window_wait_frames"] += int(diag["aligned_valid"] and not diag["window_ready"])
            item["missing_or_invalid_frames"] += int(not diag["aligned_valid"])
            if previous["detected"] is True and not diag["aligned_valid"]:
                item["detection_losses"] += 1
            if previous["detected"] is False and diag["aligned_valid"]:
                item["detection_recoveries"] += 1
            if previous["inference_valid"] and not diag["inference_valid"]:
                previous["awaiting_recovery"] = True
            if previous["awaiting_recovery"] and diag["inference_valid"]:
                item["inference_recoveries"] += 1
                previous["awaiting_recovery"] = False
            previous.update(detected=diag["aligned_valid"], inference_valid=diag["inference_valid"])
            item["last"] = dict(diag, control_relative_seconds=teleoperator.last_timestamp,
                                detection_unix_ms=teleoperator.last_observation.hands.timestamp)

    def write(self, path):
        import json
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self.data["acceptance"] = "Pending human camera review; exit status alone does not establish tracking success"
        path.write_text(json.dumps(self.data, indent=2, allow_nan=False), encoding="utf-8")
        print(f"camera_report={path.resolve()} frames={self.data['frames']} reason={self.data['exit_reason']}", flush=True)


def _runtime_configuration(args):
    from teleoperation.contracts.checkpoints import validate_checkpoint_selection
    with_hands = getattr(args, "with_hands", False)
    if bool(args.arms_only) == bool(with_hands):
        raise ValueError("Select exactly one of --arms-only or --with-hands")
    paths = [getattr(args, name, None) for name in ("checkpoint", "left_checkpoint", "right_checkpoint")]
    validate_checkpoint_selection(*paths, required=with_hands)
    if with_hands and args.attach_hands is False:
        raise ValueError("--with-hands conflicts with --no-attach-hands")
    if args.attach_hands is None:
        args.attach_hands = bool(with_hands)
    if with_hands:
        for path in paths:
            if path is not None and not Path(path).is_file():
                raise FileNotFoundError(f"Hand checkpoint not found: {path}")
    if args.frames < 0 or args.camera_index < 0:
        raise ValueError("Frames and camera index must be non-negative")
    for name in ("width", "height", "fps", "simulation_steps"):
        if getattr(args, name) <= 0:
            raise ValueError(f"{name} must be positive")
    for name in ("model_asset_path", "pose_model_asset_path"):
        if not Path(getattr(args, name)).is_file():
            raise FileNotFoundError(f"{name} not found: {getattr(args, name)}")
    inputs = [getattr(args, name) for name in ("calibration", "urdf", "model_asset_path", "pose_model_asset_path")]
    if Path(args.report).resolve() in {Path(path).resolve() for path in inputs + paths if path is not None}:
        raise ValueError("Report path must not overwrite an input asset")
    if Path(args.report).suffix.lower() != ".json":
        raise ValueError("Report must end with .json")
    calibration = ArmCalibration.load(args.calibration)
    if "synthetic" in str(calibration.metadata.get("capture_kind", "")).lower() or ".synthetic." in Path(args.calibration).name.lower():
        raise ValueError("Synthetic calibration is not valid for live camera control; capture a real neutral calibration")
    visibility = args.pose_min_visibility
    if visibility is None:
        visibility = calibration.metadata.get("min_visibility", 0.7)
    if not np.isfinite(visibility) or not 0 <= visibility <= 1:
        raise ValueError("Pose minimum visibility must be within [0,1]")
    return calibration, visibility


def _runtime_preview(camera, raw, controller, actual, window, teleoperator=None):
    import cv2
    from .world_landmark_capture import _draw_landmarks, HAND_CONNECTIONS
    from teleoperation.retargeting.arm.ik import _pose_matrix, _pose_error
    canvas = camera.last_bgr.copy()
    for row, (side, color) in enumerate((("left", (0, 255, 0)), ("right", (255, 180, 0)))):
        _draw_landmarks(canvas, raw.upper_body.arms[side], ((0, 1), (1, 2)), color, side)
        diag = controller.diagnostics[side]
        ik = "WORKSPACE" if diag["workspace_rejected"] else "OK" if diag["ik_success"] else "FAIL" if diag["ik_attempted"] else "--"
        status = f"{side}: Pose={'OK' if diag['observation_valid'] else '--'} IK={ik} {diag['status']}"
        if diag["orientation_held"]:
            status += " direction HOLD"
        pose = actual["arms"][side]["pose"]
        fk = controller.solvers[side].specification.robot.fkine(controller.controllers[side].q).A
        control_error = _pose_error(fk, _pose_matrix(pose))[0]
        target_error = (f"{1000 * _pose_error(_pose_matrix(diag['target_pose']), _pose_matrix(pose))[0]:.1f}mm"
                        if diag["observation_valid"] else "--")
        state = f"EE=({pose[0]:.2f},{pose[1]:.2f},{pose[2]:.2f})m target err={target_error} control err={1000 * control_error:.1f}mm"
        for offset, text in enumerate((status, state)):
            cv2.putText(canvas, text, (10, 23 + row * 52 + offset * 23), cv2.FONT_HERSHEY_SIMPLEX, 0.42, color, 1)
    if teleoperator is not None:
        for row, (side, color) in enumerate((("left", (0, 255, 0)), ("right", (255, 180, 0)))):
            _draw_landmarks(canvas, raw.hands.hands[side], HAND_CONNECTIONS, color, side + "-hand")
            diag = teleoperator.hand_diagnostics[side]
            status = (f"{side} hand: detected={int(diag['detected'])} aligned={int(diag['aligned_valid'])} "
                      f"window={int(diag['window_ready'])} inference={int(diag['inference_valid'])} "
                      f"streak={diag['valid_streak']}/3")
            cv2.putText(canvas, status, (10, 135 + row * 23), cv2.FONT_HERSHEY_SIMPLEX, 0.42, color, 1)
        cv2.putText(canvas, f"device={teleoperator.device}; hand processing={1000 * teleoperator.hand_processing_seconds:.1f}ms",
                    (10, 181), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255), 1)
        for row, label in enumerate(getattr(teleoperator, "hand_model_labels", ())):
            cv2.putText(canvas, label, (10, 204 + row * 20), cv2.FONT_HERSHEY_SIMPLEX, 0.36, (255, 255, 255), 1)
    footer_y = 211 + 20 * len(getattr(teleoperator, "hand_model_labels", ())) if teleoperator is not None else 135
    cv2.putText(canvas, "TRON2A simulation; q/Esc closes", (10, footer_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    cv2.imshow(window, canvas)
    key = cv2.waitKey(1) & 0xFF
    return key in (ord("q"), 27) or cv2.getWindowProperty(window, cv2.WND_PROP_VISIBLE) < 1


def run_camera(args):
    from teleoperation.applications.dual import Tron2ADualArmController, DualTeleoperator
    from teleoperation.simulation.tron2a_world import Tron2ADualArmWorld
    camera = world = report = teleoperator = None
    preview_open = False
    window = "TRON2A camera arms simulation"
    code = 0
    try:
        calibration, visibility = _runtime_configuration(args)
        report = CameraRunReport(args)
        with_hands = getattr(args, "with_hands", False)
        if with_hands:
            from .hand_realtime import load_palm_local_retargeter
            from .hand_checkpoints import checkpoint_options, checkpoint_metadata, checkpoint_description
            options = checkpoint_options(args)
            options = {name: None if path is None else str(Path(path).resolve()) for name, path in options.items()}
            model = load_palm_local_retargeter(device=args.device, landmark_space="world", **options)
            teleoperator = DualTeleoperator(model, args.device, calibration, args.urdf,
                                           input_mode="camera_world", orientation_mode=args.orientation_mode)
            teleoperator.hand_model_labels = [f"{name}: {'/'.join(Path(path).parts[-4:])}"
                                              for name, path in checkpoint_metadata(options).items()]
            controller = teleoperator.arms
            window = "TRON2A camera dual arms and hands simulation"
            report.data.update(hand_checkpoints=checkpoint_metadata(options), inference_device=str(args.device),
                               hand_coordinate_alignment=model.coordinate_alignment, hand_landmark_space="world")
            print(f"World hand model: {checkpoint_description(options)} device={args.device}", flush=True)
        else:
            controller = Tron2ADualArmController(calibration, args.urdf, input_mode="camera_world", orientation_mode=args.orientation_mode)
        report.data.update(orientation_mode=args.orientation_mode, attach_hands=args.attach_hands,
                           hand_reference_only=not with_hands,
                           extrinsics_calibrated=calibration.metadata.get("extrinsics_calibrated", False))
        world = Tron2ADualArmWorld(args.urdf, gui=args.gui, display=args.display,
                                  attach_hands=args.attach_hands, safe_q=controller.safe_q,
                                  hand_mount_pose=calibration.hand_mount_pose, hand_reference_only=not with_hands)
        camera = MediaPipeCameraInput(model_asset_path=args.model_asset_path, camera_index=args.camera_index,
                    width=args.width, height=args.height, fps=args.fps, invalid_hand_as_missing=True,
                    pose_model_asset_path=args.pose_model_asset_path, pose_min_visibility=visibility)
        clock = RelativeControlClock()
        print(f"TRON2A {'dual arms and hands' if with_hands else 'arm-only'} simulation. Start neutral; rotation/scale are configured, not estimated. q/Esc exits preview.", flush=True)
        while args.frames == 0 or report.data["frames"] < args.frames:
            if not world.is_connected():
                report.data["exit_reason"] = "simulation_window_closed"
                break
            started = time.perf_counter()
            raw = camera.next_teleop_observation()
            captured = time.perf_counter()
            metadata = raw.hands.metadata
            if (metadata.get("arm_world_space") != "mediapipe_pose_world_meters"
                    or metadata.get("timestamp_unit") != "unix_ms"
                    or raw.upper_body.timestamp is None or not np.isfinite(raw.upper_body.timestamp)
                    or raw.hands.timestamp != raw.upper_body.timestamp):
                raise ValueError("Camera observation source/units or matching detection timestamps invalid")
            seconds = clock.next_seconds()
            if teleoperator is not None:
                command = teleoperator.update(raw, control_seconds=seconds)
            else:
                controller.update(raw.upper_body, seconds)
            controlled = time.perf_counter()
            if teleoperator is not None:
                world.apply(command)
            else:
                world.apply_arms({side: safety.q for side, safety in controller.controllers.items()})
            for _ in range(args.simulation_steps):
                world.step()
            actual = world.actual_state()
            simulated = time.perf_counter()
            durations = dict(capture_detection_ms=captured-started,
                       retarget_ik_control_ms=controlled-captured, simulation_state_ms=simulated-controlled,
                       capture_to_actual_ms=simulated-started)
            if teleoperator is not None:
                durations["hand_processing_ms"] = teleoperator.hand_processing_seconds
                report.add_hands(teleoperator)
            report.add(raw, seconds, controller, actual, durations)
            if args.preview:
                preview_open = True
                preview_options = {} if teleoperator is None else dict(teleoperator=teleoperator)
                if _runtime_preview(camera, raw, controller, actual, window, **preview_options):
                    report.data["exit_reason"] = "preview_exit"
                    break
            # GUI and DIRECT both run at capture cadence, without accumulating
            # a control-time jump after an overdue frame.
            delay = 1 / args.fps - (time.perf_counter() - started)
            if delay > 0:
                time.sleep(delay)
        else:
            report.data["exit_reason"] = "frame_limit"
    except KeyboardInterrupt:
        code = 130
        if report is not None:
            report.data["exit_reason"] = "keyboard_interrupt"
    except Exception as error:
        if world is not None and not world.is_connected():
            if report is not None:
                report.data["exit_reason"] = "simulation_window_closed"
        else:
            code = 1
            print(f"Camera TRON2A simulation failed: {error}", flush=True)
            if report is not None:
                report.data.update(exit_reason="error", error=str(error))
    finally:
        # Every owned resource is closed even if another cleanup operation fails.
        for resource, method in ((camera, "release"), (teleoperator, "close"), (world, "close")):
            if resource is not None:
                try:
                    getattr(resource, method)()
                except Exception as error:
                    print(f"Cleanup failed: {error}")
                    code = 1
        if preview_open:
            import cv2
            try:
                cv2.destroyWindow(window)
            except cv2.error:
                pass
        if report is not None:
            try:
                report.write(args.report)
            except Exception as error:
                print(f"Report write failed: {error}")
                code = 1
    return code
