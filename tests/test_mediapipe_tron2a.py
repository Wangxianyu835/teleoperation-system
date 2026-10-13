"""TRON2A camera calibration/control with synthetic observations and real physics.

No tests open physical cameras. Preview handedness, occlusion recovery and
device reopening still require the user's hardware acceptance checks.
"""
from dataclasses import replace
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

import numpy as np
from scipy.spatial.transform import Rotation

from teleoperation.applications import mediapipe_tron2a as application
from teleoperation.cli import build_parser
from teleoperation.contracts.observations import RawHandFrame, RawTeleopObservation, UpperBodyObservation
from teleoperation.data.calibration import read_calibration, write_calibration
from teleoperation.paths import DEFAULT_TRON2A_URDF
from teleoperation.retargeting.arm.calibration import (
    ArmCalibration, CameraCalibrationConfig, NeutralPoseCollector, tron2a_reference,
)


def configuration(**kwargs):
    values = dict(
        rotation=np.eye(3), scale=np.ones(3),
        safe_q={side: np.array([0, 0, 0, -0.5, 0, 0, 0.]) for side in ("left", "right")},
        workspace_min=np.full(3, -1.), workspace_max=np.ones(3), sample_count=5,
    )
    values.update(kwargs)
    return CameraCalibrationConfig(**values)


def observation(index=0, *, shift=0., invalid=None, visibility=0.95, no_world=False,
                straight=False, missing=False):
    left = np.array([[0.2, 0., 0.], [0.2, 0.25, 0.], [0.2, 0.45, -0.15]])
    right = np.array([[-0.2, 0., 0.], [-0.2, 0.25, 0.], [-0.3, 0.45, -0.1]])
    if straight:
        left[2] = left[1] + left[1] - left[0]
    left[2, 0] += shift
    valid = {side: side != invalid for side in ("left", "right")}
    timestamp = 1800000000000 + index * 33
    metadata = dict(frame_index=index, timestamp_unit="unix_ms",
                    arm_world_space="mediapipe_pose_world_meters",
                    pose_visibility=np.full(6, visibility))
    points = {"left": left, "right": right}
    return RawTeleopObservation(
        RawHandFrame({"left": None, "right": None}, timestamp, metadata=metadata),
        UpperBodyObservation({side: None if missing else points[side].copy() for side in points},
                             valid, timestamp, "mediapipe_pose_normalized", "normalized",
                             world_arms=None if no_world else {
                                 side: None if missing else points[side].copy() for side in points}),
    )


def synthetic_calibration():
    collector = NeutralPoseCollector(configuration())
    for index in range(5):
        collector.add(observation(index), index / 30.)
    return collector.calibration(DEFAULT_TRON2A_URDF)


class NeutralCalibrationTests(unittest.TestCase):
    def test_synthetic_reference_positions_and_real_fk(self):
        calibration = synthetic_calibration()
        np.testing.assert_allclose(calibration.human_neutral_wrist["left"], [0, 0.45, -0.15])
        np.testing.assert_allclose(calibration.human_neutral_wrist["right"], [-0.1, 0.45, -0.1])
        from teleoperation.retargeting.arm.ik import load_arm_specification
        for side in ("left", "right"):
            spec = load_arm_specification(DEFAULT_TRON2A_URDF, side)
            matrix = spec.robot.fkine(calibration.safe_q[side]).A
            np.testing.assert_allclose(calibration.robot_neutral_pose[side][:3], matrix[:3, 3])
            np.testing.assert_allclose(Rotation.from_quat(calibration.robot_neutral_pose[side][3:]).as_matrix(), matrix[:3, :3], atol=1e-10)
        self.assertFalse(calibration.metadata["extrinsics_calibrated"])
        self.assertEqual(calibration.metadata["human_reference"], "each_side_own_shoulder")

    def test_global_translation_does_not_change_shoulder_relative_reference(self):
        collector = NeutralPoseCollector(configuration())
        for index in range(5):
            raw = observation(index)
            for side in ("left", "right"):
                raw.upper_body.world_arms[side][:] += [index * 0.1, -0.05, 0.2]
            collector.add(raw, index / 30.)
        calibration = collector.calibration(DEFAULT_TRON2A_URDF)
        np.testing.assert_allclose(calibration.human_neutral_wrist["left"], [0, .45, -.15], atol=1e-12)

    def test_insufficient_and_unstable_samples_are_rejected(self):
        collector = NeutralPoseCollector(configuration())
        for index in range(4):
            collector.add(observation(index), index / 30.)
        with self.assertRaisesRegex(ValueError, "Insufficient"):
            collector.calibration(DEFAULT_TRON2A_URDF)
        for index in range(4, 20):
            collector.add(observation(index, shift=.15 * (index % 2)), index / 30.)
        self.assertFalse(collector.ready)
        with self.assertRaisesRegex(ValueError, "Insufficient"):
            collector.calibration(DEFAULT_TRON2A_URDF)

    def test_missing_invalid_low_visibility_world_and_straight_elbow_reset_window(self):
        for kwargs in (dict(invalid="left"), dict(invalid="right"), dict(visibility=.1),
                       dict(no_world=True), dict(straight=True), dict(missing=True)):
            with self.subTest(kwargs=kwargs):
                collector = NeutralPoseCollector(configuration())
                collector.add(observation(0), 0.)
                self.assertFalse(collector.add(observation(1, **kwargs), .03))
                self.assertEqual(len(collector.samples), 0)
                for index in range(2, 7):
                    collector.add(observation(index), index / 30.)
                self.assertTrue(collector.ready)

    def test_nonfinite_metric_points_and_visibility_reset_window(self):
        for what in ("world", "visibility"):
            collector = NeutralPoseCollector(configuration())
            raw = observation()
            if what == "world":
                raw.upper_body.world_arms["left"][2, 0] = np.nan
            else:
                raw.hands.metadata["pose_visibility"][0] = np.nan
            self.assertFalse(collector.add(raw, 0.))
            self.assertEqual(len(collector.samples), 0)

    def test_elbow_motion_with_stationary_wrist_is_unstable(self):
        collector = NeutralPoseCollector(configuration())
        for index in range(5):
            raw = observation(index)
            raw.upper_body.world_arms["left"][1, 0] += .08 * (index % 2)
            collector.add(raw, index / 30.)
        self.assertFalse(collector.ready)

    def test_frame_gap_detection_gap_and_time_gap_restart_window(self):
        for index, seconds, repeat_detection in ((5, .03, False), (1, 1., False), (1, .03, True)):
            with self.subTest(index=index, seconds=seconds, repeat=repeat_detection):
                collector = NeutralPoseCollector(configuration())
                collector.add(observation(), 0.)
                raw = observation(index)
                if repeat_detection:
                    raw = replace(raw, hands=replace(raw.hands, timestamp=1800000000000),
                                  upper_body=replace(raw.upper_body, timestamp=1800000000000))
                collector.add(raw, seconds)
                self.assertEqual(len(collector.samples), 1)

    def test_raw_timestamp_and_source_validation(self):
        for change in ("timestamp", "unit", "space", "frame"):
            raw = observation()
            if change == "timestamp":
                raw = replace(raw, upper_body=replace(raw.upper_body, timestamp=1))
            else:
                raw.hands.metadata[{"unit": "timestamp_unit", "space": "arm_world_space", "frame": "frame_index"}[change]] = "wrong"
            with self.assertRaises(ValueError):
                NeutralPoseCollector(configuration()).add(raw, 0.)

    def test_clock_seconds_are_independent_of_raw_unix_milliseconds(self):
        with patch.object(application.time, "monotonic", side_effect=[100., 100., 100., 100.25, 100.2]):
            clock = application.RelativeControlClock()
            values = [clock.next_seconds() for _ in range(4)]
        self.assertEqual(values[0], 0.)
        self.assertEqual(values[2], .25)
        self.assertTrue(all(b > a for a, b in zip(values, values[1:])))
        raw = observation()
        timed = application.TimedCameraObservation(raw, .25)
        self.assertEqual(timed.raw.hands.timestamp, 1800000000000)
        self.assertEqual(timed.control_relative_seconds, .25)
        collector = NeutralPoseCollector(configuration())
        collector.add(raw, 0.)
        with self.assertRaisesRegex(ValueError, "monotonically"):
            collector.add(observation(1), 0.)

    def test_mapping_and_collection_configuration_validation(self):
        invalid = dict(
            rotation=[np.zeros((3, 3)), np.diag([1., 1., -1.]), np.full((3, 3), np.nan)],
            scale=[np.zeros(3), np.full(3, np.nan), np.full(3, -1.)],
            translation=[np.full(3, np.inf)], workspace_min=[np.ones(3)],
            workspace_max=[np.full(3, np.nan)], sample_count=[2, 3.5],
            min_visibility=[-1, 1.1, np.nan], stability_tolerance=[0., np.nan],
            max_sample_gap=[0.], max_acceleration=[-1., np.nan], tracking_timeout=[0.],
            min_elbow_bend_degrees=[0., 170., np.nan], max_elbow_bend_degrees=[5., 180.],
            safe_q=[{"left": np.full(7, np.inf), "right": np.zeros(7)}],
        )
        for name, values in invalid.items():
            for value in values:
                with self.subTest(name=name, value=value), self.assertRaises(ValueError):
                    configuration(**{name: value})

    def test_fk_rejects_out_of_limit_safe_pose_and_workspace(self):
        config = configuration()
        bad_q = {side: values.copy() for side, values in config.safe_q.items()}
        bad_q["right"][0] = 10.
        with self.assertRaisesRegex(ValueError, "outside URDF"):
            tron2a_reference(replace(config, safe_q=bad_q), DEFAULT_TRON2A_URDF)
        with self.assertRaisesRegex(ValueError, "outside workspace"):
            tron2a_reference(replace(config, workspace_min=np.zeros(3)), DEFAULT_TRON2A_URDF)
        with self.assertRaises(FileNotFoundError):
            tron2a_reference(config, "missing-tron2a.urdf")


class CalibrationPersistenceTests(unittest.TestCase):
    def test_strict_json_save_reload_and_no_overwrite(self):
        calibration = synthetic_calibration()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "calibration.json"
            write_calibration(path, calibration.to_dict())
            loaded = ArmCalibration.load(path)
            for name in ("rotation", "translation", "scale", "workspace_min", "workspace_max"):
                np.testing.assert_array_equal(getattr(loaded, name), getattr(calibration, name))
            for name in ("human_neutral_wrist", "robot_neutral_pose", "safe_q", "tool_offset_quat_xyzw"):
                for side in ("left", "right"):
                    np.testing.assert_array_equal(getattr(loaded, name)[side], getattr(calibration, name)[side])
            original = path.read_bytes()
            with self.assertRaises(FileExistsError):
                write_calibration(path, {"different": True})
            self.assertEqual(path.read_bytes(), original)
            self.assertFalse(list(Path(directory).glob("*.tmp")))
            trace = loaded.metadata["capture_trace"]
            self.assertEqual(trace[0]["detection_unix_ms"], 1800000000000)
            self.assertAlmostEqual(trace[-1]["control_relative_seconds"], 4 / 30.)

    def test_serialization_and_publish_failures_leave_no_result_or_temporary_file(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "calibration.json"
            with self.assertRaises(ValueError):
                write_calibration(path, {"bad": float("nan")})
            with patch("teleoperation.data.calibration.os.link", side_effect=OSError("publish failed")):
                with self.assertRaises(OSError):
                    write_calibration(path, {"valid": True})
            self.assertEqual(list(Path(directory).iterdir()), [])

    def test_loading_rejects_invalid_key_values(self):
        calibration = synthetic_calibration()
        for name, value in (("rotation", np.diag([1., 1., -1.]).tolist()),
                            ("scale", [1., float("nan"), 1.]),
                            ("translation", [0., float("inf"), 0.]),
                            ("workspace_max", [-2., -2., -2.]),
                            ("max_acceleration", -1.), ("tracking_timeout", float("nan")),
                            ("robot_neutral_pose", {side: [0., 0., 0., 0., 0., 0., 0.] for side in ("left", "right")})):
            data = calibration.to_dict()
            data[name] = value
            with self.subTest(name=name), self.assertRaises(ValueError):
                ArmCalibration.from_dict(data)

    def test_delivered_synthetic_fixture_is_loadable_and_matches_real_fk(self):
        path = Path(__file__).parent / "fixtures" / "tron2a_camera_neutral.synthetic.json"
        loaded = ArmCalibration.load(path)
        expected = synthetic_calibration()
        for side in ("left", "right"):
            np.testing.assert_allclose(loaded.robot_neutral_pose[side], expected.robot_neutral_pose[side], atol=1e-12)
            np.testing.assert_allclose(loaded.human_neutral_wrist[side], expected.human_neutral_wrist[side], atol=1e-12)
        self.assertEqual(read_calibration(path)["metadata"]["capture_kind"], "synthetic_validation_only")


class CameraApplicationTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.directory = Path(temporary.name)
        for name in ("hands.task", "pose.task"):
            (self.directory / name).touch()
        self.args = build_parser().parse_args([
            "dual", "calibrate-camera", "--output", str(self.directory / "calibration.json"),
            "--model-asset-path", str(self.directory / "hands.task"),
            "--pose-model-asset-path", str(self.directory / "pose.task"),
            "--safe-q-left", "0", "0", "0", "-0.5", "0", "0", "0",
            "--safe-q-right", "0", "0", "0", "-0.5", "0", "0", "0",
            "--rotation", "1", "0", "0", "0", "1", "0", "0", "0", "1",
            "--workspace-min", "-1", "-1", "-1", "--workspace-max", "1", "1", "1",
            "--samples", "5", "--max-frames", "10", "--no-preview",
        ])

    def test_help_does_not_open_camera_or_execute_configuration(self):
        with patch.object(application, "MediaPipeCameraInput") as camera, patch.object(application, "tron2a_reference") as fk:
            with self.assertRaises(SystemExit) as caught:
                build_parser().parse_args(["dual", "calibrate-camera", "--help"])
        self.assertEqual(caught.exception.code, 0)
        camera.assert_not_called()
        fk.assert_not_called()

    def test_success_saves_then_reloads_and_closes(self):
        camera = Mock()
        camera.next_teleop_observation.side_effect = [observation(index) for index in range(5)]
        with patch.object(application, "MediaPipeCameraInput", return_value=camera) as factory:
            self.assertEqual(application.run(self.args), 0)
        factory.assert_called_once()
        self.assertTrue(factory.call_args.kwargs["invalid_hand_as_missing"])
        self.assertEqual(camera.next_teleop_observation.call_count, 5)
        camera.release.assert_called_once()
        loaded = ArmCalibration.load(self.args.output)
        self.assertEqual(loaded.metadata["sample_count"], 5)
        times = [item["control_relative_seconds"] for item in loaded.metadata["capture_trace"]]
        self.assertTrue(all(b > a for a, b in zip(times, times[1:])))

    def test_configuration_errors_do_not_open_camera(self):
        for name, value in (("model_asset_path", self.directory / "missing.task"),
                            ("pose_model_asset_path", self.directory / "missing.task"),
                            ("urdf", self.directory / "missing.urdf"),
                            ("samples", 2), ("timeout", float("nan")), ("width", 0)):
            original = getattr(self.args, name)
            setattr(self.args, name, value)
            with self.subTest(name=name), patch.object(application, "MediaPipeCameraInput") as camera:
                self.assertEqual(application.run(self.args), 1)
                camera.assert_not_called()
                self.assertFalse(self.args.output.exists())
            setattr(self.args, name, original)
        self.args.output.write_text("keep", encoding="utf-8")
        with patch.object(application, "MediaPipeCameraInput") as camera:
            self.assertEqual(application.run(self.args), 1)
            camera.assert_not_called()
        self.assertEqual(self.args.output.read_text(encoding="utf-8"), "keep")

    def test_cancel_failure_and_insufficient_frames_close_and_do_not_save(self):
        for failure, code in ((KeyboardInterrupt(), 130), (RuntimeError("camera lost"), 1), (None, 1)):
            camera = Mock()
            camera.next_teleop_observation.side_effect = failure if failure else [observation(index, invalid="left") for index in range(10)]
            with self.subTest(failure=failure), patch.object(application, "MediaPipeCameraInput", return_value=camera):
                self.assertEqual(application.run(self.args), code)
            camera.release.assert_called_once()
            self.assertFalse(self.args.output.exists())

    def test_preview_cancel_releases_camera_and_window_then_can_restart(self):
        self.args.preview = True
        camera = Mock()
        camera.next_teleop_observation.return_value = observation()
        with patch.object(application, "MediaPipeCameraInput", return_value=camera), \
             patch.object(application, "_preview", return_value=True), \
             patch("cv2.destroyWindow") as destroy:
            self.assertEqual(application.run(self.args), 130)
        destroy.assert_called_once()
        camera.release.assert_called_once()
        self.assertFalse(self.args.output.exists())
        self.args.preview = False
        self.test_success_saves_then_reloads_and_closes()

    def test_preview_failure_and_write_failure_release_camera(self):
        for cause in ("preview", "write"):
            camera = Mock()
            camera.next_teleop_observation.side_effect = [observation(index) for index in range(5)]
            self.args.preview = cause == "preview"
            target = "_preview" if self.args.preview else "write_calibration"
            with patch.object(application, "MediaPipeCameraInput", return_value=camera), \
                 patch.object(application, target, side_effect=OSError("failure")), patch("cv2.destroyWindow"):
                self.assertEqual(application.run(self.args), 1)
            camera.release.assert_called_once()
            self.assertFalse(self.args.output.exists())

    def test_reload_failure_removes_new_result_and_releases_camera(self):
        camera = Mock()
        camera.next_teleop_observation.side_effect = [observation(index) for index in range(5)]
        with patch.object(application, "MediaPipeCameraInput", return_value=camera), \
             patch.object(ArmCalibration, "load", side_effect=ValueError("read-back failed")):
            self.assertEqual(application.run(self.args), 1)
        camera.release.assert_called_once()
        self.assertFalse(self.args.output.exists())

    def test_camera_initialization_failure_and_timeout_do_not_save(self):
        with patch.object(application, "MediaPipeCameraInput", side_effect=RuntimeError("model initialization")):
            self.assertEqual(application.run(self.args), 1)
        camera = Mock()
        with patch.object(application, "MediaPipeCameraInput", return_value=camera), \
             patch.object(application.RelativeControlClock, "next_seconds", return_value=121.):
            self.assertEqual(application.run(self.args), 1)
        camera.next_teleop_observation.assert_not_called()
        camera.release.assert_called_once()
        self.assertFalse(self.args.output.exists())





class CameraArmMappingTests(unittest.TestCase):
    def setUp(self):
        self.calibration = synthetic_calibration()

    def retargeter(self, **kwargs):
        from teleoperation.retargeting.arm.retargeter import ArmRetargeter
        return ArmRetargeter(self.calibration, input_mode="camera_world", **kwargs)

    def test_neutral_offset_rotated_scaled_axes_and_independent_shoulders(self):
        rotation = Rotation.from_euler("xyz", [0.2, -0.4, 0.7]).as_matrix()
        self.calibration = replace(self.calibration, rotation=rotation, scale=np.array([0.5, 1.2, 0.8]), translation=np.array([0.01, -0.02, 0.03]))
        for side in ("left", "right"):
            for axis in range(3):
                raw = observation()
                # Make normalized points visibly incompatible with metre mapping.
                raw.upper_body.arms[side][:] = 99
                raw.upper_body.world_arms[side][:] += [1., 2., 3.]
                delta = np.eye(3)[axis] * 0.05
                raw.upper_body.world_arms[side][2] += delta
                mapper = self.retargeter()
                target = mapper.update(raw.upper_body, side)
                expected = self.calibration.robot_neutral_pose[side][:3] + self.calibration.translation + rotation @ (self.calibration.scale * delta)
                np.testing.assert_allclose(target.pose[:3], expected, atol=1e-7)
                np.testing.assert_allclose(target.pose[3:], self.calibration.robot_neutral_pose[side][3:], atol=1e-7)
                other = "right" if side == "left" else "left"
                np.testing.assert_allclose(mapper.update(raw.upper_body, other).pose[:3], self.calibration.robot_neutral_pose[other][:3] + self.calibration.translation, atol=1e-7)

    def test_forearm_neutral_alignment_and_degenerate_position_still_updates(self):
        mapper = self.retargeter(orientation_mode="forearm")
        for side in ("left", "right"):
            neutral = mapper.update(observation().upper_body, side)
            np.testing.assert_allclose(Rotation.from_quat(neutral.pose[3:]).as_matrix(), Rotation.from_quat(self.calibration.robot_neutral_pose[side][3:]).as_matrix(), atol=1e-6)
        straight = observation(straight=True)
        target = mapper.update(straight.upper_body, "left")
        self.assertTrue(target.valid)
        self.assertTrue(mapper.orientation_held["left"])
        np.testing.assert_allclose(target.pose[3:], self.calibration.robot_neutral_pose["left"][3:], atol=1e-7)
        self.assertGreater(np.linalg.norm(target.pose[:3] - self.calibration.robot_neutral_pose["left"][:3]), 0.05)
        straight.upper_body.world_arms["left"][:] = np.nan
        self.assertFalse(mapper.update(straight.upper_body, "left").valid)
        self.assertTrue(mapper.update(straight.upper_body, "right").valid)

    def test_missing_invalid_nonfinite_and_calibration_source_checks(self):
        mapper = self.retargeter()
        for raw in (observation(invalid="left"), observation(no_world=True), observation(missing=True)):
            self.assertFalse(mapper.update(raw.upper_body, "left").valid)
        raw = observation()
        raw.upper_body.world_arms["left"][0, 0] = np.inf
        self.assertFalse(mapper.update(raw.upper_body, "left").valid)
        for key, value in (("units", "normalized"), ("human_reference", "hip"), ("coordinate_source", "other")):
            metadata = dict(self.calibration.metadata, **{key: value})
            from teleoperation.retargeting.arm.retargeter import ArmRetargeter
            with self.subTest(key=key), self.assertRaises(ValueError):
                ArmRetargeter(replace(self.calibration, metadata=metadata), input_mode="camera_world")
        with self.assertRaises(ValueError):
            mapper.update(replace(observation().upper_body, units="m"), "left")

    def test_near_straight_and_plane_sign_crossing_do_not_flip_roll(self):
        mapper = self.retargeter(orientation_mode="forearm")
        raw = observation()
        first = mapper.update(raw.upper_body, "left")
        raw.upper_body.world_arms["left"][2] = [.2, .45, -.001]
        held = mapper.update(raw.upper_body, "left")
        self.assertTrue(held.valid)
        self.assertTrue(mapper.orientation_held["left"])
        np.testing.assert_allclose(first.pose[3:], held.pose[3:])
        raw.upper_body.world_arms["left"][2] = [.2, .45, .02]
        crossed = mapper.update(raw.upper_body, "left")
        angle = (Rotation.from_quat(first.pose[3:]).inv() * Rotation.from_quat(crossed.pose[3:])).magnitude()
        self.assertLess(angle, 1.)

    def test_legacy_absolute_wrist_behavior_is_unchanged(self):
        from teleoperation.retargeting.arm.retargeter import ArmRetargeter
        mapper = ArmRetargeter(self.calibration)
        raw = observation()
        target = mapper.update(raw.upper_body, "left")
        expected = self.calibration.robot_neutral_pose["left"][:3] + raw.upper_body.arms["left"][2] - self.calibration.human_neutral_wrist["left"]
        np.testing.assert_allclose(target.pose[:3], expected, atol=1e-7)
        self.assertFalse(mapper.update(observation(straight=True).upper_body, "left").valid)


class ArmControlSafetyTests(unittest.TestCase):
    def setUp(self):
        from teleoperation.applications.dual import Tron2ADualArmController
        self.calibration = synthetic_calibration()
        self.controller = Tron2ADualArmController(self.calibration, DEFAULT_TRON2A_URDF, input_mode="camera_world")

    def test_independent_pose_without_hands_and_ik_failure(self):
        self.controller.update(observation().upper_body, 0.)
        self.assertTrue(all(d["ik_success"] for d in self.controller.diagnostics.values()))
        raw = observation(invalid="left")
        self.controller.update(raw.upper_body, .03)
        self.assertFalse(self.controller.diagnostics["left"]["ik_attempted"])
        self.assertTrue(self.controller.diagnostics["right"]["ik_success"])
        from teleoperation.contracts.arm import ArmTargetPose
        unreachable = ArmTargetPose("left", np.array([.95, .95, -.95, 0, 0, 0, 1.]))
        with patch.object(self.controller.arm_retargeter, "update", return_value=unreachable):
            self.assertFalse(self.controller.update_side(raw.upper_body, "left", .06))
        self.assertEqual(self.controller.diagnostics["left"]["status"], "loss_braking")

    def test_outside_workspace_is_rejected_without_hiding_target(self):
        raw = observation(shift=50.)
        with patch.object(self.controller.solvers["left"], "solve") as solver:
            self.assertFalse(self.controller.update_side(raw.upper_body, "left", 0.))
        solver.assert_not_called()
        self.assertTrue(self.controller.diagnostics["left"]["workspace_rejected"])
        self.assertGreater(self.controller.diagnostics["left"]["target_pose"][0], 1.)

    def test_original_dual_keeps_hand_pipeline_and_call_order(self):
        from teleoperation.applications.dual import DualTeleoperator
        from teleoperation.applications.hand_processing import hand_coordinate_alignment
        from teleoperation.contracts.hand import L21HandAngles, HandRetargetResult
        events = []
        hand = Mock(coordinate_alignment=hand_coordinate_alignment("legacy"))
        pipeline = Mock()
        pipeline.process_frame.side_effect = lambda raw: (events.append("hands_preprocess") or object(), None)
        adapter = Mock()
        adapter.retarget.side_effect = lambda window: events.append("hands_model") or HandRetargetResult(
            L21HandAngles(np.zeros(18)), L21HandAngles(np.zeros(18)), True, True)
        with patch("teleoperation.applications.hand_processing.DualHandPipeline", return_value=pipeline), patch("teleoperation.retargeting.hand.interface.PoseTransformerRetargeter", return_value=adapter):
            dual = DualTeleoperator(hand, "cpu", self.calibration, DEFAULT_TRON2A_URDF)
        def arm_update(body, side, time):
            events.append(side + "_arm")
            return True
        with patch.object(dual.arms, "update_side", side_effect=arm_update):
            command = dual.update(dict(hands={"left":None, "right":None}, arms=observation().upper_body.arms, timestamp=.1))
        self.assertEqual(events, ["hands_preprocess", "hands_model", "left_arm", "right_arm"])
        self.assertTrue(command.left_hand_valid and command.right_hand_valid)
        self.assertIs(dual.controllers, dual.arms.controllers)
        self.assertEqual(dual.arm_retargeter.input_mode, "legacy")
        dual.close()
        pipeline.reset.assert_called_once(); adapter.reset.assert_called_once()

    def test_ik_reachable_fk_targets_and_reject_unchecked_success(self):
        from teleoperation.retargeting.arm.ik import target_pose_from_matrix
        from types import SimpleNamespace
        for side, solver in self.controller.solvers.items():
            q = self.calibration.safe_q[side].copy()
            q[0] += .1
            q[1] += .1
            target = target_pose_from_matrix(solver.specification.robot.fkine(q).A, side)
            result = solver.solve(target, self.calibration.safe_q[side], self.calibration.safe_q[side])
            self.assertTrue(result.success)
            self.assertLessEqual(result.position_error, .01)
            self.assertLessEqual(result.orientation_error, .15)
            bads = (np.full(7, np.nan), solver.specification.upper + 1,
                    self.calibration.safe_q[side], np.zeros(6))
            # Use a distinctly separated FK target for the wrong-q residual check.
            q[0] += .5
            target = target_pose_from_matrix(solver.specification.robot.fkine(q).A, side)
            for bad in bads:
                with patch.object(self.controller.arm_retargeter, "update", return_value=target), patch.object(solver, "solve", return_value=SimpleNamespace(success=True, q=bad)):
                    self.assertFalse(self.controller.update_side(observation().upper_body, side, 1.))

    def test_mismatched_urdf_reference_and_unsafe_q_are_rejected(self):
        from teleoperation.applications.dual import Tron2ADualArmController
        poses = {s:p.copy() for s,p in self.calibration.robot_neutral_pose.items()}
        poses["left"][0] += .01
        with self.assertRaises(ValueError):
            Tron2ADualArmController(replace(self.calibration, robot_neutral_pose=poses), DEFAULT_TRON2A_URDF, input_mode="camera_world")
        joints = {s:q.copy() for s,q in self.calibration.safe_q.items()}
        joints["left"][0] = 99
        with self.assertRaises(ValueError):
            Tron2ADualArmController(replace(self.calibration, safe_q=joints), DEFAULT_TRON2A_URDF, input_mode="camera_world")

    def test_loss_brakes_residual_velocity_timeout_returns_and_recovers(self):
        safety = self.controller.controllers["left"]
        target = safety.safe_q + .15
        safety.update(None, False, 0.)
        np.testing.assert_array_equal(safety.q, safety.safe_q)
        for i in range(1, 9):
            safety.update(target, True, i / 100.)
        before = safety.q.copy()
        velocity = safety.velocity.copy()
        safety.update(None, False, .09)
        self.assertEqual(safety.status, "loss_braking")
        self.assertTrue(np.any(safety.q != before))
        self.assertLess(np.linalg.norm(safety.velocity), np.linalg.norm(velocity))
        for i in range(10, 55):
            safety.update(None, False, i / 100.)
        np.testing.assert_allclose(safety.velocity, 0, atol=1e-10)
        held = safety.q.copy()
        safety.update(None, False, .59)
        self.assertEqual(safety.status, "timeout_return")
        for i in range(60, 150):
            safety.update(None, False, i / 100.)
        self.assertLess(np.linalg.norm(safety.q - safety.safe_q), np.linalg.norm(held - safety.safe_q))
        previous = safety.q.copy()
        safety.update(target, True, 1.5)
        self.assertEqual(safety.status, "tracking")
        self.assertLess(np.max(np.abs(safety.q - previous)), .01)

    def test_initial_loss_bad_times_and_large_gap_do_not_jump(self):
        safety = self.controller.controllers["left"]
        for timestamp in (0., .01, .02, .6):
            safety.update(None, False, timestamp)
            np.testing.assert_array_equal(safety.q, safety.safe_q)
        for timestamp in (.6, .5, -1., np.nan, np.inf):
            previous = safety.q.copy()
            safety.update(safety.safe_q + .5, True, timestamp)
            self.assertEqual(safety.status, "time_invalid")
            np.testing.assert_array_equal(safety.q, previous)
        safety.update(safety.safe_q + .5, True, .61)
        previous = safety.q.copy()
        safety.update(safety.safe_q + .5, True, 10.)
        self.assertTrue(safety.status.startswith("time_gap_"))
        self.assertLess(np.max(np.abs(safety.q - previous)), .05)

    def test_joint_bounds_velocity_acceleration_with_irregular_times_and_reversals(self):
        for side, safety in self.controller.controllers.items():
            rng = np.random.default_rng(923)
            time = 0.
            safety.update(None, False, time)
            for i in range(1800):
                dt = rng.uniform(.001, .1)
                time += dt
                q, v = safety.q.copy(), safety.velocity.copy()
                spec = safety.specification
                desired = spec.upper if i % 400 < 200 else spec.lower
                safety.update(desired, i % 91 != 0, time)
                self.assertTrue(np.all(safety.q >= spec.lower - 1e-8), side)
                self.assertTrue(np.all(safety.q <= spec.upper + 1e-8), side)
                self.assertTrue(np.all(np.abs(safety.velocity) <= spec.velocity + 1e-8), side)
                self.assertTrue(np.all(np.abs(safety.velocity - v) <= safety.max_acceleration * dt + 1e-8), side)
                self.assertTrue(np.all(np.abs(safety.q - q) <= spec.velocity * dt + 1e-8), side)


class CameraSimulationTests(unittest.TestCase):
    def test_simplification_preserves_all_joints_inertias_and_source(self):
        import xml.etree.ElementTree as ET
        from teleoperation.simulation.tron2a_world import _prepare_urdf
        source = DEFAULT_TRON2A_URDF.read_bytes()
        with tempfile.TemporaryDirectory() as directory:
            target = _prepare_urdf(DEFAULT_TRON2A_URDF, Path(directory), display="simplified")
            a, b = ET.fromstring(source), ET.parse(target).getroot()
            self.assertEqual([ET.tostring(j) for j in a.findall("joint")], [ET.tostring(j) for j in b.findall("joint")])
            self.assertEqual([ET.tostring(j) for j in a.findall(".//inertial")], [ET.tostring(j) for j in b.findall(".//inertial")])
            self.assertEqual(b.findall(".//mesh"), [])
        self.assertEqual(DEFAULT_TRON2A_URDF.read_bytes(), source)

    def test_full_resource_resolution_checks_every_mesh_and_both_package_prefixes(self):
        from teleoperation.simulation.tron2a_world import _prepare_urdf
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "urdf").mkdir(); (root / "meshes").mkdir()
            asset = root / "meshes" / "link.STL"
            asset.write_text("stub")
            source = root / "urdf" / "robot.urdf"
            for prefix in ("tron2", "tron2a"):
                source.write_text(f'<robot><link name="base"><visual><geometry><mesh filename="package://robot_description/{prefix}/DACH_TRON2A/meshes/link.STL"/></geometry></visual></link></robot>')
                prepared = _prepare_urdf(source, root).read_text()
                self.assertIn(asset.as_posix(), prepared)
                self.assertNotIn("package://", prepared)
            asset.unlink()
            with self.assertRaisesRegex(FileNotFoundError, "simplified"):
                _prepare_urdf(source, root)

    def test_headless_left_right_and_bilateral_follow_actual_fk(self):
        try:
            import pybullet
        except ImportError:
            self.skipTest("pybullet not installed: actual simulator tracking unverified")
        from teleoperation.simulation.tron2a_world import Tron2ADualArmWorld
        from teleoperation.applications.dual import Tron2ADualArmController
        from teleoperation.retargeting.arm.ik import target_pose_from_matrix, _pose_matrix, _pose_error
        calibration = synthetic_calibration()
        for moving in (("left",), ("right",), ("left", "right")):
            ctrl = Tron2ADualArmController(calibration, DEFAULT_TRON2A_URDF, input_mode="camera_world")
            world = Tron2ADualArmWorld(DEFAULT_TRON2A_URDF, gui=False, display="simplified", attach_hands=False, safe_q=calibration.safe_q)
            temporary = Path(world._temporary.name)
            try:
                for side in ("left", "right"):
                    initial = world.actual_state()["arms"][side]["pose"]
                    np.testing.assert_allclose(initial[:3], calibration.robot_neutral_pose[side][:3], atol=1e-6)
                targets = {}
                for side in ("left", "right"):
                    q = calibration.safe_q[side].copy()
                    if side in moving:
                        q[0] += .12; q[1] += .08
                    targets[side] = target_pose_from_matrix(ctrl.solvers[side].specification.robot.fkine(q).A, side)
                with patch.object(ctrl.arm_retargeter, "update", side_effect=lambda body, side: targets[side]):
                    for frame in range(100):
                        ctrl.update(observation().upper_body, frame / 30.)
                        world.apply_arms({s:c.q for s,c in ctrl.controllers.items()})
                        for _ in range(8):
                            world.step()
                actual = world.actual_state()
                for side in targets:
                    errors = _pose_error(_pose_matrix(targets[side].pose), _pose_matrix(actual["arms"][side]["pose"]))
                    self.assertLessEqual(errors[0], .02, (moving, side, errors))
                    self.assertLessEqual(errors[1], .15)
                    spec = ctrl.solvers[side].specification
                    self.assertTrue(np.all(actual["arms"][side]["q"] >= spec.lower - 1e-5))
                    self.assertTrue(np.all(actual["arms"][side]["q"] <= spec.upper + 1e-5))
                self.assertLess(max(map(abs, actual["noncontrolled_joint_drift"].values()), default=0), 1e-5)
            finally:
                world.close()
                world.close()
            self.assertFalse(pybullet.isConnected(world.client))
            self.assertFalse(temporary.exists())

    def test_initialization_failure_disconnects_and_cleans_temp(self):
        try:
            import pybullet
        except ImportError:
            self.skipTest("pybullet not installed: connection cleanup unverified")
        from teleoperation.simulation.tron2a_world import Tron2ADualArmWorld
        client = pybullet.connect(pybullet.DIRECT)
        with patch.object(pybullet, "connect", return_value=client), patch.object(pybullet, "loadURDF", side_effect=RuntimeError("load failed")):
            with self.assertRaisesRegex(RuntimeError, "load failed"):
                Tron2ADualArmWorld(DEFAULT_TRON2A_URDF, gui=False, display="simplified", attach_hands=False)
        self.assertFalse(pybullet.isConnected(client))

    def test_optional_hand_geometry_holds_fixed_reference(self):
        try:
            import pybullet
        except ImportError:
            self.skipTest("pybullet not installed: hand geometry reference unverified")
        from teleoperation.simulation.tron2a_world import Tron2ADualArmWorld
        calibration = synthetic_calibration()
        world = Tron2ADualArmWorld(DEFAULT_TRON2A_URDF, gui=False, display="simplified", attach_hands=True,
                                   hand_reference_only=True, safe_q=calibration.safe_q)
        try:
            for _ in range(80):
                world.step()
            for side, body in world.hand_bodies.items():
                states = pybullet.getJointStates(body, world.hand_indices[side], physicsClientId=world.client)
                np.testing.assert_allclose([s[0] for s in states], 0, atol=1e-10)
                np.testing.assert_allclose([s[1] for s in states], 0, atol=1e-10)
        finally:
            world.close()

    def test_pose_world_mapping_through_real_ik_safety_and_simulation(self):
        try:
            import pybullet
        except ImportError:
            self.skipTest("pybullet not installed: complete pose-world pipeline unverified")
        from teleoperation.simulation.tron2a_world import Tron2ADualArmWorld
        from teleoperation.applications.dual import Tron2ADualArmController
        from teleoperation.retargeting.arm.ik import _pose_matrix, _pose_error
        calibration = synthetic_calibration()
        for moving in (("left",), ("right",), ("left", "right")):
            controller = Tron2ADualArmController(calibration, DEFAULT_TRON2A_URDF, input_mode="camera_world")
            world = Tron2ADualArmWorld(DEFAULT_TRON2A_URDF, gui=False, display="simplified", attach_hands=False, safe_q=calibration.safe_q)
            try:
                for frame in range(150):
                    raw = observation(frame)
                    if frame:
                        for side in moving:
                            raw.upper_body.world_arms[side][2] += [.015, -.01, .01]
                    controller.update(raw.upper_body, frame / 30.)
                    world.apply_arms({s:c.q for s,c in controller.controllers.items()})
                    for _ in range(8):
                        world.step()
                actual = world.actual_state()
                for side in ("left", "right"):
                    self.assertTrue(controller.diagnostics[side]["ik_success"])
                    target = controller.diagnostics[side]["target_pose"]
                    errors = _pose_error(_pose_matrix(target), _pose_matrix(actual["arms"][side]["pose"]))
                    self.assertLessEqual(errors[0], .02, (moving, side, errors))
                    self.assertLessEqual(errors[1], .15)
            finally:
                world.close()


class CameraRuntimeTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.directory = Path(self.temporary.name)
        self.calibration = synthetic_calibration()
        # Synthetic observations may exercise orchestration in tests only.
        write_calibration(self.directory / "neutral.json", self.calibration.to_dict())
        for model in ("hands.task", "pose.task"):
            (self.directory / model).write_bytes(b"mock")
        self.args = build_parser().parse_args(["dual", "camera", "--arms-only", "--calibration", str(self.directory / "neutral.json"),
            "--model-asset-path", str(self.directory / "hands.task"), "--pose-model-asset-path", str(self.directory / "pose.task"),
            "--display", "simplified", "--no-gui", "--no-preview", "--frames", "2", "--report", str(self.directory / "report.json")])

    def test_help_is_static_and_lists_both_modes_and_checkpoints(self):
        import contextlib
        import io
        stream = io.StringIO()
        with contextlib.redirect_stdout(stream), patch.object(application, "MediaPipeCameraInput") as camera:
            with self.assertRaises(SystemExit) as caught:
                build_parser().parse_args(["dual", "camera", "--help"])
        self.assertEqual(caught.exception.code, 0)
        self.assertIn("--arms-only", stream.getvalue())
        for option in ("--with-hands", "--checkpoint", "--left-checkpoint", "--right-checkpoint", "--device"):
            self.assertIn(option, stream.getvalue())
        camera.assert_not_called()

    def test_preview_includes_actual_pose_and_separate_tracking_errors(self):
        from teleoperation.applications.dual import Tron2ADualArmController
        controller = Tron2ADualArmController(self.calibration, DEFAULT_TRON2A_URDF, input_mode="camera_world")
        raw = observation(invalid="left")
        controller.update(raw.upper_body, 0.)
        actual = dict(arms={side: dict(pose=self.calibration.robot_neutral_pose[side]) for side in ("left", "right")})
        camera = Mock(last_bgr=np.zeros((240, 640, 3), dtype=np.uint8))
        with patch("cv2.imshow"), patch("cv2.waitKey", return_value=27), patch("cv2.putText") as text:
            self.assertTrue(application._runtime_preview(camera, raw, controller, actual, "test"))
        messages = [call.args[1] for call in text.call_args_list]
        self.assertTrue(any("EE=" in message and "target err=" in message and "control err=" in message for message in messages))
        self.assertTrue(any("Pose=--" in message for message in messages))

    def test_real_headless_runtime_missing_hands_report_and_no_hand_models(self):
        try:
            import pybullet
        except ImportError:
            self.skipTest("pybullet not installed: runtime simulator unverified")
        camera = Mock()
        camera.next_teleop_observation.side_effect = [observation(), observation(1, invalid="left")]
        with patch.object(application, "MediaPipeCameraInput", return_value=camera), patch("teleoperation.retargeting.hand.predictor.create_twohand_retargeter", side_effect=AssertionError("hand checkpoint loaded")), patch("teleoperation.applications.hand_processing.DualHandPipeline", side_effect=AssertionError("hand window created")), patch.object(application.time, "sleep"):
            self.assertEqual(application.run_camera(self.args), 0)
        camera.release.assert_called_once()
        report = read_calibration(self.args.report)
        self.assertEqual(report["frames"], 2)
        self.assertEqual(report["sides"]["left"]["invalid_observations"], 1)
        self.assertEqual(report["sides"]["right"]["ik_success"], 2)
        self.assertIn("actual_vs_target_position_m", report["sides"]["right"]["metrics"])
        self.assertEqual(report["sides"]["right"]["last"]["detection_unix_ms"], observation(1).upper_body.timestamp)
        self.assertLess(report["sides"]["right"]["last"]["control_relative_seconds"], 10)
        self.assertIn("capture_to_actual_ms", report["latency"])
        self.assertFalse(report["real_camera_verified"])

    def test_invalid_configuration_and_synthetic_fixture_do_not_open_devices(self):
        for name, value in (("frames", -1), ("simulation_steps", 0), ("pose_min_visibility", np.nan), ("report", self.args.calibration),
                            ("calibration", Path(__file__).parent / "fixtures/tron2a_camera_neutral.synthetic.json")):
            old = getattr(self.args, name)
            setattr(self.args, name, value)
            with patch.object(application, "MediaPipeCameraInput") as camera, patch("teleoperation.simulation.tron2a_world.Tron2ADualArmWorld") as world:
                self.assertEqual(application.run_camera(self.args), 1)
                camera.assert_not_called(); world.assert_not_called()
            setattr(self.args, name, old)

    def test_interrupt_capture_and_preview_errors_close_resources_and_report(self):
        world = Mock()
        world.actual_state.return_value = dict(arms={s: dict(q=q, velocity=np.zeros(7), pose=self.calibration.robot_neutral_pose[s]) for s,q in self.calibration.safe_q.items()}, noncontrolled_joint_drift={})
        for error, code in ((KeyboardInterrupt(), 130), (RuntimeError("capture lost"), 1)):
            camera = Mock()
            camera.next_teleop_observation.side_effect = error
            world.reset_mock()
            with patch.object(application, "MediaPipeCameraInput", return_value=camera), patch("teleoperation.simulation.tron2a_world.Tron2ADualArmWorld", return_value=world):
                self.assertEqual(application.run_camera(self.args), code)
            camera.release.assert_called_once(); world.close.assert_called_once()
            self.assertTrue(self.args.report.is_file())
        self.args.preview = True
        for result in (True, RuntimeError("preview failed")):
            camera = Mock(); camera.next_teleop_observation.return_value = observation()
            world.reset_mock()
            with patch.object(application, "MediaPipeCameraInput", return_value=camera), patch("teleoperation.simulation.tron2a_world.Tron2ADualArmWorld", return_value=world), patch.object(application, "_runtime_preview", side_effect=result if isinstance(result, Exception) else None, return_value=True), patch("cv2.destroyWindow") as destroy:
                self.assertEqual(application.run_camera(self.args), 1 if isinstance(result, Exception) else 0)
            camera.release.assert_called_once(); world.close.assert_called_once(); destroy.assert_called_once()

    def test_camera_initialization_failure_and_simulation_window_close_release_world(self):
        world = Mock()
        with patch.object(application, "MediaPipeCameraInput", side_effect=RuntimeError("camera init failed")), patch("teleoperation.simulation.tron2a_world.Tron2ADualArmWorld", return_value=world):
            self.assertEqual(application.run_camera(self.args), 1)
        world.close.assert_called_once()
        self.assertEqual(read_calibration(self.args.report)["exit_reason"], "error")
        world.reset_mock()
        world.is_connected.side_effect = [True, False]
        camera = Mock()
        camera.next_teleop_observation.side_effect = RuntimeError("window closed during capture")
        with patch.object(application, "MediaPipeCameraInput", return_value=camera), patch("teleoperation.simulation.tron2a_world.Tron2ADualArmWorld", return_value=world):
            self.assertEqual(application.run_camera(self.args), 0)
        camera.release.assert_called_once(); world.close.assert_called_once()
        self.assertEqual(read_calibration(self.args.report)["exit_reason"], "simulation_window_closed")


def dual_world_observation(index=0, left=True, right=True, *, invalid_pose=None):
    from tests.test_world_hand_pipeline import world_frame, observation as hand_observation
    hands = hand_observation(world_frame(index, left, right))
    pose = observation(index, invalid=invalid_pose)
    hands = replace(hands, metadata={**hands.metadata, **pose.hands.metadata})
    return RawTeleopObservation(hands, replace(pose.upper_body, timestamp=hands.timestamp))


def world_predictor():
    from teleoperation.contracts.coordinates import WORLD_PALM_LOCAL_COORDINATE_ALIGNMENT
    model = Mock(coordinate_alignment=WORLD_PALM_LOCAL_COORDINATE_ALIGNMENT)
    def predict(payload, device):
        return dict(hands={side: None if values is None else np.linspace(0, .3, 18, dtype=np.float32)
                          for side, values in payload["hands"].items()})
    model.predict.side_effect = predict
    return model


def write_canonical_checkpoint(path):
    """Real model state for integration plumbing, not an accuracy benchmark."""
    import torch
    from teleoperation.retargeting.hand.config import L21
    from teleoperation.retargeting.hand.transformer import PoseTransformer
    from teleoperation.contracts.coordinates import WORLD_PALM_LOCAL_METADATA, CANONICAL_ANGLE_CONVENTION
    torch.manual_seed(123)
    model = PoseTransformer(**L21.model_kwargs())
    torch.save(dict(model_pos=model.state_dict(), hand_side="shared", training_robot="right",
                    angle_convention=CANONICAL_ANGLE_CONVENTION, **WORLD_PALM_LOCAL_METADATA), path)
    return path


class CameraDualTeleoperatorTests(unittest.TestCase):
    def setUp(self):
        from teleoperation.applications.dual import DualTeleoperator
        self.dual = DualTeleoperator(world_predictor(), "cpu", synthetic_calibration(), DEFAULT_TRON2A_URDF,
                                     input_mode="camera_world")
        self.addCleanup(self.dual.close)

    def test_three_frame_wait_single_loss_recovery_and_independent_pose(self):
        commands = [self.dual.update(dual_world_observation(i), control_seconds=i / 30) for i in range(3)]
        self.assertEqual([c.left_hand_valid for c in commands], [False, False, True])
        self.assertTrue(all(c.left_arm_valid and c.right_arm_valid for c in commands))
        held = commands[-1].left_hand.copy()
        lost = self.dual.update(dual_world_observation(3, left=False), control_seconds=.1)
        self.assertFalse(lost.left_hand_valid)
        self.assertTrue(lost.right_hand_valid and lost.left_arm_valid)
        np.testing.assert_array_equal(lost.left_hand, held)
        self.assertEqual(self.dual.hand_pipeline.valid_streak["left"], 0)
        for i in (4, 5, 6):
            cmd = self.dual.update(dual_world_observation(i, invalid_pose="right"), control_seconds=i / 30)
            self.assertEqual(cmd.left_hand_valid, i == 6)
            self.assertTrue(cmd.right_hand_valid and cmd.left_arm_valid)
            self.assertFalse(cmd.right_arm_valid)
        raw = dual_world_observation(7)
        raw.upper_body.world_arms["left"][:] = np.nan
        cmd = self.dual.update(raw, control_seconds=7 / 30)
        self.assertTrue(cmd.left_hand_valid and cmd.right_hand_valid and cmd.right_arm_valid)
        self.assertFalse(cmd.left_arm_valid)
        self.assertFalse(self.dual.arms.diagnostics["left"]["ik_attempted"])

    def test_seconds_raw_preservation_48d_order_and_array_independence(self):
        from teleoperation.retargeting.hand.angles import angle18_to_dofs
        raw = dual_world_observation()
        original_metadata = dict(raw.hands.metadata)
        for i in range(3):
            command = self.dual.update(raw, control_seconds=i * .033)
        self.assertEqual(command.timestamp, .066)
        self.assertIs(self.dual.last_observation, raw)
        self.assertEqual(raw.hands.metadata, original_metadata)
        self.assertEqual(raw.hands.timestamp, 1_700_000_000_000)
        payload = self.dual.hand_retargeter.retargeter.predict.call_args.args[0]
        self.assertEqual(payload["timestamp"], .066)
        self.assertEqual(payload["metadata"]["raw_timestamp_ms"], raw.hands.timestamp)
        self.assertEqual(payload["metadata"]["timestamp_unit"], "relative_seconds")
        flat = command.flatten()
        self.assertEqual(flat.shape, (48,))
        self.assertTrue(np.isfinite(flat).all())
        for name, start, stop in (("left_arm", 0, 7), ("left_hand", 7, 24), ("right_arm", 24, 31), ("right_hand", 31, 48)):
            np.testing.assert_array_equal(flat[start:stop], getattr(command, name))
        np.testing.assert_allclose(command.left_hand, angle18_to_dofs(np.linspace(0, .3, 18, dtype=np.float32)))
        before = command.flatten().copy()
        self.dual.update(dual_world_observation(3), control_seconds=.1)
        for side in ("left", "right"):
            self.dual.controllers[side].q[:] = 0
            self.dual.hand_q[side][:] = .2
            self.assertFalse(np.shares_memory(getattr(command, side + "_hand"), self.dual.hand_q[side]))
            self.assertFalse(np.shares_memory(getattr(command, side + "_arm"), self.dual.controllers[side].q))
        np.testing.assert_array_equal(command.flatten(), before)

    def test_invalid_times_reject_before_inference_or_arm_state_change(self):
        self.dual.update(dual_world_observation(), control_seconds=.1)
        for seconds in (None, .1, .09, -1, np.nan, np.inf, True, "0.2"):
            q = {s:c.q.copy() for s,c in self.dual.controllers.items()}
            with self.subTest(seconds=seconds), patch.object(self.dual.arms, "update_side") as arms:
                with self.assertRaisesRegex(ValueError, "control_seconds"):
                    self.dual.update(dual_world_observation(1), control_seconds=seconds)
                arms.assert_not_called()
            for side in q:
                np.testing.assert_array_equal(self.dual.controllers[side].q, q[side])
            self.assertEqual(self.dual.hand_pipeline.valid_streak["left"], 0)
        self.dual.update(dual_world_observation(2), control_seconds=.2)

    def test_bad_world_contract_and_model_alignment_refuse_normalized_fallback(self):
        from teleoperation.applications.dual import DualTeleoperator
        from teleoperation.contracts.coordinates import PALM_LOCAL_COORDINATE_ALIGNMENT
        for kwargs in (dict(input_mode="invalid"), dict(input_mode="camera_world", hand_preprocessing="palm-local")):
            with self.assertRaises(ValueError):
                DualTeleoperator(world_predictor(), "cpu", synthetic_calibration(), DEFAULT_TRON2A_URDF, **kwargs)
        model = Mock(coordinate_alignment=PALM_LOCAL_COORDINATE_ALIGNMENT)
        with self.assertRaisesRegex(ValueError, "mismatch"):
            DualTeleoperator(model, "cpu", synthetic_calibration(), DEFAULT_TRON2A_URDF, input_mode="camera_world")
        raw = dual_world_observation()
        with self.assertRaisesRegex(ValueError, "world_hands"):
            self.dual.update(replace(raw, hands=replace(raw.hands, world_hands=None)), control_seconds=0.)
        self.dual.close()
        self.assertEqual(self.dual.hand_pipeline.valid_streak, {"left":0, "right":0})
        self.assertEqual(self.dual.hand_diagnostics, {})

    def test_ik_failure_keeps_safe_output_and_fingers_continue(self):
        from types import SimpleNamespace
        for i in range(2):
            self.dual.update(dual_world_observation(i), control_seconds=i / 30)
        with patch.object(self.dual.solvers["left"], "solve", return_value=SimpleNamespace(success=False, q=np.full(7, np.nan))):
            cmd = self.dual.update(dual_world_observation(2), control_seconds=2 / 30)
        self.assertFalse(cmd.left_arm_valid)
        self.assertTrue(cmd.right_arm_valid and cmd.left_hand_valid and cmd.right_hand_valid)
        self.assertTrue(np.isfinite(cmd.flatten()).all())
        np.testing.assert_allclose(cmd.left_arm, self.dual.safe_q["left"])

    def test_shared_model_matches_hand_and_dual_windows_flags_and_native_angles(self):
        from teleoperation.applications.hand_processing import MediaPipePalmLocalPipeline
        from teleoperation.applications.hand_realtime import load_palm_local_retargeter
        from teleoperation.retargeting.hand.interface import PoseTransformerRetargeter
        from teleoperation.retargeting.hand.angles import angle18_to_dofs
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        checkpoint = write_canonical_checkpoint(Path(directory.name) / "shared.pth")
        model = load_palm_local_retargeter(checkpoint, device="cpu", landmark_space="world")
        self.dual.hand_retargeter = PoseTransformerRetargeter(model, "cpu")
        pipeline = MediaPipePalmLocalPipeline("world")
        original = PoseTransformerRetargeter(model, "cpu")
        for i, left in enumerate((True, True, True, False, True, True, True, True)):
            raw = dual_world_observation(i, left=left, invalid_pose="right" if i == 7 else None)
            seconds = i / 30
            frame = replace(raw.hands, timestamp=seconds, metadata={**raw.hands.metadata,
                "raw_timestamp_ms":raw.hands.timestamp, "timestamp_unit":"relative_seconds"})
            window, canonical = pipeline.process_observation(frame)
            reference = None if window is None else original.retarget(window)
            with patch.object(self.dual.hand_retargeter, "retarget", wraps=self.dual.hand_retargeter.retarget) as infer:
                command = self.dual.update(raw, control_seconds=seconds)
            for side in ("left", "right"):
                diag = self.dual.hand_diagnostics[side]
                self.assertEqual(diag["aligned_valid"], canonical.hands[side] is not None)
                self.assertEqual(diag["valid_streak"], pipeline.valid_streak[side])
                if canonical.hands[side] is not None:
                    np.testing.assert_allclose(self.dual.hand_pipeline.current_hands[side], canonical.hands[side], atol=1e-6, rtol=0)
                expected = None if reference is None else getattr(reference, side + "_angles")
                self.assertEqual(getattr(command, side + "_hand_valid"), expected is not None)
                self.assertEqual(diag["window_ready"], window is not None and window.hands[side] is not None)
                if expected is not None:
                    np.testing.assert_allclose(infer.call_args.args[0].hands[side], window.hands[side], atol=1e-6, rtol=0)
                    np.testing.assert_allclose(getattr(command, side + "_hand"), angle18_to_dofs(expected.values), atol=1e-6, rtol=0)


class CameraFullRuntimeTests(unittest.TestCase):
    def setUp(self):
        CameraRuntimeTests.setUp(self)
        self.args.arms_only = False
        self.args.with_hands = True
        self.args.attach_hands = None
        self.args.checkpoint = self.directory / "world.pth"
        self.args.checkpoint.write_bytes(b"loader mocked only in orchestration tests")

    def fake_world(self):
        world = Mock()
        world.actual_state.return_value = dict(arms={s: dict(q=q, velocity=np.zeros(7), pose=self.calibration.robot_neutral_pose[s])
            for s,q in self.calibration.safe_q.items()}, noncontrolled_joint_drift={})
        return world

    def test_cli_exclusive_modes_checkpoint_selection_and_arms_default(self):
        import contextlib
        import io
        base = ["dual", "camera", "--calibration", "neutral.json"]
        parser = build_parser()
        self.assertIsNone(parser.parse_args(base + ["--arms-only"]).checkpoint)
        for options in ([], ["--arms-only", "--with-hands"],
                        ["--with-hands", "--left-checkpoint", "left.pth"],
                        ["--with-hands", "--checkpoint", "shared.pth", "--left-checkpoint", "l.pth", "--right-checkpoint", "r.pth"]):
            with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                parser.parse_args(base + options)
        paired = parser.parse_args(base + ["--with-hands", "--left-checkpoint", "l.pth", "--right-checkpoint", "r.pth", "--device", "cpu"])
        self.assertIsNone(paired.checkpoint)
        self.assertEqual(paired.left_checkpoint, Path("l.pth"))

    def test_bad_full_configuration_rejected_before_camera_or_world(self):
        for changes in (dict(attach_hands=False), dict(checkpoint=None), dict(checkpoint=Path("missing.pth")),
                        dict(left_checkpoint=self.args.checkpoint, right_checkpoint=self.args.checkpoint),
                        dict(report=self.args.checkpoint)):
            args = type(self.args)(**{**vars(self.args), **changes})
            with self.subTest(changes=changes), patch.object(application, "MediaPipeCameraInput") as camera, \
                 patch("teleoperation.simulation.tron2a_world.Tron2ADualArmWorld") as world:
                self.assertEqual(application.run_camera(args), 1)
                camera.assert_not_called(); world.assert_not_called()
        with patch("teleoperation.applications.hand_realtime.load_palm_local_retargeter", side_effect=ValueError("coordinate mismatch")), \
             patch.object(application, "MediaPipeCameraInput") as camera, \
             patch("teleoperation.simulation.tron2a_world.Tron2ADualArmWorld") as world:
            self.assertEqual(application.run_camera(self.args), 1)
            camera.assert_not_called(); world.assert_not_called()

    def test_complete_apply_and_report_loss_recovery_statistics(self):
        self.args.frames = 7
        camera, world = Mock(), self.fake_world()
        camera.next_teleop_observation.side_effect = [dual_world_observation(i, left=i != 3) for i in range(7)]
        with patch.object(application, "MediaPipeCameraInput", return_value=camera), \
             patch("teleoperation.simulation.tron2a_world.Tron2ADualArmWorld", return_value=world) as factory, \
             patch("teleoperation.applications.hand_realtime.load_palm_local_retargeter", return_value=world_predictor()) as loader, \
             patch.object(application.RelativeControlClock, "next_seconds", side_effect=[i/30 for i in range(7)]), \
             patch.object(application.time, "sleep"):
            self.assertEqual(application.run_camera(self.args), 0)
        self.assertTrue(factory.call_args.kwargs["attach_hands"])
        self.assertFalse(factory.call_args.kwargs["hand_reference_only"])
        self.assertEqual(loader.call_args.kwargs["landmark_space"], "world")
        self.assertEqual(loader.call_args.kwargs["checkpoint_path"], str(self.args.checkpoint.resolve()))
        world.apply_arms.assert_not_called()
        self.assertEqual(world.apply.call_count, 7)
        self.assertEqual(world.step.call_count, 7 * self.args.simulation_steps)
        self.assertEqual(camera.next_teleop_observation.call_count, 7)
        commands = [c.args[0] for c in world.apply.call_args_list]
        self.assertEqual([c.left_hand_valid for c in commands], [False, False, True, False, False, False, True])
        report = read_calibration(self.args.report)
        left = report["hands"]["left"]
        self.assertEqual(left["inference_valid_frames"], 2)
        self.assertEqual(left["window_wait_frames"], 4)
        self.assertEqual(left["missing_or_invalid_frames"], 1)
        self.assertEqual(left["detection_losses"], 1)
        self.assertEqual(left["detection_recoveries"], 1)
        self.assertEqual(left["inference_recoveries"], 1)
        self.assertEqual(report["inference_device"], "cpu")
        self.assertEqual(report["hands"]["left"]["last"]["detection_unix_ms"], dual_world_observation(6).hands.timestamp)
        self.assertIn("hand_processing_ms", report["latency"])
        camera.release.assert_called_once(); world.close.assert_called_once()

    def test_full_preview_shows_independent_hand_states_device_and_weights(self):
        from teleoperation.applications.dual import DualTeleoperator
        dual = DualTeleoperator(world_predictor(), "cpu", self.calibration, DEFAULT_TRON2A_URDF, input_mode="camera_world")
        self.addCleanup(dual.close)
        for i in range(3):
            raw = dual_world_observation(i, left=False)
            dual.update(raw, control_seconds=i/30)
        dual.hand_model_labels = ["left_checkpoint: world_test_v1/left/model_best.pth"]
        camera = Mock(last_bgr=np.zeros((480, 640, 3), dtype=np.uint8))
        with patch("cv2.imshow"), patch("cv2.waitKey", return_value=27), patch("cv2.putText") as text:
            self.assertTrue(application._runtime_preview(camera, raw, dual.arms, self.fake_world().actual_state(), "test", teleoperator=dual))
        messages = [call.args[1] for call in text.call_args_list]
        self.assertTrue(any("left hand: detected=0" in message and "inference=0" in message for message in messages))
        self.assertTrue(any("right hand: detected=1" in message and "window=1 inference=1" in message for message in messages))
        self.assertTrue(any("device=cpu" in message for message in messages))
        self.assertIn(dual.hand_model_labels[0], messages)

    def test_full_lifecycle_initialization_failure_error_cancel_and_restart(self):
        from teleoperation.applications.dual import DualTeleoperator
        for failure in ("world", "camera", "capture", "interrupt", "preview", "exit", "cleanup", "report", "normal"):
            dual = DualTeleoperator(world_predictor(), "cpu", self.calibration, DEFAULT_TRON2A_URDF, input_mode="camera_world")
            camera, world = Mock(), self.fake_world()
            self.args.preview = failure in ("preview", "exit")
            self.args.attach_hands = None
            if failure == "capture":
                camera.next_teleop_observation.side_effect = RuntimeError("capture failure")
            elif failure == "interrupt":
                camera.next_teleop_observation.side_effect = KeyboardInterrupt()
            else:
                camera.next_teleop_observation.side_effect = [dual_world_observation(i) for i in range(2)]
            if failure == "cleanup":
                camera.release.side_effect = RuntimeError("release failure")
            with self.subTest(failure=failure), \
                 patch("teleoperation.applications.dual.DualTeleoperator", return_value=dual), \
                 patch.object(dual, "close", wraps=dual.close) as close, \
                 patch("teleoperation.applications.hand_realtime.load_palm_local_retargeter", return_value=world_predictor()), \
                 patch.object(application, "MediaPipeCameraInput", return_value=camera,
                              side_effect=RuntimeError("init failure") if failure == "camera" else None), \
                 patch("teleoperation.simulation.tron2a_world.Tron2ADualArmWorld", return_value=world,
                       side_effect=RuntimeError("init failure") if failure == "world" else None), \
                 patch.object(application, "_runtime_preview", return_value=True,
                              side_effect=RuntimeError("preview failure") if failure == "preview" else None), \
                 patch("cv2.destroyWindow"), patch.object(application.time, "sleep"), \
                 patch.object(application.CameraRunReport, "write", side_effect=OSError("write failure") if failure == "report" else None):
                expected = 130 if failure == "interrupt" else 0 if failure in ("normal", "exit") else 1
                self.assertEqual(application.run_camera(self.args), expected)
                close.assert_called_once()
            if failure != "world":
                world.close.assert_called_once()
            if failure not in ("world", "camera"):
                camera.release.assert_called_once()
            self.assertEqual(dual.hand_pipeline.valid_streak, {"left":0, "right":0})

    def test_shared_runtime_and_native_hand_targets_reach_pybullet(self):
        import pybullet
        from teleoperation.simulation.tron2a_world import Tron2ADualArmWorld
        self.args.checkpoint = write_canonical_checkpoint(self.directory / "shared_runtime.pth")
        self.args.left_checkpoint = self.args.right_checkpoint = None
        self.args.frames = 4
        camera = Mock()
        camera.next_teleop_observation.side_effect = [dual_world_observation(i) for i in range(4)]
        worlds, commands, finger_states = [], [], []
        def create_world(*args, **kwargs):
            world = Tron2ADualArmWorld(*args, **kwargs)
            worlds.append(world)
            original_apply = world.apply
            def apply(command):
                original_apply(command)
                commands.append(command)
            world.apply = apply
            original_state = world.actual_state
            def actual_state():
                finger_states.append({side: np.array([s[0] for s in pybullet.getJointStates(
                    body, world.hand_indices[side], physicsClientId=world.client)])
                    for side, body in world.hand_bodies.items()})
                return original_state()
            world.actual_state = actual_state
            return world
        with patch.object(application, "MediaPipeCameraInput", return_value=camera), \
             patch("teleoperation.simulation.tron2a_world.Tron2ADualArmWorld", side_effect=create_world), \
             patch.object(application.time, "sleep"):
            self.assertEqual(application.run_camera(self.args), 0)
        self.assertEqual(len(commands), 4)
        self.assertTrue(commands[-1].left_hand_valid and commands[-1].right_hand_valid)
        self.assertTrue(np.any(commands[-1].left_hand != 0))
        self.assertFalse(worlds[0].hand_reference_only)
        self.assertEqual(set(worlds[0].hand_bodies), {"left", "right"})
        self.assertTrue(np.any(np.abs(finger_states[-1]["left"]) > .001))
        self.assertTrue(np.any(np.abs(finger_states[-1]["right"]) > .001))
        self.assertFalse(pybullet.isConnected(worlds[0].client))
        report = read_calibration(self.args.report)
        self.assertEqual(report["hands"]["left"]["inference_valid_frames"], 2)
        self.assertEqual(report["hand_checkpoints"], {"checkpoint":str(self.args.checkpoint.resolve())})
        camera.release.assert_called_once()

    def test_real_shared_and_paired_loader_contracts(self):
        import torch
        from tests.test_world_hand_pipeline import small_model_kwargs
        from teleoperation.applications.hand_realtime import load_palm_local_retargeter
        from teleoperation.retargeting.hand.transformer import PoseTransformer
        from teleoperation.contracts.coordinates import WORLD_PALM_LOCAL_METADATA, PALM_LOCAL_COORDINATE_ALIGNMENT, CANONICAL_ANGLE_CONVENTION
        kwargs = small_model_kwargs()
        state = PoseTransformer(**kwargs).state_dict()
        paths = {}
        for side in ("shared", "left", "right"):
            path = self.directory / (side + ".pth")
            torch.save(dict(model_pos=state, hand_side=side, training_robot="right",
                            angle_convention=CANONICAL_ANGLE_CONVENTION, **WORLD_PALM_LOCAL_METADATA), path)
            paths[side] = path
        with patch("teleoperation.retargeting.hand.config.L21", Mock(model_kwargs=Mock(return_value=kwargs))):
            shared = load_palm_local_retargeter(paths["shared"], "cpu", landmark_space="world")
            self.assertIs(shared.model_for_side("left"), shared.model_for_side("right"))
            with self.assertRaisesRegex(ValueError, "shared checkpoint"):
                load_palm_local_retargeter(device="cpu", left_checkpoint=paths["left"], right_checkpoint=paths["right"], landmark_space="world")
            for side in ("left", "right"):
                self.assertEqual(shared.model_for_side(side).hand_side, "shared")
                for key, value in shared.model_for_side(side).state_dict().items():
                    torch.testing.assert_close(value, state[key])
            for options in (dict(checkpoint_path=paths["left"]),
                            dict(left_checkpoint=paths["right"], right_checkpoint=paths["left"])):
                with self.assertRaisesRegex(ValueError, "[Ss]hared"):
                    load_palm_local_retargeter(device="cpu", **options)
            torch.save(dict(model_pos=state, coordinate_alignment=PALM_LOCAL_COORDINATE_ALIGNMENT), paths["shared"])
            with self.assertRaisesRegex(ValueError, "alignment mismatch"):
                load_palm_local_retargeter(paths["shared"], "cpu", landmark_space="world")
            for metadata in (dict(coordinate_alignment=PALM_LOCAL_COORDINATE_ALIGNMENT),
                             {**WORLD_PALM_LOCAL_METADATA, "length_unit":"cm"}):
                torch.save(dict(model_pos=state, **metadata), self.args.checkpoint)
                with patch.object(application, "MediaPipeCameraInput") as camera, \
                     patch("teleoperation.simulation.tron2a_world.Tron2ADualArmWorld") as world:
                    self.assertEqual(application.run_camera(self.args), 1)
                    camera.assert_not_called(); world.assert_not_called()


if __name__ == "__main__":
    unittest.main()
