"""Camera-only TRON2A calibration, with synthetic observations and real URDF FK.

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


if __name__ == "__main__":
    unittest.main()
