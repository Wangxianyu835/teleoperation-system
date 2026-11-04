"""Regressions for realtime arm jitter, simulation lag and target routing."""

import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

import numpy as np

from teleoperation.simulation.arm_scene_driver import ArmSceneDriver
from teleoperation.simulation.realtime_teleop import SimulationStepClock, build_parser
from tests.test_arm_ik_multi_robot import ASSETS_AVAILABLE, PYBULLET_AVAILABLE


class ArmTimingAndOptionsTests(unittest.TestCase):
    def test_live_physics_keeps_up_with_thirteen_fps_capture(self):
        now = [0.0]
        clock = SimulationStepClock(clock=lambda: now[0])
        for _ in range(130):
            now[0] += 1 / 13
            self.assertIn(clock.next_steps(), (18, 19))
        self.assertAlmostEqual(clock.total_steps / 240, 10.0, delta=1 / 240)
        self.assertEqual(clock.capped_frames, 0)

    def test_offline_steps_do_not_depend_on_processing_speed(self):
        now = [0.0]
        clock = SimulationStepClock(fixed_steps=8, realtime=False, clock=lambda: now[0])
        for delay in (0.001, 0.033, 0.2, 5.0):
            now[0] += delay
            self.assertEqual(clock.next_steps(), 8)
        self.assertEqual(clock.total_steps, 32)

    def test_long_freeze_does_not_accumulate_a_physics_burst(self):
        now = [0.0]
        clock = SimulationStepClock(clock=lambda: now[0])
        now[0] = 3.0
        self.assertEqual(clock.next_steps(), 24)
        self.assertEqual(clock.capped_frames, 1)
        now[0] += 1 / 30
        self.assertLessEqual(clock.next_steps(), 9)

    def test_gain_option_accepts_robot_names_and_repeated_overrides(self):
        args = build_parser().parse_args([
            "--arm-gain", "g1", "0.2", "--arm-gain", "h1_2", "0.5"])
        self.assertEqual(dict(args.arm_gain), {"g1": 0.2, "h1_2": 0.5})

    def test_invalid_gain_is_rejected_by_parser(self):
        for name, gain in (("unknown", "0.2"), ("g1", "0"), ("g1", "-1"),
                           ("g1", "nan"), ("g1", "inf"), ("g1", "oops")):
            with self.subTest(name=name, gain=gain), contextlib.redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit) as error:
                    build_parser().parse_args(["--arm-gain", name, gain])
                self.assertEqual(error.exception.code, 2)


class ArmOrientationRoutingTests(unittest.TestCase):
    def setUp(self):
        self.driver = ArmSceneDriver.__new__(ArmSceneDriver)
        self.driver.sides = ("left", "right")
        self.driver.applied = {"left": 0, "right": 0}
        self.driver.ik = {}
        for name in ("g1", "h1_2"):
            adapter = Mock()
            adapter.ee = {side: (0, "wrist") for side in self.driver.sides}
            adapter.ik_mode = "pose" if name == "h1_2" else "position"
            adapter.solve.return_value = {"joint": 0.0}
            self.driver.ik[name] = adapter
        self.positions = {"g1": (0.16, 0.1, 0.1), "h1_2": (0.24, 0.05, 0.19)}

    def test_each_side_keeps_each_robots_own_orientation(self):
        for side, quaternion in (("left", (0., 0., 0., 1.)), ("right", (0., 1., 0., 0.))):
            self.driver.apply_each(side, self.positions,
                                   orientation={"g1": None, "h1_2": quaternion})
            self.driver.ik["g1"].solve.assert_called_with(
                side, self.positions["g1"], orientation=None, source="base")
            self.driver.ik["h1_2"].solve.assert_called_with(
                side, self.positions["h1_2"], orientation=quaternion, source="base")

    def test_shared_quaternion_remains_supported(self):
        quaternion = (0., 0., 0., 1.)
        self.driver.apply_each("right", self.positions, orientation=quaternion)
        for name, adapter in self.driver.ik.items():
            adapter.solve.assert_called_once_with(
                "right", self.positions[name], orientation=quaternion, source="base")

    def test_missing_orientation_fails_before_any_arm_is_changed(self):
        with self.assertRaisesRegex(ValueError, "no orientation"):
            self.driver.apply_each("right", self.positions, orientation={"g1": None})
        for adapter in self.driver.ik.values():
            adapter.solve.assert_not_called()
            adapter.apply.assert_not_called()


@unittest.skipUnless(ASSETS_AVAILABLE and PYBULLET_AVAILABLE, "robot assets and PyBullet required")
class ArmSolveStateTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from teleoperation.applications.three_robot_scene import ThreeRobotHandScene
        cls.scene = ThreeRobotHandScene(sides=("left", "right"), gui=False)
        cls.driver = ArmSceneDriver(cls.scene, sides=("left", "right"))

    @classmethod
    def tearDownClass(cls):
        cls.scene.close()

    @staticmethod
    def states(adapter):
        return np.array([adapter.p.getJointState(adapter.body, index, physicsClientId=adapter.cid)[:2]
                         for index in adapter.indices])

    def test_solving_both_arms_preserves_all_positions_and_velocities(self):
        for name, adapter in self.driver.ik.items():
            for side in ("left", "right"):
                with self.subTest(robot=name, side=side):
                    index = adapter.arm_ids[side][0]
                    q = adapter.p.getJointState(adapter.body, index, physicsClientId=adapter.cid)[0]
                    adapter.p.resetJointState(adapter.body, index, q, targetVelocity=0.12,
                                               physicsClientId=adapter.cid)
                    try:
                        before = self.states(adapter)
                        values = adapter.solve(side, adapter.profile.realtime_neutral(side))
                        self.assertEqual(len(values), 7)
                        np.testing.assert_array_equal(self.states(adapter), before)
                    finally:
                        adapter.p.resetJointState(adapter.body, index, q, physicsClientId=adapter.cid)

    def test_failed_second_pass_restores_physics_and_keeps_previous_target(self):
        for name, adapter in self.driver.ik.items():
            for side in ("left", "right"):
                with self.subTest(robot=name, side=side):
                    original = adapter._ik_call
                    target = adapter.last_target_world.get(side)
                    before = self.states(adapter)
                    calls = [0]

                    def fail_second(*args):
                        calls[0] += 1
                        if calls[0] == 2:
                            raise RuntimeError("second IK pass failed")
                        return original(*args)

                    with patch.object(adapter, "_ik_call", side_effect=fail_second):
                        with self.assertRaisesRegex(RuntimeError, "second IK pass failed"):
                            adapter.solve(side, adapter.profile.realtime_neutral(side))
                    np.testing.assert_array_equal(self.states(adapter), before)
                    self.assertEqual(adapter.last_target_world.get(side), target)

    def test_elbow_solution_and_recovery_preserve_all_joint_states(self):
        for name, adapter in self.driver.ik.items():
            for side in ("left", "right"):
                with self.subTest(robot=name, side=side):
                    before = self.states(adapter)
                    points = adapter.arm_points(side)
                    world_wrist, world_elbow = points["wrist"], points["elbow"]
                    index = adapter.arm_ids[side][0]
                    adapter.p.resetJointState(adapter.body, index, before[adapter.position_of[index], 0],
                                             targetVelocity=0.17, physicsClientId=adapter.cid)
                    before = self.states(adapter)
                    try:
                        values = adapter.solve(side, world_wrist, source="world",
                                               elbow=world_elbow, elbow_pass="last")
                        self.assertEqual(len(values), 7)
                        np.testing.assert_array_equal(self.states(adapter), before)
                        target = adapter.last_target_world[side]

                        def fail_recovery(*args, **kwargs):
                            adapter._teleport(side, adapter.limits["rest"])
                            raise RuntimeError("elbow recovery failed")

                        with patch.object(adapter, "_recover_elbow_branch", side_effect=fail_recovery):
                            with self.assertRaisesRegex(RuntimeError, "elbow recovery failed"):
                                adapter.solve(side, world_wrist, source="world",
                                              elbow=world_elbow, elbow_pass="last")
                        np.testing.assert_array_equal(self.states(adapter), before)
                        self.assertEqual(adapter.last_target_world[side], target)
                    finally:
                        for joint, (angle, velocity) in zip(adapter.indices, before):
                            adapter.p.resetJointState(adapter.body, joint, angle,
                                                     targetVelocity=0., physicsClientId=adapter.cid)

    def test_current_and_live_read_current_angles_while_loaded_keeps_initial_reference(self):
        for name, adapter in self.driver.ik.items():
            with self.subTest(robot=name):
                index = adapter.arm_ids["right"][0]
                q, velocity = adapter.p.getJointState(adapter.body, index,
                                                      physicsClientId=adapter.cid)[:2]
                previous_mode = adapter.rest_poses
                try:
                    adapter.p.resetJointState(adapter.body, index, q + 0.08,
                                             physicsClientId=adapter.cid)
                    expected = self.states(adapter)[:, 0]
                    for mode in ("current", "live"):
                        adapter.rest_poses = mode
                        np.testing.assert_array_equal(adapter._rest_poses(), expected)
                    adapter.rest_poses = "loaded"
                    np.testing.assert_array_equal(adapter._rest_poses(), adapter.limits["rest"])
                    self.assertNotEqual(adapter._rest_poses()[adapter.position_of[index]],
                                        expected[adapter.position_of[index]])
                finally:
                    adapter.rest_poses = previous_mode
                    adapter.p.resetJointState(adapter.body, index, q, targetVelocity=velocity,
                                             physicsClientId=adapter.cid)

    def test_unreachable_wrist_orientation_does_not_sacrifice_position(self):
        adapter = self.driver.ik["h1_2"]
        for side in ("left", "right"):
            with self.subTest(side=side):
                before = self.states(adapter)
                # The synthetic camera palm frame is not a robot tool calibration.
                target = adapter.profile.realtime_neutral(side)
                values = adapter.solve(side, target, orientation=(0., 1., 0., 0.))
                np.testing.assert_array_equal(self.states(adapter), before)
                details = adapter.last_solve[side]
                if details["pose_candidate_residual_mm"] > adapter.profile.max_residual_mm:
                    self.assertTrue(details["position_fallback"])
                    self.assertIn(details['wrist_rejection'], ('pose_position_conflict', 'nonfinite_or_limit'))
                    for index, name in adapter.arm[side]:
                        self.assertEqual(values[name], before[adapter.position_of[index]][0])



@unittest.skipUnless(ASSETS_AVAILABLE and PYBULLET_AVAILABLE, "robot assets and PyBullet required")
class ArmRealtimeLoopTests(unittest.TestCase):
    def setUp(self):
        from teleoperation.simulation import realtime_teleop as app
        from tests.test_native_hand_replay import stub_camera_backend
        backend = patch.object(app, 'build_camera_retargeter', side_effect=stub_camera_backend)
        backend.start()
        self.addCleanup(backend.stop)

    def test_body_replay_commands_use_recorded_irregular_capture_times(self):
        from dataclasses import replace
        from types import SimpleNamespace
        from teleoperation.simulation import realtime_teleop as app
        from teleoperation.contracts.observations import CameraFrame
        from teleoperation.data.body_recording import capture_sample, save_body_recording
        from teleoperation.retargeting.hand.joint_commands import JointCommandPipeline
        from tests.smplerx_fixtures import observation, METADATA
        timeline = []
        class AuditedCommands(JointCommandPipeline):
            def update(self, result, timestamp=None):
                timeline.append(timestamp)
                return super().update(result, timestamp)
            def missing(self, key, timestamp=None):
                timeline.append(timestamp)
                return super().missing(key, timestamp)
        stamps = [10., 10.01, 10.04, 10.05, 10.12]
        rows = []
        for index, stamp in enumerate(stamps):
            raw, result = observation(index)
            raw = replace(raw, hands=replace(raw.hands, timestamp=stamp))
            source = SimpleNamespace(session_id='camera', last_result=result,
                last_frame=CameraFrame(np.zeros((4,5,3), dtype=np.uint8), stamp, index))
            rows.append(capture_sample(source, raw))
        with tempfile.TemporaryDirectory() as folder:
            path = save_body_recording(Path(folder)/'body.h5', rows, METADATA)
            args = build_parser().parse_args(['--input-body', str(path), '--no-arms',
                '--robot', 'g1', '--headless', '--no-preview', '--print-every', '0'])
            with patch.object(app, 'JointCommandPipeline', AuditedCommands), \
                    patch.object(app, '_drain_in_background', side_effect=lambda action, label: action()), \
                    contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(app.run(args), 0)
        np.testing.assert_allclose(sorted(set(timeline)), stamps)

    def test_replay_loop_routes_both_wrist_orientations_and_keeps_fixed_physics(self):
        from teleoperation.simulation import realtime_teleop as app
        from tests.test_arm_realtime_bridge import arm, open_hand

        with tempfile.TemporaryDirectory() as folder:
            capture = Path(folder) / "capture.npz"
            report = Path(folder) / "report.json"
            from tests.smplerx_fixtures import write
            write(capture, 18)
            args = build_parser().parse_args([
                "--robot", "h1_2", "--robot", "gr1_t2", "--robot", "g1",
                "--input-body", str(capture), "--report", str(report),
                "--backend", "dex", "--headless", "--no-preview",
                "--no-collision-guard", "--arm-calibrate-frames", "1", "--print-every", "0",
                "--arm-orientation", "auto"])
            with patch.object(app, "_drain_in_background", side_effect=lambda action, label: action()), \
                    contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(app.run(args), 0)
            result = json.loads(report.read_text(encoding="utf-8"))
        self.assertEqual(result["arm_failures"], {})
        self.assertEqual(result["simulation_timing"]["mode"], "fixed")
        self.assertEqual(result["simulation_timing"]["physics_steps"], 144)
        self.assertEqual(result["processed_frames"], 18)
        for side in ("left", "right"):
            self.assertEqual(result["arm"]["applied_frames"][side], 18)
            self.assertIsNotNone(result["arm"]["robots"]["h1_2"]["last_solve"][side][
                "pose_candidate_residual_mm"])
            self.assertIsNotNone(result["arm"]["robots"]["g1"]["last_solve"][side][
                "pose_candidate_residual_mm"])


if __name__ == "__main__":
    unittest.main()
