"""Exercise replay controls with real native URDFs and sentinel joint targets."""

import contextlib
import hashlib
import io
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock

import h5py
import numpy as np
import pybullet as p

from teleoperation.applications.replay import replay_hand_native as replay


class NativeHandReplayTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.path = Path(self.temp.name) / "sentinel.h5"
        self.angles = np.full((8, 18), 0.3) + np.arange(8)[:, None] * 0.03
        self.angles[:, 0] = 0
        self.write_fixture()

    def write_fixture(self, *, invalid=False, collapsed=False):
        left = self.angles.copy()
        if collapsed:
            left[2] = 0
        with h5py.File(self.path, "w") as f:
            f["frame_ids"] = np.arange(8)
            f["timestamps"] = np.arange(8) / 30
            for side, angles in (("left", left), ("right", self.angles + 0.1)):
                f[f"{side}_angles"] = angles
                valid = np.ones(8, dtype=bool)
                if invalid:
                    valid[4] = False
                f[f"{side}_valid"] = valid

    def run_replay(self, *options, interrupt_after=None):
        """Only GUI connection and wall clock are substituted; mapping stays real."""
        mapped, motors, cameras, clients = [], [], [], []
        clock = [0.0]
        steps = [0]
        real_connect, real_step = p.connect, p.stepSimulation
        real_map, real_motor = replay.NativeHandAdapter.map, p.setJointMotorControl2

        def connect(_mode):
            cid = real_connect(p.DIRECT)
            clients.append(cid)
            return cid

        def map_frame(adapter, dofs):
            result = real_map(adapter, dofs)
            mapped.append((np.r_[0.0, dofs.values].copy(), result[0].copy(), result[1]))
            return result

        def motor(body, index, mode, **kwargs):
            motors.append((index, kwargs["targetPosition"]))
            return real_motor(body, index, mode, **kwargs)

        def step(**kwargs):
            real_step(**kwargs)
            steps[0] += 1
            if interrupt_after and steps[0] >= interrupt_after:
                raise KeyboardInterrupt

        def sleep(duration):
            clock[0] += duration

        argv = ["replay_hand_native.py", "--file", str(self.path), "--substeps", "1", *options]
        try:
            with mock.patch.object(sys, "argv", argv), \
                    mock.patch.object(p, "connect", side_effect=connect), \
                    mock.patch.object(p, "isConnected", return_value=False), \
                    mock.patch.object(p, "stepSimulation", side_effect=step), \
                    mock.patch.object(p, "setJointMotorControl2", side_effect=motor), \
                    mock.patch.object(p, "resetDebugVisualizerCamera", side_effect=lambda **k: cameras.append(k)), \
                    mock.patch.object(replay.NativeHandAdapter, "map", autospec=True, side_effect=map_frame), \
                    mock.patch.object(replay.time, "monotonic", side_effect=lambda: clock[0]), \
                    mock.patch.object(replay.time, "sleep", side_effect=sleep), \
                    contextlib.redirect_stdout(io.StringIO()):
                cli_main(["replay", "native-hand", *argv[1:]])
        finally:
            for cid in clients:
                try:
                    p.disconnect(cid)
                except p.error:
                    pass  # Headless replay closes its own connection.
        return mapped, motors, cameras, clock[0], steps[0]

    def assert_same_targets(self, actual, expected):
        self.assertEqual(len(actual), len(expected))
        for got, want in zip(actual, expected):
            np.testing.assert_array_equal(got[0], want[0])
            self.assertEqual(got[1:], want[1:])

    def test_view_speed_range_loop_preserve_targets_and_validity_on_all_robots(self):
        self.write_fixture(invalid=True)
        digest = hashlib.sha256(self.path.read_bytes()).digest()
        for robot in ("h1_2", "gr1_t2", "g1"):
            with self.subTest(robot=robot):
                baseline = self.run_replay("--robot", robot, "--no-repair")[0]
                self.assertEqual(len(baseline), 14)  # Invalid original frame 4 is skipped.
                self.assertEqual(len(baseline[0][1]), {"h1_2": 12, "gr1_t2": 11, "g1": 7}[robot])
                expected = baseline[4:8]  # Original frames 2,3; end-frame 4 excluded.
                for view in ("full", "front", "side", "hands", "left-hand", "right-hand"):
                    with self.subTest(view=view):
                        mapped, motors, cameras, elapsed, steps = self.run_replay(
                            "--robot", robot, "--no-repair", "--render", "--view", view,
                            "--speed", "0.5", "--start-frame", "2", "--end-frame", "4", "--loop", "2")
                        self.assert_same_targets(mapped, expected * 2)
                        self.assertEqual(steps, 4)
                        self.assertAlmostEqual(elapsed, 2 * (1 / 30) / 0.5)
                        self.assertEqual(len(cameras), 1)
                        if view.endswith("-hand"):
                            self.assertLess(cameras[0]["cameraDistance"], 1)
                        # Confirm actual motor commands contain every mapped target.
                        for _angle, joints, _clipped in mapped:
                            for target in joints.values():
                                self.assertIn(target, [value for _index, value in motors])
        self.assertEqual(hashlib.sha256(self.path.read_bytes()).digest(), digest)

    def test_default_repair_uses_full_recording_context_at_clip_boundary(self):
        self.write_fixture(collapsed=True)
        full = self.run_replay()[0]
        clip = self.run_replay("--start-frame", "2", "--end-frame", "3")[0]
        self.assert_same_targets(clip, full[4:6])
        np.testing.assert_allclose(clip[0][0], self.angles[2])
        self.assertTrue(np.any(clip[0][0] != 0))

    def test_infinite_loop_repeats_identical_targets_until_interrupt(self):
        baseline = self.run_replay("--no-repair")[0]
        result = self.run_replay("--no-repair", "--loop", "0", interrupt_after=16)
        self.assert_same_targets(result[0], baseline * 2)
        self.assertEqual(result[4], 16)

    def test_invalid_controls_fail_before_connecting(self):
        for options in (
            ["--speed", "0"], ["--speed", "nan"], ["--speed", "inf"],
            ["--speed", "-1"], ["--loop", "-1"], ["--substeps", "-1"],
            ["--view", "unknown"], ["--start-frame", "-1"],
            ["--start-frame", "4", "--end-frame", "4"], ["--end-frame", "9"],
            ["--hand", "left", "--view", "right-hand"],
        ):
            with self.subTest(options=options), \
                    mock.patch.object(sys, "argv", ["replay", "--file", str(self.path), *options]), \
                    mock.patch.object(p, "connect") as connect, \
                    contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit) as error:
                    cli_main(["replay", "native-hand", *sys.argv[1:]])
                self.assertEqual(error.exception.code, 2)
                connect.assert_not_called()


if __name__ == "__main__":
    unittest.main()

from teleoperation.cli import main as cli_main
