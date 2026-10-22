"""Diagnostic timing, lifecycle and fixed-step numerical equivalence."""
import contextlib
import csv
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from teleoperation.simulation.arm_diagnostics import ArmDiagnostics, sample_seconds
from teleoperation.simulation import realtime_teleop as app
from tests.test_arm_ik_multi_robot import ASSETS_AVAILABLE, PYBULLET_AVAILABLE


def recording(path, frames=4):
    from tests.test_arm_realtime_bridge import arm, open_hand
    hand = open_hand()
    data = {"timestamps": np.arange(frames) / 30}
    for side in ("left", "right"):
        data[side + "_hand"] = np.stack([hand] * frames)
        data["world_" + side + "_hand"] = np.stack([(hand - hand[0]) * .19] * frames)
        data[side + "_arm"] = np.stack([arm(side, wrist_offset=(i * .001, 0, 0)) for i in range(frames)])
    if frames >= 3:
        data["left_arm"][2] = np.nan
    np.savez(path, **data)


def arguments(capture, report, *extra):
    return app.build_parser().parse_args(["--input-npz", str(capture), "--report", str(report),
        "--backend", "geometric", "--headless", "--no-preview", "--no-collision-guard",
        "--no-arm-fallback", "--arm-calibrate-frames", "1", "--print-every", "0", *extra])


class DiagnosticUnitTests(unittest.TestCase):
    def test_time_units_are_explicit_and_never_modify_input(self):
        raw = np.array([1000., 1100., 1200.])
        before = raw.copy()
        np.testing.assert_allclose(sample_seconds(raw, "milliseconds"), [0, .1, .2])
        np.testing.assert_allclose(sample_seconds(raw, "seconds"), [0, 100, 200])
        np.testing.assert_array_equal(raw, before)
        for values, unit in ((raw, None), ([0, 0], "seconds"), ([0, np.nan], "seconds"),
                             ([1, 0], "seconds"), ([[0, 1]], "seconds")):
            with self.subTest(values=values, unit=unit), self.assertRaises(ValueError):
                sample_seconds(values, unit)

    def test_stage_records_failures_and_uses_bounded_aggregates(self):
        ticks = iter([1., 1.002, 2., 2.004])
        diag = ArmDiagnostics(clock=lambda: next(ticks))
        with diag.stage("ik"):
            pass
        with self.assertRaises(RuntimeError):
            with diag.stage("ik"):
                raise RuntimeError("bad solver")
        item = diag.summary()["timings_ms"]["ik"]
        self.assertEqual(item["count"], 2)
        self.assertAlmostEqual(item["mean_ms"], 3.)
        self.assertAlmostEqual(item["max_ms"], 4.)
        self.assertNotIn("samples", diag.timings["ik"])

    def test_trace_cannot_overwrite_and_close_is_idempotent(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "trace.csv"
            diag = ArmDiagnostics(path)
            diag.close(); diag.close()
            self.assertTrue(diag.trace.file.closed)
            with self.assertRaises(FileExistsError):
                ArmDiagnostics(path)


@unittest.skipUnless(ASSETS_AVAILABLE and PYBULLET_AVAILABLE, "robot assets and PyBullet required")
class DiagnosticLoopTests(unittest.TestCase):
    def test_diagnostics_leave_targets_commands_states_and_step_counts_identical(self):
        from teleoperation.simulation.arm_scene_driver import ArmSceneDriver
        from teleoperation.simulation.arm_ik import PyBulletArmIK
        snapshots = []
        current = []
        ik_counts = [0, 0]
        run_id = [0]
        original_step, original_ik = ArmSceneDriver.step, PyBulletArmIK._ik_call

        def step(driver, *args, **kwargs):
            original_step(driver, *args, **kwargs)
            current.append({name: {side: {"q": list(adapter.joint_values(side).values()),
                "target": adapter.last_target_world[side]} for side in driver.sides}
                for name, adapter in driver.ik.items()})

        def ik(adapter, *args, **kwargs):
            ik_counts[run_id[0]] += 1
            return original_ik(adapter, *args, **kwargs)

        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            capture = root / "input.npz"
            trace = root / "trace.csv"
            recording(capture)
            reports = []
            for index in range(2):
                run_id[0] = index
                current = []
                report = root / f"run{index}.json"
                args = arguments(capture, report, *([] if not index else ["--diagnostic-trace", str(trace)]))
                with patch.object(app, "_drain_in_background", side_effect=lambda action, label: action()), \
                        patch.object(ArmSceneDriver, "step", step), patch.object(PyBulletArmIK, "_ik_call", ik), \
                        contextlib.redirect_stdout(io.StringIO()):
                    self.assertEqual(app.run(args), 0)
                snapshots.append(current)
                reports.append(json.loads(report.read_text(encoding="utf-8")))
            rows = list(csv.DictReader(trace.open(encoding="utf-8", newline="")))
        self.assertEqual(snapshots[0], snapshots[1])
        self.assertEqual(ik_counts[0], ik_counts[1])
        self.assertEqual(reports[0]["arm_last_targets"], reports[1]["arm_last_targets"])
        self.assertEqual(reports[0]["simulation_timing"], reports[1]["simulation_timing"])
        self.assertNotIn("diagnostics", reports[0])
        self.assertEqual(len(rows), 24)
        self.assertEqual({(row["robot"], row["side"]) for row in rows},
                         {(robot, side) for robot in ("g1", "h1_2", "gr1_t2") for side in ("left", "right")})
        lost = [row for row in rows if row["frame_id"] == "2" and row["side"] == "left"]
        self.assertTrue(all(row["input_valid"] == "False" and row["held"] == "True" for row in lost))
        self.assertTrue(all(row["sample_time_s"] == "" and row["timestamp_unit"] == "unspecified_legacy" for row in rows))
        self.assertIn("ik.h1_2.left", reports[1]["diagnostics"]["timings_ms"])
        self.assertIn("diagnostic_overhead", reports[1]["diagnostics"]["timings_ms"])

    def test_error_closes_trace_and_saves_partial_report(self):
        original = app.ReplayFrameSource.next_frame
        objects = []
        def make(*args, **kwargs):
            diag = ArmDiagnostics(*args, **kwargs)
            objects.append(diag)
            return diag
        def fail(source):
            if source.index == 1:
                raise RuntimeError("capture lost")
            return original(source)
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            capture, report, trace = root / "input.npz", root / "report.json", root / "trace.csv"
            recording(capture)
            with patch("teleoperation.simulation.arm_diagnostics.ArmDiagnostics", side_effect=make), \
                    patch.object(app.ReplayFrameSource, "next_frame", fail), \
                    patch.object(app, "_drain_in_background", side_effect=lambda action, label: action()), \
                    contextlib.redirect_stdout(io.StringIO()):
                with self.assertRaisesRegex(RuntimeError, "capture lost"):
                    app.run(arguments(capture, report, "--diagnostic-trace", str(trace)))
            self.assertTrue(objects[0].closed and objects[0].trace.file.closed)
            result = json.loads(report.read_text(encoding="utf-8"))
            self.assertEqual(result["exit_reason"], "error")
            self.assertEqual(result["diagnostics"]["frames"], 1)

    def test_output_collision_fails_before_building_scene(self):
        args = arguments("input.npz", "report.json", "--diagnostic-trace", "input.npz")
        with patch.object(app, "build_scene") as scene:
            with self.assertRaisesRegex(ValueError, "overwrite"):
                app.run(args)
            scene.assert_not_called()


if __name__ == "__main__":
    unittest.main()
