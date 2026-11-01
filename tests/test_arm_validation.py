"""Explicit input provenance, deterministic scenarios and offline acceptance."""
import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from teleoperation.data.arm_validation import load_validation_recording, save_validation_recording, validate_recording
from teleoperation.simulation.arm_scenarios import SCENARIOS, synthetic_recording
from teleoperation.simulation.arm_validation_metrics import analyze_rows, distribution, read_trace
from teleoperation.applications.arm_validation import main, run_case
from tests.test_arm_ik_multi_robot import ASSETS_AVAILABLE, PYBULLET_AVAILABLE


class ValidationInputTests(unittest.TestCase):
    def test_pose_fallback_rotates_the_elbow_plane_without_moving_the_wrist(self):
        from teleoperation.simulation.arm_realtime import wrist_orientation_frame, relative_wrist_rotation
        data = synthetic_recording(13, scenarios=(("neutral", 1.), ("pose_fallback", 1.)))
        for side in ("left", "right"):
            neutral, rotated = data[side + "_arm"][0], data[side + "_arm"][-1]
            np.testing.assert_array_equal(neutral[[0, 2]], rotated[[0, 2]])
            rotation = relative_wrist_rotation(rotated, wrist_orientation_frame(neutral))
            self.assertIsNotNone(rotation)
            self.assertAlmostEqual(float(np.trace(rotation)), -1., places=6)
        self.assertEqual(json.loads(str(data["metadata_json"]))["scenario_revision"], 2)

    def test_fixed_seed_preserves_scenarios_and_declares_synthetic_time(self):
        first, second = synthetic_recording(13), synthetic_recording(13)
        for key in first:
            np.testing.assert_array_equal(first[key], second[key])
        metadata = json.loads(str(first["metadata_json"]))
        self.assertEqual(metadata["source"], "synthetic")
        self.assertEqual(metadata["timestamp_unit"], "seconds")
        self.assertEqual(set(first["scenario"]), {name for name, _ in SCENARIOS})
        for interval in metadata["intervals"]:
            block = first["scenario"][interval["start_frame"]:interval["end_frame_exclusive"]]
            self.assertTrue(np.all(block == interval["name"]))
        missing = first["scenario"] == "drop_left"
        self.assertTrue(np.isnan(first["left_arm"][missing]).all())
        self.assertTrue(np.isfinite(first["right_arm"][missing]).all())

    def test_legacy_units_are_required_and_source_is_preserved(self):
        data = synthetic_recording(13, scenarios=(("neutral", 3.),))
        data.pop("metadata_json")
        data["timestamps"] = 1_700_000_000_000 + data["timestamps"] * 1000
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "old.npz"
            save_validation_recording(path, data)
            before = path.read_bytes()
            with self.assertRaisesRegex(ValueError, "Legacy"):
                load_validation_recording(path)
            with self.assertRaisesRegex(ValueError, 'Legacy'):
                load_validation_recording(path, 'milliseconds')
            self.assertEqual(path.read_bytes(), before)

    def test_malformed_inputs_and_conflicting_units_fail(self):
        original = synthetic_recording(13, scenarios=(("neutral", 1.),))
        for change in (lambda d: d.pop("left_arm"), lambda d: d.update(right_arm=np.zeros((2,3,3))),
                       lambda d: d.update(timestamps=np.zeros(len(d["timestamps"]))),
                       lambda d: d.update(scenario=np.zeros(len(d["timestamps"])))):
            data = dict(original)
            change(data)
            with self.assertRaises(ValueError):
                validate_recording(data)
        with self.assertRaisesRegex(ValueError, "conflicts"):
            validate_recording(original, "milliseconds")
        bad_units = dict(original, metadata_json=np.asarray(json.dumps({"timestamp_unit": "seconds", "length_unit": "cm"})))
        with self.assertRaisesRegex(ValueError, "world metres"):
            validate_recording(bad_units)
        with self.assertRaises(FileNotFoundError):
            load_validation_recording("camera.h5", "seconds")

    def test_percentiles_and_empty_observations_are_json_safe(self):
        stats = distribution([1, 2, 3, 4])
        self.assertEqual(stats["mean"], 2.5)
        self.assertAlmostEqual(stats["p95"], 3.85)
        self.assertIsNone(distribution([])["mean"])
        json.dumps(distribution([]), allow_nan=False)

    def test_invalid_import_fails_before_creating_output(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            data = synthetic_recording(13, scenarios=(("neutral", 1.),))
            data.pop("metadata_json")
            save_validation_recording(root / "old.npz", data)
            with contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(main(["--input-npz", str(root / "old.npz"), "--output-dir", str(root / "result")]), 1)
            self.assertFalse((root / "result").exists())

    def test_failed_acceptance_is_saved_and_returns_nonzero(self):
        case = {"checks": [{"name": "steady_error.h1_2.right.settle", "status": "fail"}],
                "passed": False, "frames": 3, "transitions": {}}
        with tempfile.TemporaryDirectory() as folder:
            destination = Path(folder) / "result"
            with patch("teleoperation.applications.arm_validation.run_case", return_value=case), \
                    contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(main(["--output-dir", str(destination)]), 1)
            report = json.loads((destination / "summary.json").read_text(encoding="utf-8"))
            self.assertFalse(report["passed"])
            self.assertEqual(len(report["cases"]), 2)
            self.assertIn("synthetic_13hz", report["recommendations"]["failed_checks"][-1])


@unittest.skipUnless(ASSETS_AVAILABLE and PYBULLET_AVAILABLE, "robot assets and PyBullet required")
class ValidationSimulationTests(unittest.TestCase):
    def test_recorded_posture_mode_checks_robot_reach_instead_of_chain_travel_box(self):
        from tests.smplerx_fixtures import write
        with tempfile.TemporaryDirectory() as folder, contextlib.redirect_stdout(io.StringIO()):
            root = Path(folder)
            path = write(root/'body.h5',40)
            result = run_case(load_validation_recording(path),root/'trace.csv',calibration_frames=3)
            rows = read_trace(root/'trace.csv')
        self.assertEqual(result['map_mode'],'posture')
        self.assertTrue(result['passed'],[c for c in result['checks'] if c['status']=='fail'])
        self.assertTrue(any(c['name'].startswith('posture_reach.') for c in result['checks']))
        self.assertFalse(any(c['name'].startswith('travel_clamp.') for c in result['checks']))
        # An unreachable target must still fail the replacement posture check.
        for row in rows:
            if row['valid']:
                row['target_base'] = [100.,100.,100.]
        from teleoperation.simulation.arm_profiles import profile_for
        rejected = analyze_rows(rows,{name:profile_for(name) for name in ('h1_2','gr1_t2','g1')},
                                map_mode='posture',robot_geometry=result['robot_geometry'])
        self.assertTrue(any(c['status']=='fail' and c['name'].startswith('posture_reach.') for c in rejected['checks']))

    def test_six_arm_case_records_hold_recovery_clamp_and_explicit_time(self):
        data = synthetic_recording(13, scenarios=(("neutral", .4), ("static", 1.2),
                    ("drop_left", .2), ("recover_left", .5), ("clamp", .5)))
        # A shortened fixture checks recorded metrics; the full CLI enforces all scenarios.
        data["metadata_json"] = np.asarray(json.dumps({"source": "test_fixture", "timestamp_unit": "seconds"}))
        with tempfile.TemporaryDirectory() as folder, contextlib.redirect_stdout(io.StringIO()):
            trace = Path(folder) / "trace.csv"
            result = run_case(validate_recording(data), trace, calibration_frames=4)
            rows = read_trace(trace)
        self.assertTrue(result["passed"], [item for item in result["checks"] if item["status"] == "fail"])
        self.assertEqual(len(rows), len(data["timestamps"]) * 6)
        self.assertLess(abs(result["simulation_timing"]["simulated_seconds"] - len(data["timestamps"])/13), 1/240 + 1e-8)
        for sides in result["transitions"].values():
            self.assertGreater(sides["left"]["held_target_delta_mm"]["count"], 0)
            self.assertEqual(sides["left"]["held_target_delta_mm"]["max"], 0.)
            self.assertGreater(sides["left"]["recovery_first_target_jump_mm"]["count"], 0)
        for row in rows:
            self.assertEqual(row["timestamp_unit"], "seconds")
            self.assertIsNotNone(row["sample_time_s"])
        json.dumps(result, allow_nan=False)
        from teleoperation.simulation.arm_profiles import profile_for
        altered = [dict(row, position_error_mm=100.) if row["robot"] == "h1_2" and
                   row["side"] == "right" and row["scenario"] == "static" else row for row in rows]
        rejected = analyze_rows(altered, {name: profile_for(name) for name in ("h1_2", "g1", "gr1_t2")})
        self.assertFalse(rejected["passed"])
        self.assertFalse(rejected["arm_overview"]["h1_2"]["right"]["spatial_checks_passed"])

    def test_failed_offline_solver_closes_scene_and_trace(self):
        from teleoperation.applications.three_robot_scene import ThreeRobotHandScene
        from teleoperation.simulation.arm_scene_driver import ArmSceneDriver
        original_close = ThreeRobotHandScene.close
        closed = []
        def close(scene):
            closed.append(scene.cid)
            original_close(scene)
        data = validate_recording(synthetic_recording(13, scenarios=(("neutral", .3),)))
        with tempfile.TemporaryDirectory() as folder:
            trace = Path(folder) / "trace.csv"
            with patch.object(ArmSceneDriver, "apply_each", side_effect=RuntimeError("IK failed")), \
                    patch.object(ThreeRobotHandScene, "close", close), contextlib.redirect_stdout(io.StringIO()):
                with self.assertRaisesRegex(RuntimeError, "IK failed"):
                    run_case(data, trace, calibration_frames=2)
            self.assertEqual(len(closed), 1)
            with trace.open("a", encoding="utf-8"):
                pass


if __name__ == "__main__":
    unittest.main()
