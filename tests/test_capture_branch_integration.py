"""Imported capture tools keep the current data and CLI contracts."""
import contextlib
import io
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import h5py
import numpy as np

from devtools import make_twohand_raw_sample as sample
from devtools import run_retargeting_pipeline as pipeline
from teleoperation.applications.hand_align import align_h5


class CaptureBranchIntegrationTests(unittest.TestCase):
    def test_generated_normalized_recording_aligns_without_changing_source(self):
        left, right, ids, timestamps = sample.make_twohand_raw(6, 13)
        with tempfile.TemporaryDirectory() as folder:
            raw, aligned = Path(folder) / "raw.h5", Path(folder) / "aligned.h5"
            sample.write_raw_h5(raw, left, right, ids, timestamps, 13)
            before = raw.read_bytes()
            result = align_h5(raw, aligned)
            self.assertEqual(raw.read_bytes(), before)
            self.assertEqual(result["frames"], 6)
            with h5py.File(aligned) as handle:
                self.assertEqual(handle.attrs["source_landmark_space"], "mediapipe_normalized")
                for side in ("left", "right"):
                    points = handle[side + "_hand_keypoints"][:]
                    self.assertEqual(points.shape, (6, 25, 3))
                    self.assertTrue(np.isfinite(points).all())
                    self.assertTrue(np.any(points))
                np.testing.assert_array_equal(handle["timestamps"][:], timestamps)

    def test_sample_rejects_invalid_timing_before_generation(self):
        for frames, fps in ((2, 30), (6, 0), (6, -1), (6, np.nan), (6, np.inf)):
            with self.subTest(frames=frames, fps=fps), self.assertRaises(ValueError):
                sample.make_twohand_raw(frames, fps)

    def test_pipeline_validation_rejects_nonfinite_predictions(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "angles.h5"
            with h5py.File(path, "w") as handle:
                handle["frame_ids"] = np.arange(4)
                for side in ("left", "right"):
                    values = np.zeros((4, 18))
                    values[2, 4] = np.nan
                    handle[side + "_angles"] = values
                    handle[side + "_valid"] = np.ones(4, dtype=bool)
            with contextlib.redirect_stdout(io.StringIO()):
                self.assertFalse(pipeline.verify_angles(path))

    def test_explicit_missing_recording_is_not_replaced_with_synthetic_data(self):
        with tempfile.TemporaryDirectory() as folder:
            args = ["pipeline", "--raw", str(Path(folder) / "missing.h5"),
                    "--work-dir", str(Path(folder) / "work")]
            with patch("sys.argv", args), patch.object(pipeline, "run_step") as run, \
                    contextlib.redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit) as error:
                    pipeline.main()
            self.assertEqual(error.exception.code, 2)
            run.assert_not_called()


if __name__ == "__main__":
    unittest.main()
