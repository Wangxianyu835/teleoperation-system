"""Imported capture tools keep the current data and CLI contracts."""
import contextlib
import io
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import h5py
import numpy as np

from devtools import run_retargeting_pipeline as pipeline
from teleoperation.applications.hand_align import align_h5


class CaptureBranchIntegrationTests(unittest.TestCase):
    def test_generated_world_recording_aligns_without_changing_source(self):
        from tests.test_world_hand_pipeline import world_frame, save_raw
        from teleoperation.contracts.coordinates import WORLD_PALM_LOCAL_COORDINATE_ALIGNMENT
        frames=[world_frame(i) for i in range(6)]
        with tempfile.TemporaryDirectory() as folder:
            raw,aligned=Path(folder)/'raw.h5',Path(folder)/'aligned.h5'
            save_raw(raw,frames)
            before=raw.read_bytes()
            result=align_h5(raw,aligned,'world')
            self.assertEqual(raw.read_bytes(),before)
            self.assertEqual(result['frames'],6)
            with h5py.File(aligned) as target, h5py.File(raw) as source:
                self.assertEqual(target.attrs['coordinate_alignment'],WORLD_PALM_LOCAL_COORDINATE_ALIGNMENT)
                for side in ('left','right'):
                    points=target[side+'_hand_keypoints'][:]
                    self.assertEqual(points.shape,(6,25,3))
                    self.assertTrue(np.isfinite(points).all())
                    self.assertTrue(np.any(points))
                np.testing.assert_array_equal(target['timestamps'][:],source['timestamps'][:])
                np.testing.assert_array_equal(target['frame_ids'][:],source['frame_ids'][:])

    def test_native_inspection_rejects_nonfinite_valid_predictions(self):
        from teleoperation.applications.hand_inspect import inspect_angle_h5
        from tests.test_hand_core_regression import write_native_recording
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'joints.h5'
            write_native_recording(path)
            with h5py.File(path, 'r+') as handle:
                handle['g1/right/raw_positions'][1, 0] = np.nan
            with self.assertRaises(ValueError):
                inspect_angle_h5(path)

    def test_explicit_missing_recording_is_not_replaced_with_synthetic_data(self):
        with tempfile.TemporaryDirectory() as folder:
            args = ["pipeline", "--raw", str(Path(folder) / "missing.h5"),
                    "--work-dir", str(Path(folder) / "work")]
            with patch("sys.argv", args), patch.object(pipeline.subprocess, "run") as run, \
                    contextlib.redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit) as error:
                    pipeline.main()
            self.assertEqual(error.exception.code, 2)
            run.assert_not_called()


if __name__ == "__main__":
    unittest.main()
