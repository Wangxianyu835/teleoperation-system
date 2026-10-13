"""Regression coverage for absent and per-side frozen Vision Pro tails."""
import contextlib
import io
from pathlib import Path
import tempfile
import unittest

import h5py
import numpy as np

from devtools import visionpro_trim


class VisionProTrimTests(unittest.TestCase):
    def check_trim(self, frozen_sides):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "raw.h5"
            output = Path(directory) / "trimmed.h5"
            originals = {}
            with h5py.File(source, "w") as handle:
                handle.attrs["source"] = "visionpro"
                handle.attrs["recording_status"] = "frames_reached"
                handle.create_dataset("timestamps", data=np.arange(6) / 30.0)
                for side in ("left", "right"):
                    transforms = np.tile(np.eye(4, dtype=np.float32), (6, 25, 1, 1))
                    motion = np.arange(6)
                    if side in frozen_sides:
                        motion = np.minimum(motion, 2)
                    transforms[:, :, 0, 3] = motion[:, None] / 100.0
                    originals[side] = transforms
                    dataset = handle.create_dataset(side + "_hand_transforms", data=transforms)
                    dataset.attrs["units"] = "m"
                    handle.create_dataset(side + "_valid", data=np.ones(6, dtype=bool))
            with contextlib.redirect_stdout(io.StringIO()):
                result = visionpro_trim.main([
                    "--input-h5", str(source), "--output", str(output), "--stale-limit", "2",
                ])
            self.assertEqual(result, 0)
            kept = 3 if frozen_sides else 6
            with h5py.File(output, "r") as handle:
                self.assertEqual(handle.attrs["source"], "visionpro")
                self.assertEqual(handle.attrs["recording_status"], "frames_reached+trimmed")
                self.assertEqual(handle.attrs["frame_count"], kept)
                self.assertEqual(handle.attrs["trimmed_frames"], 6 - kept)
                np.testing.assert_array_equal(handle["timestamps"][:], np.arange(kept) / 30.0)
                for side in ("left", "right"):
                    self.assertEqual(handle.attrs[side + "_freeze_at_frame"],
                                     3 if side in frozen_sides else -1)
                    dataset = handle[side + "_hand_transforms"]
                    self.assertEqual(dataset.attrs["units"], "m")
                    np.testing.assert_array_equal(dataset[:], originals[side][:kept])
            with h5py.File(source, "r") as handle:
                self.assertEqual(handle["timestamps"].shape, (6,))

    def test_no_frozen_tail_keeps_all_frames(self):
        self.check_trim(set())

    def test_one_frozen_side_trims_without_missing_attribute_error(self):
        for side in ("left", "right"):
            with self.subTest(side=side):
                self.check_trim({side})

    def test_both_frozen_sides_trim_the_tail(self):
        self.check_trim({"left", "right"})
