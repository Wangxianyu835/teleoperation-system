import hashlib
import tempfile
import unittest
from pathlib import Path

import h5py
import numpy as np

from scripts.diagnose_hand_coordinates import diagnose_h5
from tests.test_coordinate_contracts import synthetic_hand_pair


class CoordinateDiagnosticTests(unittest.TestCase):
    def _write(self, path, *, grouped=False, declared=False):
        with h5py.File(path, "w") as handle:
            if declared:
                handle.attrs["coordinate_frame"] = "l21"
                handle.attrs["coordinate_alignment"] = "source_to_l21_xyz"
                handle.attrs["units"] = "synthetic_unit"
                handle.attrs["source_coordinate_convention"] = "synthetic_xyz"
            container = handle.create_group("recording") if grouped else handle
            for side, points in zip(("left", "right"), synthetic_hand_pair()):
                name = f"{side}_hand_keypoints" if not grouped else ("l_glove_pos" if side == "left" else "r_glove_pos")
                data = np.repeat(points[None], 3, axis=0)
                data[1] = 0
                data[2, 3, 0] = np.nan
                container.create_dataset(name, data=data)

    def test_undeclared_units_remain_unverified_and_file_is_unchanged(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "raw.h5"
            self._write(path)
            before = hashlib.sha256(path.read_bytes()).hexdigest()
            report = diagnose_h5(path)
            self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(), before)
        self.assertEqual(report["declared_units"], "UNDECLARED")
        self.assertEqual(report["physical_scale_status"], "PHYSICAL SCALE NOT VERIFIED")
        self.assertEqual(report["physical_coordinate_status"], "MANUAL VERIFICATION REQUIRED")
        self.assertTrue(report["transform"]["proper_rotation"])
        self.assertFalse(report["aligned_metadata_consistent"])
        for side in ("left", "right"):
            self.assertEqual(report["hands"][side]["finite_nonzero_frames"], 1)
            self.assertEqual(report["hands"][side]["nonfinite_frames"], 1)
            self.assertEqual(report["hands"][side]["zero_frames"], 1)
            self.assertEqual(report["fk"][side]["declared_length_units"], "UNDECLARED")

    def test_grouped_schema_reports_declarations_without_claiming_calibration(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "grouped.h5"
            self._write(path, grouped=True, declared=True)
            report = diagnose_h5(path, sample_count=1)
        self.assertEqual(report["declared_units"], "synthetic_unit")
        self.assertEqual(report["source_coordinate_convention"], "synthetic_xyz")
        self.assertTrue(report["aligned_metadata_consistent"])
        self.assertEqual(report["physical_scale_status"], "PHYSICAL SCALE NOT VERIFIED")
        left = report["hands"]["left"]["samples"][0]["signed_volume"]
        right = report["hands"]["right"]["samples"][0]["signed_volume"]
        self.assertLess(left * right, 0)

    def test_missing_alignment_metadata_is_reported(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "partial.h5"
            self._write(path)
            with h5py.File(path, "a") as handle:
                handle.attrs["coordinate_frame"] = "l21"
            report = diagnose_h5(path)
        self.assertFalse(report["aligned_metadata_consistent"])
        self.assertEqual(report["coordinate_alignment"], "UNDECLARED")

    def test_invalid_sample_count_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "positive"):
            diagnose_h5("not_opened.h5", sample_count=0)


if __name__ == "__main__":
    unittest.main()
