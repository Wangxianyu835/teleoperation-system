"""Compatibility wrapper must preserve canonical inspection and error output."""

from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import h5py
import numpy as np


ROOT = Path(__file__).resolve().parents[1]


class InspectWrapperTests(unittest.TestCase):
    def commands(self, *args):
        return [subprocess.run(
            [sys.executable, "-B", *entry, *args], cwd=ROOT,
            capture_output=True, text=True, encoding="utf-8", timeout=30,
        ) for entry in (["inspect_angle_h5.py"], ["-m", "retargeting", "inspect"])]

    def test_help_is_canonical(self):
        wrapper, canonical = self.commands("--help")
        self.assertEqual(wrapper.returncode, 0, wrapper.stderr)
        self.assertEqual(canonical.returncode, 0, canonical.stderr)
        self.assertEqual(wrapper.stdout, canonical.stdout)
        self.assertIn("--angle-h5", wrapper.stdout)

    def test_valid_file_has_identical_summary_and_validity(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "angles.h5"
            with h5py.File(path, "w") as f:
                f["frame_ids"] = np.arange(3)
                f["timestamps"] = np.arange(3) / 30
                for side in ("left", "right"):
                    f[f"{side}_angles"] = np.zeros((3, 18), dtype=np.float32)
                    f[f"{side}_valid"] = [True, False, True]
                f.attrs["alignment"] = "palm_local_to_l21_v1"
            wrapper, canonical = self.commands("--angle-h5", str(path))
        self.assertEqual(wrapper.returncode, 0, wrapper.stderr)
        self.assertEqual(canonical.returncode, 0, canonical.stderr)
        self.assertEqual(wrapper.stdout, canonical.stdout)
        self.assertIn("frames=3", wrapper.stdout)
        self.assertIn("left_valid=2", wrapper.stdout)
        self.assertIn("right_invalid=1", wrapper.stdout)
        self.assertIn("attr.alignment=palm_local_to_l21_v1", wrapper.stdout)

    def assert_same_failure(self, path, error, message):
        wrapper, canonical = self.commands("--angle-h5", str(path))
        for result in (wrapper, canonical):
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("Traceback", result.stderr)  # Exception must not be swallowed.
            self.assertIn(error, result.stderr)
            self.assertIn(message, result.stderr)
        self.assertEqual(wrapper.returncode, canonical.returncode)
        self.assertEqual(wrapper.stderr.splitlines()[-1], canonical.stderr.splitlines()[-1])

    def test_nonexistent_file_preserves_canonical_error(self):
        with tempfile.TemporaryDirectory() as temp:
            self.assert_same_failure(Path(temp) / "missing.h5", "FileNotFoundError", "Angle H5 was not found")

    def test_invalid_schema_preserves_canonical_error(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "invalid.h5"
            with h5py.File(path, "w") as f:
                f["frame_ids"] = [0]
            self.assert_same_failure(path, "ValueError", "Angle H5 is missing datasets")


if __name__ == "__main__":
    unittest.main()
