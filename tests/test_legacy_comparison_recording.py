"""Retired lossy and native schemas cannot enter joint command replay."""
from pathlib import Path
import tempfile
import unittest

import h5py

from teleoperation.data.hand_joint_recording import iter_joint_commands


class LegacyComparisonRecordingTests(unittest.TestCase):
    def test_retired_schemas_are_rejected_before_reading_commands(self):
        with tempfile.TemporaryDirectory() as directory:
            for schema in ("dex_native_comparison_v1", "dex_native_results_v1", "transhandr_vr_results_v1"):
                with self.subTest(schema=schema):
                    path = Path(directory) / (schema + ".h5")
                    with h5py.File(path, "w") as stream:
                        stream.attrs.update(schema=schema, timestamp_unit="seconds", angle_unit="rad")
                    with self.assertRaisesRegex(ValueError, "Not a hand joint command recording"):
                        list(iter_joint_commands(path, {}))


if __name__ == "__main__":
    unittest.main()
