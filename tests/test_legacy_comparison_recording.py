"""Historical lossy mappings are diagnostics, never native hand commands."""
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

import h5py
import numpy as np

from teleoperation.data.hand_recording import LegacyComparisonWriter, iter_legacy_native_commands


class LegacyComparisonRecordingTests(unittest.TestCase):
    def test_lossy_mapping_is_saved_as_diagnostics_and_cannot_be_replayed_as_commands(self):
        key = ("g1", "right")
        model = SimpleNamespace(joint_names=("driver", "follower"), independent_names=("driver",),
                                config_hash="historical_asset", fingers=("index",))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/"comparison.h5"
            with LegacyComparisonWriter(path, {key:model}, {}) as writer:
                writer.append(0., {key:{"raw_angles":np.array([.1,.7]), "valid":True, "solve_ms":1., "error":""}},
                              {key:{"driver":.1,"follower":.7}}, {key:np.zeros((1,3))})
            with h5py.File(path) as stream:
                self.assertEqual(stream.attrs["schema"], "dex_native_comparison_v1")
                np.testing.assert_array_equal(stream["g1/right/raw_angles"][0], [.1,.7])
                np.testing.assert_array_equal(stream["g1/right/command_angles"][0], [.1,.7])
            with self.assertRaises(ValueError):
                list(iter_legacy_native_commands(path, {key:model}, specs={}))


if __name__ == "__main__":
    unittest.main()
