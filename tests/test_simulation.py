import tempfile
import unittest
from pathlib import Path

import h5py
import numpy as np

from retargeting.simulation import angle18_to_dofs, angle18_to_nodes, iter_angle_h5


class SimulationAdapterTests(unittest.TestCase):
    def test_angle_adapters_have_expected_order_and_shape(self):
        angle = np.arange(18, dtype=np.float32)
        np.testing.assert_array_equal(angle18_to_dofs(angle), angle[1:])
        np.testing.assert_array_equal(angle18_to_nodes(angle)[:18], angle)
        np.testing.assert_array_equal(angle18_to_nodes(angle)[18:], np.zeros(5))

    def test_invalid_frames_hold_previous_pose(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "angles.h5"
            angles = np.stack(
                [np.zeros(18), np.ones(18), np.full(18, 9)], dtype=np.float32
            )
            with h5py.File(path, "w") as h5_file:
                h5_file.create_dataset("frame_ids", data=np.arange(3))
                h5_file.create_dataset("timestamps", data=np.arange(3) / 30.0)
                for side in ("left", "right"):
                    h5_file.create_dataset(f"{side}_angles", data=angles)
                    h5_file.create_dataset(
                        f"{side}_valid", data=np.array([False, True, False])
                    )

            frames = list(iter_angle_h5(path))

        np.testing.assert_array_equal(frames[0].left, np.zeros(18))
        np.testing.assert_array_equal(frames[1].left, np.ones(18))
        np.testing.assert_array_equal(frames[2].left, np.ones(18))


if __name__ == "__main__":
    unittest.main()
