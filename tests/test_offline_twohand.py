import unittest

import numpy as np

from retargeting.inference import _hold_last_valid_angles, _valid_angle_rows


class OfflineTwoHandTests(unittest.TestCase):
    def test_angle_validation_rejects_nonfinite_and_out_of_range_rows(self):
        limits = np.array([[0.0, 0.0], [-0.18, 0.18]], dtype=np.float32)
        angles = np.array(
            [[0.0, 0.1], [0.0, 0.3], [np.nan, 0.1]],
            dtype=np.float32,
        )

        np.testing.assert_array_equal(
            _valid_angle_rows(angles, limits),
            np.array([True, False, False]),
        )

    def test_invalid_angles_hold_previous_valid_pose(self):
        angles = np.array(
            [[0.0, 0.0], [0.2, 0.3], [0.0, 0.0], [0.4, 0.5], [0.0, 0.0]],
            dtype=np.float32,
        )
        valid = np.array([False, True, False, True, False])

        _hold_last_valid_angles(angles, valid)

        np.testing.assert_allclose(
            angles,
            [[0.0, 0.0], [0.2, 0.3], [0.2, 0.3], [0.4, 0.5], [0.4, 0.5]],
        )


if __name__ == "__main__":
    unittest.main()
