"""Shared rotation conversion regressions, independent of landmark input."""
import unittest
import numpy as np

from teleoperation.retargeting.arm.rotations import (
    matrix_from_quaternion, rotation_matrix_to_quaternion,
)


def rodrigues(axis, angle):
    """Rotation matrix about ``axis`` by ``angle`` (test-local, no scipy)."""
    vector = np.asarray(axis, dtype=np.float64)
    vector = vector / np.linalg.norm(vector)
    cross = np.array([[0.0, -vector[2], vector[1]],
                      [vector[2], 0.0, -vector[0]],
                      [-vector[1], vector[0], 0.0]])
    return np.eye(3) + np.sin(angle) * cross + (1.0 - np.cos(angle)) * (cross @ cross)


class QuaternionTests(unittest.TestCase):
    def test_round_trip_for_known_rotations(self):
        cases = (((1, 0, 0), 0.0), ((0, 0, 1), 0.7),
                 ((0.3, 0.5, -0.8), 2.4), ((1.0, 1.0, 1.0), 3.1))
        for axis, angle in cases:
            with self.subTest(axis=axis, angle=angle):
                matrix = rodrigues(axis, angle)
                quaternion = rotation_matrix_to_quaternion(matrix)
                self.assertAlmostEqual(float(np.linalg.norm(quaternion)), 1.0, places=12)
                np.testing.assert_allclose(matrix_from_quaternion(quaternion), matrix,
                                           atol=1e-9)


if __name__ == "__main__":
    unittest.main()
