import unittest

import numpy as np
from retargeting.coordinates import (
    COORDINATE_FRAME,
    SOURCE_TO_L21_MATRIX,
    align_source_hand_coordinates,
)


class CoordinateAlignmentTests(unittest.TestCase):
    def test_fixed_alignment_applies_to_both_hands_identically(self):
        points = np.asarray(
            [[[1.0, 2.0, 3.0], [-4.0, 5.0, -6.0]]],
            dtype=np.float32,
        )
        np.testing.assert_array_equal(
            align_source_hand_coordinates(points),
            [[[-2.0, 3.0, -1.0], [-5.0, -6.0, 4.0]]],
        )

    def test_alignment_preserves_shape_and_input(self):
        points = np.arange(2 * 3 * 4 * 3, dtype=np.float32).reshape(2, 3, 4, 3)
        original = points.copy()
        aligned = align_source_hand_coordinates(points)
        self.assertEqual(aligned.shape, points.shape)
        np.testing.assert_array_equal(points, original)
        self.assertTrue(np.isfinite(aligned).all())

    def test_alignment_matrix_is_a_proper_rotation(self):
        matrix = SOURCE_TO_L21_MATRIX
        np.testing.assert_allclose(matrix @ matrix.T, np.eye(3), atol=1e-6)
        self.assertAlmostEqual(float(np.linalg.det(matrix)), 1.0, places=6)
        self.assertEqual(COORDINATE_FRAME, "l21")

    def test_invalid_inputs_are_rejected(self):
        with self.assertRaises(ValueError):
            align_source_hand_coordinates(np.zeros((3, 2), dtype=np.float32))
        with self.assertRaises(ValueError):
            align_source_hand_coordinates(
                np.asarray([[np.nan, 0.0, 0.0]], dtype=np.float32)
            )


if __name__ == "__main__":
    unittest.main()
