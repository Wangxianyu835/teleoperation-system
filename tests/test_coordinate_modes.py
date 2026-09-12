import unittest

import numpy as np
import torch

from retargeting.coordinates import (
    transform_fk_positions,
    transform_hand_coordinates,
)


class CoordinateModeTests(unittest.TestCase):
    def test_none_preserves_numpy_coordinates(self):
        points = np.arange(18, dtype=np.float32).reshape(2, 3, 3)
        transformed = transform_hand_coordinates(points, "left", "none")
        np.testing.assert_array_equal(transformed, points)
        self.assertIsNot(transformed, points)

    def test_mirror_x_only_changes_left_x(self):
        points = np.asarray(
            [[[1.0, 2.0, 3.0], [-4.0, 5.0, -6.0]]],
            dtype=np.float32,
        )
        transformed = transform_hand_coordinates(points, "left", "mirror_x")
        np.testing.assert_array_equal(
            transformed,
            [[[-1.0, 2.0, 3.0], [4.0, 5.0, -6.0]]],
        )

    def test_right_hand_is_not_mirrored(self):
        points = np.asarray([[[1.0, 2.0, 3.0]]], dtype=np.float32)
        transformed = transform_hand_coordinates(points, "right", "mirror_x")
        np.testing.assert_array_equal(transformed, points)

    def test_fk_transform_preserves_gradients_and_only_mirrors_x(self):
        positions = torch.tensor(
            [[[1.0, 2.0, 3.0], [-4.0, 5.0, -6.0]]],
            requires_grad=True,
        )
        transformed = transform_fk_positions(
            positions,
            side="left",
            mode="mirror_x",
        )
        torch.testing.assert_close(
            transformed,
            torch.tensor([[[-1.0, 2.0, 3.0], [4.0, 5.0, -6.0]]]),
        )
        transformed.sum().backward()
        torch.testing.assert_close(
            positions.grad,
            torch.tensor([[[-1.0, 1.0, 1.0], [-1.0, 1.0, 1.0]]]),
        )


if __name__ == "__main__":
    unittest.main()
