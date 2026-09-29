import unittest

import numpy as np
import torch
import torch.nn as nn

from retargeting.model import TwoHandRetargeter


class DummyRetargetModel(nn.Module):
    def forward(self, hand_window):
        value = hand_window[:, -1, 0, 0].unsqueeze(1)
        return value.expand(hand_window.shape[0], 18)


class TwoHandRetargeterTests(unittest.TestCase):
    def test_shared_model_can_predict_both_sides(self):
        retargeter = TwoHandRetargeter(DummyRetargetModel())
        left = np.ones((3, 25, 3), dtype=np.float32)
        right = np.full((3, 25, 3), 2.0, dtype=np.float32)

        output = retargeter.predict(
            {
                "hands": {"left": left, "right": right},
            },
            device="cpu",
        )

        self.assertEqual(output["hands"]["left"].shape, (18,))
        self.assertEqual(output["hands"]["right"].shape, (18,))
        np.testing.assert_array_equal(output["hands"]["left"], np.ones(18))
        np.testing.assert_array_equal(output["hands"]["right"], np.full(18, 2.0))

    def test_shared_model_can_predict_when_one_side_is_missing(self):
        retargeter = TwoHandRetargeter(DummyRetargetModel())
        left = np.ones((3, 25, 3), dtype=np.float32)

        output = retargeter.predict(
            {
                "hands": {"left": left, "right": None},
            },
            device="cpu",
        )

        self.assertIsNotNone(output["hands"]["left"])
        self.assertIsNone(output["hands"]["right"])


if __name__ == "__main__":
    unittest.main()
