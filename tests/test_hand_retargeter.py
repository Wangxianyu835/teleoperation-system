import unittest

import numpy as np
import torch
import torch.nn as nn

from retargeting.hand_core import HandWindow
from retargeting.model import TwoHandRetargeter
from retargeting.retargeter import HandCommand, HandRetargeter, PoseTransformerRetargeter


class DummyRetargetModel(nn.Module):
    def forward(self, hand_window):
        value = hand_window[:, -1, 0, 0].unsqueeze(1)
        return value.expand(hand_window.shape[0], 18)


def _window(left=None, right=None, timestamp=1.25):
    return HandWindow(
        hands={"left": left, "right": right},
        timestamp=timestamp,
        source="test",
    )


class PoseTransformerRetargeterTests(unittest.TestCase):
    def test_adapter_implements_interface_and_returns_hand_command(self):
        adapter = PoseTransformerRetargeter(TwoHandRetargeter(DummyRetargetModel()))
        left = np.ones((3, 25, 3), dtype=np.float32)
        right = np.full((3, 25, 3), 2.0, dtype=np.float32)

        self.assertIsInstance(adapter, HandRetargeter)
        command = adapter.retarget(_window(left, right))

        self.assertIsInstance(command, HandCommand)
        self.assertTrue(command.left_valid)
        self.assertTrue(command.right_valid)
        self.assertEqual(command.left_angles.shape, (18,))
        self.assertEqual(command.right_angles.shape, (18,))
        self.assertEqual(command.timestamp, 1.25)

    def test_validity_follows_missing_window_sides(self):
        adapter = PoseTransformerRetargeter(TwoHandRetargeter(DummyRetargetModel()))
        left = np.ones((3, 25, 3), dtype=np.float32)

        command = adapter.retarget(_window(left, None))

        self.assertTrue(command.left_valid)
        self.assertFalse(command.right_valid)
        self.assertIsNone(command.right_angles)

    def test_adapter_matches_existing_two_hand_retargeter(self):
        model = DummyRetargetModel()
        old = TwoHandRetargeter(model)
        adapter = PoseTransformerRetargeter(old)
        left = np.arange(3 * 25 * 3, dtype=np.float32).reshape(3, 25, 3)
        right = left + 100.0
        window = _window(left, right)

        expected = old.predict(window.to_payload(), device="cpu")["hands"]
        actual = adapter.retarget(window)

        np.testing.assert_array_equal(actual.left_angles, expected["left"])
        np.testing.assert_array_equal(actual.right_angles, expected["right"])

    def test_reset_preserves_stateless_model_output(self):
        adapter = PoseTransformerRetargeter(TwoHandRetargeter(DummyRetargetModel()))
        left = np.ones((3, 25, 3), dtype=np.float32)
        window = _window(left, None)

        before = adapter.retarget(window)
        adapter.reset()
        after = adapter.retarget(window)

        np.testing.assert_array_equal(before.left_angles, after.left_angles)
        self.assertEqual(before.left_valid, after.left_valid)
        self.assertEqual(before.right_valid, after.right_valid)


if __name__ == "__main__":
    unittest.main()
