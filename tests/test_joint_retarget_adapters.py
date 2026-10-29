"""Legacy model numerics enter the one target-native joint result interface."""
import unittest

import numpy as np

from teleoperation.contracts.hand import HandWindow
from teleoperation.contracts.hand_control import HandJointResult
from teleoperation.retargeting.hand.interface import PoseTransformerRetargeter


class ArrayPredictor:
    def predict(self, payload, device):
        return {"hands": {"left": np.arange(18, dtype=np.float64), "right": None}}


class JointRetargetAdapterTests(unittest.TestCase):
    def test_legacy_nn_drops_only_placeholder_and_preserves_invalid_side(self):
        window = HandWindow({"left": np.zeros((3, 25, 3)), "right": None}, 1., "test")
        result = PoseTransformerRetargeter(ArrayPredictor()).retarget(window)
        self.assertIsInstance(result, dict)
        self.assertIsInstance(result[("l21", "left")], HandJointResult)
        np.testing.assert_array_equal(result[("l21", "left")].positions, np.arange(1., 18.))
        self.assertFalse(result[("l21", "right")].valid)
        self.assertIsNone(result[("l21", "right")].positions)
        self.assertEqual(result[("l21", "left")].timestamp, 1.)


if __name__ == "__main__":
    unittest.main()
