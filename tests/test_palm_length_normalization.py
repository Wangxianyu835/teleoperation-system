"""One causal dimensionless preprocessing contract for training and control."""
from dataclasses import replace
import unittest

import numpy as np

from teleoperation.contracts.coordinates import WORLD_PALM_LOCAL_METADATA, validate_landmark_contract
from teleoperation.contracts.hand import HandWindow
from teleoperation.retargeting.hand.normalization import normalize_hand_window, PALM_LENGTH_PROFILE, PALM_LENGTH_ALIGNMENT


def metric_window(scales=(1., 1., 1.)):
    points = np.zeros((25, 3), dtype=np.float32)
    points[6] = [.02, .03, .01]
    points[11] = [0., 0., .08]
    points[4] = [.04, .02, .06]
    points[9] = [.03, .01, .15]
    values = np.stack([points * scale for scale in scales])
    return HandWindow({"left": values.copy(), "right": values.copy()}, 1., "h5", dict(WORLD_PALM_LOCAL_METADATA))


class PalmLengthNormalizationTests(unittest.TestCase):
    def test_uniform_scale_invariance_and_input_is_not_mutated(self):
        original = metric_window()
        before = original.hands["left"].copy()
        for scale in (.7, 1., 1.3):
            window = metric_window((scale,) * 3)
            output = normalize_hand_window(window)
            np.testing.assert_allclose(output.hands["left"][-1, 11], [0., 0., 1.], atol=1e-6)
            np.testing.assert_allclose(output.hands["left"][-1, 4], [.5, .25, .75], atol=1e-6)
            self.assertEqual(output.units, "1")
            validate_landmark_contract(PALM_LENGTH_ALIGNMENT, output.metadata, "test")
        np.testing.assert_array_equal(original.hands["left"], before)
        self.assertEqual(original.metadata["units"], "m")

    def test_three_frames_share_the_median_scale_and_are_wrist_relative(self):
        window = metric_window((1., 2., 1.))
        window.hands["left"][:] += [1., -2., 3.]
        output = normalize_hand_window(window)
        np.testing.assert_allclose(output.hands["left"][:, 11, 2], [1., 2., 1.], atol=1e-5)
        np.testing.assert_array_equal(output.hands["left"][:, 0], np.zeros((3, 3)))
        self.assertAlmostEqual(output.metadata["palm_length_m"]["left"], .08, places=6)
        self.assertEqual(output.metadata["input_profile"], PALM_LENGTH_PROFILE)

    def test_invalid_side_is_not_replaced_with_the_other_side_or_epsilon(self):
        for value in (0., 1e-6, np.nan, np.inf):
            window = metric_window()
            window.hands["left"][1, 11] = [0., 0., value]
            output = normalize_hand_window(window)
            self.assertIsNone(output.hands["left"])
            self.assertIsNotNone(output.hands["right"])
            self.assertIsNone(output.metadata["palm_length_m"]["left"])
        missing = replace(metric_window(), hands={"left": None, "right": None})
        self.assertEqual(normalize_hand_window(missing).hands, {"left": None, "right": None})

    def test_repeated_or_wrong_source_preprocessing_is_rejected(self):
        output = normalize_hand_window(metric_window())
        with self.assertRaises(ValueError):
            normalize_hand_window(output)
        for metadata in ({}, {**WORLD_PALM_LOCAL_METADATA, "units": "mm"}):
            with self.assertRaises(ValueError):
                normalize_hand_window(replace(metric_window(), metadata=metadata))

    def test_wrong_window_size_or_degenerate_palm_is_invalid(self):
        window = metric_window()
        window.hands["left"][:] = 0.
        self.assertIsNone(normalize_hand_window(window).hands["left"])
        short = replace(window, hands={"left": np.zeros((2, 25, 3)), "right": None})
        with self.assertRaises(ValueError):
            normalize_hand_window(short)


if __name__ == "__main__":
    unittest.main()
