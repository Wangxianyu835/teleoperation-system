import unittest

import numpy as np

from teleoperation.contracts.validation import (
    HAND_KEYPOINTS,
    RECEPTIVE_FIELD,
    legacy_visionpro_to_window,
    validate_arm_input,
    validate_retarget_input,
)
from teleoperation.retargeting.hand.tracking import HandIdentityTracker
from tests.support import CanonicalWindowFixture
from teleoperation.retargeting.hand.topology import mediapipe21_to_hand25


class InputAdapterTests(unittest.TestCase):
    def test_mediapipe_mapping_has_canonical_shape_and_wrist_origin(self):
        points = np.arange(21 * 3, dtype=np.float32).reshape(21, 3)
        converted = mediapipe21_to_hand25(points)

        self.assertEqual(converted.shape, (HAND_KEYPOINTS, 3))
        np.testing.assert_array_equal(converted[0], np.zeros(3, dtype=np.float32))
        np.testing.assert_allclose(converted[6], points[5] - points[0])
        np.testing.assert_allclose(
            converted[5],
            0.5 * (points[5] - points[0]),
        )

    def test_window_buffer_keeps_missing_hand_explicit(self):
        buffer = CanonicalWindowFixture()
        points = np.zeros((21, 3), dtype=np.float32)

        self.assertIsNone(buffer.update(left_hand=points, source="test"))
        self.assertIsNone(buffer.update(left_hand=points, source="test"))
        payload = buffer.update(left_hand=points, source="test")

        self.assertIsNotNone(payload)
        validate_retarget_input(payload)
        self.assertEqual(payload["hands"]["left"].shape, (3, 25, 3))
        self.assertIsNone(payload["hands"]["right"])

    def test_legacy_visionpro_shape_is_normalized(self):
        legacy = np.zeros((25, RECEPTIVE_FIELD, 3), dtype=np.float32)
        normalized = legacy_visionpro_to_window(legacy)
        self.assertEqual(normalized.shape, (RECEPTIVE_FIELD, 25, 3))

    def test_arm_contract_accepts_optional_frame_and_rejects_bad_shape(self):
        self.assertTrue(validate_arm_input(None, "left"))

        with self.assertRaisesRegex(ValueError, "left_arm shape"):
            validate_arm_input(np.zeros((2, 3), dtype=np.float32), "left")

    def test_realtime_contract_can_require_arms_container(self):
        payload = {
            "hands": {
                "left": np.zeros((RECEPTIVE_FIELD, HAND_KEYPOINTS, 3), dtype=np.float32),
                "right": None,
            },
        }

        validate_retarget_input(payload)
        with self.assertRaisesRegex(KeyError, "arms"):
            validate_retarget_input(payload, require_arms=True)

        payload["arms"] = {"left": None, "right": None}
        self.assertTrue(validate_retarget_input(payload, require_arms=True))

    def test_identity_tracker_corrects_single_hand_label_switch(self):
        tracker = HandIdentityTracker()
        shape = np.linspace(0.0, 0.2, 21 * 3, dtype=np.float32).reshape(21, 3)
        left = shape + [0.70, 0.0, 0.0]
        right = shape + [0.30, 0.0, 0.0]
        tracker.update(left_hand=left, right_hand=right)

        switched = tracker.update(right_hand=left + [0.01, 0.0, 0.0])

        self.assertIsNotNone(switched["left"])
        self.assertIsNone(switched["right"])

    def test_identity_tracker_rejects_unmatched_jump(self):
        tracker = HandIdentityTracker()
        shape = np.linspace(0.0, 0.2, 21 * 3, dtype=np.float32).reshape(21, 3)
        tracker.update(right_hand=shape + [0.30, 0.0, 0.0])

        result = tracker.update(right_hand=shape + [0.60, 0.0, 0.0])

        self.assertIsNone(result["left"])
        self.assertIsNone(result["right"])

if __name__ == "__main__":
    unittest.main()
