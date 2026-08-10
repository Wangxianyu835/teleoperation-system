import tempfile
import unittest
from pathlib import Path

import numpy as np

from config.retarget_io import (
    HAND_KEYPOINTS,
    RECEPTIVE_FIELD,
    legacy_visionpro_to_window,
    validate_retarget_input,
)
from input_adapters.hand_keypoints import (
    HandWindowBuffer,
    mediapipe21_to_hand25,
)
from input_adapters.npy_replay_adapter import NpyReplayAdapter


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
        buffer = HandWindowBuffer()
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

    def test_npy_replay_emits_three_frame_window(self):
        points = np.zeros((21, 3), dtype=np.float32)
        frames = np.array(
            [
                {
                    "frame_id": index,
                    "timestamp": float(index),
                    "left_hand": points + index,
                    "right_hand": None,
                }
                for index in range(3)
            ],
            dtype=object,
        )

        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "hands.npy"
            np.save(path, frames)
            payload = NpyReplayAdapter(path).next_input()

        self.assertIsNotNone(payload)
        self.assertEqual(payload["source"], "mediapipe_approx")
        self.assertEqual(payload["hands"]["left"].shape, (3, 25, 3))


if __name__ == "__main__":
    unittest.main()
