import tempfile
import unittest
from pathlib import Path

import h5py
import numpy as np
import torch

from teleoperation.retargeting.hand.transformer import PoseTransformer
from teleoperation.learning.dataset import TwoHandH5Dataset
from teleoperation.retargeting.hand.exporting import _hold_last_valid_angles
from teleoperation.applications.hand_processing import CanonicalHandPipeline
from teleoperation.retargeting.hand.angles import angle18_to_dofs
from teleoperation.applications.replay.canonical import iter_angle_h5
from teleoperation.contracts.constants import HAND_KEYPOINTS
from teleoperation.retargeting.hand.tracking import HandIdentityTracker
from tests.support import CanonicalWindowFixture
from teleoperation.retargeting.hand.topology import mediapipe21_to_hand25, wrist_relative
from tests.fixtures.hands import (
    continuous_three_frames,
    fixed_mediapipe_hand,
    identity_jump,
    left_hand,
    missing_left,
    missing_right,
    nan_frame,
    right_hand,
    zero_frame,
)


class SyntheticFixtureTests(unittest.TestCase):
    def test_fixture_set_contains_both_sides_invalid_frames_and_missing_sides(self):
        self.assertEqual(left_hand().shape, (21, 3))
        self.assertEqual(right_hand().shape, (21, 3))
        self.assertEqual(continuous_three_frames().shape, (3, 21, 3))
        self.assertTrue(np.isnan(nan_frame()).any())
        np.testing.assert_array_equal(zero_frame(), np.zeros((21, 3), dtype=np.float32))
        self.assertIsNone(missing_left())
        self.assertIsNone(missing_right())


class MediaPipeConversionRegressionTests(unittest.TestCase):
    def test_mediapipe21_to_hand25_has_expected_shape_roots_and_wrist(self):
        points = fixed_mediapipe_hand()
        converted = mediapipe21_to_hand25(points)

        self.assertEqual(converted.shape, (HAND_KEYPOINTS, 3))
        # The function performs wrist-relative normalization after copying MP0.
        np.testing.assert_array_equal(converted[0], np.zeros(3, dtype=np.float32))
        for output_index, mediapipe_mcp in ((5, 5), (10, 9), (15, 13), (20, 17)):
            expected = 0.5 * (points[mediapipe_mcp] - points[0])
            np.testing.assert_allclose(converted[output_index], expected, atol=1e-6)

        np.testing.assert_allclose(converted[6], points[5] - points[0], atol=1e-6)
        np.testing.assert_allclose(converted[11], points[9] - points[0], atol=1e-6)
        np.testing.assert_allclose(converted[16], points[13] - points[0], atol=1e-6)
        np.testing.assert_allclose(converted[21], points[17] - points[0], atol=1e-6)

    def test_wrist_relative_zeroes_wrist_and_is_translation_invariant(self):
        points = fixed_mediapipe_hand()[:7]
        translated = points + np.asarray([2.0, -3.0, 4.0], dtype=np.float32)

        relative = wrist_relative(points)
        translated_relative = wrist_relative(translated)

        np.testing.assert_array_equal(relative[0], np.zeros(3, dtype=np.float32))
        np.testing.assert_allclose(relative, translated_relative, atol=1e-6)


class HandWindowRegressionTests(unittest.TestCase):
    def test_window_requires_three_frames_and_preserves_chronological_order(self):
        buffer = CanonicalWindowFixture()
        frames = continuous_three_frames()

        self.assertIsNone(buffer.update(left_hand=frames[0], source="test"))
        self.assertIsNone(buffer.update(left_hand=frames[1], source="test"))
        payload = buffer.update(left_hand=frames[2], source="test")

        self.assertIsNotNone(payload)
        window = payload["hands"]["left"]
        self.assertEqual(window.shape, (3, 25, 3))
        expected = np.stack([mediapipe21_to_hand25(frame) for frame in frames])
        np.testing.assert_allclose(window, expected)
        self.assertIsNone(payload["hands"]["right"])

    def test_h5_and_realtime_paths_produce_identical_canonical_windows(self):
        frames = continuous_three_frames()
        with tempfile.TemporaryDirectory() as temp_dir:
            h5_path = Path(temp_dir) / "hands.h5"
            with h5py.File(h5_path, "w") as h5_file:
                h5_file.attrs["coordinate_frame"] = "l21"
                h5_file.attrs["coordinate_alignment"] = "source_to_l21_xyz"
                h5_file.create_dataset("frame_ids", data=np.arange(3))
                h5_file.create_dataset("timestamps", data=np.arange(3) / 30.0)
                h5_file.create_dataset("left_hand_keypoints", data=frames)
                h5_file.create_dataset(
                    "right_hand_keypoints",
                    data=np.zeros_like(frames),
                )
            h5_window = TwoHandH5Dataset(h5_path).samples[0]["left_input"]

        realtime_style = CanonicalHandPipeline()
        realtime_payload = None

        for index, frame in enumerate(frames):
            realtime_payload = realtime_style.update_detections(
                detections=[("left", frame)],
                timestamp=index / 30.0,
                source="mediapipe_approx",
            )

        self.assertIsNotNone(realtime_payload)
        realtime_window = realtime_payload["hands"]["left"]
        self.assertEqual(h5_window.shape, (3, 25, 3))
        self.assertEqual(realtime_window.shape, (3, 25, 3))
        np.testing.assert_array_equal(h5_window, realtime_window)

    def test_missing_left_and_missing_right_stay_explicit(self):
        right_only = CanonicalWindowFixture()
        for _ in range(2):
            self.assertIsNone(
                right_only.update(
                    left_hand=missing_left(),
                    right_hand=right_hand(),
                    source="test",
                )
            )
        right_payload = right_only.update(
            left_hand=missing_left(),
            right_hand=right_hand(),
            source="test",
        )
        self.assertIsNone(right_payload["hands"]["left"])
        self.assertEqual(right_payload["hands"]["right"].shape, (3, 25, 3))

        left_only = CanonicalWindowFixture()
        for _ in range(2):
            self.assertIsNone(
                left_only.update(
                    left_hand=left_hand(),
                    right_hand=missing_right(),
                    source="test",
                )
            )
        left_payload = left_only.update(
            left_hand=left_hand(),
            right_hand=missing_right(),
            source="test",
        )
        self.assertEqual(left_payload["hands"]["left"].shape, (3, 25, 3))
        self.assertIsNone(left_payload["hands"]["right"])


class IdentityResetRegressionTests(unittest.TestCase):
    def test_jump_is_rejected_side_buffer_is_cleared_and_recovery_needs_three_frames(self):
        tracker = HandIdentityTracker()
        buffer = CanonicalWindowFixture()
        normal = fixed_mediapipe_hand()

        for frame in continuous_three_frames():
            tracked = tracker.update_detections([("left", frame)])
            buffer.update(left_hand=tracked["left"], source="test")
        self.assertIsNotNone(buffer.buffer._window_or_none("left"))

        rejected = tracker.update_detections([("left", identity_jump())])
        self.assertIsNone(rejected["left"])
        buffer.reset_side("left")
        self.assertIsNone(buffer.buffer._window_or_none("left"))

        for index in range(2):
            tracked = tracker.update_detections([("left", normal)])
            self.assertIsNone(
                buffer.update(left_hand=tracked["left"], source="test")
            )
        tracked = tracker.update_detections([("left", normal)])
        recovered = buffer.update(left_hand=tracked["left"], source="test")
        self.assertIsNotNone(recovered)
        self.assertEqual(recovered["hands"]["left"].shape, (3, 25, 3))

    def test_nonfinite_and_zero_frames_are_not_trackable(self):
        tracker = HandIdentityTracker()
        tracker.update(left_hand=left_hand())
        self.assertIsNone(tracker.update(left_hand=nan_frame())["left"])
        self.assertIsNone(tracker.update(left_hand=zero_frame())["left"])


class ModelAndAngleContractRegressionTests(unittest.TestCase):
    def test_pose_transformer_contract_is_b3jcx_to_b18(self):
        model = PoseTransformer(
            num_frame=3,
            in_num_joints=25,
            in_chans=3,
            out_num_joint=18,
            out_chans=1,
            embed_dim_ratio=8,
            spatial_depth=1,
            temporal_depth=1,
            num_heads=1,
            drop_path_rate=0.0,
        ).eval()
        input_window = torch.zeros((2, 3, 25, 3), dtype=torch.float32)

        with torch.no_grad():
            output = model(input_window)

        self.assertEqual(tuple(output.shape), (2, 18))

    def test_angle18_to_dofs_drops_only_fixed_dim_zero(self):
        angle = np.arange(18, dtype=np.float32)
        self.assertEqual(float(angle[0]), 0.0)
        np.testing.assert_array_equal(angle18_to_dofs(angle), angle[1:])


class InvalidOfflineRegressionTests(unittest.TestCase):
    def test_invalid_bit_leading_zero_and_later_hold_previous(self):
        angles = np.stack(
            [
                np.zeros(18, dtype=np.float32),
                np.ones(18, dtype=np.float32),
                np.full(18, 7.0, dtype=np.float32),
            ]
        )
        valid = np.asarray([False, True, False], dtype=bool)

        _hold_last_valid_angles(angles, valid)
        self.assertFalse(bool(valid[0]))
        self.assertTrue(bool(valid[1]))
        self.assertFalse(bool(valid[2]))
        np.testing.assert_array_equal(angles[0], np.zeros(18, dtype=np.float32))
        np.testing.assert_array_equal(angles[1], np.ones(18, dtype=np.float32))
        np.testing.assert_array_equal(angles[2], np.ones(18, dtype=np.float32))

    def test_angle_h5_iterator_preserves_invalid_bits_and_hold_policy(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "angles.h5"
            values = np.stack(
                [
                    np.zeros(18, dtype=np.float32),
                    np.ones(18, dtype=np.float32),
                    np.full(18, 7.0, dtype=np.float32),
                ]
            )
            with h5py.File(path, "w") as h5_file:
                h5_file.create_dataset("frame_ids", data=np.arange(3))
                h5_file.create_dataset("timestamps", data=np.arange(3) / 30.0)
                for side in ("left", "right"):
                    h5_file.create_dataset(f"{side}_angles", data=values)
                    h5_file.create_dataset(
                        f"{side}_valid", data=np.asarray([False, True, False])
                    )

            frames = list(iter_angle_h5(path))

        self.assertFalse(frames[0].left_valid)
        self.assertTrue(frames[1].left_valid)
        self.assertFalse(frames[2].left_valid)
        np.testing.assert_array_equal(frames[0].left, np.zeros(18, dtype=np.float32))
        np.testing.assert_array_equal(frames[1].left, np.ones(18, dtype=np.float32))
        np.testing.assert_array_equal(frames[2].left, np.ones(18, dtype=np.float32))


if __name__ == "__main__":
    unittest.main()
