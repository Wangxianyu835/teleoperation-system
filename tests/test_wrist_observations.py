"""Regression specifications for source-space wrists. Not device acceptance."""
import unittest
from unittest.mock import patch
import numpy as np
from scipy.spatial.transform import Rotation

from teleoperation.contracts.wrist import WristObservation
from teleoperation.inputs.wrist import vr_wrist, VR_REFERENCE
from teleoperation.inputs.visionpro import VisionProInput
from teleoperation.retargeting.arm.wrist_orientation import WristTracker


class WristObservationTests(unittest.TestCase):
    def test_vr_accepts_sdk_singleton_and_rejects_reflection(self):
        matrix = np.eye(4)
        matrix[:3, 3] = [.1, .2, .3]
        wrist = vr_wrist(matrix[None], 1.)
        self.assertTrue(wrist.orientation_valid)
        self.assertEqual(wrist.reference_frame, VR_REFERENCE)
        np.testing.assert_allclose(wrist.position, matrix[:3, 3])
        matrix[0, 0] = -1
        self.assertFalse(vr_wrist(matrix).orientation_valid)

    def test_missing_and_placeholder_vr_do_not_create_valid_identity(self):
        for value in (None, np.eye(4), np.full((4, 4), np.nan)):
            self.assertFalse(vr_wrist(value).orientation_valid)

    def test_hold_preserves_delta_and_source_change_realigns(self):
        tracker = WristTracker(np.eye(3), calibration_frames=1)
        tracker.update(vr_wrist(np.diag([1, 1, 1, 1]) + np.array(
            [[0, 0, 0, .1], [0, 0, 0, .2], [0, 0, 0, .3], [0, 0, 0, 0]]), 1.), .03)
        self.assertEqual(tracker.status, "aligned")
        before = tracker.delta.as_quat().copy()
        tracker.update(None, .6)
        self.assertEqual(tracker.status, "lost_hold")
        np.testing.assert_array_equal(tracker.delta.as_quat(), before)
        tracker.update(WristObservation(np.array([0., 0., 0., 1.]), "smplx_camera_axes",
                       "smplx_wrist_local", "smplerx", .01, orientation_valid=True), .03)
        self.assertEqual(tracker.generation, 2)

    def test_smplerx_camera_axes_are_supported(self):
        tracker = WristTracker(np.eye(3), calibration_frames=1)
        first = WristObservation(np.array([0., 0., 0., 1.]), "smplx_camera_axes",
                                 "smplx_wrist_local", "smplerx", 1.0,
                                 orientation_valid=True)
        tracker.update(first, .03)
        self.assertEqual(tracker.status, "aligned")
        moved = WristObservation(
            Rotation.from_euler("z", .2).as_quat(), "smplx_camera_axes",
            "smplx_wrist_local", "smplerx", 1.1, orientation_valid=True,
        )
        tracker.update(moved, .03)
        self.assertNotEqual(tracker.reason, "unsupported wrist reference frame")

    def test_cached_sdk_packet_does_not_gain_new_wrist_timestamps(self):
        matrix = np.eye(4)
        matrix[:3, 3] = [.1, .2, .3]
        class Stream:
            payload = {"left_wrist": matrix[None]}
            def get_latest(self):
                return self.payload
        source = VisionProInput("unused", "left", Stream())
        with patch("teleoperation.inputs.visionpro.time.time", side_effect=[100., 101.]):
            first = source.next_observation()
            second = source.next_observation()
        self.assertEqual(first.wrists["left"].timestamp, second.wrists["left"].timestamp)
        tracker = WristTracker(np.eye(3))
        tracker.update(first.wrists["left"], .03)
        tracker.update(second.wrists["left"], .03)
        self.assertIsNone(tracker.neutral)


if __name__ == "__main__":
    unittest.main()
