"""Wrist-target estimation: signs, mirroring, degeneracy, quaternion round-trip."""

import unittest
from pathlib import Path

import h5py
import numpy as np

from teleoperation.retargeting.arm.wrist_targets import (
    WristMapping,
    matrix_from_quaternion,
    palm_frame_from_landmarks,
    robot_orientation,
    rotation_matrix_to_quaternion,
    wrist_pose_from_landmarks,
)

ROOT = Path(__file__).resolve().parents[1]
SAMPLE_HAND = ROOT / "datasets" / "samples" / "human_hand_demo_right.h5"
SAMPLE_AVAILABLE = SAMPLE_HAND.is_file()


def sample_hand(index=0, translation=(0.0, 0.0, 0.0)):
    """One realistic (21,3) hand from the committed synthetic sample."""
    with h5py.File(SAMPLE_HAND, "r") as handle:
        points = np.asarray(handle["keypoints_3d"][:], dtype=np.float64)
    return points[index] + np.asarray(translation, dtype=np.float64)


def rodrigues(axis, angle):
    """Rotation matrix about ``axis`` by ``angle`` (test-local, no scipy)."""
    vector = np.asarray(axis, dtype=np.float64)
    vector = vector / np.linalg.norm(vector)
    cross = np.array([[0.0, -vector[2], vector[1]],
                      [vector[2], 0.0, -vector[0]],
                      [-vector[1], vector[0], 0.0]])
    return np.eye(3) + np.sin(angle) * cross + (1.0 - np.cos(angle)) * (cross @ cross)


@unittest.skipUnless(SAMPLE_AVAILABLE, "datasets/samples/human_hand_demo_right.h5 is missing")
class PalmFrameTests(unittest.TestCase):
    def test_frame_is_orthonormal_and_right_handed(self):
        frame = palm_frame_from_landmarks(sample_hand())
        self.assertIsNotNone(frame)
        np.testing.assert_allclose(frame.T @ frame, np.eye(3), atol=1e-9)
        self.assertAlmostEqual(float(np.linalg.det(frame)), 1.0, places=9)
        np.testing.assert_allclose(frame[:, 2], np.cross(frame[:, 0], frame[:, 1]),
                                   atol=1e-9)

    def test_degenerate_inputs_return_none(self):
        cases = {"missing": None, "wrong shape": np.zeros((20, 3)),
                 "non finite": np.full((21, 3), np.nan), "all equal": np.zeros((21, 3))}
        for name, points in cases.items():
            with self.subTest(name=name):
                self.assertIsNone(palm_frame_from_landmarks(points))


@unittest.skipUnless(SAMPLE_AVAILABLE, "datasets/samples/human_hand_demo_right.h5 is missing")
class WristPoseTests(unittest.TestCase):
    def setUp(self):
        self.hand = sample_hand()

    def test_image_offsets_map_with_the_documented_signs(self):
        mapping = WristMapping()
        base = wrist_pose_from_landmarks(self.hand, "left", mapping)
        moved_x = wrist_pose_from_landmarks(self.hand + (0.1, 0.0, 0.0), "left", mapping)
        moved_y = wrist_pose_from_landmarks(self.hand + (0.0, 0.1, 0.0), "left", mapping)
        moved_z = wrist_pose_from_landmarks(self.hand + (0.0, 0.0, -0.1), "left", mapping)
        self.assertTrue(base.valid)
        # image x -> robot Y (lateral, +Y is the operator's left)
        np.testing.assert_allclose(moved_x.position - base.position,
                                   [0.0, 0.1 * mapping.scale, 0.0], atol=1e-9)
        # image y grows downwards -> robot Z (up) decreases
        np.testing.assert_allclose(moved_y.position - base.position,
                                   [0.0, 0.0, -0.1 * mapping.scale], atol=1e-9)
        # image z negative = closer to the camera -> robot X (forward)
        np.testing.assert_allclose(moved_z.position - base.position,
                                   [0.1 * mapping.scale, 0.0, 0.0], atol=1e-9)

    def test_position_is_side_independent(self):
        """One formula covers both hands: the mirroring comes from robot +Y."""
        mapping = WristMapping()
        shifted = self.hand + (0.04, -0.02, 0.01)
        left = wrist_pose_from_landmarks(shifted, "left", mapping)
        right = wrist_pose_from_landmarks(shifted, "right", mapping)
        np.testing.assert_allclose(left.position, right.position, atol=1e-12)
        np.testing.assert_allclose(matrix_from_quaternion(left.quaternion),
                                   matrix_from_quaternion(right.quaternion), atol=1e-12)

    def test_missing_or_degenerate_hand_is_invalid(self):
        target = wrist_pose_from_landmarks(None, "right")
        self.assertFalse(target.valid)
        self.assertTrue(target.reason)
        np.testing.assert_array_equal(target.quaternion, [0.0, 0.0, 0.0, 1.0])
        with self.assertRaisesRegex(ValueError, "side"):
            wrist_pose_from_landmarks(self.hand, "middle")

    def test_orientation_is_proper_and_tracks_the_palm(self):
        mapping = WristMapping()
        target = wrist_pose_from_landmarks(self.hand, "right", mapping)
        matrix = matrix_from_quaternion(target.quaternion)
        np.testing.assert_allclose(matrix.T @ matrix, np.eye(3), atol=1e-9)
        self.assertAlmostEqual(float(np.linalg.det(matrix)), 1.0, places=9)
        expected = robot_orientation(palm_frame_from_landmarks(self.hand), mapping)
        np.testing.assert_allclose(matrix, expected, atol=1e-9)
        np.testing.assert_allclose(rotation_matrix_to_quaternion(matrix),
                                   target.quaternion, atol=1e-12)

    def test_rotating_the_hand_rotates_the_target(self):
        mapping = WristMapping()
        palm = palm_frame_from_landmarks(self.hand)
        rotation = rodrigues(palm[:, 1], 0.4)
        wrist = self.hand[0]
        rotated = (rotation @ (self.hand - wrist).T).T + wrist
        base = wrist_pose_from_landmarks(self.hand, "right", mapping)
        moved = wrist_pose_from_landmarks(rotated, "right", mapping)
        self.assertTrue(moved.valid)
        np.testing.assert_allclose(moved.position, base.position, atol=1e-9)
        change = float(np.linalg.norm(matrix_from_quaternion(moved.quaternion)
                                     - matrix_from_quaternion(base.quaternion)))
        self.assertGreater(change, 0.1)


class QuaternionTests(unittest.TestCase):
    def test_round_trip_for_known_rotations(self):
        cases = (((1, 0, 0), 0.0), ((0, 0, 1), 0.7),
                 ((0.3, 0.5, -0.8), 2.4), ((1.0, 1.0, 1.0), 3.1))
        for axis, angle in cases:
            with self.subTest(axis=axis, angle=angle):
                matrix = rodrigues(axis, angle)
                quaternion = rotation_matrix_to_quaternion(matrix)
                self.assertAlmostEqual(float(np.linalg.norm(quaternion)), 1.0, places=12)
                np.testing.assert_allclose(matrix_from_quaternion(quaternion), matrix,
                                           atol=1e-9)

    def test_mapping_validation(self):
        cases = ({"scale": 0.0}, {"axis_order": ("X", "X", "Z")},
                 {"axis_signs": (1.0, 1.0)}, {"base_offset": (1.0, 2.0)},
                 {"position_order": ("X", "X", "Z")}, {"position_signs": (1.0, -1.0)})
        for kwargs in cases:
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                WristMapping(**kwargs)


if __name__ == "__main__":
    unittest.main()
