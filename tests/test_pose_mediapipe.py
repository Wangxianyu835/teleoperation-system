"""MediaPipe pose arm subset parsing."""

from types import SimpleNamespace as NS
import unittest

import numpy as np

from teleoperation.inputs.pose_mediapipe import parse_pose_result


def pose_landmarks(visibility=0.9):
    points = [NS(x=0.0, y=0.0, z=0.0, visibility=visibility) for _ in range(33)]
    for index, value in zip((11, 12, 13, 14, 15, 16), (0.1, 0.2, 0.3, 0.4, 0.5, 0.6)):
        points[index] = NS(
            x=value,
            y=value + 0.01,
            z=value + 0.02,
            visibility=visibility,
        )
    return points


def world_landmarks(visibility=0.9):
    points = pose_landmarks(visibility)
    return [
        NS(x=point.x + 1.0, y=point.y + 1.0, z=point.z + 1.0, visibility=point.visibility)
        for point in points
    ]


def result(landmarks=True, world=True, visibility=0.9):
    pose = landmarks if isinstance(landmarks, list) else (
        [pose_landmarks(visibility)] if landmarks else []
    )
    pose_world = world if isinstance(world, list) else (
        [world_landmarks(visibility)] if world else []
    )
    return NS(
        pose_landmarks=pose,
        pose_world_landmarks=pose_world,
    )


class PoseMediaPipeTests(unittest.TestCase):
    def test_parses_shoulder_elbow_wrist_in_project_order(self):
        parsed = parse_pose_result(result())
        self.assertEqual(parsed["pose_keypoints"].shape, (6, 3))
        self.assertEqual(parsed["pose_world_keypoints"].shape, (6, 3))
        self.assertEqual(parsed["left_arm_keypoints"].shape, (3, 3))
        self.assertEqual(parsed["right_arm_keypoints"].shape, (3, 3))
        np.testing.assert_allclose(parsed["left_arm_keypoints"][:, 0], [0.1, 0.3, 0.5])
        np.testing.assert_allclose(parsed["right_arm_keypoints"][:, 0], [0.2, 0.4, 0.6])
        self.assertTrue(parsed["left_arm_valid"])
        self.assertTrue(parsed["right_arm_valid"])
        self.assertEqual(parsed["metadata"]["arm_world_origin"], "hip_midpoint")
        self.assertEqual(parsed["metadata"]["arm_length_unit"], "m")

    def test_missing_pose_returns_invalid_nan_fields(self):
        parsed = parse_pose_result(result(landmarks=False, world=False))
        self.assertTrue(np.isnan(parsed["pose_keypoints"]).all())
        self.assertFalse(parsed["left_arm_valid"])
        self.assertFalse(parsed["right_arm_valid"])

    def test_low_visibility_invalidates_only_affected_side(self):
        landmarks = pose_landmarks(0.9)
        landmarks[11] = NS(x=0.1, y=0.1, z=0.1, visibility=0.01)
        parsed = parse_pose_result(result(landmarks=[landmarks]))
        self.assertFalse(parsed["left_arm_valid"])
        self.assertTrue(parsed["right_arm_valid"])

    def test_missing_world_landmarks_keep_metric_fields_invalid(self):
        parsed = parse_pose_result(result(world=False))
        self.assertTrue(np.isnan(parsed["pose_world_keypoints"]).all())
        self.assertFalse(parsed["left_arm_valid"])
        self.assertFalse(parsed["right_arm_valid"])


if __name__ == "__main__":
    unittest.main()
