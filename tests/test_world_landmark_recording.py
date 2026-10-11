"""Combined hand/body capture persistence and compatibility."""

import os
from pathlib import Path
from types import SimpleNamespace
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

import numpy as np

from teleoperation.applications import world_landmark_capture
from teleoperation.cli import build_parser
from teleoperation.data.command_h5 import _load_arm_observations
from teleoperation.data.hand_body_recording import (
    read_hand_body_recording,
    write_hand_body_recording,
)
from teleoperation.data.hand_h5 import load_twohand_h5
from teleoperation.data.world_landmark_recording import (
    read_world_landmark_recording,
    write_world_landmark_recording,
)

ROOT = Path(__file__).resolve().parents[1]


def points(value):
    return np.full((21, 3), value, dtype=np.float32)


def arm_points(value):
    return np.full((3, 3), value, dtype=np.float32)


def frame(index, world_left=None, world_right=None):
    return {
        "frame_id": index,
        "timestamp": 1000 + index,
        "world_left": world_left,
        "world_right": world_right,
    }


def combined_frame(index):
    return {
        "frame_id": index,
        "timestamp": 1000 + index,
        "left_hand_keypoints": points(0.01 + index * 0.001),
        "right_hand_keypoints": points(0.02 + index * 0.001),
        "left_world_landmarks": points(0.01 + index * 0.001),
        "right_world_landmarks": points(0.02 + index * 0.001),
        "left_arm_keypoints": arm_points(0.1 + index * 0.01),
        "right_arm_keypoints": arm_points(0.2 + index * 0.01),
        "left_arm_world_keypoints": arm_points(0.1 + index * 0.01),
        "right_arm_world_keypoints": arm_points(0.2 + index * 0.01),
        "left_arm_valid": True,
        "right_arm_valid": True,
        "pose_visibility": np.ones(6, dtype=np.float32),
        "metadata": {"image_width": 320, "image_height": 240},
    }


class LegacyWorldRecordingTests(unittest.TestCase):
    def test_legacy_round_trip_still_works(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "world.h5"
            write_world_landmark_recording(
                path,
                [
                    frame(0, world_left=points(0.01)),
                    frame(1, world_right=points(0.02)),
                ],
                metadata={"camera_index": 1},
            )
            arrays, attributes = read_world_landmark_recording(path)

        self.assertEqual(arrays["left_valid"].tolist(), [True, False])
        self.assertEqual(arrays["right_valid"].tolist(), [False, True])
        self.assertEqual(attributes["source_landmark_space"], "mediapipe_world_meters")


class CombinedCaptureTests(unittest.TestCase):
    def test_combined_round_trip_contains_both_spaces(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "capture.h5"
            write_hand_body_recording(path, [combined_frame(0), combined_frame(1)])
            arrays, attributes = read_hand_body_recording(path)

        self.assertEqual(arrays["left_hand_keypoints"].shape, (2, 21, 3))
        self.assertEqual(arrays["left_world_landmarks"].shape, (2, 21, 3))
        self.assertEqual(arrays["left_arm_keypoints"].shape, (2, 3, 3))
        self.assertEqual(arrays["left_arm_world_keypoints"].shape, (2, 3, 3))
        self.assertEqual(attributes["source_landmark_space"], "mediapipe_normalized")
        self.assertEqual(attributes["hand_world_landmark_space"], "mediapipe_world_meters")
        self.assertEqual(attributes["arm_world_space"], "mediapipe_pose_world_meters")
        self.assertEqual(attributes["length_unit"], "m")
        self.assertEqual(attributes["image_width"], 320)
        self.assertEqual(attributes["image_height"], 240)

    def test_combined_capture_is_readable_by_hand_and_arm_pipelines(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "capture.h5"
            write_hand_body_recording(path, [combined_frame(0), combined_frame(1)])
            frame_ids, timestamps, hands = load_twohand_h5(path, require_aligned=False)
            arm_ids, arm_times, arms, arm_valid = _load_arm_observations(path)

        np.testing.assert_array_equal(frame_ids, [0, 1])
        np.testing.assert_array_equal(arm_ids, frame_ids)
        np.testing.assert_allclose(timestamps, arm_times)
        self.assertEqual(hands["left"].shape, (2, 21, 3))
        self.assertEqual(arms["right"].shape, (2, 3, 3))
        self.assertTrue(arm_valid["left"].all())
        self.assertTrue(arm_valid["right"].all())

    def test_capture_app_writes_combined_output(self):
        class FakeCamera:
            def __init__(self):
                self.released = False
                self.index = 0

            def release(self, *, wait_for_mediapipe=True, close_landmarker=True):
                self.released = True

            def next_frame(self, *, include_world=False, include_pose=False):
                self.assert_include_world = include_world
                self.assert_include_pose = include_pose
                value = combined_frame(self.index)
                self.index += 1
                return value

        fake = FakeCamera()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "capture.h5"
            args = SimpleNamespace(
                model_asset_path="hand.task",
                pose_model_asset_path="pose.task",
                pose_min_visibility=0.2,
                camera_index=2,
                frames=2,
                output=path,
                width=320,
                height=240,
                fps=30,
                preview=False,
                max_fps=0,
            )
            with mock.patch.object(
                world_landmark_capture,
                "_ensure_model",
                side_effect=lambda path, urls: Path(path),
            ), mock.patch.object(
                world_landmark_capture,
                "MediaPipeCameraInput",
                return_value=fake,
            ):
                self.assertEqual(world_landmark_capture.main(args), 0)
            arrays, attributes = read_hand_body_recording(path)

        self.assertTrue(fake.released)
        self.assertTrue(fake.assert_include_world)
        self.assertTrue(fake.assert_include_pose)
        self.assertEqual(arrays["timestamps"].shape, (2,))
        self.assertEqual(attributes["source_landmark_space"], "mediapipe_normalized")

    def test_record_world_command_is_registered(self):
        args = build_parser().parse_args(
            ["hand", "record-world", "--output", "capture.h5"]
        )
        self.assertEqual(args.command, "record-world")
        self.assertEqual(args.output, Path("capture.h5"))
        self.assertTrue(args.preview)
        self.assertEqual(args.max_fps, 5.0)
        self.assertTrue(args.fast_exit)
        self.assertIn("pose_model_asset_path", vars(args))

    def test_root_runner_is_directly_runnable(self):
        result = subprocess.run(
            [sys.executable, "-B", str(ROOT / "record_world_landmarks.py"), "--help"],
            cwd=ROOT,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=30,
            env={**os.environ, "PYTHONUTF8": "1", "PYTHONIOENCODING": "utf-8"},
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("--pose-model-asset-path", result.stdout)
        self.assertIn("--no-preview", result.stdout)

    def test_capture_module_is_directly_runnable(self):
        result = subprocess.run(
            [
                sys.executable,
                "-B",
                str(ROOT / "src" / "teleoperation" / "applications" / "world_landmark_capture.py"),
                "--help",
            ],
            cwd=ROOT,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=30,
            env={**os.environ, "PYTHONUTF8": "1", "PYTHONIOENCODING": "utf-8"},
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("--pose-model-asset-path", result.stdout)


if __name__ == "__main__":
    unittest.main()
