"""World landmark capture persistence and CLI registration."""

import os
from pathlib import Path
from types import SimpleNamespace
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

import numpy as np

from teleoperation.apps import world_landmark_capture
from teleoperation.cli import build_parser
from teleoperation.data.world_landmark_recording import (
    read_world_landmark_recording,
    write_world_landmark_recording,
)

ROOT = Path(__file__).resolve().parents[1]


def points(value):
    return np.full((21, 3), value, dtype=np.float32)


def frame(index, world_left=None, world_right=None):
    return {
        "frame_id": index,
        "timestamp": 1000 + index,
        "world_left": world_left,
        "world_right": world_right,
    }


class WorldLandmarkRecordingTests(unittest.TestCase):
    def test_round_trip_preserves_validity_units_and_timestamps(self):
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

        self.assertEqual(arrays["frame_ids"].tolist(), [0, 1])
        self.assertEqual(arrays["timestamps"].tolist(), [1000.0, 1001.0])
        self.assertEqual(arrays["left_world_landmarks"].shape, (2, 21, 3))
        self.assertEqual(arrays["right_world_landmarks"].shape, (2, 21, 3))
        self.assertEqual(arrays["left_valid"].tolist(), [True, False])
        self.assertEqual(arrays["right_valid"].tolist(), [False, True])
        self.assertTrue(np.isnan(arrays["left_world_landmarks"][1]).all())
        self.assertTrue(np.isnan(arrays["right_world_landmarks"][0]).all())
        self.assertEqual(attributes["source_landmark_space"], "mediapipe_world_meters")
        self.assertEqual(attributes["length_unit"], "m")
        self.assertEqual(attributes["camera_index"], "1")

    def test_writer_rejects_empty_and_nonfinite_captures(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "world.h5"
            with self.assertRaisesRegex(ValueError, "at least one frame"):
                write_world_landmark_recording(path, [])
            invalid = points(0.01)
            invalid[3, 1] = np.nan
            with self.assertRaisesRegex(ValueError, "finite"):
                write_world_landmark_recording(path, [frame(0, world_left=invalid)])

    def test_capture_app_writes_the_selected_output(self):
        class FakeCamera:
            def __init__(self):
                self.released = False

            def release(self, *, wait_for_mediapipe=True, close_landmarker=True):
                self.released = True

            def next_frame(self, *, include_world=False):
                self.assert_include_world = include_world
                index = self.index
                self.index += 1
                return frame(index, world_left=points(0.01 * (index + 1)))

        fake = FakeCamera()
        fake.index = 0
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "capture.h5"
            args = SimpleNamespace(
                model_asset_path="model.task",
                camera_index=2,
                frames=3,
                output=path,
                width=640,
                height=480,
                fps=30,
                preview=False,
                max_fps=0,
            )
            with mock.patch.object(
                world_landmark_capture,
                "MediaPipeCameraInput",
                return_value=fake,
            ):
                self.assertEqual(world_landmark_capture.main(args), 0)
            arrays, attributes = read_world_landmark_recording(path)

        self.assertTrue(fake.released)
        self.assertTrue(fake.assert_include_world)
        self.assertEqual(arrays["timestamps"].shape, (3,))
        self.assertEqual(attributes["camera_index"], "2")
        self.assertEqual(attributes["source_landmark_space"], "mediapipe_world_meters")

    def test_record_world_command_is_registered(self):
        args = build_parser().parse_args(
            ["hand", "record-world", "--output", "capture.h5"]
        )
        self.assertEqual(args.command, "record-world")
        self.assertEqual(args.output, Path("capture.h5"))
        self.assertTrue(args.preview)
        self.assertEqual(args.max_fps, 10.0)
        self.assertTrue(args.fast_exit)

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
        self.assertIn("--frames", result.stdout)
        self.assertIn("--no-preview", result.stdout)
        self.assertIn("--max-fps", result.stdout)

    def test_capture_module_is_directly_runnable(self):
        result = subprocess.run(
            [
                sys.executable,
                "-B",
                str(ROOT / "src" / "teleoperation" / "apps" / "world_landmark_capture.py"),
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
        self.assertIn("--frames", result.stdout)


if __name__ == "__main__":
    unittest.main()
