"""Combined hand/body capture persistence and compatibility."""

import os
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

import numpy as np

from teleoperation.applications import world_landmark_capture
from teleoperation.cli import build_parser, main as cli_main
from teleoperation.paths import (
    DEFAULT_MEDIAPIPE_ASSET,
    DEFAULT_WORLD_CAPTURE_DIR,
)
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


def frame(index, world_left=None, world_right=None):
    return {
        "frame_id": index,
        "timestamp": 1000 + index,
        "world_left": world_left,
        "world_right": world_right,
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
        from tests.smplerx_fixtures import write
        from teleoperation.data.body_recording import load_body_recording, replay_observation
        with tempfile.TemporaryDirectory() as directory:
            path = write(Path(directory) / 'capture.h5')
            arrays, attributes = read_hand_body_recording(path)
            recording = load_body_recording(path)
        self.assertEqual(arrays['left_hand_keypoints'].shape, (3,21,3))
        self.assertEqual(attributes['source'], 'smplerx')
        self.assertEqual(recording.metadata['capture_kind'], 'synthetic_validation_only')
        self.assertEqual(replay_observation(recording.samples[0]).wrists['left'].source, 'smplerx')

    def test_combined_capture_is_readable_by_hand_and_arm_pipelines(self):
        from tests.smplerx_fixtures import write
        with tempfile.TemporaryDirectory() as directory:
            path = write(Path(directory) / 'capture.h5')
            frame_ids, timestamps, hands = load_twohand_h5(path, require_aligned=False)
            arm_ids, arm_times, arms, arm_valid = _load_arm_observations(path)
        np.testing.assert_array_equal(frame_ids, [0,1,2])
        np.testing.assert_array_equal(arm_ids, frame_ids)
        np.testing.assert_allclose(timestamps, arm_times)
        self.assertEqual(hands['left'].shape, (3,21,3))
        self.assertEqual(arms['right'].shape, (3,3,3))
        self.assertTrue(arm_valid['left'].all())

    def test_capture_app_writes_combined_output(self):
        from tests.smplerx_fixtures import observation
        from teleoperation.data.body_recording import load_body_recording
        class FakeCamera:
            def __init__(self): self.index = 0; self.released = False
            def capture_sample(self):
                from teleoperation.contracts.observations import CameraFrame
                self.index += 1
                return CameraFrame(np.zeros((4,5,3),dtype=np.uint8), 10. + (self.index-1)/30., self.index)
            def process_sample(self, frame): return observation(self.index-1)[0].hands
            def release(self, **kwargs): self.released = True
        class Client:
            def __init__(self, *args, **kwargs): pass
            def submit(self, frame, rgb): self.frame = frame
            def poll(self): return observation(self.frame.sequence-1)[1]
        fake = FakeCamera()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'capture.h5'
            with mock.patch.object(world_landmark_capture, '_ensure_model', side_effect=lambda path,urls: Path(path)), mock.patch.object(world_landmark_capture, 'MediaPipeCameraInput', return_value=fake), mock.patch.object(world_landmark_capture, 'SmplerXWorkerTransport') as worker, mock.patch.object(world_landmark_capture, 'SmplerXClient', Client), mock.patch.object(world_landmark_capture.os, '_exit') as fast_exit:
                self.assertEqual(cli_main(['hand','record-world','--output',str(path),'--frames','2','--no-preview','--max-fps','0']),0)
                fast_exit.assert_called_once_with(0)
                worker.return_value.close.assert_called_once()
            recording = load_body_recording(path)
        self.assertTrue(fake.released)
        self.assertEqual(len(recording.samples),2)
        self.assertIn('worker_packet',recording.samples[0])

    def test_record_world_command_is_registered(self):
        args = build_parser().parse_args(
            ["hand", "record-world", "--output", "capture.h5"]
        )
        self.assertEqual(args.command, "record-world")
        self.assertEqual(args.output, Path("capture.h5"))
        self.assertTrue(args.preview)
        self.assertEqual(args.max_fps, 5.0)
        self.assertTrue(args.fast_exit)
        self.assertIn("smplerx_worker_python", vars(args))
        self.assertEqual(args.model_asset_path, DEFAULT_MEDIAPIPE_ASSET)
        self.assertFalse(args.no_arms)

    def test_record_world_help_works_without_site_packages(self):
        result = subprocess.run(
            [sys.executable, "-B", "-S", "-m", "teleoperation", "hand", "record-world", "--help"],
            cwd=ROOT,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=30,
            env={**os.environ, "PYTHONUTF8": "1", "PYTHONIOENCODING": "utf-8",
                 "PYTHONPATH": str(ROOT / "src")},
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("--smplerx-checkpoint", result.stdout)
        self.assertIn("--no-preview", result.stdout)
        self.assertIn("--output-dir", result.stdout)
        self.assertIn("SMPL-X", result.stdout)

    def test_automatic_output_uses_default_directory_and_shanghai_time(self):
        args = build_parser().parse_args(["hand", "record-world"])
        self.assertIsNone(args.output)
        self.assertEqual(args.output_dir, DEFAULT_WORLD_CAPTURE_DIR)
        started_at = datetime(2026, 10, 11, 12, 30, 0,
                              tzinfo=timezone(timedelta(hours=8)))
        with mock.patch.object(world_landmark_capture, "datetime") as clock, \
                mock.patch.object(Path, "exists", return_value=False):
            clock.now.return_value = started_at
            output = world_landmark_capture._output_path(args)
            clock.now.assert_called_once_with(timezone(timedelta(hours=8)))
        self.assertEqual(output, DEFAULT_WORLD_CAPTURE_DIR / "hand_body_20261011_123000.h5")

    def test_automatic_output_avoids_existing_recordings(self):
        with tempfile.TemporaryDirectory() as directory:
            args = build_parser().parse_args([
                "hand", "record-world", "--output-dir", directory,
            ])
            existing = Path(directory) / "hand_body_20261011_123000.h5"
            existing.write_bytes(b"existing capture")
            with mock.patch.object(world_landmark_capture, "datetime") as clock:
                clock.now.return_value = datetime(2026, 10, 11, 12, 30, 0)
                output = world_landmark_capture._output_path(args)
            self.assertEqual(output, existing.with_stem(existing.stem + "_01"))
            self.assertEqual(existing.read_bytes(), b"existing capture")

    def test_explicit_output_filename_is_preserved(self):
        args = SimpleNamespace(output=Path("named/session.hdf5"), output_dir=Path("unused"))
        self.assertEqual(world_landmark_capture._output_path(args), args.output)

    def test_output_and_output_dir_are_mutually_exclusive(self):
        with mock.patch("sys.stderr"), self.assertRaises(SystemExit) as failure:
            build_parser().parse_args([
                "hand", "record-world", "--output", "capture.h5", "--output-dir", "other",
            ])
        self.assertEqual(failure.exception.code, 2)


if __name__ == "__main__":
    unittest.main()
