"""Camera -> capture H5 entry point: contract tests with no camera, no MediaPipe.

Everything here drives ``apps/camera_record.py`` with an in-process frame stub, so
the suite never opens a camera, never imports cv2/mediapipe and never writes
outside a temporary directory. The committed synthetic hand
(``datasets/samples/human_hand_demo_right.h5``) supplies realistic (21,3) rows.
"""

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import h5py
import numpy as np

from teleoperation.apps.camera_record import (
    MODEL_ASSET_CANDIDATES,
    AngleCollector,
    build_parser,
    default_model_asset,
    main,
    record_frames,
)
from teleoperation.apps.camera_record import run as camera_run
from teleoperation.apps.hand_align import align_h5
from teleoperation.apps.hand_inspect import inspect_angle_h5
from teleoperation.data.capture_h5 import (
    CAPTURE_SIDE_KEYS,
    VALID_SIDE_KEYS,
    CaptureH5Writer,
    read_capture_summary,
)
from teleoperation.data.hand_h5 import load_twohand_h5, read_angle_arrays
from teleoperation.retargeting.hand.config import HAND_ANGLE_DIM

ROOT = Path(__file__).resolve().parents[1]
SAMPLE_HAND = ROOT / "datasets" / "samples" / "human_hand_demo_right.h5"
START_MS = 1_700_000_000_000
SAMPLE_AVAILABLE = SAMPLE_HAND.is_file()


def sample_rows(count):
    """The first ``count`` (21,3) rows of the committed synthetic hand."""
    with h5py.File(SAMPLE_HAND, "r") as handle:
        points = np.asarray(handle["keypoints_3d"][:], dtype=np.float32)
    if count > points.shape[0]:
        raise ValueError(f"the sample hand only has {points.shape[0]} frames")
    return points[:count]


def sample_frames(count, side="right"):
    """Frames shaped exactly like ``MediaPipeCameraInput.next_frame()`` output."""
    frames = []
    for index, row in enumerate(sample_rows(count)):
        frame = {
            "left": None,
            "right": None,
            "timestamp": START_MS + index * 33,
            "metadata": {"frame_index": index, "camera_index": 0,
                         "image_width": 640, "image_height": 480},
        }
        frame[side] = row
        frames.append(frame)
    return frames


class StubCamera:
    """Stand-in for ``MediaPipeCameraInput``: only next_frame()/close() are used."""

    def __init__(self, frames):
        self.frames = list(frames)
        self.index = 0
        self.closed = False

    def next_frame(self):
        if self.index >= len(self.frames):
            raise RuntimeError("stub camera ran out of frames")
        frame = self.frames[self.index]
        self.index += 1
        return frame

    def close(self):
        self.closed = True


@unittest.skipUnless(SAMPLE_AVAILABLE, "datasets/samples/human_hand_demo_right.h5 is missing")
class CaptureWriterTests(unittest.TestCase):
    """Layout, zero-fill for a missing hand, and the guards around append()."""

    def test_layout_zeros_and_attributes(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "capture.h5"
            writer = CaptureH5Writer(
                path, camera_index=0, image_width=640, image_height=480, fps=30,
                model_asset_path="hand_landmarker.task",
            )
            for frame in sample_frames(3):
                writer.append(frame)
            summary = writer.close()
            self.assertEqual(summary["frames"], 3)
            self.assertEqual(summary["left_valid_frames"], 0)
            self.assertEqual(summary["right_valid_frames"], 3)
            self.assertEqual(summary["any_hand_frames"], 3)
            self.assertTrue(summary["closed"])
            self.assertAlmostEqual(summary["duration_seconds"], 0.066, places=6)
            with h5py.File(path, "r") as handle:
                self.assertEqual(handle[CAPTURE_SIDE_KEYS["left"]].shape, (3, 21, 3))
                self.assertEqual(handle[CAPTURE_SIDE_KEYS["left"]].dtype, np.float32)
                self.assertEqual(handle[CAPTURE_SIDE_KEYS["right"]].shape, (3, 21, 3))
                self.assertFalse(np.any(handle[CAPTURE_SIDE_KEYS["left"]][:]))
                self.assertFalse(np.asarray(handle[VALID_SIDE_KEYS["left"]][:]).any())
                self.assertTrue(np.asarray(handle[VALID_SIDE_KEYS["right"]][:]).all())
                np.testing.assert_array_equal(handle["frame_ids"][:], [0, 1, 2])
                np.testing.assert_allclose(handle["timestamps"][:], [0.0, 0.033, 0.066])
                np.testing.assert_array_equal(
                    handle["unix_ms"][:], [START_MS, START_MS + 33, START_MS + 66])
                self.assertEqual(handle.attrs["frames"], 3)
                self.assertEqual(handle.attrs["missing_hand_encoding"], "zeros")
                self.assertEqual(handle.attrs["source_landmark_space"],
                                 "mediapipe_normalized")
                self.assertEqual(handle.attrs["timestamp_unit"], "relative_seconds")
                self.assertEqual(handle.attrs["camera_index"], 0)
            read_back = read_capture_summary(path)
            self.assertEqual(read_back["frames"], 3)
            self.assertEqual(read_back["datasets"]["right_hand_keypoints"], (3, 21, 3))

    def test_append_guards(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "capture.h5"
            writer = CaptureH5Writer(path)
            first = sample_frames(1)[0]
            writer.append(first)
            with self.assertRaisesRegex(ValueError, "increase strictly"):
                writer.append(first)
            with self.assertRaisesRegex(ValueError, "shape"):
                writer.append({**first, "timestamp": first["timestamp"] + 1,
                               "right": np.zeros((20, 3), dtype=np.float32)})
            broken = np.zeros((21, 3), dtype=np.float32)
            broken[0, 0] = np.nan
            with self.assertRaisesRegex(ValueError, "finite"):
                writer.append({**first, "timestamp": first["timestamp"] + 2,
                               "right": broken})
            writer.update_attributes({"observed_image_width": 640})
            writer.close()
            with self.assertRaisesRegex(ValueError, "closed"):
                writer.append({**first, "timestamp": first["timestamp"] + 3})
            self.assertEqual(
                read_capture_summary(path)["attributes"]["observed_image_width"], 640)


class FakeClock:
    """Deterministic stand-in for ``time.perf_counter`` (advances per call)."""

    def __init__(self, step=0.5):
        self.now = 0.0
        self.step = step

    def __call__(self):
        current = self.now
        self.now += self.step
        return current


@unittest.skipUnless(SAMPLE_AVAILABLE, "datasets/samples/human_hand_demo_right.h5 is missing")
class RecordFramesTests(unittest.TestCase):
    """Loop limits, the per-frame hook, and the teammate readers downstream."""

    def test_max_frames_stops_and_the_hook_sees_every_row(self):
        with tempfile.TemporaryDirectory() as folder:
            writer = CaptureH5Writer(Path(folder) / "capture.h5")
            source = StubCamera(sample_frames(8))
            seen = []

            def hook(row, frame, frame_id, elapsed):
                seen.append((row, frame_id, round(elapsed, 3)))

            summary = record_frames(source, writer, max_frames=5, progress_every=0,
                                    on_frame=hook)
            writer.close()
            self.assertEqual(summary["frames"], 5)
            self.assertEqual([item[0] for item in seen], [0, 1, 2, 3, 4])
            self.assertEqual([item[1] for item in seen], [0, 1, 2, 3, 4])
            self.assertEqual(seen[-1][2], 0.132)
            self.assertIn("loop_fps", summary)
            self.assertEqual(source.index, 5)

    def test_max_seconds_stops_the_loop(self):
        with tempfile.TemporaryDirectory() as folder:
            writer = CaptureH5Writer(Path(folder) / "capture.h5")
            summary = record_frames(StubCamera(sample_frames(8)), writer, max_seconds=1.0,
                                    progress_every=0, clock=FakeClock(0.5))
            writer.close()
            self.assertEqual(summary["frames"], 2)

    def test_no_limit_is_a_programming_error(self):
        with tempfile.TemporaryDirectory() as folder:
            writer = CaptureH5Writer(Path(folder) / "capture.h5")
            with self.assertRaisesRegex(ValueError, "max_frames"):
                record_frames(StubCamera(sample_frames(4)), writer, max_frames=0,
                              max_seconds=0.0)
            writer.close()

    def test_align_accepts_the_recorded_capture(self):
        """The teammate's align step must read our file and count the zero frames."""
        with tempfile.TemporaryDirectory() as folder:
            raw = Path(folder) / "raw.h5"
            aligned = Path(folder) / "aligned.h5"
            frames = sample_frames(4)
            frames[2] = {**frames[2], "right": None}
            writer = CaptureH5Writer(raw)
            for frame in frames:
                writer.append(frame)
            writer.close()
            report = align_h5(raw, aligned)
            self.assertEqual(report["frames"], 4)
            self.assertEqual(report["sides"]["right"]["zero_source_frames"], 1)
            self.assertEqual(report["sides"]["left"]["zero_source_frames"], 4)
            frame_ids, timestamps, hands = load_twohand_h5(aligned)
            np.testing.assert_array_equal(frame_ids, [0, 1, 2, 3])
            self.assertEqual(timestamps.shape, (4,))
            self.assertEqual(hands["right"].shape, (4, 25, 3))
            self.assertEqual(hands["left"].shape, (4, 25, 3))


@unittest.skipUnless(SAMPLE_AVAILABLE, "datasets/samples/human_hand_demo_right.h5 is missing")
class AngleCollectorTests(unittest.TestCase):
    """The optional 18D side product must be an inspectable angle H5."""

    def test_angle_h5_is_readable_by_hand_inspect(self):
        with tempfile.TemporaryDirectory() as folder:
            raw = Path(folder) / "capture.h5"
            angles_path = Path(folder) / "angles.h5"
            frames = sample_frames(40)
            collector = AngleCollector(calibration_frames=5)
            writer = CaptureH5Writer(raw)
            summary = record_frames(StubCamera(frames), writer, max_frames=len(frames),
                                    progress_every=0, on_frame=collector)
            writer.close()
            self.assertEqual(summary["frames"], 40)
            angles = collector.write(angles_path, {"capture_h5": str(raw)})
            self.assertGreater(angles["right_valid_frames"], 0)
            self.assertEqual(angles["left_valid_frames"], 0)
            self.assertTrue(angles["calibration_status"]["right"])
            self.assertFalse(angles["calibration_status"]["left"])
            arrays, attrs = read_angle_arrays(angles_path)
            self.assertEqual(arrays["right_angles"].shape, (40, HAND_ANGLE_DIM))
            self.assertTrue(np.isfinite(arrays["right_angles"]).all())
            self.assertFalse(np.asarray(arrays["left_valid"][:]).any())
            self.assertFalse(np.any(arrays["left_angles"][:]))
            self.assertEqual(int(np.asarray(arrays["right_valid"][:]).sum()),
                             angles["right_valid_frames"])
            self.assertEqual(attrs["backend"], "geometric")
            self.assertEqual(attrs["backend_weights"], "none")
            self.assertEqual(attrs["capture_h5"], str(raw))
            inspected = inspect_angle_h5(angles_path)
            self.assertEqual(inspected["frames"], 40)
            self.assertEqual(inspected["right"]["valid"], angles["right_valid_frames"])
            self.assertEqual(inspected["right"]["nonfinite"], 0)
            self.assertEqual(inspected["left"]["nonfinite"], 0)


@unittest.skipUnless(SAMPLE_AVAILABLE, "datasets/samples/human_hand_demo_right.h5 is missing")
class CameraRecordRunTests(unittest.TestCase):
    """Full ``run()`` path with a patched camera: real writer, angles, report."""

    def test_run_writes_capture_and_angles_without_hardware(self):
        """Replacement for MediaPipeCameraInput: same constructor, no hardware."""

        class StubCameraInput:
            def __init__(self, **kwargs):
                self.kwargs = kwargs
                self.source = StubCamera(sample_frames(40))

            def next_frame(self):
                return self.source.next_frame()

            def close(self):
                self.source.close()

        with tempfile.TemporaryDirectory() as folder, patch(
                "teleoperation.inputs.mediapipe.MediaPipeCameraInput", StubCameraInput):
            raw = Path(folder) / "capture.h5"
            angles = Path(folder) / "angles.h5"
            report = Path(folder) / "report.json"
            args = build_parser().parse_args([
                "--output-h5", str(raw), "--angles-h5", str(angles),
                "--frames", "40", "--seconds", "0", "--progress-every", "0",
                "--model-asset-path", str(SAMPLE_HAND), "--report", str(report),
            ])
            exit_code = camera_run(args)
            self.assertEqual(exit_code, 0)
            capture = read_capture_summary(raw)
            self.assertEqual(capture["frames"], 40)
            self.assertEqual(capture["attributes"]["camera_index"], 0)
            self.assertEqual(capture["attributes"]["observed_image_width"], 640)
            self.assertEqual(inspect_angle_h5(angles)["frames"], 40)
            payload = json.loads(report.read_text(encoding="utf-8"))
            self.assertEqual(payload["exit_code"], 0)
            self.assertEqual(payload["valid_frames"]["right"], 40)
            self.assertIn("loop_fps", payload)
            self.assertEqual(payload["angles"]["right_valid_frames"],
                             inspect_angle_h5(angles)["right"]["valid"])


class CameraRecordCliTests(unittest.TestCase):
    """Defaults and guards of the command line; nothing here opens a camera."""

    def test_defaults_need_at_least_one_limit(self):
        args = build_parser().parse_args([])
        self.assertEqual(args.frames, 0)
        self.assertGreater(args.seconds, 0.0)
        self.assertEqual(args.camera_index, 0)
        self.assertIsNone(args.angles_h5)
        with self.assertRaises(SystemExit):
            main(["--frames", "0", "--seconds", "0"])

    def test_model_asset_discovery(self):
        if not any(candidate.is_file() for candidate in MODEL_ASSET_CANDIDATES):
            self.skipTest("no hand_landmarker.task in this checkout")
        self.assertTrue(default_model_asset().is_file())


if __name__ == "__main__":
    unittest.main()
