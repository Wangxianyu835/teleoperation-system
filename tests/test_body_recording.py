import tempfile
import unittest
from pathlib import Path

import h5py
import numpy as np

from teleoperation.data.body_recording import load_body_recording, save_body_recording


class BodyRecordingTests(unittest.TestCase):
    def _sample(self):
        return {
            "sample_id": "camera:12",
            "capture_time_s": 1.25,
            "completed_time_s": 1.31,
            "valid": True,
            "reason": "ok",
            "joints_root_m": np.arange(9, dtype=np.float32).reshape(3, 3),
            "global_rotations": np.broadcast_to(np.eye(3, dtype=np.float32), (3, 3, 3)).copy(),
            "wrist_quat_left": np.array([0., 0., 0., 1.], dtype=np.float32),
            "wrist_quat_right": np.array([0., 0., 0., 1.], dtype=np.float32),
        }

    def test_roundtrip_preserves_capture_time_and_model_provenance(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "body.h5"
            save_body_recording(path, [self._sample()], {
                "source": "smplerx", "model_id": "s32", "model_hash": "abc123",
                "time_unit": "monotonic_s", "calibration_version": "arm-v2",
            })
            recording = load_body_recording(path)
        self.assertEqual(recording.metadata["model_hash"], "abc123")
        self.assertEqual(recording.samples[0]["sample_id"], "camera:12")
        self.assertAlmostEqual(recording.samples[0]["capture_time_s"], 1.25)
        np.testing.assert_allclose(recording.samples[0]["joints_root_m"], self._sample()["joints_root_m"])

    def test_legacy_arm_recording_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "legacy.h5"
            with h5py.File(path, "w") as handle:
                handle.attrs["schema_version"] = "hand_body_capture_h5_v1"
                handle.attrs["arm_world_space"] = "mediapipe_pose_world_meters"
            with self.assertRaisesRegex(ValueError, "SMPL-X"):
                load_body_recording(path)

    def test_missing_model_hash_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(ValueError, "model_hash"):
                save_body_recording(Path(directory) / "body.h5", [self._sample()], {
                    "source": "smplerx", "model_id": "s32", "time_unit": "monotonic_s",
                })


if __name__ == "__main__":
    unittest.main()
