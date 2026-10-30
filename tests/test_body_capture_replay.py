"""The body/hand record must replay through the live tracking path."""
import tempfile
import unittest
from pathlib import Path
import numpy as np
from tests.test_smplerx_camera_source import FakeCamera, FakeHands, FakeClient, body_result
from teleoperation.applications.camera_teleop_source import CameraTeleopSource
from teleoperation.data.body_recording import save_body_recording, load_body_recording


class BodyCaptureReplayTests(unittest.TestCase):
    def test_capture_roundtrip_keeps_raw_body_hands_and_rgb(self):
        from teleoperation.data.body_recording import capture_sample, replay_observation
        camera = FakeCamera()
        source = CameraTeleopSource(camera, FakeHands(), FakeClient(body_result()), session_id='camera')
        raw = source.next_teleop_observation()
        row = capture_sample(source, raw)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'session.h5'
            save_body_recording(path, [row], {'source': 'smplerx', 'model_id': 's32',
                                 'model_hash': 'd' * 64, 'time_unit': 'monotonic_s'})
            restored = load_body_recording(path).samples[0]
        replayed = replay_observation(restored)
        np.testing.assert_array_equal(restored['rgb'], source.last_frame.rgb)
        np.testing.assert_array_equal(replayed.upper_body.world_arms['left'], raw.upper_body.world_arms['left'])
        self.assertEqual(replayed.wrists['left'].source, 'smplerx')
        self.assertEqual(replayed.hands.timestamp, raw.hands.timestamp)

    def test_recording_reader_rejects_forged_empty_model_hash(self):
        import h5py, json
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'session.h5'
            save_body_recording(path, [{'sample_id': 's:1', 'capture_time_s': 1.}],
                                {'source': 'smplerx', 'model_id': 's32', 'model_hash': 'h', 'time_unit': 'monotonic_s'})
            with h5py.File(path, 'r+') as handle:
                metadata = json.loads(handle.attrs['metadata_json'])
                metadata['model_hash'] = ''
                handle.attrs['metadata_json'] = json.dumps(metadata)
            with self.assertRaisesRegex(ValueError, 'model_hash'):
                load_body_recording(path)
