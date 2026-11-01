"""The body/hand record must replay through the live tracking path."""
import tempfile
import unittest
from pathlib import Path
import numpy as np
from tests.test_smplerx_camera_source import FakeCamera, FakeHands, FakeClient, body_result
from teleoperation.applications.camera_teleop_source import CameraTeleopSource
from teleoperation.data.body_recording import save_body_recording, load_body_recording


class BodyCaptureReplayTests(unittest.TestCase):
    def test_retired_nested_wrists_are_removed_from_capture_and_legacy_replay(self):
        from dataclasses import replace
        from types import SimpleNamespace
        from tests.smplerx_fixtures import observation
        from teleoperation.contracts.observations import CameraFrame
        from teleoperation.contracts.wrist import WristObservation
        from teleoperation.data.body_recording import capture_sample, replay_observation
        from teleoperation.inputs.smplerx import decode_packet, encode_packet
        raw, result = observation(0)
        retired = WristObservation(np.array([0., 0., 0., 1.]),
                                   'mediapipe_hand_world_axes', 'old_palm', 'camera_palm',
                                   raw.hands.timestamp, orientation_valid=True)
        raw = replace(raw, hands=replace(raw.hands, wrists={'left': retired}))
        source = SimpleNamespace(last_frame=CameraFrame(np.zeros((4, 5, 3), dtype=np.uint8),
                                                       raw.hands.timestamp, 0),
                                 session_id='camera', last_result=result)
        row = capture_sample(source, raw)
        packet = decode_packet(row['observation_packet'].tobytes())
        self.assertIsNone(packet['payload']['hands']['wrists'])
        self.assertEqual(raw.hands.wrists['left'].source, 'camera_palm')
        packet['payload']['hands']['wrists'] = {'left': {**vars(retired),
                                                       'orientation_xyzw': np.array([np.nan])}}
        row['observation_packet'] = np.frombuffer(encode_packet('observation', packet['payload']), dtype=np.uint8)
        restored = replay_observation(row)
        self.assertIsNone(restored.hands.wrists)
        for side in ('left', 'right'):
            self.assertEqual(restored.wrists[side].source, 'smplerx')
            np.testing.assert_array_equal(restored.wrists[side].orientation_xyzw,
                                          raw.wrists[side].orientation_xyzw)
            np.testing.assert_array_equal(restored.hands.world_hands[side],
                                          raw.hands.world_hands[side])

    def test_live_and_replay_keep_identical_irregular_capture_intervals(self):
        from dataclasses import replace
        from unittest.mock import Mock
        from tests.smplerx_fixtures import observation
        from teleoperation.data.body_recording import capture_sample
        from teleoperation.simulation.realtime_teleop import CameraFrameSource, BodyReplayFrameSource
        from teleoperation.applications.hand_processing import MediaPipePalmLocalPipeline
        from teleoperation.contracts.observations import CameraFrame
        source = Mock()
        live = CameraFrameSource.__new__(CameraFrameSource)
        live.pipeline = MediaPipePalmLocalPipeline('world')
        live._source, live._start_timestamp = source, None
        live._last_capture_time_s = live._capture_dt = None
        live.body_enabled = True
        rows, intervals = [], []
        for index,stamp in enumerate((10.,10.05,10.22,10.25)):
            raw,result = observation(index)
            raw = replace(raw,hands=replace(raw.hands,timestamp=stamp))
            source.next_teleop_observation.return_value = raw
            source.last_frame = CameraFrame(np.zeros((4,5,3),dtype=np.uint8),stamp,index)
            source.session_id,source.last_result = 'camera',result
            live.next_frame()
            rows.append(capture_sample(source,raw))
            intervals.append(live.frame_dt)
        with tempfile.TemporaryDirectory() as directory:
            from tests.smplerx_fixtures import METADATA
            path = save_body_recording(Path(directory)/'irregular.h5',rows,METADATA)
            replay = BodyReplayFrameSource(path,'world')
            restored = []
            for _ in rows:
                replay.next_frame()
                restored.append(replay.frame_dt)
        self.assertEqual(intervals,restored)
        np.testing.assert_allclose(intervals[1:],[.05,.17,.03])

    def test_body_replay_with_absent_hands_does_not_require_metric_hand_geometry(self):
        from teleoperation.data.body_recording import capture_sample
        from teleoperation.simulation.realtime_teleop import BodyReplayFrameSource
        source = CameraTeleopSource(FakeCamera(),FakeHands(),FakeClient(body_result()),session_id='camera')
        raw = source.next_teleop_observation()
        with tempfile.TemporaryDirectory() as directory:
            from tests.smplerx_fixtures import METADATA
            path = save_body_recording(Path(directory)/'no-hands.h5',[capture_sample(source,raw)],METADATA)
            replay = BodyReplayFrameSource(path,'world')
            self.assertIsNone(replay.next_frame())
            self.assertTrue(replay.arms().valid['left'])

    def test_combined_file_feeds_hand_reader_and_body_frame_source(self):
        from teleoperation.data.body_recording import capture_sample
        from teleoperation.data.hand_h5 import load_twohand_h5
        from teleoperation.simulation.realtime_teleop import BodyReplayFrameSource
        camera = FakeCamera()
        source = CameraTeleopSource(camera, FakeHands(), FakeClient(body_result()), session_id='camera')
        raw = source.next_teleop_observation()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'session.h5'
            save_body_recording(path, [capture_sample(source, raw)],
                                {'source': 'smplerx', 'model_id': 's32', 'model_hash': 'h', 'time_unit': 'monotonic_s'})
            ids, times, hands = load_twohand_h5(path, require_aligned=False)
            replay = BodyReplayFrameSource(path, 'normalized')
            replay.next_frame()
            np.testing.assert_array_equal(replay.arms().world_arms['left'], raw.upper_body.world_arms['left'])
            self.assertEqual(replay.wrists()['left'].source, 'smplerx')
            replay.next_frame()
            self.assertTrue(replay.exhausted)
        self.assertEqual(ids.tolist(), [0])
        self.assertEqual(times.tolist(), [raw.hands.timestamp])
        self.assertTrue(np.isnan(hands['left']).all())

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
