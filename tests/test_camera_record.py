"""Standalone recorder delegates to current raw-world capture and alignment."""
import contextlib
import io
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

import h5py
import numpy as np
from teleoperation.applications import camera_record, offline_h5_record
from teleoperation.contracts.coordinates import WORLD_PALM_LOCAL_COORDINATE_ALIGNMENT
from teleoperation.learning.dataset import TwoHandH5Dataset
from tests.test_world_hand_pipeline import world_frame


class CameraRecordTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)
        self.raw = self.root / 'raw.h5'
        self.aligned = self.root / 'aligned.h5'

    def options(self, *extra):
        return ['--output-h5', str(self.raw), '--aligned-output', str(self.aligned),
                '--model-asset-path', 'hand.task', '--no-preview', *extra]

    def test_adapter_forwards_limits_and_explicit_world_input(self):
        args = camera_record.build_parser().parse_args(self.options('--frames','4','--seconds','2'))
        with patch.object(offline_h5_record, 'run', return_value=7) as record:
            self.assertEqual(camera_record.run(args),7)
        forwarded = record.call_args.args[0]
        self.assertEqual(forwarded.output,self.raw)
        self.assertEqual(forwarded.aligned_output,self.aligned)
        self.assertEqual(forwarded.landmark_space,'world')
        self.assertEqual(forwarded.frames,4)
        self.assertEqual(forwarded.seconds,2.)
        self.assertTrue(forwarded.no_preview)

    def test_records_world_and_preview_arrays_then_aligns(self):
        frames = [world_frame(i) for i in range(5)]
        capture = Mock()
        capture.next_frame.side_effect = frames
        with patch.object(offline_h5_record,'MediaPipeCameraInput',return_value=capture), contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(camera_record.main(self.options('--frames','5','--seconds','0')),0)
        capture.close.assert_called_once()
        self.assertEqual(capture.next_frame.call_count,5)
        for call in capture.next_frame.call_args_list:
            self.assertEqual(call.kwargs,{'include_world':True})
        with h5py.File(self.raw) as stream:
            self.assertEqual(stream.attrs['recording_status'],'frame_limit')
            for side in ('left','right'):
                np.testing.assert_array_equal(stream[side+'_world_landmarks'][:],np.stack([f['world_'+side] for f in frames]))
                np.testing.assert_array_equal(stream[side+'_hand_keypoints'][:],np.stack([f[side] for f in frames]))
        dataset = TwoHandH5Dataset(self.aligned)
        self.assertEqual(dataset.coordinate_alignment,WORLD_PALM_LOCAL_COORDINATE_ALIGNMENT)
        self.assertEqual(dataset.side_counts(),{'left':3,'right':3})

    def test_duration_limit_stops_capture_and_aligns_complete_frames(self):
        capture = Mock()
        capture.next_frame.side_effect = [world_frame(i) for i in range(5)]
        with patch.object(offline_h5_record,'MediaPipeCameraInput',return_value=capture), patch.object(
                offline_h5_record.time,'monotonic',side_effect=[0.,0.,.1,1.]), contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(camera_record.main(self.options('--frames','0','--seconds','.5')),0)
        capture.close.assert_called_once()
        self.assertEqual(capture.next_frame.call_count,2)
        with h5py.File(self.raw) as stream:
            self.assertEqual(stream.attrs['recording_status'],'duration_limit')
            self.assertEqual(stream['timestamps'].shape,(2,))
        self.assertTrue(self.aligned.exists())

    def test_capture_failure_keeps_complete_raw_rows_and_closes_camera(self):
        capture = Mock()
        capture.next_frame.side_effect = [world_frame(0),world_frame(1),RuntimeError('capture failed')]
        with patch.object(offline_h5_record,'MediaPipeCameraInput',return_value=capture), contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(camera_record.main(self.options('--frames','5','--seconds','0')),1)
        capture.close.assert_called_once()
        self.assertFalse(self.aligned.exists())
        with h5py.File(self.raw) as stream:
            self.assertEqual(stream.attrs['recording_status'],'error')
            self.assertTrue(all(value.shape[0]==2 for value in stream.values()))

    def test_existing_output_rejected_before_camera_open(self):
        self.raw.write_bytes(b'preserve me')
        with patch.object(offline_h5_record,'MediaPipeCameraInput') as capture:
            with self.assertRaises(FileExistsError):
                camera_record.main(self.options('--frames','1'))
        capture.assert_not_called()
        self.assertEqual(self.raw.read_bytes(),b'preserve me')

    def test_invalid_limits_fail_before_capture(self):
        for limits in (('--frames','-1'),('--seconds','nan'),('--seconds','-1'),('--frames','0','--seconds','0')):
            with self.subTest(limits=limits), patch.object(offline_h5_record,'MediaPipeCameraInput') as capture, contextlib.redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit) as error:
                    camera_record.main(self.options(*limits))
                self.assertEqual(error.exception.code,2)
                capture.assert_not_called()

    def test_retired_angle_output_option_is_rejected(self):
        with contextlib.redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit) as error:
                camera_record.build_parser().parse_args(['--angles-h5','old.h5'])
        self.assertEqual(error.exception.code,2)


if __name__ == '__main__':
    unittest.main()
