"""World coordinates, chronological windows and capture recovery without hardware."""

import unittest
from unittest.mock import Mock, patch

import numpy as np

from teleoperation.contracts.observations import RawHandFrame
from teleoperation.applications.hand_realtime import (
    MediaPipeCameraWorkflow,
    MediaPipePalmLocalPipeline,
)


SIDES = ("left", "right")


def synthetic_hand_pair():
    left = np.zeros((21, 3), dtype=np.float32)
    bases = np.array([[.018, .008, .010], [.015, .030, 0], [0, .035, 0],
                      [-.012, .030, 0], [-.022, .023, .005]], dtype=np.float32)
    for finger, base in enumerate(bases):
        left[1 + 4 * finger:5 + 4 * finger] = np.array([1, 1.35, 1.65, 2])[:, None] * base
    right = left * np.array([-1, 1, 1], dtype=np.float32)
    return left + np.array([-.2, .4, .1]), right + np.array([.2, .4, .1])


def raw_frame(index, left=True, right=True):
    pair = synthetic_hand_pair()
    frame = {"timestamp": index / 30.0, "metadata": {"frame_index": index}}
    for side, raw, present in zip(SIDES, pair, (left, right)):
        points = raw.astype(np.float32)
        # Change finger shape as well as camera pose, so old/new frames differ.
        points[[4, 8, 12, 16, 20], 2] += index * 0.002
        angle = index * 0.04
        rotation = np.array([[np.cos(angle), -np.sin(angle), 0],
                             [np.sin(angle), np.cos(angle), 0], [0, 0, 1]], dtype=np.float32)
        points = points @ rotation.T + np.array([0.003 * index, 0.0, 0.0], dtype=np.float32)
        frame[side] = points if present else None
    return frame


def world_raw_frame(index, left=True, right=True):
    frame = raw_frame(index,left,right)
    for side in SIDES:
        frame['world_'+side] = None if frame[side] is None else frame[side].copy()
    frame['metadata'].update(world_landmark_space='mediapipe_world_meters',world_length_unit='m')
    return frame


def offline_reference(sequence, bases=None):
    """Independent chronological lists using the world numerical transform."""
    from teleoperation.retargeting.hand.coordinates import align_mediapipe_hand
    histories = {side:[] for side in SIDES}
    outputs = []
    for frame in sequence:
        current, windows = {}, {}
        for side in SIDES:
            raw = frame['world_'+side]
            aligned = None
            try:
                array = np.asarray(raw,dtype=np.float32)
                if array.shape == (21,3) and np.isfinite(array).all():
                    candidate = align_mediapipe_hand(array,side=side)
                    if np.isfinite(candidate).all() and np.any(candidate):
                        aligned = candidate
            except ValueError:
                pass
            current[side] = aligned
            histories[side] = [] if aligned is None else (histories[side]+[aligned])[-3:]
            windows[side] = np.stack(histories[side]) if len(histories[side])==3 else None
        outputs.append((current,windows))
    return outputs


class MediaPipeRealtimeTests(unittest.TestCase):
    def test_world_frame_coordinates_and_windows_match_independent_reference(self):
        for bad in ('missing','zero','degenerate','nan','inf'):
            frames=[world_raw_frame(i) for i in range(9)]
            values={'missing':None,'zero':np.zeros((21,3)),
                    'degenerate':np.arange(63).reshape(21,3),
                    'nan':np.full((21,3),np.nan),'inf':np.full((21,3),np.inf)}
            frames[3]['world_left']=values[bad]
            pipeline=MediaPipePalmLocalPipeline('world')
            for index,(frame,(current,windows)) in enumerate(zip(frames,offline_reference(frames))):
                with self.subTest(bad=bad,index=index):
                    window=pipeline.process_window(frame)
                    for side in SIDES:
                        if current[side] is None:
                            self.assertIsNone(pipeline.current_hands[side])
                            self.assertEqual(pipeline.valid_streak[side],0)
                        else:
                            np.testing.assert_array_equal(pipeline.current_hands[side],current[side])
                        actual=None if window is None else window.hands[side]
                        if windows[side] is None:
                            self.assertIsNone(actual)
                        else:
                            np.testing.assert_array_equal(actual,windows[side])
                    if index in (3,4,5):
                        self.assertIsNone(window.hands['left'])
                        self.assertIsNotNone(window.hands['right'])
                    if index==6:
                        self.assertIsNotNone(window.hands['left'])

    def test_palm_length_profile_matches_independent_median_and_recovers(self):
        from teleoperation.contracts.coordinates import PALM_LENGTH_PROFILE, palm_length_metadata
        frames=[world_raw_frame(i) for i in range(9)]
        frames[3]['world_left']=None
        # Vary world size to detect per-frame normalization instead of the causal median.
        for index,frame in enumerate(frames):
            for side in SIDES:
                if frame['world_'+side] is not None:
                    frame['world_'+side] *= 1.+index*.15
        pipeline=MediaPipePalmLocalPipeline('world',input_profile=PALM_LENGTH_PROFILE)
        for index,(frame,(_,expected)) in enumerate(zip(frames,offline_reference(frames))):
            actual=pipeline.process_window(frame)
            if actual is None:
                self.assertLess(index,2)
                continue
            for name,value in palm_length_metadata().items():
                self.assertEqual(actual.metadata[name],value)
            for side in SIDES:
                metric=expected[side]
                if metric is None:
                    self.assertIsNone(actual.hands[side])
                    continue
                metric=metric.astype(np.float64)
                metric-=metric[:,0:1]
                median=float(np.median(np.linalg.norm(metric[:,11],axis=1)))
                np.testing.assert_array_equal(actual.hands[side],(metric/median).astype(np.float32))
                self.assertAlmostEqual(actual.metadata['palm_length_m'][side],median)
            if index in (3,4,5):
                self.assertIsNone(actual.hands['left'])
                self.assertIsNotNone(actual.hands['right'])
            if index==6:
                self.assertIsNotNone(actual.hands['left'])

    def test_reset_discards_all_history_and_requires_three_new_frames(self):
        pipeline=MediaPipePalmLocalPipeline('world')
        for index in range(3): pipeline.process_window(world_raw_frame(index))
        pipeline.reset()
        self.assertEqual(pipeline.valid_streak,dict.fromkeys(SIDES,0))
        self.assertIsNone(pipeline.process_window(world_raw_frame(4)))
        self.assertIsNone(pipeline.process_window(world_raw_frame(5)))
        actual=pipeline.process_window(world_raw_frame(6))
        expected=offline_reference([world_raw_frame(i) for i in (4,5,6)])[-1][1]
        for side in SIDES:
            np.testing.assert_array_equal(actual.hands[side],expected[side])

    def test_world_alignment_precedes_buffer_and_never_loads_robot_basis(self):
        with patch('teleoperation.retargeting.hand.processing.build_l21_reference_basis',side_effect=AssertionError('robot basis')):
            pipeline=MediaPipePalmLocalPipeline('world')
            append=pipeline.buffer.append
            def observe(frame):
                for side in SIDES:
                    np.testing.assert_array_equal(frame.hands[side],pipeline.current_hands[side])
                return append(frame)
            with patch.object(pipeline.buffer,'append',side_effect=observe) as buffered:
                for index in range(3): pipeline.process_window(world_raw_frame(index))
                self.assertEqual(buffered.call_count,3)

    def test_capture_error_clears_history_and_releases_camera(self):
        capture=Mock()
        frames=[world_raw_frame(i) for i in range(3)]
        capture.next_observation.side_effect=[RawHandFrame(
            {s:f[s] for s in SIDES},f['timestamp'],'mediapipe_approx',f['metadata'],
            world_hands={s:f['world_'+s] for s in SIDES}) for f in frames]+[RuntimeError('capture failed')]
        with patch('teleoperation.applications.hand_realtime.MediaPipeCameraInput',return_value=capture):
            workflow=MediaPipeCameraWorkflow('model.task',landmark_space='world')
            for _ in frames: workflow.next_window()
            with self.assertRaisesRegex(RuntimeError,'capture failed'): workflow.next_window()
        self.assertEqual(workflow.processor.valid_streak,dict.fromkeys(SIDES,0))
        self.assertIsNone(workflow.last_raw_frame)
        capture.close.assert_called_once()

    def test_world_workflow_preserves_raw_timestamp_and_uses_relative_output(self):
        capture=Mock()
        frames=[world_raw_frame(i) for i in range(3)]
        for index,frame in enumerate(frames):
            frame['timestamp']=100000+index*33
            frame['metadata']['timestamp_unit']='unix_ms'
        capture.next_observation.side_effect=[RawHandFrame(
            {s:f[s] for s in SIDES},f['timestamp'],'mediapipe_approx',f['metadata'],
            world_hands={s:f['world_'+s] for s in SIDES}) for f in frames]
        with patch('teleoperation.applications.hand_realtime.MediaPipeCameraInput',return_value=capture), patch(
                'teleoperation.applications.hand_realtime.time.time',side_effect=[100,100.1,100.2,100.3]):
            with MediaPipeCameraWorkflow('model.task',landmark_space='world') as workflow:
                self.assertIsNone(workflow.next_window())
                self.assertIsNone(workflow.next_window())
                window=workflow.next_window()
                self.assertEqual(window.timestamp,.3)
                self.assertEqual(window.metadata['raw_timestamp_ms'],100066)
                self.assertEqual(workflow.last_raw_frame['timestamp'],100066)
                for call in capture.next_observation.call_args_list:
                    self.assertEqual(call.kwargs,{'include_world':True})
        capture.close.assert_called_once()


if __name__ == '__main__':
    unittest.main()
