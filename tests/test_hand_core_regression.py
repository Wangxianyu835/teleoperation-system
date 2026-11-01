import tempfile
import unittest
from pathlib import Path

import h5py
import numpy as np
import torch

from teleoperation.contracts.constants import HAND_KEYPOINTS
from teleoperation.retargeting.hand.topology import mediapipe21_to_hand25, wrist_relative
from tests.fixtures.hands import (
    continuous_three_frames,
    fixed_mediapipe_hand,
    left_hand,
    missing_left,
    missing_right,
    nan_frame,
    right_hand,
    zero_frame,
)


class SyntheticFixtureTests(unittest.TestCase):
    def test_fixture_set_contains_both_sides_invalid_frames_and_missing_sides(self):
        self.assertEqual(left_hand().shape, (21, 3))
        self.assertEqual(right_hand().shape, (21, 3))
        self.assertEqual(continuous_three_frames().shape, (3, 21, 3))
        self.assertTrue(np.isnan(nan_frame()).any())
        np.testing.assert_array_equal(zero_frame(), np.zeros((21, 3), dtype=np.float32))
        self.assertIsNone(missing_left())
        self.assertIsNone(missing_right())


class MediaPipeConversionRegressionTests(unittest.TestCase):
    def test_mediapipe21_to_hand25_has_expected_shape_roots_and_wrist(self):
        points = fixed_mediapipe_hand()
        converted = mediapipe21_to_hand25(points)

        self.assertEqual(converted.shape, (HAND_KEYPOINTS, 3))
        # The function performs wrist-relative normalization after copying MP0.
        np.testing.assert_array_equal(converted[0], np.zeros(3, dtype=np.float32))
        for output_index, mediapipe_mcp in ((5, 5), (10, 9), (15, 13), (20, 17)):
            expected = 0.5 * (points[mediapipe_mcp] - points[0])
            np.testing.assert_allclose(converted[output_index], expected, atol=1e-6)

        np.testing.assert_allclose(converted[6], points[5] - points[0], atol=1e-6)
        np.testing.assert_allclose(converted[11], points[9] - points[0], atol=1e-6)
        np.testing.assert_allclose(converted[16], points[13] - points[0], atol=1e-6)
        np.testing.assert_allclose(converted[21], points[17] - points[0], atol=1e-6)

    def test_wrist_relative_zeroes_wrist_and_is_translation_invariant(self):
        points = fixed_mediapipe_hand()[:7]
        translated = points + np.asarray([2.0, -3.0, 4.0], dtype=np.float32)

        relative = wrist_relative(points)
        translated_relative = wrist_relative(translated)

        np.testing.assert_array_equal(relative[0], np.zeros(3, dtype=np.float32))
        np.testing.assert_allclose(relative, translated_relative, atol=1e-6)


class InvalidOfflineRegressionTests(unittest.TestCase):
    def test_invalid_result_is_not_relabelled_as_valid_when_command_is_held(self):
        from teleoperation.contracts.hand_control import HandJointResult
        from teleoperation.retargeting.hand.joint_commands import JointCommandPipeline
        from teleoperation.robots.hand_spec import load_hand_spec
        spec=load_hand_spec('g1','right')
        pipeline=JointCommandPipeline({spec.key:spec},filter_mode='none')
        invalid=HandJointResult(*spec.key,spec.spec_hash,spec.independent_joint_names,None,0.,False)
        first=pipeline.update(invalid,timestamp=0.)
        np.testing.assert_array_equal(first.positions,spec.q_reference)
        valid=HandJointResult(*spec.key,spec.spec_hash,spec.independent_joint_names,spec.bounds.mean(1),1/30)
        second=pipeline.update(valid,timestamp=1/30)
        third=pipeline.update(invalid,timestamp=2/30)
        self.assertFalse(invalid.valid)
        self.assertIsNone(invalid.positions)
        np.testing.assert_array_equal(second.positions,third.positions)


def write_native_recording(path, valid=(False, True, False), hand_model='g1', side='right'):
    """Small real recording fixture shared by inspection and replay regressions."""
    from teleoperation.contracts.hand_control import HandJointResult
    from teleoperation.data.hand_joint_recording import HandJointResultWriter
    from teleoperation.retargeting.hand.joint_commands import JointCommandPipeline
    from teleoperation.robots.hand_spec import load_hand_spec
    spec = load_hand_spec(hand_model, side)
    pipeline = JointCommandPipeline({spec.key: spec}, filter_mode='none')
    with HandJointResultWriter(path, {spec.key: spec}, {}) as writer:
        for index, is_valid in enumerate(valid):
            timestamp = index / 30.
            result = HandJointResult(*spec.key, spec.spec_hash, spec.independent_joint_names,
                                     spec.bounds.mean(1) if is_valid else None,
                                     timestamp, is_valid)
            command = pipeline.update(result, timestamp=timestamp)
            writer.append(index, timestamp, {spec.key: result}, {spec.key: command})
    return spec


class NativeRecordingRegressionTests(unittest.TestCase):
    def test_invalid_results_stay_invalid_while_command_holds(self):
        from teleoperation.data.hand_joint_recording import iter_joint_commands
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'joints.h5'
            spec = write_native_recording(path)
            frames = list(iter_joint_commands(path, {spec.key: spec}))
            self.assertEqual([timestamp for timestamp, _ in frames], [0., 1/30., 2/30.])
            np.testing.assert_array_equal(frames[1][1][spec.key].positions, frames[2][1][spec.key].positions)
            with h5py.File(path) as stream:
                group = stream['g1/right']
                np.testing.assert_array_equal(group['valid'][:], [False, True, False])
                self.assertTrue(np.isnan(group['raw_positions'][0]).all())
                self.assertTrue(np.isnan(group['raw_positions'][2]).all())
                self.assertTrue(np.isfinite(group['command_positions'][:]).all())

    def test_old_angle_file_is_rejected_by_native_replay(self):
        from teleoperation.data.hand_joint_recording import iter_joint_commands
        from teleoperation.robots.hand_spec import load_hand_spec
        spec = load_hand_spec('l21', 'right')
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'old.h5'
            with h5py.File(path, 'w') as stream:
                stream['timestamps'] = [0.]
                stream['right_angles'] = np.zeros((1,18))
            with self.assertRaisesRegex(ValueError, 'hand joint command recording'):
                list(iter_joint_commands(path, {spec.key: spec}))

    def test_shared_model_batch_and_streaming_use_identical_native_signs(self):
        from teleoperation.contracts.coordinates import PALM_LENGTH_PROFILE
        from teleoperation.applications.hand_processing import MediaPipePalmLocalPipeline
        from teleoperation.retargeting.hand.interface import SharedJointNNRetargeter
        from teleoperation.retargeting.hand.predictor import build_joint_model
        from teleoperation.retargeting.hand.angles import canonical_to_native_target_angles
        from teleoperation.robots.hand_spec import load_hand_spec
        from tests.test_joint_checkpoints import SMALL
        from tests.test_world_hand_pipeline import world_frame
        torch.manual_seed(7)
        specs = {('l21',side):load_hand_spec('l21',side) for side in ('left','right')}
        canonical = specs[('l21','right')]
        model = build_joint_model(canonical, SMALL).eval()
        retargeter = SharedJointNNRetargeter(specs, model)
        pipeline = MediaPipePalmLocalPipeline('world', input_profile=PALM_LENGTH_PROFILE)
        windows = []
        for index in range(7):
            window = pipeline.process_window(world_frame(index))
            if window is not None:
                windows.append(window)
        self.assertEqual(len(windows), 5)
        for key,spec in specs.items():
            batch = torch.as_tensor(np.stack([window.hands[spec.side] for window in windows]))
            with torch.no_grad():
                expected = canonical_to_native_target_angles(model(batch),canonical,spec).numpy()
            for index, window in enumerate(windows):
                result = retargeter.retarget(window)[key]
                self.assertTrue(result.valid)
                self.assertEqual(result.positions.shape,(17,))
                np.testing.assert_allclose(result.positions,expected[index],atol=1e-6,rtol=1e-5)


if __name__ == '__main__':
    unittest.main()
