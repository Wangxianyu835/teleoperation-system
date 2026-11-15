"""Device choice must not select a different hand retargeting algorithm."""
from pathlib import Path
import tempfile
import unittest
import importlib.util

import h5py
import numpy as np
import torch

from teleoperation.contracts.observations import RawHandFrame
from teleoperation.applications import hand_processing
from teleoperation.applications.hand_processing import MediaPipePalmLocalPipeline
from teleoperation.cli import build_parser, main
from teleoperation.robots.hand_spec import load_hand_spec
from teleoperation.retargeting.hand.topology import expand_hand21_to25
from tests.test_world_hand_pipeline import world_frame, observation
from tests.test_joint_checkpoints import SMALL


def paired_observations(index, missing=()):
    camera = observation(world_frame(index))
    transforms = {}
    for side in ('left', 'right'):
        # Invert the documented SDK -> anatomical metric axes, independently
        # of the production adapter. Preserve all bent-finger z coordinates.
        axes = (np.array([[0., -1., 0.], [0., 0., -1.], [1., 0., 0.]])
                if side == 'left' else
                np.array([[0., -1., 0.], [0., 0., 1.], [-1., 0., 0.]]))
        matrices = np.repeat(np.eye(4)[None], 25, axis=0)
        matrices[:, :3, 3] = expand_hand21_to25(camera.world_hands[side]) @ axes
        transforms[side] = None if side in missing else matrices
        if side in missing:
            camera.hands[side] = None
            camera.world_hands[side] = None
    return camera, RawHandFrame(transforms, camera.timestamp, 'visionpro',
                               geometry_encoding={s: 'transforms25' for s in transforms})


class UnifiedHandInputTests(unittest.TestCase):
    def pipeline(self):
        factory = getattr(hand_processing, 'UnifiedHandPipeline', None)
        self.assertIsNotNone(factory, 'Both inputs need the same public hand pipeline')
        return factory()

    def test_both_devices_produce_same_contract_values_and_camera_numerics(self):
        rgb, vr = self.pipeline(), self.pipeline()
        original = MediaPipePalmLocalPipeline('world', input_profile='camera_palm_length_window3_v1')
        for i in range(5):
            camera, headset = paired_observations(i)
            a, b = rgb.process_window(camera), vr.process_window(headset)
            old = original.process_window(camera)
            if i < 2:
                self.assertIsNone(a)
                self.assertIsNone(b)
                continue
            self.assertEqual(a.metadata['input_profile'], 'hand_palm_length_window3_v1')
            self.assertEqual(a.metadata['input_profile'], b.metadata['input_profile'])
            self.assertEqual(b.source, 'visionpro')
            self.assertEqual(b.metadata['source_landmark_space'], 'canonical_hand_meters')
            for side in ('left', 'right'):
                np.testing.assert_allclose(a.hands[side], b.hands[side], atol=2e-6)
                np.testing.assert_allclose(a.hands[side], old.hands[side], atol=0, rtol=0)

    def test_missing_side_needs_three_new_samples_and_other_side_continues(self):
        pipeline = self.pipeline()
        for i in range(7):
            _, raw = paired_observations(i, ('left',) if i == 3 else ())
            window = pipeline.process_window(raw)
            if i >= 2:
                self.assertIsNotNone(window.hands['right'])
                self.assertEqual(window.hands['left'] is None, i in (3, 4, 5))

    def test_device_switch_and_invalid_input_clear_temporal_history(self):
        pipeline = self.pipeline()
        for i in range(3):
            pipeline.process_window(paired_observations(i)[0])
        self.assertIsNone(pipeline.process_window(paired_observations(3)[1]))
        self.assertIsNone(pipeline.process_window(paired_observations(4)[1]))
        self.assertIsNotNone(pipeline.process_window(paired_observations(5)[1]))
        raw = paired_observations(6)[1]
        raw.metadata['coordinate_alignment'] = 'already_processed'
        with self.assertRaises(ValueError):
            pipeline.process_window(raw)
        self.assertIsNone(pipeline.process_window(paired_observations(7)[1]))

    def test_same_shared_nn_and_command_pipeline_accept_both_devices(self):
        from teleoperation.retargeting.hand.predictor import build_joint_model
        from teleoperation.retargeting.hand.interface import SharedJointNNRetargeter
        from teleoperation.retargeting.hand.joint_commands import JointCommandPipeline
        from teleoperation.applications.hand_control import _process
        torch.set_num_threads(1)
        rgb, vr = self.pipeline(), self.pipeline()
        for i in range(3):
            a, b = paired_observations(i)
            a, b = rgb.process_window(a), vr.process_window(b)
        for name in ('l21', 'h1_2', 'gr1_t2', 'g1'):
            specs = {(name, s): load_hand_spec(name, s) for s in ('left', 'right')}
            right = specs[(name, 'right')]
            model = build_joint_model(right, SMALL)
            backend = SharedJointNNRetargeter(specs, model)
            ra, ca, _ = _process(a, specs, backend, JointCommandPipeline(specs))
            rb, cb, _ = _process(b, specs, backend, JointCommandPipeline(specs))
            for key in specs:
                self.assertTrue(ra[key].valid and rb[key].valid)
                np.testing.assert_allclose(ra[key].positions, rb[key].positions, atol=1e-6)
                np.testing.assert_allclose(ca[key].positions, cb[key].positions, atol=1e-6)

    def test_cli_device_and_backend_are_independent(self):
        for device in ('camera', 'visionpro'):
            for backend in ('nn', 'dex'):
                argv = ['hand', 'control', '--source', device, '--hand-model', 'g1', '--backend', backend]
                if device == 'visionpro':
                    argv += ['--input-h5', 'raw.h5']
                args = build_parser().parse_args(argv)
                self.assertEqual((args.source, args.backend), (device, backend))

    def test_vr_shortcut_uses_common_nn_by_default(self):
        args = build_parser().parse_args(['hand', 'vr', '--ip', 'headset'])
        self.assertEqual(args.backend, 'nn')
        self.assertEqual(args.source, 'visionpro')
        self.assertEqual(args.input_profile, 'hand_palm_length_window3_v1')

    def test_malformed_vr_side_does_not_poison_other_hand(self):
        pipeline = self.pipeline()
        for i in range(3):
            _, raw = paired_observations(i)
            raw.hands['left'][2, :3, 3] = np.nan
            window = pipeline.process_window(raw)
        self.assertIsNone(window.hands['left'])
        self.assertTrue(np.isfinite(window.hands['right']).all())

    def test_camera_workflow_uses_shared_pipeline_and_keeps_source(self):
        from unittest.mock import Mock, patch
        from teleoperation.applications.hand_realtime import MediaPipeCameraWorkflow
        from dataclasses import replace
        capture = Mock()
        capture.next_observation.side_effect = [
            replace(paired_observations(i)[0], source='mediapipe_hands') for i in range(3)]
        with patch('teleoperation.applications.hand_realtime.MediaPipeCameraInput', return_value=capture):
            with MediaPipeCameraWorkflow('asset.task', input_profile='hand_palm_length_window3_v1') as camera:
                for _ in range(3):
                    window = camera.next_window()
                self.assertEqual(window.source, 'mediapipe_hands')
                self.assertEqual(window.metadata['input_profile'], 'hand_palm_length_window3_v1')
                self.assertIsNotNone(camera.processor.current_hands['left'])
        capture.close.assert_called_once()

    def test_vr_wrong_units_and_topology_are_rejected_before_buffering(self):
        pipeline = self.pipeline()
        raw = paired_observations(0)[1]
        raw.metadata['units'] = 'mm'
        with self.assertRaises(ValueError):
            pipeline.process_window(raw)
        raw = paired_observations(1)[1]
        raw.geometry_encoding['left'] = 'landmarks25'
        with self.assertRaises(ValueError):
            pipeline.process_window(raw)

    @unittest.skipUnless(importlib.util.find_spec('dex_retargeting'), 'requires dex environment')
    def test_same_dex_solver_for_both_devices_all_targets(self):
        from teleoperation.retargeting.hand.dex_native import NormalizedDexRetargeter
        from teleoperation.retargeting.hand.joint_commands import JointCommandPipeline
        from teleoperation.applications.hand_control import _process
        rgb, vr = self.pipeline(), self.pipeline()
        for i in range(3):
            a, b = paired_observations(i)
            a, b = rgb.process_window(a), vr.process_window(b)
        for name in ('l21', 'h1_2', 'gr1_t2', 'g1'):
            first = NormalizedDexRetargeter([name], ('left', 'right'))
            second = NormalizedDexRetargeter([name], ('left', 'right'))
            specs = first.specs
            ra, ca, _ = _process(a, specs, first, JointCommandPipeline(specs))
            rb, cb, _ = _process(b, specs, second, JointCommandPipeline(specs))
            for key in specs:
                self.assertTrue(ra[key].valid, ra[key].error)
                self.assertTrue(rb[key].valid, rb[key].error)
                np.testing.assert_allclose(ra[key].positions, rb[key].positions, atol=1e-5)
                np.testing.assert_allclose(ca[key].positions, cb[key].positions, atol=1e-5)

    def test_vr_h5_uses_existing_shared_nn_and_generic_results(self):
        from teleoperation.retargeting.hand.predictor import build_joint_model
        from teleoperation.retargeting.hand.checkpoint import shared_joint_checkpoint_payload, save_joint_checkpoint
        from teleoperation.data.hand_joint_recording import iter_joint_commands
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source, weights, result = root/'raw.h5', root/'nn.pth', root/'result.h5'
            with h5py.File(source, 'w') as stream:
                stream.attrs['source'] = 'visionpro'
                stream['timestamps'] = np.arange(7) / 30
                for side in ('left', 'right'):
                    stream[side+'_hand_transforms'] = np.stack([
                        paired_observations(i)[1].hands[side] for i in range(7)])
                    stream[side+'_valid'] = np.array([True, True, True, side != 'left', True, True, True])
            spec = load_hand_spec('l21', 'right')
            save_joint_checkpoint(weights, shared_joint_checkpoint_payload(build_joint_model(spec, SMALL), spec))
            self.assertEqual(main(['hand', 'control', '--source', 'visionpro', '--input-h5', str(source),
                '--hand-model', 'l21', '--backend', 'nn', '--hand', 'both', '--checkpoint', str(weights),
                '--device', 'cpu', '--headless', '--no-collision-guard', '--record-results', str(result)]), 0)
            specs = {('l21', side): load_hand_spec('l21', side) for side in ('left', 'right')}
            rows = list(iter_joint_commands(result, specs))
            self.assertEqual(len(rows), 7)
            with h5py.File(result) as stream:
                np.testing.assert_array_equal(stream['l21/left/valid'][:], [0, 0, 1, 0, 0, 0, 1])
                np.testing.assert_array_equal(stream['l21/right/valid'][:], [0, 0, 1, 1, 1, 1, 1])
            shortcut = root / 'shortcut.h5'
            self.assertEqual(main(['hand', 'vr', '--backend', 'nn', '--input-h5', str(source),
                '--hand-model', 'l21', '--hand', 'both', '--checkpoint', str(weights), '--fps', '0',
                '--headless', '--no-collision-guard', '--output-h5', str(shortcut)]), 0)
            with h5py.File(result) as direct, h5py.File(shortcut) as short:
                for side in ('left', 'right'):
                    np.testing.assert_array_equal(direct[f'l21/{side}/command_positions'][:],
                                                  short[f'l21/{side}/command_positions'][:])


if __name__ == '__main__':
    unittest.main()
