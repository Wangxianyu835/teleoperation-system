"""Both methods share one normalized target and independent coordinate result."""
import importlib.util
import unittest

import numpy as np
import torch

from teleoperation.applications.hand_processing import MediaPipePalmLocalPipeline
from teleoperation.contracts.hand import HandWindow
from teleoperation.retargeting.hand.interface import JointNNRetargeter, SharedJointNNRetargeter
from teleoperation.retargeting.hand.predictor import build_joint_model
from teleoperation.robots.hand_spec import load_hand_spec
from tests.test_joint_checkpoints import SMALL
from tests.test_world_hand_pipeline import world_frame


class NormalizedRetargeterTests(unittest.TestCase):
    def window(self):
        pipeline = MediaPipePalmLocalPipeline('world', input_profile='camera_palm_length_window3_v1')
        for i in range(3):
            window = pipeline.process_window(world_frame(i))
        return window

    def test_native_nn_shapes_sides_and_missing_input(self):
        torch.set_num_threads(1)
        specs = {('l21',s):load_hand_spec('l21',s) for s in ('left','right')}
        nn = JointNNRetargeter(specs,{k:build_joint_model(s,SMALL) for k,s in specs.items()})
        window = self.window()
        results = nn.retarget(window)
        for key,spec in specs.items():
            result = results[key]
            self.assertTrue(result.valid)
            self.assertEqual(result.joint_names,spec.independent_joint_names)
            self.assertEqual(result.positions.shape,(17,))
            spec.expand(result.positions)
        missing = HandWindow({'left':None,'right':window.hands['right']},window.timestamp,window.source,window.metadata)
        self.assertFalse(nn.retarget(missing)[('l21','left')].valid)

    def test_shared_nn_converts_one_canonical_output_to_both_native_sides(self):
        class FixedModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.register_buffer('values', torch.tensor(
                    [[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9,
                      1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7]],
                    dtype=torch.float32,
                ))

            def forward(self, inputs):
                return self.values.expand(inputs.shape[0], -1)

        torch.set_num_threads(1)
        specs = {('l21', side): load_hand_spec('l21', side) for side in ('left', 'right')}
        nn = SharedJointNNRetargeter(specs, FixedModel())
        results = nn.retarget(self.window())
        np.testing.assert_allclose(results[('l21', 'right')].positions, np.arange(1, 18) / 10)
        expected_left = np.arange(1, 18, dtype=np.float32) / 10
        expected_left[[0, 3, 6, 9]] *= -1
        np.testing.assert_allclose(results[('l21', 'left')].positions, expected_left)

    def test_shared_nn_adapts_registered_native_hand_side_limits(self):
        class FixedModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.register_buffer('values', torch.full((1, 7), 0.1))

            def forward(self, inputs):
                return self.values.expand(inputs.shape[0], -1)

        specs = {('g1', side): load_hand_spec('g1', side) for side in ('left', 'right')}
        results = SharedJointNNRetargeter(specs, FixedModel()).retarget(self.window())
        expected_left = np.full(7, 0.1, dtype=np.float32)
        expected_left[[1, 2, 4, 5, 6]] *= -1
        np.testing.assert_allclose(results[('g1', 'left')].positions, expected_left)
        np.testing.assert_allclose(results[('g1', 'right')].positions, 0.1)

    @unittest.skipUnless(importlib.util.find_spec('dex_retargeting'), 'requires dex environment')
    def test_dex_all_registered_targets_preserve_identity_and_scale_equivalence(self):
        from teleoperation.retargeting.hand.dex_native import NormalizedDexRetargeter
        from teleoperation.retargeting.hand.hand_model import build_hand_targets
        for name in ('l21','h1_2','gr1_t2','g1'):
            dex = NormalizedDexRetargeter([name],('left','right'))
            window = self.window()
            results = dex.retarget(window)
            for key,spec in dex.specs.items():
                self.assertTrue(results[key].valid, results[key].error)
                self.assertEqual(results[key].positions.shape,(spec.dof,))
                spec.expand(results[key].positions)
                solver = dex.solvers[key]
                self.assertAlmostEqual(solver.sequence.optimizer.huber_loss.beta,.02*spec.reference_palm_length_m)
                targets = build_hand_targets(window,spec)
                np.testing.assert_allclose(targets.positions_native_m/spec.reference_palm_length_m,targets.vectors)
            missing = HandWindow({'left':None,'right':None},window.timestamp,window.source,window.metadata)
            self.assertFalse(dex.retarget(missing)[(name,'left')].valid)


if __name__ == '__main__': unittest.main()
