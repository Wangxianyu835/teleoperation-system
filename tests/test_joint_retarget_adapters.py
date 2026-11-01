"""Native joint inference preserves the complete target output and validity."""
import unittest
import numpy as np
import torch
from teleoperation.contracts.hand import HandWindow
from teleoperation.contracts.coordinates import palm_length_metadata
from teleoperation.retargeting.hand.interface import JointNNRetargeter
from teleoperation.robots.hand_spec import load_hand_spec

class ArrayPredictor(torch.nn.Module):
    def __init__(self, values):
        super().__init__()
        self.register_buffer('values', torch.as_tensor(values, dtype=torch.float32))
        self.calls = 0
    def forward(self, points):
        self.calls += 1
        return self.values.expand(len(points), -1)

class JointRetargetAdapterTests(unittest.TestCase):
    def test_native_output_keeps_first_joint_and_missing_side(self):
        for name in ('g1', 'l21'):
            specs = {(name, side): load_hand_spec(name, side) for side in ('left', 'right')}
            models = {key: ArrayPredictor(spec.bounds.mean(1)) for key, spec in specs.items()}
            window = HandWindow({'left': np.ones((3,25,3)), 'right': None}, 1., 'test', palm_length_metadata())
            results = JointNNRetargeter(specs, models).retarget(window)
            for key, result in results.items():
                with self.subTest(target=key):
                    self.assertEqual(result.timestamp, 1.)
                    self.assertEqual(result.joint_names, specs[key].independent_joint_names)
                    self.assertEqual(result.spec_hash, specs[key].spec_hash)
                    if key[1] == 'left':
                        np.testing.assert_allclose(result.positions, specs[key].bounds.mean(1))
                        self.assertEqual(len(result.positions), specs[key].dof)
                    else:
                        self.assertFalse(result.valid)
                        self.assertIsNone(result.positions)
                        self.assertEqual(models[key].calls, 0)

    def test_bad_model_output_is_invalid(self):
        spec = load_hand_spec('l21','right')
        window = HandWindow({'left': None, 'right': np.ones((3,25,3))}, 1., 'test', palm_length_metadata())
        for values in (np.zeros(18), np.full(spec.dof,np.nan)):
            result = JointNNRetargeter({spec.key:spec},{spec.key:ArrayPredictor(values)}).retarget(window)[spec.key]
            self.assertFalse(result.valid)
            self.assertIsNone(result.positions)
            self.assertIn('invalid model output',result.error)

    def test_metric_window_rejected_before_inference(self):
        spec = load_hand_spec('l21','right')
        model = ArrayPredictor(np.zeros(spec.dof))
        window = HandWindow({'left':None,'right':np.ones((3,25,3))},1.,'test')
        with self.assertRaises(ValueError):
            JointNNRetargeter({spec.key:spec},{spec.key:model}).retarget(window)
        self.assertEqual(model.calls,0)
