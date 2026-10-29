"""A reachable real L21 FK target exercises the normalized upstream solver."""
import importlib.util
import unittest

import numpy as np
import torch

from teleoperation.contracts.coordinates import palm_length_metadata
from teleoperation.contracts.hand import HandWindow
from teleoperation.robots.hand_spec import load_hand_spec
from teleoperation.retargeting.hand.hand_model import DifferentiableHandModel, build_hand_targets


@unittest.skipUnless(importlib.util.find_spec('dex_retargeting'),'requires dex environment')
class NormalizedDexTests(unittest.TestCase):
    def test_reachable_l21_target_and_nn_share_native_result_identity(self):
        from teleoperation.retargeting.hand.dex_native import NormalizedDexRetargeter
        from teleoperation.retargeting.hand.interface import JointNNRetargeter
        from teleoperation.retargeting.hand.predictor import build_joint_model
        from tests.test_joint_checkpoints import SMALL
        torch.set_num_threads(1)
        for side in ('left','right'):
            spec=load_hand_spec('l21',side)
            fk=DifferentiableHandModel(spec)
            q=.3*spec.bounds.mean(1)+.7*spec.q_reference
            frames=(*spec.palm_frames,*spec.tip_frames)
            points=fk.positions(torch.tensor(q[None],dtype=torch.float64),frames).detach().numpy()[0]
            canonical=(points-points[:1]) @ spec.native_to_canonical/spec.reference_palm_length_m
            human=np.zeros((25,3),dtype=np.float32)
            human[[0,6,11]]=canonical[:3]
            human[list(spec.human_tip_indices)]=canonical[3:]
            window=HandWindow({s:np.repeat(human[None],3,axis=0) if s==side else None for s in ('left','right')},0.,'fixture',palm_length_metadata())
            target=build_hand_targets(window,spec)
            dex=NormalizedDexRetargeter(['l21'],(side,))
            solved=dex.retarget(window)[spec.key]
            self.assertTrue(solved.valid,solved.error)
            actual=fk.vectors(torch.tensor(solved.positions[None].copy(),dtype=torch.float64)).detach().numpy()[0]
            residual=np.linalg.norm((actual-target.vectors)*spec.reference_palm_length_m,axis=1)
            self.assertLess(float(residual.max()),.005)
            nn=JointNNRetargeter({spec.key:spec},{spec.key:build_joint_model(spec,SMALL)})
            neural=nn.retarget(window)[spec.key]
            self.assertEqual((neural.key,neural.spec_hash,neural.joint_names),(solved.key,solved.spec_hash,solved.joint_names))
            self.assertEqual(neural.positions.shape,solved.positions.shape)


if __name__=='__main__': unittest.main()
