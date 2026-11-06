import importlib.util
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import numpy as np
import torch
from teleoperation.robots.hand_spec import load_hand_spec
from teleoperation.applications.hand_processing import MediaPipePalmLocalPipeline
from teleoperation.retargeting.hand.predictor import build_joint_model
from teleoperation.retargeting.hand.checkpoint import shared_joint_checkpoint_payload,save_joint_checkpoint,load_shared_joint_checkpoint
from teleoperation.retargeting.hand.interface import SharedJointNNRetargeter
from tests.test_joint_checkpoints import SMALL
from tests.test_world_hand_pipeline import world_frame


class HandBackendFactoryTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        self.folder=tempfile.TemporaryDirectory();self.addCleanup(self.folder.cleanup)
        self.weights={}
        for name in ('l21','g1','h1_2'):
            spec=load_hand_spec(name,'right');path=Path(self.folder.name)/(name+'.pth')
            save_joint_checkpoint(path,shared_joint_checkpoint_payload(build_joint_model(spec,SMALL),spec))
            self.weights[name]=path
        pipeline=MediaPipePalmLocalPipeline('world',input_profile='camera_palm_length_window3_v1')
        for i in range(3):self.window=pipeline.process_window(world_frame(i))

    def test_common_factory_preserves_each_selected_side_and_loads_each_model_once(self):
        from teleoperation.applications.hand_backend import build_hand_retargeter_from_specs
        specs={key:load_hand_spec(*key) for key in (('g1','right'),('h1_2','left'))}
        with patch('teleoperation.retargeting.hand.checkpoint.load_shared_joint_checkpoint',wraps=load_shared_joint_checkpoint) as load:
            common=build_hand_retargeter_from_specs(specs,'nn',checkpoints={n:self.weights[n] for n in ('g1','h1_2')})
            self.assertEqual(load.call_count,2)
        result=common.retarget(self.window)
        self.assertEqual(set(result),set(specs))
        for key,spec in specs.items():
            canonical=load_hand_spec(key[0],'right')
            model,_=load_shared_joint_checkpoint(self.weights[key[0]],canonical,'cpu')
            expected=SharedJointNNRetargeter({key:spec},model,canonical_spec=canonical).retarget(self.window)[key]
            np.testing.assert_array_equal(result[key].positions,expected.positions)

    def test_camera_hand_and_dual_boundaries_preserve_the_same_l21_predictions(self):
        from teleoperation.applications.hand_backend import build_hand_retargeter_from_specs,build_camera_retargeter
        from teleoperation.applications.hand_control import _selection
        from teleoperation.applications.dual import load_camera_hand_retargeter
        specs={('l21',s):load_hand_spec('l21',s) for s in ('left','right')}
        args=SimpleNamespace(input_profile='camera_palm_length_window3_v1',hand='both',hand_model='l21',backend='nn',checkpoint=self.weights['l21'],device='cpu')
        retargeters=[build_hand_retargeter_from_specs(specs,'nn',checkpoints={'l21':self.weights['l21']}),
            build_camera_retargeter(('l21',),('left','right'),'nn',{'l21':self.weights['l21']})[1],
            _selection(args)[1],load_camera_hand_retargeter(self.weights['l21'])]
        expected=retargeters[0].retarget(self.window)
        for retargeter in retargeters[1:]:
            actual=retargeter.retarget(self.window)
            for key in specs:np.testing.assert_array_equal(actual[key].positions,expected[key].positions)

    def test_empty_arm_only_selection_loads_no_hand_model_and_rejects_unused_weights(self):
        from teleoperation.applications.hand_backend import build_hand_retargeter_from_specs
        with patch('teleoperation.retargeting.hand.checkpoint.load_shared_joint_checkpoint') as load:
            self.assertEqual(build_hand_retargeter_from_specs({},'nn').retarget(self.window),{})
            load.assert_not_called()
            with self.assertRaises(ValueError):build_hand_retargeter_from_specs({},'nn',checkpoints=self.weights)

    @unittest.skipUnless(importlib.util.find_spec('dex_retargeting'),'requires dex environment')
    def test_dex_keeps_only_selected_model_side_pairs(self):
        from teleoperation.applications.hand_backend import build_hand_retargeter_from_specs
        from teleoperation.retargeting.hand.dex_native import NormalizedDexRetargeter
        specs={key:load_hand_spec(*key) for key in (('g1','right'),('h1_2','left'))}
        actual=build_hand_retargeter_from_specs(specs,'dex').retarget(self.window)
        self.assertEqual(set(actual),set(specs))
        for key in specs:
            expected=NormalizedDexRetargeter((key[0],),(key[1],)).retarget(self.window)[key]
            np.testing.assert_allclose(actual[key].positions,expected.positions,atol=1e-8)
