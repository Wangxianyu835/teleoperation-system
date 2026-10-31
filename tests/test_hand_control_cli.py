import contextlib
import io
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import h5py
import torch

from teleoperation.cli import main, build_parser
from teleoperation.applications.hand_align import align_h5
from teleoperation.retargeting.hand.predictor import build_joint_model
from teleoperation.retargeting.hand.checkpoint import (
    joint_checkpoint_payload,
    save_joint_checkpoint,
    shared_joint_checkpoint_payload,
)
from teleoperation.robots.hand_spec import load_hand_spec
from tests.test_world_hand_pipeline import world_frame, save_raw
from tests.test_joint_checkpoints import SMALL


class HandControlCLITests(unittest.TestCase):
    def test_missing_camera_dependency_fails_before_gui_or_camera(self):
        import importlib
        from unittest.mock import MagicMock
        real_import=importlib.import_module
        def importing(name,*args,**kwargs):
            if name in ('cv2','mediapipe'): raise ModuleNotFoundError('camera dependency missing')
            return real_import(name,*args,**kwargs)
        with tempfile.TemporaryDirectory() as directory:
            weights=Path(directory)/'target.pth'
            spec=load_hand_spec('l21','right')
            save_joint_checkpoint(weights,joint_checkpoint_payload(build_joint_model(spec,SMALL),spec))
            scene=MagicMock(); scene.alive.return_value=False
            scene.joint_values.return_value=spec.expand(spec.q_reference)
            with patch('importlib.import_module',side_effect=importing), \
                 patch('teleoperation.applications.hand_control._scene',return_value=(scene,MagicMock())) as factory, \
                 patch('teleoperation.applications.hand_realtime.MediaPipeCameraWorkflow') as camera, \
                 patch('teleoperation.applications.realtime_hand_sim.resolve_model_asset',return_value=Path('asset.task')):
                with self.assertRaisesRegex(ModuleNotFoundError,'camera dependency missing'):
                    main(['hand','control','--hand-model','l21','--backend','nn','--hand','right','--checkpoint',str(weights),'--frames','1','--headless','--no-collision-guard'])
                factory.assert_not_called(); camera.assert_not_called()

    def test_scene_initialization_failure_releases_actual_bullet_connection(self):
        import pybullet as p
        from teleoperation.applications.hand_control import _scene
        connect=p.connect
        for name in ('l21','g1'):
            with self.subTest(hand_model=name):
                connected=[]
                def track(*args,**kwargs):
                    cid=connect(*args,**kwargs); connected.append(cid); return cid
                spec=load_hand_spec(name,'right')
                with patch.object(p,'connect',side_effect=track),patch.object(p,'loadURDF',side_effect=RuntimeError('asset load failed')):
                    with self.assertRaisesRegex(RuntimeError,'asset load failed'):
                        _scene({spec.key:spec},True)
                try:
                    self.assertTrue(connected)
                    self.assertFalse(any(p.isConnected(cid) for cid in connected))
                finally:
                    for cid in connected:
                        if p.isConnected(cid): p.disconnect(cid)

    def test_control_uses_same_nn_results_and_native_pybullet_executor(self):
        from teleoperation.applications.hand_processing import MediaPipePalmLocalPipeline
        from dataclasses import replace
        torch.set_num_threads(1)
        pipeline=MediaPipePalmLocalPipeline('world',input_profile='camera_palm_length_window3_v1')
        windows=[pipeline.process_window(world_frame(i)) for i in range(5)]
        # Use ready windows with increasing acquisition timestamps for recording.
        ready=[replace(window,timestamp=i/30) for i,window in enumerate(w for w in windows if w is not None)]
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory); weights=root/'target.pth'; output=root/'live.h5'
            spec=load_hand_spec('l21','right')
            save_joint_checkpoint(weights,joint_checkpoint_payload(build_joint_model(spec,SMALL),spec))
            from unittest.mock import MagicMock
            camera=MagicMock()
            camera.__enter__.return_value=camera
            camera.next_window.side_effect=ready
            with patch('teleoperation.applications.hand_realtime.MediaPipeCameraWorkflow',return_value=camera), \
                 patch('teleoperation.applications.hand_control._camera_dependencies'), \
                 patch('teleoperation.applications.realtime_hand_sim.resolve_model_asset',return_value=Path('asset.task')):
                self.assertEqual(main(['hand','control','--hand-model','l21','--backend','nn','--hand','right',
                    '--checkpoint',str(weights),'--frames','3','--headless','--record-results',str(output)]),0)
            camera.__exit__.assert_called_once()
            with h5py.File(output,'r') as stream:
                self.assertEqual(stream['l21/right/valid'][:].sum(),3)
                self.assertEqual(stream['l21/right/command_positions'].shape,(3,17))

    def test_list_and_train_target_selection(self):
        with contextlib.redirect_stdout(io.StringIO()) as output:
            self.assertEqual(main(['hand','list-models']),0)
        self.assertIn('l21',output.getvalue())
        args = build_parser().parse_args(['hand','train','--input','human.h5','--run-name','example','--hand-model','l21','g1'])
        self.assertEqual(args.hand_model,['l21','g1'])
        self.assertFalse(hasattr(args, 'hand_side'))
        configured=build_parser().parse_args(['hand','train','--input','human.h5','--run-name','init',
            '--hand-model','g1','--checkpoint-root','outputs/models','--input-profile','camera_palm_length_window3_v1',
            '--init-checkpoint','target.pth'])
        self.assertEqual(configured.checkpoint_root,Path('outputs/models'))
        self.assertEqual(configured.init_checkpoint,Path('target.pth'))

    def test_shared_new_nn_checkpoint_is_selected_once_for_both_sides(self):
        from teleoperation.applications.hand_control import _selection
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'shared.pth'
            spec = load_hand_spec('l21', 'right')
            save_joint_checkpoint(path, shared_joint_checkpoint_payload(build_joint_model(spec, SMALL), spec))
            args = build_parser().parse_args([
                'hand', 'control', '--hand-model', 'l21', '--backend', 'nn', '--hand', 'both',
                '--checkpoint', str(path), '--device', 'cpu',
            ])
            specs, retargeter = _selection(args)
            self.assertEqual(set(specs), {('l21', 'left'), ('l21', 'right')})
            self.assertEqual(type(retargeter).__name__, 'SharedJointNNRetargeter')

    def test_nn_export_generic_schema_and_invalid_checkpoint_fails_before_camera(self):
        torch.set_num_threads(1)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            raw, data, weights, output = [root/name for name in ('raw.h5','aligned.h5','nn.pth','result.h5')]
            save_raw(raw,[world_frame(i) for i in range(8)])
            align_h5(raw,data)
            spec = load_hand_spec('l21','right')
            model = build_joint_model(spec,SMALL)
            save_joint_checkpoint(weights,joint_checkpoint_payload(model,spec))
            self.assertEqual(main(['hand','export','--input',str(data),'--output',str(output),
                                   '--hand-model','l21','--backend','nn','--hand','right','--checkpoint',str(weights)]),0)
            with h5py.File(output,'r') as stream:
                self.assertEqual(stream.attrs['schema'],'hand_joint_results_v1')
                self.assertEqual(stream['l21/right/raw_positions'].shape,(8,17))
                self.assertEqual(list(stream['l21/right/valid'][:3]),[False,False,True])
            self.assertEqual(main(['hand','inspect','--angle-h5',str(output)]),0)
            with patch('teleoperation.applications.hand_realtime.MediaPipeCameraWorkflow') as camera:
                with self.assertRaises(ValueError):
                    main(['hand','control','--hand-model','g1','--backend','nn','--hand','right','--checkpoint',str(weights)])
                camera.assert_not_called()


if __name__ == '__main__': unittest.main()
