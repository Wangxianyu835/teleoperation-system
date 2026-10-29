"""Small real FK training runs exercise persistence and deterministic resume."""
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

import torch

from teleoperation.applications.hand_align import align_h5
from teleoperation.learning.trainer import train_joint_models
from teleoperation.paths import training_run_directory
from teleoperation.contracts.coordinates import PALM_LENGTH_PROFILE
from tests.test_world_hand_pipeline import save_raw, world_frame
from tests.test_joint_checkpoints import SMALL


class JointTrainingTests(unittest.TestCase):
    def test_strict_weight_initialization_is_preflighted_for_every_target(self):
        from unittest.mock import patch
        from teleoperation.learning import trainer
        from teleoperation.retargeting.hand.predictor import build_joint_model
        from teleoperation.retargeting.hand.checkpoint import joint_checkpoint_payload,save_joint_checkpoint
        from teleoperation.robots.hand_spec import load_hand_spec
        torch.set_num_threads(1)
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory); data=root/'aligned.h5'; raw=root/'raw.h5'; weights=root/'init.pth'
            save_raw(raw,[world_frame(i) for i in range(40)]); align_h5(raw,data)
            spec=load_hand_spec('g1','right'); model=build_joint_model(spec,SMALL)
            save_joint_checkpoint(weights,joint_checkpoint_payload(model,spec))
            kwargs=dict(input_path=data,root=root/'models',hand_models=['g1'],run_name='init',epochs=1,device='cpu',init_checkpoint=weights)
            seen=[]
            original=trainer._epoch
            def observe(current,*args,**options):
                if not seen:
                    for name,value in current.state_dict().items():
                        torch.testing.assert_close(value,model.state_dict()[name],atol=0,rtol=0)
                    seen.append(True)
                return original(current,*args,**options)
            with patch.object(trainer,'_epoch',side_effect=observe):
                train_joint_models(**kwargs)
            self.assertTrue(seen)
            with self.assertRaises(ValueError):
                train_joint_models(**{**kwargs,'hand_models':['g1','l21'],'run_name':'bad'})
            for name in ('g1','l21'):
                self.assertFalse(training_run_directory(root/'models',name,'right',PALM_LENGTH_PROFILE,'bad').exists())
            with self.assertRaises(ValueError):
                train_joint_models(**{**kwargs,'run_name':'conflict','resume':root/'elsewhere.pth'})

    def test_shared_data_per_target_weights_and_exact_resume(self):
        torch.set_num_threads(1)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            raw, data = root/'raw.h5', root/'human.h5'
            save_raw(raw, [world_frame(i) for i in range(40)])
            align_h5(raw, data)
            before = hashlib.sha256(data.read_bytes()).hexdigest()
            kwargs = dict(input_path=data, root=root/'models', hand_side='right',
                          batch_size=8, val_ratio=.25, architecture=SMALL, device='cpu')
            train_joint_models(hand_models=['g1', 'l21'], run_name='first', epochs=1, **kwargs)
            runs = [training_run_directory(kwargs['root'], n, 'right', PALM_LENGTH_PROFILE, 'first') for n in ('g1','l21')]
            identities = []
            for run in runs:
                self.assertEqual({p.name for p in run.iterdir()},
                                 {'model_best.pth','model_last.pth','config.json','dataset.json','metrics.jsonl','training.log'})
                identities.append(json.loads((run/'dataset.json').read_text()))
            self.assertEqual(identities[0], identities[1])
            with self.assertRaises(FileExistsError):
                train_joint_models(hand_models=['g1'], run_name='first', epochs=1, **kwargs)
            train_joint_models(hand_models=['g1'], run_name='first', epochs=2, resume=runs[0]/'model_last.pth', **kwargs)
            train_joint_models(hand_models=['g1'], run_name='full', epochs=2, **kwargs)
            full = training_run_directory(kwargs['root'], 'g1','right',PALM_LENGTH_PROFILE,'full')
            a = torch.load(runs[0]/'model_last.pth', weights_only=True)
            b = torch.load(full/'model_last.pth', weights_only=True)
            for name, value in a['state_dict'].items():
                torch.testing.assert_close(value, b['state_dict'][name], atol=0, rtol=0)
            self.assertEqual(hashlib.sha256(data.read_bytes()).hexdigest(), before)
            # Sidecar files alone must not authorize optimizer state from another run.
            checkpoint=runs[0]/'model_last.pth'
            state=torch.load(checkpoint,weights_only=True)
            state['training_config']['run_name']='another_run'
            torch.save(state,checkpoint)
            with self.assertRaisesRegex(ValueError,'checkpoint.*identity'):
                train_joint_models(hand_models=['g1'],run_name='first',epochs=3,resume=checkpoint,**kwargs)


if __name__ == '__main__':
    unittest.main()
